"""Checks that need an NVIDIA GPU (and, where marked, the cuPotential plugin).

    spikes/run.sh -m pytest tests/gpu -m gpu -q
"""
import numpy as np
import openmm as mm
import pytest

from cufep.alchemy.factory import make_alchemical_system
from cufep.alchemy.schedule import fixed_schedule
from cufep.backends import get_backend
from cufep.backends.cupotential import CuPotentialBackend
from cufep.envcheck import check_environment
from cufep.sampling.hrex import ReplicaExchangeSampler
from cufep.sampling.states import StateEnergyEvaluator
from cufep.sampling.storage import read_samples
from cufep.sampling.worker import LegSpec

pytestmark = pytest.mark.gpu

HAS_CUDA = "CUDA" in [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
HAS_PLUGIN = CuPotentialBackend.available()[0]
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA platform")
needs_plugin = pytest.mark.skipif(not (HAS_CUDA and HAS_PLUGIN), reason="cuPotential plugin not available")


@needs_cuda
def test_check_environment_passes():
    lines = []
    assert check_environment(0, lines.append)
    assert any("stock OpenMM on GPU 0: OK" in line for line in lines)


@pytest.fixture(scope="module")
def big_water_leg():
    """A 3.2 nm water box (large enough for the plugin) with one water as the ligand."""
    import openmm.app as app
    import openmm.unit as u

    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(3.2, 3.2, 3.2) * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    stock = get_backend("openmm")
    integ = stock.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=1)
    ctx = stock.make_context(stock.prepare_system(system), integ, 0)
    ctx.setPositions(mod.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 300)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 1)
    integ.step(2000)
    st = ctx.getState(getPositions=True)
    pos = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer))
    del ctx
    ligand = [a.index for a in list(mod.topology.residues())[20].atoms()]
    alch = make_alchemical_system(system, ligand)
    schedule = fixed_schedule(["electrostatics", "sterics"], {"electrostatics": 3, "sterics": 4})
    return alch, schedule, pos, box, ligand


@needs_plugin
def test_backends_agree_at_every_state(big_water_leg):
    import openmm.unit as u

    alch, schedule, pos, box, _ = big_water_leg
    rows = {}
    for name in ("openmm", "cupotential"):
        backend = get_backend(name)
        ctx = backend.make_context(backend.prepare_system(alch.system),
                                   backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        rows[name] = StateEnergyEvaluator(ctx, schedule.states, 300.0, 1.0).reduced_potentials()[0]
        del ctx
    diff = (rows["cupotential"] - rows["cupotential"][0]) - (rows["openmm"] - rows["openmm"][0])
    assert np.max(np.abs(diff)) < 0.1          # kT; stock PME itself is only this accurate (docs/M0_REPORT.md)
    assert abs(rows["cupotential"][0] - rows["openmm"][0]) / abs(rows["openmm"][0]) < 1e-4


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_state_energies_are_right_straight_after_loading_a_configuration(big_water_leg, backend):
    """Energies evaluated without dynamics after setPositions must equal those after the dynamics.

    OpenMM rebuilds the neighbour list of an interaction-group CustomNonbondedForce (the
    soft-core force) only in a computation that also runs the standard nonbonded kernel. If
    the soft-core force group is evaluated on its own after the positions were replaced, the
    list is stale and the energies are wrong by several kT (docs/M0_REPORT.md).
    """
    import openmm.unit as u

    from cufep.sampling.states import set_state

    alch, schedule, pos, box, _ = big_water_leg
    be = get_backend(backend)
    integ = be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=3)
    ctx = be.make_context(be.prepare_system(alch.system), integ, 0)
    ev = StateEnergyEvaluator(ctx, schedule.states, 300.0, None)
    k = next(i for i, s in enumerate(schedule.states) if s.lambda_electrostatics == 0.0 and s.lambda_sterics == 1.0)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 5)
    set_state(ctx, schedule.states[k])
    integ.step(500)
    st = ctx.getState(getPositions=True)
    early_pos, early_box = st.getPositions(asNumpy=True), st.getPeriodicBoxVectors()
    after_md = ev.reduced_potentials(restore=schedule.states[k])[0]
    integ.step(10000)                                  # 20 ps: the ligand's neighbours have changed
    ctx.setPeriodicBoxVectors(*early_box)
    ctx.setPositions(early_pos)
    loaded = ev.reduced_potentials()[0]
    usable = np.abs(after_md - after_md[k]) < 60.0
    error = ((loaded - loaded[k]) - (after_md - after_md[k]))[usable]
    assert np.max(np.abs(error)) < 0.01, error         # kT


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_with_worker_processes(tmp_path, big_water_leg, backend):
    alch, schedule, pos, box, ligand = big_water_leg
    n_gpus = mm.Platform.getPlatformByName("CUDA").getPropertyDefaultValue("DeviceIndex")
    devices = [0, 1] if _gpu_count() >= 2 else [0]
    spec = LegSpec(name="gpu-test", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3, frame_atoms=list(ligand))
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(8, record_stride=2, checkpoint_stride=4)
        assert s.iteration == 8
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        assert s.restore() and s.iteration == 8
        s.run(4)
    data = read_samples(tmp_path / "hrex")
    k = len(schedule.states)
    assert data["u"].shape == (12, k, k)
    own = data["u"][np.arange(12)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own))
    assert all(sorted(row.tolist()) == list(range(k)) for row in data["states"])
    _ = n_gpus


def _gpu_count() -> int:
    from cufep.scheduler import detect_gpus
    return len(detect_gpus(max_utilization=101))


@needs_cuda
def test_killed_worker_is_restarted_and_the_iteration_repeated(tmp_path, big_water_leg):
    """Kill one worker between iterations: the next iteration must still complete for all replicas."""
    alch, schedule, pos, box, ligand = big_water_leg
    devices = [0, 1] if _gpu_count() >= 2 else [0, 0]
    spec = LegSpec(name="kill-test", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend="openmm", temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3)
    messages = []
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=messages.append) as s:
        s.initialize(pos, box)
        s.run(3, checkpoint_stride=0)
        before = s.arrays.pos[s.cur].copy()
        s.workers[1].proc.kill()
        s.workers[1].proc.join(10)
        s.run(3, checkpoint_stride=0)
        assert s.iteration == 6 and s.restarts == 1
        assert np.all(np.isfinite(s.arrays.pos[s.cur])) and not np.allclose(before, s.arrays.pos[s.cur])
    data = read_samples(tmp_path / "hrex")
    k = len(schedule.states)
    assert data["u"].shape == (6, k, k)
    own = data["u"][np.arange(6)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own))
    assert any("restarted" in m for m in messages)
