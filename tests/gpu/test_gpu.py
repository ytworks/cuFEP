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
from cufep.engine.protocol import make_state_evaluator
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


#: How far a backend may lie from stock OpenMM in double precision with an Ewald tolerance of 1e-6, at the
#: tolerance a leg runs with (5e-4): the differences between the windows, in kT. Measured on 31
#: configurations of ``rest2_leg`` with each water (spikes/m31_water_models_on_the_backends.py, 2026-11-24):
#: the plugin at most 0.022 (TIP3P) and 0.044 (OPC), stock OpenMM at most 0.118 and 0.156; the levels are
#: about twice the largest. The plugin against stock, which these tests held to 0.1 kT on one configuration
#: each, is the two errors together, mostly that of stock PME: it reached 0.126 and 0.152, above 0.1 on 1
#: and 5 of the 31 configurations (0.114 on the configuration of ``rest2_leg`` with OPC)
EWALD_LEVEL_KT = {"cupotential": 0.1, "openmm": 0.3}
#: the same for the plugin's potential energy and its forces on the atoms (virtual sites left out), relative.
#: Between windows everything that does not depend on lambda cancels: the energy among the waters, and the
#: forces a virtual site hands on to its atoms (review of 29fb9df)
PLUGIN_ENERGY_LEVEL = 1e-4
PLUGIN_FORCE_LEVEL = 1e-3


def assert_backends_give_the_energies_of_a_tight_ewald_sum(alch_system, states, pos, box, pressure=None):
    """Each backend gives the differences between the windows that a tight Ewald sum gives, so the two
    agree and neither is held to the error of the other's PME; the plugin also its potential energy and
    its forces, in the configuration as it is loaded."""
    import openmm.unit as u

    tight = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(alch_system))
    for force in tight.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setEwaldErrorTolerance(1e-6)
    atoms = [i for i in range(alch_system.getNumParticles()) if not alch_system.isVirtualSite(i)]
    rows, energy, forces = {}, {}, {}
    for name, backend, system in (("reference", get_backend("openmm", precision="double"), tight),
                                  ("openmm", get_backend("openmm"), alch_system),
                                  ("cupotential", get_backend("cupotential"), alch_system)):
        ctx = backend.make_context(backend.prepare_system(system),
                                   backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        state = ctx.getState(getEnergy=True, getForces=True)
        energy[name] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        forces[name] = state.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)[atoms]
        rows[name] = make_state_evaluator(ctx, states, 300.0, pressure).reduced_potentials()[0]
        del ctx
    reference = rows["reference"] - rows["reference"][0]
    measured = {}
    for name, level in EWALD_LEVEL_KT.items():
        error = (rows[name] - rows[name][0]) - reference
        measured[name] = float(np.max(np.abs(error)))
        assert measured[name] < level, (name, error)                     # kT
    measured["energy"] = abs(energy["cupotential"] - energy["reference"]) / abs(energy["reference"])
    measured["forces"] = float(np.linalg.norm(forces["cupotential"] - forces["reference"])
                               / np.linalg.norm(forces["reference"]))
    assert measured["energy"] < PLUGIN_ENERGY_LEVEL, measured
    assert measured["forces"] < PLUGIN_FORCE_LEVEL, measured


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
    # the configuration the backends are compared on is made on the CPU platform with one thread and fixed
    # seeds, so that it is the same in every run: MD on a GPU is not reproducible to the bit, and with a
    # configuration that changed from run to run this comparison failed once in four runs at 1.08e-4 (review
    # of PR 1 on tests whose input is random). The CPU platform with several threads sums forces in an order
    # that varies, and two runs with 4 threads differed by 0.385 nm RMS after 2000 steps (review of PR 1)
    stock = get_backend("openmm", platform="CPU")
    integ = stock.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=1)
    ctx = mm.Context(stock.prepare_system(system), integ, mm.Platform.getPlatformByName("CPU"),
                     {"Threads": "1", "DeterministicForces": "true"})
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
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4})
    return alch, schedule, pos, box, ligand


@needs_plugin
def test_backends_agree_at_every_state(big_water_leg):
    alch, schedule, pos, box, _ = big_water_leg
    assert_backends_give_the_energies_of_a_tight_ewald_sum(alch.system, schedule.states, pos, box, pressure=1.0)


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
    # states unfavourable by 60 kT or more have no weight; favourable ones are compared however low (review of PR 1)
    usable = (after_md - after_md[k]) < 60.0
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
def test_workers_can_be_added_and_removed_between_iterations(tmp_path, big_water_leg):
    """The scheduler hands GPUs to a running job and takes them back; no data may be lost."""
    alch, schedule, pos, box, ligand = big_water_leg
    spec = LegSpec(name="elastic", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend="openmm", temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3)
    k = len(schedule.states)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[0], log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(2, checkpoint_stride=0)
        assert len(s.workers) == 1 and not s.set_devices([0])            # nothing to change
        assert s.set_devices([0, 0, 0]) and len(s.workers) == 3          # three workers share GPU 0
        s.run(2, checkpoint_stride=0)
        assert s.set_devices([0]) and len(s.workers) == 1
        s.run(2, checkpoint_stride=0)
        assert s.set_devices([0] * 50) and len(s.workers) == k           # never more workers than replicas
        s.run(2, checkpoint_stride=0)
        assert s.iteration == 8 and s.restarts == 0
        assert [it for it, _ in s.device_history] == [0, 2, 4, 6]
        assert 0.0 < s.busy_seconds <= s.worker_seconds_available * 1.05
    data = read_samples(tmp_path / "hrex")
    assert data["u"].shape == (8, k, k)
    assert all(sorted(row.tolist()) == list(range(k)) for row in data["states"])
    assert np.all(np.isfinite(data["u"][np.arange(8)[:, None], np.arange(k), data["states"]]))


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_killed_worker_is_restarted_and_the_iteration_repeated(tmp_path, big_water_leg, backend):
    """Kill one worker between iterations: the next iteration must still complete for all replicas."""
    alch, schedule, pos, box, ligand = big_water_leg
    devices = [0, 1] if _gpu_count() >= 2 else [0, 0]
    spec = LegSpec(name="kill-test", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
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


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_precalculation_windows_run_in_a_pool_of_workers(big_water_leg, backend):
    """Adaptive lambda scheduling with the windows of an iteration running in worker processes."""
    import openmm.unit as u

    from cufep.alchemy.schedule import AlsSettings, adaptive_schedule
    from cufep.sampling.hrex import WindowPool
    from cufep.sampling.states import AlchemicalState

    alch, schedule, pos, box, _ = big_water_leg
    be = get_backend(backend)
    system = be.prepare_system(alch.system)
    system.addForce(be.make_isotropic_barostat(1.0 * u.bar, 300 * u.kelvin, 25, seed=2))
    integ = be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=4)
    ctx = be.make_context(system, integ, 0)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 6)
    spec = LegSpec(name="pool", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states[:2]], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=2.0,
                   steps_per_iteration=50, seed=9)
    settings = AlsSettings(candidates=21, initial_windows={"discharge": 3, "vanish": 4}, window_ps=1.0,
                           equilibration_ps=0.2, sample_interval_ps=0.05, walk_ps=0.2, max_windows=10)
    messages = []
    with WindowPool(spec, [0, 0, 0], log=messages.append) as pool:
        result, configs = adaptive_schedule(ctx, integ, ["discharge", "vanish"], 300.0, 2.0, settings,
                                            messages.append, pool=pool)
    assert result.provenance["windows_run_concurrently"] is True
    assert result.states[0] == AlchemicalState(1.0, 1.0, 1.0) and result.states[-1] == AlchemicalState(1.0, 0.0, 0.0)
    assert 6 <= len(result.states) <= 10 and len(configs) == len(result.states)
    assert all(np.all(np.isfinite(c[0])) for c in configs)
    last = {rec["stage"]: rec for rec in result.provenance["iterations"]}
    assert set(last) == {"discharge", "vanish"}
    assert all(rec["status"] in ("converged", "saturated", "max_windows") for rec in last.values())


# ------------------------------------------------------------------------------------- REST2
@pytest.fixture(scope="module", params=["amber14/tip3p.xml", "amber14/opc.xml"], ids=["tip3p", "opc"])
def rest2_leg(request):
    """Two dipeptides in a 3.2 nm water box; one is the ligand, part of the other is hot too.

    In 3-site water and in OPC, whose fourth site is a virtual site: the states, the two backends and
    the replica exchange below are held to the same numbers with both.
    """
    from conftest import build_dipeptides_in_water

    system, top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2, water=request.param)
    if request.param == "amber14/opc.xml":
        assert any(system.isVirtualSite(i) for i in range(system.getNumParticles()))
    alch = make_alchemical_system(system, ligand, rest2=True, rest2_atoms=other[:12])
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4}, rest2_scale=3.0)
    return alch, schedule, pos, box, ligand


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_rest2_state_energies_match_setting_each_state(rest2_leg, backend):
    """The polynomial evaluation of all REST2 state energies equals switching the Context to each state."""
    import openmm.unit as u

    from cufep.sampling.states import set_state

    alch, schedule, pos, box, _ = rest2_leg
    assert max(s.rest2_scale for s in schedule.states) == 3.0
    be = get_backend(backend)
    integ = be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=3)
    ctx = be.make_context(be.prepare_system(alch.system), integ, 0)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 5)
    k = len(schedule.states) // 2
    set_state(ctx, schedule.states[k])
    integ.step(250)                                    # dynamics in a scaled state
    red = StateEnergyEvaluator(ctx, schedule.states, 300.0, None).reduced_potentials(restore=schedule.states[k])[0]
    kt = 8.31446261815324e-3 * 300.0
    direct = []
    for state in schedule.states:
        set_state(ctx, state)
        direct.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / kt)
    direct = np.array(direct)
    error = (red - red[0]) - (direct - direct[0])
    assert np.max(np.abs(error)) < 0.02, error         # kT, single-precision energies of a 3,300-atom box


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_md_with_the_barostat_leaves_every_virtual_site_in_its_place_and_the_water_rigid(rest2_leg, backend):
    """Review of 29fb9df: the plugin has an integrator and a barostat of its own, and no test held what they
    do to a virtual site. After NPT MD on each backend the box has changed, every constraint has its length
    and every virtual site is where the System puts it (``computeVirtualSites`` moves none). In the
    configurations a leg stored after its MD on the plugin with OPC these are 5e-8 nm and 5e-10 nm."""
    import openmm.unit as u

    alch, _schedule, pos, box, _ = rest2_leg
    be = get_backend(backend)
    system = be.prepare_system(alch.system)
    system.addForce(be.make_isotropic_barostat(1.0 * u.bar, 300 * u.kelvin, 5, seed=3))
    integ = be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=3)
    ctx = be.make_context(system, integ, 0)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 5)
    integ.step(1000)                                   # 200 attempts to change the volume
    state = ctx.getState(getPositions=True)
    assert abs(state.getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3) - np.linalg.det(box)) > 1e-3
    x = np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    constraints = [system.getConstraintParameters(k) for k in range(system.getNumConstraints())]
    i, j = np.array([c[0] for c in constraints]), np.array([c[1] for c in constraints])
    length = np.array([c[2].value_in_unit(u.nanometer) for c in constraints])
    assert np.abs(np.linalg.norm(x[i] - x[j], axis=1) - length).max() < 1e-6          # nm
    ctx.computeVirtualSites()
    moved = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)) - x
    assert np.abs(moved).max() < 1e-6                                                # nm


@needs_plugin
def test_rest2_backends_agree(rest2_leg):
    """With REST2, in 3-site water and in OPC, whose fourth site is a virtual site."""
    alch, schedule, pos, box, _ = rest2_leg
    assert_backends_give_the_energies_of_a_tight_ewald_sum(alch.system, schedule.states, pos, box, pressure=1.0)


@needs_plugin
def test_backends_agree_with_annihilated_electrostatics_and_the_squared_soft_core():
    """``alchemy.annihilate_electrostatics`` and ``alchemy.softcore_b: 2``, with REST2: the plugin
    takes the NonbondedForce whose ligand exceptions carry no charge product, the Coulomb energy
    inside the ligand and the soft-core stay on stock kernels that share its global parameters,
    and the reduced potentials of every state agree between the two backends as they do without
    these settings; on each backend the polynomial evaluation equals setting each state."""
    import openmm.unit as u
    from conftest import build_dipeptides_in_water

    from cufep.sampling.states import set_state

    system, _top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
    alch = make_alchemical_system(system, ligand, softcore_alpha=1.0, rest2=True, rest2_atoms=other[:12],
                                  annihilate_electrostatics=True, softcore_b=2)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 5, "vanish": 4}, rest2_scale=3.0)
    kt = 8.31446261815324e-3 * 300.0
    rows = {}
    for name in ("openmm", "cupotential"):
        backend = get_backend(name)
        ctx = backend.make_context(backend.prepare_system(alch.system),
                                   backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        rows[name] = StateEnergyEvaluator(ctx, schedule.states, 300.0, None).reduced_potentials()[0]
        direct = []
        for state in schedule.states:
            set_state(ctx, state)
            direct.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / kt)
        direct = np.array(direct)
        error = (rows[name] - rows[name][0]) - (direct - direct[0])
        assert np.max(np.abs(error)) < 0.02, (name, error)
        del ctx
    assert_backends_give_the_energies_of_a_tight_ewald_sum(alch.system, schedule.states, pos, box)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_with_rest2(tmp_path, rest2_leg, backend):
    alch, schedule, pos, box, ligand = rest2_leg
    devices = [0, 1] if _gpu_count() >= 2 else [0]
    spec = LegSpec(name="gpu-rest2", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3, frame_atoms=list(ligand))
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(10, record_stride=2, checkpoint_stride=5)
    data = read_samples(tmp_path / "hrex")
    k = len(schedule.states)
    assert data["u"].shape == (10, k, k)
    own = data["u"][np.arange(10)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own)) and np.all(np.isfinite(data["u"][:, :, [0, k - 1]]))
    assert all(sorted(row.tolist()) == list(range(k)) for row in data["states"])
    # replicas exchange: swaps between neighbouring windows are accepted. Measured in five runs of each water
    # and backend (2026-11-24): 97 to 205 accepted in the 10 iterations; the level is a third of the fewest.
    # Not "the states of the replicas changed from one iteration to the next": a pair that swaps several times
    # in an iteration ends where it began half the time, and stock OpenMM on a GPU does not give the same
    # trajectory in every run. With OPC the states changed in 2 to 5 of the 9 steps between iterations, and in
    # one run of the suite in none
    assert data["swaps"].shape == (10, k - 1, 2)
    assert data["swaps"][:, :, 1].sum() > 30 and (data["swaps"][:, :, 1] <= data["swaps"][:, :, 0]).all()


@needs_plugin
def test_the_system_of_a_validation_with_a_rest2_pocket_agrees_on_both_backends():
    """Review of PR 1 (P2): `cufep validate` compares the System that a run samples, with REST2 and its region;
    here the pocket region holds the whole second peptide and the windows in the middle carry the scale 1.5."""
    from conftest import build_dipeptides_in_water

    from cufep.config.schema import Config
    from cufep.validate import validation_system

    system, top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
    prep = {"topology": top, "positions": pos, "ligand": ligand, "protein": other,
            "input_chains": {c.id: c.id for c in top.chains()}}
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc": "none"},
                                 "sampling": {"rest2": {"scale": 1.5, "region": "pocket", "pocket_cutoff_nm": 5.0}}})
    built, schedule = validation_system(cfg, system, prep, box, "complex")
    assert built.rest2_on and built.hot_atoms == sorted(other)
    assert_backends_give_the_energies_of_a_tight_ewald_sum(built.alch.system, schedule.states, pos, box)


# ------------------------------------------------------------------- MD for the restraint
@needs_cuda
@pytest.mark.parametrize("precision", ["mixed", "single"])
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_the_trajectories_for_a_restraint_run_on_the_backend_of_the_leg(backend, precision, tmp_path):
    """The unrestrained NPT trajectories from which ``restraints.selection: interactions`` chooses a
    restraint run on the backend that is configured, two processes at a time on one GPU and in the
    process of the job alike, and each file records that backend. No fallback is allowed here: a
    plugin that is configured and not used fails the test instead of passing on stock OpenMM. In
    mixed and in single precision: plain MD runs in the one of the YAML."""
    from conftest import build_dipeptides_in_water

    from cufep.sampling import restraint_md
    from cufep.sampling.equilibrate import EquilibrationSettings

    system, _top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
    solute = sorted(ligand + other)
    settings = EquilibrationSettings(production_ns=0.002, frame_interval_ps=0.5, platform="CUDA", precision=precision,
                                     timestep_fs=2.0, seed=5, md_backend=backend, fallback="none",
                                     temperature_K=300.0, pressure_bar=1.0, barostat_frequency=25)

    def tasks(directory, n):
        xml = mm.XmlSerializer.serialize(system)
        return [restraint_md.TrajectoryTask(
            index=k, path=str(restraint_md.trajectory_file(directory, k)), system_xml=xml, positions=pos,
            velocities=np.zeros_like(pos), box=box, frame_atoms=solute, settings=settings, timestep_fs=2.0,
            made_with={"trajectory": k}) for k in range(n)]

    lines = []
    summaries = restraint_md.run_trajectories(tasks(tmp_path / "processes", 3), [0, 0], lines.append)
    assert [(s["index"], s["backend"], s["frames"], s["notes"]) for s in summaries] == \
        [(k, backend, 4, []) for k in range(3)]
    assert not [line for line in lines if "stock OpenMM" in line]
    read = restraint_md.read_trajectories(tmp_path / "processes", 3)
    assert read.backends == [backend] * 3 and read.timesteps_fs == [2.0] * 3 and read.notes == []
    assert read.frames.shape == (12, len(solute), 3) and np.all(np.isfinite(read.frames))
    assert read.time_ps.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0] * 3)
    firsts = [read.frames[read.trajectory == k][-1] for k in range(3)]
    assert not np.allclose(firsts[0], firsts[1], atol=1e-4) and not np.allclose(firsts[1], firsts[2], atol=1e-4)
    # NPT: the box moves, and the solute stays what it was (no atom flew off)
    assert not np.allclose(read.boxes, read.boxes[0]) and np.abs(read.boxes - box).max() < 0.2
    assert np.abs(read.frames - pos[solute]).max() < 1.5

    inline = restraint_md.run_trajectories(tasks(tmp_path / "inline", 2), [0], lines.append, inline=True)
    assert [s["backend"] for s in inline] == [backend] * 2
    assert restraint_md.read_trajectories(tmp_path / "inline", 2).frames.shape == (8, len(solute), 3)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_the_pre_equilibration_of_a_leg_runs_in_single_precision_where_it_is_asked(backend, monkeypatch):
    """``backend.precision: single`` is the precision of the plain MD of a leg too (the owner's decision of
    2026-11-21; earlier versions made mixed of it where the settings of the YAML become those of the MD): the
    minimisation, NVT and NPT on stock OpenMM and the unrestrained MD on the backend of the leg, without a
    fallback and without a recovery. The settings are those the leg makes of the YAML, and the precision is
    read from every Context that was made (review of eb6f89f: the test gave the MD its settings directly and
    looked at no Context)."""
    from conftest import build_dipeptides_in_water

    from cufep.backends.base import Backend
    from cufep.config.schema import Config
    from cufep.sampling.equilibrate import pre_equilibrate
    from cufep.workflow import plain_md_settings

    system, _top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
    solute = sorted(ligand + other)
    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "backend": {"name": backend, "precision": "single", "fallback": "none"}})
    settings = plain_md_settings(cfg, 2.0, "isotropic", 3, nvt_ns=0.002, npt_ns=0.002, production_ns=0.004,
                                 frame_interval_ps=1.0, minimize_iterations=200, restrained_atoms=solute)
    assert (settings.precision, settings.md_backend, settings.fallback) == ("single", backend, "none")
    precisions = []
    make = Backend.make_context

    def made(self, *args, **kwargs):
        context = make(self, *args, **kwargs)
        precisions.append(context.getPlatform().getPropertyValue(context, "Precision"))
        return context

    monkeypatch.setattr(Backend, "make_context", made)
    lines = []
    result = pre_equilibrate(system, pos, box, solute, settings, 0, lines.append)
    assert precisions and set(precisions) == {"single"}
    assert result.backend == backend and result.notes == [] and result.timestep_fs == 2.0
    assert result.frames.shape == (4, len(solute), 3) and np.all(np.isfinite(result.positions))
    assert np.all(np.isfinite(result.velocities)) and np.abs(result.box - box).max() < 0.2
    assert not [line for line in lines if "stock OpenMM" in line or "2 fs" in line]


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_a_pre_equilibration_that_holds_the_solute_runs_on_the_backend_of_the_leg(backend):
    """``restraints.selection: interactions`` holds the heavy atoms of the solute where they were given while
    the surroundings settle; for a bilayer the MD after NVT and NPT runs with them held, on the backend of the
    leg. Without a fallback: the plugin runs it with the position restraints, and the held atoms stay."""
    from conftest import build_dipeptides_in_water

    from cufep.sampling.equilibrate import EquilibrationSettings, pre_equilibrate

    system, top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
    atoms = list(top.atoms())
    heavy = [i for i in sorted(ligand + other) if atoms[i].element.symbol != "H"]
    # 10,000 kJ/mol/nm^2: an atom then moves 0.016 nm along an axis at 300 K (the root of kT / k), and the
    # 0.1 nm below is more than six times that. (With the 1,000 of a leg the same is 0.05 nm and one of twenty
    # atoms was 0.102 nm away once: the level must not be one that the noise of an MD on a GPU can reach.)
    settings = EquilibrationSettings(nvt_ns=0.002, npt_ns=0.002, production_ns=0.02, frame_interval_ps=5.0,
                                     minimize_iterations=200, platform="CUDA", timestep_fs=2.0, seed=3,
                                     md_backend=backend, fallback="none", restrained_atoms=heavy, hold_solute=True,
                                     restraint_k=10000.0)
    held = pre_equilibrate(system, pos, box, sorted(ligand + other), settings, 0, lambda line: None)
    assert held.backend == backend and held.notes == [] and held.frames.shape[0] == 4
    assert np.linalg.norm(held.positions[heavy] - pos[heavy], axis=1).max() < 0.1



@pytest.fixture(scope="module", params=["amber14/tip3p.xml", "amber14/opc.xml"], ids=["tip3p", "opc"])
def moves_leg(request):
    """Two dipeptides in a 3.2 nm water box, one of them the ligand, with what a mover of water
    relocation moves needs of it (a sphere of the paper's radius around the ligand). In 3-site water and
    in OPC, whose waters have a fourth site, a virtual one: the moves treat the four as one body."""
    from conftest import build_dipeptides_in_water

    from cufep.sampling.relocation_mc import relocation_spec

    system, top, pos, box, ligand, _other = build_dipeptides_in_water(box_nm=3.2, water=request.param)
    cutoff = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce)).getCutoffDistance()._value
    relocation = relocation_spec(system, top, pos, box, ligand, cutoff, 0.5, 1, 300.0, trial_poses=256,
                                 inner_fraction=0.5, interval_ps=0.4, radius_margin_nm=0.3)
    alch = make_alchemical_system(system, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 2, "vanish": 3})
    return alch, schedule, pos, box, ligand, relocation


def test_the_trial_energies_on_the_gpu_are_those_of_the_processor(moves_leg):
    """The numbers a proposal needs (the scores of all waters, the energies of the trial poses at
    the cutoff and at the radius of the scores, the energy of a pose with every other water) come
    from kernels on the CUDA device of the Context; the NumPy stand-in is the reference. Both are
    the same function in double precision, so they agree to rounding: in a triclinic cell, with
    the ligand at intermediate lambda values, for poses anywhere in the cell. A mover on a CUDA
    Context uses the device, and then makes the proposal the processor makes from the same random
    numbers."""
    import openmm.unit as u

    from cufep.sampling.relocation_mc import SCORE_RADIUS_NM, GpuStandIn, RelocationMover, StandIn, random_rotations

    alch, schedule, pos, box, _ligand, relocation = moves_leg
    pos = np.asarray(pos, dtype=float)
    box = np.asarray(box, dtype=float) + np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.3, 0.5, 0.0]])
    rng = np.random.default_rng(5)
    moved = 11
    atoms = relocation.waters[moved]
    local = pos[atoms] - pos[atoms[0]]
    poses = (rng.random((512, 3)) @ box)[:, None, :] + np.einsum("kij,sj->ksi", random_rotations(rng, 512), local)
    gpu = GpuStandIn(relocation, 0)
    for lambda_electrostatics, lambda_sterics in ((1.0, 1.0), (0.4, 1.0), (0.0, 0.3)):
        reference = StandIn(relocation, pos, box, lambda_electrostatics, lambda_sterics)
        device = gpu.load(pos, box, lambda_electrostatics, lambda_sterics)
        for a, b in ((reference.scores(), device.scores()),
                     (reference.pose_energies(poses, moved, relocation.cutoff_nm),
                      device.pose_energies(poses, moved, relocation.cutoff_nm)),
                     (reference.pose_energies(poses[:32], moved, SCORE_RADIUS_NM),
                      device.pose_energies(poses[:32], moved, SCORE_RADIUS_NM)),
                     (reference.with_waters(poses[:2], moved), device.with_waters(poses[:2], moved))):
            assert a.shape == b.shape and np.isfinite(a).all()
            assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    # a mover on a CUDA Context computes on the device, and its proposal is that of the processor
    backend = get_backend("openmm")
    ctx = backend.make_context(backend.prepare_system(alch.system),
                               backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
    state = schedule.states[2]
    on_device, on_processor = RelocationMover(relocation, ctx, seed=9), RelocationMover(relocation, None, seed=9)
    assert on_device.engine == "the GPU of the Context" and on_processor.engine == "the processor"
    for out in (False, True, False, True):
        a, b = on_device.proposal(pos, box, state, out), on_processor.proposal(pos, box, state, out)
        assert a["moved"] == b["moved"] and a["chosen"] == b["chosen"]
        assert a["log_q_ratio"] == pytest.approx(b["log_q_ratio"], rel=1e-9, abs=1e-9)


#: How far the energy change of a relocated water may lie, in a backend, from stock OpenMM in double
#: precision with an Ewald tolerance of 1e-6, in kT. Measured on 40 proposals in each of three windows of
#: ``moves_leg`` with each water (spikes/m32_relocated_water_on_the_backends.py, 2026-11-24): the plugin at
#: most 0.024 (TIP3P) and 0.035 (OPC), stock OpenMM at most 0.079 and 0.225; the levels are about twice
#: the largest and more. The plugin against stock, which this test held to 0.1 kT with TIP3P, is the two
#: errors together, mostly that of stock PME: at most 0.086 with TIP3P, and with OPC up to 0.231 and
#: above 0.1 for 52 of the 120 proposals
MOVE_LEVEL_KT = {"cupotential": 0.1, "openmm": 0.5}


@needs_plugin
def test_the_energy_change_of_a_relocated_water_is_that_of_a_tight_ewald_sum(moves_leg):
    """The acceptance of a water relocation move needs one number of the backend: the change of
    the potential energy of the simulation when the water is put at its new pose, in the state the
    replica is simulated in. For the same proposals (they are made on the processor, from the
    same configuration) the plugin and stock OpenMM each give the change that a tight Ewald sum
    gives, so the two agree and neither is held to the error of the other's PME. With OPC the moved
    water has a virtual site, which is put at its new place with the three atoms."""
    import openmm.unit as u

    from cufep.sampling.relocation_mc import RelocationMover
    from cufep.sampling.states import set_state

    alch, schedule, pos, box, _ligand, relocation = moves_leg
    state = schedule.states[2]                               # charges half off
    proposals = RelocationMover(relocation, None, seed=3)
    moves = [proposals.proposal(np.asarray(pos, dtype=float), np.asarray(box, dtype=float), state, out)
             for out in (False, True, False, True, False, True)]
    assert all(move is not None for move in moves)
    tight = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(alch.system))
    for force in tight.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setEwaldErrorTolerance(1e-6)
    kt = 8.31446261815324e-3 * 300.0
    changes = {}
    for name, backend, system in (("reference", get_backend("openmm", precision="double"), tight),
                                  ("openmm", get_backend("openmm"), alch.system),
                                  ("cupotential", get_backend("cupotential"), alch.system)):
        ctx = backend.make_context(backend.prepare_system(system),
                                   backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        set_state(ctx, state)
        mover = RelocationMover(relocation, ctx, seed=3)
        base = mover._potential(pos)
        assert mover._potential(pos) == pytest.approx(base, abs=0.05)      # the same number when asked again
        changes[name] = np.array([mover._potential(move["y"]) - base for move in moves]) / kt
        del ctx
    for name, level in MOVE_LEVEL_KT.items():
        assert np.max(np.abs(changes[name] - changes["reference"])) < level, (name, changes)
    assert np.abs(changes["reference"]).max() > 1.0                        # (moves that change the energy)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_with_water_relocation_moves(tmp_path, moves_leg, backend):
    """Replica exchange with worker processes and the moves: every replica attempts one move after
    the MD of an iteration in which the interval was passed (0.4 ps: every second iteration of 50
    steps of 4 fs), in the state it is simulated in; the samples hold the attempts by window and the
    energies stay finite."""
    alch, schedule, pos, box, ligand, relocation = moves_leg
    devices = [0, 1] if _gpu_count() >= 2 else [0]
    spec = LegSpec(name="gpu-moves", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3, frame_atoms=list(ligand), relocation=relocation)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(10, record_stride=2, checkpoint_stride=5)
    data = read_samples(tmp_path / "hrex")
    k = len(schedule.states)
    own = data["u"][np.arange(10)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own)) and np.all(np.isfinite(data["u"][:, :, [0, k - 1]]))
    moves = data["relocations"]
    attempted = moves[:, :, 0] + moves[:, :, 2]
    assert moves.shape == (10, k, 4) and attempted.sum(axis=1).tolist() == [0, k] * 5
    assert (moves[:, :, 1] <= moves[:, :, 0]).all() and (moves[:, :, 3] <= moves[:, :, 2]).all()


# ------------------------------------------------------------------- grand canonical moves of water
@pytest.fixture(scope="module", params=["amber14/tip3p.xml", "amber14/opc.xml"], ids=["tip3p", "opc"])
def gcmc_leg(request):
    """Two dipeptides in a 3.2 nm water box, one of them the ligand, as a leg with grand canonical moves of
    water is made of it: the specification of the moves from the system with every water, the alchemical
    System from the system with the ghosts of the start switched off. The excess chemical potential is 4
    kcal/mol above that of water, so that the few moves of a test insert waters."""
    from conftest import build_dipeptides_in_water

    from cufep.sampling.gcmc import gcmc_spec, ghosts_off

    system, top, pos, box, ligand, _other = build_dipeptides_in_water(box_nm=3.2, water=request.param)
    cutoff = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce)).getCutoffDistance()._value
    moves = gcmc_spec(system, top, pos, box, ligand, cutoff, 0.5, 1, 300.0, interval_ps=0.4, attempts=400,
                      batch_size=200, radius_margin_nm=0.3, excess_chemical_potential_kcal_per_mol=-2.0,
                      standard_volume_nm3=0.0303, ghost_waters=None)
    plain = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ghosts_off(plain, moves.waters[moves.ghosts].reshape(-1))
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 2, "vanish": 3})
    return make_alchemical_system(plain, ligand), make_alchemical_system(system, ligand), schedule, pos, box, \
        ligand, moves


def test_the_trial_energies_with_ghosts_on_the_gpu_are_those_of_the_processor(gcmc_leg):
    """The trials of the grand canonical moves are evaluated by the kernels on the CUDA device of the
    Context, with the waters that are switched off left out: the energies of poses of a water and of waters
    where they are equal those of the NumPy stand-in, in a triclinic cell, with the ligand at intermediate
    lambda values."""
    from cufep.sampling.relocation_mc import GpuStandIn, StandIn, random_rotations

    _alch, _all_on, _schedule, pos, box, _ligand, moves = gcmc_leg
    pos = np.asarray(pos, dtype=float)
    box = np.asarray(box, dtype=float) + np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.3, 0.5, 0.0]])
    rng = np.random.default_rng(5)
    on = moves.initial_on()
    on[rng.choice(np.flatnonzero(on), size=40, replace=False)] = False       # ghosts all over the cell
    active = np.ones(len(pos), dtype=bool)
    active[moves.waters[~on].reshape(-1)] = False
    ghost = int(np.flatnonzero(~on)[0])
    local = pos[moves.waters[ghost]] - pos[moves.waters[ghost][0]]
    poses = (rng.random((256, 3)) @ box)[:, None, :] + np.einsum("kij,sj->ksi", random_rotations(rng, 256), local)
    which = np.flatnonzero(on)[::37]
    gpu = GpuStandIn(moves, 0)
    for lambda_electrostatics, lambda_sterics in ((1.0, 1.0), (0.4, 1.0), (0.0, 0.3)):
        reference = StandIn(moves, pos, box, lambda_electrostatics, lambda_sterics, active=active)
        device = gpu.load(pos, box, lambda_electrostatics, lambda_sterics, active=active)
        for a, b in ((reference.pose_energies(poses, ghost, moves.cutoff_nm),
                      device.pose_energies(poses, ghost, moves.cutoff_nm)),
                     (reference.water_energies(which, moves.cutoff_nm), device.water_energies(which, moves.cutoff_nm))):
            assert a.shape == b.shape and np.isfinite(a).all()
            assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
        everything = gpu.load(pos, box, lambda_electrostatics, lambda_sterics).water_energies(which, moves.cutoff_nm)
        assert np.abs(everything - reference.water_energies(which, moves.cutoff_nm)).max() > 1.0   # (ghosts count)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_waters_switched_in_a_context_give_the_energies_of_a_system_built_so(gcmc_leg, backend):
    """A Context of the leg (its System has the ghosts of the start off) whose ghosts are switched on gives
    the energies of the System with every water, in every window, and switched back those it had: on stock
    OpenMM to its single precision, on the plugin too (measured: the same numbers; the risk check of the
    plan, spikes/m33_ghost_waters_on_the_backends.py). With OPC the charge that is switched is on a
    virtual site."""
    import openmm.unit as u

    from cufep.sampling.gcmc import WaterSwitch

    alch, all_on, schedule, pos, box, _ligand, moves = gcmc_leg
    be = get_backend(backend)

    def context_of(system):
        ctx = be.make_context(be.prepare_system(system),
                              be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds, seed=3), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        return ctx

    def numbers(ctx):
        ctx.setPositions(pos)
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return energy, StateEnergyEvaluator(ctx, schedule.states, 300.0, None).reduced_potentials()[0]

    leg, every = context_of(alch.system), context_of(all_on.system)
    (e_leg, u_leg), (e_every, u_every) = numbers(leg), numbers(every)
    assert abs(e_every - e_leg) > 100.0                                    # kJ/mol: the ghosts of the start
    switch = WaterSwitch(leg, moves)
    assert switch.apply(np.ones(len(moves.waters), dtype=bool)) == len(moves.ghosts)
    e_on, u_on = numbers(leg)
    assert e_on == pytest.approx(e_every, abs=0.05) and np.allclose(u_on, u_every, rtol=0.0, atol=0.02)
    assert switch.apply(moves.initial_on()) == len(moves.ghosts)
    e_off, u_off = numbers(leg)
    assert e_off == pytest.approx(e_leg, abs=0.05) and np.allclose(u_off, u_leg, rtol=0.0, atol=0.02)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_with_grand_canonical_moves_of_water(tmp_path, gcmc_leg, backend):
    """Replica exchange with worker processes and the moves, without a barostat: every replica makes one
    move after the MD of an iteration in which the interval was passed (0.4 ps: every second iteration of
    50 steps of 4 fs), in the state it is simulated in, with the waters it has; the samples hold the counts
    by window, the checkpoint the waters of every replica, the energies stay finite and the cell is the one
    the leg started in. A leg that is resumed goes on with the waters it had."""
    alch, _all_on, schedule, pos, box, ligand, moves = gcmc_leg
    devices = [0, 1] if _gpu_count() >= 2 else [0]
    spec = LegSpec(name="gpu-gcmc", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend=backend, temperature_K=300.0, pressure_bar=None, barostat="none", timestep_fs=4.0,
                   steps_per_iteration=50, seed=3, frame_atoms=list(ligand), gcmc=moves)
    k, n_waters = len(schedule.states), len(moves.waters)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(10, record_stride=2, checkpoint_stride=5)
    data = read_samples(tmp_path / "hrex")
    own = data["u"][np.arange(10)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own)) and np.all(np.isfinite(data["u"][:, :, [0, k - 1]]))
    counts = data["gcmc"]
    assert counts.shape == (10, k, 5)
    assert (counts[:, :, 0] + counts[:, :, 2]).sum(axis=1).tolist() == [0, k * moves.attempts] * 5
    assert (counts[::2, :, 4] == -1).all() and (counts[1::2, :, 4] >= 0).all()
    assert counts[:, :, 1].sum() > 0                                        # (insertions were made)
    assert np.allclose(data["volume"], abs(np.linalg.det(box)), rtol=1e-6)  # no barostat
    with np.load(tmp_path / "hrex" / "checkpoint.npz") as checkpoint:
        lists = checkpoint["water_on"]
    made = int(counts[:, :, 1].sum() - counts[:, :, 3].sum())
    assert lists.shape == (k, n_waters) and int(lists.sum()) == k * (n_waters - len(moves.ghosts)) + made
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=devices, log=lambda m: None) as s:
        assert s.restore() and np.array_equal(s.arrays.won[s.cur] > 0.5, lists)
        s.run(2, record_stride=2, checkpoint_stride=2)
    assert read_samples(tmp_path / "hrex")["gcmc"].shape == (12, k, 5)


@needs_cuda
def test_a_leg_whose_ghosts_run_out_stops_and_no_worker_is_restarted(tmp_path, gcmc_leg):
    """A worker that cannot try an insertion because no ghost is left says so, and the sampler stops the leg
    with that reason (:class:`cufep.sampling.hrex.MovesError`). It is not a failure of the worker: before,
    the worker was restarted and its replicas run again with other random numbers, and a second such stop
    counted as a failure of the backend, after which a leg is repeated on stock OpenMM (review of the
    moves, 2026-11-25)."""
    from dataclasses import replace

    from cufep.sampling.hrex import MovesError, WorkerError

    alch, _all_on, schedule, pos, box, ligand, moves = gcmc_leg
    eager = replace(moves, excess_chemical_potential_kj_per_mol=40.0)        # insertions at once
    spec = LegSpec(name="gpu-gcmc", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=alch.system.getNumParticles(),
                   backend="openmm", temperature_K=300.0, pressure_bar=None, barostat="none", timestep_fs=4.0,
                   steps_per_iteration=100, seed=3, frame_atoms=list(ligand), gcmc=eager)
    one_ghost = np.ones(len(moves.waters), dtype=bool)
    one_ghost[moves.ghosts[0]] = False
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[0], log=lambda m: None) as s:
        s.initialize(pos, box, water_on=one_ghost)
        with pytest.raises(MovesError, match="no ghost water is left") as stopped:
            s.run(20, record_stride=2, checkpoint_stride=5)
        assert not isinstance(stopped.value, WorkerError) and s.restarts == 0


# ---- relative transformations (cufep.rbfe) ------------------------------------------------------------------
@pytest.fixture(scope="module", params=[False, True], ids=["plain", "rest2"])
def hybrid_leg(request):
    """Toluene into phenol in a 3.2 nm box of water (large enough for the plugin), the hybrid System of the two;
    with REST2 a dozen water atoms are hot too."""
    from conftest import build_hybrid_edge

    from cufep.rbfe.hybrid import make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge(box_nm=3.2)
    rest2 = request.param
    hot = list(range(len(ligand_a), len(ligand_a) + 12)) if rest2 else []
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=rest2,
                                rest2_atoms=hot)
    schedule = fixed_schedule(["morph"], {"morph": 8}, rest2_scale=3.0 if rest2 else 1.0)
    return hybrid, schedule, box


@needs_plugin
def test_hybrid_backends_agree_at_every_state(hybrid_leg):
    """The states of a hybrid System: each backend gives the differences between the windows that a tight Ewald
    sum gives (the plugin carries the charges of the ligand as offsets under five global parameters)."""
    hybrid, schedule, box = hybrid_leg
    assert_backends_give_the_energies_of_a_tight_ewald_sum(hybrid.system, schedule.states, hybrid.positions, box,
                                                           pressure=1.0)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_hybrid_state_energies_match_setting_each_state(hybrid_leg, backend):
    """The evaluation of all hybrid state energies (a quadratic in each half of lambda for the main group)
    equals switching the Context to each state, after dynamics in a scaled state."""
    import openmm.unit as u

    from cufep.sampling.states import set_state

    hybrid, schedule, box = hybrid_leg
    be = get_backend(backend)
    integ = be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=3)
    ctx = be.make_context(be.prepare_system(hybrid.system), integ, 0)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(hybrid.positions)
    k = len(schedule.states) // 2
    set_state(ctx, schedule.states[k])
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 5)
    integ.step(250)
    red = make_state_evaluator(ctx, schedule.states, 300.0, None).reduced_potentials(restore=schedule.states[k])[0]
    kt = 8.31446261815324e-3 * 300.0
    direct = []
    for state in schedule.states:
        set_state(ctx, state)
        direct.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / kt)
    direct = np.array(direct)
    error = (red - red[0]) - (direct - direct[0])
    assert np.max(np.abs(error)) < 0.02, error         # kT, as the test of the REST2 states of the ABFE


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_of_a_hybrid_leg(tmp_path, hybrid_leg, backend):
    """Replica exchange of the hybrid states on the GPU, with the barostat: finite energies and accepted swaps."""
    hybrid, schedule, box = hybrid_leg
    spec = LegSpec(name="gpu-hybrid", system_xml=mm.XmlSerializer.serialize(hybrid.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=hybrid.system.getNumParticles(),
                   protocol="hybrid", backend=backend, temperature_K=300.0, pressure_bar=1.0, barostat="isotropic",
                   timestep_fs=2.0, steps_per_iteration=50, seed=3, frame_atoms=list(hybrid.ligand_atoms))
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[0], log=lambda m: None) as s:
        s.initialize(hybrid.positions, box)
        s.run(10, record_stride=2, checkpoint_stride=5)
    data = read_samples(tmp_path / "hrex")
    k = len(schedule.states)
    assert data["u"].shape == (10, k, k) and np.all(np.isfinite(data["u"]))
    assert data["swaps"][:, :, 1].sum() > 0


@pytest.fixture(scope="module")
def hybrid_gcmc_leg():
    """The hybrid of toluene into phenol in a 3.2 nm box with grand canonical moves of water as a relative complex
    leg makes them (cufep.rbfe.leg.hybrid_gcmc): the hybrid with every water, the hybrid with the ghosts of the
    start off, and the specification of the moves; the excess chemical potential 4 kcal/mol above that of water,
    so that the few moves of a test insert waters."""
    from conftest import build_hybrid_edge, hybrid_edge_topology

    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_gcmc
    from cufep.sampling.gcmc import ghosts_off

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge(box_nm=3.2)
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    topology = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system)
    rcfg = RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"},
                                      "sampling": {"gcmc": {"interval_ps": 0.2, "attempts": 400, "batch_size": 200,
                                                            "excess_chemical_potential_kcal_per_mol": -2.0}}})
    moves = hybrid_gcmc(rcfg, hybrid, topology, box)
    every = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(hybrid.system))
    ghosts_off(hybrid.system, moves.waters[moves.ghosts].reshape(-1))
    schedule = fixed_schedule(["morph"], {"morph": 5})
    return moves, hybrid, every, schedule, box


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_waters_switched_in_a_context_of_a_hybrid_give_the_energies_of_a_hybrid_built_so(hybrid_gcmc_leg, backend):
    """A Context of a hybrid leg with its ghosts off whose ghosts are switched on gives the energies of the hybrid
    with every water in every window (charges in the NonbondedForce the plugin replaces, Lennard-Jones of the
    ligand in the custom force on the stock kernels), and switched back those it had."""
    import openmm.unit as u

    from cufep.sampling.gcmc import WaterSwitch

    moves, hybrid, every, schedule, box = hybrid_gcmc_leg
    be = get_backend(backend)

    def context_of(system):
        ctx = be.make_context(be.prepare_system(system),
                              be.make_integrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtoseconds, seed=3), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(hybrid.positions)
        return ctx

    def numbers(ctx):
        ctx.setPositions(hybrid.positions)
        return make_state_evaluator(ctx, schedule.states, 300.0, None).reduced_potentials()[0]

    leg, all_on = context_of(hybrid.system), context_of(every)
    u_leg, u_every = numbers(leg), numbers(all_on)
    assert np.abs(u_every - u_leg).max() > 10.0                          # kT: the ghosts of the start
    switch = WaterSwitch(leg, moves)
    assert switch.apply(np.ones(len(moves.waters), dtype=bool)) == len(moves.ghosts)
    assert np.allclose(numbers(leg), u_every, rtol=0.0, atol=0.02)
    assert switch.apply(moves.initial_on()) == len(moves.ghosts)
    assert np.allclose(numbers(leg), u_leg, rtol=0.0, atol=0.02)


@needs_cuda
@pytest.mark.parametrize("backend", ["openmm", pytest.param("cupotential", marks=needs_plugin)])
def test_replica_exchange_of_a_hybrid_leg_with_grand_canonical_moves_of_water(tmp_path, hybrid_gcmc_leg, backend):
    """Replica exchange of a hybrid leg with the moves and no barostat: one move per replica every 0.2 ps (every
    second iteration of 50 steps of 2 fs) in the state of the hybrid it is simulated in; finite energies, counts by
    window, insertions made, the cell fixed."""
    moves, hybrid, _every, schedule, box = hybrid_gcmc_leg
    spec = LegSpec(name="gpu-hybrid-gcmc", system_xml=mm.XmlSerializer.serialize(hybrid.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=hybrid.system.getNumParticles(),
                   protocol="hybrid", backend=backend, temperature_K=300.0, pressure_bar=None, barostat="none",
                   timestep_fs=2.0, steps_per_iteration=50, seed=3, frame_atoms=list(hybrid.ligand_atoms), gcmc=moves)
    k = len(schedule.states)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[0], log=lambda m: None) as s:
        s.initialize(hybrid.positions, box)
        s.run(10, record_stride=2, checkpoint_stride=5)
    data = read_samples(tmp_path / "hrex")
    own = data["u"][np.arange(10)[:, None], np.arange(k), data["states"]]
    assert np.all(np.isfinite(own))
    counts = data["gcmc"]
    assert counts.shape == (10, k, 5)
    assert (counts[:, :, 0] + counts[:, :, 2]).sum(axis=1).tolist() == [0, k * moves.attempts] * 5
    assert counts[:, :, 1].sum() > 0
    assert np.allclose(data["volume"], abs(np.linalg.det(box)), rtol=1e-6)


@needs_plugin
def test_backends_agree_at_every_state_of_a_hybrid_that_changes_the_charge():
    """Toluene into phenolate with an alchemical water that is Na+ in B: each backend gives the differences between
    the windows that a tight Ewald sum gives (the water's charges as offsets under lambda_electrostatics_core in the
    NonbondedForce the plugin replaces, its Lennard-Jones in the custom force on the stock kernels)."""
    from conftest import build_hybrid_edge, hybrid_edge_topology

    from cufep.rbfe.charge import alchemical_water, ion_parameters, water_into_ion
    from cufep.rbfe.hybrid import make_hybrid_system

    ring = {**{i: i for i in range(1, 7)}, **{10 + k: 7 + k for k in range(5)}}
    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge("[O-]c1ccccc1", ring,
                                                                                         box_nm=3.2)
    top_a = hybrid_edge_topology(system_a.getNumParticles(), ligand_a, system_a)
    water = alchemical_water(top_a, pos_a, 0.8)
    water_into_ion(system_b, [i - len(ligand_a) for i in water],
                   ion_parameters(system_a, top_a, +1, mm.app.ForceField("amber14/tip3p.xml")))
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, alchemical_water=water)
    schedule = fixed_schedule(["morph"], {"morph": 8})
    assert_backends_give_the_energies_of_a_tight_ewald_sum(hybrid.system, schedule.states, hybrid.positions, box,
                                                           pressure=1.0)

