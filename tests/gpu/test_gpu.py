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
@pytest.fixture(scope="module")
def rest2_leg():
    """Two dipeptides in a 3.2 nm water box; one is the ligand, part of the other is hot too."""
    from conftest import build_dipeptides_in_water

    system, top, pos, box, ligand, other = build_dipeptides_in_water(box_nm=3.2)
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


@needs_plugin
def test_rest2_backends_agree(rest2_leg):
    import openmm.unit as u

    alch, schedule, pos, box, _ = rest2_leg
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
    assert np.max(np.abs(diff)) < 0.1, diff            # kT, as without REST2 (the fixture has no dispersion correction)


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
    assert np.any(data["states"][1:] != data["states"][:-1])               # replicas exchange


@needs_plugin
def test_the_system_of_a_validation_with_a_rest2_pocket_agrees_on_both_backends():
    """Review of PR 1 (P2): `cufep validate` compares the System that a run samples, with REST2 and its region;
    here the pocket region holds the whole second peptide and the windows in the middle carry the scale 1.5."""
    import openmm.unit as u
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
    rows = {}
    for name in ("openmm", "cupotential"):
        backend = get_backend(name)
        ctx = backend.make_context(backend.prepare_system(built.alch.system),
                                   backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
        ctx.setPositions(pos)
        rows[name] = StateEnergyEvaluator(ctx, schedule.states, 300.0, None).reduced_potentials()[0]
        del ctx
    diff = (rows["cupotential"] - rows["cupotential"][0]) - (rows["openmm"] - rows["openmm"][0])
    assert np.max(np.abs(diff)) < 0.1, diff


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

