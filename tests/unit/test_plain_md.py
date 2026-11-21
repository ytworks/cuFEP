"""The unrestrained MD of the fully interacting system: which backend runs it and how it recovers."""
import numpy as np
import openmm.unit as unit
import pytest
from conftest import lj_particles

from cufep.backends.base import BackendError
from cufep.sampling import equilibrate


def settings(**changes):
    values = dict(nvt_ns=0.0001, npt_ns=0.0001, production_ns=0.0004, frame_interval_ps=0.1, platform="Reference",
                  precision="double", timestep_fs=4.0, seed=3, md_backend="cupotential", minimize_iterations=20)
    values.update(changes)
    return equilibrate.EquilibrationSettings(**values)


def test_the_unrestrained_md_runs_on_the_backend_of_the_leg(other_backend):
    """Minimisation, NVT and NPT run on stock OpenMM; the unrestrained MD, and only it, on the backend
    that the configuration names."""
    system, positions, box = lj_particles()
    lines = []
    result = equilibrate.pre_equilibrate(system, positions, box, list(range(8)), settings(), None, lines.append)
    assert other_backend.contexts == 1                       # the Context of the unrestrained MD
    assert set(other_backend.asked) == {"openmm", "cupotential"}
    assert result.notes == [] and result.timestep_fs == 4.0
    assert result.frames.shape == (4, 8, 3) and result.simulated_ns == pytest.approx(0.0004)
    assert not [line for line in lines if "stock OpenMM" in line]


def test_a_backend_that_cannot_run_the_system_leaves_the_md_to_stock_openmm(other_backend):
    system, positions, box = lj_particles()
    other_backend.refuse = True
    lines = []
    result = equilibrate.pre_equilibrate(system, positions, box, list(range(8)), settings(), None, lines.append)
    assert other_backend.contexts == 0 and result.frames.shape[0] == 4 and result.backend == "openmm"
    assert lines == ["pre-equilibration MD on stock OpenMM (cupotential cannot run it: this System is not for the "
                     "plugin)"]
    # the result says so too, as it does for a run that failed (a recovery is in the log and the result)
    assert result.notes == ["the pre-equilibration MD ran on stock OpenMM: cupotential cannot run it (this System is "
                            "not for the plugin)"]


def test_an_md_that_fails_on_the_backend_is_repeated_on_stock_openmm_and_says_so(other_backend):
    system, positions, box = lj_particles()
    other_backend.fail = True
    lines = []
    result = equilibrate.pre_equilibrate(system, positions, box, list(range(8)), settings(), None, lines.append)
    assert result.frames.shape[0] == 4 and result.timestep_fs == 4.0
    assert result.notes == ["the pre-equilibration MD failed on cupotential and was repeated on stock OpenMM"]
    assert lines == ["pre-equilibration MD failed on cupotential (the plugin stopped); repeating it on stock OpenMM"]


def test_an_unstable_md_is_repeated_once_at_2_fs_and_says_so(monkeypatch):
    """A run that fails at the time step of the configuration is made again from the start at 2 fs;
    one that fails there too is an error that names both."""
    system, positions, box = lj_particles()
    run = equilibrate._production
    failing = {4.0}

    def production(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs):
        if dt_fs in failing:
            raise equilibrate.EquilibrationError("energy became non-finite during the pre-equilibration MD")
        return run(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs)

    monkeypatch.setattr(equilibrate, "_production", production)
    lines = []
    result = equilibrate.pre_equilibrate(system, positions, box, list(range(8)), settings(md_backend="openmm"), None,
                                         lines.append)
    assert result.timestep_fs == 2.0
    assert result.notes == ["time step lowered automatically to 2 fs after an unstable run at 4 fs"]
    assert lines == ["pre-equilibration at 4 fs failed: energy became non-finite during the pre-equilibration MD"]
    failing.add(2.0)
    with pytest.raises(equilibrate.EquilibrationError, match=r"failed at every time step tried \(\[4.0, 2.0\]\)"):
        equilibrate.pre_equilibrate(system, positions, box, list(range(8)), settings(md_backend="openmm"), None,
                                    lines.append)


def test_the_plain_md_of_a_leg_takes_its_settings_from_the_configuration():
    """Temperature, pressure, friction, the frequency of the barostat, the backend, its platform and
    its precision are those of the YAML for every plain MD of a leg."""
    from cufep.config.schema import Config
    from cufep.workflow import plain_md_settings

    cfg = Config.model_validate({
        "system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
        "thermodynamics": {"temperature_K": 310.0, "pressure_bar": 2.0},
        "dynamics": {"friction_per_ps": 0.5, "barostat_frequency": 50},
        "backend": {"name": "openmm", "platform": "CPU", "precision": "double"}})
    made = plain_md_settings(cfg, 3.0, "membrane", 17, production_ns=1.5, frame_interval_ps=10.0)
    assert (made.temperature_K, made.pressure_bar) == (310.0, 2.0)
    assert (made.friction_per_ps, made.barostat_frequency) == (0.5, 50)
    assert (made.md_backend, made.platform, made.precision) == ("openmm", "CPU", "double")
    assert (made.timestep_fs, made.barostat, made.seed, made.production_ns, made.frame_interval_ps) == \
        (3.0, "membrane", 17, 1.5, 10.0)
    single = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                    "backend": {"precision": "single"}})
    # single precision too (the owner's decision of 2026-11-21 after the review of f5a77ce: plain MD ran in
    # mixed precision where single was asked, which no document said and the log did not show)
    assert plain_md_settings(single, 4.0, "isotropic", 1).precision == "single"
    assert plain_md_settings(single, 4.0, "isotropic", 1).md_backend == "cupotential"


def configured(fallback="openmm", precision="mixed"):
    """The settings of a short plain MD as a configuration with this ``backend`` section gives them."""
    from cufep.config.schema import Config
    from cufep.workflow import plain_md_settings

    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "backend": {"name": "cupotential", "fallback": fallback, "precision": precision}})
    made = plain_md_settings(cfg, 4.0, "isotropic", 3, nvt_ns=0.0001, npt_ns=0.0001, production_ns=0.0004,
                             frame_interval_ps=0.1, minimize_iterations=20)
    made.platform = "Reference"                 # the tests have no GPU; the backend is other_backend
    return made


def test_without_a_fallback_a_backend_that_cannot_run_the_md_stops_the_leg(other_backend):
    """``backend.fallback: none`` holds for the plain MD of a leg as for its alchemical run: stock
    OpenMM does not take over. Before this was so, the MD moved to stock OpenMM whatever the
    setting, and only the log said so."""
    system, positions, box = lj_particles()
    other_backend.refuse = True
    lines = []
    with pytest.raises(BackendError, match="cannot run the pre-equilibration MD .* and backend.fallback is none"):
        equilibrate.pre_equilibrate(system, positions, box, list(range(8)), configured("none"), None, lines.append)
    assert lines == [] and other_backend.contexts == 0
    # with the fallback of the default, stock OpenMM runs it
    assert equilibrate.pre_equilibrate(system, positions, box, list(range(8)), configured(), None,
                                       lines.append).backend == "openmm"


def test_without_a_fallback_an_md_that_fails_on_the_backend_is_not_repeated_on_stock_openmm(other_backend):
    system, positions, box = lj_particles()
    other_backend.fail = True
    lines = []
    with pytest.raises(equilibrate.EquilibrationError, match=r"failed at every time step tried .*the plugin stopped"):
        equilibrate.pre_equilibrate(system, positions, box, list(range(8)), configured("none"), None, lines.append)
    assert lines[0] == ("pre-equilibration MD failed on cupotential (the plugin stopped); backend.fallback is none, "
                        "so it is not repeated on stock OpenMM")
    assert not [line for line in lines if "repeating it on stock OpenMM" in line]


def test_the_plain_md_runs_the_backend_in_the_precision_of_the_alchemical_run(other_backend):
    """The plugin has no double precision: asked for it, the alchemical run of a leg uses the plugin
    in mixed precision. So does its plain MD (it gave the plugin "double", which the plugin refuses,
    and ran on stock OpenMM instead)."""
    system, positions, box = lj_particles()
    result = equilibrate.pre_equilibrate(system, positions, box, list(range(8)), configured(precision="double"),
                                         None, lambda line: None)
    assert other_backend.precisions == ["mixed"] and result.backend == "cupotential"
    from cufep.backends.base import backend_precision

    assert [backend_precision("cupotential", p) for p in ("mixed", "single", "double")] == ["mixed", "single", "mixed"]
    assert [backend_precision("openmm", p) for p in ("mixed", "single", "double")] == ["mixed", "single", "double"]


def test_the_backend_is_asked_about_the_box_the_md_starts_in_and_gets_the_system_in_that_box(other_backend,
                                                                                            monkeypatch):
    """Review of eb6f89f (P2). The backend was asked whether it can run a copy of the System in the box the MD
    starts in, and then made its Context of the System as it was prepared, whose box can be another one: a
    plugin that needs 3 x the cutoff found 2.71 nm enough and then refused the 2.69 nm of the prepared System,
    outside the fallback, and the leg stopped. Both are the System in the box of the start; the threads of
    the CPU platform are those of the configuration."""
    system, positions, box = lj_particles()
    started_in = np.asarray(box, dtype=float) * 1.01
    seen, threads = [], []

    def edge(of):
        return round(of.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometer), 6)

    prepare = other_backend.prepare_system

    def asked(self, checked):
        seen.append(("asked", edge(checked)))

    def given(self, to_run):
        seen.append(("given", edge(to_run)))
        threads.append(self.cpu_threads)
        return prepare(self, to_run)

    monkeypatch.setattr(other_backend, "check_system", asked)
    monkeypatch.setattr(other_backend, "prepare_system", given)
    start = (positions, np.zeros_like(positions), started_in)
    equilibrate.unrestrained_md(system, start, list(range(8)), settings(cpu_threads=3), 4.0, None, lambda line: None)
    assert {kind for kind, _edge in seen} == {"asked", "given"}
    assert {length for _kind, length in seen} == {round(float(started_in[0][0]), 6)}
    assert set(threads) == {3}
    assert edge(system) == round(float(np.asarray(box)[0][0]), 6)         # the System of the caller is as it was

    # the settings of the YAML give the threads (resources.cpu_threads_per_worker) to every plain MD of a leg
    from cufep.config.schema import Config
    from cufep.workflow import plain_md_settings

    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "resources": {"cpu_threads_per_worker": 4}})
    assert plain_md_settings(cfg, 4.0, "isotropic", 1).cpu_threads == 4


def test_a_pre_equilibration_can_hold_the_solute_where_it_was_given(monkeypatch):
    """``restraints.selection: interactions`` chooses the restraint from MD that starts from the pose that was
    given, as the method of Ding et al. starts it from the minimised input (the owner's decision of 2026-11-21:
    after 10 ns of free MD two runs of the A2A example had the ligand in two states and chose two restraints).
    The pre-equilibration of such a leg holds the solute through NVT and NPT and ends there; that of a bilayer
    goes on with the solute still held, without the wall of the first two parts. The other selections are as
    they were: free MD after NVT and NPT."""
    from cufep.config.schema import Config
    from cufep.workflow import equil_made_with, pre_equilibration_parts, settings_differences

    def config(**sections):
        return Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"}, **sections})

    default, searched = config(), config(restraints={"selection": "auto"})
    assert pre_equilibration_parts(default, "complex", False) == (0.0, True)
    assert pre_equilibration_parts(default, "complex", True) == (10.0, True)
    assert pre_equilibration_parts(default, "solvent", False) == (0.5, False)
    assert pre_equilibration_parts(searched, "complex", False) == (5.0, False)
    assert pre_equilibration_parts(searched, "complex", True) == (10.0, False)
    # the two are different pre-equilibrations, and a file of an earlier version is one of free MD
    held, free = equil_made_with(default, "complex", False, 7), equil_made_with(searched, "complex", False, 7)
    assert settings_differences(free, held) == ["pre_equilibration.md_ns: 5.0 -> not given",
                                                "solute_held: not recorded -> True"]
    assert "solute_held" not in equil_made_with(default, "solvent", False, 7)
    # md_ns is the free MD of the other selections: a change of it is no other pre-equilibration of this one (the
    # fourth pass of Codex: a run with the default selection was refused for it), and it is one of theirs
    other = {"sampling": {"pre_equilibration": {"complex": {"md_ns": 1.0}}}}
    assert settings_differences(held, equil_made_with(config(**other), "complex", False, 7)) == []
    assert settings_differences(held, equil_made_with(config(**other), "complex", True, 7)) == [
        "membrane_equilibration_ns: None -> 10.0"]
    assert settings_differences(free, equil_made_with(config(restraints={"selection": "auto"}, **other), "complex",
                                                      False, 7)) == ["pre_equilibration.md_ns: 5.0 -> 1.0"]

    system, positions, box = lj_particles()
    made = []
    context = equilibrate._context

    def seen(system, backend, settings, dt_fs, device, barostat, restraint_reference, seed, barostat_seed=None,
             wall=True):
        made.append((barostat, restraint_reference is not None, wall))
        return context(system, backend, settings, dt_fs, device, barostat, restraint_reference, seed, barostat_seed,
                       wall)

    monkeypatch.setattr(equilibrate, "_context", seen)
    held_atoms = [0, 1, 2, 3]
    # no MD after the NPT: the result is its end, with the one frame of it
    ends = equilibrate.pre_equilibrate(system, positions, box, list(range(8)),
                                       settings(md_backend="openmm", production_ns=0.0, hold_solute=True,
                                                restrained_atoms=held_atoms), None, lambda line: None)
    assert made == [(False, True, True), (True, True, True)]                  # NVT and NPT, both with the solute held
    assert ends.simulated_ns == 0.0 and ends.frames.shape == (1, 8, 3) and ends.backend == "openmm"
    assert np.allclose(ends.frames[0], ends.positions[list(range(8))], atol=1e-6) and ends.notes == []
    moved = np.linalg.norm(ends.positions - np.asarray(positions), axis=1)
    assert moved[held_atoms].max() < 0.12 and np.all(np.isfinite(ends.velocities))
    # MD after the NPT with the solute held: its Context has the position restraints and no wall
    made.clear()
    goes_on = equilibrate.pre_equilibrate(system, positions, box, list(range(8)),
                                          settings(md_backend="openmm", hold_solute=True, restrained_atoms=held_atoms),
                                          None, lambda line: None)
    assert made[-1] == (True, True, False) and goes_on.simulated_ns == pytest.approx(0.0004)
    made.clear()
    equilibrate.pre_equilibrate(system, positions, box, list(range(8)),
                                settings(md_backend="openmm", restrained_atoms=held_atoms), None, lambda line: None)
    assert made[-1] == (True, False, False)                                   # the free MD of the other selections

