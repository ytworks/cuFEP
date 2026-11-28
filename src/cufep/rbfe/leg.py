"""One leg of an edge: the hybrid System of its end states, sampled by Hamiltonian replica exchange.

A leg of the protocol ``hybrid`` (:mod:`cufep.engine.protocol`) on the engine of a leg (:mod:`cufep.engine.leg`):
the pre-equilibration of the hybrid System at lambda 0 (ligand A, the unique atoms of B dummies), the lambda
windows, the replica exchange (:func:`cufep.workflow.sample_leg`), the fallback to stock OpenMM, and the analysis
of the samples (:func:`cufep.engine.leg.path_result`): the free energy of transforming A into B in the leg.

The lambda windows (``lambda_schedule``)
    adaptive, ``legs: each`` (default): each leg runs its own precalculation of adaptive lambda scheduling, the
        windows of an iteration at the same time on the GPUs it is granted (as the ABFE).
    adaptive, ``legs: solvent``: the solvent leg runs the precalculation and the complex leg takes its windows, as
        Midgley et al. 2025 decide the windows of a relative transformation from the ligand in solution.
    fixed: the windows of ``windows``.
    Windows that a leg does not choose itself start, all of them, from the end of its pre-equilibration, minimised in
    their own state (OpenFE minimises every replica, 5000 steps) and 10 ps of MD in it, all windows at the same time
    on the GPUs of the leg; the equilibration of the replica exchange then relaxes them further.

Grand canonical moves of water (``sampling.gcmc``, on by default in the complex leg)
    As in the ABFE (:mod:`cufep.sampling.gcmc`): the waters farthest from the ligand are ghosts from the start, the
    pre-equilibration with its barostat settles the cell without them, and the windows and the replica exchange keep
    the cell of its last configuration, without a barostat, with the moves in a sphere around the heavy atoms of the
    hybrid ligand (those of A and of B).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from cufep.engine.jobs import Job
from cufep.workflow import KCAL, WorkflowError

#: the stage of the one coordinate of a relative transformation
STAGES = ["morph"]
#: how long the complex leg waits between two looks for the windows of the solvent leg
WAIT_SECONDS = 20.0
#: how the windows that a leg does not choose itself get the configurations of their replicas: from the end of the
#: pre-equilibration, minimised in their own state (OpenFE's 5000 steps of minimisation) and a short MD in it
PRE_EQUILIBRATION_START = "the end of the pre-equilibration, minimised in each window"
MINIMIZE_STEPS = 5000
START_MD_PS = 10.0
#: how the result of a leg tells the user to run it again
RBFE_COMMAND = "cufep rbfe run CONFIG --edge NAME"


@dataclass(kw_only=True)
class EdgeLegJob(Job):
    """A leg of an edge: ``edge_dir`` is the directory of the edge, ``config`` the configuration of the relative
    run."""

    edge: str
    edge_dir: str

    unit_key: ClassVar[str] = "edge"

    @property
    def name(self) -> str:
        return self.edge

    @property
    def unit_dir(self) -> Path:
        return Path(self.edge_dir)

    def runner(self):
        return run_hybrid_leg_safe


def run_hybrid_leg_safe(job: EdgeLegJob) -> dict:
    """:func:`run_hybrid_leg` that records a failure instead of raising (for the job processes)."""
    import traceback

    from cufep.workflow import begin_leg, write_leg_error

    begin_leg(job.leg_dir, job)
    try:
        return run_hybrid_leg(job)
    except Exception as exc:  # noqa: BLE001
        err = write_leg_error(job.leg_dir, f"{type(exc).__name__}: {exc}", job, traceback.format_exc(limit=12))
        print(f"[{job.tag}] FAILED: {err['error']}", flush=True)
        return err


def hybrid_topology(top_a, top_b, unique_b: list[int]):
    """The topology of the hybrid System: that of A, and the unique atoms of B in a residue LIG of a chain of their
    own (their bonds among themselves; those to the core are terms of the System)."""
    import openmm.app as app

    out = app.Topology()
    out.setPeriodicBoxVectors(top_a.getPeriodicBoxVectors())
    atoms = {}
    for chain in top_a.chains():
        c = out.addChain(chain.id)
        for res in chain.residues():
            r = out.addResidue(res.name, c, res.id, res.insertionCode)
            for atom in res.atoms():
                atoms[("a", atom.index)] = out.addAtom(atom.name, atom.element, r, atom.id)
    for bond in top_a.bonds():
        out.addBond(atoms[("a", bond[0].index)], atoms[("a", bond[1].index)])
    if unique_b:
        b_atoms = list(top_b.atoms())
        r = out.addResidue("LIG", out.addChain("X"))
        for j in unique_b:
            atoms[("b", j)] = out.addAtom(b_atoms[j].name + "B", b_atoms[j].element, r)
        wanted = set(unique_b)
        for bond in top_b.bonds():
            i, j = bond[0].index, bond[1].index
            if i in wanted and j in wanted:
                out.addBond(atoms[("b", i)], atoms[("b", j)])
    return out


@dataclass
class HybridLeg:
    hybrid: object                # cufep.rbfe.hybrid.HybridSystem
    xml: str
    prep: dict                    # what the shared functions read of a prepared system (cufep.workflow.load_prepared)
    hot_atoms: list
    hot_residues: list
    barostat: str
    membrane_info: dict
    info: dict


def build_leg(rcfg, edge_dir: Path, leg: str) -> HybridLeg:
    """The hybrid System of a leg from its prepared end states, and a prepared system of it."""
    import openmm as mm

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.prep import load_end_states
    from cufep.workflow import rest2_region

    end = load_end_states(edge_dir, leg)
    a, b, info = end["a"], end["b"], end["info"]
    rest2 = rcfg.sampling.rest2
    rest2_on = rest2.scale > 1.0
    hot_atoms, hot_residues = [], []
    if rest2_on and leg == "complex":
        hot_atoms, hot_residues = rest2_region(rest2, a["topology"], a["positions"], a["ligand"], a["protein"],
                                               info.get("input_chains"))
    hybrid = make_hybrid_system(mm.XmlSerializer.deserialize(a["system_xml"]), a["positions"], a["ligand"],
                                mm.XmlSerializer.deserialize(b["system_xml"]), b["positions"], b["ligand"],
                                end["core_map"], softcore_alpha=rcfg.alchemy.softcore_alpha, rest2=rest2_on,
                                rest2_atoms=hot_atoms)
    unique_b = sorted(j for j in b["ligand"] if j not in set(end["core_map"].values()))
    topology = hybrid_topology(a["topology"], b["topology"], unique_b)
    xml = mm.XmlSerializer.serialize(hybrid.system)
    prep = {"input_chains": info.get("input_chains"), "system_xml": xml, "topology": topology,
            "positions": hybrid.positions, "box": a["box"], "ligand": list(hybrid.ligand_atoms),
            "solute": sorted(set(a["solute"]) | set(hybrid.unique_new)), "protein": list(a["protein"])}
    leg_info = info["legs"][leg]
    return HybridLeg(hybrid=hybrid, xml=xml, prep=prep, hot_atoms=hot_atoms, hot_residues=hot_residues,
                     barostat=leg_info["barostat"], membrane_info=leg_info.get("membrane_info") or {}, info=info)


def hybrid_settings(rcfg, leg: str | None = None) -> dict:
    """The settings of a relative run that the samples of a leg depend on beyond its System; for the leg that makes
    grand canonical moves of water (the complex leg) also those of the moves, as the ABFE records them
    (:func:`cufep.workflow.gcmc_record`; the pressure of the samples is that of the cell they are fixed in)."""
    from cufep.workflow import gcmc_in, gcmc_record

    settings = {"rbfe": {"alchemy": rcfg.alchemy.model_dump(mode="json"),
                         "mapping": rcfg.mapping.model_dump(mode="json")},
                "sampling.rest2": rcfg.sampling.rest2.model_dump(mode="json")}
    if gcmc_in(rcfg, leg):
        settings["gcmc"] = gcmc_record(rcfg)
    return settings


def hybrid_gcmc(rcfg, built: HybridLeg, leg: str, log=print):
    """The grand canonical moves of water of a leg (:class:`cufep.sampling.gcmc.GcmcSpec`), or None for a leg
    without them. With them the waters that are ghosts at the start are switched off in the hybrid System of
    ``built`` (changed in place, with its XML and its prepared system), so that every MD of the leg has them: as
    :func:`cufep.workflow.leg_system` makes the System of a leg of the ABFE. The soft-core of the stand-in energy is
    that of the ABFE (cuFEP's own: the stand-in decides only which trials are tested with the full potential)."""
    import openmm as mm

    from cufep.config.schema import AlchemyConfig
    from cufep.sampling.gcmc import GcmcError, gcmc_spec, ghosts_off, hybrid_parameters
    from cufep.workflow import gcmc_in, gcmc_water_values

    if not gcmc_in(rcfg, leg):
        return None
    g, prep, stand_in = rcfg.sampling.gcmc, built.prep, AlchemyConfig()
    mu, volume = gcmc_water_values(rcfg)
    try:
        spec = gcmc_spec(built.hybrid.system, prep["topology"], prep["positions"], np.asarray(prep["box"], dtype=float),
                         prep["ligand"], rcfg.dynamics.cutoff_nm, stand_in.softcore_alpha, stand_in.softcore_b,
                         rcfg.thermodynamics.temperature_K, g.interval_ps, g.attempts, g.batch_size,
                         g.radius_margin_nm, mu, volume, None if g.ghost_waters == "auto" else int(g.ghost_waters))
    except GcmcError as exc:
        raise WorkflowError(f"sampling.gcmc cannot be used with this system: {exc}") from exc
    spec.hybrid = hybrid_parameters(built.hybrid, spec.ligand)
    ghosts_off(built.hybrid.system, spec.waters[spec.ghosts].reshape(-1))
    built.xml = mm.XmlSerializer.serialize(built.hybrid.system)
    built.prep = {**prep, "system_xml": built.xml}
    log(f"grand canonical moves of water: sphere of {10 * spec.radius_nm:.1f} A around the heavy atoms of the hybrid "
        f"ligand, Adams value B = {spec.adams:.3f} (excess chemical potential "
        f"{spec.excess_chemical_potential_kj_per_mol / KCAL:.3f} kcal/mol, standard volume "
        f"{1000 * spec.standard_volume_nm3:.3f} A^3), {len(spec.ghosts)} of the {len(spec.waters)} waters are ghosts "
        "at the start (those farthest from the ligand); the windows and the replica exchange have no barostat")
    return spec


def hybrid_leg_spec(rcfg, built: HybridLeg, gcmc, leg: str, name: str, states: list[dict], backend: str,
                    timestep_fs: float, steps_per_iteration: int, **more):
    """What the workers of a relative leg are given (:class:`cufep.sampling.worker.LegSpec`), from the configuration
    and the hybrid System of the leg (``built``, with its ghosts off where ``gcmc`` gives the moves). Its
    ``conditions()`` are what the samples record, so the run of a leg and the check before a run
    (:func:`check_leg_reuse`) make it here, in one place. ``timestep_fs`` is the time step of the pre-equilibration,
    ``more`` what only a run needs (seed, frame atoms, the share of the GPU)."""
    from cufep.engine.protocol import HYBRID
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import _precision

    barostat = built.barostat
    return LegSpec(name=name, system_xml=built.xml, states=states, n_atoms=built.hybrid.system.getNumParticles(),
                   protocol=HYBRID, backend=backend, precision=_precision(rcfg, backend),
                   platform=rcfg.backend.platform, temperature_K=rcfg.thermodynamics.temperature_K,
                   pressure_bar=None if gcmc is not None else rcfg.thermodynamics.pressure_bar,
                   barostat="none" if gcmc is not None else barostat,
                   barostat_frequency=rcfg.dynamics.barostat_frequency, exchange=rcfg.sampling.exchange,
                   friction_per_ps=rcfg.dynamics.friction_per_ps, timestep_fs=timestep_fs,
                   steps_per_iteration=steps_per_iteration, cpu_threads=rcfg.resources.cpu_threads_per_worker,
                   box_safety=1.3 if barostat == "membrane" and gcmc is None else None,
                   settings=hybrid_settings(rcfg, leg), gcmc=gcmc, **more)


def check_leg_reuse(rcfg, edge_dir: Path, leg: str, repeat: int) -> None:
    """Before a run writes or samples anything: the lambda windows and the samples (or the checkpoint of their
    start) that the directory of a leg holds must be ones this run can go on from. The checks the leg makes where
    it uses the files (:func:`run_hybrid_leg`: the settings its windows were chosen with, and every condition of
    its samples with the checksum of the System, for which the hybrid System of this run is built), made first
    from the files alone, so that a run that will be refused leaves the directory, its record of the configuration
    and the results of its legs as they were. Second review of PR #21: a finished edge run again with another
    chemical potential of water, or a run of an earlier version of the hybrid continued, was refused by its legs
    after the run had replaced ``resolved_config.yaml`` and the legs had set their results aside. Raises
    WorkflowError or ConditionsError."""
    from cufep.engine.leg import check_kept_windows, start_conditions
    from cufep.sampling.storage import other_conditions, stored_conditions
    from cufep.workflow import (
        _choose_backend,
        _schedule_of_this_leg,
        check_windows_settings,
        has_starts_of,
        planned_iterations,
        settings_differences,
        start_record_of,
        starts_are_of,
    )

    leg_dir = Path(edge_dir) / f"repeat_{repeat}" / leg
    sched_file, start_file, hrex = leg_dir / "schedule.yaml", leg_dir / "start_configs.npz", leg_dir / "hrex"
    if not sched_file.exists():
        return                 # no windows yet (samples without their windows: check_samples_have_their_windows)
    quiet = lambda _message: None                                    # noqa: E731  the leg logs it when it runs
    settings = hybrid_settings(rcfg, leg)
    schedule = _schedule_of_this_leg(sched_file, leg, STAGES)
    rest2_scale = rcfg.sampling.rest2.scale if rcfg.sampling.rest2.scale > 1.0 else 1.0
    if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
        raise WorkflowError(f"the lambda schedule in {sched_file} was made with a REST2 scale of "
                            f"{schedule.rest2_scale:g} but sampling.rest2.scale is {rcfg.sampling.rest2.scale:g}. "
                            "Nothing was changed. Use the same value, or start the leg anew with --reschedule.")
    if start_file.exists():
        made = start_record_of(start_file)
        if starts_are_of(made, schedule):                  # start configurations of other windows are made again
            check_windows_settings(made, settings, start_file, quiet)
    n_stored, conditions = stored_conditions(hrex)
    if n_stored == 0:
        # no samples. A leg stopped at the checkpoint of the start of its replica exchange, whose start
        # configurations are made again, holds the conditions recorded with that checkpoint against those of this
        # run (check_kept_windows in run_hybrid_leg), as it does those of samples: they are compared here too
        # (third review of PR #21: such a leg of a System this version builds otherwise was refused after the run
        # had replaced resolved_config.yaml). With its start configurations the replica exchange takes the
        # conditions of this run for a leg without samples (LegStorage.check_conditions)
        if not (hrex / "checkpoint.npz").exists() or has_starts_of(leg_dir, schedule):
            return
        conditions = start_conditions(leg_dir)
    if conditions is None:
        raise WorkflowError(f"the samples in {hrex} do not record their conditions; nothing was changed")
    # the settings first: no System has to be built to compare them
    if isinstance(conditions.get("settings"), dict):
        differing = settings_differences(conditions["settings"], settings)
        if differing:
            raise other_conditions(n_stored, hrex, differing)
    # then every condition, with the checksum of the System this run would sample: the hybrid System is built
    equil_file = leg_dir / "equil.npz"
    if not equil_file.exists():
        raise WorkflowError(f"{hrex} holds samples of this leg but {equil_file}, which they started from, is not "
                            "there. Nothing was changed; run the calculation in a new output.dir")
    with np.load(equil_file) as eq:
        timestep_fs = float(eq["timestep_fs"])
    built = build_leg(rcfg, Path(edge_dir), leg)
    gcmc = hybrid_gcmc(rcfg, built, leg, quiet)
    plan = planned_iterations(rcfg.sampling.time_per_iteration_ps, timestep_fs, rcfg.sampling.equilibration(leg),
                              rcfg.sampling.production_ns)
    # the samples of a leg stay on the backend they were made with; a leg without samples runs on the one this
    # run chooses for its System (cufep.engine.leg.chosen_backend)
    backend = (str(conditions.get("backend") or rcfg.backend.name) if n_stored else
               _choose_backend(rcfg, built.hybrid.system, quiet)[0])
    spec = hybrid_leg_spec(rcfg, built, gcmc, leg, f"{Path(edge_dir).name}/r{repeat}/{leg}",
                           [state.to_dict() for state in schedule.states], backend, timestep_fs,
                           plan.steps_per_iteration)
    check_kept_windows(leg_dir, schedule, spec.conditions())


def leg_reuse_problem(config: dict, edge_dir: str, leg: str, repeat: int) -> str | None:
    """:func:`check_leg_reuse` for a process of its own (the legs of a run are checked at the same time): the
    refusal as text, or None."""
    from cufep.rbfe.config import RbfeConfig
    from cufep.sampling.storage import ConditionsError

    try:
        check_leg_reuse(RbfeConfig.model_validate(config), Path(edge_dir), leg, repeat)
    except (WorkflowError, ConditionsError) as exc:
        return str(exc)
    return None


def _schedule_from_solvent(rcfg, edge_dir: Path, repeat: int, log, control=None):
    """The windows of the solvent leg of the edge, when it has them; waits for them."""
    from cufep.alchemy.schedule import Schedule

    solvent = Path(edge_dir) / f"repeat_{repeat}" / "solvent"
    told = False
    while True:
        if (solvent / "schedule.yaml").exists() and (solvent / "start_configs.npz").exists():
            schedule = Schedule.load(solvent / "schedule.yaml")
            schedule.notes.append("the windows of the solvent leg (lambda_schedule.adaptive.legs: solvent)")
            return schedule
        if (solvent / "error.json").exists():
            raise WorkflowError("the solvent leg of this edge failed before it had its windows "
                                f"({solvent / 'error.json'})")
        if not told:
            log("waiting for the lambda windows of the solvent leg (lambda_schedule.adaptive.legs: solvent)")
            told = True
            if control is not None:
                control.serial()
        time.sleep(WAIT_SECONDS)


def run_hybrid_leg(job: EdgeLegJob) -> dict:
    """Run (or resume, or extend) one leg of an edge and return its result."""
    from cufep.alchemy.schedule import AlsSettings, schedule_from_windows
    from cufep.engine.leg import (
        adaptive_windows,
        check_kept_windows,
        chosen_backend,
        finish_leg,
        goes_on_from_its_samples,
        kept_start_record,
        leg_context,
        open_leg,
        pre_equilibrated,
        record_of_kept_windows,
        refuse_samples_without_windows,
        starts_from,
        with_fallback,
    )
    from cufep.engine.protocol import stage_state
    from cufep.mps import thread_percentage
    from cufep.rbfe.config import RbfeConfig
    from cufep.workflow import (
        _schedule_of_this_leg,
        planned_iterations,
        publish_windows,
        sample_leg,
        stored_windows,
    )

    rcfg = RbfeConfig.model_validate(job.config)
    run = open_leg(job, rcfg)
    leg_dir, log, seed = run.leg_dir, run.log, run.seed
    edge_dir = Path(job.edge_dir)
    built = build_leg(rcfg, edge_dir, job.leg)
    gcmc = hybrid_gcmc(rcfg, built, job.leg, log)          # with the moves: the ghosts of the start switched off
    hybrid, prep, barostat = built.hybrid, built.prep, built.barostat
    n_atoms = hybrid.system.getNumParticles()
    is_complex = job.leg == "complex"
    rest2 = rcfg.sampling.rest2
    rest2_scale = rest2.scale if rest2.scale > 1.0 else 1.0
    log(f"hybrid: {len(hybrid.core)} core atoms, {len(hybrid.unique_old)} unique to "
        f"{built.info['ligand_a']}, {len(hybrid.unique_new)} unique to {built.info['ligand_b']}; "
        + (f"REST2 on the ligand and {len(built.hot_atoms)} atoms of {len(built.hot_residues)} residues "
           f"({' '.join(built.hot_residues)}), {rest2_scale:g} times the temperature in the middle"
           if rest2_scale > 1.0 else "REST2 off"))

    # ---- 1. pre-equilibration of the hybrid System at lambda 0 ---------------------------------
    eq, _recorded = pre_equilibrated(run, prep, hybrid.system, barostat, built.membrane_info)
    timestep_fs = float(eq["timestep_fs"])
    start = (np.asarray(eq["positions"]), np.asarray(eq["box"]), np.asarray(eq["velocities"]))
    # (with grand canonical moves of water the windows and the replica exchange have no barostat: they keep the cell
    # of this last configuration, cufep.workflow.alchemical_box)

    # ---- 2. backend -------------------------------------------------------------------------------
    backend_name = chosen_backend(run, hybrid.system)
    iteration_ps = rcfg.sampling.time_per_iteration_ps
    plan = planned_iterations(iteration_ps, timestep_fs, rcfg.sampling.equilibration(job.leg), job.production_ns)
    n_equil, n_prod = plan.n_equilibration, plan.n_production
    ckpt_stride = max(1, int(round(rcfg.sampling.checkpoint_interval_ps / iteration_ps)))
    atoms = list(prep["topology"].atoms())
    frame_atoms = prep["solute"] if is_complex else list(hybrid.ligand_atoms)
    where = {a: i for i, a in enumerate(frame_atoms)}
    frame_ligand = [where[i] for i in hybrid.atoms_a if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    frame_ca = [where[i] for i in prep["protein"] if atoms[i].name == "CA" and i in where] if is_complex else []
    # the settings the samples and the windows of the leg depend on beyond its System
    settings = hybrid_settings(rcfg, job.leg)
    ends = [stage_state(STAGES[0], 0.0).to_dict(), stage_state(STAGES[-1], 1.0).to_dict()]

    def spec_of(states: list[dict], backend: str, **more):
        return hybrid_leg_spec(rcfg, built, gcmc, job.leg, job.tag, states, backend, timestep_fs,
                               plan.steps_per_iteration, mps_thread_percentage=thread_percentage(job.oversubscription),
                               **more)


    def from_pre_equilibration(schedule) -> list:
        """The start of every window that the leg does not choose itself: the end of the pre-equilibration,
        minimised in the state of the window (OpenFE's minimisation, ``MINIMIZE_STEPS``) and ``START_MD_PS`` of MD
        in it (:func:`cufep.engine.leg.starts_from`). The atoms unique to the other ligand do not interact at lambda
        0, so water can sit where they are: a window in which they interact starts from that only after its
        minimisation. The minimisation and this MD run on stock OpenMM, as the minimisation, NVT and NPT of every
        pre-equilibration do (cufep.sampling.equilibrate): OpenMM's minimiser moves to a Context of the CPU platform
        where the forces are very large, and the cuPotential plugin has no kernels there (the ejm_31 -> ejm_49
        complex leg of the second run of the TYK2 example failed so). Only the start configurations come from it;
        the samples are of the backend of the leg."""
        return starts_from(run, spec_of(ends, "openmm", seed=seed + 29), schedule, start, hybrid.system, n_atoms,
                           MINIMIZE_STEPS, START_MD_PS)

    def check_scale(schedule) -> None:
        if abs(schedule.rest2_scale - rest2_scale) > 1e-9:
            raise WorkflowError("the lambda schedule of this leg was made with a REST2 scale of "
                                f"{schedule.rest2_scale:g} but sampling.rest2.scale is {rest2.scale:g}")

    def run_with(backend: str):
        ls = rcfg.lambda_schedule
        # ---- 3. lambda windows ---------------------------------------------------------------
        stored = stored_windows(leg_dir, job.leg, STAGES, settings, log)
        if stored is not None:
            schedule, configs, _made = stored
            log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg already has")
        else:
            t0 = time.perf_counter()
            a = ls.adaptive
            starting = {"start_configurations": PRE_EQUILIBRATION_START, "start_md_ps": START_MD_PS,
                        "minimize_steps": MINIMIZE_STEPS, "start_backend": "openmm"}
            if (leg_dir / "schedule.yaml").exists():
                # the windows this leg has, without start configurations of them (none, or of other windows): its
                # samples are of these windows, so they are kept and only their start configurations are made
                # again, whatever lambda_schedule asks now (review of PR #18). A schedule that is not one of this
                # leg stops the run here, before anything of the leg is written
                schedule = _schedule_of_this_leg(leg_dir / "schedule.yaml", job.leg, STAGES)
                check_scale(schedule)
                production = spec_of([st.to_dict() for st in schedule.states], backend, seed=seed + 17,
                                     frame_atoms=frame_atoms, frame_ligand=frame_ligand, frame_ca=frame_ca)
                check_kept_windows(leg_dir, schedule, production.conditions())
                record_of_kept_windows(leg_dir, schedule, settings, log)   # the settings the windows were chosen with
                log(f"lambda schedule ({schedule.mode}): the {len(schedule.states)} windows this leg has, without "
                    "the start configurations of them; they are made again for these windows (the settings of "
                    "lambda_schedule are not applied to them; cufep rbfe run --reschedule determines the windows "
                    "again)")
                cfgs = from_pre_equilibration(schedule)
                # samples taken up as they were made keep their start (unknown without a record of these windows),
                # and how the file was made now is said apart; samples started anew start from these (reviews of
                # PR #18)
                how = kept_start_record(leg_dir, schedule, settings, {**starting, "windows": "those this leg had"},
                                        log)
            elif goes_on_from_its_samples(leg_dir):
                refuse_samples_without_windows(leg_dir)
            elif ls.mode == "fixed":
                values = ls.windows
                schedule = schedule_from_windows(STAGES, {"morph": values},
                                                 {"morph": values if isinstance(values, int) else len(values)},
                                                 rest2_scale)
                cfgs = from_pre_equilibration(schedule)
                how = {**starting, "windows": "given"}
            elif a.legs == "each" or job.leg == "solvent":
                als = AlsSettings(overlap_threshold=a.overlap_threshold, overlap_matrix=a.overlap_matrix,
                                  candidates=a.candidates, initial_windows={"morph": a.initial_windows},
                                  window_ps=a.window(job.leg), equilibration_ps=a.equilibration(job.leg),
                                  sample_interval_ps=a.sample_interval_ps, start_configurations=a.start_configurations,
                                  walk_ps=a.walk_ps, max_windows=a.max_windows, rest2_scale=float(rest2_scale))
                pool_spec = spec_of(ends, backend, seed=seed + 23)
                # the Context of this process, which walks through the windows of the first iteration
                ctx, integ = leg_context(pool_spec, hybrid.system, start[1], start[0], start[2], run.device0, seed)
                schedule, cfgs = adaptive_windows(run, pool_spec, ctx, integ, STAGES, als, n_atoms)
                del ctx, integ
                how = {"start_configurations": als.start_configurations,
                       **({"walk_ps": float(als.walk_ps)} if als.start_configurations == "walk" else {}),
                       "windows": "chosen by this precalculation"}
            else:
                schedule = _schedule_from_solvent(rcfg, edge_dir, job.repeat, log, job.control)
                cfgs = from_pre_equilibration(schedule)
                how = {**starting, "windows": "those of the solvent leg"}
            record = {**how, "settings": settings, "states": [s.to_dict() for s in schedule.states]}
            configs = publish_windows(leg_dir, schedule, cfgs, record, leg=job.leg, edge=job.edge,
                                      repeat=job.repeat, seed=seed, backend=backend)
            run.timed("lambda_schedule", t0)
            log(f"lambda schedule ({schedule.mode}): {len(schedule.states)} windows in {time.perf_counter() - t0:.0f} s"
                f" ({how['windows']})")
        check_scale(schedule)
        # ---- 4. replica exchange ----------------------------------------------------------------
        spec = spec_of([s.to_dict() for s in schedule.states], backend, seed=seed + 17, frame_atoms=frame_atoms,
                       frame_ligand=frame_ligand, frame_ca=frame_ca)
        return sample_leg(spec, leg_dir / "hrex", job, configs, n_equil, n_equil + n_prod, rcfg, log, run.timing,
                          run.timed, ckpt_stride), schedule

    (samples, schedule), backend_name = with_fallback(run, backend_name, run_with)

    # ---- 5. analysis ---------------------------------------------------------------------------------
    t_analysis = time.perf_counter()
    from cufep.workflow import record_digest

    run_record = {"edge": job.edge, "leg": job.leg, "repeat": job.repeat, "backend": backend_name,
                  "timestep_fs": timestep_fs, "notes": list(run.notes), "origin_not_recorded": [],
                  # the prepared systems of the edge, which the legs of one ddG must share (edge_result)
                  "prepared_systems_sha256": record_digest(built.info.get("made_from")),
                  "rest2": {"scale": float(rest2_scale), "hot_protein_atoms": len(built.hot_atoms),
                            "hot_residues": built.hot_residues, "torsion_terms_scaled": hybrid.rest2_torsions,
                            "scale_of_window": [float(st.rest2_scale) for st in schedule.states]}}
    result = hybrid_leg_result(leg_dir, rcfg, run_record, built, samples=samples, schedule=schedule)
    run.timed("analysis", t_analysis)
    finish_leg(run, result, backend_name, n_atoms, {"edge": job.edge})
    a = result["analysis"]
    log(f"dG(A -> B, {job.leg}) = {result['delta_g'] / KCAL:.2f} ± {result['delta_g_error'] / KCAL:.2f} kcal/mol; "
        f"min overlap {a['min_overlap']:.3f}, fewest independent samples in a window "
        f"{a['effective_samples_per_state']:.0f}, round trips {a['round_trips']}")
    return result


def hybrid_leg_result(leg_dir: Path, analysis, run_record: dict, built: HybridLeg | None = None, samples=None,
                      schedule=None) -> dict:
    """The result of a leg of an edge: the analysis of its samples (:func:`cufep.engine.leg.path_result`), whose
    free energy along the path (``delta_g``) is that of transforming A into B in the leg, and what the hybrid
    topology is made of."""
    from cufep.engine.leg import path_result
    from cufep.workflow import gcmc_numbers, water_moves_of

    result, of = path_result(leg_dir, analysis, run_record, {"edge": run_record.get("edge")}, samples=samples,
                             schedule=schedule, command=RBFE_COMMAND)
    # the grand canonical moves of water of the production that is analysed, by lambda window (None: none)
    result["gcmc"] = gcmc_numbers(of["samples"], of["first"])
    # and what the samples were made with: whether the leg made the moves, and with which chemical potential and
    # standard volume of water. The complex legs of one network must agree in it (cufep.rbfe.run)
    result["water_moves"] = water_moves_of(of["attrs"].get("conditions"))
    if built is not None:
        h = built.hybrid
        result["hybrid"] = {"ligand_a": built.info["ligand_a"], "ligand_b": built.info["ligand_b"],
                            "core_atoms": len(h.core), "unique_old_atoms": len(h.unique_old),
                            "unique_new_atoms": len(h.unique_new), "lomap_score": built.info.get("lomap_score")}
    return result
