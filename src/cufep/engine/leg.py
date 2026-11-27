"""The steps of a leg that do not depend on its protocol, for the ABFE (:func:`cufep.workflow.run_leg`) and the
relative calculation (:func:`cufep.rbfe.leg.run_hybrid_leg`).

A leg is a System whose lambda states go from one end state to the other (:mod:`cufep.engine.protocol`). Its run:
the pre-equilibration of the System at the first end (:func:`pre_equilibrated`), the backend
(:func:`chosen_backend`), the lambda windows with the start configurations of their replicas (stored, chosen by a
precalculation of adaptive lambda scheduling, :func:`adaptive_windows`, or given and started from a configuration,
:func:`starts_from`), the replica exchange (:func:`cufep.workflow.sample_leg`), the repetition of the whole leg on
stock OpenMM when the cuPotential plugin fails (:func:`with_fallback`), and the analysis of its samples
(:func:`path_result`): the free energy along the path of the leg (``delta_g``), its standard error, the checks of
the analysis and the windows. What a protocol adds (the restraint of the ABFE, the hybrid topology of a relative
leg) and how it names its result are its own.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: how the result of a leg tells the user to run it again: the command and its option for the unit of the leg
ABFE_COMMAND = "cufep run CONFIG --ligand NAME"


@dataclass
class LegRun:
    """What the run of one leg carries through its steps."""

    job: object                     # the job of the leg (cufep.engine.jobs)
    cfg: object                     # its configuration (cufep.config.schema.CommonConfig)
    leg_dir: Path
    log: object
    seed: int
    device0: int | None
    notes: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=lambda: {"phases": {}, "started": time.time()})

    def timed(self, name: str, t_start: float) -> None:
        self.timing["phases"][name] = self.timing["phases"].get(name, 0.0) + (time.perf_counter() - t_start)


def open_leg(job, cfg, set_aside_pre_equilibration: str | None = None) -> LegRun:
    """The directory and the log of a leg, with what it had set aside when its windows are determined again
    (``--reschedule``). ``set_aside_pre_equilibration``: why the pre-equilibration of the leg cannot be used again
    (:func:`equil_ghosts_problem`); with ``--reschedule`` it is set aside with the windows and samples, and the leg
    starts anew from its prepared system (third review of PR #19)."""
    from cufep.workflow import _Log, archive_leg, leg_seed

    leg_dir = job.leg_dir
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(leg_dir / "log.txt", prefix=f"[{job.tag}] ")
    if job.reschedule:
        previous = archive_leg(leg_dir, pre_equilibration=set_aside_pre_equilibration is not None)
        if previous is not None:
            log(f"the lambda windows are determined again; the windows and samples this leg had are in "
                f"{previous.name}/"
                + ("" if set_aside_pre_equilibration is None else
                   f", with its pre-equilibration, which cannot be used again ({set_aside_pre_equilibration}): "
                   "it is made anew"))
    return LegRun(job=job, cfg=cfg, leg_dir=leg_dir, log=log, seed=leg_seed(cfg, job.repeat, job.leg),
                  device0=job.devices[0] if job.devices else None)


#: beside ``equil.npz`` of a leg with grand canonical moves of water: the ghost waters its pre-equilibration was made
#: with (rows of the waters of the moves)
EQUIL_GHOSTS = "equil_ghosts.json"


def equil_ghosts_problem(equil_file: Path, gcmc, earlier_ghosts=None) -> str | None:
    """Why the pre-equilibration of a leg with grand canonical moves of water cannot start the replica exchange of
    this run: its ghost waters are not those of this run (second review of PR #19: a change of the radius of the
    sphere changed the number of ghosts of ``auto`` from 139 to 160, and the leg was fixed in the cell equilibrated
    with 139). Its record (:data:`EQUIL_GHOSTS`), else ``earlier_ghosts``, the ghosts that the versions which wrote no
    record chose for it (None: not known). None for a leg without the moves or without a pre-equilibration, and for
    one whose ghosts are those of this run. It matters only for a leg that starts its replica exchange from the
    pre-equilibration (:func:`starts_anew`): samples record the System they were made with, ghosts and all."""
    equil_file = Path(equil_file)
    if gcmc is None or not equil_file.exists():
        return None
    record = equil_file.parent / EQUIL_GHOSTS
    current = sorted(int(i) for i in gcmc.ghosts)
    if record.exists():
        stored = recorded_ghosts(record, len(gcmc.waters))
        if stored is None:
            # what it was made with is not known: a pre-equilibration that cannot be used, set aside by --reschedule
            # (eleventh review of PR #19: an empty record stopped the leg with an error of JSON, also with --reschedule)
            return f"its record of ghost waters ({record}) cannot be read"
        if stored == current:
            return None
        return (f"it was made with {len(stored)} ghost waters and this run has {len(current)}"
                + ("" if len(stored) != len(current) else " others"))
    if earlier_ghosts is not None and sorted(int(i) for i in earlier_ghosts) == current:
        return None
    return NO_GHOST_RECORD


def ghost_record(record: Path, n_waters: int | None = None) -> dict | None:
    """A record of the ghost waters of a pre-equilibration (:data:`EQUIL_GHOSTS`) as it is written: its ghosts, a set
    of numbers of waters (integers, not negative, each once, below ``n_waters`` where it is given), in ascending order,
    and ``start_centre`` (a name) where it gives one. None where it is not that (empty, not JSON, without its list of
    ghosts, a number that is not an integer: twelfth review of PR #19, 1.9 was taken as 1 and 1e999 stopped the leg;
    a number twice or of no water: thirteenth review, [1, 1] passed for [1], and [426] of 426 waters stopped the
    leg)."""
    try:
        data = json.loads(Path(record).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("ghosts"), list):
        return None
    ghosts = data["ghosts"]
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ghosts):
        return None
    if len(set(ghosts)) != len(ghosts) or (n_waters is not None and any(i >= int(n_waters) for i in ghosts)):
        return None
    if "start_centre" in data and not isinstance(data["start_centre"], str):
        return None
    return {"ghosts": sorted(ghosts), **({"start_centre": data["start_centre"]} if "start_centre" in data else {})}


def recorded_ghosts(record: Path, n_waters: int | None = None) -> list[int] | None:
    """The ghost waters of a record of a pre-equilibration (:data:`EQUIL_GHOSTS`), in ascending order; None where it
    cannot be read (:func:`ghost_record`)."""
    data = ghost_record(record, n_waters)
    return None if data is None else data["ghosts"]


#: why a pre-equilibration without a record of its ghost waters cannot start a replica exchange that needs it
NO_GHOST_RECORD = ("it records no ghost waters (an earlier version wrote it) and they are not known to be those of "
                   "this run")


def starts_anew(leg_dir: Path, reschedule: bool) -> bool:
    """Whether the replica exchange of a leg starts from its pre-equilibration in this run: with ``--reschedule``
    (its samples are set aside), or without samples or the checkpoint of their start
    (:func:`goes_on_from_its_samples`). A leg that goes on from its samples, or is analysed again from complete
    ones, does not make configurations from its pre-equilibration (third review of PR #19: a finished leg was refused
    on a resume and lost its result)."""
    return bool(reschedule) or not goes_on_from_its_samples(leg_dir)


def _cell_differs(boxes, cell) -> bool:
    return not np.allclose(np.asarray(boxes, dtype=float), np.asarray(cell, dtype=float)[None], rtol=0.0, atol=1e-9)


def checkpoint_cell_problem(leg_dir: Path, cell) -> str | None:
    """Why the checkpoint of the replica exchange of a leg with grand canonical moves of water cannot be continued:
    its replicas are in another cell than ``cell``, the one the pre-equilibration fixes (third and fifth reviews of
    PR #19: a checkpoint of the cell of an earlier pre-equilibration was restored, also one without samples, whose
    conditions are taken as new). None without a checkpoint."""
    ckpt = Path(leg_dir) / "hrex" / "checkpoint.npz"
    if not ckpt.exists():
        return None
    with np.load(ckpt, allow_pickle=False) as data:
        if not _cell_differs(data["boxes"], cell):
            return None
    return (f"the replicas of the checkpoint {ckpt} are in another cell than the pre-equilibration of the leg fixes: "
            "they were made from another pre-equilibration")


def start_cell_problem(leg_dir: Path, cell) -> str | None:
    """Why the start configurations of the windows of a leg with grand canonical moves of water cannot be used:
    they are the ones the leg uses (its windows have them: a schedule, and start configurations of its windows, as
    :func:`cufep.workflow.stored_windows` takes them; others are made again) and are in another cell than ``cell``,
    the one the pre-equilibration fixes (third review of PR #19: a pre-equilibration made again left the run in the
    cell of the one before; fifth review: start configurations that would be made again were refused)."""
    from cufep.alchemy.schedule import Schedule
    from cufep.workflow import start_record_of, starts_are_of

    sched_file, start_file = Path(leg_dir) / "schedule.yaml", Path(leg_dir) / "start_configs.npz"
    if not (sched_file.exists() and start_file.exists()):
        return None
    schedule = Schedule.load(sched_file)
    with np.load(start_file) as start:
        boxes = np.asarray(start["boxes"], dtype=float)
    if not starts_are_of(start_record_of(start_file), schedule) or len(boxes) != len(schedule.states):
        return None
    if not _cell_differs(boxes, cell):
        return None
    return (f"the start configurations of its windows ({start_file}) are in another cell than its pre-equilibration "
            f"fixes ({Path(leg_dir) / 'equil.npz'}): they were made from another pre-equilibration")


def fixed_cell_problems(leg_dir: Path) -> list[str]:
    """What of a leg with grand canonical moves of water is not in the cell its pre-equilibration fixes: the start
    configurations it uses and the checkpoint it goes on from (:func:`start_cell_problem`,
    :func:`checkpoint_cell_problem`), for the check of a run before it writes anything. Nothing without a
    pre-equilibration."""
    from cufep.workflow import alchemical_box

    equil_file = Path(leg_dir) / "equil.npz"
    if not equil_file.exists():
        return []
    with np.load(equil_file) as eq:
        cell = alchemical_box(eq, True)
    return [why for why in (start_cell_problem(leg_dir, cell), checkpoint_cell_problem(leg_dir, cell)) if why]


def unfinished_archive_problem(leg_dir: Path, leg: str, repeat: int, reschedule: bool) -> str | None:
    """A leg whose archive (``--reschedule``) was begun and not completed is not run as it is: part of what it had is
    set aside, the rest is still in place (review of 2861b93, for the ABFE; fifth review of PR #19, for the relative
    legs). None with ``--reschedule``, which completes it, or without such an archive."""
    unfinished = sorted(Path(leg_dir).glob(".previous_*.partial"))
    if not unfinished or reschedule:
        return None
    return (f"the {leg} leg of repeat {repeat} was being started anew (--reschedule) when a run stopped: "
            f"{unfinished[0]} holds a part of what the leg had. Nothing was changed. Run this leg with --reschedule "
            "again to complete that")


def unusable_pre_equilibration(equil_file: Path, why: str, reschedule_sets_it_aside: bool) -> str:
    """The refusal of a pre-equilibration that cannot start the replica exchange of a leg, with the way on: a relative
    leg is computed anew with ``--reschedule``, which sets the pre-equilibration aside (:func:`open_leg`); an absolute
    leg has made its restraint and the MD for it from it, and is computed in a new output directory."""
    return (f"the pre-equilibration of this leg in {equil_file} cannot be used again: {why}. Nothing was changed; "
            + ("run the leg with --reschedule, which sets it aside with the windows, samples and result of the leg in "
               "previous_<n>/ and computes the leg anew, or run the calculation in a new output.dir"
               if reschedule_sets_it_aside else
               "the restraint of the leg and the MD for it were made from it: run the calculation in a new output.dir"))


def check_equil_ghosts(equil_file: Path, gcmc, earlier_ghosts=None, reschedule_sets_it_aside: bool = True) -> None:
    """Refuse a pre-equilibration that cannot start the replica exchange of a leg (:func:`equil_ghosts_problem`):
    the check in the leg, after the run has made it for all its legs before it wrote anything."""
    from cufep.workflow import WorkflowError

    why = equil_ghosts_problem(equil_file, gcmc, earlier_ghosts)
    if why is not None:
        raise WorkflowError(unusable_pre_equilibration(equil_file, why, reschedule_sets_it_aside))


def pre_equilibrated(run: LegRun, prep: dict, system, barostat: str, membrane_info: dict, gcmc=None):
    """The pre-equilibration of the leg (``equil.npz``), made if the leg has none, checked against the settings of
    this run: ``(its arrays, whether its file records what it was made with)``. With grand canonical moves of water
    (``gcmc``) the ghost waters it is made with are recorded first (:data:`EQUIL_GHOSTS`); whether one that is there
    can start the replica exchange the leg checks before anything is set aside (:func:`check_equil_ghosts`)."""
    from cufep.files import publish_text
    from cufep.workflow import check_equil_file, run_pre_equilibration

    equil_file = run.leg_dir / "equil.npz"
    leg = run.job.leg
    if not equil_file.exists():
        if gcmc is not None:
            publish_text(run.leg_dir / EQUIL_GHOSTS, json.dumps(
                {"ghosts": [int(i) for i in gcmc.ghosts], "waters": int(len(gcmc.waters)),
                 **({} if gcmc.start_centre == "ligand" else {"start_centre": gcmc.start_centre})}))
        t0 = time.perf_counter()
        run_pre_equilibration(run.cfg, leg, prep, system, barostat, membrane_info, run.seed, equil_file, run.device0,
                              run.log)
        run.timed("pre_equilibration", t0)
    eq = np.load(equil_file)
    in_membrane = leg == "complex" and barostat == "membrane"
    return eq, check_equil_file(eq, equil_file, run.cfg, leg, in_membrane, run.seed, run.log)


def chosen_backend(run: LegRun, system) -> str:
    """The backend the leg runs on: the one of the configuration (or stock OpenMM where the plugin cannot run the
    System), and the backend of its samples if it has some (samples of two backends are never mixed)."""
    from cufep.sampling.storage import stored_backend
    from cufep.workflow import _choose_backend

    backend_name, notes = _choose_backend(run.cfg, system, run.log)
    run.notes += notes
    made_with = stored_backend(run.leg_dir / "hrex")
    if made_with is not None and made_with != backend_name:
        run.log(f"this leg goes on with {made_with}, the backend its samples were made with "
                f"(this run would have chosen {backend_name})")
        run.notes.append(f"the leg was continued on {made_with}, the backend its samples were made with")
        backend_name = made_with
    return backend_name


def leg_context(spec, system, box, positions, velocities, device, seed: int):
    """A Context of this process for the windows of a leg: the System prepared by the backend of ``spec``, its
    barostat (none for ``spec.barostat`` none), the integrator, and the configuration (``positions`` may be a
    function of the Context, for positions that depend on its molecules). ``seed`` is the seed of the leg."""
    import openmm as mm
    import openmm.unit as u

    from cufep.sampling.worker import backend_of

    backend = backend_of(spec)
    prepared = backend.prepare_system(system)
    T = spec.temperature_K * u.kelvin
    if spec.barostat == "membrane":
        prepared.addForce(backend.make_membrane_barostat(spec.pressure_bar * u.bar, T, spec.barostat_frequency,
                                                         seed=seed + 5))
    elif spec.barostat != "none":
        prepared.addForce(backend.make_isotropic_barostat(spec.pressure_bar * u.bar, T, spec.barostat_frequency,
                                                          seed=seed + 5))
    integ = backend.make_integrator(T, spec.friction_per_ps / u.picosecond, spec.timestep_fs * u.femtoseconds,
                                    seed=seed + 6)
    ctx = backend.make_context(prepared, integ, device)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(positions(ctx) if callable(positions) else positions)
    ctx.setVelocities(velocities)
    return ctx, integ


#: OpenMM's minimisation of a configuration moved into the fixed cell of the grand canonical moves, before any MD in it
#: (steps; OpenFE's 5000, as the windows of a relative leg minimise their starts)
CELL_MINIMIZE_STEPS = 5000


def minimised_in_cell(spec, system, cell, positions, velocities, device, seed: int) -> np.ndarray:
    """The end of a pre-equilibration moved into the fixed cell of the grand canonical moves
    (:func:`cufep.workflow.alchemical_box`, :func:`cufep.workflow.positions_in_cell`; ``positions`` may be a function
    of the Context, as for :func:`leg_context`), minimised there before any MD in it: OpenMM's minimiser, on stock
    OpenMM, in the first state of ``spec`` (its backend, barostat and seed are replaced), ``CELL_MINIMIZE_STEPS``
    steps. Every molecule moved whole keeps its geometry but not its contacts: the cell of the moves is the mean cell
    of the pre-equilibration, a few per cent from its last one (A2A 4g~4kU: z -2.0 %, a largest force 5.7e4
    kJ/mol/nm against 5.9e3, energy +14600 kJ/mol), and MD on cuPotential from such a start stopped in its first 5 ps
    (a kernel's contract check, ``cuda_matrix_migrate``; ``spikes/out/m48``), which under MPS stopped every worker of
    the run with it. Positions in nm."""
    import dataclasses

    import openmm as mm
    import openmm.unit as u

    from cufep.engine.protocol import state_from_dict
    from cufep.sampling.states import set_state

    stock = dataclasses.replace(spec, backend="openmm", barostat="none")
    ctx, integ = leg_context(stock, system, cell, positions, velocities, device, seed)
    set_state(ctx, state_from_dict(spec.states[0], spec.protocol))
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, CELL_MINIMIZE_STEPS)
    out = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    del ctx, integ
    return np.asarray(out)


def granted_devices(job, n_windows: int, work_per_window: float, what: str) -> list:
    """The GPUs for ``n_windows`` windows that run at the same time: those the scheduler grants (pinned: the job
    cannot move these workers), else those of the job."""
    devices = list(job.devices)
    if job.control is not None:
        granted = job.control.request_devices(n_windows, n_windows * work_per_window / max(1, job.oversubscription),
                                              pinned=True, what=what)
        devices = granted or devices
    return devices


def adaptive_windows(run: LegRun, pool_spec, ctx, integ, stages, als, n_atoms: int, inline_mover=None):
    """The lambda windows of a precalculation of adaptive lambda scheduling and the start configurations of their
    replicas, from the configuration of ``ctx``. The windows of an iteration run at the same time on the GPUs the
    scheduler grants; without workers one after another on ``ctx`` (the same windows from the same start
    configurations either way, :func:`cufep.alchemy.schedule.adaptive_schedule`). ``inline_mover`` makes the
    moves of water of the windows on ``ctx`` in that case: a function that returns ``(mover, interval in steps)``."""
    from cufep.alchemy.candidates import initial_window_indices
    from cufep.alchemy.schedule import adaptive_schedule
    from cufep.sampling.hrex import WindowPool
    from cufep.workflow import md_work

    cfg, job = run.cfg, run.job
    T, timestep_fs = pool_spec.temperature_K, pool_spec.timestep_fs
    n_first = sum(len(initial_window_indices(als.initial_windows[st], als.candidates)) for st in stages) \
        - (len(stages) - 1)
    work = md_work(als.window_ps + als.equilibration_ps, n_atoms, cfg.sampling.time_per_iteration_ps)
    devices = granted_devices(job, n_first, work,
                              f"the precalculation of the lambda schedule ({n_first} lambda windows)")
    if job.inline or len(devices) < 2:
        mover, interval_steps = inline_mover() if inline_mover is not None else (None, 0)
        return adaptive_schedule(ctx, integ, stages, T, timestep_fs, als, run.log, mover=mover,
                                 interval_steps=interval_steps)
    with WindowPool(pool_spec, devices[:n_first], run.log) as pool:
        return adaptive_schedule(ctx, integ, stages, T, timestep_fs, als, run.log, pool=pool)


def starts_from(run: LegRun, start_spec, schedule, start: tuple, system, n_atoms: int, minimize_steps: int,
                md_ps: float) -> list:
    """The start configurations of windows that a leg does not choose itself, from one configuration ``start``
    (positions, box, velocities): each minimised in the state of its window (``minimize_steps``) and ``md_ps`` of
    MD in it, all windows at the same time on the GPUs the leg is granted (in this process one after another
    where it has no workers), on the backend of ``start_spec``."""
    from cufep.alchemy.schedule import run_window
    from cufep.sampling.hrex import WindowPool
    from cufep.steps import md_steps
    from cufep.workflow import md_work

    job = run.job
    tasks = [{"state": state.to_dict(), "protocol": state.protocol, "config": start, "stages": [], "candidates": 2,
              "temperature_K": start_spec.temperature_K, "equilibration_steps": md_steps(md_ps, start_spec.timestep_fs),
              "n_samples": 0, "steps_per_sample": 1, "minimize_steps": minimize_steps} for state in schedule.states]
    devices = granted_devices(job, len(tasks), md_work(md_ps, n_atoms, run.cfg.sampling.time_per_iteration_ps),
                              f"the start of {len(tasks)} lambda windows")
    if not job.inline and len(devices) >= 2:
        with WindowPool(start_spec, devices[:len(tasks)], run.log) as pool:
            done = pool.run(tasks)
    else:
        ctx, integ = leg_context(start_spec, system, start[1], start[0], start[2], run.device0, run.seed)
        # the moves of water a worker of the pool makes in these windows (cufep.sampling.worker, window_mover): the
        # same here, so that the start does not depend on the GPUs a leg is granted (review of PR #19)
        mover, interval_steps = window_moves(start_spec, ctx)
        done = [run_window(ctx, integ, task, mover, interval_steps) for task in tasks]
        del ctx, integ
    return [d["config"] for d in done]


def window_moves(spec, context) -> tuple:
    """(mover, MD steps between its moves) of the moves of water of ``spec`` on ``context`` for a window run in this
    process, as a worker of a pool makes them (its first worker's stream of random numbers); (None, 0) without."""
    from cufep.sampling.worker import GCMC_STREAM, RELOCATION_STREAM, stream_seed
    from cufep.steps import md_steps

    if spec.gcmc is not None:
        from cufep.sampling.gcmc import GcmcMover

        return (GcmcMover(spec.gcmc, context, stream_seed(spec.seed, 0, 0, GCMC_STREAM)),
                md_steps(spec.gcmc.interval_ps, spec.timestep_fs))
    if spec.relocation is not None:
        from cufep.sampling.relocation_mc import RelocationMover

        return (RelocationMover(spec.relocation, context, stream_seed(spec.seed, 0, 0, RELOCATION_STREAM)),
                md_steps(spec.relocation.interval_ps, spec.timestep_fs))
    return None, 0


def with_fallback(run: LegRun, backend_name: str, run_with):
    """``run_with(backend_name)``; if the cuPotential plugin fails in a leg that may be repeated, what the failed run
    left is set aside and the whole leg is run again on stock OpenMM (samples of two backends are never mixed).
    Returns ``(what run_with returned, the backend that made it)``."""
    from cufep.sampling.storage import stored_conditions
    from cufep.workflow import (
        WorkflowError,
        backend_fallback_refusal,
        leg_with_samples_stays_on_its_backend,
        set_aside_failed_backend,
    )

    # a leg that has samples from an earlier run is continued, never repeated from its start
    had_samples = stored_conditions(run.leg_dir / "hrex")[0] > 0
    try:
        return run_with(backend_name), backend_name
    except Exception as exc:  # noqa: BLE001  classified here; everything that is not repeated is raised
        refusal = backend_fallback_refusal(exc, backend_name, run.cfg.backend.fallback, had_samples)
        if refusal is not None:
            if refusal == "the leg had samples from an earlier run":
                raise WorkflowError(leg_with_samples_stays_on_its_backend(exc)) from exc
            raise
        kept = set_aside_failed_backend(run.leg_dir)
        run.log(f"cuPotential run failed ({str(exc)[:300]}); repeating this leg on stock OpenMM; what the failed run "
                f"left is kept in {kept.name}/")
        run.notes.append("the cuPotential run failed; the whole leg was repeated on stock OpenMM "
                         f"(samples from the two backends are never mixed; the failed run is kept in {kept.name}/)")
        return run_with("openmm"), "openmm"


def goes_on_from_its_samples(leg_dir: Path) -> bool:
    """Whether the replica exchange of a leg goes on from what its directory holds (its samples, or the checkpoint
    of its start, which is written before the first sample, :meth:`ReplicaExchangeSampler.initialize`) and not
    from the start configurations of its windows (third review of PR #18)."""
    from cufep.sampling.storage import stored_conditions

    return stored_conditions(Path(leg_dir) / "hrex")[0] > 0 or (Path(leg_dir) / "hrex" / "checkpoint.npz").exists()


#: the keys of the record of start configurations that say how they were made
START_KEYS = ("start_configurations", "walk_ps", "start_md_ps", "minimize_steps", "start_backend")


def kept_start_record(leg_dir: Path, schedule, settings: dict, made_now: dict, log=print) -> dict:
    """The record of start configurations made again for windows a leg keeps (``made_now``: how they are made now,
    with ``windows``). A leg that has samples (or the checkpoint of their start) keeps their start, the one the
    earlier record gives for these windows (:func:`record_of_kept_windows`; none if it names other windows or there
    is none: the start is not known), with what the earlier record says it does not record, and says how the file
    was made now apart (``start_configurations_made_again``); the record becomes that only when the replica exchange
    initializes from these configurations (:func:`record_the_start_used`, called by
    :func:`cufep.workflow.sample_leg`; it does not when it restores the checkpoint, analyses a complete leg, or
    refuses the leg). A leg without samples starts from them: the record is ``made_now``. One rule for every way a
    leg keeps its windows (reviews of PR #18)."""
    if not goes_on_from_its_samples(leg_dir):
        return dict(made_now)
    earlier = record_of_kept_windows(leg_dir, schedule, settings, log)
    again = {key: made_now[key] for key in START_KEYS if key in made_now}
    rest = {key: value for key, value in made_now.items() if key not in START_KEYS}
    if earlier:
        # the earlier record of these windows says what is not recorded of them; the mark of made_now that their
        # origin is not recorded is of start configurations without a record (sixth review of PR #18)
        rest.pop("origin_not_recorded", None)
    return {**rest, **earlier, "start_configurations_made_again": again}


def record_the_start_used(leg_dir: Path) -> None:
    """Called when the replica exchange of a leg initializes from its start configurations (not from the checkpoint
    of its samples): if their record keeps the start of earlier samples with the start made again apart
    (:func:`kept_start_record`), it becomes the start made again, the one these samples start from (fifth and sixth
    reviews of PR #18: switched before the initialization, the start of the earlier samples was lost when the
    replica exchange refused the leg)."""
    import json as _json

    from cufep.files import publish_npz
    from cufep.workflow import start_record_of

    start_file = Path(leg_dir) / "start_configs.npz"
    if not start_file.exists():
        return
    made = start_record_of(start_file)
    again = made.get("start_configurations_made_again")
    if not isinstance(again, dict):
        return
    record = {**{key: value for key, value in made.items()
                 if key not in START_KEYS and key != "start_configurations_made_again"}, **again}
    # the start of these samples is known now: what is not recorded is at most the origin of the windows (seventh
    # review of PR #18: the mark of an earlier version said the start configurations too)
    from cufep.workflow import WINDOWS_ORIGINS

    if record.get("origin_not_recorded") == WINDOWS_ORIGINS[0]:
        record["origin_not_recorded"] = WINDOWS_ORIGINS[1]
    with np.load(start_file) as start:
        arrays = {key: start[key] for key in start.files if key != "made_with"}
    publish_npz(start_file, made_with=_json.dumps(record), **arrays)


def check_samples_have_their_windows(leg_dir: Path, states) -> None:
    """Before a run moves or writes anything of a leg (before ``begin_leg``): if its replica exchange goes on from
    samples (or the checkpoint of their start), the leg must have the windows of those samples: ``states`` are the
    states of the schedule it will use (its own, or the one it is given), None if it has none. Samples that do not
    record their conditions are not compared. Raises WorkflowError (fifth review of PR #18: a missing schedule was
    refused after the result of the leg had been set aside, and the leg counted as failed)."""
    from cufep.sampling.storage import _same, stored_layout
    from cufep.workflow import WorkflowError

    if not goes_on_from_its_samples(leg_dir):
        return
    if states is None:
        refuse_samples_without_windows(leg_dir)
    stored = stored_layout(Path(leg_dir) / "hrex")[1].get("conditions")
    # samples that do not record their states are not compared (their conditions are checked where they are used)
    if isinstance(stored, dict) and "states" in stored and not _same(stored["states"], [dict(s) for s in states]):
        raise WorkflowError(f"the lambda windows {Path(leg_dir)} is to use are not those of its samples in "
                            f"{Path(leg_dir) / 'hrex'}; nothing was changed (--reschedule starts the leg anew and "
                            "keeps what it had)")


def check_kept_windows(leg_dir: Path, schedule, conditions: dict, refuse_unrecorded: bool = True) -> None:
    """Before anything of a leg is written for windows it keeps (start configurations made again, a schedule
    taken): if its replica exchange goes on from its samples (:func:`goes_on_from_its_samples`), the windows must
    be those of its samples and every condition they were made under that of the replica exchange of this run
    (``conditions``: ``LegSpec.conditions()``), as the replica exchange compares them when it opens the samples,
    after the files would have been written (reviews of PR #18). Raises WorkflowError or ConditionsError; returns
    for a leg that does not go on from samples. Samples that do not record their conditions (an earlier version of
    the ABFE) are refused with ``refuse_unrecorded``, else left to the checks of their own."""
    from cufep.sampling.storage import _same, conditions_differences, other_conditions, stored_layout
    from cufep.workflow import WorkflowError

    if not goes_on_from_its_samples(leg_dir):
        return
    hrex = Path(leg_dir) / "hrex"
    parts, attrs = stored_layout(hrex)
    stored = attrs.get("conditions")
    if not isinstance(stored, dict):
        if refuse_unrecorded:
            raise WorkflowError(f"the samples in {hrex} do not record their conditions; nothing was changed")
        return
    if not _same(stored.get("states"), [state.to_dict() for state in schedule.states]):
        raise WorkflowError(f"the lambda windows of {Path(leg_dir) / 'schedule.yaml'} are not those of its samples in "
                            f"{hrex}; nothing was changed (--reschedule starts the leg anew and keeps what it had)")
    differing = conditions_differences(stored, conditions)
    if differing:
        raise other_conditions(sum(rows for _path, rows, _tail, _frames in parts), hrex, differing)


def record_of_kept_windows(leg_dir: Path, schedule, settings: dict, log=print) -> dict:
    """What the record of the start configurations of a leg says of how the start configurations of the windows
    it keeps were made, if it names these windows (else empty), after checking that they were made with the
    settings of this run (:func:`cufep.workflow.check_windows_settings`; second review of PR #18). A leg whose
    replica exchange goes on from its samples keeps this as the start of its samples, and records the start
    configurations made again apart."""
    from cufep.sampling.storage import _same
    from cufep.workflow import check_windows_settings, start_record_of

    start_file = Path(leg_dir) / "start_configs.npz"
    try:
        made = start_record_of(start_file) if start_file.exists() else {}
    except (OSError, ValueError, KeyError):
        made = {}
    if made.get("states") is None or not _same(made["states"], [state.to_dict() for state in schedule.states]):
        return {}
    check_windows_settings(made, settings, start_file, log)
    return {key: value for key, value in made.items() if key not in ("settings", "states")}


def refuse_samples_without_windows(leg_dir: Path) -> None:
    """A leg whose replica exchange goes on from samples (or the checkpoint of their start) that has no schedule
    of their windows cannot be continued: refused before new windows are written (review of PR #18)."""
    from cufep.workflow import WorkflowError

    if goes_on_from_its_samples(leg_dir):
        raise WorkflowError(f"{leg_dir} has samples (or the checkpoint of their start) but no schedule.yaml of the "
                            "windows they are of: the leg cannot be continued; nothing was changed (--reschedule "
                            "starts it anew and keeps what it had)")


def finish_leg(run: LegRun, result: dict, backend_name: str, n_atoms: int, identity: dict) -> None:
    """Write the timing of the leg (a diagnostic, which decides nothing) and its result."""
    from cufep.files import publish_text
    from cufep.workflow import write_leg_result

    run.timing.update(finished=time.time(), backend=backend_name, atoms=n_atoms, **identity, leg=run.job.leg,
                      repeat=run.job.repeat)
    publish_text(run.leg_dir / "timing.json", json.dumps(run.timing, indent=1))
    write_leg_result(run.leg_dir, result)


def equilibration_files(leg_dir: Path) -> tuple[list[str], dict]:
    """The notes of the pre-equilibration of a leg and what its file says of the bilayer, without verdicts
    (:func:`cufep.workflow.equil_notes`, :func:`cufep.workflow.bilayer_numbers`); a leg without the file gives
    none."""
    from cufep.workflow import bilayer_numbers, equil_notes, leg_files

    equil_file = leg_files(leg_dir)["equil.npz"]
    if equil_file is None:
        return [], {"bilayer": None}
    eq = np.load(equil_file)
    return equil_notes(eq), {"bilayer": bilayer_numbers(eq)}


def path_result(leg_dir: Path, analysis, run: dict, identity: dict, samples: dict | None = None, schedule=None,
                n_equilibration: int | None = None, files=equilibration_files, command: str = ABFE_COMMAND):
    """The analysis of the samples of a leg (needs no GPU): the free energy along the path of the leg from its
    first to its last state (``delta_g``: MBAR minus the tail correction of the same iterations; ``delta_g_error``
    its standard error, :class:`cufep.analysis.mbar.TimeRangeEstimator`), the checks of the analysis, the windows,
    the structure and the notes. Returns ``(result, what a protocol may add from: the attributes of the samples,
    their temperature, the first iteration of the production analysed, the samples)``.

    ``analysis`` gives the settings of :func:`cufep.workflow.analysis_settings` (a whole configuration, or its
    ``analysis`` section alone), and nothing else of a configuration enters: temperature, length of an iteration
    and the end of the equilibration are those of the samples, so that a leg cannot be analysed as if it had been
    run under the settings of another run. ``run`` holds what only the run of the leg knows (its notes among
    them), ``identity`` what the result says the leg is of. ``files(leg_dir)`` gives the notes of the files of the
    leg and what they say as numbers; ``command`` is how the warnings tell the user to run the leg again. One
    function for a leg that has just been sampled, a result made again by an analysis of a run directory, and a
    result made again because the method or the settings of the analysis have changed.
    """
    from cufep.alchemy.schedule import Schedule
    from cufep.analysis.mbar import NoCorrection, analyze_leg
    from cufep.analysis.structure import structural_analysis
    from cufep.sampling.storage import equilibration_boundary, read_samples
    from cufep.workflow import (
        _ANALYSIS_NOTES,
        WorkflowError,
        analysis_problems,
        analysis_record,
        analysis_settings,
        leg_quality,
        pressure_of_samples,
        production_start,
        quality_warnings,
        start_configurations_of,
        windows_origin_of,
    )

    settings = analysis_settings(analysis)
    a = settings.analysis
    if samples is None:
        samples = read_samples(leg_dir / "hrex")
    attrs = samples["attrs"]
    if "temperature_K" not in attrs or "iteration_ps" not in attrs:
        raise WorkflowError(f"the samples in {leg_dir} do not record their temperature and the length of an iteration")
    temperature, iteration_ps = float(attrs["temperature_K"]), float(attrs["iteration_ps"])
    if n_equilibration is None:
        n_equilibration = equilibration_boundary(attrs, leg_dir)
        if n_equilibration is None:
            raise WorkflowError(f"the samples in {leg_dir} are of a run that did not finish and do not record where "
                                "its production begins")
    problems = analysis_problems(attrs, settings)
    if problems:
        raise WorkflowError(f"the samples in {leg_dir} cannot be analysed with these settings: {'; '.join(problems)}")
    if schedule is None and (leg_dir / "schedule.yaml").exists():
        schedule = Schedule.load(leg_dir / "schedule.yaml")
    # the production that is analysed: from the end of the equilibration of the run, or with
    # analysis.detect_equilibration from where the detection puts it; the free energy and the
    # numbers of the structure are of the same iterations
    first, detected = production_start(samples, n_equilibration, a)
    try:
        res = analyze_leg(samples, temperature, n_equilibration=first, n_fractions=a.convergence_points,
                          n_blocks=a.convergence_blocks, error_blocks=a.error_blocks,
                          part_error_blocks=a.part_error_blocks, tail=a.lrc, min_overlap=a.min_overlap)
    except NoCorrection as exc:
        raise WorkflowError(f"the samples in {leg_dir}: {exc}") from exc
    structure = structural_analysis(samples, first)
    conditions = attrs.get("conditions") if isinstance(attrs.get("conditions"), dict) else {}
    # the notes of the files of the leg, and what they say as numbers
    notes, file_numbers = files(leg_dir)
    # what the run noted besides, without the verdicts of earlier versions
    notes += [n for n in (run.get("notes") or []) if n not in notes and not str(n).startswith(_ANALYSIS_NOTES)]
    origin_not_recorded = list(run.get("origin_not_recorded") or [])
    # what the files of the leg say of its windows, whoever asks (review of 8dc042e: `cufep analyze`
    # of a leg that was stopped before its first result did not have the note of its run)
    windows_origin = windows_origin_of(leg_dir)
    if windows_origin and windows_origin not in origin_not_recorded:
        origin_not_recorded.append(windows_origin)
    # the outer radius to which the tail energy of these samples was recorded; the energies cannot
    # be turned into those of another radius
    tail = conditions.get("lrc") if isinstance(conditions.get("lrc"), dict) else {}
    result = {
        **identity, "leg": run.get("leg"), "repeat": run.get("repeat"), "backend": run.get("backend"),
        # the pressure is the one the samples were made under; samples that do not record it have none
        "temperature_K": temperature, "pressure_bar": pressure_of_samples(conditions),
        "prepared_systems_sha256": run.get("prepared_systems_sha256"),
        "timestep_fs": run.get("timestep_fs"), "n_states": res.n_states,
        "production_ns": float(res.n_iterations_used * iteration_ps / 1000.0),
        "n_iterations_stored": int(samples["u"].shape[0]), "n_equilibration_iterations": int(n_equilibration),
        # with analysis.detect_equilibration: the iterations of the production that the detection left
        # out after the equilibration of the run, with the statistical inefficiency and the number of
        # uncorrelated samples of the effective energy from there on (None: the detection is off)
        "equilibration_detected": detected,
        # the free energy along the path of the leg (MBAR minus the tail correction of the same iterations),
        # its standard error (cufep.analysis.mbar.TimeRangeEstimator), and the MBAR part alone
        "delta_g": res.delta_g, "delta_g_error": res.delta_g_error,
        "delta_g_mbar": res.delta_g_mbar, "delta_g_mbar_error": res.delta_g_mbar_error,
        "lrc": res.correction, "lrc_method": a.lrc, "lrc_long_cutoff_nm": tail.get("long_cutoff_nm"),
        "lrc_reweight": res.lrc_reweight,
        "lrc_isotropic": res.lrc_isotropic, "lrc_error": res.correction_error, "structure": structure,
        # numbers of the files of the leg, without a verdict
        **file_numbers,
        "analysis": res.to_dict(), "units": "kJ/mol", "analysis_record": analysis_record(settings), "run": run,
    }
    result["quality"] = leg_quality(result["analysis"], a)
    result["lambda_windows"] = schedule.windows() if schedule is not None else {}
    # what this leg used although an earlier version had written it without saying what it was made from
    result["origin_not_recorded"] = origin_not_recorded
    if origin_not_recorded:
        notes.append("written by an earlier version of cuFEP without a record of what it was made from, and used "
                     f"as it is: {', '.join(origin_not_recorded)}. Nothing in these files contradicts this run; "
                     "that they were made with its settings could not be checked")
    if schedule is not None:
        notes += [n for n in schedule.notes if n not in notes]
    result["notes"] = notes + quality_warnings(result, a, command)
    # windows that adaptive lambda scheduling added to its start, per stage (None: fixed windows)
    start = schedule.provenance.get("initial_windows") if schedule is not None and schedule.mode == "adaptive" else None
    result["lambda_windows_added"] = (
        {name: len(values) - int(start[name]) for name, values in schedule.windows().items() if name in start}
        if start else None)
    # how the windows got the configurations their replicas started from (None: the files do not say)
    result["start_configurations"] = start_configurations_of(leg_dir)
    result["rest2"] = run.get("rest2")
    return result, {"attrs": attrs, "temperature": temperature, "first": first, "samples": samples,
                    "conditions": conditions}
