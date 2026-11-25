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


def open_leg(job, cfg) -> LegRun:
    """The directory and the log of a leg, with what it had set aside when its windows are determined again
    (``--reschedule``)."""
    from cufep.workflow import _Log, archive_leg, leg_seed

    leg_dir = job.leg_dir
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(leg_dir / "log.txt", prefix=f"[{job.tag}] ")
    if job.reschedule:
        previous = archive_leg(leg_dir)
        if previous is not None:
            log(f"the lambda windows are determined again; the windows and samples this leg had are in "
                f"{previous.name}/")
    return LegRun(job=job, cfg=cfg, leg_dir=leg_dir, log=log, seed=leg_seed(cfg, job.repeat, job.leg),
                  device0=job.devices[0] if job.devices else None)


def pre_equilibrated(run: LegRun, prep: dict, system, barostat: str, membrane_info: dict):
    """The pre-equilibration of the leg (``equil.npz``), made if the leg has none, checked against the settings of
    this run: ``(its arrays, whether its file records what it was made with)``."""
    from cufep.workflow import check_equil_file, run_pre_equilibration

    equil_file = run.leg_dir / "equil.npz"
    leg = run.job.leg
    if not equil_file.exists():
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
        done = [run_window(ctx, integ, task) for task in tasks]
        del ctx, integ
    return [d["config"] for d in done]


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
