"""Top-level driver: configuration -> prepared systems -> legs -> binding free energies."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from cufep import jsonio, mps
from cufep.analysis.plots import plot_interactions, plot_repeat
from cufep.benchmarks.runner import LEG_FLAGS, flags_of
from cufep.config.auto import AutoReport, resolve
from cufep.config.schema import Config, check_configuration, dump_config, read_settings
from cufep.files import publish_text, publish_with
from cufep.scheduler import GpuMonitor, run_jobs
from cufep.workflow import (
    KCAL,
    LegJob,
    WorkflowError,
    _Log,
    check_before_jobs,
    check_interaction_inputs,
    check_leg_reuse,
    check_partner_legs,
    check_prepared,
    combine_ligand,
    ligand_directories,
    ligands_of_run,
    prepare_ligand,
    refresh_leg_result,
    select_names,
    settings_differences,
    stored_ligand_result,
)


def plan(cfg: Config, log=print, ligands=None) -> tuple[Config, AutoReport, object, list]:
    """Static part of a run: prepare the protein, read the ligands, resolve ``auto`` settings."""
    from cufep.prep.forcefield import ForceFieldSpec, make_forcefield
    from cufep.prep.protein import prepare_protein

    check_configuration(cfg)
    if ligands is None:
        # the ligands first: what they are called decides where anything is written
        ligands = ligands_of_run(cfg)
        # what restraints.selection: interactions needs, before any MD: a plan is of every complex leg.
        # A run gives its ligands and has checked those whose complex leg it runs (:func:`run`)
        check_interaction_inputs(cfg, ligands)
    names = [lig.name for lig in ligands]
    if cfg.system.cofactors:
        raise WorkflowError("system.cofactors is not implemented yet in this version")
    ff = make_forcefield(ForceFieldSpec(protein=cfg.forcefield.protein, water=cfg.forcefield.water), [])
    protein = prepare_protein(cfg.system.protein, forcefield=ff, ph=cfg.thermodynamics.ph)
    for line in protein.report.lines():
        log(f"protein: {line}")
    resolved, report = resolve(cfg, protein)
    report.warnings += protein.report.warnings
    if protein.report.protonation_assigned:
        report.warnings.append("protonation states were assigned automatically (no hydrogens in the PDB): "
                               + ", ".join(protein.report.protonation_assigned[:40]))
    log(f"{len(ligands)} ligand(s): {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    return resolved, report, protein, ligands


def lock_output_directory(out_dir: Path):
    """Take the lock of an output directory, or stop if another run holds it.

    Two runs writing into one directory overwrite each other's files and fail later with an
    unrelated message about a file that cannot be opened. The lock is released when the
    returned file is closed or the process ends.
    """
    import fcntl

    from cufep.processes import legs_still_running

    handle = open(Path(out_dir) / ".cufep.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise WorkflowError(f"another cufep run is using the output directory {out_dir}; wait for it to end, "
                            "or give this run another directory (output.dir or -o)") from None
    # a job of an earlier run whose scheduler ended without stopping it may still be writing
    running = legs_still_running(out_dir)
    if running:
        handle.close()
        raise WorkflowError(f"a job of an earlier run is still running in {', '.join(running)} of {out_dir}; it "
                            "stops by itself now that its run has ended. Wait for it, then run again")
    return handle


def select_jobs(names: list[str], repeats: int, only_ligands=None, only_legs=None) -> list[tuple[str, int, str]]:
    """(ligand, repeat, leg) of the jobs of a run, restricted to the given ligands and legs."""
    only_ligands, only_legs = list(only_ligands or []), list(only_legs or [])
    select_names(names, only_ligands, "--ligand")
    wrong = [leg for leg in only_legs if leg not in ("complex", "solvent")]
    if wrong:
        raise WorkflowError(f"--leg: {', '.join(wrong)} is neither complex nor solvent")
    return [(name, r, leg) for name in names if not only_ligands or name in only_ligands
            for r in range(repeats) for leg in ("complex", "solvent") if not only_legs or leg in only_legs]


def run(cfg: Config, reschedule: bool = False, log=None, only_ligands=None, only_legs=None) -> list[dict]:
    """Run the whole calculation for every ligand in the SDF. Safe to call again to resume.

    Called again in the same output directory, a leg goes on from where it stopped: an
    unfinished leg resumes, and a finished leg runs on if ``sampling.production_ns`` is now
    longer. With ``reschedule`` a leg starts anew with windows determined again; what it had is
    moved to ``previous_<n>/`` in its directory. ``only_ligands`` and ``only_legs`` restrict
    this to some legs; the binding free energies are then put together again with the results
    the other legs already have.
    """
    check_configuration(cfg)                     # before the output directory is made
    ligands = ligands_of_run(cfg, only_ligands)  # so are the names of the ligands
    # and what restraints.selection: interactions needs, of the ligands whose complex leg this run runs:
    # a ligand that is left out (--ligand), or of which only the solvent leg is run (--leg), needs no restraint
    with_complex_leg = {name for name, _repeat, leg in select_jobs([lig.name for lig in ligands], 1, only_ligands,
                                                                    only_legs) if leg == "complex"}
    check_interaction_inputs(cfg, [lig for lig in ligands if lig.name in with_complex_leg])
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_output_directory(out_dir)
    try:
        return _run(cfg, out_dir, reschedule, log, only_ligands, only_legs, ligands)
    finally:
        lock.close()


def _run(cfg: Config, out_dir: Path, reschedule: bool, log, only_ligands=None, only_legs=None,
         ligands=None) -> list[dict]:
    log = log or _Log(out_dir / "cufep.log")
    t_run = time.time()
    resolved, report, protein, ligands = plan(cfg, log, ligands)
    selected = select_jobs([lig.name for lig in ligands], resolved.sampling.repeats, only_ligands, only_legs)
    partial = bool(only_ligands or only_legs)
    if partial:
        log("only " + ", ".join(sorted({f"{name}/{leg}" for name, _, leg in selected}))
            + " is run; the other legs keep the results they have")
    resolved.output.dir = out_dir
    for key, value, reason in report.decisions:
        log(f"auto: {key} = {value} ({reason})")
    for w in report.warnings:
        log(f"WARNING: {w}")
    membrane = bool(resolved.membrane.enabled)
    gpus = list(resolved.resources.gpus) if resolved.resources.gpus else [None]
    cfg_dict = resolved.model_dump(mode="json")

    lig_dirs = ligand_directories(out_dir, [lig.name for lig in ligands])
    to_run = {name for name, _, _ in selected}
    repeats = resolved.sampling.repeats
    # Before anything is written or sampled, for all legs of this run: what the output directory
    # holds must be usable under this configuration, and the finished legs that are not part of
    # the run must be legs of the same calculation. A run that is refused here leaves the
    # directory, and its record of the configuration that made its results, as they were.
    try:
        for lig in ligands:
            if lig.name not in to_run:
                continue
            mine = {(r, leg) for name, r, leg in selected if name == lig.name}
            info = check_prepared(resolved, lig, lig_dirs[lig.name], membrane, resolved.solvation.box_shape,
                                  lambda message: None)
            check_partner_legs(lig_dirs[lig.name], lig.name, mine, repeats, resolved, info)
            if info is not None:
                check_before_jobs(resolved, lig_dirs[lig.name], mine)
            for r, leg in sorted(mine):
                if info is not None:
                    check_leg_reuse(resolved, lig_dirs[lig.name], leg, r, info, resolved.sampling.production(leg),
                                    reschedule, log)
    except Exception as exc:
        dump_config(resolved, out_dir / "resolved_config.refused.yaml")
        log(f"refused: {exc}")
        log("the configuration of this attempt is in resolved_config.refused.yaml; resolved_config.yaml and the "
            "results are unchanged")
        raise
    write_run_record(out_dir, resolved, report, protein, log)
    for lig in ligands:
        if lig.name in to_run:
            prepare_ligand(resolved, lig, protein, lig_dirs[lig.name], membrane, resolved.solvation.box_shape, log)
            check_before_jobs(resolved, lig_dirs[lig.name],             # before any job starts
                              {(r, leg) for name, r, leg in selected if name == lig.name})

    t_prepared = time.time()
    sizes = {name: json.loads((lig_dirs[name] / "prep" / "prep.json").read_text()) for name in to_run}
    jobs = [LegJob(ligand=name, leg=leg, repeat=r, lig_dir=str(lig_dirs[name]), devices=[], config=cfg_dict,
                   production_ns=resolved.sampling.production(leg), reschedule=reschedule,
                   n_atoms=int(sizes[name].get(f"n_atoms_{leg}", 0)))
            for name, r, leg in selected]
    log(f"{len(jobs)} job(s): {len(ligands)} ligand(s) x 2 legs x {repeats} repeat(s); per lambda window "
        f"{resolved.sampling.equilibration('complex'):g} ns equilibration and "
        f"{resolved.sampling.production('complex'):g} ns production (complex leg)")
    real_gpus = [g for g in gpus if g is not None]
    factor = resolved.resources.oversubscription_factor
    monitor = GpuMonitor(real_gpus)              # maps the run's devices to nvidia-smi's before MPS starts
    daemon = mps.start(log) if real_gpus and factor != 1 else None
    monitor.start()
    try:
        results = run_jobs(jobs, gpus, log, oversubscription=factor, mps=daemon is not None)
    finally:
        monitor.stop()
        if daemon is not None:
            daemon.stop()
    failed_now: dict[str, list[str]] = {}
    for r in results:
        if "error" in r:
            log(f"FAILED {r['ligand']}/r{r['repeat']}/{r['leg']}: {r['error']}")
            failed_now.setdefault(r["ligand"], []).append(f"{r['leg']} leg" + (f" of repeat {r['repeat']}"
                                                                              if repeats > 1 else ""))
    # a run of some legs only leaves the timing of the whole run in place
    write_timing(out_dir, jobs, t_run, t_prepared, monitor, daemon is not None, log,
                 file_name="timing_rerun.json" if partial else "timing.json")

    # legs of the ligands of this run that this run did not sample keep the result of an earlier run:
    # where that is of another analysis (version of the method, settings), it is made again from the
    # samples. Ligands that are not part of this run (--ligand) are not touched at all: they keep the
    # result they have, combined with the repeats and the settings of the run that made it (review of
    # PR 1: a run of one ligand with one repeat recombined another ligand of three repeats as one)
    for lig in ligands:
        if lig.name not in to_run:
            continue
        for r in range(repeats):
            for leg in ("complex", "solvent"):
                refresh_leg_result(lig_dirs[lig.name] / f"repeat_{r}" / leg, resolved, log)
    summary = [combine_ligand(lig_dirs[lig.name], lig.name, repeats, resolved) if lig.name in to_run
               else stored_ligand_result(lig_dirs[lig.name], lig.name) for lig in ligands]
    for s in summary:
        # legs that this run was to do and that failed; the command ends with an error because of them
        s["failed_in_this_run"] = failed_now.get(s["ligand"], [])
        # the legs this command ran, which decide how it ends (failed_legs)
        s["selected_legs"] = sorted([r, leg] for name, r, leg in selected if name == s["ligand"])
        # a ligand that is not part of this run keeps the repeats of the run that combined it
        s["n_repeats_planned"] = repeats if s["ligand"] in to_run else int(s.get("repeats_planned") or 0)
    write_summary(out_dir, summary, ligands)
    for lig in ligands:
        if lig.name not in to_run:
            continue
        for r in range(repeats):
            try:
                plot_repeat(lig_dirs[lig.name] / f"repeat_{r}")
                # what a restraint chosen from interactions rests on (no figure for another selection)
                plot_interactions(lig_dirs[lig.name] / f"repeat_{r}" / "complex")
            except Exception as exc:  # noqa: BLE001  a figure never fails a run
                log(f"WARNING: no analysis figure for {lig.name}, repeat {r}: {type(exc).__name__}: {exc}")
    for s in summary:
        log(result_line(s))
        for w in s.get("warnings", []):
            log(f"  warning: {w}")
    return summary


def result_line(s: dict) -> str:
    """The line of the log that gives a ligand's binding free energy and everything its result
    says about its legs (:data:`cufep.benchmarks.runner.LEG_FLAGS`)."""
    flags = flags_of(s)
    warnings = f"; {len(s['warnings'])} warning(s):" if s.get("warnings") else ""
    if "dg_bind" not in s:
        return (f"RESULT {s['ligand']}: no binding free energy" + "".join(f"; {flag}" for flag in flags)
                + ("" if flags else "; no complete repeat") + warnings)
    if s.get("dg_bind_std") is not None:
        err = f" ± {s['dg_bind_std']:.2f} kcal/mol (standard deviation over {s['n_repeats']} repeats)"
    elif s.get("dg_bind_error") is not None:
        # of the one calculation: from time blocks of its two legs, or MBAR's own where a leg is
        # too short for blocks (that leg carries a note)
        how = "standard error from time blocks" if s.get("dg_bind_error_from_time_blocks") else "standard error"
        err = f" ± {s['dg_bind_error']:.2f} kcal/mol ({how}, 1 repeat)"
    elif s.get("dg_bind_mbar_error") is not None:  # a result of before the standard error from time blocks
        err = f" ± {s['dg_bind_mbar_error']:.2f} kcal/mol (MBAR uncertainty, 1 repeat)"
    else:
        err = " (no standard error)"
    # a pair of neighbouring windows without overlap leaves the free energy undefined, whatever the
    # standard error says; say so next to the number
    lowest = [v for v in (s.get("min_overlap"), s.get("min_overlap_used")) if v is not None]
    if s.get("not_reliable") and lowest:
        flags = [f"{flag} (neighbouring lambda windows overlap by only {min(lowest):.3f})"
                 if flag.startswith("NOT RELIABLE") else flag for flag in flags]
    return (
        f"RESULT {s['ligand']}: dG_bind = {s['dg_bind']:.2f}{err}" + "".join(f"; {flag}" for flag in flags) + warnings
    )


def write_run_record(out_dir: Path, resolved: Config, report, protein, log=print) -> None:
    """Write the configuration of this run (``resolved_config.yaml``), the report of the automatic
    choices and the prepared protein, after the run has been found usable for the directory.

    If the directory holds the configuration of an earlier run that differs in more than the
    resources and the output directory, that file is kept as ``resolved_config.previous_<n>.yaml``:
    results of the earlier run may still be there (legs this run does not sample).
    """
    path = out_dir / "resolved_config.yaml"
    new = resolved.model_dump(mode="json")
    earlier = read_settings(path) if path.exists() else None

    def calculation(data: dict) -> dict:
        return {key: value for key, value in data.items() if key not in ("resources", "output")}

    if earlier is not None and settings_differences(calculation(earlier), calculation(new)):
        n = 1
        while (out_dir / f"resolved_config.previous_{n}.yaml").exists():
            n += 1
        path.replace(out_dir / f"resolved_config.previous_{n}.yaml")
        log(f"the configuration differs from the one of the earlier run in this directory, which is kept as "
            f"resolved_config.previous_{n}.yaml")
    dump_config(resolved, path)
    (out_dir / "resolved_config.refused.yaml").unlink(missing_ok=True)
    publish_text(out_dir / "auto_report.md", report.to_markdown())
    protein.write_pdb(out_dir / "protein_prepared.pdb")


def failed_legs(summary: list[dict]) -> list[str]:
    """Why the command ends with an error, as "ligand: ..." (empty if it does not): every leg this
    command ran that failed or has no free energy, and every ligand of such a leg whose legs were
    made under different conditions. What other ligands or other legs gave does not decide it
    (review of PR 1: a leg of this run without a free energy ended with 0 because another ligand
    had a result, and a solvent leg run alone in a new directory ended with 1 because its complex
    leg was not there yet). One rule for `cufep run` and `cufep benchmark run`."""
    out = []
    for s in summary:
        ran = {(int(r), leg) for r, leg in s.get("selected_legs", [])}
        if not ran:
            continue
        failed = [(int(r), leg) for r, leg in s.get("failed_leg_ids", [])]
        for r, leg in sorted(set(failed) & ran):
            out.append(f"{s['ligand']}: {leg} leg" + (f" of repeat {r}" if s.get("n_repeats_planned", 1) > 1 else "")
                       + " failed or has no free energy")
        if s.get("inconsistent_legs"):
            out.append(f"{s['ligand']}: legs made under different conditions ({'; '.join(s['inconsistent_legs'])})")
    return out


def waiting_for_partner(summary: list[dict]) -> list[str]:
    """Ligands whose legs of this command all have a result but no binding free energy yet, because a
    leg that this command did not run is not there: not an error, a note."""
    out = []
    for s in summary:
        ran = {(int(r), leg) for r, leg in s.get("selected_legs", [])}
        failed = {(int(r), leg) for r, leg in s.get("failed_leg_ids", [])}
        if ran and not (ran & failed) and s.get("dg_bind") is None and not s.get("inconsistent_legs"):
            missing = sorted({leg for leg in ("complex", "solvent")} - {leg for _r, leg in ran})
            out.append(f"{s['ligand']}: the legs of this run have a result; the binding free energy needs the "
                       f"{' and '.join(missing) or 'other'} leg as well")
    return out


def write_summary(out_dir: Path, summary: list[dict], ligands) -> None:
    exp = {}
    for lig in ligands:
        for key in ("r_exp_dg", "exp_dg", "dG_exp"):
            if key in lig.properties:
                try:
                    exp[lig.name] = float(lig.properties[key])
                except ValueError:
                    pass
    def table(fh) -> None:
        w = csv.writer(fh)
        # one column for each thing a ligand's result can say about its legs (LEG_FLAGS)
        w.writerow(["ligand", "dg_bind_kcal_mol", "std_over_repeats", "standard_error", "mbar_error", "n_repeats",
                    "exp_dg_kcal_mol",
                    "lowest_overlap", "lowest_overlap_of_the_samples_used"] + [key for key, _name in LEG_FLAGS]
                   + ["warnings"])
        for s in summary:
            w.writerow([s["ligand"], _fmt(s.get("dg_bind")), _fmt(s.get("dg_bind_std")), _fmt(s.get("dg_bind_error")),
                        _fmt(s.get("dg_bind_mbar_error")), s.get("n_repeats", 0), _fmt(exp.get(s["ligand"])),
                        _fmt(s.get("min_overlap")), _fmt(s.get("min_overlap_used"))]
                       + ["; ".join(str(item) for item in s.get(key) or []) for key, _name in LEG_FLAGS]
                       + [len(s.get("warnings", []))])

    publish_with(out_dir / "summary.csv", table, newline="")
    publish_text(out_dir / "summary.json", jsonio.dumps(summary, indent=1))


def write_timing(out_dir: Path, jobs, t_run: float, t_prepared: float, monitor, mps_used: bool, log,
                 file_name: str = "timing.json") -> None:
    """Where the time went: per phase and per job, and how busy the GPUs were (``file_name``)."""
    legs = []
    for job in jobs:
        f = job.leg_dir / "timing.json"
        if f.exists():
            legs.append(json.loads(f.read_text()))
    now = time.time()
    phases: dict[str, float] = {}
    for leg in legs:
        for name, seconds in leg["phases"].items():
            phases[name] = phases.get(name, 0.0) + seconds
    # GPU utilization over the periods in which at least one job was sampling
    intervals = [(leg["finished"] - leg["phases"].get("analysis", 0.0) - leg["phases"].get("sampling", 0.0),
                  leg["finished"] - leg["phases"].get("analysis", 0.0)) for leg in legs if "finished" in leg]
    data = {"wall_seconds": now - t_run, "preparation_seconds": t_prepared - t_run,
            "jobs_seconds": now - t_prepared, "mps": mps_used, "gpus": list(monitor.gpus),
            "gpu_utilization_percent": {"whole_run": monitor.mean(), "while_sampling": monitor.mean_over(intervals)},
            "phase_seconds_summed_over_jobs": phases, "jobs": legs}
    publish_text(out_dir / file_name, json.dumps(data, indent=1))
    log(
        f"timing: {data['wall_seconds'] / 60:.1f} min in total; "
        f"preparation {data['preparation_seconds'] / 60:.1f} min; "
        + "; ".join(f"{name} {seconds / 60:.1f} min" for name, seconds in phases.items())
        + " (summed over jobs)"
    )
    util = data["gpu_utilization_percent"]["while_sampling"]
    if util and all(v is not None for v in util.values()):
        log("timing: GPU utilization while sampling: " + ", ".join(f"GPU {g} {v:.0f}%" for g, v in util.items()))
    for leg in legs:
        smp = leg.get("sampling")
        if smp and smp.get("iterations"):
            busy = smp.get("worker_busy_fraction")
            log(f"timing: {leg['ligand']}/r{leg['repeat']}/{leg['leg']}: {smp['lambda_windows']} lambda windows, "
                f"{smp['seconds_per_iteration']:.3f} s per iteration, workers busy "
                f"{100 * busy:.0f}%" if busy is not None else "")


def _fmt(x) -> str:
    return "" if x is None else f"{x:.3f}"


__all__ = ["plan", "run", "failed_legs", "waiting_for_partner", "KCAL"]
