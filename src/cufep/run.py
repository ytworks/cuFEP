"""Top-level driver: configuration -> prepared systems -> legs -> binding free energies."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from cufep.config.auto import AutoReport, resolve
from cufep.config.schema import Config, dump_config
from cufep import mps
from cufep.scheduler import GpuMonitor, run_jobs
from cufep.workflow import KCAL, LegJob, WorkflowError, _Log, combine_ligand, prepare_ligand, safe_name


def plan(cfg: Config, log=print) -> tuple[Config, AutoReport, object, list]:
    """Static part of a run: prepare the protein, read the ligands, resolve ``auto`` settings."""
    from cufep.prep.forcefield import ForceFieldSpec, make_forcefield
    from cufep.prep.ligand import load_ligands
    from cufep.prep.protein import prepare_protein

    ff = make_forcefield(ForceFieldSpec(protein=cfg.forcefield.protein, water=cfg.forcefield.water), [])
    protein = prepare_protein(cfg.system.protein, forcefield=ff, ph=cfg.thermodynamics.ph)
    for line in protein.report.lines():
        log(f"protein: {line}")
    ligands = load_ligands(cfg.system.ligand)
    if cfg.system.ligand_names:
        wanted = set(cfg.system.ligand_names)
        missing = wanted - {lig.name for lig in ligands}
        if missing:
            raise WorkflowError(f"ligands not found in {cfg.system.ligand}: {', '.join(sorted(missing))}")
        ligands = [lig for lig in ligands if lig.name in wanted]
    names = [lig.name for lig in ligands]
    if len(set(safe_name(n) for n in names)) != len(names):
        raise WorkflowError("ligand names in the SDF are not unique; give every record a distinct title line")
    if cfg.system.cofactors:
        raise WorkflowError("system.cofactors is not implemented yet in this version")
    resolved, report = resolve(cfg, protein)
    report.warnings += protein.report.warnings
    if protein.report.protonation_assigned:
        report.warnings.append("protonation states were assigned automatically (no hydrogens in the PDB): "
                               + ", ".join(protein.report.protonation_assigned[:40]))
    log(f"{len(ligands)} ligand(s): {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    return resolved, report, protein, ligands


def run(cfg: Config, reschedule: bool = False, log=None) -> list[dict]:
    """Run the whole calculation for every ligand in the SDF. Safe to call again to resume."""
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log or _Log(out_dir / "cufep.log")
    t_run = time.time()
    resolved, report, protein, ligands = plan(cfg, log)
    resolved.output.dir = out_dir
    dump_config(resolved, out_dir / "resolved_config.yaml")
    (out_dir / "auto_report.md").write_text(report.to_markdown())
    for key, value, reason in report.decisions:
        log(f"auto: {key} = {value} ({reason})")
    for w in report.warnings:
        log(f"WARNING: {w}")
    protein.write_pdb(out_dir / "protein_prepared.pdb")
    membrane = bool(resolved.membrane.enabled)
    gpus = list(resolved.resources.gpus) if resolved.resources.gpus else [None]
    cfg_dict = resolved.model_dump(mode="json")

    lig_dirs = {}
    for lig in ligands:
        lig_dir = out_dir / safe_name(lig.name)
        prepare_ligand(resolved, lig, protein, lig_dir, membrane, resolved.solvation.box_shape, log)
        lig_dirs[lig.name] = lig_dir

    t_prepared = time.time()
    repeats = resolved.sampling.repeats
    sizes = {lig.name: json.loads((lig_dirs[lig.name] / "prep" / "prep.json").read_text()) for lig in ligands}
    jobs = [LegJob(ligand=lig.name, leg=leg, repeat=r, lig_dir=str(lig_dirs[lig.name]), devices=[], config=cfg_dict,
                   production_ns=resolved.sampling.production(leg), reschedule=reschedule,
                   n_atoms=int(sizes[lig.name].get(f"n_atoms_{leg}", 0)))
            for lig in ligands for r in range(repeats) for leg in ("complex", "solvent")]
    log(f"{len(jobs)} job(s): {len(ligands)} ligand(s) x 2 legs x {repeats} repeat(s); per lambda window "
        f"{resolved.sampling.equilibration('complex'):g} ns equilibration and "
        f"{resolved.sampling.production('complex'):g} ns production (complex leg)")
    real_gpus = [g for g in gpus if g is not None]
    factor = resolved.resources.oversubscription_factor
    daemon = mps.start(log) if real_gpus and factor != 1 else None
    monitor = GpuMonitor(real_gpus).start()
    try:
        results = run_jobs(jobs, gpus, log, oversubscription=factor, mps=daemon is not None)
    finally:
        monitor.stop()
        if daemon is not None:
            daemon.stop()
    for r in results:
        if "error" in r:
            log(f"FAILED {r['ligand']}/r{r['repeat']}/{r['leg']}: {r['error']}")
    write_timing(out_dir, jobs, t_run, t_prepared, monitor, daemon is not None, log)

    summary = [combine_ligand(lig_dirs[lig.name], lig.name, repeats) for lig in ligands]
    write_summary(out_dir, summary, ligands)
    for s in summary:
        if "dg_bind" in s:
            if s.get("dg_bind_std") is not None:
                err = f" ± {s['dg_bind_std']:.2f} kcal/mol (standard deviation over {s['n_repeats']} repeats)"
            else:
                err = f" ± {s['dg_bind_mbar_error']:.2f} kcal/mol (MBAR uncertainty, 1 repeat)"
            log(f"RESULT {s['ligand']}: dG_bind = {s['dg_bind']:.2f}{err}"
                + (f"; {len(s['warnings'])} warning(s), see summary.json" if s.get("warnings") else ""))
        else:
            log(f"RESULT {s['ligand']}: no complete repeat")
    return summary


def write_summary(out_dir: Path, summary: list[dict], ligands) -> None:
    exp = {}
    for lig in ligands:
        for key in ("r_exp_dg", "exp_dg", "dG_exp"):
            if key in lig.properties:
                try:
                    exp[lig.name] = float(lig.properties[key])
                except ValueError:
                    pass
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand", "dg_bind_kcal_mol", "std_over_repeats", "mbar_error", "n_repeats", "exp_dg_kcal_mol",
                    "warnings"])
        for s in summary:
            w.writerow([s["ligand"], _fmt(s.get("dg_bind")), _fmt(s.get("dg_bind_std")),
                        _fmt(s.get("dg_bind_mbar_error")), s.get("n_repeats", 0), _fmt(exp.get(s["ligand"])),
                        len(s.get("warnings", []))])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


def write_timing(out_dir: Path, jobs, t_run: float, t_prepared: float, monitor, mps_used: bool, log) -> None:
    """Where the time went: per phase and per job, and how busy the GPUs were (timing.json)."""
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
    (out_dir / "timing.json").write_text(json.dumps(data, indent=1))
    log(f"timing: {data['wall_seconds'] / 60:.1f} min in total; preparation {data['preparation_seconds'] / 60:.1f} min; "
        + "; ".join(f"{name} {seconds / 60:.1f} min" for name, seconds in phases.items()) + " (summed over jobs)")
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


__all__ = ["plan", "run", "KCAL"]
