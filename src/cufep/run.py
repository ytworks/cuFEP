"""Top-level driver: configuration -> prepared systems -> legs -> binding free energies."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from cufep.config.auto import AutoReport, resolve
from cufep.config.schema import Config, dump_config
from cufep.scheduler import run_jobs
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

    lengths = list(resolved.sampling.production_ns)
    repeats = resolved.sampling.repeats

    def jobs_for(names_legs, length):
        return [LegJob(ligand=name, leg=leg, repeat=r, lig_dir=str(lig_dirs[name]), devices=[],
                       config=cfg_dict, production_ns=length, reschedule=reschedule)
                for name, leg in names_legs for r in range(repeats)]

    todo = [(lig.name, leg) for lig in ligands for leg in ("complex", "solvent")]
    for n, length in enumerate(lengths):
        if not todo:
            break
        log(f"sampling round {n + 1}: {len(todo)} leg(s) x {repeats} repeat(s), production {length:g} ns per state")
        results = run_jobs(jobs_for(todo, length), gpus, log)
        failed = [r for r in results if "error" in r]
        if failed:
            for r in failed:
                log(f"FAILED {r['ligand']}/r{r['repeat']}/{r['leg']}: {r['error']}")
        # extend a leg (all of its repeats together) only if a pre-declared criterion is not met
        todo_next = []
        if n + 1 < len(lengths):
            for name, leg in todo:
                mine = [r for r in results if r.get("ligand") == name and r.get("leg") == leg and "error" not in r]
                if mine and not all(r.get("sampling_sufficient", True) for r in mine):
                    todo_next.append((name, leg))
                    reasons = sorted({x for r in mine for x in r.get("sampling_reasons", [])})
                    log(f"{name}/{leg}: extending to {lengths[n + 1]:g} ns ({'; '.join(reasons)})")
        todo = todo_next

    summary = [combine_ligand(lig_dirs[lig.name], lig.name, repeats) for lig in ligands]
    write_summary(out_dir, summary, ligands)
    for s in summary:
        if "dg_bind" in s:
            sd = f" ± {s['dg_bind_std']:.2f}" if s.get("dg_bind_std") is not None else ""
            log(f"RESULT {s['ligand']}: dG_bind = {s['dg_bind']:.2f}{sd} kcal/mol ({s['n_repeats']} repeat(s))")
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


def _fmt(x) -> str:
    return "" if x is None else f"{x:.3f}"


__all__ = ["plan", "run", "KCAL"]
