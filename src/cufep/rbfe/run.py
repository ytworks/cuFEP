"""Driver of a relative run: configuration -> network -> prepared edges -> legs -> ddG and dG.

Output directory
    ``resolved_config.yaml``  the configuration of the run, auto settings resolved
    ``network.json``          the ligands and the edges (given or planned by LOMAP)
    ``edges/<A>~<B>/``        an edge: ``prep/`` (end states, mapping), ``repeat_<r>/{complex,solvent}/`` (legs)
    ``abfe/``                 the ABFE run of the reference ligands (``dg.references: [abfe]``)
    ``edges.csv``, ``ligands.csv``, ``summary.json``  the results; ``cufep.log``, ``timing.json``
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import yaml

from cufep import jsonio
from cufep.files import publish_text, publish_with
from cufep.rbfe.config import RbfeConfig
from cufep.rbfe.network import Edge, cycle_closures, dg_from_abfe, dg_from_experiment, edge_ddg
from cufep.rbfe.prep import edge_name
from cufep.workflow import KCAL, WorkflowError, _Log, check_interaction_inputs, select_ligands


def ligands_of(rcfg: RbfeConfig) -> list:
    from cufep.prep.ligand import load_ligands

    ligands = select_ligands(load_ligands(rcfg.system.ligands), rcfg.system.ligand_names, "system.ligand_names")
    names = [lig.name for lig in ligands]
    if len(set(names)) != len(names):
        raise WorkflowError("ligand names in the SDF are not unique; give every record a distinct title line")
    if any("~" in name or "/" in name for name in names):
        raise WorkflowError("a ligand name has '~' or '/', which name the directories of the edges")
    return ligands


def experiment_of(rcfg: RbfeConfig, ligands) -> dict[str, float]:
    """Experimental binding free energies (kcal/mol) of the ligands that have one: the CSV of
    ``system.experiment`` or the tags of the SDF records."""
    out: dict[str, float] = {}
    if rcfg.system.experiment is not None:
        with open(rcfg.system.experiment, newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["ligand"].strip()] = float(row["dg"])
        return {lig.name: out[lig.name] for lig in ligands if lig.name in out}
    for lig in ligands:
        for key in ("r_exp_dg", "exp_dg", "dG_exp"):
            if key in lig.properties:
                try:
                    out[lig.name] = float(lig.properties[key])
                    break
                except ValueError:
                    pass
    return out


def edges_of(rcfg: RbfeConfig, ligands, log=print) -> tuple[list[tuple[str, str]], str]:
    """The edges of the run and how they were obtained."""
    names = [lig.name for lig in ligands]
    if rcfg.network.edges is not None:
        edges = [(a, b) for a, b in rcfg.network.edges]
        unknown = sorted({n for e in edges for n in e} - set(names))
        if unknown:
            raise WorkflowError(f"network.edges names ligands that are not ligands of the run: {', '.join(unknown)}")
        how = "given (network.edges)"
    else:
        from cufep.rbfe.mapping import MappingSettings, plan_network

        m = rcfg.mapping
        settings = MappingSettings(atom_max_distance=m.atom_max_distance_A,
                                   map_hydrogens_on_hydrogens_only=m.map_hydrogens_on_hydrogens_only,
                                   map_exact_ring_matches_only=m.map_exact_ring_matches_only,
                                   allow_partial_fused_rings=m.allow_partial_fused_rings,
                                   allow_bond_breaks=m.allow_bond_breaks)
        n = rcfg.network
        edges = plan_network(ligands, settings, n.distance_cutoff, n.max_path_length, n.require_cycle_covering)
        how = "planned by LOMAP"
    import networkx as nx

    graph = nx.Graph(edges)
    graph.add_nodes_from(names)
    if not nx.is_connected(graph):
        raise WorkflowError("the edges do not connect all ligands of the run: "
                            + "; ".join(", ".join(sorted(c)) for c in nx.connected_components(graph)))
    log(f"{len(edges)} edge(s) ({how}): " + ", ".join(f"{a} -> {b}" for a, b in edges))
    return edges, how


def run_rbfe(rcfg: RbfeConfig, reschedule: bool = False, log=None, only_edges=None) -> dict:
    """Run the whole relative calculation. Safe to call again to resume."""
    from cufep.run import lock_output_directory

    out_dir = Path(rcfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_output_directory(out_dir)
    try:
        return _run(rcfg, out_dir, reschedule, log or _Log(out_dir / "cufep.log"), only_edges)
    finally:
        lock.close()


def _run(rcfg: RbfeConfig, out_dir: Path, reschedule: bool, log, only_edges=None) -> dict:
    from cufep.rbfe.leg import HybridLegJob
    from cufep.rbfe.prep import prepare_edge
    from cufep.run import execute_jobs, finish_run, lock_output_directory, plan, prepare_run, write_timing

    t_run = time.time()
    ligands = ligands_of(rcfg)
    if len(ligands) < 2:
        raise WorkflowError("a relative calculation needs at least two ligands")
    refs = list(rcfg.dg.abfe_ligands)
    missing = sorted(set(refs) - {lig.name for lig in ligands})
    if missing:
        raise WorkflowError(f"dg.abfe_ligands names ligands that are not ligands of the run: {', '.join(missing)}")
    resolved, report, protein, ligands = plan(rcfg.shared(), log, ligands)
    # the settings auto resolved for the shared sections are those of the relative run
    rcfg = RbfeConfig.model_validate({**rcfg.model_dump(mode="json"),
                                      "backend": resolved.backend.model_dump(mode="json"),
                                      "resources": resolved.resources.model_dump(mode="json"),
                                      "solvation": resolved.solvation.model_dump(mode="json"),
                                      "output": {"dir": str(out_dir)}})
    for key, value, reason in report.decisions:
        log(f"auto: {key} = {value} ({reason})")
    for w in report.warnings:
        log(f"WARNING: {w}")
    edges, how = edges_of(rcfg, ligands, log)
    names = [edge_name(a, b) for a, b in edges]
    selected = [e for e, name in zip(edges, names) if not only_edges or name in only_edges]
    unknown = sorted(set(only_edges or []) - set(names))
    if unknown:
        raise WorkflowError(f"--edge: {', '.join(unknown)} is not an edge of the run (edges: {', '.join(names)})")
    experiment = experiment_of(rcfg, ligands)
    write_run_files(out_dir, rcfg, [lig.name for lig in ligands], edges, how, experiment)
    publish_text(out_dir / "auto_report.md", report.to_markdown())
    protein.write_pdb(out_dir / "protein_prepared.pdb")
    by_name = {lig.name: lig for lig in ligands}
    box_shape = rcfg.solvation.box_shape
    infos = {}
    for a, b in selected:
        infos[edge_name(a, b)] = prepare_edge(rcfg, by_name[a], by_name[b], protein,
                                              out_dir / "edges" / edge_name(a, b), box_shape, log)
    cfg_dict = rcfg.model_dump(mode="json")
    repeats = rcfg.sampling.repeats
    jobs = [HybridLegJob(ligand=edge_name(a, b), leg=leg, repeat=r, lig_dir=str(out_dir / "edges" / edge_name(a, b)),
                         devices=[], config=cfg_dict, production_ns=float(rcfg.sampling.production_ns),
                         reschedule=reschedule, n_atoms=int(infos[edge_name(a, b)]["legs"][leg]["n_atoms_a"]))
            for a, b in selected for r in range(repeats) for leg in ("solvent", "complex")]
    abfe_run = None
    abfe_lock = None
    if "abfe" in rcfg.dg.references:
        abfe_dir = out_dir / "abfe"
        abfe_dir.mkdir(parents=True, exist_ok=True)
        abfe_lock = lock_output_directory(abfe_dir)
        abfe_cfg = rcfg.abfe_config(refs)
        # what the restraint of the ABFE needs, before any MD (as `cufep run` checks it)
        check_interaction_inputs(abfe_cfg, [by_name[name] for name in refs])
        abfe_run = prepare_run(abfe_cfg, abfe_dir, reschedule, _Log(abfe_dir / "cufep.log"),
                               ligands=[by_name[name] for name in refs])
    t_prepared = time.time()
    log(f"{len(jobs)} job(s) of {len(selected)} edge(s) x 2 legs x {repeats} repeat(s)"
        + (f" and {len(abfe_run.jobs)} of the ABFE of {', '.join(refs)}" if abfe_run else "")
        + f"; per lambda window {rcfg.sampling.equilibration('complex'):g} ns equilibration and "
        f"{rcfg.sampling.production_ns:g} ns production")
    gpus = list(rcfg.resources.gpus) if rcfg.resources.gpus else [None]
    try:
        results, monitor, mps_used = execute_jobs(jobs + (abfe_run.jobs if abfe_run else []), gpus,
                                                  rcfg.resources.oversubscription_factor, log)
        abfe_summary = finish_run(abfe_run, results[len(jobs):], monitor, mps_used) if abfe_run else []
    finally:
        if abfe_lock is not None:
            abfe_lock.close()
    for r in results[:len(jobs)]:
        if "error" in r:
            log(f"FAILED {r.get('ligand')}/r{r.get('repeat')}/{r.get('leg')}: {r['error']}")
    write_timing(out_dir, jobs, t_run, t_prepared, monitor, mps_used, log)
    summary = combine_network(out_dir, [lig.name for lig in ligands], edges, repeats, rcfg, experiment, abfe_summary)
    write_results(out_dir, summary)
    for line in result_lines(summary):
        log(line)
    return summary


def write_run_files(out_dir: Path, rcfg: RbfeConfig, ligands: list[str], edges, how: str,
                    experiment: dict[str, float]) -> None:
    """The record of a run that its results are made from again (:func:`analyse_directory`): the configuration
    and the network."""
    publish_text(out_dir / "resolved_config.yaml", yaml.safe_dump(rcfg.model_dump(mode="json"), sort_keys=False))
    publish_text(out_dir / "network.json", json.dumps({"ligands": list(ligands), "edges": [list(e) for e in edges],
                                                       "edges_from": how, "experiment_kcal_mol": experiment},
                                                      indent=1))


def edge_result(edge_dir: Path, a: str, b: str, repeats: int) -> dict:
    """ddG of an edge from the results of its legs (kcal/mol); repeats are averaged, their standard deviation
    given (as the ABFE combines repeats)."""
    out = {"edge": edge_name(a, b), "ligand_a": a, "ligand_b": b, "repeats": []}
    for r in range(repeats):
        legs = {}
        for leg in ("complex", "solvent"):
            f = Path(edge_dir) / f"repeat_{r}" / leg / "result.json"
            legs[leg] = json.loads(f.read_text()) if f.exists() else None
        if not all(legs.values()) or any(legs[leg].get("delta_g") is None for leg in legs):
            out["repeats"].append({"repeat": r, "missing": [leg for leg in legs if not legs[leg]
                                                             or legs[leg].get("delta_g") is None]})
            continue
        c, s = legs["complex"], legs["solvent"]
        ddg, se = edge_ddg(c["delta_g"], c["delta_g_error"], s["delta_g"], s["delta_g_error"])
        out["repeats"].append({
            "repeat": r, "ddg": ddg / KCAL, "se": se / KCAL, "complex_dg": c["delta_g"] / KCAL,
            "complex_se": c["delta_g_error"] / KCAL, "solvent_dg": s["delta_g"] / KCAL,
            "solvent_se": s["delta_g_error"] / KCAL,
            "windows": {leg: legs[leg]["n_states"] for leg in legs},
            "lowest_overlap": {leg: legs[leg]["quality"]["overlap"]["lowest"] for leg in legs},
            "not_reliable": [leg for leg in legs if not legs[leg]["quality"]["overlap"]["ok"]],
            "fewest_independent_samples": {leg: legs[leg]["quality"]["samples"].get("counted") for leg in legs},
            "backend": {leg: legs[leg].get("backend") for leg in legs},
            "warnings": [f"{leg}: {n}" for leg in legs for n in legs[leg].get("notes", [])]})
    done = [r for r in out["repeats"] if "ddg" in r]
    if done:
        values = [r["ddg"] for r in done]
        out["ddg"] = sum(values) / len(values)
        if len(done) > 1:
            mean = out["ddg"]
            out["ddg_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            out["se"] = out["ddg_std"] / math.sqrt(len(values))
        else:
            out["se"] = done[0]["se"]
        out["not_reliable"] = sorted({leg for r in done for leg in r["not_reliable"]})
    return out


def combine_network(out_dir: Path, ligands: list[str], edges: list[tuple[str, str]], repeats: int, rcfg,
                    experiment: dict[str, float], abfe_summary: list[dict]) -> dict:
    """The results of the edges, the cycle closures and the dG of every ligand by each reference."""
    results = [edge_result(out_dir / "edges" / edge_name(a, b), a, b, repeats) for a, b in edges]
    measured = [Edge(r["ligand_a"], r["ligand_b"], r["ddg"], r["se"]) for r in results if "ddg" in r]
    summary = {"ligands": ligands, "edges": results, "experiment_kcal_mol": experiment,
               "cycle_closures": cycle_closures(measured), "dg": {}, "notes": []}
    for r in results:
        if experiment.get(r["ligand_a"]) is not None and experiment.get(r["ligand_b"]) is not None:
            r["exp_ddg"] = experiment[r["ligand_b"]] - experiment[r["ligand_a"]]
    import networkx as nx

    graph = nx.Graph([(e.ligand_a, e.ligand_b) for e in measured])
    graph.add_nodes_from(ligands)
    if not measured or not nx.is_connected(graph):
        summary["notes"].append("the edges with a ddG do not connect all ligands: no dG")
        return summary
    if "experiment" in rcfg.dg.references:
        if experiment:
            values = dg_from_experiment(ligands, measured, experiment)
            summary["dg"]["experiment"] = {name: {"dg": v, "se": e} for name, (v, e) in values.items()}
        else:
            summary["notes"].append("dg.references has experiment, but no ligand has an experimental value")
    if "abfe" in rcfg.dg.references:
        abfe = {}
        for s in abfe_summary:
            if s.get("dg_bind") is not None:
                se = s.get("dg_bind_error") if s.get("dg_bind_std") is None else s["dg_bind_std"]
                if se:
                    abfe[s["ligand"]] = (float(s["dg_bind"]), float(se))
        summary["abfe_kcal_mol"] = {name: {"dg": v, "se": e} for name, (v, e) in abfe.items()}
        if abfe:
            summary["dg"]["abfe"] = {name: {"dg": v, "se": e}
                                     for name, (v, e) in dg_from_abfe(ligands, measured, abfe).items()}
        else:
            summary["notes"].append("dg.references has abfe, but no reference ligand has an ABFE result")
    return summary


def write_results(out_dir: Path, summary: dict) -> None:
    def fmt(x) -> str:
        return "" if x is None else f"{x:.3f}"

    def edges_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["edge", "ligand_a", "ligand_b", "ddg_kcal_mol", "standard_error", "exp_ddg_kcal_mol",
                    "complex_dg", "solvent_dg", "windows_complex", "windows_solvent", "lowest_overlap_complex",
                    "lowest_overlap_solvent", "not_reliable", "backend_complex", "backend_solvent"])
        for e in summary["edges"]:
            r = next((x for x in e["repeats"] if "ddg" in x), {})
            w.writerow([e["edge"], e["ligand_a"], e["ligand_b"], fmt(e.get("ddg")), fmt(e.get("se")),
                        fmt(e.get("exp_ddg")), fmt(r.get("complex_dg")), fmt(r.get("solvent_dg")),
                        (r.get("windows") or {}).get("complex", ""), (r.get("windows") or {}).get("solvent", ""),
                        fmt((r.get("lowest_overlap") or {}).get("complex")),
                        fmt((r.get("lowest_overlap") or {}).get("solvent")), "; ".join(e.get("not_reliable", [])),
                        (r.get("backend") or {}).get("complex", ""), (r.get("backend") or {}).get("solvent", "")])

    def ligands_table(fh) -> None:
        w = csv.writer(fh)
        refs = list(summary["dg"])
        w.writerow(["ligand", "exp_dg_kcal_mol"] + [f"dg_{ref}_kcal_mol" for ref in refs]
                   + [f"se_{ref}" for ref in refs] + ["abfe_dg_kcal_mol"])
        abfe = summary.get("abfe_kcal_mol", {})
        for name in summary["ligands"]:
            w.writerow([name, fmt(summary["experiment_kcal_mol"].get(name))]
                       + [fmt(summary["dg"][ref].get(name, {}).get("dg")) for ref in refs]
                       + [fmt(summary["dg"][ref].get(name, {}).get("se")) for ref in refs]
                       + [fmt(abfe.get(name, {}).get("dg"))])

    publish_with(out_dir / "edges.csv", edges_table, newline="")
    publish_with(out_dir / "ligands.csv", ligands_table, newline="")
    publish_text(out_dir / "summary.json", jsonio.dumps(summary, indent=1))


def result_lines(summary: dict) -> list[str]:
    lines = []
    for e in summary["edges"]:
        if "ddg" in e:
            exp = f" (experiment {e['exp_ddg']:+.2f})" if e.get("exp_ddg") is not None else ""
            flag = f"; NOT RELIABLE ({', '.join(e['not_reliable'])} leg)" if e.get("not_reliable") else ""
            lines.append(f"RESULT {e['edge']}: ddG = {e['ddg']:+.2f} ± {e['se']:.2f} kcal/mol{exp}{flag}")
        else:
            lines.append(f"RESULT {e['edge']}: no ddG")
    for c in summary["cycle_closures"]:
        lines.append(f"cycle {' -> '.join(c['ligands'])}: sum of ddG {c['sum_kcal_mol']:+.2f} ± {c['se_kcal_mol']:.2f}")
    for ref, values in summary["dg"].items():
        for name, v in values.items():
            exp = summary["experiment_kcal_mol"].get(name)
            lines.append(f"dG ({ref} reference) {name}: {v['dg']:.2f} ± {v['se']:.2f} kcal/mol"
                         + (f" (experiment {exp:.2f})" if exp is not None else ""))
    lines += [f"note: {n}" for n in summary["notes"]]
    return lines


def failed_edges(summary: dict) -> list[str]:
    return [e["edge"] for e in summary["edges"] if "ddg" not in e]


def analyse_directory(out_dir: Path) -> dict:
    """The results of a relative run from the results its legs have (and those of its ABFE), as the run makes
    them; writes nothing."""
    out_dir = Path(out_dir)
    rcfg = RbfeConfig.model_validate(yaml.safe_load((out_dir / "resolved_config.yaml").read_text()))
    network = json.loads((out_dir / "network.json").read_text())
    abfe_file = out_dir / "abfe" / "summary.json"
    abfe_summary = json.loads(abfe_file.read_text()) if abfe_file.exists() else []
    return combine_network(out_dir, network["ligands"], [tuple(e) for e in network["edges"]], rcfg.sampling.repeats,
                           rcfg, {k: float(v) for k, v in network.get("experiment_kcal_mol", {}).items()}, abfe_summary)
