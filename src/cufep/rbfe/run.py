"""Driver of a relative run: configuration -> network -> prepared edges -> legs -> ddG and dG.

Output directory
    ``resolved_config.yaml``  the configuration of the run, auto settings resolved
    ``network.json``          the ligands and the edges (given or planned by LOMAP)
    ``edges/<A>~<B>/``        an edge: ``prep/`` (end states, mapping), ``repeat_<r>/{complex,solvent}/`` (legs)
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
from cufep.rbfe.config import RbfeConfig, RbfeSampling
from cufep.rbfe.network import Edge, compound_dg, cycle_closures, edge_ddg, ligand_dg, network_ddg
from cufep.rbfe.prep import edge_name
from cufep.workflow import KCAL, WorkflowError, _Log, select_ligands


def ligands_of(rcfg: RbfeConfig) -> list:
    from cufep.prep.ligand import load_ligands

    ligands = select_ligands(load_ligands(rcfg.system.ligands), rcfg.system.ligand_names, "system.ligand_names")
    names = [lig.name for lig in ligands]
    if len(set(names)) != len(names):
        raise WorkflowError("ligand names in the SDF are not unique; give every record a distinct title line")
    if any("~" in name or "/" in name for name in names):
        raise WorkflowError("a ligand name has '~' or '/', which name the directories of the edges")
    return ligands


def states_of(rcfg: RbfeConfig, ligands) -> dict | None:
    """The compounds of the ligands (``system.states``; None without it): ``{"compound": {ligand: compound}, "form":
    {ligand: form}, "fraction": {ligand: population in solution}, "symmetry": {compound: symmetry number}}``.
    Refused (before anything is written) where it does not name every ligand of the run, where two states of one form
    of a compound have other populations, where the populations of the forms of a compound do not add up to 1, or
    where a compound has two symmetry numbers."""
    if rcfg.system.states is None:
        return None
    path = rcfg.system.states
    with open(path, newline="") as fh:
        rows = {r["ligand"].strip(): r for r in csv.DictReader(fh)}
    names = [lig.name for lig in ligands]
    missing = [n for n in names if n not in rows]
    if missing:
        raise WorkflowError(f"system.states ({path}) does not name the ligands {', '.join(missing)}")
    out = {"compound": {}, "form": {}, "fraction": {}, "symmetry": {}}
    problems = []
    for n in names:
        r = rows[n]
        compound = r["compound"].strip()
        out["compound"][n] = compound
        out["form"][n] = (r.get("form") or "").strip()
        out["fraction"][n] = float(r.get("solution_fraction") or 1.0)
        if not 0.0 < out["fraction"][n] <= 1.0:
            problems.append(f"{n}: solution_fraction {out['fraction'][n]:g} is not in (0, 1]")
    # what belongs to a compound is checked over all its rows in the file (a run may take some of its states)
    for n, r in rows.items():
        sigma = int(r.get("symmetry_number") or 1)
        if sigma < 1:
            problems.append(f"{n}: symmetry_number {sigma} is not a positive number")
        if out["symmetry"].setdefault(r["compound"].strip(), sigma) != sigma:
            problems.append(f"compound {r['compound'].strip()} has two symmetry numbers")
    out["symmetry"] = {c: s for c, s in out["symmetry"].items() if c in set(out["compound"].values())}
    # the forms of a compound are those of all its rows in the file (a run may take some of its states)
    for compound in set(out["compound"].values()):
        forms: dict[str, set] = {}
        for r in rows.values():
            if r["compound"].strip() == compound:
                forms.setdefault((r.get("form") or "").strip(), set()).add(
                    round(float(r.get("solution_fraction") or 1.0), 12))
        if any(len(v) > 1 for v in forms.values()):
            problems.append(f"compound {compound}: states of one form have other solution fractions")
        elif abs(sum(next(iter(v)) for v in forms.values()) - 1.0) > 1e-6:
            problems.append(f"compound {compound}: the solution fractions of its forms add up to "
                            f"{sum(next(iter(v)) for v in forms.values()):g}, not 1")
    if problems:
        raise WorkflowError(f"system.states ({path}): " + "; ".join(problems))
    # the states of the file that the run leaves out, by compound (the dG of such a compound combines the others)
    out["left_out"] = {c: sorted(n for n, r in rows.items() if r["compound"].strip() == c and n not in names)
                       for c in sorted(set(out["compound"].values()))}
    out["left_out"] = {c: v for c, v in out["left_out"].items() if v}
    return out


def compound_experiment(states: dict, experiment: dict[str, float]) -> dict[str, float]:
    """The experimental dG of every compound: that of its states (the value of a compound is given for each of its
    states, the tags of their records); refused where two states of one compound have other values."""
    out: dict[str, float] = {}
    for ligand, compound in states["compound"].items():
        if ligand in experiment:
            if compound in out and abs(out[compound] - experiment[ligand]) > 1e-6:
                raise WorkflowError(f"the states of compound {compound} have other experimental values")
            out[compound] = float(experiment[ligand])
    return out


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
        from cufep.rbfe.mapping import mapping_settings, plan_network

        m = rcfg.mapping
        settings = mapping_settings(m)
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


def check_windows_of_the_solvent_leg(rcfg: RbfeConfig, out_dir: Path, edges, only_legs) -> None:
    """Refuses, before the edges are prepared, a run of complex legs alone that take the windows of the solvent leg
    (`lambda_schedule.adaptive.legs: solvent`) when a solvent leg has none yet: with no solvent leg to run, the
    complex leg would wait for ever."""
    s = rcfg.lambda_schedule
    if not only_legs or "solvent" in only_legs or s.mode != "adaptive" or s.adaptive.legs != "solvent":
        return
    missing = [f"{edge_name(a, b)}/repeat_{r}" for a, b in edges for r in range(rcfg.sampling.repeats)
               if not all((out_dir / "edges" / edge_name(a, b) / f"repeat_{r}" / "solvent" / f).exists()
                          for f in ("schedule.yaml", "start_configs.npz"))]
    if missing:
        raise WorkflowError("--leg complex with lambda_schedule.adaptive.legs: solvent takes the windows of the "
                            f"solvent leg, which {', '.join(missing)} does not have yet: run the solvent leg first")


def check_legs_have_their_windows(out_dir: Path, edges, repeats: int, only_legs) -> None:
    """Before a run moves or writes anything: every leg it runs that has samples (or the checkpoint of their start)
    has the schedule of their windows (:func:`cufep.engine.leg.check_samples_have_their_windows`), so that a leg is
    not refused after its result was set aside (fifth review of PR #18, the class of the ABFE's)."""
    from cufep.alchemy.schedule import Schedule
    from cufep.engine.leg import check_samples_have_their_windows

    for a, b in edges:
        for r in range(repeats):
            for leg in ("solvent", "complex"):
                if only_legs and leg not in only_legs:
                    continue
                leg_dir = out_dir / "edges" / edge_name(a, b) / f"repeat_{r}" / leg
                sched = leg_dir / "schedule.yaml"
                check_samples_have_their_windows(
                    leg_dir, [state.to_dict() for state in Schedule.load(sched).states] if sched.exists() else None)


def run_rbfe(rcfg: RbfeConfig, reschedule: bool = False, log=None, only_edges=None, only_legs=None) -> dict:
    """Run the whole relative calculation. Safe to call again to resume."""
    from cufep.run import lock_output_directory

    out_dir = Path(rcfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_output_directory(out_dir)
    try:
        return _run(rcfg, out_dir, reschedule, log or _Log(out_dir / "cufep.log"), only_edges, only_legs)
    finally:
        lock.close()


def _run(rcfg: RbfeConfig, out_dir: Path, reschedule: bool, log, only_edges=None, only_legs=None) -> dict:
    from cufep.rbfe.leg import EdgeLegJob
    from cufep.rbfe.prep import prepare_edge, prepared_edge
    from cufep.run import execute_jobs, prepare_protein_and_settings, write_timing

    t_run = time.time()
    ligands = ligands_of(rcfg)
    if len(ligands) < 2:
        raise WorkflowError("a relative calculation needs at least two ligands")
    resolved, report, protein = prepare_protein_and_settings(rcfg, rcfg.system.protein, log)
    names = [lig.name for lig in ligands]
    log(f"{len(ligands)} ligand(s): {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    rcfg = RbfeConfig.model_validate({**resolved.model_dump(mode="json"), "output": {"dir": str(out_dir)}})
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
    check_windows_of_the_solvent_leg(rcfg, out_dir, selected, only_legs)
    from cufep.rbfe.charge import charge_change_problems

    problems = charge_change_problems(ligands, edges)
    if problems:
        raise WorkflowError("; ".join(problems))
    by_name = {lig.name: lig for lig in ligands}
    box_shape = rcfg.solvation.box_shape
    # every prepared edge of the network, before the run writes anything: those it runs, and those whose results it
    # combines (second review of PR #17: an edge of another protein was combined); a refused run keeps the record
    # of the run whose results the directory holds (review of PR #17)
    for a, b in edges:
        prepared_edge(rcfg, by_name[a], by_name[b], out_dir / "edges" / edge_name(a, b), box_shape)
    if not reschedule:
        check_legs_have_their_windows(out_dir, selected, rcfg.sampling.repeats, only_legs)
    experiment = experiment_of(rcfg, ligands)
    states = states_of(rcfg, ligands)
    if states is not None:
        compound_experiment(states, experiment)            # refused before anything is written
    infos = {}
    for a, b in selected:
        infos[edge_name(a, b)] = prepare_edge(rcfg, by_name[a], by_name[b], protein,
                                              out_dir / "edges" / edge_name(a, b), box_shape, log)
    # the record of the run after its edges are prepared: an edge whose preparation stops (no alchemical water, no
    # mapping) leaves the record of the run whose results the directory holds (review of PR #19)
    write_run_files(out_dir, rcfg, [lig.name for lig in ligands], edges, how, experiment, states)
    publish_text(out_dir / "auto_report.md", report.to_markdown())
    protein.write_pdb(out_dir / "protein_prepared.pdb")
    cfg_dict = rcfg.model_dump(mode="json")
    repeats = rcfg.sampling.repeats
    jobs = [EdgeLegJob(edge=edge_name(a, b), leg=leg, repeat=r, edge_dir=str(out_dir / "edges" / edge_name(a, b)),
                         devices=[], config=cfg_dict, production_ns=float(rcfg.sampling.production_ns),
                         reschedule=reschedule, n_atoms=int(infos[edge_name(a, b)]["legs"][leg]["n_atoms_a"]))
            for a, b in selected for r in range(repeats) for leg in ("solvent", "complex")
            if not only_legs or leg in only_legs]
    t_prepared = time.time()
    log(f"{len(jobs)} job(s) of {len(selected)} edge(s) x 2 legs x {repeats} repeat(s); per lambda window "
        f"{rcfg.sampling.equilibration('complex'):g} ns equilibration and "
        f"{rcfg.sampling.production_ns:g} ns production")
    gpus = list(rcfg.resources.gpus) if rcfg.resources.gpus else [None]
    results, monitor, mps_used = execute_jobs(jobs, gpus, rcfg.resources.oversubscription_factor, log)
    failed_now = failed_jobs(results)
    for line in failed_now:
        log(f"FAILED {line}")
    write_timing(out_dir, jobs, t_run, t_prepared, monitor, mps_used, log)
    summary = combine_network(out_dir, [lig.name for lig in ligands], edges, repeats, experiment, states,
                              rcfg.thermodynamics.temperature_K)
    # what the exit code of `cufep rbfe run` says: whether a job of this run failed (review of PR #17)
    summary["failed_in_this_run"] = failed_now
    write_results(out_dir, summary)
    for line in result_lines(summary):
        log(line)
    return summary


def write_run_files(out_dir: Path, rcfg: RbfeConfig, ligands: list[str], edges, how: str,
                    experiment: dict[str, float], states: dict | None = None) -> None:
    """The record of a run that its results are made from again (:func:`analyse_directory`): the configuration
    and the network (with the compounds of its states, if it has some, and the temperature that combines them)."""
    publish_text(out_dir / "resolved_config.yaml", yaml.safe_dump(rcfg.model_dump(mode="json"), sort_keys=False))
    record = {"ligands": list(ligands), "edges": [list(e) for e in edges], "edges_from": how,
              "experiment_kcal_mol": experiment}
    if states is not None:
        record.update(states=states, temperature_K=float(rcfg.thermodynamics.temperature_K))
    publish_text(out_dir / "network.json", json.dumps(record, indent=1))


def edge_result(edge_dir: Path, a: str, b: str, repeats: int) -> dict:
    """ddG of an edge from the results of its legs (kcal/mol); repeats are averaged, their standard deviation
    given (as the ABFE combines repeats).

    A repeat counts only with both legs and the standard error of both (a leg whose error could not be determined
    is named under ``no_standard_error``). The legs of all repeats that count must have been made under the same
    conditions (:func:`cufep.workflow.cycle_differences`, the check of the ABFE: temperature, pressure, prepared
    systems ...); if they were not, the edge gets no ddG and says what differs (``inconsistent_legs``) (review of
    PR #17: legs at 298 K and 310 K were subtracted). The free energies of the legs are the means over the same
    repeats as the ddG, which is their difference."""
    from cufep.workflow import cycle_differences

    out = {"edge": edge_name(a, b), "ligand_a": a, "ligand_b": b, "repeats": []}
    complete = []
    for r in range(repeats):
        legs = {}
        for leg in ("complex", "solvent"):
            f = Path(edge_dir) / f"repeat_{r}" / leg / "result.json"
            legs[leg] = json.loads(f.read_text()) if f.exists() else None
        if not all(legs.values()) or any(legs[leg].get("delta_g") is None for leg in legs):
            out["repeats"].append({"repeat": r, "missing": [leg for leg in legs if not legs[leg]
                                                             or legs[leg].get("delta_g") is None]})
            continue
        no_error = [leg for leg in legs if legs[leg].get("delta_g_error") is None]
        if no_error:
            out["repeats"].append({"repeat": r, "no_standard_error": no_error})
            continue
        c, s = legs["complex"], legs["solvent"]
        # the state of the water moves of the complex leg, compared among all repeats below (review of PR #19)
        c = {**c, "grand_canonical_state": water_moves_of(c, Path(edge_dir) / f"repeat_{r}" / "complex")}
        # the two legs of the repeat are compared before they are subtracted (second review of PR #17)
        pair = [(f"complex leg of repeat {r}", c), (f"solvent leg of repeat {r}", s)]
        differ, _ = cycle_differences(pair)
        if differ:
            out["repeats"].append({"repeat": r, "inconsistent_legs": differ, "complex_dg": c["delta_g"] / KCAL,
                                   "solvent_dg": s["delta_g"] / KCAL})
            continue
        complete += pair
        ddg, se = edge_ddg(c["delta_g"], c["delta_g_error"], s["delta_g"], s["delta_g_error"])
        out["repeats"].append({
            "repeat": r, "ddg": ddg / KCAL, "se": se / KCAL, "complex_dg": c["delta_g"] / KCAL,
            "complex_se": c["delta_g_error"] / KCAL, "solvent_dg": s["delta_g"] / KCAL,
            "solvent_se": s["delta_g_error"] / KCAL,
            "windows": {leg: legs[leg]["n_states"] for leg in legs},
            "lowest_overlap": {leg: legs[leg]["quality"]["overlap"]["lowest"] for leg in legs},
            # the overlap among the samples MBAR estimates from, the other check behind NOT RELIABLE
            "lowest_overlap_used": {leg: (legs[leg]["quality"]["overlap"].get("used") or {}).get("lowest")
                                    for leg in legs},
            "not_reliable": [leg for leg in legs if not legs[leg]["quality"]["overlap"]["ok"]],
            "fewest_independent_samples": {leg: legs[leg]["quality"]["samples"].get("counted") for leg in legs},
            "backend": {leg: legs[leg].get("backend") for leg in legs},
            "warnings": [f"{leg}: {n}" for leg in legs for n in legs[leg].get("notes", [])]})
    inconsistent, not_compared = cycle_differences(complete)
    # the complex legs of all repeats in one grand canonical ensemble of water, whatever their order (second review
    # of PR #19: a first repeat of an earlier version let two later ones of other chemical potentials pass)
    moves = [(where, res["grand_canonical_state"]) for where, res in complete if "grand_canonical_state" in res]
    inconsistent += moves_differ(moves)
    inconsistent += [item for r in out["repeats"] for item in r.get("inconsistent_legs", [])
                     if item not in inconsistent]
    if inconsistent:
        out["inconsistent_legs"] = inconsistent
    if not_compared:
        out["not_compared"] = not_compared
    done = [r for r in out["repeats"] if "ddg" in r] if not inconsistent else []
    if done:
        values = [r["ddg"] for r in done]
        out["ddg"] = sum(values) / len(values)
        out["complex_dg"] = sum(r["complex_dg"] for r in done) / len(done)
        out["solvent_dg"] = sum(r["solvent_dg"] for r in done) / len(done)
        if len(done) > 1:
            mean = out["ddg"]
            out["ddg_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            out["se"] = out["ddg_std"] / math.sqrt(len(values))
        else:
            out["se"] = done[0]["se"]
        out["not_reliable"] = sorted({leg for r in done for leg in r["not_reliable"]})
        # what the table gives of the legs: over the repeats of the ddG (second review of PR #17: the first only)
        out["lowest_overlap"] = {leg: min(r["lowest_overlap"][leg] for r in done) for leg in ("complex", "solvent")}
        out["lowest_overlap_used"] = {leg: min((r["lowest_overlap_used"][leg] for r in done
                                                if r["lowest_overlap_used"][leg] is not None), default=None)
                                      for leg in ("complex", "solvent")}
        out["windows"] = {leg: sorted({r["windows"][leg] for r in done}) for leg in ("complex", "solvent")}
        out["backend"] = {leg: sorted({str(r["backend"][leg]) for r in done}) for leg in ("complex", "solvent")}
        # the conditions of the legs, which the edges of one network must share (combine_network)
        out["conditions"] = {key: complete[0][1].get(key) for key, _ in NETWORK_CONDITIONS
                             if key != "grand_canonical"}
        out["conditions"]["grand_canonical"] = merged_moves([state for _w, state in moves])
    return out


#: what the edges combined into one network must share, as their legs record it: the thermodynamic state (with the
#: grand canonical moves of water of the complex leg, :func:`water_moves_of`) and the treatment of the dispersion tail.
#: (The prepared systems differ from edge to edge; what they are made from besides the ligands is compared by
#: network_differences.)
NETWORK_CONDITIONS = (("temperature_K", "temperature (K)"), ("pressure_bar", "pressure (bar)"),
                      ("lrc_method", "correction for the dispersion tail"),
                      ("grand_canonical", "grand canonical moves of water of the complex leg"))


def water_moves_of(result: dict, leg_dir: Path | None = None) -> dict:
    """The state of the grand canonical moves of water of a leg: whether it makes them, and the excess chemical
    potential and the standard volume where they are known (review of PR #19: complex legs with other chemical
    potentials of water were combined into one network). The record of the leg (``run.grand_canonical``); for a
    result of an earlier version, the settings its samples recorded (``hrex``, ``conditions.settings.gcmc``), else
    whether it holds counts of moves (with or without, the two numbers not known; second review of PR #19)."""
    recorded = (result.get("run") or {}).get("grand_canonical")
    if recorded is not None:
        return recorded
    if leg_dir is not None:
        from cufep.rbfe.leg import water_moves_record
        from cufep.sampling.storage import stored_conditions

        _n, conditions = stored_conditions(Path(leg_dir) / "hrex")
        settings = (conditions or {}).get("settings")
        if isinstance(settings, dict):
            return water_moves_record(settings.get("gcmc"))
    return {"enabled": result.get("gcmc") is not None}


def moves_differ(states: list[tuple[str, dict]]) -> list[str]:
    """What differs among the states of the water moves of complex legs (``(where, state)``), whatever their order:
    with and without the moves, and the excess chemical potential or the standard volume among those that know them."""
    out = []
    if len({bool(s.get("enabled")) for _w, s in states}) > 1:
        out.append("grand canonical moves of water: " + ", ".join(
            f"{where} {'with' if s.get('enabled') else 'without'}" for where, s in states))
    for key, name in (("excess_chemical_potential_kcal_per_mol",
                       "grand canonical moves of water, excess chemical potential (kcal/mol)"),
                      ("standard_volume_nm3", "grand canonical moves of water, standard volume (nm^3)")):
        known = [(where, s[key]) for where, s in states if s.get(key) is not None]
        if len({round(float(v), 9) for _w, v in known}) > 1:
            out.append(f"{name}: " + ", ".join(f"{where} {v:g}" for where, v in known))
    return out


def merged_moves(states: list[dict]) -> dict:
    """One state of the water moves of legs that agree (:func:`moves_differ`): the numbers that any of them knows."""
    out: dict = {}
    for s in states:
        for key, value in s.items():
            if value is not None:
                out[key] = value
    return out


#: what the prepared systems of all edges of one network must have been made from alike (rbfe.prep.edge_made_from)
SHARED_PREPARATION = ("protein_sha256", "forcefield", "membrane", "solvation", "box_shape", "backend",
                      "thermodynamics", "dynamics")


def network_differences(out_dir: Path, results: list[dict]) -> list[str]:
    """What differs among the edges with a ddG that would be combined into one network: the conditions of their
    legs (:data:`NETWORK_CONDITIONS`), and what their prepared systems were made from besides the two ligands
    (:data:`SHARED_PREPARATION`: protein, force field, pH ...), and that a ligand of one name is one molecule in all
    of them (the digests of their records). Second review of PR #17: an edge run at 298 K and one at 310 K, or with
    another protein, gave one set of dG; third: two molecules under one name were one node."""
    from cufep.sampling.storage import _same

    out = []
    done = [r for r in results if "ddg" in r]
    # the water moves of the complex legs: compared as the repeats of an edge are (moves_differ)
    out += moves_differ([(r["edge"], (r.get("conditions") or {})["grand_canonical"]) for r in done
                         if (r.get("conditions") or {}).get("grand_canonical") is not None])
    for key, name in NETWORK_CONDITIONS:
        if key == "grand_canonical":
            continue
        values = [(r["edge"], (r.get("conditions") or {}).get(key)) for r in done]
        known = [(edge, value) for edge, value in values if value is not None]
        if any(not _same(value, known[0][1]) for _edge, value in known[1:]):
            out.append(f"{name}: " + ", ".join(f"{edge} {value}" for edge, value in known))
    made, molecules = [], {}
    for r in done:
        f = Path(out_dir) / "edges" / r["edge"] / "prep" / "prep.json"
        record = json.loads(f.read_text()).get("made_from") if f.exists() else None
        if record is not None:
            made.append((r["edge"], {key: record.get(key) for key in SHARED_PREPARATION}))
            # the molecule (digest of its record) that each end of the edge is (third review of PR #17)
            for end in ("ligand_a", "ligand_b"):
                molecules.setdefault(r[end], {}).setdefault(record.get(end), []).append(r["edge"])
    for name, digests in molecules.items():
        if len(digests) > 1:
            out.append(f"ligand {name} is another molecule in " + " than in ".join(
                ", ".join(edges) for edges in digests.values()))
    for key in SHARED_PREPARATION:
        if any(not _same(m[key], made[0][1][key]) for _edge, m in made[1:]):
            out.append(f"the prepared systems ({key}): " + ", ".join(
                edge for edge, m in made if not _same(m[key], made[0][1][key])) + f" differ from {made[0][0]}")
    return out


def combine_network(out_dir: Path, ligands: list[str], edges: list[tuple[str, str]], repeats: int,
                    experiment: dict[str, float], states: dict | None = None, temperature_K: float = 298.15) -> dict:
    """The results of the edges, the cycle closures and the dG of every ligand from the edges alone
    (:func:`cufep.rbfe.network.ligand_dg`); with ``states`` (:func:`states_of`) the dG of every compound from its
    states (:func:`cufep.rbfe.network.compound_dg`), which then set the constant of the dG of the states too."""
    results = [edge_result(out_dir / "edges" / edge_name(a, b), a, b, repeats) for a, b in edges]
    compound_of = (states or {}).get("compound") or {}
    for r in results:
        same = compound_of and compound_of.get(r["ligand_a"]) == compound_of.get(r["ligand_b"])
        # two states of one compound: experiment has no ddG between them (review of PR #19: +0.00 was given)
        if not same and experiment.get(r["ligand_a"]) is not None and experiment.get(r["ligand_b"]) is not None:
            r["exp_ddg"] = experiment[r["ligand_b"]] - experiment[r["ligand_a"]]
            # of states: that of their compounds, not of what the edge computes (second review of PR #19)
            r["exp_ddg_of"] = "the compounds" if compound_of else "the ligands"
            if compound_of:
                r["exp_compounds"] = [compound_of[r["ligand_a"]], compound_of[r["ligand_b"]]]
    differing = network_differences(out_dir, results)
    if differing:
        # each edge gives its ddG; no dG and no cycle closure is made from edges of other conditions
        return {"ligands": ligands, "edges": results, "experiment_kcal_mol": experiment, "cycle_closures": [],
                "dg": None, "inconsistent_edges": differing,
                "notes": ["NO NETWORK: the edges with a ddG were not made under the same conditions ("
                          + "; ".join(differing) + "); their ddG are given one by one, without cycle closures "
                          "and without dG"]}
    measured = [Edge(r["ligand_a"], r["ligand_b"], r["ddg"], r["se"]) for r in results if "ddg" in r]
    summary = {"ligands": ligands, "edges": results, "experiment_kcal_mol": experiment,
               "cycle_closures": cycle_closures(measured), "dg": None, "notes": []}
    import networkx as nx

    graph = nx.Graph([(e.ligand_a, e.ligand_b) for e in measured])
    graph.add_nodes_from(ligands)
    if not measured or not nx.is_connected(graph):
        summary["notes"].append("the edges with a ddG do not connect all ligands: no dG")
        return summary
    if states is None:
        dg = ligand_dg(ligands, measured, experiment)
        summary["dg"] = {"reference": dg["reference"], "reference_ligands": dg["reference_ligands"],
                         "values": {name: {"dg": v, "se": e} for name, (v, e) in dg["values"].items()}}
    else:
        from cufep.sampling.states import R_GAS

        by_compound = compound_dg(ligands, measured, states["compound"], compound_experiment(states, experiment),
                                  states["fraction"], states["symmetry"], R_GAS * float(temperature_K) / KCAL)
        summary["dg"] = {"reference": by_compound["reference"],
                         "reference_compounds": by_compound["reference_compounds"],
                         "values": by_compound["states"]}
        summary["compounds"] = {"temperature_K": float(temperature_K),
                                "experiment_kcal_mol": compound_experiment(states, experiment),
                                "values": by_compound["compounds"]}
        for c, left in (states.get("left_out") or {}).items():
            summary["notes"].append(f"compound {c}: its states {', '.join(left)} of system.states are not in the run; "
                                    "its dG combines the others")
        dg = {"reference": by_compound["reference"]}
    # the ddG that the whole network gives every edge (that of an edge without its own ddG too)
    inferred = network_ddg(ligands, measured, [(r["ligand_a"], r["ligand_b"]) for r in results])
    for r in results:
        r["network_ddg"], r["network_se"] = inferred[(r["ligand_a"], r["ligand_b"])]
    if dg["reference"] == "mean of the network":
        summary["notes"].append("no ligand has an experimental value: the dG are relative to the mean of the network")
    return summary


def write_results(out_dir: Path, summary: dict) -> None:
    def fmt(x) -> str:
        return "" if x is None else f"{x:.3f}"

    def edges_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["edge", "ligand_a", "ligand_b", "ddg_kcal_mol", "standard_error", "exp_ddg_kcal_mol",
                    "exp_ddg_of", "network_ddg_kcal_mol", "network_standard_error",
                    "complex_dg", "solvent_dg", "windows_complex", "windows_solvent", "lowest_overlap_complex",
                    "lowest_overlap_solvent", "lowest_overlap_used_complex", "lowest_overlap_used_solvent",
                    "not_reliable", "backend_complex", "backend_solvent"])
        def joined(values) -> str:
            return "/".join(str(v) for v in values or [])

        for e in summary["edges"]:
            windows, overlap, backend = e.get("windows") or {}, e.get("lowest_overlap") or {}, e.get("backend") or {}
            used = e.get("lowest_overlap_used") or {}
            w.writerow([e["edge"], e["ligand_a"], e["ligand_b"], fmt(e.get("ddg")), fmt(e.get("se")),
                        fmt(e.get("exp_ddg")), e.get("exp_ddg_of", "") if e.get("exp_ddg") is not None else "",
                        fmt(e.get("network_ddg")), fmt(e.get("network_se")),
                        fmt(e.get("complex_dg")), fmt(e.get("solvent_dg")),
                        joined(windows.get("complex")), joined(windows.get("solvent")),
                        fmt(overlap.get("complex")), fmt(overlap.get("solvent")), fmt(used.get("complex")),
                        fmt(used.get("solvent")), "; ".join(e.get("not_reliable", [])),
                        joined(backend.get("complex")), joined(backend.get("solvent"))])

    def ligands_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["ligand", "dg_kcal_mol", "standard_error", "exp_dg_kcal_mol", "reference"])
        dg = summary.get("dg") or {"values": {}, "reference": ""}
        for name in summary["ligands"]:
            v = dg["values"].get(name, {})
            w.writerow([name, fmt(v.get("dg")), fmt(v.get("se")), fmt(summary["experiment_kcal_mol"].get(name)),
                        dg["reference"]])

    def compounds_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["compound", "dg_kcal_mol", "standard_error", "exp_dg_kcal_mol", "symmetry_number", "states"])
        c = summary["compounds"]
        for name, v in c["values"].items():
            w.writerow([name, fmt(v["dg"]), fmt(v["se"]), fmt(c["experiment_kcal_mol"].get(name)), v["symmetry_number"],
                        "; ".join(f"{s} {weight:.3f}" for s, weight in v["weights"].items())])

    publish_with(out_dir / "edges.csv", edges_table, newline="")
    publish_with(out_dir / "ligands.csv", ligands_table, newline="")
    if summary.get("compounds"):
        publish_with(out_dir / "compounds.csv", compounds_table, newline="")
    else:
        (out_dir / "compounds.csv").unlink(missing_ok=True)     # never the compounds of an earlier state
    publish_text(out_dir / "summary.json", jsonio.dumps(summary, indent=1))


def result_lines(summary: dict) -> list[str]:
    lines = []
    for e in summary["edges"]:
        if "ddg" in e:
            exp = ""
            if e.get("exp_ddg") is not None:
                of = (f" of the compounds {' -> '.join(e['exp_compounds'])}"
                      if e.get("exp_ddg_of") == "the compounds" and e.get("exp_compounds") else "")
                exp = f" (experiment{of} {e['exp_ddg']:+.2f})"
            flag = f"; NOT RELIABLE ({', '.join(e['not_reliable'])} leg)" if e.get("not_reliable") else ""
            lines.append(f"RESULT {e['edge']}: ddG = {e['ddg']:+.2f} ± {e['se']:.2f} kcal/mol{exp}{flag}")
        elif e.get("inconsistent_legs"):
            lines.append(f"RESULT {e['edge']}: no ddG; its legs were not made under the same conditions "
                         f"({'; '.join(e['inconsistent_legs'])})")
        elif any(r.get("no_standard_error") for r in e["repeats"]):
            legs = sorted({leg for r in e["repeats"] for leg in r.get("no_standard_error", [])})
            lines.append(f"RESULT {e['edge']}: no ddG; the standard error of the {' and '.join(legs)} leg could not "
                         "be determined")
        else:
            lines.append(f"RESULT {e['edge']}: no ddG")
    for c in summary["cycle_closures"]:
        lines.append(f"cycle {' -> '.join(c['ligands'])}: sum of ddG {c['sum_kcal_mol']:+.2f} ± {c['se_kcal_mol']:.2f}")
    if summary.get("compounds"):
        c = summary["compounds"]
        for name, v in c["values"].items():
            exp = c["experiment_kcal_mol"].get(name)
            states = ", ".join(f"{s} {w:.2f}" for s, w in v["weights"].items())
            lines.append(f"dG {name}: {v['dg']:.2f} ± {v['se']:.2f} kcal/mol (relative to the "
                         f"{summary['dg']['reference']}; states {states}"
                         + (f"; symmetry number {v['symmetry_number']}" if v["symmetry_number"] > 1 else "") + ")"
                         + (f"; experiment {exp:.2f}" if exp is not None else ""))
    elif summary.get("dg"):
        for name, v in summary["dg"]["values"].items():
            exp = summary["experiment_kcal_mol"].get(name)
            lines.append(f"dG {name}: {v['dg']:.2f} ± {v['se']:.2f} kcal/mol (relative to the "
                         f"{summary['dg']['reference']})" + (f"; experiment {exp:.2f}" if exp is not None else ""))
    lines += [f"note: {n}" for n in summary["notes"]]
    return lines


def failed_jobs(results: list[dict]) -> list[str]:
    """The legs of this run that failed: an error, or no finite free energy (windows that the samples do not connect;
    second review of PR #17: such a leg ended the run with 0). What the exit code of `cufep rbfe run` says."""
    failed = []
    for r in results:
        tag = f"{r.get('edge', r.get('ligand'))}/r{r.get('repeat')}/{r.get('leg')}"
        if "error" in r:
            failed.append(f"{tag}: {r['error']}")
        elif r.get("delta_g") is None or not math.isfinite(float(r["delta_g"])):
            failed.append(f"{tag}: no free energy")
    return failed


def failed_edges(summary: dict) -> list[str]:
    return [e["edge"] for e in summary["edges"] if "ddg" not in e]


def analyse_directory(out_dir: Path) -> dict:
    """The results of a relative run from the results its legs have, as the run makes
    them; writes nothing. Of the configuration of the run only what combining the results needs is read (the
    number of repeats): the checks of a configuration are for a run that simulates, and a directory that an
    earlier version wrote under other checks is analysed all the same (review of PR #18)."""
    out_dir = Path(out_dir)
    stored = yaml.safe_load((out_dir / "resolved_config.yaml").read_text()) or {}
    repeats = (stored.get("sampling") or {}).get("repeats", RbfeSampling.model_fields["repeats"].default)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise WorkflowError(f"{out_dir / 'resolved_config.yaml'}: sampling.repeats is {repeats!r}, not a number of "
                            "repeats")
    network = json.loads((out_dir / "network.json").read_text())
    return combine_network(out_dir, network["ligands"], [tuple(e) for e in network["edges"]], repeats,
                           {k: float(v) for k, v in network.get("experiment_kcal_mol", {}).items()},
                           network.get("states"), float(network.get("temperature_K", 298.15)))
