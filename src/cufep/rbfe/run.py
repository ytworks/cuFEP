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

import numpy as np
import yaml

from cufep import jsonio
from cufep.files import publish_text, publish_with
from cufep.rbfe.config import RbfeConfig, RbfeSampling
from cufep.rbfe.network import Edge, cycle_closures, edge_ddg, ligand_dg
from cufep.rbfe.prep import edge_name
from cufep.workflow import KCAL, WATER_RULE, WorkflowError, _Log, select_ligands


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
        from cufep.rbfe.mapping import plan_network

        n = rcfg.network
        edges = plan_network(ligands, rcfg.mapping.settings(), n.distance_cutoff, n.max_path_length,
                             n.require_cycle_covering)
        how = "planned by LOMAP"
    import networkx as nx

    graph = nx.Graph(edges)
    graph.add_nodes_from(names)
    if not nx.is_connected(graph):
        raise WorkflowError("the edges do not connect all ligands of the run: "
                            + "; ".join(", ".join(sorted(c)) for c in nx.connected_components(graph)))
    log(f"{len(edges)} edge(s) ({how}): " + ", ".join(f"{a} -> {b}" for a, b in edges))
    return edges, how


def mapped_edges(rcfg: RbfeConfig, ligands, edges) -> list[dict]:
    """The atom mapping of every edge as the preparation makes it (Kartograf with the settings of the run), its
    counts and its LOMAP score: what `cufep rbfe plan` and `cufep benchmark plan` show before anything runs."""
    from cufep.rbfe.mapping import kartograf_mapping, lomap_score

    by_name = {lig.name: lig for lig in ligands}
    out = []
    for a, b in edges:
        mapping = kartograf_mapping(by_name[a], by_name[b], rcfg.mapping.settings())
        out.append({"ligand_a": a, "ligand_b": b, "mapped": len(mapping),
                    "unique_a": by_name[a].n_atoms - len(mapping), "unique_b": by_name[b].n_atoms - len(mapping),
                    "lomap_score": lomap_score(by_name[a], by_name[b], mapping)})
    return out


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


def check_pre_equilibrations(rcfg: RbfeConfig, out_dir: Path, edges, infos: dict, only_legs) -> None:
    """Before a run moves or writes anything: the pre-equilibration that a leg it runs has is one this run can use
    (:func:`cufep.workflow.check_equil_file`, the check the leg makes when it begins). Review of PR #21: a
    directory of a version before the grand canonical moves of water, run again with the same YAML, was refused by
    its legs one by one, after the run had replaced ``resolved_config.yaml`` and every leg had set its result
    aside, and the edges that had a ddG had none. The refusal of such a directory says how it is continued as it
    was made."""
    from cufep.workflow import check_equil_file, gcmc_in, leg_seed

    for a, b in edges:
        info = infos.get(edge_name(a, b))
        if info is None:
            continue                                        # not prepared yet: it has no legs
        for r in range(rcfg.sampling.repeats):
            for leg in ("solvent", "complex"):
                equil_file = out_dir / "edges" / edge_name(a, b) / f"repeat_{r}" / leg / "equil.npz"
                if (only_legs and leg not in only_legs) or not equil_file.exists():
                    continue
                with np.load(equil_file) as eq:
                    stored = json.loads(str(eq["made_with"])) if "made_with" in eq.files else None
                    in_membrane = leg == "complex" and info["legs"][leg].get("barostat") == "membrane"
                    try:
                        check_equil_file(eq, equil_file, rcfg, leg, in_membrane, leg_seed(rcfg, r, leg),
                                         lambda _message: None)
                    except WorkflowError as exc:
                        if stored is not None and "gcmc" not in stored and gcmc_in(rcfg, leg):
                            raise WorkflowError(
                                f"{exc} This leg was made without grand canonical moves of water, which a relative "
                                "calculation makes in its complex leg by default: with sampling.gcmc.enabled: false "
                                "the calculation is continued as it was made.") from exc
                        raise


#: how many legs are checked at the same time before a run (:func:`check_legs_reuse`): the hybrid System of a leg
#: with samples is built for its checksum, which takes seconds for a complex leg (tens in a bilayer)
CHECK_WORKERS = 8


def check_legs_reuse(rcfg: RbfeConfig, out_dir: Path, edges, infos: dict, only_legs, log=print) -> None:
    """Before a run moves or writes anything: the lambda windows and the samples of every leg it runs are ones it
    can go on from (:func:`cufep.rbfe.leg.check_leg_reuse`: the settings, and the checksum of the System of this
    run). The legs that have windows are checked at the same time, in processes of their own; the first refusal,
    in the order of the legs, stops the run."""
    from cufep.rbfe.leg import leg_reuse_problem

    todo = [(str(out_dir / "edges" / edge_name(a, b)), leg, r)
            for a, b in edges if infos.get(edge_name(a, b)) is not None
            for r in range(rcfg.sampling.repeats) for leg in ("solvent", "complex")
            if (not only_legs or leg in only_legs)
            and (out_dir / "edges" / edge_name(a, b) / f"repeat_{r}" / leg / "schedule.yaml").exists()]
    if not todo:
        return
    config = rcfg.model_dump(mode="json")
    workers = min(len(todo), CHECK_WORKERS)
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        log(f"checking the lambda windows and samples that {len(todo)} leg(s) have before the run")
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            problems = list(pool.map(leg_reuse_problem, [config] * len(todo), *zip(*todo)))
    else:
        problems = [leg_reuse_problem(config, *item) for item in todo]
    for problem in problems:
        if problem:
            raise WorkflowError(problem)


def read_leg_result(file: Path) -> dict | None:
    """The result of a leg (``result.json``), or None for a leg without one. Raises WorkflowError, with the name
    of the file, for one that cannot be read (truncated, or not the record of a leg): the results of a directory
    are combined from every leg, and a file that fails there fails for all of them."""
    file = Path(file)
    if not file.exists():
        return None
    try:
        result = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"{file} cannot be read ({type(exc).__name__}: {exc}); it is damaged. Put the file "
                            "back, or run the leg again") from exc
    if not isinstance(result, dict):
        raise WorkflowError(f"{file} is not the result of a leg; it is damaged. Put the file back, or run the leg "
                            "again")
    return result


def read_summary(stored: Path) -> dict | None:
    """The summary a directory has, or None: there is none, or it cannot be read."""
    try:
        summary = json.loads(Path(stored).read_text())
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, dict) and isinstance(summary.get("edges"), list) else None


def stored_edges(summary: dict | None) -> list[dict]:
    """The edges of the summary a directory has (:func:`read_summary`) that were made from the results of their
    legs (:func:`made_from_results`)."""
    return [e for e in (summary or {}).get("edges", [])
            if isinstance(e, dict) and "edge" in e and made_from_results(e)]


#: what the grand canonical moves of water of a complex leg are named by where their record is gone
WATER_RECORD = "(the record of its samples, from which the summary has the grand canonical moves of water of the leg)"


def water_moves_lost(held, read) -> bool:
    """Whether what a summary holds of the grand canonical moves of water of a complex leg (``held``: a value of
    the water, the rule of the mover) is not given any more by what is read of the leg now (``read``)."""
    return isinstance(held, dict) and isinstance(read, dict) and any(
        held.get(key) is not None and read.get(key) is None for key in (*WATER_VALUES, WATER_RULE))


def leg_being_run(leg_dir: Path) -> bool:
    """Whether a run began the leg and did not finish it: its marker of a started leg is there (the run was
    stopped), or the error of a leg that failed. Such a leg has no result, which is an ordinary state of a
    directory; ``result.previous.json`` does not say it, since it stays when the leg has finished again (sixth
    review of PR #21)."""
    from cufep.workflow import STARTED_FILE

    return (Path(leg_dir) / STARTED_FILE).exists() or (Path(leg_dir) / "error.json").exists()


def check_results_combined(out_dir: Path, edges, selected, only_legs, repeats: int) -> None:
    """Before a run writes anything: the results it will combine at its end without sampling them must still be
    what the directory made them from. The run writes a summary of all edges when it ends, as `cufep rbfe analyze
    --write` does, and what is gone is not compared there: an edge whose repeats the summary refused a ddG for
    their other values of the water is combined once the record of the samples of one repeat, or the result of
    one, is moved away (fifth review of PR #21, found for --write; the same state through the run). Refused:

    - an edge with the result of a leg and without the record of its prepared systems (``prep/prep.json``): the
      prepared edges of a network are compared by that record, and an edge without it is left out;
    - a leg result that the summary of the directory read and that is gone, for a leg this run does not sample;
      not where a run began the leg and did not finish it (:func:`leg_being_run`), which is an ordinary state of
      a directory;
    - what the summary holds of the grand canonical moves of water of a complex leg this run does not sample and
      the result of that leg, with the record of its samples, gives no more (:func:`water_moves_lost`).

    Each leg is held to the summary by itself, whatever the other leg of its repeat has now, and only the repeats
    that this run combines (those of its ``sampling.repeats``). Sixth review of PR #21: with --leg solvent, a
    complex leg whose record was gone was not looked at while the solvent leg of its repeat had no result, and
    was combined when that leg had finished; a result moved away passed for the ``result.previous.json`` of an
    earlier, finished run of the leg; a repeat beyond those of the configuration was asked for. Seventh: a result
    that cannot be read (a truncated file) was looked at only for a complex leg of a repeat the summary read; the
    run sampled the other edge and failed when it combined the results. Where the summary
    cannot be read, the ddG that ``edges.csv`` holds of the edges this run does not run must be those their legs
    give (:func:`tables_changed`). Raises WorkflowError."""
    out_dir = Path(out_dir)
    kept = read_summary(out_dir / "summary.json")
    had = {e["edge"]: e for e in stored_edges(kept)}
    chosen = {edge_name(a, b) for a, b in selected}
    if kept is None:
        # no summary that can be read: of the edges this run does not run, the tables are the record (the run
        # writes them again when it ends, with what the legs of those edges give then)
        changed = tables_changed(out_dir, [edge_result(out_dir / "edges" / edge_name(a, b), a, b, repeats)
                                           for a, b in edges if edge_name(a, b) not in chosen], ligands=False)
        if changed:
            shown = "; ".join(changed[:3]) + (" ..." if len(changed) > 3 else "")
            raise WorkflowError(
                f"{out_dir / 'summary.json'} cannot be read or is not there, and edges.csv holds {len(changed)} "
                f"value(s) of edges this run does not run that the results of their legs do not give ({shown}): "
                "the run would write them again from what is there. Nothing was changed. Put the files back, run "
                "those edges too, or run the calculation in a new output.dir")

    def sampled(name: str, leg: str) -> bool:
        return name in chosen and (not only_legs or leg in only_legs)

    gone = []
    for a, b in edges:
        name = edge_name(a, b)
        edge_dir, files = out_dir / "edges" / name, []
        if not (edge_dir / "prep" / "prep.json").exists() and list(edge_dir.glob("repeat_*/*/result.json")):
            files.append("prep/prep.json")
        held = had.get(name) or {}
        read = set(repeats_read(held))
        entries = {entry.get("repeat"): entry for entry in held.get("repeats") or [] if entry.get("repeat") in read}
        for repeat in range(repeats):                             # the repeats this run combines
            entry = entries.get(repeat)                           # what the summary read of the repeat, if it did
            for leg in ("complex", "solvent"):
                leg_dir = edge_dir / f"repeat_{repeat}" / leg
                if sampled(name, leg):
                    continue
                if not (leg_dir / "result.json").exists():
                    if entry is not None and not leg_being_run(leg_dir):
                        files.append(f"repeat_{repeat}/{leg}/result.json")
                    continue
                # every result the run will combine must be one it can read then, whatever the summary holds of it
                try:
                    result = read_leg_result(leg_dir / "result.json")
                except WorkflowError:
                    files.append(f"repeat_{repeat}/{leg}/result.json (it cannot be read)")
                    continue
                if leg == "complex" and entry is not None and water_moves_lost(
                        entry.get("water_moves"), water_moves_of_leg(result, leg_dir)):
                    files.append(f"repeat_{repeat}/complex/hrex {WATER_RECORD}")
        if files:
            gone.append((name, files))
    if gone:
        shown = "; ".join(f"{edge}: {', '.join(files)}" for edge, files in gone[:3]) + (" ..." if len(gone) > 3 else "")
        raise WorkflowError(
            f"{out_dir} holds the results of {len(gone)} edge(s) which this run would combine without sampling them, "
            f"made from files that are not in the directory or cannot be read ({shown}): combined again, they would "
            "not be compared as they were. Nothing was changed. Put the files back, or run the calculation in a "
            "new output.dir")


def run_rbfe(rcfg: RbfeConfig, reschedule: bool = False, log=None, only_edges=None, only_legs=None,
             experiment: dict[str, float] | None = None) -> dict:
    """Run the whole relative calculation. Safe to call again to resume. ``experiment`` (kcal/mol by ligand) takes
    the place of ``system.experiment`` and the tags of the SDF: a benchmark gives the values of its table."""
    from cufep.run import lock_output_directory

    out_dir = Path(rcfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_output_directory(out_dir)
    try:
        return _run(rcfg, out_dir, reschedule, log or _Log(out_dir / "cufep.log"), only_edges, only_legs, experiment)
    finally:
        lock.close()


def _run(rcfg: RbfeConfig, out_dir: Path, reschedule: bool, log, only_edges=None, only_legs=None,
         experiment: dict[str, float] | None = None) -> dict:
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
    by_name = {lig.name: lig for lig in ligands}
    box_shape = rcfg.solvation.box_shape
    # every prepared edge of the network, before the run writes anything: those it runs, and those whose results it
    # combines (second review of PR #17: an edge of another protein was combined); a refused run keeps the record
    # of the run whose results the directory holds (review of PR #17)
    prepared = {edge_name(a, b): prepared_edge(rcfg, by_name[a], by_name[b], out_dir / "edges" / edge_name(a, b),
                                               box_shape) for a, b in edges}
    check_pre_equilibrations(rcfg, out_dir, selected, prepared, only_legs)
    if not reschedule:
        # (with --reschedule the windows and samples of the legs are set aside: they start anew)
        check_legs_have_their_windows(out_dir, selected, rcfg.sampling.repeats, only_legs)
        check_legs_reuse(rcfg, out_dir, selected, prepared, only_legs, log)
    check_results_combined(out_dir, edges, selected, only_legs, rcfg.sampling.repeats)
    experiment = (experiment_of(rcfg, ligands) if experiment is None
                  else {lig.name: float(experiment[lig.name]) for lig in ligands if lig.name in experiment})
    write_run_files(out_dir, rcfg, [lig.name for lig in ligands], edges, how, experiment)
    publish_text(out_dir / "auto_report.md", report.to_markdown())
    protein.write_pdb(out_dir / "protein_prepared.pdb")
    infos = {}
    for a, b in selected:
        infos[edge_name(a, b)] = prepare_edge(rcfg, by_name[a], by_name[b], protein,
                                              out_dir / "edges" / edge_name(a, b), box_shape, log)
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
    summary = combine_network(out_dir, [lig.name for lig in ligands], edges, repeats, experiment)
    # what the exit code of `cufep rbfe run` says: whether a job of this run failed (review of PR #17)
    summary["failed_in_this_run"] = failed_now
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


#: the two numbers of the water that grand canonical moves are made with
WATER_VALUES = ("excess_chemical_potential_kcal_per_mol", "standard_volume_nm3")


def water_moves_of_leg(result: dict, leg_dir: Path) -> dict:
    """The grand canonical moves of water a complex leg was sampled with (:func:`cufep.workflow.water_moves_of`):
    as its result records them; for a result that does not (a version before the record), as its samples do. A
    result of a version before a relative leg could make the moves (it has no ``gcmc`` at all) was made without
    them. A result with the counts of the moves whose samples are not there was made with them, and with which
    values of the water is not recorded: ``{"enabled": True}`` alone. Whether the moves were made is so known
    for every leg, apart from the values (second review of PR #21: such a leg was not compared at all and went
    into one network with legs without the moves). The rule of the mover for the velocities of an inserted water
    (:data:`cufep.workflow.WATER_RULE`) is that of the result, or for a result of a version before it was recorded
    there, that of its samples; None (or no such key) where neither records one (third review of PR #21: the rule
    did not reach the result of a leg, and the results did not tell the edges of the earlier mover)."""
    from cufep.sampling.storage import recorded_conditions
    from cufep.workflow import water_moves_of

    recorded = result.get("water_moves")
    if isinstance(recorded, dict) and (not recorded.get("enabled") or WATER_RULE in recorded):
        return recorded
    if not isinstance(recorded, dict) and ("gcmc" not in result or result["gcmc"] is None):
        return {"enabled": False}
    # from the record of the samples alone: a result is combined also where the parts of its samples were set
    # aside, which a reader of the samples refuses (fourth review of PR #21)
    of_samples = water_moves_of(recorded_conditions(Path(leg_dir) / "hrex"))
    if isinstance(recorded, dict):
        # a result of a version that recorded the values of the water and not yet the rule of the mover
        return {**recorded, WATER_RULE: (of_samples or {}).get(WATER_RULE)}
    return of_samples if of_samples is not None else {"enabled": True}


def water_values_recorded(moves: dict) -> bool:
    """Whether the values of the water of the moves are known: of a leg without the moves there are none to know."""
    return not moves.get("enabled") or all(moves.get(key) is not None for key in WATER_VALUES)


def water_moves_differ(a: dict, b: dict) -> bool:
    """Whether two complex legs were sampled with other grand canonical moves of water, as far as it is known:
    one with and one without them, or with another value of the water where both record it."""
    from cufep.sampling.storage import _same

    if bool(a.get("enabled")) != bool(b.get("enabled")):
        return True
    return any(a.get(key) is not None and b.get(key) is not None and not _same(a[key], b[key])
               for key in WATER_VALUES)


def any_water_moves_differ(moves: list[dict]) -> bool:
    return any(water_moves_differ(a, b) for i, a in enumerate(moves) for b in moves[i + 1:])


def water_moves_text(moves: dict) -> str:
    if not moves.get("enabled"):
        return "none"
    if not water_values_recorded(moves):
        return "made, the values of the water not recorded"
    return (f"excess chemical potential {moves['excess_chemical_potential_kcal_per_mol']:g} kcal/mol, standard "
            f"volume {moves['standard_volume_nm3']:g} nm3")


#: how the grand canonical moves of water of the complex legs are named where they differ
WATER_MOVES = "grand canonical moves of water in the complex leg"


def edge_result(edge_dir: Path, a: str, b: str, repeats: int) -> dict:
    """ddG of an edge from the results of its legs (kcal/mol); repeats are averaged, their standard deviation
    given (as the ABFE combines repeats).

    A repeat counts only with both legs and the standard error of both (a leg whose error could not be determined
    is named under ``no_standard_error``). The legs of all repeats that count must have been made under the same
    conditions (:func:`cufep.workflow.cycle_differences`, the check of the ABFE: temperature, pressure, prepared
    systems ...); if they were not, the edge gets no ddG and says what differs (``inconsistent_legs``) (review of
    PR #17: legs at 298 K and 310 K were subtracted). The free energies of the legs are the means over the same
    repeats as the ddG, which is their difference. The complex legs of the repeats must also have been sampled with
    the same grand canonical moves of water (:func:`water_moves_of_leg`: with or without them, the same chemical
    potential and standard volume of water), or the edge has no ddG (review of PR #21)."""
    from cufep.workflow import cycle_differences

    out = {"edge": edge_name(a, b), "ligand_a": a, "ligand_b": b, "repeats": []}
    prep = Path(edge_dir) / "prep" / "prep.json"
    if prep.exists():
        # the LOMAP score of the mapping the edge was prepared with: in the summary, so that a report made again from
        # the summary alone has it (review of PR #21)
        score = json.loads(prep.read_text()).get("lomap_score")
        out["lomap_score"] = None if score is None else float(score)
    complete = []
    for r in range(repeats):
        legs = {}
        for leg in ("complex", "solvent"):
            legs[leg] = read_leg_result(Path(edge_dir) / f"repeat_{r}" / leg / "result.json")
        if not all(legs.values()) or any(legs[leg].get("delta_g") is None for leg in legs):
            out["repeats"].append({"repeat": r, "missing": [leg for leg in legs if not legs[leg]
                                                             or legs[leg].get("delta_g") is None]})
            continue
        no_error = [leg for leg in legs if legs[leg].get("delta_g_error") is None]
        if no_error:
            out["repeats"].append({"repeat": r, "no_standard_error": no_error})
            continue
        c, s = legs["complex"], legs["solvent"]
        # the two legs of the repeat are compared before they are subtracted (second review of PR #17)
        pair = [(f"{leg} leg of repeat {r}", legs[leg]) for leg in legs]
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
            # the legs with a window below the level of independent samples of the analysis (Mey et al. 2020)
            "few_samples": [leg for leg in legs if legs[leg]["quality"]["samples"].get("ok") is False],
            "water_moves": water_moves_of_leg(c, Path(edge_dir) / f"repeat_{r}" / "complex"),
            "backend": {leg: legs[leg].get("backend") for leg in legs},
            "warnings": [f"{leg}: {n}" for leg in legs for n in legs[leg].get("notes", [])]})
    inconsistent, not_compared = cycle_differences(complete)
    inconsistent += [item for r in out["repeats"] for item in r.get("inconsistent_legs", [])
                     if item not in inconsistent]
    moves = [(r["repeat"], r["water_moves"]) for r in out["repeats"] if "ddg" in r]
    if any_water_moves_differ([value for _repeat, value in moves]):
        inconsistent.append(f"{WATER_MOVES}: " + ", ".join(f"repeat {repeat} {water_moves_text(value)}"
                                                         for repeat, value in moves))
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
        out["few_samples"] = sorted({leg for r in done for leg in r["few_samples"]})
        out["fewest_independent_samples"] = {
            leg: min((r["fewest_independent_samples"][leg] for r in done
                      if r["fewest_independent_samples"][leg] is not None), default=None)
            for leg in ("complex", "solvent")}
        # what the table gives of the legs: over the repeats of the ddG (second review of PR #17: the first only)
        out["lowest_overlap"] = {leg: min(r["lowest_overlap"][leg] for r in done) for leg in ("complex", "solvent")}
        out["lowest_overlap_used"] = {leg: min((r["lowest_overlap_used"][leg] for r in done
                                                if r["lowest_overlap_used"][leg] is not None), default=None)
                                      for leg in ("complex", "solvent")}
        out["windows"] = {leg: sorted({r["windows"][leg] for r in done}) for leg in ("complex", "solvent")}
        out["backend"] = {leg: sorted({str(r["backend"][leg]) for r in done}) for leg in ("complex", "solvent")}
        # the conditions of the legs, which the edges of one network must share (combine_network)
        out["conditions"] = {key: complete[0][1].get(key) for key, _ in NETWORK_CONDITIONS}
        # of the complex legs alone (the solvent leg never makes the moves): what the repeats of the ddG agree in,
        # with the values of the water from a repeat that records them; and the repeats that do not, which stay
        # named whatever the others record
        merged = {"enabled": bool(moves[0][1].get("enabled"))}
        for key in WATER_VALUES:
            recorded = [value[key] for _repeat, value in moves if value.get(key) is not None]
            if recorded and merged["enabled"]:
                merged[key] = recorded[0]
        if merged["enabled"]:
            # the rule of the mover for the velocities of an inserted water, where all repeats of the ddG record one
            # rule (None: none recorded, or not one rule; the repeats say which, and those without one are named)
            rules = {value.get(WATER_RULE) for _repeat, value in moves}
            merged[WATER_RULE] = rules.pop() if len(rules) == 1 else None
        out["conditions"]["water_moves"] = merged
        out["water_values_not_recorded"] = [repeat for repeat, value in moves
                                            if not water_values_recorded(value)]
        out["water_rule_not_recorded"] = [repeat for repeat, value in moves
                                          if value.get("enabled") and value.get(WATER_RULE) is None]
    return out


def water_rules(results: list[dict]) -> dict[str | None, list[str]]:
    """The rules of the mover for the velocities of an inserted water under which the complex legs of the edges
    with a ddG made their grand canonical moves (None: a leg that records none), each with its edges; an edge
    whose repeats are not of one rule is named with the repeats of each. Edges without the moves have none."""
    out: dict[str | None, list[str]] = {}
    for r in results:
        by_rule: dict[str | None, list[int]] = {}
        for repeat in r["repeats"] if "ddg" in r else []:
            moves = repeat.get("water_moves") if "ddg" in repeat else None
            if isinstance(moves, dict) and moves.get("enabled"):
                by_rule.setdefault(moves.get(WATER_RULE), []).append(repeat["repeat"])
        for rule, repeats in by_rule.items():
            out.setdefault(rule, []).append(r["edge"] if len(by_rule) == 1 else
                                            f"{r['edge']} (repeat {', '.join(str(n) for n in repeats)})")
    return out


#: what the edges combined into one network must share, as their legs record it: the thermodynamic state and the
#: treatment of the dispersion tail. (The prepared systems differ from edge to edge; what they are made from besides
#: the ligands is compared by network_differences.)
NETWORK_CONDITIONS = (("temperature_K", "temperature (K)"), ("pressure_bar", "pressure (bar)"),
                      ("lrc_method", "correction for the dispersion tail"))
#: what the prepared systems of all edges of one network must have been made from alike (rbfe.prep.edge_made_from)
SHARED_PREPARATION = ("protein_sha256", "forcefield", "membrane", "solvation", "box_shape", "backend",
                      "thermodynamics", "dynamics")


def network_differences(out_dir: Path, results: list[dict]) -> list[str]:
    """What differs among the edges with a ddG that would be combined into one network: the conditions of their
    legs (:data:`NETWORK_CONDITIONS`), and what their prepared systems were made from besides the two ligands
    (:data:`SHARED_PREPARATION`: protein, force field, pH ...), and that a ligand of one name is one molecule in all
    of them (the digests of their records). Second review of PR #17: an edge run at 298 K and one at 310 K, or with
    another protein, gave one set of dG; third: two molecules under one name were one node. Review of PR #21: so
    did edges whose complex legs were sampled with other grand canonical moves of water (another chemical potential
    after a change of the YAML between two runs of --edge, or with and without the moves)."""
    from cufep.sampling.storage import _same

    out = []
    done = [r for r in results if "ddg" in r]
    for key, name in NETWORK_CONDITIONS:
        values = [(r["edge"], (r.get("conditions") or {}).get(key)) for r in done]
        known = [(edge, value) for edge, value in values if value is not None]
        if any(not _same(value, known[0][1]) for _edge, value in known[1:]):
            out.append(f"{name}: " + ", ".join(f"{edge} {value}" for edge, value in known))
    # the complex legs: with or without the moves is known of every edge, the values of the water of most
    moves = [(r["edge"], (r.get("conditions") or {}).get("water_moves")) for r in done]
    moves = [(edge, value) for edge, value in moves if isinstance(value, dict)]
    if any_water_moves_differ([value for _edge, value in moves]):
        out.append(f"{WATER_MOVES}: " + ", ".join(f"{edge} {water_moves_text(value)}" for edge, value in moves))
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
                    experiment: dict[str, float]) -> dict:
    """The results of the edges, the cycle closures and the dG of every ligand from the edges alone
    (:func:`cufep.rbfe.network.ligand_dg`)."""
    results = [edge_result(out_dir / "edges" / edge_name(a, b), a, b, repeats) for a, b in edges]
    for r in results:
        if experiment.get(r["ligand_a"]) is not None and experiment.get(r["ligand_b"]) is not None:
            r["exp_ddg"] = experiment[r["ligand_b"]] - experiment[r["ligand_a"]]
    # what could not be compared is said whatever comes of the comparison
    unknown = [f"{r['edge']} (repeat {', '.join(str(n) for n in r['water_values_not_recorded'])})"
               for r in results if "ddg" in r and r.get("water_values_not_recorded")]
    water_notes = [] if not unknown else [
        f"{WATER_MOVES}: made, and the excess chemical potential and the standard volume of the water not "
        f"recorded, for {', '.join(unknown)} (results of an earlier version whose samples are not there); those "
        "values were not compared for them"]
    # edges whose movers were not of one rule, or of which that cannot be told, give their ddG to one network, and
    # which are of which is said (third review of PR #21: an edge finished by the mover before 2026-11-28 and one
    # added by this version gave one set of dG, and nothing told them apart); so is a network none of whose edges
    # records a rule
    rules = water_rules(results)
    # (what is known of a leg without a rule: nothing but that; said with what the movers did until the rule)
    earlier = "(until 2026-11-28 the mover left those velocities as they were)"
    known = {rule: edges for rule, edges in rules.items() if rule is not None}
    unrecorded = rules.get(None, [])
    if len(known) > 1:
        # rules that are known to differ (fourth review of PR #21: an edge without a recorded rule was said to
        # be of another rule; it is one of which the rule cannot be told)
        water_notes.append(
            f"{WATER_MOVES}: the edges with a ddG were not sampled under one rule for the velocities of an inserted "
            "water: " + "; ".join(f"{rule} for {', '.join(edges)}" for rule, edges in known.items())
            + (f"; no rule recorded {earlier} for {', '.join(unrecorded)}" if unrecorded else ""))
    elif known and unrecorded:
        ((rule, edges),) = known.items()
        water_notes.append(
            f"{WATER_MOVES}: no rule for the velocities of an inserted water is recorded for "
            f"{', '.join(unrecorded)} {earlier}; whether every edge with a ddG was sampled under one rule cannot be "
            f"told ({rule} for {', '.join(edges)})")
    elif unrecorded:
        water_notes.append(
            f"{WATER_MOVES}: no rule for the velocities of an inserted water is recorded for any of the "
            f"{len(unrecorded)} edge(s) with a ddG whose complex legs made the moves {earlier}")
    differing = network_differences(out_dir, results)
    if differing:
        # each edge gives its ddG; no dG and no cycle closure is made from edges of other conditions
        return {"ligands": ligands, "edges": results, "experiment_kcal_mol": experiment, "cycle_closures": [],
                "dg": None, "inconsistent_edges": differing,
                "notes": ["NO NETWORK: the edges with a ddG were not made under the same conditions ("
                          + "; ".join(differing) + "); their ddG are given one by one, without cycle closures "
                          "and without dG", *water_notes]}
    measured = [Edge(r["ligand_a"], r["ligand_b"], r["ddg"], r["se"]) for r in results if "ddg" in r]
    summary = {"ligands": ligands, "edges": results, "experiment_kcal_mol": experiment,
               "cycle_closures": cycle_closures(measured), "dg": None, "notes": list(water_notes)}
    import networkx as nx

    graph = nx.Graph([(e.ligand_a, e.ligand_b) for e in measured])
    graph.add_nodes_from(ligands)
    if not measured or not nx.is_connected(graph):
        summary["notes"].append("the edges with a ddG do not connect all ligands: no dG")
        return summary
    dg = ligand_dg(ligands, measured, experiment)
    summary["dg"] = {"reference": dg["reference"], "reference_ligands": dg["reference_ligands"],
                     "values": {name: {"dg": v, "se": e} for name, (v, e) in dg["values"].items()}}
    if dg["reference"] == "mean of the network":
        summary["notes"].append("no ligand has an experimental value: the dG are relative to the mean of the network")
    return summary


def write_results(out_dir: Path, summary: dict) -> None:
    fmt = table_number                 # (the one rounding of the tables, which a table read again is held to)

    def edges_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["edge", "ligand_a", "ligand_b", "ddg_kcal_mol", "standard_error", "exp_ddg_kcal_mol",
                    "complex_dg", "solvent_dg", "windows_complex", "windows_solvent", "lowest_overlap_complex",
                    "lowest_overlap_solvent", "lowest_overlap_used_complex", "lowest_overlap_used_solvent",
                    "not_reliable", "backend_complex", "backend_solvent", "fewest_independent_samples_complex",
                    "fewest_independent_samples_solvent", "few_independent_samples", "lomap_score"])
        def joined(values) -> str:
            return "/".join(str(v) for v in values or [])

        for e in summary["edges"]:
            windows, overlap, backend = e.get("windows") or {}, e.get("lowest_overlap") or {}, e.get("backend") or {}
            used = e.get("lowest_overlap_used") or {}
            fewest = e.get("fewest_independent_samples") or {}
            w.writerow([e["edge"], e["ligand_a"], e["ligand_b"], fmt(e.get("ddg")), fmt(e.get("se")),
                        fmt(e.get("exp_ddg")), fmt(e.get("complex_dg")), fmt(e.get("solvent_dg")),
                        joined(windows.get("complex")), joined(windows.get("solvent")),
                        fmt(overlap.get("complex")), fmt(overlap.get("solvent")), fmt(used.get("complex")),
                        fmt(used.get("solvent")), "; ".join(e.get("not_reliable", [])),
                        joined(backend.get("complex")), joined(backend.get("solvent")),
                        fmt(fewest.get("complex")), fmt(fewest.get("solvent")), "; ".join(e.get("few_samples", [])),
                        fmt(e.get("lomap_score"))])

    def ligands_table(fh) -> None:
        w = csv.writer(fh)
        w.writerow(["ligand", "dg_kcal_mol", "standard_error", "exp_dg_kcal_mol", "reference"])
        dg = summary.get("dg") or {"values": {}, "reference": ""}
        for name in summary["ligands"]:
            v = dg["values"].get(name, {})
            w.writerow([name, fmt(v.get("dg")), fmt(v.get("se")), fmt(summary["experiment_kcal_mol"].get(name)),
                        dg["reference"]])

    publish_with(out_dir / "edges.csv", edges_table, newline="")
    publish_with(out_dir / "ligands.csv", ligands_table, newline="")
    publish_text(out_dir / "summary.json", jsonio.dumps(summary, indent=1))


def result_lines(summary: dict) -> list[str]:
    lines = []
    for e in summary["edges"]:
        if "ddg" in e:
            exp = f" (experiment {e['exp_ddg']:+.2f})" if e.get("exp_ddg") is not None else ""
            flag = f"; NOT RELIABLE ({', '.join(e['not_reliable'])} leg)" if e.get("not_reliable") else ""
            if e.get("few_samples"):
                flag += f"; few independent samples ({', '.join(e['few_samples'])} leg)"
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
    if summary.get("dg"):
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


def repeats_read(edge: dict) -> list[int]:
    """The repeats of an edge of a summary of which the results of both legs were read: those with a ddG of their
    own, and those whose legs gave none together (made under other conditions, or without a standard error).
    What the summary says of the edge was made from them, whether the edge has a ddG or was refused one (the
    repeats of an edge that were sampled with other moves of water each have their ddG, and the edge none)."""
    return [r["repeat"] for r in edge.get("repeats") or []
            if "repeat" in r and any(key in r for key in ("ddg", "inconsistent_legs", "no_standard_error"))]


def made_from_results(edge: dict) -> bool:
    """Whether a summary holds something of an edge that was made from the results of its legs: a ddG, or
    repeats that were read (:func:`repeats_read`)."""
    return edge.get("ddg") is not None or bool(repeats_read(edge))


def files_gone(out_dir: Path, edge: dict, combined: int | None = None) -> list[str]:
    """Of an edge of the summary a directory has that was made from the results of its legs
    (:func:`made_from_results`): the files which its ddG, or the refusal of one, and what the summary holds of
    the edge were made from and which the directory does not have, by their path in the directory of the edge.
    They are the result of both legs of every repeat that was read, and the record of the prepared edge (its
    LOMAP score is read there, and the edges of a network are compared by it: an edge without it would be left
    out of that comparison). For a summary that does not name the repeats of its ddG: any result of a leg.
    ``combined``: the number of repeats the directory combines now (``sampling.repeats`` of its record); a
    repeat beyond them is not read again and its files are not asked for. Nor is the result of a leg that a run
    began and did not finish (:func:`leg_being_run`): its repeat is not combined, by the run and by
    `cufep rbfe analyze --write` alike."""
    edge_dir = Path(out_dir) / "edges" / edge["edge"]
    repeats = [r for r in repeats_read(edge) if combined is None or r < combined]
    # (a leg that a run began and did not finish has no result: the state of the directory, as for a run)
    missing = [f"repeat_{r}/{leg}/result.json" for r in repeats for leg in ("complex", "solvent")
               if not (edge_dir / f"repeat_{r}" / leg / "result.json").exists()
               and not leg_being_run(edge_dir / f"repeat_{r}" / leg)]
    if not (edge_dir / "prep" / "prep.json").exists():
        missing.append("prep/prep.json")
    if not repeats and not list(edge_dir.glob("repeat_*/*/result.json")):
        missing.insert(0, "repeat_*/*/result.json")
    return missing


def not_made_again(stored: Path, gone: list[tuple[str, list[str]]]) -> str:
    """The refusal of :func:`rewrite_results`: the edges of the summary ``stored`` with the files that are gone."""
    shown = "; ".join(f"{edge}: {', '.join(files)}" for edge, files in gone[:3]) + (" ..." if len(gone) > 3 else "")
    return (f"{stored} holds the results of {len(gone)} edge(s) made from files that are not in the directory "
            f"({shown}): made again from what is there, such an edge would have no ddG, or another one, or one that "
            "it was refused, or not all the summary holds of it. Nothing was written. A report of the summary as it "
            "is: cufep benchmark report NAME DIRECTORY --protocol rbfe")


def water_record_gone(stored: dict, made: dict | None) -> list[str]:
    """Of an edge of the summary a directory has (``stored``; with a ddG, or refused one for its repeats) and the
    same edge made again from the results of its legs (``made``): the repeats of whose complex leg the summary
    holds something of the grand
    canonical moves of water (a value of the water, the rule of the mover) that is not read any more. A result
    that does not record it itself has it from the record of its samples (:func:`water_moves_of_leg`), so that
    record is gone; named by its path in the directory of the edge. (The review of 7f4a1e6: with the samples of
    such a leg moved away the values of its water were not recorded and not compared any more, and a summary
    without a network, for other values of the water, was replaced by one with a network. Fifth review of PR #21,
    P1: the same for the repeats of one edge, which the summary had refused a ddG for their other values of the
    water; that edge was not looked at, having no ddG, and got the mean of the two repeats.)"""
    again = {r["repeat"]: r for r in (made or {}).get("repeats") or [] if "repeat" in r}
    gone = [f"repeat_{repeat['repeat']}/complex/hrex {WATER_RECORD}" for repeat in stored.get("repeats") or []
            if "ddg" in repeat and water_moves_lost(repeat.get("water_moves"),
                                                    (again.get(repeat.get("repeat")) or {}).get("water_moves"))]
    # a summary that holds them of the edge alone, not of its repeats (with them, a repeat that is read no more,
    # its result set aside by a run, takes its values out of the edge's without any record being gone)
    of_repeats = any(isinstance(repeat.get("water_moves"), dict) for repeat in stored.get("repeats") or [])
    if not of_repeats and water_moves_lost((stored.get("conditions") or {}).get("water_moves"),
                                           ((made or {}).get("conditions") or {}).get("water_moves")):
        gone.append(f"repeat_*/complex/hrex {WATER_RECORD}")
    return gone


def table_number(x) -> str:
    """A number as the tables of a run give it (three decimals; empty for none)."""
    return "" if x is None else f"{x:.3f}"


def table_values(out_dir: Path, name: str, key: str, value: str) -> dict[str, tuple[str, str]]:
    """``(value, standard error)`` as text, by ``key``, of the rows of a table of a directory (``edges.csv``,
    ``ligands.csv``); empty where the table is not there or cannot be read."""
    try:
        with open(Path(out_dir) / name, newline="") as fh:
            return {row[key]: ((row.get(value) or "").strip(), (row.get("standard_error") or "").strip())
                    for row in csv.DictReader(fh) if row.get(key)}
    except (OSError, csv.Error):
        return {}


def tables_changed(out_dir: Path, results: list[dict], dg: dict | None = None, ligands: bool = True) -> list[str]:
    """What the tables of a directory hold and the results of its legs, made again (``results``: edges as
    :func:`edge_result` gives them; ``dg``: the dG of a summary), would give otherwise: the ddG of an edge of
    ``edges.csv`` and, with ``ligands``, the dG of a ligand of ``ligands.csv``, each with its standard error, as
    the tables write them. For a directory whose summary cannot be read (or is not there): the tables are then
    the record of what it had, and they are made again only from legs that give the same. Sixth review of PR
    #21: a truncated summary.json in a copy of benchmarks/results/tyk2_rbfe let `cufep rbfe analyze --write`
    empty the ddG of 22 edges and the dG of 16 compounds; and its review: so would a table whose edge had the
    mean of two repeats, with one of them moved away, get the value of the other."""
    def said(pair: tuple[str, str]) -> str:
        return "none" if not pair[0] else f"{pair[0]} ± {pair[1]}" if pair[1] else pair[0]

    out = []
    made = {e["edge"]: (table_number(e.get("ddg")), table_number(e.get("se"))) for e in results}
    for name, held in table_values(out_dir, "edges.csv", "edge", "ddg_kcal_mol").items():
        if name in made and made[name] != held or name not in made and held[0]:
            out.append(f"the ddG of {name}: {said(held)} in edges.csv, {said(made.get(name, ('', '')))} from the legs")
    if ligands:
        values = (dg or {}).get("values") or {}
        for name, held in table_values(out_dir, "ligands.csv", "ligand", "dg_kcal_mol").items():
            now = (table_number((values.get(name) or {}).get("dg")), table_number((values.get(name) or {}).get("se")))
            if now != held:
                out.append(f"the dG of {name}: {said(held)} in ligands.csv, {said(now)} from the legs")
    return out


def rewrite_results(out_dir: Path) -> dict:
    """`cufep rbfe analyze --write`: the tables and the summary of a run made again from the results its legs have
    (:func:`analyse_directory`), as the run writes them. Under the lock of the directory from before the legs are
    read until the files are written, as a run holds it: a run in the directory could else finish a leg between the
    two, and its new summary be replaced by one that does not have the leg. Refused, with nothing written, where
    the summary the directory has holds the ddG of an edge, or the repeats for which it refused one, and the
    directory not every file that was made from (:func:`files_gone`, :func:`water_record_gone`): written again,
    such an edge would have no ddG, that of the repeats that are left, one it was refused, or conditions that are
    compared no more.
    (Second review of PR #21: a copy of the results without the directories of the edges,
    ``benchmarks/results/``; third: one leg of a repeat set aside, which left the other leg and passed.) The
    network and the number of repeats are those the directory records now (``network.json``,
    ``resolved_config.yaml``), as for the run."""
    from cufep.run import lock_output_directory

    out_dir = Path(out_dir)
    lock = lock_output_directory(out_dir)
    try:
        stored = out_dir / "summary.json"
        kept = read_summary(stored)
        had = stored_edges(kept)
        repeats = recorded_repeats(out_dir)
        gone = [(e["edge"], files_gone(out_dir, e, repeats)) for e in had]
        gone = [(edge, files) for edge, files in gone if files]
        if gone:
            raise WorkflowError(not_made_again(stored, gone))
        summary = analyse_directory(out_dir)
        made = {e["edge"]: e for e in summary["edges"]}
        gone = [(e["edge"], water_record_gone(e, made.get(e["edge"]))) for e in had]
        gone = [(edge, files) for edge, files in gone if files]
        if gone:
            raise WorkflowError(not_made_again(stored, gone))
        if kept is None:
            # no summary that can be read: the tables say what the directory had
            changed = tables_changed(out_dir, summary["edges"], summary.get("dg"))
            if changed:
                shown = "; ".join(changed[:3]) + (" ..." if len(changed) > 3 else "")
                raise WorkflowError(
                    f"{stored} cannot be read or is not there, and the tables of the directory hold {len(changed)} "
                    f"value(s) that the results of the legs in it do not give ({shown}): no record says what they "
                    "were made from. Nothing was written. With the legs they were made from, the summary is made "
                    "again; to make the results from the legs as they are, move edges.csv and ligands.csv away.")
        write_results(out_dir, summary)
        return summary
    finally:
        lock.close()


def recorded_repeats(out_dir: Path) -> int:
    """The number of repeats that the record of a run directory gives (``sampling.repeats`` of
    ``resolved_config.yaml``): those its results are combined from."""
    stored = yaml.safe_load((Path(out_dir) / "resolved_config.yaml").read_text()) or {}
    repeats = (stored.get("sampling") or {}).get("repeats", RbfeSampling.model_fields["repeats"].default)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise WorkflowError(f"{Path(out_dir) / 'resolved_config.yaml'}: sampling.repeats is {repeats!r}, not a "
                            "number of repeats")
    return repeats


def analyse_directory(out_dir: Path) -> dict:
    """The results of a relative run from the results its legs have, as the run makes
    them; writes nothing. Of the configuration of the run only what combining the results needs is read (the
    number of repeats): the checks of a configuration are for a run that simulates, and a directory that an
    earlier version wrote under other checks is analysed all the same (review of PR #18)."""
    out_dir = Path(out_dir)
    repeats = recorded_repeats(out_dir)
    network = json.loads((out_dir / "network.json").read_text())
    return combine_network(out_dir, network["ligands"], [tuple(e) for e in network["edges"]], repeats,
                           {k: float(v) for k, v in network.get("experiment_kcal_mol", {}).items()})
