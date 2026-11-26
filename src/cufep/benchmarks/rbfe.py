"""The benchmark sets run by relative calculations (``cufep benchmark run NAME --protocol rbfe``).

A set runs with the settings of its relative example (``rbfe_example`` in the registry): everything in that YAML
but the input files, the choice of ligands, the edges and the output directory. ``full`` runs every ligand of the
set and ``subset`` the five compounds of the registry, both on the network that LOMAP plans (``network.edges`` not
given); a set with ``rbfe_states`` runs every state of those compounds (A2A: the 29 states of the published map) and
combines them into compounds (``system.states``). ``smoke`` runs the first edge of the example as the example runs
it, with the sampling shortened as for the absolute benchmarks.

The report compares
    the dG of every ligand with experiment, with the published FEP+ predictions of the set and with the absolute
    calculation of cuFEP recorded in ``benchmarks/results/`` (``abfe_results``), and
    the ddG of every edge (its own, and the one the whole network gives it) with experiment and with the published
    FEP+ prediction of the same edge (``edge_predictions``),
by the statistics of the public benchmark (Ross et al., Commun. Chem. 2023): the pairwise RMSE over all pairs of
ligands, R^2 and Kendall tau of the dG, and the edgewise RMSE. Numbers only; no verdict.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cufep.benchmarks.runner import (
    CUFEP_COLOUR,
    FEPPLUS_COLOUR,
    GUIDE_COLOUR,
    SMOKE_SAMPLING,
    STATISTICS,
    TIERS,
    _pyplot,
    _save_figure,
    _shorten,
    data_root,
    load_registry,
    read_reference,
)
from cufep.benchmarks.stats import bootstrap, metrics

#: the files a relative report writes (the figures are removed when there is nothing to draw)
REPORT_FILES = ("rbfe_report.md", "rbfe_report_ligands.csv", "rbfe_report_edges.csv")
FIGURES = ("rbfe_report.png", "rbfe_report_edges.png")
LIGAND_COLUMNS = ("ligand", "exp", "cufep_rbfe", "cufep_rbfe_error", "fepplus", "cufep_abfe")
EDGE_COLUMNS = ("edge", "ligand_a", "ligand_b", "exp_ddg", "exp_ddg_of", "ddg", "ddg_error", "network_ddg",
                "network_ddg_error",
                "fepplus_bennett_ddg", "fepplus_bennett_error", "fepplus_ccc_ddg", "fepplus_ccc_error",
                "lowest_overlap_complex", "lowest_overlap_solvent", "not_reliable")


def is_relative_run(run_dir: str | Path) -> bool:
    """A directory of ``cufep rbfe run`` (or of a relative benchmark): it holds the network of the run, or (a
    directory that keeps the summary alone) a summary of edges (sixth review of PR #19: such a directory went to the
    report of the absolute calculation, which failed on it)."""
    run_dir = Path(run_dir)
    if (run_dir / "network.json").exists():
        return True
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(summary, dict) and isinstance(summary.get("edges"), list) and "ligands" in summary


def _entry(name: str) -> dict:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    entry = registry[name]
    if "rbfe_example" not in entry:
        raise KeyError(f"benchmark '{name}' has no relative example")
    return entry


def rbfe_benchmark_config(name: str, tier: str, out_dir: str | Path | None = None):
    """The configuration of a relative benchmark tier (the module's docstring)."""
    from cufep.rbfe.config import RbfeConfig, load_rbfe_config

    entry, root = _entry(name), data_root()
    if tier not in TIERS:
        raise KeyError(f"unknown tier '{tier}' (available: {', '.join(TIERS)})")
    example = load_rbfe_config(root / entry["rbfe_example"])
    data = example.model_dump(mode="json")
    data["system"].update(protein=str(root / entry["protein"]), ligands=str(root / entry["rbfe_ligands"]))
    data["output"]["dir"] = str(out_dir or Path("runs") / f"benchmark_rbfe_{name}_{tier}")
    if tier == "smoke":
        first = list((example.network.edges or [])[:1])
        if not first:
            raise KeyError(f"the relative example of '{name}' gives no edge for the smoke tier")
        # the edge as the example runs it: its ligands are records of the set, one per compound
        data["system"].update(ligands=str(root / entry["ligands"]), states=None, ligand_names=list(first[0]))
        data["network"]["edges"] = [list(first[0])]
        _shorten(data, {"sampling": SMOKE_SAMPLING,
                        "lambda_schedule": {"adaptive": {"window_ps": 10, "equilibration_ps": 0}},
                        "membrane": {"equilibration_ns": 0.5}})
    else:
        compounds = list(entry["subset"]) if tier == "subset" else None
        if entry.get("rbfe_states"):
            states = read_states(name)
            data["system"]["states"] = str(root / entry["rbfe_states"])
            data["system"]["ligand_names"] = None if compounds is None else [
                s for s, r in states.items() if r["compound"] in compounds]
        else:
            data["system"]["ligand_names"] = compounds
        data["network"]["edges"] = None
    return RbfeConfig.model_validate(data)


def read_states(name: str) -> dict[str, dict]:
    """The rows of ``rbfe_states`` of a set by state (empty for a set without one)."""
    entry = _entry(name)
    if not entry.get("rbfe_states"):
        return {}
    with open(data_root() / entry["rbfe_states"]) as fh:
        return {r["ligand"].strip(): r for r in csv.DictReader(fh)}


def _record_names(name: str, by_state: bool = False) -> dict[str, str]:
    """The name in a relative run of each SDF record of the set: of the states of ``rbfe_states`` (``by_state``), or
    of the record ``ligands.csv`` chose for each compound (record -> compound)."""
    if by_state:
        return {r["sdf_record"].strip(): s for s, r in read_states(name).items()}
    entry = _entry(name)
    with open(data_root() / entry["reference"]) as fh:
        return {r["sdf_record"].strip(): r["compound"].strip() for r in csv.DictReader(fh)}


def published_edges(name: str, by_state: bool = False) -> dict[tuple[str, str], dict]:
    """The published FEP+ prediction of every edge of the set whose two ends are ligands of a relative run (states of
    ``rbfe_states`` with ``by_state``), as (A, B) -> {bennett, bennett_error, ccc, ccc_error}, ddG = dG(B) - dG(A) in
    kcal/mol (the file's ``Lig 1``, ``Lig 2``; its experimental ddG is the difference of the experimental dG of the
    two)."""
    entry = _entry(name)
    names = _record_names(name, by_state)
    out = {}
    with open(data_root() / entry["edge_predictions"]) as fh:
        for row in csv.DictReader(fh):
            a, b = names.get(row["Lig 1"].strip()), names.get(row["Lig 2"].strip())
            if a is None or b is None:
                continue                               # an end that is not a ligand of the run
            out[(a, b)] = {"bennett": float(row["Bennett ddG (kcal/mol)"]),
                           "bennett_error": float(row["Bennett std. error (kcal/mol)"]),
                           "ccc": float(row["CCC ddG (kcal/mol)"]),
                           "ccc_error": float(row["CCC std. error (kcal/mol)"])}
    return out


def _published(edges: dict, a: str, b: str) -> dict | None:
    """The published prediction of the edge A -> B, also when it was published as B -> A (signs turned)."""
    if (a, b) in edges:
        return edges[(a, b)]
    if (b, a) in edges:
        e = edges[(b, a)]
        return {"bennett": -e["bennett"], "bennett_error": e["bennett_error"], "ccc": -e["ccc"],
                "ccc_error": e["ccc_error"]}
    return None


def stored_abfe(name: str) -> dict[str, float]:
    """The dG of every ligand of the absolute calculation of the set recorded in ``benchmarks/results/``."""
    entry = _entry(name)
    path = data_root() / entry.get("abfe_results", "")
    if not entry.get("abfe_results") or not path.is_file():
        return {}
    with open(path) as fh:
        return {r["ligand"]: float(r["cufep"]) for r in csv.DictReader(fh) if r.get("cufep")}


def edge_errors(pred, exp) -> dict[str, float | None]:
    """RMSE and MUE of the ddG of edges against experiment (the edgewise errors of Ross et al. 2023)."""
    err = np.asarray(pred, dtype=float) - np.asarray(exp, dtype=float)
    if not len(err):
        return {"n": 0, "rmse": None, "mue": None}
    return {"n": len(err), "rmse": float(np.sqrt(np.mean(err ** 2))), "mue": float(np.mean(np.abs(err)))}


def states_of_a_summary(summary: dict, name: str | None = None) -> dict | None:
    """The states of the compounds (``{"compound": state -> compound}``) as a summary of a relative run records them,
    for a directory that keeps the summary without its network (seventh review of PR #19: the report of an A2A
    summary alone compared the experiment of compounds with the dG of their 29 states, and an edge between two states
    of one compound with an experimental ddG of 0): the compound of each state whose experimental value it has
    (``experiment_of_compound``), the states the dG of each compound combines (``compounds``), and the compounds of
    the two ends of an edge whose experiment is that of the compounds (``exp_compounds``; a summary of an earlier
    version of a network that did not yet connect its states has neither of the others, eighth review). A state still
    without a compound is taken from the states of the benchmark ``name`` where it is one of them, and a summary with
    a name that only a state of the benchmark has is of a run of states whatever else it records. None for a run
    without states; a state whose compound is not known is not in the mapping, and the report leaves it out of what
    it compares with experiment."""
    compound_of = dict(summary.get("experiment_of_compound") or {})
    for compound, value in ((summary.get("compounds") or {}).get("values") or {}).items():
        for state in (value or {}).get("weights") or {}:
            compound_of.setdefault(state, compound)
    of_states = False
    for edge in summary.get("edges") or []:
        if edge.get("exp_ddg_of") == "the compounds":
            of_states = True
            ends = edge.get("exp_compounds") or []
            if len(ends) == 2:
                compound_of.setdefault(edge["ligand_a"], ends[0])
                compound_of.setdefault(edge["ligand_b"], ends[1])
    ligands = list(summary.get("ligands") or [])
    of_benchmark = read_states(name) if name is not None and _entry(name).get("rbfe_states") else {}
    # what a summary records of a run without states: the version that writes its directory writes the compounds of
    # a run of states too; the reference of the dG of ligands; the experiment of an edge as that of its ligands
    # (tenth review of PR #19: a run of the ligands 4h and 4jS without system.states lost its dG to the names)
    without_states = ("run_directory" in summary or bool((summary.get("dg") or {}).get("reference_ligands"))
                      or any(e.get("exp_ddg_of") == "the ligands" for e in summary.get("edges") or []))
    if not compound_of and not of_states and without_states:
        return None
    # a summary that records neither (2ecbcd2, a network that gave no dG): a name that only a state of the benchmark
    # has (4jS, not a compound) makes it a run of states (ninth review: an edge within 4j was compared with an
    # experimental ddG of 0)
    compounds_of_benchmark = set(read_reference(name)) if of_benchmark else set()
    if any(s in of_benchmark and s not in compounds_of_benchmark for s in ligands):
        of_states = True
    if not compound_of and not of_states:
        return None
    for state, row in of_benchmark.items():
        if state in ligands:
            compound_of.setdefault(state, row["compound"])
    return {"compound": compound_of}


def dg_with_experiment(summary: dict, states: dict | None, experiment: dict[str, float],
                       temperature_K: float) -> tuple[dict, str] | None:
    """The dG of the ligands (or, with ``states``, of the compounds) of a relative run and their standard errors,
    computed again from the ddG of its edges by the estimate of the run (:func:`cufep.rbfe.network.ligand_dg`,
    :func:`cufep.rbfe.network.compound_dg`) with ``experiment`` as the reference: ``({name: {"dg", "se"}},
    reference)``. None where the summary has no dG, or does not record what the estimate needs (the compound,
    population and symmetry number of every state)."""
    from cufep.rbfe.network import Edge, compound_dg, ligand_dg
    from cufep.sampling.states import R_GAS
    from cufep.workflow import KCAL

    if not (summary.get("dg") or {}).get("values"):
        return None
    measured = [Edge(e["ligand_a"], e["ligand_b"], float(e["ddg"]), float(e["se"])) for e in summary["edges"]
                if e.get("ddg") is not None and e.get("se") is not None]
    names = list(summary["ligands"])
    if states is None:
        out = ligand_dg(names, measured, {n: v for n, v in experiment.items() if n in set(names)})
        return {n: {"dg": v, "se": e} for n, (v, e) in out["values"].items()}, out["reference"]
    compound_of = states.get("compound") or {}
    if any(n not in compound_of for n in names) or "fraction" not in states or "symmetry" not in states:
        return None
    compounds = set(compound_of[n] for n in names)
    out = compound_dg(names, measured, compound_of, {c: v for c, v in experiment.items() if c in compounds},
                      states["fraction"], states["symmetry"], R_GAS * float(temperature_K) / KCAL)
    return {c: {"dg": v["dg"], "se": v["se"]} for c, v in out["compounds"].items()}, out["reference"]


def analysed_summary(run_dir: Path) -> dict:
    """The summary of a relative run as its directory gives it now: from its legs where it is the directory of a run
    (``edges/``, ``resolved_config.yaml``, ``network.json``), whatever results its legs have at the moment (fourth
    review of PR #19: with every leg run again and none finished yet, the ddG of an earlier summary.json came back);
    else (a directory that holds the summary alone) its ``summary.json``."""
    from cufep.rbfe.run import analyse_directory

    run_dir = Path(run_dir)
    if ((run_dir / "edges").is_dir() and (run_dir / "resolved_config.yaml").exists()
            and (run_dir / "network.json").exists()):
        return analyse_directory(run_dir)
    return json.loads((run_dir / "summary.json").read_text())


def report(name: str, run_dir: str | Path, write: bool = True) -> str:
    """Compare a relative run with experiment, with FEP+ and with the absolute calculation; returns Markdown.

    The files are those of the run as it is now (the figures of an earlier state are removed). A directory with the
    results of its legs is analysed from them as `cufep rbfe analyze` analyses it
    (:func:`cufep.rbfe.run.analyse_directory`), so that both take the same edges (third review of PR #19: legs that
    the analysis leaves out came back from an earlier summary.json); one without them is reported from its
    summary.json."""
    run_dir = Path(run_dir)
    summary = analysed_summary(run_dir)
    reference = read_reference(name)
    abfe = stored_abfe(name)
    network = run_dir / "network.json"
    record = json.loads(network.read_text()) if network.exists() else {}
    # the states of the compounds as the run recorded them, whether its network gave dG or not (review of PR #19: an
    # incomplete network lost them, and an edge between two states of one compound got an experimental ddG of 0)
    # the network records whether the run had states; a summary alone is read for them (states_of_a_summary)
    # a summary of this version records the states of its compounds as the estimate takes them (thirteenth review of
    # PR #19: without their populations the report could not compute the dG of a summary alone again)
    states = (record.get("states") if network.exists()
              else summary.get("states") if isinstance(summary.get("states"), dict)
              else states_of_a_summary(summary, name))
    compound_of = dict((states or {}).get("compound") or {})
    compounds = summary.get("compounds")
    published = published_edges(name, by_state=bool(states))
    experiment = {k: float(v) for k, v in (summary.get("experiment_kcal_mol") or {}).items()}
    unmapped = [s for s in summary["ligands"] if states and s not in compound_of]
    if states:
        # the dG are those of the compounds, each from its states (system.states); a state whose compound is not
        # known is compared with nothing (eighth review of PR #19)
        names_of = list(dict.fromkeys(compound_of[s] for s in summary["ligands"] if s in compound_of))
        values = (compounds or {}).get("values") or {}
        experiment_of = {compound_of[s]: v for s, v in experiment.items() if s in compound_of}
    else:
        values, names_of = (summary.get("dg") or {}).get("values") or {}, list(summary["ligands"])
        experiment_of = experiment

    def experiment_of_name(name_of: str):
        """The experimental dG of a ligand, or of a compound where the ligands are states: the value the run was
        given, else that of the benchmark."""
        return experiment_of.get(name_of, reference.get(name_of, {}).get("exp"))

    def experiment_of_end(x: str):
        """The experimental dG an end of an edge is compared with: of its compound for a state, else of the ligand
        (:func:`experiment_of_name`; eleventh review of PR #19: the table of ligands took the benchmark's and the
        edges did not)."""
        name_of = compound_of.get(x) if states else x
        if name_of is None:
            return None
        return experiment_of_name(name_of)

    # the origin of the dG of the run: the mean of the experimental values it was given, or (none given) the mean of
    # the network, which is not that of experiment (eleventh review of PR #19: a ddG reproduced exactly gave a raw RMSE
    # of 10.4 kcal/mol). Then the dG and their standard errors are computed again from the edges, by the estimate of
    # the run, with the experimental values of this table (twelfth review: dG shifted to them kept the standard errors
    # of the mean of the network); a summary that does not record what that needs is shifted, and says that its
    # standard errors are those of the mean of the network
    origin = ((summary.get("dg") or {}).get("reference") or "")
    how = "run"
    if origin and origin != "mean of the experimental values":
        temperature = float((compounds or {}).get("temperature_K") or record.get("temperature_K") or 298.15)
        resolved = {n: experiment_of_name(n) for n in names_of}
        again = dg_with_experiment(summary, states or None, {n: v for n, v in resolved.items() if v is not None},
                                   temperature)
        if again is not None:
            values, origin = again
            how = "again"
        else:
            how = "shifted"
    ligands = []
    for lig in names_of:
        ref = reference.get(lig, {})
        v = values.get(lig) or {}
        ligands.append({"ligand": lig, "exp": experiment_of.get(lig, ref.get("exp")), "cufep_rbfe": v.get("dg"),
                        "cufep_rbfe_error": v.get("se"), "fepplus": ref.get("fepplus"), "cufep_abfe": abfe.get(lig)})
    shifted = None
    if how == "shifted":
        scored_rows = [r for r in ligands if r["cufep_rbfe"] is not None and r["exp"] is not None]
        if scored_rows:
            shifted = (sum(r["exp"] for r in scored_rows)
                       - sum(r["cufep_rbfe"] for r in scored_rows)) / len(scored_rows)
            for r in ligands:
                if r["cufep_rbfe"] is not None:
                    r["cufep_rbfe"] = r["cufep_rbfe"] + shifted
    edges = []
    for e in summary["edges"]:
        a, b = e["ligand_a"], e["ligand_b"]
        pub = _published(published, a, b) or {}
        exp_ddg = e.get("exp_ddg")
        if exp_ddg is None and experiment_of_end(a) is not None and experiment_of_end(b) is not None:
            exp_ddg = experiment_of_end(b) - experiment_of_end(a)
        if states and (a not in compound_of or b not in compound_of or compound_of[a] == compound_of[b]):
            # two states of one compound: experiment has no ddG between them (review of PR #19: 0 in the figure);
            # a state whose compound is not known: not compared (eighth review)
            exp_ddg = None
        overlap = e.get("lowest_overlap") or {}
        edges.append({"edge": e["edge"], "ligand_a": a, "ligand_b": b, "exp_ddg": exp_ddg,
                      "exp_ddg_of": "the compounds" if states else "the ligands", "ddg": e.get("ddg"),
                      "ddg_error": e.get("se"), "network_ddg": e.get("network_ddg"),
                      "network_ddg_error": e.get("network_se"), "fepplus_bennett_ddg": pub.get("bennett"),
                      "fepplus_bennett_error": pub.get("bennett_error"), "fepplus_ccc_ddg": pub.get("ccc"),
                      "fepplus_ccc_error": pub.get("ccc_error"), "lowest_overlap_complex": overlap.get("complex"),
                      "lowest_overlap_solvent": overlap.get("solvent"),
                      "not_reliable": "; ".join(e.get("not_reliable") or [])})

    title = load_registry()[name]["title"]
    with_dg = [r for r in ligands if r["cufep_rbfe"] is not None]
    with_ddg = [r for r in edges if r["ddg"] is not None]
    edges_from = record.get("edges_from", "")
    what = "Compounds" if states else "Ligands"
    lines = [f"# cuFEP relative benchmark report: {name}", "", title, "",
             f"{what} with a dG: {len(with_dg)} of {len(ligands)}; edges with a ddG: {len(with_ddg)} of {len(edges)}"
             + (f" (edges {edges_from})." if edges_from else "."), "",
             *([] if not states else [
                 f"The {len(summary['ligands'])} ligands of the network are states of {len(ligands)} compounds "
                 "(system.states): the dG of a compound combines its states, -kT ln sum P exp(-dG/kT) - kT ln s "
                 "(docs/rbfe.md, \"States of a compound\"). Experiment measures compounds: the ddG compared with it "
                 "are those of the pairs of compounds the edges join, from the dG of the compounds; an edge between "
                 "states is compared with the published prediction of the same edge only.", ""]),
             ("The network of the run gave no dG (the notes of the run below say why), and no dG of cuFEP is "
              "compared with experiment; the ddG of the edges are"
              if not origin else
              f"The dG of cuFEP come from the edges alone, their mean over the {what.lower()} with an experimental "
              "value set to the mean of those values (`docs/rbfe.md`, \"The network\")"
              if how == "run" and origin == "mean of the experimental values" else
              "The dG of cuFEP come from the edges alone; the run was given no experimental values and set their mean "
              "over the network to 0 (`docs/rbfe.md`, \"The network\"): here the dG and their standard errors are "
              f"computed again from the ddG of its edges by the same estimate, their mean over the {what.lower()} "
              "with an experimental value of this table set to the mean of those values"
              if how == "again" and origin == "mean of the experimental values" else
              "The dG of cuFEP come from the edges alone, their mean over the network set to 0 (no experimental "
              "value: they are not on the scale of experiment, and the raw errors compare values of other origins)"
              if shifted is None else
              "The dG of cuFEP come from the edges alone; the run was given no experimental values and set their mean "
              "over the network to 0 (`docs/rbfe.md`, \"The network\"): here they are shifted by "
              f"{shifted:+.2f} kcal/mol to the mean of the experimental values of the {what.lower()} compared, and "
              "their standard errors are those of the run, against the mean of the network (the summary does not "
              "record what computing them again needs)")
             + "; the published FEP+ values are relative "
             "calculations shifted to the experimental mean of the whole published set, with 20 ns per lambda window "
             "(Ross et al. 2023), and cuFEP's absolute values (`benchmarks/results/`) are not shifted. The pairwise "
             "RMSE, R^2 and Kendall tau do not depend on a shift.", ""]
    for note in summary.get("notes") or []:
        lines.append(f"Note of the run: {note}")
    if unmapped:
        lines.append(f"States whose compound the summary does not record ({', '.join(unmapped)}) are compared with "
                     "nothing (the experiment measures compounds).")
    if summary.get("notes") or unmapped:
        lines.append("")

    lines += ["## Per ligand (kcal/mol)", "",
              "| ligand | experiment | cuFEP RBFE | standard error | FEP+ | cuFEP ABFE |",
              "|---|---|---|---|---|---|"]

    def num(x, fmt=".2f") -> str:
        return "" if x is None else format(x, fmt)

    for r in sorted(ligands, key=lambda r: (r["exp"] is None, r["exp"] or 0.0)):
        lines.append(f"| {r['ligand']} | {num(r['exp'])} | {num(r['cufep_rbfe'])} | {num(r['cufep_rbfe_error'])} | "
                     f"{num(r['fepplus'])} | {num(r['cufep_abfe'])} |")

    scored = [r for r in with_dg if r["exp"] is not None]
    lines += ["", "## Agreement of the dG with experiment", ""]
    if len(scored) >= 2:
        exp = [r["exp"] for r in scored]
        series = {"cuFEP RBFE": [r["cufep_rbfe"] for r in scored]}
        if all(r["fepplus"] is not None for r in scored):
            series["FEP+"] = [r["fepplus"] for r in scored]
        if all(r["cufep_abfe"] is not None for r in scored):
            series["cuFEP ABFE"] = [r["cufep_abfe"] for r in scored]
        stats = {k: (metrics(v, exp), bootstrap(v, exp) if len(scored) >= 4 else {}) for k, v in series.items()}

        def cell(method: str, key: str) -> str:
            m, ci = stats[method]
            if m[key] is None:
                return "N/A"
            return f"{m[key]:.2f}" + (f" [{ci[key][0]:.2f}, {ci[key][1]:.2f}]" if key in ci else "")

        lines += [f"{len(scored)} ligands with a dG and an experimental value; 95 % bootstrap intervals over "
                  "ligands in brackets.", "", "| statistic | " + " | ".join(series) + " |",
                  "|---|" + "---|" * len(series)]
        lines += [f"| {label} | " + " | ".join(cell(k, key) for k in series) + " |" for key, label in STATISTICS]
    else:
        lines.append("Fewer than two ligands have a dG and an experimental value.")

    lines += ["", "## Per edge (kcal/mol)", "",
              "ddG = dG(B) - dG(A). Own: the edge's two legs; network: dG(B) - dG(A) of the estimate from all edges "
              "(what the cycle closure correction gives an edge); FEP+: the published prediction of the same edge "
              "(Bennett and cycle closure corrected), where the published map has it. "
              + ("The experiment of an edge between states is that of their two compounds (none within one "
                 "compound); it is not what the edge computes." if states else ""), "",
              f"| edge | {'experiment (of the compounds)' if states else 'experiment'} | cuFEP own | cuFEP network | "
              "FEP+ Bennett | FEP+ CCC | lowest overlap complex, solvent | NOT RELIABLE |",
              "|---|---|---|---|---|---|---|---|"]
    for r in edges:
        own = "" if r["ddg"] is None else f"{r['ddg']:+.2f} ± {r['ddg_error']:.2f}"
        net = "" if r["network_ddg"] is None else f"{r['network_ddg']:+.2f} ± {r['network_ddg_error']:.2f}"
        lines.append(f"| {r['edge']} | {num(r['exp_ddg'], '+.2f')} | {own} | {net} | "
                     f"{num(r['fepplus_bennett_ddg'], '+.2f')} | {num(r['fepplus_ccc_ddg'], '+.2f')} | "
                     f"{num(r['lowest_overlap_complex'], '.3f')}, {num(r['lowest_overlap_solvent'], '.3f')} | "
                     f"{r['not_reliable']} |")

    lines += ["", "## Agreement of the ddG of the edges with experiment", "",
              "| ddG | edges | RMSE | MUE |", "|---|---|---|---|"]

    def edge_row(label: str, pairs: list[tuple[float, float]]) -> str:
        m = edge_errors([p for p, _ in pairs], [x for _, x in pairs])
        return f"| {label} | {m['n']} | {num(m['rmse'])} | {num(m['mue'])} |"

    pair_rows = []
    if states:
        # experiment against the ddG of the pairs of compounds that the edges join (review of PR #19: a state edge
        # and the experiment of its compounds are not one quantity, e.g. kT ln 2 for two equal binding modes)
        pairs = list(dict.fromkeys(tuple(sorted((compound_of[r["ligand_a"]], compound_of[r["ligand_b"]])))
                                   for r in edges if r["ligand_a"] in compound_of and r["ligand_b"] in compound_of
                                   and compound_of[r["ligand_a"]] != compound_of[r["ligand_b"]]))
        # the experimental value of the table of compounds (that of the run, else of the benchmark; second review)
        dg_of = {r["ligand"]: r for r in ligands}
        for ca, cb in pairs:
            if dg_of[ca]["exp"] is None or dg_of[cb]["exp"] is None:
                continue
            ours = (None if dg_of[ca]["cufep_rbfe"] is None or dg_of[cb]["cufep_rbfe"] is None
                    else dg_of[cb]["cufep_rbfe"] - dg_of[ca]["cufep_rbfe"])
            theirs = (None if dg_of[ca]["fepplus"] is None or dg_of[cb]["fepplus"] is None
                      else dg_of[cb]["fepplus"] - dg_of[ca]["fepplus"])
            pair_rows.append({"edge": f"{ca}~{cb}", "exp_ddg": dg_of[cb]["exp"] - dg_of[ca]["exp"], "ddg": ours,
                              "ddg_error": None, "fepplus_bennett_ddg": theirs})
        lines.append(edge_row("cuFEP, the pairs of compounds the edges join",
                              [(r["ddg"], r["exp_ddg"]) for r in pair_rows if r["ddg"] is not None]))
        lines.append(edge_row("FEP+ (published dG of the compounds), the same pairs",
                              [(r["fepplus_bennett_ddg"], r["exp_ddg"]) for r in pair_rows
                               if r["fepplus_bennett_ddg"] is not None]))
        same_edges = [(r["ddg"], r["fepplus_bennett_ddg"]) for r in edges
                      if r["ddg"] is not None and r["fepplus_bennett_ddg"] is not None]
        m = edge_errors([p for p, _ in same_edges], [x for _, x in same_edges])
        lines += ["", f"Edges between states against the published prediction of the same edge (FEP+ Bennett, no "
                      f"experiment): {m['n']} edges, RMSE {num(m['rmse'])}, MUE {num(m['mue'])} kcal/mol."]
    else:
        scored_edges = [r for r in edges if r["exp_ddg"] is not None]
        lines.append(edge_row("cuFEP, own", [(r["ddg"], r["exp_ddg"]) for r in scored_edges if r["ddg"] is not None]))
        lines.append(edge_row("cuFEP, network", [(r["network_ddg"], r["exp_ddg"]) for r in scored_edges
                                                 if r["network_ddg"] is not None]))
        common = [r for r in scored_edges if r["ddg"] is not None and r["fepplus_bennett_ddg"] is not None]
        lines.append(edge_row("cuFEP, own, on the edges FEP+ also has", [(r["ddg"], r["exp_ddg"]) for r in common]))
        lines.append(edge_row("FEP+ Bennett, the same edges",
                              [(r["fepplus_bennett_ddg"], r["exp_ddg"]) for r in common]))
        names = set(summary["ligands"])
        own_map = [(v["bennett"], experiment_of_end(b) - experiment_of_end(a)) for (a, b), v in published.items()
                   if a in names and b in names and experiment_of_end(a) is not None
                   and experiment_of_end(b) is not None]
        lines.append(edge_row("FEP+ Bennett, its whole map among these ligands", own_map))

    text = "\n".join(lines) + "\n"
    if write:
        _save(run_dir, text, ligands, edges, name, pair_rows if states else None,
              errors_of_the_run=shifted is not None)
    return text


def axis_range(rows: list[dict], x: str, ys) -> tuple[float, float]:
    """The range of both axes of a figure of calculated against experimental values: every value and the ends of
    every error bar (eleventh review of PR #19: they were cut off at the axes), 1 kcal/mol beyond them."""
    values = [r[x] for r in rows] + [r[k] + s * ((r[err] or 0.0) if err else 0.0) for r in rows
                                     for k, err, _ in ys if r[k] is not None for s in (-1.0, 1.0)]
    return min(values) - 1.0, max(values) + 1.0


#: of a summary whose dG are shifted (it does not record what computing them again needs): what their standard errors
#: are about, in a column of the table of the ligands and in the figure (thirteenth review of PR #19)
ERRORS_OF_THE_RUN = "the dG of the run, about the mean of its network, not the shifted values"


def _save(run_dir: Path, text: str, ligands: list[dict], edges: list[dict], name: str, pairs=None,
          errors_of_the_run: bool = False) -> None:
    from cufep.files import publish_text, publish_with

    publish_text(run_dir / "rbfe_report.md", text)
    columns_of_ligands = LIGAND_COLUMNS
    if errors_of_the_run:
        columns_of_ligands = LIGAND_COLUMNS + ("cufep_rbfe_error_of",)
        ligands = [{**r, "cufep_rbfe_error_of": ERRORS_OF_THE_RUN if r.get("cufep_rbfe_error") is not None else ""}
                   for r in ligands]
    for path, columns, rows in ((run_dir / "rbfe_report_ligands.csv", columns_of_ligands, ligands),
                                (run_dir / "rbfe_report_edges.csv", EDGE_COLUMNS, edges)):
        def table(fh, columns=columns, rows=rows) -> None:
            w = csv.DictWriter(fh, fieldnames=list(columns), restval="")
            w.writeheader()
            w.writerows(rows)

        publish_with(path, table, newline="")
    for figure in FIGURES:
        (run_dir / figure).unlink(missing_ok=True)     # never the figure of an earlier state of the run
    _plot(run_dir, ligands, edges if pairs is None else pairs, name, pairs is not None, errors_of_the_run)


def _plot(run_dir: Path, ligands: list[dict], edges: list[dict], name: str, of_compounds: bool = False,
          errors_of_the_run: bool = False) -> None:
    """The figures: the dG against experiment, and the ddG against experiment (of the edges, or with states of the
    pairs of compounds the edges join). ``errors_of_the_run``: the error bars of the dG are those of the run about the
    mean of its network (shifted dG), which the figure says."""
    plt = _pyplot()
    if plt is None:
        return
    for figure, rows, x, ys, label in (
            ("rbfe_report.png", [r for r in ligands if r["cufep_rbfe"] is not None and r["exp"] is not None], "exp",
             (("cufep_rbfe", "cufep_rbfe_error", "cuFEP RBFE"), ("fepplus", None, "FEP+ (published)")),
             "dG"),
            ("rbfe_report_edges.png", [r for r in edges if r["ddg"] is not None and r["exp_ddg"] is not None],
             "exp_ddg", (("ddg", "ddg_error", "cuFEP, compounds" if of_compounds else "cuFEP, own"),
                         ("fepplus_bennett_ddg", None,
                          "FEP+ (published dG of the compounds)" if of_compounds else "FEP+ Bennett (same edge)")),
             "ddG")):
        if not rows:
            continue
        lo, hi = axis_range(rows, x, ys)
        fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
        ax.plot([lo, hi], [lo, hi], color=GUIDE_COLOUR, lw=1)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.92", zorder=0)
        for (key, err, what), colour, marker in zip(ys, (CUFEP_COLOUR, FEPPLUS_COLOUR), ("o", "s")):
            part = [r for r in rows if r[key] is not None]
            if not part:
                continue
            ax.errorbar([r[x] for r in part], [r[key] for r in part],
                        yerr=[r[err] or 0.0 for r in part] if err else None, fmt=marker, ms=6, color=colour,
                        mfc=colour if marker == "o" else "none", capsize=2, elinewidth=1, label=what)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"experiment {label} (kcal/mol)")
        ax.set_ylabel(f"calculated {label} (kcal/mol)")
        ax.set_title(f"{name}: relative calculations, {label}"
                     + ("\n(error bars: standard errors about the mean of the network)"
                        if errors_of_the_run and label == "dG" else ""), fontsize=9 if errors_of_the_run else None)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _save_figure(fig, run_dir / figure)
        plt.close(fig)
