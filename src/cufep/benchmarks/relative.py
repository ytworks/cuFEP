"""The built-in benchmarks by relative binding free energies (`cufep benchmark ... --protocol rbfe`).

A set runs with the settings of its relative example (``rbfe_example`` in the registry): everything in that
configuration but the input files, which are the set's, the ligands and their network, and the output
directory. ``full`` runs every compound of the set on the LOMAP network of ``cufep rbfe`` (``network.edges``
of the example is not used); ``smoke`` runs the first edge of the example alone, with shortened sampling (never
lengthening a setting of the example). The experimental values are those of the set's table
(``ligands.csv``), which the relative calculation also uses to set the one constant of the dG.
"""
from __future__ import annotations

import csv
from pathlib import Path

from cufep.benchmarks.runner import _shorten, data_root, load_registry, read_reference

TIERS = ("smoke", "full")
#: the shortened settings of the smoke tier (the numbers of the absolute smoke tier, for the sections a relative
#: configuration has)
SMOKE = {"sampling": {"pre_equilibration": {"complex": {"nvt_ns": 0.02, "npt_ns": 0.02, "md_ns": 0.3},
                                            "solvent": {"nvt_ns": 0.01, "npt_ns": 0.02, "md_ns": 0.05}},
                      "equilibration_ns": 0.05, "production_ns": 0.1, "repeats": 1},
         "lambda_schedule": {"adaptive": {"window_ps": 10, "equilibration_ps": 0}},
         "membrane": {"equilibration_ns": 0.5}}


def _entry(name: str) -> dict:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    return registry[name]


def example_path(name: str) -> Path:
    return data_root() / _entry(name)["rbfe_example"]


def benchmark_rbfe_config(name: str, tier: str, out_dir: str | Path | None = None):
    """The configuration of a relative benchmark run (:class:`cufep.rbfe.config.RbfeConfig`)."""
    from cufep.rbfe.config import RbfeConfig, load_rbfe_config

    if tier not in TIERS:
        raise KeyError(f"unknown tier '{tier}' of a relative benchmark (available: {', '.join(TIERS)})")
    entry, root = _entry(name), data_root()
    example = load_rbfe_config(example_path(name))
    data = example.model_dump(mode="json")
    data["system"].update(protein=str(root / entry["protein"]), ligands=str(root / entry["ligands"]),
                          experiment=None)
    data["output"]["dir"] = str(out_dir or Path("runs") / f"benchmark_rbfe_{name}_{tier}")
    if tier == "smoke":
        first = [list(example.network.edges[0])] if example.network.edges else None
        if first is None:
            raise ValueError(f"{example_path(name)} gives no edge for the smoke tier")
        data["system"]["ligand_names"] = first[0]
        data["network"]["edges"] = first
        _shorten(data, SMOKE)
    else:
        data["system"]["ligand_names"] = None
        data["network"]["edges"] = None
    return RbfeConfig.model_validate(data)


def experiment(name: str) -> dict[str, float]:
    """The experimental dG (kcal/mol) of every compound of the set, from its table."""
    return {compound: ref["exp"] for compound, ref in read_reference(name).items()}


def published_edges(name: str) -> dict[frozenset, dict]:
    """The edges of the published map of the set with their FEP+ predictions (``edge_predictions_*_out.csv`` of the
    public benchmark), keyed by the pair of compounds; each with the direction it was computed in.

    The map names its records (A2A: "4e D", 4jS ...), the set its compounds: a record is the compound whose record
    it is in the set's table (``sdf_record``), and an edge of another state of a compound (its other rotamer, its
    deprotonated form) is not an edge of the set (review of PR #21: the A2A edges were looked up by record names)."""
    root = data_root()
    with open(root / _entry(name)["reference"], newline="") as fh:
        compound_of = {r["sdf_record"].strip(): r["compound"].strip() for r in csv.DictReader(fh)}
    out = {}
    with open(root / _entry(name)["fepplus_edges"], newline="") as fh:
        for row in csv.DictReader(fh):
            records = row["Lig 1"].strip(), row["Lig 2"].strip()
            if not all(r in compound_of for r in records):
                continue
            a, b = (compound_of[r] for r in records)
            out[frozenset((a, b))] = {
                "ligand_a": a, "ligand_b": b, "records": list(records), "exp_ddg": float(row["Exp. ddG (kcal/mol)"]),
                "bennett_ddg": float(row["Bennett ddG (kcal/mol)"]),
                "bennett_se": float(row["Bennett std. error (kcal/mol)"]),
                "ccc_ddg": float(row["CCC ddG (kcal/mol)"]), "ccc_se": float(row["CCC std. error (kcal/mol)"])}
    return out


def oriented(value: float | None, published: dict, a: str, b: str) -> float | None:
    """A ddG of the published edge in the direction a -> b."""
    if value is None:
        return None
    return value if (published["ligand_a"], published["ligand_b"]) == (a, b) else -value


def plan_rows(name: str, rcfg) -> tuple[list, list[tuple[str, str]], str, list[dict]]:
    """The ligands, the edges, how they were obtained and, for every edge, its mapping and LOMAP score and whether
    the published map of the set has it: what `cufep benchmark plan` shows before anything runs."""
    from cufep.rbfe.run import edges_of, ligands_of, mapped_edges

    ligands = ligands_of(rcfg)
    edges, how = edges_of(rcfg, ligands, lambda _line: None)
    published = published_edges(name)
    rows = mapped_edges(rcfg, ligands, edges)
    for r in rows:
        r["in_published_map"] = frozenset((r["ligand_a"], r["ligand_b"])) in published
    return ligands, edges, how, rows


def plan_lines(name: str, tier: str, rcfg) -> list[str]:
    ligands, edges, how, rows = plan_rows(name, rcfg)
    published = published_edges(name)
    names = {lig.name for lig in ligands}
    in_map = sum(r["in_published_map"] for r in rows)
    of_these = sum(1 for key in published if key <= names)
    lines = [f"benchmark {name}, tier {tier}, relative: {len(ligands)} ligand(s), {len(edges)} edge(s) ({how})",
             f"{in_map} of the {len(edges)} edges are in the published map of the set, which has {of_these} edges "
             "between these ligands",
             "", "| edge | atoms mapped | unique to A | unique to B | LOMAP score | in the published map |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['ligand_a']} -> {r['ligand_b']} | {r['mapped']} | {r['unique_a']} | {r['unique_b']} | "
              f"{r['lomap_score']:.2f} | {'yes' if r['in_published_map'] else 'no'} |" for r in rows]
    return lines


# ---- the report: the run against experiment, FEP+ and OpenFE -------------------------------------------------

#: the statistics of the dG of the compounds against experiment, in the order of the report
DG_STATISTICS = (("rmse", "RMSE (kcal/mol)"), ("mue", "MUE (kcal/mol)"),
                 ("pairwise_rmse", "pairwise ddG RMSE (kcal/mol)"), ("pearson_r2", "Pearson R^2"),
                 ("kendall_tau", "Kendall tau"))
#: those of the ddG of the edges against the experimental ddG
EDGE_STATISTICS = (("rmse", "RMSE (kcal/mol)"), ("mue", "MUE (kcal/mol)"))
#: how the published values that the report compares with were computed
FEPPLUS_SETTINGS = ("the published FEP+ predictions of the public benchmark (Ross et al., Commun. Chem. 2023; "
                    "`ligands.csv` and `edge_predictions_*_out.csv` of the set)")
OPENFE_SETTINGS = ("the published results of the OpenFE benchmark (IndustryBenchmarks2024 v1.0.0, Baumann et al.): "
                   "OpenFE 1.0.1, relative hybrid topology protocol, 11 lambda windows of 5 ns, three repeats per edge "
                   "averaged, the LOMAP network of the set")
#: the published uncertainty of the ddG of an OpenFE edge (``stdev DDG_bind``): sqrt(SD_complex^2 + SD_solvent^2) of
#: the standard deviations of the three repeats of each leg (so for all 22 edges of TYK2), not the standard deviation
#: of three ddG (ejm_49 -> ejm_50: 0.1364 against 0.0587 kcal/mol; review of PR #21)
OPENFE_EDGE_ERROR = "the standard deviation propagated from the three repeats of the complex and of the solvent leg"
OPENFE_COLOUR = "#1baf7a"                              # categorical slot 3 of the palette of the ABFE figures
LIGAND_COLUMNS = ("ligand", "exp", "cufep", "cufep_error", "fepplus", "openfe", "openfe_error", "cufep_minus_exp",
                  "fepplus_minus_exp", "openfe_minus_exp")
EDGE_COLUMNS = ("edge", "ligand_a", "ligand_b", "lomap_score", "exp_ddg", "cufep_ddg", "cufep_error",
                "cufep_network_ddg", "in_published_map", "fepplus_bennett_ddg", "fepplus_bennett_error",
                "fepplus_ccc_ddg", "in_openfe_network", "openfe_ddg", "openfe_error", "windows_complex",
                "windows_solvent", "lowest_overlap_complex", "lowest_overlap_solvent", "backend_complex",
                "backend_solvent", "not_reliable", "fewest_independent_samples_complex",
                "fewest_independent_samples_solvent", "few_independent_samples", "why_no_ddg")


def openfe_results(name: str) -> tuple[dict[str, dict], dict[frozenset, dict]]:
    """The published OpenFE results of the set: the dG of every compound and the ddG of every edge (in the direction
    it was computed in, B - A), each with its published uncertainty; empty where the project published none."""
    entry, root = _entry(name), data_root()
    if not entry.get("openfe_dg"):
        return {}, {}
    dg, edges = {}, {}
    with open(root / entry["openfe_dg"], newline="") as fh:
        for row in csv.DictReader(fh):
            dg[row["ligand name"].strip()] = {"dg": float(row["DG (kcal/mol)"]),
                                              "error": float(row["uncertainty (kcal/mol)"])}
    with open(root / entry["openfe_edges"], newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = row["ligand_A"].strip(), row["ligand_B"].strip()
            edges[frozenset((a, b))] = {"ligand_a": a, "ligand_b": b, "ddg": float(row["DDG_bind (kcal/mol)"]),
                                        "error": float(row["stdev DDG_bind (kcal/mol)"])}
    return dg, edges


def _lomap_score(run_dir: Path, edge: dict) -> float | None:
    """The LOMAP score of the mapping of an edge: as the summary of the run records it; for a summary that does not
    (a version before the record), from the prepared edge, where the directory has it. A report made again from
    the files that are kept of a run (``benchmarks/results/``, without the prepared edges) has the scores (review
    of PR #21: all of them were empty there)."""
    import json

    if edge.get("lomap_score") is not None:
        return float(edge["lomap_score"])
    f = run_dir / "edges" / edge["edge"] / "prep" / "prep.json"
    if not f.exists():
        return None
    score = json.loads(f.read_text()).get("lomap_score")
    return None if score is None else float(score)


def check_run_is_of_the_set(name: str, ligands: list[str], run_dir: Path, every: bool = True) -> None:
    """Refuses, before a report writes or removes anything, a run directory that is not of the benchmark ``name``:
    one with ligands that are not compounds of the set (``every``: all of them must be; else at least one, for
    the absolute report, which lists ligands it does not know apart). Review of PR #21: `cufep benchmark report
    tyk2 <the A2A results>` skipped all 17 compounds, wrote an empty report under the name of TYK2 over the
    report of A2A and removed its figures. A run of some compounds of the set (the smoke tier) is one of the
    set."""
    reference = read_reference(name)
    unknown = [ligand for ligand in ligands if ligand not in reference]
    if (unknown and every) or (ligands and len(unknown) == len(ligands)):
        shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise ValueError(f"{run_dir} is not a run of the benchmark {name}: its ligands {shown} are not compounds of "
                         "that set. Nothing was written.")


def _why_no_ddg(e: dict) -> str:
    if e.get("inconsistent_legs"):
        return "legs made under different conditions: " + "; ".join(e["inconsistent_legs"])
    for r in e.get("repeats", []):
        if r.get("no_standard_error"):
            return "no standard error of the " + " and ".join(r["no_standard_error"]) + " leg"
        if r.get("missing"):
            return "no result of the " + " and ".join(r["missing"]) + " leg"
    return "no result"


#: what the table says of a check whose numbers the summary of a run does not hold
NOT_RECORDED = "not recorded"


def samples_level(run_dir: Path) -> float | None:
    """The level of independent samples of the analysis of a run (``analysis.min_effective_samples`` of its
    recorded configuration), or None where the directory does not say."""
    import yaml

    f = run_dir / "resolved_config.yaml"
    if not f.exists():
        return None
    level = ((yaml.safe_load(f.read_text()) or {}).get("analysis") or {}).get("min_effective_samples")
    return float(level) if isinstance(level, (int, float)) and not isinstance(level, bool) else None


def independent_samples_of(edge: dict, level: float | None) -> tuple[dict, list[str] | None]:
    """(the window with the fewest independent samples of each leg, the legs below the level of the analysis) of
    an edge of a summary: as the summary records them; for a summary of a version before that record, from the
    repeats of its ddG, which hold the numbers of their legs, against ``level``. The legs are None, not recorded,
    where the summary gives no ground for them: no repeat with the numbers, or no level (second review of PR #21:
    a report of such a summary said that no edge had few independent samples, of an edge with 31)."""
    if "few_samples" in edge and "fewest_independent_samples" in edge:
        return edge["fewest_independent_samples"] or {}, list(edge["few_samples"] or [])
    if edge.get("ddg") is None:
        return {}, []                                       # no ddG: no repeat was taken
    counted = [r["fewest_independent_samples"] for r in edge.get("repeats", [])
               if "ddg" in r and isinstance(r.get("fewest_independent_samples"), dict)]
    fewest = {leg: min((c[leg] for c in counted if c.get(leg) is not None), default=None)
              for leg in ("complex", "solvent")}
    if not counted or level is None or any(value is None for value in fewest.values()):
        return fewest, None
    return fewest, [leg for leg in ("complex", "solvent") if fewest[leg] < level]


def _rows(name: str, run_dir: Path, summary: dict) -> tuple[list[dict], list[dict], bool]:
    level = samples_level(run_dir)
    reference = read_reference(name)
    fep_edges = published_edges(name)
    of_dg, of_edges = openfe_results(name)
    dg = summary.get("dg") or {}
    shifted = dg.get("reference") == "mean of the experimental values"
    values = dg.get("values") or {}
    ligands = []
    for compound in summary["ligands"]:
        ref = reference.get(compound)
        if ref is None:
            continue
        mine = values.get(compound) if shifted else None
        of = of_dg.get(compound)
        ligands.append({
            "ligand": compound, "exp": ref["exp"], "cufep": None if mine is None else mine["dg"],
            "cufep_error": None if mine is None else mine["se"], "fepplus": ref["fepplus"],
            "openfe": None if of is None else of["dg"], "openfe_error": None if of is None else of["error"],
            "cufep_minus_exp": None if mine is None else mine["dg"] - ref["exp"],
            "fepplus_minus_exp": ref["fepplus"] - ref["exp"],
            "openfe_minus_exp": None if of is None else of["dg"] - ref["exp"]})
    edges = []
    for e in summary["edges"]:
        a, b, key = e["ligand_a"], e["ligand_b"], frozenset((e["ligand_a"], e["ligand_b"]))
        fep, of = fep_edges.get(key), of_edges.get(key)
        windows, overlap, backend = e.get("windows") or {}, e.get("lowest_overlap") or {}, e.get("backend") or {}
        exp = None if a not in reference or b not in reference else reference[b]["exp"] - reference[a]["exp"]
        network = values[b]["dg"] - values[a]["dg"] if a in values and b in values else None
        fewest, few = independent_samples_of(e, level)
        edges.append({
            "edge": e["edge"], "ligand_a": a, "ligand_b": b, "lomap_score": _lomap_score(run_dir, e),
            "exp_ddg": exp, "cufep_ddg": e.get("ddg"), "cufep_error": e.get("se"), "cufep_network_ddg": network,
            "in_published_map": fep is not None,
            "fepplus_bennett_ddg": None if fep is None else oriented(fep["bennett_ddg"], fep, a, b),
            "fepplus_bennett_error": None if fep is None else fep["bennett_se"],
            "fepplus_ccc_ddg": None if fep is None else oriented(fep["ccc_ddg"], fep, a, b),
            "in_openfe_network": of is not None if of_edges else None,
            "openfe_ddg": None if of is None else oriented(of["ddg"], of, a, b),
            "openfe_error": None if of is None else of["error"],
            "windows_complex": "/".join(str(n) for n in windows.get("complex") or []),
            "windows_solvent": "/".join(str(n) for n in windows.get("solvent") or []),
            "lowest_overlap_complex": overlap.get("complex"), "lowest_overlap_solvent": overlap.get("solvent"),
            "backend_complex": "/".join(backend.get("complex") or []),
            "backend_solvent": "/".join(backend.get("solvent") or []),
            "not_reliable": "; ".join(e.get("not_reliable") or []),
            # the window with the fewest independent samples of each leg, and the legs below the level of the
            # analysis (review of PR #21: the table gave the standard error alone)
            "fewest_independent_samples_complex": fewest.get("complex"),
            "fewest_independent_samples_solvent": fewest.get("solvent"),
            "few_independent_samples": NOT_RECORDED if few is None else "; ".join(few),
            "why_no_ddg": "" if e.get("ddg") is not None else _why_no_ddg(e)})
    return ligands, edges, shifted


def _statistics(series: dict[str, list], exp: list, keys, centre: bool = False) -> dict[str, dict[str, str]]:
    """Each method's statistics against experiment, with 95 % bootstrap intervals (four or more values), as text.
    ``centre``: of each method's values with their mean set to the experimental mean of the values compared, in the
    estimate and in every resample of its interval (:func:`cufep.benchmarks.stats.bootstrap`)."""
    from cufep.benchmarks.stats import bootstrap, metrics
    from cufep.benchmarks.stats import centred as centred_values

    out = {}
    for method, pred in series.items():
        m = metrics(centred_values(pred, exp) if centre else pred, exp)
        ci = bootstrap(pred, exp, centre=centre) if len(pred) >= 4 else {}
        out[method] = {}
        for key, _label in keys:
            if m[key] is None:
                out[method][key] = "N/A"
            else:
                out[method][key] = f"{m[key]:.2f}" + (f" [{ci[key][0]:.2f}, {ci[key][1]:.2f}]" if key in ci else "")
    return out


def _flags(e: dict) -> str:
    """The flags of an edge in the table: the legs that do not meet a level of the two checks of the analysis."""
    return "; ".join(f"{name}: {e[key].replace('; ', ', ')}" for key, name in (
        ("not_reliable", "NOT RELIABLE"), ("few_independent_samples", "few independent samples")) if e[key])


def _f(value, fmt: str = ".2f", missing: str = "") -> str:
    return missing if value is None else format(value, fmt)


def _cufep_settings(run_dir: Path) -> str:
    import yaml

    f = run_dir / "resolved_config.yaml"
    if not f.exists():
        return "cuFEP: the settings of the run are not in this directory"
    cfg = yaml.safe_load(f.read_text()) or {}
    sampling, schedule = cfg.get("sampling") or {}, cfg.get("lambda_schedule") or {}
    membrane = " in a POPC bilayer" if (cfg.get("membrane") or {}).get("enabled") else ""
    return (f"cuFEP: `cufep rbfe` with the settings of the set's relative example{membrane}, "
            f"{sampling.get('production_ns')} ns of production per lambda window, "
            f"calculations per edge: {sampling.get('repeats', 1)}, lambda windows {schedule.get('mode')}, "
            f"backend {(cfg.get('backend') or {}).get('name')} (`resolved_config.yaml`)")


def report(name: str, run_dir: str | Path, write: bool = True) -> str:
    """Compare a relative benchmark run with experiment, FEP+ and OpenFE; returns Markdown. Written as the run is
    now: edges without a ddG are listed with the reason, and the files of an earlier state are replaced."""
    import json

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    check_run_is_of_the_set(name, list(summary["ligands"]), run_dir)
    network = run_dir / "network.json"
    edges_from = json.loads(network.read_text()).get("edges_from") if network.exists() else None
    ligands, edges, shifted = _rows(name, run_dir, summary)
    of_dg, of_edges = openfe_results(name)
    entry = _entry(name)
    done = [e for e in edges if e["cufep_ddg"] is not None]
    how = edges_from or "how the edges were obtained is not recorded"
    lines = [f"# cuFEP relative benchmark report: {name}", "", entry["title"], "",
             f"Edges with a ddG: {len(done)} of {len(edges)} ({how}); "
             f"compounds with a dG: {sum(r['cufep'] is not None for r in ligands)} of {len(ligands)}.", "",
             "Compared with:", "", f"- {_cufep_settings(run_dir)}", f"- FEP+: {FEPPLUS_SETTINGS}",
             f"- OpenFE: {OPENFE_SETTINGS}" if of_dg else
             "- OpenFE: the OpenFE benchmark published no result for this set; its columns are empty", "",
             "The table of the compounds gives the dG as estimated and as published: cuFEP's maximum likelihood "
             "estimate from its edges referenced to the mean of the experimental values of the compounds of this run, "
             "FEP+ and OpenFE referenced to that of their own sets. The statistics and the figure of the dG centre "
             "every method on the compounds compared. The numbers are given without a verdict.", ""]
    for note in summary.get("notes") or []:
        lines.append(f"Note of the run: {note}")
    if not shifted:
        lines += ["", "The run has no dG referenced to the mean of the experimental values: the compounds are compared "
                      "by their ddG alone.", ""]
    compared = [r for r in ligands if r["cufep"] is not None and (not of_dg or r["openfe"] is not None)]
    if compared:
        exp = [r["exp"] for r in compared]
        series = {"cuFEP": [r["cufep"] for r in compared], "FEP+": [r["fepplus"] for r in compared]}
        if of_dg:
            series["OpenFE"] = [r["openfe"] for r in compared]
        stats = _statistics(series, exp, DG_STATISTICS, centre=True)
        lines += ["## dG of the compounds against experiment", "",
                  f"The same {len(compared)} compounds for every method, each method centred on them (its mean set to "
                  "the mean of their experimental values, as cinnabar's plots of dG centre them), so that RMSE and MUE "
                  "compare like with like; pairwise RMSE, R^2 and Kendall tau do not depend on it. 95 % bootstrap "
                  "intervals over compounds in brackets, each resample centred on its own compounds.", "",
                  "| statistic | " + " | ".join(series) + " |", "|---|" + "---|" * len(series)]
        lines += [f"| {label} | " + " | ".join(stats[m][key] for m in series) + " |" for key, label in DG_STATISTICS]
        lines.append("")
    network = ("cuFEP from the network",) if any(e["cufep_network_ddg"] is not None for e in done) else ()
    groups = [(f"all {len(done)} edges of the run", done, ("cuFEP", *network))]
    in_map = [e for e in done if e["in_published_map"]]
    groups.append((f"the {len(in_map)} edges also in the published map", in_map,
                   ("cuFEP", "FEP+ Bennett", "FEP+ cycle closure")))
    if of_edges:
        in_of = [e for e in done if e["in_openfe_network"]]
        groups.append((f"the {len(in_of)} edges also in OpenFE's network", in_of, ("cuFEP", "OpenFE")))
    column = {"cuFEP": "cufep_ddg", "cuFEP from the network": "cufep_network_ddg",
              "FEP+ Bennett": "fepplus_bennett_ddg", "FEP+ cycle closure": "fepplus_ccc_ddg", "OpenFE": "openfe_ddg"}
    lines += ["## ddG of the edges against experiment", "",
              "cuFEP: the ddG of the edge itself; from the network: the difference of the two dG of the estimate. "
              "FEP+ Bennett: the ddG of the edge as published; cycle closure: the published ddG after the correction "
              f"of the cycles. OpenFE: the mean of its three repeats (± {OPENFE_EDGE_ERROR}, as published). Every "
              "ddG in the direction of the edge of this run; 95 % bootstrap intervals over edges in brackets.", "",
              "| edges | n | statistic | " + " | ".join(column) + " |", "|---|---|---|" + "---|" * len(column)]
    for label, members, methods in groups:
        members = [e for e in members if e["exp_ddg"] is not None and all(e[column[m]] is not None for m in methods)]
        if not members:
            lines.append(f"| {label} | 0 | | " + " | ".join("" for _ in column) + " |")
            continue
        stats = _statistics({m: [e[column[m]] for e in members] for m in methods}, [e["exp_ddg"] for e in members],
                            EDGE_STATISTICS)
        for key, stat in EDGE_STATISTICS:
            lines.append(f"| {label} | {len(members)} | {stat} | "
                         + " | ".join(stats[m][key] if m in methods else "" for m in column) + " |")
    lines += ["", "## Compounds (kcal/mol)", "",
              "| compound | experiment | cuFEP | SE | FEP+ | OpenFE | cuFEP - exp | FEP+ - exp | OpenFE - exp |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ligands, key=lambda r: r["exp"]):
        lines.append(f"| {r['ligand']} | {r['exp']:.2f} | {_f(r['cufep'], '.2f', 'no dG')} | {_f(r['cufep_error'])} | "
                     f"{r['fepplus']:.2f} | {_f(r['openfe'], '.2f', '-')} | {_f(r['cufep_minus_exp'], '+.2f')} | "
                     f"{r['fepplus_minus_exp']:+.2f} | {_f(r['openfe_minus_exp'], '+.2f', '-')} |")
    lines += ["", "## Edges (kcal/mol)", "",
              "Windows, lowest overlap of neighbouring windows (all iterations), fewest independent samples in a "
              "window and backend, each of the complex and the solvent leg; SE: standard error of cuFEP's ddG; "
              f"OpenFE: mean of its three repeats ± {OPENFE_EDGE_ERROR}. Flags, the two checks of the analysis: NOT "
              "RELIABLE names a leg whose overlap is below its level (Klimovich et al. 2015); few independent "
              "samples names a leg with a window below the level of the analysis (50 by default, Mey et al. 2020), "
              "for which more sampling is advisable. `-`: the edge is not in that network.", "",
              "| edge | LOMAP score | experiment | cuFEP | SE | from the network | FEP+ Bennett | FEP+ cycle closure | "
              "OpenFE | windows | lowest overlap | fewest independent samples | backend | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in edges:
        ddg = _f(e["cufep_ddg"], '+.2f') if e["cufep_ddg"] is not None else f"no ddG ({e['why_no_ddg']})"
        of = "" if not of_edges else _f(e["openfe_ddg"], "+.2f", "-") + (
            f" ± {e['openfe_error']:.2f}" if e["openfe_error"] is not None else "")
        lines.append(
            f"| {e['ligand_a']} -> {e['ligand_b']} | {_f(e['lomap_score'])} | {_f(e['exp_ddg'], '+.2f')} | {ddg} | "
            f"{_f(e['cufep_error'])} | {_f(e['cufep_network_ddg'], '+.2f')} | "
            f"{_f(e['fepplus_bennett_ddg'], '+.2f', '-')} | {_f(e['fepplus_ccc_ddg'], '+.2f', '-')} | {of} | "
            f"{e['windows_complex']}, {e['windows_solvent']} | "
            f"{_f(e['lowest_overlap_complex'], '.3f')}, {_f(e['lowest_overlap_solvent'], '.3f')} | "
            f"{_f(e['fewest_independent_samples_complex'], '.0f')}, "
            f"{_f(e['fewest_independent_samples_solvent'], '.0f')} | "
            f"{e['backend_complex']}, {e['backend_solvent']} | {_flags(e)} |")
    stock = [e["edge"] for e in edges if any(b and b != "cupotential" for b in
                                              (e["backend_complex"], e["backend_solvent"]))]
    flagged = {name: [e["edge"] for e in edges if e[key] and e[key] != NOT_RECORDED] for key, name in (
        ("not_reliable", "NOT RELIABLE"), ("few_independent_samples", "few independent samples"))}
    lines += ["", "Edges with a flag: " + "; ".join(
        f"{name}: {', '.join(found) if found else 'none'}" for name, found in flagged.items()) + "."]
    unknown = [e["edge"] for e in edges if e["few_independent_samples"] == NOT_RECORDED]
    if unknown:
        lines += ["", f"Few independent samples: {NOT_RECORDED} for {', '.join(unknown)} (the summary of the run "
                      "does not hold the numbers of their legs, or the directory not the level of its analysis)."]
    lines += ["", "Legs on another backend than the cuPotential plugin (the fallback to stock OpenMM, whose reason "
                  f"the log of the leg gives): {', '.join(stock) if stock else 'none'}."]
    closures = summary.get("cycle_closures") or []
    if closures:
        lines += ["", "## Cycle closures (kcal/mol)", "", "| cycle | sum of ddG | standard error |", "|---|---|---|"]
        lines += [f"| {' -> '.join(c['ligands'])} | {c['sum_kcal_mol']:+.2f} | {c['se_kcal_mol']:.2f} |"
                  for c in closures]
    text = "\n".join(lines) + "\n"
    if write:
        _save(run_dir, name, text, ligands, edges, bool(of_dg))
    return text


def _save(run_dir: Path, name: str, text: str, ligands: list[dict], edges: list[dict], with_openfe: bool) -> None:
    from cufep.files import publish_text, publish_with

    publish_text(run_dir / "rbfe_report.md", text)
    for path, rows, columns in ((run_dir / "rbfe_report_ligands.csv", ligands, LIGAND_COLUMNS),
                                (run_dir / "rbfe_report_edges.csv", edges, EDGE_COLUMNS)):
        def table(fh, rows=rows, columns=columns) -> None:
            w = csv.DictWriter(fh, fieldnames=list(columns), restval="")
            w.writeheader()
            w.writerows(rows)

        publish_with(path, table, newline="")
    dg_rows = [r for r in ligands if r["cufep"] is not None]
    edge_rows = [e for e in edges if e["cufep_ddg"] is not None and e["exp_ddg"] is not None]
    for path, rows, plot in ((run_dir / "rbfe_report.png", dg_rows, _plot_dg),
                             (run_dir / "rbfe_report_edges.png", edge_rows, _plot_edges)):
        path.unlink(missing_ok=True)                   # never the figure of an earlier state of the run
        if rows:
            plot(rows, path, name, with_openfe)


def _scatter(rows: list[dict], path: Path, title: str, exp_key: str, series: list[tuple], xlabel: str) -> None:
    """Calculated against experiment: one series per method (key, error key, label, colour, marker, filled), the
    diagonal and a band of 1 kcal/mol, the axes holding every value with its whole error bar."""
    from cufep.benchmarks.runner import GUIDE_COLOUR, _pyplot, _save_figure
    from cufep.benchmarks.stats import metrics

    plt = _pyplot()
    if plt is None:
        return
    values = [r[exp_key] for r in rows]
    for key, err, *_ in series:
        values += [r[key] + s * (r.get(err) or 0.0) for r in rows if r.get(key) is not None for s in (-1.0, 1.0)]
    lo, hi = min(values) - 1.0, max(values) + 1.0
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([lo, hi], [lo, hi], color=GUIDE_COLOUR, lw=1)
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.92", zorder=0)
    for key, err, label, colour, marker, filled in series:
        part = [r for r in rows if r.get(key) is not None]
        if not part:
            continue
        m = metrics([r[key] for r in part], [r[exp_key] for r in part])
        stat = (f"{label}, {len(part)}: RMSE {m['rmse']:.2f}, Kendall τ "
                + ("N/A" if m["kendall_tau"] is None else f"{m['kendall_tau']:.2f}"))
        ax.errorbar([r[exp_key] for r in part], [r[key] for r in part],
                    yerr=[r.get(err) or 0.0 for r in part] if err else None, fmt=marker, ms=6, color=colour,
                    mfc=colour if filled else "white", capsize=2, elinewidth=1, label=stat, zorder=3 if filled else 2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(xlabel.replace("experiment", "calculated"))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.12))
    _save_figure(fig, path)
    plt.close(fig)


def _plot_dg(rows: list[dict], path: Path, name: str, with_openfe: bool) -> None:
    """The dG of every method, each centred on the compounds it is shown for, as the statistics centre them."""
    from cufep.benchmarks.runner import CUFEP_COLOUR, FEPPLUS_COLOUR
    from cufep.benchmarks.stats import centred

    rows = [dict(r) for r in rows]
    for key in ("cufep", "fepplus", "openfe"):
        shown = [r for r in rows if r.get(key) is not None]
        if shown:
            for r, v in zip(shown, centred([r[key] for r in shown], [r["exp"] for r in shown])):
                r[key] = float(v)
    series = [("cufep", "cufep_error", "cuFEP (relative; standard error)", CUFEP_COLOUR, "o", True),
              ("fepplus", None, "FEP+ (published)", FEPPLUS_COLOUR, "s", False)]
    if with_openfe:
        series.append(("openfe", "openfe_error", "OpenFE (published)", OPENFE_COLOUR, "^", False))
    _scatter(rows, path, f"{name}: dG of the compounds (each method centred)", "exp", series,
             "experiment dG (kcal/mol)")


def _plot_edges(rows: list[dict], path: Path, name: str, with_openfe: bool) -> None:
    from cufep.benchmarks.runner import CUFEP_COLOUR, FEPPLUS_COLOUR

    series = [("cufep_ddg", "cufep_error", "cuFEP (the edge; standard error)", CUFEP_COLOUR, "o", True),
              ("fepplus_bennett_ddg", "fepplus_bennett_error", "FEP+ Bennett (published map)", FEPPLUS_COLOUR, "s",
               False)]
    if with_openfe:
        series.append(("openfe_ddg", "openfe_error", "OpenFE (published; mean of 3 repeats, SD propagated from the "
                       "legs)", OPENFE_COLOUR, "^", False))
    _scatter(rows, path, f"{name}: ddG of the edges", "exp_ddg", series, "experiment ddG (kcal/mol)")
