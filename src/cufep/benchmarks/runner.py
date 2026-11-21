"""Run and report the built-in benchmarks."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path

import yaml

from cufep.benchmarks.stats import bootstrap, metrics
from cufep.config.schema import Config

TIERS = ("smoke", "subset", "full")
SMOKE_SAMPLING = {"pre_equilibration": {"complex": {"nvt_ns": 0.02, "npt_ns": 0.02, "md_ns": 0.3},
                                        "solvent": {"nvt_ns": 0.01, "npt_ns": 0.02, "md_ns": 0.05}},
                  "equilibration_ns": 0.05, "production_ns": 0.1, "repeats": 1}


def data_root() -> Path:
    env = os.environ.get("CUFEP_DATA")
    if env and (Path(env) / "benchmarks").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "benchmarks" / "THIRD_PARTY").exists():
            return parent
    return Path.cwd()


def load_registry() -> dict:
    return yaml.safe_load((Path(__file__).parent / "registry.yaml").read_text())


def read_reference(name: str) -> dict[str, dict]:
    entry = load_registry()[name]
    with open(data_root() / entry["reference"]) as fh:
        return {r["compound"]: {"exp": float(r["exp_dg_kcal_mol"]), "fepplus": float(r["fepplus_dg_kcal_mol"])}
                for r in csv.DictReader(fh)}


def _shorten(into: dict, values: dict) -> None:
    """Shorten the nested settings ``into`` to ``values``, key by key: a number becomes the smaller of
    the two (a shorter setting of the example stays), a per-leg setting gets the smaller number for
    each leg, and a section is never replaced as a whole."""
    for key, value in values.items():
        here = into.get(key)
        if isinstance(value, dict) and isinstance(here, dict):
            _shorten(here, value)
        elif isinstance(here, dict) and all(isinstance(v, (int, float)) for v in here.values()):
            into[key] = {leg: min(v, value) for leg, v in here.items()}
        elif isinstance(here, (int, float)) and not isinstance(here, bool):
            into[key] = min(here, value)
        else:
            into[key] = copy.deepcopy(value)


def example_notices(name: str) -> tuple[Path, list[str]]:
    """The configuration file of the set's example and the notices about it (settings of earlier
    versions, a threshold without its overlap matrix): a benchmark run reads that file, so it
    says of it what ``cufep run`` says."""
    from cufep.config.schema import load_config_with_notices

    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    path = data_root() / registry[name]["example"]
    return path, load_config_with_notices(path)[1]


def benchmark_config(name: str, tier: str, out_dir: str | Path | None = None) -> Config:
    """Configuration of a benchmark tier.

    The settings are those of the set's example (``example`` in the registry): everything in its
    configuration but the input files, which are the set's, the selection of ligands and the
    output directory. So a set runs as its example runs, and the example's ligand is one of the
    set's. ``full`` runs every ligand, ``subset`` five; ``smoke`` runs one and also shortens the
    sampling and the MD from which a restraint is chosen (never lengthens a setting of the example).
    """
    from cufep.config.schema import load_config

    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    if tier not in TIERS:
        raise KeyError(f"unknown tier '{tier}' (available: {', '.join(TIERS)})")
    entry, root = registry[name], data_root()
    data = load_config(root / entry["example"]).model_dump(mode="json")
    data["system"].update(protein=str(root / entry["protein"]), ligand=str(root / entry["ligands"]))
    data["output"]["dir"] = str(out_dir or Path("runs") / f"benchmark_{name}_{tier}")
    data["system"]["ligand_names"] = {"smoke": entry["subset"][:1], "subset": list(entry["subset"]), "full": None}[tier]
    if tier == "smoke":
        _shorten(data, {"sampling": SMOKE_SAMPLING,
                        "lambda_schedule": {"adaptive": {"window_ps": 10, "equilibration_ps": 0}},
                        "membrane": {"equilibration_ns": 0.5},
                        # the MD from which a restraint is chosen (restraints.selection: interactions)
                        "restraints": {"interactions": {"length_ns": 0.3, "discard_ns": 0.1}}})
    return Config.model_validate(data)


#: what a ligand's result can say about its legs: key in the summary and how the report names it.
#: The same list is behind the flags of the report, the columns of summary.csv and the result line.
#: The verdicts on the analysis are the two checks with the literature's levels
#: (:data:`cufep.analysis.mbar.QUALITY_CHECKS`: overlap, uncorrelated samples); checks of cuFEP's own
#: that earlier versions made (halves, BAR against MBAR, NOT CONVERGED) are not shown, also not from
#: results those versions wrote (owner, 2026-11-20).
LEG_FLAGS = (("failed_legs", "FAILED"), ("inconsistent_legs", "LEGS MADE UNDER DIFFERENT CONDITIONS"),
             ("not_reliable", "NOT RELIABLE (overlap)"), ("old_analysis", "old analysis"),
             ("few_samples", "few independent samples"),
             ("origin_not_recorded", "origin of files not recorded"))
#: the flag of a binding free energy whose legs were not all analysed by this version of the
#: analysis: the summary names the versions of its legs (analysis_versions) since version 5; one
#: that does not was made before
NOT_CHECKED = "checks of this version not made"


def not_checked(ligand_summary: dict) -> str | None:
    """The flag of a binding free energy whose legs were analysed otherwise than by this version, or None."""
    from cufep.analysis.mbar import ANALYSIS_VERSION

    if "dg_bind" not in ligand_summary:
        return None
    versions = ligand_summary.get("analysis_versions")
    if versions is None:
        return f"{NOT_CHECKED} (analysed by an earlier version of cuFEP)"
    if list(versions) == [ANALYSIS_VERSION]:
        return None
    shown = ", ".join("not recorded" if v is None else str(v) for v in versions)
    return f"{NOT_CHECKED} (analysed by version{'s' if len(versions) > 1 else ''} {shown} of the analysis)"


def flags_of(ligand_summary: dict) -> list[str]:
    """The flags of one ligand, in the order of :data:`LEG_FLAGS`.

    A result that an earlier version of cuFEP analysed is not shown as if it had passed the
    checks of this one: it carries the flag that the checks of this version were not made.
    """
    out = []
    for key, name in LEG_FLAGS:
        items = ligand_summary.get(key) or []
        if items:
            out.append(f"{name}: " + ", ".join(str(item).split(":")[0] if key in ("old_analysis", "origin_not_recorded")
                                               else str(item) for item in items))
    flag = not_checked(ligand_summary)
    if flag:
        out.append(flag)
    return out


def overlap_places(ligand_summary: dict) -> str:
    """Where the legs of a NOT RELIABLE result have neighbouring lambda windows below the level of
    overlap: leg, pair of windows, stage, lambda values and the overlap, among all iterations and
    among the samples of the estimate (``overlap_below_level`` of the summary). A summary of an
    earlier version names the legs only; empty if no leg is NOT RELIABLE."""
    places = ligand_summary.get("overlap_below_level")
    if places is None:
        legs = ligand_summary.get("not_reliable") or []
        return "; ".join(f"{leg} (the windows are not in this summary.json)" for leg in legs)
    grouped: dict[str, list[str]] = {}
    for p in places:
        pair, lam = p.get("windows"), p.get("lambda")
        where = p["leg"] if not pair else f"{p['leg']}, lambda windows {pair[0]} and {pair[1]}"
        if p.get("stage") is not None and lam:
            where += f" ({p['stage']} stage, lambda {lam[0]:g} and {lam[1]:g})"
        value = "not defined" if p.get("overlap") is None else f"{p['overlap']:.3f}"
        grouped.setdefault(where, []).append(f"{value} among {p['of']}")
    return "; ".join(f"{where}: {', '.join(values)}" for where, values in grouped.items())


#: the statistics of the agreement with experiment, in the order of the report
STATISTICS = (("mean_signed_error", "mean signed error (kcal/mol)"), ("rmse", "RMSE, raw (kcal/mol)"),
              ("mue", "MUE, raw (kcal/mol)"), ("centred_rmse", "RMSE after removing the mean error (kcal/mol)"),
              ("pairwise_rmse", "pairwise ddG RMSE (kcal/mol)"), ("kendall_tau", "Kendall tau"),
              ("spearman", "Spearman rho"), ("pearson_r", "Pearson r"), ("pearson_r2", "Pearson R^2"))
#: those of them shown for each group of the breakdown by the check of overlap
BREAKDOWN_STATISTICS = ("mean_signed_error", "rmse", "mue", "centred_rmse")


def overlap_level(summary: list[dict]) -> str:
    """The level of overlap below which the results of a run are NOT RELIABLE, as their analysis
    recorded it (``analysis_record``; the default where a summary of an earlier version has none)."""
    from cufep.config.schema import AnalysisConfig

    levels = {float((s.get("analysis_record") or {}).get("min_overlap", AnalysisConfig().min_overlap))
              for s in summary}
    return " or ".join(f"{level:g}" for level in sorted(levels)) or f"{AnalysisConfig().min_overlap:g}"


def breakdown_by_overlap(rows: list[dict], unreliable: set[str], unchecked: set[str], level: str) -> list[str]:
    """Markdown lines: the agreement with experiment of the NOT RELIABLE ligands and of the others,
    each for cuFEP and for FEP+ on the same ligands. Both groups are also in the statistics of all
    ligands; this shows how the results below the level of overlap compare. Ligands whose check was
    not made by this version (``unchecked``) are named, never counted as meeting the level."""
    groups = [("NOT RELIABLE", [r for r in rows if r["ligand"] in unreliable]),
              ("the others", [r for r in rows if r["ligand"] not in unreliable])]
    out = ["## Agreement with experiment by the check of overlap", ""]
    not_made = sorted((r for r in rows if r["ligand"] in unchecked), key=lambda r: r["exp"])
    note = ("" if not not_made else " The checks of this version were not made for "
            + ", ".join(r["ligand"] for r in not_made) + " (analysed by an earlier version).")
    if not groups[0][1]:
        return out + [f"No ligand with a binding free energy is NOT RELIABLE.{note}"]
    names = ", ".join(r["ligand"] for r in sorted(groups[0][1], key=lambda r: r["exp"]))
    out += [f"NOT RELIABLE (overlap below {level} in a leg, Klimovich et al. 2015): {names}. They are counted in "
            "the statistics above; here they and the other ligands are compared with experiment apart, cuFEP and "
            "FEP+ on the same ligands (N/A: not defined for so few ligands). As above, the raw FEP+ statistics are "
            f"not like for like; the RMSE after removing the mean error is.{note}", ""]
    header, rule = "| statistic |", "|---|"
    cells: dict[str, str] = {key: "" for key in BREAKDOWN_STATISTICS}
    for group, members in groups:
        header += f" cuFEP, {group} ({len(members)}) | FEP+, the same ligands |"
        rule += "---|---|"
        exp = [r["exp"] for r in members]
        for method in ("cufep", "fepplus"):
            m = metrics([r[method] for r in members], exp) if members else None
            for key in BREAKDOWN_STATISTICS:
                value = None if m is None else m[key]
                cells[key] += " N/A |" if value is None else f" {value:.2f} |"
    labels = dict(STATISTICS)
    return out + [header, rule] + [f"| {labels[key]} |{cells[key]}" for key in BREAKDOWN_STATISTICS]


#: the columns of benchmark_report.csv: cufep_error is the standard error of one calculation
#: (dg_bind_error; empty for a result of before it), cufep_std the standard deviation over repeats;
#: cufep_minus_exp and fepplus_minus_exp the signed errors; overlap_below_level where a NOT RELIABLE
#: result is below the level (:func:`overlap_places`)
REPORT_COLUMNS = ("ligand", "exp", "fepplus", "cufep", "cufep_error", "cufep_std", "n_repeats", "cufep_minus_exp",
                  "fepplus_minus_exp", "windows_complex", "windows_solvent",
                  "windows_added_complex", "windows_added_solvent", "lowest_overlap", "lowest_overlap_used",
                  "overlap_below_level", "flags", "warnings")


def report(name: str, run_dir: str | Path, write: bool = True) -> str:
    """Compare a run with experiment and with FEP+; returns Markdown.

    The report, its table (``benchmark_report.csv``) and its figure are those of the run as it
    is now: ligands without a binding free energy are listed with the reason, and a report with
    no ligand at all is written as such, so that the files of an earlier, better state of the
    directory are not left to be read as the current result.
    """
    run_dir = Path(run_dir)
    reference = read_reference(name)
    summary = json.loads((run_dir / "summary.json").read_text())
    rows, without = [], []
    unreliable: set[str] = set()                       # ligands whose result is NOT RELIABLE (overlap)
    unchecked: set[str] = set()                        # ligands whose checks this version did not make
    level = overlap_level(summary)
    for s in summary:
        ref = reference.get(s["ligand"])
        if ref is None:
            continue
        if s.get("dg_bind") is None:                   # none, or not a number (written as null), as run.py reads it
            without.append({"ligand": s["ligand"], "exp": ref["exp"], "fepplus": ref["fepplus"], "cufep": None,
                            "cufep_error": None, "cufep_std": None, "n_repeats": s.get("n_repeats", 0),
                            "cufep_minus_exp": None, "fepplus_minus_exp": ref["fepplus"] - ref["exp"],
                            "overlap_below_level": overlap_places(s),
                            "flags": "; ".join(flags_of(s) or ["no complete repeat"]),
                            "warnings": len(s.get("warnings", []))})
            continue
        if s.get("not_reliable"):
            unreliable.add(s["ligand"])
        if not_checked(s):
            unchecked.add(s["ligand"])
        first = (s.get("repeats") or [{}])[0]
        added = first.get("lambda_windows_added") or {}
        flags = flags_of(s)
        rows.append(
            {
                "ligand": s["ligand"],
                "exp": ref["exp"],
                "fepplus": ref["fepplus"],
                "cufep": s["dg_bind"],
                "cufep_error": s.get("dg_bind_error"),
                "cufep_std": s.get("dg_bind_std"),
                "n_repeats": s["n_repeats"],
                "cufep_minus_exp": s["dg_bind"] - ref["exp"],
                "fepplus_minus_exp": ref["fepplus"] - ref["exp"],
                "windows_complex": (first.get("lambda_windows") or {}).get("complex"),
                "windows_solvent": (first.get("lambda_windows") or {}).get("solvent"),
                "windows_added_complex": None if added.get("complex") is None else sum(added["complex"].values()),
                "windows_added_solvent": None if added.get("solvent") is None else sum(added["solvent"].values()),
                "lowest_overlap": s.get("min_overlap"),
                "lowest_overlap_used": s.get("min_overlap_used"),
                "overlap_below_level": overlap_places(s),
                "flags": "; ".join(flags),
                "warnings": len(s.get("warnings", [])),
            }
        )
    title = load_registry()[name]["title"]
    lines = [f"# cuFEP benchmark report: {name}", "", title, "",
             f"Ligands with a result: {len(rows)} of {len(summary)} run ({len(reference)} in the set).", ""]
    if without:
        lines += ["## Ligands without a binding free energy", "", "| ligand | experiment | why |", "|---|---|---|"]
        lines += [f"| {r['ligand']} | {r['exp']:.2f} | {r['flags']} |" for r in sorted(without, key=lambda r: r["exp"])]
        lines.append("")

    def save(text: str) -> None:
        from cufep.files import publish_text, publish_with

        publish_text(run_dir / "benchmark_report.md", text)

        def table(fh) -> None:
            w = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS), restval="")
            w.writeheader()
            w.writerows(rows + without)

        publish_with(run_dir / "benchmark_report.csv", table, newline="")
        for figure, plot in ((run_dir / "benchmark_report.png", _plot),
                             (run_dir / "benchmark_report_ligands.png", _plot_ligands)):
            figure.unlink(missing_ok=True)             # never the figure of an earlier state of the run
            if rows:
                plot(rows, figure, name, unreliable, level)

    if not rows:
        lines.append("No ligand has a binding free energy.")
        text = "\n".join(lines) + "\n"
        if write:
            save(text)
        return text
    lines += ["## Per ligand (kcal/mol)", "",
              "Every ligand with a binding free energy, NOT RELIABLE ones included (marked). Standard error: of one "
              "calculation, from its own trajectories (- for a result of an earlier version that did not record it). "
              "SD over repeats: only where several independent calculations were run. The last two columns are the "
              "signed errors against experiment.", "",
              "| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ | cuFEP - experiment "
              "| FEP+ - experiment |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["exp"]):
        sd = "" if r["cufep_std"] is None else f"{r['cufep_std']:.2f}"
        se = "-" if r["cufep_error"] is None else f"{r['cufep_error']:.2f}"
        mark = " (NOT RELIABLE)" if r["ligand"] in unreliable else ""
        lines.append(f"| {r['ligand']}{mark} | {r['exp']:.2f} | {r['cufep']:.2f} | {se} | {sd} | {r['n_repeats']} | "
                     f"{r['fepplus']:.2f} | {r['cufep_minus_exp']:+.2f} | {r['fepplus_minus_exp']:+.2f} |")

    def count(value) -> str:
        return "" if value is None else str(value)

    lines += [
        "",
        "## Lambda windows and checks of the analysis",
        "",
        "Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap "
        "of neighbouring windows in the production (from all iterations with equal numbers of samples per "
        "window, and among the samples the free energy is estimated from), and the legs that do not meet a "
        "level of the checks (first repeat). The checks are the two of the literature: NOT RELIABLE (overlap "
        f"below {level}, Klimovich et al. 2015) says that the free energy of a leg rests on too little overlap of "
        "neighbouring windows; few independent samples (fewer than 50 "
        "in a window, Mey et al. 2020) says that more sampling is advisable. A result that an earlier version "
        "of cuFEP analysed carries the flag that the checks of this version were not made; they are made when "
        "its legs are analysed again. The last column says where a NOT RELIABLE result is below the level: "
        "leg, pair of lambda windows, stage, lambda values and overlap.",
        "",
        "| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags "
        "| overlap below the level |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["exp"]):
        def cell_windows(leg: str, r=r) -> str:
            n, extra = r[f"windows_{leg}"], r[f"windows_added_{leg}"]
            return count(n) + ("" if extra is None else f" ({extra})")
        low = "" if r["lowest_overlap"] is None else f"{r['lowest_overlap']:.3f}"
        used = "" if r["lowest_overlap_used"] is None else f"{r['lowest_overlap_used']:.3f}"
        lines.append(f"| {r['ligand']} | {cell_windows('complex')} | {cell_windows('solvent')} | {low} | {used} | "
                     f"{r['flags']} | {r['overlap_below_level']} |")
    scheduled = [r[f"windows_added_{leg}"] for r in rows for leg in ("complex", "solvent")
                 if r[f"windows_added_{leg}"] is not None]
    if scheduled:
        lines += [
            "",
            f"Adaptive lambda scheduling added windows in {sum(1 for n in scheduled if n > 0)} of "
            f"{len(scheduled)} legs. {sum(1 for r in rows if r['flags'])} of {len(rows)} ligands have a flag.",
        ]
    exp = [r["exp"] for r in rows]
    table = {"cuFEP": [r["cufep"] for r in rows], "FEP+": [r["fepplus"] for r in rows]}
    stats = {k: (metrics(v, exp), bootstrap(v, exp) if len(rows) >= 4 else {}) for k, v in table.items()}

    def cell(method: str, key: str) -> str:
        m, ci = stats[method]
        if m[key] is None:
            return "N/A"                               # not defined for these ligands
        text = f"{m[key]:.2f}"
        if key in ci:
            text += f" [{ci[key][0]:.2f}, {ci[key][1]:.2f}]"
        return text

    lines += ["", "## Agreement with experiment", "",
              "Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands "
              "of this report: every ligand with a binding free energy, NOT RELIABLE ones included. FEP+ values "
              "come from relative calculations that were shifted as a whole to the experimental mean of the "
              "published set; over a part of that set their mean signed error is not zero. Only the rows below "
              "the first three compare the two methods like for like.", "",
              "| statistic | cuFEP | FEP+ |", "|---|---|---|"]
    for key, label in STATISTICS:
        lines.append(f"| {label} | {cell('cuFEP', key)} | {cell('FEP+', key)} |")
    if len(rows) < 8:
        lines += ["", f"Only {len(rows)} ligands: the intervals are wide and rank statistics are indicative at best."]
    lines += ["", *breakdown_by_overlap(rows, unreliable, unchecked, level)]
    text = "\n".join(lines) + "\n"
    if write:
        save(text)
    return text


def error_bar(row: dict) -> float:
    """The error bar of a ligand in the figure: the standard deviation over repeats where several
    were run, else the standard error of the one calculation; 0 where neither is recorded."""
    if row.get("n_repeats", 0) > 1 and row.get("cufep_std") is not None:
        return float(row["cufep_std"])
    return float(row["cufep_error"]) if row.get("cufep_error") is not None else 0.0


def error_bar_label(rows: list[dict]) -> str:
    kinds = {"SD over repeats" if r.get("n_repeats", 0) > 1 and r.get("cufep_std") is not None
             else "standard error" if r.get("cufep_error") is not None else "none recorded" for r in rows}
    return "error bars: " + ", ".join(sorted(kinds))


def figure_range(rows: list[dict], margin: float = 1.0) -> tuple[float, float]:
    """The range of both axes of the figure: every experimental value, every FEP+ value and every
    cuFEP value with its whole error bar, and a margin (review of PR 1: a prediction far from the
    experiment fell outside the axes, with its error bar)."""
    values = [r["exp"] for r in rows] + [r["fepplus"] for r in rows if r.get("fepplus") is not None]
    values += [r["cufep"] + sign * error_bar(r) for r in rows for sign in (-1.0, 1.0)]
    return min(values) - margin, max(values) + margin


#: the colours of the figures (categorical slots 1 and 2 of a palette checked for colour-vision
#: deficiencies; experiment is drawn in neutral ink) and the marker of each series
CUFEP_COLOUR, FEPPLUS_COLOUR, EXPERIMENT_COLOUR, GUIDE_COLOUR = "#2a78d6", "#eb6834", "#0b0b0b", "0.55"


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    return plt


def _save_figure(fig, path: Path) -> None:
    from cufep.files import publish_with

    # the canvas grows to take the legend and its statistics; the axes keep their size
    publish_with(path, lambda fh: fig.savefig(fh, format=Path(path).suffix.lstrip(".") or "png", bbox_inches="tight"),
                 mode="wb")


def figure_statistics(rows: list[dict]) -> list[str]:
    """The lines of statistics a figure shows under its legend: for cuFEP and for FEP+ over all the
    ligands of the figure (NOT RELIABLE ones included), the correlation coefficients and the errors,
    as the tables of the report give them (N/A where not defined)."""
    exp = [r["exp"] for r in rows]

    def value(m: dict, key: str) -> str:
        return "N/A" if m[key] is None else f"{m[key]:.2f}"

    lines = []
    for method, column in (("cuFEP", "cufep"), ("FEP+", "fepplus")):
        m = metrics([r[column] for r in rows], exp)
        lines.append(f"{method}, {len(rows)} ligands: Pearson r {value(m, 'pearson_r')}, Kendall \u03c4 "
                     f"{value(m, 'kendall_tau')}, RMSE {value(m, 'rmse')}, RMSE after removing the mean error "
                     f"{value(m, 'centred_rmse')} kcal/mol")
    return lines


def _legend(ax, rows: list[dict], **kwargs) -> None:
    """The legend of a figure with the lines of :func:`figure_statistics` below its series."""
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    extra = figure_statistics(rows)
    handles += [Line2D([], [], linestyle="none", marker="") for _ in extra]
    ax.legend(handles, labels + extra, frameon=False, fontsize=8, **kwargs)


def _cufep_series(ax, xs: dict, ys: dict, rows: list[dict], unreliable: set[str], level: str,
                  bars: bool = True) -> None:
    """cuFEP's points: filled circles, the NOT RELIABLE ones open; each series names its error bars."""
    for part, what, kind in (([r for r in rows if r["ligand"] not in unreliable], "cuFEP (absolute", {}),
                             ([r for r in rows if r["ligand"] in unreliable],
                              f"cuFEP, NOT RELIABLE (overlap below {level}", {"mfc": "white"})):
        if part:
            ax.errorbar([xs[r["ligand"]] for r in part], [ys[r["ligand"]] for r in part],
                        yerr=[error_bar(r) for r in part] if bars else None, fmt="o", ms=6, color=CUFEP_COLOUR,
                        capsize=2, elinewidth=1, label=f"{what}; {error_bar_label(part)})", zorder=3, **kind)


def _plot(rows: list[dict], path: Path, name: str, unreliable: set[str], level: str) -> None:
    """The figure of the report: every ligand with a result against experiment, the NOT RELIABLE
    ones as open circles, FEP+ as open squares; the diagonal and a band of 1 kcal/mol."""
    plt = _pyplot()
    if plt is None:
        return
    lo, hi = figure_range(rows)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([lo, hi], [lo, hi], color=GUIDE_COLOUR, lw=1)
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.92", zorder=0)
    exp = {r["ligand"]: r["exp"] for r in rows}
    _cufep_series(ax, exp, {r["ligand"]: r["cufep"] for r in rows}, rows, unreliable, level)
    ax.plot(list(exp.values()), [r["fepplus"] for r in rows], "s", ms=6, mfc="none", color=FEPPLUS_COLOUR,
            label="FEP+ (relative, shifted)", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("experiment (kcal/mol)")
    ax.set_ylabel("calculated (kcal/mol)")
    ax.set_title(f"{name}: binding free energies")
    _legend(ax, rows, loc="upper left", bbox_to_anchor=(0.0, -0.12))
    _save_figure(fig, path)
    plt.close(fig)


def _plot_ligands(rows: list[dict], path: Path, name: str, unreliable: set[str], level: str) -> None:
    """The figure of the report ligand by ligand, in the order of experiment: above, the binding
    free energy of experiment, cuFEP (with its error bar) and FEP+; below, the signed errors of
    cuFEP and FEP+ against experiment, with zero and a band of 1 kcal/mol."""
    plt = _pyplot()
    if plt is None:
        return
    ordered = sorted(rows, key=lambda r: r["exp"])
    x = {r["ligand"]: i for i, r in enumerate(ordered)}
    shift = 0.18                                       # experiment in the middle, the two methods beside it
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(max(6.0, 0.48 * len(ordered) + 2.0), 6.4),
                                      dpi=150, gridspec_kw={"height_ratios": [3, 2]})
    left = {k: v - shift for k, v in x.items()}
    right = [v + shift for v in x.values()]
    top.plot(list(x.values()), [r["exp"] for r in ordered], "D", ms=6, color=EXPERIMENT_COLOUR, label="experiment",
             zorder=4)
    _cufep_series(top, left, {r["ligand"]: r["cufep"] for r in rows}, rows, unreliable, level)
    top.plot(right, [r["fepplus"] for r in ordered], "s", ms=6, mfc="none", color=FEPPLUS_COLOUR,
             label="FEP+ (relative, shifted)", zorder=2)
    top.set_ylabel("binding free energy (kcal/mol)")
    top.set_title(f"{name}: binding free energy of each ligand", pad=96)
    _legend(top, rows, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    width = len(ordered) - 0.5
    bottom.fill_between([-0.5, width], [-1, -1], [1, 1], color="0.92", zorder=0)
    bottom.axhline(0.0, color=GUIDE_COLOUR, lw=1)
    _cufep_series(bottom, left, {r["ligand"]: r["cufep"] - r["exp"] for r in rows}, rows, unreliable, level)
    bottom.plot(right, [r["fepplus"] - r["exp"] for r in ordered], "s", ms=6, mfc="none", color=FEPPLUS_COLOUR,
                zorder=2)
    bottom.set_ylabel("minus experiment (kcal/mol)")
    bottom.set_xticks(list(x.values()))
    bottom.set_xticklabels([r["ligand"] for r in ordered], rotation=60, ha="right", fontsize=8)
    bottom.set_xlim(-0.5, width)
    for ax in (top, bottom):
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    _save_figure(fig, path)
    plt.close(fig)


def freeze_schedules(name: str, run_dir: str | Path, dest: str | Path | None = None) -> Path:
    """Copy the lambda schedules of a finished run to ``dest``, by default ``RUN_DIR/frozen_schedules``.

    A later run with ``lambda_schedule: {mode: fixed, file: <that directory>}`` then uses
    exactly these states instead of searching again. The default lies in the run directory, which
    the user can write; the data of the official image (``CUFEP_DATA=/opt/cufep``) belong to root
    and are read only (review of PR 1: the default under ``benchmarks/<name>/`` there could not be
    written by the user of the image).
    """
    import shutil

    from cufep.files import publish

    run_dir = Path(run_dir)
    dest = Path(dest) if dest else run_dir / "frozen_schedules"
    n = 0
    for sched in sorted(run_dir.glob("*/repeat_*/*/schedule.yaml")):
        leg, repeat, ligand = sched.parent.name, sched.parent.parent.name.split("_")[1], sched.parent.parent.parent.name
        target = dest / ligand / f"{leg}_r{repeat}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        publish(target, lambda tmp, sched=sched: shutil.copy(sched, tmp))
        n += 1
    if n == 0:
        raise FileNotFoundError(f"no schedule.yaml found under {run_dir}")
    return dest
