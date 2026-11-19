"""Run and report the built-in benchmarks."""
from __future__ import annotations

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


def benchmark_config(name: str, tier: str, out_dir: str | Path | None = None) -> Config:
    """Configuration of a benchmark tier: only the inputs and, for ``smoke``, shortened sampling."""
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(registry)})")
    if tier not in TIERS:
        raise KeyError(f"unknown tier '{tier}' (available: {', '.join(TIERS)})")
    entry, root = registry[name], data_root()
    data: dict = {"system": {"protein": str(root / entry["protein"]), "ligand": str(root / entry["ligands"])},
                  "membrane": {"enabled": bool(entry.get("membrane", False))},
                  "output": {"dir": str(out_dir or Path("runs") / f"benchmark_{name}_{tier}")}}
    if tier == "smoke":
        data["system"]["ligand_names"] = entry["subset"][:1]
        data["sampling"] = dict(SMOKE_SAMPLING)
        data["lambda_schedule"] = {"adaptive": {"window_ps": 10, "equilibration_ps": 0}}
        data["membrane"].update(equilibration_ns=0.5)
    elif tier == "subset":
        data["system"]["ligand_names"] = list(entry["subset"])
    return Config.model_validate(data)


#: what a ligand's result can say about its legs: key in the summary and how the report names it.
#: The same list is behind the flags of the report, the columns of summary.csv and the result line.
LEG_FLAGS = (("failed_legs", "FAILED"), ("inconsistent_legs", "LEGS MADE UNDER DIFFERENT CONDITIONS"),
             ("not_reliable", "NOT RELIABLE (overlap)"), ("old_analysis", "old analysis"),
             ("few_samples", "few independent samples"), ("halves_differ", "halves differ"),
             ("halves_not_compared", "halves not compared"), ("estimators_disagree", "BAR and MBAR differ"),
             ("origin_not_recorded", "origin of files not recorded"))


#: what a ligand's result of an earlier version of cuFEP says with keys that this version no longer
#: writes: the verdicts of checks that were replaced. They are shown as what that version said;
#: they are not columns of summary.csv, which only this version writes.
EARLIER_FLAGS = (("not_converged", "NOT CONVERGED by the check of an earlier version"),
                 ("convergence_not_assessed", "convergence not assessed by an earlier version"))
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
    checks of this one: it carries the verdicts of that version (:data:`EARLIER_FLAGS`) and the
    flag that the checks of this version were not made.
    """
    out = []
    for key, name in LEG_FLAGS + EARLIER_FLAGS:
        items = ligand_summary.get(key) or []
        if items:
            out.append(f"{name}: " + ", ".join(str(item).split(":")[0] if key in ("old_analysis", "origin_not_recorded")
                                               else str(item) for item in items))
    flag = not_checked(ligand_summary)
    if flag:
        out.append(flag)
    return out


#: the columns of benchmark_report.csv: cufep_error is the standard error of one calculation
#: (dg_bind_error; empty for a result of before it), cufep_std the standard deviation over repeats
REPORT_COLUMNS = ("ligand", "exp", "fepplus", "cufep", "cufep_error", "cufep_std", "n_repeats", "windows_complex",
                  "windows_solvent",
                  "windows_added_complex", "windows_added_solvent", "lowest_overlap", "lowest_overlap_used", "flags",
                  "warnings")


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
    for s in summary:
        ref = reference.get(s["ligand"])
        if ref is None:
            continue
        if "dg_bind" not in s:
            without.append({"ligand": s["ligand"], "exp": ref["exp"], "fepplus": ref["fepplus"], "cufep": None,
                            "cufep_error": None, "cufep_std": None, "n_repeats": s.get("n_repeats", 0),
                            "flags": "; ".join(flags_of(s) or ["no complete repeat"]),
                            "warnings": len(s.get("warnings", []))})
            continue
        first = (s.get("repeats") or [{}])[0]
        added = first.get("lambda_windows_added") or {}
        flags = flags_of(s)
        rows.append({"ligand": s["ligand"], "exp": ref["exp"], "fepplus": ref["fepplus"], "cufep": s["dg_bind"],
                     "cufep_error": s.get("dg_bind_error"), "cufep_std": s.get("dg_bind_std"), "n_repeats": s["n_repeats"],
                     "windows_complex": (first.get("lambda_windows") or {}).get("complex"),
                     "windows_solvent": (first.get("lambda_windows") or {}).get("solvent"),
                     "windows_added_complex": None if added.get("complex") is None else sum(added["complex"].values()),
                     "windows_added_solvent": None if added.get("solvent") is None else sum(added["solvent"].values()),
                     "lowest_overlap": s.get("min_overlap"), "lowest_overlap_used": s.get("min_overlap_used"),
                     "flags": "; ".join(flags),
                     "warnings": len(s.get("warnings", []))})
    title = load_registry()[name]["title"]
    lines = [f"# cuFEP benchmark report: {name}", "", title, "",
             f"Ligands with a result: {len(rows)} of {len(summary)} run ({len(reference)} in the set).", ""]
    if without:
        lines += ["## Ligands without a binding free energy", "", "| ligand | experiment | why |", "|---|---|---|"]
        lines += [f"| {r['ligand']} | {r['exp']:.2f} | {r['flags']} |" for r in sorted(without, key=lambda r: r["exp"])]
        lines.append("")

    def save(text: str) -> None:
        (run_dir / "benchmark_report.md").write_text(text)
        with open(run_dir / "benchmark_report.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS), restval="")
            w.writeheader()
            w.writerows(rows + without)
        figure = run_dir / "benchmark_report.png"
        figure.unlink(missing_ok=True)                 # never the figure of an earlier state of the run
        if rows:
            _plot(rows, figure, name)

    if not rows:
        lines.append("No ligand has a binding free energy.")
        text = "\n".join(lines) + "\n"
        if write:
            save(text)
        return text
    lines += ["## Per ligand (kcal/mol)", "",
              "Standard error: of one calculation, from its own trajectories (- for a result of an earlier version "
              "that did not record it). SD over repeats: only where several independent calculations were run.", "",
              "| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["exp"]):
        sd = "" if r["cufep_std"] is None else f"{r['cufep_std']:.2f}"
        se = "-" if r["cufep_error"] is None else f"{r['cufep_error']:.2f}"
        lines.append(f"| {r['ligand']} | {r['exp']:.2f} | {r['cufep']:.2f} | {se} | {sd} | {r['n_repeats']} | "
                     f"{r['fepplus']:.2f} |")

    def count(value) -> str:
        return "" if value is None else str(value)

    lines += ["", "## Lambda windows and checks of the analysis", "",
              "Windows per leg, how many of them adaptive lambda scheduling added to its start, the lowest overlap "
              "of neighbouring windows in the production (from all iterations with equal numbers of samples per "
              "window, and among the samples the free energy is estimated from), and the legs that do not meet a "
              "level of the checks (first repeat). NOT RELIABLE (overlap below 0.03) is the one that makes a result "
              "unusable; the others say that more sampling is advisable. A result that an earlier version of cuFEP "
              "analysed carries the verdicts of that version and the flag that the checks of this version were not "
              "made; they are made when its legs are analysed again.", "",
              "| ligand | windows complex (added) | windows solvent (added) | lowest overlap | of the samples used | flags |",
              "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["exp"]):
        def cell_windows(leg: str) -> str:
            n, extra = r[f"windows_{leg}"], r[f"windows_added_{leg}"]
            return count(n) + ("" if extra is None else f" ({extra})")
        low = "" if r["lowest_overlap"] is None else f"{r['lowest_overlap']:.3f}"
        used = "" if r["lowest_overlap_used"] is None else f"{r['lowest_overlap_used']:.3f}"
        lines.append(f"| {r['ligand']} | {cell_windows('complex')} | {cell_windows('solvent')} | {low} | {used} | "
                     f"{r['flags']} |")
    scheduled = [r[f"windows_added_{leg}"] for r in rows for leg in ("complex", "solvent")
                 if r[f"windows_added_{leg}"] is not None]
    if scheduled:
        lines += ["", f"Adaptive lambda scheduling added windows in {sum(1 for n in scheduled if n > 0)} of "
                      f"{len(scheduled)} legs. {sum(1 for r in rows if r['flags'])} of {len(rows)} ligands have a flag."]
    exp = [r["exp"] for r in rows]
    table = {"cuFEP": [r["cufep"] for r in rows], "FEP+": [r["fepplus"] for r in rows]}
    stats = {k: (metrics(v, exp), bootstrap(v, exp) if len(rows) >= 4 else {}) for k, v in table.items()}

    def cell(method: str, key: str) -> str:
        m, ci = stats[method]
        text = f"{m[key]:.2f}"
        if key in ci:
            text += f" [{ci[key][0]:.2f}, {ci[key][1]:.2f}]"
        return text

    lines += ["", "## Agreement with experiment", "",
              "Values in brackets are 95 % bootstrap intervals over ligands. All numbers are for the ligands "
              "of this report. FEP+ values come from relative calculations that were shifted as a whole to the "
              "experimental mean of the published set; over a part of that set their mean signed error is not "
              "zero. Only the rows below the first three compare the two methods like for like.", "",
              "| statistic | cuFEP | FEP+ |", "|---|---|---|"]
    labels = [("mean_signed_error", "mean signed error (kcal/mol)"), ("rmse", "RMSE, raw (kcal/mol)"),
              ("mue", "MUE, raw (kcal/mol)"), ("centred_rmse", "RMSE after removing the mean error (kcal/mol)"),
              ("pairwise_rmse", "pairwise ddG RMSE (kcal/mol)"), ("kendall_tau", "Kendall tau"),
              ("spearman", "Spearman rho"), ("pearson_r2", "Pearson R^2")]
    for key, label in labels:
        lines.append(f"| {label} | {cell('cuFEP', key)} | {cell('FEP+', key)} |")
    if len(rows) < 8:
        lines += ["", f"Only {len(rows)} ligands: the intervals are wide and rank statistics are indicative at best."]
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


def _plot(rows: list[dict], path: Path, name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    exp = [r["exp"] for r in rows]
    lo, hi = figure_range(rows)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1)
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.9", zorder=0)
    ax.errorbar(exp, [r["cufep"] for r in rows], yerr=[error_bar(r) for r in rows], fmt="o",
                color="#1f6f8b", label="cuFEP (absolute; " + error_bar_label(rows) + ")", capsize=2)
    ax.plot(exp, [r["fepplus"] for r in rows], "s", mfc="none", color="#c0392b", label="FEP+ (relative, shifted)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("experiment (kcal/mol)"); ax.set_ylabel("calculated (kcal/mol)")
    ax.set_title(f"{name}: binding free energies"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def freeze_schedules(name: str, run_dir: str | Path, dest: str | Path | None = None) -> Path:
    """Copy the lambda schedules of a finished run to ``benchmarks/<name>/schedules``.

    A later run with ``lambda_schedule: {mode: fixed, file: <that directory>}`` then uses
    exactly these states instead of searching again.
    """
    import shutil

    run_dir = Path(run_dir)
    dest = Path(dest) if dest else data_root() / "benchmarks" / name / "schedules"
    n = 0
    for sched in sorted(run_dir.glob("*/repeat_*/*/schedule.yaml")):
        leg, repeat, ligand = sched.parent.name, sched.parent.parent.name.split("_")[1], sched.parent.parent.parent.name
        target = dest / ligand / f"{leg}_r{repeat}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sched, target)
        n += 1
    if n == 0:
        raise FileNotFoundError(f"no schedule.yaml found under {run_dir}")
    return dest
