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
SMOKE_SAMPLING = {"nvt_ns": 0.02, "npt_ns": 0.02, "pre_equilibration_ns": 0.3, "solvent_nvt_ns": 0.01,
                  "solvent_npt_ns": 0.02, "solvent_pre_equilibration_ns": 0.05, "equilibration_ns": 0.05,
                  "production_ns": [0.1], "repeats": 1}


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
                  "output": {"dir": str(out_dir or Path("runs") / f"benchmark_{name}_{tier}")}}
    if tier == "smoke":
        data["system"]["ligand_names"] = entry["subset"][:1]
        data["sampling"] = dict(SMOKE_SAMPLING)
        data["lambda_schedule"] = {"pilot_ps_per_window": 10}
        data["membrane"] = {"equilibration_ns": 0.5, "max_equilibration_ns": 0.5}
    elif tier == "subset":
        data["system"]["ligand_names"] = list(entry["subset"])
    return Config.model_validate(data)


def report(name: str, run_dir: str | Path, write: bool = True) -> str:
    """Compare a finished run with experiment and with FEP+; returns Markdown."""
    run_dir = Path(run_dir)
    reference = read_reference(name)
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = []
    for s in summary:
        ref = reference.get(s["ligand"])
        if ref is None or "dg_bind" not in s:
            continue
        rows.append({"ligand": s["ligand"], "exp": ref["exp"], "fepplus": ref["fepplus"], "cufep": s["dg_bind"],
                     "cufep_std": s.get("dg_bind_std"), "n_repeats": s["n_repeats"],
                     "warnings": len(s.get("warnings", []))})
    title = load_registry()[name]["title"]
    lines = [f"# cuFEP benchmark report: {name}", "", title, "",
             f"Ligands with a result: {len(rows)} of {len(summary)} run ({len(reference)} in the set).", ""]
    if not rows:
        lines.append("No finished ligand yet.")
        return "\n".join(lines) + "\n"
    lines += ["## Per ligand (kcal/mol)", "", "| ligand | experiment | cuFEP | SD over repeats | repeats | FEP+ |",
              "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["exp"]):
        sd = "" if r["cufep_std"] is None else f"{r['cufep_std']:.2f}"
        lines.append(f"| {r['ligand']} | {r['exp']:.2f} | {r['cufep']:.2f} | {sd} | {r['n_repeats']} | {r['fepplus']:.2f} |")
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
              "Values in brackets are 95 % bootstrap intervals over ligands. FEP+ values come from relative "
              "calculations shifted to the experimental mean, so only the rows below the first three compare "
              "the two methods like for like.", "",
              "| statistic | cuFEP | FEP+ |", "|---|---|---|"]
    labels = [("mean_signed_error", "mean signed error (kcal/mol)"), ("rmse", "RMSE, raw (kcal/mol)"),
              ("mue", "MUE, raw (kcal/mol)"), ("centred_rmse", "RMSE after removing the mean error (kcal/mol)"),
              ("pairwise_rmse", "pairwise ddG RMSE (kcal/mol)"), ("kendall_tau", "Kendall tau"),
              ("spearman", "Spearman rho"), ("pearson_r2", "Pearson R^2")]
    for key, label in labels:
        fep = "0 by construction" if key == "mean_signed_error" else cell("FEP+", key)
        lines.append(f"| {label} | {cell('cuFEP', key)} | {fep} |")
    if len(rows) < 8:
        lines += ["", f"Only {len(rows)} ligands: the intervals are wide and rank statistics are indicative at best."]
    text = "\n".join(lines) + "\n"
    if write:
        (run_dir / "benchmark_report.md").write_text(text)
        with open(run_dir / "benchmark_report.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        _plot(rows, run_dir / "benchmark_report.png", name)
    return text


def _plot(rows: list[dict], path: Path, name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    exp = [r["exp"] for r in rows]
    lo, hi = min(exp) - 2.5, max(exp) + 2.5
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1)
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="0.9", zorder=0)
    ax.errorbar(exp, [r["cufep"] for r in rows], yerr=[r["cufep_std"] or 0.0 for r in rows], fmt="o",
                color="#1f6f8b", label="cuFEP (absolute)", capsize=2)
    ax.plot(exp, [r["fepplus"] for r in rows], "s", mfc="none", color="#c0392b", label="FEP+ (relative, shifted)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("experiment (kcal/mol)"); ax.set_ylabel("calculated (kcal/mol)")
    ax.set_title(f"{name}: binding free energies"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
