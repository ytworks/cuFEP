"""Tables of the accuracy pages of the wiki (docs/wiki/Accuracy.md, Accuracy-ja.md) from a recorded
benchmark run (benchmarks/results/<set>/benchmark_report.csv), in English and Japanese.

The statistics are those of `cufep benchmark report` (the same functions, the same bootstrap seed),
so the pages show the numbers of the report. Both languages get the same numbers; only the headers
and labels are translated. Run inside the image:

    scripts/in_docker.sh python spikes/wiki_tables.py tyk2 a2a

Output: spikes/out/wiki_tables/<set>.<lang>.md, and the figures of the run copied to
docs/wiki/images/<set>_scatter.png and <set>_ligands.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from cufep.benchmarks.runner import BREAKDOWN_STATISTICS, STATISTICS
from cufep.benchmarks.stats import bootstrap, metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes" / "out" / "wiki_tables"

LABELS = {
    "en": dict(STATISTICS),
    "ja": {"mean_signed_error": "平均符号誤差 (kcal/mol)", "rmse": "RMSE、生 (kcal/mol)", "mue": "MUE、生 (kcal/mol)",
           "centred_rmse": "平均誤差を除いた RMSE (kcal/mol)", "pairwise_rmse": "ペアワイズ ddG RMSE (kcal/mol)",
           "kendall_tau": "Kendall τ", "spearman": "Spearman ρ", "pearson_r": "Pearson r",
           "pearson_r2": "Pearson R²"},
}
WORDS = {
    "en": {"statistic": "statistic", "same": "FEP+, the same ligands", "nr": "NOT RELIABLE", "others": "the others",
           "per_ligand": ("ligand", "experiment", "cuFEP", "standard error", "cuFEP - experiment", "FEP+",
                          "FEP+ - experiment", "windows complex / solvent", "lowest overlap", "flags"),
           "without": ("ligand", "experiment", "why")},
    "ja": {"statistic": "統計量", "same": "FEP+、同じリガンド", "nr": "NOT RELIABLE", "others": "それ以外",
           "per_ligand": ("リガンド", "実験値", "cuFEP", "標準誤差", "cuFEP − 実験値", "FEP+", "FEP+ − 実験値",
                          "窓 complex / solvent", "最小 overlap", "フラグ"),
           "without": ("リガンド", "実験値", "理由")},
}


#: the published per-target results of Felis (ByteDance Seed), arXiv:2603.22274 version 3
#: (2026-11-09), appendix tables "Felis ABFE statistics ..." and "FEP+ 20 ns RBFE prediction metrics":
#: R^2, Spearman rho, Kendall tau, offset b = <exp - pred>, offset-adjusted RMSE and MAE (kcal/mol),
#: correct order ratio (pairs whose experimental values differ by more than 2 kcal/mol). Each Felis
#: value per ligand is the mean of three independent calculations; per-ligand values are not published.
FELIS = {
    "tyk2": {"Felis, 5 ns x 3, ByteFF25 AM1-BCC": (0.69, 0.82, 0.68, 0.04, 0.82, 0.56, 0.97),
             "Felis, 3 ns x 3, ByteFF25 AM1-BCC": (0.64, 0.83, 0.67, 0.23, 0.89, 0.64, 0.97),
             "Felis, 5 ns x 3, ByteFF25 ABCG2": (0.39, 0.66, 0.48, 0.09, 1.06, 0.76, 0.91),
             "FEP+ as the paper gives it": (0.86, 0.93, 0.81, 0.00, 0.47, 0.37, 1.00)},
    "a2a": {"Felis, 5 ns x 3, ByteFF25 AM1-BCC": (0.55, 0.81, 0.65, 2.74, 1.16, 0.88, 1.00),
            "Felis, 3 ns x 3, ByteFF25 AM1-BCC": (0.53, 0.75, 0.57, 2.98, 1.20, 0.92, 1.00),
            "Felis, 5 ns x 3, ByteFF25 ABCG2": (0.57, 0.87, 0.72, 2.04, 1.07, 0.75, 0.94),
            "FEP+ as the paper gives it": (0.50, 0.73, 0.59, 0.16, 1.08, 0.79, 1.00)},
}
FELIS_LABELS = {"ja": {"Felis, 5 ns x 3, ByteFF25 AM1-BCC": "Felis、5 ns × 3 回、ByteFF25 AM1-BCC",
                       "Felis, 3 ns x 3, ByteFF25 AM1-BCC": "Felis、3 ns × 3 回、ByteFF25 AM1-BCC",
                       "Felis, 5 ns x 3, ByteFF25 ABCG2": "Felis、5 ns × 3 回、ByteFF25 ABCG2",
                       "FEP+ as the paper gives it": "FEP+ (論文の表の値)"}}
FELIS_HEADER = {"en": ("method", "ligands", "R^2", "Spearman rho", "Kendall tau", "offset b", "RMSE after offset",
                       "MAE after offset", "correct order ratio"),
                "ja": ("手法", "リガンド", "R²", "Spearman ρ", "Kendall τ", "オフセット b", "補正後 RMSE",
                       "補正後 MAE", "順序正解率")}


def felis_metrics(pred: list[float], exp: list[float]) -> tuple:
    """The statistics of the Felis paper, by its definitions: R^2, Spearman rho, Kendall tau, the
    offset b = <exp - pred>, RMSE and MAE of pred + b, and the fraction of correctly ranked pairs
    among those whose experimental values differ by more than 2 kcal/mol (None if there is none)."""
    import numpy as np

    pred, exp = np.asarray(pred, dtype=float), np.asarray(exp, dtype=float)
    m = metrics(pred, exp)
    b = float(np.mean(exp - pred))
    err = pred + b - exp
    pairs = [(i, j) for i in range(len(exp)) for j in range(i + 1, len(exp)) if abs(exp[i] - exp[j]) > 2.0]
    correct = [np.sign(pred[i] - pred[j]) == np.sign(exp[i] - exp[j]) for i, j in pairs]
    return (m["pearson_r2"], m["spearman"], m["kendall_tau"], b, float(np.sqrt(np.mean(err ** 2))),
            float(np.mean(np.abs(err))), float(np.mean(correct)) if pairs else None)


def felis_table(name: str, lang: str) -> str:
    """cuFEP (this run) and FEP+ (the published predictions) by the definitions of the Felis paper,
    beside the rows that paper publishes for the same target."""
    rows, _ = read_rows(name)
    exp = [float(r["exp"]) for r in rows]

    def line(label: str, n, values) -> str:
        cells = ["N/A" if v is None else f"{v:.2f}".replace("-0.00", "0.00") for v in values]
        return f"| {label} | {n} | " + " | ".join(cells) + " |"

    this_run = {"en": "cuFEP, this run (5 ns x 1)", "ja": "cuFEP、この計算 (5 ns × 1 回)"}[lang]
    fep = {"en": "FEP+, computed here from the published predictions", "ja": "FEP+、公開予測値からここで計算"}[lang]
    out = ["| " + " | ".join(FELIS_HEADER[lang]) + " |", "|" + "---|" * len(FELIS_HEADER[lang])]
    out.append(line(this_run, len(rows), felis_metrics([float(r["cufep"]) for r in rows], exp)))
    out.append(line(fep, len(rows), felis_metrics([float(r["fepplus"]) for r in rows], exp)))
    for label, values in FELIS[name].items():
        out.append(line(FELIS_LABELS.get(lang, {}).get(label, label), "", values))
    return "\n".join(out) + "\n"


def number(text: str) -> float | None:
    return float(text) if text not in ("", None) else None


def read_rows(name: str) -> tuple[list[dict], list[dict]]:
    with open(ROOT / "benchmarks" / "results" / name / "benchmark_report.csv") as fh:
        rows = list(csv.DictReader(fh))
    with_result = [r for r in rows if r["cufep"] != ""]
    return with_result, [r for r in rows if r["cufep"] == ""]


def cell(m: dict, ci: dict, key: str) -> str:
    if m[key] is None:
        return "N/A"
    text = f"{m[key]:.2f}"
    return text + (f" [{ci[key][0]:.2f}, {ci[key][1]:.2f}]" if key in ci else "")


def tables(name: str, lang: str) -> str:
    rows, without = read_rows(name)
    w, labels = WORDS[lang], LABELS[lang]
    exp = [float(r["exp"]) for r in rows]
    out = []
    # the statistics of all ligands with a result (NOT RELIABLE ones included), as the report gives them
    stats = {}
    for method, column in (("cuFEP", "cufep"), ("FEP+", "fepplus")):
        values = [float(r[column]) for r in rows]
        stats[method] = (metrics(values, exp), bootstrap(values, exp) if len(rows) >= 4 else {})
    out += [f"| {w['statistic']} | cuFEP | FEP+ |", "|---|---|---|"]
    out += [f"| {labels[key]} | {cell(*stats['cuFEP'], key)} | {cell(*stats['FEP+'], key)} |" for key, _ in STATISTICS]
    # the breakdown by the check of overlap
    unreliable = [r for r in rows if "NOT RELIABLE" in r["flags"]]
    others = [r for r in rows if "NOT RELIABLE" not in r["flags"]]
    if unreliable:
        header, rule, cells = f"| {w['statistic']} |", "|---|", {k: "" for k in BREAKDOWN_STATISTICS}
        for group, members in ((w["nr"], unreliable), (w["others"], others)):
            header += f" cuFEP, {group} ({len(members)}) | {w['same']} |"
            rule += "---|---|"
            e = [float(r["exp"]) for r in members]
            for column in ("cufep", "fepplus"):
                m = metrics([float(r[column]) for r in members], e) if members else None
                for key in BREAKDOWN_STATISTICS:
                    value = None if m is None else m[key]
                    cells[key] += " N/A |" if value is None else f" {value:.2f} |"
        out += ["", header, rule] + [f"| {labels[k]} |{cells[k]}" for k in BREAKDOWN_STATISTICS]
    # every ligand
    out += ["", "| " + " | ".join(w["per_ligand"]) + " |", "|" + "---|" * len(w["per_ligand"])]
    for r in sorted(rows, key=lambda r: float(r["exp"])):
        se = "" if r["cufep_error"] == "" else f"{float(r['cufep_error']):.2f}"
        low = "" if r["lowest_overlap"] == "" else f"{float(r['lowest_overlap']):.3f}"
        flags = "; ".join(x for x in (r["flags"], r["overlap_below_level"]) if x)
        out.append(f"| {r['ligand']} | {float(r['exp']):.2f} | {float(r['cufep']):.2f} | {se} | "
                   f"{float(r['cufep_minus_exp']):+.2f} | {float(r['fepplus']):.2f} | "
                   f"{float(r['fepplus_minus_exp']):+.2f} | {r['windows_complex']} / {r['windows_solvent']} | "
                   f"{low} | {flags} |")
    if without:
        out += ["", "| " + " | ".join(w["without"]) + " |", "|---|---|---|"]
        out += [f"| {r['ligand']} | {float(r['exp']):.2f} | {r['flags']} |"
                for r in sorted(without, key=lambda r: float(r["exp"]))]
    return "\n".join(out) + "\n"


def figures(name: str) -> None:
    """Copy the figures of the recorded run into the images of the wiki, unchanged."""
    import shutil

    images = ROOT / "docs" / "wiki" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for source, target in (("benchmark_report.png", f"{name}_scatter.png"),
                           ("benchmark_report_ligands.png", f"{name}_ligands.png")):
        shutil.copyfile(ROOT / "benchmarks" / "results" / name / source, images / target)
        print(images / target)


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        for lang in ("en", "ja"):
            (OUT / f"{name}.{lang}.md").write_text(tables(name, lang))
            (OUT / f"{name}.felis.{lang}.md").write_text(felis_table(name, lang))
            print(OUT / f"{name}.{lang}.md", OUT / f"{name}.felis.{lang}.md")
        figures(name)


if __name__ == "__main__":
    main(sys.argv[1:])
