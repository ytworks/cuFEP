"""Two runs of the same benchmark set, compared ligand by ligand: the binding free energy and the
free energy of each leg (kcal/mol), and the mean, the standard deviation and the standard error of
the differences. Reads the two run directories (their ``benchmark_report.csv`` and the results of
the legs of repeat 0); writes nothing. Numbers only.

    scripts/in_docker.sh python spikes/m27_paired_runs.py RUN_A RUN_B
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

KCAL = 4.184


def table(run: Path) -> dict:
    with open(run / "benchmark_report.csv") as handle:
        rows = {r["ligand"]: r for r in csv.DictReader(handle)}
    out = {}
    for name, r in rows.items():
        legs = {}
        for leg in ("complex", "solvent"):
            result = json.loads((run / name / "repeat_0" / leg / "result.json").read_text())
            legs[leg] = (result["delta_g_decouple"] / KCAL, result.get("start_configurations", "not recorded"))
        out[name] = (float(r["exp"]), float(r["cufep"]), float(r["cufep_error"]), legs)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = table(Path(sys.argv[1])), table(Path(sys.argv[2]))
    differences = {"binding": [], "complex leg": [], "solvent leg": []}
    print("ligand experiment | A (standard error) | B (standard error) | B-A | complex leg A, B, B-A | "
          "solvent leg A, B, B-A | start configurations of the complex leg A / B")
    for name in a:
        if name not in b:
            continue
        exp, ga, sa, la = a[name]
        _, gb, sb, lb = b[name]
        ca, cb = la["complex"][0], lb["complex"][0]
        wa, wb = la["solvent"][0], lb["solvent"][0]
        differences["binding"].append(gb - ga)
        differences["complex leg"].append(cb - ca)
        differences["solvent leg"].append(wb - wa)
        print(f"{name} {exp:.2f} | {ga:.2f} ({sa:.2f}) | {gb:.2f} ({sb:.2f}) | {gb - ga:+.2f} | "
              f"{ca:.2f} {cb:.2f} {cb - ca:+.2f} | {wa:.2f} {wb:.2f} {wb - wa:+.2f} | "
              f"{la['complex'][1]} / {lb['complex'][1]}")
    for label, values in differences.items():
        x = np.array(values)
        print(f"{label}: {len(x)} ligands, mean of B-A {x.mean():+.2f}, standard deviation {x.std(ddof=1):.2f}, "
              f"standard error of the mean {x.std(ddof=1) / np.sqrt(len(x)):.2f}")


if __name__ == "__main__":
    main()
