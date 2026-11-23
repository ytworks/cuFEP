"""The ligands a run has finished so far, each against the same ligand of an earlier run of the
set and against experiment (kcal/mol): the binding free energy, the two legs, how the windows of
the complex leg were started, and the mean change by that. Reads the two run directories (the
earlier one has its ``benchmark_report.csv``); writes nothing. Numbers only.

    scripts/in_docker.sh python spikes/m30_ligands_against_a_run.py EARLIER_RUN RUN
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

KCAL = 4.184


def started(run: Path, ligand: str) -> str:
    """How the windows of the complex leg got their start configurations, from its schedule: the
    setting, where it is recorded; in a run of an earlier version, whether its windows ran at the
    same time (a walk gave their starts) or one after another (each from the one before it)."""
    schedule = yaml.safe_load((run / ligand / "repeat_0" / "complex" / "schedule.yaml").read_text())
    record = schedule.get("provenance", schedule)
    if record.get("start_configurations"):
        return str(record["start_configurations"])
    concurrent = record.get("windows_run_concurrently")
    return "not recorded" if concurrent is None else ("walk" if concurrent else "sequential")


def leg(run: Path, ligand: str, name: str) -> dict | None:
    path = run / ligand / "repeat_0" / name / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    earlier, run = Path(sys.argv[1]), Path(sys.argv[2])
    with open(earlier / "benchmark_report.csv") as handle:
        rows = {r["ligand"]: r for r in csv.DictReader(handle)}
    changes: dict[str, list[float]] = {}
    errors = {"earlier": [], "now": []}
    print("ligand | experiment | earlier -> now (standard error) | error earlier -> now | complex leg earlier -> now | "
          "solvent leg earlier -> now | windows of the complex leg started: earlier / now")
    for ligand, row in rows.items():
        result = run / ligand / "result.json"
        if not result.exists():
            continue
        now = json.loads(result.read_text())
        if now.get("dg_bind") is None:
            continue
        exp, before = float(row["exp"]), float(row["cufep"])
        legs = {(which, name): leg(where, ligand, name) for which, where in (("earlier", earlier), ("now", run))
                for name in ("complex", "solvent")}
        value = {key: r["delta_g_decouple"] / KCAL for key, r in legs.items()}
        how = [started(earlier, ligand), started(run, ligand)]
        changes.setdefault(how[0], []).append(now["dg_bind"] - before)
        errors["earlier"].append(before - exp)
        errors["now"].append(now["dg_bind"] - exp)
        print(f"{ligand} | {exp:.2f} | {before:.2f} -> {now['dg_bind']:.2f} ({now['dg_bind_error']:.2f}) | "
              f"{before - exp:+.2f} -> {now['dg_bind'] - exp:+.2f} | {value['earlier', 'complex']:.2f} -> "
              f"{value['now', 'complex']:.2f} | {value['earlier', 'solvent']:.2f} -> {value['now', 'solvent']:.2f} | "
              f"{how[0]} / {how[1]}")
    for name, values in changes.items():
        x = np.array(values)
        spread = f", standard deviation {x.std(ddof=1):.2f}" if len(x) > 1 else ""
        print(f"earlier started {name}: {len(x)} ligands, mean change {x.mean():+.2f}{spread}")
    if errors["now"]:
        a, b = np.array(errors["earlier"]), np.array(errors["now"])
        print(f"these {len(a)} ligands: mean signed error {a.mean():+.2f} -> {b.mean():+.2f}; root mean square error "
              f"{np.sqrt((a ** 2).mean()):.2f} -> {np.sqrt((b ** 2).mean()):.2f}")


if __name__ == "__main__":
    main()
