"""The thermodynamic length of the stages of stored runs, and what it predicts against what the
runs measured (plan docs/plans/2026-11-25-thermodynamic-length-schedule.md, the measurement V0).

    scripts/in_docker.sh python spikes/m36_thermodynamic_length.py \
        runs/benchmark_tyk2_full_e77a6c5 runs/benchmark_a2a_full_e77a6c5 --out spikes/out/m36

No simulation and no GPU: everything is read from the samples of the production of every leg of
the runs (``hrex/``), its ``schedule.yaml`` and its ``result.json``. For every pair of neighbouring
lambda windows:

- the length between the two (cufep.alchemy.thermodynamic_length.pair_distance: from the energy
  difference between them in the samples of each), and the two standard deviations it is made of;
- the acceptance of an exchange: the one the run recorded, the mean over all pairs of the samples
  of the two windows, and erfc(d / 2), which a normally distributed difference gives;
- the overlap of the pair alone from the samples of the two windows (the estimator the rule of
  adaptive lambda scheduling reads), the one a normally distributed difference gives, the
  prediction of A3FE's source, 1 / (2 + d^2 / 2), and the one the precalculation of the leg had
  for the pair (the last record of its stage in ``schedule.yaml``);
- the element of the overlap matrix of the leg, as its result has it.

Also where along a stage its length lies: the length of a pair per unit of lambda, by the tenth of
lambda that the middle of the pair is in.

For every stage the sum of the lengths of its pairs, and how many windows a spacing would give
that corresponds to an acceptance (ceil(L / s) intervals, s = 2 erfcinv(acceptance)); for every
leg the windows it had and those totals. The lengths between windows that are far apart leave the
relation that makes them add up, so a sum is the nearer to the length of the stage the closer the
windows of the run were.

Three tables are written to ``--out`` (pairs.csv, stages.csv, legs.csv) and a summary is printed.
Numbers only: nothing is judged here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

TARGETS = (0.55, 0.35, 0.27, 0.23)        # acceptances: 0.5 kcal/mol, the preprint, pair overlap 0.30, 1.0 kcal/mol


def leg_rows(run: Path, leg_dir: Path) -> tuple[list[dict], list[dict], dict] | str:
    from cufep.alchemy.schedule import pair_overlap
    from cufep.alchemy.thermodynamic_length import (
        exchange_acceptance,
        gaussian_acceptance,
        gaussian_pair_overlap,
        length_of_acceptance,
        pair_differences,
        pair_distance,
    )
    from cufep.analysis.mbar import count_round_trips, stage_boundaries, swap_acceptance
    from cufep.sampling.storage import StorageError, equilibration_boundary, read_samples

    try:
        samples = read_samples(leg_dir / "hrex")
    except (StorageError, OSError) as exc:
        return f"its samples cannot be read ({type(exc).__name__}: {exc})"
    attrs = samples["attrs"]
    first = equilibration_boundary(attrs, leg_dir)
    if first is None:
        return "its samples do not say where the production begins"
    u, states = np.asarray(samples["u"])[first:], np.asarray(samples["states"])[first:]
    if len(u) < 8:
        return "fewer than 8 iterations of its production are stored"
    schedule = yaml.safe_load((leg_dir / "schedule.yaml").read_text())
    result = json.loads((leg_dir / "result.json").read_text())
    analysis = result.get("analysis", {})
    recorded = swap_acceptance(samples, first, len(u))
    bounds = stage_boundaries(attrs["states"])
    provenance = schedule.get("provenance") or {}
    last_record = {}
    for record in provenance.get("iterations", []):
        last_record[record["stage"]] = record
    ligand, leg = leg_dir.parent.parent.name, leg_dir.name
    ident = {"run": run.name, "ligand": ligand, "leg": leg}
    differences = pair_differences(u, states)
    replica_in = np.argsort(states, axis=1)
    it = np.arange(len(u))
    in_leg = analysis.get("overlap_neighbour") or []
    pairs, stages = [], []
    for stage, (lo, hi) in bounds.items():
        lambdas = schedule["windows"][stage]
        record = last_record.get(stage, {})
        final = record.get("lambda") == [round(float(v), 6) for v in lambdas]

        def before(key, i, record=record, final=final):
            """What the last record of the precalculation of this stage has for pair ``i``, if it is of these
            windows."""
            values = record.get(key) if final else None
            return values[i] if values else None

        length = 0.0
        for k in range(lo, hi):
            x, y = differences[k]
            d = pair_distance(x, y)
            length += d
            lower, upper = replica_in[:, k], replica_in[:, k + 1]
            energies = [np.stack([u[it, r, k], u[it, r, k + 1]], axis=1) for r in (lower, upper)]
            alone = float(pair_overlap(energies, [0, 1])[0])
            row = {**ident, "stage": stage, "pair": k,
                   "lambda_lower": lambdas[k - lo], "lambda_upper": lambdas[k - lo + 1],
                   "distance": d, "sd_lower": float(np.std(x, ddof=1)), "sd_upper": float(np.std(y, ddof=1)),
                   "acceptance_recorded": None if recorded is None else recorded[k],
                   "acceptance_of_samples": exchange_acceptance(x, y),
                   "acceptance_gaussian": float(gaussian_acceptance(d)),
                   "pair_overlap": alone, "pair_overlap_gaussian": float(gaussian_pair_overlap(d)),
                   "pair_overlap_a3fe": 1.0 / (2.0 + 0.5 * d * d),
                   "pair_overlap_precalculation": before("pair_overlap", k - lo),
                   "stage_overlap_precalculation": before("superdiagonal", k - lo),
                   "overlap_in_leg": in_leg[k] if k < len(in_leg) else None}
            pairs.append(row)
        stage_row = {**ident, "stage": stage, "windows": hi - lo + 1, "length": length}
        for a in TARGETS:
            stage_row[f"intervals_at_{a}"] = max(1, math.ceil(length / float(length_of_acceptance(a))))
        stages.append(stage_row)
    accepted = [p["acceptance_recorded"] for p in pairs if p["acceptance_recorded"] is not None]
    ns = len(u) * float(attrs.get("iteration_ps", 0.0)) / 1000.0
    trips = count_round_trips(states)
    leg_row = {**ident, "windows": len(attrs["states"]), "length": sum(s["length"] for s in stages),
               "production_ns": ns, "round_trips": trips, "round_trips_per_ns": trips / ns if ns else None,
               "acceptance_lowest": min(accepted) if accepted else None,
               "acceptance_highest": max(accepted) if accepted else None,
               "acceptance_cv": float(np.std(accepted) / np.mean(accepted)) if accepted else None,
               "pair_overlap_lowest": min(p["pair_overlap"] for p in pairs),
               "overlap_in_leg_lowest": analysis.get("min_overlap"),
               "overlap_matrix_of_the_precalculation": provenance.get("overlap_matrix"),
               "overlap_threshold": provenance.get("overlap_threshold")}
    for a in TARGETS:
        leg_row[f"windows_at_{a}"] = sum(s[f"intervals_at_{a}"] for s in stages) + 1
    return pairs, stages, leg_row


def quantiles(values) -> str:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return "-"
    lo, mid, hi = np.quantile(values, [0.05, 0.5, 0.95])
    return f"{mid:.4f} [{lo:.4f}, {hi:.4f}] (n {len(values)})"


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else f"{v:.6g}" if isinstance(v, float) else v
                             for k, v in row.items()})


def main() -> None:
    from cufep.alchemy.thermodynamic_length import gaussian_pair_overlap, length_of_acceptance

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("spikes/out/m36"))
    args = parser.parse_args()
    pairs, stages, legs = [], [], []
    for run in args.runs:
        for result in sorted(run.glob("*/repeat_*/*/result.json")):
            leg_dir = result.parent
            if not (leg_dir / "schedule.yaml").exists() or not (leg_dir / "hrex").is_dir():
                continue
            rows = leg_rows(run, leg_dir)
            if isinstance(rows, str):
                print(f"{leg_dir}: left out, {rows}")
                continue
            pairs += rows[0]
            stages += rows[1]
            legs.append(rows[2])
            print(f"{leg_dir}: {rows[2]['windows']} windows, length {rows[2]['length']:.1f} kT", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "pairs.csv", pairs)
    write(args.out / "stages.csv", stages)
    write(args.out / "legs.csv", legs)

    def ratio(row, measured, predicted):
        return None if row[measured] is None or not row[predicted] else row[measured] / row[predicted]

    print(f"\n{len(legs)} legs, {len(pairs)} pairs of neighbouring windows. Median [5 %, 95 %].")
    for leg in ("complex", "solvent"):
        for stage in ("restrain", "discharge", "vanish"):
            rows = [p for p in pairs if p["leg"] == leg and p["stage"] == stage]
            if not rows:
                continue
            print(f"\n{leg}, {stage}")
            print("  distance between neighbours, kT:            ", quantiles(p["distance"] for p in rows))
            print("  sd in the upper window / in the lower:      ",
                  quantiles(p["sd_upper"] / p["sd_lower"] for p in rows))
            print("  acceptance recorded:                        ", quantiles(p["acceptance_recorded"] for p in rows))
            print("  acceptance recorded / erfc(d / 2):          ",
                  quantiles(ratio(p, "acceptance_recorded", "acceptance_gaussian") for p in rows))
            print("  acceptance recorded / of the samples:       ",
                  quantiles(ratio(p, "acceptance_recorded", "acceptance_of_samples") for p in rows))
            print("  pair overlap:                               ", quantiles(p["pair_overlap"] for p in rows))
            print("  pair overlap / normal distribution:         ",
                  quantiles(ratio(p, "pair_overlap", "pair_overlap_gaussian") for p in rows))
            print("  pair overlap / A3FE's 1 / (2 + d^2 / 2):    ",
                  quantiles(ratio(p, "pair_overlap", "pair_overlap_a3fe") for p in rows))
            print("  pair overlap / that of the precalculation:  ",
                  quantiles(ratio(p, "pair_overlap", "pair_overlap_precalculation") for p in rows))
            sums = [s for s in stages if s["leg"] == leg and s["stage"] == stage]
            print("  length of the stage, kT:                    ", quantiles(s["length"] for s in sums))
            print("  windows of the stage:                       ", quantiles(float(s["windows"]) for s in sums))
            for a in TARGETS:
                print(f"  windows at acceptance {a:.2f} (s {float(length_of_acceptance(a)):.2f} kT):     ",
                      quantiles(float(s[f"intervals_at_{a}"] + 1) for s in sums))
    for leg in ("complex", "solvent"):
        rows = [r for r in legs if r["leg"] == leg]
        if not rows:
            continue
        print(f"\n{leg} legs")
        print("  windows:                    ", quantiles(float(r["windows"]) for r in rows))
        for a in TARGETS:
            print(f"  windows at acceptance {a:.2f}:  ", quantiles(float(r[f"windows_at_{a}"]) for r in rows))
        print("  lowest acceptance recorded: ", quantiles(r["acceptance_lowest"] for r in rows))
        print("  highest acceptance recorded:", quantiles(r["acceptance_highest"] for r in rows))
        print("  sd / mean of the acceptances:", quantiles(r["acceptance_cv"] for r in rows))
        print("  round trips per ns:         ", quantiles(r["round_trips_per_ns"] for r in rows))
    # where along a stage its length lies: the length of a pair per unit of lambda, by the tenth of lambda
    # that the middle of the pair is in
    print("\nlength per unit of lambda (kT), median over the pairs whose middle is in each tenth of lambda")
    for leg in ("complex", "solvent"):
        for stage in ("restrain", "discharge", "vanish"):
            tenths = [[] for _ in range(10)]
            for p in pairs:
                if p["leg"] == leg and p["stage"] == stage:
                    middle = 0.5 * (p["lambda_lower"] + p["lambda_upper"])
                    tenths[min(int(middle * 10.0), 9)].append(p["distance"] / (p["lambda_upper"] - p["lambda_lower"]))
            if any(tenths):
                print(f"  {leg:8s} {stage:9s} " + " ".join(f"{np.median(t):6.1f}" if t else "     -" for t in tenths))
    # a pair placed at the distance of an acceptance of 0.35 would measure this share of the overlap a
    # normal distribution gives it; below 0.30 / that overlap the present rule of the complex leg adds a window
    level = 0.30 / float(gaussian_pair_overlap(length_of_acceptance(0.35)))
    for leg in ("complex", "solvent"):
        for stage in ("restrain", "discharge", "vanish"):
            shares = [ratio(p, "pair_overlap", "pair_overlap_gaussian")
                      for p in pairs if p["leg"] == leg and p["stage"] == stage]
            shares = [s for s in shares if s is not None]
            if shares:
                print(f"{leg}, {stage}: pair overlap at or below {level:.3f} of the normal one in "
                      f"{sum(s <= level for s in shares)} of {len(shares)} pairs")
    print(f"\nwritten: {args.out}/pairs.csv, stages.csv, legs.csv")


if __name__ == "__main__":
    main()
