"""Where adaptive lambda scheduling put its windows in stored legs, how slowly each place was
sampled, and a start for the precalculation that follows both.

    scripts/in_docker.sh python spikes/m18_window_design.py runs/benchmark_tyk2_full \
        runs/benchmark_a2a_full runs/tyk2_spread_1b770da_1 --windows 30 59

Read from every leg of the given runs whose windows adaptive lambda scheduling chose
(``<ligand>/repeat_<n>/<leg>/result.json``): the lambda windows of each stage and the statistical
inefficiency g of each window (in iterations of the replica exchange). Nothing is computed again
and nothing is written to the runs.

Printed per leg kind (complex, solvent) and stage, by tenths of lambda:

- the density of the windows: the mean number of intervals between neighbouring windows that
  lie in the tenth (an interval counts by the part of it that lies there);
- g: the geometric mean and the largest value of the windows in the tenth, and how many of
  them have fewer than 50 independent samples, counted as the check of the analysis counts them
  (``cufep.analysis.mbar.counted_samples``: the production divided by g, and not more than the
  samples MBAR got). The window at the end of a stage is the first of the next one; it is
  listed once, at lambda 1.0 of the earlier stage.

The start (plan docs/plans/2026-11-21-als-initial-window-design.md, cuFEP's own design). The
variance that an interval of width d contributes to the free energy is about d^2 * s^2 * g, with
s the spread of the energy difference per unit of lambda and g the statistical inefficiency there.
For a given number of windows the sum is smallest with a density of windows proportional to
s * sqrt(g). The density with which adaptive lambda scheduling placed its windows stands for s
(it places them at equal overlap), so every interval of a stored leg gets the weight sqrt(g),
with g the geometric mean of its two windows, spread evenly over the interval. The weights of
all legs are added. A leg that starts with N windows gets its N - 1 intervals shared among the
stages by their weight (at least one each). Within a stage every candidate step (0.01) gets a
part of the stage's intervals in proportion to its weight, but not more than one interval:
windows are candidates, so a step cannot hold two. What that limit takes from a step goes to
the other steps in proportion to their weights. The windows then lie where the running sum of
these parts reaches 1, 2, ..., each moved to its nearest candidate; because no step holds more
than one interval, no two windows share a candidate and none has to be moved aside.

The tables and the start are also written to spikes/out/window_design/ (untracked).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from cufep.analysis.mbar import counted_samples

CANDIDATES = 101
STAGES = {"complex": ["restrain", "discharge", "vanish"], "solvent": ["discharge", "vanish"]}
LEVEL = 50.0          # analysis.min_effective_samples, the level of Mey et al. 2020


def legs_of(run_dir: Path, leg: str):
    """(name, windows per stage, g per window, independent samples per window) of every leg of
    this kind in the run whose windows adaptive lambda scheduling chose."""
    for result_file in sorted(run_dir.glob(f"*/repeat_*/{leg}/result.json")):
        schedule_file = result_file.with_name("schedule.yaml")
        if not schedule_file.exists() or yaml.safe_load(schedule_file.read_text()).get("mode") != "adaptive":
            continue
        result = json.loads(result_file.read_text())
        analysis = result.get("analysis") or {}
        # the samples of a window as the check of the analysis counts them (not more than MBAR got)
        g, n = analysis.get("window_statistical_inefficiency"), counted_samples(analysis)
        windows = result.get("lambda_windows") or {}
        if g is None or not n or any(stage not in windows for stage in STAGES[leg]):
            continue
        yield f"{run_dir.name}/{result_file.parts[-4]}/{result_file.parts[-3]}", windows, g, n


def per_stage(leg: str, windows: dict, values: list) -> dict:
    """The values of the windows of each stage: the window at the end of a stage is the first
    of the next one, and is listed once."""
    out, offset = {}, 0
    for stage in STAGES[leg]:
        n = len(windows[stage])
        out[stage] = [float(v) for v in values[offset:offset + n]]
        offset += n - 1
    return out


def spread(lo: float, hi: float, weight: float, bins: np.ndarray) -> None:
    """Add ``weight`` to the bins of width 0.01, by the part of [lo, hi] that lies in each."""
    top = CANDIDATES - 1
    for j in range(int(math.floor(lo * top + 1e-9)), min(top, int(math.ceil(hi * top - 1e-9)))):
        part = min(hi, (j + 1) / top) - max(lo, j / top)
        if part > 0:
            bins[j] += weight * part / (hi - lo)


def collect(run_dirs: list[Path], leg: str) -> dict:
    top = CANDIDATES - 1
    stages = STAGES[leg]
    out = {"legs": [], "density": {s: np.zeros(top) for s in stages}, "weight": {s: np.zeros(top) for s in stages},
           "log_g": {s: [[] for _ in range(10)] for s in stages}, "largest_g": {s: [0.0] * 10 for s in stages},
           "below": {s: [0] * 10 for s in stages}}
    for run_dir in run_dirs:
        for name, windows, g, n in legs_of(run_dir, leg):
            out["legs"].append(name)
            g_of, n_of = per_stage(leg, windows, g), per_stage(leg, windows, n)
            for stage in STAGES[leg]:
                lam = [float(v) for v in windows[stage]]
                for i in range(len(lam) - 1):
                    slow = math.sqrt(max(g_of[stage][i], 1.0) * max(g_of[stage][i + 1], 1.0))
                    spread(lam[i], lam[i + 1], 1.0, out["density"][stage])
                    spread(lam[i], lam[i + 1], math.sqrt(slow), out["weight"][stage])
                first_of_leg = stage == STAGES[leg][0]
                for j, (value, g_window, n_window) in enumerate(zip(lam, g_of[stage], n_of[stage])):
                    if j == 0 and not first_of_leg:
                        continue                  # the window at a stage boundary is listed with the earlier stage
                    tenth = min(9, int(value * 10 + 1e-9))
                    out["log_g"][stage][tenth].append(math.log(max(g_window, 1.0)))
                    out["largest_g"][stage][tenth] = max(out["largest_g"][stage][tenth], g_window)
                    out["below"][stage][tenth] += int(n_window < LEVEL)
    return out


def shared_among_stages(total: dict, intervals: int, most: int) -> dict:
    """The intervals of a leg shared among its stages in proportion to their weights: at least
    one each, at most ``most`` (a stage has no more intervals than candidate steps). One at a
    time to the stage that is furthest below its share."""
    stages = list(total)
    if intervals < len(stages) or intervals > most * len(stages):
        raise ValueError(f"{intervals + 1} windows cannot be placed on {len(stages)} stage(s) of {most + 1} candidates")
    whole = sum(total.values())
    count = {s: 1 for s in stages}
    for _ in range(intervals - len(stages)):
        count[max((s for s in stages if count[s] < most), key=lambda s: intervals * total[s] / whole - count[s])] += 1
    return count


def intervals_per_step(weight: np.ndarray, intervals: int) -> np.ndarray:
    """How many of the ``intervals`` of a stage each candidate step gets: in proportion to its
    weight, and at most one, since windows are candidates and two of them cannot lie within one
    step. What the limit takes from a step goes to the other steps in proportion to their weights."""
    weight = np.asarray(weight, dtype=float)
    out, free, left = np.zeros(len(weight)), weight > 0.0, float(intervals)
    while free.any():
        scale = left / weight[free].sum()
        full = free & (weight * scale >= 1.0)
        if not full.any():
            out[free] = weight[free] * scale
            break
        out[full] = 1.0
        free &= ~full
        left -= float(full.sum())
    return out


def start_of(data: dict, leg: str, n_windows: int) -> dict:
    """The lambda values of each stage for a leg that starts with ``n_windows`` windows."""
    top = CANDIDATES - 1
    stages = STAGES[leg]
    count = shared_among_stages({s: float(data["weight"][s].sum()) for s in stages}, n_windows - 1, top)
    grid = np.arange(CANDIDATES) / top
    out = {}
    for stage in stages:
        cumulative = np.concatenate([[0.0], np.cumsum(intervals_per_step(data["weight"][stage], count[stage]))])
        # no step holds more than one interval, so neighbouring windows are a step or more apart and
        # their nearest candidates differ
        inner = [int(math.floor(float(np.interp(m, cumulative, grid)) * top + 0.5)) for m in range(1, count[stage])]
        values = [0] + inner + [top]
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"the windows of the {stage} stage do not fall on distinct candidates: {values}")
        out[stage] = [round(i / top, 2) for i in values]
    return out


def tenths(bins: np.ndarray, n_legs: int) -> list[float]:
    return [float(bins[10 * t:10 * t + 10].sum()) / max(1, n_legs) for t in range(10)]


def report(data: dict, leg: str, title: str) -> dict:
    n_legs = len(data["legs"])
    weight_of_leg = sum(sum(tenths(data["weight"][s], n_legs)) for s in STAGES[leg])
    print(f"\n## {title}: {leg} leg, {n_legs} legs")
    print("tenth of lambda        " + " ".join(f"{t / 10:.1f}-{(t + 1) / 10:.1f}" for t in range(10)))
    table = {}
    for stage in STAGES[leg]:
        density = tenths(data["density"][stage], n_legs)
        weight = tenths(data["weight"][stage], n_legs)
        mean_g = [math.exp(sum(v) / len(v)) if v else None for v in data["log_g"][stage]]
        table[stage] = {"intervals_per_tenth": density, "weight_per_tenth": weight, "geometric_mean_g": mean_g,
                        "largest_g": data["largest_g"][stage], "windows": [len(v) for v in data["log_g"][stage]],
                        "windows_below_50_samples": data["below"][stage]}
        print(f"{stage:10s} intervals   " + " ".join(f"{v:7.2f}" for v in density))
        print(f"{'':10s} g, mean     " + " ".join(f"{v:7.1f}" if v is not None else "      -" for v in mean_g))
        print(f"{'':10s} g, largest  " + " ".join(f"{v:7.0f}" for v in data["largest_g"][stage]))
        print(f"{'':10s} below 50    " + " ".join(
            f"{b:3d}/{len(v):<3d}" for b, v in zip(data["below"][stage], data["log_g"][stage])))
        print(f"{'':10s} weight      " + " ".join(f"{v:7.2f}" for v in weight)
              + f"   share of the stage {sum(weight) / max(1e-12, weight_of_leg):.2f}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--windows", type=int, nargs="*", default=[30, 59],
                    help="numbers of windows a complex leg starts with")
    ap.add_argument("--out", default="spikes/out/window_design")
    args = ap.parse_args()
    run_dirs = [Path(r) for r in args.runs]
    summary = {"runs": [str(r) for r in run_dirs], "level_of_independent_samples": LEVEL}
    for leg in ("complex", "solvent"):
        for run_dir in run_dirs:
            one = collect([run_dir], leg)
            if one["legs"]:
                summary[f"{leg}:{run_dir.name}"] = report(one, leg, run_dir.name)
        pooled = collect(run_dirs, leg)
        summary[f"{leg}:legs"] = pooled["legs"]
        if not pooled["legs"]:
            print(f"\n## {leg} leg: none of the runs has one whose windows adaptive lambda scheduling chose")
            continue
        summary[f"{leg}:all"] = report(pooled, leg, "all runs")
        if leg == "complex":
            summary["start"] = {}
            for n in args.windows:
                start = start_of(pooled, leg, n)
                summary["start"][str(n)] = start
                print(f"\n# start of the complex leg with {n} windows "
                      f"({', '.join(f'{s} {len(v)}' for s, v in start.items())})")
                print(yaml.safe_dump({"lambda_schedule": {"adaptive": {"initial_windows": {"complex": start}}}},
                                     default_flow_style=None, sort_keys=False, width=100).rstrip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "window_design.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwritten: {out / 'window_design.json'}")


if __name__ == "__main__":
    main()
