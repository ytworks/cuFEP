"""Read finished legs again with the convergence analysis: is a leg in which adaptive lambda
scheduling added no window a leg whose overlap and convergence are in order?

For every leg below the given run directories that has samples: the windows adaptive lambda
scheduling started with and ended with, the lowest overlap its precalculation read, and from
the production the lowest overlap, the free energy of each half against the whole, BAR against
MBAR and the verdicts of ``cufep.analysis.mbar.quality_checks``. Nothing below ``runs/`` is
written; the table goes to standard output and the rows to a JSON file.

    spikes/run.sh spikes/m11_convergence_report.py runs/benchmark_tyk2_subset_r3 runs/val_tyk2_adaptive \\
        [--out spikes/out/m11_convergence.json] [--jobs 8]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

KCAL = 4.184


def one_leg(leg_dir: str) -> dict | None:
    from cufep.analysis.mbar import analyze_leg, quality_checks
    from cufep.sampling.storage import read_samples

    d = Path(leg_dir)
    try:
        samples = read_samples(d / "hrex")
        temperature = float(samples["attrs"].get("temperature_K", 298.15))
        a = analyze_leg(samples, temperature).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"leg_dir": leg_dir, "error": f"{type(exc).__name__}: {exc}"}
    q = quality_checks(a)
    kt = 8.31446261815324e-3 * temperature / KCAL
    row = {"leg_dir": leg_dir, "run": d.parts[-4], "ligand": d.parts[-3], "repeat": d.parts[-2], "leg": d.parts[-1],
           "windows": a["n_states"], "production_iterations": a["n_iterations_used"],
           "delta_g": a["delta_g"] / KCAL, "error": a["delta_g_error"] / KCAL,
           "first_half": a["first_half"][0] / KCAL, "second_half": a["second_half"][0] / KCAL,
           "halves_kT": q["halves"].get("difference_kT"), "halves_sigma": q["halves"].get("difference_sigma"),
           "bar": a["bar_delta_g"] / KCAL, "lowest_overlap": a["min_overlap"], "round_trips": a["round_trips"],
           "fewest_samples": q["samples"].get("fewest"), "slowest_window": q["samples"].get("window"),
           "g_slowest": q["samples"].get("statistical_inefficiency"),
           "overlap_ok": q["overlap"]["ok"], "samples_ok": q["samples"]["ok"], "halves_ok": q["halves"]["ok"],
           "halves_reason": q["halves"].get("reason"), "estimators_ok": q["estimators"]["ok"],
           "mode": None, "start": None, "added": None, "precalculation_lowest_overlap": None, "threshold": None,
           "rest2": 1.0}
    sched = d / "schedule.yaml"
    if sched.exists():
        s = yaml.safe_load(sched.read_text())
        row["mode"] = s.get("mode")
        row["rest2"] = float(s.get("rest2_scale", 1.0))
        prov = s.get("provenance") or {}
        if s.get("mode") == "adaptive" and "initial_windows" in prov:
            start = prov["initial_windows"]
            final = {k: len(v) for k, v in s["windows"].items()}
            last = {}
            for it in prov.get("iterations", []):
                last[it["stage"]] = it
            lows = [min(it["superdiagonal"]) for it in last.values() if it.get("superdiagonal")]
            row.update(start="/".join(str(start[k]) for k in final), added=sum(final[k] - start[k] for k in final),
                       precalculation_lowest_overlap=min(lows) if lows else None, threshold=prov.get("overlap_threshold"))
    return row


def main() -> None:
    args = sys.argv[1:]
    out = Path("spikes/out/m11_convergence.json")
    jobs = 8
    runs = []
    while args:
        a = args.pop(0)
        if a == "--out":
            out = Path(args.pop(0))
        elif a == "--jobs":
            jobs = int(args.pop(0))
        else:
            runs.append(Path(a))
    legs = sorted(str(p.parent.parent) for run in runs for p in run.glob("*/repeat_*/*/hrex/samples.h5"))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = [r for r in pool.map(one_leg, legs) if r is not None]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"{'run':30s} {'ligand':7s} {'leg':7s} win prod_low     dG     ±   | fewest samples (window, g) | halves: kT  sigma | "
          f"overlap samples halves BAR/MBAR  trips")
    for r in rows:
        if "leg" not in r:
            print(f"{r['leg_dir']}: {r['error']}")
            continue

        def flag(v):
            return {True: "ok", False: "below", None: "-"}[v]

        halves = "    -      -" if r["halves_kT"] is None else f"{r['halves_kT']:5.1f}  {r['halves_sigma']:5.1f}"
        print(f"{r['run'][:30]:30s} {r['ligand'][:7]:7s} {r['leg']:7s} {r['windows']:3d}  {r['lowest_overlap']:.3f}  "
              f"{r['delta_g']:7.2f} {r['error']:5.2f}  |   {r['fewest_samples']:6.0f}  ({r['slowest_window']:2d}, {r['g_slowest']:5.0f})     | "
              f"{halves}       | {flag(r['overlap_ok']):6s} {flag(r['samples_ok']):6s}  {flag(r['halves_ok']):6s} "
              f"{flag(r['estimators_ok']):6s}  {r['round_trips']:4d}")
    done = [r for r in rows if "leg" in r]
    for leg in ("complex", "solvent"):
        sub = [r for r in done if r["leg"] == leg]
        if sub:
            print(f"{leg}: {len(sub)} legs; overlap below 0.03 in {sum(r['overlap_ok'] is False for r in sub)}; "
                  f"fewer than 50 independent samples in a window in {sum(r['samples_ok'] is False for r in sub)}; "
                  f"halves differ by more than 1 kT in {sum(r['halves_ok'] is False for r in sub)}, not compared in "
                  f"{sum(r['halves_reason'] is not None for r in sub)}; BAR and MBAR more than 1 kcal/mol apart in "
                  f"{sum(r['estimators_ok'] is False for r in sub)}")


if __name__ == "__main__":
    main()
