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

import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

KCAL = 4.184


def halves_info(a: dict):
    """(difference of the two halves in kT, in standard errors of the difference), or (None, None): numbers only."""
    import numpy as np

    try:
        first, second = a["first_half"], a["second_half"]
        difference = abs(first[0] - second[0])
        sigma = float(np.hypot(first[1], second[1]))
    except (TypeError, IndexError, KeyError):
        return None, None
    if not np.isfinite(difference):
        return None, None
    kt = 0.0083144626 * float(a.get("temperature_K", 298.15))
    return difference / kt, (difference / sigma if sigma else None)


def one_leg(leg_dir: str) -> dict | None:
    # the one way to the result of a leg directory, with the settings and the equilibration that
    # cufep analyze uses (review of PR 1: this script analysed without the tail correction)
    from cufep.workflow import analyse_leg_directory

    d = Path(leg_dir)
    try:
        done = analyse_leg_directory(d)
    except Exception as exc:  # noqa: BLE001
        return {"leg_dir": leg_dir, "error": f"{type(exc).__name__}: {exc}"}
    result = done["result"]
    a, q, tail = result["analysis"], result["quality"], result["lrc_method"]
    if not all(isinstance(result.get(key), (int, float)) and np.isfinite(result[key])
               for key in ("delta_g_decouple", "delta_g_error")):
        return {"leg_dir": leg_dir, "error": "the free energy or its standard error is not finite"}

    def kcal(value):
        return None if value is None else value / KCAL

    row = {"leg_dir": leg_dir, "run": d.parts[-4], "ligand": d.parts[-3], "repeat": d.parts[-2], "leg": d.parts[-1],
           "windows": a["n_states"], "production_iterations": a["n_iterations_used"],
           "delta_g": kcal(result["delta_g_decouple"]), "error": kcal(result["delta_g_error"]),
           # the standard error MBAR gives, that of every number of time blocks, and the number reported
           "error_mbar": kcal(a.get("delta_g_mbar_error")), "error_n_blocks": a.get("delta_g_error_n_blocks"),
           "error_from": a.get("delta_g_error_from"), "tail": tail,
           # the settings the leg was analysed with, and where they came from (as cufep analyze)
           "analysis_record": result.get("analysis_record"), "settings_from": done["source"],
           "error_blocks": {count: kcal(value) for count, value in (a.get("delta_g_error_blocks") or {}).items()},
           "first_half": kcal(a["first_half"][0]), "second_half": kcal(a["second_half"][0]),
           # the halves as numbers, without a verdict (no check of cuFEP's own, the owner 2026-11-20)
           "halves_kT": halves_info(a)[0], "halves_sigma": halves_info(a)[1],
           "bar": kcal(a["bar_delta_g"]), "lowest_overlap": a["min_overlap"], "round_trips": a["round_trips"],
           "lowest_overlap_used": a["min_overlap_used"], "overlap_below": q["overlap"]["below"],
           "fewest_samples": q["samples"].get("fewest"), "slowest_window": q["samples"].get("window"),
           "kept_samples": q["samples"].get("kept"), "neighbour": q["samples"].get("neighbour"),
           "g_slowest": q["samples"].get("statistical_inefficiency"),
           "overlap_ok": q["overlap"]["ok"], "samples_ok": q["samples"]["ok"],
           # the levels this leg was judged by (its stored record of the analysis, or the defaults): the
           # literature's, the only checks
           "levels": {"min_overlap": q["overlap"]["threshold"], "min_effective_samples": q["samples"]["threshold"]},
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
            row.update(
                start="/".join(str(start[k]) for k in final),
                added=sum(final[k] - start[k] for k in final),
                precalculation_lowest_overlap=min(lows) if lows else None,
                threshold=prov.get("overlap_threshold"),
            )
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
    if jobs <= 1:
        rows = [r for r in map(one_leg, legs) if r is not None]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = [r for r in pool.map(one_leg, legs) if r is not None]
    out.parent.mkdir(parents=True, exist_ok=True)
    from cufep import jsonio

    out.write_text(jsonio.dumps(rows, indent=1))
    print(
        f"{'run':30s} {'ligand':7s} {'leg':7s} win  overlap: all  used      dG     ±   | fewest samples: estimated "
        f"kept "
        f"(window, to, g) | halves (no verdict): kT  sigma | overlap samples  trips"
    )
    for r in rows:
        if "leg" not in r:
            print(f"{r['leg_dir']}: {r['error']}")
            continue

        def flag(v):
            return {True: "ok", False: "below", None: "-"}[v]

        sigma = "    -" if r["halves_sigma"] is None else f"{r['halves_sigma']:5.1f}"
        halves = "    -      -" if r["halves_kT"] is None else f"{r['halves_kT']:5.1f}  {sigma}"
        used = "    -" if r["lowest_overlap_used"] is None else f"{r['lowest_overlap_used']:.3f}"
        to = {-1: "prev", 1: "next"}.get(r["neighbour"], "-")
        energy = "      -     -" if r["delta_g"] is None else f"{r['delta_g']:7.2f} {r['error'] or 0.0:5.2f}"
        print(
            f"{r['run'][:30]:30s} {r['ligand'][:7]:7s} {r['leg']:7s} {r['windows']:3d}         "
            f"{r['lowest_overlap']:.3f} {used}  "
            f"{energy}  |          {r['fewest_samples']:6.0f}   {r['kept_samples']:5d}  ({r['slowest_window']:2d}, "
            f"{to:4s}, "
            f"{r['g_slowest']:5.0f})     | {halves}       | {flag(r['overlap_ok']):6s} {flag(r['samples_ok']):6s}  "
            f"{r['round_trips']:4d}"
        )
    done = [r for r in rows if "leg" in r]
    import numpy as np

    print()
    print(f"{'run':30s} {'ligand':7s} {'leg':7s}  error: reported (blocks)  MBAR | error with blocks: number: error "
          f"| samples: kept  level  check")
    for r in done:
        # the numbers of blocks this leg was evaluated with (review of PR 1: a leg with 7 blocks fell out
        # of fixed columns of 40, 20, 10 and 5)
        blocks = "  ".join(f"{c}: " + ("-" if v is None else f"{v:.2f}") for c, v in r["error_blocks"].items())
        level = r["levels"]["min_effective_samples"]
        source = str(r["error_n_blocks"]) if r["error_from"] == "time blocks" else "anal"
        mbar = "    -" if r["error_mbar"] is None else f"{r['error_mbar']:5.2f}"   # pymbar may give none
        print(f"{r['run'][:30]:30s} {r['ligand'][:7]:7s} {r['leg']:7s}          {r['error']:5.2f}  ({source:>4s})  "
              f"{mbar} | {blocks} |         {r['kept_samples']:5d}  {level:5g}  {'ok' if r['samples_ok'] else 'below'}")
    for leg in ("complex", "solvent"):
        sub = [r for r in done if r["leg"] == leg]
        if sub:
            def spread(values) -> str:
                values = np.asarray([v for v in values if v is not None], dtype=float)    # the numbers there are
                if not values.size:
                    return "none"
                return f"{values.min():.2f}-{values.max():.2f}, median {np.median(values):.2f}"

            # only the legs whose standard error does come from time blocks are counted as such
            from_blocks = [r["error"] for r in sub if r["error_from"] == "time blocks"]
            analytical = [r["error"] for r in sub if r["error_from"] != "time blocks"]
            reported = Counter(r["error_n_blocks"] for r in sub if r["error_from"] == "time blocks")
            print(f"{leg}: standard error from time blocks in {len(from_blocks)} legs: {spread(from_blocks)} kcal/mol; "
                  f"the analytical one in {len(analytical)}: {spread(analytical)}; of MBAR alone "
                  f"{spread([r['error_mbar'] for r in sub])}; samples check below each leg's level in "
                  f"{sum(r['samples_ok'] is False for r in sub)}; number of blocks reported: "
                  + (", ".join(f"{c}: {m}" for c, m in sorted(reported.items(), reverse=True)) or "none"))
    print()
    for leg in ("complex", "solvent"):
        sub = [r for r in done if r["leg"] == leg]
        if sub:
            # every leg is judged by the levels it was analysed with (review of PR 1: the summary named the
            # default levels for legs judged by others); the levels that occur are listed with their legs
            def levels(key: str, sub=sub) -> str:
                return ", ".join(
                    f"{value:g} in {m}" for value, m in sorted(Counter(r["levels"][key] for r in sub).items())
                )

            print(f"{leg}: {len(sub)} legs; below their level of overlap ({levels('min_overlap')}) in "
                  f"{sum(r['overlap_ok'] is False for r in sub)} "
                  f"(all iterations: {sum('all iterations' in r['overlap_below'] for r in sub)}, the samples of the "
                  f"estimate: {sum('the samples of the estimate' in r['overlap_below'] for r in sub)}); "
                  f"fewer independent samples in a window than their level ({levels('min_effective_samples')}) in "
                  f"{sum(r['samples_ok'] is False for r in sub)}")


if __name__ == "__main__":
    main()
