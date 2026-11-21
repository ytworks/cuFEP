"""Runs of one ligand with several window settings, side by side: the runs of the candidates of
plan docs/plans/2026-11-21-als-initial-window-design.md and the runs they are compared with.

    scripts/in_docker.sh python spikes/m19_window_runs.py --ligand ejm_31 --experiment -9.54 \
        now=runs/tyk2_interactions_720cdb1,runs/tyk2_spread_1b770da_1,runs/tyk2_spread_1b770da_2 \
        A=runs/tyk2_windows_a_<commit>_1,runs/tyk2_windows_a_<commit>_2 --production-ns 1 2 3 4 5

Every argument ``name=run,run,...`` is a group of runs of the same settings. Printed (Markdown):

- per run: the binding free energy with its standard error and its difference from experiment,
  the legs, the release of the restraint; the lambda windows of the complex leg per stage, how
  many of them adaptive lambda scheduling added and where; the notes of the precalculation
  (``max_windows``, ``saturated``); the lowest overlap of neighbouring windows in the matrix of
  the leg (level 0.03) and the fewest independent samples of a window (level 50) with its place;
  the round trips;
- per group: mean, standard deviation and range of the binding free energy and of the legs, and
  against the experiment the difference of the mean and the root mean square of the differences
  of the runs;
- with ``--production-ns``: the same per group for a production cut to each of the given lengths.
  The legs are analysed again from the first part of their stored samples by the function that
  analyses a leg (``leg_result_from_samples``) and combined by the function that combines the
  legs of a ligand (``combine_ligand``), with the settings of the analysis that the run recorded,
  so a shorter production carries its own checks. Results of this go to
  ``spikes/out/window_runs/`` (untracked); nothing is written to the runs.
- with ``--times``: the time of each phase as the run recorded it (``timing.json``). These are
  for the owner only: no run time of the cuPotential plugin goes into a tracked file.

Numbers only; no verdict is attached to any of them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from cufep.analysis.mbar import counted_samples

KCAL = 4.184
STAGES = {"complex": ["restrain", "discharge", "vanish"], "solvent": ["discharge", "vanish"]}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def place_of(windows: dict, leg: str, index: int) -> str:
    """Stage and lambda of the window with this number in the leg (a window at a stage boundary
    is named by the stage that ends there)."""
    offset = 0
    for stage in STAGES[leg]:
        values = windows[stage]
        if index < offset + len(values):
            return f"{stage} {values[index - offset]:g}"
        offset += len(values) - 1
    return "?"


def leg_facts(leg_dir: Path, leg: str) -> dict:
    result = read_json(leg_dir / "result.json")
    schedule = yaml.safe_load((leg_dir / "schedule.yaml").read_text()) if (leg_dir / "schedule.yaml").exists() else {}
    if result is None:
        return {"missing": True}
    a, windows = result["analysis"], result["lambda_windows"]
    samples = counted_samples(a)                 # as the check of the analysis counts them
    fewest = int(np.argmin(samples))
    prov = schedule.get("provenance", {})
    inserted = [f"{rec['stage']} {rec['section'][0]:g}-{rec['section'][1]:g}"
                for rec in prov.get("iterations", []) if rec.get("status") == "inserted"]
    last = {}
    for rec in prov.get("iterations", []):
        last[rec["stage"]] = rec
    return {
        "delta_g": a["delta_g"] / KCAL, "error": a["delta_g_error"] / KCAL,
        "windows": {s: len(windows[s]) for s in STAGES[leg] if s in windows}, "n_states": a["n_states"],
        "start": prov.get("initial_windows"), "added": result.get("lambda_windows_added"),
        "overlap_matrix": prov.get("overlap_matrix", "stage"), "threshold": prov.get("overlap_threshold"),
        "iterations": max([rec["iteration"] for rec in prov.get("iterations", [])], default=0),
        "inserted": inserted, "status": {s: rec.get("status") for s, rec in last.items()},
        "lowest_read": {s: min(rec.get("pair_overlap") or rec.get("superdiagonal") or [float("nan")])
                        for s, rec in last.items()},
        "schedule_notes": schedule.get("notes", []),
        "min_overlap": a["min_overlap"], "fewest_samples": samples[fewest],
        "fewest_place": place_of(windows, leg, fewest),
        "below_50": sum(1 for v in samples if v < 50.0), "round_trips": a.get("round_trips"),
        "lambda": windows,
    }


def run_facts(run: Path, ligand: str) -> dict:
    lig = read_json(run / ligand / "result.json")
    rep = (lig or {}).get("repeats", [{}])[0] if lig else {}
    return {"run": run.name, "dg_bind": (lig or {}).get("dg_bind"), "error": (lig or {}).get("dg_bind_error"),
            "dg_complex": rep.get("dg_complex"), "dg_solvent": rep.get("dg_solvent"),
            "release": rep.get("restraint_release"), "not_reliable": (lig or {}).get("not_reliable"),
            "few_samples": (lig or {}).get("few_samples"),
            "legs": {leg: leg_facts(run / ligand / "repeat_0" / leg, leg) for leg in ("complex", "solvent")}}


def cut_production(run: Path, ligand: str, production_ns: float, scratch: Path) -> dict | None:
    """The result of the ligand from the first ``production_ns`` of the production of both legs,
    by the analysis of the package; None if a leg holds less."""
    from cufep.config.schema import Config
    from cufep.sampling.storage import read_samples
    from cufep.workflow import combine_ligand, leg_result_from_samples, run_record_of, write_leg_result

    cfg = Config.model_validate(yaml.safe_load((run / "resolved_config.yaml").read_text()))
    target = scratch / run.name / f"{production_ns:g}ns" / ligand
    if target.exists():
        shutil.rmtree(target)
    for leg in ("complex", "solvent"):
        leg_dir = run / ligand / "repeat_0" / leg
        result = read_json(leg_dir / "result.json")
        samples = read_samples(leg_dir / "hrex")
        n_eq = int(result["n_equilibration_iterations"])
        end = n_eq + int(round(production_ns * 1000.0 / float(samples["attrs"]["iteration_ps"])))
        if end > samples["u"].shape[0]:
            return None
        cut = dict(samples)
        for key in ("u", "states", "volume", "swaps"):
            if cut.get(key) is not None:
                cut[key] = cut[key][:end]
        for rows, keys in (("lrc_iter", ("lrc_iter", "lrc_du")), ("frame_iter", ("frame_iter", "frames", "frame_box"))):
            if cut.get(rows) is not None and len(cut[rows]):
                keep = np.asarray(cut[rows]) < end
                for key in keys:
                    if cut.get(key) is not None:
                        cut[key] = np.asarray(cut[key])[keep]
        res = leg_result_from_samples(leg_dir, cfg.analysis, run_record_of(result, leg_dir), samples=cut,
                                      n_equilibration=n_eq)
        out = target / "repeat_0" / leg
        out.mkdir(parents=True)
        write_leg_result(out, res)
    combined = combine_ligand(target, ligand, 1, analysis=cfg.analysis, publish=False)
    rep = combined.get("repeats", [{}])[0]
    legs = {leg: read_json(target / "repeat_0" / leg / "result.json")["analysis"] for leg in ("complex", "solvent")}
    return {"dg_bind": combined.get("dg_bind"), "error": combined.get("dg_bind_error"),
            "dg_complex": rep.get("dg_complex"), "dg_solvent": rep.get("dg_solvent"),
            "min_overlap": min(legs[leg]["min_overlap"] for leg in legs),
            "fewest_samples": min(min(counted_samples(legs[leg])) for leg in legs),
            "not_reliable": combined.get("not_reliable"), "few_samples": combined.get("few_samples")}


def spread(values: list) -> str:
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return "- | - | -"
    sd = f"{v.std(ddof=1):.2f}" if len(v) > 1 else "-"
    return f"{v.mean():.2f} | {sd} | {v.max() - v.min():.2f}"


def from_experiment(values: list, experiment: float | None) -> str:
    """Mean of the values minus the experiment, and the root mean square of their differences
    from it (the two together: where the runs lie and how far a single run is off)."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0 or experiment is None:
        return "- | -"
    return f"{v.mean() - experiment:+.2f} | {float(np.sqrt(np.mean((v - experiment) ** 2))):.2f}"


def fmt(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="name=run,run,...")
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--experiment", type=float, default=None, help="kcal/mol")
    ap.add_argument("--production-ns", type=float, nargs="*", default=[])
    ap.add_argument("--times", action="store_true")
    ap.add_argument("--out", default="spikes/out/window_runs")
    args = ap.parse_args()
    groups = {}
    for item in args.groups:
        name, _, runs = item.partition("=")
        groups[name] = [Path(r) for r in runs.split(",") if r]
    facts = {name: [run_facts(run, args.ligand) for run in runs] for name, runs in groups.items()}

    print(f"## {args.ligand}: binding free energy per run (kcal/mol)"
          + (f"; experiment {args.experiment:g}" if args.experiment is not None else ""))
    print("| group | run | dG_bind | SE | dG_bind - exp | complex leg | solvent leg | release | flags |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, runs in facts.items():
        for f in runs:
            off = None if f["dg_bind"] is None or args.experiment is None else f["dg_bind"] - args.experiment
            flags = "; ".join(
                ([f"NOT RELIABLE: {', '.join(f['not_reliable'])}"] if f["not_reliable"] else [])
                + ([f"few independent samples: {', '.join(f['few_samples'])}"] if f["few_samples"] else []))
            print(f"| {name} | {f['run']} | {fmt(f['dg_bind'])} | {fmt(f['error'])} | {fmt(off)} | "
                  f"{fmt(f['dg_complex'])} | {fmt(f['dg_solvent'])} | {fmt(f['release'])} | {flags or '-'} |")

    print("\n## per group: mean, standard deviation, largest minus smallest; against experiment the mean "
          "difference and the root mean square difference of the runs (kcal/mol)")
    print("| group | runs | dG_bind mean | SD | range | mean - exp | RMS of dG_bind - exp | complex mean | SD | range "
          "| solvent mean | SD | range |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, runs in facts.items():
        done = [f for f in runs if f["dg_bind"] is not None]
        bind = [f["dg_bind"] for f in done]
        print(f"| {name} | {len(done)} | {spread(bind)} | {from_experiment(bind, args.experiment)} | "
              f"{spread([f['dg_complex'] for f in done])} | {spread([f['dg_solvent'] for f in done])} |")

    for leg in ("complex", "solvent"):
        print(f"\n## {leg} leg: lambda windows and the checks of the analysis")
        print("| group | run | windows (" + " / ".join(STAGES[leg]) + ") | start | added | overlap read (threshold) | "
              "iterations | status | lowest overlap in the leg (level 0.03) | fewest independent samples (level 50) "
              "at | windows below 50 | round trips |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, runs in facts.items():
            for f in runs:
                g = f["legs"][leg]
                if g.get("missing"):
                    print(f"| {name} | {f['run']} | no result | | | | | | | | | |")
                    continue
                per_stage = " / ".join(str(g["windows"].get(s, "-")) for s in STAGES[leg])
                start = " / ".join(str((g["start"] or {}).get(s, "-")) for s in STAGES[leg])
                added = " / ".join(str((g["added"] or {}).get(s, "-")) for s in STAGES[leg])
                status = ", ".join(f"{s} {v}" for s, v in g["status"].items())
                print(f"| {name} | {f['run']} | {g['n_states']} ({per_stage}) | {start} | {added} | "
                      f"{g['overlap_matrix']} ({g['threshold']}) | {g['iterations']} | {status} | "
                      f"{g['min_overlap']:.3f} | {g['fewest_samples']:.0f} at {g['fewest_place']} | {g['below_50']} | "
                      f"{g['round_trips']} |")
        for name, runs in facts.items():
            for f in runs:
                g = f["legs"][leg]
                if g.get("missing"):
                    continue
                print(f"\n{name} {f['run']} {leg}: lowest overlap the rule read at its end: "
                      + ", ".join(f"{s} {v:.3f}" for s, v in g["lowest_read"].items()))
                print("  sections it added a window to, in order: " + ("; ".join(g["inserted"]) or "none"))
                for stage in STAGES[leg]:
                    print(f"  {stage}: " + " ".join(f"{v:g}" for v in g["lambda"].get(stage, [])))
                for note in g["schedule_notes"]:
                    print(f"  note: {note}")

    if args.production_ns:
        scratch = Path(args.out)
        table = {}
        for name, runs in groups.items():
            for run in runs:
                for ns in args.production_ns:
                    try:
                        table[(name, run.name, ns)] = cut_production(run, args.ligand, ns, scratch)
                    except Exception as exc:  # noqa: BLE001 - a part without an estimate is reported, not hidden
                        table[(name, run.name, ns)] = {"problem": f"{type(exc).__name__}: {exc}"}
        print("\n## production cut to a shorter length: per group (kcal/mol)")
        print("| group | production ns | runs with a value | dG_bind per run | mean | SD | range | mean - exp | "
              "RMS of dG_bind - exp | complex SD | lowest overlap | fewest independent samples |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, runs in groups.items():
            for ns in args.production_ns:
                rows = [table[(name, run.name, ns)] for run in runs]
                good = [r for r in rows if r and r.get("dg_bind") is not None]
                values = ", ".join(fmt(r.get("dg_bind")) if r and "problem" not in r else "no value" for r in rows)
                complex_sd = spread([r["dg_complex"] for r in good]).split(" | ")[1]
                bind = [r["dg_bind"] for r in good]
                print(f"| {name} | {ns:g} | {len(good)} | {values} | {spread(bind)} | "
                      f"{from_experiment(bind, args.experiment)} | "
                      f"{complex_sd} | " + (f"{min(r['min_overlap'] for r in good):.3f} | "
                                            f"{min(r['fewest_samples'] for r in good):.0f} |" if good else "- | - |"))
        for key, row in table.items():
            if row and "problem" in row:
                print(f"\n{key[0]} {key[1]} at {key[2]:g} ns: {row['problem']}")
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / f"{args.ligand}_cut.json").write_text(json.dumps(
            {f"{k[0]}|{k[1]}|{k[2]:g}": v for k, v in table.items()}, indent=1))

    if args.times:
        print("\n## time of each phase as the runs recorded it, in seconds (for the owner; not for a tracked file)")
        for name, runs in groups.items():
            for run in runs:
                timing = read_json(run / "timing.json") or {}
                phases = {f"{job.get('leg')}:{k}": round(v) for job in timing.get("jobs", [])
                          for k, v in (job.get("phases") or {}).items()}
                per_iteration = {job.get("leg"): round((job.get("sampling") or {}).get("seconds_per_iteration", 0), 3)
                                 for job in timing.get("jobs", [])}
                print(f"{name} {run.name}: wall {round(timing.get('wall_seconds', 0))}; {phases}; "
                      f"seconds per iteration {per_iteration}; GPUs {timing.get('gpus')}")


if __name__ == "__main__":
    main()
