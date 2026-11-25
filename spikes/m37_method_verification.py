"""The verification runs of plan docs/plans/2026-11-25-thermodynamic-length-schedule.md (stage 4) side by
side: every run of ``lambda_schedule.adaptive.method: thermodynamic_length`` against the run of the present
method from the same image (the control) and the stored runs of the present method, with the measures the
plan's section "Verification" names, and the comparison at equal cost.

    scripts/in_docker.sh python spikes/m37_method_verification.py [--out spikes/out/m37]

Per run and leg: the free energies (binding, legs, stages, the release of the restraint) with their standard
errors and the difference from experiment; the windows per stage, split into the start, what the rule of
adaptive lambda scheduling added, what the thermodynamic length added and what the rule added after that
(repairs); the windows run in the precalculation; the simulated time (precalculation, production with its
equilibration); the overlap of neighbouring windows (the two matrices the check of the analysis reads, and the
pair alone from the production samples), the acceptance of exchanges between neighbours (lowest, highest,
coefficient of variation), the thermodynamic length between neighbours and of each stage; the round trips and
their rate; the fewest independent samples of a window and the largest statistical inefficiency; the halves.
At equal cost: the production of each leg of a run of the method cut to the windows x ns of its control
(5 ns x windows of the control / windows of the run), analysed again by the analysis of the package and the
legs combined as a run combines them (as ``spikes/m19_window_runs.py --production-ns`` does).

Writes ``report.json`` (every number, also per pair of windows and per state, for the figures) and prints
Markdown tables. Numbers only; no verdict. Run times are not read: they are for the owner only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m19_window_runs import LEGS, STAGES, ligand_dir, lowest_overlap, read_json  # noqa: E402

from cufep.analysis.mbar import counted_samples  # noqa: E402

KCAL = 4.184
PRODUCTION_NS = 5.0

# experiment and published prediction: the columns exp_dg_kcal_mol and the prediction of the benchmark files
# (benchmarks/tyk2/ligands.csv, benchmarks/a2a/ligands.csv)
GROUPS = [
    {"system": "ejm_31, TIP3P", "ligand": "ejm_31", "experiment": -9.54, "published_prediction": -9.91, "runs": [
        ("control", "runs/tyk2_als_eeb2351_1", "eeb2351", "als"),
        ("method", "runs/tyk2_tl_eeb2351_1", "eeb2351", "thermodynamic_length"),
        ("method", "runs/tyk2_tl_eeb2351_2", "eeb2351", "thermodynamic_length"),
        ("method", "runs/tyk2_tl_eeb2351_3", "eeb2351", "thermodynamic_length"),
        ("stored", "runs/tyk2_windows_a_5e2cf33_1", "5e2cf33", "als"),
        ("stored", "runs/tyk2_windows_a_5e2cf33_2", "5e2cf33", "als"),
        ("stored", "runs/tyk2_windows_a_5e2cf33_3", "5e2cf33", "als"),
        ("stored", "runs/benchmark_tyk2_full_e77a6c5", "e77a6c5", "als")]},
    {"system": "ejm_31, OPC", "ligand": "ejm_31", "experiment": -9.54, "published_prediction": -9.91, "runs": [
        ("control", "runs/tyk2_als_opc_eeb2351_1", "eeb2351", "als"),
        ("method", "runs/tyk2_tl_opc_eeb2351_1", "eeb2351", "thermodynamic_length"),
        ("stored", "runs/benchmark_tyk2_full_opc_29fb9df", "29fb9df", "als")]},
    {"system": "A2A 4e, bilayer", "ligand": "4e", "experiment": -12.074, "published_prediction": -11.010, "runs": [
        ("control", "runs/a2a_als_eeb2351_1", "eeb2351", "als"),
        ("method", "runs/a2a_tl_eeb2351_1", "eeb2351", "thermodynamic_length"),
        ("stored", "runs/benchmark_a2a_full_e77a6c5", "e77a6c5", "als")]},
]


def production(leg_dir: Path, n_eq: int):
    """u and states of the production, and u as [iteration, state the configuration is in, state evaluated] (kT)."""
    from cufep.sampling.storage import read_samples

    samples = read_samples(leg_dir / "hrex")
    u, states = np.asarray(samples["u"])[n_eq:], np.asarray(samples["states"])[n_eq:]
    it = np.arange(len(u))[:, None]
    return u, states, u[it, np.argsort(states, axis=1)]


def pairs_of(schedule: dict) -> list[dict]:
    """Stage and lambda of every pair of neighbouring states of a leg (a pair is of the stage of its upper state)."""
    names = schedule["stages"]                     # stage_of_state holds the number of the stage in this list
    stage_of, lam = [names[i] for i in schedule["stage_of_state"]], schedule["stage_lambda"]
    out = []
    for i in range(len(stage_of) - 1):
        stage = stage_of[i + 1]
        lower = lam[i] if stage_of[i] == stage else 0.0
        out.append({"stage": stage, "lambda": [round(float(lower), 4), round(float(lam[i + 1]), 4)]})
    return out


def window_counts(schedule: dict, result: dict, leg: str) -> dict:
    """Per stage: the start, the windows the rule ended with before any length was measured, what the lengths
    added, what the rule added after that, and the end."""
    prov = schedule.get("provenance", {})
    start = prov.get("initial_windows") or {}
    end = {s: len(v) for s, v in (result.get("lambda_windows") or {}).items()}
    passes = (prov.get("length_additions") or {}).get("passes") or []
    out = {}
    for s in STAGES[leg]:
        if passes:
            rule = len(passes[0]["stages"][s]["lambda"])
            length = sum(len(p["stages"][s].get("added_lambda") or []) for p in passes)
        else:
            rule, length = end.get(s), 0
        out[s] = {"start": start.get(s), "rule": rule, "length_added": length,
                  "repair": end.get(s) - rule - length, "end": end.get(s)}
    return out


def leg_numbers(leg_dir: Path, leg: str) -> dict:
    from cufep.alchemy.schedule import pair_overlap
    from cufep.alchemy.thermodynamic_length import pair_differences, pair_distance

    result = read_json(leg_dir / "result.json")
    schedule = yaml.safe_load((leg_dir / "schedule.yaml").read_text())
    a, prov = result["analysis"], schedule.get("provenance", {})
    n_eq, stored = int(result["n_equilibration_iterations"]), int(result["n_iterations_stored"])
    iteration_ps = 1000.0 * float(result["production_ns"]) / (stored - n_eq)
    k = int(a["n_states"])
    acc = np.asarray(a.get("swap_acceptance_neighbour") or [], dtype=float)
    counted = counted_samples(a)
    ineff = np.asarray(a.get("window_statistical_inefficiency") or [], dtype=float)
    pairs = pairs_of(schedule)
    u, states, by_state = production(leg_dir, n_eq)
    alone = pair_overlap([by_state[:, i, :] for i in range(k)], list(range(k)))
    # the lengths between neighbours: as the analysis records them since stage 2 of the plan, and for results of
    # an earlier analysis from the production samples by the same functions (the analysis computes them so)
    distance, distance_source = a.get("neighbour_distance"), "analysis"
    if not distance:
        distance, distance_source = [pair_distance(x, y) for x, y in pair_differences(u, states)], "production samples"
    stage_length = a.get("stage_length")
    if not stage_length:
        stage_length = {s: float(sum(d for d, p in zip(distance, pairs) if p["stage"] == s)) for s in STAGES[leg]}
    out = {
        "dg": result["delta_g_decouple"] / KCAL, "error": result["delta_g_error"] / KCAL,
        "stage_dg": {s: v / KCAL for s, v in (a.get("stage_delta_g") or {}).items()},
        "halves": [(a.get("first_half") or [None])[0], (a.get("second_half") or [None])[0]],
        "windows": window_counts(schedule, result, leg), "n_windows": k,
        "lambda": result.get("lambda_windows"),
        "precalculation_windows_run": prov.get("windows_simulated"),
        "precalculation_ns": (prov.get("windows_simulated") or 0)
        * (float(prov.get("window_ps", 0)) + float(prov.get("equilibration_ps", 0))) / 1000.0,
        "production_ns_all_windows": k * stored * iteration_ps / 1000.0,
        "overlap_matrix_lowest": a.get("min_overlap"), "overlap_matrix_of_estimate_lowest": a.get("min_overlap_used"),
        "overlap_check": lowest_overlap(a),
        "overlap_pair_alone": [float(x) for x in alone],
        "acceptance": [float(x) for x in acc],
        "acceptance_lowest": float(acc.min()), "acceptance_highest": float(acc.max()),
        "acceptance_cv": float(acc.std() / acc.mean()),
        "distance": [float(x) for x in distance], "distance_source": distance_source,
        "stage_length": stage_length,
        "round_trips": a.get("round_trips"),
        "round_trips_per_ns": (a.get("round_trips") or 0) / float(result["production_ns"]),
        "fewest_independent_samples": float(min(counted)), "windows_below_50": int(sum(c < 50 for c in counted)),
        "largest_inefficiency": float(ineff.max()), "largest_inefficiency_state": int(ineff.argmax()),
        "restraint": (result.get("restraint_choice") or {}).get("anchors"),
        "pairs": pairs,
        "f_k": {json.dumps([round(s["lambda_restraints"], 4), round(s["lambda_electrostatics"], 4),
                            round(s["lambda_sterics"], 4)]): f / KCAL for s, f in zip(schedule["states"], a["f_k"])},
    }
    return out


def cut_by_leg(run: Path, ligand: str, production_ns: dict, target: Path, reuse: bool = False) -> dict:
    """The result of the ligand from the first ``production_ns[leg]`` of the production of each leg, by the
    analysis of the package with the settings each leg recorded, the legs combined as a run combines them.
    With ``reuse`` the legs analysed by an earlier call for the same lengths are taken as they are."""
    from cufep.sampling.storage import read_samples
    from cufep.workflow import (
        analysis_settings_of,
        combine_ligand,
        leg_result_from_samples,
        run_record_of,
        write_leg_result,
    )

    lig_dir = ligand_dir(run, ligand)
    target = target / lig_dir.name
    stamp = target / "production_ns.json"
    reuse = reuse and stamp.exists() and json.loads(stamp.read_text()) == production_ns
    if target.exists() and not reuse:
        shutil.rmtree(target)
    settings = None
    for leg in LEGS:
        leg_dir = lig_dir / "repeat_0" / leg
        result = read_json(leg_dir / "result.json")
        samples = read_samples(leg_dir / "hrex")
        settings = analysis_settings_of(result.get("analysis_record"), samples["attrs"])
        if reuse:
            continue
        n_eq = int(result["n_equilibration_iterations"])
        end = n_eq + int(round(production_ns[leg] * 1000.0 / float(samples["attrs"]["iteration_ps"])))
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
        res = leg_result_from_samples(leg_dir, settings, run_record_of(result, leg_dir), samples=cut,
                                      n_equilibration=n_eq)
        out = target / "repeat_0" / leg
        out.mkdir(parents=True)
        write_leg_result(out, res)
    stamp.write_text(json.dumps(production_ns))
    combined = combine_ligand(target, ligand, 1, analysis=settings.analysis, publish=False)
    rep = (combined.get("repeats") or [{}])[0]
    legs = {leg: read_json(target / "repeat_0" / leg / "result.json") for leg in LEGS}
    return {"production_ns": production_ns, "dg_bind": combined.get("dg_bind"), "error": combined.get("dg_bind_error"),
            "dg_complex": rep.get("dg_complex"), "dg_solvent": rep.get("dg_solvent"),
            "leg_error": {leg: legs[leg]["delta_g_error"] / KCAL for leg in LEGS},
            "overlap_check": min(lowest_overlap(legs[leg]["analysis"]) for leg in LEGS),
            "fewest_independent_samples": min(min(counted_samples(legs[leg]["analysis"])) for leg in LEGS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="spikes/out/m37")
    ap.add_argument("--reuse-equal-cost", action="store_true",
                    help="take the legs cut to the same lengths by an earlier run of this script as they are")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for group in GROUPS:
        ligand, runs = group["ligand"], []
        control_windows = None
        for role, run, image, method in group["runs"]:
            lig = read_json(ligand_dir(Path(run), ligand) / "result.json")
            rep = lig["repeats"][0]
            legs = {leg: leg_numbers(ligand_dir(Path(run), ligand) / "repeat_0" / leg, leg)
                    for leg in LEGS}
            entry = {"role": role, "run": run, "image": image, "method": method,
                     "dg_bind": lig["dg_bind"], "error": lig["dg_bind_error"],
                     "dg_complex": rep["dg_complex"], "dg_solvent": rep["dg_solvent"],
                     "restraint_release": rep["restraint_release"], "lrc": [rep["lrc_complex"], rep["lrc_solvent"]],
                     "restraint": rep.get("restraint_choice", {}).get("anchors"),
                     "interaction": {k: rep.get("restraint_choice", {}).get("interaction", {}).get(k)
                                     for k in ("residue", "type")},
                     "legs": legs}
            if role == "control":
                control_windows = {leg: legs[leg]["n_windows"] for leg in LEGS}
            if role == "method":
                ns = {leg: PRODUCTION_NS * control_windows[leg] / legs[leg]["n_windows"] for leg in LEGS}
                entry["equal_cost"] = cut_by_leg(Path(run), ligand, ns, out / "equal_cost" / Path(run).name,
                                                 args.reuse_equal_cost)
            runs.append(entry)
            print(f"{group['system']} {role} {run}: {lig['dg_bind']:.2f} ± {lig['dg_bind_error']:.2f}", flush=True)
        keys = ("system", "ligand", "experiment", "published_prediction")
        report.append({k: group[k] for k in keys} | {"runs": runs})
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print_tables(report)


def print_tables(report: list) -> None:
    for g in report:
        print(f"\n## {g['system']} (experiment {g['experiment']:.2f}, "
              f"published prediction {g['published_prediction']:.2f})\n")
        print("| run | role | dG | dG - exp | complex | solvent | release | windows | precalc windows run | "
              "simulated ns (precalc + production) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in g["runs"]:
            c, s = r["legs"]["complex"], r["legs"]["solvent"]
            ns = sum(r["legs"][leg]["precalculation_ns"] + r["legs"][leg]["production_ns_all_windows"] for leg in LEGS)
            print(f"| {Path(r['run']).name} | {r['role']} | {r['dg_bind']:.2f} ± {r['error']:.2f} | "
                  f"{r['dg_bind'] - g['experiment']:+.2f} | {c['dg']:.2f} ± {c['error']:.2f} | "
                  f"{s['dg']:.2f} ± {s['error']:.2f} | "
                  f"{r['restraint_release']:.2f} | {c['n_windows']} / {s['n_windows']} | "
                  f"{c['precalculation_windows_run']} / {s['precalculation_windows_run']} | {ns:.0f} |")
        for leg in LEGS:
            print(f"\n{leg}: | run | windows per stage (start > rule > +length > +repair = end) | overlap check | "
                  "pair alone lowest | acceptance lowest / highest / CV | longest distance kT | stage lengths kT | "
                  "round trips (per ns) | fewest samples | largest inefficiency |")
            for r in g["runs"]:
                n = r["legs"][leg]
                w = "; ".join(f"{s} {v['start']}>{v['rule']}>+{v['length_added']}>+{v['repair']}={v['end']}"
                              for s, v in n["windows"].items())
                print(f"| {Path(r['run']).name} | {w} | {n['overlap_check']:.3f} | "
                      f"{min(n['overlap_pair_alone'], default=float('nan')):.3f} | "
                      f"{n['acceptance_lowest']:.2f} / {n['acceptance_highest']:.2f} / {n['acceptance_cv']:.2f} | "
                      f"{max(n['distance'], default=float('nan')):.2f} | "
                      + ", ".join(f"{s} {v:.1f}" for s, v in (n['stage_length'] or {}).items())
                      + f" | {n['round_trips']} ({n['round_trips_per_ns']:.1f}) | "
                      f"{n['fewest_independent_samples']:.0f} | {n['largest_inefficiency']:.0f} |")
        for r in g["runs"]:
            e = r.get("equal_cost")
            if e:
                print(f"\nequal cost {Path(r['run']).name}: production {e['production_ns']['complex']:.2f} / "
                      f"{e['production_ns']['solvent']:.2f} ns: dG {e['dg_bind']:.2f} ± {e['error']:.2f}, "
                      f"complex {e['dg_complex']:.2f} ± {e['leg_error']['complex']:.2f}, "
                      f"solvent {e['dg_solvent']:.2f} ± {e['leg_error']['solvent']:.2f}, "
                      f"overlap check {e['overlap_check']:.3f}, fewest samples {e['fewest_independent_samples']:.0f}")


if __name__ == "__main__":
    main()
