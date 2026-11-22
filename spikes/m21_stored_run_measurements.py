"""Measurements on the stored full benchmark runs (plan
docs/plans/2026-11-22-accuracy-causes-and-fixes.md, step 2). Reads the runs; writes only under
spikes/out/m21/. Numbers only: nothing here decides anything.

    scripts/in_docker.sh python spikes/m21_stored_run_measurements.py paths
    scripts/in_docker.sh python spikes/m21_stored_run_measurements.py stages [--jobs 16]
    scripts/in_docker.sh python spikes/m21_stored_run_measurements.py pocket [--jobs 4] [--set a2a]
    scripts/in_docker.sh python spikes/m21_stored_run_measurements.py bilayer
    scripts/in_docker.sh python spikes/m21_stored_run_measurements.py detect [--jobs 4]

detect   where automatic equilibration detection puts the start of the production of every leg,
         what the leg gives from there on, and the statistics of a set with it
paths   how the start configurations of each leg were made (schedule.yaml), the signed error, the
         parts of the binding free energy and their change from the earlier run
stages   the free energy of each stage from parts of the production (halves, fifths, the last 3
         and 4 ns), the pairs of windows with sterics 0.5 to 0.2, and the statistics of a set if a
         part of the production were taken
pocket   what is in the place of the ligand in every lambda window at the start and at the end of
         the replica-exchange run: water (by molecule), protein atoms, the XY area of the cell
bilayer  XY area of the cell and water near the ligand through the pre-equilibration

The runs: "new" is the run of e77a6c5, "old" the run of 3b45bfe. "chain" and "walk" name how the
first windows of a leg got their start configurations: with one worker each window from the end of
the one before it (chain), with several workers from a short walk through the windows (walk).
No speed or run time is read.
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes/out/m21"
RUNS = {"tyk2": {"new": ROOT / "runs/benchmark_tyk2_full_e77a6c5", "old": ROOT / "runs/benchmark_tyk2_full"},
        "a2a": {"new": ROOT / "runs/benchmark_a2a_full_e77a6c5", "old": ROOT / "runs/benchmark_a2a_full"}}
KCAL = 4.184
LIPIDS = {"PA", "PC", "OL", "POPC", "POP"}
PATHS = ("chain", "walk")
#: lambda values are stored as 1 - x: 0.2 is 0.19999999999999996 and 0.15 is 0.15000000000000002
EPS = 1e-9


# ------------------------------------------------------------------------------------ reading
def report(run: Path) -> dict[str, dict]:
    """benchmark_report.csv of a run: experiment, prediction and signed error per ligand."""
    with (run / "benchmark_report.csv").open() as fh:
        return {r["ligand"]: {"exp": float(r["exp"]), "dg": float(r["cufep"]), "error": float(r["cufep_minus_exp"])}
                for r in csv.DictReader(fh) if r["cufep"] not in ("", None)}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def leg_dir(run: Path, ligand: str, leg: str) -> Path:
    return run / ligand / "repeat_0" / leg


def path_of(run: Path, ligand: str, leg: str) -> str:
    """How the first windows of the leg got their start configurations: "walk", "chain", or "?"
    if the schedule does not record it."""
    file = leg_dir(run, ligand, leg) / "schedule.yaml"
    if not file.exists():
        return "?"
    flag = (yaml.safe_load(file.read_text()).get("provenance") or {}).get("windows_run_concurrently")
    return "?" if flag is None else ("walk" if flag else "chain")


def mean_sd(values) -> tuple[float, float, int]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return (float(x.mean()) if len(x) else np.nan, float(x.std(ddof=1)) if len(x) > 1 else np.nan, len(x))


def difference(a, b) -> str:
    """Difference of the means of two groups with its standard error (the variances over the counts)."""
    (ma, sa, na), (mb, sb, nb) = mean_sd(a), mean_sd(b)
    if na < 2 or nb < 2:
        return "not defined (a group of fewer than two)"
    return f"{ma - mb:+.2f} +- {np.sqrt(sa ** 2 / na + sb ** 2 / nb):.2f}"


def summary(x) -> str:
    x = np.asarray(x, dtype=float)
    return (f"mean {np.nanmean(x):+.2f}, SD {np.nanstd(x, ddof=1):.2f}, negative in {int((x < 0).sum())} of "
            f"{int(np.isfinite(x).sum())}")


# ------------------------------------------------------------------------------------ paths
def ligand_parts(run: Path, ligand: str) -> dict | None:
    lig = read_json(run / ligand / "result.json")
    if lig is None or not lig.get("repeats"):
        return None
    rep = lig["repeats"][0]
    out = {"dg_bind": lig.get("dg_bind"), "complex": rep.get("dg_complex"), "solvent": rep.get("dg_solvent"),
           "release": rep.get("restraint_release")}
    for leg in ("complex", "solvent"):
        res = read_json(leg_dir(run, ligand, leg) / "result.json") or {}
        a = res.get("analysis") or {}
        out[f"{leg}_path"] = path_of(run, ligand, leg)
        out[f"{leg}_windows"] = res.get("n_states")
        out[f"{leg}_tail"] = None if res.get("lrc") is None else res["lrc"] / KCAL
        out[f"{leg}_stages"] = {k: v / KCAL for k, v in (a.get("stage_delta_g") or {}).items()}
        out[f"{leg}_round_trips"] = a.get("round_trips")
    return out


def change(new: dict, old: dict | None, key: str, sub: str | None = None) -> float:
    """new - old of one part (of one stage with ``sub``); NaN where one of the two is missing."""
    a = new[key] if sub is None else new[key].get(sub)
    b = (old or {}).get(key) if sub is None else ((old or {}).get(key) or {}).get(sub)
    return np.nan if a is None or b is None else a - b


def paths() -> None:
    result = {}
    for name, runs in RUNS.items():
        new, old = report(runs["new"]), report(runs["old"])
        rows = {}
        for lig in sorted(new, key=lambda k: new[k]["exp"]):
            rows[lig] = {"exp": new[lig]["exp"], "error_new": new[lig]["error"],
                         "error_old": old.get(lig, {}).get("error"),
                         "new": ligand_parts(runs["new"], lig), "old": ligand_parts(runs["old"], lig)}
        result[name] = rows
        print(f"\n===== {name}: parts of the binding free energy, kcal/mol "
              "(new = e77a6c5, old = 3b45bfe; d = new - old). complex = MBAR part - tail(c): the tail correction is "
              "the number that is subtracted =====")
        print("ligand   exp    | error new  old | path new c/s   old c/s | complex new   d | solvent new   d | "
              "release new   d | tail(c) new   d | restrain d | discharge d | vanish d | windows c new old")
        for lig, r in rows.items():
            n, o = r["new"], r["old"] or {}
            error_old = np.nan if r["error_old"] is None else r["error_old"]
            stages = " | ".join(f"{change(n, o, 'complex_stages', s):+8.2f}"
                                for s in ("restrain", "discharge", "vanish"))
            print(f"{lig:7s} {r['exp']:6.2f} | {r['error_new']:+6.2f} {error_old:+6.2f} |"
                  f" {n['complex_path']:>5s}/{n['solvent_path']:<5s}"
                  f" {o.get('complex_path', '?'):>5s}/{o.get('solvent_path', '?'):<5s} |"
                  f" {n['complex']:7.2f} {change(n, o, 'complex'):+6.2f} |"
                  f" {n['solvent']:7.2f} {change(n, o, 'solvent'):+6.2f} |"
                  f" {n['release']:7.2f} {change(n, o, 'release'):+6.2f} |"
                  f" {n['complex_tail']:7.2f} {change(n, o, 'complex_tail'):+6.2f} | {stages} |"
                  f" {n['complex_windows']} {o.get('complex_windows')}")
        for which in ("new", "old"):
            for leg in ("complex", "solvent"):
                groups = {p: [r[f"error_{which}"] for r in rows.values() if r[which] and r[which][f"{leg}_path"] == p]
                          for p in PATHS}
                text = "; ".join("{}: mean {:+.2f}, SD {:.2f}, n {}".format(p, *mean_sd(v))
                                 for p, v in groups.items() if v)
                print(f"  signed error by the path of the {leg} leg, {which} run: {text}; chain - walk "
                      f"{difference(groups['chain'], groups['walk'])}")
        # the change between the runs by the pair of paths of the complex leg
        pairs: dict[str, list] = {}
        for r in rows.values():
            n, o = r["new"], r["old"]
            if n and o and n["complex"] is not None and o["complex"] is not None:
                key = f"{o['complex_path']} -> {n['complex_path']}"
                pairs.setdefault(key, []).append((n["complex"] + n["release"]) - (o["complex"] + o["release"]))
        for key, v in sorted(pairs.items()):
            print(f"  complex leg + release, new - old, path {key}: mean {np.mean(v):+.2f}, values "
                  + " ".join(f"{x:+.2f}" for x in v))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paths.json").write_text(json.dumps(result, indent=1, default=float))


# ------------------------------------------------------------------------------------ stages
def part_free_energies(samples: dict, a: int, b: int, temperature_K: float) -> np.ndarray | None:
    """Free energies of the windows (kcal/mol, the first is zero) from the iterations [a, b) of the
    stored samples: the functions of the analysis of the package, in the order analyze_leg calls
    them (decorrelated window by window, MBAR), applied to the part on its own. No tail
    correction, no error.

    For the whole production this is the MBAR part of the result (checked in ``stages``). For a
    part it is not what the result reports as a half or a block: the analysis of the package
    keeps for every part the samples that the decorrelation of the WHOLE production kept
    (TimeRangeEstimator), and leaves a part out if a window keeps fewer than four of them there.
    Here the samples of a part are decorrelated by the correlation seen inside that part, which
    is shorter, so a part keeps more samples and every leg has every part. The two differ leg by
    leg by a few tenths of a kcal/mol (review of this script); the difference of the halves that
    the result reports is printed beside the one from here."""
    from cufep.analysis.mbar import (
        R_GAS,
        decorrelated,
        neighbour_inefficiencies,
        solve_decorrelated,
        solve_mbar,
        statistical_inefficiency,
    )

    u, states = samples["u"][a:b], samples["states"][a:b]
    try:
        stride = max(1, int(np.ceil(statistical_inefficiency(u, states))))
        f_start, _df, _overlap = solve_mbar(u[::stride], states[::stride])
        table = neighbour_inefficiencies(u, states)
        g = np.array([np.nanmax(row) if np.isfinite(row).any() else 1.0 for row in table])
        u_kn, n_k = decorrelated(u, states, g)
        f_k, _df_k, _by_number = solve_decorrelated(u_kn, n_k, initial_f_k=f_start, overlap=True)
    except Exception:  # noqa: BLE001  a part without a solution has no numbers
        return None
    return np.asarray(f_k) * R_GAS * temperature_K / KCAL


def leg_stages(args) -> dict:
    run, ligand, leg = args
    import warnings

    from cufep.analysis.mbar import stage_boundaries
    from cufep.sampling.storage import read_samples

    warnings.filterwarnings("ignore")
    directory = leg_dir(Path(run), ligand, leg)
    res = read_json(directory / "result.json")
    samples = read_samples(directory / "hrex")
    attrs = samples["attrs"]
    n_eq, total = int(res["n_equilibration_iterations"]), int(samples["u"].shape[0])
    n = total - n_eq
    per_ns = int(round(1000.0 / float(attrs["iteration_ps"])))
    temperature = float(res["temperature_K"])
    ranges = {"whole": (n_eq, total), "half_1": (n_eq, n_eq + n // 2), "half_2": (n_eq + n // 2, total)}
    for i in range(5):
        ranges[f"fifth_{i + 1}"] = (n_eq + i * (n // 5), n_eq + (i + 1) * (n // 5))
    for ns in (3, 4):
        if n > ns * per_ns:
            ranges[f"last_{ns}ns"] = (total - ns * per_ns, total)
    f = {}
    for key, (a, b) in ranges.items():
        values = part_free_energies(samples, a, b, temperature)
        f[key] = None if values is None else values.tolist()
    a = res["analysis"]
    states = attrs["states"]
    first, second = a.get("first_half"), a.get("second_half")
    reported = None
    if first and second and first[0] is not None and second[0] is not None:
        reported = (second[0] - first[0]) / KCAL            # of the leg (MBAR minus tail), as the result has it
    return {"ligand": ligand, "leg": leg, "f": f, "stages": stage_boundaries(states), "halves_reported": reported,
            "sterics": [s["lambda_sterics"] for s in states],
            "electrostatics": [s["lambda_electrostatics"] for s in states],
            "mbar_reported": a["delta_g_mbar"] / KCAL, "overlap_neighbour": a.get("overlap_neighbour"),
            "swap_neighbour": a.get("swap_acceptance_neighbour"), "production_iterations": n,
            "iterations_per_ns": per_ns}


def stage_value(row: dict, part: str, first: int, last: int) -> float:
    f = row["f"].get(part)
    return np.nan if f is None else f[last] - f[first]


def halves(row: dict, first: int, last: int) -> float:
    """Second half minus first half of the production, of the free energy between two windows."""
    return stage_value(row, "half_2", first, last) - stage_value(row, "half_1", first, last)


def stage_table(name: str, leg: str, legs: dict, table: dict, run: Path) -> None:
    names = list(next(iter(legs.values()))["stages"])
    print(f"\n--- {name} {leg} leg: second half minus first half of the production, by stage, kcal/mol "
          "(each half decorrelated on its own; 'leg' is the sum of the stages, 'result' what result.json has "
          "for the leg, with the samples of the whole production) ---")
    print("ligand   error  path  | " + " ".join(f"{s:>9s}" for s in names) + " |    leg result | vanish "
          "sterics>=0.5  <0.5 | pairs with sterics 0.5-0.2: change, lowest overlap, lowest swap acceptance")
    by_stage = {s: [] for s in names}
    upper, lower, kind, whole_leg, reported = [], [], [], [], []
    for lig, r in legs.items():
        path = path_of(run, lig, leg)
        kind.append(path)
        values = {s: halves(r, *r["stages"][s]) for s in names}
        for s in names:
            by_stage[s].append(values[s])
        a, b = r["stages"]["vanish"]
        ls = np.array(r["sterics"])
        split = max((k for k in range(a, b + 1) if ls[k] >= 0.5 - EPS), default=a)    # last window with sterics >= 0.5
        up, low = halves(r, a, split), halves(r, split, b)
        upper.append(up)
        lower.append(low)
        mid = [k for k in range(a, b) if ls[k + 1] >= 0.2 - EPS and ls[k] <= 0.5 + EPS]    # pairs inside 0.5 .. 0.2
        mid_change = halves(r, mid[0], mid[-1] + 1) if mid else np.nan
        ov = min(r["overlap_neighbour"][k] for k in mid) if mid and r["overlap_neighbour"] else np.nan
        sw = min(r["swap_neighbour"][k] for k in mid) if mid and r["swap_neighbour"] else np.nan
        total = halves(r, 0, len(ls) - 1)
        whole_leg.append(total)
        reported.append(np.nan if r["halves_reported"] is None else r["halves_reported"])
        print(f"{lig:7s} {table[lig]['error']:+6.2f} {path:5s} | " + " ".join(f"{values[s]:+9.2f}" for s in names)
              + f" | {total:+6.2f} {reported[-1]:+6.2f} | {up:+15.2f} {low:+6.2f} | {len(mid):2d} pairs "
              f"{mid_change:+6.2f} {ov:6.3f} {sw:6.3f}")
    both = np.isfinite(reported)
    print(f"  leg: {summary(whole_leg)}; in the {int(both.sum())} legs whose result has both halves: here "
          f"{np.mean(np.array(whole_leg)[both]):+.2f}, result {np.mean(np.array(reported)[both]):+.2f}")
    for s in names:
        x = np.array(by_stage[s])
        print(f"  {s:9s}: {summary(x)}")
        for p in PATHS:
            sel = [v for v, k in zip(x, kind) if k == p and np.isfinite(v)]
            if sel:
                print(f"      {p}: mean {np.mean(sel):+.2f}, n {len(sel)}")
    print(f"  vanish, sterics >= 0.5: {summary(upper)}")
    print(f"  vanish, sterics < 0.5: {summary(lower)}")
    fifths = {s: np.array([[stage_value(r, f"fifth_{i + 1}", *r["stages"][s]) for i in range(5)]
                           for r in legs.values()]) for s in names}
    print("  fifths of the production, mean over the ligands, relative to the first fifth: "
          + "; ".join(f"{s} " + " ".join(f"{v:+.2f}" for v in np.nanmean(fifths[s] - fifths[s][:, :1], axis=0))
                      for s in names))


def stages(jobs: int) -> None:
    from cufep.benchmarks.stats import metrics

    tasks = [(str(runs["new"]), lig, leg) for runs in RUNS.values() for lig in report(runs["new"])
             for leg in ("complex", "solvent")]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        done = list(pool.map(leg_stages, tasks))
    by = {(Path(t[0]).name, r["ligand"], r["leg"]): r for t, r in zip(tasks, done)}
    result = {}
    for name, runs in RUNS.items():
        table = report(runs["new"])
        order = sorted(table, key=lambda k: table[k]["error"])
        rows = {leg: {lig: by[(runs["new"].name, lig, leg)] for lig in order} for leg in ("complex", "solvent")}
        result[name] = rows
        worst = max(abs(r["f"]["whole"][-1] - r["mbar_reported"]) for leg in rows.values() for r in leg.values()
                    if r["f"]["whole"] is not None)
        print(f"\n===== {name}: the whole production by this script against the reported MBAR part: largest "
              f"difference {worst:.4f} kcal/mol =====")
        for leg, legs in rows.items():
            stage_table(name, leg, legs, table, runs["new"])
        # the statistics of the set if a part of the production were taken (MBAR part of both legs;
        # the tail correction and the release of the restraint as reported)
        print(f"\n--- {name}: statistics of the set from parts of the production (kcal/mol) ---")
        exp = np.array([table[lig]["exp"] for lig in order])
        for part in ("whole", "half_1", "half_2", "last_4ns", "last_3ns"):
            pred = []
            for lig in order:
                c, s = rows["complex"][lig], rows["solvent"][lig]
                if c["f"].get(part) is None or s["f"].get(part) is None:
                    pred.append(np.nan)
                    continue
                shift = (c["f"][part][-1] - c["f"]["whole"][-1]) - (s["f"][part][-1] - s["f"]["whole"][-1])
                pred.append(table[lig]["dg"] - shift)
            pred = np.array(pred)
            ok = np.isfinite(pred)
            m = metrics(pred[ok], exp[ok])
            print(f"  {part:9s}: n {int(ok.sum())}, mean signed error {m['mean_signed_error']:+.2f}, RMSE after "
                  f"removing it {m['centred_rmse']:.2f}, Kendall tau {m['kendall_tau']:+.2f}, "
                  f"Pearson r {m['pearson_r']:+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stages.json").write_text(json.dumps(result, indent=1, default=float))


# ------------------------------------------------------------------------------------ detect
def leg_detected(args) -> dict:
    """Where automatic equilibration detection (cufep.analysis.mbar.detect_equilibration) puts the
    start of the production of a stored leg, and the MBAR part of the leg from there on."""
    run, ligand, leg = args
    import warnings

    from cufep.analysis.mbar import detect_equilibration
    from cufep.sampling.storage import read_samples

    warnings.filterwarnings("ignore")
    directory = leg_dir(Path(run), ligand, leg)
    res = read_json(directory / "result.json")
    samples = read_samples(directory / "hrex")
    n_eq, total = int(res["n_equilibration_iterations"]), int(samples["u"].shape[0])
    temperature, iteration_ps = float(res["temperature_K"]), float(samples["attrs"]["iteration_ps"])
    found = detect_equilibration(samples["u"][n_eq:], samples["states"][n_eq:])
    t = 0 if found is None else int(found["iterations"])
    whole = part_free_energies(samples, n_eq, total, temperature)
    after = part_free_energies(samples, n_eq + t, total, temperature)
    return {"ligand": ligand, "leg": leg, "iterations": t, "ns": t * iteration_ps / 1000.0,
            "production_ns": (total - n_eq) * iteration_ps / 1000.0,
            "statistical_inefficiency": None if found is None else found["statistical_inefficiency"],
            "effective_samples": None if found is None else found["effective_samples"],
            "whole": None if whole is None else float(whole[-1]), "after": None if after is None else float(after[-1])}


def detect(jobs: int) -> None:
    from cufep.benchmarks.stats import metrics

    tasks = [(str(runs["new"]), lig, leg) for runs in RUNS.values() for lig in report(runs["new"])
             for leg in ("complex", "solvent")]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        done = list(pool.map(leg_detected, tasks))
    by = {(Path(t[0]).name, r["ligand"], r["leg"]): r for t, r in zip(tasks, done)}
    result = {}
    for name, runs in RUNS.items():
        table = report(runs["new"])
        order = sorted(table, key=lambda k: table[k]["error"])
        rows = {lig: {leg: by[(runs["new"].name, lig, leg)] for leg in ("complex", "solvent")} for lig in order}
        result[name] = rows
        print(f"\n===== {name}: automatic equilibration detection on the production of the stored legs (the MBAR "
              "part of both legs from the detected start; the tail correction and the release of the restraint as "
              "reported), kcal/mol =====")
        print("ligand   error | complex: left out ns, g, change of the leg | solvent: left out ns, g, change of the "
              "leg | error with the detection")
        pred, left = [], {"complex": [], "solvent": []}
        for lig in order:
            c, s = rows[lig]["complex"], rows[lig]["solvent"]
            cells, shift = [], 0.0
            for leg, sign in ((c, 1.0), (s, -1.0)):
                d = np.nan if leg["whole"] is None or leg["after"] is None else leg["after"] - leg["whole"]
                shift += sign * d
                left[leg["leg"]].append(leg["ns"])
                g = np.nan if leg["statistical_inefficiency"] is None else leg["statistical_inefficiency"]
                cells.append(f"{leg['ns']:5.2f} {g:6.1f} {d:+6.2f}")
            pred.append(table[lig]["dg"] - shift)
            print(f"{lig:7s} {table[lig]['error']:+6.2f} | {cells[0]} | {cells[1]} | "
                  f"{pred[-1] - table[lig]['exp']:+6.2f}")
        exp, pred = np.array([table[lig]["exp"] for lig in order]), np.array(pred)
        ok = np.isfinite(pred)
        reported = np.array([table[lig]["dg"] for lig in order])
        for label, values in (("as reported", reported), ("with the detection", pred)):
            m = metrics(values[ok], exp[ok])
            print(f"  {label:18s}: n {int(ok.sum())}, mean signed error {m['mean_signed_error']:+.2f}, RMSE after "
                  f"removing it {m['centred_rmse']:.2f}, Kendall tau {m['kendall_tau']:+.2f}, "
                  f"Pearson r {m['pearson_r']:+.2f}")
        for leg, values in left.items():
            x = np.array(values)
            print(f"  {leg} leg: left out {x.mean():.2f} ns on average (least {x.min():.2f}, most {x.max():.2f}) of "
                  f"{rows[order[0]][leg]['production_ns']:g} ns")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detect.json").write_text(json.dumps(result, indent=1, default=float))


# ------------------------------------------------------------------------------------ pocket
def lengths_angles(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    alpha = np.degrees(np.arccos(np.einsum("ij,ij->i", b, c) / (lb * lc)))
    beta = np.degrees(np.arccos(np.einsum("ij,ij->i", a, c) / (la * lc)))
    gamma = np.degrees(np.arccos(np.einsum("ij,ij->i", a, b) / (la * lb)))
    return np.stack([la, lb, lc], axis=1), np.stack([alpha, beta, gamma], axis=1)


def fitted(reference: np.ndarray, target: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``points`` moved by the rigid motion that puts ``reference`` on ``target`` (Kabsch)."""
    rc, tc = reference.mean(axis=0), target.mean(axis=0)
    h = (reference - rc).T @ (target - tc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (points - rc) @ rot.T + tc


def leg_pocket(args) -> dict:
    run, ligand = args
    import mdtraj as md

    from cufep.sampling.storage import read_samples

    run = Path(run)
    leg = leg_dir(run, ligand, "complex")
    ref = md.load(str(run / ligand / "prep/complex.pdb"))
    top = ref.topology
    prep = np.load(run / ligand / "prep/complex.npz")
    atoms = list(top.atoms)
    lig_heavy = np.array([int(i) for i in prep["ligand"] if atoms[int(i)].element.symbol != "H"])
    prot_heavy = np.array([int(i) for i in prep["protein"] if atoms[int(i)].element.symbol != "H"])
    water_o = np.array([a.index for a in atoms if a.residue.is_water and a.element.symbol == "O"])
    lipid = np.array([a.index for a in atoms if a.residue.name in LIPIDS and a.element.symbol != "H"], dtype=int)
    ca = np.array([int(i) for i in prep["protein"] if atoms[int(i)].name == "CA"])
    near_ca = np.intersect1d(md.compute_neighbors(ref, 1.2, lig_heavy, haystack_indices=ca, periodic=False)[0], ca)
    ref_xyz = ref.xyz[0].astype(float)
    states = yaml.safe_load((leg / "schedule.yaml").read_text())["states"]
    stored = read_samples(leg / "hrex")["states"]

    def describe(positions: np.ndarray, boxes: np.ndarray, label: np.ndarray) -> list[dict]:
        t = md.Trajectory(positions.astype(np.float32), top)
        t.unitcell_lengths, t.unitcell_angles = lengths_angles(boxes)
        w35 = md.compute_neighbors(t, 0.35, lig_heavy, haystack_indices=water_o)
        w20 = md.compute_neighbors(t, 0.20, lig_heavy, haystack_indices=water_o)
        p20 = md.compute_neighbors(t, 0.20, lig_heavy, haystack_indices=prot_heavy)
        none = [np.zeros(0)] * t.n_frames
        l35 = md.compute_neighbors(t, 0.35, lig_heavy, haystack_indices=lipid) if len(lipid) else none
        # the same region in every frame: the ligand of the prepared pose, put on the pocket of the frame
        ghost = md.Trajectory(positions.astype(np.float32), top)
        ghost.unitcell_lengths, ghost.unitcell_angles = t.unitcell_lengths, t.unitcell_angles
        for f in range(ghost.n_frames):
            ghost.xyz[f, lig_heavy] = fitted(ref_xyz[near_ca], positions[f, near_ca], ref_xyz[lig_heavy])
        g35 = md.compute_neighbors(ghost, 0.35, lig_heavy, haystack_indices=water_o)
        g20 = md.compute_neighbors(ghost, 0.20, lig_heavy, haystack_indices=water_o)
        q20 = md.compute_neighbors(ghost, 0.20, lig_heavy, haystack_indices=prot_heavy)
        rows = []
        for r in range(t.n_frames):
            inside = sorted({f"{atoms[int(i)].residue.name}{atoms[int(i)].residue.resSeq}" for i in q20[r]})
            rows.append({"replica": r, "state": int(label[r]),
                         "water35": len(w35[r]), "water20": len(w20[r]), "protein20": len(p20[r]),
                         "lipid35": len(l35[r]),
                         "fixed_water35": [int(i) for i in g35[r]], "fixed_water20": [int(i) for i in g20[r]],
                         "fixed_protein20": len(q20[r]), "fixed_residues": inside,
                         "area": float(np.linalg.norm(np.cross(boxes[r, 0], boxes[r, 1])))})
        return rows

    start = np.load(leg / "start_configs.npz")
    end = np.load(leg / "hrex/checkpoint.npz")
    # a replica is in the window it was simulated in: the first and the last stored row of the
    # states (checkpoint.npz holds the assignment after the last exchange, which is another one)
    rows_start = describe(start["positions"], start["boxes"], stored[0])
    rows_end = describe(end["positions"], end["boxes"], stored[-1])
    return {"ligand": ligand, "states": states, "start": rows_start, "end": rows_end,
            "visited": [[int(stored[:, r].min()), int(stored[:, r].max())] for r in range(stored.shape[1])],
            "checkpoint_iteration": int(end["iteration"]), "stored_iterations": int(stored.shape[0]),
            "n_ligand_heavy": int(len(lig_heavy)), "membrane": bool(len(lipid))}


def cell(x: dict | None) -> str:
    if x is None:
        return "    -    -    -      -"
    return f"{len(x['fixed_water20']):4d} {len(x['fixed_water35']):4d} {x['fixed_protein20']:4d} {x['area']:6.2f}"


def windows_table(name: str, r: dict, error: float, path: str) -> list[dict]:
    """Print the windows of one leg; return its rows of the two summaries (decoupled end, coupled end)."""
    lig = r["ligand"]
    by_start = {x["state"]: x for x in r["start"]}
    by_end = {x["state"]: x for x in r["end"]}
    by_replica = {x["replica"]: x for x in r["end"]}
    print(f"\n== {name} {lig}: error {error:+.2f}, path {path}, {r['n_ligand_heavy']} heavy atoms; "
          f"checkpoint at iteration {r['checkpoint_iteration']}, {r['stored_iterations']} stored")
    print("win restr elec ster | start: w2.0 w3.5 p2.0 area | end: w2.0 w3.5 p2.0 area | replica of the start: "
          "ends in window, entered, left, windows visited")
    for k, st in enumerate(r["states"]):
        a, b = by_start.get(k), by_end.get(k)
        rep = by_replica[a["replica"]] if a else None
        moved = ""
        if a and rep:
            s0, s1 = set(a["fixed_water35"]), set(rep["fixed_water35"])
            lo, hi = r["visited"][a["replica"]]
            moved = f"{rep['state']:3d} {len(s1 - s0):3d} {len(s0 - s1):3d}   {lo}-{hi}"
        print(f"{k:3d} {st['lambda_restraints']:.2f} {st['lambda_electrostatics']:.2f} {st['lambda_sterics']:.2f} | "
              f"      {cell(a)} |     {cell(b)} | {moved}")
    out = []
    decoupled = [k for k, st in enumerate(r["states"]) if st["lambda_sterics"] <= 0.15 + EPS]
    for label, sel in (("sterics <= 0.15", decoupled),
                       ("coupled (first three windows)", [0, 1, 2])):
        a = [by_start[k] for k in sel if k in by_start]
        b = [by_end[k] for k in sel if k in by_end]
        row = {"set": name, "ligand": lig, "path": path, "error": error, "where": label}
        for which, group in (("start", a), ("end", b)):
            row[f"{which}_w20"] = np.mean([len(x["fixed_water20"]) for x in group])
            row[f"{which}_w35"] = np.mean([len(x["fixed_water35"]) for x in group])
            row[f"{which}_p20"] = np.mean([x["fixed_protein20"] for x in group])
            row[f"{which}_area"] = np.mean([x["area"] for x in group])
        out.append(row)
    return out


def pocket_summary(name: str, label: str, rows: list[dict], membrane: bool) -> None:
    """``membrane``: the systems hold a bilayer, so |a x b| of the cell is the area of the bilayer and
    the cell changes it on its own. Without one (a rhombic dodecahedron) it only follows the volume
    of the cell, and its statistics are left out."""
    from scipy import stats

    print(f"\n--- {name}, windows {label}: mean per leg at the start -> at the end ---")
    print("ligand   error  path  | water 2.0 A    | water 3.5 A    | protein 2.0 A  | "
          + ("XY area of the bilayer nm2" if membrane else "|a x b| of the cell nm2 (no bilayer)"))
    for s in rows:
        print(f"{s['ligand']:7s} {s['error']:+6.2f} {s['path']:5s} | {s['start_w20']:5.2f} -> {s['end_w20']:5.2f} | "
              f"{s['start_w35']:5.2f} -> {s['end_w35']:5.2f} | {s['start_p20']:5.2f} -> {s['end_p20']:5.2f} | "
              f"{s['start_area']:6.2f} -> {s['end_area']:6.2f}")
    for key in ("w20", "w35", "p20") + (("area",) if membrane else ()):
        d = np.array([s[f"end_{key}"] - s[f"start_{key}"] for s in rows])
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
        print(f"  {key}: end - start mean {d.mean():+.2f} (positive in {int((d > 0).sum())}, negative in "
              f"{int((d < 0).sum())} of {len(d)}; Wilcoxon p {p:.3f})")
        for which in ("start", "end"):
            groups = {q: [s[f"{which}_{key}"] for s in rows if s["path"] == q] for q in PATHS}
            print(f"      at the {which}: chain mean {mean_sd(groups['chain'])[0]:.2f} (n {len(groups['chain'])}), "
                  f"walk mean {mean_sd(groups['walk'])[0]:.2f} (n {len(groups['walk'])}); chain - walk "
                  f"{difference(groups['chain'], groups['walk'])}")
        x = np.array([s[f"end_{key}"] for s in rows])
        e = np.array([s["error"] for s in rows])
        if np.ptp(x) > 0:
            rho = stats.spearmanr(x, e)
            print(f"      at the end against the signed error: Spearman {rho[0]:+.2f} (p {rho[1]:.2f})")


def pocket(jobs: int, only: list[str]) -> None:
    result = {}
    for name in only:
        runs = RUNS[name]
        table = report(runs["new"])
        order = sorted(table, key=lambda k: table[k]["error"])
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(leg_pocket, [(str(runs["new"]), lig) for lig in order]))
        result[name] = {r["ligand"]: r for r in done}
        print(f"\n===== {name}: the place of the ligand at the start and at the end of the replica-exchange run =====")
        print("Counts in the region of the prepared pose put on the pocket of each configuration: water oxygens\n"
              "within 2.0 A and 3.5 A of a ligand heavy atom position, protein heavy atoms within 2.0 A.\n"
              "'entered/left': water molecules (3.5 A) of the same replica that are there at the end and were not\n"
              "at the start, and the reverse.")
        rows = []
        for r in done:
            lig = r["ligand"]
            rows.extend(windows_table(name, r, table[lig]["error"], path_of(runs["new"], lig, "complex")))
        for label in ("sterics <= 0.15", "coupled (first three windows)"):
            pocket_summary(name, label, [s for s in rows if s["where"] == label], all(r["membrane"] for r in done))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"pocket_{name}.json").write_text(json.dumps(result[name], indent=1, default=float))


# ------------------------------------------------------------------------------------ bilayer
def series_numbers(time: np.ndarray, x: np.ndarray) -> tuple[list[float], float, float]:
    """Means of the fifths of a series, its slope per ns over the whole and over the last half."""
    q, half = len(x) // 5, len(x) // 2
    fifths = [float(x[i * q:(i + 1) * q].mean()) for i in range(5)]
    return fifths, float(np.polyfit(time, x, 1)[0]), float(np.polyfit(time[half:], x[half:], 1)[0])


def bilayer() -> None:
    result = {}
    for which in ("new", "old"):
        run = RUNS["a2a"][which]
        table = report(run)
        print(f"\n===== a2a, {which} run: the MD of the pre-equilibration of the complex after its NVT and NPT "
              "(equil.npz), fifths of the stored series. The two runs are two protocols: in the new one the "
              "heavy atoms of protein and ligand are held through this MD, in the old one they are free, so the "
              "numbers of the water are not comparable between the runs =====")
        print("ligand   error  path  solute | ns    | XY area nm2, fifths            slope/ns all, last half | water "
              "within 6 A of a ligand atom (hydrogens too), fifths   slope/ns all, last half")
        rows = {}
        for lig in sorted(table, key=lambda k: table[k]["error"]):
            eq = np.load(leg_dir(run, lig, "complex") / "equil.npz", allow_pickle=True)
            made = json.loads(str(eq["made_with"])) if "made_with" in eq.files else {}
            held = {True: "held", False: "free"}.get(made.get("solute_held"), "?")
            area = np.atleast_1d(np.asarray(eq["area_nm2"], dtype=float)) if "area_nm2" in eq.files else np.zeros(0)
            water = np.zeros(0)
            if "pocket_waters" in eq.files:
                water = np.atleast_1d(np.asarray(eq["pocket_waters"], dtype=float))
            ns = float(eq["simulated_ns"]) if "simulated_ns" in eq.files else np.nan
            if len(area) < 10 or len(water) != len(area):
                print(f"{lig:7s} no series ({len(area)} values of the area, {len(water)} of the water)")
                continue
            time = (np.arange(len(area)) + 1.0) * ns / len(area)        # a frame at the end of every interval
            fa, sa, sa2 = series_numbers(time, area)
            fw, sw, sw2 = series_numbers(time, water)
            rows[lig] = {"error": table[lig]["error"], "path": path_of(run, lig, "complex"), "ns": ns,
                         "area_fifths": fa, "area_slope": sa, "area_slope_last_half": sa2, "water_fifths": fw,
                         "water_slope": sw, "water_slope_last_half": sw2, "frames": len(area), "solute": held}
            print(f"{lig:7s} {table[lig]['error']:+6.2f} {rows[lig]['path']:5s} {held:6s} | {ns:5.2f} | "
                  + " ".join(f"{v:6.2f}" for v in fa) + f"  {sa:+6.3f} {sa2:+6.3f} | "
                  + " ".join(f"{v:5.1f}" for v in fw) + f"  {sw:+6.3f} {sw2:+6.3f}")
        if rows:
            for key, label in (("area_slope", "area, slope over the whole series"),
                               ("area_slope_last_half", "area, slope over the last half"),
                               ("water_slope", "water, slope over the whole series"),
                               ("water_slope_last_half", "water, slope over the last half")):
                x = np.array([r[key] for r in rows.values()])
                print(f"  {label}: mean {x.mean():+.3f} per ns, negative in {int((x < 0).sum())} of {len(x)}")
            last = np.array([r["area_fifths"][4] - r["area_fifths"][0] for r in rows.values()])
            print(f"  area, last fifth minus first fifth: mean {last.mean():+.2f} nm2, range {last.min():+.2f} to "
                  f"{last.max():+.2f}")
        result[which] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bilayer.json").write_text(json.dumps(result, indent=1, default=float))


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in ("paths", "stages", "detect", "pocket", "bilayer"):
        raise SystemExit(__doc__)
    jobs, only = (16 if args[0] == "stages" else 4), list(RUNS)
    rest = args[1:]
    while rest:
        a = rest.pop(0)
        if a == "--jobs":
            jobs = int(rest.pop(0))
        elif a == "--set":
            only = [rest.pop(0)]
        else:
            raise SystemExit(f"unknown argument {a}")
    if args[0] == "paths":
        paths()
    elif args[0] == "stages":
        stages(jobs)
    elif args[0] == "detect":
        detect(jobs)
    elif args[0] == "pocket":
        pocket(jobs, only)
    else:
        bilayer()


if __name__ == "__main__":
    main()
