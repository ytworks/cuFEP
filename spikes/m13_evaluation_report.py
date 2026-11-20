"""The table of plan revision 13, B6: for each run, one calculation of one compound, the binding free
energy with its standard error, the experiment and the signed error, and for each leg every check of
the analysis on that same trajectory.

    spikes/run.sh spikes/m13_evaluation_report.py runs/tyk2_r13_b3:ejm_31:-9.54 runs/a2a_r13_b4:4e:-12.07 ...

Nothing is written; the table goes to standard output (Markdown).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KCAL = 4.184
KT = 0.0083144626 * 298.15            # kJ/mol


def leg_line(name: str, res: dict) -> str:
    """One leg: the two checks of the literature (made again with the code of the tree and the default levels:
    overlap >= 0.03, Mey's 50 uncorrelated samples) and, as numbers without a verdict, the halves and BAR - MBAR."""
    from cufep.analysis.mbar import quality_checks

    a = res["analysis"]
    q = quality_checks(a)
    s = q["samples"]
    first, second = a.get("first_half"), a.get("second_half")
    try:
        halves = abs(first[0] - second[0])
        sigma = float(np.hypot(first[1], second[1]))
        compared = bool(np.isfinite(halves))
    except (TypeError, IndexError):
        compared = False
    halves_cell = f"{halves / KT:.2f} kT ({halves / sigma:.1f} σ)" if compared else "-"
    added = res.get("lambda_windows_added") or 0
    added = sum(added.values()) if isinstance(added, dict) else added
    windows = res.get("n_states") or a.get("n_states")
    used = a.get("min_overlap_used")
    bar, mbar = a.get("bar_delta_g"), a.get("delta_g_mbar")
    bar_cell = "-" if bar is None or mbar is None else f"{abs(bar - mbar) / KCAL:.2f}"
    g = s.get("statistical_inefficiency")
    return (
        f"| {name} | {windows} (+{added}) | {res['delta_g_decouple'] / KCAL:.2f} ± {res['delta_g_error'] / KCAL:.2f} "
        f"| {a['min_overlap']:.3f} / {'-' if used is None else f'{used:.3f}'} "
        f"| {s.get('fewest', float('nan')):.0f} / {s.get('kept')} (window {s.get('window')}, g "
        f"{'-' if g is None else f'{g:.0f}'}) "
        f"| {'; '.join(k for k in ('overlap', 'samples') if q[k].get('ok') is False) or 'none'} "
        f"| {halves_cell} | {bar_cell} |"
    )


def main() -> None:
    for spec in sys.argv[1:]:
        run, ligand, experiment = spec.split(":")
        run, experiment = Path(run), float(experiment)
        lig = json.loads((run / ligand / "result.json").read_text())
        dg = lig.get("dg_bind")
        print(f"### {run.name} ({ligand})\n")
        if dg is None:
            print(f"no binding free energy: {lig.get('warnings')}\n")
            continue
        error = lig.get("dg_bind_error", lig.get("dg_bind_mbar_error", float("nan")))
        print(
            f"dG_bind {dg:.2f} ± {error:.2f} kcal/mol (standard error, one calculation; "
            f"MBAR alone ± {lig.get('dg_bind_mbar_error', float('nan')):.2f}); experiment {experiment:.2f}; signed "
            f"error "
            f"{dg - experiment:+.2f}\n"
        )
        print(
            "| leg | windows (added) | dG of the leg, kcal/mol | lowest overlap: all / used | fewest samples: "
            "estimated / kept "
            "| checks not met (overlap 0.03, 50 samples) | halves (no verdict) | BAR - MBAR, kcal/mol (no verdict) |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for leg in ("complex", "solvent"):
            res = json.loads((run / ligand / "repeat_0" / leg / "result.json").read_text())
            print(leg_line(leg, res))
        equil = run / ligand / "repeat_0" / "complex" / "equil.npz"
        if equil.exists():
            with np.load(equil) as e:
                if "membrane_drift" in e.files:
                    drift = json.loads(str(e["membrane_drift"]))
                    if drift:
                        water = drift.get("pocket_water_change")
                        print(f"\nbilayer between the last two quarters of its equilibration (no verdict): area "
                              f"{100 * drift['area_change']:+.2f} %, water molecules near the ligand "
                              f"{'none' if water is None else f'{water:+.2f}'}")
        notes = [w for w in lig.get("warnings", [])]
        if notes:
            print("\nwarnings: " + "; ".join(notes))
        print()


if __name__ == "__main__":
    main()
