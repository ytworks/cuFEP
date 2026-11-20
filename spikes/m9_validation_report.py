"""Summary of one or more finished runs for the protocol validation (plan step 9).

For every repeat: free energies of the legs and of binding, lambda windows per stage, the
lowest overlap and exchange acceptance between neighbouring windows, and how far replicas
travelled. Timings are printed only with --timing (they are for internal use).

    spikes/run.sh spikes/m9_validation_report.py runs/val_tyk2_adaptive runs/val_tyk2_fixed [--timing] [--pairs]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KJ = 4.184
args = [a for a in sys.argv[1:] if not a.startswith("--")]
timing = "--timing" in sys.argv
pairs = "--pairs" in sys.argv          # overlap and exchange of every pair of neighbouring windows

for run in map(Path, args):
    print(f"\n===== {run}")
    for lig_dir in sorted(p for p in run.iterdir() if (p / "prep").exists()):
        binds = []
        for rep in sorted(lig_dir.glob("repeat_*")):
            legs = {}
            for leg in ("complex", "solvent"):
                f = rep / leg / "result.json"
                if f.exists():
                    legs[leg] = json.loads(f.read_text())
            if len(legs) < 2:
                print(f"{lig_dir.name} {rep.name}: incomplete ({', '.join(legs) or 'no leg finished'})")
                continue
            c, s = legs["complex"], legs["solvent"]
            bind = (s["delta_g_decouple"] - c["delta_g_decouple"] - c["restraint_release"]) / KJ
            binds.append(bind)
            print(
                f"{lig_dir.name} {rep.name}: dG_bind {bind:7.2f} kcal/mol | complex {c['delta_g_decouple'] / KJ:6.2f} "
                f"(MBAR ± {c['delta_g_mbar_error'] / KJ:.2f}) solvent {s['delta_g_decouple'] / KJ:6.2f} "
                f"(± {s['delta_g_mbar_error'] / KJ:.2f}) release {c['restraint_release'] / KJ:6.2f}"
            )
            for leg, r in legs.items():
                a = r["analysis"]
                windows = r.get("lambda_windows", {})
                acc = [
                    x
                    for x in (a.get("swap_acceptance_neighbour") or a.get("neighbour_transition_fraction") or [])
                    if x is not None
                ]
                stage = ", ".join(f"{k} {v / KJ:.2f}" for k, v in a["stage_delta_g"].items())
                print(
                    f"    {leg:8s} {r['n_states']:2d} windows ("
                    + ", ".join(f"{k} {len(v)}" for k, v in windows.items())
                    + f"); stages {stage}; "
                    f"lowest overlap {a['min_overlap']:.3f}; neighbour exchange lowest {min(acc):.2f} mean "
                    f"{np.mean(acc):.2f}; "
                    f"round trips {a['round_trips']}; N_eff/window {a['effective_samples_per_state']:.0f}; "
                    f"forward/reverse gap {a['forward_reverse_gap'] / KJ:.2f}"
                )
                for note in r.get("notes", []):
                    print(f"        note: {note}")
                if pairs:
                    k = 0
                    for name, vals in windows.items():
                        n = len(vals) - 1
                        print(f"        {name:9s} lambda   " + " ".join(f"{v:5.2f}" for v in vals))
                        print(
                            f"        {'':9s} overlap     "
                            + " ".join(f"{v:5.3f}" for v in a["overlap_neighbour"][k : k + n])
                        )
                        print(f"        {'':9s} exchange    " + " ".join(f"{v:5.3f}" for v in acc[k:k + n]))
                        k += n
                if timing and (rep / leg / "timing.json").exists():
                    t = json.loads((rep / leg / "timing.json").read_text())
                    smp = t.get("sampling", {})
                    print("        timing: " + ", ".join(f"{k} {v:.0f} s" for k, v in t["phases"].items())
                          + (f"; {smp['seconds_per_iteration']:.3f} s/iteration, workers busy "
                             f"{100 * (smp.get('worker_busy_fraction') or 0):.0f}%" if smp else ""))
        if binds:
            sd = f", SD {np.std(binds, ddof=1):.2f}" if len(binds) > 1 else ""
            print(f"{lig_dir.name}: mean dG_bind {np.mean(binds):.2f} kcal/mol over {len(binds)} repeat(s){sd}")
    if timing and (run / "timing.json").exists():
        t = json.loads((run / "timing.json").read_text())
        util = t["gpu_utilization_percent"]["while_sampling"]
        print(
            f"timing: wall {t['wall_seconds'] / 60:.1f} min, preparation {t['preparation_seconds'] / 60:.1f} min, MPS "
            f"{t['mps']}; "
            "GPU utilization while sampling: " + ", ".join(f"{g}: {v:.0f}%" for g, v in util.items() if v is not None)
        )
