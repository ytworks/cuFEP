"""The results of relative runs of one edge side by side, leg by leg and repeat by repeat, as numbers without a
verdict: the free energy of each leg and the ddG with their standard errors, the checks of the analysis (lowest
overlap, fewest independent samples), round trips, how the windows were started, the MD of each leg, and the terminal
flip moves by lambda window. For the comparisons of the plan of 2026-11-28 (issue #27). Reads only.

With ``--timing`` the wall-clock seconds of each leg's phases are printed too (a diagnostic of the machine it ran on;
times measured with the cuPotential plugin go to no tracked file).

    scripts/in_docker.sh python spikes/m44_runs_side_by_side.py [--timing] RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KCAL = 4.184


def main(*args: str) -> None:
    timing = "--timing" in args
    for run_dir in (Path(a) for a in args if a != "--timing"):
        for edge_dir in sorted(path for path in (run_dir / "edges").iterdir() if path.is_dir()):
            print(f"== {run_dir}  {edge_dir.name}")
            for repeat_dir in sorted(edge_dir.glob("repeat_*")):
                legs = {}
                for leg in ("complex", "solvent"):
                    file = repeat_dir / leg / "result.json"
                    if file.exists():
                        legs[leg] = json.loads(file.read_text())
                for leg, r in legs.items():
                    a = r["analysis"]
                    md = r.get("md_ps") or {}
                    flips = r.get("terminal_flip")
                    dg, error = r["delta_g"] / KCAL, r["delta_g_error"] / KCAL
                    line = (f"  {repeat_dir.name} {leg}: dG {dg:+.3f} ± {error:.3f} "
                            f"kcal/mol; {r['n_states']} windows, {r['production_ns']:g} ns; lowest overlap "
                            f"{a['min_overlap']:.3f}, fewest independent samples "
                            f"{a['effective_samples_per_state']:.0f}, "
                            f"round trips {a['round_trips']}; start: {r.get('start_configurations')}; MD "
                            f"{md.get('lambda_windows')} + {md.get('replica_exchange')} = {md.get('total')} ps")
                    if flips:
                        line += (f"; flips accepted/attempted by window "
                                 f"{[f'{x}/{y}' for x, y in zip(flips['accepted'], flips['attempted'])]}")
                    print(line)
                    if timing:
                        t = json.loads((repeat_dir / leg / "timing.json").read_text())
                        print(f"      seconds: {({k: round(v) for k, v in (t.get('phases') or {}).items()})}, backend "
                              f"{t.get('backend')}, {(t.get('sampling') or {}).get('workers')} workers")
                if len(legs) == 2:
                    ddg = (legs["complex"]["delta_g"] - legs["solvent"]["delta_g"]) / KCAL
                    err = (legs["complex"]["delta_g_error"] ** 2 + legs["solvent"]["delta_g_error"] ** 2) ** 0.5 / KCAL
                    print(f"  {repeat_dir.name} ddG {ddg:+.3f} ± {err:.3f} kcal/mol")
        table = run_dir / "edges.csv"
        if table.exists():
            header, *rows = [line.split(",") for line in table.read_text().splitlines()]
            for row in rows:
                d = dict(zip(header, row))
                print(f"  edges.csv: ddG {d['ddg_kcal_mol']} ± {d['standard_error']}, "
                      f"experiment {d['exp_ddg_kcal_mol']}")
        if timing and (run_dir / "timing.json").exists():
            t = json.loads((run_dir / "timing.json").read_text())
            print(f"  run: wall {t.get('wall_seconds'):.0f} s, preparation {t.get('preparation_seconds'):.0f} s, jobs "
                  f"{t.get('jobs_seconds'):.0f} s, GPUs {t.get('gpus')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
