"""Free energy per stage from 1 ns blocks, and how far replicas travel along lambda, per repeat.

    spikes/run.sh spikes/m2_repeat_blocks.py
"""
from pathlib import Path

import numpy as np
import yaml

from cufep.analysis.mbar import solve_mbar
from cufep.sampling.storage import read_samples

KT = 8.31446261815324e-3 * 298.15 / 4.184     # kcal/mol
lig_dir = Path("runs/tyk2_ejm_31/ejm_31")
for leg_name in ("complex", "solvent"):
    print(f"===== {leg_name} leg: free energy (kcal/mol) from 1 ns blocks of the production (iterations 400-2400)")
    for r in range(3):
        leg = lig_dir / f"repeat_{r}/{leg_name}"
        samples = read_samples(leg / "hrex")              # every stored part, to its count
        u = samples["u"][:2400]
        states = samples["states"][:2400]
        sched = yaml.safe_load((leg / "schedule.yaml").read_text())["states"]
        n = len(sched)
        le = np.array([s["lambda_electrostatics"] for s in sched])
        ls = np.array([s["lambda_sterics"] for s in sched])
        i_el0 = int(np.argmax(le == 0.0))            # first window with charges off
        i_r1 = int(np.argmax(np.array([s["lambda_restraints"] for s in sched]) == 1.0))
        rows = []
        for b in range(5):
            sl = slice(400 + 400 * b, 800 + 400 * b)
            f = solve_mbar(u[sl], states[sl])[0]
            f = np.asarray(f) * KT
            rows.append((f[i_r1] - f[0], f[i_el0] - f[i_r1], f[-1] - f[i_el0], f[-1] - f[0]))
        rows = np.array(rows)
        print(
            f"repeat {r} ({n} windows): total per block "
            + " ".join(f"{x:6.2f}" for x in rows[:, 3])
            + " | discharge "
            + " ".join(f"{x:6.2f}" for x in rows[:, 1])
            + " | vanish "
            + " ".join(f"{x:6.2f}" for x in rows[:, 2])
        )
        # replica mobility: range of window indices visited by each replica during the production
        span = states[400:2400].max(0) - states[400:2400].min(0)
        print(
            f"          window range visited by a replica: median {np.median(span):.0f}, max {span.max()} of {n - 1}; "
            f"replicas that reached both ends: "
            f"{int(np.sum((states[400:2400].min(0) == 0) & (states[400:2400].max(0) == n - 1)))}"
        )
