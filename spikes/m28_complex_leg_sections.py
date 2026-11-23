"""Complex legs of one ligand with the same lambda windows, side by side: the free energy of each
section of the path (MBAR, kcal/mol; the sections hold the REST2 scaling of their windows, so they
are compared between legs, not read alone), the water relocation moves that were accepted, and
what is in the place of the ligand in every window at the start and at the end of the run (water
oxygens within 0.35 nm and protein heavy atoms within 0.2 nm of the heavy atoms of the ligand of
the prepared pose, fitted on the pocket). Reads the runs; writes nothing. Numbers only.

    scripts/in_docker.sh python spikes/m28_complex_leg_sections.py LIGAND RUN [RUN ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m21_stored_run_measurements import leg_pocket  # noqa: E402

KCAL = 4.184


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ligand, runs = sys.argv[1], sys.argv[2:]
    for run in runs:
        leg = Path(run) / ligand / "repeat_0" / "complex"
        result = json.loads((leg / "result.json").read_text())
        f = np.array(result["analysis"]["f_k"]) / KCAL                      # stored in kJ/mol
        pocket = leg_pocket((run, ligand))
        states = pocket["states"]
        sterics = np.array([s["lambda_sterics"] for s in states])
        charges = np.array([s["lambda_electrostatics"] for s in states])
        restraint = np.array([s["lambda_restraints"] for s in states])
        marks = {"restraint on": int(np.flatnonzero(restraint >= 1.0)[0]),
                 "charges off": int(np.flatnonzero(charges <= 0.0)[0]),
                 "sterics 0.5": int(np.flatnonzero(sterics <= 0.5)[0]),
                 "sterics 0.25": int(np.flatnonzero(sterics <= 0.25)[0]), "sterics 0": len(states) - 1}
        print(f"{run}: complex leg {result['delta_g_decouple'] / KCAL:.2f} +- {result['delta_g_error'] / KCAL:.2f} "
              f"(MBAR {f[-1] - f[0]:.2f}), {len(states)} windows")
        last, parts = 0, []
        for name, index in marks.items():
            parts.append(f"to {name} (window {index}) {f[index] - f[last]:+.2f}")
            last = index
        print("   " + "; ".join(parts))
        moves = result.get("relocation_mc")
        if moves:
            for way in ("into", "out_of"):
                accepted, attempted = moves[f"accepted_{way}_the_sphere"], moves[f"attempted_{way}_the_sphere"]
                print(f"   moves accepted {way.replace('_', ' ')} the sphere, by window: "
                      + " ".join(str(n) for n in accepted) + f" ({sum(accepted)} of {sum(attempted)} attempted)")
        for label in ("start", "end"):
            rows = sorted(pocket[label], key=lambda row: row["state"])
            water = " ".join(f"{len(r['fixed_water35']):2d}" for r in rows)
            protein = " ".join(f"{r['fixed_protein20']:2d}" for r in rows)
            print(f"   water in the place of the ligand, {label}:   {water}")
            print(f"   protein in the place of the ligand, {label}: {protein}")


if __name__ == "__main__":
    main()
