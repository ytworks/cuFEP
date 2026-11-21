"""Count the interactions of the trajectories of a complex leg again, by ProLIF's own run, and
compare with what the leg recorded (``interactions.json``).

    scripts/in_docker.sh python spikes/m17_recount_interactions.py runs/<name>/<ligand>/repeat_0/complex

The leg found its interactions with molecules that are made once and take the coordinates of each
frame (:mod:`cufep.analysis.interactions`). Here ProLIF converts protein and ligand again for every
frame from an MDAnalysis universe, as its ``Fingerprint.run`` does, with the ligand's chemistry
perceived by MDAnalysis and not taken from the prepared molecule. Printed: for every residue and
type of interaction the frames in which it is there by both counts, and the largest difference.
The frames are those the leg analysed (later than ``discard_ns`` of its record), with the ligand
in the periodic image next to the protein.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(leg_dir: str) -> int:
    import MDAnalysis as mda
    import prolif

    from cufep.alchemy.restraints import backbone_hosts, nearest_image_of_ligand
    from cufep.analysis.interactions import INTERACTIONS, VICINITY_CUTOFF_A, protein_atoms_of
    from cufep.sampling.restraint_md import read_trajectories
    from cufep.workflow import load_prepared

    leg = Path(leg_dir)
    report = json.loads((leg / "interactions.json").read_text())
    settings = report["made_with"]["interactions"]
    prep = load_prepared(leg.parent.parent / "prep", "complex")
    topology, ligand = prep["topology"], [int(i) for i in prep["ligand"]]
    analysed = read_trajectories(leg, int(settings["trajectories"])).after(float(settings["discard_ns"]))
    frame_atoms = np.asarray(analysed.frame_atoms, dtype=int)
    row = {int(a): i for i, a in enumerate(frame_atoms)}
    c_alpha = [row[trio[1]] for trio in backbone_hosts(topology).values() if trio[1] in row]
    xyz = nearest_image_of_ligand(analysed.frames, np.array([row[i] for i in ligand]),
                                  np.asarray(analysed.frames, dtype=float)[:, c_alpha, :].mean(axis=1),
                                  np.asarray(analysed.boxes, dtype=float))
    n = xyz.shape[0]
    whole = np.zeros((n, topology.getNumAtoms(), 3), dtype=np.float32)
    whole[:, frame_atoms, :] = xyz * 10.0
    universe = mda.Universe(topology, whole, format="MEMORY")
    universe.residues.resids = np.arange(universe.residues.n_residues)
    fingerprint = prolif.Fingerprint(list(INTERACTIONS), count=True, vicinity_cutoff=VICINITY_CUTOFF_A)
    fingerprint.run(universe.trajectory, universe.atoms[np.array(ligand)],
                    universe.atoms[protein_atoms_of(topology, among=row)], n_jobs=8, progress=False)
    again: dict[tuple[int, str], int] = {}
    for f in range(n):
        for key in {(int(residue.number), str(name)) for (_l, residue), by_type in fingerprint.ifp.get(f, {}).items()
                    for name in by_type}:
            again[key] = again.get(key, 0) + 1
    recorded = {(int(r["residue"]), r["interaction"]): int(r["frames"]) for r in report["residues"]}
    names = {int(r["residue"]): r["residue_name"] for r in report["residues"]}
    print(f"{n} frames; recorded {len(recorded)} residue/type pairs, counted again {len(again)}")
    worst = 0
    for key in sorted(set(recorded) | set(again), key=lambda k: -max(recorded.get(k, 0), again.get(k, 0))):
        a, b = recorded.get(key, 0), again.get(key, 0)
        worst = max(worst, abs(a - b))
        print(f"  {names.get(key[0], key[0])!s:10s} {key[1]:11s} recorded {a:4d}  again {b:4d}")
    print(f"largest difference: {worst} frame(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
