"""Count the interactions of the trajectories of a complex leg again, by ProLIF's own run, and
compare with what the leg recorded (``interactions.json``).

    scripts/in_docker.sh python spikes/m17_recount_interactions.py runs/<name>/<ligand>/repeat_0/complex

The leg found its interactions with molecules that are made once and take the coordinates of each
frame (:mod:`cufep.analysis.interactions`). Here ProLIF converts protein and ligand again for every
frame from an MDAnalysis universe, as its ``Fingerprint.run`` does, with the ligand's chemistry
perceived by MDAnalysis and not taken from the prepared molecule. Printed: for every ligand atom
L1, residue and type of interaction the frames in which it is there by both counts and the number
of frames that only one of the two counts has, and the largest such number (a count per residue
and type alone would not see two ligand atoms or two sets of frames that changed places). The
ligand atom that ProLIF names is given its L1 by the rule of the leg (``LigandGraph.anchor_atom``):
the count is a second one of the interactions and of their frames, not of that rule, which the
unit tests hold to known molecules.
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
    from openff.toolkit import Molecule

    from cufep.alchemy.interaction_anchors import LigandGraph
    from cufep.alchemy.restraints import ligand_next_to_protein
    from cufep.analysis.interactions import INTERACTIONS, VICINITY_CUTOFF_A, protein_atoms_of, with_rdkit_aromaticity
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
    protein_rows = np.array([row[int(i)] for i in protein_atoms_of(topology, among=row)])
    xyz = ligand_next_to_protein(analysed.frames, np.array([row[i] for i in ligand]), protein_rows,
                                 np.asarray(analysed.boxes, dtype=float))
    n = xyz.shape[0]
    whole = np.zeros((n, topology.getNumAtoms(), 3), dtype=np.float32)
    whole[:, frame_atoms, :] = xyz * 10.0
    universe = mda.Universe(topology, whole, format="MEMORY")
    universe.residues.resids = np.arange(universe.residues.n_residues)
    fingerprint = prolif.Fingerprint(list(INTERACTIONS), count=True, vicinity_cutoff=VICINITY_CUTOFF_A)
    fingerprint.run(universe.trajectory, universe.atoms[np.array(ligand)],
                    universe.atoms[protein_atoms_of(topology, among=row)], n_jobs=8, progress=False)
    # the atom L1 of an interaction by the rule of the leg, from the ligand atom ProLIF names first
    rdmol = Molecule.from_json((leg.parent.parent / "prep" / "ligand.json").read_text()).to_rdkit()
    graph = LigandGraph.from_rdkit(with_rdkit_aromaticity(rdmol))
    again: dict[tuple[int, int, str], set[int]] = {}
    for f in range(n):
        for (_l, residue), by_type in fingerprint.ifp.get(f, {}).items():
            for name, occurrences in by_type.items():
                for occurrence in occurrences:
                    l1 = graph.anchor_atom(int(occurrence["parent_indices"]["ligand"][0]))
                    if l1 is not None:
                        again.setdefault((int(ligand[l1]), int(residue.number), str(name)), set()).add(f)
    recorded = {(int(r["ligand_atom"]), int(r["residue"]), r["interaction"]): {int(f) for f in r["in_frames"]}
                for r in report["interactions"]}
    names = {(int(r["ligand_atom"]), int(r["residue"])): f"{r['ligand_atom_name']} {r['residue_name']}"
             for r in report["interactions"]}
    print(f"{n} frames; recorded {len(recorded)} interactions (ligand atom, residue, type), counted again {len(again)}")
    worst = 0
    for key in sorted(set(recorded) | set(again), key=lambda k: -max(len(recorded.get(k, ())), len(again.get(k, ())))):
        a, b = recorded.get(key, set()), again.get(key, set())
        worst = max(worst, len(a ^ b))
        print(f"  {names.get(key[:2], key[:2])!s:24s} {key[2]:11s} recorded {len(a):4d}  again {len(b):4d}  "
              f"frames in one count only {len(a ^ b):4d}")
    print(f"largest number of frames in one count only: {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
