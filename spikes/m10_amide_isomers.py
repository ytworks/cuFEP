"""Count amide bonds of the ligand that left their input isomer, per lambda window, at the end of a run.

REST2 lowers every torsion barrier of the hot region. If the scale is too high, the amide
bonds (omega) of the ligand isomerise in the hot windows and the isomers spread over the
windows through replica exchange.

    spikes/run.sh spikes/m10_amide_isomers.py runs/benchmark_tyk2_subset_rest2 runs/benchmark_tyk2_subset_r3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from rdkit import Chem


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :], x[..., l, :] - x[..., k, :]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def main() -> None:
    from cufep.workflow import load_prepared

    amide = Chem.MolFromSmarts("[OX1]=[CX3]-[NX3]-[#6]")
    for run in map(Path, sys.argv[1:]):
        print(run.name)
        for lig_dir in sorted(p for p in run.iterdir() if (p / "prep").exists()):
            mol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
            quads = sorted({m for m in mol.GetSubstructMatches(amide)})
            if not quads:
                continue
            for rep in sorted(lig_dir.glob("repeat_*")):
                out = [f"  {lig_dir.name} {rep.name}:"]
                for leg in ("complex", "solvent"):
                    ck_file = rep / leg / "hrex" / "checkpoint.npz"
                    if not ck_file.exists():
                        continue
                    prep = load_prepared(lig_dir / "prep", leg)
                    ligand = np.array(prep["ligand"])
                    reference = prep["positions"][ligand]
                    ck = np.load(ck_file)
                    x = ck["positions"][np.argsort(ck["state_of_replica"])][:, ligand, :]
                    flipped = np.zeros(len(x), dtype=int)
                    for q in quads:
                        ref = dihedral(reference, *q)
                        now = dihedral(x, *q)
                        flipped += (np.abs((now - ref + 180.0) % 360.0 - 180.0) > 90.0)
                    out.append(f"{leg} {int((flipped > 0).sum())} of {len(x)} windows with an isomerised amide ("
                               + "".join("x" if f else "." for f in flipped) + ")")
                print(" ".join(out[:1]) + " " + "; ".join(out[1:]) + f"   [{len(quads)} amide torsions]")


if __name__ == "__main__":
    main()
