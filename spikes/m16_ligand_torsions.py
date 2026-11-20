"""Torsions of the ligand in a benchmark run, for every ligand and both legs. Reads only.

For every torsion across a non-ring single bond between heavy atoms (amide bonds marked), and the
state it has in the prepared input pose (the bound pose of the benchmark): the fraction of the
frames of the coupled window (lambda 0, all of the production) in which it is within 90 degrees of
that state, how often it changed between the two states there, and in how many windows it is
turned (more than 90 degrees away) at the end of the run (checkpoint). A torsion is listed if it
was turned anywhere; the free energy of each leg is given for comparison.

    scripts/in_docker.sh python spikes/m16_ligand_torsions.py runs/benchmark_tyk2_full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from rdkit import Chem
from stored_frames import stored_frames

KCAL = 4.184
ROTATABLE = Chem.MolFromSmarts("[!$([NH2]);!$([CH3]);!D1]-&!@[!$([NH2]);!$([CH3]);!D1]")


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :], x[..., l, :] - x[..., k, :]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def turned(angle, reference) -> np.ndarray:
    return np.abs((angle - reference + 180.0) % 360.0 - 180.0) > 90.0


def torsions_of(mol) -> list[tuple[str, tuple[int, int, int, int]]]:
    out = []
    for j, k in mol.GetSubstructMatches(ROTATABLE):
        a, b = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
        i = next(n.GetIdx() for n in sorted(a.GetNeighbors(), key=lambda n: -n.GetAtomicNum()) if n.GetIdx() != k)
        l = next(n.GetIdx() for n in sorted(b.GetNeighbors(), key=lambda n: -n.GetAtomicNum()) if n.GetIdx() != j)
        amide = (a.GetSymbol(), b.GetSymbol()) in (("C", "N"), ("N", "C")) and any(
            n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
            for c in (a, b) if c.GetSymbol() == "C" for n in c.GetNeighbors())
        label = "-".join(f"{mol.GetAtomWithIdx(x).GetSymbol()}{x}" for x in (i, j, k, l))
        out.append((label + (" amide" if amide else ""), (i, j, k, l)))
    return out


def main(run_dir: str) -> None:
    from cufep.workflow import load_prepared

    run = Path(run_dir)
    print("| ligand | leg | dG of the leg | torsion | input angle | coupled window: fraction as input | changes | "
          "windows turned at the end |")
    print("|---|---|---|---|---|---|---|---|")
    for lig_dir in sorted(p for p in run.iterdir() if (p / "prep" / "ligand.json").exists()):
        mol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
        torsions = torsions_of(mol)
        for leg in ("complex", "solvent"):
            d = lig_dir / "repeat_0" / leg
            if not (d / "result.json").exists():
                continue
            dg = json.loads((d / "result.json").read_text())["delta_g_decouple"] / KCAL
            prep = load_prepared(lig_dir / "prep", leg)
            lig = np.asarray(prep["ligand"], dtype=int)
            reference = np.asarray(prep["positions"])[lig]
            atoms = list(np.load(d / "equil.npz")["frame_atoms"])
            frames = stored_frames(d / "hrex")[:, [atoms.index(a) for a in lig], :]
            ck = np.load(d / "hrex" / "checkpoint.npz")
            ends = ck["positions"][np.argsort(ck["state_of_replica"])][:, lig, :]
            shown = False
            for label, q in torsions:
                ref = float(dihedral(reference, *q))
                series = turned(dihedral(frames, *q), ref)
                at_end = turned(dihedral(ends, *q), ref)
                if not series.any() and not at_end.any():
                    continue
                shown = True
                changes = int(np.count_nonzero(np.diff(series.astype(int))))
                marks = "".join("x" if t else "." for t in at_end)
                print(f"| {lig_dir.name} | {leg} | {dg:.2f} | {label} | {ref:.0f} | {1 - series.mean():.2f} "
                      f"({len(series)} frames) | {changes} | {int(at_end.sum())} of {len(at_end)} ({marks}) |")
            if not shown:
                print(f"| {lig_dir.name} | {leg} | {dg:.2f} | none turned ({len(torsions)} torsions) | | | | |")


if __name__ == "__main__":
    main(sys.argv[1])
