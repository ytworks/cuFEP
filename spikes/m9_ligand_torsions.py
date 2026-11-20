"""Which torsion of the ligand differs between repeats of the solvent leg?

For every rotatable bond of the ligand: the distribution of its dihedral angle in the frames
that the coupled window (lambda 0) wrote during replica exchange, and in the start
configurations of all windows, for each repeat.

    spikes/run.sh spikes/m9_ligand_torsions.py runs/val_tyk2_adaptive ejm_31 [solvent|complex]
"""
import sys
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from rdkit import Chem
from stored_frames import stored_frames


def dihedral(x, i, j, k, l):
    b0, b1, b2 = x[..., i, :] - x[..., j, :], x[..., k, :] - x[..., j, :], x[..., l, :] - x[..., k, :]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def main() -> None:
    run_dir, ligand = Path(sys.argv[1]), sys.argv[2]
    leg = sys.argv[3] if len(sys.argv) > 3 else "solvent"
    lig_dir = run_dir / ligand
    from cufep.workflow import load_prepared
    prep = load_prepared(lig_dir / "prep", leg)
    lig = np.asarray(prep["ligand"], dtype=int)
    mol = Molecule.from_json((lig_dir / "prep" / "ligand.json").read_text()).to_rdkit()
    patt = Chem.MolFromSmarts("[!$([NH2]);!$([CH3]);!D1]-&!@[!$([NH2]);!$([CH3]);!D1]")
    torsions = []
    for j, k in mol.GetSubstructMatches(patt):
        a, b = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
        i = next(n.GetIdx() for n in sorted(a.GetNeighbors(), key=lambda n: -n.GetAtomicNum()) if n.GetIdx() != k)
        l = next(n.GetIdx() for n in sorted(b.GetNeighbors(), key=lambda n: -n.GetAtomicNum()) if n.GetIdx() != j)
        label = "-".join(f"{mol.GetAtomWithIdx(x).GetSymbol()}{x}" for x in (i, j, k, l))
        amide = (a.GetSymbol(), b.GetSymbol()) in (("C", "N"), ("N", "C")) and any(
            n.GetSymbol() == "O" and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
            for c in (a, b) if c.GetSymbol() == "C" for n in c.GetNeighbors())
        torsions.append((label + (" (amide)" if amide else ""), (i, j, k, l)))
    print(f"{ligand}, {leg} leg: {Chem.MolToSmiles(Chem.RemoveHs(mol))}")

    def summary(x):
        """Fractions of the frames near cis (0), +90, trans (180), -90."""
        bins = np.array([np.mean(np.abs(x) < 45), np.mean((x >= 45) & (x < 135)), np.mean(np.abs(x) >= 135),
                         np.mean((x <= -45) & (x > -135))])
        return "cis %.2f  +90 %.2f  trans %.2f  -90 %.2f" % tuple(bins)

    for rep_dir in sorted(lig_dir.glob("repeat_*")):
        d = rep_dir / leg
        if not (d / "hrex" / "samples.h5").exists():
            continue
        eq = np.load(d / "equil.npz")
        atoms = list(eq["frame_atoms"])
        local = np.array([atoms.index(a) for a in lig])
        frames = stored_frames(d / "hrex")[:, local, :]
        start = np.load(d / "start_configs.npz")["positions"][:, lig, :]
        pre = eq["frames"][:, local, :]
        print(f"\n{rep_dir.name}: {len(pre)} frames of pre-equilibration, {len(start)} start configurations, "
              f"{len(frames)} frames of the coupled window")
        for label, (i, j, k, l) in torsions:
            print(f"  {label:26s} pre-equilibration {summary(dihedral(pre, i, j, k, l))} | start "
                  f"{summary(dihedral(start, i, j, k, l))} | coupled window {summary(dihedral(frames, i, j, k, l))}")


if __name__ == "__main__":
    main()
