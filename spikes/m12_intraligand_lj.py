"""How much Lennard-Jones energy inside a ligand lies beyond the cutoff?

cuFEP makes every pair of ligand atoms an exception of the NonbondedForce, so that the
interactions inside the ligand stay when the ligand is decoupled. An exception has no cutoff.
In the original System the pairs that are four or more bonds apart are ordinary pairs, cut
off at ``dynamics.cutoff_nm``; OpenFE (openmmtools) keeps that cutoff for them. The coupled
end of a cuFEP leg therefore has, in addition to the original System, the Lennard-Jones
energy of those pairs at distances beyond the cutoff.

This script gives its size for the input pose of every ligand of the benchmarks: the number
of such pairs and their energy with the OpenFF parameters. No simulation, no GPU.

    spikes/run.sh spikes/m12_intraligand_lj.py [--cutoff 0.9] [--out spikes/out/m12_intraligand_lj.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KCAL = 4.184


def beyond_cutoff(ligand, forcefield, cutoff_nm: float) -> dict:
    from rdkit import Chem

    offmol = ligand.offmol
    labels = forcefield.label_molecules(offmol.to_topology())[0]["vdW"]
    n = offmol.n_atoms
    sigma, epsilon = np.zeros(n), np.zeros(n)
    for (i,), p in labels.items():
        sigma[i] = (
            p.sigma.m_as("nanometer") if p.sigma is not None else p.rmin_half.m_as("nanometer") * 2 / 2 ** (1 / 6)
        )
        epsilon[i] = p.epsilon.m_as("kilojoule / mole")
    pos = ligand.positions_nm()
    bonds_apart = Chem.GetDistanceMatrix(offmol.to_rdkit())
    r = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    i, j = np.triu_indices(n, k=1)
    ordinary = bonds_apart[i, j] >= 4                 # 1-2 and 1-3 are excluded, 1-4 are exceptions already
    far = ordinary & (r[i, j] > cutoff_nm)
    s = 0.5 * (sigma[i[far]] + sigma[j[far]])
    e = np.sqrt(epsilon[i[far]] * epsilon[j[far]])
    x = (s / r[i[far], j[far]]) ** 6
    energy = float(np.sum(4.0 * e * (x * x - x)))
    heavy = [a for a in range(n) if offmol.atoms[a].atomic_number > 1]
    return {"ligand": ligand.name, "atoms": n, "heavy_atoms": len(heavy), "longest_distance_nm": float(r.max()),
            "ordinary_pairs": int(ordinary.sum()), "pairs_beyond_cutoff": int(far.sum()),
            "lj_beyond_cutoff_kJ_mol": energy, "lj_beyond_cutoff_kcal_mol": energy / KCAL}


def main() -> None:
    from openff.toolkit import ForceField

    from cufep.benchmarks.runner import data_root, load_registry
    from cufep.config.schema import Config
    from cufep.prep.ligand import load_ligands

    args = sys.argv[1:]
    cutoff = float(args[args.index("--cutoff") + 1]) if "--cutoff" in args else Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}).dynamics.cutoff_nm
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path("spikes/out/m12_intraligand_lj.json")
    forcefield = ForceField("openff-2.2.1.offxml")
    rows = []
    for name, entry in load_registry().items():
        import csv
        with open(data_root() / entry["reference"]) as fh:
            wanted = {r["compound"] for r in csv.DictReader(fh)}
        for ligand in load_ligands(data_root() / entry["ligands"]):
            if ligand.name in wanted:
                rows.append(dict(beyond_cutoff(ligand, forcefield, cutoff), benchmark=name))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"Lennard-Jones energy inside the ligand beyond {cutoff:g} nm, input poses, OpenFF 2.2.1")
    print(f"{'benchmark':9s} {'ligand':8s} heavy  longest distance (nm)  pairs beyond  energy (kJ/mol)  (kcal/mol)")
    for r in rows:
        print(
            f"{r['benchmark']:9s} {r['ligand']:8s} {r['heavy_atoms']:4d}   {r['longest_distance_nm']:8.2f}             "
            f"{r['pairs_beyond_cutoff']:6d}       {r['lj_beyond_cutoff_kJ_mol']:8.3f}      "
            f"{r['lj_beyond_cutoff_kcal_mol']:7.3f}"
        )
    for name in sorted({r["benchmark"] for r in rows}):
        e = [r["lj_beyond_cutoff_kcal_mol"] for r in rows if r["benchmark"] == name]
        print(f"{name}: {len(e)} ligands, {min(e):.3f} to {max(e):.3f} kcal/mol, mean {np.mean(e):.3f}, "
              f"spread between ligands (max - min) {max(e) - min(e):.3f}")


if __name__ == "__main__":
    main()
