"""How much the AM1-BCC charges of a ligand depend on the conformer they are computed on (plan
docs/plans/2026-11-22-accuracy-causes-and-fixes.md, step 2h). cuFEP computes them once, on the input
pose (cufep.prep.ligand.assign_partial_charges). Here the same call is made on the input pose and
on other conformers of the same molecule, and the charges are compared. Numbers only.

    scripts/in_docker.sh python spikes/m23_charge_conformers.py tyk2 ejm_49 ejm_54 ejm_31
    scripts/in_docker.sh python spikes/m23_charge_conformers.py a2a 4k 4e [--conformers 6]

Per ligand: the largest and the root-mean-square difference of an atomic charge between the input
pose and each other conformer, the atom of the largest difference, and the change of the dipole
moment of the conformer when its charges are replaced by those of the input pose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes/out/m23"
SETS = {"tyk2": "benchmarks/tyk2/ligands.sdf", "a2a": "benchmarks/a2a/ligands.sdf"}
DEBYE_PER_E_NM = 48.0321          # 1 e nm in debye


def charges_on(ligand, conformer) -> np.ndarray:
    """AM1-BCC charges of the ligand computed on one conformer, by the function of the package
    that computes them on the input pose (no cache)."""
    from cufep.prep.ligand import Ligand, assign_partial_charges

    work = type(ligand.offmol)(ligand.offmol)
    work._conformers = [conformer]
    copy = Ligand(name=ligand.name, offmol=work)
    assign_partial_charges(copy, "am1bcc")
    return np.asarray(copy.offmol.partial_charges.m_as("elementary_charge"), dtype=float)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in SETS:
        raise SystemExit(__doc__)
    name, wanted, n_conformers = args[0], [], 6
    rest = args[1:]
    while rest:
        a = rest.pop(0)
        if a == "--conformers":
            n_conformers = int(rest.pop(0))
        else:
            wanted.append(a)
    from openff.units import unit

    from cufep.prep.ligand import load_ligands

    ligands = {lig.name: lig for lig in load_ligands(ROOT / SETS[name])}
    result = {}
    print(f"{name}: AM1-BCC charges on the input pose and on up to {n_conformers} other conformers "
          "(e; dipole in debye)")
    print("ligand   conformer  heavy-atom RMSD from the pose (A) | largest |dq|  atom | RMS dq | dipole with own "
          "charges, with those of the pose")
    for lig_name in wanted or list(ligands):
        mol = ligands[lig_name].offmol
        pose = mol.conformers[0]
        q0 = charges_on(ligands[lig_name], pose)
        other = type(mol)(mol)
        other.generate_conformers(n_conformers=n_conformers, rms_cutoff=1.0 * unit.angstrom)
        heavy = [i for i, a in enumerate(mol.atoms) if a.atomic_number > 1]
        rows = []
        for k, conformer in enumerate(other.conformers):
            q = charges_on(ligands[lig_name], conformer)
            x = np.asarray(conformer.m_as("nanometer"), dtype=float)
            x0 = np.asarray(pose.m_as("nanometer"), dtype=float)
            # RMSD of the heavy atoms after superposition (Kabsch)
            a, b = x[heavy] - x[heavy].mean(axis=0), x0[heavy] - x0[heavy].mean(axis=0)
            u, _s, vt = np.linalg.svd(a.T @ b)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
            rmsd = 10.0 * float(np.sqrt(((a @ rot.T - b) ** 2).sum(axis=1).mean()))
            dq = q - q0
            worst = int(np.argmax(np.abs(dq)))
            centred = x - x.mean(axis=0)
            dipole_own = float(np.linalg.norm(q @ centred)) * DEBYE_PER_E_NM
            dipole_pose = float(np.linalg.norm(q0 @ centred)) * DEBYE_PER_E_NM
            atom = mol.atoms[worst]
            rows.append({"conformer": k, "rmsd_A": rmsd, "max_abs_dq": float(np.abs(dq).max()),
                         "atom": f"{atom.symbol}{worst}", "rms_dq": float(np.sqrt((dq ** 2).mean())),
                         "dipole_own_D": dipole_own, "dipole_pose_charges_D": dipole_pose})
            print(f"{lig_name:8s} {k:9d}  {rmsd:8.2f} | {rows[-1]['max_abs_dq']:12.4f}  {rows[-1]['atom']:>4s} | "
                  f"{rows[-1]['rms_dq']:6.4f} | {dipole_own:6.2f}  {dipole_pose:6.2f}", flush=True)
        result[lig_name] = {"n_atoms": mol.n_atoms, "conformers": rows,
                            "largest_abs_dq": max((r["max_abs_dq"] for r in rows), default=None)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
