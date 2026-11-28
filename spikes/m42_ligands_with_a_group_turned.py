"""The two ligands of a prepared edge with the terminal group that terminal flip moves turn given in its other
orientation: an SDF of the two records whose atoms beyond the pivot are turned by half a turn about the bond of
the group (the coordinates of the input, nothing else changed). The input of a calculation that starts from the
other orientation, for the comparison of the two starts with and without the moves (plan of 2026-11-28, issue #27).

The group is that of the hybrid of the edge (:func:`cufep.rbfe.leg.ligand_flip_groups`, complex leg); in each ligand
its own atoms beyond the pivot are turned about its own anchor and pivot.

With OUT_CACHE the charges the run of RUN_DIR used for the two ligands (its ``cache/``) are written there for the
turned records, so that a calculation from OUT.sdf has the charges of the one it is compared with: the cache is
keyed by chemistry and coordinates, and AM1-BCC charges of another conformer are other charges.

    scripts/in_docker.sh python spikes/m42_ligands_with_a_group_turned.py LIGANDS.sdf RUN_DIR EDGE OUT.sdf [OUT_CACHE]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(ligands_sdf: str, run_dir: str, edge: str, out: str, out_cache: str | None = None) -> None:
    from rdkit import Chem

    from cufep.rbfe.leg import edge_ligands, ligand_flip_groups
    from cufep.sampling.terminal_flip import FlipGroup, _beyond, half_turn

    edge_dir = Path(run_dir) / "edges" / edge
    name_a, name_b = edge.split("~")
    mol_a, mol_b, mapping = edge_ligands(edge_dir, "complex")
    groups, _record = ligand_flip_groups(edge_dir, "complex")
    if len(groups) != 1:
        raise SystemExit(f"{edge}: {len(groups)} groups are turned; this script is for an edge with one")
    group = groups[0]
    axes = {name_a: (group.anchor, group.pivot), name_b: (mapping[group.anchor], mapping[group.pivot])}
    records = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(ligands_sdf, removeHs=False)}
    writer = Chem.SDWriter(out)
    box = 1000.0 * np.eye(3)                              # no periodic images in an input file
    for name, prepared in ((name_a, mol_a), (name_b, mol_b)):
        mol = Chem.Mol(records[name])
        if [a.GetAtomicNum() for a in mol.GetAtoms()] != [a.GetAtomicNum() for a in prepared.GetAtoms()]:
            raise SystemExit(f"{name}: the atoms of the record are not those of the prepared ligand")
        anchor, pivot = axes[name]
        neighbours = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in mol.GetAtoms()}
        mobile = tuple(sorted(_beyond(neighbours, pivot, anchor)))
        xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        turned, _rotation = half_turn(xyz, box, FlipGroup(anchor=anchor, pivot=pivot, mobile=mobile))
        for i, row in enumerate(turned):
            mol.GetConformer().SetAtomPosition(i, row.tolist())
        moved = np.linalg.norm(turned - xyz, axis=1)
        far = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in mobile
               if moved[i] > 0.5 and mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        print(f"{name}: anchor {anchor} ({mol.GetAtomWithIdx(anchor).GetSymbol()}), pivot {pivot}, {len(mobile)} atoms "
              f"turned, displaced by up to {moved.max():.2f} A; heavy atoms that moved more than 0.5 A: {far}")
        writer.write(mol)
    writer.close()
    if out_cache:
        import shutil

        from cufep.prep.ligand import load_ligands

        before = {lig.name: lig.digest() for lig in load_ligands(ligands_sdf)}
        Path(out_cache).mkdir(parents=True, exist_ok=True)
        for lig in load_ligands(out):
            source = Path(run_dir) / "cache" / f"charges_am1bcc_{before[lig.name]}.json"
            shutil.copyfile(source, Path(out_cache) / f"charges_am1bcc_{lig.digest()}.json")
            print(f"{lig.name}: the charges of {source.name} for the turned record ({lig.digest()})")


if __name__ == "__main__":
    main(*sys.argv[1:])
