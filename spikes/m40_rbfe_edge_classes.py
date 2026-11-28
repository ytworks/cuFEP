"""What the changes of the plan of 2026-11-28 (issue #27) apply to, edge by edge, for the prepared edges of a run
directory of ``cufep rbfe`` or of a relative benchmark. No experimental value is read, nothing is run, and the
directory is only read.

Per edge, from ``prep/`` (the two ligands and the atom mapping of the complex leg):

* terminal flip Monte Carlo: the terminal aromatic groups the moves would turn, and those of either ligand that
  are left out with the reason (:func:`cufep.sampling.terminal_flip.hybrid_flip_groups`, on the bond graph of the
  two ligands: the unique atoms of B are numbered after the atoms of A, hydrogens are held by constraints);
* junctions: for each end, the groups of dummy atoms, the core atom each is bonded to (the bridge), how many
  atoms of the physical ligand of that end the bridge is bonded to (1 terminal, 2 dual, 3 triple junction, in the
  words of Fleck, Wieder and Boresch 2021), whether the hybridisation of the bridge differs between the ligands,
  and whether a group is bonded to more than one core atom (a ring that closes at one end);
* REST2: the atoms unique to a ligand, and the core atoms whose element, Lennard-Jones parameters or bonded
  neighbours change; of the others, how many change their charge by more than 0.01 and 0.05 e.

    scripts/in_docker.sh python spikes/m40_rbfe_edge_classes.py runs/rbfe_bench_tyk2_957a9cf [OUT.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def ligand(path: Path):
    from openff.toolkit import Molecule

    off = Molecule.from_json(path.read_text())
    return off, off.to_rdkit()


def junctions(mol_phys, mol_dummy, phys_of_dummy: dict[int, int]) -> list[dict]:
    """The groups of atoms of ``mol_dummy`` that are dummies at the end of ``mol_phys`` (``phys_of_dummy``: core
    atom of mol_dummy -> atom of mol_phys)."""
    core = set(phys_of_dummy)
    dummies = [a.GetIdx() for a in mol_dummy.GetAtoms() if a.GetIdx() not in core]
    seen, out = set(), []
    for start in dummies:
        if start in seen:
            continue
        group, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for n in mol_dummy.GetAtomWithIdx(i).GetNeighbors():
                j = n.GetIdx()
                if j not in core and j not in group:
                    group.add(j)
                    stack.append(j)
        seen |= group
        bridges = sorted({n.GetIdx() for i in group for n in mol_dummy.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() in core})
        entry = {"dummy_atoms": len(group), "heavy_dummy_atoms": sum(mol_dummy.GetAtomWithIdx(i).GetAtomicNum() > 1
                                                                     for i in group),
                 "bridges": len(bridges)}
        if len(bridges) == 1:
            p_dummy = mol_dummy.GetAtomWithIdx(bridges[0])
            p_phys = mol_phys.GetAtomWithIdx(phys_of_dummy[bridges[0]])
            # the terms of the dummies are those of their own ligand, in which the bridge is bonded to these core
            # atoms: no term of the hybrid joins unique atoms of both ligands. Their number is what the junction is in
            # the words of Fleck et al. (1 terminal, 2 dual, 3 triple; a triple junction can be separated exactly
            # only where the bridge is planar in the ligand of the dummies)
            core_neighbours = sum(n.GetIdx() in core for n in p_dummy.GetNeighbors())
            planar = str(p_dummy.GetHybridization()) in ("SP", "SP2") or p_dummy.GetIsAromatic()
            kind = {1: "terminal", 2: "dual", 3: "triple, planar" if planar else "triple, not planar"}.get(
                core_neighbours, f"{core_neighbours} core neighbours")
            entry.update(physical_neighbours=p_phys.GetDegree(), core_neighbours=core_neighbours, junction=kind,
                         bridge_element=p_phys.GetSymbol(),
                         bridge_hybridisation=[str(p_phys.GetHybridization()), str(p_dummy.GetHybridization())],
                         hybridisation_changes=p_phys.GetHybridization() != p_dummy.GetHybridization(),
                         bridge_in_ring=p_phys.IsInRing())
        out.append(entry)
    return out


def changed_core(off_a, off_b, mol_a, mol_b, mapping: dict[int, int]) -> dict:
    """Core atoms by what changes between the ligands (charges of the molecules as prepared, if they have them)."""
    qa = None if off_a.partial_charges is None else [float(q) for q in off_a.partial_charges.m_as("elementary_charge")]
    qb = None if off_b.partial_charges is None else [float(q) for q in off_b.partial_charges.m_as("elementary_charge")]
    element = sum(mol_a.GetAtomWithIdx(i).GetAtomicNum() != mol_b.GetAtomWithIdx(j).GetAtomicNum()
                  for i, j in mapping.items())
    next_to_unique = sum(any(n.GetIdx() not in mapping for n in mol_a.GetAtomWithIdx(i).GetNeighbors())
                         or any(n.GetIdx() not in set(mapping.values()) for n in mol_b.GetAtomWithIdx(j).GetNeighbors())
                         for i, j in mapping.items())
    out = {"core_atoms": len(mapping), "core_element_changes": int(element),
           "core_bonded_to_a_unique_atom": int(next_to_unique)}
    if qa is not None and qb is not None:
        dq = [abs(qa[i] - qb[j]) for i, j in mapping.items()]
        out.update(core_charge_changes_above_0_01=sum(d > 0.01 for d in dq),
                   core_charge_changes_above_0_05=sum(d > 0.05 for d in dq), largest_core_charge_change=max(dq))
    return out


def main(run_dir: str, out: str | None = None) -> None:
    from cufep.sampling.terminal_flip import hybrid_flip_groups, ligand_graph

    rows = []
    edges = Path(run_dir) / "edges"
    for edge_dir in sorted(path for path in edges.iterdir() if (path / "prep" / "prep.json").exists()):
        prep = edge_dir / "prep"
        info = json.loads((prep / "prep.json").read_text())
        (off_a, mol_a), (off_b, mol_b) = ligand(prep / "ligand_a.json"), ligand(prep / "ligand_b.json")
        mapping = {int(i): int(j) for i, j in info["legs"]["complex"]["mapping"].items()}
        to_a, to_b, bonds, held = ligand_graph(mol_a, mol_b, mapping)
        groups, record = hybrid_flip_groups(mol_a, mol_b, mapping, to_a, to_b, bonds, held)
        b_to_a = {j: i for i, j in mapping.items()}
        row = {"edge": edge_dir.name,
               "unique_a": mol_a.GetNumAtoms() - len(mapping), "unique_b": mol_b.GetNumAtoms() - len(mapping),
               "flip_groups": [g.to_dict() for g in groups], "terminal_groups": record,
               # dummies at the end of A are the unique atoms of B, and the other way round
               "junctions_at_a": junctions(mol_a, mol_b, b_to_a), "junctions_at_b": junctions(mol_b, mol_a, mapping),
               **changed_core(off_a, off_b, mol_a, mol_b, mapping)}
        rows.append(row)
        left_out = sorted({r["why_not"] for r in record if not r["used"]})

        def kinds(js):
            return ", ".join(f"{j['dummy_atoms']} atoms/{j.get('physical_neighbours', '?')} neighbours"
                             + ("/hybridisation changes" if j.get("hybridisation_changes") else "")
                             + (f"/{j['bridges']} bridges" if j["bridges"] != 1 else "") for j in js) or "none"

        print(f"{row['edge']}: flips {len(groups)} group(s)"
              + (f" (mobile atoms {[len(g.mobile) for g in groups]})" if groups else "")
              + (f"; left out: {'; '.join(left_out)}" if left_out else "")
              + f" | dummies at A: {kinds(row['junctions_at_a'])} | at B: {kinds(row['junctions_at_b'])}"
              + f" | core {row['core_atoms']}, element changes {row['core_element_changes']}, bonded to a unique atom "
              f"{row['core_bonded_to_a_unique_atom']}, charge changes > 0.01 e "
              f"{row.get('core_charge_changes_above_0_01')}, > 0.05 e {row.get('core_charge_changes_above_0_05')}",
              flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
