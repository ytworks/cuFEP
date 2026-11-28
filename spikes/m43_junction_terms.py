"""The junctions of an edge in detail: for each end, the groups of dummy atoms, the core atom each is bonded to (the
bridge) with its neighbours in the physical ligand of that end, and the bonded terms of the ligand the dummies
belong to that join dummy atoms to physical atoms (the terms the hybrid keeps at full strength at that end): the
angles and the torsions, each with its atoms. Reads the prepared solvent leg of an edge; nothing is run or written.

For the rule of item 3 of the plan of 2026-11-28 (issue #27): which of these terms Fleck, Wieder and Boresch (J.
Chem. Theory Comput. 2021, 17, 4403) keep at a terminal, a dual and a planar triple junction.

    scripts/in_docker.sh python spikes/m43_junction_terms.py runs/rbfe_bench_tyk2_957a9cf "jmc_28~jmc_30" [...]
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(run_dir: str, *edges: str) -> None:
    import openmm as mm

    from cufep.rbfe.leg import edge_ligands
    from cufep.rbfe.prep import load_end_states

    for edge in edges:
        edge_dir = Path(run_dir) / "edges" / edge
        mol_a, mol_b, mapping = edge_ligands(edge_dir, "solvent")
        end = load_end_states(edge_dir, "solvent")
        print(f"==== {edge}: {len(mapping)} core atoms")
        b_to_a = {j: i for i, j in mapping.items()}
        for end_name, phys, dummy_mol, dummy_end, to_phys in (
                ("A", mol_a, mol_b, end["b"], b_to_a), ("B", mol_b, mol_a, end["a"], mapping)):
            # the dummies at this end are the unique atoms of the other ligand; their terms are those of its System
            system = mm.XmlSerializer.deserialize(dummy_end["system_xml"])
            index = {int(s): k for k, s in enumerate(dummy_end["ligand"])}          # System atom -> atom of the ligand
            core = set(to_phys)

            def label(k: int, dummy_mol=dummy_mol, core=core, to_phys=to_phys, phys=phys) -> str:
                atom = dummy_mol.GetAtomWithIdx(k)
                if k in core:
                    p = phys.GetAtomWithIdx(to_phys[k])
                    change = "" if p.GetSymbol() == atom.GetSymbol() else f">{atom.GetSymbol()}"
                    return f"{p.GetSymbol()}{to_phys[k]}{change}"
                return f"[{atom.GetSymbol()}{k}]"                                      # a dummy at this end

            dummies = [a.GetIdx() for a in dummy_mol.GetAtoms() if a.GetIdx() not in core]
            print(f"-- end {end_name}: {len(dummies)} dummy atoms {[label(k) for k in dummies]}")
            for k in sorted({n.GetIdx() for d in dummies for n in dummy_mol.GetAtomWithIdx(d).GetNeighbors()} & core):
                p, q = phys.GetAtomWithIdx(to_phys[k]), dummy_mol.GetAtomWithIdx(k)
                print(f"   bridge {label(k)}: {p.GetHybridization()} with {p.GetDegree()} neighbours "
                      f"{[f'{n.GetSymbol()}{n.GetIdx()}' for n in p.GetNeighbors()]} at this end"
                      f"{', in a ring' if p.IsInRing() else ''}{', aromatic' if p.GetIsAromatic() else ''}; "
                      f"{q.GetHybridization()} with {q.GetDegree()} in the other ligand")
            for force in system.getForces():
                if isinstance(force, mm.HarmonicAngleForce):
                    rows = []
                    for t in range(force.getNumAngles()):
                        i, j, m, theta, k_ = force.getAngleParameters(t)
                        if all(x in index for x in (i, j, m)):
                            atoms = [index[x] for x in (i, j, m)]
                            n_dummy = sum(a not in core for a in atoms)
                            if 0 < n_dummy < 3:
                                rows.append(f"{'-'.join(label(a) for a in atoms)} ({theta._value * 57.29578:.0f} deg, "
                                            f"{k_._value:.0f})")
                    print(f"   angles with dummy and physical atoms ({len(rows)}): " + "; ".join(rows))
                if isinstance(force, mm.PeriodicTorsionForce):
                    rows: dict[str, list[str]] = {}
                    for t in range(force.getNumTorsions()):
                        i, j, m, n, per, _phase, k_ = force.getTorsionParameters(t)
                        if all(x in index for x in (i, j, m, n)) and k_._value != 0.0:
                            atoms = [index[x] for x in (i, j, m, n)]
                            n_dummy = sum(a not in core for a in atoms)
                            if 0 < n_dummy < 4:
                                rows.setdefault("-".join(label(a) for a in atoms), []).append(f"n{per}:{k_._value:.2f}")
                    print(f"   torsions with dummy and physical atoms ({len(rows)}):")
                    for atoms, terms in rows.items():
                        print(f"      {atoms}  {' '.join(terms)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
