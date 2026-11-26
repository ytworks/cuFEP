"""The REST2 region of the perturbed atoms of a hybrid ligand (``sampling.rest2.region: perturbed``).

The rule is the default hot atom selection of the relative calculations of the public benchmark, as the help of
its program states it ("Hot Atoms Dialog Box", read in full):

    "The perturbed atoms are grouped into sets in which the atoms are contiguously connected. These can be
    considered as "fragments". The hot atom set is at this stage considered to be empty.
    Each fragment is considered in turn, from the one with the fewest atoms to the one with the most. The bonds from
    the rest of the ligand to the fragment are examined:
      If a bond is rotatable, the atom that is not in the fragment is added to it, with its immediate terminal
      attachments (e.g. -H, -F, =O). Atoms from the fragment that are not already in the hot atom set are added to it.
      If the bond attaches the set to a ring, the ring is added to the fragment and a search is made for rotatable
      bonds attached to the ring (which may be a fused ring system). The atoms in the enlarged fragment are added
      to the hot atom set if they are not already in it.
      If the bond is not rotatable, the atoms are added to the fragment but not to the hot atom set. This would
      happen for a hydrogen or a methyl group, for example.
    The process is repeated (in the same order of fragments as before, not by the new size) to find rotatable
    bonds for the enlarged fragments.
    The process stops when the next addition would exceed the maximum number of heavy atoms, which is 24 if the hot
    region does not contain a ring, or 30 if the hot region contains a ring."

Where the help is silent (cuFEP's own):
- the perturbed atoms are the atoms unique to A or to B and the mapped atoms whose element changes;
- a bond is rotatable as RDKit's ``Lipinski.RotatableBondSmarts`` counts it on the heavy atoms: a single bond
  outside a ring between two heavy atoms that each have another heavy neighbour and no triple bond (so that a bond
  to a methyl group or a halogen is not rotatable, as the help says of a methyl group); a bond of the hybrid is
  judged in the end state that has it (A for its atoms, B for a bond to an atom unique to B);
- "immediate terminal attachments" are the neighbours whose only neighbour is that atom;
- "the ring is added" adds the fused ring system of the atom with the terminal attachments of its atoms;
- the three cases are taken in the order of the help: a rotatable bond, else a bond to a ring, else a bond that is
  not rotatable;
- a pass examines the bonds of each fragment in the order of their atoms; passes are repeated until one adds
  nothing; "the next addition" that would pass the limit ends the selection (it is not made, the selection keeps
  what it had);
- the hydrogens of every heavy atom of the region are in it (the region is used atom by atom).
The protein has no hot atoms in this region (the help's "Receptor hot atom ASL" is empty by default).
"""
from __future__ import annotations

#: the limits of the help: heavy atoms of the hot region without and with a ring
MAX_HEAVY_ATOMS = 24
MAX_HEAVY_ATOMS_WITH_RING = 30


def _end_graph(molecule, to_hybrid: dict[int, int]) -> dict:
    """What the rule reads of one end state, in hybrid indices: neighbours, elements, rotatable bonds, the fused
    ring system of every ring atom."""
    from rdkit import Chem

    neighbours: dict[int, set[int]] = {to_hybrid[a.GetIdx()]: set() for a in molecule.GetAtoms()}
    heavy = {to_hybrid[a.GetIdx()]: a.GetAtomicNum() > 1 for a in molecule.GetAtoms()}
    element = {to_hybrid[a.GetIdx()]: a.GetAtomicNum() for a in molecule.GetAtoms()}
    triple = {to_hybrid[a.GetIdx()] for a in molecule.GetAtoms()
              if any(b.GetBondType() == Chem.BondType.TRIPLE for b in a.GetBonds())}
    for b in molecule.GetBonds():
        i, j = to_hybrid[b.GetBeginAtomIdx()], to_hybrid[b.GetEndAtomIdx()]
        neighbours[i].add(j)
        neighbours[j].add(i)

    def heavy_degree(i: int) -> int:
        return sum(1 for j in neighbours[i] if heavy[j])

    rotatable = set()
    for b in molecule.GetBonds():
        i, j = to_hybrid[b.GetBeginAtomIdx()], to_hybrid[b.GetEndAtomIdx()]
        if (b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing() and heavy[i] and heavy[j]
                and heavy_degree(i) >= 2 and heavy_degree(j) >= 2 and i not in triple and j not in triple):
            rotatable.add(frozenset((i, j)))
    rings = [set(to_hybrid[k] for k in ring) for ring in molecule.GetRingInfo().AtomRings()]
    systems: list[set[int]] = []
    for ring in rings:
        joined = [s for s in systems if s & ring]
        merged = set(ring).union(*joined) if joined else set(ring)
        systems = [s for s in systems if not s & ring] + [merged]
    system_of = {i: s for s in systems for i in s}
    return {"neighbours": neighbours, "heavy": heavy, "element": element, "rotatable": rotatable,
            "ring_system": system_of, "rings": rings}


def perturbed_hot_atoms(molecule_a, molecule_b, to_hybrid_a: dict[int, int], to_hybrid_b: dict[int, int],
                        core: set[int]) -> set[int]:
    """The hot atoms (hybrid indices) of the rule of the module for the hybrid of ``molecule_a`` and ``molecule_b``
    (RDKit molecules with their hydrogens; ``to_hybrid_*`` their atom indices -> hybrid indices; ``core`` the mapped
    hybrid atoms)."""
    a, b = _end_graph(molecule_a, to_hybrid_a), _end_graph(molecule_b, to_hybrid_b)
    in_a, in_b = set(a["neighbours"]), set(b["neighbours"])
    atoms = in_a | in_b

    def end_of(i: int, j: int) -> dict:
        """The end state that has the bond i-j (A where both are atoms of A)."""
        return a if i in in_a and j in in_a else b

    neighbours = {i: a["neighbours"].get(i, set()) | b["neighbours"].get(i, set()) for i in atoms}
    heavy = {**b["heavy"], **a["heavy"]}
    changed = {i for i in core if a["element"][i] != b["element"][i]}
    perturbed = (atoms - core) | changed

    def terminal(v: int, end: dict) -> set[int]:
        return {w for w in end["neighbours"][v] if len(end["neighbours"][w]) == 1}

    # the fragments: connected sets of perturbed atoms, fewest atoms first (ties by their lowest atom)
    fragments, seen = [], set()
    for start in sorted(perturbed):
        if start in seen:
            continue
        group, todo = set(), [start]
        while todo:
            i = todo.pop()
            if i in group:
                continue
            group.add(i)
            todo += [j for j in neighbours[i] if j in perturbed and j not in group]
        seen |= group
        fragments.append(group)
    fragments.sort(key=lambda f: (len(f), min(f)))

    hot: set[int] = set()

    def fits(candidate: set[int]) -> bool:
        heavy_atoms = sum(1 for i in candidate if heavy[i])
        # "contains a ring": a whole ring of either end is in the region (one atom of a ring is not; second review of
        # PR #19)
        ring = any(r <= candidate for r in a["rings"] + b["rings"])
        return heavy_atoms <= (MAX_HEAVY_ATOMS_WITH_RING if ring else MAX_HEAVY_ATOMS)

    changed_any, stopped = True, False
    while changed_any and not stopped:
        changed_any = False
        for fragment in fragments:
            for u in sorted(fragment):
                for v in sorted(neighbours[u] - fragment):
                    if v in fragment:
                        continue                                     # added by an earlier bond of this pass
                    end = end_of(u, v)
                    if frozenset((u, v)) in end["rotatable"]:
                        grown = fragment | {v} | terminal(v, end)
                        candidate = hot | grown
                    elif v in end["ring_system"]:
                        ring = end["ring_system"][v]
                        grown = fragment | ring | {w for r in ring for w in terminal(r, end)}
                        candidate = hot | grown
                    else:
                        fragment.add(v)                              # not rotatable: the fragment only
                        changed_any = True
                        continue
                    if not fits(candidate):
                        stopped = True
                        break
                    fragment |= grown
                    hot = candidate
                    changed_any = True
                if stopped:
                    break
            if stopped:
                break
    # the hydrogens of the heavy atoms of the region
    hot |= {j for i in list(hot) if heavy[i] for j in neighbours[i] if not heavy[j]}
    return hot
