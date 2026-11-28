"""The junctions of a hybrid topology: where atoms that are dummies at an end are bonded to atoms of both ligands,
and which bonded terms between the dummies and the physical atoms are kept at that end.

Why
    At an end of a relative transformation the atoms unique to the other ligand are dummies: their nonbonded
    interactions are off, their bonded terms are not. The free energy is that of the physical ligand only if the
    dummies contribute a constant to the partition function, whatever the configuration of the physical atoms. With
    every bonded term kept, redundant terms between dummies and physical atoms make that contribution depend on
    physical coordinates (two dihedral terms from one dummy to two physical atoms depend on the angle between
    those two), and the end is then not the physical ligand.

The published rule
    Fleck, Wieder and Boresch, J. Chem. Theory Comput. 2021, 17, 4403 ("Dummy Atoms in Alchemical Free Energy
    Calculations"), sections 2.3 and 3.3. The physical bridge atom X is the physical atom bonded to dummies; the
    junction is named by the number of physical groups bound to X.

    * Terminal junction (one physical group R on X): "one can always keep all bond angle terms"; of the dihedrals,
      "one has to identify the physical atoms two bonds away from the physical bridge atom. For one of them, all
      dihedral angles reaching into the dummy group(s) can be kept. Any dihedral angle terms that originate from
      the respective other physical atoms and reach into the dummy group(s) have to be deleted." No term is
      modified.
    * Dual junction (two): the dummy bridge atom by a bond and two angles of 90 degrees and 100 kcal/mol/rad^2 with
      every dihedral from it into the physical part deleted, or by a bond, an angle and one dihedral of
      periodicity 1; terms with dummies of two groups on one bridge deleted.
    * Triple junction (three): exact only where the physical part is planar (all terms to one physical neighbour
      deleted, two angles of 90 degrees); otherwise only weak angles mitigate.

What a junction is in cuFEP's hybrid (cuFEP's reading of the rule for this topology)
    A hydrogen is never mapped onto a heavy atom, and no term of the hybrid joins atoms unique to A with atoms
    unique to B. The bonded terms of a dummy are those of its own ligand, so the physical atoms it is tied to are
    core atoms that are its neighbours' neighbours in its own ligand. The physical groups on the bridge that the
    rule counts are therefore the core atoms bonded to the bridge in the ligand of the dummies (a methyl carbon
    that becomes a nitrile carbon has four physical neighbours at the methyl end, and the dummy nitrogen is tied
    to one of them: a terminal junction).

What is treated (``alchemy.dummy_bonded_terms: separated``)
    Terminal junctions, by the rule above: of the proper torsions Q-R-X-D (R the one core atom on the bridge X, Q a
    core atom on R, D a dummy on X) those through one Q are kept and the others go to zero towards the end at
    which D is a dummy; every angle and every other term is kept. Which Q is cuFEP's choice where the paper leaves
    it free (it kept the one along the main chain): the atom with the most bonds, then the lowest number. Dual and
    triple junctions, and dummies bonded to more than one core atom (a ring that closes at one end), are left as
    they are and named in the record.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Junction:
    """A core atom with dummies on it at one end."""

    #: the end at which the atoms are dummies: "A" (lambda 0, the atoms unique to B) or "B"
    end: str
    bridge: int
    #: the dummies bonded to the bridge
    dummies: tuple[int, ...]
    #: the core atoms bonded to the bridge in the ligand of the dummies
    core_neighbours: tuple[int, ...]
    kind: str
    #: whether the terms of this junction are chosen by the rule (else they are all kept)
    separated: bool
    #: terminal junction: the core atom two bonds from the bridge through which the dihedrals are kept (None: none)
    kept_through: int | None = None
    #: the core atoms two bonds from the bridge whose dihedrals into the dummies go to zero at this end
    removed_through: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {"end": self.end, "bridge": int(self.bridge), "dummies": [int(i) for i in self.dummies],
                "core_neighbours": [int(i) for i in self.core_neighbours], "kind": self.kind,
                "separated": bool(self.separated),
                "kept_through": None if self.kept_through is None else int(self.kept_through),
                "removed_through": [int(i) for i in self.removed_through]}


KINDS = {1: "terminal", 2: "dual", 3: "triple"}


def junctions_of(end: str, bonds, core, dummies) -> list[Junction]:
    """The junctions at one end. ``bonds``: the bonded pairs of the ligand the dummies belong to; ``core``: the atoms
    of both ligands; ``dummies``: the atoms of that ligand that the other does not have (all in one numbering)."""
    core, dummies = set(core), set(dummies)
    neighbours: dict[int, set[int]] = {}
    for i, j in bonds:
        neighbours.setdefault(int(i), set()).add(int(j))
        neighbours.setdefault(int(j), set()).add(int(i))

    def bridges_of_group(start: int) -> set[int]:
        """The core atoms the group of dummies of ``start`` is bonded to."""
        group, stack = {start}, [start]
        while stack:
            for j in neighbours.get(stack.pop(), ()):
                if j in dummies and j not in group:
                    group.add(j)
                    stack.append(j)
        return {j for i in group for j in neighbours.get(i, ()) if j in core}

    found = []
    for bridge in sorted(i for i in core if neighbours.get(i, set()) & dummies):
        on_bridge = tuple(sorted(neighbours[bridge] & dummies))
        physical = tuple(sorted(neighbours[bridge] & core))
        kind = KINDS.get(len(physical), f"{len(physical)} core neighbours")
        one_bridge = all(bridges_of_group(d) == {bridge} for d in on_bridge)
        if not one_bridge:
            kind += ", dummies bonded to more than one core atom"
        if kind != "terminal":
            found.append(Junction(end, bridge, on_bridge, physical, kind, separated=False))
            continue
        beyond = sorted(neighbours[physical[0]] & core - {bridge}, key=lambda q: (-len(neighbours[q]), q))
        found.append(Junction(end, bridge, on_bridge, physical, kind, separated=True,
                              kept_through=beyond[0] if beyond else None, removed_through=tuple(sorted(beyond[1:]))))
    return found


def removed_torsions(junctions: list[Junction], torsions) -> list[int]:
    """Which of ``torsions`` (their atoms, four each) go to zero at the end of ``junctions``: the proper torsions
    Q-R-X-D of a terminal junction with Q one of ``removed_through``. Returns their positions in ``torsions``."""
    terminal = [j for j in junctions if j.separated and j.kind == "terminal"]
    patterns = {(q, j.core_neighbours[0], j.bridge, d) for j in terminal for q in j.removed_through for d in j.dummies}
    return [k for k, atoms in enumerate(torsions)
            if tuple(int(i) for i in atoms) in patterns or tuple(int(i) for i in reversed(atoms)) in patterns]
