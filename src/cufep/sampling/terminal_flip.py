"""Terminal flip Monte Carlo: a terminal aromatic group of the ligand is turned by half a turn about the bond that
joins it to the rest of the ligand, and the move is accepted with the potential energy of the replica's state.

Why
    A ring whose two faces or edges differ (a substituent in one ortho or meta position, a ring atom of another
    element) has two orientations about its exocyclic bond. In a pocket the barrier between them is crossed rarely
    in the nanoseconds of a lambda window, so a calculation samples the orientation it was started in. The move
    proposes the other one directly.

The published method and what is taken from it
    Terminal flip Monte Carlo as SOMD2 implements it (OpenBioSim/somd2, ``runner/_samplers/_terminal_flip.py``,
    read 2026-11-28; the two papers on it, doi:10.1021/acs.jctc.6c00177 and doi:10.1021/acs.jctc.6c00593, could
    not be read here). SOMD2: the anchor is a non-ring atom with at least two connections, the pivot a ring atom
    bonded to it, the mobile atoms everything reached from the pivot without crossing the anchor; the angle is
    360/n degrees from the geometry of the ring (180 for a planar ring); one group is chosen uniformly and turned
    by a uniformly chosen multiple of the angle that is not a full turn; the Metropolis criterion on the total
    potential energy of the Context decides, and the positions are put back on rejection; one attempt at the start
    of each cycle of the replica exchange; the velocities are left or drawn anew.

cuFEP's rule (:func:`terminal_groups`, :func:`hybrid_flip_groups`)
    Narrower than SOMD2's, by the first scope of issue #27; every difference is cuFEP's own.

    * The axis is a bond that is in no ring; the pivot is an atom of an aromatic ring. (SOMD2 takes any ring, and
      only a non-ring anchor; here the anchor may be an atom of another ring, as in a biaryl.)
    * The group is terminal: beyond the axis lies one ring system (the pivot's, with the rings fused to it) and
      atoms in no ring. The side of the anchor must not be such a group itself (which side is the terminal one
      would not be determined). SOMD2 instead skips groups above a number of atoms, if one is given.
    * The two orientations differ: the two ring neighbours of the pivot are not equivalent atoms of the molecule
      (canonical ranks without tie breaking). A phenyl or a 2,6-dichlorophenyl ring is not turned: its half turn
      gives the configuration it had with the names of equivalent atoms exchanged.
    * In a hybrid of two ligands the ring of the pivot and the anchor are core atoms (the ring is in both
      ligands), the group is terminal in both, and the two orientations differ in at least one. The mobile atoms
      are those beyond the pivot in the bond graph of the hybrid: the unique atoms of both ligands turn with the
      ring. A group whose mobile atoms reach the anchor another way (a ring closed in one ligand only) or are
      held by a constraint to an atom that does not turn is left out.
    * The angle is half a turn for every group, fixed when the leg is set up: the exocyclic bond of an aromatic
      atom lies in the plane of its ring, so half a turn about it maps the plane of the ring onto itself. No
      angle is taken from the configuration.

The move (:class:`FlipMover`)
    One group is chosen uniformly. Its mobile atoms are turned by half a turn about the line through anchor and
    pivot (positions relative to the pivot by the nearest image), and their velocities by the same rotation, so
    that constrained bonds inside the group keep their length and the velocities no component along them, and the
    kinetic energy is what it was. The move is its own reverse and is proposed with the same probability from
    both configurations, so it is accepted with min(1, exp(-(U' - U)/kT)), U the potential energy of the Context
    in the replica's state (lambda and REST2 as they are set): detailed balance in that state. A rejected move
    leaves positions and velocities as they were. Two evaluations of the energy per attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cufep.sampling.relocation_mc import KB_KJ_PER_MOL_K

#: what the conditions of the samples say of the rule and of the move (changed with either)
FLIP_RULE = "aromatic terminal groups whose two orientations differ; half a turn, velocities turned with it"


@dataclass(frozen=True)
class FlipGroup:
    """A terminal group: the atoms of the axis (they do not move) and the atoms that turn about it."""

    anchor: int
    pivot: int
    mobile: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"anchor": int(self.anchor), "pivot": int(self.pivot), "mobile": [int(i) for i in self.mobile]}


@dataclass
class FlipSpec:
    """What a mover needs: the groups (atom indices of the System), the temperature and the interval."""

    groups: list[FlipGroup]
    temperature_K: float
    interval_ps: float
    #: per group, where it was found (names of the ligands' atoms), for the log and the result
    names: list[str] = field(default_factory=list)

    def record(self) -> dict:
        """What the conditions of the samples say of the moves."""
        return {"rule": FLIP_RULE, "interval_ps": float(self.interval_ps),
                "groups": [g.to_dict() for g in self.groups]}


# ------------------------------------------------------------------------------------------- the rule
def _beyond(neighbours: dict[int, set[int]], pivot: int, anchor: int) -> set[int] | None:
    """The atoms reached from ``pivot`` without using the bond pivot-anchor, the pivot left out; None if the
    anchor is reached another way (the bond is in a ring)."""
    seen, stack = {pivot}, [pivot]
    while stack:
        i = stack.pop()
        for j in neighbours[i]:
            if i == pivot and j == anchor:
                continue
            if j == anchor:
                return None
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen - {pivot}


def _ring_systems(mol) -> dict[int, int]:
    """Atom -> number of its ring system (rings that share an atom are one system); atoms in no ring are absent."""
    system: dict[int, int] = {}
    groups: list[set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        ring = set(ring)
        touching = [g for g in groups if g & ring]
        for g in touching:
            ring |= g
            groups.remove(g)
        groups.append(ring)
    for k, g in enumerate(groups):
        for i in g:
            system[i] = k
    return system


def terminal_groups(mol) -> list[dict]:
    """The terminal aromatic groups of a molecule (an RDKit molecule with its hydrogens), by the rule of the
    module: for each ``anchor``, ``pivot``, ``mobile`` (sorted atom indices), ``ring`` (the atoms of the pivot's
    ring system) and ``distinct`` (whether the two orientations differ)."""
    from rdkit import Chem

    neighbours = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in mol.GetAtoms()}
    systems = _ring_systems(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    def side(pivot: int, anchor: int) -> set[int] | None:
        """The atoms beyond the axis if they are a terminal group with an aromatic pivot, else None."""
        if not mol.GetAtomWithIdx(pivot).GetIsAromatic() or pivot not in systems:
            return None
        mobile = _beyond(neighbours, pivot, anchor)
        if mobile is None or {systems[i] for i in mobile | {pivot} if i in systems} != {systems[pivot]}:
            return None
        return mobile

    found = []
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for pivot, anchor in ((i, j), (j, i)):
            mobile = side(pivot, anchor)
            if mobile is None:
                continue
            # the terminal side of the bond is the smaller one (heavy atoms); with as many on both sides it is
            # not determined, and neither is turned
            beyond = len((mobile | {pivot}) & heavy)
            if not beyond < len(heavy) - beyond:
                continue
            ring = {k for k, s in systems.items() if s == systems[pivot]}
            beside = sorted(neighbours[pivot] & ring)
            distinct = len(beside) != 2 or ranks[beside[0]] != ranks[beside[1]]
            found.append({"anchor": anchor, "pivot": pivot, "mobile": sorted(mobile), "ring": sorted(ring),
                          "distinct": bool(distinct)})
    return sorted(found, key=lambda g: (g["anchor"], g["pivot"]))


def ligand_graph(mol_a, mol_b, mapping: dict[int, int]) -> tuple[dict[int, int], dict[int, int], set, set]:
    """The two ligands as one graph without a System: the atoms of A as they are numbered, the unique atoms of B
    after them. Returns ``(atom of A -> atom of the graph, atom of B -> atom of the graph, bonded pairs, pairs
    with a hydrogen)``; the last are the pairs a System with constrained bonds to hydrogen holds."""
    n_a = mol_a.GetNumAtoms()
    b_to_a = {j: i for i, j in mapping.items()}
    unique_b = [j for j in range(mol_b.GetNumAtoms()) if j not in b_to_a]
    to_a = {i: i for i in range(n_a)}
    to_b = {**b_to_a, **{j: n_a + k for k, j in enumerate(unique_b)}}
    bonds, with_hydrogen = set(), set()
    for mol, to in ((mol_a, to_a), (mol_b, to_b)):
        for bond in mol.GetBonds():
            i, j = to[bond.GetBeginAtomIdx()], to[bond.GetEndAtomIdx()]
            pair = (min(i, j), max(i, j))
            bonds.add(pair)
            if 1 in (bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()):
                with_hydrogen.add(pair)
    return to_a, to_b, bonds, with_hydrogen


def hybrid_flip_groups(mol_a, mol_b, mapping: dict[int, int], to_hybrid_a: dict[int, int],
                       to_hybrid_b: dict[int, int], bonds, constraints=()) -> tuple[list[FlipGroup], list[dict]]:
    """The groups of a hybrid of ligands A and B that the moves turn, and a record of every terminal group of
    either ligand with what was decided of it (``used``, or ``why_not``).

    ``mol_a``, ``mol_b``: the ligands (RDKit molecules with hydrogens); ``mapping``: atom of A -> atom of B (the
    core); ``to_hybrid_*``: atom of the ligand -> atom of the hybrid System; ``bonds``: the bonded pairs of the
    hybrid System (both ligands); ``constraints``: its constrained pairs.
    """
    core_a, core_b = set(mapping), set(mapping.values())
    b_to_a = {j: i for i, j in mapping.items()}
    of_a = {(g["anchor"], g["pivot"]): g for g in terminal_groups(mol_a)}
    of_b = {(g["anchor"], g["pivot"]): g for g in terminal_groups(mol_b)}
    neighbours: dict[int, set[int]] = {}
    for i, j in bonds:
        neighbours.setdefault(int(i), set()).add(int(j))
        neighbours.setdefault(int(j), set()).add(int(i))
    groups, record, seen = [], [], set()

    def note(ligand: str, g: dict, hybrid_axis, why_not: str | None, mobile=()) -> None:
        record.append({"found_in": ligand, "anchor": g["anchor"], "pivot": g["pivot"],
                       "mobile_atoms_of_the_ligand": len(g["mobile"]), "distinct": g["distinct"],
                       "hybrid_axis": None if hybrid_axis is None else [int(hybrid_axis[0]), int(hybrid_axis[1])],
                       "used": why_not is None, "why_not": why_not, "mobile_atoms": len(mobile)})

    for ligand, mine, other, core, to_hybrid, to_other in (
            ("A", of_a, of_b, core_a, to_hybrid_a, mapping), ("B", of_b, of_a, core_b, to_hybrid_b, b_to_a)):
        for (anchor, pivot), g in mine.items():
            if anchor not in core or not set(g["ring"]) <= core:
                note(ligand, g, None, "the ring or the anchor is in one ligand only")
                continue
            axis = (to_hybrid[anchor], to_hybrid[pivot])
            if axis in seen:
                continue                                              # found in A already
            seen.add(axis)
            twin = other.get((to_other[anchor], to_other[pivot]))
            if twin is None:
                note(ligand, g, axis, "not a terminal group of the other ligand")
                continue
            if not (g["distinct"] or twin["distinct"]):
                note(ligand, g, axis, "the two orientations do not differ in either ligand")
                continue
            mobile = _beyond(neighbours, axis[1], axis[0])
            if mobile is None:
                note(ligand, g, axis, "the atoms beyond the pivot reach the anchor in the hybrid")
                continue
            held = [pair for pair in constraints if (int(pair[0]) in mobile) != (int(pair[1]) in mobile)]
            if held:
                note(ligand, g, axis, "a constraint joins an atom that turns to one that does not", mobile)
                continue
            groups.append(FlipGroup(anchor=int(axis[0]), pivot=int(axis[1]), mobile=tuple(sorted(mobile))))
            note(ligand, g, axis, None, mobile)
    return groups, record


# ------------------------------------------------------------------------------------------- the move
def half_turn(positions: np.ndarray, box: np.ndarray, group: FlipGroup) -> tuple[np.ndarray, np.ndarray]:
    """``positions`` (nm) with the mobile atoms of ``group`` turned by half a turn about the line through its anchor
    and its pivot, and the matrix they were turned with. The atoms are taken relative to the pivot by the nearest
    image, so a ligand that lies across the edge of the cell is turned as one molecule."""
    from cufep.alchemy.lrc import nearest_image

    x = np.array(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    pivot = x[group.pivot]
    axis = nearest_image((pivot - x[group.anchor])[None], box)[0]
    axis = axis / np.linalg.norm(axis)
    rotation = 2.0 * np.outer(axis, axis) - np.eye(3)              # half a turn about ``axis``
    mobile = np.asarray(group.mobile, dtype=np.int64)
    relative = nearest_image(x[mobile] - pivot, box)
    x[mobile] = pivot + relative @ rotation.T
    return x, rotation


class FlipMover:
    """Attempts terminal flips in one OpenMM Context (the module's docstring)."""

    def __init__(self, spec: FlipSpec, context, seed: int) -> None:
        if not spec.groups:
            raise ValueError("no terminal group to turn")
        self.spec = spec
        self.context = context
        self.rng = np.random.default_rng(int(seed))
        self.beta = 1.0 / (KB_KJ_PER_MOL_K * float(spec.temperature_K))

    def _potential(self) -> float:
        import openmm.unit as u

        return float(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))

    def attempt(self, state=None) -> dict:
        """One move on the configuration of the Context, in the state the Context is set to. Returns ``group``
        (its number), ``accepted`` and ``log_r`` (-(U' - U)/kT). The Context is changed only by a move that is
        accepted."""
        import openmm.unit as u

        st = self.context.getState(getPositions=True, getVelocities=True)
        x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        v = np.asarray(st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond), dtype=float)
        box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        k = int(self.rng.integers(len(self.spec.groups)))
        group = self.spec.groups[k]
        y, rotation = half_turn(x, box, group)
        u_x = self._potential()
        self.context.setPositions(y)
        log_r = -self.beta * (self._potential() - u_x)
        result = {"group": k, "accepted": False, "log_r": float(log_r)}
        if not np.isfinite(log_r) or np.log(self.rng.random()) >= log_r:
            self.context.setPositions(x)
            return result
        mobile = np.asarray(group.mobile, dtype=np.int64)
        v[mobile] = v[mobile] @ rotation.T
        self.context.setVelocities(v)
        result["accepted"] = True
        return result
