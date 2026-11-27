"""The bonded terms of dummy atoms where one mapped atom carries unmapped atoms of both ligands.

Source: M. Fleck, M. Wieder and S. Boresch, "Dummy atoms in alchemical free energy calculations", J. Chem. Theory
Comput. 17, 4403-4419 (2021), cited as Fleck 2021 with its sections.

Words (Fleck 2021, sec. 2.3). At the end where a ligand is absent (A at lambda 1, B at lambda 0) its unmapped atoms
are dummy atoms. ``X`` is a mapped atom bonded to them (the physical bridge atom); a dummy group is a connected part of
the unmapped atoms of that ligand bonded to X through one atom ``D`` (the dummy bridge atom); the anchors ``P`` are the
mapped neighbours of X. The unmapped atoms of the other ligand at X have no term with the dummy atoms (a unique old and
a unique new atom never interact, :mod:`cufep.rbfe.hybrid`), so that the dummy group is anchored to the anchors alone:
"since, in effect, we have eliminated the acetone branch from consideration, this reduces our task to that of a triple
junction" (sec. 2.4). The number of anchors is the type of the junction: one, terminal (sec. 2.3.1); two, dual
(sec. 2.3.2); three, triple, nonplanar since X then has four bonds in each ligand (sec. 2.3.3).

Why. OpenFE's hybrid keeps every bonded term with an unmapped atom at full strength at every lambda. Where both
ligands have unmapped atoms on X, the dummy bridge atom of one ligand then keeps its angles to the anchors and its
dihedrals across the bonds of X: a rotation of the other ligand about such a bond drags the dummy with it against its
dihedrals (ejm_42~ejm_54 of TYK2: the urea C-N bond of ejm_54 at lambda 1 had a torsional barrier of 88.4 kJ/mol
instead of its own 53.6, and its solvent leg never reached its conformer with an intramolecular hydrogen bond).
Fleck 2021: "The derivation of eq 2 relies crucially on using exactly three, nonredundant degrees of freedom (three
bonded energy terms) per dummy atom" (sec. 2.1.1); with them the dummy group adds the same factor to both legs and
cannot change the physical end state.

The rules (at the dummy end; terms with no mapped atom are kept: "Any energy terms involving only dummy atoms from the
same dummy group can and should always be kept", sec. 2.3; a term that touches unmapped atoms of the ligand outside
the groups of X, or groups of two such atoms X, is deleted):

* terminal: "one can always keep all bond angle terms"; of the dihedrals Z-P-X-D from the atoms Z two bonds away from X,
  "For one of them, all dihedral angles reaching into the dummy group(s) can be kept. Any dihedral angle terms that
  originate from the respective other physical atoms and reach into the dummy group(s) have to be deleted" (sec.
  2.3.1);
* dual: "There are two possibilities to anchor the dummy bridge atom DC with respect to the physical molecule. The
  first option is to use a consecutive bond, angle, and dihedral angle term ... To prevent flapping, the periodicity
  of the dihedral angle term has to be removed" (sec. 2.3.2): the bond X-D and the angle P-X-D are kept, the
  dihedral D-X-P-Z gets periodicity 1 and 100 kcal/mol with its minimum where the dummy is ("The DC–N-C5 angle was
  deleted, and the DC–N-C1-C2 dihedral kept instead ... The force constant of the dihedral angle was increased to
  100 kcal/mol, its periodicity changed to 1", TOL2PYR-2), the other angle and every other dihedral from D into the
  physical molecule are deleted. Where no anchor has a mapped neighbour other than X, the second possibility, "set
  the equilibrium values of the bond angle terms to the dummy atom to 90°, together with increasing the corresponding
  force constants" to 100 kcal/(mol rad^2) (Table 3), with "all dihedral angle terms starting from DC into the
  physical system" deleted. Of the dihedrals D'-D-X-P of the deeper dummy atoms, "all dihedral angle terms must
  either terminate at atom C1 or C5; the respective other dihedrals need to be deleted" (those to the anchor of the
  dihedral are kept); with two dummy groups at X, "any bonded terms which depend on dummy atoms from both groups
  should be deleted";
* triple (nonplanar): the three angles P-X-D are kept with "force constants ... set to a low value (<5 kcal/(mol
  rad2))", 3.55 kcal/(mol rad^2) in the calculations of Fleck 2021 (sec. 3, Table 2), and "their equilibrium values
  should be adapted so that the minimum energy of the physical molecule is not affected"; "all dihedrals starting from
  the dummy bridge atom into the physical molecule need to be removed"; dihedrals D'-D-X-P to one anchor only (as for
  the dual junction, sec. 2.3.3).

Force constants: Fleck 2021 computed with CHARMM, whose angle energy is K (theta - theta0)^2; OpenMM's is
(k/2) (theta - theta0)^2, so k = 2 K.

cuFEP's own choices where the source is silent (``docs/rbfe.md``, "Dummy atoms where both ligands have unmapped atoms
on one atom"):

* improper torsions with D and an anchor are deleted at a dual and a triple junction (they would be terms beyond the
  three of sec. 2.1.1; the source does not treat force-field impropers);
* the first bond, angle and dihedral of the two possibilities of a dual junction, which leaves the dummy where it is
  (the 90 degrees of the second move a group out of the plane of X along lambda, which "may reduce phase-space
  overlap and necessitate additional λ-states", sec. 2.1.2: windows of short legs of toluene into phenol no longer
  overlapped); its anchor P and atom Z: the first, by the index of the hybrid, with a Z bonded to more than one atom;
  the anchor that keeps the dihedrals of the deeper dummy atoms of a triple junction, and the atom Z of a terminal
  junction, is the one of the lowest index;
* the adapted equilibrium values of a triple junction: the direction of D that minimises the sum of the squared
  deviations of its three angles from those of the force field, with X and the anchors at the prepared positions of
  the other ligand (held there instead of minimised together with the dummy atom, as Fleck 2021 does with a force
  constant of 0.05 kcal/(mol rad^2));
* along lambda, each term goes linearly from that of the force field (where its ligand exists) to that of the dummy,
  with the steric lambda of its ligand (``lambda_sterics_delete`` for A, ``lambda_sterics_insert`` for B): a group has
  its own geometry wherever it interacts fully.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KCAL = 4.184
#: the force constant of the angles to a dummy bridge atom at a dual junction: 100 kcal/(mol rad^2) of CHARMM (Fleck
#: 2021, Table 3), in OpenMM's convention (kJ/(mol rad^2))
K_DUAL = 2.0 * 100.0 * KCAL
#: the force constant of the three angles at a triple junction: 3.55 kcal/(mol rad^2) of CHARMM (Fleck 2021, sec. 3)
K_TRIPLE = 2.0 * 3.55 * KCAL
#: the dihedral that anchors a dummy bridge atom at a dual junction: periodicity 1, 100 kcal/mol (Fleck 2021, Table 3,
#: TOL2PYR-2; CHARMM's dihedral energy K (1 + cos(n phi - delta)) is OpenMM's), kJ/mol
K_ANCHOR = 100.0 * KCAL
RIGHT_ANGLE = math.pi / 2.0

TERMINAL, DUAL, TRIPLE = "terminal", "dual", "triple"
_TYPES = {1: TERMINAL, 2: DUAL, 3: TRIPLE}


class JunctionError(ValueError):
    """Unmapped atoms on one atom with both ligands in a shape the rules of Fleck 2021 do not cover."""


@dataclass(frozen=True)
class Junction:
    #: the mapped atom X (hybrid index)
    atom: int
    #: its mapped neighbours, ascending
    anchors: tuple[int, ...]
    #: the dummy groups of this ligand at X: (dummy bridge atom, atoms of the group)
    groups: tuple[tuple[int, frozenset], ...]

    @property
    def kind(self) -> str:
        return _TYPES[len(self.anchors)]


def _component(neighbours, start: int, members: set) -> set:
    out, todo = {start}, [start]
    while todo:
        x = todo.pop()
        for y in neighbours.get(x, ()):
            if y in members and y not in out:
                out.add(y)
                todo.append(y)
    return out


def find_junctions(neighbours: dict, core: set, unique: set, other_neighbours: dict, other_unique: set) -> list:
    """The junctions of one ligand: every mapped atom bonded to unmapped atoms of this ligand (``unique``) and of the
    other (``other_unique``), with the bonds of each ligand (``neighbours``, ``other_neighbours``: atom -> bonded atoms,
    hybrid indices). Raises JunctionError where a dummy group is bonded to X through two atoms or to another mapped
    atom (Fleck 2021 assumes "that the dummy groups are always disjoint", sec. 2.3), or X has no anchor or more than
    three."""
    out = []
    for x in sorted(core):
        own = sorted(y for y in neighbours.get(x, ()) if y in unique)
        if not own or not any(y in other_unique for y in other_neighbours.get(x, ())):
            continue
        anchors = tuple(sorted(y for y in neighbours.get(x, ()) if y in core))
        if not 1 <= len(anchors) <= 3:
            raise JunctionError(f"mapped atom {x} carries unmapped atoms of both ligands and has {len(anchors)} mapped "
                                "neighbours; the rules for dummy atoms (Fleck et al. 2021) cover one to three")
        groups, seen = [], set()
        for d in own:
            if d in seen:
                raise JunctionError(f"the unmapped atoms bonded to mapped atom {x} form a ring through it; the rules "
                                    "for dummy atoms (Fleck et al. 2021) take a group bonded to it through one atom")
            atoms = _component(neighbours, d, unique)
            seen |= atoms
            outside = {y for a in atoms for y in neighbours.get(a, ()) if y in core and y != x}
            if outside:
                raise JunctionError(f"the unmapped atoms bonded to mapped atom {x} are also bonded to mapped atoms "
                                    f"{sorted(outside)}; the rules for dummy atoms (Fleck et al. 2021) take a group "
                                    "bonded to the mapped atoms through one atom")
            groups.append((d, frozenset(atoms)))
        out.append(Junction(x, anchors, tuple(groups)))
    return out


def adapted_angles(x_pos, anchor_pos, d_direction, theta0) -> list[float]:
    """The equilibrium values of the angles P-X-D of a triple junction (radians): the direction of D that minimises
    the sum of the squared deviations of the three angles from ``theta0``, with X at ``x_pos`` and the anchors at
    ``anchor_pos`` (nm); the angles of that direction. The fit starts from ``d_direction`` and from the direction
    opposite to the sum of the bonds to the anchors (the free site of a tetrahedral X), and keeps the better."""
    from scipy.optimize import minimize

    x = np.asarray(x_pos, dtype=float)
    e = [np.asarray(p, dtype=float) - x for p in anchor_pos]
    e = [v / np.linalg.norm(v) for v in e]
    t0 = np.asarray(theta0, dtype=float)

    def angles(v):
        v = v / np.linalg.norm(v)
        return np.array([math.acos(max(-1.0, min(1.0, float(np.dot(v, ei))))) for ei in e])

    def cost(v):
        return float(np.sum((angles(v) - t0) ** 2))

    starts = [np.asarray(d_direction, dtype=float)]
    free = -np.sum(e, axis=0)
    if np.linalg.norm(free) > 1e-6:
        starts.append(free)
    fits = [minimize(cost, s / np.linalg.norm(s), method="Nelder-Mead",
                     options={"xatol": 1e-12, "fatol": 1e-16, "maxiter": 20000}) for s in starts]
    best = min(fits, key=lambda r: r.fun)
    return [float(a) for a in angles(best.x)]


class DummyTerms:
    """The terms of one ligand at its dummy end by the rules of Fleck 2021 (module docstring).

    ``neighbours``: bonds of this ligand (atom -> bonded atoms, hybrid indices); ``core``, ``unique``: the mapped atoms
    and this ligand's unmapped ones; ``other_neighbours``, ``other_unique``: the same of the other ligand; ``angles``:
    the angle terms of this ligand, (i, j, k) with i < k -> (theta0 rad, k kJ/(mol rad^2)); ``own_positions``,
    ``other_positions``: atom -> position (nm) in the prepared end state of this ligand and of the other."""

    def __init__(self, neighbours, core, unique, other_neighbours, other_unique, angles, own_positions,
                 other_positions):
        self.neighbours = {k: set(v) for k, v in neighbours.items()}
        self.core, self.unique = set(core), set(unique)
        self.junctions = find_junctions(self.neighbours, self.core, self.unique, other_neighbours, set(other_unique))
        self._where = {}                       # dummy atom -> (index of its junction, index of its group)
        for n, j in enumerate(self.junctions):
            for g, (_d, atoms) in enumerate(j.groups):
                for a in atoms:
                    self._where[a] = (n, g)
        self._theta0 = {}                      # angle key -> adapted equilibrium value (triple junctions)
        #: dummy bridge atom of a dual junction -> (anchor P, atom Z, phase) of its dihedral D-X-P-Z, or None where no
        #: anchor has a mapped neighbour other than X (then the two angles at 90 degrees)
        self._anchor = {}
        for j in self.junctions:
            if j.kind == DUAL:
                for d, _atoms in j.groups:
                    self._anchor[d] = self._dihedral_anchor(j, d, own_positions)
            if j.kind != TRIPLE:
                continue
            for d, _atoms in j.groups:
                keys = [_angle_key(p, j.atom, d) for p in j.anchors]
                present = [(p, k) for p, k in zip(j.anchors, keys) if k in angles]
                if not present:
                    continue
                values = adapted_angles(other_positions[j.atom], [other_positions[p] for p, _k in present],
                                        np.asarray(own_positions[d]) - np.asarray(own_positions[j.atom]),
                                        [angles[k][0] for _p, k in present])
                for (_p, k), value in zip(present, values):
                    self._theta0[k] = value

    def _dihedral_anchor(self, j: Junction, d: int, positions):
        """(P, Z, phase) of the dihedral D-X-P-Z that anchors the dummy bridge atom ``d`` of a dual junction: of the
        anchors P and their mapped neighbours Z other than X, the first with a Z bonded to more than one atom (not a
        terminal atom that turns with its group: "problems may arise if the 'anchor' of the dihedral angle term is
        itself a rotamer, such as a methyl group", Fleck 2021, sec. 2.3.2), else the first; the phase puts the minimum
        of the term at the dihedral of the prepared end state of this ligand ("this modified energy term positions the
        dummy atom in the correct geometry"). None where no anchor has such a Z."""
        pairs = [(p, z) for p in j.anchors for z in sorted(self.neighbours.get(p, ()))
                 if z in self.core and z != j.atom]
        if not pairs:
            return None
        inner = [(p, z) for p, z in pairs if len(self.neighbours.get(z, ())) > 1]
        p, z = (inner or pairs)[0]
        phi = _dihedral(*(np.asarray(positions[a], dtype=float) for a in (d, j.atom, p, z)))
        # K (1 + cos(phi - phase)) is lowest at phi - phase = pi
        return p, z, float(math.remainder(phi - math.pi, 2.0 * math.pi))

    def anchor_torsions(self) -> list:
        """The dihedral terms that anchor the dummy bridge atoms of dual junctions, which the force field does not have:
        ((D, X, P, Z), phase rad, force constant kJ/mol), periodicity 1, at the dummy end only."""
        out = []
        for j in self.junctions:
            for d, _atoms in j.groups:
                anchor = self._anchor.get(d)
                if anchor is not None:
                    p, z, phase = anchor
                    out.append(((d, j.atom, p, z), phase, K_ANCHOR))
        return out

    def _kept_anchor(self, j: Junction, d: int) -> int:
        """The anchor to which the dihedrals of the deeper dummy atoms of the group of ``d`` are kept."""
        anchor = self._anchor.get(d)
        return anchor[0] if anchor is not None else j.anchors[0]

    def changed(self) -> bool:
        return bool(self.junctions)

    def _classify(self, atoms):
        """(junction, groups touched) of a term with atoms of a dummy group, None for another term; "delete" for a
        term that touches dummy atoms of two junctions or unmapped atoms outside the groups of its junction."""
        touched = {self._where[a] for a in atoms if a in self._where}
        if not touched:
            return None
        if len({n for n, _g in touched}) > 1:
            return "delete"
        n = next(iter(touched))[0]
        j = self.junctions[n]
        if any(a in self.unique and a not in self._where for a in atoms):
            return "delete"
        return j, {g for _n, g in touched}

    def angle(self, key, theta0: float, k: float):
        """The angle term ``key`` at the dummy end: None where it is not changed, else (theta0, k), k 0 for a deleted
        term."""
        c = self._classify(key)
        if c is None:
            return None
        if c == "delete":
            return (theta0, 0.0)
        j, groups = c
        physical = [a for a in key if a in self.core]
        if physical == [j.atom]:
            return (theta0, 0.0) if len(groups) > 1 and j.kind != TERMINAL else None
        if not physical:
            return None
        if j.kind == TERMINAL:
            return None
        if j.kind == DUAL:
            d = next(a for a in key if a in self._where)
            anchor = self._anchor.get(d)
            if anchor is None:
                return (RIGHT_ANGLE, K_DUAL)
            # bond, angle and dihedral: the angle to the anchor of the dihedral unchanged, the other deleted
            return None if anchor[0] in key else (theta0, 0.0)
        return (self._theta0.get(_angle_key(*key), theta0), K_TRIPLE)

    def _proper(self, atoms) -> bool:
        return all(atoms[i + 1] in self.neighbours.get(atoms[i], ()) for i in range(3))

    def torsion(self, atoms):
        """The torsion term on ``atoms`` at the dummy end: None where it is not changed, True where it is kept, False
        where it is deleted."""
        c = self._classify(atoms)
        if c is None:
            return None
        if c == "delete":
            return False
        j, groups = c
        physical = [a for a in atoms if a in self.core]
        if not physical:
            return None
        if physical == [j.atom]:
            return False if len(groups) > 1 and j.kind != TERMINAL else None
        if not self._proper(atoms):
            return None if j.kind == TERMINAL else False
        seq = list(atoms) if atoms[0] in self._where else list(reversed(atoms))
        bridges = {d for d, _a in j.groups}
        if seq[0] in bridges and seq[1] == j.atom and seq[2] in j.anchors and seq[3] in self.core:
            # D-X-P-Z: from the dummy bridge atom into the physical molecule
            if j.kind != TERMINAL:
                return False
            return seq[3] == self._z_of(j.atom, seq[2])
        if seq[0] in self._where and seq[1] in bridges and seq[2] == j.atom and seq[3] in j.anchors:
            # D'-D-X-P: a deeper dummy atom anchored to an anchor
            if j.kind == TERMINAL:
                return True
            return seq[3] == (self._kept_anchor(j, seq[1]) if j.kind == DUAL else j.anchors[0])
        return None if j.kind == TERMINAL else False

    def _z_of(self, x: int, p: int) -> int | None:
        z = sorted(y for y in self.neighbours.get(p, ()) if y in self.core and y != x)
        return z[0] if z else None

    def notes(self, name: str) -> list[str]:
        return [f"mapped atom {j.atom} carries unmapped atoms of both ligands: the dummy atoms of {name} there have "
                f"the bonded terms of a {j.kind} junction (Fleck et al. 2021)" for j in self.junctions]


def _angle_key(a, b, c):
    return (a, b, c) if a < c else (c, b, a)


def _dihedral(p0, p1, p2, p3) -> float:
    """The dihedral angle p0-p1-p2-p3 (radians), with OpenMM's sign."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return float(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
