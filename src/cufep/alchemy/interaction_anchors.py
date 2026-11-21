"""Boresch anchors from the protein-ligand interactions of an unrestrained MD.

The rule of ``restraints.selection: interactions``. It is the one of Ding, Liu, Wang, Xia, Wang
and Yan, "Development and large-scale benchmarks of a protein-ligand absolute binding free
energy toolkit" (arXiv:2603.22274 v3, sec. 2.1), read together with the source of that work
(github.com/ByteDance-Seed/felis at commit d6dff91, ``felis/protocols/boresch/``) where the paper
gives no number. The paper states the thresholds, which are settings here
(:class:`InteractionSettings`, ``restraints.interactions`` in the configuration); the constants
that only the source gives rank the candidates and are the named constants of this module.
``docs/configuration.md`` quotes both and says where cuFEP differs. No code of that work is used.

In words. Every interaction that ProLIF finds between the ligand and a residue in a frame is
given to one ligand atom, L1 (:func:`anchor_atom`). An interaction (L1, residue, type) is kept if
it is there in at least ``min_frame_fraction`` of the frames; the kept types of one (L1, residue)
are merged. The hosts P1, P2, P3 are the backbone N, CA and C of the residue in each of their six
orders, the guests L2 and L3 are bonded heavy atoms that are no chain ends. A set of anchors is a
candidate if the angles L1-L2-L3 and P1-L1-L2 lie within ``angle_range_deg`` often enough, and the
candidates are ranked by how often P1-L1-L2 and L1-P1-P2 do, by how far those angles are from 90
degrees, by the secondary structure of the residue and by the interaction. If that gives no
anchors, the residues whose backbone is nearest to the ligand are taken instead
(:func:`rank_by_distance`); if that gives none either, there is no restraint and the calculation
stops (:class:`cufep.alchemy.restraints.RestraintError`).

The angles are those of the restraint: L1-P1-P2 is theta_A (H1-H0-G0) and P1-L1-L2 is theta_B
(H0-G0-G1) of :class:`cufep.alchemy.restraints.BoreschRestraint`, with H0, H1, H2 = P1, P2, P3 and
G0, G1, G2 = L1, L2, L3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cufep.alchemy.restraints import RestraintError
from cufep.config.schema import frames_needed  # noqa: F401  (the one rounding of a fraction of the frames)

#: the interactions of ProLIF that can carry anchors, with the weight of each in the ranking and
#: in the merging of the types of one (L1, residue): ionic 0.9, cation-pi 0.7, hydrogen bond 0.6,
#: halogen bond 0.4, aromatic stacking 0.1 (the order of the paper, the numbers of the source,
#: anchor_scoring.py)
INTERACTION_WEIGHTS = {
    "Anionic": 0.9, "Cationic": 0.9,
    "CationPi": 0.7, "PiCation": 0.7,
    "HBDonor": 0.6, "HBAcceptor": 0.6,
    "XBAcceptor": 0.4, "XBDonor": 0.4,
    "EdgeToFace": 0.1, "FaceToFace": 0.1,
}
#: the interactions that count for the bonus of a residue with two of them: all but the stacking
BONUS_INTERACTIONS = frozenset(name for name, weight in INTERACTION_WEIGHTS.items() if weight > 0.1)
#: what the candidates of the distance method are called; it has no weight
CONTACT = "contact"
#: ligand atoms that are never L1: the S and both O of a sulfonyl group, both O of a carboxylic
#: acid, an ester or a carboxylate (SMARTS and the positions in each match; anchor_constants.py of
#: the source. The paper names "sulfonyl sulfur and oxygen atoms and carboxyl oxygen atoms"; the
#: second pattern also matches the oxygens of an ester outside a ring)
EXCLUDED_SMARTS = (("[#16](=O)=O", (0, 1, 2)), ("C(=O)[O!R]", (1, 2)), ("C(=O)[O-]", (1, 2)))
#: L1 is the interacting atom or the first suitable atom within this many bonds of it
BOND_REACH = 3
#: the ranking: the count of frames of an angle is multiplied by 1 while the two mean deviations
#: from 90 degrees add up to TAPER_START_DEG or less, by TAPER_FLOOR from TAPER_END_DEG on, and by
#: the straight line between (TAPER_FLOOR and TAPER_SPAN are the 0.7 and 0.3 of the source, which
#: are written out so that the product with a count rounds as it does there)
TAPER_START_DEG, TAPER_END_DEG, TAPER_FLOOR, TAPER_SPAN = 40.0, 60.0, 0.7, 0.3
#: ... the counts are turned into SCORE_LEVELS levels of max(MIN_LEVEL_FRAMES, frames // SCORE_LEVELS)
#: frames each, a residue of stable secondary structure is raised by STRUCTURE_LEVELS of them
SCORE_LEVELS, MIN_LEVEL_FRAMES, STRUCTURE_LEVELS = 50, 10, 3
#: ... and a residue with two kept interactions of BONUS_INTERACTIONS gets this many levels
BONUS_LEVELS = int(SCORE_LEVELS * 0.15)
#: the distance method ranks by the two counts together in steps of this fraction of the frames
DISTANCE_RESOLUTION = 0.02
#: the elements of the ligand atoms of the distance method: the first group that the ligand has,
#: each group together with those before it (anchor_ligand.py of the source)
DISTANCE_ELEMENTS = (("N", "O"), ("P", "S"), ("I", "Br", "Cl", "F"), ("Si", "C"))


@dataclass
class InteractionSettings:
    """The thresholds of the rule (``restraints.interactions``), with the values of the paper."""

    min_frame_fraction: float = 0.40
    angle_range_deg: tuple[float, float] = (45.0, 135.0)
    stable_structure_fraction: float = 0.5
    second_interaction_ratio: float = 0.6
    fallback_frame_fraction: float = 0.15
    fallback_cutoff_nm: float = 1.0


# --------------------------------------------------------------------------- the ligand
@dataclass
class LigandGraph:
    """What the rule needs of the ligand: its bonds and which atoms can be anchors."""

    neighbours: list[list[int]]          # bonded atoms of every atom, ascending
    heavy: list[bool]
    symbols: list[str]
    excluded: frozenset[int]

    @classmethod
    def from_rdkit(cls, rdmol) -> "LigandGraph":
        from rdkit import Chem

        neighbours = [sorted(n.GetIdx() for n in atom.GetNeighbors()) for atom in rdmol.GetAtoms()]
        excluded: set[int] = set()
        for smarts, positions in EXCLUDED_SMARTS:
            for match in rdmol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
                excluded.update(match[i] for i in positions)
        return cls(neighbours, [atom.GetAtomicNum() > 1 for atom in rdmol.GetAtoms()],
                   [atom.GetSymbol() for atom in rdmol.GetAtoms()], frozenset(excluded))

    def chain_end(self, atom: int) -> bool:
        """Whether ``atom`` is bonded to one atom only ("terminal"; hydrogens count as bonded atoms,
        so the carbon of a methyl group is no chain end)."""
        return len(self.neighbours[atom]) == 1

    def within_bonds(self, atom: int, bonds: int) -> list[int]:
        """The atoms up to ``bonds`` bonds from ``atom``, nearest first; those equally far in the
        order in which they are reached, the bonded atoms of each atom by ascending number (the
        source walks through a set there, whose order Python does not define)."""
        found = [atom]
        seen = {atom}
        shell = [atom]
        for _ in range(bonds):
            reached = []
            for a in shell:
                for b in self.neighbours[a]:
                    if b not in seen:
                        seen.add(b)
                        reached.append(b)
            found += reached
            shell = reached
        return found[1:]

    def suitable(self, atom: int) -> bool:
        """Whether ``atom`` can be L1: a heavy atom that is not excluded and has at least two
        heavy atoms bonded to it."""
        return (self.heavy[atom] and atom not in self.excluded
                and sum(self.heavy[b] for b in self.neighbours[atom]) > 1)

    def anchor_atom(self, atom: int) -> int | None:
        """L1 for an interaction of ``atom``: the atom itself if it is suitable, else the first
        suitable atom within :data:`BOND_REACH` bonds; None if there is none."""
        for a in (atom, *self.within_bonds(atom, BOND_REACH)):
            if self.suitable(a):
                return a
        return None

    def guest_pairs(self, l1: int) -> list[tuple[int, int]]:
        """Every (L2, L3) for ``l1``: L2 a heavy atom bonded to L1 that is no chain end, L3 a
        heavy atom bonded to L2, no chain end and not L1."""
        pairs = []
        for l2 in self.neighbours[l1]:
            if not self.heavy[l2] or self.chain_end(l2):
                continue
            pairs += [(l2, l3) for l3 in self.neighbours[l2]
                      if self.heavy[l3] and not self.chain_end(l3) and l3 != l1]
        return pairs

    def has_guest_anchors(self) -> bool:
        """Whether any atom of the ligand can be L1 with an L2 and an L3: without one neither
        method can give anchors, which is known before any MD is run."""
        return any(self.suitable(a) and self.guest_pairs(a) for a in range(len(self.heavy)))


# --------------------------------------------------------------------------- the frames
@dataclass
class Frames:
    """The analysed frames in the words of the rule. Every index is a number of the System.

    ``xyz`` (frames x frame atoms x 3, nm) holds whole molecules with the ligand in the periodic
    image nearest to the protein (:func:`cufep.alchemy.restraints.ligand_next_to_protein`), so
    that an angle between protein and ligand atoms is that of the neighbours and not of images
    (the source measures the angles as the coordinates were written).
    """

    xyz: np.ndarray
    atoms: np.ndarray                          # the System's atom number of every frame atom
    ligand: list[int]                          # the ligand atoms in the order of its molecule
    #: residue -> its backbone (N, CA, C): :func:`cufep.alchemy.restraints.backbone_hosts`
    hosts: dict[int, tuple[int, int, int]]
    #: residue -> the fraction of the frames in which it is in a stable secondary structure:
    #: :func:`cufep.alchemy.restraints.stable_structure_fractions`
    stable_fraction: dict[int, float] = field(default_factory=dict)
    boxes: np.ndarray | None = None            # frames x 3 x 3, for the distance method
    trajectory: np.ndarray | None = None       # which trajectory every frame is of
    #: residue -> the atoms named N, CA or C of a protein residue that is no host (a cap), for the distance method
    other_backbone: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atoms = np.asarray(self.atoms, dtype=int)
        self._row = {int(a): i for i, a in enumerate(self.atoms)}

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def of(self, atom: int, frames=None) -> np.ndarray:
        """The positions of one atom of the System, in all frames or in ``frames``."""
        column = self.xyz[:, self._row[int(atom)], :] if frames is None else self.xyz[frames, self._row[int(atom)], :]
        return np.asarray(column, dtype=float)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The angle a-b-c at b in degrees, frame by frame."""
    v1, v2 = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("fi,fi->f", v1, v2) / np.sqrt(np.einsum("fi,fi->f", v1, v1) * np.einsum("fi,fi->f", v2, v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def frames_in_range(angles: np.ndarray, bounds) -> tuple[int, float]:
    """(frames in which the angle lies within ``bounds``, both ends included; mean distance of the
    angle from 90 degrees over all the frames given). An angle that is not defined is in no range
    and counts as 90 degrees away."""
    low, high = bounds
    with np.errstate(invalid="ignore"):
        inside = int(np.count_nonzero((angles >= low) & (angles <= high)))
    away = np.nan_to_num(np.abs(angles - 90.0), nan=90.0)
    return inside, float(away.mean()) if len(away) else 90.0


# --------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    """One set of anchors and the numbers it is ranked by."""

    host: tuple[int, int, int]           # P1, P2, P3 = H0, H1, H2
    guest: tuple[int, int, int]          # L1, L2, L3 = G0, G1, G2
    residue: int
    interaction: str                     # the merged type, or CONTACT
    n_frames: int                        # frames of the interaction (or of the contact)
    n_theta_a: int                       # frames with L1-P1-P2 in range
    n_theta_b: int                       # frames with P1-L1-L2 in range
    n_host: int                          # frames with P1-P2-P3 in range
    n_guest: int                         # frames with L1-L2-L3 in range
    away_theta_a: float                  # mean |angle - 90| in degrees, over the frames of the candidate
    away_theta_b: float
    away_host: float
    away_guest: float
    stable: bool = False
    score: float | None = None           # the interaction method only
    parts: dict = field(default_factory=dict)
    refused: str | None = None           # why it was passed over although ranked before the chosen one

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in ("residue", "interaction", "n_frames", "n_theta_a", "n_theta_b", "n_host",
                                             "n_guest", "away_theta_a", "away_theta_b", "away_host", "away_guest",
                                             "stable", "score", "refused")}
        out.update(host=[int(i) for i in self.host], guest=[int(i) for i in self.guest], parts=dict(self.parts))
        return out


def _expand(frames: Frames, graph: LigandGraph, l1_local: int, residue: int, interaction: str, where: np.ndarray,
            needed: int, bounds) -> list[Candidate]:
    """The candidates of one (L1, residue): the six orders of the backbone with every (L2, L3)
    whose L1-L2-L3 and P1-L1-L2 lie within ``bounds`` in at least ``needed`` of the frames
    ``where`` of the interaction. L1-P1-P2 and P1-P2-P3 are counted and decide nothing here."""
    if l1_local in graph.excluded or residue not in frames.hosts:
        return []
    n, ca, c = frames.hosts[residue]
    orders = ((n, ca, c), (n, c, ca), (ca, n, c), (ca, c, n), (c, n, ca), (c, ca, n))
    l1 = frames.ligand[l1_local]
    x_l1 = frames.of(l1, where)
    out = []
    for p1, p2, p3 in orders:
        x_p1, x_p2, x_p3 = frames.of(p1, where), frames.of(p2, where), frames.of(p3, where)
        n_host, away_host = frames_in_range(angle_deg(x_p1, x_p2, x_p3), bounds)
        n_a, away_a = frames_in_range(angle_deg(x_l1, x_p1, x_p2), bounds)
        for l2_local, l3_local in graph.guest_pairs(l1_local):
            l2, l3 = frames.ligand[l2_local], frames.ligand[l3_local]
            x_l2 = frames.of(l2, where)
            n_guest, away_guest = frames_in_range(angle_deg(x_l1, x_l2, frames.of(l3, where)), bounds)
            n_b, away_b = frames_in_range(angle_deg(x_p1, x_l1, x_l2), bounds)
            if n_guest >= needed and n_b >= needed:
                out.append(Candidate((p1, p2, p3), (l1, l2, l3), residue, interaction, len(where), n_a, n_b,
                                     n_host, n_guest, away_a, away_b, away_host, away_guest))
    return out


# --------------------------------------------------------------------------- interactions
@dataclass
class InteractionTable:
    """How often every interaction is there. ``frames_of[(l1, residue, type)]`` are the frames
    (ascending) of an interaction given to the ligand atom ``l1`` (a number of the ligand's
    molecule); ``kept`` are those that reach the threshold."""

    n_frames: int
    needed: int
    frames_of: dict[tuple[int, int, str], np.ndarray]
    kept: list[tuple[int, int, str]]

    def rows(self, frames: Frames | None = None) -> list[dict]:
        """One row per interaction for the output: counts, the fraction of the frames, the counts
        of each trajectory, whether it is kept, and the frames it is in."""
        rows = []
        for key in sorted(self.frames_of, key=lambda k: (-len(self.frames_of[k]), k)):
            where = self.frames_of[key]
            row = {"ligand_atom": int(key[0] if frames is None else frames.ligand[key[0]]), "residue": int(key[1]),
                   "interaction": key[2], "frames": int(len(where)),
                   "fraction": float(len(where) / self.n_frames) if self.n_frames else 0.0,
                   "kept": key in set(self.kept), "in_frames": [int(f) for f in where]}
            if frames is not None and frames.trajectory is not None:
                of = np.asarray(frames.trajectory)[where]
                row["frames_of_trajectory"] = {int(t): int(np.count_nonzero(of == t))
                                               for t in np.unique(frames.trajectory)}
            rows.append(row)
        return rows


def interaction_table(records, graph: LigandGraph, n_frames: int, settings: InteractionSettings) -> InteractionTable:
    """Count the interactions of the frames. ``records[f]`` holds, for frame f, one
    ``(ligand atom, residue, type)`` for every interaction that ProLIF reports there: the first
    ligand atom of the interaction, as a number of the ligand's molecule. An interaction counts
    once in a frame, for the atom L1 that :meth:`LigandGraph.anchor_atom` gives it; types that
    carry no anchors and atoms without an L1 are left out."""
    found: dict[tuple[int, int, str], list[int]] = {}
    for f, frame in enumerate(records):
        here = set()
        for atom, residue, kind in frame:
            if kind not in INTERACTION_WEIGHTS:
                continue
            l1 = graph.anchor_atom(int(atom))
            if l1 is not None:
                here.add((l1, int(residue), str(kind)))
        for key in here:
            found.setdefault(key, []).append(f)
    needed = frames_needed(n_frames, settings.min_frame_fraction)
    frames_of = {key: np.array(sorted(where), dtype=int) for key, where in found.items()}
    kept = sorted(key for key, where in frames_of.items() if len(where) >= needed)
    return InteractionTable(int(n_frames), needed, frames_of, kept)


def _merged(table: InteractionTable) -> dict[tuple[int, int], tuple[str, np.ndarray]]:
    """The kept interactions of every (L1, residue) as one: the frames of all its types together,
    and the type of the largest weight (of two of the same weight the first by name: the weight
    is all that the ranking reads)."""
    merged: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for l1, residue, kind in table.kept:
        where = table.frames_of[(l1, residue, kind)]
        if (l1, residue) not in merged:
            merged[(l1, residue)] = (kind, where)
            continue
        old, frames = merged[(l1, residue)]
        best = min((old, kind), key=lambda k: (-INTERACTION_WEIGHTS[k], k))
        merged[(l1, residue)] = (best, np.union1d(frames, where))
    return merged


def _residues_with_two(table: InteractionTable, ratio: float) -> set[int]:
    """The residues that get the bonus: at least two kept interactions of
    :data:`BONUS_INTERACTIONS` (before the merging), the second most frequent of them in more than
    ``ratio`` of the frames of the most frequent."""
    counts: dict[int, list[int]] = {}
    for l1, residue, kind in table.kept:
        if kind in BONUS_INTERACTIONS:
            counts.setdefault(residue, []).append(len(table.frames_of[(l1, residue, kind)]))
    out = set()
    for residue, numbers in counts.items():
        if len(numbers) >= 2:
            first, second = sorted(numbers, reverse=True)[:2]
            if second / first > ratio:
                out.add(residue)
    return out


def _level(count: int, most: int, step: int) -> int:
    """The level of a count of frames: SCORE_LEVELS for the largest count ``most``, one less for
    every whole ``step`` frames below it (one more for every whole step above it, which a raised
    count can be)."""
    if most >= count:
        return SCORE_LEVELS - (most - count) // step
    return SCORE_LEVELS + (count - most) // step


def rank_by_interactions(frames: Frames, graph: LigandGraph, table: InteractionTable,
                         settings: InteractionSettings) -> list[Candidate]:
    """The candidates of the kept interactions, best first."""
    candidates: list[Candidate] = []
    for (l1, residue), (kind, where) in _merged(table).items():
        candidates += _expand(frames, graph, l1, residue, kind, where, table.needed, settings.angle_range_deg)
    if not candidates:
        return []
    with_two = _residues_with_two(table, settings.second_interaction_ratio)
    most = max(max(c.n_theta_a, c.n_theta_b) for c in candidates)
    step = max(MIN_LEVEL_FRAMES, frames.n // SCORE_LEVELS)
    for c in candidates:
        away = c.away_theta_a + c.away_theta_b
        if away > TAPER_END_DEG:
            taper = TAPER_FLOOR
        elif away > TAPER_START_DEG:
            taper = TAPER_FLOOR + (TAPER_END_DEG - away) / (TAPER_END_DEG - TAPER_START_DEG) * TAPER_SPAN
        else:
            taper = 1.0
        c.stable = frames.stable_fraction.get(c.residue, 0.0) > settings.stable_structure_fraction
        raised = STRUCTURE_LEVELS * step * int(c.stable)
        level_a = _level(int(taper * c.n_theta_a) + raised, most, step)
        level_b = _level(int(taper * c.n_theta_b) + raised, most, step)
        bonus = BONUS_LEVELS if c.residue in with_two else 0
        weight = INTERACTION_WEIGHTS[c.interaction]
        c.score = float(level_a + level_b + weight + bonus)
        c.parts = {"taper": float(taper), "level_theta_a": int(level_a), "level_theta_b": int(level_b),
                   "weight": float(weight), "bonus": int(bonus), "frames_per_level": int(step)}
    # the score in hundredths, as the source compares it; then the atoms, so that the order is one
    return sorted(candidates, key=lambda c: (-int(c.score * 100), c.host, c.guest))


# --------------------------------------------------------------------------- distance
def distance_atoms(graph: LigandGraph) -> list[int]:
    """The ligand atoms of the distance method (numbers of the ligand's molecule): the atoms that
    are L1 for some atom of the ligand, of the first group of :data:`DISTANCE_ELEMENTS` that the
    ligand has among them."""
    anchors = sorted({a for a in (graph.anchor_atom(i) for i in range(len(graph.heavy))) if a is not None})
    elements: tuple[str, ...] = ()
    for group in DISTANCE_ELEMENTS:
        elements += group
        chosen = [a for a in anchors if graph.symbols[a] in elements]
        if chosen:
            return chosen
    return []


def contacts(frames: Frames, graph: LigandGraph, settings: InteractionSettings) -> dict[int, tuple[int, np.ndarray]]:
    """The contacts of the distance method: ``{residue: (L1, frames)}``. In every frame the
    nearest pair of a ligand atom of :func:`distance_atoms` and a backbone atom (N, CA, C of the
    host residues) is taken, across the periodic boundary, if it is within
    ``fallback_cutoff_nm``; the frame is one of the residue of that backbone atom. L1 of a residue
    is the ligand atom that is the nearest one in most of its frames (the earliest of them if
    several are)."""
    from cufep.alchemy.lrc import nearest_image

    selected = distance_atoms(graph)
    backbone = [(atom, residue) for residue, trio in sorted(frames.hosts.items()) for atom in trio]
    if not selected or not backbone:
        return {}
    # the backbone atoms of residues that cannot hold anchors (a cap has a C or an N and no three of them) are
    # looked at too, as in the source, which takes every atom named N, CA or C of the protein and drops the
    # residues without anchors afterwards: a frame in which such an atom is the nearest is a frame of no host,
    # not of the host that comes next (review of 877ab88)
    backbone += [(atom, None) for _residue, atoms in sorted(frames.other_backbone.items()) for atom in atoms]
    lig = np.stack([frames.of(frames.ligand[a]) for a in selected], axis=1)           # frames x ligand atoms x 3
    bb = np.stack([frames.of(atom) for atom, _residue in backbone], axis=1)            # frames x backbone x 3
    nearest: list[tuple[int, int, int]] = []                                           # frame, L1, residue
    for f in range(frames.n):
        delta = (lig[f][:, None, :] - bb[f][None, :, :]).reshape(-1, 3)
        if frames.boxes is not None:
            delta = nearest_image(delta, np.asarray(frames.boxes[f], dtype=float))
        dist = np.sqrt((delta * delta).sum(axis=1))
        k = int(np.argmin(dist))
        if dist[k] <= settings.fallback_cutoff_nm and backbone[k % len(backbone)][1] is not None:
            nearest.append((f, selected[k // len(backbone)], backbone[k % len(backbone)][1]))
    counts: dict[int, dict[int, int]] = {}
    where: dict[int, list[int]] = {}
    for f, l1, residue in nearest:
        counts.setdefault(residue, {})
        counts[residue][l1] = counts[residue].get(l1, 0) + 1
        where.setdefault(residue, []).append(f)
    out = {}
    for residue, numbers in counts.items():
        l1 = max(numbers, key=lambda a: numbers[a])            # the first of equal counts: dicts keep their order
        out[residue] = (l1, np.array(where[residue], dtype=int))
    return out


def rank_by_distance(frames: Frames, graph: LigandGraph, settings: InteractionSettings) -> list[Candidate]:
    """The candidates of the distance method, best first: the residues with a contact in at
    least ``fallback_frame_fraction`` of the frames, the angle condition with that fraction, and
    the order by the two counts together (in steps of :data:`DISTANCE_RESOLUTION` of the frames),
    then by the two deviations from 90 degrees, then by those of the host and guest angles."""
    needed = frames_needed(frames.n, settings.fallback_frame_fraction)
    candidates: list[Candidate] = []
    for residue, (l1, where) in sorted(contacts(frames, graph, settings).items()):
        if len(where) >= needed:
            candidates += _expand(frames, graph, l1, residue, CONTACT, where, needed, settings.angle_range_deg)
    resolution = max(1, int(DISTANCE_RESOLUTION * frames.n))
    for c in candidates:
        c.stable = frames.stable_fraction.get(c.residue, 0.0) > settings.stable_structure_fraction
        c.parts = {"frames_per_step": int(resolution)}
    return sorted(candidates, key=lambda c: (-((c.n_theta_b + c.n_theta_a) // resolution),
                                             c.away_theta_b + c.away_theta_a, c.away_host + c.away_guest,
                                             c.host, c.guest))


# --------------------------------------------------------------------------- the choice
@dataclass
class AnchorChoice:
    """The anchors that were chosen and what they were chosen from."""

    host: tuple[int, int, int]
    guest: tuple[int, int, int]
    method: str                          # "interactions" | "distance"
    chosen: Candidate
    table: InteractionTable
    ranked: dict[str, list[Candidate]]   # per method that was tried, in its order


def choose_anchors(frames: Frames, graph: LigandGraph, records, settings: InteractionSettings | None = None,
                   problem_of=None) -> AnchorChoice:
    """The anchors of the rule: the best candidate of the interactions, or, if they give none,
    of the distance method.

    ``problem_of(host, guest)`` says why a set of anchors cannot be used (or None): the rules that
    every restraint of cuFEP is held to (:func:`cufep.alchemy.restraints.anchor_geometry_problem`).
    A candidate that is refused is passed over and keeps the reason. With no candidate left in
    either method there is no restraint: :class:`~cufep.alchemy.restraints.RestraintError`.
    """
    settings = settings or InteractionSettings()
    table = interaction_table(records, graph, frames.n, settings)
    ranked: dict[str, list[Candidate]] = {}
    for method, rank in (("interactions", lambda: rank_by_interactions(frames, graph, table, settings)),
                         ("distance", lambda: rank_by_distance(frames, graph, settings))):
        ranked[method] = rank()
        for candidate in ranked[method]:
            candidate.refused = problem_of(candidate.host, candidate.guest) if problem_of is not None else None
            if candidate.refused is None:
                return AnchorChoice(candidate.host, candidate.guest, method, candidate, table, ranked)
    tried = ", ".join(f"{len(v)} of the {k}" for k, v in ranked.items())
    # the reasons of the first ones (the leg ends here and writes no interactions.json to read them in)
    first = [c for candidates in ranked.values() for c in candidates if c.refused][:3]
    why = "".join(f"; host {list(c.host)} guest {list(c.guest)}: {c.refused}" for c in first)
    raise RestraintError(
        "no Boresch anchors can be chosen from the interactions of the MD: no interaction of the ligand with a "
        f"residue is kept in {table.needed} of {frames.n} frames with anchors at usable angles, and neither does the "
        f"backbone nearest to the ligand give any ({tried} candidates, each refused{why}). Give the anchors "
        "(restraints.selection: manual) or use restraints.selection: auto")
