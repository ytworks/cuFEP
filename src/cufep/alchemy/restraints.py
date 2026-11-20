"""Boresch restraints: force, standard-state correction and automatic atom selection.

Geometry (host atoms H0, H1, H2 on the protein; guest atoms G0, G1, G2 on the ligand):

    r      = |H0 - G0|
    thetaA = angle(H1, H0, G0)          thetaB = angle(H0, G0, G1)
    phiA   = dihedral(H2, H1, H0, G0)   phiB   = dihedral(H1, H0, G0, G1)
    phiC   = dihedral(H0, G0, G1, G2)

The automatic search follows OpenFE's ABFE protocol (itself based on Baumann et al. and
Alibay et al.): guest anchors are three bonded atoms of a rigid (aromatic) ring near the
ligand centre; host anchors are three bonded backbone atoms in stable secondary structure
with low RMSF, 0.5-1.5 nm from the guest anchor. Candidates with near-collinear or
unstable geometry are rejected and the remaining one with the smallest fluctuations,
measured in units of the restraint width, is taken. Force constants are OpenFE's defaults.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import asdict, dataclass

import numpy as np
import openmm as mm
import openmm.unit as unit

from cufep.alchemy.factory import LAMBDA_RESTRAINTS, RESTRAINT_GROUP

KCAL = 4.184
#: OpenFE defaults: 10 kcal/mol/A^2 and 80 kcal/mol/rad^2
K_R_DEFAULT = 10.0 * KCAL * 100.0          # kJ/mol/nm^2
K_ANGLE_DEFAULT = 80.0 * KCAL              # kJ/mol/rad^2
STANDARD_VOLUME_NM3 = 1.6605390666         # 1 mol/L
R_GAS = 8.31446261815324e-3                # kJ/mol/K


class RestraintError(RuntimeError):
    """No acceptable restraint could be chosen."""


@dataclass
class BoreschRestraint:
    host_atoms: tuple[int, int, int]       # (H0, H1, H2)
    guest_atoms: tuple[int, int, int]      # (G0, G1, G2)
    r0: float                              # nm
    theta_a0: float                        # rad
    theta_b0: float
    phi_a0: float
    phi_b0: float
    phi_c0: float
    k_r: float = K_R_DEFAULT
    k_theta_a: float = K_ANGLE_DEFAULT
    k_theta_b: float = K_ANGLE_DEFAULT
    k_phi_a: float = K_ANGLE_DEFAULT
    k_phi_b: float = K_ANGLE_DEFAULT
    k_phi_c: float = K_ANGLE_DEFAULT

    # -- force ------------------------------------------------------------------
    def create_force(self) -> mm.CustomCompoundBondForce:
        energy = (
            f"{LAMBDA_RESTRAINTS}*E;"
            "E=0.5*k_r*(distance(p3,p4)-r0)^2"
            "+0.5*k_ta*(angle(p2,p3,p4)-ta0)^2+0.5*k_tb*(angle(p3,p4,p5)-tb0)^2"
            "+0.5*k_pa*da^2+0.5*k_pb*db^2+0.5*k_pc*dc^2;"
            "da=xa-floor(xa/twopi+0.5)*twopi; xa=dihedral(p1,p2,p3,p4)-pa0;"
            "db=xb-floor(xb/twopi+0.5)*twopi; xb=dihedral(p2,p3,p4,p5)-pb0;"
            "dc=xc-floor(xc/twopi+0.5)*twopi; xc=dihedral(p3,p4,p5,p6)-pc0;"
            f"twopi={2 * math.pi!r}"
        )
        force = mm.CustomCompoundBondForce(6, energy)
        force.setName("cufep_boresch_restraint")
        force.addGlobalParameter(LAMBDA_RESTRAINTS, 1.0)
        names = ["k_r", "r0", "k_ta", "ta0", "k_tb", "tb0", "k_pa", "pa0", "k_pb", "pb0", "k_pc", "pc0"]
        for n in names:
            force.addPerBondParameter(n)
        h0, h1, h2 = self.host_atoms
        g0, g1, g2 = self.guest_atoms
        force.addBond([h2, h1, h0, g0, g1, g2],
                      [self.k_r, self.r0, self.k_theta_a, self.theta_a0, self.k_theta_b, self.theta_b0,
                       self.k_phi_a, self.phi_a0, self.k_phi_b, self.phi_b0, self.k_phi_c, self.phi_c0])
        force.setUsesPeriodicBoundaryConditions(True)
        force.setForceGroup(RESTRAINT_GROUP)
        return force

    # -- geometry ---------------------------------------------------------------
    def values(self, positions: np.ndarray) -> np.ndarray:
        """The six restraint coordinates for positions of shape (..., n_atoms, 3)."""
        h0, h1, h2 = (positions[..., i, :] for i in self.host_atoms)
        g0, g1, g2 = (positions[..., i, :] for i in self.guest_atoms)
        return np.stack([_dist(h0, g0), _angle(h1, h0, g0), _angle(h0, g0, g1),
                         _dihedral(h2, h1, h0, g0), _dihedral(h1, h0, g0, g1), _dihedral(h0, g0, g1, g2)], axis=-1)

    def energy(self, positions: np.ndarray) -> np.ndarray:
        """Restraint energy (kJ/mol, lambda_restraints = 1)."""
        v = self.values(positions)
        ref = np.array([self.r0, self.theta_a0, self.theta_b0, self.phi_a0, self.phi_b0, self.phi_c0])
        k = np.array([self.k_r, self.k_theta_a, self.k_theta_b, self.k_phi_a, self.k_phi_b, self.k_phi_c])
        d = v - ref
        d[..., 3:] = (d[..., 3:] + np.pi) % (2 * np.pi) - np.pi
        return 0.5 * np.sum(k * d * d, axis=-1)

    # -- standard state ----------------------------------------------------------
    def release_free_energy(self, temperature_K: float, method: str = "numerical") -> float:
        """Free energy (kJ/mol) of releasing the restraint on the *decoupled* ligand into
        the standard-state volume: ``-kT ln(8 pi^2 V0 / Z_restrained)``. Negative.

        ``numerical`` integrates the restraint partition function exactly (the six
        coordinates are separable); ``analytical`` is Boresch's Gaussian approximation.
        """
        kt = R_GAS * temperature_K
        if method == "analytical":
            z = (self.r0 ** 2 * math.sin(self.theta_a0) * math.sin(self.theta_b0) * (2 * math.pi * kt) ** 3
                 / math.sqrt(self.k_r * self.k_theta_a * self.k_theta_b * self.k_phi_a * self.k_phi_b * self.k_phi_c))
        elif method == "numerical":
            from scipy import integrate
            beta = 1.0 / kt
            r_hi = self.r0 + 12.0 * math.sqrt(kt / self.k_r)
            z_r = integrate.quad(lambda r: r * r * math.exp(-0.5 * beta * self.k_r * (r - self.r0) ** 2),
                                 0.0, r_hi, points=[self.r0], limit=200)[0]
            z = z_r
            for k, t0 in ((self.k_theta_a, self.theta_a0), (self.k_theta_b, self.theta_b0)):
                z *= integrate.quad(lambda t: math.sin(t) * math.exp(-0.5 * beta * k * (t - t0) ** 2),
                                    0.0, math.pi, points=[t0], limit=200)[0]
            for k in (self.k_phi_a, self.k_phi_b, self.k_phi_c):
                z *= integrate.quad(lambda p: math.exp(-0.5 * beta * k * p * p), -math.pi, math.pi,
                                    points=[0.0], limit=200)[0]
        else:
            raise ValueError("method must be 'numerical' or 'analytical'")
        return -kt * math.log(8.0 * math.pi ** 2 * STANDARD_VOLUME_NM3 / z)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["host_atoms"] = list(self.host_atoms)
        d["guest_atoms"] = list(self.guest_atoms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BoreschRestraint":
        d = dict(d)
        d["host_atoms"] = tuple(int(i) for i in d["host_atoms"])
        d["guest_atoms"] = tuple(int(i) for i in d["guest_atoms"])
        return cls(**d)


# --------------------------------------------------------------------------- geometry
def _dist(a, b):
    return np.linalg.norm(b - a, axis=-1)


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)


def circular_mean(x: np.ndarray) -> float:
    return float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()))


def circular_std(x: np.ndarray) -> float:
    r = min(1.0, float(np.hypot(np.sin(x).mean(), np.cos(x).mean())))
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12)))))


def kabsch_align(frames: np.ndarray, ref: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Superpose every frame on ``ref`` using the atoms ``sel``; returns aligned copies."""
    out = np.empty_like(frames)
    ref_c = ref[sel].mean(axis=0)
    q = ref[sel] - ref_c
    for f in range(frames.shape[0]):
        c = frames[f, sel].mean(axis=0)
        p = frames[f, sel] - c
        u_, _s, vt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(u_ @ vt))
        rot = u_ @ np.diag([1.0, 1.0, d]) @ vt
        out[f] = (frames[f] - c) @ rot + ref_c
    return out


# --------------------------------------------------------------------------- search
@dataclass
class RestraintSearchSettings:
    k_r: float = K_R_DEFAULT
    k_angle: float = K_ANGLE_DEFAULT
    rmsf_cutoff_nm: float = 0.1
    host_min_distance_nm: float = 0.5
    host_max_distance_nm: float = 1.5
    dssp_filter: bool = True
    #: theta_A and theta_B must stay this far (in kT of their force constant) from 0 and pi in every
    #: frame: OpenFE's check_angle_not_flat (contributed by V. Gapsys)
    flat_angle_energy_kt: float = 10.0
    max_guest_anchors: int = 8
    max_host_candidates: int = 60


#: the six restrained coordinates of a Boresch restraint
COORDINATES = ("r", "theta_A", "theta_B", "phi_A", "phi_B", "phi_C")
#: the four angles of the anchors: the two restrained ones and the angles of the host and of the
#: guest triple, which the first and the last dihedral angle need away from a straight line
ANCHOR_ANGLES = ("H2-H1-H0", "H1-H0-G0 (theta_A)", "H0-G0-G1 (theta_B)", "G0-G1-G2")


def coordinate_fluctuations(values: np.ndarray) -> dict[str, float]:
    """Standard deviation of each restrained coordinate over frames (``values``: frames x 6), the
    dihedral angles on the circle."""
    v = np.asarray(values, dtype=float)
    stds = [float(v[:, 0].std()), float(v[:, 1].std()), float(v[:, 2].std()),
            circular_std(v[:, 3]), circular_std(v[:, 4]), circular_std(v[:, 5])]
    return dict(zip(COORDINATES, stds))


def anchor_angles(restraint: "BoreschRestraint", positions: np.ndarray) -> np.ndarray:
    """The four angles of the anchors (ANCHOR_ANGLES) for positions of shape (..., n_atoms, 3)."""
    h0, h1, h2 = (positions[..., i, :] for i in restraint.host_atoms)
    g0, g1, g2 = (positions[..., i, :] for i in restraint.guest_atoms)
    return np.stack([_angle(h2, h1, h0), _angle(h1, h0, g0), _angle(h0, g0, g1), _angle(g0, g1, g2)], axis=-1)


#: OpenFE's is_collinear (from Yank): three atoms in a row are taken as on a straight line when the
#: inner product of their two normalised bond vectors exceeds this in magnitude
COLLINEAR_THRESHOLD = 0.9


def collinear_frames(restraint: "BoreschRestraint", frames: np.ndarray) -> np.ndarray:
    """For every frame, whether the six anchors H2-H1-H0-G0-G1-G2 are on a straight line by OpenFE's
    rule (:data:`COLLINEAR_THRESHOLD`, any three in a row)."""
    chain = [*reversed(restraint.host_atoms), *restraint.guest_atoms]
    p = np.asarray(frames, dtype=float)
    if p.ndim == 2:
        p = p[None]
    out = np.zeros(p.shape[0], dtype=bool)
    for i in range(4):
        v1 = p[:, chain[i + 1], :] - p[:, chain[i], :]
        v2 = p[:, chain[i + 2], :] - p[:, chain[i + 1], :]
        cos = np.einsum("fi,fi->f", v1, v2) / np.sqrt(np.einsum("fi,fi->f", v1, v1) * np.einsum("fi,fi->f", v2, v2))
        out |= np.abs(cos) > COLLINEAR_THRESHOLD
    return out


#: the three atoms in a row of the chain H2-H1-H0-G0-G1-G2 whose two bond vectors span the planes of
#: the dihedral angles phi_A (the first two), phi_B (the middle two) and phi_C (the last two)
ANCHOR_TRIPLES = ("H2-H1-H0", "H1-H0-G0", "H0-G0-G1", "G0-G1-G2")


def undefined_dihedral_planes(restraint: "BoreschRestraint", positions: np.ndarray) -> list[str]:
    """The rows of three anchors (:data:`ANCHOR_TRIPLES`) that span no plane in ``positions`` (one
    frame): the cross product of their two bond vectors is zero or not finite, so a dihedral angle
    of the restraint is not defined there and its force is NaN (review of 7b49066: a nitrile whose
    three atoms were exactly on a line in the last frame). Whether a plane exists, not a level."""
    chain = [*reversed(restraint.host_atoms), *restraint.guest_atoms]
    p = np.asarray(positions, dtype=float)
    out = []
    for k, name in enumerate(ANCHOR_TRIPLES):
        cross = np.cross(p[chain[k + 1]] - p[chain[k]], p[chain[k + 2]] - p[chain[k + 1]])
        norm = float(np.linalg.norm(cross))
        if not (np.isfinite(norm) and norm > 0.0):
            out.append(name)
    return out


def anchor_geometry_problem(restraint: "BoreschRestraint", frames: np.ndarray, flat_margin: float,
                            trajectory: bool = True, reference_copies=()) -> str | None:
    """Why the anchors of ``restraint`` are refused over ``frames`` (positions with the ligand in the
    periodic image nearest the host), or None. The rules are two of OpenFE's (EvaluateBoreschAtoms in
    openfe.protocols.restraint_utils.geometry.boresch.host) and nothing of cuFEP's own (the owner,
    2026-11-20; earlier versions also held all four angles to the flat margin and the last frame to
    30 degrees):

    * theta_A and theta_B stay ``flat_margin`` away from 0 and pi in every frame: the angle at which
      the restraint energy reaches 10 kT (OpenFE's check_angle_not_flat, contributed by V. Gapsys);
    * the six anchors H2-H1-H0-G0-G1-G2 are not on a straight line in every frame (OpenFE's
      ``not all(collinear)`` with is_collinear, from Yank: :func:`collinear_frames`): an acetonitrile
      whose N-C-C was chosen as guest anchors gave forces of NaN (review of PR 1).

    The second rule is one of a trajectory: ``trajectory=False`` (a single pose, such as the
    prepared one before the pre-equilibration) applies the first alone (review of 7b49066: a
    prepared pose with theta_A of 25 degrees, on a line by is_collinear, was refused although the
    frames of the pre-equilibration, 25 to 40 degrees, give the restraint). For a trajectory the
    dihedral angles must also be defined in its last frame, from which the reference values come
    (:func:`undefined_dihedral_planes`), and in every one of ``reference_copies``: other copies of
    that configuration, such as the frame before it was superposed on the binding site and the
    positions the alchemical run starts from. A copy can differ from ``frames[-1]`` by rounding,
    which can make three anchors exactly on a line in one and not in the other (review of 10589a6:
    after the superposition a nitrile that was exactly straight was taken for bent, and the force of
    the run was NaN).

    OpenFE's other conditions there (each dihedral within +-150 degrees, limits on the circular
    variance of the angles and dihedrals) were never part of cuFEP and are not applied."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 2:
        frames = frames[None]
    angles = np.atleast_2d(anchor_angles(restraint, frames))
    for k in (1, 2):
        name = ANCHOR_ANGLES[k]
        low, high = float(angles[:, k].min()), float(angles[:, k].max())
        if low < flat_margin or high > np.pi - flat_margin:
            return (f"the anchor angle {name} comes within {np.degrees(flat_margin):.0f} degrees of a straight line "
                    f"({np.degrees(low):.1f} to {np.degrees(high):.1f} degrees over the frames), where the restraint "
                    "energy is below 10 kT (OpenFE's check_angle_not_flat)")
    if not trajectory:
        return None
    if bool(np.all(collinear_frames(restraint, frames))):
        return ("the six anchors H2-H1-H0-G0-G1-G2 are on a straight line in every frame (three in a row with "
                f"|cos| above {COLLINEAR_THRESHOLD:g}, OpenFE's is_collinear): the dihedral angles of the restraint "
                "are not defined there")
    for copy in (frames[-1], *reference_copies):
        flat = undefined_dihedral_planes(restraint, copy)
        if flat:
            return (f"the anchors {', '.join(flat)} are exactly on a line in the last frame, from which the reference "
                    "values come: the dihedral angles of the restraint are not defined there")
    return None


def anchor_margins(settings: "RestraintSearchSettings", temperature_K: float) -> float:
    """The angle (radians) from 0 and pi of :func:`anchor_geometry_problem`: where the energy of a
    restrained angle reaches ``flat_angle_energy_kt`` kT."""
    kt = R_GAS * temperature_K
    return math.sqrt(2.0 * settings.flat_angle_energy_kt * kt / settings.k_angle)


@dataclass
class RestraintSearchResult:
    restraint: BoreschRestraint
    score: float
    fluctuations: dict[str, float]         # std of each coordinate over the trajectory
    ligand_rmsd_nm: float
    n_candidates: int
    notes: list[str]


def _stable_secondary_structure(topology, atoms: list, frames: np.ndarray, protein_atoms: np.ndarray) -> set[int]:
    """Residue indices (OpenMM) that are helix or strand in most frames, away from element ends."""
    import mdtraj as md

    sub = md.Topology.from_openmm(topology).subset(protein_atoms)
    stride = max(1, frames.shape[0] // 10)
    traj = md.Trajectory(frames[::stride][:, protein_atoms, :], sub)
    dssp = md.compute_dssp(traj, simplified=True)             # (n_frames, n_residues) of H/E/C/NA
    stable = ((dssp == "H") | (dssp == "E")).mean(axis=0) >= 0.8
    res_index: list[int] = []
    for a in protein_atoms:
        r = atoms[a].residue.index
        if not res_index or res_index[-1] != r:
            res_index.append(r)
    keep: set[int] = set()
    i, n = 0, min(len(stable), len(res_index))
    while i < n:
        if not stable[i]:
            i += 1
            continue
        j = i
        while j < n and stable[j]:
            j += 1
        if j - i >= 5:                                         # drop the end residues of each element
            keep.update(res_index[i + 1:j - 1])
        i = j
    return keep


def _guest_anchor_candidates(rdmol, ligand_atoms: list[int], rmsf: np.ndarray, rmsf_cutoff: float, limit: int):
    """Bonded triples (G0, G1, G2) in ligand-local indices, most central first (OpenFE's rule)."""
    from rdkit import Chem

    heavy = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() > 1]
    rings = [set(r) for r in rdmol.GetRingInfo().AtomRings()
             if all(rdmol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    pool: set[int] = set()
    for ring in rings:
        if max(rmsf[i] for i in ring) < rmsf_cutoff:
            pool |= ring
    ring_only = len(pool) >= 3
    if not ring_only:
        pool = {i for i in heavy if rmsf[i] < rmsf_cutoff}
    if len(pool) < 3:
        return []
    dmat = Chem.GetDistanceMatrix(rdmol)
    hm = dmat[np.ix_(heavy, heavy)]
    centre = heavy[int(np.argmin(hm.max(axis=1)))]
    conf = rdmol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(rdmol.GetNumAtoms())])
    order = sorted(pool, key=lambda i: float(np.linalg.norm(xyz[i] - xyz[centre])))
    # symmetry classes: prefer anchors that are not interchangeable with another atom
    ranks = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
    multiplicity = {i: ranks.count(ranks[i]) for i in heavy}
    triples = []
    for g0 in order:
        for nb1 in rdmol.GetAtomWithIdx(g0).GetNeighbors():
            g1 = nb1.GetIdx()
            if g1 not in pool:
                continue
            for nb2 in nb1.GetNeighbors():
                g2 = nb2.GetIdx()
                if g2 == g0 or g2 not in pool:
                    continue
                if ring_only and not any({g0, g1, g2} <= ring for ring in rings):
                    continue
                sym = multiplicity[g0] + multiplicity[g1] + multiplicity[g2] - 3
                triples.append((sym, len(triples), (g0, g1, g2)))
    triples.sort(key=lambda t: (t[0] > 0, t[1]))               # unique atoms first, then centrality
    return [t[2] for t in triples[:limit]]


#: how the note of earlier versions about a ligand that moved from its pose began (a verdict with a level of
#: cuFEP's own, restraints.pose_rmsd_warning_nm: no longer made, and removed from stored notes)
POSE_NOTE = "the ligand moved away from its"


def nearest_image_of_ligand(positions_nm: np.ndarray, ligand: np.ndarray, anchor: np.ndarray,
                            box_nm: np.ndarray) -> np.ndarray:
    """Copy of the coordinates with the ligand, as a whole, in the periodic image nearest to ``anchor``.

    ``positions_nm`` is (n_atoms, 3) or (n_frames, n_atoms, 3); ``anchor`` and ``box_nm`` (3 x 3
    box vectors, OpenMM's reduced form) are one per frame accordingly. The ligand may be any
    number of cells away: coordinates of a trajectory are not wrapped into the box.
    """
    from cufep.alchemy.lrc import minimum_image

    single = positions_nm.ndim == 2
    pos = np.array(positions_nm[None] if single else positions_nm, dtype=float)
    anchors = np.asarray(anchor, dtype=float).reshape(-1, 3)
    boxes = np.asarray(box_nm, dtype=float).reshape(-1, 3, 3)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    for f in range(pos.shape[0]):
        box = boxes[min(f, len(boxes) - 1)]
        delta = pos[f, ligand].mean(axis=0) - anchors[min(f, len(anchors) - 1)]
        # the lattice vector that brings the ligand next to the anchor, however many cells that is,
        # and its neighbours (in a skewed cell the nearest image need not be that one)
        candidates = minimum_image(delta[None], box)[0] - delta + shifts @ box
        best = candidates[np.argmin(np.linalg.norm(delta + candidates, axis=1))]
        pos[f, ligand] += best
    return pos[0] if single else pos


def _c_alpha_atoms(topology) -> np.ndarray:
    from cufep.prep.protein import AMINO_ACIDS

    return np.array([a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS and a.name == "CA"], dtype=int)


def binding_site_frames(topology, frames_nm: np.ndarray, ligand: np.ndarray, ligand_heavy: np.ndarray,
                        host_max_distance_nm: float, frame_boxes_nm: np.ndarray | None = None):
    """The frames in the frame of the binding site: ``(frames, aligned, site)``.

    With the box vectors of the frames, the ligand is first brought to the periodic image nearest
    to the protein (``frames``). Then every frame is superposed on the backbone atoms around the
    ligand (``site``, the C-alpha atoms if there are fewer than 12), so that domain motions elsewhere
    do not count as flexibility of the site (OpenFE's "local" RMSF): ``aligned``."""
    from cufep.prep.protein import AMINO_ACIDS

    ca = _c_alpha_atoms(topology)
    if frame_boxes_nm is not None:
        # the ligand in the periodic image nearest to the protein, whatever image it was written in
        frames_nm = nearest_image_of_ligand(frames_nm, ligand, frames_nm[:, ca, :].mean(axis=1), frame_boxes_nm)
    backbone_all = np.array([a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS
                             and a.name in ("N", "CA", "C")], dtype=int)
    centroid = frames_nm[0, ligand_heavy, :].mean(axis=0)
    near = backbone_all[np.linalg.norm(frames_nm[0, backbone_all, :] - centroid, axis=1) < host_max_distance_nm + 0.5]
    site = near if len(near) >= 12 else ca
    aligned = kabsch_align(frames_nm, frames_nm[0], site)
    aligned = kabsch_align(aligned, aligned.mean(axis=0), site)
    return frames_nm, aligned, site


def pose_rmsd_in_site(topology, aligned: np.ndarray, site: np.ndarray, ligand: np.ndarray, ligand_heavy: np.ndarray,
                      rdmol, input_pose_nm: np.ndarray | None = None, input_box_nm: np.ndarray | None = None) -> float:
    """The heavy-atom RMSD of the ligand from its input pose (from the first frame if none is given)
    over the second half of the frames ``aligned`` on the binding site (:func:`binding_site_frames`).

    The input is brought to the periodic image of its ligand nearest to its protein and superposed
    on the mean of the frames by the same atoms. Atoms that the symmetry of the molecule
    interchanges may swap places (OpenFE reports the same symmetry-corrected RMSD)."""
    mean = aligned.mean(axis=0)
    if input_pose_nm is None:
        reference = aligned[0, ligand_heavy, :]
    else:
        given = np.array(input_pose_nm, dtype=float)
        if input_box_nm is not None:
            given = nearest_image_of_ligand(given, ligand, given[_c_alpha_atoms(topology)].mean(axis=0), input_box_nm)
        reference = kabsch_align(given[None], mean, site)[0][ligand_heavy, :]
    rmsd = None
    for perm in heavy_atom_symmetries(rdmol):
        d = np.sqrt(((aligned[:, ligand_heavy[perm], :] - reference) ** 2).sum(axis=2).mean(axis=1))
        rmsd = d if rmsd is None else np.minimum(rmsd, d)
    return float(rmsd[len(rmsd) // 2:].mean())


def ligand_pose_rmsd(topology, frames_nm: np.ndarray, ligand_atoms, rdmol, input_pose_nm: np.ndarray,
                     frame_boxes_nm: np.ndarray | None = None, input_box_nm: np.ndarray | None = None,
                     host_max_distance_nm: float = RestraintSearchSettings.host_max_distance_nm) -> float:
    """How far the ligand of the frames is from its input pose (nm): the measure of the restraint
    search (:func:`binding_site_frames`, :func:`pose_rmsd_in_site`), for anchors that were not
    chosen by it (review of PR 1: with manual anchors a ligand that left the site gave no warning)."""
    lig = np.asarray(ligand_atoms, dtype=int)
    heavy = lig[[i for i, a in enumerate(rdmol.GetAtoms()) if a.GetAtomicNum() > 1]]
    _frames, aligned, site = binding_site_frames(topology, frames_nm, lig, heavy, host_max_distance_nm, frame_boxes_nm)
    return pose_rmsd_in_site(topology, aligned, site, lig, heavy, rdmol, input_pose_nm, input_box_nm)


def find_boresch_restraint(topology, frames_nm: np.ndarray, ligand_atoms: list[int], rdmol,
                           temperature_K: float = 298.15,
                           settings: RestraintSearchSettings | None = None,
                           input_pose_nm: np.ndarray | None = None, frame_boxes_nm: np.ndarray | None = None,
                           input_box_nm: np.ndarray | None = None, start_nm: np.ndarray | None = None,
                           start_box_nm: np.ndarray | None = None) -> RestraintSearchResult:
    """Choose Boresch restraint atoms from an MD trajectory of the bound complex.

    ``frames_nm`` has shape (n_frames, n_atoms, 3) with whole (unwrapped) molecules;
    ``rdmol`` is the ligand with atoms in the same order as ``ligand_atoms``.

    ``input_pose_nm`` are the coordinates of the prepared input, in the same atom order. With
    them the pose check asks how far the ligand is from the pose that was given, and not from
    the first frame of a trajectory that may begin after the ligand has moved. With the box
    vectors of the frames and of the input, the ligand is first brought to the periodic image
    nearest to the protein.

    ``start_nm`` (with its box ``start_box_nm``) are the positions the alchemical run starts from,
    of which the last frame is a copy: the dihedral angles of the chosen anchors must be defined
    there too (:func:`anchor_geometry_problem`).
    """
    s = settings or RestraintSearchSettings()
    kt = R_GAS * temperature_K
    atoms = list(topology.atoms())
    lig = np.asarray(ligand_atoms, dtype=int)
    from cufep.prep.protein import AMINO_ACIDS
    protein = np.array([a.index for a in atoms if a.residue.name in AMINO_ACIDS], dtype=int)
    ca = np.array([a.index for a in atoms if a.residue.name in AMINO_ACIDS and a.name == "CA"], dtype=int)
    if len(ca) < 4:
        raise RestraintError("too few protein C-alpha atoms for a restraint search")
    notes: list[str] = []

    heavy_local = [i for i, a in enumerate(rdmol.GetAtoms()) if a.GetAtomicNum() > 1]
    lig_heavy = lig[heavy_local]
    frames_nm, aligned, site = binding_site_frames(topology, frames_nm, lig, lig_heavy, s.host_max_distance_nm,
                                                   frame_boxes_nm)
    mean = aligned.mean(axis=0)
    rmsf = np.sqrt(((aligned - mean) ** 2).sum(axis=2).mean(axis=0))

    # Internal flexibility of the ligand: superpose the ligand on itself. A small ligand that
    # rattles or flips in its pocket is still rigid; whether its pose is stable relative to
    # the protein is judged separately, by the pose RMSD and by the restraint coordinates.
    lig_frames = frames_nm[:, lig, :]
    lig_self = kabsch_align(lig_frames, lig_frames[0], np.array(heavy_local))
    lig_self = kabsch_align(lig_self, lig_self.mean(axis=0), np.array(heavy_local))
    lig_rmsf = np.sqrt(((lig_self - lig_self.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))

    # How far the ligand moved from the input pose, in the frame of the binding site: a number of the
    # result, without a verdict (OpenFE reports the ligand RMSD without a limit)
    tail = pose_rmsd_in_site(topology, aligned, site, lig, lig_heavy, rdmol, input_pose_nm, input_box_nm)

    guests = _guest_anchor_candidates(rdmol, list(lig), lig_rmsf, s.rmsf_cutoff_nm, s.max_guest_anchors)
    if not guests:
        raise RestraintError("no suitable ligand atoms for the restraint (ligand too flexible or too small)")

    # host pool: backbone atoms, stable secondary structure, low RMSF
    backbone = [a.index for a in atoms if a.residue.name in AMINO_ACIDS and a.name in ("N", "CA", "C")]
    pool = np.array(backbone, dtype=int)
    if s.dssp_filter:
        stable = _stable_secondary_structure(topology, atoms, aligned, protein)
        filtered = np.array([i for i in backbone if atoms[i].residue.index in stable], dtype=int)
        if len(filtered) >= 20:
            pool = filtered
        else:
            notes.append("secondary-structure filter left too few atoms; using all backbone atoms")
    pool = pool[rmsf[pool] < s.rmsf_cutoff_nm]
    if len(pool) < 3:
        raise RestraintError("no protein backbone atoms with low enough RMSF near the ligand")
    pool_set = set(int(i) for i in pool)
    bonded: dict[int, set[int]] = {i: set() for i in pool_set}
    for b in topology.bonds():
        i, j = b[0].index, b[1].index
        if i in pool_set and j in pool_set:
            bonded[i].add(j); bonded[j].add(i)

    sig_r = math.sqrt(kt / s.k_r)
    sig_a = math.sqrt(kt / s.k_angle)
    flat_margin = anchor_margins(s, temperature_K)
    # the configuration the reference values come from, as the run has it: the last frame before it was
    # superposed on the binding site, and the positions the run starts from
    copies = [frames_nm[-1]]
    if start_nm is not None:
        start = np.asarray(start_nm, dtype=float)
        if start_box_nm is not None:
            start = nearest_image_of_ligand(start, lig, start[ca].mean(axis=0), np.asarray(start_box_nm, dtype=float))
        copies.append(start)
    best = None
    n_tested = 0
    for g_local in guests:
        g0, g1, g2 = (int(lig[i]) for i in g_local)
        d_g0 = np.linalg.norm(aligned[:, pool, :] - aligned[:, [g0], :], axis=2)      # (frames, pool)
        ok = (d_g0.min(axis=0) >= s.host_min_distance_nm) & (d_g0.mean(axis=0) <= s.host_max_distance_nm)
        cand = pool[ok][np.argsort(d_g0.mean(axis=0)[ok])][: s.max_host_candidates]
        for h0 in cand:
            h0 = int(h0)
            for h1 in bonded[h0]:
                for h2 in bonded[h1]:
                    if h2 == h0:
                        continue
                    n_tested += 1
                    r = BoreschRestraint((h0, h1, h2), (g0, g1, g2), 0, 0, 0, 0, 0, 0)
                    # OpenFE's rules: theta_A and theta_B away from 0 and pi, the anchors not on a line; and
                    # the dihedral angles defined where the run starts
                    if anchor_geometry_problem(r, aligned, flat_margin, reference_copies=copies) is not None:
                        continue
                    v = r.values(aligned)                                           # (frames, 6)
                    # Reference values: those of the final frame, from which the alchemical run
                    # starts (as OpenFE). A mean over the trajectory is a geometry that a ligand
                    # which moved between two positions never has.
                    reference = [float(x) for x in v[-1]]
                    stds = list(coordinate_fluctuations(v).values())
                    score = (stds[0] / sig_r) ** 2 + sum((x / sig_a) ** 2 for x in stds[1:])
                    if best is None or score < best[0]:
                        best = (score, (h0, h1, h2), (g0, g1, g2), reference, stds)
        if best is not None:
            break          # OpenFE: the first (most central) guest anchor with a valid host anchor
    if best is None:
        raise RestraintError("no host/guest atom combination gave a stable, non-collinear restraint geometry")
    score, host, guest, ref, stds = best
    restraint = BoreschRestraint(host, guest, ref[0], ref[1], ref[2], ref[3], ref[4], ref[5],
                                 s.k_r, s.k_angle, s.k_angle, s.k_angle, s.k_angle, s.k_angle)
    return RestraintSearchResult(restraint=restraint, score=float(score),
                                 fluctuations=dict(zip(COORDINATES, stds)), ligand_rmsd_nm=tail,
                                 n_candidates=n_tested, notes=notes)


def with_force_constants_from_fluctuations(restraint: BoreschRestraint, fluctuations: dict[str, float],
                                           temperature_K: float, limits: tuple[float, float] = (0.25, 4.0)
                                           ) -> BoreschRestraint:
    """The same restraint with force constants kT / sigma^2 from the observed fluctuations.

    A harmonic restraint with this force constant has the width the coordinate shows in the
    plain MD of the bound state, so it holds the ligand where it is without squeezing it. A3FE
    and SOMD2 derive Boresch force constants this way. Each
    value is kept within ``limits`` times the fixed default (10 kcal/mol/A^2, 80 kcal/mol/rad^2).
    """
    kt = R_GAS * temperature_K
    names = {"k_r": "r", "k_theta_a": "theta_A", "k_theta_b": "theta_B", "k_phi_a": "phi_A", "k_phi_b": "phi_B",
             "k_phi_c": "phi_C"}
    values = {}
    for field_name, label in names.items():
        default = K_R_DEFAULT if field_name == "k_r" else K_ANGLE_DEFAULT
        sigma = float(fluctuations[label])
        k = kt / (sigma * sigma) if sigma > 0 else default * limits[1]
        values[field_name] = float(min(default * limits[1], max(default * limits[0], k)))
    return dataclasses.replace(restraint, **values)


def ligand_symmetry_number(rdmol) -> int:
    """Number of heavy-atom graph automorphisms of the ligand (reported, not applied)."""
    from rdkit import Chem

    heavy = Chem.RemoveHs(rdmol)
    return max(1, len(heavy.GetSubstructMatches(heavy, uniquify=False, maxMatches=10000)))


def heavy_atom_symmetries(rdmol, limit: int = 512) -> list[np.ndarray]:
    """Permutations of the heavy atoms (in their own numbering) that map the molecule onto itself.

    The identity comes first. At most ``limit`` permutations are returned.
    """
    from rdkit import Chem

    n_heavy = sum(1 for a in rdmol.GetAtoms() if a.GetAtomicNum() > 1)
    identity = np.arange(n_heavy)
    heavy = Chem.RemoveHs(rdmol)
    if heavy.GetNumAtoms() != n_heavy:                 # hydrogens RDKit keeps; no correction then
        return [identity]
    perms = [np.asarray(m, dtype=int) for m in heavy.GetSubstructMatches(heavy, uniquify=False, maxMatches=limit)]
    return [identity] + [m for m in perms if not np.array_equal(m, identity)]
