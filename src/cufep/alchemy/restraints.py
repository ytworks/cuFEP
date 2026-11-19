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
    flat_angle_energy_kt: float = 10.0     # a restrained angle must be this far from 0 / pi
    min_equilibrium_angle_deg: float = 30.0
    max_guest_anchors: int = 8
    max_host_candidates: int = 60
    pose_rmsd_warning_nm: float = 0.3


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


#: how the note about a ligand that left its pose begins
POSE_NOTE = "the ligand moved away from its"


def pose_note(rmsd_nm: float, warning_nm: float, reference: str = "input") -> str | None:
    """The note for a ligand whose heavy atoms are further than ``warning_nm`` from the pose it is
    compared with (``reference``: "input" or "starting"), or None."""
    if rmsd_nm <= warning_nm:
        return None
    return (f"{POSE_NOTE} {reference} pose in the pre-equilibration MD (heavy-atom RMSD {10 * rmsd_nm:.1f} A over the "
            f"second half, above restraints.pose_rmsd_warning_nm = {10 * warning_nm:g} A); the result is for the pose "
            "it moved to")


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


def find_boresch_restraint(topology, frames_nm: np.ndarray, ligand_atoms: list[int], rdmol,
                           temperature_K: float = 298.15,
                           settings: RestraintSearchSettings | None = None,
                           input_pose_nm: np.ndarray | None = None, frame_boxes_nm: np.ndarray | None = None,
                           input_box_nm: np.ndarray | None = None) -> RestraintSearchResult:
    """Choose Boresch restraint atoms from an MD trajectory of the bound complex.

    ``frames_nm`` has shape (n_frames, n_atoms, 3) with whole (unwrapped) molecules;
    ``rdmol`` is the ligand with atoms in the same order as ``ligand_atoms``.

    ``input_pose_nm`` are the coordinates of the prepared input, in the same atom order. With
    them the pose check asks how far the ligand is from the pose that was given, and not from
    the first frame of a trajectory that may begin after the ligand has moved. With the box
    vectors of the frames and of the input, the ligand is first brought to the periodic image
    nearest to the protein.
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
    if frame_boxes_nm is not None:
        # the ligand in the periodic image nearest to the protein, whatever image it was written in
        frames_nm = nearest_image_of_ligand(frames_nm, lig, frames_nm[:, ca, :].mean(axis=1), frame_boxes_nm)

    # Frame of the binding site: superpose on the backbone atoms around the ligand, so that
    # domain motions elsewhere do not count as flexibility of the site (OpenFE's "local" RMSF).
    backbone_all = np.array([a.index for a in atoms if a.residue.name in AMINO_ACIDS and a.name in ("N", "CA", "C")],
                            dtype=int)
    centroid = frames_nm[0, lig_heavy, :].mean(axis=0)
    near = backbone_all[np.linalg.norm(frames_nm[0, backbone_all, :] - centroid, axis=1) < s.host_max_distance_nm + 0.5]
    site = near if len(near) >= 12 else ca
    aligned = kabsch_align(frames_nm, frames_nm[0], site)
    aligned = kabsch_align(aligned, aligned.mean(axis=0), site)
    mean = aligned.mean(axis=0)
    rmsf = np.sqrt(((aligned - mean) ** 2).sum(axis=2).mean(axis=0))

    # Internal flexibility of the ligand: superpose the ligand on itself. A small ligand that
    # rattles or flips in its pocket is still rigid; whether its pose is stable relative to
    # the protein is judged separately, by the pose RMSD and by the restraint coordinates.
    lig_frames = frames_nm[:, lig, :]
    lig_self = kabsch_align(lig_frames, lig_frames[0], np.array(heavy_local))
    lig_self = kabsch_align(lig_self, lig_self.mean(axis=0), np.array(heavy_local))
    lig_rmsf = np.sqrt(((lig_self - lig_self.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))

    # How far the ligand moved from the input pose, in the frame of the binding site. Atoms that
    # the symmetry of the molecule interchanges may swap places (OpenFE reports the same
    # symmetry-corrected RMSD). The run goes on in any case, as in OpenFE; a large value is
    # reported with the result.
    if input_pose_nm is None:
        reference = aligned[0, lig_heavy, :]         # no input structure given: the first frame
    else:
        given = np.array(input_pose_nm, dtype=float)
        if input_box_nm is not None:
            given = nearest_image_of_ligand(given, lig, given[ca].mean(axis=0), input_box_nm)
        reference = kabsch_align(given[None], mean, site)[0][lig_heavy, :]
    rmsd = None
    for perm in heavy_atom_symmetries(rdmol):
        d = np.sqrt(((aligned[:, lig_heavy[perm], :] - reference) ** 2).sum(axis=2).mean(axis=1))
        rmsd = d if rmsd is None else np.minimum(rmsd, d)
    tail = float(rmsd[len(rmsd) // 2:].mean())
    note = pose_note(tail, s.pose_rmsd_warning_nm, "input" if input_pose_nm is not None else "starting")
    if note:
        notes.append(note)

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
    flat_margin = math.sqrt(2.0 * s.flat_angle_energy_kt * kt / s.k_angle)
    eq_margin = math.radians(s.min_equilibrium_angle_deg)
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
                    v = r.values(aligned)                                           # (frames, 6)
                    ang = v[:, 1:3]
                    if ang.min() < flat_margin or ang.max() > math.pi - flat_margin:
                        continue
                    # Reference values: those of the final frame, from which the alchemical run
                    # starts (as OpenFE). A mean over the trajectory is a geometry that a ligand
                    # which moved between two positions never has.
                    reference = [float(x) for x in v[-1]]
                    if not all(eq_margin <= m <= math.pi - eq_margin for m in reference[1:3]):
                        continue
                    stds = [float(v[:, 0].std()), float(ang[:, 0].std()), float(ang[:, 1].std()),
                            circular_std(v[:, 3]), circular_std(v[:, 4]), circular_std(v[:, 5])]
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
    labels = ["r", "theta_A", "theta_B", "phi_A", "phi_B", "phi_C"]
    return RestraintSearchResult(restraint=restraint, score=float(score),
                                 fluctuations=dict(zip(labels, stds)), ligand_rmsd_nm=tail,
                                 n_candidates=n_tested, notes=notes)


def with_force_constants_from_fluctuations(restraint: BoreschRestraint, fluctuations: dict[str, float],
                                           temperature_K: float, limits: tuple[float, float] = (0.25, 4.0)
                                           ) -> BoreschRestraint:
    """The same restraint with force constants kT / sigma^2 from the observed fluctuations.

    A harmonic restraint with this force constant has the width the coordinate shows in the
    plain MD of the bound state, so it holds the ligand where it is without squeezing it. A3FE
    and SOMD2 derive Boresch force constants this way (Flare: "constraint strengths"). Each
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
