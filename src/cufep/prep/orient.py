"""Membrane protein detection and orientation from a bare structure.

A hydrophobic-slab search in the spirit of TMDET / ANVIL: among all slab orientations,
positions and thicknesses, find the one that best separates solvent-exposed hydrophobic
residues (inside) from exposed polar residues (outside), and count the helices or strands
that cross it. A protein is called a membrane protein only if a good slab exists *and*
secondary-structure elements span it. Buried residues are ignored: protein cores are
hydrophobic in soluble proteins too.

The up/down sense of the normal is arbitrary; cuFEP builds symmetric bilayers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HYDROPHOBIC = {"PHE", "GLY", "ILE", "LEU", "MET", "VAL", "TRP", "TYR", "ALA", "CYS"}
#: theoretical maximum side-chain accessibility (Tien et al. 2013), A^2
MAX_SASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
            "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
            "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
_CANON = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
          "LYN": "LYS"}
EXPOSED_FRACTION = 0.20
THICKNESSES_NM = (2.4, 2.8, 3.2, 3.6)
SCORE_MEMBRANE = 0.25
SCORE_SOLUBLE = 0.12


@dataclass
class MembraneVerdict:
    is_membrane: bool
    confident: bool
    reason: str
    score: float = 0.0
    n_crossing_helices: int = 0
    n_crossing_strands: int = 0
    normal: np.ndarray | None = None
    centre_nm: np.ndarray | None = None
    thickness_nm: float = 0.0


def _sphere_points(n: int) -> np.ndarray:
    """Roughly uniform directions on a hemisphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n                         # 1 .. 0
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _residue_table(topology, positions_nm: np.ndarray):
    """Per-residue arrays for protein residues: CA position, hydrophobic flag, exposed flag, DSSP."""
    import mdtraj as md

    from cufep.prep.protein import AMINO_ACIDS

    atoms = [a for a in topology.atoms() if a.residue.name in AMINO_ACIDS]
    idx = np.array([a.index for a in atoms], dtype=int)
    sub = md.Topology.from_openmm(topology).subset(idx)
    traj = md.Trajectory(positions_nm[idx][None, :, :], sub)
    sasa = md.shrake_rupley(traj, mode="residue")[0] * 100.0                 # A^2
    dssp = md.compute_dssp(traj, simplified=True)[0]
    ca, names, chain = [], [], []
    for res in sub.residues:
        a = next((x for x in res.atoms if x.name == "CA"), None)
        ca.append(traj.xyz[0, a.index] if a is not None else np.full(3, np.nan))
        names.append(_CANON.get(res.name, res.name))
        chain.append(res.chain.index)
    ca = np.asarray(ca, dtype=float)
    rel = np.array([s / MAX_SASA.get(n, 200.0) for s, n in zip(sasa, names)])
    ok = np.isfinite(ca[:, 0])
    return (ca[ok], np.array([n in HYDROPHOBIC for n in names])[ok], (rel > EXPOSED_FRACTION)[ok],
            np.asarray(dssp)[ok], np.asarray(chain)[ok])


def _segments(dssp: np.ndarray, chain: np.ndarray, code: str, min_len: int) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(dssp)
    while i < n:
        if dssp[i] != code:
            i += 1
            continue
        j = i
        while j < n and dssp[j] == code and chain[j] == chain[i]:
            j += 1
        if j - i >= min_len:
            out.append((i, j - 1))
        i = j
    return out


def detect_membrane(topology, positions_nm: np.ndarray, n_directions: int = 1200) -> MembraneVerdict:
    """Decide whether the protein is a transmembrane protein and find its slab."""
    ca, hydrophobic, exposed, dssp, chain = _residue_table(topology, np.asarray(positions_nm, dtype=float))
    n_res = len(ca)
    if n_res < 40:
        return MembraneVerdict(False, True, f"only {n_res} residues: treated as soluble")
    helices = _segments(dssp, chain, "H", 14)
    strands = _segments(dssp, chain, "E", 6)
    if not helices and len(strands) < 8:
        return MembraneVerdict(False, True, "no helix or beta-barrel long enough to span a bilayer")

    centre = ca.mean(axis=0)
    x = ca - centre
    pho = exposed & hydrophobic
    phi = exposed & ~hydrophobic
    n_pho, n_phi = int(pho.sum()), int(phi.sum())
    if n_pho < 5 or n_phi < 5:
        return MembraneVerdict(False, False, "too few exposed residues to judge")
    best = (-2.0, None, 0.0, 0.0)
    for normal in _sphere_points(n_directions):
        z = x @ normal
        zp, zh = np.sort(z[pho]), np.sort(z[phi])
        centres = np.arange(z.min(), z.max() + 1e-9, 0.1)
        for d in THICKNESSES_NM:
            lo, hi = centres - 0.5 * d, centres + 0.5 * d
            tp = np.searchsorted(zp, hi) - np.searchsorted(zp, lo)          # hydrophobic inside
            fp = np.searchsorted(zh, hi) - np.searchsorted(zh, lo)          # polar inside
            fn, tn = n_pho - tp, n_phi - fp
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)
            mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1e-9), -1.0)
            k = int(np.argmax(mcc))
            if mcc[k] > best[0]:
                best = (float(mcc[k]), normal, float(centres[k]), d)
    score, normal, c, d = best
    z = x @ normal

    def crossing(segs) -> int:
        count = 0
        for a, b in segs:
            za, zb = z[a], z[b]
            lo, hi = min(za, zb), max(za, zb)
            if lo < c - 0.35 * d and hi > c + 0.35 * d:
                count += 1
        return count

    n_h, n_s = crossing(helices), crossing(strands)
    slab_centre = centre + c * normal
    kw = dict(score=score, n_crossing_helices=n_h, n_crossing_strands=n_s, normal=normal,
              centre_nm=slab_centre, thickness_nm=d)
    summary = (f"hydrophobic slab score {score:.2f}, {n_h} helices and {n_s} strands crossing a "
               f"{10 * d:.0f} A slab")
    spanning = n_h + (n_s if n_s >= 8 else 0)
    if spanning == 0 or score < SCORE_SOLUBLE:
        return MembraneVerdict(False, True, f"soluble: {summary}", **kw)
    if (n_h >= 3 or n_s >= 8) and score >= SCORE_MEMBRANE:
        return MembraneVerdict(True, True, f"membrane protein: {summary}", **kw)
    return MembraneVerdict(spanning >= 2 and score >= SCORE_MEMBRANE, False, f"ambiguous: {summary}", **kw)


def orientation_matrix(normal: np.ndarray) -> np.ndarray:
    """Rotation that maps ``normal`` onto +z (rows are the new x, y, z axes)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, helper)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return np.stack([a, b, n])


def orient_positions(positions_nm: np.ndarray, verdict: MembraneVerdict) -> np.ndarray:
    """Coordinates with the membrane normal along z and the slab centre at the origin."""
    rot = orientation_matrix(verdict.normal)
    return (np.asarray(positions_nm, dtype=float) - verdict.centre_nm) @ rot.T


def tilt_and_shift(verdict: MembraneVerdict, ref_normal: np.ndarray, ref_centre_nm: np.ndarray) -> tuple[float, float]:
    """Angle (deg) between normals, ignoring sense, and distance (nm) of the slab centre
    from the reference membrane plane."""
    n = verdict.normal / np.linalg.norm(verdict.normal)
    r = np.asarray(ref_normal, dtype=float) / np.linalg.norm(ref_normal)
    angle = float(np.degrees(np.arccos(min(1.0, abs(float(n @ r))))))
    shift = float(abs((verdict.centre_nm - np.asarray(ref_centre_nm, dtype=float)) @ r))
    return angle, shift
