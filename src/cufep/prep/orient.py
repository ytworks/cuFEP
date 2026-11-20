"""Orientation of a membrane protein for bilayer building.

Whether a system gets a membrane is the user's choice (``membrane.enabled`` in the
configuration). This module only answers the next question: given a protein that is to be
embedded, where is the membrane plane? The coordinates in a PDB file are arbitrary, so the
membrane normal and centre are estimated from the structure by three independent methods,
and the estimate is used only if at least two of them agree:

* a transfer-energy model (a simplified PPM-like water-to-membrane transfer energy),
* a hydrophobic-slab search (exposed hydrophobic residues inside, exposed polar outside),
* ``memembed`` from AmberTools, when it is installed.

``membrane.orientation: input`` skips all of this for structures that are already
oriented (membrane normal along z, membrane centre at z = 0, e.g. from the OPM database).

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
MAX_TILT_DIFFERENCE_DEG = 15.0
MAX_DEPTH_DIFFERENCE_NM = 0.5


class OrientationError(RuntimeError):
    """The membrane orientation could not be determined reliably."""


@dataclass
class Orientation:
    method: str
    normal: np.ndarray
    centre_nm: np.ndarray
    thickness_nm: float
    score: float = 0.0

    def difference(self, other: "Orientation") -> tuple[float, float]:
        """(tilt in degrees ignoring the sense of the normal, depth offset in nm)."""
        a = self.normal / np.linalg.norm(self.normal)
        b = other.normal / np.linalg.norm(other.normal)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))
        return tilt, float(abs((self.centre_nm - other.centre_nm) @ b))


def _sphere_points(n: int) -> np.ndarray:
    """Roughly uniform directions on a hemisphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n                         # 1 .. 0
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


# ============================================================================ slab search
def _exposed_residues(topology, positions_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C-alpha positions, hydrophobic flags and solvent-exposed flags of the protein residues."""
    import mdtraj as md

    from cufep.prep.protein import AMINO_ACIDS

    idx = np.array([a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS], dtype=int)
    sub = md.Topology.from_openmm(topology).subset(idx)
    traj = md.Trajectory(positions_nm[idx][None, :, :], sub)
    sasa = md.shrake_rupley(traj, mode="residue")[0] * 100.0                 # A^2
    ca, names = [], []
    for res in sub.residues:
        a = next((x for x in res.atoms if x.name == "CA"), None)
        ca.append(traj.xyz[0, a.index] if a is not None else np.full(3, np.nan))
        names.append(_CANON.get(res.name, res.name))
    ca = np.asarray(ca, dtype=float)
    rel = np.array([s / MAX_SASA.get(n, 200.0) for s, n in zip(sasa, names)])
    ok = np.isfinite(ca[:, 0])
    return ca[ok], np.array([n in HYDROPHOBIC for n in names])[ok], (rel > EXPOSED_FRACTION)[ok]


def slab_orientation(topology, positions_nm: np.ndarray, n_directions: int = 1200) -> Orientation:
    """Slab that best separates exposed hydrophobic residues (inside) from exposed polar ones.

    Buried residues are ignored: protein cores are hydrophobic everywhere. The score is the
    Matthews correlation between "inside the slab" and "hydrophobic".
    """
    ca, hydrophobic, exposed = _exposed_residues(topology, np.asarray(positions_nm, dtype=float))
    centre = ca.mean(axis=0)
    x = ca - centre
    pho = exposed & hydrophobic
    phi = exposed & ~hydrophobic
    n_pho, n_phi = int(pho.sum()), int(phi.sum())
    if n_pho < 5 or n_phi < 5:
        raise OrientationError("too few solvent-exposed residues for the hydrophobic-slab search")
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
    return Orientation("slab", normal, centre + c * normal, d, score=score)


# ============================================================================ transfer energy
#: atomic solvation parameters for water -> hydrocarbon transfer, kcal/mol/A^2
_SIGMA = {"C": -0.022, "S": -0.010, "N": 0.053, "O": 0.057}
_CHARGED = {("LYS", "NZ"): 0.25, ("ARG", "NH1"): 0.25, ("ARG", "NH2"): 0.25, ("ARG", "NE"): 0.15,
            ("ASP", "OD1"): 0.25, ("ASP", "OD2"): 0.25, ("GLU", "OE1"): 0.25, ("GLU", "OE2"): 0.25}
_INTERFACE_NM = 0.09


def _transfer_weights(topology, positions_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mdtraj as md

    from cufep.prep.protein import AMINO_ACIDS

    atoms = [a for a in topology.atoms()
             if a.residue.name in AMINO_ACIDS and a.element is not None and a.element.symbol != "H"]
    idx = np.array([a.index for a in atoms], dtype=int)
    sub = md.Topology.from_openmm(topology).subset(idx)
    asa = md.shrake_rupley(md.Trajectory(positions_nm[idx][None], sub), mode="atom")[0] * 100.0
    sig = np.array([_CHARGED.get((_CANON.get(a.residue.name, a.residue.name), a.name),
                                 _SIGMA.get(a.element.symbol, 0.0)) for a in atoms])
    w = asa * sig
    keep = np.abs(w) > 1e-6
    return positions_nm[idx][keep], w[keep]


def _transfer_search(xyz, w, normals, half_grid, step):
    origin = xyz.mean(axis=0)
    x = xyz - origin
    best = (np.inf, None, 0.0, 0.0)
    for n in normals:
        z = x @ n
        centres = np.arange(z.min() + 0.5, z.max() - 0.5 + 1e-9, step)
        if len(centres) == 0:
            continue
        dz = np.abs(z[None, :] - centres[:, None])
        for h in half_grid:
            e = (w[None, :] / (1.0 + np.exp((dz - h) / _INTERFACE_NM))).sum(axis=1)
            k = int(np.argmin(e))
            if e[k] < best[0]:
                best = (float(e[k]), n, float(centres[k]), float(h))
    e, n, c, h = best
    if n is None:
        raise OrientationError("the protein is too small for the transfer-energy search")
    return e, n, origin + c * n, 2.0 * h


def transfer_energy_orientation(topology, positions_nm: np.ndarray) -> Orientation:
    """Slab that minimises an implicit water-to-membrane transfer energy (a simplified PPM-like model):
    exposed carbon and sulphur are favourable inside the slab, exposed nitrogen and oxygen cost."""
    xyz, w = _transfer_weights(topology, np.asarray(positions_nm, dtype=float))
    e, n, c, d = _transfer_search(xyz, w, _sphere_points(900), (1.0, 1.2, 1.4, 1.6, 1.8), 0.1)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400, 3)) * np.tan(np.radians(12.0)) / 2.0
    cand = n[None, :] + v - (v @ n)[:, None] * n[None, :]
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    e, n, c, d = _transfer_search(xyz, w, np.vstack([n[None, :], cand]), np.arange(1.0, 1.85, 0.1), 0.05)
    return Orientation("transfer", n, c, d, score=e)


# ============================================================================ memembed
def memembed_orientation(topology, positions_nm: np.ndarray, timeout: float = 900.0) -> Orientation | None:
    """Orientation from the ``memembed`` program (AmberTools), or None if it is unavailable or fails."""
    import shutil
    import subprocess
    import tempfile

    import openmm.app as app
    import openmm.unit as unit

    from cufep.prep.protein import AMINO_ACIDS

    exe = shutil.which("memembed")
    if exe is None:
        return None
    positions_nm = np.asarray(positions_nm, dtype=float)
    top = app.Topology()
    chain = top.addChain("A")
    pos, ca_in = [], []
    n_res = 0
    for res in topology.residues():
        if res.name not in AMINO_ACIDS:
            continue
        n_res += 1
        new = top.addResidue(_CANON.get(res.name, res.name), chain, id=str(n_res))
        for a in res.atoms():
            if a.element is None or a.element.symbol == "H":
                continue
            top.addAtom(a.name, a.element, new)
            pos.append(positions_nm[a.index])
            if a.name == "CA":
                ca_in.append(positions_nm[a.index])
    with tempfile.TemporaryDirectory(prefix="cufep-memembed-") as tmp:
        src, dst = f"{tmp}/in.pdb", f"{tmp}/out.pdb"
        with open(src, "w") as fh:
            app.PDBFile.writeFile(top, np.asarray(pos) * unit.nanometer, fh, keepIds=True)
        try:
            subprocess.run([exe, "-s", "3", "-o", dst, src], capture_output=True, text=True, timeout=timeout,
                           check=True, cwd=tmp)
            ca_out = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(dst)
                               if l.startswith("ATOM") and l[12:16].strip() == "CA"]) / 10.0
        except Exception:  # noqa: BLE001 - memembed is optional
            return None
    ca_in = np.asarray(ca_in)
    if ca_out.shape != ca_in.shape or len(ca_in) < 4:
        return None
    # memembed writes the protein with the membrane normal along z and the membrane centre at the
    # origin; recover that frame in input coordinates from the superposition out -> in.
    co, ci = ca_out.mean(axis=0), ca_in.mean(axis=0)
    u_, _s, vt = np.linalg.svd((ca_out - co).T @ (ca_in - ci))
    rot = u_ @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u_ @ vt))]) @ vt
    return Orientation("memembed", np.array([0.0, 0.0, 1.0]) @ rot, ci - co @ rot, 0.0)


# ============================================================================ consensus
def consensus_orientation(topology, positions_nm: np.ndarray, use_memembed: bool = True, log=print) -> Orientation:
    """Membrane orientation supported by at least two independent methods.

    Two orientations agree if their normals differ by less than 15 degrees and their centres
    by less than 5 A along the normal. The transfer-energy result is preferred when it is part
    of an agreeing pair. Raises :class:`OrientationError` when no two methods agree.
    """
    positions_nm = np.asarray(positions_nm, dtype=float)
    found: list[Orientation] = []
    for method in (transfer_energy_orientation, slab_orientation):
        try:
            found.append(method(topology, positions_nm))
        except OrientationError as exc:
            log(f"orientation: {method.__name__} failed ({exc})")
    if use_memembed:
        m = memembed_orientation(topology, positions_nm)
        if m is not None:
            found.append(m)
    agreeing: list[tuple[int, int]] = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            tilt, depth = found[i].difference(found[j])
            ok = tilt < MAX_TILT_DIFFERENCE_DEG and depth < MAX_DEPTH_DIFFERENCE_NM
            log(f"orientation: {found[i].method} vs {found[j].method}: tilt {tilt:.1f} deg, depth "
                f"{10 * depth:.1f} A {'(agree)' if ok else '(differ)'}")
            if ok:
                agreeing.append((i, j))
    if not agreeing:
        raise OrientationError(
            "the membrane orientation methods (" + ", ".join(o.method for o in found) + ") do not agree. "
            "Supply a structure that is already oriented (membrane normal along z, membrane centre at z = 0, "
            "e.g. from the OPM database) and set `membrane.orientation: input`, or pick one method with "
            "`membrane.orientation: transfer | slab | memembed`.")
    members = {k for pair in agreeing for k in pair}
    transfer = next((k for k in members if found[k].method == "transfer"), None)
    chosen = found[transfer if transfer is not None else min(members)]
    if chosen.thickness_nm <= 0:
        chosen.thickness_nm = next((o.thickness_nm for o in found if o.thickness_nm > 0), 3.0)
    return chosen


def orientation_matrix(normal: np.ndarray) -> np.ndarray:
    """Rotation that maps ``normal`` onto +z (rows are the new x, y, z axes)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, helper)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return np.stack([a, b, n])


def orient_positions(positions_nm: np.ndarray, orientation: Orientation) -> np.ndarray:
    """Coordinates with the membrane normal along z and the membrane centre at the origin."""
    rot = orientation_matrix(orientation.normal)
    return (np.asarray(positions_nm, dtype=float) - orientation.centre_nm) @ rot.T
