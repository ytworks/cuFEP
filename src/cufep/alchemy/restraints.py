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

import dataclasses
import math
from dataclasses import asdict, dataclass

import numpy as np
import openmm as mm

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
                z *= integrate.quad(lambda t, k=k, t0=t0: math.sin(t) * math.exp(-0.5 * beta * k * (t - t0) ** 2),
                                    0.0, math.pi, points=[t0], limit=200)[0]
            for k in (self.k_phi_a, self.k_phi_b, self.k_phi_c):
                z *= integrate.quad(lambda p, k=k: math.exp(-0.5 * beta * k * p * p), -math.pi, math.pi,
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
    #: restraints.force_constants: auto: a candidate is also held to that rule with the force constants
    #: its fluctuations give it (:func:`constants_from_fluctuations`)
    constants_from_fluctuations: bool = False
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


def flat_margin_of(k_angle: float, temperature_K: float, energy_kt: float = 10.0) -> float:
    """The angle (radians) from 0 and pi at which a restrained angle of force constant ``k_angle``
    (kJ/mol/rad^2) has the energy ``energy_kt`` kT."""
    return math.sqrt(2.0 * float(energy_kt) * R_GAS * float(temperature_K) / float(k_angle))


def flat_angle_problem(theta: np.ndarray, margins) -> str | None:
    """Why theta_A and theta_B (``theta``: configurations x 2, radians) break OpenFE's
    check_angle_not_flat, or None: each must stay its margin away from 0 and pi in every
    configuration. ``margins`` is one angle for both or one for each: the margin is that of the
    force constant the angle is restrained with (:func:`flat_margin_of`)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    both = (float(margins),) * 2 if np.ndim(margins) == 0 else tuple(float(m) for m in margins)
    for k, name in enumerate(ANCHOR_ANGLES[1:3]):
        low, high = float(theta[:, k].min()), float(theta[:, k].max())
        if low < both[k] or high > np.pi - both[k]:
            return (f"the anchor angle {name} comes within {np.degrees(both[k]):.0f} degrees of a straight line "
                    f"({np.degrees(low):.1f} to {np.degrees(high):.1f} degrees over the frames), where the restraint "
                    "energy is below 10 kT (OpenFE's check_angle_not_flat)")
    return None


def anchor_geometry_problem(restraint: "BoreschRestraint", frames: np.ndarray, flat_margin,
                            trajectory: bool = True, reference_copies=(),
                            last_frame_is_reference: bool = True) -> str | None:
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
    # flat_margin None: the rule of the flat angle is asked by the caller, with the force constants and the
    # reference angles the restraint gets (flat_rule_problem)
    flat = None if flat_margin is None else \
        flat_angle_problem(np.atleast_2d(anchor_angles(restraint, frames))[:, 1:3], flat_margin)
    if flat or not trajectory:
        return flat
    if bool(np.all(collinear_frames(restraint, frames))):
        return ("the six anchors H2-H1-H0-G0-G1-G2 are on a straight line in every frame (three in a row with "
                f"|cos| above {COLLINEAR_THRESHOLD:g}, OpenFE's is_collinear): the dihedral angles of the restraint "
                "are not defined there")
    # the last frame where the reference values are its values (the search, given anchors); for reference values
    # from histograms no frame is one, and which frame comes last decides nothing
    for copy in ((frames[-1], *reference_copies) if last_frame_is_reference else tuple(reference_copies)):
        flat = undefined_dihedral_planes(restraint, copy)
        if flat:
            return (f"the anchors {', '.join(flat)} are exactly on a line in the last frame, from which the reference "
                    "values come: the dihedral angles of the restraint are not defined there")
    return None


@dataclass
class AnchorMeasurement:
    """What the frames say about a set of anchors."""

    problem: str | None                    # why the anchors are refused (anchor_geometry_problem), or None
    values: np.ndarray                     # frames x 6: the six coordinates in every frame
    start_values: np.ndarray | None        # the six coordinates where the alchemical run starts
    #: theta_A and theta_B of every configuration that is held to OpenFE's check_angle_not_flat: the frames,
    #: and the start where it is held to the rules of the frames
    held_theta: np.ndarray | None = None


def measure_anchors(host_atoms, guest_atoms, ligand_atoms, frames, frame_atoms, frame_boxes,
                    flat_margin: float | None, start=None, start_box=None,
                    start_among_frames: bool = False) -> AnchorMeasurement:
    """The six coordinates of the anchors over ``frames`` and whether the anchors can be used: one
    measurement for anchors that are given (``restraints.selection: manual``) and for anchors
    chosen from interactions (``interactions``).

    ``frames`` (frames x frame atoms x 3, nm) are the positions of the atoms ``frame_atoms`` (their
    numbers in the System), whole molecules that are not wrapped into the box. The force of the
    restraint measures across the periodic boundary, so the six anchors of every frame are laid
    out as the force sees them (:func:`as_the_force_measures`; from the raw coordinates a ligand
    one cell away gave r0 = 5.4 nm in a 5 nm cell, review of PR 1; with the ligand in the image
    nearest to the centre of the host anchors, H0 at 0 and G0 at 2.1 nm in a cell of 4 nm gave
    r = 2.1 nm where the force measures 1.9 nm, review of eb6f89f). ``ligand_atoms`` is not read
    any more. ``flat_margin`` None leaves OpenFE's rule of the flat angle to the caller, who asks
    it with the force constants and the reference angles of the restraint
    (:func:`flat_rule_problem` on ``held_theta``).

    ``start`` (the positions of the whole System, with ``start_box``) is the configuration the
    alchemical run starts from. The dihedral angles of the anchors must be defined there
    (:func:`anchor_geometry_problem`). With ``start_among_frames`` it is also held to the rule of
    every frame (theta_A and theta_B away from a straight line): for reference values that are not
    those of the last frame, so that no frame need be the start. The rule of a trajectory (not on
    a line in every frame) is one of the frames alone: a start that is bent does not make anchors
    usable that are on a line in every frame the reference values come from (review of eb6f89f).
    """
    frame_atoms = np.asarray(frame_atoms, dtype=int)
    host, guest = [int(i) for i in host_atoms], [int(i) for i in guest_atoms]
    row = {int(a): i for i, a in enumerate(frame_atoms)}
    chain = [*reversed(host), *guest]                         # H2, H1, H0, G0, G1, G2: the particles of the force
    xyz = as_the_force_measures(np.asarray(frames)[:, [row[a] for a in chain], :], frame_boxes)
    probe = BoreschRestraint((2, 1, 0), (3, 4, 5), 0, 0, 0, 0, 0, 0)
    copies, begin = [], None
    if start is not None:
        begin = np.asarray(start, dtype=float)[chain]
        if start_box is not None:
            begin = as_the_force_measures(begin[None], np.asarray(start_box, dtype=float)[None])[0]
        copies.append(begin)
    problem = anchor_geometry_problem(probe, xyz, flat_margin, reference_copies=copies,
                                      last_frame_is_reference=not start_among_frames)
    held = xyz
    if start_among_frames and begin is not None:
        held = np.concatenate([xyz, begin[None]])
        if flat_margin is not None:
            problem = problem or anchor_geometry_problem(probe, begin, flat_margin, trajectory=False)
    return AnchorMeasurement(problem, probe.values(xyz), None if begin is None else probe.values(begin),
                             np.atleast_2d(anchor_angles(probe, held))[:, 1:3])


def as_the_force_measures(chain_nm: np.ndarray, boxes_nm) -> np.ndarray:
    """The six anchors H2, H1, H0, G0, G1, G2 (``chain_nm``: frames x 6 x 3) laid out as the force
    of the restraint sees them: each from the one before it by the periodic displacement OpenMM
    computes (:func:`cufep.alchemy.lrc.minimum_image`; the force is periodic). The distance, the
    two angles and the three dihedral angles of the result are the ones the force restrains, in
    any cell and whatever images the atoms are written in.

    That is not the nearest image where a displacement is longer than half the smallest height of
    a skewed cell: H0 to G0 of (0, 0, 1.5) nm in the rhombic dodecahedron of 4 nm is 3.125 nm for
    the force, and a reference of 1.5 nm would cost 5523 kJ/mol there; and anchors of one molecule
    need not be bonded neighbours: H0 to H1 of 2.5 nm in a cube of 4 nm is -1.5 nm for the force,
    and theta_A is 120 degrees where the written coordinates give 60 (review of 877ab88)."""
    from cufep.alchemy.lrc import minimum_image

    written = np.asarray(chain_nm, dtype=float)
    boxes = np.asarray(boxes_nm, dtype=float).reshape(-1, 3, 3)
    out = np.empty_like(written)
    out[:, 0] = written[:, 0]
    for f in range(written.shape[0]):
        steps = minimum_image(written[f, 1:] - written[f, :-1], boxes[min(f, len(boxes) - 1)])
        out[f, 1:] = written[f, 0] + np.cumsum(steps, axis=0)
    return out


def flat_rule_problem(k_theta_a: float, k_theta_b: float, theta: np.ndarray, temperature_K: float,
                      flat_angle_energy_kt: float = 10.0) -> str | None:
    """OpenFE's check_angle_not_flat for a restraint with these force constants (kJ/mol/rad^2) on
    theta_A and theta_B: ``theta`` (configurations x 2, radians) holds every angle that must be
    10 kT from 0 and pi: those of the frames, of the configuration the run starts from and the
    reference angles of the restraint itself."""
    margins = (flat_margin_of(k_theta_a, temperature_K, flat_angle_energy_kt),
               flat_margin_of(k_theta_b, temperature_K, flat_angle_energy_kt))
    return flat_angle_problem(theta, margins)


def constants_from_fluctuations(restraint: "BoreschRestraint", values: np.ndarray, held_theta: np.ndarray,
                                temperature_K: float, flat_angle_energy_kt: float = 10.0):
    """``restraints.force_constants: auto`` for anchors that passed the rules with the configured
    force constants: ``(the restraint with force constants from the fluctuations of values, why
    it cannot be used with them or None)``. OpenFE's check_angle_not_flat is a rule of the force
    constant the angle has: a weaker one puts 10 kT farther from a straight line (44 degrees at
    20 kcal/mol/rad^2 where 80 gives 22), so theta_A and theta_B (``held_theta``, of every
    configuration held to the rule) are asked again with the constants the restraint gets
    (review of eb6f89f: a start at 30 degrees passed, 4.6 kT from a straight line)."""
    out = with_force_constants_from_fluctuations(restraint, coordinate_fluctuations(values), temperature_K)
    return out, flat_rule_problem(out.k_theta_a, out.k_theta_b, held_theta, temperature_K, flat_angle_energy_kt)


def reference_values_problem(values: np.ndarray, angle_range_deg=(45.0, 135.0)) -> str | None:
    """Why anchors with the coordinates ``values`` (frames x 6) give no reference values by
    :func:`reference_values_from_histograms`, or None: in no frame do theta_A and theta_B both lie
    within ``angle_range_deg``."""
    deg = np.degrees(np.asarray(values, dtype=float)[:, 1:3])
    low, high = (float(x) for x in angle_range_deg)
    if not bool(np.any(np.all((deg >= low) & (deg <= high), axis=1))):
        return (f"in no frame do both anchor angles (theta_A, theta_B) lie within {low:g} to {high:g} "
                "degrees: the anchors give no reference values for the restraint")
    return None


#: restraints.selection: interactions takes the reference value of a coordinate from a histogram
#: of this many bins (the source of Ding et al., commit d6dff91, main_boresch_restraints.py)
REFERENCE_BINS = 20


def reference_values_from_histograms(values: np.ndarray, angle_range_deg=(45.0, 135.0),
                                     bins: int = REFERENCE_BINS) -> np.ndarray:
    """The reference values of ``restraints.selection: interactions``: for every coordinate the
    centre of the fullest bin of its histogram over the frames in which theta_A and theta_B both
    lie within ``angle_range_deg`` (``values``: frames x 6 in nm and radians; the result likewise).

    The rule of the source of Ding et al. (arXiv:2603.22274; ``filter_mask_histogram_plot.py`` and
    ``find_ideal_geometry.py`` at commit d6dff91; the paper says only that the reference
    geometries come from the trajectory): ``bins`` bins, over 0 to 180 degrees for the two angles,
    over -180 to 180 degrees for the three dihedral angles and over the range of the data for the
    distance, the centre rounded to 0.1 degree and 0.001 nm. The bins of a dihedral angle cover
    the circle, so that a peak at 180 degrees lies in two neighbouring bins as a peak on any other
    edge of a bin does (the source also makes histograms from 0 to 360 degrees, for its figure:
    they have the same bins).

    Without a frame in the range there is no reference: :class:`RestraintError`.
    """
    v = np.asarray(values, dtype=float)
    deg = np.degrees(v[:, 1:])
    low, high = (float(x) for x in angle_range_deg)
    inside = (deg[:, 0] >= low) & (deg[:, 0] <= high) & (deg[:, 1] >= low) & (deg[:, 1] <= high)
    none = reference_values_problem(v, angle_range_deg)
    if none:
        raise RestraintError(none)

    def fullest(x, span):
        if span is None and float(np.ptp(x)) < 1.0e-9:
            return float(np.mean(x))                 # a distance that does not vary has no range to divide into bins
        counts, edges = np.histogram(x, bins=int(bins), range=span)
        k = int(np.argmax(counts))
        return 0.5 * (edges[k] + edges[k + 1])

    r0 = round(fullest(v[inside, 0], None), 3)
    angles = [round(fullest(deg[inside, i], (0.0, 180.0)), 1) for i in (0, 1)]
    dihedrals = [round(fullest(deg[inside, i], (-180.0, 180.0)), 1) for i in (2, 3, 4)]
    return np.array([r0, *np.radians(angles), *np.radians(dihedrals)])


#: the secondary structures that count as stable for restraints.selection: interactions, as DSSP
#: names them: alpha, 3-10 and pi helix, extended strand and bridge (Ding et al., sec. 2.1)
STABLE_STRUCTURES = "HGIEB"


def stable_structure_fractions(topology, frames, frame_atoms) -> dict[int, float]:
    """For every amino-acid residue (its index in the topology) the fraction of ``frames`` in which
    it is in one of :data:`STABLE_STRUCTURES`, by DSSP as MDTraj computes it. ``frames`` (frames x
    frame atoms x 3, nm) hold the atoms ``frame_atoms``.

    The paper names these five structures; its source uses a DSSP that tells helix, strand and
    loop apart only, so that a 3-10 or pi helix and a bridge are a loop there.
    """
    import mdtraj as md

    from cufep.prep.protein import AMINO_ACIDS

    atoms = list(topology.atoms())
    row = {int(a): i for i, a in enumerate(np.asarray(frame_atoms, dtype=int))}
    protein = np.array([a.index for a in atoms if a.residue.name in AMINO_ACIDS and a.index in row], dtype=int)
    if not len(protein):
        return {}
    residues: list[int] = []
    for a in protein:
        index = atoms[a].residue.index
        if not residues or residues[-1] != index:
            residues.append(index)
    sub = md.Topology.from_openmm(topology).subset(protein)
    xyz = np.asarray(frames, dtype=np.float32)[:, [row[int(a)] for a in protein], :]
    codes = md.compute_dssp(md.Trajectory(xyz, sub), simplified=False)             # frames x residues
    if codes.shape[1] != len(residues):
        raise RestraintError(f"DSSP gave {codes.shape[1]} residues for {len(residues)} of the protein")
    stable = np.isin(codes, list(STABLE_STRUCTURES)).mean(axis=0)
    return {int(r): float(x) for r, x in zip(residues, stable)}


def backbone_hosts(topology) -> dict[int, tuple[int, int, int]]:
    """The residues that can hold the host anchors of ``restraints.selection: interactions``, with
    their backbone atoms (N, CA, C) as numbers of the System: the amino-acid residues that have
    exactly one atom of each name. A cap has no such three atoms and is none."""
    from cufep.prep.protein import AMINO_ACIDS

    hosts = {}
    for residue in topology.residues():
        if residue.name not in AMINO_ACIDS:
            continue
        named: dict[str, list[int]] = {}
        for atom in residue.atoms():
            named.setdefault(atom.name, []).append(atom.index)
        if all(len(named.get(name, ())) == 1 for name in ("N", "CA", "C")):
            hosts[residue.index] = (named["N"][0], named["CA"][0], named["C"][0])
    return hosts


def anchor_margins(settings: "RestraintSearchSettings", temperature_K: float) -> float:
    """The angle (radians) from 0 and pi of :func:`anchor_geometry_problem`: where the energy of a
    restrained angle reaches ``flat_angle_energy_kt`` kT."""
    return flat_margin_of(settings.k_angle, temperature_K, settings.flat_angle_energy_kt)


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
    return ligand_in_the_image_of(positions_nm, ligand, ligand, anchor, box_nm)


def ligand_in_the_image_of(positions_nm: np.ndarray, ligand: np.ndarray, of_atoms, anchor: np.ndarray,
                           box_nm: np.ndarray) -> np.ndarray:
    """Copy of the coordinates with the ligand moved, as a whole, by the lattice vector that brings
    the centre of its atoms ``of_atoms`` into the periodic image nearest to ``anchor``. Shapes as for
    :func:`nearest_image_of_ligand`."""
    from cufep.alchemy.lrc import nearest_image

    single = positions_nm.ndim == 2
    pos = np.array(positions_nm[None] if single else positions_nm, dtype=float)
    anchors = np.asarray(anchor, dtype=float).reshape(-1, 3)
    boxes = np.asarray(box_nm, dtype=float).reshape(-1, 3, 3)
    for f in range(pos.shape[0]):
        delta = pos[f, of_atoms].mean(axis=0) - anchors[min(f, len(anchors) - 1)]
        # the lattice vector that brings them next to the anchor, however many cells that is
        pos[f, ligand] += nearest_image(delta[None], boxes[min(f, len(boxes) - 1)])[0] - delta
    return pos[0] if single else pos


def ligand_next_to_protein(positions_nm: np.ndarray, ligand: np.ndarray, protein: np.ndarray,
                           box_nm: np.ndarray) -> np.ndarray:
    """Copy of the coordinates with the ligand, as a whole, in the periodic image in which it lies
    next to the protein: the image nearest to that atom of ``protein`` (its C-alpha atoms) which is
    nearest to the centre of the ligand in any image. Shapes as for :func:`nearest_image_of_ligand`.

    The one rule of everything that looks at the bound ligand without knowing its pocket (the
    interactions of an MD, the search of ``restraints.selection: auto``, the RMSD of the pose).
    It is not the image nearest to the centre of the protein: a ligand on the outer face of a
    lopsided complex is farther from that centre than half a cell, and the image nearest to the
    centre then lies in the solvent on the far side (review of f5a77ce: a domain of 52 residues on
    one of 477, cell 9.05 nm, ligand 4.93 nm from the centre of the C-alpha atoms and 0.3 nm from
    its pocket; the image nearest to the centre is 1.2 nm from every backbone atom).
    Coordinates without protein atoms are returned as they are."""
    from cufep.alchemy.lrc import nearest_image

    protein = np.asarray(protein, dtype=int)
    single = positions_nm.ndim == 2
    pos = np.array(positions_nm[None] if single else positions_nm, dtype=float)
    if len(protein) == 0:
        return pos[0] if single else pos
    boxes = np.asarray(box_nm, dtype=float).reshape(-1, 3, 3)
    anchors = np.empty((pos.shape[0], 3))
    for f in range(pos.shape[0]):
        # every protein atom from the centre of the ligand, in its nearest image
        delta = nearest_image(pos[f, protein] - pos[f, ligand].mean(axis=0), boxes[min(f, len(boxes) - 1)])
        anchors[f] = pos[f, protein[int(np.argmin((delta * delta).sum(axis=1)))]]
    moved = nearest_image_of_ligand(pos, ligand, anchors, boxes)
    return moved[0] if single else moved


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
        frames_nm = ligand_next_to_protein(frames_nm, ligand, ca, frame_boxes_nm)
    backbone_all = np.array([a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS
                             and a.name in ("N", "CA", "C")], dtype=int)
    centroid = frames_nm[0, ligand_heavy, :].mean(axis=0)
    near = backbone_all[np.linalg.norm(frames_nm[0, backbone_all, :] - centroid, axis=1) < host_max_distance_nm + 0.5]
    site = near if len(near) >= 12 else ca
    aligned = kabsch_align(frames_nm, frames_nm[0], site)
    aligned = kabsch_align(aligned, aligned.mean(axis=0), site)
    return frames_nm, aligned, site


def pose_rmsd_in_site(topology, aligned: np.ndarray, site: np.ndarray, ligand: np.ndarray, ligand_heavy: np.ndarray,
                      rdmol, input_pose_nm: np.ndarray | None = None, input_box_nm: np.ndarray | None = None,
                      second_half: bool = True) -> float:
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
            given = ligand_next_to_protein(given, ligand, _c_alpha_atoms(topology), input_box_nm)
        reference = kabsch_align(given[None], mean, site)[0][ligand_heavy, :]
    rmsd = None
    for perm in heavy_atom_symmetries(rdmol):
        d = np.sqrt(((aligned[:, ligand_heavy[perm], :] - reference) ** 2).sum(axis=2).mean(axis=1))
        rmsd = d if rmsd is None else np.minimum(rmsd, d)
    # of the second half of one trajectory (the pre-equilibration); of all frames where they are the analysed
    # part of several (the MD from which a restraint is chosen)
    return float(rmsd[len(rmsd) // 2 if second_half else 0:].mean())


def ligand_pose_rmsd(topology, frames_nm: np.ndarray, ligand_atoms, rdmol, input_pose_nm: np.ndarray,
                     frame_boxes_nm: np.ndarray | None = None, input_box_nm: np.ndarray | None = None,
                     host_max_distance_nm: float = RestraintSearchSettings.host_max_distance_nm,
                     second_half: bool = True) -> float:
    """How far the ligand of the frames is from its input pose (nm): the measure of the restraint
    search (:func:`binding_site_frames`, :func:`pose_rmsd_in_site`), for anchors that were not
    chosen by it (review of PR 1: with manual anchors a ligand that left the site gave no warning)."""
    lig = np.asarray(ligand_atoms, dtype=int)
    heavy = lig[[i for i, a in enumerate(rdmol.GetAtoms()) if a.GetAtomicNum() > 1]]
    _frames, aligned, site = binding_site_frames(topology, frames_nm, lig, heavy, host_max_distance_nm, frame_boxes_nm)
    return pose_rmsd_in_site(topology, aligned, site, lig, heavy, rdmol, input_pose_nm, input_box_nm, second_half)


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
            bonded[i].add(j)
            bonded[j].add(i)

    sig_r = math.sqrt(kt / s.k_r)
    sig_a = math.sqrt(kt / s.k_angle)
    flat_margin = anchor_margins(s, temperature_K)
    # the configuration the reference values come from, as the run has it: the last frame before it was
    # superposed on the binding site, and the positions the run starts from
    copies = [frames_nm[-1]]
    if start_nm is not None:
        start = np.asarray(start_nm, dtype=float)
        if start_box_nm is not None:
            start = ligand_next_to_protein(start, lig, ca, np.asarray(start_box_nm, dtype=float))
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
                    if anchor_geometry_problem(r, aligned, None if s.constants_from_fluctuations else flat_margin,
                                               reference_copies=copies) is not None:
                        continue
                    v = r.values(aligned)                                           # (frames, 6)
                    if s.constants_from_fluctuations and constants_from_fluctuations(
                            r, v, v[:, 1:3], temperature_K, s.flat_angle_energy_kt)[1] is not None:
                        continue
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
