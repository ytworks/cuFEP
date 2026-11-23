"""Water relocation moves between a sphere around the ligand and the rest of the cell.

The move is RelocationMC of Ding et al. (arXiv:2603.22274 v3, "RelocationMC water-sampling
protocol" and the appendix "RelocationMC workflow and acceptance-ratio derivation"), written from
the paper. It moves a whole water molecule between the inside and the outside of a sphere centred
on the ligand; the number of particles and the box do not change.

The rule of the paper, as it is implemented here
-------------------------------------------------
With x the configuration and beta = 1 / kT:

1. A direction is chosen (probability ``g``): the source region S is the sphere or the rest of
   the cell, the target region T the other one.
2. Every water M of S gets a score s_M, its interaction with everything else, and the water D
   that is moved is chosen with probability exp(beta s_D) / Z_S(x), Z_S(x) the sum of
   exp(beta s_M) over the waters of S: a water that is held weakly is preferred.
3. K anchors are drawn, each with an independent uniform orientation of the conformation of D,
   and sorted into the two regions afterwards. A trial pose i has the weight
   w_i = exp(-beta u_i) / h(i), u_i its interaction with everything but D and h the density its
   anchor was drawn with. Among the poses of T one, C, is chosen with probability w_C / W_f,
   W_f the sum of the weights of the poses of T.
4. The reverse pool is the same K poses with the chosen one replaced by the pose D has; W_r is
   the sum of the weights of its poses of S. Z_T(y) is the sum of exp(beta s_M) over the waters
   of T in the proposed configuration y, the moved water among them.
5. The move is accepted with probability min(1, r),

       ln r = ln(g_r / g_f) + ln Z_S(x) - ln Z_T(y) + beta (s_C - s_D)
              + ln W_f - ln W_r - beta [dU - (u_C - u_D)],

   dU = U(y) - U(x) the change of the potential energy of the simulation.
6. An accepted water keeps the velocity of its centre of mass; its velocities relative to it are
   turned with the molecule.

Where the paper is silent or this module differs (cuFEP's own)
---------------------------------------------------------------
- The energies u and the scores s. The paper computes them with a CUDA kernel: the direct-space
  and reciprocal-space PME interaction of every pose, so that dU - (u_C - u_D) is only a
  correction of the finite mesh. OpenMM gives no energy per molecule, so here u is a stand-in:
  Lennard-Jones and reaction-field Coulomb (dielectric 78.5) within the cutoff of the run, the
  ligand with its charges scaled by lambda_electrostatics and its soft-core at lambda_sterics;
  REST2 is left out of it. The score s is the same within a shorter radius. What the stand-in
  misses is in dU - (u_C - u_D), which uses the potential energy of the Context in the state the
  replica is simulated in: the move keeps the distribution of that state exactly, whatever the
  stand-in is (the Hastings ratio above holds for any u and s). A poor stand-in costs acceptance,
  not correctness.
- The density of the anchors. The paper draws them uniformly in the whole cell (h constant, so it
  cancels and no volume appears). With ``inner_fraction`` a share of them is drawn uniformly in
  the sphere and the rest uniformly outside it; h then differs between the regions and stays in
  the weights, as in the formula of the paper before its h is cancelled. None is the paper's
  uniform draw.
- The paper names neither the point of a water that is put on an anchor nor the one that decides
  its region: the oxygen here, for both. The two directions are chosen with probability 1/2.
- The cell can be triclinic (the paper's kernel needs an orthorhombic one).
- When a move is attempted (``attempt_is_due``): the paper attempts one every 1000 MD steps (4 ps);
  here one is attempted at the end of the MD of an iteration during which a multiple of
  ``interval_ps`` was passed, the same number of attempts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cufep.alchemy.lrc import nearest_image

KB_KJ_PER_MOL_K = 8.31446261815324e-3
ONE_4PI_EPS0 = 138.93545764438198
#: dielectric of the reaction field of the stand-in energy
DIELECTRIC = 78.5
#: radius (nm) within which the score of a water, which decides how likely it is the one that is
#: moved, counts its interactions: its first two shells of neighbours
SCORE_RADIUS_NM = 0.5
#: the probability of proposing a move out of the sphere (the other direction: one minus it)
OUT_PROBABILITY = 0.5


def attempt_is_due(iteration: int, steps_per_iteration: int, interval_steps: int) -> bool:
    """Whether a move is attempted after the MD of ``iteration`` (counted from 0): if a multiple of
    ``interval_steps`` MD steps of the replica's trajectory was passed during it. A function of
    the iteration alone, so that a leg that is resumed attempts its moves where it would have."""
    before, after = int(iteration) * int(steps_per_iteration), (int(iteration) + 1) * int(steps_per_iteration)
    return after // int(interval_steps) > before // int(interval_steps)


def random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` rotation matrices, uniform over orientations (the Haar measure): unit quaternions
    from four normal numbers."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


def sphere_radius(ligand_heavy_positions: np.ndarray, margin_nm: float) -> float:
    """Radius of the sphere: half the largest distance between two heavy atoms of the ligand in
    the pose that is given (the molecule whole), plus ``margin_nm`` (the paper: 0.3 nm)."""
    x = np.asarray(ligand_heavy_positions, dtype=float)
    return 0.5 * float(np.max(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2))) + float(margin_nm)


@dataclass
class RelocationSpec:
    """What a mover needs of a leg (picklable; made by the workflow from the prepared system)."""

    waters: np.ndarray            # (number of waters, sites per water) atom indices, the oxygen first
    ligand: np.ndarray            # atom indices of the ligand
    ligand_heavy: np.ndarray      # its heavy atoms: the centre of the sphere
    radius_nm: float
    charge: np.ndarray            # per atom, of the fully interacting System (e)
    sigma: np.ndarray             # nm
    epsilon: np.ndarray           # kJ/mol
    mass: np.ndarray              # amu (0 for a virtual site)
    cutoff_nm: float
    softcore_alpha: float
    softcore_b: int
    temperature_K: float
    trial_poses: int
    inner_fraction: float | None  # None: anchors uniform in the whole cell (the paper)
    interval_ps: float


def cell_heights(box: np.ndarray) -> np.ndarray:
    """Distances between the opposite faces of the cell (rows of ``box`` are its vectors)."""
    volume = abs(float(np.linalg.det(box)))
    areas = [np.linalg.norm(np.cross(box[(k + 1) % 3], box[(k + 2) % 3])) for k in range(3)]
    return volume / np.asarray(areas)


class Environment:
    """The atoms of one configuration with their periodic images within a margin of the cell, for
    energies of points with everything within a radius of them (minimum image, any cell)."""

    def __init__(self, positions: np.ndarray, box: np.ndarray, margin_nm: float) -> None:
        from scipy.spatial import cKDTree

        self.box = np.asarray(box, dtype=float)
        self.inverse = np.linalg.inv(self.box)
        heights = cell_heights(self.box)
        if 2.0 * margin_nm >= float(heights.min()):
            raise ValueError(f"a radius of {margin_nm:g} nm is not less than half the smallest height of the cell "
                             f"({heights.min():g} nm)")
        frac = np.asarray(positions, dtype=float) @ self.inverse
        frac -= np.floor(frac)
        low, high = frac * heights < margin_nm, (1.0 - frac) * heights < margin_nm
        points, index = [frac @ self.box], [np.arange(len(frac))]
        for shift in np.ndindex(3, 3, 3):
            shift = np.asarray(shift) - 1
            if not shift.any():
                continue
            # an image shifted by +1 along an axis lies beyond the far face: within the margin of
            # the cell only if the atom is within the margin of the near face, and the reverse
            keep = np.ones(len(frac), dtype=bool)
            for axis in range(3):
                if shift[axis] == 1:
                    keep &= low[:, axis]
                elif shift[axis] == -1:
                    keep &= high[:, axis]
            if keep.any():
                points.append((frac[keep] + shift) @ self.box)
                index.append(np.flatnonzero(keep))
        self.atom = np.concatenate(index)                     # the atom of every point
        self.tree = cKDTree(np.concatenate(points))
        self.margin = float(margin_nm)

    def wrapped(self, points: np.ndarray) -> np.ndarray:
        frac = np.asarray(points, dtype=float) @ self.inverse
        return (frac - np.floor(frac)) @ self.box

    def pairs(self, points: np.ndarray, radius_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(index of the point, atom, distance) of every atom image within ``radius_nm`` of every
        point."""
        from scipy.spatial import cKDTree

        if radius_nm > self.margin + 1e-12:
            raise ValueError("the radius is larger than the margin the images were made for")
        found = cKDTree(self.wrapped(points)).sparse_distance_matrix(self.tree, float(radius_nm), output_type="ndarray")
        return found["i"], self.atom[found["j"]], found["v"]


class StandIn:
    """The stand-in energy of points with the atoms of one configuration in one alchemical state
    (the module's docstring): Lennard-Jones and reaction-field Coulomb within a radius."""

    def __init__(self, spec: RelocationSpec, positions: np.ndarray, box: np.ndarray, lambda_electrostatics: float,
                 lambda_sterics: float) -> None:
        self.environment = Environment(positions, box, spec.cutoff_nm)
        self.charge = np.array(spec.charge, dtype=float)
        self.charge[spec.ligand] *= float(lambda_electrostatics)
        self.sigma = np.asarray(spec.sigma, dtype=float)
        self.epsilon = np.asarray(spec.epsilon, dtype=float)
        # the soft-core of the ligand: x = 1 / (alpha (1 - lambda)^b + (r / sigma)^6), times lambda
        self.soft = np.zeros(len(self.charge))
        self.soft[spec.ligand] = float(spec.softcore_alpha) * (1.0 - float(lambda_sterics)) ** int(spec.softcore_b)
        self.scale = np.ones(len(self.charge))
        self.scale[spec.ligand] = float(lambda_sterics)

    def pair_energies(self, points: np.ndarray, charge: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray,
                      radius_nm: float, skip: np.ndarray | None = None,
                      own: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(point, atom, energy in kJ/mol) of every pair of a point, with the given charge and
        Lennard-Jones parameters, and an atom within ``radius_nm`` of it. ``skip`` are atoms that
        are left out for every point; ``own`` gives per point a number, and an atom with the same
        number in ``self.group`` is left out for it (the atoms of the point's own molecule)."""
        i, j, r = self.environment.pairs(points, radius_nm)
        keep = r > 0.0
        if skip is not None and len(skip):
            mask = np.zeros(len(self.charge), dtype=bool)
            mask[np.asarray(skip)] = True
            keep &= ~mask[j]
        if own is not None:
            keep &= self.group[j] != np.asarray(own)[i]
        i, j, r = i[keep], j[keep], r[keep]
        k_rf = (DIELECTRIC - 1.0) / ((2.0 * DIELECTRIC + 1.0) * radius_nm ** 3)
        c_rf = 3.0 * DIELECTRIC / ((2.0 * DIELECTRIC + 1.0) * radius_nm)
        energy = ONE_4PI_EPS0 * np.asarray(charge)[i] * self.charge[j] * (1.0 / r + k_rf * r * r - c_rf)
        eps = np.sqrt(np.asarray(epsilon)[i] * self.epsilon[j])
        lj = eps > 0.0
        if lj.any():
            sig = 0.5 * (np.asarray(sigma)[i][lj] + self.sigma[j][lj])
            x = 1.0 / (self.soft[j][lj] + (r[lj] / sig) ** 6)
            energy[lj] += self.scale[j][lj] * 4.0 * eps[lj] * x * (x - 1.0)
        return i, j, energy

    def energies(self, points: np.ndarray, charge: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray,
                 radius_nm: float, skip: np.ndarray | None = None, own: np.ndarray | None = None) -> np.ndarray:
        """Energy (kJ/mol) of every point with all atoms within ``radius_nm`` of it
        (:meth:`pair_energies`)."""
        i, _j, energy = self.pair_energies(points, charge, sigma, epsilon, radius_nm, skip, own)
        return np.bincount(i, weights=energy, minlength=len(points))

    def by_water(self, points: np.ndarray, charge: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray,
                 radius_nm: float, skip: np.ndarray | None = None) -> np.ndarray:
        """Energy (kJ/mol) of the points together with each water (:meth:`set_groups`): what a
        molecule at these points adds to the energy every water has with everything else."""
        _i, j, energy = self.pair_energies(points, charge, sigma, epsilon, radius_nm, skip)
        water = self.group[j]
        return np.bincount(water[water >= 0], weights=energy[water >= 0], minlength=int(self.group.max()) + 1)

    def set_groups(self, n_atoms: int, waters: np.ndarray) -> None:
        """Number the atoms by the water they belong to (-1: none), for ``own`` of :meth:`energies`."""
        self.group = np.full(n_atoms, -1, dtype=np.int64)
        self.group[waters] = np.arange(len(waters))[:, None]


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    top = float(values.max())
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.exp(values - top).sum()))


class RelocationMover:
    """Attempts water relocation moves in one OpenMM Context (the module's docstring)."""

    def __init__(self, spec: RelocationSpec, context, seed: int) -> None:
        self.spec = spec
        self.context = context
        self.rng = np.random.default_rng(int(seed))
        self.beta = 1.0 / (KB_KJ_PER_MOL_K * float(spec.temperature_K))
        self.waters = np.asarray(spec.waters, dtype=np.int64)
        self.n_sites = self.waters.shape[1]
        if spec.inner_fraction is not None and not 0.0 < float(spec.inner_fraction) < 1.0:
            raise ValueError("inner_fraction is a number between 0 and 1, or none")
        if SCORE_RADIUS_NM > spec.cutoff_nm:
            raise ValueError("the cutoff is shorter than the radius of the scores")

    # -- the potential energy of the simulation ------------------------------------------------------
    def _potential(self, positions: np.ndarray | None = None) -> float:
        import openmm.unit as u

        if positions is not None:
            self.context.setPositions(positions)
        # all force groups in one computation (the neighbour lists of the soft-core force are
        # brought up to date only together with the standard nonbonded kernel)
        return float(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))

    # -- the proposal ----------------------------------------------------------------------------------
    def _anchors(self, centre: np.ndarray, box: np.ndarray, p_in: float) -> tuple[np.ndarray, np.ndarray]:
        """(anchors, whether each is in the sphere): each one in the sphere with probability
        ``p_in``, uniform there, else uniform in the rest of the cell."""
        k, radius = int(self.spec.trial_poses), float(self.spec.radius_nm)
        inner = self.rng.random(k) < p_in
        anchors = np.empty((k, 3))
        n_in = int(inner.sum())
        direction = self.rng.normal(size=(n_in, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        anchors[inner] = centre + direction * (radius * self.rng.random(n_in) ** (1.0 / 3.0))[:, None]
        todo = np.flatnonzero(~inner)
        while len(todo):
            trial = self.rng.random((len(todo), 3)) @ box
            outside = np.linalg.norm(nearest_image(trial - centre, box), axis=1) >= radius
            anchors[todo[outside]] = trial[outside]
            todo = todo[~outside]
        return anchors, inner

    def centre(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
        """The centre of the sphere: the geometric centre of the heavy atoms of the ligand, the
        molecule taken whole."""
        heavy = x[self.spec.ligand_heavy]
        return heavy[0] + nearest_image(heavy - heavy[0], box).mean(axis=0)

    def inside(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Whether each water is in the sphere: by its oxygen."""
        distance = np.linalg.norm(nearest_image(x[self.waters[:, 0]] - self.centre(x, box), box), axis=1)
        return distance < float(self.spec.radius_nm)

    def proposal(self, x: np.ndarray, box: np.ndarray, state, out: bool, moved: int | None = None,
                 poses: np.ndarray | None = None, anchor_inside: np.ndarray | None = None,
                 chosen: int | None = None) -> dict | None:
        """One move proposed from the configuration ``x`` in the alchemical state ``state``: out of
        the sphere (``out``) or into it. Returns None if none can be proposed (no water to take,
        no trial pose in the target region); else ``y``, the proposed configuration, and
        ``log_q_ratio``, with which the move is accepted with probability min(1, r),

            ln r = log_q_ratio - beta (U(y) - U(x)),

        and what was drawn: ``moved`` (the water), ``poses`` (the K trial poses of its sites),
        ``anchor_inside`` (whether each is in the sphere), ``chosen`` (the pose it is put in) and
        ``rotation`` of that pose. Given instead of drawn, they make the proposal a function of its
        arguments: the reverse of a move is the move back with the same poses, the chosen one
        replaced by the pose the water had, and its ``log_q_ratio`` is the negative."""
        spec, beta = self.spec, self.beta
        radius = float(spec.radius_nm)
        centre = self.centre(x, box)
        inside = self.inside(x, box)
        source, target = (inside, ~inside) if out else (~inside, inside)
        if not source.any():
            return None
        volume = abs(float(np.linalg.det(box)))
        v_in = 4.0 / 3.0 * np.pi * radius ** 3
        p_in = v_in / volume if spec.inner_fraction is None else float(spec.inner_fraction)
        log_h_in, log_h_out = np.log(p_in / v_in), np.log((1.0 - p_in) / (volume - v_in))
        log_h_source, log_h_target = (log_h_in, log_h_out) if out else (log_h_out, log_h_in)

        energy = StandIn(spec, x, box, state.lambda_electrostatics, state.lambda_sterics)
        energy.set_groups(len(x), self.waters)
        q, sigma, epsilon = spec.charge[self.waters[0]], spec.sigma[self.waters[0]], spec.epsilon[self.waters[0]]
        n_w, n_s = self.waters.shape
        # the scores of all waters in x: the interaction of each with everything else
        per_site = energy.energies(x[self.waters.reshape(-1)], np.tile(q, n_w), np.tile(sigma, n_w),
                                   np.tile(epsilon, n_w), SCORE_RADIUS_NM, own=np.repeat(np.arange(n_w), n_s))
        score = per_site.reshape(n_w, n_s).sum(axis=1)
        log_z_source = logsumexp(beta * score[source])
        if moved is None:
            members = np.flatnonzero(source)
            weights = np.exp(beta * score[members] - log_z_source)
            moved = int(self.rng.choice(members, p=weights / weights.sum()))
        elif not source[moved]:
            raise ValueError("the water that is to be moved is not in the source region")
        atoms = self.waters[moved]
        # K trial poses: an anchor for the oxygen and a uniform orientation of the conformation of D
        rotations = None
        if poses is None:
            local = nearest_image(x[atoms] - x[atoms[0]], box)
            anchors, anchor_inside = self._anchors(centre, box, p_in)
            rotations = random_rotations(self.rng, len(anchors))
            poses = anchors[:, None, :] + np.einsum("kij,sj->ksi", rotations, local)
        # the target region is the sphere, unless the move goes out of it
        in_target = np.asarray(anchor_inside, dtype=bool) != out
        if not in_target.any():
            return None
        k = len(poses)
        trial = energy.energies(poses.reshape(-1, 3), np.tile(q, k), np.tile(sigma, k), np.tile(epsilon, k),
                                spec.cutoff_nm, skip=atoms).reshape(k, n_s).sum(axis=1)
        u_own = float(energy.energies(x[atoms], q, sigma, epsilon, spec.cutoff_nm, skip=atoms).sum())
        log_w_forward = logsumexp(-beta * trial[in_target]) - log_h_target
        if not np.isfinite(log_w_forward):
            return None
        if chosen is None:
            candidates = np.flatnonzero(in_target)
            p = np.exp(-beta * trial[candidates] - (log_w_forward + log_h_target))
            chosen = int(self.rng.choice(candidates, p=p / p.sum()))
        elif not in_target[chosen]:
            raise ValueError("the pose that is chosen is not in the target region")
        u_new = float(trial[chosen])
        # the reverse pool: the other poses, with the pose D has in place of the chosen one
        log_w_reverse = logsumexp(np.append(-beta * trial[~in_target], -beta * u_own)) - log_h_source
        # the scores of the waters of the target region in the proposed configuration: those it has
        # in x, each with its interaction with D moved from the old pose to the new one, and D itself
        y = x.copy()
        y[atoms] = poses[chosen]
        others = np.flatnonzero(target)
        changed = score[others].copy()
        for sign, pose in ((-1.0, x[atoms]), (1.0, poses[chosen])):
            changed += sign * energy.by_water(pose, q, sigma, epsilon, SCORE_RADIUS_NM, skip=atoms)[others]
        s_new = float(energy.energies(poses[chosen], q, sigma, epsilon, SCORE_RADIUS_NM, skip=atoms).sum())
        log_z_target = logsumexp(np.append(beta * changed, beta * s_new))
        direction = np.log((1.0 - OUT_PROBABILITY) / OUT_PROBABILITY) * (1.0 if out else -1.0)
        log_q_ratio = (direction + log_z_source - log_z_target + beta * (s_new - score[moved])
                       + log_w_forward - log_w_reverse + beta * (u_new - u_own))
        return {"y": y, "log_q_ratio": float(log_q_ratio), "moved": int(moved), "poses": poses,
                "anchor_inside": np.asarray(anchor_inside, dtype=bool), "chosen": int(chosen),
                "rotation": None if rotations is None else rotations[chosen]}

    def attempt(self, state) -> dict:
        """One move on the configuration of the Context, in the alchemical state ``state`` (which
        the Context is set to). Returns ``direction`` ("in": into the sphere, "out": out of it) and
        ``accepted``, and ``log_r`` (ln r) where a move could be proposed; one that cannot be
        proposed is a move that was not accepted. The Context is changed only by a move that is
        accepted."""
        import openmm.unit as u

        st = self.context.getState(getPositions=True)
        x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        out = bool(self.rng.random() < OUT_PROBABILITY)
        result = {"direction": "out" if out else "in", "accepted": False}
        move = self.proposal(x, box, state, out)
        if move is None:
            return result
        u_x = self._potential()
        u_y = self._potential(move["y"])
        log_r = move["log_q_ratio"] - self.beta * (u_y - u_x)
        result["log_r"] = float(log_r)
        if not np.isfinite(log_r) or np.log(self.rng.random()) >= log_r:
            self.context.setPositions(x)
            return result
        # accepted: the velocity of the centre of mass is kept, the rest turns with the molecule
        atoms = self.waters[move["moved"]]
        velocities = np.asarray(self.context.getState(getVelocities=True).getVelocities(asNumpy=True)
                                .value_in_unit(u.nanometer / u.picosecond), dtype=float)
        mass = np.asarray(self.spec.mass, dtype=float)[atoms]
        com = (mass[:, None] * velocities[atoms]).sum(axis=0) / mass.sum()
        velocities[atoms] = com + (velocities[atoms] - com) @ move["rotation"].T
        self.context.setVelocities(velocities)
        result["accepted"] = True
        return result


class RelocationError(ValueError):
    """The moves cannot be made in this system as they are asked for."""


def relocation_spec(system, topology, positions: np.ndarray, box: np.ndarray, ligand, cutoff_nm: float,
                    softcore_alpha: float, softcore_b: int, temperature_K: float, trial_poses: int,
                    inner_fraction: float | None, interval_ps: float, radius_margin_nm: float) -> RelocationSpec:
    """What a mover needs, from the fully interacting ``system`` of a leg (the one the alchemical
    System is made from), its ``topology`` and the pose that was prepared (``positions``, ``box``).

    The waters that can be moved are the residues named HOH: all of them. As in the paper they
    must be alike (the same charge and Lennard-Jones parameters site by site) and must not
    interact within themselves (every nonbonded exception inside one is zero), so that the energy
    of a water with everything else is the energy of its sites with all other atoms. The sphere
    must fit into the cell: twice its radius, and twice the cutoff, less than its smallest height.
    Raises RelocationError otherwise."""
    import openmm as mm
    import openmm.unit as u

    force = next((f for f in system.getForces() if isinstance(f, mm.NonbondedForce)), None)
    if force is None:
        raise RelocationError("the System has no NonbondedForce")
    units = (u.elementary_charge, u.nanometer, u.kilojoule_per_mole)
    parameters = np.array([[value.value_in_unit(unit) for value, unit in zip(force.getParticleParameters(i), units)]
                           for i in range(system.getNumParticles())])
    mass = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
    atoms = list(topology.atoms())
    waters = []
    for residue in topology.residues():
        if residue.name != "HOH":
            continue
        members = [a.index for a in residue.atoms()]
        oxygen = [i for i in members if atoms[i].element is not None and atoms[i].element.symbol == "O"]
        if len(oxygen) != 1:
            raise RelocationError(f"water {residue.id} has {len(oxygen)} oxygen atoms")
        waters.append(oxygen + [i for i in members if i != oxygen[0]])
    if not waters:
        raise RelocationError("the system has no water (residues named HOH)")
    if len({len(w) for w in waters}) != 1:
        raise RelocationError("the waters do not all have the same number of sites")
    waters = np.array(waters, dtype=np.int64)
    if not np.allclose(parameters[waters], parameters[waters[0]][None], rtol=0.0, atol=1e-12):
        raise RelocationError("the waters do not all have the same nonbonded parameters")
    water_of = np.full(len(atoms), -1, dtype=np.int64)
    water_of[waters] = np.arange(len(waters))[:, None]
    for k in range(force.getNumExceptions()):
        i, j, qq, _sigma, epsilon = force.getExceptionParameters(k)
        if water_of[i] >= 0 and water_of[i] == water_of[j] and (
                abs(qq.value_in_unit(u.elementary_charge ** 2)) > 0.0
                or abs(epsilon.value_in_unit(u.kilojoule_per_mole)) > 0.0):
            raise RelocationError("a water has a nonbonded interaction between its own sites")
    ligand = np.array(sorted(int(i) for i in ligand), dtype=np.int64)
    heavy = np.array([i for i in ligand if atoms[i].element is not None and atoms[i].element.symbol != "H"],
                     dtype=np.int64)
    if not len(heavy):
        raise RelocationError("the ligand has no heavy atom")
    box = np.asarray(box, dtype=float)
    pose = np.asarray(positions, dtype=float)[heavy]
    pose = pose[0] + nearest_image(pose - pose[0], box)
    radius = sphere_radius(pose, radius_margin_nm)
    smallest = float(cell_heights(box).min())
    if 2.0 * radius >= smallest or 2.0 * float(cutoff_nm) >= smallest:
        raise RelocationError(f"the sphere (radius {radius:.2f} nm) or the cutoff ({float(cutoff_nm):g} nm) is not "
                              f"less than half the smallest height of the cell ({smallest:.2f} nm)")
    if float(cutoff_nm) < SCORE_RADIUS_NM:
        raise RelocationError(f"the cutoff ({float(cutoff_nm):g} nm) is shorter than the radius of the scores "
                              f"({SCORE_RADIUS_NM:g} nm)")
    return RelocationSpec(waters=waters, ligand=ligand, ligand_heavy=heavy, radius_nm=radius,
                          charge=parameters[:, 0].copy(), sigma=parameters[:, 1].copy(),
                          epsilon=parameters[:, 2].copy(), mass=mass, cutoff_nm=float(cutoff_nm),
                          softcore_alpha=float(softcore_alpha), softcore_b=int(softcore_b),
                          temperature_K=float(temperature_K), trial_poses=int(trial_poses),
                          inner_fraction=None if inner_fraction is None else float(inner_fraction),
                          interval_ps=float(interval_ps))
