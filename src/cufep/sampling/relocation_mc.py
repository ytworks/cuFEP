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
- When a move is attempted (``attempts_due``): the paper attempts one every 1000 MD steps (4 ps);
  here one is attempted, at the end of the MD of an iteration, for every multiple of
  ``interval_ps`` that was passed during it: the same number of attempts.
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


def attempts_due(iteration: int, steps_per_iteration: int, interval_steps: int) -> int:
    """How many moves are attempted after the MD of ``iteration`` (counted from 0): one for every
    multiple of ``interval_steps`` MD steps of the replica's trajectory that was passed during it,
    so that the number of attempts is that of the interval whatever the length of an iteration
    (review of 723b12f: with an interval shorter than an iteration only one move was attempted
    per iteration, fewer than the setting says). A function of the iteration alone, so that a
    leg that is resumed attempts its moves where it would have."""
    before, after = int(iteration) * int(steps_per_iteration), (int(iteration) + 1) * int(steps_per_iteration)
    return after // int(interval_steps) - before // int(interval_steps)


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


def _bond_pairs(bonds) -> np.ndarray:
    return np.asarray(list(bonds) if bonds is not None else [], dtype=np.int64).reshape(-1, 2)


def _molecules(n: int, pairs: np.ndarray) -> list[list[int]]:
    """The connected parts of ``n`` points joined by ``pairs``, each in the order of its points, in the order of their
    first points."""
    neighbours: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in pairs:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    seen, out = set(), []
    for start in range(n):
        if start in seen:
            continue
        group, todo = {start}, [start]
        while todo:
            i = todo.pop()
            for j in neighbours[i]:
                if j not in group:
                    group.add(j)
                    todo.append(j)
        seen |= group
        out.append(sorted(group))
    return out


def nearest_first_is_whole(points: np.ndarray, bonds, box: np.ndarray) -> bool:
    """Whether the images of the points nearest to the first point are the pose of :func:`whole_pose`: they keep every
    bond whole, and the geometric centre of every molecule after the first is at the image nearest to that of the
    first. It is then the pose of every version before :func:`whole_pose` (the images nearest to the first atom)."""
    x, box = np.asarray(points, dtype=float), np.asarray(box, dtype=float)
    pairs = _bond_pairs(bonds)
    simple = x[0] + nearest_image(x - x[0], box)
    if len(pairs):
        d = simple[pairs[:, 1]] - simple[pairs[:, 0]]
        if not np.allclose(d, nearest_image(d, box), rtol=0.0, atol=1e-9):
            return False
    molecules = _molecules(len(x), pairs)
    first = simple[molecules[0]].mean(axis=0)
    for members in molecules[1:]:
        d = (simple[members].mean(axis=0) - first)[None]
        if not np.allclose(d, nearest_image(d, box), rtol=0.0, atol=1e-9):
            return False
    return True


def whole_pose(points: np.ndarray, bonds, box: np.ndarray) -> np.ndarray:
    """The points (n x 3, nm) of molecules in a periodic cell with every molecule whole: each along its ``bonds``
    (pairs of indices of the points), every molecule after the first with its geometric centre at the image nearest to
    that of the first. Where the images nearest to the first point are that pose (:func:`nearest_first_is_whole`),
    those images, the numbers of before. (Second review of PR #19: a ligand of 2.1 nm in a cell of 4 nm was broken by
    the images nearest to its first atom, and its sphere took 2.225 nm instead of 1.35; third review: the second end
    state of an edge, whole, was put at the image nearest to the first atom of the first, 2.255 nm instead of
    1.395.)"""
    x, box = np.asarray(points, dtype=float), np.asarray(box, dtype=float)
    if nearest_first_is_whole(x, bonds, box):
        return x[0] + nearest_image(x - x[0], box)
    pairs = _bond_pairs(bonds)
    neighbours: dict[int, list[int]] = {i: [] for i in range(len(x))}
    for i, j in pairs:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    out = np.full_like(x, np.nan)
    first_centre = None
    for members in _molecules(len(x), pairs):
        start = members[0]
        out[start] = x[start]
        todo = [start]
        while todo:
            i = todo.pop()
            for j in sorted(neighbours[i]):
                if np.isnan(out[j, 0]):
                    out[j] = out[i] + nearest_image((x[j] - x[i])[None], box)[0]
                    todo.append(j)
        centre = out[members].mean(axis=0)
        if first_centre is None:
            first_centre = centre
        else:
            out[members] += first_centre + nearest_image((centre - first_centre)[None], box)[0] - centre
    return out


def heavy_bonds(topology, heavy, bonds=None) -> np.ndarray:
    """The bonds between the atoms ``heavy`` (pairs of positions in ``heavy``): those of the topology and ``bonds``
    (pairs of atoms of the System, those a topology does not hold: of a hybrid, the bonds of an atom unique to B with
    the core; third review of PR #19)."""
    where = {int(a): k for k, a in enumerate(heavy)}
    pairs = {tuple(sorted((where[a.index], where[b.index]))) for a, b in topology.bonds()
             if a.index in where and b.index in where}
    pairs |= {tuple(sorted((where[int(i)], where[int(j)]))) for i, j in (bonds or ())
              if int(i) in where and int(j) in where}
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


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
    ligand_heavy_bonds: np.ndarray | None = None   # the bonds between them (pairs of positions in ligand_heavy)


def image_within_half_a_cell(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """The image of every difference vector whose fractional components are all within a half. It
    is the nearest image whenever the nearest one is shorter than half the smallest height of the
    cell, so a distance compared with a radius that :func:`check_cell` has passed is decided
    rightly by it (the kernels of :class:`GpuStandIn` use the same rule)."""
    frac = np.asarray(delta, dtype=float) @ np.linalg.inv(box)
    return (frac - np.rint(frac)) @ box


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
                 lambda_sterics: float, active: np.ndarray | None = None) -> None:
        """``active`` (per atom; None: every atom): an atom that is not active has no charge and no
        Lennard-Jones here, as a water that is switched off has none in the simulation
        (:mod:`cufep.sampling.gcmc`)."""
        self.spec = spec
        self.positions = np.asarray(positions, dtype=float)
        self.set_groups(len(self.positions), np.asarray(spec.waters))
        self.environment = Environment(positions, box, spec.cutoff_nm)
        self.charge = np.array(spec.charge, dtype=float)
        self.charge[spec.ligand] *= float(lambda_electrostatics)
        self.sigma = np.asarray(spec.sigma, dtype=float)
        self.epsilon = np.array(spec.epsilon, dtype=float)
        if active is not None:
            off = ~np.asarray(active, dtype=bool)
            self.charge[off], self.epsilon[off] = 0.0, 0.0
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

    # -- what a proposal needs (the same three numbers from :class:`GpuStandIn`) ---------------------
    def _site_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = np.asarray(self.spec.waters)[0]
        return self.spec.charge[first], self.spec.sigma[first], self.spec.epsilon[first]

    def scores(self) -> np.ndarray:
        """The score of every water: its energy with everything else within the radius of the
        scores."""
        waters = np.asarray(self.spec.waters)
        n_w, n_s = waters.shape
        q, sigma, epsilon = self._site_parameters()
        per_site = self.energies(self.positions[waters.reshape(-1)], np.tile(q, n_w), np.tile(sigma, n_w),
                                 np.tile(epsilon, n_w), SCORE_RADIUS_NM, own=np.repeat(np.arange(n_w), n_s))
        return per_site.reshape(n_w, n_s).sum(axis=1)

    def pose_energies(self, poses: np.ndarray, moved: int, radius_nm: float) -> np.ndarray:
        """The energy of a water in each of ``poses`` (number, sites, 3) with everything within
        ``radius_nm`` but the atoms of the water ``moved``, whose poses they are."""
        k, n_s = len(poses), np.asarray(self.spec.waters).shape[1]
        q, sigma, epsilon = self._site_parameters()
        atoms = np.asarray(self.spec.waters)[int(moved)]
        return self.energies(np.asarray(poses, dtype=float).reshape(-1, 3), np.tile(q, k), np.tile(sigma, k),
                             np.tile(epsilon, k), float(radius_nm), skip=atoms).reshape(k, n_s).sum(axis=1)

    def water_energies(self, which: np.ndarray, radius_nm: float) -> np.ndarray:
        """The energy of each of the waters ``which``, where it is, with everything else within
        ``radius_nm``."""
        which = np.asarray(which, dtype=np.int64)
        waters = np.asarray(self.spec.waters)
        n_s = waters.shape[1]
        if not len(which):
            return np.zeros(0)
        q, sigma, epsilon = self._site_parameters()
        m = len(which)
        per_site = self.energies(self.positions[waters[which].reshape(-1)], np.tile(q, m), np.tile(sigma, m),
                                 np.tile(epsilon, m), float(radius_nm), own=np.repeat(which, n_s))
        return per_site.reshape(m, n_s).sum(axis=1)

    def with_waters(self, poses: np.ndarray, moved: int) -> np.ndarray:
        """(pose, water): the energy of the water ``moved`` in each of ``poses`` with every other
        water within the radius of the scores (0 with itself)."""
        q, sigma, epsilon = self._site_parameters()
        atoms = np.asarray(self.spec.waters)[int(moved)]
        return np.stack([self.by_water(pose, q, sigma, epsilon, SCORE_RADIUS_NM, skip=atoms) for pose in poses])


#: the kernels of :class:`GpuStandIn` (CUDA C, energies in double precision). The atoms of the
#: configuration are sorted by the cell of a grid whose cells are at least 1/``reach`` of the radius
#: high, so that everything within the radius of a point is in the cells up to ``reach`` away from
#: its own. A pair is first measured in single precision against the radius plus a margin (1e-3 nm,
#: far above the error of that measure) and, if it may be within the radius, measured and computed
#: in double precision: the numbers are those of double precision alone, most pairs cost a third.
#: Fractional coordinates: the image of a difference with all three components within a half is the
#: nearest one whenever the distance is less than half the smallest height of the cell, which
#: :func:`check_cell` holds the radius to.
_CUDA_SOURCE = r"""
extern "C" __global__
void site_energies(const double* frac, const float* near, const double* par, const int* group,
                   const int* cell_start, const int* dims, const double* box, const float* box32,
                   const double* inv, const double* sites, const double* site_par, const int* own,
                   const double radius, const float r2near, const int reach, const double k_rf,
                   const double c_rf, const double coulomb, const int m, double* out)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    double f[3];
    float h[3];
    int c[3], first[3], count[3];
    for (int d = 0; d < 3; ++d) {
        f[d] = sites[3 * i] * inv[d] + sites[3 * i + 1] * inv[3 + d] + sites[3 * i + 2] * inv[6 + d];
        f[d] -= floor(f[d]);
        h[d] = (float)f[d];
        c[d] = min((int)(f[d] * dims[d]), dims[d] - 1);
        // the cells up to `reach` away, or every cell of the axis once if there are fewer
        if (dims[d] >= 2 * reach + 1) { count[d] = 2 * reach + 1; first[d] = -reach; }
        else { count[d] = dims[d]; first[d] = -c[d]; }
    }
    const double qi = site_par[3 * i], si = site_par[3 * i + 1], ei = site_par[3 * i + 2];
    const double r2max = radius * radius;
    const int mine = own[i];
    double total = 0.0;
    for (int a = 0; a < count[0]; ++a) {
        const int cx = (c[0] + first[0] + a + dims[0]) % dims[0];
        for (int b = 0; b < count[1]; ++b) {
            const int cy = (c[1] + first[1] + b + dims[1]) % dims[1];
            for (int g = 0; g < count[2]; ++g) {
                const int cz = (c[2] + first[2] + g + dims[2]) % dims[2];
                const int cell = cx + dims[0] * (cy + dims[1] * cz);
                for (int j = cell_start[cell]; j < cell_start[cell + 1]; ++j) {
                    if (mine >= 0 && group[j] == mine) continue;
                    float ex = near[3 * j] - h[0], ey = near[3 * j + 1] - h[1], ez = near[3 * j + 2] - h[2];
                    ex -= rintf(ex); ey -= rintf(ey); ez -= rintf(ez);
                    const float px = ex * box32[0] + ey * box32[3] + ez * box32[6];
                    const float py = ex * box32[1] + ey * box32[4] + ez * box32[7];
                    const float pz = ex * box32[2] + ey * box32[5] + ez * box32[8];
                    if (px * px + py * py + pz * pz > r2near) continue;
                    double dx = frac[3 * j] - f[0], dy = frac[3 * j + 1] - f[1], dz = frac[3 * j + 2] - f[2];
                    dx -= rint(dx); dy -= rint(dy); dz -= rint(dz);
                    const double x = dx * box[0] + dy * box[3] + dz * box[6];
                    const double y = dx * box[1] + dy * box[4] + dz * box[7];
                    const double z = dx * box[2] + dy * box[5] + dz * box[8];
                    const double r2 = x * x + y * y + z * z;
                    if (r2 > r2max || r2 <= 0.0) continue;
                    const double inv_r = 1.0 / sqrt(r2);
                    double e = coulomb * qi * par[5 * j] * (inv_r + k_rf * r2 - c_rf);
                    const double eps = ei * par[5 * j + 2];          // the square roots of the two epsilon
                    if (eps > 0.0) {
                        const double sig = 0.5 * (si + par[5 * j + 1]);
                        const double soft = par[5 * j + 3];
                        double s;                                    // 1 / (soft + (r / sigma)^6)
                        if (soft == 0.0) { const double a = sig * sig * inv_r * inv_r; s = a * a * a; }
                        else { const double t2 = r2 / (sig * sig); s = 1.0 / (soft + t2 * t2 * t2); }
                        e += par[5 * j + 4] * 4.0 * eps * s * (s - 1.0);
                    }
                    total += e;
                }
            }
        }
    }
    out[i] = total;
}

extern "C" __global__
void poses_with_waters(const double* waters, const double* site_par, const double* poses, const double* box,
                       const double* inv, const double radius, const double k_rf, const double c_rf,
                       const double coulomb, const int n_w, const int n_s, const int n_p, const int skip,
                       double* out)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_w * n_p) return;
    const int p = t / n_w, w = t % n_w;
    double total = 0.0;
    if (w != skip) {
        const double r2max = radius * radius;
        for (int a = 0; a < n_s; ++a) {
            const double* pa = poses + 3 * (p * n_s + a);
            for (int b = 0; b < n_s; ++b) {
                const double* wb = waters + 3 * (w * n_s + b);
                const double ux = wb[0] - pa[0], uy = wb[1] - pa[1], uz = wb[2] - pa[2];
                double dx = ux * inv[0] + uy * inv[3] + uz * inv[6];
                double dy = ux * inv[1] + uy * inv[4] + uz * inv[7];
                double dz = ux * inv[2] + uy * inv[5] + uz * inv[8];
                dx -= rint(dx); dy -= rint(dy); dz -= rint(dz);
                const double x = dx * box[0] + dy * box[3] + dz * box[6];
                const double y = dx * box[1] + dy * box[4] + dz * box[7];
                const double z = dx * box[2] + dy * box[5] + dz * box[8];
                const double r2 = x * x + y * y + z * z;
                if (r2 > r2max || r2 <= 0.0) continue;
                const double inv_r = 1.0 / sqrt(r2);
                double e = coulomb * site_par[3 * a] * site_par[3 * b] * (inv_r + k_rf * r2 - c_rf);
                const double eps = site_par[3 * a + 2] * site_par[3 * b + 2];
                if (eps > 0.0) {
                    const double sig = 0.5 * (site_par[3 * a + 1] + site_par[3 * b + 1]);
                    const double q = sig * sig * inv_r * inv_r;
                    const double s = q * q * q;
                    e += 4.0 * eps * s * (s - 1.0);
                }
                total += e;
            }
        }
    }
    out[t] = total;
}
"""


class GpuStandIn:
    """The numbers of :class:`StandIn` that a proposal needs, computed on a CUDA device with CuPy:
    the same energy function in double precision, the atoms found through a grid of cells instead
    of a tree. One object per mover; :meth:`load` gives it the configuration of an attempt."""

    #: cells of the grid per radius, and how far (in cells) the atoms within the radius can be
    REACH = 2
    #: what is added to the radius for the first measure of a pair, in single precision (nm)
    NEAR_MARGIN_NM = 1e-3

    def __init__(self, spec: RelocationSpec, device: int) -> None:
        import cupy as cp

        self.cp, self.spec = cp, spec
        self.device = cp.cuda.Device(int(device))
        # the device of the Context is the current one of the process from here on: `with self.device` returns to the
        # device that was current before, the first (0) where none was set, and CUDA 12 makes a context there (the
        # full TYK2 benchmark: every worker with the moves held one on GPU 0 too, whatever its GPU, until GPU 0 had more
        # than the 60 clients of an MPS server and the workers that started then failed: cudaErrorDevicesUnavailable)
        self.device.use()
        waters = np.asarray(spec.waters)
        self.n_w, self.n_s = waters.shape
        group = np.full(len(spec.charge), -1, dtype=np.int32)
        group[waters] = np.arange(self.n_w, dtype=np.int32)[:, None]
        first = waters[0]
        # per site of a water: charge, sigma and the square root of epsilon
        site_par = np.stack([spec.charge[first], spec.sigma[first], np.sqrt(spec.epsilon[first])],
                            axis=1).astype(np.float64)
        with self.device:
            module = cp.RawModule(code=_CUDA_SOURCE)
            self.site_kernel = module.get_function("site_energies")
            self.water_kernel = module.get_function("poses_with_waters")
            self.group = cp.asarray(group)
            self.site_par = cp.asarray(site_par)
            self.water_index = cp.asarray(waters.reshape(-1).astype(np.int64))
            self.score_par = cp.asarray(np.tile(site_par, (self.n_w, 1)))
            self.score_own = cp.asarray(np.repeat(np.arange(self.n_w, dtype=np.int32), self.n_s))
        self.parameters = {}                     # by (lambda_electrostatics, lambda_sterics): on the device

    def load(self, positions: np.ndarray, box: np.ndarray, lambda_electrostatics: float,
             lambda_sterics: float, active: np.ndarray | None = None) -> GpuStandIn:
        """The configuration and the alchemical state the next numbers are of; ``active`` as in
        :class:`StandIn`."""
        cp, spec = self.cp, self.spec
        box = np.asarray(box, dtype=np.float64)
        inverse = np.linalg.inv(box)
        key = (float(lambda_electrostatics), float(lambda_sterics))
        if key not in self.parameters:
            # per atom: charge, sigma, the square root of epsilon, the soft-core term and the factor
            # of its Lennard-Jones
            par = np.empty((len(spec.charge), 5))
            par[:, 0], par[:, 1], par[:, 2] = spec.charge, spec.sigma, np.sqrt(spec.epsilon)
            par[:, 3], par[:, 4] = 0.0, 1.0
            par[spec.ligand, 0] *= key[0]
            par[spec.ligand, 3] = float(spec.softcore_alpha) * (1.0 - key[1]) ** int(spec.softcore_b)
            par[spec.ligand, 4] = key[1]
            if len(self.parameters) >= 128:      # a worker has the states of one leg: far fewer
                self.parameters.clear()
            with self.device:
                self.parameters[key] = cp.asarray(par)
        self.heights = cell_heights(box)
        with self.device:
            self.box, self.inverse = cp.asarray(box.ravel()), cp.asarray(inverse.ravel())
            self.box32 = cp.asarray(box.ravel().astype(np.float32))
            self.positions = cp.asarray(np.ascontiguousarray(positions, dtype=np.float64))
            # x @ inverse, written out: no library but the kernels CuPy compiles itself is needed
            rows = cp.asarray(inverse)
            frac = (self.positions[:, 0:1] * rows[0] + self.positions[:, 1:2] * rows[1]
                    + self.positions[:, 2:3] * rows[2])
            self.frac = frac - cp.floor(frac)
            self.par = self.parameters[key]
            if active is not None:
                off = cp.asarray(np.flatnonzero(~np.asarray(active, dtype=bool)))
                self.par = self.par.copy()
                self.par[off, 0] = 0.0                    # no charge
                self.par[off, 2] = 0.0                    # no Lennard-Jones (the square root of epsilon)
            self.grids = {}
        return self

    def _grid(self, radius_nm: float):
        """The atoms sorted by the cells of a grid whose cells are at least ``radius_nm`` / REACH
        high: (fractional coordinates, the same in single precision, parameters, water of each
        atom, where the atoms of each cell start, cells per axis)."""
        if radius_nm not in self.grids:
            cp = self.cp
            dims = np.maximum(1, np.floor(self.heights * (self.REACH / float(radius_nm)))).astype(np.int32)
            n = cp.asarray(dims)
            cell = cp.minimum((self.frac * n).astype(cp.int32), n - 1)
            number = cell[:, 0] + int(dims[0]) * (cell[:, 1] + int(dims[1]) * cell[:, 2])
            order = cp.argsort(number)
            start = cp.searchsorted(number[order], cp.arange(int(dims.prod()) + 1, dtype=number.dtype))
            frac = cp.ascontiguousarray(self.frac[order])
            self.grids[radius_nm] = (frac, frac.astype(cp.float32), cp.ascontiguousarray(self.par[order]),
                                     cp.ascontiguousarray(self.group[order]), start.astype(cp.int32), n)
        return self.grids[radius_nm]

    def _sites(self, sites, site_par, own, radius_nm: float):
        """Energy of every site with all atoms within ``radius_nm`` but those of the water ``own``."""
        cp = self.cp
        radius = float(radius_nm)
        k_rf = (DIELECTRIC - 1.0) / ((2.0 * DIELECTRIC + 1.0) * radius ** 3)
        c_rf = 3.0 * DIELECTRIC / ((2.0 * DIELECTRIC + 1.0) * radius)
        frac, near, par, group, start, dims = self._grid(radius)
        m = int(sites.shape[0])
        out = cp.empty(m, dtype=cp.float64)
        self.site_kernel(((m + 127) // 128,), (128,),
                         (frac, near, par, group, start, dims, self.box, self.box32, self.inverse,
                          cp.ascontiguousarray(sites), cp.ascontiguousarray(site_par), own, np.float64(radius),
                          np.float32((radius + self.NEAR_MARGIN_NM) ** 2), np.int32(self.REACH), np.float64(k_rf),
                          np.float64(c_rf), np.float64(ONE_4PI_EPS0), np.int32(m), out))
        return out

    def scores(self) -> np.ndarray:
        with self.device:
            per_site = self._sites(self.positions[self.water_index], self.score_par, self.score_own, SCORE_RADIUS_NM)
            return self.cp.asnumpy(per_site.reshape(self.n_w, self.n_s).sum(axis=1))

    def pose_energies(self, poses: np.ndarray, moved: int, radius_nm: float) -> np.ndarray:
        cp = self.cp
        k = len(poses)
        with self.device:
            sites = cp.asarray(np.ascontiguousarray(poses, dtype=np.float64).reshape(-1, 3))
            own = cp.full(k * self.n_s, int(moved), dtype=cp.int32)
            per_site = self._sites(sites, cp.tile(self.site_par, (k, 1)), own, radius_nm)
            return cp.asnumpy(per_site.reshape(k, self.n_s).sum(axis=1))

    def water_energies(self, which: np.ndarray, radius_nm: float) -> np.ndarray:
        cp = self.cp
        which = np.asarray(which, dtype=np.int64)
        m = len(which)
        if not m:
            return np.zeros(0)
        with self.device:
            index = self.water_index.reshape(self.n_w, self.n_s)[cp.asarray(which)].reshape(-1)
            own = cp.asarray(np.repeat(which, self.n_s).astype(np.int32))
            per_site = self._sites(self.positions[index], cp.tile(self.site_par, (m, 1)), own, radius_nm)
            return cp.asnumpy(per_site.reshape(m, self.n_s).sum(axis=1))

    def with_waters(self, poses: np.ndarray, moved: int) -> np.ndarray:
        cp = self.cp
        k = len(poses)
        radius = float(SCORE_RADIUS_NM)
        k_rf = (DIELECTRIC - 1.0) / ((2.0 * DIELECTRIC + 1.0) * radius ** 3)
        c_rf = 3.0 * DIELECTRIC / ((2.0 * DIELECTRIC + 1.0) * radius)
        with self.device:
            out = cp.empty(k * self.n_w, dtype=cp.float64)
            self.water_kernel(((k * self.n_w + 127) // 128,), (128,),
                              (cp.ascontiguousarray(self.positions[self.water_index]), self.site_par,
                               cp.asarray(np.ascontiguousarray(poses, dtype=np.float64).reshape(-1, 3)), self.box,
                               self.inverse, np.float64(radius), np.float64(k_rf), np.float64(c_rf),
                               np.float64(ONE_4PI_EPS0), np.int32(self.n_w), np.int32(self.n_s), np.int32(k),
                               np.int32(int(moved)), out))
            return cp.asnumpy(out.reshape(k, self.n_w))


def gpu_stand_in(spec: RelocationSpec, context) -> GpuStandIn | None:
    """The stand-in on the CUDA device of ``context``, or None where there is none (another
    platform, or an environment without CuPy): the NumPy stand-in is used then, which gives the
    same numbers on the processor and is much slower."""
    if context is None:                          # a mover that only makes proposals
        return None
    platform = context.getPlatform()
    if platform.getName() != "CUDA":
        return None
    try:
        import cupy  # noqa: F401
    except ImportError:
        return None
    return GpuStandIn(spec, int(platform.getPropertyValue(context, "DeviceIndex").split(",")[0]))


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
        # where the stand-in energies are computed: the CUDA device of the Context, else the processor
        self.gpu = gpu_stand_in(spec, context)
        self.engine = "the GPU of the Context" if self.gpu is not None else "the processor"

    def _stand_in(self, x: np.ndarray, box: np.ndarray, state):
        if self.gpu is not None:
            return self.gpu.load(x, box, state.lambda_electrostatics, state.lambda_sterics)
        return StandIn(self.spec, x, box, state.lambda_electrostatics, state.lambda_sterics)

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
            outside = np.linalg.norm(image_within_half_a_cell(trial - centre, box), axis=1) >= radius
            anchors[todo[outside]] = trial[outside]
            todo = todo[~outside]
        return anchors, inner

    def centre(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
        """The centre of the sphere: the geometric centre of the heavy atoms of the ligand, the
        molecule taken whole."""
        return whole_pose(x[self.spec.ligand_heavy], self.spec.ligand_heavy_bonds, box).mean(axis=0)

    def inside(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Whether each water is in the sphere: by its oxygen."""
        distance = np.linalg.norm(image_within_half_a_cell(x[self.waters[:, 0]] - self.centre(x, box), box), axis=1)
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
        if not (np.isfinite(x).all() and np.isfinite(box).all()):
            # a configuration that is not finite (a simulation that failed): nothing is proposed from
            # it, and what the run does about such a replica is not held up here (the draw of the
            # anchors would not end: no point is at a distance from a centre that is not a number)
            return None
        check_cell(box, radius, float(spec.cutoff_nm))            # the cell as it is now
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

        energy = self._stand_in(x, box, state)
        # the scores of all waters in x: the interaction of each with everything else
        score = energy.scores()
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
            poses = anchors[:, None, :] + np.matmul(local, rotations.transpose(0, 2, 1))
        # the target region is the sphere, unless the move goes out of it
        in_target = np.asarray(anchor_inside, dtype=bool) != out
        if not in_target.any():
            return None
        # the energies of the K poses and of the pose D has, in one computation
        trial = energy.pose_energies(np.concatenate([poses, x[atoms][None]]), moved, spec.cutoff_nm)
        trial, u_own = trial[:-1], float(trial[-1])
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
        old_and_new = energy.with_waters(np.stack([x[atoms], poses[chosen]]), moved)
        changed = score[others] - old_and_new[0, others] + old_and_new[1, others]
        s_new = float(energy.pose_energies(poses[chosen][None], moved, SCORE_RADIUS_NM)[0])
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
        # (a site without a mass, the virtual site of a 4-site water, has no velocity of its own
        # and gets none)
        atoms = self.waters[move["moved"]]
        atoms = atoms[np.asarray(self.spec.mass, dtype=float)[atoms] > 0.0]
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


def check_cell(box: np.ndarray, radius_nm: float, cutoff_nm: float) -> None:
    """The sphere and the cutoff must be less than half the smallest height of the cell: else the
    nearest image of a water does not decide its region, nor that of an atom its interaction.
    Checked when a leg is built and at every attempt, with the cell of that moment (a barostat
    changes it)."""
    smallest = float(cell_heights(np.asarray(box, dtype=float)).min())
    if 2.0 * radius_nm >= smallest or 2.0 * cutoff_nm >= smallest:
        raise RelocationError(f"the sphere (radius {radius_nm:.2f} nm) or the cutoff ({cutoff_nm:g} nm) is not "
                              f"less than half the smallest height of the cell ({smallest:.2f} nm)")


def relocation_spec(system, topology, positions: np.ndarray, box: np.ndarray, ligand, cutoff_nm: float,
                    softcore_alpha: float, softcore_b: int, temperature_K: float, trial_poses: int,
                    inner_fraction: float | None, interval_ps: float, radius_margin_nm: float,
                    bonds=None) -> RelocationSpec:
    """What a mover needs, from the fully interacting ``system`` of a leg (the one the alchemical
    System is made from), its ``topology`` and the pose that was prepared (``positions``, ``box``).

    The waters that can be moved are the residues named HOH: all of them. As in the paper they
    must be alike (the same charge and Lennard-Jones parameters site by site) and must not
    interact within themselves (every nonbonded exception inside one is zero), so that the energy
    of a water with everything else is the energy of its sites with all other atoms. The sphere
    must fit into the cell: twice its radius, and twice the cutoff, less than its smallest height.
    The ligand is taken whole along the bonds of the topology and ``bonds`` (pairs of atoms of the System that the
    topology does not hold). Raises RelocationError otherwise."""
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
    bonds = heavy_bonds(topology, heavy, bonds)
    radius = sphere_radius(whole_pose(np.asarray(positions, dtype=float)[heavy], bonds, box), radius_margin_nm)
    check_cell(box, radius, float(cutoff_nm))
    if float(cutoff_nm) < SCORE_RADIUS_NM:
        raise RelocationError(f"the cutoff ({float(cutoff_nm):g} nm) is shorter than the radius of the scores "
                              f"({SCORE_RADIUS_NM:g} nm)")
    return RelocationSpec(waters=waters, ligand=ligand, ligand_heavy=heavy, radius_nm=radius,
                          charge=parameters[:, 0].copy(), sigma=parameters[:, 1].copy(),
                          epsilon=parameters[:, 2].copy(), mass=mass, cutoff_nm=float(cutoff_nm),
                          softcore_alpha=float(softcore_alpha), softcore_b=int(softcore_b),
                          temperature_K=float(temperature_K), trial_poses=int(trial_poses),
                          inner_fraction=None if inner_fraction is None else float(inner_fraction),
                          interval_ps=float(interval_ps), ligand_heavy_bonds=bonds)
