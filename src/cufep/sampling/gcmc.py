"""Grand canonical Monte Carlo of water in a sphere around the ligand (``sampling.gcmc``).

Water exchanges slowly between a buried binding site and the bulk, and the water relocation moves
of :mod:`cufep.sampling.relocation_mc`, which keep the number of waters, changed nothing on the
A2A set (plan ``docs/plans/2026-11-24-gcmc-water.md``). Here the number of waters in a sphere around
the ligand changes: the sphere is in equilibrium with a reservoir of bulk water.

The method and its sources
--------------------------
- Ghost waters (grand: Samways, Bruce Macdonald and Essex, J. Chem. Inf. Model. 2020, 60, 4436;
  its ``utils.add_ghosts``: "This is to avoid changing the number of particles throughout a
  simulation. Instead, we can just switch between 'ghost' and 'real' waters"). The System keeps its
  particles. A water that is deleted stays as a ghost: a molecule without nonbonded interactions.
  An insertion puts a ghost at a place in the sphere and gives it its interactions.
- The acceptance (Ross et al., J. Chem. Theory Comput. 2017, 13, 6373, eq 1, 2 and 7; the same in
  grand's ``samplers.py``):

      P(insert) = min[1, exp(B) exp(-beta dE) / (N + 1)]
      P(delete) = min[1, N exp(-B) exp(-beta dE)]
      B = beta mu'_sol + ln(V_sphere / V°)

  mu'_sol is the excess chemical potential of water in bulk water ("the hydration free energy of
  the simulated water model"), V° the standard state volume of bulk water, N the number of waters
  that interact and have their oxygen in the sphere (grand: every water that is not a ghost,
  counted from the positions at every call, so that water which diffused in or out during the MD
  counts). Insertion or deletion with probability 1/2 each; the oxygen of an inserted water is put
  at a point drawn uniformly in the sphere, the molecule turned at random; the water to delete is
  drawn uniformly among the N. A deletion with N = 0 is an attempt that does nothing (grand).
- Constant volume (Ross 2017: "the volume must remain constant. This is so that the simulations
  correctly sample the grand canonical (muVT) ensemble"; grand stops with "GCMC must be used at
  constant volume" if the System has a barostat): the leg that makes these moves has no barostat.
- Batches of trials with an approximate potential (the scheme of Ross et al., J. Chem. Theory
  Comput. 2020, 16, 6061, as the white paper of loch, OpenBioSim, states it; the paper itself
  could not be obtained): a batch of trials is evaluated at once from one configuration, each is
  accepted or not by the rule above, the first accepted one is applied, the rest of the batch is
  thrown away, and the count of attempts goes on from the trial that was applied. The trials are
  evaluated with an approximate potential U' and the one that passed is tested again with the full
  potential U of the simulation, "min(1, exp[-beta((U(j) - U(i)) - (U'(j) - U'(i))])" (loch, after
  Gelb, J. Chem. Phys. 2003, 118, 7747); "If a candidate is found to fail the full potential
  correction criterion, the process continues to the next candidate from the batch". Each trial is
  then one Metropolis step with the full potential in two stages, and what U' is changes how many
  trials pass, not what is sampled.

cuFEP's own (the sources are silent or do it otherwise)
-------------------------------------------------------
- U' is the stand-in energy of :mod:`cufep.sampling.relocation_mc` (Lennard-Jones and
  reaction-field Coulomb within the cutoff, the ligand in its alchemical state, on the GPU of the
  Context), U the potential energy of the Context in the state the replica is simulated in.
- The ghosts are waters of the prepared system: those whose oxygen is farthest from the ligand
  (grand adds molecules; here the prepared system stays as it is, and the box settles without
  them in the pre-equilibration, which runs with a barostat).
- A water with a virtual site (a 4-site model) is switched and placed as one body with it (grand
  takes TIP3P only).
- An inserted water keeps the velocities it had as a ghost (grand's sampler sets positions only;
  a ghost is integrated and thermostatted like every other molecule).
- The sphere is that of the water relocation moves: centred on the geometric centre of the heavy
  atoms of the ligand, radius half the largest distance between two of them in the prepared pose
  plus ``radius_margin_nm``.
- With no ghost left the calculation stops with the reason, as grand does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cufep.alchemy.lrc import nearest_image
from cufep.sampling.relocation_mc import (
    KB_KJ_PER_MOL_K,
    RelocationError,
    StandIn,
    check_cell,
    gpu_stand_in,
    image_within_half_a_cell,
    random_rotations,
    relocation_spec,
)

#: what a move reports, in this order in the samples of a leg (``gcmc``): trials of an insertion,
#: insertions made, trials of a deletion, deletions made, waters in the sphere after the move
COUNTS = ("insertions_attempted", "insertions_accepted", "deletions_attempted", "deletions_accepted",
          "waters_in_the_sphere")


class GcmcError(RelocationError):
    """The moves cannot be made in this system as they are asked for."""


@dataclass
class GcmcSpec:
    """What a mover needs of a leg (picklable; made by the workflow from the prepared system).

    The fields the stand-in energy reads have the names of
    :class:`cufep.sampling.relocation_mc.RelocationSpec`; ``charge``, ``sigma`` and ``epsilon`` are
    those of the system with every water interacting."""

    waters: np.ndarray            # (number of waters, sites per water) atom indices, the oxygen first
    ligand: np.ndarray
    ligand_heavy: np.ndarray
    radius_nm: float
    charge: np.ndarray            # per atom (e)
    sigma: np.ndarray             # nm
    epsilon: np.ndarray           # kJ/mol
    mass: np.ndarray              # amu (0 for a virtual site)
    cutoff_nm: float
    softcore_alpha: float
    softcore_b: int
    temperature_K: float
    interval_ps: float
    attempts: int                 # trials of one move
    batch_size: int               # trials evaluated at once
    excess_chemical_potential_kj_per_mol: float
    standard_volume_nm3: float
    ghosts: np.ndarray            # the waters (rows of ``waters``) that are ghosts at the start
    #: (lambda_electrostatics, lambda_sterics) of the ligand in the stand-in energy U' of every state; None: those of
    #: the state (an ABFE leg). A relative leg has no such pair: its ligand interacts in U' with the parameters of
    #: both end states at full strength, which the test with the full potential corrects (cuFEP's own)
    stand_in_lambdas: tuple[float, float] | None = None
    ligand_heavy_bonds: np.ndarray | None = None   # the bonds between the heavy atoms (pairs of positions in them)

    @property
    def adams(self) -> float:
        """B = beta mu'_sol + ln(V_sphere / V°)."""
        volume = 4.0 / 3.0 * np.pi * float(self.radius_nm) ** 3
        return (float(self.excess_chemical_potential_kj_per_mol) / (KB_KJ_PER_MOL_K * float(self.temperature_K))
                + float(np.log(volume / float(self.standard_volume_nm3))))

    def initial_on(self) -> np.ndarray:
        """Which waters interact at the start (per water)."""
        on = np.ones(len(self.waters), dtype=bool)
        on[np.asarray(self.ghosts, dtype=np.int64)] = False
        return on


def farthest_waters(waters: np.ndarray, positions: np.ndarray, box: np.ndarray, ligand_heavy: np.ndarray,
                    n: int, bonds=None) -> np.ndarray:
    """The ``n`` waters whose oxygen is farthest from the geometric centre of the heavy atoms of the
    ligand (the molecule whole along ``bonds``, nearest image), in ascending order of their number: the ghosts of the
    start."""
    x, box = np.asarray(positions, dtype=float), np.asarray(box, dtype=float)
    centre = _centre(x, box, ligand_heavy, bonds)
    distance = np.linalg.norm(nearest_image(x[np.asarray(waters)[:, 0]] - centre, box), axis=1)
    if int(n) > len(distance):
        raise GcmcError(f"{int(n)} ghost waters are asked for and the system has {len(distance)} waters")
    order = np.lexsort((np.arange(len(distance)), -distance))        # farthest first, ties by number
    return np.sort(order[:int(n)])


def epsilon_parameters() -> dict[str, tuple[str, ...]]:
    """The forces besides the NonbondedForce that hold the Lennard-Jones of the atoms with the ligand, by name, and
    their per-particle parameters of epsilon: the soft-core force of the ABFE (:mod:`cufep.alchemy.factory`) and the
    Lennard-Jones of the hybrid ligand with its environment (:mod:`cufep.rbfe.hybrid`, epsilon at either end)."""
    from cufep.alchemy.factory import SOFTCORE_FORCE_NAME
    from cufep.rbfe.hybrid import HYBRID_LJ_FORCE_NAME

    return {SOFTCORE_FORCE_NAME: ("epsilon",), HYBRID_LJ_FORCE_NAME: ("epsilonA", "epsilonB")}


def lennard_jones_forces(system) -> list[tuple]:
    """(force, indices of its epsilon parameters) of every force of :func:`epsilon_parameters` in ``system``."""
    import openmm as mm

    known, out = epsilon_parameters(), []
    for force in system.getForces():
        if isinstance(force, mm.CustomNonbondedForce) and force.getName() in known:
            names = [force.getPerParticleParameterName(k) for k in range(force.getNumPerParticleParameters())]
            out.append((force, [names.index(n) for n in known[force.getName()]]))
    return out


def ghosts_off(system, atoms) -> None:
    """Take the nonbonded interactions of ``atoms`` out of ``system`` (changed in place): charge 0
    and epsilon 0 in its NonbondedForce, and epsilon 0 in the forces of :func:`lennard_jones_forces` (the hybrid
    System of a relative leg has one). For the System of a leg with the moves before anything is made of it, so
    that its plain MD, its alchemical System and its tail correction all have the ghosts of the start."""
    import openmm as mm

    force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    for i in (int(a) for a in atoms):
        charge, sigma, epsilon = force.getParticleParameters(i)
        force.setParticleParameters(i, charge * 0.0, sigma, epsilon * 0.0)
    for other, indices in lennard_jones_forces(system):
        for i in (int(a) for a in atoms):
            values = list(other.getParticleParameters(i))
            for k in indices:
                values[k] = 0.0
            other.setParticleParameters(i, values)


def every_water_on(system, spec: GcmcSpec) -> None:
    """Give every water of ``system`` (changed in place) the nonbonded parameters it has when it
    interacts: the reverse of :func:`ghosts_off`, for what is made of the system with all its waters
    (the tail correction, which leaves out the ghosts of each configuration itself)."""
    import openmm as mm

    force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    for i in (int(a) for a in np.asarray(spec.waters).reshape(-1)):
        force.setParticleParameters(i, float(spec.charge[i]), float(spec.sigma[i]), float(spec.epsilon[i]))
    for other, indices in lennard_jones_forces(system):
        for i in (int(a) for a in np.asarray(spec.waters).reshape(-1)):
            values = list(other.getParticleParameters(i))
            for k in indices:
                values[k] = float(spec.epsilon[i])
            other.setParticleParameters(i, values)


def gcmc_spec(system, topology, positions: np.ndarray, box: np.ndarray, ligand, cutoff_nm: float,
              softcore_alpha: float, softcore_b: int, temperature_K: float, interval_ps: float, attempts: int,
              batch_size: int, radius_margin_nm: float, excess_chemical_potential_kcal_per_mol: float,
              standard_volume_nm3: float, ghost_waters: int | None,
              radius_from_nm: np.ndarray | None = None, radius_from_bonds=None, bonds=None) -> GcmcSpec:
    """What a mover needs, from the ``system`` of a leg with every water interacting, its
    ``topology`` and the pose that was prepared. The waters and the sphere are found as for the
    water relocation moves (:func:`cufep.sampling.relocation_mc.relocation_spec`, with its
    conditions). ``ghost_waters`` None: as many as the sphere would hold if it were all bulk water
    (its volume by the standard volume, rounded up; cuFEP's own: the sources give examples of 10 to
    50 and no rule). ``radius_from_nm``: the heavy atoms whose largest distance gives the radius, where they are not
    those of the ligand in ``positions`` (the two end states of a relative leg), with ``radius_from_bonds`` the bonds
    between them (pairs of positions in them), each molecule taken whole (:func:`whole_pose`). ``bonds``: pairs of
    atoms of the System bonded that the topology does not hold (of a hybrid: the bonds of both end states), for the
    ligand taken whole in the check of the cell and in the moves. Raises GcmcError where the moves cannot be made."""
    try:
        base = relocation_spec(system, topology, positions, box, ligand, cutoff_nm, softcore_alpha, softcore_b,
                               temperature_K, 1, None, interval_ps, radius_margin_nm, bonds)
    except GcmcError:
        raise
    except RelocationError as exc:
        raise GcmcError(str(exc)) from exc
    if radius_from_nm is not None:
        import dataclasses

        from cufep.sampling.relocation_mc import sphere_radius, whole_pose

        pose = whole_pose(np.asarray(radius_from_nm, dtype=float), radius_from_bonds, np.asarray(box, dtype=float))
        base = dataclasses.replace(base, radius_nm=sphere_radius(pose, radius_margin_nm))
        check_cell(np.asarray(box, dtype=float), base.radius_nm, float(cutoff_nm))
    if int(attempts) < 1 or int(batch_size) < 1:
        raise GcmcError("attempts and batch_size are at least 1")
    if not float(standard_volume_nm3) > 0.0:
        raise GcmcError("the standard volume is a positive number")
    if ghost_waters is None:
        ghost_waters = int(np.ceil(4.0 / 3.0 * np.pi * base.radius_nm ** 3 / float(standard_volume_nm3)))
    if int(ghost_waters) < 1:
        raise GcmcError("at least one ghost water is needed")
    ghosts = farthest_waters(base.waters, positions, box, base.ligand_heavy, int(ghost_waters), base.ligand_heavy_bonds)
    inside = np.linalg.norm(
        nearest_image(np.asarray(positions, dtype=float)[base.waters[ghosts, 0]]
                      - _centre(np.asarray(positions, dtype=float), np.asarray(box, dtype=float), base.ligand_heavy,
                                base.ligand_heavy_bonds),
                      np.asarray(box, dtype=float)), axis=1) < base.radius_nm
    if inside.any():
        raise GcmcError(f"{int(ghost_waters)} ghost waters would take waters of the sphere itself")
    return GcmcSpec(waters=base.waters, ligand=base.ligand, ligand_heavy=base.ligand_heavy, radius_nm=base.radius_nm,
                    charge=base.charge, sigma=base.sigma, epsilon=base.epsilon, mass=base.mass,
                    cutoff_nm=base.cutoff_nm, softcore_alpha=base.softcore_alpha, softcore_b=base.softcore_b,
                    temperature_K=base.temperature_K, interval_ps=float(interval_ps), attempts=int(attempts),
                    batch_size=int(batch_size),
                    excess_chemical_potential_kj_per_mol=4.184 * float(excess_chemical_potential_kcal_per_mol),
                    standard_volume_nm3=float(standard_volume_nm3), ghosts=ghosts,
                    ligand_heavy_bonds=base.ligand_heavy_bonds)


def _centre(x: np.ndarray, box: np.ndarray, ligand_heavy: np.ndarray, bonds=None) -> np.ndarray:
    """The geometric centre of the heavy atoms of the ligand, the molecule whole along ``bonds``."""
    from cufep.sampling.relocation_mc import whole_pose

    return whole_pose(x[np.asarray(ligand_heavy)], bonds, box).mean(axis=0)


class WaterSwitch:
    """Switches waters off and on in a live Context: for every site of the water, charge and
    epsilon in the force that holds the nonbonded parameters of all atoms (the stock
    NonbondedForce, or the cuPotential plugin's force that took its place) and epsilon in the
    forces of :func:`lennard_jones_forces` (the soft-core force of the ligand of the ABFE, the
    Lennard-Jones of the hybrid ligand of a relative leg), if the System has one; then
    ``updateParametersInContext``.

    A Context that was switched so gives the energies and forces of a System built so, on both
    backends (``spikes/m33_ghost_waters_on_the_backends.py``, the tests). ``on`` is what the
    Context has when this object is made: the System it was made of."""

    def __init__(self, context, spec: GcmcSpec, on: np.ndarray | None = None) -> None:
        self.context, self.waters = context, np.asarray(spec.waters, dtype=np.int64)
        system = context.getSystem()
        forces = [system.getForce(i) for i in range(system.getNumForces())]
        found = [f for f in forces if type(f).__name__ == "NonbondedForce"]
        if len(found) != 1:
            raise GcmcError(f"the System has {len(found)} forces with the nonbonded parameters of the atoms, not one")
        self.nonbonded = found[0]
        self.others = lennard_jones_forces(system)
        self.charge, self.sigma, self.epsilon = (np.asarray(v, dtype=float) for v in
                                                 (spec.charge, spec.sigma, spec.epsilon))
        self.on = spec.initial_on() if on is None else np.array(on, dtype=bool)

    def apply(self, on: np.ndarray) -> int:
        """Make the Context have the waters ``on`` (per water) and no other; returns how many were
        switched."""
        on = np.asarray(on, dtype=bool)
        changed = np.flatnonzero(on != self.on)
        if not len(changed):
            return 0
        for water in changed:
            keep = 1.0 if on[water] else 0.0
            for i in (int(a) for a in self.waters[water]):
                self.nonbonded.setParticleParameters(i, keep * self.charge[i], self.sigma[i], keep * self.epsilon[i])
                for other, indices in self.others:
                    values = list(other.getParticleParameters(i))
                    for k in indices:
                        values[k] = keep * self.epsilon[i]
                    other.setParticleParameters(i, values)
        self.nonbonded.updateParametersInContext(self.context)
        for other, _indices in self.others:
            other.updateParametersInContext(self.context)
        self.on = on.copy()
        return len(changed)


class GcmcMover:
    """Makes the moves in one OpenMM Context (the module's docstring). The Context is one whose
    System has the ghosts of the start switched off (:func:`ghosts_off`); which waters interact in
    it at any time is ``on`` (per water), which a worker that loads another replica sets with
    :meth:`set_on`."""

    def __init__(self, spec: GcmcSpec, context, seed: int) -> None:
        self.spec, self.context = spec, context
        self.rng = np.random.default_rng(int(seed))
        self.beta = 1.0 / (KB_KJ_PER_MOL_K * float(spec.temperature_K))
        self.waters = np.asarray(spec.waters, dtype=np.int64)
        self.b = spec.adams
        self.switch = None if context is None else WaterSwitch(context, spec)
        self._on = spec.initial_on()
        self.gpu = gpu_stand_in(spec, context)
        self.engine = "the GPU of the Context" if self.gpu is not None else "the processor"

    # -- which waters interact ---------------------------------------------------------------------
    @property
    def on(self) -> np.ndarray:
        return self._on.copy()

    def set_on(self, on: np.ndarray) -> None:
        on = np.asarray(on, dtype=bool)
        if on.shape != self._on.shape:
            raise GcmcError("the list of the waters that interact does not fit this system")
        if self.switch is not None:
            self.switch.apply(on)
        self._on = on.copy()

    # -- the configuration -------------------------------------------------------------------------
    def centre(self, x: np.ndarray, box: np.ndarray) -> np.ndarray:
        return _centre(x, box, self.spec.ligand_heavy, self.spec.ligand_heavy_bonds)

    def in_sphere(self, x: np.ndarray, box: np.ndarray, on: np.ndarray | None = None) -> np.ndarray:
        """The waters that interact and have their oxygen in the sphere (their numbers)."""
        on = self._on if on is None else on
        distance = np.linalg.norm(image_within_half_a_cell(x[self.waters[:, 0]] - self.centre(x, box), box), axis=1)
        return np.flatnonzero(on & (distance < float(self.spec.radius_nm)))

    def _potential(self) -> float:
        import openmm.unit as u

        return float(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))

    def _stand_in(self, x: np.ndarray, box: np.ndarray, state, on: np.ndarray):
        active = np.ones(len(self.spec.charge), dtype=bool)
        active[self.waters[~on].reshape(-1)] = False
        fixed = self.spec.stand_in_lambdas
        electrostatics, sterics = (state.lambda_electrostatics, state.lambda_sterics) if fixed is None else fixed
        if self.gpu is not None:
            return self.gpu.load(x, box, electrostatics, sterics, active=active)
        return StandIn(self.spec, x, box, electrostatics, sterics, active=active)

    # -- one batch of trials -------------------------------------------------------------------------
    def configuration(self, x: np.ndarray, box: np.ndarray, state, on: np.ndarray) -> dict:
        """What every batch of trials from one configuration needs, made once for all of them: the
        stand-in energy of the configuration with the waters ``on``, the centre of the sphere, the
        waters in it (``inside``) with the stand-in energy each has with everything else
        (``u_water``: what a deletion takes away), and the ghosts."""
        inside = self.in_sphere(x, box, on)
        energy = self._stand_in(x, box, state, on)
        return {"x": x, "on": on, "energy": energy, "centre": self.centre(x, box), "inside": inside,
                "u_water": energy.water_energies(inside, self.spec.cutoff_nm) if len(inside) else np.zeros(0),
                "ghosts": np.flatnonzero(~on)}

    def trials(self, configuration: dict, k: int) -> dict:
        """``k`` trials from a configuration (:meth:`configuration`), each with the logarithm of its
        acceptance by the stand-in energy: ``insert`` (per trial), ``log_p``, ``u`` (the change of
        the stand-in energy, kJ/mol), ``ghost`` (the water an insertion switches on), ``poses`` (of
        its sites, per insertion), ``delete`` (the water of each deletion; -1 where the sphere is
        empty) and ``uniform`` (the number each is accepted by)."""
        spec, rng = self.spec, self.rng
        x, inside = configuration["x"], configuration["inside"]
        n = len(inside)
        insert = rng.integers(2, size=k) == 1
        n_in = int(insert.sum())
        out = {"insert": insert, "log_p": np.full(k, -np.inf), "u": np.zeros(k), "n": n, "ghost": -1,
               "poses": np.zeros((0, self.waters.shape[1], 3)), "delete": np.full(k, -1, dtype=np.int64)}
        if n_in:
            ghosts = configuration["ghosts"]
            if not len(ghosts):
                raise GcmcError(
                    "no ghost water is left, so no water can be inserted: the calculation needs more of them "
                    "(sampling.gcmc.ghost_waters)")
            ghost = int(rng.choice(ghosts))
            atoms = self.waters[ghost]
            local = x[atoms] - x[atoms[0]]                      # the molecule as it is, about its oxygen
            direction = rng.normal(size=(n_in, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            points = configuration["centre"] + direction * (
                float(spec.radius_nm) * rng.random(n_in) ** (1.0 / 3.0))[:, None]
            poses = points[:, None, :] + np.einsum("kij,sj->ksi", random_rotations(rng, n_in), local)
            u_in = configuration["energy"].pose_energies(poses, ghost, spec.cutoff_nm)
            out["ghost"], out["poses"] = ghost, poses
            out["u"][insert] = u_in
            out["log_p"][insert] = self.b - np.log(n + 1.0) - self.beta * u_in
        n_out = k - n_in
        if n_out and n:
            pick = rng.integers(n, size=n_out)
            u_water = configuration["u_water"]
            out["delete"][~insert] = inside[pick]
            out["u"][~insert] = -u_water[pick]
            out["log_p"][~insert] = np.log(float(n)) - self.b + self.beta * u_water[pick]
        out["uniform"] = rng.random(k)
        return out

    def attempt(self, state) -> dict:
        """One move on the configuration of the Context, in the alchemical state ``state`` (which
        the Context is set to): ``spec.attempts`` trials in batches. Returns the counts
        (:data:`COUNTS`). The Context and :attr:`on` are changed by every trial that is accepted.

        The batches of a move are made from the same configuration until a trial is accepted, so
        what does not depend on the trials (:meth:`configuration`) is made once and again only
        after an accepted trial."""
        import openmm.unit as u

        spec = self.spec
        st = self.context.getState(getPositions=True)
        x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        box = np.asarray(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        counts = dict.fromkeys(COUNTS, 0)
        if not (np.isfinite(x).all() and np.isfinite(box).all()):
            return counts                                       # a simulation that failed: the run says so
        check_cell(box, float(spec.radius_nm), float(spec.cutoff_nm))
        on, energy_x, done, configuration = self._on.copy(), None, 0, None
        # a trial that passes the first test is put into the Context for the second one. If it is not
        # accepted the Context is not put back at once: the next such trial replaces it, and the end of the
        # move leaves the Context with the waters and positions of the configuration that was reached
        displaced = False
        try:
            while done < int(spec.attempts):
                k = min(int(spec.batch_size), int(spec.attempts) - done)
                if configuration is None:
                    configuration = self.configuration(x, box, state, on)
                batch = self.trials(configuration, k)
                passed = np.flatnonzero(np.log(batch["uniform"]) < batch["log_p"])
                used, insertion = k, 0
                for j in passed:
                    if energy_x is None:
                        energy_x = self._potential()
                    new_on, y = on.copy(), x
                    if batch["insert"][j]:
                        new_on[batch["ghost"]] = True
                        y = x.copy()
                        y[self.waters[batch["ghost"]]] = batch["poses"][int(batch["insert"][:j].sum())]
                        self.context.setPositions(y)
                        displaced = True
                    else:
                        new_on[batch["delete"][j]] = False
                    self.switch.apply(new_on)
                    energy_y = self._potential()
                    log_r = -self.beta * ((energy_y - energy_x) - batch["u"][j])
                    if np.isfinite(log_r) and np.log(self.rng.random()) < log_r:
                        displaced = displaced and not batch["insert"][j]
                        on, x, energy_x, configuration = new_on, y, energy_y, None
                        self._on = on.copy()
                        used, insertion = int(j) + 1, 1 if batch["insert"][j] else -1
                        break
                tried = batch["insert"][:used]
                counts["insertions_attempted"] += int(tried.sum())
                counts["deletions_attempted"] += int(used - tried.sum())
                counts["insertions_accepted"] += int(insertion == 1)
                counts["deletions_accepted"] += int(insertion == -1)
                done += used
        finally:
            self.switch.apply(on)
            if displaced:
                self.context.setPositions(x)
        counts["waters_in_the_sphere"] = len(self.in_sphere(x, box, on))
        return counts
