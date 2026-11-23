"""Water relocation moves (``sampling.relocation_mc``; RelocationMC of Ding et al., arXiv:2603.22274
v3): the stand-in energy against OpenMM, the acceptance ratio against its own reverse, and the
distribution the moves keep against a known answer."""
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from cufep.sampling.relocation_mc import (
    KB_KJ_PER_MOL_K,
    SCORE_RADIUS_NM,
    RelocationError,
    RelocationMover,
    RelocationSpec,
    StandIn,
    attempts_due,
    random_rotations,
    sphere_radius,
)
from cufep.sampling.states import AlchemicalState

SHEAR = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.4, 0.0]])


def nonbonded(system) -> np.ndarray:
    """(charge, sigma, epsilon) per particle of the NonbondedForce of ``system``."""
    force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    units = (u.elementary_charge, u.nanometer, u.kilojoule_per_mole)
    return np.array([[value.value_in_unit(unit) for value, unit in zip(force.getParticleParameters(i), units)]
                     for i in range(system.getNumParticles())])


def water_spec(system, ligand, radius_nm: float = 0.45, **more) -> RelocationSpec:
    """The specification of the moves for a box of three-site waters one of which is the ligand."""
    parameters = nonbonded(system)
    n = system.getNumParticles()
    waters = np.array([[i, i + 1, i + 2] for i in range(0, n, 3) if i not in ligand])
    mass = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(n)])
    values = dict(waters=waters, ligand=np.array(ligand), ligand_heavy=np.array(ligand[:1]), radius_nm=radius_nm,
                  charge=parameters[:, 0], sigma=parameters[:, 1], epsilon=parameters[:, 2], mass=mass, cutoff_nm=0.9,
                  softcore_alpha=0.5, softcore_b=1, temperature_K=300.0, trial_poses=64, inner_fraction=0.5,
                  interval_ps=4.0)
    return RelocationSpec(**{**values, **more})


def reaction_field_energy(system, positions, box, cutoff_nm: float, silent=()) -> float:
    """Potential energy (kJ/mol) of the nonbonded interactions of ``system`` with a reaction field
    of dielectric 78.5 at ``cutoff_nm``; the atoms ``silent`` have no charge and no Lennard-Jones."""
    source = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    copy, force = mm.System(), mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm)
    force.setReactionFieldDielectric(78.5)
    force.setUseDispersionCorrection(False)
    for i in range(system.getNumParticles()):
        copy.addParticle(1.0)
        q, sigma, epsilon = source.getParticleParameters(i)
        force.addParticle(q * 0.0 if i in silent else q, sigma, epsilon * 0.0 if i in silent else epsilon)
    for k in range(source.getNumExceptions()):
        i, j, qq, sigma, epsilon = source.getExceptionParameters(k)
        force.addException(i, j, qq, sigma, epsilon)
    copy.addForce(force)
    context = mm.Context(copy, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    context.setPositions(positions)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


@pytest.mark.parametrize("cell", ["cubic", "triclinic"])
def test_the_stand_in_energy_is_the_reaction_field_energy_of_openmm(water_ligand_system, cell):
    """The stand-in energy of a water with everything else (Lennard-Jones and reaction-field
    Coulomb within a radius, minimum image) is what OpenMM gives for the same water in the same
    place: the energy of the system minus the energy with that water made silent. In a cubic and
    in a triclinic cell, for a water where it is and moved to another pose, at the cutoff and at
    the radius of the scores."""
    system, pos, box, ligand = water_ligand_system
    box = box + SHEAR if cell == "triclinic" else box
    spec = water_spec(system, ligand)
    rng = np.random.default_rng(7)
    for water in (3, 40, 111):
        atoms = spec.waters[water]
        q, sigma, epsilon = spec.charge[atoms], spec.sigma[atoms], spec.epsilon[atoms]
        for moved in (False, True):
            x = pos.copy()
            if moved:
                local = x[atoms] - x[atoms[0]]
                x[atoms] = rng.random(3) @ box + local @ random_rotations(rng, 1)[0].T
            energy = StandIn(spec, x, box, 1.0, 1.0)
            energy.set_groups(len(x), spec.waters)
            for radius in (spec.cutoff_nm, SCORE_RADIUS_NM):
                expected = reaction_field_energy(system, x, box, radius) \
                    - reaction_field_energy(system, x, box, radius, silent=set(atoms.tolist()))
                skipped = energy.energies(x[atoms], q, sigma, epsilon, radius, skip=atoms).sum()
                own = energy.energies(x[atoms], q, sigma, epsilon, radius, own=np.full(3, water)).sum()
                assert skipped == pytest.approx(expected, rel=1e-9, abs=1e-6)
                assert own == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_the_stand_in_follows_the_alchemical_state_of_the_ligand(water_ligand_system):
    """The ligand is in the stand-in with its charges times lambda_electrostatics and with the
    soft-core of the leg at lambda_sterics (x = 1 / (alpha (1 - lambda)^b + (r / sigma)^6), times
    lambda): the energy of a water next to it is the sum over its pairs by that rule, and a ligand
    that is decoupled is not there."""
    system, pos, box, ligand = water_ligand_system
    spec = water_spec(system, ligand, softcore_b=2)
    centre = pos[ligand[0]]
    distance = np.linalg.norm(pos[spec.waters[:, 0]] - centre - np.round((pos[spec.waters[:, 0]] - centre) / 2.0) * 2.0,
                              axis=1)
    atoms = spec.waters[int(np.argmin(distance))]                    # the water nearest to the ligand
    q, sigma, epsilon = spec.charge[atoms], spec.sigma[atoms], spec.epsilon[atoms]

    def energy(lambda_electrostatics, lambda_sterics, without_ligand=False):
        used = water_spec(system, ligand, softcore_b=2)
        if without_ligand:
            used.charge, used.epsilon = used.charge.copy(), used.epsilon.copy()
            used.charge[list(ligand)] = 0.0
            used.epsilon[list(ligand)] = 0.0
        return StandIn(used, pos, box, lambda_electrostatics, lambda_sterics) \
            .energies(pos[atoms], q, sigma, epsilon, 0.9, skip=atoms).sum()

    rest = energy(1.0, 1.0, without_ligand=True)
    assert energy(0.0, 0.0) == pytest.approx(rest, abs=1e-9)          # decoupled: not there
    le, ls = 0.6, 0.3
    expected = rest
    k_rf, c_rf = (78.5 - 1.0) / ((2 * 78.5 + 1.0) * 0.9 ** 3), 3 * 78.5 / ((2 * 78.5 + 1.0) * 0.9)
    for i in atoms:
        for j in ligand:
            d = pos[i] - pos[j]
            r = float(np.linalg.norm(d - np.round(d / 2.0) * 2.0))
            if r >= 0.9:
                continue
            expected += 138.93545764438198 * spec.charge[i] * le * spec.charge[j] * (1 / r + k_rf * r * r - c_rf)
            eps = np.sqrt(spec.epsilon[i] * spec.epsilon[j])
            if eps > 0:
                x = 1.0 / (0.5 * (1 - ls) ** 2 + (r / (0.5 * (spec.sigma[i] + spec.sigma[j]))) ** 6)
                expected += ls * 4 * eps * x * (x - 1)
    assert energy(le, ls) == pytest.approx(expected, rel=1e-10)
    assert abs(energy(le, ls) - rest) > 1.0                           # (the ligand is felt)


@pytest.mark.parametrize("first", ["in", "out"])
@pytest.mark.parametrize("inner_fraction", [None, 0.5])
def test_the_ratio_of_a_move_is_the_inverse_of_that_of_the_move_back(water_ligand_system, first, inner_fraction):
    """A move and the move back: the water returned to the pose it had, from the pool of the same
    trial poses with the chosen one replaced by that pose. The proposal ratio of the one is the
    inverse of that of the other, to rounding, with water that interacts, a ligand at intermediate
    lambda values, the squared soft-core, anchors uniform in the cell (the paper) and anchors half
    of which are drawn in the sphere. That holds only if every term of the ratio is the one the
    reverse move computes from the proposed configuration: the scores of the target region after
    the move, the weights of the reverse pool, the densities of the anchors."""
    system, pos, box, ligand = water_ligand_system
    spec = water_spec(system, ligand, softcore_b=2, trial_poses=256, inner_fraction=inner_fraction)
    mover = RelocationMover(spec, None, seed=11)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=0.6, lambda_sterics=0.4)
    out = first == "out"
    assert 2 <= mover.inside(pos, box).sum() < len(spec.waters)
    forward = mover.proposal(pos, box, state, out)
    assert forward is not None and forward["anchor_inside"][forward["chosen"]] != out
    atoms = spec.waters[forward["moved"]]
    assert not np.allclose(forward["y"][atoms], pos[atoms]) and mover.inside(forward["y"], box)[forward["moved"]] != out
    others = np.setdiff1d(np.arange(len(pos)), atoms)
    assert np.array_equal(forward["y"][others], pos[others])
    poses, anchor_inside = forward["poses"].copy(), forward["anchor_inside"].copy()
    poses[forward["chosen"]] = pos[atoms]
    anchor_inside[forward["chosen"]] = out                 # the pose the water had: in the region it left
    back = mover.proposal(forward["y"], box, state, not out, moved=forward["moved"], poses=poses,
                          anchor_inside=anchor_inside, chosen=forward["chosen"])
    assert np.allclose(back["y"], pos, atol=1e-12)
    assert back["log_q_ratio"] == pytest.approx(-forward["log_q_ratio"], abs=1e-8)
    assert abs(forward["log_q_ratio"]) > 1e-3


def ideal_waters(n_waters: int, edge: float, well_kt: float, radius: float, seed: int):
    """``n_waters`` rigid three-site molecules without any interaction in a cubic cell, and one
    particle at its centre that is the ligand; with ``well_kt``, an oxygen within ``radius`` of
    that particle has an energy of that many kT less (300 K). Returns (system, positions, box)."""
    rng = np.random.default_rng(seed)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0), mm.Vec3(0, 0, edge))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9)
    centre = np.full(3, 0.5 * edge)
    positions = [centre]
    system.addParticle(12.0)
    nb.addParticle(0.0, 0.3, 0.0)
    shape = np.array([[0.0, 0.0, 0.0], [0.0957, 0.0, 0.0], [-0.024, 0.0927, 0.0]])
    for _ in range(n_waters):
        while True:                                        # all waters start outside the sphere
            anchor = rng.random(3) * edge
            if np.linalg.norm(anchor - centre) > radius + 0.05:
                break
        positions.extend(anchor + shape @ random_rotations(rng, 1)[0].T)
        for mass in (15.999, 1.008, 1.008):
            system.addParticle(mass)
            nb.addParticle(0.0, 0.3, 0.0)
    system.addForce(nb)
    if well_kt:
        well = mm.CustomExternalForce(f"-{well_kt * KB_KJ_PER_MOL_K * 300.0}*step({radius}-periodicdistance(x,y,z,"
                                      f"{centre[0]},{centre[1]},{centre[2]}))")
        for k in range(n_waters):
            well.addParticle(1 + 3 * k, [])
        system.addForce(well)
    return system, np.array(positions), np.eye(3) * edge


@pytest.mark.parametrize("inner_fraction", [None, 0.5])
@pytest.mark.parametrize("well_kt", [0.0, 2.0])
def test_the_moves_keep_the_distribution_of_waters_between_the_sphere_and_the_rest(inner_fraction, well_kt):
    """Twenty waters that interact with nothing: each is in the sphere with the probability of
    its volume, or, where an oxygen in the sphere has an energy 2 kT lower, with
    V_in e^2 / (V_in e^2 + V_out), independently, so the number in the sphere is binomial. The
    moves alone, started with every water outside, give its mean and its variance, with anchors
    uniform in the cell and with half of them in the sphere. The stand-in energy knows nothing of
    the well: it reaches the acceptance only through the potential energy of the Context (with
    the sign of that term turned the mean would be 0.3 in place of 9.7)."""
    n, edge, radius = 20, 2.0, 0.6
    system, positions, box = ideal_waters(n, edge, well_kt, radius, seed=3)
    spec = water_spec(system, [0], radius_nm=radius, trial_poses=16, inner_fraction=inner_fraction)
    spec.waters = np.array([[1 + 3 * k, 2 + 3 * k, 3 + 3 * k] for k in range(n)])
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    mover = RelocationMover(spec, context, seed=5)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    assert mover.inside(positions, box).sum() == 0
    counts, accepted = [], {"in": 0, "out": 0}
    for attempt in range(3000):
        result = mover.attempt(state)
        accepted[result["direction"]] += result["accepted"]
        if attempt >= 300:
            x = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
            counts.append(int(mover.inside(x, box).sum()))
    v_in = 4.0 / 3.0 * np.pi * radius ** 3
    weight = v_in * np.exp(well_kt)
    p = weight / (weight + edge ** 3 - v_in)
    counts = np.array(counts)
    # tolerances: four times the scatter over eight chains of this length with other seeds. Without
    # the well (exact mean 2.26, variance 2.01) the means had a standard deviation of 0.06 to 0.08
    # and the variances of 0.08 to 0.15; with it (9.70 and 5.00) 0.15, and 0.4 to 0.6
    assert counts.mean() == pytest.approx(n * p, abs=0.6 if well_kt else 0.35)
    assert counts.var() == pytest.approx(n * p * (1 - p), abs=2.4 if well_kt else 0.6)
    assert accepted["in"] > 100 and accepted["out"] > 100


def test_a_water_that_is_moved_keeps_its_shape_and_the_velocity_of_its_centre_of_mass():
    """An accepted move changes one water: its sites are turned and shifted as one body, the
    velocity of its centre of mass is the one it had, its velocities relative to it are turned
    with it (the kinetic energy is the same), and no other atom changes. A move that is not
    accepted changes nothing."""
    n, edge, radius = 20, 2.0, 0.6
    system, positions, box = ideal_waters(n, edge, 0.0, radius, seed=4)
    spec = water_spec(system, [0], radius_nm=radius, trial_poses=16)
    spec.waters = np.array([[1 + 3 * k, 2 + 3 * k, 3 + 3 * k] for k in range(n)])
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    context.setVelocities(np.random.default_rng(8).normal(size=positions.shape))
    mover = RelocationMover(spec, context, seed=6)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)

    def snapshot():
        st = context.getState(getPositions=True, getVelocities=True)
        return (np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer)),
                np.asarray(st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)))

    seen = {True: 0, False: 0}
    for _ in range(200):
        (x0, v0), result, (x1, v1) = snapshot(), mover.attempt(state), snapshot()
        seen[result["accepted"]] += 1
        changed = np.flatnonzero(np.any(x0 != x1, axis=1))
        if not result["accepted"]:
            assert len(changed) == 0 and np.array_equal(v0, v1)
            continue
        atoms = spec.waters[(changed[0] - 1) // 3]
        assert np.array_equal(changed, atoms)
        assert np.array_equal(np.flatnonzero(np.any(v0 != v1, axis=1)), atoms)
        for a in range(3):
            for b in range(a):
                assert np.linalg.norm(x1[atoms[a]] - x1[atoms[b]]) == pytest.approx(
                    np.linalg.norm(x0[atoms[a]] - x0[atoms[b]]), abs=1e-12)
        mass = spec.mass[atoms][:, None]
        assert np.allclose((mass * v1[atoms]).sum(axis=0), (mass * v0[atoms]).sum(axis=0), atol=1e-12)
        assert (mass * v1[atoms] ** 2).sum() == pytest.approx((mass * v0[atoms] ** 2).sum(), rel=1e-12)
        # the velocities relative to the centre of mass turned as the molecule did
        com0, com1 = (mass * x0[atoms]).sum(axis=0) / mass.sum(), (mass * x1[atoms]).sum(axis=0) / mass.sum()
        vcom = (mass * v0[atoms]).sum(axis=0) / mass.sum()
        angular0 = np.cross(x0[atoms] - com0, mass * (v0[atoms] - vcom)).sum(axis=0)
        angular1 = np.cross(x1[atoms] - com1, mass * (v1[atoms] - vcom)).sum(axis=0)
        assert np.linalg.norm(angular1) == pytest.approx(np.linalg.norm(angular0), rel=1e-10)
    assert seen[True] > 20 and seen[False] > 20


def test_one_move_is_attempted_for_every_interval_of_md():
    """After the MD of an iteration one move is attempted for every multiple of the interval the
    trajectory of the replica passed in it: with iterations of 625 steps and an interval of 1000
    steps (2.5 ps and 4 ps at 4 fs), 1000 attempts in 1600 iterations, and the same ones whatever
    iteration a leg is resumed at. An interval shorter than an iteration gives several attempts
    after one iteration, as many as the setting says (review of 723b12f: it gave one, 16 where
    10000 steps hold 40 intervals of 250)."""
    due = [attempts_due(iteration, 625, 1000) for iteration in range(1600)]
    assert sum(due) == 1000 and due[:5] == [0, 1, 0, 1, 1]
    assert [attempts_due(iteration, 625, 250) for iteration in range(4)] == [2, 3, 2, 3]
    assert sum(attempts_due(iteration, 625, 250) for iteration in range(16)) == 40
    assert sum(attempts_due(iteration, 625, 6250) for iteration in range(100)) == 10


def test_a_cell_that_became_too_small_for_the_sphere_stops_the_moves(water_ligand_system):
    """The sphere and the cutoff are compared with the cell when the leg is built and again at
    every attempt, with the cell the Context has then: a barostat changes the cell, and in one
    whose smallest height is no more than twice the radius the nearest image of a water no longer
    decides its region. The attempt stops with the reason (review of 723b12f: only the cell the
    leg was built with was compared)."""
    system, pos, box, ligand = water_ligand_system
    height = float(np.linalg.norm(box[0]))
    spec = water_spec(system, ligand, radius_nm=0.49 * height, trial_poses=16)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    context.setPositions(pos)
    mover = RelocationMover(spec, context, seed=3)
    state = AlchemicalState(lambda_restraints=1.0, lambda_electrostatics=1.0, lambda_sterics=1.0)
    assert mover.attempt(state)["direction"] in ("in", "out")
    context.setPeriodicBoxVectors(*[mm.Vec3(*(0.97 * np.asarray(r))) for r in box])
    with pytest.raises(RelocationError, match="not less than half the smallest height of the cell"):
        mover.attempt(state)


def test_the_radius_of_the_sphere_is_that_of_the_paper():
    """Half the largest distance between two heavy atoms of the ligand plus the margin (0.3 nm)."""
    atoms = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 1.2, 0.0]])
    assert sphere_radius(atoms, 0.3) == pytest.approx(0.5 * 1.3 + 0.3)


def test_a_leg_with_the_moves_stores_them_by_window_and_goes_on_after_a_stop(tmp_path, water_ligand_system):
    """Replica exchange of a small leg with the moves (stock OpenMM, CPU): after the MD of an
    iteration in which the interval was passed, every replica attempts one move in the state it is
    simulated in. The samples hold, per iteration and lambda window, the moves attempted and
    accepted in each direction; a leg that is resumed attempts its moves in the iterations it would
    have (the rule is one of the iteration), and its result gives the numbers of the production by
    window."""
    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.sampling.hrex import ReplicaExchangeSampler
    from cufep.sampling.storage import read_samples
    from cufep.sampling.worker import LegSpec
    from cufep.workflow import relocation_numbers

    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 2})
    k = len(schedule.states)
    # 10 steps of 2 fs per iteration, one move for every 0.04 ps (20 steps): every second iteration
    spec = LegSpec(name="moves", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=1.0,
                   barostat="isotropic", barostat_frequency=5, timestep_fs=2.0, steps_per_iteration=10, seed=5,
                   relocation=water_spec(system, ligand, trial_poses=32, interval_ps=0.04))

    def sampler():
        return ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=lambda m: None)

    with sampler() as s:
        s.initialize(pos, box)
        s.run(5, record_stride=1, checkpoint_stride=5)
    moves = read_samples(tmp_path / "hrex")["relocations"]
    assert moves.shape == (5, k, 4)
    attempted = moves[:, :, 0] + moves[:, :, 2]
    assert attempted.sum(axis=1).tolist() == [0, k, 0, k, 0] and (attempted[1] == 1).all()
    assert (moves[:, :, 1] <= moves[:, :, 0]).all() and (moves[:, :, 3] <= moves[:, :, 2]).all()
    with sampler() as s:
        assert s.restore()
        s.run(3, record_stride=1, checkpoint_stride=3)
    samples = read_samples(tmp_path / "hrex")
    attempted = samples["relocations"][:, :, 0] + samples["relocations"][:, :, 2]
    assert attempted.sum(axis=1).tolist() == [0, k, 0, k, 0, k, 0, k]
    numbers = relocation_numbers(samples, 2)                      # the iterations from the third on
    assert [a + b for a, b in zip(numbers["attempted_into_the_sphere"], numbers["attempted_out_of_the_sphere"])] \
        == [3] * k
    assert sum(numbers["accepted_into_the_sphere"]) == int(samples["relocations"][2:, :, 1].sum())
    # samples without the moves have no such numbers
    assert relocation_numbers({"u": samples["u"]}, 0) is None
    # an interval shorter than an iteration (5 steps in 10): two moves of every replica after each
    from dataclasses import replace

    short = replace(spec, name="short", relocation=water_spec(system, ligand, trial_poses=32, interval_ps=0.01))
    with ReplicaExchangeSampler(short, tmp_path / "short", devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(2, record_stride=1, checkpoint_stride=2)
    moves = read_samples(tmp_path / "short")["relocations"]
    assert (moves[:, :, 0] + moves[:, :, 2] == 2).all()


def test_the_moves_are_recorded_with_the_samples_of_the_leg_that_makes_them():
    """The conditions of the samples name the settings of the moves for the complex leg of a run
    that makes them, and for no other leg: a leg sampled with the moves is not continued without
    them or with other ones, and the samples of every leg without them, also the solvent leg of a
    run with them and every leg of an earlier version, keep the record they have. A configuration
    of an earlier version, which has no such section, is the one without moves."""
    from cufep.config.schema import Config
    from cufep.workflow import SETTINGS_ADDED, samples_settings, settings_differences

    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}}
    default = Config.model_validate(base)
    moving = Config.model_validate(dict(base, sampling={"relocation_mc": {"enabled": True, "trial_poses": 4096,
                                                                           "inner_fraction": 0.5}}))
    assert "relocation_mc" not in samples_settings(default, "complex")
    assert samples_settings(moving, "solvent") == samples_settings(default, "solvent") == samples_settings(default)
    record = samples_settings(moving, "complex")
    assert record["relocation_mc"] == {"enabled": True, "interval_ps": 4.0, "trial_poses": 4096,
                                       "inner_fraction": 0.5, "radius_margin_nm": 0.3}
    assert settings_differences(samples_settings(default, "complex"), record)[0].startswith(
        "relocation_mc: not recorded -> ")
    assert SETTINGS_ADDED["sampling"]["relocation_mc"] == default.sampling.relocation_mc.model_dump(mode="json")
    with pytest.raises(ValueError):
        Config.model_validate(dict(base, sampling={"relocation_mc": {"inner_fraction": 1.0}}))


def test_what_a_mover_needs_is_taken_from_the_system_and_what_it_cannot_treat_is_refused():
    """The specification of the moves from a prepared system: every water (residue HOH) with its
    oxygen first, the parameters of the fully interacting System, the sphere of the paper around
    the heavy atoms of the ligand. Refused: waters that are not alike, a water that interacts
    within itself, and a cell in which the sphere or the cutoff does not fit."""
    import openmm.app as app

    from cufep.sampling.relocation_mc import RelocationError, relocation_spec

    forcefield = app.ForceField("amber14/tip3p.xml")
    model = app.Modeller(app.Topology(), [])
    model.addSolvent(forcefield, boxSize=mm.Vec3(2.0, 2.0, 2.0) * u.nanometer)
    positions = np.asarray(model.positions.value_in_unit(u.nanometer))
    box = np.eye(3) * 2.0

    def build():
        return forcefield.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer)

    def spec(system, **more):
        values = dict(cutoff_nm=0.9, softcore_alpha=0.5, softcore_b=1, temperature_K=298.15, trial_poses=64,
                      inner_fraction=None, interval_ps=4.0, radius_margin_nm=0.3)
        values.update(more)
        # the ligand: the atoms of two waters, so that the sphere has a size of its own
        return relocation_spec(system, model.topology, positions, box, [0, 1, 2, 3, 4, 5], **values)

    made = spec(build())
    n_waters = model.topology.getNumResidues()
    assert made.waters.shape == (n_waters, 3) and made.waters[5].tolist() == [15, 16, 17]
    assert made.ligand_heavy.tolist() == [0, 3] and made.ligand.tolist() == [0, 1, 2, 3, 4, 5]
    separation = positions[3] - positions[0]
    separation -= np.round(separation / 2.0) * 2.0
    assert made.radius_nm == pytest.approx(0.5 * np.linalg.norm(separation) + 0.3)
    assert made.charge[0] == pytest.approx(-0.834) and made.charge[1] == pytest.approx(0.417)
    assert made.epsilon[1] == 0.0 and made.mass[0] == pytest.approx(15.999, abs=0.01)

    unlike = build()
    force = next(f for f in unlike.getForces() if isinstance(f, mm.NonbondedForce))
    force.setParticleParameters(30, -0.8, 0.3, 0.6)
    with pytest.raises(RelocationError, match="not all have the same nonbonded parameters"):
        spec(unlike)
    bonded = build()
    force = next(f for f in bonded.getForces() if isinstance(f, mm.NonbondedForce))
    i, j, _qq, sigma, epsilon = force.getExceptionParameters(0)
    force.setExceptionParameters(0, i, j, 0.1, sigma, epsilon)
    with pytest.raises(RelocationError, match="between its own sites"):
        spec(bonded)
    with pytest.raises(RelocationError, match="less than half the smallest height"):
        spec(build(), cutoff_nm=1.0)
    with pytest.raises(RelocationError, match="less than half the smallest height"):
        spec(build(), radius_margin_nm=0.9)


def test_a_window_of_the_precalculation_attempts_one_move_for_every_interval(water_ligand_system):
    """In a window of the precalculation of adaptive lambda scheduling a move is attempted each time
    the MD of the window, its equilibration and its samples together, has gone on by the interval,
    in the state of the window; cutting the MD at those steps changes nothing of it (a mover that
    moves nothing gives the energies and the last configuration of a window without one)."""
    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.schedule import run_window

    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)

    def window(mover=None, interval_steps=0):
        integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
        integrator.setRandomNumberSeed(31)
        context = mm.Context(alch.system, integrator, mm.Platform.getPlatformByName("CPU"),
                             {"Threads": "1", "DeterministicForces": "true"})
        context.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
        context.setPositions(pos)
        context.setVelocitiesToTemperature(300.0, 32)
        st = context.getState(getPositions=True, getVelocities=True)
        config = (st.getPositions(asNumpy=True).value_in_unit(u.nanometer), box,
                  st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond))
        task = {"state": {"lambda_restraints": 1.0, "lambda_electrostatics": 0.5, "lambda_sterics": 1.0},
                "config": config, "stages": ["discharge"], "candidates": 5, "temperature_K": 300.0,
                "equilibration_steps": 50, "n_samples": 4, "steps_per_sample": 20,
                "leg_stages": ["discharge", "vanish"], "rest2_scale": 1.0}
        return run_window(context, integrator, task, mover, interval_steps)

    class Still:
        def __init__(self):
            self.states = []

        def attempt(self, state):
            self.states.append(state.lambda_electrostatics)

    still = Still()
    with_moves, without = window(still, 30), window()
    assert still.states == [0.5] * 4                              # at steps 30, 60, 90 and 120 of 130
    assert np.array_equal(with_moves["energies"]["discharge"], without["energies"]["discharge"])
    assert np.array_equal(with_moves["config"][0], without["config"][0])
