"""Alchemical factory, force-group energy evaluation, restraints and tail energies."""
import math

import numpy as np
import openmm as mm
import pytest
from conftest import potential, reference_context

from cufep.alchemy.factory import (
    LAMBDA_ELECTROSTATICS,
    LAMBDA_STERICS,
    MAIN_GROUP,
    SOFTCORE_GROUP,
    AlchemyError,
    make_alchemical_system,
    set_lambdas,
)
from cufep.alchemy.lrc import TailEnergy, minimum_image, tail_free_energy
from cufep.alchemy.restraints import BoreschRestraint
from cufep.backends.base import clone_system
from cufep.sampling.states import AlchemicalState, StateEnergyEvaluator, set_state, validate_states


@pytest.fixture(scope="module")
def alch(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    return make_alchemical_system(system, ligand)


def test_coupled_state_reproduces_the_original_system(water_ligand_system, alch):
    system, pos, box, ligand = water_ligand_system
    e0 = potential(reference_context(system, pos, box))
    ctx = reference_context(alch.system, pos, box)
    set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: 1.0, LAMBDA_STERICS: 1.0})
    assert potential(ctx) == pytest.approx(e0, abs=2e-3)


def test_pairs_inside_the_ligand_have_no_cutoff_at_the_coupled_end():
    """Review of PR 1. Every pair of ligand atoms is an exception of the NonbondedForce, and an
    exception has no cutoff: the coupled end has, on top of the original System, the Lennard-Jones
    energy of ligand atoms that are further apart than the cutoff (protocol.md, "Interactions
    inside the ligand"). This pins that documented difference, for a pair and for a long molecule.
    """
    cutoff, sigma, epsilon, box = 0.9, 0.4, 1.0, 4.0

    def lj(r):
        x = (sigma / r) ** 6
        return 4.0 * epsilon * (x * x - x)

    def both_ends(ligand_positions):
        """Energy of the original System and of the coupled end; two far atoms are the environment."""
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.PME)
        nb.setCutoffDistance(cutoff)
        nb.setUseDispersionCorrection(False)
        positions = [mm.Vec3(*p) for p in ligand_positions] + [mm.Vec3(3.0, 3.0, 3.0), mm.Vec3(3.0, 3.0, 3.4)]
        for _ in positions:
            system.addParticle(39.9)
            nb.addParticle(0.0, sigma, epsilon)
        system.addForce(nb)
        ligand = list(range(len(ligand_positions)))
        alch = make_alchemical_system(system, ligand)
        pos = np.array([[p.x, p.y, p.z] for p in positions])
        cell = np.eye(3) * box
        ctx = reference_context(alch.system, pos, cell)
        set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: 1.0, LAMBDA_STERICS: 1.0})
        return potential(reference_context(system, pos, cell)), potential(ctx)

    # two ligand atoms inside the cutoff: the coupled end is the original System
    original, coupled = both_ends([(1.0, 1.0, 1.0), (1.85, 1.0, 1.0)])
    assert coupled == pytest.approx(original, abs=1e-6) and original == pytest.approx(lj(0.85) + lj(0.4), abs=1e-6)
    # beyond the cutoff: the original System has no energy for the pair, the coupled end has its tail
    original, coupled = both_ends([(1.0, 1.0, 1.0), (1.95, 1.0, 1.0)])
    assert original == pytest.approx(lj(0.4), abs=1e-6)
    assert coupled - original == pytest.approx(lj(0.95), abs=1e-6) and lj(0.95) == pytest.approx(-0.0222, abs=1e-4)
    # a long molecule: the difference is the sum over its pairs beyond the cutoff
    chain = [(0.6 + 0.5 * k, 1.0, 1.0) for k in range(4)]           # 1.5 nm from end to end
    original, coupled = both_ends(chain)
    far = sum(lj(0.5 * (b - a)) for a in range(4) for b in range(a + 1, 4) if 0.5 * (b - a) > cutoff)
    assert far == pytest.approx(2 * lj(1.0) + lj(1.5), abs=1e-9) and far < -1e-2
    assert original == pytest.approx(3 * lj(0.5) + lj(0.4), abs=1e-6)
    assert coupled - original == pytest.approx(far, abs=1e-6)


def test_decoupled_state_has_no_ligand_environment_interaction(water_ligand_system, alch):
    system, pos, box, ligand = water_ligand_system
    ref = clone_system(system)
    nb = next(f for f in ref.getForces() if isinstance(f, mm.NonbondedForce))
    for i in ligand:
        _q, sig, _eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, sig, 0.0)
    e_ref = potential(reference_context(ref, pos, box))
    ctx = reference_context(alch.system, pos, box)
    set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: 0.0, LAMBDA_STERICS: 0.0})
    assert potential(ctx) == pytest.approx(e_ref, abs=2e-3)
    # and moving the decoupled ligand does not change the energy
    moved = pos.copy()
    moved[ligand] += np.array([0.31, -0.2, 0.17])
    ctx.setPositions(moved)
    assert potential(ctx) == pytest.approx(e_ref, abs=2e-3)


def test_force_groups_add_up_and_main_group_is_quadratic_in_lambda_elec(water_ligand_system, alch):
    system, pos, box, ligand = water_ligand_system
    ctx = reference_context(alch.system, pos, box)
    g = {}
    for le in (0.0, 0.25, 0.5, 0.75, 1.0):
        set_lambdas(ctx, {LAMBDA_ELECTROSTATICS: le, LAMBDA_STERICS: 0.6})
        g[le] = potential(ctx, {MAIN_GROUP})
        assert potential(ctx) == pytest.approx(g[le] + potential(ctx, {SOFTCORE_GROUP}), abs=1e-6)
    a2 = 2 * (g[1.0] - 2 * g[0.5] + g[0.0])
    a1 = g[1.0] - g[0.0] - a2
    for le in (0.25, 0.75):
        assert g[0.0] + a1 * le + a2 * le * le == pytest.approx(g[le], abs=1e-4)


def test_evaluator_matches_brute_force(water_ligand_system, alch):
    system, pos, box, ligand = water_ligand_system
    states = [AlchemicalState(1, 1.0, 1.0), AlchemicalState(1, 0.7, 1.0), AlchemicalState(1, 0.4, 1.0),
              AlchemicalState(1, 0.1, 1.0), AlchemicalState(1, 0.0, 1.0), AlchemicalState(1, 0.0, 0.5),
              AlchemicalState(1, 0.0, 0.1), AlchemicalState(1, 0.0, 0.0)]
    validate_states(states)
    ctx = reference_context(alch.system, pos, box)
    T, P = 298.15, 1.0
    red, vol = StateEnergyEvaluator(ctx, states, T, P).reduced_potentials()
    kt = 8.31446261815324e-3 * T
    for k, s in enumerate(states):
        set_state(ctx, s)
        brute = (potential(ctx) + P * vol * 0.0602214076) / kt
        assert red[k] == pytest.approx(brute, abs=1e-4)
    assert vol == pytest.approx(abs(np.linalg.det(box)), rel=1e-6)


def test_states_need_charges_off_before_sterics():
    with pytest.raises(ValueError):
        validate_states([AlchemicalState(1, 1.0, 1.0), AlchemicalState(1, 0.5, 0.5)])


def test_factory_rejects_bad_input(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    with pytest.raises(AlchemyError):
        make_alchemical_system(system, [])
    alch = make_alchemical_system(system, ligand)
    with pytest.raises(AlchemyError):
        make_alchemical_system(alch.system, ligand)        # already alchemical


# ----------------------------------------------------------------------------- restraints
def _restraint():
    return BoreschRestraint((2, 1, 0), (3, 4, 5), r0=0.8, theta_a0=1.6, theta_b0=1.3,
                            phi_a0=0.9, phi_b0=-3.0, phi_c0=2.2)


def test_restraint_force_matches_numpy_energy():
    r = _restraint()
    system = mm.System()
    for _ in range(6):
        system.addParticle(12.0)
    # the force uses periodic distances; a large box keeps the random test points unwrapped
    system.setDefaultPeriodicBoxVectors(mm.Vec3(30, 0, 0), mm.Vec3(0, 30, 0), mm.Vec3(0, 0, 30))
    force = r.create_force()
    system.addForce(force)
    rng = np.random.default_rng(3)
    for _ in range(25):
        pos = rng.normal(scale=0.6, size=(6, 3)) + 15.0
        ctx = reference_context(system, pos, np.eye(3) * 30)
        assert potential(ctx) == pytest.approx(float(r.energy(pos)), rel=1e-6, abs=1e-6)
        ctx.setParameter("lambda_restraints", 0.25)
        assert potential(ctx) == pytest.approx(0.25 * float(r.energy(pos)), rel=1e-6, abs=1e-6)


def test_restraint_values_at_reference_geometry_give_zero_energy():
    # build six points that realise the reference values exactly
    def place(a, b, c, dist, angle, dihedral):
        bc = (c - b) / np.linalg.norm(c - b)
        n = np.cross(b - a, bc)
        n /= np.linalg.norm(n)
        m = np.cross(n, bc)
        d = np.array([-dist * math.cos(angle), dist * math.sin(angle) * math.cos(dihedral),
                      dist * math.sin(angle) * math.sin(dihedral)])
        return c + d[0] * bc + d[1] * m + d[2] * n

    r = _restraint()
    h2, h1, h0 = np.array([0.0, 0.0, 0.0]), np.array([0.15, 0.0, 0.0]), np.array([0.2, 0.14, 0.0])
    g0 = place(h2, h1, h0, r.r0, r.theta_a0, r.phi_a0)
    g1 = place(h1, h0, g0, 0.14, r.theta_b0, r.phi_b0)
    g2 = place(h0, g0, g1, 0.14, 1.9, r.phi_c0)
    pos = np.array([h2, h1, h0, g0, g1, g2])        # host_atoms = (2, 1, 0): H0 is atom 2
    v = r.values(pos)
    assert v[0] == pytest.approx(r.r0, abs=1e-9)
    assert v[1] == pytest.approx(r.theta_a0, abs=1e-9) and v[2] == pytest.approx(r.theta_b0, abs=1e-9)
    assert float(r.energy(pos)) == pytest.approx(0.0, abs=1e-9)


def test_standard_state_correction_numerical_vs_analytical():
    r = _restraint()
    num, ana = r.release_free_energy(298.15, "numerical"), r.release_free_energy(298.15, "analytical")
    assert num < 0 and ana < 0
    assert num == pytest.approx(ana, abs=0.15)             # stiff restraint: Gaussian limit is good
    # by hand: -kT ln(8 pi^2 V0 sqrt(prod K) / (r0^2 sin sin (2 pi kT)^3))
    kt = 8.31446261815324e-3 * 298.15
    z = (0.8 ** 2 * math.sin(1.6) * math.sin(1.3) * (2 * math.pi * kt) ** 3
         / math.sqrt(4184.0 * 334.72 ** 5))
    assert ana == pytest.approx(-kt * math.log(8 * math.pi ** 2 * 1.6605390666 / z), rel=1e-12)
    # a soft restraint near a flat angle deviates from the Gaussian formula
    soft = BoreschRestraint((2, 1, 0), (3, 4, 5), 0.8, 0.5, 1.3, 0.9, -3.0, 2.2, k_theta_a=20.0)
    assert abs(soft.release_free_energy(298.15, "numerical") - soft.release_free_energy(298.15, "analytical")) > 0.05


def test_restraint_serialisation_round_trip():
    r = _restraint()
    assert BoreschRestraint.from_dict(r.to_dict()) == r


# ----------------------------------------------------------------------------- tail energy
def test_minimum_image_triclinic():
    box = np.array([[3.0, 0, 0], [0, 3.0, 0], [1.5, 1.5, 2.1]])
    rng = np.random.default_rng(1)
    d = rng.uniform(-6, 6, size=(200, 3))
    mi = minimum_image(d, box)
    shifts = np.array([i * box[0] + j * box[1] + k * box[2] for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)
                       for k in (-2, -1, 0, 1, 2)])
    brute = np.min(np.linalg.norm(d[:, None, :] + shifts[None, :, :], axis=2), axis=1)
    # the reduced-cell reduction is exact for all distances below half the shortest height
    near = brute < 1.0
    assert np.allclose(np.linalg.norm(mi, axis=1)[near], brute[near], atol=1e-9)


def test_tail_energy_matches_brute_force():
    rng = np.random.default_rng(2)
    box = np.eye(3) * 5.0
    pos = rng.uniform(0, 5, size=(400, 3))
    lig = np.arange(3)
    tail = TailEnergy(lig, np.array([0.33, 0.30, 0.11]), np.array([0.4, 0.6, 0.0]), np.arange(3, 400),
                      np.full(397, 0.32), np.full(397, 0.5), cutoff_nm=0.9, long_cutoff_nm=2.0)
    e = tail.energy(pos, box)
    brute = 0.0
    for a, i in enumerate(lig):
        for j in range(3, 400):
            d = pos[j] - pos[i]
            d -= np.round(d / 5.0) * 5.0
            r = np.linalg.norm(d)
            if 0.9 < r < 2.0:
                s, eps = 0.5 * (tail.lig_sigma[a] + 0.32), math.sqrt(tail.lig_epsilon[a] * 0.5)
                brute += 4 * eps * ((s / r) ** 12 - (s / r) ** 6)
    homogeneous = e - brute
    assert homogeneous < 0 and abs(homogeneous) < 0.2 * abs(brute)
    again = TailEnergy.from_dict(tail.to_dict())
    assert again.energy(pos, box) == pytest.approx(e, rel=1e-12)


def test_the_tail_never_begins_inside_the_real_space_cutoff():
    """Review of PR 1 (P2): with an outer radius of 0.45 nm and a cutoff of 0.9 nm the explicit shell was empty
    and the homogeneous tail began at 0.45 nm, adding the dispersion between 0.45 and 0.9 nm a second time.
    The tail begins at the cutoff at the latest; a cell too small for an explicit shell gives the homogeneous
    tail from the cutoff."""
    import openmm as mm
    import openmm.unit as unit

    rng = np.random.default_rng(5)
    box = np.eye(3) * 5.0
    pos = rng.uniform(0, 5, size=(200, 3))
    args = (np.arange(2), np.array([0.33, 0.30]), np.array([0.4, 0.6]), np.arange(2, 200), np.full(198, 0.32),
            np.full(198, 0.5))
    at_cutoff = TailEnergy(*args, cutoff_nm=0.9, long_cutoff_nm=0.9)
    inside = TailEnergy(*args, cutoff_nm=0.9, long_cutoff_nm=0.45)
    assert inside.outer_radius(box) == 0.9 and inside.energy(pos, box) == pytest.approx(at_cutoff.energy(pos, box))
    homogeneous = (16.0 * np.pi / 125.0) * (at_cutoff._s12 / (9.0 * 0.9 ** 9) - at_cutoff._s6 / (3.0 * 0.9 ** 3))
    assert at_cutoff.energy(pos, box) == pytest.approx(homogeneous)
    small = np.eye(3) * 1.82                       # 0.49 x 1.82 nm = 0.892 nm is inside the cutoff
    assert TailEnergy(*args, cutoff_nm=0.9, long_cutoff_nm=2.0).outer_radius(small) == 0.9
    # the tail is made from the System's own cutoff, and an outer radius inside it is refused
    system = mm.System()
    nb = mm.NonbondedForce()
    for _ in range(4):
        system.addParticle(12.0)
        nb.addParticle(0.0, 0.3, 0.5)
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    system.addForce(nb)
    params = [(0.0, 0.3, 0.5)]
    TailEnergy.from_system(system, [0], params, 0.9, 2.0)
    with pytest.raises(ValueError, match="the System cuts the nonbonded interactions at 0.9 nm"):
        TailEnergy.from_system(system, [0], params, 1.0, 2.0)
    with pytest.raises(ValueError, match="must be larger than the real-space cutoff"):
        TailEnergy.from_system(system, [0], params, 0.9, 0.45)


def test_tail_free_energy_limits():
    assert tail_free_energy(np.full(50, -3.0), 298.15)[0] == pytest.approx(-3.0)
    dg, se = tail_free_energy(np.array([-2.0, -4.0]), 298.15)
    assert -4.0 < dg < -3.0 and se > 0
    # review of PR 1: no tail energy is no estimate (it was a correction of 0 with an error of 0)
    with pytest.raises(ValueError, match="needs at least one recorded tail energy"):
        tail_free_energy(np.array([]), 298.15)
