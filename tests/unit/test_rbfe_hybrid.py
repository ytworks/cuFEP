"""The hybrid System of a relative transformation (cufep.rbfe.hybrid) and the energies of its states
(cufep.rbfe.states), on the CPU with stock OpenMM.

Toluene (A) becomes phenol (B) in a small box of water: the ring and its hydrogens are the core, the methyl
group the unique old atoms, the hydroxyl the unique new ones.
"""
from __future__ import annotations

import math

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from conftest import TOLUENE, build_hybrid_edge, potential, reference_context

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields")


@pytest.fixture(scope="module")
def edge():
    return build_hybrid_edge()


def _hybrid(edge, **kwargs):
    from cufep.rbfe.hybrid import make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, _box = edge
    return make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, **kwargs)


def _set(ctx, state):
    for name, value in state.as_parameters().items():
        if name in ctx.getParameters():
            ctx.setParameter(name, value)


def _valence_of(system, atoms, positions, box) -> float:
    """Energy of the bonds, angles and torsions of ``system`` that have an atom of ``atoms``."""
    keep = mm.System()
    for i in range(system.getNumParticles()):
        keep.addParticle(system.getParticleMass(i))
    keep.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    atoms = set(atoms)
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            g = mm.HarmonicBondForce()
            for k in range(f.getNumBonds()):
                i, j, r, kk = f.getBondParameters(k)
                if {i, j} & atoms:
                    g.addBond(i, j, r, kk)
        elif isinstance(f, mm.HarmonicAngleForce):
            g = mm.HarmonicAngleForce()
            for k in range(f.getNumAngles()):
                i, j, m, t, kk = f.getAngleParameters(k)
                if {i, j, m} & atoms:
                    g.addAngle(i, j, m, t, kk)
        elif isinstance(f, mm.PeriodicTorsionForce):
            g = mm.PeriodicTorsionForce()
            for k in range(f.getNumTorsions()):
                i, j, m, n, p, ph, kk = f.getTorsionParameters(k)
                if {i, j, m, n} & atoms:
                    g.addTorsion(i, j, m, n, p, ph, kk)
        else:
            continue
        keep.addForce(g)
    return potential(reference_context(keep, positions, box))


@pytest.mark.parametrize("rest2", [False, True])
def test_the_ends_of_the_hybrid_are_the_two_ligands_and_the_dummies_keep_their_bonded_terms(edge, rest2):
    """At lambda 0 the hybrid System has the energy of System A plus the bonded terms of the unique atoms of B
    (dummies that do not interact), at lambda 1 that of System B plus the bonded terms of the unique atoms of
    A: the rules of OpenFE's hybrid topology, in cuFEP's layout. With REST2 (the ends are not scaled) the
    same, with water molecules in the hot region."""
    from cufep.rbfe.states import HybridState

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    hot = list(range(len(ligand_a), len(ligand_a) + 9)) if rest2 else []
    hybrid = _hybrid(edge, rest2=rest2, rest2_atoms=hot)
    positions = hybrid.positions
    h_to_b = {h: j for j, h in hybrid.b_to_hybrid.items()}
    positions_b = np.array([positions[hybrid.b_to_hybrid[j]] for j in range(system_b.getNumParticles())])
    ctx = reference_context(hybrid.system, positions, box)

    _set(ctx, HybridState(0.0))
    expected = potential(reference_context(system_a, positions[: system_a.getNumParticles()], box)) \
        + _valence_of(system_b, [h_to_b[h] for h in hybrid.unique_new], positions_b, box)
    assert potential(ctx) == pytest.approx(expected, abs=2e-3)

    _set(ctx, HybridState(1.0))
    expected = potential(reference_context(system_b, positions_b, box)) \
        + _valence_of(system_a, hybrid.unique_old, positions[: system_a.getNumParticles()], box)
    assert potential(ctx) == pytest.approx(expected, abs=2e-3)


def test_the_plugin_gets_only_charges_of_the_ligand(edge):
    """The design that keeps the hybrid on the cuPotential plugin (spikes/m36): the NonbondedForce, which the
    plugin replaces, has no Lennard-Jones of a ligand atom and offsets on charges only, under the three
    electrostatic lambdas and REST2."""
    from cufep.rbfe.hybrid import ELECTROSTATIC_LAMBDAS

    hybrid = _hybrid(edge, rest2=True, rest2_atoms=[len(edge[2])])
    (nb,) = [f for f in hybrid.system.getForces() if isinstance(f, mm.NonbondedForce)]
    for i in hybrid.ligand_atoms:
        assert nb.getParticleParameters(i)[2].value_in_unit(u.kilojoule_per_mole) == 0.0
    names = {nb.getGlobalParameterName(k) for k in range(nb.getNumGlobalParameters())}
    assert names == set(ELECTROSTATIC_LAMBDAS) | {"rest2_dq", "rest2_ds"}
    for k in range(nb.getNumParticleParameterOffsets()):
        name, i, dq, dsig, deps = nb.getParticleParameterOffset(k)
        assert dsig == 0.0
        assert deps == 0.0 or (name == "rest2_ds" and i not in hybrid.ligand_atoms)


@pytest.mark.parametrize("rest2", [False, True])
def test_the_evaluator_gives_the_energy_of_every_state(edge, rest2):
    """The cheap evaluation (a quadratic in each half of lambda for the main group, the others per value)
    equals setting each state and computing the energy, for a configuration after MD in the middle."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.rbfe.states import HybridEnergyEvaluator, HybridState
    from cufep.sampling.states import R_GAS

    *_, box = edge
    hot = list(range(len(edge[2]), len(edge[2]) + 12)) if rest2 else []
    hybrid = _hybrid(edge, rest2=rest2, rest2_atoms=hot)
    integ = mm.LangevinMiddleIntegrator(298.15 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(3)
    ctx = mm.Context(hybrid.system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(hybrid.positions)
    _set(ctx, HybridState(0.5, 1.6 if rest2 else 1.0))
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
    integ.step(50)
    schedule = fixed_schedule(["morph"], {"morph": 8}, rest2_scale=2.0 if rest2 else 1.0)
    states = schedule.states
    assert all(isinstance(s, HybridState) for s in states) and states[0].lambda_hybrid == 0.0
    assert (max(s.rest2_scale for s in states) == pytest.approx(2.0)) == rest2
    evaluator = HybridEnergyEvaluator(ctx, states, 298.15, None)
    fast, _volume = evaluator.reduced_potentials()
    beta = 1.0 / (R_GAS * 298.15)
    slow = []
    for state in states:
        _set(ctx, state)
        slow.append(beta * potential(ctx))
    assert np.allclose(fast, slow, rtol=0, atol=2e-6 * np.max(np.abs(slow)))


def test_a_group_that_becomes_itself_gives_the_same_energy_forwards_and_backwards():
    """Toluene into toluene with only the ring mapped: the methyl group of A is deleted and an identical one of
    B, at the same place, inserted. With the old and new copies on top of each other every state x has the
    energy of the state 1 - x: the delete and insert halves of the rules mirror each other."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridState

    same = {**{i: i for i in range(1, 7)}, **{10 + k: 10 + k for k in range(5)}}
    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge(TOLUENE, same)
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    positions = hybrid.positions.copy()
    b_unique = sorted(set(ligand_b) - set(core.values()))
    old_of_b = {j: ligand_a[j - ligand_b[0]] for j in b_unique}
    for j in b_unique:
        positions[hybrid.b_to_hybrid[j]] = positions[old_of_b[j]]
    ctx = reference_context(hybrid.system, positions, box)
    for x in (0.1, 0.3, 0.45, 0.5):
        _set(ctx, HybridState(x))
        forwards = potential(ctx)
        _set(ctx, HybridState(1.0 - x))
        assert potential(ctx) == pytest.approx(forwards, abs=1e-4), x


def test_a_mapping_that_changes_a_constrained_length_or_names_other_atoms_is_refused(edge):
    from cufep.alchemy.factory import AlchemyError
    from cufep.rbfe.hybrid import make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, _box = edge
    with pytest.raises(AlchemyError, match="not atoms of the ligands"):
        make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b,
                           {**core, len(ligand_a) + 3: ligand_b[0]})
    # a methyl hydrogen of toluene mapped onto the hydroxyl hydrogen of phenol: C-H and O-H are constrained
    # with other lengths
    with pytest.raises(AlchemyError, match="constraint"):
        make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b,
                           {**core, 0: ligand_b[0], 7: ligand_b[7]})


def test_lambda_functions_switch_the_unique_atoms_in_halves():
    from cufep.rbfe.states import lambda_functions

    first, middle, second = lambda_functions(0.25), lambda_functions(0.5), lambda_functions(0.75)
    assert first["lambda_electrostatics_delete"] == 0.5 and first["lambda_sterics_insert"] == 0.5
    assert first["lambda_sterics_delete"] == 0.0 and first["lambda_electrostatics_insert"] == 0.0
    assert middle["lambda_electrostatics_delete"] == 1.0 and middle["lambda_sterics_insert"] == 1.0
    assert middle["lambda_sterics_delete"] == 0.0 and middle["lambda_electrostatics_insert"] == 0.0
    assert second["lambda_sterics_delete"] == 0.5 and second["lambda_electrostatics_insert"] == 0.5
    assert all(math.isclose(v, 0.75) for k, v in second.items() if k.endswith(("core", "bonds", "angles", "torsions")))


def test_a_window_where_the_other_ligand_interacts_starts_only_after_a_minimisation(edge):
    """The bug of the first run of the TYK2 example (2026-11-25): every window of the complex leg started from the
    end of its pre-equilibration at lambda 0, where the atoms unique to B do not interact and water can sit on them;
    the windows in which they interact went to NaN at once. A water put on an atom unique to B: at lambda 1 that
    configuration has an energy no MD survives (the windows of that run failed at their first step), and after the
    window is minimised in its own state (run_window, ``minimize_steps``, as the leg starts such windows) its MD
    stays finite."""
    from cufep.alchemy.schedule import run_window
    from cufep.rbfe.states import HybridState

    *_, box = edge
    hybrid = _hybrid(edge)
    positions = hybrid.positions.copy()
    water_o = next(i for i in range(len(edge[2]), edge[0].getNumParticles())
                   if edge[0].getParticleMass(i).value_in_unit(u.dalton) > 15.0)
    shift = positions[hybrid.unique_new[0]] - positions[water_o]
    positions[water_o:water_o + 3] += shift + 0.02             # the water's oxygen on the unique atom

    def context():
        integ = mm.LangevinMiddleIntegrator(298.15 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
        integ.setRandomNumberSeed(2)
        ctx = mm.Context(hybrid.system, integ, mm.Platform.getPlatformByName("CPU"),
                         {"Threads": "1", "DeterministicForces": "true"})
        return ctx, integ

    ctx, _integ = context()
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(positions)
    _set(ctx, HybridState(1.0))
    assert potential(ctx) > 1e6                                # kJ/mol: an overlap of hard Lennard-Jones
    ctx, integ = context()
    task = {"state": HybridState(1.0).to_dict(), "config": (positions, box, np.zeros_like(positions)), "stages": [],
            "candidates": 2, "temperature_K": 298.15, "equilibration_steps": 200, "n_samples": 0,
            "steps_per_sample": 1, "minimize_steps": 500}
    out = run_window(ctx, integ, task)
    assert np.all(np.isfinite(out["config"][0])) and potential(ctx) < 0.0


def _copy_with(system, kind, add):
    copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    add(next(f for f in copy.getForces() if isinstance(f, kind)))
    return copy


def test_a_torsion_that_a_system_has_twice_counts_twice_at_its_end(edge):
    """Regression (review of PR #17): torsions were keyed by their atoms and parameters, as OpenFE keys them, so a
    term that a System has twice became one term and the end of the hybrid was not that System. One term twice in
    both Systems, another twice in A only: each end is its System."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridState

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    f = next(f for f in system_a.getForces() if isinstance(f, mm.PeriodicTorsionForce))
    atoms = next(f.getTorsionParameters(k)[:4] for k in range(f.getNumTorsions())
                 if set(f.getTorsionParameters(k)[:4]) <= set(core))
    # phases that give energies far from zero at a planar ring
    both, only_a = (1, 0.7, 5.0), (2, 0.3, 4.0)

    def twice_in_a(t):
        for per, phase, k in (both, both, only_a, only_a):
            t.addTorsion(*atoms, per, phase, k)

    def twice_in_b(t):
        for _ in range(2):
            t.addTorsion(*[core[i] for i in atoms], *both)

    sys_a = _copy_with(system_a, mm.PeriodicTorsionForce, twice_in_a)
    sys_b = _copy_with(system_b, mm.PeriodicTorsionForce, twice_in_b)
    hybrid = make_hybrid_system(sys_a, pos_a, ligand_a, sys_b, pos_b, ligand_b, core)
    positions = hybrid.positions
    h_to_b = {h: j for j, h in hybrid.b_to_hybrid.items()}
    positions_b = np.array([positions[hybrid.b_to_hybrid[j]] for j in range(sys_b.getNumParticles())])
    ctx = reference_context(hybrid.system, positions, box)
    _set(ctx, HybridState(0.0))
    expected = potential(reference_context(sys_a, positions[: sys_a.getNumParticles()], box)) \
        + _valence_of(sys_b, [h_to_b[h] for h in hybrid.unique_new], positions_b, box)
    assert potential(ctx) == pytest.approx(expected, abs=2e-3)
    _set(ctx, HybridState(1.0))
    expected = potential(reference_context(sys_b, positions_b, box)) \
        + _valence_of(sys_a, hybrid.unique_old, positions[: sys_a.getNumParticles()], box)
    assert potential(ctx) == pytest.approx(expected, abs=2e-3)


def test_bonds_and_angles_of_the_environment_that_differ_between_a_and_b_are_refused(edge):
    """Regression (review of PR #17): the hybrid takes the bonded terms of the environment from A and did not look at
    those of B. A bond or an angle of the environment in one System only, or with other parameters, is refused; the
    same term in both is taken."""
    from cufep.alchemy.factory import AlchemyError
    from cufep.rbfe.hybrid import make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, _box = edge
    n_a = len(ligand_a)
    # two oxygens of the water: n_a + 0 and n_a + 3 in A, 0 and 3 in B (B has its ligand last)
    bond_a = lambda k: lambda f: f.addBond(n_a, n_a + 3, 0.3, k)  # noqa: E731
    bond_b = lambda k: lambda f: f.addBond(0, 3, 0.3, k)  # noqa: E731
    same = make_hybrid_system(_copy_with(system_a, mm.HarmonicBondForce, bond_a(100.0)), pos_a, ligand_a,
                              _copy_with(system_b, mm.HarmonicBondForce, bond_b(100.0)), pos_b, ligand_b, core)
    assert same.system.getNumParticles() > 0
    cases = [(system_a, _copy_with(system_b, mm.HarmonicBondForce, bond_b(100.0)), "bond .* in B but not in A"),
             (_copy_with(system_a, mm.HarmonicBondForce, bond_a(100.0)),
              _copy_with(system_b, mm.HarmonicBondForce, bond_b(200.0)), "bond .* other parameters"),
             (_copy_with(system_a, mm.HarmonicAngleForce, lambda f: f.addAngle(n_a, n_a + 3, n_a + 6, 1.9, 50.0)),
              system_b, "angle .* in A but not in B")]
    for sys_a, sys_b, message in cases:
        with pytest.raises(AlchemyError, match=message):
            make_hybrid_system(sys_a, pos_a, ligand_a, sys_b, pos_b, ligand_b, core)
