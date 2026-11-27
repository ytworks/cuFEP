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


_BONDED = (mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce, mm.CustomAngleForce,
           mm.CustomTorsionForce)


def _bonded_context(system, x: float):
    """A Context of the bonded terms of a hybrid ``system`` alone, at lambda ``x``."""
    from cufep.rbfe.hybrid import CORE_BOND_FORCE_NAME
    from cufep.rbfe.states import HybridState

    s = mm.System()
    for i in range(system.getNumParticles()):
        s.addParticle(system.getParticleMass(i))
    for f in system.getForces():
        if isinstance(f, _BONDED) or (isinstance(f, mm.CustomBondForce) and f.getName() == CORE_BOND_FORCE_NAME):
            g = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))
            g.setForceGroup(0)
            s.addForce(g)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    _set(ctx, HybridState(x))
    return ctx


def _dummies_at_their_minimum(hybrid, x: float, positions: np.ndarray) -> np.ndarray:
    """``positions`` with the dummy atoms of the end ``x`` (the unique atoms of the other ligand) at the minimum of the
    bonded terms, the other atoms held (the dummies interact with nothing at that end): the lowest of the minima from
    where they are and from where the hybrid has them, each also with every dummy group turned by 120 and 240 degrees
    about its bond to the mapped atom X and about each bond of X to a mapped atom (a dummy group has minima of its own,
    its rotamers and the sites about X, and a start in one of them is not a coupling to the other atoms)."""
    from scipy.optimize import minimize

    ctx = _bonded_context(hybrid.system, x)
    dummies = np.array(hybrid.unique_new if x == 0.0 else hybrid.unique_old)
    kj, nm = u.kilojoule_per_mole, u.nanometer
    bonds = {}
    for a, b in hybrid.bonds:
        bonds.setdefault(a, set()).add(b)
        bonds.setdefault(b, set()).add(a)
    dummy_set, core = set(int(d) for d in dummies), set(hybrid.core)
    groups = []                                    # (X, bridge D, atoms of the group)
    for d in sorted(dummy_set):
        for xa in sorted(bonds.get(d, ())):
            if xa in core:
                group, todo = {d}, [d]
                while todo:
                    a = todo.pop()
                    for b in bonds.get(a, ()):
                        if b in dummy_set and b not in group:
                            group.add(b)
                            todo.append(b)
                groups.append((xa, d, sorted(group)))

    def turned(y, axis_from, axis_to, atoms, degrees):
        k = (y[axis_to] - y[axis_from]) / np.linalg.norm(y[axis_to] - y[axis_from])
        t = math.radians(degrees)
        v = y[atoms] - y[axis_from]
        out = y.copy()
        out[atoms] = y[axis_from] + v * math.cos(t) + np.cross(k, v) * math.sin(t) \
            + np.outer(v @ k, k) * (1 - math.cos(t))
        return out

    def energy(flat):
        y = positions.copy()
        y[dummies] = flat.reshape(-1, 3)
        ctx.setPositions(y)
        st = ctx.getState(getEnergy=True, getForces=True)
        return st.getPotentialEnergy().value_in_unit(kj), \
            -np.asarray(st.getForces(asNumpy=True).value_in_unit(kj / nm))[dummies].reshape(-1)

    bases = [positions.copy(), positions.copy()]
    bases[1][dummies] = np.asarray(hybrid.positions, dtype=float)[dummies]
    starts = []
    for base in bases:
        starts.append(base[dummies])
        for xa, d, group in groups:
            for a, b in [(xa, d)] + [(xa, p) for p in sorted(bonds.get(xa, ())) if p in core]:
                for degrees in (120.0, 240.0):
                    starts.append(turned(base, a, b, group, degrees)[dummies])
    best = min((minimize(energy, s.reshape(-1), jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "gtol": 1e-7}) for s in starts), key=lambda r: r.fun)
    out = positions.copy()
    out[dummies] = best.x.reshape(-1, 3)
    return out


def _ligand_bonds(system, atoms) -> dict:
    atoms, out = set(atoms), {}
    pairs = [system.getConstraintParameters(k)[:2] for k in range(system.getNumConstraints())]
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            pairs += [f.getBondParameters(k)[:2] for k in range(f.getNumBonds())]
    for i, j in pairs:
        if i in atoms and j in atoms:
            out.setdefault(i, set()).add(j)
            out.setdefault(j, set()).add(i)
    return out


def _turned(positions: np.ndarray, bonds: dict, i: int, j: int, degrees: float) -> np.ndarray:
    """``positions`` with the atoms on the side of j of the bond i-j turned about it."""
    side, todo = {j}, [j]
    while todo:
        a = todo.pop()
        for b in bonds.get(a, ()):
            if b != i and b not in side:
                side.add(b)
                todo.append(b)
    side = sorted(side - {j})
    k = (positions[j] - positions[i]) / np.linalg.norm(positions[j] - positions[i])
    t = math.radians(degrees)
    v = positions[side] - positions[j]
    out = positions.copy()
    out[side] = positions[j] + v * math.cos(t) + np.cross(k, v) * math.sin(t) + np.outer(v @ k, k) * (1 - math.cos(t))
    return out


def _excess_at_the_end(edge, hybrid, x: float, configurations) -> list[float]:
    """E(hybrid at the end x) - E(the System of the ligand of that end) for each configuration of the hybrid (nm; the
    dummy atoms where their minimisation starts), the dummy atoms at their minimum: what the hybrid adds to the ligand
    of the end."""
    from cufep.rbfe.states import HybridState

    system_a, _pa, _la, system_b, _pb, _lb, _core, box = edge
    system, to_h = (system_a, hybrid.a_to_hybrid) if x == 0.0 else (system_b, hybrid.b_to_hybrid)
    out = []
    for config in configurations:
        positions = _dummies_at_their_minimum(hybrid, x, np.asarray(config, dtype=float))
        own = np.array([positions[to_h[k]] for k in range(system.getNumParticles())])
        ctx = reference_context(hybrid.system, positions, box)
        _set(ctx, HybridState(x))
        out.append(potential(ctx) - potential(reference_context(system, own, box)))
    return out


def _moved_at_random(hybrid, x: float, seed: int = 3):
    """The prepared hybrid configuration and three with the atoms of the ligand of the end ``x`` moved at random."""
    base = np.asarray(hybrid.positions, dtype=float)
    atoms = list(hybrid.atoms_a if x == 0.0 else hybrid.atoms_b)
    rng = np.random.default_rng(seed)
    out = [base]
    for _ in range(3):
        moved = base.copy()
        moved[atoms] += rng.normal(0.0, 0.004, (len(atoms), 3))
        out.append(moved)
    return out


@pytest.mark.parametrize("rest2", [False, True])
def test_the_ends_of_the_hybrid_are_the_two_ligands(edge, rest2):
    """At lambda 0 the hybrid System has the energy of System A plus that of the dummy atoms (the unique atoms of B,
    which interact with nothing), at lambda 1 that of System B plus that of the unique atoms of A: with the dummies at
    their minimum, what the hybrid adds is the same wherever the atoms of the ligand of the end are (the ligand's atoms
    moved at random). Toluene into phenol: the methyl group and the hydroxyl are on one ring carbon, a dual junction of
    cufep.rbfe.junction; version 2 of the hybrid, which kept every term of the dummies, added from 0.37 to 1.05 kJ/mol
    at lambda 0 over these configurations. With REST2 (the ends are not scaled) the same, with water molecules in the
    hot region."""
    _system_a, _pa, ligand_a, *_rest = edge
    hot = list(range(len(ligand_a), len(ligand_a) + 9)) if rest2 else []
    hybrid = _hybrid(edge, rest2=rest2, rest2_atoms=hot)
    for x in (0.0, 1.0):
        excess = _excess_at_the_end(edge, hybrid, x, _moved_at_random(hybrid, x))
        assert max(excess) - min(excess) < 2e-3, (x, excess)


#: three shapes of a mapped atom with unmapped atoms of both ligands (RDKit's AddHs order of the atoms; the mapping of
#: A onto B), and the bond of that atom turned: N-methylpropanamide into 1,3-dimethylurea (the carbonyl carbon carries
#: the ethyl of A and the NH-CH3 of B: dual junction, the C-N bond of the amide turned, as ejm_42~ejm_54 of TYK2),
#: chloroethane into ethane (the CH2Cl carbon carries Cl and an H of B: triple junction, the C-C bond turned) and
#: toluene into (trifluoromethyl)benzene (the methyl carbon carries the hydrogens of A and the fluorines of B: terminal
#: junction, the bond to the ring turned)
JUNCTIONS = {
    "dual": ("CCC(=O)NC", "CNC(=O)NC", {2: 2, 3: 3, 4: 4, 5: 5, 11: 10, 12: 11, 13: 12, 14: 13}, (4, 2)),
    "triple": ("CCCl", "CC", {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}, (0, 1)),
    "terminal": ("Cc1ccccc1", "FC(F)(F)c1ccccc1",
                 {0: 1, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14}, (1, 0)),
}


@pytest.mark.parametrize("shape", sorted(JUNCTIONS))
def test_a_rotation_of_a_ligand_at_a_mapped_atom_with_unmapped_atoms_of_both_does_not_turn_the_dummies(shape):
    """Where a mapped atom carries unmapped atoms of both ligands, the ligand of an end turns about the bonds of that
    atom as it would alone: with the dummy atoms at their minimum, what the hybrid adds is the same at every angle
    (Fleck, Wieder and Boresch 2021: three nonredundant bonded terms per dummy atom). Version 2 of the hybrid kept every
    term of the dummies, whose dihedrals into the other ligand made the dummy follow the turn against them (TYK2
    ejm_42~ejm_54: 34.9 kJ/mol more at 90 degrees than at 0 on the urea C-N bond of ejm_54; here, on the code before
    the change, at lambda 0 from 24.97 to 59.74 kJ/mol over the angles for the dual junction, from 3.22 to 6.31 for
    the triple); the terminal junction turned no dummy before either."""
    smiles_a, smiles_b, mapping, (i, j) = JUNCTIONS[shape]
    edge = build_hybrid_edge(smiles_b, mapping, smiles_a=smiles_a)
    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    hybrid = _hybrid(edge)
    assert any(shape in note for note in hybrid.dummy_junctions), hybrid.dummy_junctions
    bonds = {}
    for a, b in hybrid.bonds:
        bonds.setdefault(a, set()).add(b)
        bonds.setdefault(b, set()).add(a)
    base = np.asarray(hybrid.positions, dtype=float)
    for x in (0.0, 1.0):
        # the atoms of both ligands on the side of j turn, the dummy atoms as the start of their minimisation
        turned = [_turned(base, bonds, ligand_a[i], ligand_a[j], d) for d in (0, 60, 90, 120, 180)]
        excess = _excess_at_the_end(edge, hybrid, x, turned)
        assert max(excess) - min(excess) < 2e-3, (shape, x, excess)


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
            "steps_per_sample": 1, "minimize_steps": 500, "protocol": "hybrid"}
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
    # at each end the hybrid gains what its System gains (the hybrid of the plain Systems is the reference, so that
    # the dummy atoms, whose terms cufep.rbfe.junction sets, count the same on both sides)
    plain = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    hybrid = make_hybrid_system(sys_a, pos_a, ligand_a, sys_b, pos_b, ligand_b, core)
    positions = hybrid.positions
    positions_b = np.array([positions[hybrid.b_to_hybrid[j]] for j in range(sys_b.getNumParticles())])
    ends = {0.0: (sys_a, system_a, positions[: sys_a.getNumParticles()]), 1.0: (sys_b, system_b, positions_b)}
    for x, (doubled, alone, own) in ends.items():
        ctx, ref = reference_context(hybrid.system, positions, box), reference_context(plain.system, positions, box)
        _set(ctx, HybridState(x))
        _set(ref, HybridState(x))
        gained = potential(reference_context(doubled, own, box)) - potential(reference_context(alone, own, box))
        assert abs(gained) > 1.0
        assert potential(ctx) - potential(ref) == pytest.approx(gained, abs=2e-3), x


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
