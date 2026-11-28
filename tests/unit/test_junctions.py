"""The junctions of a hybrid topology (cufep.rbfe.junctions) and ``alchemy.dummy_bonded_terms: separated``: which
junction is which, which terms go to zero at an end, and that the dummies of a terminal junction then contribute a
constant to the partition function whatever the physical atoms do."""
from __future__ import annotations

import math

import numpy as np
import openmm as mm
import pytest
from conftest import build_hybrid_edge, potential, reference_context

from cufep.rbfe.junctions import junctions_of, removed_torsions

KT = 8.31446261815324e-3 * 300.0


# ---- which junction is which --------------------------------------------------------------------------------
# A methyl group on R (0) becomes a nitrile: X (4) is the carbon of both, 5, 6, 7 the hydrogens of A, 8 the nitrogen
# of B. R holds Q1 (1, which has a further atom 9), Q2 (2) and a hydrogen (3)
CORE = {0, 1, 2, 3, 4, 9}
SHARED = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 9)}
BONDS_A = SHARED | {(4, 5), (4, 6), (4, 7)}
BONDS_B = SHARED | {(4, 8)}


def test_a_junction_is_named_by_the_core_atoms_on_the_bridge_in_the_ligand_of_the_dummies():
    """The methyl carbon has four physical neighbours at the end of A, and the dummy nitrogen is tied to one of them
    by the terms of B: a terminal junction, as the three dummy hydrogens at the end of B are. Of the atoms two bonds
    from the bridge, the dihedrals through the one with the most bonds (then the lowest number) are kept."""
    (at_a,) = junctions_of("A", BONDS_B, CORE, {8})
    assert (at_a.bridge, at_a.dummies, at_a.core_neighbours, at_a.kind) == (4, (8,), (0,), "terminal")
    assert at_a.separated and at_a.kept_through == 1 and at_a.removed_through == (2, 3)
    (at_b,) = junctions_of("B", BONDS_A, CORE, {5, 6, 7})
    assert (at_b.bridge, at_b.dummies, at_b.kind, at_b.kept_through, at_b.removed_through) == (
        4, (5, 6, 7), "terminal", 1, (2, 3))
    assert at_b.to_dict()["end"] == "B"
    # the terms of A that go to zero at the end of B: the proper torsions Q-R-X-D through 2 and 3, either way round
    torsions = [(1, 0, 4, 5), (2, 0, 4, 5), (6, 4, 0, 2), (3, 0, 4, 7), (9, 1, 0, 4), (0, 4, 5, 6), (5, 4, 0, 1)]
    assert removed_torsions([at_b], torsions) == [1, 2, 3]
    assert removed_torsions([at_a], torsions) == []                         # the nitrogen has none of them


def test_junctions_the_rule_does_not_treat_are_named_and_left():
    """Two or three core atoms on the bridge (a dual, a triple junction), and dummies bonded to more than one core
    atom (a ring that closes at one end): named, and none of their terms is removed."""
    # a hydrogen (5) of an aromatic carbon X (0) with ring neighbours 1 and 2
    (dual,) = junctions_of("B", {(0, 1), (0, 2), (0, 5), (1, 3), (2, 4)}, {0, 1, 2, 3, 4}, {5})
    assert (dual.kind, dual.separated, dual.core_neighbours) == ("dual", False, (1, 2))
    # a methyl carbon (5, with hydrogens 6, 7, 8) on an sp3 carbon X (0) that keeps three core neighbours
    (triple,) = junctions_of("A", {(0, 1), (0, 2), (0, 3), (0, 5), (5, 6), (5, 7), (5, 8)}, {0, 1, 2, 3}, {5, 6, 7, 8})
    assert (triple.kind, triple.separated, triple.dummies) == ("triple", False, (5,))
    # dummies 5-6 bonded to the core atoms 0 and 3
    ring = junctions_of("A", {(0, 1), (1, 2), (2, 3), (0, 5), (5, 6), (6, 3)}, {0, 1, 2, 3}, {5, 6})
    assert [j.bridge for j in ring] == [0, 3] and not any(j.separated for j in ring)
    assert all("more than one core atom" in j.kind for j in ring)
    for junction in (dual, triple, *ring):
        assert removed_torsions([junction], [(1, 0, 5, 6), (2, 0, 5, 6), (3, 0, 5, 6)]) == []


# ---- the dummies of a terminal junction contribute a constant ------------------------------------------------
K_TORSION = 6.0                                    # kJ/mol: 2.4 kT, so that the redundant term matters


def _toy(delta_deg: float, phi_deg: float = 0.0):
    """Q1, Q2 on R, the bridge X on R, and D on X (atoms 0 to 4): positions with Q2 turned by ``delta_deg`` from Q1
    about the axis R-X and D by ``phi_deg``; System A has D with the torsions Q1-R-X-D and Q2-R-X-D, System B is the
    four other atoms."""
    def around_x(origin, sign, turn_deg):
        theta, turn = math.radians(109.5), math.radians(turn_deg)
        return origin + 0.11 * np.array([sign * math.cos(math.pi - theta), math.sin(theta) * math.cos(turn),
                                         math.sin(theta) * math.sin(turn)])

    r, x = np.array([1.0, 1.0, 1.0]), np.array([1.15, 1.0, 1.0])
    positions = np.array([around_x(r, -1.0, 0.0), around_x(r, -1.0, delta_deg), r, x, around_x(x, 1.0, phi_deg)])

    def system(n: int) -> mm.System:
        s = mm.System()
        for _ in range(n):
            s.addParticle(12.0)
        s.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
        bonds, angles, torsions = mm.HarmonicBondForce(), mm.HarmonicAngleForce(), mm.PeriodicTorsionForce()
        pairs = [(0, 2), (1, 2), (2, 3)] + ([(3, 4)] if n == 5 else [])
        for i, j in pairs:
            bonds.addBond(i, j, 0.11 if 2 in (i, j) and 3 not in (i, j) or 4 in (i, j) else 0.15, 2000.0)
        for i, j, m in [(0, 2, 1), (0, 2, 3), (1, 2, 3)] + ([(2, 3, 4)] if n == 5 else []):
            angles.addAngle(i, j, m, math.radians(109.5), 400.0)
        if n == 5:
            for q in (0, 1):
                torsions.addTorsion(q, 2, 3, 4, 3, 0.0, K_TORSION)
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.PME)
        nb.setCutoffDistance(0.9)
        nb.setUseDispersionCorrection(False)
        for _ in range(n):
            nb.addParticle(0.0, 0.3, 0.0)
        nb.createExceptionsFromBonds(pairs, 0.5, 0.5)
        for force in (bonds, angles, torsions, nb):
            s.addForce(force)
        return s

    return system(5), system(4), positions


@pytest.mark.parametrize("terms, constant", [("all", False), ("separated", True)])
def test_the_dummy_of_a_terminal_junction_contributes_a_constant_whatever_the_physical_atoms_do(terms, constant):
    """At the end at which D is a dummy, the integral of exp(-U/kT) over the turn of D about the bond R-X, relative
    to the physical molecule alone, is the dummy's contribution to the partition function as far as it can depend
    on the physical atoms (its bond and its angle are terms of D with R and X alone).

    With every term kept (all) the two torsions Q1-R-X-D and Q2-R-X-D, K (1 + cos 3 phi) each, add up to
    2 K + 2 K cos(3 delta / 2) cos(3 phi - 3 delta / 2), delta the turn between Q1 and Q2: the integral is
    2 pi exp(-2 K / kT) I0(2 K cos(3 delta / 2) / kT) and changes with delta, a physical coordinate. By the rule of
    Fleck et al. for a terminal junction (separated) the torsion through Q2 is off at this end, and the integral
    is 2 pi exp(-K / kT) I0(K / kT) for every delta. At the other end, where D is physical, both are on."""
    from scipy.special import i0

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridState

    box = 3.0 * np.eye(3)
    n = 720
    logs, expected = [], []
    for delta in (120.0, 100.0, 80.0):
        system_a, system_b, positions = _toy(delta)
        hybrid = make_hybrid_system(system_a, positions, [0, 1, 2, 3, 4], system_b, positions[:4], [0, 1, 2, 3],
                                    {i: i for i in range(4)}, dummy_bonded_terms=terms)
        assert hybrid.unique_old == (4,) and hybrid.unique_new == ()
        ctx = reference_context(hybrid.system, positions, box)
        physical = reference_context(system_b, positions[:4], box)
        for name, value in HybridState(1.0).as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        weights = []
        for k in range(n):
            x = _toy(delta, 360.0 * k / n)[2]
            ctx.setPositions(x)
            weights.append(math.exp(-(potential(ctx) - potential(physical)) / KT))
        logs.append(math.log(2.0 * math.pi * float(np.mean(weights))))
        amplitude = K_TORSION / KT if constant else 2.0 * K_TORSION * math.cos(math.radians(1.5 * delta)) / KT
        offset = (1.0 if constant else 2.0) * K_TORSION / KT
        expected.append(math.log(2.0 * math.pi * float(i0(amplitude))) - offset)
        # the end at which D is physical: the energy of System A, both torsions on
        for name, value in HybridState(0.0).as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        ctx.setPositions(positions)
        assert potential(ctx) == pytest.approx(potential(reference_context(system_a, positions, box)), abs=1e-6)
        assert hybrid.torsions_off_at_an_end == (1 if constant else 0)
    assert logs == pytest.approx(expected, abs=1e-6)                       # the analytic values
    spread = max(logs) - min(logs)
    if constant:
        assert spread < 1e-9
    else:
        assert spread > 1.5                                                 # kT: 1.99 between 120 and 80 degrees


# ---- a real molecule: the methyl group of toluene becomes a nitrile ---------------------------------------------
#: toluene (Cc1ccccc1) -> benzonitrile (N#Cc1ccccc1): the methyl carbon onto the nitrile carbon, the ring and its
#: hydrogens; the three methyl hydrogens (7, 8, 9) are unique to A, the nitrogen (0) to B
NITRILE_MAP = {0: 1, **{i: i + 1 for i in range(1, 7)}, **{10 + k: 8 + k for k in range(5)}}


@pytest.fixture(scope="module")
def nitrile_edge():
    return build_hybrid_edge(smiles_b="N#Cc1ccccc1", core_b=NITRILE_MAP)


@pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields")
def test_the_ends_of_a_hybrid_with_separated_terms_are_those_of_all_terms_less_the_torsions_that_are_off(nitrile_edge):
    """Both junctions of toluene to benzonitrile are terminal. With ``separated`` the hybrid has, at the end at
    which the methyl hydrogens are dummies (lambda 1), the energy it has with every term kept less that of the
    torsions H-C-C-C through the ring carbon that is not kept (computed from System A), and at lambda 0, where they
    are physical, the same energy as with every term kept. With REST2 the same."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridState

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = nitrile_edge
    for rest2 in (False, True):
        hot = list(range(len(ligand_a), len(ligand_a) + 9)) if rest2 else []
        kept, separated = (
            make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=rest2, rest2_atoms=hot,
                               dummy_bonded_terms=terms) for terms in ("all", "separated"))
        assert kept.junctions == () and kept.torsions_off_at_an_end == 0
        kinds = {(j["end"], j["kind"], len(j["dummies"]), j["separated"]) for j in separated.junctions}
        assert kinds == {("A", "terminal", 1, True), ("B", "terminal", 3, True)}
        (at_b,) = [j for j in separated.junctions if j["end"] == "B"]
        assert at_b["bridge"] == 0 and at_b["core_neighbours"] == [1] and at_b["dummies"] == [7, 8, 9]
        assert {at_b["kept_through"], *at_b["removed_through"]} == {2, 6}             # the two ortho carbons
        # the energy of the torsions of A through the ortho carbon that is not kept
        off = mm.System()
        for _atom in range(system_a.getNumParticles()):
            off.addParticle(1.0)
        off.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())
        (torsions,) = [f for f in system_a.getForces() if isinstance(f, mm.PeriodicTorsionForce)]
        chosen, n_terms = mm.PeriodicTorsionForce(), 0
        for k in range(torsions.getNumTorsions()):
            i, j, m, n, per, phase, kk = torsions.getTorsionParameters(k)
            atoms = (i, j, m, n) if i in at_b["removed_through"] else (n, m, j, i)
            if atoms[0] in at_b["removed_through"] and atoms[1:3] == (1, 0) and atoms[3] in (7, 8, 9):
                chosen.addTorsion(i, j, m, n, per, phase, kk)
                n_terms += 1
        off.addForce(chosen)
        assert separated.torsions_off_at_an_end == n_terms and n_terms >= 3
        removed = potential(reference_context(off, pos_a, box))
        energies = {}
        for name, hybrid in (("all", kept), ("separated", separated)):
            ctx = reference_context(hybrid.system, hybrid.positions, box)
            for lam in (0.0, 1.0):
                for parameter, value in HybridState(lam).as_parameters().items():
                    if parameter in ctx.getParameters():
                        ctx.setParameter(parameter, value)
                energies[name, lam] = potential(ctx)
        assert energies["separated", 0.0] == pytest.approx(energies["all", 0.0], abs=1e-6)
        assert energies["separated", 1.0] == pytest.approx(energies["all", 1.0] - removed, abs=1e-6)


def test_the_default_keeps_every_term_and_builds_the_system_it_built():
    """``dummy_bonded_terms: all`` (the default) is the hybrid as it was: the same System to the byte whether the
    argument is given or not, and a value that is neither is refused."""
    from cufep.rbfe.hybrid import AlchemyError, make_hybrid_system

    system_a, system_b, positions = _toy(120.0)
    args = (system_a, positions, [0, 1, 2, 3, 4], system_b, positions[:4], [0, 1, 2, 3], {i: i for i in range(4)})
    default, given = make_hybrid_system(*args), make_hybrid_system(*args, dummy_bonded_terms="all")
    assert mm.XmlSerializer.serialize(default.system) == mm.XmlSerializer.serialize(given.system)
    separated = make_hybrid_system(*args, dummy_bonded_terms="separated")
    assert mm.XmlSerializer.serialize(separated.system) != mm.XmlSerializer.serialize(default.system)
    with pytest.raises(AlchemyError, match="all or separated"):
        make_hybrid_system(*args, dummy_bonded_terms="none")


def test_the_settings_of_the_samples_name_separated_terms_and_say_nothing_of_the_default():
    """The settings recorded with the samples of a leg: with every term kept (the default) they are those of the
    records made before the setting existed, so that those samples are taken as they are; ``separated`` is named."""
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.leg import hybrid_settings

    system = {"protein": "p.pdb", "ligands": "l.sdf"}
    default = RbfeConfig.model_validate({"system": system})
    assert default.alchemy.dummy_bonded_terms == "all"
    assert hybrid_settings(default, "solvent")["rbfe"]["alchemy"] == {"softcore_alpha": 0.85}
    separated = RbfeConfig.model_validate({"system": system, "alchemy": {"dummy_bonded_terms": "separated"}})
    assert hybrid_settings(separated, "solvent")["rbfe"]["alchemy"] == {
        "softcore_alpha": 0.85, "dummy_bonded_terms": "separated"}
    with pytest.raises(ValueError, match="dummy_bonded_terms"):
        RbfeConfig.model_validate({"system": system, "alchemy": {"dummy_bonded_terms": "none"}})
