"""REST2 on a part of the hybrid ligand (sampling.rest2.region: perturbed): the rule of the hot atoms on molecules with
a known answer, the energies of every state against a System with the REST2 scaling written into its parameters
(charges of hot atoms times sqrt(s), their epsilon times s, the torsions of four hot atoms times s), the evaluation of
all states, and the System of the whole ligand hot unchanged. The charges of an alchemical water are cold. On the CPU
with stock OpenMM."""
from __future__ import annotations

import math

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from conftest import build_hybrid_edge, hybrid_edge_topology, potential, reference_context

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields")


def _pair(a_smiles: str, b_smiles: str):
    """Two molecules with their hydrogens, B the heavy skeleton of A plus its own atoms: the hybrid indices (A as it
    is, the atoms unique to B after it) and the mapped atoms (heavy atoms by the skeleton, hydrogens by their parents
    in order)."""
    from rdkit import Chem

    a, b = Chem.AddHs(Chem.MolFromSmiles(a_smiles)), Chem.AddHs(Chem.MolFromSmiles(b_smiles))
    skeleton = Chem.RemoveHs(a)
    heavy = dict(zip(range(skeleton.GetNumAtoms()), b.GetSubstructMatch(skeleton)))
    mapping = dict(heavy)
    for i, j in heavy.items():
        ha = [n.GetIdx() for n in a.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1]
        hb = [n.GetIdx() for n in b.GetAtomWithIdx(j).GetNeighbors() if n.GetAtomicNum() == 1]
        mapping.update(zip(ha, hb))
    to_a = {i: i for i in range(a.GetNumAtoms())}
    unique_b = [j for j in range(b.GetNumAtoms()) if j not in set(mapping.values())]
    to_b = {j: i for i, j in mapping.items()}
    to_b.update({j: a.GetNumAtoms() + k for k, j in enumerate(unique_b)})
    return a, b, to_a, to_b, set(mapping)


def test_the_hot_atoms_grow_from_the_perturbation_along_rotatable_bonds_and_rings():
    """Butylbenzene into 4-butylphenol: the para hydrogen (unique to A) and the hydroxyl (unique to B) are on the ring
    by bonds that are not rotatable, so the ring with its hydrogens is added; from the ring the rotatable bonds of the
    chain add its CH2 groups with their hydrogens; the bond to the methyl group is not rotatable: the methyl stays cold.
    A chain of 28 carbons stops at 30 heavy atoms (the limit with a ring)."""
    from cufep.rbfe.hot_region import MAX_HEAVY_ATOMS_WITH_RING, perturbed_hot_atoms

    a, b, to_a, to_b, core = _pair("CCCCc1ccccc1", "CCCCc1ccc(O)cc1")
    hot = perturbed_hot_atoms(a, b, to_a, to_b, core)
    methyl = {0} | {n.GetIdx() for n in a.GetAtomWithIdx(0).GetNeighbors() if n.GetAtomicNum() == 1}
    everything = set(to_a.values()) | set(to_b.values())
    assert everything - hot == methyl
    long_a, long_b, to_a, to_b, core = _pair("C" * 28 + "c1ccccc1", "C" * 28 + "c1ccc(O)cc1")
    hot = perturbed_hot_atoms(long_a, long_b, to_a, to_b, core)
    heavy = {i for i in range(long_a.GetNumAtoms()) if long_a.GetAtomWithIdx(i).GetAtomicNum() > 1}
    heavy |= {to_b[j] for j in range(long_b.GetNumAtoms()) if long_b.GetAtomWithIdx(j).GetAtomicNum() > 1}
    assert len(hot & heavy) == MAX_HEAVY_ATOMS_WITH_RING
    chain_hot = sorted(i for i in hot & heavy if i < 28)
    assert chain_hot == list(range(28 - len(chain_hot), 28))                # from the ring outwards
    # second review of PR #19: from a perturbation at the end of a chain of 23 carbons the region reaches 24 heavy
    # atoms before the ring; one more atom, the first of the ring, is not "a ring", so the limit is 24 and the ring
    # stays cold (it was let in, under the limit of 30)
    from cufep.rbfe.hot_region import MAX_HEAVY_ATOMS

    a, b, to_a, to_b, core = _pair("c1ccccc1" + "C" * 23, "c1ccccc1" + "C" * 23 + "O")
    hot = perturbed_hot_atoms(a, b, to_a, to_b, core)
    element = {to_b[j]: b.GetAtomWithIdx(j).GetAtomicNum() for j in range(b.GetNumAtoms())}
    element.update({i: a.GetAtomWithIdx(i).GetAtomicNum() for i in range(a.GetNumAtoms())})
    heavy = {i for i in hot if element[i] > 1}
    assert len(heavy) == MAX_HEAVY_ATOMS and not heavy & set(range(6))


def test_a_bond_that_closes_a_ring_is_read_in_the_end_state_that_has_it():
    """Eleventh review of PR #19: propane into cyclopropane with the three carbons mapped. The bond of the two end
    carbons is in B only; it was read in A (both atoms are atoms of A), where it is neither rotatable nor a bond to a
    ring, and the region stayed empty. Read in B it is a bond to a ring: the ring and every hydrogen is hot."""
    from cufep.rbfe.hot_region import perturbed_hot_atoms

    a, b, to_a, to_b, core = _pair("CCC", "C1CC1")
    assert {to_a[0], to_a[1], to_a[2]} <= core
    hot = perturbed_hot_atoms(a, b, to_a, to_b, core)
    assert hot == set(to_a.values()) | set(to_b.values())
    assert perturbed_hot_atoms(a, b, to_a, to_b, core, rule=1) == set()     # version 1, which legs record otherwise


def _scaled(system, hot: set[int], s: float, ligand: set[int]):
    """``system`` with the REST2 scaling of the atoms ``hot`` written into its parameters: charge times sqrt(s),
    epsilon times s; the exceptions of the ligand times the factors of their atoms; the torsions of four hot atoms
    times s."""
    out = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    root = math.sqrt(s)
    for force in out.getForces():
        if isinstance(force, mm.NonbondedForce):
            for i in hot:
                q, sig, eps = force.getParticleParameters(i)
                force.setParticleParameters(i, q * root, sig, eps * s)
            for k in range(force.getNumExceptions()):
                i, j, qq, sig, eps = force.getExceptionParameters(k)
                f = (root if i in hot else 1.0) * (root if j in hot else 1.0)
                if i in ligand or j in ligand:
                    force.setExceptionParameters(k, i, j, qq * f, sig, eps * f)
        elif isinstance(force, mm.PeriodicTorsionForce):
            for k in range(force.getNumTorsions()):
                i, j, m, n, per, phase, kk = force.getTorsionParameters(k)
                if all(x in hot for x in (i, j, m, n)):
                    force.setTorsionParameters(k, i, j, m, n, per, phase, kk * s)
    return out


def _change_a_core_torsion(system_b, ligand_b: list[int], core: dict[int, int]) -> int:
    """Double the force constant of the first proper torsion of four core atoms of B and turn its phase by pi, so that
    it carries energy in the pose of A (a term the hybrid then holds once for A and once for B); returns how many
    terms changed."""
    core_b = set(core.values())
    for force in system_b.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            for k in range(force.getNumTorsions()):
                i, j, m, n, per, phase, kk = force.getTorsionParameters(k)
                if all(x in core_b for x in (i, j, m, n)) and kk.value_in_unit(u.kilojoule_per_mole) > 0.5:
                    force.setTorsionParameters(k, i, j, m, n, per, phase + math.pi * u.radian, 2.0 * kk)
                    return 1
    return 0


def _at(ctx, state) -> float:
    for name, value in state.as_parameters().items():
        if name in ctx.getParameters():
            ctx.setParameter(name, value)
    return potential(ctx)


def _ring_and_unique(hybrid) -> list[int]:
    """A part of the ligand of the test edge: the unique atoms and the ring carbons (the ring hydrogens cold)."""
    return sorted(set(hybrid.unique_old) | set(hybrid.unique_new) | {1, 2, 3, 4, 5, 6})


@pytest.mark.parametrize("case", ["part of the ligand", "alchemical water", "a core torsion that changes"])
def test_the_states_have_the_energy_of_the_rest2_scaling_written_into_the_system(case):
    """At x 0.3 and 0.7 and REST2 scale 1.5 (s = 1/1.5) the hybrid with REST2 on a part of the ligand (or on the whole
    ligand with an alchemical water, whose charges are cold) has the energy of the hybrid without REST2 made from end
    states whose hot atoms carry the scaling in their parameters; at the ends, the energy without scaling."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridState

    if case == "alchemical water":
        from cufep.rbfe.charge import alchemical_water, ion_parameters, water_into_ion

        ring = {**{i: i for i in range(1, 7)}, **{10 + k: 7 + k for k in range(5)}}
        system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge("[O-]c1ccccc1", ring)
        top_a = hybrid_edge_topology(system_a.getNumParticles(), ligand_a, system_a, box)
        water = alchemical_water(top_a, pos_a, 0.8)
        water_into_ion(system_b, [i - len(ligand_a) for i in water],
                       ion_parameters(system_a, top_a, +1, mm.app.ForceField("amber14/tip3p.xml")))
        more = {"alchemical_water": water}
    else:
        system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge()
        more = {}
        if case == "a core torsion that changes":
            # review of PR #19: the REST2 factor multiplied the term of A of a core torsion alone
            assert _change_a_core_torsion(system_b, ligand_b, core) == 1
    plain = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, **more)
    hot = _ring_and_unique(plain) if case == "part of the ligand" else list(plain.ligand_atoms)
    rest2 = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True,
                               rest2_ligand_atoms=hot if case == "part of the ligand" else None, **more)
    assert (set(rest2.rest2_ligand_atoms) != set(rest2.ligand_atoms)) == (case == "part of the ligand")
    ctx = reference_context(rest2.system, rest2.positions, box)
    s = 1.0 / 1.5
    h_to_b = {h: j for j, h in rest2.b_to_hybrid.items()}
    hot_a = {h for h in hot if h < system_a.getNumParticles()}
    hot_b = {h_to_b[h] for h in hot if h in h_to_b}
    scaled = make_hybrid_system(_scaled(system_a, hot_a, s, set(ligand_a)), pos_a, ligand_a,
                                _scaled(system_b, hot_b, s, set(ligand_b)), pos_b, ligand_b, core, **more)
    expected_ctx = reference_context(scaled.system, scaled.positions, box)
    unscaled_ctx = reference_context(plain.system, plain.positions, box)
    for x in (0.3, 0.7):
        assert _at(ctx, HybridState(x, 1.5)) == pytest.approx(_at(expected_ctx, HybridState(x)), abs=2e-3)
    for x in (0.0, 1.0):
        assert _at(ctx, HybridState(x)) == pytest.approx(_at(unscaled_ctx, HybridState(x)), abs=2e-3)


@pytest.mark.parametrize("case", ["part of the ligand", "alchemical water", "a core torsion that changes"])
def test_the_evaluator_gives_every_rest2_state_of_a_hybrid_with_cold_charges(case):
    """The evaluation of all states (nine evaluations of the main group per half, the others in a = sqrt(s)) equals
    setting each state, along a leg with REST2 2.0 in the middle, after MD in a scaled state. Before the evaluation in
    a, the Lennard-Jones of an alchemical water with its environment was scaled with the ligand's."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridEnergyEvaluator, HybridState
    from cufep.sampling.states import R_GAS

    if case == "alchemical water":
        from cufep.rbfe.charge import alchemical_water, ion_parameters, water_into_ion

        ring = {**{i: i for i in range(1, 7)}, **{10 + k: 7 + k for k in range(5)}}
        system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge("[O-]c1ccccc1", ring)
        top_a = hybrid_edge_topology(system_a.getNumParticles(), ligand_a, system_a, box)
        water = alchemical_water(top_a, pos_a, 0.8)
        water_into_ion(system_b, [i - len(ligand_a) for i in water],
                       ion_parameters(system_a, top_a, +1, mm.app.ForceField("amber14/tip3p.xml")))
        hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True,
                                    alchemical_water=water)
    elif case == "a core torsion that changes":
        system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge()
        assert _change_a_core_torsion(system_b, ligand_b, core) == 1
        hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True)
    else:
        system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge()
        plain = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
        hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True,
                                    rest2_ligand_atoms=_ring_and_unique(plain))
    integ = mm.LangevinMiddleIntegrator(298.15 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(3)
    ctx = mm.Context(hybrid.system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(hybrid.positions)
    _at(ctx, HybridState(0.5, 1.6))
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
    integ.step(50)
    states = fixed_schedule(["morph"], {"morph": 8}, rest2_scale=2.0).states
    evaluator = HybridEnergyEvaluator(ctx, states, 298.15, None)
    assert evaluator.by_atom == (case != "a core torsion that changes")
    fast, _ = evaluator.reduced_potentials()
    beta = 1.0 / (R_GAS * 298.15)
    slow = np.array([beta * _at(ctx, state) for state in states])
    assert np.allclose(fast, slow, rtol=0, atol=2e-6 * np.max(np.abs(slow)))


def test_the_whole_ligand_hot_is_the_system_of_before_and_the_absolute_calculation_refuses_the_region():
    from cufep.config.schema import Config
    from cufep.rbfe.config import RbfeConfig
    from cufep.rbfe.hybrid import COLD_ELECTROSTATIC_LAMBDAS, make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge()
    before = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True)
    named = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True,
                               rest2_ligand_atoms=before.ligand_atoms)
    assert mm.XmlSerializer.serialize(named.system) == mm.XmlSerializer.serialize(before.system)
    (nb,) = [f for f in before.system.getForces() if isinstance(f, mm.NonbondedForce)]
    names = {nb.getGlobalParameterName(k) for k in range(nb.getNumGlobalParameters())}
    assert not names & set(COLD_ELECTROSTATIC_LAMBDAS)
    part = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=True,
                              rest2_ligand_atoms=_ring_and_unique(before))
    (nb,) = [f for f in part.system.getForces() if isinstance(f, mm.NonbondedForce)]
    assert nb.getNumGlobalParameters() == 8                      # three hot and three cold charges, rest2_dq and _ds
    RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"},
                               "sampling": {"rest2": {"region": "perturbed"}}})
    with pytest.raises(ValueError, match="region of a relative calculation"):
        Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                               "sampling": {"rest2": {"region": "perturbed"}}})
