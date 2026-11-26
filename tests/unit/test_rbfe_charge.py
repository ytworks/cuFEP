"""Edges that change the net charge (cufep.rbfe.charge): an alchemical water keeps the charge of the box. Toluene (A)
becomes phenolate (B, charge -1) in a box of water; a water far from the ligand is a Na+ ion in B. On the CPU with
stock OpenMM."""
from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from conftest import build_hybrid_edge, hybrid_edge_topology, potential, reference_context, valence_energy

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields", "mdtraj")

PHENOLATE = "[O-]c1ccccc1"
#: toluene -> phenolate: the ring carbons and the five ring hydrogens (phenolate's hydrogens 7 to 11)
PHENOLATE_MAP = {**{i: i for i in range(1, 7)}, **{10 + k: 7 + k for k in range(5)}}


@pytest.fixture(scope="module")
def charged_edge():
    """The edge, the water chosen by the rule in A, the ion of the force field (the box has none) and System B with
    that water as the ion."""
    from cufep.rbfe.charge import alchemical_water, ion_parameters, water_into_ion

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = build_hybrid_edge(PHENOLATE, PHENOLATE_MAP)
    top_a = hybrid_edge_topology(system_a.getNumParticles(), ligand_a, system_a)
    water = alchemical_water(top_a, pos_a, 0.8)
    ion = ion_parameters(system_a, top_a, +1, mm.app.ForceField("amber14/tip3p.xml"))
    shift = len(ligand_a)                      # B: the same waters, its ligand last
    b_with_ion = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_b))
    water_into_ion(b_with_ion, [i - shift for i in water], ion)
    return system_a, pos_a, ligand_a, b_with_ion, pos_b, ligand_b, core, box, water, ion, top_a


def _total_charge(system) -> float:
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    return sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge)
               for i in range(system.getNumParticles()))


def test_the_water_is_the_first_one_far_from_the_solutes_and_becomes_the_ion_of_the_sign(charged_edge):
    """The rule of OpenFE: the water of the lowest residue index with no atom within 0.8 nm of the solutes, in A; in B
    its oxygen is a positive ion for a ligand that gains a negative charge (dq = q(A) - q(B) = +1, Na+ of the force
    field when the box has none) and its hydrogens have charge 0, so that A and B have one total charge."""
    import mdtraj

    from cufep.rbfe.charge import alchemical_water

    system_a, pos_a, ligand_a, b_with_ion, _pb, _lb, _core, box, water, ion, top_a = charged_edge
    assert ion["name"] == "NA" and ion["charge"] == pytest.approx(1.0) and ion["from"] == "the force field"
    traj = mdtraj.Trajectory(np.asarray(pos_a)[None], mdtraj.Topology.from_openmm(top_a))
    traj.unitcell_vectors = np.asarray(box)[None]
    far = mdtraj.compute_distances(traj, [[i, j] for i in ligand_a for j in water]).min()
    assert far >= 0.8
    def nearest(residue) -> float:
        return float(min(mdtraj.compute_distances(traj, [[i, a.index] for i in ligand_a for a in residue.atoms()])[0]))

    first = min((r for r in top_a.residues() if r.name == "HOH" and nearest(r) >= 0.8), key=lambda r: r.index)
    assert water == [a.index for a in first.atoms()]
    assert alchemical_water(top_a, pos_a, 0.8) == water                     # the same water every time
    assert _total_charge(system_a) == pytest.approx(_total_charge(b_with_ion), abs=1e-6)


def test_the_ends_of_a_hybrid_that_changes_the_charge_are_a_and_b_with_the_ion(charged_edge):
    """At lambda 0 the hybrid with the alchemical water has the energy of System A (with the water) plus the bonded
    terms of the unique atoms of B, at lambda 1 that of System B with the ion plus those of the unique atoms of A; its
    total charge is that of both ends at every lambda; the NonbondedForce, which the plugin replaces, holds only
    charges of the water (offsets under lambda_electrostatics_core, no Lennard-Jones)."""
    from cufep.rbfe.hybrid import LAMBDA_ELECTROSTATICS_CORE, make_hybrid_system
    from cufep.rbfe.states import HybridState

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box, water, _ion, _top = charged_edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, alchemical_water=water)
    assert hybrid.alchemical_water == tuple(sorted(water))
    positions = hybrid.positions
    h_to_b = {h: j for j, h in hybrid.b_to_hybrid.items()}
    positions_b = np.array([positions[hybrid.b_to_hybrid[j]] for j in range(system_b.getNumParticles())])
    ctx = reference_context(hybrid.system, positions, box)

    def at(lam: float) -> float:
        for name, value in HybridState(lam).as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        return potential(ctx)

    expected_a = potential(reference_context(system_a, positions[: system_a.getNumParticles()], box)) \
        + valence_energy(system_b, [h_to_b[h] for h in hybrid.unique_new], positions_b, box)
    expected_b = potential(reference_context(system_b, positions_b, box)) \
        + valence_energy(system_a, hybrid.unique_old, positions[: system_a.getNumParticles()], box)
    assert at(0.0) == pytest.approx(expected_a, abs=2e-3) and at(1.0) == pytest.approx(expected_b, abs=2e-3)
    assert abs(expected_a - expected_b) > 10.0                               # kJ/mol: a water is not an ion
    (nb,) = [f for f in hybrid.system.getForces() if isinstance(f, mm.NonbondedForce)]
    assert nb.getParticleParameters(water[0])[2].value_in_unit(u.kilojoule_per_mole) == 0.0
    offsets = [nb.getParticleParameterOffset(k) for k in range(nb.getNumParticleParameterOffsets())]
    assert {(name, i) for name, i, *_ in offsets if i in water} == {(LAMBDA_ELECTROSTATICS_CORE, i) for i in water}
    # the two ends have one total charge (in between, OpenFE's lambda functions take the charges of the unique atoms
    # of A away in the first half and give those of B in the second, as for every edge)
    for lam in (0.0, 1.0):
        total = 0.0
        state = HybridState(lam).as_parameters()
        for i in range(hybrid.system.getNumParticles()):
            total += nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge)
        for name, _atom, dq, _ds, _de in offsets:
            total += state.get(name, 0.0) * dq
        assert total == pytest.approx(_total_charge(system_a), abs=1e-6)


def test_the_evaluator_gives_the_energy_of_every_state_with_the_alchemical_water(charged_edge):
    """The evaluation of all states (a quadratic in each half of lambda for the main group) equals setting each
    state, with the water's charge offsets in the main group."""
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.states import HybridEnergyEvaluator, HybridState
    from cufep.sampling.states import R_GAS

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box, water, _ion, _top = charged_edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, alchemical_water=water)
    integ = mm.LangevinMiddleIntegrator(298.15 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    integ.setRandomNumberSeed(3)
    ctx = mm.Context(hybrid.system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(hybrid.positions)
    for name, value in HybridState(0.5).as_parameters().items():
        if name in ctx.getParameters():
            ctx.setParameter(name, value)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
    integ.step(50)
    states = fixed_schedule(["morph"], {"morph": 8}).states
    fast, _ = HybridEnergyEvaluator(ctx, states, 298.15, None).reduced_potentials()
    beta = 1.0 / (R_GAS * 298.15)
    slow = []
    for state in states:
        for name, value in state.as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        slow.append(beta * potential(ctx))
    assert np.allclose(fast, slow, rtol=0, atol=2e-6 * np.max(np.abs(slow)))


def test_a_change_of_more_than_one_charge_is_refused():
    from cufep.rbfe.charge import charge_change_problems

    class Lig:
        def __init__(self, name, q):
            self.name, self.formal_charge = name, q

    ligands = [Lig("a", 0), Lig("b", -1), Lig("c", -2), Lig("d", 1)]
    assert charge_change_problems(ligands, [("a", "b"), ("b", "d"), ("a", "d")]) == [
        "the edge b -> d changes the net charge by +2 (from -1 to +1); an alchemical water takes a change of one "
        "charge only"]
    assert len(charge_change_problems(ligands, [("a", "c")])) == 1
