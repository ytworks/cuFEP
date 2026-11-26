"""Grand canonical moves of water in the complex leg of a relative calculation (on by default): the hybrid System
with waters switched off in a Context is the hybrid System built without them, at every state; the moves run in the
states of the hybrid; the setting is of the complex leg only. On the CPU with stock OpenMM."""
from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from conftest import build_hybrid_edge, hybrid_edge_topology

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields")


@pytest.fixture(scope="module")
def edge():
    return build_hybrid_edge()


def _rbfe_config(**sampling):
    from cufep.rbfe.config import RbfeConfig

    return RbfeConfig.model_validate({"system": {"protein": "p.pdb", "ligands": "l.sdf"},
                                      "sampling": sampling or {}})


def _energies(system, positions, box, states) -> list[float]:
    from cufep.rbfe.states import HybridState

    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    ctx.setPositions(positions)
    out = []
    for lam in states:
        for name, value in HybridState(lam, 1.0).as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        out.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole))
    return out, ctx


def test_waters_switched_off_in_a_context_of_the_hybrid_are_waters_it_was_built_without(edge):
    """The moves switch waters in the Context of a hybrid System (charge and epsilon in the NonbondedForce, epsilon of
    both end states in the Lennard-Jones of the ligand with its environment). At lambda 0, 0.3, 0.7 and 1 the
    Context gives the energy of the hybrid System made from end states without those waters, and of the hybrid
    System whose ghosts were switched off after it was made (as a leg makes it); before the fix of the switch the
    Lennard-Jones of a ghost with the ligand stayed."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_gcmc, molecules_of
    from cufep.sampling.gcmc import WaterSwitch, ghosts_off

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    n = hybrid.system.getNumParticles()
    topology = hybrid_edge_topology(n, list(hybrid.ligand_atoms), hybrid.system)
    rcfg = _rbfe_config(gcmc={"ghost_waters": 6})
    spec = hybrid_gcmc(rcfg, hybrid, topology, box)
    assert spec.stand_in_lambdas == (1.0, 1.0) and len(spec.ghosts) == 6
    # the ligand of the stand-in energy has the parameters of both end states
    unique_new = hybrid.unique_new[0]
    assert spec.epsilon[unique_new] == pytest.approx(hybrid.parameters_b[unique_new][2])
    ghosts = spec.waters[spec.ghosts].reshape(-1)
    # the hybrid made from end states without the ghosts (environment indices of A; B has its ligand last)
    off_a = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_a))
    off_b = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_b))
    ghosts_off(off_a, ghosts)
    ghosts_off(off_b, ghosts - len(ligand_a))
    built_without = make_hybrid_system(off_a, pos_a, ligand_a, off_b, pos_b, ligand_b, core).system
    switched_after = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(hybrid.system))
    ghosts_off(switched_after, ghosts)
    lambdas = (0.0, 0.3, 0.7, 1.0)
    expected, _ = _energies(built_without, hybrid.positions, box, lambdas)
    every, ctx = _energies(hybrid.system, hybrid.positions, box, lambdas)
    assert max(abs(a - b) for a, b in zip(every, expected)) > 1.0                # kJ/mol: six waters
    after, _ = _energies(switched_after, hybrid.positions, box, lambdas)
    assert np.allclose(after, expected, rtol=0.0, atol=1e-6)
    switch = WaterSwitch(ctx, spec, on=np.ones(len(spec.waters), dtype=bool))
    assert switch.apply(spec.initial_on()) == 6
    got = []
    from cufep.rbfe.states import HybridState

    for lam in lambdas:
        for name, value in HybridState(lam, 1.0).as_parameters().items():
            if name in ctx.getParameters():
                ctx.setParameter(name, value)
        got.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole))
    assert np.allclose(got, expected, rtol=0.0, atol=1e-6)
    # the molecules: the whole hybrid ligand is one, every water one
    molecules = molecules_of(hybrid.system, topology, hybrid.bonds)
    ligand = next(m for m in molecules if hybrid.ligand_atoms[0] in m)
    assert sorted(ligand) == list(hybrid.ligand_atoms) and len(molecules) == 1 + len(spec.waters)


def test_moves_are_made_in_the_states_of_the_hybrid(edge):
    """A move in a state of the hybrid (which has no lambda_electrostatics of its own) takes the stand-in energy of
    the ligand at the fixed lambdas, counts its trials, and leaves the Context with the waters it says interact."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_gcmc
    from cufep.rbfe.states import HybridState
    from cufep.sampling.gcmc import GcmcMover, ghosts_off

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    topology = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system)
    spec = hybrid_gcmc(_rbfe_config(gcmc={"attempts": 200, "batch_size": 50}), hybrid, topology, box)
    ghosts_off(hybrid.system, spec.waters[spec.ghosts].reshape(-1))
    state = HybridState(0.4, 1.0)
    _, ctx = _energies(hybrid.system, hybrid.positions, box, (0.4,))
    mover = GcmcMover(spec, ctx, seed=11)
    counts = mover.attempt(state)
    assert counts["insertions_attempted"] + counts["deletions_attempted"] >= 50
    on = mover.on
    rebuilt = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(hybrid.system))
    from cufep.sampling.gcmc import every_water_on

    every_water_on(rebuilt, spec)
    ghosts_off(rebuilt, spec.waters[~on].reshape(-1))
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True)
    expected, _ = _energies(rebuilt, x, box, (0.4,))
    assert ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole) == \
        pytest.approx(expected[0], abs=1e-6)


def test_the_moves_are_on_by_default_in_the_complex_leg_only_and_recorded_there():
    from cufep.rbfe.leg import hybrid_settings
    from cufep.workflow import gcmc_in

    rcfg = _rbfe_config()
    assert rcfg.sampling.gcmc.enabled and gcmc_in(rcfg, "complex") and not gcmc_in(rcfg, "solvent")
    assert "gcmc" in hybrid_settings(rcfg, "complex") and "gcmc" not in hybrid_settings(rcfg, "solvent")
    assert hybrid_settings(rcfg, "complex")["gcmc"]["cell_pressure_bar"] == rcfg.thermodynamics.pressure_bar
    off = _rbfe_config(gcmc={"enabled": False})
    assert "gcmc" not in hybrid_settings(off, "complex")
    # a section that gives other settings of the moves keeps them on (they went off when only `attempts` was given)
    assert _rbfe_config(gcmc={"attempts": 5000}).sampling.gcmc.enabled
