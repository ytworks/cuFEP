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


def test_the_sphere_of_an_edge_is_the_same_both_ways(edge):
    """Review of PR #19: the radius of the sphere came from the hybrid, which holds the mapped atoms where A has them;
    with a mapped atom of B 0.1 nm from where A has it, A -> B and B -> A had spheres of other radii (and other Adams
    values). The radius comes from the heavy atoms of both end states: one sphere both ways."""
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_gcmc

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    pos_b = np.array(pos_b, dtype=float)
    centre = pos_a[ligand_a].mean(axis=0)
    far = max((i for i in core if system_a.getParticleMass(i).value_in_unit(mm.unit.dalton) > 2.0),
              key=lambda i: np.linalg.norm(pos_a[i] - centre))                 # the heavy mapped atom farthest out
    outwards = (pos_a[far] - centre) / np.linalg.norm(pos_a[far] - centre)
    pos_b[core[far]] = pos_a[far] + 0.1 * outwards                         # B has it 0.1 nm farther out
    rcfg = _rbfe_config(gcmc={"ghost_waters": 6})

    def heavy(system, positions, ligand):
        return [positions[i] for i in ligand if system.getParticleMass(i).value_in_unit(mm.unit.dalton) > 2.0]

    both = heavy(system_a, pos_a, ligand_a) + heavy(system_b, pos_b, ligand_b)
    forward = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    backward = make_hybrid_system(system_b, pos_b, ligand_b, system_a, pos_a, ligand_a, {j: i for i, j in core.items()})
    radii = []
    for hybrid, n in ((forward, forward.system.getNumParticles()), (backward, backward.system.getNumParticles())):
        top = hybrid_edge_topology(n, list(hybrid.ligand_atoms), hybrid.system, box)
        radii.append((hybrid_gcmc(rcfg, hybrid, top, box).radius_nm,
                      hybrid_gcmc(rcfg, hybrid, top, box, both).radius_nm))
    assert abs(radii[0][0] - radii[1][0]) > 0.005                           # from the hybrid: not one sphere
    assert radii[0][1] == pytest.approx(radii[1][1], abs=1e-12)


def test_a_pre_equilibration_of_other_ghost_waters_is_not_used_again(tmp_path, monkeypatch):
    """Second review of PR #19 (P2): with `ghost_waters: auto` a radius of 1.05 nm instead of 1.00 nm gives 160 ghosts
    instead of 139; equil.npz recorded `auto`, and a leg resumed after its pre-equilibration was fixed in the cell
    equilibrated with the other waters. The ghosts are recorded before the pre-equilibration is made (equil_ghosts.json)
    and a pre-equilibration of other ghosts, or of an earlier version whose ghosts are not known, is refused before
    anything is written; one of the same ghosts is used again."""
    import json
    from types import SimpleNamespace

    from cufep import workflow
    from cufep.engine.leg import EQUIL_GHOSTS, pre_equilibrated

    made = []

    def fake_pre_equilibration(cfg, leg, prep, system, barostat, membrane_info, seed, equil_file, device, log):
        assert (equil_file.parent / EQUIL_GHOSTS).exists()          # recorded before
        made.append(1)
        np.savez(equil_file, positions=np.zeros((1, 3)))

    monkeypatch.setattr(workflow, "run_pre_equilibration", fake_pre_equilibration)
    monkeypatch.setattr(workflow, "check_equil_file", lambda *args, **kwargs: True)
    run = SimpleNamespace(leg_dir=tmp_path, job=SimpleNamespace(leg="complex"), cfg=None, seed=1, device0=None,
                          log=lambda *a: None, timed=lambda *a: None)
    moves = SimpleNamespace(ghosts=np.arange(139), waters=np.zeros((900, 3)))
    pre_equilibrated(run, {}, None, "isotropic", {}, moves)
    assert json.loads((tmp_path / EQUIL_GHOSTS).read_text())["ghosts"] == list(range(139)) and made == [1]
    pre_equilibrated(run, {}, None, "isotropic", {}, moves)
    assert made == [1]
    more = SimpleNamespace(ghosts=np.arange(160), waters=moves.waters)
    with pytest.raises(workflow.WorkflowError, match="made with 139 ghost waters and this run has 160"):
        pre_equilibrated(run, {}, None, "isotropic", {}, more)
    (tmp_path / EQUIL_GHOSTS).unlink()                               # a pre-equilibration of an earlier version
    with pytest.raises(workflow.WorkflowError, match="records no ghost waters"):
        pre_equilibrated(run, {}, None, "isotropic", {}, moves)
    pre_equilibrated(run, {}, None, "isotropic", {}, moves, earlier_ghosts=np.arange(139))
    assert made == [1] and not (tmp_path / EQUIL_GHOSTS).exists()


def test_the_sphere_of_a_relative_leg_takes_the_bonds_of_the_atoms_unique_to_b(edge):
    """Third review of PR #19 (P2): the topology of a hybrid holds no bond between an atom unique to B and the core
    (they are terms of the System), so the oxygen of phenol was a molecule of its own in the sphere of the moves (with
    the chain of the review, the core along x to 1.95 nm and an atom of B at 2.10 nm in a cell of 4 nm, the cell was
    refused). The moves take the bonds of the hybrid: its heavy atoms are one molecule, in the check of the cell and
    in the moves. (The test topology holds no bond of the ligand at all.)"""
    import networkx as nx

    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_gcmc

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    top = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system, box)
    moves = hybrid_gcmc(_rbfe_config(gcmc={"ghost_waters": 6}), hybrid, top, box)
    heavy = moves.ligand_heavy.tolist()
    assert set(hybrid.unique_new) & set(heavy)                              # the oxygen of phenol
    graph = nx.Graph([tuple(pair) for pair in moves.ligand_heavy_bonds.tolist()])
    graph.add_nodes_from(range(len(heavy)))
    assert nx.number_connected_components(graph) == 1
