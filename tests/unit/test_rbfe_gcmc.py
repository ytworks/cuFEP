"""Grand canonical moves of water in a leg of a relative calculation (cufep.sampling.gcmc on the hybrid System of
cufep.rbfe.hybrid), on the CPU with stock OpenMM: toluene (A) into phenol (B) in a box of water."""
from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from conftest import build_hybrid_edge, potential, reference_context

pytestmark = pytest.mark.needs("openff.toolkit", "rdkit", "openmmforcefields")


@pytest.fixture(scope="module")
def edge():
    return build_hybrid_edge(with_topologies=True)


def _hybrid(edge, rest2=False):
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import hybrid_topology

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box, (top_a, top_b) = edge
    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, rest2=rest2,
                                rest2_atoms=[])
    unique_b = sorted(j for j in ligand_b if j not in set(core.values()))
    return hybrid, hybrid_topology(top_a, top_b, unique_b)


def _spec(edge, hybrid, topology):
    from cufep.sampling.gcmc import gcmc_spec, hybrid_parameters

    *_, box, _tops = edge
    spec = gcmc_spec(hybrid.system, topology, hybrid.positions, np.asarray(box), list(hybrid.ligand_atoms), 0.9, 0.5,
                     1, 298.15, 4.0, 200, 100, 0.3, -6.09, 0.030345, 4)
    spec.hybrid = hybrid_parameters(hybrid, spec.ligand)
    return spec


def _set(ctx, x, scale=1.0):
    from cufep.rbfe.states import HybridState

    for name, value in HybridState(x, scale).as_parameters().items():
        if name in ctx.getParameters():
            ctx.setParameter(name, value)


def _without_water(edge, water: list[int]):
    """The hybrid built from end states in which the atoms of ``water`` have no charge and no epsilon: what a ghost
    is, built from the start."""
    from cufep.rbfe.hybrid import make_hybrid_system

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box, _tops = edge
    out = []
    # the environment is the same in both Systems, after ligand A in System A and before ligand B in System B
    for system, shift in ((system_a, 0), (system_b, -len(ligand_a))):
        copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
        nb = next(f for f in copy.getForces() if isinstance(f, mm.NonbondedForce))
        for i in water:
            q, s, e = nb.getParticleParameters(i + shift)
            nb.setParticleParameters(i + shift, q * 0.0, s, e * 0.0)
        out.append(copy)
    return make_hybrid_system(out[0], pos_a, ligand_a, out[1], pos_b, ligand_b, core, rest2=False, rest2_atoms=[])


def test_a_ghost_water_has_no_interaction_in_any_force_of_the_hybrid(edge):
    """ghosts_off takes a water out of the NonbondedForce and out of the Lennard-Jones of the hybrid ligand with its
    environment: the hybrid then has, at every lambda, the energy of a hybrid built from end states without that
    water's interactions. Without the second force the ghost would still feel the ligand's Lennard-Jones."""
    from cufep.sampling.gcmc import ghosts_off

    *_, box, _tops = edge
    hybrid, topology = _hybrid(edge)
    spec = _spec(edge, hybrid, topology)
    # the water nearest the ligand, which the ligand's Lennard-Jones reaches
    x = hybrid.positions
    centre = x[list(spec.ligand_heavy)].mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(x[spec.waters[:, 0]] - centre, axis=1)))
    atoms = [int(a) for a in spec.waters[nearest]]
    ghosts_off(hybrid.system, atoms)
    expected = _without_water(edge, atoms)
    ctx, ref = reference_context(hybrid.system, x, box), reference_context(expected.system, x, box)
    for lam in (0.0, 0.3, 0.7, 1.0):
        _set(ctx, lam)
        _set(ref, lam)
        assert potential(ctx) == pytest.approx(potential(ref), abs=1e-6 * abs(potential(ref)) + 1e-4)


@pytest.mark.parametrize("rest2", [False, True])
def test_switching_a_water_in_a_live_context_gives_the_system_built_so(edge, rest2):
    """WaterSwitch on a Context of the hybrid: a water switched off gives the energy of the System with that water
    a ghost (ghosts_off), at every lambda and REST2 scale; switched on again, the energy it had."""
    from cufep.sampling.gcmc import WaterSwitch, ghosts_off

    *_, box, _tops = edge
    hybrid, topology = _hybrid(edge, rest2=rest2)
    spec = _spec(edge, hybrid, topology)
    spec.ghosts = np.zeros(0, dtype=np.int64)                         # every water interacts in this Context
    x = hybrid.positions
    centre = x[list(spec.ligand_heavy)].mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(x[spec.waters[:, 0]] - centre, axis=1)))
    ctx = reference_context(hybrid.system, x, box)
    switch = WaterSwitch(ctx, spec)
    off = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(hybrid.system))
    ghosts_off(off, spec.waters[nearest])
    ref = reference_context(off, x, box)
    for lam, scale in ((0.3, 1.5 if rest2 else 1.0), (0.8, 1.0)):
        _set(ctx, lam, scale)
        _set(ref, lam, scale)
        before = potential(ctx)
        on = np.ones(len(spec.waters), dtype=bool)
        on[nearest] = False
        switch.apply(on)
        assert potential(ctx) == pytest.approx(potential(ref), abs=1e-6 * abs(potential(ref)) + 1e-4)
        switch.apply(np.ones(len(spec.waters), dtype=bool))
        assert potential(ctx) == pytest.approx(before, abs=1e-6 * abs(before) + 1e-4)


def test_the_stand_in_gives_each_atom_of_the_hybrid_ligand_its_state(edge):
    """At lambda 0 the atoms of the hybrid ligand have the charges and Lennard-Jones of A (those unique to B none),
    at 1 those of B (those unique to A none), in the middle the mapped ones half way and the unique ones of A without
    charge (they lose it in the first half) and those of B without (they gain it in the second)."""
    from cufep.sampling.gcmc import hybrid_ligand_parameters

    hybrid, topology = _hybrid(edge)
    spec = _spec(edge, hybrid, topology)
    h = spec.hybrid
    kind = h["kind"]
    assert set(kind) == {0, 1, 2}
    for x, end in ((0.0, "a"), (1.0, "b")):
        charge, _sigma, epsilon, soft, scale = hybrid_ligand_parameters(spec, x)
        other = 2 if end == "a" else 1
        here = kind != other
        assert np.allclose(charge, h[f"charge_{end}"]) and np.allclose(epsilon[here], h[f"epsilon_{end}"][here])
        assert np.all(scale[kind == other] == 0.0) and np.all(scale[kind != other] == 1.0)
        assert np.all(soft[kind == 0] == 0.0)
    charge, _sigma, _epsilon, _soft, scale = hybrid_ligand_parameters(spec, 0.5)
    assert np.allclose(charge[kind == 0], 0.5 * (h["charge_a"] + h["charge_b"])[kind == 0])
    assert np.all(charge[kind != 0] == 0.0) and np.all(scale[kind != 0] == 1.0)


def test_moves_are_made_on_a_hybrid_context_and_keep_its_energy_that_of_its_waters(edge):
    """A move of the mover on a Context of the hybrid (the NumPy stand-in of the hybrid ligand's state): the counts
    of a move, and after it the Context has the energy of the System with the waters that the mover says interact."""
    from cufep.rbfe.states import HybridState
    from cufep.sampling.gcmc import GcmcMover, ghosts_off

    *_, box, _tops = edge
    hybrid, topology = _hybrid(edge)
    spec = _spec(edge, hybrid, topology)
    ghosts_off(hybrid.system, spec.waters[spec.ghosts].reshape(-1))
    ctx = reference_context(hybrid.system, hybrid.positions, box)
    state = HybridState(0.4)
    _set(ctx, 0.4)
    mover = GcmcMover(spec, ctx, seed=3)
    totals = np.zeros(4, dtype=int)
    for _ in range(5):
        counts = mover.attempt(state)
        totals += [counts["insertions_attempted"], counts["insertions_accepted"], counts["deletions_attempted"],
                   counts["deletions_accepted"]]
    assert totals[0] + totals[2] == 5 * spec.attempts
    assert totals[1] + totals[3] >= 1                                  # the moves do something in this box
    rebuilt = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(hybrid.system))
    from cufep.sampling.gcmc import every_water_on

    every_water_on(rebuilt, spec)
    ghosts_off(rebuilt, spec.waters[~mover.on].reshape(-1))
    import openmm.unit as u

    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    ref = reference_context(rebuilt, x, box)
    _set(ref, 0.4)
    assert potential(ctx) == pytest.approx(potential(ref), abs=1e-6 * abs(potential(ref)) + 1e-4)
