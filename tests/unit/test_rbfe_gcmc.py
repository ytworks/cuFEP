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


def test_the_sphere_of_an_edge_is_the_same_both_ways(edge, tmp_path):
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
    # eleventh review of PR #19: the ghosts of the start were the waters farthest from the centre of the hybrid, which
    # holds the mapped atoms where A has them: with B 0.3 nm along x from A the two directions chose other waters. They
    # are counted from the mapped atoms midway between the two end states: the same waters both ways
    from cufep.rbfe.leg import centre_positions_of

    pos_b = np.array(edge[4], dtype=float)
    pos_b[ligand_b] += [0.3, 0.0, 0.0]
    a_end = {"positions": pos_a, "ligand": ligand_a, "box": box, "system": system_a}
    b_end = {"positions": pos_b, "ligand": ligand_b, "box": box, "system": system_b}
    chosen = {}
    for first, second, mapping in ((a_end, b_end, core), (b_end, a_end, {j: i for i, j in core.items()})):
        hybrid = make_hybrid_system(first["system"], first["positions"], first["ligand"], second["system"],
                                    second["positions"], second["ligand"], mapping)
        top = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system, box)
        both = (heavy(first["system"], first["positions"], first["ligand"])
                + heavy(second["system"], second["positions"], second["ligand"]))
        for rule, centre in (("hybrid", None), ("midway", centre_positions_of(hybrid, first, second))):
            moves = hybrid_gcmc(rcfg, hybrid, top, box, both, centre_positions=centre)
            oxygens = np.asarray(hybrid.positions)[moves.waters[moves.ghosts, 0]]
            chosen.setdefault(rule, []).append(sorted(map(tuple, np.round(oxygens, 6))))
    assert chosen["hybrid"][0] != chosen["hybrid"][1]
    assert chosen["midway"][0] == chosen["midway"][1]
    # a leg whose pre-equilibration or samples an earlier version made keeps their ghosts, those of the centre of the
    # hybrid: where its record names no centre and has them, or without a record where it goes on from its samples
    import json

    from cufep.rbfe.leg import ghosts_of_the_leg

    hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)
    top = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system, box)
    both = heavy(system_a, pos_a, ligand_a) + heavy(system_b, pos_b, ligand_b)
    midway_at = centre_positions_of(hybrid, a_end, b_end)
    midway = hybrid_gcmc(rcfg, hybrid, top, box, both, centre_positions=midway_at)
    earlier = hybrid_gcmc(rcfg, hybrid, top, box, both)
    assert midway.start_centre == "midway" and list(midway.ghosts) != list(earlier.ghosts)
    leg_dir = tmp_path / "leg"
    leg_dir.mkdir()
    (leg_dir / "equil.npz").write_bytes(b"made by an earlier version")
    for record, kept in (({"ghosts": earlier.ghosts.tolist()}, earlier),
                         ({"ghosts": earlier.ghosts.tolist(), "start_centre": "midway"}, midway),
                         ({"ghosts": midway.ghosts.tolist()}, midway), (None, midway)):
        (leg_dir / "equil_ghosts.json").unlink(missing_ok=True)
        if record is not None:
            (leg_dir / "equil_ghosts.json").write_text(json.dumps(record))
        moves, _centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, False, midway=midway_at)
        assert list(moves.ghosts) == list(kept.ghosts)
    (leg_dir / "hrex").mkdir()
    (leg_dir / "hrex" / "checkpoint.npz").write_bytes(b"")              # goes on from its samples
    going_on, _centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, False, midway=midway_at)
    assert list(going_on.ghosts) == list(earlier.ghosts)
    anew, _centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, True, midway=midway_at)
    assert list(anew.ghosts) == list(midway.ghosts)


def test_a_leg_keeps_the_ghosts_of_its_samples_and_one_configuration_has_one_sphere_both_ways(edge, tmp_path,
                                                                                               monkeypatch):
    """Twelfth review of PR #19, on the edge with B 0.3 nm along x from A (the centres of the start differ):
    (P2) a leg that goes on from its samples chose by the record of its ghosts alone: a record that was lost chose the
    ghosts of the earlier centre, one that could not be read those of the new, and the System of the samples was
    another. The ghosts are those whose System has the checksum the samples recorded;
    (P2) the ghosts of an earlier version were checked around the new centre before they were taken up, and a leg with
    a valid record was refused: they are checked around the centre they were counted from;
    (P2, the moves) the sphere of a move is around the heavy atoms of the hybrid ligand in the configuration of that
    move, the same atoms both ways: one configuration has the same waters in its sphere in A -> B and B -> A (the
    hybrids start from other configurations, the mapped atoms of A or of B)."""
    import hashlib
    import json

    from cufep.rbfe import leg as leg_module
    from cufep.rbfe.hybrid import make_hybrid_system
    from cufep.rbfe.leg import centre_positions_of, ghosts_of_the_leg, hybrid_gcmc, start_moves
    from cufep.sampling.gcmc import GcmcMover, ghosts_off, start_ghosts_in_sphere
    from cufep.workflow import WorkflowError

    system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core, box = edge
    pos_b = np.array(pos_b, dtype=float)
    pos_b[ligand_b] += [0.3, 0.0, 0.0]
    a_end = {"positions": pos_a, "ligand": ligand_a, "box": box}
    b_end = {"positions": pos_b, "ligand": ligand_b, "box": box}

    def heavy(system, positions, ligand):
        return [positions[i] for i in ligand if system.getParticleMass(i).value_in_unit(mm.unit.dalton) > 2.0]

    def hybrid_of():
        return make_hybrid_system(system_a, pos_a, ligand_a, system_b, pos_b, ligand_b, core)

    hybrid = hybrid_of()
    top = hybrid_edge_topology(hybrid.system.getNumParticles(), list(hybrid.ligand_atoms), hybrid.system, box)
    both = (heavy(system_a, pos_a, ligand_a) + heavy(system_b, pos_b, ligand_b), None)
    rcfg = _rbfe_config(gcmc={"ghost_waters": 6})
    midway_positions = centre_positions_of(hybrid, a_end, b_end)
    midway = hybrid_gcmc(rcfg, hybrid, top, box, *both, centre_positions=midway_positions)
    earlier = hybrid_gcmc(rcfg, hybrid, top, box, *both)
    assert list(midway.ghosts) != list(earlier.ghosts)

    def digest(ghosts) -> str:
        """The checksum of the System of the leg with these ghosts, as the leg makes it (build_leg)."""
        made = hybrid_of()
        ghosts_off(made.system, midway.waters[np.asarray(ghosts)].reshape(-1))
        return hashlib.sha256(mm.XmlSerializer.serialize(made.system).encode()).hexdigest()

    leg_dir = tmp_path / "leg"
    (leg_dir / "hrex").mkdir(parents=True)
    (leg_dir / "hrex" / "checkpoint.npz").write_bytes(b"")             # goes on from its samples
    record = leg_dir / "equil_ghosts.json"
    for text, made_with, kept in ((None, midway, midway), ("", earlier, earlier), ("{}", midway, midway),
                                  (json.dumps({"ghosts": midway.ghosts.tolist(), "start_centre": "midway"}), earlier,
                                   earlier),
                                  (None, None, earlier)):                # samples of neither: the earlier rule
        record.unlink(missing_ok=True)
        if text is not None:
            record.write_text(text)
        stored = digest(made_with.ghosts) if made_with is not None else "0" * 64
        monkeypatch.setattr(leg_module, "_samples_system_digest", lambda leg_dir, stored=stored: stored)
        moves, _centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, False, hybrid.system,
                                            midway=midway_positions)
        assert list(moves.ghosts) == list(kept.ghosts)
    # (thirteenth review, P2) a record of numbers of no water ([426] of 426 waters, 2**63) or of a number twice is not a
    # record: the checksum of the samples still finds the ghosts; [1, 1] is not taken for [1], and the same set in
    # another order is that set
    n = len(midway.waters)
    stored = digest(earlier.ghosts)
    monkeypatch.setattr(leg_module, "_samples_system_digest", lambda leg_dir: stored)
    for text in (json.dumps({"ghosts": [n]}), json.dumps({"ghosts": [2 ** 63]}),
                 json.dumps({"ghosts": [int(earlier.ghosts[0])] * 2})):
        record.write_text(text)
        moves, _centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, False, hybrid.system,
                                            midway=midway_positions)
        assert list(moves.ghosts) == list(earlier.ghosts)
    record.write_text(json.dumps({"ghosts": [int(i) for i in earlier.ghosts[::-1]]}))
    moves, centres = ghosts_of_the_leg(midway, hybrid.positions, box, leg_dir, False, hybrid.system,
                                            midway=midway_positions)
    assert list(moves.ghosts) == list(earlier.ghosts) and centres == ("ligand",)
    monkeypatch.undo()
    # (thirteenth review, P2) the centres the ghosts are checked around: the one the record names, else every centre
    # whose rule gives them (one ghost: both rules choose the same water), and the start passes around one of them
    one = _rbfe_config(gcmc={"ghost_waters": 1})
    single = hybrid_gcmc(one, hybrid, top, box, *both, centre_positions=midway_positions)
    assert list(single.ghosts) == list(hybrid_gcmc(one, hybrid, top, box, *both).ghosts)
    leg_one = tmp_path / "one"
    leg_one.mkdir()
    (leg_one / "equil.npz").write_bytes(b"made by an earlier version")
    (leg_one / "equil_ghosts.json").write_text(json.dumps({"ghosts": single.ghosts.tolist()}))
    centres = ghosts_of_the_leg(single, hybrid.positions, box, leg_one, False, midway=midway_positions)[1]
    assert centres == ("midway", "ligand")
    (leg_one / "equil_ghosts.json").write_text(json.dumps({"ghosts": single.ghosts.tolist(), "start_centre": "ligand"}))
    assert ghosts_of_the_leg(single, hybrid.positions, box, leg_one, False, midway=midway_positions)[1] == (
        "ligand",)
    import cufep.sampling.gcmc as gcmc_module

    (leg_one / "equil_ghosts.json").write_text(json.dumps({"ghosts": single.ghosts.tolist()}))
    with monkeypatch.context() as patch:                  # a ghost in the sphere of the new centre alone
        patch.setattr(gcmc_module, "start_ghosts_in_sphere",
                      lambda spec, x, box, ghosts, centre, heavy=None: centre is midway_positions)
        assert list(start_moves(one, hybrid, top, box, both, midway_positions, leg_one, False).ghosts) == list(
            single.ghosts)
        (leg_one / "equil_ghosts.json").write_text(json.dumps({"ghosts": single.ghosts.tolist(),
                                                              "start_centre": "midway"}))
        with pytest.raises(WorkflowError, match="would take waters of the sphere itself"):
            start_moves(one, hybrid, top, box, both, midway_positions, leg_one, False)
        # (fourteenth review, P2) ghosts chosen anew (the number of ghosts changed: the record of one ghost is not taken
        # up) are checked around the centre of this version alone, though both rules choose them
        (leg_one / "equil_ghosts.json").write_text(json.dumps({"ghosts": single.ghosts.tolist()}))
        two = _rbfe_config(gcmc={"ghost_waters": 2})
        anew = hybrid_gcmc(two, hybrid, top, box, *both, centre_positions=midway_positions, check_start=False)
        assert ghosts_of_the_leg(anew, hybrid.positions, box, leg_one, False, midway=midway_positions)[1] == (
            "midway",)
        with pytest.raises(WorkflowError, match="would take waters of the sphere itself"):
            start_moves(two, hybrid, top, box, both, midway_positions, leg_one, False)
    # the check of the start around the centre the ghosts were counted from, with B 0.3 nm along y from A: as many
    # ghosts as waters outside the sphere around the centre of the hybrid (403 of 426); around the new centre fewer are
    # outside (395), and a new leg has ghosts inside
    from cufep.alchemy.lrc import nearest_image
    from cufep.sampling.gcmc import _centre

    along_y = np.array(edge[4], dtype=float)
    along_y[ligand_b] += [0.0, 0.3, 0.0]
    y_hybrid = make_hybrid_system(system_a, pos_a, ligand_a, system_b, along_y, ligand_b, core)
    y_top = hybrid_edge_topology(y_hybrid.system.getNumParticles(), list(y_hybrid.ligand_atoms), y_hybrid.system, box)
    y_both = (heavy(system_a, pos_a, ligand_a) + heavy(system_b, along_y, ligand_b), None)
    y_midway = centre_positions_of(y_hybrid, a_end, {"positions": along_y, "ligand": ligand_b, "box": box})
    spec = hybrid_gcmc(rcfg, y_hybrid, y_top, box, *y_both)
    oxygens = np.asarray(y_hybrid.positions)[spec.waters[:, 0]]

    def outside(positions) -> int:
        centre = _centre(np.asarray(positions, dtype=float), box, spec.ligand_heavy, spec.ligand_heavy_bonds)
        return int((np.linalg.norm(nearest_image(oxygens - centre, box), axis=1) >= spec.radius_nm).sum())

    n = outside(y_hybrid.positions)
    assert outside(y_midway) < n
    many = _rbfe_config(gcmc={"ghost_waters": n})
    with pytest.raises(WorkflowError, match="would take waters of the sphere itself"):
        start_moves(many, y_hybrid, y_top, box, y_both, y_midway)          # a new leg
    of_earlier = hybrid_gcmc(many, y_hybrid, y_top, box, *y_both)           # the ghosts an earlier version made
    assert not start_ghosts_in_sphere(of_earlier, y_hybrid.positions, box, of_earlier.ghosts)
    old = tmp_path / "old"
    old.mkdir()
    (old / "equil.npz").write_bytes(b"made by an earlier version")
    (old / "equil_ghosts.json").write_text(json.dumps({"ghosts": of_earlier.ghosts.tolist()}))
    kept = start_moves(many, y_hybrid, y_top, box, y_both, y_midway, old, False)
    assert list(kept.ghosts) == list(of_earlier.ghosts) and kept.start_centre == "ligand"
    # one configuration, both ways: the waters of the sphere of a move
    backward = make_hybrid_system(system_b, pos_b, ligand_b, system_a, pos_a, ligand_a, {j: i for i, j in core.items()})
    top_b = hybrid_edge_topology(backward.system.getNumParticles(), list(backward.ligand_atoms), backward.system, box)
    moves_b = hybrid_gcmc(rcfg, backward, top_b, box, *both)
    x_f = np.asarray(hybrid.positions, dtype=float)
    x_b = np.array(backward.positions, dtype=float)
    for i, h in hybrid.a_to_hybrid.items():
        x_b[backward.b_to_hybrid[i]] = x_f[h]
    for j, h in hybrid.b_to_hybrid.items():
        x_b[backward.a_to_hybrid[j]] = x_f[h]
    forward_mover, backward_mover = GcmcMover(midway, None, 1), GcmcMover(moves_b, None, 1)
    on = np.ones(len(midway.waters), dtype=bool)
    in_forward = forward_mover.in_sphere(x_f, box, on)
    assert len(in_forward) and list(in_forward) == list(backward_mover.in_sphere(x_b, box, on))
    assert list(in_forward) != list(backward_mover.in_sphere(np.asarray(backward.positions), box, on))


def test_a_pre_equilibration_of_other_ghost_waters_is_not_used_again(tmp_path, monkeypatch):
    """Second review of PR #19 (P2): with `ghost_waters: auto` a radius of 1.05 nm instead of 1.00 nm gives 160 ghosts
    instead of 139; equil.npz recorded `auto`, and a leg resumed after its pre-equilibration was fixed in the cell
    equilibrated with the other waters. The ghosts are recorded before the pre-equilibration is made (equil_ghosts.json)
    and a pre-equilibration of other ghosts, or of an earlier version whose ghosts are not known, is refused
    (check_equil_ghosts: the leg calls it before anything is set aside, the run for all its legs before it writes
    anything); one of the same ghosts is used again."""
    import json
    from types import SimpleNamespace

    from cufep import workflow
    from cufep.engine.leg import EQUIL_GHOSTS, check_equil_ghosts, pre_equilibrated

    made = []

    def fake_pre_equilibration(cfg, leg, prep, system, barostat, membrane_info, seed, equil_file, device, log):
        assert (equil_file.parent / EQUIL_GHOSTS).exists()          # recorded before
        made.append(1)
        np.savez(equil_file, positions=np.zeros((1, 3)))

    monkeypatch.setattr(workflow, "run_pre_equilibration", fake_pre_equilibration)
    monkeypatch.setattr(workflow, "check_equil_file", lambda *args, **kwargs: True)
    run = SimpleNamespace(leg_dir=tmp_path, job=SimpleNamespace(leg="complex"), cfg=None, seed=1, device0=None,
                          log=lambda *a: None, timed=lambda *a: None)
    moves = SimpleNamespace(ghosts=np.arange(139), waters=np.zeros((900, 3)), start_centre="ligand")
    pre_equilibrated(run, {}, None, "isotropic", {}, moves)
    assert json.loads((tmp_path / EQUIL_GHOSTS).read_text())["ghosts"] == list(range(139)) and made == [1]
    pre_equilibrated(run, {}, None, "isotropic", {}, moves)
    assert made == [1]
    equil = tmp_path / "equil.npz"
    check_equil_ghosts(equil, moves)
    more = SimpleNamespace(ghosts=np.arange(160), waters=moves.waters)
    with pytest.raises(workflow.WorkflowError, match="made with 139 ghost waters and this run has 160"):
        check_equil_ghosts(equil, more)
    (tmp_path / EQUIL_GHOSTS).unlink()                               # a pre-equilibration of an earlier version
    with pytest.raises(workflow.WorkflowError, match="records no ghost waters"):
        check_equil_ghosts(equil, moves)
    check_equil_ghosts(equil, moves, earlier_ghosts=np.arange(139))
    pre_equilibrated(run, {}, None, "isotropic", {}, moves)
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
