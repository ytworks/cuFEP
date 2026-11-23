"""A complex leg whose alchemical run is made in water while its restraint is chosen in the bilayer
(``membrane.complex_leg: water``): two prepared systems of one pose, the restraint of the one used
in the other."""
import json
from pathlib import Path

import numpy as np
import pytest
import test_interactions
from test_interactions import BOX, trajectories_by_hand

#: TYK2 with ejm_31 as prepared (the fixture of test_interactions, used by the tests of this module too)
tyk2 = test_interactions.tyk2

STAGE = {"name": "complex_bilayer", "barostat": "membrane", "membrane_info": {}}


def turned(positions: np.ndarray) -> np.ndarray:
    """The same configuration turned about its centre and moved, as the build of a bilayer lays the complex
    into its plane."""
    a, b = 0.7, -0.4
    rot = (np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
           @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]]))
    centre = positions.mean(axis=0)
    return (positions - centre) @ rot.T + centre + np.array([0.3, -0.2, 0.1])


def with_a_water_before_the_ligand(topology, positions, ligand):
    """The same complex with one water molecule more before the ligand: (topology, positions in nm, ligand
    atoms). Two builds of one pose differ like this when only one of them removes a crystal water."""
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.solvate import ligand_atom_indices

    of_ligand = set(ligand)
    out = app.Modeller(topology, positions * u.nanometer)
    out.delete([a for a in out.topology.atoms() if a.index in of_ligand])
    only_ligand = app.Modeller(topology, positions * u.nanometer)
    only_ligand.delete([a for a in only_ligand.topology.atoms() if a.index not in of_ligand])
    water = app.Topology()
    residue = water.addResidue("HOH", water.addChain())
    oxygen = water.addAtom("O", app.Element.getBySymbol("O"), residue)
    for name in ("H1", "H2"):
        water.addBond(oxygen, water.addAtom(name, app.Element.getBySymbol("H"), residue))
    corner = positions.min(axis=0) - 1.0
    out.add(water, (corner + np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])) * u.nanometer)
    out.add(only_ligand.topology, only_ligand.positions)
    return (out.topology, np.asarray(out.positions.value_in_unit(u.nanometer), dtype=float),
            list(ligand_atom_indices(out.topology)))


def prepared(topology, positions, ligand) -> dict:
    """A prepared complex as :func:`cufep.workflow.load_prepared` gives it, without a System."""
    from cufep.workflow import _solute_of

    return dict(_solute_of(topology, ligand), positions=positions, box=BOX, input_chains=None,
                system_xml="<not read: the MD of these tests is made by hand/>")


def two_systems(tyk2):
    """TYK2 with ejm_31 as the complex of the alchemical run, and the same pose turned, with a water before
    the ligand, as the complex in the bilayer: (the first, the second)."""
    topology, pos, ligand, _rdmol, _protein = tyk2
    top2, pos2, ligand2 = with_a_water_before_the_ligand(topology, pos, ligand)
    return prepared(topology, pos, ligand), prepared(top2, turned(pos2), ligand2)


def test_the_complex_leg_in_water_needs_a_bilayer_for_a_restraint_from_interactions():
    """``membrane.complex_leg: water`` builds the bilayer for the MD from which the restraint is chosen:
    with a selection that has no such MD the configuration is refused when it is read. The box of
    water that ``auto`` gives the complex of the alchemical run is then the one of a complex in
    water; a bilayer in the alchemical run keeps its rectangular box."""
    from pydantic import ValidationError

    from cufep.config.auto import resolve
    from cufep.config.schema import Config

    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "backend": {"name": "openmm", "platform": "CPU"}}
    in_water = {"enabled": True, "complex_leg": "water"}
    for selection in ("auto", "manual"):
        restraints = {"selection": selection, "host_atoms": [0, 1, 2], "guest_atoms": [3, 4, 5]}
        with pytest.raises(ValidationError,
                           match="membrane.complex_leg is 'water'.*restraints.selection: interactions"):
            Config.model_validate({**base, "membrane": in_water, "restraints": restraints})
        Config.model_validate({**base, "membrane": {"enabled": True}, "restraints": restraints})
        # without a bilayer the setting says nothing, as the other settings of the membrane
        Config.model_validate({**base, "membrane": {"complex_leg": "water"}, "restraints": restraints})
    shapes = {}
    for name, membrane in (("water", in_water), ("bilayer", {"enabled": True}), ("none", {})):
        resolved, report = resolve(Config.model_validate({**base, "membrane": membrane}))
        shapes[name] = (resolved.solvation.box_shape,
                        next(reason for key, _value, reason in report.decisions if key == "solvation.box_shape"))
    assert [shape for shape, _reason in shapes.values()] == ["dodecahedron", "cube", "dodecahedron"]
    assert "membrane.complex_leg: water" in shapes["water"][1] and "complex_leg" not in shapes["none"][1]


def test_the_two_preparations_are_told_apart_by_their_record(tmp_path):
    """The record of a preparation says that the complex of the alchemical run is in water; one made
    with the default says nothing of the setting, so every earlier preparation is still used. A
    directory of the one is refused by a run of the other before anything is written."""
    from cufep.config.schema import Config
    from cufep.prep.ligand import load_ligands
    from cufep.workflow import (
        MEMBRANE_SETTINGS_ADDED,
        SETTINGS_ADDED,
        WorkflowError,
        prep_contradictions,
        prep_made_from,
        prepare_ligand,
        settings_differences,
    )

    repo = Path(__file__).resolve().parents[2]
    ligand = load_ligands(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf")[0]

    def config(**membrane):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                      "output": {"dir": str(tmp_path / "out")}, "membrane": membrane})

    bilayer, water = config(enabled=True), config(enabled=True, complex_leg="water")
    of_bilayer = prep_made_from(bilayer, ligand, True, "cube")
    assert "complex_leg" not in of_bilayer["membrane"] and SETTINGS_ADDED["membrane"] is MEMBRANE_SETTINGS_ADDED
    assert settings_differences(of_bilayer, prep_made_from(water, ligand, True, "cube")) == [
        "membrane.complex_leg: not recorded -> water"]
    # a file without a record of its origin holds the bilayer complex alone
    older = {"ligand": ligand.name, "membrane": True, "complex_barostat": "membrane", "box_shape": "cube"}
    assert prep_contradictions(older, ligand.name, True, "cube") == []
    assert prep_contradictions(older, ligand.name, True, "cube", "water") == ["complex_barostat: membrane -> isotropic"]
    lig_dir = tmp_path / "out" / ligand.name
    (lig_dir / "prep").mkdir(parents=True)
    (lig_dir / "prep" / "prep.json").write_text(json.dumps({"ligand": ligand.name, "made_from": of_bilayer}))
    assert prepare_ligand(bilayer, ligand, None, lig_dir, True, "cube")["ligand"] == ligand.name
    with pytest.raises(WorkflowError, match="cannot be used again.*membrane.complex_leg: not recorded -> water"):
        prepare_ligand(water, ligand, None, lig_dir, True, "cube")
    assert sorted(p.name for p in (lig_dir / "prep").iterdir()) == ["prep.json"]


def test_a_restraint_names_the_same_atoms_in_another_system_of_the_same_solute(tyk2):
    """Two prepared systems of one pose number the ligand differently when one of them lost a water
    that comes before it. The restraint chosen in the one is the same restraint in the other: the
    energy of its force is the same for the solute turned and moved, and the restraint with the
    numbers of the first system is another one there. Systems that are not of the same protein and
    ligand, atom by atom, are refused."""
    import openmm as mm
    from conftest import potential, reference_context

    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.workflow import WorkflowError, restraint_in_system, solute_correspondence

    topology, pos, ligand, rdmol, _protein = tyk2
    here, there = two_systems(tyk2)
    atom_map = solute_correspondence(here, there)
    assert all(atom_map[i] == i for i in here["protein"]) and len(atom_map) == len(pos)
    assert [atom_map[i] for i in ligand] == [i + 3 for i in ligand] == there["ligand"]

    atoms = list(topology.atoms())
    residue = next(r for r in topology.residues() if r.name == "VAL" and r.id == "981")
    host = tuple(next(a.index for a in residue.atoms() if a.name == name) for name in ("N", "CA", "C"))
    heavy = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() > 1]
    first = next(i for i in heavy if sum(n.GetAtomicNum() > 1 for n in rdmol.GetAtomWithIdx(i).GetNeighbors()) > 1)
    second, third = [n.GetIdx() for n in rdmol.GetAtomWithIdx(first).GetNeighbors() if n.GetAtomicNum() > 1][:2]
    guest = (ligand[second], ligand[first], ligand[third])
    restraint = BoreschRestraint(host, guest, 0.6, 1.2, 1.5, 0.4, -1.1, 2.0)
    moved = restraint_in_system(restraint, atom_map)
    assert moved.host_atoms == host and moved.guest_atoms == tuple(i + 3 for i in guest)
    assert (moved.r0, moved.theta_a0, moved.k_r, moved.k_phi_c) == (restraint.r0, restraint.theta_a0, restraint.k_r,
                                                                   restraint.k_phi_c)

    def energy(of, positions):
        system = mm.System()
        for _ in positions:
            system.addParticle(12.0)
        system.addForce(of.create_force())
        return potential(reference_context(system, positions, BOX))

    in_the_first = energy(restraint, pos)
    assert in_the_first > 1.0 and atoms[guest[0]].element.symbol != "H"
    assert energy(moved, there["positions"]) == pytest.approx(in_the_first, rel=1e-6)
    assert abs(energy(restraint, there["positions"]) - in_the_first) > 1.0

    renamed = list(there["topology"].atoms())[there["ligand"][0]]
    name, renamed.name = renamed.name, "X9"
    with pytest.raises(WorkflowError, match="not of the same protein and ligand: atom .* is LIG.*nothing was written"):
        solute_correspondence(here, there)
    renamed.name = name
    with pytest.raises(WorkflowError, match=f"{len(pos)} atoms of protein and ligand in one and {len(pos) - 1}"):
        solute_correspondence(here, dict(there, solute=there["solute"][:-1]))
    # two chains of one sequence and numbering differ in their name alone: an atom of another chain is another atom
    chain = next(iter(there["topology"].chains()))
    kept, chain.id = chain.id, "Z"
    with pytest.raises(WorkflowError, match=f"of chain {kept} in one and .* of chain Z in the other"):
        solute_correspondence(here, there)
    chain.id = kept
    assert solute_correspondence(there, here) == {b: a for a, b in atom_map.items()}


def stage_inputs(tyk2, tmp_path, monkeypatch, name="ejm_31"):
    """What the restraint stage of a complex leg in water gets: the configuration, the job, the directory
    of the leg, the complex in water with the start of its alchemical run, the complex in the bilayer (read
    in the place of ``prep/complex_bilayer.*``), and the pre-equilibrations that were made (by hand)."""
    from cufep import workflow
    from cufep.config.schema import Config

    water, bilayer = two_systems(tyk2)
    cfg = Config.model_validate({"system": {"protein": "protein.pdb", "ligand": "ligand.sdf"},
                                 "membrane": {"enabled": True, "complex_leg": "water", "equilibration_ns": 0.5},
                                 "restraints": {"selection": "interactions", "interactions": dict(
                                     trajectories=2, length_ns=0.2, discard_ns=0.1, frame_interval_ps=10.0)}})
    leg_dir = tmp_path / name / "repeat_0" / "complex"
    leg_dir.mkdir(parents=True)
    job = workflow.LegJob(ligand=name, leg="complex", repeat=0, lig_dir=str(tmp_path / name), devices=[], config={},
                          production_ns=5.0, inline=True)
    # the alchemical run starts from another configuration than the MD in the bilayer: the ligand a little moved
    begin = water["positions"].copy()
    begin[water["ligand"]] += np.array([0.02, -0.01, 0.015])
    eq = {"positions": begin, "velocities": np.zeros_like(begin), "box": BOX, "timestep_fs": 4.0}
    made = []

    def pre_equilibration(cfg_, leg, prep, system, barostat, membrane_info, seed, equil_file, device, log=print,
                          what="pre-equilibration"):
        made.append((leg, barostat, system))
        np.savez(equil_file, positions=prep["positions"], velocities=np.zeros_like(prep["positions"]), box=BOX,
                 timestep_fs=4.0, simulated_ns=cfg_.membrane.equilibration_ns, simulated_ns_is_integrated=True,
                 notes="[]", made_with=json.dumps(workflow.equil_made_with(cfg_, leg, barostat == "membrane", seed)))

    monkeypatch.setattr(workflow, "run_pre_equilibration", pre_equilibration)
    monkeypatch.setattr(workflow, "load_prepared", lambda prep_dir, system: {"complex_bilayer": bilayer}[system])
    return cfg, job, leg_dir, water, eq, bilayer, made


def test_the_restraint_of_a_leg_in_water_is_chosen_in_the_bilayer_and_names_the_atoms_of_the_leg(
        tyk2, tmp_path, monkeypatch):
    """The complex in the bilayer is pre-equilibrated, its trajectories start from there and the
    restraint is chosen from them; the anchors are then the same atoms as the complex in water
    numbers them, and they were held to the rules at the configuration the alchemical run starts
    from. The record ties the restraint to both pre-equilibrations, and the analysis of the
    interactions stays the one of this restraint."""
    from cufep import workflow
    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.analysis.plots import _is_of_the_restraint
    from cufep.sampling.restraint_md import start_digest

    _topology, _pos, _ligand, rdmol, _protein = tyk2
    cfg, job, leg_dir, water, eq, bilayer, made_equil = stage_inputs(tyk2, tmp_path, monkeypatch)
    made = trajectories_by_hand(monkeypatch, bilayer["positions"])
    seed, lines = workflow.leg_seed(cfg, 0, "complex"), []
    restraint, info, _md, _equil = workflow.restraint_from_the_bilayer(cfg, job, leg_dir, water, eq, STAGE, rdmol,
                                                                       seed, None, lines.append)
    assert made_equil == [("complex", "membrane", None)] and [task.index for task in made] == [0, 1]
    assert all(task.settings.barostat == "membrane" and len(task.positions) == len(bilayer["positions"])
               for task in made)
    # the anchors: atoms of protein and ligand of the complex in water, the same atoms as were chosen
    assert workflow.restraint_anchor_problems(restraint.host_atoms, restraint.guest_atoms, water) == []
    chosen = info["md_system"]
    assert chosen["name"] == "complex_bilayer" and list(restraint.host_atoms) == chosen["host_atoms"]
    assert [i + 3 for i in restraint.guest_atoms] == chosen["guest_atoms"]
    assert workflow.restraint_choice(info)["md_system"] == "complex_bilayer"
    assert any("chosen from MD of the complex in the bilayer" in line for line in lines)
    # held to the rules at the start of the alchemical run, whose values the record gives
    probe = BoreschRestraint(tuple(restraint.host_atoms), tuple(restraint.guest_atoms), 0, 0, 0, 0, 0, 0)
    at_start = probe.values(eq["positions"])
    recorded = info["start_values"]
    assert recorded["r"] == pytest.approx(at_start[0], abs=1e-6)
    assert [recorded[key] for key in ("theta_A", "theta_B", "phi_A", "phi_B", "phi_C")] == pytest.approx(
        list(np.degrees(at_start[1:])), abs=1e-4)
    in_the_bilayer = BoreschRestraint(tuple(chosen["host_atoms"]), tuple(chosen["guest_atoms"]), 0, 0, 0, 0, 0, 0)
    assert not np.allclose(in_the_bilayer.values(bilayer["positions"]), at_start, atol=1e-4)
    # both pre-equilibrations are in the record, and the leg holds its restraint to both
    md_start = np.load(leg_dir / workflow.RESTRAINT_EQUIL)
    assert info["start"] == start_digest(md_start["positions"], md_start["velocities"], md_start["box"])
    assert info["alchemical_start"] == start_digest(eq["positions"], eq["velocities"], eq["box"])
    rfile = leg_dir / "restraint.json"
    stored = {"restraint": restraint.to_dict(), "info": info, "made_with": workflow.restraint_made_with(cfg)}
    rfile.write_text(json.dumps(stored))
    np.savez(leg_dir / "equil.npz", **eq)
    workflow.check_restraint_of_leg(stored, rfile, leg_dir)
    assert _is_of_the_restraint(leg_dir / "interactions.json", rfile)
    np.savez(leg_dir / "equil.npz", **dict(eq, positions=eq["positions"] + 0.001))
    with pytest.raises(workflow.WorkflowError, match="held to the rules at another configuration"):
        workflow.check_restraint_of_leg(stored, rfile, leg_dir)
    np.savez(leg_dir / "equil.npz", **eq)
    kept = (leg_dir / workflow.RESTRAINT_EQUIL).read_bytes()
    np.savez(leg_dir / workflow.RESTRAINT_EQUIL, **dict(md_start, positions=md_start["positions"] + 0.001))
    with pytest.raises(workflow.WorkflowError, match="chosen from MD that started from another configuration.*"
                                                     "restraint_equil.npz"):
        workflow.check_restraint_of_leg(stored, rfile, leg_dir)
    (leg_dir / workflow.RESTRAINT_EQUIL).unlink()
    with pytest.raises(workflow.WorkflowError, match="restraint_equil.npz is not there.*cannot be checked"):
        workflow.restraint_release_of(leg_dir, {"states": [{"lambda_restraints": 0.0}]}, 298.15)
    (leg_dir / workflow.RESTRAINT_EQUIL).write_bytes(kept)
    assert workflow.restraint_release_of(leg_dir, {"states": [{"lambda_restraints": 0.0}]}, 298.15)[0] != 0.0

    # the stage again: nothing is made a second time, and the restraint is the same
    again = workflow.restraint_from_the_bilayer(cfg, job, leg_dir, water, eq, STAGE, rdmol, seed, None,
                                                lambda line: None)
    assert len(made_equil) == 1 and len(made) == 2
    assert again[0].to_dict() == restraint.to_dict() and again[1] == info


def test_the_stage_in_the_bilayer_goes_on_from_every_file_it_has_written(tyk2, tmp_path, monkeypatch):
    """A leg that stops after the pre-equilibration in the bilayer, or after one of the trajectories,
    goes on with what it has and gets the restraint of a leg that was not stopped. Before any job,
    the trajectories it has are held to the pre-equilibration in the bilayer, not to the one of the
    alchemical run."""
    from cufep import workflow

    _topology, _pos, _ligand, rdmol, _protein = tyk2
    cfg, job, leg_dir, water, eq, bilayer, made_equil = stage_inputs(tyk2, tmp_path, monkeypatch)
    seed = workflow.leg_seed(cfg, 0, "complex")
    quiet = lambda line: None                                                                # noqa: E731
    made = trajectories_by_hand(monkeypatch, bilayer["positions"], stop_at=1)
    with pytest.raises(RuntimeError, match="the machine stopped"):
        workflow.restraint_from_the_bilayer(cfg, job, leg_dir, water, eq, STAGE, rdmol, seed, None, quiet)
    assert len(made_equil) == 1 and [task.index for task in made] == [0]
    assert not (leg_dir / "interactions.json").exists()

    # before any job of the next run: the trajectory that is there is one of this run
    np.savez(leg_dir / "equil.npz", **eq, simulated_ns=0.0, simulated_ns_is_integrated=True, notes="[]",
             made_with=json.dumps(workflow.equil_made_with(cfg, "complex", False, seed)))
    lig_dir = tmp_path / "ejm_31"
    prep_info = {"complex_barostat": "isotropic", "restraint_system": STAGE}
    workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)
    with pytest.raises(workflow.WorkflowError, match="MD 0 for the restraint of this leg .*barostat"):
        workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, {"complex_barostat": "isotropic"}, 5.0)
    kept = (leg_dir / workflow.RESTRAINT_EQUIL).read_bytes()
    (leg_dir / workflow.RESTRAINT_EQUIL).unlink()
    with pytest.raises(workflow.WorkflowError, match="restraint_equil.npz, which it started from, is not there"):
        workflow.check_leg_reuse(cfg, lig_dir, "complex", 0, prep_info, 5.0)
    (leg_dir / workflow.RESTRAINT_EQUIL).write_bytes(kept)
    # a bilayer that was equilibrated for another length than this run asks for is not the one of this run:
    # refused before any job and by the leg itself, which asks with the same function whether it has its
    # restraint or not (a review: a leg with a restraint whose prepared systems were made again used it unasked)
    data = cfg.model_dump(mode="json")
    data["membrane"]["equilibration_ns"] = 0.25
    shorter = type(cfg).model_validate(data)
    for check in (lambda: workflow.check_leg_reuse(shorter, lig_dir, "complex", 0, prep_info, 5.0),
                  lambda: workflow.check_restraint_equil(shorter, leg_dir, prep_info, seed)):
        with pytest.raises(workflow.WorkflowError, match="restraint_equil.npz cannot be used again.*"
                                                         "membrane_equilibration_ns: 0.5 -> 0.25"):
            check()
    workflow.check_restraint_equil(cfg, leg_dir, prep_info, seed)
    workflow.check_restraint_equil(shorter, leg_dir, {"complex_barostat": "isotropic"}, seed)   # no such system

    more = trajectories_by_hand(monkeypatch, bilayer["positions"])
    restraint, info, _md, _equil = workflow.restraint_from_the_bilayer(cfg, job, leg_dir, water, eq, STAGE, rdmol,
                                                                       seed, None, quiet)
    assert len(made_equil) == 1 and [task.index for task in more] == [1]

    # a leg that was not stopped, in a directory of its own
    cfg2, job2, leg_dir2, water2, eq2, bilayer2, _made = stage_inputs(tyk2, tmp_path, monkeypatch, name="unstopped")
    trajectories_by_hand(monkeypatch, bilayer2["positions"])
    whole, info2, _md, _equil = workflow.restraint_from_the_bilayer(cfg2, job2, leg_dir2, water2, eq2, STAGE, rdmol,
                                                                    seed, None, quiet)
    assert whole.to_dict() == restraint.to_dict() and info2 == info
    assert (leg_dir2 / "interactions.json").read_bytes() == (leg_dir / "interactions.json").read_bytes()


def test_the_result_of_a_leg_in_water_gives_the_bilayer_of_its_restraint_through_every_way_in(tmp_path):
    """The bilayer of such a leg is the one of the MD for its restraint: how it changed at the end of
    its equilibration is a number of the result, taken from that pre-equilibration, and what that
    pre-equilibration notes is said of it. The same from the leg and from an archive of it."""
    from test_sampling import synthetic_leg

    from cufep.workflow import RESTRAINT_EQUIL, archive_leg, leg_files, notes_of_leg_files

    leg = tmp_path / "lig" / "repeat_0" / "complex"
    synthetic_leg(leg)
    area = np.concatenate([np.full(200, 66.0), np.full(100, 65.0), np.full(100, 65.0 * 0.985)])
    np.savez(leg / "equil.npz", notes=json.dumps(["a note of the leg"]), pocket_waters=np.zeros(0),
             area_nm2=np.zeros(0), simulated_ns=0.0)
    assert notes_of_leg_files(leg)[1]["bilayer"] is None
    np.savez(leg / RESTRAINT_EQUIL, notes=json.dumps(["the time step was reduced to 2 fs"]), area_nm2=area,
             pocket_waters=np.full(400, 6.0), simulated_ns=10.0)
    notes, numbers = notes_of_leg_files(leg)
    assert notes == ["a note of the leg", "pre-equilibration of the complex in the bilayer, for the restraint: "
                                          "the time step was reduced to 2 fs"]
    assert numbers["bilayer"] == {"area_change": pytest.approx(-0.015), "pocket_water_change": 0.0}
    archived = archive_leg(leg)
    assert leg_files(archived)[RESTRAINT_EQUIL] == archived / RESTRAINT_EQUIL and (leg / RESTRAINT_EQUIL).exists()
    assert notes_of_leg_files(archived) == (notes, numbers)


def small_complex(tmp_path, out_dir):
    """An alanine dipeptide as the protein with one crystal water, and benzene next to it as the ligand, with
    its partial charges in the cache of the output directory: (prepared protein, ligand)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    from conftest import _DIPEPTIDE_BONDS, _DIPEPTIDE_HEAVY
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cufep.prep.ligand import load_ligands
    from cufep.prep.protein import PreparedProtein

    top, pos = app.Topology(), []
    chain, previous_c = top.addChain("A"), None
    for number, res_name in enumerate(("ACE", "ALA", "NME"), start=1):
        res, local = top.addResidue(res_name, chain, str(number)), {}
        for rn, name, element, xyz in _DIPEPTIDE_HEAVY:
            if rn == res_name:
                local[name] = top.addAtom(name, app.Element.getBySymbol(element), res)
                pos.append(mm.Vec3(*(0.1 * np.array(xyz) + 1.0)))
        for a, b in _DIPEPTIDE_BONDS[res_name]:
            top.addBond(local[a], local[b])
        if previous_c is not None:
            top.addBond(previous_c, local["N"])
        previous_c = local["C"] if res_name != "NME" else None
    top.addAtom("O", app.Element.getBySymbol("O"), top.addResidue("HOH", top.addChain("W"), "1"))
    pos.append(mm.Vec3(1.3, 0.4, 1.0))
    mod = app.Modeller(top, pos * u.nanometer)
    mod.addHydrogens(app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml"),
                     platform=mm.Platform.getPlatformByName("Reference"))
    protein = PreparedProtein(mod.topology, mod.positions, None, {"A": "A"})

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(benzene, randomSeed=7)
    conformer = benzene.GetConformer()
    xyz = np.array(conformer.GetPositions())
    for i, row in enumerate(xyz - xyz.mean(axis=0) + np.array([13.0, 10.0, 16.0])):          # in Angstrom
        conformer.SetAtomPosition(i, row.tolist())
    benzene.SetProp("_Name", "benzene")
    writer = Chem.SDWriter(str(tmp_path / "ligand.sdf"))
    writer.write(benzene)
    writer.close()
    ligand = load_ligands(tmp_path / "ligand.sdf")[0]
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache" / f"charges_am1bcc_{ligand.digest()}.json").write_text(
        json.dumps({"charges": [-0.13] * 6 + [0.13] * 6}))
    return protein, ligand


def test_a_complex_leg_in_water_is_prepared_as_two_systems_of_one_pose(tmp_path, monkeypatch):
    """The preparation writes the complex in water, on which the alchemical run is made, and the
    complex in the bilayer for the MD of the restraint, and its record says which is which. The build
    of the bilayer (here in its place: the complex turned, its crystal water removed, a rectangular
    box) numbers the ligand differently; the solute is the same atoms in both. A build that gives
    another solute stops the preparation before anything is written."""
    import openmm as mm
    import openmm.unit as u

    from cufep import workflow
    from cufep.config.schema import Config

    out = tmp_path / "out"
    protein, ligand = small_complex(tmp_path, out)
    cfg = Config.model_validate({"system": {"protein": str(tmp_path / "protein.pdb"),
                                            "ligand": str(tmp_path / "ligand.sdf")},
                                 "output": {"dir": str(out)}, "backend": {"name": "openmm"},
                                 "membrane": {"enabled": True, "complex_leg": "water"}})
    rename = []

    def bilayer_in_its_place(modeller, forcefield, cfg_, log=print):
        modeller.positions = turned(np.asarray(modeller.positions.value_in_unit(u.nanometer))) * u.nanometer
        modeller.delete([r for r in modeller.topology.residues() if r.name == "HOH"])
        for atom in modeller.topology.atoms():
            if atom.name in rename:
                atom.name = "CX"
        modeller.addSolvent(forcefield, boxSize=mm.Vec3(3.0, 3.0, 3.0) * u.nanometer)
        return {"phosphate_plane_nm": 1.0, "warnings": ["a warning of the build"]}

    monkeypatch.setattr("cufep.prep.membrane.add_membrane", bilayer_in_its_place)
    lig_dir = out / ligand.name
    lines = []
    info = workflow.prepare_ligand(cfg, ligand, protein, lig_dir, True, "dodecahedron", lines.append)
    assert info == json.loads((lig_dir / "prep" / "prep.json").read_text())
    stage = workflow.restraint_system(info)
    assert (info["complex_barostat"], info["membrane"], info["membrane_info"]) == ("isotropic", True, {})
    assert stage == {"name": "complex_bilayer", "barostat": "membrane", "n_atoms": stage["n_atoms"],
                     "membrane_info": {"phosphate_plane_nm": 1.0}}
    assert info["warnings"] == ["membrane: a warning of the build"]
    assert info["made_from"]["membrane"]["complex_leg"] == "water"
    assert f"complex in the bilayer, for the MD of the restraint, {stage['n_atoms']} atoms" in lines[-1]
    water, bilayer = (workflow.load_prepared(lig_dir / "prep", name) for name in ("complex", "complex_bilayer"))
    assert (len(water["positions"]), len(bilayer["positions"])) == (info["n_atoms_complex"], stage["n_atoms"])
    # the crystal water (three atoms) comes before the ligand and is in the complex in water alone
    assert water["protein"] == bilayer["protein"] and [i - 3 for i in water["ligand"]] == bilayer["ligand"]
    atom_map = workflow.solute_correspondence(bilayer, water)
    assert [atom_map[i] for i in bilayer["ligand"]] == water["ligand"]
    # the two boxes: of a complex in water, and the rectangular one of the bilayer
    assert not np.allclose(water["box"], np.diag(np.diag(water["box"])))
    assert np.allclose(bilayer["box"], 3.0 * np.eye(3))
    # a run with the bilayer in the alchemical run is refused in this directory, and the other way round
    in_bilayer = Config.model_validate(dict(cfg.model_dump(mode="json"), membrane={"enabled": True}))
    with pytest.raises(workflow.WorkflowError, match="cannot be used again.*membrane.complex_leg: water -> "):
        workflow.check_prepared(in_bilayer, ligand, lig_dir, True, "dodecahedron")

    rename.append("CB")
    other = out / "another"
    with pytest.raises(workflow.WorkflowError, match="not of the same protein and ligand: atom .* is ALA2:CX of chain "
                                                     "A in one and ALA2:CB of chain A in the other"):
        workflow.prepare_ligand(cfg, ligand, protein, other, True, "dodecahedron", lines.append)
    assert list((other / "prep").iterdir()) == []
