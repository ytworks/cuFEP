"""Ions of a membrane system, the numbers of a bilayer, and the notes of stored data."""
import json

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.prep.membrane import add_ions


def _counts(modeller, forcefield):
    names = [r.name for r in modeller.topology.residues()]
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(
        nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles())
    )
    return {"water": names.count("HOH"), "NA": names.count("NA"), "CL": names.count("CL"), "charge": round(charge, 6)}


@pytest.fixture()
def peptides_in_water(dipeptides_in_water):
    _system, topology, positions, _box, _second, _first = dipeptides_in_water
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    def build(extra_sodium: bool = False):
        modeller = app.Modeller(topology, positions * u.nanometer)
        if extra_sodium:                                         # a system with a net charge of +1
            ion = app.Topology()
            ion.addAtom("NA", app.Element.getBySymbol("Na"), ion.addResidue("NA", ion.addChain()))
            box = modeller.topology.getPeriodicBoxVectors()
            modeller.add(ion, [mm.Vec3(0.1, 0.1, 0.1)] * u.nanometer)
            modeller.topology.setPeriodicBoxVectors(box)
        return modeller

    return build, forcefield


def test_a_neutral_system_without_salt_keeps_all_its_water(peptides_in_water):
    """Review of PR 1: with 0 M and no net charge the function reported no ion and still made a water a Cl-."""
    build, forcefield = peptides_in_water
    modeller = build()
    before = _counts(modeller, forcefield)
    assert before["charge"] == 0.0 and before["NA"] == before["CL"] == 0 and before["water"] > 100
    info = add_ions(modeller, forcefield, 0.0, 0.0, log=lambda m: None)
    assert (info["n_sodium"], info["n_chloride"], info["net_charge_before_ions"]) == (0, 0, 0)
    assert _counts(modeller, forcefield) == before


def test_reported_ions_are_the_ions_in_the_topology(peptides_in_water):
    build, forcefield = peptides_in_water
    # salt in a neutral system: as many of each as reported, one water gone for each ion
    modeller = build()
    before = _counts(modeller, forcefield)
    info = add_ions(modeller, forcefield, 0.5, 0.0, log=lambda m: None)
    after = _counts(modeller, forcefield)
    assert info["n_sodium"] == info["n_chloride"] >= 1
    assert (after["NA"], after["CL"]) == (info["n_sodium"], info["n_chloride"])
    assert after["water"] == before["water"] - info["n_sodium"] - info["n_chloride"] and after["charge"] == 0.0
    # a net charge without salt: only the counter-ion
    modeller = build(extra_sodium=True)
    before = _counts(modeller, forcefield)
    assert before["charge"] == 1.0
    info = add_ions(modeller, forcefield, 0.0, 0.0, log=lambda m: None)
    after = _counts(modeller, forcefield)
    assert (info["n_sodium"], info["n_chloride"], info["net_charge_before_ions"]) == (0, 1, 1)
    assert (after["NA"], after["CL"], after["water"], after["charge"]) == (1, 1, before["water"] - 1, 0.0)


def test_how_the_bilayer_changed_is_given_as_numbers_without_a_verdict():
    """The owner, 2026-11-20: no verdict with a level of cuFEP's own. How the area of the bilayer and the
    waters near the ligand changed from the third to the fourth quarter of its equilibration are numbers;
    nothing says "settled" (earlier: 1 % and 1.5 waters), and nothing makes the run longer for it."""
    from cufep.sampling.equilibrate import EquilibrationResult, EquilibrationSettings, membrane_drift

    rng = np.random.default_rng(1)
    steady = 64.0 + rng.normal(scale=0.2, size=400)
    waters = 6.0 + rng.normal(scale=0.5, size=400)
    out = membrane_drift(steady, waters)
    assert set(out) == {"area_change", "pocket_water_change"}
    assert abs(out["area_change"]) < 0.002 and abs(out["pocket_water_change"]) < 0.3
    shrinking = steady * np.linspace(1.10, 1.0, 400)               # the area still falls by 2.5 % per quarter
    assert membrane_drift(shrinking, waters)["area_change"] == pytest.approx(-0.024, abs=0.004)
    filling = waters + np.linspace(0.0, 12.0, 400)                 # three more waters near the ligand per quarter
    assert membrane_drift(steady, filling)["pocket_water_change"] == pytest.approx(3.0, abs=0.3)
    assert membrane_drift(steady, [])["pocket_water_change"] is None
    assert membrane_drift(steady[:12], waters[:12]) == {"area_change": None, "pocket_water_change": None}
    # no levels and no extension
    assert not set(EquilibrationSettings.__dataclass_fields__) & {
        "max_production_ns",
        "area_tolerance",
        "water_tolerance",
    }
    assert "settled" not in EquilibrationResult.__dataclass_fields__


#: the notes with levels of cuFEP's own that version 9 wrote on a bilayer, a pose and a structure
BILAYER_VERDICT = (
    "the bilayer had not settled after 10 ns of equilibration: between the last two quarters the area in "
    "the membrane plane changed by -1.5 % (tolerance 1 %). A longer membrane.equilibration_ns, in a new "
    "output.dir, gives it more time"
)
POSE_VERDICT = ("the ligand moved away from its input pose in the pre-equilibration MD (heavy-atom RMSD 2.5 A over the "
                "second half, above restraints.pose_rmsd_warning_nm = 2 A); the result is for the pose it moved to")
UNCOMPARED_POSE = (
    "the distance of the ligand from its input pose was measured by an earlier version of cuFEP in another "
    "way (or not stored) and could not be compared with restraints.pose_rmsd_warning_nm of this run"
)
STRUCTURE_VERDICT = ("the ligand's centre of mass drifted 6.0 A in the coupled state; it may have left the pocket")


def test_the_verdicts_that_earlier_versions_stored_are_removed_and_the_numbers_kept(tmp_path):
    """The notes of a stored pre-equilibration and of a stored restraint lose the verdicts with levels of
    cuFEP's own that earlier versions wrote into them; the other notes stay, and the numbers behind the
    verdicts are given as they are (the area fell by 1.5 %, the ligand was 2.5 A from its input pose)."""
    from cufep.config.schema import Config
    from cufep.workflow import bilayer_numbers, equil_notes, restraint_made_with, restraint_notes

    area = np.concatenate([np.full(200, 66.0), np.full(100, 65.0), np.full(100, 65.0 * 0.985)])
    np.savez(tmp_path / "equil.npz", notes=json.dumps(["the time step was reduced to 2 fs", BILAYER_VERDICT]),
             area_nm2=area, pocket_waters=np.full(400, 6.0), simulated_ns=10.0)
    eq = np.load(tmp_path / "equil.npz")
    assert equil_notes(eq) == ["the time step was reduced to 2 fs"]
    assert bilayer_numbers(eq) == {"area_change": pytest.approx(-0.015), "pocket_water_change": 0.0}
    np.savez(tmp_path / "solvent.npz", notes="[]", area_nm2=area, pocket_waters=np.zeros(0), simulated_ns=1.0)
    assert bilayer_numbers(np.load(tmp_path / "solvent.npz")) is None                 # not a bilayer

    stored = {"info": {"selection": "auto", "ligand_rmsd_A": 2.5, "ligand_rmsd_reference": "input pose",
                       "notes": ["a note of the search", POSE_VERDICT, UNCOMPARED_POSE]}}
    assert restraint_notes(stored) == ["a note of the search"]
    assert restraint_notes({"info": {"selection": "manual"}}) == []
    # the settings of the restraints hold no level any more, and a file that names one is read with a notice
    assert not any("pose" in key or "warning" in key for key in restraint_made_with(Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}})))


def test_a_leg_result_gives_the_bilayer_and_the_pose_as_numbers_without_verdicts(tmp_path):
    """The result of a leg gives how the bilayer changed at the end of its equilibration and how far the
    ligand was from its input pose as numbers, from the files of the leg. The verdicts with levels of
    cuFEP's own that version 9 wrote (in those files, in the notes of the run, in a stored result) are
    gone from the result that is made again, which every result of version 9 is."""
    from conftest import edit_attributes
    from test_sampling import synthetic_leg

    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.config.schema import Config
    from cufep.workflow import (
        analysis_record,
        archive_leg,
        combine_ligand,
        leg_files,
        leg_result_from_samples,
        notes_of_leg_files,
        refresh_leg_result,
        write_leg_result,
    )

    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "membrane": {"enabled": True},
                                 "analysis": {"lrc": "none"}})
    assert not any("tolerance" in key or "pose" in key for key in analysis_record(cfg))
    lig = tmp_path / "lig"
    legs = {name: lig / "repeat_0" / name for name in ("complex", "solvent")}
    for name, leg in legs.items():
        synthetic_leg(leg)
        edit_attributes(leg / "hrex", {"conditions": {"barostat": "membrane" if name == "complex" else "isotropic",
                                                      "pressure_bar": 1.0}})   # the conditions say what kind of leg
    area = np.concatenate([np.full(200, 66.0), np.full(100, 65.0), np.full(100, 65.0 * 0.985)])
    np.savez(legs["complex"] / "equil.npz", notes=json.dumps(["the time step was reduced to 2 fs", BILAYER_VERDICT]),
             area_nm2=area, pocket_waters=np.full(400, 6.0), simulated_ns=10.0)
    np.savez(legs["solvent"] / "equil.npz", notes="[]", area_nm2=area, pocket_waters=np.zeros(0), simulated_ns=1.0)
    restraint = BoreschRestraint((0, 1, 2), (3, 4, 5), 0.8, 1.6, 1.6, 0.0, 2.0, 1.0)
    (legs["complex"] / "restraint.json").write_text(json.dumps({"restraint": restraint.to_dict(), "info": {
        "selection": "auto", "ligand_rmsd_A": 2.5, "ligand_rmsd_reference": "input pose",
        "notes": ["a note of the search", POSE_VERDICT]}}))

    def run(name):
        # the run of version 9 kept the verdicts among its notes too
        return {"ligand": "lig", "leg": name, "repeat": 0, "backend": "openmm", "timestep_fs": 4.0,
                "notes": ["a note of the run", BILAYER_VERDICT, STRUCTURE_VERDICT], "prepared_systems_sha256": "abc"}

    kept = ["the time step was reduced to 2 fs", "a note of the search", "a note of the run"]
    for name, leg in legs.items():
        write_leg_result(leg, leg_result_from_samples(leg, cfg, run(name)))
    finished = json.loads((legs["complex"] / "result.json").read_text())
    assert finished["notes"] == kept
    assert finished["bilayer"] == {"area_change": pytest.approx(-0.015), "pocket_water_change": 0.0}
    assert finished["ligand_rmsd_from_input_A"] == 2.5
    solvent = json.loads((legs["solvent"] / "result.json").read_text())
    assert solvent["notes"] == ["a note of the run"]
    assert solvent["bilayer"] is None and solvent["ligand_rmsd_from_input_A"] is None
    said = []
    assert refresh_leg_result(legs["complex"], cfg, said.append) == "current" and said == []

    # a result of version 9 that holds the verdicts: made again, without them
    old = dict(finished, notes=kept[:2] + [BILAYER_VERDICT, POSE_VERDICT, "a note of the run"],
               analysis_record=dict(finished["analysis_record"], version=9, membrane_area_tolerance=0.01,
                                    membrane_pocket_water_tolerance=1.5, pose_rmsd_warning_nm=0.3))
    old.pop("bilayer")
    old.pop("ligand_rmsd_from_input_A")
    (legs["complex"] / "result.json").write_text(json.dumps(old))
    assert refresh_leg_result(legs["complex"], cfg, said.append) == "reanalysed"
    assert "version 9 of the analysis" in said[0]
    again = json.loads((legs["complex"] / "result.json").read_text())
    assert again["notes"] == kept and again["bilayer"] == finished["bilayer"]
    assert again["ligand_rmsd_from_input_A"] == 2.5 and again["delta_g_decouple"] == finished["delta_g_decouple"]
    combined = combine_ligand(lig, "lig", 1, cfg)
    assert not any("settled" in w or "moved away" in w or "drifted" in w for w in combined["warnings"])
    assert "dg_bind" in combined
    # a result whose structure numbers are of the earlier rule for the periodic image of the ligand (its record
    # has no version of them) is made again too, with the free energy it had (review of eb6f89f: it stayed as
    # "current", with a ligand RMSD of the length of the cell where the ligand was far from the centre)
    before = dict(again, analysis_record={k: v for k, v in again["analysis_record"].items() if k != "structure"})
    assert again["analysis_record"]["structure"] == 2 and again["analysis_record"]["version"] == 10
    (legs["complex"] / "result.json").write_text(json.dumps(before))
    assert refresh_leg_result(legs["complex"], cfg, said.append) == "reanalysed"
    once_more = json.loads((legs["complex"] / "result.json").read_text())
    assert once_more["analysis_record"] == again["analysis_record"]
    assert once_more["delta_g_decouple"] == again["delta_g_decouple"]

    # review of PR 1: a leg run again with --reschedule keeps its earlier samples in previous_<n>/; the archive
    # holds copies of the files of the leg, and an archive of an earlier version is read with the files above it
    archived = archive_leg(legs["complex"])
    assert (archived / "equil.npz").exists() and (archived / "restraint.json").exists()
    from_archive = leg_result_from_samples(archived, cfg, run("complex"))
    assert from_archive["notes"] == kept and from_archive["bilayer"] == finished["bilayer"]
    (archived / "equil.npz").unlink()
    (archived / "restraint.json").unlink()
    assert leg_files(archived) == {"equil.npz": legs["complex"] / "equil.npz",
                                   "restraint.json": legs["complex"] / "restraint.json"}
    assert leg_result_from_samples(archived, cfg, run("complex"))["ligand_rmsd_from_input_A"] == 2.5
    for name in ("hrex", "result.json"):
        (archived / name).rename(legs["complex"] / name)

    # what a restraint chosen from interactions rests on is a number of its file too, and of no other restraint:
    # the same through a new result, a result made again, the result of the ligand and an archive
    assert finished["restraint_choice"] is None and again["restraint_choice"] is None
    choice = {"method": "interactions", "anchors": {"host": ["VAL981:N", "VAL981:CA", "VAL981:C"],
                                                    "guest": ["LIG1:N2x", "LIG1:C7x", "LIG1:C6x"]},
              "interaction": {"residue": "VAL981", "type": "HBAcceptor", "frames": 744, "of_frames": 800},
              "score": 107.6, "candidates_tested": 32, "candidates_refused": 0,
              "md": {"trajectories": 4, "frames_analysed": 800, "backends": ["cupotential"] * 4,
                     "timesteps_fs": [4.0] * 4}}
    (legs["complex"] / "restraint.json").write_text(json.dumps({"restraint": restraint.to_dict(), "info": dict(
        choice, selection="interactions", ligand_rmsd_A=2.5, ligand_rmsd_reference="input pose",
        notes=["MD 2 for the restraint: time step lowered automatically to 2 fs after an unstable run at 4 fs"],
        versions={"prolif": "2.2.3"})}))
    from_interactions = leg_result_from_samples(legs["complex"], cfg, run("complex"))
    assert from_interactions["restraint_choice"] == choice
    assert "MD 2 for the restraint: time step lowered automatically to 2 fs after an unstable run at 4 fs" \
        in from_interactions["notes"]
    write_leg_result(legs["complex"], from_interactions)
    assert refresh_leg_result(legs["complex"], cfg, said.append) == "current"
    assert combine_ligand(lig, "lig", 1, cfg)["repeats"][0]["restraint_choice"] == choice
    # `cufep analyze` prints it too (review of eb6f89f: the command printed the analysis and left it out)
    from click.testing import CliRunner

    from cufep.cli import main

    printed = json.loads(CliRunner().invoke(main, ["analyze", str(legs["complex"])]).output)
    assert printed["restraint_choice"] == choice
    assert "restraint_choice" not in json.loads(CliRunner().invoke(main, ["analyze", str(legs["solvent"])]).output)
    assert leg_result_from_samples(archive_leg(legs["complex"]), cfg, run("complex"))["restraint_choice"] == choice

    # without the files there are neither notes nor numbers of them
    assert notes_of_leg_files(legs["solvent"].parent / "nothing") == (
        [], {"bilayer": None, "ligand_rmsd_from_input_A": None, "restraint_choice": None})



def _waters(oxygens_nm):
    """A topology of water molecules with their oxygens at ``oxygens_nm`` and their positions (nm)."""
    top = app.Topology()
    chain = top.addChain()
    positions = []
    for k, o in enumerate(np.asarray(oxygens_nm, dtype=float)):
        res = top.addResidue("HOH", chain, id=str(k + 1))
        ao = top.addAtom("O", app.element.oxygen, res)
        h1 = top.addAtom("H1", app.element.hydrogen, res)
        h2 = top.addAtom("H2", app.element.hydrogen, res)
        top.addBond(ao, h1)
        top.addBond(ao, h2)
        positions += [o, o + [0.0957, 0.0, 0.0], o + [-0.024, 0.0927, 0.0]]
    return top, np.asarray(positions)


def test_waters_that_overlap_across_the_boundary_of_a_trimmed_bilayer_are_removed():
    """Review of 771a7ab (B5, the A2A example): trim_membrane removed waters that overlapped a lipid across the new
    periodic boundary, but not waters that overlapped each other there; two waters on the two sides of the new boundary
    ended with their oxygens 0.092 A apart, and the pre-equilibration ended in NaN at 4 and at 2 fs. Of every pair of
    overlapping waters in the final box one goes; a water that overlaps another molecule goes."""
    from cufep.prep.membrane import CLASH_NM, _kind, _molecules, boundary_water_overlaps

    edge, height = 3.0, 4.0
    built = (6.0, 6.0, 8.0)                                             # the box before trimming: none of these met
    oxygens = [[0.0046, 1.0, 1.0], [3.0 - 0.0046, 1.0, 1.0],            # 0.092 A apart through x
               [1.5, 0.1, 2.0], [1.5, 2.95, 2.0],                         # 0.15 nm apart through y
               [1.0, 1.0, 0.05], [1.0, 1.0, 3.93],                        # 0.12 nm apart through z
               [1.5, 1.5, 1.0], [2.0, 2.0, 3.0]]                          # far from everything
    top, pos = _waters(oxygens)
    atoms = list(top.atoms())
    kinds = np.array([_kind(a.residue) for a in atoms])
    heavy = np.array([a.element != app.element.hydrogen for a in atoms])
    molecules = _molecules(top)
    removed = boundary_water_overlaps(pos, kinds, heavy, molecules, set(), (edge, edge, height), built)
    gone = sorted({atoms[i].residue.index for i in removed})
    assert len(gone) == 3 and {0, 1} & set(gone) and {2, 3} & set(gone) and {4, 5} & set(gone)
    assert 6 not in gone and 7 not in gone and all(len([i for i in removed if atoms[i].residue.index == r]) == 3
                                                    for r in gone)
    # what is left overlaps nothing in the final box
    left = [i for i in range(len(atoms)) if heavy[i] and i not in removed]
    box = np.array([edge, edge, height])
    d = np.linalg.norm((pos[left][:, None] - pos[left][None] + box / 2) % box - box / 2, axis=2)
    assert d[np.triu_indices(len(left), 1)].min() >= CLASH_NM
    # atoms that are already removed do not count: with the first water gone, its partner stays
    without_first = {atoms[i].residue.index for i in boundary_water_overlaps(
        pos, kinds, heavy, molecules, {a.index for a in atoms if a.residue.index == 0}, (edge, edge, height), built)}
    assert 1 not in without_first and len(without_first) == 2
    # a water that the new boundary brings within 0.1 nm of the protein goes whatever else
    protein_kinds = kinds.copy()
    protein_kinds[[a.index for a in list(top.residues())[7].atoms()]] = "protein"
    close = pos.copy()
    close[[a.index for a in list(top.residues())[6].atoms()]] += pos[21] - pos[18] + [0.1, 0.0, -height]
    taken = {atoms[i].residue.index for i in boundary_water_overlaps(close, protein_kinds, heavy, molecules, set(),
                                                                     (edge, edge, height), built)}
    assert 6 in taken and 7 not in taken
    # review of 7b49066: a contact that the built system had is not the new boundary's, and stays
    close[[a.index for a in list(top.residues())[6].atoms()]] += [0.0, 0.0, height]
    taken = {atoms[i].residue.index for i in boundary_water_overlaps(close, protein_kinds, heavy, molecules, set(),
                                                                     (edge, edge, height), built)}
    assert 6 not in taken


def test_trimming_keeps_the_contacts_of_the_built_system():
    """Review of 7b49066 (P2): every close pair in the trimmed box was resolved, so a water coordinating a Mg ion in
    the middle of the box (0.194 nm, the distance of Amber/TIP3P minimised on OpenMM's Reference platform) was removed
    when a 6 nm box was cut to 3 nm. Only pairs that the new boundary brings together are resolved: the Mg water stays,
    and of two waters 0.04 nm apart through the new boundary one goes. The same rule holds for lipids and the waters
    near them (contact_is_new)."""
    from cufep.prep.membrane import _kind, _molecules, boundary_water_overlaps, contact_is_new

    top, pos = _waters([[1.5, 1.5 + 0.194, 3.0], [0.02, 1.0, 1.0], [2.98, 1.0, 1.0]])
    chain = list(top.chains())[0]
    mg = top.addAtom("MG", app.element.magnesium, top.addResidue("MG", chain, id="9"))
    pos = np.vstack([pos, [[1.5, 1.5, 3.0]]])
    atoms = list(top.atoms())
    kinds = np.array([_kind(a.residue) for a in atoms])
    assert kinds[mg.index] == "ion"
    heavy = np.array([a.element != app.element.hydrogen for a in atoms])
    molecules = _molecules(top)
    gone = {atoms[i].residue.index for i in boundary_water_overlaps(pos, kinds, heavy, molecules, set(),
                                                                      (3.0, 3.0, 6.0), (6.0, 6.0, 6.0))}
    assert 0 not in gone and len(gone & {1, 2}) == 1 and len(gone) == 1
    built, trimmed = (6.0, 6.0, 6.0), (3.0, 3.0, 6.0)
    assert not contact_is_new(pos, 0, mg.index, built, trimmed) and contact_is_new(pos, 3, 6, built, trimmed)
    # review of 10589a6 (P3): a water exactly 0.240 nm from the ion (CLASH_NM, which the search of the trimmed box
    # includes) was taken for a new contact and removed; the same image gives the same distance in both boxes
    from cufep.prep.membrane import CLASH_NM

    at_level = pos.copy()
    at_level[[0, 1, 2]] += [0.0, CLASH_NM - 0.194, 0.0]
    assert np.linalg.norm(at_level[0] - at_level[mg.index]) == pytest.approx(CLASH_NM, abs=1e-12)
    gone = {atoms[i].residue.index for i in boundary_water_overlaps(at_level, kinds, heavy, molecules, set(), trimmed,
                                                                      built)}
    assert 0 not in gone and len(gone) == 1 and not contact_is_new(at_level, 0, mg.index, built, trimmed)
