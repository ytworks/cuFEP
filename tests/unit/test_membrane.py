"""Ions of a membrane system, and notes on a bilayer made again from stored data."""
import json

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import pytest

from cufep.prep.membrane import add_ions


def _counts(modeller, forcefield):
    names = [r.name for r in modeller.topology.residues()]
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles()))
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


def test_a_bilayer_that_still_changes_is_recognised():
    from cufep.sampling.equilibrate import membrane_settled

    rng = np.random.default_rng(1)
    steady = 64.0 + rng.normal(scale=0.2, size=400)
    waters = 6.0 + rng.normal(scale=0.5, size=400)
    out = membrane_settled(steady, waters, 0.01, 1.5)
    assert out["settled"] is True and abs(out["area_change"]) < 0.002 and abs(out["pocket_water_change"]) < 0.3
    shrinking = steady * np.linspace(1.10, 1.0, 400)               # the area still falls by 2.5 % per quarter
    out = membrane_settled(shrinking, waters, 0.01, 1.5)
    assert out["settled"] is False and out["area_change"] == pytest.approx(-0.024, abs=0.004)
    filling = waters + np.linspace(0.0, 12.0, 400)                 # three more waters near the ligand per quarter
    out = membrane_settled(steady, filling, 0.01, 1.5)
    assert out["settled"] is False and out["pocket_water_change"] == pytest.approx(3.0, abs=0.3)
    assert membrane_settled(steady, [], 0.01, 1.5)["pocket_water_change"] is None
    assert membrane_settled(steady[:12], waters[:12], 0.01, 1.5)["settled"] is None     # too short to say


def test_stored_verdicts_are_made_again_with_the_levels_of_this_run(tmp_path):
    """Review of PR 1: a leg that ran again returned the notes of its first run: a bilayer whose area
    changed by 1.5 % stayed "settled" or "not settled" whatever membrane.area_tolerance said now, and
    the note about the pose kept the level of the first run."""
    from cufep.config.schema import Config
    from cufep.workflow import equil_notes, restraint_made_with, restraint_notes

    def config(**sections):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, **sections})

    # the area falls by 1.5 % from the third to the fourth quarter; the file holds the note of a run
    # that had a tolerance of 1 %, and a note that has nothing to do with the bilayer
    area = np.concatenate([np.full(200, 66.0), np.full(100, 65.0), np.full(100, 65.0 * 0.985)])
    old_note = ("the bilayer had not settled after 10 ns of equilibration: between the last two quarters the area "
                "in the membrane plane changed by -1.5 % (tolerance 1 %). A longer membrane.equilibration_ns, in a "
                "new output.dir, gives it more time")
    np.savez(tmp_path / "equil.npz", notes=json.dumps(["the time step was reduced to 2 fs", old_note]), area_nm2=area,
             pocket_waters=np.full(400, 6.0), simulated_ns=10.0)
    eq = np.load(tmp_path / "equil.npz")
    strict = equil_notes(eq, config(membrane={"enabled": True, "area_tolerance": 0.01}), True)
    assert strict == ["the time step was reduced to 2 fs", old_note.replace("(tolerance 1 %)", "(tolerance 1 %) and the "
                      "number of water molecules near the ligand by +0.0 (tolerance 1.5)")]
    assert equil_notes(eq, config(membrane={"enabled": True, "area_tolerance": 0.02}), True) \
        == ["the time step was reduced to 2 fs"]                                 # 1.5 % is within 2 %
    waters = equil_notes(eq, config(membrane={"enabled": True, "area_tolerance": 0.02, "pocket_water_tolerance": 0.5}), True)
    assert waters == ["the time step was reduced to 2 fs"]                       # the waters did not change
    assert equil_notes(eq, config(), False) == ["the time step was reduced to 2 fs", old_note]   # not a bilayer leg

    # the pose: 2.5 A from the input pose, stored with the note of a run whose level was 2 A
    stored = {"info": {"selection": "auto", "ligand_rmsd_A": 2.5, "ligand_rmsd_reference": "input pose",
                       "notes": ["a note of the search",
                                 "the ligand moved away from its input pose in the pre-equilibration MD (heavy-atom "
                                 "RMSD 2.5 A over the second half, above restraints.pose_rmsd_warning_nm = 2 A); the "
                                 "result is for the pose it moved to"]}}
    assert restraint_notes(stored, config(restraints={"pose_rmsd_warning_nm": 0.3})) == ["a note of the search"]
    again = restraint_notes(stored, config(restraints={"pose_rmsd_warning_nm": 0.2}))
    assert again == stored["info"]["notes"]
    # the level is not part of what a restraint is made with: changing it does not refuse the file
    assert restraint_made_with(config(restraints={"pose_rmsd_warning_nm": 0.2})) == restraint_made_with(config())
    # a file whose distance was measured otherwise, or that holds none: kept, and said so
    older = {"info": {"selection": "auto", "ligand_rmsd_A": 0.8, "notes": ["a note of the search"]}}
    notes = restraint_notes(older, config(restraints={"pose_rmsd_warning_nm": 0.05}))
    assert notes[0] == "a note of the search" and "could not be compared with restraints.pose_rmsd_warning_nm" in notes[1]
    assert restraint_notes({"info": {"selection": "manual"}}, config()) == []


def test_the_notes_on_bilayer_and_pose_of_a_leg_that_is_not_run_follow_the_levels_of_the_run(tmp_path):
    """Review of PR 1: a complex leg finished with membrane.area_tolerance 0.02 (the area changed by 1.5 %:
    no note). A run of the solvent leg alone with 0.01 found the result of the complex leg "current" and
    combined it without a note; made again for another reason, it kept the notes of the old run."""
    import h5py

    from test_sampling import synthetic_leg

    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.config.schema import Config
    from cufep.workflow import (analysis_record, combine_ligand, leg_result_from_samples, notes_of_leg_files,
                                refresh_leg_result, write_leg_result)

    def config(area=0.02, pose=0.3, **analysis):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                      "membrane": {"enabled": True, "area_tolerance": area},
                                      "restraints": {"pose_rmsd_warning_nm": pose}, "analysis": {"lrc": "none", **analysis}})

    lig = tmp_path / "lig"
    legs = {name: lig / "repeat_0" / name for name in ("complex", "solvent")}
    for name, leg in legs.items():
        synthetic_leg(leg)
        with h5py.File(leg / "hrex" / "samples.h5", "a") as h5:        # the conditions say what kind of leg it is
            h5.attrs["conditions"] = json.dumps({"barostat": "membrane" if name == "complex" else "isotropic",
                                                 "pressure_bar": 1.0})
    # the files of the complex leg: the area falls by 1.5 % between the last two quarters, the ligand is
    # 2.5 A from its input pose; both files hold a note that depends on no level
    area = np.concatenate([np.full(200, 66.0), np.full(100, 65.0), np.full(100, 65.0 * 0.985)])
    np.savez(legs["complex"] / "equil.npz", notes=json.dumps(["the time step was reduced to 2 fs"]), area_nm2=area,
             pocket_waters=np.full(400, 6.0), simulated_ns=10.0)
    np.savez(legs["solvent"] / "equil.npz", notes="[]", area_nm2=area, pocket_waters=np.zeros(0), simulated_ns=1.0)
    restraint = BoreschRestraint((0, 1, 2), (3, 4, 5), 0.8, 1.6, 1.6, 0.0, 2.0, 1.0)
    (legs["complex"] / "restraint.json").write_text(json.dumps({"restraint": restraint.to_dict(), "info": {
        "selection": "auto", "ligand_rmsd_A": 2.5, "ligand_rmsd_reference": "input pose", "notes": ["a note of the search"]}}))

    def run(name):
        return {"ligand": "lig", "leg": name, "repeat": 0, "backend": "openmm", "timestep_fs": 4.0,
                "notes": ["a note of the run"], "prepared_systems_sha256": "abc"}

    first = config(area=0.02, pose=0.3)
    for name, leg in legs.items():
        write_leg_result(leg, leg_result_from_samples(leg, first, run(name)))
    kept = ["the time step was reduced to 2 fs", "a note of the search", "a note of the run"]
    finished = json.loads((legs["complex"] / "result.json").read_text())
    assert finished["notes"] == kept and finished["run"]["notes"] == ["a note of the run"]
    assert finished["analysis_record"] == analysis_record(first) and finished["analysis_record"]["membrane_area_tolerance"] == 0.02
    assert json.loads((legs["solvent"] / "result.json").read_text())["notes"] == ["a note of the run"]
    # the same levels again, although they are not the defaults: nothing is made again
    said = []
    assert refresh_leg_result(legs["complex"], first, said.append) == "current" and said == []

    # the next run asks for 1 % and 2 A and samples the solvent leg alone: the complex leg is made again
    strict = config(area=0.01, pose=0.2)
    assert refresh_leg_result(legs["complex"], strict, said.append) == "reanalysed"
    assert "membrane_area_tolerance: 0.02 -> 0.01; pose_rmsd_warning_nm: 0.3 -> 0.2" in said[0]
    again = json.loads((legs["complex"] / "result.json").read_text())
    assert again["notes"][0] == "the time step was reduced to 2 fs" and again["notes"][-1] == "a note of the run"
    assert again["notes"][1].startswith("the bilayer had not settled after 10 ns of equilibration: between the last two "
                                        "quarters the area in the membrane plane changed by -1.5 % (tolerance 1 %)")
    assert again["notes"][2] == "a note of the search"
    assert again["notes"][3].startswith("the ligand moved away from its input pose in the pre-equilibration MD "
                                        "(heavy-atom RMSD 2.5 A over the second half, above "
                                        "restraints.pose_rmsd_warning_nm = 2 A)")
    assert len(again["notes"]) == 5 and again["delta_g_decouple"] == finished["delta_g_decouple"]
    assert refresh_leg_result(legs["solvent"], strict, said.append) == "reanalysed"       # its record names the levels too
    combined = combine_ligand(lig, "lig", 1, strict)
    assert any(w.startswith("repeat 0 complex: the bilayer had not settled") for w in combined["warnings"])
    assert any(w.startswith("repeat 0 complex: the ligand moved away from its input pose") for w in combined["warnings"])
    assert combined["old_analysis"] == [] and "dg_bind" in combined
    assert refresh_leg_result(legs["complex"], strict, said.append) == "current"

    # back to the first levels: the notes of the stricter run do not stay
    assert refresh_leg_result(legs["complex"], first, said.append) == "reanalysed"
    assert json.loads((legs["complex"] / "result.json").read_text())["notes"] == kept
    # a result of a version whose run kept these notes itself: they are judged anew, not doubled
    old = dict(again, run=dict(again["run"], notes=again["notes"][:4] + ["a note of the run"]))
    (legs["complex"] / "result.json").write_text(json.dumps(old))
    assert leg_result_from_samples(legs["complex"], first, old["run"])["notes"] == kept
    assert leg_result_from_samples(legs["complex"], strict, old["run"])["notes"] == again["notes"]
    # without the files nothing can be judged again: what the run noted stays as it is
    notes, judged = notes_of_leg_files(legs["solvent"].parent / "nothing", strict, {})
    assert notes == [] and judged == ()
    (legs["complex"] / "equil.npz").unlink()
    without = leg_result_from_samples(legs["complex"], first, old["run"])["notes"]
    assert without[0] == "a note of the search" and any(n.startswith("the bilayer had not settled") for n in without)
    assert not any(n.startswith("the ligand moved away") for n in without)

