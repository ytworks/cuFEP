"""Input checks, membrane detection, GPU allocation and result combination."""
import json

import numpy as np
import pytest

from cufep.prep.ligand import LigandError, load_ligands
from cufep.prep.orient import consensus_orientation, orient_positions, slab_orientation, transfer_energy_orientation
from cufep.prep.protein import prepare_protein
from cufep.scheduler import Demand, allocate, oversubscription_for
from cufep.workflow import LegJob, check_ligand, combine_ligand


@pytest.fixture(scope="module")
def tyk2(repo):
    return prepare_protein(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb")


@pytest.fixture(scope="module")
def ligands(repo):
    return load_ligands(repo / "benchmarks/tyk2/inputs/tyk2_ligands.sdf")


def test_ligands_load_with_names_and_experimental_values(ligands):
    assert len(ligands) == 16
    lig = next(l for l in ligands if l.name == "ejm_31")
    assert lig.n_atoms == 32 and lig.formal_charge == 0
    assert float(lig.properties["r_exp_dg"]) == pytest.approx(-9.54)


def test_bound_pose_passes_and_displaced_pose_stops(tyk2, ligands):
    lig = ligands[0]
    assert check_ligand(lig, tyk2.topology, tyk2.positions_nm()) == []
    from openff.units import unit
    moved = load_ligands(lig.source)[0]
    moved.offmol._conformers = [moved.offmol.conformers[0] + np.array([40.0, 0.0, 0.0]) * unit.angstrom]
    with pytest.raises(LigandError, match="pocket"):
        check_ligand(moved, tyk2.topology, tyk2.positions_nm())


def test_charged_ligand_stops(repo, tyk2):
    ligs = load_ligands(repo / "benchmarks/a2a/inputs/a2a_hip278_ligands.sdf")
    charged = next(l for l in ligs if l.formal_charge != 0)
    with pytest.raises(LigandError, match="formal charge"):
        check_ligand(charged, tyk2.topology, tyk2.positions_nm())


def test_ligand_without_hydrogens_stops(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    w = Chem.SDWriter(str(tmp_path / "noh.sdf"))
    w.write(Chem.RemoveHs(mol))
    w.close()
    with pytest.raises(LigandError, match="hydrogens"):
        load_ligands(tmp_path / "noh.sdf")


def test_membrane_orientation_methods_agree_for_a_gpcr(repo):
    a2a = prepare_protein(repo / "benchmarks/a2a/inputs/a2a_hip278_protein.pdb")
    pos = a2a.positions_nm()
    transfer, slab = transfer_energy_orientation(a2a.topology, pos), slab_orientation(a2a.topology, pos)
    tilt, depth = transfer.difference(slab)
    assert tilt < 15.0 and depth < 0.5
    assert transfer.score < 0 and 2.0 <= transfer.thickness_nm <= 4.0
    messages = []
    chosen = consensus_orientation(a2a.topology, pos, use_memembed=False, log=messages.append)
    assert chosen.method == "transfer" and any("(agree)" in m for m in messages)
    z = orient_positions(pos, chosen)[:, 2]
    assert z.min() < -1.5 and z.max() > 1.5                      # the protein spans the membrane along z
    # rotating the input must not change the result: the answer is a property of the structure
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = transfer_energy_orientation(a2a.topology, pos @ q.T + 1.0)
    back = rotated.normal @ q
    assert abs(float(back @ transfer.normal)) > np.cos(np.radians(8.0))


def _loads(plan, demands, gpus):
    """Fraction of every GPU that the plan uses."""
    load = {g: 0.0 for g in gpus}
    for d in demands:
        for g in plan[d.job]:
            load[g] += 1.0 / d.factor
    return load


def test_oversubscription_factor_follows_the_system_size():
    assert oversubscription_for(2_565) == 16 and oversubscription_for(34_078) == 8
    assert oversubscription_for(74_000) == 4 and oversubscription_for(500_000) == 2
    assert oversubscription_for(2_565, setting=3) == 3
    assert oversubscription_for(2_565, mps=False) == 1 and oversubscription_for(2_565, setting=8, mps=False) == 1


def test_one_ligand_on_eight_gpus_while_the_complex_is_still_in_plain_md():
    """The solvent leg samples on all GPUs while the complex leg holds a single worker."""
    gpus = list(range(8))
    demands = [Demand(job=0, phase="serial", factor=8, devices=[0]),
               Demand(job=1, phase="sampling", factor=16, max_workers=14, work=1.0)]
    plan = allocate(gpus, demands)
    assert plan[0] == [0]                                      # the running MD stays where it is
    assert len(plan[1]) == 14                                  # one worker per replica, never more
    assert max(plan[1].count(g) for g in gpus) <= 2            # spread over the GPUs
    assert all(v <= 1.0 + 1e-9 for v in _loads(plan, demands, gpus).values())


def test_capacity_is_shared_in_proportion_to_the_remaining_work():
    gpus = [0]
    demands = [Demand(job=0, phase="sampling", factor=8, max_workers=18, work=3.0),
               Demand(job=1, phase="sampling", factor=16, max_workers=18, work=1.0)]
    plan = allocate(gpus, demands)
    assert len(plan[0]) == 6 and len(plan[1]) == 4             # 3/4 and 1/4 of one GPU
    assert _loads(plan, demands, gpus)[0] == pytest.approx(1.0)
    # the job that is nearly done needs less; the other one gets the rest
    demands[1].work = 0.01
    plan = allocate(gpus, demands)
    assert len(plan[0]) == 7 and len(plan[1]) == 2             # 7/8 of the GPU, and the 1/8 that is left over
    assert _loads(plan, demands, gpus)[0] == pytest.approx(1.0)


def test_three_repeats_use_all_gpus_and_every_job_keeps_one_worker():
    gpus = list(range(8))
    demands = [Demand(job=i, phase="sampling", factor=8, max_workers=18, work=1.0) for i in range(3)]
    plan = allocate(gpus, demands)
    assert [len(plan[i]) for i in range(3)] == [18, 18, 18]    # 54 workers fit into 64 places
    load = _loads(plan, demands, gpus)
    assert max(load.values()) <= 1.0 + 1e-9 and min(load.values()) >= 0.75
    crowded = [Demand(job=i, phase="sampling", factor=1, max_workers=18, work=1.0) for i in range(12)]
    plan = allocate(gpus, crowded)                             # no MPS and more jobs than GPUs
    assert all(len(plan[i]) == 1 for i in range(12))


def test_workers_stay_where_they_are_when_a_job_joins():
    gpus = [0, 1, 2, 3]
    a = Demand(job=0, phase="sampling", factor=2, max_workers=8, work=1.0)
    first = allocate(gpus, [a])[0]
    assert first == [0, 0, 1, 1, 2, 2, 3, 3]
    a.devices = first
    plan = allocate(gpus, [a, Demand(job=1, phase="serial", factor=2)])
    assert len(plan[0]) == 7 and plan[1] in ([0], [1], [2], [3])
    assert sum(1 for g in gpus if plan[0].count(g) == 2) == 3  # only one worker had to stop
    assert allocate([], [a]) == {0: []}


def test_combination_follows_the_thermodynamic_cycle(tmp_path):
    def leg(r, name, dg, release=0.0):
        d = tmp_path / f"repeat_{r}" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": dg, "delta_g_mbar_error": 0.4, "restraint_release": release,
            "lrc": -1.0, "backend": "openmm",
            "analysis": {"stage_delta_g": {"vanish": dg}, "min_overlap": 0.02 if (r, name) == (1, "complex") else 0.2},
            "n_states": 14, "production_ns": 5.0, "notes": ["a note"] if r == 0 and name == "complex" else []}))
    for r, (c, s) in enumerate([(160.0, 80.0), (164.184, 80.0)]):
        leg(r, "complex", c, release=-40.0)
        leg(r, "solvent", s)
    out = combine_ligand(tmp_path, "lig", 3)
    # dG_bind = solvent - complex - release, converted to kcal/mol
    assert out["n_repeats"] == 2 and "repeat 2 is incomplete" in out["warnings"]
    assert "repeat 0 complex: a note" in out["warnings"]
    assert out["repeats"][0]["lambda_windows"] == {"complex": 14, "solvent": 14}
    # the lowest overlap of any leg is carried into the summary, where the result line reads it
    assert out["min_overlap"] == 0.02 and out["repeats"][0]["min_overlap"] == {"complex": 0.2, "solvent": 0.2}
    assert out["repeats"][0]["dg_bind"] == pytest.approx((80.0 - 160.0 + 40.0) / 4.184)
    assert out["dg_bind"] == pytest.approx((-40.0 - 44.184) / 2 / 4.184)
    assert out["dg_bind_std"] == pytest.approx(np.std([-40.0, -44.184], ddof=1) / 4.184)
    # legs written before the checks existed carry no verdict
    assert out["not_reliable"] == [] and out["few_samples"] == [] and out["halves_differ"] == []
    assert out["halves_not_compared"] == [] and out["estimators_disagree"] == []


def test_checks_of_the_analysis_reach_the_leg_notes_and_the_ligand_result(tmp_path):
    from cufep.analysis.mbar import quality_checks
    from cufep.config.schema import Config
    from cufep.workflow import leg_quality, quality_warnings

    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}})
    kcal = 4.184
    # kT is 0.59 kcal/mol: the halves of the steady leg differ by 0.5 kT, those of the other by 2.9 kT
    steady = {"delta_g": 100.0 * kcal, "delta_g_error": 0.2 * kcal, "min_overlap": 0.11, "overlap_neighbour": [0.2, 0.11],
              "first_half": [100.2 * kcal, 0.3 * kcal], "second_half": [99.9 * kcal, 0.3 * kcal],
              "bar_delta_g": 100.1 * kcal, "stage_delta_g": {"vanish": 1.0}, "n_states": 3, "temperature_K": 298.15,
              "window_effective_samples": [400.0, 120.0, 300.0], "window_statistical_inefficiency": [6.0, 20.0, 8.0]}
    poor = dict(steady, first_half=[100.9 * kcal, 0.3 * kcal], second_half=[99.2 * kcal, 0.3 * kcal],
                min_overlap=0.02, overlap_neighbour=[0.2, 0.02], bar_delta_g=101.5 * kcal,
                window_effective_samples=[400.0, 12.0, 300.0], window_statistical_inefficiency=[6.0, 200.0, 8.0])
    assert quality_warnings({"analysis": steady}, cfg) == []
    assert leg_quality(steady, cfg) == quality_checks(steady)                  # the defaults are the same
    notes = quality_warnings({"analysis": poor}, cfg)
    assert len(notes) == 4
    # the overlap comes first, and it alone says NOT RELIABLE
    assert notes[0].startswith("NOT RELIABLE: the overlap between lambda windows 1 and 2 is 0.020, below 0.03")
    assert not any("NOT" in note for note in notes[1:])
    assert notes[1].startswith("few independent samples: lambda window 1 has 12 (its energy difference to the next "
                               "window is correlated over 200 iterations); 50 are recommended for MBAR")
    assert notes[2].startswith("the two halves of the production differ by 1.70 kcal/mol (2.9 kT, 4.0 standard errors "
                               "of the difference; first half 100.90, second half 99.20, all of it 100.00 ± 0.20). "
                               "More than 1 kT indicates poor convergence")
    assert "sampling.production_ns" in notes[1] and "sampling.production_ns" in notes[2]
    assert notes[3].startswith("BAR between neighbouring lambda windows gives 101.50 kcal/mol and MBAR 100.00; a "
                               "difference above 1 kcal/mol")
    staged = quality_warnings({"analysis": poor, "leg": "complex",
                               "lambda_windows": {"restrain": [0.0, 1.0], "vanish": [0.0, 0.5, 1.0]}}, cfg)
    assert staged[0].startswith("NOT RELIABLE: the overlap between lambda windows 1 and 2 (vanish stage, lambda 0 and 0.5) "
                                "is 0.020")
    assert "--leg complex --reschedule" in staged[0] and "--leg complex --production-ns" in staged[2]
    assert staged[1].startswith("few independent samples: lambda window 1 (vanish stage) has 12")
    # the overlap among the samples of the estimate, and what MBAR gets from the slowest window
    thinned = dict(steady, overlap_used_neighbour=[0.2, 0.012], window_kept_samples=[400, 12, 300],
                   window_inefficiency_neighbour=[1, -1, -1], window_effective_samples=[400.0, 12.4, 300.0],
                   window_statistical_inefficiency=[6.0, 200.0, 8.0])
    notes_thinned = quality_warnings({"analysis": thinned, "leg": "complex"}, cfg)
    assert len(notes_thinned) == 2
    assert notes_thinned[0].startswith(
        "NOT RELIABLE: among the samples the free energy is estimated from, the overlap between lambda windows 1 "
        "and 2 is 0.012, below 0.03; window 1 gives it 12 samples and window 2 gives 300.")
    assert "--leg complex --production-ns" in notes_thinned[0] and "--reschedule" in notes_thinned[0]
    assert notes_thinned[1].startswith(
        "few independent samples: lambda window 1 has 12 and gives MBAR 12 samples (its energy difference to the "
        "previous window is correlated over 200 iterations); 50 are recommended for MBAR")
    # the levels are settings
    lenient = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                     "analysis": {"min_overlap": 0.01, "min_effective_samples": 10,
                                                  "halves_tolerance_kT": 3.0, "estimators_tolerance_kcal_mol": 2.0}})
    assert quality_warnings({"analysis": poor}, lenient) == []
    # checks stored by an older version (no "halves") are made again from the analysis
    assert len(quality_warnings({"analysis": poor, "quality": {"overlap": {"ok": True}, "convergence": {"ok": True}}}, cfg)) == 4

    # a production too short to compare its halves: said so, and kept apart from halves that differ
    too_short = dict(steady, first_half=[float("nan")] * 2, second_half=[float("nan")] * 2,
                     convergence_note="lambda window 1 holds 2 independent sample(s) in a half of the production")
    verdict = quality_checks(too_short)["halves"]
    assert verdict["ok"] is None and "2 independent" in verdict["reason"]
    note = quality_warnings({"analysis": too_short, "leg": "solvent"}, cfg)
    assert len(note) == 1 and note[0].startswith("the two halves of the production could not be compared: lambda "
                                                 "window 1 holds 2")
    assert "--leg solvent --production-ns" in note[0]
    short_dir = tmp_path / "short"
    for name in ("complex", "solvent"):
        d = short_dir / "repeat_0" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": steady["delta_g"], "delta_g_mbar_error": 0.4, "restraint_release": 0.0,
            "lrc": 0.0, "backend": "openmm", "analysis": too_short if name == "solvent" else steady, "n_states": 15,
            "production_ns": 0.1, "notes": [], "quality": quality_checks(too_short if name == "solvent" else steady)}))
    out = combine_ligand(short_dir, "lig", 1)
    assert out["halves_not_compared"] == ["solvent leg"] and out["halves_differ"] == [] and out["few_samples"] == []

    for name, analysis in (("complex", poor), ("solvent", steady)):
        d = tmp_path / "repeat_0" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": analysis["delta_g"], "delta_g_mbar_error": 0.4, "restraint_release": 0.0,
            "lrc": 0.0, "backend": "openmm", "analysis": analysis, "n_states": 15, "production_ns": 5.0, "notes": [],
            "quality": quality_checks(analysis), "lambda_windows_added": {"vanish": 2} if name == "solvent" else None}))
    out = combine_ligand(tmp_path, "lig", 1)
    assert out["not_reliable"] == ["complex leg"] and out["halves_differ"] == ["complex leg"]
    assert out["few_samples"] == ["complex leg"] and out["estimators_disagree"] == ["complex leg"]
    assert out["halves_not_compared"] == []
    assert out["repeats"][0]["lambda_windows_added"] == {"complex": None, "solvent": {"vanish": 2}}

    # a result written before these checks: its old verdict on convergence is not carried over
    old_dir = tmp_path / "old"
    for name in ("complex", "solvent"):
        d = old_dir / "repeat_0" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": 400.0, "delta_g_mbar_error": 0.4, "restraint_release": 0.0, "lrc": 0.0,
            "backend": "openmm", "analysis": {"min_overlap": 0.02, "stage_delta_g": {}}, "n_states": 15, "production_ns": 5.0, "notes": [],
            "quality": {"overlap": {"ok": name == "solvent"}, "convergence": {"ok": False}, "estimators": {"ok": False}}}))
    out = combine_ligand(old_dir, "lig", 1)
    assert out["not_reliable"] == ["complex leg"] and out["halves_differ"] == [] and out["estimators_disagree"] == []


def test_a_leg_that_starts_anew_keeps_what_it_had(tmp_path):
    """--reschedule on a finished leg: nothing is deleted, pre-equilibration and restraint stay."""
    from cufep.workflow import archive_leg, window_stage

    leg = tmp_path / "repeat_0" / "complex"
    assert archive_leg(leg) is None                                   # nothing there yet
    (leg / "hrex").mkdir(parents=True)
    for name in ("hrex/samples.h5", "schedule.yaml", "start_configs.npz", "result.json", "timing.json", "equil.npz",
                 "restraint.json", "log.txt"):
        (leg / name).write_text(name)
    first = archive_leg(leg)
    assert first == leg / "previous_1"
    assert sorted(f.name for f in first.iterdir()) == ["hrex", "result.json", "schedule.yaml", "start_configs.npz",
                                                       "timing.json"]
    assert (first / "hrex" / "samples.h5").read_text() == "hrex/samples.h5"
    assert sorted(f.name for f in leg.iterdir()) == ["equil.npz", "log.txt", "previous_1", "restraint.json"]
    assert archive_leg(leg) is None                                   # only what must stay is left
    (leg / "schedule.yaml").write_text("new")
    assert archive_leg(leg) == leg / "previous_2" and (first / "schedule.yaml").read_text() == "schedule.yaml"

    windows = {"restrain": [0.0, 0.5, 1.0], "discharge": [0.0, 0.25, 0.5, 0.75, 1.0], "vanish": [0.0, 0.5, 1.0]}
    assert window_stage(windows, 0) == ("restrain", 0.0, 0.5)
    assert window_stage(windows, 2) == ("discharge", 0.0, 0.25)       # window 2 ends restrain and starts discharge
    assert window_stage(windows, 5) == ("discharge", 0.75, 1.0)
    assert window_stage(windows, 7) == ("vanish", 0.5, 1.0) and window_stage(windows, 8) is None


def _leg_result(dg, production_ns=5.0, leg="complex"):
    return {"leg": leg, "delta_g_decouple": dg, "delta_g_mbar_error": 0.4, "restraint_release": 0.0, "lrc": 0.0,
            "backend": "openmm", "analysis": {"stage_delta_g": {"vanish": dg}, "min_overlap": 0.2}, "n_states": 15,
            "production_ns": production_ns, "notes": []}


def test_legs_made_under_different_conditions_are_not_combined(tmp_path):
    """Review of PR 1: a solvent leg at 298.15 K and a complex leg at 315 K gave a binding free energy
    without a word, and so did two legs with different corrections for the dispersion tail."""
    from cufep.run import failed_legs

    def write(lig, repeat, leg, **fields):
        d = lig / f"repeat_{repeat}" / leg
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(dict(_leg_result(160.0 if leg == "complex" else 80.0, leg=leg),
                                                       **fields)))

    usual = {"temperature_K": 298.15, "lrc_method": "reweight"}
    whole = dict(usual, pressure_bar=1.0, prepared_systems_sha256="abc")       # all that a result of today records
    write(tmp_path / "a", 0, "solvent", **usual)
    write(tmp_path / "a", 0, "complex", temperature_K=315.0, lrc_method="reweight")
    out = combine_ligand(tmp_path / "a", "lig", 1)
    assert "dg_bind" not in out and out["n_repeats"] == 0
    assert out["inconsistent_legs"] == ["temperature (K): complex leg 315.0, solvent leg 298.15"]
    assert out["warnings"][0].startswith("NO BINDING FREE ENERGY: the legs were not made under the same conditions "
                                         "(temperature (K): complex leg 315.0, solvent leg 298.15)")
    # the command ends with an error because of it
    assert failed_legs([dict(out, failed_in_this_run=[])]) == [
        "lig: legs made under different conditions (temperature (K): complex leg 315.0, solvent leg 298.15)"]

    write(tmp_path / "b", 0, "solvent", **usual)
    write(tmp_path / "b", 0, "complex", temperature_K=298.15, lrc_method="isotropic")
    out = combine_ligand(tmp_path / "b", "lig", 1)
    assert "dg_bind" not in out
    assert out["inconsistent_legs"] == ["correction for the dispersion tail: complex leg isotropic, solvent leg reweight"]

    # pressure and the prepared systems are part of the cycle as well, where a result records them
    write(tmp_path / "p", 0, "solvent", pressure_bar=1.0, prepared_systems_sha256="abc", **usual)
    write(tmp_path / "p", 0, "complex", pressure_bar=2.0, prepared_systems_sha256="xyz", **usual)
    assert combine_ligand(tmp_path / "p", "lig", 1)["inconsistent_legs"] == [
        "pressure (bar): complex leg 2.0, solvent leg 1.0",
        "prepared systems (input, force field, solvation): complex leg xyz, solvent leg abc"]

    # two repeats, each consistent in itself, at two temperatures: they are not repeats of one calculation
    for repeat, temperature in ((0, 298.15), (1, 315.0)):
        for leg in ("complex", "solvent"):
            write(tmp_path / "c", repeat, leg, temperature_K=temperature, lrc_method="reweight")
    out = combine_ligand(tmp_path / "c", "lig", 2)
    assert "dg_bind" not in out and len(out["inconsistent_legs"]) == 1
    assert "complex leg of repeat 0 298.15" in out["inconsistent_legs"][0]
    assert "solvent leg of repeat 1 315.0" in out["inconsistent_legs"][0]

    # what may differ between the legs of one calculation does not stop the combination
    write(tmp_path / "d", 0, "solvent", backend="cupotential", timestep_fs=4.0, n_states=14, **whole)
    write(tmp_path / "d", 0, "complex", backend="openmm", timestep_fs=2.0, n_states=19, rest2={"scale": 1.5},
          origin_not_recorded=["the prepared systems", "the samples"], **whole)
    out = combine_ligand(tmp_path / "d", "lig", 1)
    assert out["dg_bind"] == pytest.approx((80.0 - 160.0) / 4.184) and out["inconsistent_legs"] == []
    # a leg that used files of an earlier version whose origin is not recorded is named in the ligand's result
    assert out["origin_not_recorded"] == ["complex leg: the prepared systems, the samples"]
    assert out["not_compared"] == [] and failed_legs([dict(out, failed_in_this_run=[])]) == []

    # results of an earlier version that lack the fields: combined, and the result says what was not compared
    write(tmp_path / "e", 0, "solvent")
    write(tmp_path / "e", 0, "complex", temperature_K=298.15)
    out = combine_ligand(tmp_path / "e", "lig", 1)
    assert "dg_bind" in out and out["inconsistent_legs"] == []
    assert out["not_compared"] == [
        "temperature (K): not recorded for the solvent leg",
        "pressure (bar): not recorded for the complex leg, the solvent leg",
        "prepared systems (input, force field, solvation): not recorded for the complex leg, the solvent leg",
        "correction for the dispersion tail: not recorded for the complex leg, the solvent leg"]
    assert sum("could not be compared between the legs" in w for w in out["warnings"]) == 4

    # before anything is sampled: a leg is not run under other conditions than the finished legs that
    # this run does not sample and that it would be combined with
    from cufep.config.schema import Config
    from cufep.workflow import WorkflowError, check_partner_legs

    def config(temperature):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                      "thermodynamics": {"temperature_K": temperature}})

    write(tmp_path / "f", 0, "solvent", pressure_bar=1.0, **usual)
    with pytest.raises(WorkflowError, match="temperature .K.: the solvent leg of repeat 0, which is finished and not "
                                            "part of this run, was made at 298.15; this run uses 315.0.*Nothing was run"):
        check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, config(315.0))
    check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, config(298.15))                 # the same conditions
    check_partner_legs(tmp_path / "f", "lig", {(0, "complex"), (0, "solvent")}, 1, config(315.0))  # its own samples decide
    with pytest.raises(WorkflowError, match="the solvent leg of repeat 0"):                        # a further repeat
        check_partner_legs(tmp_path / "f", "lig", {(1, "complex"), (1, "solvent")}, 2, config(315.0))


def test_a_leg_whose_analysis_gave_no_free_energy_is_a_failed_leg(tmp_path):
    """Windows without any overlap: the result file holds null, and the ligand gets no number from it."""
    from cufep import jsonio

    for name, dg in (("complex", float("nan")), ("solvent", 80.0)):
        d = tmp_path / "repeat_0" / name
        d.mkdir(parents=True)
        result = dict(_leg_result(dg, leg=name), delta_g_mbar_error=0.4)
        result["analysis"] = dict(result["analysis"], min_overlap_used=0.05)
        (d / "result.json").write_text(jsonio.dumps(result))
    assert json.loads((tmp_path / "repeat_0" / "complex" / "result.json").read_text())["delta_g_decouple"] is None
    out = combine_ligand(tmp_path, "lig", 1)
    assert "dg_bind" not in out and out["failed_legs"] == ["complex leg"]
    assert any("the analysis gave no free energy" in w for w in out["warnings"])
    (tmp_path / "repeat_0" / "complex" / "result.json").write_text(
        jsonio.dumps(dict(result, leg="complex", delta_g_decouple=160.0, restraint_release=-40.0)))
    out = combine_ligand(tmp_path, "lig", 1)
    assert out["failed_legs"] == [] and out["min_overlap_used"] == 0.05
    assert out["repeats"][0]["min_overlap_used"] == {"complex": 0.05, "solvent": 0.05}


def test_a_leg_that_fails_while_it_is_continued_does_not_count_with_its_old_result(tmp_path, monkeypatch):
    """Review of PR 1: a finished 5 ns leg whose continuation to 10 ns failed kept its old free energy."""
    import cufep.workflow as workflow
    from cufep.workflow import archive_leg, begin_leg, leg_state, read_result, run_leg_safe

    lig = tmp_path / "lig"
    for leg in ("complex", "solvent"):
        d = lig / "repeat_0" / leg
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(_leg_result(160.0 if leg == "complex" else 80.0, leg=leg)))
    assert "dg_bind" in combine_ligand(lig, "lig", 1)

    def fails(job):
        raise RuntimeError("the GPU was lost")

    monkeypatch.setattr(workflow, "run_leg", fails)
    job = LegJob(ligand="lig", leg="complex", repeat=0, lig_dir=str(lig), devices=[], config={}, production_ns=10.0)
    res = run_leg_safe(job)
    leg_dir = lig / "repeat_0" / "complex"
    assert "the GPU was lost" in res["error"] and res["previous_result"]["production_ns"] == 5.0
    assert not (leg_dir / "result.json").exists()
    assert json.loads((leg_dir / "result.previous.json").read_text())["delta_g_decouple"] == 160.0
    assert leg_state(leg_dir)[0] == "failed" and "error" in read_result(job)
    out = combine_ligand(lig, "lig", 1)
    assert "dg_bind" not in out and out["n_repeats"] == 0 and out["failed_legs"] == ["complex leg"]
    assert any("FAILED: RuntimeError: the GPU was lost" in w and "result.previous.json" in w for w in out["warnings"])

    # the leg runs again and finishes: the error is gone and the new result counts
    begin_leg(leg_dir)
    assert leg_state(leg_dir)[0] == "failed" and leg_state(leg_dir)[1]["interrupted"]      # started, nothing written yet
    (leg_dir / "result.json").write_text(json.dumps(_leg_result(164.184, production_ns=10.0)))
    out = combine_ligand(lig, "lig", 1)
    assert out["failed_legs"] == [] and out["repeats"][0]["production_ns"] == {"complex": 10.0, "solvent": 5.0}

    # a directory of an older version can hold a result and a later error: the later file counts
    (leg_dir / "error.json").write_text(json.dumps({"error": "later"}))
    import os
    os.utime(leg_dir / "result.json", (1.0, 1.0))
    assert leg_state(leg_dir)[0] == "failed"

    # started anew, the leg keeps the result it had under its own name in the archive
    begin_leg(leg_dir)
    (leg_dir / "schedule.yaml").write_text("windows")
    previous = archive_leg(leg_dir)
    assert (previous / "result.json").exists() and not (previous / "result.previous.json").exists()


def test_a_leg_whose_process_was_killed_is_a_failed_leg(tmp_path):
    """Review of PR 1: a job process killed before it wrote error.json left a leg that read as never run:
    the command ended with an error, but result, summary and report listed no failed leg."""
    from cufep.workflow import STARTED_FILE, begin_leg, leg_state, read_result, record_ended_job, write_leg_result

    lig = tmp_path / "lig"
    for repeat in (0, 1):
        for leg in ("complex", "solvent"):
            d = lig / f"repeat_{repeat}" / leg
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps(_leg_result(160.0 if leg == "complex" else 80.0, leg=leg)))
    job = LegJob(ligand="lig", leg="complex", repeat=1, lig_dir=str(lig), devices=[], config={}, production_ns=10.0)
    begin_leg(job.leg_dir, job)                                 # the scheduler starts the job ...
    assert (job.leg_dir / STARTED_FILE).exists() and not (job.leg_dir / "result.json").exists()
    # ... and nothing more is written: the process was killed, and so was the scheduler (or Ctrl-C)
    state, data = leg_state(job.leg_dir)
    assert state == "failed" and data["interrupted"] and (data["ligand"], data["leg"], data["repeat"]) == ("lig", "complex", 1)
    assert read_result(job)["error"].startswith("the run of this leg was interrupted before it wrote a result")
    out = combine_ligand(lig, "lig", 2)
    assert out["failed_legs"] == ["complex leg of repeat 1"] and out["n_repeats"] == 1 and "dg_bind" in out
    assert any("interrupted before it wrote a result" in w for w in out["warnings"])

    # the scheduler sees the dead process: what it knows goes into the files of the leg
    err = record_ended_job(job, -9)
    assert err["error"] == "the job process ended without a result (exit code -9, killed by signal 9)"
    assert err["previous_result"]["delta_g_decouple_kJ_mol"] == 160.0 and (err["leg"], err["repeat"]) == ("complex", 1)
    assert json.loads((job.leg_dir / "error.json").read_text())["error"] == err["error"]
    assert not (job.leg_dir / STARTED_FILE).exists()
    assert combine_ligand(lig, "lig", 2)["failed_legs"] == ["complex leg of repeat 1"]

    # a process that ended after it had written its result is not turned into a failure
    begin_leg(job.leg_dir, job)
    write_leg_result(job.leg_dir, _leg_result(161.0))
    assert record_ended_job(job, -9)["delta_g_decouple"] == 161.0 and not (job.leg_dir / "error.json").exists()
    assert leg_state(job.leg_dir)[0] == "finished" and not (job.leg_dir / STARTED_FILE).exists()
    assert not list(job.leg_dir.glob("*.tmp"))                  # written in one step

    # a result file that is cut off is a failed leg, not an exception while the legs are combined
    (job.leg_dir / "result.json").write_text('{"leg": "complex", "delta_g_dec')
    state, data = leg_state(job.leg_dir)
    assert state == "failed" and "cannot be read" in data["error"]
    assert combine_ligand(lig, "lig", 2)["failed_legs"] == ["complex leg of repeat 1"]

    # a leg that never started is still only missing
    assert leg_state(lig / "repeat_2" / "complex") == ("missing", None)


def test_commands_end_with_an_error_when_a_leg_failed(tmp_path, monkeypatch):
    import yaml
    from click.testing import CliRunner

    import cufep.run as driver
    from cufep.cli import main

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "output": {"dir": str(tmp_path / "out")}}))
    good = {"ligand": "a", "dg_bind": -9.0, "failed_in_this_run": []}
    bad = {"ligand": "b", "dg_bind": -8.0, "failed_in_this_run": ["complex leg"]}
    assert driver.failed_legs([good, bad]) == ["b: complex leg"] and driver.failed_legs([good]) == []

    monkeypatch.setattr(driver, "run", lambda cfg, **kwargs: [good])
    assert CliRunner().invoke(main, ["run", str(cfg)]).exit_code == 0
    monkeypatch.setattr(driver, "run", lambda cfg, **kwargs: [good, bad])
    result = CliRunner().invoke(main, ["run", str(cfg), "--ligand", "b", "--leg", "complex", "--production-ns", "10"])
    assert result.exit_code == 1 and "failed: b: complex leg" in result.output


def test_files_of_an_earlier_run_are_reused_only_for_the_same_input_and_settings(repo, tmp_path, ligands):
    """Review of PR 1: with membrane.enabled changed, the solvated system of the earlier run was used again."""
    import shutil

    from cufep.config.schema import Config
    from cufep.workflow import (WorkflowError, check_made_from, equil_contradictions, equil_in_membrane,
                                equil_made_with, prep_contradictions, prep_made_from, prepare_ligand,
                                restraint_contradictions, restraint_made_with, settings_differences)

    protein = tmp_path / "protein.pdb"
    shutil.copy(repo / "benchmarks/tyk2/inputs/tyk2_protein.pdb", protein)

    def config(**sections):
        return Config.model_validate({"system": {"protein": str(protein), "ligand": "l.sdf"},
                                      "output": {"dir": str(tmp_path / "out")}, **sections})

    cfg, lig = config(), ligands[0]
    made = prep_made_from(cfg, lig, False, "dodecahedron")
    assert settings_differences(made, prep_made_from(config(), lig, False, "dodecahedron")) == []
    assert settings_differences(made, prep_made_from(cfg, lig, True, "cube")) == [
        "membrane.enabled: False -> True", "membrane.lipid: not recorded -> POPC",
        "membrane.orientation: not recorded -> auto", "membrane.padding_nm: not recorded -> 1.5",
        "solvation.box_shape: dodecahedron -> cube"]
    assert settings_differences(made, prep_made_from(config(forcefield={"ligand_charges": "am1bcc-nagl"}), lig, False,
                                                     "dodecahedron")) == ["forcefield.ligand_charges: am1bcc -> am1bcc-nagl"]
    assert [d.split(":")[0] for d in settings_differences(made, prep_made_from(cfg, ligands[1], False, "dodecahedron"))] \
        == ["ligand"]
    protein.write_text(protein.read_text() + "REMARK changed\n")
    assert [d.split(":")[0] for d in settings_differences(made, prep_made_from(cfg, lig, False, "dodecahedron"))] \
        == ["protein_sha256"]
    made = prep_made_from(cfg, lig, False, "dodecahedron")

    # the preparation stops before it does anything when the prepared systems are of other settings
    lig_dir = tmp_path / "out" / lig.name
    (lig_dir / "prep").mkdir(parents=True)
    (lig_dir / "prep" / "prep.json").write_text(json.dumps({"ligand": lig.name, "made_from": made}))
    assert prepare_ligand(cfg, lig, None, lig_dir, False, "dodecahedron")["ligand"] == lig.name
    with pytest.raises(WorkflowError, match="the prepared systems of jmc_23 in .*prep cannot be used again"
                                            ".*membrane.enabled: False -> True.*needs a new output.dir"):
        prepare_ligand(config(membrane={"enabled": True}), lig, None, lig_dir, True, "cube")
    # a file of a version that kept no record of what it was made from, with what such a file holds:
    # refused where it says something else than this run, used otherwise
    notes = []
    older = {"ligand": lig.name, "n_atoms_complex": 50000, "n_atoms_solvent": 3000, "complex_barostat": "isotropic",
             "membrane": False, "box_shape": "dodecahedron", "solvent_box_shape": "cube", "membrane_info": {},
             "warnings": []}
    (lig_dir / "prep" / "prep.json").write_text(json.dumps(older))
    assert prep_contradictions(older, lig.name, True, "dodecahedron") == ["membrane: False -> True",
                                                                          "complex_barostat: isotropic -> membrane"]
    with pytest.raises(WorkflowError, match="did not record what it was made from, and what it does hold is not "
                                            "what this run asks for .membrane: False -> True; complex_barostat: "
                                            "isotropic -> membrane.*needs a new output.dir"):
        prepare_ligand(config(membrane={"enabled": True}), lig, None, lig_dir, True, "dodecahedron", log=notes.append)
    with pytest.raises(WorkflowError, match="box_shape: dodecahedron -> cube"):
        prepare_ligand(cfg, lig, None, lig_dir, False, "cube", log=notes.append)
    assert notes == []
    assert prepare_ligand(cfg, lig, None, lig_dir, False, "dodecahedron", log=notes.append)["ligand"] == lig.name
    assert "did not record what it was made from; nothing it holds contradicts this run" in notes[0]
    assert check_made_from("x", tmp_path, None, {}, log=notes.append) is False and check_made_from("x", tmp_path, {}, {})

    # the pre-equilibration of such a version: the time step in the file is the one that was used.
    # The area of the box is recorded for every system, the water molecules near the ligand only in a
    # bilayer (a soluble system was once taken for one because of its area)
    np.savez(tmp_path / "old_equil.npz", timestep_fs=2.0, simulated_ns=5.0, area_nm2=np.full(40, 61.7),
             pocket_waters=np.zeros(0), notes="[]")
    old_eq = np.load(tmp_path / "old_equil.npz")
    assert equil_in_membrane(old_eq) is False
    assert equil_contradictions(old_eq, cfg, "complex", False) == []                  # 2 fs used, 4 fs configured
    assert equil_contradictions(old_eq, config(dynamics={"timestep_fs": 1.0}), "complex", False) \
        == ["time step: 2 fs -> 1 fs"]
    assert equil_contradictions(old_eq, config(sampling={"pre_equilibration": {"complex": {"md_ns": 1.0}}}),
                                "complex", False) == ["length of the MD: 5 ns -> 1 ns"]
    assert equil_contradictions(old_eq, config(membrane={"enabled": True}), "complex", True) \
        == ["length of the MD: 5 ns -> 10 ns", "bilayer: no -> yes"]
    np.savez(tmp_path / "old_membrane.npz", timestep_fs=4.0, simulated_ns=15.0, area_nm2=np.full(40, 66.0),
             pocket_waters=np.full(40, 6.0))
    old_membrane = np.load(tmp_path / "old_membrane.npz")
    assert equil_in_membrane(old_membrane) is True
    assert equil_contradictions(old_membrane, config(membrane={"enabled": True}), "complex", True) == []   # extended
    assert "bilayer: yes -> no" in equil_contradictions(old_membrane, cfg, "complex", False)
    np.savez(tmp_path / "older.npz", timestep_fs=4.0, simulated_ns=5.0)          # a file that does not tell
    assert equil_in_membrane(np.load(tmp_path / "older.npz")) is None
    assert equil_contradictions(np.load(tmp_path / "older.npz"), cfg, "complex", False) == []
    # the restraint of such a version: all six force constants it holds (kJ/mol/nm^2 and kJ/mol/rad^2)
    angles = {key: 80.0 * 4.184 for key in ("k_theta_a", "k_theta_b", "k_phi_a", "k_phi_b", "k_phi_c")}
    old_restraint = {"restraint": {"host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6], "k_r": 10.0 * 4.184 * 100.0,
                                   **angles},
                     "info": {"selection": "auto", "force_constants": "openfe"}}
    assert restraint_contradictions(old_restraint, cfg) == []
    assert restraint_contradictions(old_restraint, config(restraints={"force_constants": "auto"})) \
        == ["restraints.force_constants: openfe -> auto"]
    assert restraint_contradictions(old_restraint, config(restraints={"k_r_kcal_per_A2": 20.0})) \
        == ["restraints.k_r_kcal_per_A2: 10 -> 20"]
    # review of PR 1: the distance constant kept and the angle constant changed from 80 to 20 went through
    assert restraint_contradictions(old_restraint, config(restraints={"k_angle_kcal_per_rad2": 20.0})) == [
        "restraints.k_angle_kcal_per_rad2: k_theta_a 80, k_theta_b 80, k_phi_a 80, k_phi_b 80, k_phi_c 80 -> 20"]
    one_differs = {"restraint": dict(old_restraint["restraint"], k_phi_b=40.0 * 4.184), "info": old_restraint["info"]}
    assert restraint_contradictions(one_differs, cfg) == ["restraints.k_angle_kcal_per_rad2: k_phi_b 40 -> 80"]
    only_distance = {"restraint": {"k_r": 10.0 * 4.184 * 100.0}, "info": old_restraint["info"]}
    assert restraint_contradictions(only_distance, config(restraints={"k_angle_kcal_per_rad2": 20.0})) == []
    assert restraint_contradictions(old_restraint, config(restraints={
        "selection": "manual", "host_atoms": [1, 2, 9], "guest_atoms": [4, 5, 6]})) \
        == ["restraints.selection: auto -> manual", "restraints.host_atoms: [1, 2, 3] -> [1, 2, 9]"]
    # with manual selection the constants are the configured ones whatever force_constants says
    manual = {"restraint": dict(old_restraint["restraint"]), "info": {"selection": "manual"}}
    assert restraint_contradictions(manual, config(restraints={
        "selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6], "force_constants": "auto",
        "k_angle_kcal_per_rad2": 20.0})) == [
        "restraints.k_angle_kcal_per_rad2: k_theta_a 80, k_theta_b 80, k_phi_a 80, k_phi_b 80, k_phi_c 80 -> 20"]

    # pre-equilibration and restraint of a leg
    eq = equil_made_with(cfg, "complex", False, 7)
    longer = config(sampling={"pre_equilibration": {"complex": {"md_ns": 1.0}}})
    assert settings_differences(eq, equil_made_with(longer, "complex", False, 7)) == ["pre_equilibration.md_ns: 5.0 -> 1.0"]
    assert settings_differences(eq, equil_made_with(longer, "solvent", False, 7))[0].startswith("pre_equilibration.md_ns")
    assert settings_differences(eq, equil_made_with(config(thermodynamics={"temperature_K": 310.0}), "complex", False, 7)) \
        == ["temperature_K: 298.15 -> 310.0"]
    in_membrane = equil_made_with(config(membrane={"enabled": True, "equilibration_ns": 20.0}), "complex", True, 7)
    assert settings_differences(equil_made_with(config(membrane={"enabled": True}), "complex", True, 7), in_membrane) \
        == ["membrane_equilibration_ns: 10.0 -> 20.0"]
    with pytest.raises(WorkflowError, match="the pre-equilibration of this leg in .*equil.npz cannot be used again"):
        check_made_from("the pre-equilibration of this leg", tmp_path / "equil.npz", eq, in_membrane)
    stiff = restraint_made_with(config(restraints={"k_r_kcal_per_A2": 20.0, "pose_rmsd_warning_nm": 0.5}))
    assert settings_differences(restraint_made_with(cfg), stiff) == ["k_r_kcal_per_A2: 10.0 -> 20.0"]


def test_a_run_that_is_refused_leaves_the_record_of_the_directory_as_it_was(tmp_path, monkeypatch, ligands):
    """Review of PR 1: a run stopped because its configuration did not fit the directory had already
    overwritten resolved_config.yaml, which then described a configuration that made none of the results."""
    import cufep.run as driver
    from cufep.config.auto import AutoReport
    from cufep.config.schema import Config, read_settings
    from cufep.sampling.storage import ConditionsError, LegStorage
    from cufep.workflow import WorkflowError, check_leg_reuse, prep_made_from, samples_settings

    lig = ligands[0]
    out = tmp_path / "out"

    def config(**sections):
        return Config.model_validate({"system": {"protein": str(tmp_path / "p.pdb"), "ligand": "l.sdf"},
                                      "output": {"dir": str(out)}, "resources": {"gpus": []},
                                      "solvation": {"box_shape": "dodecahedron"}, **sections})

    class Protein:
        def write_pdb(self, path):
            Path(path).write_text("ATOM")

    from pathlib import Path
    monkeypatch.setattr(driver, "plan", lambda cfg, log: (cfg.model_copy(deep=True), AutoReport(), Protein(), [lig]))
    monkeypatch.setattr(driver, "run_jobs", lambda jobs, *args, **kwargs: [])
    monkeypatch.setattr(driver, "write_timing", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "refresh_leg_result", lambda *args, **kwargs: "current")
    monkeypatch.setattr(driver, "plot_repeat", lambda *args, **kwargs: None)
    log = []

    # an earlier run left: its configuration, the prepared systems, and samples of the solvent leg
    first = config()
    lig_dir = out / lig.name
    (lig_dir / "prep").mkdir(parents=True)
    (lig_dir / "prep" / "prep.json").write_text(json.dumps({
        "ligand": lig.name, "complex_barostat": "isotropic", "membrane": False, "box_shape": "dodecahedron",
        "made_from": prep_made_from(first, lig, False, "dodecahedron")}))
    hrex = lig_dir / "repeat_0" / "solvent" / "hrex"
    storage = LegStorage(hrex, 2, 2)
    storage.check_conditions({"temperature_K": 298.15, "pressure_bar": 1.0, "friction_per_ps": 1.0,
                              "settings": samples_settings(first)})
    for _ in range(6):
        storage.append(np.zeros((2, 2)), np.arange(2), np.ones(2))
    storage.close()
    driver._run(first, out, False, log.append)
    record = (out / "resolved_config.yaml").read_bytes()
    assert (out / "protein_prepared.pdb").exists() and not list(out.glob("resolved_config.previous_*"))

    # another soft-core: refused before anything is written
    other = config(alchemy={"softcore_alpha": 0.8})
    with pytest.raises(ConditionsError, match="alchemy.softcore_alpha: 0.5 -> 0.8.*They are left as they are"):
        driver._run(other, out, False, log.append)
    assert (out / "resolved_config.yaml").read_bytes() == record
    assert read_settings(out / "resolved_config.refused.yaml")["alchemy"]["softcore_alpha"] == 0.8
    assert any(line.startswith("refused: ") for line in log) and not list(out.glob("resolved_config.previous_*"))
    # another membrane setting: the prepared systems refuse, the record stays
    with pytest.raises(WorkflowError, match="membrane.enabled: False -> True"):
        driver._run(config(membrane={"enabled": True}), out, False, log.append)
    assert (out / "resolved_config.yaml").read_bytes() == record

    # a run that differs only in its resources is the same calculation: no history
    driver._run(config(resources={"gpus": [], "cpu_threads_per_worker": 3}), out, False, log.append)
    assert not list(out.glob("resolved_config.previous_*")) and not (out / "resolved_config.refused.yaml").exists()
    assert read_settings(out / "resolved_config.yaml")["resources"]["cpu_threads_per_worker"] == 3
    # a longer production is a change of the calculation: the earlier record is kept, once
    driver._run(config(sampling={"production_ns": 20.0}), out, False, log.append)
    assert [f.name for f in out.glob("resolved_config.previous_*")] == ["resolved_config.previous_1.yaml"]
    assert read_settings(out / "resolved_config.previous_1.yaml")["sampling"]["production_ns"] == 10.0
    assert read_settings(out / "resolved_config.yaml")["sampling"]["production_ns"] == 20.0

    # the checks of a leg, from its files alone
    prep_info = json.loads((lig_dir / "prep" / "prep.json").read_text())
    check_leg_reuse(first, lig_dir, "solvent", 0, prep_info, 10.0)
    with pytest.raises(ConditionsError, match="temperature_K: 298.15 -> 310.0"):
        check_leg_reuse(config(thermodynamics={"temperature_K": 310.0}), lig_dir, "solvent", 0, prep_info, 10.0)
    with pytest.raises(ConditionsError, match="rest2.scale: 1.5 -> 2.0"):
        check_leg_reuse(config(sampling={"rest2": {"scale": 2.0}}), lig_dir, "solvent", 0, prep_info, 10.0)
    check_leg_reuse(other, lig_dir, "solvent", 0, prep_info, 10.0, reschedule=True)      # the leg starts anew
    check_leg_reuse(other, lig_dir, "solvent", 1, prep_info, 10.0)                       # a leg without files
    # samples of a version that did not record their conditions: enough of them are analysed, too few
    # are not continued
    import h5py
    with h5py.File(hrex / "samples.h5", "a") as h5:
        del h5.attrs["conditions"]
    with pytest.raises(ConditionsError, match="did not record the conditions.*this run asks for 4400.*not continued"):
        check_leg_reuse(first, lig_dir, "solvent", 0, prep_info, 10.0)
    check_leg_reuse(config(sampling={"equilibration_ns": 0.0025, "production_ns": 0.005}), lig_dir, "solvent", 0,
                    prep_info, 0.005)


def test_a_run_can_be_restricted_to_ligands_and_legs():
    from cufep.run import select_jobs
    from cufep.workflow import WorkflowError

    names = ["a", "b"]
    assert select_jobs(names, 1) == [("a", 0, "complex"), ("a", 0, "solvent"), ("b", 0, "complex"), ("b", 0, "solvent")]
    assert select_jobs(names, 2, only_ligands=("b",), only_legs=("complex",)) == [("b", 0, "complex"), ("b", 1, "complex")]
    assert select_jobs(names, 1, only_legs=["solvent"]) == [("a", 0, "solvent"), ("b", 0, "solvent")]
    with pytest.raises(WorkflowError, match="not among the ligands"):
        select_jobs(names, 1, only_ligands=["c"])
    with pytest.raises(WorkflowError, match="neither complex nor solvent"):
        select_jobs(names, 1, only_legs=["bound"])


def test_analysis_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from cufep.analysis.mbar import quality_checks
    from cufep.analysis.plots import plot_repeat

    assert plot_repeat(tmp_path) is None                              # no finished leg
    analysis = {"delta_g": 400.0, "delta_g_error": 0.8, "min_overlap": 0.02, "overlap_neighbour": [0.2, 0.02],
                "overlap_matrix": [[0.8, 0.2, 0.0], [0.2, 0.78, 0.02], [0.0, 0.02, 0.98]],
                "forward": [[0.5, 402.0, 1.1], [1.0, 400.0, 0.8]], "reverse": [[0.5, 398.5, 1.1], [1.0, 400.0, 0.8]],
                "first_half": [402.0, 1.1], "second_half": [398.5, 1.1], "bar_delta_g": 400.2,
                "window_effective_samples": [300.0, 40.0, 250.0], "window_statistical_inefficiency": [8.0, 60.0, 9.6],
                "blocks": [[0.0, 0.5, 402.0, 1.1], [0.5, 1.0, 398.5, 1.1]]}
    d = tmp_path / "solvent"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"ligand": "lig", "repeat": 0, "leg": "solvent", "analysis": analysis,
                                               "quality": quality_checks(analysis)}))
    out = plot_repeat(tmp_path)
    assert out == tmp_path / "analysis.png" and out.stat().st_size > 10000


@pytest.mark.parametrize("shape,ratio", [("cube", 1.0), ("dodecahedron", 2 ** -0.5)])
def test_solvation_box_meets_the_minimum_height(shape, ratio):
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.solvate import SolvationSpec, solvate

    ff = app.ForceField("amber14/tip3p.xml")

    def box_for(points_nm, min_edge):
        seed = app.Modeller(app.Topology(), [])
        seed.addSolvent(ff, boxSize=(2.0, 2.0, 2.0) * u.nanometer)
        waters = list(seed.topology.residues())
        keep = app.Modeller(seed.topology, seed.positions)
        keep.delete(waters[1:])                                   # one water molecule as the "solute"
        pos = np.asarray(keep.positions.value_in_unit(u.nanometer))
        shift = np.asarray(points_nm) - pos.mean(axis=0)
        keep.positions = (pos + shift) * u.nanometer
        solvate(keep, ff, SolvationSpec(padding_nm=1.0, box_shape=shape, ionic_strength_M=0.0, min_edge_nm=min_edge))
        vec = keep.topology.getPeriodicBoxVectors().value_in_unit(u.nanometer)
        return float(vec[0][0]), float(vec[2][2])

    width, height = box_for([0.0, 0.0, 0.0], 2.9)
    assert height >= 2.9 - 1e-6 and height == pytest.approx(width * ratio, rel=1e-6)
    assert width < 2.9 / ratio + 0.2                               # no larger than needed
    width_free, _ = box_for([0.0, 0.0, 0.0], 0.0)
    assert width_free < width                                      # the rule, not the padding, set the size


def test_a_second_run_in_the_same_output_directory_is_refused(tmp_path):
    from cufep.run import lock_output_directory
    from cufep.workflow import WorkflowError

    first = lock_output_directory(tmp_path)
    with pytest.raises(WorkflowError, match="another cufep run is using the output directory"):
        lock_output_directory(tmp_path)
    first.close()
    lock_output_directory(tmp_path).close()                # free again once the first run has ended
