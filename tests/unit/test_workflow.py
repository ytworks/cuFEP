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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_ligands_load_with_names_and_experimental_values(ligands):
    assert len(ligands) == 16
    lig = next(l for l in ligands if l.name == "ejm_31")
    assert lig.n_atoms == 32 and lig.formal_charge == 0
    assert float(lig.properties["r_exp_dg"]) == pytest.approx(-9.54)


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_the_place_of_the_ligand_is_not_judged(ligands):
    """The owner, 2026-11-20: no verdict with a level of cuFEP's own. Earlier versions stopped on a ligand with less
    than half of its heavy atoms near the protein (a pose 40 A away); the place of the ligand is now the user's."""
    lig = ligands[0]
    check_ligand(lig)
    from openff.units import unit
    moved = load_ligands(lig.source)[0]
    moved.offmol._conformers = [moved.offmol.conformers[0] + np.array([40.0, 0.0, 0.0]) * unit.angstrom]
    check_ligand(moved)                                       # 40 A away: not stopped


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_charged_ligand_stops(repo):
    ligs = load_ligands(repo / "benchmarks/a2a/inputs/a2a_hip278_ligands.sdf")
    charged = next(l for l in ligs if l.formal_charge != 0)
    with pytest.raises(LigandError, match="formal charge"):
        check_ligand(charged)


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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
    # no MPS and more jobs than GPUs: a state the scheduler never makes (a job starts only when its first worker
    # fits, tests/unit/test_scheduler.py); the strict allocation refuses it, the plain one still gives each a worker
    assert allocate(gpus, crowded, strict=True) is None
    plan = allocate(gpus, crowded)
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
    # the checks are made again from the numbers of every leg, whatever version wrote it: a leg of before the
    # checks with an overlap of 0.02 is below the literature's 0.03; one without numbers of samples is not judged
    assert out["not_reliable"] == ["complex leg of repeat 1"] and out["few_samples"] == []
    assert not any(key in out for key in ("halves_differ", "halves_not_compared", "halves_below_level",
                                          "estimators_disagree"))
    # results of before the standard error from time blocks hold the one of MBAR alone: it is what is combined
    assert out["dg_bind_error"] == out["dg_bind_mbar_error"] == pytest.approx(np.hypot(0.4, 0.4) / 4.184)
    assert (
        out["dg_bind_error_from_time_blocks"] is False and out["repeats"][0]["error"] == out["repeats"][0]["mbar_error"]
    )

    # the standard error of a leg is the one from time blocks; that of MBAR is kept next to it
    def with_blocks(r, name, error, n_blocks):
        path = tmp_path / f"repeat_{r}" / name / "result.json"
        data = json.loads(path.read_text())
        data["delta_g_error"] = error
        data["analysis"]["delta_g_error_n_blocks"] = n_blocks
        path.write_text(json.dumps(data))
    for r in range(2):
        with_blocks(r, "complex", 1.2, 5)
        with_blocks(r, "solvent", 0.5, 20)
    out = combine_ligand(tmp_path, "lig", 2)
    assert (
        out["dg_bind_error"] == pytest.approx(np.hypot(1.2, 0.5) / 4.184)
        and out["dg_bind_error_from_time_blocks"] is True
    )
    assert out["dg_bind_mbar_error"] == pytest.approx(np.hypot(0.4, 0.4) / 4.184)
    assert (
        out["repeats"][1]["error"] == pytest.approx(1.3 / 4.184) and out["repeats"][1]["error_from_time_blocks"] is True
    )
    # one leg too short for blocks reports the error of MBAR: the binding free energy says "standard error" alone
    with_blocks(1, "solvent", 0.4, None)
    out = combine_ligand(tmp_path, "lig", 2)
    assert out["dg_bind_error_from_time_blocks"] is False
    assert out["dg_bind_error"] == pytest.approx(np.sqrt((1.3 ** 2 + np.hypot(1.2, 0.4) ** 2) / 2) / 4.184)


def test_checks_of_the_analysis_reach_the_leg_notes_and_the_ligand_result(tmp_path):
    from cufep.analysis.mbar import quality_checks
    from cufep.config.schema import Config
    from cufep.workflow import leg_quality, overlap_below_level, quality_warnings

    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}})
    kcal = 4.184
    # kT is 0.59 kcal/mol: the halves of the steady leg differ by 0.5 kT, those of the other by 2.9 kT
    steady = {
        "delta_g": 100.0 * kcal,
        "delta_g_error": 0.2 * kcal,
        "min_overlap": 0.11,
        "overlap_neighbour": [0.2, 0.11],
        "first_half": [100.2 * kcal, 0.3 * kcal],
        "second_half": [99.9 * kcal, 0.3 * kcal],
        "bar_delta_g": 100.1 * kcal,
        "stage_delta_g": {"vanish": 1.0},
        "n_states": 3,
        "temperature_K": 298.15,
        "window_effective_samples": [400.0, 120.0, 300.0],
        "window_statistical_inefficiency": [6.0, 20.0, 8.0],
    }
    poor = dict(steady, first_half=[100.9 * kcal, 0.3 * kcal], second_half=[99.2 * kcal, 0.3 * kcal],
                min_overlap=0.02, overlap_neighbour=[0.2, 0.02], bar_delta_g=101.5 * kcal,
                window_effective_samples=[400.0, 12.0, 300.0], window_statistical_inefficiency=[6.0, 200.0, 8.0])
    assert quality_warnings({"analysis": steady}, cfg) == []
    assert leg_quality(steady, cfg) == quality_checks(steady)                  # the defaults are the same
    notes = quality_warnings({"analysis": poor}, cfg)
    # the owner, 2026-11-20: the only verdicts are the two checks with the literature's levels. Halves 2.9 kT apart
    # and BAR 1.5 kcal/mol from MBAR give no message (they were checks of cuFEP's own)
    assert len(notes) == 2
    assert set(quality_checks(poor)) == {"overlap", "samples", "uncertainty"}
    assert not any("halves" in note or "BAR between" in note or "CONVERGED" in note for note in notes)
    # the overlap comes first, and it alone says NOT RELIABLE
    assert notes[0].startswith("NOT RELIABLE: the overlap between lambda windows 1 and 2 is 0.020, below 0.03")
    assert not any("NOT" in note for note in notes[1:])
    assert notes[1].startswith(
        "few independent samples: lambda window 1 has 12 (its energy difference to the next "
        "window is correlated over 200 iterations); 50 are recommended for MBAR (Mey et al. 2020)"
    )
    assert "sampling.production_ns" in notes[1]
    staged = quality_warnings({"analysis": poor, "leg": "complex",
                               "lambda_windows": {"restrain": [0.0, 1.0], "vanish": [0.0, 0.5, 1.0]}}, cfg)
    assert staged[0].startswith(
        "NOT RELIABLE: the overlap between lambda windows 1 and 2 (vanish stage, lambda 0 and 0.5) is 0.020"
    )
    assert "--leg complex --reschedule" in staged[0] and "--leg complex --production-ns" in staged[1]
    assert staged[1].startswith("few independent samples: lambda window 1 (vanish stage) has 12")
    # where the leg is below the level, as the summary of the ligand keeps it for the benchmark report
    assert overlap_below_level({"analysis": poor, "lambda_windows": {"restrain": [0.0, 1.0],
                                                                     "vanish": [0.0, 0.5, 1.0]}}, cfg) == [
        {"of": "all iterations", "overlap": 0.02, "windows": [1, 2], "stage": "vanish", "lambda": [0.0, 0.5]}]
    assert overlap_below_level({"analysis": steady}, cfg) == []
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
    assert overlap_below_level({"analysis": thinned}, cfg) == [
        {"of": "the samples of the estimate", "overlap": 0.012, "windows": [1, 2], "stage": None, "lambda": None}]
    assert notes_thinned[1].startswith(
        "few independent samples: lambda window 1 has 12 and gives MBAR 12 samples (its energy difference to the "
        "previous window is correlated over 200 iterations); 50 are recommended for MBAR (Mey et al. 2020)")
    # the number of samples is the literature's: production / g of the window, and not more than MBAR gets. A
    # standard error from time blocks 3 times the one of MBAR no longer reduces it (that reduction was cuFEP's own)
    travelling = dict(steady, window_kept_samples=[400, 90, 300], window_effective_samples=[400.0, 120.0, 300.0],
                      delta_g_error=0.6 * kcal, delta_g_error_mbar=0.2 * kcal, delta_g_error_n_blocks=5)
    verdict = quality_checks(travelling)["samples"]
    assert verdict["ok"] is True and verdict["kept"] == 90 and verdict["fewest"] == 120.0 and verdict["counted"] == 90.0
    assert "error_ratio" not in verdict and "independent" not in verdict
    assert quality_warnings({"analysis": travelling}, cfg) == []
    # the levels are settings
    lenient = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                     "analysis": {"min_overlap": 0.01, "min_effective_samples": 10}})
    assert quality_warnings({"analysis": poor}, lenient) == []
    assert overlap_below_level({"analysis": poor}, lenient) == []
    # checks stored by an older version are not used: they are made again from the analysis
    assert len(quality_warnings({"analysis": poor, "quality": {"overlap": {"ok": True}, "halves": {"ok": False},
                                                              "estimators": {"ok": False}}}, cfg)) == 2
    # a production too short to compare its halves has no message about them
    too_short = dict(steady, first_half=[float("nan")] * 2, second_half=[float("nan")] * 2,
                     convergence_note="lambda window 1 holds 2 independent sample(s) in a half of the production")
    assert quality_warnings({"analysis": too_short, "leg": "solvent"}, cfg) == []

    for name, analysis in (("complex", poor), ("solvent", steady)):
        d = tmp_path / "repeat_0" / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "leg": name, "delta_g_decouple": analysis["delta_g"], "delta_g_mbar_error": 0.4, "restraint_release": 0.0,
            "lrc": 0.0, "backend": "openmm", "analysis": analysis, "n_states": 15, "production_ns": 5.0,
            # notes of an earlier version, with its own verdicts: made again, and those of removed checks are gone
            "notes": ["a note about the pose", "the two halves of the production differ by 1.70 kcal/mol (2.9 kT)",
                      "BAR between neighbouring lambda windows gives 101.50 kcal/mol and MBAR 100.00"],
            "quality": {"overlap": {"ok": True}, "halves": {"ok": False}, "estimators": {"ok": False}},
            "lambda_windows_added": {"vanish": 2} if name == "solvent" else None}))
    out = combine_ligand(tmp_path, "lig", 1)
    assert out["not_reliable"] == ["complex leg"] and out["few_samples"] == ["complex leg"]
    assert out["overlap_below_level"] == [{"leg": "complex leg", "of": "all iterations", "overlap": 0.02,
                                           "windows": [1, 2], "stage": None, "lambda": None}]
    assert not any(key in out for key in ("halves_differ", "halves_not_compared", "halves_below_level",
                                          "estimators_disagree"))
    assert out["repeats"][0]["lambda_windows_added"] == {"complex": None, "solvent": {"vanish": 2}}
    assert "repeat 0 complex: a note about the pose" in out["warnings"]
    assert not any("halves" in w or "BAR between" in w for w in out["warnings"])
    assert any(w.startswith("repeat 0 complex: NOT RELIABLE") for w in out["warnings"])


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
    assert sorted(f.name for f in first.iterdir()) == ["equil.npz", "hrex", "restraint.json", "result.json",
                                                       "schedule.yaml", "start_configs.npz", "timing.json"]
    # the pre-equilibration and the restraint stay with the leg; the archive has copies of them
    assert (first / "equil.npz").read_text() == "equil.npz" == (leg / "equil.npz").read_text()
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
    # all that a result of today records
    whole = dict(usual, pressure_bar=1.0, prepared_systems_sha256="abc", lrc_long_cutoff_nm=2.0)
    write(tmp_path / "a", 0, "solvent", **usual)
    write(tmp_path / "a", 0, "complex", temperature_K=315.0, lrc_method="reweight")
    out = combine_ligand(tmp_path / "a", "lig", 1)
    assert "dg_bind" not in out and out["n_repeats"] == 0
    assert out["inconsistent_legs"] == ["temperature (K): complex leg 315.0, solvent leg 298.15"]
    assert out["warnings"][0].startswith("NO BINDING FREE ENERGY: the legs were not made under the same conditions "
                                         "(temperature (K): complex leg 315.0, solvent leg 298.15)")
    # the command ends with an error because of it
    assert failed_legs([dict(out, selected_legs=[[0, "solvent"]])]) == [
        "lig: legs made under different conditions (temperature (K): complex leg 315.0, solvent leg 298.15)"]
    assert failed_legs([dict(out, selected_legs=[])]) == []                # a ligand this command did not run

    write(tmp_path / "b", 0, "solvent", **usual)
    write(tmp_path / "b", 0, "complex", temperature_K=298.15, lrc_method="isotropic")
    out = combine_ligand(tmp_path / "b", "lig", 1)
    assert "dg_bind" not in out
    assert out["inconsistent_legs"] == [
        "correction for the dispersion tail: complex leg isotropic, solvent leg reweight"
    ]

    # pressure and the prepared systems are part of the cycle as well, where a result records them
    write(tmp_path / "p", 0, "solvent", pressure_bar=1.0, prepared_systems_sha256="abc", **usual)
    write(tmp_path / "p", 0, "complex", pressure_bar=2.0, prepared_systems_sha256="xyz", **usual)
    assert combine_ligand(tmp_path / "p", "lig", 1)["inconsistent_legs"] == [
        "pressure (bar): complex leg 2.0, solvent leg 1.0",
        "prepared systems (input, force field, solvation): complex leg xyz, solvent leg abc"]

    # review of PR 1: two legs corrected by reweighting whose tail energies were recorded to different
    # outer radii were combined (the stored energies cannot be turned into those of another radius)
    write(tmp_path / "t", 0, "solvent", **dict(whole, lrc_long_cutoff_nm=1.5))
    write(tmp_path / "t", 0, "complex", **whole)
    out = combine_ligand(tmp_path / "t", "lig", 1)
    assert "dg_bind" not in out
    assert out["inconsistent_legs"] == ["outer radius of the tail correction (nm): complex leg 2.0, solvent leg 1.5"]
    # the isotropic correction does not depend on that radius, and neither does none
    for method in ("isotropic", "none"):
        write(tmp_path / method, 0, "solvent", **dict(whole, lrc_method=method, lrc_long_cutoff_nm=1.5))
        write(tmp_path / method, 0, "complex", **dict(whole, lrc_method=method))
        out = combine_ligand(tmp_path / method, "lig", 1)
        assert "dg_bind" in out and out["inconsistent_legs"] == [] and out["not_compared"] == []
    # a reweighted leg whose result does not record the radius is named
    write(tmp_path / "u", 0, "solvent", **dict(whole, lrc_long_cutoff_nm=None))
    write(tmp_path / "u", 0, "complex", **whole)
    assert combine_ligand(tmp_path / "u", "lig", 1)["not_compared"] == [
        "outer radius of the tail correction (nm): not recorded for the solvent leg"]

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
    assert out["not_compared"] == [] and failed_legs([dict(out, selected_legs=[[0, "solvent"]])]) == []

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
    with pytest.raises(WorkflowError, match="temperature .K.: the legs this run samples 315.0, the solvent leg of "
                                            "repeat 0 .finished, not part of this run. 298.15.*Nothing was run"):
        check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, config(315.0))
    check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, config(298.15))  # the same conditions
    check_partner_legs(
        tmp_path / "f", "lig", {(0, "complex"), (0, "solvent")}, 1, config(315.0)
    )  # its own samples decide
    with pytest.raises(WorkflowError, match="the solvent leg of repeat 0"):                        # a further repeat
        check_partner_legs(tmp_path / "f", "lig", {(1, "complex"), (1, "solvent")}, 2, config(315.0))
    # the outer radius of the tail: what the samples of the finished leg hold against what this run asks for
    from cufep.sampling.storage import LegStorage
    storage = LegStorage(tmp_path / "f" / "repeat_0" / "solvent" / "hrex", 2, 2)
    storage.check_conditions({"temperature_K": 298.15, "lrc": {"cutoff_nm": 0.9, "long_cutoff_nm": 1.0}})
    storage.set_meta(lrc_s6=1.0, lrc_s12=1.0, lrc_cutoff_nm=0.9, n_equilibration_iterations=0)
    for it in range(8):
        storage.append(np.zeros((2, 2)), np.arange(2), np.ones(2))
        storage.append_record(it, {"lrc_du": -1.0})
    storage.close()
    with pytest.raises(
        WorkflowError,
        match="outer radius of the tail correction .nm.: the legs this run samples 2.0, "
        "the solvent leg of repeat 0 .finished, not part of this run. 1.0.*the "
        "solvent leg of repeat 0 .finished, not part of this run. has samples that "
        "cannot be analysed with the settings of this run: their tail energies were "
        "recorded to an outer radius of 1 nm.*Nothing was run",
    ):
        check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, config(298.15))
    shorter = Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc_cutoff_nm": 1.0}}
    )
    check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, shorter)  # the same radius
    isotropic = Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc": "isotropic"}}
    )
    check_partner_legs(tmp_path / "f", "lig", {(0, "complex")}, 1, isotropic)                       # does not use it


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
    assert leg_state(leg_dir)[0] == "failed" and leg_state(leg_dir)[1]["interrupted"]  # started, nothing written yet
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
    assert (
        state == "failed"
        and data["interrupted"]
        and (data["ligand"], data["leg"], data["repeat"]) == ("lig", "complex", 1)
    )
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
    cfg.write_text(
        yaml.safe_dump({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "output": {"dir": str(tmp_path / "out")}})
    )
    good = {"ligand": "a", "dg_bind": -9.0, "selected_legs": [[0, "complex"], [0, "solvent"]], "failed_leg_ids": []}
    bad = {"ligand": "b", "dg_bind": -8.0, "selected_legs": [[0, "complex"]], "failed_leg_ids": [[0, "complex"]]}
    assert driver.failed_legs([good, bad]) == ["b: complex leg failed or has no free energy"]
    assert driver.failed_legs([good]) == []

    monkeypatch.setattr(driver, "run", lambda cfg, **kwargs: [good])
    assert CliRunner().invoke(main, ["run", str(cfg)]).exit_code == 0
    monkeypatch.setattr(driver, "run", lambda cfg, **kwargs: [good, bad])
    result = CliRunner().invoke(main, ["run", str(cfg), "--ligand", "b", "--leg", "complex", "--production-ns", "10"])
    assert result.exit_code == 1 and "failed: b: complex leg failed or has no free energy" in result.output


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_files_of_an_earlier_run_are_reused_only_for_the_same_input_and_settings(repo, tmp_path, ligands):
    """Review of PR 1: with membrane.enabled changed, the solvated system of the earlier run was used again."""
    import shutil

    from cufep.config.schema import Config
    from cufep.workflow import (
        WorkflowError,
        check_made_from,
        equil_contradictions,
        equil_in_membrane,
        equil_made_with,
        prep_contradictions,
        prep_made_from,
        prepare_ligand,
        restraint_contradictions,
        restraint_made_with,
        settings_differences,
    )

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
    assert settings_differences(
        made, prep_made_from(config(forcefield={"ligand_charges": "am1bcc-nagl"}), lig, False, "dodecahedron")
    ) == ["forcefield.ligand_charges: am1bcc -> am1bcc-nagl"]
    assert [
        d.split(":")[0] for d in settings_differences(made, prep_made_from(cfg, ligands[1], False, "dodecahedron"))
    ] == ["ligand"]
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
    assert check_made_from("x", tmp_path, None, {}, log=notes.append) is False and check_made_from(
        "x", tmp_path, {}, {}
    )

    # the pre-equilibration of such a version: the time step in the file is the one that was used.
    # The area of the box is recorded for every system, the water molecules near the ligand only in a
    # bilayer (a soluble system was once taken for one because of its area)
    np.savez(tmp_path / "old_equil.npz", timestep_fs=2.0, simulated_ns=5.0, area_nm2=np.full(40, 61.7),
             pocket_waters=np.zeros(0), notes="[]")
    old_eq = np.load(tmp_path / "old_equil.npz")
    assert equil_in_membrane(old_eq) is False

    def searched_then(**sections):
        """The selection of those versions (the search), whose pre-equilibration ends with free MD."""
        return config(restraints={"selection": "auto"}, **sections)

    # under the default of today the pre-equilibration of a complex leg holds the pose that was given and has
    # no free MD: a file of free MD is not one of it
    held = "the solute during the pre-equilibration: free -> held (restraints.selection: interactions)"
    assert equil_contradictions(old_eq, cfg, "complex", False) == ["length of the MD: 5 ns -> 0 ns", held]
    assert equil_contradictions(old_eq, searched_then(), "complex", False) == []     # 2 fs used, 4 fs configured
    assert equil_contradictions(old_eq, searched_then(dynamics={"timestep_fs": 1.0}), "complex", False) \
        == ["time step: 2 fs -> 1 fs"]
    assert equil_contradictions(old_eq, searched_then(sampling={"pre_equilibration": {"complex": {"md_ns": 1.0}}}),
                                "complex", False) == ["length of the MD: 5 ns -> 1 ns"]
    assert equil_contradictions(old_eq, searched_then(membrane={"enabled": True}), "complex", True) \
        == ["length of the MD: 5 ns -> 10 ns", "bilayer: no -> yes"]
    np.savez(tmp_path / "old_membrane.npz", timestep_fs=4.0, simulated_ns=15.0, area_nm2=np.full(40, 66.0),
             pocket_waters=np.full(40, 6.0))
    old_membrane = np.load(tmp_path / "old_membrane.npz")
    assert equil_in_membrane(old_membrane) is True
    # the 10 ns of a bilayer are as long as those the default asks for, and were free: a file without a record
    # is not the pre-equilibration that holds the solute (the fourth pass of Codex: it was used, and the MD for
    # the restraint started from the end of 10 ns of free MD)
    assert equil_contradictions(old_membrane, config(membrane={"enabled": True}), "complex", True) == [held]
    assert equil_contradictions(old_membrane, searched_then(membrane={"enabled": True}), "complex", True) == []
    assert "bilayer: yes -> no" in equil_contradictions(old_membrane, searched_then(), "complex", False)
    np.savez(tmp_path / "older.npz", timestep_fs=4.0, simulated_ns=5.0)          # a file that does not tell
    assert equil_in_membrane(np.load(tmp_path / "older.npz")) is None
    assert equil_contradictions(np.load(tmp_path / "older.npz"), searched_then(), "complex", False) == []
    # review of 037bdf1 (P2): versions before revision 15 integrated whole frame intervals of 20 ps and recorded the
    # length asked for. A file of 50 ps holds 40 ps and is refused whatever its record says; one of today holds 50 ps
    from cufep.workflow import check_equil_file, leg_seed

    short = config(sampling={"pre_equilibration": {"solvent": {"md_ns": 0.05}}})
    seed = leg_seed(short, 0, "solvent")
    record = json.dumps(equil_made_with(short, "solvent", False, seed))
    np.savez(tmp_path / "rounded.npz", timestep_fs=4.0, simulated_ns=0.05, made_with=record)
    with pytest.raises(WorkflowError, match="holds 40 ps of unrestrained MD where 50 ps were asked for"):
        check_equil_file(np.load(tmp_path / "rounded.npz"), tmp_path / "rounded.npz", short, "solvent", False, seed)
    assert equil_contradictions(np.load(tmp_path / "rounded.npz"), short, "solvent", False) == [
        "length of the MD: 0.04 ns -> 0.05 ns"]
    np.savez(tmp_path / "integrated.npz", timestep_fs=4.0, simulated_ns=0.05, simulated_ns_is_integrated=True,
             made_with=record)
    assert check_equil_file(np.load(tmp_path / "integrated.npz"), tmp_path / "integrated.npz", short, "solvent", False,
                            seed) is True
    whole = config(sampling={"pre_equilibration": {"solvent": {"md_ns": 0.04}}})
    np.savez(tmp_path / "whole.npz", timestep_fs=4.0, simulated_ns=0.04,
             made_with=json.dumps(equil_made_with(whole, "solvent", False, seed)))
    assert check_equil_file(np.load(tmp_path / "whole.npz"), tmp_path / "whole.npz", whole, "solvent", False, seed)
    # the platform is part of the record (review of eb6f89f: a pre-equilibration on the Reference platform was used
    # by a run on the CPU platform). A file of a version that recorded none says nothing of its platform and is
    # used as before, on whatever platform (review of 877ab88: it was taken for one of CUDA)
    on_reference = config(sampling={"pre_equilibration": {"solvent": {"md_ns": 0.04}}},
                          backend={"name": "openmm", "platform": "Reference"})
    on_cpu = config(sampling={"pre_equilibration": {"solvent": {"md_ns": 0.04}}},
                    backend={"name": "openmm", "platform": "CPU"})
    record = equil_made_with(on_reference, "solvent", False, seed)
    assert record["platform"] == "Reference"
    np.savez(tmp_path / "reference.npz", timestep_fs=4.0, simulated_ns=0.04, made_with=json.dumps(record))
    assert check_equil_file(np.load(tmp_path / "reference.npz"), tmp_path / "reference.npz", on_reference, "solvent",
                            False, seed)
    with pytest.raises(WorkflowError, match="platform: Reference -> CPU"):
        check_equil_file(np.load(tmp_path / "reference.npz"), tmp_path / "reference.npz", on_cpu, "solvent", False,
                         seed)
    earlier = {key: value for key, value in record.items() if key != "platform"}
    np.savez(tmp_path / "earlier.npz", timestep_fs=4.0, simulated_ns=0.04, made_with=json.dumps(earlier))
    for cfg_now in (on_reference, on_cpu):
        assert check_equil_file(np.load(tmp_path / "earlier.npz"), tmp_path / "earlier.npz", cfg_now, "solvent", False,
                                seed)
    # the fifth pass of Codex: the first two commits that held the solute (720cdb1, 48213b7) recorded md_ns, which
    # such a pre-equilibration does not have and its record no longer holds. Their files are this pre-equilibration
    # (the two example runs of 720cdb1 were refused: "pre_equilibration.md_ns: 5.0 -> not given"); a file of free
    # MD is not, whatever its md_ns
    holding = config()
    seed = leg_seed(holding, 0, "complex")
    record = equil_made_with(holding, "complex", False, seed)
    assert record["solute_held"] is True and "md_ns" not in record["pre_equilibration"]
    of_720cdb1 = {**record, "pre_equilibration": {**record["pre_equilibration"], "md_ns": 5.0}}
    np.savez(tmp_path / "held.npz", timestep_fs=4.0, simulated_ns=0.0, simulated_ns_is_integrated=True,
             made_with=json.dumps(of_720cdb1))
    assert check_equil_file(np.load(tmp_path / "held.npz"), tmp_path / "held.npz", holding, "complex", False, seed)
    free = {key: value for key, value in of_720cdb1.items() if key != "solute_held"}
    np.savez(tmp_path / "free.npz", timestep_fs=4.0, simulated_ns=5.0, simulated_ns_is_integrated=True,
             made_with=json.dumps(free))
    with pytest.raises(WorkflowError, match="md_ns: 5.0 -> not given; solute_held: not recorded -> True"):
        check_equil_file(np.load(tmp_path / "free.npz"), tmp_path / "free.npz", holding, "complex", False, seed)
    # the precision the file says its MD ran in is the one of the platform where the platform has its own (the
    # fourth pass of Codex: a run on the Reference platform, which is double, was recorded as mixed)
    from cufep.workflow import precision_that_ran

    assert precision_that_ran(on_reference, "openmm") == "that of the Reference platform"
    assert precision_that_ran(config(backend={"precision": "double"}), "cupotential") == "mixed"
    assert precision_that_ran(config(backend={"precision": "double"}), "openmm") == "double"
    # the restraint of such a version: all six force constants it holds (kJ/mol/nm^2 and kJ/mol/rad^2)
    angles = {key: 80.0 * 4.184 for key in ("k_theta_a", "k_theta_b", "k_phi_a", "k_phi_b", "k_phi_c")}
    old_restraint = {"restraint": {"host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6], "k_r": 10.0 * 4.184 * 100.0,
                                   **angles},
                     "info": {"selection": "auto", "force_constants": "openfe"}}

    def searched(**settings):
        """A configuration with the selection of those versions, the search (the default until 2026-11-21)."""
        return config(restraints={"selection": "auto", **settings})

    assert restraint_contradictions(old_restraint, searched()) == []
    assert restraint_contradictions(old_restraint, searched(force_constants="auto")) \
        == ["restraints.force_constants: openfe -> auto"]
    # the selection "interactions" came later, with settings of its own. The record of a restraint of another
    # selection is what it was before (commit cda6965), so that the restraints of earlier runs stay usable; a
    # restraint of the search is not one of a run that chooses from interactions, recorded or not
    from_md = config()
    assert from_md.restraints.selection == "interactions"
    assert restraint_made_with(searched()) == {
        "type": "boresch", "selection": "auto", "force_constants": "openfe", "k_r_kcal_per_A2": 10.0,
        "k_angle_kcal_per_rad2": 80.0, "host_atoms": None, "guest_atoms": None, "rmsf_cutoff_nm": 0.1,
        "host_min_distance_nm": 0.5, "host_max_distance_nm": 1.5, "dssp_filter": True}
    assert restraint_made_with(from_md)["interactions"] == {
        "trajectories": 4, "length_ns": 3.0, "discard_ns": 1.0, "frame_interval_ps": 10.0, "min_frame_fraction": 0.4,
        "angle_range_deg": [45.0, 135.0], "stable_structure_fraction": 0.5, "second_interaction_ratio": 0.6,
        "fallback_frame_fraction": 0.15, "fallback_cutoff_nm": 1.0}
    # the settings of a record are compared as they are (reviews of eb6f89f and 877ab88: with a tolerance of
    # 1e-9, 0.3999999999 and 0.4 were one setting, which is 319 and 320 frames; a rank that turns over at
    # stable_structure_fraction 0.5 and not at 0.4999999999; a frame every 2500 and every 2501 steps; and the
    # seeds 1000000000 and 1000000001 were one seed, whose trajectories were used together)
    for section, key, one, other in (("interactions", "min_frame_fraction", 0.3999999999, 0.4),
                                     ("interactions", "stable_structure_fraction", 0.4999999999, 0.5),
                                     ("interactions", "frame_interval_ps", 10.0019999999, 10.0020000001)):
        records = [restraint_made_with(config(restraints={section: {key: value}})) for value in (one, other)]
        assert settings_differences(*records) == [f"{section}.{key}: {one} -> {other}"]
    assert settings_differences({"seed": 1000000000}, {"seed": 1000000001}) == ["seed: 1000000000 -> 1000000001"]
    assert settings_differences({"k": 4, "range": [45.0, 135]}, {"k": 4.0, "range": (45, 135.0)}) == []
    assert restraint_contradictions(old_restraint, from_md) == ["restraints.selection: auto -> interactions"]
    with pytest.raises(WorkflowError, match="selection: auto -> interactions"):
        check_made_from("the restraint of this leg", tmp_path / "restraint.json", restraint_made_with(searched()),
                        restraint_made_with(from_md))
    assert check_made_from("the restraint of this leg", tmp_path / "restraint.json", restraint_made_with(from_md),
                           restraint_made_with(config(restraints={"selection": "interactions"})))
    with pytest.raises(WorkflowError, match="interactions.length_ns: 3.0 -> 2.0"):
        check_made_from("the restraint of this leg", tmp_path / "restraint.json", restraint_made_with(from_md),
                        restraint_made_with(config(restraints={"selection": "interactions",
                                                               "interactions": {"length_ns": 2.0}})))
    assert restraint_contradictions(old_restraint, searched(k_r_kcal_per_A2=20.0)) \
        == ["restraints.k_r_kcal_per_A2: 10 -> 20"]
    # review of PR 1: the distance constant kept and the angle constant changed from 80 to 20 went through
    assert restraint_contradictions(old_restraint, searched(k_angle_kcal_per_rad2=20.0)) == [
        "restraints.k_angle_kcal_per_rad2: k_theta_a 80, k_theta_b 80, k_phi_a 80, k_phi_b 80, k_phi_c 80 -> 20"]
    one_differs = {"restraint": dict(old_restraint["restraint"], k_phi_b=40.0 * 4.184), "info": old_restraint["info"]}
    assert restraint_contradictions(one_differs, searched()) == ["restraints.k_angle_kcal_per_rad2: k_phi_b 40 -> 80"]
    only_distance = {"restraint": {"k_r": 10.0 * 4.184 * 100.0}, "info": old_restraint["info"]}
    assert restraint_contradictions(only_distance, searched(k_angle_kcal_per_rad2=20.0)) == []
    assert restraint_contradictions(old_restraint, config(restraints={
        "selection": "manual", "host_atoms": [1, 2, 9], "guest_atoms": [4, 5, 6]})) \
        == ["restraints.selection: auto -> manual", "restraints.host_atoms: [1, 2, 3] -> [1, 2, 9]"]
    # manual anchors take their constants from their fluctuations under auto (review of PR 1); a manual restraint
    # of an earlier version, which ignored auto, holds the configured constants and is refused under auto
    manual = {"restraint": dict(old_restraint["restraint"]), "info": {"selection": "manual"}}
    assert restraint_contradictions(manual, config(restraints={
        "selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6], "force_constants": "auto",
        "k_angle_kcal_per_rad2": 20.0})) == [
        "restraints.force_constants: the configured constants (a manual restraint of an earlier version) -> auto"]
    assert restraint_contradictions(manual, config(restraints={
        "selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6],
        "k_angle_kcal_per_rad2": 20.0})) == [
        "restraints.k_angle_kcal_per_rad2: k_theta_a 80, k_theta_b 80, k_phi_a 80, k_phi_b 80, k_phi_c 80 -> 20"]
    # independent review of c691e37 (P2): such a file of 64751ca had a record of its origin (with auto), so the
    # contradiction above, which counts for files without one, was never asked and the 10/80 restraint was used under
    # a run that records auto. It is refused whatever its record says, by the check of the pre-flight and of the leg
    from cufep.workflow import check_restraint_file

    manual_auto = config(restraints={"selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6],
                                     "force_constants": "auto"})
    of_64751ca = dict(manual, made_with=restraint_made_with(manual_auto))
    assert check_made_from("the restraint of this leg", tmp_path / "restraint.json", of_64751ca["made_with"],
                           restraint_made_with(manual_auto))                     # the record alone lets it through
    with pytest.raises(WorkflowError, match="manual restraint of a version that used the configured force constants"):
        check_restraint_file(of_64751ca, tmp_path / "restraint.json", manual_auto)
    with pytest.raises(WorkflowError, match="did not record what it was made from"):
        check_restraint_file(manual, tmp_path / "restraint.json", manual_auto)   # without a record: as before
    # a file of today records its constants and the rule its anchors were held to: used, and no contradiction is
    # made up for it
    from cufep.workflow import FLAT_ANGLE_RULE

    today = {"restraint": dict(old_restraint["restraint"]), "made_with": restraint_made_with(manual_auto),
             "info": {"selection": "manual", "force_constants": "auto", "flat_angle_rule": FLAT_ANGLE_RULE}}
    assert check_restraint_file(today, tmp_path / "restraint.json", manual_auto) is True
    assert restraint_contradictions(today, manual_auto) == []
    # one with constants from fluctuations of a version that asked OpenFE's rule of the flat angle with the
    # configured constants (review of 877ab88: theta_A at 30 degrees passed with 80 kcal/mol/rad^2, the restraint
    # got 20, 4.6 kT from a straight line, and a later version went on to sample with it)
    before = dict(today, info={"selection": "manual", "force_constants": "auto"})
    with pytest.raises(WorkflowError, match="held the anchors to OpenFE's rule of the flat angle with the configured"):
        check_restraint_file(before, tmp_path / "restraint.json", manual_auto)
    manual_openfe = config(restraints={"selection": "manual", "host_atoms": [1, 2, 3], "guest_atoms": [4, 5, 6]})
    assert check_restraint_file(dict(manual, made_with=restraint_made_with(manual_openfe)), tmp_path / "restraint.json",
                                manual_openfe) is True                            # the configured constants were asked

    # pre-equilibration and restraint of a leg
    eq = equil_made_with(cfg, "complex", False, 7)
    longer = config(sampling={"pre_equilibration": {"complex": {"md_ns": 1.0}}})
    # the free MD of a pre-equilibration is a setting of the selections that have it (the search here)
    searched_longer = config(restraints={"selection": "auto"},
                             sampling={"pre_equilibration": {"complex": {"md_ns": 1.0}}})
    assert settings_differences(equil_made_with(searched(), "complex", False, 7),
                                equil_made_with(searched_longer, "complex", False, 7)) == [
        "pre_equilibration.md_ns: 5.0 -> 1.0"]
    assert settings_differences(eq, equil_made_with(longer, "complex", False, 7)) == []
    assert "pre_equilibration.md_ns: not recorded -> 0.5" in settings_differences(
        eq, equil_made_with(longer, "solvent", False, 7))
    assert settings_differences(
        eq, equil_made_with(config(thermodynamics={"temperature_K": 310.0}), "complex", False, 7)
    ) == ["temperature_K: 298.15 -> 310.0"]
    in_membrane = equil_made_with(config(membrane={"enabled": True, "equilibration_ns": 20.0}), "complex", True, 7)
    assert settings_differences(equil_made_with(config(membrane={"enabled": True}), "complex", True, 7), in_membrane) \
        == ["membrane_equilibration_ns: 10.0 -> 20.0"]
    with pytest.raises(WorkflowError, match="the pre-equilibration of this leg in .*equil.npz cannot be used again"):
        check_made_from("the pre-equilibration of this leg", tmp_path / "equil.npz", eq, in_membrane)
    # backend.fallback decides whether the pre-equilibration may be repeated on stock OpenMM: a file made with the
    # fallback is not one of a run without it (review of f5a77ce: "no difference: the file is used"). The record
    # of a run with it is the one every earlier version wrote, whose files stay usable
    without_fallback = equil_made_with(config(backend={"fallback": "none"}), "complex", False, 7)
    assert settings_differences(eq, without_fallback) == ["fallback: not recorded -> none"]
    assert "fallback" not in eq
    # plain MD runs in single precision where it is asked (the owner's decision of 2026-11-21). Earlier versions
    # ran it in mixed precision then and recorded "single": such a file is not one of an MD in single precision
    single = equil_made_with(config(backend={"precision": "single"}), "complex", False, 7)
    earlier = {key: value for key, value in single.items() if key != "plain_md_precision"}
    assert earlier == dict(eq, precision="single")                      # what an earlier version recorded
    assert settings_differences(earlier, single) == ["plain_md_precision: not recorded -> single"]
    assert "plain_md_precision" not in eq
    # double asked of the plugin, which has none: earlier versions ran that MD on stock OpenMM in double
    # precision and recorded cupotential and double; the plugin runs it in mixed precision now (review of eb6f89f)
    double = equil_made_with(config(backend={"precision": "double"}), "complex", False, 7)
    assert double["backend"] == "cupotential" and double["plain_md_precision"] == "mixed"
    stock = equil_made_with(config(backend={"name": "openmm", "precision": "double"}), "complex", False, 7)
    assert "plain_md_precision" not in stock
    # the platform is part of the record, as it is of a trajectory for the restraint: a pre-equilibration on
    # the Reference platform was used again by a run on the CPU platform. A record without it is one of CUDA
    on_cpu = equil_made_with(config(backend={"name": "openmm", "platform": "CPU"}), "complex", False, 7)
    reference = equil_made_with(config(backend={"name": "openmm", "platform": "Reference"}), "complex", False, 7)
    assert settings_differences(reference, on_cpu) == ["platform: Reference -> CPU"] and eq["platform"] == "CUDA"
    stiff = restraint_made_with(config(restraints={"k_r_kcal_per_A2": 20.0}))
    assert settings_differences(restraint_made_with(cfg), stiff) == ["k_r_kcal_per_A2: 10.0 -> 20.0"]


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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

    monkeypatch.setattr(
        driver, "plan", lambda cfg, log, ligands=None: (cfg.model_copy(deep=True), AutoReport(), Protein(), [lig])
    )
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
    # samples of a version that did not record their conditions are held to what their own file says:
    # six iterations of 2.5 ps at 298.15 K, two of them equilibration
    import h5py
    from conftest import as_earlier_version

    with h5py.File(as_earlier_version(hrex), "a") as h5:
        del h5.attrs["conditions"]
        h5.attrs["temperature_K"], h5.attrs["iteration_ps"], h5.attrs["n_equilibration_iterations"] = 298.15, 2.5, 2
    with pytest.raises(ConditionsError, match="did not record the conditions.*equilibration per lambda window: 2 "
                                              "iterations .0.005 ns. -> 400 .1 ns.; production per lambda window: the "
                                              "samples hold 4 iterations .0.01 ns. and 10 ns are 4000 iterations.*"
                                              "--reschedule"):
        check_leg_reuse(first, lig_dir, "solvent", 0, prep_info, 10.0)
    short = {"equilibration_ns": 0.005, "production_ns": 0.01}
    check_leg_reuse(config(sampling=short), lig_dir, "solvent", 0, prep_info, 0.01)          # what they hold
    with pytest.raises(ConditionsError, match="temperature: 298.15 K -> 310 K"):
        check_leg_reuse(config(sampling=short, thermodynamics={"temperature_K": 310.0}), lig_dir, "solvent", 0,
                        prep_info, 0.01)


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_samples_without_a_record_of_their_settings_are_checked_before_anything_is_written(
    tmp_path, monkeypatch, ligands
):
    """Review of PR 1: one version of cuFEP recorded the conditions of the samples with a checksum of the
    System and without the settings behind it. A run with another soft-core passed the checks before the
    run; the sampler then found the other System, when resolved_config.yaml was already rewritten, the
    finished result moved away and the leg marked as failed."""
    import hashlib
    from pathlib import Path

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    import cufep.run as driver
    from cufep.alchemy.factory import make_alchemical_system
    from cufep.alchemy.schedule import fixed_schedule
    from cufep.config.auto import AutoReport
    from cufep.config.schema import Config
    from cufep.sampling.storage import ConditionsError, LegStorage
    from cufep.workflow import (
        WorkflowError,
        _write_system,
        alchemical_leg,
        check_leg_reuse,
        conditions_of_this_run,
        equil_made_with,
        leg_seed,
        leg_spec,
        load_prepared,
        prep_made_from,
    )

    lig = ligands[0]
    out = tmp_path / "out"

    def config(**sections):
        return Config.model_validate({"system": {"protein": str(tmp_path / "p.pdb"), "ligand": "l.sdf"},
                                      "output": {"dir": str(out)}, "resources": {"gpus": []},
                                      "solvation": {"box_shape": "dodecahedron"}, **sections})

    first, other = config(), config(alchemy={"softcore_alpha": 0.8})
    assert first.alchemy.softcore_alpha == 0.5
    # the prepared solvent system of the ligand: a small water box in which one molecule is the ligand
    ff = app.ForceField("amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    ligand_atoms = [a.index for a in list(mod.topology.residues())[10].atoms()]
    lig_dir = out / lig.name
    (lig_dir / "prep").mkdir(parents=True)
    _write_system(lig_dir / "prep", "solvent", mod, system, ligand_atoms)
    prep_info = {"ligand": lig.name, "complex_barostat": "isotropic", "membrane": False, "box_shape": "dodecahedron",
                 "made_from": prep_made_from(first, lig, False, "dodecahedron")}
    (lig_dir / "prep" / "prep.json").write_text(json.dumps(prep_info))
    # what an earlier run of the leg left: pre-equilibration, lambda windows, samples and a result
    leg_dir = lig_dir / "repeat_0" / "solvent"
    leg_dir.mkdir(parents=True)
    box = np.diag([1.9, 1.9, 1.9])
    # the length its record was made for (sampling.pre_equilibration.solvent.md_ns: 0.5 ns), all of it integrated
    np.savez(leg_dir / "equil.npz", box=box, timestep_fs=2.0, simulated_ns=0.5,
             made_with=json.dumps(equil_made_with(first, "solvent", False, leg_seed(first, 0, "solvent"))))
    schedule = fixed_schedule(
        ["discharge", "vanish"], {"discharge": 1, "vanish": 2}, rest2_scale=first.sampling.rest2.scale
    )
    schedule.save(leg_dir / "schedule.yaml", leg="solvent")
    states = [state.to_dict() for state in schedule.states]

    # the System as the run of a leg made it before the function existed: box, alchemical System, XML
    prep = load_prepared(lig_dir / "prep", "solvent")
    by_hand = mm.XmlSerializer.deserialize(prep["system_xml"])
    by_hand.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in box])
    alch = make_alchemical_system(by_hand, prep["ligand"], 0.5, rest2=True, rest2_atoms=[],
                                  bonds=[(b[0].index, b[1].index) for b in prep["topology"].bonds()])
    checksum = hashlib.sha256(mm.XmlSerializer.serialize(alch.system).encode()).hexdigest()
    built = alchemical_leg(first, mm.XmlSerializer.deserialize(prep["system_xml"]), prep, box, None, False)
    assert hashlib.sha256(built.xml.encode()).hexdigest() == checksum and built.rest2_on and built.hot_atoms == []
    assert built.tail["long_cutoff_nm"] == 2.0                    # analysis.lrc: reweight is the default
    spec = leg_spec(first, built, states, "isotropic", 2.0, "openmm", 7, name="a leg", frame_atoms=[1, 2])
    assert spec.name == "a leg" and spec.seed == 7 and spec.frame_atoms == [1, 2] and spec.steps_per_iteration == 1250
    assert spec.settings["alchemy"]["softcore_alpha"] == 0.5 and spec.conditions()["system_sha256"] == checksum
    current = conditions_of_this_run(first, lig_dir, "solvent", 0, prep_info, "openmm", 6)
    assert current == spec.conditions()

    # the samples of that version: all conditions, the tail with all its parameters, no settings
    conditions = dict(current, lrc=built.tail)
    del conditions["settings"]
    storage = LegStorage(leg_dir / "hrex", len(states), len(states))
    assert storage.check_conditions(conditions) == "new"
    for _ in range(6):
        storage.append(np.zeros((len(states), len(states))), np.arange(len(states)), np.ones(len(states)))
    storage.close()
    (leg_dir / "result.json").write_text(json.dumps({"leg": "solvent", "delta_g_decouple": 1.0}))

    check_leg_reuse(first, lig_dir, "solvent", 0, prep_info, 10.0)                # the run that made them
    refusal = "made under other conditions than this run asks for .the System .force field, soft-core, restraint or " \
              "REST2 region.*They are left as they are"
    with pytest.raises(ConditionsError, match=refusal):
        check_leg_reuse(other, lig_dir, "solvent", 0, prep_info, 10.0)
    # another temperature is refused by the pre-equilibration already, before a System is built
    with pytest.raises(WorkflowError, match="the pre-equilibration of this leg.*temperature_K: 298.15 -> 310.0"):
        check_leg_reuse(config(thermodynamics={"temperature_K": 310.0}), lig_dir, "solvent", 0, prep_info, 10.0)
    check_leg_reuse(other, lig_dir, "solvent", 0, prep_info, 10.0, reschedule=True)      # the leg starts anew

    # a run with the other soft-core is refused before anything is written
    class Protein:
        def write_pdb(self, path):
            Path(path).write_text("ATOM")

    monkeypatch.setattr(
        driver, "plan", lambda cfg, log, ligands=None: (cfg.model_copy(deep=True), AutoReport(), Protein(), [lig])
    )
    monkeypatch.setattr(driver, "run_jobs", lambda jobs, *args, **kwargs: pytest.fail("the run was not refused"))
    (out / "resolved_config.yaml").write_text("the record of the run that made the samples")
    before = {path: path.read_bytes() for path in (out / "resolved_config.yaml", leg_dir / "result.json",
                                                   *sorted((leg_dir / "hrex").iterdir()))}
    log = []
    with pytest.raises(ConditionsError, match=refusal):
        driver._run(other, out, False, log.append)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert sorted((leg_dir / "hrex").iterdir()) == sorted(path for path in before if path.parent == leg_dir / "hrex")
    assert not (leg_dir / "started.json").exists() and not (leg_dir / "result.previous.json").exists()
    assert (out / "resolved_config.refused.yaml").exists() and any(line.startswith("refused: ") for line in log)

    # without a file from which the System is built, nothing shows that the conditions are the same
    (leg_dir / "schedule.yaml").rename(leg_dir / "schedule.kept")
    with pytest.raises(ConditionsError, match="recorded a checksum of their System and not the settings behind it.*"
                                              "the lambda schedule .schedule.yaml. of this leg is not there. Nothing "
                                              "was changed"):
        check_leg_reuse(first, lig_dir, "solvent", 0, prep_info, 10.0)


def tail_leg(leg_dir, lrc="reweight", radius=2.0, tail_in_production=True, n=120, n_equilibration=20):
    """A leg directory as a finished run leaves it, with tail data recorded to ``radius`` (or none)."""
    from cufep.sampling.storage import LegStorage

    rng = np.random.default_rng(4)
    storage = LegStorage(leg_dir / "hrex", 3, 3)
    conditions = {"temperature_K": 298.15, "pressure_bar": 1.0, "barostat": "isotropic",
                  "lrc": None if lrc == "none" else {"cutoff_nm": 0.9, "long_cutoff_nm": radius, "n_ligand": 3,
                                                    "n_environment": 30, "s6": 1.0, "s12": 1.0, "sha256": "x"}}
    storage.check_conditions(conditions)
    meta = {"temperature_K": 298.15, "iteration_ps": 2.5, "backend": "openmm", "leg": leg_dir.name}
    if lrc != "none":
        meta.update(lrc_s6=1.0, lrc_s12=1.0, lrc_cutoff_nm=0.9)
    storage.set_meta(**meta)
    storage.set_plan(n_equilibration, n)
    for it in range(n):
        x = rng.normal(size=3)
        storage.append(np.repeat((0.5 * x ** 2)[:, None], 3, axis=1), np.arange(3), np.ones(3))
        if lrc != "none" and it % 4 == 0 and (tail_in_production or it < n_equilibration):
            storage.append_record(it, {"lrc_du": -2.0 + 0.1 * rng.normal()})
    storage.close()


def test_what_an_analysis_needs_from_the_samples_is_checked_in_one_place_before_any_sampling(tmp_path):
    """Review of PR 1: (1) cufep analyze --config with analysis.lrc_cutoff_nm 3.0 on samples whose tail
    energies were recorded to 2.0 nm gave the same result as 2.0, without a word and without the radius;
    (2) a complex leg finished with analysis.lrc none, and the solvent leg alone was run with reweight: the
    check before the run passed, and the two legs could not be combined after the solvent leg had been
    sampled in full."""
    import yaml
    from click.testing import CliRunner

    from cufep.cli import main
    from cufep.config.schema import AnalysisConfig, Config
    from cufep.sampling.storage import stored_attributes
    from cufep.workflow import (
        WorkflowError,
        analysis_problems,
        analysis_record,
        analysis_settings_of,
        check_partner_legs,
        leg_result_from_samples,
    )

    def config(**analysis):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": analysis})

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    tail_leg(leg, radius=2.0)
    attrs = stored_attributes(leg / "hrex")[1]
    assert analysis_problems(attrs, config()) == [] and analysis_problems(attrs, config(lrc="isotropic")) == []
    assert analysis_problems(attrs, config(lrc_cutoff_nm=3.0)) == [
        "their tail energies were recorded to an outer radius of 2 nm and analysis.lrc_cutoff_nm is 3 (the radius "
        "cannot be changed after the sampling)"]
    assert analysis_problems(attrs, config(lrc="none", lrc_cutoff_nm=3.0)) == []       # no correction, no radius
    # cufep analyze: the radius that was used is printed; another one given explicitly is refused
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["lrc_long_cutoff_nm"] == 2.0 and shown["lrc_cutoff_nm"] == 0.9 and shown["lrc_method"] == "reweight"
    for radius, refused in ((2.0, False), (3.0, True)):
        path = tmp_path / f"r{radius}.yaml"
        path.write_text(yaml.safe_dump({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                        "analysis": {"lrc_cutoff_nm": radius}}))
        out = CliRunner().invoke(main, ["analyze", str(leg), "--config", str(path)])
        assert (out.exit_code == 1) is refused
        if refused:
            assert "recorded to an outer radius of 2 nm and analysis.lrc_cutoff_nm is 3" in out.output
        else:
            assert json.loads(out.output)["lrc_long_cutoff_nm"] == 2.0
    # the record of the settings holds the radius; one of an earlier version takes it from the samples
    record = analysis_record(config())
    assert record["lrc_cutoff_nm"] == 2.0
    del record["lrc_cutoff_nm"]
    leg15 = tmp_path / "lig15" / "repeat_0" / "solvent"
    tail_leg(leg15, radius=1.5)
    attrs15 = stored_attributes(leg15 / "hrex")[1]
    assert analysis_settings_of(record, attrs15).analysis.lrc_cutoff_nm == 1.5
    assert analysis_settings_of(record).analysis.lrc_cutoff_nm == 2.0
    # samples without tail data refuse an analysis with a correction, wherever it is made
    none = tmp_path / "lig" / "repeat_0" / "complex"
    tail_leg(none, lrc="none")
    with pytest.raises(WorkflowError, match="they hold no data for the tail correction .they were made with "
                                            "analysis.lrc: none. and analysis.lrc is reweight"):
        leg_result_from_samples(none, AnalysisConfig(), {"leg": "complex"})
    (none / "result.json").write_text(json.dumps({"leg": "complex", "temperature_K": 298.15, "pressure_bar": 1.0,
                                                  "delta_g_decouple": 1.0, "delta_g_mbar_error": 0.1}))
    with pytest.raises(WorkflowError, match="the complex leg of repeat 0 .finished, not part of this run. has samples "
                                            "that cannot be analysed with the settings of this run: they hold no data "
                                            "for the tail correction.*Nothing was run"):
        check_partner_legs(tmp_path / "lig", "lig", {(0, "solvent")}, 1, config())
    check_partner_legs(tmp_path / "lig", "lig", {(0, "solvent")}, 1, config(lrc="none"))   # what they were made with
    # a finished leg whose tail energies were all recorded in its equilibration
    early = tmp_path / "early" / "repeat_0" / "complex"
    tail_leg(early, tail_in_production=False)
    (early / "result.json").write_text(json.dumps({"leg": "complex", "temperature_K": 298.15, "pressure_bar": 1.0,
                                                   "delta_g_decouple": 1.0, "delta_g_mbar_error": 0.1}))
    with pytest.raises(WorkflowError, match="the tail correction of their production cannot be estimated: 0 tail "
                                            "energies were recorded in these iterations; 4 are needed"):
        check_partner_legs(tmp_path / "early", "lig", {(0, "solvent")}, 1, config())


def test_the_run_and_the_check_of_stored_samples_count_iterations_by_one_rule():
    """Review of PR 1: with 2.5 ps per iteration and 3 fs steps a run makes 833 steps per iteration (2.499 ps)
    and 4000 iterations for 10 ns; read again with the same settings, its samples were refused as 0.004 ns
    too short."""
    from cufep.config.schema import Config
    from cufep.workflow import iterations_for, planned_iterations, unrecorded_samples_problems

    plan = planned_iterations(2.5, 3.0, 1.0, 10.0)
    assert plan.steps_per_iteration == 833 and plan.iteration_ps == pytest.approx(2.499)
    assert (plan.n_equilibration, plan.n_production) == (400, 4000)
    assert iterations_for(0.0, 2.5, 1) == 1 and iterations_for(0.001, 2.5, 4) == 4
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                 "dynamics": {"timestep_fs": 3.0}, "sampling": {"equilibration_ns": 1.0}})
    attrs = {"temperature_K": 298.15, "iteration_ps": 833 * 3.0e-3, "n_equilibration_iterations": 400,
             "n_planned_iterations": 4400}
    from pathlib import Path
    assert unrecorded_samples_problems(attrs, 4400, Path("leg"), cfg, "solvent", 10.0, None, 3.0) == []
    short = unrecorded_samples_problems(attrs, 4399, Path("leg"), cfg, "solvent", 10.0, None, 3.0)
    assert short == ["production per lambda window: the samples hold 3999 iterations (9.9935 ns) and 10 ns are 4000 "
                     "iterations (samples of unrecorded conditions are not added to)"]
    # an equilibration of 10 ns at 3 fs: 4000 iterations by the run's rule (4002 by the length an iteration had)
    long = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                  "dynamics": {"timestep_fs": 3.0}, "sampling": {"equilibration_ns": 10.0}})
    attrs = dict(attrs, n_equilibration_iterations=4000, n_planned_iterations=8000)
    assert unrecorded_samples_problems(attrs, 8000, Path("leg"), long, "solvent", 10.0, None, 3.0) == []


@pytest.mark.parametrize("state", ["as made", "other radius", "no tail data", "tail only in equilibration"])
def test_one_state_of_a_leg_gives_one_answer_through_every_way_in(tmp_path, state):
    """The same leg through every way to its result: the analysis of a run (leg_result_from_samples), the
    refresh of a leg that a run does not sample, the check before a run of a finished leg it does not
    sample, cufep analyze of the leg directory, of its hrex directory and of an archived copy, and cufep
    analyze --config. Each gives the same free energy, or each refuses for the same reason."""
    import shutil

    import yaml
    from click.testing import CliRunner

    from cufep.cli import main
    from cufep.config.schema import Config
    from cufep.workflow import (
        WorkflowError,
        analysis_record,
        check_partner_legs,
        leg_result_from_samples,
        refresh_leg_result,
    )

    lig = tmp_path / "lig"
    leg = lig / "repeat_0" / "complex"
    tail_leg(leg, lrc="none" if state == "no tail data" else "reweight",
             tail_in_production=state != "tail only in equilibration")
    data = {"system": {"protein": "p.pdb", "ligand": "l.sdf"},
            "analysis": {"lrc_cutoff_nm": 3.0 if state == "other radius" else 2.0}}
    cfg = Config.model_validate(data)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(data))
    reason = {"as made": None, "other radius": "recorded to an outer radius of 2 nm and analysis.lrc_cutoff_nm is 3",
              "no tail data": "they hold no data for the tail correction",
              "tail only in equilibration": "tail energies"}[state]
    run = {"ligand": "lig", "leg": "complex", "repeat": 0, "backend": "openmm", "notes": []}
    # the record of the result is of another version, so that the refresh has to analyse the samples
    (leg / "result.json").write_text(json.dumps({"leg": "complex", "temperature_K": 298.15, "pressure_bar": 1.0,
                                                 "delta_g_decouple": 0.0, "delta_g_mbar_error": 0.1, "run": run,
                                                 "analysis_record": dict(analysis_record(cfg), version=1)}))
    archive = leg / "previous_1"
    archive.mkdir()
    shutil.copytree(leg / "hrex", archive / "hrex")
    shutil.copy(leg / "result.json", archive / "result.json")

    answers = {}
    try:
        answers["the run's analysis"] = leg_result_from_samples(leg, cfg, run)["delta_g_decouple"]
    except WorkflowError as exc:
        answers["the run's analysis"] = str(exc)
    log = []
    outcome = refresh_leg_result(leg, cfg, log.append)
    answers["the refresh"] = json.loads((leg / "result.json").read_text())["delta_g_decouple"] \
        if outcome == "reanalysed" else log[-1]
    try:
        check_partner_legs(lig, "lig", {(0, "solvent")}, 1, cfg)
        answers["the check before a run"] = "accepted"
    except WorkflowError as exc:
        answers["the check before a run"] = str(exc)
    for name, args in (("analyze", [str(leg)]), ("analyze of hrex", [str(leg / "hrex")]),
                       ("analyze of the archive", [str(archive)]),
                       ("analyze --config", [str(leg), "--config", str(config_file)])):
        if name != "analyze --config":                 # the settings of the leg's result are those of cfg
            pass
        out = CliRunner().invoke(main, ["analyze", *args] if name != "analyze --config" else ["analyze", *args])
        answers[name] = json.loads(out.output)["delta_g_decouple"] if out.exit_code == 0 else out.output

    if reason is None:
        values = {name: value for name, value in answers.items() if name != "the check before a run"}
        assert all(isinstance(value, float) for value in values.values()), answers
        assert max(values.values()) - min(values.values()) < 1e-9, answers
        assert answers["the check before a run"] == "accepted"
    elif state == "tail only in equilibration":
        # the samples say what they lack only once their production is read: the check before a run
        # reads the iterations of the tail energies, every analysis reads the production
        assert all("tail energies" in str(value) and ("0 tail" in str(value) or "production" in str(value))
                   for value in answers.values()), answers
    else:
        assert all(isinstance(value, str) and reason in value for value in answers.values()), answers


def test_the_check_before_a_run_takes_a_finished_leg_as_it_will_be_combined(tmp_path):
    """Review of PR 1: (1) the check before a run read temperature and pressure from the old result of a
    finished leg, while its analysis again takes them from its samples: 310 K samples with a result of
    298.15 K passed and were refused after the run, 298.15 K samples with a result of 310 K were refused
    before it; (2) a leg whose samples were gone, with lrc_method reweight and a radius of 1.5 nm in its
    result, passed and was refused at the combination; (3) tail energies were counted from the boundary
    on, also beyond the stored rows, where the analysis counts the production only."""
    import h5py
    from conftest import as_earlier_version, edit_attributes

    from cufep.config.schema import Config
    from cufep.workflow import WorkflowError, analysis_record, check_partner_legs

    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}})

    def finished(lig, temperature_in_samples, temperature_in_result, record=None):
        leg = lig / "repeat_0" / "complex"
        tail_leg(leg)
        edit_attributes(leg / "hrex", {"temperature_K": temperature_in_samples})
        (leg / "result.json").write_text(json.dumps({
            "leg": "complex", "temperature_K": temperature_in_result, "pressure_bar": 1.0, "delta_g_decouple": 1.0,
            "delta_g_mbar_error": 0.1, "lrc_method": "reweight", "lrc_long_cutoff_nm": 2.0,
            **({"analysis_record": record} if record else {})}))
        return leg

    finished(tmp_path / "a", 310.0, 298.15)
    with pytest.raises(WorkflowError, match="temperature .K.: the legs this run samples 298.15, the complex leg of "
                                            "repeat 0 .finished, not part of this run. 310.0"):
        check_partner_legs(tmp_path / "a", "lig", {(0, "solvent")}, 1, cfg)
    finished(tmp_path / "b", 298.15, 310.0)
    check_partner_legs(tmp_path / "b", "lig", {(0, "solvent")}, 1, cfg)
    # a result made with the settings of this run is combined as it is: its own temperature counts
    finished(tmp_path / "c", 298.15, 310.0, record=analysis_record(cfg))
    with pytest.raises(WorkflowError, match="temperature .K.: the legs this run samples 298.15, the complex leg"):
        check_partner_legs(tmp_path / "c", "lig", {(0, "solvent")}, 1, cfg)
    # samples gone: the result is combined as it is, and its correction must be that of this run
    gone = finished(tmp_path / "d", 298.15, 298.15)
    import shutil
    shutil.rmtree(gone / "hrex")
    data = json.loads((gone / "result.json").read_text())
    (gone / "result.json").write_text(json.dumps(dict(data, lrc_long_cutoff_nm=1.5)))
    with pytest.raises(WorkflowError, match="outer radius of the tail correction .nm.: the legs this run samples 2.0, "
                                            "the complex leg of repeat 0 .finished, not part of this run. 1.5"):
        check_partner_legs(tmp_path / "d", "lig", {(0, "solvent")}, 1, cfg)
    (gone / "result.json").write_text(json.dumps(dict(data, lrc_method="isotropic")))
    with pytest.raises(WorkflowError, match="correction for the dispersion tail: the legs this run samples reweight, "
                                            "the complex leg of repeat 0 .finished, not part of this run. isotropic"):
        check_partner_legs(tmp_path / "d", "lig", {(0, "solvent")}, 1, cfg)
    # the tail energies of the production [boundary, stored end), as the analysis takes them
    beyond = tmp_path / "e" / "repeat_0" / "complex"
    tail_leg(beyond, tail_in_production=False)
    with h5py.File(as_earlier_version(beyond / "hrex"), "a") as h5:   # records with rows beyond the stored ones
        h5["lrc_iter"].resize(40, axis=0)
        h5["lrc_iter"][5:] = np.arange(120, 155)
        h5["lrc_du"].resize(40, axis=0)
        h5["lrc_du"][5:] = -2.0
        h5["committed_counts"][1] = 40
    (beyond / "result.json").write_text(json.dumps({"leg": "complex", "temperature_K": 298.15, "pressure_bar": 1.0,
                                                    "delta_g_decouple": 1.0, "delta_g_mbar_error": 0.1}))
    with pytest.raises(WorkflowError, match="the tail correction of their production cannot be estimated: 0 tail "
                                            "energies were recorded in these iterations"):
        check_partner_legs(tmp_path / "e", "lig", {(0, "solvent")}, 1, cfg)
    # the isotropic correction needs both sums of the ligand, as its analysis does
    partial = tmp_path / "f" / "repeat_0" / "complex"
    tail_leg(partial)
    edit_attributes(partial / "hrex", remove=("lrc_s12",))
    (partial / "result.json").write_text((beyond / "result.json").read_text())
    isotropic = Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc": "isotropic"}}
    )
    with pytest.raises(WorkflowError, match="no Lennard-Jones sums of the ligand"):
        check_partner_legs(tmp_path / "f", "lig", {(0, "solvent")}, 1, isotropic)


def test_analyze_takes_the_recorded_radius_and_an_equilibration_given_exactly(tmp_path):
    """Review of PR 1: (1) samples without any record of their settings, with tail energies recorded to 1.5
    nm, were refused by cufep analyze against the default of 2.0 nm; (2) --equilibration-ns 0 dropped the
    first iteration of the production, and a boundary of an old file whose iteration was 2.499 ps for 2.5 ps
    asked could not be given exactly."""
    from click.testing import CliRunner

    from cufep.cli import main

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    tail_leg(leg, radius=1.5, n_equilibration=0, n=40)
    shown = CliRunner().invoke(main, ["analyze", str(leg)])
    assert shown.exit_code == 0, shown.output
    out = json.loads(shown.output)
    assert out["lrc_long_cutoff_nm"] == 1.5 and out["analysis_record"]["lrc_cutoff_nm"] == 1.5
    assert out["analysis_settings_from"].startswith("the defaults")
    zero = json.loads(CliRunner().invoke(main, ["analyze", str(leg), "--equilibration-ns", "0"]).output)
    assert zero["n_equilibration_iterations"] == 0 and zero["n_iterations_used"] == 40
    exact = json.loads(CliRunner().invoke(main, ["analyze", str(leg), "--equilibration-iterations", "7"]).output)
    assert exact["n_equilibration_iterations"] == 7 and exact["n_iterations_used"] == 33


def test_an_archive_that_failed_half_way_is_completed_and_never_read_half(tmp_path, monkeypatch):
    """Review of PR 1: the archive of --reschedule moved the samples and the result first and then copied the
    pre-equilibration; a copy cut short (disk full) left 20 bytes that were read in place of the whole file,
    and a second call found nothing more to move."""
    import shutil

    from cufep.workflow import archive_leg, leg_files

    leg = tmp_path / "repeat_0" / "complex"
    (leg / "hrex").mkdir(parents=True)
    for name in ("hrex/samples.h5", "result.json", "schedule.yaml"):
        (leg / name).write_text(name)
    (leg / "equil.npz").write_bytes(b"x" * 1000)
    (leg / "restraint.json").write_text("{}")
    real_copy = shutil.copy2
    calls = []

    def short_copy(src, dst, *args, **kwargs):
        calls.append(dst)
        if len(calls) == 1:
            with open(dst, "wb") as fh:
                fh.write(b"x" * 20)
            raise OSError(28, "No space left on device")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", short_copy)
    with pytest.raises(OSError, match="No space left"):
        archive_leg(leg)
    assert not (leg / "previous_1").exists()                       # no half archive to be read
    assert (leg / "hrex").exists() or list(leg.glob(".previous_1.partial/hrex"))
    previous = archive_leg(leg)                                     # the second call completes the same one
    assert previous == leg / "previous_1" and not list(leg.glob(".previous_*"))
    assert (previous / "equil.npz").read_bytes() == b"x" * 1000 and (previous / "hrex" / "samples.h5").exists()
    assert leg_files(previous)["equil.npz"] == previous / "equil.npz"
    assert archive_leg(leg) is None


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
def test_a_leg_is_not_continued_from_a_checkpoint_that_does_not_record_its_rows(tmp_path, ligands):
    """Review of PR 1: such a checkpoint (earlier versions) is refused before anything is run when the leg is
    still to be continued; a leg that holds all its iterations is analysed as it is."""
    from cufep.config.schema import Config
    from cufep.sampling.storage import ConditionsError, LegStorage
    from cufep.workflow import check_leg_reuse, old_checkpoint_refusal, prep_made_from, samples_settings

    cfg = Config.model_validate({"system": {"protein": str(tmp_path / "p.pdb"), "ligand": "l.sdf"},
                                 "sampling": {"equilibration_ns": 0.005, "production_ns": 0.01}})
    lig = ligands[0]
    lig_dir = tmp_path / "out" / lig.name
    (lig_dir / "prep").mkdir(parents=True)
    prep_info = {"ligand": lig.name, "complex_barostat": "isotropic", "membrane": False, "box_shape": "dodecahedron",
                 "made_from": prep_made_from(cfg, lig, False, "dodecahedron")}
    hrex = lig_dir / "repeat_0" / "solvent" / "hrex"

    def leg(rows, extra):
        import shutil
        shutil.rmtree(hrex, ignore_errors=True)
        storage = LegStorage(hrex, 2, 2)
        storage.check_conditions({"temperature_K": 298.15, "pressure_bar": 1.0, "friction_per_ps": 1.0,
                                  "settings": samples_settings(cfg)})
        for _ in range(rows):
            storage.append(np.zeros((2, 2)), np.arange(2), np.ones(2))
        storage.write_checkpoint(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), [0, 1], rows, extra)
        storage.close()

    leg(3, {"start": 0})                       # 2 + 4 iterations are planned: it would be continued
    assert "does not record how many of its iterations were stored" in old_checkpoint_refusal(hrex)
    with pytest.raises(
        ConditionsError,
        match="the solvent leg of repeat 0: its checkpoint .* was written by an earlier version.*Nothing was changed",
    ):
        check_leg_reuse(cfg, lig_dir, "solvent", 0, prep_info, 0.01)
    leg(6, {"start": 0})                       # all its iterations, but nothing shows which rows are its own
    with pytest.raises(ConditionsError, match="does not record how many of its iterations were stored"):
        check_leg_reuse(cfg, lig_dir, "solvent", 0, prep_info, 0.01)
    result = hrex.parent / "result.json"
    result.write_text(json.dumps({"n_iterations_stored": 6}))     # the result of its finished run, from these rows
    check_leg_reuse(cfg, lig_dir, "solvent", 0, prep_info, 0.01)   # analysed as it is
    result.unlink()
    leg(3, {"start": 0, "rows": 3})            # a checkpoint of this version is continued
    assert old_checkpoint_refusal(hrex) is None
    check_leg_reuse(cfg, lig_dir, "solvent", 0, prep_info, 0.01)
    # complete: the checkpoint names all the iterations and the samples hold exactly those (review of PR 1:
    # samples stored twice after an interrupt were taken as complete and analysed with their duplicates)
    from cufep.workflow import leg_is_complete
    leg(6, {"start": 0, "rows": 6})
    assert leg_is_complete(hrex, 6) and not leg_is_complete(hrex, 7)
    storage = LegStorage(hrex, 2, 2)
    storage.append(np.zeros((2, 2)), np.arange(2), np.ones(2))
    storage.close()                            # a row the checkpoint does not name
    assert not leg_is_complete(hrex, 6)
    leg(6, {"start": 0})                       # an earlier version: complete only with a result of exactly these rows
    assert not leg_is_complete(hrex, 6)        # review of PR 1: a warm-up and later rows can add up to its iteration
    result.write_text(json.dumps({"n_iterations_stored": 6}))
    assert leg_is_complete(hrex, 6)
    result.write_text(json.dumps({"n_iterations_stored": 4}))
    assert not leg_is_complete(hrex, 6)
    result.unlink()


def test_the_equilibration_of_cufep_analyze_is_given_one_way_and_is_a_length(tmp_path):
    """Review of PR 1: --equilibration-ns 0.05 with --equilibration-iterations 0 used all 120 rows, and
    --equilibration-ns -0.5 was taken for 0."""
    from click.testing import CliRunner

    from cufep.cli import main

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    tail_leg(leg, n_equilibration=20, n=120)
    both = CliRunner().invoke(
        main, ["analyze", str(leg), "--equilibration-ns", "0.05", "--equilibration-iterations", "0"]
    )
    assert both.exit_code == 1 and "either as a length or as a number of iterations, not both" in both.output
    for bad in ("-0.5", "nan", "inf"):
        out = CliRunner().invoke(main, ["analyze", str(leg), "--equilibration-ns", bad])
        assert out.exit_code != 0 and ("nanoseconds of 0 or more" in out.output or "Invalid value" in out.output), bad
    out = json.loads(CliRunner().invoke(main, ["analyze", str(leg), "--equilibration-ns", "0.05"]).output)
    assert out["n_equilibration_iterations"] == 20 and out["n_iterations_used"] == 100


def test_a_complete_leg_is_analysed_with_the_boundary_of_this_run(tmp_path):
    """Review of PR 1 (f515d9a, 1): a leg finished as 20 + 100 iterations, run again as 40 + 80, was taken as
    complete and analysed with the old boundary (100 rows, 1.686020 kJ/mol where 80 rows give 1.677377)."""
    from test_sampling import synthetic_leg

    from cufep.config.schema import Config
    from cufep.sampling.storage import LegStorage
    from cufep.workflow import leg_result_from_samples, take_complete_leg

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    synthetic_leg(leg, n=120, n_equilibration=20)
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc": "none"}})
    run = {"ligand": "lig", "leg": "solvent", "repeat": 0, "backend": "openmm", "notes": []}
    old = leg_result_from_samples(leg, cfg, run)
    assert old["analysis"]["n_iterations_used"] == 100 and old["n_equilibration_iterations"] == 20
    said = []
    take_complete_leg(LegStorage(leg / "hrex", 4, 4), 40, 120, cfg, said.append)
    assert any("the first 40 of its iterations are equilibration in this run (20 in the run that made them)" in line
               for line in said)
    new = leg_result_from_samples(leg, cfg, run)
    assert new["analysis"]["n_iterations_used"] == 80 and new["n_equilibration_iterations"] == 40
    assert new["delta_g_decouple"] != old["delta_g_decouple"]


def test_windows_without_connecting_samples_give_no_free_energy_and_no_binding_free_energy(tmp_path):
    """Review of PR 1 (f515d9a, 2): two windows with no overlap at all (cross energies 1500 kT, 256 iterations)
    gave each leg 0 +- 0, and summary.csv and the benchmark a binding free energy of 0.000 +- 0.000. The whole
    production is tested as every part is: such a leg has no free energy, and the ligand none."""
    from cufep.config.schema import AnalysisConfig
    from cufep.sampling.storage import LegStorage
    from cufep.workflow import combine_ligand, leg_result_from_samples, write_leg_result

    lig = tmp_path / "lig"
    rng = np.random.default_rng(4)
    for level in (0.03, 0.0):
        for name in ("complex", "solvent"):
            leg = lig / f"level_{level}" / "repeat_0" / name
            storage = LegStorage(leg / "hrex", 2, 2)
            for _ in range(256):
                u = np.full((2, 2), 1500.0)
                u[0, 0], u[1, 1] = rng.normal(0.0, 1.0, 2) ** 2
                storage.append(u, np.arange(2), np.ones(2))
            storage.set_meta(temperature_K=298.15, iteration_ps=2.5, backend="openmm", leg=name)
            storage.set_plan(0, 256)
            storage.close()
            run = {"ligand": "lig", "leg": name, "repeat": 0, "backend": "openmm", "notes": [], "pressure_bar": 1.0}
            result = leg_result_from_samples(leg, AnalysisConfig(lrc="none", min_overlap=level), run)
            assert result["delta_g_decouple"] != result["delta_g_decouple"]          # NaN: no free energy
            assert result["analysis"]["not_determined"].startswith("the samples do not connect the lambda windows")
            assert any(note.startswith("NO FREE ENERGY: the samples do not connect") for note in result["notes"])
            write_leg_result(leg, result)
        out = combine_ligand(lig / f"level_{level}", "lig", 1, AnalysisConfig(lrc="none", min_overlap=level))
        assert "dg_bind" not in out and out["failed_legs"] == ["complex leg", "solvent leg"], level


def test_a_run_can_be_restricted_to_ligands_and_legs():
    from cufep.run import select_jobs
    from cufep.workflow import WorkflowError

    names = ["a", "b"]
    assert select_jobs(names, 1) == [("a", 0, "complex"), ("a", 0, "solvent"), ("b", 0, "complex"), ("b", 0, "solvent")]
    assert select_jobs(names, 2, only_ligands=("b",), only_legs=("complex",)) == [
        ("b", 0, "complex"),
        ("b", 1, "complex"),
    ]
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


@pytest.mark.parametrize("water", ["amber14/tip3p.xml", "amber14/opc3.xml", "amber14/spce.xml",
                                   "amber14/tip4pew.xml", "amber14/tip4pfb.xml", "amber14/opc.xml"])
def test_every_water_model_of_the_table_gives_a_system(water):
    """Review of PR 1 (P2): ``forcefield.water: amber14/tip4pew.xml`` was solvated with 3-site waters and
    createSystem found one site too few. A water that comes with the input (a crystal water, 3 sites) gets
    its extra site too."""
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.solvate import SolvationSpec, solvate
    from cufep.prep.water import water_model_of

    ff = app.ForceField("amber14/protein.ff14SB.xml", water)
    seed = app.Modeller(app.Topology(), [])
    seed.addSolvent(app.ForceField("amber14/tip3p.xml"), boxSize=(2.0, 2.0, 2.0) * u.nanometer)
    keep = app.Modeller(seed.topology, seed.positions)
    keep.delete(list(seed.topology.residues())[1:])            # one 3-site crystal water
    solvate(keep, ff, SolvationSpec(padding_nm=1.0, box_shape="cube", ionic_strength_M=0.0, water=water))
    system = ff.createSystem(keep.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    sites = water_model_of(water).sites
    assert system.getNumParticles() == sites * keep.topology.getNumResidues()
    # the reviewer's failure: a 4-site force field with a box of 3-site waters
    if sites == 4:
        wrong = app.Modeller(seed.topology, seed.positions)
        wrong.delete(list(seed.topology.residues())[1:])
        with pytest.raises(Exception, match="missing 1 extra site"):
            wrong.addSolvent(ff, model="tip3p", padding=1.0 * u.nanometer)
            ff.createSystem(wrong.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer)


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


@pytest.mark.parametrize("cutoff,height", [(1.2, 2.592), (0.9, 2.121)])
def test_a_stock_openmm_box_is_at_least_twice_the_cutoff_high(cutoff, height):
    """Review of 037bdf1 (P2): for stock OpenMM a small solute in a dodecahedron with 1.5 nm padding and a 1.2 nm cutoff
    got a box 2.121 nm high, and no Context could be made. With the rule of 2 x cutoff and the margin of 8 % the box
    is 2.592 nm high and the PME energy is computed on the Reference platform; with a 0.9 nm cutoff it stays 2.121
    nm."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.solvate import SolvationSpec, box_height_problem, box_heights_nm, min_box_height_nm, solvate

    ff = app.ForceField("amber14/tip3p.xml")

    def solvated(min_edge):
        seed = app.Modeller(app.Topology(), [])
        seed.addSolvent(ff, boxSize=(2.0, 2.0, 2.0) * u.nanometer)
        keep = app.Modeller(seed.topology, seed.positions)
        keep.delete(list(seed.topology.residues())[1:])            # one water molecule as the solute
        solvate(keep, ff, SolvationSpec(padding_nm=1.5, box_shape="dodecahedron", ionic_strength_M=0.0,
                                        min_edge_nm=min_edge))
        box = np.array([[float(v) for v in vec.value_in_unit(u.nanometer)]
                        for vec in keep.topology.getPeriodicBoxVectors()])
        return keep, box

    def pme_energy(modeller):
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff * u.nanometer)
        context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
        context.setPositions(modeller.positions)
        return context.getState(getEnergy=True).getPotentialEnergy()

    old, old_box = solvated(0.0)                                    # the rule before: none for stock OpenMM
    assert box_heights_nm(old_box).min() == pytest.approx(2.121, abs=1e-3)
    if cutoff > 0.5 * 2.121:
        assert "cannot make a Context" in box_height_problem(old_box, cutoff)
        with pytest.raises(Exception, match="cutoff"):
            pme_energy(old)
    new, new_box = solvated(min_box_height_nm(cutoff, "openmm"))
    assert box_heights_nm(new_box).min() == pytest.approx(height, abs=1e-3)
    assert box_height_problem(new_box, cutoff) is None
    assert np.isfinite(pme_energy(new).value_in_unit(u.kilojoule_per_mole))
    assert min_box_height_nm(cutoff, "cupotential") == pytest.approx(1.08 * 3.0 * cutoff)


def test_a_second_run_in_the_same_output_directory_is_refused(tmp_path):
    from cufep.run import lock_output_directory
    from cufep.workflow import WorkflowError

    first = lock_output_directory(tmp_path)
    with pytest.raises(WorkflowError, match="another cufep run is using the output directory"):
        lock_output_directory(tmp_path)
    first.close()
    lock_output_directory(tmp_path).close()                # free again once the first run has ended


#: what a ligand result records about what it was made of since revision 15 (a result of an earlier version lacks it)
RECORD_OF_THE_COMBINATION = ("repeats_planned", "analysis_record", "leg_digests")


def _write_ligand_legs(lig_dir, dg_per_repeat):
    """Finished legs of a ligand whose repeats give the binding free energies ``dg_per_repeat`` (kcal/mol)."""
    from cufep.workflow import KCAL, write_leg_result

    for r, dg in enumerate(dg_per_repeat):
        for leg, value in (("complex", -dg * KCAL), ("solvent", 0.0)):
            d = lig_dir / f"repeat_{r}" / leg
            d.mkdir(parents=True, exist_ok=True)
            write_leg_result(d, _leg_result(value, leg=leg))


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
@pytest.mark.parametrize("version", ["this", "earlier"])
def test_a_run_of_one_ligand_shows_the_others_as_their_legs_give_them(tmp_path, monkeypatch, ligands, version):
    """Review of PR 1 (P1): ligand B had a result of three repeats (mean -2.0 kcal/mol); a run of ligand A alone with
    one repeat recombined B as one repeat (-1.0). Review of 037bdf1 (P2): once the complex leg of B's last repeat had
    been interrupted, a run of A alone showed B's old mean of three repeats and success again; B's own repeats give
    -1.5 and a failed leg, and after that leg was run again -3.0. In a result of today and of an earlier version, and
    B's
    files are never written."""
    import csv
    from pathlib import Path

    import cufep.run as driver
    from cufep import workflow
    from cufep.config.auto import AutoReport
    from cufep.config.schema import Config

    a, b = ligands[0], ligands[1]
    out = tmp_path / "out"
    cfg = Config.model_validate({"system": {"protein": str(tmp_path / "p.pdb"), "ligand": "l.sdf"},
                                 "output": {"dir": str(out)}, "resources": {"gpus": []},
                                 "solvation": {"box_shape": "dodecahedron"}})

    class Protein:
        def write_pdb(self, path):
            Path(path).write_text("ATOM")

    monkeypatch.setattr(
        driver, "plan", lambda cfg, log, ligands=None: (cfg.model_copy(deep=True), AutoReport(), Protein(), [a, b])
    )
    monkeypatch.setattr(driver, "prepare_ligand", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "run_jobs", lambda jobs, *args, **kwargs: [])
    monkeypatch.setattr(driver, "write_timing", lambda *args, **kwargs: None)
    refreshed = []
    monkeypatch.setattr(
        driver, "refresh_leg_result", lambda leg_dir, *args, **kwargs: refreshed.append(leg_dir) or "current"
    )
    plotted = []
    monkeypatch.setattr(driver, "plot_repeat", lambda repeat_dir, *args, **kwargs: plotted.append(repeat_dir))
    (out / a.name / "prep").mkdir(parents=True)
    (out / a.name / "prep" / "prep.json").write_text(json.dumps({"n_atoms_complex": 10, "n_atoms_solvent": 5}))
    b_dir = out / b.name
    _write_ligand_legs(b_dir, [-1.0, -2.0, -3.0])
    stored = workflow.combine_ligand(b_dir, b.name, 3, cfg)            # B's own run, three repeats
    assert stored["dg_bind"] == pytest.approx(-2.0) and stored["repeats_planned"] == 3
    if version == "earlier":
        (b_dir / "result.json").write_text(
            json.dumps({k: v for k, v in stored.items() if k not in RECORD_OF_THE_COMBINATION})
        )

    def b_files():
        return {
            p.relative_to(b_dir): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(b_dir.rglob("*"))
            if p.is_file()
        }

    def run_a():
        before = b_files()
        summary = driver._run(cfg, out, False, lambda m: None, only_ligands=[a.name])
        assert b_files() == before                                   # nothing of B is written
        entry = next(s for s in summary if s["ligand"] == b.name)
        assert (
            entry["not_part_of_this_run"] is True and entry["n_repeats_planned"] == 3 and entry["selected_legs"] == []
        )
        assert json.loads((out / "summary.json").read_text())[1].get("dg_bind") == pytest.approx(entry.get("dg_bind"))
        row = next(r for r in csv.DictReader((out / "summary.csv").open()) if r["ligand"] == b.name)
        return entry, row

    entry, row = run_a()
    assert (
        entry["dg_bind"] == pytest.approx(-2.0)
        and entry["n_repeats"] == 3
        and not entry.get("combined_again_from_legs")
    )
    assert row["dg_bind_kcal_mol"] == "-2.000" and row["n_repeats"] == "3"

    workflow.begin_leg(b_dir / "repeat_2" / "complex")  # B's last complex leg starts again and is interrupted
    entry, row = run_a()
    assert entry["dg_bind"] == pytest.approx(-1.5) and entry["n_repeats"] == 2 and entry["combined_again_from_legs"]
    assert entry["failed_leg_ids"] == [[2, "complex"]] and entry["warnings"][0].startswith("not part of this run")
    assert row["dg_bind_kcal_mol"] == "-1.500" and row["n_repeats"] == "2"
    assert driver.failed_legs(json.loads((out / "summary.json").read_text())) == []   # A's run did not fail

    from cufep.workflow import KCAL
    workflow.write_leg_result(b_dir / "repeat_2" / "complex", _leg_result(6.0 * KCAL))   # run again: -6.0
    entry, row = run_a()
    assert entry["dg_bind"] == pytest.approx(-3.0) and entry["n_repeats"] == 3 and entry["failed_leg_ids"] == []
    assert row["dg_bind_kcal_mol"] == "-3.000"
    assert all(b.name not in str(d) for d in refreshed + plotted)


def test_a_ligand_not_part_of_a_run_keeps_its_own_repeats_and_analysis_settings(tmp_path):
    """The read-only combination of a ligand that a run does not include uses the repeats and the analysis record
    of the run that combined it, not those of the run in progress; one without a result of its own is combined over
    the repeats it has."""
    from cufep import workflow
    from cufep.config.schema import Config

    lig = tmp_path / "b"
    _write_ligand_legs(lig, [-1.0, -2.0, -3.0, -4.0])
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"min_overlap": 0.05}})
    stored = workflow.combine_ligand(lig, "b", 3, cfg)                 # repeat 3 is not part of B's result
    assert stored["dg_bind"] == pytest.approx(-2.0)
    assert workflow.stored_ligand_result(lig, "b")["dg_bind"] == pytest.approx(-2.0)
    workflow.write_leg_result(lig / "repeat_0" / "solvent", _leg_result(0.0, leg="solvent"))   # the same result again
    assert not workflow.stored_ligand_result(lig, "b").get("combined_again_from_legs")
    workflow.write_leg_result(
        lig / "repeat_0" / "solvent", dict(_leg_result(0.0, leg="solvent"), analysis_record={"version": 1})
    )
    again = workflow.stored_ligand_result(lig, "b")
    assert again["combined_again_from_legs"] and again["n_repeats"] == 3 and again["repeats_planned"] == 3
    assert again["analysis_record"]["min_overlap"] == 0.05             # B's settings, not the defaults
    assert any("version 1 of the analysis" in item for item in again["old_analysis"])
    (lig / "result.json").unlink()
    alone = workflow.stored_ligand_result(lig, "b")
    assert alone["n_repeats"] == 4 and alone["dg_bind"] == pytest.approx(-2.5)
    assert "did not combine them" in alone["warnings"][0]
    assert workflow.stored_ligand_result(tmp_path / "none", "x")["warnings"] == [
        "not part of this run, and no result of an earlier run"]


@pytest.mark.parametrize("version", ["this", "earlier"])
def test_a_ligand_not_part_of_a_run_shows_the_verdicts_of_this_version(tmp_path, version):
    """Review of 7b49066 (P2): a ligand B combined by analysis version 8 whose legs had not changed was shown as stored
    in a run of ligand A alone: the summary listed B's legs under the verdicts of that version and the log carried
    its warnings on the halves and BAR, checks that are no longer made. A stored result of another version is combined
    again, read-only, with its repeats and analysis settings: the same free energy, the checks of this version, and
    none of B's files written. The same for a result of an earlier version without a record of its combination."""
    from cufep import workflow
    from cufep.analysis.mbar import ANALYSIS_VERSION
    from cufep.config.schema import Config

    lig = tmp_path / "b"
    _write_ligand_legs(lig, [-1.0, -2.0, -3.0])
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}})
    stored = workflow.combine_ligand(lig, "b", 3, cfg)
    old_verdicts = ["repeat 0 complex: NOT CONVERGED: the two halves of the production differ by 2.5 kT",
                    "repeat 1 solvent: BAR between neighbouring lambda windows and MBAR differ by 1.2 kcal/mol"]
    old = dict(stored, warnings=stored["warnings"] + old_verdicts, not_converged=["complex leg of repeat 0"],
               analysis_record=dict(stored["analysis_record"], version=8, halves_tolerance_kT=1.0))
    if version == "earlier":
        old = {k: v for k, v in old.items() if k not in RECORD_OF_THE_COMBINATION}
    (lig / "result.json").write_text(json.dumps(old))
    before = {p: p.read_bytes() for p in sorted(lig.rglob("*")) if p.is_file()}
    shown = workflow.stored_ligand_result(lig, "b")
    assert shown["dg_bind"] == pytest.approx(-2.0) and shown["n_repeats"] == 3 and shown["not_part_of_this_run"]
    assert not any(w in shown["warnings"] for w in old_verdicts) and "not_converged" not in shown
    assert {p: p.read_bytes() for p in sorted(lig.rglob("*")) if p.is_file()} == before     # nothing of B written
    if version == "this":
        assert shown["analysis_record"]["version"] == 8 and shown["repeats_planned"] == 3   # B's own settings
    # a result of this version with unchanged legs is shown as it is stored
    (lig / "result.json").write_text(json.dumps(dict(stored, warnings=stored["warnings"] + ["a stored note"])))
    assert stored["analysis_record"]["version"] == ANALYSIS_VERSION
    assert "a stored note" in workflow.stored_ligand_result(lig, "b")["warnings"]


def test_a_new_result_replaces_the_old_one_only_when_it_is_complete(tmp_path, monkeypatch):
    """Review of PR 1 (P2): the result was moved aside before the new one was written; a failed write left the leg
    with none, and later runs read it as not finished."""
    from cufep import workflow
    from cufep.config.schema import AnalysisConfig

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    (leg / "hrex").mkdir(parents=True)
    (leg / "hrex" / "samples.h5").write_text("")
    old = {"analysis_record": {"version": 0}, "delta_g_decouple": 1.0}
    (leg / "result.json").write_text(json.dumps(old))
    monkeypatch.setattr(workflow, "leg_result_from_samples", lambda *args, **kwargs: {"delta_g_decouple": 2.0})
    monkeypatch.setattr(workflow, "run_record_of", lambda *args, **kwargs: {})
    real = workflow._write_json

    def full_disk(path, data):
        if path.name == "result.json":
            raise OSError(28, "No space left on device")
        real(path, data)

    monkeypatch.setattr(workflow, "_write_json", full_disk)
    with pytest.raises(OSError, match="No space"):
        workflow.refresh_leg_result(leg, AnalysisConfig(), lambda m: None)
    assert json.loads((leg / "result.json").read_text()) == old              # still the leg's result
    monkeypatch.setattr(workflow, "_write_json", real)
    assert workflow.refresh_leg_result(leg, AnalysisConfig(), lambda m: None) == "reanalysed"
    assert json.loads((leg / "result.json").read_text()) == {"delta_g_decouple": 2.0}
    assert json.loads((leg / "result.replaced.json").read_text()) == old


def test_a_repeat_without_a_current_leg_keeps_no_old_figure(tmp_path):
    """Review of PR 1 (P2): after both legs of a repeat failed, analysis.png of the earlier success stayed."""
    from cufep.analysis.plots import plot_repeat

    (tmp_path / "analysis.png").write_bytes(b"\x89PNG old")
    assert plot_repeat(tmp_path) is None and not (tmp_path / "analysis.png").exists()


def test_validate_takes_the_lock_of_the_output_directory(tmp_path, monkeypatch):
    """Review of PR 1 (P1): `cufep validate` made the prepared systems of a directory while `cufep run` was making
    them there."""
    from cufep import run as driver
    from cufep import validate as validation
    from cufep import workflow
    from cufep.backends.cupotential import CuPotentialBackend
    from cufep.config.schema import Config

    class Ligand:
        name = "lig"

    cfg = Config.model_validate(
        {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "output": {"dir": str(tmp_path / "out")}}
    )
    monkeypatch.setattr(CuPotentialBackend, "available", staticmethod(lambda: (True, "")))
    monkeypatch.setattr(workflow, "ligands_of_run", lambda cfg, names=(): [Ligand()])
    monkeypatch.setattr(driver, "plan", lambda cfg, echo=print, ligands=None: (cfg, None, None, ligands))
    (tmp_path / "out").mkdir()
    held = driver.lock_output_directory(tmp_path / "out")
    try:
        with pytest.raises(workflow.WorkflowError, match="another cufep run is using the output directory"):
            validation.validate(cfg, None, 0, lambda m: None)
    finally:
        held.close()
