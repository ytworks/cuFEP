"""Benchmark registry, statistics and report; structural analysis."""
import csv
import json

import numpy as np
import pytest

from cufep.analysis.structure import structural_analysis
from cufep.benchmarks.curate import compound_id, evenly_spaced_subset
from cufep.benchmarks.runner import benchmark_config, data_root, load_registry, read_reference, report
from cufep.benchmarks.stats import bootstrap, metrics
from cufep.prep.ligand import load_ligands


def test_metrics_on_known_numbers():
    exp = np.array([-10.0, -9.0, -8.0, -7.0])
    pred = exp + 2.0                                   # perfect ranking, constant offset
    m = metrics(pred, exp)
    assert m["mean_signed_error"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)
    assert m["centred_rmse"] == pytest.approx(0.0, abs=1e-12) and m["pairwise_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["kendall_tau"] == pytest.approx(1.0) and m["spearman"] == pytest.approx(1.0)
    m = metrics(exp[::-1], exp)
    assert m["kendall_tau"] == pytest.approx(-1.0)
    # pairwise RMSE equals sqrt(2 N / (N - 1)) times the centred RMSE
    rng = np.random.default_rng(0)
    pred = exp + rng.normal(size=4)
    m = metrics(pred, exp)
    assert m["pairwise_rmse"] == pytest.approx(m["centred_rmse"] * np.sqrt(2 * 4 / 3))


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    exp = rng.uniform(-12, -7, 16)
    pred = exp + rng.normal(0.5, 0.8, 16)
    m, ci = metrics(pred, exp), bootstrap(pred, exp, n_boot=400, seed=2)
    for key in ("rmse", "mue", "centred_rmse", "kendall_tau"):
        assert ci[key][0] <= m[key] <= ci[key][1]


def test_registry_and_curated_files_are_consistent():
    registry = load_registry()
    assert set(registry) == {"tyk2", "a2a"}
    for name, n in (("tyk2", 16), ("a2a", 17)):
        entry, ref = registry[name], read_reference(name)
        assert len(ref) == n
        ligands = load_ligands(data_root() / entry["ligands"])
        assert sorted(l.name for l in ligands) == sorted(ref)
        assert all(l.formal_charge == 0 for l in ligands)
        for lig in ligands:
            assert float(lig.properties["r_exp_dg"]) == pytest.approx(ref[lig.name]["exp"], abs=2e-3)
        with open(data_root() / entry["reference"]) as fh:
            rows = list(csv.DictReader(fh))
        assert entry["subset"] == evenly_spaced_subset(rows)


def test_compound_names():
    assert compound_id("4b D") == "4b" and compound_id("4jS") == "4j" and compound_id("4kU") == "4k"
    assert compound_id("4g") == "4g" and compound_id("ejm_31") == "ejm_31"


def test_benchmark_tiers(tmp_path):
    smoke = benchmark_config("tyk2", "smoke", tmp_path / "s")
    assert smoke.system.ligand_names == ["jmc_23"] and smoke.sampling.repeats == 1
    assert smoke.sampling.production_ns == 0.1 and smoke.lambda_schedule.adaptive.window_ps == 10
    assert smoke.lambda_schedule.adaptive.equilibration("complex") == 0.0
    subset = benchmark_config("a2a", "subset", tmp_path / "x")
    assert subset.system.ligand_names == load_registry()["a2a"]["subset"]
    assert subset.sampling.repeats == 1 and subset.membrane.enabled is True     # one run per ligand, as FEP+
    full = benchmark_config("tyk2", "full")
    assert full.system.ligand_names is None and full.sampling.production_ns == 10.0
    assert full.membrane.enabled is False and smoke.membrane.enabled is False
    assert benchmark_config("a2a", "smoke").membrane.enabled is True
    with pytest.raises(KeyError):
        benchmark_config("nope", "full")
    with pytest.raises(KeyError):
        benchmark_config("tyk2", "huge")


def test_report_from_a_synthetic_run(tmp_path):
    ref = read_reference("tyk2")
    rng = np.random.default_rng(4)
    summary = [{"ligand": name, "dg_bind": v["exp"] - 1.5 + rng.normal(0, 0.6), "dg_bind_std": 0.4, "n_repeats": 3,
                "warnings": []} for name, v in ref.items()]
    summary.append({"ligand": "unfinished", "n_repeats": 0, "warnings": ["repeat 0 is incomplete"]})
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = report("tyk2", tmp_path)
    assert "Ligands with a result: 16 of 17 run (16 in the set)." in text
    assert "| mean signed error (kcal/mol) | -1." in text and "0 by construction" not in text
    # over the whole curated TYK2 set the FEP+ values have no mean error ...
    assert [line.split("|")[3].split("[")[0].strip() for line in text.splitlines() if line.startswith("| mean signed")] == ["0.00"]
    # ... over the five ligands of the subset they have (review of PR 1: the table said "0 by construction")
    subset = {"jmc_23", "jmc_30", "ejm_45", "ejm_48", "ejm_44"}
    part = tmp_path / "subset"
    part.mkdir()
    (part / "summary.json").write_text(json.dumps([s for s in summary if s["ligand"] in subset]))
    text = report("tyk2", part)
    expected = np.mean([ref[name]["fepplus"] - ref[name]["exp"] for name in subset])
    assert expected == pytest.approx(0.244, abs=0.001)
    assert [line.split("|")[3].split("[")[0].strip() for line in text.splitlines() if line.startswith("| mean signed")] == ["0.24"]
    assert "over a part of that set their mean signed error is not zero" in text
    assert (tmp_path / "benchmark_report.md").exists() and (tmp_path / "benchmark_report.csv").exists()
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 16


def _samples(shift):
    rng = np.random.default_rng(0)
    ca = rng.normal(scale=1.0, size=(30, 3))
    lig = rng.normal(scale=0.2, size=(8, 3)) + np.array([0.3, 0.0, 0.0])
    frames = []
    for f in range(12):
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        moved = lig + (shift * f / 11.0)
        xyz = np.vstack([ca, moved]) + rng.normal(scale=0.005, size=(38, 3))
        frames.append(xyz @ rot.T + rng.normal(size=3))            # arbitrary rigid motion of the whole complex
    return {"frames": np.array(frames, dtype=np.float32), "frame_iter": np.arange(12) * 4,
            "attrs": {"frame_ca": list(range(30)), "frame_ligand": list(range(30, 38))}}


def test_structural_analysis_ignores_tumbling_and_flags_a_leaving_ligand():
    still = structural_analysis(_samples(np.zeros(3)))
    assert still["ligand_rmsd_max_A"] < 0.5 and still["warnings"] == []
    gone = structural_analysis(_samples(np.array([1.2, 0.0, 0.0])))
    assert gone["ligand_com_drift_last_half_A"] > 5.0 and len(gone["warnings"]) == 2
    assert structural_analysis({"attrs": {}}) is None


def test_a_report_is_that_of_the_run_as_it_is_now(tmp_path):
    """Review of PR 1: after every extension had failed, the report, table and figure of the earlier,
    successful state still showed the old binding free energy."""
    ref = read_reference("tyk2")
    good = {"ligand": "jmc_23", "dg_bind": -12.34, "dg_bind_std": None, "n_repeats": 1, "warnings": []}
    (tmp_path / "summary.json").write_text(json.dumps([good]))
    assert "-12.34" in report("tyk2", tmp_path)
    assert "-12.34" in (tmp_path / "benchmark_report.csv").read_text()
    (tmp_path / "benchmark_report.png").write_text("the figure of that state")      # also where no figure can be made

    failed = {"ligand": "jmc_23", "n_repeats": 0, "failed_legs": ["complex leg"],
              "warnings": ["repeat 0 complex: FAILED: the GPU was lost.", "repeat 0 is incomplete"]}
    (tmp_path / "summary.json").write_text(json.dumps([failed]))
    text = report("tyk2", tmp_path)
    assert "Ligands with a result: 0 of 1 run" in text and "No ligand has a binding free energy." in text
    assert f"| jmc_23 | {ref['jmc_23']['exp']:.2f} | FAILED: complex leg |" in text
    assert (tmp_path / "benchmark_report.md").read_text() == text and "-12.34" not in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["cufep"] == "" and rows[0]["flags"] == "FAILED: complex leg"
    assert not (tmp_path / "benchmark_report.png").exists()

    # some ligands with a result and some without: both are in the report and in the table
    (tmp_path / "summary.json").write_text(json.dumps([
        failed, dict(good, ligand="jmc_30"), {"ligand": "ejm_45", "n_repeats": 0, "warnings": []}]))
    text = report("tyk2", tmp_path)
    assert "Ligands with a result: 1 of 3 run" in text and "## Ligands without a binding free energy" in text
    assert "| ejm_45 |" in text and "no complete repeat" in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert [row["ligand"] for row in csv.DictReader(fh)] == ["jmc_30", "jmc_23", "ejm_45"]


def test_every_check_reaches_the_report_the_summary_and_the_result_line(tmp_path):
    """Review of PR 1: a leg whose only failed check was BAR against MBAR was named in summary.json and
    nowhere a person looks: not in the flags of the report, not in summary.csv, not in the result line."""
    from types import SimpleNamespace

    from cufep.benchmarks.runner import LEG_FLAGS, flags_of
    from cufep.run import result_line, write_summary

    only_estimators = {"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_std": None, "dg_bind_mbar_error": 0.3,
                       "n_repeats": 1, "warnings": ["repeat 0 complex: BAR between neighbouring lambda windows gives ..."],
                       "failed_legs": [], "not_reliable": [], "few_samples": [], "halves_differ": [],
                       "halves_not_compared": [], "estimators_disagree": ["complex leg"], "min_overlap": 0.08}
    assert flags_of(only_estimators) == ["BAR and MBAR differ: complex leg"]
    (tmp_path / "summary.json").write_text(json.dumps([only_estimators]))
    assert "| BAR and MBAR differ: complex leg |" in report("tyk2", tmp_path)
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert list(csv.DictReader(fh))[0]["flags"] == "BAR and MBAR differ: complex leg"
    assert result_line(only_estimators) == ("RESULT jmc_23: dG_bind = -11.00 ± 0.30 kcal/mol (MBAR uncertainty, 1 repeat)"
                                            "; BAR and MBAR differ: complex leg; 1 warning(s):")
    write_summary(tmp_path, [only_estimators], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["estimators_disagree"] == "complex leg" and row["few_samples"] == ""
    assert all(key in row for key, _name in LEG_FLAGS)                       # one column per thing that can be said

    # all of them at once, in the order of their weight; the overlap is given with the number
    everything = dict(only_estimators, failed_legs=["solvent leg of repeat 1"], not_reliable=["complex leg"],
                      inconsistent_legs=[], old_analysis=["solvent leg: version 2 of the analysis; this run uses version 3"],
                      few_samples=["complex leg"], halves_differ=["complex leg"], halves_not_compared=["solvent leg"],
                      origin_not_recorded=["complex leg: the samples"], min_overlap=0.02, min_overlap_used=0.012)
    assert flags_of(everything) == [
        "FAILED: solvent leg of repeat 1", "NOT RELIABLE (overlap): complex leg", "old analysis: solvent leg",
        "few independent samples: complex leg", "halves differ: complex leg", "halves not compared: solvent leg",
        "BAR and MBAR differ: complex leg", "origin of files not recorded: complex leg"]
    assert "; NOT RELIABLE (overlap): complex leg (neighbouring lambda windows overlap by only 0.012);" in result_line(everything)
    nothing = {"ligand": "x", "n_repeats": 0, "warnings": []}
    assert result_line(nothing) == "RESULT x: no binding free energy; no complete repeat"
    assert result_line(dict(nothing, inconsistent_legs=["temperature (K): complex leg 315.0, solvent leg 298.15"])) \
        == ("RESULT x: no binding free energy; LEGS MADE UNDER DIFFERENT CONDITIONS: temperature (K): complex leg "
            "315.0, solvent leg 298.15")
