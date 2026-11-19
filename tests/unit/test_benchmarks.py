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


def test_a_summary_of_an_earlier_version_is_not_shown_as_if_it_had_passed_the_checks(tmp_path):
    """Review of PR 1: `benchmark report` on a summary.json of an earlier version (not_converged:
    ["complex leg"], convergence_not_assessed: ["solvent leg"]) gave empty flags in the table and the CSV:
    that version's verdicts were dropped and the checks of this version, which it had not made, looked passed."""
    from types import SimpleNamespace

    from cufep.benchmarks.runner import LEG_FLAGS, NOT_CHECKED, flags_of
    from cufep.run import result_line, write_summary

    assert NOT_CHECKED == "checks of this version not made"
    earlier_version = f"{NOT_CHECKED} (analysed by an earlier version of cuFEP)"
    # as the versions before the corrected analysis wrote it
    earlier = {"ligand": "jmc_23", "units": "kcal/mol", "dg_bind": -11.0, "dg_bind_std": None, "dg_bind_mbar_error": 0.3,
               "n_repeats": 1, "warnings": [], "failed_legs": [], "not_reliable": [], "not_converged": ["complex leg"],
               "convergence_not_assessed": ["solvent leg"], "estimators_disagree": [], "min_overlap": 0.08,
               "repeats": [{"lambda_windows": {"complex": 16, "solvent": 13}}]}
    said = ["NOT CONVERGED by the check of an earlier version: complex leg",
            "convergence not assessed by an earlier version: solvent leg", earlier_version]
    assert flags_of(earlier) == said
    (tmp_path / "summary.json").write_text(json.dumps([earlier]))
    text = report("tyk2", tmp_path)
    assert f"| 16 | 13 | 0.080 |  | {'; '.join(said)} |" in text
    assert "A result that an earlier version of cuFEP analysed carries the verdicts of that version" in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert list(csv.DictReader(fh))[0]["flags"] == "; ".join(said)
    assert result_line(earlier).endswith("(MBAR uncertainty, 1 repeat); " + "; ".join(said))
    # a result of the first versions, with no key of any check, and one of the analysis before the time blocks
    oldest = {"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_std": None, "dg_bind_mbar_error": 0.3, "n_repeats": 1}
    assert flags_of(oldest) == [earlier_version]
    before_blocks = dict(oldest, few_samples=["complex leg"], halves_differ=[], estimators_disagree=[], not_reliable=[])
    assert flags_of(before_blocks) == ["few independent samples: complex leg", earlier_version]
    # a result of this version with nothing to say has no flag, and a ligand without a result is not "checked" either
    from cufep.analysis.mbar import ANALYSIS_VERSION
    current = dict(before_blocks, few_samples=[], dg_bind_error=0.5, dg_bind_error_from_time_blocks=True,
                   analysis_versions=[ANALYSIS_VERSION])
    assert flags_of(current) == [] and flags_of({"ligand": "x", "n_repeats": 0, "warnings": ["not run"]}) == []
    # the keys of earlier versions are no columns of the table this version writes
    write_summary(tmp_path, [current], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        columns = list(csv.DictReader(fh).fieldnames)
    assert "not_converged" not in columns and all(key in columns for key, _name in LEG_FLAGS)


def test_the_report_shows_the_standard_error_of_one_calculation(tmp_path):
    """Review of PR 1: with dg_bind -11.00, dg_bind_error 0.75 and one repeat the Markdown table and the CSV of
    the benchmark report had no error and the figure an error bar of 0."""
    from cufep.benchmarks.runner import error_bar, error_bar_label

    one = {"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_error": 0.75, "dg_bind_std": None, "n_repeats": 1,
           "warnings": [], "min_overlap": 0.08}
    older = {"ligand": "jmc_30", "dg_bind": -10.5, "dg_bind_std": None, "dg_bind_mbar_error": 0.3, "n_repeats": 1,
             "warnings": [], "min_overlap": 0.08}
    three = {"ligand": "ejm_31", "dg_bind": -9.0, "dg_bind_error": 0.5, "dg_bind_std": 1.2, "n_repeats": 3,
             "warnings": [], "min_overlap": 0.08}
    (tmp_path / "summary.json").write_text(json.dumps([one, older, three]))
    text = report("tyk2", tmp_path)
    assert "| ligand | experiment | cuFEP | standard error | SD over repeats | repeats | FEP+ |" in text
    assert "| -11.00 | 0.75 |  | 1 |" in text and "| -10.50 | - |  | 1 |" in text and "| -9.00 | 0.50 | 1.20 | 3 |" in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        rows = {row["ligand"]: row for row in csv.DictReader(fh)}
    assert rows["jmc_23"]["cufep_error"] == "0.75" and rows["jmc_23"]["cufep_std"] == ""
    assert rows["jmc_30"]["cufep_error"] == "" and rows["ejm_31"]["cufep_std"] == "1.2"
    bars = {r["ligand"]: error_bar(r) for r in rows.values() for r in [dict(r, n_repeats=int(r["n_repeats"]),
            cufep_error=float(r["cufep_error"]) if r["cufep_error"] else None,
            cufep_std=float(r["cufep_std"]) if r["cufep_std"] else None)]}
    assert bars == {"jmc_23": 0.75, "jmc_30": 0.0, "ejm_31": 1.2}
    assert error_bar_label([{"n_repeats": 1, "cufep_error": 0.75}, {"n_repeats": 1, "cufep_error": None}]) == \
        "error bars: none recorded, standard error"


def test_a_summary_names_the_version_of_the_analysis_of_its_legs():
    """Review of PR 1: a summary of version 4 has dg_bind_error, which was taken for the mark of this version:
    its results were shown as checked, though version 4 could not see the halves of the corrected value."""
    from cufep.analysis.mbar import ANALYSIS_VERSION
    from cufep.benchmarks.runner import NOT_CHECKED, flags_of

    base = {"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_error": 0.5, "dg_bind_std": None, "n_repeats": 1}
    assert flags_of(dict(base)) == [f"{NOT_CHECKED} (analysed by an earlier version of cuFEP)"]
    assert flags_of(dict(base, analysis_versions=[4])) == [f"{NOT_CHECKED} (analysed by version 4 of the analysis)"]
    assert flags_of(dict(base, analysis_versions=[None, ANALYSIS_VERSION])) == [
        f"{NOT_CHECKED} (analysed by versions not recorded, {ANALYSIS_VERSION} of the analysis)"]
    assert flags_of(dict(base, analysis_versions=[ANALYSIS_VERSION])) == []


def test_the_figure_shows_a_prediction_far_from_experiment_with_its_whole_error_bar():
    """Review of PR 1: ejm_31, experiment -9.54, prediction -13.00 +- 1.25 kcal/mol: the axes were set from
    the experimental values ([-12.04, -7.04]) and the point and its error bar were outside."""
    from cufep.benchmarks.runner import figure_range

    rows = [{"exp": -9.54, "fepplus": -9.2, "cufep": -13.0, "cufep_error": 1.25, "cufep_std": None, "n_repeats": 1}]
    lo, hi = figure_range(rows)
    assert lo <= -14.25 and hi >= -9.2 and lo <= -9.54 <= hi


def test_every_check_reaches_the_report_the_summary_and_the_result_line(tmp_path):
    """Review of PR 1: a leg whose only failed check was BAR against MBAR was named in summary.json and
    nowhere a person looks: not in the flags of the report, not in summary.csv, not in the result line."""
    from types import SimpleNamespace

    from cufep.benchmarks.runner import LEG_FLAGS, flags_of
    from cufep.run import result_line, write_summary

    from cufep.analysis.mbar import ANALYSIS_VERSION
    only_estimators = {"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_std": None, "dg_bind_mbar_error": 0.3,
                       "dg_bind_error": 0.5, "dg_bind_error_from_time_blocks": True, "analysis_versions": [ANALYSIS_VERSION],
                       "n_repeats": 1, "warnings": ["repeat 0 complex: BAR between neighbouring lambda windows gives ..."],
                       "failed_legs": [], "not_reliable": [], "few_samples": [], "halves_differ": [],
                       "halves_not_compared": [], "estimators_disagree": ["complex leg"], "min_overlap": 0.08}
    assert flags_of(only_estimators) == ["BAR and MBAR differ: complex leg"]
    (tmp_path / "summary.json").write_text(json.dumps([only_estimators]))
    assert "| BAR and MBAR differ: complex leg |" in report("tyk2", tmp_path)
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert list(csv.DictReader(fh))[0]["flags"] == "BAR and MBAR differ: complex leg"
    assert result_line(only_estimators) == ("RESULT jmc_23: dG_bind = -11.00 ± 0.50 kcal/mol (standard error from time "
                                            "blocks, 1 repeat); BAR and MBAR differ: complex leg; 1 warning(s):")
    write_summary(tmp_path, [only_estimators], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["estimators_disagree"] == "complex leg" and row["few_samples"] == ""
    assert all(key in row for key, _name in LEG_FLAGS)                       # one column per thing that can be said
    assert row["mbar_error"] == "0.300" and row["standard_error"] == "0.500"
    # the standard error of one calculation is the one from time blocks; that of MBAR stays in its column
    blocks = dict(only_estimators, dg_bind_error=0.9, dg_bind_error_from_time_blocks=True)
    assert result_line(blocks).startswith("RESULT jmc_23: dG_bind = -11.00 ± 0.90 kcal/mol (standard error from time "
                                          "blocks, 1 repeat); BAR and MBAR differ")
    assert " ± 0.90 kcal/mol (standard error, 1 repeat);" in result_line(dict(blocks, dg_bind_error_from_time_blocks=False))
    write_summary(tmp_path, [blocks], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["standard_error"] == "0.900" and row["mbar_error"] == "0.300"

    # all of them at once, in the order of their weight; the overlap is given with the number
    everything = dict(only_estimators, failed_legs=["solvent leg of repeat 1"], not_reliable=["complex leg"],
                      inconsistent_legs=[], old_analysis=["solvent leg: version 2 of the analysis; this run uses version 3"],
                      few_samples=["complex leg"], halves_differ=["complex leg"], halves_not_compared=["solvent leg"],
                      halves_below_level=["complex leg"],
                      origin_not_recorded=["complex leg: the samples"], min_overlap=0.02, min_overlap_used=0.012)
    assert flags_of(everything) == [
        "FAILED: solvent leg of repeat 1", "NOT RELIABLE (overlap): complex leg", "old analysis: solvent leg",
        "few independent samples: complex leg", "halves differ: complex leg", "halves not compared: solvent leg",
        "halves compared below the level of overlap: complex leg",
        "BAR and MBAR differ: complex leg", "origin of files not recorded: complex leg"]
    assert "; NOT RELIABLE (overlap): complex leg (neighbouring lambda windows overlap by only 0.012);" in result_line(everything)
    nothing = {"ligand": "x", "n_repeats": 0, "warnings": []}
    assert result_line(nothing) == "RESULT x: no binding free energy; no complete repeat"
    assert result_line(dict(nothing, inconsistent_legs=["temperature (K): complex leg 315.0, solvent leg 298.15"])) \
        == ("RESULT x: no binding free energy; LEGS MADE UNDER DIFFERENT CONDITIONS: temperature (K): complex leg "
            "315.0, solvent leg 298.15")
