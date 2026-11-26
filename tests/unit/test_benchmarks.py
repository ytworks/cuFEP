"""Benchmark registry, statistics and report; structural analysis."""
import csv
import json

import numpy as np
import pytest

from cufep.analysis.structure import structural_analysis
from cufep.benchmarks.curate import compound_id, evenly_spaced_subset
from cufep.benchmarks.runner import (
    benchmark_config,
    data_root,
    example_notices,
    load_registry,
    overlap_places,
    read_reference,
    report,
)
from cufep.benchmarks.stats import bootstrap, metrics
from cufep.config.schema import load_config
from cufep.prep.ligand import load_ligands


def test_a_benchmark_run_says_of_the_file_of_its_example_what_cufep_run_says(tmp_path, monkeypatch):
    """Review of the defaults of 2026-11-22 (Codex): a benchmark reads the configuration of its
    example and dropped the notices about it, so an example with a threshold and no overlap matrix
    was read otherwise than by earlier versions without a word."""
    import yaml
    from click.testing import CliRunner

    from cufep import cli
    from cufep import run as driver

    assert example_notices("tyk2")[1] == [] and example_notices("a2a")[1] == []
    relative = load_registry()["tyk2"]["example"]
    settings = yaml.safe_load((data_root() / relative).read_text())
    settings["lambda_schedule"] = {"adaptive": {"overlap_threshold": 0.18}}
    (tmp_path / "benchmarks").mkdir()
    example = tmp_path / relative
    example.parent.mkdir(parents=True)
    example.write_text(yaml.safe_dump(settings))
    monkeypatch.setenv("CUFEP_DATA", str(tmp_path))
    path, notices = example_notices("tyk2")
    assert path == example and len(notices) == 1
    assert notices[0].startswith("lambda_schedule.adaptive.overlap_threshold is given without overlap_matrix: in "
                                 "the complex leg the threshold (0.18) is compared with the overlap of the two windows")

    def not_run(cfg):
        raise RuntimeError("not run in this test")

    monkeypatch.setattr(driver, "run", not_run)
    result = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--tier", "smoke", "-o", str(tmp_path / "out")])
    printed = result.output + (result.stderr if getattr(result, "stderr_bytes", None) is not None else "")
    assert result.exit_code != 0 and "not run in this test" in printed
    assert f"note: {example}: lambda_schedule.adaptive.overlap_threshold is given without overlap_matrix" in printed


def test_metrics_on_known_numbers():
    exp = np.array([-10.0, -9.0, -8.0, -7.0])
    pred = exp + 2.0                                   # perfect ranking, constant offset
    m = metrics(pred, exp)
    assert m["mean_signed_error"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)
    assert m["centred_rmse"] == pytest.approx(0.0, abs=1e-12) and m["pairwise_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["kendall_tau"] == pytest.approx(1.0) and m["spearman"] == pytest.approx(1.0)
    assert m["pearson_r"] == pytest.approx(1.0) and m["pearson_r2"] == pytest.approx(1.0)
    m = metrics(exp[::-1], exp)
    assert m["kendall_tau"] == pytest.approx(-1.0) and m["pearson_r"] == pytest.approx(-1.0)
    assert m["pearson_r2"] == pytest.approx(1.0)
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


@pytest.mark.needs("openff.toolkit", "rdkit", "pdbfixer")
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
    assert smoke.sampling.production_ns == 0.1
    assert smoke.lambda_schedule.adaptive.window("complex") == smoke.lambda_schedule.adaptive.window("solvent") == 10
    assert smoke.lambda_schedule.adaptive.equilibration("complex") == 0.0
    a2a_smoke = benchmark_config("a2a", "smoke", tmp_path / "m")
    assert a2a_smoke.membrane.enabled is True and a2a_smoke.membrane.equilibration_ns == 0.5
    assert benchmark_config("a2a", "subset", tmp_path / "x").system.ligand_names == load_registry()["a2a"]["subset"]
    assert benchmark_config("tyk2", "full").system.ligand_names is None
    with pytest.raises(KeyError):
        benchmark_config("nope", "full")
    with pytest.raises(KeyError):
        benchmark_config("tyk2", "huge")


def _sdf_records(path) -> dict[str, str]:
    return {r.split("\n")[0].strip(): r for r in path.read_text().split("$$$$\n") if r.strip()}


@pytest.mark.parametrize("name", ["tyk2", "a2a"])
def test_a_benchmark_set_runs_with_the_settings_of_its_example(name, tmp_path):
    """A set runs as its example runs: the same settings, only the input files and the output
    directory differ; and the example is a member of the set (the same protein file, its ligand's
    record in the set), so the set's result for that ligand is the example's calculation."""
    entry, root = load_registry()[name], data_root()
    example = load_config(root / entry["example"])

    def settings(cfg) -> dict:
        data = cfg.model_dump(mode="json")
        data["system"].pop("protein"), data["system"].pop("ligand"), data["system"].pop("ligand_names")
        data.pop("output")
        return data

    for tier, names in (("subset", entry["subset"]), ("full", None)):
        cfg = benchmark_config(name, tier, tmp_path / tier)
        assert settings(cfg) == settings(example)
        assert cfg.system.ligand_names == names
        assert cfg.system.protein == root / entry["protein"] and cfg.system.ligand == root / entry["ligands"]
    assert example.system.protein.read_bytes() == (root / entry["protein"]).read_bytes()
    (ligand, record), = _sdf_records(example.system.ligand).items()
    assert _sdf_records(root / entry["ligands"])[ligand] == record


def test_the_tiers_choose_the_ligands_and_smoke_never_lengthens_a_setting_of_the_example(tmp_path, monkeypatch):
    """Independent review of PR 4: an example that selects ligands made the full tier run only those, and the
    smoke tier set 0.5 ns of membrane equilibration where the example had 0.2."""
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "examples" / "membrane_a2a").mkdir(parents=True)
    (tmp_path / "examples" / "membrane_a2a" / "config.yaml").write_text(
        "system: {protein: protein.pdb, ligand: ligand.sdf, ligand_names: [4e]}\n"
        "membrane: {enabled: true, equilibration_ns: 0.2}\n"
        "sampling: {production_ns: {complex: 0.05, solvent: 5.0}, rest2: {region: ligand}}\n")
    monkeypatch.setenv("CUFEP_DATA", str(tmp_path))
    example = load_config(tmp_path / "examples" / "membrane_a2a" / "config.yaml")
    assert benchmark_config("a2a", "full").system.ligand_names is None
    smoke = benchmark_config("a2a", "smoke", tmp_path / "s")
    assert smoke.system.ligand_names == ["4e"] and smoke.membrane.equilibration_ns == 0.2
    assert smoke.sampling.production_ns.complex == 0.05 and smoke.sampling.production_ns.solvent == 0.1
    assert smoke.sampling.rest2 == example.sampling.rest2 and smoke.membrane.enabled is True
    assert smoke.analysis == example.analysis
    # the MD from which the restraint is chosen is shortened too, and nothing else of the restraint
    assert (smoke.restraints.interactions.length_ns, smoke.restraints.interactions.discard_ns) == (0.3, 0.1)
    shortened = example.restraints.model_dump(mode="json")
    shortened["interactions"].update(length_ns=0.3, discard_ns=0.1)
    assert smoke.restraints.model_dump(mode="json") == shortened
    (tmp_path / "examples" / "membrane_a2a" / "config.yaml").write_text(
        "system: {protein: protein.pdb, ligand: ligand.sdf}\nmembrane: {enabled: true}\n"
        "restraints: {interactions: {length_ns: 0.2, discard_ns: 0.05}}\n")
    shorter = benchmark_config("a2a", "smoke", tmp_path / "s").restraints.interactions
    assert (shorter.length_ns, shorter.discard_ns) == (0.2, 0.05)


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
    assert [
        line.split("|")[3].split("[")[0].strip() for line in text.splitlines() if line.startswith("| mean signed")
    ] == ["0.00"]
    # ... over the five ligands of the subset they have (review of PR 1: the table said "0 by construction")
    subset = {"jmc_23", "jmc_30", "ejm_45", "ejm_48", "ejm_44"}
    part = tmp_path / "subset"
    part.mkdir()
    (part / "summary.json").write_text(json.dumps([s for s in summary if s["ligand"] in subset]))
    text = report("tyk2", part)
    expected = np.mean([ref[name]["fepplus"] - ref[name]["exp"] for name in subset])
    assert expected == pytest.approx(0.244, abs=0.001)
    assert [
        line.split("|")[3].split("[")[0].strip() for line in text.splitlines() if line.startswith("| mean signed")
    ] == ["0.24"]
    assert "over a part of that set their mean signed error is not zero" in text
    assert (tmp_path / "benchmark_report.md").exists() and (tmp_path / "benchmark_report.csv").exists()
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 16


def test_not_reliable_ligands_are_compared_with_experiment_and_fepplus_like_the_others(tmp_path):
    """The owner, 2026-11-20: NOT RELIABLE results are compared properly, not left out. They count in the
    statistics of all ligands, have their signed errors and the place below the level of overlap, and the
    breakdown compares them and the others with experiment, FEP+ on the same ligands."""
    ref = read_reference("tyk2")
    dg = {"jmc_23": -12.5, "ejm_47": -9.2, "ejm_49": -10.75, "ejm_31": -13.04}
    place = {"leg": "complex leg", "windows": [13, 14], "stage": "vanish", "lambda": [0.75, 0.88]}
    unreliable = {
        "ejm_31": [dict(place, of="all iterations", overlap=0.024), dict(place, of="the samples of the estimate",
                                                                           overlap=0.021)],
        "ejm_49": [{"leg": "solvent leg", "of": "all iterations", "overlap": 0.011, "windows": [0, 1],
                    "stage": "discharge", "lambda": [0.0, 0.25]}],
    }
    summary = [{"ligand": name, "dg_bind": value, "dg_bind_error": 0.5, "dg_bind_std": None, "n_repeats": 1,
                "warnings": [], "min_overlap": 0.08, "not_reliable": [p["leg"] for p in unreliable.get(name, [])][:1],
                "overlap_below_level": unreliable.get(name, [])} for name, value in dg.items()]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = report("tyk2", tmp_path)

    def err(names, method):
        values = [(dg[n] if method == "cufep" else ref[n]["fepplus"]) - ref[n]["exp"] for n in names]
        return np.mean(values), np.sqrt(np.mean(np.square(values)))

    # every ligand is in the statistics of all ligands
    mse_all = err(dg, "cufep")[0]
    assert [line.split("|")[2].split("[")[0].strip() for line in text.splitlines()
            if line.startswith("| mean signed")][0] == f"{mse_all:.2f}"
    # ... with its signed errors, marked, and the place below the level
    assert (f"| ejm_31 (NOT RELIABLE) | -9.54 | -13.04 | 0.50 |  | 1 | {ref['ejm_31']['fepplus']:.2f} | -3.50 | "
            f"{ref['ejm_31']['fepplus'] + 9.54:+.2f} |") in text
    assert ("| complex leg, lambda windows 13 and 14 (vanish stage, lambda 0.75 and 0.88): 0.024 among all "
            "iterations, 0.021 among the samples of the estimate |") in text
    # the breakdown: both groups, cuFEP and FEP+ on the same ligands
    assert ("| statistic | cuFEP, NOT RELIABLE (2) | FEP+, the same ligands | cuFEP, the others (2) | FEP+, the same "
            "ligands |") in text
    row = [line for line in text.splitlines() if line.startswith("| RMSE, raw")][-1]
    expected = [err(["ejm_31", "ejm_49"], "cufep")[1], err(["ejm_31", "ejm_49"], "fepplus")[1],
                err(["jmc_23", "ejm_47"], "cufep")[1], err(["jmc_23", "ejm_47"], "fepplus")[1]]
    assert [float(x) for x in row.split("|")[2:6]] == pytest.approx(expected, abs=0.005)
    assert "NOT RELIABLE (overlap below 0.03 in a leg, Klimovich et al. 2015): ejm_31, ejm_49." in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        rows = {r["ligand"]: r for r in csv.DictReader(fh)}
    assert float(rows["ejm_31"]["cufep_minus_exp"]) == pytest.approx(-3.5)
    assert rows["ejm_49"]["overlap_below_level"].startswith("solvent leg, lambda windows 0 and 1 (discharge stage")
    assert rows["jmc_23"]["overlap_below_level"] == "" and (tmp_path / "benchmark_report.png").exists()
    # a group of one ligand has no error around its mean; no NOT RELIABLE ligand: said so
    (tmp_path / "summary.json").write_text(json.dumps([s for s in summary if s["ligand"] != "ejm_49"]))
    row = [line for line in report("tyk2", tmp_path).splitlines() if line.startswith("| RMSE after")][-1]
    assert row.split("|")[2:4] == [" N/A ", " N/A "]
    (tmp_path / "summary.json").write_text(json.dumps([s for s in summary if s["ligand"] not in unreliable]))
    text = report("tyk2", tmp_path)
    assert "No ligand with a binding free energy is NOT RELIABLE." in text and "meet the level" not in text
    # the level is the one the analysis of the run recorded
    level = [dict(s, analysis_record={"min_overlap": 0.05}) for s in summary]
    (tmp_path / "summary.json").write_text(json.dumps(level))
    text = report("tyk2", tmp_path)
    assert "NOT RELIABLE (overlap below 0.05, Klimovich" in text and "below 0.03" not in text
    # a summary of an earlier version names the legs only
    assert overlap_places({"not_reliable": ["complex leg"]}) == "complex leg (the windows are not in this summary.json)"
    # the figures carry the correlation coefficients and the errors of both methods over all their ligands
    from cufep.benchmarks.runner import figure_statistics

    lines = figure_statistics([{"ligand": n, "exp": ref[n]["exp"], "cufep": v, "fepplus": ref[n]["fepplus"]}
                               for n, v in dg.items()])
    r = np.corrcoef(list(dg.values()), [ref[n]["exp"] for n in dg])[0, 1]
    assert lines[0].startswith(f"cuFEP, 4 ligands: Pearson r {r:.2f}, Kendall")
    assert lines[1].startswith("FEP+, 4 ligands")
    assert figure_statistics([{"ligand": "x", "exp": -9.0, "cufep": -10.0, "fepplus": -9.5}])[0] == (
        "cuFEP, 1 ligands: Pearson r N/A, Kendall \u03c4 N/A, RMSE 1.00, RMSE after removing the mean error N/A "
        "kcal/mol")


def test_ligands_an_earlier_version_analysed_are_not_said_to_meet_the_level(tmp_path):
    """Independent review of PR 4: for a summary of an earlier version (no checks of this version) the breakdown
    said that the ligands "meet the level of overlap"; a binding free energy written as null stopped the report."""
    old = [{"ligand": name, "dg_bind": value, "dg_bind_mbar_error": 0.3, "dg_bind_std": None, "n_repeats": 1}
           for name, value in (("jmc_23", -12.0), ("jmc_30", -11.0))]
    old.append({"ligand": "ejm_31", "dg_bind": None, "n_repeats": 1, "not_reliable": ["solvent leg"], "warnings": []})
    (tmp_path / "summary.json").write_text(json.dumps(old))
    text = report("tyk2", tmp_path)
    assert ("No ligand with a binding free energy is NOT RELIABLE. The checks of this version were not made for "
            "jmc_23, jmc_30 (analysed by an earlier version).") in text
    assert "| ejm_31 | -9.54 | NOT RELIABLE (overlap): solvent leg" in text and "Ligands with a result: 2 of 3" in text


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


def test_structural_analysis_ignores_tumbling_and_measures_a_leaving_ligand():
    """The RMSD and the drift of the centre of the ligand are numbers without a verdict (the owner, 2026-11-20:
    the warnings above 3 A and 5 A, levels of cuFEP's own, are gone)."""
    still = structural_analysis(_samples(np.zeros(3)))
    assert still["ligand_rmsd_max_A"] < 0.5 and "warnings" not in still
    gone = structural_analysis(_samples(np.array([1.2, 0.0, 0.0])))
    assert gone["ligand_com_drift_last_half_A"] > 5.0 and "warnings" not in gone
    assert structural_analysis({"attrs": {}}) is None


def test_a_report_is_that_of_the_run_as_it_is_now(tmp_path):
    """Review of PR 1: after every extension had failed, the report, table and figure of the earlier,
    successful state still showed the old binding free energy."""
    ref = read_reference("tyk2")
    good = {"ligand": "jmc_23", "dg_bind": -12.34, "dg_bind_std": None, "n_repeats": 1, "warnings": []}
    (tmp_path / "summary.json").write_text(json.dumps([good]))
    assert "-12.34" in report("tyk2", tmp_path)
    assert "-12.34" in (tmp_path / "benchmark_report.csv").read_text()
    assert (tmp_path / "benchmark_report_ligands.png").exists()            # the figure ligand by ligand
    (tmp_path / "benchmark_report.png").write_text("the figure of that state")      # also where no figure can be made
    (tmp_path / "benchmark_report_ligands.png").write_text("the figure of that state")

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
    assert not (tmp_path / "benchmark_report.png").exists() and not (tmp_path / "benchmark_report_ligands.png").exists()

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
    earlier = {
        "ligand": "jmc_23",
        "units": "kcal/mol",
        "dg_bind": -11.0,
        "dg_bind_std": None,
        "dg_bind_mbar_error": 0.3,
        "n_repeats": 1,
        "warnings": [],
        "failed_legs": [],
        "not_reliable": [],
        "not_converged": ["complex leg"],
        "convergence_not_assessed": ["solvent leg"],
        "estimators_disagree": [],
        "min_overlap": 0.08,
        "repeats": [{"lambda_windows": {"complex": 16, "solvent": 13}}],
    }
    # the owner, 2026-11-20: the verdicts of checks of cuFEP's own (NOT CONVERGED, halves, BAR against MBAR) are not
    # shown, also not from summaries of the versions that made them; the flag says the checks of this version are not
    # made
    said = [earlier_version]
    assert flags_of(earlier) == said
    (tmp_path / "summary.json").write_text(json.dumps([earlier]))
    text = report("tyk2", tmp_path)
    assert f"| 16 | 13 | 0.080 |  | {'; '.join(said)} |" in text
    assert "NOT CONVERGED" not in text and "carries the flag that the checks of this version were not made" in text
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
    assert (
        "| -11.00 | 0.75 |  | 1 |" in text and "| -10.50 | - |  | 1 |" in text and "| -9.00 | 0.50 | 1.20 | 3 |" in text
    )
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
    """Review of PR 1: a leg whose only failed check was named in summary.json and nowhere a person looks: not in the
    flags of the report, not in summary.csv, not in the result line. The checks are the two of the literature (the
    owner, 2026-11-20): the keys of the checks of cuFEP's own that earlier versions wrote are no flags and no
    columns."""
    from types import SimpleNamespace

    from cufep.analysis.mbar import ANALYSIS_VERSION, QUALITY_CHECKS
    from cufep.benchmarks.runner import LEG_FLAGS, flags_of
    from cufep.run import result_line, write_summary

    assert QUALITY_CHECKS == ("overlap", "samples")
    assert [key for key, _name in LEG_FLAGS] == ["failed_legs", "inconsistent_legs", "not_reliable", "old_analysis",
                                                 "few_samples", "origin_not_recorded"]
    only_samples = {
        "ligand": "jmc_23",
        "dg_bind": -11.0,
        "dg_bind_std": None,
        "dg_bind_mbar_error": 0.3,
        "dg_bind_error": 0.5,
        "dg_bind_error_from_time_blocks": True,
        "analysis_versions": [ANALYSIS_VERSION],
        "n_repeats": 1,
        "warnings": ["repeat 0 complex: few independent samples: ..."],
        "failed_legs": [],
        "not_reliable": [],
        "few_samples": ["complex leg"],
        "min_overlap": 0.08,
        # keys of an earlier version, for checks that are no longer made: shown nowhere
        "halves_differ": ["complex leg"],
        "estimators_disagree": ["complex leg"],
    }
    assert flags_of(only_samples) == ["few independent samples: complex leg"]
    (tmp_path / "summary.json").write_text(json.dumps([only_samples]))
    text = report("tyk2", tmp_path)
    assert "| few independent samples: complex leg |" in text and "halves" not in text and "BAR and MBAR" not in text
    with open(tmp_path / "benchmark_report.csv") as fh:
        assert list(csv.DictReader(fh))[0]["flags"] == "few independent samples: complex leg"
    assert result_line(only_samples) == ("RESULT jmc_23: dG_bind = -11.00 ± 0.50 kcal/mol (standard error from time "
                                         "blocks, 1 repeat); few independent samples: complex leg; 1 warning(s):")
    write_summary(tmp_path, [only_samples], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["few_samples"] == "complex leg" and "halves_differ" not in row and "estimators_disagree" not in row
    assert all(key in row for key, _name in LEG_FLAGS)                       # one column per thing that can be said
    assert row["mbar_error"] == "0.300" and row["standard_error"] == "0.500"
    # the standard error of one calculation is the one from time blocks; that of MBAR stays in its column
    blocks = dict(only_samples, dg_bind_error=0.9, dg_bind_error_from_time_blocks=True)
    assert result_line(blocks).startswith("RESULT jmc_23: dG_bind = -11.00 ± 0.90 kcal/mol (standard error from time "
                                          "blocks, 1 repeat); few independent samples")
    assert " ± 0.90 kcal/mol (standard error, 1 repeat);" in result_line(
        dict(blocks, dg_bind_error_from_time_blocks=False)
    )
    write_summary(tmp_path, [blocks], [SimpleNamespace(name="jmc_23", properties={})])
    with open(tmp_path / "summary.csv") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["standard_error"] == "0.900" and row["mbar_error"] == "0.300"

    # all of them at once, in the order of their weight; the overlap is given with the number
    everything = dict(
        only_samples,
        failed_legs=["solvent leg of repeat 1"],
        not_reliable=["complex leg"],
        inconsistent_legs=[],
        old_analysis=["solvent leg: version 2 of the analysis; this run uses version 3"],
        few_samples=["complex leg"],
        halves_not_compared=["solvent leg"],
        halves_below_level=["complex leg"],
        origin_not_recorded=["complex leg: the samples"],
        min_overlap=0.02,
        min_overlap_used=0.012,
    )
    assert flags_of(everything) == [
        "FAILED: solvent leg of repeat 1", "NOT RELIABLE (overlap): complex leg", "old analysis: solvent leg",
        "few independent samples: complex leg", "origin of files not recorded: complex leg"]
    assert "; NOT RELIABLE (overlap): complex leg (neighbouring lambda windows overlap by only 0.012);" in result_line(
        everything
    )
    nothing = {"ligand": "x", "n_repeats": 0, "warnings": []}
    assert result_line(nothing) == "RESULT x: no binding free energy; no complete repeat"
    assert result_line(dict(nothing, inconsistent_legs=["temperature (K): complex leg 315.0, solvent leg 298.15"])) \
        == ("RESULT x: no binding free energy; LEGS MADE UNDER DIFFERENT CONDITIONS: temperature (K): complex leg "
            "315.0, solvent leg 298.15")


def test_frozen_schedules_go_to_the_run_directory_by_default(tmp_path, monkeypatch):
    """Review of PR 1 (P2): in the official image CUFEP_DATA=/opt/cufep belongs to root and the user of the image
    could not write the default benchmarks/<name>/schedules there. The default is the run directory."""
    from cufep.benchmarks.runner import freeze_schedules

    data = tmp_path / "data"
    (data / "benchmarks").mkdir(parents=True)
    (data / "benchmarks").chmod(0o555)                 # read only, as in the image
    monkeypatch.setenv("CUFEP_DATA", str(data))
    run = tmp_path / "run"
    for leg in ("complex", "solvent"):
        d = run / "ejm_31" / "repeat_0" / leg
        d.mkdir(parents=True)
        (d / "schedule.yaml").write_text(f"leg: {leg}\n")
    try:
        dest = freeze_schedules("tyk2", run)
    finally:
        (data / "benchmarks").chmod(0o755)
    assert dest == run / "frozen_schedules"
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.yaml")) == ["ejm_31/complex_r0.yaml",
                                                                                     "ejm_31/solvent_r0.yaml"]
    assert not list((data / "benchmarks").iterdir())


def test_a_relative_benchmark_runs_with_the_settings_of_its_relative_example(tmp_path):
    """`--protocol rbfe`: a set runs with every setting of its relative example but the input files, the choice of
    ligands, the edges (LOMAP's network of the ligands) and the output directory; the smoke tier runs the first edge
    of the example, shortened, and the example's protein and ligands are those of the set."""
    from cufep.benchmarks.rbfe import rbfe_benchmark_config
    from cufep.rbfe.config import load_rbfe_config

    for name in ("tyk2", "a2a"):
        entry, root = load_registry()[name], data_root()
        example = load_rbfe_config(root / entry["rbfe_example"])

        def settings(cfg) -> dict:
            data = cfg.model_dump(mode="json")
            for key in ("protein", "ligands", "ligand_names"):
                data["system"].pop(key)
            data["network"].pop("edges"), data.pop("output")
            return data

        states = {}
        if entry.get("rbfe_states"):
            with open(root / entry["rbfe_states"]) as fh:
                states = {r["ligand"]: r["compound"] for r in csv.DictReader(fh)}
        for tier, names in (("subset", entry["subset"]), ("full", None)):
            cfg = rbfe_benchmark_config(name, tier, tmp_path / tier)
            data = settings(cfg)
            data["system"].pop("states")
            expected = settings(example)
            expected["system"].pop("states")
            assert data == expected
            if states:
                # every state of the compounds of the tier, combined into compounds
                assert cfg.system.states == root / entry["rbfe_states"]
                assert cfg.system.ligand_names == (None if names is None else
                                                   [s for s, c in states.items() if c in names])
            else:
                assert cfg.system.ligand_names == names and cfg.system.states is None
            assert cfg.network.edges is None
            assert cfg.system.protein == root / entry["protein"] and cfg.system.ligands == root / entry["rbfe_ligands"]
        smoke = rbfe_benchmark_config(name, "smoke", tmp_path / "s")
        first = tuple(example.network.edges[0])
        assert [tuple(e) for e in smoke.network.edges] == [first] and smoke.system.ligand_names == list(first)
        assert smoke.system.ligands == root / entry["ligands"] and smoke.system.states is None
        assert smoke.sampling.production_ns == 0.1 and smoke.lambda_schedule.adaptive.window("complex") == 10
        assert smoke.membrane.enabled is (name == "a2a")
        assert example.system.protein.read_bytes() == (root / entry["protein"]).read_bytes()
        # the example's records were written again by RDKit: the same atoms at the same places as the set's
        records = _sdf_records(root / entry["ligands"])
        for lig, record in _sdf_records(example.system.ligands).items():
            assert _atoms(record) == _atoms(records[lig])
    with pytest.raises(KeyError):
        rbfe_benchmark_config("tyk2", "huge")


def _atoms(record: str) -> list[tuple]:
    """The elements and coordinates (A, 4 decimals) of a V2000 record."""
    lines = record.split("\n")
    n = int(lines[3][:3])
    return [(line.split()[3], tuple(round(float(x), 4) for x in line.split()[:3])) for line in lines[4:4 + n]]


def _relative_run(path, name: str, noise: float = 0.4, seed: int = 3) -> dict:
    """A directory of a relative run of the whole set NAME: dG of every ligand and ddG of the published edges with
    known errors (experiment + noise), as cufep rbfe run writes them."""
    from cufep.benchmarks.rbfe import published_edges

    ref = read_reference(name)
    rng = np.random.default_rng(seed)
    dg = {lig: v["exp"] + noise * rng.normal() for lig, v in ref.items()}
    edges = []
    for (a, b), _ in published_edges(name).items():
        ddg = dg[b] - dg[a] + 0.2 * rng.normal()
        edges.append({"edge": f"{a}~{b}", "ligand_a": a, "ligand_b": b, "ddg": ddg, "se": 0.3,
                      "network_ddg": dg[b] - dg[a], "network_se": 0.2, "exp_ddg": ref[b]["exp"] - ref[a]["exp"],
                      "lowest_overlap": {"complex": 0.1, "solvent": 0.2}, "not_reliable": []})
    summary = {"ligands": list(ref), "edges": edges, "experiment_kcal_mol": {k: v["exp"] for k, v in ref.items()},
               "cycle_closures": [], "notes": [],
               "dg": {"reference": "mean of the experimental values", "reference_ligands": list(ref),
                      "values": {k: {"dg": v, "se": 0.15} for k, v in dg.items()}}}
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "network.json").write_text(json.dumps({"ligands": list(ref), "edges": [[e["ligand_a"], e["ligand_b"]]
                                                                                   for e in edges],
                                                   "edges_from": "planned by LOMAP"}))
    return summary


def test_the_relative_report_gives_the_statistics_of_the_public_benchmark(tmp_path):
    """The report of a relative run: the pairwise RMSE, R^2 and tau of the dG against experiment for cuFEP, the
    published predictions and the recorded absolute run on the same ligands; the edgewise errors of the edges' own
    ddG, of the network's, and of the published ones on the same edges, matched also when published the other way
    round. The same text through `cufep benchmark report`."""
    from click.testing import CliRunner

    from cufep import cli
    from cufep.benchmarks.rbfe import published_edges, report, stored_abfe

    summary = _relative_run(tmp_path / "run", "tyk2")
    text = report("tyk2", tmp_path / "run")
    ref = read_reference("tyk2")
    names = list(ref)
    exp = [ref[n]["exp"] for n in names]
    ours = metrics([summary["dg"]["values"][n]["dg"] for n in names], exp)
    theirs = metrics([ref[n]["fepplus"] for n in names], exp)
    abfe = stored_abfe("tyk2")
    assert len(abfe) == 16
    absolute = metrics([abfe[n] for n in names], exp)
    row = next(line for line in text.splitlines() if line.startswith("| pairwise ddG RMSE"))
    cells = [c.split("[")[0].strip() for c in row.split("|")[2:5]]
    assert cells == [f"{m['pairwise_rmse']:.2f}" for m in (ours, theirs, absolute)]
    assert "Ligands with a dG: 16 of 16; edges with a ddG: 24 of 24 (edges planned by LOMAP)." in text
    published = published_edges("tyk2")
    assert len(published) == 24
    own = [(e["ddg"], e["exp_ddg"]) for e in summary["edges"]]
    rmse = float(np.sqrt(np.mean([(p - x) ** 2 for p, x in own])))
    assert f"| cuFEP, own | 24 | {rmse:.2f} |" in text
    bennett = float(np.sqrt(np.mean([(v["bennett"] - (ref[b]["exp"] - ref[a]["exp"])) ** 2
                                     for (a, b), v in published.items()])))
    assert f"| FEP+ Bennett, the same edges | 24 | {bennett:.2f} |" in text
    for f in ("rbfe_report.md", "rbfe_report_ligands.csv", "rbfe_report_edges.csv", "rbfe_report.png",
              "rbfe_report_edges.png"):
        assert (tmp_path / "run" / f).exists()
    # an edge run the other way round meets the published prediction with its sign turned
    turned = dict(summary)
    a, b = summary["edges"][0]["ligand_a"], summary["edges"][0]["ligand_b"]
    turned["edges"] = [dict(summary["edges"][0], edge=f"{b}~{a}", ligand_a=b, ligand_b=a,
                            ddg=-summary["edges"][0]["ddg"], exp_ddg=-summary["edges"][0]["exp_ddg"])]
    (tmp_path / "turned").mkdir()
    (tmp_path / "turned" / "summary.json").write_text(json.dumps(turned))
    report("tyk2", tmp_path / "turned")
    rows = list(csv.DictReader((tmp_path / "turned" / "rbfe_report_edges.csv").open()))
    assert float(rows[0]["fepplus_bennett_ddg"]) == pytest.approx(-published[(a, b)]["bennett"])
    result = CliRunner().invoke(cli.main, ["benchmark", "report", "tyk2", str(tmp_path / "run")])
    assert result.exit_code == 0 and result.output == text + "\n"


def test_a_relative_benchmark_run_is_the_relative_run_of_its_tier(tmp_path, monkeypatch):
    """`cufep benchmark run NAME --protocol rbfe` runs the configuration of the tier through the driver of
    `cufep rbfe run`, prints the report of the directory and ends with 1 when a leg of this run failed; a relative
    directory has no lambda schedules to freeze."""
    from click.testing import CliRunner

    from cufep import cli
    from cufep.benchmarks.rbfe import rbfe_benchmark_config
    from cufep.rbfe import run as rbfe_run

    seen = {}

    def fake_run(rcfg, **kwargs):
        seen["cfg"] = rcfg
        summary = _relative_run(rcfg.output.dir, "tyk2")
        summary["failed_in_this_run"] = ["jmc_23~jmc_30/r0/complex: boom"]
        return summary

    monkeypatch.setattr(rbfe_run, "run_rbfe", fake_run)
    out = tmp_path / "out"
    result = CliRunner().invoke(cli.main, ["benchmark", "run", "tyk2", "--tier", "subset", "--protocol", "rbfe",
                                           "-o", str(out), "--gpus", "0,1"])
    expected = rbfe_benchmark_config("tyk2", "subset", out)
    expected.resources.gpus = [0, 1]
    assert seen["cfg"].model_dump() == expected.model_dump()
    assert result.exit_code == 1 and "# cuFEP relative benchmark report: tyk2" in result.output
    assert (out / "rbfe_report.md").exists()
    frozen = CliRunner().invoke(cli.main, ["benchmark", "freeze", "tyk2", str(out)])
    assert frozen.exit_code != 0 and "is a relative run" in frozen.output


def test_the_states_of_a2a_are_the_records_of_the_published_map_with_their_corrections():
    """`rbfe_states` of A2A holds every record of the unmodified SDF as a state of its compound, as curate.py makes it:
    the deprotonated phenols of 4e (both orientations), 4f and 4o with the populations of their forms at pH 7.5 from
    the solvent pKa of Ross et al. 2023 (Supplementary Table 17; Azimi et al. 2022, eqs 12-14), the ten compounds of
    Supplementary Table 16 with symmetry number 2, and each state with the experimental value of its compound."""
    from cufep.benchmarks.curate import A2A_ASSAY_PH, A2A_SOLVENT_PKA, A2A_SYMMETRY_CORRECTED, compound_id, read_sdf

    entry = load_registry()["a2a"]
    with open(data_root() / entry["rbfe_states"]) as fh:
        rows = list(csv.DictReader(fh))
    records = read_sdf(data_root() / "benchmarks/a2a/inputs/a2a_hip278_ligands.sdf")
    assert [r["sdf_record"] for r in rows] == [name for name, _ in records] and len(rows) == 29
    ref = read_reference("a2a")
    assert {r["compound"] for r in rows} == set(ref)
    by_record = dict(records)
    for r in rows:
        assert r["compound"] == compound_id(r["sdf_record"]) and r["ligand"] == "_".join(r["sdf_record"].split())
        assert float(r["exp_dg_kcal_mol"]) == pytest.approx(ref[r["compound"]]["exp"], abs=2e-3)
        assert int(r["symmetry_number"]) == (2 if r["compound"] in A2A_SYMMETRY_CORRECTED else 1)
        deprotonated = "deprotonated" in r["sdf_record"]
        assert int(r["formal_charge"]) == (-1 if deprotonated else 0)
        if r["compound"] in A2A_SOLVENT_PKA:
            alpha = 10.0 ** (A2A_ASSAY_PH - A2A_SOLVENT_PKA[r["compound"]])
            share = alpha / (1 + alpha) if deprotonated else 1 / (1 + alpha)
            assert float(r["solution_fraction"]) == pytest.approx(share, abs=1e-6)
        else:
            assert float(r["solution_fraction"]) == 1.0 and not deprotonated
        written = _sdf_records(data_root() / entry["rbfe_ligands"])[r["ligand"]]
        assert _atoms(written) == _atoms(by_record[r["sdf_record"]])
    assert sum(1 for r in rows if "deprotonated" in r["sdf_record"]) == 4


def _states_run(path, name: str, summary: dict, states: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "network.json").write_text(json.dumps({"ligands": summary["ligands"], "edges": [
        [e["ligand_a"], e["ligand_b"]] for e in summary["edges"]], "edges_from": "planned by LOMAP",
        "states": {"compound": {s: r["compound"] for s, r in states.items()}}}))


def test_the_relative_report_of_states_compares_compounds(tmp_path):
    """A relative run of states: the report compares the dG of the compounds with experiment and the published
    predictions of the compounds, matches the published edges by state, compares experiment with the pairs of compounds
    the edges join, and an edge between states only with the published prediction of the same edge."""
    from cufep.benchmarks.rbfe import published_edges, read_states, report

    ref = read_reference("a2a")
    states = read_states("a2a")
    rng = np.random.default_rng(5)
    dg_state = {s: ref[r["compound"]]["exp"] + 0.3 * rng.normal() for s, r in states.items()}
    published = published_edges("a2a", by_state=True)
    assert ("4b_U", "4b_D") in published and len(published) == 46
    edges = [{"edge": f"{a}~{b}", "ligand_a": a, "ligand_b": b, "ddg": dg_state[b] - dg_state[a], "se": 0.3,
              "network_ddg": dg_state[b] - dg_state[a], "network_se": 0.2,
              "lowest_overlap": {"complex": 0.1, "solvent": 0.2}, "not_reliable": []} for a, b in published]
    compounds = {c: {"dg": v["exp"] + 0.5 * rng.normal(), "se": 0.2, "symmetry_number": 1,
                     "weights": {s: 1.0 for s, r in states.items() if r["compound"] == c}} for c, v in ref.items()}
    summary = {"ligands": list(states), "edges": edges, "notes": [], "cycle_closures": [],
               "experiment_kcal_mol": {s: ref[r["compound"]]["exp"] for s, r in states.items()},
               "dg": {"reference": "mean of the experimental values", "values": {}},
               "compounds": {"temperature_K": 298.15, "experiment_kcal_mol": {c: v["exp"] for c, v in ref.items()},
                             "values": compounds}}
    _states_run(tmp_path, "a2a", summary, states)
    text = report("a2a", tmp_path)
    assert "Compounds with a dG: 17 of 17; edges with a ddG: 46 of 46 (edges planned by LOMAP)." in text
    names = list(ref)
    ours = metrics([compounds[n]["dg"] for n in names], [ref[n]["exp"] for n in names])
    row = next(line for line in text.splitlines() if line.startswith("| pairwise ddG RMSE"))
    assert row.split("|")[2].split("[")[0].strip() == f"{ours['pairwise_rmse']:.2f}"
    pairs = {tuple(sorted((states[a]["compound"], states[b]["compound"]))) for a, b in published
             if states[a]["compound"] != states[b]["compound"]}
    expected = float(np.sqrt(np.mean([((compounds[cb]["dg"] - compounds[ca]["dg"]) - (ref[cb]["exp"] - ref[ca]["exp"]))
                                      ** 2 for ca, cb in pairs])))
    assert f"| cuFEP, the pairs of compounds the edges join | {len(pairs)} | {expected:.2f} |" in text
    rows = {r["edge"]: r for r in csv.DictReader((tmp_path / "rbfe_report_edges.csv").open())}
    assert rows["4b_U~4b_D"]["exp_ddg"] == "" and rows["4b_U~4b_D"]["exp_ddg_of"] == "the compounds"


def test_the_report_of_states_keeps_them_when_the_network_gives_no_dg(tmp_path):
    """Review of PR #19 (P2): with an incomplete network (no compounds in the summary) the report lost which compound
    each state is: the edge 4jS~4jR got an experimental ddG of +0.00 and lost the published Bennett value +0.84.
    The states come from network.json."""
    from cufep.benchmarks.rbfe import published_edges, read_states, report

    states = read_states("a2a")
    published = published_edges("a2a", by_state=True)
    assert published[("4jS", "4jR")]["bennett"] == pytest.approx(0.84, abs=0.01)
    summary = {"ligands": list(states), "edges": [
        {"edge": "4jS~4jR", "ligand_a": "4jS", "ligand_b": "4jR", "ddg": 0.5, "se": 0.3, "not_reliable": []}],
        "notes": ["the edges with a ddG do not connect all ligands: no dG"], "cycle_closures": [], "dg": None,
        "experiment_kcal_mol": {s: float(r["exp_dg_kcal_mol"]) for s, r in states.items()}}
    _states_run(tmp_path, "a2a", summary, states)
    report("a2a", tmp_path)
    row = next(csv.DictReader((tmp_path / "rbfe_report_edges.csv").open()))
    assert row["exp_ddg"] == "" and float(row["fepplus_bennett_ddg"]) == pytest.approx(0.84, abs=0.01)


def test_experiment_is_compared_with_the_compounds_not_with_an_edge_between_states(tmp_path):
    """Review of PR #19 (P2): A with two equal binding modes and B with one, every state at dG 0: the edge A1 -> B
    computes 0 and the compounds differ by kT ln 2 (+0.41 kcal/mol at 298 K). Experiment is compared with the pair of
    compounds: with an experimental ddG of +0.41 the error is 0, not 0.41."""
    from cufep.benchmarks.rbfe import read_states, report

    kt_ln2 = 0.0019872 * 298.15 * np.log(2.0)
    states = {s: r for s, r in read_states("a2a").items() if r["compound"] in ("4b", "4e")}
    two = [s for s in states if states[s]["compound"] == "4b"]
    one = [s for s in states if states[s]["compound"] == "4e"][0]
    exp = {"4b": -10.0 - kt_ln2, "4e": -10.0}
    compounds = {"4b": {"dg": -10.0 - kt_ln2, "se": 0.1, "symmetry_number": 1, "weights": {s: 0.5 for s in two}},
                 "4e": {"dg": -10.0, "se": 0.1, "symmetry_number": 1, "weights": {one: 1.0}}}
    summary = {"ligands": two + [one], "edges": [
        {"edge": f"{two[0]}~{one}", "ligand_a": two[0], "ligand_b": one, "ddg": 0.0, "se": 0.1, "not_reliable": []}],
        "notes": [], "cycle_closures": [], "dg": {"reference": "mean of the experimental values", "values": {}},
        "experiment_kcal_mol": {s: exp[states[s]["compound"]] for s in two + [one]},
        "compounds": {"temperature_K": 298.15, "experiment_kcal_mol": exp, "values": compounds}}
    _states_run(tmp_path, "a2a", summary, states)
    text = report("a2a", tmp_path)
    assert "| cuFEP, the pairs of compounds the edges join | 1 | 0.00 |" in text


def test_the_pairs_of_compounds_take_the_experiment_of_the_table_of_compounds(tmp_path):
    """Second review of PR #19 (P3): a run that records no experimental values (none in its configuration) has those
    of the benchmark in the table of compounds, and the pairs of compounds against experiment were left out (0 pairs).
    The pairs take the experimental value of the table."""
    from cufep.benchmarks.rbfe import read_states, report

    ref = read_reference("a2a")
    states = {s: r for s, r in read_states("a2a").items() if r["compound"] in ("4b", "4e")}
    two = [s for s in states if states[s]["compound"] == "4b"]
    one = [s for s in states if states[s]["compound"] == "4e"][0]
    compounds = {"4b": {"dg": ref["4b"]["exp"], "se": 0.1, "symmetry_number": 1, "weights": {s: 0.5 for s in two}},
                 "4e": {"dg": ref["4e"]["exp"], "se": 0.1, "symmetry_number": 1, "weights": {one: 1.0}}}
    summary = {"ligands": two + [one], "edges": [
        {"edge": f"{two[0]}~{one}", "ligand_a": two[0], "ligand_b": one, "ddg": 0.0, "se": 0.1, "not_reliable": []}],
        "notes": [], "cycle_closures": [], "dg": {"reference": "the first ligand", "values": {}},
        "experiment_kcal_mol": {}, "compounds": {"temperature_K": 298.15, "experiment_kcal_mol": {},
                                                 "values": compounds}}
    _states_run(tmp_path, "a2a", summary, states)
    text = report("a2a", tmp_path)
    assert "| cuFEP, the pairs of compounds the edges join | 1 | 0.00 |" in text


def test_a_relative_summary_alone_is_reported_as_a_relative_run_from_the_command_line(tmp_path):
    """Sixth review of PR #19 (P2): `cufep benchmark report tyk2 DIR` of a directory that keeps the summary of a
    relative run alone (no network.json) went to the report of the absolute calculation, which failed on it. Such a
    summary is that of a relative run."""
    from click.testing import CliRunner

    from cufep import cli
    from cufep.benchmarks.rbfe import report

    _relative_run(tmp_path / "run", "tyk2")
    (tmp_path / "run" / "network.json").unlink()
    result = CliRunner().invoke(cli.main, ["benchmark", "report", "tyk2", str(tmp_path / "run")])
    assert result.exit_code == 0, result.output
    assert result.output == report("tyk2", tmp_path / "run") + "\n"
    assert result.output.startswith("# cuFEP relative benchmark report: tyk2")


def test_a_summary_of_states_alone_is_reported_as_the_run_with_its_network(tmp_path):
    """Seventh review of PR #19 (P1): the report of an A2A summary without its network.json compared the experiment of
    the 17 compounds with the dG of their 29 states, and the edge between the two states 4jS and 4jR of 4j with an
    experimental ddG of 0. The states come from the summary (experiment_of_compound, the states of each compound):
    the same compounds, statistics and edges as with the network."""
    import shutil

    from cufep.benchmarks.rbfe import published_edges, read_states, report

    ref = read_reference("a2a")
    states = read_states("a2a")
    rng = np.random.default_rng(7)
    dg_state = {s: ref[r["compound"]]["exp"] + 0.3 * rng.normal() for s, r in states.items()}
    published = published_edges("a2a", by_state=True)
    edges = [{"edge": f"{a}~{b}", "ligand_a": a, "ligand_b": b, "ddg": dg_state[b] - dg_state[a], "se": 0.3,
              "network_ddg": dg_state[b] - dg_state[a], "network_se": 0.2, "not_reliable": []} for a, b in published]
    compounds = {c: {"dg": v["exp"] + 0.5 * rng.normal(), "se": 0.2, "symmetry_number": 1,
                     "weights": {s: 1.0 for s, r in states.items() if r["compound"] == c}} for c, v in ref.items()}
    summary = {"ligands": list(states), "edges": edges, "notes": [], "cycle_closures": [],
               "experiment_kcal_mol": {s: ref[r["compound"]]["exp"] for s, r in states.items()},
               "experiment_of_compound": {s: r["compound"] for s, r in states.items()},
               "dg": {"reference": "mean of the experimental values", "values": {}},
               "compounds": {"temperature_K": 298.15, "experiment_kcal_mol": {c: v["exp"] for c, v in ref.items()},
                             "values": compounds}}
    _states_run(tmp_path / "with", "a2a", summary, states)
    shutil.copytree(tmp_path / "with", tmp_path / "alone")
    (tmp_path / "alone" / "network.json").unlink()
    texts = {name: report("a2a", tmp_path / name) for name in ("with", "alone")}
    for table in ("rbfe_report_ligands.csv", "rbfe_report_edges.csv"):
        assert (tmp_path / "with" / table).read_text() == (tmp_path / "alone" / table).read_text(), table
    assert "Compounds with a dG: 17 of 17" in texts["alone"]
    rows = {name: [line for line in text.splitlines() if line.startswith("| ")] for name, text in texts.items()}
    assert rows["with"] == rows["alone"]
    edge = next(r for r in csv.DictReader((tmp_path / "alone" / "rbfe_report_edges.csv").open())
                if r["edge"] == "4jS~4jR")
    assert edge["exp_ddg"] == ""
    # a summary of an earlier version without experiment_of_compound: the states of each compound
    del summary["experiment_of_compound"]
    (tmp_path / "alone" / "summary.json").write_text(json.dumps(summary))
    report("a2a", tmp_path / "alone")
    assert (tmp_path / "with" / "rbfe_report_ligands.csv").read_text() == (
        tmp_path / "alone" / "rbfe_report_ligands.csv").read_text()


def test_a_partial_summary_of_states_of_an_earlier_version_gives_no_experiment_within_a_compound(tmp_path):
    """Eighth review of PR #19 (P2): a summary of a6c4dda whose edges did not yet connect the states (no compounds, no
    experiment_of_compound) was taken for a run without states: the edge between 4jS and 4jR of 4j got an experimental
    ddG of 0.00 from the experiment of their compound. The states come from the edges whose experiment is that of the
    compounds (exp_compounds) and from the states of the benchmark; a state whose compound is still not known is
    compared with nothing."""
    from cufep.benchmarks.rbfe import read_states, report

    states = read_states("a2a")
    exp = {s: float(r["exp_dg_kcal_mol"]) for s, r in states.items()}
    summary = {"ligands": list(states) + ["unknown"], "cycle_closures": [], "dg": None,
               "notes": ["the edges with a ddG do not connect all ligands: no dG"],
               "experiment_kcal_mol": {**exp, "unknown": -9.0},
               "edges": [{"edge": "4h~4jS", "ligand_a": "4h", "ligand_b": "4jS", "ddg": -0.5, "se": 0.3,
                          "exp_ddg": exp["4jS"] - exp["4h"], "exp_ddg_of": "the compounds",
                          "exp_compounds": ["4h", "4j"], "not_reliable": []},
                         {"edge": "4jS~4jR", "ligand_a": "4jS", "ligand_b": "4jR", "ddg": 0.4, "se": 0.3,
                          "not_reliable": []},
                         {"edge": "4jS~unknown", "ligand_a": "4jS", "ligand_b": "unknown", "ddg": 0.2, "se": 0.3,
                          "not_reliable": []}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = report("a2a", tmp_path)
    rows = {r["edge"]: r for r in csv.DictReader((tmp_path / "rbfe_report_edges.csv").open())}
    assert rows["4jS~4jR"]["exp_ddg"] == "" and rows["4jS~unknown"]["exp_ddg"] == ""
    assert float(rows["4h~4jS"]["exp_ddg"]) == pytest.approx(exp["4jS"] - exp["4h"])
    assert "States whose compound the summary does not record (unknown)" in text


def test_a_summary_of_2ecbcd2_with_states_alone_is_taken_for_a_run_of_states(tmp_path):
    """Ninth review of PR #19 (P2): a summary of 2ecbcd2 (no experiment_of_compound, no compounds, no notes of the
    compounds on its edges) of the network of the 29 states of A2A in which only 4jS -> 4jR had a ddG was taken for a
    run without states: the edge within 4j was compared with an experimental ddG of 0 (+0.4 of error). A name that
    only a state of the benchmark has makes it a run of states; a summary of compounds stays one."""
    from cufep.benchmarks.rbfe import read_states, report

    states = read_states("a2a")
    exp = {s: float(r["exp_dg_kcal_mol"]) for s, r in states.items()}
    old = {"ligands": list(states), "cycle_closures": [], "dg": None, "experiment_kcal_mol": exp,
           "notes": ["the edges with a ddG do not connect all ligands: no dG"],
           "edges": [{"edge": "4jS~4jR", "ligand_a": "4jS", "ligand_b": "4jR", "ddg": 0.4, "se": 0.3,
                      "not_reliable": []}]}
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "summary.json").write_text(json.dumps(old))
    report("a2a", tmp_path / "old")
    row = next(csv.DictReader((tmp_path / "old" / "rbfe_report_edges.csv").open()))
    assert row["exp_ddg"] == "" and row["exp_ddg_of"] == "the compounds"
    ref = read_reference("a2a")
    compounds = {"ligands": list(ref), "cycle_closures": [], "dg": None, "notes": [],
                 "experiment_kcal_mol": {c: v["exp"] for c, v in ref.items()},
                 "edges": [{"edge": "4b~4e", "ligand_a": "4b", "ligand_b": "4e", "ddg": 0.4, "se": 0.3,
                            "not_reliable": []}]}
    (tmp_path / "compounds").mkdir()
    (tmp_path / "compounds" / "summary.json").write_text(json.dumps(compounds))
    report("a2a", tmp_path / "compounds")
    row = next(csv.DictReader((tmp_path / "compounds" / "rbfe_report_edges.csv").open()))
    assert float(row["exp_ddg"]) == pytest.approx(ref["4e"]["exp"] - ref["4b"]["exp"])
    assert row["exp_ddg_of"] == "the ligands"


def test_a_run_of_ligands_named_as_states_is_reported_as_a_run_of_ligands(tmp_path):
    """Tenth review of PR #19 (P2): 4h and 4jS run as ligands, without system.states: their dG were in the summary
    with the reference of ligands, and the report took the run for one of states by the name 4jS and gave "Compounds
    with a dG: 0 of 2", with its network.json and without it. The network says whether the run had states, and a
    summary that records its format is read by it; the names decide only for a summary that records neither."""
    import shutil

    from cufep.benchmarks.rbfe import report

    exp = {"4h": -10.0, "4jS": -10.887}
    summary = {"ligands": ["4h", "4jS"], "cycle_closures": [], "notes": [], "run_directory": "runs/x",
               "experiment_kcal_mol": exp,
               "dg": {"reference": "mean of the experimental values", "reference_ligands": ["4h", "4jS"],
                      "values": {"4h": {"dg": -10.2, "se": 0.2}, "4jS": {"dg": -10.6, "se": 0.2}}},
               "edges": [{"edge": "4h~4jS", "ligand_a": "4h", "ligand_b": "4jS", "ddg": -0.4, "se": 0.3,
                          "exp_ddg": exp["4jS"] - exp["4h"], "exp_ddg_of": "the ligands", "not_reliable": []}]}
    (tmp_path / "with").mkdir()
    (tmp_path / "with" / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "with" / "network.json").write_text(json.dumps({"ligands": ["4h", "4jS"], "edges": [["4h", "4jS"]],
                                                                "edges_from": "given"}))
    shutil.copytree(tmp_path / "with", tmp_path / "alone")
    (tmp_path / "alone" / "network.json").unlink()
    for name in ("with", "alone"):
        text = report("a2a", tmp_path / name)
        assert "Ligands with a dG: 2 of 2" in text, name
        rows = {r["ligand"]: r for r in csv.DictReader((tmp_path / name / "rbfe_report_ligands.csv").open())}
        assert float(rows["4jS"]["cufep_rbfe"]) == pytest.approx(-10.6), name


def test_the_report_compares_edges_with_the_experiment_of_its_table_and_dg_of_any_origin(tmp_path):
    """Eleventh review of PR #19: (P2) without ejm_31 and ejm_46 in the experiment of the run, the table of ligands
    took the benchmark's values and the edge ejm_31 -> ejm_46 (-1.770 kcal/mol in the benchmark) had none and left
    the statistics of the edges; (P2) a run without experimental values sets the mean of its dG over the network to 0
    (+0.885 and -0.885 for a ddG of -1.770 reproduced exactly), and the report compared them with -9.540 and -11.310:
    a raw RMSE of 10.425. The edges take the experiment of the table, and dG of another origin are shifted to the mean
    of the experiment compared, which the report says; (P3) the axes of the figures hold the ends of the error bars."""
    from cufep.benchmarks.rbfe import axis_range, report

    ref = read_reference("tyk2")
    exact = ref["ejm_46"]["exp"] - ref["ejm_31"]["exp"]
    summary = {"ligands": ["ejm_31", "ejm_46"], "cycle_closures": [], "notes": [], "experiment_kcal_mol": {},
               "dg": {"reference": "mean of the network", "reference_ligands": [],
                      "values": {"ejm_31": {"dg": -exact / 2, "se": 0.1}, "ejm_46": {"dg": exact / 2, "se": 0.1}}},
               "edges": [{"edge": "ejm_31~ejm_46", "ligand_a": "ejm_31", "ligand_b": "ejm_46", "ddg": exact,
                          "se": 0.2, "network_ddg": exact, "network_se": 0.2, "not_reliable": []}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = report("tyk2", tmp_path)
    row = next(csv.DictReader((tmp_path / "rbfe_report_edges.csv").open()))
    assert float(row["exp_ddg"]) == pytest.approx(exact, abs=1e-6)
    assert "| cuFEP, own | 1 | 0.00 |" in text
    raw = next(line for line in text.splitlines() if line.startswith("| RMSE, raw"))
    assert raw.split("|")[2].strip().startswith("0.00") and "computed again from the ddG of its edges" in text
    assert axis_range([{"exp": -1.77, "calc": -1.77, "err": 3.0}], "exp", (("calc", "err", "calc"),))[1] \
        == pytest.approx(-1.77 + 3.0 + 1.0)
    # twelfth review of PR #19: ejm_31 -> X computed as +2 +- 1 kcal/mol without experimental values, the benchmark
    # with ejm_31 alone: the run gives (-1, +1) +- (0.5, 0.5) about the mean of the network; on the scale of ejm_31
    # (-9.54) the same estimate gives the standard errors (0, 1), which the dG shifted to it had not
    summary = {"ligands": ["ejm_31", "xlig"], "cycle_closures": [], "notes": [], "experiment_kcal_mol": {},
               "dg": {"reference": "mean of the network", "reference_ligands": [],
                      "values": {"ejm_31": {"dg": -1.0, "se": 0.5}, "xlig": {"dg": 1.0, "se": 0.5}}},
               "edges": [{"edge": "ejm_31~xlig", "ligand_a": "ejm_31", "ligand_b": "xlig", "ddg": 2.0, "se": 1.0,
                          "network_ddg": 2.0, "network_se": 1.0, "not_reliable": []}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    report("tyk2", tmp_path)
    rows = {r["ligand"]: r for r in csv.DictReader((tmp_path / "rbfe_report_ligands.csv").open())}
    assert float(rows["ejm_31"]["cufep_rbfe"]) == pytest.approx(ref["ejm_31"]["exp"], abs=1e-9)
    assert float(rows["xlig"]["cufep_rbfe"]) == pytest.approx(ref["ejm_31"]["exp"] + 2.0, abs=1e-9)
    assert float(rows["ejm_31"]["cufep_rbfe_error"]) == pytest.approx(0.0, abs=1e-9)
    assert float(rows["xlig"]["cufep_rbfe_error"]) == pytest.approx(1.0, abs=1e-9)
    # without an experimental value the dG stay about the mean of the network, and the report says so
    summary["ligands"], summary["edges"][0]["ligand_a"] = ["ylig", "xlig"], "ylig"
    summary["dg"]["values"] = {"ylig": {"dg": -1.0, "se": 0.5}, "xlig": {"dg": 1.0, "se": 0.5}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert "their mean over the network set to 0 (no experimental value" in report("tyk2", tmp_path)


def test_the_report_of_a_summary_alone_gives_the_errors_of_the_values_it_shows(tmp_path):
    """Thirteenth review of PR #19: (P2) a summary.json alone of a run of states (A2A: state h of compound 4h, state x
    of compound X, h -> x +2 kcal/mol, no experimental values in the run, 4h alone in the table) was shifted, and its
    table and figure gave the standard errors of the mean of the network; the summary records what the estimate takes
    of the states, and the report computes the dG and their errors again on the scale of 4h. A summary of an earlier
    version that does not record them is shifted, and its table says what the errors are about. (P2) a network that
    gave no dG was said to have been given no experimental value."""
    from test_rbfe_network import _leg_result, _write_legs

    from cufep.benchmarks.rbfe import report
    from cufep.rbfe.run import combine_network

    run = tmp_path / "run"
    _write_legs(run, "h~x", 0, _leg_result(-10.0), _leg_result(-12.0))
    states = {"compound": {"h": "4h", "x": "X"}, "form": {"h": "", "x": ""}, "fraction": {"h": 1.0, "x": 1.0},
              "symmetry": {"4h": 1, "X": 1}}
    summary = combine_network(run, ["h", "x"], [("h", "x")], 1, {}, states, 298.15)
    se_edge = summary["edges"][0]["se"]
    alone = tmp_path / "alone"
    alone.mkdir()
    (alone / "summary.json").write_text(json.dumps(summary))
    report("a2a", alone)
    rows = {r["ligand"]: r for r in csv.DictReader((alone / "rbfe_report_ligands.csv").open())}
    exp_4h = read_reference("a2a")["4h"]["exp"]
    assert float(rows["4h"]["cufep_rbfe"]) == pytest.approx(exp_4h, abs=1e-6)
    assert float(rows["X"]["cufep_rbfe"]) == pytest.approx(exp_4h + 2.0, abs=1e-6)
    assert float(rows["4h"]["cufep_rbfe_error"]) == pytest.approx(0.0, abs=1e-9)
    assert float(rows["X"]["cufep_rbfe_error"]) == pytest.approx(se_edge, abs=1e-9)
    # a summary of an earlier version without the states: shifted, and the table says what the errors are about
    del summary["states"]
    (alone / "summary.json").write_text(json.dumps(summary))
    text = report("a2a", alone)
    rows = list(csv.DictReader((alone / "rbfe_report_ligands.csv").open()))
    assert "shifted by" in text and all(r["cufep_rbfe_error_of"].startswith("the dG of the run") for r in rows)
    # no dG from the network: said so, not "no experimental value"
    summary["dg"] = None
    (alone / "summary.json").write_text(json.dumps(summary))
    text = report("a2a", alone)
    assert "The network of the run gave no dG" in text and "no experimental value" not in text
