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
    assert smoke.sampling.production_ns == [0.1]
    subset = benchmark_config("a2a", "subset", tmp_path / "x")
    assert subset.system.ligand_names == load_registry()["a2a"]["subset"]
    assert subset.sampling.repeats == 3 and subset.membrane.enabled == "auto"
    full = benchmark_config("tyk2", "full")
    assert full.system.ligand_names is None and full.sampling.production_ns == [5.0, 10.0]
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
    assert "| mean signed error (kcal/mol) | -1." in text and "0 by construction" in text
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
