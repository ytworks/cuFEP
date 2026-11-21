"""A number that the data do not determine is never reported as one, and a command ends by what it was asked
to do (review of PR 1 on a3462a5)."""
import json
from unittest import mock

import numpy as np
import pytest

import cufep.analysis.mbar as mbar
from cufep.analysis.mbar import analyze_leg
from cufep.benchmarks.stats import bootstrap, metrics


def _independent(u, states):
    return np.ones((u.shape[2], 2))


def test_the_whole_production_needs_the_samples_every_part_needs():
    """P2: two windows, eight iterations, three samples per window after decorrelation: every part of the
    production had no estimate, the whole gave -0.948 +- 0.946 kJ/mol. It now has none, and says why."""
    rng = np.random.default_rng(4)
    n = 8
    x = np.stack([rng.normal(0.0, 1.0, n), rng.normal(0.5, 1.0, n)], axis=1)
    samples = {"u": np.stack([0.5 * x ** 2, 0.5 * (x - 0.5) ** 2], axis=2), "states": np.tile(np.arange(2), (n, 1))}
    every_third = lambda u, states: np.full((u.shape[2], 2), 2.7)              # noqa: E731
    with mock.patch.object(mbar, "neighbour_inefficiencies", every_third):
        res = analyze_leg(samples, 298.15, n_equilibration=0)
        assert res.window_kept_samples == [3, 3]
        assert res.not_determined == ("lambda window 0 keeps 3 sample(s) in the production; 4 are needed for an "
                                      "estimate")
        assert np.isnan(res.delta_g) and np.isnan(res.delta_g_error)
        with mock.patch.object(mbar, "too_few_samples", lambda counts, where="there": ""):
            before = analyze_leg(samples, 298.15, n_equilibration=0)       # what the analysis gave without the rule
        assert np.isfinite(before.delta_g) and before.not_determined == ""


def test_the_error_of_the_bar_sum_is_one_of_the_sum():
    """P2: neighbouring BAR pairs share a window. Three windows whose ends are one Hamiltonian, the samples of the
    last the same as those of the first: the sum is exactly 0 in every resample, while the error in quadrature of
    the pairs is not 0 (the reviewer's example gave 0.158 kT, this one 0.0287 kT). The error now comes from the
    same time blocks as that of MBAR."""
    rng = np.random.default_rng(11)
    n = 600
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.7, 1.0, n)
    x = np.stack([a, b, a], axis=1)                    # the third replica repeats the first
    centres = np.array([0.0, 0.7, 0.0])
    u = 0.5 * (x[:, :, None] - centres[None, None, :]) ** 2
    samples = {"u": u, "states": np.tile(np.arange(3), (n, 1))}
    with mock.patch.object(mbar, "neighbour_inefficiencies", _independent):
        res = analyze_leg(samples, 298.15, n_equilibration=0, error_blocks=(40, 20, 10, 5))
    kt = mbar.R_GAS * 298.15
    pairs_error = mbar.bar_between_neighbours(*mbar.decorrelated(u, samples["states"], np.ones(3)))[1]
    assert pairs_error == pytest.approx(0.0287, abs=0.0005)               # what was reported before
    assert res.bar_delta_g == pytest.approx(0.0, abs=1e-9)
    assert res.bar_delta_g_error == pytest.approx(0.0, abs=1e-9 * kt) and res.bar_delta_g_error_n_blocks == 40


def test_the_bar_error_follows_the_blocks_of_mbar_on_ordinary_windows():
    """On independent windows the jackknife of the sum agrees with the error in quadrature of the pairs."""
    rng = np.random.default_rng(12)
    n = 2000
    centres = np.array([0.0, 0.6, 1.2, 1.8])
    x = centres[None, :] + rng.normal(size=(n, 4))
    u = 0.5 * (x[:, :, None] - centres[None, None, :]) ** 2
    samples = {"u": u, "states": np.tile(np.arange(4), (n, 1))}
    with mock.patch.object(mbar, "neighbour_inefficiencies", _independent):
        res = analyze_leg(samples, 298.15, n_equilibration=0, error_blocks=(40, 20, 10, 5))
    kt = mbar.R_GAS * 298.15
    pairs_error = mbar.bar_between_neighbours(*mbar.decorrelated(u, samples["states"], np.ones(4)))[1] * kt
    assert res.bar_delta_g_error == pytest.approx(pairs_error, rel=0.35)


# ---- benchmark statistics ---------------------------------------------------------------------------------

def test_every_resample_counts_for_the_errors():
    """P2: errors [0, 0, 0, 4]: over all 256 resamples the 95 % interval of the mean signed error is [0, 3];
    dropping resamples of fewer than three different ligands gave [0, 2]."""
    exp = np.array([-9.0, -8.0, -7.0, -6.0])
    pred = exp + np.array([0.0, 0.0, 0.0, 4.0])
    ci = bootstrap(pred, exp, n_boot=20000, seed=1)
    assert ci["mean_signed_error"] == (0.0, 3.0)


def test_correlations_are_computed_where_they_are_defined_and_shown_as_na_where_not():
    """P2: two ligands, predictions [-10, -8], experiment [-9, -7]: R^2 is 1, and was shown as 0.00."""
    m = metrics([-10.0, -8.0], [-9.0, -7.0])
    assert m["pearson_r2"] == pytest.approx(1.0) and m["spearman"] == pytest.approx(1.0)
    assert m["kendall_tau"] == pytest.approx(1.0)
    one = metrics([-10.0], [-9.0])
    assert all(one[key] is None for key in ("pearson_r2", "pearson_r", "spearman", "kendall_tau"))
    assert one["pairwise_rmse"] is None and one["mean_signed_error"] == -1.0
    flat = metrics([-10.0, -10.0, -10.0], [-9.0, -7.0, -8.0])
    assert flat["pearson_r2"] is None and flat["kendall_tau"] is None


def test_the_report_shows_na_for_an_undefined_statistic(tmp_path, monkeypatch, repo):
    from cufep.benchmarks import runner

    summary = [{"ligand": "jmc_23", "dg_bind": -11.0, "dg_bind_error": 0.3, "n_repeats": 1, "warnings": []}]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    text = runner.report("tyk2", tmp_path)
    assert "| Pearson R^2 | N/A | N/A |" in text and "| Kendall tau | N/A | N/A |" in text


# ---- how a command ends ----------------------------------------------------------------------------------

def _summary(ligand, selected, failed=(), dg=None, inconsistent=()):
    s = {"ligand": ligand, "selected_legs": [list(x) for x in selected], "failed_leg_ids": [list(x) for x in failed],
         "n_repeats_planned": 1, "inconsistent_legs": list(inconsistent), "warnings": []}
    if dg is not None:
        s["dg_bind"] = dg
    return s


@pytest.mark.parametrize("command", [["run", "config.yaml"], ["benchmark", "run", "tyk2"]])
def test_a_command_ends_by_the_legs_it_ran(tmp_path, monkeypatch, command):
    """P2: a leg of this run without a free energy ended with 0 because another ligand had a result; a solvent leg
    run alone in a new directory ended with 1 because the complex leg was not there yet."""
    from click.testing import CliRunner

    from cufep import cli
    from cufep import run as driver
    from cufep.benchmarks import runner

    (tmp_path / "config.yaml").write_text("system: {protein: p.pdb, ligand: l.sdf}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "report", lambda name, directory: "report")
    cases = {
        "a leg of this run has no free energy": ([_summary("a", [(0, "solvent")], failed=[(0, "solvent")]),
                                                  _summary("b", [(0, "complex"), (0, "solvent")], dg=-9.0)], 1),
        "a solvent leg alone, its complex leg not there yet": ([_summary("a", [(0, "solvent")])], 0),
        "a failed leg that this run did not run": ([_summary("a", [(0, "solvent")], failed=[(0, "complex")])], 0),
        "every leg of this run has a result": ([_summary("a", [(0, "complex"), (0, "solvent")], dg=-9.0)], 0),
        "legs made under different conditions": ([_summary("a", [(0, "solvent")], inconsistent=["temperature"])], 1),
    }
    for name, (summary, code) in cases.items():
        monkeypatch.setattr(driver, "run", lambda *a, summary=summary, **k: summary)
        result = CliRunner().invoke(cli.main, command)
        assert result.exit_code == code, (name, result.output)
        if name.startswith("a solvent leg alone"):
            assert "the binding free energy needs the complex leg as well" in result.output


def test_the_ligand_records_which_legs_have_no_free_energy(tmp_path):
    from cufep.workflow import combine_ligand

    for leg, result in (("complex", None), ("solvent", {"leg": "solvent", "delta_g_decouple": float("nan")})):
        d = tmp_path / "repeat_0" / leg
        d.mkdir(parents=True)
        if result is None:
            (d / "error.json").write_text(json.dumps({"error": "the job process ended without a result"}))
        else:
            (d / "result.json").write_text(json.dumps(result))
    out = combine_ligand(tmp_path, "lig", 1)
    assert sorted(map(tuple, out["failed_leg_ids"])) == [(0, "complex"), (0, "solvent")]


# ---- review of PR 1 on 64751ca: the isotropic tail and the acceptance of exchange -------------------------

def test_the_isotropic_tail_is_averaged_over_the_volumes_of_the_coupled_state():
    """P2: in NPT the homogeneous tail is C / V in every configuration; 298.15 K, volumes 90, 90, 110, 110 nm^3,
    s6 = 120, s12 = 0, cutoff 1 nm: C / <V> gave -20.106 kJ/mol, -kT ln <exp(-C / (kT V))> is -21.062."""
    series = mbar.TailSeries("isotropic", np.arange(4), np.array([90.0, 90.0, 110.0, 110.0]), 298.15,
                             s6=120.0, s12=0.0, cutoff_nm=1.0)
    value, error, n, why = series.estimate(slice(None))
    assert why == "" and n == 4 and value == pytest.approx(-21.062, abs=0.001)
    c = 16.0 * np.pi * (-120.0 / 3.0)
    assert c / 100.0 == pytest.approx(-20.106, abs=0.001)                    # what was reported before
    assert series.weights(slice(None))[0] is not None


def test_the_acceptance_of_exchange_counts_the_proposals():
    """P2: ten windows of one Hamiltonian mixed for 100 iterations accept every proposed swap; the measure reported
    as the acceptance (replicas that had moved to a neighbouring window after an iteration) gave 0.14-0.23."""
    from cufep.sampling.hrex import mix_replicas

    rng = np.random.default_rng(3)
    k, n = 10, 100
    state = np.arange(k)
    states, swaps = [], []
    for _ in range(n):
        counts = np.zeros((k - 1, 2), dtype="i8")
        states.append(state.copy())
        state, _acc, _att = mix_replicas(np.zeros((k, k)), state, rng, neighbour_counts=counts)
        swaps.append(counts)
    samples = {"swaps": np.array(swaps), "states": np.array(states)}
    acceptance = mbar.swap_acceptance(samples, 0, n)
    assert acceptance == [1.0] * (k - 1)
    fraction = mbar.neighbour_swap_fraction(np.array(states))
    assert max(fraction) < 0.35                                             # far from the true acceptance
    assert mbar.swap_acceptance({"states": np.array(states)}, 0, n) is None  # samples of an earlier version


def test_swaps_are_stored_with_their_rows_and_absent_where_a_part_lacks_them(tmp_path):
    from cufep.sampling.storage import LegStorage, read_samples

    leg = tmp_path / "hrex"
    s = LegStorage(leg, 2, 2)
    s.append(np.zeros((2, 2)), np.arange(2), np.ones(2), swaps=np.array([[3, 2]]))
    s.append(np.zeros((2, 2)), np.arange(2), np.ones(2), swaps=np.array([[4, 1]]))
    s.flush()
    assert read_samples(leg)["swaps"].tolist() == [[[3, 2]], [[4, 1]]]
    s.append(np.zeros((2, 2)), np.arange(2), np.ones(2))                    # a part without them
    s.flush()
    s.close()
    data = read_samples(leg)
    assert "swaps" not in data and data["u"].shape[0] == 3
