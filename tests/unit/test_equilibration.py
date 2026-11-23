"""Automatic equilibration detection of the analysis (``analysis.detect_equilibration``): the rule
of openmmtools (Chodera 2016), checked against a series whose answer is known and against pymbar."""
import numpy as np
import pytest

from cufep.analysis.mbar import EQUILIBRATION_ORIGINS, detect_equilibration


def as_samples(series: np.ndarray):
    """One replica in one state whose reduced potential is the series."""
    series = np.asarray(series, dtype=float)
    return series[:, None, None], np.zeros((len(series), 1), dtype=int)


def relaxing(n: int = 2000, start: int = 300, seed: int = 5) -> np.ndarray:
    """White noise that sits 10 standard deviations higher during its first ``start`` values."""
    series = np.random.default_rng(seed).normal(size=n)
    series[:start] += 10.0
    return series


def test_the_start_is_where_the_most_uncorrelated_samples_are_left():
    """A series that is 10 standard deviations off for its first 300 values of 2000 and white noise
    after: from any earlier origin the step makes the series look correlated over hundreds of
    values, from 300 on (an origin: the 100 origins are 20 apart) all 1701 values count."""
    found = detect_equilibration(*as_samples(relaxing()))
    assert found["iterations"] == 300
    assert found["statistical_inefficiency"] == pytest.approx(1.0, abs=0.3)
    assert found["effective_samples"] == pytest.approx(1701.0 / found["statistical_inefficiency"])


def test_the_detection_is_that_of_pymbar_on_the_origins_of_openmmtools():
    """The same origin, statistical inefficiency and number of samples as pymbar's
    detect_equilibration with the origins 20 apart, for a series whose best origin is not the
    first one (pymbar may take the first, openmmtools never does)."""
    from pymbar import timeseries

    series = relaxing(seed=11)
    t, g, n_effective = timeseries.detect_equilibration(series, fast=True, nskip=len(series) // EQUILIBRATION_ORIGINS)
    found = detect_equilibration(*as_samples(series))
    assert t > 0 and found["iterations"] == t
    assert found["statistical_inefficiency"] == pytest.approx(g, rel=1e-6)
    assert found["effective_samples"] == pytest.approx(n_effective, rel=1e-6)


def test_the_first_origin_is_never_taken():
    """A series that is the same from every origin (alternating values, so the samples that are
    left only fall as the origin moves on): pymbar takes the first origin, the rule of openmmtools
    the second (the first 1 % of the series goes)."""
    from pymbar import timeseries

    series = np.where(np.arange(2000) % 2 == 0, 1.0, -1.0)
    assert timeseries.detect_equilibration(series, fast=True, nskip=20)[0] == 0
    found = detect_equilibration(*as_samples(series))
    assert found["iterations"] == 20
    g = timeseries.statistical_inefficiency(series[20:], fast=True)
    assert found["statistical_inefficiency"] == pytest.approx(g)
    assert found["effective_samples"] == pytest.approx(1981 / g)
    assert detect_equilibration(*as_samples(series), origins=50)["iterations"] == 40


def test_the_series_is_the_effective_energy_of_the_replicas():
    """The series is the sum over the replicas of the reduced potential at the state each was
    simulated in: a step that only those energies carry is found, whatever the other states hold."""
    rng = np.random.default_rng(3)
    n, n_rep, n_states = 1000, 3, 4
    states = rng.integers(0, n_states, size=(n, n_rep))
    u = rng.normal(size=(n, n_rep, n_states)) * 50.0                   # the energies at the other states: noise
    own = rng.normal(size=(n, n_rep)) / np.sqrt(n_rep)
    own[:200] += 10.0 / n_rep
    u[np.arange(n)[:, None], np.arange(n_rep)[None, :], states] = own
    assert detect_equilibration(u, states)["iterations"] == 200


def test_a_series_without_two_origins_or_without_change_has_no_start():
    assert detect_equilibration(*as_samples(np.arange(2.0))) is None
    assert detect_equilibration(*as_samples(np.full(500, 3.0))) is None


def relaxing_leg(leg_dir, n=1020, n_equilibration=20, relax=300, tail_until=None):
    """A leg directory as a finished run leaves it: three replicas whose energies sit 10 kT higher
    during the first ``relax`` iterations of the production, and a tail energy every fourth
    iteration (until iteration ``tail_until``, if given)."""
    from cufep.sampling.storage import LegStorage

    rng, tail = np.random.default_rng(4), np.random.default_rng(5)
    storage = LegStorage(leg_dir / "hrex", 3, 3)
    storage.check_conditions({"temperature_K": 298.15, "pressure_bar": 1.0, "barostat": "isotropic",
                              "lrc": {"cutoff_nm": 0.9, "long_cutoff_nm": 2.0, "n_ligand": 3, "n_environment": 30,
                                      "s6": 1.0, "s12": 1.0, "sha256": "x"}})
    storage.set_meta(temperature_K=298.15, iteration_ps=2.5, backend="openmm", leg=leg_dir.name, lrc_s6=1.0,
                     lrc_s12=1.0, lrc_cutoff_nm=0.9)
    storage.set_plan(n_equilibration, n)
    for it in range(n):
        x = rng.normal(size=3)
        high = 10.0 if it < n_equilibration + relax else 0.0
        storage.append(np.repeat((0.5 * x ** 2 + high)[:, None], 3, axis=1), np.arange(3), np.ones(3))
        if it % 4 == 0 and (tail_until is None or it < tail_until):
            storage.append_record(it, {"lrc_du": -2.0 + 0.1 * tail.normal()})
    storage.close()


def test_the_detection_moves_the_start_of_the_analysis_of_a_leg_only_where_it_is_asked_for(tmp_path):
    """``analysis.detect_equilibration`` through every way to a result. Off (the default) the
    result is the one of the production after the equilibration of the run, and its record of the
    analysis is the one every earlier result has. On, the iterations the detection leaves out go
    as well: the result is that of an analysis given that start, records what was left out, and
    keeps the end of the equilibration of the run; `cufep analyze` gives it with a configuration
    that asks for it and with the settings a result records; and the check of what an analysis
    needs from the samples takes the same production."""
    from cufep import jsonio
    from cufep.config.schema import AnalysisConfig, Config
    from cufep.sampling.storage import read_samples
    from cufep.workflow import (
        WorkflowError,
        analyse_leg_directory,
        leg_result_from_samples,
        samples_analysis_problems,
    )

    leg, run = tmp_path / "repeat_0" / "solvent", {"leg": "solvent"}
    relaxing_leg(leg)
    off, on = AnalysisConfig(), AnalysisConfig(detect_equilibration=True)
    plain = leg_result_from_samples(leg, off, run)
    assert plain["equilibration_detected"] is None and plain["analysis"]["n_iterations_used"] == 1000
    assert "detect_equilibration" not in plain["analysis_record"]

    samples = read_samples(leg / "hrex")
    expected = detect_equilibration(samples["u"][20:], samples["states"][20:])
    t = expected["iterations"]
    # the energies are high for 300 iterations: none of them is kept (which of the later origins has
    # the most uncorrelated samples follows the noise of the series)
    assert 300 <= t < 1000
    found = leg_result_from_samples(leg, on, run)
    assert found["equilibration_detected"] == expected and found["n_equilibration_iterations"] == 20
    assert found["analysis"]["n_iterations_used"] == 1000 - t
    assert found["production_ns"] == pytest.approx((1000 - t) * 2.5 / 1000.0)
    assert found["analysis_record"]["detect_equilibration"] is True
    # the numbers are those of an analysis that is given that start, and not those of the default
    given = leg_result_from_samples(leg, off, run, n_equilibration=20 + t)
    for key in ("delta_g_decouple", "delta_g_error", "lrc"):
        assert found[key] == given[key]
    assert found["analysis"]["f_k"] == given["analysis"]["f_k"]
    assert found["lrc"] != plain["lrc"]

    # cufep analyze: with a configuration that asks for it, and with the settings a result records
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                 "analysis": {"detect_equilibration": True}})
    assert analyse_leg_directory(leg, cfg)["result"]["delta_g_decouple"] == found["delta_g_decouple"]
    assert analyse_leg_directory(leg)["result"]["equilibration_detected"] is None       # no result yet: defaults
    (leg / "result.json").write_text(jsonio.dumps(found))
    again = analyse_leg_directory(leg)
    assert again["source"] == "the settings the leg's result records"
    assert again["result"]["equilibration_detected"] == expected and again["n_equilibration"] == 20

    # tail energies only before the start the detection finds: nothing is missing without it, and
    # with it the check before a run says what the analysis then says
    short = tmp_path / "short" / "repeat_0" / "solvent"
    relaxing_leg(short, tail_until=320)
    assert samples_analysis_problems(short, off) == []
    assert "tail correction of their production cannot be estimated" in samples_analysis_problems(short, on)[0]
    leg_result_from_samples(short, off, run)
    with pytest.raises(WorkflowError, match="tail correction"):
        leg_result_from_samples(short, on, run)
