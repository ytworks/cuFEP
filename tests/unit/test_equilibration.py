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
