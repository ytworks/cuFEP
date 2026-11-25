"""The thermodynamic length of a stage, and what a length says about two neighbouring windows
(cufep.alchemy.thermodynamic_length), on windows whose answers are known.

Unit-width harmonic windows: the energy difference between two of them is normally distributed,
their distance is that of their centres, an exchange between them is accepted with probability
erfc(d / 2), and the length of a path is the distance its centre travels.
"""
import math

import numpy as np
import pytest

from cufep.alchemy.schedule import pair_overlap
from cufep.alchemy.thermodynamic_length import (
    exchange_acceptance,
    gaussian_acceptance,
    gaussian_pair_overlap,
    gradient_sd,
    integrated_length,
    length_of_acceptance,
    pair_differences,
    pair_distance,
    windows_added_by_length,
)
from cufep.analysis.mbar import analyze_leg
from cufep.sampling.hrex import mix_replicas


def two_windows(d, samples=4000, seed=1):
    """Two unit-width harmonic windows ``d`` apart: the energy of the second minus that of the first
    in the samples of each, and the energies of the samples at both windows (what pair_overlap takes)."""
    rng = np.random.default_rng(seed)
    centres = np.array([0.0, d])
    energies = [0.5 * (rng.normal(c, 1.0, size=samples)[:, None] - centres[None, :]) ** 2 for c in centres]
    return energies[0][:, 1] - energies[0][:, 0], energies[1][:, 1] - energies[1][:, 0], energies


def test_two_windows_a_length_apart_exchange_and_overlap_as_the_length_says():
    """The tolerances are four times the scatter of five seeds (0.008 for the acceptance, 0.003
    for the overlap, 0.5 % for the distance)."""
    for d in (0.5, 1.0, 2.0, 3.0):
        x, y, energies = two_windows(d)
        assert pair_distance(x, y) == pytest.approx(d, rel=0.02)
        assert gaussian_acceptance(d) == pytest.approx(math.erfc(d / 2.0))
        assert exchange_acceptance(x, y) == pytest.approx(math.erfc(d / 2.0), abs=0.03)
        # the overlap that the rule of adaptive lambda scheduling reads for a pair alone
        assert pair_overlap(energies, [0, 1])[0] == pytest.approx(gaussian_pair_overlap(d), abs=0.012)
    assert gaussian_pair_overlap(0.0) == pytest.approx(0.5)
    assert length_of_acceptance(gaussian_acceptance(1.3)) == pytest.approx(1.3)


def test_the_acceptance_is_the_mean_over_all_pairs_of_samples():
    """Against the sum over every pair, with energy differences that overflow an exponential and
    infinite ones (configurations with overlapping atoms). A difference that is not a number gives
    no acceptance: it was counted as an exchange that is always accepted (review of stage 1 of the
    plan of 2026-11-25: ``[0.0], [nan]`` gave 1.0 and ``[0.0, 1.0], [nan, -50.0]`` gave 0.5)."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(0.0, 30.0, 300), rng.normal(-5.0, 40.0, 257)
    x[:4] = [1.0e7, -1.0e7, 800.0, np.inf]
    y[:3] = [900.0, -2.0e6, -np.inf]
    every_pair = np.mean(np.exp(np.minimum(y[None, :] - x[:, None], 0.0)))
    assert exchange_acceptance(x, y) == pytest.approx(every_pair, rel=1e-12)
    assert exchange_acceptance([0.0], [0.0]) == 1.0
    assert exchange_acceptance([1.0], [0.0]) == pytest.approx(math.exp(-1.0))
    assert exchange_acceptance([1000005.0], [1000003.0]) == pytest.approx(math.exp(-2.0))
    assert exchange_acceptance([-np.inf, 0.0], [0.0]) == 1.0
    assert math.isnan(exchange_acceptance([0.0], [np.nan])) and math.isnan(exchange_acceptance([np.nan], [0.0]))
    assert math.isnan(exchange_acceptance([0.0, 1.0], [np.nan, -50.0]))
    assert math.isnan(pair_distance([0.0, np.inf], [0.0, 1.0])) and math.isnan(pair_distance([1.0], [2.0, 3.0]))
    # infinite differences of the same sign in both windows have no difference (they gave 1.0), and
    # values whose squares overflow have no standard deviation (they gave inf): review of stage 2
    assert math.isnan(exchange_acceptance([np.inf], [np.inf])) and math.isnan(exchange_acceptance([-np.inf], [-np.inf]))
    assert math.isnan(pair_distance([1.0e200, -1.0e200, 3.0e200], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="samples of both"):
        exchange_acceptance([], [0.0])
    with pytest.raises(ValueError, match="above 0 and at most 1"):
        length_of_acceptance(1.5)

def test_the_length_of_a_stage_is_the_distance_its_windows_travel():
    """A path that is steep near lambda = 1, as a soft-core stage is: x(lambda) = 6 lambda +
    30 lambda^6 on 101 candidates, 36 long. The gradient at a window is the central difference of
    the candidates next to it; 21 windows integrate to 36.09 by the trapezoidal rule (three seeds:
    36.12 +- 0.05)."""
    top = 100
    grid = np.linspace(0.0, 1.0, top + 1)
    centre = 6.0 * grid + 30.0 * grid ** 6
    windows = [5 * i for i in range(21)]
    rng = np.random.default_rng(1)
    sds = []
    for i in windows:
        energies = 0.5 * (rng.normal(centre[i], 1.0, size=4000)[:, None] - centre[None, :]) ** 2
        lo, hi = max(i - 1, 0), min(i + 1, top)
        sds.append(gradient_sd(energies, i, 1.0 / top))
        assert sds[-1] == pytest.approx((centre[hi] - centre[lo]) * top / (hi - lo), rel=0.05)
    length = integrated_length(grid[windows], sds)
    assert length[0] == 0.0 and np.all(np.diff(length) > 0.0)
    assert length[-1] == pytest.approx(36.0, rel=0.01)
    assert length[10] == pytest.approx(centre[50] - centre[0], rel=0.03)      # half of lambda, a tenth of the length
    with pytest.raises(ValueError, match="must increase"):
        integrated_length([0.0, 0.5, 0.5], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="not one of the 101 candidates"):
        gradient_sd(energies, -2, 1.0 / top)                     # an index from the end gave a number


def test_the_differences_of_a_pair_follow_the_windows_through_the_exchanges():
    """Replicas change windows from iteration to iteration: the samples of a window are those of
    whichever replica holds it."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=(6, 4, 4))
    states = np.array([rng.permutation(4) for _ in range(6)])
    pairs = pair_differences(u, states)
    assert len(pairs) == 3
    for j, (x, y) in enumerate(pairs):
        for it in range(6):
            lower, upper = list(states[it]).index(j), list(states[it]).index(j + 1)
            assert x[it] == u[it, lower, j + 1] - u[it, lower, j]
            assert y[it] == u[it, upper, j + 1] - u[it, upper, j]
    states[2] = [0, 0, 2, 3]
    with pytest.raises(ValueError, match="one replica in each state"):
        pair_differences(u, states)


def test_the_analysis_of_a_leg_gives_the_lengths_between_its_windows():
    """Six unit-width harmonic windows at unequal distances whose replicas exchange: the length
    between two neighbours is the distance of their centres, and a stage is as long as its pairs
    together (3000 iterations: a length scatters by 1.3 %)."""
    rng = np.random.default_rng(4)
    centres = np.array([0.0, 0.5, 1.5, 2.5, 3.0, 5.0])
    state_of = np.arange(6)
    u, states = np.zeros((3000, 6, 6)), np.zeros((3000, 6), dtype=int)
    for it in range(3000):
        x = rng.normal(centres[state_of], 1.0)
        u[it] = 0.5 * (x[:, None] - centres[None, :]) ** 2
        states[it] = state_of
        state_of = mix_replicas(u[it], state_of, rng)[0]
    # a solvent leg: discharge over the first three windows, vanish over the last four
    lambdas = ((1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.6), (0.0, 0.3), (0.0, 0.0))
    state_dicts = [{"lambda_restraints": 1.0, "lambda_electrostatics": e, "lambda_sterics": s} for e, s in lambdas]
    res = analyze_leg({"u": u, "states": states, "attrs": {"states": state_dicts}}, 298.15)
    assert res.neighbour_distance == pytest.approx(np.diff(centres), rel=0.05)
    assert res.stage_length == pytest.approx({"discharge": 1.5, "vanish": 3.5}, rel=0.05)
    assert res.to_dict()["neighbour_distance"] == res.neighbour_distance
    # samples that do not say which states their windows are: the pairs, no stages
    bare = analyze_leg({"u": u, "states": states}, 298.15)
    assert bare.neighbour_distance == res.neighbour_distance and bare.stage_length == {}
    # an iteration that does not hold one replica in each state: no lengths, and the analysis is made
    # all the same (it ended with a ValueError: review of stage 2 of the plan of 2026-11-25)
    states[7] = [0, 0, 2, 3, 4, 5]
    broken = analyze_leg({"u": u, "states": states, "attrs": {"states": state_dicts}}, 298.15)
    assert broken.neighbour_distance == [] and broken.stage_length == {}
    assert set(broken.stage_delta_g) == {"discharge", "vanish"}


def test_windows_are_added_where_a_start_is_more_than_a_step_apart_and_no_window_is_taken_away():
    """Intervals of a start that are 1.0, 1.5, 2.8, 0.6 and 0.3 kT long, at a step of 1.32: the second
    gets one window and the third two, evenly in lambda within the interval, and the others none."""
    start, lengths = [0, 20, 40, 60, 80, 100], [0.0, 1.0, 2.5, 5.3, 5.9, 6.2]
    assert windows_added_by_length([(start, lengths)], 1.32, 99) == ([[30, 47, 53]], [False])
    # room for two or four windows only, over two stages: they go where the step is the longest, and
    # the answer says in which stages steps stay longer than asked for
    two_stages = [([0, 50, 100], [0.0, 3.0, 3.1]), ([0, 100], [0.0, 4.0])]
    assert windows_added_by_length(two_stages, 1.32, 2) == ([[25], [50]], [True, True])
    assert windows_added_by_length(two_stages, 1.32, 4) == ([[17, 33], [33, 67]], [False, True])
    assert windows_added_by_length(two_stages, 1.32, 99) == ([[17, 33], [25, 50, 75]], [False, False])
    # an interval takes no more windows than it has candidates between its ends
    assert windows_added_by_length([([0, 2, 100], [0.0, 9.0, 9.5])], 1.32, 99) == ([[1]], [False])
    # a stage without a length, and one whose windows are close enough: nothing
    assert windows_added_by_length([([0, 100], [0.0, 0.0]), ([0, 50, 100], [0.0, 1.3, 2.6])], 1.32, 9) == (
        [[], []], [False, False])
    # the path of the test above from 21 evenly spaced windows that are 0.3 to 8.3 apart: every
    # interval is cut into the fewest steps that are 1.32 long or less, where it has the candidates
    lambdas = np.linspace(0.0, 1.0, 21)
    lengths = integrated_length(lambdas, 6.0 + 180.0 * lambdas ** 5)
    added, limited = windows_added_by_length([([5 * i for i in range(21)], lengths)], 1.32, 99)
    new = np.histogram(added[0], bins=[5 * i for i in range(21)])[0]
    for d, n in zip(np.diff(lengths), new):
        assert n == 4 or (d / (n + 1) <= 1.32 and (n == 0 or d / n > 1.32))
    assert limited == [False] and new[0] == 0 and new[-1] == 4 and sorted(set(added[0])) == added[0]
    with pytest.raises(ValueError, match="numbers that do not fall"):
        windows_added_by_length([([0, 50, 100], [0.0, np.nan, 1.0])], 1.32, 9)
    with pytest.raises(ValueError, match="increasing candidates"):
        windows_added_by_length([([0, 50, 50], [0.0, 1.0, 2.0])], 1.32, 9)
