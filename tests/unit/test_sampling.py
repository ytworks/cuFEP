"""Replica mixing, MBAR, storage and an inline replica-exchange run on the CPU."""
import json

import numpy as np
import openmm as mm
import pytest

from cufep.alchemy.factory import make_alchemical_system
from cufep.alchemy.schedule import fixed_schedule
from cufep.analysis.mbar import (analyze_leg, assemble, count_round_trips, solve_mbar, stage_boundaries,
                                 statistical_inefficiency)
from cufep.sampling.hrex import ReplicaExchangeSampler, mix_replicas
from cufep.sampling.storage import LegStorage, read_samples
from cufep.sampling.worker import LegSpec


def harmonic_samples(n_iter, k_spring, x0, rng, exchange=True):
    """Exact Gibbs sampling of K harmonic states (beta = 1) with replica mixing."""
    k = len(k_spring)
    sig = 1.0 / np.sqrt(k_spring)
    state_of = np.arange(k)
    u = np.zeros((n_iter, k, k))
    states = np.zeros((n_iter, k), dtype=int)
    xs = np.zeros((n_iter, k))
    for it in range(n_iter):
        x = rng.normal(x0[state_of], sig[state_of])
        u[it] = 0.5 * k_spring[None, :] * (x[:, None] - x0[None, :]) ** 2
        states[it] = state_of
        xs[it] = x
        if exchange:
            state_of, _, _ = mix_replicas(u[it], state_of, rng)
    return u, states, xs


def ar1(n, rho, rng):
    """A stationary series of unit variance whose correlation from one step to the next is ``rho``."""
    z = np.empty(n)
    z[0] = rng.normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal()
    return z


def test_mixing_is_a_permutation_and_accepts_downhill_swaps():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6, 6))
    new, acc, att = mix_replicas(u, np.arange(6), rng)
    assert sorted(new.tolist()) == list(range(6)) and 0 < acc <= att
    # two replicas each sitting in the other's favourite state must swap
    u2 = np.array([[50.0, 0.0], [0.0, 50.0]])
    new, _, _ = mix_replicas(u2, np.array([0, 1]), rng, n_attempts=20)
    assert new.tolist() == [1, 0]


def test_mixing_depends_only_on_the_seed_and_iteration():
    u = np.random.default_rng(5).normal(size=(8, 8))
    a = mix_replicas(u, np.arange(8), np.random.default_rng([7, 3, 0x5EED]))[0]
    b = mix_replicas(u, np.arange(8), np.random.default_rng([7, 3, 0x5EED]))[0]
    c = mix_replicas(u, np.arange(8), np.random.default_rng([7, 4, 0x5EED]))[0]
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()


def test_mixing_preserves_each_state_distribution():
    """Detailed balance: with exchange, the samples collected in state k still follow state k."""
    rng = np.random.default_rng(11)
    k_spring = np.array([1.0, 2.0, 4.0, 8.0])
    x0 = np.array([0.0, 0.4, 0.8, 1.2])
    u, states, xs = harmonic_samples(6000, k_spring, x0, rng)
    assert np.mean(states[1:] != states[:-1]) > 0.2                   # replicas really move
    for k in range(4):
        x = xs[states == k]
        se = 1 / np.sqrt(k_spring[k] * len(x))
        assert abs(x.mean() - x0[k]) < 5 * se
        assert x.var() == pytest.approx(1 / k_spring[k], rel=0.1)


def test_mbar_recovers_harmonic_free_energies():
    rng = np.random.default_rng(3)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    x0 = np.linspace(0.0, 1.0, 6)
    u, states, _ = harmonic_samples(3000, k_spring, x0, rng)
    f, df, overlap = solve_mbar(u, states)
    exact = 0.5 * np.log(k_spring / k_spring[0])                      # -ln Z_k / Z_0
    assert np.all(np.abs(f - exact) < 5 * np.maximum(df, 0.01))
    assert np.allclose(overlap.sum(axis=1), 1.0, atol=1e-6)
    u_kn, n_k = assemble(u, states)
    assert u_kn.shape == (6, 18000) and n_k.tolist() == [3000] * 6
    assert statistical_inefficiency(u, states) < 1.5                  # independent samples
    assert count_round_trips(states) > 5


def test_mbar_start_does_not_change_the_solution_and_jax_is_off():
    """The solve starts from BAR or from given free energies; the answer is that from zeros."""
    import pymbar

    from cufep.analysis.mbar import _pymbar
    _pymbar()
    assert pymbar.mbar_solvers.use_jit is False                       # NumPy solver, see _pymbar
    rng = np.random.default_rng(5)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    u, states, _ = harmonic_samples(1500, k_spring, np.linspace(0.0, 1.0, 6), rng)
    u_kn, n_k = assemble(u, states)
    zeros = pymbar.MBAR(u_kn - u_kn.min(), n_k, solver_protocol="robust").compute_free_energy_differences()
    f, df, overlap = solve_mbar(u, states)                             # BAR start
    assert np.allclose(f, zeros["Delta_f"][0], atol=1e-8) and np.allclose(df, zeros["dDelta_f"][0], atol=1e-8)
    f2, df2, none = solve_mbar(u, states, initial_f_k=f + 0.3, overlap=False)
    assert none is None and np.allclose(f2, f, atol=1e-8) and np.allclose(df2, df, atol=1e-8)
    # a window without samples cannot start from BAR; the solve still works
    keep = states != 2
    u_kn2 = u.reshape(-1, 6)[keep.reshape(-1)].T
    order = np.argsort(states.reshape(-1)[keep.reshape(-1)], kind="stable")
    from cufep.analysis.mbar import overlap_matrix
    o = overlap_matrix(u_kn2[:, order], np.array([1500, 1500, 0, 1500, 1500, 1500]))
    assert o.shape == (6, 6) and np.all(np.isfinite(o))


def test_analysis_of_a_leg_with_a_window_that_overlaps_no_other_is_bounded():
    """Such a leg has no defined free energy; the solver must still stop (it once ran for an hour)."""
    import time

    rng = np.random.default_rng(7)
    k_spring = np.array([1.0, 1.0, 1.5, 2.2, 3.3, 5.0])
    x0 = np.array([-60.0, 0.0, 0.3, 0.6, 0.9, 1.2])                  # window 0 is far from all others
    u, states, _ = harmonic_samples(1500, k_spring, x0, rng)
    assert np.all(states[:, 0] == 0)                                  # it never exchanges
    t0 = time.perf_counter()
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert time.perf_counter() - t0 < 60.0
    assert res.min_overlap < 1e-6
    # no part of such a leg has a determined free energy (review of PR 1: the solver returned its start
    # value for them): only the whole is shown, flagged by its own overlap; the parts say why they are not
    assert [point[0] for point in res.forward] == [1.0] == [point[0] for point in res.reverse]
    assert res.forward[-1][1] == res.delta_g and res.reverse[-1][1] == res.delta_g
    assert len(res.parts_not_determined) == 18 + 5 and "below 0.03: their free energy is not determined" in \
        res.parts_not_determined[0]
    assert res.blocks == [] and "the first half (iterations 0-750 of the production): lambda windows 0 and 1 overlap" \
        in res.convergence_note
    assert all(value is None for value in res.delta_g_error_blocks.values()) and res.delta_g_error_from == "analytical"


def test_bar_between_neighbours_agrees_with_mbar():
    from cufep.analysis.mbar import bar_between_neighbours

    rng = np.random.default_rng(11)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    u, states, _ = harmonic_samples(3000, k_spring, np.linspace(0.0, 1.0, 6), rng)
    f, df, _ = solve_mbar(u, states)
    bar, bar_error = bar_between_neighbours(*assemble(u, states))
    exact = 0.5 * np.log(k_spring[-1] / k_spring[0])
    assert abs(bar - exact) < 5 * bar_error and abs(bar - f[-1]) < 2 * df[-1]
    assert 0.5 * df[-1] < bar_error < 3 * df[-1]


def test_each_lambda_window_is_decorrelated_on_its_own():
    """One slow lambda window: its samples are thinned by its own correlation time.

    The statistical inefficiency of the reduced potential summed over the replicas hides such a
    window; with it, the uncertainty of the leg came out too small.
    """
    from cufep.analysis.mbar import decorrelated, kept_iterations, quality_checks, window_inefficiencies

    rng = np.random.default_rng(31)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2])
    x0 = np.linspace(0.0, 0.6, 4)
    n, k = 4000, 4
    sig = 1.0 / np.sqrt(k_spring)
    # window 2 follows an AR(1) process with correlation 0.9 from one iteration to the next: the
    # statistical inefficiency of x is (1 + 0.9) / (1 - 0.9) = 19, that of x squared 9.5; an energy
    # difference holds both
    rho = 0.9
    z = np.zeros(n)
    z[0] = rng.normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal()
    x = rng.normal(x0, sig, size=(n, k))
    x[:, 2] = x0[2] + sig[2] * z
    u = 0.5 * k_spring[None, None, :] * (x[:, :, None] - x0[None, None, :]) ** 2
    states = np.tile(np.arange(k), (n, 1))                           # no exchange

    g = window_inefficiencies(u, states)
    assert 8.0 < g[2] < 25.0 and np.all(g[[0, 1, 3]] < 1.5)
    assert statistical_inefficiency(u, states) < 0.5 * g[2]           # the summed potential does not show it
    u_kn, n_k = decorrelated(u, states, g)
    assert n_k[2] == int(np.ceil(n / np.ceil(g[2]))) and all(n_k[j] in (n, n // 2) for j in (0, 1, 3))
    assert u_kn.shape == (k, n_k.sum())
    # the samples of window 2 are those of every ceil(g)-th iteration, in order
    first = int(n_k[:2].sum())
    assert np.array_equal(u_kn[:, first:first + n_k[2]], u[::int(np.ceil(g[2])), 2, :].T)
    # a part of the production holds those samples of the whole that lie in it
    stride = int(np.ceil(g[2]))
    part_kn, part_n = decorrelated(u, states, g, 1000, 3000)
    inside = [it for it in range(0, n, stride) if 1000 <= it < 3000]
    assert part_n[2] == len(inside) and kept_iterations(1000, 3000, g[2]).tolist() == inside
    assert np.array_equal(part_kn[:, int(part_n[:2].sum())], u[inside[0], 2, :])
    halves = [decorrelated(u, states, g, 0, n // 2)[1], decorrelated(u, states, g, n // 2, n)[1]]
    assert np.array_equal(halves[0] + halves[1], n_k)                 # the two halves are the whole

    res = analyze_leg({"u": u, "states": states}, 298.15)
    kt = 8.31446261815324e-3 * 298.15
    assert res.slowest_window == 2 and res.window_statistical_inefficiency == pytest.approx(g.tolist())
    assert res.window_effective_samples[2] == pytest.approx(n / g[2])
    assert res.effective_samples_per_state == pytest.approx(n / g[2])
    exact = 0.5 * np.log(k_spring[-1] / k_spring[0]) * kt
    assert abs(res.delta_g - exact) < 5 * res.delta_g_error
    # the uncertainty is larger than that from all samples taken as independent
    naive = solve_mbar(u, states)[1][-1] * kt
    assert res.delta_g_error > 1.15 * naive
    q = quality_checks(res.to_dict())
    assert q["samples"]["ok"] and q["samples"]["window"] == 2 and q["samples"]["fewest"] == pytest.approx(n / g[2])
    assert q["samples"]["statistical_inefficiency"] == pytest.approx(g[2]) and q["samples"]["threshold"] == 50.0
    assert q["samples"]["kept"] == n_k[2] == res.window_kept_samples[2]
    few = quality_checks(res.to_dict(), min_effective_samples=1000.0)["samples"]
    assert few["ok"] is False and few["window"] == 2


def test_checks_of_a_leg_use_the_levels_of_the_literature():
    """Overlap 0.03 first; 50 independent samples; halves within 1 kT; BAR and MBAR within 1 kcal/mol."""
    from cufep.analysis.mbar import quality_checks

    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    x0 = np.linspace(0.0, 1.0, 6)
    u, states, _ = harmonic_samples(3000, k_spring, x0, np.random.default_rng(3))
    steady = analyze_leg({"u": u, "states": states}, 298.15)
    kt = 8.31446261815324e-3 * 298.15
    assert len(steady.forward) == 10 and steady.forward[4][0] == pytest.approx(0.5)
    assert steady.forward[-1][1] == steady.delta_g and steady.reverse[-1][1] == steady.delta_g
    assert np.array(steady.overlap_matrix).shape == (6, 6)
    assert steady.overlap_neighbour[2] == pytest.approx(steady.overlap_matrix[2][3], abs=1e-4)
    assert steady.forward_reverse_gap == pytest.approx(abs(steady.first_half[0] - steady.second_half[0]))
    q = quality_checks(steady.to_dict())
    assert q["overlap"]["ok"] and q["samples"]["ok"] and q["halves"]["ok"] and q["estimators"]["ok"]
    assert q["halves"]["tolerance"] == pytest.approx(kt) and q["halves"]["tolerance_kT"] == 1.0
    assert q["halves"]["difference_kT"] == pytest.approx(steady.forward_reverse_gap / kt)
    assert q["estimators"]["tolerance"] == pytest.approx(4.184) and q["uncertainty"] == steady.delta_g_error
    assert abs(steady.bar_delta_g - steady.delta_g) < 0.1 * kt

    # the last state becomes stiffer half way through the production, as when a slow degree of
    # freedom changes: its free energy rises by 0.5 ln(30 / 10.7) = 0.52 kT in the second half
    late = k_spring.copy()
    late[-1] = 30.0
    u2, states2, _ = harmonic_samples(3000, late, x0, np.random.default_rng(4))
    changed = analyze_leg({"u": np.concatenate([u, u2]), "states": np.concatenate([states, states2])}, 298.15)
    assert (changed.second_half[0] - changed.first_half[0]) / kt == pytest.approx(0.52, abs=0.1)
    # five consecutive blocks: the first two lie before the change, the last two after it
    assert [(b[0], b[1]) for b in changed.blocks] == [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    assert (np.mean([b[2] for b in changed.blocks[3:]]) - np.mean([b[2] for b in changed.blocks[:2]])) / kt \
        == pytest.approx(0.52, abs=0.15)
    assert len(steady.blocks) == 5 and max(abs(b[2] - steady.delta_g) for b in steady.blocks) < 0.3 * kt
    # 0.52 kT is below the level of 1 kT although it is many standard errors: the level is that of
    # the literature, in kT, and the difference in standard errors is reported next to it
    q = quality_checks(changed.to_dict())
    assert q["overlap"]["ok"] and q["halves"]["ok"] and q["halves"]["difference_sigma"] > 5.0
    assert q["halves"]["difference_kT"] == pytest.approx(0.52, abs=0.1)
    assert quality_checks(changed.to_dict(), halves_tolerance_kT=0.3)["halves"]["ok"] is False

    # overlap below the threshold, and an analysis that lacks the numbers of a check
    assert quality_checks(steady.to_dict(), min_overlap=0.9)["overlap"]["ok"] is False
    old = {"delta_g": 10.0, "delta_g_error": 0.5, "min_overlap": 0.2, "overlap_neighbour": [0.2, 0.3]}
    q = quality_checks(old)
    assert q["overlap"] == {"ok": True, "lowest": 0.2, "windows": [0, 1], "threshold": 0.03, "all_ok": True,
                            "used": None, "below": []}
    assert q["samples"]["ok"] is None and q["halves"]["ok"] is None and q["estimators"]["ok"] is None
    # an analysis made before every window was decorrelated on its own: its halves carry no verdict
    assert quality_checks(dict(old, first_half=[12.0, 0.1], second_half=[8.0, 0.1], bar_delta_g=20.0))["halves"]["ok"] is None
    new = dict(old, window_effective_samples=[100.0, 30.0, 80.0], window_statistical_inefficiency=[1.0, 3.4, 1.2])
    q = quality_checks(dict(new, first_half=[float("nan"), float("nan")], second_half=[10.1, 0.7], bar_delta_g=15.0))
    assert q["halves"]["ok"] is False and "difference" not in q["halves"]      # half of the data gives no estimate
    assert q["estimators"]["ok"] is False and q["samples"]["ok"] is False and q["samples"]["window"] == 1
    q = quality_checks(dict(new, first_half=[10.5, 0.4], second_half=[8.5, 0.3], bar_delta_g=12.0))
    assert q["halves"]["ok"] and q["halves"]["difference_kT"] == pytest.approx(2.0 / kt)   # 0.81 kT
    assert q["halves"]["difference_sigma"] == pytest.approx(4.0) and q["estimators"]["ok"]
    assert quality_checks(dict(new, first_half=[10.5, 0.4], second_half=[7.5, 0.3]))["halves"]["ok"] is False


def test_halves_of_the_production_share_no_iteration_and_a_short_production_is_not_assessed(monkeypatch):
    """Review of PR 1: with 12 iterations and a stride of 3 both "halves" were all of the data and agreed."""
    import cufep.analysis.mbar as mbar
    from cufep.analysis.mbar import quality_checks

    # two states; in the second half of the production the second state is much stiffer
    x0 = np.array([0.0, 0.4])
    u1, s1, _ = harmonic_samples(6, np.array([1.0, 1.0]), x0, np.random.default_rng(21))
    u2, s2, _ = harmonic_samples(6, np.array([1.0, 30.0]), x0, np.random.default_rng(22))
    samples = {"u": np.concatenate([u1, u2]), "states": np.concatenate([s1, s2])}
    kt = 8.31446261815324e-3 * 298.15

    # correlated samples (statistical inefficiency 2.125 -> every third iteration): a half holds two
    monkeypatch.setattr(mbar, "neighbour_inefficiencies", lambda u, states: np.full((u.shape[2], 2), 2.125))
    short = analyze_leg(samples, 298.15)
    assert np.isnan(short.first_half[0]) and np.isnan(short.second_half[0]) and np.isnan(short.forward_reverse_gap)
    assert short.convergence_note == ("the first half (iterations 0-6 of the production): lambda window 0 keeps 2 "
                                      "sample(s) there; 4 are needed for an estimate; the second half (iterations "
                                      "6-12 of the production): lambda window 0 keeps 2 sample(s) there; 4 are needed "
                                      "for an estimate")
    # the samples are iterations 0, 3, 6 and 9: a forward part holds four of them from 10 iterations on, a
    # reverse part only as the whole
    assert [round(point[0] * 12) for point in short.forward] == [10, 11, 12] and short.forward[-1][1] == short.delta_g
    assert [point[0] for point in short.reverse] == [1.0]
    assert sum(text.startswith("reverse point") for text in short.parts_not_determined) == 9
    assert short.blocks == []
    verdict = quality_checks(short.to_dict())["halves"]
    assert verdict["ok"] is None and verdict["reason"] == short.convergence_note

    # independent samples: each half is estimated from its own six iterations
    monkeypatch.setattr(mbar, "neighbour_inefficiencies", lambda u, states: np.ones((u.shape[2], 2)))
    res = analyze_leg(samples, 298.15, n_blocks=3)
    first = solve_mbar(samples["u"][:6], samples["states"][:6])[0][-1] * kt
    second = solve_mbar(samples["u"][6:], samples["states"][6:])[0][-1] * kt
    assert res.convergence_note == "" and abs(second - first) > 0.5 * kt       # the two halves really differ
    assert res.first_half[0] == pytest.approx(first, abs=1e-6) and res.second_half[0] == pytest.approx(second, abs=1e-6)
    assert res.forward_reverse_gap == pytest.approx(abs(first - second), abs=1e-6)
    # the forward and reverse series: only parts with enough samples, at the fraction that was used
    used = [round(point[0] * 12) for point in res.forward]
    assert used == [4, 5, 6, 7, 8, 10, 11, 12] and [round(point[0] * 12) for point in res.reverse] == used
    assert all(point[0] == m / 12 for point, m in zip(res.forward, used))
    assert res.forward[2][1] == pytest.approx(first, abs=1e-6) and res.reverse[2][1] == pytest.approx(second, abs=1e-6)
    assert [(b[0], b[1]) for b in res.blocks] == [(0.0, 4 / 12), (4 / 12, 8 / 12), (8 / 12, 1.0)]
    assert res.blocks[0][2] == pytest.approx(solve_mbar(samples["u"][:4], samples["states"][:4])[0][-1] * kt, abs=1e-6)


def test_a_window_is_decorrelated_by_the_slower_of_its_two_neighbours():
    """Review of PR 1: a motion that is slow only in the energy difference to the previous window was missed.

    Three windows of the same curvature at (0, 0), (1.5, 0) and (1.5, 0.8), no exchange. In the middle
    window x is correlated from one iteration to the next by 0.98: its energy difference to the next
    window depends on y alone and looks independent, that to the previous window does not.
    """
    from cufep.analysis.mbar import decorrelated, neighbour_inefficiencies, quality_checks, solve_decorrelated

    centres = np.array([[0.0, 0.0], [1.5, 0.0], [1.5, 0.8]])

    def leg(seed, n=600):
        rng = np.random.default_rng(seed)
        xy = rng.normal(size=(n, 3, 2)) + centres[None]
        xy[:, 1, 0] = 1.5 + ar1(n, 0.98, rng)
        return 0.5 * ((xy[:, :, None, :] - centres[None, None]) ** 2).sum(axis=-1), np.tile(np.arange(3), (n, 1))

    u, states = leg(0)
    table = neighbour_inefficiencies(u, states)
    assert np.isnan(table[0, 0]) and np.isnan(table[2, 1])                  # no such neighbour
    assert table[1, 0] > 40.0 and table[1, 1] < 1.5                          # slow towards window 0 only
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert res.window_inefficiency_neighbour == [1, -1, -1]
    assert res.window_statistical_inefficiency[1] == pytest.approx(table[1, 0])
    assert res.window_kept_samples[1] == int(np.ceil(600 / np.ceil(table[1, 0]))) and res.slowest_window == 1
    stored = res.to_dict()
    assert stored["window_inefficiency_previous"][0] is None
    assert stored["window_inefficiency_previous"][1] == pytest.approx(table[1, 0])
    assert stored["window_inefficiency_next"][1] == pytest.approx(table[1, 1])
    samples = quality_checks(stored)["samples"]
    assert samples["window"] == 1 and samples["neighbour"] == -1 and samples["kept"] == res.window_kept_samples[1]

    # the end states have the same free energy. Over independent runs the reported standard error must
    # be about as large as the scatter of the estimates; from the next window alone it is a third of it
    both, next_only = [], []
    for seed in range(24):
        u, states = leg(seed)
        table = neighbour_inefficiencies(u, states)
        for store, g in ((both, np.nanmax(table, axis=1)),
                         (next_only, np.where(np.isfinite(table[:, 1]), table[:, 1], table[:, 0]))):
            f, df = solve_decorrelated(*decorrelated(u, states, g))
            store.append((f[-1], df[-1]))
    both, next_only = np.array(both), np.array(next_only)
    scatter, reported = both[:, 0].std(ddof=1), np.sqrt((both[:, 1] ** 2).mean())
    assert reported > 0.7 * scatter and abs(both[:, 0].mean()) < 3 * scatter / np.sqrt(len(both))
    assert np.sqrt((next_only[:, 1] ** 2).mean()) < 0.5 * next_only[:, 0].std(ddof=1)
    # the standard error that a leg reports (time blocks where they can be used) agrees with the scatter too
    kt = 8.31446261815324e-3 * 298.15
    legs = [analyze_leg(dict(zip(("u", "states"), leg(seed))), 298.15) for seed in range(24)]
    assert 0.7 * scatter < np.sqrt(np.mean([(r.delta_g_error / kt) ** 2 for r in legs])) < 1.4 * scatter
    # the reported error is that of the time blocks or the analytical one, whichever is larger
    for r in legs:
        usable = [value for value in r.delta_g_error_blocks.values() if value is not None]
        assert r.delta_g_error == pytest.approx(max(usable + [r.delta_g_mbar_error]))
        assert r.delta_g_error_from == ("time blocks" if usable and max(usable) >= r.delta_g_mbar_error else "analytical")
    # 40 blocks of 15 iterations are shorter than the interval at which the slow window keeps its samples
    assert legs[0].delta_g_error_blocks["40"] is None and legs[0].delta_g_error_blocks["5"] is not None
    assert legs[0].delta_g_error_blocks_note["40"].startswith("a block of 15 iteration(s) is shorter than the interval of ")


def travelling_samples(seed, n_iter=600, rho=0.97, k=8):
    """The model of the review of PR 1: ``k`` harmonic windows of the same curvature with centres from
    -0.4 to 0.4 (the end states have the same free energy), started from the stationary distribution.
    From one iteration to the next a configuration keeps ``rho`` of its distance from the centre of the
    window it is in, and the replicas exchange: a slow configuration travels from window to window."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(-0.4, 0.4, k)
    state_of = np.arange(k)
    x = rng.normal(centres[state_of], 1.0)
    u, states = np.zeros((n_iter, k, k)), np.zeros((n_iter, k), dtype=int)
    for it in range(n_iter):
        if it:
            centre = centres[state_of]
            x = centre + rho * (x - centre) + np.sqrt(1 - rho ** 2) * rng.normal(size=k)
        u[it] = 0.5 * (x[:, None] - centres[None, :]) ** 2
        states[it] = state_of
        state_of = mix_replicas(u[it], state_of, rng)[0]
    return {"u": u, "states": states}


def test_the_standard_error_of_a_leg_is_that_of_time_blocks_common_to_all_windows():
    """Review of PR 1: in replica exchange a slow configuration travels between the windows. Every window
    looked decorrelated on its own, the standard error of MBAR was a quarter of the scatter of independent
    runs (0.027 against 0.106 kT over 48 seeds), and all four checks passed."""
    from cufep.analysis.mbar import (BLOCK_COUNTS, TailSeries, TimeRangeEstimator, decorrelated, free_energies_from,
                                     kept_iterations, kept_sample_iterations, quality_checks, solve_decorrelated,
                                     window_inefficiencies)

    kt = 8.31446261815324e-3 * 298.15

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    legs = [analyze_leg(travelling_samples(seed), 298.15) for seed in range(48)]
    scatter = np.std([leg.delta_g for leg in legs], ddof=1)
    assert abs(np.mean([leg.delta_g for leg in legs])) < 3 * scatter / np.sqrt(len(legs))      # the truth is 0
    assert 0.6 * scatter < rms([leg.delta_g_error for leg in legs]) < 1.3 * scatter
    assert rms([leg.delta_g_mbar_error for leg in legs]) < 0.5 * scatter
    # the largest of the errors of the four numbers of blocks is the one that is reported
    first = legs[0]
    assert list(first.delta_g_error_blocks) == [str(count) for count in BLOCK_COUNTS] == ["40", "20", "10", "5"]
    assert first.delta_g_error == max(first.delta_g_error_blocks.values()) and first.delta_g_error_from == "time blocks"
    assert first.delta_g_error == first.delta_g_error_blocks[str(first.delta_g_error_n_blocks)]
    assert first.delta_g_error_note == "" and first.delta_g_error_blocks_note == {} and first.halves_error_blocks == [5, 5]
    # without a tail correction the value is the MBAR part, and so are its errors
    assert first.delta_g == first.delta_g_mbar and first.delta_g_mbar_error_blocks == first.delta_g_error_blocks
    # this leg passed all four checks; the samples check now counts what the standard error leaves
    checks = quality_checks(first.to_dict())
    assert checks["overlap"]["ok"] and checks["halves"]["ok"] and checks["estimators"]["ok"]
    assert first.round_trips > 100 and checks["uncertainty"] == first.delta_g_error
    samples = checks["samples"]
    assert samples["kept"] >= 50 and samples["fewest"] >= 50 and samples["ok"] is False
    assert samples["error_ratio"] == pytest.approx(max(first.delta_g_mbar_error_blocks.values()) / first.delta_g_mbar_error)
    assert samples["error_ratio"] > 2.0
    assert samples["independent"] == pytest.approx(min(samples["kept"], samples["fewest"]) / samples["error_ratio"] ** 2)
    assert not any(quality_checks(leg.to_dict())["samples"]["ok"] for leg in legs)

    # without correlation the blocks give what MBAR gives, and the samples count as they are
    control = [analyze_leg(travelling_samples(seed, rho=0.0), 298.15) for seed in range(48)]
    scatter = np.std([leg.delta_g for leg in control], ddof=1)
    assert 0.9 * scatter < rms([leg.delta_g_error for leg in control]) < 1.35 * scatter
    assert 0.8 * scatter < rms([leg.delta_g_mbar_error for leg in control]) < 1.2 * scatter
    assert all(quality_checks(leg.to_dict())["samples"]["ok"] for leg in control)

    # the jackknife itself: ten blocks of 60 iterations, each estimate without one of them by pymbar
    data = travelling_samples(0)
    g = window_inefficiencies(data["u"], data["states"])
    u_kn, n_k = decorrelated(data["u"], data["states"], g)
    iteration = kept_sample_iterations(600, g)
    assert len(iteration) == n_k.sum() and np.array_equal(iteration[:n_k[0]], kept_iterations(0, 600, g[0]))
    assert np.array_equal(kept_sample_iterations(600, g, 300, 600),
                          np.concatenate([kept_iterations(300, 600, x) for x in g]))
    f_k, _df = solve_decorrelated(u_kn, n_k)
    window = np.repeat(np.arange(8), n_k)
    without = []
    for b in range(10):
        keep = (iteration < 60 * b) | (iteration >= 60 * (b + 1))
        counts = np.bincount(window[keep], minlength=8)
        f, _df = solve_decorrelated(u_kn[:, keep], counts)
        without.append(f[-1])
        if b in (0, 7):                    # the solver of the blocks gives pymbar's free energies
            assert free_energies_from(u_kn[:, keep], counts, f_k) == pytest.approx(f, abs=1e-6)
    by_hand = np.sqrt(9 / 10 * np.sum((np.array(without) - np.mean(without)) ** 2))
    none = TailSeries("none", np.zeros(0), np.zeros(0), 298.15)
    estimator = TimeRangeEstimator(u_kn, n_k, iteration, none, kt, 0.03, int(np.ceil(g.max())), f_k)
    base = estimator.estimate(*estimator.masks(0, 600), f_k)
    value, mbar_part, why = estimator.jackknife(0, 600, 10, base)
    assert why == "" and value == pytest.approx(by_hand * kt, abs=1e-6) == mbar_part
    assert value == pytest.approx(first.delta_g_error_blocks["10"], abs=1e-6)
    # a number of blocks without which a window keeps too few samples is not used
    few = np.concatenate([np.arange(0, 100, 2), [0, 1, 2, 3, 50]])           # window 1 has four samples in block 1
    sparse = TimeRangeEstimator(np.zeros((2, 55)), np.array([50, 5]), few, none, kt, 0.03, 2, np.zeros(2))
    whole = sparse.estimate(*sparse.masks(0, 100), np.zeros(2))
    assert whole.determined and sparse.jackknife(0, 100, 5, whole) == (
        None, None, "without block 1: lambda window 1 keeps 1 sample(s) there; 4 are needed for an estimate")
    assert free_energies_from(np.zeros((2, 5)), np.array([5, 0]), np.zeros(2)) is None       # a window without samples


def leg_with_tail(seed=0, n=600, k=3, tail=None, stride=4, identical=True):
    """Samples of a leg whose end states have known free energies, with tail energies recorded every
    ``stride`` iterations: ``tail(iterations, rng)`` gives them."""
    rng = np.random.default_rng(seed)
    if identical:                                   # the Hamiltonians of all windows are the same: dG = 0
        x = rng.normal(size=(n, k))
        u = np.repeat((0.5 * x ** 2)[:, :, None], k, axis=2)
        states = np.tile(np.arange(k), (n, 1))
    else:
        u, states, _ = harmonic_samples(n, np.linspace(1.0, 4.0, k), np.linspace(0.0, 0.6, k), rng)
    iterations = np.arange(0, n, stride)
    return {"u": u, "states": states, "volume": np.ones((n, k)), "lrc_iter": iterations,
            "lrc_du": tail(iterations, rng), "attrs": {}}


def test_the_standard_error_and_the_halves_are_those_of_the_value_with_its_tail_correction():
    """Review of PR 1: the free energy of a leg is MBAR minus the tail correction, but its standard error and
    the comparison of its halves were made from MBAR alone. Three windows with the same Hamiltonian (MBAR
    gives 0), the tail energy of the complex leg 0 in the first half and -8 kJ/mol in the second, that of the
    solvent leg -2 kJ/mol throughout: the binding free energy was -1.0464 +- 0.0000 kcal/mol, all checks passed.
    The corrected halves of the complex leg differ by 8 kJ/mol (3.23 kT), and the same ten blocks give a
    standard error of 0.183 kcal/mol once the correction is estimated in each."""
    from cufep.analysis.mbar import quality_checks

    kt, kcal = 8.31446261815324e-3 * 298.15, 4.184
    complex_leg = analyze_leg(leg_with_tail(tail=lambda it, rng: np.where(it < 300, 0.0, -8.0)), 298.15,
                              n_equilibration=0, tail="reweight")
    solvent_leg = analyze_leg(leg_with_tail(tail=lambda it, rng: np.full(it.size, -2.0)), 298.15,
                              n_equilibration=0, tail="reweight")
    assert complex_leg.delta_g_mbar == pytest.approx(0.0, abs=1e-9) and complex_leg.correction_samples == 150
    whole = -kt * np.log(0.5 + 0.5 * np.exp(8.0 / kt))
    assert complex_leg.correction == pytest.approx(whole) and complex_leg.delta_g == pytest.approx(-whole)
    dg_bind = (solvent_leg.delta_g - complex_leg.delta_g) / kcal
    assert dg_bind == pytest.approx(-1.0464, abs=5e-4)
    # ten blocks of 60 iterations, each holding 15 tail energies, by hand
    without = [-kt * np.log((60 + 75 * np.exp(8.0 / kt)) / 135)] * 5 + [-kt * np.log((75 + 60 * np.exp(8.0 / kt)) / 135)] * 5
    by_hand = np.sqrt(9 / 10 * np.sum((np.array(without) - np.mean(without)) ** 2))
    assert complex_leg.delta_g_error_blocks["10"] == pytest.approx(by_hand) and by_hand / kcal == pytest.approx(0.183, abs=5e-4)
    assert complex_leg.delta_g_error >= by_hand and complex_leg.delta_g_error_from == "time blocks"
    assert complex_leg.delta_g_mbar_error_blocks["10"] == pytest.approx(0.0, abs=1e-9)      # the MBAR part has none
    # the halves are of the corrected value: 0 and 8 kJ/mol
    assert complex_leg.first_half[0] == pytest.approx(0.0, abs=1e-9) and complex_leg.second_half[0] == pytest.approx(8.0)
    checks = quality_checks(complex_leg.to_dict())
    assert checks["halves"]["ok"] is False and checks["halves"]["difference_kT"] == pytest.approx(8.0 / kt)
    assert round(checks["halves"]["difference_kT"], 2) == 3.23
    # BAR and MBAR are compared without the correction, on the same samples: they agree
    assert checks["estimators"]["ok"] and checks["estimators"]["mbar"] == complex_leg.delta_g_mbar
    # the forward and reverse series and the consecutive blocks are of the corrected value too
    assert complex_leg.forward[-1][1] == complex_leg.delta_g and complex_leg.blocks[0][2] == pytest.approx(0.0, abs=1e-9)
    assert complex_leg.blocks[-1][2] == pytest.approx(8.0)
    # a constant correction has no spread: the solvent leg's error is the analytical one of zero
    assert solvent_leg.delta_g == pytest.approx(2.0) and solvent_leg.delta_g_error == pytest.approx(0.0, abs=1e-9)


def test_a_part_of_the_production_whose_windows_do_not_overlap_has_no_estimate(monkeypatch):
    """Review of PR 1: two windows, 600 iterations; the windows overlap in the first 120 iterations only, then
    their cross energies are above 1000 kT. The whole production overlaps by 0.10. Without samples that
    connect the windows the MBAR equations hold for any free energy, so the solver returned the value it was
    started from (-10, 2 or 10 kT alike); both halves were reported as the same 4.958 kJ/mol and all checks
    passed."""
    from cufep.analysis.mbar import TailSeries, TimeRangeEstimator, decorrelated, free_energies_from, quality_checks

    import cufep.analysis.mbar as mbar

    # every iteration counts as independent: the jump of the cross energies at iteration 120 would otherwise
    # make the statistical inefficiency so long that a half keeps too few samples, which is not the point here
    monkeypatch.setattr(mbar, "neighbour_inefficiencies", lambda u, states: np.ones((u.shape[2], 2)))
    rng = np.random.default_rng(3)
    n, centres = 600, np.array([0.0, 1.0])
    x = centres[None, :] + rng.normal(size=(n, 2))                    # replica r stays in window r
    u = 0.5 * (x[:, :, None] - centres[None, None, :]) ** 2
    u[120:, 0, 1] = u[120:, 1, 0] = 1500.0
    states = np.tile(np.arange(2), (n, 1))
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert res.min_overlap > 0.03
    # the trap itself: the second half alone, solved from three starts, returns each start
    g = np.ones(2)
    second_kn, second_n = decorrelated(u[300:], states[300:], g)
    for start in (-10.0, 2.0, 10.0):
        assert free_energies_from(second_kn, second_n, np.array([0.0, start]))[-1] == pytest.approx(start)
    from cufep.analysis.mbar import kept_sample_iterations
    estimator = TimeRangeEstimator(second_kn, second_n, kept_sample_iterations(300, g), TailSeries(
        "none", np.zeros(0), np.zeros(0), 298.15), 2.479, 0.03, int(np.ceil(g.max())), np.array([0.0, 2.0]))
    trapped = estimator.estimate(*estimator.masks(0, 300), np.array([0.0, 2.0]))
    assert not trapped.determined and trapped.value is None
    assert trapped.reason.startswith("lambda windows 0 and 1 overlap by 0.000 among these samples, below 0.03")
    # the halves are not compared, and say why
    assert np.isnan(res.first_half[0]) and np.isnan(res.second_half[0])
    assert "the second half (iterations 300-600 of the production): lambda windows 0 and 1 overlap by 0.000" in \
        res.convergence_note
    halves = quality_checks(res.to_dict())["halves"]
    assert halves["ok"] is None and halves["reason"] == res.convergence_note
    # a number of blocks without whose first block nothing connects the windows is not used
    assert res.delta_g_error_blocks["5"] is None
    assert res.delta_g_error_blocks_note["5"].startswith("without block 1: lambda windows 0 and 1 overlap by")
    # the reverse series has no estimate before it reaches back into the first 120 iterations
    reverse_fractions = [round(point[0], 2) for point in res.reverse]
    assert reverse_fractions == [0.9, 1.0] and len(res.forward) == 10
    assert sum(text.startswith("reverse point") for text in res.parts_not_determined) == 8


def test_a_tail_correction_without_observations_in_the_production_is_no_correction(tmp_path):
    """Review of PR 1: with analysis.lrc reweight and tail energies recorded only in the equilibration, the
    analysis reported a correction of 0.0 with an error of 0.0."""
    from click.testing import CliRunner

    from cufep.analysis.mbar import NoCorrection
    from cufep.cli import main
    from cufep.config.schema import AnalysisConfig
    from cufep.workflow import WorkflowError, leg_result_from_samples

    samples = leg_with_tail(tail=lambda it, rng: np.full(it.size, -2.0))
    samples["lrc_iter"], samples["lrc_du"] = np.array([0, 4, 8, 12]), np.full(4, -2.0)
    with pytest.raises(NoCorrection, match="0 tail energies were recorded in these iterations; 4 are needed"):
        analyze_leg(samples, 298.15, n_equilibration=20, tail="reweight")
    samples["lrc_iter"], samples["lrc_du"] = np.zeros(0, dtype=int), np.zeros(0)
    with pytest.raises(NoCorrection, match="the samples hold no tail energies"):
        analyze_leg(samples, 298.15, n_equilibration=20, tail="reweight")
    with pytest.raises(NoCorrection, match="no Lennard-Jones sums of the ligand"):
        analyze_leg(samples, 298.15, n_equilibration=20, tail="isotropic")
    assert analyze_leg(samples, 298.15, n_equilibration=20, tail="none").correction == 0.0

    # through a leg directory: the run's analysis and cufep analyze refuse alike
    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    storage = LegStorage(leg / "hrex", 3, 3)
    storage.set_meta(temperature_K=298.15, iteration_ps=2.5, backend="openmm", leg="solvent", lrc_s6=1.0, lrc_s12=1.0,
                     lrc_cutoff_nm=0.9)
    storage.set_plan(20, 600, {"version": 5, "lrc": "reweight"})
    u = leg_with_tail(tail=lambda it, rng: it * 0.0)["u"]
    for it in range(600):
        storage.append(u[it], np.arange(3), np.ones(3))
        if it < 20 and it % 4 == 0:
            storage.append_record(it, {"lrc_du": -2.0})
    storage.close()
    with pytest.raises(WorkflowError, match="the tail correction .reweight. cannot be estimated from the production: 0 tail"):
        leg_result_from_samples(leg, AnalysisConfig(lrc="reweight"), {"leg": "solvent"})
    shown = CliRunner().invoke(main, ["analyze", str(leg)])
    assert shown.exit_code == 1 and "cannot be estimated from the production" in shown.output


def test_every_part_is_the_reported_quantity_and_its_error_agrees_with_the_scatter():
    """Synthetic truth: six harmonic windows (dG = 0.5 ln(k_last / k_first) kT) and tail energies drawn
    from a normal distribution (correction mu - s^2 / 2kT). Over independent runs the reported value is
    unbiased and its standard error agrees with the scatter; every part equals MBAR of its own samples
    minus the correction of its own tail energies; the reported error is never below the analytical one."""
    from cufep.alchemy.lrc import tail_free_energy
    from cufep.analysis.mbar import decorrelated, solve_decorrelated, window_inefficiencies

    kt = 8.31446261815324e-3 * 298.15
    mu, sd = -3.0, 1.5
    truth = 0.5 * np.log(4.0 / 1.0) * kt - (mu - sd ** 2 / (2 * kt))
    legs = []
    for seed in range(32):
        samples = leg_with_tail(seed, n=600, k=6, identical=False, tail=lambda it, rng: rng.normal(mu, sd, it.size))
        legs.append((samples, analyze_leg(samples, 298.15, n_equilibration=0, tail="reweight")))
    values = np.array([res.delta_g for _, res in legs])
    scatter = values.std(ddof=1)
    assert abs(values.mean() - truth) < 3 * scatter / np.sqrt(len(values))
    assert 0.8 * scatter < np.sqrt(np.mean([res.delta_g_error ** 2 for _, res in legs])) < 1.4 * scatter
    for samples, res in legs:
        analytic = np.hypot(res.delta_g_mbar_error, res.correction_error)
        assert res.delta_g_error >= analytic - 1e-12 and res.delta_g == pytest.approx(res.delta_g_mbar - res.correction)
    # the halves by hand: MBAR of the half's own samples minus the correction of the half's own tail energies
    samples, res = legs[0]
    g = np.array(res.window_statistical_inefficiency)
    for (a, b), half in (((0, 300), res.first_half), ((300, 600), res.second_half)):
        f, _df = solve_decorrelated(*decorrelated(samples["u"], samples["states"], g, a, b))
        inside = (samples["lrc_iter"] >= a) & (samples["lrc_iter"] < b)
        assert half[0] == pytest.approx(f[-1] * kt - tail_free_energy(samples["lrc_du"][inside], 298.15)[0], abs=1e-6)
    # so is a point of the forward series
    m = 300
    f, _df = solve_decorrelated(*decorrelated(samples["u"], samples["states"], g, 0, m))
    point = [p for p in res.forward if p[0] == 0.5][0]
    inside = samples["lrc_iter"] < m
    assert point[1] == pytest.approx(f[-1] * kt - tail_free_energy(samples["lrc_du"][inside], 298.15)[0], abs=1e-6)


def test_a_leg_too_short_for_time_blocks_reports_the_error_of_mbar(monkeypatch):
    """The standard error is never left without a value: where no number of time blocks can be used it is
    the one MBAR gives, and the leg says so."""
    import cufep.analysis.mbar as mbar
    from cufep.analysis.mbar import quality_checks
    from cufep.config.schema import AnalysisConfig
    from cufep.workflow import quality_warnings

    k_spring, x0 = np.array([1.0, 1.3]), np.array([0.0, 0.2])
    u, states, _ = harmonic_samples(4, k_spring, x0, np.random.default_rng(5))
    four = analyze_leg({"u": u, "states": states}, 298.15)
    assert four.delta_g_error == four.delta_g_mbar_error and np.isfinite(four.delta_g_error)
    assert four.delta_g_error_from == "analytical"
    assert four.delta_g_error_n_blocks is None and four.delta_g_error_blocks == {"40": None, "20": None, "10": None, "5": None}
    assert four.delta_g_error_note.startswith("no number of time blocks could be used: 40 blocks: a block of 0 "
                                              "iteration(s) is shorter than the interval of 1 iterations")
    assert "; 5 blocks: a block of 0 iteration(s)" in four.delta_g_error_note
    stored = json.loads(json.dumps(four.to_dict()))
    samples = quality_checks(stored)["samples"]
    assert samples["error_ratio"] is None and samples["independent"] == 4.0 and samples["ok"] is False
    notes = quality_warnings({"analysis": stored, "leg": "solvent"}, AnalysisConfig())
    assert sum(note.startswith("the standard error of this leg is the analytical one of MBAR and the tail correction, "
                               "which takes the samples as independent of each other (no number of time blocks could "
                               "be used: 40 blocks: ") for note in notes) == 1
    assert not any("standard error from time blocks is" in note for note in notes)

    # twelve iterations of which every third is kept: a block of five is shorter than that interval
    u, states, _ = harmonic_samples(12, k_spring, x0, np.random.default_rng(6))
    monkeypatch.setattr(mbar, "neighbour_inefficiencies", lambda u, states: np.full((u.shape[2], 2), 2.125))
    twelve = analyze_leg({"u": u, "states": states}, 298.15)
    assert twelve.window_kept_samples == [4, 4] and twelve.delta_g_error == twelve.delta_g_mbar_error
    assert twelve.delta_g_error_n_blocks is None and twelve.halves_error_blocks == [None, None]
    assert "5 blocks: a block of 2 iteration(s) is shorter than the interval of 3 iterations at which the slowest " \
           "lambda window keeps its samples" in twelve.delta_g_error_note
    # with every iteration kept, five blocks of two or three iterations can be used
    monkeypatch.setattr(mbar, "neighbour_inefficiencies", lambda u, states: np.ones((u.shape[2], 2)))
    whole = analyze_leg({"u": u, "states": states}, 298.15)
    assert whole.delta_g_error_note == "" and whole.delta_g_error_blocks["40"] is None
    usable = [value for value in whole.delta_g_error_blocks.values() if value is not None]
    assert usable and whole.delta_g_error == max(usable + [whole.delta_g_mbar_error])
    # the numbers of blocks are settings of the analysis (analysis.error_blocks, part_error_blocks)
    three = analyze_leg({"u": u, "states": states}, 298.15, error_blocks=[6, 3, 3], part_error_blocks=3)
    assert list(three.delta_g_error_blocks) == ["6", "3"] and three.delta_g == whole.delta_g
    # a half holds six iterations: three blocks of two leave four samples, two blocks of three do not
    assert three.halves_error_blocks in ([3, 3], [3, None], [None, 3], [None, None])
    assert three.first_half[1] >= 0 and three.second_half[1] >= 0
    two = analyze_leg({"u": u, "states": states}, 298.15, part_error_blocks=2)
    assert two.halves_error_blocks == [None, None] and two.first_half[1] > 0       # the analytical error of the half
    assert three.delta_g_error == max([v for v in three.delta_g_error_blocks.values() if v is not None]
                                      + [three.delta_g_mbar_error])


def test_the_samples_check_counts_what_mbar_gets(monkeypatch):
    """Review of PR 1: 65 iterations with statistical inefficiencies 1.209 and 1.038 give MBAR 33 samples
    per window (every second iteration), not the 53.8 the check counted."""
    import cufep.analysis.mbar as mbar
    from cufep.analysis.mbar import quality_checks

    u, states, _ = harmonic_samples(65, np.array([1.0, 1.3]), np.array([0.0, 0.2]), np.random.default_rng(8))
    monkeypatch.setattr(mbar, "neighbour_inefficiencies",
                        lambda u, states: np.array([[np.nan, 1.209], [1.038, np.nan]]))
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert res.window_kept_samples == [33, 33]
    assert res.window_effective_samples == pytest.approx([65 / 1.209, 65 / 1.038])
    samples = quality_checks(res.to_dict())["samples"]
    assert samples["ok"] is False and samples["kept"] == 33 and samples["window"] == 0
    assert samples["fewest"] == pytest.approx(53.76, abs=0.01)                # the estimate alone would pass
    assert quality_checks(res.to_dict(), min_effective_samples=33.0)["samples"]["ok"] is True


def test_the_overlap_is_also_taken_from_the_samples_of_the_estimate():
    """Review of PR 1: the overlap that was checked came from other samples than the free energy.

    Two windows 4.25 standard deviations apart, each correlated by 0.7 from one iteration to the next. A
    noise of 20 kT that both energies of a configuration share hides the correlation from the summed
    potential, so the matrix of all iterations uses every second one and the estimate every seventh.
    """
    from cufep.analysis.mbar import decorrelated, overlap_with_equal_weights, quality_checks, solve_decorrelated

    rng = np.random.default_rng(298)
    n = 700
    x = np.stack([ar1(n, 0.7, rng), 4.25 + ar1(n, 0.7, rng)], axis=1)
    noise = rng.normal(scale=20.0, size=(n, 2))
    u = 0.5 * (x[:, :, None] - np.array([0.0, 4.25])[None, None]) ** 2 + noise[:, :, None]
    res = analyze_leg({"u": u, "states": np.tile(np.arange(2), (n, 1))}, 298.15)
    assert res.window_kept_samples == [100, 100]
    assert res.min_overlap == pytest.approx(0.0337, abs=5e-4)                # passes the level of 0.03
    assert res.min_overlap_used == pytest.approx(0.0247, abs=5e-4)           # the samples of the estimate do not
    # with equal numbers of samples the normalisation of equal numbers is pymbar's own matrix
    assert np.allclose(res.overlap_used_matrix, res.overlap_used_weighted_by_number, atol=2e-4)
    overlap = quality_checks(res.to_dict())["overlap"]
    assert overlap["all_ok"] is True and overlap["used"]["ok"] is False and overlap["ok"] is False
    assert overlap["below"] == ["the samples of the estimate"]
    assert overlap["used"]["windows"] == [0, 1] and overlap["used"]["samples"] == [100, 100]

    # different numbers of samples: pymbar's matrix follows the numbers, the normalisation of equal
    # numbers stays where the overlap of the windows is
    u3, s3, _ = harmonic_samples(2000, np.array([1.0, 1.5, 2.2]), np.array([0.0, 0.5, 1.0]),
                                 np.random.default_rng(5), exchange=False)
    equal_kn, equal_n = decorrelated(u3, s3, np.ones(3))
    f_equal, _df, o_equal = solve_decorrelated(equal_kn, equal_n, overlap=True)
    assert np.allclose(overlap_with_equal_weights(equal_kn, equal_n, f_equal), o_equal, atol=1e-6)
    thin_kn, thin_n = decorrelated(u3, s3, np.array([1.0, 100.0, 1.0]))
    assert thin_n.tolist() == [2000, 20, 2000]
    f_thin, _df, o_thin = solve_decorrelated(thin_kn, thin_n, overlap=True)
    normalised = overlap_with_equal_weights(thin_kn, thin_n, f_thin)
    assert np.allclose(normalised, normalised.T) and np.allclose(normalised.sum(axis=1), 1.0, atol=0.1)
    assert normalised[0, 1] == pytest.approx(o_equal[0, 1], rel=0.25)
    assert o_thin[0, 1] < 0.05 * o_equal[0, 1] and o_thin[1, 0] > o_equal[1, 0]     # 20 samples next to 2000


def test_results_are_json_that_every_parser_reads():
    """Review of PR 1: a production too short for its halves wrote NaN, and identical halves Infinity."""
    import json

    from cufep import jsonio
    from cufep.analysis.mbar import quality_checks

    u, states, _ = harmonic_samples(4, np.array([1.0, 1.3]), np.array([0.0, 0.2]), np.random.default_rng(2))
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert np.isnan(res.first_half[0]) and np.isnan(res.forward_reverse_gap)        # in memory
    stored = res.to_dict()
    assert stored["first_half"] == [None, None] and stored["forward_reverse_gap"] is None
    assert "4 are needed" in stored["convergence_note"]

    def not_json(token):
        raise ValueError(f"{token} is not JSON")

    back = json.loads(jsonio.dumps({"analysis": stored, "quality": quality_checks(stored)}), parse_constant=not_json)
    assert back["quality"]["halves"]["ok"] is None and back["quality"]["halves"]["reason"] == stored["convergence_note"]
    assert jsonio.clean({"a": float("inf"), "b": np.float64("nan"), "c": np.int64(3), "d": (1.0, np.array([2.0, np.nan])),
                         "e": np.bool_(True)}) == {"a": None, "b": None, "c": 3, "d": [1.0, [2.0, None]], "e": True}

    base = {"delta_g": 10.0, "delta_g_error": 0.0, "min_overlap": 0.2, "overlap_neighbour": [0.2],
            "window_effective_samples": [100.0, 100.0], "window_statistical_inefficiency": [1.0, 1.0]}
    same = quality_checks(dict(base, first_half=[10.0, 0.0], second_half=[10.0, 0.0]))["halves"]
    assert same["ok"] is True and same["difference_sigma"] == 0.0                   # was Infinity
    apart = quality_checks(dict(base, first_half=[10.5, 0.0], second_half=[10.0, 0.0]))["halves"]
    assert apart["difference_sigma"] is None and "is zero" in apart["difference_sigma_note"]
    # values that a result file holds as null
    missing = quality_checks(dict(base, first_half=[None, None], second_half=[10.0, 0.1], bar_delta_g=None,
                                  bar_note="no estimate"))
    assert missing["halves"]["ok"] is False and "difference" not in missing["halves"]
    assert missing["estimators"]["ok"] is None and missing["estimators"]["reason"] == "no estimate"
    json.loads(jsonio.dumps(missing), parse_constant=not_json)
    no_energy = quality_checks(dict(base, delta_g=None, delta_g_error=None, min_overlap=None, overlap_neighbour=[None]))
    assert no_energy["overlap"]["ok"] is False and no_energy["uncertainty"] is None


def test_stage_boundaries():
    s = fixed_schedule(["restrain", "discharge", "vanish"], {"restrain": 2, "discharge": 3, "vanish": 4})
    b = stage_boundaries([x.to_dict() for x in s.states])
    assert b == {"restrain": (0, 2), "discharge": (2, 5), "vanish": (5, 9)}
    s = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4})
    assert stage_boundaries([x.to_dict() for x in s.states]) == {"discharge": (0, 3), "vanish": (3, 7)}


def test_storage_append_truncate_and_checkpoint(tmp_path):
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    for it in range(5):
        st.append(np.full((3, 3), float(it)), np.array([0, 1, 2]), np.ones(3))
        if it % 2 == 0:
            st.append_record(it, {"lrc_du": -1.0 * it, "frame": np.zeros((2, 3), np.float32),
                                  "box": np.eye(3, dtype=np.float32)})
    st.write_checkpoint(np.zeros((3, 4, 3)), np.zeros((3, 4, 3)), np.zeros((3, 3, 3)), [2, 0, 1], 3)
    st.close()
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    ck = st.read_checkpoint()
    assert ck["iteration"] == 3 and ck["state_of_replica"].tolist() == [2, 0, 1]
    st.truncate(ck["iteration"])
    assert st.n_iterations == 3
    st.close()
    data = read_samples(tmp_path / "leg")
    assert data["u"].shape == (3, 3, 3) and data["lrc_iter"].tolist() == [0, 2] and data["frames"].shape[0] == 2
    with pytest.raises(ValueError):
        LegStorage(tmp_path / "leg", 4, 4)


def _rows(n0, n, k=3):
    """``n`` distinct rows from ``n0`` on: (u, states, volume) with values that name their iteration."""
    return [(np.full((k, k), float(it)), np.roll(np.arange(k), it), np.full(k, 10.0 + it)) for it in range(n0, n0 + n)]


def _fill(storage, rows, n0):
    for i, row in enumerate(rows):
        storage.append(*row)
        it = n0 + i
        if it % 2 == 0:
            storage.append_record(it, {"lrc_du": -float(it), "frame": np.full((2, 3), it, np.float32),
                                       "box": np.eye(3, dtype=np.float32) * it})


def _assert_exactly(directory, n):
    """The file holds iterations 0 .. n-1 once each, with their records, and nothing else."""
    data = read_samples(directory)
    assert data["u"].shape[0] == data["states"].shape[0] == data["volume"].shape[0] == n
    assert [float(x) for x in data["u"][:, 0, 0]] == [float(it) for it in range(n)]
    assert [list(row) for row in data["states"]] == [list(np.roll(np.arange(3), it)) for it in range(n)]
    assert data["volume"][:, 0].tolist() == [10.0 + it for it in range(n)]
    assert data["lrc_iter"].tolist() == list(range(0, n, 2)) and data["lrc_du"].tolist() == [-float(it) for it in range(0, n, 2)]
    assert data["frame_iter"].tolist() == list(range(0, n, 2)) and data["frames"][:, 0, 0].tolist() == list(range(0, n, 2))
    assert data["frame_box"].shape[0] == len(data["frame_iter"])


WRITE_STEPS = ("/u", "/states", "/volume", "/lrc_iter", "/lrc_du", "/frame_iter", "/frames", "/frame_box", "publish")


@pytest.mark.parametrize("step", WRITE_STEPS)
def test_a_write_that_fails_at_any_step_stores_nothing_twice_and_nothing_unwritten(tmp_path, monkeypatch, step):
    """Review of PR 1: 6 iterations stored, 4 more in memory; the write of `states` failed once after `u`
    had been extended. The retry at close appended the 4 rows after the 10 of `u`: 14 rows, rows 7-10 twice
    in the energies and four rows of zeros in the states. Every step of a write fails once here."""
    import cufep.sampling.storage as storage_module

    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    _fill(st, _rows(0, 6), 0)
    assert st.write_checkpoint(np.zeros((3, 4, 3)), np.zeros((3, 4, 3)), np.zeros((3, 3, 3)), [0, 1, 2], 6)
    _fill(st, _rows(6, 4), 6)
    real_write, real_publish, failed = storage_module._write_at, storage_module._publish, []

    def write_at(dataset, offset, values):
        real_write(dataset, offset, values)                       # the data reaches the file, then it fails
        if dataset.name == step and not failed and len(values):
            failed.append(step)
            raise OSError(f"injected failure after writing {step}")

    def publish(h5, *counts):
        if step == "publish" and not failed:
            failed.append(step)
            raise OSError("injected failure while publishing")
        real_publish(h5, *counts)

    monkeypatch.setattr(storage_module, "_write_at", write_at)
    monkeypatch.setattr(storage_module, "_publish", publish)
    with pytest.raises(OSError, match="injected"):
        st.flush()
    assert failed == [step] and st.n_iterations == 10 and st.n_stored == 6
    _assert_exactly(tmp_path / "leg", 6)                          # every reader sees the state before the write
    assert LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2).n_iterations == 6
    st.close()                                                    # the retry, as when the sampler is closed
    _assert_exactly(tmp_path / "leg", 10)
    again = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)   # opened again, it goes on from there
    assert again.n_iterations == 10
    _fill(again, _rows(10, 2), 10)
    again.close()
    _assert_exactly(tmp_path / "leg", 12)


@pytest.mark.parametrize("step", ("publish", "resize"))
def test_a_truncation_that_fails_leaves_one_state_of_the_samples(tmp_path, monkeypatch, step):
    import h5py

    import cufep.sampling.storage as storage_module

    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    _fill(st, _rows(0, 10), 0)
    st.close()
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    real_publish = storage_module._publish
    if step == "publish":
        def publish(h5, *counts):
            raise OSError("injected failure while publishing")
        monkeypatch.setattr(storage_module, "_publish", publish)
    else:
        real_resize = h5py.Dataset.resize

        def resize(self, *args, **kwargs):
            if self.name == "/volume":
                raise OSError("injected failure while shortening")
            return real_resize(self, *args, **kwargs)
        monkeypatch.setattr(h5py.Dataset, "resize", resize)
    with pytest.raises(OSError, match="injected"):
        st.truncate(7)
    monkeypatch.setattr(storage_module, "_publish", real_publish)
    if step == "publish":
        _assert_exactly(tmp_path / "leg", 10)                     # nothing changed
        assert st.n_stored == 10
        st.truncate(7)
    else:
        _assert_exactly(tmp_path / "leg", 7)                      # the smaller count was published first
        assert st.n_stored == 7
        monkeypatch.undo()
    _fill(st, _rows(7, 3), 7)
    st.close()
    _assert_exactly(tmp_path / "leg", 10)


def test_samples_of_earlier_versions_are_read_and_continued_from_what_all_their_datasets_hold(tmp_path):
    """A file without the stored count (earlier versions) holds the iterations that all three row
    datasets hold; it is continued from there, and from then on it has the count."""
    import h5py

    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    _fill(st, _rows(0, 8), 0)
    st.close()
    with h5py.File(tmp_path / "leg" / "samples.h5", "a") as h5:   # as an earlier version killed while writing left it
        del h5.attrs["committed"]
        h5["u"].resize(10, axis=0)
        h5["lrc_iter"].resize(5, axis=0)
        h5["lrc_iter"][4] = 8
        h5["lrc_du"].resize(5, axis=0)
    _assert_exactly(tmp_path / "leg", 8)
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    assert st.n_iterations == 8
    _fill(st, _rows(8, 4), 8)
    st.close()
    _assert_exactly(tmp_path / "leg", 12)
    from cufep.sampling.storage import stored_attributes, stored_conditions
    assert stored_attributes(tmp_path / "leg")[0] == 12 and stored_conditions(tmp_path / "leg")[0] == 12
    assert read_samples(tmp_path / "leg")["attrs"]["committed"] == {"iterations": 12, "tail_records": 6, "frame_records": 6}
    # a count that the datasets do not hold is a damaged file, not a smaller sample
    with h5py.File(tmp_path / "leg" / "samples.h5", "a") as h5:
        h5["states"].resize(11, axis=0)
    from cufep.sampling.storage import StorageError
    with pytest.raises(StorageError, match="states are shorter than the 12 iterations"):
        read_samples(tmp_path / "leg")


def test_the_storage_says_when_it_could_not_store_and_restore_refuses_a_file_shorter_than_its_checkpoint(
        tmp_path, tiny_leg, monkeypatch):
    import cufep.sampling.storage as storage_module
    from cufep.sampling.hrex import SamplerError
    from cufep.sampling.storage import StorageError

    st = LegStorage(tmp_path / "leg", 3, 3)
    _fill(st, _rows(0, 2), 0)

    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(storage_module, "_publish", broken)
    with pytest.raises(StorageError, match="the 2 iteration.s. after iteration 0 could not be stored .* disk full"):
        st.close()
    monkeypatch.undo()
    st.close()                                                     # the rows stayed in memory: stored now
    assert read_samples(tmp_path / "leg")["u"].shape[0] == 2

    spec, pos, box = tiny_leg
    log = []
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=log.append) as s:
        s.initialize(pos, box)
        s.run(4, record_stride=2, checkpoint_stride=2)
    # the sampler: an error of the storage at close is raised, unless another error is already on its way
    s = ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=log.append)
    s.storage.append(np.zeros((4, 4)), np.arange(4), np.ones(4))
    monkeypatch.setattr(storage_module, "_publish", broken)
    with pytest.raises(StorageError, match="disk full"):
        s.close()
    s = ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=log.append)
    s.storage.append(np.zeros((4, 4)), np.arange(4), np.ones(4))
    with pytest.raises(ValueError, match="the error of the run"):
        with s:
            raise ValueError("the error of the run")
    assert any("could not be stored" in line for line in log)
    monkeypatch.undo()
    assert not issubclass(StorageError, SamplerError)
    # a file that holds fewer iterations than its checkpoint names is not continued
    import h5py
    with h5py.File(tmp_path / "hrex" / "samples.h5", "a") as h5:
        frames = int(np.sum(h5["frame_iter"][:] < 2))
        h5.attrs["committed"] = json.dumps({"iterations": 2, "tail_records": 0, "frame_records": frames})
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=log.append) as s:
        with pytest.raises(StorageError, match="holds 2 iterations, fewer than the 4 that its checkpoint names"):
            s.restore()


def test_a_final_checkpoint_that_cannot_be_stored_is_no_failure_of_the_backend(tmp_path, tiny_leg, monkeypatch):
    """Review of PR 1: the last checkpoint met a reader that held samples.h5 for the whole wait; the error was
    a SamplerError, which run_leg took for a failure of cuPotential: it set the samples aside and repeated
    the whole leg on stock OpenMM."""
    import openmm as mm

    import cufep.sampling.storage as storage_module
    from cufep.sampling.hrex import SamplerError, WorkerError
    from cufep.sampling.storage import StorageBusyError
    from cufep.workflow import backend_fallback_refusal

    spec, pos, box = tiny_leg
    monkeypatch.setattr(storage_module, "WRITE_WAIT_S", 0.3)
    monkeypatch.setattr(storage_module, "CHECKPOINT_WAIT_S", 0.1)
    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(2, record_stride=2, checkpoint_stride=2)
        reader = hold_samples(tmp_path / "hrex" / "samples.h5", "r")
        try:
            with pytest.raises(StorageBusyError, match="the samples up to iteration 5 could not be stored: another "
                                                       "process held .* The checkpoint at iteration 2 and the samples "
                                                       "up to it are kept; running the leg again resumes it"):
                s.run(3, record_stride=2, checkpoint_stride=100)
        finally:
            release(reader)
    assert s.storage.read_checkpoint()["iteration"] == 2
    # what run_leg makes of each kind of error
    busy = StorageBusyError("held")
    assert backend_fallback_refusal(busy, "cupotential", "openmm", False) == "not a failure of the backend"
    assert backend_fallback_refusal(SamplerError("the checkpoint does not match"), "cupotential", "openmm", False) \
        == "not a failure of the backend"
    assert backend_fallback_refusal(WorkerError("worker 3 died"), "cupotential", "openmm", False) is None
    assert backend_fallback_refusal(mm.OpenMMException("CUDA error"), "cupotential", "openmm", False) is None
    assert backend_fallback_refusal(WorkerError("died"), "cupotential", "openmm", True) == "the leg had samples from an earlier run"
    assert backend_fallback_refusal(WorkerError("died"), "openmm", "openmm", False) == "the leg runs on stock OpenMM already"
    assert backend_fallback_refusal(WorkerError("died"), "cupotential", "none", False) == "backend.fallback is none"
    assert backend_fallback_refusal(OSError("disk"), "cupotential", "openmm", False) == "not a failure of the backend"


def test_the_samples_of_a_failed_backend_run_are_kept_and_archived(tmp_path):
    """The samples of every cuPotential run that failed are kept (earlier ones were deleted), and they go
    with the leg into previous_<n>/."""
    from cufep.workflow import archive_leg, set_aside_failed_backend

    leg = tmp_path / "leg"
    for n in range(3):
        (leg / "hrex").mkdir(parents=True)
        (leg / "hrex" / "samples.h5").write_text(f"run {n}")
        (leg / "schedule.yaml").write_text(f"schedule {n}")
        kept = set_aside_failed_backend(leg)
        assert kept.name == ("hrex_failed_cupotential" if n == 0 else f"hrex_failed_cupotential_{n + 1}")
        assert (kept / "samples.h5").read_text() == f"run {n}" and (kept / "schedule.yaml").read_text() == f"schedule {n}"
        assert not (leg / "hrex").exists() and not (leg / "schedule.yaml").exists()
    previous = archive_leg(leg)
    assert sorted(p.name for p in previous.iterdir()) == ["hrex_failed_cupotential", "hrex_failed_cupotential_2",
                                                          "hrex_failed_cupotential_3"]


HOLD_SAMPLES = ("import sys, h5py\n"
                "handle = h5py.File(sys.argv[1], sys.argv[2])\n"
                "print('open', flush=True)\n"
                "sys.stdin.readline()\n")


def hold_samples(path, mode):
    """Another process that has a samples file open in ``mode`` until its standard input is closed."""
    import subprocess
    import sys

    holder = subprocess.Popen([sys.executable, "-c", HOLD_SAMPLES, str(path), mode], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "open"
    return holder


def release(holder):
    holder.stdin.close()
    holder.wait(timeout=60)


def test_another_process_reads_the_samples_of_a_leg_that_is_being_sampled(tmp_path):
    """Review of PR 1: `cufep analyze` of a leg whose sampler was alive ended with "unable to lock file":
    the sampler held samples.h5 open for writing from its first to its last iteration."""
    import subprocess
    import sys
    import time

    from cufep.sampling.storage import open_samples

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    u, states, _ = harmonic_samples(107, np.array([1.0, 1.6, 2.6, 4.2]), np.linspace(0.0, 0.6, 4), np.random.default_rng(12))
    storage = LegStorage(leg / "hrex", 4, 4, n_frame_atoms=2)
    storage.set_meta(temperature_K=310.0, iteration_ps=2.5, backend="openmm", leg="solvent")
    storage.set_plan(4, 400)
    state = (np.zeros((4, 5, 3)), np.zeros((4, 5, 3)), np.zeros((4, 3, 3)), np.arange(4))
    for it in range(107):
        storage.append(u[it], states[it], np.ones(4))
        if it % 4 == 0:
            storage.append_record(it, {"lrc_du": -0.1 * it, "frame": np.full((2, 3), it, np.float32),
                                       "box": np.eye(3, dtype=np.float32)})
        if it == 99:
            assert storage.write_checkpoint(*state, 100) is True
    # the sampler is alive: 100 iterations are in the file, 7 more in its memory
    assert storage.n_iterations == 107
    read = "import sys; from cufep.sampling.storage import read_samples; d = read_samples(sys.argv[1]); " \
           "print(d['u'].shape[0], d['states'].shape[0], d['lrc_iter'].tolist()[-1], d['frames'].shape[0])"
    other = subprocess.run([sys.executable, "-c", read, str(leg / "hrex")], capture_output=True, text=True, timeout=120)
    assert other.returncode == 0 and other.stdout.split() == ["100", "100", "96", "25"], other.stderr
    analyze = subprocess.run([sys.executable, "-m", "cufep.cli", "analyze", str(leg)], capture_output=True, text=True,
                             timeout=300)
    assert analyze.returncode == 0, analyze.stderr
    shown = json.loads(analyze.stdout)
    assert shown["n_iterations_stored"] == 100 and shown["n_iterations_used"] == 96 and shown["n_planned_iterations"] == 400
    assert read_samples(leg / "hrex")["u"].shape[0] == 100                  # and this process too
    assert np.array_equal(read_samples(leg / "hrex")["u"], u[:100])

    # another process is reading the file just when a checkpoint is due: the checkpoint is postponed, the
    # samples stay in memory, and the next one stores them all
    reader = hold_samples(leg / "hrex" / "samples.h5", "r")
    t0 = time.monotonic()
    assert storage.write_checkpoint(*state, 107, wait_s=0.3) is False and 0.25 < time.monotonic() - t0 < 30.0
    assert storage.read_checkpoint()["iteration"] == 100 and storage.n_iterations == 107
    assert storage.flush(wait_s=0.0) is False
    release(reader)
    assert storage.write_checkpoint(*state, 107, wait_s=30.0) is True
    assert storage.read_checkpoint()["iteration"] == 107 and storage.n_iterations == 107
    data = read_samples(leg / "hrex")
    assert np.array_equal(data["u"], u) and np.array_equal(data["states"], states) and data["lrc_iter"].tolist()[-1] == 104
    assert data["frames"].shape == (27, 2, 3) and data["frame_iter"].tolist() == list(range(0, 107, 4))
    assert storage.flush() is True                                          # nothing is left in memory

    # a reader waits for the sampler that is storing a checkpoint, as long as it is told to
    writer = hold_samples(leg / "hrex" / "samples.h5", "a")
    with pytest.raises(OSError, match="unable to lock file"):
        open_samples(leg / "hrex" / "samples.h5", "r", wait_s=0.3)
    import threading
    threading.Timer(1.0, release, args=(writer,)).start()
    t0 = time.monotonic()
    assert read_samples(leg / "hrex")["u"].shape[0] == 107 and time.monotonic() - t0 > 0.5

    # rows that are dropped before they were stored, and rows of a file whose datasets differ in length
    storage.append(u[0], states[0], np.ones(4))
    storage.append_record(107, {"lrc_du": 1.0})
    storage.truncate(107)
    assert storage.n_iterations == 107
    storage.close()
    assert read_samples(leg / "hrex")["lrc_iter"].tolist()[-1] == 104
    import h5py
    with h5py.File(leg / "hrex" / "samples.h5", "a") as h5:                # as if killed while storing
        h5["u"].resize(110, axis=0)
        h5["lrc_iter"].resize(28, axis=0)
        h5["lrc_iter"][27] = 108
        h5["lrc_du"].resize(28, axis=0)
    data = read_samples(leg / "hrex")
    assert data["u"].shape[0] == data["states"].shape[0] == data["volume"].shape[0] == 107
    assert len(data["lrc_iter"]) == len(data["lrc_du"]) == 27


def test_a_checkpoint_that_meets_a_reader_is_postponed_and_the_run_goes_on(tmp_path, tiny_leg, monkeypatch):
    """The sampler of a leg does not stop, and loses nothing, when its samples file is held by a reader at
    the moment of a checkpoint; the end of the run waits for the file."""
    import cufep.sampling.storage as storage_module

    spec, pos, box = tiny_leg
    monkeypatch.setattr(storage_module, "CHECKPOINT_WAIT_S", 0.2)
    held, log = [], []

    def note(message):
        log.append(message)
        if "postponed" in message and held:
            release(held.pop())                   # the reader goes away after the first postponed checkpoint

    with ReplicaExchangeSampler(spec, tmp_path / "hrex", devices=[None], inline=True, log=note) as s:
        s.initialize(pos, box)
        held.append(hold_samples(tmp_path / "hrex" / "samples.h5", "r"))
        s.run(6, record_stride=2, checkpoint_stride=2)
        assert s.iteration == 6 and s.storage.n_iterations == 6
        assert s.storage.read_checkpoint()["iteration"] == 6
    postponed = [message for message in log if "postponed" in message]
    assert len(postponed) == 1 and postponed[0] == (
        "[tiny] the checkpoint at iteration 2 is postponed to the next one: another process holds samples.h5 open; "
        "the samples stay in memory")
    data = read_samples(tmp_path / "hrex")
    assert data["u"].shape[0] == 6 and data["frame_iter"].tolist() == [0, 2, 4]


@pytest.fixture(scope="module")
def tiny_leg(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 2})
    spec = LegSpec(name="tiny", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=1.0,
                   barostat="isotropic", timestep_fs=2.0, steps_per_iteration=10, seed=5, frame_atoms=list(ligand))
    return spec, pos, box


def test_inline_replica_exchange_runs_resumes_and_analyses(tmp_path, tiny_leg):
    spec, pos, box = tiny_leg
    leg = tmp_path / "hrex"
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(6, record_stride=2, checkpoint_stride=3)
        assert s.iteration == 6 and sorted(s.state_of_replica.tolist()) == [0, 1, 2, 3]
        final = s.state_of_replica.copy()
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        assert s.restore() and s.iteration == 6
        assert s.state_of_replica.tolist() == final.tolist()
        s.run(6, record_stride=2, checkpoint_stride=3)
        s.storage.set_meta(n_equilibration_iterations=2)
    data = read_samples(leg)
    assert data["u"].shape == (12, 4, 4) and np.all(np.isfinite(data["u"][np.arange(12)[:, None], np.arange(4), data["states"]]))
    assert all(sorted(row.tolist()) == [0, 1, 2, 3] for row in data["states"])
    assert data["frames"].shape[1:] == (3, 3) and len(data["frame_iter"]) >= 4
    res = analyze_leg(data, 300.0)
    assert res.n_states == 4 and res.n_iterations_used == 10
    assert np.isfinite(res.delta_g) and set(res.stage_delta_g) == {"discharge", "vanish"}
    # decoupling a water molecule from water costs free energy (hydration free energy ~ -6 kcal/mol).
    # Ten iterations are far too few for the analysis of a leg, which thins every window by its own
    # correlation and may keep one or two of them (its result then depends on the machine); the
    # samples themselves say it when all of them are taken
    assert solve_mbar(data["u"][2:], data["states"][2:], overlap=False)[0][-1] > 0
    assert min(res.window_kept_samples) >= 1 and max(res.window_kept_samples) <= 10


def test_a_resumed_leg_does_not_repeat_its_thermal_noise(tmp_path, monkeypatch):
    """Review of PR 1: a leg resumed from its checkpoint drew the Langevin noise of its first steps again.

    The seed of a worker was made of the seed of the leg and the number of the worker, and a resumed leg
    numbers its workers from 0 again. Particles without forces show the noise directly: with the
    LangevinMiddle integrator v(t + dt) = exp(-gamma dt) v(t) + noise.
    """
    import shutil

    import cufep.sampling.hrex as hrex
    from cufep.sampling.hrex import SamplerError
    from cufep.sampling.states import AlchemicalState
    from cufep.sampling.worker import BAROSTAT_STREAM, stream_seed

    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    spec = LegSpec(name="free", system_xml=mm.XmlSerializer.serialize(system),
                   states=[AlchemicalState().to_dict(), AlchemicalState(lambda_sterics=0.5).to_dict()], n_atoms=4,
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=298.15, pressure_bar=None,
                   barostat="none", friction_per_ps=1.0, timestep_fs=2.0, steps_per_iteration=1, seed=11)
    decay = np.exp(-1.0 * 0.002)

    def sampler(directory):
        return ReplicaExchangeSampler(spec, directory, devices=[None], inline=True, exchange=False, log=lambda m: None)

    def noise_of_steps(s, n):
        v = [s.arrays.vel[s.cur].copy()]
        for _ in range(n):
            s.run(1, record_stride=0, checkpoint_stride=1000)       # one step, and a checkpoint at its end
            v.append(s.arrays.vel[s.cur].copy())
        v = np.array(v)
        return v[1:] - decay * v[:-1]

    def fresh(directory):
        with sampler(directory) as s:
            s.initialize(np.zeros((4, 3)), 3.0 * np.eye(3), velocities=np.zeros((4, 3)))
            assert s.start_number == 0
            noise = noise_of_steps(s, 10)
            assert s.storage.read_checkpoint()["extra"] == {"start": 0}
            with pytest.raises(SamplerError, match="after the workers of this sampler were started"):
                s.restore()
            return noise

    def resumed(directory, start):
        with sampler(directory) as s:
            assert s.restore() and s.start_number == start
            noise = noise_of_steps(s, 10)
            assert s.storage.read_checkpoint()["extra"] == {"start": start}
            return noise

    def correlation(a, b):
        return abs(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))

    first = fresh(tmp_path / "a")
    assert np.array_equal(first, fresh(tmp_path / "b"))             # the same seed gives the same run
    # it is the thermal noise: sqrt(1 - decay^2) * sqrt(kT / m) = 0.029 nm/ps for 12 Da at 298 K
    assert first.std() == pytest.approx(np.sqrt(1 - decay ** 2) * np.sqrt(8.31446e-3 * 298.15 / 12.0), rel=0.25)
    shutil.copytree(tmp_path / "a", tmp_path / "c")
    second = resumed(tmp_path / "a", 1)
    assert np.array_equal(second, resumed(tmp_path / "c", 1))       # a resume from the same checkpoint too
    assert correlation(first, second) < 0.3                         # it was 1.000: the same numbers again
    third = resumed(tmp_path / "a", 2)                              # and every further start has its own
    assert correlation(first, third) < 0.3 and correlation(second, third) < 0.3
    # a checkpoint of a version that did not count the starts: the leg has started once before
    with sampler(tmp_path / "b") as s:
        s.storage.write_checkpoint(s.arrays.pos[0], s.arrays.vel[0], s.arrays.box[0], np.arange(2), 10)
        assert s.restore() and s.start_number == 1

    # the seeds: within the range of OpenMM and of the plugin, never zero, one per start, worker and purpose
    seeds = {stream_seed(11, start, worker, purpose) for start in range(3) for worker in range(40)
             for purpose in (0, BAROSTAT_STREAM)}
    assert len(seeds) == 240 and all(1 <= value < 2 ** 31 for value in seeds)
    assert 1 <= stream_seed(-5, 0, 0) < 2 ** 31                     # a negative seed in the configuration

    # a worker that is started again in the middle of a run gets a new number, and with it a new stream
    class Fake:
        def __init__(self, ctx, spec, device, names, worker_id, start=0):
            self.device, self.worker_id, self.start = device, worker_id, start

        def wait_ready(self, timeout):
            return "ready"

        def stop(self):
            pass

    monkeypatch.setattr(hrex, "_Process", Fake)
    with ReplicaExchangeSampler(spec, tmp_path / "r", devices=[None, None], inline=False, log=lambda m: None) as s:
        s.start_number = 3
        s.workers, s._next_worker_id = [Fake(None, spec, None, None, 0, 3), Fake(None, spec, None, None, 1, 3)], 2
        s._restart_worker(0)
        assert (s.workers[0].worker_id, s.workers[0].start, s._next_worker_id) == (2, 3, 3)
        assert s.workers[1].worker_id == 1


def test_samples_are_continued_only_under_the_conditions_they_were_made_under(tmp_path, tiny_leg):
    """Review of PR 1: a leg ended at 300 K was reopened at 315 K and analysed as if nothing had changed."""
    import dataclasses

    import h5py

    from cufep.sampling.storage import ConditionsError, stored_backend

    spec, pos, box = tiny_leg
    leg = tmp_path / "hrex"
    assert stored_backend(leg) is None
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        assert s.conditions_of_storage == "new"
        s.initialize(pos, box)
        s.run(6, record_stride=2, checkpoint_stride=3)
    assert stored_backend(leg) == "openmm"

    def stored():
        with h5py.File(leg / "samples.h5", "r") as h5:
            return h5["u"].shape[0], float(h5.attrs["temperature_K"]), str(h5.attrs["states"]), h5["u"][:].copy()

    before = stored()
    assert before[:2] == (6, 300.0)
    other_states = [dict(st) for st in spec.states]
    other_states[1]["lambda_electrostatics"] = 0.4
    changes = {"temperature_K": dataclasses.replace(spec, temperature_K=315.0),
               "the lambda states": dataclasses.replace(spec, states=other_states),
               "the System": dataclasses.replace(spec, system_xml=spec.system_xml + "\n"),
               "steps_per_iteration": dataclasses.replace(spec, steps_per_iteration=20),
               "pressure_bar": dataclasses.replace(spec, pressure_bar=2.0),
               "backend": dataclasses.replace(spec, backend="cupotential")}
    for what, changed in changes.items():
        with pytest.raises(ConditionsError, match=what) as err:
            ReplicaExchangeSampler(changed, leg, devices=[None], inline=True, log=lambda m: None)
        assert "left as they are" in str(err.value) and "new output.dir" in str(err.value)
        after = stored()
        assert after[:3] == before[:3] and np.array_equal(after[3], before[3])     # nothing was written

    # the same conditions: the run goes on, here to a longer production
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        assert s.conditions_of_storage == "same" and s.restore() and s.iteration == 6
        s.run(2, record_stride=2, checkpoint_stride=3)
    assert stored()[0] == 8



def test_the_tail_correction_is_recorded_with_the_samples_as_a_summary(tmp_path, tiny_leg):
    """The conditions of the samples held every parameter of every environment atom of the tail
    correction: 0.6 MB for a kinase and 1.7 MB for a receptor in a bilayer, in one attribute. They hold
    its summary now; samples that hold the parameters are compared as their summary."""
    import dataclasses

    import h5py

    from cufep.alchemy.lrc import TailEnergy
    from cufep.config.schema import AnalysisConfig
    from cufep.sampling.storage import ConditionsError
    from cufep.sampling.worker import tail_summary
    from cufep.workflow import leg_result_from_samples

    spec, _pos, _box = tiny_leg
    rng = np.random.default_rng(9)
    tail = TailEnergy(ligand=np.arange(3), lig_sigma=np.full(3, 0.3), lig_epsilon=np.full(3, 0.4), env=np.arange(3, 30003),
                      env_sigma=rng.uniform(0.2, 0.4, 30000), env_epsilon=rng.uniform(0.1, 0.8, 30000),
                      cutoff_nm=0.9, long_cutoff_nm=2.0)
    with_tail = dataclasses.replace(spec, lrc=tail.to_dict())
    conditions = with_tail.conditions()
    assert conditions["lrc"] == tail.summary() == tail_summary(tail.summary()) and conditions["lrc"]["n_environment"] == 30000
    assert len(json.dumps(conditions["lrc"])) < 400 < 1_000_000 < len(json.dumps(tail.to_dict()))
    assert spec.conditions()["lrc"] is None

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    synthetic_leg(leg)
    with h5py.File(leg / "hrex" / "samples.h5", "a") as h5:                # as the version before wrote it
        h5.attrs["conditions"] = json.dumps(dict(conditions, lrc=tail.to_dict()))
    storage = LegStorage(leg / "hrex", 4, 4)
    assert storage.check_conditions(conditions) == "same"
    other_radius = dataclasses.replace(with_tail, lrc=dataclasses.replace(tail, long_cutoff_nm=1.5).to_dict())
    with pytest.raises(ConditionsError, match="the tail correction .long_cutoff_nm: 2.0 -> 1.5."):
        storage.check_conditions(other_radius.conditions())
    other_atoms = dataclasses.replace(with_tail, lrc=dataclasses.replace(tail, env_epsilon=tail.env_epsilon * 1.01).to_dict())
    with pytest.raises(ConditionsError, match="the Lennard-Jones parameters of the ligand's dispersion tail"):
        storage.check_conditions(other_atoms.conditions())
    with pytest.raises(ConditionsError, match="the tail correction .recorded with the samples or not."):
        storage.check_conditions(spec.conditions())
    storage.close()
    # the result of a leg says to which radius its tail energies were recorded
    run = {"ligand": "lig", "leg": "solvent", "repeat": 0, "backend": "openmm", "notes": []}
    assert leg_result_from_samples(leg, AnalysisConfig(lrc="none"), run)["lrc_long_cutoff_nm"] == 2.0


def test_samples_of_unrecorded_conditions_are_read_but_not_continued(tmp_path, tiny_leg, water_ligand_system):
    """Review of PR 1: six iterations written by an earlier version with a soft-core alpha of 0.5 were
    continued with 0.8, and all ten rows were then labelled with the hash of the 0.8 System.

    That version recorded neither the soft-core nor a hash of its System, and it rewrote temperature,
    backend and states whenever it opened the file: nothing on the disk shows what such samples were
    made with. They are read and analysed, never added to, and their file is not touched.
    """
    import dataclasses
    import hashlib

    import h5py

    from cufep.sampling.storage import ConditionsError

    spec, pos, box = tiny_leg
    leg = tmp_path / "hrex"
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(6, record_stride=2, checkpoint_stride=3)
        s.storage.set_meta(n_equilibration_iterations=2)
    with h5py.File(leg / "samples.h5", "a") as h5:                     # as the earlier version left it
        del h5.attrs["conditions"]

    def content():
        with h5py.File(leg / "samples.h5", "r") as h5:
            return ({name: h5[name][:].tobytes() for name in h5}, {key: str(h5.attrs[key]) for key in h5.attrs})

    before, checkpoint = content(), hashlib.sha256((leg / "checkpoint.npz").read_bytes()).hexdigest()
    system, _pos, _box, ligand = water_ligand_system
    other_alpha = dataclasses.replace(spec, system_xml=mm.XmlSerializer.serialize(
        make_alchemical_system(system, ligand, 0.8).system))
    for attempt in (other_alpha, spec):                                # even the same settings cannot be shown
        with ReplicaExchangeSampler(attempt, leg, devices=[None], inline=True, log=lambda m: None) as s:
            assert s.conditions_of_storage == "unrecorded" and s.unrecorded
            for step in (s.restore, lambda: s.run(4, record_stride=2, checkpoint_stride=3),
                         lambda: s.initialize(pos, box)):
                with pytest.raises(ConditionsError, match="did not record the conditions they were made under"
                                                          ".*can be analysed.*--reschedule"):
                    step()
        assert content() == before                                     # no dataset and no attribute changed
        assert hashlib.sha256((leg / "checkpoint.npz").read_bytes()).hexdigest() == checkpoint
    with h5py.File(leg / "samples.h5", "r") as h5:
        assert "conditions" not in h5.attrs and h5["u"].shape[0] == 6  # no conditions of today on old rows
    # the samples are still read and analysed
    assert analyze_leg(read_samples(leg), 300.0).n_iterations_used == 4


def test_an_interrupted_leg_is_not_analysed_with_its_equilibration(tmp_path, tiny_leg):
    """Review of PR 1: the end of the equilibration was written only when a run had finished, so the
    analysis of an interrupted leg counted its equilibration as production."""
    from click.testing import CliRunner

    from cufep.analysis.mbar import NoProduction
    from cufep.cli import main
    from cufep.sampling.storage import equilibration_boundary

    spec, pos, box = tiny_leg
    leg = tmp_path / "leg"
    # 8 iterations of equilibration and 6 of production are planned; the run stops after 6 iterations
    with ReplicaExchangeSampler(spec, leg / "hrex", devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.storage.set_plan(8, 14)
        s.run(6, record_stride=2, checkpoint_stride=3)
    data = read_samples(leg / "hrex")
    assert data["attrs"]["n_equilibration_iterations"] == 8 and data["attrs"]["n_planned_iterations"] == 14
    assert equilibration_boundary(data["attrs"], leg) == 8
    with pytest.raises(NoProduction, match="6 iteration.s., of which the first 8 are equilibration .the run was "
                                           "planned for 14.: 0 iteration.s. of production"):
        analyze_leg(data, 300.0)
    result = CliRunner().invoke(main, ["analyze", str(leg)])
    assert result.exit_code == 1 and "no production yet" in result.output

    # it goes on and stops again in its production: only the production is analysed
    with ReplicaExchangeSampler(spec, leg / "hrex", devices=[None], inline=True, log=lambda m: None) as s:
        assert s.restore() and s.iteration == 6
        s.storage.set_plan(8, 14)
        s.run(6, record_stride=2, checkpoint_stride=3)
    assert analyze_leg(read_samples(leg / "hrex"), 300.0).n_iterations_used == 4
    result = CliRunner().invoke(main, ["analyze", str(leg)])
    assert result.exit_code == 0
    shown = json.loads(result.output)
    assert shown["n_iterations_used"] == 4 and shown["n_equilibration_iterations"] == 8
    assert shown["n_iterations_stored"] == 12 and shown["n_planned_iterations"] == 14

    # the plan is on the disk before the first iteration, not only in the open file
    early = tmp_path / "early" / "hrex"
    with ReplicaExchangeSampler(spec, early, devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.storage.set_plan(8, 14)
        assert read_samples(early)["attrs"]["n_planned_iterations"] == 14

    # a run of an earlier version that did not finish: the boundary reads 0 and is not known
    old = tmp_path / "old"
    with ReplicaExchangeSampler(spec, old / "hrex", devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(6, record_stride=2, checkpoint_stride=3)
    attrs = read_samples(old / "hrex")["attrs"]
    assert "n_planned_iterations" not in attrs and equilibration_boundary(attrs, old) is None
    result = CliRunner().invoke(main, ["analyze", str(old)])
    assert result.exit_code == 1 and "--equilibration-ns" in result.output
    # 2 fs and 10 steps per iteration: 0.00004 ns are two iterations
    result = CliRunner().invoke(main, ["analyze", str(old), "--equilibration-ns", "0.00004"])
    assert result.exit_code == 0 and json.loads(result.output)["n_iterations_used"] == 4
    # once such a leg has a result, the number in its file is the one that run wrote
    (old / "result.json").write_text("{}")
    assert equilibration_boundary(attrs, old) == 0


def synthetic_leg(leg_dir, n=300, temperature=310.0, n_equilibration=20, lambda_states=None):
    """A leg directory with harmonic samples, as a finished run leaves it (no tail data, no schedule)."""
    u, states, _ = harmonic_samples(n, np.array([1.0, 1.6, 2.6, 4.2]), np.linspace(0.0, 0.6, 4), np.random.default_rng(12))
    storage = LegStorage(leg_dir / "hrex", 4, 4)
    for it in range(n):
        storage.append(u[it], states[it], np.ones(4))
    storage.set_meta(temperature_K=temperature, iteration_ps=2.5, backend="openmm", leg="solvent")
    if lambda_states is not None:
        storage.set_meta(states=lambda_states)
    storage.set_plan(n_equilibration, n)
    storage.close()


def attempted(analysis: dict, name: str) -> int:
    """How many points of the forward (or reverse) series an analysis made: those it shows and those whose
    free energy was not determined (parts_not_determined)."""
    return len(analysis[name]) + sum(text.startswith(f"{name} point") for text in analysis["parts_not_determined"])


def test_analyze_says_what_the_run_said(tmp_path):
    """Review of PR 1: `cufep analyze` judged a leg by the default levels, not by those of its run: a leg
    that its run called NOT RELIABLE at analysis.min_overlap 0.2 passed, with 10 points instead of 4."""
    import yaml
    from click.testing import CliRunner

    from cufep.analysis.mbar import ANALYSIS_VERSION
    from cufep.cli import main
    from cufep.config.schema import AnalysisConfig
    from cufep.workflow import WorkflowError, analysis_record, leg_result_from_samples, write_leg_result

    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    synthetic_leg(leg)
    settings = AnalysisConfig(lrc="none", min_overlap=0.5, convergence_points=4, convergence_blocks=2)
    run = {"ligand": "lig", "leg": "solvent", "repeat": 0, "backend": "openmm", "timestep_fs": 4.0,
           "notes": ["a note of the run"]}
    result = leg_result_from_samples(leg, settings, run)
    # temperature, length and boundary are those of the samples; no configuration is asked for them
    assert result["temperature_K"] == 310.0 and result["production_ns"] == pytest.approx(280 * 2.5 / 1000.0)
    assert result["analysis_record"] == analysis_record(settings) and result["analysis_record"]["version"] == ANALYSIS_VERSION == 5
    # the standard error of the leg: that of time blocks or the analytical one, whichever is larger. At the
    # overlap level of 0.5 asked for here no part of these samples is determined, so it is the analytical one
    usable = [value for value in result["analysis"]["delta_g_error_blocks"].values() if value is not None]
    analytic = float(np.hypot(result["delta_g_mbar_error"], result["lrc_error"]))
    assert result["delta_g_error"] == result["analysis"]["delta_g_error"] == pytest.approx(max(usable + [analytic]))
    assert result["delta_g_mbar_error"] == result["analysis"]["delta_g_mbar_error"]
    assert result["delta_g_decouple"] == result["analysis"]["delta_g"] == result["delta_g_mbar"] - result["lrc"]
    assert result["quality"]["uncertainty"] == result["delta_g_error"]
    assert result["quality"]["overlap"]["threshold"] == 0.5 and result["quality"]["overlap"]["ok"] is False
    assert attempted(result["analysis"], "forward") == 4 and result["analysis"]["blocks"] == []
    assert sum(text.startswith("block ") and " of 2 " in text for text in result["analysis"]["parts_not_determined"]) == 2
    assert result["notes"][0] == "a note of the run" and result["notes"][1].startswith("NOT RELIABLE")
    assert result["run"] == run and result["ligand"] == "lig" and result["lrc"] == 0.0
    write_leg_result(leg, result)

    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"] == "the settings the leg's result records"
    assert shown["quality"] == json.loads(json.dumps(result["quality"]))
    assert attempted(shown, "forward") == 4 and shown["blocks"] == [] and shown["notes"] == result["notes"]
    assert shown["delta_g_decouple"] == pytest.approx(result["delta_g_decouple"]) and shown["temperature_K"] == 310.0
    assert json.loads((leg / "result.json").read_text())["notes"] == result["notes"]      # the command only prints

    # other settings on request; the temperature of the configuration is not one of them
    base = {"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "thermodynamics": {"temperature_K": 350.0}}
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({**base, "analysis": {"lrc": "none"}}))
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg), "--config", str(other)]).output)
    assert shown["analysis_settings_from"].startswith("--config") and shown["temperature_K"] == 310.0
    assert shown["quality"]["overlap"] == dict(shown["quality"]["overlap"], threshold=0.03, ok=True)
    assert len(shown["forward"]) == 10 and len(shown["blocks"]) == 5 and shown["notes"] == ["a note of the run"]

    # these samples hold no data for a tail correction: asking for one stops
    with pytest.raises(WorkflowError, match="without the data for a correction of the dispersion tail"):
        leg_result_from_samples(leg, AnalysisConfig(lrc="reweight"), run)
    other.write_text(yaml.safe_dump({**base, "analysis": {"lrc": "isotropic"}}))
    stopped = CliRunner().invoke(main, ["analyze", str(leg), "--config", str(other)])
    assert stopped.exit_code == 1 and "cannot be analysed with analysis.lrc: isotropic" in stopped.output

    # a result of a version that recorded no settings: the defaults, and the command says so; the
    # notes of that version's analysis are not carried over
    older = {k: v for k, v in result.items() if k not in ("analysis_record", "run")}
    older["notes"] = ["a note of the run", "NOT CONVERGED: the first half of the production gives ..."]
    (leg / "result.json").write_text(json.dumps(json.loads(__import__("cufep.jsonio", fromlist=["dumps"]).dumps(older))))
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"].startswith("the defaults") and shown["lrc_method"] == "none"
    assert shown["quality"]["overlap"]["threshold"] == 0.03 and shown["notes"] == ["a note of the run"]


def test_a_leg_whose_first_run_was_interrupted_is_judged_by_the_levels_of_that_run(tmp_path):
    """Review of PR 1: a leg without a result was analysed with the default settings: 100 iterations of a
    run with analysis.min_overlap 0.45, 4 points and 2 blocks passed at 0.03 with 10 points and 5 blocks."""
    from click.testing import CliRunner

    from cufep.cli import main
    from cufep.config.schema import Config
    from cufep.workflow import analysis_record, analysis_settings, analysis_settings_of

    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"},
                                 "analysis": {"lrc": "none", "min_overlap": 0.45, "convergence_points": 4,
                                              "convergence_blocks": 2},
                                 "membrane": {"area_tolerance": 0.02}, "restraints": {"pose_rmsd_warning_nm": 0.2}})
    record = analysis_record(cfg)
    assert record["min_overlap"] == 0.45 and record["membrane_area_tolerance"] == 0.02
    assert record["error_blocks"] == [40, 20, 10, 5] and record["part_error_blocks"] == 5
    assert record["membrane_pocket_water_tolerance"] == 1.5 and record["pose_rmsd_warning_nm"] == 0.2
    # the settings a record describes give the same record, and an analysis section alone the default levels
    assert analysis_record(analysis_settings_of(record)) == record and analysis_settings(cfg) is cfg
    alone = analysis_record(cfg.analysis)
    assert alone == dict(record, membrane_area_tolerance=0.01, pose_rmsd_warning_nm=0.3)

    # the run stores them with the samples before its first iteration; it is interrupted after 100
    leg = tmp_path / "lig" / "repeat_0" / "solvent"
    u, states, _ = harmonic_samples(100, np.array([1.0, 1.6, 2.6, 4.2]), np.linspace(0.0, 0.6, 4), np.random.default_rng(12))
    storage = LegStorage(leg / "hrex", 4, 4)
    storage.set_meta(temperature_K=310.0, iteration_ps=2.5, backend="openmm", leg="solvent")
    storage.set_plan(4, 400, record)
    for it in range(100):
        storage.append(u[it], states[it], np.ones(4))
    storage.close()
    assert read_samples(leg / "hrex")["attrs"]["analysis_record"] == record
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"] == "the settings of the run that made the samples, which has no result (yet)"
    assert shown["quality"]["overlap"]["threshold"] == 0.45 and shown["quality"]["overlap"]["ok"] is False
    assert attempted(shown, "forward") == 4 and shown["analysis_record"] == record
    assert shown["n_planned_iterations"] == 400 and shown["n_iterations_used"] == 96
    # the result the leg had before that run is of an earlier run: the run that made the samples comes first
    earlier = dict(record, min_overlap=0.03, convergence_points=10)
    (leg / "result.previous.json").write_text(json.dumps({"analysis_record": earlier, "lrc_method": "none"}))
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["quality"]["overlap"]["threshold"] == 0.45 and attempted(shown, "forward") == 4
    # a finished leg says what its result says
    (leg / "result.json").write_text(json.dumps({"analysis_record": earlier, "lrc_method": "none"}))
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"] == "the settings the leg's result records"
    assert shown["quality"]["overlap"]["threshold"] == 0.03 and len(shown["forward"]) == 10
    (leg / "result.json").unlink()
    # samples of a run that stored no settings: the result the leg had before, then the defaults
    import h5py
    with h5py.File(leg / "hrex" / "samples.h5", "a") as h5:
        del h5.attrs["analysis_record"]
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"] == "the settings the result records that the leg had before"
    assert shown["quality"]["overlap"]["threshold"] == 0.03 and len(shown["forward"]) == 10
    (leg / "result.previous.json").unlink()
    shown = json.loads(CliRunner().invoke(main, ["analyze", str(leg)]).output)
    assert shown["analysis_settings_from"].startswith("the defaults") and shown["analysis_record"] == dict(
        alone, min_overlap=0.03, convergence_points=10, convergence_blocks=5)


def test_a_result_of_another_analysis_is_made_again_before_the_legs_are_combined(tmp_path):
    """Review of PR 1: when only one leg was run again, its new analysis was combined with the free energy
    and the uncertainty that an earlier method of analysis had given the other leg."""
    import shutil

    from cufep import jsonio
    from cufep.analysis.mbar import ANALYSIS_VERSION
    from cufep.config.schema import AnalysisConfig
    from cufep.workflow import (analysis_record, combine_ligand, leg_result_from_samples, refresh_leg_result,
                                write_leg_result)

    lig, settings, results = tmp_path / "lig", AnalysisConfig(lrc="none"), {}
    for leg in ("complex", "solvent"):
        synthetic_leg(lig / "repeat_0" / leg)
        run = {"ligand": "lig", "leg": leg, "repeat": 0, "backend": "openmm", "timestep_fs": 4.0,
               "notes": ["a note of the run"], "pressure_bar": 1.0, "prepared_systems_sha256": "abc"}
        results[leg] = leg_result_from_samples(lig / "repeat_0" / leg, settings, run)
        write_leg_result(lig / "repeat_0" / leg, results[leg])
    solvent = lig / "repeat_0" / "solvent"
    current = (results["solvent"]["delta_g_decouple"] - results["complex"]["delta_g_decouple"]) / 4.184

    # the solvent leg has a result as an earlier version wrote it: other numbers, the checks of that
    # version, no record of the analysis
    earlier = {k: v for k, v in results["solvent"].items() if k not in ("analysis_record", "run")}
    earlier.update(delta_g_decouple=results["solvent"]["delta_g_decouple"] + 2.0 * 4.184, delta_g_mbar_error=0.01,
                   quality={"overlap": {"ok": True}, "convergence": {"ok": False}},
                   notes=["a note of the run", "NOT CONVERGED: the first half of the production gives ..."])
    (solvent / "result.json").write_text(jsonio.dumps(earlier))
    named = (f"solvent leg: made before the version of the analysis was recorded; this run uses version "
             f"{ANALYSIS_VERSION}; not checked: independent samples, halves of the production, BAR against MBAR")
    mixed = combine_ligand(lig, "lig", 1, settings)
    assert mixed["dg_bind"] == pytest.approx(current + 2.0) and mixed["old_analysis"] == [named]

    messages = []
    assert refresh_leg_result(solvent, settings, messages.append) == "reanalysed"
    assert refresh_leg_result(lig / "repeat_0" / "complex", settings, messages.append) == "current"
    assert len(messages) == 1 and "kept as result.replaced.json" in messages[0]
    assert json.loads((solvent / "result.replaced.json").read_text())["delta_g_mbar_error"] == 0.01
    again = json.loads((solvent / "result.json").read_text())
    assert again["analysis_record"] == analysis_record(settings)
    assert again["delta_g_decouple"] == pytest.approx(results["solvent"]["delta_g_decouple"])
    assert again["notes"][0] == "a note of the run" and not any(n.startswith("NOT CONVERGED") for n in again["notes"])
    assert refresh_leg_result(solvent, settings, messages.append) == "current" and len(messages) == 1   # once is enough
    out = combine_ligand(lig, "lig", 1, settings)
    assert out["dg_bind"] == pytest.approx(current) and out["old_analysis"] == []

    # other settings of the analysis are another analysis as well
    fewer = AnalysisConfig(lrc="none", convergence_points=4)
    assert refresh_leg_result(solvent, fewer, messages.append) == "reanalysed"
    assert "convergence_points: 10 -> 4" in messages[-1]
    assert len(json.loads((solvent / "result.json").read_text())["analysis"]["forward"]) == 4

    # without its samples a result cannot be made again: it is combined as it is, and the ligand's
    # result names the leg and the checks its result does not hold
    (solvent / "result.json").write_text(jsonio.dumps(earlier))
    shutil.rmtree(solvent / "hrex")
    assert refresh_leg_result(solvent, settings, messages.append) == "stale"
    assert "its samples are not there" in messages[-1] and not (solvent / "result.json.tmp").exists()
    out = combine_ligand(lig, "lig", 1, settings)
    assert out["dg_bind"] == pytest.approx(current + 2.0) and out["old_analysis"] == [named]
    assert any("solvent leg" in w and "combined as they are" in w for w in out["warnings"])
    assert refresh_leg_result(lig / "repeat_0" / "nothing", settings, messages.append) == "not finished"


def test_samples_of_unrecorded_conditions_must_be_what_the_run_asks_for(tmp_path):
    """Review of PR 1, two findings on samples that an earlier version wrote without their conditions and
    that are analysed as they are: the release of the restraint was taken at the temperature of the
    configuration (and the pressure filled in from it), and 0.10 ns of production passed for 0.50 ns
    because the rows were counted with the iteration length of the configuration."""
    import h5py

    from cufep.alchemy.restraints import BoreschRestraint
    from cufep.config.schema import AnalysisConfig, Config
    from cufep.sampling.storage import stored_attributes
    from cufep.workflow import (WorkflowError, combine_ligand, leg_result_from_samples, restraint_release_of,
                                unrecorded_samples_problems, write_leg_result)

    def config(**sections):
        return Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, **sections})

    bound = [x.to_dict() for x in fixed_schedule(["restrain", "discharge", "vanish"],
                                                  {"restrain": 2, "discharge": 2, "vanish": 2}).states]
    free = [x.to_dict() for x in fixed_schedule(["discharge", "vanish"], {"discharge": 2, "vanish": 3}).states]
    lig = tmp_path / "lig"
    complex_dir, solvent_dir = lig / "repeat_0" / "complex", lig / "repeat_0" / "solvent"
    # 60 iterations of 2.5 ps at 300 K, the first 20 equilibration: 0.10 ns of production
    for leg_dir, lambda_states in ((complex_dir, bound), (solvent_dir, free)):
        synthetic_leg(leg_dir, n=60, temperature=300.0, n_equilibration=20, lambda_states=lambda_states)
        with h5py.File(leg_dir / "hrex" / "samples.h5", "a") as h5:        # as an earlier version left a finished leg
            del h5.attrs["n_planned_iterations"]
    n, attrs = stored_attributes(complex_dir / "hrex")
    assert n == 60 and "conditions" not in attrs and attrs["temperature_K"] == 300.0

    def problems(cfg, production_ns, held=attrs, lambda_states=bound, timestep_fs=4.0):
        return unrecorded_samples_problems(held, n, complex_dir, cfg, "complex", production_ns, lambda_states, timestep_fs)

    asked = {"equilibration_ns": 0.05, "production_ns": 0.1}
    same = config(thermodynamics={"temperature_K": 300.0}, sampling=asked)
    assert problems(same, 0.1) == []
    # the reviewer's case: an iteration of 10 ps and 0.50 ns of production are asked for; 55 rows "were enough"
    longer = config(thermodynamics={"temperature_K": 300.0},
                    sampling={"time_per_iteration_ps": 10.0, "equilibration_ns": 0.05, "production_ns": 0.5})
    assert problems(longer, 0.5) == [
        "length of an iteration: 2.5 ps -> 10 ps",
        "production per lambda window: the samples hold 0.1 ns and 0.5 ns are asked for (samples of unrecorded "
        "conditions are not added to)"]
    assert problems(same, 0.2) == ["production per lambda window: the samples hold 0.1 ns and 0.2 ns are asked for "
                                   "(samples of unrecorded conditions are not added to)"]
    assert problems(config(thermodynamics={"temperature_K": 310.0}, sampling=asked), 0.1) == ["temperature: 300 K -> 310 K"]
    assert problems(config(thermodynamics={"temperature_K": 300.0}, sampling=dict(asked, equilibration_ns=0.1)), 0.1) \
        == ["equilibration per lambda window: 0.05 ns -> 0.1 ns"]
    assert problems(same, 0.1, lambda_states=free) == ["the lambda states"]
    # states of a version before REST2 are the same states; 2.5 ps at 3 fs are 833 steps
    assert problems(same, 0.1, held=dict(attrs, states=[{k: v for k, v in x.items() if k != "rest2_scale"}
                                                         for x in bound])) == []
    assert problems(same, 0.0999, held=dict(attrs, iteration_ps=2.499), timestep_fs=3.0) == []
    # a run that did not finish and did not record where its production begins
    unknown = problems(same, 0.1, held=dict(attrs, n_equilibration_iterations=0))
    assert len(unknown) == 1 and "does not record where its production begins" in unknown[0]
    assert "does not record the length of an iteration" in problems(same, 0.1, held={"temperature_K": 300.0})[-1]

    # the release of the restraint: from restraint.json at the temperature of the samples
    restraint = BoreschRestraint(host_atoms=(1, 2, 3), guest_atoms=(4, 5, 6), r0=0.45, theta_a0=1.2, theta_b0=1.3,
                                 phi_a0=0.3, phi_b0=-0.5, phi_c0=1.0)
    run = {"ligand": "lig", "leg": "complex", "repeat": 0, "backend": "openmm", "timestep_fs": 4.0, "notes": []}
    with pytest.raises(WorkflowError, match="restraint.json is not there"):
        leg_result_from_samples(complex_dir, AnalysisConfig(lrc="none"), run)
    (complex_dir / "restraint.json").write_text(json.dumps({"restraint": restraint.to_dict(), "info": {}}))
    at_300 = restraint.release_free_energy(300.0, "numerical")
    assert restraint_release_of(complex_dir, attrs, 300.0) == pytest.approx((at_300, restraint.release_free_energy(300.0, "analytical")))
    assert abs(restraint.release_free_energy(315.0, "numerical") - at_300) > 1.0   # kJ/mol: the temperature matters
    assert restraint_release_of(complex_dir / "previous_1", attrs, 300.0)[0] == pytest.approx(at_300)
    assert restraint_release_of(solvent_dir, stored_attributes(solvent_dir / "hrex")[1], 300.0) == (0.0, 0.0)
    bound_result = leg_result_from_samples(complex_dir, AnalysisConfig(lrc="none"), run)
    assert bound_result["temperature_K"] == 300.0 and bound_result["restraint_release"] == pytest.approx(at_300)
    assert bound_result["pressure_bar"] is None                    # not recorded, and not filled in from a configuration
    free_result = leg_result_from_samples(solvent_dir, AnalysisConfig(lrc="none"), dict(run, leg="solvent"))
    assert free_result["restraint_release"] == 0.0
    write_leg_result(complex_dir, bound_result)
    write_leg_result(solvent_dir, free_result)
    out = combine_ligand(lig, "lig", 1)
    assert out["dg_bind"] == pytest.approx((free_result["delta_g_decouple"] - bound_result["delta_g_decouple"] - at_300) / 4.184)
    assert "pressure (bar): not recorded for the complex leg, the solvent leg" in out["not_compared"]
    # samples that do record their conditions give their pressure
    with h5py.File(solvent_dir / "hrex" / "samples.h5", "a") as h5:
        h5.attrs["conditions"] = json.dumps({"pressure_bar": 1.0})
    assert leg_result_from_samples(solvent_dir, AnalysisConfig(lrc="none"), dict(run, leg="solvent"))["pressure_bar"] == 1.0


def test_an_interruption_before_the_first_checkpoint_does_not_leave_samples_twice(tmp_path, tiny_leg):
    """Review of PR 1: two iterations before the first checkpoint stayed in front of the rerun's six."""
    spec, pos, box = tiny_leg
    for name, remove_checkpoint in (("with_start_checkpoint", False), ("as_older_versions_left_it", True)):
        leg = tmp_path / name
        with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
            s.initialize(pos, box)
            at_start = (leg / "checkpoint.npz").read_bytes()
            s.run(2, record_stride=1, checkpoint_stride=5)
            assert s.storage.n_iterations == 2
        # the process is killed during its third iteration: two iterations are in the storage and
        # the checkpoint on disk is the one of the start (older versions had written none yet)
        if remove_checkpoint:
            (leg / "checkpoint.npz").unlink()
        else:
            (leg / "checkpoint.npz").write_bytes(at_start)
        with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
            if not s.restore():
                assert remove_checkpoint
                s.initialize(pos, box)
            assert s.iteration == 0 and s.storage.n_iterations == 0
            s.run(6, record_stride=1, checkpoint_stride=5)
            s.storage.set_meta(n_equilibration_iterations=2)
        data = read_samples(leg)
        assert data["u"].shape[0] == 6 and data["frame_iter"].tolist() == sorted(set(data["frame_iter"].tolist()))
        assert len(data["lrc_iter"]) == len(set(data["lrc_iter"].tolist()))
        assert analyze_leg(data, 300.0).n_iterations_used == 4

