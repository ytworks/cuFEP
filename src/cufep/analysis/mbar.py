"""MBAR analysis of one replica-exchange leg."""
from __future__ import annotations

import contextlib
import io
import os
from dataclasses import asdict, dataclass, field

import numpy as np

R_GAS = 8.31446261815324e-3   # kJ/mol/K
KCAL = 4.184
#: The version of the method by which a leg is analysed. A result records it; a result of another
#: version is made again from its samples before it is combined with others.
#:   1  one statistical inefficiency for all lambda windows, from the summed reduced potential
#:   2  one per window, from its energy difference to the next window
#:   3  the larger of those to both neighbours; overlap also among the samples of the estimate;
#:      the samples check counts what MBAR gets
#:   4  standard error from time blocks common to all windows (delete-one-block jackknife), also
#:      for the halves; the samples check counts what that error leaves of the samples
ANALYSIS_VERSION = 4
#: numbers of time blocks the standard error of a leg is computed with; the largest of the errors
#: they give is the one reported (:func:`block_standard_error`). The numbers are cuFEP's choice:
#: the literature asks for blocks longer than the correlation time and for a standard error that
#: no longer grows with the length of the blocks (Grossfield et al. 2018, section 7.3.2), and
#: gives no numbers. Defaults of ``analysis.error_blocks`` and ``analysis.halves_error_blocks``.
BLOCK_COUNTS = (40, 20, 10, 5)
#: number of time blocks for the standard error of one half of the production
HALF_BLOCK_COUNT = 5
#: a part of the production gets a free energy estimate of its own only if it holds at least this
#: many statistically independent samples per lambda window
MIN_INDEPENDENT_SAMPLES = 4


class NoProduction(ValueError):
    """The samples hold no production to analyse (yet)."""


def _pymbar():
    # pymbar solves with JAX when JAX is installed. A job process once stopped for good inside
    # that solver (a deadlock in JAX's array conversion) after the sampling had finished, so
    # pymbar is told to use NumPy; the variable is read when pymbar is first imported.
    # PYMBAR_DISABLE_JAX=0 in the environment brings JAX back.
    os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pymbar
        from pymbar import timeseries
    return pymbar, timeseries


@dataclass
class LegAnalysis:
    n_states: int
    n_iterations_used: int
    #: statistical inefficiency of the reduced potential summed over the replicas, in iterations;
    #: used only for the overlap matrix (equal numbers of samples per window)
    statistical_inefficiency: float
    #: independent samples, as estimated, in the lambda window that has the fewest
    #: (see window_effective_samples and window_kept_samples)
    effective_samples_per_state: float
    #: free energy of the last state relative to the first, kJ/mol, by MBAR from samples that are
    #: decorrelated window by window, and its standard error: the largest of delta_g_error_blocks,
    #: or MBAR's own (delta_g_error_mbar) where no number of blocks can be used
    #: (delta_g_error_note says why)
    delta_g: float
    delta_g_error: float
    #: the standard error MBAR gives for these samples; it takes them as independent
    delta_g_error_mbar: float = float("nan")
    #: the standard error by the delete-one-block jackknife for every number of time blocks that
    #: was asked for (key: the number as text; None where that number cannot be used, with the
    #: reason in delta_g_error_blocks_note), and the number whose error is reported
    delta_g_error_blocks: dict[str, float | None] = field(default_factory=dict)
    delta_g_error_blocks_note: dict[str, str] = field(default_factory=dict)
    delta_g_error_n_blocks: int | None = None
    delta_g_error_note: str = ""
    #: statistical inefficiency of every lambda window, in iterations: the larger of those of its
    #: energy differences to the previous and to the next window, in the configurations of the window
    window_statistical_inefficiency: list[float] = field(default_factory=list)
    #: the two it is the larger of (NaN where there is no such neighbour), and which neighbour
    #: gave it: -1 the previous window, +1 the next
    window_inefficiency_previous: list[float] = field(default_factory=list)
    window_inefficiency_next: list[float] = field(default_factory=list)
    window_inefficiency_neighbour: list[int] = field(default_factory=list)
    #: independent samples per lambda window as estimated: production iterations / statistical
    #: inefficiency
    window_effective_samples: list[float] = field(default_factory=list)
    #: samples every lambda window gives to MBAR: every ceil(statistical inefficiency)-th
    #: iteration, so never more than the estimate above
    window_kept_samples: list[int] = field(default_factory=list)
    #: the window with the fewest samples (the smaller of the two numbers above)
    slowest_window: int = 0
    #: cumulative free energy at every state relative to the first, kJ/mol
    f_k: list[float] = field(default_factory=list)
    df_k: list[float] = field(default_factory=list)
    stage_delta_g: dict[str, float] = field(default_factory=dict)
    overlap_neighbour: list[float] = field(default_factory=list)
    min_overlap: float = 0.0
    round_trips: int = 0
    swap_acceptance_neighbour: list[float] = field(default_factory=list)
    #: the whole overlap matrix of the windows (overlap_neighbour is its first off-diagonal),
    #: from all iterations with the same number of samples in every window
    overlap_matrix: list[list[float]] = field(default_factory=list)
    #: the overlap matrix of the samples the free energy is estimated from (window_kept_samples
    #: per window), in the normalisation of equal numbers of samples per window
    #: (overlap_with_equal_weights); overlap_used_neighbour is its first off-diagonal
    overlap_used_matrix: list[list[float]] = field(default_factory=list)
    overlap_used_neighbour: list[float] = field(default_factory=list)
    min_overlap_used: float = float("nan")
    #: pymbar's own matrix of those samples, O[i][j] = N_j * sum_n W_ni W_nj: with different
    #: numbers of samples O[i][j] and O[j][i] differ by the ratio of the numbers. For the record;
    #: no level is applied to it
    overlap_used_weighted_by_number: list[list[float]] = field(default_factory=list)
    #: free energy from growing parts of the production, counted from its start (forward) and
    #: from its end (reverse): (fraction of the production, dG, error)
    forward: list[tuple[float, float, float]] = field(default_factory=list)
    reverse: list[tuple[float, float, float]] = field(default_factory=list)
    #: (dG, error) from the first half and from the second half of the production alone; the
    #: halves do not share an iteration. The error is that of a number of time blocks within
    #: the half, or MBAR's own where they cannot be used: halves_error_blocks holds the number of
    #: blocks behind each of the two errors (None: MBAR's own)
    first_half: tuple[float, float] = (float("nan"), float("nan"))
    second_half: tuple[float, float] = (float("nan"), float("nan"))
    halves_error_blocks: list[int | None] = field(default_factory=list)
    #: why the halves have no estimate, if the production is too short for them
    convergence_note: str = ""
    #: difference between the two halves
    forward_reverse_gap: float = 0.0
    #: free energy from consecutive blocks of the production that do not overlap:
    #: (start, end as fractions of the production, dG, error)
    blocks: list[tuple[float, float, float, float]] = field(default_factory=list)
    #: sum of BAR between neighbouring windows, a second estimator on the same data
    bar_delta_g: float = float("nan")
    bar_delta_g_error: float = float("nan")
    #: why BAR has no estimate, if it has none
    bar_note: str = ""
    lrc_delta_g: float = 0.0
    lrc_error: float = 0.0
    lrc_samples: int = 0
    #: the same correction in the homogeneous (isotropic) approximation that OpenMM's
    #: long-range correction and OpenFE use; reported for comparison
    lrc_isotropic: float = 0.0
    temperature_K: float = 298.15

    def to_dict(self) -> dict:
        """For a result file: a number that has no value (NaN) is None there, see cufep.jsonio."""
        from cufep.jsonio import clean

        return clean(asdict(self))


def assemble(u: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``u[iter, replica, state]`` and ``states[iter, replica]`` into pymbar's (u_kn, N_k)."""
    n_iter, n_rep, n_states = u.shape
    order = np.argsort(states.reshape(-1), kind="stable")           # samples grouped by sampled state
    u_kn = u.reshape(n_iter * n_rep, n_states)[order].T
    n_k = np.bincount(states.reshape(-1), minlength=n_states)
    return np.ascontiguousarray(u_kn), n_k


def _mbar(u_kn: np.ndarray, n_k: np.ndarray, initial_f_k=None, max_iterations: int = 1000):
    """A solved ``pymbar.MBAR``.

    Solving from zeros takes the adaptive solver more than a hundred iterations for a leg of
    thirty windows. It starts from the given free energies instead, or from BAR between
    neighbouring windows, which leaves a few iterations; the solution is the same.

    The solver is pymbar's adaptive one (a Newton-Raphson or a self-consistent step, whichever
    lowers the gradient more) on its own; the "robust" protocol follows it with L-BFGS-B.
    ``max_iterations`` bounds it. If a window overlaps with no other, its free energy is not
    determined and the solver never meets its convergence test; with pymbar's limit of 10,000
    iterations per stage the analysis of such a leg took most of an hour.
    """
    pymbar, _ = _pymbar()
    # Overlapping-atom configurations give astronomically large energies in coupled states;
    # cap them so the solver stays finite (their weight is zero either way).
    u_kn = np.minimum(np.asarray(u_kn, dtype=float) - np.min(u_kn), 1.0e6)
    n_k = np.asarray(n_k)
    if initial_f_k is not None:
        start = {"initial_f_k": np.asarray(initial_f_k, dtype=float)}
    elif np.all(n_k > 0):
        start = {"initialize": "BAR"}
    else:
        start = {}

    def solve(**kwargs):
        # pymbar raises the solver's own limit to maximum_iterations and writes the result into
        # the protocol it is given, so both are set and the protocol is made anew for each solve
        protocol = (dict(method="adaptive", options=dict(maxiter=max_iterations)),)
        return pymbar.MBAR(u_kn, n_k, solver_protocol=protocol, maximum_iterations=max_iterations, **kwargs)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if start:
            try:
                return solve(**start)
            except Exception:  # noqa: BLE001  the starting point failed; solve from zeros
                pass
        return solve()


def solve_mbar(u: np.ndarray, states: np.ndarray, initial_f_k=None, overlap: bool = True,
               max_iterations: int = 1000):
    """Return (f_k, df_k relative to state 0, overlap matrix or None) in kT."""
    u_kn, n_k = assemble(u, states)
    mbar = _mbar(u_kn, n_k, initial_f_k, max_iterations)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = mbar.compute_free_energy_differences()
        matrix = np.asarray(mbar.compute_overlap()["matrix"]) if overlap else None
    return np.asarray(res["Delta_f"][0]), np.asarray(res["dDelta_f"][0]), matrix


def neighbour_inefficiencies(u: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Statistical inefficiencies of every lambda window towards its two neighbours, in iterations.

    Row j holds those of the energy differences between window j and window j - 1 (column 0) and
    between window j and window j + 1 (column 1), over the configurations window j holds from
    iteration to iteration, whichever replica holds them; NaN where there is no such neighbour.

    Free energy estimators need uncorrelated samples (Klimovich, Shirts and Mobley 2015, section
    2.5, who take dU/dlambda or "any of the potential energy differences"). alchemlyb's default
    (``decorrelate_u_nk``, method ``dE``) takes the difference to the next window only. MBAR uses
    the energies of a configuration in both neighbours, and a degree of freedom that is slow can
    change the difference to one neighbour and leave the other alone, so both are examined. Where
    replicas hardly exchange, a window keeps its configurations for a long time and these series
    show it; the reduced potential summed over all replicas does not.
    """
    _, timeseries = _pymbar()
    n, _n_rep, k = u.shape
    replica_in = np.argsort(states, axis=1)                    # replica that is in window j at each iteration
    it = np.arange(n)
    table = np.full((k, 2), np.nan)
    for j in range(k):
        r = replica_in[:, j]
        for column, other in ((0, j - 1), (1, j + 1)):
            if not 0 <= other < k:
                continue
            series = np.minimum(u[it, r, other] - u[it, r, j], 1.0e6)
            g = 1.0
            if n >= 8 and not np.allclose(series, series[0]):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        g = max(1.0, float(timeseries.statistical_inefficiency(series)))
                    except Exception:  # noqa: BLE001
                        g = 1.0
            table[j, column] = g
    return table


def window_inefficiencies(u: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Statistical inefficiency of every lambda window: the larger of its two in
    :func:`neighbour_inefficiencies`."""
    table = neighbour_inefficiencies(u, states)
    return np.array([np.nanmax(row) if np.isfinite(row).any() else 1.0 for row in table])


def kept_iterations(start: int, stop: int, g_window: float) -> np.ndarray:
    """The iterations in [start, stop) that a window with statistical inefficiency ``g_window`` keeps.

    Every ceil(g)-th iteration of the production, counted from its first one (uniform intervals,
    alchemlyb's "conservative" slicing). A part of the production keeps those of them that lie in
    it, so the samples of the parts are the samples of the whole, as in alchemlyb, where the
    series are decorrelated first and then divided.
    """
    stride = max(1, int(np.ceil(g_window)))
    return np.arange(-(-start // stride) * stride, stop, stride)


def decorrelated(u: np.ndarray, states: np.ndarray, g: np.ndarray, start: int = 0, stop: int | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """pymbar's (u_kn, N_k) for the iterations [start, stop), every window subsampled on its own.

    Window j contributes the configurations it held at :func:`kept_iterations`, whichever replica
    held them, so the windows contribute different numbers of samples.
    """
    n, _n_rep, k = u.shape
    stop = n if stop is None else stop
    replica_in = np.argsort(states, axis=1)                    # replica that is in window j at each iteration
    rows, n_k = [], []
    for j in range(k):
        keep = kept_iterations(start, stop, g[j])
        rows.append(u[keep, replica_in[keep, j], :])
        n_k.append(len(keep))
    return np.ascontiguousarray(np.concatenate(rows, axis=0).T), np.array(n_k)


def kept_sample_iterations(n: int, g: np.ndarray, start: int = 0, stop: int | None = None) -> np.ndarray:
    """The iteration of every sample (column) that :func:`decorrelated` returns for the same arguments."""
    stop = n if stop is None else stop
    return np.concatenate([kept_iterations(start, stop, g_window) for g_window in g])


def free_energies_from(u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray, tolerance: float = 1.0e-8,
                       max_iterations: int = 25) -> np.ndarray | None:
    """MBAR free energies (kT, relative to the first state) of samples grouped by state, by
    Newton-Raphson from the nearby solution ``f_k``. None if that does not converge.

    The estimate without one block of the production differs little from that of all of it, so a
    few steps from there solve the MBAR equations (sum_n P_ni = N_i, with P_ni the probability
    that sample n is of state i); pymbar would set up each of these problems anew and compute a
    covariance matrix that nobody reads. The result is pymbar's.
    """
    from scipy.special import logsumexp

    n_k = np.asarray(n_k, dtype=float)
    if np.any(n_k <= 0):
        return None
    f = np.asarray(f_k, dtype=float) - float(f_k[0])
    log_n = np.log(n_k)
    with np.errstate(all="ignore"):
        for _ in range(max_iterations):
            log_p = (log_n + f)[:, None] - u_kn
            p = np.exp(log_p - logsumexp(log_p, axis=0))
            total = p.sum(axis=1)
            gradient = total - n_k
            if not np.all(np.isfinite(gradient)):
                return None
            if np.max(np.abs(gradient) / n_k) < tolerance:
                return f
            hessian = np.diag(total) - p @ p.T
            try:
                step = np.linalg.solve(hessian[1:, 1:], gradient[1:])
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(step)):
                return None
            f[1:] -= step
    return None


def block_standard_error(u_kn: np.ndarray, n_k: np.ndarray, iteration: np.ndarray, start: int, stop: int,
                         n_blocks: int, f_k: np.ndarray, longest_interval: int) -> tuple[float | None, str]:
    """Standard error (kT) of the free energy of the last state relative to the first by the
    delete-one-block jackknife over blocks of time: (error, "") or (None, why this number of
    blocks cannot be used).

    The iterations [start, stop) are divided into ``n_blocks`` consecutive blocks of the same
    length, the same for all lambda windows. The free energy is estimated ``n_blocks`` times,
    each time without the samples of one block; the standard error is
    sqrt((B - 1) / B * sum_b (dG_b - mean)^2).

    Why blocks of time and not the standard error MBAR gives: in replica exchange a configuration
    that changes slowly travels from window to window. Each window then looks decorrelated on
    its own (its statistical inefficiency is short), and the samples of different windows are
    still correlated with each other, which MBAR's error does not see: for eight harmonic windows
    and a correlation of 0.97 from iteration to iteration it was a quarter of the scatter of
    independent runs (review of PR 1). A block of time holds what all windows sampled in it, so
    the correlation between the windows stays inside the blocks. "For correlated data, blocks
    that are uncorrelated are the unit that is resampled"; with few of them the jackknife is the
    method of choice; and the ensemble of a replica-exchange simulation "is constructed from"
    correlated trajectories that return many times (Grossfield et al. 2018, sections 7.6 and 8).

    ``iteration`` is the iteration of every sample (:func:`kept_sample_iterations`), ``f_k`` the
    solution for all of them, ``longest_interval`` the largest interval at which a window keeps
    its samples. A number of blocks is not used if a block is shorter than that interval (a
    block must hold a sample of every window) or if, without a block, a window keeps fewer than
    ``MIN_INDEPENDENT_SAMPLES`` samples.
    """
    n_k = np.asarray(n_k)
    k = len(n_k)
    edges = np.array([start + int(round(b * (stop - start) / n_blocks)) for b in range(n_blocks + 1)])
    shortest = int(np.diff(edges).min())
    if shortest < max(1, int(longest_interval)):
        return None, (f"a block of {shortest} iteration(s) is shorter than the interval of {int(longest_interval)} "
                      "iterations at which the slowest lambda window keeps its samples")
    window = np.repeat(np.arange(k), n_k)
    block = np.searchsorted(edges, iteration, side="right") - 1
    inside = np.zeros((n_blocks, k), dtype=int)
    np.add.at(inside, (block, window), 1)
    left = n_k[None, :] - inside
    if left.min() < MIN_INDEPENDENT_SAMPLES:
        b, j = np.unravel_index(int(np.argmin(left)), left.shape)
        return None, (f"without block {b + 1} lambda window {j} keeps {int(left[b, j])} sample(s); "
                      f"{MIN_INDEPENDENT_SAMPLES} are needed for an estimate")
    estimates = np.empty(n_blocks)
    for b in range(n_blocks):
        keep = block != b
        f = free_energies_from(u_kn[:, keep], left[b], f_k)
        if f is None:
            try:
                f, _df = solve_decorrelated(u_kn[:, keep], left[b], initial_f_k=f_k, max_iterations=30)
            except Exception:  # noqa: BLE001
                f = None
        if f is None or not np.isfinite(f[-1]):
            return None, f"the samples without block {b + 1} gave no estimate"
        estimates[b] = f[-1]
    return float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((estimates - estimates.mean()) ** 2))), ""


def solve_decorrelated(u_kn: np.ndarray, n_k: np.ndarray, initial_f_k=None, max_iterations: int = 1000,
                       overlap: bool = False):
    """(f_k, df_k relative to state 0) in kT from samples grouped by state.

    With ``overlap`` also the overlap matrix of these very samples (None if it cannot be
    computed): pymbar's ``O[i, j] = N_j * sum_n W_ni W_nj``. Its rows sum to 1; it is symmetric
    only if all windows hold the same number of samples.
    """
    mbar = _mbar(u_kn, n_k, initial_f_k, max_iterations)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = mbar.compute_free_energy_differences()
        matrix = None
        if overlap:
            try:
                matrix = np.asarray(mbar.compute_overlap()["matrix"], dtype=float)
            except Exception:  # noqa: BLE001
                matrix = None
    f, df = np.asarray(res["Delta_f"][0]), np.asarray(res["dDelta_f"][0])
    return (f, df, matrix) if overlap else (f, df)


def overlap_with_equal_weights(u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray) -> np.ndarray:
    """Overlap matrix of samples grouped by state, every window weighing the same whatever its
    number of samples.

    pymbar's matrix is ``O[i, j] = N_j * sum_n W_ni W_nj`` with the MBAR weights W. With the same
    number of samples in every window it is symmetric and measures how much the windows share
    their configurations. With different numbers it also measures the numbers: a window that
    keeps 30 samples next to one that keeps 3000 has O near 0.003 in one direction however well
    the two overlap. The numbers of samples have a check of their own, so the overlap is taken
    here in the normalisation of equal numbers:

        O[i, j] = sum_n w_n P_ni P_nj,   P_ni = exp(f_i - u_i(x_n)) / sum_k exp(f_k - u_k(x_n)),

    where w_n = 1 / N_s for a sample of window s (the samples of every window weigh the same in
    total) and f the free energies of the estimate. For equal numbers this is pymbar's matrix
    exactly. It is symmetric; its rows sum to 1 up to the sampling error.
    """
    from scipy.special import logsumexp

    n_k = np.asarray(n_k)
    reduced = np.minimum(np.asarray(u_kn, dtype=float) - np.min(u_kn), 1.0e6)
    log_p = np.asarray(f_k, dtype=float)[:, None] - reduced
    p = np.exp(log_p - logsumexp(log_p, axis=0))
    weights = np.repeat(1.0 / np.maximum(n_k, 1), n_k)
    return (p * weights) @ p.T


def overlap_matrix(u_kn: np.ndarray, n_k: np.ndarray) -> np.ndarray:
    """MBAR overlap matrix for reduced energies ``u_kn[state, sample]`` (samples grouped by state)."""
    mbar = _mbar(u_kn, n_k)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return np.asarray(mbar.compute_overlap()["matrix"])


def bar_between_neighbours(u_kn: np.ndarray, n_k: np.ndarray) -> tuple[float, float]:
    """Free energy of the last window relative to the first as the sum of BAR between neighbours.

    Returns (dG, standard error) in kT from samples grouped by state (``assemble`` or
    ``decorrelated``). BAR uses only the two windows of a pair, MBAR all windows at once; where
    the windows overlap well the two agree closely, so a difference is a sign of windows that
    are too far apart or too little sampled.
    """
    _pymbar()
    from pymbar import other_estimators

    u_kn = np.minimum(np.asarray(u_kn, dtype=float) - np.min(u_kn), 1.0e6)
    first = np.concatenate([[0], np.cumsum(n_k)])
    total, variance = 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            np.errstate(all="ignore"):
        for k in range(len(n_k) - 1):
            lower = u_kn[:, first[k]:first[k + 1]]             # samples of window k
            upper = u_kn[:, first[k + 1]:first[k + 2]]         # samples of window k + 1
            res = other_estimators.bar(lower[k + 1] - lower[k], upper[k] - upper[k + 1])
            total += float(res["Delta_f"])
            variance += float(res["dDelta_f"]) ** 2
    return total, float(np.sqrt(variance))


def _number(value) -> float | None:
    """``value`` as a finite float, or None for None, NaN, an infinity or something that is no number."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def quality_checks(analysis: dict, min_overlap: float = 0.03, min_effective_samples: float = 50.0,
                   halves_tolerance_kT: float = 1.0, estimators_tolerance_kcal: float = 1.0) -> dict:
    """What the analysis of a leg says about its reliability. ``analysis`` holds energies in kJ/mol.

    Every level is one that the literature gives; none of them is cuFEP's own.

    overlap      every pair of neighbouring lambda windows overlaps by at least ``min_overlap``.
                 This is the first condition: the variance of the estimate is a function of the
                 overlap matrix and the number of samples, and "values as low as 0.03 [seem] to be
                 tolerable to yield a reliable free energy estimate"; below it "the free energy
                 estimate itself will likely be substantially incorrect" (Klimovich, Shirts and
                 Mobley 2015, sections 3.4 and 3.5; also the warning level of Flare). Two matrices
                 are examined, both in the normalisation of equal numbers of samples per window:
                 that of all iterations, and that of the samples the free energy is estimated
                 from (``used``, :func:`overlap_with_equal_weights`). ``below`` names the one
                 that is under the level.
    samples      the window with the fewest independent samples has at least
                 ``min_effective_samples`` ("it is recommended that the effective number of
                 uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators are used", Mey et
                 al. 2020, section 8.2). Both the estimate (iterations / statistical inefficiency,
                 ``fewest``) and the number of samples MBAR really gets from that window
                 (``kept``) must reach it, and so must what the standard error leaves of them
                 (``independent``): where the standard error from time blocks is ``error_ratio``
                 times the one MBAR gives for the kept samples, the samples carry the
                 information of 1 / error_ratio^2 as many independent ones (the variance of an
                 estimate falls as one over their number). The windows are decorrelated one by
                 one; this is the correlation that is left between them. A ratio below 1 counts
                 as 1.
    halves       the free energies from the first and from the second half of the production
                 differ by at most ``halves_tolerance_kT`` ("differences over 1 kBT indicating
                 poor convergence", Mey et al. 2020, figure 9, where it is applied to each step of
                 a relative perturbation; here to the leg). A difference is an indication, not a
                 proof, and a converged estimate still fluctuates: the difference is also given
                 in units of its own standard error.
    estimators   BAR between neighbouring windows and MBAR differ by at most
                 ``estimators_tolerance_kcal`` (differences "larger than 1 kcal/mol ... would
                 indicate poor convergence", Mey et al. 2020, section 8.4; the literature compares
                 TI with BAR-type estimators, cuFEP records no dU/dlambda).

    ``ok`` is None where the analysis does not hold the numbers for a check; for the halves that
    includes a production too short to give each half an estimate, with the reason in ``reason``.
    A number that has no value is None, in ``analysis`` and in what is returned.
    """
    kt = R_GAS * float(analysis.get("temperature_K", 298.15))
    final, error = _number(analysis.get("delta_g")), _number(analysis.get("delta_g_error"))
    kept = analysis.get("window_kept_samples") or []
    # standard error from time blocks / standard error of MBAR (None: no blocks, or a result without)
    mbar_error = _number(analysis.get("delta_g_error_mbar"))
    ratio = error / mbar_error if analysis.get("delta_g_error_n_blocks") and error is not None and mbar_error else None

    neighbour = [_number(x) for x in analysis.get("overlap_neighbour", [])]
    lowest = _number(analysis["min_overlap"]) if "min_overlap" in analysis else 1.0
    all_ok = bool(lowest is not None and lowest >= min_overlap)
    pair = int(np.argmin([0.0 if x is None else x for x in neighbour])) if neighbour else None
    overlap = {"ok": all_ok, "lowest": lowest, "windows": None if pair is None else [pair, pair + 1],
               "threshold": min_overlap, "all_ok": all_ok, "used": None}
    used = analysis.get("overlap_used_neighbour")
    if used:
        values = [0.0 if _number(x) is None else _number(x) for x in used]
        i = int(np.argmin(values))
        overlap["used"] = {"ok": bool(values[i] >= min_overlap), "lowest": values[i], "windows": [i, i + 1],
                           "samples": [int(kept[i]), int(kept[i + 1])] if len(kept) > i + 1 else None}
        overlap["ok"] = bool(all_ok and overlap["used"]["ok"])
    overlap["below"] = ([] if all_ok else ["all iterations"]) \
        + (["the samples of the estimate"] if overlap["used"] and not overlap["used"]["ok"] else [])
    out = {"overlap": overlap}

    samples = {"ok": None, "threshold": float(min_effective_samples)}
    per_window = analysis.get("window_effective_samples")
    if per_window:
        counts = [min(float(e), float(kept[j])) if len(kept) == len(per_window) else float(e)
                  for j, e in enumerate(per_window)]
        j = int(np.argmin(counts))
        independent = counts[j] / max(1.0, ratio ** 2) if ratio is not None else counts[j]
        samples.update(ok=bool(independent >= min_effective_samples), fewest=float(per_window[j]), window=j,
                       statistical_inefficiency=float(analysis["window_statistical_inefficiency"][j]),
                       error_ratio=ratio, independent=float(independent))
        if len(kept) == len(per_window):
            samples["kept"] = int(kept[j])
        towards = analysis.get("window_inefficiency_neighbour") or []
        if len(towards) == len(per_window):
            samples["neighbour"] = int(towards[j])
    out["samples"] = samples

    halves = [analysis.get("first_half"), analysis.get("second_half")]
    tolerance = halves_tolerance_kT * kt
    conv = {"ok": None, "tolerance": tolerance, "tolerance_kT": float(halves_tolerance_kT), "final": final}
    if analysis.get("convergence_note"):
        conv["reason"] = str(analysis["convergence_note"])
    elif per_window and all(h is not None for h in halves):
        first, second = _number(halves[0][0]), _number(halves[1][0])
        conv.update(first_half=first, second_half=second)
        if first is not None and second is not None:
            difference = abs(first - second)
            errors = [_number(halves[0][1]), _number(halves[1][1])]
            sigma = None if None in errors else float(np.hypot(*errors))
            if sigma:
                in_sigma = difference / sigma
            elif sigma == 0.0 and difference == 0.0:
                in_sigma = 0.0                               # identical halves
            else:
                in_sigma = None
                conv["difference_sigma_note"] = ("the standard error of the difference is zero" if sigma == 0.0
                                                 else "a half has no standard error")
            conv.update(difference=difference, difference_kT=difference / kt, ok=bool(difference <= tolerance),
                        difference_sigma=in_sigma)
        elif final is not None:
            conv["ok"] = False       # half of the data does not determine the free energy
    out["halves"] = conv

    bar = _number(analysis.get("bar_delta_g"))
    tolerance = estimators_tolerance_kcal * KCAL
    est = {"ok": None, "tolerance": tolerance, "mbar": final}
    if per_window and bar is not None and final is not None:
        est.update(bar=bar, ok=bool(abs(bar - final) <= tolerance))
    elif analysis.get("bar_note"):
        est["reason"] = str(analysis["bar_note"])
    out["estimators"] = est
    out["uncertainty"] = error
    return out


def statistical_inefficiency(u: np.ndarray, states: np.ndarray) -> float:
    """g of the replica-summed reduced potential at the sampled states (in iterations)."""
    _, timeseries = _pymbar()
    n_iter, n_rep, _ = u.shape
    series = u[np.arange(n_iter)[:, None], np.arange(n_rep)[None, :], states].sum(axis=1)
    if n_iter < 8 or np.allclose(series, series[0]):
        return 1.0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            g = float(timeseries.statistical_inefficiency(series))
        except Exception:  # noqa: BLE001
            g = 1.0
    return max(1.0, g)


def count_round_trips(states: np.ndarray) -> int:
    """Total number of complete end-to-end-and-back passages over all replicas."""
    n_states = int(states.max()) + 1
    trips = 0
    for r in range(states.shape[1]):
        traj = states[:, r]
        ends = traj[(traj == 0) | (traj == n_states - 1)]
        if len(ends) < 2:
            continue
        changes = int(np.sum(ends[1:] != ends[:-1]))
        trips += changes // 2
    return trips


def neighbour_swap_fraction(states: np.ndarray) -> list[float]:
    """Fraction of iterations after which a replica left state k for state k+1 (or back)."""
    n_states = int(states.max()) + 1
    moved = np.zeros(n_states - 1)
    a, b = states[:-1], states[1:]
    for k in range(n_states - 1):
        moved[k] = np.mean(np.any((a == k) & (b == k + 1), axis=1) | np.any((a == k + 1) & (b == k), axis=1))
    return moved.tolist()


def stage_boundaries(state_dicts: list[dict]) -> dict[str, tuple[int, int]]:
    """Index ranges (first, last window) of the restrain, discharge and vanish stages."""
    lr = [s.get("lambda_restraints", 1.0) for s in state_dicts]
    le = [s.get("lambda_electrostatics", 1.0) for s in state_dicts]
    ls = [s.get("lambda_sterics", 1.0) for s in state_dicts]
    n = len(state_dicts)
    out: dict[str, tuple[int, int]] = {}
    i_r = max((i for i in range(n) if le[i] == 1.0 and ls[i] == 1.0), default=0)
    i_e = max((i for i in range(n) if ls[i] == 1.0), default=i_r)
    if i_r > 0 and lr[0] < 1.0:
        out["restrain"] = (0, i_r)
    if i_e > i_r:
        out["discharge"] = (i_r if "restrain" in out else 0, i_e)
    if n - 1 > i_e:
        out["vanish"] = (i_e, n - 1)
    return out


def analyze_leg(samples: dict, temperature_K: float, n_equilibration: int | None = None,
                n_fractions: int = 10, n_blocks: int = 5, error_blocks=BLOCK_COUNTS,
                halves_error_blocks: int = HALF_BLOCK_COUNT) -> LegAnalysis:
    """Analyse samples loaded with :func:`cufep.sampling.storage.read_samples`.

    The free energy comes from samples that are decorrelated window by window
    (:func:`neighbour_inefficiencies`, :func:`decorrelated`), as Klimovich, Shirts and Mobley
    2015 and alchemlyb do. Its standard error comes from time blocks common to all windows
    (:func:`block_standard_error`, for each number of ``error_blocks``; the largest is reported),
    and so do the errors of the two halves (``halves_error_blocks`` blocks each). The points of
    the forward and reverse series and the consecutive blocks carry the error MBAR gives for
    their samples.
    ``n_fractions`` is the number of points of the forward and reverse
    series, ``n_blocks`` the number of consecutive blocks with an estimate of their own. A part
    of the production gets an estimate only if every window holds ``MIN_INDEPENDENT_SAMPLES``
    independent samples in it; a point or block that does not is left out, and halves that do
    not are reported with the reason.
    """
    from cufep.alchemy.lrc import tail_free_energy

    kt = R_GAS * temperature_K
    attrs = samples.get("attrs", {})
    if n_equilibration is None:
        n_equilibration = int(attrs.get("n_equilibration_iterations", 0))
    stored = int(samples["u"].shape[0])
    u = samples["u"][n_equilibration:]
    states = samples["states"][n_equilibration:]
    if u.shape[0] < 4:
        planned = attrs.get("n_planned_iterations")
        raise NoProduction(
            f"the samples hold {stored} iteration(s), of which the first {n_equilibration} are equilibration"
            + ("" if planned is None else f" (the run was planned for {int(planned)})")
            + f": {max(0, stored - n_equilibration)} iteration(s) of production, and 4 are the least for an analysis")
    n, n_states = u.shape[0], u.shape[2]
    # the overlap matrix: equal numbers of samples per window, so that its elements mean the
    # same for every window (statistical inefficiency of the summed reduced potential)
    g_sum = statistical_inefficiency(u, states)
    stride = max(1, int(np.ceil(g_sum)))
    f_start, _df, overlap = solve_mbar(u[::stride], states[::stride])
    neighbour = [float(overlap[i, i + 1]) for i in range(n_states - 1)]
    # the free energy and its uncertainty: every window decorrelated on its own, by the slower of
    # its energy differences to the two neighbours
    table = neighbour_inefficiencies(u, states)
    g = np.array([np.nanmax(row) if np.isfinite(row).any() else 1.0 for row in table])
    towards = [1 if np.isfinite(row[1]) and not row[0] > row[1] else -1 for row in table]
    u_kn, n_k = decorrelated(u, states, g)
    f_k, df_k, by_number = solve_decorrelated(u_kn, n_k, initial_f_k=f_start, overlap=True)
    # the overlap among these very samples, in the normalisation of equal numbers per window
    try:
        used = overlap_with_equal_weights(u_kn, n_k, f_k)
    except Exception:  # noqa: BLE001
        used = None
    if used is not None and not np.all(np.isfinite(used)):
        used = None
    used_neighbour = [] if used is None else [float(used[i, i + 1]) for i in range(n_states - 1)]
    effective = n / g
    slowest = int(np.argmin(np.minimum(effective, n_k)))
    # the standard error: time blocks common to all windows, for each number of blocks
    longest = int(np.ceil(g.max()))
    lowest = float(np.min(u_kn))                               # as pymbar gets them, see _mbar
    reduced = np.minimum(u_kn - lowest, 1.0e6)
    iteration = kept_sample_iterations(n, g)
    block_errors: dict[str, float | None] = {}
    block_notes: dict[str, str] = {}
    for count in dict.fromkeys(int(count) for count in error_blocks):
        value, why = block_standard_error(reduced, n_k, iteration, 0, n, count, f_k, longest)
        block_errors[str(count)] = None if value is None else float(value * kt)
        if why:
            block_notes[str(count)] = why
    mbar_error = float(df_k[-1] * kt)
    usable = {count: value for count, value in block_errors.items() if value is not None}
    if usable:
        n_blocks_used = int(max(usable, key=usable.get))
        error, error_note = usable[str(n_blocks_used)], ""
    else:
        n_blocks_used, error = None, mbar_error
        error_note = ("no number of time blocks could be used: "
                      + "; ".join(f"{count} blocks: {why}" for count, why in block_notes.items()))
    half_blocks: dict[tuple[int, int], int] = {}

    def partial(a: int, b: int, n_time_blocks: int = 0):
        """(dG, error) of the iterations [a, b), or None if a window has too few independent samples there.

        The error is MBAR's; with ``n_time_blocks`` it is that of so many time blocks within
        the part where they can be used (``half_blocks`` then holds the number for the part).
        """
        part_kn, part_n = decorrelated(u, states, g, a, b)
        if part_n.min() < MIN_INDEPENDENT_SAMPLES:
            return None
        try:
            # starts from the solution of all data and then converges in a few iterations; a
            # part that needs more has a window without overlap and no defined answer
            f, df = solve_decorrelated(part_kn, part_n, initial_f_k=f_k, max_iterations=30)
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")
        part_error = float(df[-1])
        if n_time_blocks and np.isfinite(f[-1]):
            value, _why = block_standard_error(np.minimum(part_kn - lowest, 1.0e6), part_n,
                                               kept_sample_iterations(n, g, a, b), a, b, n_time_blocks, f, longest)
            if value is not None:
                part_error = value
                half_blocks[(a, b)] = n_time_blocks
        return float(f[-1] * kt), float(part_error * kt)

    forward, reverse = [], []
    everything = (float(f_k[-1] * kt), float(df_k[-1] * kt))
    for i in range(1, n_fractions + 1):
        m = int(round(i * n / n_fractions))
        fwd, rev = (everything, everything) if m >= n else (partial(0, m), partial(n - m, n))
        if fwd is None or rev is None:
            continue
        # the fraction that is really used is the one that is stored and plotted
        forward.append((m / n, *fwd))
        reverse.append((m / n, *rev))
    # the halves: split in the middle of the time series, no iteration in both
    mid = n // 2
    first_half, second_half = partial(0, mid, int(halves_error_blocks)), partial(mid, n, int(halves_error_blocks))
    convergence_note = ""
    if first_half is None or second_half is None:
        first_half = second_half = (float("nan"), float("nan"))
        half_blocks.clear()
        in_half = min(len(kept_iterations(0, mid, g[slowest])), len(kept_iterations(mid, n, g[slowest])))
        convergence_note = (f"lambda window {slowest} holds {in_half} independent sample(s) in a half of the production "
                            f"(statistical inefficiency {g[slowest]:.0f} of {n} iterations); "
                            f"{MIN_INDEPENDENT_SAMPLES} are needed for an estimate")
    gap = abs(first_half[0] - second_half[0])
    blocks = []
    edges = [int(round(k * n / n_blocks)) for k in range(n_blocks + 1)] if n_blocks >= 2 else []
    parts = [partial(a, b) for a, b in zip(edges[:-1], edges[1:])]
    if parts and all(part is not None for part in parts):
        blocks = [(a / n, b / n, *part) for a, b, part in zip(edges[:-1], edges[1:], parts)]
    bar_note = ""
    try:
        bar, bar_error = bar_between_neighbours(u_kn, n_k)
        bar, bar_error = bar * kt, bar_error * kt
    except Exception as exc:  # noqa: BLE001  a pair of windows without overlap has no BAR estimate
        bar = bar_error = float("nan")
        bar_note = f"BAR gave no estimate for a pair of neighbouring windows ({type(exc).__name__})"
    if not bar_note and not np.isfinite(bar):
        bar_note = "BAR gave no finite estimate for a pair of neighbouring windows"

    stages = {}
    state_dicts = attrs.get("states", [])
    if isinstance(state_dicts, list) and len(state_dicts) == n_states:
        for name, (a, b) in stage_boundaries(state_dicts).items():
            stages[name] = float((f_k[b] - f_k[a]) * kt)

    lrc_dg = lrc_se = 0.0
    lrc_n = 0
    if "lrc_du" in samples and len(samples["lrc_du"]):
        mask = samples["lrc_iter"] >= n_equilibration
        du = samples["lrc_du"][mask]
        lrc_n = int(du.size)
        lrc_dg, lrc_se = tail_free_energy(du, temperature_K)

    lrc_iso = 0.0
    if "lrc_s6" in attrs:
        rc = float(attrs.get("lrc_cutoff_nm", 0.9))
        coupled = states == 0
        volume = float(samples["volume"][n_equilibration:][coupled].mean()) if coupled.any() else float("nan")
        if np.isfinite(volume) and volume > 0:
            lrc_iso = float((16.0 * np.pi / volume) * (float(attrs["lrc_s12"]) / (9.0 * rc ** 9)
                                                       - float(attrs["lrc_s6"]) / (3.0 * rc ** 3)))

    return LegAnalysis(
        n_states=n_states, n_iterations_used=int(n), statistical_inefficiency=float(g_sum),
        effective_samples_per_state=float(effective[slowest]), delta_g=float(f_k[-1] * kt),
        delta_g_error=float(error), delta_g_error_mbar=mbar_error, delta_g_error_blocks=block_errors,
        delta_g_error_blocks_note=block_notes, delta_g_error_n_blocks=n_blocks_used, delta_g_error_note=error_note,
        halves_error_blocks=[half_blocks.get((0, mid)), half_blocks.get((mid, n))],
        window_statistical_inefficiency=[float(x) for x in g],
        window_inefficiency_previous=[float(x) for x in table[:, 0]],
        window_inefficiency_next=[float(x) for x in table[:, 1]], window_inefficiency_neighbour=towards,
        window_effective_samples=[float(x) for x in effective], window_kept_samples=[int(x) for x in n_k],
        slowest_window=slowest,
        f_k=(f_k * kt).tolist(), df_k=(df_k * kt).tolist(), stage_delta_g=stages,
        overlap_neighbour=neighbour, min_overlap=float(min(neighbour)) if neighbour else 1.0,
        round_trips=count_round_trips(states), swap_acceptance_neighbour=neighbour_swap_fraction(states),
        overlap_matrix=np.round(overlap, 4).tolist(),
        overlap_used_matrix=[] if used is None else np.round(used, 4).tolist(),
        overlap_used_weighted_by_number=[] if by_number is None else np.round(by_number, 4).tolist(),
        overlap_used_neighbour=used_neighbour,
        min_overlap_used=float(min(used_neighbour)) if used_neighbour else float("nan"),
        forward=forward, reverse=reverse, first_half=first_half, second_half=second_half,
        convergence_note=convergence_note, forward_reverse_gap=float(gap), blocks=blocks,
        bar_delta_g=float(bar), bar_delta_g_error=float(bar_error), bar_note=bar_note,
        lrc_delta_g=lrc_dg, lrc_error=lrc_se, lrc_samples=lrc_n, lrc_isotropic=lrc_iso,
        temperature_K=temperature_K)
