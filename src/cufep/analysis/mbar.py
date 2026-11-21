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
#:   5  the standard error, the halves and every other part of the production are estimated for
#:      the free energy that is reported (MBAR minus the tail correction of the same iterations);
#:      a part counts only if its estimate is determined; the reported standard error is never
#:      smaller than the analytical one
#:   6  the analytical error includes the correlation of MBAR and the correction (from the
#:      jackknife, else their sum as an upper bound); a part is determined if its samples fix the
#:      solution of its MBAR equations (windows_determined), and a determined part below
#:      analysis.min_overlap is used with a warning; blocks without an estimate leave the others.
#:      The numbers of a fixed input are pinned by a test, so that a change of them cannot keep
#:      the version
#:   7  the whole production needs as many samples in every window as any part of it
#:      (too_few_samples); the standard error of the sum of BAR between neighbouring windows comes
#:      from the same time blocks as that of MBAR (neighbouring pairs share a window, so the errors
#:      of the pairs are not independent)
#:   8  the isotropic tail correction is -kT ln < exp(-C / (kT V)) > over the volumes of the coupled
#:      state, not C / <V>; the acceptance of exchange between neighbouring windows is accepted /
#:      proposed swaps where the samples record them (the measure reported before is kept as
#:      neighbour_transition_fraction)
#:   9  the verdicts on a leg are the two checks with the literature's levels only (overlap, uncorrelated
#:      samples); the comparison of the two halves and of BAR with MBAR, and the reduction of the
#:      samples by the ratio of two standard errors, are no longer checks (owner, 2026-11-20). The
#:      free energies and errors are those of version 8
#:  10  no other verdict either: the notes on the bilayer and on the pose of the ligand, with levels of
#:      cuFEP's own, are gone; how the bilayer changed and how far the ligand was from its input pose
#:      are numbers of the result. The record of the analysis holds the analysis section only
ANALYSIS_VERSION = 10
#: numbers of time blocks the standard error of a leg is computed with; the largest of the errors
#: they give is the one reported (:meth:`TimeRangeEstimator.part`). The numbers are cuFEP's choice:
#: the literature asks for blocks longer than the correlation time and for a standard error that
#: no longer grows with the length of the blocks (Grossfield et al. 2018, section 7.3.2), and
#: gives no numbers. Defaults of ``analysis.error_blocks`` and ``analysis.part_error_blocks``.
BLOCK_COUNTS = (40, 20, 10, 5)
#: the relative tolerance to which :func:`free_energies_from` solves the MBAR equations
SOLVER_TOLERANCE = 1.0e-8
#: a part is determined if the tolerance of the solver leaves its free energies uncertain by at
#: most this fraction of their statistical standard error and by at most RESOLUTION_KT
#: (:func:`windows_determined`)
NEGLIGIBLE_FRACTION = 0.01
#: kT; finer than the 0.01 kcal/mol (0.017 kT at 298 K) to which free energies are reported
RESOLUTION_KT = 0.01
#: number of time blocks for the standard error of a part of the production (a half, a point
#: of the forward and reverse series, a consecutive block); default of analysis.part_error_blocks
PART_BLOCK_COUNT = 5
#: a part of the production gets a free energy estimate of its own only if it holds at least this
#: many statistically independent samples per lambda window
MIN_INDEPENDENT_SAMPLES = 4


class NoProduction(ValueError):
    """The samples hold no production to analyse (yet)."""


class NoCorrection(ValueError):
    """The tail correction that was asked for cannot be estimated from these samples."""


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
    #: the free energy of decoupling that is reported, kJ/mol: that of the last state relative to
    #: the first by MBAR, minus the tail correction (correction), both from the same iterations;
    #: and its standard error (see delta_g_error_from)
    delta_g: float
    delta_g_error: float
    #: the MBAR part alone (samples decorrelated window by window) and the standard error MBAR
    #: gives for it, which takes the samples as independent
    delta_g_mbar: float = float("nan")
    delta_g_mbar_error: float = float("nan")
    #: the tail correction of the coupled state (correction_method: reweight, isotropic or none),
    #: its standard error for independent samples, and the number of observations it rests on
    correction_method: str = "none"
    correction: float = 0.0
    correction_error: float = 0.0
    correction_samples: int = 0
    #: for the correction by reweighting: the effective number of its observations (1 / sum w^2 of
    #: the normalised exponential weights) and the largest weight; a correction that rests on a few
    #: configurations shows here. No level is applied
    correction_effective_samples: float | None = None
    correction_largest_weight: float | None = None
    #: the correlation of the errors of MBAR and of the correction, from the jackknife replicates of
    #: the most time blocks that can be used (None where none can be used or there is no correction)
    correction_correlation: float | None = None
    #: the standard error of delta_g by the delete-one-block jackknife over time blocks common to
    #: all windows, for every number of blocks that was asked for (key: the number as text; None
    #: where that number cannot be used, with the reason in delta_g_error_blocks_note), and the
    #: same for the MBAR part alone
    delta_g_error_blocks: dict[str, float | None] = field(default_factory=dict)
    delta_g_mbar_error_blocks: dict[str, float | None] = field(default_factory=dict)
    delta_g_error_blocks_note: dict[str, str] = field(default_factory=dict)
    #: where delta_g_error comes from: "time blocks" (the largest of delta_g_error_blocks, from
    #: delta_g_error_n_blocks blocks), "analytical" (the standard errors MBAR and the correction
    #: give for independent samples, combined with their correlation from the jackknife:
    #: correction_correlation), or "analytical upper bound" (their sum, where no number of blocks
    #: can be used and the correlation is not known; delta_g_error_note says why none can)
    delta_g_error_from: str = ""
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
    #: acceptance of exchange between the replicas in neighbouring windows over the production:
    #: accepted / proposed swaps per pair; None where the samples do not record them (earlier
    #: versions) or nothing was proposed
    swap_acceptance_neighbour: list[float | None] | None = None
    #: fraction of production iterations after which a replica had moved from window k to k + 1 or
    #: back (what earlier versions reported as the acceptance; it misses moves undone within one
    #: iteration of mixing)
    neighbour_transition_fraction: list[float] = field(default_factory=list)
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
    #: delta_g from growing parts of the production, counted from its start (forward) and from its
    #: end (reverse): (fraction of the production, dG, error). Every part is estimated as the
    #: whole is (MBAR minus the correction of its own iterations, its error from time blocks
    #: within it or the analytical one, whichever is larger)
    forward: list[tuple[float, float, float]] = field(default_factory=list)
    reverse: list[tuple[float, float, float]] = field(default_factory=list)
    #: (dG, error) of delta_g from the first half and from the second half of the production
    #: alone; the halves do not share an iteration. halves_error_blocks holds the number of time
    #: blocks behind each of the two errors (None: the analytical one)
    first_half: tuple[float, float] = (float("nan"), float("nan"))
    second_half: tuple[float, float] = (float("nan"), float("nan"))
    halves_error_blocks: list[int | None] = field(default_factory=list)
    #: why the free energy of the whole production is not determined (its samples do not connect
    #: the lambda windows, :func:`windows_determined`); delta_g and its errors are then NaN
    not_determined: str = ""
    #: why the halves have no estimate (too few samples, windows that the samples of a half do
    #: not connect, no tail samples in a half), if they have none
    convergence_note: str = ""
    #: the halves have estimates, but the windows of a half overlap below analysis.min_overlap:
    #: which half, and by how much (their comparison rests on too little overlap)
    halves_warning: str = ""
    #: difference between the two halves
    forward_reverse_gap: float = 0.0
    #: delta_g from consecutive blocks of the production that do not overlap:
    #: (start, end as fractions of the production, dG, error)
    blocks: list[tuple[float, float, float, float]] = field(default_factory=list)
    #: the points of the forward and reverse series and the consecutive blocks that have no
    #: estimate, and why (the free energy of their samples is not determined)
    parts_not_determined: list[str] = field(default_factory=list)
    #: the halves, points and blocks that have an estimate although their neighbouring windows
    #: overlap below analysis.min_overlap, and by how much (shown, with this warning)
    parts_below_level: list[str] = field(default_factory=list)
    #: sum of BAR between neighbouring windows, a second estimator on the same data
    bar_delta_g: float = float("nan")
    #: its standard error from the same time blocks as that of MBAR, and the number of blocks; None
    #: and why where no number of blocks can be used
    bar_delta_g_error: float = float("nan")
    bar_delta_g_error_n_blocks: int | None = None
    bar_delta_g_error_note: str = ""
    #: why BAR has no estimate, if it has none
    bar_note: str = ""
    #: the tail correction of the whole production by both methods, where the samples hold what
    #: it needs (None otherwise): by reweighting the recorded tail energies, and in the
    #: homogeneous (isotropic) approximation that OpenMM's long-range correction and OpenFE
    #: use. For comparison; the one that is used is ``correction``
    lrc_reweight: float | None = None
    lrc_isotropic: float | None = None
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
        matrix = overlap_of(mbar) if overlap else None
    return np.asarray(res["Delta_f"][0]), np.asarray(res["dDelta_f"][0]), matrix


def overlap_of(mbar) -> np.ndarray:
    """pymbar's overlap matrix, O[i, j] = N_j sum_n W_ni W_nj (``MBAR.compute_overlap``), without the
    eigenvalues that method also computes: ``numpy.linalg.eig`` of that matrix does not converge on
    some platforms for a degenerate one (34 windows of one Hamiltonian failed in CI), and only the
    matrix is used."""
    w = np.asarray(mbar.W_nk, dtype=float)
    return (w.T @ w) * np.asarray(mbar.N_k, dtype=float)[None, :]


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


def free_energies_from(u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray, tolerance: float = SOLVER_TOLERANCE,
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


@dataclass
class TailSeries:
    """What the tail correction of a selection of the production is computed from.

    ``times`` is the iteration of every observation counted from the start of the production;
    ``values`` are the recorded tail energy differences of the coupled state (kJ/mol, method
    reweight) or the box volumes of the replica in the coupled state (nm^3, method isotropic).
    The method none has no observations and a correction of zero.
    """
    method: str
    times: np.ndarray
    values: np.ndarray
    temperature_K: float
    s6: float = 0.0
    s12: float = 0.0
    cutoff_nm: float = 0.0

    def estimate(self, select) -> tuple[float | None, float | None, int, str]:
        """(correction in kJ/mol, its standard error for independent observations, number of
        observations, why there is none) of the observations ``select`` picks."""
        if self.method == "none":
            return 0.0, 0.0, 0, ""
        values = self.values[select]
        what = "tail energies" if self.method == "reweight" else "volumes of the coupled state"
        bad = int(np.sum(~np.isfinite(values))) if self.method == "reweight" else \
            int(np.sum(~(np.isfinite(values) & (values > 0))))
        if bad:
            kind = "not finite" if self.method == "reweight" else "not finite and positive"
            return None, None, int(values.size), (f"{bad} of the {values.size} {what} in these iterations are {kind}; "
                                                   "the samples are damaged")
        if values.size < MIN_INDEPENDENT_SAMPLES:
            return None, None, int(values.size), (f"{values.size} {what} were recorded in these iterations; "
                                                   f"{MIN_INDEPENDENT_SAMPLES} are needed for the tail correction")
        if self.method == "reweight":
            from cufep.alchemy.lrc import tail_free_energy

            value, error = tail_free_energy(values, self.temperature_K)
            return float(value), float(error), int(values.size), ""
        # the homogeneous tail is C / V in every configuration; in NPT the free energy of adding it is
        # -kT ln < exp(-C / (kT V)) >, the same estimate as for the recorded tail energies, not C / <V>
        # (review of PR 1: 4 volumes 90, 90, 110, 110 nm^3 gave -20.106 instead of -21.062 kJ/mol)
        from cufep.alchemy.lrc import tail_free_energy

        value, error = tail_free_energy(self.isotropic_energies(values), self.temperature_K)
        return float(value), float(error), int(values.size), ""

    def isotropic_energies(self, volumes: np.ndarray) -> np.ndarray:
        """The homogeneous dispersion tail of every configuration, kJ/mol, from its box volume."""
        c = 16.0 * np.pi * (self.s12 / (9.0 * self.cutoff_nm ** 9) - self.s6 / (3.0 * self.cutoff_nm ** 3))
        return c / np.asarray(volumes, dtype=float)

    def weights(self, select) -> tuple[float | None, float | None]:
        """For a correction by exponential averaging (reweight, isotropic): the effective number of its
        observations, 1 / sum w^2 with the normalised exponential weights w (the effective sample
        number of pymbar's ``compute_effective_sample_number``), and the largest weight. None for
        the method none."""
        if self.method == "none":
            return None, None
        energies = np.asarray(self.values[select], dtype=float)
        if self.method == "isotropic":
            energies = self.isotropic_energies(energies)
        x = -energies / (R_GAS * self.temperature_K)
        if not x.size or not np.all(np.isfinite(x)):
            return None, None
        w = np.exp(x - x.max())
        w /= w.sum()
        return float(1.0 / np.sum(w ** 2)), float(w.max())


def tail_series(samples: dict, method: str, n_equilibration: int, n: int, temperature_K: float) -> TailSeries:
    """The observations for the tail correction ``method`` in the production [n_equilibration,
    n_equilibration + n) of ``samples``. Raises NoCorrection if the samples hold none of them."""
    attrs = samples.get("attrs", {})
    empty = np.zeros(0)
    if method == "none":
        return TailSeries("none", empty, empty, temperature_K)
    if method == "reweight":
        iterations = np.asarray(samples.get("lrc_iter", empty), dtype=int)
        values = np.asarray(samples.get("lrc_du", empty), dtype=float)
        if values.size == 0:
            raise NoCorrection("the samples hold no tail energies (they were made with analysis.lrc: none); the "
                               "correction by reweighting needs them")
        times = iterations - int(n_equilibration)
        inside = (times >= 0) & (times < n)
        return TailSeries("reweight", times[inside], values[inside], temperature_K)
    if method == "isotropic":
        if not all(key in attrs for key in ("lrc_s6", "lrc_s12", "lrc_cutoff_nm")):
            raise NoCorrection(
                "the samples hold no Lennard-Jones sums of the ligand (they were made with analysis.lrc: "
                "none); the isotropic correction needs them"
            )
        states = np.asarray(samples["states"][n_equilibration:n_equilibration + n])
        coupled = np.argsort(states, axis=1)[:, 0]                    # the replica in state 0
        volumes = np.asarray(samples["volume"][n_equilibration:n_equilibration + n], dtype=float)
        return TailSeries("isotropic", np.arange(n), volumes[np.arange(n), coupled], temperature_K,
                          float(attrs["lrc_s6"]), float(attrs["lrc_s12"]), float(attrs["lrc_cutoff_nm"]))
    raise ValueError(f"unknown tail correction {method!r}")


@dataclass
class Determinacy:
    """Whether the samples of a part determine its free energies (:func:`windows_determined`)."""
    determined: bool
    #: the smallest eigenvalue of the Hessian of the MBAR equations with window 0 fixed
    smallest: float
    #: how far from the solution the solver may stop along the least determined combination of
    #: free energies (kT), and the standard error of that combination (kT); inf if not fixed at all
    numerical: float
    statistical: float
    #: the two groups of windows that the least determined combination moves against each other
    groups: list[list[int]]


def windows_determined(reduced_u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray) -> Determinacy:
    """Whether the samples determine the free energies ``f_k`` (kT) that solve their MBAR equations.

    The Newton step of the solver (:func:`free_energies_from`) uses the Hessian of the MBAR
    equations, H = diag(N) - P P^T with P_in the probability that sample n is of state i. It is
    the Laplacian of a graph whose edge between windows i and j weighs sum_n P_in P_jn, the
    samples the two share. The free energies (relative to window 0) are unique if and only if this
    graph is connected, that is if H with window 0 fixed has no zero eigenvalue; without samples
    that connect two groups of windows the MBAR equations hold for any difference of their free
    energies, and a solver started from a value returns it. With a smallest eigenvalue lambda
    the solver, which stops where every |gradient_i| / N_i is below its tolerance, may stop up
    to tolerance * |N| / lambda away from the solution, while the statistical standard error of
    that combination of free energies is about 1 / sqrt(lambda) (the Hessian is the information
    of the equations; Shirts and Chodera 2008). The part is determined if the first is at most
    ``NEGLIGIBLE_FRACTION`` of the second and at most ``RESOLUTION_KT``: the samples, not the
    start of the solver, fix the free energies to the precision they are reported with (review
    of PR 1: with the relative bound alone, windows that shared 1e-8 of a sample had a standard
    error of thousands of kT and a solver that could stop 30 kT away; its halves were the start
    of the solver, the whole, and agreed). A connected part of low precision is determined,
    however long its chain of windows (review of PR 1: a chain of 24 windows that overlap by
    0.25 was called unconnected, and 34 windows of one Hamiltonian, each pair sharing 1/34,
    likewise); its standard error is large and its overlap is a check of its own.
    """
    from scipy.special import logsumexp

    n_k = np.asarray(n_k, dtype=float)
    k = len(n_k)
    if k < 2:
        return Determinacy(True, float("inf"), 0.0, 0.0, [list(range(k))])
    with np.errstate(all="ignore"):
        log_p = (np.log(n_k) + np.asarray(f_k, dtype=float))[:, None] - np.asarray(reduced_u_kn, dtype=float)
        p = np.exp(log_p - logsumexp(log_p, axis=0))
    hessian = np.diag(p.sum(axis=1)) - p @ p.T
    if not np.all(np.isfinite(hessian)):
        return Determinacy(False, float("nan"), float("inf"), float("inf"), [list(range(k))])
    values, vectors = np.linalg.eigh(hessian[1:, 1:])
    smallest = float(values[0])
    if smallest > 0.0:
        numerical = SOLVER_TOLERANCE * float(np.linalg.norm(n_k)) / smallest
        statistical = 1.0 / np.sqrt(smallest)
    else:
        numerical = statistical = float("inf")
    determined = bool(smallest > 0.0 and numerical <= min(NEGLIGIBLE_FRACTION * statistical, RESOLUTION_KT))
    # the windows that the least determined combination moves together, against the others
    v = np.concatenate([[0.0], vectors[:, 0]])
    middle = (v.max() + v.min()) / 2.0
    side = v > middle
    groups = sorted(([int(i) for i in np.flatnonzero(side == s)] for s in (False, True) if np.any(side == s)),
                    key=lambda g: g[0])
    return Determinacy(determined, smallest, float(numerical), float(statistical), groups)


def not_determined_reason(reduced_u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray) -> str:
    """Why the samples do not determine the free energies ``f_k`` (:func:`windows_determined`), or
    "" if they do. The one test of every part of the production and of the whole of it."""
    try:
        fixed = windows_determined(reduced_u_kn, n_k, f_k)
    except Exception:  # noqa: BLE001  a matrix that cannot be decomposed determines nothing
        fixed = None
    if fixed is not None and fixed.determined:
        return ""
    if fixed is None:
        why = "whether they do could not be computed"
    elif not np.isfinite(fixed.numerical):
        why = "the samples these groups share do not fix the difference of their free energies"
    else:
        why = (f"the samples these groups share fix the difference of their free energies only to within "
               f"{fixed.numerical:.2g} kT (standard error {fixed.statistical:.2g} kT), more than "
               f"{min(NEGLIGIBLE_FRACTION * fixed.statistical, RESOLUTION_KT):.2g} kT")
    shown = "" if fixed is None else f" ({_windows_shown(fixed.groups)})"
    return f"the samples do not connect the lambda windows{shown}: {why}, so it is not determined"


def _windows_shown(groups: list[list[int]]) -> str:
    return "; ".join(
        f"windows {g[0]}-{g[-1]}" if g == list(range(g[0], g[-1] + 1)) else "windows " + ", ".join(map(str, g))
        for g in groups
    )


@dataclass
class PartEstimate:
    """The free energy of a selection of the samples of a leg, or why it is not used (``reason``;
    ``kind`` "not determined": too few samples, no solution, windows not connected, no correction).
    A determined part whose neighbouring windows overlap below analysis.min_overlap is used, and
    ``warning`` says so."""
    reason: str = ""
    kind: str = ""
    warning: str = ""
    f: np.ndarray | None = None            # MBAR free energies, kT, relative to the first state
    mbar: float | None = None              # kJ/mol
    mbar_error: float | None = None        # analytical, kJ/mol (None where it was not computed)
    correction: float | None = None        # kJ/mol
    correction_error: float | None = None
    correction_samples: int = 0
    value: float | None = None             # mbar - correction
    lowest_overlap: float | None = None

    @property
    def determined(self) -> bool:
        return not self.reason


class TimeRangeEstimator:
    """The free energy of decoupling of a leg from any selection of the iterations of its
    production: MBAR on the samples of those iterations minus the tail correction of the same
    iterations. The whole production, every replicate of the jackknife, the halves, the points of
    the forward and reverse series and the consecutive blocks are all estimated here, so that an
    error or a check is always one of the quantity that is reported.

    A selection is used only if its free energy is determined: every window keeps at least
    ``MIN_INDEPENDENT_SAMPLES`` samples in it, the MBAR equations converge, the samples determine
    their solution (:func:`windows_determined`: windows that the samples do not connect have no
    determined difference of free energies, and a solver started from a value returns it), and
    the tail correction has its observations in it. A determined selection whose neighbouring
    windows overlap among its samples by less than ``min_overlap`` (the level of Klimovich,
    Shirts and Mobley 2015, section 3.5, below which an estimate "will likely be substantially
    incorrect") is used, with a warning: its estimate and its error are shown, and the warning
    says that they rest on too little overlap (review of PR 1: a connected part of low precision
    is shown with its error and a warning, a part that is not determined is not).
    """

    def __init__(self, u_kn: np.ndarray, n_k: np.ndarray, iteration: np.ndarray, tail: TailSeries, kt: float,
                 min_overlap: float, longest_interval: int, f_whole: np.ndarray) -> None:
        self.u_kn = np.asarray(u_kn, dtype=float)
        # as pymbar gets them (see _mbar): shifted and capped, the same for every selection
        self.reduced = np.minimum(self.u_kn - np.min(self.u_kn), 1.0e6)
        self.n_k = np.asarray(n_k)
        self.window = np.repeat(np.arange(len(self.n_k)), self.n_k)
        self.iteration = np.asarray(iteration)
        self.tail, self.kt, self.min_overlap = tail, float(kt), float(min_overlap)
        self.longest = max(1, int(longest_interval))
        self.f_whole = np.asarray(f_whole, dtype=float)

    def estimate(self, columns: np.ndarray, observations: np.ndarray, f_start: np.ndarray,
                 analytic: bool = False) -> PartEstimate:
        """The estimate from the samples ``columns`` and the tail observations ``observations``
        (boolean masks), solved from ``f_start``; with ``analytic`` also MBAR's standard error."""
        k = len(self.n_k)
        counts = np.bincount(self.window[columns], minlength=k)
        few = too_few_samples(counts)
        if few:
            return PartEstimate(reason=few, kind="not determined")
        f = free_energies_from(self.reduced[:, columns], counts, f_start)
        df = None
        if f is None or analytic:
            try:
                solved, df = solve_decorrelated(self.u_kn[:, columns], counts, max_iterations=30,
                                                initial_f_k=f_start if f is None else f)
                f = solved if f is None else f
            except Exception:  # noqa: BLE001  no solution of the MBAR equations
                df = None
        if f is None or not np.all(np.isfinite(f)):
            return PartEstimate(reason="the MBAR equations of these samples have no solution", kind="not determined")
        # determined: the samples fix the solution (windows_determined). This does not depend on
        # analysis.min_overlap, which is a level of reliability and a warning of its own below.
        reason = not_determined_reason(self.reduced[:, columns], counts, f)
        if reason:
            return PartEstimate(reason=reason, kind="not determined")
        try:
            overlap = overlap_with_equal_weights(self.u_kn[:, columns], counts, f)
            neighbour = np.array([overlap[i, i + 1] for i in range(k - 1)])
        except Exception:  # noqa: BLE001
            neighbour = np.array([np.nan])
        lowest = float(np.min(neighbour)) if neighbour.size else 1.0
        warning = ""
        if not (np.isfinite(lowest) and lowest >= self.min_overlap):
            i = int(np.nanargmin(neighbour)) if np.isfinite(neighbour).any() else 0
            shown = "not defined" if not np.isfinite(lowest) else f"{lowest:.3f}"
            warning = (f"lambda windows {i} and {i + 1} overlap by {shown} among these samples, below "
                       f"analysis.min_overlap ({self.min_overlap:g}), the level of a reliable estimate")
        correction, correction_error, n_obs, why = self.tail.estimate(observations)
        if correction is None:
            return PartEstimate(reason=why, correction_samples=n_obs, kind="not determined")
        mbar = float(f[-1] * self.kt)
        mbar_error = None if df is None or not np.isfinite(df[-1]) else float(df[-1] * self.kt)
        return PartEstimate(f=f, mbar=mbar, mbar_error=mbar_error, correction=correction,
                            correction_error=correction_error, correction_samples=n_obs, value=mbar - correction,
                            lowest_overlap=lowest, warning=warning)

    def bar(self, columns: np.ndarray) -> float:
        """The sum of BAR between neighbouring windows from the samples ``columns``, kJ/mol."""
        counts = np.bincount(self.window[columns], minlength=len(self.n_k))
        return float(bar_between_neighbours(self.u_kn[:, columns], counts)[0] * self.kt)

    def bar_error(self, a: int, b: int, block_counts) -> tuple[float | None, int | None, str]:
        """Standard error (kJ/mol) of the sum of BAR between neighbouring windows over the
        iterations [a, b): the delete-one-block jackknife over the same time blocks as the error of
        MBAR (:meth:`jackknife`), the largest of the numbers of blocks that can be used, and that
        number. Two neighbouring pairs share a window, so the errors of the pairs are not
        independent and their sum in quadrature is no error of the sum (review of PR 1: three
        windows whose ends are one Hamiltonian give a sum of exactly 0 and that error 0.158 kT).
        (None, None, why) if no number of blocks can be used."""
        columns, _ = self.masks(a, b)
        best: tuple[float, int] | None = None
        notes = []
        for n_blocks in dict.fromkeys(int(c) for c in block_counts):
            edges = np.array([a + int(round(i * (b - a) / n_blocks)) for i in range(n_blocks + 1)])
            shortest = int(np.diff(edges).min())
            if shortest < self.longest:
                notes.append(f"{n_blocks} blocks: a block of {shortest} iteration(s) is shorter than the interval of "
                             f"{self.longest} iterations at which the slowest lambda window keeps its samples")
                continue
            values = []
            for i in range(n_blocks):
                in_block, _ = self.masks(int(edges[i]), int(edges[i + 1]))
                kept = columns & ~in_block
                counts = np.bincount(self.window[kept], minlength=len(self.n_k))
                few = too_few_samples(counts)
                try:
                    value = None if few else self.bar(kept)
                except BarNotDetermined as exc:
                    few, value = str(exc), None
                except Exception as exc:  # noqa: BLE001  a pair without overlap has no BAR estimate
                    few = f"BAR gave no estimate ({type(exc).__name__})"
                    value = None
                if value is None or not np.isfinite(value):
                    notes.append(f"{n_blocks} blocks: without block {i + 1}: {few or 'BAR gave no finite estimate'}")
                    values = []
                    break
                values.append(value)
            if values:
                x = np.asarray(values)
                spread = float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((x - x.mean()) ** 2)))
                if best is None or spread > best[0]:
                    best = (spread, n_blocks)
        if best is None:
            return None, None, "no number of time blocks could be used: " + "; ".join(notes)
        return best[0], best[1], ""

    def masks(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """The samples and the tail observations of the iterations [a, b) of the production."""
        return ((self.iteration >= a) & (self.iteration < b),
                (self.tail.times >= a) & (self.tail.times < b))

    def jackknife(self, a: int, b: int, n_blocks: int, base: PartEstimate,
                  replicates: list | None = None) -> tuple[float | None, float | None, str]:
        """Standard errors (kJ/mol) of the value and of its MBAR part from the iterations [a, b) by
        the delete-one-block jackknife over ``n_blocks`` consecutive time blocks of the same length,
        common to all windows: sqrt((B - 1) / B * sum_b (x_b - mean)^2) with x_b the estimate
        without block b. (None, None, why) if this number of blocks cannot be used: a block is
        shorter than the interval at which the slowest window keeps its samples, or the estimate
        without a block is not determined.
        """
        edges = np.array([a + int(round(i * (b - a) / n_blocks)) for i in range(n_blocks + 1)])
        shortest = int(np.diff(edges).min())
        if shortest < self.longest:
            return None, None, (f"a block of {shortest} iteration(s) is shorter than the interval of {self.longest} "
                                "iterations at which the slowest lambda window keeps its samples")
        columns, observations = self.masks(a, b)
        values, mbars = [], []
        for i in range(n_blocks):
            in_block, observed_in_block = self.masks(int(edges[i]), int(edges[i + 1]))
            without = self.estimate(columns & ~in_block, observations & ~observed_in_block, base.f)
            if not without.determined:
                return None, None, f"without block {i + 1}: {without.reason}"
            values.append(without.value)
            mbars.append(without.mbar)
        if replicates is not None:                       # (MBAR, correction) of every replicate
            replicates[:] = [(m, m - v) for m, v in zip(mbars, values)]

        def spread(x) -> float:
            x = np.asarray(x)
            return float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((x - x.mean()) ** 2)))

        return spread(values), spread(mbars), ""

    def part(self, a: int, b: int, block_counts, base: PartEstimate | None = None) -> dict:
        """The estimate of the iterations [a, b) with its standard error.

        The error is the largest of the jackknife errors of ``block_counts`` that can be used, or
        the analytical one (MBAR and the correction for independent samples) if that is larger
        or none can be used. Returns a dict with ``estimate`` (PartEstimate), ``error``,
        ``error_from``, ``n_blocks``, the jackknife errors per number of blocks (``blocks``,
        ``mbar_blocks``), ``block_notes`` and ``note``.
        """
        if base is None:
            columns, observations = self.masks(a, b)
            base = self.estimate(columns, observations, self.f_whole, analytic=True)
        out = {"estimate": base, "error": None, "error_from": "", "n_blocks": None, "blocks": {}, "mbar_blocks": {},
               "block_notes": {}, "note": "", "correlation": None}
        if not base.determined:
            return out
        replicates_of: dict[int, list] = {}
        for count in dict.fromkeys(int(c) for c in block_counts):
            replicates_of[count] = []
            value, mbar, why = self.jackknife(a, b, count, base, replicates_of[count])
            out["blocks"][str(count)], out["mbar_blocks"][str(count)] = value, mbar
            if why:
                out["block_notes"][str(count)] = why
        usable = {count: value for count, value in out["blocks"].items() if value is not None}
        # The analytical error. MBAR and the correction are estimated from the same configurations of
        # the coupled state, so their errors are correlated (review of PR 1: -0.79 in a two-state model).
        # Their correlation is taken from the jackknife replicates of the most blocks that can be used;
        # where no number of blocks can be used it is not known, and the sum of the two standard errors,
        # an upper bound of the error of their difference (Cauchy-Schwarz), is taken instead.
        out["correlation"] = None
        # an error that is not a finite number is no error: the analytical one is then not available
        # (review of PR 1: a NaN from pymbar replaced a usable error from time blocks)
        sm = base.mbar_error if base.mbar_error is not None and np.isfinite(base.mbar_error) else None
        sc = base.correction_error if base.correction_error is not None and np.isfinite(base.correction_error) else None
        if sm is None or sc is None:
            analytic = None
        elif sc == 0.0:
            analytic, kind = float(sm), "analytical"
        elif usable:
            pairs = np.array(replicates_of[max(int(c) for c in usable)])
            dm, dc = pairs[:, 0] - pairs[:, 0].mean(), pairs[:, 1] - pairs[:, 1].mean()
            rho = float(np.sum(dm * dc) / np.sqrt(np.sum(dm ** 2) * np.sum(dc ** 2))) \
                if np.sum(dm ** 2) > 0 and np.sum(dc ** 2) > 0 else 0.0
            rho = min(1.0, max(-1.0, rho))
            out["correlation"] = rho
            analytic, kind = float(np.sqrt(max(0.0, sm ** 2 + sc ** 2 - 2.0 * rho * sm * sc))), "analytical"
        else:
            analytic, kind = float(sm + sc), "analytical upper bound"
        best = max(usable, key=usable.get) if usable else None
        if best is not None and (analytic is None or usable[best] >= analytic):
            out.update(error=usable[best], error_from="time blocks", n_blocks=int(best))
        elif analytic is not None:
            out.update(error=analytic, error_from=kind)
        if not usable:
            out["note"] = ("no number of time blocks could be used: "
                           + "; ".join(f"{count} blocks: {why}" for count, why in out["block_notes"].items()))
        return out


def too_few_samples(counts, where: str = "there") -> str:
    """Why samples with ``counts`` per lambda window give no estimate, or "": every window must
    keep ``MIN_INDEPENDENT_SAMPLES``. One rule for every part of a production and for the whole of
    it (review of PR 1: the whole production with three samples per window gave -0.948 +- 0.946
    kJ/mol, where every part of it would have had no estimate)."""
    counts = np.asarray(counts)
    if counts.size and counts.min() < MIN_INDEPENDENT_SAMPLES:
        j = int(np.argmin(counts))
        return (f"lambda window {j} keeps {int(counts[j])} sample(s) {where}; {MIN_INDEPENDENT_SAMPLES} are needed "
                "for an estimate")
    return ""


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
                matrix = overlap_of(mbar)
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
        return overlap_of(mbar)


class BarNotDetermined(ValueError):
    """BAR between two neighbouring windows is not determined by their samples."""


def bar_between_neighbours(u_kn: np.ndarray, n_k: np.ndarray) -> tuple[float, float]:
    """Free energy of the last window relative to the first as the sum of BAR between neighbours.

    Returns (dG, standard error of the pairs in quadrature) in kT from samples grouped by state
    (``assemble`` or ``decorrelated``). BAR uses only the two windows of a pair, MBAR all windows
    at once; where the windows overlap well the two agree closely, so a difference is a sign of
    windows that are too far apart or too little sampled. The second number ignores that
    neighbouring pairs share a window; the analysis reports the error of the sum from time blocks
    (:meth:`TimeRangeEstimator.bar_error`).

    A pair counts only if its two windows determine their difference by the rule of every MBAR
    estimate (:func:`too_few_samples`, :func:`not_determined_reason` with the BAR solution):
    BarNotDetermined otherwise, never a number that the samples do not fix (pymbar's BAR returns a
    finite value for two windows that share no sample).
    """
    _pymbar()
    from pymbar import other_estimators

    u_kn = np.minimum(np.asarray(u_kn, dtype=float) - np.min(u_kn), 1.0e6)
    first = np.concatenate([[0], np.cumsum(n_k)])
    total, variance = 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            np.errstate(all="ignore"):
        for k in range(len(n_k) - 1):
            pair_counts = np.asarray(n_k[k:k + 2])
            few = too_few_samples(pair_counts, "there")
            if few:
                raise BarNotDetermined(f"lambda windows {k} and {k + 1}: {few}")
            lower = u_kn[:, first[k]:first[k + 1]]             # samples of window k
            upper = u_kn[:, first[k + 1]:first[k + 2]]         # samples of window k + 1
            res = other_estimators.bar(lower[k + 1] - lower[k], upper[k] - upper[k + 1])
            delta = float(res["Delta_f"])
            pair = u_kn[k:k + 2, first[k]:first[k + 2]]
            why = not_determined_reason(pair - np.min(pair), pair_counts, np.array([0.0, delta])) \
                if np.isfinite(delta) else "BAR gave no finite estimate"
            if why:
                raise BarNotDetermined(f"lambda windows {k} and {k + 1}: {why}")
            total += delta
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


def counted_samples(analysis: dict) -> list[float]:
    """The uncorrelated samples of every lambda window as the check of the samples counts them: the
    production divided by the statistical inefficiency of the window (``window_effective_samples``),
    and not more than the samples MBAR really gets from it (``window_kept_samples``). One count for
    :func:`quality_checks` and for every table that says how many samples a window has."""
    per_window = analysis.get("window_effective_samples") or []
    kept = analysis.get("window_kept_samples") or []
    return [min(float(e), float(kept[j])) if len(kept) == len(per_window) else float(e)
            for j, e in enumerate(per_window)]


def quality_checks(analysis: dict, min_overlap: float = 0.03, min_effective_samples: float = 50.0) -> dict:
    """What the analysis of a leg says about its reliability. ``analysis`` holds energies in kJ/mol.

    Two checks, each with the number the literature gives and nothing of cuFEP's own (the owner,
    2026-11-20: no criteria of cuFEP's own; a comparison of the two halves of the production and one
    of BAR with MBAR were removed, together with a reduction of the number of samples by the ratio
    of two standard errors):

    overlap      every pair of neighbouring lambda windows overlaps by at least ``min_overlap``:
                 "values as low as 0.03 [seem] to be tolerable to yield a reliable free energy
                 estimate"; below it "the free energy estimate itself will likely be substantially
                 incorrect" (Klimovich, Shirts and Mobley 2015, section 3.5). Two matrices are
                 examined, both in the normalisation of equal numbers of samples per window: that of
                 all iterations, and that of the samples the free energy is estimated from
                 (``used``, :func:`overlap_with_equal_weights`).
                 ``below`` names the one that is under the level.
    samples      the window with the fewest uncorrelated samples has at least
                 ``min_effective_samples``: "it is recommended that the effective number of
                 uncorrelated samples Neff >= ~50 if the BAR or MBAR estimators are used" (Mey et
                 al. 2020, section 8.2). The number of a window is its production divided by the
                 statistical inefficiency of its energy difference to the neighbouring window
                 (``fewest``), and not more than the samples MBAR really gets from it (``kept``).

    ``ok`` is None where the analysis does not hold the numbers for a check. The forward and
    reverse estimates, the two halves and BAR stay in the analysis as numbers without a verdict
    (OpenFE shows its forward and reverse series without one).
    """
    error = _number(analysis.get("delta_g_error"))
    kept = analysis.get("window_kept_samples") or []
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
        counts = counted_samples(analysis)
        j = int(np.argmin(counts))
        samples.update(ok=bool(counts[j] >= min_effective_samples), fewest=float(per_window[j]), window=j,
                       statistical_inefficiency=float(analysis["window_statistical_inefficiency"][j]),
                       counted=float(counts[j]))
        if len(kept) == len(per_window):
            samples["kept"] = int(kept[j])
        towards = analysis.get("window_inefficiency_neighbour") or []
        if len(towards) == len(per_window):
            samples["neighbour"] = int(towards[j])
    out["samples"] = samples
    out["uncertainty"] = error
    return out


#: the checks of :func:`quality_checks`: the only verdicts on a leg, each with the level the literature gives
QUALITY_CHECKS = ("overlap", "samples")


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


def swap_acceptance(samples: dict, n_equilibration: int, n: int) -> list[float | None] | None:
    """Accepted / proposed swaps between the replicas in neighbouring windows over the iterations of
    the production, per pair (None for a pair without proposals); None if the samples do not record
    the swaps (samples of earlier versions)."""
    if "swaps" not in samples:
        return None
    counts = np.asarray(samples["swaps"][n_equilibration:n_equilibration + n], dtype=float).sum(axis=0)
    return [float(a / p) if p > 0 else None for p, a in counts]


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
                part_error_blocks: int = PART_BLOCK_COUNT, tail: str = "none",
                min_overlap: float = 0.03) -> LegAnalysis:
    """Analyse samples loaded with :func:`cufep.sampling.storage.read_samples`.

    The free energy that is reported is that of decoupling, MBAR minus the tail correction
    ``tail`` (reweight, isotropic or none), and every part of the analysis is of that quantity
    (:class:`TimeRangeEstimator`): the standard error of the whole from time blocks common to all
    windows (for each number of ``error_blocks``), the two halves, the ``n_fractions`` points of
    the forward and reverse series and the ``n_blocks`` consecutive blocks, each with the error
    of ``part_error_blocks`` time blocks within it. MBAR is solved on samples that are
    decorrelated window by window (:func:`neighbour_inefficiencies`, :func:`decorrelated`), as
    Klimovich, Shirts and Mobley 2015 and alchemlyb do. A part whose free energy is not
    determined (see TimeRangeEstimator) is left out with the reason; halves that are not
    determined are reported as not compared. A part whose windows overlap below ``min_overlap``
    is kept and listed with a warning (``parts_below_level``, ``halves_warning``). BAR between neighbouring windows is
    compared with the MBAR part, on the same samples. Raises NoProduction and NoCorrection.
    """
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
    series = tail_series(samples, tail, n_equilibration, n, temperature_K)
    # the overlap matrix: equal numbers of samples per window, so that its elements mean the
    # same for every window (statistical inefficiency of the summed reduced potential)
    g_sum = statistical_inefficiency(u, states)
    stride = max(1, int(np.ceil(g_sum)))
    f_start, _df, overlap = solve_mbar(u[::stride], states[::stride])
    neighbour = [float(overlap[i, i + 1]) for i in range(n_states - 1)]
    # the free energy: every window decorrelated on its own, by the slower of its energy
    # differences to the two neighbours
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

    estimator = TimeRangeEstimator(u_kn, n_k, kept_sample_iterations(n, g), series, kt, min_overlap,
                                   int(np.ceil(g.max())), f_k)
    # the whole production: its MBAR solution is the one above; the overlap of its windows is a
    # check of its own (NOT RELIABLE), so it is not required here
    everything, everything_obs = estimator.masks(0, n)
    correction, correction_error, n_obs, why = series.estimate(everything_obs)
    if correction is None:
        raise NoCorrection(f"the tail correction ({tail}) cannot be estimated from the production: {why}")
    mbar, mbar_error = float(f_k[-1] * kt), float(df_k[-1] * kt)
    weights = series.weights(everything_obs)
    # the whole production is determined by the test of every part (review of PR 1: two windows that
    # shared no sample gave 0 +- 0 at analysis.min_overlap 0, and the ligand a binding free energy).
    # If it is not, the leg has no free energy and says why (not_determined)
    not_determined = too_few_samples(n_k, "in the production") or not_determined_reason(estimator.reduced, n_k, f_k)
    if not_determined:
        mbar = mbar_error = float("nan")
        base = PartEstimate(reason=not_determined, kind="not determined")
    else:
        base = PartEstimate(f=f_k, mbar=mbar, mbar_error=mbar_error, correction=correction,
                            correction_error=correction_error, correction_samples=n_obs, value=mbar - correction)
    whole = estimator.part(0, n, error_blocks, base=base)

    parts_not_determined: list[str] = []
    parts_below_level: list[str] = []
    warnings_of: dict[str, str] = {}

    def part(a: int, b: int, label: str):
        """(dG, error, number of time blocks behind the error) of the iterations [a, b), or a reason.
        A part below the level of overlap is returned, and its warning listed."""
        where = f"{label} (iterations {a}-{b} of the production)"
        result = whole if (a, b) == (0, n) else estimator.part(a, b, [part_error_blocks])
        if not result["estimate"].determined:
            return f"{where}: {result['estimate'].reason}"
        if result["error"] is None:
            return f"{where}: no standard error could be computed"
        if result["estimate"].warning:
            warnings_of[label] = f"{where}: {result['estimate'].warning}"
            parts_below_level.append(warnings_of[label])
        return result["estimate"].value, result["error"], result["n_blocks"]

    forward, reverse = [], []
    for i in range(1, n_fractions + 1):
        m = int(round(i * n / n_fractions))
        fwd, rev = part(0, m, f"forward point {m / n:.2f}"), part(n - m, n, f"reverse point {m / n:.2f}")
        # the fraction that is really used is the one that is stored and plotted
        for out, got in ((forward, fwd), (reverse, rev)):
            if isinstance(got, str):
                parts_not_determined.append(got)
            else:
                out.append((m / n, float(got[0]), float(got[1])))
    # the halves: split in the middle of the time series, no iteration in both
    mid = n // 2
    halves = [part(0, mid, "the first half"), part(mid, n, "the second half")]
    convergence_note = "; ".join(h for h in halves if isinstance(h, str))
    if convergence_note:
        first_half = second_half = (float("nan"), float("nan"))
        half_blocks = [None, None]
        halves_warning = ""
    else:
        first_half, second_half = ((float(h[0]), float(h[1])) for h in halves)
        half_blocks = [halves[0][2], halves[1][2]]
        halves_warning = "; ".join(warnings_of[label] for label in ("the first half", "the second half")
                                   if label in warnings_of)
    gap = abs(first_half[0] - second_half[0])
    blocks = []
    edges = [int(round(k * n / n_blocks)) for k in range(n_blocks + 1)] if n_blocks >= 2 else []
    consecutive = [part(a, b, f"block {i + 1} of {n_blocks}") for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    # every block that has an estimate is kept at its place; the others are listed
    blocks = [(a / n, b / n, float(c[0]), float(c[1])) for a, b, c in zip(edges[:-1], edges[1:], consecutive)
              if not isinstance(c, str)]
    parts_not_determined += [c for c in consecutive if isinstance(c, str)]
    bar_note, bar_error, bar_blocks, bar_error_note = "", float("nan"), None, ""
    try:
        bar = estimator.bar(everything)
    except BarNotDetermined as exc:
        bar = float("nan")
        bar_note = f"BAR has no estimate: {exc}"
    except Exception as exc:  # noqa: BLE001  a pair of windows without overlap has no BAR estimate
        bar = float("nan")
        bar_note = f"BAR gave no estimate for a pair of neighbouring windows ({type(exc).__name__})"
    if not bar_note and not np.isfinite(bar):
        bar_note = "BAR gave no finite estimate for a pair of neighbouring windows"
    if not bar_note:
        error, bar_blocks, bar_error_note = estimator.bar_error(0, n, error_blocks)
        bar_error = float("nan") if error is None else float(error)

    stages = {}
    state_dicts = attrs.get("states", [])
    if isinstance(state_dicts, list) and len(state_dicts) == n_states and not not_determined:
        for name, (a, b) in stage_boundaries(state_dicts).items():
            stages[name] = float((f_k[b] - f_k[a]) * kt)

    def whole_correction(method: str) -> float | None:
        """The correction of the whole production by another method, for comparison."""
        try:
            value = tail_series(samples, method, n_equilibration, n, temperature_K).estimate(slice(None))[0]
        except Exception:  # noqa: BLE001  a value for comparison only: missing, never an error of the analysis
            return None
        return None if value is None or not np.isfinite(value) else float(value)

    return LegAnalysis(
        n_states=n_states, n_iterations_used=int(n), statistical_inefficiency=float(g_sum),
        effective_samples_per_state=float(effective[slowest]),
        delta_g=float(base.value) if base.value is not None else float("nan"), not_determined=not_determined,
        # no standard error at all (no time blocks, no finite analytical one): none, never a number
        delta_g_error=float(whole["error"]) if whole["error"] is not None else float("nan"),
        delta_g_mbar=mbar, delta_g_mbar_error=mbar_error,
        correction_method=tail, correction=float(correction), correction_error=float(correction_error),
        correction_samples=int(n_obs), correction_effective_samples=weights[0], correction_largest_weight=weights[1],
        correction_correlation=whole["correlation"],
        delta_g_error_blocks=whole["blocks"],
        delta_g_mbar_error_blocks=whole["mbar_blocks"], delta_g_error_blocks_note=whole["block_notes"],
        delta_g_error_from=whole["error_from"], delta_g_error_n_blocks=whole["n_blocks"],
        delta_g_error_note=whole["note"], halves_error_blocks=half_blocks,
        window_statistical_inefficiency=[float(x) for x in g],
        window_inefficiency_previous=[float(x) for x in table[:, 0]],
        window_inefficiency_next=[float(x) for x in table[:, 1]], window_inefficiency_neighbour=towards,
        window_effective_samples=[float(x) for x in effective], window_kept_samples=[int(x) for x in n_k],
        slowest_window=slowest,
        f_k=(f_k * kt).tolist(), df_k=(df_k * kt).tolist(), stage_delta_g=stages,
        overlap_neighbour=neighbour, min_overlap=float(min(neighbour)) if neighbour else 1.0,
        round_trips=count_round_trips(states), neighbour_transition_fraction=neighbour_swap_fraction(states),
        swap_acceptance_neighbour=swap_acceptance(samples, n_equilibration, n),
        overlap_matrix=np.round(overlap, 4).tolist(),
        overlap_used_matrix=[] if used is None else np.round(used, 4).tolist(),
        overlap_used_weighted_by_number=[] if by_number is None else np.round(by_number, 4).tolist(),
        overlap_used_neighbour=used_neighbour,
        min_overlap_used=float(min(used_neighbour)) if used_neighbour else float("nan"),
        forward=forward, reverse=reverse, first_half=first_half, second_half=second_half,
        convergence_note=convergence_note, halves_warning=halves_warning, forward_reverse_gap=float(gap), blocks=blocks,
        parts_not_determined=parts_not_determined, parts_below_level=parts_below_level,
        bar_delta_g=float(bar), bar_delta_g_error=float(bar_error), bar_delta_g_error_n_blocks=bar_blocks,
        bar_delta_g_error_note=bar_error_note, bar_note=bar_note,
        lrc_reweight=whole_correction("reweight"), lrc_isotropic=whole_correction("isotropic"),
        temperature_K=temperature_K)
