"""MBAR analysis of one replica-exchange leg."""
from __future__ import annotations

import contextlib
import io
import os
from dataclasses import asdict, dataclass, field

import numpy as np

R_GAS = 8.31446261815324e-3   # kJ/mol/K
KCAL = 4.184


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
    statistical_inefficiency: float
    effective_samples_per_state: float
    #: free energy of the last state relative to the first, kJ/mol (and MBAR standard error)
    delta_g: float
    delta_g_error: float
    #: cumulative free energy at every state relative to the first, kJ/mol
    f_k: list[float] = field(default_factory=list)
    df_k: list[float] = field(default_factory=list)
    stage_delta_g: dict[str, float] = field(default_factory=dict)
    overlap_neighbour: list[float] = field(default_factory=list)
    min_overlap: float = 0.0
    round_trips: int = 0
    swap_acceptance_neighbour: list[float] = field(default_factory=list)
    #: the whole overlap matrix of the windows (overlap_neighbour is its first off-diagonal)
    overlap_matrix: list[list[float]] = field(default_factory=list)
    #: free energy from growing parts of the production, counted from its start (forward) and
    #: from its end (reverse): (fraction of the production, dG, error)
    forward: list[tuple[float, float, float]] = field(default_factory=list)
    reverse: list[tuple[float, float, float]] = field(default_factory=list)
    #: (dG, error) from the first half and from the second half of the production alone
    first_half: tuple[float, float] = (float("nan"), float("nan"))
    second_half: tuple[float, float] = (float("nan"), float("nan"))
    #: difference between the two halves
    forward_reverse_gap: float = 0.0
    #: sum of BAR between neighbouring windows, a second estimator on the same data
    bar_delta_g: float = float("nan")
    bar_delta_g_error: float = float("nan")
    lrc_delta_g: float = 0.0
    lrc_error: float = 0.0
    lrc_samples: int = 0
    #: the same correction in the homogeneous (isotropic) approximation that OpenMM's
    #: long-range correction and OpenFE use; reported for comparison
    lrc_isotropic: float = 0.0
    temperature_K: float = 298.15

    def to_dict(self) -> dict:
        return asdict(self)


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


def overlap_matrix(u_kn: np.ndarray, n_k: np.ndarray) -> np.ndarray:
    """MBAR overlap matrix for reduced energies ``u_kn[state, sample]`` (samples grouped by state)."""
    mbar = _mbar(u_kn, n_k)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return np.asarray(mbar.compute_overlap()["matrix"])


def bar_between_neighbours(u: np.ndarray, states: np.ndarray) -> tuple[float, float]:
    """Free energy of the last window relative to the first as the sum of BAR between neighbours.

    Returns (dG, standard error) in kT, from ``u[iter, replica, state]`` and
    ``states[iter, replica]``. BAR uses only the two windows of a pair, MBAR all windows at
    once; where the windows overlap well the two agree closely, so a difference is a sign of
    windows that are too far apart or too little sampled.
    """
    _pymbar()
    from pymbar import other_estimators

    n_states = u.shape[2]
    flat_u = np.minimum(u.reshape(-1, n_states) - np.min(u), 1.0e6)
    flat_s = states.reshape(-1)
    total, variance = 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            np.errstate(all="ignore"):
        for k in range(n_states - 1):
            lower, upper = flat_u[flat_s == k], flat_u[flat_s == k + 1]
            res = other_estimators.bar(lower[:, k + 1] - lower[:, k], upper[:, k] - upper[:, k + 1])
            total += float(res["Delta_f"])
            variance += float(res["dDelta_f"]) ** 2
    return total, float(np.sqrt(variance))


def quality_checks(analysis: dict, min_overlap: float = 0.03, tolerance_sigma: float = 2.0) -> dict:
    """What the analysis of a leg says about its reliability. Energies in the units of ``analysis``.

    Three checks, after Klimovich, Shirts and Mobley, J. Comput. Aided Mol. Des. 2015, 29, 397:

    overlap      every pair of neighbouring windows overlaps by at least ``min_overlap``
                 ("values as low as 0.03 ... tolerable"; also the warning level of Flare).
    convergence  the free energy from the first half of the production alone, and from the second
                 half alone, lies within ``tolerance_sigma`` MBAR standard errors of the free
                 energy from all of it ("both sets of data should converge rapidly to within
                 uncertainty of the final value"; the guideline gives no factor, 2 is cuFEP's).
    estimators   BAR between neighbouring windows and MBAR agree within the same tolerance (the
                 guideline compares TI with BAR-type estimators; cuFEP records no dU/dlambda).

    ``ok`` is None where the analysis does not hold the numbers for a check.
    """
    final, error = float(analysis["delta_g"]), float(analysis["delta_g_error"])
    tolerance = tolerance_sigma * error
    neighbour = list(analysis.get("overlap_neighbour", []))
    lowest = float(analysis.get("min_overlap", 1.0))
    pair = int(np.argmin(neighbour)) if neighbour else None
    out = {"overlap": {"ok": bool(lowest >= min_overlap), "lowest": lowest,
                       "windows": None if pair is None else [pair, pair + 1], "threshold": min_overlap}}

    halves = [analysis.get("first_half"), analysis.get("second_half")]
    conv = {"ok": None, "tolerance": tolerance, "final": final}
    if all(h is not None for h in halves):
        first, second = float(halves[0][0]), float(halves[1][0])
        conv.update(first_half=first, second_half=second)
        if np.isfinite(first) and np.isfinite(second):
            conv["ok"] = bool(max(abs(first - final), abs(second - final)) <= tolerance)
        elif np.isfinite(final):
            conv["ok"] = False       # half of the data does not determine the free energy
    out["convergence"] = conv

    bar = analysis.get("bar_delta_g")
    est = {"ok": None, "tolerance": tolerance, "mbar": final}
    if bar is not None and np.isfinite(float(bar)):
        est.update(bar=float(bar), ok=bool(abs(float(bar) - final) <= tolerance))
    out["estimators"] = est
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
                n_fractions: int = 10) -> LegAnalysis:
    """Analyse samples loaded with :func:`cufep.sampling.storage.read_samples`.

    ``n_fractions`` is the number of points of the forward and reverse series.
    """
    from cufep.alchemy.lrc import tail_free_energy

    kt = R_GAS * temperature_K
    attrs = samples.get("attrs", {})
    if n_equilibration is None:
        n_equilibration = int(attrs.get("n_equilibration_iterations", 0))
    u = samples["u"][n_equilibration:]
    states = samples["states"][n_equilibration:]
    if u.shape[0] < 4:
        raise ValueError("not enough production iterations to analyse")
    g = statistical_inefficiency(u, states)
    stride = max(1, int(np.ceil(g)))
    f_k, df_k, overlap = solve_mbar(u[::stride], states[::stride])
    n_states = u.shape[2]
    neighbour = [float(overlap[i, i + 1]) for i in range(n_states - 1)]

    def partial(sl):
        try:
            # starts from the solution of all data and then converges in a few iterations; a
            # part that needs more has a window without overlap and no defined answer
            f, df, _ = solve_mbar(u[sl][::stride], states[sl][::stride], initial_f_k=f_k, overlap=False,
                                  max_iterations=30)
            return float(f[-1] * kt), float(df[-1] * kt)
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")

    n = u.shape[0]
    forward, reverse = [], []
    everything = (float(f_k[-1] * kt), float(df_k[-1] * kt))
    for i in range(1, n_fractions + 1):
        frac = i / n_fractions
        m = max(4 * stride, int(round(frac * n)))
        forward.append((frac, *(everything if m >= n else partial(slice(0, m)))))
        reverse.append((frac, *(everything if m >= n else partial(slice(n - m, n)))))
    half = max(4 * stride, n // 2)
    first_half, second_half = partial(slice(0, half)), partial(slice(n - half, n))
    gap = abs(first_half[0] - second_half[0])
    try:
        bar, bar_error = bar_between_neighbours(u[::stride], states[::stride])
        bar, bar_error = bar * kt, bar_error * kt
    except Exception:  # noqa: BLE001  a pair of windows without overlap has no BAR estimate
        bar = bar_error = float("nan")

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
        n_states=n_states, n_iterations_used=int(n), statistical_inefficiency=float(g),
        effective_samples_per_state=float(n / g), delta_g=float(f_k[-1] * kt), delta_g_error=float(df_k[-1] * kt),
        f_k=(f_k * kt).tolist(), df_k=(df_k * kt).tolist(), stage_delta_g=stages,
        overlap_neighbour=neighbour, min_overlap=float(min(neighbour)) if neighbour else 1.0,
        round_trips=count_round_trips(states), swap_acceptance_neighbour=neighbour_swap_fraction(states),
        overlap_matrix=np.round(overlap, 4).tolist(),
        forward=forward, reverse=reverse, first_half=first_half, second_half=second_half,
        forward_reverse_gap=float(gap), bar_delta_g=float(bar), bar_delta_g_error=float(bar_error),
        lrc_delta_g=lrc_dg, lrc_error=lrc_se, lrc_samples=lrc_n, lrc_isotropic=lrc_iso,
        temperature_K=temperature_K)
