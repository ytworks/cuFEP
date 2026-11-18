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
    # pymbar pulls in JAX when present; keep it on the CPU and quiet.
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
    forward: list[tuple[float, float, float]] = field(default_factory=list)   # (fraction, dG, error)
    reverse: list[tuple[float, float, float]] = field(default_factory=list)
    forward_reverse_gap: float = 0.0
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


def solve_mbar(u: np.ndarray, states: np.ndarray):
    """Return (f_k, df_k relative to state 0, overlap matrix) in kT."""
    pymbar, _ = _pymbar()
    u_kn, n_k = assemble(u, states)
    # Overlapping-atom configurations give astronomically large energies in coupled states;
    # cap them so the solver stays finite (their weight is zero either way).
    u_kn = np.minimum(u_kn - u_kn.min(), 1.0e6)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mbar = pymbar.MBAR(u_kn, n_k, solver_protocol="robust")
        res = mbar.compute_free_energy_differences()
        overlap = mbar.compute_overlap()["matrix"]
    return np.asarray(res["Delta_f"][0]), np.asarray(res["dDelta_f"][0]), np.asarray(overlap)


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
    """Index ranges (first, last state) of the restraint, electrostatics and sterics stages."""
    lr = [s.get("lambda_restraints", 1.0) for s in state_dicts]
    le = [s.get("lambda_electrostatics", 1.0) for s in state_dicts]
    ls = [s.get("lambda_sterics", 1.0) for s in state_dicts]
    n = len(state_dicts)
    out: dict[str, tuple[int, int]] = {}
    i_r = max((i for i in range(n) if le[i] == 1.0 and ls[i] == 1.0), default=0)
    i_e = max((i for i in range(n) if ls[i] == 1.0), default=i_r)
    if i_r > 0 and lr[0] < 1.0:
        out["restraints"] = (0, i_r)
    if i_e > i_r:
        out["electrostatics"] = (i_r if "restraints" in out else 0, i_e)
    if n - 1 > i_e:
        out["sterics"] = (i_e, n - 1)
    return out


def analyze_leg(samples: dict, temperature_K: float, n_equilibration: int | None = None,
                n_fractions: int = 5) -> LegAnalysis:
    """Analyse samples loaded with :func:`cufep.sampling.storage.read_samples`."""
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
            f, df, _ = solve_mbar(u[sl][::stride], states[sl][::stride])
            return float(f[-1] * kt), float(df[-1] * kt)
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")

    n = u.shape[0]
    forward, reverse = [], []
    for i in range(1, n_fractions + 1):
        frac = i / n_fractions
        m = max(4 * stride, int(round(frac * n)))
        forward.append((frac, *partial(slice(0, m))))
        reverse.append((frac, *partial(slice(n - m, n))))
    half = max(4 * stride, n // 2)
    gap = abs(partial(slice(0, half))[0] - partial(slice(n - half, n))[0])

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
        forward=forward, reverse=reverse, forward_reverse_gap=float(gap),
        lrc_delta_g=lrc_dg, lrc_error=lrc_se, lrc_samples=lrc_n, lrc_isotropic=lrc_iso,
        temperature_K=temperature_K)
