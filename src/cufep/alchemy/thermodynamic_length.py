"""Thermodynamic length along a stage, and what a length says about two neighbouring lambda windows.

The metric and its integration are those of Clark, Robb, Cole and Michel, J. Chem. Theory Comput.
2024, 20, 7806 (A3FE), section 2.1.1 and eq 19, as the source of that program computes them
(``a3fe/analyse/process_grads.py``, ``get_integrated_error`` and ``calculate_optimal_lam_vals``):

1. At each lambda value that was simulated, take the standard deviation of the gradient of the
   energy with respect to lambda over the samples of that value ("root variance of the gradients").
2. Integrate it over lambda with the trapezoidal rule. The integral from the start of the stage is
   the length up to there; the paper calls a step of that integral a thermodynamic speed.

cuFEP takes the gradient as a central difference of the reduced energies that the precalculation of
adaptive lambda scheduling computes at all candidate lambda values of a stage, so a length here is
in kT (the paper's speeds are in kcal/mol: times kT). With REST2 the scale of the hot region changes
along the leg with lambda, and the difference of the candidate energies holds that change too.

What a length says about two windows that are ``d`` apart, if the energy difference between them
is normally distributed (their own samples then give it the same variance, ``d ** 2``, and means
that differ by ``d ** 2``):

- an exchange of their configurations is accepted with probability ``erfc(d / 2)``: with ``X`` the
  difference in a configuration of the first and ``Y`` in one of the second, the acceptance is
  ``min(1, exp(Y - X))`` and ``X - Y`` is normal with mean ``d ** 2`` and variance ``2 d ** 2``.
  Walter and Bereau, arXiv:2610.11714 v1, section S1.8, space their windows by this relation
  ("we assumed Gaussian distributed exchange energies. The acceptance of a pair then depends only
  on the distance between the two states, analogous to a thermodynamic length");
- the overlap of the pair alone (the element of the overlap matrix of the two windows, with as many
  samples of each) is ``exp(-d ** 2 / 8) * E[1 / (2 cosh(d z / 2))]`` over a standard normal ``z``:
  0.5 for ``d = 0``, 0.398 for 1, 0.225 for 2.

Nothing here decides anything: these are numbers, and the functions that measure them from samples.
A number that cannot be had from its input (an energy difference that is not a number, a single
sample) is NaN, never a value that looks like a measurement.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gradient_sd", "integrated_length", "pair_differences", "pair_distance", "exchange_acceptance",
           "stage_record", "gaussian_acceptance", "length_of_acceptance", "gaussian_pair_overlap"]


def gradient_sd(energies: np.ndarray, index: int, spacing: float) -> float:
    """Standard deviation over the samples of a window of the gradient of the reduced energy with
    respect to lambda, in kT.

    ``energies`` holds the samples of the window at all candidates of the stage, shape (samples,
    candidates), ``index`` is the candidate of the window and ``spacing`` the lambda between two
    candidates. The gradient is the central difference of the two candidates next to the window,
    and the one-sided difference at an end of the stage.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError("a gradient needs the energies at two candidates or more, as (samples, candidates)")
    if not 0 <= index < energies.shape[1]:
        raise ValueError(f"candidate {index} is not one of the {energies.shape[1]} candidates")
    lo, hi = max(index - 1, 0), min(index + 1, energies.shape[1] - 1)
    return _sd((energies[:, hi] - energies[:, lo]) / ((hi - lo) * spacing))


def _sd(values: np.ndarray) -> float:
    """The standard deviation of samples; NaN for fewer than two or for one that is not finite."""
    values = np.asarray(values, dtype=float).ravel()
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        sd = float(np.std(values, ddof=1))
    return sd if np.isfinite(sd) else float("nan")       # values whose squares overflow have none


def integrated_length(lambdas, sds) -> np.ndarray:
    """The length from the first window to each window: the trapezoidal integral over lambda of
    the standard deviations of the gradient at the windows (0 at the first window)."""
    lambdas, sds = np.asarray(lambdas, dtype=float), np.asarray(sds, dtype=float)
    if lambdas.shape != sds.shape or lambdas.ndim != 1 or len(lambdas) < 2:
        raise ValueError("one standard deviation per window, for at least two windows")
    if np.any(np.diff(lambdas) <= 0.0):
        raise ValueError("the lambda values of the windows must increase")
    steps = 0.5 * (sds[1:] + sds[:-1]) * np.diff(lambdas)
    return np.concatenate([[0.0], np.cumsum(steps)])


def pair_differences(u: np.ndarray, states: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """For every pair of neighbouring windows of a replica exchange run, the reduced energy of the
    upper window minus that of the lower one, in the configurations the lower window holds from
    iteration to iteration and in those the upper window holds.

    ``u[iteration, replica, state]`` and ``states[iteration, replica]`` as the samples store them
    (one replica per state). Entry ``k`` is ``(x, y)`` for windows ``k`` and ``k + 1``.
    """
    u, states = np.asarray(u, dtype=float), np.asarray(states)
    n, _replicas, k = u.shape
    if states.shape != (n, k) or not np.array_equal(np.sort(states, axis=1), np.broadcast_to(np.arange(k), (n, k))):
        raise ValueError("every iteration must have one replica in each state")
    replica_in = np.argsort(states, axis=1)                     # the replica that is in window j at each iteration
    it = np.arange(n)
    out = []
    for j in range(k - 1):
        lower, upper = replica_in[:, j], replica_in[:, j + 1]
        out.append((u[it, lower, j + 1] - u[it, lower, j], u[it, upper, j + 1] - u[it, upper, j]))
    return out


def pair_distance(x, y) -> float:
    """The length between two windows from the energy difference between them in the samples of
    each (``x`` in those of the first, ``y`` in those of the second): the root of the mean of the
    two variances, in kT. For a normally distributed difference both variances are the square of
    the length; windows far apart leave that relation."""
    return float(np.sqrt(0.5 * (_sd(x) ** 2 + _sd(y) ** 2)))


def exchange_acceptance(x, y) -> float:
    """The acceptance of an exchange between a configuration of one window and a configuration of
    another, averaged over all pairs of their samples.

    ``x`` is the reduced energy of the second window minus that of the first in the samples of the
    first, ``y`` the same difference in the samples of the second. A pair is accepted with
    probability ``min(1, exp(y - x))``, the rule of the replica exchange. NaN if ``y - x`` is not a
    number for some pair: a difference that is not a number, or infinite ones of the same sign in
    both windows.
    """
    x, y = np.asarray(x, dtype=float).ravel(), np.sort(np.asarray(y, dtype=float).ravel())
    if len(x) == 0 or len(y) == 0:
        raise ValueError("an acceptance needs samples of both windows")
    same_infinity = any(np.any(x == v) and np.any(y == v) for v in (np.inf, -np.inf))
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or same_infinity:
        return float("nan")
    below = np.searchsorted(y, x, side="left")                  # the y smaller than each x: accepted with exp(y - x)
    log_sums = np.concatenate([[-np.inf], np.logaddexp.accumulate(y)])      # log of the sum of exp(y) over the smallest
    with np.errstate(invalid="ignore"):
        # at most `below`: every term is at most 1; no y below an x, no term
        partly = np.where(below > 0, np.exp(log_sums[below] - x), 0.0)
    return float(np.mean((len(y) - below + partly) / len(y)))


def stage_record(energies: list[np.ndarray], windows: list[int], spacing: float) -> dict:
    """The lengths of one stage from the samples of its windows, as numbers for a record.

    ``energies[i]`` holds the samples of window ``i`` at all candidates of the stage, shape
    (samples, candidates), and ``windows`` the candidates of the windows, in increasing order.
    ``gradient_sd`` and ``length`` (from the first window, by :func:`integrated_length`) have one
    value per window; ``pair_distance`` and ``exchange_acceptance`` one per pair of neighbouring
    windows, from the energy difference between the two in the samples of each.
    """
    sds = [gradient_sd(e, i, spacing) for e, i in zip(energies, windows)]
    distances, acceptances = [], []
    for (ea, a), (eb, b) in zip(zip(energies, windows), zip(energies[1:], windows[1:])):
        x, y = np.asarray(ea)[:, b] - np.asarray(ea)[:, a], np.asarray(eb)[:, b] - np.asarray(eb)[:, a]
        distances.append(pair_distance(x, y))
        acceptances.append(exchange_acceptance(x, y))
    return {"gradient_sd": sds, "length": integrated_length([i * spacing for i in windows], sds).tolist(),
            "pair_distance": distances, "exchange_acceptance": acceptances}


def gaussian_acceptance(distance):
    """The acceptance of an exchange between two windows ``distance`` apart whose energy difference
    is normally distributed: ``erfc(distance / 2)``."""
    from scipy.special import erfc

    return erfc(0.5 * np.asarray(distance, dtype=float))


def length_of_acceptance(acceptance):
    """The distance at which :func:`gaussian_acceptance` is ``acceptance`` (above 0, at most 1)."""
    from scipy.special import erfcinv

    acceptance = np.asarray(acceptance, dtype=float)
    if np.any(~(acceptance > 0.0)) or np.any(acceptance > 1.0):
        raise ValueError("an acceptance is above 0 and at most 1")
    return 2.0 * erfcinv(acceptance)


def gaussian_pair_overlap(distance):
    """The overlap of two windows alone that are ``distance`` apart and whose energy difference is
    normally distributed: the element of the overlap matrix of the two, with as many samples of
    each. By a quadrature of 200 points: to 1e-7 of the value up to a distance of 5 and to 1e-4
    up to 8, where the overlap is 5e-5; beyond that less of the value is right."""
    z, w = np.polynomial.hermite_e.hermegauss(200)
    d = np.asarray(distance, dtype=float)[..., None]
    t = np.abs(0.5 * d * z)
    half_sech = np.exp(-t) / (1.0 + np.exp(-2.0 * t))           # 1 / (2 cosh t), without the overflow of cosh
    return np.exp(-0.125 * d[..., 0] ** 2) * np.sum(w * half_sech, axis=-1) / np.sqrt(2.0 * np.pi)
