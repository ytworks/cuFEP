"""Statistics for comparing predicted and experimental binding free energies.

Absolute calculations (cuFEP) are judged on their raw values. FEP+ values in the public
benchmark come from relative calculations shifted to the experimental mean, so their mean
error is zero by construction. The like-for-like comparison is therefore made after
removing each method's mean signed error ("centred"), on pairwise differences, and on
rank correlation.

A statistic that is not defined for the data (a correlation of one ligand or of a constant
series, a pairwise error or an error around the mean of one ligand) is None, shown as N/A, never
a number (review of PR 1: two ligands with R^2 = 1 were reported as 0.00).
"""
from __future__ import annotations

import numpy as np

#: statistics of the agreement between two series that need two different values in each
CORRELATIONS = ("kendall_tau", "spearman", "pearson_r", "pearson_r2")


def _defined(a: np.ndarray, b: np.ndarray) -> bool:
    """A correlation of ``a`` and ``b`` is defined: two points or more, neither series constant."""
    return len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import kendalltau

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not _defined(a, b):
        return None
    t = kendalltau(a, b).statistic
    return float(t) if np.isfinite(t) else None


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not _defined(a, b):
        return None
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None


def pearson_r(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not _defined(a, b):
        return None
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else None


def pearson_r2(a: np.ndarray, b: np.ndarray) -> float | None:
    r = pearson_r(a, b)
    return None if r is None else r * r


def metrics(pred, exp) -> dict[str, float | None]:
    pred, exp = np.asarray(pred, dtype=float), np.asarray(exp, dtype=float)
    err = pred - exp
    centred = err - err.mean()
    i, j = np.triu_indices(len(pred), k=1)
    pair = (pred[i] - pred[j]) - (exp[i] - exp[j])
    return {"n": float(len(pred)), "mean_signed_error": float(err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mue": float(np.mean(np.abs(err))),
            "centred_rmse": float(np.sqrt(np.mean(centred ** 2))) if len(pred) >= 2 else None,
            "pairwise_rmse": float(np.sqrt(np.mean(pair ** 2))) if len(pair) else None,
            "kendall_tau": kendall_tau(pred, exp), "spearman": spearman(pred, exp), "pearson_r": pearson_r(pred, exp),
            "pearson_r2": pearson_r2(pred, exp)}


def bootstrap(pred, exp, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile confidence intervals from resampling ligands with replacement.

    Every resample counts for every statistic it defines: the errors are defined for every
    resample, a correlation only where the resample has two different values in both series, and
    a resample without it is left out for that statistic alone (review of PR 1: resamples of fewer
    than three different ligands were dropped for every statistic, and for the errors
    [0, 0, 0, 4] the interval of the mean signed error was [0, 2] instead of [0, 3]).
    """
    pred, exp = np.asarray(pred, dtype=float), np.asarray(exp, dtype=float)
    rng = np.random.default_rng(seed)
    keys = list(metrics(pred, exp))
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics(pred[idx], exp[idx])
        for k in keys:
            if m[k] is not None and np.isfinite(m[k]):
                samples[k].append(m[k])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items() if v}
