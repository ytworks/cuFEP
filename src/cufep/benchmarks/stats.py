"""Statistics for comparing predicted and experimental binding free energies.

Absolute calculations (cuFEP) are judged on their raw values. FEP+ values in the public
benchmark come from relative calculations shifted to the experimental mean, so their mean
error is zero by construction. The like-for-like comparison is therefore made after
removing each method's mean signed error ("centred"), on pairwise differences, and on
rank correlation.
"""
from __future__ import annotations

import numpy as np


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    t = kendalltau(a, b).statistic
    return float(t) if np.isfinite(t) else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def metrics(pred, exp) -> dict[str, float]:
    pred, exp = np.asarray(pred, dtype=float), np.asarray(exp, dtype=float)
    err = pred - exp
    centred = err - err.mean()
    i, j = np.triu_indices(len(pred), k=1)
    pair = (pred[i] - pred[j]) - (exp[i] - exp[j])
    r = np.corrcoef(pred, exp)[0, 1] if len(pred) > 2 and pred.std() > 0 and exp.std() > 0 else 0.0
    return {"n": float(len(pred)), "mean_signed_error": float(err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mue": float(np.mean(np.abs(err))), "centred_rmse": float(np.sqrt(np.mean(centred ** 2))),
            "pairwise_rmse": float(np.sqrt(np.mean(pair ** 2))) if len(pair) else 0.0,
            "kendall_tau": kendall_tau(pred, exp), "spearman": spearman(pred, exp), "pearson_r2": float(r * r)}


def bootstrap(pred, exp, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile confidence intervals from resampling ligands with replacement."""
    pred, exp = np.asarray(pred, dtype=float), np.asarray(exp, dtype=float)
    rng = np.random.default_rng(seed)
    keys = list(metrics(pred, exp))
    samples = {k: [] for k in keys}
    n = len(pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        m = metrics(pred[idx], exp[idx])
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items() if v}
