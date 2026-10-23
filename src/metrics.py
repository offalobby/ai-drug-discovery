"""Ranking, enrichment and regression metrics with uncertainty estimates.

Enrichment definitions (N molecules, A actives, budget fraction f):
    k                = ceil(f * N)
    expected random  = k * A / N
    EF@f             = (hits in top k / k) / (A / N) = hits / expected random

Tied scores at the selection boundary are resolved by their expectation under random
tie-breaking, so the result never depends on the input row order.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import (average_precision_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)


def budget_size(n: int, fraction: float) -> int:
    """Number of molecules selected under a top-`fraction` screening budget (at least 1)."""
    return max(1, math.ceil(fraction * n - 1e-9))


def hits_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Actives among the k highest scores; boundary ties count by expectation (may be fractional)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if not 0 < k <= len(s):
        raise ValueError(f"k must be in [1, {len(s)}], got {k}")
    threshold = np.sort(s)[::-1][k - 1]
    above = s > threshold
    tied = s == threshold
    n_needed_from_ties = k - above.sum()
    return float(y[above].sum() + n_needed_from_ties * y[tied].sum() / tied.sum())


def enrichment(y_true: Sequence[int], scores: Sequence[float], fraction: float) -> dict:
    """Hits, precision, expected random hits, EF and hypergeometric p-value at a budget fraction."""
    y = np.asarray(y_true, dtype=int)
    n, a = len(y), int(y.sum())
    k = budget_size(n, fraction)
    hits = hits_at_k(y, scores, k)
    expected = k * a / n
    # P(random selection of k without replacement finds >= observed hits). Fractional tie-hits
    # are rounded down, which makes the p-value conservative.
    observed_int = int(math.floor(hits + 1e-9))
    p_value = float(hypergeom.sf(observed_int - 1, n, a, k)) if a > 0 else float("nan")
    return {
        "fraction": fraction,
        "n_pool": n,
        "n_actives": a,
        "n_selected": k,
        "hits": hits,
        "precision": hits / k,
        "prevalence": a / n,
        "expected_random_hits": expected,
        "enrichment_factor": hits / expected if expected > 0 else float("nan"),
        "recall": hits / a if a > 0 else float("nan"),
        "random_p_value": p_value,
        # Central 95% range of hits a uniformly random selection of k molecules would obtain
        "random_hits_95pct_interval": [int(hypergeom.ppf(0.025, n, a, k)), int(hypergeom.ppf(0.975, n, a, k))]
        if a > 0 else [0, 0],
    }


def bootstrap_ef(y_true, scores, fraction: float = 0.05, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap interval for EF by resampling held-out molecules with replacement."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        values.append(enrichment(y[idx], s[idx], fraction)["enrichment_factor"])
    values = np.array(values)
    return {
        "fraction": fraction,
        "n_boot_used": int(len(values)),
        "ci_low": float(np.quantile(values, alpha / 2)),
        "ci_high": float(np.quantile(values, 1 - alpha / 2)),
        "median": float(np.median(values)),
    }


def classification_summary(y_true, scores, fractions=(0.01, 0.05, 0.10)) -> dict:
    """ROC-AUC, average precision and enrichment at several budgets."""
    y = np.asarray(y_true, dtype=int)
    out = {
        "n": int(len(y)),
        "n_actives": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() > 0 else float("nan"),
    }
    for f in fractions:
        out[f"enrichment@{int(round(f * 100))}%"] = enrichment(y, scores, f)
    return out


def regression_summary(y_true, y_pred) -> dict:
    """MAE, RMSE and R^2 in the target's own units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_regression(y_true, y_pred, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap intervals for MAE, RMSE and R^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    samples = {"mae": [], "rmse": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        summary = regression_summary(y_true[idx], y_pred[idx])
        for key in samples:
            samples[key].append(summary[key])
    return {key: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
            for key, v in samples.items()}
