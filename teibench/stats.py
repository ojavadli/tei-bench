"""Statistical analysis for the paired before/after design.

Primary test: paired t-test on per-agent deltas (n agents).
Robustness:  Wilcoxon signed-rank (non-parametric, no normality assumption).
Effect size: Cohen's d_z (paired), with bootstrap 95% CI on the mean delta.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def analyze_paired(before: Sequence[float], after: Sequence[float]) -> dict:
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    assert b.shape == a.shape and b.ndim == 1 and len(b) >= 2
    deltas = a - b
    n = len(deltas)
    mean_d = float(deltas.mean())
    sd_d = float(deltas.std(ddof=1))

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank (skip zero-diffs handled by scipy); guard all-equal
    try:
        if np.allclose(deltas, 0):
            w_stat, w_p = float("nan"), 1.0
        else:
            w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Cohen's d_z for paired data
    d_z = mean_d / sd_d if sd_d > 0 else float("inf") if mean_d != 0 else 0.0

    # Bootstrap 95% CI on the mean delta
    rng = np.random.default_rng(12345)
    boot = np.array([
        rng.choice(deltas, size=n, replace=True).mean() for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Win/loss/tie
    wins = int((deltas > 1e-9).sum())
    losses = int((deltas < -1e-9).sum())
    ties = n - wins - losses

    return {
        "n": n,
        "mean_before": float(b.mean()),
        "mean_after": float(a.mean()),
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "median_delta": float(np.median(deltas)),
        "t_stat": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_p),
        "cohen_dz": float(d_z),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rel_improvement_pct": (mean_d / b.mean() * 100.0) if b.mean() else float("nan"),
    }


def holm_bonferroni(pvalues: dict) -> dict:
    """Holm-Bonferroni correction across multiple endpoints."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i, (name, p) in enumerate(items):
        out[name] = min(1.0, p * (m - i))
    # enforce monotonicity
    prev = 0.0
    for name, _ in items:
        out[name] = max(out[name], prev)
        prev = out[name]
    return out


__all__ = ["analyze_paired", "holm_bonferroni"]
