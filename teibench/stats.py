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


def sign_test(before: Sequence[float], after: Sequence[float]) -> dict:
    """Exact binomial sign test on non-zero paired differences."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    pos = int((d > 1e-9).sum()); neg = int((d < -1e-9).sum()); ties = int(len(d) - pos - neg)
    m = pos + neg
    if m == 0:
        return {"pos": pos, "neg": neg, "ties": ties, "p_value": 1.0}
    p = float(stats.binomtest(min(pos, neg), m, 0.5, alternative="two-sided").pvalue)
    return {"pos": pos, "neg": neg, "ties": ties, "p_value": p}


def permutation_test(before: Sequence[float], after: Sequence[float],
                     iters: int = 20000, seed: int = 7) -> dict:
    """Paired sign-flip permutation test on the mean difference (two-sided)."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(iters, len(d)))
    null = (signs * np.abs(d)).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return {"observed_mean": obs, "p_value": p}


def tost_equivalence(before: Sequence[float], after: Sequence[float],
                     bound: float = 0.05) -> dict:
    """Two One-Sided Tests for equivalence: is the paired mean diff within +/-bound?
    Returns the larger of the two one-sided p-values (the TOST p); equivalent if < 0.05."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    n = len(d); mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / np.sqrt(n)
    if se == 0:
        return {"bound": bound, "p_tost": 0.0 if abs(mean) < bound else 1.0,
                "equivalent": abs(mean) < bound}
    t_lower = (mean - (-bound)) / se   # H0: diff <= -bound
    t_upper = (mean - bound) / se      # H0: diff >= +bound
    p_lower = float(stats.t.sf(t_lower, df=n - 1))       # one-sided, diff > -bound
    p_upper = float(stats.t.cdf(t_upper, df=n - 1))      # one-sided, diff < +bound
    p_tost = max(p_lower, p_upper)
    return {"bound": bound, "p_tost": p_tost, "equivalent": p_tost < 0.05}


def power_mde(before: Sequence[float], after: Sequence[float],
              alpha: float = 0.05, power: float = 0.80) -> dict:
    """Minimum detectable effect (paired) at given n, sd of diffs, alpha, power;
    plus post-hoc power for the observed effect. Normal approximation."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    n = len(d); sd = float(d.std(ddof=1)); mean = float(d.mean())
    z_a = stats.norm.ppf(1 - alpha / 2); z_b = stats.norm.ppf(power)
    mde = (z_a + z_b) * sd / np.sqrt(n) if n > 0 else float("nan")
    # post-hoc power for observed |mean|
    ncp = abs(mean) / (sd / np.sqrt(n)) if sd > 0 else float("inf")
    post_power = float(stats.norm.cdf(ncp - z_a) + stats.norm.cdf(-ncp - z_a))
    return {"n": n, "sd_delta": sd, "mde_80pct": float(mde),
            "observed_delta": mean, "post_hoc_power": min(1.0, post_power)}


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


__all__ = ["analyze_paired", "holm_bonferroni", "sign_test", "permutation_test",
           "tost_equivalence", "power_mde"]
