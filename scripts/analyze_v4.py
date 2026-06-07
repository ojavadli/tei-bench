"""Analyze the TWO-PHASE run (results_v4/): baseline vs two_phase on all 31 agents,
plus the per-phase attribution (structural-only / structural+prompt / baseline) and a
do-no-harm slope (baseline -> two_phase: up / tie / down).

Emits paper/tables/teiv4.tex, paper/numbers_v4.tex, paper/figures/{teiv4_slope,
teiv4_arms}.png, results_v4/_summary.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.stats import analyze_paired, sign_test, permutation_test

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.dpi": 200,
})
GREEN, RED, GREY, NAVY = "#2f7d4f", "#a13b3b", "#9aa6b2", "#27496d"
fmt_p = lambda p: f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_v4")
    args = ap.parse_args()
    RES = ROOT / args.results_dir
    recs = {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())
            for p in sorted(RES.glob("*.json")) if not p.name.startswith("_")}
    if not recs:
        print(f"{args.results_dir}/ empty — run run_experiment_v4.py first."); return
    tids = sorted(recs); n = len(tids)
    base = [recs[t]["arms"]["baseline"]["objective_mean"] for t in tids]
    tp = [recs[t]["arms"]["two_phase"]["objective_mean"] for t in tids]

    c = analyze_paired(base, tp)
    sign = sign_test(base, tp); perm = permutation_test(base, tp)
    up = sum(1 for i in range(n) if tp[i] > base[i] + 1e-9)
    down = sum(1 for i in range(n) if tp[i] < base[i] - 1e-9)
    tie = n - up - down
    phases = Counter(recs[t]["arms"]["two_phase"]["optimization"]["deployed_phase"] for t in tids)
    # how often did the prompt phase add over the structural best (on validation)?
    prompt_added = sum(1 for t in tids
                       if recs[t]["arms"]["two_phase"]["optimization"]["prompt_best_val"]
                       > recs[t]["arms"]["two_phase"]["optimization"]["struct_best_val"] + 1e-9)
    hidx = [i for i, v in enumerate(base) if v < 0.9]
    head = analyze_paired([base[i] for i in hidx], [tp[i] for i in hidx]) if len(hidx) >= 2 else None

    summary = {"n": n, "baseline_mean": float(np.mean(base)), "two_phase_mean": float(np.mean(tp)),
               "contrast": c, "sign": sign, "perm": perm, "up": up, "down": down, "tie": tie,
               "deployed_phase_counts": dict(phases), "prompt_added_over_struct": prompt_added,
               "headroom": ({"n": len(hidx), "stats": head} if head else None),
               "task_ids": tids}
    (RES / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # slope baseline -> two_phase
    task_order = sorted(range(n), key=lambda i: base[i])
    fig, ax = plt.subplots(figsize=(3.4, 3.8))
    for i in task_order:
        b, a = base[i], tp[i]
        col = GREEN if a > b + 1e-9 else (RED if a < b - 1e-9 else GREY)
        ax.plot([0, 1], [b, a], "-", color=col, alpha=0.6, lw=1.2)
        ax.scatter([0, 1], [b, a], color=col, s=14, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "TEI two-phase"]); ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 1.04); ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Per-task baseline $\\rightarrow$ two-phase TEI (30+30)\n{up} up \u00b7 {tie} tie \u00b7 {down} down",
                 fontsize=8.5); ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(FIG / "teiv4_slope.png"); plt.close(fig)

    # arms bar (truncated)
    means = [np.mean(base), np.mean(tp)]
    ci = [1.96 * np.std(base, ddof=1) / np.sqrt(n), 1.96 * np.std(tp, ddof=1) / np.sqrt(n)]
    lo = min(m - x for m, x in zip(means, ci)); hi = max(m + x for m, x in zip(means, ci)); pad = (hi - lo) * 0.2 + 0.01
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.bar([0, 1], means, yerr=ci, capsize=4, color=[GREY, NAVY], edgecolor="#333", linewidth=0.5,
           error_kw={"elinewidth": 0.9, "ecolor": "#333"})
    for i, m in enumerate(means):
        ax.text(i, means[i] + ci[i] + pad * 0.15, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Two-phase TEI"])
    ax.set_ylim(max(0, lo - pad), hi + pad); ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Two-phase TEI vs baseline ($n={n}$)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "teiv4_arms.png"); plt.close(fig)

    # table
    tab = [r"\begin{tabular}{lr}", r"\toprule", f"Arm (two-phase, $n={n}$) & Held-out acc. \\\\",
           r"\midrule", f"Baseline & {np.mean(base):.3f} \\\\",
           f"Two-phase TEI (30+30) & {np.mean(tp):.3f} \\\\", r"\bottomrule", r"\end{tabular}",
           r"", r"\vspace{4pt}", r"",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Contrast & $\Delta$ & 95\% CI & $p$ & $d_z$ \\", r"\midrule",
           (f"Two-phase $-$ baseline & {c['mean_delta']:+.3f} & "
            f"[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}] & {fmt_p(c['t_p_value'])} & {c['cohen_dz']:.2f} \\\\"),
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "teiv4.tex").write_text("\n".join(tab), encoding="utf-8")

    m = {"vfourN": n, "vfourBase": f"{np.mean(base):.3f}", "vfourTei": f"{np.mean(tp):.3f}",
         "vfourDelta": f"{c['mean_delta']:+.3f}", "vfourP": fmt_p(c['t_p_value']),
         "vfourDz": f"{c['cohen_dz']:.2f}", "vfourCIlo": f"{c['ci95_low']:+.3f}",
         "vfourCIhi": f"{c['ci95_high']:+.3f}", "vfourUp": up, "vfourDown": down, "vfourTie": tie,
         "vfourSignP": fmt_p(sign["p_value"]), "vfourStructIters": 30, "vfourPromptIters": 30,
         "vfourPromptAdded": prompt_added}
    if head:
        m.update({"vfourHeadN": len(hidx), "vfourHeadDelta": f"{head['mean_delta']:+.3f}",
                  "vfourHeadP": fmt_p(head['t_p_value']), "vfourHeadDz": f"{head['cohen_dz']:.2f}"})
    (ROOT / "paper" / "numbers_v4.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")

    print(f"== Two-phase (n={n}) ==  baseline={np.mean(base):.3f} two_phase={np.mean(tp):.3f}")
    print(f"  delta={c['mean_delta']:+.3f} p={fmt_p(c['t_p_value'])} dz={c['cohen_dz']:.2f} "
          f"W/L/T={up}/{down}/{tie} sign_p={fmt_p(sign['p_value'])}")
    print(f"  deployed phases: {dict(phases)}")
    print(f"  prompt phase added over structural on: {prompt_added}/{n}")
    if head:
        print(f"  headroom n={len(hidx)}: delta={head['mean_delta']:+.3f} p={fmt_p(head['t_p_value'])} dz={head['cohen_dz']:.2f}")
    print("Wrote tables/teiv4.tex, numbers_v4.tex, figures, _summary.json")


if __name__ == "__main__":
    main()
