"""Analyze TEI v3 on the FULL suite (results_v3_full/) against the recorded v2 arms
(results_v2/: baseline, random, objective_reflection, tei) on all 31 tasks.

This is the >=31-experiment, full-suite test of the redesigned loop. Reports:
  * arm means (baseline / random / obj-only / TEI v2 / TEI v3)
  * paired contrasts tei_v3 vs baseline / random / tei(v2) / objref (n=31)
  * win/loss/tie and whether do-no-harm removed regressions
  * headroom subset breakdown
  * efficiency: tasks triaged, tasks where the posterior gate shipped baseline,
    judge evaluations saved by successive halving

Emits paper/tables/teiv3.tex, paper/numbers_v3.tex, paper/figures/teiv3_arms.png,
paper/figures/teiv3_forest.png, results_v3_full/_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.stats import analyze_paired, sign_test, permutation_test, tost_equivalence

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results_v2"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})
PALETTE = {"baseline": "#9aa6b2", "random": "#7e9bbd", "objective_reflection": "#6a8fae",
           "tei": "#5a86b0", "tei_v3": "#27496d"}
LABEL = {"baseline": "Baseline", "random": "Random", "objective_reflection": "Obj-only",
         "tei": "TEI (v2)", "tei_v3": "TEI v3"}
GREEN, RED, GREY = "#2f7d4f", "#a13b3b", "#9aa6b2"
fmt_p = lambda p: f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def load(d):
    return {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())
            for p in sorted(d.glob("*.json")) if not p.name.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_v3_full")
    args = ap.parse_args()
    V3 = ROOT / args.results_dir
    v3 = load(V3); v2 = load(V2)
    if not v3:
        print(f"{args.results_dir}/ empty — run run_experiment_v3.py --all first."); return

    tids = sorted(set(v3) & set(v2))
    n = len(tids)
    print(f"Full-suite v3: {len(v3)} v3 tasks, joined with v2 -> n={n}")

    obj = {a: [v2[t]["arms"][a]["objective_mean"] for t in tids]
           for a in ["baseline", "random", "objective_reflection", "tei"]}
    obj["tei_v3"] = [v3[t]["arms"]["tei_v3"]["objective_mean"] for t in tids]

    contrasts = {
        "teiv3_vs_baseline": ("tei_v3", "baseline"),
        "teiv3_vs_random": ("tei_v3", "random"),
        "teiv3_vs_tei": ("tei_v3", "tei"),
        "teiv3_vs_objref": ("tei_v3", "objective_reflection"),
    }
    cs = {k: analyze_paired(obj[b], obj[a]) for k, (a, b) in contrasts.items()}
    extra = {"sign_vs_baseline": sign_test(obj["baseline"], obj["tei_v3"]),
             "perm_vs_baseline": permutation_test(obj["baseline"], obj["tei_v3"]),
             "sign_vs_random": sign_test(obj["random"], obj["tei_v3"]),
             "perm_vs_random": permutation_test(obj["random"], obj["tei_v3"])}

    # headroom subset (baseline < 0.9)
    hidx = [i for i, v in enumerate(obj["baseline"]) if v < 0.9]
    head = None
    if len(hidx) >= 2:
        hb = [obj["baseline"][i] for i in hidx]; ht = [obj["tei_v3"][i] for i in hidx]
        hr = [obj["random"][i] for i in hidx]
        head = {"n": len(hidx), "vs_baseline": analyze_paired(hb, ht),
                "vs_random": analyze_paired(hr, ht),
                "task_ids": [tids[i] for i in hidx]}

    # efficiency
    triaged = sum(1 for t in v3 if v3[t]["arms"]["tei_v3"]["optimization"].get("triaged"))
    ship_base = sum(1 for t in v3 if v3[t]["arms"]["tei_v3"]["optimization"].get("shipped_baseline")
                    or v3[t]["arms"]["tei_v3"]["optimization"].get("triaged"))
    judge_saved = sum(v3[t]["arms"]["tei_v3"]["optimization"].get("judge_calls_saved", 0) for t in v3)
    # regressions: tasks where tei_v3 < baseline on test
    regress = sum(1 for i in range(n) if obj["tei_v3"][i] < obj["baseline"][i] - 1e-9)

    arms_means = {a: float(np.mean(obj[a])) for a in
                  ["baseline", "random", "objective_reflection", "tei", "tei_v3"]}
    summary = {"n": n, "arm_means": arms_means, "contrasts": cs, "extra": extra,
               "headroom": head, "n_triaged": triaged, "n_shipped_baseline": ship_base,
               "judge_calls_saved": judge_saved, "n_regressions": regress, "task_ids": tids}
    (V3 / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------- figure: full-suite arm means (truncated) ----------
    order = ["baseline", "random", "objective_reflection", "tei", "tei_v3"]
    means = [np.mean(obj[a]) for a in order]
    ci = [1.96 * np.std(obj[a], ddof=1) / np.sqrt(n) for a in order]
    lo = min(m - c for m, c in zip(means, ci)); hi = max(m + c for m, c in zip(means, ci))
    pad = (hi - lo) * 0.14
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(range(len(order)), means, yerr=ci, capsize=3.5,
           color=[PALETTE[a] for a in order], edgecolor="#333", linewidth=0.5,
           error_kw={"elinewidth": 0.9, "ecolor": "#333"})
    for i, m in enumerate(means):
        ax.text(i, means[i] + ci[i] + pad * 0.15, f"{m:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[a] for a in order], rotation=18, ha="right")
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Full suite ($n={n}$): TEI v3 vs.\\ all arms", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "teiv3_arms.png"); plt.close(fig)

    # ---------- forest: per-task tei_v3 - baseline (full suite) ----------
    deltas = [(tids[i], obj["tei_v3"][i] - obj["baseline"][i]) for i in range(n)]
    deltas = [d for d in deltas if abs(d[1]) > 1e-9]
    deltas.sort(key=lambda x: x[1])
    if deltas:
        fig, ax = plt.subplots(figsize=(3.4, max(2.4, 0.22 * len(deltas) + 0.8)))
        colors = [GREEN if d > 0 else RED for _, d in deltas]
        ax.barh(range(len(deltas)), [d for _, d in deltas], color=colors,
                edgecolor="#333", linewidth=0.4)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(range(len(deltas))); ax.set_yticklabels([t for t, _ in deltas], fontsize=6.5)
        ax.set_xlabel("TEI v3 $-$ baseline (held-out)")
        nz = len(deltas); up = sum(1 for _, d in deltas if d > 0)
        ax.set_title(f"Per-task TEI v3 $-$ baseline ({up}/{nz} moved up; rest triaged to 0)", fontsize=8)
        ax.grid(axis="y", alpha=0)
        fig.tight_layout(); fig.savefig(FIG / "teiv3_forest.png"); plt.close(fig)

    # ---- slope: baseline -> TEI v3 (do-no-harm: every line is up or flat, none down) ----
    task_order = sorted(range(n), key=lambda i: obj["baseline"][i])
    up = down = flat = 0
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    for i in task_order:
        b, a = obj["baseline"][i], obj["tei_v3"][i]
        c = GREEN if a > b + 1e-9 else (RED if a < b - 1e-9 else GREY)
        up += a > b + 1e-9; down += a < b - 1e-9; flat += abs(a - b) <= 1e-9
        ax.plot([0, 1], [b, a], "-", color=c, alpha=0.6, lw=1.2)
        ax.scatter([0, 1], [b, a], color=c, s=14, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "TEI v3"]); ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 1.04); ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Per-task baseline $\\rightarrow$ TEI v3 (do-no-harm)\n"
                 f"{up} up \u00b7 {flat} tie \u00b7 {down} down", fontsize=8.5)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(FIG / "slope_v3.png"); plt.close(fig)

    # ---------- table ----------
    def crow(label, s):
        return (f"{label} & {s['mean_delta']:+.3f} & [{s['ci95_low']:+.3f},{s['ci95_high']:+.3f}] "
                f"& {fmt_p(s['t_p_value'])} & {s['cohen_dz']:.2f} \\\\")
    rows = "\n".join(f"{LABEL[a]} & {np.mean(obj[a]):.3f} \\\\" for a in order)
    tab = [r"\begin{tabular}{lr}", r"\toprule",
           f"Arm (full suite, $n={n}$) & Held-out acc. \\\\", r"\midrule", rows,
           r"\bottomrule", r"\end{tabular}", r"", r"\vspace{4pt}", r"",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Contrast & $\Delta$ & 95\% CI & $p$ & $d_z$ \\", r"\midrule",
           crow("TEI v3 $-$ baseline", cs["teiv3_vs_baseline"]),
           crow("TEI v3 $-$ random", cs["teiv3_vs_random"]),
           crow("TEI v3 $-$ TEI (v2)", cs["teiv3_vs_tei"]),
           crow("TEI v3 $-$ obj-only", cs["teiv3_vs_objref"]),
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "teiv3.tex").write_text("\n".join(tab), encoding="utf-8")

    tb, tr, tt, to = (cs["teiv3_vs_baseline"], cs["teiv3_vs_random"],
                      cs["teiv3_vs_tei"], cs["teiv3_vs_objref"])
    m = {
        "vthreeN": n,
        "vthreeBase": f"{arms_means['baseline']:.3f}", "vthreeRand": f"{arms_means['random']:.3f}",
        "vthreeObjref": f"{arms_means['objective_reflection']:.3f}",
        "vthreeTeiTwo": f"{arms_means['tei']:.3f}", "vthreeTeiV": f"{arms_means['tei_v3']:.3f}",
        "vthreeBaseDelta": f"{tb['mean_delta']:+.3f}", "vthreeBaseP": fmt_p(tb['t_p_value']),
        "vthreeBaseDz": f"{tb['cohen_dz']:.2f}", "vthreeBaseCIlo": f"{tb['ci95_low']:+.3f}",
        "vthreeBaseCIhi": f"{tb['ci95_high']:+.3f}",
        "vthreeBaseW": tb['wins'], "vthreeBaseL": tb['losses'], "vthreeBaseT": tb['ties'],
        "vthreeSignBaseP": fmt_p(extra['sign_vs_baseline']['p_value']),
        "vthreeRandDelta": f"{tr['mean_delta']:+.3f}", "vthreeRandP": fmt_p(tr['t_p_value']),
        "vthreeRandDz": f"{tr['cohen_dz']:.2f}",
        "vthreeTeiDelta": f"{tt['mean_delta']:+.3f}", "vthreeTeiP": fmt_p(tt['t_p_value']),
        "vthreeObjrefDelta": f"{to['mean_delta']:+.3f}", "vthreeObjrefP": fmt_p(to['t_p_value']),
        "vthreeTriaged": triaged, "vthreeShipBase": ship_base, "vthreeJudgeSaved": judge_saved,
        "vthreeRegress": regress,
    }
    if head:
        hb_, hr_ = head["vs_baseline"], head["vs_random"]
        m.update({"vthreeHeadN": head["n"],
                  "vthreeHeadDelta": f"{hb_['mean_delta']:+.3f}", "vthreeHeadP": fmt_p(hb_['t_p_value']),
                  "vthreeHeadDz": f"{hb_['cohen_dz']:.2f}",
                  "vthreeHeadRandDelta": f"{hr_['mean_delta']:+.3f}", "vthreeHeadRandP": fmt_p(hr_['t_p_value'])})
    (ROOT / "paper" / "numbers_v3.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")

    print("== TEI v3 (full suite, n=%d) ==" % n)
    print("arm means:", {a: round(arms_means[a], 3) for a in order})
    for k, c in cs.items():
        print(f"  {k:20} d={c['mean_delta']:+.3f} p={fmt_p(c['t_p_value'])} dz={c['cohen_dz']:.2f} "
              f"W/L/T={c['wins']}/{c['losses']}/{c['ties']}")
    print(f"  regressions (tei_v3<baseline on test): {regress}; triaged={triaged}; "
          f"shipped_baseline={ship_base}; judge_saved={judge_saved}")
    if head:
        print(f"  headroom n={head['n']}: vs baseline d={head['vs_baseline']['mean_delta']:+.3f} "
              f"p={fmt_p(head['vs_baseline']['t_p_value'])} dz={head['vs_baseline']['cohen_dz']:.2f}; "
              f"vs random d={head['vs_random']['mean_delta']:+.3f} p={fmt_p(head['vs_random']['t_p_value'])}")
    print("Wrote tables/teiv3.tex, numbers_v3.tex, figures, _summary.json")


if __name__ == "__main__":
    main()
