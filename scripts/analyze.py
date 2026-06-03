"""Analysis + figures + LaTeX tables from results/.

Produces:
  paper/figures/slope_objective.png      paired before→after slope chart
  paper/figures/delta_hist.png           distribution of per-agent objective deltas
  paper/figures/gpa_dimensions.png       per-dimension before/after bars
  paper/figures/headroom_scatter.png     improvement vs baseline difficulty
  paper/tables/per_agent.tex             per-agent results table
  paper/tables/aggregate.tex             aggregate statistics table
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.gpa_judge import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ACCENT = "#2563EB"
GREEN = "#16A34A"
GREY = "#94A3B8"


def load():
    recs = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        recs.append(json.loads(p.read_text()))
    summary = json.loads((RESULTS / "_full_summary.json").read_text())
    return recs, summary


def fig_slope(recs):
    recs = sorted(recs, key=lambda r: r["baseline_test"]["objective_mean"])
    fig, ax = plt.subplots(figsize=(6.5, 7))
    n_up = n_flat = n_down = 0
    for i, r in enumerate(recs):
        b = r["baseline_test"]["objective_mean"]
        a = r["final_test"]["objective_mean"]
        if a > b + 1e-9:
            color, n_up = GREEN, n_up + 1
        elif a < b - 1e-9:
            color, n_down = "#DC2626", n_down + 1
        else:
            color, n_flat = GREY, n_flat + 1
        ax.plot([0, 1], [b, a], "-", color=color, alpha=0.65, lw=1.8, zorder=1)
        ax.scatter([0, 1], [b, a], color=color, s=26, zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline\nprompt", "After TEI\n(held-out)"])
    ax.set_ylabel("Objective score (accuracy / EM)")
    ax.set_title("Per-agent objective score, baseline → TEI (held-out test)")
    ax.text(0.5, 0.04, f"{n_up} improved  ·  {n_flat} unchanged  ·  {n_down} declined",
            transform=ax.transAxes, ha="center", fontsize=10, color="#334155")
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.02, 1.04)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "slope_objective.png", dpi=150); plt.close(fig)


def fig_delta_hist(recs):
    deltas = [r["delta_objective"] for r in recs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(deltas, bins=12, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#475569", lw=1, ls="--")
    ax.axvline(float(np.mean(deltas)), color=GREEN, lw=2,
               label=f"mean Δ = {np.mean(deltas):+.3f}")
    ax.set_xlabel("Δ objective (TEI − baseline), per agent")
    ax.set_ylabel("# agents"); ax.legend()
    ax.set_title("Distribution of per-agent objective improvement")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "delta_hist.png", dpi=150); plt.close(fig)


def fig_gpa_dims(summary):
    dims = summary["gpa_dimensions"]
    names = [d.replace("_", "\n") for d in DIMENSIONS]
    before = [dims[d]["mean_before"] for d in DIMENSIONS]
    after = [dims[d]["mean_after"] for d in DIMENSIONS]
    x = np.arange(len(DIMENSIONS)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w/2, before, w, label="Baseline", color=GREY)
    ax.bar(x + w/2, after, w, label="After TEI", color=ACCENT)
    for i, d in enumerate(DIMENSIONS):
        ax.text(x[i] + w/2, after[i] + 0.01, f"{after[i]-before[i]:+.02f}",
                ha="center", fontsize=8, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("GPA dimension score"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("GPA dimensions, baseline → TEI (held-out, judge=Sonnet)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "gpa_dimensions.png", dpi=150); plt.close(fig)


def fig_headroom(recs):
    b = np.array([r["baseline_test"]["objective_mean"] for r in recs])
    d = np.array([r["delta_objective"] for r in recs])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(b, d, color=ACCENT, s=36, alpha=0.8)
    if len(b) >= 2:
        coef = np.polyfit(b, d, 1)
        xs = np.linspace(b.min(), b.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="#475569", ls="--", lw=1.3,
                label=f"slope={coef[0]:.2f}")
        ax.legend()
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("Baseline objective (task difficulty / headroom)")
    ax.set_ylabel("Δ objective from TEI")
    ax.set_title("TEI improvement vs. baseline headroom")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "headroom_scatter.png", dpi=150); plt.close(fig)


def _fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def table_per_agent(recs):
    recs = sorted(recs, key=lambda r: r["industry"])
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Agent & Industry & Base & TEI & $\Delta$obj & $\Delta$GPA \\", r"\midrule",
    ]
    for r in recs:
        tid = r["task_id"].replace("_", r"\_")
        ind = r["industry"].replace("&", r"\&")
        lines.append(
            f"\\texttt{{{tid}}} & {ind} & "
            f"{r['baseline_test']['objective_mean']:.2f} & "
            f"{r['final_test']['objective_mean']:.2f} & "
            f"{r['delta_objective']:+.2f} & {r['delta_gpa']:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "per_agent.tex").write_text("\n".join(lines), encoding="utf-8")


def table_aggregate(summary):
    def row(name, s):
        return (f"{name} & {s['mean_before']:.3f} & {s['mean_after']:.3f} & "
                f"{s['mean_delta']:+.3f} & [{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}] & "
                f"{s['t_stat']:.2f} & {_fmt_p(s['t_p_value'])} & {s['cohen_dz']:.2f} \\\\")
    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Endpoint & Base & TEI & $\Delta$ & 95\% CI & $t$ & $p$ & $d_z$ \\", r"\midrule",
        row("Objective (primary)", summary["objective"]),
        row("GPA aggregate", summary["gpa"]),
        r"\midrule",
    ]
    for d in DIMENSIONS:
        lines.append(row("\\quad " + d.replace("_", " "), summary["gpa_dimensions"][d]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "aggregate.tex").write_text("\n".join(lines), encoding="utf-8")


def emit_numbers(summary):
    o, g = summary["objective"], summary["gpa"]
    h = summary.get("headroom_subset") or {}
    hs = h.get("stats") or {}
    def p(x): return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"
    macros = {
        "nAgents": summary["n_agents"],
        "objBase": f"{o['mean_before']:.3f}", "objAfter": f"{o['mean_after']:.3f}",
        "objDelta": f"{o['mean_delta']:+.3f}", "objCILow": f"{o['ci95_low']:+.3f}",
        "objCIHigh": f"{o['ci95_high']:+.3f}", "objT": f"{o['t_stat']:.2f}",
        "objP": p(o["t_p_value"]), "objDz": f"{o['cohen_dz']:.2f}",
        "objWins": o["wins"], "objLosses": o["losses"], "objTies": o["ties"],
        "gpaBase": f"{g['mean_before']:.3f}", "gpaAfter": f"{g['mean_after']:.3f}",
        "gpaDelta": f"{g['mean_delta']:+.3f}", "gpaP": p(g["t_p_value"]),
        "headN": h.get("n", 0),
        "headDelta": f"{hs.get('mean_delta', 0):+.3f}" if hs else "n/a",
        "headP": p(hs["t_p_value"]) if hs else "n/a",
        "headDz": f"{hs.get('cohen_dz', 0):.2f}" if hs else "n/a",
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    recs, summary = load()
    print(f"Loaded {len(recs)} agent records.")
    fig_slope(recs); fig_delta_hist(recs); fig_gpa_dims(summary); fig_headroom(recs)
    table_per_agent(recs); table_aggregate(summary); emit_numbers(summary)
    print("Wrote figures →", FIG)
    print("Wrote tables  →", TAB)
    print("Wrote numbers → paper/numbers.tex")


if __name__ == "__main__":
    main()
