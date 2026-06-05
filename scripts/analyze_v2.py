"""Analysis for the v2 ablation experiment (results_v2/).

Computes, for the four arms (baseline / random / objective_reflection / tei):
  * mean held-out objective per arm
  * paired contrasts: tei-baseline, tei-random, tei-objref, random-baseline
    (paired t, Wilcoxon, Cohen d_z, bootstrap CI), Holm-corrected
  * GPA aggregate + 4 dimensions (baseline vs tei)
  * pre-specified subgroups: public-only, synthetic(authored)-only, headroom(<0.9)
Emits: figures, LaTeX tables, paper/numbers_v2.tex macros, results_v2/_summary.json,
       results_v2/per_agent.csv
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.stats import analyze_paired, holm_bonferroni
from teibench.gpa_judge import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
ARMS = ["baseline", "random", "objective_reflection", "tei"]
ARM_LABEL = {"baseline": "Baseline", "random": "Random search",
             "objective_reflection": "Obj-only reflection", "tei": "TEI (full)"}
ACCENT, GREEN, GREY, RED = "#2563EB", "#16A34A", "#94A3B8", "#DC2626"


def load():
    recs = [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))
            if not p.name.startswith("_")]
    return recs


def arm_obj(recs, arm):
    return [r["arms"][arm]["objective_mean"] for r in recs]


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    recs = load()
    n = len(recs)
    print(f"Loaded {n} tasks from results_v2/")
    assert n >= 2

    obj = {arm: arm_obj(recs, arm) for arm in ARMS}
    contrasts = {
        "tei_vs_baseline": ("tei", "baseline"),
        "tei_vs_random": ("tei", "random"),
        "tei_vs_objref": ("tei", "objective_reflection"),
        "random_vs_baseline": ("random", "baseline"),
        "objref_vs_baseline": ("objective_reflection", "baseline"),
    }
    cstats = {k: analyze_paired(obj[b], obj[a]) for k, (a, b) in contrasts.items()}
    holm = holm_bonferroni({k: v["t_p_value"] for k, v in cstats.items()})

    # GPA (baseline vs tei)
    bg = [r["arms"]["baseline"]["gpa_mean"] for r in recs]
    tg = [r["arms"]["tei"]["gpa_mean"] for r in recs]
    gpa_stats = analyze_paired(bg, tg)
    dim_stats = {}
    for d in DIMENSIONS:
        bd = [r["arms"]["baseline"]["gpa_dims"].get(d, 0.0) for r in recs]
        td = [r["arms"]["tei"]["gpa_dims"].get(d, 0.0) for r in recs]
        dim_stats[d] = analyze_paired(bd, td)

    # subgroups
    pub = [r for r in recs if r.get("is_public")]
    syn = [r for r in recs if not r.get("is_public")]
    sub = {}
    for name, group in [("public", pub), ("synthetic", syn)]:
        if len(group) >= 2:
            sub[name] = {"n": len(group),
                         "stats": analyze_paired(arm_obj(group, "baseline"), arm_obj(group, "tei"))}
    head_idx = [i for i, v in enumerate(obj["baseline"]) if v < 0.9]
    headroom = ({"n": len(head_idx),
                 "stats": analyze_paired([obj["baseline"][i] for i in head_idx],
                                         [obj["tei"][i] for i in head_idx])}
                if len(head_idx) >= 2 else None)

    summary = {
        "n_agents": n, "arm_means": {a: float(np.mean(obj[a])) for a in ARMS},
        "contrasts": cstats, "holm_corrected_p": holm,
        "gpa": gpa_stats, "gpa_dimensions": dim_stats,
        "subgroups": sub, "headroom_subset": headroom,
    }
    (RES / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figures ----
    # 1. arm means with bootstrap CI (across-agent mean +/- t CI)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    means = [np.mean(obj[a]) for a in ARMS]
    sems = [np.std(obj[a], ddof=1) / np.sqrt(n) for a in ARMS]
    cols = [GREY, "#a78bfa", "#60a5fa", ACCENT]
    ax.bar(range(len(ARMS)), means, yerr=[1.96 * s for s in sems], capsize=5, color=cols)
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("Held-out objective (mean ± 95% CI)"); ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: held-out objective by optimization arm (n=%d)" % n)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "ablation_arms.png", dpi=150); plt.close(fig)

    # 2. slope baseline -> tei
    order = sorted(range(n), key=lambda i: obj["baseline"][i])
    fig, ax = plt.subplots(figsize=(6, 6.5))
    up = flat = down = 0
    for i in order:
        b, a = obj["baseline"][i], obj["tei"][i]
        c = GREEN if a > b + 1e-9 else (RED if a < b - 1e-9 else GREY)
        up += a > b + 1e-9; down += a < b - 1e-9; flat += abs(a - b) <= 1e-9
        ax.plot([0, 1], [b, a], "-", color=c, alpha=0.65, lw=1.7)
        ax.scatter([0, 1], [b, a], color=c, s=24)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "TEI (held-out)"])
    ax.set_ylabel("Objective score"); ax.set_ylim(-0.02, 1.04); ax.set_xlim(-0.25, 1.25)
    ax.text(0.5, 0.03, f"{up} improved · {flat} unchanged · {down} declined",
            transform=ax.transAxes, ha="center", fontsize=10, color="#334155")
    ax.set_title("Per-agent objective, baseline → TEI"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "slope_objective_v2.png", dpi=150); plt.close(fig)

    # 3. public vs synthetic
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [("Public\n(n=%d)" % len(pub), pub), ("Synthetic\n(n=%d)" % len(syn), syn)]
    xs = np.arange(len(groups)); w = 0.38
    ax.bar(xs - w/2, [np.mean(arm_obj(g, "baseline")) for _, g in groups], w, label="Baseline", color=GREY)
    ax.bar(xs + w/2, [np.mean(arm_obj(g, "tei")) for _, g in groups], w, label="TEI", color=ACCENT)
    ax.set_xticks(xs); ax.set_xticklabels([nm for nm, _ in groups]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out objective"); ax.legend(); ax.set_title("Public vs. synthetic tasks")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "public_vs_synthetic.png", dpi=150); plt.close(fig)

    # ---- tables ----
    def row(label, s):
        return (f"{label} & {s['mean_before']:.3f} & {s['mean_after']:.3f} & {s['mean_delta']:+.3f} & "
                f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}] & {s['t_stat']:.2f} & {fmt_p(s['t_p_value'])} "
                f"& {s['cohen_dz']:.2f} \\\\")
    abl = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r"Contrast & A & B & $\Delta$ & 95\% CI & $t$ & $p$ & $d_z$ \\", r"\midrule",
           row("TEI $-$ baseline", cstats["tei_vs_baseline"]),
           row("TEI $-$ random", cstats["tei_vs_random"]),
           row("TEI $-$ obj-only refl.", cstats["tei_vs_objref"]),
           row("random $-$ baseline", cstats["random_vs_baseline"]),
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "ablation.tex").write_text("\n".join(abl), encoding="utf-8")

    # per-agent (all arms)
    pa = [r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Agent & Domain & Base & Rand & ObjRef & TEI \\", r"\midrule"]
    for r in sorted(recs, key=lambda x: x["industry"]):
        tid = r["task_id"].replace("_", r"\_"); ind = r["industry"].replace("&", r"\&")
        a = r["arms"]
        pa.append(f"\\texttt{{{tid}}} & {ind} & {a['baseline']['objective_mean']:.2f} & "
                  f"{a['random']['objective_mean']:.2f} & {a['objective_reflection']['objective_mean']:.2f} & "
                  f"{a['tei']['objective_mean']:.2f} \\\\")
    pa += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "per_agent_v2.tex").write_text("\n".join(pa), encoding="utf-8")

    with (RES / "per_agent.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "domain", "is_public", "n_test", "obj_baseline", "obj_random",
                    "obj_objref", "obj_tei", "tei_minus_baseline", "gpa_baseline", "gpa_tei"])
        for r in recs:
            a = r["arms"]
            w.writerow([r["task_id"], r["industry"], r["is_public"], r["n_test"],
                        f"{a['baseline']['objective_mean']:.4f}", f"{a['random']['objective_mean']:.4f}",
                        f"{a['objective_reflection']['objective_mean']:.4f}", f"{a['tei']['objective_mean']:.4f}",
                        f"{a['tei']['objective_mean']-a['baseline']['objective_mean']:.4f}",
                        f"{a['baseline'].get('gpa_mean',0):.4f}", f"{a['tei'].get('gpa_mean',0):.4f}"])

    # ---- macros for the paper ----
    tb = cstats["tei_vs_baseline"]; tr = cstats["tei_vs_random"]; to = cstats["tei_vs_objref"]
    rb = cstats["random_vs_baseline"]
    m = {
        "nAgents": n,
        "armBase": f"{np.mean(obj['baseline']):.3f}", "armRand": f"{np.mean(obj['random']):.3f}",
        "armObjref": f"{np.mean(obj['objective_reflection']):.3f}", "armTei": f"{np.mean(obj['tei']):.3f}",
        "teiBaseDelta": f"{tb['mean_delta']:+.3f}", "teiBaseCIlo": f"{tb['ci95_low']:+.3f}",
        "teiBaseCIhi": f"{tb['ci95_high']:+.3f}", "teiBaseT": f"{tb['t_stat']:.2f}",
        "teiBaseP": fmt_p(tb['t_p_value']), "teiBaseDz": f"{tb['cohen_dz']:.2f}",
        "teiBaseW": tb['wins'], "teiBaseL": tb['losses'], "teiBaseTie": tb['ties'],
        "teiRandDelta": f"{tr['mean_delta']:+.3f}", "teiRandP": fmt_p(tr['t_p_value']), "teiRandDz": f"{tr['cohen_dz']:.2f}",
        "teiObjrefDelta": f"{to['mean_delta']:+.3f}", "teiObjrefP": fmt_p(to['t_p_value']), "teiObjrefDz": f"{to['cohen_dz']:.2f}",
        "randBaseDelta": f"{rb['mean_delta']:+.3f}", "randBaseP": fmt_p(rb['t_p_value']),
        "gpaBase": f"{gpa_stats['mean_before']:.3f}", "gpaTei": f"{gpa_stats['mean_after']:.3f}",
        "gpaDelta": f"{gpa_stats['mean_delta']:+.3f}", "gpaP": fmt_p(gpa_stats['t_p_value']),
        "holmTeiBaseP": fmt_p(holm['tei_vs_baseline']), "holmTeiRandP": fmt_p(holm['tei_vs_random']),
        "holmTeiObjrefP": fmt_p(holm['tei_vs_objref']),
        "pubN": len(pub), "synN": len(syn),
    }
    if "public" in sub:
        m["pubDelta"] = f"{sub['public']['stats']['mean_delta']:+.3f}"
        m["pubP"] = fmt_p(sub['public']['stats']['t_p_value'])
    if "synthetic" in sub:
        m["synDelta"] = f"{sub['synthetic']['stats']['mean_delta']:+.3f}"
        m["synP"] = fmt_p(sub['synthetic']['stats']['t_p_value'])
    if headroom:
        m["headN"] = headroom["n"]; m["headDelta"] = f"{headroom['stats']['mean_delta']:+.3f}"
        m["headP"] = fmt_p(headroom['stats']['t_p_value']); m["headDz"] = f"{headroom['stats']['cohen_dz']:.2f}"
    (ROOT / "paper" / "numbers_v2.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")

    print("== v2 summary ==")
    print(f"arm means: " + "  ".join(f"{a}={np.mean(obj[a]):.3f}" for a in ARMS))
    for k, c in cstats.items():
        print(f"  {k:20} d={c['mean_delta']:+.3f} p={fmt_p(c['t_p_value'])} dz={c['cohen_dz']:.2f} "
              f"holm={fmt_p(holm[k])} W/L/T={c['wins']}/{c['losses']}/{c['ties']}")
    print("Wrote figures, tables, numbers_v2.tex, _summary.json, per_agent.csv")


if __name__ == "__main__":
    main()
