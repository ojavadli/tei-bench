"""Analysis for the v2 ablation experiment (results_v2/).

Four arms (baseline / random / objective_reflection / tei):
  * mean held-out objective per arm
  * paired contrasts (paired t, Wilcoxon, sign test, permutation test, Cohen d_z,
    bootstrap CI), Holm-corrected
  * equivalence (TOST) + power/MDE for the headline contrasts (n=31 and headroom n)
  * GPA aggregate + 4 dimensions (baseline vs tei)
  * subgroups: public-only, synthetic-only, headroom(<0.9)
Emits arXiv-styled figures (truncated axes), LaTeX tables, numbers_v2.tex, summary, csv.
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
from teibench.stats import (analyze_paired, holm_bonferroni, sign_test,
                            permutation_test, tost_equivalence, power_mde)
from teibench.gpa_judge import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
ARMS = ["baseline", "random", "objective_reflection", "tei"]
ARM_LABEL = {"baseline": "Baseline", "random": "Random\nsearch",
             "objective_reflection": "Obj-only\nreflection", "tei": "TEI\n(full)"}

# arXiv-style matplotlib defaults: serif (matches Computer Modern), muted palette.
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.dpi": 200,
})
PALETTE = ["#9aa6b2", "#7e9bbd", "#5a86b0", "#27496d"]  # gray -> navy (muted)
NAVY, GREEN, RED, GREY = "#27496d", "#2f7d4f", "#a13b3b", "#9aa6b2"


def load():
    return [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))
            if not p.name.startswith("_")]


def arm_obj(recs, arm):
    return [r["arms"][arm]["objective_mean"] for r in recs]


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    recs = load()
    n = len(recs)
    print(f"Loaded {n} tasks from results_v2/")
    obj = {arm: arm_obj(recs, arm) for arm in ARMS}

    contrasts_def = {
        "tei_vs_baseline": ("tei", "baseline"), "tei_vs_random": ("tei", "random"),
        "tei_vs_objref": ("tei", "objective_reflection"),
        "random_vs_baseline": ("random", "baseline"),
        "objref_vs_baseline": ("objective_reflection", "baseline"),
    }
    cstats = {k: analyze_paired(obj[b], obj[a]) for k, (a, b) in contrasts_def.items()}
    holm = holm_bonferroni({k: v["t_p_value"] for k, v in cstats.items()})

    # Extra robustness on the headline contrast (TEI vs baseline), full suite
    tb_a, tb_b = obj["tei"], obj["baseline"]
    extra_full = {
        "sign": sign_test(tb_b, tb_a), "perm": permutation_test(tb_b, tb_a),
        "tost05": tost_equivalence(tb_b, tb_a, bound=0.05),
        "power": power_mde(tb_b, tb_a),
    }

    # GPA
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
    for name, g in [("public", pub), ("synthetic", syn)]:
        if len(g) >= 2:
            sub[name] = {"n": len(g), "stats": analyze_paired(arm_obj(g, "baseline"), arm_obj(g, "tei"))}
    head_idx = [i for i, v in enumerate(obj["baseline"]) if v < 0.9]
    headroom = None
    if len(head_idx) >= 2:
        hb = [obj["baseline"][i] for i in head_idx]; ht = [obj["tei"][i] for i in head_idx]
        headroom = {"n": len(head_idx), "stats": analyze_paired(hb, ht),
                    "tost05": tost_equivalence(hb, ht, 0.05), "power": power_mde(hb, ht),
                    "sign": sign_test(hb, ht), "perm": permutation_test(hb, ht),
                    "task_ids": [recs[i]["task_id"] for i in head_idx]}
    n_ceiling = int(sum(1 for v in obj["baseline"] if v >= 0.999))

    summary = {
        "n_agents": n, "arm_means": {a: float(np.mean(obj[a])) for a in ARMS},
        "contrasts": cstats, "holm_corrected_p": holm, "extra_full": extra_full,
        "gpa": gpa_stats, "gpa_dimensions": dim_stats, "subgroups": sub,
        "headroom_subset": headroom, "n_ceiling_tasks": n_ceiling,
    }
    (RES / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------- FIGURES (arXiv style, truncated axes) ----------
    means = [np.mean(obj[a]) for a in ARMS]
    ci = [1.96 * np.std(obj[a], ddof=1) / np.sqrt(n) for a in ARMS]
    lo = min(m - c for m, c in zip(means, ci)); hi = max(m + c for m, c in zip(means, ci))
    pad = (hi - lo) * 0.12
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    bars = ax.bar(range(4), means, yerr=ci, capsize=3.5, color=PALETTE,
                  edgecolor="#333", linewidth=0.5, error_kw={"elinewidth": 0.9, "ecolor": "#333"})
    for i, m in enumerate(means):
        ax.text(i, means[i] + ci[i] + pad * 0.18, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_ylim(max(0.0, lo - pad), hi + pad)   # truncated so the small differences are visible
    ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Held-out accuracy by optimization arm ($n={n}$)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "ablation_arms.png"); plt.close(fig)

    # slope baseline->tei
    order = sorted(range(n), key=lambda i: obj["baseline"][i])
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    up = down = flat = 0
    for i in order:
        b, a = obj["baseline"][i], obj["tei"][i]
        c = GREEN if a > b + 1e-9 else (RED if a < b - 1e-9 else GREY)
        up += a > b + 1e-9; down += a < b - 1e-9; flat += abs(a - b) <= 1e-9
        ax.plot([0, 1], [b, a], "-", color=c, alpha=0.6, lw=1.2)
        ax.scatter([0, 1], [b, a], color=c, s=14, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "TEI"]); ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 1.04); ax.set_ylabel("Held-out objective accuracy")
    ax.set_title(f"Per-task baseline $\\rightarrow$ TEI\n{up} up · {flat} tie · {down} down", fontsize=8.5)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(FIG / "slope_objective_v2.png"); plt.close(fig)

    # public vs synthetic (truncated)
    groups = [("Public\n(n=%d)" % len(pub), pub), ("Synthetic\n(n=%d)" % len(syn), syn)]
    xs = np.arange(2); w = 0.36
    bm = [np.mean(arm_obj(g, "baseline")) for _, g in groups]
    tm = [np.mean(arm_obj(g, "tei")) for _, g in groups]
    allv = bm + tm; glo, ghi = min(allv), max(allv); gpad = (ghi - glo) * 0.5 + 0.02
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.bar(xs - w/2, bm, w, label="Baseline", color=PALETTE[0], edgecolor="#333", linewidth=0.5)
    ax.bar(xs + w/2, tm, w, label="TEI", color=PALETTE[3], edgecolor="#333", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([nm for nm, _ in groups])
    ax.set_ylim(max(0.0, glo - gpad), ghi + gpad)
    ax.set_ylabel("Held-out objective accuracy"); ax.legend(frameon=False)
    ax.set_title("Public vs.\\ synthetic tasks (y-axis truncated)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "public_vs_synthetic.png"); plt.close(fig)

    # headroom forest: per-task delta with task labels (headroom subset)
    if headroom:
        hidx = sorted(head_idx, key=lambda i: obj["tei"][i] - obj["baseline"][i])
        deltas = [obj["tei"][i] - obj["baseline"][i] for i in hidx]
        labels = [recs[i]["task_id"] for i in hidx]
        fig, ax = plt.subplots(figsize=(3.4, max(2.4, 0.22 * len(hidx) + 0.8)))
        colors = [GREEN if d > 0 else (RED if d < 0 else GREY) for d in deltas]
        ax.barh(range(len(hidx)), deltas, color=colors, edgecolor="#333", linewidth=0.4)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(range(len(hidx))); ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("TEI $-$ baseline (held-out)")
        hs = headroom["stats"]
        ax.set_title(f"Headroom subset ($n={headroom['n']}$): mean $\\Delta={hs['mean_delta']:+.3f}$\n"
                     f"$p={fmt_p(hs['t_p_value'])}$, $d_z={hs['cohen_dz']:.2f}$", fontsize=8)
        ax.grid(axis="y", alpha=0)
        fig.tight_layout(); fig.savefig(FIG / "headroom_forest.png"); plt.close(fig)

    # ---------- TABLES ----------
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

    pa = [r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Agent & Domain & Base & Rand & ObjRef & TEI \\", r"\midrule"]
    for r in sorted(recs, key=lambda x: x["industry"]):
        tid = r["task_id"].replace("_", r"\_"); ind = r["industry"].replace("&", r"\&")
        a = r["arms"]
        pa.append(f"\\texttt{{\\small {tid}}} & {ind} & {a['baseline']['objective_mean']:.2f} & "
                  f"{a['random']['objective_mean']:.2f} & {a['objective_reflection']['objective_mean']:.2f} & "
                  f"{a['tei']['objective_mean']:.2f} \\\\")
    pa += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "per_agent_v2.tex").write_text("\n".join(pa), encoding="utf-8")

    DIM_LABEL = {"target_alignment": "Target alignment", "reasoning_soundness": "Reasoning soundness",
                 "execution_accuracy": "Execution accuracy", "output_integrity": "Output integrity"}
    gd = [r"\begin{tabular}{lrrrr}", r"\toprule",
          r"GPA dimension & Baseline & TEI & $\Delta$ & $p$ \\", r"\midrule"]
    for d in DIMENSIONS:
        s = dim_stats[d]
        gd.append(f"{DIM_LABEL.get(d, d)} & {s['mean_before']:.3f} & {s['mean_after']:.3f} & "
                  f"{s['mean_delta']:+.3f} & {fmt_p(s['t_p_value'])} \\\\")
    gd += [r"\midrule",
           f"Aggregate & {gpa_stats['mean_before']:.3f} & {gpa_stats['mean_after']:.3f} & "
           f"{gpa_stats['mean_delta']:+.3f} & {fmt_p(gpa_stats['t_p_value'])} \\\\",
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "gpa_dims.tex").write_text("\n".join(gd), encoding="utf-8")

    # benchmark composition summary (by metric and by source)
    from collections import Counter
    met = Counter(r["metric"] for r in recs)
    n_pub = sum(1 for r in recs if r.get("is_public")); n_syn = n - n_pub
    tot_test = sum(r["n_test"] for r in recs)
    ts = [r"\begin{tabular}{lr}", r"\toprule", r"Benchmark composition & Count \\", r"\midrule",
          f"Tasks (total) & {n} \\\\",
          f"\\quad public (HuggingFace) & {n_pub} \\\\",
          f"\\quad authored / synthetic & {n_syn} \\\\", r"\midrule"]
    for mname, cnt in sorted(met.items(), key=lambda kv: -kv[1]):
        ts.append(f"metric: {mname.replace('_', ' ')} & {cnt} \\\\")
    ts += [r"\midrule",
           f"Held-out test examples (total) & {tot_test} \\\\",
           f"\\quad mean per task & {tot_test/n:.1f} \\\\",
           f"Tasks at ceiling ($\\geq$0.999 baseline) & {n_ceiling} \\\\",
           f"Headroom tasks (baseline $<$0.9) & {len(head_idx)} \\\\",
           r"\bottomrule", r"\end{tabular}"]
    (TAB / "task_summary.tex").write_text("\n".join(ts), encoding="utf-8")

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

    # ---------- MACROS ----------
    tb = cstats["tei_vs_baseline"]; tr = cstats["tei_vs_random"]; to = cstats["tei_vs_objref"]
    rb = cstats["random_vs_baseline"]
    m = {
        "nAgents": n, "nCeiling": n_ceiling,
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
        "signP": fmt_p(extra_full['sign']['p_value']), "permP": fmt_p(extra_full['perm']['p_value']),
        "tostBound": "0.05", "tostP": fmt_p(extra_full['tost05']['p_tost']),
        "tostEquiv": "yes" if extra_full['tost05']['equivalent'] else "no",
        "mdeFull": f"{extra_full['power']['mde_80pct']:.3f}",
        "powerFull": f"{extra_full['power']['post_hoc_power']:.2f}",
        "pubN": len(pub), "synN": len(syn),
    }
    if "public" in sub:
        m["pubDelta"] = f"{sub['public']['stats']['mean_delta']:+.3f}"; m["pubP"] = fmt_p(sub['public']['stats']['t_p_value'])
    if "synthetic" in sub:
        m["synDelta"] = f"{sub['synthetic']['stats']['mean_delta']:+.3f}"; m["synP"] = fmt_p(sub['synthetic']['stats']['t_p_value'])
    if headroom:
        hs = headroom["stats"]
        m["headN"] = headroom["n"]; m["headDelta"] = f"{hs['mean_delta']:+.3f}"
        m["headP"] = fmt_p(hs['t_p_value']); m["headDz"] = f"{hs['cohen_dz']:.2f}"
        m["headMde"] = f"{headroom['power']['mde_80pct']:.3f}"
        m["headPower"] = f"{headroom['power']['post_hoc_power']:.2f}"
    (ROOT / "paper" / "numbers_v2.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")

    print("== v2 summary ==")
    print("arm means:", "  ".join(f"{a}={np.mean(obj[a]):.3f}" for a in ARMS))
    print(f"ceiling tasks (>=0.999): {n_ceiling}")
    for k, c in cstats.items():
        print(f"  {k:20} d={c['mean_delta']:+.3f} p={fmt_p(c['t_p_value'])} dz={c['cohen_dz']:.2f} holm={fmt_p(holm[k])}")
    print(f"  TOST(±0.05) p={fmt_p(extra_full['tost05']['p_tost'])} equiv={extra_full['tost05']['equivalent']}")
    print(f"  MDE(80%)={extra_full['power']['mde_80pct']:.3f}  post-hoc power={extra_full['power']['post_hoc_power']:.2f}")
    if headroom:
        print(f"  headroom n={headroom['n']} d={headroom['stats']['mean_delta']:+.3f} "
              f"p={fmt_p(headroom['stats']['t_p_value'])} dz={headroom['stats']['cohen_dz']:.2f} "
              f"MDE={headroom['power']['mde_80pct']:.3f}")
    print("Wrote figures (truncated, arXiv-styled), tables, numbers_v2.tex, summary, csv")


if __name__ == "__main__":
    main()
