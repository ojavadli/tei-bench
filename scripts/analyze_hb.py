"""Analyze the high-budget experiment (results_v2_hb/): 20 iterations on the
headroom subset, arms baseline/random/objective_reflection/tei/opro.

Answers two reviewer critiques directly:
  (1) Budget: does the null survive a realistic budget? (20 iters vs the 6-iter
      main run on the SAME tasks).
  (2) External optimizer: TEI vs an OPRO-style instruction optimizer.

Emits paper/tables/highbudget.tex and paper/numbers_hb.tex (real numbers).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.stats import analyze_paired, tost_equivalence, power_mde

ROOT = Path(__file__).resolve().parent.parent
HB = ROOT / "results_v2_hb"
V2 = ROOT / "results_v2"
ARMS = ["baseline", "random", "objective_reflection", "tei", "opro"]


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    recs = {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())
            for p in sorted(HB.glob("*.json")) if not p.name.startswith("_")}
    if not recs:
        print("results_v2_hb/ empty — run the high-budget experiment first."); return
    tids = sorted(recs)
    n = len(tids)
    obj = {a: [recs[t]["arms"][a]["objective_mean"] for t in tids] for a in ARMS}
    print(f"High-budget: {n} tasks @ 20 iters. arm means:",
          {a: round(float(np.mean(obj[a])), 3) for a in ARMS})

    contrasts = {
        "tei_vs_baseline": ("tei", "baseline"), "tei_vs_random": ("tei", "random"),
        "tei_vs_opro": ("tei", "opro"), "opro_vs_baseline": ("opro", "baseline"),
    }
    cs = {k: analyze_paired(obj[b], obj[a]) for k, (a, b) in contrasts.items()}
    tost = tost_equivalence(obj["baseline"], obj["tei"], 0.05)
    pw = power_mde(obj["baseline"], obj["tei"])

    # budget effect: high-budget TEI vs 6-iter TEI on the SAME tasks
    budget = None
    v2 = {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())
          for p in V2.glob("*.json") if not p.name.startswith("_")}
    common = [t for t in tids if t in v2]
    if len(common) >= 2:
        hb_tei = [recs[t]["arms"]["tei"]["objective_mean"] for t in common]
        lo_tei = [v2[t]["arms"]["tei"]["objective_mean"] for t in common]
        budget = analyze_paired(lo_tei, hb_tei)
        print(f"budget effect (20-iter TEI - 6-iter TEI), n={len(common)}: "
              f"d={budget['mean_delta']:+.3f} p={fmt_p(budget['t_p_value'])}")

    # table
    def row(label, a):
        m = np.mean(obj[a]); 
        return f"{label} & {m:.3f} \\\\"
    rows = "\n".join(row(l, a) for l, a in
                     [("Baseline", "baseline"), ("Random search", "random"),
                      ("Obj-only reflection", "objective_reflection"),
                      ("OPRO-style", "opro"), ("TEI (full)", "tei")])
    def crow(label, s):
        return (f"{label} & {s['mean_delta']:+.3f} & [{s['ci95_low']:+.3f},{s['ci95_high']:+.3f}] "
                f"& {fmt_p(s['t_p_value'])} & {s['cohen_dz']:.2f} \\\\")
    tab = [r"\begin{tabular}{lr}", r"\toprule",
           f"Arm (20 iters, $n={n}$) & Held-out acc. \\\\", r"\midrule", rows,
           r"\bottomrule", r"\end{tabular}", r"", r"\vspace{4pt}", r"",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Contrast & $\Delta$ & 95\% CI & $p$ & $d_z$ \\", r"\midrule",
           crow("TEI $-$ baseline", cs["tei_vs_baseline"]),
           crow("TEI $-$ random", cs["tei_vs_random"]),
           crow("TEI $-$ OPRO", cs["tei_vs_opro"]),
           crow("OPRO $-$ baseline", cs["opro_vs_baseline"]),
           r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "highbudget.tex").write_text("\n".join(tab), encoding="utf-8")

    tb = cs["tei_vs_baseline"]; tr = cs["tei_vs_random"]; to = cs["tei_vs_opro"]; ob = cs["opro_vs_baseline"]
    m = {
        "hbN": n, "hbIters": 20,
        "hbBase": f"{np.mean(obj['baseline']):.3f}", "hbRand": f"{np.mean(obj['random']):.3f}",
        "hbObjref": f"{np.mean(obj['objective_reflection']):.3f}",
        "hbTei": f"{np.mean(obj['tei']):.3f}", "hbOpro": f"{np.mean(obj['opro']):.3f}",
        "hbTeiBaseDelta": f"{tb['mean_delta']:+.3f}", "hbTeiBaseP": fmt_p(tb['t_p_value']),
        "hbTeiBaseDz": f"{tb['cohen_dz']:.2f}",
        "hbTeiRandDelta": f"{tr['mean_delta']:+.3f}", "hbTeiRandP": fmt_p(tr['t_p_value']),
        "hbTeiRandDz": f"{tr['cohen_dz']:.2f}",
        "hbTeiOproDelta": f"{to['mean_delta']:+.3f}", "hbTeiOproP": fmt_p(to['t_p_value']),
        "hbTeiOproDz": f"{to['cohen_dz']:.2f}",
        "hbOproBaseDelta": f"{ob['mean_delta']:+.3f}", "hbOproBaseP": fmt_p(ob['t_p_value']),
    }
    if budget:
        m["hbBudgetDelta"] = f"{budget['mean_delta']:+.3f}"
        m["hbBudgetP"] = fmt_p(budget['t_p_value'])
        m["hbBudgetN"] = len(common)
    (ROOT / "paper" / "numbers_hb.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")

    hb_summary = {
        "n": n, "iters": 20,
        "arm_means": {a: float(np.mean(obj[a])) for a in ARMS},
        "contrasts": {k: cs[k] for k in cs},
        "budget_effect": budget, "tost05": tost, "power": pw,
    }
    (HB / "_summary.json").write_text(json.dumps(hb_summary, indent=2), encoding="utf-8")
    print("Wrote tables/highbudget.tex, numbers_hb.tex, results_v2_hb/_summary.json")
    for k, c in cs.items():
        print(f"  {k:18} d={c['mean_delta']:+.3f} p={fmt_p(c['t_p_value'])} dz={c['cohen_dz']:.2f}")
    print(f"  TOST(0.05) p={fmt_p(tost['p_tost'])} equiv={tost['equivalent']}  MDE80={pw['mde_80pct']:.3f}")


if __name__ == "__main__":
    main()
