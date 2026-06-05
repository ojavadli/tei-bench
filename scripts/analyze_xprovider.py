"""Cross-provider analysis: compare TEI's baseline->TEI objective gain when the
AGENT is an OpenAI model (results_v2_gpt/) vs the Claude agent (results_v2/),
on the SAME pre-specified task subset. Emits paper/tables/xprovider.tex.

Tests whether the effect is Claude-specific (the 'single model family' critique).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.stats import analyze_paired

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "paper" / "tables"

# Pre-specified cross-provider subset (fixed in advance, not chosen by outcome).
SUBSET = [
    "fin_banking_intent", "health_triage", "edu_gsm8k", "trust_safety_hate",
    "it_email_spam", "telecom_churn_reason", "media_news_desk",
    "insurance_claim_type", "social_emotion", "consumer_sentiment_5",
]


def arm_obj(results_dir, subset):
    base, tei, ids = [], [], []
    for tid in subset:
        p = Path(results_dir) / f"{tid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        base.append(d["arms"]["baseline"]["objective_mean"])
        tei.append(d["arms"]["tei"]["objective_mean"])
        ids.append(tid)
    return base, tei, ids


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def row(model, base, tei):
    if len(base) < 2:
        return f"{model} & -- & -- & -- \\\\"
    s = analyze_paired(base, tei)
    return (f"{model} (n={len(base)}) & {s['mean_before']:.3f} & {s['mean_after']:.3f} "
            f"& {s['mean_delta']:+.3f} ($p$={fmt_p(s['t_p_value'])}) \\\\")


def main():
    cb, ct, ids = arm_obj(ROOT / "results_v2", SUBSET)
    gb, gt, gids = arm_obj(ROOT / "results_v2_gpt", SUBSET)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Agent model (judge = Sonnet 4.5) & Baseline & TEI & $\Delta$ \\", r"\midrule",
             row("claude-haiku-4-5", cb, ct),
             row("gpt-4o-mini", gb, gt),
             r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "xprovider.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote tables/xprovider.tex")
    print("  claude subset:", row("claude-haiku-4-5", cb, ct))
    print("  gpt subset:   ", row("gpt-4o-mini", gb, gt))


if __name__ == "__main__":
    main()
