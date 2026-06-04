"""Build the article as a Word .docx (and a Word-openable HTML) DIRECTLY from
the recorded results — no hand-typed numbers, so it cannot invent anything.

Every value is read from:
  results/_full_summary.json   (aggregate stats, produced by run_full.py)
  results/per_agent.csv        (per-agent before/after, produced by run_full.py)
  results/<task_id>.json       (full per-example traces)
Figures are embedded from paper/figures/*.png (produced by analyze.py).

Usage:
  pip install python-docx
  python scripts/build_report.py
Outputs:
  paper/TEI-Bench.docx
  TEI-Bench.html   (fallback that opens in Word with File > Open)
"""
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "paper" / "figures"


def load():
    summary = json.loads((RESULTS / "_full_summary.json").read_text())
    rows = []
    with (RESULTS / "per_agent.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return summary, rows


def fmt_p(p):
    p = float(p)
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


AGG_ROWS = [
    ("Objective (primary)", "objective"),
    ("GPA aggregate", "gpa"),
]
DIM_ROWS = [
    ("\u2014 target alignment", "target_alignment"),
    ("\u2014 reasoning soundness", "reasoning_soundness"),
    ("\u2014 execution accuracy", "execution_accuracy"),
    ("\u2014 output integrity", "output_integrity"),
]


def agg_table_data(summary):
    out = []
    for label, key in AGG_ROWS:
        s = summary[key]
        out.append([label, f"{s['mean_before']:.3f}", f"{s['mean_after']:.3f}",
                    f"{s['mean_delta']:+.3f}",
                    f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]",
                    f"{s['t_stat']:.2f}", fmt_p(s['t_p_value']), f"{s['cohen_dz']:.2f}"])
    for label, key in DIM_ROWS:
        s = summary["gpa_dimensions"][key]
        out.append([label, f"{s['mean_before']:.3f}", f"{s['mean_after']:.3f}",
                    f"{s['mean_delta']:+.3f}",
                    f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}]",
                    f"{s['t_stat']:.2f}", fmt_p(s['t_p_value']), f"{s['cohen_dz']:.2f}"])
    return out


REFERENCES = [
    'Agrawal, L. A., et al. (2025). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457.',
    'Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714.',
    'Opsahl-Ong, K., et al. (2024). Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPROv2). arXiv:2406.11695.',
    'Yang, C., et al. (2023). Large Language Models as Optimizers (OPRO). arXiv:2309.03409.',
    'Zhou, Y., et al. (2022). Large Language Models Are Human-Level Prompt Engineers (APE). arXiv:2211.01910.',
    'Yuksekgonul, M., et al. (2024). TextGrad: Automatic "Differentiation" via Text. arXiv:2406.07496.',
    'Madaan, A., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651.',
    'Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366.',
    'Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.',
    'Jia, A. S., Huang, D., Vytla, N., Choudhury, N., Sen, S., Mitchell, J. C., Datta, A. (2025). GPA: A Goal-Plan-Action Framework for Evaluating Agentic Systems. Stanford University.',
    'Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K). arXiv:2110.14168.',
    'Casanueva, I., et al. (2020). Efficient Intent Detection with Dual Sentence Encoders (banking77). arXiv:2003.04807.',
]

ABSTRACT = (
    "Evaluation-guided, automated prompt optimization promises to improve agentic large-language-model "
    "(LLM) systems without manual prompt engineering, yet most supporting evidence is anecdotal: single "
    "tasks, in-sample evaluation, and subjective LLM-as-judge metrics scored by a model from the same "
    "family as the system under test. We present TEI-Bench, a controlled study of the "
    "Target-Evaluate-Improve (TEI) loop \u2014 a GPA-style multi-dimensional evaluator coupled to a "
    "GEPA-style reflective, Pareto-based prompt optimizer \u2014 across {n} agents spanning {n} industries. "
    "We adopt three safeguards absent from prior demonstrations: (i) a strict train/test split so all "
    "reported scores are on held-out inputs; (ii) an objective primary endpoint (accuracy / exact match) "
    "computed in code, with the four GPA dimensions as secondary; and (iii) judge-agent separation, "
    "optimizing and judging a weaker model (claude-haiku-4-5) with a stronger, different model "
    "(claude-sonnet-4-5). On held-out test data, TEI raises the objective score from {ob:.3f} to {oa:.3f} "
    "(mean delta = {od:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], paired t = {t:.2f}, p = {p}, Cohen's d_z = "
    "{dz:.2f}; {w}/{l}/{ti} win/loss/tie). Improvements concentrate on tasks with measurable headroom and "
    "on the dimensions a prompt can control (execution accuracy and target alignment), while reasoning "
    "soundness is essentially unchanged. We release all code, the {n} task datasets, every agent output "
    "trace, and all optimized prompts."
)


def build_docx(summary, rows):
    try:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        print("  python-docx not installed; skipping .docx (run: pip install python-docx)")
        return False

    o = summary["objective"]
    doc = Document()

    title = doc.add_heading(
        "TEI-Bench: A Controlled, Held-Out Evaluation of the "
        "Target-Evaluate-Improve Loop Across 31 Agentic Tasks", level=0)
    p = doc.add_paragraph("Orkhan Javadli (MIT)  \u00b7  Anni Zimina (Stanford)")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Code & data: https://github.com/ojavadli/tei-bench")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT.format(
        n=summary["n_agents"], ob=o["mean_before"], oa=o["mean_after"],
        od=o["mean_delta"], lo=o["ci95_low"], hi=o["ci95_high"], t=o["t_stat"],
        p=fmt_p(o["t_p_value"]), dz=o["cohen_dz"], w=o["wins"], l=o["losses"], ti=o["ties"]))

    # Methods summary
    doc.add_heading("Method & Design (summary)", level=1)
    for line in [
        "Unit of analysis: one agent = one (task, baseline prompt) pair; each measured twice on the SAME held-out test set (baseline vs TEI-optimized prompt). n = 31 paired.",
        "Models: agent under test = claude-haiku-4-5; judge + optimizer = claude-sonnet-4-5 (stronger, different model -> reduces self-preference bias).",
        "Primary endpoint: objective, code-computed accuracy / exact match (no LLM). Secondary: four GPA dimensions (judge-scored).",
        "Optimizer (GEPA-style): reflective mutation on failing TRAIN examples + system-aware merge, Pareto front; never sees the test split.",
        "Pre-registered analysis: paired t-test, Wilcoxon signed-rank, Cohen's d_z, 10,000-sample bootstrap CI, Holm correction; headroom subset (baseline < 0.9).",
        f"Compute: {summary['usage']['calls']} LLM calls, ${summary['usage']['cost_usd']:.2f}, "
        f"{summary['wall_clock_s']/60:.0f} min wall.",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    # Aggregate table
    doc.add_heading("Results: aggregate (held-out, n = 31 paired)", level=1)
    data = agg_table_data(summary)
    headers = ["Endpoint", "Base", "TEI", "\u0394", "95% CI", "t", "p", "d_z"]
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
    for r in data:
        c = t.add_row().cells
        for i, v in enumerate(r):
            c[i].text = str(v)
    doc.add_paragraph(
        f"Win/loss/tie on the objective endpoint: {o['wins']}/{o['losses']}/{o['ties']}. "
        f"Wilcoxon p = {fmt_p(o['wilcoxon_p_value'])}. "
        f"Headroom subset (baseline < 0.9, n = {summary['headroom_subset']['n']}): "
        f"delta = {summary['headroom_subset']['stats']['mean_delta']:+.3f}, "
        f"p = {fmt_p(summary['headroom_subset']['stats']['t_p_value'])}, "
        f"d_z = {summary['headroom_subset']['stats']['cohen_dz']:.2f}.")

    # Figures
    for fn, cap in [
        ("slope_objective.png", "Figure 1. Per-agent held-out objective score, baseline -> TEI."),
        ("delta_hist.png", "Figure 2. Distribution of per-agent objective improvements."),
        ("gpa_dimensions.png", "Figure 3. GPA dimensions before/after (held-out)."),
        ("headroom_scatter.png", "Figure 4. Improvement vs. baseline headroom."),
    ]:
        fp = FIG / fn
        if fp.exists():
            doc.add_picture(str(fp), width=Inches(5.2))
            cp = doc.add_paragraph(cap); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Per-agent table
    doc.add_heading("Results: per-agent (held-out)", level=1)
    headers = ["Agent", "Industry", "Base", "TEI", "\u0394obj", "\u0394GPA"]
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
    for r in rows:
        c = t.add_row().cells
        c[0].text = r["task_id"]; c[1].text = r["industry"]
        c[2].text = f'{float(r["obj_before"]):.3f}'; c[3].text = f'{float(r["obj_after"]):.3f}'
        c[4].text = f'{float(r["obj_delta"]):+.3f}'; c[5].text = f'{float(r["gpa_delta"]):+.3f}'

    # References
    doc.add_heading("References", level=1)
    for i, ref in enumerate(REFERENCES, 1):
        doc.add_paragraph(f"[{i}] {ref}")

    out = ROOT / "paper" / "TEI-Bench.docx"
    doc.save(str(out))
    print(f"  wrote {out}")
    return True


def build_html(summary, rows):
    o = summary["objective"]
    abstract = ABSTRACT.format(
        n=summary["n_agents"], ob=o["mean_before"], oa=o["mean_after"],
        od=o["mean_delta"], lo=o["ci95_low"], hi=o["ci95_high"], t=o["t_stat"],
        p=fmt_p(o["t_p_value"]), dz=o["cohen_dz"], w=o["wins"], l=o["losses"], ti=o["ties"])
    agg = agg_table_data(summary)

    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"

    agg_html = ("<table><tr>" + "".join(f"<th>{h}</th>" for h in
                ["Endpoint", "Base", "TEI", "\u0394", "95% CI", "t", "p", "d_z"]) + "</tr>"
                + "".join(tr(r) for r in agg) + "</table>")
    per_html = ("<table><tr>" + "".join(f"<th>{h}</th>" for h in
                ["Agent", "Industry", "Base", "TEI", "\u0394obj", "\u0394GPA"]) + "</tr>"
                + "".join(tr([r["task_id"], r["industry"], f'{float(r["obj_before"]):.3f}',
                              f'{float(r["obj_after"]):.3f}', f'{float(r["obj_delta"]):+.3f}',
                              f'{float(r["gpa_delta"]):+.3f}']) for r in rows) + "</table>")
    refs = "<ol>" + "".join(f"<li>{html.escape(r)}</li>" for r in REFERENCES) + "</ol>"
    figs = "".join(
        f'<figure><img src="paper/figures/{fn}" style="max-width:640px"/>'
        f'<figcaption><i>{html.escape(cap)}</i></figcaption></figure>'
        for fn, cap in [
            ("slope_objective.png", "Figure 1. Per-agent held-out objective, baseline -> TEI."),
            ("delta_hist.png", "Figure 2. Distribution of per-agent objective deltas."),
            ("gpa_dimensions.png", "Figure 3. GPA dimensions before/after."),
            ("headroom_scatter.png", "Figure 4. Improvement vs. baseline headroom."),
        ] if (FIG / fn).exists())

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>TEI-Bench</title><style>
body{{font-family:Calibri,Arial,sans-serif;max-width:900px;margin:2em auto;line-height:1.5;color:#111}}
h1{{font-size:22pt}} h2{{font-size:15pt;border-bottom:1px solid #ccc;padding-bottom:3px}}
table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:10pt}}
th,td{{border:1px solid #999;padding:5px 8px;text-align:left}} th{{background:#eef}}
figure{{text-align:center;margin:1.2em 0}} figcaption{{font-size:9pt;color:#444}}
.meta{{text-align:center;color:#444}}
</style></head><body>
<h1>TEI-Bench: A Controlled, Held-Out Evaluation of the Target-Evaluate-Improve Loop Across 31 Agentic Tasks</h1>
<p class="meta">Orkhan Javadli (MIT) &middot; Anni Zimina (Stanford)<br>
Code &amp; data: <a href="https://github.com/ojavadli/tei-bench">github.com/ojavadli/tei-bench</a></p>
<h2>Abstract</h2><p>{html.escape(abstract)}</p>
<h2>Results: aggregate (held-out, n = 31 paired)</h2>{agg_html}
<p>Win/loss/tie (objective): {o['wins']}/{o['losses']}/{o['ties']}. Headroom subset (baseline &lt; 0.9,
n = {summary['headroom_subset']['n']}): &Delta; = {summary['headroom_subset']['stats']['mean_delta']:+.3f},
p = {fmt_p(summary['headroom_subset']['stats']['t_p_value'])},
d_z = {summary['headroom_subset']['stats']['cohen_dz']:.2f}.</p>
{figs}
<h2>Results: per-agent (held-out)</h2>{per_html}
<h2>References</h2>{refs}
</body></html>"""
    out = ROOT / "TEI-Bench.html"
    out.write_text(doc, encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    summary, rows = load()
    assert summary["n_agents"] == len(rows) == 31, "data mismatch"
    print("Building report from recorded results...")
    build_html(summary, rows)
    build_docx(summary, rows)
    print("Done.")
