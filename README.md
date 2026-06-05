# TEI-Bench

A **controlled, held-out evaluation** of the Target–Evaluate–Improve (TEI) loop
across **31 agentic tasks spanning 31 industries**.

> 📄 **Read the full article:** [`ARTICLE.md`](ARTICLE.md) · 📝 [`paper/TEI-Bench.docx`](paper/TEI-Bench.docx) · 📑 [`paper/main.tex`](paper/main.tex)
>
> **Headline finding (v2, pre-registered `prereg-v2`, held-out, n=31, 4-arm ablation).** A *cautionary,
> nuanced* result. Once the output-format confound is removed (universal `FINAL:` answer contract) and a
> random-search baseline is added, the large gain seen under a naive scorer (a v1 pilot reported +0.175,
> p≈4e-6) **collapses on the full suite**: arm means are nearly tied — baseline **0.852**, random **0.867**,
> objective-only-reflection **0.865**, TEI **0.867**; TEI does **not** beat the baseline (Δ=+0.015, p=0.41,
> Holm=1.0), is **equivalent within ±0.05** (TOST p=0.03), and is **indistinguishable from random prompt
> search** (Δ=0.000). **But** the full-suite null is dominated by 12/31 ceiling tasks. On the **headroom
> subset (n=12) at a 3.3× budget (20 iters)**, reflective TEI **does** beat the baseline (**+0.078, p=0.034,
> d_z=0.70**) and an **OPRO-style external optimizer** by the same margin (OPRO gains **exactly 0.000** over
> baseline) — yet **still cannot be separated from budget-matched random paraphrase** (+0.067, p=0.102), and
> extra budget barely matters (+0.011, p=0.49). **Conclusion:** evaluation-guided optimization shows no
> aggregate benefit on a competent baseline; reflective optimization helps only on non-saturated tasks,
> where it beats a baseline and a non-reflective optimizer but not blind paraphrase search. Earlier v1 is in
> `results/` (provenance only); the corrected run is in `results_v2/`; the high-budget arm is in `results_v2_hb/`.

TEI-Bench measures whether evaluation-guided, GEPA-style reflective prompt
optimization (the "Improve" half of [tei-loop](https://github.com/ojavadli/tei-loop))
produces *real, generalizing* improvements — not in-sample, not self-judged,
not single-task.

## Why this is credible

| Safeguard | What we do |
|---|---|
| **Held-out test** | TEI optimizes only on the train split; every reported number is on unseen test inputs. |
| **Objective primary endpoint** | Accuracy / exact-match computed in *code* (no LLM). The four GPA dimensions are *secondary*. |
| **Judge ≠ agent** | Agent = `claude-haiku-4-5`; judge + optimizer = `claude-sonnet-4-5` (stronger, different model → no self-preference). |
| **Fair baselines** | Baseline prompts include the label set; TEI must earn gains via format/disambiguation/decision rules. |
| **Difficulty calibration** | Tasks span 2–20 classes; pre-registered headroom-subset analysis (baseline < 0.9). |
| **Powered statistics** | n = 31 paired: paired *t* + Wilcoxon + sign + permutation tests, Cohen's d_z, bootstrap 95% CI, Holm correction, **TOST equivalence**, and **power / minimum-detectable-effect**. |
| **Honest null tooling** | We report equivalence (TOST, ±0.05), MDE₈₀, and post-hoc power — not just `p > 0.05` — and disclose that 12/31 tasks are at ceiling. |

## Layout

```
teibench/         # the harness (LLM layer, GPA judge, scorers, optimizer w/ tei|objref|random|opro modes, stats)
tasks/            # 31 task datasets (committed JSON: train/test gold)
results_v2/       # PRIMARY corrected run: per-arm records (every prompt, output trace, score) + summary + CSV
results_v2_gpt/   # cross-provider spot-check (agent = gpt-4o-mini)
results_v2_hb/    # high-budget (20-iteration) + OPRO-style arm on the headroom subset
results/          # v1 pilot, retained ONLY for provenance (had the scoring artifact; do not cite)
paper/            # arXiv paper (LaTeX, 2-column) + auto-generated figures/tables/numbers + large appendix
scripts/          # build_*_tasks.py, run_experiment.py, analyze_v2.py, analyze_hb.py, build_appendix.py
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python scripts/build_public_tasks.py      # public-dataset agents (HuggingFace)
python scripts/build_authored_tasks.py    # authored industry agents (generate+validate)
python scripts/build_pilot_tasks.py       # 3 seed agents

# Main 4-arm ablation (baseline / random / objective_reflection / tei), held-out, ~$20:
python scripts/run_experiment.py --iterations 6 --concurrency 10

# High-budget + OPRO external optimizer on the headroom subset (baseline < 0.9):
python scripts/run_experiment.py --iterations 20 --modes random objective_reflection tei opro \
    --only <headroom task ids> --results-dir results_v2_hb

python scripts/analyze_v2.py              # figures (arXiv style), tables, paper macros (numbers_v2.tex)
python scripts/analyze_hb.py              # high-budget table + macros (numbers_hb.tex)
python scripts/build_appendix.py          # regenerate the large appendix from recorded data
cd paper && tectonic main.tex             # build the PDF (main 2-column + appendix)
```

The `.cache/` directory makes re-runs near-free (responses are content-addressed).
Every number in the paper is generated by macro from `results_v2/` — there are no hand-typed results.

## Models

- **Agent under test:** `claude-haiku-4-5`
- **Judge + optimizer:** `claude-sonnet-4-5`

## License

Code: MIT. Public-dataset tasks retain their original dataset licenses (see each
task's `source` field). Authored tasks: CC-BY-4.0.

## Citation

```bibtex
@misc{javadli2026teibench,
  title  = {TEI-Bench: A Controlled, Held-Out Evaluation of the
            Target-Evaluate-Improve Loop Across 31 Agentic Tasks},
  author = {Javadli, Orkhan and Zimina, Anni},
  year   = {2026},
  note   = {https://github.com/ojavadli/tei-loop}
}
```
