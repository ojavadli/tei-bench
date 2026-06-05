# TEI-Bench

A **controlled, held-out evaluation** of the Target–Evaluate–Improve (TEI) loop
across **31 agentic tasks spanning 31 industries**.

> 📄 **Read the full article:** [`ARTICLE.md`](ARTICLE.md) · 📝 [`paper/TEI-Bench.docx`](paper/TEI-Bench.docx) · 📑 [`paper/main.tex`](paper/main.tex)
>
> **Headline finding (v2, pre-registered `prereg-v2`, held-out, n=31, 4-arm ablation).** This is a
> *cautionary / negative* result. Once the output-format confound is removed (universal `FINAL:`
> answer contract) and a random-search baseline is added, the large gain seen under a naive scorer
> (a v1 pilot reported +0.175, p≈4e-6) **collapses**: arm means are nearly tied — baseline **0.852**,
> random **0.867**, objective-only-reflection **0.865**, TEI **0.867**. TEI does **not** significantly
> beat the baseline (Δ=+0.015, p=0.41, Holm=1.0) and is **statistically indistinguishable from random
> prompt search** (Δ=0.000, p=1.0). Conclusion: on single-turn classification with a competent baseline,
> the marginal value of evaluation-guided prompt optimization is unproven and easily overstated by naive
> scoring. The earlier v1 results are retained in `results/` for provenance; the corrected run is in `results_v2/`.

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
| **Powered statistics** | n = 31 paired: paired *t*-test + Wilcoxon + Cohen's d_z + bootstrap 95% CI, Holm-corrected. |

## Layout

```
teibench/        # the harness (LLM layer, GPA judge, scorers, GEPA optimizer, TEI procedure, stats)
tasks/           # 31 task datasets (committed JSON: train/test gold)
results/         # per-agent records (every prompt, output trace, score) + summary + CSV
paper/           # arXiv paper (LaTeX) + auto-generated figures/tables/numbers
scripts/         # build_*_tasks.py, run_full.py, analyze.py
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python scripts/build_public_tasks.py     # public-dataset agents (HuggingFace)
python scripts/build_authored_tasks.py   # authored industry agents (generate+validate)
python scripts/build_pilot_tasks.py      # 3 seed agents
python scripts/run_full.py               # full 31-agent run (~1.5–2h, ~$15)
python scripts/analyze.py                # figures, tables, paper macros
```

The `.cache/` directory makes re-runs near-free (responses are content-addressed).

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
