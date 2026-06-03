---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- agents
- llm-evaluation
- prompt-optimization
- tei-loop
- gpa
- gepa
pretty_name: TEI-Bench
size_categories:
- n<1K
---

# TEI-Bench

A controlled, held-out benchmark for the **Target–Evaluate–Improve (TEI)** loop:
31 agentic tasks across 31 industries, each with a naive baseline prompt and a
train/test split, used to measure whether evaluation-guided reflective prompt
optimization produces real, generalizing improvement.

- **Paper / code:** https://github.com/ojavadli/tei-loop
- **Agent under test:** `claude-haiku-4-5` · **Judge + optimizer:** `claude-sonnet-4-5`

## Files

| File | Contents |
|---|---|
| `tasks.jsonl` | One row per example: `task_id, industry, metric, split, labels, query, gold, source`. |
| `results.jsonl` | One row per agent: baseline + optimized prompt, before/after objective + GPA, deltas. |

## Methodology (why the numbers are trustworthy)

- **Held-out:** optimization uses only `split=train`; all reported scores are on `split=test`.
- **Objective primary endpoint:** code-computed accuracy / exact match (no LLM), with the four GPA dimensions as secondary.
- **Judge ≠ agent:** a stronger, different model judges and optimizes a weaker agent → no self-preference bias.
- **Fair baselines:** baseline prompts include the label set; TEI earns gains via format/disambiguation/decision rules.
- **Powered statistics:** n = 31 paired (paired *t*-test + Wilcoxon + Cohen's d_z + bootstrap CI, Holm-corrected).

## Task composition

- ~14 tasks from public datasets (HuggingFace; original licenses apply, see each row's `source`).
- ~14 authored industry tasks (Sonnet-generated, Sonnet-validated; only label-agreed examples kept).
- 3 seed tasks (finance / healthcare / education).

## Citation

```bibtex
@misc{javadli2026teibench,
  title  = {TEI-Bench: A Controlled, Held-Out Evaluation of the Target-Evaluate-Improve Loop Across 31 Agentic Tasks},
  author = {Javadli, Orkhan and Zimina, Anni},
  year   = {2026}
}
```
