# Pre-registered analysis plan

This plan was fixed in code (`scripts/run_full.py`, `scripts/analyze.py`,
`teibench/stats.py`) **before** the full 31-agent run was executed. It is
recorded here for transparency.

## Hypothesis
**H1.** Applying the TEI loop (reflective Pareto prompt optimization guided by
GPA-style evaluation) to an agent's baseline prompt increases the agent's
objective score on a held-out test set, on average across diverse tasks.

**H0.** Mean per-agent change in held-out objective score is zero.

## Design
- Within-subjects (paired). Unit of analysis = one agent (task + baseline prompt). n = 31.
- Each agent evaluated on the **same held-out test split** under (A) baseline prompt and (B) TEI-optimized prompt.
- Optimization uses the **train split only**. The test split is never seen during optimization.

## Endpoints
- **Primary:** objective score (code-computed: label exact-match accuracy for classification; numeric exact match for arithmetic), averaged over `eval_repeats` runs.
- **Secondary:** the four GPA dimensions (target alignment, reasoning soundness, execution accuracy, output integrity) and their mean, scored by an LLM judge that is a different, stronger model than the agent.

## Statistical analysis
- **Primary test:** two-sided paired *t*-test on per-agent objective deltas (B − A).
- **Robustness:** Wilcoxon signed-rank test.
- **Effect size:** Cohen's d_z (paired); 95% CI on the mean delta via 10,000-sample bootstrap.
- **Multiplicity:** Holm–Bonferroni across {objective, GPA aggregate, 4 GPA dimensions}.
- **Significance threshold:** α = 0.05 on the Holm-corrected primary endpoint.

## Pre-registered secondary analysis
- **Headroom subset:** agents with baseline objective < 0.90 (a saturated metric cannot improve). Same paired tests restricted to this subset.

## Models (fixed a priori)
- Agent under test: `claude-haiku-4-5`.
- Judge + optimizer: `claude-sonnet-4-5`.

## Stopping rule
- All 31 agents are run once with `seed=0`, `iterations=8`, `eval_repeats=2`. No agent is added or removed based on its outcome.
