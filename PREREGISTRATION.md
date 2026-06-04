# Pre-registered analysis plan — TEI-Bench v2

This plan is committed **before** the v2 experiment is executed
(`scripts/run_experiment.py`). It supersedes the v1 plan.

## Honest disclosure of the v1 → v2 change
The initial (v1) run had two problems flagged in external review, which v2 fixes:
1. **Config mismatch.** The v1 `PREREGISTRATION.md` named `iterations=8, eval_repeats=2`,
   but the executed v1 run used `iterations=6, eval_repeats=1` (chosen for compute/runtime).
   That is a real inconsistency. We do not paper over it: v1 results are retained in
   `results/` for provenance, and v2 is a clean re-run whose config is fixed in *this*
   document and in code **before** execution.
2. **Scoring artifact.** The v1 label scorer extracted the predicted label by its *latest*
   mention, which mis-scored verbose baselines. v2 replaces this with a universal
   `FINAL: <answer>` output contract applied identically to every arm; the scorer reads
   only that line. This removes the format confound (and may shrink the headline effect —
   we report whatever happens).

## Hypotheses
- **H1.** TEI improves held-out objective accuracy over the competent baseline prompt. (paired, n=31)
- **H2.** TEI beats undirected **random** prompt search of the same budget. (the missing optimizer baseline)
- **H3.** GPA-guided selection adds value over **objective-only reflection** (TEI ≥ objective_reflection).

H0 for each: zero mean paired difference on held-out objective.

## Design
- Within-subjects (paired). Unit = one agent (task + baseline prompt). n = 31.
- Four arms, evaluated on the **same held-out test split** per task:
  `baseline`, `random`, `objective_reflection`, `tei`.
- Optimization (all optimized arms) uses the **train split only**; the test split is never seen.
- Universal `FINAL: <answer>` output contract on every arm; objective scorer reads only that line.

## Fixed configuration (a priori)
- Agent under test: `claude-haiku-4-5`. Judge + optimizer: `claude-sonnet-4-5`.
- Cross-provider confirmatory arm: agent `gpt-4o-mini`, judge `claude-sonnet-4-5`, on a fixed task subset.
- `iterations = 6`, `minibatch = 10`, `seed = 0`, `eval_repeats = 1` (per-task bootstrap CIs quantify uncertainty).
- Test sizes: public-dataset tasks = 30; authored tasks = 24; three seed tasks = 15.

## Endpoints
- **Primary:** objective, code-computed accuracy / exact match (no LLM), read from the FINAL line.
- **Secondary:** four GPA-inspired judge dimensions (target alignment, reasoning soundness,
  execution accuracy, output integrity) and their mean — scored for `baseline` and `tei`.

## Statistical analysis
- Primary: two-sided paired t-test on per-agent objective deltas, for each contrast
  (tei−baseline, tei−random, tei−objective_reflection, random−baseline).
- Robustness: Wilcoxon signed-rank. Effect size: Cohen's d_z. Per-contrast 10,000-sample bootstrap 95% CI.
- Per-task: bootstrap 95% CI on each arm's objective over its test examples.
- Multiplicity: Holm correction across the reported contrasts.
- Pre-specified subgroup analyses: **public-only** tasks and **synthetic(authored)-only** tasks, separately.
- Pre-specified headroom subset: baseline objective < 0.9.

## Stopping rule
All 31 tasks run once at `seed=0`. No task is added or removed based on its outcome.
The cross-provider subset is fixed in advance (listed in the run command), not chosen by result.

## Reporting commitment
We report all four arms, all contrasts, the public/synthetic split, ties and losses, and the
per-dimension breakdown — including any result unfavorable to TEI.
