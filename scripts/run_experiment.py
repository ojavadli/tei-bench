"""Full ablation experiment (v2) — addresses the peer-review critiques.

For every task we evaluate FOUR arms on the SAME held-out test split:
  baseline               : the competent baseline prompt (no optimization)
  random                 : undirected paraphrase search (same budget)
  objective_reflection   : reflective optimization, objective-only selection
  tei                    : reflective optimization, GPA-guided Pareto selection

All arms use the universal FINAL: output contract (scorer reads only that line),
so the format confound is removed. The judge (GPA) runs only for baseline + tei.
Per-arm objective on test is reported with a bootstrap 95% CI over test examples.
Every per-example test output is recorded to results_v2/<task_id>.json.

Usage:
  python scripts/run_experiment.py --iterations 6 --concurrency 10
  python scripts/run_experiment.py --only fin_banking_intent edu_gsm8k   # subset
  python scripts/run_experiment.py --agent-model gpt-4o-mini --suffix _gpt --only ...  # cross-provider
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench import LLM, Usage, load_all_tasks
from teibench.agent import evaluate_split
from teibench.optimizer import optimize
from teibench.gpa_judge import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
JUDGE_MODEL = "claude-sonnet-4-5"
OPT_MODEL = "claude-sonnet-4-5"
ARMS = ["baseline", "random", "objective_reflection", "tei"]


def boot_ci(vals, iters=10000, seed=1):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return [float(a.mean()) if len(a) else 0.0] * 2
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


async def eval_arm(llm, *, agent_model, system_prompt, task, run_judge):
    ev = await evaluate_split(
        llm, agent_model=agent_model, judge_model=JUDGE_MODEL,
        system_prompt=system_prompt, task=task, examples=task.test, run_judge=run_judge)
    objs = [e.objective for e in ev.examples]
    rec = {
        "objective_mean": ev.objective_mean,
        "objective_ci95": boot_ci(objs),
        "prompt": system_prompt,
        "examples": [{"query": e.query, "gold": e.gold, "output": e.output,
                      "objective": e.objective, "gpa": e.gpa} for e in ev.examples],
    }
    if run_judge:
        rec["gpa_mean"] = ev.gpa_mean
        rec["gpa_dims"] = ev.gpa_dims
    return rec


async def run_task(llm, task, *, agent_model, num_iterations, minibatch, seed, log):
    t0 = time.time()
    arms = {}
    # baseline (judge on)
    arms["baseline"] = await eval_arm(
        llm, agent_model=agent_model, system_prompt=task.baseline_prompt,
        task=task, run_judge=True)
    log.append(f"  [{task.task_id}] baseline obj={arms['baseline']['objective_mean']:.3f}")
    # optimization arms
    for mode in ("random", "objective_reflection", "tei"):
        opt = await optimize(
            llm, agent_model=agent_model, judge_model=JUDGE_MODEL, optimizer_model=OPT_MODEL,
            task=task, baseline_prompt=task.baseline_prompt, train=task.train,
            mode=mode, num_iterations=num_iterations, minibatch=minibatch, seed=seed, log=log)
        arms[mode] = await eval_arm(
            llm, agent_model=agent_model, system_prompt=opt["best_prompt"],
            task=task, run_judge=(mode == "tei"))
        arms[mode]["optimization"] = {"history": opt["history"], "front": opt["front"]}
        log.append(f"  [{task.task_id}] {mode} obj={arms[mode]['objective_mean']:.3f}")
    return {
        "task_id": task.task_id, "industry": task.industry, "metric": task.metric,
        "source": task.source,
        "is_public": "HuggingFace" in task.source,
        "n_train": task.n_train, "n_test": task.n_test,
        "agent_model": agent_model, "judge_model": JUDGE_MODEL, "optimizer_model": OPT_MODEL,
        "seed": seed, "arms": arms, "duration_s": round(time.time() - t0, 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-model", default="claude-haiku-4-5")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--results-dir", default="results_v2")
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir
    results_dir.mkdir(exist_ok=True)
    usage = Usage()
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=args.concurrency, usage=usage)
    tasks = load_all_tasks(TASKS_DIR)
    if args.only:
        tasks = [t for t in tasks if t.task_id in args.only]

    print(f"EXPERIMENT v2 | {len(tasks)} tasks | agent={args.agent_model} "
          f"judge={JUDGE_MODEL} arms={ARMS} iters={args.iterations}", flush=True)
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        out_path = results_dir / f"{task.task_id}.json"
        if out_path.exists():
            print(f"[{i}/{len(tasks)}] {task.task_id} — exists, skipping", flush=True)
            continue
        log = []
        rec = await run_task(llm, task, agent_model=args.agent_model,
                             num_iterations=args.iterations, minibatch=args.minibatch,
                             seed=args.seed, log=log)
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        a = rec["arms"]
        print(f"[{i}/{len(tasks)}] {task.task_id:<28} "
              f"base={a['baseline']['objective_mean']:.3f} "
              f"rand={a['random']['objective_mean']:.3f} "
              f"objref={a['objective_reflection']['objective_mean']:.3f} "
              f"tei={a['tei']['objective_mean']:.3f}  "
              f"[{usage.report()}]", flush=True)

    print(f"\nDONE {len(tasks)} tasks in {(time.time()-t0)/60:.1f} min | {usage.report()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
