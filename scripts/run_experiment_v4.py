"""TEI v4 experiment: the explicit TWO-PHASE loop requested.

For each of the 31 agents:
  Phase A: <struct_iters> iterations of STRUCTURAL fixes -> deploy the highest-scoring
           structure (scored on a held-out validation split V).
  Phase B: starting from that structure, <prompt_iters> iterations of PROMPT
           optimizations -> deploy the highest-scoring prompt.
A final do-no-harm gate confirms the winner on an INDEPENDENT split (ship baseline
unless confirmed better), so the loop never deploys a score-decreasing change.

Records, per agent on the held-out TEST split: baseline arm and two_phase arm
(objective + 4 evaluation dimensions), plus the per-phase validation bests and the
deployed prompts. Models: agent (haiku by default) under test; optimizer/judge a
Sonnet (<= 4.6). API key read from ANTHROPIC_API_KEY (never written to disk).

Usage:
  ANTHROPIC_API_KEY=... python scripts/run_experiment_v4.py \
      --struct-iters 30 --prompt-iters 30 --optimizer-model claude-sonnet-4-6 \
      --judge-model claude-sonnet-4-6 --agent-model claude-haiku-4-5 --concurrency 12
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
from teibench.optimizer import optimize_two_phase

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"


def boot_ci(vals, iters=10000, seed=1):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return [float(a.mean()) if len(a) else 0.0] * 2
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


async def eval_arm(llm, *, agent_model, judge_model, system_prompt, task):
    ev = await evaluate_split(llm, agent_model=agent_model, judge_model=judge_model,
                              system_prompt=system_prompt, task=task, examples=task.test, run_judge=True)
    objs = [e.objective for e in ev.examples]
    return {"objective_mean": ev.objective_mean, "objective_ci95": boot_ci(objs),
            "gpa_mean": ev.gpa_mean, "gpa_dims": ev.gpa_dims, "prompt": system_prompt,
            "examples": [{"query": e.query, "gold": e.gold, "output": e.output,
                          "objective": e.objective, "gpa": e.gpa} for e in ev.examples]}


async def run_task(llm, task, *, agent_model, judge_model, optimizer_model,
                   struct_iters, prompt_iters, seed, log):
    t0 = time.time()
    arms = {"baseline": await eval_arm(llm, agent_model=agent_model, judge_model=judge_model,
                                       system_prompt=task.baseline_prompt, task=task)}
    opt = await optimize_two_phase(
        llm, agent_model=agent_model, judge_model=judge_model, optimizer_model=optimizer_model,
        task=task, baseline_prompt=task.baseline_prompt, train=task.train,
        struct_iters=struct_iters, prompt_iters=prompt_iters, seed=seed, log=log)
    arms["two_phase"] = await eval_arm(llm, agent_model=agent_model, judge_model=judge_model,
                                       system_prompt=opt["best_prompt"], task=task)
    arms["two_phase"]["optimization"] = {k: opt[k] for k in
        ("baseline_val", "struct_best_val", "prompt_best_val", "final_val",
         "shipped_baseline", "ship_prob_better", "deployed_phase", "struct_iters",
         "prompt_iters", "struct_prompt", "trajA", "trajB")}
    return {"task_id": task.task_id, "industry": task.industry, "metric": task.metric,
            "source": task.source, "is_public": "HuggingFace" in task.source,
            "n_train": task.n_train, "n_test": task.n_test, "agent_model": agent_model,
            "judge_model": judge_model, "optimizer_model": optimizer_model, "seed": seed,
            "arms": arms, "duration_s": round(time.time() - t0, 1)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-iters", type=int, default=30)
    ap.add_argument("--prompt-iters", type=int, default=30)
    ap.add_argument("--agent-model", default="claude-haiku-4-5")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--optimizer-model", default="claude-sonnet-4-6")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--results-dir", default="results_v4")
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir; results_dir.mkdir(exist_ok=True)
    usage = Usage()
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=args.concurrency, usage=usage)
    tasks = load_all_tasks(TASKS_DIR)
    if args.only:
        tasks = [t for t in tasks if t.task_id in args.only]

    print(f"EXPERIMENT v4 (two-phase) | {len(tasks)} tasks | agent={args.agent_model} "
          f"opt/judge={args.optimizer_model} | {args.struct_iters} struct + "
          f"{args.prompt_iters} prompt iters/agent", flush=True)
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        out_path = results_dir / f"{task.task_id}.json"
        if out_path.exists():
            print(f"[{i}/{len(tasks)}] {task.task_id} — exists, skipping", flush=True); continue
        log = []
        rec = await run_task(llm, task, agent_model=args.agent_model, judge_model=args.judge_model,
                             optimizer_model=args.optimizer_model, struct_iters=args.struct_iters,
                             prompt_iters=args.prompt_iters, seed=args.seed, log=log)
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        a = rec["arms"]; o = a["two_phase"]["optimization"]
        print(f"[{i}/{len(tasks)}] {task.task_id:<28} base={a['baseline']['objective_mean']:.3f} "
              f"two_phase={a['two_phase']['objective_mean']:.3f} "
              f"[struct_val={o['struct_best_val']:.2f} prompt_val={o['prompt_best_val']:.2f} "
              f"deploy={o['deployed_phase']}]  [{usage.report()}]", flush=True)

    print(f"\nDONE {len(tasks)} tasks in {(time.time()-t0)/60:.1f} min | {usage.report()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
