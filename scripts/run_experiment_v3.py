"""TEI v3 experiment: the redesigned, validation-gated, triaged loop.

For each task we record TWO arms on the SAME held-out test split:
  baseline   : the competent baseline prompt (judge on) -- cache-hit from v2
  tei_v3     : optimize_v3 (validation-gated selection, successive halving,
               structural-fix mutation, why-better/why-worse memory, do-no-harm,
               headroom triage), then evaluate the frozen prompt on test.

The random / objective_reflection / tei / opro arms for the SAME headroom tasks
already exist in results_v2_hb/ (20-iter run), so analyze_v3 joins on task_id to
contrast tei_v3 against them. We also run a few CEILING tasks to show triage skips
optimization and ships the baseline.

Usage:
  python scripts/run_experiment_v3.py --iterations 12 --concurrency 10
  python scripts/run_experiment_v3.py --only market_research_subjectivity ...
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
from teibench.optimizer import optimize_v3

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
V2 = ROOT / "results_v2"
JUDGE_MODEL = "claude-sonnet-4-5"
OPT_MODEL = "claude-sonnet-4-5"


def boot_ci(vals, iters=10000, seed=1):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return [float(a.mean()) if len(a) else 0.0] * 2
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def headroom_and_ceiling():
    head, ceil = [], []
    for p in sorted(V2.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        b = d["arms"]["baseline"]["objective_mean"]
        if b < 0.9:
            head.append(d["task_id"])
        elif b >= 0.999:
            ceil.append(d["task_id"])
    return head, ceil


async def eval_arm(llm, *, agent_model, system_prompt, task, run_judge):
    ev = await evaluate_split(
        llm, agent_model=agent_model, judge_model=JUDGE_MODEL,
        system_prompt=system_prompt, task=task, examples=task.test, run_judge=run_judge)
    objs = [e.objective for e in ev.examples]
    rec = {"objective_mean": ev.objective_mean, "objective_ci95": boot_ci(objs),
           "prompt": system_prompt,
           "examples": [{"query": e.query, "gold": e.gold, "output": e.output,
                         "objective": e.objective, "gpa": e.gpa} for e in ev.examples]}
    if run_judge:
        rec["gpa_mean"] = ev.gpa_mean; rec["gpa_dims"] = ev.gpa_dims
    return rec


async def run_task(llm, task, *, agent_model, num_iterations, seed, log):
    t0 = time.time()
    arms = {}
    arms["baseline"] = await eval_arm(llm, agent_model=agent_model,
                                      system_prompt=task.baseline_prompt, task=task, run_judge=True)
    opt = await optimize_v3(
        llm, agent_model=agent_model, judge_model=JUDGE_MODEL, optimizer_model=OPT_MODEL,
        task=task, baseline_prompt=task.baseline_prompt, train=task.train,
        num_iterations=num_iterations, seed=seed, log=log)
    arms["tei_v3"] = await eval_arm(llm, agent_model=agent_model,
                                    system_prompt=opt["best_prompt"], task=task, run_judge=True)
    arms["tei_v3"]["optimization"] = {
        "triaged": opt["triaged"], "judge_calls_saved": opt["judge_calls_saved"],
        "n_promoted": opt["n_promoted"], "best_val_obj": opt["best_val_obj"],
        "baseline_val_obj": opt["baseline_val_obj"],
        "ship_prob_better": opt.get("ship_prob_better"),
        "shipped_baseline": opt.get("shipped_baseline"),
        "front": opt["front"], "why_log": opt["why_log"], "history": opt["history"]}
    return {"task_id": task.task_id, "industry": task.industry, "metric": task.metric,
            "source": task.source, "is_public": "HuggingFace" in task.source,
            "n_train": task.n_train, "n_test": task.n_test, "agent_model": agent_model,
            "judge_model": JUDGE_MODEL, "optimizer_model": OPT_MODEL, "seed": seed,
            "arms": arms, "duration_s": round(time.time() - t0, 1)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-model", default="claude-haiku-4-5")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="run all 31 tasks")
    ap.add_argument("--n-ceiling", type=int, default=4)
    ap.add_argument("--results-dir", default="results_v3")
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir
    results_dir.mkdir(exist_ok=True)
    usage = Usage()
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=args.concurrency, usage=usage)
    tasks = load_all_tasks(TASKS_DIR)

    if args.only:
        want = set(args.only)
    elif args.all:
        want = {t.task_id for t in tasks}
        print(f"Running ALL {len(want)} tasks (full suite).")
    else:
        head, ceil = headroom_and_ceiling()
        want = set(head) | set(ceil[: args.n_ceiling])
        print(f"Headroom tasks: {len(head)} | ceiling (triage demo): {min(args.n_ceiling, len(ceil))}")
    tasks = [t for t in tasks if t.task_id in want]

    print(f"EXPERIMENT v3 | {len(tasks)} tasks | agent={args.agent_model} "
          f"judge={JUDGE_MODEL} iters={args.iterations}", flush=True)
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        out_path = results_dir / f"{task.task_id}.json"
        if out_path.exists():
            print(f"[{i}/{len(tasks)}] {task.task_id} — exists, skipping", flush=True)
            continue
        log = []
        rec = await run_task(llm, task, agent_model=args.agent_model,
                             num_iterations=args.iterations, seed=args.seed, log=log)
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        a = rec["arms"]; o = a["tei_v3"]["optimization"]
        tag = "TRIAGED" if o["triaged"] else f"saved={o['judge_calls_saved']}"
        print(f"[{i}/{len(tasks)}] {task.task_id:<28} base={a['baseline']['objective_mean']:.3f} "
              f"tei_v3={a['tei_v3']['objective_mean']:.3f} [{tag}]  [{usage.report()}]", flush=True)

    print(f"\nDONE {len(tasks)} tasks in {(time.time()-t0)/60:.1f} min | {usage.report()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
