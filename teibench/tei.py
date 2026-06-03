"""The TEI procedure for a single agent (one experimental unit).

Pipeline (controlled, held-out):
  1. BASELINE  — evaluate the naive baseline prompt on the TEST split
                 (repeated `eval_repeats` times to quantify variance).
  2. OPTIMIZE  — run GEPA-style reflective Pareto optimization on the
                 TRAIN split only, yielding an optimized prompt.
  3. FINAL     — evaluate the optimized prompt on the same held-out TEST
                 split (same repeats).

The paired contrast (baseline_test vs final_test) on held-out data is the
unit that feeds the across-agent statistics.
"""
from __future__ import annotations

import time
from typing import Optional

from .agent import evaluate_split
from .llm import LLM
from .optimizer import optimize


async def _repeated_test_eval(
    llm, *, agent_model, judge_model, system_prompt, task, repeats, agent_temperature,
):
    objs, gpas, dim_runs, example_sets = [], [], [], []
    for r in range(repeats):
        ev = await evaluate_split(
            llm, agent_model=agent_model, judge_model=judge_model,
            system_prompt=system_prompt, task=task, examples=task.test,
            agent_temperature=agent_temperature,
        )
        objs.append(ev.objective_mean)
        gpas.append(ev.gpa_mean)
        dim_runs.append(ev.gpa_dims)
        example_sets.append([
            {"query": e.query, "gold": e.gold, "output": e.output,
             "objective": e.objective, "gpa": e.gpa}
            for e in ev.examples
        ])
    dims = {}
    if dim_runs and dim_runs[0]:
        for d in dim_runs[0]:
            dims[d] = sum(run[d] for run in dim_runs) / len(dim_runs)
    return {
        "objective_runs": objs,
        "gpa_runs": gpas,
        "objective_mean": sum(objs) / len(objs),
        "gpa_mean": sum(gpas) / len(gpas),
        "gpa_dims": dims,
        # keep the first repeat's full per-example trace for the record
        "examples": example_sets[0],
    }


async def run_tei_on_agent(
    llm: LLM,
    task,
    *,
    agent_model: str,
    judge_model: str,
    optimizer_model: str,
    num_iterations: int = 8,
    minibatch: int = 8,
    eval_repeats: int = 2,
    seed: int = 0,
    agent_temperature: float = 0.0,
    log: Optional[list] = None,
) -> dict:
    log = log if log is not None else []
    t0 = time.time()
    log.append(f"  ── TEI on [{task.task_id}] ({task.industry}) "
               f"metric={task.metric} n_train={task.n_train} n_test={task.n_test}")

    # 1) BASELINE on held-out test
    baseline = await _repeated_test_eval(
        llm, agent_model=agent_model, judge_model=judge_model,
        system_prompt=task.baseline_prompt, task=task,
        repeats=eval_repeats, agent_temperature=agent_temperature,
    )
    log.append(f"    baseline TEST: obj={baseline['objective_mean']:.3f} "
               f"gpa={baseline['gpa_mean']:.3f}")

    # 2) OPTIMIZE on train only
    opt = await optimize(
        llm, agent_model=agent_model, judge_model=judge_model,
        optimizer_model=optimizer_model, task=task,
        baseline_prompt=task.baseline_prompt, train=task.train,
        num_iterations=num_iterations, minibatch=minibatch, seed=seed, log=log,
    )

    # 3) FINAL (optimized prompt) on the SAME held-out test
    final = await _repeated_test_eval(
        llm, agent_model=agent_model, judge_model=judge_model,
        system_prompt=opt["best_prompt"], task=task,
        repeats=eval_repeats, agent_temperature=agent_temperature,
    )
    log.append(f"    final    TEST: obj={final['objective_mean']:.3f} "
               f"gpa={final['gpa_mean']:.3f}  "
               f"Δobj={final['objective_mean']-baseline['objective_mean']:+.3f} "
               f"Δgpa={final['gpa_mean']-baseline['gpa_mean']:+.3f}")

    return {
        "task_id": task.task_id,
        "industry": task.industry,
        "metric": task.metric,
        "n_train": task.n_train,
        "n_test": task.n_test,
        "source": task.source,
        "agent_model": agent_model,
        "judge_model": judge_model,
        "optimizer_model": optimizer_model,
        "seed": seed,
        "eval_repeats": eval_repeats,
        "baseline_prompt": task.baseline_prompt,
        "optimized_prompt": opt["best_prompt"],
        "baseline_test": baseline,
        "final_test": final,
        "delta_objective": final["objective_mean"] - baseline["objective_mean"],
        "delta_gpa": final["gpa_mean"] - baseline["gpa_mean"],
        "optimization": {
            "front": opt["front"], "history": opt["history"],
            "baseline_train_obj": opt["baseline_train_obj"],
            "best_train_obj": opt["best_train_obj"],
        },
        "duration_s": round(time.time() - t0, 1),
    }


__all__ = ["run_tei_on_agent"]
