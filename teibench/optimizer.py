"""Prompt optimizer with three modes, for the ablation the reviewers asked for.

  * mode="tei"                  : reflective mutation on FAILING train examples
                                  + system-aware merge, selection by a Pareto
                                  front over (objective, GPA). The full method.
  * mode="objective_reflection" : same reflective mutation + merge, but selection
                                  by OBJECTIVE ONLY (no GPA). Isolates the value
                                  of the GPA-guided evaluation signal.
  * mode="random"               : undirected paraphrase of the prompt (the
                                  optimizer never sees failures), selection by
                                  objective only. Tests whether TEI beats random
                                  prompt search of the same budget.

All modes use the SAME iteration budget and the SAME train minibatches, and
NONE of them touch the test split. Only mode="tei" calls the GPA judge during
optimization (the other arms are objective-only by construction, which also
makes them cheaper).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .agent import evaluate_split
from .llm import LLM


@dataclass
class Candidate:
    prompt: str
    objective: float = 0.0
    gpa: float = 0.0
    iteration: int = 0
    strategy: str = "baseline"

    def composite(self, w_obj: float) -> float:
        return w_obj * self.objective + (1 - w_obj) * self.gpa

    def dominates(self, other: "Candidate") -> bool:
        return (self.objective >= other.objective and self.gpa >= other.gpa
                and (self.objective > other.objective or self.gpa > other.gpa))


_REFLECT_SYSTEM = """You are an expert prompt engineer improving the SYSTEM PROMPT
of a task agent. You are given the current system prompt and several examples
where the agent FAILED (its output did not match the reference answer).

Diagnose the failure pattern, then rewrite the system prompt so the agent would
handle these and similar cases correctly. You may add explicit instructions,
output-format constraints, decision rules, or label definitions. Keep it focused
and general (do NOT hard-code answers to the specific examples). Return ONLY the
improved system prompt text, nothing else."""

_MERGE_SYSTEM = """You are an expert prompt engineer. You are given two strong
SYSTEM PROMPTS for the same task agent. Produce a single improved system prompt
that combines the best instructions and decision rules from both, without
redundancy or contradiction. Return ONLY the merged system prompt text."""

_PARAPHRASE_SYSTEM = """You are an expert prompt engineer. Rewrite the SYSTEM PROMPT
below into a different but equally reasonable phrasing for the same task. You may
restructure, rephrase, and lightly expand instructions, but you are NOT given any
information about which examples the agent got wrong. Return ONLY the rewritten
system prompt text, nothing else."""


def _failures(split_eval, k: int = 4) -> list:
    fails = sorted(split_eval.examples, key=lambda r: r.objective)
    return [r for r in fails if r.objective < 1.0][:k] or fails[:k]


async def optimize(
    llm: LLM,
    *,
    agent_model: str,
    judge_model: str,
    optimizer_model: str,
    task,
    baseline_prompt: str,
    train: list,
    mode: str = "tei",
    num_iterations: int = 6,
    minibatch: int = 10,
    seed: int = 0,
    log: Optional[list] = None,
) -> dict:
    """Optimize the prompt on the train split under the given ablation mode."""
    assert mode in ("tei", "objective_reflection", "random")
    rng = random.Random(seed)
    log = log if log is not None else []
    run_judge = (mode == "tei")
    w_obj = 0.5 if mode == "tei" else 1.0
    score = lambda c: c.composite(w_obj)

    def _mb() -> list:
        return train if len(train) <= minibatch else rng.sample(train, minibatch)

    base_eval = await evaluate_split(
        llm, agent_model=agent_model, judge_model=judge_model,
        system_prompt=baseline_prompt, task=task, examples=_mb(), run_judge=run_judge)
    base = Candidate(prompt=baseline_prompt, objective=base_eval.objective_mean,
                     gpa=base_eval.gpa_mean, iteration=0, strategy="baseline")
    pool: list[Candidate] = [base]
    best = base
    history = [{"iter": 0, "obj": base.objective, "gpa": base.gpa, "strategy": "baseline"}]
    last_eval = base_eval
    log.append(f"    [opt:{mode}] baseline train obj={base.objective:.3f} gpa={base.gpa:.3f}")

    for it in range(1, num_iterations + 1):
        if mode == "random":
            strategy = "paraphrase"
        else:
            strategy = "merge" if (len(pool) >= 2 and rng.random() < 0.3) else "mutation"

        if strategy == "paraphrase":
            src = rng.choice(pool)
            new_prompt = await llm.complete(
                model=optimizer_model, system=_PARAPHRASE_SYSTEM,
                user=f"SYSTEM PROMPT:\n{src.prompt}", temperature=0.9,
                max_tokens=1200, nonce=f"para-{seed}-{it}")
        elif strategy == "merge":
            ranked = sorted(pool, key=score, reverse=True)[:4]
            a, b = rng.sample(ranked, 2) if len(ranked) >= 2 else (ranked[0], ranked[0])
            new_prompt = await llm.complete(
                model=optimizer_model, system=_MERGE_SYSTEM,
                user=f"SYSTEM PROMPT A:\n{a.prompt}\n\nSYSTEM PROMPT B:\n{b.prompt}",
                temperature=0.8, max_tokens=1200, nonce=f"merge-{seed}-{it}")
        else:  # mutation (reflective)
            parent = max(pool, key=score)
            fails = _failures(last_eval, k=4)
            fail_block = "\n\n".join(
                f"QUERY: {f.query}\nAGENT OUTPUT: {f.output[:300]}\nREFERENCE: {f.gold}"
                for f in fails)
            new_prompt = await llm.complete(
                model=optimizer_model, system=_REFLECT_SYSTEM,
                user=(f"TASK: {task.instruction}\n\nCURRENT SYSTEM PROMPT:\n{parent.prompt}"
                      f"\n\nFAILING EXAMPLES:\n{fail_block}\n\nReturn the improved system prompt only."),
                temperature=0.8, max_tokens=1200, nonce=f"mut-{seed}-{it}")

        new_prompt = new_prompt.strip()
        if not new_prompt or len(new_prompt) < 20:
            history.append({"iter": it, "skipped": True, "strategy": strategy})
            continue

        ev = await evaluate_split(
            llm, agent_model=agent_model, judge_model=judge_model,
            system_prompt=new_prompt, task=task, examples=_mb(), run_judge=run_judge)
        cand = Candidate(prompt=new_prompt, objective=ev.objective_mean,
                         gpa=ev.gpa_mean, iteration=it, strategy=strategy)
        last_eval = ev
        pool.append(cand)
        if score(cand) > score(best):
            best = cand
        history.append({"iter": it, "obj": cand.objective, "gpa": cand.gpa,
                        "strategy": strategy})
        log.append(f"    [opt:{mode}] iter {it:>2} {strategy:<10} obj={cand.objective:.3f} "
                   f"gpa={cand.gpa:.3f} (best obj={best.objective:.3f})")

    if mode == "tei":
        front = [c for c in pool if not any(o.dominates(c) for o in pool)]
    else:
        front = sorted(pool, key=score, reverse=True)[:5]

    return {
        "mode": mode,
        "best_prompt": best.prompt,
        "best_train_obj": best.objective,
        "best_train_gpa": best.gpa,
        "baseline_train_obj": base.objective,
        "baseline_train_gpa": base.gpa,
        "front": [{"obj": c.objective, "gpa": c.gpa, "iter": c.iteration,
                   "strategy": c.strategy} for c in front],
        "history": history,
    }


__all__ = ["optimize", "Candidate"]
