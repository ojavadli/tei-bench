"""GEPA-style reflective Pareto prompt optimizer (the 'Improve' in TEI).

Faithful port of the tei-loop optimization strategy, adapted for a
controlled study:

  * Maintains a Pareto front over (objective, gpa_aggregate).
  * Reflective mutation: the optimizer model is shown the current prompt and
    a sample of FAILING train examples (query, agent output, gold) and asked
    to produce an improved system prompt that would fix those failures.
  * System-aware merge: combine two strong parents into one prompt.
  * Candidates are scored on TRAIN minibatches only — the test split is
    never touched here. This is what makes the held-out result meaningful.

Returns the best candidate by composite train score.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
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

    def composite(self, w_obj: float = 0.5) -> float:
        return w_obj * self.objective + (1 - w_obj) * self.gpa

    def dominates(self, other: "Candidate") -> bool:
        return (
            self.objective >= other.objective
            and self.gpa >= other.gpa
            and (self.objective > other.objective or self.gpa > other.gpa)
        )


_REFLECT_SYSTEM = """You are an expert prompt engineer improving the SYSTEM PROMPT
of a task agent. You are given the current system prompt and several examples
where the agent FAILED (its output did not match the reference answer).

Diagnose the failure pattern, then rewrite the system prompt so the agent
would handle these and similar cases correctly. You may add explicit
instructions, output-format constraints, decision rules, or label
definitions. Keep it focused and general (do NOT hard-code answers to the
specific examples — that would not generalise). Return ONLY the improved
system prompt text, nothing else."""

_MERGE_SYSTEM = """You are an expert prompt engineer. You are given two strong
SYSTEM PROMPTS for the same task agent. Produce a single improved system
prompt that combines the best instructions and decision rules from both,
without redundancy or contradiction. Return ONLY the merged system prompt
text, nothing else."""


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
    num_iterations: int = 8,
    minibatch: int = 8,
    seed: int = 0,
    w_obj: float = 0.5,
    log: Optional[list] = None,
) -> dict:
    """Run reflective Pareto optimization on the train split.

    Returns dict with best_prompt, front, history, and the baseline train eval.
    """
    rng = random.Random(seed)
    log = log if log is not None else []

    def _mb() -> list:
        if len(train) <= minibatch:
            return train
        return rng.sample(train, minibatch)

    # Baseline candidate, evaluated on a train minibatch
    base_mb = _mb()
    base_eval = await evaluate_split(
        llm, agent_model=agent_model, judge_model=judge_model,
        system_prompt=baseline_prompt, task=task, examples=base_mb,
    )
    base = Candidate(
        prompt=baseline_prompt, objective=base_eval.objective_mean,
        gpa=base_eval.gpa_mean, iteration=0, strategy="baseline",
    )
    front: list[Candidate] = [base]
    best = base
    history = [{"iter": 0, "obj": base.objective, "gpa": base.gpa,
                "composite": base.composite(w_obj), "strategy": "baseline"}]
    log.append(f"    [opt] baseline train: obj={base.objective:.3f} gpa={base.gpa:.3f}")

    last_fail_eval = base_eval
    for it in range(1, num_iterations + 1):
        strategy = "merge" if (len(front) >= 2 and rng.random() < 0.3) else "mutation"

        if strategy == "merge":
            a, b = rng.sample(front, 2)
            new_prompt = await llm.complete(
                model=optimizer_model, system=_MERGE_SYSTEM,
                user=f"SYSTEM PROMPT A:\n{a.prompt}\n\nSYSTEM PROMPT B:\n{b.prompt}",
                temperature=0.8, max_tokens=1200, nonce=f"merge-{seed}-{it}",
            )
        else:
            parent = max(front, key=lambda c: c.composite(w_obj))
            fails = _failures(last_fail_eval, k=4)
            fail_block = "\n\n".join(
                f"QUERY: {f.query}\nAGENT OUTPUT: {f.output[:300]}\nREFERENCE: {f.gold}"
                for f in fails
            )
            new_prompt = await llm.complete(
                model=optimizer_model, system=_REFLECT_SYSTEM,
                user=(
                    f"TASK: {task.instruction}\n\n"
                    f"CURRENT SYSTEM PROMPT:\n{parent.prompt}\n\n"
                    f"FAILING EXAMPLES:\n{fail_block}\n\n"
                    f"Return the improved system prompt only."
                ),
                temperature=0.8, max_tokens=1200, nonce=f"mut-{seed}-{it}",
            )

        new_prompt = new_prompt.strip()
        if not new_prompt or len(new_prompt) < 20:
            history.append({"iter": it, "skipped": True, "strategy": strategy})
            continue

        mb = _mb()
        ev = await evaluate_split(
            llm, agent_model=agent_model, judge_model=judge_model,
            system_prompt=new_prompt, task=task, examples=mb,
        )
        cand = Candidate(
            prompt=new_prompt, objective=ev.objective_mean,
            gpa=ev.gpa_mean, iteration=it, strategy=strategy,
        )
        last_fail_eval = ev

        # Update Pareto front
        front = [c for c in front if not cand.dominates(c)]
        if not any(c.dominates(cand) for c in front):
            front.append(cand)
        if cand.composite(w_obj) > best.composite(w_obj):
            best = cand

        history.append({
            "iter": it, "obj": cand.objective, "gpa": cand.gpa,
            "composite": cand.composite(w_obj), "strategy": strategy,
            "front_size": len(front),
        })
        log.append(
            f"    [opt] iter {it:>2} {strategy:<8} "
            f"obj={cand.objective:.3f} gpa={cand.gpa:.3f} "
            f"comp={cand.composite(w_obj):.3f} (best={best.composite(w_obj):.3f})"
        )

    return {
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
