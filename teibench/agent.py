"""The agent under test + dataset evaluation.

An 'agent' here is the minimal unit that the TEI loop optimises: a single
system prompt driving a task. We deliberately keep the agent a thin
prompt-conditioned LLM call so that the ONLY thing TEI changes is the
prompt — isolating the effect we want to measure. (The full tei-loop tool
also does structural code fixes; for a controlled paired study we hold the
code fixed and vary only the prompt, which is the cleaner scientific
manipulation and the one the poster's headline rests on.)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .gpa_judge import DIMENSIONS, gpa_judge
from .llm import LLM
from .scorers import score_example


@dataclass
class ExampleResult:
    query: str
    gold: str
    output: str
    objective: float
    gpa: dict = field(default_factory=dict)


@dataclass
class SplitEval:
    objective_mean: float
    gpa_mean: float
    gpa_dims: dict
    examples: list[ExampleResult]

    def composite(self, w_obj: float = 0.5) -> float:
        return w_obj * self.objective_mean + (1 - w_obj) * self.gpa_mean


async def run_agent(
    llm: LLM, agent_model: str, system_prompt: str, query: str,
    *, temperature: float = 0.0, max_tokens: int = 800,
) -> str:
    return await llm.complete(
        model=agent_model, system=system_prompt, user=query,
        temperature=temperature, max_tokens=max_tokens,
    )


async def evaluate_split(
    llm: LLM,
    *,
    agent_model: str,
    judge_model: str,
    system_prompt: str,
    task,                      # Task dataclass (see tasks.py)
    examples: list,            # list of {"query","gold"}
    run_judge: bool = True,
    agent_temperature: float = 0.0,
) -> SplitEval:
    """Run the agent on every example, score objective + (optionally) GPA."""
    async def _one(ex) -> ExampleResult:
        out = await run_agent(
            llm, agent_model, system_prompt, ex["query"],
            temperature=agent_temperature,
        )
        obj = score_example(task.metric, out, ex["gold"], task.labels)
        gpa = {}
        if run_judge:
            gpa = await gpa_judge(
                llm, judge_model,
                task_instruction=task.instruction,
                query=ex["query"], gold=str(ex["gold"]), output=out,
            )
        return ExampleResult(
            query=ex["query"], gold=str(ex["gold"]), output=out,
            objective=obj, gpa=gpa,
        )

    results = await asyncio.gather(*[_one(ex) for ex in examples])
    obj_mean = sum(r.objective for r in results) / len(results)
    if run_judge:
        dims = {d: sum(r.gpa.get(d, 0.0) for r in results) / len(results) for d in DIMENSIONS}
        gpa_mean = sum(dims.values()) / len(dims)
    else:
        dims, gpa_mean = {}, 0.0
    return SplitEval(
        objective_mean=obj_mean, gpa_mean=gpa_mean,
        gpa_dims=dims, examples=list(results),
    )


__all__ = ["run_agent", "evaluate_split", "SplitEval", "ExampleResult"]
