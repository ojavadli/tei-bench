"""GPA-style 4-dimension LLM-as-judge (the TEI evaluation framework).

This is the *secondary* endpoint. It scores an agent trace on the four
TEI dimensions from the tei-loop project:

  * target_alignment   — did the output pursue what was actually asked?
  * reasoning_soundness — are the intermediate steps coherent + grounded?
  * execution_accuracy  — were the task mechanics (format, constraints,
                          required fields) performed correctly?
  * output_integrity    — is the output accurate, complete, non-fabricated?

The judge model is deliberately a DIFFERENT, stronger model than the agent
under test (agent = Haiku, judge = Sonnet) to reduce self-preference bias.
The judge is given the gold answer as reference context (reference-based
LLM judging is markedly more reliable than reference-free).

Each dimension is scored 0–100 by the judge and returned in [0,1].
"""
from __future__ import annotations

from .llm import LLM

DIMENSIONS = [
    "target_alignment",
    "reasoning_soundness",
    "execution_accuracy",
    "output_integrity",
]

_JUDGE_SYSTEM = """You are a rigorous, impartial evaluator of AI-agent outputs.
You score outputs on four independent dimensions, each 0-100.

Definitions:
- target_alignment: Did the response address exactly what the task asked,
  without drifting to a different question or adding off-target content?
- reasoning_soundness: Are the stated or implied reasoning steps coherent,
  logically valid, and grounded in the input (no leaps, no contradictions)?
- execution_accuracy: Were the task mechanics performed correctly — correct
  format, required fields present, constraints obeyed, label drawn from the
  allowed set when applicable?
- output_integrity: Is the final answer correct, complete, and free of
  fabrication or hallucination, given the reference answer?

Be calibrated and strict. A correct, well-formed answer that matches the
reference should score 85-100 on the relevant dimensions; a wrong or
fabricated answer should score below 40 on output_integrity regardless of
how fluent it is. Judge the substance, not the verbosity.

Respond with ONLY a JSON object of the form:
{"target_alignment": <int 0-100>, "reasoning_soundness": <int 0-100>,
 "execution_accuracy": <int 0-100>, "output_integrity": <int 0-100>,
 "rationale": "<one sentence>"}"""

_JUDGE_USER = """TASK GIVEN TO THE AGENT:
{task_instruction}

AGENT INPUT (the specific query):
{query}

REFERENCE / GOLD ANSWER (ground truth for your judgement):
{gold}

AGENT OUTPUT (to be scored):
{output}

Score the four dimensions now as JSON."""


async def gpa_judge(
    llm: LLM,
    judge_model: str,
    *,
    task_instruction: str,
    query: str,
    gold: str,
    output: str,
    nonce: str = "",
) -> dict:
    """Return {dimension: score_in_0_1, ..., 'aggregate': float, 'rationale': str}."""
    user = _JUDGE_USER.format(
        task_instruction=task_instruction,
        query=query,
        gold=gold,
        output=output[:4000],
    )
    try:
        data = await llm.complete_json(
            model=judge_model,
            system=_JUDGE_SYSTEM,
            user=user,
            temperature=0.0,
            max_tokens=400,
            nonce=nonce,
        )
    except Exception:
        # Defensive: a malformed judge response scores neutral-low so it
        # neither inflates nor crashes the run. Logged upstream.
        data = {d: 50 for d in DIMENSIONS}

    out = {}
    for d in DIMENSIONS:
        try:
            out[d] = max(0.0, min(1.0, float(data.get(d, 50)) / 100.0))
        except (TypeError, ValueError):
            out[d] = 0.5
    out["aggregate"] = sum(out[d] for d in DIMENSIONS) / len(DIMENSIONS)
    out["rationale"] = str(data.get("rationale", ""))[:200]
    return out


__all__ = ["gpa_judge", "DIMENSIONS"]
