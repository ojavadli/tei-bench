"""Objective, code-computed scorers (NO LLM involved).

These produce the *primary* endpoint. They are deterministic functions of
(prediction, gold) with no model in the loop.

Scoring contract (v2 — removes the format confound flagged in review):
Every agent, in EVERY experimental condition, is instructed to end its reply
with a single line `FINAL: <answer>`. The scorer reads ONLY that final line.
This eliminates the previous "latest label mention in the reasoning" artifact,
where a verbose baseline that named other labels mid-reasoning was mis-scored.
Because the same contract is applied identically to baseline and optimized
conditions, the paired comparison stays fair while the artifact is removed.

A conservative fallback (last non-empty line, then whole-text) handles outputs
that fail to emit a FINAL line, so non-compliant responses are still scored.

Each scorer returns a float in [0, 1].
"""
from __future__ import annotations

import re
from typing import Sequence


def _norm(s: str) -> str:
    # Treat underscores/hyphens as spaces so label identity is compared, not
    # punctuation style (gold 'card_arrival' == output 'card arrival').
    s = str(s).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)


_FINAL_RE = re.compile(r"^\s*final\s*(?:answer)?\s*[:\-]\s*(.+?)\s*$",
                       re.IGNORECASE | re.MULTILINE)


def build_output_contract(metric: str, labels: Sequence[str] | None) -> str:
    """The universal output-format instruction appended to EVERY condition's
    system prompt. Identical across baseline and all optimized arms."""
    if metric == "classification":
        opts = ", ".join(labels or [])
        return (
            "\n\nIMPORTANT OUTPUT FORMAT: You may reason briefly, but you MUST end "
            "your response with a single final line in EXACTLY this format and "
            "write nothing after it:\nFINAL: <label>\n"
            f"where <label> is copied verbatim from this set: {opts}."
        )
    if metric == "numeric":
        return (
            "\n\nIMPORTANT OUTPUT FORMAT: You may show work, but you MUST end your "
            "response with a single final line in EXACTLY this format and write "
            "nothing after it:\nFINAL: <number>\n"
            "where <number> is only the final numeric answer (digits only)."
        )
    return ""


def _final_span(output: str) -> str | None:
    """Return the content of the LAST `FINAL:` line, if present."""
    matches = _FINAL_RE.findall(output or "")
    if matches:
        return matches[-1].strip()
    return None


# ----------------------------- classification -----------------------------

def _match_label(span: str, labels: Sequence[str]) -> str | None:
    """Find which allowed label the span denotes (exact, then unique substring)."""
    ns = _norm(span)
    norm_labels = {lbl: _norm(lbl) for lbl in labels}
    # exact match
    for lbl, nlbl in norm_labels.items():
        if ns == nlbl:
            return lbl
    # unique substring: span contains exactly one label (longest wins on ties)
    contained = [lbl for lbl, nlbl in norm_labels.items() if nlbl and nlbl in ns]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        # prefer the longest label name (most specific)
        return max(contained, key=lambda l: len(norm_labels[l]))
    # label contains span (model abbreviated)
    rev = [lbl for lbl, nlbl in norm_labels.items() if ns and ns in nlbl]
    if len(rev) == 1:
        return rev[0]
    return None


def _fallback_label(output: str, labels: Sequence[str]) -> str | None:
    """No FINAL line: scan last non-empty line, then whole text (first label)."""
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    for span in ([lines[-1]] if lines else []) + [output or ""]:
        lbl = _match_label(span, labels)
        if lbl:
            return lbl
    return None


def score_classification(output: str, gold: str, labels: Sequence[str]) -> float:
    span = _final_span(output)
    pred = _match_label(span, labels) if span is not None else None
    if pred is None:
        pred = _fallback_label(output, labels)
    if pred is None:
        return 0.0
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


# --------------------------------- numeric ---------------------------------

_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _num(text: str) -> float | None:
    cands = _NUM_RE.findall((text or "").replace(",", ""))
    if not cands:
        return None
    try:
        return float(re.sub(r"[^\d.\-]", "", cands[-1]))
    except ValueError:
        return None


def score_numeric(output: str, gold: str, tol: float = 1e-6) -> float:
    span = _final_span(output)
    pred = _num(span) if span is not None else None
    if pred is None:
        pred = _num(output)
    g = _num(str(gold))
    if pred is None or g is None:
        return 0.0
    return 1.0 if abs(pred - g) <= tol * max(1.0, abs(g)) else 0.0


# --------------------------- token-f1 / contains ---------------------------

def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))


def score_token_f1(output: str, gold: str) -> float:
    span = _final_span(output)
    text = span if span is not None else output
    pred_t, gold_t = _tokens(text), _tokens(gold)
    if not pred_t and not gold_t:
        return 1.0
    if not pred_t or not gold_t:
        return 0.0
    gold_count: dict[str, int] = {}
    for t in gold_t:
        gold_count[t] = gold_count.get(t, 0) + 1
    pred_count: dict[str, int] = {}
    for t in pred_t:
        pred_count[t] = pred_count.get(t, 0) + 1
    overlap = sum(min(c, gold_count.get(t, 0)) for t, c in pred_count.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_t)
    recall = overlap / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def score_contains_all(output: str, required: Sequence[str]) -> float:
    if not required:
        return 1.0
    no = _norm(output)
    hits = sum(1 for r in required if _norm(r) in no)
    return hits / len(required)


def score_example(metric: str, output: str, gold, labels=None) -> float:
    if metric == "classification":
        return score_classification(output, gold, labels or [])
    if metric == "numeric":
        return score_numeric(output, str(gold))
    if metric == "token_f1":
        return score_token_f1(output, str(gold))
    if metric == "contains_all":
        req = gold if isinstance(gold, (list, tuple)) else [gold]
        return score_contains_all(output, req)
    raise ValueError(f"Unknown metric: {metric}")


__all__ = [
    "build_output_contract", "score_classification", "score_numeric",
    "score_token_f1", "score_contains_all", "score_example",
]
