"""Objective, code-computed scorers (NO LLM involved).

These produce the *primary* endpoint for the study. Because they are
deterministic functions of (prediction, gold) with no model in the loop,
they are immune to the "LLM-judge is circular / self-preferring" critique
that applies to subjective evaluation.

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


def _extract_tail_label(output: str, labels: Sequence[str]) -> str | None:
    """Deterministically extract a predicted label from free-form text.

    Strategy (no LLM): normalise; if the text has an 'answer:'/'label:'/
    'category:' marker, search the span after the LAST such marker first;
    otherwise search the whole text. Among allowed labels that appear as a
    word, pick the one whose last occurrence is latest (models tend to end
    with their final answer).
    """
    norm_out = _norm(output)
    norm_labels = {lbl: _norm(lbl) for lbl in labels}

    spans = [norm_out]
    m = list(re.finditer(r"(?:answer|label|category|classification|intent)\s*[:\-]", norm_out))
    if m:
        spans.insert(0, norm_out[m[-1].end():])

    for span in spans:
        best, best_pos = None, -1
        for lbl, nlbl in norm_labels.items():
            # word-ish boundary match
            for mm in re.finditer(re.escape(nlbl), span):
                if mm.start() > best_pos:
                    best, best_pos = lbl, mm.start()
        if best is not None:
            return best
    return None


def score_classification(output: str, gold: str, labels: Sequence[str]) -> float:
    pred = _extract_tail_label(output, labels)
    if pred is None:
        return 0.0
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _last_number(text: str) -> float | None:
    cands = _NUM_RE.findall(text.replace(",", ""))
    if not cands:
        return None
    try:
        return float(re.sub(r"[^\d.\-]", "", cands[-1]))
    except ValueError:
        return None


def score_numeric(output: str, gold: str, tol: float = 1e-6) -> float:
    pred = _last_number(output)
    g = _last_number(str(gold))
    if pred is None or g is None:
        return 0.0
    return 1.0 if abs(pred - g) <= tol * max(1.0, abs(g)) else 0.0


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))


def score_token_f1(output: str, gold: str) -> float:
    """Token-level F1 — standard for short-answer QA / extraction."""
    pred_t = _tokens(output)
    gold_t = _tokens(gold)
    if not pred_t and not gold_t:
        return 1.0
    if not pred_t or not gold_t:
        return 0.0
    common: dict[str, int] = {}
    gold_count: dict[str, int] = {}
    for t in gold_t:
        gold_count[t] = gold_count.get(t, 0) + 1
    overlap = 0
    pred_count: dict[str, int] = {}
    for t in pred_t:
        pred_count[t] = pred_count.get(t, 0) + 1
    for t, c in pred_count.items():
        overlap += min(c, gold_count.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_t)
    recall = overlap / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def score_contains_all(output: str, required: Sequence[str]) -> float:
    """Fraction of required substrings present (case-insensitive)."""
    if not required:
        return 1.0
    no = _norm(output)
    hits = sum(1 for r in required if _norm(r) in no)
    return hits / len(required)


def score_example(metric: str, output: str, gold, labels=None) -> float:
    """Dispatch to the right scorer based on the task's declared metric."""
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
    "score_classification", "score_numeric", "score_token_f1",
    "score_contains_all", "score_example",
]
