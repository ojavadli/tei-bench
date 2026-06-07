"""LLM foundation layer for TEI-Bench.

A thin, instrumented wrapper around the Anthropic API with:
  * token + cost accounting (per-model, aggregate)
  * exponential-backoff retries on transient errors
  * an on-disk response cache keyed by the full call signature, so re-runs
    are cheap and the experiment is replayable without re-spending budget
  * a global concurrency semaphore

Design note (scientific validity): the cache key includes a `nonce` field.
Deterministic calls (agent at temp 0, judge at temp 0) use nonce="" so they
hit the cache. Stochastic calls (optimizer proposing candidate prompts at
high temperature) pass a unique nonce per iteration so each is a distinct,
replayable sample rather than collapsing to one cached response.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic


# Approximate Anthropic list prices (USD per 1M tokens) as of mid-2026.
# Used ONLY for compute-budget reporting; scientific results do not depend
# on these numbers. Update freely.
PRICES = {
    "claude-haiku-4-5":   {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-5":  {"in": 3.00, "out": 15.00},
    "claude-sonnet-4-6":  {"in": 3.00, "out": 15.00},
    "claude-opus-4-1":    {"in": 15.00, "out": 75.00},
    # OpenAI (approx, for budget reporting only)
    "gpt-4o-mini":        {"in": 0.15, "out": 0.60},
    "gpt-4.1-mini":       {"in": 0.40, "out": 1.60},
    "gpt-4o":             {"in": 2.50, "out": 10.00},
    "gpt-4.1":            {"in": 2.00, "out": 8.00},
}


def _is_openai(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))


def _price(model: str) -> dict:
    for k, v in PRICES.items():
        if model.startswith(k):
            return v
    return {"in": 3.00, "out": 15.00}


@dataclass
class Usage:
    """Mutable token + cost accumulator, broken down by model."""
    by_model: dict = field(default_factory=dict)
    calls: int = 0
    cache_hits: int = 0

    def add(self, model: str, in_tok: int, out_tok: int, cached: bool) -> None:
        self.calls += 1
        if cached:
            self.cache_hits += 1
            return
        m = self.by_model.setdefault(model, {"in": 0, "out": 0})
        m["in"] += in_tok
        m["out"] += out_tok

    def cost_usd(self) -> float:
        total = 0.0
        for model, t in self.by_model.items():
            p = _price(model)
            total += t["in"] / 1e6 * p["in"] + t["out"] / 1e6 * p["out"]
        return total

    def tokens(self) -> tuple[int, int]:
        return (
            sum(t["in"] for t in self.by_model.values()),
            sum(t["out"] for t in self.by_model.values()),
        )

    def report(self) -> str:
        ti, to = self.tokens()
        return (
            f"calls={self.calls} cache_hits={self.cache_hits} "
            f"in_tok={ti:,} out_tok={to:,} cost=${self.cost_usd():.4f}"
        )


class LLM:
    """Instrumented Anthropic client with disk cache + cost accounting."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        max_concurrency: int = 6,
        usage: Optional[Usage] = None,
    ):
        self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._openai = None  # created lazily on first OpenAI call
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max_concurrency)
        self.usage = usage or Usage()

    def _openai_client(self):
        if self._openai is None:
            from openai import AsyncOpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set (needed for gpt-* models)")
            self._openai = AsyncOpenAI(api_key=key)
        return self._openai

    def _key(self, model, system, user, temperature, max_tokens, nonce) -> str:
        blob = json.dumps(
            [model, system, user, temperature, max_tokens, nonce],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    async def _call_openai(self, model, system, user, temperature, max_tokens):
        """OpenAI chat completion with param fallbacks for newer models."""
        client = self._openai_client()
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        params = {"model": model, "messages": msgs,
                  "max_tokens": max_tokens, "temperature": temperature}
        try:
            resp = await client.chat.completions.create(**params)
        except Exception as e:
            err = str(e).lower()
            if "max_tokens" in err and "max_completion_tokens" in err:
                params.pop("max_tokens"); params["max_completion_tokens"] = max_tokens
            if "temperature" in err:
                params.pop("temperature", None)
            resp = await client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        nonce: str = "",
        max_retries: int = 5,
    ) -> str:
        """Return the model's text completion, using cache when possible."""
        key = self._key(model, system, user, temperature, max_tokens, nonce)
        cpath = self._cache_path(key)
        if cpath.exists():
            try:
                data = json.loads(cpath.read_text(encoding="utf-8"))
                self.usage.add(model, data.get("in", 0), data.get("out", 0), cached=True)
                return data["text"]
            except Exception:
                pass

        delay = 2.0
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                async with self._sem:
                    if _is_openai(model):
                        text, in_tok, out_tok = await self._call_openai(
                            model, system, user, temperature, max_tokens)
                    else:
                        resp = await self._client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            system=system,
                            messages=[{"role": "user", "content": user}],
                            temperature=temperature,
                        )
                        text = resp.content[0].text if resp.content else ""
                        in_tok = resp.usage.input_tokens if resp.usage else 0
                        out_tok = resp.usage.output_tokens if resp.usage else 0
                self.usage.add(model, in_tok, out_tok, cached=False)
                cpath.write_text(
                    json.dumps({"text": text, "in": in_tok, "out": out_tok}),
                    encoding="utf-8",
                )
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                # Retry on rate limits / overload / transient 5xx
                if any(s in msg for s in ("rate", "overload", "529", "500", "502", "503", "timeout", "connection")):
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")

    async def complete_json(self, **kwargs) -> dict:
        """Completion that must return a JSON object. Strips code fences."""
        text = await self.complete(**kwargs)
        return parse_json(text)


def parse_json(text: str) -> dict:
    """Best-effort JSON extraction from a model response."""
    t = text.strip()
    if t.startswith("```"):
        # strip ```json ... ``` fences
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # find first { and last }
    if not t.startswith("{"):
        i = t.find("{")
        if i >= 0:
            t = t[i:]
    if not t.endswith("}"):
        j = t.rfind("}")
        if j >= 0:
            t = t[: j + 1]
    return json.loads(t)


def parse_json_list(text: str) -> list:
    """Best-effort extraction of a JSON list of strings from a model response."""
    import json as _json
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        t = t[i : j + 1]
    try:
        data = _json.loads(t)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback: split lines that look like quoted/bulleted items
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if len(line) >= 2 and line[0] in "\"'" and line[-1] in "\"'":
            items.append(line[1:-1])
    return items


__all__ = ["LLM", "Usage", "parse_json", "parse_json_list", "PRICES"]
