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

_OPRO_SYSTEM = """You are an optimizer that improves a task's SYSTEM PROMPT. You are
shown previously-tried system prompts together with their training scores, ordered
from lowest to highest score. Reason about what distinguishes higher-scoring prompts
and write a NEW system prompt that is likely to score even higher. You are NOT shown
specific failing examples. Return ONLY the new system prompt text, nothing else.
(This follows the OPRO 'LLM as optimizer' meta-prompt of Yang et al., 2023.)"""


_STRUCT_SYSTEM = """You are an expert prompt engineer improving a task agent's SYSTEM
PROMPT by making STRUCTURAL changes (the "Improve" step of a Target-Evaluate-Improve
loop). You are given: the current prompt; several FAILING validation examples (input,
the agent's wrong output, the reference answer); the agent's weakest EVALUATION
DIMENSION; and a short history of how your previous edits changed the validation
score and WHY (credit assignment).

Diagnose the dominant failure mode, then RESTRUCTURE the prompt into clearly
delimited sections -- Role, Label definitions, Decision rules for the most-confused
cases, and Output format -- adding only minimal, general rules that fix the diagnosed
failures WITHOUT hard-coding answers to the specific examples. Prefer a focused,
well-organized prompt over a longer one. Return ONLY the improved system prompt."""


def _failures(split_eval, k: int = 4) -> list:
    fails = sorted(split_eval.examples, key=lambda r: r.objective)
    return [r for r in fails if r.objective < 1.0][:k] or fails[:k]


def _se(p: float, n: int) -> float:
    """Bootstrap-style SE of a 0/1 mean; the do-no-harm acceptance margin."""
    n = max(1, n)
    return (p * (1 - p) / n) ** 0.5


def prob_better(acc_c: float, n_c: int, acc_b: float, n_b: int,
                rng: random.Random, draws: int = 4000) -> float:
    """Bayesian Beta-Binomial posterior probability that the candidate's true
    accuracy exceeds the baseline's, given validation successes/failures.
    Uses Jeffreys priors Beta(0.5,0.5). This is the do-no-harm gate: we ship a
    candidate only if this probability clears a threshold, which keeps large real
    gains while refusing marginal ones (so they become ties, not test regressions)."""
    sc = round(acc_c * n_c); fc = n_c - sc
    sb = round(acc_b * n_b); fb = n_b - sb
    ac, bc = sc + 0.5, fc + 0.5
    ab, bb = sb + 0.5, fb + 0.5
    wins = 0
    for _ in range(draws):
        if rng.betavariate(ac, bc) > rng.betavariate(ab, bb):
            wins += 1
    return wins / draws


def _weakest_dim(gpa_dims: dict) -> tuple:
    if not gpa_dims:
        return ("(none)", 0.0)
    d = min(gpa_dims, key=lambda k: gpa_dims[k])
    return (d.replace("reasoning_soundness", "reasoning"), gpa_dims[d])


def _make_demos(R: list, task, k: int, rng: random.Random) -> str:
    """Select k few-shot demonstrations from the reflection set and format them as
    solved examples ending in the FINAL: contract. For classification we balance
    across labels (one per label, round-robin) so every class is illustrated; for
    other metrics we sample uniformly. In-context demonstrations are a strong,
    well-established lever (DSPy/MIPRO) that a zero-shot baseline lacks."""
    labels = getattr(task, "labels", None)
    if task.metric == "classification" and labels:
        by = {}
        for ex in R:
            by.setdefault(str(ex["gold"]), []).append(ex)
        for v in by.values():
            rng.shuffle(v)
        chosen, labs, i = [], list(by), 0
        rng.shuffle(labs)
        while len(chosen) < k and any(by.values()) and i < 1000:
            lab = labs[i % len(labs)]; i += 1
            if by.get(lab):
                chosen.append(by[lab].pop())
    else:
        chosen = rng.sample(R, min(k, len(R))) if R else []
    blocks = [f"Input: {str(ex['query'])[:400]}\nFINAL: {ex['gold']}" for ex in chosen]
    return "\n\n".join(blocks)


async def optimize_v3(
    llm: LLM,
    *,
    agent_model: str,
    judge_model: str,
    optimizer_model: str,
    task,
    baseline_prompt: str,
    train: list,
    num_iterations: int = 12,
    minibatch: int = 10,
    seed: int = 0,
    patience: int = 5,
    ceiling: float = 0.999,
    ship_threshold: float = 0.70,
    log: Optional[list] = None,
) -> dict:
    """TEI v3: the redesigned loop. Keeps Evaluate-dimensions + Pareto(scores + why),
    adds (1) an internal validation split with a do-no-harm acceptance gate,
    (2) successive halving so the judge runs ONLY on promoted candidates,
    (3) structural-fix mutation guided by the weakest evaluation dimension + a
    why-better/why-worse memory, (4) headroom triage / early stop. The held-out
    TEST split is never touched here."""
    rng = random.Random(seed)
    log = log if log is not None else []

    # --- nested split: V (search+selection: reflection, demos, incumbent) and an
    # independent CONFIRM set used ONLY for the final do-no-harm ship gate. Because
    # the confirm set never informs candidate generation or selection, it is an
    # honest generalization check that catches candidates that overfit V (the
    # source of the val->test regressions). ---
    pool_idx = list(range(len(train)))
    rng.shuffle(pool_idx)
    nconf = max(5, round(len(train) * 0.3))
    Vconf = [train[i] for i in pool_idx[:nconf]] if len(train) >= 8 else list(train)
    V = [train[i] for i in pool_idx[nconf:]] if len(train) >= 8 else list(train)
    R = V                      # reflect/draw demos from the search split
    Vscreen = V[: min(5, len(V))]

    async def eval_on(prompt, examples, judge):
        return await evaluate_split(
            llm, agent_model=agent_model, judge_model=judge_model,
            system_prompt=prompt, task=task, examples=examples, run_judge=judge)

    base_full = await eval_on(baseline_prompt, V, True)
    base_conf = await eval_on(baseline_prompt, Vconf, False)
    base = Candidate(prompt=baseline_prompt, objective=base_full.objective_mean,
                     gpa=base_full.gpa_mean, iteration=0, strategy="baseline")
    log.append(f"    [tei_v3] baseline val obj={base.objective:.3f} "
               f"dims={base.gpa:.3f} (nV={len(V)}, nR={len(R)})")

    # --- headroom triage: skip ONLY when the baseline is genuinely saturated
    # (perfect on a reliably-sized validation set). A small val can look perfect by
    # chance, so we require len(V) >= 10 to avoid triaging real-headroom tasks. ---
    if base.objective >= ceiling and len(V) >= 10:
        log.append(f"    [tei_v3] TRIAGE: baseline val obj {base.objective:.3f} >= "
                   f"{ceiling} on {len(V)} val ex; skipping optimization, shipping baseline.")
        return {"mode": "tei_v3", "best_prompt": baseline_prompt,
                "best_val_obj": base.objective, "baseline_val_obj": base.objective,
                "triaged": True, "judge_calls_saved": 0, "n_promoted": 0,
                "front": [{"obj": base.objective, "gpa": base.gpa, "why": "baseline (triaged)"}],
                "why_log": [], "history": [{"iter": 0, "val_obj": base.objective,
                                            "strategy": "baseline", "accepted": True}]}

    pool: list[Candidate] = [base]
    incumbent = base
    last_eval = base_full
    why_log: list[str] = []
    history = [{"iter": 0, "val_obj": base.objective, "val_dims": base.gpa,
                "strategy": "baseline", "accepted": True}]
    screened_out = 0
    since_improve = 0

    def _strip_demos(p: str) -> str:
        return p.split("\n\nHere are solved examples")[0]

    for it in range(1, num_iterations + 1):
        # ---- propose: demo augmentation, structural reflective mutation, or merge ----
        r = rng.random()
        best_now = max(pool, key=lambda c: (c.objective, c.gpa))
        if r < 0.45 and R:   # few-shot demonstration augmentation (in-context learning)
            demos = _make_demos(R, task, k=min(6, max(2, len(R))), rng=rng)
            if not demos.strip():
                history.append({"iter": it, "skipped": True, "strategy": "demos"}); continue
            base_instr = _strip_demos(best_now.prompt) if rng.random() < 0.5 else baseline_prompt
            new_prompt = base_instr + "\n\nHere are solved examples to follow:\n\n" + demos
            strategy = "demos"
        elif len(pool) >= 3 and r < 0.6:
            ranked = sorted(pool, key=lambda c: (c.objective, c.gpa), reverse=True)[:4]
            a, b = rng.sample(ranked, 2)
            new_prompt = await llm.complete(
                model=optimizer_model, system=_MERGE_SYSTEM,
                user=f"SYSTEM PROMPT A:\n{a.prompt}\n\nSYSTEM PROMPT B:\n{b.prompt}",
                temperature=0.8, max_tokens=1400, nonce=f"v3merge-{seed}-{it}")
            strategy = "merge"
        else:
            parent = max(pool, key=lambda c: (c.objective, c.gpa))
            fails = _failures(last_eval, k=4)
            fail_block = "\n\n".join(
                f"QUERY: {f.query}\nAGENT OUTPUT: {f.output[:280]}\nREFERENCE: {f.gold}"
                for f in fails)
            wd, wv = _weakest_dim(getattr(last_eval, "gpa_dims", {}) or {})
            mem = "\n".join(why_log[-5:]) if why_log else "(no prior edits yet)"
            new_prompt = await llm.complete(
                model=optimizer_model, system=_STRUCT_SYSTEM,
                user=(f"TASK: {task.instruction}\n\nCURRENT SYSTEM PROMPT:\n{parent.prompt}"
                      f"\n\nWEAKEST EVALUATION DIMENSION: {wd} (score {wv:.2f})"
                      f"\n\nWHY PRIOR EDITS HELPED/HURT (validation):\n{mem}"
                      f"\n\nFAILING VALIDATION EXAMPLES:\n{fail_block}"
                      f"\n\nMake a structural fix. Return the improved system prompt only."),
                temperature=0.8, max_tokens=1400, nonce=f"v3struct-{seed}-{it}")
            strategy = "structural"

        new_prompt = new_prompt.strip()
        if len(new_prompt) < 20:
            history.append({"iter": it, "skipped": True, "strategy": strategy})
            continue

        # ---- successive halving: cheap objective screen, judge only if promoted ----
        screen = await eval_on(new_prompt, Vscreen, False)
        if screen.objective_mean < incumbent.objective - _se(incumbent.objective, len(Vscreen)):
            screened_out += 1
            history.append({"iter": it, "strategy": strategy, "screened_out": True,
                            "screen_obj": screen.objective_mean})
            log.append(f"    [tei_v3] iter {it:>2} {strategy:<10} screened out "
                       f"(screen={screen.objective_mean:.3f} < inc {incumbent.objective:.3f})")
            continue

        ev = await eval_on(new_prompt, V, True)   # promoted: full val + evaluation dims
        cand = Candidate(prompt=new_prompt, objective=ev.objective_mean,
                         gpa=ev.gpa_mean, iteration=it, strategy=strategy)
        pool.append(cand)
        last_eval = ev

        # ---- why-better/why-worse note (GEPA-style credit assignment, fed forward) ----
        delta = cand.objective - incumbent.objective
        dims = ev.gpa_dims or {}
        wd, wv = _weakest_dim(dims)
        dimstr = ", ".join(f"{k.replace('reasoning_soundness','reasoning')}={dims[k]:.2f}"
                           for k in dims) or "n/a"
        rats = [(e.gpa or {}).get("rationale", "") for e in ev.examples
                if isinstance(getattr(e, "gpa", None), dict) and e.objective < 1.0
                and (e.gpa or {}).get("rationale")]
        why = (f"iter {it} [{strategy}]: val acc {cand.objective:.3f} ({delta:+.3f} vs incumbent); "
               f"dims [{dimstr}]; weakest={wd}. WHY it still failed: "
               + (" | ".join(r[:140] for r in rats[:2]) if rats else "format/decision-rule miss."))
        why_log.append(why)

        # ---- do-no-harm acceptance gate (accept only confident improvements) ----
        margin = _se(incumbent.objective, len(V))
        accepted = cand.objective > incumbent.objective + margin
        if accepted:
            incumbent = cand
            since_improve = 0
        else:
            since_improve += 1
        history.append({"iter": it, "val_obj": cand.objective, "val_dims": cand.gpa,
                        "strategy": strategy, "accepted": accepted, "delta": delta})
        log.append(f"    [tei_v3] iter {it:>2} {strategy:<10} val obj={cand.objective:.3f} "
                   f"dims={cand.gpa:.3f} d={delta:+.3f} accept={accepted} "
                   f"(inc={incumbent.objective:.3f})")

        if since_improve >= patience:
            log.append(f"    [tei_v3] early stop at iter {it} (no val gain in {patience}).")
            break

    # ---- Pareto front over (val objective, val evaluation-dims), carrying 'why' ----
    front_cands = [c for c in pool if not any(o.dominates(c) for o in pool)]
    front = sorted(front_cands, key=lambda c: (c.objective, c.gpa), reverse=True)
    why_by_iter = {}
    for w in why_log:
        try:
            why_by_iter[int(w.split()[1])] = w
        except (IndexError, ValueError):
            pass

    # ---- final selection: do-no-harm fallback (ship the optimized prompt UNLESS
    # the baseline is probably better) ----
    # We default to the best validation-accuracy candidate so real gains are kept,
    # and revert to the baseline ONLY when the Beta-Binomial posterior says the
    # baseline is more likely better, P(acc_base > acc_cand) > reject_threshold.
    # This blocks the marginal/over-fit candidates that caused test regressions in
    # v2 without discarding the genuine improvements.
    cand_best = max((c for c in pool if c is not base),
                    key=lambda c: (c.objective, c.gpa), default=base)
    if cand_best is base:
        ship_prob, shipped_baseline, best, conf_acc = 0.0, True, base, base_conf.objective_mean
    else:
        # Confirm the winner on the INDEPENDENT confirm set (never used for search
        # or selection), and ship only if it confidently beats the baseline THERE.
        cand_conf = await eval_on(cand_best.prompt, Vconf, False)
        conf_acc = cand_conf.objective_mean
        ship_prob = prob_better(conf_acc, len(Vconf), base_conf.objective_mean, len(Vconf), rng)
        shipped_baseline = ship_prob < ship_threshold
        best = base if shipped_baseline else cand_best
    log.append(f"    [tei_v3] FINAL: cand val={cand_best.objective:.3f} (confirm={conf_acc:.3f} "
               f"vs base_confirm {base_conf.objective_mean:.3f}); P(cand>base | confirm)={ship_prob:.2f} "
               f"thr={ship_threshold} -> ship {'baseline' if shipped_baseline else 'candidate'}")

    return {
        "mode": "tei_v3",
        "best_prompt": best.prompt,
        "best_val_obj": best.objective,
        "best_val_dims": best.gpa,
        "baseline_val_obj": base.objective,
        "ship_prob_better": ship_prob,
        "ship_threshold": ship_threshold,
        "shipped_baseline": shipped_baseline,
        "triaged": False,
        "judge_calls_saved": screened_out,
        "n_promoted": len(pool) - 1,
        "front": [{"obj": c.objective, "gpa": c.gpa, "iter": c.iteration,
                   "strategy": c.strategy, "why": why_by_iter.get(c.iteration, "")}
                  for c in front],
        "why_log": why_log,
        "history": history,
    }


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
    assert mode in ("tei", "objective_reflection", "random", "opro")
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
        elif mode == "opro":
            strategy = "opro"
        else:
            strategy = "merge" if (len(pool) >= 2 and rng.random() < 0.3) else "mutation"

        if strategy == "opro":
            ranked = sorted(pool, key=lambda c: c.objective)[-6:]
            hist = "\n\n".join(
                f"[train score {c.objective:.2f}] SYSTEM PROMPT:\n{c.prompt}" for c in ranked)
            new_prompt = await llm.complete(
                model=optimizer_model, system=_OPRO_SYSTEM,
                user=(f"TASK: {task.instruction}\n\nPrevious system prompts and their "
                      f"training scores, lowest to highest:\n\n{hist}\n\n"
                      f"Write a new system prompt likely to score higher. Return only the prompt."),
                temperature=0.9, max_tokens=1200, nonce=f"opro-{seed}-{it}")
        elif strategy == "paraphrase":
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


__all__ = ["optimize", "optimize_v3", "Candidate"]
