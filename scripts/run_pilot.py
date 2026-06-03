"""Run the TEI pilot on the 3 pilot tasks with REAL Claude calls.

Measures real per-agent cost + time so we can project the full 31-agent run.
Writes one results JSON per agent to results/ and a pilot summary.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench import LLM, Usage, load_all_tasks, run_tei_on_agent
from teibench.stats import analyze_paired

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-sonnet-4-5"
OPT_MODEL = "claude-sonnet-4-5"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="task_ids to run")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    usage = Usage()
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=args.concurrency, usage=usage)

    tasks = load_all_tasks(TASKS_DIR)
    if args.only:
        tasks = [t for t in tasks if t.task_id in args.only]
    print(f"Running pilot on {len(tasks)} agents: {[t.task_id for t in tasks]}")
    print(f"agent={AGENT_MODEL} judge={JUDGE_MODEL} opt={OPT_MODEL} "
          f"iters={args.iterations} repeats={args.repeats}\n")

    records = []
    log: list[str] = []
    t0 = time.time()
    for t in tasks:
        rec = await run_tei_on_agent(
            llm, t, agent_model=AGENT_MODEL, judge_model=JUDGE_MODEL,
            optimizer_model=OPT_MODEL, num_iterations=args.iterations,
            minibatch=args.minibatch, eval_repeats=args.repeats, seed=0, log=log,
        )
        records.append(rec)
        (RESULTS_DIR / f"{rec['task_id']}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n".join(log)); log.clear()
        print(f"    [usage so far] {usage.report()}\n")

    # Pilot-level paired analysis (small n — just a sanity signal)
    before_obj = [r["baseline_test"]["objective_mean"] for r in records]
    after_obj = [r["final_test"]["objective_mean"] for r in records]
    before_gpa = [r["baseline_test"]["gpa_mean"] for r in records]
    after_gpa = [r["final_test"]["gpa_mean"] for r in records]

    summary = {
        "n_agents": len(records),
        "models": {"agent": AGENT_MODEL, "judge": JUDGE_MODEL, "optimizer": OPT_MODEL},
        "config": vars(args),
        "objective": analyze_paired(before_obj, after_obj) if len(records) >= 2 else None,
        "gpa": analyze_paired(before_gpa, after_gpa) if len(records) >= 2 else None,
        "per_agent": [
            {"task_id": r["task_id"], "industry": r["industry"],
             "obj_before": r["baseline_test"]["objective_mean"],
             "obj_after": r["final_test"]["objective_mean"],
             "gpa_before": r["baseline_test"]["gpa_mean"],
             "gpa_after": r["final_test"]["gpa_mean"],
             "duration_s": r["duration_s"]}
            for r in records
        ],
        "usage": {
            "calls": usage.calls, "cache_hits": usage.cache_hits,
            "tokens_in": usage.tokens()[0], "tokens_out": usage.tokens()[1],
            "cost_usd": round(usage.cost_usd(), 4),
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("PILOT SUMMARY")
    print("=" * 64)
    for pa in summary["per_agent"]:
        print(f"  {pa['task_id']:<22} obj {pa['obj_before']:.3f}→{pa['obj_after']:.3f}  "
              f"gpa {pa['gpa_before']:.3f}→{pa['gpa_after']:.3f}  ({pa['duration_s']}s)")
    if summary["objective"]:
        o = summary["objective"]
        print(f"\n  Objective Δ mean = {o['mean_delta']:+.3f}  "
              f"(t={o['t_stat']:.2f}, p={o['t_p_value']:.4f}, d_z={o['cohen_dz']:.2f})")
    print(f"\n  Cost: ${summary['usage']['cost_usd']:.4f}  "
          f"({usage.report()})")
    print(f"  Wall clock: {summary['wall_clock_s']}s")
    n = summary["n_agents"]
    if n:
        print(f"\n  PROJECTION → 31 agents: "
              f"~${summary['usage']['cost_usd']/n*31:.2f}, "
              f"~{summary['wall_clock_s']/n*31/60:.1f} min wall "
              f"(uncached; cache makes re-runs near-free)")


if __name__ == "__main__":
    asyncio.run(main())
