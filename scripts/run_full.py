"""Full 31-agent TEI run. Records per-agent results + paired statistics.

Outputs:
  results/<task_id>.json     full per-agent record (prompts, traces, scores)
  results/_full_summary.json deck-level stats (objective, gpa, per-dimension)
  results/per_agent.csv      one row per agent for the paper's tables
"""
import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench import LLM, Usage, load_all_tasks, run_tei_on_agent
from teibench.stats import analyze_paired, holm_bonferroni
from teibench.gpa_judge import DIMENSIONS

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
AGENT_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-sonnet-4-5"
OPT_MODEL = "claude-sonnet-4-5"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    usage = Usage()
    llm = LLM(cache_dir=str(ROOT / ".cache"), max_concurrency=args.concurrency, usage=usage)
    tasks = load_all_tasks(TASKS_DIR)
    if args.only:
        tasks = [t for t in tasks if t.task_id in args.only]

    print(f"FULL RUN: {len(tasks)} agents | agent={AGENT_MODEL} judge={JUDGE_MODEL} "
          f"opt={OPT_MODEL} iters={args.iterations} repeats={args.repeats}", flush=True)
    t0 = time.time()
    records = []
    for i, t in enumerate(tasks, 1):
        log: list[str] = []
        rec = await run_tei_on_agent(
            llm, t, agent_model=AGENT_MODEL, judge_model=JUDGE_MODEL,
            optimizer_model=OPT_MODEL, num_iterations=args.iterations,
            minibatch=args.minibatch, eval_repeats=args.repeats, seed=0, log=log,
        )
        records.append(rec)
        (RESULTS_DIR / f"{rec['task_id']}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{i:>2}/{len(tasks)}] {rec['task_id']:<30} "
              f"obj {rec['baseline_test']['objective_mean']:.3f}→{rec['final_test']['objective_mean']:.3f} "
              f"({rec['delta_objective']:+.3f})  "
              f"gpa {rec['baseline_test']['gpa_mean']:.3f}→{rec['final_test']['gpa_mean']:.3f} "
              f"({rec['delta_gpa']:+.3f})  {rec['duration_s']}s  "
              f"[{usage.report()}]", flush=True)

    # ---- aggregate stats ----
    bo = [r["baseline_test"]["objective_mean"] for r in records]
    ao = [r["final_test"]["objective_mean"] for r in records]
    bg = [r["baseline_test"]["gpa_mean"] for r in records]
    ag = [r["final_test"]["gpa_mean"] for r in records]

    obj_stats = analyze_paired(bo, ao)
    gpa_stats = analyze_paired(bg, ag)

    dim_stats = {}
    for d in DIMENSIONS:
        bd = [r["baseline_test"]["gpa_dims"].get(d, 0.0) for r in records]
        ad = [r["final_test"]["gpa_dims"].get(d, 0.0) for r in records]
        dim_stats[d] = analyze_paired(bd, ad)

    # headroom subset (baseline objective < 0.9), pre-registered
    head_idx = [i for i, v in enumerate(bo) if v < 0.9]
    headroom = None
    if len(head_idx) >= 2:
        headroom = analyze_paired([bo[i] for i in head_idx], [ao[i] for i in head_idx])

    pcorr = holm_bonferroni({
        "objective": obj_stats["t_p_value"], "gpa": gpa_stats["t_p_value"],
        **{f"dim_{d}": dim_stats[d]["t_p_value"] for d in DIMENSIONS},
    })

    summary = {
        "n_agents": len(records),
        "models": {"agent": AGENT_MODEL, "judge": JUDGE_MODEL, "optimizer": OPT_MODEL},
        "config": vars(args),
        "objective": obj_stats,
        "gpa": gpa_stats,
        "gpa_dimensions": dim_stats,
        "headroom_subset": {"n": len(head_idx), "stats": headroom},
        "holm_corrected_p": pcorr,
        "usage": {"calls": usage.calls, "cache_hits": usage.cache_hits,
                  "tokens_in": usage.tokens()[0], "tokens_out": usage.tokens()[1],
                  "cost_usd": round(usage.cost_usd(), 4)},
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "_full_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (RESULTS_DIR / "per_agent.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "industry", "metric", "n_classes", "n_test",
                    "obj_before", "obj_after", "obj_delta",
                    "gpa_before", "gpa_after", "gpa_delta", "duration_s"])
        for r in records:
            t = next(t for t in tasks if t.task_id == r["task_id"])
            w.writerow([r["task_id"], r["industry"], r["metric"],
                        len(t.labels) if t.labels else "", r["n_test"],
                        f"{r['baseline_test']['objective_mean']:.4f}",
                        f"{r['final_test']['objective_mean']:.4f}",
                        f"{r['delta_objective']:.4f}",
                        f"{r['baseline_test']['gpa_mean']:.4f}",
                        f"{r['final_test']['gpa_mean']:.4f}",
                        f"{r['delta_gpa']:.4f}", r["duration_s"]])

    print("\n" + "=" * 70)
    print("FULL RUN COMPLETE")
    print("=" * 70)
    o, g = obj_stats, gpa_stats
    print(f"OBJECTIVE (primary): {o['mean_before']:.3f} → {o['mean_after']:.3f}  "
          f"Δ={o['mean_delta']:+.3f}  95%CI[{o['ci95_low']:+.3f},{o['ci95_high']:+.3f}]  "
          f"t={o['t_stat']:.2f} p={o['t_p_value']:.2e} d_z={o['cohen_dz']:.2f}  "
          f"W={o['wins']}/L={o['losses']}/T={o['ties']}")
    print(f"GPA (secondary):     {g['mean_before']:.3f} → {g['mean_after']:.3f}  "
          f"Δ={g['mean_delta']:+.3f}  t={g['t_stat']:.2f} p={g['t_p_value']:.2e} d_z={g['cohen_dz']:.2f}")
    if headroom:
        h = headroom
        print(f"HEADROOM subset n={summary['headroom_subset']['n']}: "
              f"Δ={h['mean_delta']:+.3f} p={h['t_p_value']:.2e} d_z={h['cohen_dz']:.2f}")
    print(f"\nCost: ${summary['usage']['cost_usd']:.2f} | Wall: {summary['wall_clock_s']/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
