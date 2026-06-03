"""Export TEI-Bench as a HuggingFace-ready dataset + results bundle.

Produces huggingface/:
  tasks.jsonl       one row per (task, split, example) with gold
  results.jsonl     one row per agent: prompts + scores + deltas
  README.md         dataset card (written separately)
Push with: huggingface-cli upload <repo> huggingface/ . (needs HF token)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench import load_all_tasks

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "huggingface"
HF.mkdir(exist_ok=True)


def export_tasks():
    rows = []
    for t in load_all_tasks(ROOT / "tasks"):
        for split in ("train", "test"):
            for ex in getattr(t, split):
                rows.append({
                    "task_id": t.task_id, "industry": t.industry,
                    "metric": t.metric, "split": split,
                    "labels": t.labels, "query": ex["query"], "gold": ex["gold"],
                    "source": t.source,
                })
    with (HF / "tasks.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  tasks.jsonl: {len(rows)} rows")


def export_results():
    res_dir = ROOT / "results"
    rows = []
    for p in sorted(res_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        rows.append({
            "task_id": r["task_id"], "industry": r["industry"], "metric": r["metric"],
            "agent_model": r["agent_model"], "judge_model": r["judge_model"],
            "baseline_prompt": r["baseline_prompt"], "optimized_prompt": r["optimized_prompt"],
            "obj_before": r["baseline_test"]["objective_mean"],
            "obj_after": r["final_test"]["objective_mean"],
            "delta_objective": r["delta_objective"],
            "gpa_before": r["baseline_test"]["gpa_mean"],
            "gpa_after": r["final_test"]["gpa_mean"],
            "delta_gpa": r["delta_gpa"],
        })
    with (HF / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  results.jsonl: {len(rows)} agents")


if __name__ == "__main__":
    print("Exporting HuggingFace bundle...")
    export_tasks()
    try:
        export_results()
    except Exception as e:
        print("  (results not ready yet:", e, ")")
    print("Wrote", HF)
