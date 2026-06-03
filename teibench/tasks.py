"""Task registry + loader.

Each task is one 'agent' in the study: an industry-flavoured problem with a
naive baseline system prompt and a labelled train/test split. Tasks are
stored as JSON files under tasks/ so the exact data is version-controlled
and the experiment is fully reproducible.

JSON schema (one file per task):
{
  "task_id": "fin_banking_intent",
  "industry": "Finance / Banking",
  "instruction": "Classify the customer's banking request into one intent.",
  "metric": "classification",          # classification|numeric|token_f1|contains_all
  "labels": ["card_arrival", "..."],  # required iff metric == classification
  "baseline_prompt": "You are a banking assistant. Classify the request.",
  "source": "banking77 (CC-BY-4.0), https://...",
  "train": [{"query": "...", "gold": "card_arrival"}, ...],
  "test":  [{"query": "...", "gold": "card_arrival"}, ...]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Task:
    task_id: str
    industry: str
    instruction: str
    metric: str
    baseline_prompt: str
    train: list
    test: list
    labels: Optional[list] = None
    source: str = ""

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def load_task(path: str | Path) -> Task:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Task(
        task_id=d["task_id"],
        industry=d["industry"],
        instruction=d["instruction"],
        metric=d["metric"],
        baseline_prompt=d["baseline_prompt"],
        train=d["train"],
        test=d["test"],
        labels=d.get("labels"),
        source=d.get("source", ""),
    )


def load_all_tasks(tasks_dir: str | Path) -> list[Task]:
    tasks = [load_task(p) for p in sorted(Path(tasks_dir).glob("*.json"))]
    return tasks


def save_task(task: Task, tasks_dir: str | Path) -> Path:
    Path(tasks_dir).mkdir(parents=True, exist_ok=True)
    p = Path(tasks_dir) / f"{task.task_id}.json"
    p.write_text(json.dumps({
        "task_id": task.task_id, "industry": task.industry,
        "instruction": task.instruction, "metric": task.metric,
        "labels": task.labels, "baseline_prompt": task.baseline_prompt,
        "source": task.source, "train": task.train, "test": task.test,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


__all__ = ["Task", "load_task", "load_all_tasks", "save_task"]
