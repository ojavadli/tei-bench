"""TEI-Bench: a controlled, held-out evaluation of the TEI loop across agents."""
from teibench.llm import LLM, Usage
from teibench.tasks import Task, load_task, load_all_tasks, save_task
from teibench.tei import run_tei_on_agent
from teibench.stats import analyze_paired, holm_bonferroni

__version__ = "0.1.0"
__all__ = [
    "LLM", "Usage", "Task", "load_task", "load_all_tasks", "save_task",
    "run_tei_on_agent", "analyze_paired", "holm_bonferroni",
]
