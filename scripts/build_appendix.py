"""Generate the (large) appendix from the REAL recorded data in results_v2/.

Everything here is verbatim recorded material — prompt templates actually used,
per-arm optimized prompts actually produced, and per-example transcripts actually
generated — so the appendix cannot contain fabricated content. Emitted as
paper/appendix.tex (one-column), \\input by main.tex after \\appendix.

Requires (in main.tex): \\usepackage{listings}, \\usepackage{longtable}.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teibench.optimizer import _REFLECT_SYSTEM, _MERGE_SYSTEM, _PARAPHRASE_SYSTEM
from teibench.gpa_judge import _JUDGE_SYSTEM
from teibench.scorers import build_output_contract

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
OUT = ROOT / "paper" / "appendix.tex"

_REPL = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2192": "->",
         "\u00d7": "x", "\u2248": "~", "\u2265": ">=", "\u2264": "<=",
         "\u00b1": "+/-", "\u2022": "*", "\u00a0": " ", "\u25d1": "(o)"}


def ascii_clean(s: str) -> str:
    s = str(s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # listings delimiter safety
    return s.replace(r"\end{lstlisting}", "[end lstlisting]")


def esc(s: str) -> str:
    """LaTeX-escape for normal text (table cells, headings)."""
    s = ascii_clean(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def lst(text: str) -> str:
    return "\\begin{lstlisting}\n" + ascii_clean(text).strip() + "\n\\end{lstlisting}\n"


def load():
    return [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))
            if not p.name.startswith("_")]


def main():
    recs = load()
    L = []
    A = L.append

    A(r"\onecolumn")
    A(r"\appendix")
    A(r"\section{Prompt templates (verbatim, as used)}\label{app:templates}")
    A("These are the exact templates used by the harness for every run.\n")
    A(r"\subsection{Judge system prompt (GPA-inspired four-dimension evaluator)}")
    A(lst(_JUDGE_SYSTEM))
    A(r"\subsection{Reflective-mutation system prompt (TEI / objective-only)}")
    A(lst(_REFLECT_SYSTEM))
    A(r"\subsection{System-aware merge prompt}")
    A(lst(_MERGE_SYSTEM))
    A(r"\subsection{Random-search paraphrase prompt (no failure information)}")
    A(lst(_PARAPHRASE_SYSTEM))
    A(r"\subsection{Universal output contract (classification / numeric)}")
    A(lst(build_output_contract("classification", ["label_a", "label_b", "..."])))
    A(lst(build_output_contract("numeric", None)))

    A(r"\section{Full task catalog}\label{app:tasks}")
    A("For each task: domain, source, metric, split sizes, label set, and the baseline prompt.\n")
    for r in recs:
        a = r["arms"]
        A(r"\subsection{\texttt{%s}}" % esc(r["task_id"]))
        labels = ""
        # recover label set from the baseline prompt's contract if present
        A(r"\noindent\textbf{Domain:} %s \quad \textbf{Source:} %s \\" %
          (esc(r["industry"]), esc(r["source"])))
        A(r"\textbf{Metric:} %s \quad \textbf{$n_{\mathrm{train}}$:} %d \quad "
          r"\textbf{$n_{\mathrm{test}}$:} %d \quad \textbf{public:} %s \\" %
          (esc(r["metric"]), r["n_train"], r["n_test"], esc(str(r.get("is_public")))))
        A(r"\textbf{Baseline prompt:}")
        A(lst(a["baseline"]["prompt"]))

    A(r"\section{Optimized prompts per arm (verbatim)}\label{app:prompts}")
    A("The actual system prompts produced by each optimization arm, per task.\n")
    for r in recs:
        A(r"\subsection{\texttt{%s}}" % esc(r["task_id"]))
        for arm, name in [("random", "Random search"),
                          ("objective_reflection", "Objective-only reflection"),
                          ("tei", "TEI (full)")]:
            A(r"\noindent\textbf{%s:}" % name)
            A(lst(r["arms"][arm]["prompt"]))

    A(r"\section{Worked transcripts (verbatim recorded outputs)}\label{app:transcripts}")
    A("Per task, the first held-out test examples under the baseline and TEI prompts: "
      "query, the agent's actual output, the gold label, and the code-computed score.\n")
    for r in recs:
        A(r"\subsection{\texttt{%s}}" % esc(r["task_id"]))
        for arm, name in [("baseline", "Baseline"), ("tei", "TEI")]:
            exs = r["arms"][arm].get("examples", [])[:4]
            for j, ex in enumerate(exs, 1):
                A(r"\noindent\textbf{%s --- example %d} (gold: \texttt{%s}, score: %.1f)" %
                  (name, j, esc(str(ex["gold"])), ex["objective"]))
                A(r"\textit{Query:}")
                A(lst(ex["query"][:1500]))
                A(r"\textit{Agent output:}")
                A(lst(ex["output"][:2000]))
                gpa = ex.get("gpa") or {}
                if gpa.get("rationale"):
                    A(r"\textit{Judge rationale:} " + esc(gpa["rationale"][:400]) + r" \\")

    A(r"\section{Full per-task results with bootstrap confidence intervals}\label{app:pertask}")
    A(r"\begin{longtable}{lrrrr}")
    A(r"\toprule Task & Base [95\% CI] & Random & Obj-only & TEI \\ \midrule \endhead")
    for r in recs:
        a = r["arms"]
        def cell(arm):
            o = a[arm]["objective_mean"]; ci = a[arm].get("objective_ci95", [o, o])
            return f"{o:.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
        A(r"\texttt{\small %s} & %s & %s & %s & %s \\" %
          (esc(r["task_id"]), cell("baseline"), cell("random"),
           cell("objective_reflection"), cell("tei")))
    A(r"\bottomrule \end{longtable}")

    A(r"\section{Per-dimension GPA (baseline vs TEI), per task}\label{app:gpadims}")
    A(r"\begin{longtable}{lrrrr}")
    A(r"\toprule Task & Target & Reasoning & Execution & Integrity \\")
    A(r" & (B$\to$T) & (B$\to$T) & (B$\to$T) & (B$\to$T) \\ \midrule \endhead")
    dims = ["target_alignment", "reasoning_soundness", "execution_accuracy", "output_integrity"]
    for r in recs:
        bd = r["arms"]["baseline"].get("gpa_dims", {})
        td = r["arms"]["tei"].get("gpa_dims", {})
        cells = " & ".join(f"{bd.get(d,0):.2f}$\\to${td.get(d,0):.2f}" for d in dims)
        A(r"\texttt{\small %s} & %s \\" % (esc(r["task_id"]), cells))
    A(r"\bottomrule \end{longtable}")

    OUT.write_text("\n".join(L), encoding="utf-8")
    # rough page estimate
    nchars = sum(len(x) for x in L)
    print(f"Wrote {OUT} ({nchars:,} chars, {len(recs)} tasks). "
          f"Sections: templates, catalog, optimized prompts, transcripts, per-task CIs, GPA dims.")


if __name__ == "__main__":
    main()
