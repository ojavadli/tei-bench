#!/usr/bin/env bash
# Deterministic paper build. Order matters:
#   make_assets.py  -> numbers.tex, tables/*.tex, figures/*.png, figures/_fig_data.json
#   make_narrative.py -> a1_result.tex  (independent)
#   make_figures.py -> figures/fig_*.png (needs figures/_fig_data.json + _paper_recompute.json)
#   tectonic       -> main.pdf
set -euo pipefail
cd "$(dirname "$0")"
PY="${TEI_PY:-$HOME/swebench-agents/_venv_aider/bin/python}"
echo "== make_assets =="   ; "$PY" make_assets.py
echo "== make_narrative ==" ; "$PY" make_narrative.py
echo "== make_figures =="  ; "$PY" make_figures.py
echo "== tectonic =="      ; tectonic main.tex --print 2>&1 | tail -5
cp -f main.pdf TEI-SWE.pdf
echo "== done: $(pwd)/main.pdf =="
