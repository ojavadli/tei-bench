#!/usr/bin/env python3
"""Shared academic figure style for TEI-SWE data plots.

The manuscript compiles with tectonic (XeTeX) using Latin Modern; there is no
standalone xelatex/pdflatex on this machine, so matplotlib's PGF backend (which
must shell out to a LaTeX binary to measure text) is unavailable. Instead we
emit true VECTOR PDF figures whose text is set in the manuscript's own Latin
Modern typeface (the lmroman*.otf shipped in the tectonic bundle), embedded as a
subsetted font. Result: figure labels are crisp vector glyphs in the same family
as the body text -- no raster pixels -- which is the owner-approved alternative
when a real xelatex is absent ("verified to contain true vector text and paths").
"""
import glob
import os
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# --- register the manuscript's Latin Modern fonts from the tectonic cache ---
_cands = glob.glob(os.path.expanduser(
    "~/Library/Caches/Tectonic/bundles/data/*/lmroman10-regular.otf"))
_BUNDLE = os.path.dirname(_cands[0]) if _cands else None
_LOADED = []
if _BUNDLE:
    for _fn in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf",
                "lmroman9-regular.otf", "lmroman8-regular.otf", "lmroman12-regular.otf"):
        _p = os.path.join(_BUNDLE, _fn)
        if os.path.exists(_p):
            fm.fontManager.addfont(_p); _LOADED.append(_fn)
LM = "Latin Modern Roman"

# --- one restrained academic palette ---
INK = "#1a1a1a"      # near-black: text, primary rules
ACCENT = "#2f5c8a"   # muted steel blue: TEI / deployed data series only
GRAYD = "#6b7280"    # baseline / control / ceiling
GRAYM = "#9aa1a9"    # mid gray
GRAYL = "#c9ced4"    # light gray: individual population lines
FRAME = "#3a3a3a"    # plot-box frame
GRID = "#d6d6d6"     # subtle gridlines (behind data)

plt.rcParams.update({
    "font.family": "serif", "font.serif": [LM, "CMU Serif", "cmr10"],
    # mathtext uses matplotlib's Computer Modern, the same family as Latin Modern,
    # so $n{=}26$, $\Delta$, $\approx$ render consistently with the LM body text.
    "mathtext.fontset": "cm", "text.usetex": False, "text.parse_math": True,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.titleweight": "normal", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    # framed plot box (all four spines) + subtle gridlines behind the data +
    # ticks pointing in: the polished scientific-tool / MATLAB-grade look.
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": FRAME, "axes.linewidth": 0.7, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.alpha": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "lines.linewidth": 1.2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": FRAME, "ytick.color": FRAME,
    "pdf.fonttype": 3, "pdf.compression": 6, "figure.dpi": 300, "savefig.dpi": 300,
})

OUT = os.path.expanduser("~/swebench-agents/paper/figures")


def save(fig, name):
    """Write a vector PDF (never PNG) at the figure's intended physical size."""
    assert name.endswith(".pdf"), "main figures must be vector PDF"
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name, "(LM fonts:", len(_LOADED), ")")
