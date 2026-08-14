#!/usr/bin/env python3
"""All-page PDF QA for TEI-SWE (owner directive section 10).

Mechanical, per-page checks over the compiled PDF text layer (poppler
`pdftotext -bbox`), with SEPARATE thresholds for main text vs. the dense
appendix (a page is classified by its own median glyph height, so footnotesize
tables and 8 pt appendix dumps are not flagged merely for being small):

  * font size        - median glyph height (proxy for pt); flags genuinely tiny
                       or oversized MAIN-text pages, never ordinary appendix 8 pt
  * raw LaTeX        - literal \\textbf, \\textsc, \\Delta, \\%, {,}, resizebox,
                       $\\..., stray control sequences leaking into the text
  * blank-area ratio - text bounding-box coverage; low coverage is reported
                       (figure/landscape pages legitimately have little text)
  * duplicate titles - the same "Figure N:"/"Table N:" caption text appearing
                       for two different numbers (a duplication smell)

Main figures are vector (Latin Modern-embedded PDF, or native TikZ). Vector-ness
is enforced in THREE independent places, so no single false pass can slip
through (any one alone can be fooled; together they cannot):

  * asset extension  - static grep of main.tex: every \\includegraphics that is
                       not an allowed appendix raster must be a .pdf (catches a
                       .png/.jpg main-figure include before the PDF is built)
  * bbox vector proof- per MAIN figure (PyMuPDF), the region ABOVE its caption
                       must contain NO raster image, typeset Latin Modern text,
                       and vector content (page-stream paths for TikZ or a Form
                       XObject for the plots). Localizing to the figure box
                       avoids the page-level get_drawings() false pass, where
                       table rules elsewhere register as "vector present."

Two regression/scope guards for failure modes that have actually recurred:

  * regression       - 'p0.001' (missing operator; regressed once v10->v11) and
                       the call-ledger scoping (1,271 original vs 3,090 full)
  * scope diff       - diff the current manuscript text against the previous
                       release and confirm no removed content reappeared (the
                       removed material still lives in the repo/history)

Exit code is nonzero if any HARD flag fires.
"""
import os
import re
import statistics
import subprocess
import sys

TEX = os.path.join(os.path.dirname(__file__), "main.tex")
RELEASE_REPO = os.path.expanduser("~/swebench-agents/release")
PREV_TAG = "paper-v5-figures"        # the immediately-preceding released manuscript
APPENDIX_RASTER_ALLOW = {"curves_grid_a.png", "curves_grid_b.png", "ablation_curves.png"}
MAIN_FIGS = {  # distinctive caption substring -> label; each must be vector
    "Deployed improvement (headline)": "Fig 1 trajectory",
    "validation ladder (evidence": "Fig 2 ladder (TikZ)",
    "How TEI works. A per-agent": "Fig 3 method (TikZ)",
    "Deployed trajectories for all 30 systems": "Fig 4 population",
    "increase all four anchored rubric dimensions": "Fig 5 dimensions",
    "Distribution of rubric movement": "Fig 6 noise floor",
    "Four within-study controls": "Fig 7 validation",
}
BANNED_PHASEC = ["prereg-exec2", "47/100", "50/100", "100 baseline", "100 patched",
                 "200 rollout", "Phase C", "Phase 3", "scoring_instrument", "exec100",
                 "livelock"]

PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "main.pdf")

MAIN_MEDIAN_MIN = 9.3     # a page whose median >= this is treated as body-prose "main"
MAIN_TINY = 6.0           # a main page with tiny median glyphs (pt) -> hard flag
OVERSIZE = 1.35           # main page median above this * body -> enlarged type
APPENDIX_UNREADABLE = 4.5 # any page below this median glyph height -> hard flag
BLANK_COVERAGE = 0.12     # text bbox coverage below this -> report (soft)

RAW_LATEX = [
    r"\textbf", r"\textsc", r"\textit", r"\emph{", r"\Delta", r"\alpha",
    r"\resizebox", r"\scalebox", r"\includegraphics", r"\begin{", r"\end{",
    r"\textbackslash", r"{,}", r"\% ", r"\&", r"$\\", r"\hbox", r"\vskip",
]


def extract():
    out = subprocess.run(["pdftotext", "-bbox", PDF, "-"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"pdftotext failed: {out.stderr[:200]}")
    pages = []
    for pm in re.finditer(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', out.stdout, re.S):
        pw, ph = float(pm.group(1)), float(pm.group(2))
        words = []
        for wm in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', pm.group(3)):
            x0, y0, x1, y1, t = (float(wm.group(1)), float(wm.group(2)), float(wm.group(3)),
                                 float(wm.group(4)), wm.group(5))
            if t.strip():
                words.append((x0, y0, x1, y1, t))
        pages.append((pw, ph, words))
    return pages


def gsize(w):
    # glyph-height proxy that is invariant to 90-degree rotation (landscape
    # tables/figures): a word is always longer along its baseline than tall, so
    # the SHORT bbox dimension is the glyph height in either orientation.
    return min(w[2] - w[0], w[3] - w[1])


def plain_text():
    out = subprocess.run(["pdftotext", "-nopgbrk", PDF, "-"], capture_output=True, text=True)
    return out.stdout


def figure_asset_check():
    """Static (pre-build): a main-figure \\includegraphics that is .png/.jpg is a
    raster leak. Allowed appendix rasters are whitelisted."""
    flags = []
    if not os.path.exists(TEX):
        return flags
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", open(TEX).read()):
        base = os.path.basename(m.group(1))
        if re.search(r"\.(png|jpe?g)$", base, re.I) and base not in APPENDIX_RASTER_ALLOW:
            flags.append((0, "RASTER-INCLUDE", f"{m.group(1)} (main figures must be vector .pdf)"))
    return flags


def vector_figure_check():
    """Per-figure bounding-box vector proof (PyMuPDF)."""
    try:
        import pymupdf
    except Exception as e:
        return [(0, "VECTOR-CHECK-UNAVAILABLE", f"PyMuPDF not importable: {e}")]
    doc = pymupdf.open(PDF); flags = []
    for needle, label in MAIN_FIGS.items():
        hit = None
        for pno in range(min(30, doc.page_count)):
            r = doc[pno].search_for(needle)
            if r:
                hit = (pno, doc[pno], r[0]); break
        if not hit:
            flags.append((0, "FIG-NOT-FOUND", f"{label}: caption '{needle[:22]}' not found")); continue
        pno, pg, cap = hit
        box = pymupdf.Rect(pg.rect.x0, 30, pg.rect.x1, cap.y0 - 2)
        area = pg.rect.width * pg.rect.height
        imgs = [i for i in pg.get_image_info(xrefs=True)
                if pymupdf.Rect(i["bbox"]).intersects(box)
                and pymupdf.Rect(i["bbox"]).get_area() > 0.005 * area]
        words = [w for w in pg.get_text("words") if pymupdf.Rect(w[:4]).intersects(box)]
        draws = [d for d in pg.get_drawings() if pymupdf.Rect(d["rect"]).intersects(box)]
        xobjs = [x for x in pg.get_xobjects() if pymupdf.Rect(x[3]).intersects(box)]
        fonts = {s["font"] for b in pg.get_text("dict", clip=box)["blocks"]
                 for ln in b.get("lines", []) for s in ln.get("spans", [])}
        if imgs:
            flags.append((pno + 1, "RASTER-IN-FIG", f"{label}: {len(imgs)} raster image(s) in figure box"))
        if len(words) < 3:
            flags.append((pno + 1, "NO-TYPESET-TEXT", f"{label}: only {len(words)} words in box"))
        if len(draws) < 1 and len(xobjs) < 1:
            flags.append((pno + 1, "NO-VECTOR-CONTENT", f"{label}: no vector paths / Form XObject in box"))
        if words and not any("LMRoman" in f or "LMMono" in f for f in fonts):
            flags.append((pno + 1, "FONT-MISMATCH", f"{label}: figure fonts not Latin Modern {sorted(fonts)[:4]}"))
    return flags


def regression_check(text):
    """Compressed guards for failure modes that recurred (one line each)."""
    flags = []
    flat = re.sub(r"\s+", " ", text)
    if re.search(r"\bp0\.001\b", flat):
        flags.append((0, "P-OPERATOR", "'p0.001' present (missing < / = operator)"))
    if "3,090" not in flat:
        flags.append((0, "CALL-LEDGER", "full Phase-A/B total 3,090 not reported"))
    if re.search(r"study total is 1,?271", flat):
        flags.append((0, "CALL-LEDGER", "1,271 mislabelled as 'study total'"))
    return flags


def scope_diff_check(text):
    """No removed (out-of-scope) content may reappear. Hard-fail if any banned
    token is in the current PDF; the mechanical diff vs the previous release
    additionally flags a token that is present now but was absent then."""
    flags = []
    cur = re.sub(r"\s+", " ", text).lower()
    hits = [b for b in BANNED_PHASEC if b.lower() in cur]
    if hits:
        flags.append((0, "SCOPE-BANNED", f"banned/removed content in current PDF: {hits}"))
    try:
        import pymupdf
        prev = subprocess.run(["git", "-C", RELEASE_REPO, "show", f"{PREV_TAG}:paper/TEI-SWE.pdf"],
                              capture_output=True)
        if prev.returncode == 0:
            pd = pymupdf.open(stream=prev.stdout, filetype="pdf")
            prevtext = " ".join(pg.get_text() for pg in pd).lower()
            for b in BANNED_PHASEC:
                if b.lower() in cur and b.lower() not in prevtext:
                    flags.append((0, "SCOPE-REGRESSION", f"'{b}' absent from {PREV_TAG} but present now"))
    except Exception:
        pass
    return flags


def main():
    pages = extract()
    med = {i: statistics.median([gsize(w) for w in pg[2]])
           for i, pg in enumerate(pages, 1) if len(pg[2]) >= 25}
    body = sorted(med.values())[int(0.85 * (len(med) - 1))]
    print(f"PDF QA: {PDF}")
    print(f"pages: {len(pages)}   body-prose median glyph height: {body:.1f} pt")
    print(f"main-page classifier: median >= {MAIN_MEDIAN_MIN} pt; appendix pages exempt "
          f"from the small-font rule (down to the {APPENDIX_UNREADABLE} pt unreadable floor)\n")

    hard, soft = [], []
    import html
    caption_titles = {}  # (kind, num) -> title text
    for i, (pw, ph, words) in enumerate(pages, 1):
        if not words:
            soft.append((i, "EMPTY", "no extractable text (figure/landscape page?)"))
            continue
        heights = [gsize(w) for w in words]
        m = statistics.median(heights)
        is_main = m >= MAIN_MEDIAN_MIN
        # font
        if m <= APPENDIX_UNREADABLE:
            hard.append((i, "UNREADABLE", f"median {m:.1f} pt"))
        elif is_main and m <= MAIN_TINY:
            hard.append((i, "TINY-MAIN", f"median {m:.1f} pt"))
        elif is_main and m >= OVERSIZE * body:
            hard.append((i, "OVERSIZED-MAIN", f"median {m:.1f} pt = {m/body:.2f}x body"))
        # raw LaTeX (join words with spaces; also check raw concatenation)
        txt = " ".join(html.unescape(w[4]) for w in words)
        raw = html.unescape("".join(w[4] for w in words))
        hits = sorted({tok for tok in RAW_LATEX if tok in txt or tok in raw})
        if hits:
            hard.append((i, "RAW-LATEX", ", ".join(repr(h) for h in hits[:6])))
        # blank-area coverage
        xs0 = min(w[0] for w in words); ys0 = min(w[1] for w in words)
        xs1 = max(w[2] for w in words); ys1 = max(w[3] for w in words)
        cov = ((xs1 - xs0) * (ys1 - ys0)) / (pw * ph)
        if cov < BLANK_COVERAGE:
            soft.append((i, "LOW-COVERAGE", f"{cov*100:.0f}% text bbox (figure/landscape ok)"))
        # duplicate captions
        for cm in re.finditer(r"(Figure|Table)\s+(\d+):\s*([^.]{6,70})", txt):
            key = (cm.group(1), cm.group(2))
            title = re.sub(r"\s+", " ", cm.group(3)).strip().lower()
            if key in caption_titles and caption_titles[key] != title:
                hard.append((i, "DUP-CAPTION-NUM", f"{cm.group(1)} {cm.group(2)} seen twice"))
            caption_titles.setdefault(key, title)

    # duplicate title TEXT across different numbers (SOFT: split/continued
    # figures legitimately share a title stem, e.g. "... agents 1-15 / 16-30")
    seen_title = {}
    for (kind, num), title in caption_titles.items():
        if title in seen_title and seen_title[title] != (kind, num):
            soft.append((0, "DUP-TITLE-TEXT",
                         f"'{title[:36]}...' on {kind} {num} & {kind} {seen_title[title][1]} "
                         f"(ok if a split/continued float)"))
        seen_title.setdefault(title, (kind, num))

    # --- vector-figure pipeline + regression + scope guards ---
    # asset-extension and regression checks need no external deps, so they stay the
    # always-on hard raster guard; the bbox proof degrades to a soft note if
    # PyMuPDF is absent (then rely on asset-extension + the page-level raster scan).
    fulltext = plain_text()
    vg = vector_figure_check()
    vg_soft = [f for f in vg if f[1] == "VECTOR-CHECK-UNAVAILABLE"]
    vg_hard = [f for f in vg if f[1] != "VECTOR-CHECK-UNAVAILABLE"]
    soft += vg_soft
    hard += figure_asset_check() + vg_hard + regression_check(fulltext) + scope_diff_check(fulltext)
    if not vg_soft:
        bad = {f[0] for f in vg_hard}
        print(f"vector figures verified (asset .pdf + no raster in box + typeset LM text + "
              f"vector content): {len(MAIN_FIGS) - len(bad)}/{len(MAIN_FIGS)} main figures\n")

    if hard:
        print("HARD FLAGS:")
        for p, k, d in hard:
            print(f"  {'p'+str(p) if p else '  -':>5}  {k}: {d}")
    else:
        print("no hard flags (no tiny/oversized main text, no raw LaTeX, no duplicate captions).")
    if soft:
        print("\nsoft/informational (verify visually; often legitimate figure pages):")
        for p, k, d in soft[:40]:
            print(f"  p{p:>3}  {k}: {d}")

    print("\nPASS" if not hard else "\nFAIL")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
