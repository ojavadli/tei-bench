# REVISION_LOG_15 — professional figure restyle, portrait Table 5, "six thousand" in prose

Owner-requested polish pass. Build: 0 errors, 0 undefined refs, 0 overfull boxes
> 5 pt, 176 pp. pdf_qa 7/7 vector figures + PASS; consistency 7 surfaces; 0
Phase-C.

## Figures — scientific-tool (MATLAB-grade) styling
`figstyle.py`: every data figure (Figs 1, 4, 5, 6, 7) now has a full thin plot
box (all four spines), subtle gridlines behind the data (horizontal on the
value axis; vertical for the horizontal dumbbell / dot plots), and ticks
pointing in. Latin Modern text and the single muted accent are retained. This
replaces the previous open L-frame / no-grid look, which read as a default plot
rather than a finished scientific figure. Still true vector PDF (0 raster on the
main-figure pages; LM subsets embedded).

## Table 5 — portrait, readable
Converted from a full-page landscape `sidewaystable` to a portrait `table[p]` at
footnotesize. The System column was widened-but-bounded to `p{2.5cm}` (was an
unbounded `l` that dominated the width and, when squeezed, collided with Split);
system names truncate at 18 chars. All 14 columns are legible in portrait
without microtype; `adjustbox max width` is a shrink-only safety.

## "Six thousand" in prose
Every running-text / caption mention of 6{,}000 candidate versions is now spelled
"six thousand" (main text and the appendix technique caption). Numerals are kept
where they belong: table cells (e.g., the study-summary "Total candidate
versions" row), the method-figure data annotation, and formula-style breakdowns
(3{,}000 structural + 3{,}000 prompt).

No new experiments; raw data and git history untouched.
