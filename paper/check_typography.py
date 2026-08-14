#!/usr/bin/env python3
"""Typography auditor for the compiled PDF (Task 8).

Extracts per-word bounding boxes from the PDF *text layer* via poppler's
`pdftotext -bbox`, and reports, per page, the distribution of glyph heights
(a proxy for font size in points). It flags the specific failure this paper
had --- narrow tables *enlarged* to text width by \\resizebox, i.e. table type
rendered LARGER than body prose --- plus microscopic body text.

Caveat printed at runtime: figures are embedded as rasterized PNGs, so their
internal label/tick text is NOT in the PDF text layer and cannot be measured
here; figure fonts are set explicitly in make_figures.py (8--9.5 pt) and are
verified by rendered-page visual inspection, which the directive names as the
final authority. Section headings are excluded from the oversize test by
requiring a DENSE cluster of large words (headings are short lines).
"""
import os
import re
import statistics
import subprocess
import sys

PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "main.pdf")

# Robust logic: within one font size, individual word heights vary a lot
# (ascenders/descenders/caps), so a single global mode is meaningless. Instead
# compare each page's MEDIAN word height to the document body median (= the
# median of per-page medians of text-heavy pages). An enlarged table makes a
# page's median jump; a normal page's median tracks body size regardless of a
# few tall words or a section heading.
OVERSIZE_RATIO = 1.28      # page median above this * body => enlarged type on the page
MICRO_RATIO = 0.62         # page median below this * body => microscopic type
MIN_WORDS = 25             # ignore near-empty (figure) pages when setting the body ref


def extract():
    out = subprocess.run(["pdftotext", "-bbox", PDF, "-"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"pdftotext failed: {out.stderr[:200]}")
    pages = []
    for pm in re.finditer(r"<page\b[^>]*>(.*?)</page>", out.stdout, re.S):
        heights = []
        for wm in re.finditer(r'<word[^>]*yMin="([\d.]+)"[^>]*yMax="([\d.]+)">([^<]*)</word>', pm.group(1)):
            if wm.group(3).strip():
                heights.append(round(float(wm.group(2)) - float(wm.group(1)), 1))
        pages.append(heights)
    return pages


def main():
    pages = extract()
    if not pages:
        sys.exit("no text extracted (is this a text PDF?)")
    medians = {i: statistics.median(pg) for i, pg in enumerate(pages, 1) if len(pg) >= MIN_WORDS}
    # Body PROSE is the largest normal text size; captions/tables/appendix are
    # smaller and dominate page count (160+ dense appendix pages here), so the
    # median-of-medians under-reads. Use a high percentile to lock onto prose.
    _sm = sorted(medians.values())
    body = _sm[int(0.85 * (len(_sm) - 1))]
    print(f"typography audit: {PDF}")
    print(f"pages with text: {sum(1 for pg in pages if pg)}   "
          f"body median glyph height: {body:.1f} pt (proxy for body font size)")
    print(f"test: per-page MEDIAN height outside [{MICRO_RATIO:.2f}x, {OVERSIZE_RATIO:.2f}x] body "
          f"= [{MICRO_RATIO*body:.1f}, {OVERSIZE_RATIO*body:.1f}] pt  (catches resizebox-enlarged "
          f"tables and microscopic blocks)")
    print("note: figure-internal text is rasterized (PNG), not in the text layer, so it is not "
          "measurable here; figure fonts are set explicitly in make_figures.py (8-9.5 pt) and "
          "verified by rendered-page visual inspection (the directive's final authority).\n")

    flags = []
    for i, pg in enumerate(pages, 1):
        if len(pg) < MIN_WORDS:
            continue
        med = statistics.median(pg)
        if med >= OVERSIZE_RATIO * body:
            flags.append((i, "OVERSIZED-TYPE", f"page median {med:.1f}pt = {med/body:.2f}x body"))
        elif med <= MICRO_RATIO * body:
            flags.append((i, "MICROSCOPIC-TYPE", f"page median {med:.1f}pt = {med/body:.2f}x body"))

    if flags:
        print("FLAGGED PAGES (median-based; inspect visually):")
        for pageno, kind, detail in flags:
            print(f"  p{pageno:>3}  {kind}: {detail}")
    else:
        print("no page has an anomalous median glyph height "
              "(no resizebox-enlarged tables, no microscopic text blocks).")

    print("\nper-page median glyph height (first 20 pages; '*' = figure/sparse page):")
    for i, pg in enumerate(pages[:20], 1):
        if len(pg) < MIN_WORDS:
            print(f"  p{i:>3}  * ({len(pg)} words)")
            continue
        print(f"  p{i:>3}  median {statistics.median(pg):.1f}pt   tallest {max(pg):.1f}pt   "
              f"words {len(pg)}")

    print("\nPASS" if not flags else "\nFAIL")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
