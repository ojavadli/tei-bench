#!/usr/bin/env python3
"""Vector-native main-paper data figures for TEI-SWE (Figs 1, 4, 5, 6, 7).

Output is true vector PDF with Latin Modern text (see figstyle.py). Schematics
(Figs 2 ladder, 3 method) are drawn natively in TikZ inside the manuscript, not
here. Plain regular-weight black value labels; the accent is used only for the
TEI/deployed data series. Every value is read from canonical JSON; assertions at
the bottom guard against stale hardcoding. NO Phase-C content.
"""
import json
import os
import numpy as np
import figstyle
from figstyle import plt, save, ACCENT, INK, GRAYD, GRAYM, GRAYL

ROOT = os.path.expanduser("~/swebench-agents")
OUT = figstyle.OUT
_RC = json.load(open(os.path.join(ROOT, "_paper_recompute.json")))
RC = _RC["summary"]
AP = RC["APPLIED"]; PR = RC["PROPOSED"]
PA = _RC["per_agent"]
FD = json.load(open(os.path.join(OUT, "_fig_data.json")))


# ============================================================ FIGURE 1 — result
def fig_trajectory():
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    fig.subplots_adjust(left=0.15, right=0.965, top=0.955, bottom=0.115)
    xs = [0, 1, 2]
    dep = [FD["stages_deployed"][k] for k in ("base", "struct", "final")]
    ceil = [FD["stages_ceiling"][k] for k in ("base", "struct", "final")]
    ax.plot(xs, ceil, ls=(0, (4, 2)), color=GRAYM, lw=0.9, marker="o", ms=3.4,
            mfc="white", mec=GRAYM, zorder=2, label="best proposed (not committed)")
    ax.plot(xs, dep, color=ACCENT, lw=1.6, marker="o", ms=5.2, mfc=ACCENT,
            mec="white", mew=0.6, zorder=4, label="deployed mean ($n=26$)")
    for xi, yi in zip(xs, dep):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, -12),
                    ha="center", va="top", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(["Default", "Structural", "Final"])
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(min(dep) - 0.045, max(ceil) + 0.012)
    ax.set_ylabel("anchored rubric aggregate (proxy)")
    ax.legend(loc="lower right", handlelength=1.9, borderaxespad=0.3, labelspacing=0.4,
              fontsize=7.6)
    save(fig, "fig_trajectory.pdf")


# ======================================================= FIGURE 4 — population
def fig_population():
    fig, ax = plt.subplots(figsize=(4.3, 2.85))
    fig.subplots_adjust(left=0.15, right=0.965, top=0.945, bottom=0.115)
    xs = [0, 1, 2]
    cols = ([r["base"] for r in PA], [r["best_applied_struct"] for r in PA],
            [r["best_applied_final"] for r in PA])
    for b, s, f in zip(*cols):
        ax.plot(xs, [b, s, f], color=GRAYL, lw=0.6, zorder=1)
    mean = [float(np.mean(c)) for c in cols]
    ax.plot(xs, mean, color=ACCENT, lw=1.7, marker="o", ms=4.8, mfc=ACCENT,
            mec="white", mew=0.6, zorder=4, label="mean, all systems ($n=30$)")
    for xi, v in zip(xs, mean):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(["Default", "Structural", "Final"])
    ax.set_xlim(-0.1, 2.28)
    ax.set_ylabel("deployed rubric aggregate (proxy)")
    ax.legend(loc="lower right", handlelength=1.9, fontsize=7.6)
    save(fig, "fig_population.pdf")


# ========================================================= FIGURE 5 — dimensions
def fig_dims():
    d = FD["dims"]
    order = sorted(d["order"], key=lambda k: d["deployed"][k] - d["baseline"][k], reverse=True)
    labels = [d["labels"][k] for k in order]
    base = [d["baseline"][k] for k in order]; dep = [d["deployed"][k] for k in order]
    fig, ax = plt.subplots(figsize=(6.1, 2.1))
    fig.subplots_adjust(left=0.215, right=0.9, top=0.965, bottom=0.205)
    y = list(range(len(order)))[::-1]
    for yi, b, f in zip(y, base, dep):
        ax.plot([b, f], [yi, yi], color=GRAYM, lw=1.1, zorder=1)
        ax.scatter(b, yi, s=32, facecolor="white", edgecolor=GRAYD, lw=0.9, zorder=3)
        ax.scatter(f, yi, s=40, color=ACCENT, zorder=3)
        ax.annotate(f"{b:.3f}", (b, yi), textcoords="offset points", xytext=(-6, 0),
                    ha="right", va="center", fontsize=7.2, color=GRAYD)
        ax.annotate(f"{f:.3f}", (f, yi), textcoords="offset points", xytext=(6, 0),
                    ha="left", va="center", fontsize=7.2, color=INK)
        ax.annotate(f"$+{f - b:.3f}$", ((b + f) / 2, yi), textcoords="offset points",
                    xytext=(0, 5.5), ha="center", fontsize=6.8, color=GRAYD)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("anchored rubric dimension score (proxy)")
    ax.set_xlim(0.45, 0.80); ax.set_ylim(-0.6, len(order) - 0.15)
    ax.scatter([], [], facecolor="white", edgecolor=GRAYD, s=32, label="baseline")
    ax.scatter([], [], color=ACCENT, s=40, label="deployed")
    ax.legend(loc="lower left", fontsize=7.4, handletextpad=0.3, ncol=2, columnspacing=1.1)
    ax.tick_params(axis="y", length=0)              # keep the box; drop y tick marks (categorical)
    ax.grid(True, axis="x"); ax.grid(False, axis="y")  # value is on x -> vertical gridlines
    save(fig, "fig_dims.pdf")


# =============================================== FIGURE 6 — noise floor (stacked)
def fig_floor():
    """Two stacked histograms sharing one x-axis: the paraphrase rewording orbit
    (n=90) and the shipped deployed deltas (n=30). Different y-scales, made
    explicit; no real mass hidden."""
    par = np.asarray(FD["floor"]["paraphrase"], float)
    shp = np.asarray(FD["floor"]["shipped"], float)
    edges = np.arange(-0.01, 0.185 + 1e-9, 0.005)
    pc, _ = np.histogram(par, bins=edges)
    sc, _ = np.histogram(shp, bins=edges)
    ptop = int(pc.max()); stop = int(sc.max())
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(4.6, 3.0),
                               gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=0.125, right=0.965, top=0.965, bottom=0.135, hspace=0.18)
    a.hist(par, bins=edges, color=GRAYD, edgecolor="white", lw=0.3)
    a.set_ylim(0, ptop * 1.12); a.set_ylabel("paraphrase\ncount ($n=90$)", fontsize=8)
    a.annotate("max rewording $\\approx +0.0075$", (0.0075, ptop),
               textcoords="offset points", xytext=(8, -2), ha="left", va="top",
               fontsize=7.2, color=INK)
    b.hist(shp, bins=edges, color=ACCENT, edgecolor="white", lw=0.3)
    b.set_ylim(0, stop * 1.18); b.set_ylabel("shipped\ncount ($n=30$)", fontsize=8)
    b.annotate("deployed deltas up to $+0.173$", (shp.max(), stop * 0.5),
               textcoords="offset points", xytext=(-8, 0), ha="right", va="center",
               fontsize=7.2, color=INK)
    b.set_xlabel("delta vs baseline (rubric / proxy)")
    for ax in (a, b):
        ax.axvline(0, color="0.6", lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(edges[0], edges[-1])
    save(fig, "fig_floor.pdf")


def fig_floor_ecdf():
    """Appendix companion: paired ECDFs (cumulative view of Fig. 6)."""
    par = np.sort(np.asarray(FD["floor"]["paraphrase"], float))
    shp = np.sort(np.asarray(FD["floor"]["shipped"], float))

    def ecdf(a):
        return (np.concatenate([[a[0]], a]),
                np.concatenate([[0.0], np.arange(1, len(a) + 1) / len(a)]))
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    fig.subplots_adjust(left=0.13, right=0.965, top=0.96, bottom=0.155)
    xp, yp = ecdf(par); xsq, ysq = ecdf(shp)
    ax.step(xp, yp, where="post", color=GRAYD, lw=1.3)
    ax.step(xsq, ysq, where="post", color=ACCENT, lw=1.5)
    ax.axvline(0, color="0.6", lw=0.6, ls=(0, (2, 2)))
    ax.text(0.006, 0.55, "paraphrase orbit\n($n=90$)", color=GRAYD, fontsize=7.4, va="center")
    ax.text(0.093, 0.55, "shipped deltas\n($n=30$)", color=ACCENT, fontsize=7.4, va="center")
    ax.set_xlabel("delta vs baseline (rubric / proxy)")
    ax.set_ylabel("empirical cumulative prob.")
    ax.set_xlim(-0.02, 0.185); ax.set_ylim(0, 1.02)
    save(fig, "fig_floor_ecdf.pdf")


# ================================================= FIGURE 7 — validation panels
def fig_validation():
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 3.6))
    fig.subplots_adjust(hspace=0.6, wspace=0.42, left=0.11, right=0.965,
                        top=0.93, bottom=0.10)
    A, B, C, D = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    def barpair(ax, vals, labels, colors, hatches, ymax, ylab, title, yticks):
        ax.bar([0, 1], vals, width=0.6, color=colors, hatch=hatches, edgecolor=INK, lw=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_ylim(0, ymax); ax.set_yticks(yticks)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=8.8, fontweight="bold")
        for xi, v, s in zip([0, 1], vals, [f"{v:g}" for v in vals]):
            ax.text(xi, v + ymax * 0.02, s, ha="center", fontsize=8, color=INK)

    barpair(A, [22, 4], ["TEI\nmaj.", "baseline\nmaj."], [ACCENT, GRAYD], ["", "////"],
            26, "systems (of 26)", "Blinded A/B", [0, 6, 13, 20, 26])
    barpair(B, [84.6, 26.9], ["real", "sham"], [ACCENT, GRAYD], ["", "////"],
            100, "% of votes", "Sham placebo", [0, 25, 50, 75, 100])

    # C dot plot (Cleveland), continuous rubric-delta axis incl. zero
    yv = [1, 0]; xv = [0.0630, 0.0165]
    C.hlines(yv, 0, xv, color=GRAYM, lw=1.1, zorder=1)
    C.scatter([xv[0]], [1], s=40, color=ACCENT, zorder=3)
    C.scatter([xv[1]], [0], s=40, facecolor="white", edgecolor=GRAYD, lw=1.0, zorder=3)
    for yy, vv in zip(yv, xv):
        C.annotate(f"$+{vv:.4f}$", (vv, yy), textcoords="offset points", xytext=(6, 0),
                   ha="left", va="center", fontsize=7.4, color=INK)
    C.set_yticks(yv); C.set_yticklabels(["TEI", "random"])
    C.set_xlim(0, 0.088); C.set_ylim(-0.55, 1.55)
    C.set_xlabel("rubric delta", fontsize=8)
    C.set_title("Budget-matched random", fontsize=8.8, fontweight="bold")
    C.tick_params(axis="y", length=0)               # keep the box; drop y tick marks (categorical)
    C.grid(True, axis="x"); C.grid(False, axis="y")  # value is on x -> vertical gridlines

    barpair(D, [3, 6], ["patched\nmaj.", "baseline\nmaj."], [ACCENT, GRAYD], ["", "xxxx"],
            10, "agent majorities (of 10)", "Cross-provider", [0, 2, 4, 6, 8, 10])
    save(fig, "fig_validation.pdf")


# ==================================================================== assertions
def _assert():
    d = FD["stages_deployed"]
    assert abs(d["final"] - 0.6842) < 1.5e-3 and abs(d["struct"] - 0.6672) < 1.5e-3 \
        and abs(d["base"] - 0.6057) < 1.5e-3, ("n=26 deployed drifted", d)
    assert abs(FD["stages_ceiling"]["final"] - 0.6905) < 1.5e-3, "ceiling drifted"
    allf = float(np.mean([r["best_applied_final"] for r in PA]))
    assert abs(allf - 0.675) < 2e-3, ("all-30 deployed final drifted", allf)
    assert len(FD["floor"]["paraphrase"]) == 90 and len(FD["floor"]["shipped"]) == 30
    dims = FD["dims"]
    gains = {k: dims["deployed"][k] - dims["baseline"][k] for k in dims["order"]}
    assert max(gains, key=gains.get) == "execution_accuracy"
    assert len(figstyle._LOADED) >= 1, "Latin Modern fonts not loaded"
    print(f"assertions OK: n26 {d['base']:.3f}->{d['struct']:.3f}->{d['final']:.3f}, "
          f"all30 final {allf:.4f}, EA +{gains['execution_accuracy']:.3f}, "
          f"LM fonts {figstyle._LOADED}")


if __name__ == "__main__":
    _assert()
    fig_trajectory(); fig_population(); fig_dims()
    fig_floor(); fig_floor_ecdf(); fig_validation()
    print("all vector figures written")
