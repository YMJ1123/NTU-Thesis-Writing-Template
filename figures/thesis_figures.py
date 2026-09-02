#!/usr/bin/env python3
"""Thesis figures, drawn in the same house style as the manuscript figures.

Every number here is quoted from a table in Chapter 4, so the figures and the
tables cannot drift apart: see the SOURCE comment above each dataset.

The earlier raster figures in this directory were produced from per-sample
prediction dumps under /work/ymj1123ntu, which no longer exists. Figures whose
underlying points cannot be reconstructed from the thesis tables (the abundance
scatter, the ROC curve and the per-genus accuracy scatter) were retired rather
than reproduced at lower fidelity; the quantities they showed are reported in
Tables 4.x and Appendix B.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import apply_style, audit_text, BLUE, TEAL, ORANGE, GREY

OUT = Path(__file__).parent
apply_style()


def _tag(ax, letter):
    """Panel tag in the manuscript's convention: bold, above-left of the axes."""
    ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def _bars(ax, labels, values, colors, fmt="{:.1f}", pad=1.0):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.62, color=colors, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v),
                ha="center", va="bottom", fontsize=8)
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — read-level accuracy against sample-level utility.
# SOURCE: Table "Sample-level evaluation across three models" (tab:sample-model-comparison)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sample_level_models():
    models = ["MT\n6-mer", "NT-\nGenus", "MT\n13-mer"]
    colors = [ORANGE, BLUE, TEAL]

    read_acc = [48.82, 67.07, 87.42]           # genus Top-1, %
    pearson = [0.984, 0.993, 0.999]            # abundance correlation
    roc_auc = [0.569, 0.705, 0.900]            # binary detection
    sens95 = [9.4, 17.2, 47.7]                 # sensitivity at 95% specificity, %

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    fig.subplots_adjust(wspace=0.40, bottom=0.22, top=0.86)

    ax = axes[0]
    _bars(ax, models, read_acc, colors, fmt="{:.1f}%", pad=1.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Genus Top-1 (%)")
    ax.set_title("Read level")
    _tag(ax, "a")

    ax = axes[1]
    _bars(ax, models, pearson, colors, fmt="{:.3f}", pad=0.0015)
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("Pearson $r$")
    ax.set_title("Relative abundance")
    _tag(ax, "b")

    ax = axes[2]
    x = np.arange(len(models))
    w = 0.36
    b1 = ax.bar(x - w / 2, roc_auc, width=w, color=colors, zorder=3)
    b2 = ax.bar(x + w / 2, [s / 100 for s in sens95], width=w, color=colors,
                alpha=0.45, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Presence / absence")
    for b, v in zip(b1, roc_auc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(b2, sens95):
        ax.text(b.get_x() + b.get_width() / 2, v / 100 + 0.02, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7.5, alpha=0.85)
    ax.legend([b1[0], b2[0]], ["ROC AUC", "Sens.\\ @ 95% spec."],
              loc="upper left", frameon=False, handlelength=1.1)
    _tag(ax, "c")

    audit_text(fig, "sample_level_models")
    fig.savefig(OUT / "sample_level_models.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote sample_level_models.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — detection sensitivity against true abundance.
# SOURCE: Table "Detection sensitivity by true relative abundance"
#         (tab:sample-detection-abundance) and tab:sample-model-comparison.
# ─────────────────────────────────────────────────────────────────────────────
def fig_detection_by_abundance():
    bins = ["0.01–0.1%", "0.1–1%", "1–5%", ">5%"]
    counts = [651, 7654, 2975, 720]
    sens = [93.9, 97.6, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(bins))
    bars = ax.bar(x, sens, width=0.6, color=BLUE, zorder=3)
    for b, v, n in zip(bars, sens, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
        ax.text(b.get_x() + b.get_width() / 2, 90.6, f"n={n:,}",
                ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(90, 101.5)
    ax.set_xlabel("True relative abundance of the genus")
    ax.set_ylabel("Detection sensitivity (%)")

    audit_text(fig, "detection_by_abundance")
    fig.savefig(OUT / "detection_by_abundance.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote detection_by_abundance.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — data scaling, restyled from the original generate_figures.py.
# SOURCE: Table "Genus classification across the full data-scale series" (tab:scaling)
# ─────────────────────────────────────────────────────────────────────────────
def fig_data_scaling():
    reads = np.array([0.5e6, 5e6, 50e6, 250e6])
    acc = np.array([55.29, 63.05, 67.07, 67.29])
    mt13 = {50e6: 87.42, 250e6: 98.7}

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.semilogx(reads, acc, "o-", color=BLUE, zorder=3,
                label="NT-Genus (6-mer)")
    ax.semilogx(list(mt13), list(mt13.values()), "s--", color=TEAL, zorder=3,
                label="MetaTransformer (13-mer)")

    # The log-linear projection the first three points invite, and which the
    # fourth refutes: fit on 0.5M-50M, extend to 250M.
    fit = np.polyfit(np.log10(reads[:3]), acc[:3], 1)
    xs = np.logspace(np.log10(0.5e6), np.log10(250e6), 50)
    ax.plot(xs, np.polyval(fit, np.log10(xs)), ":", color=GREY, lw=1.3,
            zorder=2, label="log-linear fit (0.5M–50M)")

    ax.annotate("+0.22 pp", xy=(250e6, 67.29), xytext=(250e6, 58),
                ha="center", fontsize=8, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.set_xlabel("Training reads")
    ax.set_ylabel("Genus Top-1, RC TTA (%)")
    ax.set_ylim(50, 102)
    ax.legend(loc="center left", frameon=False)

    audit_text(fig, "data_scaling")
    fig.savefig(OUT / "data_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote data_scaling.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — RC TTA gain, restyled. SOURCE: Table tab:rc-tta.
# ─────────────────────────────────────────────────────────────────────────────
def fig_rc_tta():
    labels = ["v3\n500K", "NT-Genus-5M", "NT-Genus\n50M", "DNABERT", "DNABERT-2",
              "KmerFormer\n(1 layer)"]
    gains = [1.44, 1.03, 0.78, 0.58, 1.53, 0.08]
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE, GREY]

    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    bars = ax.bar(np.arange(len(labels)), gains, width=0.6, color=colors, zorder=3)
    for b, v in zip(bars, gains):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"+{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("RC TTA gain (pp)")
    ax.set_ylim(0, 1.85)
    ax.axhline(0.12, color="#c1432e", lw=0.9, ls="--", zorder=2)
    ax.text(-0.35, 0.17, "run-to-run s.d.", fontsize=7.5,
            color="#c1432e", ha="left")

    audit_text(fig, "rc_tta_benefit")
    fig.savefig(OUT / "rc_tta_benefit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote rc_tta_benefit.pdf")


if __name__ == "__main__":
    fig_sample_level_models()
    fig_detection_by_abundance()
    fig_data_scaling()
    fig_rc_tta()
