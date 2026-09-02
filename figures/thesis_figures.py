#!/usr/bin/env python3
"""Every figure in the thesis, drawn in the manuscript's house style.

THE RULE: a figure is drawn at the size it is printed at, so \\includegraphics
never scales it. Scaling is what made the old figure set inconsistent --- label
text ranged from 4.9pt to 10.8pt across ten figures, because each was drawn at
some other width and then squeezed or stretched to fit. Draw at WIDE or NARROW
below, place with the matching \\includegraphics width, and 9pt stays 9pt.

  \\textwidth = 425pt = 5.90in   (A4 595pt, 3cm margins both sides)

Data provenance: every number is quoted from a table in Chapter 4 or Appendix B,
named in the SOURCE comment above each dataset, so figures and tables cannot
drift apart. The epoch-by-epoch series for the two training figures come from
tab:v3-dynamics and tab:v8-dynamics (removed from the text in commit 6b80ffd,
still in its history) and tab:spv4-progress in Appendix B.

Retired: the abundance scatter, the detection ROC curve and the per-genus
accuracy scatter were drawn from per-sample prediction dumps under
/work/ymj1123ntu, which no longer exists. They plotted quantities the tables
already report, so they were dropped rather than reproduced at lower fidelity.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import apply_style, audit_text, BLUE, TEAL, ORANGE, GREY, RED

OUT = Path(__file__).parent
apply_style()

WIDE = 5.90      # place with \includegraphics[width=\textwidth]
NARROW = 4.13    # place with \includegraphics[width=0.70\textwidth]


def _tag(ax, letter):
    ax.text(-0.20, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def _label_bars(ax, bars, values, fmt="{:.1f}", pad=1.0, size=8):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,
                fmt.format(v), ha="center", va="bottom", fontsize=size)


def _save(fig, name):
    audit_text(fig, name)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# ─── 4.5 data scaling ────────────────────────────────────────────────────────
# SOURCE: tab:scaling
def fig_data_scaling():
    reads = np.array([0.5e6, 5e6, 50e6, 250e6])
    acc = np.array([55.29, 63.05, 67.07, 67.29])

    fig, ax = plt.subplots(figsize=(NARROW, 2.85))
    fit = np.polyfit(np.log10(reads[:3]), acc[:3], 1)
    xs = np.logspace(np.log10(0.4e6), np.log10(3e8), 60)
    ax.plot(xs, np.polyval(fit, np.log10(xs)), ":", color=GREY, lw=1.2, zorder=2,
            label="log-linear fit, 0.5M–50M")
    ax.semilogx(reads, acc, "o-", color=BLUE, zorder=3, label="NT-Genus, 6-mer")
    ax.semilogx([50e6, 250e6], [87.42, 98.7], "s--", color=TEAL, zorder=3,
                label="MetaTransformer, 13-mer")
    ax.annotate("+0.22 pp", xy=(250e6, 67.29), xytext=(0.9e8, 54),
                fontsize=8, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.set_xlim(3.5e5, 4.5e8)
    ax.set_ylim(50, 105)
    ax.set_xlabel("Training reads")
    ax.set_ylabel("Genus Top-1, RC TTA (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    _save(fig, "data_scaling")


# ─── 4.3/4.5 training dynamics ───────────────────────────────────────────────
# SOURCE: tab:v3-dynamics, tab:v8-dynamics (commit 6b80ffd history)
def fig_training_dynamics():
    v3_ep = [1, 5, 10, 15, 20, 28, 35]
    v3_tr = [45.25, 51.34, 54.25, 56.25, 58.08, 60.88, 62.69]
    v3_va = [46.29, 51.27, 52.39, 53.26, 53.88, 53.92, 53.71]
    v8_ep = [1, 5, 10, 14, 15, 20, 25, 29]
    v8_tr = [49.96, 55.97, 58.93, 60.60, 58.80, 61.82, 62.89, 63.32]
    v8_va = [51.80, 57.04, 59.42, 60.56, 59.18, 60.59, 61.77, 62.02]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.24)
    for ax, ep, tr, va, title, colour, letter in [
        (axes[0], v3_ep, v3_tr, v3_va, "500K reads (imbalanced)", ORANGE, "a"),
        (axes[1], v8_ep, v8_tr, v8_va, "5M reads (balanced)", BLUE, "b"),
    ]:
        ax.plot(ep, tr, "--", color=colour, lw=1.4, alpha=0.75, label="Train")
        ax.plot(ep, va, "-", color=colour, lw=1.8, label="Validation")
        ax.fill_between(ep, va, tr, color=colour, alpha=0.12, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 37)
        _tag(ax, letter)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(43, 67)
    axes[0].annotate("train–val gap\n9.0 pp at epoch 35", xy=(35, 58.2),
                     xytext=(11, 44.2), fontsize=7.5, color="#444444",
                     arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[1].axvline(15, color=RED, lw=0.9, ls=":", zorder=2)
    axes[1].text(16.0, 44.2, "LR reset on resume", fontsize=7, color=RED)
    axes[1].annotate("gap ≤ 1.3 pp throughout", xy=(29, 62.7), xytext=(6.5, 65.4),
                     fontsize=7.5, color="#444444",
                     arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    _save(fig, "training_dynamics")


# ─── 4.14 species-level learning curve ───────────────────────────────────────
# SOURCE: tab:spv4-progress (Appendix B)
def fig_learning_curves():
    ep = [1, 6, 7, 12, 13, 18, 26, 30]
    acc = [14.23, 15.91, 15.12, 15.51, 15.85, 16.37, 17.31, 17.55]
    f1 = [12.43, 14.05, 13.36, 13.65, 13.97, 14.46, 15.28, 15.70]

    fig, ax = plt.subplots(figsize=(NARROW, 2.75))
    for x in (7, 13):
        ax.axvline(x, color=RED, lw=0.8, ls=":", zorder=2)
    ax.plot(ep, acc, "o-", color=BLUE, zorder=3, label="Validation Top-1")
    ax.plot(ep, f1, "s--", color=GREY, lw=1.4, zorder=3, label="Macro F1")
    ax.text(7.5, 11.9, "LR reset", fontsize=7, color=RED)
    ax.text(13.5, 17.9, "resume fixed", fontsize=7, color=RED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Species, 1,535 classes (%)")
    ax.set_xlim(0, 32)
    ax.set_ylim(11.3, 18.9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _save(fig, "learning_curves")


# ─── 4.7 RC TTA ──────────────────────────────────────────────────────────────
# SOURCE: tab:rc-tta
def fig_rc_tta():
    labels = ["v3\n500K", "NT-Genus\n5M", "NT-Genus\n50M", "DNABERT",
              "DNABERT-2", "KmerFormer\n1 layer"]
    gains = [1.44, 1.03, 0.78, 0.58, 1.53, 0.08]
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE, GREY]

    fig, ax = plt.subplots(figsize=(WIDE, 2.5))
    bars = ax.bar(np.arange(len(labels)), gains, width=0.58, color=colors, zorder=3)
    _label_bars(ax, bars, gains, fmt="+{:.2f}", pad=0.05)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("RC TTA gain (pp)")
    ax.set_ylim(0, 1.82)
    ax.axhline(0.12, color=RED, lw=0.9, ls="--", zorder=2)
    ax.text(-0.42, 0.19, "run-to-run s.d.", fontsize=7.5, color=RED, ha="left")
    _save(fig, "rc_tta_benefit")


# ─── 4.10 depth at a fixed 6-mer tokenizer ───────────────────────────────────
# SOURCE: tab:v11-results
def fig_depth_sweep():
    depths = [1, 8, 16, 29]
    acc = [53.92, 65.44, 67.94, 69.52]
    params = [0.90, 2.29, 3.88, 6.45]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.7))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    vals = [48.87, 67.08, 69.52]
    bars = ax.bar(np.arange(3), vals, width=0.6,
                  color=[ORANGE, GREY, BLUE], zorder=3)
    _label_bars(ax, bars, vals, fmt="{:.2f}%", pad=1.4)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["MT\n1 layer", "NT-v2\n500M", "KmerFormer\n29 layers"])
    ax.set_ylim(0, 84)
    ax.set_ylabel("Genus Top-1, RC TTA (%)")
    ax.set_title("At a shared 6-mer tokenizer")
    _tag(ax, "a")

    ax = axes[1]
    ax.axhline(67.08, color=GREY, ls="--", lw=1.2, zorder=2)
    ax.plot(depths, acc, "o-", color=BLUE, zorder=3)
    ax.text(-1.2, 67.6, "NT-v2, 500M", fontsize=7.5, color="#444444", ha="left")
    for d, a, p in zip(depths, acc, params):
        ax.annotate(f"{p:g}M", xy=(d, a), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="#555555")
    ax.set_xlabel("Encoder layers")
    ax.set_ylabel("Genus Top-1 (%)")
    ax.set_xticks(depths)
    ax.set_xlim(-2, 33)
    ax.set_ylim(49, 73.5)
    ax.set_title("Depth alone: $+15.60$ pp")
    _tag(ax, "b")
    _save(fig, "depth_sweep_6mer")


# ─── 4.12 the 13-mer arms ────────────────────────────────────────────────────
# SOURCE: tab:thirteenmer
def fig_thirteenmer():
    rows = [
        ("exact 13-mer, 1 layer",      91.15, TEAL),
        ("MetaTransformer, 1 layer",   87.47, GREY),
        ("hashed $d128$, 1 layer",     86.80, BLUE),
        ("exact 13-mer, 16 layers",    85.63, TEAL),
        ("hashed $d128$, 16 layers",   83.83, BLUE),
        ("hashed $d64$, 16 layers",    79.26, BLUE),
        ("6-mer, 16 layers",           67.94, ORANGE),
    ][::-1]

    # tight bbox adds the y-label column, so the drawn width is set below WIDE
    fig, ax = plt.subplots(figsize=(WIDE - 0.30, 2.9))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r[1] for r in rows], height=0.62,
                   color=[r[2] for r in rows], zorder=3)
    for b, r in zip(bars, rows):
        ax.text(r[1] + 0.5, b.get_y() + b.get_height() / 2, f"{r[1]:.2f}%",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(62, 97)
    ax.set_xlabel("Genus Top-1, RC TTA (%)")
    ax.grid(axis="y", visible=False)
    _save(fig, "thirteenmer_arms")


# ─── 4.10 the soil crossover ─────────────────────────────────────────────────
# SOURCE: 4.10, "The crossover replicates on a second catalogue"
def fig_soil():
    x = [5, 50]
    fig, ax = plt.subplots(figsize=(NARROW, 2.75))
    ax.plot(x, [31.98, 39.84], "o-", color=GREY, label="NT-v2, pre-trained")
    ax.plot(x, [27.42, 43.68], "s-", color=BLUE, label="KmerFormer, from scratch")
    ax.annotate("$+4.6$ pp", xy=(5.4, 29.5), fontsize=8, color="#444444")
    ax.annotate("$-3.8$ pp", xy=(28, 43.4), fontsize=8, color="#444444")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["5M", "50M"])
    ax.set_xlim(3.6, 78)
    ax.set_ylim(24, 49)
    ax.set_xlabel("Training reads (soil, 309 genera)")
    ax.set_ylabel("Genus Top-1 (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _save(fig, "soil_data_scale")


# ─── 4.15 read accuracy against sample-level utility ─────────────────────────
# SOURCE: tab:sample-model-comparison
def fig_sample_level_models():
    models = ["MT\n6-mer", "NT-\nGenus", "MT\n13-mer"]
    colors = [ORANGE, BLUE, TEAL]
    read_acc = [48.82, 67.07, 87.42]
    pearson = [0.984, 0.993, 0.999]
    roc_auc = [0.569, 0.705, 0.900]
    sens95 = [9.4, 17.2, 47.7]

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.6))
    fig.subplots_adjust(wspace=0.46)

    ax = axes[0]
    bars = ax.bar(np.arange(3), read_acc, width=0.6, color=colors, zorder=3)
    _label_bars(ax, bars, read_acc, fmt="{:.1f}%", pad=1.5)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(models)
    ax.set_ylim(0, 102); ax.set_ylabel("Genus Top-1 (%)")
    ax.set_title("(a)  Read level")

    ax = axes[1]
    bars = ax.bar(np.arange(3), pearson, width=0.6, color=colors, zorder=3)
    _label_bars(ax, bars, pearson, fmt="{:.3f}", pad=0.0018)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(models)
    ax.set_ylim(0.95, 1.008); ax.set_ylabel("Pearson $r$")
    ax.set_title("(b)  Relative abundance")

    ax = axes[2]
    w = 0.34
    b1 = ax.bar(np.arange(3) - w / 2, roc_auc, width=w, color=colors, zorder=3)
    b2 = ax.bar(np.arange(3) + w / 2, [s / 100 for s in sens95], width=w,
                color=colors, alpha=0.42, zorder=3)
    for b, v in zip(b1, roc_auc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    for b, v in zip(b2, sens95):
        ax.text(b.get_x() + b.get_width() / 2, v / 100 + 0.02, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7, alpha=0.9)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Score")
    ax.legend([b1[0], b2[0]], ["ROC AUC", "Sens. @ 95% spec."],
              loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2,
              frameon=False, handlelength=1.0, fontsize=7, columnspacing=1.0)
    ax.set_title("(c)  Presence / absence")
    _save(fig, "sample_level_models")


# ─── 4.15 detection sensitivity by abundance ─────────────────────────────────
# SOURCE: tab:sample-detection-abundance
def fig_detection_by_abundance():
    bins = ["0.01–0.1%", "0.1–1%", "1–5%", ">5%"]
    counts = [651, 7654, 2975, 720]
    sens = [93.9, 97.6, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(NARROW, 2.6))
    bars = ax.bar(np.arange(4), sens, width=0.6, color=BLUE, zorder=3)
    for b, v, n in zip(bars, sens, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
        ax.text(b.get_x() + b.get_width() / 2, 90.4, f"n={n:,}",
                ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(bins)
    ax.set_ylim(90, 101.8)
    ax.set_xlabel("True relative abundance of the genus")
    ax.set_ylabel("Detection sensitivity (%)")
    _save(fig, "detection_by_abundance")


if __name__ == "__main__":
    for fn in (fig_data_scaling, fig_training_dynamics, fig_learning_curves,
               fig_rc_tta, fig_depth_sweep, fig_thirteenmer, fig_soil,
               fig_sample_level_models, fig_detection_by_abundance):
        fn()
