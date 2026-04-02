#!/usr/bin/env python3
"""Generate thesis figures for token-level GFM classifier."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent
RESULTS = Path("/work/ymj1123ntu/token_level_gfm_classifier/results")

# ── Consistent style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "train": "#4878CF",
    "val":   "#D65F5F",
    "v3":    "#E07B39",
    "v8":    "#4878CF",
    "v9":    "#6BBF59",
    "v11":   "#9B59B6",
    "meta":  "#C0392B",
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training dynamics — 500K (overfitting) vs 5M vs 50M
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_dynamics():
    v3  = pd.read_csv(RESULTS / "nt_token_genus_lora_v3/training_history.csv")
    v8  = pd.read_csv(RESULTS / "nt_token_genus_lora_v8_5M/training_history.csv")
    v9  = pd.read_csv(RESULTS / "nt_token_genus_lora_v9_50M/training_history.csv")

    # v8: epochs 1-14 are clean (first run); epoch 15 has LR reset → mark with dashed
    v8_clean = v8[v8["epoch"] <= 14]
    v8_resume = v8[v8["epoch"] >= 14]   # overlap at 14 for continuity

    # v9: epochs 1-10 are first run, 11+ are LR-reset resume
    v9_clean = v9[v9["epoch"] <= 10]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    datasets = [
        (axes[0], v3,       None,      "v3 — 500K reads (imbalanced)", COLORS["v3"]),
        (axes[1], v8_clean, v8_resume, "v8 — 5M reads (balanced)",      COLORS["v8"]),
        (axes[2], v9_clean, None,      "v9 — 50M reads (balanced)",      COLORS["v9"]),
    ]

    for ax, df_main, df_resume, title, color in datasets:
        # Training accuracy
        ax.plot(df_main["epoch"], df_main["train_acc"] * 100,
                color=color, lw=1.5, ls="--", alpha=0.7, label="Train")
        # Validation accuracy
        ax.plot(df_main["epoch"], df_main["val_acc"] * 100,
                color=color, lw=2.0, label="Validation")

        if df_resume is not None:
            ax.plot(df_resume["epoch"], df_resume["train_acc"] * 100,
                    color=color, lw=1.5, ls="--", alpha=0.4)
            ax.plot(df_resume["epoch"], df_resume["val_acc"] * 100,
                    color=color, lw=2.0, alpha=0.4)
            # Mark LR reset
            ax.axvline(x=15, color="gray", lw=1.0, ls=":", alpha=0.8)
            ax.text(15.3, ax.get_ylim()[0] + 1, "LR reset", fontsize=8,
                    color="gray", va="bottom")

        # Shade train-val gap region
        ax.fill_between(df_main["epoch"],
                         df_main["train_acc"] * 100,
                         df_main["val_acc"] * 100,
                         alpha=0.08, color=color)

        ax.set_title(title, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(left=1)

    # Highlight best val on v3
    best_v3 = v3.loc[v3["val_acc"].idxmax()]
    axes[0].annotate(f'Best: {best_v3["val_acc"]*100:.1f}%',
                     xy=(best_v3["epoch"], best_v3["val_acc"]*100),
                     xytext=(best_v3["epoch"] - 8, best_v3["val_acc"]*100 - 2.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                     fontsize=9, color="gray")

    # Highlight best val on v9
    best_v9 = v9_clean.loc[v9_clean["val_acc"].idxmax()]
    axes[2].annotate(f'Best: {best_v9["val_acc"]*100:.1f}%',
                     xy=(best_v9["epoch"], best_v9["val_acc"]*100),
                     xytext=(best_v9["epoch"] - 5, best_v9["val_acc"]*100 - 2.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                     fontsize=9, color="gray")

    fig.suptitle("Training Dynamics: Overfitting Decreases with Data Scale",
                 fontsize=13, y=1.02)

    plt.savefig(OUT / "training_dynamics.pdf", bbox_inches="tight")
    plt.savefig(OUT / "training_dynamics.png", bbox_inches="tight")
    plt.close()
    print("Saved training_dynamics.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Data scaling law
# ─────────────────────────────────────────────────────────────────────────────
def fig_data_scaling():
    # Our results
    reads  = np.array([500_000, 5_000_000, 50_000_000])
    acc    = np.array([55.29, 63.05, 66.06])

    # MetaTransformer (estimated ~614M balanced reads, 98.3% recall)
    meta_reads = 614_000_000
    meta_acc   = 98.3

    # Fit log-linear curve to our three points
    log_reads = np.log10(reads)
    coeffs = np.polyfit(log_reads, acc, 1)   # linear in log-space
    x_fit  = np.logspace(np.log10(4e5), np.log10(6e8), 300)
    y_fit  = np.polyval(coeffs, np.log10(x_fit))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Fitted curve
    ax.plot(x_fit, y_fit, color="#4878CF", lw=1.5, ls="--", alpha=0.6,
            label="Log-linear fit (ours)")

    # Our data points
    ax.scatter(reads, acc, color=COLORS["v9"], s=90, zorder=5, label="This work")
    labels = ["v4\n500K", "v8\n5M", "v9\n50M"]
    offsets = [(-0.25, -3.5), (-0.25, 1.5), (0.08, 1.5)]
    for x, y, lbl, (dx, dy) in zip(reads, acc, labels, offsets):
        ax.annotate(f"{lbl}\n({y:.1f}%)",
                    xy=(x, y), xytext=(x * 10**dx, y + dy),
                    fontsize=8.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    # MetaTransformer
    ax.scatter([meta_reads], [meta_acc], color=COLORS["meta"],
               s=100, marker="*", zorder=5, label="MetaTransformer")
    ax.annotate(f"MetaTransformer\n~614M reads\n({meta_acc}% recall)",
                xy=(meta_reads, meta_acc),
                xytext=(meta_reads * 0.18, meta_acc - 9),
                fontsize=8.5, ha="center", color=COLORS["meta"],
                arrowprops=dict(arrowstyle="->", color=COLORS["meta"], lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("Training reads (log scale)")
    ax.set_ylabel("Genus RC TTA Accuracy (%)")
    ax.set_title("Data Scaling: Genus-Level Classification Accuracy vs.\ Training Volume")
    ax.set_ylim(45, 102)
    ax.set_xlim(2e5, 2e9)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate gains
    ax.annotate("", xy=(5e6, 63.05), xytext=(5e5, 55.29),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(1.5e6, 59.5, "+7.76 pp\n(10×)", fontsize=8, ha="center", color="gray")

    ax.annotate("", xy=(5e7, 66.06), xytext=(5e6, 63.05),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(1.5e7, 65.1, "+3.01 pp\n(10×)", fontsize=8, ha="center", color="gray")

    plt.savefig(OUT / "data_scaling.pdf", bbox_inches="tight")
    plt.savefig(OUT / "data_scaling.png", bbox_inches="tight")
    plt.close()
    print("Saved data_scaling.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Backbone ablation — v9 vs v11
# ─────────────────────────────────────────────────────────────────────────────
def fig_backbone_ablation():
    metrics = ["Micro\nAccuracy", "Balanced\nAccuracy", "F1\n(macro)", "Top-3\nAccuracy", "Top-5\nAccuracy"]
    v9_vals  = [66.06, 37.52, 43.05, 83.74, 89.56]
    v11_vals = [53.88, 22.24, 25.78, 73.85, 81.86]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    bars_v9  = ax.bar(x - width/2, v9_vals,  width, label="v9 (NT-v2 + LoRA, 29-layer)",
                      color=COLORS["v9"],  alpha=0.88, edgecolor="white")
    bars_v11 = ax.bar(x + width/2, v11_vals, width, label="v11 (Shallow, 1-layer, random init)",
                      color=COLORS["v11"], alpha=0.88, edgecolor="white")

    # Value labels (inside bar top to avoid crowding)
    for bar in bars_v9:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 2.5, f"{h:.1f}",
                ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")
    for bar in bars_v11:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 2.5, f"{h:.1f}",
                ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")

    # Gap annotations: placed above the taller bar with a bracket-style indicator
    for i, (v9, v11) in enumerate(zip(v9_vals, v11_vals)):
        gap = v9 - v11
        top = max(v9, v11)
        ax.annotate(f"Δ+{gap:.1f}",
                    xy=(x[i], top + 4.5),
                    fontsize=8, ha="center", color="dimgray",
                    style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Backbone Ablation: Pre-trained NT-v2 vs.\\ Shallow Transformer (50M balanced reads)")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.savefig(OUT / "backbone_ablation.pdf", bbox_inches="tight")
    plt.savefig(OUT / "backbone_ablation.png", bbox_inches="tight")
    plt.close()
    print("Saved backbone_ablation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: RC TTA benefit across experiments
# ─────────────────────────────────────────────────────────────────────────────
def fig_rc_tta():
    exps   = ["v3\n500K", "v4\n500K", "v5b\n500K\nLA", "v7\n500K\nRC-cons.",
              "v8\n5M", "v9\n50M\n(NT-v2)", "v11\n50M\n(shallow)"]
    fwd    = [53.92, 53.75, 52.29, 54.10, 62.02, 65.29, 53.80]
    rc_tta = [55.36, 55.29, 53.58, 55.18, 63.05, 66.06, 53.88]
    gains  = [r - f for r, f in zip(rc_tta, fwd)]

    x = np.arange(len(exps))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.1)

    # Top: grouped bar chart
    color_fwd    = "#7BAFD4"
    color_rctta  = "#2166AC"

    ax1.bar(x - width/2, fwd,    width, label="Forward only",   color=color_fwd,   alpha=0.88, edgecolor="white")
    ax1.bar(x + width/2, rc_tta, width, label="+ RC TTA",        color=color_rctta, alpha=0.88, edgecolor="white")

    ax1.set_xticks([])
    ax1.set_ylabel("Genus Accuracy (%)")
    ax1.set_title("RC TTA Benefit Across Experiments")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(40, 72)
    ax1.set_xlim(-0.6, len(exps) - 0.4)

    # Separate v11 visually
    ax1.axvline(x=5.5, color="lightgray", lw=1.2, ls="--")
    ax1.text(5.52, 68, "← NT-v2 backbone  |  random init →",
             fontsize=8, color="gray", va="top")

    # Bottom: gain
    bar_colors = [COLORS["v9"] if g > 0.5 else COLORS["v11"] for g in gains]
    ax2.bar(x, gains, width=0.55, color=bar_colors, alpha=0.85, edgecolor="white")
    for i, g in enumerate(gains):
        ax2.text(x[i], g + 0.02, f"+{g:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(exps, fontsize=8.5)
    ax2.set_ylabel("Gain (pp)")
    ax2.set_ylim(0, 2.0)
    ax2.set_xlim(-0.6, len(exps) - 0.4)
    ax2.axvline(x=5.5, color="lightgray", lw=1.2, ls="--")

    plt.savefig(OUT / "rc_tta_benefit.pdf", bbox_inches="tight")
    plt.savefig(OUT / "rc_tta_benefit.png", bbox_inches="tight")
    plt.close()
    print("Saved rc_tta_benefit.pdf")


if __name__ == "__main__":
    fig_training_dynamics()
    fig_data_scaling()
    fig_backbone_ablation()
    fig_rc_tta()
    print("All figures generated.")
