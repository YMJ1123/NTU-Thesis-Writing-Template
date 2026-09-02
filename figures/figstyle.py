"""Shared matplotlib style for all manuscript figures (unified fonts/sizes).
Import and call apply_style() at the top of every figure script."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext

# Palette (paper accent = OUP bibblue 0,63,114)
BLUE   = "#003f72"   # primary / NT-v2
TEAL   = "#2a9d8f"   # MT 13-mer
ORANGE = "#e07a2f"   # MT 6-mer / secondary
GREY   = "#8a8f98"   # Kraken2 / neutral
LGREY  = "#c9ccd1"
RED    = "#c1432e"   # negative / genus-balanced
GREEN  = "#4c956c"
PALETTE = [BLUE, TEAL, ORANGE, GREY, RED, GREEN]


def audit_text(fig, name, tol=0.5):
    """Pairwise bounding-box overlap check over every visible Text artist.

    Returns the list of colliding pairs (empty == clean) and prints a report.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Matplotlib keeps tick objects whose locations fall outside the view
    # interval. They are clipped at draw time and never appear in the output,
    # but they still report a window extent, so measuring them invents
    # collisions with real labels. Exclude them before comparing anything.
    phantom = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            for tick, loc in zip(axis.get_major_ticks(),
                                 axis.get_majorticklocs()):
                if not lo <= loc <= hi:
                    phantom.update({id(tick.label1), id(tick.label2)})

    items = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in phantom:
            continue
        bb = t.get_window_extent(renderer=renderer)
        if bb.width > 0 and bb.height > 0:
            items.append((t.get_text().replace("\n", " / "), bb))
    clashes = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > tol and oy > tol:
                clashes.append((items[i][0], items[j][0],
                                round(ox, 1), round(oy, 1)))
    if clashes:
        print(f"  !! {name}: {len(clashes)} overlapping text pair(s)")
        for a, b, ox, oy in clashes:
            print(f"     '{a}'  x  '{b}'   ({ox} x {oy} px)")
    else:
        print(f"  ok {name}: {len(items)} text artists, no overlaps")
    return clashes

def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.titlesize": 10,
        # Not bold. A bold axes title is emphasis applied to every panel
        # heading whether or not it needs any, and it set the tone the value
        # labels and Venn counts then followed. Panel tags stay bold: they are
        # structural markers, not text a reader reads.
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,   # grid lines and ticks stay behind bars/lines
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed as TrueType (no Type-3), journal-safe
        "ps.fonttype": 42,
    })

if __name__ == "__main__":
    apply_style()
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot([1,2,3],[1,4,9], marker="o", color=BLUE, label="test $r=0.99$")
    ax.set_xlabel("x label"); ax.set_ylabel("y label"); ax.set_title("Font test")
    ax.legend()
    fig.savefig("/tmp/claude-27474/-home-ymj1123ntu/bc482539-0db5-47b0-8470-5c579748a179/scratchpad/fonttest.pdf")
    print("wrote fonttest.pdf")
