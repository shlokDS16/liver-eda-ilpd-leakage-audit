"""Shared plotting style for every figure in the report.

One accent colour (oxblood) carries the "disease" class, a muted teal carries
"no disease", and everything else stays in greys so the eye goes to the data.
"""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

INK = "#1F2328"
MUTED = "#6B7280"
RULE = "#D9D4CC"
OXBLOOD = "#8C2F39"
TEAL = "#2F5D62"
SAND = "#C9A66B"
SLATE = "#4B5563"
PAPER = "#FFFFFF"

CLASS_COLORS = {1: OXBLOOD, 0: TEAL}
CLASS_NAMES = {1: "Liver disease", 0: "No disease"}
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("ox", ["#F7F3EE", "#D9A5A0", OXBLOOD, "#4A1219"])
DIV_CMAP = mpl.colors.LinearSegmentedColormap.from_list("div", [TEAL, "#F7F3EE", OXBLOOD])
MODEL_COLORS = [OXBLOOD, TEAL, SAND, SLATE, "#A3505B", "#6E9497", "#9C8A6B", "#9CA3AF"]


def apply():
    mpl.rcParams.update({
        "font.family": "Bahnschrift",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#EEEAE4",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "text.color": INK,
        "axes.unicode_minus": False,
    })


def _plain(v, _):
    return f"{v:g}"


def save(fig, path):
    # log axes: plain numbers instead of 10^x mathtext (Bahnschrift has no math minus glyph)
    for ax in fig.axes:
        for axis, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if sc == "log":
                axis.set_major_formatter(mpl.ticker.FuncFormatter(_plain))
                axis.set_minor_formatter(mpl.ticker.NullFormatter())
    fig.savefig(path)
    plt.close(fig)
