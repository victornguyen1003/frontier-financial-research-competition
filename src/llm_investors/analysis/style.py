"""Shared figure style. Categorical slots 1-3 of the validated reference palette (they pass
colorblind checks for any pair), a blue/red diverging pair with a gray midpoint, and recessive
axes. Colors follow the entity: an architecture keeps its color in every figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ARCH_COLORS = {"none": "#2a78d6", "log": "#eb6834", "reflect": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BUY, SELL, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 150,
    })
