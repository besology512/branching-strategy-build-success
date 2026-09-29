"""One visual language for every figure in the repository.

Two families carry meaning throughout: branching features/strategies built on short-lived
work vs. the rest. Colours were validated for colour-vision deficiency; every figure also
encodes the same distinction with marker shape or direct labels, so nothing relies on
colour alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"          # standard vs branching
AQUA, YELLOW, MAGENTA = "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"

# Fixed colour per strategy (never re-assigned by rank)
STRATEGY_COLORS = {
    "One Branch": AQUA,
    "Trunk-based": BLUE,
    "Feature Branching": YELLOW,
    "Git Flow": ORANGE,
}
STRATEGY_ORDER = ["One Branch", "Trunk-based", "Feature Branching", "Git Flow"]


def use():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"figure -> {path}")
