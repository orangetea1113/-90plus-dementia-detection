"""Shared matplotlib style, palette and helpers for the plot_*.py scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE    = Path(__file__).parent
RESULTS = HERE / "results"
FIGS    = RESULTS / "figures"

# ── palette (validated reference categorical slots 1-4, light surface) ────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
SLOTS     = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

MODEL_LABEL = {"sgbt": "SGBT", "elasticnet": "Elastic Net"}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    AXIS,
    "axes.labelcolor":   INK_2,
    "axes.titlecolor":   INK,
    "axes.titlesize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelcolor":  INK_2,
    "ytick.labelcolor":  INK_2,
    "legend.frameon":    False,
    "savefig.dpi":       200,
    "savefig.facecolor": SURFACE,
})


NO_FILTER = -1.0   # sentinel for the no-CCC-filter baseline (blank in the csv)


def _cond(c: float, prefix: str = "") -> str:
    return "All\n(no filter)" if c == NO_FILTER else f"{prefix}≥{c:.2f}"


def _xticklabels(summary: pd.DataFrame) -> list[str]:
    t = summary.drop_duplicates("ccc_threshold").sort_values("ccc_threshold")
    return [f"{_cond(c)}\nn={n}".replace("(no filter)\n", "")
            for c, n in zip(t["ccc_threshold"], t["n_features"])]
