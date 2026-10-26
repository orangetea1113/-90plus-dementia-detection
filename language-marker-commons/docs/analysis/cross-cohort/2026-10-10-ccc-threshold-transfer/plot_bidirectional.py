"""Figure for the bidirectional experiment (results/bidirectional_*.csv).

One panel per model; x = CCC threshold; four series:
within-cohort CV (90+, I-CONECT; mean ± SD over seeds) and transfer in both
directions (95% bootstrap CI). Reuses plot_style.py.

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/plot_bidirectional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import plot_style as P                 # noqa: E402  (sets rcParams)
plt = P.plt

BID = P.RESULTS

# (key in summary, label, color, marker, kind)
SERIES = [
    ("90plus_cv",         "90+ CV",             P.SLOTS[0], "o", "cv"),
    ("iconect_cv",        "I-CONECT CV",        P.SLOTS[2], "o", "cv"),
    ("90plus_to_iconect", "90+ → I-CONECT",     P.SLOTS[1], "s", "tr"),
    ("iconect_to_90plus", "I-CONECT → 90+",     P.SLOTS[3], "s", "tr"),
]


def main() -> None:
    summ = pd.read_csv(BID / "bidirectional_summary.csv").fillna({"ccc_threshold": P.NO_FILTER})
    tr = pd.read_csv(BID / "bidirectional_transfer_results.csv").fillna({"ccc_threshold": P.NO_FILTER})
    models = list(P.MODEL_LABEL)
    ths = sorted(summ["ccc_threshold"].unique())
    x = np.arange(len(ths))
    offs = np.linspace(-0.21, 0.21, len(SERIES))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, m in zip(axes, models):
        s = summ[summ["model"] == m].sort_values("ccc_threshold")
        ax.axhline(0.5, color=P.AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.text(x[-1] + 0.45, 0.5, "chance", color=P.MUTED, fontsize=8,
                va="bottom", ha="right")
        ax.grid(axis="y", color=P.GRID, lw=0.6)
        ax.set_axisbelow(True)

        for (key, label, col, mk, kind), o in zip(SERIES, offs):
            xs = x + o
            if kind == "cv":
                mu = s[f"{key}_auc"].to_numpy()
                sd = s[f"{key}_std"].to_numpy()
                ax.errorbar(xs, mu, yerr=sd, color=col, lw=1.8, marker=mk, ms=5,
                            mec=P.SURFACE, mew=1.2, elinewidth=1.2, capsize=0,
                            label=f"{label} (mean ± SD)")
            else:
                src, tgt = key.split("_to_")
                t = tr[(tr["model"] == m) & (tr["source"] == src)].sort_values("ccc_threshold")
                a = t["transfer_auc"].to_numpy()
                ax.errorbar(xs, a, yerr=[a - t["ci_lo"], t["ci_hi"] - a], color=col,
                            lw=1.8, marker=mk, ms=5, mec=P.SURFACE, mew=1.2,
                            elinewidth=1.2, capsize=0, label=f"{label} (95% CI)")

        ax.set_title(P.MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xticks(x, P._xticklabels(s))
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_xlabel("Q-Q CCC threshold (features kept)")

    axes[0].set_ylabel("AUC (MCI vs normal)")
    axes[0].set_ylim(0.40, 0.75)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper left", ncol=4, fontsize=9,
               bbox_to_anchor=(0.005, 0.95), labelcolor=P.INK_2)
    fig.suptitle("Within-cohort baselines and bidirectional transfer, 90+ ⇄ I-CONECT",
                 x=0.01, ha="left", y=1.0, fontsize=12, color=P.INK)
    out = P.FIGS / "bidirectional_auc.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
