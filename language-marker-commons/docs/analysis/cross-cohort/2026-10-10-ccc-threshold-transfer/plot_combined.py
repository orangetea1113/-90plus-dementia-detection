"""One-axis summary: SGBT / Elastic Net, 90+ CV AUC and 90+ → I-CONECT transfer AUC vs CCC threshold.

Reads results/bidirectional_summary.csv (written by bidirectional.py; 12 thresholds up to 0.995).

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/plot_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import plot_style as P                 # noqa: E402  (sets rcParams)
plt = P.plt

# (model, column, label, color, marker)
SERIES = [
    ("sgbt",       "90plus_to_iconect_auc", "SGBT Transfer (I-CONECT)",       "#2a78d6", "o"),
    ("elasticnet", "90plus_to_iconect_auc", "ElasticNet Transfer (I-CONECT)", "#eb6834", "o"),
    ("sgbt",       "90plus_cv_auc",  "SGBT CV (90+)",                  "#7fbbea", "o"),
    ("elasticnet", "90plus_cv_auc",  "ElasticNet CV (90+)",            "#f5a277", "o"),
]


def main() -> None:
    s = pd.read_csv(P.RESULTS / "bidirectional_summary.csv").fillna({"ccc_threshold": P.NO_FILTER})
    s = s.sort_values("ccc_threshold")
    t = s.drop_duplicates("ccc_threshold")
    x = np.arange(len(t))
    labels = [("All" if c == P.NO_FILTER else "≥" + (f"{c:.3f}"[:-1] if f"{c:.3f}".endswith("0") else f"{c:.3f}")) + f"\n({n})"
              for c, n in zip(t["ccc_threshold"], t["n_features"])]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.grid(axis="y", color=P.GRID, lw=0.6)
    ax.set_axisbelow(True)
    for m, col, label, color, mk in SERIES:
        d = s[s["model"] == m]
        ax.plot(x, d[col].to_numpy(), color=color, lw=2, marker=mk, ms=7,
                mec=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Q-Q CCC threshold (features kept)")
    ax.set_ylabel("AUC (MCI vs normal)")
    ax.set_ylim(0.5, 0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              fontsize=9, labelcolor=P.INK_2, handlelength=1.5)
    fig.tight_layout()
    out = P.FIGS / "auc_combined.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
