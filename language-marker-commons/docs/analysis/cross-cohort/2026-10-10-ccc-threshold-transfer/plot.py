"""Figures for the CCC-threshold transfer experiment.

Reads results/{cv_results,summary}.csv written by run.py and writes
results/figures/*.png.

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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

CV_COLOR = SLOTS[0]   # 90+ internal CV
TR_COLOR = SLOTS[1]   # transfer to I-CONECT
MODEL_LABEL = {"sgbt": "SGBT", "elasticnet": "Elastic Net"}
FAMILIES = [("n_ling", "Linguistic"), ("n_temp", "Temporal"),
            ("n_sem", "Semantic"), ("n_ac", "Acoustic")]

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


def plot_auc_vs_threshold(cv: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """One panel per model: 90+ CV AUC (seeds + mean±SD) vs I-CONECT transfer AUC (95% CI)."""
    thresholds = sorted(summary["ccc_threshold"].unique())
    x = np.arange(len(thresholds))
    xpos = dict(zip(thresholds, x))
    off = 0.12   # horizontal nudge so CV and transfer marks don't overlap

    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4.4), sharey=True)

    for ax, m in zip(axes, models):
        s  = summary[summary["model"] == m].sort_values("ccc_threshold")
        c  = cv[cv["model"] == m]
        xs = s["ccc_threshold"].map(xpos).to_numpy()

        ax.axhline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.text(x[-1] + 0.45, 0.5, "chance", color=MUTED, fontsize=8,
                va="bottom", ha="right")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

        # 90+ CV: individual seeds (faint) + mean ± SD band
        jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(c))
        ax.scatter(c["ccc_threshold"].map(xpos) - off + jitter, c["oof_auc"],
                   s=9, color=CV_COLOR, alpha=0.35, lw=0, zorder=2)
        mu, sd = s["cv_auc_mean"].to_numpy(), s["cv_auc_std"].to_numpy()
        ax.fill_between(xs - off, mu - sd, mu + sd, color=CV_COLOR, alpha=0.15,
                        lw=0, zorder=1)
        ax.plot(xs - off, mu, color=CV_COLOR, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.5, zorder=3, label="90+ CV (mean ± SD, 10 seeds)")

        # transfer: point + bootstrap 95% CI
        ta = s["transfer_auc"].to_numpy()
        ax.errorbar(xs + off, ta,
                    yerr=[ta - s["ci_lo"].to_numpy(), s["ci_hi"].to_numpy() - ta],
                    color=TR_COLOR, lw=2, marker="s", ms=5, mec=SURFACE, mew=1.5,
                    capsize=0, elinewidth=1.2, zorder=3,
                    label="→ I-CONECT transfer (95% bootstrap CI)")

        # direct labels on the end points only
        ax.annotate(f"{mu[-1]:.3f}", (xs[-1] - off, mu[-1]), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=8, color=INK_2)
        ax.annotate(f"{ta[-1]:.3f}", (xs[-1] + off, ta[-1]), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=INK_2)

        ax.set_title(MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xticks(x, _xticklabels(summary))
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_xlabel("Q-Q CCC threshold (features kept)")

    axes[0].set_ylabel("AUC (MCI vs normal)")
    axes[0].set_ylim(0.40, 0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles, labels, loc="upper left", ncol=2, fontsize=9,
               bbox_to_anchor=(0.005, 0.95), labelcolor=INK_2)
    fig.suptitle("Within-cohort CV vs cross-cohort transfer, by CCC threshold",
                 x=0.01, ha="left", y=1.0, fontsize=12, color=INK)

    out = FIGS / "auc_vs_ccc_threshold.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_composition(summary: pd.DataFrame) -> Path:
    """Stacked horizontal bars: how many features of each family survive each threshold."""
    t = summary.drop_duplicates("ccc_threshold").sort_values("ccc_threshold")
    y = np.arange(len(t))[::-1]   # lowest threshold on top

    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(t))
    for (col, label), color in zip(FAMILIES, SLOTS):
        w = t[col].to_numpy(dtype=float)
        ax.barh(y, w, left=left, height=0.62, color=color, label=label,
                edgecolor=SURFACE, linewidth=2)
        for yi, li, wi in zip(y, left, w):
            if wi >= 12:   # label only segments wide enough to hold the number
                ax.text(li + wi / 2, yi, f"{int(wi)}", ha="center", va="center",
                        fontsize=8, color=INK)
        left += w

    for yi, total in zip(y, left):
        ax.text(total + 5, yi, f"{int(total)}", va="center", fontsize=9, color=INK_2)

    ax.set_yticks(y, [_cond(c, "CCC ").replace("\n", " ") for c in t["ccc_threshold"]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Number of features")
    ax.set_xlim(0, left.max() * 1.08)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.16),
              fontsize=9, labelcolor=INK_2, handlelength=1.2)
    ax.set_title("Features kept at each CCC threshold, by family",
                 loc="left", y=1.18, fontsize=12)
    fig.tight_layout()

    out = FIGS / "feature_composition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_random_subset(rand: pd.DataFrame, rsum: pd.DataFrame) -> Path:
    """Rows = draw mode, cols = model: random-subset AUCs (null) vs the CCC subset."""
    thresholds = sorted(rsum["ccc_threshold"].unique())
    x = np.arange(len(thresholds))
    xpos = dict(zip(thresholds, x))
    models = list(MODEL_LABEL)
    modes = [("stratified", "Family-matched random subsets"),
             ("unstratified", "Size-matched random subsets (any family)")]

    fig, axes = plt.subplots(len(modes), len(models), figsize=(12, 7.6),
                             sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for (mode, mode_label), row in zip(modes, axes):
        for m, ax in zip(models, row):
            r = rand[(rand["mode"] == mode) & (rand["model"] == m) & (rand["draw"] >= 0)]
            s = rsum[(rsum["mode"] == mode) & (rsum["model"] == m)].sort_values("ccc_threshold")
            xs = s["ccc_threshold"].map(xpos).to_numpy()

            ax.axhline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)

            ax.scatter(r["ccc_threshold"].map(xpos) + rng.uniform(-0.18, 0.18, len(r)),
                       r["auc"], s=6, color=MUTED, alpha=0.35, lw=0, zorder=1,
                       label="Random subset (one dot per draw)")
            ax.vlines(xs, s["rand_lo"], s["rand_hi"], color=INK_2, lw=1.2, zorder=2,
                      label="Random 2.5–97.5th pct")
            ax.scatter(xs, s["ccc_auc"], marker="s", s=42, color=TR_COLOR,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label="CCC-selected subset")
            for xi, a, p in zip(xs, s["ccc_auc"], s["p_one_sided"]):
                ax.annotate(f"p={p:.2f}", (xi, a), xytext=(9, 0),
                            textcoords="offset points", va="center",
                            fontsize=7.5, color=INK_2)

            ax.set_title(f"{MODEL_LABEL[m]} · {mode_label}", loc="left",
                         fontweight="bold", fontsize=10)
            ax.set_xticks(x, [f"≥{t:.2f}\nn={n}" for t, n in
                              zip(s["ccc_threshold"], s["n_features"])])
            ax.set_xlim(x[0] - 0.5, x[-1] + 0.6)

    for ax in axes[:, 0]:
        ax.set_ylabel("I-CONECT transfer AUC")
    for ax in axes[-1]:
        ax.set_xlabel("Q-Q CCC threshold (features kept)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.005, 0.965), labelcolor=INK_2, markerscale=1.5)
    n = int(rsum["n_draws"].iloc[0])
    fig.suptitle(f"CCC-selected features vs {n} random subsets of the same size "
                 "(p = share of random draws ≥ CCC)", x=0.01, ha="left", y=1.0,
                 fontsize=12, color=INK)

    out = FIGS / "random_subset_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    cv      = pd.read_csv(RESULTS / "cv_results.csv").fillna({"ccc_threshold": NO_FILTER})
    summary = pd.read_csv(RESULTS / "summary.csv").fillna({"ccc_threshold": NO_FILTER})
    for out in (plot_auc_vs_threshold(cv, summary), plot_feature_composition(summary)):
        print(f"wrote {out}")
    rdir = RESULTS / "random_subset"
    if (rdir / "summary.csv").exists():
        out = plot_random_subset(pd.read_csv(rdir / "random_auc.csv"),
                                 pd.read_csv(rdir / "summary.csv"))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
