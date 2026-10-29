"""All figures for the CCC-threshold transfer experiment (results/figures/*.png).

Groups (each reads the csvs written by the matching experiment script and is skipped if they are missing):

  random      random_subset_control.png  <- validity.py random
  auc         auc_combined.png (CV + transfer AUC vs CCC threshold, one axis)   <- bidirectional.py
  bidir       bidirectional_auc.png (both directions + within-cohort CV)        <- bidirectional.py
  importance  fi_*.png                                            <- feature_importance.py

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/plot.py [group ...]   # default: all
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cohorts import subset_name  # noqa: E402

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

TR_COLOR = SLOTS[1]   # transfer to I-CONECT
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


def _thr(c: float) -> str:
    """0.50 … 0.99 with two decimals, 0.995 with three (so it is not shown as 0.99)."""
    return f"{c:.2f}" if round(c, 2) == c else f"{c:.3f}"


def _cond(c: float, prefix: str = "") -> str:
    return "All\n(no filter)" if c == NO_FILTER else f"{prefix}≥{_thr(c)}"


def _xticklabels(summary: pd.DataFrame) -> list[str]:
    t = summary.drop_duplicates("ccc_threshold").sort_values("ccc_threshold")
    return [f"{_cond(c)}\nn={n}".replace("(no filter)\n", "")
            for c, n in zip(t["ccc_threshold"], t["n_features"])]

# ── threshold / composition / random-subset figures ───────────────────────

def plot_random_subset(rand: pd.DataFrame, rsum: pd.DataFrame) -> Path:
    """One panel per model: random-subset AUCs (null) vs the CCC subset."""
    thresholds = sorted(rsum["ccc_threshold"].unique())
    x = np.arange(len(thresholds))
    xpos = dict(zip(thresholds, x))
    models = list(MODEL_LABEL)
    fig, axes = plt.subplots(1, len(models), figsize=(15, 4.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for m, ax in zip(models, axes):
        r = rand[(rand["model"] == m) & (rand["draw"] >= 0)]
        s = rsum[rsum["model"] == m].sort_values("ccc_threshold")
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
            ax.annotate(f"p={p:.3f}", (xi, a), xytext=(8, 0),
                        textcoords="offset points", va="center",
                        fontsize=7, color=INK_2)

        ax.set_title(MODEL_LABEL[m], loc="left", fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xticks(x, [f"≥{_thr(t)}\nn={n}" for t, n in
                          zip(s["ccc_threshold"], s["n_features"])])
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.6)

    axes[0].set_ylabel("I-CONECT transfer AUC")
    for ax in axes:
        ax.set_xlabel("Q-Q CCC threshold (features kept)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(handles, labels, loc="upper left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.005, 0.965), labelcolor=INK_2, markerscale=1.5)
    n = int(rsum["n_draws"].iloc[0])
    fig.suptitle(f"CCC-selected features vs {n} random subsets of the same size drawn from all features "
                 "(p = share of random draws ≥ CCC)", x=0.01, ha="left", y=1.0,
                 fontsize=12, color=INK)

    out = FIGS / "random_subset_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── AUC vs threshold, one axis ────────────────────────────────────────────

# (model, column, label, color, marker)
COMBINED_SERIES = [
    ("sgbt",       "90plus_to_iconect_auc", "SGBT Transfer (I-CONECT)",       "#2a78d6", "o"),
    ("elasticnet", "90plus_to_iconect_auc", "ElasticNet Transfer (I-CONECT)", "#eb6834", "o"),
    ("sgbt",       "90plus_cv_auc",  "SGBT CV (90+)",                  "#7fbbea", "o"),
    ("elasticnet", "90plus_cv_auc",  "ElasticNet CV (90+)",            "#f5a277", "o"),
]


def plot_auc_combined() -> None:
    s = pd.read_csv(RESULTS / "bidirectional_summary.csv").fillna({"ccc_threshold": NO_FILTER})
    s = s.sort_values("ccc_threshold")
    t = s.drop_duplicates("ccc_threshold")
    x = np.arange(len(t))
    labels = [("All" if c == NO_FILTER else "≥" + (f"{c:.3f}"[:-1] if f"{c:.3f}".endswith("0") else f"{c:.3f}")) + f"\n({n})"
              for c, n in zip(t["ccc_threshold"], t["n_features"])]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for m, col, label, color, mk in COMBINED_SERIES:
        d = s[s["model"] == m]
        ax.plot(x, d[col].to_numpy(), color=color, lw=2, marker=mk, ms=7,
                mec=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Q-Q CCC threshold (features kept)")
    ax.set_ylabel("AUC (MCI vs normal)")
    ax.set_ylim(0.5, 0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              fontsize=9, labelcolor=INK_2, handlelength=1.5)
    fig.tight_layout()
    out = FIGS / "auc_combined.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

# ── bidirectional ─────────────────────────────────────────────────────────

# (key in summary, label, color, marker, kind)
BID_SERIES = [
    ("90plus_cv",         "90+ CV",             SLOTS[0], "o", "cv"),
    ("iconect_cv",        "I-CONECT CV",        SLOTS[2], "o", "cv"),
    ("90plus_to_iconect", "90+ → I-CONECT",     SLOTS[1], "s", "tr"),
    ("iconect_to_90plus", "I-CONECT → 90+",     SLOTS[3], "s", "tr"),
]


def plot_bidirectional() -> None:
    summ = pd.read_csv(RESULTS / "bidirectional_summary.csv").fillna({"ccc_threshold": NO_FILTER})
    tr = pd.read_csv(RESULTS / "bidirectional_transfer_results.csv").fillna({"ccc_threshold": NO_FILTER})
    models = list(MODEL_LABEL)
    ths = sorted(summ["ccc_threshold"].unique())
    x = np.arange(len(ths))
    offs = np.linspace(-0.21, 0.21, len(BID_SERIES))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, m in zip(axes, models):
        s = summ[summ["model"] == m].sort_values("ccc_threshold")
        ax.axhline(0.5, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.text(x[-1] + 0.45, 0.5, "chance", color=MUTED, fontsize=8,
                va="bottom", ha="right")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

        for (key, label, col, mk, kind), o in zip(BID_SERIES, offs):
            xs = x + o
            if kind == "cv":
                mu = s[f"{key}_auc"].to_numpy()
                sd = s[f"{key}_std"].to_numpy()
                ax.errorbar(xs, mu, yerr=sd, color=col, lw=1.8, marker=mk, ms=5,
                            mec=SURFACE, mew=1.2, elinewidth=1.2, capsize=0,
                            label=f"{label} (mean ± SD)")
            else:
                src, tgt = key.split("_to_")
                t = tr[(tr["model"] == m) & (tr["source"] == src)].sort_values("ccc_threshold")
                a = t["transfer_auc"].to_numpy()
                ax.errorbar(xs, a, yerr=[a - t["ci_lo"], t["ci_hi"] - a], color=col,
                            lw=1.8, marker=mk, ms=5, mec=SURFACE, mew=1.2,
                            elinewidth=1.2, capsize=0, label=f"{label} (95% CI)")

        ax.set_title(MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xticks(x, _xticklabels(s))
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_xlabel("Q-Q CCC threshold (features kept)")

    axes[0].set_ylabel("AUC (MCI vs normal)")
    axes[0].set_ylim(0.40, 0.75)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper left", ncol=4, fontsize=9,
               bbox_to_anchor=(0.005, 0.95), labelcolor=INK_2)
    fig.suptitle("Within-cohort baselines and bidirectional transfer, 90+ ⇄ I-CONECT",
                 x=0.01, ha="left", y=1.0, fontsize=12, color=INK)
    out = FIGS / "bidirectional_auc.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

# ── feature importance ────────────────────────────────────────────────────

FI = RESULTS / "feature_importance"
FI_FAMILIES = ["linguistic", "temporal", "semantic", "acoustic"]
FAM_COLOR = dict(zip(FI_FAMILIES, SLOTS))
VIEW_LABEL = {"90plus": "90+ model", "iconect": "I-CONECT model",
              "ctrl_90plus_n156": "90+ subsample\n(n = I-CONECT)",
              "transfer_90_to_ic": "90+ model,\nscored on I-CONECT"}
N_LABEL = 6


def _subset(t) -> str:
    return subset_name(None if pd.isna(t) else t)


def fi_scatter(pf: pd.DataFrame, subset: str) -> None:
    models = [m for m in MODEL_LABEL if m in pf["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        d = pf[pf["model"] == m]
        ax.axhline(0, color=AXIS, lw=1, zorder=0)
        ax.axvline(0, color=AXIS, lw=1, zorder=0)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for f in FI_FAMILIES:
            s = d[d["family"] == f]
            ax.scatter(s["imp_90"], s["imp_ic"], s=22, color=FAM_COLOR[f], alpha=0.8,
                       edgecolor=SURFACE, lw=0.6, label=f"{f} ({len(s)})")
        lab = pd.concat([d.nlargest(N_LABEL, "imp_90"), d.nlargest(N_LABEL, "imp_ic")])
        for _, r in lab.drop_duplicates("feature").iterrows():
            ax.annotate(r["feature"][:32], (r["imp_90"], r["imp_ic"]), fontsize=7,
                        color=INK_2, xytext=(4, 3), textcoords="offset points")
        ax.set_title(MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xlabel("90+ model: OOB permutation importance (ΔAUC)")
        ax.set_ylabel("I-CONECT model: OOB permutation importance (ΔAUC)")
    axes[0][0].legend(loc="upper right", fontsize=8, labelcolor=INK_2)
    fig.suptitle(f"Feature importance, 90+-trained vs I-CONECT-trained ({subset})",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, f"fi_scatter_{subset}.png")


def fi_family_bars(fs: pd.DataFrame, subset: str) -> None:
    models = [m for m in MODEL_LABEL if m in fs["model"].unique()]
    views = list(VIEW_LABEL)
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 3.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(views))[::-1]
    for ax, m in zip(axes[0], models):
        w = (fs[fs["model"] == m].pivot(index="view", columns="family", values="share")
             .reindex(index=views, columns=FI_FAMILIES).fillna(0))
        left = np.zeros(len(views))
        for f in FI_FAMILIES:
            ax.barh(y, w[f], left=left, height=0.62, color=FAM_COLOR[f],
                    edgecolor=SURFACE, lw=2, label=f)
            for yi, l, v in zip(y, left, w[f]):
                if v >= 0.08:
                    ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color=SURFACE)
            left += w[f].to_numpy()
        ax.set_yticks(y, [VIEW_LABEL[v] for v in views])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xlabel("share of positive permutation importance")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(h, l, loc="upper left", ncol=4, fontsize=9, bbox_to_anchor=(0.005, 0.93),
               labelcolor=INK_2)
    fig.suptitle(f"Which feature families each model relies on ({subset})",
                 x=0.01, ha="left", y=1.0, fontsize=12, color=INK)
    _save(fig, f"fi_family_{subset}.png")


def fi_weights(pf: pd.DataFrame, subset: str, k: int = 15) -> None:
    """Side-by-side weights of the 90+ and I-CONECT models, union of each one's top k.

    Elastic Net: signed bootstrap-mean coefficient (orange = higher → MCI), faded
    when selected in < 50% of bootstraps. SGBT: OOB permutation importance ± SD.
    """
    specs = {"elasticnet": ("coef", "Elastic Net coefficient (per SD, bootstrap mean)"),
             "sgbt": ("imp", "SGBT OOB permutation importance (ΔAUC, mean ± SD)")}
    for m, (kind, xlabel) in specs.items():
        d = pf[pf["model"] == m]
        if d.empty:
            continue
        key = (lambda c: d[c].abs()) if kind == "coef" else (lambda c: d[c])
        top = set(d.loc[key(f"{kind}_90").nlargest(k).index, "feature"]) | \
              set(d.loc[key(f"{kind}_ic").nlargest(k).index, "feature"])
        d = d[d["feature"].isin(top)].sort_values(f"{kind}_90", key=abs if kind == "coef" else None)
        y = np.arange(len(d))
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.27 * len(d) + 1.6), sharey=True, sharex=True)
        for ax, (who, title) in zip(axes, (("90", "Trained on 90+ (n=868)"),
                                           ("ic", "Trained on I-CONECT (n=156)"))):
            v = d[f"{kind}_{who}"].to_numpy()
            ax.axvline(0, color=AXIS, lw=1, zorder=0)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if kind == "coef":
                sel = d[f"coef_{who}_sel"].to_numpy() >= 0.5
                col = np.where(v > 0, SLOTS[1], SLOTS[0])
                for yi, vi, ci, si in zip(y, v, col, sel):
                    ax.barh(yi, vi, height=0.7, color=ci, alpha=1.0 if si else 0.3)
            else:
                ax.barh(y, v, height=0.7, color=SLOTS[3],
                        xerr=d[f"imp_{who}_sd"], error_kw={"ecolor": MUTED, "lw": 0.8})
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9)
        axes[0].set_yticks(y, [f"{f[:44]}  ·{fam[:3]}" for f, fam in zip(d["feature"], d["family"])],
                           fontsize=7.5)
        note = ("orange = higher in MCI, blue = lower; faded = selected in < 50% of bootstraps"
                if kind == "coef" else "error bars: SD over bootstraps")
        fig.suptitle(f"{MODEL_LABEL[m]} feature weights, 90+ vs I-CONECT ({subset}; "
                     f"union of each model's top {k})\n{note}",
                     x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout()
        _save(fig, f"fi_weights_{m}_{subset}.png")


def fi_agreement(summ: pd.DataFrame, ctrl: pd.DataFrame) -> None:
    summ = summ.assign(subset=summ["ccc_threshold"].map(_subset))
    ctrl = ctrl.assign(subset=ctrl["ccc_threshold"].map(_subset))
    rows = list(summ[["subset", "model", "view"]].itertuples(index=False, name=None))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.4))
    ax.axvline(0, color=AXIS, lw=1, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for i, (sub, m, v) in enumerate(rows):
        yi = len(rows) - 1 - i
        c = ctrl[(ctrl["subset"] == sub) & (ctrl["model"] == m) & (ctrl["view"] == v)]
        ax.scatter(c["spearman"], np.full(len(c), yi), s=16, color=MUTED, alpha=0.6,
                   label="90+ subsample (n = I-CONECT) vs full 90+" if i == 0 else None)
        o = summ[(summ["subset"] == sub) & (summ["model"] == m) & (summ["view"] == v)].iloc[0]
        ax.scatter(o["spearman_90_vs_ic"], yi, s=60, marker="D", color=SLOTS[1],
                   edgecolor=SURFACE, lw=1.2, zorder=3,
                   label="I-CONECT vs full 90+" if i == 0 else None)
    ax.set_yticks(range(len(rows))[::-1],
                  [f"{s} · {MODEL_LABEL[m]} · {v}" for s, m, v in rows])
    ax.set_xlabel("Spearman ρ of feature importance with the full-90+ model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / (0.45 * len(rows) + 1.4)),
              ncol=2, fontsize=8, labelcolor=INK_2)
    ax.set_title("Is the I-CONECT model further from 90+ than a small 90+ sample is?",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "fi_agreement.png")


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_feature_importance() -> None:
    pf = pd.read_csv(FI / "per_feature.csv")
    fs = pd.read_csv(FI / "family_share.csv")
    for t in pf["ccc_threshold"].unique():
        sub = _subset(t)
        same = (lambda df: df[df["ccc_threshold"].isna()] if pd.isna(t)
                else df[df["ccc_threshold"] == t])
        fi_scatter(same(pf), sub)
        fi_family_bars(same(fs), sub)
        fi_weights(same(pf), sub)
    fi_agreement(pd.read_csv(FI / "summary.csv"), pd.read_csv(FI / "ctrl_draws.csv"))

def plot_random() -> None:
    out = plot_random_subset(pd.read_csv(RESULTS / "random_subset_auc.csv"),
                             pd.read_csv(RESULTS / "random_subset_summary.csv"))
    print(f"wrote {out}")


GROUPS = {
    "random":     (plot_random, RESULTS / "random_subset_auc.csv"),
    "auc":        (plot_auc_combined, RESULTS / "bidirectional_summary.csv"),
    "bidir":      (plot_bidirectional, RESULTS / "bidirectional_summary.csv"),
    "importance": (plot_feature_importance, RESULTS / "feature_importance" / "per_feature.csv"),
}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for g in sys.argv[1:] or GROUPS:
        fn, needs = GROUPS[g]
        if needs.exists():
            fn()
        else:
            print(f"skip {g}: {needs} not found")


if __name__ == "__main__":
    main()
