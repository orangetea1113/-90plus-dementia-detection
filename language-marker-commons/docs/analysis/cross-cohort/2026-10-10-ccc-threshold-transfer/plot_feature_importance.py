"""Figures for feature_importance.py (results/feature_importance/*.csv).

  fi_scatter_<subset>.png   imp_90 vs imp_ic per feature, one panel per model,
                            colored by family; top features labeled
  fi_family_<subset>.png    share of positive importance per family, for the
                            90+ model, I-CONECT model, n=156 90+ control and
                            90+ → IC transfer
  fi_weights_<model>_<subset>.png
                            per-feature weights side by side (EN signed coef,
                            SGBT permutation importance), union of top 15
  fi_agreement.png         90+ vs I-CONECT Spearman ρ against the small-n
                            control distribution, per model × view × subset

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/plot_feature_importance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import plot_style as P                 # noqa: E402  (sets rcParams)
from cohorts import subset_name        # noqa: E402
plt = P.plt

FI = P.RESULTS / "feature_importance"
FAMILIES = ["linguistic", "temporal", "semantic", "acoustic"]
FAM_COLOR = dict(zip(FAMILIES, P.SLOTS))
VIEW_LABEL = {"90plus": "90+ model", "iconect": "I-CONECT model",
              "ctrl_90plus_n156": "90+ subsample\n(n = I-CONECT)",
              "transfer_90_to_ic": "90+ model,\nscored on I-CONECT"}
N_LABEL = 6


def _subset(t) -> str:
    return subset_name(None if pd.isna(t) else t)


def scatter(pf: pd.DataFrame, subset: str) -> None:
    models = [m for m in P.MODEL_LABEL if m in pf["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 5.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        d = pf[pf["model"] == m]
        ax.axhline(0, color=P.AXIS, lw=1, zorder=0)
        ax.axvline(0, color=P.AXIS, lw=1, zorder=0)
        ax.grid(color=P.GRID, lw=0.6)
        ax.set_axisbelow(True)
        for f in FAMILIES:
            s = d[d["family"] == f]
            ax.scatter(s["imp_90"], s["imp_ic"], s=22, color=FAM_COLOR[f], alpha=0.8,
                       edgecolor=P.SURFACE, lw=0.6, label=f"{f} ({len(s)})")
        lab = pd.concat([d.nlargest(N_LABEL, "imp_90"), d.nlargest(N_LABEL, "imp_ic")])
        for _, r in lab.drop_duplicates("feature").iterrows():
            ax.annotate(r["feature"][:32], (r["imp_90"], r["imp_ic"]), fontsize=7,
                        color=P.INK_2, xytext=(4, 3), textcoords="offset points")
        ax.set_title(P.MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xlabel("90+ model: OOB permutation importance (ΔAUC)")
        ax.set_ylabel("I-CONECT model: OOB permutation importance (ΔAUC)")
    axes[0][0].legend(loc="upper right", fontsize=8, labelcolor=P.INK_2)
    fig.suptitle(f"Feature importance, 90+-trained vs I-CONECT-trained ({subset})",
                 x=0.01, ha="left", fontsize=12, color=P.INK)
    fig.tight_layout()
    _save(fig, f"fi_scatter_{subset}.png")


def family_bars(fs: pd.DataFrame, subset: str) -> None:
    models = [m for m in P.MODEL_LABEL if m in fs["model"].unique()]
    views = list(VIEW_LABEL)
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 3.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(views))[::-1]
    for ax, m in zip(axes[0], models):
        w = (fs[fs["model"] == m].pivot(index="view", columns="family", values="share")
             .reindex(index=views, columns=FAMILIES).fillna(0))
        left = np.zeros(len(views))
        for f in FAMILIES:
            ax.barh(y, w[f], left=left, height=0.62, color=FAM_COLOR[f],
                    edgecolor=P.SURFACE, lw=2, label=f)
            for yi, l, v in zip(y, left, w[f]):
                if v >= 0.08:
                    ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color=P.SURFACE)
            left += w[f].to_numpy()
        ax.set_yticks(y, [VIEW_LABEL[v] for v in views])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(P.MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xlabel("share of positive permutation importance")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(h, l, loc="upper left", ncol=4, fontsize=9, bbox_to_anchor=(0.005, 0.93),
               labelcolor=P.INK_2)
    fig.suptitle(f"Which feature families each model relies on ({subset})",
                 x=0.01, ha="left", y=1.0, fontsize=12, color=P.INK)
    _save(fig, f"fi_family_{subset}.png")


def weights(pf: pd.DataFrame, subset: str, k: int = 15) -> None:
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
            ax.axvline(0, color=P.AXIS, lw=1, zorder=0)
            ax.grid(axis="x", color=P.GRID, lw=0.6)
            ax.set_axisbelow(True)
            if kind == "coef":
                sel = d[f"coef_{who}_sel"].to_numpy() >= 0.5
                col = np.where(v > 0, P.SLOTS[1], P.SLOTS[0])
                for yi, vi, ci, si in zip(y, v, col, sel):
                    ax.barh(yi, vi, height=0.7, color=ci, alpha=1.0 if si else 0.3)
            else:
                ax.barh(y, v, height=0.7, color=P.SLOTS[3],
                        xerr=d[f"imp_{who}_sd"], error_kw={"ecolor": P.MUTED, "lw": 0.8})
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9)
        axes[0].set_yticks(y, [f"{f[:44]}  ·{fam[:3]}" for f, fam in zip(d["feature"], d["family"])],
                           fontsize=7.5)
        note = ("orange = higher in MCI, blue = lower; faded = selected in < 50% of bootstraps"
                if kind == "coef" else "error bars: SD over bootstraps")
        fig.suptitle(f"{P.MODEL_LABEL[m]} feature weights, 90+ vs I-CONECT ({subset}; "
                     f"union of each model's top {k})\n{note}",
                     x=0.01, ha="left", fontsize=11, color=P.INK)
        fig.tight_layout()
        _save(fig, f"fi_weights_{m}_{subset}.png")


def agreement(summ: pd.DataFrame, ctrl: pd.DataFrame) -> None:
    summ = summ.assign(subset=summ["ccc_threshold"].map(_subset))
    ctrl = ctrl.assign(subset=ctrl["ccc_threshold"].map(_subset))
    rows = list(summ[["subset", "model", "view"]].itertuples(index=False, name=None))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.4))
    ax.axvline(0, color=P.AXIS, lw=1, zorder=0)
    ax.grid(axis="x", color=P.GRID, lw=0.6)
    ax.set_axisbelow(True)
    for i, (sub, m, v) in enumerate(rows):
        yi = len(rows) - 1 - i
        c = ctrl[(ctrl["subset"] == sub) & (ctrl["model"] == m) & (ctrl["view"] == v)]
        ax.scatter(c["spearman"], np.full(len(c), yi), s=16, color=P.MUTED, alpha=0.6,
                   label="90+ subsample (n = I-CONECT) vs full 90+" if i == 0 else None)
        o = summ[(summ["subset"] == sub) & (summ["model"] == m) & (summ["view"] == v)].iloc[0]
        ax.scatter(o["spearman_90_vs_ic"], yi, s=60, marker="D", color=P.SLOTS[1],
                   edgecolor=P.SURFACE, lw=1.2, zorder=3,
                   label="I-CONECT vs full 90+" if i == 0 else None)
    ax.set_yticks(range(len(rows))[::-1],
                  [f"{s} · {P.MODEL_LABEL[m]} · {v}" for s, m, v in rows])
    ax.set_xlabel("Spearman ρ of feature importance with the full-90+ model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / (0.45 * len(rows) + 1.4)),
              ncol=2, fontsize=8, labelcolor=P.INK_2)
    ax.set_title("Is the I-CONECT model further from 90+ than a small 90+ sample is?",
                 loc="left", fontsize=11, color=P.INK)
    fig.tight_layout()
    _save(fig, "fi_agreement.png")


def _save(fig, name: str) -> None:
    P.FIGS.mkdir(parents=True, exist_ok=True)
    out = P.FIGS / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    pf = pd.read_csv(FI / "per_feature.csv")
    fs = pd.read_csv(FI / "family_share.csv")
    for t in pf["ccc_threshold"].unique():
        sub = _subset(t)
        same = (lambda df: df[df["ccc_threshold"].isna()] if pd.isna(t)
                else df[df["ccc_threshold"] == t])
        scatter(same(pf), sub)
        family_bars(same(fs), sub)
        weights(same(pf), sub)
    agreement(pd.read_csv(FI / "summary.csv"), pd.read_csv(FI / "ctrl_draws.csv"))


if __name__ == "__main__":
    main()
