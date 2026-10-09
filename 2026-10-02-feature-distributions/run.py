"""Full cross-cohort distribution analysis: 90+ vs I-CONECT, 547 features.

Feature families (four modalities)
----------------------------------
  ling       (99)   recording_features.parquet  (LIWC, syntactic, lexical, response length)
  temporal   (24)   recording_temporal_*.parquet  (pt_* participant timing, ix_* interaction)
  acoustic  (420)   recording_acoustic_*.parquet  (eGeMAPS 88 base + MFCC 52 base,
                                                  x 3 stats _whole/_mean/_std)
  semantic    (4)   recording_semantic_*.parquet
  Total: 547

  Acoustic note: each base feature has three stat variants (_whole / _mean / _std);
  they are grouped into one 3-column figure per base feature (trio plot).
  I-CONECT acoustic coverage: ~86/156 subjects (54%), so the acoustic non-NaN
  threshold is set separately via --min-nonnan-ac (default 0.5).

Outputs
-------
  results/all_features_stats.csv        - per-feature stats (Cohen's d, KS) for all
                                          547 features
  results/domain_auc.csv                - per-family transfer-risk scalars
  results/plots/
    ling/        {slug}.png             - KDE plots for ling features (99 files)
    temporal/    {slug}.png             - KDE plots for temporal features (24 files)
    semantic/    {slug}.png             - KDE plots for semantic features (4 files)
    acoustic/    {slug}.png             - trio plots (_whole/_mean/_std) per acoustic
                                          base feature (140 files)
    family/      summary_all_features.png - 4-panel summary over all 547 features
                 summary_{family}.png   - 3-panel summary per family (4 files)
                 _overview_by_family.png - |d| strip per family
                 transfer_risk.png      - family x risk-metric heatmap
                 ic_subgroups.png       - I-CONECT long vs short conversations vs 90+
  results/normalized/                   - per-cohort robust z-score (median / IQR):
    feature_shape_stats.csv             - per feature: Q-Q CCC on 5-95th pct (primary), KS (robust z),
                                          Q-Q Pearson r and slope, raw SD ratio, raw d/KS
    plots/<family>/{slug}.png           - robust-z densities + Q-Q plot, one per feature

Usage
-----
# Local (laptop/workstation):
    python run.py --data-dir /path/to/dataset

  Expects:
    <data-dir>/i-conect/recording_features.parquet
    <data-dir>/i-conect/recording_temporal_sam_v0_ds2.parquet
    <data-dir>/i-conect/recording_semantic_sam_v0_ds2.parquet
    <data-dir>/i-conect/recording_acoustic_sam_v0_ds2.parquet
    <data-dir>/90plus/recording_features.parquet
    <data-dir>/90plus/recording_temporal_whisper-lv3-pyannote.parquet
    <data-dir>/90plus/recording_semantic_whisper-lv3-pyannote.parquet
    <data-dir>/90plus/recording_acoustic_whisper-lv3-pyannote.parquet
    <data-dir>/90plus/role_map.parquet

# Great Lakes (uses lmc.io + config/paths.yaml):
    python run.py

Options:
    --data-dir PATH        local data directory (see above)
    --min-nonnan FLOAT     fraction non-NaN subjects required per cohort for
                           ling/temporal/semantic features (default 0.8)
    --min-nonnan-ac FLOAT  same threshold for acoustic features (default 0.5)
    --no-plots             skip all PNGs, output CSV only
    --family-only [FAM..]  only redraw family-level plots from the existing stats CSV
    --normalized-only      only redo the robust z-score shape analysis (results/normalized/)
    --paths FILE           path to paths.yaml (GL mode only, default config/paths.yaml)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
PLOTS_DIR  = HERE / "results" / "plots"

def _feat_subdir(family: str) -> Path:
    """Map a feature family string to its output subdirectory."""
    prefix = family.split(":")[0]          # 'ling', 'temporal', 'semantic', 'acoustic'
    return PLOTS_DIR / prefix

PLOTS_FAMILY  = PLOTS_DIR / "family"

IC_COLOR, NP_COLOR = "#4477AA", "#EE7733"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})

KEY = ["dataset", "subject_id", "visit_id", "transcript_id"]

META_COLS = {
    "subject_id", "visit_id", "transcript_id", "recording_id",
    "asr_config", "dataset", "transcript_revision",
}


# ── family lookup ─────────────────────────────────────────────────────────────
FAMILIES = ["ling", "temporal", "acoustic", "semantic"]


def family_of(feat: str, ling_cols: set, temp_cols: set, sem_cols: set, ac_cols: set = frozenset()) -> str:
    """Map a feature to one of the four modality families in FAMILIES."""
    if feat in ac_cols or feat.startswith(("egemaps_", "mfcc_")):
        return "acoustic"
    if feat in sem_cols or feat.startswith("sm_"):
        return "semantic"
    if feat in temp_cols or feat.startswith(("pt_", "ix_")):
        return "temporal"
    return "ling"


def slugify(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()[:60]


# ── data loading ──────────────────────────────────────────────────────────────
def load_local(data_dir: Path):
    """Load from a local dataset/ directory. Returns (ic_subj, np_subj, feat_cols, family_sets)."""
    data_dir = Path(data_dir)

    # ── I-CONECT ──────────────────────────────────────────────────────────────
    ic_ling = pd.read_parquet(data_dir / "i-conect" / "recording_features.parquet")
    ic_ling = ic_ling[ic_ling["asr_config"] == "sam_v0_ds2"]

    ic_temp = pd.read_parquet(data_dir / "i-conect" / "recording_temporal_sam_v0_ds2.parquet")

    ic_sem_path = data_dir / "i-conect" / "recording_semantic_sam_v0_ds2.parquet"
    ic_sem = pd.read_parquet(ic_sem_path) if ic_sem_path.exists() else None

    # ── 90+ (role_usable filter) ──────────────────────────────────────────────
    np_ling = pd.read_parquet(data_dir / "90plus" / "recording_features.parquet")
    np_ling = np_ling[np_ling["asr_config"] == "whisper-lv3-pyannote"]
    rm = pd.read_parquet(data_dir / "90plus" / "role_map.parquet")
    usable = set(rm.loc[rm["role_usable"], "transcript_id"])
    np_ling = np_ling[np_ling["transcript_id"].isin(usable)]

    np_temp = pd.read_parquet(
        data_dir / "90plus" / "recording_temporal_whisper-lv3-pyannote.parquet"
    )
    np_temp = np_temp[np_temp["transcript_id"].isin(usable)]

    np_sem_path = data_dir / "90plus" / "recording_semantic_whisper-lv3-pyannote.parquet"
    np_sem = pd.read_parquet(np_sem_path) if np_sem_path.exists() else None
    if np_sem is not None:
        np_sem = np_sem[np_sem["transcript_id"].isin(usable)]

    # ── feature column lists ──────────────────────────────────────────────────
    ling_cols = [c for c in ic_ling.columns if c not in META_COLS]
    temp_cols = [c for c in ic_temp.columns if c not in META_COLS]
    sem_cols  = [c for c in ic_sem.columns  if c not in META_COLS] if ic_sem is not None else []

    ic_ac_path = data_dir / "i-conect" / "recording_acoustic_sam_v0_ds2.parquet"
    np_ac_path = data_dir / "90plus" / "recording_acoustic_whisper-lv3-pyannote.parquet"
    ic_ac = pd.read_parquet(ic_ac_path) if ic_ac_path.exists() else None
    np_ac = pd.read_parquet(np_ac_path) if np_ac_path.exists() else None
    if np_ac is not None:
        np_ac = np_ac[np_ac["transcript_id"].isin(usable)]
    # use intersection so only features present in BOTH cohorts are included
    if ic_ac is not None and np_ac is not None:
        ic_ac_feats = [c for c in ic_ac.columns if c not in META_COLS]
        np_ac_feats = set(c for c in np_ac.columns if c not in META_COLS)
        ac_cols = [c for c in ic_ac_feats if c in np_ac_feats]
    else:
        ac_cols = []

    all_feats = ling_cols + temp_cols + sem_cols + ac_cols

    # ── merge ling/temp/sem at transcript level, aggregate to subject ────────
    non_ac_feats = ling_cols + temp_cols + sem_cols

    merge_keys_t = [k for k in KEY if k in ic_temp.columns]
    ic = ic_ling.merge(ic_temp[merge_keys_t + temp_cols], on=merge_keys_t, how="inner")
    if ic_sem is not None:
        merge_keys_s = [k for k in KEY if k in ic_sem.columns]
        ic = ic.merge(ic_sem[merge_keys_s + sem_cols], on=merge_keys_s, how="left")
    ic_subj = ic.groupby("subject_id")[non_ac_feats].mean()

    # acoustic: aggregate directly at subject level, then outer-join to ic_subj
    if ic_ac is not None and ac_cols:
        ic_ac_subj = ic_ac.groupby("subject_id")[ac_cols].mean()
        ic_subj = ic_subj.join(ic_ac_subj, how="left")

    merge_keys_t = [k for k in KEY if k in np_temp.columns]
    np_ = np_ling.merge(np_temp[merge_keys_t + temp_cols], on=merge_keys_t, how="inner")
    if np_sem is not None:
        merge_keys_s = [k for k in KEY if k in np_sem.columns]
        np_ = np_.merge(np_sem[merge_keys_s + sem_cols], on=merge_keys_s, how="left")
    np_subj = np_.groupby("subject_id")[non_ac_feats].mean()

    if np_ac is not None and ac_cols:
        np_ac_subj = np_ac.groupby("subject_id")[ac_cols].mean()
        np_subj = np_subj.join(np_ac_subj, how="left")

    family_sets = {
        "ling": set(ling_cols),
        "temp": set(temp_cols),
        "sem":  set(sem_cols),
        "ac":   set(ac_cols),
    }
    return ic_subj, np_subj, all_feats, family_sets


def load_gl(paths_yaml: str):
    """Load from Great Lakes via lmc.io. Returns same tuple as load_local."""
    _REPO = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(_REPO))
    from lmc import io  # noqa: E402
    from lmc.features.linguistic.iconect99 import FEATURE_NAMES as LING, FEATURE_BLOCKS  # noqa
    from lmc.features.temporal.names import TEMPORAL_FEATURE_NAMES as TEMP  # noqa
    try:
        from lmc.features.linguistic.highsignal import HIGHSIGNAL_FEATURE_NAMES as HS
    except ImportError:
        HS = []

    paths = io.load_paths(paths_yaml)
    data_dir = Path(paths["local_data_dir"])

    ling_cols = list(LING)
    temp_cols = list(TEMP)
    sem_cols  = []

    all_feats = ling_cols + temp_cols

    ic_f = io.read_table("iconect", "recording_features", paths)
    ic_f = ic_f[ic_f["asr_config"] == "sam_v0_ds2"]
    ic_t = pd.read_parquet(data_dir / "processed" / "iconect" / "recording_temporal_sam_v0_ds2.parquet")
    ic   = ic_f.merge(ic_t[[k for k in KEY if k in ic_t] + temp_cols], on=[k for k in KEY if k in ic_t], how="inner")
    ic_subj = ic.groupby("subject_id")[all_feats].mean()

    np_f = pd.read_parquet(data_dir / "processed" / "the90plus" / "recording_features.parquet")
    np_f = np_f[np_f["asr_config"] == "whisper-lv3-pyannote"]
    rm   = pd.read_parquet(data_dir / "processed" / "the90plus" / "role_map.parquet")
    np_f = np_f[np_f["transcript_id"].isin(set(rm.loc[rm["role_usable"], "transcript_id"]))]
    np_t = pd.read_parquet(data_dir / "processed" / "the90plus" / "recording_temporal_whisper-lv3-pyannote.parquet")
    np_  = np_f.merge(np_t[[k for k in KEY if k in np_t] + temp_cols], on=[k for k in KEY if k in np_t], how="inner")
    np_subj = np_.groupby("subject_id")[all_feats].mean()

    family_sets = {"ling": set(ling_cols), "temp": set(temp_cols), "sem": set(sem_cols)}
    return ic_subj, np_subj, all_feats, family_sets


# ── statistics ────────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / max(n_a + n_b - 2, 1)
    pooled_sd = np.sqrt(pooled_var)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")


def feature_stats(feat: str, ic_subj: pd.DataFrame, np_subj: pd.DataFrame,
                  family_sets: dict) -> dict | None:
    ic_v = ic_subj[feat].dropna().values
    np_v = np_subj[feat].dropna().values
    if len(ic_v) < 5 or len(np_v) < 5:
        return None

    ks_stat, ks_p = stats.ks_2samp(ic_v, np_v)
    cd   = cohens_d(ic_v, np_v)
    abs_d = abs(cd) if not np.isnan(cd) else float("nan")

    def iqr(v): return float(np.percentile(v, 75) - np.percentile(v, 25))

    fam = family_of(feat, family_sets["ling"], family_sets["temp"], family_sets["sem"], family_sets.get("ac", frozenset()))
    shift = (
        "negligible" if abs_d < 0.2 else
        "small"      if abs_d < 0.5 else
        "medium"     if abs_d < 0.8 else
        "large"
    ) if not np.isnan(abs_d) else "unknown"

    return dict(
        feature=feat, family=fam,
        n_iconect=int(len(ic_v)), ic_mean=float(ic_v.mean()),
        ic_median=float(np.median(ic_v)), ic_iqr=iqr(ic_v), ic_std=float(ic_v.std(ddof=1)),
        n_90plus=int(len(np_v)),  np_mean=float(np_v.mean()),
        np_median=float(np.median(np_v)), np_iqr=iqr(np_v), np_std=float(np_v.std(ddof=1)),
        cohens_d=cd, abs_d=abs_d,
        ks_stat=float(ks_stat), ks_pval=float(ks_p),
        shift_category=shift,
    )


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_feature(feat: str, s: dict, ic_v: np.ndarray, np_v: np.ndarray,
                 out_dir: Path | None = None) -> None:
    slug = slugify(feat)
    out  = (out_dir or PLOTS_DIR) / f"{slug}.png"

    xmin, xmax = min(ic_v.min(), np_v.min()), max(ic_v.max(), np_v.max())
    pad   = (xmax - xmin) * 0.10
    xgrid = np.linspace(xmin - pad, xmax + pad, 500)

    try:
        kde_ic = gaussian_kde(ic_v, bw_method="scott")(xgrid)
        kde_np = gaussian_kde(np_v, bw_method="scott")(xgrid)
        has_kde = True
    except Exception:
        has_kde = False

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6.5),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
        sharex=True,
    )

    # ── top: KDE overlay + rug + median lines ─────────────────────────────────
    if has_kde:
        ax_top.fill_between(xgrid, kde_ic, alpha=0.20, color=IC_COLOR)
        ax_top.plot(xgrid, kde_ic, color=IC_COLOR, lw=2)
        ax_top.fill_between(xgrid, kde_np, alpha=0.20, color=NP_COLOR)
        ax_top.plot(xgrid, kde_np, color=NP_COLOR, lw=2)
    else:
        ax_top.hist(ic_v, bins=25, density=True, color=IC_COLOR, alpha=0.3)
        ax_top.hist(np_v, bins=25, density=True, color=NP_COLOR, alpha=0.3)

    ax_top.axvline(s["ic_median"], color=IC_COLOR, lw=1.4, ls="--", alpha=0.85)
    ax_top.axvline(s["np_median"], color=NP_COLOR, lw=1.4, ls="--", alpha=0.85)

    rng = np.random.default_rng(0)
    rug_y = ax_top.get_ylim()[1] * 0.015 if has_kde else 0
    ax_top.plot(ic_v, rng.uniform(0, rug_y + 1e-9, len(ic_v)), "|",
                color=IC_COLOR, alpha=0.15, ms=5)
    ax_top.plot(np_v, rng.uniform(0, rug_y + 1e-9, len(np_v)), "|",
                color=NP_COLOR, alpha=0.08, ms=5)

    sig = "p<0.001" if s["ks_pval"] < 0.001 else f"p={s['ks_pval']:.3f}"
    annot = (
        f"Cohen's d = {s['cohens_d']:+.3f}  ({s['shift_category']})\n"
        f"KS = {s['ks_stat']:.3f},  {sig}"
    )
    ax_top.text(0.98, 0.97, annot, transform=ax_top.transAxes, fontsize=8.2,
                va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=GRID, alpha=0.92))
    ax_top.set_ylabel("Density")
    ax_top.set_title(
        f"Cross-cohort: {feat[:65]}\n[{s['family']}]",
        loc="left", weight="bold", fontsize=9,
    )
    ax_top.legend(
        handles=[
            Patch(color=IC_COLOR, alpha=0.6, label=f"I-CONECT (n={s['n_iconect']})"),
            Patch(color=NP_COLOR, alpha=0.6, label=f"90+ (n={s['n_90plus']})"),
        ],
        frameon=False, loc="upper left", fontsize=9,
    )
    ax_top.grid(axis="x", color=GRID, lw=0.5)
    ax_top.set_axisbelow(True)
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(bottom=False)

    # ── bottom: Δ density = KDE_IC − KDE_90+ ─────────────────────────────────
    if has_kde:
        delta = kde_ic - kde_np
        pos_mask = delta >= 0
        ax_bot.fill_between(xgrid, delta, where=pos_mask,
                            color=IC_COLOR, alpha=0.35, label="IC > 90+")
        ax_bot.fill_between(xgrid, delta, where=~pos_mask,
                            color=NP_COLOR, alpha=0.35, label="90+ > IC")
        ax_bot.plot(xgrid, delta, color=INK, lw=1.0, alpha=0.6)
        ax_bot.axhline(0, color=MUTED, lw=0.8, ls="-")
        ax_bot.legend(frameon=False, fontsize=8, loc="upper right",
                      handlelength=1.2, handletextpad=0.5)
    else:
        ax_bot.text(0.5, 0.5, "KDE unavailable", transform=ax_bot.transAxes,
                    ha="center", va="center", color=MUTED)

    ax_bot.set_ylabel("Δ density\n(IC − 90+)", fontsize=8.5)
    ax_bot.set_xlabel(feat, labelpad=5)
    ax_bot.grid(axis="x", color=GRID, lw=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.3f}" if v != 0 else "0")
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



# ── acoustic trio plot (whole / mean / std in one figure) ─────────────────────
def acoustic_base(feat: str) -> str:
    """Strip the trailing _whole/_mean/_std stat suffix."""
    return re.sub(r"_(whole|mean|std)$", "", feat)


def plot_acoustic_trio(
    base: str,
    stat_rows: dict,           # {"whole": s_dict, "mean": s_dict, "std": s_dict}
    stat_vals: dict,           # {"whole": (ic_v, np_v), ...}
    out_dir: Path | None = None,
) -> None:
    """Three-column two-panel figure for one acoustic base feature."""
    slug = slugify(base)
    out  = (out_dir or PLOTS_DIR) / f"{slug}.png"

    stats_order = ["whole", "mean", "std"]
    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 5.5),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08, "wspace": 0.32},
    )
    # link x-axes within each column manually
    for col in range(3):
        axes[1, col].sharex(axes[0, col])

    for col, stat in enumerate(stats_order):
        s    = stat_rows.get(stat)
        vals = stat_vals.get(stat)
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if s is None or vals is None:
            ax_top.set_visible(False)
            ax_bot.set_visible(False)
            continue

        ic_v, np_v = vals
        xmin = min(ic_v.min(), np_v.min())
        xmax = max(ic_v.max(), np_v.max())
        pad   = (xmax - xmin) * 0.10 or 0.5
        xgrid = np.linspace(xmin - pad, xmax + pad, 400)

        try:
            kde_ic  = gaussian_kde(ic_v, bw_method="scott")(xgrid)
            kde_np  = gaussian_kde(np_v, bw_method="scott")(xgrid)
            has_kde = True
        except Exception:
            has_kde = False

        # top panel
        if has_kde:
            ax_top.fill_between(xgrid, kde_ic, alpha=0.18, color=IC_COLOR)
            ax_top.plot(xgrid, kde_ic, color=IC_COLOR, lw=1.6,
                        label=f"I-CONECT (n={s['n_iconect']})")
            ax_top.fill_between(xgrid, kde_np, alpha=0.18, color=NP_COLOR)
            ax_top.plot(xgrid, kde_np, color=NP_COLOR, lw=1.6,
                        label=f"90+ (n={s['n_90plus']})")

        ylim = ax_top.get_ylim()
        ymax = ylim[1] if has_kde else 1.0
        ax_top.axvline(float(np.median(ic_v)), color=IC_COLOR, lw=1.1, ls="--", alpha=0.75)
        ax_top.axvline(float(np.median(np_v)), color=NP_COLOR, lw=1.1, ls="--", alpha=0.75)
        ax_top.eventplot(ic_v, orientation="horizontal",
                         lineoffsets=-ymax * 0.06, linelengths=ymax * 0.08,
                         linewidths=0.6, color=IC_COLOR, alpha=0.4)
        ax_top.eventplot(np_v, orientation="horizontal",
                         lineoffsets=-ymax * 0.12, linelengths=ymax * 0.08,
                         linewidths=0.6, color=NP_COLOR, alpha=0.3)

        stats_txt = (
            f"d={s['cohens_d']:+.2f}  KS={s['ks_stat']:.2f}"
        )
        ax_top.text(0.97, 0.97, stats_txt, transform=ax_top.transAxes,
                    ha="right", va="top", fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.85))
        ax_top.set_title(f"_{stat}", fontsize=9, weight="bold", pad=3)
        ax_top.set_ylabel("density" if col == 0 else "", fontsize=8)
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(color=GRID, lw=0.4)
        ax_top.set_axisbelow(True)
        if col == 0:
            ax_top.legend(fontsize=7.5, frameon=False, loc="upper left")

        # bottom panel (delta density)
        if has_kde:
            delta    = kde_ic - kde_np
            pos_mask = delta >= 0
            ax_bot.fill_between(xgrid, delta, where=pos_mask,
                                color=IC_COLOR, alpha=0.35, label="IC > 90+")
            ax_bot.fill_between(xgrid, delta, where=~pos_mask,
                                color=NP_COLOR, alpha=0.35, label="90+ > IC")
            ax_bot.axhline(0, color=INK, lw=0.6)
        ax_bot.set_ylabel("Δ density" if col == 0 else "", fontsize=7.5)
        ax_bot.set_xlabel("value", fontsize=8)
        ax_bot.grid(color=GRID, lw=0.4)
        ax_bot.set_axisbelow(True)

    short = base[:80] + ("…" if len(base) > 80 else "")
    fig.suptitle(short, fontsize=9, weight="bold", x=0.02, ha="left")
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ── family overview plots ─────────────────────────────────────────────────────
_FAM_ORDER = FAMILIES
_FAM_COLOR = {
    "ling":     "#4477AA",
    "temporal": "#EE7733",
    "acoustic": "#AA3377",
    "semantic": "#228833",
}
_SHIFT_COLOR = {
    "negligible": "#BBBBBB",
    "small":      "#88BBDD",
    "medium":     "#EE7733",
    "large":      "#CC3311",
}



def plot_cross_family_overview(df: pd.DataFrame) -> None:
    """One-page overview: strip + IQR bar of |Cohen's d| per family, sorted by median."""
    families = [f for f in _FAM_ORDER if f in df["family"].values]
    if not families:
        return

    medians = {f: df[df["family"] == f]["abs_d"].median() for f in families}
    families_sorted = sorted(families, key=lambda f: medians[f])

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(families_sorted) * 0.7 + 1.5)))
    rng = np.random.default_rng(0)

    for yi, fam in enumerate(families_sorted):
        vals = df[df["family"] == fam]["abs_d"].dropna().values
        if len(vals) == 0:
            continue
        color = _FAM_COLOR.get(fam, "#AAAAAA")
        jitter = rng.uniform(-0.22, 0.22, len(vals))
        ax.scatter(vals, yi + jitter, color=color, alpha=0.55,
                   s=28, zorder=3, edgecolors="none")
        ax.plot([np.median(vals)] * 2, [yi - 0.35, yi + 0.35],
                color=color, lw=2.5, solid_capstyle="round", zorder=4)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.barh(yi, q3 - q1, left=q1, height=0.5,
                color=color, alpha=0.20, zorder=2)

    for xv, lab in [(0.2, "small"), (0.5, "med"), (0.8, "large")]:
        ax.axvline(xv, color=MUTED, lw=0.7, ls="--", alpha=0.7)
        ax.text(xv + 0.01, len(families_sorted) - 0.5, lab,
                fontsize=7, color=MUTED, va="top")

    ax.set_yticks(range(len(families_sorted)))
    ax.set_yticklabels(families_sorted, fontsize=9)
    ax.set_xlabel("|Cohen's d|", labelpad=6)
    ax.set_title(
        "Cross-cohort shift by feature family\n"
        "(dots = features; bar = IQR; tick = median)",
        loc="left", weight="bold", fontsize=9,
    )
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = PLOTS_FAMILY / "_overview_by_family.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview saved: {out}")








def plot_pca_panel(ax, ic_raw: pd.DataFrame, np_raw: pd.DataFrame, title: str) -> None:
    """PCA scatter of subjects on complete cases only (no imputation).

    Missingness is not random: I-CONECT acoustic features exist almost only for
    the long-conversation subgroup, so median-imputing them collapses the other
    subjects onto an artificial line. Subjects with any NaN in these features
    are dropped instead, and the counts are reported in the legend.
    """
    from sklearn.decomposition import PCA

    ic_cc = ic_raw.dropna(how="any")
    np_cc = np_raw.dropna(how="any")
    n_ic, n_np = len(ic_cc), len(np_cc)
    try:
        if n_ic < 3 or n_np < 3:
            raise ValueError(f"too few complete cases (IC {n_ic}, 90+ {n_np})")
        combined = pd.concat([ic_cc, np_cc], axis=0)
        f_std    = combined.std().replace(0, np.nan).fillna(1)
        X_all    = ((combined - combined.mean()) / f_std).values
        pca      = PCA(n_components=min(2, X_all.shape[1]))
        coords   = pca.fit_transform(X_all)
        ve       = pca.explained_variance_ratio_

        def _lab(name, n, n_tot):
            drop = n_tot - n
            return f"{name} (n={n}" + (f"; {drop} with NaN dropped)" if drop else ")")

        ax.scatter(coords[:n_ic, 0], coords[:n_ic, 1],
                   color=IC_COLOR, alpha=0.55, s=22, edgecolors="none",
                   label=_lab("I-CONECT", n_ic, len(ic_raw)), zorder=3)
        ax.scatter(coords[n_ic:, 0], coords[n_ic:, 1] if len(ve) > 1 else np.zeros(n_np),
                   color=NP_COLOR, alpha=0.30, s=16, edgecolors="none",
                   label=_lab("90+", n_np, len(np_raw)), zorder=2)
        ax.set_xlabel(f"PC1 ({ve[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({ve[1]:.1%})" if len(ve) > 1 else "", fontsize=8)
        ax.legend(fontsize=8, frameon=False)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=MUTED)
    ax.set_title(title, loc="left", weight="bold", fontsize=9)
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)


def support_overlap(ic_v: np.ndarray, np_v: np.ndarray) -> float:
    """Fraction of IC values inside the 90+ [5th, 95th] percentile range."""
    if len(np_v) < 10 or len(ic_v) == 0:
        return float("nan")
    lo, hi = np.percentile(np_v, 5), np.percentile(np_v, 95)
    return float(np.mean((ic_v >= lo) & (ic_v <= hi)))


def plot_family_summary(
    df: pd.DataFrame,
    ic_subj: pd.DataFrame,
    np_subj: pd.DataFrame,
    family: str,
    top_n: int = 20,
) -> None:
    """Three-panel family summary.

    A (top-left)  : PCA scatter of subjects on complete cases (IC vs 90+).
    B (right)     : Per-feature lollipop of Cohen's d, sorted by d, marker colour =
                    KS, right-hand label = support overlap. Families with more
                    than `top_n` features show the `top_n` largest-KS features.
    C (bot-left)  : |d| vs KS for every feature in the family, colour = support
                    overlap; shaded box = shape change (|d| < 0.5, KS > 0.3).
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize

    sub = df[df["family"] == family].copy()
    if len(sub) < 2:
        return
    feats = sub["feature"].tolist()
    sub["overlap"] = [support_overlap(ic_subj[f].dropna().values,
                                      np_subj[f].dropna().values) for f in feats]

    out = PLOTS_FAMILY / f"summary_{slugify(family)}.png"
    fig = plt.figure(figsize=(15, 9.5))
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.1],
                   hspace=0.38, wspace=0.85)
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_lol = fig.add_subplot(gs[:, 1])

    # ── A: PCA ─────────────────────────────────────────────────────────────
    plot_pca_panel(ax_pca, ic_subj[feats], np_subj[feats],
                   "A  PCA of subjects (complete cases)")

    ks_norm = Normalize(vmin=0, vmax=1)
    ks_cmap = plt.get_cmap("viridis_r")

    # ── B: per-feature lollipop ───────────────────────────────────────────
    shown = sub if len(sub) <= top_n else sub.nlargest(top_n, "ks_stat")
    shown = shown.sort_values("cohens_d")
    y = np.arange(len(shown))
    ax_lol.hlines(y, 0, shown["cohens_d"], color=GRID, lw=1.6, zorder=1)
    ax_lol.scatter(shown["cohens_d"], y, c=shown["ks_stat"], cmap=ks_cmap,
                   norm=ks_norm, s=60, edgecolors=INK, linewidths=0.4, zorder=3)
    for xv, ls in [(0, "-"), (-0.8, ":"), (0.8, ":")]:
        ax_lol.axvline(xv, color=MUTED if xv else INK, lw=0.8, ls=ls, alpha=0.7)
    ax_lol.set_yticks(y)
    ax_lol.set_yticklabels([f[:40] + ("…" if len(f) > 40 else "")
                            for f in shown["feature"]], fontsize=7.5)
    ax_lol.set_ylim(-0.7, len(shown) - 0.3)
    # support overlap as right-hand labels (red = < 50%)
    ax_ov = ax_lol.secondary_yaxis("right")
    ax_ov.set_yticks(y)
    ax_ov.set_yticklabels([f"{v:.0%}" if np.isfinite(v) else "—"
                           for v in shown["overlap"]], fontsize=7.5)
    for lab, v in zip(ax_ov.get_yticklabels(), shown["overlap"]):
        if np.isfinite(v) and v < 0.5:
            lab.set_color("#CC3311"); lab.set_fontweight("bold")
    ax_ov.tick_params(length=0)
    ax_ov.set_ylabel("support overlap (IC inside 90+ [5th–95th])", fontsize=8)
    ax_lol.set_xlabel("Cohen's d  (I-CONECT − 90+)", fontsize=8)
    sel = "all features" if len(sub) <= top_n else f"top {top_n} of {len(sub)} by KS"
    ax_lol.set_title(f"B  Per-feature shift ({sel})", loc="left",
                     weight="bold", fontsize=9)
    ax_lol.grid(axis="x", color=GRID, lw=0.4); ax_lol.set_axisbelow(True)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=ks_norm, cmap=ks_cmap),
                      ax=ax_lol, location="top", fraction=0.025, pad=0.05, aspect=40)
    cb.set_label("KS statistic", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ── C: |d| vs KS, colour = support overlap ────────────────────────────
    ov_cmap = plt.get_cmap("RdYlGn")
    n_shape = int(((sub["abs_d"] < 0.5) & (sub["ks_stat"] > 0.3)).sum())
    ax_sc.fill_between([0, 0.5], 0.3, 1.0, color=MUTED, alpha=0.12, lw=0,
                       label=f"shape change: |d|<0.5, KS>0.3 ({n_shape})")
    sc = ax_sc.scatter(sub["abs_d"], sub["ks_stat"], c=sub["overlap"],
                       cmap=ov_cmap, vmin=0, vmax=1, s=26, alpha=0.8,
                       edgecolors="none", zorder=3)
    ax_sc.axvline(0.8, color=NP_COLOR, lw=0.8, ls="--", alpha=0.6,
                  label="|d|=0.8 (large)")
    ax_sc.set_ylim(0, 1.02); ax_sc.set_xlim(left=0)
    ax_sc.set_xlabel("|Cohen's d|", fontsize=8)
    ax_sc.set_ylabel("KS statistic", fontsize=8)
    ax_sc.legend(fontsize=7, frameon=False, loc="lower right")
    ax_sc.set_title(f"C  |d| vs KS  (all {len(sub)} features)", loc="left",
                    weight="bold", fontsize=9)
    ax_sc.grid(color=GRID, lw=0.4); ax_sc.set_axisbelow(True)
    cb2 = fig.colorbar(sc, ax=ax_sc, fraction=0.04, pad=0.02)
    cb2.set_label("support overlap", fontsize=8); cb2.ax.tick_params(labelsize=7)

    n_large = int((sub["abs_d"] >= 0.8).sum())
    n_risk  = int((sub["overlap"] < 0.5).sum())
    fig.suptitle(
        f"{family}  ·  {len(sub)} features  ·  "
        f"{n_large} large-shift (|d|≥0.8)  ·  "
        f"{n_risk} with support overlap < 50%",
        fontsize=10, weight="bold",
    )
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Family summary: {out.name}")


def plot_domain_auc(
    df: pd.DataFrame,
    ic_subj: pd.DataFrame,
    np_subj: pd.DataFrame,
) -> None:
    """Cross-family transfer-risk dashboard.

    Computes per family:
      - domain discriminability AUC (logistic regression, 5-fold CV, complete cases)
      - mean |Cohen's d|
      - mean KS statistic
      - mean support overlap rate, and the share of features with overlap < 50%
      - mean |deltaR| (correlation structure divergence; saved to the CSV only,
        because subgroup mixing inflates it and it is not a transfer-risk score)

    Saves results/domain_auc.csv and plots family/transfer_risk.png: a heatmap of
    AUC, mean |d|, mean KS and the share of features with overlap < 50%, families
    sorted by AUC. Colours use fixed absolute scales per column, not min-max across
    families, so a family is not painted green just for being the best of a bad lot.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    families = sorted(df["family"].unique())
    records  = []
    for fam in families:
        sub   = df[df["family"] == fam]
        feats = sub["feature"].tolist()
        ic_X  = ic_subj[feats].copy()
        np_X  = np_subj[feats].copy()

        # AUC on complete cases: IC acoustic features are missing for whole
        # subjects (not at random), and imputing those rows with the pooled
        # median would plant fake "typical 90+" IC samples.
        ic_cc = ic_X.dropna(how="any")
        np_cc = np_X.dropna(how="any")
        X = pd.concat([ic_cc, np_cc], axis=0).values
        y = np.array([0] * len(ic_cc) + [1] * len(np_cc))
        if min(len(ic_cc), len(np_cc)) >= 10:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf",   LogisticRegression(C=0.1, max_iter=500, random_state=42)),
            ])
            try:
                auc = float(cross_val_score(
                    pipe, X, y,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="roc_auc",
                ).mean())
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        # mean |d| and mean KS
        mean_abs_d = float(sub["abs_d"].mean())
        mean_ks    = float(sub["ks_stat"].mean())

        # support overlap: mean, and share of features below the 50% risk line
        overlaps = np.array([support_overlap(ic_subj[f].dropna().values,
                                             np_subj[f].dropna().values) for f in feats])
        ov_ok = overlaps[np.isfinite(overlaps)]
        mean_overlap = float(ov_ok.mean()) if len(ov_ok) else np.nan
        frac_ov_lt50 = float(np.mean(ov_ok < 0.5)) if len(ov_ok) else np.nan

        # mean |deltaR|
        R_ic = ic_X.dropna(how="all").corr().values
        R_np = np_X.dropna(how="all").corr().values
        if R_ic.shape == R_np.shape and len(feats) >= 3:
            tri = np.abs(R_ic - R_np)[np.triu_indices(len(feats), k=1)]
            mean_abs_dr = float(np.nanmean(tri))
        else:
            mean_abs_dr = np.nan

        records.append({
            "family":       fam,
            "auc":          round(auc, 3),
            "mean_abs_d":   round(mean_abs_d, 3),
            "mean_ks":      round(mean_ks, 3),
            "mean_overlap": round(mean_overlap, 3) if not np.isnan(mean_overlap) else np.nan,
            "frac_overlap_lt50": round(frac_ov_lt50, 3),
            "mean_abs_dR":  round(mean_abs_dr,  3) if not np.isnan(mean_abs_dr)  else np.nan,
            "n_feats":      len(feats),
            "n_ic_auc":     len(ic_cc),
            "n_np_auc":     len(np_cc),
        })
        print(f"    {fam:<10} AUC={auc:.3f}  |d|={mean_abs_d:.2f}  KS={mean_ks:.2f}"
              f"  ovlp={mean_overlap:.2f}  ovlp<50%={frac_ov_lt50:.0%}  |dR|={mean_abs_dr:.2f}")

    auc_df = pd.DataFrame(records).sort_values("auc", ascending=False)
    auc_df.to_csv(HERE / "results" / "domain_auc.csv", index=False)

    # ── heatmap: families x metrics, fixed absolute colour scales ───────
    # (label, value column, scale lo, scale hi); colour = clip((v-lo)/(hi-lo), 0, 1),
    # red = higher transfer risk
    metrics = [
        ("Domain AUC\n(scale 0.5\u20131.0)",                       "auc",               0.5, 1.0),
        ("Mean |Cohen's d|\n(scale 0\u20132)",                     "mean_abs_d",        0.0, 2.0),
        ("Mean KS\n(scale 0\u20131)",                              "mean_ks",           0.0, 1.0),
        ("Features with support\noverlap < 50%  (scale 0\u20131)", "frac_overlap_lt50", 0.0, 1.0),
    ]
    fam_labels = auc_df["family"].tolist()
    heat_data  = np.full((len(fam_labels), len(metrics)), np.nan)
    heat_norm  = np.full_like(heat_data, np.nan)
    for c, (_, col, lo, hi) in enumerate(metrics):
        heat_data[:, c] = auc_df[col].values
        heat_norm[:, c] = np.clip((auc_df[col].values - lo) / (hi - lo), 0, 1)

    fig, ax = plt.subplots(figsize=(10, max(3.8, len(fam_labels) * 0.6 + 1.3)))
    im = ax.imshow(heat_norm, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")

    # annotate cells with raw values (share column as a percentage)
    for r in range(len(fam_labels)):
        for c, (_, col, _, _) in enumerate(metrics):
            val = heat_data[r, c]
            if np.isnan(val):
                txt = "\u2014"
            elif col == "frac_overlap_lt50":
                txt = f"{val:.0%}"
            elif col in ("auc", "mean_ks"):
                txt = f"{val:.3f}"
            else:
                txt = f"{val:.2f}"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=9, color=INK, weight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ylabels = [f"{row['family']}\n(AUC on {int(row['n_ic_auc'])} IC)"
               if int(row["n_ic_auc"]) < 150 else row["family"]
               for _, row in auc_df.iterrows()]
    ax.set_yticks(range(len(fam_labels)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_title(
        "Transfer risk by feature family  (red = higher risk; fixed absolute colour scales)",
        fontsize=9, weight="bold", pad=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="risk (0 = low end of the column scale, 1 = high end)")
    fig.text(0.01, 0.005,
             "AUC uses complete cases only; |dR| (correlation-structure change) is in domain_auc.csv.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS_FAMILY / "transfer_risk.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Transfer risk heatmap saved.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Local dataset/ directory (i-conect/ and 90plus/ subdirs). "
                         "Omit to use Great Lakes via lmc.io.")
    ap.add_argument("--min-nonnan", type=float, default=0.8,
                    help="Min fraction of non-NaN subjects per cohort (default 0.8).")
    ap.add_argument("--min-nonnan-ac", type=float, default=0.5,
                    help="Min non-NaN fraction for acoustic features (default 0.5; IC acoustic only covers ~54%% of subjects).")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip per-feature PNG; output stats CSV only.")
    ap.add_argument("--paths", default="config/paths.yaml",
                    help="Path to paths.yaml (GL mode only).")
    ap.add_argument("--family-only", nargs="*", metavar="FAMILY", default=None,
                    help="Only regenerate family-level plots, reusing "
                         "results/all_features_stats.csv from a previous full run. "
                         "Optionally list families (ling temporal acoustic semantic) to "
                         "redraw just those summaries; use 'all' for the "
                         "all-features combined summary.")
    ap.add_argument("--normalized-only", action="store_true",
                    help="Only redo the per-cohort robust z-score shape analysis "
                         "(results/normalized/), reusing results/all_features_stats.csv.")
    args = ap.parse_args()

    if args.data_dir is not None:
        print(f"Mode: local  ({args.data_dir})")
        ic_subj, np_subj, all_feats, family_sets = load_local(args.data_dir)
    else:
        print("Mode: Great Lakes (lmc.io)")
        ic_subj, np_subj, all_feats, family_sets = load_gl(args.paths)

    print(f"  I-CONECT: {len(ic_subj)} subjects")
    print(f"  90+ :     {len(np_subj)} subjects")
    print(f"  Features to screen: {len(all_feats)}")

    if args.normalized_only:
        stats_csv = HERE / "results" / "all_features_stats.csv"
        if not stats_csv.exists():
            ap.error(f"--normalized-only needs {stats_csv}; run once without it first.")
        run_normalized_analysis(pd.read_csv(stats_csv), ic_subj, np_subj,
                                plots=not args.no_plots)
        return

    if args.family_only is not None:
        stats_csv = HERE / "results" / "all_features_stats.csv"
        if not stats_csv.exists():
            ap.error(f"--family-only needs {stats_csv}; run once without it first.")
        df = pd.read_csv(stats_csv)
        families = args.family_only or None
        if families:
            unknown = sorted(set(families) - set(df["family"]) - {"all"})
            if unknown:
                ap.error(f"unknown family {unknown}; available: 'all' (all-features "
                         f"summary), {sorted(df['family'].unique())}")
        print(f"  Loaded {len(df)} feature stats from {stats_csv}")
        generate_family_plots(df, ic_subj, np_subj, families)
        return

    ac_feat_set = family_sets.get("ac", set())
    def _passes_filter(f: str) -> bool:
        if f not in ic_subj.columns or f not in np_subj.columns:
            return False
        thr = args.min_nonnan_ac if f in ac_feat_set else args.min_nonnan
        return (ic_subj[f].notna().mean() >= thr
                and np_subj[f].notna().mean() >= thr)
    keep = [f for f in all_feats if _passes_filter(f)]
    print(f"  Passing non-NaN filter: {len(keep)} "
          f"(ling/temp/sem ≥{args.min_nonnan:.0%}, acoustic ≥{args.min_nonnan_ac:.0%})")

    if not args.no_plots:
        for _d in [PLOTS_DIR / s for s in
                   ("ling", "temporal", "semantic", "acoustic", "family")]:
            _d.mkdir(parents=True, exist_ok=True)

    # group acoustic features by base name (strip _whole/_mean/_std)
    ac_feat_set = family_sets.get("ac", set())
    ac_bases: dict = {}  # base -> {stat: feat}
    non_ac_keep: list = []
    for f in keep:
        if f in ac_feat_set:
            base = acoustic_base(f)
            stat = re.sub(r".*_(whole|mean|std)$", r"\1", f)
            ac_bases.setdefault(base, {})[stat] = f
        else:
            non_ac_keep.append(f)

    rows = []
    total_items = len(non_ac_keep) + len(ac_bases)
    done = 0

    # ── non-acoustic features (individual plots) ──────────────────────
    for feat in non_ac_keep:
        s = feature_stats(feat, ic_subj, np_subj, family_sets)
        if s is None:
            done += 1; continue
        rows.append(s)
        if not args.no_plots:
            ic_v = ic_subj[feat].dropna().values
            np_v = np_subj[feat].dropna().values
            if len(ic_v) >= 3 and len(np_v) >= 3:
                plot_feature(feat, s, ic_v, np_v, out_dir=_feat_subdir(s["family"]))
        done += 1
        if done % 25 == 0:
            print(f"  … {done}/{total_items} done")

    # ── acoustic features (trio plots: whole/mean/std per base) ───────
    for base, stat_map in ac_bases.items():
        stat_rows_d: dict = {}
        stat_vals: dict = {}
        for stat, feat in stat_map.items():
            s = feature_stats(feat, ic_subj, np_subj, family_sets)
            if s is None:
                continue
            rows.append(s)
            stat_rows_d[stat] = s
            ic_v = ic_subj[feat].dropna().values
            np_v = np_subj[feat].dropna().values
            if len(ic_v) >= 3 and len(np_v) >= 3:
                stat_vals[stat] = (ic_v, np_v)
        if not args.no_plots and stat_rows_d:
            _ac_fam = next(iter(stat_rows_d.values()))["family"]
            plot_acoustic_trio(base, stat_rows_d, stat_vals,
                               out_dir=_feat_subdir(_ac_fam))
        done += 1
        if done % 25 == 0:
            print(f"  … {done}/{total_items} done")

    df = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    out_csv = HERE / "results" / "all_features_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} features)")
    if not args.no_plots:
        print(f"Plots: {PLOTS_DIR}  ({len(df)} files)")

    print("\n── Per-family summary (mean |Cohen's d|, sorted ascending) ────────")
    fam_summary = df.groupby("family").agg(
        n_feat=("abs_d", "count"),
        mean_abs_d=("abs_d", "mean"),
        n_large=("abs_d", lambda x: (x >= 0.8).sum()),
        mean_ks=("ks_stat", "mean"),
    ).sort_values("mean_abs_d")
    for fam, row in fam_summary.iterrows():
        print(f"  {fam:<28} mean|d|={row.mean_abs_d:.3f}  "
              f"KS={row.mean_ks:.3f}  "
              f"({int(row.n_feat)} feats, {int(row.n_large)} large-shift)")

    print(f"\nTotal: {len(keep)} features screened | "
          f"{(df.abs_d >= 0.8).sum()} large-shift (|d|≥0.8) | "
          f"{(df.ks_pval < 0.05).sum()} KS-significant (α=0.05)")

    if not args.no_plots:
        generate_family_plots(df, ic_subj, np_subj)

    run_normalized_analysis(df, ic_subj, np_subj, plots=not args.no_plots)



def plot_all_features_summary(
    df: pd.DataFrame,
    ic_subj: pd.DataFrame,
    np_subj: pd.DataFrame,
) -> None:
    """Four-panel summary across ALL features (all families combined).

    Identical panel layout to plot_family_summary but uses every feature in df,
    colour-coded by family in the PCA scatter.  Saves to
    results/plots/family/summary_all_features.png.

    A (top-left)  : PCA scatter of subjects, coloured IC vs 90+ (family shapes).
    B (top-right) : Histogram of Cohen's d across ALL features, by shift tier.
    C (bot-left)  : Histogram of support-overlap rates for ALL features.
    D (bot-right) : Scatter of |d| vs KS, coloured by family; the upper-left
                    quadrant (|d| < 0.5, KS > 0.3) flags shape change.
    """
    feats = [f for f in df["feature"].tolist()
             if f in ic_subj.columns and f in np_subj.columns]
    n_f   = len(feats)
    if n_f < 2:
        return

    ic_raw = ic_subj[feats].copy()
    np_raw = np_subj[feats].copy()

    PLOTS_FAMILY.mkdir(parents=True, exist_ok=True)
    out = PLOTS_FAMILY / "summary_all_features.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                              gridspec_kw={"hspace": 0.42, "wspace": 0.35})
    ax_pca, ax_d, ax_ovlp, ax_scatter = axes.flat

    # ── A: PCA scatter (complete cases) ────────────────────────────────────
    plot_pca_panel(ax_pca, ic_raw, np_raw,
                   "A  PCA of subjects (all features, complete cases)")

    # ── B: Cohen's d histogram — all features ─────────────────────────────
    d_vals = df["cohens_d"].dropna().values
    bins   = np.linspace(d_vals.min() - 0.1, d_vals.max() + 0.1, 40)
    tiers  = [
        ("negligible", MUTED,     lambda v: np.abs(v) <  0.2),
        ("small",      "#88CCEE", lambda v: (np.abs(v) >= 0.2) & (np.abs(v) < 0.5)),
        ("medium",     IC_COLOR,  lambda v: (np.abs(v) >= 0.5) & (np.abs(v) < 0.8)),
        ("large",      NP_COLOR,  lambda v: np.abs(v) >= 0.8),
    ]
    bottom = np.zeros(len(bins) - 1)
    for label, color, mask in tiers:
        vals = d_vals[mask(d_vals)]
        if len(vals) == 0:
            continue
        h, _ = np.histogram(vals, bins=bins)
        ax_d.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom,
                 color=color, align="edge", alpha=0.80,
                 label=f"{label} ({mask(d_vals).sum()})")
        bottom += h
    ax_d.axvline(0,    color=INK,      lw=0.8, ls="--", alpha=0.6)
    ax_d.axvline(-0.8, color=NP_COLOR, lw=0.7, ls=":",  alpha=0.5)
    ax_d.axvline( 0.8, color=NP_COLOR, lw=0.7, ls=":",  alpha=0.5)
    n_large = int((np.abs(d_vals) >= 0.8).sum())
    ax_d.text(0.97, 0.95, f"{n_large}/{len(d_vals)} large-shift",
              transform=ax_d.transAxes, ha="right", va="top",
              fontsize=8, color=NP_COLOR, weight="bold")
    ax_d.set_xlabel("Cohen's d  (I-CONECT − 90+)", fontsize=8)
    ax_d.set_ylabel("# features", fontsize=8)
    ax_d.legend(fontsize=7, frameon=False, ncol=2)
    ax_d.set_title("B  Cohen's d  — all features", loc="left",
                    weight="bold", fontsize=9)
    ax_d.grid(color=GRID, lw=0.4); ax_d.set_axisbelow(True)

    # ── C: Support overlap — all features ─────────────────────────────────
    ovlp = np.array([support_overlap(ic_subj[f].dropna().values,
                                     np_subj[f].dropna().values) for f in feats])
    ovlp = ovlp[np.isfinite(ovlp)]
    if len(ovlp):
        ax_ovlp.hist(ovlp, bins=25, range=(0, 1),
                     color=IC_COLOR, alpha=0.70, edgecolor="white", lw=0.4)
        ax_ovlp.axvline(0.5, color=NP_COLOR, lw=1.4, ls="--",
                         label="50% threshold (high risk)")
        ax_ovlp.axvline(0.8, color=MUTED,    lw=1.2, ls=":",
                         label="80% threshold (safe)")
        pct_risk = float(np.mean(ovlp < 0.5)) * 100
        ax_ovlp.text(0.03, 0.95,
                     f"{pct_risk:.0f}% of features\nhigh extrapolation risk",
                     transform=ax_ovlp.transAxes, va="top", fontsize=8,
                     color=NP_COLOR, weight="bold")
        ax_ovlp.legend(fontsize=8, frameon=False)
    ax_ovlp.set_xlabel("Fraction of IC subjects inside 90+ [5th–95th pct]", fontsize=8)
    ax_ovlp.set_ylabel("# features", fontsize=8)
    ax_ovlp.set_xlim(0, 1)
    ax_ovlp.set_title("C  Support overlap — all features",
                       loc="left", weight="bold", fontsize=9)
    ax_ovlp.grid(color=GRID, lw=0.4); ax_ovlp.set_axisbelow(True)

    # ── D: |d| vs KS scatter, colour = family ─────────────────────────────
    fam_list = [f for f in _FAM_ORDER if f in df["family"].values]
    fam_color_map = _FAM_COLOR
    for fam in fam_list:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        ax_scatter.scatter(
            sub["abs_d"].values,
            sub["ks_stat"].values,
            color=fam_color_map[fam], alpha=0.45, s=18,
            edgecolors="none", label=fam, zorder=3,
        )
    ax_scatter.axvline(0.8, color=NP_COLOR, lw=0.8, ls="--", alpha=0.6,
                        label="|d|=0.8 (large)")
    # shape-change quadrant: mean barely moves but the ECDFs separate
    n_shape = int(((df["abs_d"] < 0.5) & (df["ks_stat"] > 0.3)).sum())
    ax_scatter.fill_between([0, 0.5], 0.3, 1.0, color=MUTED, alpha=0.12, lw=0,
                            label=f"shape change: |d|<0.5, KS>0.3 ({n_shape})")
    ax_scatter.set_ylim(0, 1.02)
    ax_scatter.set_xlabel("|Cohen's d|", fontsize=8)
    ax_scatter.set_ylabel("KS statistic", fontsize=8)
    ax_scatter.legend(fontsize=7, frameon=False, ncol=1,
                       loc="lower right", markerscale=1.4)
    ax_scatter.set_title("D  |d| vs KS  (colour = family)",
                          loc="left", weight="bold", fontsize=9)
    ax_scatter.grid(color=GRID, lw=0.4); ax_scatter.set_axisbelow(True)

    _how = "all"
    n_ic_sub = ic_raw.dropna(how=_how).shape[0]
    n_np_sub = np_raw.dropna(how=_how).shape[0]
    fig.suptitle(
        f"All features combined  ·  {n_f} features  ·  "
        f"IC n≈{n_ic_sub}  90+ n≈{n_np_sub}",
        fontsize=10, weight="bold",
    )
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  All-features summary: summary_all_features.png")

# ── I-CONECT long / short conversation subgroups ──────────────────────────────
# I-CONECT conversation durations are bimodal: the short subgroup ends at ~900 s
# and the long one starts at ~1000 s (90+ max is ~1085 s, 99th pct ~530 s).
LONG_CONV_SEC = 900.0
SUBGROUP_FEATURES = [  # (feature, display label) - length-driven first, then cohort-level
    ("ix_conversation_dur_sec", "Conversation duration (s)"),
    ("pt_total_words", "Participant total words"),
    ("Adverbs", "Adverbs (LIWC count)"),
    ("Assent", "Assent (LIWC count)"),
    ("Simple TTR", "Simple TTR"),
    ("sm_content_diversity", "Content diversity"),
    ("Measure of lexical textual diversity [mtld_ma_bid]", "MTLD (bidirectional)"),
    ("ix_patient_talk_ratio", "Participant talk-time ratio"),
    ("pt_pause_ratio", "Participant pause ratio"),
    ("ix_turn_rate_per_min", "Turns per minute"),
]


def plot_ic_subgroups(df: pd.DataFrame, ic_subj: pd.DataFrame, np_subj: pd.DataFrame) -> None:
    """Compare the I-CONECT long- and short-conversation subgroups against 90+.

    Writes results/ic_subgroup_summary.csv (selected features: medians and support
    overlap per subgroup), results/ic_subgroup_family.csv (per family: share of
    features whose support overlap is < 50%, per subgroup) and
    plots/family/ic_subgroups.png (A: duration distribution, B: per-feature
    overlap, C: per-family share of out-of-range features).
    """
    dur = ic_subj["ix_conversation_dur_sec"]
    groups = {"IC long": ic_subj[dur > LONG_CONV_SEC], "IC short": ic_subj[dur <= LONG_CONV_SEC]}
    g_color = {"IC long": "#1F3D7A", "IC short": "#7FA7D9"}

    rows = []
    for feat, label in SUBGROUP_FEATURES:
        if feat not in ic_subj.columns:
            continue
        np_v = np_subj[feat].dropna().values
        row = {"feature": feat, "label": label,
               "family": df.loc[df["feature"] == feat, "family"].iloc[0],
               "median_90plus": float(np.median(np_v))}
        for g, sub in groups.items():
            v = sub[feat].dropna().values
            key = g.lower().replace(" ", "_")
            row[f"median_{key}"] = float(np.median(v))
            row[f"overlap_{key}"] = support_overlap(v, np_v)
            row[f"n_{key}"] = int(len(v))
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(HERE / "results" / "ic_subgroup_summary.csv", index=False)

    fam_rows = []
    for fam in FAMILIES:
        feats = df.loc[df["family"] == fam, "feature"].tolist()
        rec = {"family": fam, "n_feats": len(feats)}
        for g, sub in groups.items():
            key = g.lower().replace(" ", "_")
            ov = np.array([support_overlap(sub[f].dropna().values, np_subj[f].dropna().values)
                           for f in feats])
            ok = ov[np.isfinite(ov)]
            rec[f"frac_overlap_lt50_{key}"] = round(float(np.mean(ok < 0.5)), 3) if len(ok) else np.nan
            rec[f"n_subjects_{key}"] = int(sub[feats].dropna(how="all").shape[0])
        fam_rows.append(rec)
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(HERE / "results" / "ic_subgroup_family.csv", index=False)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1, 1.25, 0.9], "wspace": 0.55})

    # A: conversation duration, log x
    bins = np.logspace(np.log10(30), np.log10(2500), 45)
    ax_a.hist(np_subj["ix_conversation_dur_sec"].dropna(), bins=bins, density=True,
              color=NP_COLOR, alpha=0.45, label=f"90+ (n={np_subj['ix_conversation_dur_sec'].notna().sum()})")
    for g, sub in groups.items():
        ax_a.hist(sub["ix_conversation_dur_sec"].dropna(), bins=bins, density=True,
                  color=g_color[g], alpha=0.75, label=f"{g} (n={len(sub)})")
    ax_a.axvline(LONG_CONV_SEC, color=INK, lw=1, ls="--")
    ax_a.text(LONG_CONV_SEC * 1.04, ax_a.get_ylim()[1] * 0.92, f"split {LONG_CONV_SEC:.0f} s",
              fontsize=8, color=INK)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Conversation duration (s, log scale)", fontsize=8)
    ax_a.set_ylabel("Density", fontsize=8)
    ax_a.legend(fontsize=8, frameon=False, loc="upper left")
    ax_a.set_title("A  I-CONECT splits into two conversation lengths", loc="left",
                   weight="bold", fontsize=9)
    ax_a.grid(color=GRID, lw=0.4); ax_a.set_axisbelow(True)

    # B: support overlap per feature and subgroup
    y = np.arange(len(feat_df))[::-1]
    for g in groups:
        key = g.lower().replace(" ", "_")
        ax_b.scatter(feat_df[f"overlap_{key}"], y, s=70, color=g_color[g], label=g,
                     edgecolors=INK, linewidths=0.4, zorder=3)
    for yi, (_, r) in zip(y, feat_df.iterrows()):
        ax_b.plot([r["overlap_ic_long"], r["overlap_ic_short"]], [yi, yi],
                  color=GRID, lw=2, zorder=1)
    ax_b.axvline(0.5, color=NP_COLOR, lw=1.2, ls="--", label="50% (high risk)")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(feat_df["label"], fontsize=8)
    ax_b.set_xlim(-0.03, 1.03)
    ax_b.set_xlabel("Support overlap: share of subgroup inside 90+ [5th–95th pct]", fontsize=8)
    ax_b.legend(fontsize=8, frameon=False, loc="lower right")
    ax_b.set_title("B  Which shifts come from the long subgroup only?", loc="left",
                   weight="bold", fontsize=9)
    ax_b.grid(axis="x", color=GRID, lw=0.4); ax_b.set_axisbelow(True)

    # C: share of out-of-range features per family and subgroup
    x = np.arange(len(fam_df)); bw = 0.38
    for k, g in enumerate(groups):
        key = g.lower().replace(" ", "_")
        vals = fam_df[f"frac_overlap_lt50_{key}"].values
        bars = ax_c.bar(x + (k - 0.5) * bw, vals, bw, color=g_color[g], label=g)
        for b_, v, n in zip(bars, vals, fam_df[f"n_subjects_{key}"]):
            ax_c.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.0%}\nn={n}",
                      ha="center", va="bottom", fontsize=6.5, color=INK)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(fam_df["family"], fontsize=8)
    ax_c.set_ylim(0, 1.05)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_c.set_ylabel("Features with support overlap < 50%", fontsize=8)
    ax_c.legend(fontsize=8, frameon=False, loc="upper right")
    ax_c.set_title("C  Out-of-range features per family", loc="left", weight="bold", fontsize=9)
    ax_c.grid(axis="y", color=GRID, lw=0.4); ax_c.set_axisbelow(True)

    fig.savefig(PLOTS_FAMILY / "ic_subgroups.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  IC subgroup comparison: ic_subgroups.png, ic_subgroup_summary.csv, ic_subgroup_family.csv")


# ── per-cohort robust z-score: shape difference left after removing the shift ─
# Each feature is robust-z-scored separately within each cohort,
#   z = (x - median) / IQR      (fallback: (x - median) / SD when the IQR is 0),
# which removes location and scale. A plain mean/SD z-score is avoided because a
# few extreme values inflate the SD and squash the rest of the distribution,
# producing spurious shape differences. Per feature we then measure
#   qq_ccc    PRIMARY. Lin's concordance correlation between the two cohorts'
#             5th..95th percentiles (robust z). 1 = the Q-Q points lie on the
#             identity line, i.e. same shape. Unlike Pearson r it also penalises a
#             slope or offset away from the diagonal. The outer percentiles are
#             left out because robust z leaves the tails unbounded (up to hundreds of
#             IQRs) and, with 156 I-CONECT subjects, the 1st/99th percentiles rest on
#             one or two people; including them lets single subjects decide the CCC.
#   ks_rz     KS statistic between the two robust-z samples (supporting metric;
#             most sensitive to the bulk of the distribution)
#   qq_pearson_r, qq_slope   Pearson r and least-squares slope of the same Q-Q
#             points, for reference (r alone ignores the slope)
#   sd_ratio  SD(I-CONECT) / SD(90+) on raw values (normalising hides scale)
# Outputs go to results/normalized/ so the raw-data figures are left untouched.
NORM_DIR = HERE / "results" / "normalized"
QQ_QUANTILES = np.arange(5, 96)     # percentiles entering the CCC
QQ_PLOT_QUANTILES = np.arange(1, 100)  # percentiles drawn in the Q-Q plot
CCC_OK, CCC_WARN = 0.95, 0.90  # >= OK: same shape; < WARN: clearly different


def slugify_full(name: str) -> str:
    """Untruncated slug: slugify() cuts at 60 chars, which can merge the
    _whole/_mean/_std variants of long acoustic names."""
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()


def robust_z(v: np.ndarray) -> tuple[np.ndarray | None, str]:
    """(x - median) / IQR; falls back to the SD when the IQR is 0."""
    med = np.median(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    if iqr > 0:
        return (v - med) / iqr, ""
    sd = v.std(ddof=1)
    if sd > 0:
        return (v - med) / sd, "IQR = 0, scaled by SD"
    return None, "constant"


def concordance(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (x.var() + y.var() + (mx - my) ** 2))


def shape_stats(ic_v: np.ndarray, np_v: np.ndarray) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    sd_ic, sd_np = ic_v.std(ddof=1), np_v.std(ddof=1)
    res = {"sd_ratio": float(sd_ic / sd_np) if sd_np > 0 else np.nan,
           "qq_ccc": np.nan, "ks_rz": np.nan, "ks_rz_pval": np.nan,
           "qq_pearson_r": np.nan, "qq_slope": np.nan, "note": ""}
    z_ic, n_ic = robust_z(ic_v)
    z_np, n_np = robust_z(np_v)
    notes = [f"I-CONECT: {n_ic}" if n_ic else "", f"90+: {n_np}" if n_np else ""]
    res["note"] = "; ".join(n for n in notes if n)
    if z_ic is None or z_np is None:
        return res, None, None
    ks, p = stats.ks_2samp(z_ic, z_np)
    res["ks_rz"], res["ks_rz_pval"] = float(ks), float(p)
    q_ic, q_np = np.percentile(z_ic, QQ_QUANTILES), np.percentile(z_np, QQ_QUANTILES)
    res["qq_ccc"] = concordance(q_np, q_ic)
    if np.ptp(q_ic) > 0 and np.ptp(q_np) > 0:
        res["qq_pearson_r"] = float(np.corrcoef(q_np, q_ic)[0, 1])
        res["qq_slope"] = float(np.polyfit(q_np, q_ic, 1)[0])
    return res, z_ic, z_np


def plot_normalized_feature(feat: str, fam: str, row: dict,
                            z_ic: np.ndarray, z_np: np.ndarray, out: Path) -> None:
    """Left: robust-z densities of both cohorts. Right: Q-Q plot of percentiles 1-99."""
    fig, (ax_k, ax_q) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                     gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.28})
    lo = min(np.percentile(z_ic, 0.5), np.percentile(z_np, 0.5))
    hi = max(np.percentile(z_ic, 99.5), np.percentile(z_np, 99.5))
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - pad, hi + pad, 400)
    for z, col, lab in [(z_np, NP_COLOR, f"90+ (n={len(z_np)})"),
                        (z_ic, IC_COLOR, f"I-CONECT (n={len(z_ic)})")]:
        try:
            k = gaussian_kde(z, bw_method="scott")(xs)
            ax_k.fill_between(xs, k, color=col, alpha=0.2)
            ax_k.plot(xs, k, color=col, lw=2, label=lab)
        except Exception:  # degenerate sample: fall back to a histogram
            ax_k.hist(z, bins=30, density=True, color=col, alpha=0.35, label=lab)
    ax_k.set_xlim(xs[0], xs[-1])
    ax_k.set_ylim(bottom=0)
    ax_k.set_xlabel("robust z = (x − median) / IQR, within each cohort", fontsize=8.5)
    ax_k.set_ylabel("Density", fontsize=8.5)
    ax_k.legend(frameon=False, fontsize=8, loc="upper left")
    ax_k.grid(axis="x", color=GRID, lw=0.5); ax_k.set_axisbelow(True)
    ax_k.set_title(f"After robust z-score: {feat[:62]}\n[{fam}]", loc="left",
                   weight="bold", fontsize=9)
    ax_k.text(0.98, 0.97,
              f"Q–Q CCC = {row['qq_ccc']:.3f}\nKS (robust z) = {row['ks_rz']:.3f}\n"
              f"SD ratio (raw) = {row['sd_ratio']:.2f}\nraw d = {row['cohens_d_raw']:+.2f}",
              transform=ax_k.transAxes, ha="right", va="top", fontsize=8, family="monospace",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID, alpha=0.92))

    q_np, q_ic = np.percentile(z_np, QQ_PLOT_QUANTILES), np.percentile(z_ic, QQ_PLOT_QUANTILES)
    inner = np.isin(QQ_PLOT_QUANTILES, QQ_QUANTILES)
    # axis limits from the percentiles used in the CCC, so a few extreme points
    # do not shrink the informative part of the plot
    lo_l = min(q_np[inner].min(), q_ic[inner].min()); hi_l = max(q_np[inner].max(), q_ic[inner].max())
    pad_l = 0.08 * (hi_l - lo_l) if hi_l > lo_l else 1.0
    lim = [lo_l - pad_l, hi_l + pad_l]
    ax_q.plot(lim, lim, color=MUTED, lw=1, ls="--", label="identity (same shape)")
    ccc = row["qq_ccc"]
    pt_col = "#2a9d60" if ccc >= CCC_OK else "#d97706" if ccc >= CCC_WARN else "#d63939"
    ax_q.scatter(q_np[inner], q_ic[inner], s=12, color=pt_col, alpha=0.85, zorder=3,
                 label="percentiles 5–95 (CCC)")
    ax_q.scatter(np.clip(q_np[~inner], *lim), np.clip(q_ic[~inner], *lim), s=10, color=MUTED,
                 alpha=0.5, zorder=2, label="1–4, 96–99 (not used)")
    ax_q.set_xlim(lim); ax_q.set_ylim(lim)
    for q in (25, 50, 75):
        ax_q.scatter(np.percentile(z_np, q), np.percentile(z_ic, q), s=40, facecolor="none",
                     edgecolor=INK, lw=1.2, zorder=4)
    ax_q.set_xlabel("90+ quantile (robust z)", fontsize=8.5)
    ax_q.set_ylabel("I-CONECT quantile (robust z)", fontsize=8.5)
    ax_q.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax_q.grid(color=GRID, lw=0.5); ax_q.set_axisbelow(True)
    ax_q.set_title(f"Q–Q plot · CCC = {ccc:.3f}  (circles: quartiles)", loc="left",
                   weight="bold", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)


def run_normalized_analysis(df: pd.DataFrame, ic_subj: pd.DataFrame,
                            np_subj: pd.DataFrame, plots: bool = True) -> pd.DataFrame:
    """Per-feature shape comparison after per-cohort robust z-scoring.

    Writes results/normalized/feature_shape_stats.csv (sorted by qq_ccc, least
    similar first) and, if `plots`, results/normalized/plots/<family>/<slug>.png.
    """
    print("\n── Per-cohort robust z-score: shape differences per feature ─────")
    slugs = [slugify_full(f) for f in df["feature"]]
    assert len(set(slugs)) == len(slugs), "feature slugs collide"
    rows = []
    for i, r in enumerate(df.itertuples()):
        ic_v = ic_subj[r.feature].dropna().values
        np_v = np_subj[r.feature].dropna().values
        res, z_ic, z_np = shape_stats(ic_v, np_v)
        row = {"feature": r.feature, "family": r.family,
               "n_iconect": len(ic_v), "n_90plus": len(np_v),
               "cohens_d_raw": r.cohens_d, "ks_raw": r.ks_stat,
               **res, "plot": f"{r.family}/{slugs[i]}.png" if z_ic is not None else ""}
        rows.append(row)
        if plots and z_ic is not None:
            plot_normalized_feature(r.feature, r.family, row, z_ic, z_np,
                                    NORM_DIR / "plots" / row["plot"])
        if (i + 1) % 100 == 0:
            print(f"  … {i + 1}/{len(df)} features")

    out = pd.DataFrame(rows).sort_values("qq_ccc", ascending=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(NORM_DIR / "feature_shape_stats.csv", index=False)
    ok = out["qq_ccc"].notna()
    print(f"  Saved {NORM_DIR / 'feature_shape_stats.csv'} ({len(out)} features; "
          f"skipped {out.loc[~ok, 'feature'].tolist()})")
    print(f"  Q-Q CCC median {out.loc[ok, 'qq_ccc'].median():.3f}; "
          f">= {CCC_OK}: {(out['qq_ccc'] >= CCC_OK).sum()}, "
          f"< {CCC_WARN}: {(out['qq_ccc'] < CCC_WARN).sum()} | KS median "
          f"{out.loc[ok, 'ks_raw'].median():.3f} (raw) -> {out.loc[ok, 'ks_rz'].median():.3f} (robust z)")
    summarize_normalized(out, plots=plots)
    return out


def summarize_normalized(out: pd.DataFrame, plots: bool = True) -> pd.DataFrame:
    """Score statistics of the normalised shape comparison.

    Writes results/normalized/family_summary.csv (one row per family plus 'all':
    CCC quartiles, the share of features in each CCC tier, KS before/after) and,
    if `plots`, results/normalized/ccc_summary.png (A: CCC histogram by family,
    B: tier shares per family, C: CCC vs KS after normalisation).
    """
    ok = out[out["qq_ccc"].notna()].copy()
    ok["tier"] = np.where(ok["qq_ccc"] >= CCC_OK, "same",
                          np.where(ok["qq_ccc"] >= CCC_WARN, "moderate", "different"))
    rows = []
    for fam in [*FAMILIES, "all"]:
        sub = ok if fam == "all" else ok[ok["family"] == fam]
        if sub.empty:
            continue
        rows.append({
            "family": fam, "n_features": len(sub),
            "ccc_median": sub["qq_ccc"].median(),
            "ccc_q25": sub["qq_ccc"].quantile(0.25), "ccc_q75": sub["qq_ccc"].quantile(0.75),
            "ccc_min": sub["qq_ccc"].min(),
            "n_same": int((sub["tier"] == "same").sum()),
            "n_moderate": int((sub["tier"] == "moderate").sum()),
            "n_different": int((sub["tier"] == "different").sum()),
            "frac_same": (sub["tier"] == "same").mean(),
            "frac_moderate": (sub["tier"] == "moderate").mean(),
            "frac_different": (sub["tier"] == "different").mean(),
            "ks_raw_median": sub["ks_raw"].median(), "ks_rz_median": sub["ks_rz"].median(),
            "ccc_ks_spearman": sub[["qq_ccc", "ks_rz"]].corr("spearman").iloc[0, 1]
            if len(sub) > 2 else np.nan,
        })
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(NORM_DIR / "family_summary.csv", index=False)
    if not plots:
        return summ

    fams = [f for f in FAMILIES if f in set(ok["family"])]
    tier_col = {"same": "#2a9d60", "moderate": "#d97706", "different": "#d63939"}
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.25, 1, 1.1], "wspace": 0.32})

    # A: CCC histogram, stacked by family
    bins = np.linspace(min(0.0, ok["qq_ccc"].min()), 1.0, 41)
    bottoms = np.zeros(len(bins) - 1)
    for fam in fams:
        h, _ = np.histogram(ok.loc[ok["family"] == fam, "qq_ccc"], bins=bins)
        ax_a.bar(bins[:-1], h, width=np.diff(bins), bottom=bottoms, align="edge",
                 color=_FAM_COLOR[fam], alpha=0.85, label=f"{fam} ({int((ok['family'] == fam).sum())})")
        bottoms += h
    for x, lab in [(CCC_WARN, f"{CCC_WARN}"), (CCC_OK, f"{CCC_OK}")]:
        ax_a.axvline(x, color=INK, lw=1, ls="--")
        ax_a.text(x, ax_a.get_ylim()[1] * 0.97, lab, fontsize=8, ha="right", va="top", rotation=90)
    ax_a.set_xlabel("Q–Q CCC (5th–95th percentiles, robust z)", fontsize=8.5)
    ax_a.set_ylabel("# features", fontsize=8.5)
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")
    ax_a.grid(axis="y", color=GRID, lw=0.5); ax_a.set_axisbelow(True)
    ax_a.set_title("A  Distribution of CCC", loc="left", weight="bold", fontsize=9)

    # B: share of features per tier, per family (+ all)
    labels = [*fams, "all"]
    srows = summ.set_index("family").loc[labels]
    left = np.zeros(len(labels))
    for tier in ("same", "moderate", "different"):
        vals = srows[f"frac_{tier}"].values
        ax_b.barh(range(len(labels)), vals, left=left, color=tier_col[tier], alpha=0.85,
                  label=f"{tier} ({'≥' + str(CCC_OK) if tier == 'same' else str(CCC_WARN) + '–' + str(CCC_OK) if tier == 'moderate' else '<' + str(CCC_WARN)})")
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.06:
                ax_b.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax_b.set_yticks(range(len(labels)))
    ax_b.set_yticklabels([f"{f}\n(n={int(srows.loc[f, 'n_features'])})" for f in labels], fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1)
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_b.set_xlabel("Share of features", fontsize=8.5)
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax_b.set_title("B  Shape similarity tiers", loc="left", weight="bold", fontsize=9)

    # C: CCC vs KS after normalisation
    for fam in fams:
        s_ = ok[ok["family"] == fam]
        ax_c.scatter(s_["ks_rz"], s_["qq_ccc"], s=14, color=_FAM_COLOR[fam], alpha=0.6,
                     edgecolors="none", label=fam)
    for y in (CCC_OK, CCC_WARN):
        ax_c.axhline(y, color=MUTED, lw=0.8, ls="--")
    ax_c.axvline(0.3, color=MUTED, lw=0.8, ls=":")
    rho = ok[["qq_ccc", "ks_rz"]].corr("spearman").iloc[0, 1]
    ax_c.set_xlabel("KS after robust z", fontsize=8.5)
    ax_c.set_ylabel("Q–Q CCC", fontsize=8.5)
    ax_c.legend(frameon=False, fontsize=7.5, loc="lower left", markerscale=1.4)
    ax_c.grid(color=GRID, lw=0.4); ax_c.set_axisbelow(True)
    ax_c.set_title(f"C  CCC vs KS (rank correlation {rho:.2f})", loc="left", weight="bold", fontsize=9)

    fig.savefig(NORM_DIR / "ccc_summary.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  Score statistics: ccc_summary.png, family_summary.csv")
    return summ


def generate_family_plots(
    df: pd.DataFrame,
    ic_subj: pd.DataFrame,
    np_subj: pd.DataFrame,
    families: list | None = None,
) -> None:
    """Cross-family overview, transfer-risk heatmap and per-family summaries.
    If `families` is given, only those per-family summaries are redrawn."""
    PLOTS_FAMILY.mkdir(parents=True, exist_ok=True)
    print("\n── Generating family overview plots ──────────────────────────────────")
    if families is None:
        plot_cross_family_overview(df)
        print("  Computing transfer risk metrics...")
        plot_domain_auc(df, ic_subj, np_subj)
        plot_ic_subgroups(df, ic_subj, np_subj)
        families = ["all", *df["family"].unique()]
    for fam in families:
        if fam == "all":
            print("  All-features combined summary...")
            plot_all_features_summary(df, ic_subj, np_subj)
        else:
            plot_family_summary(df, ic_subj, np_subj, fam)
    print(f"  Family plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
