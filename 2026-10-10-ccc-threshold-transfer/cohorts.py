"""Shared data loading and CCC feature selection for this analysis.

Imported by bidirectional.py and random_subset.py: cohort
loaders (90+ Whisper, I-CONECT DeepSpeech2), subject-level aggregation of the
four feature families, the two-cohort availability filter, and Q-Q CCC
selection from the 2026-10-02 feature-distribution analysis.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── repo imports ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from lmc.features.acoustic.names import recording_acoustic_columns   # noqa: E402
from lmc.features.temporal.names import TEMPORAL_FEATURE_NAMES as TEMP  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
LING_CSV = _REPO / "lmc/features/linguistic/iconect99.csv"
CCC_CSV = (
    _REPO / "docs/analysis/cross-cohort/2026-10-02-feature-distributions"
    / "results/normalized/feature_shape_stats.csv"
)


def _load_ling_feature_names() -> list[str]:
    """Load the canonical 99 names without importing the LIWC extractor."""
    spec = pd.read_csv(LING_CSV).sort_values("idx").reset_index(drop=True)
    duplicate = spec["feature_name"].duplicated(keep=False)
    names = []
    for name, computation, is_duplicate in zip(
        spec["feature_name"], spec["computation"], duplicate
    ):
        if is_duplicate and "lex_div." in computation:
            measure = computation.split("lex_div.", 1)[1].split("(", 1)[0]
            names.append(f"{name} [{measure}]")
        else:
            names.append(name)
    if len(names) != len(set(names)) or len(names) != 99:
        raise ValueError(f"expected 99 unique linguistic features, got {len(names)}")
    return names


LING = _load_ling_feature_names()

# ── constants ─────────────────────────────────────────────────────────────────
LABEL_MAP      = {"normal": 0, "mci": 1}
CCC_THRESHOLDS = [None, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95]   # None = no CCC filter (all features)
# full grid: the original thresholds plus finer cut-offs at the top end. Later thresholds are
# appended so anything seeded by grid position (validity.py random draws) keeps its seeds.
THRESHOLDS     = CCC_THRESHOLDS + [0.96, 0.97, 0.98, 0.99, 0.995]
SEEDS          = list(range(10))
MIN_NONNAN_LING = 0.80   # ling / temporal / semantic
MIN_NONNAN_AC   = 0.50   # acoustic (I-CONECT coverage ≈54%)
N_JOBS          = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))   # parallel CV seeds

SEMANTIC_COLS = [
    "sm_content_density",
    "sm_content_diversity",
    "sm_content_freq",
    "sm_semantic_spread",
]
ACOUSTIC_COLS = recording_acoustic_columns()   # 420 cols
ACOUSTIC_SET  = set(ACOUSTIC_COLS)

KEY = ["dataset", "subject_id", "visit_id", "transcript_id"]


# ── data loading ──────────────────────────────────────────────────────────────

def _merge_keys(df: pd.DataFrame) -> list[str]:
    return [k for k in KEY if k in df.columns]


def _load_subject_features(
    base: Path,
    asr: str,
    ling_cols: list[str],
    temp_cols: list[str],
    sem_cols: list[str],
    ac_cols: list[str],
    usable: set | None = None,
) -> pd.DataFrame:
    """Load all four feature families and aggregate to subject level (mean)."""

    # ── ling ──────────────────────────────────────────────────────────────────
    ling = pd.read_parquet(base / "recording_features.parquet")
    ling = ling[ling["asr_config"] == asr]
    if usable:
        ling = ling[ling["transcript_id"].isin(usable)]

    # ── temporal ──────────────────────────────────────────────────────────────
    temp = pd.read_parquet(base / f"recording_temporal_{asr}.parquet")
    if usable:
        temp = temp[temp["transcript_id"].isin(usable)]
    mk = _merge_keys(temp)
    merged = ling.merge(
        temp[mk + [c for c in temp_cols if c in temp.columns]],
        on=mk, how="inner",
    )

    # ── semantic (optional table) ─────────────────────────────────────────────
    sem_path = base / f"recording_semantic_{asr}.parquet"
    if sem_path.exists() and sem_cols:
        sem = pd.read_parquet(sem_path)
        if usable:
            sem = sem[sem["transcript_id"].isin(usable)]
        avail = [c for c in sem_cols if c in sem.columns]
        if avail:
            merged = merged.merge(
                sem[_merge_keys(sem) + avail],
                on=_merge_keys(sem), how="left",
            )

    # ── aggregate ling + temporal + semantic to subject ───────────────────────
    non_ac = [c for c in ling_cols + temp_cols + sem_cols if c in merged.columns]
    subj = merged.groupby("subject_id")[non_ac].mean()

    # ── acoustic (join at subject level) ─────────────────────────────────────
    ac_path = base / f"recording_acoustic_{asr}.parquet"
    if ac_path.exists() and ac_cols:
        ac = pd.read_parquet(ac_path)
        if usable:
            ac = ac[ac["transcript_id"].isin(usable)]
        avail_ac = [c for c in ac_cols if c in ac.columns]
        if avail_ac:
            ac_subj = ac.groupby("subject_id")[avail_ac].mean()
            subj = subj.join(ac_subj, how="left")

    return subj


def load_90plus(paths: dict) -> tuple[pd.DataFrame, pd.Series]:
    base = Path(paths["local_data_dir"]) / "processed" / "the90plus"

    rm = pd.read_parquet(base / "role_map.parquet")
    usable = set(rm.loc[rm["role_usable"], "transcript_id"])

    X = _load_subject_features(
        base, "whisper-lv3-pyannote",
        list(LING), list(TEMP), SEMANTIC_COLS, ACOUSTIC_COLS, usable,
    )

    vm = pd.read_parquet(base / "visit_meta.parquet")
    vm = vm[vm["diagnosis_group"].isin({"normal", "mci"})]
    label_mode = (
        vm.groupby("subject_id")["diagnosis_group"]
          .agg(lambda s: s.mode().iloc[0])
          .map(LABEL_MAP)
    )

    X = X.loc[X.index.isin(label_mode.index)]
    y = label_mode.reindex(X.index).dropna().astype(int)
    X = X.loc[y.index]

    print(f"  90+    : {len(X)} subjects | "
          f"MCI={int(y.sum())} ({y.mean():.1%}) | features={X.shape[1]}")
    return X, y


def load_iconect(paths: dict) -> tuple[pd.DataFrame, pd.Series]:
    base = Path(paths["local_data_dir"]) / "processed" / "iconect"

    X = _load_subject_features(
        base, "sam_v0_ds2",
        list(LING), list(TEMP), SEMANTIC_COLS, ACOUSTIC_COLS,
    )

    labels = pd.read_parquet(base / "mci_label_baseline.parquet")
    labels = labels[labels["mci_label"].isin({"normal", "mci"})]
    label_s = labels.set_index("subject_id")["mci_label"].map(LABEL_MAP)

    X = X.loc[X.index.isin(label_s.index)]
    y = label_s.reindex(X.index).dropna().astype(int)
    X = X.loc[y.index]

    print(f"  I-CONECT: {len(X)} subjects | "
          f"MCI={int(y.sum())} ({y.mean():.1%}) | features={X.shape[1]}")
    return X, y


# ── CCC feature selection ─────────────────────────────────────────────────────

def load_ccc_table() -> pd.DataFrame:
    df = pd.read_csv(CCC_CSV)
    df["qq_ccc"] = pd.to_numeric(df["qq_ccc"], errors="coerce")
    return df


def select_by_ccc(ccc_df: pd.DataFrame, thresh: float | None,
                  available: list[str]) -> list[str]:
    if thresh is None:
        return list(available)
    passing = set(ccc_df.loc[ccc_df["qq_ccc"] >= thresh, "feature"])
    return [f for f in available if f in passing]


def availability_filter(X_90: pd.DataFrame, X_ic: pd.DataFrame) -> list[str]:
    """Keep features with sufficient non-NaN coverage in both cohorts."""
    common = [f for f in X_90.columns if f in X_ic.columns]

    def ok(f: str) -> bool:
        thr = MIN_NONNAN_AC if f in ACOUSTIC_SET else MIN_NONNAN_LING
        return (X_90[f].notna().mean() >= thr and
                X_ic[f].notna().mean() >= thr)

    return [f for f in common if ok(f)]


def _label(thresh: float | None) -> str:
    return "all features" if thresh is None else f"CCC≥{thresh:g}"


def subset_name(thresh: float | None) -> str:
    """File/label stem: all, ccc_0.50, …, ccc_0.99, ccc_0.995."""
    if thresh is None or pd.isna(thresh):
        return "all"
    return f"ccc_{thresh:.2f}" if round(thresh, 2) == thresh else f"ccc_{thresh:.3f}"


# ── progress ──────────────────────────────────────────────────────────────────

class Progress:
    """Print `[done/total pct | elapsed | ETA] msg` lines (flushed, log-friendly)."""

    def __init__(self, total: int):
        self.total, self.done, self.t0 = total, 0, time.time()

    def step(self, msg: str) -> None:
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done)
        print(f"[{self.done}/{self.total} {self.done / self.total:4.0%} | "
              f"{el / 60:5.1f}m elapsed | ETA {eta / 60:5.1f}m] {msg}", flush=True)
