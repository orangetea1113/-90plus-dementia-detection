"""CCC-threshold feature selection for cross-cohort transfer: 90+ → I-CONECT.

Hypothesis
----------
Features whose normalized distributions are most similar between cohorts
(high Q-Q CCC after per-cohort robust z-score) should yield better
cross-cohort transfer.

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/run.py

Run from the repository root on Great Lakes (see submit.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── repo imports ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

# Add this analysis dir so sibling modules (models, evaluate) are importable
sys.path.insert(0, str(Path(__file__).parent))

from lmc import io                                                   # noqa: E402
from lmc.features.acoustic.names import recording_acoustic_columns   # noqa: E402
from lmc.features.temporal.names import TEMPORAL_FEATURE_NAMES as TEMP  # noqa: E402
from models import MODELS                                            # noqa: E402
from evaluate import cv_on_90plus, transfer_to_iconect               # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
LING_CSV = _REPO / "lmc/features/linguistic/iconect99.csv"
RESULTS = HERE / "results"
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
    return "all features" if thresh is None else f"CCC≥{thresh:.2f}"


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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--transfer-only", action="store_true",
                    help="skip 90+ CV; refit transfer models and write only "
                         "transfer_predictions.csv (existing CSVs untouched)")
    args = ap.parse_args()
    seeds = [] if args.transfer_only else SEEDS

    paths = io.load_paths(args.paths)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "feature_subsets").mkdir(exist_ok=True)

    print("=" * 60)
    print("Loading features & labels…")
    print("=" * 60)
    X_90, y_90 = load_90plus(paths)
    X_ic, y_ic = load_iconect(paths)

    avail = availability_filter(X_90, X_ic)
    X_90, X_ic = X_90[avail], X_ic[avail]
    print(f"\nCommon features after availability filter: {len(avail)}")

    ccc_df = load_ccc_table()
    print(f"Features with CCC data: "
          f"{ccc_df['feature'].isin(set(avail)).sum()}/{len(avail)}")

    ling_set = set(LING)
    temp_set = set(TEMP)
    sem_set  = set(SEMANTIC_COLS)

    cv_rows, tr_rows, pred_rows = [], [], []
    # one step per CV seed + one per transfer fit, for every (threshold, model)
    prog = Progress(len(CCC_THRESHOLDS) * len(MODELS) * (len(seeds) + 1))

    print("\n" + "=" * 60)
    for thresh in CCC_THRESHOLDS:
        sel = select_by_ccc(ccc_df, thresh, avail)
        if not sel:
            print(f"{_label(thresh)}: no features — skipped")
            continue

        (RESULTS / "feature_subsets" / ("all.txt" if thresh is None else f"ccc_{thresh:.2f}.txt")).write_text(
            "\n".join(sel)
        )

        n_l = sum(1 for f in sel if f in ling_set)
        n_t = sum(1 for f in sel if f in temp_set)
        n_s = sum(1 for f in sel if f in sem_set)
        n_a = sum(1 for f in sel if f in ACOUSTIC_SET)
        print(f"\n{_label(thresh)} | {len(sel)} features "
              f"(ling={n_l} temp={n_t} sem={n_s} ac={n_a})")

        X90_s = X_90[sel]
        Xic_s = X_ic[sel]

        for mname, mfn in MODELS.items():
            seed_aucs = []
            # seeds run in parallel; results are yielded in seed order
            results = Parallel(n_jobs=N_JOBS, return_as="generator")(
                delayed(cv_on_90plus)(X90_s, y_90, seed, mfn) for seed in seeds
            )
            for seed, res in zip(seeds, results):
                seed_aucs.append(res["oof_auc"])
                prog.step(f"{_label(thresh)} {mname} CV seed {seed} "
                          f"AUC={res['oof_auc']:.3f}")
                cv_rows.append({
                    "ccc_threshold": thresh,
                    "n_features":    len(sel),
                    "model":         mname,
                    "seed":          seed,
                    **{k: res[k] for k in ("oof_auc", "ci_lo", "ci_hi", "bal_acc")},
                    "n_ling": n_l, "n_temp": n_t, "n_sem": n_s, "n_ac": n_a,
                })

            cv_mu  = float(np.mean(seed_aucs)) if seed_aucs else np.nan
            cv_std = float(np.std(seed_aucs)) if seed_aucs else np.nan

            tr = transfer_to_iconect(X90_s, y_90, Xic_s, y_ic, mfn)
            pred_rows.append(pd.DataFrame({
                "ccc_threshold": thresh,
                "model":         mname,
                "subject_id":    y_ic.index,
                "y":             y_ic.to_numpy(),
                "proba":         tr["proba"],
            }))
            tr_rows.append({
                "ccc_threshold": thresh,
                "n_features":    len(sel),
                "model":         mname,
                "cv_auc_mean":   cv_mu,
                "cv_auc_std":    cv_std,
                **{k: tr[k] for k in ("transfer_auc", "ci_lo", "ci_hi",
                                       "bal_acc", "n_iconect", "n_mci_iconect")},
                "n_ling": n_l, "n_temp": n_t, "n_sem": n_s, "n_ac": n_a,
            })

            prog.step(f"{_label(thresh)} {mname} transfer → I-CONECT "
                      f"AUC={tr['transfer_auc']:.3f}")
            print(f"  {mname:>11s}  90+ CV={cv_mu:.3f}±{cv_std:.3f}  "
                  f"ICONECT={tr['transfer_auc']:.3f} "
                  f"[{tr['ci_lo']:.3f}, {tr['ci_hi']:.3f}]")

        # checkpoint after every threshold so a timeout keeps finished results
        pd.concat(pred_rows).to_csv(RESULTS / "transfer_predictions.csv", index=False)
        if not args.transfer_only:
            pd.DataFrame(cv_rows).to_csv(RESULTS / "cv_results.csv",       index=False)
            pd.DataFrame(tr_rows).to_csv(RESULTS / "transfer_results.csv", index=False)
            pd.DataFrame(tr_rows).to_csv(RESULTS / "summary.csv",          index=False)

    print(f"\nDone. Results → {RESULTS}/")


if __name__ == "__main__":
    main()
