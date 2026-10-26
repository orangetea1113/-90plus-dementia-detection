"""CV and cross-cohort transfer evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5


# ── robust z-score ────────────────────────────────────────────────────────────

def rz_fit(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute per-feature median and IQR from X.

    Features with IQR == 0 have their IQR set to NaN so that
    rz_transform fills them with 0 (constant → no information).
    """
    med = X.median(skipna=True)
    iqr = X.quantile(0.75) - X.quantile(0.25)
    iqr[iqr == 0] = np.nan
    return med, iqr


def rz_transform(X: pd.DataFrame, med: pd.Series, iqr: pd.Series) -> np.ndarray:
    """Apply (X - median) / IQR; remaining NaN → 0."""
    return ((X - med) / iqr).fillna(0.0).to_numpy(dtype=float)


# ── AUC with bootstrap CI ─────────────────────────────────────────────────────

def bootstrap_auc(
    y: np.ndarray,
    proba: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (AUC, ci_lo_2.5%, ci_hi_97.5%) via non-parametric bootstrap."""
    auc = roc_auc_score(y, proba)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boot = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            boot.append(roc_auc_score(y[b], proba[b]))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return float(auc), float(lo), float(hi)


# ── within-cohort cross-validation ────────────────────────────────────────────

def cv_within(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    model_fn,
    return_oof: bool = False,
) -> dict:
    """StratifiedGroupKFold CV within one cohort.

    Groups are subject_id values (X.index). Robust z-score normalization is
    fit on the training fold only to avoid leakage.

    Returns
    -------
    dict with keys: oof_auc, ci_lo, ci_hi, bal_acc (+ oof if return_oof)
    """
    groups = X.index.to_numpy()
    y_arr  = y.to_numpy()
    skf    = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_p  = np.zeros(len(y_arr))

    for train_idx, test_idx in skf.split(X, y_arr, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr        = y_arr[train_idx]

        med, iqr = rz_fit(X_tr)
        Xtr_n    = rz_transform(X_tr, med, iqr)
        Xte_n    = rz_transform(X_te, med, iqr)

        clf = model_fn(seed)
        clf.fit(Xtr_n, y_tr)
        oof_p[test_idx] = clf.predict_proba(Xte_n)[:, 1]

    auc, lo, hi = bootstrap_auc(y_arr, oof_p)
    bal = balanced_accuracy_score(y_arr, (oof_p >= 0.5).astype(int))
    out = {"oof_auc": auc, "ci_lo": lo, "ci_hi": hi, "bal_acc": bal}
    if return_oof:
        out["oof"] = oof_p
    return out


# ── cross-cohort transfer ─────────────────────────────────────────────────────

def transfer(
    X_src: pd.DataFrame,
    y_src: pd.Series,
    X_tgt: pd.DataFrame,
    y_tgt: pd.Series,
    model_fn,
    seed: int = 0,
) -> dict:
    """Fit on all of the source cohort, evaluate on all of the target cohort.

    Each cohort is normalized by its own median/IQR (consistent with how the
    Q-Q CCC was computed in the 2026-10-02 analysis).

    Returns
    -------
    dict with keys: transfer_auc, ci_lo, ci_hi, bal_acc, n_target,
                    n_mci_target, proba (per-subject P(MCI), aligned with
                    y_tgt.index)
    """
    med_s, iqr_s = rz_fit(X_src)
    Xs_n = rz_transform(X_src, med_s, iqr_s)

    med_t, iqr_t = rz_fit(X_tgt)
    Xt_n = rz_transform(X_tgt, med_t, iqr_t)

    clf = model_fn(seed)
    clf.fit(Xs_n, y_src.to_numpy())
    p_t = clf.predict_proba(Xt_n)[:, 1]

    y_t = y_tgt.to_numpy()
    auc, lo, hi = bootstrap_auc(y_t, p_t)
    bal = balanced_accuracy_score(y_t, (p_t >= 0.5).astype(int))
    return {
        "transfer_auc": auc,
        "ci_lo":        lo,
        "ci_hi":        hi,
        "bal_acc":      bal,
        "n_target":     len(y_t),
        "n_mci_target": int(y_t.sum()),
        "proba":        p_t,
    }
