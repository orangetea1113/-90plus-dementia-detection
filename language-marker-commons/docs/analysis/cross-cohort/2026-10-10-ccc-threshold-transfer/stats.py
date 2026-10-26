"""Significance tests on the I-CONECT transfer predictions.

Reads the 90+ → I-CONECT rows of results/bidirectional_transfer_predictions.csv
(written by bidirectional.py) and writes:

  results/permutation.csv       transfer AUC vs chance (label permutation)
  results/paired_bootstrap.csv  AUC(threshold) − AUC(all features), paired

The paired bootstrap resamples the same I-CONECT subjects for every
feature subset, so subject-level difficulty cancels in the difference.

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE    = Path(__file__).parent
RESULTS = HERE / "results"

N_BOOT = 2000
N_PERM = 5000
SEED   = 0
ALL    = "all"   # label for the no-filter baseline


def _subset(thresh: float) -> str:
    if pd.isna(thresh):
        return ALL
    return f"ccc_{thresh:.2f}" if round(thresh, 2) == thresh else f"ccc_{thresh:.3f}"


def _read_90plus_to_iconect(name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / name)
    df = df[(df["source"] == "90plus") & (df["target"] == "iconect")]
    return df.assign(subset=df["ccc_threshold"].map(_subset))


def rank_auc(y: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC from precomputed score ranks.

    y may be 1-D or (n_draws, n) so permuted labels are scored at once.
    """
    n1 = y.sum(axis=-1)
    n0 = y.shape[-1] - n1
    return ((y * ranks).sum(axis=-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ── load ──────────────────────────────────────────────────────────────────────

def load_predictions() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Return (y, {model: {subset: proba}}) with subjects aligned across subsets."""
    df = _read_90plus_to_iconect("bidirectional_transfer_predictions.csv")

    wide = df.pivot_table(index="subject_id", columns=["model", "subset"],
                          values="proba")
    y = df.groupby("subject_id")["y"].first().reindex(wide.index).to_numpy()
    if wide.isna().any().any():
        raise ValueError("subjects differ across subsets — cannot pair")

    preds: dict[str, dict[str, np.ndarray]] = {}
    for model, subset in wide.columns:
        preds.setdefault(model, {})[subset] = wide[(model, subset)].to_numpy()
    return y, preds


def check_against_summary(y: np.ndarray, preds: dict) -> None:
    """Predictions should reproduce the AUCs in transfer_results; warn if not."""
    summ = _read_90plus_to_iconect("bidirectional_transfer_results.csv")
    for _, r in summ.iterrows():
        p = preds.get(r["model"], {}).get(r["subset"])
        if p is None:
            continue
        auc = float(rank_auc(y, rankdata(p)))
        if abs(auc - r["transfer_auc"]) > 1e-6:
            print(f"WARNING {r['model']} {r['subset']}: refit AUC {auc:.4f} "
                  f"≠ transfer_results {r['transfer_auc']:.4f}")


# ── tests ─────────────────────────────────────────────────────────────────────

def permutation_test(y: np.ndarray, preds: dict) -> pd.DataFrame:
    """One-sided p-value for AUC > 0.5 by permuting I-CONECT labels."""
    rng = np.random.default_rng(SEED)
    y_perm = np.stack([rng.permutation(y) for _ in range(N_PERM)])

    rows = []
    for model, subsets in preds.items():
        for subset, p in subsets.items():
            ranks = rankdata(p)
            obs   = float(rank_auc(y, ranks))
            null  = rank_auc(y_perm, ranks)
            rows.append({
                "model":      model,
                "subset":     subset,
                "auc":        obs,
                "null_95":    float(np.percentile(null, 95)),
                "p_perm":     (1 + int((null >= obs).sum())) / (1 + N_PERM),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(y: np.ndarray, preds: dict) -> pd.DataFrame:
    """CI and two-sided p for AUC(subset) − AUC(all), same resamples throughout."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    draws = []
    while len(draws) < N_BOOT:
        b = rng.integers(0, n, n)
        if 0 < y[b].sum() < n:          # need both classes
            draws.append(b)

    rows = []
    for model, subsets in preds.items():
        boot = {s: np.array([rank_auc(y[b], rankdata(p[b])) for b in draws])
                for s, p in subsets.items()}
        base_obs = float(rank_auc(y, rankdata(subsets[ALL])))
        for subset, p in subsets.items():
            if subset == ALL:
                continue
            diff = boot[subset] - boot[ALL]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            # two-sided bootstrap p: how often the difference crosses zero
            p_two = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            rows.append({
                "model":      model,
                "subset":     subset,
                "auc":        float(rank_auc(y, rankdata(p))),
                "auc_all":    base_obs,
                "diff":       float(rank_auc(y, rankdata(p))) - base_obs,
                "diff_ci_lo": float(lo),
                "diff_ci_hi": float(hi),
                "p_boot":     float(p_two),
            })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    y, preds = load_predictions()
    print(f"I-CONECT: n={len(y)}  MCI={int(y.sum())}  "
          f"models={list(preds)}  subsets={len(next(iter(preds.values())))}")
    check_against_summary(y, preds)

    perm = permutation_test(y, preds)
    perm.to_csv(RESULTS / "permutation.csv", index=False)
    print(f"\nPermutation test (AUC > 0.5, {N_PERM} perms)")
    print(perm.to_string(index=False, float_format="%.3f"))

    boot = paired_bootstrap(y, preds)
    boot.to_csv(RESULTS / "paired_bootstrap.csv", index=False)
    print(f"\nPaired bootstrap vs all features ({N_BOOT} resamples, uncorrected)")
    print(boot.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
