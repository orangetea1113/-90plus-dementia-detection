"""Size-matched random-subset control for the CCC-threshold transfer experiment.

Question
--------
Does a CCC-selected feature subset transfer 90+ → I-CONECT better than a
random subset of the same size? For each threshold, draw N random subsets
and refit the same transfer model on each; the CCC subset's AUC is then
ranked against that null distribution.

Two draw modes:
  stratified    same per-family counts (ling/temp/sem/ac) as the CCC subset
                → is CCC choosing better features *within* each family?
  unstratified  same total count, drawn from all features
                → does CCC selection beat random, family shift included?

The same random subsets are used for SGBT and Elastic Net.

Usage
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/random_subset.py
        [--n-draws 200]

Writes results/random_subset_{auc,summary}.csv.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import rz_fit, rz_transform                  # noqa: E402
from models import MODELS                                  # noqa: E402
from cohorts import (                                          # noqa: E402
    ACOUSTIC_SET, CCC_THRESHOLDS, LING, N_JOBS, SEMANTIC_COLS, TEMP, Progress,
    availability_filter, load_90plus, load_ccc_table, load_iconect, select_by_ccc,
)
from lmc import io                                         # noqa: E402  (repo path set by cohorts)

HERE  = Path(__file__).parent
OUT   = HERE / "results"
MODES = ["stratified", "unstratified"]
TRANSFER_SEED = 0   # same model seed as bidirectional.py's transfer fit


def _family(f: str, ling: set, temp: set, sem: set) -> str:
    if f in ling:
        return "ling"
    if f in temp:
        return "temp"
    if f in sem:
        return "sem"
    if f in ACOUSTIC_SET:
        return "ac"
    raise ValueError(f"unknown family for {f}")


def draw_subset(rng: np.random.Generator, mode: str, sel: list[str],
                avail: list[str], fam: dict[str, str]) -> list[str]:
    """Random subset matching `sel` in total size (and family counts if stratified)."""
    if mode == "unstratified":
        return list(rng.choice(avail, len(sel), replace=False))
    out = []
    for f_name in ("ling", "temp", "sem", "ac"):
        k = sum(1 for f in sel if fam[f] == f_name)
        pool = [f for f in avail if fam[f] == f_name]
        out += list(rng.choice(pool, k, replace=False))
    return out


def fit_auc(X90: pd.DataFrame, y90: np.ndarray, Xic: pd.DataFrame,
            yic: np.ndarray, cols: list[str], model_fn) -> float:
    """Fit on 90+ (pre-normalized), return I-CONECT AUC."""
    clf = model_fn(TRANSFER_SEED)
    clf.fit(X90[cols].to_numpy(), y90)
    return float(roc_auc_score(yic, clf.predict_proba(Xic[cols].to_numpy())[:, 1]))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, thr, model), g in df.groupby(["mode", "ccc_threshold", "model"]):
        r   = g.loc[g["draw"] >= 0, "auc"].to_numpy()
        obs = float(g.loc[g["draw"] == -1, "auc"].iloc[0])
        rows.append({
            "mode": mode, "ccc_threshold": thr, "model": model,
            "n_features": int(g["n_features"].iloc[0]), "n_draws": len(r),
            "ccc_auc":    obs,
            "rand_mean":  float(r.mean()),
            "rand_sd":    float(r.std()),
            "rand_lo":    float(np.percentile(r, 2.5)),
            "rand_hi":    float(np.percentile(r, 97.5)),
            "pctile":     float((r < obs).mean()),          # share of draws beaten
            "p_one_sided": (1 + int((r >= obs).sum())) / (1 + len(r)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--n-draws", type=int, default=200)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paths = io.load_paths(args.paths)
    X_90, y_90 = load_90plus(paths)
    X_ic, y_ic = load_iconect(paths)
    avail = availability_filter(X_90, X_ic)

    # per-feature robust z-score, each cohort its own stats (as in bidirectional.py) —
    # column-wise, so normalizing once then slicing equals per-subset fitting
    X90n = pd.DataFrame(rz_transform(X_90[avail], *rz_fit(X_90[avail])), columns=avail)
    Xicn = pd.DataFrame(rz_transform(X_ic[avail], *rz_fit(X_ic[avail])), columns=avail)
    y90, yic = y_90.to_numpy(), y_ic.to_numpy()

    ling_set, temp_set, sem_set = set(LING), set(TEMP), set(SEMANTIC_COLS)
    fam = {f: _family(f, ling_set, temp_set, sem_set) for f in avail}
    ccc_df = load_ccc_table()
    thresholds = [t for t in CCC_THRESHOLDS if t is not None]

    prog = Progress(len(MODES) * len(thresholds))
    rows = []
    for mi, mode in enumerate(MODES):
        for ti, thr in enumerate(thresholds):
            sel = select_by_ccc(ccc_df, thr, avail)
            # draw −1 = the CCC subset itself; draws seeded per (mode, threshold, draw)
            subsets = [sel] + [
                draw_subset(np.random.default_rng([mi, ti, d]), mode, sel, avail, fam)
                for d in range(args.n_draws)
            ]
            jobs = [(d - 1, m, cols) for d, cols in enumerate(subsets) for m in MODELS]
            aucs = Parallel(n_jobs=N_JOBS)(
                delayed(fit_auc)(X90n, y90, Xicn, yic, cols, MODELS[m])
                for _, m, cols in jobs
            )
            rows += [{"mode": mode, "ccc_threshold": thr, "model": m, "draw": d,
                      "n_features": len(sel), "auc": a}
                     for (d, m, _), a in zip(jobs, aucs)]

            df = pd.DataFrame(rows)
            df.to_csv(OUT / "random_subset_auc.csv", index=False)   # checkpoint
            s = summarize(df[(df["mode"] == mode) & (df["ccc_threshold"] == thr)])
            prog.step(f"{mode} CCC≥{thr:.2f} n={len(sel)} | " + "  ".join(
                f"{r.model}: CCC={r.ccc_auc:.3f} rand={r.rand_mean:.3f} "
                f"p={r.p_one_sided:.3f}" for r in s.itertuples()))

    summ = summarize(pd.DataFrame(rows))
    summ.to_csv(OUT / "random_subset_summary.csv", index=False)
    print("\n" + summ.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
