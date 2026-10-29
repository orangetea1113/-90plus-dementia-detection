"""Within-cohort baselines + bidirectional transfer: 90+ ⇄ I-CONECT.

Fills the full train × test matrix for every (CCC threshold, model):

    train \\ test     90+                 I-CONECT
    90+               CV (within)         transfer 90+ → IC
    I-CONECT          transfer IC → 90+   CV (within)

The diagonal (5-fold StratifiedGroupKFold × 10 seeds, fold-wise robust
z-score) is the within-cohort ceiling each transfer AUC should be read
against. Off-diagonal cells fit on all of one cohort and predict all of the
other, each cohort normalized by its own median/IQR.

Loaders, availability filter and CCC selection come from cohorts.py.

Usage (from the repo root)
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/bidirectional.py

Writes results/bidirectional_{cv_results,transfer_results,summary,
transfer_predictions}.csv and feature_subsets/ccc_<thr>.txt.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import MODELS, cv_within, transfer   # noqa: E402
from cohorts import (                                                 # noqa: E402
    ACOUSTIC_SET, LING, N_JOBS, SEEDS, SEMANTIC_COLS, TEMP, THRESHOLDS,
    Progress, _label, availability_filter, subset_name, load_90plus, load_ccc_table,
    load_iconect, select_by_ccc,
)
from lmc import io                                                # noqa: E402  (repo path set by cohorts)

OUT = Path(__file__).parent / "results"

COHORTS = ("90plus", "iconect")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated CCC thresholds overriding the default grid")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of CV seeds per cohort (default 10)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    thresholds = ([float(t) for t in args.thresholds.split(",")]
                  if args.thresholds else THRESHOLDS)
    
    paths = io.load_paths(args.paths)
    (OUT / "feature_subsets").mkdir(parents=True, exist_ok=True)

    print("Loading features & labels…")
    X_90, y_90 = load_90plus(paths)
    X_ic, y_ic = load_iconect(paths)
    avail = availability_filter(X_90, X_ic)
    data = {"90plus": (X_90[avail], y_90), "iconect": (X_ic[avail], y_ic)}
    print(f"Common features after availability filter: {len(avail)}")

    ccc_df = load_ccc_table()
    ling_set, temp_set, sem_set = set(LING), set(TEMP), set(SEMANTIC_COLS)

    cv_rows, tr_rows, pred_rows, summ_rows = [], [], [], []
    # per (threshold, model): seeds × 2 cohorts CV + 2 transfer fits
    prog = Progress(len(thresholds) * len(MODELS) * (2 * len(seeds) + 2))

    for thresh in thresholds:
        sel = select_by_ccc(ccc_df, thresh, avail)
        if not sel:
            print(f"{_label(thresh)}: no features — skipped")
            continue
        (OUT / "feature_subsets" / f"{subset_name(thresh)}.txt").write_text("\n".join(sel))
        fam = {
            "n_ling": sum(f in ling_set for f in sel),
            "n_temp": sum(f in temp_set for f in sel),
            "n_sem":  sum(f in sem_set for f in sel),
            "n_ac":   sum(f in ACOUSTIC_SET for f in sel),
        }
        print(f"\n{_label(thresh)} | {len(sel)} features {fam}")

        for mname, mfn in MODELS.items():
            cell = {}   # (train, test) → AUC (CV mean on the diagonal)

            # ── diagonal: within-cohort CV ────────────────────────────────────
            for cohort in COHORTS:
                X, y = data[cohort]
                X = X[sel]
                results = Parallel(n_jobs=N_JOBS, return_as="generator")(
                    delayed(cv_within)(X, y, seed, mfn) for seed in seeds
                )
                aucs = []
                for seed, res in zip(seeds, results):
                    aucs.append(res["oof_auc"])
                    prog.step(f"{_label(thresh)} {mname} CV {cohort} seed {seed} "
                              f"AUC={res['oof_auc']:.3f}")
                    cv_rows.append({
                        "cohort": cohort, "ccc_threshold": thresh,
                        "n_features": len(sel), "model": mname, "seed": seed,
                        **{k: res[k] for k in ("oof_auc", "ci_lo", "ci_hi", "bal_acc")},
                        "n_subjects": len(y), "n_mci": int(y.sum()), **fam,
                    })
                cell[(cohort, cohort)] = (float(np.mean(aucs)), float(np.std(aucs)))

            # ── off-diagonal: transfer in both directions ─────────────────────
            for src, tgt in (("90plus", "iconect"), ("iconect", "90plus")):
                Xs, ys = data[src]
                Xt, yt = data[tgt]
                tr = transfer(Xs[sel], ys, Xt[sel], yt, mfn)
                prog.step(f"{_label(thresh)} {mname} transfer {src} → {tgt} "
                          f"AUC={tr['transfer_auc']:.3f}")
                tr_rows.append({
                    "source": src, "target": tgt, "ccc_threshold": thresh,
                    "n_features": len(sel), "model": mname,
                    **{k: tr[k] for k in ("transfer_auc", "ci_lo", "ci_hi", "bal_acc",
                                           "n_target", "n_mci_target")},
                    "n_source": len(ys), **fam,
                })
                pred_rows.append(pd.DataFrame({
                    "source": src, "target": tgt, "ccc_threshold": thresh,
                    "model": mname, "subject_id": yt.index,
                    "y": yt.to_numpy(), "proba": tr["proba"],
                }))
                cell[(src, tgt)] = (tr["transfer_auc"], np.nan)

            # ── one wide summary row per (threshold, model) ───────────────────
            r = {"ccc_threshold": thresh, "n_features": len(sel), "model": mname}
            for (src, tgt), (auc, sd) in cell.items():
                key = f"{src}_cv" if src == tgt else f"{src}_to_{tgt}"
                r[f"{key}_auc"] = auc
                if src == tgt:
                    r[f"{key}_std"] = sd
            # transfer gap: target-cohort CV ceiling minus transfer AUC
            r["gap_to_iconect"] = r["iconect_cv_auc"] - r["90plus_to_iconect_auc"]
            r["gap_to_90plus"]  = r["90plus_cv_auc"]  - r["iconect_to_90plus_auc"]
            summ_rows.append(r)

            print(f"  {mname:>11s}  CV 90+={r['90plus_cv_auc']:.3f}  "
                  f"CV IC={r['iconect_cv_auc']:.3f}  "
                  f"90+→IC={r['90plus_to_iconect_auc']:.3f}  "
                  f"IC→90+={r['iconect_to_90plus_auc']:.3f}")

        # checkpoint after every threshold
        pd.DataFrame(cv_rows).to_csv(OUT / "bidirectional_cv_results.csv", index=False)
        pd.DataFrame(tr_rows).to_csv(OUT / "bidirectional_transfer_results.csv", index=False)
        pd.DataFrame(summ_rows).to_csv(OUT / "bidirectional_summary.csv", index=False)
        pd.concat(pred_rows).to_csv(OUT / "bidirectional_transfer_predictions.csv", index=False)

    print(f"\nDone. Results → {OUT}/")


if __name__ == "__main__":
    main()
