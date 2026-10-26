"""Which features does each cohort's model rely on? 90+-trained vs I-CONECT-trained.

Question
--------
The 90+ model transfers to I-CONECT (AUC ≈ 0.60) better than an I-CONECT
model does under its own CV (≈ 0.50). Do the two models weight the same
features? And for the 90+ model, which features carry the signal that
survives transfer, and which ones hurt on I-CONECT?

Importance views (per CCC subset × model)
-----------------------------------------
  imp_90 / imp_ic   OOB permutation importance (AUC drop). Fit on a stratified
                    bootstrap of one cohort, permute each feature on the
                    out-of-bag subjects. Averaged over --n-boot bootstraps.
                    Works the same way for SGBT and Elastic Net.
  coef_90 / coef_ic Elastic Net only: bootstrap-mean coefficient (per IQR-unit
                    after StandardScaler) and selection frequency (coef ≠ 0).
  imp_tr_90_to_ic   Permutation importance of the full-90+ model evaluated on
                    I-CONECT: positive = feature helps the transfer, negative =
                    it misleads on I-CONECT. (imp_tr_ic_to_90 is the reverse.)
  uni_auc_90 / _ic  Single-feature AUC (> 0.5 = higher in MCI), model-free.

Small-n control
---------------
I-CONECT has n=156, so its importances are noisy. A low 90+ vs I-CONECT rank
correlation could come from small n alone. The control redoes the I-CONECT
procedure on --n-ctrl stratified 90+ subsamples of size n_ic (same bootstrap
count). It compares their agreement with full-90+ against the I-CONECT
agreement. p_below_ctrl = fraction of control replicates that agree with
full-90+ *less* than I-CONECT does. Small p means I-CONECT differs from 90+
more than small n explains.

Each cohort is robust-z-scored once with its own median/IQR, as in transfer()
(label-free). Loaders, availability filter and CCC selection come from cohorts.py.

Usage (from the repo root)
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/feature_importance.py
        [--thresholds all,0.50] [--n-boot 50] [--n-ctrl 10] [--n-repeats 3]

Writes results/feature_importance/{per_feature,summary,family_share,ctrl_draws}.csv.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import rz_fit, rz_transform                         # noqa: E402
from models import MODELS                                         # noqa: E402
from cohorts import (                                             # noqa: E402
    ACOUSTIC_SET, LING, N_JOBS, SEMANTIC_COLS, TEMP, Progress, _label,
    availability_filter, load_90plus, load_ccc_table, load_iconect,
    select_by_ccc, subset_name,
)
from lmc import io                                                # noqa: E402  (repo path set by cohorts)

OUT = Path(__file__).parent / "results" / "feature_importance"

TOP_K = (20, 50)
N_REPEATS_TRANSFER = 30   # permutation repeats when scoring on the other cohort


def family(f: str) -> str:
    if f in ACOUSTIC_SET:
        return "acoustic"
    if f in TEMP:
        return "temporal"
    if f in SEMANTIC_COLS:
        return "semantic"
    return "linguistic"


# ── one bootstrap: fit, OOB permutation importance, EN coefficients ──────────

def _stratified(rng: np.random.Generator, y: np.ndarray, pool: np.ndarray,
                n: int, replace: bool) -> np.ndarray:
    """Draw n indices from pool with the class ratio of y[pool]."""
    out = []
    for c in (0, 1):
        idx = pool[y[pool] == c]
        k = int(round(n * len(idx) / len(pool)))
        out.append(rng.choice(idx, k, replace=replace))
    return np.concatenate(out)


def _boot_one(X: np.ndarray, y: np.ndarray, pool: np.ndarray, model_fn,
              seed: int, n_repeats: int) -> tuple[np.ndarray, np.ndarray | None, float]:
    rng = np.random.default_rng(seed)
    boot = _stratified(rng, y, pool, len(pool), replace=True)
    oob = np.setdiff1d(pool, boot)

    clf = model_fn(seed)
    clf.fit(X[boot], y[boot])
    pi = permutation_importance(clf, X[oob], y[oob], scoring="roc_auc",
                                n_repeats=n_repeats, random_state=seed, n_jobs=1)
    coef = clf.named_steps["clf"].coef_.ravel().copy() if hasattr(clf, "named_steps") else None
    oob_auc = roc_auc_score(y[oob], clf.predict_proba(X[oob])[:, 1])
    return pi.importances_mean, coef, oob_auc


def boot_importance(X: np.ndarray, y: np.ndarray, pool: np.ndarray, model_fn,
                    seeds: list[int], n_repeats: int, prog: Progress, tag: str) -> dict:
    """Aggregate _boot_one over seeds → mean/sd importance, EN coef stats, OOB AUC."""
    res = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(_boot_one)(X, y, pool, model_fn, s, n_repeats) for s in seeds
    )
    imps, coefs, aucs = [], [], []
    for s, (imp, coef, auc) in zip(seeds, res):
        imps.append(imp)
        aucs.append(auc)
        if coef is not None:
            coefs.append(coef)
        prog.step(f"{tag} boot {s} OOB AUC={auc:.3f}")
    imps = np.vstack(imps)
    out = {"imp": imps.mean(0), "imp_sd": imps.std(0),
           "imp_pos": (imps > 0).mean(0), "oob_auc": float(np.mean(aucs))}
    if coefs:
        coefs = np.vstack(coefs)
        out.update(coef=coefs.mean(0), coef_sel=(coefs != 0).mean(0))
    return out


def transfer_importance(Xs, ys, Xt, yt, model_fn, n_repeats: int) -> np.ndarray:
    """Full-source model, permutation importance scored on the target cohort."""
    clf = model_fn(0).fit(Xs, ys)
    pi = permutation_importance(clf, Xt, yt, scoring="roc_auc",
                                n_repeats=n_repeats, random_state=0, n_jobs=N_JOBS)
    return pi.importances_mean


# ── agreement metrics ─────────────────────────────────────────────────────────

def _topk(v: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-v)[:k])


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    r = {"spearman": spearmanr(a, b).statistic}
    for k in TOP_K:
        ta, tb = _topk(a, k), _topk(b, k)
        r[f"jaccard_top{k}"] = len(ta & tb) / len(ta | tb)
    return r


def sign_agreement(ca, sa, cb, sb, min_sel: float = 0.5) -> tuple[float, int]:
    """Share of features selected in both EN models (freq ≥ min_sel) with equal sign."""
    both = (sa >= min_sel) & (sb >= min_sel)
    if not both.any():
        return np.nan, 0
    return float((np.sign(ca[both]) == np.sign(cb[both])).mean()), int(both.sum())


def family_share(v: np.ndarray, fams: np.ndarray) -> dict[str, float]:
    """Share of total positive importance per family."""
    pos = np.clip(v, 0, None)
    tot = pos.sum()
    return {f: float(pos[fams == f].sum() / tot) if tot > 0 else np.nan
            for f in ("linguistic", "temporal", "semantic", "acoustic")}


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_thresholds(s: str) -> list[float | None]:
    return [None if t.strip() == "all" else float(t) for t in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--thresholds", default="all,0.50",
                    help="comma-separated CCC thresholds; 'all' = no filter")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n-boot", type=int, default=50, help="bootstraps per cohort")
    ap.add_argument("--n-ctrl", type=int, default=10, help="90+ small-n control replicates")
    ap.add_argument("--n-repeats", type=int, default=3, help="permutation repeats on OOB")
    args = ap.parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    models = {m: MODELS[m] for m in args.models.split(",")}
    seeds = list(range(args.n_boot))

    paths = io.load_paths(args.paths)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading features & labels…")
    X_90, y_90 = load_90plus(paths)
    X_ic, y_ic = load_iconect(paths)
    avail = availability_filter(X_90, X_ic)
    print(f"Common features after availability filter: {len(avail)}")
    ccc = load_ccc_table().set_index("feature")["qq_ccc"]

    # one label-free robust z-score per cohort, as in transfer()
    Z_90 = pd.DataFrame(rz_transform(X_90[avail], *rz_fit(X_90[avail])),
                        index=X_90.index, columns=avail)
    Z_ic = pd.DataFrame(rz_transform(X_ic[avail], *rz_fit(X_ic[avail])),
                        index=X_ic.index, columns=avail)
    y90, yic = y_90.to_numpy(), y_ic.to_numpy()
    n_ic = len(yic)

    # fixed 90+ subsamples of size n_ic, shared across thresholds and models
    rng = np.random.default_rng(12345)
    ctrl_pools = [np.sort(_stratified(rng, y90, np.arange(len(y90)), n_ic, replace=False))
                  for _ in range(args.n_ctrl)]

    per_feat, summ_rows, share_rows, ctrl_rows = [], [], [], []
    prog = Progress(len(thresholds) * len(models) * (2 + args.n_ctrl) * len(seeds))

    for thresh in thresholds:
        sel = select_by_ccc(ccc.reset_index(), thresh, avail)
        A90, Aic = Z_90[sel].to_numpy(), Z_ic[sel].to_numpy()
        fams = np.array([family(f) for f in sel])
        print(f"\n{_label(thresh)} | {len(sel)} features")

        uni_90 = np.array([roc_auc_score(y90, A90[:, j]) for j in range(len(sel))])
        uni_ic = np.array([roc_auc_score(yic, Aic[:, j]) for j in range(len(sel))])

        for mname, mfn in models.items():
            tag = f"{subset_name(thresh)} {mname}"
            r90 = boot_importance(A90, y90, np.arange(len(y90)), mfn, seeds,
                                  args.n_repeats, prog, f"{tag} 90+")
            ric = boot_importance(Aic, yic, np.arange(n_ic), mfn, seeds,
                                  args.n_repeats, prog, f"{tag} IC")
            rctrl = [boot_importance(A90, y90, pool, mfn, seeds, args.n_repeats,
                                     prog, f"{tag} ctrl{i}")
                     for i, pool in enumerate(ctrl_pools)]
            tr_fwd = transfer_importance(A90, y90, Aic, yic, mfn, N_REPEATS_TRANSFER)
            tr_rev = transfer_importance(Aic, yic, A90, y90, mfn, N_REPEATS_TRANSFER)

            # ── per-feature table ─────────────────────────────────────────────
            df = pd.DataFrame({
                "ccc_threshold": thresh, "model": mname, "feature": sel,
                "family": fams, "qq_ccc": ccc.reindex(sel).to_numpy(),
                "imp_90": r90["imp"], "imp_90_sd": r90["imp_sd"], "imp_90_pos": r90["imp_pos"],
                "imp_ic": ric["imp"], "imp_ic_sd": ric["imp_sd"], "imp_ic_pos": ric["imp_pos"],
                "imp_ctrl": np.mean([c["imp"] for c in rctrl], 0),
                "imp_tr_90_to_ic": tr_fwd, "imp_tr_ic_to_90": tr_rev,
                "uni_auc_90": uni_90, "uni_auc_ic": uni_ic,
            })
            df["rank_90"] = df["imp_90"].rank(ascending=False, method="min").astype(int)
            df["rank_ic"] = df["imp_ic"].rank(ascending=False, method="min").astype(int)
            df["rank_tr_90_to_ic"] = df["imp_tr_90_to_ic"].rank(ascending=False, method="min").astype(int)
            if "coef" in r90:
                df["coef_90"], df["coef_90_sel"] = r90["coef"], r90["coef_sel"]
                df["coef_ic"], df["coef_ic_sel"] = ric["coef"], ric["coef_sel"]
            per_feat.append(df)

            # ── agreement: views to compare, each vs full-90+ ─────────────────
            views = {"perm": (r90["imp"], ric["imp"], [c["imp"] for c in rctrl])}
            if "coef" in r90:
                views["en_abs_coef"] = (np.abs(r90["coef"]), np.abs(ric["coef"]),
                                        [np.abs(c["coef"]) for c in rctrl])
            for view, (v90, vic, vctrl) in views.items():
                obs = agreement(v90, vic)
                ctrl = [agreement(v90, vc) for vc in vctrl]
                for i, c in enumerate(ctrl):
                    ctrl_rows.append({"ccc_threshold": thresh, "model": mname,
                                      "view": view, "replicate": i, **c})
                row = {"ccc_threshold": thresh, "n_features": len(sel),
                       "model": mname, "view": view,
                       "oob_auc_90": r90["oob_auc"], "oob_auc_ic": ric["oob_auc"],
                       "oob_auc_ctrl": float(np.mean([c["oob_auc"] for c in rctrl]))}
                for k, val in obs.items():
                    cv = np.array([c[k] for c in ctrl])
                    row[f"{k}_90_vs_ic"] = val
                    row[f"{k}_ctrl_mean"] = float(cv.mean()) if len(cv) else np.nan
                    row[f"{k}_ctrl_min"] = float(cv.min()) if len(cv) else np.nan
                    row[f"{k}_p_below_ctrl"] = float((cv <= val).mean()) if len(cv) else np.nan
                if view == "perm":
                    # does what each model relies on line up with what transfers?
                    row["spearman_90_vs_tr"] = spearmanr(r90["imp"], tr_fwd).statistic
                    row["spearman_ic_vs_tr"] = spearmanr(ric["imp"], tr_fwd).statistic
                    row["spearman_90_vs_ccc"] = spearmanr(
                        r90["imp"], df["qq_ccc"], nan_policy="omit").statistic
                    row["uni_auc_dir_agree"] = float(
                        (np.sign(uni_90 - 0.5) == np.sign(uni_ic - 0.5)).mean())
                else:
                    row["en_sign_agree"], row["en_n_both_sel"] = sign_agreement(
                        r90["coef"], r90["coef_sel"], ric["coef"], ric["coef_sel"])
                summ_rows.append(row)

            for view, v in (("90plus", r90["imp"]), ("iconect", ric["imp"]),
                            ("ctrl_90plus_n156", df["imp_ctrl"].to_numpy()),
                            ("transfer_90_to_ic", tr_fwd)):
                share_rows += [{"ccc_threshold": thresh, "model": mname, "view": view,
                                "family": f, "share": s}
                               for f, s in family_share(v, fams).items()]

            _report(df, summ_rows[-len(views):])

        # checkpoint after every threshold
        pd.concat(per_feat).to_csv(OUT / "per_feature.csv", index=False)
        pd.DataFrame(summ_rows).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(share_rows).to_csv(OUT / "family_share.csv", index=False)
        pd.DataFrame(ctrl_rows).to_csv(OUT / "ctrl_draws.csv", index=False)

    print(f"\nDone. Results → {OUT}/")


def _report(df: pd.DataFrame, rows: list[dict], k: int = 15) -> None:
    """Console: top-k per view side by side + agreement stats."""
    cols = {"90+ model": "imp_90", "I-CONECT model": "imp_ic",
            "90+ → IC (perm on IC)": "imp_tr_90_to_ic"}
    tops = {name: df.nlargest(k, c)[["feature", c]].to_numpy() for name, c in cols.items()}
    print("  " + " | ".join(f"{n:<42s}" for n in cols))
    for i in range(k):
        print("  " + " | ".join(f"{t[i][0][:34]:<34s} {t[i][1]:+.4f}" for t in tops.values()))
    worst = df.nsmallest(5, "imp_tr_90_to_ic")
    print("  hurts transfer: " + ", ".join(
        f"{f} ({v:+.4f})" for f, v in zip(worst["feature"], worst["imp_tr_90_to_ic"])))
    for r in rows:
        print(f"  [{r['view']}] spearman 90+ vs IC = {r['spearman_90_vs_ic']:+.3f} "
              f"(small-n ctrl mean {r['spearman_ctrl_mean']:+.3f}, "
              f"p_below={r['spearman_p_below_ctrl']:.2f}) | "
              f"top20 Jaccard {r['jaccard_top20_90_vs_ic']:.2f} "
              f"(ctrl {r['jaccard_top20_ctrl_mean']:.2f})")


if __name__ == "__main__":
    main()
