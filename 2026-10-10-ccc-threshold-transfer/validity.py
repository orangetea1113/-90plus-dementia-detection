"""Validity tests for the 90+ → I-CONECT transfer AUCs, over the full CCC-threshold grid.

Three tests, one data load. Each asks whether a CCC-filtered subset's transfer AUC means anything:

  random       Is a CCC-selected subset better than a random subset of the same size?
               For each threshold, draw N random subsets of the same total size from all
               features (family mix not matched) and refit the same transfer model on each;
               the CCC subset's AUC is ranked against that null. The same random subsets
               are used for SGBT and Elastic Net.
  permutation  Is the transfer AUC better than chance? (I-CONECT label permutation)
  bootstrap    Is any subset better than all features? Paired bootstrap of
               AUC(threshold) − AUC(all): the same I-CONECT subjects are resampled for
               every subset, so subject-level difficulty cancels in the difference.

permutation and bootstrap refit the 90+ → I-CONECT model once per (threshold, model)
with the same seed as bidirectional.py, so they need no prediction file from it. The fit
is checked against results/bidirectional_transfer_results.csv when that file exists, and
rows already in permutation.csv / paired_bootstrap.csv are compared with the new ones
before being overwritten.

Usage (from the repo root)
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/validity.py
        [random] [permutation] [bootstrap]          # default: all three
        [--n-draws 200] [--n-perm 5000] [--n-boot 2000]
        [--thresholds 0.5,0.98] [--out DIR]         # subset / other dir, for test runs

Writes results/{random_subset_auc,random_subset_summary,permutation,paired_bootstrap}.csv.
random_subset_auc.csv (every draw) is git-ignored.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import MODELS, rz_fit, rz_transform                  # noqa: E402
from cohorts import (                                              # noqa: E402
    N_JOBS, THRESHOLDS, Progress, _label,
    availability_filter, load_90plus, load_ccc_table, load_iconect, select_by_ccc,
    subset_name,
)
from lmc import io                                                 # noqa: E402  (repo path set by cohorts)

HERE  = Path(__file__).parent
RESULTS = HERE / "results"
TESTS = ("permutation", "bootstrap", "random")
RANDOM_STREAM = 1   # seed stream of the random draws; 1 = the "any features" stream of the earlier
                    # two-mode version (0 was family-matched, now dropped), kept so results are unchanged
TRANSFER_SEED = 0   # same model seed as bidirectional.py's transfer fit
SEED = 0
ALL = subset_name(None)   # label for the no-filter baseline


# ── data ──────────────────────────────────────────────────────────────────────

def load_data(paths: dict) -> dict:
    """Both cohorts robust-z-scored on their own median/IQR, features in the common set.

    Normalizing is column-wise, so normalizing once and slicing equals fitting per subset.
    I-CONECT subjects are sorted by id so permutation / bootstrap draws are reproducible.
    """
    X_90, y_90 = load_90plus(paths)
    X_ic, y_ic = load_iconect(paths)
    avail = availability_filter(X_90, X_ic)
    X_ic, y_ic = X_ic.sort_index(), y_ic.sort_index()

    def norm(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(rz_transform(X[avail], *rz_fit(X[avail])),
                            index=X.index, columns=avail)

    return {"avail": avail, "ccc": load_ccc_table(),
            "X90": norm(X_90), "y90": y_90.to_numpy(),
            "Xic": norm(X_ic), "yic": y_ic.to_numpy()}


def fit_proba(d: dict, cols: list[str], model_fn) -> np.ndarray:
    """Fit on 90+, return P(MCI) for every I-CONECT subject."""
    clf = model_fn(TRANSFER_SEED)
    clf.fit(d["X90"][cols].to_numpy(), d["y90"])
    return clf.predict_proba(d["Xic"][cols].to_numpy())[:, 1]


def rank_auc(y: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC from precomputed score ranks.

    y may be 1-D or (n_draws, n) so permuted labels are scored at once.
    """
    n1 = y.sum(axis=-1)
    n0 = y.shape[-1] - n1
    return ((y * ranks).sum(axis=-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _compare(old_path: Path, new: pd.DataFrame, keys: list[str], cols: list[str]) -> None:
    """Print the largest change in rows that exist in both the old file and the new run."""
    if not old_path.exists():
        return
    m = pd.read_csv(old_path).merge(new, on=keys, suffixes=("_old", ""))
    if m.empty:
        return
    worst = max(float((m[c + "_old"] - m[c]).abs().max()) for c in cols)
    print(f"  vs existing {old_path.name}: {len(m)} shared rows, max |Δ| = {worst:.2g}")


# ── permutation + paired bootstrap (need one transfer fit per threshold × model) ──

def transfer_predictions(d: dict, thresholds: list) -> dict[str, dict[str, np.ndarray]]:
    """{model: {subset: P(MCI) on I-CONECT}}, one 90+ → I-CONECT fit per cell."""
    cells = []
    for thr in thresholds:
        sel = select_by_ccc(d["ccc"], thr, d["avail"])
        if sel:
            cells += [(thr, m, sel) for m in MODELS]
    prog = Progress(len(cells))
    out: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODELS}
    res = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(fit_proba)(d, sel, MODELS[m]) for _, m, sel in cells)
    for (thr, m, sel), p in zip(cells, res):
        out[m][subset_name(thr)] = p
        prog.step(f"transfer fit {_label(thr)} {m} n={len(sel)} "
                  f"AUC={float(rank_auc(d['yic'], rankdata(p))):.3f}")
    return out


def check_against_bidirectional(y: np.ndarray, preds: dict, results: Path) -> None:
    """The refit should reproduce the AUCs bidirectional.py wrote; warn if not."""
    path = results / "bidirectional_transfer_results.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[(t["source"] == "90plus") & (t["target"] == "iconect")]
    bad = 0
    for _, r in t.iterrows():
        p = preds.get(r["model"], {}).get(subset_name(r["ccc_threshold"]))
        if p is None:
            continue
        auc = float(rank_auc(y, rankdata(p)))
        if abs(auc - r["transfer_auc"]) > 1e-6:
            bad += 1
            print(f"WARNING {r['model']} {subset_name(r['ccc_threshold'])}: refit AUC "
                  f"{auc:.4f} ≠ bidirectional {r['transfer_auc']:.4f}")
    print("  refit reproduces bidirectional_transfer_results.csv" if not bad
          else f"  {bad} refit AUCs differ from bidirectional_transfer_results.csv")


def permutation_test(y: np.ndarray, preds: dict, n_perm: int) -> pd.DataFrame:
    """One-sided p-value for AUC > 0.5 by permuting I-CONECT labels."""
    rng = np.random.default_rng(SEED)
    y_perm = np.stack([rng.permutation(y) for _ in range(n_perm)])

    rows = []
    for model, subsets in preds.items():
        for subset, p in subsets.items():
            ranks = rankdata(p)
            obs   = float(rank_auc(y, ranks))
            null  = rank_auc(y_perm, ranks)
            rows.append({
                "model":   model,
                "subset":  subset,
                "auc":     obs,
                "null_95": float(np.percentile(null, 95)),
                "p_perm":  (1 + int((null >= obs).sum())) / (1 + n_perm),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(y: np.ndarray, preds: dict, n_boot: int) -> pd.DataFrame:
    """CI and two-sided p for AUC(subset) − AUC(all), same resamples throughout."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    draws = []
    while len(draws) < n_boot:
        b = rng.integers(0, n, n)
        if 0 < y[b].sum() < n:          # need both classes
            draws.append(b)

    rows = []
    for model, subsets in preds.items():
        if ALL not in subsets:
            continue
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
            auc = float(rank_auc(y, rankdata(p)))
            rows.append({
                "model":      model,
                "subset":     subset,
                "auc":        auc,
                "auc_all":    base_obs,
                "diff":       auc - base_obs,
                "diff_ci_lo": float(lo),
                "diff_ci_hi": float(hi),
                "p_boot":     float(p_two),
            })
    return pd.DataFrame(rows)


# ── random-subset control ─────────────────────────────────────────────────────

def draw_subset(rng: np.random.Generator, sel: list[str], avail: list[str]) -> list[str]:
    """Random subset of `avail` with the same total size as `sel`."""
    return list(rng.choice(avail, len(sel), replace=False))


def fit_auc(d: dict, cols: list[str], model_fn) -> float:
    return float(roc_auc_score(d["yic"], fit_proba(d, cols, model_fn)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (thr, model), g in df.groupby(["ccc_threshold", "model"]):
        r   = g.loc[g["draw"] >= 0, "auc"].to_numpy()
        obs = float(g.loc[g["draw"] == -1, "auc"].iloc[0])
        rows.append({
            "ccc_threshold": thr, "model": model,
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


def random_subset_control(d: dict, thresholds: list[float], n_draws: int,
                          out: Path) -> pd.DataFrame:
    # draws are seeded by the threshold's position in the full grid, so a threshold
    # keeps its subsets whichever other thresholds are run
    grid = [t for t in THRESHOLDS if t is not None]
    prog = Progress(len(thresholds))
    rows = []
    for thr in thresholds:
        ti = grid.index(thr)
        sel = select_by_ccc(d["ccc"], thr, d["avail"])
        # draw −1 = the CCC subset itself
        subsets = [sel] + [
            draw_subset(np.random.default_rng([RANDOM_STREAM, ti, k]), sel, d["avail"])
            for k in range(n_draws)
        ]
        jobs = [(k - 1, m, cols) for k, cols in enumerate(subsets) for m in MODELS]
        aucs = Parallel(n_jobs=N_JOBS)(
            delayed(fit_auc)(d, cols, MODELS[m]) for _, m, cols in jobs)
        rows += [{"ccc_threshold": thr, "model": m, "draw": k,
                  "n_features": len(sel), "auc": a}
                 for (k, m, _), a in zip(jobs, aucs)]

        df = pd.DataFrame(rows)
        df.to_csv(out / "random_subset_auc.csv", index=False)   # checkpoint
        s = summarize(df[df["ccc_threshold"] == thr])
        prog.step(f"CCC≥{thr:g} n={len(sel)} | " + "  ".join(
            f"{r.model}: CCC={r.ccc_auc:.3f} rand={r.rand_mean:.3f} "
            f"p={r.p_one_sided:.3f}" for r in s.itertuples()))
    return summarize(pd.DataFrame(rows))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # no argparse choices=: with nargs="*" an empty list is rejected as an invalid choice
    ap.add_argument("tests", nargs="*", metavar="TEST",
                    help=f"tests to run, any of {', '.join(TESTS)} (default: all)")
    ap.add_argument("--paths", default="config/paths.yaml")
    ap.add_argument("--n-draws", type=int, default=200, help="random subsets per threshold")
    ap.add_argument("--n-perm", type=int, default=5000, help="label permutations")
    ap.add_argument("--n-boot", type=int, default=2000, help="paired bootstrap resamples")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated subset of the grid to run ('all' = no filter); "
                         "use with --out so partial results do not replace the full ones")
    ap.add_argument("--out", default=str(RESULTS), help="output directory (default results/)")
    args = ap.parse_args()

    bad = [t for t in args.tests if t not in TESTS]
    if bad:
        ap.error(f"unknown test {bad}; choose from {', '.join(TESTS)}")
    tests = args.tests or list(TESTS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.thresholds:
        thresholds = [None if t.strip() == "all" else float(t) for t in args.thresholds.split(",")]
        unknown = [t for t in thresholds if t not in THRESHOLDS]
        if unknown:
            raise SystemExit(f"not in the grid {THRESHOLDS}: {unknown}")
    else:
        thresholds = list(THRESHOLDS)

    d = load_data(io.load_paths(args.paths))
    y = d["yic"]
    print(f"90+: n={len(d['y90'])}  I-CONECT: n={len(y)} MCI={int(y.sum())}  "
          f"features={len(d['avail'])}  thresholds={len(thresholds)}  tests={tests}")

    if {"permutation", "bootstrap"} & set(tests):
        print("\n── transfer fits (90+ → I-CONECT) ──")
        preds = transfer_predictions(d, thresholds)
        check_against_bidirectional(y, preds, RESULTS)

        if "permutation" in tests:
            perm = permutation_test(y, preds, args.n_perm)
            print(f"\nPermutation test (AUC > 0.5, {args.n_perm} perms)")
            print(perm.to_string(index=False, float_format="%.3f"))
            _compare(out / "permutation.csv", perm, ["model", "subset"], ["auc", "p_perm"])
            perm.to_csv(out / "permutation.csv", index=False)

        if "bootstrap" in tests:
            boot = paired_bootstrap(y, preds, args.n_boot)
            print(f"\nPaired bootstrap vs all features ({args.n_boot} resamples, uncorrected)")
            print(boot.to_string(index=False, float_format="%.3f"))
            _compare(out / "paired_bootstrap.csv", boot, ["model", "subset"],
                     ["auc", "diff", "p_boot"])
            boot.to_csv(out / "paired_bootstrap.csv", index=False)

    if "random" in tests:
        print("\n── random-subset control ──")
        rthr = [t for t in thresholds if t is not None]
        summ = random_subset_control(d, rthr, args.n_draws, out)
        _compare(out / "random_subset_summary.csv", summ,
                 ["ccc_threshold", "model"], ["ccc_auc", "rand_mean", "p_one_sided"])
        summ.to_csv(out / "random_subset_summary.csv", index=False)
        print("\n" + summ.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
