"""Build the standalone external-facing report (report/ccc-threshold-transfer-report.html).

Reads only aggregate outputs already in results/ (and the 2026-10-02 distribution
analysis), fills report_template.html, and writes one self-contained HTML file: inline
CSS, inline JS, inline SVG charts, no external requests. No subject-level data is
embedded: per-feature rows are feature names plus cohort-level statistics.

Usage (from the repo root)
-----
    python docs/analysis/cross-cohort/2026-10-10-ccc-threshold-transfer/build_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FI = RES / "feature_importance"
DIST = HERE.parent / "2026-10-02-feature-distributions" / "results"
OUT = HERE / "report" / "ccc-threshold-transfer-report.html"
TEMPLATE = HERE / "report_template.html"

DATE = "2026-10-25"
FAMILIES = ["linguistic", "temporal", "semantic", "acoustic"]
FAM_FROM_STATS = {"ling": "linguistic", "temporal": "temporal",
                  "semantic": "semantic", "acoustic": "acoustic"}
EFFECT_TIERS = [("negligible", 0.0, 0.2), ("small", 0.2, 0.5),
                ("medium", 0.5, 0.8), ("large", 0.8, np.inf)]
STRONG = (0.03, 0.05, 0.08)     # |univariate AUC - 0.5| cut-offs for direction agreement
N_TOP = 10                      # top features per model in the weight chart


def _thr(x) -> float | None:
    return None if pd.isna(x) else float(round(x, 3))


def _nan_none(v):
    return None if pd.isna(v) else float(v)


def transfer_block() -> dict:
    summ = pd.read_csv(RES / "bidirectional_summary.csv")
    tr = pd.read_csv(RES / "bidirectional_transfer_results.csv")
    out = {}
    for m in ("sgbt", "elasticnet"):
        s = summ[summ["model"] == m]
        t = tr[(tr["model"] == m) & (tr["source"] == "90plus")]
        t = t.set_index(t["ccc_threshold"].map(_thr))
        rows = []
        for _, r in s.iterrows():
            th = _thr(r["ccc_threshold"])
            c = t.loc[[th]].iloc[0] if th in t.index else t[t.index.isna()].iloc[0]
            rows.append({
                "thr": th, "n": int(r["n_features"]),
                "cv90": float(r["90plus_cv_auc"]), "cv90_sd": float(r["90plus_cv_std"]),
                "cvic": float(r["iconect_cv_auc"]), "cvic_sd": float(r["iconect_cv_std"]),
                "tr": float(c["transfer_auc"]), "lo": float(c["ci_lo"]), "hi": float(c["ci_hi"]),
                "rev": float(r["iconect_to_90plus_auc"]),
            })
        out[m] = sorted(rows, key=lambda r: -1 if r["thr"] is None else r["thr"])
    return out


def stats_block() -> dict:
    rnd = pd.read_csv(RES / "random_subset_summary.csv")
    pb = pd.read_csv(RES / "paired_bootstrap.csv")
    pm = pd.read_csv(RES / "permutation.csv")
    return {
        "random": [{"mode": r["mode"], "thr": float(r["ccc_threshold"]), "model": r["model"],
                    "n": int(r["n_features"]), "auc": float(r["ccc_auc"]),
                    "mean": float(r["rand_mean"]), "lo": float(r["rand_lo"]),
                    "hi": float(r["rand_hi"]), "p": float(r["p_one_sided"])}
                   for _, r in rnd.iterrows()],
        "paired": [{"model": r["model"], "subset": r["subset"], "diff": float(r["diff"]),
                    "lo": float(r["diff_ci_lo"]), "hi": float(r["diff_ci_hi"]),
                    "p": float(r["p_boot"])} for _, r in pb.iterrows()],
        "perm": [{"model": r["model"], "subset": r["subset"], "auc": float(r["auc"]),
                  "p": float(r["p_perm"])} for _, r in pm.iterrows()],
    }


def distribution_block() -> dict:
    allf = pd.read_csv(DIST / "all_features_stats.csv")
    tiers = []
    for name, lo, hi in EFFECT_TIERS:
        tiers.append({"tier": name, "n": int(((allf["abs_d"] >= lo) & (allf["abs_d"] < hi)).sum())})
    fam = pd.read_csv(DIST / "normalized" / "family_summary.csv")
    ccc = pd.read_csv(DIST / "normalized" / "feature_shape_stats.csv")
    ccc["fam"] = ccc["family"].map(FAM_FROM_STATS)
    bins = np.arange(0.2, 1.0001, 0.02)
    hist = {}
    for f in FAMILIES:
        v = ccc.loc[ccc["fam"] == f, "qq_ccc"].dropna().to_numpy()
        hist[f] = np.histogram(np.clip(v, bins[0], bins[-1] - 1e-9), bins=bins)[0].astype(int).tolist()
    return {
        "tiers": tiers,
        "n_features": int(len(allf)),
        "frac_large": float((allf["abs_d"] >= 0.8).mean()),
        "families": [{"family": FAM_FROM_STATS[r["family"]], "n": int(r["n_features"]),
                      "median": float(r["ccc_median"]), "same": int(r["n_same"]),
                      "moderate": int(r["n_moderate"]), "different": int(r["n_different"]),
                      "ks_raw": float(r["ks_raw_median"]), "ks_rz": float(r["ks_rz_median"])}
                     for _, r in fam.iterrows() if r["family"] != "all"],
        "hist_bins": [round(float(b), 2) for b in bins],
        "hist": hist,
        "n_ccc": int(ccc["qq_ccc"].notna().sum()),
        "frac_same": float((ccc["qq_ccc"].dropna() >= 0.95).mean()),
    }


def weights_block() -> dict:
    pf = pd.read_csv(FI / "per_feature.csv")
    pf = pf[pf["ccc_threshold"].isna()]
    en = pf[pf["model"] == "elasticnet"].copy()
    sg = pf[pf["model"] == "sgbt"].copy()

    top = set(en.reindex(en["coef_90"].abs().nlargest(N_TOP).index)["feature"]) | \
          set(en.reindex(en["coef_ic"].abs().nlargest(N_TOP).index)["feature"])
    e = en[en["feature"].isin(top)].sort_values("coef_90", key=abs, ascending=False)
    en_rows = [{"feature": r["feature"], "family": r["family"],
                "b90": float(r["coef_90"]), "s90": float(r["coef_90_sel"]),
                "bic": float(r["coef_ic"]), "sic": float(r["coef_ic_sel"]),
                "u90": float(r["uni_auc_90"]), "uic": float(r["uni_auc_ic"])}
               for _, r in e.iterrows()]

    topg = set(sg.nlargest(10, "imp_90")["feature"]) | set(sg.nlargest(10, "imp_ic")["feature"])
    g = sg[sg["feature"].isin(topg)].sort_values("imp_90", ascending=False)
    sg_rows = [{"feature": r["feature"], "family": r["family"],
                "i90": float(r["imp_90"]), "sd90": float(r["imp_90_sd"]),
                "iic": float(r["imp_ic"]), "sdic": float(r["imp_ic_sd"])}
               for _, r in g.iterrows()]

    # direction agreement vs 90+ effect size (univariate AUC). Features constant in
    # I-CONECT (AUC exactly 0.5; only ix_interruption_rate) have no direction: excluded.
    sg_v = sg[sg["uni_auc_ic"] != 0.5]
    d90, dic = sg_v["uni_auc_90"] - 0.5, sg_v["uni_auc_ic"] - 0.5
    agree = []
    for c in STRONG:
        m = d90.abs() >= c
        agree.append({"cut": c, "n": int(m.sum()),
                      "agree": float((np.sign(d90[m]) == np.sign(dic[m])).mean())})
    strong = d90.abs() >= 0.05
    agree_fam = []
    for f in FAMILIES:
        m = strong & (sg_v["family"] == f)
        if m.sum():
            agree_fam.append({"family": f, "n": int(m.sum()),
                              "agree": float((np.sign(d90[m]) == np.sign(dic[m])).mean())})

    # direction flips vs transfer importance, among the 90+ model's top-100 features
    flips = []
    for name, d in (("elasticnet", en), ("sgbt", sg)):
        d = d[d["uni_auc_ic"] != 0.5]
        d = d.assign(flip=np.sign(d["uni_auc_90"] - .5) != np.sign(d["uni_auc_ic"] - .5))
        used = d[d["rank_90"] <= 100]
        flips.append({"model": name,
                      "flip_mean": float(used.loc[used["flip"], "imp_tr_90_to_ic"].mean()),
                      "flip_n": int(used["flip"].sum()),
                      "same_mean": float(used.loc[~used["flip"], "imp_tr_90_to_ic"].mean()),
                      "same_n": int((~used["flip"]).sum())})

    summ = pd.read_csv(FI / "summary.csv")
    summ = summ[summ["ccc_threshold"].isna()]
    ctrl = pd.read_csv(FI / "ctrl_draws.csv")
    ctrl = ctrl[ctrl["ccc_threshold"].isna()]
    agr = []
    for _, r in summ.iterrows():
        c = ctrl[(ctrl["model"] == r["model"]) & (ctrl["view"] == r["view"])]["spearman"]
        agr.append({"model": r["model"], "view": r["view"], "obs": float(r["spearman_90_vs_ic"]),
                    "ctrl": [float(x) for x in c], "oob90": float(r["oob_auc_90"]),
                    "oobic": float(r["oob_auc_ic"]), "oobctrl": float(r["oob_auc_ctrl"])})

    return {"en": en_rows, "sgbt": sg_rows, "agree": agree, "agree_fam": agree_fam,
            "flips": flips, "agreement": agr,
            "n_sel90": int((en["coef_90_sel"] >= .5).sum()),
            "n_selic": int((en["coef_ic_sel"] >= .5).sum())}


def collect() -> dict:
    return {"transfer": transfer_block(), "stats": stats_block(),
            "dist": distribution_block(), "weights": weights_block(), "date": DATE}


def main() -> None:
    data = collect()
    if "--print" in sys.argv:
        print(json.dumps(data, indent=1)[:12000])
        return
    html = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", payload).replace("__DATE__", DATE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
