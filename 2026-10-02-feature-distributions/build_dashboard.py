"""Build dashboard.html from the outputs of run.py.

Reads   results/all_features_stats.csv, results/domain_auc.csv, results/plots/**
        dashboard_style.css  (page stylesheet)
Writes  dashboard.html  (family-level images are embedded as base64; the per-feature
                         gallery lazy-loads results/plots/<cat>/<file>.png, so keep
                         dashboard.html next to results/)

Usage:  python build_dashboard.py        (run run.py first)
"""
import base64
import html
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
PLOTS = RES / "plots"
FAMILIES = ["ling", "temporal", "acoustic", "semantic"]
LONG_CONV_SEC = 900  # keep in sync with run.py
FAM_LABEL = {
    "ling": "ling · language",
    "temporal": "temporal · timing & interaction",
    "acoustic": "acoustic · eGeMAPS + MFCC",
    "semantic": "semantic",
}
TIERS = [("large", "#d63939"), ("medium", "#d97706"), ("small", "#2d7dd2"), ("negligible", "#8892a8")]
TIER_COLOR = dict(TIERS)


def slugify(name: str) -> str:  # same rule as run.py
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()[:60]


def acoustic_base(feat: str) -> str:
    return re.sub(r"_(whole|mean|std)$", "", feat)


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def auc_color(v: float) -> str:
    return "#d63939" if v >= 0.95 else "#d97706" if v >= 0.85 else "#2d7dd2" if v >= 0.75 else "#52b788"


def esc(s) -> str:
    return html.escape(str(s))


df = pd.read_csv(RES / "all_features_stats.csv")
auc = pd.read_csv(RES / "domain_auc.csv").set_index("family")
fams = [f for f in FAMILIES if f in set(df["family"])]
n_feat, n_large = len(df), int((df["abs_d"] >= 0.8).sum())
n_shape = int(((df["abs_d"] < 0.5) & (df["ks_stat"] > 0.3)).sum())

# ── transfer-risk SVGs ──────────────────────────────────────────────────────
order = sorted(fams, key=lambda f: -auc.loc[f, "auc"])
row_h, top = 34, 16
x0, w = 100, 460  # bar axis: 0..1 over w px
h1 = top + row_h * len(order) + 44
svg = [f'<svg viewBox="0 0 620 {h1}" style="width:100%;max-width:620px;height:auto;color:var(--fg)">']
for i, f in enumerate(order):
    v = float(auc.loc[f, "auc"]); y = top + i * row_h
    svg.append(f'<rect x="{x0}" y="{y}" width="{v * w:.0f}" height="24" fill="{auc_color(v)}" rx="2" opacity=".82"/>')
    svg.append(f'<text x="{x0 + v * w + 5:.0f}" y="{y + 16}" font-size="11" fill="currentColor">{v:.3f}</text>')
    svg.append(f'<text x="{x0 - 6}" y="{y + 16}" font-size="11" fill="currentColor" text-anchor="end">{f}</text>')
y_end = top + row_h * len(order)
for t in (0.25, 0.5, 0.75, 1.0):
    xx = x0 + t * w
    svg.append(f'<line x1="{xx:.0f}" y1="{top}" x2="{xx:.0f}" y2="{y_end}" stroke="var(--border)" stroke-width="1"/>')
    svg.append(f'<text x="{xx:.0f}" y="{y_end + 14}" font-size="10" fill="var(--muted)" text-anchor="middle">{t:.2f}</text>')
x5 = x0 + 0.5 * w
svg.append(f'<line x1="{x5:.0f}" y1="{top}" x2="{x5:.0f}" y2="{y_end}" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4,3" opacity=".7"/>')
svg.append(f'<text x="{x0 + w / 2:.0f}" y="{y_end + 32}" font-size="10" fill="var(--muted)" text-anchor="middle">'
           'Domain discriminability AUC (L2 logistic, 5-fold CV, complete cases)</text></svg>')
svg_auc = "".join(svg)

svg = [f'<svg viewBox="0 0 620 {top + row_h * len(order) + 34}" style="width:100%;max-width:620px;height:auto;color:var(--fg)">']
bw = 400
for i, f in enumerate(order):
    sub = df[df["family"] == f]; n = len(sub); y = top + i * row_h; x = x0
    for tier, col in TIERS:
        k = int((sub["shift_category"] == tier).sum())
        if not k:
            continue
        ww = bw * k / n
        svg.append(f'<rect x="{x:.0f}" y="{y}" width="{ww:.0f}" height="24" fill="{col}" opacity=".82" rx="2"/>')
        if ww > 22:
            svg.append(f'<text x="{x + ww / 2:.0f}" y="{y + 16}" font-size="10" fill="white" text-anchor="middle">{k}</text>')
        x += ww
    svg.append(f'<text x="{x0 - 6}" y="{y + 16}" font-size="11" fill="currentColor" text-anchor="end">{f}</text>')
    svg.append(f'<text x="{x0 + bw + 6}" y="{y + 16}" font-size="10" fill="var(--muted)">n={n}</text>')
ly = top + row_h * len(order) + 6
for j, (tier, col) in enumerate(TIERS):
    lx = x0 + j * 95
    svg.append(f'<rect x="{lx}" y="{ly}" width="12" height="10" fill="{col}" rx="2"/>')
    svg.append(f'<text x="{lx + 16}" y="{ly + 9}" font-size="10" fill="var(--muted)">{tier}</text>')
svg.append("</svg>")
svg_shift = "".join(svg)

# ── family browser ──────────────────────────────────────────────────────────
# ── per-figure explanations shown under each family summary ─────────────────
# Wording refers to figures verified against results/ when written; if the data or the
# long/short split change, re-check the numbers quoted here.
FIG_NOTES = {
    "all_features": [
        ("A · PCA", "All 547 features are used, so only the 86 I-CONECT subjects with acoustic data "
         "(70 dropped) and 1,110 90+ subjects with no missing value are shown. 90+ forms one cloud around "
         "the origin. I-CONECT splits in two: 69 long-conversation subjects far to the right, fully "
         "separated from 90+, and 17 short-conversation subjects at the edge of the 90+ cloud. The "
         "separation between the cohorts is mostly the long subgroup versus 90+."),
        ("B · Cohen's d histogram", "358 features have d > 0 (I-CONECT higher) and 189 have d < 0. "
         "The spike at d ≈ 1.8–2.4 holds 91 features, 63 of them language counts that grow with "
         "conversation length. The 12 features with d ≤ −2 are mostly acoustic (10), plus "
         "interruption rate and turns per minute."),
        ("C · Support overlap", "U-shaped. 264 features (48%) keep at least 80% of I-CONECT inside the "
         "90+ range and are safe to transfer; 167 (31%) fall below 50% and force a 90+-trained model to "
         "extrapolate. 46 of those are below 10% (40 acoustic, 4 temporal, 2 language)."),
        ("D · |d| vs KS", "The two statistics agree on the trunk (rank correlation 0.81). Three "
         "regions matter: the grey box on the left (66 features, 62 acoustic) where the mean hardly moves "
         "but the distribution does; the horizontal band at |d| ≈ 2, KS ≈ 0.45 (language counts "
         "driven by the long subgroup); and the top right, where both statistics are large."),
    ],
    "ling": [
        ("A · PCA", "The first component alone explains 70% of the variance and is simply how much "
         "was said (word count, function words, clauses). 90+ and the short I-CONECT subgroup sit on top of "
         "each other on the left; the long subgroup stretches far to the right. The language family "
         "differs mainly in volume, not in style."),
        ("B · Per-feature shift", "The 20 features with the largest KS are all I-CONECT-higher counts "
         "(assent, numbers, negations, nonfluencies, adverbs, function words, clauses). For 11 of them "
         "overlap is 45–49%, i.e. about half of I-CONECT (the long subgroup) is out of range. The most "
         "extreme are Nonfluencies (9%) and Assent (10%): even short conversations are out of range, "
         "which suggests different transcription conventions for fillers. Sentence-structure counts "
         "(T-units, clauses per sentence) are also lower, at 20–43%."),
        ("C · |d| vs KS", "Most points lie on a horizontal band at |d| ≈ 1.5–2.5 with KS ≈ 0.45: "
         "d is large but only part of I-CONECT is shifted. The green cluster at the bottom left (15 features "
         "with |d| < 0.5) consists entirely of ratios and length-corrected measures (per-clause or "
         "per-T-unit rates, MTLD). Only 2 features fall in the shape-change box."),
    ],
    "temporal": [
        ("A · PCA", "PC1 (33%) is conversation scale: speech time, duration, word count and turns. "
         "90+ clusters on the left. The long I-CONECT subgroup is far to the right. The short subgroup has "
         "a similar scale to 90+ but lies below it on PC2: participants speak only about 13% of the time "
         "(90+: 52%), so even short I-CONECT conversations are interviewer-led."),
        ("B · Per-feature shift", "Two different kinds of shift. Positive (scale) features such as "
         "duration, interviewer turns (overlap 8%) and total speech or words (37–47%) are driven by the "
         "long subgroup. Negative (pacing) features such as interruption rate (0%), turns per minute (6%), "
         "pause ratio (13%) and mean pause (16%) are out of range in both subgroups, so they are not a "
         "length effect."),
        ("C · |d| vs KS", "The red points at the top right are the out-of-range features described above. "
         "The row at KS ≈ 0.45 holds the count features that only the long subgroup shifts. The two "
         "features in the grey box are mean and median response latency: the range overlaps (98–99%), "
         "but the distribution shape does not."),
    ],
    "acoustic": [
        ("A · PCA", "Only 86 of 156 I-CONECT subjects have acoustic data (70 dropped), 69 of them from the "
         "long subgroup. PC1 (19%) is dominated by the standard-deviation aggregates of MFCCs and their "
         "deltas; PC2 (11%) by loudness level. The long subgroup lies completely outside the 90+ 95% range; "
         "the 17 short-conversation subjects sit at the edge of the 90+ cloud."),
        ("B · Per-feature shift", "Shows the top 20 of 420 features by KS: all have KS above 0.93 and "
         "overlap of 0–2%, i.e. the cohorts do not overlap at all. Positive shifts (d = +3.5 to +7.2) are "
         "the variability of loudness, spectral flux and MFCC 1–8; negative shifts (d ≈ −2) are the "
         "high-order MFCC delta statistics and the 20th loudness percentile."),
        ("C · |d| vs KS", "Polarised: a large red group at the top right (|d| > 2, KS > 0.8, overlap near "
         "0) and a large green group at the bottom left that can be used as is. The grey box holds 62 shape-change "
         "features, 94% of all such features in the dataset; d cannot see them, KS can."),
    ],
    "semantic": [
        ("A · PCA", "PC1 (42%) contrasts semantic spread with content-word frequency; PC2 (30%) is content "
         "diversity. The short I-CONECT subgroup lies inside the 90+ cloud; the long subgroup sits below it "
         "because longer texts have lower content diversity."),
        ("B · Per-feature shift", "The only family in which no feature is below 50% overlap. Content density "
         "(d = −1.13) has 80% overlap but a high KS (0.72): I-CONECT is more concentrated, not out of range. "
         "Content diversity (d = −1.05) is a length artefact (long subgroup 0.50, short 0.86, 90+ 0.81). "
         "Content-word frequency (d = −0.06) is essentially identical in the two cohorts."),
        ("C · |d| vs KS", "No point lies in the shape-change box. Three of four features have |d| > 0.8, but "
         "all keep most of I-CONECT inside the 90+ range, so the family is the least risky to transfer once "
         "length-sensitive measures are normalised."),
    ],
}


def notes_html(key: str) -> str:
    items = "".join(f"<li><b>{esc(h)}</b>{esc(t)}</li>" for h, t in FIG_NOTES[key])
    return f'<ul class="fp-notes">{items}</ul>'


tabs = ['<button class="ftab active" data-fam="all_features">all features</button>']
panels = []
all_img = b64(PLOTS / "family" / "summary_all_features.png")
n_aucfams = int((auc["auc"] >= 0.99).sum())
panels.append(f'''<div class="fpanel" id="fp-all_features">
  <div class="fp-hdr"><h3>all features (combined)</h3>
    <div class="cats-row">
      <span class="chip2" style="border-color:#d63939;color:#d63939">{n_feat} features · {len(fams)} families</span>
      <span class="chip2" style="border-color:#4477AA;color:#4477AA">IC n≈156 · 90+ n≈1120</span>
      <span class="chip2" style="border-color:#8892a8;color:#8892a8">PCA on complete cases (86 IC · 1110 90+)</span>
    </div>
  </div>
  <div class="metric-row">
    <div class="metric"><div class="mval" style="color:#d63939">{n_large}</div><div class="mlbl">Large-shift (|d|≥0.8)</div></div>
    <div class="metric"><div class="mval" style="color:#d97706">{n_shape}</div><div class="mlbl">Shape change (|d|&lt;0.5, KS&gt;0.3)</div></div>
    <div class="metric"><div class="mval">{n_feat}</div><div class="mlbl">Total features</div></div>
    <div class="metric"><div class="mval" style="color:#52b788">{len(fams)}</div><div class="mlbl">Families</div></div>
  </div>
  <div class="img-wrap"><img src="{all_img}" alt="all features summary" class="fam-img"></div>
  {notes_html("all_features")}
</div>''')
for f in fams:
    sub = df[df["family"] == f]; n = len(sub)
    chips = "".join(
        f'<span class="chip2" style="border-color:{c};color:{c}">{t}: {int((sub["shift_category"] == t).sum())} '
        f'({(sub["shift_category"] == t).mean():.0%})</span>'
        for t, c in TIERS if (sub["shift_category"] == t).any())
    shape = int(((sub["abs_d"] < 0.5) & (sub["ks_stat"] > 0.3)).sum())
    chips += f'<span class="chip2" style="border-color:#8892a8;color:#8892a8">shape change: {shape}</span>'
    a = auc.loc[f]
    note = f' (IC n={int(a["n_ic_auc"])})' if int(a["n_ic_auc"]) < 150 else ""
    tabs.append(f'<button class="ftab" data-fam="{f}">{f}</button>')
    panels.append(f'''<div class="fpanel" id="fp-{f}" style="display:none">
      <div class="fp-hdr"><h3>{esc(FAM_LABEL[f])}</h3><div class="cats-row">{chips}</div></div>
      <div class="metric-row">
      <div class="metric"><div class="mval">{a["auc"]:.3f}</div><div class="mlbl">Domain AUC{note}</div></div>
      <div class="metric"><div class="mval">{a["mean_abs_d"]:.3f}</div><div class="mlbl">Mean |d|</div></div>
      <div class="metric"><div class="mval">{a["mean_ks"]:.3f}</div><div class="mlbl">Mean KS</div></div>
      <div class="metric"><div class="mval">{a["mean_overlap"]:.3f}</div><div class="mlbl">Mean overlap</div></div>
      <div class="metric"><div class="mval">{a["frac_overlap_lt50"]:.0%}</div><div class="mlbl">Features with overlap &lt; 50%</div></div>
      <div class="metric"><div class="mval">{n}</div><div class="mlbl">Features</div></div>
    </div>
      <div class="img-wrap"><img src="{b64(PLOTS / "family" / f"summary_{f}.png")}" alt="{f}" class="fam-img"></div>
      {notes_html(f)}
    </div>''')

# ── top-20 table ────────────────────────────────────────────────────────────
rows = []
for r in df.nlargest(20, "abs_d").itertuples():
    col = TIER_COLOR[r.shift_category]
    rows.append(
        f'<tr><td class="mono" title="{esc(r.feature)}">{esc(r.feature)}</td>'
        f'<td><span class="chip" style="background:{col}22;color:{col}">{r.shift_category}</span></td>'
        f'<td class="muted-cell">{r.family}</td><td class="num">{r.abs_d:.2f}</td>'
        f'<td class="num">{r.ks_stat:.3f}</td><td class="num">{r.ks_pval:.2e}</td></tr>')
top_rows = "".join(rows)

# ── gallery catalog: one entry per plot file ────────────────────────────────
catalog = {}
for cat in FAMILIES:
    sub = df[df["family"] == cat]
    if cat == "acoustic":  # one trio plot per base feature; report the strongest variant
        sub = sub.assign(key=[slugify(acoustic_base(x)) for x in sub["feature"]])
        sub = sub.loc[sub.groupby("key")["abs_d"].idxmax()]
        meta = {r.key: r for r in sub.itertuples()}
    else:
        meta = {slugify(r.feature): r for r in sub.itertuples()}
    entries = []
    for p in sorted((PLOTS / cat).glob("*.png")):
        m = meta.get(p.stem)
        entries.append({"f": p.name,
                        "d": round(float(m.abs_d), 3) if m else None,
                        "s": m.shift_category if m else "",
                        "k": round(float(m.ks_stat), 3) if m else None})
    catalog[cat] = entries
n_plots = sum(len(v) for v in catalog.values())
cat_tabs = "".join(
    f'<button class="tab{" active" if c == "ling" else ""}" data-cat="{c}">{c} ({len(catalog[c])})</button>'
    for c in fams)

# ── I-CONECT subgroups (written by run.py: plot_ic_subgroups) ───────────────
sg_feat = pd.read_csv(RES / "ic_subgroup_summary.csv")
sg_fam = pd.read_csv(RES / "ic_subgroup_family.csv").set_index("family")
n_long = int(sg_feat["n_ic_long"].max()); n_short = int(sg_feat["n_ic_short"].max())
dur = sg_feat.set_index("feature").loc["ix_conversation_dur_sec"]


def fmt_med(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def ov_cell(v: float) -> str:
    col = "#d63939" if v < 0.5 else "var(--fg)"
    w = "600" if v < 0.5 else "400"
    return f'<td class="num" style="color:{col};font-weight:{w}">{v:.0%}</td>'


sg_rows = "".join(
    f'<tr><td>{esc(r.label)}</td><td class="muted-cell">{r.family}</td>'
    f'<td class="num">{fmt_med(r.median_90plus)}</td><td class="num">{fmt_med(r.median_ic_short)}</td>'
    f'<td class="num">{fmt_med(r.median_ic_long)}</td>{ov_cell(r.overlap_ic_short)}{ov_cell(r.overlap_ic_long)}</tr>'
    for r in sg_feat.itertuples())

# ── key findings: numbers pulled from the results so they stay in sync ──────
ling_small = int(((df["family"] == "ling") & (df["abs_d"] < 0.5)).sum())
shape_ac = int(((df["family"] == "acoustic") & (df["abs_d"] < 0.5) & (df["ks_stat"] > 0.3)).sum())
ac_ic = int(auc.loc["acoustic", "n_ic_auc"]) if "acoustic" in auc.index else 0
sgf = lambda fam, g: sg_fam.loc[fam, f"frac_overlap_lt50_ic_{g}"]
findings = [
    ("The two cohorts are almost perfectly separable.",
     f"A classifier trained on any single family tells I-CONECT from 90+ subjects with AUC "
     f"{auc['auc'].min():.2f}–{auc['auc'].max():.2f}; {n_large} of {n_feat} features "
     f"({n_large / n_feat:.0%}) have a large shift (|d| ≥ 0.8). Models trained on 90+ should "
     f"not be applied to I-CONECT without adaptation.", "#s-risk"),
    ("Most of the shift comes from one I-CONECT subgroup.",
     f"I-CONECT conversations are either short (n={n_short}, median {dur.median_ic_short:,.0f} s) "
     f"or long (n={n_long}, median {dur.median_ic_long:,.0f} s; 90+ median {dur.median_90plus:,.0f} s). "
     f"In the language family, {sgf('ling', 'long'):.0%} of features fall outside the 90+ range for "
     f"the long subgroup but only {sgf('ling', 'short'):.0%} for the short one: raw counts "
     f"(LIWC categories, clauses, words) simply grow with conversation length.", "#s-subgroups"),
    ("Timing and turn-taking differ in both subgroups.",
     f"Pause ratio, long pauses, turns per minute and interruption rate are far from 90+ even in "
     f"short conversations ({sgf('temporal', 'short'):.0%} of temporal features out of range for the "
     f"short subgroup). This is not a length effect; it points to differences in transcription, "
     f"segmentation or diarization that need checking.", "#s-subgroups"),
    ("Cohen's d alone misses part of the shift.",
     f"{n_shape} features have |d| < 0.5 but KS > 0.3: the mean barely moves while the distribution "
     f"shape or spread changes. {shape_ac} of them are acoustic. Conversely, d overstates the shift "
     f"of count features driven by the long subgroup.", "#s-howto"),
    ("Ratio-based and semantic features transfer best.",
     f"All {ling_small} language features with |d| < 0.5 are ratios or length-corrected measures "
     f"(per-clause / per-T-unit rates, MTLD). The semantic family has no feature outside the 90+ "
     f"range (AUC {auc.loc['semantic', 'auc']:.2f}). Length-sensitive measures such as Simple TTR "
     f"and content diversity should be avoided or normalised.", "#s-family"),
]
findings_html = "".join(
    f'<li><div class="kf-head">{esc(h)}</div><div class="kf-body">{esc(b)} '
    f'<a href="{link}">details →</a></div></li>' for h, b, link in findings)

# ── representative examples (report section 2.3.2) ──────────────────────────
adv = sg_feat.set_index("feature").loc["Adverbs"]
EXAMPLES = [  # (feature, plot subdir, label, pattern tag, tag colour, takeaway)
    ("sm_content_freq", "semantic", "Mean content-word frequency",
     "Baseline: no shift", "#52b788",
     "Medians almost coincide (5.38 vs 5.39) and d is negligible. I-CONECT is only a little more "
     "concentrated, which the small but non-zero KS picks up. This is what a transferable feature "
     "looks like."),
    ("ix_n_interviewer_turns", "temporal", "Interviewer turns",
     "Pure location shift", "#d63939",
     "90+ conversations have about 25 interviewer turns, I-CONECT about 90; the two curves barely "
     "overlap and cross only once. d and KS agree, so d summarises this shift well."),
    ("ix_mean_response_latency_sec", "temporal", "Mean participant response latency",
     "Shape change that d misses", "#d97706",
     "By d this is negligible, yet the curves clearly differ: I-CONECT is a narrow peak around 0.7 s, "
     "while 90+ is wider and spreads into negative (overlapping) and long latencies. 90+'s tails "
     "inflate the pooled SD and deflate d; KS shows the moderate difference."),
    ("Nonfluencies", "ling", "Nonfluencies (um, uh)",
     "Heavy tail deflates d", "#d97706",
     "Almost every 90+ subject sits near 0, while I-CONECT spreads to 150+. I-CONECT's huge variance "
     "dominates the pooled SD, so d (already large) still understates the separation; KS shows a "
     "single low threshold separates most subjects."),
    ("pt_long_pause_per_min", "temporal", "Long pauses (> 2 s) per minute",
     "Variance compression", "#d97706",
     "I-CONECT subjects all pause rarely (0–0.5 per minute) and look alike, while 90+ peaks near "
     "1 per minute with a tail past 10. d only just passes the large threshold, but KS is high: a "
     "90+-trained model would only ever see I-CONECT at the far-left edge of its range."),
    ("Adverbs", "ling", "Adverbs (LIWC count)",
     "d inflated by a subgroup", "#7a5af5",
     f"One of the largest language shifts by d, yet KS is only moderate. I-CONECT is bimodal: the "
     f"short-conversation subgroup matches 90+ (median {adv.median_ic_short:.1f} vs "
     f"{adv.median_90plus:.1f}), while the long subgroup sits at {adv.median_ic_long:.0f}. d averages "
     f"the two into one big shift; raw counts simply grow with conversation length."),
]
ex_cards = []
for k, (feat, sub_dir, label, tag, tag_col, text) in enumerate(EXAMPLES):
    r = df.set_index("feature").loc[feat]
    std_ratio = r["ic_std"] / r["np_std"] if r["np_std"] > 0 else float("nan")
    ex_cards.append(f'''<div class="ex-card">
      <div class="ex-hdr"><span class="ex-letter">{"ABCDEF"[k]}</span>
        <span class="chip" style="background:{tag_col}22;color:{tag_col}">{esc(tag)}</span></div>
      <div class="ex-title">{esc(label)} <span class="ex-feat">{esc(feat)} · {r["family"]}</span></div>
      <img src="{b64(PLOTS / sub_dir / f"{slugify(feat)}.png")}" alt="{esc(feat)}">
      <div class="ex-stats">
        <div><span>{r["cohens_d"]:+.2f}</span>Cohen's d</div>
        <div><span>{r["ks_stat"]:.2f}</span>KS</div>
        <div><span>{std_ratio:.2f}</span>SD ratio (IC / 90+)</div>
      </div>
      <p class="ex-text">{esc(text)}</p>
    </div>''')
examples_html = "".join(ex_cards)

# ── normalised view (written by run.py: run_normalized_analysis) ────────────
norm_csv = RES / "normalized" / "feature_shape_stats.csv"
norm = pd.read_csv(norm_csv) if norm_csv.exists() else None
if norm is not None:
    norm["note"] = norm["note"].fillna("")
    nok = norm[norm["qq_ccc"].notna()]
    skipped = norm[norm["qq_ccc"].isna()]
    fallback = nok[nok["note"] != ""]
    CCC_OK, CCC_WARN = 0.95, 0.90  # keep in sync with run.py
    norm_catalog = {
        fam: [{"f": r.plot, "n": r.feature, "c": round(r.qq_ccc, 3), "k": round(r.ks_rz, 3),
               "sr": round(r.sd_ratio, 2), "d": round(r.cohens_d_raw, 2), "kr": round(r.ks_raw, 3)}
              for r in nok[nok["family"] == fam].itertuples()]
        for fam in fams}
    norm_tabs = "".join(
        f'<button class="tab{" active" if c == "ling" else ""}" data-cat="{c}">{c} ({len(norm_catalog[c])})</button>'
        for c in fams)

    summ = pd.read_csv(RES / "normalized" / "family_summary.csv").set_index("family")
    stats_rows = ""
    for fam in [*fams, "all"]:
        if fam not in summ.index:
            continue
        q = summ.loc[fam]
        bold = ' style="font-weight:600"' if fam == "all" else ""
        stats_rows += (
            f'<tr{bold}><td>{fam}</td><td class="num">{int(q.n_features)}</td>'
            f'<td class="num">{q.ccc_median:.3f}</td>'
            f'<td class="num muted-cell">{q.ccc_q25:.3f} \u2013 {q.ccc_q75:.3f}</td>'
            f'<td class="num">{q.ccc_min:.3f}</td>'
            f'<td class="num" style="color:#2a9d60">{int(q.n_same)} ({q.frac_same:.0%})</td>'
            f'<td class="num" style="color:#d97706">{int(q.n_moderate)} ({q.frac_moderate:.0%})</td>'
            f'<td class="num" style="color:#d63939">{int(q.n_different)} ({q.frac_different:.0%})</td>'
            f'<td class="num">{q.ks_raw_median:.3f}</td><td class="num">{q.ks_rz_median:.3f}</td></tr>')
    ccc_img = b64(RES / "normalized" / "ccc_summary.png")
    skipped_txt = "; ".join(f"{esc(r.feature)} ({esc(r.note)})" for r in skipped.itertuples()) or "none"
    fallback_txt = "; ".join(f"{esc(r.feature)} ({esc(r.note)})" for r in fallback.itertuples())
    fallback_html = f" IQR fallback: {fallback_txt}." if fallback_txt else ""
    n_ok = int((nok["qq_ccc"] >= CCC_OK).sum())
    n_mid = int(((nok["qq_ccc"] >= CCC_WARN) & (nok["qq_ccc"] < CCC_OK)).sum())
    n_bad = int((nok["qq_ccc"] < CCC_WARN).sum())
    # ── numbers for the summary block ──────────────────────────────────────
    nok = nok.assign(abs_d=nok["cohens_d_raw"].abs())
    big = nok[nok["abs_d"] >= 0.8]
    n_big, n_big_same = len(big), int((big["qq_ccc"] >= CCC_OK).sum())
    n_ks_raw, n_ks_rz = int((nok["ks_raw"] > 0.3).sum()), int((nok["ks_rz"] > 0.3).sum())
    ks_raw_med, ks_rz_med = nok["ks_raw"].median(), nok["ks_rz"].median()
    tier = lambda f, lo, hi: int(((nok["family"] == f) & (nok["qq_ccc"] >= lo) & (nok["qq_ccc"] < hi)).sum())
    n_f = lambda f: int((nok["family"] == f).sum())
    diff = {f: tier(f, -9, CCC_WARN) for f in fams}
    same = {f: tier(f, CCC_OK, 9) for f in fams}
    pct = lambda a, b: f"{a / b:.0%}"
    ling_ks = nok.loc[nok["family"] == "ling", "ks_rz"].median()
    ling_ks_raw = nok.loc[nok["family"] == "ling", "ks_raw"].median()
    tdiff = nok[(nok["family"] == "temporal") & (nok["qq_ccc"] < CCC_WARN)]["feature"].tolist()
    mfd = nok[nok["feature"].str.match(r"mfcc_d\d+_mean_whole$")]
    n_mfd, n_mfd_bad = len(mfd), int((mfd["qq_ccc"] < CCC_WARN).sum())
    n_ac_mfd_bad = int(((nok["family"] == "acoustic") & nok["feature"].str.startswith("mfcc_d")
                        & (nok["qq_ccc"] < CCC_WARN)).sum())
    n_ccc_only = int(((nok["qq_ccc"] < CCC_WARN) & (nok["ks_rz"] < 0.15)).sum())
    n_same = int((nok["qq_ccc"] >= CCC_OK).sum())
    n_same_sd = int(((nok["qq_ccc"] >= CCC_OK) & ((nok["sd_ratio"] < 0.5) | (nok["sd_ratio"] > 2))).sum())
    rho = nok[["qq_ccc", "ks_rz"]].corr("spearman").iloc[0, 1]
    summary_html = f'''<div class="sec" id="n-summary" style="margin-top:8px">
  <div class="sec-title">Summary · What Is Left After Normalisation</div>
  <div class="stat-row">
    <div class="sc"><div class="sv">{ks_raw_med:.2f} → {ks_rz_med:.2f}</div><div class="sl">Median KS<br><span style="font-size:11px;color:var(--muted)">raw → robust z-score</span></div></div>
    <div class="sc"><div class="sv" style="color:#2a9d60">{pct(n_same, len(nok))}</div><div class="sl">Same shape<br><span style="font-size:11px;color:var(--muted)">CCC ≥ {CCC_OK} · {n_same} of {len(nok)} features</span></div></div>
    <div class="sc"><div class="sv" style="color:#d63939">{pct(n_bad, len(nok))}</div><div class="sl">Different shape<br><span style="font-size:11px;color:var(--muted)">CCC &lt; {CCC_WARN} · {n_bad} features</span></div></div>
    <div class="sc"><div class="sv" style="color:#d63939">{pct(diff["ling"], n_f("ling"))}</div><div class="sl">Language features still different<br><span style="font-size:11px;color:var(--muted)">temporal {pct(diff["temporal"], n_f("temporal"))} · acoustic {pct(diff["acoustic"], n_f("acoustic"))} · semantic {pct(diff["semantic"], n_f("semantic"))}</span></div></div>
  </div>
  <div class="card">
    <ol class="kf">
      <li><div class="kf-head">Per-cohort normalisation resolves most of the shift.</div><div class="kf-body">{n_ks_raw} features had a raw KS above 0.3; after the robust z-score only {n_ks_rz} do. Of the {n_big} features with a large raw shift (|d| ≥ 0.8), {n_big_same} ({pct(n_big_same, n_big)}) have the same shape in the two cohorts (CCC ≥ {CCC_OK}): for them the difference was only location and scale, which each cohort's own median and IQR remove.</div></li>
      <li><div class="kf-head">The language family is the exception: its shape differs.</div><div class="kf-body">{diff["ling"]} of {n_f("ling")} language features ({pct(diff["ling"], n_f("ling"))}) have CCC &lt; {CCC_WARN}, and their median KS barely moves ({ling_ks_raw:.2f} → {ling_ks:.2f}). These are count features that are bimodal in I-CONECT (a short-conversation and a long-conversation subgroup); a z-score shifts and rescales both modes but cannot merge them. Only {same["ling"]} language features keep the same shape, and 19 of those are ratios or length-corrected measures (per-clause and per-T-unit rates, mean lengths, MTLD and TTR variants).</div></li>
      <li><div class="kf-head">Temporal: scale features differ, pacing features line up.</div><div class="kf-body">{same["temporal"]} of {n_f("temporal")} temporal features have the same shape. The {diff["temporal"]} that do not are all size or share features ({", ".join(esc(x) for x in tdiff)}), driven by the long-conversation subgroup. Pacing features that were far from 90+ in raw terms (pause ratio, long pauses per minute, turns per minute, response latency) have CCC ≥ 0.97, so their gap is a location shift: normalisation removes it, and any real difference in pacing between the cohorts with it.</div></li>
      <li><div class="kf-head">Acoustic: mostly the same shape, with one weak spot.</div><div class="kf-body">{same["acoustic"]} of {n_f("acoustic")} acoustic features ({pct(same["acoustic"], n_f("acoustic"))}) match. {diff["acoustic"]} do not, and {n_ac_mfd_bad} of those are MFCC delta statistics; the worst are the whole-recording means of the deltas ({n_mfd_bad} of {n_mfd} have CCC &lt; {CCC_WARN}), which have a long one-sided tail in I-CONECT. Acoustic scores rest on only 86 I-CONECT subjects, 69 from the long subgroup.</div></li>
      <li><div class="kf-head">Semantic: no feature differs.</div><div class="kf-body">{same["semantic"]} of {n_f("semantic")} semantic features have the same shape and the other is moderate. Content diversity ({nok.set_index("feature").loc["sm_content_diversity", "qq_ccc"]:.2f}) is the weakest, consistent with the length effect seen in the raw view.</div></li>
      <li><div class="kf-head">CCC adds information that KS misses.</div><div class="kf-body">The two scores mostly agree (rank correlation {rho:.2f}), but {n_ccc_only} features have CCC &lt; {CCC_WARN} while KS stays below 0.15: a difference in the tails that the largest-gap KS does not see.</div></li>
      <li><div class="kf-head">Caveat: same shape is necessary for transfer, not sufficient.</div><div class="kf-body">The scores assume each cohort is normalised with its own statistics, and say nothing about whether a feature relates to cognition in the same way. Normalising also hides scale: {n_same_sd} of the {n_same} same-shape features still have a raw SD ratio outside 0.5–2, so a model that sees raw values would still be shifted on them.</div></li>
    </ol>
  </div>
</div>

'''
    norm_view = summary_html + f'''<div class="sec" id="n-intro" style="margin-top:8px">
  <div class="sec-title">What This View Shows</div>
  <div class="two-col">
    <div class="card">
      <p class="note" style="font-weight:600;color:var(--fg)">Method</p>
      <ul class="howto">
        <li>Every feature is <b>robust-z-scored separately within each cohort</b>: z = (x − median) / IQR, using that cohort's own median and interquartile range. This removes the location (mean) shift and the scale difference, so what is left is the difference in distribution <b>shape</b> (skew, tails, bimodality).</li>
        <li>Median/IQR is used instead of mean/SD because a few extreme values inflate the SD and squash the rest of the distribution, which creates spurious shape differences.</li>
        <li>Shape similarity is judged mainly by the <b>Q–Q concordance (CCC)</b>: pair the two cohorts' 5th–95th percentiles and measure how closely they follow the identity line.</li>
        <li>Each plot shows the two robust-z densities (left) and the Q–Q plot (right; coloured points enter the CCC, grey points 1–4 and 96–99 do not). Points on the dashed diagonal mean identical shape.</li>
        <li>Single-feature analysis only; the raw-data figures in the other tab are unchanged.</li>
      </ul>
    </div>
    <div class="card">
      <p class="note" style="font-weight:600;color:var(--fg)">Metrics</p>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Metric</th><th>Meaning</th><th>Rule of thumb</th></tr></thead>
        <tbody>
          <tr><td><b>Q–Q CCC</b> (primary)</td><td>Lin's concordance correlation of the two cohorts' 5th–95th percentiles. Unlike Pearson r it also penalises a slope or offset away from the diagonal</td><td class="muted-cell"><span style="color:#2a9d60">≥ {CCC_OK}</span> same shape · <span style="color:#d97706">{CCC_WARN}–{CCC_OK}</span> moderate · <span style="color:#d63939">&lt; {CCC_WARN}</span> different</td></tr>
          <tr><td>KS (robust z)</td><td>KS statistic between the two robust-z samples; most sensitive to the bulk of the distribution</td><td class="muted-cell">&lt; 0.1 similar; &gt; 0.3 clearly different</td></tr>
          <tr><td>SD ratio (raw)</td><td>SD(I-CONECT) / SD(90+) before normalisation, kept because normalising hides scale differences</td><td class="muted-cell">far from 1 = scale difference</td></tr>
          <tr><td>raw d, raw KS</td><td>The statistics before normalisation, for comparison</td><td class="muted-cell">—</td></tr>
        </tbody>
      </table></div>
      <p class="note" style="margin-top:10px">Result: {n_ok} features with CCC ≥ {CCC_OK}, {n_mid} between {CCC_WARN} and {CCC_OK}, {n_bad} below {CCC_WARN}. CCC and KS mostly agree; when they do not, CCC usually flags a tail difference that KS (which looks at the single largest gap) misses. A similar shape after normalisation is necessary for transfer but not sufficient: it assumes each cohort is normalised with its own statistics and says nothing about whether the feature relates to cognition the same way. Skipped: {skipped_txt}.{fallback_html}</p>
    </div>
  </div>
</div>

<div class="sec" id="n-stats">
  <div class="sec-title">Score Statistics · Q–Q CCC across {len(nok)} features</div>
  <div class="card">
    <div class="img-wrap" style="max-width:1100px;margin-inline:auto"><img src="{ccc_img}" alt="Q–Q CCC statistics" class="fam-img"></div>
    <ul class="fp-notes" style="max-width:1100px;margin-inline:auto">
      <li><b>A · Distribution</b>Most features pile up near 1 (median {nok["qq_ccc"].median():.3f}), with a long tail of low values. The tail is mostly acoustic (MFCC delta statistics); the cluster around 0.85 is language.</li>
      <li><b>B · Tiers</b>Same shape (CCC ≥ {CCC_OK}) for {n_ok} features ({n_ok / len(nok):.0%}), moderate for {n_mid}, different (&lt; {CCC_WARN}) for {n_bad} ({n_bad / len(nok):.0%}). The language family is the least similar: its count features are bimodal in I-CONECT (long and short conversations), and a robust z-score cannot remove that.</li>
      <li><b>C · CCC vs KS</b>The two scores mostly agree (rank correlation {nok[["qq_ccc", "ks_rz"]].corr("spearman").iloc[0, 1]:.2f}). The language features form a separate group with KS 0.3–0.5 and CCC around 0.85. At the bottom left (KS low, CCC low) are mainly acoustic features with a long one-sided tail in I-CONECT, which CCC picks up and KS does not.</li>
    </ul>
    <p class="note" style="margin-top:14px">Per family. CCC columns: median, interquartile range, minimum. Tier columns: number of features (share) with CCC ≥ {CCC_OK} (same), {CCC_WARN}–{CCC_OK} (moderate) and &lt; {CCC_WARN} (different). KS columns: median before normalisation and after robust z-scoring (the normalisation removes most of the KS in every family except language).</p>
    <div class="tbl-wrap"><table>
      <thead><tr><th>Family</th><th style="text-align:right">Features</th><th style="text-align:right">CCC median</th><th style="text-align:right">CCC IQR</th><th style="text-align:right">CCC min</th><th style="text-align:right">Same</th><th style="text-align:right">Moderate</th><th style="text-align:right">Different</th><th style="text-align:right">KS raw (median)</th><th style="text-align:right">KS robust z (median)</th></tr></thead>
      <tbody>{stats_rows}</tbody>
    </table></div>
  </div>
</div>

<div class="sec" id="n-gallery">
  <div class="sec-title">Per-Feature Plots after Robust z-score · {len(nok)} features</div>
  <div class="card">
    <div class="tab-bar" id="ncat-tabs">{norm_tabs}</div>
    <div class="gallery-controls">
      <input class="search-box" id="ngal-search" type="search" placeholder="Filter by feature name…">
      <select class="sort-select" id="ngal-sort">
        <option value="c_asc" selected>Sort: Q–Q CCC ↑ (least similar first)</option>
        <option value="c_desc">Sort: Q–Q CCC ↓ (most similar first)</option>
        <option value="k_desc">Sort: KS (robust z) ↓</option>
        <option value="kr_desc">Sort: raw KS ↓</option>
        <option value="name">Sort: name</option>
      </select>
      <span class="count-label" id="ngal-count"></span>
    </div>
    <div id="nfeat-grid" class="feat-grid"></div>
  </div>
</div>'''
else:
    norm_catalog, norm_view = {}, '<p class="note">Run run.py (or run.py --normalized-only) to populate this view.</p>'

css = (HERE / "dashboard_style.css").read_text(encoding="utf-8")
page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Cross-Cohort Transfer Analysis</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
{css}</style>
</head>
<body>

<header class="hdr">
  <div>
    <h1>Cross-Cohort Transfer Analysis · 90+ Study → I-CONECT</h1>
    <p>Language marker distribution shift · {n_feat} features · {len(fams)} families (ling / temporal / acoustic / semantic) · {n_large} large-shift (|d|≥0.8)</p>
  </div>
</header>

<div class="view-tabs" id="view-tabs">
  <button class="vtab active" data-view="raw">Raw distributions</button>
  <button class="vtab" data-view="norm">After robust z-score normalization</button>
</div>

<div id="view-raw">
<nav class="nav">
  <a href="#s-findings">Key Findings</a>
  <a href="#s-subgroups">I-CONECT Subgroups</a>
  <a href="#s-howto">How to Read</a>
  <a href="#s-examples">Examples</a>
  <a href="#s-risk">Transfer Risk</a>
  <a href="#s-heatmap">Heatmap</a>
  <a href="#s-overview">Overview Strip</a>
  <a href="#s-family">Family Browser</a>
  <a href="#s-top">Top Features</a>
  <a href="#s-gallery">Feature Gallery</a>
</nav>

<div class="stat-row" style="margin-top:12px">
  <div class="sc"><div class="sv">{n_feat}</div><div class="sl">Features screened</div></div>
  <div class="sc"><div class="sv" style="color:#d63939">{n_large}</div><div class="sl">Large shift |d|≥0.8 ({n_large / n_feat:.0%})</div></div>
  <div class="sc"><div class="sv" style="color:#d97706">{n_shape}</div><div class="sl">Shape change<br><span style="font-size:11px;color:var(--muted)">|d|&lt;0.5 but KS&gt;0.3</span></div></div>
  <div class="sc"><div class="sv" style="color:#d63939">{n_aucfams}/{len(fams)}</div><div class="sl">Families with AUC ≥ 0.99<br><span style="font-size:11px;color:var(--muted)">lowest: {auc["auc"].idxmin()} ({auc["auc"].min():.3f})</span></div></div>
</div>

<div class="sec" id="s-findings">
  <div class="sec-title">Key Findings</div>
  <div class="card"><ol class="kf">{findings_html}</ol></div>
</div>

<div class="sec" id="s-subgroups">
  <div class="sec-title">I-CONECT Splits into Long and Short Conversations</div>
  <div class="card">
    <p class="note">I-CONECT conversation durations are bimodal: the short subgroup ends at about 900 s and the long subgroup starts at about 1,000 s (90+ conversations rarely exceed 530 s). Subjects are split at {LONG_CONV_SEC:.0f} s. Comparing each subgroup with 90+ separates shifts caused by conversation length (only the long subgroup is out of range) from shifts that hold for the whole cohort (both subgroups out of range).</p>
    <div class="img-wrap"><img src="{b64(PLOTS / "family" / "ic_subgroups.png")}" alt="I-CONECT subgroup comparison" class="fam-img"></div>
    <p class="note" style="margin-top:14px">Medians per group, and support overlap = share of the subgroup inside the 90+ [5th, 95th] percentile range (red &lt; 50%).</p>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Feature</th><th>Family</th><th style="text-align:right">90+ median</th><th style="text-align:right">IC short median</th><th style="text-align:right">IC long median</th><th style="text-align:right">IC short overlap</th><th style="text-align:right">IC long overlap</th></tr></thead>
        <tbody>{sg_rows}</tbody>
      </table>
    </div>
    <ul class="howto" style="margin-top:14px">
      <li><b>Length-driven</b> (long subgroup only): total words, LIWC counts such as Adverbs, Simple TTR, content diversity. Normalising by word count or speaking time should remove most of this shift.</li>
      <li><b>Cohort-level</b> (both subgroups): pause ratio and turns per minute; Assent is also high in short conversations, which suggests different transcription conventions for fillers and back-channels.</li>
      <li><b>Short subgroup only</b>: participant talk-time ratio (short conversations are interviewer-led: median {sg_feat.set_index("feature").loc["ix_patient_talk_ratio", "median_ic_short"]:.2f} vs 90+ {sg_feat.set_index("feature").loc["ix_patient_talk_ratio", "median_90plus"]:.2f}).</li>
      <li>Acoustic features exist for {ac_ic} I-CONECT subjects, of whom {int(sg_fam.loc["acoustic", "n_subjects_ic_long"])} are from the long subgroup, so acoustic results mainly describe the long conversations.</li>
    </ul>
  </div>
</div>

<div class="sec" id="s-howto">
  <div class="sec-title">How to Read This Dashboard</div>
  <div class="two-col">
    <div class="card">
      <p class="note" style="font-weight:600;color:var(--fg)">Metrics</p>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Metric</th><th>What it measures</th><th>Rule of thumb</th></tr></thead>
        <tbody>
          <tr><td>Cohen's d</td><td>Difference in means, in pooled-SD units; sign = direction (positive: I-CONECT higher)</td><td class="muted-cell">|d| 0.2 / 0.5 / 0.8 = small / medium / large</td></tr>
          <tr><td>KS statistic</td><td>Largest gap between the two cumulative distributions; sensitive to any change in location, spread or shape</td><td class="muted-cell">0 = identical, 1 = fully separable; KS &gt; 0.3 is a clear difference</td></tr>
          <tr><td>Support overlap</td><td>Share of I-CONECT subjects inside the 90+ [5th, 95th] percentile range of a feature</td><td class="muted-cell">&lt; 50% = a 90+-trained model has to extrapolate</td></tr>
          <tr><td>Domain AUC</td><td>How well an L2 logistic regression on one family's features tells I-CONECT from 90+ subjects (5-fold CV)</td><td class="muted-cell">0.5 = indistinguishable, 1.0 = perfectly separable</td></tr>
        </tbody>
      </table></div>
    </div>
    <div class="card">
      <p class="note" style="font-weight:600;color:var(--fg)">Reading the |d| vs KS plots</p>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Region</th><th>Meaning</th><th>Action</th></tr></thead>
        <tbody>
          <tr><td>Low |d|, low KS</td><td>Distributions match</td><td class="muted-cell">Use as is</td></tr>
          <tr><td>High |d|, high KS</td><td>Whole distribution shifted or out of range</td><td class="muted-cell">Domain adaptation, or drop</td></tr>
          <tr><td>|d| &lt; 0.5, KS &gt; 0.3 (grey box)</td><td>Shape or spread changed while the mean stays put; d misses it</td><td class="muted-cell">Quantile transform, or drop</td></tr>
          <tr><td>|d| 1.5–2.5, KS ≈ 0.45 (band)</td><td>Only part of I-CONECT is shifted (the long subgroup); d overstates the shift</td><td class="muted-cell">Normalise by length, then re-check</td></tr>
        </tbody>
      </table></div>
      <p class="note" style="font-weight:600;color:var(--fg);margin-top:14px">Data caveats</p>
      <ul class="howto">
        <li>PCA plots and domain AUC use complete cases only (no imputation). Acoustic AUC is based on {ac_ic} of 156 I-CONECT subjects.</li>
        <li>The interruption rate is 0 for every I-CONECT subject: the transcripts have no overlapping turns, so this is a pipeline artefact, not behaviour.</li>
        <li>Each statistic compares one subject-level value per person (mean over that subject's recordings).</li>
      </ul>
    </div>
  </div>
</div>

<div class="sec" id="s-examples">
  <div class="sec-title">Representative Examples · Six Kinds of Shift</div>
  <p class="note">Each plot: top = density of I-CONECT (blue) and 90+ (orange), dashed lines = medians, ticks at the bottom = individual subjects; bottom = density difference (blue: more I-CONECT there, orange: more 90+). A and B show the two easy cases; C–F show where Cohen's d alone misleads and KS or the plot is needed.</p>
  <div class="ex-grid">{examples_html}</div>
</div>

<div class="sec" id="s-risk">
  <div class="sec-title">Transfer Risk · Family Overview</div>
  <div class="two-col">
    <div class="card"><p class="note">Domain discriminability AUC — how separable IC and 90+ subjects are by family features. AUC → 1.0 = high transfer risk. Computed on complete cases only (acoustic: 86 of 156 IC subjects, 69 of them from the long-conversation subgroup).</p>{svg_auc}</div>
    <div class="card"><p class="note">Cohen's d shift-category distribution per family (stacked, proportional).</p>{svg_shift}</div>
  </div>
</div>

<div class="sec" id="s-heatmap">
  <div class="sec-title">Transfer Risk Heatmap</div>
  <div class="card">
    <p class="note">Four scalars per family: domain AUC, mean |d|, mean KS, and the share of features whose support overlap is below 50% (extrapolation risk). Colours use fixed absolute scales per column (shown under each column name), not min-max across families, so a family is not painted green just for being the best of the four. The mean |ΔR| correlation divergence is kept in domain_auc.csv only.</p>
    <img src="{b64(PLOTS / "family" / "transfer_risk.png")}" alt="Transfer risk heatmap">
  </div>
</div>

<div class="sec" id="s-overview">
  <div class="sec-title">Cross-Family Feature Overview</div>
  <div class="card">
    <p class="note">|Cohen's d| per feature within each family (dots = features, bar = IQR, tick = median).</p>
    <img src="{b64(PLOTS / "family" / "_overview_by_family.png")}" alt="Cross-family overview">
  </div>
</div>

<div class="sec" id="s-family">
  <div class="sec-title">Family Summary Browser</div>
  <div class="card">
    <p class="note">Each family figure: A = PCA of subjects (complete cases); B = per-feature Cohen's d, marker colour = KS, right-hand label = support overlap (red &lt; 50%; families with &gt;20 features show the 20 largest by KS); C = |d| vs KS, colour = support overlap, grey box = shape-change zone.</p>
    <div class="tab-bar" id="fam-tabs">{"".join(tabs)}</div>
    <div id="fam-panels">
{"".join(panels)}</div>
  </div>
</div>

<div class="sec" id="s-top">
  <div class="sec-title">Top 20 Features by |Cohen's d|</div>
  <div class="card">
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Feature</th><th>Shift</th><th>Family</th><th style="text-align:right">|d|</th><th style="text-align:right">KS</th><th style="text-align:right">KS p-val</th></tr></thead>
        <tbody>{top_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<div class="sec" id="s-gallery">
  <div class="sec-title">Per-Feature Plot Gallery · {n_feat} features in {n_plots} plots</div>
  <div class="card">
    <p class="note">Acoustic plots show the _whole / _mean / _std variants of one base feature; the label shows the variant with the largest |d|.</p>
    <div class="tab-bar" id="cat-tabs">{cat_tabs}</div>
    <div class="gallery-controls">
      <input class="search-box" id="gal-search" type="search" placeholder="Filter by feature name…">
      <select class="sort-select" id="gal-sort">
        <option value="name">Sort: name</option>
        <option value="d_desc" selected>Sort: |d| ↓</option>
        <option value="d_asc">Sort: |d| ↑</option>
        <option value="ks_desc">Sort: KS ↓</option>
        <option value="shift">Sort: shift tier</option>
      </select>
      <span class="count-label" id="gal-count"></span>
    </div>
    <div id="feat-grid" class="feat-grid"></div>
  </div>
</div>
</div>

<div id="view-norm" style="display:none">
{norm_view}
</div>

<button id="theme-btn" title="Toggle theme" onclick="toggleTheme()">🌓</button>

<script>
// ── embedded data ──────────────────────────────────────────────────────
// CATALOG[cat] = [{{f: file, d: |d|, s: shift tier, k: KS}}]
const CATALOG = {json.dumps(catalog)};
const SHIFT_COLOR = {json.dumps(TIER_COLOR)};
// dashboard.html lives next to results/; plots are results/plots/<cat>/<file>.png
const IMG_BASE = "results/plots";

// ── family tabs ────────────────────────────────────────────────────────
document.getElementById("fam-tabs").addEventListener("click", e => {{
  const b = e.target.closest(".ftab"); if (!b) return;
  document.querySelectorAll(".ftab").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  document.querySelectorAll(".fpanel").forEach(p => p.style.display = "none");
  const p = document.getElementById("fp-" + b.dataset.fam);
  if (p) p.style.display = "";
}});

// ── gallery ────────────────────────────────────────────────────────────
let currentCat = "ling";
let observer;

function buildGallery(cat, query, sortMode) {{
  const grid = document.getElementById("feat-grid");
  let items = (CATALOG[cat] || []).map(e => ({{...e, key: e.f.replace(/\\.png$/, "")}}));
  if (query) {{
    const q = query.toLowerCase().replace(/[^a-z0-9]+/g, "_");
    items = items.filter(x => x.key.includes(q));
  }}
  const TIER = {{large:0, medium:1, small:2, negligible:3}};
  const v = x => (x === null || x === undefined) ? -1 : x;
  if (sortMode === "d_desc") items.sort((a, b) => v(b.d) - v(a.d));
  else if (sortMode === "d_asc") items.sort((a, b) => v(a.d) - v(b.d));
  else if (sortMode === "ks_desc") items.sort((a, b) => v(b.k) - v(a.k));
  else if (sortMode === "shift") items.sort((a, b) => (TIER[a.s] ?? 9) - (TIER[b.s] ?? 9));
  else items.sort((a, b) => a.f.localeCompare(b.f));

  document.getElementById("gal-count").textContent = items.length + " plots";
  if (observer) observer.disconnect();

  grid.innerHTML = items.map(it => {{
    const src = `${{IMG_BASE}}/${{cat}}/${{it.f}}`;
    const col = SHIFT_COLOR[it.s] || "#aaa";
    const chip = it.s ? `<span class="chip" style="background:${{col}}22;color:${{col}}">${{it.s}}</span>` : "";
    const stat = it.d === null ? "" : `|d|=${{it.d.toFixed(2)}} · KS=${{it.k.toFixed(2)}}`;
    return `<div class="feat-card">
      <div class="lazy-placeholder" data-src="${{src}}"><span>loading…</span></div>
      <div class="fc-info">
        <span class="fc-name" title="${{it.f}}">${{it.key}}</span>
        ${{chip}}
        <span class="fc-d">${{stat}}</span>
      </div>
    </div>`;
  }}).join("");

  observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
      if (!entry.isIntersecting) return;
      const ph = entry.target, src = ph.dataset.src;
      if (!src || ph.dataset.loaded) return;
      ph.dataset.loaded = "1";
      const img = new Image();
      img.onload = () => {{
        ph.innerHTML = ""; ph.appendChild(img);
        ph.style.minHeight = ""; ph.style.background = "";
        img.style.width = "100%"; img.style.display = "block"; img.style.borderRadius = "4px 4px 0 0";
        observer.unobserve(ph);
      }};
      img.onerror = () => {{
        ph.textContent = "⚠ not found: " + src; ph.style.fontSize = "10px";
        observer.unobserve(ph);
      }};
      img.src = src;
    }});
  }}, {{rootMargin: "400px"}});
  grid.querySelectorAll(".lazy-placeholder").forEach(ph => observer.observe(ph));
}}

document.getElementById("cat-tabs").addEventListener("click", e => {{
  const b = e.target.closest(".tab"); if (!b) return;
  document.querySelectorAll("#cat-tabs .tab").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  currentCat = b.dataset.cat;
  document.getElementById("gal-search").value = "";
  buildGallery(currentCat, "", document.getElementById("gal-sort").value);
}});
let searchTimer;
document.getElementById("gal-search").addEventListener("input", e => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => buildGallery(currentCat, e.target.value, document.getElementById("gal-sort").value), 200);
}});
document.getElementById("gal-sort").addEventListener("change", e => {{
  buildGallery(currentCat, document.getElementById("gal-search").value, e.target.value);
}});
buildGallery("ling", "", "d_desc");

// ── view tabs: raw vs z-score normalised ───────────────────────────────
// NORM_CATALOG[cat] = [{{f: plot path, n: feature, c: Q-Q CCC, k: KS (robust z), sr: SD ratio, d: raw d, kr: raw KS}}]
const NORM_CATALOG = {json.dumps(norm_catalog)};
const NORM_BASE = "results/normalized/plots";
let nCat = "ling", nObserver, normBuilt = false;

function lazyObserve(grid, obs) {{
  grid.querySelectorAll(".lazy-placeholder").forEach(ph => obs.observe(ph));
}}
function makeObserver() {{
  const obs = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
      if (!entry.isIntersecting) return;
      const ph = entry.target, src = ph.dataset.src;
      if (!src || ph.dataset.loaded) return;
      ph.dataset.loaded = "1";
      const img = new Image();
      img.onload = () => {{
        ph.innerHTML = ""; ph.appendChild(img);
        ph.style.minHeight = ""; ph.style.background = "";
        img.style.width = "100%"; img.style.display = "block"; img.style.borderRadius = "4px 4px 0 0";
        obs.unobserve(ph);
      }};
      img.onerror = () => {{ ph.textContent = "⚠ not found: " + src; ph.style.fontSize = "10px"; obs.unobserve(ph); }};
      img.src = src;
    }});
  }}, {{rootMargin: "400px"}});
  return obs;
}}

function buildNormGallery(cat, query, sortMode) {{
  const grid = document.getElementById("nfeat-grid");
  if (!grid) return;
  let items = (NORM_CATALOG[cat] || []).slice();
  if (query) {{ const q = query.toLowerCase(); items = items.filter(x => x.n.toLowerCase().includes(q)); }}
  if (sortMode === "c_asc") items.sort((a, b) => a.c - b.c);
  else if (sortMode === "c_desc") items.sort((a, b) => b.c - a.c);
  else if (sortMode === "k_desc") items.sort((a, b) => b.k - a.k);
  else if (sortMode === "kr_desc") items.sort((a, b) => b.kr - a.kr);
  else items.sort((a, b) => a.n.localeCompare(b.n));
  document.getElementById("ngal-count").textContent = items.length + " plots";
  if (nObserver) nObserver.disconnect();
  grid.innerHTML = items.map(it => {{
    const col = it.c < 0.90 ? "#d63939" : it.c < 0.95 ? "#d97706" : "#2a9d60";
    return `<div class="feat-card">
      <div class="lazy-placeholder" data-src="${{NORM_BASE}}/${{it.f}}"><span>loading…</span></div>
      <div class="fc-info">
        <span class="fc-name" title="${{it.n}}">${{it.n}}</span>
        <span class="chip" style="background:${{col}}22;color:${{col}}">CCC ${{it.c.toFixed(3)}}</span>
        <span class="fc-d">KS ${{it.k.toFixed(2)}}</span>
      </div>
    </div>`;
  }}).join("");
  nObserver = makeObserver();
  lazyObserve(grid, nObserver);
}}

document.getElementById("view-tabs").addEventListener("click", e => {{
  const b = e.target.closest(".vtab"); if (!b) return;
  document.querySelectorAll(".vtab").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  const v = b.dataset.view;
  document.getElementById("view-raw").style.display = v === "raw" ? "" : "none";
  document.getElementById("view-norm").style.display = v === "norm" ? "" : "none";
  if (v === "norm" && !normBuilt) {{ buildNormGallery(nCat, "", "c_asc"); normBuilt = true; }}
  window.scrollTo({{top: 0}});
}});
const nTabs = document.getElementById("ncat-tabs");
if (nTabs) {{
  nTabs.addEventListener("click", e => {{
    const b = e.target.closest(".tab"); if (!b) return;
    nTabs.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    nCat = b.dataset.cat;
    document.getElementById("ngal-search").value = "";
    buildNormGallery(nCat, "", document.getElementById("ngal-sort").value);
  }});
  let nTimer;
  document.getElementById("ngal-search").addEventListener("input", e => {{
    clearTimeout(nTimer);
    nTimer = setTimeout(() => buildNormGallery(nCat, e.target.value, document.getElementById("ngal-sort").value), 200);
  }});
  document.getElementById("ngal-sort").addEventListener("change", e => {{
    buildNormGallery(nCat, document.getElementById("ngal-search").value, e.target.value);
  }});
}}

// ── theme toggle ────────────────────────────────────────────────────────
function toggleTheme() {{
  const r = document.documentElement;
  r.setAttribute("data-theme", r.getAttribute("data-theme") === "dark" ? "light" : "dark");
  try {{ localStorage.setItem("theme", r.getAttribute("data-theme")); }} catch (e) {{}}
}}
try {{
  const t = localStorage.getItem("theme");
  if (t) document.documentElement.setAttribute("data-theme", t);
}} catch (e) {{}}
</script>
</body>
</html>
'''
(HERE / "dashboard.html").write_text(page, encoding="utf-8")
print(f"dashboard.html written: {len(page) / 1e6:.1f} MB, {n_plots} gallery plots, families={fams}")
