# Cross-Cohort Feature Distribution Analysis: 90+ vs I-CONECT (2026-10-02)

**Goal.** For every downstream feature extracted from both cohorts, characterise the per-cohort
distribution and quantify how different the distributions are. This is a purely descriptive,
label-free study — no cognitive-status labels are used. It extends the alignment study
(`../2026-07-25-90plus-vs-iconect-alignment/`) by adding per-feature distribution plots and
a shape-sensitive difference metric (KS test) to the already-computed Cohen's d.

**Caveat (inherited from alignment study).** The distributional shift confounds three
independent sources: **GENRE** (I-CONECT: semi-structured ~30-min video-chat conversation vs
90+: structured cognitive-assessment Q&A), **ASR system** (DeepSpeech2 vs Whisper-large-v3),
and **POPULATION** (age ~75–90 vs all ≥90). This analysis *describes* the shift; it does not
attribute it to any single cause.

## Execution plan

### Step 0 — Prerequisites (data on Great Lakes / turbo)

All processed feature tables live under `data/processed/` (gitignored; pulled from turbo):

| dataset | table | asr_config | rows | content |
|---|---|---|---|---|
| `iconect` | `recording_features.parquet` | `sam_v0_ds2` | ~1,400 recs | 99 linguistic features |
| `iconect` | `recording_temporal_sam_v0_ds2.parquet` | `sam_v0_ds2` | ~1,400 recs | 24 temporal/interaction features |
| `iconect` | `recording_highsignal_sam_v0_ds2.parquet` | `sam_v0_ds2` | ~1,400 recs | 6 high-signal markers |
| `the90plus` | `recording_features.parquet` | `whisper-lv3-pyannote` | ~7,800 recs | 99 linguistic features |
| `the90plus` | `recording_temporal_whisper-lv3-pyannote.parquet` | `whisper-lv3-pyannote` | ~7,800 recs | 24 temporal/interaction features |
| `the90plus` | `recording_highsignal_whisper-lv3-pyannote.parquet` | `whisper-lv3-pyannote` | ~7,800 recs | 6 high-signal markers |
| `the90plus` | `role_map.parquet` | — | 7,899 | role usability filter |

If `recording_highsignal_*.parquet` has not yet been built, run:
```bash
python -m pipelines.build_highsignal_features --dataset iconect --asr-config sam_v0_ds2
python -m pipelines.build_highsignal_features --dataset the90plus --asr-config whisper-lv3-pyannote
```

Acoustic features (`recording_acoustic_*.parquet`, 140 features) are available for I-CONECT
but may not yet exist for 90+; they are **excluded from the initial analysis** to keep the
comparison clean (ASR confound is already present; acoustic is doubly confounded by the
different recording setups and microphones). They can be added as a separate sub-analysis.

### Step 1 — Data loading and subject-level aggregation

Pattern (mirrors `../2026-07-25-90plus-vs-iconect-alignment/run.py`):

```python
# I-CONECT: merge linguistic + temporal + high-signal; filter to sam_v0_ds2
# 90+: merge same tables; filter to whisper-lv3-pyannote + role_usable==True

# Aggregate to subject level: mean across visits/transcripts per subject_id
# Result: one row per subject, columns = feature values
```

**Grain**: subject-level mean. This matches the alignment study and the univariate AUC results.
Recording-level grain would also be valid for within-session analyses.

### Step 2 — Feature availability filter

Retain a feature only if ≥ 80% of subjects in **each** cohort have a non-NaN value.
The alignment study confirmed 123/123 linguistic+temporal features pass this threshold.
High-signal features may have higher NaN rates (especially `hs_semantic_similarity_*`,
which requires ≥2 sentences) — apply the same 80% filter.

### Step 3 — Missing value handling

- For Cohen's d / KS: use pairwise complete (drop NaN within each cohort's
  vector for that feature). No imputation — the goal is to describe the actual distribution.
- Report per-feature n for each cohort.

### Step 4 — Distribution plot per feature

For each feature, a single figure with:
- **Kernel density estimate (KDE)** for both cohorts, overlaid, semi-transparent fill
- **Median line** (vertical dashed) per cohort
- **Rug plot** (horizontal tick marks at bottom, jittered) for raw values
- Cohort colors: I-CONECT `#4477AA`, 90+ `#EE7733` (Paul Tol colorblind-safe pair,
  consistent with the alignment study)
- Annotation box: Cohen's d (+ shift tier), KS stat + p

### Step 5 — Distribution difference metrics

For each feature:

| metric | method | rationale |
|---|---|---|
| **Cohen's d** | `(μ_IC − μ_90) / pooled_SD` | standardised mean difference; matches existing alignment CSV |
| **KS statistic** | `scipy.stats.ks_2samp` | non-parametric; detects any distributional difference regardless of shape; p < 0.05 → significant |

The two are complementary: d captures location shift (and its direction); |d| < 0.5 with
KS > 0.3 flags shape/scale change that d misses. Wasserstein (normalised by pooled SD) and
KDE-based Jensen-Shannon distance were computed in an earlier run and dropped: normalised
Wasserstein equals |d| exactly whenever one cohort stochastically dominates the other
(Spearman ρ = 0.98 with |d| across 547 features), and JS was largely redundant with KS
while being sensitive to KDE bandwidth and outliers.

### Step 6 — Summary table

`results/all_features_stats.csv` — one row per feature:

```
feature, family, n_iconect, n_90plus, ic_mean, ic_median, ic_iqr, np_mean, np_median, np_iqr,
ic_std, np_std, cohens_d, abs_d, ks_stat, ks_pval, shift_category
```

Where `shift_category` = "negligible" |d|<0.2, "small" <0.5, "medium" <0.8, "large" ≥0.8.

### Step 7 — Batch plots

`results/plots/{feature_slug}.png` — one PNG per feature (123 plots).

### Output layout

```
docs/analysis/cross-cohort/2026-10-02-feature-distributions/
├── README.md                        ← this file
├── demo/
│   ├── run_demo.py                  ← one-feature demo (MTLD)
│   ├── mtld_distribution.png        ← generated by run_demo.py on GL
│   └── mtld_stats.json              ← generated by run_demo.py on GL
├── run.py                           ← full batch analysis (all 547 features), incl. the
│                                       per-cohort robust z-score shape analysis (results/normalized/;
│                                       redo only that part with --normalized-only)
├── build_dashboard.py               ← builds dashboard.html from results/ (run after run.py)
├── dashboard_style.css              ← stylesheet used by build_dashboard.py
├── dashboard.html                   ← interactive dashboard (key findings, I-CONECT subgroups,
│                                       how-to-read guide, 4 families, gallery of 267 plots)
└── results/
    ├── all_features_stats.csv       ← generated by run.py on GL
    ├── domain_auc.csv               ← per-family AUC, mean |d|, mean KS, overlap
    ├── ic_subgroup_summary.csv      ← I-CONECT long vs short conversations (split 900 s):
    │                                   medians and support overlap for key features
    ├── ic_subgroup_family.csv       ← same split: share of out-of-range features per family
    ├── normalized/
    │   ├── feature_shape_stats.csv  ← per feature after per-cohort robust z-score (median/IQR):
    │   │                               Q–Q CCC on 5th–95th pct (primary), KS (robust z),
    │   │                               Q–Q Pearson r and slope, raw SD ratio, raw d / KS
    │   └── plots/<family>/<slug>.png ← robust-z densities + Q–Q plot (546 features)
    └── plots/
        └── {feature_slug}.png       ← one per feature
```

**Scripts run from the repo root on Great Lakes** (where `data/processed/` is populated):
```bash
# Demo (one feature):
python docs/analysis/cross-cohort/2026-10-02-feature-distributions/demo/run_demo.py

# Full batch (all features, ~123+ plots):
python docs/analysis/cross-cohort/2026-10-02-feature-distributions/run.py
```

### Feature inventory

Features are grouped into four modality families (`family` column; see `family_of` in
`run.py`). Numbers below are from the current run (`results/domain_auc.csv`).

| family | n features | contents | mean \|Cohen's d\| | domain AUC |
|---|---|---|---|---|
| `ling` | 99 | LIWC, syntactic (L2SCA-inspired), lexical diversity (TTR/MTLD), response length | 1.70 | 0.998 |
| `temporal` | 24 | `pt_*` participant timing + `ix_*` turn-taking / interaction | 1.50 | 1.000 |
| `acoustic` | 420 | eGeMAPS (88 base) + MFCC (52 base) × `_whole/_mean/_std` | 1.10 | 1.000 (86 IC complete cases) |
| `semantic` | 4 | content density / diversity / frequency, semantic spread | 0.78 | 0.931 |

Within `ling`, count-type features (LIWC categories, clause/T-unit counts) carry most of
the shift and scale with conversation length; ratio / length-corrected features (MTLD,
per-clause and per-T-unit ratios) show |d| < 0.5.

## Demo: MTLD (see `demo/`)

See `demo/run_demo.py` and its outputs. Feature chosen: **Measure of lexical textual
diversity [mtld_ma_wrap]** (`ling` family, lexical-diversity subset, |d|=0.33) — among the most genre-robust features,
a length-corrected vocabulary diversity index, and meaningful for the research question
(lexical diversity declines in MCI/dementia). Small-to-moderate shift makes distributions
partially overlapping and informative to visualise.
