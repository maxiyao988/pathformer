# Research Workflow: Multi-Frequency Stock Forecasting with PathFormer

**Thesis title (draft):**
*Cross-Frequency Temporal Alignment and Adaptive Fusion for Stock Return Prediction: A Multi-Frequency Extension of PathFormer*

---

## 1. Core Research Question

> Given Hourly, Half-Day, Daily, and Weekly OHLCV data for a single stock (FSLR),
> can a cross-frequency alignment + adaptive fusion framework outperform
> single-frequency baselines in predicting future log returns at horizons
> of 5D, 10D, and 20D?

---

## 2. Innovation over Vanilla PathFormer

| Dimension | PathFormer (original) | This work |
|---|---|---|
| Frequency | Single frequency | Multiple real market frequencies |
| Scale | Multiple patch sizes within one frequency | Multiple frequencies, each with multi-scale patches |
| Alignment | N/A | Daily Anchor Date as unified prediction point |
| Fusion | N/A | Cross-frequency adaptive fusion / frequency routing |
| Domain | Weather, ETT (public benchmarks) | Financial OHLCV (FSLR) |

---

## 3. Data Pipeline

### 3.1 Datasets

| Frequency | Source file | Coverage | Rows |
|---|---|---|---|
| Hourly | `FSLR_hourly_bloomberg.csv` | 2018-05-31 → 2026-06-05 | 14,038 |
| Half-Day | `FSLR_halfday_full.csv` (to be built) | same as hourly | ~4,028 |
| Daily | `FSLR_daily.csv` | 2006-11-17 → 2026-06-05 | 4,916 |
| Weekly | `FSLR_weekly.csv` | 2006-11-13 → 2026-06-08 | 1,022 |

### 3.2 Half-Day Definition

- **Morning session**: 09:30 – 12:30 (first 3 hourly bars)
- **Afternoon session**: 13:30 – 15:30 (last 2 hourly bars, if 7-bar day)
- Aggregation: `open` = first bar open, `high` = max, `low` = min, `close` = last bar close, `volume` = sum

### 3.3 Alignment Scheme

```
Anchor Date  D  (each trading day in Daily dataset)
│
├── Hourly   → last H_h  bars with timestamp < market open on D+1
├── Half-Day → last H_hd half-day sessions with date ≤ D
├── Daily    → last H_d  daily bars with date ≤ D
└── Weekly   → last H_w  weekly bars with week_end ≤ D
│
└── Target y → log return from close(D) to close(D + horizon)
                horizon ∈ {5, 10, 20} trading days
```

**Information constraint**: all inputs must be observable at end-of-day D.
No look-ahead leakage.

### 3.4 Features per bar

`[open, high, low, close, volume]` — 5 features, standardized per sample with RevIN.

---

## 4. Lookback Window Experiments (Two-Stage)

### Stage 1 — Per-Frequency Sensitivity

Fix all other frequencies at their default window. Vary one frequency at a time.

| Frequency | Candidates | Default |
|---|---|---|
| Hourly | 24 / **48** / 96 | 48 |
| Half-Day | 20 / **40** / 60 | 40 |
| Daily | 60 / **90** / 120 | 90 |
| Weekly | 26 / **52** / 78 | 52 |

Metric: Validation MSE on 20D horizon.
Select best window per frequency → call it `W*`.

### Stage 2 — Global Configuration Comparison

Compare three complete configurations on the test set:

| Config | Hourly | Half-Day | Daily | Weekly |
|---|---|---|---|---|
| Short | 24 | 20 | 60 | 26 |
| Default (`W*`) | 48 | 40 | 90 | 52 |
| Long | 96 | 60 | 120 | 78 |

---

## 5. Model Architecture

```
Input per frequency f:
  X_f  ∈  R^(B × H_f × 5)

Step 1 — Intra-frequency multi-scale encoding (PathFormer AMS per frequency)
  Z_f  ∈  R^(B × H_f × d_model)

Step 2 — Cross-frequency fusion (new contribution)
  Option A: Concat + linear projection
  Option B: Cross-attention (Daily queries, others as key/value)
  Option C: Adaptive frequency routing (MoE-style gate)

Step 3 — Prediction head
  ŷ  ∈  R^(B,)   for regression (return)
  ŷ  ∈  R^(B, 2) for classification (up/down)
```

The cross-frequency fusion module (Step 2) is the primary architectural contribution.

---

## 6. Experiment Plan

### 6.1 Main Experiments

**Task**: Predict future log return at horizon ∈ {5D, 10D, 20D}

| Model | Description |
|---|---|
| LR | Linear Regression on concatenated raw features |
| Transformer-D | Vanilla Transformer, daily only |
| PathFormer-D | PathFormer, daily only |
| **Ours-AllFreq** | PathFormer + cross-frequency fusion, all 4 frequencies |

Primary metric: **MSE, MAE, Directional Accuracy (DA), Pearson Correlation**

### 6.2 Ablation Study — Frequency Contribution

Remove one frequency at a time, keep others fixed:

| Config | H | HD | D | W |
|---|---|---|---|---|
| Ours-AllFreq | ✓ | ✓ | ✓ | ✓ |
| -Hourly | ✗ | ✓ | ✓ | ✓ |
| -HalfDay | ✓ | ✗ | ✓ | ✓ |
| -Weekly | ✓ | ✓ | ✓ | ✗ |
| Daily-only | ✗ | ✗ | ✓ | ✗ |

This directly answers: *"Does each frequency contribute?"*

### 6.3 Fusion Mechanism Comparison

| Config | Fusion |
|---|---|
| Concat | Simple concatenation + MLP |
| CrossAttn | Cross-attention (Daily as query) |
| FreqRoute | Adaptive frequency routing (proposed) |

### 6.4 Prediction Task Variants

| Task | Target | Loss |
|---|---|---|
| Regression | log return (continuous) | MSE + MAE |
| Classification | sign of return (binary) | BCE, report DA + F1 |

---

## 7. Data Split

| Set | Ratio | Date range (approx.) |
|---|---|---|
| Train | 70% | 2018-05 → 2023-10 |
| Validation | 15% | 2023-10 → 2024-10 |
| Test | 15% | 2024-10 → 2026-06 |

Split is **chronological** (no random shuffle). No future leakage.

---

## 8. Evaluation Metrics

| Metric | Formula | Task |
|---|---|---|
| MSE | mean squared error | regression |
| MAE | mean absolute error | regression |
| DA | % of correct sign predictions | both |
| Pearson r | correlation(ŷ, y) | regression |
| F1 | harmonic mean precision/recall | classification |

---

## 9. Task Checklist

```
[x] Task 0   Standardize data format (processed/)
[x] Task 1   Full EDA (hourly / daily / weekly)
[ ] Task 3   Rebuild half-day dataset from new hourly
[ ] Task 4   Verify cross-frequency alignment + info constraint
[ ] Task 5   Build multiscale_v2 dataset (all 3 horizons)
[ ] Task 6   Dataset audit (shapes, NaN, leakage check)
[ ] Task 7a  Stage 1 lookback sensitivity experiments
[ ] Task 7b  Stage 2 global config comparison
[ ] Task 8   Main experiments (4 models × 3 horizons)
[ ] Task 9   Ablation: frequency contribution
[ ] Task 10  Ablation: fusion mechanism
[ ] Task 11  Classification variant
[ ] Task 12  Write up results
```

---

## 10. File Structure (target)

```
pathformer/
├── dataset/
│   ├── processed/          ← Task 0 output (standardized source)
│   │   ├── FSLR_hourly_bloomberg.csv
│   │   ├── FSLR_daily.csv
│   │   └── FSLR_weekly.csv
│   ├── merged/             ← Task 3 output
│   │   └── FSLR_halfday_full.csv
│   ├── multiscale_v2/      ← Task 5 output
│   │   ├── X_hourly.npy    (N, H_h,  5)
│   │   ├── X_halfday.npy   (N, H_hd, 5)
│   │   ├── X_daily.npy     (N, H_d,  5)
│   │   ├── X_weekly.npy    (N, H_w,  5)
│   │   ├── y_5d.npy        (N,)
│   │   ├── y_10d.npy       (N,)
│   │   ├── y_20d.npy       (N,)
│   │   └── meta.csv        anchor_date / split / horizon
│   └── eda/                ← Task 1 output
├── scripts/python/
│   ├── task0_standardize_format.py
│   ├── task1_eda.py
│   ├── task3_build_halfday.py
│   ├── task4_verify_alignment.py
│   ├── task5_build_multiscale_v2.py
│   ├── task6_audit_dataset.py
│   ├── task7_lookback_sensitivity.py
│   ├── task8_main_experiments.py
│   ├── task9_ablation_frequency.py
│   └── task10_ablation_fusion.py
└── pathformer/
    └── models/
        ├── PathFormer.py          ← original
        └── MultiFreqPathFormer.py ← to be built
```

---

## 11. Key Design Decisions to Justify in Paper

1. **Why Daily Anchor?** → Daily is the natural decision frequency for a portfolio manager. Hourly and weekly data provide finer and coarser context respectively.

2. **Why these lookback windows?** → Justified by Stage 1 sensitivity experiments.

3. **Why log return, not price?** → Stationarity; comparable across time.

4. **Why 5D/10D/20D?** → Covers short-term (weekly), medium-term (biweekly/monthly) horizons relevant to rebalancing cycles.

5. **Why not longer history for hourly?** → Hourly only available from 2018; longer windows reduce sample count; memory constraints.
