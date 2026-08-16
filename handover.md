# Green Energy Stock Prediction Project - Handover

## Project Goal

Develop a single-modality, multi-scale stock forecasting framework for green energy equities using machine learning and transformer-based architectures.

Primary research question:

Can multi-scale temporal representations improve stock return prediction compared with traditional forecasting approaches?

Long-term objective:

Design a hybrid architecture that explicitly models:

* Short-term dynamics (Attention)
* Medium-term dynamics (Linear Forecasting)
* Long-term dynamics (Pooling / Trend Extraction)

and combines them through a learnable fusion layer.

Potential publication targets:

* AI for Finance
* Financial Machine Learning
* Time Series Forecasting
* Quantitative Finance Conferences

---

# Stock Universe

## Solar

* FSLR
* ENPH
* SEDG
* CSIQ
* RUN

## Hydrogen (future expansion)

* PLUG
* BE
* FCEL

## Renewable Utilities (future expansion)

* NEE
* CWEN

## ETFs

* ICLN
* TAN
* SPY

Sample period:

2016-01-01 to 2025-12-31

---

# Current Project Structure

Root:

* project_progress.md
* handover.md
* dataset/
* scripts/
* pathformer/

Important folders:

dataset/finance/

Contains:

* fslr.csv
* enph.csv
* sedg.csv
* csiq.csv
* run.csv
* icln_raw.csv
* tan_raw.csv
* spy_raw.csv
* solar_universe.csv
* solar_excess_return.csv

dataset/financial_ml/

Contains:

* X_raw.npy
* y_raw.npy
* X_excess.npy
* y_excess.npy
* X_market.npy
* y_market.npy
* meta_market.csv

scripts/python/

Contains:

* baseline_linear_raw.py
* baseline_linear_excess.py
* baseline_linear_raw_timesplit.py
* baseline_linear_excess_timesplit.py
* build_market_augmented_dataset.py
* build_time_split_dataset.py
* evaluate_fslr.py
* meta_test.py

---

# Completed Work

## Phase 0

PathFormer Reproduction

Dataset:

Weather

Status:

Completed successfully.

Purpose:

Verify environment and training pipeline.

---

## Phase 1

Single Stock Forecasting

Stock:

FSLR

Features:

* Open return
* High return
* Low return
* Close return
* Log volume

Model:

PathFormer

Sequence Length:

96

Prediction Horizon:

8

Results:

MSE:

1.1533

MAE:

0.6887

Correlation:

-0.033

Direction Accuracy:

48.1%

Conclusion:

No meaningful predictive power.

---

## Phase 2

Multi-Stock Solar Dataset

Universe:

* FSLR
* ENPH
* SEDG
* CSIQ
* RUN

Observations:

12560

Purpose:

Increase sample size and learn sector-level patterns.

---

## Phase 3

Label Engineering

Constructed:

### Raw Return

Future 5-day stock return

### Excess Return

Future 5-day stock return minus ICLN benchmark return

Generated:

solar_excess_return.csv

---

## Phase 4

Linear Benchmarks

Time Split:

Train:

2016-2022

Validation:

2023

Test:

2024-2025

### Raw Return

Correlation:

-0.0459

Direction Accuracy:

48.0%

### Excess Return

Correlation:

-0.0186

Direction Accuracy:

50.5%

Key finding:

OHLCV alone contains little predictive information.

---

## Phase 5

Market-Augmented Dataset

Added:

* ICLN return
* TAN return
* SPY return

Feature dimension:

5 → 8

Generated:

X_market.npy

Shape:

(12055, 96, 8)

Generated:

y_market.npy

Generated:

meta_market.csv

---

# Current Blocker

Market-augmented benchmark cannot run.

Diagnostic:

NaN in X:

0

NaN in y:

8

Inf in X:

0

Inf in y:

113

Most likely cause:

Incorrect label construction.

Need to verify whether:

dataset/finance/*.csv

contain:

* returns

or

* prices

Current suspicion:

The "close" column already contains transformed returns.

If true:

Current future-return calculation is invalid.

---

# Research Lessons So Far

1. Directly applying PathFormer to a single stock does not generate alpha.

2. Multi-stock learning is necessary.

3. Proper time-based evaluation is critical.

4. Benchmark construction is more important than model complexity at the current stage.

5. Signal discovery should precede architecture innovation.

---

# Recommended Next Steps

Priority 1

Fix market-augmented labels.

Verify:

* NaN source
* Inf source
* return vs price issue

---

Priority 2

Run:

baseline_linear_market_timesplit.py

Goal:

Determine whether market information improves predictive performance.

---

Priority 3

Technical Factor Engineering

Add:

* 5-day momentum
* 20-day momentum
* 60-day momentum
* 20-day volatility
* 60-day volatility
* MA20
* MA60
* RSI

Goal:

Determine whether predictive information exists in classical technical signals.

---

Priority 4

Benchmark Ladder

Build:

Linear Regression

↓

MLP

↓

LSTM

↓

Transformer

↓

PathFormer

This ladder is required before proposing a new architecture.

---

Priority 5

Core Research Contribution

Implement hybrid multi-scale model:

Short-Term Branch:

Attention

Medium-Term Branch:

Linear Forecasting

Long-Term Branch:

Pooling / Trend Extraction

Fusion Layer:

Learnable aggregation

This is currently the strongest candidate for a publishable contribution.

---

# Current Working Philosophy

Do not add model complexity before demonstrating signal existence.

Research order:

Signal Discovery

↓

Feature Engineering

↓

Benchmarks

↓

Deep Learning

↓

Hybrid Architecture

Not:

Transformer

↓

Another Transformer

↓

Yet Another Transformer
