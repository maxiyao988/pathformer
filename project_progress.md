# Green Energy Stock Prediction Project

## Scope (Current Active Track)

**FROZEN PANEL DECISION (2026-08-16):** **Primary panel = the 17-stock balanced Daily + Weekly panel, common period 2016-01-25 to 2026-06-05** (`AES, BEP, BLDP, BLNK, CSIQ, CWEN, DQ, ENPH, FCEL, FSLR, HASI, JKS, NEE, ORA, PLUG, RUN, SEDG` — see `panel_candidate_universe_summary.csv`, `n_tickers=17` row). This is the universe all Phase 2+ modeling (Ridge/LSTM/Transformer/PathFormer, Experiment 2/3/4) targets. **The full 24-stock universe is NOT deleted** — it remains in `panel_universe.GREEN_ENERGY_UNIVERSE` and the built dataset under `dataset/multiscale_dataset/panel/`, retained explicitly as a future robustness / unbalanced-panel extension, not as the primary experiment.

**Pre-freeze verification (ran 2026-08-16, PASSED):** `scripts/python/panel_verify_17stock_sample_index.py` confirmed that for the 17-stock panel's train split, Daily-only, Weekly-only, and Daily+Weekly settings all resolve to the exact same 31,008-sample `(ticker, anchor_date)` index (not just by construction — checked empirically via non-NaN/non-Inf masks on `X_daily`/`X_weekly`). This was the required sanity gate before freezing the panel choice.

**Current active track (2026-08-16, supersedes all wording below that implies a 4-frequency panel):** the active research track is a **Daily + Weekly multi-stock panel** with frequency-specific encoders, late fusion, and an optional per-frequency adaptive scale router. This is a deliberate scope reduction from the earlier "20–50 tickers x Hourly/Half-Day/Daily/Weekly" framing, driven by a verified data constraint: panel-wide Hourly/Half-Day OHLCV does not exist yet (see Experiment 2's data-availability note below). **FSLR is retained as a single-stock, full-frequency (Hourly/Half-Day/Daily/Weekly) case-study / diagnostic model-comparison track.** It provides both the advisor-requested Experiment-1 model-comparison setting (one ML model, one DL model, two improved Transformer algorithms) and the diagnostic evidence showing why naive full-frequency fusion is not a viable panel blueprint. It is not the main panel proving ground or the final router-interpretability setting.

Project focus was single-stock, multi-scale, single-modality modeling on FSLR using PathFormer-style time-series baselines. This FSLR-only setup is preserved as the case-study / diagnostic model-comparison track that motivates the stable panel design, not as the main panel experimental subject going forward.

- Ticker: FSLR
- Usable aligned sample period: 2011-11-02 to 2026-05-07
- Raw coverage differs by frequency; see dataset artifacts for exact start/end dates.
- Data source: Yahoo Finance and Bloomberg
- Frequencies: hourly, halfday, daily, weekly
- Features: OHLCV-derived arrays in multi-scale windows
- Targets: future log return at 5d / 10d / 20d
- Split protocol: 70% train / 15% val / 15% test (time-ordered)

---

## Current Objective

**Build and evaluate a stable Daily + Weekly panel framework** (frequency-specific encoders + late fusion + optional adaptive scale router), while preserving FSLR as the single-stock case-study / diagnostic model-comparison track that motivates the stable panel design. The FSLR Task 8 protocol below is historical/completed context, not the active objective — those items are paused per the Advisor Pivot section.

Primary scripts (FSLR historical reference, already completed):

- scripts/python/task8_baseline_pathformer.py
- scripts/python/task8_baseline_pathformer_latefusion.py
- scripts/python/task8_pathformer_multiseed_robustness.py

Primary outputs (FSLR historical reference, already completed):

- dataset/audit/task8_latefusion_seed0.csv
- dataset/audit/task8_latefusion_seed1.csv
- dataset/audit/task8_latefusion_seed21.csv
- dataset/audit/task8_latefusion_seed42.csv
- dataset/audit/task8_latefusion_seed3407.csv

---

## Completed Milestones

### Data and Pipeline

- Multi-frequency data ingestion and cleaning completed.
- Cross-frequency alignment audit completed with no leakage in retained samples.
- Multi-scale dataset rebuilt and audited.
- Lookback selection study completed; working window set (w_star):
  - hourly 24, halfday 20, daily 90, weekly 26

### Task 8 Baselines (Established)

- Linear baseline completed.
- Vanilla Transformer baseline completed.
- SWiM-style baseline completed.
- PathFormer baseline implemented and repeatedly stress-tested.

### Advisor-Directed Protocol (Completed)

- Static baselines completed (`zero`, `train_mean`) in the same split/metric pipeline.
- Tiny-subset overfit diagnostics completed with contiguous windows (`tiny_subset_size=64`).
- Per-horizon target standardization confirmed (5d / 10d / 20d independent stats).
- Loss switched to Huber in active experiments.
- Batch size and LR constrained to advisor-approved ranges in formal runs (`batch_size=32`, `lr=1e-4`).
- Multi-frequency late fusion implemented with separate encoders per frequency.
- Concat vs Gated late-fusion comparison completed.
- Fixed-seed robustness run completed with seeds: `0, 1, 21, 42, 3407`.
- Rank-based metric reporting completed (`test_rank_corr`, i.e., IC proxy).

---

## Baseline Snapshot (Reference)

### Linear (test)

- 5d: MSE 0.00654394, MAE 0.059478, Corr -0.0096, DA 0.4617
- 10d: MSE 0.01168564, MAE 0.081490, Corr -0.0263, DA 0.4781
- 20d: MSE 0.02461462, MAE 0.123228, Corr -0.0415, DA 0.4690

### Vanilla Transformer (test)

- 5d: MSE 0.00609633, MAE 0.056006, Corr -0.0919, DA 0.5401
- 10d: MSE 0.01046809, MAE 0.075947, Corr 0.1553, DA 0.5164
- 20d: MSE 0.02078742, MAE 0.112402, Corr 0.2787, DA 0.5292

### SWiM-style (test)

- 5d: MSE 0.00609194, MAE 0.055916, Corr 0.1579, DA 0.5365
- 10d: MSE 0.01049132, MAE 0.076055, Corr 0.2492, DA 0.5164
- 20d: MSE 0.02183441, MAE 0.115850, Corr 0.3090, DA 0.5292

---

## PathFormer Debug Timeline (Condensed)

### Phase A: Single-Backbone Instability (Rejected)

The original single-backbone PathFormer (`concatenate all frequencies -> one backbone`) repeatedly alternated between two failure modes:

- amplitude explosion (pred std far above target std), and
- constant-collapse (pred std near 1e-9 to 1e-8).

Diagnostic checkpoints showed scale pathologies were structural and persisted despite broad hyperparameter sweeps.

### Phase B: Teacher-Aligned Refactor

Implemented separate encoder per frequency (hourly/halfday/daily/weekly) and late fusion (`concat` vs `gated`) in:

- `scripts/python/task8_baseline_pathformer_latefusion.py`

This removed catastrophic instability and enabled reproducible multi-seed runs.

### Phase C: 5-Seed Formal Run Completed

Formal run completed on seeds `0, 1, 21, 42, 3407` with `batch_size=32`, `lr=1e-4`, and Huber loss.

Mean±Std summary (test set):

- 5d / concat: MSE 0.006104 ± 0.000033, MAE 0.056212 ± 0.000259, Corr -0.0575 ± 0.0358, Rank Corr -0.0457 ± 0.0361, DA 0.5245 ± 0.0235
- 5d / gated: MSE 0.006086 ± 0.000015, MAE 0.056051 ± 0.000120, Corr -0.0730 ± 0.0284, Rank Corr -0.0721 ± 0.0778, DA 0.5219 ± 0.0161
- 10d / concat: MSE 0.010414 ± 0.000148, MAE 0.075881 ± 0.000763, Corr 0.0774 ± 0.0687, Rank Corr 0.0339 ± 0.0913, DA 0.5299 ± 0.0397
- 10d / gated: MSE 0.010507 ± 0.000024, MAE 0.075926 ± 0.000097, Corr -0.0722 ± 0.0525, Rank Corr -0.0256 ± 0.1483, DA 0.5142 ± 0.0033
- 20d / concat: MSE 0.021465 ± 0.000298, MAE 0.113641 ± 0.000949, Corr 0.0062 ± 0.1230, Rank Corr 0.0086 ± 0.1104, DA 0.5197 ± 0.0319
- 20d / gated: MSE 0.021273 ± 0.000031, MAE 0.113206 ± 0.000033, Corr 0.0453 ± 0.0578, Rank Corr 0.0319 ± 0.1451, DA 0.5288 ± 0.0008

**Terminology note:** these are single-stock time-series `Rank Corr` (Spearman rank correlation between predicted and realized returns over time), not a cross-sectional IC. For the panel, the correct metric is a **cross-sectional Rank IC**, computed per test date across tickers and then averaged over dates — the two are not directly comparable across the FSLR and panel tracks. See `PathFormer Debug Timeline` naming above for where the older `RankCorr(IC)` label was used loosely.

---

## Ridge Panel Baseline — DONE / FROZEN

- Universe: frozen 17-stock balanced panel
- Daily / Weekly / Daily+Weekly
- Horizons: 5d / 10d / 20d
- Formal solver: **SVD**
- Alpha selected using validation only
- Cross-sectional Rank IC computed by date
- Numerical solver robustness check passed
- Final solver ambiguity resolved: **Ridge is frozen**
- No leakage
- Ridge does not beat naive baseline on pooled MSE
- Weekly-only has the best pooled MSE at 5d / 10d / 20d
- Daily+Weekly has the best mean cross-sectional Rank IC at 5d / 10d / 20d
- Ridge predictions are under-dispersed

This is the frozen formal machine-learning benchmark for the panel.

Relevant files:

- `scripts/python/panel_baseline_ridge.py`
- `scripts/python/panel_ridge_validation.py`
- `scripts/python/panel_ridge_solver_final_check.py`
- `dataset/audit/panel_ridge_solver_final_check.csv`
- `dataset/audit/panel_ridge_alpha_validation_curve.csv`
- `dataset/audit/panel_ridge_solver_final_predictions.csv`
- `dataset/audit/panel_ridge_solver_final_summary.txt`
- `dataset/audit/panel_naive_baseline_summary.csv`

### Final solver decision

The final numerical-stability check passed with the following solver agreement:

- auto vs lsqr_tight min prediction corr ≈ 0.9999
- auto vs SVD min prediction corr ≈ 0.9999
- lsqr_tight vs SVD min prediction corr ≈ 1.0000
- Best-MSE frequency choice does not change across solvers
- Best-Rank-IC frequency choice does not change across solvers
- Ridge freeze decision = **True**

SVD is the formal Ridge solver for the project benchmark.

---

## LSTM Panel Baseline — SINGLE-SEED COMPLETE / ARCHITECTURE FROZEN

The research question remains explicit:

> Does nonlinear sequential modeling add value over Ridge?

This LSTM baseline is now a completed single-seed development benchmark for the frozen panel. It is not the final multi-seed paper result.

### Completed implementation

The panel-ready LSTM baseline exists and has been run successfully:

- Script: `scripts/python/panel_baseline_lstm.py`

Architecture and training protocol:

Daily-only:

```text
Daily sequence
→ 1-layer unidirectional LSTM
→ final hidden representation
→ linear head
→ scalar return prediction
```

Weekly-only:

```text
Weekly sequence
→ 1-layer unidirectional LSTM
→ final hidden representation
→ linear head
→ scalar return prediction
```

Daily+Weekly:

```text
Daily
→ Daily LSTM ─┐
               ├→ concat → MLP head → scalar prediction
Weekly         │
→ Weekly LSTM ─┘
```

Settings:

- hidden_size = 64
- num_layers = 1
- unidirectional
- no attention
- no router
- no ticker embeddings
- no dropout branch introduced
- seed = 42
- batch_size = 32
- learning_rate = 1e-4
- Adam optimizer
- HuberLoss(delta=1.0)
- gradient clipping max_norm=1.0
- max_epochs = 100
- early stopping patience = 10
- best validation checkpoint restored before test evaluation
- MPS used on Apple Silicon when available
- same frozen data / split / normalization / metrics as Ridge

The full 3 frequency settings × 3 horizons = 9 experiments completed successfully.

Total runtime: approximately 29m 53s.

No numerical instability was observed:

- no NaN/Inf
- no output explosion
- no FSLR-style constant collapse
- early stopping behaved normally
- predictions remain under-dispersed overall

### Single-seed results

Best MSE setting by horizon:

- 5d: Daily-only
  - MSE = 0.010046
- 10d: Daily+Weekly
  - MSE = 0.019699
- 20d: Daily+Weekly
  - MSE = 0.040723

Best mean cross-sectional Rank IC by horizon:

- 5d: Daily+Weekly
  - Rank IC = 0.157300
- 10d: Weekly-only
  - Rank IC = 0.070441
- 20d: Daily+Weekly
  - Rank IC = 0.082596

Aggregate comparison vs the frozen SVD Ridge benchmark:

- MSE wins vs Ridge: 6 / 9
- MSE wins vs naive baseline: 0 / 9
- Rank IC wins vs Ridge: 6 / 9

Frequency-complementarity pattern:

- 5d: Daily+Weekly Rank IC > Daily-only = True; Daily+Weekly Rank IC > Weekly-only = True
- 10d: Daily+Weekly Rank IC > Daily-only = True; Daily+Weekly Rank IC > Weekly-only = False
- 20d: Daily+Weekly Rank IC > Daily-only = True; Daily+Weekly Rank IC > Weekly-only = True

Mean LSTM PredStd / TrueStd:

- 0.3249

### Interpretation

- LSTM improves over Ridge in several point-forecast and ranking configurations.
- However, LSTM does not beat the naive point-forecast benchmark on pooled MSE in any of the 9 configurations.
- Therefore it is not appropriate to claim that LSTM establishes strong absolute return-level predictability.
- The strongest evidence is in relative / cross-sectional ranking information rather than absolute return magnitude prediction.
- Different horizons favor different frequency representations:
  - 5d → Daily+Weekly best Rank IC
  - 10d → Weekly-only best Rank IC
  - 20d → Daily+Weekly best Rank IC
- This supports the project's multi-scale motivation without claiming final proof of universal superiority for one frequency setting.
- These results are a single-seed descriptive benchmark only.
- Formal multi-seed robustness with 5 seeds remains pending.

### Remaining robustness work

- LSTM 5-seed robustness has not yet been run
- LSTM architecture and protocol are frozen for this development benchmark
- The formal statement is:
  - "LSTM baseline: COMPLETE (single-seed development benchmark)."
  - "Architecture and training protocol frozen."
  - "Formal multi-seed robustness pending."

Relevant outputs:

- `dataset/audit/panel_lstm_test_predictions.csv`
- `dataset/audit/panel_lstm_summary_metrics.csv`
- `dataset/audit/panel_lstm_training_history.csv`
- `dataset/audit/panel_lstm_rank_ic_by_date.csv`
- `dataset/audit/panel_lstm_vs_ridge_summary.csv`
- `dataset/audit/panel_lstm_final_summary.txt`

### Emerging Panel Evidence

1. Linear and nonlinear models both struggle to beat naive zero/mean predictions in pooled MSE, highlighting the difficulty of absolute stock-return level forecasting.
2. Cross-sectional ranking contains more promising signal than absolute return magnitude prediction.
3. LSTM improves over Ridge in 6 / 9 Rank-IC configurations while remaining numerically stable.
4. Optimal temporal representation is horizon-dependent:
   - 5d: Daily+Weekly
   - 10d: Weekly
   - 20d: Daily+Weekly
   for LSTM Rank IC.
5. This is consistent with, but does not yet prove, the adaptive multi-scale hypothesis: a fixed temporal scale or frequency may not be optimal across all forecasting horizons and market states.
6. The completed Vanilla Transformer control provides an additional check on whether the horizon-dependent temporal-frequency pattern persists under generic self-attention. The next analytical step is the SWiM-style improved Transformer under the same frozen panel protocol.

Do not claim profitable trading performance, causal effects, or universal superiority of Daily+Weekly or LSTM over naive forecasting.

---

## Vanilla Transformer Panel Development Baseline — DONE / FROZEN

The Vanilla Transformer is an additional methodological control in the panel methodological control ladder. It is not one of the advisor's explicitly required "improved Transformer" algorithms. Its role is to isolate the effect of generic self-attention relative to the recurrent LSTM baseline, and to separate that effect from the incremental value of structured multi-scale / adaptive mechanisms in the later PathFormer branch.

### Completed implementation

The panel-ready Vanilla Transformer baseline exists and has completed its formal nominal-seed-42 development benchmark:

- Script: `scripts/python/panel_baseline_vanilla_transformer.py`

Architecture:

Single-frequency Daily / Weekly:

```text
input [B,T,F]
-> Linear(F,64)
-> fixed sinusoidal positional encoding
-> 2-layer TransformerEncoder
-> final-timestep representation encoded[:, -1, :]
-> Linear(64,1)
```

Daily + Weekly:

```text
Daily TransformerBranch
Weekly TransformerBranch

-> independent frequency-specific representations
-> concatenate [B,128]
-> Linear(128,64)
-> ReLU
-> Linear(64,1)
```

Architecture constants:

- d_model = 64
- nhead = 4
- num_encoder_layers = 2
- dim_feedforward = 128
- dropout = 0.1
- activation = ReLU

No causal mask.
No router.
No multi-scale patching.
No ticker embedding.
No SWiM mechanism.
No PathFormer mechanism.
No raw-sequence early fusion.

Parameter counts:

- daily_only = 67,393
- weekly_only = 67,393
- daily_weekly = 142,977

Note: an earlier implementation contained unused scalar heads inside the dual-frequency branches. This was identified during code review and refactored before formal training. The final frozen architecture above does not contain those unused heads.

### Frozen training protocol

Vanilla is intentionally matched to the frozen LSTM panel development protocol.

- seed = 42
- batch size = 32
- learning rate = 1e-4
- optimizer = Adam
- loss = HuberLoss(delta=1.0)
- max epochs = 100
- early stopping patience = 10
- gradient clipping = 1.0
- early stopping criterion: validation Huber loss
- best checkpoint restored before test evaluation

Frozen panel:

- 17 stocks
- train = 31,008
- validation = 6,647
- test = 6,664

Same sample index, anchor dates, labels, split, train-only per-ticker/per-frequency/per-feature normalization, and evaluation metrics as the frozen Ridge/LSTM panel protocol.

### Pre-training validation note

Before formal training, the implementation passed the defined validation gate:

- py_compile passed
- import passed
- frozen panel invariant checks passed
- Daily forward/backward smoke passed
- Weekly forward/backward smoke passed
- Daily+Weekly forward/backward smoke passed
- parameter-count audit passed
- explicit safe CLI implemented
- default script run performs smoke validation only
- single experiment requires explicit `--train --frequency ... --horizon ...`
- full benchmark requires explicit `--train --all`

The Rank IC implementation was independently validated against a pure pandas rank-then-Pearson implementation on the Daily-5d test sample:

- n_dates_total = 392
- n_dates_valid = 392
- max_abs_difference = 2.7755575615628914e-17
- median_abs_difference = 0.0
- mean_abs_difference = 7.080503983578804e-20
- all_close_1e12 = True

This specific Rank IC implementation was numerically validated on the relevant Daily-5d path. The SciPy / NumPy version warning did not materially alter the values in this validation path; this does not imply general environment compatibility beyond this check.

### Formal Vanilla development benchmark results

The full nominal-seed-42 MPS benchmark completed successfully.

- Total runtime: 1h 9m 44s
- Completed: 9 / 9 experiments

Daily only:

- 5d: test MSE = 0.010093; mean Rank IC = -0.004543; PredStd/TrueStd = 0.0835; best epoch = 10; best validation loss = 0.004145; runtime = 6m 46s; no_extreme_scale_pathology = False
- 10d: test MSE = 0.019791; mean Rank IC = 0.024641; PredStd/TrueStd = 0.1670; best epoch = 4; best validation loss = 0.007967; runtime = 4m 43s; no_extreme_scale_pathology = False
- 20d: test MSE = 0.038512; mean Rank IC = 0.038159; PredStd/TrueStd = 0.3349; best epoch = 4; best validation loss = 0.014788; runtime = 4m 44s; no_extreme_scale_pathology = True

Weekly only:

- 5d: test MSE = 0.010473; mean Rank IC = 0.095904; PredStd/TrueStd = 0.2315; best epoch = 10; best validation loss = 0.003870; runtime = 6m 28s; no_extreme_scale_pathology = True
- 10d: test MSE = 0.021070; mean Rank IC = 0.010942; PredStd/TrueStd = 0.1670; best epoch = 1; best validation loss = 0.007755; runtime = 3m 34s; no_extreme_scale_pathology = False
- 20d: test MSE = 0.040470; mean Rank IC = -0.041517; PredStd/TrueStd = 0.2259; best epoch = 1; best validation loss = 0.014030; runtime = 3m 43s; no_extreme_scale_pathology = True

Daily + Weekly:

- 5d: test MSE = 0.010066; mean Rank IC = 0.086375; PredStd/TrueStd = 0.2305; best epoch = 11; best validation loss = 0.003925; runtime = 13m 57s; no_extreme_scale_pathology = True
- 10d: test MSE = 0.019343; mean Rank IC = 0.071210; PredStd/TrueStd = 0.2583; best epoch = 11; best validation loss = 0.008065; runtime = 14m 16s; no_extreme_scale_pathology = True
- 20d: test MSE = 0.041350; mean Rank IC = -0.029437; PredStd/TrueStd = 0.2907; best epoch = 7; best validation loss = 0.015108; runtime = 11m 33s; no_extreme_scale_pathology = True

### Scientific interpretation

The key result is that no single temporal-frequency configuration is universally optimal.

For point accuracy:

- 5d: Daily+Weekly MSE = 0.010066; Daily = 0.010093; Weekly = 0.010473
- 10d: Daily+Weekly MSE = 0.019343; Daily = 0.019791; Weekly = 0.021070
- 20d: Daily = 0.038512; Weekly = 0.040470; Daily+Weekly = 0.041350

For cross-sectional mean Rank IC:

- 5d: Weekly = 0.095904; Daily+Weekly = 0.086375; Daily = -0.004543
- 10d: Daily+Weekly = 0.071210; Daily = 0.024641; Weekly = 0.010942
- 20d: Daily = 0.038159; Daily+Weekly = -0.029437; Weekly = -0.041517

Interpretation:

- 5d: Weekly provides the strongest cross-sectional ranking signal, while Daily+Weekly has slightly better point MSE than Daily and Weekly.
- 10d: Daily+Weekly is best on both MSE and mean Rank IC; this is the strongest Vanilla evidence of complementary Daily/Weekly information.
- 20d: Daily-only is best on both MSE and mean Rank IC; adding Weekly information hurts ranking and point accuracy.

Core scientific takeaway: temporal-frequency usefulness is strongly horizon dependent. Fixed multi-frequency fusion is not uniformly superior. Adding more temporal-frequency information can help, provide little incremental benefit, or hurt through information dilution / negative transfer. This supports the motivation for adaptive temporal-scale selection but does not yet prove that the adaptive router itself works.

The Vanilla Transformer remains numerically stable on the frozen panel, but its optimal temporal-frequency configuration is strongly horizon dependent: Weekly features provide the strongest 5-day cross-sectional ranking signal, Daily+Weekly fusion performs best at 10 days, while Daily-only features dominate at 20 days. The fact that fixed multi-frequency fusion is not uniformly superior motivates adaptive temporal-scale selection rather than indiscriminate frequency aggregation.

### Dispersion / stability interpretation

Vanilla did not exhibit the catastrophic instability previously seen in the original FSLR single-backbone PathFormer.

No:

- output-scale explosion
- NaN
- MSE in extreme hundreds/thousands
- literal constant collapse

However, some configurations remain materially under-dispersed. The most notable case is Daily 5d, with PredStd/TrueStd = 0.0835. Daily 10d and Weekly 10d both have 0.1670. Longer-horizon configurations generally show healthier dispersion in this run, but this should not be over-generalized from only 9 configurations.

Metric interpretation should remain separate:

- MSE / MAE: point forecast accuracy
- Pearson Corr: linear association
- mean Rank IC: cross-sectional ranking
- Direction Accuracy: sign prediction
- PredStd/TrueStd: prediction-dispersion calibration diagnostic

Positive Rank IC does not by itself prove strong return predictability.

### Reproducibility caveat

Before the full 9-group benchmark, a single-config nominal seed-42 Daily-5d smoke-training run produced approximately:

- MSE = 0.010023
- mean Rank IC = 0.027975
- best epoch = 11
- PredStd/TrueStd = 0.097139

The formal full-run nominal seed-42 Daily-5d result was:

- MSE = 0.010093
- mean Rank IC = -0.004543
- best epoch = 10
- PredStd/TrueStd = 0.0835

Therefore, nominal seed 42 on Apple MPS does not guarantee bitwise run-to-run deterministic reproduction. This is not treated as a model failure. It is an MPS / stochastic reproducibility caveat to be handled in the later multi-seed robustness stage. The full 9-group benchmark results above are the official Vanilla development baseline, and the earlier single-config run is treated only as smoke/development evidence. This strengthens the need for at least 5 seeds and mean ± std before formal robustness claims are made.

---

## Current Assessment

The current project status is now materially different from the earlier pre-freeze narrative.

- FSLR full-frequency naive-fusion PathFormer remains a negative diagnostic result and should remain documented as such.
- The active mainline is the frozen 17-stock Daily + Weekly panel on the shared sample index.
- Shared panel infrastructure is stable: loader, split logic, train-only normalization, and cross-sectional Rank IC are all in place and used by Ridge, LSTM, and Vanilla Transformer.
- Ridge ML baseline is complete and frozen with SVD.
- LSTM DL baseline is complete as a single-seed development benchmark and its architecture/training protocol are frozen.
- Vanilla Transformer is now complete as the generic self-attention control baseline under the same frozen panel protocol; its 9-configuration development benchmark is complete and frozen.
- Vanilla is numerically stable for the frozen panel, but does not uniformly dominate LSTM or naive baselines across all frequencies and horizons.
- Vanilla shows clear horizon-dependent frequency preference: Weekly is strongest at 5d ranking, Daily+Weekly is strongest at 10d, and Daily-only is strongest at 20d.
- Fixed Daily+Weekly fusion is useful at some horizons but not uniformly superior, which strengthens the motivation for structured multi-scale and adaptive scale-selection mechanisms.
- Our panel methodological control ladder is: Ridge → nonlinear recurrence (LSTM) → generic self-attention control (Vanilla Transformer) → structured/windowed-attention Transformer (SWiM-style) → adaptive multi-scale Transformer (PathFormer).
- This control ladder complements the advisor's literal experimental spine; it is not the advisor's literal Experiment 2.
- The advisor's literal experimental spine is: Experiment 1 = FSLR diagnostic model comparison, Experiment 2 = panel frequency-configuration comparison, Experiment 3 = Daily+Weekly PathFormer mechanism ablation, Experiment 4 = robustness + interpretability.
- The old statement that "panel-level ranking quality will be reported separately once Experiment 2 is run" is no longer correct; panel Rank IC is already being computed and reported for the Ridge, LSTM, and Vanilla baselines.

For the historical FSLR late-fusion experiments, `concat` remains the safer default branch for interpretation and `gated` remains a secondary ablation. This does not pre-select the fusion mechanism for the formal panel PathFormer benchmark, which remains pending but is now downstream of the completed Vanilla control.

Important caveat: overlapping future-return targets induce serial dependence within the panel sample. Descriptive ICIR and mean Rank IC should not be interpreted as formal statistical significance without the later multi-seed robustness and appropriate serial-dependence-aware inference step.

---

## Advisor Pivot (2026-07-08): Adaptive Multi-Scale Selection as Core Contribution

The advisor has redefined the project's core contribution and expanded the experimental scope. This section supersedes the priority of "Immediate Next Actions (Reporting-Oriented)" below; those items are paused, not deleted.

### Updated Guidance from Advisor Feedback (2026-08-16)

The advisor's latest feedback clarifies that the FSLR full-frequency experiments were a diagnostic stress test, not a valid blueprint for the panel main experiment. The key correction is explicit:

- A1–A5 correctly demonstrate that naive full-frequency fusion is unstable in the FSLR single-stock regime.
- This instability is not a reason to replicate the same structure on the panel; it is a reason to redesign the panel baseline.
- The panel should not be treated as a simple sample-size expansion of the FSLR full-frequency experiment.
- The correct panel framework is a more stable architecture: frequency-specific encoders + late fusion + optional adaptive router, with the router applied within frequency branches before cross-frequency fusion.

This shifts the project from "full-frequency naive fusion everywhere" to a cleaner decomposition of responsibilities:

- Frequency-specific encoder: model each frequency's own local/short-term or long-term structure separately.
- Late fusion: combine informative frequency-specific representations without forcing a single fragile all-frequency backbone.
- Optional adaptive router: used only after a stable per-frequency representation exists, and preferably within each frequency branch before cross-frequency mixing.

### Status Decision (Revised and Frozen Baseline)

- **FSLR full-frequency naive fusion is a negative result and should remain a diagnostic narrative, not a panel baseline.**
- **The new stable panel baseline is frequency-specific encoder + late fusion (+ optional router).**
- **Adaptive routing should be introduced only after the frequency-specific encoders are stable.** The router is not to be used as the default way of mixing all four frequencies from the start.
- **The core contribution remains adaptive multi-scale selection, but now as a per-frequency scale-selection problem first, and a cross-frequency fusion problem second.**

### Stable Panel Architecture (Advised Structure)

The advisor's recommended architecture is best summarized as:

- Hourly encoder → short-term, high-frequency dynamics
- Half-Day encoder → intermediate short-term structure
- Daily encoder → medium-term trend and volatility structure
- Weekly encoder → low-frequency regime information
- Fusion module → `concat` or `gated` late fusion of frequency-specific features
- Optional router → per-frequency internal scale router, not a naive all-frequency mixed router at the start
- Output head → unified prediction for 5d / 10d / 20d future return or direction

This design directly addresses the issue discovered in A5: all-frequency early fusion is unstable, while frequency-specific modeling plus late fusion is a more defensible and testable baseline.

### Correct Interpretation of the FSLR Stress Test

The FSLR experimental results should now be framed as follows:

1. **Patch size matters.** A1 shows that different frequencies and horizons prefer different patch sizes, which supports the multi-scale motivation.
2. **Naive full-frequency fusion is not a stable baseline.** A5 demonstrates that all-frequency combination is prone to output-scale explosion and instability.
3. **Static weighting and adaptive routing are not yet stable enough in the FSLR full-frequency setting.** A3-A4 show limited or inconsistent gains, with instability in several cases.
4. **Therefore, the panel must be redesigned around a more robust architecture.** The goal is to preserve the adaptive multi-scale idea while removing the unstable full-frequency collapse mechanism.

### Revised Experimental Design (Advisor-Aligned)

#### Experiment 1 — FSLR Case Study / Diagnostic Model Comparison

This remains a diagnostic / case-study track, but it is not just a historical narrative. The advisor's literal follow-up after listing the four experiments was: "For 1, choose one machine learning model, one deep learning model, and two improved Transformer algorithms." Accordingly, Experiment 1 should be recorded as a model comparison on the FSLR track, while still retaining the A1–A5 failure-analysis narrative as the diagnostic evidence behind that model selection.

Conceptually, the FSLR model set should include:

- one machine-learning model: current Linear baseline
- one deep-learning model: FSLR plain LSTM baseline, implementation status to be audited / likely missing
- two improved Transformer algorithms: SWiM-style and PathFormer
- optional additional conventional Transformer reference: Vanilla Transformer may remain as a supplementary control, but it is not one of the two advisor-requested improved Transformers

The A1–A5 results remain essential and should be retained as the scientific explanation for why full-frequency naive fusion is unstable. The correct interpretation is that Experiment 1 combines the required FSLR model comparison with the diagnostic failure analysis, rather than treating those as mutually exclusive categories.

#### Experiment 2 — Panel Frequency-Configuration Comparison

This is the advisor's literal Experiment 2 definition: not a model-family comparison, but a frequency-configuration comparison under the stable frequency-specific encoder + late-fusion framework.

Scientific question:

How does predictive performance change across different temporal-frequency configurations under the stable panel architecture?

Conceptual advisor frequency set:

- Hourly only
- Half-Day only
- Daily only
- Weekly only
- Hourly + Daily
- Daily + Weekly
- All frequencies

Current data-supported panel subset:

- Daily only
- Weekly only
- Daily + Weekly

Blocked / deferred because panel-wide intraday data are unavailable:

- Hourly only
- Half-Day only
- Hourly + Daily
- All frequencies

**Data-availability constraint (verified against the repo, 2026-08-16):** the panel dataset built by `panel_build_multiscale_dataset.py` only contains **Daily + Weekly** windows for all 24 tickers (`dataset/multiscale_dataset/panel/<TICKER>/` has `X_daily.npy` / `X_weekly.npy` / `y_5d,10d,20d.npy` only). Hourly and Half-Day OHLCV at panel scale do not exist yet — true Bloomberg-quality Hourly/Half-Day history is currently FSLR-only (see the Universe Expansion note above: `yfinance` 1h bars are capped at ~730 days, too short for a proper train/val/test split across 24 tickers). This means the originally-listed combo set (Hourly only / Half-Day only / Hourly + Daily) **cannot be run on the panel today** without first solving the Hourly/Half-Day data-sourcing problem.

The current panel implementation is therefore a data-constrained implementation of the advisor's frequency-specific design, not a conceptual rejection of the intraday branches. The review principle remains: the panel architecture should respect the advisor's stable pattern of frequency-specific encoders + late fusion + optional per-frequency adaptive routing; the missing intraday data simply prevents the full conceptual frequency set from being executed at this stage.

#### Experiment 3 — Daily + Weekly PathFormer Mechanism Ablation

This is the advisor-specified mechanism experiment for the panel. The working configuration is fixed as **Daily + Weekly**. The question is not "which frequency combination should we pick later?" but rather, once the stable Daily+Weekly setting is selected, which PathFormer mechanism contributes most to performance?

Advisor-specified mechanism comparison:

1. Single-scale
2. Fixed multi-scale
3. Static learned scale weight
4. Adaptive router

This should remain a mechanism-level ablation on the fixed Daily+Weekly configuration. Any future change to this configuration should be explicitly revisited with the advisor rather than silently reopened in the markdown.

#### Experiment 4 — Robustness and Interpretability

- Multi-seed robustness: at least 5 seeds, report mean ± std.
- Report metrics: MAE, Corr, cross-sectional Rank IC (computed per test date across tickers, then averaged over dates), Pred Std / True Std, and direction accuracy where relevant.
- Router-weight interpretation: analyze how the router activates under high-volatility vs low-volatility and trending vs range-bound regimes.
- Interpretability should be based on a stable model configuration, not on a model whose full-frequency baseline already failed. **FSLR's adaptive router is not stable (see A4), so it is not a valid router-interpretability subject — this analysis must run on the selected stable panel adaptive configuration, with Daily+Weekly currently the working configuration.** If the panel adaptive router also proves unstable, report interpretability findings as a failure diagnostic, not as a positive contribution.

### Panel Normalization Protocol (New — Required for Experiment 2/3)

The repository retains a broader 24-stock dataset, but the frozen primary experiment uses the balanced 17-stock panel. Because these 17 stocks still have materially different volatility and scale profiles, normalization remains essential.

Required protocol:

- Normalize features per ticker and per frequency using **train-set-only** statistics (no leakage from val/test).
- Primary target/evaluation space: raw future log returns.
- Volatility-scaled targets remain an optional robustness extension and have NOT yet been part of the frozen Ridge/LSTM benchmark.
- Report metrics both **pooled** (all tickers together) and **ticker-averaged** (metric computed per ticker, then averaged) — these can diverge a lot and both should be shown.
- Compute Rank IC **cross-sectionally by date** (rank predictions vs realized returns across tickers on the same date), then average across test dates — do not compute a single pooled time-series rank correlation and call it "IC".

### Stable Architecture Principle for Panel Deep Models

The advisor-aligned panel architecture is a frequency-specific encoder + late fusion design, not a single shared early-fusion backbone. The correct abstraction for a single frequency is:

Frequency sequence
→ frequency-specific encoder
→ representation
→ prediction head

For the daily/weekly panel, the intended structure is:

Daily
→ Daily encoder ──────────┐
                          ├→ late fusion → output head
Weekly                    │
→ Weekly encoder ─────────┘

This principle applies consistently across model families:

- LSTM
- Vanilla Transformer
- SWiM-style Transformer
- PathFormer

Do not concatenate raw Daily and Weekly sequences and feed them into one shared early-fusion backbone. The encoder changes across model families; the frequency-specific + late-fusion experimental skeleton stays stable.

The PathFormer contribution is therefore best described as:

- primary adaptive mechanism: per-frequency adaptive scale selection within each frequency branch,
- followed by: stable cross-frequency late fusion.

This is distinct from a global adaptive router that chooses directly between Daily and Weekly at the full panel level. The intended PathFormer architecture is conceptually:

Daily sequence
→ Daily multi-scale paths
→ Daily scale router
→ Daily representation
                         ┐
                         ├→ late fusion → output
Weekly sequence         │
→ Weekly multi-scale paths
→ Weekly scale router
→ Weekly representation

This is directly aligned with the advisor's original statement that PathFormer addresses the multi-scale problem within each frequency, while late fusion addresses information complementarity between frequencies.

### New Active Roadmap (Advisor-Aligned)

1. **DONE — Preserve FSLR A1–A5 as a negative-result diagnostic case study** to explain why naive full-frequency fusion fails and why the panel baseline must be redesigned.
2. **DONE — Freeze the 17-stock Daily + Weekly panel infrastructure** and standardize the train/val/test split, normalization, and evaluation protocol used by the frozen panel baselines.
3. **IN PROGRESS — Complete the panel methodological control ladder** across the current data-supported configurations: Daily only, Weekly only, and Daily + Weekly.
   - Ridge: DONE / FROZEN
   - LSTM: DONE / FROZEN development benchmark
   - Vanilla Transformer control: DONE / FROZEN development control
   - SWiM-style improved Transformer A: NEXT
   - Adaptive Multi-Scale PathFormer improved Transformer B: PENDING AFTER SWiM
4. **PARTIALLY EXECUTABLE — Advisor Experiment 2 frequency comparison** on the panel architecture: Daily only, Weekly only, Daily + Weekly are current-data-supported; Hourly only, Half-Day only, Hourly + Daily, and All frequencies remain blocked on panel-wide intraday data.
5. **PENDING — Experiment 3: Daily + Weekly PathFormer mechanism ablation** comparing single-scale vs fixed multi-scale vs static learned scale weight vs adaptive router.
6. **PENDING — Experiment 4: 5-seed robustness and router interpretation** with mean ± std reporting and regime analysis.
7. **PENDING — Final paper writeup**.

This roadmap separates the advisor's literal four-experiment spine from the additional panel methodological control ladder. The model-control ladder remains scientifically useful but is not the literal definition of Experiment 2.

### Panel Pipeline Status

- `scripts/python/panel_universe.py` — finalized the green-energy panel universe and ticker filtering.
- `scripts/python/panel_download_daily_weekly.py` — daily/weekly OHLCV download pipeline.
- `scripts/python/panel_build_multiscale_dataset.py` — builds the per-ticker panel dataset at Daily and Weekly scales only.
- `scripts/python/panel_common.py` — active shared loader / split / train-only normalization / metrics implementation for the frozen panel.
- `scripts/python/panel_baseline_ridge.py` — completed and frozen Ridge baseline for the panel.
- `scripts/python/panel_ridge_validation.py` — completed Ridge validation and naive comparison workflow.
- `scripts/python/panel_ridge_solver_final_check.py` — completed final SVD solver confirmation.
- `scripts/python/panel_baseline_lstm.py` — completed single-seed LSTM development benchmark for the panel.
- `scripts/python/panel_adaptive_scale_experiment.py` — exploratory PathFormer-family prototype; not yet the formal panel main benchmark.

Current implemented formal/development baselines:

- Naive baseline: DONE
- Ridge: DONE / FROZEN
- LSTM: DONE / FROZEN development architecture/protocol
- Vanilla Transformer: DONE / FROZEN development control
- SWiM-style / windowed-attention Transformer: NEXT
- Late-Fusion PathFormer / adaptive multi-scale PathFormer: prototype exists, but formal benchmark version remains pending

The current priority is to continue from the completed Ridge + LSTM + Vanilla control ladder toward the next benchmark model, the SWiM-style improved Transformer, while keeping the earlier FSLR diagnostic evidence as historical context.

### FSLR Full-Frequency Case Study (A1–A7) as Historical Evidence

The A1–A7 sequence should be preserved as the historical evidence showing why the old full-frequency setting is not a viable mainline path. It remains valuable as a stress-test narrative and as a cautionary baseline, but it should not be re-used as the direct panel experiment template.

- **A1**: patch-size search; confirms scale heterogeneity.
- **A2**: fixed multiscale vs single scale; partial gains but inconsistent.
- **A3**: static learned scale weight; not a stable improvement.
- **A4**: adaptive router; unstable and not clearly superior in the FSLR setting.
- **A5**: all-frequency ablation; demonstrates instability and scale explosion.
- **A6 / A7**: remain secondary and should not dominate the main narrative when the base model class is already known to be unstable.

### Final Interpretation

The project has now moved from a failed all-frequency replication attempt toward a more defensible scientific strategy:

- Keep the adaptive multi-scale idea as the core contribution.
- Remove the unstable full-frequency naive fusion from the panel mainline.
- Build the panel around an architecture that respects frequency-specific structure first, then combines them with stable late fusion and optional router mechanisms.
- Use FSLR as a diagnostic case study that explains why the more robust panel design is necessary.

This is the version of the plan consistent with the advisor's latest feedback and with the empirical evidence from A1–A5.

---

## Detailed Execution Plan (Step-by-Step, Status-Tagged)

Legend:
- [Done]: completed and supported by current code / dataset artifacts.
- [In Progress]: active or partially implemented.
- [Pending]: not yet executed.

### Phase 0 — Research Gate: confirm the scientific narrative

- [Done] Freeze the negative-result narrative for FSLR full-frequency naive fusion.
  - The evidence is the A1–A5 stress-test sequence: patch-size heterogeneity exists, but full-frequency fusion is unstable and often produces scale explosion / output collapse.
  - This is now a diagnostic result, not a panel mainline model.
- [Done] Align the project narrative around adaptive multi-scale selection as the core contribution.
  - The contribution is not “throw all frequencies into one model,” but “use frequency-specific modeling and selective fusion across scales.”
- [Done] Record the revised advisor interpretation in this file.
  - The new baseline should be stable, modular, and publication-friendly.

### Phase 1 — Data and fixed panel universe

- [Done] Finalize the panel ticker universe.
  - Script: `scripts/python/panel_universe.py`
  - Output: green-energy universe with delisted names removed.
  - Current project direction: Daily + Weekly panel only, consistent with the advisor's stable panel recommendation.
- [Done] Build the panel raw OHLCV dataset.
  - Script: `scripts/python/panel_download_daily_weekly.py`
  - Data location: `dataset/finance/panel_raw/`
- [Done] Build the per-ticker multi-scale panel dataset.
  - Script: `scripts/python/panel_build_multiscale_dataset.py`
  - Data location: `dataset/multiscale_dataset/panel/<TICKER>/`
  - Content: Daily and Weekly windows with aligned targets for 5d / 10d / 20d future returns.
- [Done] Confirm the panel data infrastructure is ready for modeling.
  - The repository now contains the expected panel pipeline from raw download to per-ticker windowed samples.

### Phase 2 — Establish stable baseline models for the panel

- [Done] Shared baseline family structure chosen.
  - Machine learning: Ridge
  - Deep learning: LSTM
  - Conventional Transformer control: Vanilla Transformer
  - Improved Transformer A: SWiM-style / windowed-attention Transformer
  - Improved Transformer B / proposed model: Adaptive Multi-Scale PathFormer
- [Done] Ridge panel-ready ML baseline written and run.
- [Done] Ridge numerical validation completed; SVD frozen.
- [Done] LSTM panel-ready DL baseline written.
- [Done] LSTM single-seed 9-configuration benchmark completed.
- [Done] LSTM architecture and training protocol frozen for the development benchmark.
- [Done] Vanilla Transformer panel development baseline implemented and run under the same frozen panel_common loader, split, normalization, targets, and metrics.
- [Done] Vanilla Transformer 9-configuration nominal-seed-42 benchmark completed and frozen as the generic self-attention control.
- [Pending] Port/build the SWiM-style improved Transformer panel baseline.
- [Pending] Build the formal Adaptive Multi-Scale PathFormer panel baseline.
- [Pending] Complete the five-model / family single-seed comparison: Ridge → LSTM → Vanilla Transformer → SWiM-style Transformer → Adaptive Multi-Scale PathFormer.
- [Pending] Multi-seed robustness after architecture comparison and selection.

The primary next action in Phase 2 is now to implement the SWiM-style improved Transformer on the same frozen panel backbone, while keeping the Vanilla Transformer as the completed control baseline.

### Phase 2a — Foundation-First Build Plan

**Principle:** the frozen panel sample index, split logic, and train-only normalization are now established and used by the actual baselines. This is the foundation layer that was verified before the later deep-learning steps.

Status update:

- [Done] Panel audit and coverage analysis completed.
- [Done] Universe-duration trade-off analysis completed.
- [Done] 17-stock universe freeze completed.
- [Done] Sample-index verification completed and passed.
- [Done] `scripts/python/panel_common.py` is in place and actively used.
- [Done] Ridge script completed and run.
- [Done] Ridge validation completed and run.
- [Done] Ridge solver final check completed and run.
- [Done] Formal solver is SVD.
- [Done] Ridge freeze completed.
- [Done] LSTM script completed and run.
- [Done] LSTM single-seed benchmark completed.
- [Done] Vanilla Transformer implementation audit and validation completed.
- [Done] Vanilla Transformer 9-configuration development benchmark completed.

Final status line for this phase:

- [Done] Ridge verification gate passed; deep-learning expansion authorized.
- [Done] LSTM single-seed development benchmark passed its stability gate.
- [Done] Vanilla Transformer development benchmark passed its control-benchmark gate.

### Phase 3A — Panel Methodological / Architecture Controls — IN PROGRESS

This is the additional model-control ladder used to attribute whether the observed temporal-frequency patterns are architecture-specific or persist across model families. It is not the advisor's literal Experiment 2.

Completed model families:

- [Done] Ridge × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
- [Done — single seed] LSTM × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
- [Done — single seed / full 9-config benchmark] Vanilla Transformer control × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d

Pending model families in order:

1. [Next] SWiM-style improved Transformer A × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
2. [Pending] Adaptive Multi-Scale PathFormer improved Transformer B × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d

Blocked / deferred:

- [Blocked on data] Hourly only, Half-Day only, Hourly + Daily, All frequencies
  - Requires a panel-wide intraday OHLCV source; the existing panel dataset is only Daily + Weekly.

Current evidence from the completed control benchmarks:

- 5d best Rank IC: Weekly-only for Vanilla; Daily+Weekly for LSTM
- 10d best Rank IC: Daily+Weekly for Vanilla and LSTM
- 20d best Rank IC: Daily-only for Vanilla; Daily+Weekly for LSTM

These results provide strong evidence that the preferred temporal representation is horizon-dependent rather than universally dominated by one frequency setting. This pattern motivates, but does not yet validate, adaptive multi-scale or adaptive frequency selection.

The Vanilla Transformer remains a control and does not satisfy one of the advisor-requested two improved-Transformer algorithm slots. It is an additional conventional-attention control, and the next formal step remains the SWiM-style improved Transformer.

### Experiment 2 — Panel Frequency-Configuration Comparison

This is the advisor-defined frequency-configuration experiment, separate from the model-control ladder above. The scientific question is how predictive performance changes across temporal-frequency configurations under the stable frequency-specific encoder + late-fusion framework.

Advisor-defined configurations:

- Hourly only
- Half-Day only
- Daily only
- Weekly only
- Hourly + Daily
- Daily + Weekly
- All frequencies

Current data-supported panel subset:

- Daily only
- Weekly only
- Daily + Weekly

Blocked / deferred because panel-wide intraday data are unavailable:

- Hourly only
- Half-Day only
- Hourly + Daily
- All frequencies

### Phase 4 — Experiment 3: Daily + Weekly PathFormer Mechanism Ablation

This is the PathFormer mechanism experiment that follows the panel frequency comparison. **Daily + Weekly is the advisor-specified working configuration** for Experiment 3.

- [Pending] Build the Daily+Weekly mechanism ablation ladder.
  - Single-scale
  - Fixed multi-scale
  - Static learned scale weight
  - Adaptive router
  - Optional dual-attention checks: full dual attention / intra-only / inter-only
- [Pending] Run the above on the same deterministic panel split and consistent metrics.
- [Pending] Determine whether the adaptive router provides improvement only after the frequency-specific representation is stable.
- [Pending] Decide whether the router should be considered a core mechanism or only an optional extension.

The current implementation should not silently reopen the Experiment-3 frequency configuration based only on intermediate results. If the team later finds strong evidence to revisit the configuration, that should be done explicitly with the advisor rather than by silently altering the experiment definition in the markdown.

### Phase 5 — Robustness and reproducibility (Experiment 4)

- [Pending] Run at least 5 seeds for the selected main model family.
  - recommended seeds: 0, 1, 21, 42, 3407
- [Pending] Report mean ± std for all main metrics.
- [Pending] Verify the selected model is not only good on one seed.
- [Pending] Confirm that stability is preserved in correlation, rank IC, and scale metrics, not only MSE.

Important clarification:

- Current Ridge results are deterministic / frozen.
- Current LSTM results are seed=42 development results.
- LSTM 5-seed robustness has not yet been run.
- Do not spend compute on multi-seed robustness until the next transformer and PathFormer baselines are stabilized.

### Phase 6 — Router interpretability analysis

- [Pending] Save per-sample router weights for the stable adaptive model.
  - This should be done only after the model is stable enough to interpret.
- [Pending] Analyze regime dependence.
  - High-volatility vs low-volatility periods
  - Trending vs range-bound regimes
  - 5d / 10d / 20d horizons
- [Pending] Analyze patch-size activation patterns.
  - which scales are consistently preferred under different market states?
- [Pending] Link router behavior to financial intuition.
  - local short-term microstructure vs medium-term trend vs low-frequency regime information

### Phase 7 — FSLR case-study writeup (diagnostic narrative)

- [Done] Preserve FSLR as a stress-test diagnostic setting, not as the mainline model target.
- [Pending] Write the FSLR section as a methodological cautionary study.
  - A1 patch-size search confirms scale heterogeneity.
  - A2 fixed multiscale helps partially but inconsistently.
  - A3 static weighting does not provide a stable improvement.
  - A4 adaptive router is not clearly stable in FSLR full-frequency mode.
  - A5 all-frequency fusion is unstable and should not be used as the panel baseline.
- [Pending] Conclude that FSLR is an important negative-result and stress-test case, but not the final mainline benchmark.

### Phase 8 — Final paper structure (recommended)

- [Pending] Section 1: FSLR diagnostic analysis
  - Why full-frequency naive fusion fails.
  - Why the router must be introduced only in a robust architecture.
- [Pending] Section 2: Panel main comparison
  - stable frequency combinations
  - model benchmarking
  - identify the strongest panel baseline
- [Pending] Section 3: Ablation study
  - Core mechanism validation on the fixed Daily+Weekly Experiment-3 configuration
  - Daily+Weekly is the advisor-specified working configuration for the Experiment-3 PathFormer mechanism ablation. Any future change to this configuration should be explicitly revisited with the advisor rather than silently changed based on intermediate results.
  - single-scale vs fixed multiscale vs static weight vs adaptive router
- [Pending] Section 4: Robustness and interpretation
  - 5-seed mean ± std
  - router activation analysis
  - financial regime interpretation

### Phase 9 — Decision gate before formal adaptive-router ablation / interpretation

Current status:

- Ridge stable: PASS
- LSTM stable: PASS as a single-seed development benchmark
- Vanilla Transformer control: PASS as a frozen development benchmark
- SWiM-style improved Transformer A: pending
- Adaptive Multi-Scale PathFormer improved Transformer B: pending
- Formal adaptive-router ablation / interpretation: wait until the model-family comparison is complete

The gate is no longer: "Should we start deep models?"

It is now: "After Ridge, LSTM, the Vanilla control, the SWiM-style improved Transformer, and the PathFormer development run, is there sufficient evidence to justify the formal adaptive multi-scale / router mechanism study?"

---

## Historical FSLR Reporting Actions (Paused)

1. Audit the repository to determine whether a plain FSLR LSTM baseline already exists. Do not claim it exists unless verified. Status: "FSLR plain LSTM baseline: implementation status to be audited / likely missing."
2. Complete the advisor-requested FSLR core model-comparison set: Linear + LSTM + SWiM-style + PathFormer.
3. Retain Vanilla Transformer as an optional supplementary conventional Transformer reference, not one of the two advisor-requested improved Transformer slots.
4. Preserve the A1–A5 diagnostic narrative and the concat/gated historical findings separately from the core model-comparison table, and write report text with explicit advisor checklist mapping and final protocol statement.

---

## Immediate Execution Checklist (Most Urgent)

1. [Done] Freeze the 17-stock balanced Daily + Weekly panel and shared sample index.
2. [Done] Freeze the Ridge ML baseline with SVD.
3. [Done] Complete the LSTM single-seed development benchmark.
4. [Done] Implement and validate the Vanilla Transformer control using the exact same `panel_common` loader, split, normalization, targets, and metrics.
5. [Done] Run the Vanilla Transformer development benchmark on Daily / Weekly / Daily+Weekly × 5d / 10d / 20d, nominal seed=42.
6. [Next] Port/build the SWiM-style improved Transformer A for the panel.
7. [Pending] Run the SWiM-style panel comparison within the additional methodological control ladder.
8. [Pending] Build the formal Adaptive Multi-Scale PathFormer panel model.
9. [Pending] Run the Adaptive PathFormer main comparison within the additional methodological control ladder.
10. [Pending] Complete the additional panel methodological control ladder across Ridge, LSTM, Vanilla Transformer, SWiM-style Transformer, and Adaptive PathFormer.
11. [Pending] Execute the advisor-defined Experiment 2 frequency comparison on the current data-supported set: Daily only, Weekly only, Daily + Weekly.
12. [Pending] Run the Advisor Experiment 3 Daily+Weekly PathFormer mechanism ablation: single / fixed / static / adaptive.
13. [Pending] Run 5-seed robustness on the selected model family.
14. [Pending] Produce router / regime interpretation analysis.
15. [Pending] Finalize the panel-results writeup and the FSLR diagnostic section.

Current control ladder:

Naive → Ridge → LSTM → Vanilla Transformer [control] → SWiM-style Transformer [Improved Transformer A] → Adaptive Multi-Scale PathFormer [Improved Transformer B / proposed model]

with status:

- Naive: DONE
- Ridge: DONE / FROZEN
- LSTM: DONE / FROZEN development benchmark
- Vanilla Transformer: DONE / FROZEN development control
- SWiM-style panel Transformer: NEXT
- Adaptive Multi-Scale PathFormer: PENDING AFTER SWiM
- Mechanism ablation: PENDING
- Formal robustness: PENDING
- Interpretability: PENDING

---

## Next-Round Experimental Plan (Superseded)

The previous FSLR-only heterogeneous-operator plan (Group 1 homogeneous operator comparison, Group 2 heterogeneous hybrids, Group 3 scale ablation on Hybrid A, and the horizon-wise 5d/10d/20d analysis framework) is **superseded by the Advisor Pivot above** and is no longer part of the active mainline. It is not reproduced here to avoid confusing it with the current active plan; the FSLR diagnostic narrative now used going forward is the A1–A7 sequence documented above, not this earlier ablation ladder.

## Archived Tracks (Not Current Mainline)

The following were previous exploration branches and are now archived from the active narrative:

- Multi-stock cross-sectional branch (solar basket + ETF-augmented labels) — superseded by the current Daily+Weekly green-energy panel.
- Market-augmented linear diagnostics (NaN/Inf issue found and fixed historically).
- Vision/candlestick image pipeline concept notes.
- The FSLR-only heterogeneous-operator Group 1/2/3 ablation plan and its horizon-wise analysis framework (see "Next-Round Experimental Plan (Superseded)" above).

These branches are not deleted from repository artifacts, but are no longer the active progress storyline.

