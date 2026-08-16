# Green Energy Stock Prediction Project

## Scope (Current Active Track)

**Current active track (2026-08-16, supersedes all wording below that implies a 4-frequency panel):** the active research track is a **Daily + Weekly multi-stock panel** with frequency-specific encoders, late fusion, and an optional per-frequency adaptive scale router. This is a deliberate scope reduction from the earlier "20–50 tickers x Hourly/Half-Day/Daily/Weekly" framing, driven by a verified data constraint: panel-wide Hourly/Half-Day OHLCV does not exist yet (see Experiment 2's data-availability note below). **FSLR is retained only as a single-stock, full-frequency (Hourly/Half-Day/Daily/Weekly) diagnostic case study** that explains why naive full-frequency fusion is not a viable panel blueprint — it is not the router-interpretability success case (see correction below).

Project focus was single-stock, multi-scale, single-modality modeling on FSLR using PathFormer-style time-series baselines. This FSLR-only setup is preserved as the validated diagnostic reference pipeline, not as the main experimental subject going forward.

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

**Build and evaluate a stable Daily + Weekly panel framework** (frequency-specific encoders + late fusion + optional adaptive scale router), while preserving the FSLR full-frequency experiments purely as diagnostic evidence for why that design is necessary. The FSLR Task 8 protocol below is historical/completed context, not the active objective — those items are paused per the Advisor Pivot section.

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

## Current Assessment

Advisor-requested experimental checklist is completed and reproducible.

- Single-backbone PathFormer track is closed as a negative result due to structural scale instability.
- Late-fusion track is stable enough for reporting and comparison.
- Ranking signal remains weak overall (time-series Rank Corr is close to zero in many FSLR settings), but protocol compliance and robustness reporting are complete. Panel-level ranking quality will be reported separately using cross-sectional Rank IC once Experiment 2 is run.
- `concat` is the safer default branch for interpretation; `gated` is retained as ablation with occasional low-variance degeneration.

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

#### Experiment 1 — FSLR Case Study / Diagnostic Analysis

This remains a diagnostic, not a mainline benchmark, and does **not** need new ML/DL baselines — A1–A5 already provide sufficient evidence.

- Reuse the existing A1–A5 results (patch-size search, single vs fixed multiscale, fixed vs static weight, static vs adaptive router, frequency ablation).
- Keep the focus on understanding failure modes, not on proving a full-frequency router works.
- This section should document that FSLR is a stress-test / case-study setting for multi-scale modeling and full-frequency instability, not the main experimental proving ground, and **not** the router-interpretability success case (router interpretability belongs in Experiment 4, on the stable panel model only).

#### Experiment 2 — Panel Main Comparison

This is the primary experimental block. **The "1 machine learning + 1 deep learning + 2 Transformer / improved-Transformer baselines" requirement belongs here, not in Experiment 1** — FSLR already has enough baseline coverage from Task 8 and A1–A5; the panel is what still needs a fresh, comparable baseline set:

| Type | Model |
|---|---|
| Machine Learning | Ridge or XGBoost |
| Deep Learning | LSTM or MLP |
| Transformer baseline | Vanilla Transformer |
| Improved Transformer | Late-Fusion PathFormer / Adaptive Multi-Scale PathFormer |

**Data-availability constraint (verified against the repo, 2026-08-16):** the panel dataset built by `panel_build_multiscale_dataset.py` only contains **Daily + Weekly** windows for all 24 tickers (`dataset/multiscale_dataset/panel/<TICKER>/` has `X_daily.npy` / `X_weekly.npy` / `y_5d,10d,20d.npy` only). Hourly and Half-Day OHLCV at panel scale do not exist yet — true Bloomberg-quality Hourly/Half-Day history is currently FSLR-only (see the Universe Expansion note above: `yfinance` 1h bars are capped at ~730 days, too short for a proper train/val/test split across 24 tickers). This means the originally-listed combo set (Hourly only / Half-Day only / Hourly + Daily) **cannot be run on the panel today** without first solving the Hourly/Half-Day data-sourcing problem.

Revised, data-honest combo list for the panel main comparison:

- Daily only
- Weekly only
- Daily + Weekly (primary multi-scale candidate)
- Optional/deferred, blocked on data: Hourly only, Half-Day only, Hourly + Daily, All frequencies — only runnable if/when a panel-wide Hourly/Half-Day source is secured (e.g. a paid intraday vendor, or accepting the ~730-day `yfinance` 1h window as a shorter, separate sub-panel experiment).

The main decision is to keep the model family stable and compare meaningful frequency combinations instead of repeating the unstable full-frequency setting, and to be explicit in the writeup that the Hourly/Half-Day panel tier is a data-constrained future extension, not a silently dropped experiment.

#### Experiment 3 — Ablation Study (on the most stable frequency combination)

Daily + Weekly is the primary candidate and will be used for the core ablation **if confirmed as the most stable setting in Experiment 2** (Experiment 2 has not been run yet as of this writing — see Phase 3 status below). Do not treat Daily + Weekly as a pre-decided answer; the ablation configuration is selected only after Experiment 2's panel main comparison confirms it.

Compare:

1. Single-scale
2. Fixed multiscale
3. Static learned scale weight
4. Adaptive router

This directly tests the mechanism of interest without combining everything into an unstable all-frequency architecture.

#### Experiment 4 — Robustness and Interpretability

- Multi-seed robustness: at least 5 seeds, report mean ± std.
- Report metrics: MAE, Corr, cross-sectional Rank IC (computed per test date across tickers, then averaged over dates), Pred Std / True Std, and direction accuracy where relevant.
- Router-weight interpretation: analyze how the router activates under high-volatility vs low-volatility and trending vs range-bound regimes.
- Interpretability should be based on a stable model configuration, not on a model whose full-frequency baseline already failed. **FSLR's adaptive router is not stable (see A4), so it is not a valid router-interpretability subject — this analysis must run on the stable Daily + Weekly panel adaptive model.** If the panel adaptive router also proves unstable, report interpretability findings as a failure diagnostic, not as a positive contribution.

### Panel Normalization Protocol (New — Required for Experiment 2/3)

Pooling 24 tickers with very different volatility profiles (e.g. NEE vs PLUG/FCEL) means a naive pooled MSE will be dominated by the highest-volatility names unless normalization is explicit. Required protocol:

- Normalize features per ticker and per frequency using **train-set-only** statistics (no leakage from val/test).
- Evaluate targets both in raw log-return space and, if needed, volatility-scaled return space.
- Report metrics both **pooled** (all tickers together) and **ticker-averaged** (metric computed per ticker, then averaged) — these can diverge a lot and both should be shown.
- Compute Rank IC **cross-sectionally by date** (rank predictions vs realized returns across tickers on the same date), then average across test dates — do not compute a single pooled time-series rank correlation and call it "IC".

### Conceptual Full-Frequency Architecture (Advisor-Advised)

The advisor's preferred panel architecture, as a general design, is:

```
Hourly Encoder
Half-Day Encoder
Daily Encoder
Weekly Encoder
Fusion Module (concat / gated late fusion)
Optional Router (prefer inside each frequency branch)
Output Head
```

**Current implementation: Daily Encoder + Weekly Encoder only.** Under the current data constraint (see Experiment 2), the active panel implementation instantiates only the Daily and Weekly branches; Hourly and Half-Day branches remain deferred extensions, not yet built for the panel.

The important modeling principle is:

- Do not mix all frequencies into a single fragile feature space at the first stage.
- First learn each frequency's own representation.
- Then fuse them via late fusion.
- Only after a stable frequency-specific representation exists, consider adaptive routing inside a scale branch or within a frequency-specific encoder stack.

This is a more credible and publication-oriented experimental design than simply scaling the FSLR full-frequency architecture to the panel.

### New Active Roadmap (Advisor-Aligned)

1. **Keep the FSLR A1–A5 diagnostics as a negative-result case study** to explain why naive full-frequency fusion fails and why the panel baseline must be redesigned.
2. **Refactor the panel main experiment** around frequency-specific encoders + late fusion + optional router, rather than repeating the unstable all-frequency naive fusion architecture.
3. **Run the panel main comparison** on the frequency combinations actually supported by current panel data: Daily only, Weekly only, Daily + Weekly. Treat Hourly only / Half-Day only / Hourly + Daily / All frequencies as blocked-on-data extensions until a panel-wide intraday source is resolved.
4. **Run the Daily + Weekly ablation** comparing single-scale vs fixed multiscale vs static learned scale weight vs adaptive router.
5. **Run 5-seed robustness reporting** with mean ± std across the main settings.
6. **Add router-weight interpretation analysis** only for the stable model family.
7. **Keep A6/A7 as secondary or deferred analysis**, as they are not a strong primary narrative when the underlying full-frequency baseline is already known to be unstable.

### Panel Pipeline Status

- `scripts/python/panel_universe.py` — finalized green-energy universe and ticker filtering (24 tickers, verified via `panel_build_manifest.csv`, all status `ok`).
- `scripts/python/panel_download_daily_weekly.py` — daily/weekly OHLCV download pipeline.
- `scripts/python/panel_build_multiscale_dataset.py` — builds per-ticker panel dataset at Daily and Weekly scales only (verified: `dataset/multiscale_dataset/panel/<TICKER>/` contains only `X_daily.npy`, `X_weekly.npy`, `y_5d/10d/20d.npy`, `meta.csv` — no hourly/half-day files exist for any panel ticker).
- `scripts/python/panel_adaptive_scale_experiment.py` — currently an exploratory prototype implementing only the PathFormer-family variants (`daily_only`, `weekly_only`, `fixed_multi`, `learnable_weight`, `adaptive_router`); it should be treated as a provisional experiment, not the final panel main benchmark.
- **Gap confirmed by repo search:** there is no Ridge / XGBoost / LSTM / MLP / Vanilla-Transformer panel script yet anywhere in `scripts/python/`. All non-PathFormer baselines needed for Experiment 2's "1 ML + 1 DL + 2 Transformer" checklist still need to be written from scratch for the panel; they exist today only in the FSLR-only Task 8 scripts (`task8_baseline_linear_multihorizon.py`, `task8_baseline_vanilla_transformer.py`, `task8_baseline_swim.py`), which are not yet ported to the pooled-panel loader.

The current priority is to change the panel experiment from a full-frequency replication to a stable frequency-specific baseline plus a clean, advisor-aligned ablation ladder.

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

- [Done] Decide the baseline family structure (design only, no code yet).
  - Machine learning: Ridge / XGBoost
  - Deep learning: LSTM / MLP
  - Transformer: Vanilla Transformer
  - Improved Transformer: Late-Fusion PathFormer / adaptive multi-scale PathFormer
- [Pending] **Write the panel-ready ML baseline script (Ridge and/or XGBoost).** No panel version exists yet; only `task8_baseline_linear_multihorizon.py` exists and it is FSLR-only.
- [Pending] **Write the panel-ready DL baseline script (LSTM and/or MLP).** No such script exists anywhere in the repo yet (FSLR-only baselines are Linear / Vanilla Transformer / SWiM / PathFormer, no plain LSTM/MLP).
- [Pending] **Port the Vanilla Transformer baseline to the pooled panel loader.** Only the FSLR-only `task8_baseline_vanilla_transformer.py` exists today.
- [Pending] Decide which "improved Transformer" to report: Late-Fusion PathFormer (frozen FSLR variant, needs panel port) vs the newer adaptive multi-scale PathFormer in `panel_adaptive_scale_experiment.py`.
- [Pending] Implement or standardize the benchmark runner so that all four baseline models share identical targets, splits, and metrics.
  - Inputs: panel Daily only / Weekly only / Daily + Weekly (see data-availability constraint under Experiment 2).
  - Outputs: MSE, MAE, Corr, cross-sectional Rank IC, Direction Accuracy, Pred Std / True Std.
- [Pending] Decide the exact benchmark set for the main paper table.
  - Minimum recommended set: Ridge, LSTM, Vanilla Transformer, Late-Fusion PathFormer.
  - Optional stronger model: XGBoost or adaptive router variant if it survives stability checks.

### Phase 3 — Panel main comparison (Experiment 2)

This is the main experimental block.

- [Pending] Run the single-frequency panel comparison (data-supported today).
  - Daily only
  - Weekly only
- [Pending] Run the dual-frequency panel comparison (data-supported today).
  - Daily + Weekly
- [Blocked on data] Hourly only, Half-Day only, Hourly + Daily, All frequencies.
  - Requires a panel-wide Hourly/Half-Day OHLCV source; current `yfinance` 1h history (~730 days) is likely too short for a fair 70/15/15 split across 24 tickers. Needs an explicit decision: skip entirely, use a shorter recent-window sub-panel, or source a paid intraday vendor.

**Immediate action before running Experiment 2:** the Ridge / LSTM / Vanilla Transformer / Late-Fusion PathFormer scripts for the panel do not exist yet (see Panel Pipeline Status gap above) — Phase 2 must produce runnable scripts before Phase 3 can start.
- [Pending] Compare these models with consistent evaluation metrics.
  - MSE
  - MAE
  - Corr
  - Cross-sectional Rank IC
  - Direction Accuracy
  - Pred Std / True Std
  - Stability trend across the test window
- [Pending] Select the stable frequency combination for the main insight.
  - This selection is the output of Experiment 2, not a pre-decided input; Daily + Weekly is only the leading candidate pending confirmation.

### Phase 4 — Core PathFormer ablation on the most stable frequency pair

The ablation must be restricted to the strongest stable frequency pair, not run over all combinations.

- [In Progress] Build the Daily + Weekly main ablation ladder.
  - Single-scale
  - Fixed multiscale
  - Static learned scale weight
  - Adaptive router
  - Optional dual-attention checks: full dual attention / intra-only / inter-only
- [Pending] Run the above on the same deterministic panel split and consistent metrics.
- [Pending] Determine whether the adaptive router provides improvement only after the frequency-specific representation is stable.
- [Pending] Decide whether the router should be considered a “core mechanism” or only an “optional extension.”

### Phase 5 — Robustness and reproducibility (Experiment 4)

- [Pending] Run at least 5 seeds for the selected main model family.
  - recommended seeds: 0, 1, 21, 42, 3407
- [Pending] Report mean ± std for all main metrics.
- [Pending] Verify the selected model is not only good on one seed.
- [Pending] Confirm that stability is preserved in correlation, rank IC, and scale metrics, not only MSE.

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
  - Daily + Weekly core mechanism validation
  - single-scale vs fixed multiscale vs static weight vs adaptive router
- [Pending] Section 4: Robustness and interpretation
  - 5-seed mean ± std
  - router activation analysis
  - financial regime interpretation

### Phase 9 — Decision gate before deep model expansion

- [Pending] Decision rule: only proceed to more complex router variants if the stable panel baseline surpasses or at least matches the best conventional baseline on the selected frequency pair.
- [Pending] If the adaptive router underperforms or destabilizes, keep it as an optional extension and not the main contribution.
- [Pending] If Daily + Weekly + late-fusion PathFormer is stable and competitive, proceed to the formal paper framing around “frequency-specific modeling + adaptive selection as a robust multi-scale design.”

---

## Immediate Next Actions (Reporting-Oriented, Paused)

1. Build final comparison table against Linear / Vanilla Transformer / SWiM using the same horizons and metrics.
2. Write report text with explicit advisor checklist mapping and final protocol statement.
3. Mark `concat` as primary late-fusion variant and `gated` as ablation in the main narrative.

---

## Immediate Execution Checklist (Most Urgent)

1. [Pending] Write the panel-ready Ridge / LSTM / Vanilla Transformer / Late-Fusion PathFormer scripts (none exist yet for the pooled panel loader — see Panel Pipeline Status gap).
2. [Pending] Standardize a shared benchmark runner so all four baselines use identical splits/targets/metrics.
3. [Pending] Run panel comparison over Daily only, Weekly only, Daily + Weekly (the only combos current panel data supports). Explicitly note Hourly only / Half-Day only / Hourly + Daily / All frequencies as blocked-on-data until a panel-wide intraday source is resolved.
4. [Pending] Select the best stable daily/weekly panel baseline.
5. [Pending] Run Daily + Weekly ablation: single / fixed / static / adaptive (+ optional full-dual / intra-only / inter-only attention checks).
6. [Pending] Run 5-seed robustness for the winning model.
7. [Pending] Save router-weight CSVs and produce interpretability analysis.
8. [Pending] Draft the final FSLR diagnostic section and the panel main-results section.

This ordering follows the advisor's preferred logic: first stabilize the panel baseline, then test the mechanism, and only then write the final scientific narrative.

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

