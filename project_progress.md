# Green Energy Stock Prediction Project

## Scope (Current Active Track)

**FROZEN PANEL DECISION (2026-08-16):** **Primary panel = the 17-stock balanced Daily + Weekly panel, common period 2016-01-25 to 2026-06-05** (`AES, BEP, BLDP, BLNK, CSIQ, CWEN, DQ, ENPH, FCEL, FSLR, HASI, JKS, NEE, ORA, PLUG, RUN, SEDG` — see `panel_candidate_universe_summary.csv`, `n_tickers=17` row). This is the universe all Phase 2+ modeling (Ridge/LSTM/Transformer/PathFormer, Experiment 2/3/4) targets. **The full 24-stock universe is NOT deleted** — it remains in `panel_universe.GREEN_ENERGY_UNIVERSE` and the built dataset under `dataset/multiscale_dataset/panel/`, retained explicitly as a future robustness / unbalanced-panel extension, not as the primary experiment.

**Pre-freeze verification (ran 2026-08-16, PASSED):** `scripts/python/panel_verify_17stock_sample_index.py` confirmed that for the 17-stock panel's train split, Daily-only, Weekly-only, and Daily+Weekly settings all resolve to the exact same 31,008-sample `(ticker, anchor_date)` index (not just by construction — checked empirically via non-NaN/non-Inf masks on `X_daily`/`X_weekly`). This was the required sanity gate before freezing the panel choice.

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

## Ridge Panel Baseline — DONE / FROZEN

- Universe: frozen 17-stock balanced panel
- Daily / Weekly / Daily+Weekly
- Horizons: 5d / 10d / 20d
- Formal solver: **SVD**
- Alpha selected using validation only
- Cross-sectional Rank IC computed by date
- Numerical solver robustness passed
- No leakage
- Ridge does not beat naive baseline on MSE
- Weekly-only has best MSE
- Daily+Weekly has best Rank IC
- Predictions are under-dispersed

These last three items will become the reference benchmark for LSTM / Transformer.

---

## LSTM Fair-Comparison Protocol (Mandatory)

The research question is explicit:

> Does nonlinear sequential modeling add value over Ridge?

To answer this correctly, the LSTM experiment must be designed to be fully comparable to the ridge baseline.

### Fairness requirements

- Universe: 17-stock balanced panel
- Train / Val / Test: same split as Ridge
  - Train: 31,008 samples
  - Val: 6,647 samples
  - Test: 6,664 samples
- Frequency: Daily-only / Weekly-only / Daily+Weekly
- Horizon: 5d / 10d / 20d
- Normalization: frozen, identical to Ridge
- Metric: same metric set as Ridge, including cross-sectional Rank IC by date
- Leakage: none
- Target construction: same labels, same timing, same sample index

In other words, the comparison is not "LSTM vs some custom pipeline"; it is 
"LSTM vs Ridge under the same panel, same split, same features, same targets, same normalization, same metrics".

### Architecture design (first LSTM version)

Keep the design deliberately simple.

#### Daily-only

```text
Daily sequence
→ LSTM
→ representation
→ linear head
→ return
```

#### Weekly-only

```text
Weekly sequence
→ LSTM
→ representation
→ linear head
→ return
```

#### Daily+Weekly

Do not concatenate the raw daily and weekly sequences into one long sequence.

```text
Daily
→ Daily LSTM ──┐
               ├─ concat → prediction head
Weekly         │
→ Weekly LSTM ─┘
```

This follows the same architecture philosophy as the later models:

- frequency-specific encoders
- late fusion
- shared comparison design across LSTM / Transformer / PathFormer

This is important because the goal is not to test a random nonlinear design, but to test whether nonlinear sequential modeling can learn temporal dependence that Ridge cannot.

### Three benchmark questions for LSTM

#### Benchmark A: Naive MSE

Ridge does not beat the naive point-forecast baseline on MSE. The question is:

> Can LSTM genuinely beat the naive baseline?

#### Benchmark B: Ridge Rank IC

For 5d, Ridge Daily+Weekly already has meaningful rank signal. The key question is:

> Can LSTM match or exceed Ridge Rank IC, especially at 5d?

#### Benchmark C: Prediction dispersion

Ridge predictions are under-dispersed:

- PredStd / TrueStd < 1

The key question is:

> Does LSTM produce a more realistic dispersion ratio close to 1 without the FSLR-style instability or explosion?

### Decision rule

LSTM adds value only if it improves over Ridge on the same benchmark questions under the same fairness constraints, not merely by showing a slightly lower MSE in a different protocol.

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

### Phase 2a — Foundation-First Build Plan (Do This Before Any Model)

**Principle:** build and verify one shared panel loader + chronological split + train-only normalization + metric evaluator first, and prove it end-to-end with Ridge only. If this layer is correct, LSTM / Vanilla Transformer / Late-Fusion PathFormer are just swapped-in models on top of the same verified pipeline. If this layer is wrong, every downstream deep-learning experiment is wasted effort, so no other model should be started before this passes verification.

**Step 0 (confirmed as the actual first step, before Ridge is even written):** build and audit the unified 24-stock Daily + Weekly panel benchmark dataset itself — common chronological split dates, train-only per-ticker/per-frequency normalization, and a shared sample index for all subsequent models — and output one audit CSV that proves the foundation is correct before any model touches it.

- `scripts/python/panel_common.py` implements the loader/split/normalizer (see below); a separate one-off script, e.g. `scripts/python/panel_audit_benchmark_dataset.py`, runs it once and writes `dataset/audit/panel_benchmark_dataset_audit.csv` plus a `_summary.txt`.
- Audit CSV must contain, per ticker x frequency (Daily, Weekly) x horizon (5d/10d/20d):
  - `n_train`, `n_val`, `n_test` sample counts.
  - `train_start_date`, `train_end_date`, `val_start_date`, `val_end_date`, `test_start_date`, `test_end_date` (must show the same panel-wide split-date thresholds for every ticker — the split is by shared date, not by per-ticker row fraction).
  - `train_feature_mean`, `train_feature_std` per feature (open/high/low/close/volume) — proves the normalizer is fit only on train rows.
  - A leakage check flag: `max_train_anchor_date <= split_train_cut` and `min_test_anchor_date > split_val_cut` for every ticker (must be `True` for all rows, otherwise the audit fails).
  - `shared_sample_index` — a stable integer id per (ticker, anchor_date) row so every downstream model (Ridge, LSTM, Transformer, PathFormer) references the exact same sample when logging predictions, letting per-sample results be joined/compared across models later.
- This audit CSV must be visually reviewed (row counts look reasonable, no ticker has zero test rows, split dates line up panel-wide) **before** `panel_baseline_ridge.py` is written, since a bug here invalidates every model built on top of it.

**New shared module: `scripts/python/panel_common.py`** (does not exist yet — to be created)

- `load_panel_frequency(frequency, horizon_key)` — loads `X_daily.npy`/`X_weekly.npy`/`y_<horizon>.npy`/`meta.csv` per ticker from `dataset/multiscale_dataset/panel/<TICKER>/` and concatenates across all 24 tickers in `panel_universe.GREEN_ENERGY_UNIVERSE`, returning arrays plus a parallel `ticker` and `anchor_date` array (same pattern as `panel_adaptive_scale_experiment.load_pooled_panel`, but factored out so every model script imports the same function instead of re-implementing it).
- `chronological_split(anchor_date, train_frac=0.70, val_frac=0.15)` — reuses the existing `global_time_split` date-threshold logic (one shared threshold across the whole panel, not per-ticker), returns boolean train/val/test masks. Must be the single source of truth so Ridge, LSTM, Transformer, and PathFormer are evaluated on byte-identical splits.
- `fit_train_only_normalizer(X_train)` / `apply_normalizer(X, stats)` — per-ticker, per-frequency feature normalization (mean/std) fit strictly on the train mask, then applied unchanged to val/test. Must be fit per ticker (not pooled across tickers) to respect the volatility-heterogeneity concern already documented under "Panel Normalization Protocol".
- `evaluate_predictions(y_true, y_pred, ticker, anchor_date)` — single shared metric function returning: MSE, MAE, Corr, Direction Accuracy, Pred Std / True Std (all pooled and ticker-averaged), plus cross-sectional Rank IC (Spearman rank correlation computed within each `anchor_date` group across tickers, then averaged across dates — not a single pooled time-series rank correlation).
- `save_result_row(...)` — one consistent CSV row/schema (frequency combo, horizon, model name, seed, all metrics above) so every model's output appends to the same comparison table format.

**Verification gate before touching any deep-learning model:**

1. Run `panel_common` unit-checks: split boundaries are identical across repeated calls, no ticker's train rows are dated after any test row anywhere in the panel, and per-ticker normalization stats are computed only from the train mask (assert no val/test dates leak into the mean/std computation).
2. Implement `scripts/python/panel_baseline_ridge.py` using only `panel_common` functions: Ridge regression (flattened window features) for Daily only / Weekly only / Daily + Weekly, all three horizons (5d/10d/20d).
3. Sanity-check Ridge output: MSE/MAE in a plausible range for daily log-returns (roughly similar order of magnitude to the FSLR Linear baseline numbers already on record), Pred Std not collapsed to ~0 and not exploding, Direction Accuracy near 0.5 (not a red flag either way for a linear baseline).
4. Only after Ridge passes this sanity check does Phase 2 proceed to LSTM, Vanilla Transformer, and the Late-Fusion PathFormer panel port — all reusing `panel_common` unchanged.

- [Done — script written, not yet run] **(Step 0)** Created `scripts/python/panel_audit_benchmark_dataset.py`. It loads all 24 tickers' existing `X_daily.npy`/`X_weekly.npy`/`y_5d,10d,20d.npy`/`meta.csv`, reports the common date-coverage window across all tickers (not assumed), derives ONE shared panel-wide chronological split (train/val/test cut dates from pooled anchor dates, applied identically to every ticker), builds a unified `(ticker, anchor_date, split, y_5d, y_10d, y_20d)` sample index, and locks train-only per-ticker/per-frequency normalization stats (mean/std). Outputs: `dataset/audit/panel_benchmark_data_audit.csv`, `panel_split_summary.csv`, `panel_norm_stats.csv`, `panel_sample_index.csv`, `panel_benchmark_data_audit_summary.txt`.
- [Done — ran, result flagged a problem] Ran Step 0's audit: 24/24 tickers loaded, no NaN/Inf, no leakage failures, but **common coverage across all 24 tickers is only ~2022-04-18 to ~2026-06-03 (~4 years)** — likely too short for the adaptive multi-scale / regime-interpretation research goal, which motivates the trade-off analysis below before finalizing the panel size.
- [Done — script written, not yet run] **(Step 0b, universe-vs-duration trade-off)** Created `scripts/python/panel_universe_duration_tradeoff.py` (reuses `panel_audit_benchmark_dataset.load_ticker`, no duplicated loading logic). It ranks all 24 tickers by first usable anchor_date, iteratively removes the latest-starting ticker to trace the full 24→1 common-coverage curve, computes a common-date-only (set-intersection, not union) 70/15/15 split for candidate universe sizes (24/22/20/18/17/16), flags the largest single-step coverage jump, and runs per-candidate sanity checks (no empty splits, no NaN/Inf, Daily+Weekly+5d/10d/20d all present on common dates). Does **not** choose a final universe — evidence only. Outputs: `dataset/audit/panel_ticker_date_coverage.csv`, `panel_universe_duration_tradeoff.csv`, `panel_candidate_universe_summary.csv`, `panel_universe_duration_tradeoff_summary.txt`.
- [Done — ran, universe frozen] Reviewed the trade-off and froze the panel: **17-stock panel, common period 2016-01-25 to 2026-06-05** (see "FROZEN PANEL DECISION" at the top of this file). Full 24-stock universe kept as a future robustness/unbalanced-panel extension, not deleted.
- [Done — ran, PASSED] Pre-freeze verification (`scripts/python/panel_verify_17stock_sample_index.py`): confirmed daily_only/weekly_only/daily_weekly all resolve to the identical 31,008-sample train `(ticker, anchor_date)` index for the frozen 17-stock panel.
- [Done — script written, not yet run] Created `scripts/python/panel_common.py` on the **frozen 17-stock panel**: `load_frozen_panel()` (reuses `panel_universe_duration_tradeoff`'s common-date computation, no duplicated logic), `fit_train_only_normalizer()` / `apply_normalizer()` (per-ticker, per-frequency, per-feature, train-split-only), `flatten_features()`, and `evaluate_predictions()` (MSE, MAE, Pearson Corr, Direction Accuracy, Pred Std / True Std / their ratio, cross-sectional Rank IC computed per anchor_date across tickers then averaged, all reported both pooled and ticker-averaged).
- [Done — script written, not yet run] Created `scripts/python/panel_baseline_ridge.py` on top of `panel_common`: runs Ridge x {daily_only, weekly_only, daily_weekly} x {5d, 10d, 20d} = 9 runs, with a small alpha grid `[0.1, 1, 10, 100]` selected via val-split pooled MSE, final metrics reported on test. Outputs `dataset/audit/panel_ridge_summary_metrics.csv` (9 rows, all pooled + ticker-averaged metrics) and `dataset/audit/panel_ridge_test_predictions.csv` (sample-level test predictions: ticker, anchor_date, horizon, y_true, y_pred, frequency_setting, model).
- [Done — ran] First Ridge run completed; raised a `LinAlgWarning: ill-conditioned matrix` on the Daily+Weekly setting, which triggered the Ridge validation pass below before proceeding to LSTM.
- [Done — script written, not yet run] Created `scripts/python/panel_ridge_validation.py` (Tasks A–F, does not touch dataset/universe/split/normalization, does not implement LSTM/Transformer/PathFormer):
  - **Task A**: naive `zero` and `train_mean` baselines (train-split-only mean) for 5d/10d/20d, evaluated through the same `panel_common.evaluate_predictions` pipeline. Undefined Corr/Rank IC (constant predictors) are left as explicit NaN, never replaced with 0 — `panel_common.py` was updated with a `safe_pearson_corr` guard to guarantee this. Output: `panel_naive_baseline_summary.csv`.
  - **Task B**: Ridge solver-stability check, `auto` (the original script's implicit default) vs `lsqr`, same samples/features/normalization/alpha grid/horizons/frequency settings; captures the `ill-conditioned` warning per fit, and compares predictions between solvers (Pearson corr, mean/max abs diff) to decide whether `lsqr` can be designated the formal solver (threshold: min cross-config prediction corr ≥ 0.999). Outputs: `panel_ridge_solver_stability.csv`, `panel_ridge_solver_predictions.csv`.
  - **Task C**: proper cross-sectional Rank IC — new `panel_common.rank_ic_by_date()` computes Spearman rank corr across the 17 tickers within each test anchor_date (not a pooled correlation), then mean/median/std/positive-IC-ratio/n_valid_dates/descriptive ICIR are summarized on top. Outputs: `panel_ridge_rank_ic_by_date.csv`, `panel_ridge_rank_ic_summary.csv`.
  - **Task D**: consolidated `panel_ridge_final_benchmark.csv` (Zero, Train Mean, Ridge Daily/Weekly/Daily+Weekly x 5d/10d/20d, using the Task-B-designated formal solver).
  - **Task E**: sanity checks (exact 6,664 test samples per frequency setting, identical daily/weekly/daily+weekly test sample sets, ~17 tickers per test date, no NaN/Inf in predictions, no train/val/test date-boundary leakage) — script hard-stops (`sys.exit(1)`) before writing the final summary if any `[MAJOR]` violation is found.
  - **Task F**: `panel_ridge_final_summary.txt` — lowest-MSE and highest-Rank-IC setting per horizon, Daily+Weekly vs single-frequency ranking comparison, Ridge-vs-naive MSE comparison, Pred Std/True Std dispersion read, solver-agreement verdict, and an explicit caveat that positive Rank IC alone does not establish predictive success (point-accuracy vs ranking-information vs calibration are kept separate).
- [Pending — user to run] Execute `python scripts/python/panel_ridge_validation.py`, review the printed "RIDGE BASELINE VALIDATION COMPLETE" block and the 7 output files, and confirm no `[MAJOR]` sanity violation before treating Ridge as the frozen formal ML baseline and moving on to LSTM.
- [Done — ran] Ridge validation (Tasks A-F) completed. Substantive findings so far: Weekly-only gives lowest pooled MSE for all 3 horizons; Daily+Weekly gives highest mean cross-sectional Rank IC for all 3 horizons; Ridge does not beat Zero/Train Mean on pooled MSE; Ridge predictions are under-dispersed. However, `auto` vs `lsqr` disagreed below the 0.999 threshold on several configs (e.g. daily_only/10d corr≈0.9428, daily_weekly/20d corr≈0.9712, each picking different alphas), so Ridge could not yet be frozen.
- [Done — script written, not yet run] Created `scripts/python/panel_ridge_solver_final_check.py` — the FINAL Ridge numerical-stability check (does not touch universe/split/normalization/features/targets/metric implementation/alpha grid). Compares `auto`, `svd` (numerical reference for collinear/ill-conditioned matrices), and `lsqr_tight` (`tol=1e-8`, `max_iter=100000`) across all 9 frequency x horizon configs; records the **full validation-MSE-vs-alpha curve** per solver/config (not just the selected alpha) to distinguish a flat loss surface from genuine disagreement; computes pairwise prediction agreement (`auto vs svd`, `lsqr_tight vs svd`, `auto vs lsqr_tight`); and applies interpretation rules A-D (SVD+lsqr_tight agree & auto differs → SVD formal; auto+SVD agree & lsqr_tight differs → SVD formal; all agree or disagreement is explained by a flat val-MSE curve with no change in best-MSE/best-Rank-IC setting → alpha-selection sensitivity, freeze with SVD; SVD itself changes which frequency setting is best → Ridge remains unstable, do not freeze). Outputs: `panel_ridge_solver_final_check.csv`, `panel_ridge_alpha_validation_curve.csv`, `panel_ridge_solver_final_predictions.csv`, `panel_ridge_solver_final_summary.txt`.
- [Pending — user to run] Execute `python scripts/python/panel_ridge_solver_final_check.py`, review "RIDGE FINAL SOLVER CHECK COMPLETE" (formal solver recommendation, whether best-MSE/best-Rank-IC setting changes across solvers, whether alpha selection is flat/sensitive vs genuinely unstable, whether Ridge can now be frozen) before deciding whether to proceed to LSTM.
- [Pending] Run the verification gate above and confirm Ridge results are sane before writing any LSTM/Transformer/PathFormer panel script.

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

