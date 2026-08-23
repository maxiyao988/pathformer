# Green Energy Stock Prediction Project

## Current Critical Status — 2026-08-23

### Temporal Integrity Reset (2026-08-23)

A formal temporal-integrity review identified three material issues in the old panel dataset contract. The project is therefore in a disciplined reset: repair the dataset contract, revalidate it independently, and then rerun the formal panel benchmarks before any final empirical conclusions are drawn.

This is not a cancellation of the research direction. The Adaptive Multi-Scale PathFormer architecture remains scientifically relevant, but the previous empirical panel results are not valid evidence under the repaired dataset contract.

Current authoritative status:

- Dataset Contract V2 design: FROZEN / READY TO IMPLEMENT
- Final V2 implementation: PENDING
- Final V2 empirical audit: PENDING
- Formal model reruns: BLOCKED UNTIL AUDIT PASS
- Old panel results: PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

The active workflow is now:

1. Repair/Finalize Dataset Contract V2
2. Independent temporal audit PASS
3. Rebuild/verify final panel sample universe and counts
4. Rerun deployable baselines and formal panel benchmarks
5. Run PathFormer main frequency comparison and mechanism ablation
6. Robustness / inference / interpretation

---

## Current Mainline Objective

The immediate project priority is no longer continued tuning on historical panel outputs. The project is now explicitly focused on:

- repairing the old panel dataset information-set contract,
- sealing the old V1 artifacts as historical snapshots,
- rebuilding the corrected V2 panel,
- independently auditing the V2 temporal semantics,
- then rerunning all formal panel benchmarks.

The architecture remains valid, but previous panel benchmark results are invalid as final evidence because the data contract changed.

---

## Scope (Current Active Track)

Formal panel universe (current design):

- AES
- BEP
- BLDP
- BLNK
- CSIQ
- CWEN
- DQ
- ENPH
- FCEL
- FSLR
- HASI
- JKS
- NEE
- ORA
- PLUG
- RUN
- SEDG

This is the formal 17-stock balanced Daily + Weekly panel retained for the main empirical track. The broader 24-stock universe remains as a retained extension / future robustness path, but it is not the active project mainline.

The research design remains consistent with the advisor-aligned structure:

- Experiment 1: FSLR diagnostic / case-study track
- Experiment 2: panel frequency comparison
- Experiment 3: Daily + Weekly PathFormer mechanism ablation
- Experiment 4: robustness + interpretation

The formal panel mainline is Daily + Weekly, with frequency-specific encoders and late fusion; the router remains within frequency branches rather than a global Daily-vs-Weekly frequency router.

---

## Temporal Integrity Reset — Three Critical Issues

### Issue 1 — Daily information-set definition

Old construction:

```python
daily.iloc[i - H_DAILY:i]
```

with anchor:

```python
D = daily.iloc[i]
```

This used:

- D-90 ... D-1

while the target began at:

```python
y_h(D) = log(Close[D+h] / Close[D])
```

This was NOT ordinary future leakage in the strict target-sense, but it did not match the intended information set. The final intended Daily contract is that the model uses the latest 90 Daily OHLCV bars including the anchor date itself:

- D-89 ... D

with:

```python
daily_feature_end == anchor_date
```

### Issue 2 — Provider Weekly timestamp / information availability

Provider-native Weekly rows are labeled by week start (e.g. Monday), but the row can still contain OHLCV information from the full trading week through the final trading day in that week.

The old builder selected Weekly rows using:

```python
weekly["datetime"] <= anchor_date
```

This could allow a Tuesday anchor to use a Monday-labeled Weekly row containing information from later in the same week. That was genuine feature look-ahead.

The corrected rule is:

```python
weekly_available_date = last actual Daily trading date represented in the week
```

and a Weekly bar may be used only when:

```python
weekly_available_date <= anchor_date
```

This replaces any Friday hard-coding and correctly handles holidays and shortened weeks.

### Issue 3 — Target labels crossed split boundaries

The old split was based only on anchor dates. Because returns are forward-looking, labels near the end of Train and Validation could extend into the next split.

The corrected rule is a common purge over the retained anchor universe:

```python
MAX_HORIZON = 20
```

Remove:

- final 20 common Train anchor dates
- final 20 common Validation anchor dates

Keep Test terminal and unpurged.

Use ONE common retained anchor universe for 5d / 10d / 20d. This is label-boundary overlap / purging, not ordinary feature leakage.

---

## ALL OLD PANEL EMPIRICAL RESULTS ARE PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

ALL panel empirical results generated before the temporal/data-contract fix are now explicitly classified as:

- PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

These results must not be used as final thesis / paper evidence.

This includes prior panel outputs for:

- Naive / constant baselines
- Ridge
- LSTM
- Vanilla Transformer
- SWiM-style Transformer
- Adaptive Multi-Scale PathFormer

Historical metrics remain in the project record for debugging and context, but they are not valid for formal conclusions. The architecture and implementations themselves may remain frozen or development-complete, but the old empirical panel benchmarks are invalid because the underlying panel dataset contract changed.

---

## V1 Backup / Provenance

The old dataset was sealed as a historical snapshot before the V2 rebuild.

Backup location:

- `dataset/audit/pre_temporal_fix_v1/`

Status:

- VERIFIED / SEALED HISTORICAL SNAPSHOT

Verification facts:

- backup size ≈ 292M
- formal ticker snapshot count = 17
- formal metadata count = 17
- SHA-256 checks matched for sampled provenance files including:
  - `panel_build_manifest.csv`
  - `FSLR/meta.csv`

Contents include:

- manifest
- 17 ticker metadata copies
- complete 17-ticker panel snapshot
- old panel result outputs
- README / inventory

The V1 backup must remain read-only.

---

## First V2 Rebuild / Interim Temporal Audit

The first corrected V2 rebuild completed successfully using:

```bash
python scripts/python/panel_build_multiscale_dataset.py
```

Build status:

- SUCCESS

Built directory size:

- ≈231M

The rebuild generated broader 24-ticker green-energy directories, but the formal panel loader retained the intended 17-stock formal universe:

- AES
- BEP
- BLDP
- BLNK
- CSIQ
- CWEN
- DQ
- ENPH
- FCEL
- FSLR
- HASI
- JKS
- NEE
- ORA
- PLUG
- RUN
- SEDG

Interim temporal audit result:

- Daily Window Integrity: PASS
- Weekly Availability / Look-Ahead: PASS
- Target Date and Value Integrity: PASS
- Split Purging: PASS
- Panel Consistency: PASS
- Weekly Raw-Data Semantics: FAIL

Overall result:

- FAIL

This failure was NOT due to Weekly future leakage remaining in the panel. Weekly availability itself passed with zero violations. The remaining issue concerned value semantics between provider-native Weekly bars and Weekly OHLCV reconstructed from canonical Daily data.

---

## Current Retained Formal Panel Counts (Interim V2)

These are the current interim V2 counts, not guaranteed final post-Weekly-reconstruction counts.

Formal retained panel:

- total samples = 43,571

Common dates:

- Original Train: 1822 dates, 2016-01-29 → 2023-04-25
- Retained Train: 1802 dates, 2016-01-29 → 2023-03-27
- Original Validation: 390 dates, 2023-04-26 → 2024-11-11
- Retained Validation: 370 dates, 2023-04-26 → 2024-10-14
- Original Test: 391 dates, 2024-11-12 → 2026-06-05
- Retained Test: 391 dates, unchanged

All split-boundary crossing counts were zero:

- Train crossing 5d = 0
- Train crossing 10d = 0
- Train crossing 20d = 0
- Val crossing 5d = 0
- Val crossing 10d = 0
- Val crossing 20d = 0

These counts were produced by the interim V2 using provider-native Weekly values and may shift slightly after the final Weekly-from-Daily rebuild.

---

## Weekly Raw-Semantics Forensic Investigation

The dedicated diagnostic script was:

- `scripts/python/diagnose_weekly_raw_semantics.py`

It produced:

- `dataset/audit/weekly_raw_semantics_diagnostic.csv`
- `dataset/audit/weekly_raw_semantics_by_ticker.csv`
- `dataset/audit/weekly_raw_semantics_by_period.csv`
- `dataset/audit/weekly_raw_semantics_summary.txt`

Raw provider provenance:

Daily was downloaded with:

```python
yf.download(
    ticker,
    period="max",
    interval="1d",
    auto_adjust=True,
    progress=False,
)
```

Weekly was downloaded separately with:

```python
yf.download(
    ticker,
    period="max",
    interval="1wk",
    auto_adjust=True,
    progress=False,
)
```

Therefore the provider-native Weekly series was not constructed from the saved canonical Daily data.

Installed yfinance version during the audit:

- 1.2.0

Observed relevant defaults:

- `actions=False`
- `back_adjust=False`
- `repair=False`

The saved raw panel files retained only:

- `datetime`
- `open`
- `high`
- `low`
- `close`
- `volume`

No explicit:

- `Adj Close`
- `Dividends`
- `Stock Splits`

were preserved in the existing panel raw series.

---

## Weekly Forensic Results

Total Weekly rows analyzed:

- 19,361

Exact Weekly-vs-Daily aggregation matches:

- 18,991
- 98.09%

OHLC mismatch Weekly rows:

- 370
- 1.91%

Volume mismatch rows:

- 17
- 0.09%

Field-level OHLC mismatches:

- Open = 368
- High = 250
- Low = 206
- Close = 0

This is a key result: provider Weekly Close was systematically consistent with Daily-aggregated Close, while Open / High / Low differed in the mismatching rows.

The earlier common-scale reporting was misleading. The actual common-scale OHLC mismatch rows were:

- 2 / 370

The corrected formal relevance check showed:

- exact formal 26-week window reconstruction: 201 Weekly bars were mismatched and used by the formal panel
- earliest weekly `week_start` actually used in the formal V2 panel: 2015-08-03

Formal-used mismatch counts by ticker:

- AES = 37
- BEP = 37
- CWEN = 29
- HASI = 29
- JKS = 3
- NEE = 33
- ORA = 33

Important: the earlier diagnostic statement that all 370 mismatches could enter the formal panel and that the earliest used week was 1973 was a DIAGNOSTIC BUG and has been corrected.

---

## Weekly Source Decision — Now Selected

### FINAL / SELECTED WEEKLY SOURCE DESIGN

Do NOT use provider-native:

- `interval="1wk"`

for formal model inputs.

Instead, construct Weekly deterministically from the canonical Daily OHLCV series. For each calendar week:

- Weekly Open = first Daily Open
- Weekly High = max Daily High
- Weekly Low = min Daily Low
- Weekly Close = last Daily Close
- Weekly Volume = sum Daily Volume

Define:

```python
weekly_available_date = last actual Daily trading date represented in that week
```

At anchor D, usable Weekly bars satisfy:

```python
weekly_available_date <= D
```

Use:

- latest 26 completed Weekly bars

This does not reduce the formal sample frequency to weekly. Formal samples remain DAILY-anchor samples. Weekly information refreshes once per completed week.

Therefore Daily-only / Weekly-only / Daily+Weekly continue to share the same `(ticker, anchor_date)` formal sample universe.

Scientific motivation: using one canonical Daily source for both Daily and Weekly representations isolates temporal frequency / temporal scale rather than confounding frequency with provider interval-specific adjustment semantics.

---

## Auto-Adjust Investigation

A separate Daily price-convention investigation was completed.

The raw Daily data were downloaded with:

```python
auto_adjust=True
```

The local yfinance adjusted-OHLC logic was inspected. This established that adjusted OHLC is derived using an adjustment ratio related to:

- `Adj Close / Close`

The key methodological distinction is that changing `auto_adjust=True` to `auto_adjust=False` is not just a data-cleaning choice: it changes the economic meaning of the target.

The current target:

```python
y_h(D) = log(Close[D+h] / Close[D])
```

under `auto_adjust=True` should be described as:

- forward adjusted-price log return
- or cautiously: dividend- and split-adjusted / total-return-like equity return

rather than pure raw-price return.

---

## Daily Adjustment Sensitivity Audit

The dedicated diagnostic was:

- `scripts/python/diagnose_daily_adjustment_sensitivity.py`

This was DIAGNOSTIC ONLY and did not:

- rebuild data
- alter raw CSVs
- modify frozen code
- train models

The comparison was:

A. `interval="1d"`, `auto_adjust=True`

B. `interval="1d"`, `auto_adjust=False`, `actions=True`

The diagnostic used the actual retained formal panel anchors and reproduced the existing train-only normalization logic.

Key findings:

- AES: ~100% non-unit `AdjClose/Close` ratio dates; median `|ratio - 1|` ≈ 0.377
- BEP: ~99.5% non-unit; median `|ratio - 1|` ≈ 0.397
- NEE: ~99.8% non-unit; median `|ratio - 1|` ≈ 0.573
- ORA: ~100% non-unit; median `|ratio - 1|` ≈ 0.075

Formal 90-day windows:

- effectively constant adjustment-ratio windows = 28,044
- adjustment ratio changes inside window = 15,527

Normalized-feature sensitivity under the actual existing normalization:

- Max absolute normalized difference:
  - p50 = 0.00000000
  - p90 = 0.24277201
  - p95 = 0.43446768
  - p99 = 0.71394871
  - max = 1.09281248
- Mean absolute normalized difference:
  - p50 = 0.00000000
  - p90 = 0.12135357
  - p95 = 0.31501018
  - p99 = 0.52463290
  - max = 0.79564100
- RMSE normalized difference:
  - p50 = 0.00000000
  - p90 = 0.14079998
  - p95 = 0.35281325
  - p99 = 0.58735475
  - max = 0.89019328

Target sensitivity diagnostic:

- Mean absolute adjusted-vs-price target difference:
  - 5d ≈ 0.000154
  - 10d ≈ 0.000396
  - 20d ≈ 0.000788
- Maximum absolute difference:
  - 5d ≈ 0.077679
  - 10d ≈ 0.077679
  - 20d ≈ 0.080386

The diagnostic observed:

- 4,492 target intervals containing dividend events
- 108 target intervals containing stock-split events

Important reporting caveat: the temporary target-sensitivity diagnostic used different usable denominators across 5d / 10d / 20d due to temporary download/date-alignment availability. Therefore the reported 40.6–40.8% “target differs” percentages are DIAGNOSTIC ONLY and should not be used as formal thesis statistics unless the alignment is explicitly repaired and verified.

---

## Final Daily Price-Convention Decision

Current research-design decision:

- RETAIN `auto_adjust=True` for the canonical Daily series.

Reason:

The research objective is stock/equity return prediction, not specifically capital-price appreciation excluding distributions. Therefore the preferred target is:

```python
forward adjusted-price log return
```

rather than an ex-dividend mechanical price-only return.

The sensitivity audit did not demonstrate a sufficiently material exploitable future-information mechanism that requires abandoning `auto_adjust=True`.

Retrospective adjustment alone is not automatically equivalent to usable look-ahead leakage.

The current normalizer remains:

- per ticker
- per frequency
- per feature
- fit on TRAIN only

with:

```python
X_norm = (X - train_mean) / train_std
```

This reduces sensitivity to common multiplicative rescaling and remains compatible with the panel design.

Final canonical source decision:

- Daily: yfinance `interval="1d"`, `auto_adjust=True`
- Weekly: deterministically aggregate from the same canonical Daily series
- Target: `log(adjusted Close[D+h] / adjusted Close[D])`

Use the terminology:

- forward adjusted-price log return

Do not call ordinary `auto_adjust=False` Yahoo Close a “fully raw traded price” without explicit evidence.

---

## Final Dataset Contract V2 — Current Target Specification

### Universe

Formal 17 stocks:

- AES, BEP, BLDP, BLNK, CSIQ, CWEN, DQ, ENPH, FCEL, FSLR, HASI, JKS, NEE, ORA, PLUG, RUN, SEDG

### Canonical Daily data

Provider:

- yfinance

Interval:

- 1d

Price convention:

- `auto_adjust=True`

Features:

- open
- high
- low
- close
- volume

Daily window:

- 90 Daily trading observations including anchor D

### Weekly data

No provider-native 1wk formal input is used.

Weekly is constructed deterministically from the canonical Daily series:

- first Open
- max High
- min Low
- last Close
- summed Volume

Weekly availability:

- actual final trading date represented in the week

Weekly input window:

- latest 26 completed Weekly bars

### Sample anchor

Daily trading date D.

All frequency settings use the same formal `(ticker, anchor_date)` sample universe.

### Target

For h in {5, 10, 20}:

```python
y_h(D) = log(Close[D+h] / Close[D])
```

where Close uses the canonical `auto_adjust=True` Daily series and D+h means future DAILY trading observations.

Recommended description:

- forward adjusted-price log return

### Split / purge

One common chronological panel split.

Purge:

- final 20 common Train anchor dates
- final 20 common Validation anchor dates

Test:

- terminal / unpurged

Normalization:

- per ticker
- per frequency
- per feature
- fit on Train only

---

## Important: Dataset V2 Is Not Yet Formally Accepted

Although the research-design decisions are largely frozen, the current Dataset V2 has not yet received an overall PASS.

Why:

- the current built V2 still includes provider-native Weekly values in the interim rebuild,
- the builder must next be modified so Weekly is reconstructed from canonical Daily,
- the independent audit must then be updated to validate the final Weekly-from-Daily contract.

Current status is therefore:

- Dataset Contract design: FROZEN / READY TO IMPLEMENT
- Final V2 implementation: PENDING
- Final V2 empirical audit: PENDING
- Formal model reruns: BLOCKED UNTIL AUDIT PASS

Do not describe the current dataset directory as final / frozen.

---

## Next Execution Roadmap

1. [DONE] Identify temporal integrity issues in the old panel.
2. [DONE] Seal the PRE-TEMPORAL-FIX V1 backup.
3. [DONE] Correct the Daily-window information-set logic.
4. [DONE] Correct the Weekly availability logic.
5. [DONE] Implement the common 20-date Train / Validation purge.
6. [DONE] Build the interim V2 and run the independent temporal audit.
7. [DONE] Diagnose provider Weekly-vs-Daily raw semantics.
8. [DONE] Trace yfinance Daily / Weekly provenance.
9. [DONE] Correct the forensic formal-relevance diagnostics.
10. [DONE] Decide the Weekly source: reconstruct from canonical Daily.
11. [DONE] Audit the `auto_adjust` semantics.
12. [DONE] Run the Daily adjustment sensitivity analysis.
13. [DONE] Decide the canonical Daily convention: yfinance 1d, `auto_adjust=True`.
14. [NEXT] Modify the Dataset V2 builder so Weekly is deterministic Daily aggregation.
15. [NEXT] Modify / extend the independent temporal audit to validate the final Weekly-from-Daily contract.
16. [PENDING] Rebuild final Dataset V2.
17. [PENDING] Run the independent temporal audit until OVERALL: PASS.
18. [PENDING] Freeze final V2 counts / provenance.
19. [PENDING] Rerun deployable naive baselines: Zero predictor and global Train-Mean predictor.
20. [PENDING] Rerun Ridge.
21. [PENDING] Rerun LSTM.
22. [PENDING] Rerun Vanilla Transformer.
23. [PENDING] Rerun SWiM-style Transformer.
24. [PENDING] Rerun Adaptive PathFormer on Daily-only / Weekly-only / Daily+Weekly across 5d / 10d / 20d.
25. [PENDING] Run PathFormer mechanism ablation: Single / Fixed / Static / Adaptive.
26. [PENDING] Run at least 5-seed robustness.
27. [PENDING] Use HAC / Newey-West or block-bootstrap inference where appropriate.
28. [PENDING] Run router interpretation / regime analysis.
29. [PENDING] Final thesis / paper tables and writeup.

---

## Baseline Terminology

Formal deployable naive baselines:

- Zero predictor: `y_hat = 0`
- Global Train-Mean predictor: `y_hat_test = mean(y_train)`
- Optional: per-ticker Train-Mean predictor

A constant using:

- `mean(y_test)`

is NOT a valid OOS benchmark.

If retained historically, label it:

- `Oracle test-mean constant — diagnostic only`

---

## PathFormer / Experiment Design Must Remain

Do not remove the existing advisor-aligned research design.

Experiment 1:

- FSLR diagnostic / case study

Experiment 2:

- panel frequency comparison

Experiment 3:

- Daily + Weekly PathFormer mechanism ablation

Experiment 4:

- robustness + router interpretation

PathFormer architecture design remains:

- Daily branch: 90 bars; patch scales [5, 10, 20, 30]
- Weekly branch: 26 bars; patch scales [2, 4, 8, 13]
- Daily+Weekly: independent frequency-specific branches with late concatenation
- no global Daily-vs-Weekly frequency router
- router: adaptive selection across scales WITHIN each frequency

Mechanism ablation:

- Single-scale
- Fixed multi-scale
- Static learned weighting
- Adaptive router

Do not claim that the adaptive router is already validated.

---

## Capacity / Inference Caveats

- Daily+Weekly models have larger capacity than single-frequency models.
- A Daily+Weekly performance improvement alone does not prove pure frequency complementarity unless capacity is controlled or acknowledged.
- Apple MPS is not bitwise deterministic.
- Single-seed development results are not formal robustness evidence.
- 5d / 10d / 20d overlapping forward returns induce serial dependence.
- Later formal inference should consider HAC / Newey-West or block bootstrap.

---

## Do Not Do / Current Priority

Until Dataset Contract V2 passes the independent temporal audit:

- do not tune PathFormer hyperparameters
- do not rerun long formal models
- do not interpret old panel metrics as final results
- do not claim adaptive routing works
- do not claim Weekly helps forecasting
- do not claim Daily+Weekly complementarity
- do not change the target formula
- do not add new features
- do not change normalization
- do not modify horizons
- do not change dependencies unless required
- do not overwrite old audit evidence

The current code and data remain protected: this task is documentation-only and does not alter any source, dataset, raw data, or model artifacts.

---

## Historical Benchmark Archive — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

The detailed historical benchmark sections below are retained as debugging and narrative context only. They are no longer valid for final claims. The documentation that follows is a historical archive under the old data contract, not a current benchmark verdict.

## Status of Old Panel Results (PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY)

The old panel-based Ridge, LSTM, Vanilla Transformer, SWiM, and Adaptive PathFormer results remain documented as historical and debugging artifacts, but they are no longer valid as final research evidence.

These results were produced under the old panel information-set contract. The Daily-window definition, Weekly row availability rule, and split sample universe all changed under the temporal repair, so old panel metrics must not enter final paper tables or final research conclusions.

This includes old Weekly-only and Daily+Weekly experiments, which were contaminated by week-start look-ahead leakage. Daily-only results are also no longer directly comparable because the Daily input window and the retained split sample universe are changing.

The most recent Adaptive PathFormer 10d/20d findings are retained as historical diagnostics and should be read as diagnostics only. A brief summary is: all six configurations had negative pooled Corr; Rank IC was weak or mostly negative; predictions were strongly under-dispersed; none beat the zero predictor on pooled MSE.

These are not final scientific conclusions.

---

## Completed Milestones

### Data and Pipeline

- Multi-frequency data ingestion and cleaning completed.
- Cross-frequency alignment audit completed with no leakage in retained samples.
- Multi-scale dataset rebuilt and audited.
- Lookback selection study completed; working window set (w_star):
  - hourly 24, halfday 20, daily 90, weekly 26

### Historical Completed Milestones — PRE-TEMPORAL-FIX

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

## Historical Baseline Snapshot — PRE-TEMPORAL-FIX

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

## Ridge Panel Baseline — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

- Universe: historical 17-stock balanced panel
- Daily / Weekly / Daily+Weekly
- Horizons: 5d / 10d / 20d
- Formal solver: **SVD**
- Alpha selected using validation only
- Cross-sectional Rank IC computed by date
- Numerical solver robustness check passed
- Final solver ambiguity resolved: **Ridge is frozen** as an implementation choice for the historical V1 benchmark path
- The V1 pipeline was initially treated as leakage-free, but the 2026-08-23 audit later identified Weekly week-start look-ahead and label-boundary overlap in the old panel contract.
- Ridge does not beat naive baseline on pooled MSE
- Weekly-only has the best pooled MSE at 5d / 10d / 20d
- Daily+Weekly has the best mean cross-sectional Rank IC at 5d / 10d / 20d
- Ridge predictions are under-dispersed

This was the historically frozen V1 development benchmark for the panel and must be rerun on the final Dataset V2 before it can inform formal conclusions.

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

## LSTM Panel Baseline — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

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
6. The completed Vanilla Transformer control provides an additional check on whether the horizon-dependent temporal-frequency pattern persists under generic self-attention. SWiM has subsequently been completed under the same frozen panel protocol, and the next model-development step is the Adaptive Multi-Scale PathFormer.

Do not claim profitable trading performance, causal effects, or universal superiority of Daily+Weekly or LSTM over naive forecasting.

---

## Vanilla Transformer Panel Development Baseline — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

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

## SWiM-style Panel Transformer — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY

SWiM is the improved Transformer A in the panel methodological control ladder:

Ridge
→ LSTM
→ Vanilla Transformer
→ SWiM-style Transformer
→ Adaptive Multi-Scale PathFormer

Its scientific question is:

> Does imposing structured local / shifted-window attention improve on generic global self-attention, before introducing adaptive multi-scale routing?

This is a methodological control / improved Transformer baseline. It is not the proposed final adaptive model, and it remains separate from the advisor's literal Experiment 2 definition, which is a panel frequency-configuration comparison.

### Historical FSLR SWiM forensic audit

The older FSLR SWiM implementation is still relevant as a historical comparison model, but it is not the blueprint for the panel benchmark.

Historical FSLR SWiM used:

- input = one concatenated 160-token sequence:
  - Hourly 24 + Half-Day 20 + Daily 90 + Weekly 26
- Linear(5 -> 32)
- four shifted-window blocks
- d_model = 32
- nhead = 4
- window_size = 16
- alternating shift = 0 / 8
- Pre-LN
- mean pooling over the full concatenated sequence
- one final prediction head
- no frequency-specific branches
- no late fusion
- no frequency embedding
- no multi-scale router
- no adaptive scale mechanism

Important forensic finding: the historical implementation used `torch.roll` for shifted windows but did not use a shifted-window boundary attention mask. Therefore cyclic wrap-around tokens could attend across an artificial boundary.

This historical implementation should not be copied literally into the panel benchmark.

Additional audit notes:

- `window_size=16` belonged to the historical 160-token concatenated FSLR sequence and was not reused as a panel hyperparameter.
- the historical FSLR SWiM remains useful as the improved-Transformer comparison model for the single-stock case study, but it is not the architectural blueprint for the frozen panel benchmark.

### Frozen panel SWiM architecture

Formal script:

- `scripts/python/panel_baseline_swim.py`

Frozen architecture:

Single-frequency branch:

```text
sequence [B,T,F]
-> Linear(F,64)
-> fixed sinusoidal positional encoding
-> SWiM block 1: local window attention, shift=0
-> SWiM block 2: shifted-window attention
-> masked mean pooling over valid timesteps
-> representation
-> prediction head
```

Frequency-specific parameters:

Daily:
- `window_size = 10`
- `shift_size = 5`

Weekly:
- `window_size = 4`
- `shift_size = 2`

Shared architecture:

- d_model = 64
- nhead = 4
- dim_feedforward = 128
- dropout = 0.1
- exactly 2 SWiM blocks per branch
- Post-LN residual structure
- ReLU FFN
- fixed sinusoidal positional encoding

Daily+Weekly:

```text
Daily SWiM branch ----\
                       -> concat -> Linear(128,64) -> ReLU -> Linear(64,1)
Weekly SWiM branch ---/
```

The panel SWiM contains:

- frequency-specific encoders
- late concat fusion

and contains no:

- early raw-frequency fusion
- adaptive router
- learned scale router
- multi-scale patch competition
- gated fusion
- cross-frequency attention
- ticker embedding

Parameter counts:

- daily_only = 67,393
- weekly_only = 67,393
- daily_weekly = 142,977

### Correct shifted-window implementation

The key implementation correction relative to the historical FSLR SWiM is that the panel version uses a correct local-window masking scheme rather than a cyclic wrap-around approximation.

The panel implementation uses:

- zero padding rather than repeated-last-token padding
- explicit validity mask
- shifted-window region IDs
- exact hard attention blocking for artificial wrap-around interactions
- padded keys blocked from valid queries
- padded query rows kept numerically defined with a dummy allowed key and then hard-zeroed
- reverse shift after window attention
- hard-zero invalid padded positions after sublayers
- masked mean pooling over real timesteps only

Masked mean pooling was selected because with only two local / shifted-window blocks, a final-timestep token does not have the same global receptive field as a global-attention Transformer. Therefore final-token pooling could artificially handicap the SWiM baseline.

This should be recorded as a comparison caveat, not as a bug:

- Vanilla uses final-token representation.
- SWiM uses masked mean pooling.

The two baseline designs differ both in attention mechanism and pooling strategy.

### Frozen training protocol

The panel SWiM benchmark follows the same frozen panel protocol as the other deep baselines:

- frozen 17-stock balanced panel
- Daily / Weekly / Daily+Weekly
- horizons = 5d / 10d / 20d
- seed = 42
- batch_size = 32
- learning_rate = 1e-4
- optimizer = Adam
- loss = HuberLoss(delta=1.0)
- gradient clipping max_norm = 1.0
- max_epochs = 100
- early_stopping_patience = 10
- validation Huber loss for checkpoint selection
- restore best validation checkpoint before test evaluation
- train-only per-ticker / per-frequency / per-feature normalization
- forward log return computed from auto-adjusted Close (adjusted-price log return)
- same frozen sample index as Ridge / LSTM / Vanilla
- same pooled + ticker-average evaluation
- same per-date cross-sectional Rank IC implementation

Formal execution backend:

- Apple MPS was used for the completed benchmark.

### MPS numerical-debugging history

This section is included for scientific transparency, but it is not meant to dominate the SWiM narrative.

Initial clean formal MPS Daily-5d runs produced intermittent epoch-1:

- `train_loss = NaN`
- `val_loss = NaN`

CPU full-epoch diagnostics remained finite.

A/B/C/D/E real-MPS causal diagnostics were then run as separate processes:

- A = clean production-like diagnostic
- B = explicit synchronize after model construction
- C = model parameter host read
- D = explicit synchronize immediately before training loop
- E = CPU RNG fingerprint only

All five completed:

- 969 / 969 training batches
- full validation
- finite losses

Therefore:

- there was no evidence that an explicit `torch.mps.synchronize()` was necessary to fix the problem
- no permanent synchronize workaround was introduced

A remaining concrete execution-order difference was identified.

Old formal single-frequency form:

```python
return model(x_batch.to(device)), y_batch.to(device)
```

Python evaluates this approximately as:

- `x -> device`
- `forward`
- `y -> device`

Diagnostic path used:

```python
x -> device
y -> device
forward
```

Production batched_forward was therefore changed to explicit pre-transfer:

```python
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
return model(x_batch), y_batch
```

and analogously for Daily+Weekly:

```python
x_daily -> device
x_weekly -> device
y -> device
then model forward
```

This did not alter:

- architecture
- attention mask semantics
- optimizer
- loss
- hyperparameters
- labels
- sample split
- metrics

After explicit pre-transfer:

- a clean Daily-only / 5d formal MPS run completed to early stopping
- the complete 9-configuration MPS benchmark subsequently completed without the training-loss NaN failure

Use conservative wording:

> "The explicit pre-transfer implementation was retained because it produced a successful clean formal run and a successful full benchmark. However, the exact low-level MPS root cause was not formally isolated, so this should not be described as definitive causal proof that transfer ordering alone caused the original NaN."

This wording avoids unsupported claims.

It is also important to note that:

- the `PyArrow` deprecation warning remained unrelated to model logic
- the local SciPy / NumPy compatibility warning remains an environment cleanup item, but there is no evidence from this experiment that it caused the SWiM MPS failure

### Formal 9-configuration SWiM results

The nominal-seed-42 full panel benchmark completed successfully:

- 3 frequency settings × 3 horizons = 9 experiments
- total experiment runtime ≈ 1h 20m 49s

The formal benchmark used the repository outputs:

- `scripts/python/panel_baseline_swim.py`
- `dataset/audit/panel_swim_summary_metrics.csv`
- `dataset/audit/panel_swim_test_predictions.csv`
- `dataset/audit/panel_swim_training_history.csv`
- `dataset/audit/panel_swim_rank_ic_by_date.csv`
- `dataset/audit/panel_swim_vs_baselines_summary.csv`
- `dataset/audit/panel_swim_final_summary.txt`
- `dataset/audit/panel_swim_full_9config_run.log`

DAILY ONLY

5d:
- best_epoch = 8
- epochs_trained = 18
- val_loss = 0.004226
- test_mse = 0.010510
- test_mae = 0.071562
- test_corr = -0.031094
- ticker_avg_corr = -0.044086
- mean_rank_ic = -0.048280
- median_rank_ic = -0.041667
- rank_ic_std = 0.264860
- positive_ic_ratio = 0.443878
- direction_accuracy = 0.467287
- pred_std = 0.01618801
- true_std = 0.099987
- pred_std_true_std_ratio = 0.161901

10d:
- best_epoch = 2
- epochs_trained = 12
- val_loss = 0.008011
- test_mse = 0.020151
- test_mae = 0.098892
- test_corr = -0.007269
- ticker_avg_corr = -0.035056
- mean_rank_ic = 0.020621
- median_rank_ic = 0.017157
- rank_ic_std = 0.258874
- positive_ic_ratio = 0.540816
- direction_accuracy = 0.515906
- pred_std = 0.02860983
- true_std = 0.138619
- pred_std_true_std_ratio = 0.206392

20d:
- best_epoch = 7
- epochs_trained = 17
- val_loss = 0.013977
- test_mse = 0.046321
- test_mae = 0.155787
- test_corr = -0.093693
- ticker_avg_corr = 0.008221
- mean_rank_ic = -0.079419
- median_rank_ic = -0.068627
- rank_ic_std = 0.240799
- positive_ic_ratio = 0.397959
- direction_accuracy = 0.456182
- pred_std = 0.06023693
- true_std = 0.191052
- pred_std_true_std_ratio = 0.315291

WEEKLY ONLY

5d:
- best_epoch = 17
- epochs_trained = 27
- val_loss = 0.004000
- test_mse = 0.011385
- test_mae = 0.074432
- test_corr = 0.044513
- ticker_avg_corr = 0.159486
- mean_rank_ic = 0.020262
- median_rank_ic = 0.001225
- rank_ic_std = 0.282901
- positive_ic_ratio = 0.500000
- direction_accuracy = 0.512155
- pred_std = 0.03046427
- true_std = 0.099987
- pred_std_true_std_ratio = 0.304681

10d:
- best_epoch = 17
- epochs_trained = 27
- val_loss = 0.008334
- test_mse = 0.028319
- test_mae = 0.119735
- test_corr = -0.058474
- ticker_avg_corr = 0.015638
- mean_rank_ic = -0.035014
- median_rank_ic = -0.058824
- rank_ic_std = 0.261693
- positive_ic_ratio = 0.418367
- direction_accuracy = 0.489346
- pred_std = 0.07175879
- true_std = 0.138619
- pred_std_true_std_ratio = 0.517671

20d:
- best_epoch = 8
- epochs_trained = 18
- val_loss = 0.014541
- test_mse = 0.045282
- test_mae = 0.153759
- test_corr = -0.078233
- ticker_avg_corr = -0.023422
- mean_rank_ic = -0.060881
- median_rank_ic = -0.046569
- rank_ic_std = 0.235785
- positive_ic_ratio = 0.428571
- direction_accuracy = 0.460534
- pred_std = 0.06766824
- true_std = 0.191052
- pred_std_true_std_ratio = 0.354188

DAILY + WEEKLY

5d:
- best_epoch = 8
- epochs_trained = 18
- val_loss = 0.004202
- test_mse = 0.010543
- test_mae = 0.070441
- test_corr = -0.043694
- ticker_avg_corr = 0.009967
- mean_rank_ic = -0.024711
- median_rank_ic = -0.039216
- rank_ic_std = 0.260778
- positive_ic_ratio = 0.443878
- direction_accuracy = 0.479142
- pred_std = 0.01522880
- true_std = 0.099987
- pred_std_true_std_ratio = 0.152307

10d:
- best_epoch = 2
- epochs_trained = 12
- val_loss = 0.008514
- test_mse = 0.025703
- test_mae = 0.110350
- test_corr = -0.062927
- ticker_avg_corr = -0.010852
- mean_rank_ic = -0.014862
- median_rank_ic = -0.004902
- rank_ic_std = 0.286935
- positive_ic_ratio = 0.482143
- direction_accuracy = 0.496098
- pred_std = 0.06260008
- true_std = 0.138619
- pred_std_true_std_ratio = 0.451599

20d:
- best_epoch = 1
- epochs_trained = 11
- val_loss = 0.014312
- test_mse = 0.036517
- test_mae = 0.134719
- test_corr = NaN
- ticker_avg_corr = NaN
- mean_rank_ic = NaN
- median_rank_ic = NaN
- rank_ic_std = NaN
- positive_ic_ratio = NaN
- direction_accuracy = 0.525810
- pred_std = 9.313226e-10
- true_std = 0.191052
- pred_std_true_std_ratio = 4.874717e-09

### Constant-collapse interpretation

The Daily+Weekly / 20d SWiM result is a constant-prediction collapse.

Evidence:

- `PredStd ≈ 9.3e-10`
- `PredStd / TrueStd ≈ 4.9e-9`

Therefore Pearson correlation and cross-sectional Rank IC are undefined.

This is not the earlier MPS training NaN failure. It is a model-output collapse that occurred after training remained finite.

Importantly:

- The earlier MPS issue was: training loss became NaN.
- The D+W 20d result was: training stayed finite, but predictions collapsed to an almost constant value.

Do not interpret its `MSE = 0.036517` as evidence that D+W is the best SWiM 20d model.

The naive 20d zero-return MSE is approximately 0.036503, so the collapsed SWiM is effectively behaving like an unconditional near-zero predictor.

### Scientific interpretation

Within SWiM, the mean Rank IC best by horizon among finite configurations is:

- 5d: Weekly-only = 0.020262
- 10d: Daily-only = 0.020621
- 20d: no positive configuration
  - Daily = -0.079419
  - Weekly = -0.060881
  - Daily+Weekly = undefined due to constant collapse

Point-MSE pattern:

- 5d: Daily-only = 0.010510 is lowest among valid SWiM settings
- 10d: Daily-only = 0.020151 is lowest
- 20d: D+W numerical MSE = 0.036517 must not be treated as a valid model superiority result because it is a constant-collapse cell; among non-collapsed SWiM settings, Weekly = 0.045282 < Daily = 0.046321

Core interpretation:

- fixed local / shifted-window attention does not consistently improve predictive performance
- SWiM is materially weaker than the strongest existing panel baselines in cross-sectional ranking
- fixed local receptive fields remain horizon dependent
- Daily+Weekly fixed fusion is not automatically beneficial
- at 20d, fixed D+W SWiM can collapse completely
- therefore imposing a fixed local temporal scale is not sufficient
- this strengthens the motivation for testing adaptive within-frequency multi-scale selection

This does not prove PathFormer will work, or that adaptive routing is already validated.

It also remains true that SWiM does not beat the naive point-forecast MSE benchmark in any of the 9 configurations.

For D+W 20d, the numerical closeness to naive MSE results from collapse and should be interpreted accordingly.

### Comparison caveat

The SWiM development results are:

- single nominal seed = 42
- descriptive development-benchmark results
- not a formal final paper result

Formal multi-seed robustness has not yet been performed.

Apple MPS is not bitwise deterministic even under a nominal fixed seed, as already observed in the Vanilla and SWiM development process.

Therefore:

- do not attach formal statistical significance to single-seed metric differences
- do not call small Rank IC differences conclusive
- overlapping 5d / 10d / 20d future-return targets induce serial dependence
- later inference should use serial-dependence-aware methods such as HAC / Newey-West or a block bootstrap where appropriate

---

## Constant Baseline Terminology (Required Methodological Note)

The formal OOS constant baseline must not use the TEST-SET mean. A constant defined as `mean(y_test)` is an oracle diagnostic because it uses test labels.

It may be retained only if explicitly labeled:

- `Oracle test-mean constant — diagnostic only`

The valid deployable baselines are:

- `Zero predictor`: `y_hat = 0`
- `Global train-mean predictor`: `y_hat_test = mean(y_train)`

Optionally later:

- `Per-ticker train-mean predictor`

Formal benchmark tables should use the train mean, not the test mean.

---

## Dataset Version / Provenance

The repaired dataset should be treated as a new information-set version:

- `Dataset Contract V1`: old / pre-temporal-fix / diagnostic
- `Dataset Contract V2`: design frozen; final implementation and empirical acceptance pending OVERALL: PASS

Desired V2 provenance fields:

- `dataset_contract_version = 2`
- `daily_includes_anchor = True`
- `weekly_availability_rule = actual_last_trading_day`
- `split_purge_dates = 20`
- `target_definition = forward_adjusted_price_log_return`

Old formal outputs should be backed up under the sealed V1 archive:

- `dataset/audit/pre_temporal_fix_v1/`

and post-fix results should not be silently mixed with pre-fix results.

---

## Historical Pre-Temporal-Fix Assessment — Superseded

This section reflects the project state before the 2026-08-23 temporal-integrity reset. It does not represent the current panel empirical status, which remains blocked until the final Dataset V2 passes the independent audit.

- FSLR full-frequency naive-fusion PathFormer remains a negative diagnostic result and should remain documented as such.
- The active mainline is the frozen 17-stock Daily + Weekly panel on the shared sample index.
- Shared panel infrastructure is stable: loader, split logic, train-only normalization, and cross-sectional Rank IC are all in place and used by Ridge, LSTM, Vanilla Transformer, and SWiM.
- Ridge ML baseline is complete and frozen with SVD.
- LSTM DL baseline is complete as a single-seed development benchmark and its architecture/training protocol are frozen.
- Vanilla Transformer is now complete as the generic self-attention control baseline under the same frozen panel protocol; its 9-configuration development benchmark is complete and frozen.
- SWiM-style Transformer is complete as a single-seed development benchmark and its architecture/training protocol are frozen.
- SWiM does not provide a consistent improvement over the existing panel controls; it remains a fixed-structure local-attention baseline whose 20d Daily+Weekly branch collapses to a near-constant prediction.
- The panel methodological control ladder is now: Ridge → LSTM → Vanilla Transformer → SWiM-style Transformer → Adaptive Multi-Scale PathFormer.
- The control ladder is four-fifths complete before the formal PathFormer panel model.
- Vanilla and SWiM both show horizon-dependent frequency preference, which reinforces the view that fixed temporal representations are not universally optimal.
- Fixed Daily+Weekly fusion is useful at some horizons but not uniformly superior, which strengthens the motivation for structured multi-scale and adaptive scale-selection mechanisms.
- This strengthens the motivation to test adaptive multi-scale selection without claiming the hypothesis is already proven.
- This control ladder complements the advisor's literal experimental spine; it is not the advisor's literal Experiment 2.
- The advisor's literal experimental spine is: Experiment 1 = FSLR diagnostic model comparison, Experiment 2 = panel frequency-configuration comparison, Experiment 3 = Daily+Weekly PathFormer mechanism ablation, Experiment 4 = robustness + interpretability.
- The old statement that "panel-level ranking quality will be reported separately once Experiment 2 is run" is no longer correct; panel Rank IC is already being computed and reported for the Ridge, LSTM, Vanilla, and SWiM baselines.

For the historical FSLR late-fusion experiments, `concat` remains the safer default branch for interpretation and `gated` remains a secondary ablation. This does not pre-select the fusion mechanism for the formal panel PathFormer benchmark, which remains pending but is now downstream of the completed Vanilla and SWiM controls.

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

### New Active Roadmap (Advisor-Aligned, Temporally Repaired)

The immediate workflow has been reset to the following order.

#### PHASE A — TEMPORAL REPAIR

1. Review the actual Copilot diff for:
   - `scripts/python/panel_build_multiscale_dataset.py`
   - `scripts/python/panel_common.py`
   - `scripts/python/panel_verify_temporal_integrity.py`
2. Correct any implementation / audit issues.
3. Back up old dataset manifests and model outputs.
4. Rebuild Dataset Contract V2.
5. Run the full independent temporal-integrity audit.
6. Do not proceed unless the audit passes.

#### PHASE B — CLEAN BASELINE RE-BENCHMARK

7. Audit rebuilt sample counts and target distributions for train / validation / test: `train y mean`, `validation y mean`, `test y mean` for 5d / 10d / 20d.
8. Run the zero predictor and train-mean predictor.
9. Re-run the formal Linear / Ridge baseline.
10. Re-run the formal panel benchmark controls required for the final paper: LSTM, Vanilla Transformer, SWiM on the same repaired dataset.

#### PHASE C — PATHFORMER

11. Re-run Adaptive PathFormer main frequency comparison: Daily, Weekly, Daily + Weekly across 5d / 10d / 20d.
12. Only after the main PathFormer comparison is stable, perform the PathFormer scale-mechanism ablation on the chosen stable frequency configuration:
   - Single-scale
   - Fixed multi-scale
   - Static learned scale weights
   - Adaptive router
13. Later: multi-seed robustness, router interpretation, inference accounting for overlapping labels, and HAC / Newey-West or block-bootstrap style uncertainty where appropriate.

This roadmap explicitly places the dataset repair before any formal benchmark re-run. Old results remain in the document as historical diagnostics, not as valid final evidence.

### Panel Pipeline Status

- `scripts/python/panel_universe.py` — finalized the green-energy panel universe and ticker filtering.
- `scripts/python/panel_download_daily_weekly.py` — daily/weekly OHLCV download pipeline.
- `scripts/python/panel_build_multiscale_dataset.py` — builds the per-ticker panel dataset at Daily and Weekly scales only.
- `scripts/python/panel_common.py` — active shared loader / split / train-only normalization / metrics implementation for the frozen panel.
- `scripts/python/panel_baseline_ridge.py` — completed and frozen Ridge baseline for the panel.
- `scripts/python/panel_ridge_validation.py` — completed Ridge validation and naive comparison workflow.
- `scripts/python/panel_ridge_solver_final_check.py` — completed final SVD solver confirmation.
- `scripts/python/panel_baseline_lstm.py` — completed single-seed LSTM development benchmark for the panel.
- `scripts/python/panel_baseline_swim.py` — completed single-seed SWiM development benchmark for the panel, with architecture and training protocol frozen.
- `scripts/python/panel_adaptive_scale_experiment.py` — exploratory PathFormer-family prototype; not yet the formal panel main benchmark.

Current implemented formal/development baselines:

- Naive baseline: DONE
- Ridge: DONE / FROZEN
- LSTM: DONE / FROZEN development architecture/protocol
- Vanilla Transformer: DONE / FROZEN development control
- SWiM-style / windowed-attention Transformer: DONE / FROZEN development benchmark
- Late-Fusion PathFormer / adaptive multi-scale PathFormer: prototype exists, but formal benchmark version remains pending

The current priority is now the temporal repair and clean re-benchmark, not another round of adaptive PathFormer tuning. The pathformer implementation remains frozen as a code artifact, but the formal dataset and benchmark evidence must be repaired before any final empirical comparison is made.

### FSLR Full-Frequency Case Study (A1–A7) as Historical Evidence

The A1–A7 sequence should be preserved as the historical evidence showing why the old full-frequency setting is not a viable mainline path. It remains valuable as a stress-test narrative and as a cautionary baseline, but it should not be re-used as the direct panel experiment template.

- **A1**: patch-size search; confirms scale heterogeneity.
- **A2**: fixed multiscale vs single scale; partial gains but inconsistent.
- **A3**: static learned scale weight; not a stable improvement.
- **A4**: adaptive router; unstable and not clearly superior in the FSLR setting.
- **A5**: all-frequency ablation; demonstrates instability and scale explosion.
- **A6 / A7**: remain secondary and should not dominate the main narrative when the base model class is already known to be unstable.

### Final Interpretation

The project has moved from a failed all-frequency replication attempt toward a more defensible scientific strategy:

- Keep the adaptive multi-scale idea as the core contribution.
- Remove the unstable full-frequency naive fusion from the panel mainline.
- Build the panel around an architecture that respects frequency-specific structure first, then combines them with stable late fusion and optional router mechanisms.
- Use FSLR as a diagnostic case study that explains why the more robust panel design is necessary.
- Maintain the advisor framework, but do not treat old panel results as final evidence until the repaired dataset passes the independent temporal audit.

This is the version of the plan consistent with the advisor's latest feedback and with the empirical evidence from A1–A5, under the current temporal-repair reset.

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
- [Done] SWiM-style improved Transformer panel baseline implemented and run under the same frozen panel_common loader, split, normalization, targets, and metrics.
- [Done] SWiM-style 9-configuration nominal-seed-42 benchmark completed and frozen as the structured local-attention control.
- [Next] Build the formal Adaptive Multi-Scale PathFormer panel baseline.
- [Pending] Complete the five-model / family single-seed comparison: Ridge → LSTM → Vanilla Transformer → SWiM-style Transformer → Adaptive Multi-Scale PathFormer.
- [Pending] Multi-seed robustness after architecture comparison and selection.

The primary next action in Phase 2 is now to implement the formal Adaptive Multi-Scale PathFormer panel backbone, using the completed Ridge, LSTM, Vanilla, and SWiM controls as the comparison ladder.

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
- [Done] SWiM implementation audit and architecture redesign completed.
- [Done] SWiM correct shifted-window boundary masking implemented.
- [Done] SWiM MPS execution debugging completed sufficiently to run the formal development benchmark.
- [Done] SWiM 9-configuration nominal-seed-42 benchmark completed.
- [Done] SWiM architecture/training protocol frozen.

Final status line for this phase:

- [Done] Ridge verification gate passed; deep-learning expansion authorized.
- [Done] LSTM single-seed development benchmark passed its stability gate.
- [Done] Vanilla Transformer development benchmark passed its control-benchmark gate.
- [Done] SWiM development benchmark passed the execution gate, with one explicit model-output failure cell: Daily+Weekly / 20d constant collapse.

### Phase 3A — Panel Methodological / Architecture Controls — PAUSED UNTIL DATASET V2 AUDIT PASSES

This is the additional model-control ladder used to attribute whether the observed temporal-frequency patterns are architecture-specific or persist across model families. It is not the advisor's literal Experiment 2.

Completed model families under the old temporal contract:

- [Done — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY] Ridge × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
- [Done — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY] LSTM × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
- [Done — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY] Vanilla Transformer control × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d
- [Done — PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY] SWiM-style improved Transformer A × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d

These results are historic diagnostics only. They must be re-run on the repaired dataset before contributing to final empirical conclusions. They remain useful for debugging and for understanding the old pipeline, but they do not define the current formal benchmark.

Next model family in order after the dataset repair:

1. [Next after V2 pass] Adaptive Multi-Scale PathFormer improved Transformer B × Daily / Weekly / Daily+Weekly × 5d / 10d / 20d

Blocked / deferred:

- [Blocked on data] Hourly only, Half-Day only, Hourly + Daily, All frequencies
  - Requires a panel-wide intraday OHLCV source; the existing panel dataset is only Daily + Weekly under the current repaired data scope.

Current evidence from the completed old control benchmarks is retained only as historical context. It should not be used to claim frequency superiority, weekly help, or Daily+Weekly complementarity until the cleaned V2 dataset passes the independent temporal audit.

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

This is the PathFormer mechanism experiment that follows the panel frequency comparison. **Daily + Weekly is the advisor-specified working configuration** for Experiment 3, but the first strict gate is dataset V2 temporal validity.

- [Pending] Build the Daily+Weekly mechanism ablation ladder after V2 dataset pass.
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

- Dataset V2 repair and audit gate: required before any formal benchmark interpretation
- Ridge stable under the old pipeline: historically completed, but not valid for final conclusions under the repaired data contract
- LSTM stable under the old pipeline: historically completed, but not valid for final conclusions under the repaired data contract
- Vanilla Transformer control under the old pipeline: historically completed, but not valid for final conclusions under the repaired data contract
- SWiM-style improved Transformer A under the old pipeline: historically completed, but not valid for final conclusions under the repaired data contract
- Adaptive Multi-Scale PathFormer improved Transformer B: pending / next, and will only be interpreted after the repaired V2 audit passes
- Formal adaptive-router ablation / interpretation: wait until the repaired dataset and the clean V2 benchmark sequence pass review

The gate is no longer: "Should we start deep models?"

It is now: "After the temporal repair passes the independent audit, and after the clean baseline re-benchmark is complete, will the PathFormer development run provide valid evidence for the formal adaptive multi-scale / router mechanism study?"

---

## Historical FSLR Reporting Actions (Paused)

1. Audit the repository to determine whether a plain FSLR LSTM baseline already exists. Do not claim it exists unless verified. Status: "FSLR plain LSTM baseline: implementation status to be audited / likely missing."
2. Complete the advisor-requested FSLR core model-comparison set: Linear + LSTM + SWiM-style + PathFormer.
3. Retain Vanilla Transformer as an optional supplementary conventional Transformer reference, not one of the two advisor-requested improved Transformer slots.
4. Preserve the A1–A5 diagnostic narrative and the concat/gated historical findings separately from the core model-comparison table, and write report text with explicit advisor checklist mapping and final protocol statement.

---

## Immediate Execution Checklist (Most Urgent)

1. [Current priority] Review the actual Git diff for the temporal repair:
   - `git --no-pager diff -- scripts/python/panel_build_multiscale_dataset.py scripts/python/panel_common.py scripts/python/panel_verify_temporal_integrity.py`
2. [Current priority] Correct any implementation / audit issues found in the diff review.
3. [Current priority] Back up old dataset manifests and model outputs.
4. [Current priority] Rebuild Dataset Contract V2.
5. [Current priority] Run the full independent temporal-integrity audit.
6. [Pending] Do not proceed unless the temporal audit passes.
7. [Pending] Audit rebuilt sample counts and target distributions for 5d / 10d / 20d.
8. [Pending] Run zero and train-mean baselines.
9. [Pending] Re-run the formal Linear / Ridge baseline on the repaired dataset.
10. [Pending] Re-run the panel benchmark controls required for final paper claims: LSTM, Vanilla Transformer, and SWiM on the same repaired dataset.
11. [Pending] Re-run Adaptive PathFormer main frequency comparison: Daily, Weekly, Daily + Weekly across 5d / 10d / 20d.
12. [Pending] Run the Daily+Weekly mechanism ablation: single / fixed / static / adaptive.
13. [Pending] Run 5-seed robustness on the selected model family.
14. [Pending] Produce router / regime interpretation analysis.
15. [Pending] Finalize the panel-results writeup and the FSLR diagnostic section.

Current control ladder for historical documentation:

Naive → Ridge → LSTM → Vanilla Transformer → SWiM-style Transformer → Adaptive Multi-Scale PathFormer

with status:

- Naive: DONE / historical reference only
- Ridge: DONE / PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY
- LSTM: DONE / PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY
- Vanilla Transformer: DONE / PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY
- SWiM-style panel Transformer: DONE / PRE-TEMPORAL-FIX / DIAGNOSTIC ONLY
- Adaptive Multi-Scale PathFormer: implementation frozen; formal benchmark pending V2 repair
- Mechanism ablation: PENDING after dataset V2 audit
- Formal robustness: PENDING after repaired-data benchmark
- Interpretability: PENDING after stable model selection

---

## Model-Capacity Caveat

Approximate Adaptive PathFormer parameter counts under the current design are:

- `daily_only`: ~56k
- `weekly_only`: ~56k
- `daily_weekly`: ~116k

Therefore, if Daily+Weekly outperforms a single-frequency model, this alone does not prove cross-frequency information complementarity. The larger dual-frequency model also has greater capacity. Later analysis should either include a matched-capacity control if feasible, or explicitly acknowledge the parameter-capacity confound.

---

## Do Not Do / Current Priority

Until Dataset Contract V2 passes the independent temporal audit:

DO NOT:

- tune PathFormer hyperparameters
- rerun long formal models
- interpret old panel metrics as final results
- claim adaptive routing works
- claim Weekly helps forecasting
- claim Daily+Weekly complementarity
- change the target formula
- add new features
- change normalization
- modify horizons
- change dependencies unless required
- overwrite old audit evidence

Historical Next Action at Earlier Temporal-Code-Review Stage:

```bash
git --no-pager diff -- \
  scripts/python/panel_build_multiscale_dataset.py \
  scripts/python/panel_common.py \
  scripts/python/panel_verify_temporal_integrity.py
```

This was an earlier review step for the temporal-code diff, not the current authoritative next action.

Current authoritative next action:

- modify the Dataset V2 builder so Weekly is deterministically aggregated from canonical Daily;
- modify / extend the independent audit to validate the final Weekly-from-Daily contract;
- review the code and rebuild final V2;
- require OVERALL: PASS before any model reruns.

---

## Advisor / Experiment Design Context (Retained)

The existing advisor-directed experimental framework remains valid in spirit, but the temporal repair changes the validity of the dataset/results, not the fundamental research question.

- Panel is the main empirical framework.
- PathFormer addresses multi-scale structure within frequency.
- Late fusion addresses Daily/Weekly information complementarity.
- The router is within-frequency, not a global Daily-vs-Weekly router.
- Experiment 3 compares:
  - Single
  - Fixed multi-scale
  - Static learned weights
  - Adaptive router
- Formal robustness and interpretation come later.

This reset does not erase the research question; it restores the validity conditions required to answer it.

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

