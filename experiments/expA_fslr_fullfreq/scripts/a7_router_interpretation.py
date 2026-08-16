"""
A7: Router-weight interpretability analysis.

Loads the per-test-sample router gate-weight CSVs saved by A4
(a4_router_weights_<freq>_<horizon>.csv, combine_mode="adaptive_router"),
computes two market-regime labels directly from the FSLR daily window
(X_daily, which is horizon-independent) for every sample:

  - volatility_regime: rolling std of daily log-returns within the window,
    "high"/"low" split at the median (computed once, across ALL samples).
  - trend_regime: R^2 of a linear fit of the daily close price over the
    window's time index; "trending" if R^2 >= --trend_r2_threshold (default
    0.3), else "range_bound".

Produces, per frequency, a long-format breakdown of mean router gate weight
by horizon x volatility_regime x trend_regime x patch_size, and a combined
CSV across all frequencies.

This is an ANALYSIS-ONLY script (no training); it requires A4 to have been
run first (with combine_mode="adaptive_router") for the desired frequencies.

Usage:
    python a7_router_interpretation.py [--horizons 5d,10d,20d] [--freqs hourly,daily]
"""

import argparse

import numpy as np
import pandas as pd

import common as C


def compute_regime_labels(trend_r2_threshold: float) -> pd.DataFrame:
    """Regime labels computed once from the (horizon-independent) X_daily
    window + meta.anchor_date. Uses the "5d" horizon file purely to access
    the shared X_daily/meta arrays (identical across horizons)."""
    data = C.load_fslr_multiscale("5d")
    X_daily = data["daily"]
    meta = data["meta"]

    n = X_daily.shape[0]
    vol = np.zeros(n, dtype=np.float64)
    r2 = np.zeros(n, dtype=np.float64)

    t = np.arange(X_daily.shape[1], dtype=np.float64)
    for i in range(n):
        close = X_daily[i, :, C.CLOSE_IDX].astype(np.float64)
        close = np.clip(close, 1e-6, None)
        log_ret = np.diff(np.log(close))
        vol[i] = float(np.std(log_ret))

        slope, intercept = np.polyfit(t, close, 1)
        pred = slope * t + intercept
        ss_res = float(np.sum((close - pred) ** 2))
        ss_tot = float(np.sum((close - close.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    vol_median = float(np.median(vol))
    volatility_regime = np.where(vol >= vol_median, "high", "low")
    trend_regime = np.where(r2 >= trend_r2_threshold, "trending", "range_bound")

    return pd.DataFrame(
        {
            "anchor_date": meta["anchor_date"].astype(str).values,
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="5d,10d,20d")
    parser.add_argument("--freqs", type=str, default="hourly,halfday,daily,weekly")
    parser.add_argument("--trend_r2_threshold", type=float, default=0.3)
    args = parser.parse_args()

    horizons = args.horizons.split(",")
    freqs = args.freqs.split(",")

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = C.resolve_output_path("a7_router_interpretation_progress.log")

    regimes = compute_regime_labels(args.trend_r2_threshold)
    C.log_progress(
        f"[a7] computed regime labels for {len(regimes)} samples "
        f"(volatility median-split, trend R^2 threshold={args.trend_r2_threshold})",
        log_file,
    )

    all_rows = []
    for freq in freqs:
        patch_sizes = C.PATCH_CANDIDATES[freq]
        gate_cols = [f"gate_patch_{p}" for p in patch_sizes]

        freq_rows = []
        for horizon in horizons:
            csv_path = C.resolve_output_path(f"a4_router_weights_{freq}_{horizon}.csv")
            if not csv_path.exists():
                C.log_progress(
                    f"[a7] WARNING: missing {csv_path.name} (run a4 with combine_mode="
                    f"adaptive_router for freq={freq} horizon={horizon} first), skipping",
                    log_file,
                )
                continue

            gates_df = pd.read_csv(csv_path)
            gates_df["anchor_date"] = gates_df["anchor_date"].astype(str)
            merged = gates_df.merge(regimes, on="anchor_date", how="inner")
            if merged.empty:
                C.log_progress(f"[a7] WARNING: no anchor_date overlap for freq={freq} horizon={horizon}", log_file)
                continue

            grouped = merged.groupby(["volatility_regime", "trend_regime"])[gate_cols].agg(["mean", "count"])
            for (vol_regime, trend_regime), row in grouped.iterrows():
                for p, col in zip(patch_sizes, gate_cols):
                    freq_rows.append(
                        {
                            "freq": freq,
                            "horizon": horizon,
                            "volatility_regime": vol_regime,
                            "trend_regime": trend_regime,
                            "patch_size": p,
                            "mean_gate": float(row[(col, "mean")]),
                            "n_samples": int(row[(col, "count")]),
                        }
                    )

        if freq_rows:
            freq_df = pd.DataFrame(freq_rows)
            freq_out = C.resolve_output_path(f"a7_router_interpretation_{freq}.csv")
            freq_df.to_csv(freq_out, index=False)
            C.log_progress(f"[a7] wrote {freq_out}", log_file)
            all_rows.extend(freq_rows)

    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined_out = C.resolve_output_path("a7_router_interpretation_all.csv")
        combined.to_csv(combined_out, index=False)
        C.log_progress(f"[a7] ALL DONE. Combined table at {combined_out}", log_file)
    else:
        C.log_progress("[a7] No A4 router-weight CSVs found for any requested freq/horizon; nothing written.", log_file)


if __name__ == "__main__":
    main()
