"""
Task (Advisor Pivot) - Build Panel Multi-Scale Dataset (Daily + Weekly only)

Generalizes the single-ticker `build_multiscale_dataset.py` logic to the green
energy panel (see panel_universe.py). Only Daily + Weekly frequencies are used
for the panel (Hourly/Half-Day remain FSLR-only, see project_progress.md
"Advisor Pivot" section for the data-availability rationale).

For each ticker with raw Daily/Weekly CSVs in dataset/finance/panel_raw/, this
builds daily-anchor windows:
  X_daily  : (N, daily_window, 5)
  X_weekly : (N, weekly_window, 5)
  y_5d, y_10d, y_20d : (N,) future log-return targets
  meta.csv : anchor_date, target_date_5d/10d/20d

Output per ticker: dataset/multiscale_dataset/panel/<TICKER>/
Also writes a manifest: dataset/multiscale_dataset/panel/panel_build_manifest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.python.panel_universe import GREEN_ENERGY_UNIVERSE

FEATURES = ["open", "high", "low", "close", "volume"]
H_DAILY = 90
H_WEEKLY = 26
TARGET_HORIZONS = [5, 10, 20]
MAX_HORIZON = max(TARGET_HORIZONS)

RAW_DIR = ROOT / "dataset" / "finance" / "panel_raw"
OUT_DIR = ROOT / "dataset" / "multiscale_dataset" / "panel"

# Minimum anchors required to keep a ticker in the panel (needs a usable
# 70/15/15 time-ordered split with the requested lookback windows).
MIN_SAMPLES = 200


def build_one(ticker: str) -> dict:
    record = {"ticker": ticker, "n_samples": 0, "status": "ok", "note": ""}

    daily_path = RAW_DIR / f"{ticker}_daily.csv"
    weekly_path = RAW_DIR / f"{ticker}_weekly.csv"
    if not daily_path.exists() or not weekly_path.exists():
        record["status"] = "missing_raw"
        return record

    daily = pd.read_csv(daily_path)
    weekly = pd.read_csv(weekly_path)
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    weekly["datetime"] = pd.to_datetime(weekly["datetime"])
    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = weekly.sort_values("datetime").reset_index(drop=True)

    X_daily, X_weekly = [], []
    y_by_horizon = {h: [] for h in TARGET_HORIZONS}
    meta_records = []

    for i in range(H_DAILY, len(daily) - MAX_HORIZON):
        anchor_dt = pd.Timestamp(daily.iloc[i]["datetime"])

        daily_window = daily.iloc[i - H_DAILY:i][FEATURES]
        if len(daily_window) < H_DAILY:
            continue

        weekly_subset = weekly[weekly["datetime"] <= anchor_dt]
        if len(weekly_subset) < H_WEEKLY:
            continue
        weekly_window = weekly_subset.tail(H_WEEKLY)[FEATURES]

        p0 = float(daily.iloc[i]["close"])
        targets = {}
        for h in TARGET_HORIZONS:
            p1 = float(daily.iloc[i + h]["close"])
            targets[h] = float(np.log(p1 / p0))

        X_daily.append(daily_window.values)
        X_weekly.append(weekly_window.values)
        for h in TARGET_HORIZONS:
            y_by_horizon[h].append(targets[h])

        meta_records.append(
            {
                "anchor_date": anchor_dt,
                "target_date_5d": daily.iloc[i + 5]["datetime"],
                "target_date_10d": daily.iloc[i + 10]["datetime"],
                "target_date_20d": daily.iloc[i + 20]["datetime"],
            }
        )

    n = len(X_daily)
    record["n_samples"] = n
    if n < MIN_SAMPLES:
        record["status"] = "too_few_samples"
        record["note"] = f"only {n} samples (< {MIN_SAMPLES})"
        return record

    X_daily = np.asarray(X_daily, dtype=np.float32)
    X_weekly = np.asarray(X_weekly, dtype=np.float32)
    y_5d = np.asarray(y_by_horizon[5], dtype=np.float32)
    y_10d = np.asarray(y_by_horizon[10], dtype=np.float32)
    y_20d = np.asarray(y_by_horizon[20], dtype=np.float32)
    meta = pd.DataFrame(meta_records)

    if np.isnan(X_daily).any() or np.isnan(X_weekly).any() or np.isnan(y_20d).any():
        record["status"] = "nan_found"
        record["note"] = "NaN detected in features or targets, skipped save"
        return record

    ticker_dir = OUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    np.save(ticker_dir / "X_daily.npy", X_daily)
    np.save(ticker_dir / "X_weekly.npy", X_weekly)
    np.save(ticker_dir / "y_5d.npy", y_5d)
    np.save(ticker_dir / "y_10d.npy", y_10d)
    np.save(ticker_dir / "y_20d.npy", y_20d)
    np.save(ticker_dir / "y.npy", y_20d)  # backward-compat alias
    meta.to_csv(ticker_dir / "meta.csv", index=False)

    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, ticker in enumerate(GREEN_ENERGY_UNIVERSE, start=1):
        record = build_one(ticker)
        manifest.append(record)
        print(f"[{i}/{len(GREEN_ENERGY_UNIVERSE)}] {ticker}: {record}")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUT_DIR / "panel_build_manifest.csv", index=False)

    print("\nBuild Summary")
    print("-" * 60)
    print(manifest_df.to_string(index=False))
    n_ok = (manifest_df["status"] == "ok").sum()
    print(f"\nok={n_ok} total={len(manifest_df)}")


if __name__ == "__main__":
    main()
