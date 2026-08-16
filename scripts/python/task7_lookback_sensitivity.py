"""
Task 7 (Stage 1) - Lookback Sensitivity (one frequency at a time)

Protocol:
- Daily anchor, 20D log-return target.
- Fix 3 frequencies at default lookbacks and vary 1 frequency.
- Use a simple linear regression baseline on flattened multi-frequency features.
- Evaluate on validation split only (time-ordered split).

Outputs:
- dataset/audit/task7_stage1_results.csv
- dataset/audit/task7_stage1_summary.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset"
AUDIT_DIR = DATA_DIR / "audit"

FEATURES = ["open", "high", "low", "close", "volume"]
TARGET_HORIZON = 20

DEFAULT_WINDOWS = {
    "hourly": 48,
    "halfday": 40,
    "daily": 90,
    "weekly": 52,
}

CANDIDATES = {
    "hourly": [24, 48, 96],
    "halfday": [20, 40, 60],
    "daily": [60, 90, 120],
    "weekly": [26, 52, 78],
}

SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


# =====================================================
# LOAD DATA
# =====================================================

def load_data() -> Dict[str, pd.DataFrame]:
    hourly = pd.read_csv(DATA_DIR / "processed" / "FSLR_hourly.csv")
    halfday = pd.read_csv(DATA_DIR / "merged" / "FSLR_halfday_full.csv")
    daily = pd.read_csv(DATA_DIR / "processed" / "FSLR_daily.csv")
    weekly = pd.read_csv(DATA_DIR / "processed" / "FSLR_weekly.csv")

    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    halfday["datetime"] = pd.to_datetime(halfday["datetime"])
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    weekly["datetime"] = pd.to_datetime(weekly["datetime"])

    return {
        "hourly": hourly.sort_values("datetime").reset_index(drop=True),
        "halfday": halfday.sort_values("datetime").reset_index(drop=True),
        "daily": daily.sort_values("datetime").reset_index(drop=True),
        "weekly": weekly.sort_values("datetime").reset_index(drop=True),
    }


# =====================================================
# DATASET BUILD
# =====================================================

def build_dataset(frames: Dict[str, pd.DataFrame], windows: Dict[str, int]) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    hourly = frames["hourly"]
    halfday = frames["halfday"]
    daily = frames["daily"]
    weekly = frames["weekly"]

    h_hourly = windows["hourly"]
    h_halfday = windows["halfday"]
    h_daily = windows["daily"]
    h_weekly = windows["weekly"]

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    anchors: List[pd.Timestamp] = []

    start_idx = h_daily
    end_idx = len(daily) - TARGET_HORIZON

    for i in range(start_idx, end_idx):
        anchor_dt = pd.Timestamp(daily.loc[i, "datetime"])

        daily_window = daily.iloc[i - h_daily:i][FEATURES]
        if len(daily_window) < h_daily:
            continue

        weekly_subset = weekly[weekly["datetime"] <= anchor_dt]
        if len(weekly_subset) < h_weekly:
            continue
        weekly_window = weekly_subset.tail(h_weekly)[FEATURES]

        halfday_subset = halfday[halfday["datetime"] <= anchor_dt]
        if len(halfday_subset) < h_halfday:
            continue
        halfday_window = halfday_subset.tail(h_halfday)[FEATURES]

        hourly_subset = hourly[hourly["datetime"] <= anchor_dt]
        if len(hourly_subset) < h_hourly:
            continue
        hourly_window = hourly_subset.tail(h_hourly)[FEATURES]

        p0 = float(daily.loc[i, "close"])
        p1 = float(daily.loc[i + TARGET_HORIZON, "close"])
        target = float(np.log(p1 / p0))

        # Flatten by frequency then concatenate.
        x = np.concatenate(
            [
                hourly_window.to_numpy().reshape(-1),
                halfday_window.to_numpy().reshape(-1),
                daily_window.to_numpy().reshape(-1),
                weekly_window.to_numpy().reshape(-1),
            ]
        )

        X_rows.append(x)
        y_rows.append(target)
        anchors.append(anchor_dt)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    anchor_series = pd.Series(anchors, name="anchor_date")
    return X, y, anchor_series


# =====================================================
# EVAL
# =====================================================

def evaluate_linear(X: np.ndarray, y: np.ndarray) -> dict:
    n = len(X)
    train_end = int(n * SPLIT["train"])
    val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]

    model = LinearRegression()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    mse = float(mean_squared_error(y_val, val_pred))
    mae = float(mean_absolute_error(y_val, val_pred))
    corr = float(np.corrcoef(val_pred, y_val)[0, 1]) if len(y_val) > 1 else np.nan

    return {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "val_mse": mse,
        "val_mae": mae,
        "val_corr": corr,
    }


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_data()

    results = []

    for freq, values in CANDIDATES.items():
        for v in values:
            windows = dict(DEFAULT_WINDOWS)
            windows[freq] = v

            X, y, anchors = build_dataset(frames, windows)
            if len(X) < 200:
                results.append(
                    {
                        "varied_frequency": freq,
                        "value": v,
                        "windows": str(windows),
                        "samples": int(len(X)),
                        "train_samples": 0,
                        "val_samples": 0,
                        "val_mse": np.nan,
                        "val_mae": np.nan,
                        "val_corr": np.nan,
                        "anchor_start": None,
                        "anchor_end": None,
                    }
                )
                continue

            metrics = evaluate_linear(X, y)
            results.append(
                {
                    "varied_frequency": freq,
                    "value": v,
                    "windows": str(windows),
                    "samples": int(len(X)),
                    "train_samples": metrics["train_samples"],
                    "val_samples": metrics["val_samples"],
                    "val_mse": metrics["val_mse"],
                    "val_mae": metrics["val_mae"],
                    "val_corr": metrics["val_corr"],
                    "anchor_start": str(anchors.iloc[0]) if len(anchors) else None,
                    "anchor_end": str(anchors.iloc[-1]) if len(anchors) else None,
                }
            )

            print(
                f"[{freq}={v}] samples={len(X)} "
                f"val_mse={metrics['val_mse']:.8f} val_mae={metrics['val_mae']:.6f}"
            )

    df = pd.DataFrame(results).sort_values(["varied_frequency", "val_mse"], na_position="last")
    out_csv = AUDIT_DIR / "task7_stage1_results.csv"
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("TASK 7 STAGE 1 - LOOKBACK SENSITIVITY SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Default windows: {DEFAULT_WINDOWS}")
    lines.append("")

    for freq in CANDIDATES.keys():
        sub = df[df["varied_frequency"] == freq].sort_values("val_mse", na_position="last")
        best = sub.iloc[0]
        lines.append(
            f"Best {freq:<7}: {int(best['value'])}  "
            f"val_mse={best['val_mse']:.8f}  samples={int(best['samples'])}"
        )

    lines.append("")
    lines.append("Top results by frequency:")
    for freq in CANDIDATES.keys():
        sub = df[df["varied_frequency"] == freq].sort_values("val_mse", na_position="last")
        for _, row in sub.iterrows():
            lines.append(
                f"- {freq}={int(row['value'])}: mse={row['val_mse']:.8f}, "
                f"mae={row['val_mae']:.6f}, samples={int(row['samples'])}"
            )

    out_txt = AUDIT_DIR / "task7_stage1_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":
    main()
