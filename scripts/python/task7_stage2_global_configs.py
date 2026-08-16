"""
Task 7 (Stage 2) - Global lookback configuration comparison.

Compare three full window configurations under the same protocol:
- Short:   hourly 24, halfday 20, daily 60,  weekly 26
- W*:      hourly 24, halfday 20, daily 90,  weekly 26
- Long:    hourly 96, halfday 60, daily 120, weekly 78

Model:
- Linear Regression baseline on flattened multi-frequency features.

Target:
- 20D log return (Daily anchor)

Outputs:
- dataset/audit/task7_stage2_results.csv
- dataset/audit/task7_stage2_summary.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset"
AUDIT_DIR = DATA_DIR / "audit"

FEATURES = ["open", "high", "low", "close", "volume"]
TARGET_HORIZON = 20

SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

CONFIGS = {
    "short": {"hourly": 24, "halfday": 20, "daily": 60, "weekly": 26},
    "w_star": {"hourly": 24, "halfday": 20, "daily": 90, "weekly": 26},
    "long": {"hourly": 96, "halfday": 60, "daily": 120, "weekly": 78},
}


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

    for i in range(h_daily, len(daily) - TARGET_HORIZON):
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

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), pd.Series(anchors, name="anchor_date")


def eval_config(X: np.ndarray, y: np.ndarray) -> dict:
    n = len(X)
    train_end = int(n * SPLIT["train"])
    val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    return {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "val_mse": float(mean_squared_error(y_val, pred_val)),
        "val_mae": float(mean_absolute_error(y_val, pred_val)),
        "val_corr": float(np.corrcoef(pred_val, y_val)[0, 1]) if len(y_val) > 1 else np.nan,
        "test_mse": float(mean_squared_error(y_test, pred_test)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_corr": float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else np.nan,
        "test_direction_acc": float((np.sign(pred_test) == np.sign(y_test)).mean()) if len(y_test) else np.nan,
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_data()

    rows = []

    for name, windows in CONFIGS.items():
        X, y, anchors = build_dataset(frames, windows)
        if len(X) < 300:
            raise RuntimeError(f"Not enough samples for config {name}: {len(X)}")

        metrics = eval_config(X, y)
        row = {
            "config": name,
            "windows": str(windows),
            "samples": int(len(X)),
            "anchor_start": str(anchors.iloc[0]),
            "anchor_end": str(anchors.iloc[-1]),
            **metrics,
        }
        rows.append(row)

        print(
            f"[{name}] samples={len(X)} "
            f"val_mse={metrics['val_mse']:.8f} test_mse={metrics['test_mse']:.8f} "
            f"test_da={metrics['test_direction_acc']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("val_mse")
    out_csv = AUDIT_DIR / "task7_stage2_results.csv"
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]

    lines = []
    lines.append("TASK 7 STAGE 2 - GLOBAL CONFIG COMPARISON")
    lines.append("=" * 72)
    lines.append("Configs:")
    for name, windows in CONFIGS.items():
        lines.append(f"- {name}: {windows}")
    lines.append("")

    lines.append("Results (sorted by validation MSE):")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['config']}: val_mse={r['val_mse']:.8f}, test_mse={r['test_mse']:.8f}, "
            f"test_mae={r['test_mae']:.6f}, test_corr={r['test_corr']:.4f}, "
            f"test_da={r['test_direction_acc']:.4f}, samples={int(r['samples'])}"
        )

    lines.append("")
    lines.append(
        f"Recommended config: {best['config']} with windows {best['windows']} "
        f"(val_mse={best['val_mse']:.8f})"
    )

    out_txt = AUDIT_DIR / "task7_stage2_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":
    main()
