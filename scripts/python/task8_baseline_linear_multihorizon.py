"""
Task 8 (Part A) - Linear Regression Baseline on Multi-Frequency Dataset

Run baseline experiments on 5D / 10D / 20D targets using the same dataset,
same split, and same features.

Metrics:
- MSE
- MAE
- Direction Accuracy
- Correlation

Data:
- dataset/multiscale_dataset/X_hourly.npy
- dataset/multiscale_dataset/X_halfday.npy
- dataset/multiscale_dataset/X_daily.npy
- dataset/multiscale_dataset/X_weekly.npy
- dataset/multiscale_dataset/y_5d.npy
- dataset/multiscale_dataset/y_10d.npy
- dataset/multiscale_dataset/y_20d.npy
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset" / "multiscale_dataset"
AUDIT_DIR = BASE_DIR / "dataset" / "audit"

SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def load_X() -> np.ndarray:
    X_hourly = np.load(DATASET_DIR / "X_hourly.npy")
    X_halfday = np.load(DATASET_DIR / "X_halfday.npy")
    X_daily = np.load(DATASET_DIR / "X_daily.npy")
    X_weekly = np.load(DATASET_DIR / "X_weekly.npy")

    n = X_hourly.shape[0]
    X = np.concatenate(
        [
            X_hourly.reshape(n, -1),
            X_halfday.reshape(n, -1),
            X_daily.reshape(n, -1),
            X_weekly.reshape(n, -1),
        ],
        axis=1,
    )
    return X


def split_indices(n: int) -> tuple[slice, slice, slice]:
    train_end = int(n * SPLIT["train"])
    val_end = int(n * (SPLIT["train"] + SPLIT["val"]))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def evaluate_horizon(X: np.ndarray, y: np.ndarray, horizon_name: str) -> dict:
    s_train, s_val, s_test = split_indices(len(X))

    X_train, y_train = X[s_train], y[s_train]
    X_val, y_val = X[s_val], y[s_val]
    X_test, y_test = X[s_test], y[s_test]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    val_mse = float(mean_squared_error(y_val, pred_val))
    val_mae = float(mean_absolute_error(y_val, pred_val))
    val_corr = float(np.corrcoef(pred_val, y_val)[0, 1]) if len(y_val) > 1 else np.nan
    val_da = float((np.sign(pred_val) == np.sign(y_val)).mean())

    test_mse = float(mean_squared_error(y_test, pred_test))
    test_mae = float(mean_absolute_error(y_test, pred_test))
    test_corr = float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else np.nan
    test_da = float((np.sign(pred_test) == np.sign(y_test)).mean())

    return {
        "horizon": horizon_name,
        "samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "val_mse": val_mse,
        "val_mae": val_mae,
        "val_corr": val_corr,
        "val_direction_acc": val_da,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_corr": test_corr,
        "test_direction_acc": test_da,
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    X = load_X()
    targets = {
        "5d": np.load(DATASET_DIR / "y_5d.npy"),
        "10d": np.load(DATASET_DIR / "y_10d.npy"),
        "20d": np.load(DATASET_DIR / "y_20d.npy"),
    }

    rows = []
    for name, y in targets.items():
        row = evaluate_horizon(X, y, name)
        rows.append(row)
        print(
            f"[{name}] test_mse={row['test_mse']:.8f} "
            f"test_mae={row['test_mae']:.6f} "
            f"test_corr={row['test_corr']:.4f} "
            f"test_da={row['test_direction_acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    out_csv = AUDIT_DIR / "task8_linear_baseline_results.csv"
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("TASK 8 - LINEAR BASELINE (MULTI-HORIZON)")
    lines.append("=" * 72)
    lines.append("Dataset: multiscale_dataset (w_star windows)")
    lines.append("Split: train=70%, val=15%, test=15% (time ordered)")
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['horizon']}: test_mse={r['test_mse']:.8f}, test_mae={r['test_mae']:.6f}, "
            f"test_corr={r['test_corr']:.4f}, test_da={r['test_direction_acc']:.4f}"
        )

    out_txt = AUDIT_DIR / "task8_linear_baseline_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":
    main()
