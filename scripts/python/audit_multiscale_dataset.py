from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset" / "multiscale_dataset"
AUDIT_DIR = BASE_DIR / "dataset" / "audit"
REPORT_PATH = AUDIT_DIR / "task6_dataset_audit.txt"

FEATURE_DIM = 5


# =====================================================
# LOAD
# =====================================================

X_hourly = np.load(DATASET_DIR / "X_hourly.npy")
X_halfday = np.load(DATASET_DIR / "X_halfday.npy")
X_daily = np.load(DATASET_DIR / "X_daily.npy")
X_weekly = np.load(DATASET_DIR / "X_weekly.npy")

y_5d = np.load(DATASET_DIR / "y_5d.npy")
y_10d = np.load(DATASET_DIR / "y_10d.npy")
y_20d = np.load(DATASET_DIR / "y_20d.npy")
y_compat = np.load(DATASET_DIR / "y.npy")

meta = pd.read_csv(DATASET_DIR / "meta.csv", parse_dates=["anchor_date", "target_date_5d", "target_date_10d", "target_date_20d"])


# =====================================================
# HELPERS
# =====================================================

def stats_line(arr: np.ndarray, name: str) -> list[str]:
    return [
        f"{name} mean = {arr.mean():.8f}",
        f"{name} std  = {arr.std():.8f}",
        f"{name} min  = {arr.min():.8f}",
        f"{name} max  = {arr.max():.8f}",
    ]


def nan_count(arr: np.ndarray) -> int:
    return int(np.isnan(arr).sum())


# =====================================================
# AUDIT
# =====================================================

lines = []
lines.append("=" * 72)
lines.append("TASK 6 - MULTISCALE DATASET AUDIT")
lines.append("=" * 72)

lines.append("\nShapes")
lines.append(f"X_hourly : {X_hourly.shape}")
lines.append(f"X_halfday: {X_halfday.shape}")
lines.append(f"X_daily  : {X_daily.shape}")
lines.append(f"X_weekly : {X_weekly.shape}")
lines.append(f"y_5d     : {y_5d.shape}")
lines.append(f"y_10d    : {y_10d.shape}")
lines.append(f"y_20d    : {y_20d.shape}")
lines.append(f"y (compat): {y_compat.shape}")
lines.append(f"meta     : {meta.shape}")

lines.append("\nSample Count Consistency")
n = X_hourly.shape[0]
same_count = (
    X_halfday.shape[0] == n
    and X_daily.shape[0] == n
    and X_weekly.shape[0] == n
    and y_5d.shape[0] == n
    and y_10d.shape[0] == n
    and y_20d.shape[0] == n
    and y_compat.shape[0] == n
    and len(meta) == n
)
lines.append(f"all arrays aligned: {same_count}")
lines.append(f"sample count: {n}")

lines.append("\nFeature Dimension Check")
feature_ok = (
    X_hourly.shape[2] == FEATURE_DIM
    and X_halfday.shape[2] == FEATURE_DIM
    and X_daily.shape[2] == FEATURE_DIM
    and X_weekly.shape[2] == FEATURE_DIM
)
lines.append(f"expected feature dim = {FEATURE_DIM}")
lines.append(f"feature dim valid: {feature_ok}")

lines.append("\nNaN Check")
lines.append(f"X_hourly  NaN: {nan_count(X_hourly)}")
lines.append(f"X_halfday NaN: {nan_count(X_halfday)}")
lines.append(f"X_daily   NaN: {nan_count(X_daily)}")
lines.append(f"X_weekly  NaN: {nan_count(X_weekly)}")
lines.append(f"y_5d      NaN: {nan_count(y_5d)}")
lines.append(f"y_10d     NaN: {nan_count(y_10d)}")
lines.append(f"y_20d     NaN: {nan_count(y_20d)}")
lines.append(f"meta NaN rows: {int(meta.isna().any(axis=1).sum())}")

lines.append("\nTarget Statistics")
lines.extend(stats_line(y_5d, "y_5d"))
lines.append("")
lines.extend(stats_line(y_10d, "y_10d"))
lines.append("")
lines.extend(stats_line(y_20d, "y_20d"))

lines.append("\nDate Range")
lines.append(f"anchor start: {meta['anchor_date'].min()}")
lines.append(f"anchor end  : {meta['anchor_date'].max()}")
lines.append(f"target 5d end : {meta['target_date_5d'].max()}")
lines.append(f"target 10d end: {meta['target_date_10d'].max()}")
lines.append(f"target 20d end: {meta['target_date_20d'].max()}")

lines.append("\nTemporal Validity")
mono_anchor = bool(meta["anchor_date"].is_monotonic_increasing)
lines.append(f"anchor date monotonic increasing: {mono_anchor}")

leak_5 = int((meta["target_date_5d"] <= meta["anchor_date"]).sum())
leak_10 = int((meta["target_date_10d"] <= meta["anchor_date"]).sum())
leak_20 = int((meta["target_date_20d"] <= meta["anchor_date"]).sum())
lines.append(f"5d leakage rows  (target<=anchor): {leak_5}")
lines.append(f"10d leakage rows (target<=anchor): {leak_10}")
lines.append(f"20d leakage rows (target<=anchor): {leak_20}")

compat_same = bool(np.array_equal(y_compat, y_20d))
lines.append(f"compat y equals y_20d: {compat_same}")

audit_pass = (
    same_count
    and feature_ok
    and nan_count(X_hourly) == 0
    and nan_count(X_halfday) == 0
    and nan_count(X_daily) == 0
    and nan_count(X_weekly) == 0
    and nan_count(y_5d) == 0
    and nan_count(y_10d) == 0
    and nan_count(y_20d) == 0
    and leak_5 == 0
    and leak_10 == 0
    and leak_20 == 0
)

lines.append("\nOverall")
lines.append(f"audit_pass: {audit_pass}")

text = "\n".join(lines)
print(text)

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(text + "\n", encoding="utf-8")
print(f"\nSaved report -> {REPORT_PATH}")