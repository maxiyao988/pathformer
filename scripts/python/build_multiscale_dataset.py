import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from config import CONFIG


# =====================================================
# CONFIG
# =====================================================

FEATURES = CONFIG["features"]
H_HOURLY = CONFIG["hourly_window"]
H_HALFDAY = CONFIG["halfday_window"]
H_DAILY = CONFIG["daily_window"]
H_WEEKLY = CONFIG["weekly_window"]
TARGET_HORIZONS = [5, 10, 20]
MAX_HORIZON = max(TARGET_HORIZONS)


# =====================================================
# LOAD DATA
# =====================================================

hourly = pd.read_csv(CONFIG["hourly_path"])
halfday = pd.read_csv(CONFIG["halfday_path"])
daily = pd.read_csv(CONFIG["daily_path"])
weekly = pd.read_csv(CONFIG["weekly_path"])

hourly["datetime"] = pd.to_datetime(hourly["datetime"])
halfday["datetime"] = pd.to_datetime(halfday["datetime"])
daily["datetime"] = pd.to_datetime(daily["datetime"])
weekly["datetime"] = pd.to_datetime(weekly["datetime"])

hourly = hourly.sort_values("datetime").reset_index(drop=True)
halfday = halfday.sort_values("datetime").reset_index(drop=True)
daily = daily.sort_values("datetime").reset_index(drop=True)
weekly = weekly.sort_values("datetime").reset_index(drop=True)


# =====================================================
# STORAGE
# =====================================================

X_hourly = []
X_halfday = []
X_daily = []
X_weekly = []

y_by_horizon = {h: [] for h in TARGET_HORIZONS}
meta_records = []


# =====================================================
# DAILY ANCHOR LOOP
# =====================================================

for i in range(H_DAILY, len(daily) - MAX_HORIZON):
    anchor_dt = pd.Timestamp(daily.iloc[i]["datetime"])

    daily_window = daily.iloc[i - H_DAILY:i][FEATURES]
    if len(daily_window) < H_DAILY:
        continue

    weekly_subset = weekly[weekly["datetime"] <= anchor_dt]
    if len(weekly_subset) < H_WEEKLY:
        continue
    weekly_window = weekly_subset.tail(H_WEEKLY)[FEATURES]

    halfday_subset = halfday[halfday["datetime"] <= anchor_dt]
    if len(halfday_subset) < H_HALFDAY:
        continue
    halfday_window = halfday_subset.tail(H_HALFDAY)[FEATURES]

    hourly_subset = hourly[hourly["datetime"] <= anchor_dt]
    if len(hourly_subset) < H_HOURLY:
        continue
    hourly_window = hourly_subset.tail(H_HOURLY)[FEATURES]

    p0 = float(daily.iloc[i]["close"])
    targets = {}
    for h in TARGET_HORIZONS:
        p1 = float(daily.iloc[i + h]["close"])
        targets[h] = float(np.log(p1 / p0))

    X_hourly.append(hourly_window.values)
    X_halfday.append(halfday_window.values)
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


# =====================================================
# NUMPY
# =====================================================

X_hourly = np.asarray(X_hourly, dtype=np.float32)
X_halfday = np.asarray(X_halfday, dtype=np.float32)
X_daily = np.asarray(X_daily, dtype=np.float32)
X_weekly = np.asarray(X_weekly, dtype=np.float32)

y_5d = np.asarray(y_by_horizon[5], dtype=np.float32)
y_10d = np.asarray(y_by_horizon[10], dtype=np.float32)
y_20d = np.asarray(y_by_horizon[20], dtype=np.float32)

meta = pd.DataFrame(meta_records)


# =====================================================
# SAVE
# =====================================================

save_dir = ROOT / CONFIG["output_dir"]
save_dir.mkdir(parents=True, exist_ok=True)

np.save(save_dir / "X_hourly.npy", X_hourly)
np.save(save_dir / "X_halfday.npy", X_halfday)
np.save(save_dir / "X_daily.npy", X_daily)
np.save(save_dir / "X_weekly.npy", X_weekly)

np.save(save_dir / "y_5d.npy", y_5d)
np.save(save_dir / "y_10d.npy", y_10d)
np.save(save_dir / "y_20d.npy", y_20d)

# Keep compatibility with old downstream scripts.
np.save(save_dir / "y.npy", y_20d)

meta.to_csv(save_dir / "meta.csv", index=False)


# =====================================================
# SUMMARY
# =====================================================

print("\nDataset Summary")
print("-" * 60)
print("X_hourly:", X_hourly.shape)
print("X_halfday:", X_halfday.shape)
print("X_daily:", X_daily.shape)
print("X_weekly:", X_weekly.shape)
print("y_5d:", y_5d.shape, "mean=", float(y_5d.mean()), "std=", float(y_5d.std()))
print("y_10d:", y_10d.shape, "mean=", float(y_10d.mean()), "std=", float(y_10d.std()))
print("y_20d:", y_20d.shape, "mean=", float(y_20d.mean()), "std=", float(y_20d.std()))

print("\nNaN Check")
print("-" * 60)
print("X_hourly NaN:", int(np.isnan(X_hourly).sum()))
print("X_halfday NaN:", int(np.isnan(X_halfday).sum()))
print("X_daily NaN:", int(np.isnan(X_daily).sum()))
print("X_weekly NaN:", int(np.isnan(X_weekly).sum()))
print("y_5d NaN:", int(np.isnan(y_5d).sum()))
print("y_10d NaN:", int(np.isnan(y_10d).sum()))
print("y_20d NaN:", int(np.isnan(y_20d).sum()))

if len(meta) > 0:
    print("\nAnchor Range")
    print("-" * 60)
    print("start:", meta["anchor_date"].min())
    print("end  :", meta["anchor_date"].max())