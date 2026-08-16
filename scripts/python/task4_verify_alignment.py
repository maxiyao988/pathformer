"""
Task 4 - Verify Cross-Frequency Alignment
=========================================
Validate whether Hourly, Half-Day, Daily, and Weekly data can be aligned
on the same daily anchor date without look-ahead leakage.

Alignment rule:
  - Daily is the anchor frequency.
  - For each daily anchor date D, each input frequency must use only data
    with timestamp <= D (information available up to that day).
  - The target is future log return from D to D + target_horizon.

This script reports:
  - valid samples
  - discarded samples
  - discard reasons
  - coverage of each frequency
  - leakage checks

Input files:
  - dataset/processed/FSLR_hourly.csv
  - dataset/merged/FSLR_halfday_full.csv
  - dataset/processed/FSLR_daily.csv
  - dataset/processed/FSLR_weekly.csv

Output:
  - dataset/audit/task4_alignment_report.txt
  - dataset/audit/task4_alignment_meta.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset"
AUDIT_DIR = DATA_DIR / "audit"

HOURLY_PATH = DATA_DIR / "processed" / "FSLR_hourly.csv"
HALFDAY_PATH = DATA_DIR / "merged" / "FSLR_halfday_full.csv"
DAILY_PATH = DATA_DIR / "processed" / "FSLR_daily.csv"
WEEKLY_PATH = DATA_DIR / "processed" / "FSLR_weekly.csv"

TARGET_HORIZON = 20

LOOKBACKS = {
    "hourly": 48,
    "halfday": 40,
    "daily": 90,
    "weekly": 52,
}

FEATURES = ["open", "high", "low", "close", "volume"]


# ==============================================================
# LOAD
# ==============================================================

def load_csv(path: Path, datetime_col: str = "datetime") -> pd.DataFrame:
    df = pd.read_csv(path)
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df


# ==============================================================
# HELPERS
# ==============================================================

def format_date(value) -> str:
    if pd.isna(value):
        return "NaT"
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def session_audit(halfday: pd.DataFrame) -> tuple[int, int, int]:
    """Return (days_with_both_sessions, morning_only_days, afternoon_only_days)."""
    pivot = (
        halfday.pivot_table(
            index="date",
            columns="session",
            values="datetime",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
    )
    morning_count = pivot["morning"] if "morning" in pivot.columns else pd.Series(0, index=pivot.index)
    afternoon_count = pivot["afternoon"] if "afternoon" in pivot.columns else pd.Series(0, index=pivot.index)

    both = int(((morning_count > 0) & (afternoon_count > 0)).sum())
    morning_only = int(((morning_count > 0) & (afternoon_count == 0)).sum())
    afternoon_only = int(((morning_count == 0) & (afternoon_count > 0)).sum())
    return both, morning_only, afternoon_only


# ==============================================================
# ALIGNMENT CHECK
# ==============================================================

def build_alignment_report(hourly: pd.DataFrame,
                           halfday: pd.DataFrame,
                           daily: pd.DataFrame,
                           weekly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reasons = Counter()
    records = []
    candidate_samples = 0

    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = weekly.sort_values("datetime").reset_index(drop=True)
    halfday = halfday.sort_values("datetime").reset_index(drop=True)
    hourly = hourly.sort_values("datetime").reset_index(drop=True)

    overlap_start = max(
        hourly["datetime"].min().normalize(),
        halfday["datetime"].min().normalize(),
        daily["datetime"].min().normalize(),
        weekly["datetime"].min().normalize(),
    )
    overlap_end = min(
        hourly["datetime"].max().normalize(),
        halfday["datetime"].max().normalize(),
        daily["datetime"].max().normalize(),
        weekly["datetime"].max().normalize(),
    )

    for idx in range(LOOKBACKS["daily"], len(daily) - TARGET_HORIZON):
        anchor_dt = pd.Timestamp(daily.loc[idx, "datetime"])
        anchor_date = anchor_dt.normalize()

        if anchor_date < overlap_start or anchor_date > overlap_end:
            reasons["outside_frequency_overlap"] += 1
            continue

        # Daily window: last H daily bars strictly before anchor row.
        daily_window = daily.iloc[idx - LOOKBACKS["daily"]: idx]
        if len(daily_window) < LOOKBACKS["daily"]:
            reasons["insufficient_daily_history"] += 1
            continue

        # Weekly window: latest weekly bars at or before the anchor date.
        weekly_subset = weekly[weekly["datetime"] <= anchor_date]
        if len(weekly_subset) < LOOKBACKS["weekly"]:
            reasons["insufficient_weekly_history"] += 1
            continue

        # Half-day window: latest half-day bars at or before anchor date.
        halfday_subset = halfday[halfday["datetime"] <= anchor_dt]
        if len(halfday_subset) < LOOKBACKS["halfday"]:
            reasons["insufficient_halfday_history"] += 1
            continue

        # Hourly window: latest hourly bars at or before anchor date.
        hourly_subset = hourly[hourly["datetime"] <= anchor_dt]
        if len(hourly_subset) < LOOKBACKS["hourly"]:
            reasons["insufficient_hourly_history"] += 1
            continue

        # Target requires future data.
        if idx + TARGET_HORIZON >= len(daily):
            reasons["insufficient_target_horizon"] += 1
            continue

        candidate_samples += 1

        # Leakage check: no feature timestamp may be after the anchor date.
        max_hourly = hourly_subset["datetime"].max()
        max_halfday = halfday_subset["datetime"].max()
        max_weekly = weekly_subset["datetime"].max()
        max_daily = daily_window["datetime"].max()

        leakage = []
        if max_hourly > anchor_dt:
            leakage.append("hourly")
        if max_halfday > anchor_dt:
            leakage.append("halfday")
        if max_daily > anchor_dt:
            leakage.append("daily")
        if max_weekly > anchor_dt:
            leakage.append("weekly")

        if leakage:
            reasons["information_leakage"] += 1
            continue

        p0 = daily.loc[idx, "close"]
        p1 = daily.loc[idx + TARGET_HORIZON, "close"]
        target = float(np.log(p1 / p0))

        records.append({
            "anchor_date": anchor_dt,
            "target_date": daily.loc[idx + TARGET_HORIZON, "datetime"],
            "target_horizon": TARGET_HORIZON,
            "target_log_return": target,
            "hourly_end": max_hourly,
            "halfday_end": max_halfday,
            "daily_end": max_daily,
            "weekly_end": max_weekly,
            "daily_window_len": len(daily_window),
            "weekly_window_len": len(weekly_subset.tail(LOOKBACKS["weekly"])),
            "halfday_window_len": len(halfday_subset.tail(LOOKBACKS["halfday"])),
            "hourly_window_len": len(hourly_subset.tail(LOOKBACKS["hourly"])),
        })

    meta = {
        "total_daily_rows": len(daily),
        "target_horizon": TARGET_HORIZON,
        "lookbacks": LOOKBACKS,
        "overlap_start": format_date(overlap_start),
        "overlap_end": format_date(overlap_end),
        "valid_samples": len(records),
        "candidate_samples": candidate_samples,
        "discarded_samples": max(0, candidate_samples - len(records)),
        "discard_reasons": dict(reasons),
        "daily_start": format_date(daily["datetime"].min()),
        "daily_end": format_date(daily["datetime"].max()),
        "hourly_start": format_date(hourly["datetime"].min()),
        "hourly_end": format_date(hourly["datetime"].max()),
        "halfday_start": format_date(halfday["datetime"].min()),
        "halfday_end": format_date(halfday["datetime"].max()),
        "weekly_start": format_date(weekly["datetime"].min()),
        "weekly_end": format_date(weekly["datetime"].max()),
    }

    return pd.DataFrame(records), meta


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = load_csv(HOURLY_PATH)
    halfday = load_csv(HALFDAY_PATH)
    daily = load_csv(DAILY_PATH)
    weekly = load_csv(WEEKLY_PATH)

    # Normalize half-day helper columns.
    if "date" not in halfday.columns:
        halfday["date"] = pd.to_datetime(halfday["datetime"]).dt.normalize()
    else:
        halfday["date"] = pd.to_datetime(halfday["date"]).dt.normalize()

    if "session" not in halfday.columns:
        halfday["session"] = "unknown"

    report_df, meta = build_alignment_report(hourly, halfday, daily, weekly)
    report_path = AUDIT_DIR / "task4_alignment_meta.csv"
    report_df.to_csv(report_path, index=False)

    both, morning_only, afternoon_only = session_audit(halfday)

    lines = []
    lines.append("TASK 4 - CROSS-FREQUENCY ALIGNMENT REPORT")
    lines.append("=" * 72)
    lines.append(f"Overlap window           : {meta['overlap_start']} -> {meta['overlap_end']}")
    lines.append(f"Daily anchors in overlap : {meta['candidate_samples']}")
    lines.append(f"Valid aligned samples    : {meta['valid_samples']}")
    lines.append(f"Discarded samples        : {meta['discarded_samples']}")
    lines.append("")
    lines.append("Date coverage")
    lines.append(f"  Hourly  : {meta['hourly_start']} -> {meta['hourly_end']}")
    lines.append(f"  HalfDay : {meta['halfday_start']} -> {meta['halfday_end']}")
    lines.append(f"  Daily   : {meta['daily_start']} -> {meta['daily_end']}")
    lines.append(f"  Weekly  : {meta['weekly_start']} -> {meta['weekly_end']}")
    lines.append("")
    lines.append("Lookback windows")
    for name, value in LOOKBACKS.items():
        lines.append(f"  {name:<7}: {value}")
    lines.append("")
    lines.append("Discard reasons")
    if meta["discard_reasons"]:
        for reason, count in sorted(meta["discard_reasons"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("Half-day session audit")
    lines.append(f"  Days with both sessions    : {both}")
    lines.append(f"  Days with morning only     : {morning_only}")
    lines.append(f"  Days with afternoon only   : {afternoon_only}")
    lines.append("")
    lines.append("Leakage rule")
    lines.append("  Passed: all retained samples use only data with timestamp <= anchor date")

    report_path_txt = AUDIT_DIR / "task4_alignment_report.txt"
    report_path_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Saved detailed sample table -> {report_path}")
    print(f"Saved report              -> {report_path_txt}")
    print("Task 4 complete.")


if __name__ == "__main__":
    main()
