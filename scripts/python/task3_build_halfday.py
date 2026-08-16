"""
Task 3 - Rebuild Half-Day Dataset
==================================
Aggregate standardized hourly bars into half-day sessions:

  Morning   : bars at 09:30, 10:30, 11:30, 12:30  (4 bars)
  Afternoon : bars at 13:30, 14:30, 15:30          (3 bars)

Aggregation per session:
  open   = first bar open
  high   = max of all bar highs
  low    = min of all bar lows
  close  = last bar close
  volume = sum of all bar volumes

Output columns:
  datetime  (session start timestamp, e.g. 2018-05-31 09:30:00 / 13:30:00)
  session   (morning / afternoon)
  open / high / low / close / volume

Input : dataset/processed/FSLR_hourly.csv
Output: dataset/merged/FSLR_halfday_full.csv
"""

import pandas as pd
from pathlib import Path

# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_PATH = BASE_DIR / "dataset" / "processed" / "FSLR_hourly.csv"
OUT_DIR  = BASE_DIR / "dataset" / "merged"
OUT_PATH = OUT_DIR / "FSLR_halfday_full.csv"

# Bar times that belong to each session
MORNING_TIMES   = {9, 10, 11, 12}   # 09:30, 10:30, 11:30, 12:30
AFTERNOON_TIMES = {13, 14, 15}      # 13:30, 14:30, 15:30

MIN_MORNING_BARS   = 3   # drop sessions with fewer bars
MIN_AFTERNOON_BARS = 2


# ==============================================================
# LOAD
# ==============================================================

def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ==============================================================
# ASSIGN SESSION
# ==============================================================

def assign_session(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["datetime"].dt.hour
    df = df.copy()
    df["date"]    = df["datetime"].dt.date
    df["hour"]    = hour
    df["session"] = None
    df.loc[hour.isin(MORNING_TIMES),   "session"] = "morning"
    df.loc[hour.isin(AFTERNOON_TIMES), "session"] = "afternoon"

    # Drop any bars that don't belong to either session
    unknown = df["session"].isna().sum()
    if unknown:
        print(f"WARNING: {unknown} bars with unrecognised hours dropped")
    df = df.dropna(subset=["session"]).copy()
    return df


# ==============================================================
# AGGREGATE
# ==============================================================

def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["date", "session"])

    records = []
    dropped_morning   = 0
    dropped_afternoon = 0

    for (date, session), grp in groups:
        grp = grp.sort_values("datetime")
        n   = len(grp)

        # Enforce minimum bar count
        if session == "morning" and n < MIN_MORNING_BARS:
            dropped_morning += 1
            continue
        if session == "afternoon" and n < MIN_AFTERNOON_BARS:
            dropped_afternoon += 1
            continue

        records.append({
            "datetime": grp["datetime"].iloc[0],   # session start
            "date":     date,
            "session":  session,
            "open":     grp["open"].iloc[0],
            "high":     grp["high"].max(),
            "low":      grp["low"].min(),
            "close":    grp["close"].iloc[-1],
            "volume":   int(grp["volume"].sum()),
            "bar_count": n,
        })

    if dropped_morning:
        print(f"WARNING: {dropped_morning} morning sessions dropped "
              f"(< {MIN_MORNING_BARS} bars)")
    if dropped_afternoon:
        print(f"WARNING: {dropped_afternoon} afternoon sessions dropped "
              f"(< {MIN_AFTERNOON_BARS} bars)")

    halfday = pd.DataFrame(records)
    halfday = halfday.sort_values("datetime").reset_index(drop=True)
    return halfday


# ==============================================================
# AUDIT
# ==============================================================

def audit(halfday: pd.DataFrame) -> None:
    n = len(halfday)
    print(f"\n{'='*60}")
    print("HALF-DAY DATASET AUDIT")
    print(f"{'='*60}")
    print(f"Total sessions      : {n:,}")
    print(f"Start               : {halfday['datetime'].min()}")
    print(f"End                 : {halfday['datetime'].max()}")

    by_session = halfday["session"].value_counts()
    print(f"\nMorning sessions    : {by_session.get('morning', 0):,}")
    print(f"Afternoon sessions  : {by_session.get('afternoon', 0):,}")

    # Day-level session completeness
    session_pivot = (
        halfday.pivot_table(
            index="date",
            columns="session",
            values="datetime",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
    )
    morning_count = session_pivot["morning"] if "morning" in session_pivot.columns else pd.Series(0, index=session_pivot.index)
    afternoon_count = session_pivot["afternoon"] if "afternoon" in session_pivot.columns else pd.Series(0, index=session_pivot.index)

    both = int(((morning_count > 0) & (afternoon_count > 0)).sum())
    morning_only = int(((morning_count > 0) & (afternoon_count == 0)).sum())
    afternoon_only = int(((morning_count == 0) & (afternoon_count > 0)).sum())

    print(f"Days with both sessions : {both:,}")
    print(f"Days with morning only  : {morning_only:,}")
    print(f"Days with afternoon only: {afternoon_only:,}")

    print(f"\nBar count distribution (morning):")
    print(halfday[halfday["session"] == "morning"]["bar_count"]
          .value_counts().sort_index().to_string())
    print(f"\nBar count distribution (afternoon):")
    print(halfday[halfday["session"] == "afternoon"]["bar_count"]
          .value_counts().sort_index().to_string())

    print(f"\nMissing values:")
    print(halfday[["open","high","low","close","volume"]].isnull().sum().to_string())

    # OHLC check
    bad = (
        (halfday["high"] < halfday["open"]) |
        (halfday["high"] < halfday["close"]) |
        (halfday["low"]  > halfday["open"]) |
        (halfday["low"]  > halfday["close"]) |
        (halfday["high"] < halfday["low"])
    ).sum()
    print(f"\nInvalid OHLC rows   : {bad}")

    print(f"\nPrice statistics:")
    print(halfday[["open","high","low","close"]].describe().to_string())


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hourly data ...")
    hourly = load_hourly(SRC_PATH)
    print(f"  Rows: {len(hourly):,}  Range: {hourly['datetime'].min()} -> "
          f"{hourly['datetime'].max()}")

    print("\nAssigning sessions ...")
    hourly = assign_session(hourly)

    print("Aggregating half-day sessions ...")
    halfday = aggregate_sessions(hourly)

    # Save (keep datetime + session columns; drop helper columns)
    save_cols = ["datetime", "session", "open", "high", "low", "close", "volume"]
    halfday[save_cols].to_csv(OUT_PATH, index=False)

    audit(halfday)

    print(f"\nSaved -> {OUT_PATH}")
    print("Task 3 complete.")


if __name__ == "__main__":
    main()
