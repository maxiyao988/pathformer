"""
Task 0 — Standardize Data Format
=================================
Enforce a unified schema across all source datasets:
  - Column names: datetime / open / high / low / close / volume (lowercase)
  - datetime: naive datetime64[ns] (no timezone)
  - Price columns: float64
  - volume: Int64
  - Sorted ascending by datetime
  - Duplicate timestamps removed (keep first)

Input:  dataset/multiscale/  (source files, read-only)
Output: dataset/processed/   (standardized files)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = Path(__file__).resolve().parents[2]          # pathformer/
SRC_DIR  = BASE_DIR / "dataset" / "multiscale"
OUT_DIR  = BASE_DIR / "dataset" / "processed"

SOURCES = {
    "FSLR_hourly": {
        "file":        SRC_DIR / "FSLR_hourly_clean_bloomberg_clean.csv",
        "datetime_col": "datetime",
        "dayfirst":    True,     # format: 31/5/2018 9:30
    },
    "FSLR_daily": {
        "file":        SRC_DIR / "FSLR_daily_clean.csv",
        "datetime_col": "date",
        "dayfirst":    False,
    },
    "FSLR_weekly": {
        "file":        SRC_DIR / "FSLR_weekly_clean.csv",
        "datetime_col": "date",
        "dayfirst":    False,
    },
}

# Canonical lowercase column name mapping
RENAME_MAP = {
    "datetime": "datetime",
    "date":     "datetime",
    "open":     "open",
    "Open":     "open",
    "high":     "high",
    "High":     "high",
    "low":      "low",
    "Low":      "low",
    "close":    "close",
    "Close":    "close",
    "volume":   "volume",
    "Volume":   "volume",
    "Volumn":   "volume",   # Bloomberg source typo
}

FINAL_COLS = ["datetime", "open", "high", "low", "close", "volume"]


# ==============================================================
# FUNCTIONS
# ==============================================================

def parse_volume(series: pd.Series) -> pd.Series:
    """Parse volume strings like '107.203k' or '1.2m' into integers."""
    def _parse(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower().replace(",", "")
        if s.endswith("k"):
            return float(s[:-1]) * 1_000
        if s.endswith("m"):
            return float(s[:-1]) * 1_000_000
        return float(s)
    return series.map(_parse)


def standardize(name: str, cfg: dict) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(cfg["file"])
    print(f"Raw rows: {len(df):,}  Columns: {df.columns.tolist()}")

    # 1. Normalize column names
    df = df.rename(columns=RENAME_MAP)

    # 2. Keep only target columns (drop any extras)
    df = df[[c for c in FINAL_COLS if c in df.columns]]

    # 3. Parse datetime as naive (no timezone)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=cfg["dayfirst"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

    # 4. Parse volume strings (e.g. '107.203k')
    df["volume"] = parse_volume(df["volume"])

    # 5. Cast types
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    # 6. Sort ascending
    df = df.sort_values("datetime").reset_index(drop=True)

    # 7. Drop duplicate timestamps (keep first)
    dup_count = df["datetime"].duplicated().sum()
    if dup_count:
        print(f"WARNING: {dup_count} duplicate timestamps found — keeping first occurrence")
        df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)

    # 8. Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"WARNING: missing values:\n{missing[missing > 0]}")
    else:
        print("OK: no missing values")

    # 9. OHLC consistency check
    bad = (
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"]  > df["open"]) |
        (df["low"]  > df["close"]) |
        (df["high"] < df["low"])
    ).sum()
    if bad:
        print(f"WARNING: {bad} rows with invalid OHLC relationships")
    else:
        print("OK: OHLC consistency passed")

    # 10. Summary
    print(f"Start:  {df['datetime'].min()}")
    print(f"End:    {df['datetime'].max()}")
    print(f"Rows:   {len(df):,}")
    print(f"Dtypes:\n{df.dtypes}")

    return df


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, cfg in SOURCES.items():
        df = standardize(name, cfg)
        out_path = OUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved -> {out_path}")
        results[name] = df

    # ==============================================================
    # SUMMARY REPORT
    # ==============================================================
    print("\n" + "=" * 60)
    print("TASK 0 SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<30} {'Rows':>8}  {'Start':>12}  {'End':>12}")
    print("-" * 70)
    for name, df in results.items():
        print(
            f"{name:<30} {len(df):>8,}  "
            f"{str(df['datetime'].min())[:10]:>12}  "
            f"{str(df['datetime'].max())[:10]:>12}"
        )
    print("=" * 60)
    print(f"Output dir: {OUT_DIR}")
    print("Task 0 complete.")


if __name__ == "__main__":
    main()
