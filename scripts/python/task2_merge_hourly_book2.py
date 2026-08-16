"""
Task 2 (supplement) - Merge Book2 hourly raw data into cleaned hourly dataset.

This script:
1. Loads existing cleaned hourly data from dataset/processed/FSLR_hourly.csv
2. Loads new raw hourly data from dataset/multiscale/Book2.xlsx
3. Normalizes schema to: datetime, open, high, low, close, volume
4. Converts volume units such as "254.97k" / "1.2m" into numeric values
5. Merges, sorts, and de-duplicates by datetime
6. Saves merged output and prints an audit summary

Outputs:
- dataset/processed/FSLR_hourly_merged_book2.csv
- dataset/processed/FSLR_hourly.csv (updated main file)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EXISTING_PATH = BASE_DIR / "dataset" / "processed" / "FSLR_hourly.csv"
BOOK2_PATH = BASE_DIR / "dataset" / "multiscale" / "Book2.xlsx"

MERGED_OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "FSLR_hourly_merged_book2.csv"
MAIN_OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "FSLR_hourly.csv"


# ==============================================================
# HELPERS
# ==============================================================

def parse_volume(value) -> float | np.nan:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace(",", "")
    if text == "":
        return np.nan

    if text.endswith("k"):
        return float(text[:-1]) * 1_000
    if text.endswith("m"):
        return float(text[:-1]) * 1_000_000
    if text.endswith("b"):
        return float(text[:-1]) * 1_000_000_000

    return float(text)


def normalize_book2(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volumn": "volume",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Book2 is missing required columns: {missing}")

    df = df[required].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["volume"] = df["volume"].map(parse_volume)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")

    return df


def normalize_existing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")
    return df


def ohlc_invalid_count(df: pd.DataFrame) -> int:
    return int(
        (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["high"] < df["low"])
        ).sum()
    )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:
    existing = pd.read_csv(EXISTING_PATH)
    book2 = pd.read_excel(BOOK2_PATH)

    existing = normalize_existing(existing)
    book2 = normalize_book2(book2)

    print("=" * 70)
    print("INPUT SUMMARY")
    print("=" * 70)
    print(f"Existing rows : {len(existing):,}")
    print(f"Existing range: {existing['datetime'].min()} -> {existing['datetime'].max()}")
    print(f"Book2 rows    : {len(book2):,}")
    print(f"Book2 range   : {book2['datetime'].min()} -> {book2['datetime'].max()}")

    overlap = len(set(existing["datetime"]).intersection(set(book2["datetime"])))
    print(f"Datetime overlap rows: {overlap:,}")

    # Keep Book2 values for overlap by concatenating existing first and dropping duplicates keep='last'.
    merged = pd.concat([existing, book2], axis=0, ignore_index=True)
    before = len(merged)
    merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    removed = before - len(merged)

    # Final type cleanup.
    merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").round().astype("Int64")

    print("\n" + "=" * 70)
    print("MERGE AUDIT")
    print("=" * 70)
    print(f"Merged rows            : {len(merged):,}")
    print(f"Duplicates removed     : {removed:,}")
    print(f"Merged range           : {merged['datetime'].min()} -> {merged['datetime'].max()}")
    print(f"Missing values         :\n{merged.isna().sum()}")
    print(f"Invalid OHLC rows      : {ohlc_invalid_count(merged):,}")

    MERGED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    merged.to_csv(MAIN_OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(f"- {MERGED_OUTPUT_PATH}")
    print(f"- {MAIN_OUTPUT_PATH} (updated)")


if __name__ == "__main__":
    main()
