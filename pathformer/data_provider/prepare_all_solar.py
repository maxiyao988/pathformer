import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

tickers = [
    "FSLR",
    "ENPH",
    "SEDG",
    "CSIQ",
    "RUN"
]

save_dir = Path("dataset/finance")
save_dir.mkdir(parents=True, exist_ok=True)

for ticker in tickers:

    print(f"\nProcessing {ticker}")

    df = yf.download(
        ticker,
        start="2016-01-01",
        end="2025-12-31",
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    # log return
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.log(df[col] / df[col].shift(1))

    # log volume
    df["Volume"] = np.log(df["Volume"])

    df = df.dropna()

    df = df.reset_index()

    df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    save_path = save_dir / f"{ticker.lower()}.csv"

    df.to_csv(
        save_path,
        index=False
    )

    print(
        f"Saved {ticker}:",
        df.shape,
        "->",
        save_path
    )

print("\nFinished.")