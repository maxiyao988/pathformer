import pandas as pd
import numpy as np

stocks = [
    "FSLR",
    "ENPH",
    "SEDG",
    "CSIQ",
    "RUN"
]

benchmark = pd.read_csv(
    "dataset/finance/icln_raw.csv"
)

benchmark["future_return_5d"] = np.log(
    benchmark["Close"].shift(-5)
    / benchmark["Close"]
)

benchmark = benchmark[
    ["Date","future_return_5d"]
]

benchmark.columns = [
    "date",
    "benchmark_return"
]

all_data = []

for ticker in stocks:

    print(f"Processing {ticker}")

    df = pd.read_csv(
        f"dataset/finance/{ticker.lower()}.csv"
    )

    df["future_return_5d"] = (
        df["close"]
        .rolling(5)
        .sum()
        .shift(-5)
    )

    merged = pd.merge(
        df,
        benchmark,
        on="date",
        how="inner"
    )

    merged["excess_return"] = (
        merged["future_return_5d"]
        -
        merged["benchmark_return"]
    )

    merged["ticker"] = ticker

    all_data.append(merged)

final_df = pd.concat(all_data)

final_df = final_df.dropna()

final_df.to_csv(
    "dataset/finance/solar_excess_return.csv",
    index=False
)

print(final_df.head())
print()
print(final_df.shape)