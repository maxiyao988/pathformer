import pandas as pd
import numpy as np
from pathlib import Path

SEQ_LEN = 96

df = pd.read_csv(
    "dataset/finance/solar_excess_return.csv"
)

feature_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

X = []
y = []
tickers = []
dates = []

for ticker in df["ticker"].unique():

    print(f"Processing {ticker}")

    stock_df = (
        df[df["ticker"] == ticker]
        .sort_values("date")
        .reset_index(drop=True)
    )

    features = stock_df[
        feature_cols
    ].values

    target = stock_df[
        "future_return_5d"
    ].values

    date_arr = stock_df[
        "date"
    ].values

    for i in range(
        SEQ_LEN,
        len(stock_df)
    ):

        X.append(
            features[
                i-SEQ_LEN:i
            ]
        )

        y.append(
            target[i]
        )

        tickers.append(
            ticker
        )

        dates.append(
            date_arr[i]
        )

X = np.array(X)
y = np.array(y)

save_dir = Path(
    "dataset/financial_ml"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    save_dir / "X_raw.npy",
    X
)

np.save(
    save_dir / "y_raw.npy",
    y
)

meta = pd.DataFrame({
    "ticker": tickers,
    "date": dates
})

meta.to_csv(
    save_dir / "meta_raw.csv",
    index=False
)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)
print()
print("saved.")