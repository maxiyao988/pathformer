import yfinance as yf
import pandas as pd

df = yf.download(
    "ICLN",
    start="2016-01-01",
    end="2025-12-31",
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

df.to_csv(
    "dataset/finance/icln_raw.csv",
    index=False
)

print(df.head())
print(df.shape)