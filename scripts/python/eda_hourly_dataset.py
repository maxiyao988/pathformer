import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "pathformer/dataset/processed/FSLR_hourly_bloomberg.csv"

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

df = df.sort_values("datetime").reset_index(drop=True)

print(df.head())

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

# ==========================================================
# DATE RANGE
# ==========================================================

print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)

print("Start:", df.datetime.min())
print("End  :", df.datetime.max())

years = (df.datetime.max() - df.datetime.min()).days / 365
print("Years:", years)

# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

# ==========================================================
# DUPLICATES
# ==========================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print("Duplicated rows:", df.duplicated().sum())
print("Duplicated timestamps:", df.datetime.duplicated().sum())

# ==========================================================
# TRADING DAYS
# ==========================================================

bars = df.groupby(df.datetime.dt.date).size()

print("\n" + "=" * 70)
print("TRADING DAYS")
print("=" * 70)

print("Trading days:", len(bars))

print("\nBars/day statistics:")

print(bars.describe())

print("\nDistribution:")

print(bars.value_counts().sort_index())

# ==========================================================
# PRICE LOGIC
# ==========================================================

print("\n" + "=" * 70)
print("PRICE CONSISTENCY CHECK")
print("=" * 70)

bad = df[
    (df.high < df.open)
    | (df.high < df.close)
    | (df.low > df.open)
    | (df.low > df.close)
    | (df.high < df.low)
]

print("Invalid OHLC rows:", len(bad))

# ==========================================================
# RETURN ANALYSIS
# ==========================================================

df["Return"] = df["close"].pct_change()

print("\n" + "=" * 70)
print("RETURN STATISTICS")
print("=" * 70)

print(df["Return"].describe())

print("\nLargest Returns")

print(df.nlargest(10, "Return")[["datetime", "Return"]])

print("\nSmallest Returns")

print(df.nsmallest(10, "Return")[["datetime", "Return"]])

# ==========================================================
# VOLUME
# ==========================================================

print("\n" + "=" * 70)
print("VOLUME")
print("=" * 70)

print(df["volume"].describe())

# ==========================================================
# CORRELATION
# ==========================================================

print("\n" + "=" * 70)
print("CORRELATION")
print("=" * 70)

print(
    df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].corr()
)

# ==========================================================
# PLOTS
# ==========================================================

plt.figure(figsize=(14,5))
plt.plot(df.datetime, df.close)
plt.title("Close Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df.Return.dropna(), bins=100)
plt.title("Hourly Return Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(np.log1p(df.volume), bins=80)
plt.title("Log Volume")
plt.tight_layout()
plt.show()

print("\nEDA FINISHED.")