try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance is required to run this script. Install it with: pip install yfinance"
    )
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# DOWNLOAD DATA
# ==========================================

ticker = "FSLR"

df = yf.download(
    ticker,
    start="2016-01-01",
    end="2025-12-31",
    auto_adjust=True
)

# ==========================================
# KEEP FEATURES
# ==========================================
# ==========================================
# FLATTEN MULTIINDEX
# ==========================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ==========================================
# KEEP FEATURES
# ==========================================

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]]

# ==========================================
# LOG RETURN TRANSFORMATION
# ==========================================

for col in ["Open", "High", "Low", "Close"]:

    df[col] = np.log(df[col]) - np.log(df[col].shift(1))

# volume log transform
df["Volume"] = np.log(df["Volume"])

# ==========================================
# DROP NAN
# ==========================================

df = df.dropna()

# ==========================================
# SAVE
# ==========================================

save_dir = Path("./dataset/finance")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "fslr.csv"
df = df.reset_index()

df.columns = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

df.to_csv(save_path, index=False)

print(df.head())

print(f"\nSaved to: {save_path}")