import yfinance as yf
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

TICKER = "FSLR"

save_dir = Path(
    "dataset/multiscale"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# HOURLY
# ==========================================

print()
print("=" * 50)
print("DOWNLOADING HOURLY")
print("=" * 50)

hourly = yf.download(
    TICKER,
    period="730d",
    interval="1h",
    auto_adjust=True,
    progress=False
)

print(hourly.shape)

hourly.to_csv(
    save_dir /
    "FSLR_hourly.csv"
)

# ==========================================
# DAILY
# ==========================================

print()
print("=" * 50)
print("DOWNLOADING DAILY")
print("=" * 50)

daily = yf.download(
    TICKER,
    period="max",
    interval="1d",
    auto_adjust=True,
    progress=False
)

print(daily.shape)

daily.to_csv(
    save_dir /
    "FSLR_daily.csv"
)

# ==========================================
# WEEKLY
# ==========================================

print()
print("=" * 50)
print("DOWNLOADING WEEKLY")
print("=" * 50)

weekly = yf.download(
    TICKER,
    period="max",
    interval="1wk",
    auto_adjust=True,
    progress=False
)

print(weekly.shape)

weekly.to_csv(
    save_dir /
    "FSLR_weekly.csv"
)

# ==========================================
# SUMMARY
# ==========================================

print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)

print("Hourly :", hourly.shape)
print("Daily  :", daily.shape)
print("Weekly :", weekly.shape)

print()
print("Saved to:")
print(save_dir)