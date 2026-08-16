"""
Task 1 - Full EDA
=================
Run a complete financial data quality check across all frequency datasets.

Checks:
  - Missing values
  - Duplicate timestamps
  - Date coverage
  - Trading day / bar statistics
  - OHLC consistency
  - Return distribution (mean, std, skew, kurtosis, outliers)
  - Volume distribution
  - Correlation matrix
  - Price visualization
  - Outlier detection (z-score and IQR)

Input:  dataset/processed/
Output: dataset/eda/  (report + plots)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves to file
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR  = Path(__file__).resolve().parents[2]
SRC_DIR   = BASE_DIR / "dataset" / "processed"
OUT_DIR   = BASE_DIR / "dataset" / "eda"
REPORT    = OUT_DIR / "eda_report.txt"

DATASETS = {
    "hourly":  SRC_DIR / "FSLR_hourly.csv",
    "daily":   SRC_DIR / "FSLR_daily.csv",
    "weekly":  SRC_DIR / "FSLR_weekly.csv",
}

Z_THRESHOLD   = 4.0   # z-score outlier flag
IQR_FACTOR    = 3.0   # IQR outlier flag

# ==============================================================
# HELPERS
# ==============================================================

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df


def section(title: str, width: int = 68) -> str:
    bar = "=" * width
    return f"\n{bar}\n{title}\n{bar}"


def fmt_pct(n: int, total: int) -> str:
    return f"{n:,} ({n/total*100:.2f}%)" if total else str(n)


def outliers_zscore(series: pd.Series, threshold: float) -> pd.Series:
    z = np.abs(stats.zscore(series.dropna()))
    idx = series.dropna().index[z > threshold]
    return series.loc[idx]


def outliers_iqr(series: pd.Series, factor: float) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    mask = (series < q1 - factor * iqr) | (series > q3 + factor * iqr)
    return series[mask]


# ==============================================================
# PER-DATASET EDA
# ==============================================================

def eda_dataset(name: str, df: pd.DataFrame) -> list[str]:
    lines = []
    n = len(df)

    lines.append(section(f"DATASET: {name.upper()}  ({n:,} rows)"))

    # ----------------------------------------------------------
    # 1. Date coverage
    # ----------------------------------------------------------
    lines.append("\n--- Date Coverage ---")
    lines.append(f"Start : {df['datetime'].min()}")
    lines.append(f"End   : {df['datetime'].max()}")
    span_days = (df["datetime"].max() - df["datetime"].min()).days
    lines.append(f"Span  : {span_days} days  ({span_days/365.25:.2f} years)")

    # ----------------------------------------------------------
    # 2. Missing values
    # ----------------------------------------------------------
    lines.append("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.any():
        lines.append(missing[missing > 0].to_string())
    else:
        lines.append("None")

    # ----------------------------------------------------------
    # 3. Duplicate timestamps
    # ----------------------------------------------------------
    lines.append("\n--- Duplicate Timestamps ---")
    dup = df["datetime"].duplicated().sum()
    lines.append(f"Count: {fmt_pct(dup, n)}")

    # ----------------------------------------------------------
    # 4. Trading bar statistics
    # ----------------------------------------------------------
    lines.append("\n--- Bar Statistics per Calendar Day ---")
    bars = df.groupby(df["datetime"].dt.date).size()
    lines.append(f"Unique trading days: {len(bars):,}")
    lines.append(bars.describe().to_string())
    lines.append("\nBar count distribution:")
    lines.append(bars.value_counts().sort_index().to_string())

    # ----------------------------------------------------------
    # 5. OHLC consistency
    # ----------------------------------------------------------
    lines.append("\n--- OHLC Consistency ---")
    bad_mask = (
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"]  > df["open"]) |
        (df["low"]  > df["close"]) |
        (df["high"] < df["low"])
    )
    bad_count = bad_mask.sum()
    lines.append(f"Invalid OHLC rows: {fmt_pct(bad_count, n)}")
    if bad_count:
        lines.append(df[bad_mask][["datetime", "open", "high", "low", "close"]].to_string())

    # ----------------------------------------------------------
    # 6. Price statistics
    # ----------------------------------------------------------
    lines.append("\n--- Price Statistics ---")
    lines.append(df[["open", "high", "low", "close"]].describe().to_string())

    # ----------------------------------------------------------
    # 7. Return distribution
    # ----------------------------------------------------------
    lines.append("\n--- Return Distribution ---")
    df["return"] = df["close"].pct_change()
    ret = df["return"].dropna()
    lines.append(f"Mean   : {ret.mean():.6f}")
    lines.append(f"Std    : {ret.std():.6f}")
    lines.append(f"Skew   : {ret.skew():.4f}")
    lines.append(f"Kurt   : {ret.kurtosis():.4f}")
    lines.append(f"Min    : {ret.min():.6f}")
    lines.append(f"Max    : {ret.max():.6f}")
    lines.append(f"p1     : {ret.quantile(0.01):.6f}")
    lines.append(f"p99    : {ret.quantile(0.99):.6f}")

    # Outliers in returns
    z_out = outliers_zscore(ret, Z_THRESHOLD)
    iqr_out = outliers_iqr(ret, IQR_FACTOR)
    lines.append(f"\nReturn outliers (|z| > {Z_THRESHOLD}): {len(z_out)}")
    lines.append(f"Return outliers (IQR x{IQR_FACTOR}):    {len(iqr_out)}")

    top5 = df.nlargest(5, "return")[["datetime", "return"]]
    bot5 = df.nsmallest(5, "return")[["datetime", "return"]]
    lines.append("\nTop 5 positive returns:")
    lines.append(top5.to_string(index=False))
    lines.append("\nTop 5 negative returns:")
    lines.append(bot5.to_string(index=False))

    # ----------------------------------------------------------
    # 8. Volume distribution
    # ----------------------------------------------------------
    lines.append("\n--- Volume Distribution ---")
    vol = df["volume"].dropna()
    lines.append(vol.describe().to_string())
    vol_z_out = outliers_zscore(vol, Z_THRESHOLD)
    lines.append(f"\nVolume outliers (|z| > {Z_THRESHOLD}): {len(vol_z_out)}")

    # ----------------------------------------------------------
    # 9. Correlation matrix
    # ----------------------------------------------------------
    lines.append("\n--- Correlation Matrix ---")
    corr_cols = ["open", "high", "low", "close", "volume"]
    lines.append(df[corr_cols].corr().round(4).to_string())

    lines.append("")
    return lines


# ==============================================================
# PLOTS
# ==============================================================

def plot_dataset(name: str, df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"FSLR EDA — {name.capitalize()}", fontsize=14, fontweight="bold")

    # 1. Close price
    ax = axes[0, 0]
    ax.plot(df["datetime"], df["close"], lw=0.8, color="steelblue")
    ax.set_title("Close Price")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=30)

    # 2. Returns
    df["return"] = df["close"].pct_change()
    ax = axes[0, 1]
    ax.hist(df["return"].dropna(), bins=100, color="steelblue", edgecolor="none")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Return")

    # 3. Volume
    ax = axes[1, 0]
    ax.bar(df["datetime"], df["volume"], width=1, color="gray", alpha=0.6)
    ax.set_title("Volume")
    ax.tick_params(axis="x", labelrotation=30)

    # 4. Log volume distribution
    ax = axes[1, 1]
    log_vol = np.log1p(df["volume"].dropna())
    ax.hist(log_vol, bins=80, color="gray", edgecolor="none")
    ax.set_title("Log Volume Distribution")

    # 5. Rolling 30-bar volatility
    ax = axes[2, 0]
    roll_vol = df["return"].rolling(30).std() * np.sqrt(252)
    ax.plot(df["datetime"], roll_vol, lw=0.8, color="firebrick")
    ax.set_title("Rolling 30-bar Annualized Volatility")
    ax.tick_params(axis="x", labelrotation=30)

    # 6. QQ plot of returns
    ax = axes[2, 1]
    ret_clean = df["return"].dropna()
    (osm, osr), (slope, intercept, _) = stats.probplot(ret_clean, dist="norm")
    ax.scatter(osm, osr, s=2, alpha=0.4, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept, color="red", lw=1)
    ax.set_title("Q-Q Plot (Returns vs Normal)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    plt.tight_layout()
    out_path = out_dir / f"eda_{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")


# ==============================================================
# CROSS-FREQUENCY COMPARISON
# ==============================================================

def cross_freq_summary(datasets: dict[str, pd.DataFrame]) -> list[str]:
    lines = [section("CROSS-FREQUENCY COMPARISON")]
    lines.append(f"\n{'Frequency':<12} {'Rows':>8}  {'Start':>12}  {'End':>12}  "
                 f"{'RetMean':>10}  {'RetStd':>10}  {'Skew':>8}  {'Kurt':>8}")
    lines.append("-" * 85)
    for name, df in datasets.items():
        ret = df["close"].pct_change().dropna()
        lines.append(
            f"{name:<12} {len(df):>8,}  "
            f"{str(df['datetime'].min())[:10]:>12}  "
            f"{str(df['datetime'].max())[:10]:>12}  "
            f"{ret.mean():>10.6f}  {ret.std():>10.6f}  "
            f"{ret.skew():>8.4f}  {ret.kurtosis():>8.4f}"
        )
    return lines


# ==============================================================
# MAIN
# ==============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_lines  = ["FSLR FINANCIAL DATA EDA REPORT",
                  f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
                  f"Source dir: {SRC_DIR}"]
    dataframes = {}

    for name, path in DATASETS.items():
        print(f"\nRunning EDA: {name} ...")
        df = load(path)
        dataframes[name] = df

        lines = eda_dataset(name, df)
        all_lines.extend(lines)
        for line in lines:
            print(line)

        plot_dataset(name, df, OUT_DIR)

    # Cross-frequency comparison
    cross = cross_freq_summary(dataframes)
    all_lines.extend(cross)
    for line in cross:
        print(line)

    # Save report
    report_text = "\n".join(all_lines)
    REPORT.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved -> {REPORT}")
    print("Task 1 complete.")


if __name__ == "__main__":
    main()
