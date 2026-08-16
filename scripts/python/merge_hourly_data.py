import pandas as pd

# ==================================================
# Bloomberg
# ==================================================

bbg = pd.read_csv(
    "dataset/multiscale/FSLR_hourly_clean_bloomberg.csv",
    encoding="utf-8",
    parse_dates=["datetime"],
    dayfirst=True
)

# 不需要 rename Date => datetime
# bbg["datetime"] 已经是 datetime 类型了

# Bloomberg 本来就是纽约时间
# 直接当作 naive datetime

# ==================================================
# Yahoo
# ==================================================

yahoo = pd.read_csv(
    "dataset/multiscale/FSLR_hourly_clean.csv"
)

yahoo["datetime"] = pd.to_datetime(
    yahoo["datetime"]
)

# 把UTC转换成纽约时间

if yahoo["datetime"].dt.tz is not None:

    yahoo["datetime"] = (
        yahoo["datetime"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

# ==================================================
# Merge
# ==================================================

df = pd.concat(
    [bbg, yahoo],
    axis=0
)

# ==================================================
# Sort
# ==================================================

df = df.sort_values(
    "datetime"
)

# ==================================================
# Remove duplicates
# ==================================================

before = len(df)

df = df.drop_duplicates(
    subset=["datetime"],
    keep="first"
)

after = len(df)

print("\nDuplicates Removed:")
print(before - after)

# ==================================================
# Final Check
# ==================================================

print("\nFinal Shape:")
print(df.shape)

print("\nStart:")
print(df["datetime"].min())

print("\nEnd:")
print(df["datetime"].max())

print("\nHead:")
print(df.head())

print("\nTail:")
print(df.tail())

# ==================================================
# Save
# ==================================================

output_file = (
    "dataset/multiscale/"
    "FSLR_hourly_clean_full.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)