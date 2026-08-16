# =====================================================
# MULTI-SCALE STOCK FORECASTING CONFIG
# =====================================================

CONFIG = {

    # =================================================
    # STOCK
    # =================================================

    "ticker": "FSLR",

    # =================================================
    # ANCHOR SCALE
    # =================================================

    "anchor_scale": "daily",

    # options:
    # hourly
    # halfday
    # daily
    # weekly

    # =================================================
    # SCALE SWITCHES
    # =================================================

    "use_hourly": True,
    "use_halfday": True,
    "use_daily": True,
    "use_weekly": True,

    # =================================================
    # FEATURES
    # =================================================

    "features": [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ],

    # =================================================
    # LOOKBACK WINDOWS
    # =================================================

    "hourly_window": 24,

    "halfday_window": 20,

    "daily_window": 90,

    "weekly_window": 26,

    # =================================================
    # TARGET
    # =================================================

    "target_horizon": 20,

    # 1
    # 5
    # 10
    # 20
    # 60

    "target_type": "return",

    # return
    # excess_return

    "target_method": "log_return",

    # log_return
    # simple_return

    # =================================================
    # DATA ALIGNMENT
    # =================================================

    "alignment_method": "end_timestamp",

    # future:
    # end_timestamp
    # nearest_timestamp
    # interpolation

    # =================================================
    # SPLIT
    # =================================================

    "train_ratio": 0.8,

    "val_ratio": 0.1,

    "test_ratio": 0.1,

    # =================================================
    # RANDOMNESS
    # =================================================

    "seed": 42,

    # =================================================
    # PATHS
    # =================================================

    "hourly_path":
        "dataset/processed/FSLR_hourly.csv",

    "halfday_path":
        "dataset/merged/FSLR_halfday_full.csv",

    "daily_path":
        "dataset/processed/FSLR_daily.csv",

    "weekly_path":
        "dataset/processed/FSLR_weekly.csv",

    # =================================================
    # OUTPUT
    # =================================================

    "output_dir":
        "dataset/multiscale_dataset"

}