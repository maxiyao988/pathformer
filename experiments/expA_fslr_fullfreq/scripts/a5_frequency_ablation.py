"""
A5: Frequency-count / frequency-combination ablation.

8 groups (as specified by the advisor), fixing the internal per-frequency
patch-scale handling to "adaptive_router" (A4's winner) over the full
patch-size candidate list, and varying ONLY which frequencies are fed in and
how the resulting per-frequency branch vectors are fused:

    group_name        active frequencies
    ----------------  -------------------------------
    single_hourly     hourly
    single_halfday    halfday
    single_daily      daily
    single_weekly     weekly
    double_short      hourly + halfday
    double_lowfreq    daily + weekly
    double_mixed      hourly + daily
    full_frequency    hourly + halfday + daily + weekly

Fusion defaults to "concat" (isolates the single variable under test: number/
choice of input frequencies). Pass --fusion adaptive_router to additionally
test the cross-frequency adaptive router fusion (CrossFreqRouter) on top.

Usage:
    python a5_frequency_ablation.py [--horizons 5d,10d,20d] [--fusion concat]
"""

import argparse
import time

import common as C

GROUPS = {
    "single_hourly": ["hourly"],
    "single_halfday": ["halfday"],
    "single_daily": ["daily"],
    "single_weekly": ["weekly"],
    "double_short": ["hourly", "halfday"],
    "double_lowfreq": ["daily", "weekly"],
    "double_mixed": ["hourly", "daily"],
    "full_frequency": ["hourly", "halfday", "daily", "weekly"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="5d,10d,20d")
    parser.add_argument("--groups", type=str, default=",".join(GROUPS.keys()))
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "adaptive_router"])
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--d_branch", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    args = parser.parse_args()

    horizons = args.horizons.split(",")
    groups = args.groups.split(",")

    out_csv = C.resolve_output_path(f"a5_frequency_ablation_{args.fusion}_results.csv")
    log_file = C.resolve_output_path(f"a5_frequency_ablation_{args.fusion}_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    jobs = [(g, h) for h in horizons for g in groups]
    C.log_progress(f"[a5] {len(jobs)} total runs (group x horizon), fusion={args.fusion}", log_file)

    overall_start = time.perf_counter()
    for job_idx, (group, horizon) in enumerate(jobs, start=1):
        freq_names = GROUPS[group]
        run_name = f"a5_{group}_{args.fusion}_{horizon}"
        C.log_progress(
            f"[a5] === run {job_idx}/{len(jobs)}: group={group} freqs={freq_names} "
            f"fusion={args.fusion} horizon={horizon} ===",
            log_file,
        )

        data = C.load_fslr_multiscale(horizon)

        def build_model(freq_names=freq_names):
            return C.MultiFrequencyRegressor(
                freq_names=freq_names,
                seq_len_map={f: C.FREQ_WINDOW[f] for f in freq_names},
                feature_dim=len(C.FEATURES),
                patch_size_map={f: list(C.PATCH_CANDIDATES[f]) for f in freq_names},
                combine_mode_map={f: "adaptive_router" for f in freq_names},
                d_model=args.d_model,
                d_ff=args.d_ff,
                d_branch=args.d_branch,
                fusion=args.fusion,
            )

        artifacts = C.train_one_run(
            build_model,
            freq_inputs=freq_names,
            data=data,
            max_epochs=args.max_epochs,
            patience=args.patience,
            run_name=run_name,
            progress_file=log_file,
        )

        row = {
            "group": group,
            "freqs": "+".join(freq_names),
            "num_freqs": len(freq_names),
            "fusion": args.fusion,
            "horizon": horizon,
            "epochs_ran": artifacts.epochs_ran,
            "elapsed_sec": artifacts.elapsed_sec,
            **artifacts.metrics,
        }
        C.save_results_csv([row], out_csv, key_cols=["group", "fusion", "horizon"])
        C.log_progress(
            f"[a5] done group={group} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a5] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
