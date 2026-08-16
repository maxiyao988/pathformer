"""
A1: Single-frequency patch-size search (FSLR case study).

For each frequency (hourly/halfday/daily/weekly) and each advisor-specified
patch-size candidate, train a single-expert PathFormer branch (combine_mode
="single": one patch size, trivial gate) to predict each return horizon
(5d/10d/20d), using ONLY that one frequency as input. This is the baseline
"which single scale works best, per frequency" search that A2 (fixed
multi-scale) and A3/A4 (learned weighting) are compared against.

Usage:
    python a1_patch_size_search.py [--horizons 5d,10d,20d] [--freqs hourly,daily]
"""

import argparse
import time
from pathlib import Path

import common as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="5d,10d,20d")
    parser.add_argument("--freqs", type=str, default="hourly,halfday,daily,weekly")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--d_branch", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    args = parser.parse_args()

    horizons = args.horizons.split(",")
    freqs = args.freqs.split(",")

    out_csv = C.resolve_output_path("a1_patch_size_search_results.csv")
    log_file = C.resolve_output_path("a1_patch_size_search_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    jobs = [(f, p, h) for h in horizons for f in freqs for p in C.PATCH_CANDIDATES[f]]
    C.log_progress(f"[a1] {len(jobs)} total runs (freq x patch x horizon)", log_file)

    overall_start = time.perf_counter()
    for job_idx, (freq, patch, horizon) in enumerate(jobs, start=1):
        run_name = f"a1_{freq}_p{patch}_{horizon}"
        C.log_progress(
            f"[a1] === run {job_idx}/{len(jobs)}: freq={freq} patch={patch} horizon={horizon} ===",
            log_file,
        )

        data = C.load_fslr_multiscale(horizon)
        seq_len = C.FREQ_WINDOW[freq]

        def build_model(freq=freq, patch=patch, seq_len=seq_len):
            return C.SingleFrequencyRegressor(
                freq_name=freq,
                seq_len=seq_len,
                feature_dim=len(C.FEATURES),
                patch_size_list=[patch],
                combine_mode="single",
                d_model=args.d_model,
                d_ff=args.d_ff,
                d_branch=args.d_branch,
            )

        artifacts = C.train_one_run(
            build_model,
            freq_inputs=[freq],
            data=data,
            max_epochs=args.max_epochs,
            patience=args.patience,
            run_name=run_name,
            progress_file=log_file,
        )

        row = {
            "freq": freq,
            "patch_size": patch,
            "horizon": horizon,
            "epochs_ran": artifacts.epochs_ran,
            "elapsed_sec": artifacts.elapsed_sec,
            **artifacts.metrics,
        }
        C.save_results_csv([row], out_csv, key_cols=["freq", "patch_size", "horizon"])
        C.log_progress(
            f"[a1] done freq={freq} patch={patch} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"test_rank_corr={artifacts.metrics['test_rank_corr']:.4f} "
            f"test_direction_acc={artifacts.metrics['test_direction_acc']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a1] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
