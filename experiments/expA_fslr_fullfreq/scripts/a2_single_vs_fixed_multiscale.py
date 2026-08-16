"""
A2: Single-scale (A1 winner) vs Fixed multi-scale ("uniform" combine_mode).

For each frequency and horizon:
  - "single": reuses A1's best patch size for that (freq, horizon), read from
    a1_patch_size_search_results.csv (lowest val_mse). If A1 hasn't been run
    yet, falls back to the middle candidate in PATCH_CANDIDATES as a
    placeholder so this script can still run standalone.
  - "fixed_multiscale": UniformGateAMS over the FULL patch-size candidate
    list for that frequency (equal, non-learned, non-adaptive weight).

Usage:
    python a2_single_vs_fixed_multiscale.py [--horizons 5d,10d,20d] [--freqs hourly,daily]
"""

import argparse
import time
from pathlib import Path

import pandas as pd

import common as C


def load_a1_winner(freq: str, horizon: str) -> int | None:
    a1_csv = C.resolve_output_path("a1_patch_size_search_results.csv")
    if not a1_csv.exists():
        return None
    df = pd.read_csv(a1_csv)
    sub = df[(df["freq"] == freq) & (df["horizon"] == horizon)]
    if sub.empty:
        return None
    best_row = sub.loc[sub["val_mse"].idxmin()]
    return int(best_row["patch_size"])


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

    out_csv = C.resolve_output_path("a2_single_vs_fixed_multiscale_results.csv")
    log_file = C.resolve_output_path("a2_single_vs_fixed_multiscale_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    jobs = [(f, h, mode) for h in horizons for f in freqs for mode in ["single", "fixed_multiscale"]]
    C.log_progress(f"[a2] {len(jobs)} total runs (freq x horizon x mode)", log_file)

    overall_start = time.perf_counter()
    for job_idx, (freq, horizon, mode) in enumerate(jobs, start=1):
        seq_len = C.FREQ_WINDOW[freq]
        candidates = C.PATCH_CANDIDATES[freq]

        if mode == "single":
            winner = load_a1_winner(freq, horizon)
            if winner is None:
                winner = candidates[len(candidates) // 2]
                C.log_progress(
                    f"[a2] WARNING: no A1 result for freq={freq} horizon={horizon}, "
                    f"falling back to placeholder patch={winner}",
                    log_file,
                )
            patch_size_list = [winner]
            combine_mode = "single"
        else:
            patch_size_list = list(candidates)
            combine_mode = "uniform"

        run_name = f"a2_{freq}_{mode}_{horizon}"
        C.log_progress(
            f"[a2] === run {job_idx}/{len(jobs)}: freq={freq} mode={mode} "
            f"patches={patch_size_list} horizon={horizon} ===",
            log_file,
        )

        data = C.load_fslr_multiscale(horizon)

        def build_model(freq=freq, patch_size_list=patch_size_list, combine_mode=combine_mode, seq_len=seq_len):
            return C.SingleFrequencyRegressor(
                freq_name=freq,
                seq_len=seq_len,
                feature_dim=len(C.FEATURES),
                patch_size_list=patch_size_list,
                combine_mode=combine_mode,
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
            "mode": mode,
            "patch_sizes": str(patch_size_list),
            "horizon": horizon,
            "epochs_ran": artifacts.epochs_ran,
            "elapsed_sec": artifacts.elapsed_sec,
            **artifacts.metrics,
        }
        C.save_results_csv([row], out_csv, key_cols=["freq", "mode", "horizon"])
        C.log_progress(
            f"[a2] done freq={freq} mode={mode} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a2] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
