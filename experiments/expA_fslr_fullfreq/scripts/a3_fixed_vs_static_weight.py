"""
A3: Fixed multi-scale ("uniform") vs Static learnable scale weight ("static_weight").

Both use the FULL patch-size candidate list per frequency; the only
difference is how the patch-size experts are combined:
  - "fixed_multiscale": equal 1/n weight, not learned (UniformGateAMS).
  - "static_weight": one global softmax weight vector, learned via backprop
    but identical for every sample (StaticWeightAMS) -- i.e. the model can
    learn "daily patch=10 is generally more useful than patch=30" but cannot
    adapt that choice per-sample/per-market-regime (that's what A4 tests).

Usage:
    python a3_fixed_vs_static_weight.py [--horizons 5d,10d,20d] [--freqs hourly,daily]
"""

import argparse
import time

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

    out_csv = C.resolve_output_path("a3_fixed_vs_static_weight_results.csv")
    log_file = C.resolve_output_path("a3_fixed_vs_static_weight_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    modes = [("fixed_multiscale", "uniform"), ("static_weight", "static_weight")]
    jobs = [(f, h, label, combine_mode) for h in horizons for f in freqs for label, combine_mode in modes]
    C.log_progress(f"[a3] {len(jobs)} total runs (freq x horizon x mode)", log_file)

    overall_start = time.perf_counter()
    for job_idx, (freq, horizon, label, combine_mode) in enumerate(jobs, start=1):
        seq_len = C.FREQ_WINDOW[freq]
        patch_size_list = list(C.PATCH_CANDIDATES[freq])

        run_name = f"a3_{freq}_{label}_{horizon}"
        C.log_progress(
            f"[a3] === run {job_idx}/{len(jobs)}: freq={freq} mode={label} "
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
            "mode": label,
            "patch_sizes": str(patch_size_list),
            "horizon": horizon,
            "epochs_ran": artifacts.epochs_ran,
            "elapsed_sec": artifacts.elapsed_sec,
            **artifacts.metrics,
        }
        C.save_results_csv([row], out_csv, key_cols=["freq", "mode", "horizon"])
        C.log_progress(
            f"[a3] done freq={freq} mode={label} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a3] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
