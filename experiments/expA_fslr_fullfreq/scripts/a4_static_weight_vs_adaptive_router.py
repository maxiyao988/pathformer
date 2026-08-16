"""
A4: Static learnable weight ("static_weight") vs Adaptive router ("adaptive_router").

This is the core comparison for the advisor's requested contribution: does
letting the router weight patch-size experts AS A FUNCTION OF THE CURRENT
INPUT WINDOW (adaptive_router = plain AMS's native noisy top-k gating) beat a
single global, non-adaptive learned weight (static_weight = StaticWeightAMS)?

Both use the FULL patch-size candidate list per frequency.

For combine_mode="adaptive_router", this script ALSO saves the per-test-sample
router gate weights (one row per test sample x patch-size, joined with
anchor_date from meta.csv) to a1_router_weights_<freq>_<horizon>.csv --
these are the inputs Experiment A7 needs for the regime-interpretation
analysis.

Usage:
    python a4_static_weight_vs_adaptive_router.py [--horizons 5d,10d,20d] [--freqs hourly,daily]
"""

import argparse
import time

import pandas as pd

import common as C


def save_router_weights(freq: str, horizon: str, data: dict, gates) -> None:
    n_total = len(data["y"])
    _, _, s_test = C.split_indices(n_total)
    meta_test = data["meta"].iloc[s_test].reset_index(drop=True)
    patch_sizes = C.PATCH_CANDIDATES[freq]
    df = pd.DataFrame(gates, columns=[f"gate_patch_{p}" for p in patch_sizes])
    df.insert(0, "anchor_date", meta_test["anchor_date"].values)
    out_path = C.resolve_output_path(f"a4_router_weights_{freq}_{horizon}.csv")
    df.to_csv(out_path, index=False)


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

    out_csv = C.resolve_output_path("a4_static_weight_vs_adaptive_router_results.csv")
    log_file = C.resolve_output_path("a4_static_weight_vs_adaptive_router_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    modes = [("static_weight", "static_weight"), ("adaptive_router", "adaptive_router")]
    jobs = [(f, h, label, combine_mode) for h in horizons for f in freqs for label, combine_mode in modes]
    C.log_progress(f"[a4] {len(jobs)} total runs (freq x horizon x mode)", log_file)

    overall_start = time.perf_counter()
    for job_idx, (freq, horizon, label, combine_mode) in enumerate(jobs, start=1):
        seq_len = C.FREQ_WINDOW[freq]
        patch_size_list = list(C.PATCH_CANDIDATES[freq])

        run_name = f"a4_{freq}_{label}_{horizon}"
        C.log_progress(
            f"[a4] === run {job_idx}/{len(jobs)}: freq={freq} mode={label} "
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

        if combine_mode == "adaptive_router" and artifacts.test_gates is not None:
            save_router_weights(freq, horizon, data, artifacts.test_gates["patch"])

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
            f"[a4] done freq={freq} mode={label} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a4] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
