"""
A6: Dual-attention ablation (intra-patch vs inter-patch attention).

Note: the 5 commonly-used variant names map to only 3 unique architectures
(reported under all 5 names for clarity/comparability, but only 3 are
actually trained):

    label          use_intra  use_inter   (== equivalent name)
    -------------  ---------  ---------   ---------------------
    full_dual      True       True
    intra_only     True       False       (== "no_inter")
    no_inter       True       False       (== "intra_only")
    inter_only     False      True        (== "no_intra")
    no_intra       False      True        (== "inter_only")

Base architecture fixed to the "best" recipe from A1-A5: full-frequency
(hourly+halfday+daily+weekly) inputs, each frequency's patch experts combined
via combine_mode="adaptive_router" over the full patch-size candidate list,
concat fusion across frequencies. Only use_intra/use_inter vary.

Usage:
    python a6_dual_attention_ablation.py [--horizons 5d,10d,20d]
"""

import argparse
import time

import common as C

# unique_label -> (use_intra, use_inter)
UNIQUE_CONFIGS = {
    "full_dual": (True, True),
    "intra_only": (True, False),
    "inter_only": (False, True),
}

# display label -> unique_label it is equivalent to
DISPLAY_TO_UNIQUE = {
    "full_dual": "full_dual",
    "intra_only": "intra_only",
    "no_inter": "intra_only",
    "inter_only": "inter_only",
    "no_intra": "inter_only",
}

FULL_FREQS = ["hourly", "halfday", "daily", "weekly"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="5d,10d,20d")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "adaptive_router"])
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--d_branch", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=C.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    args = parser.parse_args()

    horizons = args.horizons.split(",")

    out_csv = C.resolve_output_path("a6_dual_attention_ablation_results.csv")
    log_file = C.resolve_output_path("a6_dual_attention_ablation_progress.log")
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    C.set_seed(C.SEED)

    jobs = [(h, unique_label) for h in horizons for unique_label in UNIQUE_CONFIGS]
    C.log_progress(f"[a6] {len(jobs)} unique training runs x horizon (5 display labels reported)", log_file)

    overall_start = time.perf_counter()
    for job_idx, (horizon, unique_label) in enumerate(jobs, start=1):
        use_intra, use_inter = UNIQUE_CONFIGS[unique_label]
        run_name = f"a6_{unique_label}_{horizon}"
        C.log_progress(
            f"[a6] === run {job_idx}/{len(jobs)}: config={unique_label} "
            f"use_intra={use_intra} use_inter={use_inter} horizon={horizon} ===",
            log_file,
        )

        data = C.load_fslr_multiscale(horizon)

        def build_model(use_intra=use_intra, use_inter=use_inter):
            return C.MultiFrequencyRegressor(
                freq_names=FULL_FREQS,
                seq_len_map={f: C.FREQ_WINDOW[f] for f in FULL_FREQS},
                feature_dim=len(C.FEATURES),
                patch_size_map={f: list(C.PATCH_CANDIDATES[f]) for f in FULL_FREQS},
                combine_mode_map={f: "adaptive_router" for f in FULL_FREQS},
                d_model=args.d_model,
                d_ff=args.d_ff,
                d_branch=args.d_branch,
                fusion=args.fusion,
                use_intra=use_intra,
                use_inter=use_inter,
            )

        artifacts = C.train_one_run(
            build_model,
            freq_inputs=FULL_FREQS,
            data=data,
            max_epochs=args.max_epochs,
            patience=args.patience,
            run_name=run_name,
            progress_file=log_file,
        )

        display_labels = [lbl for lbl, uniq in DISPLAY_TO_UNIQUE.items() if uniq == unique_label]
        for display_label in display_labels:
            row = {
                "display_label": display_label,
                "unique_config": unique_label,
                "use_intra": use_intra,
                "use_inter": use_inter,
                "horizon": horizon,
                "epochs_ran": artifacts.epochs_ran,
                "elapsed_sec": artifacts.elapsed_sec,
                **artifacts.metrics,
            }
            C.save_results_csv([row], out_csv, key_cols=["display_label", "horizon"])

        C.log_progress(
            f"[a6] done config={unique_label} horizon={horizon} "
            f"test_mse={artifacts.metrics['test_mse']:.6f} "
            f"test_corr={artifacts.metrics['test_corr']:.4f} "
            f"(overall elapsed {C.format_seconds(time.perf_counter() - overall_start)})",
            log_file,
        )

    C.log_progress(f"[a6] ALL DONE. Results at {out_csv}", log_file)


if __name__ == "__main__":
    main()
