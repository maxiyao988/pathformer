"""
Run Task 8 PathFormer baseline with multiple seeds and build robustness tables.

Outputs:
- dataset/audit/task8_pathformer_multiseed_raw.csv
- dataset/audit/task8_pathformer_multiseed_agg.csv
- dataset/audit/task8_model_comparison_with_pathformer_multiseed.csv
- dataset/audit/task8_model_comparison_with_pathformer_multiseed_summary.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = BASE_DIR / "dataset" / "audit"
PATHFORMER_SCRIPT = BASE_DIR / "scripts" / "python" / "task8_baseline_pathformer.py"


def run_one_seed(args: argparse.Namespace, seed: int) -> Path:
    out_csv_name = f"task8_pathformer_baseline_results_seed{seed}.csv"
    out_txt_name = f"task8_pathformer_baseline_summary_seed{seed}.txt"
    progress_name = f"task8_pathformer_progress_seed{seed}.log"

    cmd = [
        sys.executable,
        str(PATHFORMER_SCRIPT),
        "--horizons",
        *args.horizons,
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(seed),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--layer_nums",
        str(args.layer_nums),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--head_constraint",
        str(args.head_constraint),
        "--head_init_std",
        str(args.head_init_std),
        "--head_init_scale",
        str(args.head_init_scale),
        "--progress_every",
        str(args.progress_every),
        "--loss_type",
        str(args.loss_type),
        "--neck_norm",
        str(args.neck_norm),
        "--progress_file",
        progress_name,
        "--out_csv_name",
        out_csv_name,
        "--out_summary_name",
        out_txt_name,
    ]

    print(f"\n=== Running seed {seed} ===")
    subprocess.run(cmd, check=True)
    return AUDIT_DIR / out_csv_name


def build_robustness_tables(seed_csvs: list[tuple[int, Path]]) -> None:
    rows = []
    for seed, csv_path in seed_csvs:
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        rows.append(df)

    raw = pd.concat(rows, ignore_index=True)
    raw_out = AUDIT_DIR / "task8_pathformer_multiseed_raw.csv"
    raw.to_csv(raw_out, index=False)

    agg = (
        raw.groupby("horizon", as_index=False)
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_corr_mean=("test_corr", "mean"),
            test_corr_std=("test_corr", "std"),
            test_rank_corr_mean=("test_rank_corr", "mean"),
            test_rank_corr_std=("test_rank_corr", "std"),
            test_direction_acc_mean=("test_direction_acc", "mean"),
            test_direction_acc_std=("test_direction_acc", "std"),
        )
        .sort_values("horizon")
    )
    agg_out = AUDIT_DIR / "task8_pathformer_multiseed_agg.csv"
    agg.to_csv(agg_out, index=False)

    linear = pd.read_csv(AUDIT_DIR / "task8_linear_baseline_results.csv")
    vanilla = pd.read_csv(AUDIT_DIR / "task8_vanilla_transformer_results.csv")
    swim = pd.read_csv(AUDIT_DIR / "task8_swim_baseline_results.csv")

    def pick(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        return df[["horizon", "test_mse", "test_mae", "test_corr", "test_direction_acc"]].rename(
            columns={
                "test_mse": f"{model_name}_test_mse",
                "test_mae": f"{model_name}_test_mae",
                "test_corr": f"{model_name}_test_corr",
                "test_direction_acc": f"{model_name}_test_da",
            }
        )

    merged = pick(linear, "Linear")
    merged = merged.merge(pick(vanilla, "Vanilla"), on="horizon", how="inner")
    merged = merged.merge(pick(swim, "SWiM"), on="horizon", how="inner")
    merged = merged.merge(agg, on="horizon", how="inner")

    out_cmp = AUDIT_DIR / "task8_model_comparison_with_pathformer_multiseed.csv"
    merged.to_csv(out_cmp, index=False)

    lines = []
    lines.append("TASK 8 - MODEL COMPARISON WITH PATHFORMER MULTI-SEED")
    lines.append("=" * 72)
    lines.append("")
    for _, r in merged.sort_values("horizon").iterrows():
        lines.append(f"[{r['horizon']}]")
        lines.append(
            f"- Linear: mse={r['Linear_test_mse']:.8f}, mae={r['Linear_test_mae']:.6f}, "
            f"corr={r['Linear_test_corr']:.4f}, da={r['Linear_test_da']:.4f}"
        )
        lines.append(
            f"- Vanilla: mse={r['Vanilla_test_mse']:.8f}, mae={r['Vanilla_test_mae']:.6f}, "
            f"corr={r['Vanilla_test_corr']:.4f}, da={r['Vanilla_test_da']:.4f}"
        )
        lines.append(
            f"- SWiM: mse={r['SWiM_test_mse']:.8f}, mae={r['SWiM_test_mae']:.6f}, "
            f"corr={r['SWiM_test_corr']:.4f}, da={r['SWiM_test_da']:.4f}"
        )
        lines.append(
            f"- PathFormer(mean±std): mse={r['test_mse_mean']:.8f}±{r['test_mse_std']:.8f}, "
            f"mae={r['test_mae_mean']:.6f}±{r['test_mae_std']:.6f}, "
            f"corr={r['test_corr_mean']:.4f}±{r['test_corr_std']:.4f}, "
            f"rank_corr={r['test_rank_corr_mean']:.4f}±{r['test_rank_corr_std']:.4f}, "
            f"da={r['test_direction_acc_mean']:.4f}±{r['test_direction_acc_std']:.4f}"
        )
        lines.append("")

    out_txt = AUDIT_DIR / "task8_model_comparison_with_pathformer_multiseed_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved -> {raw_out}")
    print(f"Saved -> {agg_out}")
    print(f"Saved -> {out_cmp}")
    print(f"Saved -> {out_txt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 21, 42, 3407])
    p.add_argument("--horizons", nargs="+", default=["5d", "10d", "20d"], choices=["5d", "10d", "20d"])
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--layer_nums", type=int, default=1)
    p.add_argument("--d_model", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--head_constraint", type=str, default="none", choices=["none", "tanh_scale"])
    p.add_argument("--head_init_std", type=float, default=1e-3)
    p.add_argument("--head_init_scale", type=float, default=0.2)
    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--neck_norm", type=str, default="layer", choices=["layer", "none"])
    p.add_argument("--progress_every", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    seed_csvs: list[tuple[int, Path]] = []
    for seed in args.seeds:
        csv_path = run_one_seed(args, seed)
        seed_csvs.append((seed, csv_path))

    build_robustness_tables(seed_csvs)


if __name__ == "__main__":
    main()
