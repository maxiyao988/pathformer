"""
Task 8 - Multi-Scale Operator Ablation Runner

Runs structured advisor-driven experiment groups by invoking
`task8_multiscale_operator_ablation.py` repeatedly with reproducible configs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT = BASE_DIR / "scripts" / "python" / "task8_multiscale_operator_ablation.py"


def exp_group_1() -> list[dict]:
    return [
        {"exp_name": "all_linear", "model_type": "all_linear", "keep_branches": "short,medium,long"},
        {"exp_name": "all_dwconv", "model_type": "all_dwconv", "keep_branches": "short,medium,long"},
        {"exp_name": "all_attention", "model_type": "all_attention", "keep_branches": "short,medium,long"},
        {"exp_name": "all_pathformer", "model_type": "all_pathformer", "keep_branches": "short,medium,long"},
    ]


def exp_group_2() -> list[dict]:
    return [
        {"exp_name": "hybrid_a", "model_type": "hybrid_a", "keep_branches": "short,medium,long"},
        {"exp_name": "hybrid_b", "model_type": "hybrid_b", "keep_branches": "short,medium,long"},
        {"exp_name": "hybrid_c", "model_type": "hybrid_c", "keep_branches": "short,medium,long"},
    ]


def exp_group_3() -> list[dict]:
    # Scale ablations on top of Hybrid A.
    return [
        {"exp_name": "hybrid_a_all", "model_type": "hybrid_a", "keep_branches": "short,medium,long"},
        {"exp_name": "hybrid_a_short_only", "model_type": "hybrid_a", "keep_branches": "short"},
        {"exp_name": "hybrid_a_medium_only", "model_type": "hybrid_a", "keep_branches": "medium"},
        {"exp_name": "hybrid_a_long_only", "model_type": "hybrid_a", "keep_branches": "long"},
        {"exp_name": "hybrid_a_short_medium", "model_type": "hybrid_a", "keep_branches": "short,medium"},
        {"exp_name": "hybrid_a_medium_long", "model_type": "hybrid_a", "keep_branches": "medium,long"},
        {"exp_name": "hybrid_a_short_long", "model_type": "hybrid_a", "keep_branches": "short,long"},
    ]


def build_experiments(which: str) -> list[dict]:
    if which == "group1":
        return exp_group_1()
    if which == "group2":
        return exp_group_2()
    if which == "group3":
        return exp_group_3()
    if which == "all":
        return exp_group_1() + exp_group_2() + exp_group_3()
    raise ValueError(f"Unsupported group: {which}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=str, default="group1", choices=["group1", "group2", "group3", "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--horizons", nargs="+", default=["5d", "10d", "20d"], choices=["5d", "10d", "20d"])
    p.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "gated"])

    p.add_argument("--short_freqs", type=str, default="hourly,halfday")
    p.add_argument("--medium_freqs", type=str, default="daily")
    p.add_argument("--long_freqs", type=str, default="weekly")

    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])

    p.add_argument("--layer_nums", type=int, default=1)
    p.add_argument("--d_model", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--d_branch", type=int, default=16)
    p.add_argument("--attn_heads", type=int, default=1)
    p.add_argument("--d_fusion", type=int, default=16)

    p.add_argument("--tiny_subset_mode", action="store_true")
    p.add_argument("--tiny_subset_size", type=int, default=64)
    p.add_argument("--tiny_subset_start", type=int, default=0)

    p.add_argument("--tag", type=str, default="v1", help="Tag appended to output file names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args.group)

    print(f"Running group={args.group} with {len(experiments)} experiment configs, seeds={args.seeds}")

    for exp in experiments:
        for seed in args.seeds:
            out_prefix = f"task8_ablation_{args.group}_{exp['exp_name']}_seed{seed}_{args.tag}"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--horizons",
                *args.horizons,
                "--model_type",
                exp["model_type"],
                "--keep_branches",
                exp["keep_branches"],
                "--fusion_type",
                args.fusion_type,
                "--short_freqs",
                args.short_freqs,
                "--medium_freqs",
                args.medium_freqs,
                "--long_freqs",
                args.long_freqs,
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
                "--loss_type",
                args.loss_type,
                "--layer_nums",
                str(args.layer_nums),
                "--d_model",
                str(args.d_model),
                "--d_ff",
                str(args.d_ff),
                "--d_branch",
                str(args.d_branch),
                "--attn_heads",
                str(args.attn_heads),
                "--d_fusion",
                str(args.d_fusion),
                "--out_csv_name",
                f"{out_prefix}.csv",
                "--out_summary_name",
                f"{out_prefix}.txt",
                "--progress_file",
                f"{out_prefix}.log",
            ]

            if args.tiny_subset_mode:
                cmd.extend(
                    [
                        "--tiny_subset_mode",
                        "--tiny_subset_size",
                        str(args.tiny_subset_size),
                        "--tiny_subset_start",
                        str(args.tiny_subset_start),
                    ]
                )

            print("\n=== Running", exp["exp_name"], "seed", seed, "===")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
