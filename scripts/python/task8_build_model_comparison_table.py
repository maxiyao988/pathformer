"""
Build Task 8 model comparison table across:
- Linear
- Vanilla Transformer
- SWiM-style
- PathFormer
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = BASE_DIR / "dataset" / "audit"


MODEL_FILES = {
    "Linear": AUDIT_DIR / "task8_linear_baseline_results.csv",
    "Vanilla": AUDIT_DIR / "task8_vanilla_transformer_results.csv",
    "SWiM": AUDIT_DIR / "task8_swim_baseline_results.csv",
    "PathFormer": AUDIT_DIR / "task8_pathformer_baseline_results.csv",
}


def main() -> None:
    rows = []
    missing = []

    for model_name, file_path in MODEL_FILES.items():
        if not file_path.exists():
            missing.append(str(file_path))
            continue
        df = pd.read_csv(file_path)
        df["model"] = model_name
        rows.append(df)

    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))

    all_df = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "model",
        "horizon",
        "test_mse",
        "test_mae",
        "test_corr",
        "test_direction_acc",
    ]
    all_df = all_df[keep_cols].copy()

    horizon_order = {"5d": 0, "10d": 1, "20d": 2}
    model_order = {"Linear": 0, "Vanilla": 1, "SWiM": 2, "PathFormer": 3}
    all_df["_h"] = all_df["horizon"].map(horizon_order)
    all_df["_m"] = all_df["model"].map(model_order)
    all_df = all_df.sort_values(["_h", "_m"]).drop(columns=["_h", "_m"])

    out_long = AUDIT_DIR / "task8_model_comparison_long.csv"
    all_df.to_csv(out_long, index=False)

    pivot = all_df.pivot(index="horizon", columns="model", values=["test_mse", "test_mae", "test_corr", "test_direction_acc"])
    pivot = pivot.sort_index(key=lambda idx: idx.map(horizon_order))
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()

    out_wide = AUDIT_DIR / "task8_model_comparison_table.csv"
    pivot.to_csv(out_wide, index=False)

    lines = []
    lines.append("TASK 8 - MODEL COMPARISON TABLE")
    lines.append("=" * 72)
    lines.append("Models: Linear, Vanilla, SWiM, PathFormer")
    lines.append("Horizons: 5d, 10d, 20d")
    lines.append("")

    for horizon in ["5d", "10d", "20d"]:
        sub = all_df[all_df["horizon"] == horizon]
        lines.append(f"[{horizon}]")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['model']}: mse={r['test_mse']:.8f}, mae={r['test_mae']:.6f}, "
                f"corr={r['test_corr']:.4f}, da={r['test_direction_acc']:.4f}"
            )
        lines.append("")

    out_txt = AUDIT_DIR / "task8_model_comparison_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved -> {out_long}")
    print(f"Saved -> {out_wide}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":
    main()
