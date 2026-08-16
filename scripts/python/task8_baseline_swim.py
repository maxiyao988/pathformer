"""
Task 8 (Part C) - SWiM-style Baseline (5D / 10D / 20D)

This script implements a lightweight shifted-window sequence mixer model
directly in-script, so no external SWiM model file is required.

Outputs are aligned with existing Task 8 format:
- dataset/audit/task8_swim_baseline_results.csv
- dataset/audit/task8_swim_baseline_summary.txt
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset" / "multiscale_dataset"
AUDIT_DIR = BASE_DIR / "dataset" / "audit"

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 25
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    mse: float
    mae: float
    corr: float
    direction_acc: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else np.nan
    direction_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return Metrics(mse=mse, mae=mae, corr=corr, direction_acc=direction_acc)


def split_indices(n: int) -> tuple[slice, slice, slice]:
    train_end = int(n * SPLIT_TRAIN)
    val_end = int(n * (SPLIT_TRAIN + SPLIT_VAL))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def load_sequence_X() -> np.ndarray:
    x_hourly = np.load(DATASET_DIR / "X_hourly.npy")
    x_halfday = np.load(DATASET_DIR / "X_halfday.npy")
    x_daily = np.load(DATASET_DIR / "X_daily.npy")
    x_weekly = np.load(DATASET_DIR / "X_weekly.npy")
    return np.concatenate([x_hourly, x_halfday, x_daily, x_weekly], axis=1).astype(np.float32)


class ShiftedWindowBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int, shift_size: int, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, dim = x.shape
        ws = self.window_size
        pad_len = (ws - (t % ws)) % ws
        if pad_len > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)

        t_pad = x.shape[1]
        n_win = t_pad // ws
        xw = x.view(bsz, n_win, ws, dim).reshape(bsz * n_win, ws, dim)
        yw, _ = self.attn(xw, xw, xw, need_weights=False)
        y = yw.reshape(bsz, n_win, ws, dim).reshape(bsz, t_pad, dim)
        if pad_len > 0:
            y = y[:, :t, :]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)

        if self.shift_size > 0:
            x_norm = torch.roll(x_norm, shifts=-self.shift_size, dims=1)

        x_attn = self._window_attention(x_norm)

        if self.shift_size > 0:
            x_attn = torch.roll(x_attn, shifts=self.shift_size, dims=1)

        x = residual + x_attn
        x = x + self.ffn(self.norm2(x))
        return x


class SWiMStyleRegressor(nn.Module):
    def __init__(self, feature_dim: int = 5, d_model: int = 32, nhead: int = 4, num_layers: int = 4, window_size: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)

        blocks = []
        for i in range(num_layers):
            shift = 0 if (i % 2 == 0) else (window_size // 2)
            blocks.append(
                ShiftedWindowBlock(
                    d_model=d_model,
                    nhead=nhead,
                    window_size=window_size,
                    shift_size=shift,
                    dropout=0.1,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def train_one_horizon(X: np.ndarray, y: np.ndarray) -> dict:
    s_train, s_val, s_test = split_indices(len(X))

    X_train = torch.from_numpy(X[s_train])
    y_train = torch.from_numpy(y[s_train].astype(np.float32))
    X_val = torch.from_numpy(X[s_val])
    y_val = torch.from_numpy(y[s_val].astype(np.float32))
    X_test = torch.from_numpy(X[s_test])
    y_test = torch.from_numpy(y[s_test].astype(np.float32))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SWiMStyleRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(float(criterion(pred, yb).item()))

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        return np.concatenate(trues), np.concatenate(preds)

    yv_true, yv_pred = collect(val_loader)
    yt_true, yt_pred = collect(test_loader)

    val_m = regression_metrics(yv_true, yv_pred)
    test_m = regression_metrics(yt_true, yt_pred)

    return {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "val_mse": val_m.mse,
        "val_mae": val_m.mae,
        "val_corr": val_m.corr,
        "val_direction_acc": val_m.direction_acc,
        "test_mse": test_m.mse,
        "test_mae": test_m.mae,
        "test_corr": test_m.corr,
        "test_direction_acc": test_m.direction_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["5d", "10d", "20d"],
        choices=["5d", "10d", "20d"],
        help="Target horizons to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    X = load_sequence_X()
    all_targets = {
        "5d": np.load(DATASET_DIR / "y_5d.npy"),
        "10d": np.load(DATASET_DIR / "y_10d.npy"),
        "20d": np.load(DATASET_DIR / "y_20d.npy"),
    }
    targets = {k: all_targets[k] for k in args.horizons}

    rows = []
    for horizon, y in targets.items():
        metrics = train_one_horizon(X, y)
        row = {"horizon": horizon, "samples": int(len(X)), **metrics}
        rows.append(row)
        print(
            f"[{horizon}] test_mse={metrics['test_mse']:.8f} "
            f"test_mae={metrics['test_mae']:.6f} test_corr={metrics['test_corr']:.4f} "
            f"test_da={metrics['test_direction_acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    out_csv = AUDIT_DIR / "task8_swim_baseline_results.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        keep_old = old[~old["horizon"].isin(df["horizon"])].copy()
        df = pd.concat([keep_old, df], ignore_index=True)
    df = df.sort_values("horizon")
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("TASK 8 - SWIM-STYLE BASELINE (MULTI-HORIZON)")
    lines.append("=" * 72)
    lines.append(f"device: {DEVICE}")
    lines.append(f"epochs<= {MAX_EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}")
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['horizon']}: test_mse={r['test_mse']:.8f}, test_mae={r['test_mae']:.6f}, "
            f"test_corr={r['test_corr']:.4f}, test_da={r['test_direction_acc']:.4f}"
        )

    out_txt = AUDIT_DIR / "task8_swim_baseline_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_txt}")


if __name__ == "__main__":
    main()
