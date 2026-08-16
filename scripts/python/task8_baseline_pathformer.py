"""
Task 8 (Part D) - PathFormer Baseline (5D / 10D / 20D)

This script reuses the repository PathFormer backbone and adds a scalar
regression head for multi-horizon return prediction.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pathformer.models.PathFormer import Model as PathFormerBackbone


DATASET_DIR = BASE_DIR / "dataset" / "multiscale_dataset"
AUDIT_DIR = BASE_DIR / "dataset" / "audit"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 8
WEIGHT_DECAY = 0.0
LR = 3e-4

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
    rank_corr: float
    direction_acc: float


@dataclass
class TrainArtifacts:
    metrics: dict
    epochs_ran: int
    elapsed_sec: float


def format_seconds(seconds: float) -> str:
    sec = int(max(0, round(seconds)))
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def log_progress(msg: str, log_file: Path | None) -> None:
    print(msg)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_output_path(name: str) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    return AUDIT_DIR / p


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else np.nan
    if len(y_true) > 1 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        rank_corr = float(spearmanr(y_pred, y_true)[0])
    else:
        rank_corr = np.nan
    direction_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return Metrics(mse=mse, mae=mae, corr=corr, rank_corr=rank_corr, direction_acc=direction_acc)


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


class PathFormerRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        layer_nums: int,
        d_model: int,
        d_ff: int,
        head_constraint: str,
        head_init_std: float,
        head_init_scale: float,
        neck_norm: str = "layer",
    ):
        super().__init__()
        cfg = SimpleNamespace(
            layer_nums=layer_nums,
            num_nodes=feature_dim,
            pred_len=1,
            seq_len=seq_len,
            k=2,
            num_experts_list=[4] * layer_nums,
            patch_size_list=[[16, 8, 5, 4] for _ in range(layer_nums)],
            d_model=d_model,
            d_ff=d_ff,
            residual_connection=1,
            revin=True,
            gpu=0,
            batch_norm=False,
        )
        self.backbone = PathFormerBackbone(cfg)
        # Teacher requirement: replace BatchNorm with LayerNorm. The backbone's
        # internal BatchNorm is disabled via cfg.batch_norm=False; a LayerNorm is
        # applied here on the pooled feature vector before the regression head.
        self.neck_norm = neck_norm
        if neck_norm == "layer":
            self.neck = nn.LayerNorm(feature_dim)
        else:
            self.neck = nn.Identity()
        self.head = nn.Linear(feature_dim, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)

        self.head_constraint = head_constraint
        # softplus(log_scale) ensures a positive output scale.
        init_log_scale = float(np.log(np.exp(max(1e-6, head_init_scale)) - 1.0))
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _aux = self.backbone(x)
        vec = out[:, -1, :]
        vec = self.neck(vec)
        raw = self.head(vec).squeeze(-1)
        if self.head_constraint == "tanh_scale":
            scale = torch.nn.functional.softplus(self.log_scale) + 1e-6
            return torch.tanh(raw) * scale
        return raw

    def forward_debug(self, x: torch.Tensor) -> dict:
        """Return intermediate tensors to diagnose where cross-sample variance
        collapses: backbone pooled vec (pre-neck), post-neck vec, and raw head
        output (pre head_constraint).
        """
        out, _aux = self.backbone(x)
        vec_raw = out[:, -1, :]
        vec_neck = self.neck(vec_raw)
        raw = self.head(vec_neck).squeeze(-1)
        return {"vec_raw": vec_raw, "vec_neck": vec_neck, "raw": raw}


def train_one_horizon(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    layer_nums: int,
    d_model: int,
    d_ff: int,
    y_standardize: bool,
    head_constraint: str,
    head_init_std: float,
    head_init_scale: float,
    horizon_name: str,
    progress_file: Path | None,
    progress_every: int,
    loss_type: str,
    huber_delta: float,
    tiny_subset_mode: bool,
    tiny_subset_size: int,
    tiny_subset_start: int,
    neck_norm: str,
    debug_backbone: bool = False,
) -> TrainArtifacts:
    if tiny_subset_mode:
        start = max(0, min(tiny_subset_start, max(0, len(X) - 1)))
        end = min(len(X), start + max(1, tiny_subset_size))
        X = X[start:end]
        y = y[start:end]
        log_progress(
            f"[{horizon_name}] tiny subset mode enabled: using CONSECUTIVE samples [{start}:{end}) size={end - start}",
            progress_file,
        )

    if len(X) < 20:
        raise ValueError(f"[{horizon_name}] not enough samples after filtering: {len(X)}")

    s_train, s_val, s_test = split_indices(len(X))

    X_train = torch.from_numpy(X[s_train])
    X_val = torch.from_numpy(X[s_val])
    X_test = torch.from_numpy(X[s_test])

    y_train_np = y[s_train].astype(np.float32)
    y_val_np = y[s_val].astype(np.float32)
    y_test_np = y[s_test].astype(np.float32)

    y_mean = float(np.mean(y_train_np))
    y_std = float(np.std(y_train_np))
    if y_std < 1e-8:
        y_std = 1.0

    if y_standardize:
        y_train_use = (y_train_np - y_mean) / y_std
        y_val_use = (y_val_np - y_mean) / y_std
        y_test_use = (y_test_np - y_mean) / y_std
    else:
        y_train_use = y_train_np
        y_val_use = y_val_np
        y_test_use = y_test_np

    y_train = torch.from_numpy(y_train_use)
    y_val = torch.from_numpy(y_val_use)
    y_test = torch.from_numpy(y_test_use)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = PathFormerRegressor(
        seq_len=X.shape[1],
        feature_dim=X.shape[2],
        layer_nums=layer_nums,
        d_model=d_model,
        d_ff=d_ff,
        head_constraint=head_constraint,
        head_init_std=head_init_std,
        head_init_scale=head_init_scale,
        neck_norm=neck_norm,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    horizon_start = time.perf_counter()
    running_epoch_secs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
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

        if debug_backbone and ((epoch % max(1, progress_every) == 0) or (epoch == 1) or (epoch == max_epochs)):
            with torch.no_grad():
                xb_dbg, _ = next(iter(train_loader))
                dbg = model.forward_debug(xb_dbg.to(DEVICE))
                vec_raw_std = float(dbg["vec_raw"].std(dim=0).mean().item())
                vec_neck_std = float(dbg["vec_neck"].std(dim=0).mean().item())
                raw_std = float(dbg["raw"].std().item())
            log_progress(
                (
                    f"[{horizon_name}] DEBUG epoch {epoch:02d}: "
                    f"backbone_vec_std(cross-sample, per-dim avg)={vec_raw_std:.6e} "
                    f"post_neck_vec_std={vec_neck_std:.6e} "
                    f"head_raw_output_std={raw_std:.6e}"
                ),
                progress_file,
            )

        epoch_sec = time.perf_counter() - epoch_start
        running_epoch_secs.append(epoch_sec)
        mean_epoch_sec = float(np.mean(running_epoch_secs))
        remain_epochs = max(0, max_epochs - epoch)
        eta_sec = remain_epochs * mean_epoch_sec

        if (epoch % max(1, progress_every) == 0) or (epoch == 1) or (epoch == max_epochs) or (bad_epochs >= patience):
            log_progress(
                (
                    f"[{horizon_name}] epoch {epoch:02d}/{max_epochs:02d} "
                    f"val_mse={mean_val:.8f} best_val_mse={best_val:.8f} "
                    f"bad_epochs={bad_epochs}/{patience} "
                    f"epoch_time={format_seconds(epoch_sec)} eta={format_seconds(eta_sec)}"
                ),
                progress_file,
            )

        if bad_epochs >= patience:
            log_progress(f"[{horizon_name}] early stopping at epoch {epoch}", progress_file)
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

    ytr_true_use, ytr_pred_use = collect(train_eval_loader)
    yv_true_use, yv_pred_use = collect(val_loader)
    yt_true_use, yt_pred_use = collect(test_loader)

    if y_standardize:
        ytr_true = ytr_true_use * y_std + y_mean
        ytr_pred = ytr_pred_use * y_std + y_mean
        yv_true = yv_true_use * y_std + y_mean
        yv_pred = yv_pred_use * y_std + y_mean
        yt_true = yt_true_use * y_std + y_mean
        yt_pred = yt_pred_use * y_std + y_mean
    else:
        ytr_true, ytr_pred = ytr_true_use, ytr_pred_use
        yv_true, yv_pred = yv_true_use, yv_pred_use
        yt_true, yt_pred = yt_true_use, yt_pred_use

    train_m = regression_metrics(ytr_true, ytr_pred)
    val_m = regression_metrics(yv_true, yv_pred)
    test_m = regression_metrics(yt_true, yt_pred)

    elapsed_sec = time.perf_counter() - horizon_start
    metrics = {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_mse": train_m.mse,
        "train_mae": train_m.mae,
        "train_pred_std": float(np.std(ytr_pred)),
        "train_true_std": float(np.std(ytr_true)),
        "val_mse": val_m.mse,
        "val_mae": val_m.mae,
        "val_corr": val_m.corr,
        "val_rank_corr": val_m.rank_corr,
        "val_direction_acc": val_m.direction_acc,
        "test_mse": test_m.mse,
        "test_mae": test_m.mae,
        "test_corr": test_m.corr,
        "test_rank_corr": test_m.rank_corr,
        "test_direction_acc": test_m.direction_acc,
        "val_pred_std": float(np.std(yv_pred)),
        "val_true_std": float(np.std(yv_true)),
        "test_pred_std": float(np.std(yt_pred)),
        "test_true_std": float(np.std(yt_true)),
        "y_train_mean": y_mean,
        "y_train_std": y_std,
    }
    return TrainArtifacts(metrics=metrics, epochs_ran=len(running_epoch_secs), elapsed_sec=elapsed_sec)


def train_static_baseline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    horizon_name: str,
    baseline_mode: str,
    progress_file: Path | None,
) -> TrainArtifacts:
    if baseline_mode not in {"zero", "train_mean"}:
        raise ValueError(f"Unsupported baseline mode: {baseline_mode}")

    s_train, s_val, s_test = split_indices(len(X))
    y_train = y[s_train].astype(np.float32)
    y_val = y[s_val].astype(np.float32)
    y_test = y[s_test].astype(np.float32)

    if baseline_mode == "zero":
        pred_value = 0.0
    else:
        pred_value = float(np.mean(y_train))

    yv_pred = np.full_like(y_val, fill_value=pred_value, dtype=np.float32)
    yt_pred = np.full_like(y_test, fill_value=pred_value, dtype=np.float32)

    val_m = regression_metrics(y_val, yv_pred)
    test_m = regression_metrics(y_test, yt_pred)

    metrics = {
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_mse": float(np.mean((y_train - float(np.mean(y_train))) ** 2)) if baseline_mode == "train_mean" else float(np.mean(y_train ** 2)),
        "train_mae": float(np.mean(np.abs(y_train - pred_value))),
        "train_pred_std": 0.0,
        "train_true_std": float(np.std(y_train)),
        "val_mse": val_m.mse,
        "val_mae": val_m.mae,
        "val_corr": val_m.corr,
        "val_rank_corr": val_m.rank_corr,
        "val_direction_acc": val_m.direction_acc,
        "test_mse": test_m.mse,
        "test_mae": test_m.mae,
        "test_corr": test_m.corr,
        "test_rank_corr": test_m.rank_corr,
        "test_direction_acc": test_m.direction_acc,
        "val_pred_std": float(np.std(yv_pred)),
        "val_true_std": float(np.std(y_val)),
        "test_pred_std": float(np.std(yt_pred)),
        "test_true_std": float(np.std(y_test)),
        "y_train_mean": float(np.mean(y_train)),
        "y_train_std": float(np.std(y_train)),
    }

    log_progress(
        (
            f"[{horizon_name}] static baseline={baseline_mode} "
            f"pred_value={pred_value:.8f} test_mse={metrics['test_mse']:.8f} "
            f"test_mae={metrics['test_mae']:.6f}"
        ),
        progress_file,
    )
    return TrainArtifacts(metrics=metrics, epochs_ran=0, elapsed_sec=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["5d", "10d", "20d"],
        choices=["5d", "10d", "20d"],
        help="Target horizons to run",
    )
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--layer_nums", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=16)
    parser.add_argument("--head_constraint", type=str, default="none", choices=["none", "tanh_scale"])
    parser.add_argument("--head_init_std", type=float, default=0.02)
    parser.add_argument("--head_init_scale", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.35)
    parser.add_argument("--neck_norm", type=str, default="layer", choices=["layer", "none"])
    parser.add_argument("--baseline_mode", type=str, default="none", choices=["none", "zero", "train_mean"])
    parser.add_argument("--tiny_subset_mode", action="store_true")
    parser.add_argument("--tiny_subset_size", type=int, default=64, help="Teacher spec: 32 or 64 CONSECUTIVE samples")
    parser.add_argument("--tiny_subset_start", type=int, default=0, help="Start index of the consecutive tiny subset window")
    parser.add_argument("--debug_backbone", action="store_true", help="Print backbone/neck/head output std per epoch for diagnosing signal collapse")
    parser.add_argument("--y_standardize", dest="y_standardize", action="store_true")
    parser.add_argument("--no_y_standardize", dest="y_standardize", action="store_false")
    parser.set_defaults(y_standardize=True)
    parser.add_argument("--progress_every", type=int, default=1, help="Print progress every N epochs")
    parser.add_argument(
        "--progress_file",
        type=str,
        default="task8_pathformer_progress.log",
        help="Progress log file under dataset/audit, use 'none' to disable file logging",
    )
    parser.add_argument(
        "--out_csv_name",
        type=str,
        default="task8_pathformer_baseline_results.csv",
        help="Output CSV file name under dataset/audit (or absolute path)",
    )
    parser.add_argument(
        "--out_summary_name",
        type=str,
        default="task8_pathformer_baseline_summary.txt",
        help="Output summary TXT file name under dataset/audit (or absolute path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    progress_file: Path | None
    if str(args.progress_file).lower() == "none":
        progress_file = None
    else:
        progress_file = resolve_output_path(args.progress_file)
        progress_file.write_text("", encoding="utf-8")

    log_progress("TASK 8 - PATHFORMER BASELINE RUN START", progress_file)
    log_progress(f"device={DEVICE}", progress_file)
    log_progress(
        (
            f"config: horizons={args.horizons}, max_epochs={args.max_epochs}, patience={args.patience}, "
            f"batch_size={args.batch_size}, layer_nums={args.layer_nums}, d_model={args.d_model}, d_ff={args.d_ff}, "
            f"lr={args.lr}, weight_decay={args.weight_decay}, seed={args.seed}, "
            f"y_standardize={args.y_standardize}, head_constraint={args.head_constraint}, "
            f"head_init_std={args.head_init_std}, head_init_scale={args.head_init_scale}, "
            f"loss_type={args.loss_type}, huber_delta={args.huber_delta}, neck_norm={args.neck_norm}, "
            f"baseline_mode={args.baseline_mode}, tiny_subset_mode={args.tiny_subset_mode}, "
            f"tiny_subset_size={args.tiny_subset_size}, tiny_subset_start={args.tiny_subset_start}"
        ),
        progress_file,
    )

    X = load_sequence_X()
    all_targets = {
        "5d": np.load(DATASET_DIR / "y_5d.npy"),
        "10d": np.load(DATASET_DIR / "y_10d.npy"),
        "20d": np.load(DATASET_DIR / "y_20d.npy"),
    }
    targets = {k: all_targets[k] for k in args.horizons}

    rows = []
    horizon_names = list(targets.keys())
    horizon_elapsed: list[float] = []
    for i, (horizon, y) in enumerate(targets.items(), start=1):
        log_progress(f"\n[{horizon}] started ({i}/{len(horizon_names)})", progress_file)
        if args.baseline_mode == "none":
            result = train_one_horizon(
                X,
                y,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                layer_nums=args.layer_nums,
                d_model=args.d_model,
                d_ff=args.d_ff,
                y_standardize=args.y_standardize,
                head_constraint=args.head_constraint,
                head_init_std=args.head_init_std,
                head_init_scale=args.head_init_scale,
                horizon_name=horizon,
                progress_file=progress_file,
                progress_every=args.progress_every,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
                tiny_subset_mode=args.tiny_subset_mode,
                tiny_subset_size=args.tiny_subset_size,
                tiny_subset_start=args.tiny_subset_start,
                neck_norm=args.neck_norm,
                debug_backbone=args.debug_backbone,
            )
        else:
            result = train_static_baseline(
                X,
                y,
                horizon_name=horizon,
                baseline_mode=args.baseline_mode,
                progress_file=progress_file,
            )
        metrics = result.metrics
        row = {"horizon": horizon, "samples": int(len(X)), **metrics}
        rows.append(row)
        horizon_elapsed.append(result.elapsed_sec)
        mean_horizon_sec = float(np.mean(horizon_elapsed))
        remain_horizons = max(0, len(horizon_names) - i)
        eta_horizons = remain_horizons * mean_horizon_sec
        log_progress(
            f"[{horizon}] train_mse={metrics['train_mse']:.8f} "
            f"train_pred_std={metrics['train_pred_std']:.6e} train_true_std={metrics['train_true_std']:.6e} "
            f"test_mse={metrics['test_mse']:.8f} "
            f"test_mae={metrics['test_mae']:.6f} test_corr={metrics['test_corr']:.4f} "
            f"test_rank_corr={metrics['test_rank_corr']:.4f} "
            f"test_da={metrics['test_direction_acc']:.4f} "
            f"test_pred_std={metrics['test_pred_std']:.6e} test_true_std={metrics['test_true_std']:.6e} "
            f"epochs_ran={result.epochs_ran} elapsed={format_seconds(result.elapsed_sec)} "
            f"global_eta={format_seconds(eta_horizons)}",
            progress_file,
        )

    df = pd.DataFrame(rows)
    out_csv = resolve_output_path(args.out_csv_name)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        keep_old = old[~old["horizon"].isin(df["horizon"])].copy()
        df = pd.concat([keep_old, df], ignore_index=True)
    df = df.sort_values("horizon")
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("TASK 8 - PATHFORMER BASELINE (MULTI-HORIZON)")
    lines.append("=" * 72)
    lines.append(f"device: {DEVICE}")
    lines.append(f"seed: {args.seed}")
    lines.append(
        f"epochs<= {args.max_epochs}, patience={args.patience}, batch={args.batch_size}, "
        f"layer_nums={args.layer_nums}, d_model={args.d_model}, d_ff={args.d_ff}"
    )
    lines.append(
        f"lr={args.lr}, weight_decay={args.weight_decay}, y_standardize={args.y_standardize}, "
        f"head_constraint={args.head_constraint}, head_init_std={args.head_init_std}, "
        f"head_init_scale={args.head_init_scale}, loss_type={args.loss_type}, "
        f"huber_delta={args.huber_delta}, neck_norm={args.neck_norm}, baseline_mode={args.baseline_mode}, "
        f"tiny_subset_mode={args.tiny_subset_mode}, tiny_subset_size={args.tiny_subset_size}, "
        f"tiny_subset_start={args.tiny_subset_start}"
    )
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['horizon']}: train_mse={r['train_mse']:.8f}, "
            f"train_pred_std={r['train_pred_std']:.6e}, train_true_std={r['train_true_std']:.6e}, "
            f"test_mse={r['test_mse']:.8f}, test_mae={r['test_mae']:.6f}, "
            f"test_corr={r['test_corr']:.4f}, test_rank_corr={r['test_rank_corr']:.4f}, "
            f"test_da={r['test_direction_acc']:.4f}, "
            f"test_pred_std={r['test_pred_std']:.6e}, test_true_std={r['test_true_std']:.6e}"
        )

    out_txt = resolve_output_path(args.out_summary_name)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_elapsed = time.perf_counter() - run_start
    log_progress("\n" + "\n".join(lines), progress_file)
    log_progress(f"\nSaved -> {out_csv}", progress_file)
    log_progress(f"Saved -> {out_txt}", progress_file)
    log_progress(f"Total elapsed: {format_seconds(total_elapsed)}", progress_file)
    if progress_file is not None:
        print(f"Saved -> {progress_file}")


if __name__ == "__main__":
    main()
