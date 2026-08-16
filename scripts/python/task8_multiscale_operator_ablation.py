"""
Task 8 - Multi-Scale Operator Ablation (Advisor-Driven)

This script implements a three-branch multi-scale regressor with configurable
operator family per branch, designed for:

1) Homogeneous operator comparison:
   - all_linear / all_dwconv / all_attention / all_pathformer
2) Heterogeneous hybrids:
   - hybrid_a / hybrid_b / hybrid_c
3) Scale ablation on a chosen hybrid:
   - keep_branches: short, medium, long, short+medium, medium+long, short+long

Default branch-frequency mapping (editable via args):
- short  -> hourly,halfday
- medium -> daily
- long   -> weekly
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

FREQ_NAMES = ["hourly", "halfday", "daily", "weekly"]
BRANCH_NAMES = ["short", "medium", "long"]

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 0.0

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


def load_sequence_X_multi() -> dict[str, np.ndarray]:
    return {
        "hourly": np.load(DATASET_DIR / "X_hourly.npy").astype(np.float32),
        "halfday": np.load(DATASET_DIR / "X_halfday.npy").astype(np.float32),
        "daily": np.load(DATASET_DIR / "X_daily.npy").astype(np.float32),
        "weekly": np.load(DATASET_DIR / "X_weekly.npy").astype(np.float32),
    }


def compute_patch_sizes(seq_len: int, num_scales: int = 4) -> list[int]:
    divisors = [d for d in range(1, seq_len + 1) if seq_len % d == 0]
    divisors = sorted(divisors, reverse=True)
    if len(divisors) >= num_scales:
        idx = np.linspace(0, len(divisors) - 1, num_scales).round().astype(int)
        chosen = [divisors[i] for i in idx]
    else:
        chosen = divisors + [1] * (num_scales - len(divisors))
    return chosen[:num_scales]


def parse_list_arg(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def get_branch_ops(model_type: str, custom_ops: str | None) -> dict[str, str]:
    if model_type == "all_linear":
        return {"short": "linear", "medium": "linear", "long": "linear"}
    if model_type == "all_dwconv":
        return {"short": "dwconv", "medium": "dwconv", "long": "dwconv"}
    if model_type == "all_attention":
        return {"short": "attention", "medium": "attention", "long": "attention"}
    if model_type == "all_pathformer":
        return {"short": "pathformer", "medium": "pathformer", "long": "pathformer"}
    if model_type == "hybrid_a":
        return {"short": "dwconv", "medium": "attention", "long": "linear"}
    if model_type == "hybrid_b":
        return {"short": "dwconv", "medium": "attention", "long": "attention"}
    if model_type == "hybrid_c":
        return {"short": "attention", "medium": "dwconv", "long": "linear"}
    if model_type == "custom":
        if not custom_ops:
            raise ValueError("custom model_type requires --custom_ops, e.g. short:dwconv,medium:attention,long:linear")
        result = {}
        for item in parse_list_arg(custom_ops):
            if ":" not in item:
                raise ValueError(f"Bad custom_ops item: {item}")
            k, v = item.split(":", 1)
            k = k.strip()
            v = v.strip()
            result[k] = v
        for b in BRANCH_NAMES:
            if b not in result:
                raise ValueError(f"custom_ops missing branch: {b}")
        return result
    raise ValueError(f"Unsupported model_type: {model_type}")


class LinearBranchEncoder(nn.Module):
    def __init__(self, seq_len: int, feature_dim: int, d_branch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feature_dim, d_branch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DWConvBranchEncoder(nn.Module):
    def __init__(self, feature_dim: int, d_branch: int):
        super().__init__()
        self.dw = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim)
        self.pw = nn.Conv1d(feature_dim, d_branch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        h = x.transpose(1, 2)
        h = self.act(self.dw(h))
        h = self.act(self.pw(h))
        return h.mean(dim=-1)


class AttentionBranchEncoder(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, d_branch: int, num_heads: int):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_branch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.mean(dim=1)
        return self.ff(h)


class PathFormerBranchEncoder(nn.Module):
    def __init__(self, seq_len: int, feature_dim: int, layer_nums: int, d_model: int, d_ff: int, d_branch: int):
        super().__init__()
        patch_sizes = compute_patch_sizes(seq_len, num_scales=4)
        cfg = SimpleNamespace(
            layer_nums=layer_nums,
            num_nodes=feature_dim,
            pred_len=1,
            seq_len=seq_len,
            k=2,
            num_experts_list=[4] * layer_nums,
            patch_size_list=[patch_sizes for _ in range(layer_nums)],
            d_model=d_model,
            d_ff=d_ff,
            residual_connection=1,
            revin=True,
            gpu=0,
            batch_norm=False,
        )
        self.backbone = PathFormerBackbone(cfg)
        self.proj = nn.Linear(feature_dim, d_branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _aux = self.backbone(x)
        return self.proj(out[:, -1, :])


class MultiFreqBranch(nn.Module):
    """One semantic branch (short/medium/long) that can consume multiple frequencies.

    If a branch maps to multiple frequencies, this module encodes each frequency
    with the same operator type and averages branch vectors.
    """

    def __init__(
        self,
        branch_name: str,
        op_type: str,
        freq_list: list[str],
        seq_lens: dict[str, int],
        feature_dim: int,
        layer_nums: int,
        d_model: int,
        d_ff: int,
        d_branch: int,
        attn_heads: int,
    ):
        super().__init__()
        self.branch_name = branch_name
        self.op_type = op_type
        self.freq_list = freq_list
        encoders = {}
        for freq in freq_list:
            if op_type == "linear":
                encoders[freq] = LinearBranchEncoder(seq_lens[freq], feature_dim, d_branch)
            elif op_type == "dwconv":
                encoders[freq] = DWConvBranchEncoder(feature_dim, d_branch)
            elif op_type == "attention":
                encoders[freq] = AttentionBranchEncoder(feature_dim, d_model, d_branch, attn_heads)
            elif op_type == "pathformer":
                encoders[freq] = PathFormerBranchEncoder(seq_lens[freq], feature_dim, layer_nums, d_model, d_ff, d_branch)
            else:
                raise ValueError(f"Unsupported operator type: {op_type}")
        self.encoders = nn.ModuleDict(encoders)

    def forward(self, x_by_freq: dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = [self.encoders[freq](x_by_freq[freq]) for freq in self.freq_list]
        return torch.stack(vecs, dim=1).mean(dim=1)


class MultiScaleAblationRegressor(nn.Module):
    def __init__(
        self,
        seq_lens: dict[str, int],
        feature_dim: int,
        branch_ops: dict[str, str],
        branch_freqs: dict[str, list[str]],
        keep_branches: list[str],
        layer_nums: int,
        d_model: int,
        d_ff: int,
        d_branch: int,
        attn_heads: int,
        fusion_type: str,
        d_fusion: int,
        head_constraint: str,
        head_init_std: float,
        head_init_scale: float,
    ):
        super().__init__()
        self.keep_branches = keep_branches
        self.fusion_type = fusion_type

        self.branches = nn.ModuleDict()
        for b in keep_branches:
            self.branches[b] = MultiFreqBranch(
                branch_name=b,
                op_type=branch_ops[b],
                freq_list=branch_freqs[b],
                seq_lens=seq_lens,
                feature_dim=feature_dim,
                layer_nums=layer_nums,
                d_model=d_model,
                d_ff=d_ff,
                d_branch=d_branch,
                attn_heads=attn_heads,
            )

        n_active = len(keep_branches)
        if fusion_type == "concat":
            fused_dim = d_branch * n_active
            self.neck = nn.LayerNorm(fused_dim)
            self.head = nn.Linear(fused_dim, 1)
        elif fusion_type == "gated":
            self.branch_proj = nn.ModuleDict({b: nn.Linear(d_branch, d_fusion) for b in keep_branches})
            self.gate = nn.Linear(d_fusion * n_active, n_active)
            self.neck = nn.LayerNorm(d_fusion)
            self.head = nn.Linear(d_fusion, 1)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)

        self.head_constraint = head_constraint
        init_log_scale = float(np.log(np.exp(max(1e-6, head_init_scale)) - 1.0))
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale, dtype=torch.float32))

    def forward(self, x_by_freq: dict[str, torch.Tensor]) -> torch.Tensor:
        vecs = {b: self.branches[b](x_by_freq) for b in self.keep_branches}

        if self.fusion_type == "concat":
            fused = torch.cat([vecs[b] for b in self.keep_branches], dim=-1)
        else:
            projected = [self.branch_proj[b](vecs[b]) for b in self.keep_branches]
            gate_input = torch.cat(projected, dim=-1)
            gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
            stacked = torch.stack(projected, dim=1)
            fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        fused = self.neck(fused)
        raw = self.head(fused).squeeze(-1)
        if self.head_constraint == "tanh_scale":
            scale = torch.nn.functional.softplus(self.log_scale) + 1e-6
            return torch.tanh(raw) * scale
        return raw


def train_one_horizon(
    X_by_freq: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    model_type: str,
    branch_ops: dict[str, str],
    branch_freqs: dict[str, list[str]],
    keep_branches: list[str],
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    layer_nums: int,
    d_model: int,
    d_ff: int,
    d_branch: int,
    attn_heads: int,
    fusion_type: str,
    d_fusion: int,
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
) -> TrainArtifacts:
    n_total = len(y)
    if tiny_subset_mode:
        start = max(0, min(tiny_subset_start, max(0, n_total - 1)))
        end = min(n_total, start + max(1, tiny_subset_size))
        X_by_freq = {name: arr[start:end] for name, arr in X_by_freq.items()}
        y = y[start:end]
        n_total = len(y)
        log_progress(
            f"[{horizon_name}/{model_type}] tiny subset mode: samples [{start}:{end}) size={n_total}",
            progress_file,
        )

    if n_total < 20:
        raise ValueError(f"[{horizon_name}/{model_type}] not enough samples after filtering: {n_total}")

    s_train, s_val, s_test = split_indices(n_total)

    seq_lens = {name: arr.shape[1] for name, arr in X_by_freq.items()}
    feature_dim = next(iter(X_by_freq.values())).shape[2]

    X_train = {name: torch.from_numpy(arr[s_train]) for name, arr in X_by_freq.items()}
    X_val = {name: torch.from_numpy(arr[s_val]) for name, arr in X_by_freq.items()}
    X_test = {name: torch.from_numpy(arr[s_test]) for name, arr in X_by_freq.items()}

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

    def make_loader(x_dict: dict[str, torch.Tensor], y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = [x_dict[name] for name in FREQ_NAMES] + [torch.from_numpy(y_arr)]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train_use, shuffle=True)
    train_eval_loader = make_loader(X_train, y_train_use, shuffle=False)
    val_loader = make_loader(X_val, y_val_use, shuffle=False)
    test_loader = make_loader(X_test, y_test_use, shuffle=False)

    model = MultiScaleAblationRegressor(
        seq_lens=seq_lens,
        feature_dim=feature_dim,
        branch_ops=branch_ops,
        branch_freqs=branch_freqs,
        keep_branches=keep_branches,
        layer_nums=layer_nums,
        d_model=d_model,
        d_ff=d_ff,
        d_branch=d_branch,
        attn_heads=attn_heads,
        fusion_type=fusion_type,
        d_fusion=d_fusion,
        head_constraint=head_constraint,
        head_init_std=head_init_std,
        head_init_scale=head_init_scale,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=huber_delta) if loss_type == "huber" else nn.MSELoss()

    def batch_to_dict(batch: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        *xs, yb = batch
        x_dict = {name: xs[i].to(DEVICE) for i, name in enumerate(FREQ_NAMES)}
        return x_dict, yb.to(DEVICE)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    horizon_start = time.perf_counter()
    running_epoch_secs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            x_dict, yb = batch_to_dict(batch)
            optimizer.zero_grad()
            pred = model(x_dict)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_dict, yb = batch_to_dict(batch)
                pred = model(x_dict)
                val_losses.append(float(criterion(pred, yb).item()))

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_sec = time.perf_counter() - epoch_start
        running_epoch_secs.append(epoch_sec)
        mean_epoch_sec = float(np.mean(running_epoch_secs))
        remain_epochs = max(0, max_epochs - epoch)
        eta_sec = remain_epochs * mean_epoch_sec

        if (epoch % max(1, progress_every) == 0) or (epoch == 1) or (epoch == max_epochs) or (bad_epochs >= patience):
            log_progress(
                (
                    f"[{horizon_name}/{model_type}] epoch {epoch:03d}/{max_epochs:03d} "
                    f"val_mse={mean_val:.8f} best_val_mse={best_val:.8f} "
                    f"bad_epochs={bad_epochs}/{patience} "
                    f"epoch_time={format_seconds(epoch_sec)} eta={format_seconds(eta_sec)}"
                ),
                progress_file,
            )

        if bad_epochs >= patience:
            log_progress(f"[{horizon_name}/{model_type}] early stopping at epoch {epoch}", progress_file)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        preds = []
        trues = []
        with torch.no_grad():
            for batch in loader:
                x_dict, yb = batch_to_dict(batch)
                pred = model(x_dict).cpu().numpy()
                preds.append(pred)
                trues.append(yb.cpu().numpy())
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
        "train_samples": int(len(y_train_np)),
        "val_samples": int(len(y_val_np)),
        "test_samples": int(len(y_test_np)),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", default=["5d", "10d", "20d"], choices=["5d", "10d", "20d"])

    p.add_argument(
        "--model_type",
        type=str,
        default="hybrid_a",
        choices=[
            "all_linear",
            "all_dwconv",
            "all_attention",
            "all_pathformer",
            "hybrid_a",
            "hybrid_b",
            "hybrid_c",
            "custom",
        ],
    )
    p.add_argument(
        "--custom_ops",
        type=str,
        default=None,
        help="Used when model_type=custom, e.g. short:dwconv,medium:attention,long:linear",
    )

    p.add_argument("--short_freqs", type=str, default="hourly,halfday")
    p.add_argument("--medium_freqs", type=str, default="daily")
    p.add_argument("--long_freqs", type=str, default="weekly")

    p.add_argument(
        "--keep_branches",
        type=str,
        default="short,medium,long",
        help="Comma-separated subset of {short,medium,long}",
    )

    p.add_argument("--fusion_type", type=str, default="concat", choices=["concat", "gated"])

    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    p.add_argument("--layer_nums", type=int, default=1)
    p.add_argument("--d_model", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--d_branch", type=int, default=16)
    p.add_argument("--attn_heads", type=int, default=1)
    p.add_argument("--d_fusion", type=int, default=16)

    p.add_argument("--head_constraint", type=str, default="none", choices=["none", "tanh_scale"])
    p.add_argument("--head_init_std", type=float, default=0.02)
    p.add_argument("--head_init_scale", type=float, default=0.2)

    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.35)

    p.add_argument("--tiny_subset_mode", action="store_true")
    p.add_argument("--tiny_subset_size", type=int, default=64)
    p.add_argument("--tiny_subset_start", type=int, default=0)

    p.add_argument("--y_standardize", dest="y_standardize", action="store_true")
    p.add_argument("--no_y_standardize", dest="y_standardize", action="store_false")
    p.set_defaults(y_standardize=True)

    p.add_argument("--progress_every", type=int, default=1)
    p.add_argument("--progress_file", type=str, default="task8_multiscale_operator_ablation.log")
    p.add_argument("--out_csv_name", type=str, default="task8_multiscale_operator_ablation_results.csv")
    p.add_argument("--out_summary_name", type=str, default="task8_multiscale_operator_ablation_summary.txt")
    return p.parse_args()


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

    branch_ops = get_branch_ops(args.model_type, args.custom_ops)

    branch_freqs = {
        "short": parse_list_arg(args.short_freqs),
        "medium": parse_list_arg(args.medium_freqs),
        "long": parse_list_arg(args.long_freqs),
    }
    keep_branches = parse_list_arg(args.keep_branches)

    for b in keep_branches:
        if b not in BRANCH_NAMES:
            raise ValueError(f"Unknown branch in keep_branches: {b}")
    for b in BRANCH_NAMES:
        if len(branch_freqs[b]) == 0:
            raise ValueError(f"Branch {b} has empty frequency list")
        for freq in branch_freqs[b]:
            if freq not in FREQ_NAMES:
                raise ValueError(f"Unknown frequency: {freq}")

    log_progress("TASK 8 - MULTI-SCALE OPERATOR ABLATION RUN START", progress_file)
    log_progress(f"device={DEVICE}", progress_file)
    log_progress(
        (
            f"config: horizons={args.horizons}, model_type={args.model_type}, branch_ops={branch_ops}, "
            f"keep_branches={keep_branches}, branch_freqs={branch_freqs}, fusion_type={args.fusion_type}, "
            f"max_epochs={args.max_epochs}, patience={args.patience}, batch_size={args.batch_size}, "
            f"layer_nums={args.layer_nums}, d_model={args.d_model}, d_ff={args.d_ff}, d_branch={args.d_branch}, "
            f"attn_heads={args.attn_heads}, d_fusion={args.d_fusion}, "
            f"lr={args.lr}, weight_decay={args.weight_decay}, seed={args.seed}, "
            f"y_standardize={args.y_standardize}, head_constraint={args.head_constraint}, "
            f"head_init_std={args.head_init_std}, head_init_scale={args.head_init_scale}, "
            f"loss_type={args.loss_type}, huber_delta={args.huber_delta}, "
            f"tiny_subset_mode={args.tiny_subset_mode}, tiny_subset_size={args.tiny_subset_size}, "
            f"tiny_subset_start={args.tiny_subset_start}"
        ),
        progress_file,
    )

    X_by_freq = load_sequence_X_multi()
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
        tag = f"{args.model_type}|{args.fusion_type}|{','.join(keep_branches)}"
        log_progress(f"\n[{horizon}/{tag}] started ({i}/{len(horizon_names)})", progress_file)

        result = train_one_horizon(
            X_by_freq,
            y,
            model_type=tag,
            branch_ops=branch_ops,
            branch_freqs=branch_freqs,
            keep_branches=keep_branches,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            layer_nums=args.layer_nums,
            d_model=args.d_model,
            d_ff=args.d_ff,
            d_branch=args.d_branch,
            attn_heads=args.attn_heads,
            fusion_type=args.fusion_type,
            d_fusion=args.d_fusion,
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
        )

        metrics = result.metrics
        row = {
            "horizon": horizon,
            "model_type": args.model_type,
            "fusion_type": args.fusion_type,
            "keep_branches": ",".join(keep_branches),
            "short_op": branch_ops["short"],
            "medium_op": branch_ops["medium"],
            "long_op": branch_ops["long"],
            "short_freqs": ",".join(branch_freqs["short"]),
            "medium_freqs": ",".join(branch_freqs["medium"]),
            "long_freqs": ",".join(branch_freqs["long"]),
            **metrics,
        }
        rows.append(row)

        horizon_elapsed.append(result.elapsed_sec)
        mean_horizon_sec = float(np.mean(horizon_elapsed))
        remain_horizons = max(0, len(horizon_names) - i)
        eta_horizons = remain_horizons * mean_horizon_sec

        log_progress(
            f"[{horizon}/{tag}] train_mse={metrics['train_mse']:.8f} "
            f"train_pred_std={metrics['train_pred_std']:.6e} train_true_std={metrics['train_true_std']:.6e} "
            f"test_mse={metrics['test_mse']:.8f} test_mae={metrics['test_mae']:.6f} "
            f"test_corr={metrics['test_corr']:.4f} test_rank_corr={metrics['test_rank_corr']:.4f} "
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
        key_cols = ["horizon", "model_type", "fusion_type", "keep_branches"]
        merge_key = df[key_cols].apply(tuple, axis=1)
        old_key = old[key_cols].apply(tuple, axis=1)
        keep_old = old[~old_key.isin(merge_key)].copy()
        df = pd.concat([keep_old, df], ignore_index=True)
    df = df.sort_values(["horizon", "model_type", "fusion_type", "keep_branches"])
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("TASK 8 - MULTI-SCALE OPERATOR ABLATION")
    lines.append("=" * 72)
    lines.append(f"device: {DEVICE}")
    lines.append(f"seed: {args.seed}")
    lines.append(
        f"model_type={args.model_type}, fusion_type={args.fusion_type}, keep_branches={','.join(keep_branches)}, "
        f"short_op={branch_ops['short']}, medium_op={branch_ops['medium']}, long_op={branch_ops['long']}"
    )
    lines.append(
        f"short_freqs={','.join(branch_freqs['short'])}, medium_freqs={','.join(branch_freqs['medium'])}, "
        f"long_freqs={','.join(branch_freqs['long'])}"
    )
    lines.append(
        f"epochs<= {args.max_epochs}, patience={args.patience}, batch={args.batch_size}, "
        f"layer_nums={args.layer_nums}, d_model={args.d_model}, d_ff={args.d_ff}, d_branch={args.d_branch}, "
        f"attn_heads={args.attn_heads}, d_fusion={args.d_fusion}"
    )
    lines.append(
        f"lr={args.lr}, weight_decay={args.weight_decay}, y_standardize={args.y_standardize}, "
        f"head_constraint={args.head_constraint}, head_init_std={args.head_init_std}, "
        f"head_init_scale={args.head_init_scale}, loss_type={args.loss_type}, huber_delta={args.huber_delta}, "
        f"tiny_subset_mode={args.tiny_subset_mode}, tiny_subset_size={args.tiny_subset_size}, "
        f"tiny_subset_start={args.tiny_subset_start}"
    )
    lines.append("")
    for _, r in df.iterrows():
        if (
            r["model_type"] == args.model_type
            and r["fusion_type"] == args.fusion_type
            and r["keep_branches"] == ",".join(keep_branches)
            and r["short_op"] == branch_ops["short"]
            and r["medium_op"] == branch_ops["medium"]
            and r["long_op"] == branch_ops["long"]
        ):
            lines.append(
                f"- {r['horizon']}: train_mse={r['train_mse']:.8f}, train_pred_std={r['train_pred_std']:.6e}, "
                f"train_true_std={r['train_true_std']:.6e}, "
                f"test_mse={r['test_mse']:.8f}, test_mae={r['test_mae']:.6f}, "
                f"test_corr={r['test_corr']:.4f}, test_rank_corr={r['test_rank_corr']:.4f}, "
                f"test_da={r['test_direction_acc']:.4f}, test_pred_std={r['test_pred_std']:.6e}, "
                f"test_true_std={r['test_true_std']:.6e}"
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
