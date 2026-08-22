"""Frozen-panel SWiM-style (local / shifted-window attention) baseline.

This script implements the structured windowed-attention improved Transformer
control under the frozen 17-stock Daily + Weekly panel protocol defined by
`panel_common.py` and the Vanilla Transformer panel baseline.

Design constraints (frozen for this v1; see project audit before changing):
- frequency-specific branches + late concat fusion (no early fusion, no
  cross-frequency attention, no router, no adaptive/multi-scale mechanism)
- exactly two window blocks per branch: block 1 = local window attention
  (shift=0), block 2 = shifted-window attention (shift = window_size // 2)
- Daily window_size=10, shift_size=5; Weekly window_size=4, shift_size=2
  (development baseline values; NOT derived from a search, NOT the historical
  FSLR window_size=16, which belonged to a different 160-token concatenated
  sequence and is not applicable here)
- Post-LN residual structure (unlike the historical FSLR SWiM, which used
  Pre-LN) to stay close to the frozen panel Vanilla Transformer's normalization
  convention
- a mathematically valid shifted-window boundary mask (the historical FSLR
  SWiM used torch.roll without any such mask -- that bug is intentionally not
  reproduced here)
- zero padding (not the historical last-timestep-repeat padding) with an
  explicit validity mask threaded through padding, cyclic shift, window
  partition/reverse, and pooling
- masked mean pooling over real timesteps only (not final-timestep pooling,
  since two local/shifted-window blocks do not give the final token the same
  global receptive field a global-attention Transformer has)
- reuses the frozen panel loader, split, normalization, training protocol,
  and evaluation functions from `panel_common.py`, and reuses several
  frequency-agnostic helpers directly from `panel_baseline_vanilla_transformer.py`
  (imported only, that file is not modified)

Default behavior is smoke/audit validation only. The full 9-experiment
benchmark is intentionally not launched by this task; formal training
requires explicit `--train` (single config) or `--train --all`.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import (
    AUDIT_DIR,
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
)
from scripts.python.panel_baseline_vanilla_transformer import (
    BATCH_SIZE,
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    FIXED_MAX_LEN,
    FREQUENCY_SETTINGS,
    LEARNING_RATE,
    MAX_EPOCHS,
    NHEAD,
    SEED,
    PositionalEncoding,
    assert_frozen_panel_checks,
    compute_rank_ic_summary,
    count_trainable_parameters,
    format_elapsed,
    prediction_scale_looks_normal,
    set_seed,
)

# =====================================================================
# FROZEN SWiM v1 DESIGN CONSTANTS -- do not tune in this task
# =====================================================================

NUM_BLOCKS = 2
WINDOW_CONFIG = {
    "daily": {"window_size": 10, "shift_size": 5},
    "weekly": {"window_size": 4, "shift_size": 2},
}

DEBUG_NAN_MODE = False
DEBUG_NAN_CONTEXT = {"frequency": None, "batch_index": None}


def _trace_rng_fingerprint_summary(prefix: str = "RNG") -> str:
    state = torch.get_rng_state()
    arr = state.detach().cpu().numpy()
    if arr.size == 0:
        return f"{prefix}=empty"
    abs_arr = np.abs(arr)
    s = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std())
    max_abs = float(abs_arr.max())
    fingerprint = int(np.frombuffer(arr.tobytes(), dtype=np.uint64).sum() % (2**63 - 1))
    return f"{prefix}: sum={s:.6f} mean={mean:.6f} std={std:.6f} max_abs={max_abs:.6f} fingerprint={fingerprint}"


def _trace_tensor_summary(name: str, x: torch.Tensor) -> None:
    arr = x.detach().cpu().numpy()
    print(
        f"[TRACE] {name}: shape={tuple(arr.shape)} "
        f"sum={float(arr.sum()):.8f} mean={float(arr.mean()):.8f} "
        f"std={float(arr.std()):.8f} max_abs={float(np.max(np.abs(arr))):.8f}"
    )


def _trace_first_batch_fingerprint(batch_idx: int, x: torch.Tensor, y: torch.Tensor) -> None:
    x_arr = x.detach().cpu().numpy()
    y_arr = y.detach().cpu().numpy()
    print(f"[TRACE] batch={batch_idx} x: shape={tuple(x_arr.shape)} sum={float(x_arr.sum()):.8f} mean={float(x_arr.mean()):.8f} std={float(x_arr.std()):.8f} max_abs={float(np.max(np.abs(x_arr))):.8f}")
    first_5 = y_arr[:5]
    print(
        f"[TRACE] batch={batch_idx} y[:5]={first_5.tolist()} "
        f"sum={float(y_arr.sum()):.8f} mean={float(y_arr.mean()):.8f} std={float(y_arr.std()):.8f}"
    )


def _model_parameter_fingerprint(model: nn.Module) -> tuple[int, float, float]:
    param_total = sum(p.numel() for p in model.parameters())
    param_sum = float(sum(p.detach().float().sum().item() for p in model.parameters()))
    param_sq_sum = float(sum((p.detach().float() ** 2).sum().item() for p in model.parameters()))
    return param_total, param_sum, param_sq_sum


# =====================================================================
# WINDOW PARTITION / REVERSE (1D, temporal axis)
# =====================================================================


def window_partition_1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, T, C] -> [B * n_win, window_size, C]. Requires T % window_size == 0."""
    B, T, C = x.shape
    assert T % window_size == 0, f"T={T} is not divisible by window_size={window_size}"
    n_win = T // window_size
    return x.reshape(B, n_win, window_size, C).reshape(B * n_win, window_size, C)


def window_reverse_1d(windows: torch.Tensor, window_size: int, B: int, T: int) -> torch.Tensor:
    """[B * n_win, window_size, C] -> [B, T, C]. Exact inverse of window_partition_1d."""
    n_win = T // window_size
    C = windows.shape[-1]
    return windows.reshape(B, n_win, window_size, C).reshape(B, T, C)


def compute_pad_len(T: int, window_size: int) -> int:
    return (window_size - (T % window_size)) % window_size


# =====================================================================
# SHIFT-BOUNDARY MASK + PADDING MASK
# =====================================================================


def compute_shift_region_ids(T_pad: int, window_size: int, shift_size: int, device) -> torch.Tensor:
    """Swin-style region labeling (1D adaptation).

    Assigns a region id to every position in the *post-shift* temporal axis
    such that two positions share a region iff they were contiguous in the
    original (pre-roll) sequence. Used to build the shifted-window attention
    mask that blocks artificial wrap-around interactions created by
    `torch.roll`. This mirrors the standard Swin Transformer `img_mask`
    construction, generalized from 2D images to a 1D temporal axis.
    """
    region_id = torch.zeros(T_pad, dtype=torch.long, device=device)
    cnt = 0
    for sl in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
        region_id[sl] = cnt
        cnt += 1
    return region_id


def build_window_masks(valid_mask: torch.Tensor, window_size: int, shift_size: int) -> torch.Tensor:
    """Build an exact additive attention mask for 1D window attention.

    valid_mask: [T_pad] bool, True = real token, False = zero-padded token.
    Returns attn_mask: [n_win, window_size, window_size] float.

    REAL QUERY semantics:
      - padded keys are strictly masked as -inf;
      - keys from different shifted-window regions are strictly masked as -inf;
      - valid same-region real-key interactions remain allowed with additive mask 0.

    PADDED QUERY semantics:
      - the query output is semantically irrelevant because it is hard-zeroed
        immediately afterward;
      - to keep the row numerically valid for softmax, we allow exactly one dummy
        self key (the first key slot in the window) and mask every other key.

    This preserves the exact hard-mask semantics required by the frozen SWiM v1
    design: artificial cyclic wrap-around interactions and padded keys are
    strictly masked for valid queries, while padded-query rows remain defined
    and are later discarded by `_hard_zero`.
    """
    device = valid_mask.device
    T_pad = valid_mask.shape[0]
    n_win = T_pad // window_size

    attn_mask = torch.full((n_win, window_size, window_size), -torch.inf, dtype=torch.float32, device=device)
    if shift_size > 0:
        valid_shifted = torch.roll(valid_mask, shifts=-shift_size, dims=0)
    else:
        valid_shifted = valid_mask
    query_valid = valid_shifted.view(n_win, window_size)
    key_valid = valid_shifted.view(n_win, window_size)

    if shift_size > 0:
        region_id = compute_shift_region_ids(T_pad, window_size, shift_size, device)
        region_windows = region_id.view(n_win, window_size)
        region_block = region_windows.unsqueeze(2) != region_windows.unsqueeze(1)
    else:
        region_block = torch.zeros((n_win, window_size, window_size), dtype=torch.bool, device=device)

    allowed_real = query_valid.unsqueeze(2) & key_valid.unsqueeze(1) & ~region_block
    attn_mask.masked_fill_(allowed_real, 0.0)

    padded_rows = ~query_valid
    if padded_rows.any():
        dummy_key = torch.zeros_like(query_valid, dtype=torch.bool)
        dummy_key[:, 0] = True
        padded_allow = padded_rows.unsqueeze(2) & dummy_key.unsqueeze(1)
        attn_mask.masked_fill_(padded_allow, 0.0)

    return attn_mask


# =====================================================================
# MODEL
# =====================================================================


class SWiMWindowBlock(nn.Module):
    """One local- or shifted-window self-attention block.

    Residual / normalization structure is Post-LN, matching the frozen panel
    Vanilla Transformer's `nn.TransformerEncoderLayer` convention (the
    historical FSLR SWiM used Pre-LN; that is intentionally not reproduced):

        attn_out = WindowAttention(x)
        x = LayerNorm(x + Dropout(attn_out))
        ffn_out = FFN(x)
        x = LayerNorm(x + Dropout(ffn_out))

    Padded positions are hard-zeroed after every sub-layer. This is required
    because padded-query rows are semantically irrelevant: they are never part
    of a real-token representation and must not leak into later blocks or the
    masked mean pool. For valid queries, attention is exact-hard-masked with
    `-inf` on padded keys and cross-region wrap-around keys; for padded-query
    rows, one harmless dummy/self key remains finite so the row stays
    numerically valid for softmax and then gets replaced by exact zero via
    `_hard_zero`.
    """

    def __init__(self, d_model: int, nhead: int, window_size: int, shift_size: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _hard_zero(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid (padded) positions with exactly 0.0.

        Uses torch.where rather than multiplying by a 0/1 mask: a degenerate
        all -inf attention row (see class docstring) produces NaN, and
        `0 * NaN == NaN` under IEEE754 -- multiplying would NOT clear it.
        torch.where genuinely replaces the value regardless of NaN/Inf.
        """
        mask = valid_mask.view(1, -1, 1).expand_as(x)
        return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T_pad, C]; valid_mask: [T_pad] bool (True = real token)."""
        B, T_pad, C = x.shape
        w = self.window_size
        n_win = T_pad // w

        if DEBUG_NAN_MODE:
            if not torch.isfinite(x).all():
                print(f"[DEBUG_NAN] BLOCK failure before attention: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size} x_nonfinite=True")
                raise RuntimeError("Non-finite input to SWiMWindowBlock.forward")

        attn_mask_win = build_window_masks(valid_mask, w, self.shift_size)  # [n_win, w, w] float, always defined
        attn_mask = attn_mask_win.unsqueeze(0).expand(B, n_win, w, w).reshape(B * n_win, w, w)
        attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
        if DEBUG_NAN_MODE:
            if not torch.isfinite(attn_mask).all() and not torch.all((attn_mask == 0.0) | torch.isinf(attn_mask)):
                print(f"[DEBUG_NAN] BAD MASK: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size} attn_mask_has_nonfinite=True")
                raise RuntimeError("Non-finite values in attention mask")
            if torch.any(torch.isinf(attn_mask).all(dim=-1) & torch.isinf(attn_mask).all(dim=-2)):
                print(f"[DEBUG_NAN] ALL-INF MASK ROW: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
                raise RuntimeError("All-inf attention row encountered in debug mode")

        residual = x
        x_shift = torch.roll(x, shifts=-self.shift_size, dims=1) if self.shift_size > 0 else x
        if DEBUG_NAN_MODE and not torch.isfinite(x_shift).all():
            print(f"[DEBUG_NAN] x_shift non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite x_shift in SWiMWindowBlock.forward")

        xw = window_partition_1d(x_shift, w)
        if DEBUG_NAN_MODE and not torch.isfinite(xw).all():
            print(f"[DEBUG_NAN] xw non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite xw in SWiMWindowBlock.forward")

        attn_out, _ = self.attn(xw, xw, xw, attn_mask=attn_mask, need_weights=False)
        if DEBUG_NAN_MODE and not torch.isfinite(attn_out).all():
            print(f"[DEBUG_NAN] attn_out non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite attention output in SWiMWindowBlock.forward")

        attn_out = window_reverse_1d(attn_out, w, B, T_pad)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=self.shift_size, dims=1)
        attn_out = self._hard_zero(attn_out, valid_mask)  # clear the harmless-but-discarded padded-query rows
        if DEBUG_NAN_MODE and not torch.isfinite(attn_out).all():
            print(f"[DEBUG_NAN] attn_out after reverse non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite attn_out after reverse")

        x = self.norm1(residual + self.dropout1(attn_out))
        if DEBUG_NAN_MODE and not torch.isfinite(x).all():
            print(f"[DEBUG_NAN] norm1 output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite norm1 output")
        x = self._hard_zero(x, valid_mask)

        ffn_out = self._hard_zero(self.ffn(x), valid_mask)
        if DEBUG_NAN_MODE and not torch.isfinite(ffn_out).all():
            print(f"[DEBUG_NAN] FFN output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite FFN output")
        x = self.norm2(x + self.dropout2(ffn_out))
        if DEBUG_NAN_MODE and not torch.isfinite(x).all():
            print(f"[DEBUG_NAN] norm2 output non-finite: frequency={DEBUG_NAN_CONTEXT['frequency']} batch={DEBUG_NAN_CONTEXT['batch_index']} block={self.shift_size}")
            raise RuntimeError("Non-finite norm2 output")
        x = self._hard_zero(x, valid_mask)
        return x


class SWiMBranch(nn.Module):
    """Per-frequency SWiM encoder: projection -> PE -> 2 window blocks -> masked mean pool."""

    def __init__(
        self,
        feature_dim: int,
        frequency_key: str,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if frequency_key not in WINDOW_CONFIG:
            raise ValueError(f"Unknown frequency_key: {frequency_key}")
        cfg = WINDOW_CONFIG[frequency_key]
        self.window_size = cfg["window_size"]
        self.shift_size = cfg["shift_size"]
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=FIXED_MAX_LEN, dropout=dropout)
        assert NUM_BLOCKS == 2, "SWiM v1 requires exactly two blocks: local + shifted-window."
        self.blocks = nn.ModuleList(
            [
                SWiMWindowBlock(d_model, nhead, self.window_size, 0, dim_feedforward, dropout),
                SWiMWindowBlock(d_model, nhead, self.window_size, self.shift_size, dim_feedforward, dropout),
            ]
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        """x: [B, T, F] -> representation [B, d_model] (or the padded encoded
        sequence [B, T_pad, d_model] plus the validity mask if
        return_sequence=True, used only by audit tests)."""
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.positional_encoding(h)

        pad_len = compute_pad_len(T, self.window_size)
        T_pad = T + pad_len
        valid_mask = torch.ones(T_pad, dtype=torch.bool, device=x.device)
        if pad_len > 0:
            zeros = torch.zeros(B, pad_len, h.shape[-1], device=h.device, dtype=h.dtype)
            h = torch.cat([h, zeros], dim=1)
            valid_mask[T:] = False

        for blk in self.blocks:
            h = blk(h, valid_mask)

        if return_sequence:
            return h, valid_mask

        rep = h.sum(dim=1) / float(T)  # padded positions are exactly 0; dividing by true length T
        return rep


class SingleFrequencySWiMRegressor(nn.Module):
    def __init__(self, feature_dim: int, frequency_key: str):
        super().__init__()
        self.branch = SWiMBranch(feature_dim=feature_dim, frequency_key=frequency_key)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.branch(x)
        return self.head(rep).squeeze(-1)


class DualFrequencySWiMRegressor(nn.Module):
    def __init__(self, daily_feature_dim: int, weekly_feature_dim: int):
        super().__init__()
        self.daily_branch = SWiMBranch(feature_dim=daily_feature_dim, frequency_key="daily")
        self.weekly_branch = SWiMBranch(feature_dim=weekly_feature_dim, frequency_key="weekly")
        self.fusion = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        daily_vec = self.daily_branch(x_daily)
        weekly_vec = self.weekly_branch(x_weekly)
        fused = torch.cat([daily_vec, weekly_vec], dim=1)
        return self.fusion(fused).squeeze(-1)


def make_model(frequency: str, daily_feature_dim: int, weekly_feature_dim: int) -> nn.Module:
    if frequency == "daily_only":
        return SingleFrequencySWiMRegressor(feature_dim=daily_feature_dim, frequency_key="daily")
    if frequency == "weekly_only":
        return SingleFrequencySWiMRegressor(feature_dim=weekly_feature_dim, frequency_key="weekly")
    if frequency == "daily_weekly":
        return DualFrequencySWiMRegressor(daily_feature_dim=daily_feature_dim, weekly_feature_dim=weekly_feature_dim)
    raise ValueError(f"Unknown frequency setting: {frequency}")


# =====================================================================
# TRAIN / EVAL (mirrors panel_baseline_vanilla_transformer.py structure)
# =====================================================================


def clear_swim_partial_outputs() -> None:
    artifacts = [
        AUDIT_DIR / "panel_swim_test_predictions.csv",
        AUDIT_DIR / "panel_swim_summary_metrics.csv",
        AUDIT_DIR / "panel_swim_training_history.csv",
        AUDIT_DIR / "panel_swim_rank_ic_by_date.csv",
        AUDIT_DIR / "panel_swim_vs_baselines_summary.csv",
        AUDIT_DIR / "panel_swim_final_summary.txt",
    ]
    for path in artifacts:
        if path.exists():
            path.unlink()


def smoke_test_models(ds, stats: dict, device: torch.device) -> dict:
    results = {}
    for frequency in FREQUENCY_SETTINGS:
        if frequency == "daily_only":
            X_all = apply_normalizer(ds, stats, "daily_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
        elif frequency == "weekly_only":
            X_all = apply_normalizer(ds, stats, "weekly_only")
            X = torch.tensor(X_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=X.shape[-1], weekly_feature_dim=X.shape[-1]).to(device)
        else:
            Xd_all, Xw_all = apply_normalizer(ds, stats, "daily_weekly")
            xd = torch.tensor(Xd_all[:8], dtype=torch.float32, device=device)
            xw = torch.tensor(Xw_all[:8], dtype=torch.float32, device=device)
            y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)
            model = make_model(frequency, daily_feature_dim=xd.shape[-1], weekly_feature_dim=xw.shape[-1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if frequency == "daily_weekly":
            pred = model(xd, xw)
        else:
            pred = model(X)
        assert pred.shape == y.shape, f"{frequency} output shape mismatch: {pred.shape} vs {y.shape}"
        loss = nn.HuberLoss(delta=1.0)(pred, y)
        loss.backward()
        all_params = [p for p in model.parameters() if p.requires_grad]
        grads_present = all(p.grad is not None and torch.isfinite(p.grad).all() for p in all_params)
        assert grads_present, f"{frequency} backprop produced missing/non-finite gradients on some parameters"
        optimizer.zero_grad()
        results[frequency] = {"ok": True, "param_count": count_trainable_parameters(model)}
    return results


def train_and_eval_frequency(ds, frequency: str, horizon: int, stats: dict, trace_nan: bool = False) -> dict:
    set_seed(SEED)
    if trace_nan:
        print(f"[TRACE] torch.initial_seed()={torch.initial_seed()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if trace_nan:
        print(f"[TRACE] selected device={device.type.upper()}")
        print("[TRACE] DEBUG_NAN_MODE = False")
        assert DEBUG_NAN_MODE is False
    experiment_start = time.perf_counter()

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    if frequency == "daily_only":
        X_all = apply_normalizer(ds, stats, "daily_only")
        X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
        X_test = torch.tensor(X_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
        X_test = torch.tensor(X_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="weekly").to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "daily_weekly":
        X_daily_all, X_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        X_d_train = torch.tensor(X_daily_all[train_mask], dtype=torch.float32)
        X_d_val = torch.tensor(X_daily_all[val_mask], dtype=torch.float32)
        X_d_test = torch.tensor(X_daily_all[test_mask], dtype=torch.float32)
        X_w_train = torch.tensor(X_weekly_all[train_mask], dtype=torch.float32)
        X_w_val = torch.tensor(X_weekly_all[val_mask], dtype=torch.float32)
        X_w_test = torch.tensor(X_weekly_all[test_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        y_test = ds.y[horizon][test_mask].copy()
        model = DualFrequencySWiMRegressor(
            daily_feature_dim=X_daily_all.shape[-1],
            weekly_feature_dim=X_weekly_all.shape[-1],
        ).to(device)
        train_loader = DataLoader(TensorDataset(X_d_train, X_w_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_d_val, X_w_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_daily, x_weekly, y_batch):
            x_daily = x_daily.to(device)
            x_weekly = x_weekly.to(device)
            y_batch = y_batch.to(device)
            return model(x_daily, x_weekly), y_batch

    else:
        raise ValueError(f"Unknown frequency setting: {frequency}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)

    if trace_nan:
        param_total, param_sum, param_sq_sum = _model_parameter_fingerprint(model)
        print(f"[TRACE] model_total_parameters={param_total}")
        print(f"[TRACE] model_param_sum={param_sum:.12f}")
        print(f"[TRACE] model_param_sq_sum={param_sq_sum:.12f}")
        print(f"[TRACE] {_trace_rng_fingerprint_summary('torch_rng_state_before_train_loader')}")

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        if trace_nan and epoch > 1:
            break
        model.train()
        train_loss_sum = 0.0
        train_weight = 0
        epoch_start = time.perf_counter()

        if frequency == "daily_weekly":
            for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                if trace_nan:
                    if not torch.isfinite(x_d_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(x_w_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(y_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if batch_idx < 3:
                        _trace_first_batch_fingerprint(batch_idx, x_d_batch, y_batch)

                logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                if trace_nan and not torch.isfinite(logits).all():
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=forward")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "forward", "frequency": frequency, "horizon": horizon}
                loss = criterion(logits, target)
                if trace_nan and not torch.isfinite(loss):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=loss")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "loss", "frequency": frequency, "horizon": horizon}
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                if trace_nan:
                    bad_grad_names = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                bad_grad_names.append((name, "missing"))
                            elif not torch.isfinite(param.grad).all():
                                bad_grad_names.append((name, "nonfinite"))
                    if bad_grad_names:
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=backward")
                        for name, reason in bad_grad_names:
                            print(f"[TRACE] gradient failure: {name} -> {reason}")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "backward", "frequency": frequency, "horizon": horizon}
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if trace_nan and not torch.isfinite(grad_norm):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=gradient clipping")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "gradient clipping", "frequency": frequency, "horizon": horizon}
                optimizer.step()
                if trace_nan:
                    for name, param in model.named_parameters():
                        if param.requires_grad and not torch.isfinite(param).all():
                            print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=optimizer step")
                            print(f"[TRACE] parameter failure: {name}")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "optimizer step", "frequency": frequency, "horizon": horizon}
        else:
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                if trace_nan:
                    if not torch.isfinite(x_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if not torch.isfinite(y_batch).all():
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=input")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "input", "frequency": frequency, "horizon": horizon}
                    if batch_idx < 3:
                        _trace_first_batch_fingerprint(batch_idx, x_batch, y_batch)

                logits, target = batched_forward(x_batch, y_batch)
                if trace_nan and not torch.isfinite(logits).all():
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=forward")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "forward", "frequency": frequency, "horizon": horizon}
                loss = criterion(logits, target)
                if trace_nan and not torch.isfinite(loss):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=loss")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "loss", "frequency": frequency, "horizon": horizon}
                batch_size_value = y_batch.shape[0]
                train_loss_sum += float(loss.item()) * batch_size_value
                train_weight += int(batch_size_value)
                loss.backward()
                if trace_nan:
                    bad_grad_names = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                bad_grad_names.append((name, "missing"))
                            elif not torch.isfinite(param.grad).all():
                                bad_grad_names.append((name, "nonfinite"))
                    if bad_grad_names:
                        print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=backward")
                        for name, reason in bad_grad_names:
                            print(f"[TRACE] gradient failure: {name} -> {reason}")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "backward", "frequency": frequency, "horizon": horizon}
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if trace_nan and not torch.isfinite(grad_norm):
                    print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=gradient clipping")
                    return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "gradient clipping", "frequency": frequency, "horizon": horizon}
                optimizer.step()
                if trace_nan:
                    for name, param in model.named_parameters():
                        if param.requires_grad and not torch.isfinite(param).all():
                            print(f"[TRACE] first failing batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=optimizer step")
                            print(f"[TRACE] parameter failure: {name}")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "optimizer step", "frequency": frequency, "horizon": horizon}

        model.eval()
        val_loss_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            if frequency == "daily_weekly":
                for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(val_loader):
                    if trace_nan:
                        if not torch.isfinite(x_d_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(x_w_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(y_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                    if trace_nan and not torch.isfinite(logits).all():
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_loss = criterion(logits, target)
                    if trace_nan and not torch.isfinite(batch_loss):
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_size_value = y_batch.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size_value
                    val_weight += int(batch_size_value)
            else:
                for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                    if trace_nan:
                        if not torch.isfinite(x_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                        if not torch.isfinite(y_batch).all():
                            print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                            return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    logits, target = batched_forward(x_batch, y_batch)
                    if trace_nan and not torch.isfinite(logits).all():
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_loss = criterion(logits, target)
                    if trace_nan and not torch.isfinite(batch_loss):
                        print(f"[TRACE] first failing validation batch={batch_idx} epoch={epoch} frequency={frequency} horizon={horizon}d device={device.type.upper()} stage=validation")
                        return {"trace_status": "failed", "epoch": epoch, "batch": batch_idx, "stage": "validation", "frequency": frequency, "horizon": horizon}
                    batch_size_value = y_batch.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size_value
                    val_weight += int(batch_size_value)

        train_loss = train_loss_sum / train_weight if train_weight else float("nan")
        val_loss = val_loss_sum / val_weight if val_weight else float("nan")
        if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)):
            raise RuntimeError(
                f"Non-finite loss detected for {frequency} / {horizon}d at epoch {epoch}: "
                f"train_loss={train_loss}, val_loss={val_loss}"
            )

        if trace_nan:
            print(f"[TRACE] trace_train_loss={train_loss:.12f}")
            print(f"[TRACE] trace_val_loss={val_loss:.12f}")
            print("[TRACE] FORMAL TRACE EPOCH + VALIDATION BOTH FINITE")
            return {
                "trace_status": "ok",
                "frequency": frequency,
                "horizon": horizon,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

        epochs_trained = epoch
        history_rows.append(
            {
                "frequency_setting": frequency,
                "horizon": f"{horizon}d",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_seconds": time.perf_counter() - epoch_start,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | best_epoch={best_epoch} | "
            f"patience={patience_counter}/{EARLY_STOPPING_PATIENCE} | "
            f"elapsed={format_elapsed(time.perf_counter() - epoch_start)}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            print(f"Best validation checkpoint = epoch {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"No valid training state for {frequency}, horizon={horizon}d")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if frequency == "daily_weekly":
            preds = []
            for x_d_batch, x_w_batch, _ in DataLoader(
                TensorDataset(X_d_test, X_w_test, torch.tensor(y_test, dtype=torch.float32)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
                preds.append(model(x_d_batch.to(device), x_w_batch.to(device)).cpu().numpy())
            y_pred = np.concatenate(preds)
        else:
            preds = []
            for x_batch, _ in DataLoader(
                TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
                preds.append(model(x_batch.to(device)).cpu().numpy())
            y_pred = np.concatenate(preds)

    if not np.all(np.isfinite(y_pred)):
        raise RuntimeError(f"Non-finite test predictions detected for {frequency} / {horizon}d")

    sanity_metric = evaluate_predictions(
        y_true=y_test, y_pred=y_pred, ticker=ds.ticker[test_mask], anchor_date=ds.anchor_date[test_mask]
    )
    ic_df = rank_ic_by_date(y_test, y_pred, ds.ticker[test_mask], ds.anchor_date[test_mask])
    rank_summary = compute_rank_ic_summary(ic_df)
    no_extreme_scale_pathology = prediction_scale_looks_normal(
        sanity_metric["pooled_pred_std"], sanity_metric["pooled_true_std"]
    )
    experiment_elapsed = time.perf_counter() - experiment_start

    return {
        "model": "swim_transformer",
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "seed": SEED,
        "n_trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "runtime_seconds": experiment_elapsed,
        "val_loss": best_val_loss,
        "test_mse": sanity_metric["pooled_mse"],
        "test_mae": sanity_metric["pooled_mae"],
        "test_corr": sanity_metric["pooled_corr"],
        "ticker_avg_mse": sanity_metric["ticker_avg_mse"],
        "ticker_avg_mae": sanity_metric["ticker_avg_mae"],
        "ticker_avg_corr": sanity_metric["ticker_avg_corr"],
        "direction_accuracy": sanity_metric["pooled_da"],
        "pred_std": sanity_metric["pooled_pred_std"],
        "true_std": sanity_metric["pooled_true_std"],
        "pred_std_true_std_ratio": sanity_metric["pooled_pred_std_over_true_std"],
        "mean_rank_ic": rank_summary["mean_rank_ic"],
        "median_rank_ic": rank_summary["median_rank_ic"],
        "rank_ic_std": rank_summary["rank_ic_std"],
        "positive_ic_ratio": rank_summary["positive_ic_ratio"],
        "rank_icir": rank_summary["rank_icir"],
        "n_valid_ic_dates": rank_summary["n_valid_ic_dates"],
        "y_pred": y_pred,
        "y_true": y_test,
        "ticker": ds.ticker[test_mask],
        "anchor_date": ds.anchor_date[test_mask],
        "rank_ic_by_date": ic_df,
        "history": history_rows,
        "no_extreme_scale_pathology": no_extreme_scale_pathology,
    }


def save_prediction_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for i in range(len(rr["ticker"])):
            rows.append(
                {
                    "ticker": rr["ticker"][i],
                    "anchor_date": rr["anchor_date"][i],
                    "horizon": rr["horizon"],
                    "frequency_setting": rr["frequency_setting"],
                    "y_true": rr["y_true"][i],
                    "y_pred": rr["y_pred"][i],
                    "seed": rr["seed"],
                    "best_epoch": rr["best_epoch"],
                    "model": rr["model"],
                }
            )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_swim_test_predictions.csv"
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_summary_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        rows.append(
            {
                "model": rr["model"],
                "frequency_setting": rr["frequency_setting"],
                "horizon": rr["horizon"],
                "seed": rr["seed"],
                "n_trainable_parameters": rr.get("n_trainable_parameters", np.nan),
                "best_epoch": rr["best_epoch"],
                "epochs_trained": rr.get("epochs_trained", rr["best_epoch"]),
                "runtime_seconds": rr.get("runtime_seconds", np.nan),
                "val_loss": rr["val_loss"],
                "test_mse": rr["test_mse"],
                "test_mae": rr["test_mae"],
                "test_corr": rr["test_corr"],
                "ticker_avg_mse": rr["ticker_avg_mse"],
                "ticker_avg_mae": rr["ticker_avg_mae"],
                "ticker_avg_corr": rr["ticker_avg_corr"],
                "mean_rank_ic": rr["mean_rank_ic"],
                "median_rank_ic": rr["median_rank_ic"],
                "rank_ic_std": rr["rank_ic_std"],
                "positive_ic_ratio": rr["positive_ic_ratio"],
                "rank_icir": rr["rank_icir"],
                "n_valid_ic_dates": rr.get("n_valid_ic_dates", np.nan),
                "direction_accuracy": rr["direction_accuracy"],
                "pred_std": rr["pred_std"],
                "true_std": rr["true_std"],
                "pred_std_true_std_ratio": rr["pred_std_true_std_ratio"],
                "no_extreme_scale_pathology": rr.get("no_extreme_scale_pathology", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_swim_summary_metrics.csv"
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_training_history(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for epoch_row in rr["history"]:
            rows.append(
                {
                    "frequency_setting": epoch_row["frequency_setting"],
                    "horizon": epoch_row["horizon"],
                    "epoch": epoch_row["epoch"],
                    "train_loss": epoch_row["train_loss"],
                    "val_loss": epoch_row["val_loss"],
                    "elapsed_seconds": epoch_row.get("elapsed_seconds", np.nan),
                }
            )
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_swim_training_history.csv"
    if append_mode and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return df


def save_rank_ic_by_date(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for res in results:
        df = res["rank_ic_by_date"].copy()
        df["frequency_setting"] = res["frequency_setting"]
        df["horizon"] = res["horizon"]
        df["model"] = "swim_transformer"
        rows.append(df.reindex(columns=["model", "frequency_setting", "horizon", "anchor_date", "n_tickers", "rank_ic"]))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path = AUDIT_DIR / "panel_swim_rank_ic_by_date.csv"
    if append_mode and path.exists():
        out.to_csv(path, mode="a", header=False, index=False)
    else:
        out.to_csv(path, index=False)
    return out


def append_experiment_outputs(res: dict) -> None:
    save_summary_rows([res], append_mode=True)
    save_prediction_rows([res], append_mode=True)
    save_training_history([res], append_mode=True)
    save_rank_ic_by_date([res], append_mode=True)


def build_baseline_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Comparison table against naive / Ridge / LSTM / Vanilla Transformer.

    PathFormer is intentionally excluded: no formal panel PathFormer benchmark
    exists yet. Metrics are kept separate (MSE/MAE, Corr, Rank IC, DA,
    PredStd/TrueStd) rather than collapsed into a single "winner" column.
    """
    naive_path = AUDIT_DIR / "panel_naive_baseline_summary.csv"
    ridge_path = AUDIT_DIR / "panel_ridge_solver_final_check.csv"
    lstm_path = AUDIT_DIR / "panel_lstm_summary_metrics.csv"
    vanilla_path = AUDIT_DIR / "panel_transformer_summary_metrics.csv"

    naive_df = pd.read_csv(naive_path) if naive_path.exists() else None
    ridge_df = pd.read_csv(ridge_path) if ridge_path.exists() else None
    if ridge_df is not None:
        ridge_df = ridge_df[ridge_df["solver_label"] == "svd"].copy()
    lstm_df = pd.read_csv(lstm_path) if lstm_path.exists() else None
    vanilla_df = pd.read_csv(vanilla_path) if vanilla_path.exists() else None

    rows = []
    for _, row in summary_df.iterrows():
        freq = row["frequency_setting"]
        horizon = row["horizon"]
        out = {
            "model": row["model"],
            "frequency_setting": freq,
            "horizon": horizon,
            "seed": row["seed"],
            "n_trainable_parameters": row["n_trainable_parameters"],
            "test_mse": row["test_mse"],
            "test_mae": row["test_mae"],
            "test_corr": row["test_corr"],
            "mean_rank_ic": row["mean_rank_ic"],
            "direction_accuracy": row["direction_accuracy"],
            "pred_std_true_std_ratio": row["pred_std_true_std_ratio"],
            "naive_available": bool(naive_df is not None),
            "ridge_available": bool(ridge_df is not None),
            "lstm_available": bool(lstm_df is not None),
            "vanilla_available": bool(vanilla_df is not None),
        }

        if naive_df is not None:
            matched = naive_df[naive_df["horizon"] == horizon].copy()
            if not matched.empty:
                matched["model"] = matched["model"].astype(str).str.lower()
                zero_row = matched[matched["model"] == "zero"]
                train_mean_row = matched[matched["model"] == "train_mean"]
                out["zero_mse"] = float(zero_row["pooled_mse"].iloc[0]) if not zero_row.empty else np.nan
                out["train_mean_mse"] = float(train_mean_row["pooled_mse"].iloc[0]) if not train_mean_row.empty else np.nan
                naive_values = [v for v in [out.get("zero_mse"), out.get("train_mean_mse")] if pd.notna(v)]
                out["best_naive_mse"] = float(np.min(naive_values)) if naive_values else np.nan
            else:
                out["zero_mse"] = np.nan
                out["train_mean_mse"] = np.nan
                out["best_naive_mse"] = np.nan

        if ridge_df is not None:
            matched = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon)]
            out["ridge_mse"] = float(matched["test_mse"].iloc[0]) if not matched.empty else np.nan
            out["ridge_corr"] = float(matched["test_corr"].iloc[0]) if not matched.empty else np.nan
            out["ridge_mean_rank_ic"] = float(matched["cross_sectional_rank_ic"].iloc[0]) if not matched.empty else np.nan

        if lstm_df is not None:
            matched = lstm_df[(lstm_df["frequency_setting"] == freq) & (lstm_df["horizon"] == horizon) & (lstm_df["model"] == "lstm")]
            out["lstm_mse"] = float(matched["test_mse"].iloc[0]) if not matched.empty else np.nan
            out["lstm_corr"] = float(matched["test_corr"].iloc[0]) if not matched.empty else np.nan
            out["lstm_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0]) if not matched.empty else np.nan

        if vanilla_df is not None:
            matched = vanilla_df[(vanilla_df["frequency_setting"] == freq) & (vanilla_df["horizon"] == horizon)]
            out["vanilla_mse"] = float(matched["test_mse"].iloc[0]) if not matched.empty else np.nan
            out["vanilla_corr"] = float(matched["test_corr"].iloc[0]) if not matched.empty else np.nan
            out["vanilla_mean_rank_ic"] = float(matched["mean_rank_ic"].iloc[0]) if not matched.empty else np.nan

        rows.append(out)
    return pd.DataFrame(rows)


def write_final_summary(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    lines = []
    lines.append("SWIM-STYLE TRANSFORMER PANEL CONTROL SUMMARY")
    lines.append("")
    lines.append("Architecture:")
    lines.append("  - frequency-specific daily/weekly SWiM branches, late concat fusion")
    lines.append("  - 2 window blocks per branch: block1 local (shift=0), block2 shifted-window")
    lines.append("  - Daily window=10 shift=5; Weekly window=4 shift=2 (development baseline)")
    lines.append("  - Post-LN residual structure; fixed sinusoidal positional encoding")
    lines.append("  - zero padding + explicit validity mask; masked mean pooling over real timesteps")
    lines.append("  - correct shifted-window boundary mask (no wrap-around leakage)")
    lines.append("  - no router, no adaptive/multi-scale mechanism, no gated/cross-attention fusion")
    lines.append("")
    lines.append("Best MSE by horizon:")
    for horizon in ["5d", "10d", "20d"]:
        dfx = summary_df[summary_df["horizon"] == horizon]
        if not dfx.empty:
            best = dfx.loc[dfx["test_mse"].idxmin()]
            lines.append(f"  {horizon}: {best['frequency_setting']} (mse={best['test_mse']:.6f})")
    lines.append("")
    lines.append("Best mean Rank IC by horizon:")
    for horizon in ["5d", "10d", "20d"]:
        dfx = summary_df[summary_df["horizon"] == horizon]
        if not dfx.empty:
            best = dfx.loc[dfx["mean_rank_ic"].idxmax()]
            lines.append(f"  {horizon}: {best['frequency_setting']} (mean_rank_ic={best['mean_rank_ic']:.6f})")
    lines.append("")
    path = AUDIT_DIR / "panel_swim_final_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not comparison_df.empty:
        pass  # comparison_df is saved separately by the caller


def full_benchmark(ds, stats: dict) -> None:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    total_experiments = len(FREQUENCY_SETTINGS) * len(HORIZONS)
    script_start = time.perf_counter()

    print("============================================================")
    print("SWIM-STYLE TRANSFORMER PANEL BASELINE RUN SETUP")
    print(f"device = {device.type.upper()}")
    print(f"total_experiments = {total_experiments}")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"learning_rate = {LEARNING_RATE}")
    print(f"max_epochs = {MAX_EPOCHS}")
    print(f"early_stopping_patience = {EARLY_STOPPING_PATIENCE}")
    print("============================================================")

    clear_swim_partial_outputs()
    print("Fresh-run initialization: cleared prior SWiM partial output files.")

    results = []
    experiment_runtimes = []
    for experiment_idx, frequency in enumerate(FREQUENCY_SETTINGS, start=1):
        for horizon in HORIZONS:
            exp_counter = (experiment_idx - 1) * len(HORIZONS) + list(HORIZONS).index(horizon) + 1
            print(f"\n[Experiment {exp_counter}/{total_experiments}]")
            print(f"frequency_setting = {frequency}")
            print(f"horizon = {horizon}d")
            res = train_and_eval_frequency(ds, frequency, int(horizon), stats)
            results.append(res)
            experiment_runtimes.append(float(res["runtime_seconds"]))
            append_experiment_outputs(res)

            completed = len(experiment_runtimes)
            total_elapsed = time.perf_counter() - script_start
            mean_experiment_time = float(np.mean(experiment_runtimes)) if experiment_runtimes else 0.0
            remaining_experiments = max(total_experiments - completed, 0)
            estimated_remaining = mean_experiment_time * remaining_experiments
            print(f"Experiment runtime: {format_elapsed(res['runtime_seconds'])}")
            print(f"n_trainable_parameters = {res['n_trainable_parameters']}")
            print(f"best_epoch = {res['best_epoch']}")
            print(f"best_validation_loss = {res['val_loss']:.6f}")
            print(f"test_mse = {res['test_mse']:.6f}")
            print(f"mean_rank_ic = {res['mean_rank_ic']:.6f}")
            print(f"PredStd/TrueStd = {res['pred_std_true_std_ratio']:.4f}")
            print(f"no_extreme_scale_pathology = {res['no_extreme_scale_pathology']}")
            print(f"\nProgress: {completed}/{total_experiments} experiments completed")
            print(f"Total elapsed: {format_elapsed(total_elapsed)}")
            print(f"Estimated remaining: ~{format_elapsed(estimated_remaining)} (approximate; early stopping epochs differ)")

    summary_df = save_summary_rows(results)
    save_prediction_rows(results)
    save_training_history(results)
    comparison_df = build_baseline_comparison(summary_df)
    comparison_df.to_csv(AUDIT_DIR / "panel_swim_vs_baselines_summary.csv", index=False)
    write_final_summary(summary_df, comparison_df)

    print("\n===========================================================")
    print("SWIM-STYLE TRANSFORMER PANEL CONTROL COMPLETE")
    print(f"Total runtime: {format_elapsed(time.perf_counter() - script_start)}")
    print(f"Completed experiments: {len(results)}/{total_experiments}")
    print("===========================================================")


# =====================================================================
# SMOKE / AUDIT TESTS
# =====================================================================


def run_check(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as exc:
        print(f"FAIL: {name}: {exc}")
        raise
    print(f"PASS: {name}")


def test_positional_encoding_shape() -> None:
    pe = PositionalEncoding(d_model=D_MODEL, max_len=FIXED_MAX_LEN, dropout=0.0)
    x = torch.zeros(2, 10, D_MODEL)
    out = pe(x)
    assert out.shape == x.shape, f"unexpected PE output shape: {out.shape}"


def test_window_partition_reverse() -> None:
    torch.manual_seed(0)
    for T, w in [(90, 10), (28, 4)]:  # divisible daily case; divisible post-pad weekly case
        x = torch.randn(3, T, 6)
        windows = window_partition_1d(x, w)
        recon = window_reverse_1d(windows, w, B=3, T=T)
        max_err = (recon - x).abs().max().item()
        assert max_err < 1e-6, f"reconstruction error too large for T={T}, w={w}: {max_err}"
        print(f"  window_partition/reverse T={T} w={w}: max_abs_reconstruction_error={max_err:.2e}")

    # non-divisible case: manually zero-pad, partition/reverse, then strip padding
    T, w = 26, 4
    pad_len = compute_pad_len(T, w)
    x = torch.randn(3, T, 6)
    x_padded = torch.cat([x, torch.zeros(3, pad_len, 6)], dim=1)
    windows = window_partition_1d(x_padded, w)
    recon_padded = window_reverse_1d(windows, w, B=3, T=T + pad_len)
    recon = recon_padded[:, :T, :]
    max_err = (recon - x).abs().max().item()
    assert max_err < 1e-6, f"reconstruction error too large for padded T={T}, w={w}: {max_err}"
    print(f"  window_partition/reverse (non-divisible) T={T} w={w} pad_len={pad_len}: max_abs_reconstruction_error={max_err:.2e}")


def test_shift_mask() -> None:
    """Toy example proving the shifted-window boundary mask matches the original-token ordering."""
    T_pad, w, shift = 20, 10, 5
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    attn_mask = build_window_masks(valid_mask, w, shift)

    original = torch.arange(T_pad)
    shifted = torch.roll(original, shifts=-shift)
    expected_shifted = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4])
    assert torch.equal(shifted, expected_shifted), f"unexpected shifted index order: {shifted.tolist()}"

    win0 = shifted[:w]
    win1 = shifted[w:]
    assert torch.equal(win0, torch.arange(5, 15)), f"unexpected shifted window0: {win0.tolist()}"
    assert torch.equal(win1, torch.tensor([15, 16, 17, 18, 19, 0, 1, 2, 3, 4])), f"unexpected shifted window1: {win1.tolist()}"

    n_win = T_pad // w
    assert attn_mask.shape == (n_win, w, w), f"attn_mask shape mismatch: {attn_mask.shape}"
    region_id = compute_shift_region_ids(T_pad, w, shift, device=torch.device("cpu"))
    expected = torch.tensor([0] * 10 + [1] * 5 + [2] * 5)
    assert torch.equal(region_id, expected), f"unexpected region ids: {region_id.tolist()}"

    win0_mask = attn_mask[0]
    win1_mask = attn_mask[1]
    assert torch.all(win0_mask == 0.0), "window0 should have no shift-boundary blocking"
    assert torch.all(win1_mask[:5, :5] == 0.0), "same-region (first half of window1) pairs must be allowed"
    assert torch.all(win1_mask[5:, 5:] == 0.0), "same-region (second half of window1) pairs must be allowed"
    assert torch.all(torch.isinf(win1_mask[:5, 5:])), "wrap-around region1<->region2 pairs must be masked with -inf"
    assert torch.all(torch.isinf(win1_mask[5:, :5])), "wrap-around region2<->region1 pairs must be masked with -inf"
    assert torch.all(torch.isfinite(win1_mask[win1_mask == 0.0])), "allowed keys in valid queries must remain finite"

    # Explicitly verify original-index mapping through cyclic shift + window partition.
    allowed_examples = [(15, 19), (0, 4), (19, 15), (4, 0)]
    blocked_examples = [(15, 0), (19, 4), (0, 15), (4, 19)]
    for a, b in allowed_examples:
        i, j = shifted.tolist().index(a), shifted.tolist().index(b)
        assert win1_mask[i % w, j % w] == 0.0, f"expected {a} <-> {b} to be allowed"
    for a, b in blocked_examples:
        i, j = shifted.tolist().index(a), shifted.tolist().index(b)
        assert torch.isinf(win1_mask[i % w, j % w]), f"expected {a} <-> {b} to be blocked"

    print("  original indices:", original.tolist())
    print("  shifted indices:", shifted.tolist())
    print("  window0:", win0.tolist())
    print("  window1:", win1.tolist())
    print("  representative allowed pairs:", [(15, 19), (0, 4)])
    print("  representative blocked wrap-around pairs:", [(15, 0), (19, 4)])


def test_hard_mask_exactness() -> None:
    """For real queries, all blocked attention entries are exactly -inf and no row is fully masked."""
    T_pad, w, shift = 20, 10, 5
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    attn_mask = build_window_masks(valid_mask, w, shift)
    win1 = attn_mask[1]

    assert torch.all(win1[:5, :5] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(win1[5:, 5:] == 0.0), "same-region valid pairs must remain exactly 0.0"
    assert torch.all(torch.isinf(win1[:5, 5:])), "real-query cross-region pairs must be exactly -inf"
    assert torch.all(torch.isinf(win1[5:, :5])), "real-query cross-region pairs must be exactly -inf"
    for row in win1:
        assert torch.any(torch.isfinite(row)), "no real-query attention row may be entirely -inf"

    T_pad2, w2, shift2 = 28, 4, 2
    original_indices = torch.arange(T_pad2)
    shifted_indices = torch.roll(original_indices, shifts=-shift2)
    expected_final_window = torch.tensor([26, 27, 0, 1])
    assert torch.equal(shifted_indices[-w2:], expected_final_window), f"unexpected final weekly shifted window: {shifted_indices[-w2:].tolist()}"

    valid_mask2 = torch.ones(T_pad2, dtype=torch.bool)
    valid_mask2[24:] = False
    shifted_validity = torch.roll(valid_mask2, shifts=-shift2)
    expected_validity = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert torch.equal(shifted_validity[-w2:], expected_validity), f"unexpected final weekly shifted validity: {shifted_validity[-w2:].tolist()}"

    attn_mask2 = build_window_masks(valid_mask2, w2, shift2)
    last_window = attn_mask2[6]
    assert torch.all(last_window[0, 0] == 0.0), "first padded query in final weekly window must retain dummy key"
    assert torch.all(torch.isinf(last_window[0, 1:])), "first padded query must not attend real keys"
    assert torch.all(last_window[1, 0] == 0.0), "second padded query in final weekly window must retain dummy key"
    assert torch.all(torch.isinf(last_window[1, 1:])), "second padded query must not attend real keys"
    assert torch.all(last_window[2, 2:] == 0.0), "real query at original token 0 must retain its valid same-region keys"
    assert torch.all(torch.isinf(last_window[2, :2])), "real query at original token 0 must not attend padded keys"
    assert torch.all(last_window[3, 2:] == 0.0), "real query at original token 1 must retain its valid same-region keys"
    assert torch.all(torch.isinf(last_window[3, :2])), "real query at original token 1 must not attend padded keys"
    assert torch.all(torch.any(torch.isfinite(last_window), dim=-1)), "every final-window query row must retain at least one finite entry"
    print("  exact hard-mask audit: blocked real-query entries are -inf; valid real pairs stay 0.0; no fully masked real row")
    print("  padded-key block check on final weekly window: real query rows keep only valid same-region keys available")


def test_weekly_shifted_padding_geometry() -> None:
    """Verify the exact Weekly shifted-window prompt geometry after zero-padding."""
    T, w, shift = 26, 4, 2
    pad_len = compute_pad_len(T, w)
    T_pad = T + pad_len
    assert T_pad == 28 and pad_len == 2, f"unexpected weekly padding geometry: pad_len={pad_len}, T_pad={T_pad}"

    original_indices = torch.arange(T_pad)
    shifted_indices = torch.roll(original_indices, shifts=-shift)
    last_window = shifted_indices[-w:]
    expected_window = torch.tensor([26, 27, 0, 1])
    assert torch.equal(last_window, expected_window), f"unexpected final shifted window: {last_window.tolist()}"

    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T:] = False
    shifted_validity = torch.roll(valid_mask, shifts=-shift)
    expected_validity = torch.tensor([False, False, True, True], dtype=torch.bool)
    assert torch.equal(shifted_validity[-w:], expected_validity), f"unexpected final shifted validity: {shifted_validity[-w:].tolist()}"

    attn_mask = build_window_masks(valid_mask, w, shift)
    final_window_mask = attn_mask[6]
    assert torch.all(torch.isfinite(final_window_mask[0, :1])), "padded query row 0 must keep one finite dummy key"
    assert torch.all(torch.isinf(final_window_mask[0, 1:])), "padded query row 0 must not attend real keys"
    assert torch.all(torch.isfinite(final_window_mask[1, :1])), "padded query row 1 must keep one finite dummy key"
    assert torch.all(torch.isinf(final_window_mask[1, 1:])), "padded query row 1 must not attend real keys"
    assert torch.all(final_window_mask[2, 2:] == 0.0), "real query 0 must be allowed to attend real token 0 and 1"
    assert torch.all(torch.isinf(final_window_mask[2, :2])), "real query 0 must not attend padded keys"
    assert torch.all(final_window_mask[3, 2:] == 0.0), "real query 1 must be allowed to attend real token 0 and 1"
    assert torch.all(torch.isinf(final_window_mask[3, :2])), "real query 1 must not attend padded keys"
    print("  weekly shifted-padding geometry:")
    print(f"    original: 0..{T-1} then PAD{T},{T + 1}")
    print(f"    shifted last window: {last_window.tolist()}")
    print(f"    shifted validity: {shifted_validity[-w:].tolist()}")
    print("    final weekly last-window mask behavior: padded rows keep dummy key; real rows allow only same-region real keys")


def test_shifted_validity_alignment() -> None:
    """The row validity used by build_window_masks() must match actual shifted-window token validity."""
    T_pad, w, shift = 28, 4, 2
    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T_pad - 2:] = False
    shifted_validity = torch.roll(valid_mask, shifts=-shift)
    attn_mask = build_window_masks(valid_mask, w, shift)
    query_valid = shifted_validity.view(T_pad // w, w)

    for win_idx in range(T_pad // w):
        for q_idx in range(w):
            actual_query_valid = bool(shifted_validity[win_idx * w + q_idx])
            assert bool(query_valid[win_idx, q_idx]) == actual_query_valid, (
                f"shifted query validity mismatch at window={win_idx}, query={q_idx}: "
                f"expected {actual_query_valid}, got {bool(query_valid[win_idx, q_idx])}"
            )
            row = attn_mask[win_idx, q_idx]
            assert torch.any(torch.isfinite(row)), (
                f"mask row for shifted-window query ({win_idx}, {q_idx}) is entirely -inf"
            )
    print("  shifted query/key validity alignment: rows correspond exactly to the shifted-window tokens they represent")


def test_attention_mask_row_safety() -> None:
    """Every row passed to MultiheadAttention must keep at least one finite entry."""
    cases = [(90, 10, 5, torch.ones(90, dtype=torch.bool)), (28, 4, 2, torch.cat([torch.ones(26, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)]))]
    for T_pad, w, shift, valid_mask in cases:
        attn_mask = build_window_masks(valid_mask, w, shift)
        assert attn_mask.shape == (T_pad // w, w, w), f"unexpected attn mask shape: {attn_mask.shape}"
        for win_idx in range(attn_mask.shape[0]):
            for q_idx in range(w):
                row = attn_mask[win_idx, q_idx]
                assert torch.any(torch.isfinite(row)), (
                    f"all -inf row in {T_pad=} window={win_idx} query={q_idx}: row={row.tolist()}"
                )
    print("  row safety audit: every real/padded attention row remains finite for Daily and Weekly shifted masks")


def test_daily_shifted_window_geometry() -> None:
    """Daily T=90/window=10/shift=5 should produce the expected wrap-around boundary geometry."""
    T, w, shift = 90, 10, 5
    original = torch.arange(T)
    shifted = torch.roll(original, shifts=-shift)
    last_window = shifted[-w:]
    expected_last_window = torch.tensor([85, 86, 87, 88, 89, 0, 1, 2, 3, 4])
    assert torch.equal(last_window, expected_last_window), f"unexpected daily final shifted window: {last_window.tolist()}"
    allowed = [(85, 89), (0, 4), (89, 85), (4, 0)]
    blocked = [(85, 0), (89, 4), (0, 85), (4, 89)]
    mask = build_window_masks(torch.ones(T, dtype=torch.bool), w, shift)[8]
    for a, b in allowed:
        ia, ib = shifted.tolist().index(a), shifted.tolist().index(b)
        assert mask[ia % w, ib % w] == 0.0, f"expected {a} <-> {b} to be allowed"
    for a, b in blocked:
        ia, ib = shifted.tolist().index(a), shifted.tolist().index(b)
        assert torch.isinf(mask[ia % w, ib % w]), f"expected {a} <-> {b} to be blocked"
    print("  original daily indices:", original[:10].tolist(), "...", original[-10:].tolist())
    print("  shifted daily final window:", last_window.tolist())
    print("  representative allowed pairs:", [(85, 89), (0, 4)])
    print("  representative blocked pairs:", [(85, 0), (89, 4)])


def test_padding_mask_invariance() -> None:
    """Prove padded content cannot leak into real-token outputs or pooling."""
    torch.manual_seed(0)
    T, w, shift, d_model = 26, 4, 2, D_MODEL
    pad_len = compute_pad_len(T, w)
    T_pad = T + pad_len
    assert pad_len == 2 and T_pad == 28, f"unexpected weekly padding geometry: pad_len={pad_len}, T_pad={T_pad}"

    block1 = SWiMWindowBlock(d_model, NHEAD, w, 0, DIM_FEEDFORWARD, dropout=0.0).eval()
    block2 = SWiMWindowBlock(d_model, NHEAD, w, shift, DIM_FEEDFORWARD, dropout=0.0).eval()

    real = torch.randn(2, T, d_model)
    zero_pad = torch.zeros(2, pad_len, d_model)
    garbage_pad = torch.randn(2, pad_len, d_model) * 1e4  # deliberately extreme values

    valid_mask = torch.ones(T_pad, dtype=torch.bool)
    valid_mask[T:] = False

    x_zero = torch.cat([real, zero_pad], dim=1)
    x_garbage = torch.cat([real, garbage_pad], dim=1)

    with torch.no_grad():
        out_zero = block2(block1(x_zero, valid_mask), valid_mask)
        out_garbage = block2(block1(x_garbage, valid_mask), valid_mask)

    assert torch.isfinite(out_zero).all(), "NaN/Inf produced with zero padding"
    assert torch.isfinite(out_garbage).all(), "NaN/Inf produced with garbage padding"

    real_diff = (out_zero[:, :T, :] - out_garbage[:, :T, :]).abs().max().item()
    assert real_diff < 1e-4, f"padded content leaked into real-token outputs: max_abs_diff={real_diff}"

    pad_zero_out = out_zero[:, T:, :]
    pad_garbage_out = out_garbage[:, T:, :]
    assert torch.allclose(pad_zero_out, torch.zeros_like(pad_zero_out), atol=1e-6), "padded positions not hard-zeroed (zero-pad case)"
    assert torch.allclose(pad_garbage_out, torch.zeros_like(pad_garbage_out), atol=1e-6), "padded positions not hard-zeroed (garbage-pad case)"

    pool_zero = out_zero.sum(dim=1) / float(T)
    pool_garbage = out_garbage.sum(dim=1) / float(T)
    pool_diff = (pool_zero - pool_garbage).abs().max().item()
    assert pool_diff < 1e-4, f"padded content leaked into masked mean pooling: max_abs_diff={pool_diff}"

    print(f"  real-token max_abs_diff (zero-pad vs garbage-pad) = {real_diff:.2e}")
    print(f"  padded-position outputs hard-zeroed in both cases -- OK")
    print(f"  pooled representation max_abs_diff (zero-pad vs garbage-pad) = {pool_diff:.2e}")


def test_forward_shapes(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:6], dtype=torch.float32, device=device)
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:6], dtype=torch.float32, device=device)

    daily_model = SingleFrequencySWiMRegressor(feature_dim=X_daily.shape[-1], frequency_key="daily").to(device)
    weekly_model = SingleFrequencySWiMRegressor(feature_dim=X_weekly.shape[-1], frequency_key="weekly").to(device)
    dual_model = DualFrequencySWiMRegressor(daily_feature_dim=X_daily.shape[-1], weekly_feature_dim=X_weekly.shape[-1]).to(device)

    out_daily = daily_model(X_daily)
    out_weekly = weekly_model(X_weekly)
    out_dual = dual_model(X_daily, X_weekly)

    assert out_daily.shape == (6,), f"daily_only output shape: {out_daily.shape}"
    assert out_weekly.shape == (6,), f"weekly_only output shape: {out_weekly.shape}"
    assert out_dual.shape == (6,), f"daily_weekly output shape: {out_dual.shape}"


def test_backward_gradients(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:8], dtype=torch.float32, device=device)
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    y = torch.tensor(ds.y[5][:8], dtype=torch.float32, device=device)

    model = DualFrequencySWiMRegressor(daily_feature_dim=X_daily.shape[-1], weekly_feature_dim=X_weekly.shape[-1]).to(device)
    pred = model(X_daily, X_weekly)
    loss = nn.HuberLoss(delta=1.0)(pred, y)
    loss.backward()

    named_params = list(model.named_parameters())
    missing = [name for name, p in named_params if p.requires_grad and p.grad is None]
    non_finite = [name for name, p in named_params if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f"parameters with no gradient (possible unused head): {missing}"
    assert not non_finite, f"parameters with non-finite gradient: {non_finite}"
    assert len(named_params) > 0, "model has no trainable parameters"


def test_nan_inf(ds, stats: dict, device: torch.device) -> None:
    X_weekly = torch.tensor(apply_normalizer(ds, stats, "weekly_only")[:8], dtype=torch.float32, device=device)
    model = SingleFrequencySWiMRegressor(feature_dim=X_weekly.shape[-1], frequency_key="weekly").to(device)
    model.eval()
    with torch.no_grad():
        out = model(X_weekly)
        seq, valid_mask = model.branch(X_weekly, return_sequence=True)
    assert torch.isfinite(out).all(), "NaN/Inf in weekly_only forward output"
    assert torch.isfinite(seq).all(), "NaN/Inf in weekly encoded sequence (pre-pool)"
    pad_positions = seq[:, ~valid_mask, :]
    assert torch.allclose(pad_positions, torch.zeros_like(pad_positions), atol=1e-6), "padded positions are not exactly zero"


def test_parameter_counts() -> dict:
    daily_model = SingleFrequencySWiMRegressor(feature_dim=5, frequency_key="daily")
    weekly_model = SingleFrequencySWiMRegressor(feature_dim=5, frequency_key="weekly")
    dual_model = DualFrequencySWiMRegressor(daily_feature_dim=5, weekly_feature_dim=5)

    counts = {
        "daily_only": count_trainable_parameters(daily_model),
        "weekly_only": count_trainable_parameters(weekly_model),
        "daily_weekly": count_trainable_parameters(dual_model),
    }
    vanilla_counts = {"daily_only": 67393, "weekly_only": 67393, "daily_weekly": 142977}
    for key in counts:
        ratio = counts[key] / vanilla_counts[key]
        print(f"  {key}: swim_params={counts[key]} vanilla_ref={vanilla_counts[key]} ratio={ratio:.3f}")
        assert counts[key] == vanilla_counts[key], (
            f"SWiM parameter count mismatch for {key}: got {counts[key]}, expected {vanilla_counts[key]}"
        )
        assert np.isclose(ratio, 1.0, atol=1e-12), f"SWiM count ratio for {key} is not 1.0: {ratio}"
    return counts


def test_deterministic_forward(ds, stats: dict, device: torch.device) -> None:
    X_daily = torch.tensor(apply_normalizer(ds, stats, "daily_only")[:4], dtype=torch.float32, device=device)
    model = SingleFrequencySWiMRegressor(feature_dim=X_daily.shape[-1], frequency_key="daily").to(device)
    model.eval()
    with torch.no_grad():
        out1 = model(X_daily)
        out2 = model(X_daily)
    assert torch.equal(out1, out2), "eval-mode forward pass is not deterministic for identical input"


def test_single_config_bookkeeping() -> None:
    """Validate the single-config purge path without launching formal training."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="swim_bookkeeping_"))
    old_audit_dir = AUDIT_DIR
    try:
        globals()["AUDIT_DIR"] = tmp_dir
        stale_files = [
            tmp_dir / "panel_swim_test_predictions.csv",
            tmp_dir / "panel_swim_summary_metrics.csv",
            tmp_dir / "panel_swim_training_history.csv",
            tmp_dir / "panel_swim_rank_ic_by_date.csv",
            tmp_dir / "panel_swim_vs_baselines_summary.csv",
            tmp_dir / "panel_swim_final_summary.txt",
        ]
        for path in stale_files:
            path.write_text("stale", encoding="utf-8")
        clear_swim_partial_outputs()
        assert all(not p.exists() for p in stale_files), "stale SWiM outputs were not cleared"

        summary_df = pd.DataFrame(
            [
                {
                    "model": "swim_transformer",
                    "frequency_setting": "daily_only",
                    "horizon": "5d",
                    "seed": SEED,
                    "n_trainable_parameters": 67393,
                    "test_mse": 1.0,
                    "test_mae": 0.5,
                    "test_corr": 0.2,
                    "mean_rank_ic": 0.01,
                    "direction_accuracy": 0.4,
                    "pred_std_true_std_ratio": 1.0,
                }
            ]
        )
        write_final_summary(summary_df, pd.DataFrame())
        assert (tmp_dir / "panel_swim_final_summary.txt").exists(), "final summary was not refreshed"
        assert "SWIM-STYLE TRANSFORMER PANEL CONTROL SUMMARY" in (tmp_dir / "panel_swim_final_summary.txt").read_text(encoding="utf-8")
        print("  single-config bookkeeping: stale outputs cleared and final summary refreshed")
    finally:
        globals()["AUDIT_DIR"] = old_audit_dir


def run_smoke_validation() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("FROZEN PANEL VALIDATION (SWiM)")
    print(f"train={int(np.sum(ds.split == 'train'))}")
    print(f"val={int(np.sum(ds.split == 'val'))}")
    print(f"test={int(np.sum(ds.split == 'test'))}")
    print(f"device={device.type.upper()}")
    print("============================================================")

    print("\n--- window/shift/padding audit tests ---")
    run_check("positional_encoding_shape", test_positional_encoding_shape)
    run_check("window_partition_reverse", test_window_partition_reverse)
    run_check("shift_boundary_mask", test_shift_mask)
    run_check("hard_mask_exactness", test_hard_mask_exactness)
    run_check("weekly_shifted_padding_geometry", test_weekly_shifted_padding_geometry)
    run_check("shifted_validity_alignment", test_shifted_validity_alignment)
    run_check("attention_mask_row_safety", test_attention_mask_row_safety)
    run_check("daily_shifted_window_geometry", test_daily_shifted_window_geometry)
    run_check("padding_mask_invariance", test_padding_mask_invariance)

    print("\n--- forward/backward/gradient audit tests ---")
    run_check("forward_shapes", lambda: test_forward_shapes(ds, stats, device))
    run_check("backward_gradients", lambda: test_backward_gradients(ds, stats, device))
    run_check("nan_inf", lambda: test_nan_inf(ds, stats, device))
    run_check("deterministic_forward", lambda: test_deterministic_forward(ds, stats, device))

    print("\n--- parameter-count audit ---")
    test_parameter_counts()

    print("\n--- single-config bookkeeping audit ---")
    test_single_config_bookkeeping()

    print("\n--- smoke_test_models (forward+backward per frequency) ---")
    smoke_results = smoke_test_models(ds, stats, device)
    for frequency, info in smoke_results.items():
        print(f"{frequency}: smoke_ok={info['ok']} param_count={info['param_count']}")

    print("\nSmoke validation passed: forward/backward, gradient, mask, and padding checks OK.")
    print("Formal 9-experiment training is intentionally not started in this stage.")


def print_finite_summary(name: str, x: np.ndarray) -> None:
    finite = np.isfinite(x)
    print(f"{name}: shape={x.shape} finite_fraction={finite.mean():.6f} min={np.min(x):.6e} max={np.max(x):.6e} mean={np.mean(x):.6e} std={np.std(x):.6e} max_abs={np.max(np.abs(x)):.6e}")


def check_tensor_finite(label: str, tensor: torch.Tensor, *, allow_inf_mask: bool = False) -> bool:
    if tensor.dtype.is_floating_point:
        finite = torch.isfinite(tensor)
        if allow_inf_mask:
            ok = bool(finite.all())
        else:
            ok = bool(finite.all())
        if not ok:
            bad = torch.nonzero(~finite)
            print(f"[DEBUG_NAN] {label} has non-finite entries: count={bad.shape[0]} first={bad[:5].tolist()}")
        return ok
    return True


def debug_nan_mode_daily_only() -> None:
    global DEBUG_NAN_MODE, DEBUG_NAN_CONTEXT
    set_seed(SEED)
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cuda" if torch.cuda.is_available() else "cpu")
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    X_all = apply_normalizer(ds, stats, "daily_only")
    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(ds.y[5][train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val = torch.tensor(ds.y[5][val_mask], dtype=torch.float32)

    print("[DEBUG_NAN] start daily_only / 5d diagnostics")
    print(f"[DEBUG_NAN] MPS available = {mps_available}")
    print(f"[DEBUG_NAN] selected device = {device.type.upper()}")
    print(f"[DEBUG_NAN] X_train shape={tuple(X_train.shape)} y_train shape={tuple(y_train.shape)}")
    print(f"[DEBUG_NAN] X_val shape={tuple(X_val.shape)} y_val shape={tuple(y_val.shape)}")
    print_finite_summary("X_train", X_train.detach().cpu().numpy())
    print_finite_summary("y_train", y_train.detach().cpu().numpy())
    if not np.isfinite(X_train.detach().cpu().numpy()).all():
        bad = np.argwhere(~np.isfinite(X_train.detach().cpu().numpy()))
        print("[DEBUG_NAN] X_train has NaN/Inf at indices:", bad[:10].tolist())
        raise RuntimeError("Non-finite normalized Daily training data detected before model execution")
    if not np.isfinite(y_train.detach().cpu().numpy()).all():
        bad = np.argwhere(~np.isfinite(y_train.detach().cpu().numpy()))
        print("[DEBUG_NAN] y_train has NaN/Inf at indices:", bad[:10].tolist())
        raise RuntimeError("Non-finite normalized Daily target data detected before model execution")

    model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            print(f"[DEBUG_NAN] initial parameter non-finite: {name}")
            raise RuntimeError(f"Non-finite initial parameter encountered: {name}")
    print("[DEBUG_NAN] initial model parameters are finite")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    total_train_batches = len(loader)
    print(f"[DEBUG_NAN] total training batches in epoch 1 = {total_train_batches}")

    DEBUG_NAN_MODE = True
    train_loss_sum = 0.0
    train_weight = 0
    completed_train_batches = 0
    train_failed = False
    failure_stage = None
    failure_batch = None

    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        DEBUG_NAN_CONTEXT["frequency"] = "daily_only"
        DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if not torch.isfinite(x_batch).all():
            print(f"[DEBUG_NAN] non-finite input batch: batch_idx={batch_idx}")
            print_finite_summary("x_batch", x_batch.detach().cpu().numpy())
            failure_stage = "input"
            failure_batch = batch_idx
            train_failed = True
            break
        if not torch.isfinite(y_batch).all():
            print(f"[DEBUG_NAN] non-finite y_batch: batch_idx={batch_idx}")
            print_finite_summary("y_batch", y_batch.detach().cpu().numpy())
            failure_stage = "input"
            failure_batch = batch_idx
            train_failed = True
            break

        optimizer.zero_grad()
        pred = model(x_batch)
        if not torch.isfinite(pred).all():
            print(f"[DEBUG_NAN] non-finite prediction at batch {batch_idx}")
            print_finite_summary("pred", pred.detach().cpu().numpy())
            failure_stage = "forward"
            failure_batch = batch_idx
            train_failed = True
            break

        loss = criterion(pred, y_batch)
        if not torch.isfinite(loss):
            print(f"[DEBUG_NAN] non-finite loss at batch {batch_idx}: loss={loss.item()}")
            failure_stage = "loss"
            failure_batch = batch_idx
            train_failed = True
            break

        loss.backward()
        bad_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    bad_grads.append((name, "missing"))
                elif not torch.isfinite(param.grad).all():
                    bad_grads.append((name, "nonfinite"))
        if bad_grads:
            for name, reason in bad_grads:
                print(f"[DEBUG_NAN] gradient issue: {name} -> {reason}")
            failure_stage = "backward"
            failure_batch = batch_idx
            train_failed = True
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[DEBUG_NAN] clip_grad_norm_ produced non-finite norm at batch {batch_idx}: {grad_norm}")
            failure_stage = "gradient clipping"
            failure_batch = batch_idx
            train_failed = True
            break
        optimizer.step()
        for name, param in model.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all():
                print(f"[DEBUG_NAN] parameter became non-finite after optimizer.step(): {name}")
                failure_stage = "optimizer step"
                failure_batch = batch_idx
                train_failed = True
                break
        if train_failed:
            break

        batch_size_value = y_batch.shape[0]
        train_loss_sum += float(loss.item()) * batch_size_value
        train_weight += int(batch_size_value)
        completed_train_batches += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == total_train_batches:
            print(f"[DEBUG_NAN] progress {batch_idx + 1}/{total_train_batches}")

    DEBUG_NAN_MODE = False
    DEBUG_NAN_CONTEXT["frequency"] = None
    DEBUG_NAN_CONTEXT["batch_index"] = None

    if train_failed:
        print(f"[DEBUG_NAN] failure reproduced during training: batch={failure_batch} stage={failure_stage}")
        return

    if completed_train_batches != total_train_batches:
        print(f"[DEBUG_NAN] training loop terminated early: completed={completed_train_batches} total={total_train_batches}")
        return

    debug_train_loss = train_loss_sum / train_weight
    print(f"[DEBUG_NAN] debug_train_loss={debug_train_loss:.12f}")
    if not np.isfinite(debug_train_loss):
        print("[DEBUG_NAN] debug_train_loss is non-finite")
        return

    model.eval()
    val_loss_sum = 0.0
    val_weight = 0
    val_failed = False
    val_failure_stage = None
    val_failure_batch = None
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if not torch.isfinite(x_batch).all():
                print(f"[DEBUG_NAN] non-finite validation input at batch={batch_idx}")
                val_failure_stage = "input"
                val_failure_batch = batch_idx
                val_failed = True
                break
            if not torch.isfinite(y_batch).all():
                print(f"[DEBUG_NAN] non-finite validation target at batch={batch_idx}")
                val_failure_stage = "input"
                val_failure_batch = batch_idx
                val_failed = True
                break

            pred = model(x_batch)
            if not torch.isfinite(pred).all():
                print(f"[DEBUG_NAN] non-finite validation prediction at batch={batch_idx}")
                val_failure_stage = "forward"
                val_failure_batch = batch_idx
                val_failed = True
                break

            loss = criterion(pred, y_batch)
            if not torch.isfinite(loss):
                print(f"[DEBUG_NAN] non-finite validation loss at batch={batch_idx}: loss={loss.item()}")
                val_failure_stage = "loss"
                val_failure_batch = batch_idx
                val_failed = True
                break
            batch_size_value = y_batch.shape[0]
            val_loss_sum += float(loss.item()) * batch_size_value
            val_weight += int(batch_size_value)

        if val_failed:
            print(f"[DEBUG_NAN] validation failure reproduced: batch={val_failure_batch} stage={val_failure_stage}")
            return

    debug_val_loss = val_loss_sum / val_weight if val_weight else float("nan")
    print(f"[DEBUG_NAN] debug_val_loss={debug_val_loss:.12f}")
    if not np.isfinite(debug_val_loss):
        print("[DEBUG_NAN] debug_val_loss is non-finite")
        return

    print("[DEBUG_NAN] FULL FIRST TRAINING EPOCH + VALIDATION BOTH FINITE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-panel SWiM-style Transformer control")
    parser.add_argument("--debug-nan", action="store_true", help="Temporary numerical diagnostic path for the first formal run; no formal outputs are written.")
    parser.add_argument("--trace-formal-nan", action="store_true", help="Run the real formal single-config training path for epoch 1 with read-only numerical tracing and no formal output writes.")
    parser.add_argument("--diagnose-mps-sync", action="store_true", help="Instructional stub for MPS sync diagnosis; use --diagnose-mps-sync-variant A/B/C/D/E to run exactly one variant.")
    parser.add_argument("--diagnose-mps-sync-variant", choices=["A", "B", "C", "D", "E"], help="Run exactly one MPS sync diagnostic variant: A/B/C/D/E. Each variant is fixed to daily_only / 5d and does not write formal outputs.")
    parser.add_argument("--train", action="store_true", help="Run formal training. Requires either --all or both --frequency and --horizon.")
    parser.add_argument("--all", action="store_true", help="Run the full 9-experiment benchmark.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, help="Single-frequency training selection: daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], help="Single-horizon training selection: 5d | 10d | 20d")
    args = parser.parse_args()

    mutually_exclusive = [
        args.debug_nan,
        args.trace_formal_nan,
        args.diagnose_mps_sync,
        args.diagnose_mps_sync_variant is not None,
        args.train,
    ]
    if sum(1 for flag in mutually_exclusive if flag) > 1:
        parser.error("Use only one of --debug-nan, --trace-formal-nan, --diagnose-mps-sync, --diagnose-mps-sync-variant, or --train.")

    if args.trace_formal_nan and (args.frequency is None or args.horizon is None):
        parser.error("--trace-formal-nan requires both --frequency and --horizon.")

    if args.diagnose_mps_sync and args.all:
        parser.error("--diagnose-mps-sync cannot be combined with --all.")
    if args.diagnose_mps_sync_variant is not None and args.all:
        parser.error("--diagnose-mps-sync-variant cannot be combined with --all.")

    if args.train:
        if args.all and (args.frequency is not None or args.horizon is not None):
            parser.error("Use either --all or a single --frequency/--horizon pair, not both.")
        if not args.all and (args.frequency is None or args.horizon is None):
            parser.error("When --train is supplied, either use --all or provide both --frequency and --horizon.")
        if args.frequency is not None and args.horizon is None:
            parser.error("If --frequency is supplied, --horizon must also be supplied.")
        if args.horizon is not None and args.frequency is None:
            parser.error("If --horizon is supplied, --frequency must also be supplied.")

    if args.diagnose_mps_sync:
        print("Use --diagnose-mps-sync-variant A|B|C|D|E to run exactly one MPS sync diagnostic variant. It is fixed to daily_only / 5d and does not write formal outputs.")
    return args


def diagnose_mps_sync_variant(variant: str) -> None:
    if not torch.backends.mps.is_available():
        print("MPS unavailable — sync diagnosis cannot be performed in this environment.")
        return

    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)
    device = torch.device("mps")
    set_seed(SEED)

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    X_all = apply_normalizer(ds, stats, "daily_only")
    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(ds.y[5][train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val = torch.tensor(ds.y[5][val_mask], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = SingleFrequencySWiMRegressor(feature_dim=X_all.shape[-1], frequency_key="daily").to(device)
    if variant == "B":
        torch.mps.synchronize()
    elif variant == "C":
        param_total, param_sum, param_sq_sum = _model_parameter_fingerprint(model)
        print(f"[DIAG] variant={variant} total_parameters={param_total}")
        print(f"[DIAG] variant={variant} param_sum={param_sum:.12f}")
        print(f"[DIAG] variant={variant} param_sq_sum={param_sq_sum:.12f}")
    elif variant == "E":
        print(f"[DIAG] variant={variant} {_trace_rng_fingerprint_summary('torch_rng_state_before_train_loader')}")

    print(f"[DIAG] variant={variant} mps_available={torch.backends.mps.is_available()} selected_device={device.type.upper()} seed={SEED} parameter_count={sum(p.numel() for p in model.parameters())} expected_train_batches={len(train_loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    if variant == "D":
        torch.mps.synchronize()

    train_loss_sum = 0.0
    train_weight = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        batch_size_value = y_batch.shape[0]
        train_loss_sum += float(loss.item()) * batch_size_value
        train_weight += int(batch_size_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    val_loss_sum = 0.0
    val_weight = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            batch_loss = criterion(logits, y_batch)
            batch_size_value = y_batch.shape[0]
            val_loss_sum += float(batch_loss.item()) * batch_size_value
            val_weight += int(batch_size_value)

    train_loss = train_loss_sum / train_weight if train_weight else float("nan")
    val_loss = val_loss_sum / val_weight if val_weight else float("nan")
    train_loss_finite = bool(np.isfinite(train_loss))
    val_loss_finite = bool(np.isfinite(val_loss))
    print(f"[DIAG] variant={variant} completed_train_batches={len(train_loader)} train_loss={train_loss:.12f} val_loss={val_loss:.12f} train_loss_finite={train_loss_finite} val_loss_finite={val_loss_finite}")


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    if args.debug_nan:
        debug_nan_mode_daily_only()
        return
    if args.trace_formal_nan:
        ds = load_frozen_panel()
        assert_frozen_panel_checks(ds)
        stats = fit_train_only_normalizer(ds)
        frequency = args.frequency
        horizon = int(args.horizon[:-1])
        train_and_eval_frequency(ds, frequency, horizon, stats, trace_nan=True)
        return
    if args.diagnose_mps_sync_variant is not None:
        diagnose_mps_sync_variant(args.diagnose_mps_sync_variant)
        return
    if args.diagnose_mps_sync:
        print("Use --diagnose-mps-sync-variant A|B|C|D|E to run exactly one MPS sync diagnostic variant. It is fixed to daily_only / 5d and does not write formal outputs.")
        return
    if args.train:
        ds = load_frozen_panel()
        assert_frozen_panel_checks(ds)
        stats = fit_train_only_normalizer(ds)
        if args.all:
            full_benchmark(ds, stats)
        else:
            clear_swim_partial_outputs()
            frequency = args.frequency
            horizon = int(args.horizon[:-1])
            result = train_and_eval_frequency(ds, frequency, horizon, stats)
            summary_df = pd.DataFrame(
                [
                    {
                        "model": result["model"],
                        "frequency_setting": result["frequency_setting"],
                        "horizon": result["horizon"],
                        "seed": result["seed"],
                        "n_trainable_parameters": result["n_trainable_parameters"],
                        "test_mse": result["test_mse"],
                        "test_mae": result["test_mae"],
                        "test_corr": result["test_corr"],
                        "mean_rank_ic": result["mean_rank_ic"],
                        "direction_accuracy": result["direction_accuracy"],
                        "pred_std_true_std_ratio": result["pred_std_true_std_ratio"],
                    }
                ]
            )
            save_summary_rows([result], append_mode=False)
            save_prediction_rows([result], append_mode=False)
            save_training_history([result], append_mode=False)
            save_rank_ic_by_date([result], append_mode=False)
            comparison_df = build_baseline_comparison(summary_df)
            if not comparison_df.empty:
                comparison_df.to_csv(AUDIT_DIR / "panel_swim_vs_baselines_summary.csv", index=False)
            write_final_summary(summary_df, comparison_df)
            print(f"Single-config run complete: {frequency} / {args.horizon}")
    else:
        run_smoke_validation()


if __name__ == "__main__":
    main()
