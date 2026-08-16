"""
Experiment A (FSLR Full-Frequency Adaptive Multi-Scale PathFormer) - Shared Core

This module holds everything shared across the A1-A7 scripts in this folder:

- FSLR full multi-scale data loading (hourly/halfday/daily/weekly, from
  dataset/multiscale_dataset/, built by scripts/python/build_multiscale_dataset.py).
- Regression metrics (MSE, MAE, Corr, Rank IC, DA, Pred Std / True Std).
- Three AMS ("router") variants built on the SAME patch-expert machinery from
  pathformer.layers.AMS.AMS, used to implement the A1 -> A4 progression:

    combine_mode      AMS class          gating behaviour
    ----------------  -----------------  --------------------------------------
    single            AMS (1 expert)     trivial (only one patch size to pick)
    uniform            UniformGateAMS    equal 1/n weight on every patch expert,
                                          not learned, not sample-adaptive
    static_weight      StaticWeightAMS   one learned softmax weight vector,
                                          shared by every sample (not adaptive)
    adaptive_router    AMS (as-is)       per-sample noisy top-k gate, function
                                          of the input window (real router)

  This mirrors the advisor's requested logic exactly: A1 (single patch) -> A2
  (fixed/uniform multi-scale) -> A3 (static learnable weight) -> A4 (adaptive
  router), all using literally the same expert Transformer_Layer modules so
  the only thing that changes between stages is how they are combined.

- PatchScaleModel: a parameterized re-implementation of
  pathformer.models.PathFormer.Model that accepts an `ams_cls` (so the four
  combine_modes above can be swapped in) and the use_intra/use_inter switches
  (for Experiment A6), used as a per-frequency branch encoder.
- CrossFreqRouter: an AMS-style noisy top-k gate across FREQUENCY branches
  (not patch experts), used by Experiment A5 for the "adaptive" fusion option
  and reused conceptually from scripts/python/panel_adaptive_scale_experiment.py.
- A generic train_one_run() training loop shared by A1/A2/A3/A4/A5/A6.
"""

from __future__ import annotations

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

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pathformer.layers.AMS import AMS
from pathformer.layers.RevIN import RevIN

EXP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_DIR / "output"

DATASET_DIR = BASE_DIR / "dataset" / "multiscale_dataset"

FREQ_NAMES = ["hourly", "halfday", "daily", "weekly"]
FEATURES = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURES.index("close")

# Lookback windows, must match config.py / build_multiscale_dataset.py.
FREQ_WINDOW = {"hourly": 24, "halfday": 20, "daily": 90, "weekly": 26}

# Advisor-specified patch-size candidates per frequency (Experiment A1).
PATCH_CANDIDATES = {
    "hourly": [2, 4, 8, 12],
    "halfday": [2, 4, 8, 10, 12],
    "daily": [5, 10, 20, 30],
    "weekly": [2, 4, 8, 12, 13, 14],
}

HORIZON_NAMES = ["5d", "10d", "20d"]

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 0.0

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# GENERIC HELPERS
# =====================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return OUTPUT_DIR / p


@dataclass
class Metrics:
    mse: float
    mae: float
    corr: float
    rank_corr: float
    direction_acc: float
    pred_std: float
    true_std: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan")
    if len(y_true) > 1 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        rank_corr = float(spearmanr(y_pred, y_true)[0])
    else:
        rank_corr = float("nan")
    direction_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return Metrics(
        mse=mse,
        mae=mae,
        corr=corr,
        rank_corr=rank_corr,
        direction_acc=direction_acc,
        pred_std=float(np.std(y_pred)),
        true_std=float(np.std(y_true)),
    )


def metrics_to_dict(prefix: str, m: Metrics) -> dict:
    return {
        f"{prefix}_mse": m.mse,
        f"{prefix}_mae": m.mae,
        f"{prefix}_corr": m.corr,
        f"{prefix}_rank_corr": m.rank_corr,
        f"{prefix}_direction_acc": m.direction_acc,
        f"{prefix}_pred_std": m.pred_std,
        f"{prefix}_true_std": m.true_std,
    }


# =====================================================================
# DATA LOADING (FSLR full multi-scale, single ticker, case study)
# =====================================================================


def load_fslr_multiscale(horizon_key: str) -> dict:
    """Loads the FSLR multi-scale arrays built by
    scripts/python/build_multiscale_dataset.py. horizon_key in {"5d","10d","20d"}.
    """
    y_name = f"y_{horizon_key}"
    data = {
        "hourly": np.load(DATASET_DIR / "X_hourly.npy").astype(np.float32),
        "halfday": np.load(DATASET_DIR / "X_halfday.npy").astype(np.float32),
        "daily": np.load(DATASET_DIR / "X_daily.npy").astype(np.float32),
        "weekly": np.load(DATASET_DIR / "X_weekly.npy").astype(np.float32),
        "y": np.load(DATASET_DIR / f"{y_name}.npy").astype(np.float32),
    }
    meta = pd.read_csv(DATASET_DIR / "meta.csv", parse_dates=["anchor_date"])
    n = min(len(data["hourly"]), len(data["halfday"]), len(data["daily"]), len(data["weekly"]), len(data["y"]), len(meta))
    for k in ["hourly", "halfday", "daily", "weekly", "y"]:
        data[k] = data[k][:n]
    data["meta"] = meta.iloc[:n].reset_index(drop=True)
    return data


def split_indices(n: int) -> tuple[slice, slice, slice]:
    train_end = int(n * SPLIT_TRAIN)
    val_end = int(n * (SPLIT_TRAIN + SPLIT_VAL))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def compute_patch_nums(seq_len: int, patch_size_list: list[int]) -> None:
    """Sanity check only; AMS itself computes patch_nums = int(seq_len/patch),
    which tolerates non-exact divisors via floor division (unfold/slicing both
    use the same truncated count internally, so this is safe, just wasteful
    for patch sizes that don't evenly divide the window)."""
    for p in patch_size_list:
        if seq_len // p < 1:
            raise ValueError(f"patch_size={p} too large for seq_len={seq_len}")


# =====================================================================
# AMS VARIANTS (A1 -> A4 progression, same expert machinery)
# =====================================================================


class UniformGateAMS(AMS):
    """'Fixed multi-scale' (Experiment A2): every patch-size expert gets an
    equal 1/num_experts weight, not learned, not sample-adaptive."""

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        batch = x.shape[0]
        gates = torch.full((batch, self.num_experts), 1.0 / self.num_experts, device=x.device, dtype=torch.float32)
        load = self._gates_to_load(gates)
        return gates, load


class StaticWeightAMS(AMS):
    """'Static learnable scale weight' (Experiment A3): one global softmax
    weight vector over patch-size experts, learned but identical for every
    sample (ignores the current window's content)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_logits = nn.Parameter(torch.zeros(self.num_experts))

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        batch = x.shape[0]
        weights = torch.softmax(self.static_logits, dim=0)
        gates = weights.unsqueeze(0).expand(batch, -1).contiguous()
        load = self._gates_to_load(gates)
        return gates, load


COMBINE_MODE_TO_AMS_CLS = {
    "single": AMS,
    "uniform": UniformGateAMS,
    "static_weight": StaticWeightAMS,
    "adaptive_router": AMS,
}


# =====================================================================
# PER-FREQUENCY BRANCH MODEL (parameterized copy of PathFormer.Model)
# =====================================================================


class PatchScaleModel(nn.Module):
    """Re-implementation of pathformer.models.PathFormer.Model, parameterized
    so the AMS class (regular / uniform-gate / static-weight) and the
    use_intra/use_inter switches can be swapped per experiment. Numerically
    identical to the original Model when ams_cls=AMS, use_intra=use_inter=True.
    """

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        pred_len: int,
        patch_size_list: list[int],
        num_experts: int,
        k: int,
        d_model: int,
        d_ff: int,
        layer_nums: int = 1,
        residual_connection: int = 1,
        revin: bool = True,
        batch_norm: bool = False,
        ams_cls: type = AMS,
        use_intra: bool = True,
        use_inter: bool = True,
    ):
        super().__init__()
        self.layer_nums = layer_nums
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.d_model = d_model
        self.residual_connection = residual_connection
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_features=num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=d_model)
        device = DEVICE
        self.AMS_lists = nn.ModuleList()
        for num in range(layer_nums):
            self.AMS_lists.append(
                ams_cls(
                    seq_len,
                    seq_len,
                    num_experts,
                    device,
                    k=k,
                    num_nodes=num_nodes,
                    patch_size=patch_size_list,
                    noisy_gating=True,
                    d_model=d_model,
                    d_ff=d_ff,
                    layer_number=num + 1,
                    residual_connection=residual_connection,
                    batch_norm=batch_norm,
                    use_intra=use_intra,
                    use_inter=use_inter,
                )
            )
        self.projections = nn.Sequential(nn.Linear(seq_len * d_model, pred_len))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        balance_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.revin:
            x = self.revin_layer(x, "norm")
        out = self.start_fc(x.unsqueeze(-1))
        batch_size = x.shape[0]

        last_gates = None
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss
            last_gates = getattr(layer, "last_gates", None)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)
        if self.revin:
            out = self.revin_layer(out, "denorm")
        return out, balance_loss, last_gates


class FrequencyBranchEncoder(nn.Module):
    """Wraps PatchScaleModel as a fixed-size vector encoder for one frequency,
    following the same convention used in scripts/python/task8_multiscale_operator_ablation.py:
    take the last forecast step (pred_len=1) over the raw feature_dim, then
    project to d_branch."""

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        patch_size_list: list[int],
        combine_mode: str,
        d_model: int,
        d_ff: int,
        d_branch: int,
        layer_nums: int = 1,
        router_k: int | None = None,
        use_intra: bool = True,
        use_inter: bool = True,
    ):
        super().__init__()
        ams_cls = COMBINE_MODE_TO_AMS_CLS[combine_mode]
        num_experts = len(patch_size_list)
        if combine_mode == "single":
            if num_experts != 1:
                raise ValueError("combine_mode='single' requires exactly one patch size")
            k = 1
        else:
            k = num_experts if router_k is None else min(router_k, num_experts)

        self.backbone = PatchScaleModel(
            seq_len=seq_len,
            num_nodes=feature_dim,
            pred_len=1,
            patch_size_list=patch_size_list,
            num_experts=num_experts,
            k=k,
            d_model=d_model,
            d_ff=d_ff,
            layer_nums=layer_nums,
            residual_connection=1,
            revin=True,
            batch_norm=False,
            ams_cls=ams_cls,
            use_intra=use_intra,
            use_inter=use_inter,
        )
        self.proj = nn.Linear(feature_dim, d_branch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        out, aux_loss, gates = self.backbone(x)
        vec = self.proj(out[:, -1, :])
        return vec, aux_loss, gates


# =====================================================================
# CROSS-FREQUENCY ROUTER (Experiment A5, "adaptive" fusion option)
# =====================================================================


class CrossFreqRouter(nn.Module):
    """AMS-style noisy top-k gating across FREQUENCY branches (not patch
    experts). Same construction as scripts/python/panel_adaptive_scale_experiment.py's
    ScaleRouter, generalized to an arbitrary number of active frequencies."""

    def __init__(self, num_freqs: int, k: int | None = None, noisy_gating: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.k = num_freqs if k is None else min(k, num_freqs)
        self.w_gate = nn.Linear(2 * num_freqs, num_freqs)
        self.w_noise = nn.Linear(2 * num_freqs, num_freqs)
        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()

    def forward(self, gate_input: torch.Tensor, training: bool) -> torch.Tensor:
        clean_logits = self.w_gate(gate_input)
        if self.noisy_gating and training:
            noise_stddev = self.softplus(self.w_noise(gate_input)) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        if self.k < self.num_freqs:
            top_vals, top_idx = logits.topk(self.k, dim=1)
            top_gates = torch.softmax(top_vals, dim=1)
            gates = torch.zeros_like(logits).scatter(1, top_idx, top_gates)
        else:
            gates = torch.softmax(logits, dim=1)
        return gates


def make_cross_freq_router_input(x_by_freq: dict[str, torch.Tensor], freq_order: list[str]) -> torch.Tensor:
    """Per-frequency [mean_logret, std_logret] summary, concatenated in freq_order."""

    def summarize(x: torch.Tensor) -> torch.Tensor:
        close = x[:, :, CLOSE_IDX].clamp_min(1e-6)
        log_ret = torch.diff(torch.log(close), dim=1)
        mean_r = log_ret.mean(dim=1, keepdim=True)
        std_r = log_ret.std(dim=1, keepdim=True)
        return torch.cat([mean_r, std_r], dim=1)

    return torch.cat([summarize(x_by_freq[f]) for f in freq_order], dim=1)


# =====================================================================
# REGRESSION HEADS (single-frequency for A1-A4, multi-frequency for A5-A6)
# =====================================================================


class SingleFrequencyRegressor(nn.Module):
    """One frequency branch + linear head -> scalar prediction. Used by A1-A4,
    where only one frequency is under test at a time."""

    def __init__(
        self,
        freq_name: str,
        seq_len: int,
        feature_dim: int,
        patch_size_list: list[int],
        combine_mode: str,
        d_model: int = 32,
        d_ff: int = 64,
        d_branch: int = 32,
        router_k: int | None = None,
        use_intra: bool = True,
        use_inter: bool = True,
    ):
        super().__init__()
        self.freq_name = freq_name
        self.encoder = FrequencyBranchEncoder(
            seq_len=seq_len,
            feature_dim=feature_dim,
            patch_size_list=patch_size_list,
            combine_mode=combine_mode,
            d_model=d_model,
            d_ff=d_ff,
            d_branch=d_branch,
            router_k=router_k,
            use_intra=use_intra,
            use_inter=use_inter,
        )
        self.head = nn.Linear(d_branch, 1)

    def forward(self, x_dict: dict[str, torch.Tensor]):
        vec, aux_loss, gates = self.encoder(x_dict[self.freq_name])
        pred = self.head(vec).squeeze(-1)
        gates_dict = {"patch": gates} if gates is not None else None
        return pred, aux_loss, gates_dict


class MultiFrequencyRegressor(nn.Module):
    """Multiple frequency branches fused into one scalar prediction. Used by
    A5 (frequency-count ablation) and A6 (dual-attention ablation).

    fusion="concat": concatenate branch vectors, single linear head.
    fusion="adaptive_router": CrossFreqRouter weights branches by market
      dynamics (per-frequency mean/std log-return), weighted-sum, then head.
    """

    def __init__(
        self,
        freq_names: list[str],
        seq_len_map: dict[str, int],
        feature_dim: int,
        patch_size_map: dict[str, list[int]],
        combine_mode_map: dict[str, str],
        d_model: int = 32,
        d_ff: int = 64,
        d_branch: int = 32,
        fusion: str = "concat",
        router_k_map: dict[str, int] | None = None,
        use_intra: bool = True,
        use_inter: bool = True,
        cross_router_k: int | None = None,
    ):
        super().__init__()
        self.freq_names = freq_names
        self.fusion = fusion
        self.d_branch = d_branch
        router_k_map = router_k_map or {}
        self.encoders = nn.ModuleDict(
            {
                f: FrequencyBranchEncoder(
                    seq_len=seq_len_map[f],
                    feature_dim=feature_dim,
                    patch_size_list=patch_size_map[f],
                    combine_mode=combine_mode_map[f],
                    d_model=d_model,
                    d_ff=d_ff,
                    d_branch=d_branch,
                    router_k=router_k_map.get(f),
                    use_intra=use_intra,
                    use_inter=use_inter,
                )
                for f in freq_names
            }
        )
        if fusion == "concat":
            self.head = nn.Linear(d_branch * len(freq_names), 1)
        elif fusion == "adaptive_router":
            self.cross_router = CrossFreqRouter(len(freq_names), k=cross_router_k)
            self.head = nn.Linear(d_branch, 1)
        else:
            raise ValueError(f"unknown fusion={fusion}")

    def forward(self, x_dict: dict[str, torch.Tensor]):
        vecs = []
        aux_total = None
        gates_dict: dict[str, torch.Tensor] = {}
        for f in self.freq_names:
            vec, aux_loss, patch_gates = self.encoders[f](x_dict[f])
            vecs.append(vec)
            aux_total = aux_loss if aux_total is None else aux_total + aux_loss
            if patch_gates is not None:
                gates_dict[f"patch_{f}"] = patch_gates

        if self.fusion == "concat":
            fused = torch.cat(vecs, dim=1)
            pred = self.head(fused).squeeze(-1)
        else:
            router_input = make_cross_freq_router_input(x_dict, self.freq_names)
            freq_gates = self.cross_router(router_input, self.training)
            gates_dict["freq"] = freq_gates
            stacked = torch.stack(vecs, dim=1)  # (B, num_freqs, d_branch)
            fused = (stacked * freq_gates.unsqueeze(-1)).sum(dim=1)
            pred = self.head(fused).squeeze(-1)

        return pred, aux_total, (gates_dict if gates_dict else None)





@dataclass
class TrainArtifacts:
    metrics: dict
    epochs_ran: int
    elapsed_sec: float
    test_gates: dict[str, np.ndarray] | None  # name -> (n_test, n_scales) gate array


def train_one_run(
    build_model_fn,
    freq_inputs: list[str],
    data: dict,
    *,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    y_standardize: bool = True,
    loss_type: str = "huber",
    huber_delta: float = 1.35,
    aux_loss_weight: float = 1.0,
    run_name: str = "run",
    progress_file: Path | None = None,
    progress_every: int = 1,
) -> TrainArtifacts:
    """Generic train/eval loop shared by A1-A6.

    build_model_fn() -> nn.Module whose forward(x_by_freq: dict[str, Tensor])
    returns (pred: Tensor[B], aux_loss: Tensor[scalar], gates: dict[str, Tensor]|None)
    where gates maps a gate-name (e.g. "patch" or "freq") to a (B, n_scales) tensor.
    """
    y = data["y"]
    n_total = len(y)
    s_train, s_val, s_test = split_indices(n_total)

    X_by_freq = {f: data[f] for f in freq_inputs}

    def subset(mask: slice) -> dict[str, torch.Tensor]:
        return {f: torch.from_numpy(X_by_freq[f][mask]) for f in freq_inputs}

    X_train, X_val, X_test = subset(s_train), subset(s_val), subset(s_test)
    y_train_np, y_val_np, y_test_np = y[s_train], y[s_val], y[s_test]

    y_mean = float(np.mean(y_train_np))
    y_std = float(np.std(y_train_np))
    if y_std < 1e-8:
        y_std = 1.0

    if y_standardize:
        y_train_use = (y_train_np - y_mean) / y_std
        y_val_use = (y_val_np - y_mean) / y_std
        y_test_use = (y_test_np - y_mean) / y_std
    else:
        y_train_use, y_val_use, y_test_use = y_train_np, y_val_np, y_test_np

    def make_loader(x_dict: dict[str, torch.Tensor], y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = [x_dict[f] for f in freq_inputs] + [torch.from_numpy(y_arr)]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train_use, shuffle=True)
    train_eval_loader = make_loader(X_train, y_train_use, shuffle=False)
    val_loader = make_loader(X_val, y_val_use, shuffle=False)
    test_loader = make_loader(X_test, y_test_use, shuffle=False)

    model = build_model_fn().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=huber_delta) if loss_type == "huber" else nn.MSELoss()

    def batch_to_dict(batch: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        *xs, yb = batch
        x_dict = {f: xs[i].to(DEVICE) for i, f in enumerate(freq_inputs)}
        return x_dict, yb.to(DEVICE)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    run_start = time.perf_counter()
    running_epoch_secs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            x_dict, yb = batch_to_dict(batch)
            optimizer.zero_grad()
            pred, aux_loss, _ = model(x_dict)
            loss = criterion(pred, yb) + aux_loss_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_dict, yb = batch_to_dict(batch)
                pred, _, _ = model(x_dict)
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
        eta_sec = max(0, max_epochs - epoch) * mean_epoch_sec

        if (epoch % max(1, progress_every) == 0) or (epoch == 1) or (epoch == max_epochs) or (bad_epochs >= patience):
            log_progress(
                f"[{run_name}] epoch {epoch:03d}/{max_epochs:03d} val_loss={mean_val:.8f} "
                f"best_val_loss={best_val:.8f} bad_epochs={bad_epochs}/{patience} "
                f"epoch_time={format_seconds(epoch_sec)} eta={format_seconds(eta_sec)}",
                progress_file,
            )

        if bad_epochs >= patience:
            log_progress(f"[{run_name}] early stopping at epoch {epoch}", progress_file)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
        preds, trues = [], []
        gates_accum: dict[str, list[np.ndarray]] | None = None
        with torch.no_grad():
            for batch in loader:
                x_dict, yb = batch_to_dict(batch)
                pred, _, gates = model(x_dict)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
                if gates is not None:
                    if gates_accum is None:
                        gates_accum = {name: [] for name in gates}
                    for name, g in gates.items():
                        gates_accum[name].append(g.cpu().numpy())
        gates_out = None
        if gates_accum is not None:
            gates_out = {name: np.concatenate(arrs, axis=0) for name, arrs in gates_accum.items()}
        return np.concatenate(trues), np.concatenate(preds), gates_out

    ytr_true_use, ytr_pred_use, _ = collect(train_eval_loader)
    yv_true_use, yv_pred_use, _ = collect(val_loader)
    yt_true_use, yt_pred_use, yt_gates = collect(test_loader)

    if y_standardize:
        ytr_true, ytr_pred = ytr_true_use * y_std + y_mean, ytr_pred_use * y_std + y_mean
        yv_true, yv_pred = yv_true_use * y_std + y_mean, yv_pred_use * y_std + y_mean
        yt_true, yt_pred = yt_true_use * y_std + y_mean, yt_pred_use * y_std + y_mean
    else:
        ytr_true, ytr_pred = ytr_true_use, ytr_pred_use
        yv_true, yv_pred = yv_true_use, yv_pred_use
        yt_true, yt_pred = yt_true_use, yt_pred_use

    train_m = regression_metrics(ytr_true, ytr_pred)
    val_m = regression_metrics(yv_true, yv_pred)
    test_m = regression_metrics(yt_true, yt_pred)

    elapsed_sec = time.perf_counter() - run_start
    metrics = {
        "train_samples": int(len(y_train_np)),
        "val_samples": int(len(y_val_np)),
        "test_samples": int(len(y_test_np)),
        **metrics_to_dict("train", train_m),
        **metrics_to_dict("val", val_m),
        **metrics_to_dict("test", test_m),
        "y_train_mean": y_mean,
        "y_train_std": y_std,
    }

    if yt_gates is not None:
        for name, g in yt_gates.items():
            for i in range(g.shape[1]):
                metrics[f"test_mean_gate_{name}_{i}"] = float(g[:, i].mean())

    return TrainArtifacts(
        metrics=metrics,
        epochs_ran=len(running_epoch_secs),
        elapsed_sec=elapsed_sec,
        test_gates=yt_gates,
    )


def save_results_csv(rows: list[dict], out_csv: Path, key_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        merge_key = df[key_cols].apply(tuple, axis=1)
        old_key = old[key_cols].apply(tuple, axis=1)
        keep_old = old[~old_key.isin(merge_key)].copy()
        df = pd.concat([keep_old, df], ignore_index=True)
    df = df.sort_values(key_cols)
    df.to_csv(out_csv, index=False)
    return df
