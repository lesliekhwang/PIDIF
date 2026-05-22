"""
Model and training utilities for Fluent DeepONet experiments.

Dataset construction lives in deeponet_fluent_dataset.py. This module re-exports
those dataset helpers for convenience, but keeps only model, normalization,
training, evaluation, and prediction code here.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeponet_fluent_dataset import (
    CELL_BRANCH_CHANNELS,
    normalize_cell_branch_with_y,
)


class FeatureNormalizer:
    """
    Channel-wise normalizer for tensors whose last dimension is feature/channel.

    It works for target tensors shaped (..., C), branch tensors (..., C), or
    point targets (B, Q, C). skip_indices can be used to leave masks and
    normalized coordinates unchanged.
    """

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        eps: float = 1.0e-6,
        skip_indices: Optional[Iterable[int]] = None,
    ):
        xt = torch.as_tensor(x, dtype=torch.float32)
        if xt.ndim < 2:
            raise ValueError(f"Expected at least 2 dimensions, got shape {tuple(xt.shape)}")

        c = int(xt.shape[-1])
        flat = xt.reshape(-1, c)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(float(eps))

        if skip_indices is not None:
            for i in skip_indices:
                mean[int(i)] = 0.0
                std[int(i)] = 1.0

        self.mean = mean
        self.std = std
        self.eps = float(eps)

    def _shape(self, x: torch.Tensor) -> Tuple[int, ...]:
        return tuple([1] * (x.ndim - 1) + [self.mean.numel()])

    def encode(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
        return (xt - self.mean.reshape(self._shape(xt))) / self.std.reshape(self._shape(xt))

    def decode(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
        return xt * self.std.reshape(self._shape(xt)) + self.mean.reshape(self._shape(xt))

    def to(self, device: Union[str, torch.device]):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": torch.tensor(float(self.eps)),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]):
        obj = cls.__new__(cls)
        obj.mean = torch.as_tensor(state["mean"], dtype=torch.float32)
        obj.std = torch.as_tensor(state["std"], dtype=torch.float32)
        eps = state.get("eps", torch.tensor(1.0e-6))
        obj.eps = float(eps.item() if torch.is_tensor(eps) else eps)
        return obj


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: str = "gelu",
    layer_norm: bool = False,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    if activation.lower() == "relu":
        act: nn.Module = nn.ReLU()
    elif activation.lower() == "tanh":
        act = nn.Tanh()
    elif activation.lower() == "silu":
        act = nn.SiLU()
    else:
        act = nn.GELU()

    layers = []
    in_dim = int(input_dim)
    for _ in range(depth - 1):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(act)
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


class PointSetBranchNet(nn.Module):
    """Permutation-invariant branch net for boundary/interface point sets."""

    def __init__(
        self,
        branch_input_dim: int,
        latent_dim: int = 128,
        output_channels: int = 4,
        point_hidden_dim: int = 128,
        point_depth: int = 3,
        global_hidden_dim: int = 128,
        global_depth: int = 3,
        aggregation: str = "mean",
        activation: str = "gelu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.branch_input_dim = int(branch_input_dim)
        self.latent_dim = int(latent_dim)
        self.output_channels = int(output_channels)
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"mean", "sum", "max"}:
            raise ValueError("aggregation must be 'mean', 'sum', or 'max'")

        self.point_encoder = _make_mlp(
            input_dim=self.branch_input_dim,
            output_dim=int(point_hidden_dim),
            hidden_dim=int(point_hidden_dim),
            depth=int(point_depth),
            activation=activation,
            layer_norm=layer_norm,
        )
        self.global_mlp = _make_mlp(
            input_dim=int(point_hidden_dim),
            output_dim=self.output_channels * self.latent_dim,
            hidden_dim=int(global_hidden_dim),
            depth=int(global_depth),
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, branch: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 3:
            raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")
        h = self.point_encoder(branch)
        if self.aggregation == "mean":
            h = h.mean(dim=1)
        elif self.aggregation == "sum":
            h = h.sum(dim=1)
        else:
            h = h.max(dim=1).values
        coeff = self.global_mlp(h)
        return coeff.reshape(branch.shape[0], self.output_channels, self.latent_dim)


class TrunkNet(nn.Module):
    """Trunk net for query coordinates."""

    def __init__(
        self,
        trunk_input_dim: int = 2,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        activation: str = "gelu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.trunk_input_dim = int(trunk_input_dim)
        self.latent_dim = int(latent_dim)
        self.net = _make_mlp(
            input_dim=self.trunk_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError(f"Expected query shape (B,Q,C), got {tuple(query.shape)}")
        return self.net(query)


class DeepONet(nn.Module):
    """
    Point-set DeepONet.

    branch: (B, M, Cb)
    query:  (B, Q, Ct) or (Q, Ct)
    output: (B, Q, Cout)
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        output_channels: int = 4,
        latent_dim: int = 128,
        branch_point_hidden_dim: int = 128,
        branch_point_depth: int = 3,
        branch_global_hidden_dim: int = 128,
        branch_global_depth: int = 3,
        trunk_hidden_dim: int = 128,
        trunk_depth: int = 4,
        aggregation: str = "mean",
        activation: str = "gelu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.branch_input_dim = int(branch_input_dim)
        self.trunk_input_dim = int(trunk_input_dim)
        self.output_channels = int(output_channels)
        self.latent_dim = int(latent_dim)
        self.branch_point_hidden_dim = int(branch_point_hidden_dim)
        self.branch_point_depth = int(branch_point_depth)
        self.branch_global_hidden_dim = int(branch_global_hidden_dim)
        self.branch_global_depth = int(branch_global_depth)
        self.trunk_hidden_dim = int(trunk_hidden_dim)
        self.trunk_depth = int(trunk_depth)
        self.aggregation = str(aggregation)
        self.activation = str(activation)
        self.layer_norm = bool(layer_norm)

        self.branch_net = PointSetBranchNet(
            branch_input_dim=self.branch_input_dim,
            latent_dim=self.latent_dim,
            output_channels=self.output_channels,
            point_hidden_dim=int(branch_point_hidden_dim),
            point_depth=int(branch_point_depth),
            global_hidden_dim=int(branch_global_hidden_dim),
            global_depth=int(branch_global_depth),
            aggregation=aggregation,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.trunk_net = TrunkNet(
            trunk_input_dim=self.trunk_input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            activation=activation,
            layer_norm=layer_norm,
        )
        self.bias = nn.Parameter(torch.zeros(self.output_channels))

    def forward(self, branch: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 3:
            raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")
        if query.ndim == 2:
            query = query.unsqueeze(0).expand(branch.shape[0], -1, -1)
        if query.ndim != 3:
            raise ValueError(f"Expected query shape (B,Q,C) or (Q,C), got {tuple(query.shape)}")
        if query.shape[0] != branch.shape[0]:
            if query.shape[0] == 1:
                query = query.expand(branch.shape[0], -1, -1)
            else:
                raise ValueError("branch and query batch dimensions do not match")

        coeff = self.branch_net(branch)      # (B, Cout, R)
        basis = self.trunk_net(query)        # (B, Q, R)
        out = torch.einsum("bcr,bqr->bqc", coeff, basis)
        return out + self.bias.reshape(1, 1, -1)

    def config(self) -> Dict[str, object]:
        return {
            "branch_input_dim": self.branch_input_dim,
            "trunk_input_dim": self.trunk_input_dim,
            "output_channels": self.output_channels,
            "latent_dim": self.latent_dim,
            "branch_point_hidden_dim": self.branch_point_hidden_dim,
            "branch_point_depth": self.branch_point_depth,
            "branch_global_hidden_dim": self.branch_global_hidden_dim,
            "branch_global_depth": self.branch_global_depth,
            "trunk_hidden_dim": self.trunk_hidden_dim,
            "trunk_depth": self.trunk_depth,
            "aggregation": self.aggregation,
            "activation": self.activation,
            "layer_norm": self.layer_norm,
        }


class RelativeL2Loss(nn.Module):
    def __init__(self, eps: float = 1.0e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.reshape(pred.shape[0], -1)
        target_f = target.reshape(target.shape[0], -1)
        num = torch.linalg.norm(pred_f - target_f, dim=1)
        den = torch.linalg.norm(target_f, dim=1).clamp_min(self.eps)
        return (num / den).mean()


def boundary_loss(
    model: nn.Module,
    branch: torch.Tensor,
    branch_channel_names: Sequence[str],
    known_masked: bool = True,
) -> torch.Tensor:
    """
    Optional boundary loss for the non-grid branch representation.

    This assumes branch physical value channels are already in the same scale as
    the model output target. With DeepONetCellDataset, that means p/T/u/v branch
    values and target values are both y-normalized.

    If known_pressure/known_temperature/known_u/known_v are present, only known
    boundary quantities contribute to the loss. This avoids forcing unknown inlet
    pressure, outlet velocity, etc. to arbitrary fill values.
    """
    names = list(branch_channel_names)
    x_ch = names.index("x_local")
    y_ch = names.index("y_local")
    value_ch = [
        names.index("boundary_pressure"),
        names.index("boundary_temperature"),
        names.index("boundary_u"),
        names.index("boundary_v"),
    ]

    query_bc = branch[..., [x_ch, y_ch]]
    target_bc = branch[..., value_ch]
    pred_bc = model(branch, query_bc)

    if known_masked and all(k in names for k in ["known_pressure", "known_temperature", "known_u", "known_v"]):
        known_ch = [
            names.index("known_pressure"),
            names.index("known_temperature"),
            names.index("known_u"),
            names.index("known_v"),
        ]
        known = branch[..., known_ch]
        denom = known.sum().clamp_min(1.0)
        return torch.sum(((pred_bc - target_bc) ** 2) * known) / denom

    return F.mse_loss(pred_bc, target_bc)


def train_deeponet_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    loss_type: str = "mse",
    lambda_bc: float = 0.0,
    branch_channel_names: Optional[Sequence[str]] = None,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    device = torch.device(device)
    rel_l2 = RelativeL2Loss()
    total = 0.0
    count = 0

    for branch, query, target, _sample_idx in loader:
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(branch, query)

        if loss_type.lower() == "relative_l2":
            field_loss = rel_l2(pred, target)
        else:
            field_loss = F.mse_loss(pred, target)

        loss = field_loss
        if lambda_bc > 0.0:
            if branch_channel_names is None:
                raise ValueError("branch_channel_names is required when lambda_bc > 0")
            loss = loss + float(lambda_bc) * boundary_loss(
                model=model,
                branch=branch,
                branch_channel_names=branch_channel_names,
            )

        loss.backward()
        optimizer.step()

        bs = int(branch.shape[0])
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


@torch.no_grad()
def evaluate_deeponet(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    y_normalizer: Optional[FeatureNormalizer] = None,
) -> Dict[str, object]:
    """Evaluate DeepONet. If y_normalizer is provided, metrics are physical-unit metrics."""
    model.eval()
    device = torch.device(device)
    mse_sum = 0.0
    rel_sum = 0.0
    count = 0
    channel_sse: Optional[torch.Tensor] = None
    channel_energy: Optional[torch.Tensor] = None
    rel_l2 = RelativeL2Loss()

    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    for branch, query, target, _sample_idx in loader:
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)
        pred = model(branch, query)

        pred_metric = pred
        target_metric = target
        if y_normalizer is not None:
            pred_metric = y_normalizer.decode(pred)
            target_metric = y_normalizer.decode(target)

        bs = int(branch.shape[0])
        mse = F.mse_loss(pred_metric, target_metric)
        rel = rel_l2(pred_metric, target_metric)
        mse_sum += float(mse.item()) * bs
        rel_sum += float(rel.item()) * bs
        count += bs

        diff = pred_metric - target_metric
        sse = torch.sum(diff ** 2, dim=(0, 1))
        energy = torch.sum(target_metric ** 2, dim=(0, 1))
        if channel_sse is None:
            channel_sse = sse.detach()
            channel_energy = energy.detach()
        else:
            channel_sse += sse.detach()
            channel_energy += energy.detach()

    if channel_sse is None or channel_energy is None:
        channel_rel = np.array([], dtype=np.float32)
    else:
        channel_rel = torch.sqrt(channel_sse / channel_energy.clamp_min(1.0e-12)).cpu().numpy()

    return {
        "mse": mse_sum / max(count, 1),
        "relative_l2": rel_sum / max(count, 1),
        "channel_relative_l2": channel_rel,
    }


@torch.no_grad()
def predict_deeponet_points(
    model: nn.Module,
    branch: Union[np.ndarray, torch.Tensor],
    query: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device],
    y_normalizer: Optional[FeatureNormalizer] = None,
    query_batch_size: int = 65536,
) -> torch.Tensor:
    """
    Predict one sample at arbitrary query points.

    branch must already be in the same preprocessing state used during training.
    For the cell-center dataset, use normalize_cell_branch_with_y before calling
    this function. The returned tensor is decoded to physical units when
    y_normalizer is supplied.
    """
    model.eval()
    device = torch.device(device)
    b = torch.as_tensor(branch, dtype=torch.float32)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    b = b.to(device)

    q_all = torch.as_tensor(query, dtype=torch.float32)
    preds = []
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    for start in range(0, q_all.shape[0], int(query_batch_size)):
        q = q_all[start:start + int(query_batch_size)].unsqueeze(0).to(device)
        pred = model(b, q).squeeze(0)
        if y_normalizer is not None:
            pred = y_normalizer.decode(pred)
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0)


@torch.no_grad()
def predict_deeponet_cell_sample(
    model: nn.Module,
    sample: Mapping[str, np.ndarray],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    branch_channel_names: Sequence[str] = CELL_BRANCH_CHANNELS,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    query_batch_size: int = 65536,
) -> torch.Tensor:
    """Predict one cell-center dataset sample in physical units."""
    branch = normalize_cell_branch_with_y(
        sample["branch"],
        branch_channel_names=branch_channel_names,
        target_y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
    )
    return predict_deeponet_points(
        model=model,
        branch=branch,
        query=sample["query"],
        device=device,
        y_normalizer=y_normalizer,
        query_batch_size=query_batch_size,
    )


# Backwards-compatible name. It also works for cell-center query points.
predict_deeponet_full_grid = predict_deeponet_points
