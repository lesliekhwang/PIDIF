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

    def forward(self, branch: torch.Tensor, branch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if branch.ndim != 3:
            raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")
        h = self.point_encoder(branch)
        
        if branch_mask is not None:
            if branch_mask.ndim != 2:
                raise ValueError(f"Expected branch_mask shape (B,M), got {tuple(branch_mask.shape)}")
            if branch_mask.shape != branch.shape[:2]:
                raise ValueError(f"branch_mask shape {tuple(branch_mask.shape)} does not match branch shape {tuple(branch.shape[:2])}")
            
            mask_bool = branch_mask.to(device=h.device, dtype=torch.bool)
            mask = mask_bool.unsqueeze(-1).to(dtype=h.dtype)
            
            if self.aggregation == "mean":
                h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            elif self.aggregation == "sum":
                h = (h * mask).sum(dim=1)
            else:
                h_masked = h.masked_fill(~mask_bool.unsqueeze(-1), torch.finfo(h.dtype).min)
                h = h_masked.max(dim=1).values
                valid_any = mask_bool.any(dim=1)
                h = torch.where(valid_any.unsqueeze(-1), h, torch.zeros_like(h))
        
        else:
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

    def forward(self, branch: torch.Tensor, query: torch.Tensor, query_batch_id: Optional[torch.Tensor] = None, branch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if branch.ndim != 3:
            raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")
        coeff = self.branch_net(branch, branch_mask)  # (B, Cout, R)
        
        # ------------------------------------------------------------
        # Concatenated mode:
        #   branch:         (B, M, Cb)
        #   branch_mask:    (B, M), True for real branch points, False for padding
        #   query:          (N_total, Cq)
        #   query_batch_id: (N_total,)
        #   output:         (N_total, Cout)
        # ------------------------------------------------------------
        if query_batch_id is not None:
            if query.ndim != 2:
                raise ValueError(
                    f"Ragged mode expects query shape (N_total,C), got {tuple(query.shape)}"
                )
            if query_batch_id.ndim != 1:
                raise ValueError(
                    f"Expected query_batch_id shape (N_total,), got {tuple(query_batch_id.shape)}"
                )
            if query.shape[0] != query_batch_id.shape[0]:
                raise ValueError("query and query_batch_id must have the same first dimension")

            basis = self.trunk_net(query)  # (N_total, R)

            if basis.ndim == 3 and basis.shape[0] == 1:
                basis = basis.squeeze(0)

            if basis.ndim != 2:
                raise ValueError(
                    f"Ragged mode expects trunk_net(query) to return (N_total,R), got {tuple(basis.shape)}"
                )

            coeff_per_query = coeff[query_batch_id]  # (N_total, Cout, R)

            out = torch.einsum("ncr,nr->nc", coeff_per_query, basis)

            return out + self.bias.reshape(1, -1)
        
        # ------------------------------------------------------------
        # Original fixed-size mode:
        #   branch: (B, M, Cb)
        #   query:  (B, Q, Cq) or (Q, Cq)
        #   output: (B, Q, Cout)
        # ------------------------------------------------------------
        if query.ndim == 2:
            query = query.unsqueeze(0).expand(branch.shape[0], -1, -1)

        if query.ndim != 3:
            raise ValueError(f"Expected query shape (B,Q,C) or (Q,C), got {tuple(query.shape)}")

        if query.shape[0] != branch.shape[0]:
            if query.shape[0] == 1:
                query = query.expand(branch.shape[0], -1, -1)
            else:
                raise ValueError("branch and query batch dimensions do not match")

        basis = self.trunk_net(query)  # (B, Q, R)

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

def relative_L2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    num = torch.linalg.norm(pred_f - target_f, dim=1)
    den = torch.linalg.norm(target_f, dim=1).clamp_min(eps)
    return (num / den).mean()
    
def ragged_mse_loss(pred, target, query_batch_id, batch_size):
    point_loss = ((pred - target) ** 2).mean(dim=-1)  # (N_total,)

    loss_sum = pred.new_zeros(batch_size)
    count = pred.new_zeros(batch_size)

    loss_sum.scatter_add_(0, query_batch_id, point_loss)
    count.scatter_add_(0, query_batch_id, torch.ones_like(point_loss))

    loss_per_sample = loss_sum / count.clamp_min(1.0)

    return loss_per_sample.mean()

def ragged_relative_l2_loss(pred, target, query_batch_id, batch_size, eps=1e-12):
    error_sq = ((pred - target) ** 2).sum(dim=-1)
    target_sq = (target ** 2).sum(dim=-1)

    error_sum = pred.new_zeros(batch_size)
    target_sum = pred.new_zeros(batch_size)

    error_sum.scatter_add_(0, query_batch_id, error_sq)
    target_sum.scatter_add_(0, query_batch_id, target_sq)

    rel = torch.sqrt(error_sum / target_sum.clamp_min(eps))

    return rel.mean()


def boundary_loss(
    model: nn.Module,
    branch: torch.Tensor,
    branch_channel_names: Sequence[str],
    output_channel_names: Optional[Sequence[str]] = None,
    known_masked: bool = True,
    branch_mask: Optional[torch.Tensor] = None,
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
    if output_channel_names is None:
        output_channel_names = [
            name[len("boundary_") :]
            for name in names
            if name.startswith("boundary_")
        ]
    output_channel_names = list(output_channel_names)

    x_ch = names.index("x_local")
    y_ch = names.index("y_local")
    value_ch = [
        names.index(f"boundary_{f}") for f in output_channel_names
    ]
    known_names = [
        f"known_{f}" for f in output_channel_names
    ]

    query_bc = branch[..., [x_ch, y_ch]]
    target_bc = branch[..., value_ch]
    pred_bc = model(branch, query_bc, branch_mask=branch_mask)

    if known_masked and all(k in names for k in known_names):
        known_ch = [
            names.index(f"known_{f}") for f in output_channel_names
        ]
        known = branch[..., known_ch]
        if branch_mask is not None:
            known = known * branch_mask.to(device=known.device, dtype=known.dtype).unsqueeze(-1)
        denom = known.sum().clamp_min(1.0)
        return torch.sum(((pred_bc - target_bc) ** 2) * known) / denom
    
    if branch_mask is not None:
        mask = branch_mask.to(device=pred_bc.device, dtype=pred_bc.dtype).unsqueeze(-1)
        denom = (mask.sum() * pred_bc.shape[-1]).clamp_min(1.0)
        return torch.sum(((pred_bc - target_bc) ** 2) * mask) / denom

    return F.mse_loss(pred_bc, target_bc)


def _field_index(
    output_channel_names: Sequence[str],
    *aliases: str,
) -> int:
    names = list(output_channel_names)
    for name in aliases:
        if name in names:
            return names.index(name)
    raise ValueError(
        f"Required output field is missing; expected one of {aliases}, got {names}"
    )


def _physics_batch_parameter(
    value,
    sample_indices: torch.Tensor,
    batch_size: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Resolve a scalar or dataset-length physics parameter for this batch."""
    tensor = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)

    tensor = tensor.reshape(-1)
    indices = sample_indices.to(device=tensor.device, dtype=torch.long)
    if indices.numel() != batch_size:
        raise ValueError(
            f"sample_indices has {indices.numel()} values for batch size {batch_size}"
        )
    if indices.numel() and int(indices.min()) < 0:
        raise ValueError("sample_indices must be non-negative")
    if indices.numel() and int(indices.max()) >= tensor.numel():
        raise ValueError(
            f"Physics parameter {name!r} has {tensor.numel()} values but "
            f"sample index {int(indices.max())} was requested"
        )
    return tensor[indices]


def _sample_pde_query_indices(
    query_batch_id: torch.Tensor,
    batch_size: int,
    max_points_per_sample: Optional[int],
) -> torch.Tensor:
    if max_points_per_sample is None:
        return torch.arange(query_batch_id.numel(), device=query_batch_id.device)
    max_points = int(max_points_per_sample)
    if max_points < 1:
        raise ValueError("max_points_per_sample must be >= 1 or None")

    selected = []
    for batch_id in range(batch_size):
        indices = torch.where(query_batch_id == batch_id)[0]
        if indices.numel() > max_points:
            order = torch.randperm(indices.numel(), device=indices.device)[:max_points]
            indices = indices[order]
        selected.append(indices)
    if not selected:
        return torch.empty(0, dtype=torch.long, device=query_batch_id.device)
    return torch.cat(selected)


def _query_gradient(
    output: torch.Tensor,
    query: torch.Tensor,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Pointwise derivatives for a point-independent DeepONet trunk evaluation."""
    if not output.requires_grad:
        return torch.zeros_like(query)
    gradient = torch.autograd.grad(
        output.sum(),
        query,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(query) + output.sum() * 0.0
    return gradient


def steady_deeponet_pde_loss(
    model: nn.Module,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    sample_indices: torch.Tensor,
    output_channel_names: Sequence[str],
    y_normalizer: FeatureNormalizer,
    physics: Mapping[str, object],
    branch_mask: Optional[torch.Tensor] = None,
    residual_weights: Optional[Mapping[str, float]] = None,
    max_points_per_sample: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Steady incompressible Navier--Stokes residual on ragged DeepONet queries.

    ``query`` contains each subdomain's local coordinates in ``[0, 1]^2``.
    Predictions are decoded to physical units and then nondimensionalized with
    each sample's velocity and length scales. This gives the same aspect-ratio
    form used by ``PINN/Modules/loss/loss_2dns.py`` while remaining correct when
    output channels are standardized for supervised training.

    Required ``physics`` entries are ``density`` (kg/m^3),
    ``kinematic_viscosity`` (m^2/s), ``x_length`` (m), ``y_length`` (m), and
    ``velocity_scale`` (m/s). Each may be a scalar or a dataset-length sequence
    indexed by ``sample_indices``. If a temperature output and
    ``thermal_diffusivity`` are both present, the steady advection-diffusion
    residual is included as ``energy``.
    """
    required_physics = (
        "density",
        "kinematic_viscosity",
        "x_length",
        "y_length",
        "velocity_scale",
    )
    missing = [name for name in required_physics if name not in physics]
    if missing:
        raise ValueError(f"Missing PDE physics parameters: {missing}")

    batch_size = int(branch.shape[0])
    selected = _sample_pde_query_indices(
        query_batch_id=query_batch_id,
        batch_size=batch_size,
        max_points_per_sample=max_points_per_sample,
    )
    if selected.numel() == 0:
        raise ValueError("Cannot compute PDE loss for an empty query batch")

    pde_query = query[selected].detach().clone().requires_grad_(True)
    pde_batch_id = query_batch_id[selected]
    pred_normalized = model(
        branch,
        pde_query,
        query_batch_id=pde_batch_id,
        branch_mask=branch_mask,
    )

    mean = y_normalizer.mean.to(
        device=pred_normalized.device,
        dtype=pred_normalized.dtype,
    )
    std = y_normalizer.std.to(
        device=pred_normalized.device,
        dtype=pred_normalized.dtype,
    )
    pred_physical = pred_normalized * std.reshape(1, -1) + mean.reshape(1, -1)

    density = _physics_batch_parameter(
        physics["density"], sample_indices, batch_size, pred_physical, "density"
    )
    viscosity = _physics_batch_parameter(
        physics["kinematic_viscosity"],
        sample_indices,
        batch_size,
        pred_physical,
        "kinematic_viscosity",
    )
    x_length = _physics_batch_parameter(
        physics["x_length"], sample_indices, batch_size, pred_physical, "x_length"
    )
    y_length = _physics_batch_parameter(
        physics["y_length"], sample_indices, batch_size, pred_physical, "y_length"
    )
    velocity_scale = _physics_batch_parameter(
        physics["velocity_scale"],
        sample_indices,
        batch_size,
        pred_physical,
        "velocity_scale",
    )
    for name, value in (
        ("density", density),
        ("kinematic_viscosity", viscosity),
        ("x_length", x_length),
        ("y_length", y_length),
        ("velocity_scale", velocity_scale),
    ):
        if torch.any(value <= 0):
            raise ValueError(f"PDE physics parameter {name!r} must be positive")

    density_q = density[pde_batch_id].unsqueeze(-1)
    x_length_q = x_length[pde_batch_id].unsqueeze(-1)
    y_length_q = y_length[pde_batch_id].unsqueeze(-1)
    velocity_q = velocity_scale[pde_batch_id].unsqueeze(-1)
    gamma = x_length_q / y_length_q
    inverse_reynolds = (
        viscosity[pde_batch_id].unsqueeze(-1) / (velocity_q * x_length_q)
    )

    p_idx = _field_index(output_channel_names, "pressure", "p")
    u_idx = _field_index(output_channel_names, "u", "u_velocity")
    v_idx = _field_index(output_channel_names, "v", "v_velocity")
    pressure = pred_physical[:, p_idx : p_idx + 1] / (
        density_q * velocity_q.square()
    )
    u_velocity = pred_physical[:, u_idx : u_idx + 1] / velocity_q
    v_velocity = pred_physical[:, v_idx : v_idx + 1] / velocity_q

    pressure_grad = _query_gradient(pressure, pde_query)
    u_grad = _query_gradient(u_velocity, pde_query)
    v_grad = _query_gradient(v_velocity, pde_query)
    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
    v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]
    p_x, p_y = pressure_grad[:, 0:1], pressure_grad[:, 1:2]

    u_x_grad = _query_gradient(u_x, pde_query)
    u_y_grad = _query_gradient(u_y, pde_query)
    v_x_grad = _query_gradient(v_x, pde_query)
    v_y_grad = _query_gradient(v_y, pde_query)
    u_xx, u_yy = u_x_grad[:, 0:1], u_y_grad[:, 1:2]
    v_xx, v_yy = v_x_grad[:, 0:1], v_y_grad[:, 1:2]

    residuals = {
        "continuity": u_x + gamma * v_y,
        "x_momentum": (
            u_velocity * u_x
            + gamma * v_velocity * u_y
            + p_x
            - inverse_reynolds * (u_xx + gamma.square() * u_yy)
        ),
        "y_momentum": (
            u_velocity * v_x
            + gamma * v_velocity * v_y
            + gamma * p_y
            - inverse_reynolds * (v_xx + gamma.square() * v_yy)
        ),
    }

    temperature_name = next(
        (name for name in ("temperature", "T") if name in output_channel_names),
        None,
    )
    if temperature_name is not None and "thermal_diffusivity" in physics:
        temperature_idx = list(output_channel_names).index(temperature_name)
        temperature_scale = physics.get(
            "temperature_scale",
            float(std[temperature_idx].detach().clamp_min(1.0e-12)),
        )
        temperature_scale_batch = _physics_batch_parameter(
            temperature_scale,
            sample_indices,
            batch_size,
            pred_physical,
            "temperature_scale",
        )
        thermal_diffusivity = _physics_batch_parameter(
            physics["thermal_diffusivity"],
            sample_indices,
            batch_size,
            pred_physical,
            "thermal_diffusivity",
        )
        if torch.any(temperature_scale_batch <= 0) or torch.any(
            thermal_diffusivity <= 0
        ):
            raise ValueError(
                "temperature_scale and thermal_diffusivity must be positive"
            )
        temperature = pred_physical[
            :, temperature_idx : temperature_idx + 1
        ] / temperature_scale_batch[pde_batch_id].unsqueeze(-1)
        temperature_grad = _query_gradient(temperature, pde_query)
        temperature_x = temperature_grad[:, 0:1]
        temperature_y = temperature_grad[:, 1:2]
        temperature_xx = _query_gradient(temperature_x, pde_query)[:, 0:1]
        temperature_yy = _query_gradient(temperature_y, pde_query)[:, 1:2]
        inverse_peclet = (
            thermal_diffusivity[pde_batch_id].unsqueeze(-1)
            / (velocity_q * x_length_q)
        )
        residuals["energy"] = (
            u_velocity * temperature_x
            + gamma * v_velocity * temperature_y
            - inverse_peclet
            * (temperature_xx + gamma.square() * temperature_yy)
        )

    weights = {
        "continuity": 1.0,
        "x_momentum": 1.0,
        "y_momentum": 1.0,
        "energy": 1.0,
    }
    if residual_weights is not None:
        unknown_weights = set(residual_weights) - set(weights)
        if unknown_weights:
            raise ValueError(
                f"Unknown PDE residual weights: {sorted(unknown_weights)}"
            )
        weights.update({name: float(value) for name, value in residual_weights.items()})

    loss_terms = {
        name: ragged_mse_loss(
            residual,
            torch.zeros_like(residual),
            pde_batch_id,
            batch_size,
        )
        for name, residual in residuals.items()
    }
    loss_terms["loss"] = sum(
        weights[name] * term for name, term in loss_terms.items()
    )
    return loss_terms


def transient_condition_losses(
    model: nn.Module,
    branch: torch.Tensor,
    branch_channel_names: Sequence[str],
    output_channel_names: Sequence[str],
    branch_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boundary and initial-condition MSE for transient branch sensors.

    Transient branch points store ``(x_local, y_local, time_local)`` along with
    ``boundary_mask``/``initial_mask``, normalized ``sensor_<field>`` values,
    and ``known_<field>`` flags. Only prescribed fields contribute:

    * inlet: known velocity components,
    * outlet: known pressure,
    * channel/cylinder walls: known velocity components,
    * initial sensors: all supplied solution fields.

    Boundary and initial points are gathered into one ragged model call so the
    point-set branch encoder is evaluated only once for both condition losses.
    """
    if branch.ndim != 3:
        raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")

    names = list(branch_channel_names)
    output_names = list(output_channel_names)
    required = [
        "x_local",
        "y_local",
        "time_local",
        "boundary_mask",
        "initial_mask",
    ]
    required += [f"sensor_{name}" for name in output_names]
    required += [f"known_{name}" for name in output_names]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            "Transient condition loss requires transient branch channels; "
            f"missing {missing}"
        )

    if branch_mask is None:
        valid_points = torch.ones(
            branch.shape[:2], dtype=torch.bool, device=branch.device
        )
    else:
        if branch_mask.shape != branch.shape[:2]:
            raise ValueError(
                f"branch_mask shape {tuple(branch_mask.shape)} does not match "
                f"branch shape {tuple(branch.shape[:2])}"
            )
        valid_points = branch_mask.to(device=branch.device, dtype=torch.bool)

    boundary_idx = names.index("boundary_mask")
    initial_idx = names.index("initial_mask")
    boundary_points = valid_points & (branch[..., boundary_idx] > 0.5)
    initial_points = valid_points & (branch[..., initial_idx] > 0.5)
    condition_points = boundary_points | initial_points

    point_indices = torch.nonzero(condition_points, as_tuple=False)
    if point_indices.numel() == 0:
        zero = branch.sum() * 0.0
        return zero, zero

    batch_indices = point_indices[:, 0]
    sensor_indices = point_indices[:, 1]
    condition_features = branch[batch_indices, sensor_indices]

    query_channels = [
        names.index("x_local"),
        names.index("y_local"),
        names.index("time_local"),
    ]
    value_channels = [names.index(f"sensor_{name}") for name in output_names]
    known_channels = [names.index(f"known_{name}") for name in output_names]

    condition_query = condition_features[:, query_channels]
    condition_target = condition_features[:, value_channels]
    known = condition_features[:, known_channels]
    condition_pred = model(
        branch,
        condition_query,
        query_batch_id=batch_indices,
        branch_mask=branch_mask,
    )

    squared_error = (condition_pred - condition_target) ** 2
    boundary_weight = (
        boundary_points[batch_indices, sensor_indices]
        .to(dtype=branch.dtype)
        .unsqueeze(-1)
        * known
    )
    initial_weight = (
        initial_points[batch_indices, sensor_indices]
        .to(dtype=branch.dtype)
        .unsqueeze(-1)
        * known
    )

    boundary_denom = boundary_weight.sum()
    initial_denom = initial_weight.sum()
    boundary_mse = (
        (squared_error * boundary_weight).sum()
        / boundary_denom.clamp_min(1.0)
    )
    initial_mse = (
        (squared_error * initial_weight).sum()
        / initial_denom.clamp_min(1.0)
    )
    return boundary_mse, initial_mse


def transient_boundary_loss(
    model: nn.Module,
    branch: torch.Tensor,
    branch_channel_names: Sequence[str],
    output_channel_names: Sequence[str],
    branch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the masked transient physical-boundary loss."""
    boundary_mse, _ = transient_condition_losses(
        model=model,
        branch=branch,
        branch_channel_names=branch_channel_names,
        output_channel_names=output_channel_names,
        branch_mask=branch_mask,
    )
    return boundary_mse


def transient_initial_condition_loss(
    model: nn.Module,
    branch: torch.Tensor,
    branch_channel_names: Sequence[str],
    output_channel_names: Sequence[str],
    branch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the masked transient initial-condition loss."""
    _, initial_mse = transient_condition_losses(
        model=model,
        branch=branch,
        branch_channel_names=branch_channel_names,
        output_channel_names=output_channel_names,
        branch_mask=branch_mask,
    )
    return initial_mse


def train_transient_deeponet_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    loss_type: str = "mse",
    lambda_bc: float = 1.0,
    lambda_ic: float = 1.0,
    branch_channel_names: Optional[Sequence[str]] = None,
    output_channel_names: Optional[Sequence[str]] = None,
) -> Tuple[float, float, float]:
    """Train one transient epoch and return field, weighted BC, and weighted IC losses."""
    if branch_channel_names is None:
        raise ValueError("branch_channel_names is required for transient condition losses")
    if output_channel_names is None:
        raise ValueError("output_channel_names is required for transient condition losses")
    if float(lambda_bc) < 0.0 or float(lambda_ic) < 0.0:
        raise ValueError("lambda_bc and lambda_ic must be non-negative")

    model.train()
    device = torch.device(device)
    total_field_loss = 0.0
    total_bc_loss = 0.0
    total_ic_loss = 0.0
    count = 0

    for batch in loader:
        if len(batch) == 6:
            branch, query, target, query_batch_id, _sample_idx, branch_mask = batch
        elif len(batch) == 5:
            branch, query, target, query_batch_id, _sample_idx = batch
            branch_mask = None
        else:
            raise ValueError(f"Expected 5 or 6 elements in batch, got {len(batch)}")

        batch_size = int(branch.shape[0])
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)
        query_batch_id = query_batch_id.to(device)
        if branch_mask is not None:
            branch_mask = branch_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(branch, query, query_batch_id, branch_mask)
        if loss_type.lower() == "relative_l2":
            field_loss = ragged_relative_l2_loss(
                pred, target, query_batch_id, batch_size
            )
        else:
            field_loss = ragged_mse_loss(
                pred, target, query_batch_id, batch_size
            )

        boundary_mse, initial_mse = transient_condition_losses(
            model=model,
            branch=branch,
            branch_channel_names=branch_channel_names,
            output_channel_names=output_channel_names,
            branch_mask=branch_mask,
        )
        weighted_bc_loss = float(lambda_bc) * boundary_mse
        weighted_ic_loss = float(lambda_ic) * initial_mse
        loss = field_loss + weighted_bc_loss + weighted_ic_loss

        loss.backward()
        optimizer.step()

        total_field_loss += float(field_loss.item()) * batch_size
        total_bc_loss += float(weighted_bc_loss.item()) * batch_size
        total_ic_loss += float(weighted_ic_loss.item()) * batch_size
        count += batch_size

    denominator = max(count, 1)
    return (
        total_field_loss / denominator,
        total_bc_loss / denominator,
        total_ic_loss / denominator,
    )



def train_deeponet_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    loss_type: str = "mse",
    lambda_bc: float = 0.0,
    lambda_pde: float = 0.0,
    branch_channel_names: Optional[Sequence[str]] = None,
    output_channel_names: Optional[Sequence[str]] = None,
    y_normalizer: Optional[FeatureNormalizer] = None,
    pde_physics: Optional[Mapping[str, object]] = None,
    pde_residual_weights: Optional[Mapping[str, float]] = None,
    n_pde_points: Optional[int] = None,
) -> tuple[float, float, float]:
    """Train one steady epoch and return field, weighted BC, and weighted PDE loss."""
    if float(lambda_bc) < 0.0 or float(lambda_pde) < 0.0:
        raise ValueError("lambda_bc and lambda_pde must be non-negative")
    if lambda_pde > 0.0:
        if output_channel_names is None:
            raise ValueError("output_channel_names is required when PDE loss is active")
        if y_normalizer is None:
            raise ValueError("y_normalizer is required when PDE loss is active")
        if pde_physics is None:
            raise ValueError("pde_physics is required when PDE loss is active")

    model.train()
    device = torch.device(device)
    total_bc_loss = 0.0
    total_pde_loss = 0.0
    total_field_loss = 0.0
    count = 0

    for batch in loader:
        if len(batch) == 6:
            branch, query, target, query_batch_id, _sample_idx, branch_mask = batch
        elif len(batch) == 5:
            branch, query, target, query_batch_id, _sample_idx = batch
            branch_mask = None
        else:
            raise ValueError(f"Expected 5 or 6 elements in batch, got {len(batch)}")
        bs = int(branch.shape[0])
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)
        query_batch_id = query_batch_id.to(device)
        sample_idx = _sample_idx.to(device)
        if branch_mask is not None:
            branch_mask = branch_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(branch, query, query_batch_id, branch_mask)
        
        if loss_type.lower() == "relative_l2":
            field_loss = ragged_relative_l2_loss(pred, target, query_batch_id, bs)
        else:
            field_loss = ragged_mse_loss(pred, target, query_batch_id, bs)
            
        total_field_loss += float(field_loss.item()) * bs

        loss = field_loss
        if lambda_bc > 0.0:
            if branch_channel_names is None:
                raise ValueError("branch_channel_names is required when boundary loss is active")
            if output_channel_names is None:
                raise ValueError("output_channel_names is required when boundary loss is active")
            bc_loss = float(lambda_bc) * boundary_loss(
                model=model,
                branch=branch,
                branch_channel_names=branch_channel_names,
                output_channel_names=output_channel_names,
                branch_mask=branch_mask,
            )
            total_bc_loss += float(bc_loss.item()) * bs
            loss = loss + bc_loss

        if lambda_pde > 0.0:
            assert output_channel_names is not None
            assert y_normalizer is not None
            assert pde_physics is not None
            pde_terms = steady_deeponet_pde_loss(
                model=model,
                branch=branch,
                query=query,
                query_batch_id=query_batch_id,
                sample_indices=sample_idx,
                output_channel_names=output_channel_names,
                y_normalizer=y_normalizer,
                physics=pde_physics,
                branch_mask=branch_mask,
                residual_weights=pde_residual_weights,
                max_points_per_sample=n_pde_points,
            )
            pde_loss = float(lambda_pde) * pde_terms["loss"]
            total_pde_loss += float(pde_loss.item()) * bs
            loss = loss + pde_loss

        loss.backward()
        optimizer.step()
        count += bs

    denominator = max(count, 1)
    return (
        total_field_loss / denominator,
        total_bc_loss / denominator,
        total_pde_loss / denominator,
    )


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
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
    for batch in loader:
        if len(batch) == 6:
            branch, query, target, query_batch_id, _sample_idx, branch_mask = batch
        elif len(batch) == 5:
            branch, query, target, query_batch_id, _sample_idx = batch
            branch_mask = None
        else:
            raise ValueError(f"Expected 5 or 6 elements in batch, got {len(batch)}")
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)
        query_batch_id = query_batch_id.to(device)
        if branch_mask is not None:
            branch_mask = branch_mask.to(device)
        pred = model(branch, query, query_batch_id, branch_mask)

        pred_metric = pred
        target_metric = target
        if y_normalizer is not None:
            pred_metric = y_normalizer.decode(pred)
            target_metric = y_normalizer.decode(target)

        bs = int(branch.shape[0])
        mse = ragged_mse_loss(pred_metric, target_metric, query_batch_id, bs)
        rel = ragged_relative_l2_loss(pred_metric, target_metric, query_batch_id, bs)
        mse_sum += float(mse.item()) * bs
        rel_sum += float(rel.item()) * bs
        count += bs

        diff = pred_metric - target_metric
        sse = torch.sum(diff ** 2, dim=0)
        energy = torch.sum(target_metric ** 2, dim=0)
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
def predict_cell_sample(
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
