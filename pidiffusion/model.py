"""Canonical point-set conditional diffusion model components."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: str = "gelu",
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build the branch MLP with the historical module layout."""

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
    """Permutation-invariant branch network for point-set conditions."""

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

    def forward(
        self,
        branch: torch.Tensor,
        branch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if branch.ndim != 3:
            raise ValueError(f"Expected branch shape (B,M,C), got {tuple(branch.shape)}")
        h = self.point_encoder(branch)

        if branch_mask is not None:
            if branch_mask.ndim != 2:
                raise ValueError(
                    f"Expected branch_mask shape (B,M), got {tuple(branch_mask.shape)}"
                )
            if branch_mask.shape != branch.shape[:2]:
                raise ValueError(
                    f"branch_mask shape {tuple(branch_mask.shape)} does not match "
                    f"branch shape {tuple(branch.shape[:2])}"
                )

            mask_bool = branch_mask.to(device=h.device, dtype=torch.bool)
            mask = mask_bool.unsqueeze(-1).to(dtype=h.dtype)

            if self.aggregation == "mean":
                h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            elif self.aggregation == "sum":
                h = (h * mask).sum(dim=1)
            else:
                h_masked = h.masked_fill(
                    ~mask_bool.unsqueeze(-1), torch.finfo(h.dtype).min
                )
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


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Create the sinusoidal timestep embedding used by the notebook model."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat(
        [torch.sin(arguments), torch.cos(arguments)],
        dim=1,
    )
    if dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=1,
        )
    return embedding


def _make_denoiser_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    depth: int = 4,
) -> nn.Sequential:
    """Build the per-query MLP used by the canonical denoiser."""

    layers = []
    in_dim = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class PointSetDiffusionDenoiser(nn.Module):
    """Point-set conditional epsilon-prediction denoiser."""

    def __init__(
        self,
        branch_input_dim: int,
        query_input_dim: int = 2,
        target_dim: int = 3,
        latent_dim: int = 128,
        time_dim: int = 128,
        branch_point_hidden_dim: int = 128,
        branch_global_hidden_dim: int = 128,
        denoiser_hidden_dim: int = 256,
        denoiser_depth: int = 4,
    ):
        super().__init__()

        self.target_dim = int(target_dim)
        self.latent_dim = int(latent_dim)
        self.time_dim = int(time_dim)

        self.branch_net = PointSetBranchNet(
            branch_input_dim=branch_input_dim,
            latent_dim=latent_dim,
            output_channels=target_dim,
            point_hidden_dim=branch_point_hidden_dim,
            point_depth=3,
            global_hidden_dim=branch_global_hidden_dim,
            global_depth=3,
            aggregation="mean",
            activation="gelu",
            layer_norm=False,
        )

        denoiser_input_dim = (
            query_input_dim
            + target_dim
            + time_dim
            + target_dim * latent_dim
        )
        self.denoiser = _make_denoiser_mlp(
            input_dim=denoiser_input_dim,
            output_dim=target_dim,
            hidden_dim=denoiser_hidden_dim,
            depth=denoiser_depth,
        )

    def forward(
        self,
        branch: torch.Tensor,
        query: torch.Tensor,
        noisy_target: torch.Tensor,
        t_query: torch.Tensor,
        query_batch_id: torch.Tensor,
        branch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        branch_coeff = self.branch_net(branch, branch_mask)
        branch_embedding = branch_coeff.reshape(branch_coeff.shape[0], -1)
        branch_embedding_query = branch_embedding[query_batch_id]
        timestep_embedding = sinusoidal_timestep_embedding(t_query, self.time_dim)
        model_input = torch.cat(
            [query, noisy_target, timestep_embedding, branch_embedding_query],
            dim=1,
        )
        return self.denoiser(model_input)
