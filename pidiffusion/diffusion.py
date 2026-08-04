"""Lightweight linear diffusion schedule and epsilon-prediction loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionSchedule:
    """Tensor representation of the linear forward diffusion schedule."""

    timesteps: int
    beta_start: float
    beta_end: float
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


def build_linear_schedule(
    timesteps: int = 1000,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> DiffusionSchedule:
    """Build the linear schedule used by the historical notebook."""

    if isinstance(timesteps, bool) or int(timesteps) <= 0:
        raise ValueError("timesteps must be a positive integer")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("beta values must be positive")
    if beta_start > beta_end:
        raise ValueError("beta_start must not exceed beta_end")

    timesteps = int(timesteps)
    betas = torch.linspace(
        float(beta_start),
        float(beta_end),
        timesteps,
        device=device,
        dtype=dtype,
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        timesteps=timesteps,
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
    )


def q_sample_points(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample noisy point targets from the forward diffusion process."""

    if noise is None:
        noise = torch.randn_like(x0)
    if noise.shape != x0.shape:
        raise ValueError(
            f"noise shape {tuple(noise.shape)} does not match x0 shape {tuple(x0.shape)}"
        )

    alpha_bar = schedule.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)
    alpha_bar_t = alpha_bar[t.to(device=x0.device, dtype=torch.long)].view(-1, 1)
    noisy = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )
    return noisy, noise


def epsilon_prediction_loss(
    model,
    branch: torch.Tensor,
    query: torch.Tensor,
    target: torch.Tensor,
    query_batch_id: torch.Tensor,
    schedule: DiffusionSchedule,
    branch_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the stochastic epsilon-prediction objective."""

    point_count = target.shape[0]
    t_query = torch.randint(
        low=0,
        high=schedule.timesteps,
        size=(point_count,),
        device=target.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target)
    noisy_target, noise = q_sample_points(
        target,
        t_query,
        schedule=schedule,
        noise=noise,
    )
    predicted_noise = model(
        branch=branch,
        query=query,
        noisy_target=noisy_target,
        t_query=t_query,
        query_batch_id=query_batch_id,
        branch_mask=branch_mask,
    )
    loss = F.mse_loss(predicted_noise, noise)
    return loss, {
        "pred_noise": predicted_noise,
        "noise": noise,
        "noisy_target": noisy_target,
        "t_query": t_query,
    }
