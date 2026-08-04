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


def _validate_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_alpha_bar_schedule(
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(alphas_cumprod):
        raise TypeError("alphas_cumprod must be a torch.Tensor")
    if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() == 0:
        raise ValueError("alphas_cumprod must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(alphas_cumprod):
        raise TypeError("alphas_cumprod must have a floating dtype")
    if not torch.isfinite(alphas_cumprod).all():
        raise ValueError("alphas_cumprod must contain only finite values")
    if not torch.all((alphas_cumprod > 0.0) & (alphas_cumprod <= 1.0)):
        raise ValueError("alphas_cumprod values must be in the interval (0, 1]")
    if alphas_cumprod.numel() > 1 and not torch.all(
        alphas_cumprod[1:] < alphas_cumprod[:-1]
    ):
        raise ValueError("alphas_cumprod must be strictly decreasing")
    return alphas_cumprod


def _validate_state_pair(
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
) -> None:
    if not torch.is_tensor(x_t) or not torch.is_tensor(epsilon_pred):
        raise TypeError("x_t and epsilon_pred must be torch.Tensor values")
    if x_t.shape != epsilon_pred.shape:
        raise ValueError(
            f"x_t shape {tuple(x_t.shape)} does not match epsilon_pred shape "
            f"{tuple(epsilon_pred.shape)}"
        )
    if not torch.is_floating_point(x_t) or not torch.is_floating_point(epsilon_pred):
        raise TypeError("x_t and epsilon_pred must have floating dtypes")
    if x_t.device != epsilon_pred.device:
        raise ValueError("x_t and epsilon_pred must be on the same device")
    if x_t.dtype != epsilon_pred.dtype:
        raise ValueError("x_t and epsilon_pred must have the same dtype")


def _validate_timestep(
    timestep: int,
    *,
    name: str,
    n_timesteps: int,
) -> int:
    timestep = _validate_integer(timestep, name)
    if timestep < 0 or timestep >= n_timesteps:
        raise ValueError(
            f"{name}={timestep} is outside the valid range [0, {n_timesteps - 1}]"
        )
    return timestep


def build_ddim_timesteps(
    num_sampling_steps: int,
    total_diffusion_steps: int,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Build the historical round-and-reverse DDIM timestep schedule."""

    num_sampling_steps = _validate_integer(num_sampling_steps, "num_sampling_steps")
    total_diffusion_steps = _validate_integer(
        total_diffusion_steps, "total_diffusion_steps"
    )
    if num_sampling_steps < 1:
        raise ValueError("num_sampling_steps must be at least one")
    if total_diffusion_steps < 1:
        raise ValueError("total_diffusion_steps must be at least one")
    if num_sampling_steps > total_diffusion_steps:
        raise ValueError(
            "num_sampling_steps cannot exceed total_diffusion_steps"
        )
    if num_sampling_steps == 1 and total_diffusion_steps > 1:
        raise ValueError(
            "At least two sampling steps are required to include both diffusion endpoints"
        )

    timesteps = torch.linspace(
        0,
        total_diffusion_steps - 1,
        steps=num_sampling_steps,
        device=device,
    ).round().long()
    timesteps = torch.unique_consecutive(timesteps)
    timesteps = torch.flip(timesteps, dims=[0])
    if int(timesteps[0]) != total_diffusion_steps - 1:
        raise RuntimeError("DDIM schedule does not start at the highest timestep")
    if int(timesteps[-1]) != 0:
        raise RuntimeError("DDIM schedule does not end at timestep zero")
    if timesteps.numel() > 1 and not torch.all(timesteps[:-1] > timesteps[1:]):
        raise RuntimeError("DDIM schedule is not strictly descending")
    return timesteps


def build_integer_segment_schedule(
    t_start: int,
    t_end: int,
    n_transitions: int,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Build one rounded, endpoint-preserving descending teacher segment."""

    t_start = _validate_integer(t_start, "t_start")
    t_end = _validate_integer(t_end, "t_end")
    n_transitions = _validate_integer(n_transitions, "n_transitions")
    if t_start <= t_end:
        raise ValueError("t_start must be greater than t_end")
    if t_end < 0:
        raise ValueError("t_end must be non-negative")
    if n_transitions < 1:
        raise ValueError("n_transitions must be at least one")

    schedule = torch.linspace(
        t_start,
        t_end,
        steps=n_transitions + 1,
        device=device,
    ).round().long()
    schedule[0] = t_start
    schedule[-1] = t_end
    if schedule.numel() != n_transitions + 1:
        raise RuntimeError("Segment schedule has an unexpected number of states")
    if not torch.all(schedule[:-1] > schedule[1:]):
        raise RuntimeError(
            "Rounded segment schedule is not strictly descending: "
            f"{schedule.detach().cpu().tolist()}"
        )
    return schedule


def _schedule_alpha_bar(
    alphas_cumprod: torch.Tensor,
    timestep: int,
    *,
    name: str,
) -> torch.Tensor:
    schedule = _validate_alpha_bar_schedule(alphas_cumprod)
    timestep = _validate_timestep(
        timestep,
        name=name,
        n_timesteps=int(schedule.numel()),
    )
    return schedule[timestep]


def ddim_step(
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
    t_current: int,
    t_next: int,
    alphas_cumprod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform one deterministic eta-zero DDIM transition and return ``(x_next, x0_pred)``."""

    _validate_state_pair(x_t, epsilon_pred)
    schedule = _validate_alpha_bar_schedule(alphas_cumprod)
    t_current = _validate_timestep(
        t_current,
        name="t_current",
        n_timesteps=int(schedule.numel()),
    )
    t_next = _validate_timestep(
        t_next,
        name="t_next",
        n_timesteps=int(schedule.numel()),
    )
    if t_next >= t_current:
        raise ValueError(
            f"Expected descending timesteps, got {t_current} -> {t_next}"
        )

    alpha_bar_current = schedule[t_current].to(
        device=x_t.device,
        dtype=x_t.dtype,
    )
    alpha_bar_next = schedule[t_next].to(
        device=x_t.device,
        dtype=x_t.dtype,
    )
    x0_pred = (
        x_t - torch.sqrt(1.0 - alpha_bar_current) * epsilon_pred
    ) / torch.sqrt(alpha_bar_current)
    x_next = (
        torch.sqrt(alpha_bar_next) * x0_pred
        + torch.sqrt(1.0 - alpha_bar_next) * epsilon_pred
    )
    return x_next, x0_pred


def final_clean_projection(
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
    timestep: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Project a timestep state to the clean field using epsilon prediction."""

    _validate_state_pair(x_t, epsilon_pred)
    alpha_bar = _schedule_alpha_bar(
        alphas_cumprod,
        timestep,
        name="timestep",
    ).to(device=x_t.device, dtype=x_t.dtype)
    return (
        x_t - torch.sqrt(1.0 - alpha_bar) * epsilon_pred
    ) / torch.sqrt(alpha_bar)


def equivalent_epsilon_target(
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    t_start: int,
    t_end: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Solve the deterministic DDIM equation for an endpoint-equivalent epsilon."""

    if not torch.is_tensor(x_start) or not torch.is_tensor(x_end):
        raise TypeError("x_start and x_end must be torch.Tensor values")
    if x_start.shape != x_end.shape:
        raise ValueError(
            f"x_start shape {tuple(x_start.shape)} does not match x_end shape "
            f"{tuple(x_end.shape)}"
        )
    if not torch.is_floating_point(x_start) or not torch.is_floating_point(x_end):
        raise TypeError("x_start and x_end must have floating dtypes")
    if x_start.device != x_end.device or x_start.dtype != x_end.dtype:
        raise ValueError("x_start and x_end must have the same device and dtype")

    schedule = _validate_alpha_bar_schedule(alphas_cumprod)
    t_start = _validate_timestep(
        t_start,
        name="t_start",
        n_timesteps=int(schedule.numel()),
    )
    t_end = _validate_timestep(
        t_end,
        name="t_end",
        n_timesteps=int(schedule.numel()),
    )
    if t_end >= t_start:
        raise ValueError(
            f"Expected descending timesteps, got {t_start} -> {t_end}"
        )

    alpha_bar_start = schedule[t_start].to(
        device=x_start.device,
        dtype=x_start.dtype,
    )
    alpha_bar_end = schedule[t_end].to(
        device=x_start.device,
        dtype=x_start.dtype,
    )
    state_coefficient = torch.sqrt(alpha_bar_end / alpha_bar_start)
    epsilon_coefficient = (
        torch.sqrt(1.0 - alpha_bar_end)
        - state_coefficient * torch.sqrt(1.0 - alpha_bar_start)
    )
    if torch.abs(epsilon_coefficient).item() < 1.0e-12:
        raise RuntimeError("Equivalent epsilon coefficient is too small")
    return (x_end - state_coefficient * x_start) / epsilon_coefficient
