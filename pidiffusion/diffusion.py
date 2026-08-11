"""Linear diffusion schedule, epsilon losses, and deterministic DDIM utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Tensor representation of the linear forward diffusion schedule."""

    timesteps: int
    beta_start: float
    beta_end: float
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


# Canonical branch schema validated by pidiffusion.data.
_BRANCH_XY = (0, 1)
_BRANCH_BOUNDARY_VALUES = (4, 5, 6)
_BRANCH_KNOWN_MASKS = (7, 8, 9)


def build_linear_schedule(
    timesteps: int = 1000,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> DiffusionSchedule:
    """Build a linear beta schedule."""

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

    if x0.ndim != 2:
        raise ValueError(f"x0 must be two-dimensional, got shape {tuple(x0.shape)}")
    if t.ndim != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError(
            "t must be one-dimensional with one timestep per x0 row; "
            f"got t={tuple(t.shape)}, x0={tuple(x0.shape)}"
        )
    if noise is None:
        noise = torch.randn_like(x0)
    if noise.shape != x0.shape:
        raise ValueError(
            f"noise shape {tuple(noise.shape)} does not match x0 shape {tuple(x0.shape)}"
        )

    t = t.to(device=x0.device, dtype=torch.long)
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= schedule.timesteps):
        raise ValueError(
            f"timestep values must lie in [0, {schedule.timesteps - 1}]"
        )

    alpha_bar = schedule.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)
    alpha_bar_t = alpha_bar[t].view(-1, 1)
    noisy = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return noisy, noise


def _validate_query_batch_id(
    query_batch_id: torch.Tensor,
    *,
    n_queries: int,
    batch_size: int,
) -> torch.Tensor:
    if query_batch_id.ndim != 1 or query_batch_id.shape[0] != n_queries:
        raise ValueError(
            "query_batch_id must be one-dimensional with one entry per query point"
        )
    query_batch_id = query_batch_id.to(dtype=torch.long)
    if n_queries == 0:
        raise ValueError("Each batch must contain at least one query point")
    if int(query_batch_id.min()) < 0 or int(query_batch_id.max()) >= batch_size:
        raise ValueError("query_batch_id contains an out-of-range sample index")
    counts = torch.bincount(query_batch_id, minlength=batch_size)
    if torch.any(counts == 0):
        raise ValueError("Every subdomain in the batch must contain at least one query point")
    return query_batch_id


def _balanced_subdomain_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    query_batch_id: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Average MSE within each subdomain, then average subdomains equally."""

    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted shape {tuple(predicted.shape)} does not match target "
            f"shape {tuple(target.shape)}"
        )
    if predicted.ndim != 2:
        raise ValueError("predicted and target must be two-dimensional")

    per_point = (predicted - target).square().mean(dim=1)
    sums = per_point.new_zeros(batch_size)
    sums.index_add_(0, query_batch_id, per_point)
    counts = torch.bincount(query_batch_id, minlength=batch_size).to(per_point.dtype)
    return (sums / counts.clamp_min(1.0)).mean()


def _resolve_subdomain_timesteps(
    *,
    batch_size: int,
    schedule: DiffusionSchedule,
    device: torch.device,
    t_subdomain: Optional[torch.Tensor],
) -> torch.Tensor:
    if t_subdomain is None:
        return torch.randint(
            low=0,
            high=schedule.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    if t_subdomain.ndim != 1 or t_subdomain.shape[0] != batch_size:
        raise ValueError(
            "t_subdomain must have shape (batch_size,), got "
            f"{tuple(t_subdomain.shape)} for batch_size={batch_size}"
        )
    t_subdomain = t_subdomain.to(device=device, dtype=torch.long)
    if t_subdomain.numel() and (
        int(t_subdomain.min()) < 0 or int(t_subdomain.max()) >= schedule.timesteps
    ):
        raise ValueError(
            f"t_subdomain values must lie in [0, {schedule.timesteps - 1}]"
        )
    return t_subdomain


def _boundary_epsilon_loss(
    *,
    model,
    branch: torch.Tensor,
    t_subdomain: torch.Tensor,
    schedule: DiffusionSchedule,
    branch_mask: Optional[torch.Tensor],
    boundary_noise: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Denoise known boundary/interface values at their exact branch coordinates."""

    if branch.ndim != 3 or branch.shape[-1] <= max(_BRANCH_KNOWN_MASKS):
        raise ValueError(
            "branch must have shape (B,M,C) with the canonical 11-channel schema"
        )

    batch_size, n_branch_points, _ = branch.shape
    valid = torch.ones(
        (batch_size, n_branch_points),
        dtype=torch.bool,
        device=branch.device,
    )
    if branch_mask is not None:
        if branch_mask.shape != (batch_size, n_branch_points):
            raise ValueError(
                f"branch_mask shape {tuple(branch_mask.shape)} does not match "
                f"branch shape {(batch_size, n_branch_points)}"
            )
        valid &= branch_mask.to(device=branch.device, dtype=torch.bool)

    known = branch[..., list(_BRANCH_KNOWN_MASKS)] > 0.5
    selected = valid & known.any(dim=-1)
    if not selected.any():
        zero = branch.sum() * 0.0
        return zero, {
            "boundary_query": branch.new_empty((0, 2)),
            "boundary_target": branch.new_empty((0, 3)),
            "boundary_noise": branch.new_empty((0, 3)),
            "boundary_pred_noise": branch.new_empty((0, 3)),
            "boundary_known_mask": torch.empty(
                (0, 3), dtype=torch.bool, device=branch.device
            ),
            "boundary_batch_id": torch.empty(
                (0,), dtype=torch.long, device=branch.device
            ),
            "boundary_t_query": torch.empty(
                (0,), dtype=torch.long, device=branch.device
            ),
        }

    all_batch_ids = torch.arange(batch_size, device=branch.device).view(-1, 1)
    all_batch_ids = all_batch_ids.expand(batch_size, n_branch_points)

    boundary_query = branch[..., list(_BRANCH_XY)][selected]
    boundary_target = branch[..., list(_BRANCH_BOUNDARY_VALUES)][selected]
    boundary_known_mask = known[selected]
    boundary_batch_id = all_batch_ids[selected]
    boundary_t_query = t_subdomain[boundary_batch_id]

    if boundary_noise is None:
        boundary_noise_selected = torch.randn_like(boundary_target)
    else:
        if boundary_noise.shape == branch[..., list(_BRANCH_BOUNDARY_VALUES)].shape:
            boundary_noise_selected = boundary_noise.to(
                device=branch.device,
                dtype=branch.dtype,
            )[selected]
        elif boundary_noise.shape == boundary_target.shape:
            boundary_noise_selected = boundary_noise.to(
                device=branch.device,
                dtype=branch.dtype,
            )
        else:
            raise ValueError(
                "boundary_noise must have shape (B,M,3) or the selected boundary "
                f"shape {tuple(boundary_target.shape)}, got {tuple(boundary_noise.shape)}"
            )

    noisy_boundary, boundary_noise_selected = q_sample_points(
        boundary_target,
        boundary_t_query,
        schedule=schedule,
        noise=boundary_noise_selected,
    )
    predicted_boundary_noise = model(
        branch=branch,
        query=boundary_query,
        noisy_target=noisy_boundary,
        t_query=boundary_t_query,
        query_batch_id=boundary_batch_id,
        branch_mask=branch_mask,
    )

    squared = (predicted_boundary_noise - boundary_noise_selected).square()
    squared = squared * boundary_known_mask.to(dtype=squared.dtype)
    known_counts_per_point = boundary_known_mask.sum(dim=1).to(dtype=squared.dtype)
    per_point = squared.sum(dim=1) / known_counts_per_point.clamp_min(1.0)

    sums = per_point.new_zeros(batch_size)
    counts = per_point.new_zeros(batch_size)
    sums.index_add_(0, boundary_batch_id, per_point)
    counts.index_add_(0, boundary_batch_id, torch.ones_like(per_point))
    valid_subdomain = counts > 0
    if not valid_subdomain.all():
        raise ValueError("Every subdomain must contain at least one known boundary value")
    loss = (sums / counts.clamp_min(1.0)).mean()

    return loss, {
        "boundary_query": boundary_query,
        "boundary_target": boundary_target,
        "boundary_noise": boundary_noise_selected,
        "boundary_pred_noise": predicted_boundary_noise,
        "boundary_known_mask": boundary_known_mask,
        "boundary_batch_id": boundary_batch_id,
        "boundary_t_query": boundary_t_query,
    }


def epsilon_prediction_loss(
    model,
    branch: torch.Tensor,
    query: torch.Tensor,
    target: torch.Tensor,
    query_batch_id: torch.Tensor,
    schedule: DiffusionSchedule,
    branch_mask: Optional[torch.Tensor] = None,
    *,
    t_subdomain: Optional[torch.Tensor] = None,
    target_noise: Optional[torch.Tensor] = None,
    boundary_loss_weight: float = 0.0,
    boundary_noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute subdomain-level epsilon loss with optional soft BC supervision.

    One timestep is sampled per subdomain and shared by every query point in that
    subdomain. Gaussian noise remains independent per point and per output field.
    The interior loss is averaged within each subdomain before subdomains are
    averaged equally. When ``boundary_loss_weight`` is positive, the same
    timestep is used to denoise known boundary/interface values at their exact
    branch coordinates.

    ``t_subdomain``, ``target_noise``, and ``boundary_noise`` may be provided by
    deterministic validation code. Training normally leaves them as ``None``.
    """

    if branch.ndim != 3:
        raise ValueError(f"branch must have shape (B,M,C), got {tuple(branch.shape)}")
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError("query and target must both be two-dimensional")
    if query.shape[0] != target.shape[0]:
        raise ValueError("query and target must contain the same number of points")
    if target.shape[1] != 3:
        raise ValueError("target must contain exactly pressure, u, and v")
    if boundary_loss_weight < 0.0:
        raise ValueError("boundary_loss_weight must be non-negative")

    batch_size = int(branch.shape[0])
    query_batch_id = _validate_query_batch_id(
        query_batch_id.to(device=target.device),
        n_queries=int(target.shape[0]),
        batch_size=batch_size,
    )
    t_subdomain = _resolve_subdomain_timesteps(
        batch_size=batch_size,
        schedule=schedule,
        device=target.device,
        t_subdomain=t_subdomain,
    )
    t_query = t_subdomain[query_batch_id]

    if target_noise is None:
        target_noise = torch.randn_like(target)
    else:
        if target_noise.shape != target.shape:
            raise ValueError(
                f"target_noise shape {tuple(target_noise.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        target_noise = target_noise.to(device=target.device, dtype=target.dtype)

    noisy_target, target_noise = q_sample_points(
        target,
        t_query,
        schedule=schedule,
        noise=target_noise,
    )
    predicted_noise = model(
        branch=branch,
        query=query,
        noisy_target=noisy_target,
        t_query=t_query,
        query_batch_id=query_batch_id,
        branch_mask=branch_mask,
    )
    interior_loss = _balanced_subdomain_mse(
        predicted_noise,
        target_noise,
        query_batch_id,
        batch_size,
    )

    boundary_loss = interior_loss.new_zeros(())
    boundary_details: dict[str, torch.Tensor] = {}
    if boundary_loss_weight > 0.0:
        boundary_loss, boundary_details = _boundary_epsilon_loss(
            model=model,
            branch=branch,
            t_subdomain=t_subdomain,
            schedule=schedule,
            branch_mask=branch_mask,
            boundary_noise=boundary_noise,
        )

    total_loss = interior_loss + float(boundary_loss_weight) * boundary_loss
    details: dict[str, Any] = {
        "pred_noise": predicted_noise,
        "noise": target_noise,
        "noisy_target": noisy_target,
        "t_query": t_query,
        "t_subdomain": t_subdomain,
        "interior_loss": interior_loss,
        "boundary_loss": boundary_loss,
        "boundary_loss_weight": float(boundary_loss_weight),
    }
    details.update(boundary_details)
    return total_loss, details


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
    """Build the round-and-reverse DDIM timestep schedule."""

    num_sampling_steps = _validate_integer(num_sampling_steps, "num_sampling_steps")
    total_diffusion_steps = _validate_integer(
        total_diffusion_steps, "total_diffusion_steps"
    )
    if num_sampling_steps < 1:
        raise ValueError("num_sampling_steps must be at least one")
    if total_diffusion_steps < 1:
        raise ValueError("total_diffusion_steps must be at least one")
    if num_sampling_steps > total_diffusion_steps:
        raise ValueError("num_sampling_steps cannot exceed total_diffusion_steps")
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
        raise ValueError(f"Expected descending timesteps, got {t_current} -> {t_next}")

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
        raise ValueError(f"Expected descending timesteps, got {t_start} -> {t_end}")

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
