"""Shared interface-physics objective for DeepONet and Field Diffusion.

This module contains only model-independent physics terms used for unknown
interior-interface optimization. Model-specific code must produce interface
predictions and then call this module.

The objective intentionally mirrors the DeepONet reference implementation:
    L = alpha_traction * L_traction
      + alpha_flux * L_flux
      + alpha_dirichlet * L_dirichlet
      + alpha_smooth * L_smooth
      + alpha_value_l2 * L_value_l2

The primary comparison protocol optimizes pressure/u/v interface traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PHYSICS_OBJECTIVE_PROTOCOL = "shared_interface_physics_deeponet_parity_v1"


@dataclass(frozen=True)
class InterfacePhysicsConfig:
    """Configuration for the shared DeepONet/Diffusion interface objective."""

    viscosity: float = 1.0 / 200.0
    length_unit_scale: float = 1.0e-3

    alpha_traction: float = 1.0
    alpha_flux: float = 1.0
    alpha_dirichlet: float = 1.0

    alpha_p: float = 1.0
    alpha_u: float = 1.0
    alpha_v: float = 1.0

    alpha_smooth: float = 1.0e-4
    alpha_value_l2: float = 0.0

    traction_scale: Optional[float] = None
    flux_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.length_unit_scale <= 0.0:
            raise ValueError("length_unit_scale must be positive")
        if self.traction_scale is not None and self.traction_scale == 0.0:
            raise ValueError("traction_scale must be nonzero")
        if self.flux_scale is not None and self.flux_scale == 0.0:
            raise ValueError("flux_scale must be nonzero")


def _as_output_stat(
    value: torch.Tensor | Sequence[float] | np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if out.numel() == 0:
        raise ValueError("Output statistic must not be empty")
    return out


def field_indices(
    output_channel_names: Sequence[str],
    required: Sequence[str] = ("pressure", "u", "v"),
) -> Dict[str, int]:
    """Return output indices for required physical fields."""
    names = list(output_channel_names)
    out: Dict[str, int] = {}
    for name in required:
        if name not in names:
            raise ValueError(f"Required output field {name!r} is missing from {names}")
        out[name] = names.index(name)
    return out


def subdomain_scales_from_metadata(
    metadata: Optional[Sequence[Mapping[str, object]]],
    n_subdomains: int,
    length_unit_scale: float,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local query-coordinate derivatives to physical derivatives."""
    device = torch.device(device)

    if metadata is None:
        ones = torch.ones(n_subdomains, dtype=torch.float32, device=device)
        return ones, ones.clone()

    if len(metadata) != n_subdomains:
        raise ValueError(
            f"metadata length {len(metadata)} does not match n_subdomains={n_subdomains}"
        )

    x_scales = []
    y_scales = []
    for item in metadata:
        if "x_left_mm" in item and "x_right_mm" in item:
            width = float(item["x_right_mm"]) - float(item["x_left_mm"])
        elif "x_left" in item and "x_right" in item:
            width = float(item["x_right"]) - float(item["x_left"])
        else:
            width = 1.0

        if "reference_length_mm" in item:
            reference_length = float(item["reference_length_mm"])
        elif "reference_length" in item:
            reference_length = float(item["reference_length"])
        else:
            reference_length = 1.0

        x_scales.append(max(width * float(length_unit_scale), 1.0e-12))
        y_scales.append(max(reference_length * float(length_unit_scale), 1.0e-12))

    return (
        torch.as_tensor(x_scales, dtype=torch.float32, device=device),
        torch.as_tensor(y_scales, dtype=torch.float32, device=device),
    )


def line_weights_from_query_y(
    q_edge: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    """Trapezoid weights for a vertical interface line integral."""
    if q_edge.ndim != 3 or q_edge.shape[-1] != 2:
        raise ValueError(f"q_edge must have shape (S, Ny, 2), got {tuple(q_edge.shape)}")
    if y_scale.ndim != 1 or y_scale.shape[0] != q_edge.shape[0]:
        raise ValueError(
            f"y_scale must have shape ({q_edge.shape[0]},), got {tuple(y_scale.shape)}"
        )

    y = q_edge[..., 1]
    if y.shape[1] < 2:
        return torch.ones_like(y)

    dy = torch.abs(y[:, 1:] - y[:, :-1]) * y_scale[:, None]
    weights = torch.zeros_like(y)
    weights[:, 0] = 0.5 * dy[:, 0]
    weights[:, -1] = 0.5 * dy[:, -1]
    if y.shape[1] > 2:
        weights[:, 1:-1] = 0.5 * (dy[:, :-1] + dy[:, 1:])
    return weights


def mean_sq_weighted(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Match the weighted mean-square convention in the DeepONet reference."""
    w = weights
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return torch.sum(w * x * x) / torch.clamp(torch.sum(w), min=1.0e-12)


def smoothness_penalty_z(z_norm: torch.Tensor) -> torch.Tensor:
    """Second-difference regularization along each interface profile."""
    if z_norm.ndim != 3:
        raise ValueError(
            f"z_norm must have shape (n_interfaces, Ny, n_fields), got {tuple(z_norm.shape)}"
        )
    if z_norm.shape[1] < 3:
        return z_norm.new_tensor(0.0)

    d2 = z_norm[:, 2:, :] - 2.0 * z_norm[:, 1:-1, :] + z_norm[:, :-2, :]
    return torch.mean(d2 * d2)


def decode_outputs(
    out_norm: torch.Tensor,
    output_mean: torch.Tensor | Sequence[float] | np.ndarray,
    output_std: torch.Tensor | Sequence[float] | np.ndarray,
) -> torch.Tensor:
    """Decode normalized outputs to physical units."""
    mean = _as_output_stat(output_mean, device=out_norm.device, dtype=out_norm.dtype)
    std = _as_output_stat(output_std, device=out_norm.device, dtype=out_norm.dtype)
    std = torch.clamp(std, min=1.0e-12)

    if out_norm.shape[-1] != mean.numel() or mean.numel() != std.numel():
        raise ValueError(
            "Output channel mismatch: "
            f"out={out_norm.shape[-1]}, mean={mean.numel()}, std={std.numel()}"
        )
    return out_norm * std + mean


def scalar_grad(
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Differentiate a scalar field at every query point with graph retention."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]


def traction_from_output_norm(
    out_norm: torch.Tensor,
    query_local: torch.Tensor,
    normal: Tuple[float, float],
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    output_mean: torch.Tensor | Sequence[float] | np.ndarray,
    output_std: torch.Tensor | Sequence[float] | np.ndarray,
    output_channel_names: Sequence[str],
    viscosity: float,
    pressure_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute traction t = sigma n from decoded p/u/v and physical gradients."""
    if not query_local.requires_grad:
        raise ValueError("query_local must have requires_grad=True for traction evaluation")

    idx = field_indices(output_channel_names, ("pressure", "u", "v"))
    out_phys = decode_outputs(out_norm, output_mean, output_std)

    pressure = out_phys[..., idx["pressure"]]
    velocity_u = out_phys[..., idx["u"]]
    velocity_v = out_phys[..., idx["v"]]

    if pressure_offset is not None:
        if pressure_offset.ndim != 1 or pressure_offset.shape[0] != out_norm.shape[0]:
            raise ValueError(
                f"pressure_offset must have shape ({out_norm.shape[0]},), "
                f"got {tuple(pressure_offset.shape)}"
            )
        pressure = pressure + pressure_offset[:, None]

    grad_u = scalar_grad(velocity_u, query_local, retain_graph=True)
    grad_v = scalar_grad(velocity_v, query_local, retain_graph=True)

    u_x = grad_u[..., 0] / x_scale[:, None]
    u_y = grad_u[..., 1] / y_scale[:, None]
    v_x = grad_v[..., 0] / x_scale[:, None]
    v_y = grad_v[..., 1] / y_scale[:, None]

    mu = float(viscosity)
    sigma_xx = -pressure + 2.0 * mu * u_x
    sigma_xy = mu * (u_y + v_x)
    sigma_yy = -pressure + 2.0 * mu * v_y

    n_x, n_y = float(normal[0]), float(normal[1])
    traction_x = sigma_xx * n_x + sigma_xy * n_y
    traction_y = sigma_xy * n_x + sigma_yy * n_y
    return torch.stack([traction_x, traction_y], dim=-1)


def edge_flux_from_output_norm(
    out_norm: torch.Tensor,
    normal: Tuple[float, float],
    weights: torch.Tensor,
    output_mean: torch.Tensor | Sequence[float] | np.ndarray,
    output_std: torch.Tensor | Sequence[float] | np.ndarray,
    output_channel_names: Sequence[str],
) -> torch.Tensor:
    """Integrate normal velocity over each vertical edge."""
    idx = field_indices(output_channel_names, ("u", "v"))
    out_phys = decode_outputs(out_norm, output_mean, output_std)

    velocity_u = out_phys[..., idx["u"]]
    velocity_v = out_phys[..., idx["v"]]
    normal_velocity = velocity_u * float(normal[0]) + velocity_v * float(normal[1])
    return torch.sum(normal_velocity * weights, dim=1)


def auto_loss_scales(
    output_std: torch.Tensor | Sequence[float] | np.ndarray,
    output_channel_names: Sequence[str],
    q_right: torch.Tensor,
    y_scale: torch.Tensor,
    config: InterfacePhysicsConfig,
) -> Tuple[float, float]:
    """Match the automatic traction/flux scaling used by the DeepONet reference."""
    std = np.asarray(
        torch.as_tensor(output_std, dtype=torch.float64).detach().cpu().numpy()
    ).reshape(-1)
    names = list(output_channel_names)

    pressure_scale = float(std[names.index("pressure")]) if "pressure" in names else 1.0
    velocity_scales = [
        float(std[names.index(field_name)])
        for field_name in ("u", "v")
        if field_name in names
    ]
    velocity_scale = max(max(velocity_scales) if velocity_scales else 1.0, 1.0e-12)

    if config.traction_scale is None:
        traction_scale = max(abs(pressure_scale), 1.0e-12)
    else:
        traction_scale = max(abs(float(config.traction_scale)), 1.0e-12)

    if config.flux_scale is None:
        with torch.no_grad():
            reference_weights = line_weights_from_query_y(q_right, y_scale)
            reference_length = float(
                torch.mean(torch.sum(reference_weights, dim=1)).detach().cpu()
            )
        flux_scale = max(abs(velocity_scale * reference_length), 1.0e-12)
    else:
        flux_scale = max(abs(float(config.flux_scale)), 1.0e-12)

    return traction_scale, flux_scale


def interface_physics_loss_torch(
    *,
    out_left_norm: torch.Tensor,
    out_right_norm: torch.Tensor,
    z_norm: torch.Tensor,
    q_left_base: torch.Tensor,
    q_right_base: torch.Tensor,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    output_mean: torch.Tensor | Sequence[float] | np.ndarray,
    output_std: torch.Tensor | Sequence[float] | np.ndarray,
    output_channel_names: Sequence[str],
    optimized_output_channels: Sequence[int],
    config: InterfacePhysicsConfig,
    pressure_offsets: Optional[torch.Tensor] = None,
    traction_scale: Optional[float] = None,
    flux_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the shared DeepONet/Diffusion unknown-interface physics loss."""
    if out_left_norm.shape != out_right_norm.shape:
        raise ValueError(
            f"Left/right output shapes differ: "
            f"{tuple(out_left_norm.shape)} vs {tuple(out_right_norm.shape)}"
        )
    if out_left_norm.ndim != 3:
        raise ValueError(
            f"Interface outputs must have shape (S, Ny, C), got {tuple(out_left_norm.shape)}"
        )

    n_subdomains, n_points, _ = out_left_norm.shape
    expected_z_shape = (max(n_subdomains - 1, 0), n_points, 3)
    if tuple(z_norm.shape) != expected_z_shape:
        raise ValueError(
            f"z_norm has shape {tuple(z_norm.shape)}, expected {expected_z_shape}"
        )

    optimized_output_channels = tuple(int(v) for v in optimized_output_channels)
    if len(optimized_output_channels) != 3:
        raise ValueError(
            "Primary parity protocol requires exactly three optimized fields: pressure/u/v"
        )

    if traction_scale is None or flux_scale is None:
        auto_traction, auto_flux = auto_loss_scales(
            output_std=output_std,
            output_channel_names=output_channel_names,
            q_right=q_right_base.detach(),
            y_scale=y_scale,
            config=config,
        )
        if traction_scale is None:
            traction_scale = auto_traction
        if flux_scale is None:
            flux_scale = auto_flux

    traction_scale = max(abs(float(traction_scale)), 1.0e-12)
    flux_scale = max(abs(float(flux_scale)), 1.0e-12)

    if n_subdomains > 1:
        opt_out = torch.as_tensor(
            optimized_output_channels,
            dtype=torch.long,
            device=out_left_norm.device,
        )
        field_weights = torch.tensor(
            [config.alpha_p, config.alpha_u, config.alpha_v],
            dtype=out_left_norm.dtype,
            device=out_left_norm.device,
        )

        pred_right = out_right_norm[:-1, :, :][:, :, opt_out]
        pred_left = out_left_norm[1:, :, :][:, :, opt_out]

        loss_dirichlet = torch.mean((pred_right - z_norm) ** 2 * field_weights)
        loss_dirichlet = loss_dirichlet + torch.mean(
            (pred_left - z_norm) ** 2 * field_weights
        )
        loss_dirichlet = loss_dirichlet + torch.mean(
            (pred_right - pred_left) ** 2 * field_weights
        )
    else:
        loss_dirichlet = z_norm.new_tensor(0.0)

    if pressure_offsets is not None:
        if pressure_offsets.ndim != 1 or pressure_offsets.shape[0] != n_subdomains:
            raise ValueError(
                f"pressure_offsets must have shape ({n_subdomains},), "
                f"got {tuple(pressure_offsets.shape)}"
            )

    traction_left = traction_from_output_norm(
        out_norm=out_left_norm,
        query_local=q_left_base,
        normal=(-1.0, 0.0),
        x_scale=x_scale,
        y_scale=y_scale,
        output_mean=output_mean,
        output_std=output_std,
        output_channel_names=output_channel_names,
        viscosity=config.viscosity,
        pressure_offset=pressure_offsets,
    )
    traction_right = traction_from_output_norm(
        out_norm=out_right_norm,
        query_local=q_right_base,
        normal=(1.0, 0.0),
        x_scale=x_scale,
        y_scale=y_scale,
        output_mean=output_mean,
        output_std=output_std,
        output_channel_names=output_channel_names,
        viscosity=config.viscosity,
        pressure_offset=pressure_offsets,
    )

    weights_right = line_weights_from_query_y(q_right_base.detach(), y_scale)
    if n_subdomains > 1:
        traction_residual = traction_right[:-1] + traction_left[1:]
        loss_traction = mean_sq_weighted(
            traction_residual / traction_scale,
            weights_right[:-1],
        )
    else:
        traction_residual = z_norm.new_zeros((0, n_points, 2))
        loss_traction = z_norm.new_tensor(0.0)

    weights_left = line_weights_from_query_y(q_left_base.detach(), y_scale)
    flux_left = edge_flux_from_output_norm(
        out_norm=out_left_norm,
        normal=(-1.0, 0.0),
        weights=weights_left,
        output_mean=output_mean,
        output_std=output_std,
        output_channel_names=output_channel_names,
    )
    flux_right = edge_flux_from_output_norm(
        out_norm=out_right_norm,
        normal=(1.0, 0.0),
        weights=weights_right,
        output_mean=output_mean,
        output_std=output_std,
        output_channel_names=output_channel_names,
    )

    flux_residual = flux_left + flux_right
    global_flux_residual = flux_left[0] + flux_right[-1]
    loss_flux = torch.mean((flux_residual / flux_scale) ** 2)
    loss_flux = loss_flux + torch.mean((global_flux_residual / flux_scale) ** 2)

    loss_smooth = smoothness_penalty_z(z_norm)
    loss_value_l2 = (
        torch.mean(z_norm * z_norm)
        if z_norm.numel()
        else z_norm.new_tensor(0.0)
    )

    total_loss = (
        config.alpha_traction * loss_traction
        + config.alpha_flux * loss_flux
        + config.alpha_dirichlet * loss_dirichlet
        + config.alpha_smooth * loss_smooth
        + config.alpha_value_l2 * loss_value_l2
    )

    info = {
        "loss": total_loss.detach(),
        "traction": loss_traction.detach(),
        "flux": loss_flux.detach(),
        "dirichlet": loss_dirichlet.detach(),
        "smooth": loss_smooth.detach(),
        "value_l2": loss_value_l2.detach(),
        "max_abs_traction_res": (
            torch.max(torch.abs(traction_residual)).detach()
            if traction_residual.numel()
            else z_norm.new_tensor(0.0)
        ),
        "max_abs_flux_res": (
            torch.max(torch.abs(flux_residual)).detach()
            if flux_residual.numel()
            else z_norm.new_tensor(0.0)
        ),
    }
    return total_loss, info
