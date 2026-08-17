"""
Evaluation helpers for the non-grid Fluent DeepONet dataset.

This version is for the cell-center dataset produced by
`deeponet_fluent_dataset.build_fluent_deeponet_cell_dataset`, where each sample has:

    sample["branch"] : boundary/interface point set, shape (M, Cb)
    sample["query"]  : original Fluent cell-center points, shape (Q_i, 2)
    sample["target"] : original Fluent cell fields, shape (Q_i, 4)

Interior interfaces can be treated as unknown at inference time. The iterative
solver initializes interior interface p/T/u/v, predicts all subdomains in
parallel, averages the two predicted traces on each shared interface, and
repeats until all interface mismatches are below tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, List

import numpy as np
import torch
from scipy.fft import dct, idct
from torch.autograd import grad

from fluent_deeponet import DeepONet, FeatureNormalizer, normalize_cell_branch_with_y, predict_deeponet_points
from plot import local_query_to_physical

Array = np.ndarray
PathLike = Union[str, Path]

@dataclass
class EdgeLayout:
    left: Array
    right: Array
    bottom: Array
    top: Array
    value_channels: Array
    known_channels: Optional[Array]
    x_channel: int
    y_channel: int
    wall_channel: int
    interface_channel: int


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_base_deeponet_checkpoint(
    checkpoint_path: PathLike,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[DeepONet, FeatureNormalizer, Dict[str, object]]:
    """Load a base DeepONet checkpoint from the updated notebook."""
    device = torch.device(device)
    ckpt = torch.load(_as_path(checkpoint_path), map_location=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain model_config and model_state_dict")
    if "y_normalizer" not in ckpt:
        raise KeyError("Checkpoint must contain y_normalizer")

    model = DeepONet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    y_normalizer = FeatureNormalizer.from_state_dict(ckpt["y_normalizer"]).to(device)
    return model, y_normalizer, dict(ckpt)


def select_sample_indices_by_case_id(
    dataset: Mapping[str, object],
    case_id: str,
    realization_id: Optional[int] = None,
) -> Array:
    """Return sample indices for one AR sorted by subdomain_id."""
    metadata = list(dataset["metadata"])
    idx = [i for i, m in enumerate(metadata) if m["case_id"] == case_id]
    if realization_id is not None:
        idx = [i for i in idx if int(metadata[i].get("realization_id", 0)) == int(realization_id)]
    if not idx:
        raise ValueError(f"No samples found for case {case_id}")
    idx = np.asarray(idx, dtype=np.int64)
    sub = np.asarray([metadata[i]["subdomain_id"] for i in idx], dtype=np.int64)
    return idx[np.argsort(sub, kind="stable")]


def infer_edge_layout(branch_template: Array, branch_channel_names: Sequence[str], tol: float = 1.0e-6) -> EdgeLayout:
    """Infer side sensor indices from the branch coordinates and masks."""
    branch = np.asarray(branch_template, dtype=np.float32)
    names = list(branch_channel_names)
    x_ch = names.index("x_local")
    y_ch = names.index("y_local")
    wall_ch = names.index("wall_mask")
    interface_ch = names.index("interface_mask")
    # Output fields are inferred from the boundary_<field> channels so this
    # works for any subset/order (e.g. isothermal runs without temperature).
    output_fields = [c[len("boundary_"):] for c in names if c.startswith("boundary_")]
    value_channels = np.asarray([names.index(f"boundary_{f}") for f in output_fields], dtype=np.int64)
    known_channels = None
    known_names = [f"known_{f}" for f in output_fields]
    if all(k in names for k in known_names):
        known_channels = np.asarray([names.index(k) for k in known_names], dtype=np.int64)

    x = branch[:, x_ch]
    y = branch[:, y_ch]
    wall = branch[:, wall_ch]
    interface = branch[:, interface_ch]

    # Interfaces are the vertical sides (min/max x); walls are bottom/top
    # polylines (smaller vs larger y_local, since y_local = y / reference_length).
    interface_pts = interface > 0.5
    wall_pts = wall > 0.5
    x_left_val = float(np.min(x[interface_pts])) if np.any(interface_pts) else 0.0
    x_right_val = float(np.max(x[interface_pts])) if np.any(interface_pts) else 0.0

    left = np.flatnonzero((np.abs(x - x_left_val) <= tol) & interface_pts)
    right = np.flatnonzero((np.abs(x - x_right_val) <= tol) & interface_pts)
    y_wall = y[wall_pts]
    y_split = float(np.median(y_wall)) if y_wall.size else 0.0
    bottom = np.flatnonzero(wall_pts & (y <= y_split))
    top = np.flatnonzero(wall_pts & (y > y_split))

    left = left[np.argsort(y[left], kind="stable")]
    right = right[np.argsort(y[right], kind="stable")]
    bottom = bottom[np.argsort(x[bottom], kind="stable")]
    top = top[np.argsort(x[top], kind="stable")]

    if left.size == 0 or right.size == 0:
        raise ValueError("Could not identify left/right interface sensors")

    return EdgeLayout(
        left=left.astype(np.int64),
        right=right.astype(np.int64),
        bottom=bottom.astype(np.int64),
        top=top.astype(np.int64),
        value_channels=value_channels,
        known_channels=known_channels,
        x_channel=x_ch,
        y_channel=y_ch,
        wall_channel=wall_ch,
        interface_channel=interface_ch,
    )


def _set_edge_profile(branch_one: Array, edge_idx: Array, value_idx: Array, profile: Array) -> None:
    edge_idx = np.asarray(edge_idx, dtype=np.int64)
    value_idx = np.asarray(value_idx, dtype=np.int64)
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (edge_idx.size, value_idx.size):
        raise ValueError(
            f"profile has shape {profile.shape}, expected {(edge_idx.size, value_idx.size)}"
        )
    branch_one[np.ix_(edge_idx, value_idx)] = profile


def _interface_value_profiles_from_branch(branch_inputs: Array, layout: EdgeLayout) -> Tuple[Array, Array]:
    """Extract left/right interface p/T/u/v profiles for all subdomains."""
    branch_arr = np.asarray(branch_inputs, dtype=np.float32)
    left_vals = branch_arr[:, layout.left, :][:, :, layout.value_channels]
    right_vals = branch_arr[:, layout.right, :][:, :, layout.value_channels]
    return left_vals.astype(np.float32, copy=False), right_vals.astype(np.float32, copy=False)


def _normal_init_stats(
    branch_inputs_for_ar: Array,
    layout: EdgeLayout,
    y_normalizer: Optional[FeatureNormalizer],
    init_std_scale: float,
) -> Tuple[Array, Array]:
    """Return physical-space mean/std for p,T,u,v initialization."""
    value_dim = len(layout.value_channels)
    if y_normalizer is not None:
        mean = y_normalizer.mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
        std = y_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if mean.size != value_dim or std.size != value_dim:
            raise ValueError(
                f"y_normalizer has {mean.size} channels but interface has {value_dim} value channels"
            )
    else:
        vals = np.asarray(branch_inputs_for_ar[..., layout.value_channels], dtype=np.float32)
        if layout.known_channels is not None:
            known = np.asarray(branch_inputs_for_ar[..., layout.known_channels], dtype=np.float32)
            mask = np.all(known > 0.5, axis=-1)
            if np.any(mask):
                vals = vals[mask]
            else:
                vals = vals.reshape(-1, value_dim)
        else:
            vals = vals.reshape(-1, value_dim)
        mean = vals.mean(axis=0).astype(np.float32)
        std = vals.std(axis=0).astype(np.float32)

    std = np.maximum(std * float(init_std_scale), 1.0e-12).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32)


def _init_value_vector(
    init_value: Optional[Union[float, Sequence[float], Array]],
    fallback: Array,
    value_dim: int,
) -> Array:
    if init_value is None:
        return np.asarray(fallback, dtype=np.float32).reshape(value_dim)
    arr = np.asarray(init_value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(value_dim, float(arr), dtype=np.float32)
    arr = arr.reshape(-1).astype(np.float32)
    if arr.size != value_dim:
        raise ValueError(f"init_value must be scalar or length {value_dim}, got shape {arr.shape}")
    return arr


def _find_wall_corner_value(
    branch_one: Array,
    layout: EdgeLayout,
    x_value: float,
    y_value: float,
    tol: float = 1.0e-6,
) -> Optional[Array]:
    x = branch_one[:, layout.x_channel]
    y = branch_one[:, layout.y_channel]
    wall = branch_one[:, layout.wall_channel]
    candidates = np.flatnonzero(
        (np.abs(x - float(x_value)) <= tol)
        & (np.abs(y - float(y_value)) <= tol)
        & (wall > 0.5)
    )
    if candidates.size == 0:
        return None
    return branch_one[int(candidates[0]), layout.value_channels].astype(np.float32)


def _preserve_wall_corner_values(
    profile: Array,
    branch_one: Array,
    edge_name: str,
    layout: EdgeLayout,
    tol: float = 1.0e-6,
) -> Array:
    """If vertical-edge endpoints coincide with wall corners, keep wall-corner values."""
    edge_name = str(edge_name).lower()
    if edge_name not in {"left", "right"}:
        return np.asarray(profile, dtype=np.float32)

    out = np.asarray(profile, dtype=np.float32).copy()
    edge_idx = layout.left if edge_name == "left" else layout.right
    if out.shape[0] < 2 or edge_idx.size < 2:
        return out

    edge_x = 0.0 if edge_name == "left" else 1.0
    edge_y = branch_one[edge_idx, layout.y_channel]

    # In the cell-center dataset, interface sensors usually exclude y=0 and y=1.
    # In that common case, there is no corner to preserve, so leave the profile unchanged.
    if abs(float(edge_y[0]) - 0.0) <= tol:
        corner = _find_wall_corner_value(branch_one, layout, edge_x, 0.0, tol=tol)
        if corner is not None:
            out[0] = corner
    if abs(float(edge_y[-1]) - 1.0) <= tol:
        corner = _find_wall_corner_value(branch_one, layout, edge_x, 1.0, tol=tol)
        if corner is not None:
            out[-1] = corner
    return out


def _sample_initial_profile(
    rng: np.random.Generator,
    mode: str,
    y_values: Array,
    mean: Array,
    std: Array,
    init_value: Optional[Union[float, Sequence[float], Array]],
    init_noise_std: float,
    gaussian_center_y: float,
    gaussian_var_y: float,
) -> Array:
    y_values = np.asarray(y_values, dtype=np.float32).reshape(-1)
    value_dim = int(mean.size)
    n_points = int(y_values.size)
    mode = str(mode).lower()

    if mode == "pointwise_random":
        return rng.normal(
            loc=mean.reshape(1, value_dim),
            scale=std.reshape(1, value_dim),
            size=(n_points, value_dim),
        ).astype(np.float32)

    if mode == "fixed_with_noise":
        base = _init_value_vector(init_value, fallback=mean, value_dim=value_dim)
        noise = rng.normal(
            loc=0.0,
            scale=float(init_noise_std),
            size=(n_points, value_dim),
        ).astype(np.float32)
        return (base.reshape(1, value_dim) + noise).astype(np.float32)

    if mode == "interfacewise_random":
        one_value = rng.normal(loc=mean, scale=std, size=(value_dim,)).astype(np.float32)
        return np.repeat(one_value.reshape(1, value_dim), n_points, axis=0)

    if mode == "gaussian_along_y":
        base = _init_value_vector(init_value, fallback=mean, value_dim=value_dim)
        # Smooth perturbation with zero mean over the sampled interface points.
        g = np.exp(
            -0.5 * ((y_values - float(gaussian_center_y)) ** 2) / float(gaussian_var_y)
        ).astype(np.float32)
        g = g - float(g.mean())
        amplitude = std.reshape(1, value_dim)
        profile = base.reshape(1, value_dim) + g.reshape(n_points, 1) * amplitude
        if init_noise_std > 0.0:
            profile = profile + rng.normal(
                loc=0.0,
                scale=float(init_noise_std),
                size=profile.shape,
            ).astype(np.float32)
        return profile.astype(np.float32)

    raise ValueError(f"Unsupported initialization mode: {mode}")


def initialize_unknown_interior_interfaces(
    branch_inputs_for_ar: Array,
    layout: EdgeLayout,
    y_normalizer: Optional[FeatureNormalizer] = None,
    random_seed: int = 0,
    init_mode: str = "pointwise_random",
    init_std_scale: float = 1.0,
    init_value: Optional[Union[float, Sequence[float], Array]] = None,
    init_noise_std: float = 1.0e-3,
    init_gaussian_center_y: float = 0.5,
    init_gaussian_var_y: float = 0.02,
    preserve_wall_corners: bool = False,
) -> Array:
    """Initialize both sides of every interior interface; exterior inlet/outlet are preserved."""
    branch = np.asarray(branch_inputs_for_ar, dtype=np.float32).copy()
    if branch.ndim != 3:
        raise ValueError(f"branch_inputs_for_ar must have shape (S,M,C), got {branch.shape}")

    init_mode = str(init_mode).lower()
    aliases = {
        "pointwise_normal": "pointwise_random",
        "pointwise_random_normal": "pointwise_random",
        "pointwise_fixed": "fixed_with_noise",
        "fixed": "fixed_with_noise",
        "fixed_value": "fixed_with_noise",
        "interface_wise_random": "interfacewise_random",
        "interfacewise_normal": "interfacewise_random",
        "interface_wise_normal": "interfacewise_random",
        "truth": "dataset_truth",
    }
    init_mode = aliases.get(init_mode, init_mode)
    if init_mode == "dataset_truth":
        return branch

    allowed = {"pointwise_random", "fixed_with_noise", "interfacewise_random", "gaussian_along_y"}
    if init_mode not in allowed:
        raise ValueError(
            "init_mode must be one of 'pointwise_random', 'fixed_with_noise', "
            "'interfacewise_random', 'gaussian_along_y', or 'dataset_truth'."
        )

    rng = np.random.default_rng(int(random_seed))
    n_sub = branch.shape[0]
    mean, std = _normal_init_stats(branch, layout, y_normalizer, init_std_scale)

    for interface_id in range(1, n_sub):
        right_y = branch[interface_id - 1, layout.right, layout.y_channel]
        left_y = branch[interface_id, layout.left, layout.y_channel]

        right_profile = _sample_initial_profile(
            rng=rng,
            mode=init_mode,
            y_values=right_y,
            mean=mean,
            std=std,
            init_value=init_value,
            init_noise_std=init_noise_std,
            gaussian_center_y=init_gaussian_center_y,
            gaussian_var_y=init_gaussian_var_y,
        )
        left_profile = _sample_initial_profile(
            rng=rng,
            mode=init_mode,
            y_values=left_y,
            mean=mean,
            std=std,
            init_value=init_value,
            init_noise_std=init_noise_std,
            gaussian_center_y=init_gaussian_center_y,
            gaussian_var_y=init_gaussian_var_y,
        )

        if preserve_wall_corners:
            right_profile = _preserve_wall_corner_values(
                right_profile, branch[interface_id - 1], "right", layout
            )
            left_profile = _preserve_wall_corner_values(
                left_profile, branch[interface_id], "left", layout
            )

        _set_edge_profile(branch[interface_id - 1], layout.right, layout.value_channels, right_profile)
        _set_edge_profile(branch[interface_id], layout.left, layout.value_channels, left_profile)

        if layout.known_channels is not None:
            branch[interface_id - 1][np.ix_(layout.right, layout.known_channels)] = 1.0
            branch[interface_id][np.ix_(layout.left, layout.known_channels)] = 1.0

    return branch.astype(np.float32, copy=False)


@torch.no_grad()
def predict_edge_profiles(
    model: DeepONet,
    branch_inputs: Array,
    layout: EdgeLayout,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
) -> Tuple[Array, Array]:
    """Predict left and right interface profiles for all subdomains in physical units."""
    model.eval()
    device = torch.device(device)
    y_normalizer = y_normalizer.to(device)
    branch_raw = np.asarray(branch_inputs, dtype=np.float32)
    n_sub = branch_raw.shape[0]

    query_channels = _query_channel_indices(model, branch_channel_names, layout)
    left_q = branch_raw[:, layout.left, :][:, :, query_channels]
    right_q = branch_raw[:, layout.right, :][:, :, query_channels]

    branch_norm = np.stack([
        normalize_cell_branch_with_y(
            b,
            branch_channel_names=branch_channel_names,
            target_y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
        )
        for b in branch_raw
    ], axis=0)

    b = torch.as_tensor(branch_norm, dtype=torch.float32, device=device)
    q_left = torch.as_tensor(left_q, dtype=torch.float32, device=device)
    q_right = torch.as_tensor(right_q, dtype=torch.float32, device=device)
    pred_left = model(b, q_left).detach().cpu().numpy().astype(np.float32)
    pred_right = model(b, q_right).detach().cpu().numpy().astype(np.float32)
    if pred_left.shape[0] != n_sub or pred_right.shape[0] != n_sub:
        raise RuntimeError("Unexpected interface prediction shape")
    return pred_left, pred_right


def _edge_grad_from_queries(
    model: DeepONet,
    branch_batch: torch.Tensor,
    query: torch.Tensor,
) -> Array:
    """Return pointwise dp/dx at each edge query location via autograd w.r.t. x and y."""
    query = query.detach().requires_grad_(True)
    pred = model(branch_batch, query)

    n_batch, n_points, n_ch = pred.shape
    grad = torch.empty((n_batch, n_points, n_ch), dtype=torch.float32, device=pred.device)
    for ch in range(n_ch):
        grad_ch = torch.autograd.grad(
            pred[..., ch],
            query,
            torch.ones([n_batch, n_points]).to(pred.device),
            retain_graph=ch < (n_ch - 1),
        )[0]
        grad[..., ch] = grad_ch[..., 0]
    return grad.detach().cpu().numpy().astype(np.float32)


def predict_edge_grad(
    model: DeepONet,
    branch_inputs: Array,
    layout: EdgeLayout,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
) -> Tuple[Array, Array]:
    """Predict dp/dx on left and right interface profiles."""
    model.eval()
    device = torch.device(device)
    y_normalizer = y_normalizer.to(device)
    branch_raw = np.asarray(branch_inputs, dtype=np.float32)
    n_sub = branch_raw.shape[0]

    query_channels = _query_channel_indices(model, branch_channel_names, layout)
    left_q = branch_raw[:, layout.left, :][:, :, query_channels]
    right_q = branch_raw[:, layout.right, :][:, :, query_channels]

    branch_norm = np.stack([
        normalize_cell_branch_with_y(
            b,
            branch_channel_names=branch_channel_names,
            target_y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
        )
        for b in branch_raw
    ], axis=0)

    b = torch.as_tensor(branch_norm, dtype=torch.float32, device=device)
    q_left = torch.as_tensor(left_q, dtype=torch.float32, device=device)
    q_right = torch.as_tensor(right_q, dtype=torch.float32, device=device)

    with torch.enable_grad():
        dpdx_left = _edge_grad_from_queries(
            model=model,
            branch_batch=b,
            query=q_left,
        )
        dpdx_right = _edge_grad_from_queries(
            model=model,
            branch_batch=b,
            query=q_right,
        )

        if dpdx_left.shape[0] != n_sub or dpdx_right.shape[0] != n_sub:
            raise RuntimeError("Unexpected interface gradient prediction shape")
        return dpdx_left, dpdx_right


@torch.no_grad()
def predict_cell_samples_physical(
    model: DeepONet,
    samples: Sequence[Mapping[str, Array]],
    branch_inputs: Array,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    query_batch_size: int = 65536,
) -> List[Array]:
    """Predict all selected variable-length cell-center samples."""
    preds = []
    for sample, branch in zip(samples, branch_inputs):
        b_norm = normalize_cell_branch_with_y(
            branch,
            branch_channel_names=branch_channel_names,
            target_y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
        )
        pred = predict_deeponet_points(
            model=model,
            branch=b_norm,
            query=sample["query"],
            device=device,
            y_normalizer=y_normalizer,
            query_batch_size=query_batch_size,
        )
        preds.append(pred.numpy().astype(np.float32))
    return preds


def compute_interface_mse_from_edges(pred_left: Array, pred_right: Array) -> Tuple[Array, Array]:
    """Compare pred_right[k] with pred_left[k+1]."""
    if pred_left.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, pred_left.shape[-1]), dtype=np.float32)
    diff = pred_right[:-1] - pred_left[1:]
    mse_by_channel = np.mean(diff ** 2, axis=1)
    mse_total = np.mean(diff ** 2, axis=(1, 2))
    return mse_total.astype(np.float32), mse_by_channel.astype(np.float32)
  
def _output_mean_std_np(y_normalizer: FeatureNormalizer) -> Tuple[Array, Array]:
    """Return output mean/std as NumPy arrays with shape (1, 4)."""
    mean = y_normalizer.mean.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    std = y_normalizer.std.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    std = np.maximum(std, 1.0e-12)
    return mean, std


def pack_interior_interface_z(
    branch_inputs: Array,
    layout: EdgeLayout,
    y_normalizer: FeatureNormalizer,
    reduce_mode: Optional[str] = None,
    dimensions: int = 16,
) -> Array:
    """
    Pack all interior interface profiles into normalized output space.

    Returns
    -------
    z:
        Normalized interface state.  Shape is
        ``(n_interfaces, n_interface_points, n_channels)`` when
        ``reduce_mode is None``, or ``(n_interfaces, dimensions, n_channels)``
        when ``reduce_mode == "dct"``.  Interface i corresponds to the shared
        boundary between subdomain i and subdomain i+1.

    Notes
    -----
    The packed value is the average of the right edge of the left subdomain
    and the left edge of the right subdomain, normalized with
    ``y_normalizer`` before optional DCT compression.
    """
    branch = np.asarray(branch_inputs, dtype=np.float32)
    if branch.ndim != 3:
        raise ValueError(f"branch_inputs must have shape (S,M,C), got {branch.shape}")
    n_channels = int(layout.value_channels.size)
    if branch.shape[0] < 2:
        n_packed = int(dimensions) if reduce_mode == "dct" else int(layout.right.size)
        return np.zeros((0, n_packed, n_channels), dtype=np.float32)

    profiles = []
    for interface_id in range(1, branch.shape[0]):
        right_profile = branch[interface_id - 1][np.ix_(layout.right, layout.value_channels)]
        left_profile = branch[interface_id][np.ix_(layout.left, layout.value_channels)]
        if right_profile.shape != left_profile.shape:
            raise ValueError(
                f"Interface {interface_id} has mismatched side shapes: "
                f"right={right_profile.shape}, left={left_profile.shape}"
            )
        profile = 0.5 * (right_profile + left_profile)
        profiles.append(profile.astype(np.float32))

    profiles = np.stack(profiles, axis=0).astype(np.float32)
    mean, std = _output_mean_std_np(y_normalizer)
    if reduce_mode == "dct":
        profile_norm = np.array(dct(profiles, axis=1, norm="ortho"), dtype=np.float32)
        profile_norm = profile_norm[:, :dimensions, :]
    else:
        profile_norm = ((profiles - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    return profile_norm


def apply_interior_interface_z(
    base_branch: Array,
    z: Array,
    layout: EdgeLayout,
    y_normalizer: FeatureNormalizer,
    reduce_mode: Optional[str] = None,
) -> Array:
    """
    Decode normalized interface profiles and write them to both sides of every
    interior interface.  Exterior inlet/outlet/wall values are preserved.
    """
    branch = np.asarray(base_branch, dtype=np.float32).copy()
    z = np.asarray(z, dtype=np.float32)
    if reduce_mode == "dct":
        z = np.array(idct(z, axis=1, norm="ortho", n=int(layout.right.size)), dtype=np.float32)
    expected = (max(branch.shape[0] - 1, 0), int(layout.right.size), int(layout.value_channels.size))
    if z.shape != expected:
        raise ValueError(f"z has shape {z.shape}, expected {expected}")

    mean, std = _output_mean_std_np(y_normalizer)
    for local_i, interface_id in enumerate(range(1, branch.shape[0])):
        profile_phys = z[local_i] * std + mean
        profile_phys = profile_phys.astype(np.float32, copy=False)
        _set_edge_profile(branch[interface_id - 1], layout.right, layout.value_channels, profile_phys)
        _set_edge_profile(branch[interface_id], layout.left, layout.value_channels, profile_phys)
        if layout.known_channels is not None:
            branch[interface_id - 1][np.ix_(layout.right, layout.known_channels)] = 1.0
            branch[interface_id][np.ix_(layout.left, layout.known_channels)] = 1.0
    return branch.astype(np.float32, copy=False)

def make_plot_dict_from_iterative_result(
    result,
    data,
    sample_indices,
    output_channel_names=None,
):
    """
    Convert iterative inference result to the dictionary format expected by
    plot.py::plot_prediction.

    Returns dict with:
        x, y, p, t, u, v, p_pred, t_pred, u_pred, v_pred
    """

    if output_channel_names is None:
        output_channel_names = list(data["output_channel_names"])

    pred_samples = result["pred_samples"]

    if len(pred_samples) != len(sample_indices):
        raise ValueError(
            f"result['pred_samples'] has length {len(pred_samples)}, "
            f"but sample_indices has length {len(sample_indices)}"
        )

    xs = []
    ys = []
    preds = []
    truths = []
    sample_ids = []

    for local_i, sample_i in enumerate(sample_indices):
        sample = data["samples"][sample_i]
        metadata = data["metadata"][sample_i]

        query_local = sample["query"]
        truth = sample["target"]
        pred = pred_samples[local_i]

        if pred.shape != truth.shape:
            raise ValueError(
                f"Shape mismatch for sample {sample_i}: "
                f"pred={pred.shape}, truth={truth.shape}"
            )

        x_phys, y_phys = local_query_to_physical(query_local, metadata)

        xs.append(x_phys)
        ys.append(y_phys)
        preds.append(pred)
        truths.append(truth)
        sample_ids.append(np.full(len(x_phys), sample_i, dtype=np.int64))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(truths, axis=0)
    sample_ids = np.concatenate(sample_ids, axis=0)

    idx = {name: output_channel_names.index(name) for name in output_channel_names}

    p = {
        "x": x,
        "y": y,
        "pred": pred,
        "truth": truth,
        "sample_id": sample_ids,
    }

    return p

# ============================================================================
# Physics-based interface inference: shared Dirichlet trace + traction/flux loss
# ============================================================================

@dataclass
class PhysicsInterfaceConfig:
    """Configuration for physics-based unknown-interface optimization.

    The optimizer keeps the trained DeepONet fixed and updates only the interior
    interface Dirichlet values written in the branch tensor.

    Important unit convention:
        The DeepONet query coordinates are local coordinates.  If metadata is
        supplied, derivatives are converted to derivatives with respect to the
        physical length unit used by metadata, multiplied by length_unit_scale.
        For your current dataset metadata names are in mm, so use
        length_unit_scale=1e-3 to compute SI derivatives in 1/m.
    """

    max_iter: int = 500
    lr: float = 5.0e-2
    optimizer: str = "adam"  # "adam" or "lbfgs"
    tol: float = 1.0e-6
    random_seed: int = 0
    init_mode: str = "fixed_with_noise"
    init_std_scale: float = 1.0
    init_value: Optional[Union[float, Sequence[float]]] = None
    init_noise_std: float = 1.0e-3
    init_gaussian_center_y: float = 0.5
    init_gaussian_var_y: float = 0.02
    preserve_wall_corners: bool = False
    query_batch_size: int = 65536

    # Which interface branch fields are optimized.  None means all fields except
    # temperature if temperature exists, i.e. pressure/u/v for a p,u,v,T output.
    optimize_fields: Optional[Sequence[str]] = None

    # Viscosity in the same physical/nondimensional convention as the decoded
    # model outputs.  If your Fluent data are dimensional, pass dynamic viscosity
    # when pressure is Pa, or kinematic viscosity if pressure is p/rho.
    # If your decoded data are nondimensionalized in the usual way, pass 1/Re.
    viscosity: float = 1.0 / 200.0

    # Multiply metadata lengths by this before computing gradients and line
    # integrals.  For mm metadata and SI velocity/pressure, use 1e-3.
    # For nondimensional coordinates, use 1.0.
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

    optimize_pressure_offsets: bool = False
    verbose: bool = True
    verbose_every: int = 25

    def __post_init__(self) -> None:
        self.max_iter = int(self.max_iter)
        self.lr = float(self.lr)
        self.optimizer = str(self.optimizer).lower()
        self.tol = float(self.tol)
        self.random_seed = int(self.random_seed)
        self.init_mode = str(self.init_mode).lower()
        self.init_std_scale = float(self.init_std_scale)
        self.init_noise_std = float(self.init_noise_std)
        self.init_gaussian_center_y = float(self.init_gaussian_center_y)
        self.init_gaussian_var_y = float(self.init_gaussian_var_y)
        self.query_batch_size = int(self.query_batch_size)
        self.viscosity = float(self.viscosity)
        self.length_unit_scale = float(self.length_unit_scale)
        self.alpha_traction = float(self.alpha_traction)
        self.alpha_flux = float(self.alpha_flux)
        self.alpha_dirichlet = float(self.alpha_dirichlet)
        self.alpha_smooth = float(self.alpha_smooth)
        self.alpha_value_l2 = float(self.alpha_value_l2)
        self.optimize_pressure_offsets = bool(self.optimize_pressure_offsets)
        self.verbose = bool(self.verbose)
        self.verbose_every = int(self.verbose_every)

        aliases = {
            "pointwise_normal": "pointwise_random",
            "pointwise_random_normal": "pointwise_random",
            "pointwise_fixed": "fixed_with_noise",
            "fixed": "fixed_with_noise",
            "fixed_value": "fixed_with_noise",
            "interface_wise_random": "interfacewise_random",
            "interfacewise_normal": "interfacewise_random",
            "interface_wise_normal": "interfacewise_random",
            "truth": "dataset_truth",
        }
        self.init_mode = aliases.get(self.init_mode, self.init_mode)
        if self.optimizer not in {"adam", "lbfgs"}:
            raise ValueError("optimizer must be 'adam' or 'lbfgs'")
        if self.max_iter < 0:
            raise ValueError("max_iter must be >= 0")
        if self.query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        if self.length_unit_scale <= 0.0:
            raise ValueError("length_unit_scale must be positive")


def _field_indices_from_names(
    output_channel_names: Sequence[str],
    required: Sequence[str] = ("pressure", "u", "v"),
) -> Dict[str, int]:
    names = list(output_channel_names)
    out: Dict[str, int] = {}
    for name in required:
        if name not in names:
            raise ValueError(f"Required output field {name!r} is missing from {names}")
        out[name] = names.index(name)
    return out


def _default_optimize_fields(output_channel_names: Sequence[str]) -> List[str]:
    names = list(output_channel_names)
    preferred = ["pressure", "u", "v"]
    out = [name for name in preferred if name in names]
    if not out:
        out = [name for name in names if name != "temperature"]
    return out


def _select_optimize_indices(
    branch_channel_names: Sequence[str],
    output_channel_names: Sequence[str],
    optimize_fields: Optional[Sequence[str]],
) -> Tuple[List[str], Array, Array]:
    """Return optimized field names, branch value-channel indices, output indices."""
    if optimize_fields is None:
        fields = _default_optimize_fields(output_channel_names)
    else:
        fields = [str(f) for f in optimize_fields]

    branch_names = list(branch_channel_names)
    output_names = list(output_channel_names)
    missing_out = [f for f in fields if f not in output_names]
    missing_branch = [f"boundary_{f}" for f in fields if f"boundary_{f}" not in branch_names]
    if missing_out:
        raise ValueError(f"optimize_fields missing from output_channel_names: {missing_out}")
    if missing_branch:
        raise ValueError(f"optimize_fields missing from branch_channel_names: {missing_branch}")

    branch_idx = np.asarray([branch_names.index(f"boundary_{f}") for f in fields], dtype=np.int64)
    out_idx = np.asarray([output_names.index(f) for f in fields], dtype=np.int64)
    return fields, branch_idx, out_idx

def _torch_normalized_branches(
    branch_raw: Array,
    branch_channel_names: Sequence[str],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float],
    local_aspect_std: Optional[float],
    device: Union[str, torch.device],
) -> torch.Tensor:
    branch_norm_np = np.stack(
        [
            normalize_cell_branch_with_y(
                b,
                branch_channel_names=branch_channel_names,
                target_y_normalizer=y_normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
            )
            for b in np.asarray(branch_raw, dtype=np.float32)
        ],
        axis=0,
    ).astype(np.float32)
    return torch.as_tensor(branch_norm_np, dtype=torch.float32, device=device)


def _make_branch_with_interface_z(
    base_branch_norm: torch.Tensor,
    z_norm: torch.Tensor,
    layout: EdgeLayout,
    opt_branch_channels: Sequence[int],
    known_branch_channels: Optional[Sequence[int]],
) -> torch.Tensor:
    """Write shared normalized interface profiles into both sides of each interface."""
    branch = base_branch_norm.clone()
    n_sub = int(branch.shape[0])
    if n_sub <= 1:
        return branch

    right_idx = torch.as_tensor(layout.right, dtype=torch.long, device=branch.device)
    left_idx = torch.as_tensor(layout.left, dtype=torch.long, device=branch.device)
    val_ch = torch.as_tensor(np.asarray(opt_branch_channels, dtype=np.int64), dtype=torch.long, device=branch.device)

    if z_norm.shape != (n_sub - 1, len(layout.right), len(opt_branch_channels)):
        raise ValueError(
            f"z_norm has shape {tuple(z_norm.shape)}, expected "
            f"{(n_sub - 1, len(layout.right), len(opt_branch_channels))}"
        )

    for interface_id in range(1, n_sub):
        profile = z_norm[interface_id - 1]
        branch[interface_id - 1, right_idx[:, None], val_ch[None, :]] = profile
        branch[interface_id, left_idx[:, None], val_ch[None, :]] = profile

        if known_branch_channels is not None:
            known_ch = torch.as_tensor(
                np.asarray(known_branch_channels, dtype=np.int64),
                dtype=torch.long,
                device=branch.device,
            )
            branch[interface_id - 1, right_idx[:, None], known_ch[None, :]] = 1.0
            branch[interface_id, left_idx[:, None], known_ch[None, :]] = 1.0

    return branch


def _edge_query_tensors(
    branch_raw: Array,
    layout: EdgeLayout,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    branch = np.asarray(branch_raw, dtype=np.float32)
    left_q = branch[:, layout.left, :][:, :, [layout.x_channel, layout.y_channel]]
    right_q = branch[:, layout.right, :][:, :, [layout.x_channel, layout.y_channel]]
    q_left = torch.as_tensor(left_q, dtype=torch.float32, device=device)
    q_right = torch.as_tensor(right_q, dtype=torch.float32, device=device)
    return q_left, q_right


def _query_channel_indices(
    model: DeepONet,
    branch_channel_names: Sequence[str],
    layout: EdgeLayout,
) -> List[int]:
    """Branch channels forming a trunk query for this model: (x, y) plus sdf when needed."""
    channels = [int(layout.x_channel), int(layout.y_channel)]
    trunk_dim = int(getattr(model, "trunk_input_dim", 2))
    if trunk_dim >= 3:
        names = list(branch_channel_names)
        if "sdf" not in names:
            raise ValueError(
                f"model.trunk_input_dim={trunk_dim} expects an 'sdf' query channel, but the "
                "branch has no 'sdf' channel. Build the dataset with include_sdf=True."
            )
        channels.append(names.index("sdf"))
    return channels


def _torch_min_distance_to_segments(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """Differentiable unsigned min distance from points (..., 2) to segments (S, 4).

    Differentiable a.e. w.r.t. ``points`` (kinks only on the medial axis), which
    is what the traction terms need when the SDF is a trunk input.
    """
    pts = points.reshape(-1, 2)
    seg = segments.to(dtype=pts.dtype, device=pts.device).reshape(-1, 4)
    a = seg[:, 0:2]
    d = seg[:, 2:4] - seg[:, 0:2]
    dd = (d * d).sum(dim=1).clamp_min(1.0e-30)
    ap = pts[:, None, :] - a[None, :, :]
    t = ((ap * d[None, :, :]).sum(dim=2) / dd[None, :]).clamp(0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * d[None, :, :]
    dist = torch.linalg.norm(pts[:, None, :] - closest, dim=2)
    return dist.min(dim=1).values.reshape(points.shape[:-1])


@dataclass
class EdgeSDFContext:
    """Recomputes the wall-SDF trunk channel for edge queries, inside autograd.

    Local edge coordinates are mapped to physical (mesh-unit) coordinates with
    per-subdomain frames, then the distance to the case wall segments is taken
    and normalized by the reference length — matching the dataset's ``sdf``
    channel while keeping the chain rule ``d(sdf)/d(x_local, y_local)`` intact.
    """

    segments: torch.Tensor   # (S, 4), mesh units, shared across the case
    x0: torch.Tensor         # (n_sub,) subdomain left edge, mesh units
    width: torch.Tensor      # (n_sub,)
    y0: torch.Tensor         # (n_sub,) local-frame y origin, mesh units
    y_height: torch.Tensor   # (n_sub,) local-frame y scale, mesh units
    ref_length: float        # mesh units

    def augment(self, q: torch.Tensor) -> torch.Tensor:
        """Append the sdf column to local edge queries q of shape (n_sub, N, 2)."""
        px = self.x0[:, None] + q[..., 0] * self.width[:, None]
        py = self.y0[:, None] + q[..., 1] * self.y_height[:, None]
        pts = torch.stack([px, py], dim=-1)
        sdf = _torch_min_distance_to_segments(pts, self.segments) / float(self.ref_length)
        return torch.cat([q, sdf.unsqueeze(-1).to(q.dtype)], dim=-1)


def make_edge_sdf_context(
    samples: Sequence[Mapping[str, Array]],
    metadata: Optional[Sequence[Mapping[str, object]]],
    device: Union[str, torch.device],
) -> Optional[EdgeSDFContext]:
    """Build an EdgeSDFContext from dataset samples, or None when unavailable."""
    if not samples or "wall_segments" not in samples[0]:
        return None
    if metadata is None:
        raise ValueError("make_edge_sdf_context requires per-sample metadata for the local frames")
    if len(metadata) != len(samples):
        raise ValueError(f"metadata length {len(metadata)} does not match samples length {len(samples)}")

    device = torch.device(device)
    segments = torch.as_tensor(np.asarray(samples[0]["wall_segments"], dtype=np.float32), device=device)

    x0, width, y0, y_height = [], [], [], []
    ref_length: Optional[float] = None
    for m in metadata:
        x_left = float(m["x_left_mm"])
        x_right = float(m["x_right_mm"])
        x0.append(x_left)
        width.append(max(x_right - x_left, 1.0e-12))
        y0.append(float(m.get("y_local_origin_mm", m.get("y_bottom_mm", 0.0))))
        y_height.append(max(float(m.get("y_local_scale_mm", m.get("reference_length_mesh", 1.0))), 1.0e-12))
        if ref_length is None:
            ref_length = float(m.get("reference_length_mesh", m.get("reference_length_mm", 1.0)))

    return EdgeSDFContext(
        segments=segments,
        x0=torch.as_tensor(x0, dtype=torch.float32, device=device),
        width=torch.as_tensor(width, dtype=torch.float32, device=device),
        y0=torch.as_tensor(y0, dtype=torch.float32, device=device),
        y_height=torch.as_tensor(y_height, dtype=torch.float32, device=device),
        ref_length=float(ref_length if ref_length is not None else 1.0),
    )


def _subdomain_scales_from_metadata(
    metadata: Optional[Sequence[Mapping[str, object]]],
    n_sub: int,
    length_unit_scale: float,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x/y scales converting local query derivatives to physical derivatives."""
    if metadata is None:
        x_scale = torch.ones(n_sub, dtype=torch.float32, device=device)
        y_scale = torch.ones(n_sub, dtype=torch.float32, device=device)
        return x_scale, y_scale

    if len(metadata) != n_sub:
        raise ValueError(f"metadata length {len(metadata)} does not match n_sub={n_sub}")

    xs = []
    ys = []
    for m in metadata:
        if "x_left_mm" in m and "x_right_mm" in m:
            width = float(m["x_right_mm"]) - float(m["x_left_mm"])
        elif "x_left" in m and "x_right" in m:
            width = float(m["x_right"]) - float(m["x_left"])
        else:
            width = 1.0

        if "reference_length_mm" in m:
            ref = float(m["reference_length_mm"])
        elif "reference_length" in m:
            ref = float(m["reference_length"])
        else:
            ref = 1.0

        xs.append(max(width * float(length_unit_scale), 1.0e-12))
        ys.append(max(ref * float(length_unit_scale), 1.0e-12))

    return (
        torch.as_tensor(xs, dtype=torch.float32, device=device),
        torch.as_tensor(ys, dtype=torch.float32, device=device),
    )


def _line_weights_from_query_y(
    q_edge: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    """Trapezoid weights for vertical interface line integrals in physical length."""
    # q_edge: (S, Ny, 2), y coordinate is local y = y_phys/ref_length.
    y = q_edge[..., 1]
    if y.shape[1] < 2:
        return torch.ones_like(y)
    dy = torch.abs(y[:, 1:] - y[:, :-1]) * y_scale[:, None]
    w = torch.zeros_like(y)
    w[:, 0] = 0.5 * dy[:, 0]
    w[:, -1] = 0.5 * dy[:, -1]
    if y.shape[1] > 2:
        w[:, 1:-1] = 0.5 * (dy[:, :-1] + dy[:, 1:])
    return w


def _decode_outputs_torch(out_norm: torch.Tensor, y_normalizer: FeatureNormalizer) -> torch.Tensor:
    y_normalizer = y_normalizer.to(out_norm.device)
    return y_normalizer.decode(out_norm)


def _scalar_grad_torch(y: torch.Tensor, x: torch.Tensor, retain_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=False,
    )[0]


def _traction_from_output_norm(
    out_norm: torch.Tensor,
    query_local: torch.Tensor,
    normal: Tuple[float, float],
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    y_normalizer: FeatureNormalizer,
    output_channel_names: Sequence[str],
    viscosity: float,
    pressure_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute t=sigma*n using decoded p/u/v and physical-coordinate gradients."""
    idx = _field_indices_from_names(output_channel_names, ("pressure", "u", "v"))
    out_phys = _decode_outputs_torch(out_norm, y_normalizer)
    p = out_phys[..., idx["pressure"]]
    u = out_phys[..., idx["u"]]
    v = out_phys[..., idx["v"]]

    if pressure_offset is not None:
        p = p + pressure_offset[:, None]

    grad_u = _scalar_grad_torch(u, query_local, retain_graph=True)
    grad_v = _scalar_grad_torch(v, query_local, retain_graph=True)

    u_x = grad_u[..., 0] / x_scale[:, None]
    u_y = grad_u[..., 1] / y_scale[:, None]
    v_x = grad_v[..., 0] / x_scale[:, None]
    v_y = grad_v[..., 1] / y_scale[:, None]

    mu = float(viscosity)
    sigma_xx = -p + 2.0 * mu * u_x
    sigma_xy = mu * (u_y + v_x)
    sigma_yy = -p + 2.0 * mu * v_y

    n_x, n_y = float(normal[0]), float(normal[1])
    t_x = sigma_xx * n_x + sigma_xy * n_y
    t_y = sigma_xy * n_x + sigma_yy * n_y
    return torch.stack([t_x, t_y], dim=-1)


def _edge_flux_from_output_norm(
    out_norm: torch.Tensor,
    q_edge: torch.Tensor,
    normal: Tuple[float, float],
    weights: torch.Tensor,
    y_normalizer: FeatureNormalizer,
    output_channel_names: Sequence[str],
) -> torch.Tensor:
    idx = _field_indices_from_names(output_channel_names, ("u", "v"))
    out_phys = _decode_outputs_torch(out_norm, y_normalizer)
    u = out_phys[..., idx["u"]]
    v = out_phys[..., idx["v"]]
    un = u * float(normal[0]) + v * float(normal[1])
    return torch.sum(un * weights, dim=1)


def _mean_sq_weighted(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return torch.sum(w * x * x) / torch.clamp(torch.sum(w), min=1.0e-12)

def _smoothness_penalty_z(z_norm: torch.Tensor) -> torch.Tensor:
    if z_norm.shape[1] < 3:
        return z_norm.new_tensor(0.0)
    d2 = z_norm[:, 2:, :] - 2.0 * z_norm[:, 1:-1, :] + z_norm[:, :-2, :]
    return torch.mean(d2 * d2)


def _initial_z_from_branch(
    init_branch: Array,
    layout: EdgeLayout,
    y_normalizer: FeatureNormalizer,
    opt_branch_channels: Sequence[int],
) -> torch.Tensor:
    """Pack shared interior interface values into normalized output units."""
    branch = np.asarray(init_branch, dtype=np.float32)
    n_sub = branch.shape[0]
    if n_sub <= 1:
        return torch.zeros((0, int(layout.right.size), len(opt_branch_channels)), dtype=torch.float32)

    y_mean = y_normalizer.mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
    y_std = np.maximum(y_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float32), 1.0e-12)

    # Map branch boundary channel index -> output-channel index by ordering.
    all_value_ch = list(np.asarray(layout.value_channels, dtype=np.int64))
    opt_out_idx = [all_value_ch.index(int(c)) for c in opt_branch_channels]

    profiles = []
    for interface_id in range(1, n_sub):
        r = branch[interface_id - 1][np.ix_(layout.right, opt_branch_channels)]
        l = branch[interface_id][np.ix_(layout.left, opt_branch_channels)]
        p = 0.5 * (r + l)
        mean = y_mean[np.asarray(opt_out_idx, dtype=np.int64)].reshape(1, -1)
        std = y_std[np.asarray(opt_out_idx, dtype=np.int64)].reshape(1, -1)
        profiles.append(((p - mean) / std).astype(np.float32))
    return torch.as_tensor(np.stack(profiles, axis=0).astype(np.float32))


def _write_final_z_to_physical_branch(
    base_branch: Array,
    z_norm: Array,
    layout: EdgeLayout,
    y_normalizer: FeatureNormalizer,
    opt_branch_channels: Sequence[int],
) -> Array:
    branch = np.asarray(base_branch, dtype=np.float32).copy()
    if branch.shape[0] <= 1:
        return branch

    y_mean = y_normalizer.mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
    y_std = np.maximum(y_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float32), 1.0e-12)
    all_value_ch = list(np.asarray(layout.value_channels, dtype=np.int64))
    opt_out_idx = np.asarray([all_value_ch.index(int(c)) for c in opt_branch_channels], dtype=np.int64)
    mean = y_mean[opt_out_idx].reshape(1, -1)
    std = y_std[opt_out_idx].reshape(1, -1)

    for local_i, interface_id in enumerate(range(1, branch.shape[0])):
        profile_phys = z_norm[local_i] * std + mean
        profile_phys = profile_phys.astype(np.float32, copy=False)
        _set_edge_profile(branch[interface_id - 1], layout.right, np.asarray(opt_branch_channels), profile_phys)
        _set_edge_profile(branch[interface_id], layout.left, np.asarray(opt_branch_channels), profile_phys)
        if layout.known_channels is not None:
            # Keep all physical interface values marked known, matching your training representation.
            branch[interface_id - 1][np.ix_(layout.right, layout.known_channels)] = 1.0
            branch[interface_id][np.ix_(layout.left, layout.known_channels)] = 1.0
    return branch.astype(np.float32, copy=False)


def _auto_loss_scales(
    y_normalizer: FeatureNormalizer,
    output_channel_names: Sequence[str],
    q_left: torch.Tensor,
    q_right: torch.Tensor,
    y_scale: torch.Tensor,
    config: PhysicsInterfaceConfig,
) -> Tuple[float, float]:
    names = list(output_channel_names)
    std = y_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float64)
    p_scale = float(std[names.index("pressure")]) if "pressure" in names else 1.0
    uv_scales = []
    for f in ("u", "v"):
        if f in names:
            uv_scales.append(float(std[names.index(f)]))
    vel_scale = max(max(uv_scales) if uv_scales else 1.0, 1.0e-12)

    if config.traction_scale is None:
        traction_scale = max(abs(p_scale), 1.0e-12)
    else:
        traction_scale = max(abs(float(config.traction_scale)), 1.0e-12)

    if config.flux_scale is None:
        with torch.no_grad():
            w_ref = _line_weights_from_query_y(q_right, y_scale)
            length_ref = float(torch.mean(torch.sum(w_ref, dim=1)).detach().cpu())
        flux_scale = max(abs(vel_scale * length_ref), 1.0e-12)
    else:
        flux_scale = max(abs(float(config.flux_scale)), 1.0e-12)
    return traction_scale, flux_scale


def physics_interface_loss_torch(
    model: DeepONet,
    base_branch_norm: torch.Tensor,
    z_norm: torch.Tensor,
    layout: EdgeLayout,
    q_left_base: torch.Tensor,
    q_right_base: torch.Tensor,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    y_normalizer: FeatureNormalizer,
    output_channel_names: Sequence[str],
    opt_branch_channels: Sequence[int],
    opt_output_channels: Sequence[int],
    config: PhysicsInterfaceConfig,
    pressure_offsets_raw: Optional[torch.Tensor] = None,
    traction_scale: float = 1.0,
    flux_scale: float = 1.0,
    sdf_ctx: Optional[EdgeSDFContext] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable physics loss for the unknown interface trace."""
    branch = _make_branch_with_interface_z(
        base_branch_norm=base_branch_norm,
        z_norm=z_norm,
        layout=layout,
        opt_branch_channels=opt_branch_channels,
        known_branch_channels=None,
    )

    q_left = q_left_base.detach().clone().requires_grad_(True)
    q_right = q_right_base.detach().clone().requires_grad_(True)

    # The (x, y) leaves stay 2-column so traction/flux gradients are taken w.r.t.
    # the coordinates; the sdf trunk channel is recomputed in-graph so its
    # spatial dependence enters those gradients through the chain rule.
    if sdf_ctx is not None:
        q_left_in = sdf_ctx.augment(q_left)
        q_right_in = sdf_ctx.augment(q_right)
    else:
        q_left_in, q_right_in = q_left, q_right

    out_left_norm = model(branch, q_left_in)
    out_right_norm = model(branch, q_right_in)

    n_sub = int(branch.shape[0])
    if pressure_offsets_raw is not None:
        zero = pressure_offsets_raw.new_zeros(1)
        p_offsets = torch.cat([zero, pressure_offsets_raw], dim=0)
        if p_offsets.numel() != n_sub:
            raise ValueError("pressure_offsets_raw must have length n_sub-1")
    else:
        p_offsets = None

    # Dirichlet consistency on the optimized interface fields in normalized output units.
    if n_sub > 1:
        opt_out = torch.as_tensor(np.asarray(opt_output_channels, dtype=np.int64), dtype=torch.long, device=branch.device)
        weights = torch.tensor([config.alpha_p, config.alpha_u, config.alpha_v], device=branch.device)
        pred_r = out_right_norm[:-1, :, :][:, :, opt_out]
        pred_l = out_left_norm[1:, :, :][:, :, opt_out]
        loss_dir = torch.mean((pred_r - z_norm) ** 2 * weights) + torch.mean((pred_l - z_norm) ** 2 * weights)
        loss_dir += torch.mean((pred_r - pred_l) ** 2 * weights)
    else:
        loss_dir = z_norm.new_tensor(0.0)

    # Traction balance across interior interfaces.
    t_left = _traction_from_output_norm(
        out_left_norm,
        q_left,
        normal=(-1.0, 0.0),
        x_scale=x_scale,
        y_scale=y_scale,
        y_normalizer=y_normalizer,
        output_channel_names=output_channel_names,
        viscosity=config.viscosity,
        pressure_offset=p_offsets,
    )
    t_right = _traction_from_output_norm(
        out_right_norm,
        q_right,
        normal=(1.0, 0.0),
        x_scale=x_scale,
        y_scale=y_scale,
        y_normalizer=y_normalizer,
        output_channel_names=output_channel_names,
        viscosity=config.viscosity,
        pressure_offset=p_offsets,
    )

    w_right = _line_weights_from_query_y(q_right_base, y_scale)
    if n_sub > 1:
        traction_res = t_right[:-1] + t_left[1:]
        loss_traction = _mean_sq_weighted(traction_res / float(traction_scale), w_right[:-1])
    else:
        traction_res = z_norm.new_zeros((0, q_right.shape[1], 2))
        loss_traction = z_norm.new_tensor(0.0)

    # Flux balance inside each subdomain: Q_left + Q_right = 0 for outward normals.
    w_left = _line_weights_from_query_y(q_left_base, y_scale)
    q_flux_left = _edge_flux_from_output_norm(
        out_left_norm,
        q_left,
        normal=(-1.0, 0.0),
        weights=w_left,
        y_normalizer=y_normalizer,
        output_channel_names=output_channel_names,
    )
    q_flux_right = _edge_flux_from_output_norm(
        out_right_norm,
        q_right,
        normal=(1.0, 0.0),
        weights=w_right,
        y_normalizer=y_normalizer,
        output_channel_names=output_channel_names,
    )
    flux_res = q_flux_left + q_flux_right
    flux_res_global = q_flux_left[0] + q_flux_right[-1]
    loss_flux = torch.mean((flux_res / float(flux_scale)) ** 2) + torch.mean((flux_res_global / float(flux_scale)) ** 2)

    loss_smooth = _smoothness_penalty_z(z_norm)
    loss_value_l2 = torch.mean(z_norm * z_norm) if z_norm.numel() else z_norm.new_tensor(0.0)

    loss = (
        config.alpha_traction * loss_traction
        + config.alpha_flux * loss_flux
        + config.alpha_dirichlet * loss_dir
        + config.alpha_smooth * loss_smooth
        + config.alpha_value_l2 * loss_value_l2
    )

    info = {
        "loss": loss.detach(),
        "traction": loss_traction.detach(),
        "flux": loss_flux.detach(),
        "dirichlet": loss_dir.detach(),
        "smooth": loss_smooth.detach(),
        "value_l2": loss_value_l2.detach(),
        "max_abs_traction_res": torch.max(torch.abs(traction_res)).detach() if traction_res.numel() else z_norm.new_tensor(0.0),
        "max_abs_flux_res": torch.max(torch.abs(flux_res)).detach() if flux_res.numel() else z_norm.new_tensor(0.0),
    }
    return loss, info


def physics_unknown_interface_inference(
    model: DeepONet,
    samples: Sequence[Mapping[str, Array]],
    branch_channel_names: Sequence[str],
    output_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    metadata: Optional[Sequence[Mapping[str, object]]] = None,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    config: Optional[PhysicsInterfaceConfig] = None,
) -> Dict[str, object]:
    """Optimize unknown interior interface Dirichlet values using traction + flux.

    This is the evaluation-time replacement for simple averaging.  It freezes the
    trained DeepONet and solves for shared interior interface branch values.

    Returns a dictionary with the same main keys as iterative_unknown_interface_inference,
    plus physics_loss_history and final optimized z values.
    """
    if config is None:
        config = PhysicsInterfaceConfig()
    device = torch.device(device)
    torch.manual_seed(int(config.random_seed))

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    y_normalizer = y_normalizer.to(device)

    branches = np.stack([np.asarray(s["branch"], dtype=np.float32) for s in samples], axis=0)
    layout = infer_edge_layout(branches[0], branch_channel_names)
    fields, opt_branch_ch, opt_out_ch = _select_optimize_indices(
        branch_channel_names=branch_channel_names,
        output_channel_names=output_channel_names,
        optimize_fields=config.optimize_fields,
    )

    if config.init_mode == "dataset_truth":
        init_branch = branches.copy()
    else:
        init_branch = initialize_unknown_interior_interfaces(
            branches,
            layout=layout,
            y_normalizer=y_normalizer,
            random_seed=config.random_seed,
            init_mode=config.init_mode,
            init_std_scale=config.init_std_scale,
            init_value=config.init_value,
            init_noise_std=config.init_noise_std,
            init_gaussian_center_y=config.init_gaussian_center_y,
            init_gaussian_var_y=config.init_gaussian_var_y,
            preserve_wall_corners=config.preserve_wall_corners,
        )

    base_branch_norm = _torch_normalized_branches(
        init_branch,
        branch_channel_names=branch_channel_names,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        device=device,
    )

    z0 = _initial_z_from_branch(
        init_branch,
        layout=layout,
        y_normalizer=y_normalizer,
        opt_branch_channels=opt_branch_ch,
    ).to(device)
    z_norm = torch.nn.Parameter(z0.clone())

    q_left, q_right = _edge_query_tensors(init_branch, layout, device=device)
    sdf_ctx: Optional[EdgeSDFContext] = None
    if int(getattr(model, "trunk_input_dim", 2)) >= 3:
        sdf_ctx = make_edge_sdf_context(samples, metadata, device)
        if sdf_ctx is None:
            raise ValueError(
                "model.trunk_input_dim >= 3 requires the sdf trunk channel: build the "
                "dataset with include_sdf=True so samples carry 'wall_segments'."
            )
    x_scale, y_scale = _subdomain_scales_from_metadata(
        metadata=metadata,
        n_sub=branches.shape[0],
        length_unit_scale=config.length_unit_scale,
        device=device,
    )
    traction_scale, flux_scale = _auto_loss_scales(
        y_normalizer=y_normalizer,
        output_channel_names=output_channel_names,
        q_left=q_left,
        q_right=q_right,
        y_scale=y_scale,
        config=config,
    )

    params: List[torch.nn.Parameter] = [z_norm]
    pressure_offsets_param: Optional[torch.nn.Parameter] = None
    if config.optimize_pressure_offsets and branches.shape[0] > 1:
        pressure_offsets_param = torch.nn.Parameter(torch.zeros(branches.shape[0] - 1, dtype=torch.float32, device=device))
        params.append(pressure_offsets_param)

    if config.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=config.lr, max_iter=20, line_search_fn="strong_wolfe")
    else:
        optimizer = torch.optim.Adam(params, lr=config.lr)

    history: List[Dict[str, float]] = []
    converged = False
    n_iter_done = 0
    prev_loss: Optional[float] = None
    loss_stagnant_iters = 0
    loss_stagnant_patience = 10

    def closure_for_loss() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return physics_interface_loss_torch(
            model=model,
            base_branch_norm=base_branch_norm,
            z_norm=z_norm,
            layout=layout,
            q_left_base=q_left,
            q_right_base=q_right,
            x_scale=x_scale,
            y_scale=y_scale,
            y_normalizer=y_normalizer,
            output_channel_names=output_channel_names,
            opt_branch_channels=opt_branch_ch,
            opt_output_channels=opt_out_ch,
            config=config,
            pressure_offsets_raw=pressure_offsets_param,
            traction_scale=traction_scale,
            flux_scale=flux_scale,
            sdf_ctx=sdf_ctx,
        )

    for iteration in range(1, int(config.max_iter) + 1):
        n_iter_done = iteration
        if config.optimizer == "lbfgs":
            saved_info: Dict[str, torch.Tensor] = {}

            def lbfgs_closure():
                optimizer.zero_grad(set_to_none=True)
                loss, info = closure_for_loss()
                loss.backward()
                saved_info.clear()
                saved_info.update(info)
                return loss

            optimizer.step(lbfgs_closure)
            loss, info = closure_for_loss()
            loss = loss.detach()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, info = closure_for_loss()
            loss.backward()
            optimizer.step()

        row = {k: float(v.detach().cpu()) for k, v in info.items()}
        row["iteration"] = float(iteration)
        row["traction_scale"] = float(traction_scale)
        row["flux_scale"] = float(flux_scale)
        history.append(row)

        metric = row["traction"] + row["flux"] + row["dirichlet"]
        converged = bool(metric <= float(config.tol))

        current_loss = row["loss"]
        if prev_loss is not None and np.isclose(
            current_loss,
            prev_loss,
            rtol=0.0,
            atol=max(float(config.tol), 1.0e-12),
        ):
            loss_stagnant_iters += 1
        else:
            loss_stagnant_iters = 0
        prev_loss = current_loss
        loss_stagnant = loss_stagnant_iters >= loss_stagnant_patience

        if config.verbose and (
            iteration == 1
            or iteration % max(config.verbose_every, 1) == 0
            or iteration == config.max_iter
            or converged
            or loss_stagnant
        ):
            print(
                f"iter={iteration:04d} | loss={row['loss']:.6e} | "
                f"traction={row['traction']:.6e} | flux={row['flux']:.6e} | "
                f"boundary={row['dirichlet']:.6e} | smooth={row['smooth']:.6e}",
                flush=True,
            )
        if converged:
            break
        if loss_stagnant:
            if config.verbose:
                print(
                    f"iter={iteration:04d} | stopping: loss unchanged for "
                    f"{loss_stagnant_patience} iterations",
                    flush=True,
                )
            break

    z_final = z_norm.detach().cpu().numpy().astype(np.float32)
    branch_final = _write_final_z_to_physical_branch(
        base_branch=init_branch,
        z_norm=z_final,
        layout=layout,
        y_normalizer=y_normalizer,
        opt_branch_channels=opt_branch_ch,
    )

    # Evaluate edge predictions and final full-cell predictions using existing helpers.
    pred_left, pred_right = predict_edge_profiles(
        model=model,
        branch_inputs=branch_final,
        layout=layout,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
    )
    mse_total, mse_by_channel = compute_interface_mse_from_edges(pred_left, pred_right)

    pred_samples = predict_cell_samples_physical(
        model=model,
        samples=samples,
        branch_inputs=branch_final,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        query_batch_size=config.query_batch_size,
    )

    pressure_offsets = None
    if pressure_offsets_param is not None:
        pressure_offsets = np.concatenate(
            [np.zeros(1, dtype=np.float32), pressure_offsets_param.detach().cpu().numpy().astype(np.float32)],
            axis=0,
        )

    return {
        "pred_samples": pred_samples,
        "branch_initial": init_branch,
        "branch_initial_random": init_branch,
        "branch_final": branch_final,
        "z_final_normalized": z_final,
        "optimized_fields": fields,
        "pressure_offsets": pressure_offsets,
        "pred_left_interface": pred_left,
        "pred_right_interface": pred_right,
        "interface_mse_history": np.asarray([mse_total], dtype=np.float32),
        "interface_mse_by_channel_history": np.asarray([mse_by_channel], dtype=np.float32),
        "physics_loss_history": history,
        "traction_scale": float(traction_scale),
        "flux_scale": float(flux_scale),
        "converged": bool(converged),
        "n_iter": int(n_iter_done),
        "layout": layout,
        "config": config,
    }


# ============================================================================
# Schwarz Neural Iteration (SNI) interface inference
# ============================================================================
#
# This is the evaluation-time replacement for the physics-based optimizer above.
# Instead of optimizing interior interface Dirichlet values with a differentiable
# traction/flux loss, SNI runs a relaxed Schwarz fixed-point iteration directly
# on the pre-decomposed subdomains (see arXiv:2504.00510, "Schwarz Neural
# Inference"):
#
#   1. Write the current interior-interface values into every subdomain's branch.
#   2. Run the frozen operator on every subdomain and evaluate the predicted
#      field on both sides of each shared interface.
#   3. Update each interior interface as the (relaxed) average of its two
#      neighbours' predicted traces:  z <- (1 - tau) * z + tau * 0.5 * (uL + uR).
#   4. Repeat until the interface mismatch stagnates / falls below tolerance.
#
# Adjacent subdomains (ordered left-to-right by subdomain_id) share a vertical
# interface: subdomain k's right edge coincides with subdomain k+1's left edge,
# sampled at the same interface sensor points. Because those sensor points are
# baked into the branch, we evaluate the shared trace directly with
# ``predict_edge_profiles`` (no interpolation needed).


@dataclass
class SNIInterfaceConfig:
    """Configuration for Schwarz Neural Iteration on unknown interior interfaces.

    The trained DeepONet stays frozen; SNI only updates the shared interior
    interface values written into the branch tensor via a relaxed Schwarz
    fixed-point map.
    """

    max_iter: int = 200
    tau: float = 0.5  # Schwarz relaxation parameter in (0, 1]
    tol: float = 1.0e-6  # convergence tolerance on the mean interface residual
    random_seed: int = 0

    # Initialization of the unknown interior interfaces (same options as the
    # physics path). "fixed_with_noise" + init_value=[0,0,0] reproduces the
    # zero start used by the reference Schwarz driver.
    init_mode: str = "fixed_with_noise"
    init_std_scale: float = 1.0
    init_value: Optional[Union[float, Sequence[float], Array]] = None
    init_noise_std: float = 0.0
    init_gaussian_center_y: float = 0.5
    init_gaussian_var_y: float = 0.02
    preserve_wall_corners: bool = False

    # Which interface branch fields participate in the Schwarz exchange. None
    # means every predicted output field (e.g. pressure/u/v).
    optimize_fields: Optional[Sequence[str]] = None

    # Stagnation-based early stop on the (rounded) mean interface residual.
    stagnation_window: int = 10
    stagnation_decimals: int = 6

    query_batch_size: int = 65536
    verbose: bool = True
    verbose_every: int = 25

    def __post_init__(self) -> None:
        self.max_iter = int(self.max_iter)
        self.tau = float(self.tau)
        self.tol = float(self.tol)
        self.random_seed = int(self.random_seed)
        self.init_mode = str(self.init_mode).lower()
        self.init_std_scale = float(self.init_std_scale)
        self.init_noise_std = float(self.init_noise_std)
        self.init_gaussian_center_y = float(self.init_gaussian_center_y)
        self.init_gaussian_var_y = float(self.init_gaussian_var_y)
        self.preserve_wall_corners = bool(self.preserve_wall_corners)
        self.stagnation_window = int(self.stagnation_window)
        self.stagnation_decimals = int(self.stagnation_decimals)
        self.query_batch_size = int(self.query_batch_size)
        self.verbose = bool(self.verbose)
        self.verbose_every = int(self.verbose_every)

        aliases = {
            "pointwise_normal": "pointwise_random",
            "pointwise_random_normal": "pointwise_random",
            "pointwise_fixed": "fixed_with_noise",
            "fixed": "fixed_with_noise",
            "fixed_value": "fixed_with_noise",
            "interface_wise_random": "interfacewise_random",
            "interfacewise_normal": "interfacewise_random",
            "interface_wise_normal": "interfacewise_random",
            "truth": "dataset_truth",
        }
        self.init_mode = aliases.get(self.init_mode, self.init_mode)
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if self.max_iter < 0:
            raise ValueError("max_iter must be >= 0")
        if self.query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")


def _interfaces_from_branch(
    branch: Array,
    layout: EdgeLayout,
    opt_branch_channels: Array,
) -> Array:
    """Read the current shared interior interface profiles from a branch.

    Returns an array of shape ``(n_interfaces, n_interface_points, n_opt)`` where
    interface ``k`` is the average of subdomain ``k``'s right edge and subdomain
    ``k+1``'s left edge over the optimized value channels.
    """
    branch = np.asarray(branch, dtype=np.float32)
    n_sub = branch.shape[0]
    n_opt = int(np.asarray(opt_branch_channels).size)
    if n_sub <= 1:
        return np.zeros((0, int(layout.right.size), n_opt), dtype=np.float32)

    profiles = []
    for interface_id in range(1, n_sub):
        right = branch[interface_id - 1][np.ix_(layout.right, opt_branch_channels)]
        left = branch[interface_id][np.ix_(layout.left, opt_branch_channels)]
        profiles.append(0.5 * (right + left))
    return np.stack(profiles, axis=0).astype(np.float32)


def _write_interfaces_to_branch(
    branch: Array,
    interfaces: Array,
    layout: EdgeLayout,
    opt_branch_channels: Array,
) -> Array:
    """Write shared interface profiles into both sides of every interior interface."""
    branch = np.asarray(branch, dtype=np.float32)
    opt_branch_channels = np.asarray(opt_branch_channels, dtype=np.int64)
    for local_i, interface_id in enumerate(range(1, branch.shape[0])):
        profile = np.asarray(interfaces[local_i], dtype=np.float32)
        _set_edge_profile(branch[interface_id - 1], layout.right, opt_branch_channels, profile)
        _set_edge_profile(branch[interface_id], layout.left, opt_branch_channels, profile)
        if layout.known_channels is not None:
            branch[interface_id - 1][np.ix_(layout.right, layout.known_channels)] = 1.0
            branch[interface_id][np.ix_(layout.left, layout.known_channels)] = 1.0
    return branch
