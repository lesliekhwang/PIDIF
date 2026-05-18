"""
Iterative interface inference for Fluent DeepONet subdomain models.

Problem solved here
-------------------
During testing, interior interface values are treated as unknown. The script:

1. keeps exterior boundaries and walls from the dataset branch input;
2. replaces all interior left/right interface values with initialized values
   (random by default, or fixed value with small noise if configured);
3. predicts all subdomains simultaneously;
4. compares the two predicted traces on each shared interface;
5. replaces both sides of each shared interface with the pointwise average of
   the two predicted traces;
6. repeats until all shared-interface MSE values are below a tolerance.

The iteration is Jacobi-style: all subdomains are predicted from the same
current interface state, and all interfaces are updated at the same time.

Expected model type
-------------------
This evaluator is intended for the base DeepONet checkpoint produced by
train_deeponet.ipynb, whose checkpoint contains:

    model_state_dict
    model_config
    branch_normalizer
    y_normalizer

The hard-BC wrapper is not used here. With a hard-BC model, the predicted edge
values would equal the current branch interface values exactly, making this
iteration degenerate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.optim as optim

from fluent_deeponet import (
    DeepONet,
    FeatureNormalizer,
    load_deeponet_dataset_h5,
)

Array = np.ndarray
PathLike = Union[str, Path]

@dataclass
class InterfaceIterationConfig:
    """Configuration for unknown-interface inference."""

    max_iter: int = 100
    tol: float = 1.0e-6
    mse_mode: str = "normalized"  # "normalized" or "physical"
    relaxation: float = 1.0
    random_seed: int = 0
    init_mode: str = "pointwise_random"  # "pointwise_random" or "gaussian_along_y"
    init_std_scale: float = 1.0
    init_value: Optional[float] = None
    init_noise_std: float = 1.0e-3
    init_gaussian_center_y: float = 0.5
    init_gaussian_var_y: float = 0.02
    query_batch_size: int = 32768
    sample_batch_size: int = 8
    preserve_wall_corners: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)
        self.relaxation = float(self.relaxation)
        self.random_seed = int(self.random_seed)
        self.init_mode = str(self.init_mode).lower()
        self.init_std_scale = float(self.init_std_scale)
        self.init_noise_std = float(self.init_noise_std)
        self.init_gaussian_center_y = float(self.init_gaussian_center_y)
        self.init_gaussian_var_y = float(self.init_gaussian_var_y)
        self.query_batch_size = int(self.query_batch_size)
        self.sample_batch_size = int(self.sample_batch_size)
        self.mse_mode = str(self.mse_mode).lower()
        if self.mse_mode not in {"normalized", "physical"}:
            raise ValueError("mse_mode must be 'normalized' or 'physical'.")
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError("relaxation must satisfy 0 < relaxation <= 1.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be >= 0.")
        if self.query_batch_size <= 0 or self.sample_batch_size <= 0:
            raise ValueError("batch sizes must be positive.")
        if self.init_mode not in {"pointwise_random", "gaussian_along_y", "dataset_truth"}:
            raise ValueError("init_mode must be 'pointwise_random' or 'gaussian_along_y' or 'dataset_truth'.")
        if self.init_noise_std < 0.0:
            raise ValueError("init_noise_std must be >= 0.")
        if self.init_gaussian_var_y <= 0.0:
            raise ValueError("init_gaussian_var_y must be > 0.")


@dataclass
class EdgeLayout:
    """Sensor indices for the four sides of one branch input sample."""

    left: Array
    right: Array
    bottom: Array
    top: Array
    value_channels: Array
    x_channel: int
    y_channel: int
    wall_channel: int
    interface_channel: int


def _as_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _normalizer_from_checkpoint(
    ckpt: Mapping[str, object],
    key: str,
) -> Optional[FeatureNormalizer]:
    if key not in ckpt or ckpt[key] is None:
        return None
    return FeatureNormalizer.from_state_dict(ckpt[key])


def load_base_deeponet_checkpoint(
    checkpoint_path: PathLike,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[DeepONet, Optional[FeatureNormalizer], Optional[FeatureNormalizer], Dict[str, object]]:
    """
    Load a base DeepONet checkpoint from train_deeponet.ipynb.

    Returns
    -------
    model, branch_normalizer, y_normalizer, checkpoint_dict
    """
    device = torch.device(device)
    ckpt = torch.load(_as_path(checkpoint_path), map_location=device)

    if "hard_model_config" in ckpt:
        raise ValueError(
            "This evaluator expects a base DeepONet checkpoint. The supplied "
            "checkpoint looks like a hard-BC checkpoint. Use the base "
            "DeepONet checkpoint from train_deeponet.ipynb."
        )
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint must contain 'model_config' and 'model_state_dict'."
        )

    model = DeepONet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    branch_normalizer = _normalizer_from_checkpoint(ckpt, "branch_normalizer")
    y_normalizer = _normalizer_from_checkpoint(ckpt, "y_normalizer")

    if branch_normalizer is not None:
        branch_normalizer = branch_normalizer.to(device)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    return model, branch_normalizer, y_normalizer, dict(ckpt)


def select_ar_subdomains(
    dataset: Mapping[str, object],
    ar: int,
    realization_id: Optional[int] = None,
) -> Array:
    """
    Return sample indices for one AR, sorted by subdomain_id.

    If the dataset contains multiple interface realizations, pass
    realization_id to select one of them.
    """
    ars = np.asarray(dataset["aspect_ratio"]).astype(int)
    subdomain_id = np.asarray(dataset["subdomain_id"]).astype(int)
    mask = ars == int(ar)

    if realization_id is not None:
        if "interface_realization_id" not in dataset:
            raise KeyError(
                "dataset does not contain interface_realization_id, but "
                "realization_id was requested."
            )
        rid = np.asarray(dataset["interface_realization_id"]).astype(int)
        mask &= rid == int(realization_id)

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No samples found for AR={ar}.")

    order = np.argsort(subdomain_id[idx], kind="stable")
    idx = idx[order]

    sorted_subs = subdomain_id[idx]
    expected = np.arange(sorted_subs.size, dtype=sorted_subs.dtype)
    if not np.array_equal(sorted_subs, expected):
        raise ValueError(
            "Selected subdomains are not exactly 0..S-1. Found "
            f"{sorted_subs.tolist()}"
        )
    return idx.astype(np.int64)


def _unique_sorted_by_coordinate(
    indices: Array,
    coord: Array,
) -> Array:
    """Sort indices by coord and keep the first index at duplicate coords."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices
    order = np.argsort(coord[indices], kind="stable")
    sorted_idx = indices[order]
    rounded = np.round(coord[sorted_idx], decimals=10)
    _, first = np.unique(rounded, return_index=True)
    return sorted_idx[np.sort(first)].astype(np.int64)


def infer_edge_layout(
    branch_template: Array,
    branch_channel_names: Sequence[str],
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    include_corner_dupes: Optional[bool] = None,
    tol: float = 1.0e-6,
) -> EdgeLayout:
    """
    Infer edge sensor indices from a branch input sample.

    For the standard dataset with include_corner_dupes=True, the builder order is
    known exactly:

        left(ny), right(ny), bottom(nx), top(nx)

    In all other cases, coordinates and masks are used.
    """
    branch = np.asarray(branch_template, dtype=np.float32)
    names = list(branch_channel_names)

    x_ch = names.index("x_local")
    y_ch = names.index("y_local")
    wall_ch = names.index("wall_mask")
    interface_ch = names.index("interface_mask")
    value_channels = np.asarray(
        [
            names.index("boundary_pressure"),
            names.index("boundary_temperature"),
            names.index("boundary_u"),
            names.index("boundary_v"),
        ],
        dtype=np.int64,
    )

    m = branch.shape[0]
    if nx is not None and ny is not None:
        nx = int(nx)
        ny = int(ny)
        if m == 2 * ny + 2 * nx:
            # Exact construction order when include_corner_dupes=True.
            left = np.arange(0, ny, dtype=np.int64)
            right = np.arange(ny, 2 * ny, dtype=np.int64)
            bottom = np.arange(2 * ny, 2 * ny + nx, dtype=np.int64)
            top = np.arange(2 * ny + nx, 2 * ny + 2 * nx, dtype=np.int64)
            return EdgeLayout(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                value_channels=value_channels,
                x_channel=x_ch,
                y_channel=y_ch,
                wall_channel=wall_ch,
                interface_channel=interface_ch,
            )

    x = branch[:, x_ch]
    y = branch[:, y_ch]
    wall = branch[:, wall_ch]
    interface = branch[:, interface_ch]

    left = np.flatnonzero((np.abs(x - 0.0) <= tol) & (interface > 0.5))
    right = np.flatnonzero((np.abs(x - 1.0) <= tol) & (interface > 0.5))
    bottom = np.flatnonzero((np.abs(y - 0.0) <= tol) & (wall > 0.5))
    top = np.flatnonzero((np.abs(y - 1.0) <= tol) & (wall > 0.5))

    left = _unique_sorted_by_coordinate(left, y)
    right = _unique_sorted_by_coordinate(right, y)
    bottom = _unique_sorted_by_coordinate(bottom, x)
    top = _unique_sorted_by_coordinate(top, x)

    if left.size == 0 or right.size == 0 or bottom.size == 0 or top.size == 0:
        raise ValueError(
            "Could not identify all four boundary edge sensor sets. Found "
            f"left={left.size}, right={right.size}, bottom={bottom.size}, top={top.size}."
        )

    return EdgeLayout(
        left=left.astype(np.int64),
        right=right.astype(np.int64),
        bottom=bottom.astype(np.int64),
        top=top.astype(np.int64),
        value_channels=value_channels,
        x_channel=x_ch,
        y_channel=y_ch,
        wall_channel=wall_ch,
        interface_channel=interface_ch,
    )


def _branch_random_stats(
    branch_inputs: Array,
    layout: EdgeLayout,
    branch_normalizer: Optional[FeatureNormalizer],
) -> Tuple[Array, Array]:
    """Mean/std used for random interface initialization in physical units."""
    value_idx = layout.value_channels
    if branch_normalizer is not None:
        mean = branch_normalizer.mean.detach().cpu().numpy()[value_idx]
        std = branch_normalizer.std.detach().cpu().numpy()[value_idx]
        return mean.astype(np.float32), std.astype(np.float32)

    values = np.asarray(branch_inputs[..., value_idx], dtype=np.float32).reshape(-1, len(value_idx))
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.maximum(std, 1.0e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _edge_profile(branch_one: Array, edge_idx: Array, value_idx: Array) -> Array:
    return np.asarray(branch_one[np.asarray(edge_idx), :][:, np.asarray(value_idx)], dtype=np.float32)


def _set_edge_profile(branch_one: Array, edge_idx: Array, value_idx: Array, profile: Array) -> None:
    edge_idx = np.asarray(edge_idx, dtype=np.int64)
    value_idx = np.asarray(value_idx, dtype=np.int64)
    profile = np.asarray(profile, dtype=np.float32)
    if profile.shape != (edge_idx.size, value_idx.size):
        raise ValueError(
            f"profile has shape {profile.shape}, expected {(edge_idx.size, value_idx.size)}"
        )
    branch_one[np.ix_(edge_idx, value_idx)] = profile


def _preserve_wall_corner_values(
    profile: Array,
    branch_one: Array,
    edge_name: str,
    layout: EdgeLayout,
) -> Array:
    """
    Optionally replace vertical-edge endpoints by known wall-corner values.

    This is only a consistency option for datasets that include duplicated corner
    sensors. It is off by default in InterfaceIterationConfig because the user
    requested random values for each interface point.
    """
    out = np.array(profile, dtype=np.float32, copy=True)
    if edge_name not in {"left", "right"}:
        return out
    if out.shape[0] < 2:
        return out

    if edge_name == "left":
        bottom_corner_idx = layout.bottom[0]
        top_corner_idx = layout.top[0]
    else:
        bottom_corner_idx = layout.bottom[-1]
        top_corner_idx = layout.top[-1]

    out[0] = branch_one[bottom_corner_idx, layout.value_channels]
    out[-1] = branch_one[top_corner_idx, layout.value_channels]
    return out


def initialize_unknown_interior_interfaces(
    branch_inputs_for_ar: Array,
    layout: EdgeLayout,
    branch_normalizer: Optional[FeatureNormalizer] = None,
    random_seed: int = 0,
    init_mode: str = "pointwise_random",
    init_std_scale: float = 1.0,
    init_value: Optional[Union[float, Sequence[float], Array]] = None,
    init_noise_std: float = 1.0e-3,
    preserve_wall_corners: bool = False,
) -> Array:
    """
    Replace interior interface values using the requested initialization mode.

    Exterior inlet/outlet interfaces are kept from the dataset.

    - init_mode="pointwise_random":
      preserve prior behavior (sample each point independently), optionally
      centered around init_value with init_noise_std.
    - init_mode="gaussian_along_y":
      build a bell-shaped profile along y with peak near init_gaussian_center_y
      and spread init_gaussian_var_y, then add optional noise.

    Interior shared interfaces are initialized independently on the two
    adjacent subdomain sides, so right edge of k and left edge of k+1 start
    out different.
    """
    branch = np.asarray(branch_inputs_for_ar, dtype=np.float32).copy()
    if branch.ndim != 3:
        raise ValueError(f"branch_inputs_for_ar must have shape (S,M,C), got {branch.shape}")

    rng = np.random.default_rng(int(random_seed))
    init_mode = str(init_mode).lower()
    if init_mode not in {"pointwise_random", "gaussian_along_y"}:
        raise ValueError("init_mode must be 'pointwise_random' or 'gaussian_along_y' or 'dataset_truth'.")
    n_sub = branch.shape[0]
    value_dim = len(layout.value_channels)
    noise_std = max(float(init_noise_std), 0.0)

    if init_value is None:
        mean, std = _branch_random_stats(branch, layout, branch_normalizer)
        mean = mean.astype(np.float32, copy=False)
        std = np.maximum(std, 1.0e-12) * float(init_std_scale)
        gauss_center = np.tile(mean, (layout.right.size, 1))
        gauss_var = np.tile(std, (layout.right.size, 1))
        
    else:
        init_value_arr = np.asarray(init_value, dtype=np.float32)
        if init_value_arr.ndim == 0:
            mean = np.full((value_dim,), float(init_value_arr), dtype=np.float32)
        else:
            mean = init_value_arr.reshape(-1).astype(np.float32, copy=False)
            if mean.size != value_dim:
                raise ValueError(
                    f"init_value must be scalar or length {value_dim}, got shape {init_value_arr.shape}"
                )
        std = np.full((value_dim,), noise_std, dtype=np.float32)
        gauss_center = np.tile(mean, (layout.right.size, 1))
        gauss_var = np.tile(std, (layout.right.size, 1))

    for interface_id in range(1, n_sub):
        if init_mode == "pointwise_random":
            # Right side of subdomain interface_id - 1.
            right_rand = rng.normal(
                loc=mean.reshape(1, value_dim),
                scale=std.reshape(1, value_dim),
                size=(layout.right.size, value_dim),
            ).astype(np.float32)
            # Left side of subdomain interface_id.
            left_rand = rng.normal(
                loc=mean.reshape(1, value_dim),
                scale=std.reshape(1, value_dim),
                size=(layout.left.size, value_dim),
            ).astype(np.float32)
        else:
            # Build smooth Gaussian profiles along y for each interface side.
            y_right = branch[interface_id - 1, layout.right, layout.y_channel].astype(np.float32)
            y_left = branch[interface_id, layout.left, layout.y_channel].astype(np.float32)
            g_right = np.exp(-0.5 * ((y_right - gauss_center) ** 2) / gauss_var).astype(np.float32)
            g_left = np.exp(-0.5 * ((y_left - gauss_center) ** 2) / gauss_var).astype(np.float32)
            right_base = g_right.reshape(-1, 1) * mean.reshape(1, value_dim)
            left_base = g_left.reshape(-1, 1) * mean.reshape(1, value_dim)
            if noise_std > 0.0:
                right_base = right_base + rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=right_base.shape,
                ).astype(np.float32)
                left_base = left_base + rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=left_base.shape,
                ).astype(np.float32)
            right_rand = right_base.astype(np.float32, copy=False)
            left_rand = left_base.astype(np.float32, copy=False)

        if preserve_wall_corners:
            right_rand = _preserve_wall_corner_values(
                right_rand, branch[interface_id - 1], "right", layout
            )
            left_rand = _preserve_wall_corner_values(
                left_rand, branch[interface_id], "left", layout
            )

        _set_edge_profile(branch[interface_id - 1], layout.right, layout.value_channels, right_rand)
        _set_edge_profile(branch[interface_id], layout.left, layout.value_channels, left_rand)

    return branch


@torch.no_grad()
def predict_all_subdomains_physical(
    model: DeepONet,
    branch_inputs: Array,
    query_coords: Array,
    device: Union[str, torch.device],
    branch_normalizer: Optional[FeatureNormalizer] = None,
    y_normalizer: Optional[FeatureNormalizer] = None,
    query_normalizer: Optional[FeatureNormalizer] = None,
    sample_batch_size: int = 8,
    query_batch_size: int = 32768,
) -> Array:
    """
    Predict all selected subdomains on the full query grid in physical units.

    Returns
    -------
    pred_flat: ndarray, shape (S, P, C_out)
    """
    model.eval()
    device = torch.device(device)

    branch_np = np.asarray(branch_inputs, dtype=np.float32)
    query_np = np.asarray(query_coords, dtype=np.float32)
    if branch_np.ndim != 3:
        raise ValueError(f"branch_inputs must have shape (S,M,C), got {branch_np.shape}")
    if query_np.ndim != 2:
        raise ValueError(f"query_coords must have shape (P,Cq), got {query_np.shape}")

    if branch_normalizer is not None:
        branch_normalizer = branch_normalizer.to(device)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
    if query_normalizer is not None:
        query_normalizer = query_normalizer.to(device)

    s_total = int(branch_np.shape[0])
    p_total = int(query_np.shape[0])
    outputs: Optional[Array] = None

    q_cpu = torch.as_tensor(query_np, dtype=torch.float32)

    for s0 in range(0, s_total, int(sample_batch_size)):
        s1 = min(s0 + int(sample_batch_size), s_total)
        b = torch.as_tensor(branch_np[s0:s1], dtype=torch.float32, device=device)
        if branch_normalizer is not None:
            b = branch_normalizer.encode(b)

        batch_preds = []
        for q0 in range(0, p_total, int(query_batch_size)):
            q1 = min(q0 + int(query_batch_size), p_total)
            q = q_cpu[q0:q1]
            if query_normalizer is not None:
                q = query_normalizer.encode(q)
            q = q.unsqueeze(0).expand(s1 - s0, -1, -1).to(device)
            pred = model(b, q)
            if y_normalizer is not None:
                pred = y_normalizer.decode(pred)
            batch_preds.append(pred.detach().cpu())

        batch_pred = torch.cat(batch_preds, dim=1).numpy().astype(np.float32, copy=False)
        if outputs is None:
            outputs = np.empty((s_total, p_total, batch_pred.shape[-1]), dtype=np.float32)
        outputs[s0:s1] = batch_pred

    if outputs is None:
        raise RuntimeError("No predictions were produced.")
    return outputs


def _flatten_to_grid(pred_flat: Array, nx: int, ny: int) -> Array:
    pred_flat = np.asarray(pred_flat, dtype=np.float32)
    if pred_flat.ndim != 3:
        raise ValueError(f"pred_flat must have shape (S,P,C), got {pred_flat.shape}")
    expected_p = int(nx) * int(ny)
    if pred_flat.shape[1] != expected_p:
        raise ValueError(
            f"pred_flat point count {pred_flat.shape[1]} does not match nx*ny={expected_p}."
        )
    return pred_flat.reshape(pred_flat.shape[0], int(nx), int(ny), pred_flat.shape[-1])


def compute_interface_mse(
    pred_grid: Array,
    y_normalizer: Optional[FeatureNormalizer] = None,
    mse_mode: str = "normalized",
) -> Tuple[Array, Array]:
    """
    Compute interface mismatch between adjacent predicted subdomains.

    Parameters
    ----------
    pred_grid:
        Shape (S, nx, ny, C). Physical units.
    y_normalizer:
        Used only when mse_mode='normalized'. The mean cancels, so only std is
        used to scale the difference.

    Returns
    -------
    mse_total:
        Shape (S-1,), mean over y and channels.
    mse_by_channel:
        Shape (S-1, C), mean over y for each channel.
    """
    pred_grid = np.asarray(pred_grid, dtype=np.float32)
    if pred_grid.ndim != 4:
        raise ValueError(f"pred_grid must have shape (S,nx,ny,C), got {pred_grid.shape}")
    if pred_grid.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, pred_grid.shape[-1]), dtype=np.float32)

    right_edges = pred_grid[:-1, -1, :, :]
    left_edges = pred_grid[1:, 0, :, :]
    diff = right_edges - left_edges

    if str(mse_mode).lower() == "normalized" and y_normalizer is not None:
        std = y_normalizer.std.detach().cpu().numpy().astype(np.float32)
        diff = diff / np.maximum(std.reshape(1, 1, -1), 1.0e-12)

    mse_by_channel = np.mean(diff ** 2, axis=1)
    mse_total = np.mean(diff ** 2, axis=(1, 2))
    return mse_total.astype(np.float32), mse_by_channel.astype(np.float32)


def update_branch_interfaces_from_predictions(
    current_branch: Array,
    pred_grid: Array,
    layout: EdgeLayout,
    relaxation: float = 1.0,
    preserve_wall_corners: bool = False,
) -> Array:
    """
    Update all interior interfaces simultaneously from predicted edge averages.

    For interface i between subdomain i-1 and i:

        new_profile = 0.5 * (pred_right_edge[i-1] + pred_left_edge[i])

    The same new_profile is assigned to both branch sides.
    """
    branch = np.asarray(current_branch, dtype=np.float32)
    pred_grid = np.asarray(pred_grid, dtype=np.float32)
    if pred_grid.ndim != 4:
        raise ValueError(f"pred_grid must have shape (S,nx,ny,C), got {pred_grid.shape}")
    if branch.shape[0] != pred_grid.shape[0]:
        raise ValueError("branch and pred_grid have different subdomain counts")

    updated = branch.copy()
    n_sub = pred_grid.shape[0]
    ny = pred_grid.shape[2]
    out_dim = pred_grid.shape[-1]

    if layout.left.size != ny or layout.right.size != ny:
        raise ValueError(
            "This evaluator assumes one vertical interface sensor per y grid point. "
            f"Got left={layout.left.size}, right={layout.right.size}, ny={ny}."
        )
    if len(layout.value_channels) != out_dim:
        raise ValueError(
            f"Branch value channel count {len(layout.value_channels)} does not match output dim {out_dim}."
        )

    relax = float(relaxation)
    for interface_id in range(1, n_sub):
        right_pred = pred_grid[interface_id - 1, -1, :, :]
        left_pred = pred_grid[interface_id, 0, :, :]
        mean_profile = 0.5 * (right_pred + left_pred)

        old_right = _edge_profile(updated[interface_id - 1], layout.right, layout.value_channels)
        old_left = _edge_profile(updated[interface_id], layout.left, layout.value_channels)

        new_right = (1.0 - relax) * old_right + relax * mean_profile
        new_left = (1.0 - relax) * old_left + relax * mean_profile

        if preserve_wall_corners:
            new_right = _preserve_wall_corner_values(
                new_right, updated[interface_id - 1], "right", layout
            )
            new_left = _preserve_wall_corner_values(
                new_left, updated[interface_id], "left", layout
            )

        _set_edge_profile(updated[interface_id - 1], layout.right, layout.value_channels, new_right)
        _set_edge_profile(updated[interface_id], layout.left, layout.value_channels, new_left)

    return updated.astype(np.float32, copy=False)


def stitch_subdomain_grids(
    subdomain_grids: Array,
    drop_duplicate_interfaces: bool = True,
) -> Array:
    """
    Stitch subdomain grids along x for plotting/saving.

    Parameters
    ----------
    subdomain_grids:
        Shape (S, nx, ny, C).
    drop_duplicate_interfaces:
        If True, remove the left column of every subdomain after the first.
        This gives stitched x-size S*(nx-1)+1.
    """
    grids = np.asarray(subdomain_grids, dtype=np.float32)
    if grids.ndim != 4:
        raise ValueError(f"subdomain_grids must have shape (S,nx,ny,C), got {grids.shape}")
    if not drop_duplicate_interfaces:
        return np.concatenate([grids[i] for i in range(grids.shape[0])], axis=0)
    pieces = [grids[0]] + [grids[i, 1:, :, :] for i in range(1, grids.shape[0])]
    return np.concatenate(pieces, axis=0)


def iterative_unknown_interface_inference(
    model: DeepONet,
    branch_inputs_for_ar: Array,
    query_coords: Array,
    nx: int,
    ny: int,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    branch_normalizer: Optional[FeatureNormalizer] = None,
    y_normalizer: Optional[FeatureNormalizer] = None,
    query_normalizer: Optional[FeatureNormalizer] = None,
    config: Optional[InterfaceIterationConfig] = None,
) -> Dict[str, object]:
    """
    Run simultaneous interface-averaging inference for one AR case.

    Parameters
    ----------
    branch_inputs_for_ar:
        Raw branch inputs for one AR, sorted by subdomain_id. Shape (S,M,Cb).
        Interior interface p/T/u/v values will be replaced before inference.
    query_coords:
        Raw trunk query coordinates, usually local [0,1]^2. Shape (nx*ny,2).
    nx, ny:
        Full grid resolution per subdomain.

    Returns
    -------
    dict with final predictions, final branch inputs, MSE history, and metadata.
    """
    if config is None:
        config = InterfaceIterationConfig()
    device = torch.device(device)

    raw_branch = np.asarray(branch_inputs_for_ar, dtype=np.float32)
    layout = infer_edge_layout(
        branch_template=raw_branch[0],
        branch_channel_names=branch_channel_names,
        nx=int(nx),
        ny=int(ny),
    )
    if config.init_mode == "dataset_truth":
        current_branch = raw_branch.copy()
    else:
        current_branch = initialize_unknown_interior_interfaces(
            branch_inputs_for_ar=raw_branch,
            layout=layout,
            branch_normalizer=branch_normalizer,
            random_seed=config.random_seed,
            init_mode=config.init_mode,
            init_std_scale=config.init_std_scale,
            init_value=config.init_value,
            init_noise_std=config.init_noise_std,
            init_gaussian_center_y=config.init_gaussian_center_y,
            init_gaussian_var_y=config.init_gaussian_var_y,
            preserve_wall_corners=config.preserve_wall_corners,
        )
    initial_random_branch = current_branch.copy()

    mse_history = []
    mse_by_channel_history = []
    converged_history = []
    final_pred_grid: Optional[Array] = None
    final_pred_flat: Optional[Array] = None
    converged = False
    n_iter_done = 0

    for iteration in range(config.max_iter + 1):
        pred_flat = predict_all_subdomains_physical(
            model=model,
            branch_inputs=current_branch,
            query_coords=query_coords,
            device=device,
            branch_normalizer=branch_normalizer,
            y_normalizer=y_normalizer,
            query_normalizer=query_normalizer,
            sample_batch_size=config.sample_batch_size,
            query_batch_size=config.query_batch_size,
        )
        pred_grid = _flatten_to_grid(pred_flat, nx=int(nx), ny=int(ny))
        mse_total, mse_by_channel = compute_interface_mse(
            pred_grid=pred_grid,
            y_normalizer=y_normalizer,
            mse_mode=config.mse_mode,
        )
        is_converged = mse_total <= config.tol
        all_converged = bool(np.all(is_converged))

        mse_history.append(mse_total)
        mse_by_channel_history.append(mse_by_channel)
        converged_history.append(is_converged)

        final_pred_flat = pred_flat
        final_pred_grid = pred_grid
        converged = all_converged
        n_iter_done = iteration

        if config.verbose:
            max_mse = float(np.max(mse_total)) if mse_total.size else 0.0
            mean_mse = float(np.mean(mse_total)) if mse_total.size else 0.0
            n_conv = int(np.count_nonzero(is_converged))
            n_int = int(is_converged.size)
            print(
                f"iter={iteration:04d} | "
                f"interface MSE max={max_mse:.6e}, mean={mean_mse:.6e} | "
                f"converged={n_conv}/{n_int}",
                flush=True,
            )

        if all_converged:
            break
        if iteration >= config.max_iter:
            break

        current_branch = update_branch_interfaces_from_predictions(
            current_branch=current_branch,
            pred_grid=pred_grid,
            layout=layout,
            relaxation=config.relaxation,
            preserve_wall_corners=config.preserve_wall_corners,
        )

    if final_pred_grid is None or final_pred_flat is None:
        raise RuntimeError("Inference did not produce predictions.")

    mse_hist_arr = np.stack(mse_history, axis=0) if mse_history else np.empty((0, 0), dtype=np.float32)
    mse_ch_hist_arr = (
        np.stack(mse_by_channel_history, axis=0)
        if mse_by_channel_history
        else np.empty((0, 0, 0), dtype=np.float32)
    )
    conv_hist_arr = (
        np.stack(converged_history, axis=0)
        if converged_history
        else np.empty((0, 0), dtype=bool)
    )

    return {
        "pred_flat": final_pred_flat,
        "pred_grid": final_pred_grid,
        "pred_stitched": stitch_subdomain_grids(final_pred_grid),
        "branch_initial_random": initial_random_branch,
        "branch_final": current_branch,
        "interface_mse_history": mse_hist_arr,
        "interface_mse_by_channel_history": mse_ch_hist_arr,
        "interface_converged_history": conv_hist_arr,
        "converged": bool(converged),
        "n_iter": int(n_iter_done),
        "config": config,
        "layout": layout,
    }


def final_truth_metrics_if_available(
    result: Mapping[str, object],
    outputs_for_ar: Optional[Array],
) -> Dict[str, Array]:
    """Optional helper for test diagnostics. Not used by the interface update."""
    if outputs_for_ar is None:
        return {}
    pred = np.asarray(result["pred_grid"], dtype=np.float32)
    truth = np.asarray(outputs_for_ar, dtype=np.float32)
    if truth.shape != pred.shape:
        raise ValueError(f"truth shape {truth.shape} does not match pred shape {pred.shape}")
    diff = pred - truth
    mse_by_channel = np.mean(diff ** 2, axis=(0, 1, 2))
    rel_l2_by_channel = np.sqrt(
        np.sum(diff ** 2, axis=(0, 1, 2))
        / np.maximum(np.sum(truth ** 2, axis=(0, 1, 2)), 1.0e-12)
    )
    return {
        "truth_mse_by_channel": mse_by_channel.astype(np.float32),
        "truth_rel_l2_by_channel": rel_l2_by_channel.astype(np.float32),
    }


def save_iteration_result_npz(
    path: PathLike,
    result: Mapping[str, object],
    dataset: Optional[Mapping[str, object]] = None,
    sample_indices: Optional[Array] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Save the main outputs of an iterative inference run."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, object] = {
        "pred_grid": np.asarray(result["pred_grid"], dtype=np.float32),
        "pred_stitched": np.asarray(result["pred_stitched"], dtype=np.float32),
        "branch_initial_random": np.asarray(result["branch_initial_random"], dtype=np.float32),
        "branch_final": np.asarray(result["branch_final"], dtype=np.float32),
        "interface_mse_history": np.asarray(result["interface_mse_history"], dtype=np.float32),
        "interface_mse_by_channel_history": np.asarray(
            result["interface_mse_by_channel_history"], dtype=np.float32
        ),
        "interface_converged_history": np.asarray(
            result["interface_converged_history"], dtype=bool
        ),
        "converged": np.asarray(bool(result["converged"])),
        "n_iter": np.asarray(int(result["n_iter"])),
    }

    if dataset is not None:
        for key in [
            "branch_channel_names",
            "trunk_channel_names",
            "output_channel_names",
        ]:
            if key in dataset:
                arrays[key] = np.asarray(dataset[key], dtype=object)
        if sample_indices is not None:
            sample_indices = np.asarray(sample_indices, dtype=np.int64)
            arrays["sample_indices"] = sample_indices
            for key in [
                "aspect_ratio",
                "subdomain_id",
                "x_left_mm",
                "x_right_mm",
                "y_bottom_mm",
                "y_top_mm",
                "interface_realization_id",
            ]:
                if key in dataset:
                    arrays[key] = np.asarray(dataset[key])[sample_indices]

    if extra is not None:
        for key, value in extra.items():
            arrays[str(key)] = value

    np.savez_compressed(path, **arrays)

def optimize_interfaces_flux_matching(
    model: torch.nn.Module,
    branch_inputs_for_ar: np.ndarray,
    query_coords: np.ndarray,
    nx: int,
    ny: int,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    branch_normalizer=None,
    y_normalizer=None,
    query_normalizer=None,
    alpha: float = 1.0,  # Weight for Dirichlet (Value) loss
    beta: float = 0.1,   # Weight for Neumann (Flux) loss
    max_iter: int = 200,
    lr: float = 0.05
) -> Dict[str, object]:
    """
    Inference-time optimization to match Dirichlet and Neumann conditions across
    interfaces. This completely replaces the Jacobi iteration.
    """
    layout = infer_edge_layout(
        branch_template=branch_inputs_for_ar[0], 
        branch_channel_names=branch_channel_names,
        nx=int(nx),
        ny=int(ny),
    )
    
    device = torch.device(device)
    model.eval()
    
    # We freeze the model weights completely
    for param in model.parameters():
        param.requires_grad = False

    S = branch_inputs_for_ar.shape[0]
    if S < 2:
        raise ValueError("Optimization requires at least 2 subdomains.")

    # 1. Isolate the specific query coordinates for the left and right interfaces
    q_np = np.asarray(query_coords, dtype=np.float32)
    
    left_q_idx = np.where(np.abs(q_np[:, layout.x_channel] - 0.0) < 1e-6)[0]
    right_q_idx = np.where(np.abs(q_np[:, layout.x_channel] - 1.0) < 1e-6)[0]
    
    # Ensure they are sorted by y so the physical points align
    left_q_idx = left_q_idx[np.argsort(q_np[left_q_idx, layout.y_channel])]
    right_q_idx = right_q_idx[np.argsort(q_np[right_q_idx, layout.y_channel])]

    q_left = torch.tensor(q_np[left_q_idx], dtype=torch.float32, device=device)
    q_right = torch.tensor(q_np[right_q_idx], dtype=torch.float32, device=device)

    # 2. Setup the Optimization Parameter
    # We optimize one shared profile per interface. Shape: (S-1, ny, num_value_channels)
    num_val_ch = len(layout.value_channels)
    base_branch = torch.tensor(branch_inputs_for_ar, dtype=torch.float32, device=device)
    
    # Initialize with the dataset mean or zeros
    mean_vals = base_branch[:, :, layout.value_channels].mean(dim=(0, 1))
    V_shared = mean_vals.view(1, 1, -1).expand(S - 1, len(left_q_idx), num_val_ch).clone()
    V_shared.requires_grad = True

    # Using Adam. L-BFGS is mathematically faster here but computing 3rd-order 
    # derivatives (Hessian of a gradient penalty) can crash PyTorch.
    optimizer = optim.Adam([V_shared], lr=lr)

    loss_history = []

    for step in range(max_iter):
        optimizer.zero_grad()

        # 3. Inject V_shared into the branch tensor (differentiable assignment)
        current_branch = base_branch.clone()
        for c_idx, ch in enumerate(layout.value_channels):
            for i in range(1, S):
                # Update right edge of domain i-1 and left edge of domain i with the SAME shared value
                current_branch[i-1, layout.right, ch] = V_shared[i-1, :, c_idx]
                current_branch[i, layout.left, ch] = V_shared[i-1, :, c_idx]

        if branch_normalizer is not None:
            branch_normalizer = branch_normalizer.to(device)
            branch_encoded = branch_normalizer.encode(current_branch)
        else:
            branch_encoded = current_branch

        b_left_domains = branch_encoded[:-1] # Predicts right edge
        b_right_domains = branch_encoded[1:] # Predicts left edge

        # Expand query coordinates for the batch and ENABLE GRADIENTS for autograd
        q_r_exp = q_right.unsqueeze(0).expand(S-1, -1, -1).clone().detach().requires_grad_(True)
        q_l_exp = q_left.unsqueeze(0).expand(S-1, -1, -1).clone().detach().requires_grad_(True)

        if query_normalizer is not None:
            query_normalizer = query_normalizer.to(device)
            q_r_net = query_normalizer.encode(q_r_exp)
            q_l_net = query_normalizer.encode(q_l_exp)
        else:
            q_r_net = q_r_exp
            q_l_net = q_l_exp

        # 4. Forward Pass (ONLY on the interfaces, incredibly fast)
        pred_right = model(b_left_domains, q_r_net)
        pred_left = model(b_right_domains, q_l_net)

        if y_normalizer is not None:
            y_normalizer = y_normalizer.to(device)
            pred_right_phys = y_normalizer.decode(pred_right)
            pred_left_phys = y_normalizer.decode(pred_left)
        else:
            pred_right_phys = pred_right
            pred_left_phys = pred_left

        # 5. Compute Robin-like Loss
        # Part A: Dirichlet (Value mismatch)
        dirichlet_loss = torch.mean((pred_right_phys - pred_left_phys) ** 2)

        # Part B: Neumann (Flux mismatch)
        neumann_loss = 0.0
        for c in range(pred_right_phys.shape[-1]):
            # Extract spatial gradients using autograd
            grad_r = torch.autograd.grad(
                outputs=pred_right_phys[..., c],
                inputs=q_r_exp,
                grad_outputs=torch.ones_like(pred_right_phys[..., c]),
                create_graph=True,   # Required to backprop through the derivative
                retain_graph=True
            )[0]
            flux_r = grad_r[..., layout.x_channel]

            grad_l = torch.autograd.grad(
                outputs=pred_left_phys[..., c],
                inputs=q_l_exp,
                grad_outputs=torch.ones_like(pred_left_phys[..., c]),
                create_graph=True,
                retain_graph=True
            )[0]
            flux_l = grad_l[..., layout.x_channel]

            neumann_loss += torch.mean((flux_r - flux_l) ** 2)

        total_loss = alpha * dirichlet_loss + beta * neumann_loss
        
        # 6. Backprop and Step
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        
        if step % 10 == 0:
            print(f"Step {step:03d} | Total Loss: {total_loss.item():.6e} | Dir: {dirichlet_loss.item():.6e} | Neu: {neumann_loss.item():.6e}")

        if total_loss.item() < 1e-6:
            print(f"Converged at step {step}")
            break

    # Once optimized, evaluate the FULL internal grid one last time to get the final fields
    with torch.no_grad():
        final_branch = base_branch.clone()
        for c_idx, ch in enumerate(layout.value_channels):
            for i in range(1, S):
                final_branch[i-1, layout.right, ch] = V_shared[i-1, :, c_idx].detach()
                final_branch[i, layout.left, ch] = V_shared[i-1, :, c_idx].detach()
                
        # Use your existing full-grid predictor
        final_pred_flat = predict_all_subdomains_physical(
            model=model,
            branch_inputs=final_branch.cpu().numpy(),
            query_coords=query_coords,
            device=device,
            branch_normalizer=branch_normalizer,
            y_normalizer=y_normalizer,
            query_normalizer=query_normalizer
        )
        final_pred_grid = _flatten_to_grid(final_pred_flat, nx, ny)

    return {
        "pred_flat": final_pred_flat,
        "pred_grid": final_pred_grid,
        "pred_stitched": stitch_subdomain_grids(final_pred_grid),
        "branch_final": final_branch.cpu().numpy(),
        "loss_history": np.array(loss_history),
        "converged": total_loss.item() < 1e-6,
        "layout": layout,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative unknown-interface inference for Fluent DeepONet."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to base DeepONet checkpoint .pt")
    parser.add_argument("--dataset-h5", required=True, help="Path to saved DeepONet dataset .h5")
    parser.add_argument("--ar", type=int, required=True, help="Aspect ratio to evaluate, e.g. 50")
    parser.add_argument("--realization-id", type=int, default=None)
    parser.add_argument("--out", default="deeponet_iterative_interface_result.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--mse-mode", choices=["normalized", "physical"], default="normalized")
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-mode",
        choices=["pointwise_random", "gaussian_along_y"],
        default="pointwise_random",
        help=(
            "Interface initialization mode: independent random per point, "
            "or Gaussian-shaped profile along y."
        ),
    )
    parser.add_argument("--init-std-scale", type=float, default=1.0)
    parser.add_argument(
        "--init-value",
        type=float,
        default=None,
        help=(
            "Fixed initial interface value for all channels. "
            "If unset, initialization is sampled from dataset statistics."
        ),
    )
    parser.add_argument(
        "--init-noise-std",
        type=float,
        default=1.0e-3,
        help="Additive Gaussian noise std (used with --init-value and gaussian mode).",
    )
    parser.add_argument(
        "--init-gaussian-center-y",
        type=float,
        default=0.5,
        help="Center y location for --init-mode gaussian_along_y.",
    )
    parser.add_argument(
        "--init-gaussian-var-y",
        type=float,
        default=0.02,
        help="Variance in y for --init-mode gaussian_along_y.",
    )
    parser.add_argument("--query-batch-size", type=int, default=32768)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument(
        "--preserve-wall-corners",
        action="store_true",
        help="Keep interface endpoint values consistent with wall-corner sensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    dataset = load_deeponet_dataset_h5(args.dataset_h5)
    sample_indices = select_ar_subdomains(
        dataset,
        ar=args.ar,
        realization_id=args.realization_id,
    )

    model, branch_normalizer, y_normalizer, ckpt = load_base_deeponet_checkpoint(
        args.checkpoint,
        device=device,
    )

    query_normalizer = None
    if "query_normalizer" in ckpt and ckpt["query_normalizer"] is not None:
        query_normalizer = FeatureNormalizer.from_state_dict(ckpt["query_normalizer"]).to(device)

    config = InterfaceIterationConfig(
        max_iter=args.max_iter,
        tol=args.tol,
        mse_mode=args.mse_mode,
        relaxation=args.relaxation,
        random_seed=args.seed,
        init_mode=args.init_mode,
        init_std_scale=args.init_std_scale,
        init_value=args.init_value,
        init_noise_std=args.init_noise_std,
        init_gaussian_center_y=args.init_gaussian_center_y,
        init_gaussian_var_y=args.init_gaussian_var_y,
        query_batch_size=args.query_batch_size,
        sample_batch_size=args.sample_batch_size,
        preserve_wall_corners=args.preserve_wall_corners,
        verbose=True,
    )

    result = iterative_unknown_interface_inference(
        model=model,
        branch_inputs_for_ar=np.asarray(dataset["branch_inputs"])[sample_indices],
        query_coords=np.asarray(dataset["query_coords"]),
        nx=int(dataset["nx"]),
        ny=int(dataset["ny"]),
        branch_channel_names=dataset["branch_channel_names"],
        device=device,
        branch_normalizer=branch_normalizer,
        y_normalizer=y_normalizer,
        query_normalizer=query_normalizer,
        config=config,
    )

    truth = None
    if "outputs" in dataset and dataset["outputs"] is not None:
        truth = np.asarray(dataset["outputs"])[sample_indices]
    metrics = final_truth_metrics_if_available(result, truth)

    save_iteration_result_npz(
        args.out,
        result=result,
        dataset=dataset,
        sample_indices=sample_indices,
        extra=metrics,
    )

    print(f"Saved iterative inference result to {args.out}")
    print(f"Converged: {result['converged']} after {result['n_iter']} iterations")
    if metrics:
        print("Truth MSE by channel:", metrics["truth_mse_by_channel"])
        print("Truth relative L2 by channel:", metrics["truth_rel_l2_by_channel"])


if __name__ == "__main__":
    main()
