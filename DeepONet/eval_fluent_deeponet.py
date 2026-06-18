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
class InterfaceIterationConfig:
    max_iter: int = 100
    tol: float = 1.0e-6
    mse_mode: str = "normalized"  # "normalized" or "physical"
    relaxation: float = 1.0
    random_seed: int = 0

    # Interface initialization. Supported modes:
    #   pointwise_random      : every interface point/channel sampled independently
    #   fixed_with_noise      : fixed scalar/vector value plus pointwise noise
    #   interfacewise_random  : one random p/T/u/v vector per interface side, broadcast along y
    #   gaussian_along_y      : smooth Gaussian-shaped profile along y
    #   dataset_truth         : keep the supplied dataset interface values
    init_mode: str = "pointwise_random"
    init_std_scale: float = 1.0
    init_value: Optional[Union[float, Sequence[float]]] = None
    init_noise_std: float = 1.0e-3
    init_gaussian_center_y: float = 0.5
    init_gaussian_var_y: float = 0.02
    preserve_wall_corners: bool = False
    
    # Dimension 
    #   dct : Discrete Cosine Transform
    # (default) None : No dimensionality reduction
    reduce_mode: Optional[str] = None
    query_batch_size: int = 65536
    sample_batch_size: int = 8
    verbose: bool = True

    def __post_init__(self) -> None:
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)
        self.mse_mode = str(self.mse_mode).lower()
        self.relaxation = float(self.relaxation)
        self.random_seed = int(self.random_seed)
        self.init_mode = str(self.init_mode).lower()
        self.init_std_scale = float(self.init_std_scale)
        self.init_noise_std = float(self.init_noise_std)
        self.init_gaussian_center_y = float(self.init_gaussian_center_y)
        self.init_gaussian_var_y = float(self.init_gaussian_var_y)
        self.query_batch_size = int(self.query_batch_size)
        self.sample_batch_size = int(self.sample_batch_size)
        self.preserve_wall_corners = bool(self.preserve_wall_corners)
        self.reduce_mode = str(self.reduce_mode).lower() if self.reduce_mode else None
        
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

        allowed_init = {
            "pointwise_random",
            "fixed_with_noise",
            "interfacewise_random",
            "gaussian_along_y",
            "dataset_truth",
        }
        if self.mse_mode not in {"normalized", "physical"}:
            raise ValueError("mse_mode must be 'normalized' or 'physical'")
        if self.init_mode not in allowed_init:
            raise ValueError(
                "init_mode must be one of "
                "'pointwise_random', 'fixed_with_noise', "
                "'interfacewise_random', 'gaussian_along_y', or 'dataset_truth'."
            )
            
        allowed_reduce = {
            "dct",
            None,
        }
        if self.reduce_mode not in allowed_reduce:
            raise ValueError(
                "reduce_mode must be one of "
                "'dct' or None."
            )
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError("relaxation must satisfy 0 < relaxation <= 1")
        if self.max_iter < 0:
            raise ValueError("max_iter must be >= 0")
        if self.query_batch_size <= 0 or self.sample_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.init_noise_std < 0.0:
            raise ValueError("init_noise_std must be >= 0")
        if self.init_gaussian_var_y <= 0.0:
            raise ValueError("init_gaussian_var_y must be > 0")


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

    left_q = branch_raw[:, layout.left, :][:, :, [layout.x_channel, layout.y_channel]]
    right_q = branch_raw[:, layout.right, :][:, :, [layout.x_channel, layout.y_channel]]

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

    left_q = branch_raw[:, layout.left, :][:, :, [layout.x_channel, layout.y_channel]]
    right_q = branch_raw[:, layout.right, :][:, :, [layout.x_channel, layout.y_channel]]

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


def update_branch_interfaces_from_edge_predictions(
    current_branch: Array,
    pred_left: Array,
    pred_right: Array,
    layout: EdgeLayout,
    relaxation: float = 1.0,
) -> Array:
    """Jacobi update: shared interface gets mean of the two predicted traces."""
    updated = np.asarray(current_branch, dtype=np.float32).copy()
    n_sub = updated.shape[0]
    relax = float(relaxation)
    for interface_id in range(1, n_sub):
        mean_profile = 0.5 * (pred_right[interface_id - 1] + pred_left[interface_id])
        old_r = updated[interface_id - 1][np.ix_(layout.right, layout.value_channels)]
        old_l = updated[interface_id][np.ix_(layout.left, layout.value_channels)]
        new_r = (1.0 - relax) * old_r + relax * mean_profile
        new_l = (1.0 - relax) * old_l + relax * mean_profile
        _set_edge_profile(updated[interface_id - 1], layout.right, layout.value_channels, new_r)
        _set_edge_profile(updated[interface_id], layout.left, layout.value_channels, new_l)
    return updated


def iterative_unknown_interface_inference(
    model: DeepONet,
    samples: Sequence[Mapping[str, Array]],
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    config: Optional[InterfaceIterationConfig] = None,
) -> Dict[str, object]:
    """Run simultaneous unknown-interface inference for variable cell-center samples."""
    if config is None:
        config = InterfaceIterationConfig()
    branches = np.stack([np.asarray(s["branch"], dtype=np.float32) for s in samples], axis=0)
    layout = infer_edge_layout(branches[0], branch_channel_names)
    if config.init_mode == "dataset_truth":
        current_branch = branches.copy()
    else:
        current_branch = initialize_unknown_interior_interfaces(
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
    initial_branch = current_branch.copy()

    mse_history = []
    mse_by_channel_history = []
    converged_history = []
    interface_left_history = []
    interface_right_history = []
    final_pred_left = None
    final_pred_right = None

    converged = False
    n_iter_done = 0
    for iteration in range(1, config.max_iter + 1):
        pred_left, pred_right = predict_edge_profiles(
            model=model,
            branch_inputs=current_branch,
            layout=layout,
            branch_channel_names=branch_channel_names,
            device=device,
            y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
        )
        mse_total, mse_by_channel = compute_interface_mse_from_edges(
            pred_left,
            pred_right,
        )
        is_converged = mse_total <= config.tol
        all_converged = bool(np.all(is_converged))
        mse_history.append(mse_total)
        mse_by_channel_history.append(mse_by_channel)
        converged_history.append(is_converged)
        final_pred_left = pred_left
        final_pred_right = pred_right
        converged = all_converged
        n_iter_done = iteration
        if config.verbose:
            max_mse = float(np.max(mse_total)) if mse_total.size else 0.0
            mean_mse = float(np.mean(mse_total)) if mse_total.size else 0.0
            print(f"iter={iteration:04d} | interface MSE max={max_mse:.6e}, mean={mean_mse:.6e} | converged={np.count_nonzero(is_converged)}/{is_converged.size}", flush=True)
        if not (all_converged or iteration >= config.max_iter):
            current_branch = update_branch_interfaces_from_edge_predictions(
                current_branch=current_branch,
                pred_left=pred_left,
                pred_right=pred_right,
                layout=layout,
                relaxation=config.relaxation,
            )

        iter_left_vals, iter_right_vals = _interface_value_profiles_from_branch(current_branch, layout)
        interface_left_history.append(iter_left_vals)
        interface_right_history.append(iter_right_vals)
        if all_converged or iteration >= config.max_iter:
            break

    pred_samples = predict_cell_samples_physical(
        model=model,
        samples=samples,
        branch_inputs=current_branch,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        query_batch_size=config.query_batch_size,
    )

    return {
        "pred_samples": pred_samples,
        "branch_initial_random": initial_branch,
        "branch_final": current_branch,
        "pred_left_interface": final_pred_left,
        "pred_right_interface": final_pred_right,
        "interface_mse_history": np.stack(mse_history, axis=0),
        "interface_mse_by_channel_history": np.stack(mse_by_channel_history, axis=0),
        "interface_converged_history": np.stack(converged_history, axis=0),
        "interface_left_values_history": np.stack(interface_left_history, axis=0),
        "interface_right_values_history": np.stack(interface_right_history, axis=0),
        "converged": bool(converged),
        "n_iter": int(n_iter_done),
        "layout": layout,
        "config": config,
    }
    
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


def _annealing_objective_from_mse(mse_total: Array, objective: str = "max") -> float:
    """Convert per-interface MSE values to a scalar annealing energy."""
    mse = np.asarray(mse_total, dtype=np.float64).reshape(-1)
    if mse.size == 0:
        return 0.0
    objective = str(objective).lower()
    if objective == "mean":
        return float(np.mean(mse))
    if objective == "sum":
        return float(np.sum(mse))
    if objective == "max":
        return float(np.max(mse))
    raise ValueError("objective must be 'max', 'mean', or 'sum'")


@torch.no_grad()
def simulated_annealing_interface_energy(
    z: Array,
    base_branch: Array,
    model: DeepONet,
    layout: EdgeLayout,
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    mse_mode: str = "normalized",
    objective: str = "max",
    reduce_mode: Optional[str] = None,
) -> Tuple[float, Dict[str, object], Array]:
    """
    Simulated-annealing objective that minimizes predicted interface mismatch.

    The scalar energy combines value mismatch (Dirichlet) and first-order
    derivative mismatch dp/dx (Neumann) across interior interfaces.  No
    smoothness penalty, range penalty, or truth penalty is included.
    """
    branch = apply_interior_interface_z(
        base_branch=base_branch,
        z=z,
        layout=layout,
        y_normalizer=y_normalizer,
        reduce_mode=reduce_mode,
    )
    pred_left, pred_right = predict_edge_profiles(
        model=model,
        branch_inputs=branch,
        layout=layout,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
    )
    mse_total, mse_by_channel = compute_interface_mse_from_edges(
        pred_left,
        pred_right,
    )
    grad_left = None
    grad_right = None
    grad_mse_total = np.zeros_like(mse_total)
    grad_mse_by_channel = np.zeros_like(mse_by_channel)
    # grad_left, grad_right = predict_edge_grad(
    #     model=model,
    #     branch_inputs=branch,
    #     layout=layout,
    #     branch_channel_names=branch_channel_names,
    #     device=device,
    #     y_normalizer=y_normalizer,
    #     local_aspect_mean=local_aspect_mean,
    #     local_aspect_std=local_aspect_std,
    # )
    # grad_mse_total, grad_mse_by_channel = compute_interface_mse_from_edges(
    #     grad_left,
    #     grad_right,
    # )
    combined_mse = mse_total + 0 * grad_mse_total
    energy = _annealing_objective_from_mse(combined_mse, objective=objective)
    details: Dict[str, object] = {
        "energy": float(energy),
        "mse_total": mse_total,
        "mse_by_channel": mse_by_channel,
        "grad_mse_total": grad_mse_total,
        "grad_mse_by_channel": grad_mse_by_channel,
        "pred_left": pred_left,
        "pred_right": pred_right,
        "grad_left": grad_left,
        "grad_right": grad_right,
    }
    return float(energy), details, branch


def _propose_interface_z(
    z: Array,
    rng: np.random.Generator,
    proposal_sigma: float,
    proposal_location: str = "channel",
    sample_mode: str = "fixed",
    reduce_mode: Optional[str] = None,
    z_clip: Optional[float] = None,
) -> Array:
    """
    Propose a new normalized interface state for simulated annealing.

    proposal_location options:
        all                : perturb every interface point/channel
        interface          : perturb one whole interface, all channels
        channel            : perturb one whole interface, one channel
        point              : perturb one interface point and one channel
    
    sample_mode options:
        fixed              : sample fixed value for all interface/channel points
        random             : sample a random value for each interface/channel point
    """
    z_new = np.asarray(z, dtype=np.float32).copy()
    if z_new.size == 0:
        return z_new

    location = str(proposal_location).lower()
    mode = str(sample_mode).lower()
    sigma = float(proposal_sigma)
    n_if, n_y, n_ch = z_new.shape

    if reduce_mode == "dct":
        mode_prob = 1 / (np.arange(n_y) + 1)
        mode_prob = mode_prob / mode_prob.sum()
        mode_sigma = 1 / (np.arange(n_y) ** 2 + 1)
        
        i = int(rng.integers(0, n_if))
        for c in range(n_ch):
            j = int(rng.choice(n_y, p=mode_prob))
            z_new[i, j, c] += rng.normal(0.0, mode_sigma[j])

    elif reduce_mode is None:
        if location == "all":
            if mode == "random":
                z_new += rng.normal(0.0, sigma, size=z_new.shape).astype(np.float32)
            elif mode == "fixed":
                delta = rng.normal(0.0, sigma, size=(n_ch,))
                z_new += delta.reshape(1, 1, -1)
        elif location == "interface":
            i = int(rng.integers(0, n_if))
            if mode == "random":
                z_new[i, :, :] += rng.normal(0.0, sigma, size=(n_y, n_ch)).astype(np.float32)
            elif mode == "fixed":
                delta = rng.normal(0.0, sigma, size=(n_ch,))
                z_new[i, :, :] += delta.reshape(1, -1)
        elif location == "channel":
            i = int(rng.integers(0, n_if))
            c = int(rng.integers(0, n_ch))
            if mode == "random":
                z_new[i, :, c] += rng.normal(0.0, sigma, size=(n_y,)).astype(np.float32)
            elif mode == "fixed":
                z_new[i, :, c] += rng.normal(0.0, sigma)
        elif location == "point":
            i = int(rng.integers(0, n_if))
            j = int(rng.integers(0, n_y))
            c = int(rng.integers(0, n_ch))
            z_new[i, j, c] += np.float32(rng.normal(0.0, sigma))
        else:
            raise ValueError(
                "proposal_mode must be 'all', 'interface', 'interface_channel', or 'point_channel'"
            )

    if z_clip is not None:
        clip = abs(float(z_clip))
        z_new = np.clip(z_new, -clip, clip)
    return z_new


def _geometric_schedule(start: float, stop: float, frac: float) -> float:
    """Geometric interpolation from start to stop for frac in [0,1]."""
    start = float(start)
    stop = float(stop)
    frac = float(np.clip(frac, 0.0, 1.0))
    if start <= 0.0 or stop <= 0.0:
        return start + frac * (stop - start)
    return start * (stop / start) ** frac


@torch.no_grad()
def simulated_annealing_unknown_interface_inference(
    model: DeepONet,
    samples: Sequence[Mapping[str, Array]],
    branch_channel_names: Sequence[str],
    device: Union[str, torch.device],
    y_normalizer: FeatureNormalizer,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    config: Optional[InterfaceIterationConfig] = None,
    temperature0: Optional[float] = None,
    temperature_min: float = 1.0e-8,
    proposal_sigma0: float = 5.0e-2,
    proposal_sigma_min: float = 1.0e-3,
    proposal_location: str = "channel",
    sample_mode: str = "fixed",
    objective: str = "max",
    z_clip: Optional[float] = None,
    verbose_every: int = 50,
) -> Dict[str, object]:
    """
    Simulated annealing inference for unknown interior interfaces.

    Interface states are proposed in normalized output space, decoded back to 
    physical units, and written to both sides of each shared interface before 
    model evaluation.
    """
    if config is None:
        config = InterfaceIterationConfig()
    rng = np.random.default_rng(int(config.random_seed))
    n_steps = int(config.max_iter)
    if n_steps < 0:
        raise ValueError("number of iterations must be >= 0")
    if proposal_sigma0 < 0.0 or proposal_sigma_min < 0.0:
        raise ValueError("proposal sigmas must be non-negative")
    reduce_mode = config.reduce_mode

    branches = np.stack([np.asarray(s["branch"], dtype=np.float32) for s in samples], axis=0)
    layout = infer_edge_layout(branches[0], branch_channel_names)

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

    z = pack_interior_interface_z(
        init_branch,
        layout=layout,
        y_normalizer=y_normalizer,
        reduce_mode=reduce_mode,
    )

    current_energy, current_details, current_branch = simulated_annealing_interface_energy(
        z=z,
        base_branch=branches,
        model=model,
        layout=layout,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        mse_mode=config.mse_mode,
        objective=objective,
        reduce_mode=reduce_mode,
    )

    print(f"Initial energy: {current_energy:.6e}")
    if temperature0 is None:
        print("Initializing temperature")
        p_de = []
        for _ in range(100):
            z_prop = _propose_interface_z(
                z=z,
                rng=rng,
                proposal_sigma=proposal_sigma0,
                proposal_location=proposal_location,
                sample_mode=sample_mode,
                z_clip=z_clip,
                reduce_mode=reduce_mode,
            )
            prop_e, _, _, = simulated_annealing_interface_energy(
                z=z_prop,
                base_branch=branches,
                model=model,
                layout=layout,
                branch_channel_names=branch_channel_names,
                device=device,
                y_normalizer=y_normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                mse_mode=config.mse_mode,
                objective=objective,
                reduce_mode=reduce_mode,
            )
            if prop_e > current_energy:
                p_de.append(prop_e - current_energy)
                
        temperature0 = float(-np.mean(p_de) / np.log(0.9))
        print(f"Initial temperature: {temperature0:.6e}")
        

    best_z = z.copy()
    best_energy = float(current_energy)
    best_branch = current_branch.copy()
    best_details = dict(current_details)

    energy_history = [float(current_energy)]
    best_energy_history = [float(best_energy)]
    accepted_history = [True]
    temperature_history = [float(temperature0)]
    proposal_sigma_history = [float(proposal_sigma0)]
    mse_history = [np.asarray(current_details["mse_total"], dtype=np.float32)]
    mse_by_channel_history = [np.asarray(current_details["mse_by_channel"], dtype=np.float32)]
    best_mse_history = [np.asarray(best_details["mse_total"], dtype=np.float32)]
    converged_history = [np.asarray(current_details["mse_total"], dtype=np.float32) <= float(config.tol)]

    for step in range(1, n_steps + 1):
        frac = step / max(n_steps, 1)
        temperature = _geometric_schedule(float(temperature0), float(temperature_min), frac)
        proposal_sigma = _geometric_schedule(float(proposal_sigma0), float(proposal_sigma_min), frac)

        z_prop = _propose_interface_z(
            z=z,
            rng=rng,
            proposal_sigma=proposal_sigma,
            proposal_location=proposal_location,
            sample_mode=sample_mode,
            z_clip=z_clip,
            reduce_mode=reduce_mode,
        )
        prop_energy, prop_details, prop_branch = simulated_annealing_interface_energy(
            z=z_prop,
            base_branch=branches,
            model=model,
            layout=layout,
            branch_channel_names=branch_channel_names,
            device=device,
            y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            mse_mode=config.mse_mode,
            objective=objective,
            reduce_mode=reduce_mode,
        )

        delta_e = float(prop_energy) - float(current_energy)
        if delta_e <= 0.0:
            accepted = True
        else:
            p_accept = np.exp(-delta_e / max(float(temperature), 1.0e-30))
            accepted = bool(rng.random() < p_accept)

        if accepted:
            z = z_prop
            current_energy = float(prop_energy)
            current_details = prop_details
            current_branch = prop_branch

            if current_energy < best_energy:
                best_z = z.copy()
                best_energy = float(current_energy)
                best_branch = current_branch.copy()
                best_details = dict(current_details)

        current_mse = np.asarray(current_details["mse_total"], dtype=np.float32)
        current_mse_by_channel = np.asarray(current_details["mse_by_channel"], dtype=np.float32)
        is_converged = current_mse <= float(config.tol)

        energy_history.append(float(current_energy))
        best_energy_history.append(float(best_energy))
        accepted_history.append(bool(accepted))
        temperature_history.append(float(temperature))
        proposal_sigma_history.append(float(proposal_sigma))
        mse_history.append(current_mse)
        mse_by_channel_history.append(current_mse_by_channel)
        best_mse_history.append(np.asarray(best_details["mse_total"], dtype=np.float32))
        converged_history.append(is_converged)

        if config.verbose and (step % max(int(verbose_every), 1) == 0 or step == n_steps):
            max_mse = float(np.max(current_mse)) if current_mse.size else 0.0
            mean_mse = float(np.mean(current_mse)) if current_mse.size else 0.0
            best_max_mse = float(np.max(best_details["mse_total"])) if len(best_details["mse_total"]) else 0.0
            print(
                f"sa_step={step:05d} | E={current_energy:.6e} | "
                f"best_E={best_energy:.6e} | max_mse={max_mse:.6e} | "
                f"mean_mse={mean_mse:.6e} | best_max_mse={best_max_mse:.6e} | "
                f"T={temperature:.3e} | sigma={proposal_sigma:.3e} | accepted={accepted}",
                flush=True,
            )

        if bool(np.all(is_converged)):
            if config.verbose:
                print(f"Simulated annealing converged at step {step}.", flush=True)
            break

    pred_samples = predict_cell_samples_physical(
        model=model,
        samples=samples,
        branch_inputs=best_branch,
        branch_channel_names=branch_channel_names,
        device=device,
        y_normalizer=y_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        query_batch_size=config.query_batch_size,
    )

    best_mse = np.asarray(best_details["mse_total"], dtype=np.float32)
    converged = bool(np.all(best_mse <= float(config.tol)))

    return {
        "pred_samples": pred_samples,
        "branch_initial": init_branch,
        "branch_initial_random": init_branch,
        "branch_final": best_branch,
        "best_z": best_z,
        "pred_left_interface": best_details["pred_left"],
        "pred_right_interface": best_details["pred_right"],
        "interface_mse_history": np.stack(mse_history, axis=0),
        "interface_mse_by_channel_history": np.stack(mse_by_channel_history, axis=0),
        "interface_converged_history": np.stack(converged_history, axis=0),
        "best_interface_mse_history": np.stack(best_mse_history, axis=0),
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "best_energy_history": np.asarray(best_energy_history, dtype=np.float64),
        "accepted_history": np.asarray(accepted_history, dtype=bool),
        "temperature_history": np.asarray(temperature_history, dtype=np.float64),
        "proposal_sigma_history": np.asarray(proposal_sigma_history, dtype=np.float64),
        "best_energy": float(best_energy),
        "converged": bool(converged),
        "n_iter": int(len(energy_history) - 1),
        "layout": layout,
        "config": config,
        "annealing_params": {
            "n_steps": int(n_steps),
            "temperature0": float(temperature0),
            "temperature_min": float(temperature_min),
            "proposal_sigma0": float(proposal_sigma0),
            "proposal_sigma_min": float(proposal_sigma_min),
            "proposal_location": str(proposal_location),
            "sample_mode": str(sample_mode),
            "objective": str(objective),
            "z_clip": None if z_clip is None else float(z_clip),
        },
    }


def save_iteration_result_npz(path: PathLike, result: Mapping[str, object], sample_indices: Optional[Array] = None) -> None:
    """Save main variable-length inference outputs. pred_samples is saved as object array."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, object] = {
        "pred_samples": np.asarray(result["pred_samples"], dtype=object),
        "branch_initial_random": np.asarray(result["branch_initial_random"], dtype=np.float32),
        "branch_final": np.asarray(result["branch_final"], dtype=np.float32),
        "interface_mse_history": np.asarray(result["interface_mse_history"], dtype=np.float32),
        "interface_mse_by_channel_history": np.asarray(result["interface_mse_by_channel_history"], dtype=np.float32),
        "interface_converged_history": np.asarray(result["interface_converged_history"], dtype=bool),
        "interface_left_values_history": np.asarray(result["interface_left_values_history"], dtype=np.float32),
        "interface_right_values_history": np.asarray(result["interface_right_values_history"], dtype=np.float32),
        "converged": np.asarray(bool(result["converged"])),
        "n_iter": np.asarray(int(result["n_iter"])),
    }
    if sample_indices is not None:
        arrays["sample_indices"] = np.asarray(sample_indices, dtype=np.int64)
    np.savez_compressed(path, **arrays)

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