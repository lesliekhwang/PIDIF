"""
DeepONet utilities for Fluent-HDF5 subdomain datasets.

This file is intentionally separate from the FNO pipeline files.  It reuses the
existing Fluent reader/build_case_subdomains function, but it does not modify any
of those files.

Data representation
-------------------
Each subdomain is represented by:

    branch_inputs: (n_samples, n_boundary_points, n_branch_features)
    query_coords:   (nx * ny, n_trunk_features)
    outputs:        (n_samples, nx, ny, n_output_fields)

Branch point features do not contain left/right-specific channels.  A boundary
point is described by its local coordinate, point-type masks, and physical
boundary values at that point:

    x_local, y_local,
    wall_mask, interface_mask,
    boundary_pressure, boundary_temperature, boundary_u, boundary_v,
    local_aspect_ratio

The two vertical subdomain boundaries both use interface_mask=1.  The top and
bottom walls both use wall_mask=1.  Coordinates tell the network where the point
is; no left/right or top/bottom one-hot side label is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from fno_fluent_dataset import build_case_subdomains

PathLike = Union[str, Path]
CaseFiles = Mapping[int, Mapping[str, PathLike]]

BRANCH_CHANNELS = [
    "x_local",
    "y_local",
    "wall_mask",
    "interface_mask",
    "boundary_pressure",
    "boundary_temperature",
    "boundary_u",
    "boundary_v",
    "local_aspect_ratio",
]

TRUNK_CHANNELS = ["x_local", "y_local"]
OUTPUT_CHANNELS = ["pressure", "temperature", "u", "v"]


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _to_nhwc(a_nchw: np.ndarray) -> np.ndarray:
    if a_nchw.ndim != 4:
        raise ValueError(f"Expected a 4-D NCHW array, got shape {a_nchw.shape}")
    return np.moveaxis(a_nchw, 1, -1)


def _field_index(channel_names: Sequence[str]) -> Dict[str, int]:
    aliases = {
        "pressure": ["pressure", "p", "SV_P"],
        "temperature": ["temperature", "t", "T", "SV_T"],
        "u": ["u", "x_velocity", "x-velocity", "SV_U"],
        "v": ["v", "y_velocity", "y-velocity", "SV_V"],
    }
    lower_to_idx = {str(name).lower(): i for i, name in enumerate(channel_names)}
    out: Dict[str, int] = {}
    for clean, names in aliases.items():
        found = None
        for name in names:
            key = str(name).lower()
            if key in lower_to_idx:
                found = lower_to_idx[key]
                break
        if found is None:
            raise ValueError(
                f"Could not find {clean!r} in output channels {list(channel_names)}"
            )
        out[clean] = int(found)
    return out


def make_query_coords(nx: int, ny: int, dtype=np.float32) -> np.ndarray:
    """Return flattened local query coordinates with shape (nx * ny, 2)."""
    x = np.linspace(0.0, 1.0, int(nx), dtype=dtype)
    y = np.linspace(0.0, 1.0, int(ny), dtype=dtype)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(dtype, copy=False)


def _stack_sensor_features(
    x_local: np.ndarray,
    y_local: np.ndarray,
    wall_mask: np.ndarray,
    interface_mask: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    local_aspect_ratio: float,
) -> np.ndarray:
    n = int(np.asarray(x_local).size)
    ar_col = np.full(n, float(local_aspect_ratio), dtype=np.float32)
    return np.column_stack(
        [
            np.asarray(x_local, dtype=np.float32).reshape(n),
            np.asarray(y_local, dtype=np.float32).reshape(n),
            np.asarray(wall_mask, dtype=np.float32).reshape(n),
            np.asarray(interface_mask, dtype=np.float32).reshape(n),
            np.asarray(pressure, dtype=np.float32).reshape(n),
            np.asarray(temperature, dtype=np.float32).reshape(n),
            np.asarray(u, dtype=np.float32).reshape(n),
            np.asarray(v, dtype=np.float32).reshape(n),
            ar_col,
        ]
    ).astype(np.float32, copy=False)


def make_deeponet_branch_inputs_for_case(
    case: Mapping[str, object],
    include_corner_dupes: bool = True,
) -> np.ndarray:
    """
    Build DeepONet branch inputs for one case returned by build_case_subdomains.

    Parameters
    ----------
    case:
        Output for one aspect-ratio case from fno_fluent_dataset.build_case_subdomains.
    include_corner_dupes:
        If True, corners appear once as interface sensors and once as wall
        sensors.  This keeps each edge profile intact and is the recommended
        default.  No side label is added, so left/right are still not specified.

        If False, duplicated corner sensors are removed by keeping the first
        occurrence.  This is useful only if your branch net is very sensitive to
        repeated coordinates.

    Returns
    -------
    branch:
        Shape (n_subdomains, n_boundary_points, 9), with channel names
        BRANCH_CHANNELS.
    """
    Y = np.asarray(case["Y"], dtype=np.float32)  # (S, C, nx, ny)
    interfaces = np.asarray(case["interfaces"], dtype=np.float32)  # (S+1, C, ny)
    channel_names = list(case["output_channel_names"])
    metadata = list(case["metadata"])
    idx = _field_index(channel_names)

    if Y.ndim != 4:
        raise ValueError(f"case['Y'] must have shape (S,C,nx,ny), got {Y.shape}")
    if interfaces.ndim != 3:
        raise ValueError(
            f"case['interfaces'] must have shape (S+1,C,ny), got {interfaces.shape}"
        )

    n_sub, _, nx, ny = Y.shape
    if interfaces.shape[0] != n_sub + 1 or interfaces.shape[-1] != ny:
        raise ValueError(f"Interface shape mismatch: Y={Y.shape}, interfaces={interfaces.shape}")
    if len(metadata) != n_sub:
        raise ValueError(f"Metadata length {len(metadata)} does not match {n_sub}")

    p_i = idx["pressure"]
    t_i = idx["temperature"]
    u_i = idx["u"]
    v_i = idx["v"]

    x_line = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_line = np.linspace(0.0, 1.0, ny, dtype=np.float32)

    branch_parts: List[np.ndarray] = []

    for s in range(n_sub):
        m = metadata[s]
        width = float(m["x_right_mm"]) - float(m["x_left_mm"])
        height = float(m["y_top_mm"]) - float(m["y_bottom_mm"])
        local_aspect = width / max(height, 1.0e-12)

        left = interfaces[s]
        right = interfaces[s + 1]

        # Both vertical sides are encoded as interface points.  There is no
        # left/right mask or side-specific physical channel.
        left_sensors = _stack_sensor_features(
            x_local=np.zeros(ny, dtype=np.float32),
            y_local=y_line,
            wall_mask=np.zeros(ny, dtype=np.float32),
            interface_mask=np.ones(ny, dtype=np.float32),
            pressure=left[p_i],
            temperature=left[t_i],
            u=left[u_i],
            v=left[v_i],
            local_aspect_ratio=local_aspect,
        )
        right_sensors = _stack_sensor_features(
            x_local=np.ones(ny, dtype=np.float32),
            y_local=y_line,
            wall_mask=np.zeros(ny, dtype=np.float32),
            interface_mask=np.ones(ny, dtype=np.float32),
            pressure=right[p_i],
            temperature=right[t_i],
            u=right[u_i],
            v=right[v_i],
            local_aspect_ratio=local_aspect,
        )

        # Top/bottom walls use values sampled from the gridded CFD field.
        # Pressure is included because the requested physical boundary vector is p/t/u/v.
        bottom_sensors = _stack_sensor_features(
            x_local=x_line,
            y_local=np.zeros(nx, dtype=np.float32),
            wall_mask=np.ones(nx, dtype=np.float32),
            interface_mask=np.zeros(nx, dtype=np.float32),
            pressure=Y[s, p_i, :, 0],
            temperature=Y[s, t_i, :, 0],
            u=Y[s, u_i, :, 0],
            v=Y[s, v_i, :, 0],
            local_aspect_ratio=local_aspect,
        )
        top_sensors = _stack_sensor_features(
            x_local=x_line,
            y_local=np.ones(nx, dtype=np.float32),
            wall_mask=np.ones(nx, dtype=np.float32),
            interface_mask=np.zeros(nx, dtype=np.float32),
            pressure=Y[s, p_i, :, -1],
            temperature=Y[s, t_i, :, -1],
            u=Y[s, u_i, :, -1],
            v=Y[s, v_i, :, -1],
            local_aspect_ratio=local_aspect,
        )

        sensors = np.concatenate(
            [left_sensors, right_sensors, bottom_sensors, top_sensors], axis=0
        )

        if not include_corner_dupes:
            # Keep the first occurrence for each rounded coordinate.  This is
            # deterministic and avoids left/right/top/bottom labels.
            coords = np.round(sensors[:, 0:2], decimals=8)
            _, keep = np.unique(coords, axis=0, return_index=True)
            sensors = sensors[np.sort(keep)]

        branch_parts.append(sensors.astype(np.float32, copy=False))

    # All subdomains in a case have the same sensor count.
    return np.stack(branch_parts, axis=0).astype(np.float32, copy=False)


def build_fluent_deeponet_dataset(
    case_files: CaseFiles,
    ar_min: int = 10,
    ar_max: int = 20,
    nx: int = 256,
    ny: int = 256,
    n_subdomains: int = 10,
    include_corner_dupes: bool = True,
    keep_cases: bool = False,
    interface_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    n_realizations: int = 1,
) -> Dict[str, object]:
    """
    Build an in-memory DeepONet dataset from Fluent HDF5 case files.

    The returned branch inputs do not have left/right-specific channels.  Each
    boundary sensor has wall/interface masks and p/t/u/v values at that point.

    Returns
    -------
    dict with keys:
        branch_inputs: (N, M, 9)
        query_coords: (nx*ny, 2)
        outputs: (N, nx, ny, 4)
        branch_channel_names, trunk_channel_names, output_channel_names
        aspect_ratio, subdomain_id, x_left_mm, x_right_mm, y_bottom_mm, y_top_mm
    """
    cases: Dict[Union[int, Tuple[int, int]], Mapping[str, object]] = {}
    metadata: List[Mapping[str, object]] = []
    branch_parts: List[np.ndarray] = []
    output_parts: List[np.ndarray] = []
    output_channel_names: Optional[List[str]] = None
    rng = np.random.default_rng() if rng is None else rng
    if n_realizations < 1:
        raise ValueError("n_realizations must be >= 1")

    for ar in range(int(ar_min), int(ar_max) + 1):
        if ar not in case_files:
            print(f"Skipping AR={ar}: not in case_files")
            continue
        for realization in range(n_realizations):
            print(f"Processing AR={ar} realization={realization}", flush=True)
            case = build_case_subdomains(
                mesh_h5=case_files[ar]["mesh"],
                dat_h5=case_files[ar]["dat"],
                aspect_ratio=ar,
                n_subdomains=n_subdomains,
                nx=nx,
                ny=ny,
                interface_jitter=interface_jitter,
                rng=rng,
            )
            
            for m in case["metadata"]:
                m["realization_id"] = realization

            branch_case = make_deeponet_branch_inputs_for_case(
                case, include_corner_dupes=include_corner_dupes
            )
            outputs_case = _to_nhwc(np.asarray(case["Y"], dtype=np.float32))

            branch_parts.append(branch_case)
            output_parts.append(outputs_case)
            metadata.extend(case["metadata"])
            output_channel_names = list(case["output_channel_names"])

            if keep_cases:
                if n_realizations > 1:
                    cases[(ar, realization)] = case
                else:
                    cases[ar] = case

    if not branch_parts:
        raise ValueError("No cases were processed. Check case_files and AR range.")

    branch_inputs = np.concatenate(branch_parts, axis=0).astype(np.float32, copy=False)
    outputs = np.concatenate(output_parts, axis=0).astype(np.float32, copy=False)
    query_coords = make_query_coords(nx=nx, ny=ny, dtype=np.float32)

    aspect_ratio = np.asarray([m["aspect_ratio"] for m in metadata], dtype=np.float32)
    subdomain_id = np.asarray([m["subdomain_id"] for m in metadata], dtype=np.int32)
    realization_id = np.asarray([m["realization_id"] for m in metadata], dtype=np.int32)
    x_left_mm = np.asarray([m["x_left_mm"] for m in metadata], dtype=np.float32)
    x_right_mm = np.asarray([m["x_right_mm"] for m in metadata], dtype=np.float32)
    y_bottom_mm = np.asarray([m["y_bottom_mm"] for m in metadata], dtype=np.float32)
    y_top_mm = np.asarray([m["y_top_mm"] for m in metadata], dtype=np.float32)
    local_aspect_ratio = np.asarray([m["local_aspect_ratio"] for m in metadata], dtype=np.float32)

    return {
        "branch_inputs": branch_inputs,
        "query_coords": query_coords,
        "outputs": outputs,
        "branch_channel_names": list(BRANCH_CHANNELS),
        "trunk_channel_names": list(TRUNK_CHANNELS),
        "output_channel_names": list(output_channel_names or OUTPUT_CHANNELS),
        "aspect_ratio": aspect_ratio,
        "subdomain_id": subdomain_id,
        "x_left_mm": x_left_mm,
        "x_right_mm": x_right_mm,
        "y_bottom_mm": y_bottom_mm,
        "y_top_mm": y_top_mm,
        "nx": int(nx),
        "ny": int(ny),
        "n_subdomains": int(n_subdomains),
        "include_corner_dupes": bool(include_corner_dupes),
        "cases": cases if keep_cases else None,
        "local_aspect_ratio": local_aspect_ratio,
        "interface_jitter": float(interface_jitter),
        "realization_id": realization_id,
        "n_realizations": int(n_realizations),
    }


def save_deeponet_dataset_h5(dataset: Mapping[str, object], output_path: PathLike) -> None:
    """Save a DeepONet dataset dictionary to HDF5."""
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "branch_inputs",
            data=np.asarray(dataset["branch_inputs"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        f.create_dataset(
            "query_coords",
            data=np.asarray(dataset["query_coords"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        f.create_dataset(
            "outputs",
            data=np.asarray(dataset["outputs"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        for key in [
            "aspect_ratio",
            "subdomain_id",
            "x_left_mm",
            "x_right_mm",
            "y_bottom_mm",
            "y_top_mm",
        ]:
            f.create_dataset(key, data=np.asarray(dataset[key]))
        f.attrs["branch_channel_names"] = "\n".join(dataset["branch_channel_names"])
        f.attrs["trunk_channel_names"] = "\n".join(dataset["trunk_channel_names"])
        f.attrs["output_channel_names"] = "\n".join(dataset["output_channel_names"])
        f.attrs["nx"] = int(dataset["nx"])
        f.attrs["ny"] = int(dataset["ny"])
        f.attrs["n_subdomains"] = int(dataset["n_subdomains"])
        f.attrs["include_corner_dupes"] = bool(dataset["include_corner_dupes"])


def load_deeponet_dataset_h5(path: PathLike) -> Dict[str, object]:
    """Load a dataset saved by save_deeponet_dataset_h5."""
    path = _as_path(path)
    with h5py.File(path, "r") as f:
        data: Dict[str, object] = {
            "branch_inputs": f["branch_inputs"][:],
            "query_coords": f["query_coords"][:],
            "outputs": f["outputs"][:],
        }
        for key in [
            "aspect_ratio",
            "subdomain_id",
            "x_left_mm",
            "x_right_mm",
            "y_bottom_mm",
            "y_top_mm",
        ]:
            data[key] = f[key][:]
        data["branch_channel_names"] = str(f.attrs["branch_channel_names"]).split("\n")
        data["trunk_channel_names"] = str(f.attrs["trunk_channel_names"]).split("\n")
        data["output_channel_names"] = str(f.attrs["output_channel_names"]).split("\n")
        data["nx"] = int(f.attrs["nx"])
        data["ny"] = int(f.attrs["ny"])
        data["n_subdomains"] = int(f.attrs["n_subdomains"])
        data["include_corner_dupes"] = bool(f.attrs["include_corner_dupes"])
        data["cases"] = None
        return data


class FeatureNormalizer:
    """
    Channel-wise normalizer for tensors whose last dimension is feature/channel.

    It works for branch tensors (N, M, C), query tensors (P, C), target tensors
    (N, nx, ny, C), or point targets (B, Q, C).

    skip_indices can be used to leave masks and normalized coordinates unchanged.
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


class DeepONetPointDataset(Dataset):
    """
    PyTorch dataset for pointwise DeepONet training.

    Each item returns:
        branch:  (M, Cb)
        query:   (Q, Ct)
        target:  (Q, Cout)
        sample_index: scalar long

    If n_query_points is None, all nx*ny points are returned.  For training on a
    256x256 grid, use a subset such as 4096 or 8192 query points per sample.
    """

    def __init__(
        self,
        branch_inputs: Union[np.ndarray, torch.Tensor],
        outputs: Union[np.ndarray, torch.Tensor],
        query_coords: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sample_indices: Optional[Sequence[int]] = None,
        n_query_points: Optional[int] = 4096,
        random_query: bool = True,
        branch_normalizer: Optional[FeatureNormalizer] = None,
        y_normalizer: Optional[FeatureNormalizer] = None,
        query_normalizer: Optional[FeatureNormalizer] = None,
    ):
        branch = torch.as_tensor(branch_inputs, dtype=torch.float32)
        y = torch.as_tensor(outputs, dtype=torch.float32)

        if branch.ndim != 3:
            raise ValueError(f"branch_inputs must have shape (N,M,Cb), got {tuple(branch.shape)}")
        if y.ndim == 4:
            n, nx, ny, c = y.shape
            y_flat = y.reshape(n, nx * ny, c)
            if query_coords is None:
                q = torch.from_numpy(make_query_coords(nx, ny)).float()
            else:
                q = torch.as_tensor(query_coords, dtype=torch.float32)
        elif y.ndim == 3:
            y_flat = y
            q = torch.as_tensor(query_coords, dtype=torch.float32)
        else:
            raise ValueError(f"outputs must have shape (N,nx,ny,C) or (N,P,C), got {tuple(y.shape)}")

        if q.ndim != 2:
            raise ValueError(f"query_coords must have shape (P,Ct), got {tuple(q.shape)}")
        if y_flat.shape[0] != branch.shape[0]:
            raise ValueError("branch_inputs and outputs have different sample counts")
        if y_flat.shape[1] != q.shape[0]:
            raise ValueError("outputs point count and query_coords point count do not match")

        if sample_indices is None:
            idx = torch.arange(branch.shape[0], dtype=torch.long)
        else:
            idx = torch.as_tensor(sample_indices, dtype=torch.long)

        self.branch = branch[idx]
        self.y_flat = y_flat[idx]
        self.query_coords = q
        self.original_indices = idx
        self.n_query_points = n_query_points
        self.random_query = bool(random_query)
        self.branch_normalizer = branch_normalizer
        self.y_normalizer = y_normalizer
        self.query_normalizer = query_normalizer

    def __len__(self) -> int:
        return int(self.branch.shape[0])

    @property
    def num_query_points_total(self) -> int:
        return int(self.query_coords.shape[0])

    def _select_query_indices(self) -> torch.Tensor:
        p = self.num_query_points_total
        if self.n_query_points is None or int(self.n_query_points) >= p:
            return torch.arange(p, dtype=torch.long)
        q = int(self.n_query_points)
        if self.random_query:
            return torch.randperm(p)[:q]
        return torch.linspace(0, p - 1, q, dtype=torch.long)

    def __getitem__(self, i: int):
        point_idx = self._select_query_indices()
        branch = self.branch[i]
        query = self.query_coords[point_idx]
        target = self.y_flat[i, point_idx]

        if self.branch_normalizer is not None:
            branch = self.branch_normalizer.encode(branch)
        if self.query_normalizer is not None:
            query = self.query_normalizer.encode(query)
        if self.y_normalizer is not None:
            target = self.y_normalizer.encode(target)

        return branch, query, target, self.original_indices[i]


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
    act: nn.Module
    if activation.lower() == "relu":
        act = nn.ReLU()
    elif activation.lower() == "tanh":
        act = nn.Tanh()
    elif activation.lower() == "silu":
        act = nn.SiLU()
    else:
        act = nn.GELU()

    layers: List[nn.Module] = []
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
    """
    Permutation-invariant branch net for boundary/interface point sets.

    Input shape:  (batch, n_boundary_points, branch_input_dim)
    Output shape: (batch, output_channels, latent_dim)
    """

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
    """
    Trunk net for query coordinates.

    Input shape:  (batch, n_query_points, trunk_input_dim)
    Output shape: (batch, n_query_points, latent_dim)
    """

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
    query:  (B, Q, Ct)
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
    model,
    branch,
    branch_normalizer=None,
    y_normalizer=None,
    branch_channel_names=None,
):
    """
    branch is the tensor already returned by DeepONetPointDataset.
    If branch_normalizer is used, branch is branch-normalized here.
    """

    device = branch.device

    if branch_channel_names is None:
        branch_channel_names = BRANCH_CHANNELS

    names = list(branch_channel_names)

    x_ch = names.index("x_local")
    y_ch = names.index("y_local")
    value_ch = [
        names.index("boundary_pressure"),
        names.index("boundary_temperature"),
        names.index("boundary_u"),
        names.index("boundary_v"),
    ]

    if branch_normalizer is not None:
        branch_normalizer = branch_normalizer.to(device)
        branch_phys = branch_normalizer.decode(branch)
    else:
        branch_phys = branch

    # Boundary query coordinates: same coordinates as branch sensors.
    q_bc = branch_phys[..., [x_ch, y_ch]]

    # Boundary target values in physical units.
    bc_target_phys = branch_phys[..., value_ch]

    # Model output is in y-normalized space if y_normalizer was used in training.
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
        bc_target = y_normalizer.encode(bc_target_phys)
    else:
        bc_target = bc_target_phys

    pred_bc = model(branch, q_bc)

    return F.mse_loss(pred_bc, bc_target)


def train_deeponet_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_type="mse",
    lambda_bc=1.0,
    branch_normalizer=None,
    y_normalizer=None,
    branch_channel_names=None,
):
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

        bc_loss = boundary_loss(
            model=model,
            branch=branch,
            branch_normalizer=branch_normalizer,
            y_normalizer=y_normalizer,
            branch_channel_names=branch_channel_names,
        )

        loss = field_loss + float(lambda_bc) * bc_loss

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
    """
    Evaluate DeepONet.  If y_normalizer is provided, metrics are reported in
    physical units after decoding.
    """
    model.eval()
    device = torch.device(device)
    mse_sum = 0.0
    rel_sum = 0.0
    count = 0
    channel_sse: Optional[torch.Tensor] = None
    channel_energy: Optional[torch.Tensor] = None
    rel_l2 = RelativeL2Loss()

    for branch, query, target, _sample_idx in loader:
        branch = branch.to(device)
        query = query.to(device)
        target = target.to(device)
        pred = model(branch, query)

        pred_metric = pred
        target_metric = target
        if y_normalizer is not None:
            y_normalizer = y_normalizer.to(device)
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
def predict_deeponet_full_grid(
    model: nn.Module,
    branch: Union[np.ndarray, torch.Tensor],
    query_coords: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device],
    branch_normalizer: Optional[FeatureNormalizer] = None,
    y_normalizer: Optional[FeatureNormalizer] = None,
    query_normalizer: Optional[FeatureNormalizer] = None,
    query_batch_size: int = 65536,
) -> torch.Tensor:
    """
    Predict one sample on the full query grid without storing all query points on
    GPU at once.

    Returns a CPU tensor shaped (P, output_channels), decoded to physical units
    if y_normalizer is provided.
    """
    model.eval()
    device = torch.device(device)
    b = torch.as_tensor(branch, dtype=torch.float32)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    if branch_normalizer is not None:
        branch_normalizer = branch_normalizer.to(device)
        b = branch_normalizer.encode(b)
    b = b.to(device)

    q_all = torch.as_tensor(query_coords, dtype=torch.float32)
    preds = []
    for start in range(0, q_all.shape[0], int(query_batch_size)):
        q = q_all[start:start + int(query_batch_size)]
        if query_normalizer is not None:
            query_normalizer = query_normalizer.to(device)
            q = query_normalizer.encode(q)
        q = q.unsqueeze(0).to(device)
        pred = model(b, q).squeeze(0)
        if y_normalizer is not None:
            y_normalizer = y_normalizer.to(device)
            pred = y_normalizer.decode(pred)
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0)
