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

from fno_fluent_dataset import (
    sample_constrained_x_edges,
    read_fluent_cell_centers,
    read_fluent_cell_fields,
    FIELD_MAP,
)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
        y_wall_flag = ((y_line == 0.0) | (y_line == 1.0)).astype(np.float32)
        x_interface_flag = ((x_line == 0.0) | (x_line == 1.0)).astype(np.float32)

        # Both vertical sides are encoded as interface points.  There is no
        # left/right mask or side-specific physical channel.
        left_sensors = _stack_sensor_features(
            x_local=np.zeros(ny, dtype=np.float32),
            y_local=y_line,
            wall_mask=y_wall_flag,
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
            wall_mask=y_wall_flag,
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
            interface_mask=x_interface_flag,
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
            interface_mask=x_interface_flag,
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




# -----------------------------------------------------------------------------
# Non-grid / cell-center DeepONet dataset construction
# -----------------------------------------------------------------------------

CELL_BRANCH_CHANNELS = [
    "x_local",
    "y_local",
    "wall_mask",
    "interface_mask",
    "boundary_pressure",
    "boundary_temperature",
    "boundary_u",
    "boundary_v",
    "known_pressure",
    "known_temperature",
    "known_u",
    "known_v",
    "local_aspect_ratio",
]

CELL_TRUNK_CHANNELS = ["x_local", "y_local"]
CELL_OUTPUT_CHANNELS = ["pressure", "temperature", "u", "v"]


def _make_global_interpolators(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
):
    """Build global interpolators for p/T/u/v from Fluent cell centers."""
    points = np.column_stack([x, y])
    linear = []
    nearest = []
    for c in range(values.shape[1]):
        linear.append(LinearNDInterpolator(points, values[:, c]))
        nearest.append(NearestNDInterpolator(points, values[:, c]))
    return linear, nearest


def _interp_fields(
    linear,
    nearest,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Interpolate p/T/u/v at query points."""
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    yq = np.asarray(yq, dtype=np.float64).reshape(-1)
    if xq.shape != yq.shape:
        raise ValueError(f"xq and yq must have same shape, got {xq.shape} and {yq.shape}")
    query = np.column_stack([xq, yq])
    out = np.empty((xq.size, len(linear)), dtype=np.float32)
    for c, (lin, near) in enumerate(zip(linear, nearest)):
        z = lin(query)
        bad = np.isnan(z)
        if np.any(bad):
            z[bad] = near(query[bad])
        out[:, c] = z.astype(np.float32)
    return out


def _stack_cell_branch_features(
    x_local: np.ndarray,
    y_local: np.ndarray,
    wall_mask: np.ndarray,
    interface_mask: np.ndarray,
    values: np.ndarray,
    known_mask: np.ndarray,
    local_aspect_ratio: float,
) -> np.ndarray:
    """
    Stack non-grid branch point features.

    values shape:
        (M, 4), ordered [pressure, temperature, u, v].
    known_mask shape:
        (M, 4), ordered [known_pressure, known_temperature, known_u, known_v].
    """
    x_local = np.asarray(x_local, dtype=np.float32).reshape(-1)
    y_local = np.asarray(y_local, dtype=np.float32).reshape(-1)
    wall_mask = np.asarray(wall_mask, dtype=np.float32).reshape(-1)
    interface_mask = np.asarray(interface_mask, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    known_mask = np.asarray(known_mask, dtype=np.float32)

    n = x_local.size
    if y_local.size != n or wall_mask.size != n or interface_mask.size != n:
        raise ValueError("x_local, y_local, wall_mask, and interface_mask must have same length")
    if values.shape != (n, 4):
        raise ValueError(f"values must have shape {(n, 4)}, got {values.shape}")
    if known_mask.shape != (n, 4):
        raise ValueError(f"known_mask must have shape {(n, 4)}, got {known_mask.shape}")

    return np.column_stack(
        [
            x_local,
            y_local,
            wall_mask,
            interface_mask,
            values,
            known_mask,
            np.full(n, float(local_aspect_ratio), dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def make_rect_channel_bc_values(
    side: str,
    n: int,
    inlet_u: float = 0.1,
    inlet_v: float = 0.0,
    inlet_T: float = 273.0,
    outlet_p: float = 0.0,
    top_T: float = 273.0,
    bottom_T: float = 275.0,
    wall_u: float = 0.0,
    wall_v: float = 0.0,
    fill_unknown_with: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return prescribed boundary values and known masks for a rectangular channel.

    values order:
        [pressure, temperature, u, v]
    known order:
        [known_pressure, known_temperature, known_u, known_v]

    Unknown fields are set to fill_unknown_with and marked with known=0.
    """
    side = str(side).lower()
    n = int(n)
    values = np.full((n, 4), float(fill_unknown_with), dtype=np.float32)
    known = np.zeros((n, 4), dtype=np.float32)

    if side == "inlet":
        values[:, 1] = float(inlet_T)
        values[:, 2] = float(inlet_u)
        values[:, 3] = float(inlet_v)
        known[:, 1] = 1.0
        known[:, 2] = 1.0
        known[:, 3] = 1.0
    elif side == "outlet":
        values[:, 0] = float(outlet_p)
        known[:, 0] = 1.0
    elif side == "top":
        values[:, 1] = float(top_T)
        values[:, 2] = float(wall_u)
        values[:, 3] = float(wall_v)
        known[:, 1] = 1.0
        known[:, 2] = 1.0
        known[:, 3] = 1.0
    elif side == "bottom":
        values[:, 1] = float(bottom_T)
        values[:, 2] = float(wall_u)
        values[:, 3] = float(wall_v)
        known[:, 1] = 1.0
        known[:, 2] = 1.0
        known[:, 3] = 1.0
    else:
        raise ValueError("side must be inlet, outlet, top, or bottom")

    return values, known


def build_fluent_deeponet_dataset(
    case_files: CaseFiles,
    ars: Sequence[int],
    n_subdomains: Union[int, Sequence[int]] = 10,
    n_interface_points: int = 256,
    n_wall_points: int = 256,
    interface_placement: str = "fixed",
    interface_jitter: float = 0.0,
    min_subdomain_width: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    field_map=FIELD_MAP,
    bc_kwargs: Optional[Mapping[str, float]] = None,
    keep_raw_case_data: bool = False,
    n_realizations: int = 1,
) -> Dict[str, object]:
    """
    Build a non-grid DeepONet dataset from Fluent HDF5 files.

    Unlike build_fluent_deeponet_dataset(), this builder does not rasterize the
    CFD data to a Cartesian grid.  It keeps original Fluent cell centers as trunk
    query points and original cell-centered p/T/u/v values as targets.

    Branch points:
        - exterior inlet/outlet/walls use prescribed boundary-condition values;
        - interior interfaces use interpolated values from the global Fluent field;
        - known_* masks mark which p/T/u/v values are prescribed/available.

    Returned samples are variable-length dictionaries:
        samples[i]["branch"]: (M, 13)
        samples[i]["query"] : (Q_i, 2)
        samples[i]["target"]: (Q_i, 4)
    """
    if rng is None:
        rng = np.random.default_rng()
    if bc_kwargs is None:
        bc_kwargs = {}

    n_interface_points = int(n_interface_points)
    n_wall_points = int(n_wall_points)
    if n_interface_points <= 1 or n_wall_points <= 1:
        raise ValueError("n_interface_points and n_wall_points must more than 1")
    if isinstance(n_subdomains, Sequence) and len(n_subdomains) != len(ars):
        raise ValueError("n_subdomains must be a sequence of the same length as ars")
    if isinstance(n_subdomains, int):
        n_subdomains = [n_subdomains] * len(ars)

    samples: List[Dict[str, np.ndarray]] = []
    metadata: List[Dict[str, object]] = []
    raw_cases: Dict[int, object] = {}

    for ar, n_sub in zip(ars, n_subdomains):
        if ar not in case_files:
            print(f"Skipping AR={ar}: not in case_files")
            continue
        
        for realization in range(n_realizations):
            print(f"Processing AR={ar} realization={realization}", flush=True)
            centers, mesh_info = read_fluent_cell_centers(case_files[ar]["mesh"])
            fields = read_fluent_cell_fields(case_files[ar]["dat"], field_map=field_map)

            channel_names = list(fields.keys())
            values_all = np.column_stack([fields[name] for name in channel_names]).astype(np.float64)
            if len(values_all) != len(centers):
                raise ValueError(
                    f"Mesh has {len(centers)} cells but .dat.h5 has {len(values_all)} values for AR={ar}."
                )

            valid = np.all(np.isfinite(centers), axis=1) & np.all(np.isfinite(values_all), axis=1)
            x = centers[valid, 0].astype(np.float64)
            y = centers[valid, 1].astype(np.float64)
            values = values_all[valid].astype(np.float64)

            xmin, xmax = float(mesh_info["x_min_mm"]), float(mesh_info["x_max_mm"])
            ymin, ymax = float(mesh_info["y_min_mm"]), float(mesh_info["y_max_mm"])
            height = ymax - ymin

            if interface_placement == "fixed":
                x_edges = np.linspace(xmin, xmax, n_sub + 1, dtype=np.float64)
                if float(interface_jitter) > 0.0:
                    base_dx = (xmax - xmin) / float(n_sub)
                    jitter_dx = float(interface_jitter) * base_dx
                    noise = rng.uniform(-jitter_dx, jitter_dx, size=n_sub - 1)
                    x_edges[1:-1] += noise
                    if np.any(np.diff(x_edges) <= 0):
                        raise RuntimeError("Jittered interfaces crossed. Reduce interface_jitter.")
            elif interface_placement == "random":
                x_edges = sample_constrained_x_edges(
                    xmin=xmin,
                    xmax=xmax,
                    n_subdomains=n_sub,
                    min_subdomain_width=min_subdomain_width,
                    rng=rng,
                )
            else:
                raise ValueError(f"Invalid interface_placement argument: {interface_placement}")

            linear, nearest = _make_global_interpolators(x, y, values)

            y_interface_phys = np.linspace(ymin, ymax, n_interface_points + 2, dtype=np.float64)[1:-1]
            y_interface_local = ((y_interface_phys - ymin) / max(height, 1.0e-12)).astype(np.float32)
            x_wall_local = np.linspace(0.0, 1.0, n_wall_points, dtype=np.float32)

            for s in range(n_sub):
                x0 = float(x_edges[s])
                x1 = float(x_edges[s + 1])
                width = x1 - x0
                local_aspect = width / max(height, 1.0e-12)

                # Trunk/target from original cell centers within this subdomain.
                cell_mask = (x >= x0) & (x <= x1)
                if not np.any(cell_mask):
                    raise ValueError(f"No cell centers in AR={ar}, subdomain={s}.")

                x_cell = x[cell_mask]
                y_cell = y[cell_mask]
                target = values[cell_mask].astype(np.float32)
                query = np.column_stack(
                    [
                        ((x_cell - x0) / max(width, 1.0e-12)).astype(np.float32),
                        ((y_cell - ymin) / max(height, 1.0e-12)).astype(np.float32),
                    ]
                ).astype(np.float32, copy=False)

                branch_parts = []

                # Left vertical side: inlet if s==0, otherwise interpolated interface.
                x_left_phys = np.full(n_interface_points, x0, dtype=np.float64)
                if s == 0:
                    vals, known = make_rect_channel_bc_values("inlet", n_interface_points, **bc_kwargs)
                else:
                    vals = _interp_fields(linear, nearest, x_left_phys, y_interface_phys)
                    known = np.ones((n_interface_points, 4), dtype=np.float32)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=np.zeros(n_interface_points, dtype=np.float32),
                        y_local=y_interface_local,
                        wall_mask=np.zeros(n_interface_points, dtype=np.float32),
                        interface_mask=np.ones(n_interface_points, dtype=np.float32),
                        values=vals,
                        known_mask=known,
                        local_aspect_ratio=local_aspect,
                    )
                )

                # Right vertical side: outlet if last subdomain, otherwise interpolated interface.
                x_right_phys = np.full(n_interface_points, x1, dtype=np.float64)
                if s == n_sub - 1:
                    vals, known = make_rect_channel_bc_values("outlet", n_interface_points, **bc_kwargs)
                else:
                    vals = _interp_fields(linear, nearest, x_right_phys, y_interface_phys)
                    known = np.ones((n_interface_points, 4), dtype=np.float32)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=np.ones(n_interface_points, dtype=np.float32),
                        y_local=y_interface_local,
                        wall_mask=np.zeros(n_interface_points, dtype=np.float32),
                        interface_mask=np.ones(n_interface_points, dtype=np.float32),
                        values=vals,
                        known_mask=known,
                        local_aspect_ratio=local_aspect,
                    )
                )

                # Bottom wall: prescribed BC values.
                vals, known = make_rect_channel_bc_values("bottom", n_wall_points, **bc_kwargs)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=x_wall_local,
                        y_local=np.zeros(n_wall_points, dtype=np.float32),
                        wall_mask=np.ones(n_wall_points, dtype=np.float32),
                        interface_mask=np.zeros(n_wall_points, dtype=np.float32),
                        values=vals,
                        known_mask=known,
                        local_aspect_ratio=local_aspect,
                    )
                )

                # Top wall: prescribed BC values.
                vals, known = make_rect_channel_bc_values("top", n_wall_points, **bc_kwargs)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=x_wall_local,
                        y_local=np.ones(n_wall_points, dtype=np.float32),
                        wall_mask=np.ones(n_wall_points, dtype=np.float32),
                        interface_mask=np.zeros(n_wall_points, dtype=np.float32),
                        values=vals,
                        known_mask=known,
                        local_aspect_ratio=local_aspect,
                    )
                )

                branch = np.concatenate(branch_parts, axis=0).astype(np.float32, copy=False)
                samples.append({"branch": branch, "query": query, "target": target})
                metadata.append(
                    {
                        "aspect_ratio": float(ar),
                        "subdomain_id": int(s),
                        "x_left_mm": float(x0),
                        "x_right_mm": float(x1),
                        "y_bottom_mm": float(ymin),
                        "y_top_mm": float(ymax),
                        "local_aspect_ratio": float(local_aspect),
                        "n_cells": int(query.shape[0]),
                    }
                )

            if keep_raw_case_data:
                raw_cases[int(ar)] = {
                    "x": x.astype(np.float32),
                    "y": y.astype(np.float32),
                    "values": values.astype(np.float32),
                    "mesh_info": mesh_info,
                    "x_edges_mm": x_edges.astype(np.float32),
                    "output_channel_names": channel_names,
                }

    if not samples:
        raise ValueError("No samples were processed. Check case_files and AR range.")

    return {
        "samples": samples,
        "metadata": metadata,
        "branch_channel_names": list(CELL_BRANCH_CHANNELS),
        "trunk_channel_names": list(CELL_TRUNK_CHANNELS),
        "output_channel_names": list(CELL_OUTPUT_CHANNELS),
        "n_subdomains": n_subdomains,
        "n_interface_points": int(n_interface_points),
        "n_wall_points": int(n_wall_points),
        "interface_jitter": float(interface_jitter),
        "raw_cases": raw_cases if keep_raw_case_data else None,
    }


class DeepONetCellDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for non-grid cell-center DeepONet training.

    Each item returns:
        branch: (M, Cb)
        query:  (Q, 2)
        target: (Q, 4)
        sample_index: scalar long

    Branch physical values [p,T,u,v] are normalized with target_y_normalizer,
    not with a separate branch normalizer.  Coordinates and masks are not changed.
    """

    def __init__(
        self,
        samples: Sequence[Mapping[str, np.ndarray]],
        sample_indices: Optional[Sequence[int]] = None,
        n_query_points: Optional[int] = 8192,
        random_query: bool = True,
        target_y_normalizer=None,
        local_aspect_mean: Optional[float] = None,
        local_aspect_std: Optional[float] = None,
        branch_channel_names: Sequence[str] = CELL_BRANCH_CHANNELS,
    ):
        self.samples = list(samples)
        if sample_indices is None:
            self.indices = np.arange(len(self.samples), dtype=np.int64)
        else:
            self.indices = np.asarray(sample_indices, dtype=np.int64)
        self.n_query_points = n_query_points
        self.random_query = bool(random_query)
        self.target_y_normalizer = target_y_normalizer
        self.local_aspect_mean = local_aspect_mean
        self.local_aspect_std = local_aspect_std
        self.branch_channel_names = list(branch_channel_names)
        self.value_idx = [
            self.branch_channel_names.index("boundary_pressure"),
            self.branch_channel_names.index("boundary_temperature"),
            self.branch_channel_names.index("boundary_u"),
            self.branch_channel_names.index("boundary_v"),
        ]
        self.aspect_idx = self.branch_channel_names.index("local_aspect_ratio")

    def __len__(self) -> int:
        return int(self.indices.size)

    def _sample_query_indices(self, n: int) -> np.ndarray:
        n = int(n)
        if self.n_query_points is None or int(self.n_query_points) >= n:
            return np.arange(n, dtype=np.int64)
        q = int(self.n_query_points)
        if self.random_query:
            return np.random.choice(n, size=q, replace=False)
        return np.linspace(0, n - 1, q).astype(np.int64)

    def _normalize_branch_physical_values_with_y(self, branch: np.ndarray) -> np.ndarray:
        if self.target_y_normalizer is None:
            return branch
        out = branch.copy()
        mean = self.target_y_normalizer.mean.detach().cpu().numpy().reshape(-1)
        std = self.target_y_normalizer.std.detach().cpu().numpy().reshape(-1)
        out[:, self.value_idx] = (out[:, self.value_idx] - mean.reshape(1, 4)) / np.maximum(
            std.reshape(1, 4), 1.0e-12
        )
        return out

    def _normalize_local_aspect(self, branch: np.ndarray) -> np.ndarray:
        if self.local_aspect_mean is None or self.local_aspect_std is None:
            return branch
        out = branch.copy()
        out[:, self.aspect_idx] = (out[:, self.aspect_idx] - float(self.local_aspect_mean)) / max(
            float(self.local_aspect_std), 1.0e-12
        )
        return out

    def __getitem__(self, i: int):
        sample_index = int(self.indices[int(i)])
        sample = self.samples[sample_index]
        branch = np.asarray(sample["branch"], dtype=np.float32)
        query_all = np.asarray(sample["query"], dtype=np.float32)
        target_all = np.asarray(sample["target"], dtype=np.float32)

        point_idx = self._sample_query_indices(query_all.shape[0])
        query = query_all[point_idx]
        target = target_all[point_idx]

        branch = self._normalize_branch_physical_values_with_y(branch)
        branch = self._normalize_local_aspect(branch)

        if self.target_y_normalizer is not None:
            target = self.target_y_normalizer.encode(target).detach().cpu().numpy().astype(np.float32)

        return (
            torch.from_numpy(branch).float(),
            torch.from_numpy(query).float(),
            torch.from_numpy(target).float(),
            torch.tensor(sample_index, dtype=torch.long),
        )
