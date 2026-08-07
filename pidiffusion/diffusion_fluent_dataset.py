"""Build randomized Fluent subdomain samples for PIDiffusion training.

Current protocol:
- one complete 2-D channel per call;
- geometry from the channel design JSON;
- random x-strip decomposition;
- original Fluent cell centers as query points;
- pressure/u/v as outputs;
- prescribed exterior BCs and CFD-interpolated interior interfaces.

The expensive global scattered-field interpolators are built once per channel
and reused across all randomized realizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

PathLike = str | Path

DEFAULT_FIELD_MAP = {"pressure": "SV_P", "u": "SV_U", "v": "SV_V"}
OUTPUT_FIELDS = ("pressure", "u", "v")
TRUNK_CHANNEL_NAMES = ("x_local", "y_local")
BRANCH_CHANNEL_NAMES = (
    "x_local",
    "y_local",
    "wall_mask",
    "interface_mask",
    "boundary_pressure",
    "boundary_u",
    "boundary_v",
    "known_pressure",
    "known_u",
    "known_v",
    "local_aspect_ratio",
)


def _numeric_keys(group: h5py.Group) -> list[str]:
    return sorted(group.keys(), key=lambda key: int(key))


def _first_dataset(group: h5py.Group) -> h5py.Dataset:
    for key in _numeric_keys(group):
        item = group[key]
        if isinstance(item, h5py.Dataset):
            return item
    raise KeyError(f"No dataset found below {group.name}")


def _read_two_node_faces(group: h5py.Group) -> np.ndarray:
    nnodes = group["nnodes"][()].astype(np.int64)
    nodes = group["nodes"][()].astype(np.int64) - 1
    if not np.all(nnodes == 2):
        raise NotImplementedError("Expected a 2-D mesh with two-node faces")
    return nodes.reshape(-1, 2)


def read_fluent_cell_centers(mesh_h5: PathLike) -> tuple[np.ndarray, dict[str, Any]]:
    """Read cell centers from the current Fluent ``.msh.h5`` layout."""

    mesh_path = Path(mesh_h5).expanduser().resolve()
    with h5py.File(mesh_path, "r") as handle:
        mesh = handle["meshes/1"]
        coords = _first_dataset(mesh["nodes/coords"])[()].astype(np.float64)

        cell_min_id = int(np.min(mesh["cells/zoneTopology/minId"][()]))
        cell_max_id = int(np.max(mesh["cells/zoneTopology/maxId"][()]))
        n_cells = cell_max_id - cell_min_id + 1

        face_nodes_group = mesh["faces/nodes"]
        c0_group = mesh["faces/c0"]
        c1_group = mesh["faces/c1"]
        zone_keys = _numeric_keys(face_nodes_group)

        if len(zone_keys) <= 1:
            raise ValueError(
                "Expected the current .msh.h5 layout with separate face-zone groups"
            )

        face_nodes_parts = []
        c0_parts = []
        c1_parts = []
        for zone_key in zone_keys:
            face_nodes = _read_two_node_faces(face_nodes_group[zone_key])
            n_faces = face_nodes.shape[0]
            c0 = (
                c0_group[zone_key][()].astype(np.int64)
                if zone_key in c0_group
                else np.zeros(n_faces, dtype=np.int64)
            )
            c1 = (
                c1_group[zone_key][()].astype(np.int64)
                if zone_key in c1_group
                else np.zeros(n_faces, dtype=np.int64)
            )
            face_nodes_parts.append(face_nodes)
            c0_parts.append(c0)
            c1_parts.append(c1)

    face_nodes = np.vstack(face_nodes_parts)
    c0 = np.concatenate(c0_parts)
    c1 = np.concatenate(c1_parts)

    valid0 = c0 > 0
    valid1 = c1 > 0
    cell_idx = np.concatenate(
        [np.repeat(c0[valid0] - cell_min_id, 2), np.repeat(c1[valid1] - cell_min_id, 2)]
    )
    node_idx = np.concatenate([face_nodes[valid0].reshape(-1), face_nodes[valid1].reshape(-1)])

    order = np.lexsort((node_idx, cell_idx))
    cell_idx = cell_idx[order]
    node_idx = node_idx[order]

    keep = np.empty(len(cell_idx), dtype=bool)
    keep[0] = True
    keep[1:] = (cell_idx[1:] != cell_idx[:-1]) | (node_idx[1:] != node_idx[:-1])
    cell_idx = cell_idx[keep]
    node_idx = node_idx[keep]

    node_count = np.bincount(cell_idx, minlength=n_cells)
    centers = np.full((n_cells, 2), np.nan, dtype=np.float64)
    valid_cells = node_count > 0
    for dim in range(2):
        sums = np.bincount(cell_idx, weights=coords[node_idx, dim], minlength=n_cells)
        centers[valid_cells, dim] = sums[valid_cells] / node_count[valid_cells]

    x_min, x_max = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
    y_min, y_max = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))
    mesh_info = {
        "mesh_h5": str(mesh_path),
        "n_nodes": int(coords.shape[0]),
        "n_cells": int(n_cells),
        "nodes_per_cell_min": int(node_count.min()),
        "nodes_per_cell_max": int(node_count.max()),
        "x_min_mm": x_min,
        "x_max_mm": x_max,
        "y_min_mm": y_min,
        "y_max_mm": y_max,
        "Lx_mm": x_max - x_min,
        "Ly_mm": y_max - y_min,
        "inferred_AR": (x_max - x_min) / (y_max - y_min),
    }
    return centers, mesh_info


def read_fluent_cell_fields(
    dat_h5: PathLike,
    field_map: Mapping[str, str] = DEFAULT_FIELD_MAP,
) -> dict[str, np.ndarray]:
    """Read selected cell-centered fields from Fluent ``.dat.h5``."""

    fields: dict[str, np.ndarray] = {}
    with h5py.File(Path(dat_h5).expanduser().resolve(), "r") as handle:
        base = handle["results/1/phase-1/cells"]
        for field_name, fluent_name in field_map.items():
            if fluent_name not in base:
                raise KeyError(f"Missing {fluent_name} in {base.name}")
            group = base[fluent_name]
            pieces = [
                group[key][()].reshape(-1)
                for key in _numeric_keys(group)
                if isinstance(group[key], h5py.Dataset)
            ]
            if not pieces:
                raise KeyError(f"No datasets found for {fluent_name}")
            fields[field_name] = np.concatenate(pieces).astype(np.float64)
    return fields


def load_channel_config(config_path: PathLike) -> dict[str, Any]:
    """Load geometry and inlet metadata from one channel design JSON."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    metadata = config.get("metadata", {})
    for key in ("x_points_mm", "deltas_mm", "L_mm"):
        if key not in metadata:
            raise KeyError(f"Channel config {path} is missing metadata.{key}")

    x_points = np.asarray(metadata["x_points_mm"], dtype=np.float64).reshape(-1)
    deltas = np.asarray(metadata["deltas_mm"], dtype=np.float64).reshape(-1)
    if x_points.shape != deltas.shape:
        raise ValueError("x_points_mm and deltas_mm must have the same shape")
    if x_points.size < 2:
        raise ValueError("x_points_mm must contain at least two points")

    order = np.argsort(x_points)
    x_points = x_points[order]
    deltas = deltas[order]
    reference_length_mm = float(metadata["L_mm"])

    return {
        "config_path": str(path),
        "x_points": x_points,
        "L_mm": reference_length_mm,
        "y_bottom_points": -deltas,
        "y_top_points": reference_length_mm + deltas,
        "AR": float(metadata["AR"]) if "AR" in metadata else None,
        "Uin_mps": float(metadata["Uin_mps"]) if "Uin_mps" in metadata else None,
    }


def sample_random_x_edges(
    x_min: float,
    x_max: float,
    n_subdomains: int,
    min_subdomain_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match the reference constrained random x-strip partition."""

    span = float(x_max) - float(x_min)
    n_subdomains = int(n_subdomains)
    min_fraction = float(min_subdomain_width)
    if n_subdomains < 1:
        raise ValueError("n_subdomains must be >= 1")
    if min_fraction < 0.0:
        raise ValueError("min_subdomain_width must be non-negative")
    if n_subdomains * min_fraction > 1.0:
        raise ValueError("n_subdomains * min_subdomain_width must be <= 1")
    if n_subdomains == 1:
        return np.asarray([x_min, x_max], dtype=np.float64)

    min_width = min_fraction * span
    slack = span - n_subdomains * min_width
    extras = rng.dirichlet(np.ones(n_subdomains, dtype=np.float64))
    widths = min_width + slack * extras
    edges = np.concatenate([[float(x_min)], float(x_min) + np.cumsum(widths)])
    edges[-1] = float(x_max)
    return edges.astype(np.float64)


def _make_interpolators(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[LinearNDInterpolator, NearestNDInterpolator]:
    """Build one multichannel interpolator pair for a complete channel."""

    points = np.column_stack([x, y])
    linear = LinearNDInterpolator(points, values)
    nearest = NearestNDInterpolator(points, values)
    return linear, nearest


def _interp_fields(
    linear: LinearNDInterpolator,
    nearest: NearestNDInterpolator,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """Interpolate all fields with nearest-neighbor fallback outside the hull."""

    x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
    y_query = np.asarray(y_query, dtype=np.float64).reshape(-1)
    if x_query.shape != y_query.shape:
        raise ValueError("x_query and y_query must have the same shape")

    query = np.column_stack([x_query, y_query])
    output = np.asarray(linear(query), dtype=np.float64)
    missing_rows = np.any(np.isnan(output), axis=1)
    if np.any(missing_rows):
        output[missing_rows] = nearest(query[missing_rows])
    return output.astype(np.float32)

def _stack_branch(
    x_local: np.ndarray,
    y_local: np.ndarray,
    wall_mask: np.ndarray,
    interface_mask: np.ndarray,
    values: np.ndarray,
    known: np.ndarray,
    local_aspect_ratio: float,
) -> np.ndarray:
    """Stack one branch sensor block in the current pressure/u/v schema."""

    x_local = np.asarray(x_local, dtype=np.float32).reshape(-1)
    y_local = np.asarray(y_local, dtype=np.float32).reshape(-1)
    wall_mask = np.asarray(wall_mask, dtype=np.float32).reshape(-1)
    interface_mask = np.asarray(interface_mask, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    known = np.asarray(known, dtype=np.float32)
    n_points = x_local.size

    if not (y_local.size == wall_mask.size == interface_mask.size == n_points):
        raise ValueError("Branch coordinates and masks must have equal lengths")
    if values.shape != (n_points, len(OUTPUT_FIELDS)):
        raise ValueError(f"Unexpected branch value shape: {values.shape}")
    if known.shape != values.shape:
        raise ValueError(f"Unexpected known-mask shape: {known.shape}")

    return np.column_stack(
        [
            x_local,
            y_local,
            wall_mask,
            interface_mask,
            values,
            known,
            np.full(n_points, local_aspect_ratio, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def _boundary_values(
    side: str,
    n_points: int,
    inlet_u: float,
    inlet_v: float,
    outlet_pressure: float,
    wall_u: float,
    wall_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exterior pressure/u/v BC values and known masks."""

    field_index = {name: index for index, name in enumerate(OUTPUT_FIELDS)}
    values = np.zeros((n_points, len(OUTPUT_FIELDS)), dtype=np.float32)
    known = np.zeros_like(values)

    def set_known(name: str, value: float) -> None:
        index = field_index[name]
        values[:, index] = float(value)
        known[:, index] = 1.0

    side = side.lower()
    if side == "inlet":
        set_known("u", inlet_u)
        set_known("v", inlet_v)
    elif side == "outlet":
        set_known("pressure", outlet_pressure)
    elif side in {"top", "bottom"}:
        set_known("u", wall_u)
        set_known("v", wall_v)
    else:
        raise ValueError("side must be inlet, outlet, top, or bottom")
    return values, known


def build_randomized_channel_dataset(
    *,
    case_id: str,
    design_path: PathLike,
    mesh_path: PathLike,
    dat_path: PathLike,
    n_subdomains: int = 10,
    n_realizations: int = 10,
    n_interface_points: int = 256,
    n_boundary_points: int = 512,
    min_subdomain_width: float = 0.01,
    rng: np.random.Generator | None = None,
    inlet_v: float = 0.0,
    outlet_pressure: float = 0.0,
    wall_u: float = 0.0,
    wall_v: float = 0.0,
    field_map: Mapping[str, str] = DEFAULT_FIELD_MAP,
) -> dict[str, Any]:
    """Build all randomized training samples for one complete CFD channel."""

    n_subdomains = int(n_subdomains)
    n_realizations = int(n_realizations)
    n_interface_points = int(n_interface_points)
    n_boundary_points = int(n_boundary_points)
    if n_subdomains < 1 or n_realizations < 1:
        raise ValueError("n_subdomains and n_realizations must be >= 1")
    if n_interface_points <= 0 or n_boundary_points <= 1:
        raise ValueError("Invalid interface or boundary sensor count")
    if tuple(field_map.keys()) != OUTPUT_FIELDS:
        raise ValueError(
            f"Current schema requires field_map keys exactly {OUTPUT_FIELDS}"
        )
    if rng is None:
        rng = np.random.default_rng()

    centers, mesh_info = read_fluent_cell_centers(mesh_path)
    fields = read_fluent_cell_fields(dat_path, field_map)
    values_all = np.column_stack([fields[name] for name in OUTPUT_FIELDS]).astype(np.float64)
    if len(values_all) != len(centers):
        raise ValueError(
            f"Case {case_id}: mesh has {len(centers)} cells but solution has {len(values_all)}"
        )

    config = load_channel_config(design_path)
    if config["AR"] is None or config["Uin_mps"] is None:
        raise ValueError(f"Case {case_id}: metadata.AR and metadata.Uin_mps are required")

    x_min = float(mesh_info["x_min_mm"])
    x_max = float(mesh_info["x_max_mm"])
    config_x = np.asarray(config["x_points"], dtype=np.float64)
    config_span = float(config_x[-1] - config_x[0])
    if config_span <= 0.0:
        raise ValueError(f"Case {case_id}: design x-points must span a positive length")

    geometry_scale = (x_max - x_min) / config_span
    wall_x = x_min + (config_x - config_x[0]) * geometry_scale
    wall_bottom = np.asarray(config["y_bottom_points"], dtype=np.float64) * geometry_scale
    wall_top = np.asarray(config["y_top_points"], dtype=np.float64) * geometry_scale
    reference_length = max(float(config["L_mm"]) * geometry_scale, 1.0e-12)
    reference_length_mm = float(config["L_mm"])
    y_origin = 0.0
    inlet_u = float(config["Uin_mps"])
    aspect_ratio = float(int(config["AR"]))

    def y_bottom(x_query):
        return np.interp(np.asarray(x_query, dtype=np.float64), wall_x, wall_bottom)

    def y_top(x_query):
        return np.interp(np.asarray(x_query, dtype=np.float64), wall_x, wall_top)

    valid = np.all(np.isfinite(centers), axis=1) & np.all(np.isfinite(values_all), axis=1)
    x = centers[valid, 0].astype(np.float64)
    y = centers[valid, 1].astype(np.float64)
    values = values_all[valid].astype(np.float64)
    if x.size == 0:
        raise ValueError(f"Case {case_id}: no finite Fluent cells remain")

    # Expensive and realization-invariant: construct once per complete channel.
    linear, nearest = _make_interpolators(x, y, values)

    y_interface_fraction = np.linspace(0.0, 1.0, n_interface_points + 2)[1:-1]
    x_wall_fraction = np.linspace(0.0, 1.0, n_boundary_points)
    samples: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []

    for realization_id in range(n_realizations):
        print(
            f"Processing case {case_id} realization={realization_id} "
            f"(n_subdomains={n_subdomains})",
            flush=True,
        )
        x_edges = sample_random_x_edges(
            x_min, x_max, n_subdomains, min_subdomain_width, rng
        )

        for subdomain_id in range(n_subdomains):
            x_left = float(x_edges[subdomain_id])
            x_right = float(x_edges[subdomain_id + 1])
            width = x_right - x_left
            inv_width = 1.0 / width
            local_aspect = width / reference_length

            yb_left, yt_left = float(y_bottom(x_left)), float(y_top(x_left))
            yb_right, yt_right = float(y_bottom(x_right)), float(y_top(x_right))

            cell_mask = (x >= x_left) & (x <= x_right)
            if not np.any(cell_mask):
                raise ValueError(
                    f"Case {case_id}, realization {realization_id}, "
                    f"subdomain {subdomain_id}: no cell centers"
                )

            x_cell = x[cell_mask]
            y_cell = y[cell_mask]
            target = values[cell_mask].astype(np.float32)
            query = np.column_stack(
                [
                    ((x_cell - x_left) * inv_width).astype(np.float32),
                    ((y_cell - y_origin) / reference_length).astype(np.float32),
                ]
            ).astype(np.float32, copy=False)

            branch_parts = []

            y_left = yb_left + y_interface_fraction * (yt_left - yb_left)
            if subdomain_id == 0:
                left_values, left_known = _boundary_values(
                    "inlet", n_interface_points, inlet_u, inlet_v,
                    outlet_pressure, wall_u, wall_v,
                )
            else:
                left_values = _interp_fields(
                    linear, nearest,
                    np.full(n_interface_points, x_left, dtype=np.float64), y_left,
                )
                left_known = np.ones((n_interface_points, len(OUTPUT_FIELDS)), dtype=np.float32)
            branch_parts.append(
                _stack_branch(
                    np.zeros(n_interface_points, dtype=np.float32),
                    ((y_left - y_origin) / reference_length).astype(np.float32),
                    np.zeros(n_interface_points, dtype=np.float32),
                    np.ones(n_interface_points, dtype=np.float32),
                    left_values, left_known, local_aspect,
                )
            )

            y_right = yb_right + y_interface_fraction * (yt_right - yb_right)
            if subdomain_id == n_subdomains - 1:
                right_values, right_known = _boundary_values(
                    "outlet", n_interface_points, inlet_u, inlet_v,
                    outlet_pressure, wall_u, wall_v,
                )
            else:
                right_values = _interp_fields(
                    linear, nearest,
                    np.full(n_interface_points, x_right, dtype=np.float64), y_right,
                )
                right_known = np.ones((n_interface_points, len(OUTPUT_FIELDS)), dtype=np.float32)
            branch_parts.append(
                _stack_branch(
                    np.ones(n_interface_points, dtype=np.float32),
                    ((y_right - y_origin) / reference_length).astype(np.float32),
                    np.zeros(n_interface_points, dtype=np.float32),
                    np.ones(n_interface_points, dtype=np.float32),
                    right_values, right_known, local_aspect,
                )
            )

            x_wall_physical = x_left + x_wall_fraction * width
            x_wall_local = x_wall_fraction.astype(np.float32)

            bottom_values, bottom_known = _boundary_values(
                "bottom", n_boundary_points, inlet_u, inlet_v,
                outlet_pressure, wall_u, wall_v,
            )
            branch_parts.append(
                _stack_branch(
                    x_wall_local,
                    ((y_bottom(x_wall_physical) - y_origin) / reference_length).astype(np.float32),
                    np.ones(n_boundary_points, dtype=np.float32),
                    np.zeros(n_boundary_points, dtype=np.float32),
                    bottom_values, bottom_known, local_aspect,
                )
            )

            top_values, top_known = _boundary_values(
                "top", n_boundary_points, inlet_u, inlet_v,
                outlet_pressure, wall_u, wall_v,
            )
            branch_parts.append(
                _stack_branch(
                    x_wall_local,
                    ((y_top(x_wall_physical) - y_origin) / reference_length).astype(np.float32),
                    np.ones(n_boundary_points, dtype=np.float32),
                    np.zeros(n_boundary_points, dtype=np.float32),
                    top_values, top_known, local_aspect,
                )
            )

            branch = np.concatenate(branch_parts, axis=0).astype(np.float32, copy=False)
            samples.append({"branch": branch, "query": query, "target": target})
            metadata.append(
                {
                    "case_id": case_id,
                    "aspect_ratio": aspect_ratio,
                    "subdomain_id": int(subdomain_id),
                    "realization_id": int(realization_id),
                    "x_left_mm": x_left,
                    "x_right_mm": x_right,
                    "y_bottom_left_mm": yb_left,
                    "y_top_left_mm": yt_left,
                    "y_bottom_right_mm": yb_right,
                    "y_top_right_mm": yt_right,
                    "y_bottom_mm": y_origin,
                    "y_top_mm": y_origin + reference_length,
                    "y_local_origin_mm": y_origin,
                    "y_local_scale_mm": reference_length,
                    "reference_length_mm": reference_length_mm,
                    "reference_length_mesh": reference_length,
                    "local_aspect_ratio": float(local_aspect),
                    "x_left_is_sharp_control_point": False,
                    "x_right_is_sharp_control_point": False,
                    "n_cells": int(query.shape[0]),
                }
            )

    expected_samples = n_subdomains * n_realizations
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"Case {case_id}: expected {expected_samples} samples, built {len(samples)}"
        )

    return {
        "samples": samples,
        "metadata": metadata,
        "branch_channel_names": list(BRANCH_CHANNEL_NAMES),
        "trunk_channel_names": list(TRUNK_CHANNEL_NAMES),
        "output_channel_names": list(OUTPUT_FIELDS),
        "case_ids": [row["case_id"] for row in metadata],
        "subdomain_id": np.asarray([row["subdomain_id"] for row in metadata], dtype=np.int32),
        "realization_id": np.asarray([row["realization_id"] for row in metadata], dtype=np.int32),
        "local_aspect_ratio": np.asarray(
            [row["local_aspect_ratio"] for row in metadata], dtype=np.float32
        ),
        "n_cells": np.asarray([row["n_cells"] for row in metadata], dtype=np.int64),
        "n_subdomains": n_subdomains,
        "n_subdomains_per_case": [n_subdomains],
        "n_subdomains_per_realization": [n_subdomains] * n_realizations,
        "n_interface_points": n_interface_points,
        "n_boundary_points": n_boundary_points,
        "interface_placement": "random",
        "interface_jitter": 0.0,
        "min_subdomain_width": float(min_subdomain_width),
        "insert_sharp_control_point_interfaces": False,
        "horizontal_interface": False,
        "horizontal_interface_jitter": 0.0,
        "n_realizations": n_realizations,
        "mesh_info": mesh_info,
    }
