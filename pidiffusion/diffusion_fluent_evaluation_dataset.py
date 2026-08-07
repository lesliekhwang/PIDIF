"""Build deterministic Fluent subdomain datasets for PIDiffusion evaluation.

This module intentionally leaves ``pidiffusion.diffusion_fluent_dataset``
unchanged because that file is provenance-locked by the finalized randomized
training HDF5.  The validated Fluent readers, interpolation helpers, branch
schema, and boundary-condition helpers are reused here; only the deterministic
x-interface placement is new.

Supported evaluation decompositions:
- ``control_points``: use every design JSON control-point x coordinate as an edge;
- ``ar1``: use equal-width strips with count ``round(channel_length / L_ref)``.

Both modes use one deterministic realization per complete channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pidiffusion.diffusion_fluent_dataset import (
    BRANCH_CHANNEL_NAMES,
    DEFAULT_FIELD_MAP,
    OUTPUT_FIELDS,
    TRUNK_CHANNEL_NAMES,
    _boundary_values,
    _interp_fields,
    _make_interpolators,
    _stack_branch,
    load_channel_config,
    read_fluent_cell_centers,
    read_fluent_cell_fields,
)

PathLike = str | Path


def _prepare_channel(
    *,
    case_id: str,
    design_path: PathLike,
    mesh_path: PathLike,
    dat_path: PathLike,
    field_map: Mapping[str, str],
) -> dict[str, Any]:
    """Load one complete CFD channel and prepare invariant geometry/fields."""

    if tuple(field_map.keys()) != OUTPUT_FIELDS:
        raise ValueError(
            f"Current schema requires field_map keys exactly {OUTPUT_FIELDS}"
        )

    centers, mesh_info = read_fluent_cell_centers(mesh_path)
    fields = read_fluent_cell_fields(dat_path, field_map)
    values_all = np.column_stack(
        [fields[name] for name in OUTPUT_FIELDS]
    ).astype(np.float64)

    if len(values_all) != len(centers):
        raise ValueError(
            f"Case {case_id}: mesh has {len(centers)} cells but solution has "
            f"{len(values_all)}"
        )

    config = load_channel_config(design_path)
    if config["AR"] is None or config["Uin_mps"] is None:
        raise ValueError(
            f"Case {case_id}: metadata.AR and metadata.Uin_mps are required"
        )

    x_min = float(mesh_info["x_min_mm"])
    x_max = float(mesh_info["x_max_mm"])
    config_x = np.asarray(config["x_points"], dtype=np.float64)
    config_span = float(config_x[-1] - config_x[0])
    if config_span <= 0.0:
        raise ValueError(
            f"Case {case_id}: design x-points must span a positive length"
        )

    geometry_scale = (x_max - x_min) / config_span
    wall_x = x_min + (config_x - config_x[0]) * geometry_scale
    wall_bottom = (
        np.asarray(config["y_bottom_points"], dtype=np.float64)
        * geometry_scale
    )
    wall_top = (
        np.asarray(config["y_top_points"], dtype=np.float64)
        * geometry_scale
    )
    reference_length = max(
        float(config["L_mm"]) * geometry_scale,
        1.0e-12,
    )

    valid = (
        np.all(np.isfinite(centers), axis=1)
        & np.all(np.isfinite(values_all), axis=1)
    )
    x = centers[valid, 0].astype(np.float64)
    y = centers[valid, 1].astype(np.float64)
    values = values_all[valid].astype(np.float64)
    if x.size == 0:
        raise ValueError(f"Case {case_id}: no finite Fluent cells remain")

    linear, nearest = _make_interpolators(x, y, values)

    return {
        "case_id": case_id,
        "mesh_info": mesh_info,
        "config": config,
        "x": x,
        "y": y,
        "values": values,
        "x_min": x_min,
        "x_max": x_max,
        "wall_x": wall_x,
        "wall_bottom": wall_bottom,
        "wall_top": wall_top,
        "reference_length": reference_length,
        "reference_length_mm": float(config["L_mm"]),
        "y_origin": 0.0,
        "inlet_u": float(config["Uin_mps"]),
        "aspect_ratio": float(int(config["AR"])),
        "linear": linear,
        "nearest": nearest,
    }


def _validate_edges(
    x_edges: np.ndarray,
    *,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Validate deterministic x-strip edges in the mesh coordinate frame."""

    edges = np.asarray(x_edges, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        raise ValueError("x_edges must contain at least two coordinates")
    if not np.all(np.isfinite(edges)):
        raise ValueError("x_edges contains non-finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("x_edges must be strictly increasing")
    if not np.isclose(edges[0], x_min, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"First edge {edges[0]} does not match mesh x_min {x_min}"
        )
    if not np.isclose(edges[-1], x_max, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Last edge {edges[-1]} does not match mesh x_max {x_max}"
        )
    return edges


def _build_channel_from_edges(
    *,
    prepared: Mapping[str, Any],
    x_edges: np.ndarray,
    interface_placement: str,
    n_interface_points: int,
    n_boundary_points: int,
    inlet_v: float,
    outlet_pressure: float,
    wall_u: float,
    wall_v: float,
) -> dict[str, Any]:
    """Build one deterministic full-channel decomposition from fixed x edges."""

    n_interface_points = int(n_interface_points)
    n_boundary_points = int(n_boundary_points)
    if n_interface_points <= 0 or n_boundary_points <= 1:
        raise ValueError("Invalid interface or boundary sensor count")

    case_id = str(prepared["case_id"])
    x = np.asarray(prepared["x"], dtype=np.float64)
    y = np.asarray(prepared["y"], dtype=np.float64)
    values = np.asarray(prepared["values"], dtype=np.float64)
    x_min = float(prepared["x_min"])
    x_max = float(prepared["x_max"])
    wall_x = np.asarray(prepared["wall_x"], dtype=np.float64)
    wall_bottom = np.asarray(prepared["wall_bottom"], dtype=np.float64)
    wall_top = np.asarray(prepared["wall_top"], dtype=np.float64)
    reference_length = float(prepared["reference_length"])
    reference_length_mm = float(prepared["reference_length_mm"])
    y_origin = float(prepared["y_origin"])
    inlet_u = float(prepared["inlet_u"])
    aspect_ratio = float(prepared["aspect_ratio"])
    linear = prepared["linear"]
    nearest = prepared["nearest"]

    edges = _validate_edges(
        x_edges,
        x_min=x_min,
        x_max=x_max,
    )
    n_subdomains = int(edges.size - 1)

    def y_bottom(x_query: np.ndarray | float) -> np.ndarray:
        return np.interp(
            np.asarray(x_query, dtype=np.float64),
            wall_x,
            wall_bottom,
        )

    def y_top(x_query: np.ndarray | float) -> np.ndarray:
        return np.interp(
            np.asarray(x_query, dtype=np.float64),
            wall_x,
            wall_top,
        )

    y_interface_fraction = np.linspace(
        0.0,
        1.0,
        n_interface_points + 2,
        dtype=np.float64,
    )[1:-1]
    x_wall_fraction = np.linspace(
        0.0,
        1.0,
        n_boundary_points,
        dtype=np.float64,
    )

    print(
        f"Processing case {case_id} realization=0 "
        f"(n_subdomains={n_subdomains}, placement={interface_placement})",
        flush=True,
    )

    samples: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []

    for subdomain_id in range(n_subdomains):
        x_left = float(edges[subdomain_id])
        x_right = float(edges[subdomain_id + 1])
        width = x_right - x_left
        inv_width = 1.0 / width
        local_aspect = width / reference_length

        yb_left = float(y_bottom(x_left))
        yt_left = float(y_top(x_left))
        yb_right = float(y_bottom(x_right))
        yt_right = float(y_top(x_right))

        cell_mask = (x >= x_left) & (x <= x_right)
        if not np.any(cell_mask):
            raise ValueError(
                f"Case {case_id}, subdomain {subdomain_id}: no cell centers"
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

        branch_parts: list[np.ndarray] = []

        y_left = yb_left + y_interface_fraction * (yt_left - yb_left)
        if subdomain_id == 0:
            left_values, left_known = _boundary_values(
                "inlet",
                n_interface_points,
                inlet_u,
                inlet_v,
                outlet_pressure,
                wall_u,
                wall_v,
            )
        else:
            left_values = _interp_fields(
                linear,
                nearest,
                np.full(n_interface_points, x_left, dtype=np.float64),
                y_left,
            )
            left_known = np.ones(
                (n_interface_points, len(OUTPUT_FIELDS)),
                dtype=np.float32,
            )
        branch_parts.append(
            _stack_branch(
                np.zeros(n_interface_points, dtype=np.float32),
                ((y_left - y_origin) / reference_length).astype(np.float32),
                np.zeros(n_interface_points, dtype=np.float32),
                np.ones(n_interface_points, dtype=np.float32),
                left_values,
                left_known,
                local_aspect,
            )
        )

        y_right = yb_right + y_interface_fraction * (yt_right - yb_right)
        if subdomain_id == n_subdomains - 1:
            right_values, right_known = _boundary_values(
                "outlet",
                n_interface_points,
                inlet_u,
                inlet_v,
                outlet_pressure,
                wall_u,
                wall_v,
            )
        else:
            right_values = _interp_fields(
                linear,
                nearest,
                np.full(n_interface_points, x_right, dtype=np.float64),
                y_right,
            )
            right_known = np.ones(
                (n_interface_points, len(OUTPUT_FIELDS)),
                dtype=np.float32,
            )
        branch_parts.append(
            _stack_branch(
                np.ones(n_interface_points, dtype=np.float32),
                ((y_right - y_origin) / reference_length).astype(np.float32),
                np.zeros(n_interface_points, dtype=np.float32),
                np.ones(n_interface_points, dtype=np.float32),
                right_values,
                right_known,
                local_aspect,
            )
        )

        x_wall_physical = x_left + x_wall_fraction * width
        x_wall_local = x_wall_fraction.astype(np.float32)

        bottom_values, bottom_known = _boundary_values(
            "bottom",
            n_boundary_points,
            inlet_u,
            inlet_v,
            outlet_pressure,
            wall_u,
            wall_v,
        )
        branch_parts.append(
            _stack_branch(
                x_wall_local,
                (
                    (y_bottom(x_wall_physical) - y_origin)
                    / reference_length
                ).astype(np.float32),
                np.ones(n_boundary_points, dtype=np.float32),
                np.zeros(n_boundary_points, dtype=np.float32),
                bottom_values,
                bottom_known,
                local_aspect,
            )
        )

        top_values, top_known = _boundary_values(
            "top",
            n_boundary_points,
            inlet_u,
            inlet_v,
            outlet_pressure,
            wall_u,
            wall_v,
        )
        branch_parts.append(
            _stack_branch(
                x_wall_local,
                (
                    (y_top(x_wall_physical) - y_origin)
                    / reference_length
                ).astype(np.float32),
                np.ones(n_boundary_points, dtype=np.float32),
                np.zeros(n_boundary_points, dtype=np.float32),
                top_values,
                top_known,
                local_aspect,
            )
        )

        branch = np.concatenate(branch_parts, axis=0).astype(
            np.float32,
            copy=False,
        )
        samples.append(
            {
                "branch": branch,
                "query": query,
                "target": target,
            }
        )
        metadata.append(
            {
                "case_id": case_id,
                "aspect_ratio": aspect_ratio,
                "subdomain_id": int(subdomain_id),
                "realization_id": 0,
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

    return {
        "samples": samples,
        "metadata": metadata,
        "branch_channel_names": list(BRANCH_CHANNEL_NAMES),
        "trunk_channel_names": list(TRUNK_CHANNEL_NAMES),
        "output_channel_names": list(OUTPUT_FIELDS),
        "case_ids": [row["case_id"] for row in metadata],
        "subdomain_id": np.asarray(
            [row["subdomain_id"] for row in metadata],
            dtype=np.int32,
        ),
        "realization_id": np.zeros(n_subdomains, dtype=np.int32),
        "local_aspect_ratio": np.asarray(
            [row["local_aspect_ratio"] for row in metadata],
            dtype=np.float32,
        ),
        "n_cells": np.asarray(
            [row["n_cells"] for row in metadata],
            dtype=np.int64,
        ),
        "n_subdomains": n_subdomains,
        "n_subdomains_per_case": [n_subdomains],
        "n_subdomains_per_realization": [n_subdomains],
        "n_interface_points": n_interface_points,
        "n_boundary_points": n_boundary_points,
        "interface_placement": interface_placement,
        "interface_jitter": 0.0,
        "min_subdomain_width": 0.0,
        "insert_sharp_control_point_interfaces": False,
        "horizontal_interface": False,
        "horizontal_interface_jitter": 0.0,
        "n_realizations": 1,
        "mesh_info": prepared["mesh_info"],
        "x_edges": edges.copy(),
    }


def build_control_point_channel_dataset(
    *,
    case_id: str,
    design_path: PathLike,
    mesh_path: PathLike,
    dat_path: PathLike,
    n_interface_points: int = 256,
    n_boundary_points: int = 512,
    inlet_v: float = 0.0,
    outlet_pressure: float = 0.0,
    wall_u: float = 0.0,
    wall_v: float = 0.0,
    field_map: Mapping[str, str] = DEFAULT_FIELD_MAP,
) -> dict[str, Any]:
    """Build one geometry-control-point evaluation decomposition."""

    prepared = _prepare_channel(
        case_id=case_id,
        design_path=design_path,
        mesh_path=mesh_path,
        dat_path=dat_path,
        field_map=field_map,
    )

    config_x = np.asarray(
        prepared["config"]["x_points"],
        dtype=np.float64,
    )
    config_span = float(config_x[-1] - config_x[0])
    x_min = float(prepared["x_min"])
    x_max = float(prepared["x_max"])
    geometry_scale = (x_max - x_min) / config_span
    x_edges = x_min + (config_x - config_x[0]) * geometry_scale

    return _build_channel_from_edges(
        prepared=prepared,
        x_edges=x_edges,
        interface_placement="control_points",
        n_interface_points=n_interface_points,
        n_boundary_points=n_boundary_points,
        inlet_v=inlet_v,
        outlet_pressure=outlet_pressure,
        wall_u=wall_u,
        wall_v=wall_v,
    )


def build_ar1_channel_dataset(
    *,
    case_id: str,
    design_path: PathLike,
    mesh_path: PathLike,
    dat_path: PathLike,
    n_interface_points: int = 256,
    n_boundary_points: int = 512,
    inlet_v: float = 0.0,
    outlet_pressure: float = 0.0,
    wall_u: float = 0.0,
    wall_v: float = 0.0,
    field_map: Mapping[str, str] = DEFAULT_FIELD_MAP,
) -> dict[str, Any]:
    """Build one equal-width nominal-AR≈1 evaluation decomposition."""

    prepared = _prepare_channel(
        case_id=case_id,
        design_path=design_path,
        mesh_path=mesh_path,
        dat_path=dat_path,
        field_map=field_map,
    )

    x_min = float(prepared["x_min"])
    x_max = float(prepared["x_max"])
    reference_length = float(prepared["reference_length"])
    n_subdomains = max(
        1,
        int(round((x_max - x_min) / reference_length)),
    )
    x_edges = np.linspace(
        x_min,
        x_max,
        n_subdomains + 1,
        dtype=np.float64,
    )

    return _build_channel_from_edges(
        prepared=prepared,
        x_edges=x_edges,
        interface_placement="ar1",
        n_interface_points=n_interface_points,
        n_boundary_points=n_boundary_points,
        inlet_v=inlet_v,
        outlet_pressure=outlet_pressure,
        wall_u=wall_u,
        wall_v=wall_v,
    )
