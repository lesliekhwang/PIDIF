"""
Non-grid Fluent-HDF5 dataset construction for point-set DeepONet models.

This module intentionally does not create Cartesian grid data.  Each subdomain
sample keeps the original Fluent cell centers as trunk/query points and the
original Fluent cell-centered p/T/u/v values as targets.

Branch representation
---------------------
Each sample has boundary/interface sensor points with features:

    x_local, y_local,
    wall_mask, interface_mask,
    boundary_pressure, boundary_temperature, boundary_u, boundary_v,
    known_pressure, known_temperature, known_u, known_v,
    local_aspect_ratio

Exterior boundaries use prescribed boundary-condition values and known_* masks.
Interior interfaces use values interpolated from the global Fluent solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

PathLike = Union[str, Path]
CaseFiles = Mapping[int, Mapping[str, PathLike]]

FIELD_MAP = {"pressure": "SV_P", "temperature": "SV_T", "u": "SV_U", "v": "SV_V"}

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

def _sorted_numeric_keys(group):
    return sorted(group.keys(), key=lambda s: int(s) if str(s).isdigit() else str(s))

def _first_dataset(group):
    for k in _sorted_numeric_keys(group):
        if isinstance(group[k], h5py.Dataset):
            return group[k]
    raise KeyError(f"No dataset found below {group.name}")

def _read_2node_faces(face_zone_group):
    nnodes = face_zone_group["nnodes"][()].astype(np.int64)
    nodes = face_zone_group["nodes"][()].astype(np.int64) - 1
    if not np.all(nnodes == 2):
        raise NotImplementedError("This code assumes a 2-D mesh where every face has 2 nodes.")
    return nodes.reshape(-1, 2)

def _auto_to_mm(coords):
    coords = coords.astype(np.float64, copy=True)
    xmax = np.nanmax(coords[:, 0])
    ymax = np.nanmax(coords[:, 1])
    # .cas.h5 stores m while .msh.h5 stores mm. Convert meter-scale data to mm.
    if xmax <= 100.0 and ymax <= 10.0:
        coords *= 1000.0
    return coords

def read_fluent_cell_centers(mesh_h5, convert_to_mm="auto"):
    """Return cell-center coordinates from Fluent .msh.h5 or .cas.h5."""
    mesh_h5 = Path(mesh_h5)
    with h5py.File(mesh_h5, "r") as f:
        mesh = f["meshes/1"]

        coords = _first_dataset(mesh["nodes/coords"])[()]
        if convert_to_mm == "auto":
            coords = _auto_to_mm(coords)
        elif convert_to_mm:
            coords = coords.astype(np.float64) * 1000.0
        else:
            coords = coords.astype(np.float64)

        cell_min_id = int(np.min(mesh["cells/zoneTopology/minId"][()]))
        cell_max_id = int(np.max(mesh["cells/zoneTopology/maxId"][()]))
        n_cells = cell_max_id - cell_min_id + 1

        face_nodes_group = mesh["faces/nodes"]
        c0_group = mesh["faces/c0"]
        c1_group = mesh["faces/c1"]
        zone_keys = _sorted_numeric_keys(face_nodes_group)

        all_face_nodes, all_c0, all_c1 = [], [], []

        if len(zone_keys) > 1:
            # .msh.h5 layout: separate groups for boundary/interior face zones.
            for zk in zone_keys:
                fn = _read_2node_faces(face_nodes_group[zk])
                nfaces = fn.shape[0]
                c0 = c0_group[zk][()].astype(np.int64) if zk in c0_group else np.zeros(nfaces, dtype=np.int64)
                c1 = c1_group[zk][()].astype(np.int64) if zk in c1_group else np.zeros(nfaces, dtype=np.int64)
                all_face_nodes.append(fn)
                all_c0.append(c0)
                all_c1.append(c1)
        else:
            # .cas.h5 layout in your example: all faces in one group.
            zk = zone_keys[0]
            fn = _read_2node_faces(face_nodes_group[zk])
            nfaces = fn.shape[0]

            c0_key = _sorted_numeric_keys(c0_group)[0]
            c0 = c0_group[c0_key][()].astype(np.int64)
            if len(c0) != nfaces:
                raise ValueError(f"Cannot align c0 values: {len(c0)} values for {nfaces} faces")

            c1_key = _sorted_numeric_keys(c1_group)[0]
            c1_raw = c1_group[c1_key][()].astype(np.int64)
            if len(c1_raw) == nfaces:
                c1 = c1_raw
            else:
                # Usually c1 is stored only for interior faces. Reinsert it by face-zone ranges.
                c1 = np.zeros(nfaces, dtype=np.int64)
                ztop = mesh["faces/zoneTopology"]
                z_c1 = ztop["c1"][()] if "c1" in ztop else None
                z_min = ztop["minId"][()]
                z_max = ztop["maxId"][()]
                raw_pos = 0
                if z_c1 is not None:
                    for has_c1, lo, hi in zip(z_c1 != 0, z_min, z_max):
                        count = int(hi - lo + 1)
                        if has_c1:
                            c1[int(lo) - 1 : int(hi)] = c1_raw[raw_pos : raw_pos + count]
                            raw_pos += count
                if raw_pos != len(c1_raw):
                    # Fallback: boundary faces first, interior faces last.
                    c1[:] = 0
                    c1[-len(c1_raw):] = c1_raw

            all_face_nodes.append(fn)
            all_c0.append(c0)
            all_c1.append(c1)

        face_nodes = np.vstack(all_face_nodes)   # 0-based node ids, shape (n_faces, 2)
        c0 = np.concatenate(all_c0)              # 1-based cell ids; 0 means no adjacent cell
        c1 = np.concatenate(all_c1)

    # Vectorized cell -> unique node pairs, then vertex-average centroids.
    valid0 = c0 > 0
    valid1 = c1 > 0
    cell_idx = np.concatenate([
        np.repeat(c0[valid0] - cell_min_id, 2),
        np.repeat(c1[valid1] - cell_min_id, 2),
    ])
    node_idx = np.concatenate([
        face_nodes[valid0].reshape(-1),
        face_nodes[valid1].reshape(-1),
    ])

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
    for d in range(2):
        sums = np.bincount(cell_idx, weights=coords[node_idx, d], minlength=n_cells)
        centers[valid_cells, d] = sums[valid_cells] / node_count[valid_cells]

    x0, x1 = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
    y0, y1 = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))
    mesh_info = {
        "mesh_h5": str(mesh_h5),
        "n_nodes": int(coords.shape[0]),
        "n_cells": int(n_cells),
        "nodes_per_cell_min": int(node_count.min()),
        "nodes_per_cell_max": int(node_count.max()),
        "x_min_mm": x0,
        "x_max_mm": x1,
        "y_min_mm": y0,
        "y_max_mm": y1,
        "Lx_mm": x1 - x0,
        "Ly_mm": y1 - y0,
        "inferred_AR": (x1 - x0) / (y1 - y0),
    }
    return centers, mesh_info

def read_fluent_cell_fields(dat_h5, field_map=FIELD_MAP):
    """Read Fluent cell-centered fields from .dat.h5."""
    fields = {}
    with h5py.File(dat_h5, "r") as f:
        base = f["results/1/phase-1/cells"]
        for clean_name, fluent_name in field_map.items():
            if fluent_name not in base:
                raise KeyError(f"Missing {fluent_name} in {base.name}")
            g = base[fluent_name]
            pieces = [g[k][()].reshape(-1) for k in _sorted_numeric_keys(g) if isinstance(g[k], h5py.Dataset)]
            fields[clean_name] = np.concatenate(pieces).astype(np.float64)
    return fields


def sample_constrained_x_edges(
    xmin: float,
    xmax: float,
    n_subdomains: int,
    min_subdomain_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample sorted x-edges with a minimum subdomain width.

    min_subdomain_width is interpreted as a fraction of the full span.
    """
    span = float(xmax) - float(xmin)
    n_subdomains = int(n_subdomains)
    min_width_frac = float(min_subdomain_width)

    if n_subdomains < 1:
        raise ValueError("n_subdomains must be >= 1")
    if min_width_frac < 0.0:
        raise ValueError("min_subdomain_width must be non-negative")
    if n_subdomains == 1:
        return np.asarray([xmin, xmax], dtype=np.float64)

    min_total = n_subdomains * min_width_frac * span
    if min_total > span:
        raise ValueError(
            "Infeasible subdomain constraints: "
            "n_subdomains * min_subdomain_width must be <= 1."
        )

    slack = span - min_total
    extras = rng.dirichlet(np.ones(n_subdomains, dtype=np.float64))
    widths = min_width_frac * span + slack * extras
    edges = np.concatenate([[float(xmin)], float(xmin) + np.cumsum(widths)])
    edges[-1] = float(xmax)
    return edges.astype(np.float64)

BRANCH_CHANNELS = CELL_BRANCH_CHANNELS
TRUNK_CHANNELS = CELL_TRUNK_CHANNELS
OUTPUT_CHANNELS = CELL_OUTPUT_CHANNELS


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _expand_n_subdomains(n_subdomains: Union[int, Sequence[int]], ars: Sequence[int]) -> List[int]:
    if isinstance(n_subdomains, (int, np.integer)):
        return [int(n_subdomains)] * len(ars)
    out = [int(v) for v in n_subdomains]
    if len(out) != len(ars):
        raise ValueError("n_subdomains must be an int or a sequence with the same length as ars")
    return out


def _ordered_values_from_fields(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in CELL_OUTPUT_CHANNELS if name not in fields]
    if missing:
        raise KeyError(f"Missing required field(s): {missing}. Available: {list(fields.keys())}")
    return np.column_stack([fields[name] for name in CELL_OUTPUT_CHANNELS]).astype(np.float64)


def _make_global_interpolators(x: np.ndarray, y: np.ndarray, values: np.ndarray):
    """Build global interpolators for p/T/u/v from Fluent cell centers."""
    points = np.column_stack([x, y])
    linear = []
    nearest = []
    for c in range(values.shape[1]):
        linear.append(LinearNDInterpolator(points, values[:, c]))
        nearest.append(NearestNDInterpolator(points, values[:, c]))
    return linear, nearest


def _interp_fields(linear, nearest, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
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
    Stack branch point features.

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
        raise ValueError("x_local, y_local, wall_mask, and interface_mask must have the same length")
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
    Return prescribed rectangular-channel boundary values and known masks.

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
    ar_list: Sequence[int],
    n_subdomains: Union[int, Sequence[int]] = 10,
    n_interface_points: int = 256,
    n_boundary_points: int = 256,
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

    No Cartesian grid is constructed.  The trunk/query points are the
    original Fluent cell centers inside each subdomain.

    Parameters
    ----------
    n_interface_points:
        Number of sensor points per vertical inlet/outlet/interface side.
    n_boundary_points:
        Number of sensor points per boundary side.
    """
    if rng is None:
        rng = np.random.default_rng()
    if bc_kwargs is None:
        bc_kwargs = {}

    sub_counts = _expand_n_subdomains(n_subdomains, ar_list)

    n_interface_points = int(n_interface_points)
    n_boundary_points = int(n_boundary_points)
    n_realizations = int(n_realizations)
    if n_interface_points <= 0 or n_boundary_points <= 1:
        raise ValueError("n_interface_points must be positive and n_boundary_points must be > 1")
    if n_realizations < 1:
        raise ValueError("n_realizations must be >= 1")

    samples: List[Dict[str, np.ndarray]] = []
    metadata: List[Dict[str, object]] = []
    raw_cases: Dict[Union[int, Tuple[int, int]], object] = {}

    for ar, n_sub in zip(ar_list, sub_counts):
        if ar not in case_files:
            print(f"Skipping AR={ar}: not in case_files")
            continue
        
        centers, mesh_info = read_fluent_cell_centers(case_files[ar]["mesh"])
        fields = read_fluent_cell_fields(case_files[ar]["dat"], field_map=field_map)
        values_all = _ordered_values_from_fields(fields)
        
        if len(values_all) != len(centers):
            raise ValueError(
                f"Mesh has {len(centers)} cells but .dat.h5 has {len(values_all)} values for AR={ar}."
            )

        for realization_id in range(n_realizations):
            print(f"Processing AR={ar} realization={realization_id}", flush=True)

            valid = np.all(np.isfinite(centers), axis=1) & np.all(np.isfinite(values_all), axis=1)
            x = centers[valid, 0].astype(np.float64)
            y = centers[valid, 1].astype(np.float64)
            values = values_all[valid].astype(np.float64)

            xmin = float(mesh_info["x_min_mm"])
            xmax = float(mesh_info["x_max_mm"])
            ymin = float(mesh_info["y_min_mm"])
            ymax = float(mesh_info["y_max_mm"])
            height = ymax - ymin

            n_sub = int(n_sub)
            if n_sub <= 0:
                raise ValueError("n_subdomains must be positive")

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
            x_wall_local = np.linspace(0.0, 1.0, n_boundary_points, dtype=np.float32)

            for s in range(n_sub):
                x0 = float(x_edges[s])
                x1 = float(x_edges[s + 1])
                width = x1 - x0
                local_aspect = width / max(height, 1.0e-12)

                if s == n_sub - 1:
                    cell_mask = (x >= x0) & (x <= x1)
                else:
                    cell_mask = (x >= x0) & (x < x1)
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

                branch_parts: List[np.ndarray] = []

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

                vals, known = make_rect_channel_bc_values("bottom", n_boundary_points, **bc_kwargs)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=x_wall_local,
                        y_local=np.zeros(n_boundary_points, dtype=np.float32),
                        wall_mask=np.ones(n_boundary_points, dtype=np.float32),
                        interface_mask=np.zeros(n_boundary_points, dtype=np.float32),
                        values=vals,
                        known_mask=known,
                        local_aspect_ratio=local_aspect,
                    )
                )

                vals, known = make_rect_channel_bc_values("top", n_boundary_points, **bc_kwargs)
                branch_parts.append(
                    _stack_cell_branch_features(
                        x_local=x_wall_local,
                        y_local=np.ones(n_boundary_points, dtype=np.float32),
                        wall_mask=np.ones(n_boundary_points, dtype=np.float32),
                        interface_mask=np.zeros(n_boundary_points, dtype=np.float32),
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
                        "realization_id": int(realization_id),
                        "x_left_mm": float(x0),
                        "x_right_mm": float(x1),
                        "y_bottom_mm": float(ymin),
                        "y_top_mm": float(ymax),
                        "local_aspect_ratio": float(local_aspect),
                        "n_cells": int(query.shape[0]),
                    }
                )

            if keep_raw_case_data:
                raw_key: Union[int, Tuple[int, int]] = int(ar) if n_realizations == 1 else (int(ar), int(realization_id))
                raw_cases[raw_key] = {
                    "x": x.astype(np.float32),
                    "y": y.astype(np.float32),
                    "values": values.astype(np.float32),
                    "mesh_info": mesh_info,
                    "x_edges_mm": x_edges.astype(np.float32),
                    "output_channel_names": list(CELL_OUTPUT_CHANNELS),
                }

    if not samples:
        raise ValueError("No samples were processed. Check case_files and AR range.")

    return {
        "samples": samples,
        "metadata": metadata,
        "branch_channel_names": list(CELL_BRANCH_CHANNELS),
        "trunk_channel_names": list(CELL_TRUNK_CHANNELS),
        "output_channel_names": list(CELL_OUTPUT_CHANNELS),
        "ars": np.asarray([m["aspect_ratio"] for m in metadata], dtype=np.float32),
        "subdomain_id": np.asarray([m["subdomain_id"] for m in metadata], dtype=np.int32),
        "realization_id": np.asarray([m["realization_id"] for m in metadata], dtype=np.int32),
        "local_aspect_ratio": np.asarray([m["local_aspect_ratio"] for m in metadata], dtype=np.float32),
        "n_cells": np.asarray([m["n_cells"] for m in metadata], dtype=np.int64),
        "n_subdomains": sub_counts if len(set(sub_counts)) > 1 else int(sub_counts[0]),
        "n_interface_points": int(n_interface_points),
        "n_boundary_points": int(n_boundary_points),
        "interface_placement": str(interface_placement),
        "interface_jitter": float(interface_jitter),
        "n_realizations": int(n_realizations),
        "raw_cases": raw_cases if keep_raw_case_data else None,
    }



def normalize_cell_branch_with_y(
    branch: np.ndarray,
    branch_channel_names: Sequence[str] = CELL_BRANCH_CHANNELS,
    target_y_normalizer=None,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    zero_unknown_values: bool = True,
) -> np.ndarray:
    """
    Normalize branch p/T/u/v with the target/output normalizer.

    Coordinates, masks, and known_* flags are left unchanged.  If known_* masks
    are present, unknown boundary quantities are set to zero in normalized space.
    """
    names = list(branch_channel_names)
    out = np.asarray(branch, dtype=np.float32).copy()
    value_idx = [
        names.index("boundary_pressure"),
        names.index("boundary_temperature"),
        names.index("boundary_u"),
        names.index("boundary_v"),
    ]

    if target_y_normalizer is not None:
        mean = target_y_normalizer.mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
        std = target_y_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float32)
        std = np.maximum(std, 1.0e-12)
        norm_values = (out[:, value_idx] - mean.reshape(1, 4)) / std.reshape(1, 4)

        known_names = ["known_pressure", "known_temperature", "known_u", "known_v"]
        if zero_unknown_values and all(k in names for k in known_names):
            known_idx = [names.index(k) for k in known_names]
            norm_values = norm_values * out[:, known_idx]

        out[:, value_idx] = norm_values.astype(np.float32)

    if local_aspect_mean is not None and local_aspect_std is not None:
        aspect_idx = names.index("local_aspect_ratio")
        out[:, aspect_idx] = (out[:, aspect_idx] - float(local_aspect_mean)) / max(float(local_aspect_std), 1.0e-12)

    return out.astype(np.float32, copy=False)


class DeepONetCellDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for variable-length Fluent cell-center DeepONet samples.

    Each item returns:
        branch:       (M, Cb)
        query:        (Q, 2)
        target:       (Q, 4)
        sample_index: scalar long
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

    def __getitem__(self, i: int):
        sample_index = int(self.indices[int(i)])
        sample = self.samples[sample_index]
        branch = np.asarray(sample["branch"], dtype=np.float32)
        query_all = np.asarray(sample["query"], dtype=np.float32)
        target_all = np.asarray(sample["target"], dtype=np.float32)

        point_idx = self._sample_query_indices(query_all.shape[0])
        query = query_all[point_idx]
        target = target_all[point_idx]

        branch = normalize_cell_branch_with_y(
            branch,
            branch_channel_names=self.branch_channel_names,
            target_y_normalizer=self.target_y_normalizer,
            local_aspect_mean=self.local_aspect_mean,
            local_aspect_std=self.local_aspect_std,
        )

        if self.target_y_normalizer is not None:
            target = self.target_y_normalizer.encode(target).detach().cpu().numpy().astype(np.float32)

        return (
            torch.from_numpy(branch).float(),
            torch.from_numpy(query).float(),
            torch.from_numpy(target).float(),
            torch.tensor(sample_index, dtype=torch.long),
        )


def save_deeponet_dataset_h5(dataset: Mapping[str, object], output_path: PathLike) -> None:
    """Save a variable-length cell dataset to HDF5."""
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(dataset["samples"])
    metadata = list(dataset["metadata"])
    with h5py.File(output_path, "w") as f:
        f.attrs["branch_channel_names"] = "\n".join(dataset["branch_channel_names"])
        f.attrs["trunk_channel_names"] = "\n".join(dataset["trunk_channel_names"])
        f.attrs["output_channel_names"] = "\n".join(dataset["output_channel_names"])
        f.attrs["n_interface_points"] = int(dataset["n_interface_points"])
        f.attrs["n_boundary_points"] = int(dataset["n_boundary_points"])
        f.attrs["include_interface_endpoints"] = bool(dataset.get("include_interface_endpoints", False))
        f.attrs["n_samples"] = len(samples)

        grp = f.create_group("samples")
        for i, sample in enumerate(samples):
            sg = grp.create_group(str(i))
            sg.create_dataset("branch", data=np.asarray(sample["branch"], dtype=np.float32), compression="gzip", compression_opts=4)
            sg.create_dataset("query", data=np.asarray(sample["query"], dtype=np.float32), compression="gzip", compression_opts=4)
            sg.create_dataset("target", data=np.asarray(sample["target"], dtype=np.float32), compression="gzip", compression_opts=4)

        meta_grp = f.create_group("metadata")
        keys = sorted({k for m in metadata for k in m.keys()})
        for key in keys:
            vals = [m.get(key, "") for m in metadata]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                meta_grp.create_dataset(key, data=np.asarray(vals))
            else:
                meta_grp.create_dataset(key, data=np.asarray(vals, dtype=h5py.string_dtype("utf-8")))


def load_deeponet_dataset_h5(path: PathLike) -> Dict[str, object]:
    """Load a variable-length cell dataset saved by save_deeponet_dataset_h5."""
    path = _as_path(path)
    with h5py.File(path, "r") as f:
        n_samples = int(f.attrs["n_samples"])
        samples = []
        for i in range(n_samples):
            sg = f["samples"][str(i)]
            samples.append(
                {
                    "branch": sg["branch"][:].astype(np.float32),
                    "query": sg["query"][:].astype(np.float32),
                    "target": sg["target"][:].astype(np.float32),
                }
            )

        metadata = []
        meta_grp = f["metadata"]
        keys = list(meta_grp.keys())
        for i in range(n_samples):
            row = {}
            for key in keys:
                value = meta_grp[key][i]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif hasattr(value, "item"):
                    value = value.item()
                row[key] = value
            metadata.append(row)

        return {
            "samples": samples,
            "metadata": metadata,
            "branch_channel_names": str(f.attrs["branch_channel_names"]).split("\n"),
            "trunk_channel_names": str(f.attrs["trunk_channel_names"]).split("\n"),
            "output_channel_names": str(f.attrs["output_channel_names"]).split("\n"),
            "n_interface_points": int(f.attrs["n_interface_points"]),
            "n_boundary_points": int(f.attrs["n_boundary_points"]),
            "include_interface_endpoints": bool(f.attrs.get("include_interface_endpoints", False)),
        }

def deeponet_cell_collate_fn(batch):
    """
    Collate DeepONet samples with fixed-size branch and variable-size query.

    Returns:
        branch:          (B, M, Cb)
        query_cat:       (N_total, 2)
        target_cat:      (N_total, 4)
        query_batch_id:  (N_total,)
        sample_idx:      (B,)
    """
    branches, queries, targets, sample_indices = zip(*batch)

    branch = torch.stack(branches, dim=0)      # (B, M, Cb)
    query_cat = torch.cat(queries, dim=0)      # (N_total, 2)
    target_cat = torch.cat(targets, dim=0)     # (N_total, 4)

    query_batch_id = torch.cat([
        torch.full((query.shape[0],), i, dtype=torch.long)
        for i, query in enumerate(queries)
    ])

    sample_idx = torch.stack(sample_indices, dim=0)

    return branch, query_cat, target_cat, query_batch_id, sample_idx