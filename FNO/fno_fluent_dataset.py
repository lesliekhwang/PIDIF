from pathlib import Path
import numpy as np
import h5py

FIELD_MAP = {"pressure": "SV_P", "temperature": "SV_T", "u": "SV_U", "v": "SV_V"}

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

def _fill_nan_grid_neighbor_mean(grid, max_iter=200):
    """
    Fill NaN cells in a (C, nx, ny) grid by repeated neighbor averaging.
    Uses only NumPy. Missing pattern is assumed identical across channels.
    """
    out = grid.copy()
    for _ in range(max_iter):
        valid = np.all(np.isfinite(out), axis=0)
        missing = ~valid
        if not np.any(missing):
            return out

        acc = np.zeros_like(out)
        cnt = np.zeros(valid.shape, dtype=np.float64)

        # 8-neighbor fill.
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue

                src_i0 = max(0, -di)
                src_i1 = out.shape[1] - max(0, di)
                src_j0 = max(0, -dj)
                src_j1 = out.shape[2] - max(0, dj)

                dst_i0 = max(0, di)
                dst_i1 = out.shape[1] - max(0, -di)
                dst_j0 = max(0, dj)
                dst_j1 = out.shape[2] - max(0, -dj)

                src_valid = valid[src_i0:src_i1, src_j0:src_j1]
                if np.any(src_valid):
                    vals = out[:, src_i0:src_i1, src_j0:src_j1]
                    acc[:, dst_i0:dst_i1, dst_j0:dst_j1] += np.where(src_valid[None, :, :], vals, 0.0)
                    cnt[dst_i0:dst_i1, dst_j0:dst_j1] += src_valid

        fill = missing & (cnt > 0)
        if not np.any(fill):
            break
        out[:, fill] = acc[:, fill] / cnt[fill]

    # Last-resort fill: channel means over valid entries.
    valid = np.all(np.isfinite(out), axis=0)
    if np.any(~valid):
        means = np.nanmean(out, axis=(1, 2))
        out[:, ~valid] = means[:, None]
    return out

def _rasterize_points_to_grid(x, y, values, x0, x1, y0, y1, nx, ny, dtype=np.float32):
    """
    Bin/average scattered cell-centered values to a fixed grid.
    Output shape is (C, nx, ny).
    """
    mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if not np.any(mask):
        raise ValueError("No cell centers found inside requested subdomain.")

    xm = x[mask]
    ym = y[mask]
    vm = values[mask]  # (n_points, C)

    ix = np.rint((xm - x0) / max(x1 - x0, 1e-12) * (nx - 1)).astype(np.int64)
    iy = np.rint((ym - y0) / max(y1 - y0, 1e-12) * (ny - 1)).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    flat = ix * ny + iy

    count = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    grid = np.full((values.shape[1], nx * ny), np.nan, dtype=np.float64)

    nonempty = count > 0
    for c in range(values.shape[1]):
        sums = np.bincount(flat, weights=vm[:, c], minlength=nx * ny)
        grid[c, nonempty] = sums[nonempty] / count[nonempty]

    grid = grid.reshape(values.shape[1], nx, ny)
    grid = _fill_nan_grid_neighbor_mean(grid)
    return grid.astype(dtype)

def _sample_vertical_profile_by_slab(x, y, values, x_line, y0, y1, ny, slab_half_width, dtype=np.float32):
    """Approximate field profile at x=x_line by averaging cells in a thin vertical slab."""
    width = float(slab_half_width)
    for _ in range(10):
        mask = (np.abs(x - x_line) <= width) & (y >= y0) & (y <= y1)
        if np.count_nonzero(mask) >= max(4, ny // 4):
            break
        width *= 2.0

    ym = y[mask]
    vm = values[mask]
    iy = np.rint((ym - y0) / max(y1 - y0, 1e-12) * (ny - 1)).astype(np.int64)
    iy = np.clip(iy, 0, ny - 1)

    count = np.bincount(iy, minlength=ny).astype(np.float64)
    prof = np.full((values.shape[1], ny), np.nan, dtype=np.float64)
    nonempty = count > 0
    for c in range(values.shape[1]):
        sums = np.bincount(iy, weights=vm[:, c], minlength=ny)
        prof[c, nonempty] = sums[nonempty] / count[nonempty]

    # Reuse 2-D filler by adding a dummy x dimension.
    prof = _fill_nan_grid_neighbor_mean(prof[:, None, :])[:, 0, :]
    return prof.astype(dtype)


def _sample_constrained_x_edges(xmin, xmax, n_subdomains, min_subdomain_width, rng):
    """
    Sample sorted x-edges with a lower bound on every subdomain width.

    min_subdomain_width is interpreted as a fraction of total span.
    For example, 0.01 means each subdomain has width >= 1% of (xmax - xmin).
    """
    span = float(xmax) - float(xmin)
    min_width_frac = float(min_subdomain_width)

    if n_subdomains < 1:
        raise ValueError(f"n_subdomains must be >= 1, got {n_subdomains}")
    if min_width_frac < 0:
        raise ValueError(
            f"min_subdomain_width must be non-negative, got {min_subdomain_width}"
        )
    if min_width_frac > 1:
        raise ValueError(
            f"min_subdomain_width must be <= 1 as a span fraction, got {min_subdomain_width}"
        )

    if n_subdomains == 1:
        return np.array([xmin, xmax], dtype=np.float64)

    min_total = n_subdomains * min_width_frac * span
    if min_total > span:
        raise ValueError(
            "Infeasible subdomain constraints: "
            f"{n_subdomains} * min_subdomain_width ({n_subdomains * min_width_frac:.6g}) "
            "must be <= 1 when min_subdomain_width is a span fraction."
        )

    slack = span - min_total

    # Dirichlet draws positive extras that sum to 1; this keeps widths smooth.
    extras = rng.dirichlet(np.ones(n_subdomains, dtype=np.float64))
    widths = (min_width_frac * span) + slack * extras
    x_edges = np.concatenate([[xmin], xmin + np.cumsum(widths)])
    x_edges[-1] = xmax
    return x_edges.astype(np.float64)

def build_case_subdomains(
    mesh_h5,
    dat_h5,
    aspect_ratio=None,
    n_subdomains=10,
    nx=64,
    ny=64,
    field_map=FIELD_MAP,
    ar_scale=50.0,
    dtype=np.float32,
    interface_placement="fixed",
    interface_jitter=0.0,
    min_subdomain_width=0.01,
    rng=None,
):
    """
    Convert one Fluent case into fixed-size FNO samples.

    Returns
    -------
    dict with
      X : (n_subdomains, 5, nx, ny)
          Input coordinate/parameter channels.
      Y : (n_subdomains, 4, nx, ny)
          Output field channels [pressure, temperature, u, v].
      interfaces : (n_subdomains + 1, 4, ny)
          Values along x-interfaces, including inlet and outlet.
    """
    centers, mesh_info = read_fluent_cell_centers(mesh_h5)
    fields = read_fluent_cell_fields(dat_h5, field_map=field_map)

    channel_names = list(fields.keys())
    values = np.column_stack([fields[name] for name in channel_names]).astype(np.float64)

    if len(values) != len(centers):
        raise ValueError(
            f"Mesh has {len(centers)} cells but .dat.h5 has {len(values)} field values. "
            "Check that the mesh/case and data files match."
        )

    valid = np.all(np.isfinite(centers), axis=1) & np.all(np.isfinite(values), axis=1)
    x = centers[valid, 0]
    y = centers[valid, 1]
    values = values[valid]

    if aspect_ratio is None:
        aspect_ratio = mesh_info["inferred_AR"]

    xmin, xmax = mesh_info["x_min_mm"], mesh_info["x_max_mm"]
    ymin, ymax = mesh_info["y_min_mm"], mesh_info["y_max_mm"]
    
    if interface_placement == "fixed":
        x_edges = np.linspace(xmin, xmax, n_subdomains + 1)
        if interface_jitter > 0:
            if rng is None:
                rng = np.random.default_rng()
            base_dx = (xmax - xmin) / n_subdomains
            jitter_dx = base_dx * interface_jitter
            noise = rng.uniform(
                low=-jitter_dx,
                high=jitter_dx,
                size=n_subdomains - 1,
            )
            
            x_edges[1:-1] += noise
            
            if np.any(np.diff(x_edges) <= 0):
                raise RuntimeError(
                    "Jittered x_edges are not strictly increasing. "
                    "Reduce the jitter range."
                )
                
    elif interface_placement == "random":
        if rng is None:
            rng = np.random.default_rng()
        x_edges = _sample_constrained_x_edges(
            xmin=xmin,
            xmax=xmax,
            n_subdomains=n_subdomains,
            min_subdomain_width=min_subdomain_width,
            rng=rng,
        )
    else:
        raise ValueError(f"Invalid interface_placement: {interface_placement}")
    
    print(f"Interface locations: {x_edges.tolist()}")
            
    y_grid = np.linspace(ymin, ymax, ny)

    X_blocks, Y_blocks, meta = [], [], []
    for i in range(n_subdomains):
        x0, x1 = x_edges[i], x_edges[i + 1]
        Y_block = _rasterize_points_to_grid(x, y, values, x0, x1, ymin, ymax, nx, ny, dtype=dtype)
        Y_blocks.append(Y_block)

        x_grid = np.linspace(x0, x1, nx)
        Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
        X_block = np.stack([
            (Xg - x0) / max(x1 - x0, 1e-12),
            (Yg - ymin) / max(ymax - ymin, 1e-12),
            (Xg - xmin) / max(xmax - xmin, 1e-12),
            np.full_like(Xg, float(aspect_ratio) / float(ar_scale)),
            np.full_like(Xg, i / max(n_subdomains - 1, 1)),
        ], axis=0).astype(dtype)
        X_blocks.append(X_block)

        meta.append({
            "aspect_ratio": float(aspect_ratio),
            "subdomain_id": int(i),
            "x_left_mm": float(x0),
            "x_right_mm": float(x1),
            "y_bottom_mm": float(ymin),
            "y_top_mm": float(ymax),
            "local_aspect_ratio": float(x1 - x0) / float(ymax - ymin),
        })

    dx_grid_each_subdomain = np.diff(x_edges) / max(nx - 1, 1)

    slab_half_widths = np.empty(n_subdomains + 1, dtype=np.float64)

    # Inlet and outlet.
    slab_half_widths[0] = dx_grid_each_subdomain[0]
    slab_half_widths[-1] = dx_grid_each_subdomain[-1]

    # Interior interfaces use the smaller neighboring grid spacing.
    if n_subdomains > 1:
        slab_half_widths[1:-1] = np.minimum(
            dx_grid_each_subdomain[:-1],
            dx_grid_each_subdomain[1:],
        )

    slab_half_widths = np.maximum(slab_half_widths, 1.0e-9)
    
    interfaces = np.stack([
        _sample_vertical_profile_by_slab(x, y, values, xe, ymin, ymax, ny, slab_half_width, dtype=dtype)
        for xe, slab_half_width in zip(x_edges, slab_half_widths)
    ], axis=0)

    return {
        "X": np.stack(X_blocks, axis=0),
        "Y": np.stack(Y_blocks, axis=0),
        "interfaces": interfaces,
        "x_edges_mm": x_edges.astype(dtype),
        "y_interface_mm": y_grid.astype(dtype),
        "input_channel_names": ["x_local", "y_norm", "x_global", "aspect_ratio_scaled", "subdomain_id_scaled"],
        "output_channel_names": channel_names,
        "metadata": meta,
        "mesh_info": mesh_info,
    }

def build_ar_dataset(case_files, ar_min=10, ar_max=20, nx=64, ny=64, n_subdomains=10):
    """
    Build one in-memory dataset for AR=10,...,20.

    case_files format:
    case_files = {
        10: {"mesh": "rect_ar10.msh.h5", "dat": "case_ar10.dat.h5"},
        11: {"mesh": "rect_ar11.msh.h5", "dat": "case_ar11.dat.h5"},
        ...
    }
    """
    cases = {}
    X_all, Y_all, meta_all = [], [], []

    for ar in range(ar_min, ar_max + 1):
        if ar not in case_files:
            print(f"Skipping AR={ar}: not in case_files")
            continue
        print(f"Processing AR={ar}")
        one = build_case_subdomains(
            mesh_h5=case_files[ar]["mesh"],
            dat_h5=case_files[ar]["dat"],
            aspect_ratio=ar,
            n_subdomains=n_subdomains,
            nx=nx,
            ny=ny,
        )
        cases[ar] = one
        X_all.append(one["X"])
        Y_all.append(one["Y"])
        meta_all.extend(one["metadata"])

    if not X_all:
        raise ValueError("No cases were processed. Check case_files and AR range.")

    first = cases[next(iter(cases))]
    return {
        "X": np.concatenate(X_all, axis=0),
        "Y": np.concatenate(Y_all, axis=0),
        "metadata": meta_all,
        "cases": cases,
        "input_channel_names": first["input_channel_names"],
        "output_channel_names": first["output_channel_names"],
    }

def standardize_channels_nchw(A, eps=1e-8):
    """Standardize array shaped (N, C, nx, ny) channel-wise."""
    mean = A.mean(axis=(0, 2, 3), keepdims=True)
    std = A.std(axis=(0, 2, 3), keepdims=True) + eps
    return (A - mean) / std, mean, std

def summarize_boundary_or_interface(profile, channel_names):
    """profile shape: (C, n_points)."""
    return {
        name: {
            "min": float(profile[i].min()),
            "max": float(profile[i].max()),
            "mean": float(profile[i].mean()),
            "std": float(profile[i].std()),
        }
        for i, name in enumerate(channel_names)
    }
