import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

from deeponet_fluent_dataset import DeepONetCellDataset
from fluent_deeponet import predict_cell_sample

PathLike = Union[str, Path]

def local_query_to_physical(query_local, metadata):
    """
    Convert local subdomain query coordinates to physical coordinates.

    query_local:
        (N, 2), columns [x_local, y_local]

    metadata:
        one entry from data["metadata"]
    """
    query_local = np.asarray(query_local, dtype=np.float32)

    # x_local is normalized to [0, 1] across the subdomain width.
    # y_local is normalized within each subdomain's local y span; with
    # horizontal_interface=True the bottom/top halves use y_local_origin_mm
    # and y_local_scale_mm (half reference length) rather than the full channel.
    x_left = float(metadata["x_left_mm"])
    x_right = float(metadata["x_right_mm"])

    x_phys = x_left + query_local[:, 0] * (x_right - x_left)

    if "y_local_origin_mm" in metadata and "y_local_scale_mm" in metadata:
        y_origin = float(metadata["y_local_origin_mm"])
        y_scale = float(metadata["y_local_scale_mm"])
        y_phys = y_origin + query_local[:, 1] * y_scale
    else:
        ref_length = float(metadata["reference_length_mm"])
        y_phys = query_local[:, 1] * ref_length

    return x_phys, y_phys


def _zone_keys(h5: h5py.File) -> List[str]:
    return sorted(h5["meshes/1/faces/nodes"].keys(), key=lambda s: int(s))


def _read_zone_face_nodes(h5: h5py.File, zone_key: str) -> List[np.ndarray]:
    base = f"meshes/1/faces/nodes/{zone_key}"
    nnodes = h5[f"{base}/nnodes"][:]
    nodes_flat = h5[f"{base}/nodes"][:] - 1  # Fluent node ids are 1-based

    faces: List[np.ndarray] = []
    offset = 0
    for n in nnodes:
        faces.append(nodes_flat[offset : offset + int(n)])
        offset += int(n)
    return faces


def _order_cell_nodes(edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Order a 2D cell's nodes into a closed polygon loop from its boundary edges."""
    if not edges:
        return None

    adjacency: dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None

    start = next(iter(adjacency))
    ordered = [start]
    prev, current = None, start
    while True:
        n0, n1 = adjacency[current]
        nxt = n0 if n0 != prev else n1
        if nxt == start:
            break
        ordered.append(nxt)
        prev, current = current, nxt
        if len(ordered) > len(adjacency):
            return None

    return ordered if len(ordered) == len(adjacency) else None


def reconstruct_triangles_and_boundary_edges(mesh_h5: PathLike):
    """Return coords, triangles, tri_to_cell, boundary edges from a Fluent HDF5 mesh."""
    with h5py.File(mesh_h5, "r") as h5:
        coords = h5["meshes/1/nodes/coords/1"][:]
        n_cells = int(h5["meshes/1/cells/zoneTopology/maxId"][:].max())
        cell_edges: List[List[Tuple[int, int]]] = [[] for _ in range(n_cells)]
        boundary_edges: List[Tuple[int, int]] = []

        for zone_key in _zone_keys(h5):
            faces = _read_zone_face_nodes(h5, zone_key)
            c0 = h5[f"meshes/1/faces/c0/{zone_key}"][:].astype(int)
            c1 = h5[f"meshes/1/faces/c1/{zone_key}"][:].astype(int)

            is_boundary_zone = np.all(c0 == 0) or np.all(c1 == 0)

            for face_nodes, left_cell, right_cell in zip(faces, c0, c1):
                if len(face_nodes) != 2:
                    raise ValueError("This script expects a 2D face to have exactly 2 nodes.")

                edge = (int(face_nodes[0]), int(face_nodes[1]))

                if is_boundary_zone:
                    boundary_edges.append(edge)

                for cell_id in (left_cell, right_cell):
                    if cell_id > 0:
                        cell_edges[cell_id - 1].append(edge)

        triangles: List[List[int]] = []
        tri_to_cell: List[int] = []
        bad: List[Tuple[int, int]] = []

        for cell_idx, edges in enumerate(cell_edges):
            ordered = _order_cell_nodes(edges)

            if ordered is None or len(ordered) not in (3, 4):
                bad.append((cell_idx + 1, 0 if ordered is None else len(ordered)))
                continue

            if len(ordered) == 3:
                triangles.append(ordered)
                tri_to_cell.append(cell_idx)
            else:
                a, b, c, d = ordered
                triangles.append([a, b, c])
                triangles.append([a, c, d])
                tri_to_cell.extend([cell_idx, cell_idx])

        if bad:
            sample = bad[:10]
            raise ValueError(
                "Expected triangular or convex quadrilateral cells, but found cells "
                f"whose faces do not form a 3- or 4-node loop. "
                f"First bad cells (id, n_nodes): {sample}"
            )

        triangles_arr = np.array(triangles, dtype=int)
        tri_to_cell_arr = np.array(tri_to_cell, dtype=int)

    return coords, triangles_arr, tri_to_cell_arr, boundary_edges


def wall_polyline_segments(
    wall_x: Sequence[float],
    wall_y_bottom: Sequence[float],
    wall_y_top: Sequence[float],
    coord_scale: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build bottom/top wall line segments for a piecewise-linear channel."""
    x_pts = np.asarray(wall_x, dtype=np.float64) * coord_scale
    yb = np.asarray(wall_y_bottom, dtype=np.float64) * coord_scale
    yt = np.asarray(wall_y_top, dtype=np.float64) * coord_scale

    if xlim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
        keep = (x_pts >= xmin) & (x_pts <= xmax)
        if np.count_nonzero(keep) < 2:
            keep = np.ones_like(x_pts, dtype=bool)
        x_pts = x_pts[keep]
        yb = yb[keep]
        yt = yt[keep]

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for xs, ys in ((x_pts, yb), (x_pts, yt)):
        for i in range(len(xs) - 1):
            segments.append((np.array([xs[i], xs[i + 1]]), np.array([ys[i], ys[i + 1]])))
    return segments


def _mesh_inside_mask(
    X: np.ndarray,
    Y: np.ndarray,
    mesh_h5: PathLike,
    coord_scale: float,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    coords, triangles, _, boundary_edges = reconstruct_triangles_and_boundary_edges(mesh_h5)
    coords = coords * coord_scale
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles=triangles)
    inside = tri.get_trifinder()(X, Y) >= 0
    segments = [(coords[i], coords[j]) for i, j in boundary_edges]
    return inside, segments


def _polyline_inside_mask(
    X: np.ndarray,
    Y: np.ndarray,
    wall_x: Sequence[float],
    wall_y_bottom: Sequence[float],
    wall_y_top: Sequence[float],
    coord_scale: float,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    x_pts = np.asarray(wall_x, dtype=np.float64) * coord_scale
    yb = np.asarray(wall_y_bottom, dtype=np.float64) * coord_scale
    yt = np.asarray(wall_y_top, dtype=np.float64) * coord_scale

    if xlim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
        keep = (x_pts >= xmin) & (x_pts <= xmax)
        if np.count_nonzero(keep) < 2:
            keep = np.ones_like(x_pts, dtype=bool)
        x_pts = x_pts[keep]
        yb = yb[keep]
        yt = yt[keep]

    poly_x = np.concatenate([x_pts, x_pts[::-1]])
    poly_y = np.concatenate([yb, yt[::-1]])
    inside = MplPath(np.column_stack([poly_x, poly_y])).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])
    ).reshape(X.shape)
    segments = wall_polyline_segments(
        wall_x, wall_y_bottom, wall_y_top, coord_scale=coord_scale, xlim=xlim
    )
    return inside, segments


def interpolate_to_plot_grid(
    x,
    y,
    z,
    n_x_plot=None,
    n_y_plot=100,
    method="linear",
    mesh_h5: Optional[PathLike] = None,
    wall_x: Optional[Sequence[float]] = None,
    wall_y_bottom: Optional[Sequence[float]] = None,
    wall_y_top: Optional[Sequence[float]] = None,
    coord_scale: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Interpolate scattered cell-center values to a regular plot grid.

    When ``mesh_h5`` or wall polyline coordinates are supplied, pixels outside the
    channel are masked to NaN so imshow respects the true (polyline) boundary.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    if xlim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
    else:
        xmin, xmax = float(np.min(x)), float(np.max(x))
    if ylim is not None:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    else:
        ymin, ymax = float(np.min(y)), float(np.max(y))

    if n_x_plot is None:
        ar_plot = round((xmax - xmin) / (ymax - ymin), 1)
        n_x_plot = max(100, int(round(n_y_plot * ar_plot)))

    xi = np.linspace(xmin, xmax, int(n_x_plot))
    yi = np.linspace(ymin, ymax, int(n_y_plot))
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), z, (Xi, Yi), method=method)

    if np.any(np.isnan(Zi)):
        Zi_nearest = griddata((x, y), z, (Xi, Yi), method="nearest")
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

    boundary_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    if mesh_h5 is not None:
        inside, boundary_segments = _mesh_inside_mask(Xi, Yi, mesh_h5, coord_scale)
        Zi = np.where(inside, Zi, np.nan)
    elif wall_x is not None and wall_y_bottom is not None and wall_y_top is not None:
        inside, boundary_segments = _polyline_inside_mask(
            Xi,
            Yi,
            wall_x,
            wall_y_bottom,
            wall_y_top,
            coord_scale=coord_scale,
            xlim=xlim,
        )
        Zi = np.where(inside, Zi, np.nan)

    extent = (xmin, xmax, ymin, ymax)
    return Xi, Yi, Zi, extent, boundary_segments


def interface_x_from_metadata(metadata_list: Sequence[Mapping[str, object]]) -> List[float]:
    """Return sorted unique internal vertical interface x-coordinates.

    Works for both plain x-strip subdomains and horizontal y-split halves.  The
    internal x-edges are the unique ``x_left_mm`` / ``x_right_mm`` values that
    lie strictly inside the channel span.
    """
    if not metadata_list:
        return []

    x_edges = sorted(
        {float(m["x_left_mm"]) for m in metadata_list}
        | {float(m["x_right_mm"]) for m in metadata_list}
    )
    if len(x_edges) < 2:
        return []

    xmin, xmax = x_edges[0], x_edges[-1]
    return [x for x in x_edges if xmin < x < xmax]


def interface_y_from_metadata(metadata_list: Sequence[Mapping[str, object]]) -> List[float]:
    """Return sorted unique horizontal interface y-coordinates, if present.

    When ``horizontal_interface=True`` during dataset construction, each sample
    metadata row carries the same ``y_center_mm`` value.
    """
    y_centers = sorted(
        {
            float(m["y_center_mm"])
            for m in metadata_list
            if m.get("y_center_mm") is not None
        }
    )
    return y_centers


def is_metis_partitioning(
    metadata: Optional[Sequence[Mapping[str, object]]] = None,
    interface_placement: Optional[str] = None,
) -> bool:
    """Return True when samples were built with ``interface_placement='metis'``."""
    if interface_placement is not None:
        return str(interface_placement).lower() == "metis"
    if metadata:
        return any("metis_partition_id" in m for m in metadata)
    return False


def metis_cut_face_midpoints_from_samples(
    samples: Sequence[Mapping[str, object]],
    metadata_list: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Return physical midpoints of METIS partition-cut faces from sample branches.

    Each METIS sample stores every cut-face interface sensor at the start of its
    branch array.  Those local coordinates are mapped back to physical space with
    the sample metadata bounding box.
    """
    points: List[np.ndarray] = []
    for sample, metadata in zip(samples, metadata_list):
        n_cut = int(metadata.get("metis_n_interface_faces", 0))
        if n_cut <= 0:
            continue
        branch = np.asarray(sample["branch"], dtype=np.float32)
        query_local = branch[:n_cut, :2]
        x_phys, y_phys = local_query_to_physical(query_local, metadata)
        points.append(np.column_stack([x_phys, y_phys]))

    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(points)


def _resolve_metis_cut_face_midpoints(
    data: Mapping[str, object],
    sample_indices: Sequence[int],
    metadata_list: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Prefer raw-case diagnostics; otherwise reconstruct from branch sensors."""
    raw_cases = data.get("raw_cases")
    if raw_cases:
        meta0 = metadata_list[0]
        case_id = meta0["case_id"]
        realization_id = int(meta0.get("realization_id", 0))
        n_realizations = int(data.get("n_realizations", 1))
        raw_key: object = case_id if n_realizations == 1 else (case_id, realization_id)
        raw = raw_cases.get(raw_key)
        if isinstance(raw, Mapping) and "metis_cut_face_midpoints" in raw:
            cut = np.asarray(raw["metis_cut_face_midpoints"], dtype=np.float64).reshape(-1, 2)
            if cut.size:
                return cut

    samples = [data["samples"][int(sid)] for sid in sample_indices]
    return metis_cut_face_midpoints_from_samples(samples, metadata_list)


def plot_prediction_imshow_from_points(
    x,
    y,
    pred,
    truth,
    field_name,
    field_idx,
    figsize=(24, 3.5),
    n_x_plot=None,
    n_y_plot=100,
    output_dir=None,
    filename_prefix="prediction",
    show=True,
    mesh_h5: Optional[PathLike] = None,
    wall_x: Optional[Sequence[float]] = None,
    wall_y_bottom: Optional[Sequence[float]] = None,
    wall_y_top: Optional[Sequence[float]] = None,
    draw_boundary: bool = True,
    draw_interfaces: bool = True,
    interface_x: Optional[Sequence[float]] = None,
    interface_y: Optional[Sequence[float]] = None,
    metis_cut_face_midpoints: Optional[np.ndarray] = None,
    interface_placement: Optional[str] = None,
    metadata: Optional[Sequence[Mapping[str, object]]] = None,
    coord_scale: float = 1.0,
    coord_unit: str = "",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Plot predicted field, CFD/truth field, and absolute error using griddata + imshow.

    When ``mesh_h5`` or wall polyline coordinates are provided, pixels outside the
    channel are masked and the true (polyline) boundary is drawn with
    ``LineCollection``, following the style of ``plot_imshow`` in ``foo.ipynb``.

    For x-strip partitioning, subdomain interfaces are drawn as dashed lines:
    vertical interfaces from unique interior ``x_left_mm`` / ``x_right_mm`` edges,
    and horizontal interfaces from ``y_center_mm`` when present.

    For METIS partitioning, pass ``interface_placement='metis'`` (or metadata
    containing ``metis_partition_id``) and supply ``metis_cut_face_midpoints``
    (or use the array returned by ``collect_predictions_for_data``).  METIS
    cut faces are drawn as white scatter markers instead of axis-aligned lines.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    pred = np.asarray(pred)
    truth = np.asarray(truth)

    _, _, z_truth, extent, boundary_segments = interpolate_to_plot_grid(
        x,
        y,
        truth[:, field_idx],
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        method="linear",
        mesh_h5=mesh_h5,
        wall_x=wall_x,
        wall_y_bottom=wall_y_bottom,
        wall_y_top=wall_y_top,
        coord_scale=coord_scale,
        xlim=xlim,
        ylim=ylim,
    )
    _, _, z_pred, _, _ = interpolate_to_plot_grid(
        x,
        y,
        pred[:, field_idx],
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        method="linear",
        mesh_h5=mesh_h5,
        wall_x=wall_x,
        wall_y_bottom=wall_y_bottom,
        wall_y_top=wall_y_top,
        coord_scale=coord_scale,
        xlim=xlim,
        ylim=ylim,
    )

    z_error = np.abs(z_truth - z_pred)

    cmap_mask = np.isfinite(z_truth)
    cmap_vals = z_truth[cmap_mask]
    if cmap_vals.size == 0:
        vmin = float(np.nanmin(truth[:, field_idx]))
        vmax = float(np.nanmax(truth[:, field_idx]))
    else:
        vmin = float(cmap_vals.min())
        vmax = float(cmap_vals.max())

    is_metis = is_metis_partitioning(metadata, interface_placement)
    if is_metis:
        interface_x_list: List[float] = []
        interface_y_list: List[float] = []
    else:
        if metadata is not None:
            if interface_x is None:
                interface_x = interface_x_from_metadata(metadata)
            if interface_y is None:
                interface_y = interface_y_from_metadata(metadata)
        interface_x_list = [float(xi) for xi in (interface_x or [])]
        interface_y_list = [float(yi) for yi in (interface_y or [])]

    metis_cut = (
        np.asarray(metis_cut_face_midpoints, dtype=np.float64).reshape(-1, 2)
        if metis_cut_face_midpoints is not None
        else np.empty((0, 2), dtype=np.float64)
    )

    x_label = f"x [{coord_unit}]" if coord_unit else "x"
    y_label = f"y [{coord_unit}]" if coord_unit else "y"

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for ax, z_plot, title, use_shared_scale in zip(
        axes,
        [z_truth, z_pred, z_error],
        [f"CFD truth {field_name}", f"Prediction {field_name}", f"{field_name} absolute error"],
        [True, True, False],
    ):
        im = ax.imshow(
            np.ma.masked_invalid(z_plot),
            extent=extent,
            origin="lower",
            cmap="jet",
            aspect="auto",
            interpolation="bilinear",
            vmin=vmin if use_shared_scale else None,
            vmax=vmax if use_shared_scale else None,
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        if draw_boundary and boundary_segments:
            ax.add_collection(LineCollection(boundary_segments, linewidths=0.8, colors="k"))
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if draw_interfaces:
            if is_metis:
                if metis_cut.size:
                    ax.scatter(
                        metis_cut[:, 0],
                        metis_cut[:, 1],
                        s=4.0,
                        c="w",
                        marker=".",
                        linewidths=0.0,
                        alpha=0.95,
                        zorder=5,
                    )
            else:
                for x_iface in interface_x_list:
                    ax.axvline(
                        x_iface,
                        color="w",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=5,
                    )
                for y_iface in interface_y_list:
                    ax.axhline(
                        y_iface,
                        color="0.85",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=5,
                    )
        fig.colorbar(im, ax=ax, pad=0.02)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{filename_prefix}_{field_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes

@torch.no_grad()
def collect_predictions_for_data(
    model,
    data,
    device,
    y_normalizer,
    local_aspect_mean,
    local_aspect_std,
    sample_indices=None,
):
    """
    Predict selected samples and concatenate their physical coordinates,
    predictions, and truth values.

    For METIS-partitioned datasets, also returns ``metis_cut_face_midpoints``
    (from ``raw_cases`` when available, otherwise reconstructed from branch
    interface sensors) and leaves ``interface_x`` / ``interface_y`` empty.
    """
    if sample_indices is None:
        sample_indices = range(len(data["samples"]))
    sample_indices = list(sample_indices)

    xs = []
    ys = []
    preds = []
    truths = []
    sample_ids = []
    metadata_list = []

    for sid in sample_indices:
        sample = data["samples"][sid]
        metadata = data["metadata"][sid]
        metadata_list.append(metadata)

        pred_phys = predict_cell_sample(
            model=model,
            sample=sample,
            device=device,
            y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            branch_channel_names=data["branch_channel_names"],
        )
        query_local = np.asarray(sample["query"], dtype=np.float32)
        truth_phys = np.asarray(sample["target"], dtype=np.float32)

        x_phys, y_phys = local_query_to_physical(query_local, metadata)

        xs.append(x_phys)
        ys.append(y_phys)
        preds.append(pred_phys)
        truths.append(truth_phys)
        sample_ids.append(np.full(len(x_phys), sid, dtype=np.int64))

    interface_placement = data.get("interface_placement")
    is_metis = is_metis_partitioning(metadata_list, interface_placement)

    result = {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "pred": np.concatenate(preds, axis=0),
        "truth": np.concatenate(truths, axis=0),
        "sample_id": np.concatenate(sample_ids, axis=0),
        "metadata": metadata_list,
        "interface_placement": interface_placement,
        "is_metis": is_metis,
    }
    if is_metis:
        result["metis_cut_face_midpoints"] = _resolve_metis_cut_face_midpoints(
            data=data,
            sample_indices=sample_indices,
            metadata_list=metadata_list,
        )
        result["interface_x"] = []
        result["interface_y"] = []
    else:
        result["interface_x"] = interface_x_from_metadata(metadata_list)
        result["interface_y"] = interface_y_from_metadata(metadata_list)
    return result


def select_samples_by_ar(data, ar, realization_id=0):
    selected = [
        i for i, m in enumerate(data["metadata"])
        if int(round(float(m["aspect_ratio"]))) == int(ar)
        and int(m.get("realization_id", 0)) == int(realization_id)
    ]

    selected = sorted(
        selected,
        key=lambda i: int(data["metadata"][i]["subdomain_id"]),
    )

    if not selected:
        raise ValueError(
            f"No samples found for AR={ar}, realization_id={realization_id}"
        )

    sub_ids = [
        int(data["metadata"][i]["subdomain_id"])
        for i in selected
    ]

    expected = list(range(len(selected)))
    if sub_ids != expected:
        raise ValueError(
            f"Selected subdomains are not consecutive 0..S-1. "
            f"Found subdomain IDs: {sub_ids}"
        )

    return selected