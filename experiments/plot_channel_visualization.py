#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VALIDATION_H5 = Path(
    "/home/nuoxu9/PIDIF/channel_diffusion_dataset/deeponet_style_dataset/"
    "channel_deeponet_style_pressure_u_v_random5_val.h5"
)
DEFAULT_GEOMETRY_ROOT = Path(
    "/home/nuoxu9/PIDIF/2d_geometry_specs/channel_water"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/nuoxu9/PIDIF/results/channel_visualization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one validation channel geometry and its stored randomized "
            "x-strip decomposition without regenerating the partition."
        )
    )
    parser.add_argument(
        "--validation-h5",
        type=Path,
        default=DEFAULT_VALIDATION_H5,
        help="Canonical randomized validation HDF5.",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=DEFAULT_GEOMETRY_ROOT,
        help="Root directory containing channel geometry JSON files.",
    )
    parser.add_argument(
        "--case-id",
        default="channel_08",
        help="Validation case id, e.g. channel_08.",
    )
    parser.add_argument(
        "--realization-id",
        type=int,
        default=0,
        help="Stored randomized decomposition realization to visualize.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf"],
        default=["png", "pdf"],
        help="Output figure formats.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return parser.parse_args()


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return value


def _as_1d(dataset: h5py.Dataset) -> np.ndarray:
    arr = dataset[...]
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1 and int(np.prod(arr.shape[1:])) == 1:
        arr = arr.reshape(arr.shape[0])
    return arr


def _find_dataset_paths(h5: h5py.File) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            basename = name.rsplit("/", 1)[-1]
            paths.setdefault(basename, []).append(name)

    h5.visititems(visitor)
    return paths


def _resolve_metadata_dataset(
    h5: h5py.File,
    paths_by_basename: dict[str, list[str]],
    candidates: Iterable[str],
    expected_length: int | None = None,
) -> tuple[str, np.ndarray]:
    attempted = []
    for key in candidates:
        for path in paths_by_basename.get(key, []):
            attempted.append(path)
            arr = _as_1d(h5[path])
            if expected_length is None or len(arr) == expected_length:
                return path, arr

    raise KeyError(
        "Could not resolve metadata dataset for candidates "
        f"{list(candidates)}. Matching paths considered: {attempted}"
    )


def load_decomposition_metadata(
    h5_path: Path,
    case_id: str,
    realization_id: int,
) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        role = _decode_scalar(h5.attrs.get("dataset_role", ""))
        split_role = _decode_scalar(h5.attrs.get("split_role", ""))

        if role and role != "canonical_randomized_validation":
            raise ValueError(
                f"Expected dataset_role='canonical_randomized_validation', found {role!r}."
            )
        if split_role and split_role != "validation":
            raise ValueError(
                f"Expected split_role='validation', found {split_role!r}."
            )

        paths = _find_dataset_paths(h5)

        case_path, case_values = _resolve_metadata_dataset(
            h5,
            paths,
            ["case_id", "case_ids"],
        )
        n_samples = len(case_values)

        realization_path, realization_values = _resolve_metadata_dataset(
            h5,
            paths,
            ["realization_id", "realization_ids"],
            expected_length=n_samples,
        )
        subdomain_path, subdomain_values = _resolve_metadata_dataset(
            h5,
            paths,
            ["subdomain_id", "subdomain_ids"],
            expected_length=n_samples,
        )
        x_left_path, x_left_values = _resolve_metadata_dataset(
            h5,
            paths,
            ["x_left_mm", "x_left"],
            expected_length=n_samples,
        )
        x_right_path, x_right_values = _resolve_metadata_dataset(
            h5,
            paths,
            ["x_right_mm", "x_right"],
            expected_length=n_samples,
        )

        decoded_cases = np.array(
            [str(_decode_scalar(v)) for v in case_values],
            dtype=object,
        )
        realization_values = np.asarray(realization_values, dtype=np.int64)
        subdomain_values = np.asarray(subdomain_values, dtype=np.int64)
        x_left_values = np.asarray(x_left_values, dtype=np.float64)
        x_right_values = np.asarray(x_right_values, dtype=np.float64)

        mask = (
            (decoded_cases == str(case_id))
            & (realization_values == int(realization_id))
        )
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            available_cases = sorted(set(decoded_cases.tolist()))
            raise ValueError(
                f"No samples found for case_id={case_id!r}, "
                f"realization_id={realization_id}. "
                f"Available case ids include: {available_cases[:10]}"
            )

        order = np.argsort(subdomain_values[indices])
        indices = indices[order]

        subdomain_ids = subdomain_values[indices]
        expected = np.arange(len(indices), dtype=np.int64)
        if not np.array_equal(subdomain_ids, expected):
            raise ValueError(
                "Selected subdomain IDs are not consecutive 0..S-1: "
                f"{subdomain_ids.tolist()}"
            )

        x_left = x_left_values[indices]
        x_right = x_right_values[indices]
        x_edges = np.unique(np.concatenate([x_left, x_right]))
        x_edges.sort()

        if len(x_edges) != len(indices) + 1:
            raise ValueError(
                "Expected one more unique x-edge than subdomains. "
                f"Found {len(indices)} subdomains and {len(x_edges)} unique edges."
            )

        interface_x = x_edges[1:-1]

        return {
            "dataset_role": role,
            "split_role": split_role,
            "sample_indices": indices,
            "subdomain_ids": subdomain_ids,
            "x_left": x_left,
            "x_right": x_right,
            "x_edges": x_edges,
            "interface_x": interface_x,
            "metadata_paths": {
                "case_id": case_path,
                "realization_id": realization_path,
                "subdomain_id": subdomain_path,
                "x_left": x_left_path,
                "x_right": x_right_path,
            },
        }


def _numeric_1d(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all():
        return None
    return arr


def _numeric_points(value: Any) -> np.ndarray | None:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        if all("x" in item and "y" in item for item in value):
            try:
                arr = np.array(
                    [[float(item["x"]), float(item["y"])] for item in value],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                return None
        else:
            return None
    else:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None

    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_dicts(value)


def _extract_geometry_from_json(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_names = [
        "x_mm",
        "wall_x_mm",
        "x",
        "wall_x",
        "x_coords_mm",
        "x_coords",
    ]
    bottom_names = [
        "wall_y_bottom_mm",
        "y_bottom_mm",
        "bottom_y_mm",
        "wall_y_bottom",
        "y_bottom",
        "bottom_y",
    ]
    top_names = [
        "wall_y_top_mm",
        "y_top_mm",
        "top_y_mm",
        "wall_y_top",
        "y_top",
        "top_y",
    ]

    # Common representation: shared x array plus lower/upper y arrays.
    for d in _walk_dicts(data):
        for x_key in x_names:
            if x_key not in d:
                continue
            x = _numeric_1d(d[x_key])
            if x is None:
                continue
            for b_key in bottom_names:
                if b_key not in d:
                    continue
                yb = _numeric_1d(d[b_key])
                if yb is None or len(yb) != len(x):
                    continue
                for t_key in top_names:
                    if t_key not in d:
                        continue
                    yt = _numeric_1d(d[t_key])
                    if yt is not None and len(yt) == len(x):
                        return x, yb, yt

    # Common representation: lower/bottom wall and upper/top wall as Nx2 points.
    lower_names = [
        "wall_bottom",
        "bottom_wall",
        "lower_wall",
        "wall_lower",
        "bottom_points",
        "lower_points",
    ]
    upper_names = [
        "wall_top",
        "top_wall",
        "upper_wall",
        "wall_upper",
        "top_points",
        "upper_points",
    ]
    for d in _walk_dicts(data):
        lower = None
        upper = None
        for key in lower_names:
            if key in d:
                lower = _numeric_points(d[key])
                if lower is not None:
                    break
        for key in upper_names:
            if key in d:
                upper = _numeric_points(d[key])
                if upper is not None:
                    break

        if lower is None or upper is None:
            continue

        lower = lower[np.argsort(lower[:, 0])]
        upper = upper[np.argsort(upper[:, 0])]

        if len(lower) == len(upper) and np.allclose(
            lower[:, 0], upper[:, 0], rtol=0.0, atol=1e-10
        ):
            return lower[:, 0], lower[:, 1], upper[:, 1]

        x = np.unique(np.concatenate([lower[:, 0], upper[:, 0]]))
        yb = np.interp(x, lower[:, 0], lower[:, 1])
        yt = np.interp(x, upper[:, 0], upper[:, 1])
        return x, yb, yt

    raise ValueError(
        "Could not identify wall coordinates in the geometry JSON. "
        "Expected either shared x + bottom/top y arrays, or bottom/top Nx2 point arrays."
    )


def find_geometry_json(root: Path, case_id: str) -> Path:
    path = root / f"{case_id}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find geometry JSON: {path}"
        )

    return path


def load_geometry(root: Path, case_id: str) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    path = find_geometry_json(root, case_id)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    units = str(data.get("units", "")).strip().lower()
    if units not in {"mm", "millimeter", "millimeters"}:
        raise ValueError(
            f"Expected geometry JSON units in millimeters, found {data.get('units')!r}."
        )

    boundaries = data.get("boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("Geometry JSON is missing a 'boundaries' mapping.")

    bottom = _numeric_points(boundaries.get("wall_bottom"))
    top = _numeric_points(boundaries.get("wall_top"))
    if bottom is None or top is None:
        raise ValueError(
            "Expected boundaries['wall_bottom'] and boundaries['wall_top'] "
            "to be lists of {'x', 'y'} points."
        )

    bottom = bottom[np.argsort(bottom[:, 0])]
    top = top[np.argsort(top[:, 0])]

    if len(bottom) == len(top) and np.allclose(
        bottom[:, 0], top[:, 0], rtol=0.0, atol=1e-12
    ):
        x = bottom[:, 0]
        y_bottom = bottom[:, 1]
        y_top = top[:, 1]
    else:
        x = np.unique(np.concatenate([bottom[:, 0], top[:, 0]]))
        y_bottom = np.interp(x, bottom[:, 0], bottom[:, 1])
        y_top = np.interp(x, top[:, 0], top[:, 1])

    x = np.asarray(x, dtype=np.float64)
    y_bottom = np.asarray(y_bottom, dtype=np.float64)
    y_top = np.asarray(y_top, dtype=np.float64)

    if np.any(y_bottom >= y_top):
        raise ValueError("Geometry JSON produced bottom-wall values >= top-wall values.")

    metadata = data.get("metadata", {})
    channel_length_mm = metadata.get("channel_length_mm")
    if channel_length_mm is not None:
        observed_length = float(x.max() - x.min())
        if not np.isclose(observed_length, float(channel_length_mm), rtol=0.0, atol=1e-9):
            raise ValueError(
                "Geometry length does not match metadata['channel_length_mm']: "
                f"{observed_length} vs {channel_length_mm}."
            )

    return path, x, y_bottom, y_top


def _style_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x [mm]", fontsize=13)
    ax.set_ylabel("y [mm]", fontsize=13)
    ax.tick_params(labelsize=11)
    # Match the senior DeepONet field plots' stretched full-channel presentation.
    ax.set_aspect("auto")


def _draw_channel(ax, x, y_bottom, y_top):
    ax.fill_between(x, y_bottom, y_top, alpha=0.12)
    ax.plot(x, y_bottom, linewidth=1.2)
    ax.plot(x, y_top, linewidth=1.2)


def plot_comparison(
    case_id: str,
    realization_id: int,
    x: np.ndarray,
    y_bottom: np.ndarray,
    y_top: np.ndarray,
    interface_x: np.ndarray,
):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y_bottom)), float(np.max(y_top))
    ypad = 0.08 * max(ymax - ymin, 1e-12)
    xlim = (xmin, xmax)
    ylim = (ymin - ypad, ymax + ypad)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 3.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    _draw_channel(axes[0], x, y_bottom, y_top)
    axes[0].set_title(f"{case_id} geometry", fontsize=15)
    _style_axis(axes[0], xlim, ylim)

    _draw_channel(axes[1], x, y_bottom, y_top)
    for xpos in interface_x:
        axes[1].axvline(
            float(xpos),
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
    axes[1].set_title(
        f"Randomized decomposition (realization {realization_id})",
        fontsize=15,
    )
    _style_axis(axes[1], xlim, ylim)

    return fig


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {path}")


def main() -> None:
    args = parse_args()

    h5_path = args.validation_h5.resolve()
    geometry_root = args.geometry_root.resolve()
    output_dir = (
        args.output_root.resolve()
        / args.case_id
        / f"random_realization_{args.realization_id:02d}"
    )

    metadata = load_decomposition_metadata(
        h5_path=h5_path,
        case_id=args.case_id,
        realization_id=args.realization_id,
    )
    geometry_path, x, y_bottom, y_top = load_geometry(
        root=geometry_root,
        case_id=args.case_id,
    )

    print("Channel visualization")
    print(f" case id          : {args.case_id}")
    print(f" realization id   : {args.realization_id}")
    print(f" validation HDF5  : {h5_path}")
    print(f" geometry JSON    : {geometry_path}")
    print(f" selected samples : {len(metadata['sample_indices'])}")
    print(f" subdomain ids    : {metadata['subdomain_ids'].tolist()}")
    print(f" x edges [mm]     : {metadata['x_edges'].tolist()}")
    print(f" metadata paths   : {metadata['metadata_paths']}")
    print(f" output dir       : {output_dir}")

    fig = plot_comparison(
        case_id=args.case_id,
        realization_id=args.realization_id,
        x=x,
        y_bottom=y_bottom,
        y_top=y_top,
        interface_x=np.asarray(metadata["interface_x"], dtype=np.float64),
    )
    save_figure(
        fig,
        output_dir,
        stem=f"{args.case_id}_geometry_and_random_decomposition",
        formats=args.formats,
        dpi=args.dpi,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
