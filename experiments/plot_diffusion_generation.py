#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "evaluate_diffusion_generation"
    / "channel08_real0_ddim20_ddim5_val"
)
DEFAULT_VALIDATION_H5 = (
    REPO_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)
DEFAULT_GEOMETRY_DIR = REPO_ROOT / "2d_geometry_specs" / "channel_water"

FIELD_NAMES = ("pressure", "u", "v")
FIELD_UNITS = {
    "pressure": "Pa",
    "u": "m/s",
    "v": "m/s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot full-channel field-diffusion generation using the same "
            "Truth | Prediction | Absolute error convention as DeepONet/plot.py."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Evaluation run directory containing predictions.h5.",
    )
    parser.add_argument(
        "--validation-h5",
        type=Path,
        default=DEFAULT_VALIDATION_H5,
        help="Canonical randomized validation HDF5 containing query/truth/metadata.",
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=DEFAULT_GEOMETRY_DIR,
        help="Directory containing channel geometry JSON files.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="channel_08",
        help="Validation case id.",
    )
    parser.add_argument(
        "--realization-id",
        type=int,
        default=0,
        help="Validation decomposition realization id.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[20, 5],
        help="DDIM sampling step counts to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--n-y-plot",
        type=int,
        default=200,
        help="Regular plotting-grid resolution in y.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--prediction-label",
        type=str,
        default=None,
        help=(
            "Optional label used in the prediction-panel title. "
            "If omitted, the script uses 'DDIM-<steps>'."
        ),
    )
    parser.add_argument(
        "--error-vmax-pressure",
        type=float,
        default=None,
        help=(
            "Fixed absolute-error colorbar vmax for pressure [Pa]. "
            "If omitted, use this run's own max error."
        ),
    )
    parser.add_argument(
        "--error-vmax-u",
        type=float,
        default=None,
        help=(
            "Fixed absolute-error colorbar vmax for u [m/s]. "
            "If omitted, use this run's own max error."
        ),
    )
    parser.add_argument(
        "--error-vmax-v",
        type=float,
        default=None,
        help=(
            "Fixed absolute-error colorbar vmax for v [m/s]. "
            "If omitted, use this run's own max error."
        ),
    )
    return parser.parse_args()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def local_query_to_physical(
    query_local: np.ndarray,
    metadata: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    query_local = np.asarray(query_local, dtype=np.float64)

    x_left = float(metadata["x_left_mm"])
    x_right = float(metadata["x_right_mm"])
    x_phys = x_left + query_local[:, 0] * (x_right - x_left)

    if (
        "y_local_origin_mm" in metadata
        and "y_local_scale_mm" in metadata
    ):
        y_origin = float(metadata["y_local_origin_mm"])
        y_scale = float(metadata["y_local_scale_mm"])
    else:
        y_origin = 0.0
        y_scale = float(metadata["reference_length_mm"])

    y_phys = y_origin + query_local[:, 1] * y_scale
    return x_phys, y_phys


def read_sample_metadata(
    h5: h5py.File,
    sample_index: int,
) -> dict[str, object]:
    metadata_group = h5["metadata"]
    return {
        key: decode_scalar(metadata_group[key][sample_index])
        for key in metadata_group.keys()
    }


def select_case_samples(
    validation_h5: Path,
    case_id: str,
    realization_id: int,
) -> list[int]:
    with h5py.File(validation_h5, "r") as h5:
        case_ids = np.array(
            [
                str(decode_scalar(value))
                for value in h5["metadata/case_id"][:]
            ],
            dtype=object,
        )
        realizations = np.asarray(
            h5["metadata/realization_id"][:],
            dtype=np.int64,
        )
        subdomains = np.asarray(
            h5["metadata/subdomain_id"][:],
            dtype=np.int64,
        )

    mask = (
        (case_ids == str(case_id))
        & (realizations == int(realization_id))
    )
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(
            f"No samples found for case_id={case_id!r}, "
            f"realization_id={realization_id}."
        )

    indices = indices[np.argsort(subdomains[indices])]
    selected_subdomains = subdomains[indices]
    expected = np.arange(len(indices), dtype=np.int64)
    if not np.array_equal(selected_subdomains, expected):
        raise ValueError(
            "Selected subdomain ids are not consecutive 0..S-1: "
            f"{selected_subdomains.tolist()}"
        )

    return indices.tolist()


def collect_full_channel(
    validation_h5: Path,
    predictions_h5: Path,
    sample_indices: Sequence[int],
    sampling_steps: int,
) -> dict[str, object]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    metadata_list: list[dict[str, object]] = []

    with h5py.File(validation_h5, "r") as source, h5py.File(
        predictions_h5, "r"
    ) as pred_h5:
        step_samples = pred_h5.get(f"ddim_{sampling_steps}/samples")
        if step_samples is None:
            raise KeyError(
                f"Missing group ddim_{sampling_steps}/samples in {predictions_h5}."
            )

        for sample_index in sample_indices:
            sample_key = str(sample_index)
            if sample_key not in step_samples:
                raise KeyError(
                    f"DDIM-{sampling_steps} prediction missing sample {sample_index}."
                )

            sample_group = source["samples"][sample_key]
            query = sample_group["query"][:].astype(np.float32)
            truth = sample_group["target"][:].astype(np.float32)
            pred = step_samples[sample_key]["prediction"][:].astype(np.float32)
            metadata = read_sample_metadata(source, sample_index)

            if truth.shape != pred.shape:
                raise ValueError(
                    f"Sample {sample_index}: truth shape {truth.shape} "
                    f"!= prediction shape {pred.shape}."
                )
            if query.shape != (truth.shape[0], 2):
                raise ValueError(
                    f"Sample {sample_index}: query shape {query.shape} is invalid."
                )

            pred_attrs = step_samples[sample_key].attrs
            for key in ("case_id", "realization_id", "subdomain_id"):
                if key in pred_attrs and key in metadata:
                    left = decode_scalar(pred_attrs[key])
                    right = metadata[key]
                    if str(left) != str(right):
                        raise ValueError(
                            f"Sample {sample_index}: prediction/source metadata "
                            f"mismatch for {key}: {left!r} vs {right!r}."
                        )

            x_phys, y_phys = local_query_to_physical(query, metadata)

            xs.append(x_phys)
            ys.append(y_phys)
            truths.append(truth)
            predictions.append(pred)
            metadata_list.append(metadata)

    interface_x = sorted(
        {
            float(m["x_left_mm"])
            for m in metadata_list
        }
        | {
            float(m["x_right_mm"])
            for m in metadata_list
        }
    )
    interface_x = interface_x[1:-1]

    return {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "truth": np.concatenate(truths, axis=0),
        "prediction": np.concatenate(predictions, axis=0),
        "metadata": metadata_list,
        "interface_x": interface_x,
    }


def load_geometry(
    geometry_json: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with geometry_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    units = str(data.get("units", "")).strip().lower()
    if units not in {"mm", "millimeter", "millimeters"}:
        raise ValueError(
            f"Expected geometry units in mm, found {data.get('units')!r}."
        )

    boundaries = data["boundaries"]
    bottom = np.asarray(
        [[float(p["x"]), float(p["y"])] for p in boundaries["wall_bottom"]],
        dtype=np.float64,
    )
    top = np.asarray(
        [[float(p["x"]), float(p["y"])] for p in boundaries["wall_top"]],
        dtype=np.float64,
    )

    bottom = bottom[np.argsort(bottom[:, 0])]
    top = top[np.argsort(top[:, 0])]

    if len(bottom) != len(top) or not np.allclose(
        bottom[:, 0],
        top[:, 0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Expected wall_bottom and wall_top to share the same x coordinates."
        )

    return bottom[:, 0], bottom[:, 1], top[:, 1]


def wall_polyline_segments(
    wall_x: np.ndarray,
    wall_y_bottom: np.ndarray,
    wall_y_top: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for ys in (wall_y_bottom, wall_y_top):
        for i in range(len(wall_x) - 1):
            segments.append(
                (
                    np.array(
                        [wall_x[i], wall_x[i + 1]],
                        dtype=np.float64,
                    ),
                    np.array(
                        [ys[i], ys[i + 1]],
                        dtype=np.float64,
                    ),
                )
            )
    return segments


def polygon_inside_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    wall_x: np.ndarray,
    wall_y_bottom: np.ndarray,
    wall_y_top: np.ndarray,
) -> np.ndarray:
    poly_x = np.concatenate([wall_x, wall_x[::-1]])
    poly_y = np.concatenate([wall_y_bottom, wall_y_top[::-1]])
    polygon = MplPath(np.column_stack([poly_x, poly_y]))
    return polygon.contains_points(
        np.column_stack([x_grid.ravel(), y_grid.ravel()])
    ).reshape(x_grid.shape)


def interpolate_to_plot_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    wall_x: np.ndarray,
    wall_y_bottom: np.ndarray,
    wall_y_top: np.ndarray,
    n_y_plot: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    xmin = float(np.min(wall_x))
    xmax = float(np.max(wall_x))
    ymin = float(np.min(wall_y_bottom))
    ymax = float(np.max(wall_y_top))

    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    n_x_plot = max(100, int(round(n_y_plot * aspect_ratio)))

    xi = np.linspace(xmin, xmax, n_x_plot)
    yi = np.linspace(ymin, ymax, n_y_plot)
    x_grid, y_grid = np.meshgrid(xi, yi)

    z_grid = griddata(
        (x, y),
        z,
        (x_grid, y_grid),
        method="linear",
    )

    if np.any(np.isnan(z_grid)):
        nearest = griddata(
            (x, y),
            z,
            (x_grid, y_grid),
            method="nearest",
        )
        z_grid = np.where(np.isnan(z_grid), nearest, z_grid)

    inside = polygon_inside_mask(
        x_grid,
        y_grid,
        wall_x,
        wall_y_bottom,
        wall_y_top,
    )
    z_grid = np.where(inside, z_grid, np.nan)

    return z_grid, (xmin, xmax, ymin, ymax)


def plot_one_field(
    x: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    field_name: str,
    field_index: int,
    sampling_steps: int,
    wall_x: np.ndarray,
    wall_y_bottom: np.ndarray,
    wall_y_top: np.ndarray,
    interface_x: Sequence[float],
    n_y_plot: int,
    output_dir: Path,
    case_id: str,
    realization_id: int,
    dpi: int,
    prediction_label: str | None,
    error_vmax: float | None,
) -> None:
    z_truth, extent = interpolate_to_plot_grid(
        x,
        y,
        truth[:, field_index],
        wall_x,
        wall_y_bottom,
        wall_y_top,
        n_y_plot,
    )
    z_prediction, _ = interpolate_to_plot_grid(
        x,
        y,
        prediction[:, field_index],
        wall_x,
        wall_y_bottom,
        wall_y_top,
        n_y_plot,
    )

    # Match the senior DeepONet plotting convention:
    # error is computed after truth/prediction are interpolated to the same grid.
    z_error = np.abs(z_truth - z_prediction)

    finite_truth = z_truth[np.isfinite(z_truth)]
    if finite_truth.size == 0:
        raise ValueError(f"No finite plotted truth values for field {field_name}.")

    # Match DeepONet/plot.py: truth defines the shared truth/prediction scale.
    vmin = float(np.min(finite_truth))
    vmax = float(np.max(finite_truth))

    finite_error = z_error[np.isfinite(z_error)]
    if finite_error.size == 0:
        raise ValueError(f"No finite plotted error values for field {field_name}.")

    error_vmin = 0.0
    if error_vmax is None:
        error_vmax = float(np.max(finite_error))

    prediction_title_prefix = (
        prediction_label if prediction_label is not None
        else f"DDIM-{sampling_steps}"
    )

    boundary_segments = wall_polyline_segments(
        wall_x,
        wall_y_bottom,
        wall_y_top,
    )
    line_segments = [
        np.column_stack([xs, ys])
        for xs, ys in boundary_segments
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 3.5),
        constrained_layout=True,
    )

    panel_specs = [
        (
            z_truth,
            f"CFD truth {field_name}",
            vmin,
            vmax,
        ),
        (
            z_prediction,
            f"{prediction_title_prefix} prediction {field_name}",
            vmin,
            vmax,
        ),
        (
            z_error,
            f"{field_name} absolute error",
            error_vmin,
            error_vmax,
        ),
    ]

    for ax, (z_plot, title, panel_vmin, panel_vmax) in zip(axes, panel_specs):
        image = ax.imshow(
            np.ma.masked_invalid(z_plot),
            extent=extent,
            origin="lower",
            cmap="jet",
            aspect="auto",
            interpolation="bilinear",
            vmin=panel_vmin,
            vmax=panel_vmax,
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x [mm]", fontsize=14)
        ax.set_ylabel("y [mm]", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

        ax.add_collection(
            LineCollection(
                line_segments,
                linewidths=0.8,
                colors="k",
            )
        )

        for x_interface in interface_x:
            ax.axvline(
                float(x_interface),
                color="w",
                linestyle="--",
                linewidth=1.0,
                alpha=0.95,
                zorder=5,
            )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(
            FIELD_UNITS[field_name],
            fontsize=12,
        )
        colorbar.ax.tick_params(labelsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{case_id}_real{realization_id:02d}_"
        f"ddim{sampling_steps}_{field_name}"
    )

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    predictions_h5 = run_dir / "predictions.h5"
    validation_h5 = args.validation_h5.resolve()
    geometry_json = (
        args.geometry_dir.resolve()
        / f"{args.case_id}.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "plots"
    )

    if not predictions_h5.is_file():
        raise FileNotFoundError(
            f"Predictions HDF5 not found: {predictions_h5}"
        )
    if not validation_h5.is_file():
        raise FileNotFoundError(
            f"Validation HDF5 not found: {validation_h5}"
        )
    if not geometry_json.is_file():
        raise FileNotFoundError(
            f"Geometry JSON not found: {geometry_json}"
        )

    sample_indices = select_case_samples(
        validation_h5,
        args.case_id,
        args.realization_id,
    )

    wall_x, wall_y_bottom, wall_y_top = load_geometry(
        geometry_json
    )

    error_vmax_by_field = {
        "pressure": args.error_vmax_pressure,
        "u": args.error_vmax_u,
        "v": args.error_vmax_v,
    }

    print("Diffusion generation field visualization")
    print(f" run dir          : {run_dir}")
    print(f" predictions HDF5 : {predictions_h5}")
    print(f" validation HDF5  : {validation_h5}")
    print(f" geometry JSON    : {geometry_json}")
    print(f" case id          : {args.case_id}")
    print(f" realization id   : {args.realization_id}")
    print(f" sample indices   : {sample_indices}")
    print(f" DDIM steps       : {args.steps}")
    print(f" output dir       : {output_dir}")
    print(f" error vmax pressure : {args.error_vmax_pressure}")
    print(f" error vmax u        : {args.error_vmax_u}")
    print(f" error vmax v        : {args.error_vmax_v}")
    print(f" prediction label    : {args.prediction_label}")

    for sampling_steps in args.steps:
        full = collect_full_channel(
            validation_h5=validation_h5,
            predictions_h5=predictions_h5,
            sample_indices=sample_indices,
            sampling_steps=sampling_steps,
        )

        print(
            f" DDIM-{sampling_steps:<3d} "
            f"| points={len(full['x'])} "
            f"| interfaces={full['interface_x']}"
        )

        for field_index, field_name in enumerate(FIELD_NAMES):
            plot_one_field(
                x=np.asarray(full["x"]),
                y=np.asarray(full["y"]),
                truth=np.asarray(full["truth"]),
                prediction=np.asarray(full["prediction"]),
                field_name=field_name,
                field_index=field_index,
                sampling_steps=sampling_steps,
                wall_x=wall_x,
                wall_y_bottom=wall_y_bottom,
                wall_y_top=wall_y_top,
                interface_x=full["interface_x"],
                n_y_plot=args.n_y_plot,
                output_dir=output_dir,
                case_id=args.case_id,
                realization_id=args.realization_id,
                dpi=args.dpi,
                prediction_label=args.prediction_label,
                error_vmax=error_vmax_by_field[field_name],
            )


if __name__ == "__main__":
    main()
