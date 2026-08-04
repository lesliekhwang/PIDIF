"""Plot persisted unknown-interface runner artifacts without running inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.interpolate import griddata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIELD_NAMES = ("pressure", "u", "v")
FIELD_UNITS = ("Pa", "m/s", "m/s")
GRID_NX = 1200
GRID_NY = 300
PLOT_DPI = 200
DISPLAY_LOW_PERCENTILE = 0.5
DISPLAY_HIGH_PERCENTILE = 99.5
ERROR_HIGH_PERCENTILE = 99.5


def _as_array(value: Any, name: str) -> np.ndarray:
    """Convert a persisted tensor-like value to a NumPy array."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise TypeError(f"Artifact field {name!r} is not array-like") from exc
    return array


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Artifact {name!r} must contain a mapping")
    return value


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return _require_mapping(json.load(handle), name)


def _load_torch_mapping(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older torch versions.
        payload = torch.load(path, map_location="cpu")
    return _require_mapping(payload, name)


def _validate_field_array(value: Any, name: str, n_points: int | None = None) -> np.ndarray:
    array = np.asarray(_as_array(value, name), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if n_points is not None and array.shape[0] != n_points:
        raise ValueError(
            f"{name} has {array.shape[0]} points but expected {n_points}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _load_run_inputs(run_directory: Path) -> dict[str, Any]:
    """Load the manifest, reconstruction, and final prediction artifacts."""

    run_directory = run_directory.expanduser().resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_directory}")

    manifest = _load_json_mapping(run_directory / "manifest.json", "manifest")
    reconstruction = _load_torch_mapping(
        run_directory / "reconstruction.pt", "reconstruction artifact"
    )
    final_prediction = _load_torch_mapping(
        run_directory / "final_prediction.pt", "final prediction artifact"
    )
    posthoc_metrics_path = run_directory / "posthoc_metrics.json"
    posthoc_metrics = (
        _load_json_mapping(posthoc_metrics_path, "posthoc metrics")
        if posthoc_metrics_path.is_file()
        else None
    )

    coordinates = np.asarray(
        _as_array(reconstruction.get("global_coordinates_mm"), "global_coordinates_mm"),
        dtype=np.float64,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "global_coordinates_mm must have shape (N, 2), "
            f"got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("global_coordinates_mm contains non-finite values")
    n_points = coordinates.shape[0]

    subdomain_id = np.asarray(
        _as_array(reconstruction.get("subdomain_id"), "subdomain_id")
    ).reshape(-1)
    if subdomain_id.shape[0] != n_points:
        raise ValueError("subdomain_id and global_coordinates_mm have different lengths")
    if not np.isfinite(subdomain_id.astype(np.float64, copy=False)).all():
        raise ValueError("subdomain_id contains non-finite values")
    if not np.equal(subdomain_id, np.rint(subdomain_id)).all():
        raise ValueError("subdomain_id must contain integer values")
    subdomain_id = subdomain_id.astype(np.int64, copy=False)
    if (subdomain_id < 0).any():
        raise ValueError("subdomain_id must be non-negative")

    inside_geometry_mask = np.asarray(
        _as_array(reconstruction.get("inside_geometry_mask"), "inside_geometry_mask")
    ).reshape(-1)
    if inside_geometry_mask.shape[0] != n_points:
        raise ValueError("inside_geometry_mask and global_coordinates_mm have different lengths")
    inside_geometry_mask = inside_geometry_mask.astype(bool, copy=False)

    prediction_value = final_prediction.get("physical")
    if prediction_value is None:
        prediction_value = reconstruction.get("physical_prediction")
    if prediction_value is None:
        raise KeyError("Neither final_prediction.physical nor reconstruction.physical_prediction exists")
    prediction = _validate_field_array(prediction_value, "physical prediction", n_points)

    reconstructed_prediction = reconstruction.get("physical_prediction")
    if reconstructed_prediction is not None:
        reconstructed_prediction_array = _validate_field_array(
            reconstructed_prediction, "reconstruction.physical_prediction", n_points
        )
        if not np.allclose(prediction, reconstructed_prediction_array, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("final prediction and reconstruction prediction differ")

    geometry = _validate_geometry(reconstruction.get("geometry"))
    field_names = tuple(reconstruction.get("field_names", FIELD_NAMES))
    field_units = tuple(reconstruction.get("field_units", FIELD_UNITS))
    if field_names != FIELD_NAMES:
        raise ValueError(f"Unsupported field_names {field_names!r}; expected {FIELD_NAMES!r}")
    if field_units != FIELD_UNITS:
        raise ValueError(f"Unsupported field_units {field_units!r}; expected {FIELD_UNITS!r}")

    plotting_metadata = reconstruction.get("plotting", {})
    if plotting_metadata is None:
        plotting_metadata = {}
    plotting_metadata = _require_mapping(plotting_metadata, "plotting metadata")
    grid_resolution = plotting_metadata.get("grid_resolution", (GRID_NX, GRID_NY))
    if not isinstance(grid_resolution, Sequence) or len(grid_resolution) != 2:
        raise ValueError("plotting.grid_resolution must contain [nx, ny]")
    nx, ny = int(grid_resolution[0]), int(grid_resolution[1])
    if nx < 2 or ny < 2:
        raise ValueError("plotting.grid_resolution values must be at least 2")

    dataset = manifest.get("dataset", {})
    dataset = _require_mapping(dataset, "manifest.dataset")
    protocol = manifest.get("protocol", {})
    protocol = _require_mapping(protocol, "manifest.protocol")
    case_id = str(dataset.get("case_id", manifest.get("case_id", run_directory.name)))
    method_name = str(
        final_prediction.get(
            "method_name",
            protocol.get("method_name", manifest.get("method_name", "unknown_interface")),
        )
    )

    return {
        "run_directory": run_directory,
        "manifest": manifest,
        "reconstruction": reconstruction,
        "final_prediction": final_prediction,
        "posthoc_metrics": posthoc_metrics,
        "coordinates": coordinates,
        "subdomain_id": subdomain_id,
        "inside_geometry_mask": inside_geometry_mask,
        "prediction": prediction,
        "geometry": geometry,
        "grid_resolution": (nx, ny),
        "case_id": case_id,
        "method_name": method_name,
    }


def _validate_geometry(value: Any) -> dict[str, np.ndarray]:
    geometry = _require_mapping(value, "reconstruction.geometry")
    required = ("x_points_mm", "wall_bottom_y_mm", "wall_top_y_mm")
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        if name not in geometry:
            raise KeyError(f"reconstruction.geometry is missing {name}")
        array = np.asarray(_as_array(geometry[name], f"geometry.{name}"), dtype=np.float64).reshape(-1)
        if array.size < 2:
            raise ValueError(f"geometry.{name} must contain at least two values")
        if not np.isfinite(array).all():
            raise ValueError(f"geometry.{name} contains non-finite values")
        arrays[name] = array
    if arrays["x_points_mm"].size != arrays["wall_bottom_y_mm"].size:
        raise ValueError("geometry x_points_mm and wall_bottom_y_mm have different lengths")
    if arrays["x_points_mm"].size != arrays["wall_top_y_mm"].size:
        raise ValueError("geometry x_points_mm and wall_top_y_mm have different lengths")
    if not np.all(np.diff(arrays["x_points_mm"]) > 0.0):
        raise ValueError("geometry.x_points_mm must be strictly increasing")
    if np.any(arrays["wall_bottom_y_mm"] >= arrays["wall_top_y_mm"]):
        raise ValueError("geometry wall bottom must be below wall top")
    interfaces = geometry.get("internal_interface_x_mm", arrays["x_points_mm"][1:-1])
    interfaces = np.asarray(_as_array(interfaces, "geometry.internal_interface_x_mm"), dtype=np.float64).reshape(-1)
    if interfaces.size != arrays["x_points_mm"].size - 2:
        raise ValueError("geometry.internal_interface_x_mm has an unexpected length")
    if interfaces.size and not np.allclose(interfaces, arrays["x_points_mm"][1:-1]):
        raise ValueError("geometry internal interfaces do not match x_points_mm")
    arrays["internal_interface_x_mm"] = interfaces
    return arrays


def _load_posthoc_truth(inputs: Mapping[str, Any]) -> np.ndarray:
    """Load CFD truth only after the caller explicitly enables post-hoc loading."""

    import h5py

    dataset = _require_mapping(inputs["manifest"].get("dataset", {}), "manifest.dataset")
    raw_path = dataset.get("resolved_path", dataset.get("path"))
    if raw_path is None:
        raise RuntimeError("Manifest does not specify a dataset path for post-hoc truth loading")
    dataset_path = Path(str(raw_path)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (REPOSITORY_ROOT / dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Post-hoc truth dataset does not exist: {dataset_path}")
    sample_indices = dataset.get("sample_indices")
    if not isinstance(sample_indices, Sequence) or isinstance(sample_indices, (str, bytes)):
        raise RuntimeError("Manifest does not specify sample_indices for post-hoc truth loading")

    targets: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as handle:
        for sample_index in sample_indices:
            target_path = f"samples/{int(sample_index)}/target"
            if target_path not in handle:
                raise KeyError(f"Post-hoc truth requires {target_path}")
            target = np.asarray(handle[target_path][...], dtype=np.float64).squeeze()
            if target.ndim == 1:
                if target.size % 3 != 0:
                    raise ValueError(f"Target {target_path} cannot be reshaped to (-1, 3)")
                target = target.reshape(-1, 3)
            elif target.ndim == 2 and target.shape[1] == 3:
                pass
            elif target.ndim == 2 and target.shape[0] == 3:
                target = target.T
            else:
                raise ValueError(f"Target {target_path} has unexpected shape {target.shape}")
            if not np.isfinite(target).all():
                raise ValueError(f"Target {target_path} contains non-finite values")
            targets.append(target)
    if not targets:
        raise RuntimeError("Post-hoc truth sample_indices is empty")
    truth = np.concatenate(targets, axis=0)
    if truth.shape != inputs["prediction"].shape:
        raise ValueError(
            f"Truth/prediction shape mismatch: truth={truth.shape}, "
            f"prediction={inputs['prediction'].shape}"
        )
    return truth


def _embedded_truth(inputs: Mapping[str, Any]) -> np.ndarray | None:
    artifacts = (("reconstruction", inputs["reconstruction"]), ("posthoc_metrics", inputs.get("posthoc_metrics")))
    for artifact_name, artifact in artifacts:
        if artifact is None:
            continue
        for name in ("physical_truth", "truth_physical", "truth"):
            if name in artifact:
                return _validate_field_array(
                    artifact[name],
                    f"{artifact_name}.{name}",
                    inputs["prediction"].shape[0],
                )
    return None


def _build_grid(inputs: Mapping[str, Any], values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Interpolate fields per subdomain and apply the piecewise wall mask."""

    coordinates = inputs["coordinates"]
    subdomain_id = inputs["subdomain_id"]
    geometry = inputs["geometry"]
    nx, ny = inputs["grid_resolution"]
    x_points = geometry["x_points_mm"]
    bottom = geometry["wall_bottom_y_mm"]
    top = geometry["wall_top_y_mm"]
    x_plot = np.linspace(float(x_points.min()), float(x_points.max()), nx)
    y_plot = np.linspace(float(bottom.min()), float(top.max()), ny)
    x_grid, y_grid = np.meshgrid(x_plot, y_plot)
    bottom_grid = np.interp(x_plot, x_points, bottom)
    top_grid = np.interp(x_plot, x_points, top)
    channel_mask = (
        (y_grid >= bottom_grid[None, :]) & (y_grid <= top_grid[None, :])
    )

    output = np.full((3, ny, nx), np.nan, dtype=np.float64)
    for subdomain_index in range(x_points.size - 1):
        point_mask = subdomain_id == subdomain_index
        column_indices = np.where(
            (x_plot >= x_points[subdomain_index])
            & (x_plot <= x_points[subdomain_index + 1])
        )[0]
        if not point_mask.any() or column_indices.size == 0:
            continue
        points = coordinates[point_mask]
        sub_values = values[point_mask]
        sub_x = x_grid[:, column_indices]
        sub_y = y_grid[:, column_indices]
        for field_index in range(3):
            try:
                interpolated = griddata(
                    points=points,
                    values=sub_values[:, field_index],
                    xi=(sub_x, sub_y),
                    method="linear",
                )
            except (ValueError, RuntimeError):
                interpolated = np.full(sub_x.shape, np.nan, dtype=np.float64)
            missing = ~np.isfinite(interpolated)
            if missing.any():
                nearest = griddata(
                    points=points,
                    values=sub_values[:, field_index],
                    xi=(sub_x, sub_y),
                    method="nearest",
                )
                interpolated[missing] = nearest[missing]
            output[field_index][:, column_indices] = interpolated
    output[:, ~channel_mask] = np.nan
    extent = [float(x_plot.min()), float(x_plot.max()), float(y_plot.min()), float(y_plot.max())]
    return output, x_grid, y_grid, extent


def _safe_limits(values: np.ndarray, symmetric: bool = False, nonnegative: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot compute display limits from an empty finite field")
    if nonnegative:
        lower = 0.0
        upper = float(np.percentile(finite, ERROR_HIGH_PERCENTILE))
    else:
        lower = float(np.percentile(finite, DISPLAY_LOW_PERCENTILE))
        upper = float(np.percentile(finite, DISPLAY_HIGH_PERCENTILE))
    if symmetric:
        radius = max(abs(lower), abs(upper))
        lower, upper = -radius, radius
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Display limits are non-finite")
    if upper <= lower:
        scale = max(abs(lower), abs(upper), 1.0)
        if nonnegative:
            lower, upper = 0.0, scale * 1.0e-6
        else:
            lower, upper = lower - 0.5 * scale, upper + 0.5 * scale
    return lower, upper


def _comparison_ranges(truth_grid: np.ndarray, prediction_grid: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    shared: list[tuple[float, float]] = []
    errors: list[tuple[float, float]] = []
    for field_index in range(3):
        combined = np.concatenate(
            [truth_grid[field_index][np.isfinite(truth_grid[field_index])],
             prediction_grid[field_index][np.isfinite(prediction_grid[field_index])]]
        )
        shared.append(_safe_limits(combined, symmetric=field_index == 2))
        errors.append(
            _safe_limits(
                np.abs(prediction_grid[field_index] - truth_grid[field_index]),
                nonnegative=True,
            )
        )
    return shared, errors


def _prediction_ranges(prediction_grid: np.ndarray) -> list[tuple[float, float]]:
    return [
        _safe_limits(prediction_grid[index], symmetric=index == 2)
        for index in range(3)
    ]


def _get_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _add_channel_geometry(axis: Any, geometry: Mapping[str, np.ndarray]) -> None:
    x_points = geometry["x_points_mm"]
    bottom = geometry["wall_bottom_y_mm"]
    top = geometry["wall_top_y_mm"]
    axis.plot(x_points, bottom, color="black", linewidth=1.25, zorder=5)
    axis.plot(x_points, top, color="black", linewidth=1.25, zorder=5)
    for interface_x in x_points[1:-1]:
        interface_bottom = float(np.interp(interface_x, x_points, bottom))
        interface_top = float(np.interp(interface_x, x_points, top))
        axis.plot(
            [interface_x, interface_x],
            [interface_bottom, interface_top],
            color="white",
            linestyle="--",
            linewidth=0.8,
            alpha=0.95,
            zorder=6,
        )
    axis.set_xlim(float(x_points.min()), float(x_points.max()))
    axis.set_ylim(float(bottom.min()), float(top.max()))
    axis.set_xlabel("x", fontsize=11)
    axis.set_ylabel("y", fontsize=11)
    axis.tick_params(labelsize=9)


def _save_figure(fig: Any, plt: Any, output_path: Path, display_ranges: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(fd)
        temp_path = Path(raw_path)
        fig.savefig(temp_path, format="png", dpi=PLOT_DPI, bbox_inches="tight")
        with temp_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
        data_axes = sum(bool(axis.images) for axis in fig.axes)
        print(
            f"Saved {output_path}; figure_size={tuple(fig.get_size_inches())}; "
            f"data_axes={data_axes}; cmap=jet; display_ranges={display_ranges}"
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        plt.close(fig)


def _maybe_show(plt: Any, show: bool) -> None:
    if show:
        plt.show()


def plot_full_comparison(inputs: Mapping[str, Any], truth: np.ndarray, output_path: Path, show: bool = False) -> None:
    """Save the notebook-style 3-by-3 truth, prediction, and error plot."""

    if truth.shape != inputs["prediction"].shape:
        raise ValueError("Truth and prediction must have the same shape")
    prediction_grid, _, _, extent = _build_grid(inputs, inputs["prediction"])
    truth_grid, _, _, _ = _build_grid(inputs, truth)
    shared_ranges, error_ranges = _comparison_ranges(truth_grid, prediction_grid)
    plt = _get_pyplot(show)
    figure, axes = plt.subplots(3, 3, figsize=(25, 10.5), constrained_layout=True)
    for field_index, (field_name, field_unit) in enumerate(zip(FIELD_NAMES, FIELD_UNITS)):
        shared_vmin, shared_vmax = shared_ranges[field_index]
        error_vmin, error_vmax = error_ranges[field_index]
        panel_data = (
            (truth_grid[field_index], f"CFD truth {field_name}", (shared_vmin, shared_vmax)),
            (prediction_grid[field_index], f"Prediction {field_name}", (shared_vmin, shared_vmax)),
            (np.abs(prediction_grid[field_index] - truth_grid[field_index]), f"{field_name} absolute error", (error_vmin, error_vmax)),
        )
        for column_index, (field_grid, title, limits) in enumerate(panel_data):
            image = axes[field_index, column_index].imshow(
                np.ma.masked_invalid(field_grid),
                extent=extent,
                origin="lower",
                cmap="jet",
                aspect="auto",
                interpolation="bilinear",
                vmin=limits[0],
                vmax=limits[1],
            )
            axes[field_index, column_index].set_title(title, fontsize=14)
            _add_channel_geometry(axes[field_index, column_index], inputs["geometry"])
            colorbar = figure.colorbar(
                image,
                ax=axes[field_index, column_index],
                pad=0.01,
                fraction=0.035,
            )
            colorbar.set_label(field_unit, fontsize=10)
            colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"{inputs['case_id']} — CFD truth vs {inputs['method_name']} prediction",
        fontsize=17,
    )
    _maybe_show(plt, show)
    _save_figure(figure, plt, output_path, {"shared": shared_ranges, "error": error_ranges})


def plot_prediction_only(inputs: Mapping[str, Any], output_path: Path, show: bool = False) -> None:
    """Save the stable one-row prediction-only layout used by DeepONet plotting."""

    prediction_grid, _, _, extent = _build_grid(inputs, inputs["prediction"])
    ranges = _prediction_ranges(prediction_grid)
    plt = _get_pyplot(show)
    figure, axes = plt.subplots(1, 3, figsize=(24, 3.5), constrained_layout=True)
    for field_index, (field_name, field_unit) in enumerate(zip(FIELD_NAMES, FIELD_UNITS)):
        image = axes[field_index].imshow(
            np.ma.masked_invalid(prediction_grid[field_index]),
            extent=extent,
            origin="lower",
            cmap="jet",
            aspect="auto",
            interpolation="bilinear",
            vmin=ranges[field_index][0],
            vmax=ranges[field_index][1],
        )
        axes[field_index].set_title(f"Prediction {field_name}", fontsize=16)
        _add_channel_geometry(axes[field_index], inputs["geometry"])
        colorbar = figure.colorbar(image, ax=axes[field_index], pad=0.02)
        colorbar.set_label(field_unit)
    figure.suptitle(
        f"{inputs['case_id']} — {inputs['method_name']} prediction",
        fontsize=17,
    )
    _maybe_show(plt, show)
    _save_figure(figure, plt, output_path, ranges)


def _read_history(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing optimization history: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Optimization history is empty")

    aliases = {
        "gradient_norm": ("gradient_norm", "gradient_l2"),
        "max_relative_mass_flux_residual": (
            "max_relative_mass_flux_residual",
            "maximum_relative_flow_error",
        ),
        "mean_relative_mass_flux_residual": (
            "mean_relative_mass_flux_residual",
            "mean_relative_flow_error",
        ),
    }
    required = (
        "step",
        "total_loss",
        "fixed_point_loss",
        "neighbor_loss",
        "mass_flux_loss",
        "wall_loss",
        "smoothness_loss",
        "pressure_transverse_loss",
        "pressure_monotonic_loss",
        "prior_loss",
        "gradient_norm",
        "max_relative_mass_flux_residual",
        "mean_relative_mass_flux_residual",
    )
    result: dict[str, np.ndarray] = {}
    for name in required:
        source_name = name
        if name in aliases:
            source_name = next((candidate for candidate in aliases[name] if candidate in rows[0]), "")
        if not source_name or source_name not in rows[0]:
            raise KeyError(f"Optimization history is missing column {name}")
        try:
            values = np.asarray([float(row[source_name]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Optimization history column {source_name} is not numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Optimization history column {source_name} contains non-finite values")
        result[name] = values
    return result


def plot_optimization_history(inputs: Mapping[str, Any], output_path: Path, show: bool = False) -> None:
    """Save a compact multi-panel optimization-history plot."""

    history = _read_history(inputs["run_directory"] / "optimization_history.csv")
    plt = _get_pyplot(show)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = (
        (axes[0, 0], ("total_loss", "fixed_point_loss", "neighbor_loss"), "Loss components", "Loss"),
        (axes[0, 1], ("mass_flux_loss", "wall_loss", "smoothness_loss", "prior_loss"), "Physics losses", "Loss"),
        (axes[0, 2], ("pressure_transverse_loss", "pressure_monotonic_loss"), "Pressure losses", "Loss"),
        (axes[1, 0], ("gradient_norm",), "Gradient norm", "Gradient norm"),
        (axes[1, 1], ("max_relative_mass_flux_residual", "mean_relative_mass_flux_residual"), "Relative mass-flux residual", "Relative residual"),
    )
    labels = {
        "total_loss": "total",
        "fixed_point_loss": "fixed point",
        "neighbor_loss": "neighbor",
        "mass_flux_loss": "mass flux",
        "wall_loss": "wall",
        "smoothness_loss": "smoothness",
        "prior_loss": "prior",
        "pressure_transverse_loss": "pressure transverse",
        "pressure_monotonic_loss": "pressure monotonic",
        "gradient_norm": "gradient norm",
        "max_relative_mass_flux_residual": "max residual",
        "mean_relative_mass_flux_residual": "mean residual",
    }
    for axis, names, title, ylabel in panels:
        for name in names:
            axis.plot(history["step"], history[name], label=labels[name])
        axis.set_title(title)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[1, 2].axis("off")
    figure.suptitle(f"{inputs['case_id']} — {inputs['method_name']} optimization history", fontsize=17)
    _maybe_show(plt, show)
    _save_figure(figure, plt, output_path, "history panels")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot persisted unknown-interface results without running inference."
    )
    parser.add_argument("--run-directory", type=Path, required=True, help="Runner artifact directory.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory; defaults to <run-directory>/figures.",
    )
    parser.add_argument(
        "--plot",
        choices=("full-comparison", "prediction-only", "optimization-history", "all"),
        default="all",
        help="Plot selection.",
    )
    parser.add_argument(
        "--allow-posthoc-truth-load",
        action="store_true",
        help="Explicitly allow loading CFD targets from the manifest dataset for comparison.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures after saving.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = _load_run_inputs(args.run_directory)
        output_directory = (
            args.output_directory.expanduser().resolve()
            if args.output_directory is not None
            else inputs["run_directory"] / "figures"
        )
        if args.plot in ("full-comparison", "all"):
            truth = _embedded_truth(inputs)
            if truth is None and args.allow_posthoc_truth_load:
                truth = _load_posthoc_truth(inputs)
            if truth is None:
                message = (
                    "Full comparison requires an explicit truth array in reconstruction.pt "
                    "or --allow-posthoc-truth-load."
                )
                if args.plot == "all":
                    print(f"Skipping full comparison: {message}")
                else:
                    raise RuntimeError(message)
            else:
                plot_full_comparison(
                    inputs,
                    truth,
                    output_directory / "full_channel_truth_prediction_error.png",
                    show=args.show,
                )
        if args.plot in ("prediction-only", "all"):
            plot_prediction_only(
                inputs,
                output_directory / "full_channel_prediction.png",
                show=args.show,
            )
        if args.plot in ("optimization-history", "all"):
            plot_optimization_history(
                inputs,
                output_directory / "optimization_history.png",
                show=args.show,
            )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
