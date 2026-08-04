"""Run fixed consistency/physics optimization for one unknown-interface case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pidiffusion.artifacts import (  # noqa: E402
    build_run_id,
    create_run_directory,
    update_manifest,
    write_manifest,
)
from pidiffusion.data import (  # noqa: E402
    FeatureNormalizer,
    normalize_diffusion_branch,
)
from pidiffusion.diffusion import (  # noqa: E402
    DiffusionSchedule,
    build_linear_schedule,
    ddim_step,
    final_clean_projection,
)
from pidiffusion.model import PointSetDiffusionDenoiser  # noqa: E402
from pidiffusion.provenance import (  # noqa: E402
    file_identity,
    git_state,
    runtime_environment,
)


DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_controlpoints.h5"
)
DEFAULT_GEOMETRY_ROOT = REPOSITORY_ROOT / "2d_geometry_specs" / "channel_water"
DEFAULT_SOURCE_NOTEBOOK = REPOSITORY_ROOT / "unknown_interface_physics_optimization.ipynb"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results"

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
TRUNK_CHANNEL_NAMES = ("x_local", "y_local")
OUTPUT_CHANNEL_NAMES = ("pressure", "u", "v")

METHOD_NAME = "consistency_physics_optimization"
MODEL_ROLE = "frozen_distilled_student"
STAGE3_SCHEMA = "progressive_distillation_stage_v1"
EXPECTED_PARAMETER_COUNT = 382083
LEFT_EDGE_START = 0
RIGHT_EDGE_START = 256
BOTTOM_EDGE_START = 512
TOP_EDGE_START = 768


@dataclass(frozen=True)
class UnknownInterfaceConfig:
    """Typed configuration for one fixed unknown-interface run."""

    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    checkpoint_path: Optional[Path] = None
    geometry_path: Optional[Path] = None
    source_notebook: Path = field(default_factory=lambda: DEFAULT_SOURCE_NOTEBOOK)
    results_root: Path = field(default_factory=lambda: DEFAULT_RESULTS_ROOT)
    case_id: str = "channel_36"
    device: str = "cuda:1"
    global_seed: int = 20260731
    inference_noise_seed: int = 8100
    boundary_noise_seed: int = 8200
    student_timesteps: tuple[int, ...] = (999, 749, 500, 250, 0)
    diffusion_steps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    num_subdomains: int = 10
    num_internal_interfaces: int = 9
    interface_points: int = 256
    geometry_tolerance_mm: float = 2.0e-5
    interface_eta_tolerance: float = 1.0e-6
    inlet_u: float = 1.0
    inlet_v: float = 0.0
    outlet_gauge_pressure: float = 0.0
    dynamic_viscosity: float = 1.003e-3
    density: float = 998.2
    consistency_steps: int = 200
    learning_rate: float = 2.0e-2
    gradient_clip: float = 1.0
    save_interval: int = 10
    fixed_point_weight: float = 1.0
    neighbor_weight: float = 0.25
    mass_flux_weight: float = 2.0
    wall_weight: float = 0.50
    smoothness_weight: float = 0.02
    pressure_transverse_weight: float = 0.10
    pressure_monotonic_weight: float = 0.25
    prior_weight: float = 0.01
    field_weights: tuple[float, float, float] = (1.0, 1.0, 0.25)
    normalized_lower_bounds: tuple[float, float, float] = (-3.0, -3.0, -8.0)
    normalized_upper_bounds: tuple[float, float, float] = (5.0, 4.0, 8.0)
    enable_posthoc_evaluation: bool = False
    run_id: Optional[str] = None


@dataclass(frozen=True)
class GeometryData:
    """Validated geometry arrays in millimetres and metres."""

    case_id: str
    x_points_mm: np.ndarray
    bottom_y_mm: np.ndarray
    top_y_mm: np.ndarray
    x_points_m: np.ndarray
    bottom_y_m: np.ndarray
    top_y_m: np.ndarray
    interface_height_m: np.ndarray
    interface_centerline_m: np.ndarray
    height_slope: np.ndarray


@dataclass(frozen=True)
class CaseData:
    """Selected case arrays and validated interface geometry."""

    case_id: str
    sample_indices: tuple[int, ...]
    all_case_ids: tuple[str, ...]
    branch_channel_names: tuple[str, ...]
    trunk_channel_names: tuple[str, ...]
    output_channel_names: tuple[str, ...]
    n_interface_points: int
    n_boundary_points: int
    include_interface_endpoints: bool
    horizontal_interface: bool
    horizontal_interface_jitter: float
    branch_raw: np.ndarray
    query_list: tuple[np.ndarray, ...]
    metadata: tuple[dict[str, Any], ...]
    query_concat_np: np.ndarray
    query_batch_id_np: np.ndarray
    subdomain_id_np: np.ndarray
    global_coordinates_mm: np.ndarray
    inside_geometry_mask_np: np.ndarray
    geometry: GeometryData
    interface_eta_np: np.ndarray
    fluid_mask_np: np.ndarray
    velocity_zero_mask_np: np.ndarray
    flow_weights_np: np.ndarray
    split_membership: Optional[str]


@dataclass
class RuntimeContext:
    """Runtime tensors shared by initialization, optimization, and prediction."""

    config: UnknownInterfaceConfig
    case: CaseData
    checkpoint: Mapping[str, Any]
    device: torch.device
    model: PointSetDiffusionDenoiser
    schedule: DiffusionSchedule
    target_mean: torch.Tensor
    target_std: torch.Tensor
    local_aspect_mean: float
    local_aspect_std: float
    truth_free_branch: torch.Tensor
    branch_mask: torch.Tensor
    query: torch.Tensor
    query_batch_id: torch.Tensor
    fixed_initial_noise: torch.Tensor
    boundary_query: torch.Tensor
    boundary_query_batch_id: torch.Tensor
    fixed_boundary_initial_noise: torch.Tensor
    flow_weights: torch.Tensor
    fluid_mask: torch.Tensor
    velocity_zero_mask: torch.Tensor
    physics_prior: torch.Tensor
    q_reference: torch.Tensor


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve(strict=False)


def _resolve_geometry_path(config: UnknownInterfaceConfig) -> Path:
    if config.geometry_path is not None:
        return _resolve_repo_path(config.geometry_path)
    return DEFAULT_GEOMETRY_ROOT / f"{config.case_id}.json"


def _manifest_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _require_file(path: Path, role: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{role} is not a regular file: {path}")


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def _decode_text_attribute(value: Any, name: str) -> str:
    value = _decode_scalar(value)
    if not isinstance(value, str):
        raise ValueError(f"HDF5 attribute {name!r} must be text")
    return value


def _as_list(value: Any, name: str) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Checkpoint field {name} must be list-like")
    return list(value)


def _compare_exact(name: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise ValueError(f"Checkpoint field {name!r} does not match the current protocol")


def _compare_numeric(name: str, expected: Any, actual: Any) -> None:
    try:
        expected_tensor = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
        actual_tensor = torch.as_tensor(actual, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"Checkpoint field {name!r} must be numeric") from exc
    if expected_tensor.shape != actual_tensor.shape or not torch.allclose(
        expected_tensor,
        actual_tensor,
        rtol=1.0e-6,
        atol=1.0e-8,
        equal_nan=False,
    ):
        raise ValueError(f"Checkpoint field {name!r} does not match the current protocol")


def _validate_device_syntax(raw_device: str) -> None:
    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device: {raw_device!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Only CPU and explicitly indexed CUDA devices are supported, got {raw_device!r}"
        )
    if device.type == "cuda" and device.index is None:
        raise ValueError(
            "CUDA device must include an explicit index, for example cuda:0"
        )


def _resolve_runtime_device(raw_device: str) -> torch.device:
    _validate_device_syntax(raw_device)
    device = torch.device(raw_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {raw_device!r}, but CUDA is not available; use --device cpu"
            )
        count = int(torch.cuda.device_count())
        if device.index is None or device.index < 0 or device.index >= count:
            raise RuntimeError(
                f"Requested CUDA device index {device.index}, but only {count} device(s) are available"
            )
    return device


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_geometry(path: Path, config: UnknownInterfaceConfig) -> GeometryData:
    _require_file(path, "Geometry JSON")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("case") != config.case_id:
        raise ValueError(
            f"Geometry case {payload.get('case')!r} does not match requested case {config.case_id!r}"
        )
    boundaries = payload.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("Geometry JSON is missing boundaries")

    geometry_metadata = payload.get("metadata", {})
    if not isinstance(geometry_metadata, Mapping):
        geometry_metadata = {}

    def _wall_points(raw: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if raw and all(isinstance(item, Mapping) for item in raw):
                try:
                    x_values = np.asarray([item["x"] for item in raw], dtype=np.float64)
                    y_values = np.asarray([item["y"] for item in raw], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Geometry boundary {name!r} has invalid point mappings") from exc
                return x_values, y_values
        try:
            values = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Geometry boundary {name!r} is not numeric") from exc
        if values.ndim == 1:
            return np.arange(values.size, dtype=np.float64), values
        if values.ndim == 2 and values.shape[1] == 2:
            return values[:, 0], values[:, 1]
        raise ValueError(f"Geometry boundary {name!r} has unsupported shape {values.shape}")

    try:
        raw_bottom = boundaries["wall_bottom"]
        raw_top = boundaries["wall_top"]
        bottom_x_mm, bottom_y_mm = _wall_points(raw_bottom, "wall_bottom")
        top_x_mm, top_y_mm = _wall_points(raw_top, "wall_top")
        raw_x_points = payload.get("x_points_mm", geometry_metadata.get("x_points_mm"))
        if raw_x_points is None:
            x_points_mm = bottom_x_mm
        else:
            x_points_mm = np.asarray(raw_x_points, dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Geometry JSON has incomplete wall or x-point arrays") from exc
    if bottom_x_mm.shape != top_x_mm.shape or not np.allclose(
        bottom_x_mm, top_x_mm, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("Geometry wall x coordinates must match")
    if x_points_mm.shape != bottom_x_mm.shape or not np.allclose(
        x_points_mm, bottom_x_mm, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("Geometry x points must match wall coordinates")
    if x_points_mm.shape != (config.num_subdomains + 1,):
        raise ValueError(f"Geometry must contain {config.num_subdomains + 1} x points")
    if bottom_y_mm.shape != x_points_mm.shape or top_y_mm.shape != x_points_mm.shape:
        raise ValueError("Geometry wall arrays must match the x-point count")
    if not np.isfinite(x_points_mm).all() or not np.isfinite(bottom_y_mm).all() or not np.isfinite(top_y_mm).all():
        raise ValueError("Geometry arrays must be finite")
    if np.any(np.diff(x_points_mm) <= 0.0):
        raise ValueError("Geometry x points must be strictly increasing")
    if np.any(top_y_mm <= bottom_y_mm):
        raise ValueError("Geometry channel heights must be positive")
    if "Uin_mps" in geometry_metadata:
        if not math.isclose(float(geometry_metadata["Uin_mps"]), config.inlet_u, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("Geometry inlet velocity does not match explicit configuration")
    x_points_m = x_points_mm * 1.0e-3
    bottom_y_m = bottom_y_mm * 1.0e-3
    top_y_m = top_y_mm * 1.0e-3
    interface_height_m = top_y_m - bottom_y_m
    interface_centerline_m = 0.5 * (top_y_m + bottom_y_m)
    height_slope = np.zeros_like(interface_height_m)
    height_slope[0] = (interface_height_m[1] - interface_height_m[0]) / (x_points_m[1] - x_points_m[0])
    height_slope[-1] = (interface_height_m[-1] - interface_height_m[-2]) / (x_points_m[-1] - x_points_m[-2])
    height_slope[1:-1] = (interface_height_m[2:] - interface_height_m[:-2]) / (
        x_points_m[2:] - x_points_m[:-2]
    )
    return GeometryData(
        case_id=config.case_id,
        x_points_mm=x_points_mm,
        bottom_y_mm=bottom_y_mm,
        top_y_mm=top_y_mm,
        x_points_m=x_points_m,
        bottom_y_m=bottom_y_m,
        top_y_m=top_y_m,
        interface_height_m=interface_height_m,
        interface_centerline_m=interface_centerline_m,
        height_slope=height_slope,
    )


def _metadata_array(handle: h5py.File, name: str) -> np.ndarray:
    path = f"metadata/{name}"
    if path not in handle:
        raise KeyError(f"Missing HDF5 metadata dataset: {path}")
    return np.asarray(handle[path][...])


def _decode_case_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(_decode_scalar(value)) for value in values], dtype=object)


def _validate_hdf5_geometry(
    metadata: Sequence[Mapping[str, Any]],
    geometry: GeometryData,
    config: UnknownInterfaceConfig,
) -> None:
    expected_left = geometry.x_points_mm[:-1]
    expected_right = geometry.x_points_mm[1:]
    actual_left = np.asarray([float(row["x_left_mm"]) for row in metadata])
    actual_right = np.asarray([float(row["x_right_mm"]) for row in metadata])
    for name, actual, expected in (
        ("x_left_mm", actual_left, expected_left),
        ("x_right_mm", actual_right, expected_right),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"HDF5 {name} does not match geometry JSON")
    for name, actual, expected in (
        ("y_bottom_left_mm", [row["y_bottom_left_mm"] for row in metadata], geometry.bottom_y_mm[:-1]),
        ("y_bottom_right_mm", [row["y_bottom_right_mm"] for row in metadata], geometry.bottom_y_mm[1:]),
        ("y_top_left_mm", [row["y_top_left_mm"] for row in metadata], geometry.top_y_mm[:-1]),
        ("y_top_right_mm", [row["y_top_right_mm"] for row in metadata], geometry.top_y_mm[1:]),
    ):
        if not np.allclose(np.asarray(actual, dtype=np.float64), expected, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"HDF5 {name} does not match geometry JSON")


def _build_flow_weights(eta: np.ndarray, heights_m: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(eta, dtype=np.float32)
    for interface_id in range(eta.shape[0]):
        valid_indices = np.flatnonzero((eta[interface_id] >= 0.0) & (eta[interface_id] <= 1.0))
        if valid_indices.size < 2:
            raise ValueError(f"Interface {interface_id} has fewer than two valid fluid points")
        eta_valid = eta[interface_id, valid_indices]
        sort_order = np.argsort(eta_valid)
        sorted_indices = valid_indices[sort_order]
        y_valid_m = eta[interface_id, sorted_indices] * heights_m[interface_id]
        if np.any(np.diff(y_valid_m) <= 0.0):
            raise ValueError(f"Interface {interface_id} eta coordinates are not strictly increasing")
        coordinate_differences = np.diff(y_valid_m)
        trapezoid = np.zeros_like(y_valid_m, dtype=np.float64)
        trapezoid[0] = coordinate_differences[0] / 2.0
        trapezoid[-1] = coordinate_differences[-1] / 2.0
        if trapezoid.size > 2:
            trapezoid[1:-1] = (coordinate_differences[:-1] + coordinate_differences[1:]) / 2.0
        weights[interface_id, sorted_indices] = trapezoid.astype(np.float32)
    return weights


def load_selected_case(
    dataset_path: Path,
    geometry_path: Path,
    config: UnknownInterfaceConfig,
) -> CaseData:
    """Load only branch/query arrays for one requested case."""

    _require_file(dataset_path, "Dataset")
    geometry = _load_geometry(geometry_path, config)
    with h5py.File(dataset_path, "r") as handle:
        attrs = handle.attrs
        n_interface_points = int(attrs["n_interface_points"])
        n_boundary_points = int(attrs["n_boundary_points"])
        if n_interface_points != config.interface_points:
            raise ValueError("HDF5 interface point count does not match the current protocol")
        if n_boundary_points != config.interface_points:
            raise ValueError("HDF5 boundary point count does not match the current protocol")
        branch_names = tuple(_decode_text_attribute(attrs["branch_channel_names"], "branch_channel_names").split("\n"))
        trunk_names = tuple(_decode_text_attribute(attrs["trunk_channel_names"], "trunk_channel_names").split("\n"))
        output_names = tuple(_decode_text_attribute(attrs["output_channel_names"], "output_channel_names").split("\n"))
        _compare_exact("branch_channel_names", BRANCH_CHANNEL_NAMES, branch_names)
        _compare_exact("trunk_channel_names", TRUNK_CHANNEL_NAMES, trunk_names)
        _compare_exact("output_channel_names", OUTPUT_CHANNEL_NAMES, output_names)
        include_endpoints = bool(attrs.get("include_interface_endpoints", False))
        horizontal_interface = bool(attrs.get("horizontal_interface", False))
        horizontal_jitter = float(attrs.get("horizontal_interface_jitter", 0.0))

        n_samples = int(attrs["n_samples"])
        case_ids_all_array = _decode_case_ids(_metadata_array(handle, "case_id"))
        subdomain_ids_all = np.asarray(_metadata_array(handle, "subdomain_id"), dtype=np.int64)
        if case_ids_all_array.shape[0] != n_samples or subdomain_ids_all.shape[0] != n_samples:
            raise ValueError("HDF5 metadata length does not match n_samples")
        all_case_ids = tuple(sorted({str(value) for value in case_ids_all_array.tolist()}))
        selected = np.flatnonzero(case_ids_all_array == config.case_id)
        if selected.size != config.num_subdomains:
            raise ValueError(
                f"Expected {config.num_subdomains} samples for {config.case_id}, found {selected.size}"
            )
        order = np.argsort(subdomain_ids_all[selected])
        selected = selected[order]
        selected_subdomain_ids = subdomain_ids_all[selected]
        if not np.array_equal(selected_subdomain_ids, np.arange(config.num_subdomains, dtype=np.int64)):
            raise ValueError("Selected subdomain IDs are not exactly 0..9")

        metadata_names = (
            "case_id",
            "subdomain_id",
            "local_aspect_ratio",
            "realization_id",
            "x_left_mm",
            "x_right_mm",
            "y_bottom_left_mm",
            "y_bottom_right_mm",
            "y_top_left_mm",
            "y_top_right_mm",
            "y_local_origin_mm",
            "y_local_scale_mm",
        )
        metadata_arrays = {name: _metadata_array(handle, name) for name in metadata_names}
        metadata: list[dict[str, Any]] = []
        branch_list: list[np.ndarray] = []
        query_list: list[np.ndarray] = []
        for sample_index in selected.tolist():
            row = {name: _decode_scalar(metadata_arrays[name][sample_index]) for name in metadata_names}
            metadata.append(row)
            sample_group = handle[f"samples/{int(sample_index)}"]
            if "branch" not in sample_group or "query" not in sample_group:
                raise KeyError(f"Sample {sample_index} is missing branch or query")
            branch_loaded = np.asarray(sample_group["branch"][...])
            query_loaded = np.asarray(sample_group["query"][...])
            if not np.issubdtype(branch_loaded.dtype, np.floating):
                raise ValueError(f"Sample {sample_index} branch must have a floating dtype")
            if not np.issubdtype(query_loaded.dtype, np.floating):
                raise ValueError(f"Sample {sample_index} query must have a floating dtype")
            branch = branch_loaded.astype(np.float32, copy=False)
            query = query_loaded.astype(np.float32, copy=False)
            if branch.shape != (config.interface_points * 4, len(BRANCH_CHANNEL_NAMES)):
                raise ValueError(f"Sample {sample_index} has unexpected branch shape {branch.shape}")
            if query.ndim != 2 or query.shape[1] != len(TRUNK_CHANNEL_NAMES):
                raise ValueError(f"Sample {sample_index} has unexpected query shape {query.shape}")
            if not np.isfinite(branch).all() or not np.isfinite(query).all():
                raise ValueError(f"Sample {sample_index} branch/query contains non-finite values")
            branch_list.append(branch)
            query_list.append(query)

    _validate_hdf5_geometry(metadata, geometry, config)
    branch_raw = np.stack(branch_list, axis=0).astype(np.float32, copy=False)
    query_concat = np.concatenate(query_list, axis=0).astype(np.float32, copy=False)
    query_batch_id = np.concatenate(
        [np.full(query.shape[0], index, dtype=np.int64) for index, query in enumerate(query_list)],
        axis=0,
    )
    subdomain_id = np.concatenate(
        [np.full(query.shape[0], index, dtype=np.int64) for index, query in enumerate(query_list)],
        axis=0,
    )
    global_query_list: list[np.ndarray] = []
    for query, row in zip(query_list, metadata):
        x_global = float(row["x_left_mm"]) + query[:, 0] * (
            float(row["x_right_mm"]) - float(row["x_left_mm"])
        )
        y_global = float(row["y_local_origin_mm"]) + query[:, 1] * float(row["y_local_scale_mm"])
        global_query_list.append(np.column_stack((x_global, y_global)).astype(np.float64, copy=False))
    global_coordinates = np.concatenate(global_query_list, axis=0)
    if not np.isfinite(global_coordinates).all():
        raise ValueError("Mapped query coordinates must be finite")
    bottom_interp = np.interp(global_coordinates[:, 0], geometry.x_points_mm, geometry.bottom_y_mm)
    top_interp = np.interp(global_coordinates[:, 0], geometry.x_points_mm, geometry.top_y_mm)
    inside_geometry_mask = (
        (global_coordinates[:, 1] >= bottom_interp - config.geometry_tolerance_mm)
        & (global_coordinates[:, 1] <= top_interp + config.geometry_tolerance_mm)
    )

    left_indices = np.arange(LEFT_EDGE_START, RIGHT_EDGE_START, dtype=np.int64)
    right_indices = np.arange(RIGHT_EDGE_START, BOTTOM_EDGE_START, dtype=np.int64)
    eta_left = branch_raw[:-1, right_indices, 1]
    eta_right = branch_raw[1:, left_indices, 1]
    if np.max(np.abs(eta_left - eta_right)) > config.interface_eta_tolerance:
        raise ValueError("Neighboring interface eta coordinates do not match")
    interface_eta = 0.5 * (eta_left + eta_right)
    fluid_mask = (interface_eta >= 0.0) & (interface_eta <= 1.0)
    velocity_zero_mask = ~fluid_mask.copy()
    for interface_id in range(config.num_internal_interfaces):
        valid = np.flatnonzero(fluid_mask[interface_id])
        if valid.size < 2:
            raise ValueError(f"Interface {interface_id} has too few valid fluid points")
        velocity_zero_mask[interface_id, valid[0]] = True
        velocity_zero_mask[interface_id, valid[-1]] = True
    flow_weights = _build_flow_weights(interface_eta, geometry.interface_height_m[1:-1])
    return CaseData(
        case_id=config.case_id,
        sample_indices=tuple(int(value) for value in selected.tolist()),
        all_case_ids=all_case_ids,
        branch_channel_names=branch_names,
        trunk_channel_names=trunk_names,
        output_channel_names=output_names,
        n_interface_points=n_interface_points,
        n_boundary_points=n_boundary_points,
        include_interface_endpoints=include_endpoints,
        horizontal_interface=horizontal_interface,
        horizontal_interface_jitter=horizontal_jitter,
        branch_raw=branch_raw,
        query_list=tuple(query_list),
        metadata=tuple(metadata),
        query_concat_np=query_concat,
        query_batch_id_np=query_batch_id,
        subdomain_id_np=subdomain_id,
        global_coordinates_mm=global_coordinates,
        inside_geometry_mask_np=inside_geometry_mask,
        geometry=geometry,
        interface_eta_np=interface_eta,
        fluid_mask_np=fluid_mask,
        velocity_zero_mask_np=velocity_zero_mask,
        flow_weights_np=flow_weights,
        split_membership=None,
    )


def _load_stage3_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load only the explicitly supported current Python Stage-3 format."""

    _require_file(path, "Stage-3 checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Checkpoint is not a current Python Stage-3 checkpoint; historical notebook checkpoints are unsupported"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Stage-3 checkpoint must contain a mapping")
    _compare_exact("schema_version", STAGE3_SCHEMA, checkpoint.get("schema_version"))
    _compare_exact("stage", 3, checkpoint.get("stage"))
    _compare_exact("stage_identity_source", "explicit_new_checkpoint", checkpoint.get("stage_identity_source"))
    required = (
        "student_model_state_dict",
        "model_config",
        "diffusion_config",
        "branch_channel_names",
        "trunk_channel_names",
        "output_channel_names",
        "split_case_ids",
        "split_indices",
        "normalization",
        "student_timesteps",
        "student_sampling_steps",
        "dataset_identity",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError("Current Stage-3 checkpoint is missing required fields: " + ", ".join(missing))
    return checkpoint


def _validate_dataset_identity(checkpoint: Mapping[str, Any], dataset_identity: Mapping[str, Any]) -> None:
    stored = checkpoint["dataset_identity"]
    if not isinstance(stored, Mapping):
        raise TypeError("Checkpoint dataset_identity must be a mapping")
    for name in ("size_bytes", "sha256"):
        if stored.get(name) is None:
            raise ValueError(f"Checkpoint dataset_identity.{name} is unavailable")
        _compare_exact(f"dataset_identity.{name}", dataset_identity[name], stored[name])
    stored_path = Path(str(stored.get("resolved_path", stored.get("path", "")))).expanduser().resolve(strict=False)
    current_path = Path(str(dataset_identity["resolved_path"])).expanduser().resolve(strict=False)
    if stored_path != current_path:
        raise ValueError("Checkpoint dataset identity path does not match the requested dataset")


def _validate_checkpoint_protocol(
    checkpoint: Mapping[str, Any],
    case: CaseData,
    config: UnknownInterfaceConfig,
    dataset_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, float, float, Optional[str]]:
    _validate_dataset_identity(checkpoint, dataset_identity)
    _compare_exact("branch_channel_names", list(BRANCH_CHANNEL_NAMES), _as_list(checkpoint["branch_channel_names"], "branch_channel_names"))
    _compare_exact("trunk_channel_names", list(TRUNK_CHANNEL_NAMES), _as_list(checkpoint["trunk_channel_names"], "trunk_channel_names"))
    _compare_exact("output_channel_names", list(OUTPUT_CHANNEL_NAMES), _as_list(checkpoint["output_channel_names"], "output_channel_names"))
    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]
    if not isinstance(model_config, Mapping) or not isinstance(diffusion_config, Mapping):
        raise TypeError("Checkpoint model_config and diffusion_config must be mappings")
    expected_model_config = {
        "branch_input_dim": len(BRANCH_CHANNEL_NAMES),
        "query_input_dim": len(TRUNK_CHANNEL_NAMES),
        "target_dim": len(OUTPUT_CHANNEL_NAMES),
        "latent_dim": 128,
        "time_dim": 128,
        "branch_point_hidden_dim": 128,
        "branch_global_hidden_dim": 128,
        "denoiser_hidden_dim": 256,
        "denoiser_depth": 4,
    }
    _compare_exact("model_config", expected_model_config, dict(model_config))
    expected_diffusion_config = {
        "T": config.diffusion_steps,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
    }
    _compare_exact("diffusion_config", expected_diffusion_config, dict(diffusion_config))
    _compare_exact("student_timesteps", list(config.student_timesteps), [int(value) for value in _as_list(checkpoint["student_timesteps"], "student_timesteps")])
    _compare_exact("student_sampling_steps", 5, int(checkpoint["student_sampling_steps"]))
    normalization = checkpoint["normalization"]
    if not isinstance(normalization, Mapping):
        raise TypeError("Checkpoint normalization must be a mapping")
    target_mean = torch.as_tensor(normalization.get("target_mean"), dtype=torch.float32, device="cpu")
    target_std = torch.as_tensor(normalization.get("target_std"), dtype=torch.float32, device="cpu")
    if target_mean.shape != (3,) or target_std.shape != (3,) or not torch.isfinite(target_mean).all() or not torch.isfinite(target_std).all() or torch.any(target_std <= 0.0):
        raise ValueError("Checkpoint target normalization must contain finite positive three-channel statistics")
    local_aspect_mean = float(normalization["local_aspect_mean"])
    local_aspect_std = float(normalization["local_aspect_std"])
    if not math.isfinite(local_aspect_mean) or not math.isfinite(local_aspect_std) or local_aspect_std <= 0.0:
        raise ValueError("Checkpoint local-aspect normalization must be finite and positive")
    split_case_ids = checkpoint["split_case_ids"]
    if not isinstance(split_case_ids, Mapping):
        raise TypeError("Checkpoint split_case_ids must be a mapping")
    split_lists = {
        name: _as_list(split_case_ids.get(name, []), f"split_case_ids.{name}")
        for name in ("train", "val", "test")
    }
    memberships = [name for name, values in split_lists.items() if case.case_id in values]
    if len(memberships) != 1:
        raise ValueError("Requested case must belong to exactly one checkpoint split")
    stored_values = [str(value) for values in split_lists.values() for value in values]
    if len(stored_values) != len(set(stored_values)):
        raise ValueError("Checkpoint split case IDs must not contain duplicates")
    if set(split_lists["train"]).intersection(split_lists["val"]) or set(split_lists["train"]).intersection(split_lists["test"]) or set(split_lists["val"]).intersection(split_lists["test"]):
        raise ValueError("Checkpoint split case IDs must be disjoint")
    stored_all_cases = sorted(set(stored_values))
    if stored_all_cases != list(case.all_case_ids):
        raise ValueError("Checkpoint split case IDs do not match the dataset case IDs")
    if not isinstance(checkpoint["student_model_state_dict"], Mapping):
        raise TypeError("student_model_state_dict must be a mapping")
    return dict(model_config), target_mean, target_std, local_aspect_mean, local_aspect_std, memberships[0]


def _prepare_truth_free_branch(case: CaseData, normalizer: FeatureNormalizer, local_aspect_mean: float, local_aspect_std: float) -> np.ndarray:
    normalized = np.stack(
        [
            normalize_diffusion_branch(
                branch,
                branch_channel_names=BRANCH_CHANNEL_NAMES,
                target_normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                zero_unknown_values=True,
            )
            for branch in case.branch_raw
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    value_indices = np.asarray([4, 5, 6], dtype=np.int64)
    known_indices = np.asarray([7, 8, 9], dtype=np.int64)
    left_indices = np.arange(LEFT_EDGE_START, RIGHT_EDGE_START, dtype=np.int64)
    right_indices = np.arange(RIGHT_EDGE_START, BOTTOM_EDGE_START, dtype=np.int64)
    for interface_id in range(len(normalized) - 1):
        normalized[interface_id, right_indices[:, None], value_indices[None, :]] = 0.0
        normalized[interface_id + 1, left_indices[:, None], value_indices[None, :]] = 0.0
        normalized[interface_id, right_indices[:, None], known_indices[None, :]] = 1.0
        normalized[interface_id + 1, left_indices[:, None], known_indices[None, :]] = 1.0
    if not np.isfinite(normalized).all():
        raise ValueError("Truth-free normalized branches contain non-finite values")
    return normalized


def _build_initial_interface_state(
    case: CaseData,
    config: UnknownInterfaceConfig,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray]:
    geometry = case.geometry
    q_reference = config.inlet_u * geometry.interface_height_m[0]
    pressure_at_station = np.zeros(config.num_subdomains + 1, dtype=np.float64)
    pressure_at_station[-1] = config.outlet_gauge_pressure
    for segment_id in range(config.num_subdomains - 1, -1, -1):
        x_left = geometry.x_points_m[segment_id]
        x_right = geometry.x_points_m[segment_id + 1]
        height_left = geometry.interface_height_m[segment_id]
        height_right = geometry.interface_height_m[segment_id + 1]
        x_grid = np.linspace(x_left, x_right, 257)
        height_grid = np.interp(x_grid, geometry.x_points_m, geometry.interface_height_m)
        gradient = -12.0 * config.dynamic_viscosity * q_reference / np.maximum(height_grid, 1.0e-12) ** 3
        pressure_drop = float(np.trapezoid(gradient, x_grid))
        pressure_at_station[segment_id] = pressure_at_station[segment_id + 1] - pressure_drop
        del height_left, height_right
    pressure_interfaces = pressure_at_station[1:-1]
    eta = case.interface_eta_np
    eta_clipped = np.clip(eta, 0.0, 1.0)
    fluid = case.fluid_mask_np
    heights = geometry.interface_height_m[1:-1]
    slopes = geometry.height_slope[1:-1]
    pressure = np.repeat(pressure_interfaces[:, None], case.n_interface_points, axis=1)
    mean_velocity = q_reference / np.maximum(heights[:, None], 1.0e-12)
    u = 6.0 * mean_velocity * eta_clipped * (1.0 - eta_clipped)
    v = mean_velocity * slopes[:, None] * (eta_clipped - 0.5)
    u = np.where(fluid, u, 0.0)
    v = np.where(fluid, v, 0.0)
    for interface_id in range(case.interface_eta_np.shape[0]):
        valid = np.flatnonzero(fluid[interface_id])
        u[interface_id, valid[0]] = 0.0
        u[interface_id, valid[-1]] = 0.0
        v[interface_id, valid[0]] = 0.0
        v[interface_id, valid[-1]] = 0.0
    physical = np.stack((pressure, u, v), axis=-1).astype(np.float32)
    target_mean_cpu = target_mean.detach().cpu()
    target_std_cpu = target_std.detach().cpu()
    normalized = (torch.from_numpy(physical) - target_mean_cpu.reshape(1, 1, 3)) / target_std_cpu.reshape(1, 1, 3)
    return normalized, physical


def _build_runtime(
    config: UnknownInterfaceConfig,
    device: torch.device,
    dataset_path: Path,
    checkpoint_path: Path,
    geometry_path: Path,
) -> RuntimeContext:
    dataset_identity = file_identity(dataset_path)
    checkpoint = _load_stage3_checkpoint(checkpoint_path)
    case = load_selected_case(dataset_path, geometry_path, config)
    model_config, target_mean_cpu, target_std_cpu, local_aspect_mean, local_aspect_std, split_membership = _validate_checkpoint_protocol(
        checkpoint,
        case,
        config,
        dataset_identity,
    )
    case = replace(case, split_membership=split_membership)
    model = PointSetDiffusionDenoiser(**model_config)
    model.load_state_dict(checkpoint["student_model_state_dict"], strict=True)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Model parameter count {parameter_count} does not match {EXPECTED_PARAMETER_COUNT}")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_mean = target_mean_cpu.to(device=device)
    target_std = target_std_cpu.to(device=device)
    normalizer = FeatureNormalizer.from_state_dict({"mean": target_mean_cpu, "std": target_std_cpu})
    truth_free_branch_np = _prepare_truth_free_branch(case, normalizer, local_aspect_mean, local_aspect_std)
    truth_free_branch = torch.from_numpy(truth_free_branch_np).to(device=device, dtype=torch.float32)
    branch_mask = torch.ones(truth_free_branch.shape[:2], device=device, dtype=torch.bool)
    query = torch.from_numpy(case.query_concat_np).to(device=device, dtype=torch.float32)
    query_batch_id = torch.from_numpy(case.query_batch_id_np).to(device=device, dtype=torch.long)
    left_indices = np.arange(LEFT_EDGE_START, RIGHT_EDGE_START, dtype=np.int64)
    right_indices = np.arange(RIGHT_EDGE_START, BOTTOM_EDGE_START, dtype=np.int64)
    boundary_query_np = np.concatenate(
        [
            truth_free_branch_np[:, left_indices, :2],
            truth_free_branch_np[:, right_indices, :2],
        ],
        axis=1,
    ).reshape(-1, 2)
    boundary_query_batch_id_np = np.repeat(np.arange(config.num_subdomains, dtype=np.int64), config.interface_points * 2)
    boundary_query = torch.from_numpy(boundary_query_np).to(device=device, dtype=torch.float32)
    boundary_query_batch_id = torch.from_numpy(boundary_query_batch_id_np).to(device=device, dtype=torch.long)
    schedule = build_linear_schedule(
        timesteps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
        dtype=torch.float32,
    )
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(config.inference_noise_seed)
    fixed_initial_noise = torch.randn((query.shape[0], 3), generator=noise_generator, device=device, dtype=torch.float32)
    boundary_generator = torch.Generator(device=device)
    boundary_generator.manual_seed(config.boundary_noise_seed)
    fixed_boundary_initial_noise = torch.randn((boundary_query.shape[0], 3), generator=boundary_generator, device=device, dtype=torch.float32)
    initial_interface_state, _ = _build_initial_interface_state(case, config, target_mean, target_std)
    initial_interface_state = initial_interface_state.to(device=device, dtype=torch.float32)
    return RuntimeContext(
        config=config,
        case=case,
        checkpoint=checkpoint,
        device=device,
        model=model,
        schedule=schedule,
        target_mean=target_mean,
        target_std=target_std,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        truth_free_branch=truth_free_branch,
        branch_mask=branch_mask,
        query=query,
        query_batch_id=query_batch_id,
        fixed_initial_noise=fixed_initial_noise,
        boundary_query=boundary_query,
        boundary_query_batch_id=boundary_query_batch_id,
        fixed_boundary_initial_noise=fixed_boundary_initial_noise,
        flow_weights=torch.from_numpy(case.flow_weights_np).to(device=device, dtype=torch.float32),
        fluid_mask=torch.from_numpy(case.fluid_mask_np).to(device=device, dtype=torch.bool),
        velocity_zero_mask=torch.from_numpy(case.velocity_zero_mask_np).to(device=device, dtype=torch.bool),
        physics_prior=initial_interface_state.detach().clone(),
        q_reference=torch.tensor(config.inlet_u * case.geometry.interface_height_m[0], device=device, dtype=torch.float32),
    )


def build_branch_from_interface_state(runtime: RuntimeContext, interface_state: torch.Tensor) -> torch.Tensor:
    """Insert one normalized interface state into the truth-free branch template."""

    expected_shape = (runtime.config.num_internal_interfaces, runtime.config.interface_points, 3)
    if tuple(interface_state.shape) != expected_shape:
        raise ValueError(f"Interface state shape {tuple(interface_state.shape)} does not match {expected_shape}")
    branch = runtime.truth_free_branch.clone()
    left_indices = torch.arange(LEFT_EDGE_START, RIGHT_EDGE_START, device=runtime.device, dtype=torch.long)
    right_indices = torch.arange(RIGHT_EDGE_START, BOTTOM_EDGE_START, device=runtime.device, dtype=torch.long)
    value_indices = torch.tensor([4, 5, 6], device=runtime.device, dtype=torch.long)
    for interface_id in range(runtime.config.num_internal_interfaces):
        branch[interface_id, right_indices[:, None], value_indices[None, :]] = interface_state[interface_id]
        branch[interface_id + 1, left_indices[:, None], value_indices[None, :]] = interface_state[interface_id]
    return branch


def differentiable_student_rollout(runtime: RuntimeContext, branch: torch.Tensor, query: torch.Tensor, query_batch_id: torch.Tensor, initial_noise: torch.Tensor) -> torch.Tensor:
    """Run the fixed five-evaluation student schedule without detaching gradients."""

    if len(runtime.config.student_timesteps) != 5:
        raise ValueError("The fixed student rollout requires exactly five timesteps")
    x_state = initial_noise
    for index in range(4):
        t_current = int(runtime.config.student_timesteps[index])
        t_next = int(runtime.config.student_timesteps[index + 1])
        t_query = torch.full((query.shape[0],), t_current, device=query.device, dtype=torch.long)
        epsilon_pred = runtime.model(
            branch=branch,
            query=query,
            noisy_target=x_state,
            t_query=t_query,
            query_batch_id=query_batch_id,
            branch_mask=runtime.branch_mask,
        )
        x_state, _ = ddim_step(x_state, epsilon_pred, t_current, t_next, runtime.schedule.alphas_cumprod)
    t_zero = torch.zeros((query.shape[0],), device=query.device, dtype=torch.long)
    epsilon_zero = runtime.model(
        branch=branch,
        query=query,
        noisy_target=x_state,
        t_query=t_zero,
        query_batch_id=query_batch_id,
        branch_mask=runtime.branch_mask,
    )
    return final_clean_projection(x_state, epsilon_zero, 0, runtime.schedule.alphas_cumprod)


def interface_normalized_to_physical(runtime: RuntimeContext, interface_state: torch.Tensor) -> torch.Tensor:
    return interface_state * runtime.target_std.reshape(1, 1, 3) + runtime.target_mean.reshape(1, 1, 3)


def _boundary_predictions(runtime: RuntimeContext, interface_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    branch = build_branch_from_interface_state(runtime, interface_state)
    prediction = differentiable_student_rollout(
        runtime,
        branch,
        runtime.boundary_query,
        runtime.boundary_query_batch_id,
        runtime.fixed_boundary_initial_noise,
    ).reshape(runtime.config.num_subdomains, runtime.config.interface_points * 2, 3)
    left = prediction[:, : runtime.config.interface_points, :]
    right = prediction[:, runtime.config.interface_points :, :]
    return right[:-1], left[1:]


def _objective_from_traces(
    runtime: RuntimeContext,
    interface_state: torch.Tensor,
    prediction_left: torch.Tensor,
    prediction_right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    field_weights = torch.as_tensor(runtime.config.field_weights, device=runtime.device, dtype=interface_state.dtype)
    fixed_left = (prediction_left - interface_state).square().mean(dim=(0, 1))
    fixed_right = (prediction_right - interface_state).square().mean(dim=(0, 1))
    fixed_point_loss = (0.5 * (fixed_left + fixed_right) * field_weights).sum() / field_weights.sum()
    neighbor_per_field = (prediction_left - prediction_right).square().mean(dim=(0, 1))
    neighbor_loss = (neighbor_per_field * field_weights).sum() / field_weights.sum()
    physical = interface_normalized_to_physical(runtime, interface_state)
    pressure = physical[..., 0]
    u = physical[..., 1]
    v = physical[..., 2]
    interface_flow_rates = (runtime.flow_weights * u).sum(dim=1)
    relative_flux_residual = (interface_flow_rates - runtime.q_reference) / runtime.q_reference
    mass_flux_loss = relative_flux_residual.square().mean()
    wall_u = u[runtime.velocity_zero_mask]
    wall_v = v[runtime.velocity_zero_mask]
    wall_loss = (wall_u / runtime.config.inlet_u).square().mean() + (wall_v / runtime.config.inlet_u).square().mean()
    second_difference = interface_state[:, 2:, :] - 2.0 * interface_state[:, 1:-1, :] + interface_state[:, :-2, :]
    smoothness_loss = second_difference.square().mean()
    pressure_means: list[torch.Tensor] = []
    pressure_transverse_terms: list[torch.Tensor] = []
    for interface_id in range(runtime.config.num_internal_interfaces):
        pressure_valid = pressure[interface_id][runtime.fluid_mask[interface_id]]
        if pressure_valid.numel() == 0:
            raise ValueError(f"Interface {interface_id} has no valid pressure points")
        pressure_mean = pressure_valid.mean()
        pressure_means.append(pressure_mean)
        pressure_transverse_terms.append(((pressure_valid - pressure_mean) / runtime.target_std[0]).square().mean())
    pressure_mean_by_interface = torch.stack(pressure_means)
    pressure_transverse_loss = torch.stack(pressure_transverse_terms).mean()
    downstream_increase = torch.relu(pressure_mean_by_interface[1:] - pressure_mean_by_interface[:-1])
    pressure_monotonic_loss = (downstream_increase / runtime.target_std[0]).square().mean()
    prior_loss = (interface_state - runtime.physics_prior).square().mean()
    total_loss = (
        runtime.config.fixed_point_weight * fixed_point_loss
        + runtime.config.neighbor_weight * neighbor_loss
        + runtime.config.mass_flux_weight * mass_flux_loss
        + runtime.config.wall_weight * wall_loss
        + runtime.config.smoothness_weight * smoothness_loss
        + runtime.config.pressure_transverse_weight * pressure_transverse_loss
        + runtime.config.pressure_monotonic_weight * pressure_monotonic_loss
        + runtime.config.prior_weight * prior_loss
    )
    return {
        "total_loss": total_loss,
        "fixed_point_loss": fixed_point_loss,
        "neighbor_loss": neighbor_loss,
        "mass_flux_loss": mass_flux_loss,
        "wall_loss": wall_loss,
        "smoothness_loss": smoothness_loss,
        "pressure_transverse_loss": pressure_transverse_loss,
        "pressure_monotonic_loss": pressure_monotonic_loss,
        "prior_loss": prior_loss,
        "gradient_norm": torch.zeros((), device=runtime.device, dtype=interface_state.dtype),
        "max_relative_mass_flux_residual": relative_flux_residual.abs().max(),
        "mean_relative_mass_flux_residual": relative_flux_residual.abs().mean(),
        "interface_flow_rates": interface_flow_rates,
        "relative_flux_residual": relative_flux_residual,
        "pressure_mean_by_interface": pressure_mean_by_interface,
    }


def compute_interface_objective(runtime: RuntimeContext, interface_state: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction_left, prediction_right = _boundary_predictions(runtime, interface_state)
    return _objective_from_traces(runtime, interface_state, prediction_left, prediction_right)


def optimize_interface(runtime: RuntimeContext, initial_interface_state: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float]], int]:
    """Optimize only the normalized internal interface state."""

    optimized = torch.nn.Parameter(initial_interface_state.detach().clone())
    optimizer = torch.optim.Adam([optimized], lr=runtime.config.learning_rate)
    lower = torch.as_tensor(runtime.config.normalized_lower_bounds, device=runtime.device, dtype=optimized.dtype).reshape(1, 1, 3)
    upper = torch.as_tensor(runtime.config.normalized_upper_bounds, device=runtime.device, dtype=optimized.dtype).reshape(1, 1, 3)
    best_loss = math.inf
    best_state = optimized.detach().clone()
    best_step = 0
    history: list[dict[str, float]] = []
    for step in range(1, runtime.config.consistency_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result = compute_interface_objective(runtime, optimized)
        loss = result["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite physics objective at step {step}")
        loss.backward()
        if optimized.grad is None or not torch.isfinite(optimized.grad).all():
            raise RuntimeError(f"Non-finite interface gradient at step {step}")
        gradient_norm = float(torch.linalg.vector_norm(optimized.grad.detach()).cpu())
        torch.nn.utils.clip_grad_norm_([optimized], runtime.config.gradient_clip)
        optimizer.step()
        with torch.no_grad():
            optimized.clamp_(lower, upper)
        with torch.no_grad():
            updated = compute_interface_objective(runtime, optimized)
        updated_loss = float(updated["total_loss"].cpu())
        row = {"step": float(step)}
        for name in (
            "total_loss",
            "fixed_point_loss",
            "neighbor_loss",
            "mass_flux_loss",
            "wall_loss",
            "smoothness_loss",
            "pressure_transverse_loss",
            "pressure_monotonic_loss",
            "prior_loss",
            "max_relative_mass_flux_residual",
            "mean_relative_mass_flux_residual",
        ):
            row[name] = float(updated[name].cpu())
        row["gradient_norm"] = gradient_norm
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        if updated_loss < best_loss:
            best_loss = updated_loss
            best_state = optimized.detach().clone()
            best_step = step
    return best_state, history, best_step


def generate_final_prediction(runtime: RuntimeContext, optimized_interface_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Generate one final full-channel prediction from the optimized interface state."""

    branch = build_branch_from_interface_state(runtime, optimized_interface_state)
    with torch.no_grad():
        normalized = differentiable_student_rollout(
            runtime,
            branch,
            runtime.query,
            runtime.query_batch_id,
            runtime.fixed_initial_noise,
        )
        physical = normalized * runtime.target_std.reshape(1, 3) + runtime.target_mean.reshape(1, 3)
    if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
        raise RuntimeError("Final prediction contains non-finite values")
    return {
        "normalized": normalized.detach().cpu(),
        "physical": physical.detach().cpu(),
    }


def reconstruct(case: CaseData, final_prediction: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Preserve pointwise concatenated output without seam averaging or overwrite."""

    return {
        "global_coordinates_mm": case.global_coordinates_mm.astype(np.float64, copy=True),
        "subdomain_id": case.subdomain_id_np.astype(np.int64, copy=True),
        "inside_geometry_mask": case.inside_geometry_mask_np.astype(bool, copy=True),
        "physical_prediction": final_prediction["physical"].numpy().astype(np.float32, copy=True),
        "pointwise_order_policy": "concatenated_subdomain_order",
        "seam_policy": "no_average_no_overwrite",
        "coordinate_policy": "local_query_to_global_mm_using_hdf5_metadata",
    }


def load_posthoc_targets(dataset_path: Path, case: CaseData) -> np.ndarray:
    """Load CFD targets only for explicitly enabled post-hoc evaluation."""

    targets: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as handle:
        for sample_index, query in zip(case.sample_indices, case.query_list):
            target_path = f"samples/{sample_index}/target"
            if target_path not in handle:
                raise KeyError(f"Post-hoc evaluation requires {target_path}")
            target = np.asarray(handle[target_path][...], dtype=np.float32).squeeze()
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
            if target.shape[0] != query.shape[0]:
                raise ValueError(f"Target/query row count differs for {target_path}")
            if not np.isfinite(target).all():
                raise ValueError(f"Target {target_path} contains non-finite values")
            targets.append(target.astype(np.float32, copy=False))
    return np.concatenate(targets, axis=0)


def compute_posthoc_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Compute physical-space metrics without changing the final prediction."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Prediction and target must have matching shape (N, 3)")
    if mask.shape != (prediction.shape[0],) or mask.dtype != bool:
        raise ValueError("Post-hoc geometry mask must have boolean shape (N,)")
    if not mask.any():
        raise ValueError("Post-hoc geometry mask contains no valid points")
    prediction_valid = prediction[mask]
    target_valid = target[mask]
    names = list(OUTPUT_CHANNEL_NAMES)
    metrics: dict[str, Any] = {"mask_fraction": float(mask.mean()), "channels": {}}
    for index, name in enumerate(names):
        error = prediction_valid[:, index] - target_valid[:, index]
        mse = float(np.mean(error**2))
        target_field = target_valid[:, index]
        variance = float(np.var(target_field))
        if variance > 0.0:
            r2 = 1.0 - mse / variance
            correlation_value = float(np.corrcoef(target_field, prediction_valid[:, index])[0, 1])
            correlation = correlation_value if math.isfinite(correlation_value) else None
        else:
            r2 = None
            correlation = None
        unit = "Pa" if name == "pressure" else "m/s"
        metrics["channels"][name] = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(error))),
            "r2": r2,
            "correlation": correlation,
            "unit": unit,
        }
    return metrics


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_history(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    fieldnames = [
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
        "learning_rate",
        "max_relative_mass_flux_residual",
        "mean_relative_mass_flux_residual",
    ]
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _plain_json_value(value: Any) -> Any:
    """Convert checkpoint values to JSON-compatible Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.Tensor, np.ndarray, np.integer, np.floating, Path)):
        return _json_default(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _artifact_identity(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity["path"] = _manifest_path(path)
    return identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_manifest(
    *,
    config: UnknownInterfaceConfig,
    run_id: str,
    run_directory: Path,
    dataset_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    source_identities: Sequence[Mapping[str, Any]],
    case: CaseData,
    checkpoint: Mapping[str, Any],
    status: str,
    created_at_utc: str,
    started_at_utc: Optional[str],
    finished_at_utc: Optional[str],
    failure_type: Optional[str],
    failure_message: Optional[str],
    selected_iteration: Optional[int],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    output_files = {name: _manifest_path(path) for name, path in artifact_paths.items() if path.exists()}
    output_identities = {name: _artifact_identity(path) for name, path in artifact_paths.items() if path.exists()}
    normalization = checkpoint.get("normalization", {})
    return {
        "schema_version": "unknown_interface_run_v1",
        "run_id": run_id,
        "timestamp_utc": created_at_utc,
        "status": status,
        "created_at_utc": created_at_utc,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "git": git_state(REPOSITORY_ROOT),
        "source_files": list(source_identities),
        "dataset": {
            **dict(dataset_identity),
            "path": _manifest_path(Path(str(dataset_identity["resolved_path"]))),
            "schema": {
                "branch": list(BRANCH_CHANNEL_NAMES),
                "query": list(TRUNK_CHANNEL_NAMES),
                "target": list(OUTPUT_CHANNEL_NAMES),
                "n_interface_points": config.interface_points,
                "n_boundary_points": config.interface_points,
            },
            "case_id": case.case_id,
            "sample_indices": list(case.sample_indices),
            "target_values_loaded": bool(config.enable_posthoc_evaluation and status == "completed"),
        },
        "checkpoint": {
            **dict(checkpoint_identity),
            "path": _manifest_path(Path(str(checkpoint_identity["resolved_path"]))),
            "format": STAGE3_SCHEMA,
            "stage": 3,
            "stage_identity_source": checkpoint.get("stage_identity_source"),
            "epoch": _plain_json_value(checkpoint.get("epoch")),
            "initialization_mode": _plain_json_value(checkpoint.get("initialization_mode")),
            "model_config": dict(checkpoint["model_config"]),
            "diffusion_config": dict(checkpoint["diffusion_config"]),
        },
        "protocol": {
            "method_name": METHOD_NAME,
            "method_selection_basis": "fixed after development-stage CFD comparison",
            "model_role": MODEL_ROLE,
            "runtime_uses_cfd_truth": False,
            "posthoc_evaluation_enabled": bool(config.enable_posthoc_evaluation),
            "posthoc_evaluation_uses_cfd_truth": bool(config.enable_posthoc_evaluation),
            "trainable_variables": {
                "name": "normalized_internal_interface_pressure_u_v",
                "shape": [config.num_internal_interfaces, config.interface_points, 3],
            },
            "subdomain_ids": list(range(config.num_subdomains)),
            "interface_pairing": "upstream_right_edge_to_downstream_left_edge_direct_order",
            "coordinate_policy": "local_query_to_global_mm_using_hdf5_metadata",
            "geometry_mask_policy": "stored_for_pointwise_reconstruction",
            "seam_policy": "no_average_no_overwrite",
            "normalizer": {
                "source": "stage3_checkpoint",
                "target_mean": _plain_json_value(normalization.get("target_mean")),
                "target_std": _plain_json_value(normalization.get("target_std")),
                "local_aspect_mean": _plain_json_value(normalization.get("local_aspect_mean")),
                "local_aspect_std": _plain_json_value(normalization.get("local_aspect_std")),
            },
            "initial_interface_method": "quasi_1d_physics_initialization",
            "objective_weights": {
                "fixed_point": config.fixed_point_weight,
                "neighbor": config.neighbor_weight,
                "mass_flux": config.mass_flux_weight,
                "wall": config.wall_weight,
                "smoothness": config.smoothness_weight,
                "pressure_transverse": config.pressure_transverse_weight,
                "pressure_monotonic": config.pressure_monotonic_weight,
                "prior": config.prior_weight,
            },
            "field_weights": list(config.field_weights),
            "optimizer": {
                "name": "Adam",
                "steps": config.consistency_steps,
                "learning_rate": config.learning_rate,
                "gradient_clip": config.gradient_clip,
                "selection": "minimum_total_physics_objective",
                "selected_iteration": selected_iteration,
            },
        },
        "inputs": {
            "inlet_u": config.inlet_u,
            "inlet_v": config.inlet_v,
            "outlet_gauge_pressure": config.outlet_gauge_pressure,
            "dynamic_viscosity": config.dynamic_viscosity,
            "density": config.density,
            "case_split_membership": case.split_membership,
            "interface_point_count": config.interface_points,
        },
        "randomness": {
            "global_seed": config.global_seed,
            "inference_noise_seed": config.inference_noise_seed,
            "boundary_noise_seed": config.boundary_noise_seed,
            "student_timesteps": list(config.student_timesteps),
            "dtype": "float32",
        },
        "model": {
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "eval_mode": True,
            "parameters_frozen": True,
        },
        "environment": runtime_environment(),
        "outputs": {
            "directory": _manifest_path(run_directory),
            "files": output_files,
            "identities": output_identities,
        },
    }


def _print_resolved_config(config: UnknownInterfaceConfig) -> None:
    values = asdict(config)
    values["dataset_path"] = str(_resolve_repo_path(config.dataset_path))
    values["checkpoint_path"] = None if config.checkpoint_path is None else str(_resolve_repo_path(config.checkpoint_path))
    values["geometry_path"] = str(_resolve_geometry_path(config))
    values["source_notebook"] = str(_resolve_repo_path(config.source_notebook))
    values["results_root"] = str(_resolve_repo_path(config.results_root))
    print(json.dumps(values, indent=2, sort_keys=True, default=str))


def _parse_timesteps(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("student timesteps must be comma-separated integers") from exc
    if values != (999, 749, 500, 250, 0):
        raise argparse.ArgumentTypeError("The fixed protocol requires 999,749,500,250,0")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[UnknownInterfaceConfig, bool]:
    """Parse a side-effect-free configuration preview or an explicit run request."""

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed consistency/physics optimization for one unknown-interface case. "
            "Without --run, only the resolved configuration is printed."
        )
    )
    parser.add_argument("--run", action="store_true", help="Enable case loading, optimization, and artifact creation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Existing HDF5 dataset path")
    parser.add_argument("--checkpoint", default=None, help="Current Python new-format Stage-3 checkpoint path")
    parser.add_argument("--geometry", default=None, help="Geometry JSON path; defaults to the case geometry file")
    parser.add_argument("--source-notebook", default=str(DEFAULT_SOURCE_NOTEBOOK), help="Source notebook path")
    parser.add_argument("--case-id", default="channel_36", help="Case identifier")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Results root")
    parser.add_argument("--run-id", default=None, help="Optional unique filesystem-safe run identifier")
    parser.add_argument("--device", default="cuda:1", help="CPU or explicitly indexed CUDA device")
    parser.add_argument("--global-seed", type=int, default=20260731)
    parser.add_argument("--inference-noise-seed", type=int, default=8100)
    parser.add_argument("--boundary-noise-seed", type=int, default=8200)
    parser.add_argument("--student-timesteps", type=_parse_timesteps, default=(999, 749, 500, 250, 0), help="Fixed schedule: 999,749,500,250,0")
    parser.add_argument("--inlet-u", type=float, default=1.0)
    parser.add_argument("--inlet-v", type=float, default=0.0)
    parser.add_argument("--outlet-gauge-pressure", type=float, default=0.0)
    parser.add_argument("--dynamic-viscosity", type=float, default=1.003e-3)
    parser.add_argument("--density", type=float, default=998.2)
    parser.add_argument("--consistency-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--enable-posthoc-evaluation", action="store_true", help="Load CFD targets only in the final post-hoc phase")
    args = parser.parse_args(argv)
    _validate_device_syntax(args.device)
    config = UnknownInterfaceConfig(
        dataset_path=Path(args.dataset),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        geometry_path=Path(args.geometry) if args.geometry else None,
        source_notebook=Path(args.source_notebook),
        results_root=Path(args.results_root),
        case_id=args.case_id,
        device=args.device,
        global_seed=args.global_seed,
        inference_noise_seed=args.inference_noise_seed,
        boundary_noise_seed=args.boundary_noise_seed,
        student_timesteps=tuple(args.student_timesteps),
        inlet_u=args.inlet_u,
        inlet_v=args.inlet_v,
        outlet_gauge_pressure=args.outlet_gauge_pressure,
        dynamic_viscosity=args.dynamic_viscosity,
        density=args.density,
        consistency_steps=args.consistency_steps,
        learning_rate=args.learning_rate,
        gradient_clip=args.gradient_clip,
        save_interval=args.save_interval,
        enable_posthoc_evaluation=bool(args.enable_posthoc_evaluation),
        run_id=args.run_id,
    )
    if args.run and config.checkpoint_path is None:
        parser.error("--checkpoint is required when --run is supplied")
    return config, bool(args.run)


def _validate_config(config: UnknownInterfaceConfig) -> None:
    _validate_device_syntax(config.device)
    if config.num_subdomains != 10 or config.num_internal_interfaces != 9 or config.interface_points != 256:
        raise ValueError("The fixed protocol requires 10 subdomains, 9 interfaces, and 256 interface points")
    if config.student_timesteps != (999, 749, 500, 250, 0):
        raise ValueError("The fixed protocol requires student timesteps 999,749,500,250,0")
    if config.diffusion_steps != 1000 or config.beta_start != 1.0e-4 or config.beta_end != 2.0e-2:
        raise ValueError("The fixed protocol requires the 1000-step linear diffusion schedule")
    if config.consistency_steps != 200 or config.learning_rate != 2.0e-2 or config.gradient_clip != 1.0:
        raise ValueError("The fixed protocol requires 200 Adam steps, learning rate 0.02, and gradient clip 1.0")
    if config.save_interval <= 0 or config.consistency_steps <= 0:
        raise ValueError("save_interval and consistency_steps must be positive")
    if config.inlet_u <= 0.0 or config.dynamic_viscosity <= 0.0 or config.density <= 0.0:
        raise ValueError("Inlet velocity, dynamic viscosity, and density must be positive")
    if len(config.field_weights) != 3 or any(value < 0.0 for value in config.field_weights) or sum(config.field_weights) <= 0.0:
        raise ValueError("field_weights must contain three non-negative values with a positive sum")


def run_unknown_interface(config: UnknownInterfaceConfig) -> Path:
    """Execute the fixed unknown-interface protocol and publish isolated artifacts."""

    _validate_config(config)
    device = _resolve_runtime_device(config.device)
    dataset_path = _resolve_repo_path(config.dataset_path)
    checkpoint_path = _resolve_repo_path(config.checkpoint_path) if config.checkpoint_path is not None else None
    geometry_path = _resolve_geometry_path(config)
    source_notebook_path = _resolve_repo_path(config.source_notebook)
    if checkpoint_path is None:
        raise ValueError("A current Python Stage-3 checkpoint path is required")
    _require_file(dataset_path, "Dataset")
    _require_file(checkpoint_path, "Stage-3 checkpoint")
    _require_file(source_notebook_path, "Source notebook")
    _require_file(geometry_path, "Geometry JSON")
    _set_global_seed(config.global_seed)
    dataset_identity = file_identity(dataset_path)
    checkpoint_identity = file_identity(checkpoint_path)
    source_identities = [
        {"role": "runner", **_artifact_identity(Path(__file__).resolve())},
        {"role": "source_notebook", **_artifact_identity(source_notebook_path)},
        {"role": "model_module", **_artifact_identity(REPOSITORY_ROOT / "pidiffusion" / "model.py")},
        {"role": "diffusion_module", **_artifact_identity(REPOSITORY_ROOT / "pidiffusion" / "diffusion.py")},
        {"role": "data_module", **_artifact_identity(REPOSITORY_ROOT / "pidiffusion" / "data.py")},
    ]
    runtime = _build_runtime(config, device, dataset_path, checkpoint_path, geometry_path)
    timestamp = datetime.now(timezone.utc)
    run_id = build_run_id(
        protocol="unknown_interface",
        case_id=config.case_id,
        checkpoint_tag="stage3",
        seed=config.global_seed,
        ddim_steps=5,
        timestamp_utc=timestamp,
    ) if config.run_id is None else config.run_id
    run_directory = create_run_directory(
        _resolve_repo_path(config.results_root),
        protocol="unknown_interface",
        case_id=config.case_id,
        run_id=run_id,
    )
    artifact_paths = {
        "initial_interface_state": run_directory / "initial_interface_state.pt",
        "optimized_interface_state": run_directory / "optimized_interface_state.pt",
        "optimization_history": run_directory / "optimization_history.csv",
        "final_prediction": run_directory / "final_prediction.pt",
        "reconstruction": run_directory / "reconstruction.pt",
    }
    if config.enable_posthoc_evaluation:
        artifact_paths["posthoc_metrics"] = run_directory / "posthoc_metrics.json"
    created_at_utc = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    started_at_utc: Optional[str] = None
    selected_iteration: Optional[int] = None
    manifest_kwargs = {
        "config": config,
        "run_id": run_id,
        "run_directory": run_directory,
        "dataset_identity": dataset_identity,
        "checkpoint_identity": checkpoint_identity,
        "source_identities": source_identities,
        "case": runtime.case,
        "checkpoint": runtime.checkpoint,
        "created_at_utc": created_at_utc,
        "artifact_paths": artifact_paths,
    }
    try:
        write_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="prepared",
                started_at_utc=None,
                finished_at_utc=None,
                failure_type=None,
                failure_message=None,
                selected_iteration=None,
            ),
        )
        started_at_utc = _utc_now()
        update_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="running",
                started_at_utc=started_at_utc,
                finished_at_utc=None,
                failure_type=None,
                failure_message=None,
                selected_iteration=None,
            ),
        )
        initial_interface_state = runtime.physics_prior.detach().clone()
        initial_physical = interface_normalized_to_physical(runtime, initial_interface_state).detach().cpu()
        _atomic_torch_save(
            artifact_paths["initial_interface_state"],
            {
                "normalized": initial_interface_state.detach().cpu(),
                "physical": initial_physical,
                "method_name": METHOD_NAME,
                "source": "quasi_1d_physics_initialization",
            },
        )
        optimized_interface_state, history, selected_iteration = optimize_interface(runtime, initial_interface_state)
        optimized_physical = interface_normalized_to_physical(runtime, optimized_interface_state).detach().cpu()
        _atomic_torch_save(
            artifact_paths["optimized_interface_state"],
            {
                "normalized": optimized_interface_state.detach().cpu(),
                "physical": optimized_physical,
                "method_name": METHOD_NAME,
                "selected_iteration": selected_iteration,
                "selection_basis": "minimum_total_physics_objective",
            },
        )
        _atomic_write_history(artifact_paths["optimization_history"], history)
        final_prediction = generate_final_prediction(runtime, optimized_interface_state)
        _atomic_torch_save(
            artifact_paths["final_prediction"],
            {
                "normalized": final_prediction["normalized"],
                "physical": final_prediction["physical"],
                "method_name": METHOD_NAME,
                "interface_state_artifact": "optimized_interface_state.pt",
            },
        )
        reconstruction = reconstruct(runtime.case, final_prediction)
        _atomic_torch_save(artifact_paths["reconstruction"], reconstruction)
        if config.enable_posthoc_evaluation:
            target = load_posthoc_targets(dataset_path, runtime.case)
            prediction = final_prediction["physical"].numpy()
            mask = reconstruction["inside_geometry_mask"]
            metrics = compute_posthoc_metrics(prediction, target, mask)
            metrics.update(
                {
                    "method_name": METHOD_NAME,
                    "runtime_uses_cfd_truth": False,
                    "posthoc_evaluation_uses_cfd_truth": True,
                    "optimized_interface_state_unchanged": True,
                }
            )
            _atomic_write_json(artifact_paths["posthoc_metrics"], metrics)
        update_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="completed",
                started_at_utc=started_at_utc,
                finished_at_utc=_utc_now(),
                failure_type=None,
                failure_message=None,
                selected_iteration=selected_iteration,
            ),
        )
    except BaseException as exc:
        if (run_directory / "manifest.json").exists():
            try:
                update_manifest(
                    run_directory,
                    _build_manifest(
                        **manifest_kwargs,
                        status="failed",
                        started_at_utc=started_at_utc,
                        finished_at_utc=_utc_now(),
                        failure_type=type(exc).__name__,
                        failure_message=str(exc).splitlines()[0][:240] or "No exception message was provided.",
                        selected_iteration=selected_iteration,
                    ),
                )
            except Exception as manifest_error:
                print(
                    f"Warning: failed to publish failure manifest without masking {type(exc).__name__}: {manifest_error}",
                    file=sys.stderr,
                )
        raise
    return run_directory


def main(argv: Optional[Sequence[str]] = None) -> int:
    config, run_requested = parse_args(argv)
    if not run_requested:
        _print_resolved_config(config)
        return 0
    run_directory = run_unknown_interface(config)
    print(f"Unknown-interface run completed: {_manifest_path(run_directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
