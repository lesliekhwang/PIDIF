"""Run the first Python implementation of progressive diffusion distillation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pidiffusion.artifacts import update_manifest, write_manifest  # noqa: E402
from pidiffusion.data import (  # noqa: E402
    DiffusionCaseSplit,
    DiffusionCellDataset,
    FeatureNormalizer,
    build_case_split,
    collate_diffusion_batch,
    fit_train_normalizers,
    load_diffusion_dataset,
)
from pidiffusion.diffusion import (  # noqa: E402
    DiffusionSchedule,
    build_ddim_timesteps,
    build_integer_segment_schedule,
    build_linear_schedule,
    ddim_step,
    equivalent_epsilon_target,
    final_clean_projection,
)
from pidiffusion.model import PointSetDiffusionDenoiser  # noqa: E402
from pidiffusion.provenance import (  # noqa: E402
    file_identity,
    git_state,
    runtime_environment,
    sha256_file,
)

try:  # Reuse the canonical baseline constructor without importing data.
    from experiments.train_diffusion import build_diffusion_model  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from train_diffusion import build_diffusion_model  # type: ignore  # noqa: E402


DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_controlpoints.h5"
)
DEFAULT_BASELINE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "point_diffusion_deeponet_style_puv_baseline_long"
    / "checkpoints"
    / "point_diffusion_baseline_long_best.pt"
)
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results"
TRUSTED_CHECKPOINT_ROOT = REPOSITORY_ROOT / "channel_diffusion_dataset"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AMBIGUOUS_RUN_IDS = {"latest", "final", "new", "updated"}


@dataclass(frozen=True)
class ProgressiveDistillationConfig:
    """Typed configuration for one one-epoch progressive distillation stage."""

    stage: Optional[int] = None
    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    baseline_checkpoint: Path = field(
        default_factory=lambda: DEFAULT_BASELINE_CHECKPOINT
    )
    stage1_checkpoint: Optional[Path] = None
    stage2_checkpoint: Optional[Path] = None
    results_root: Path = field(default_factory=lambda: DEFAULT_RESULTS_ROOT)
    device: str = "cuda:1"
    run_id: Optional[str] = None
    global_seed: int = 42
    split_seed: int = 42
    diffusion_steps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    teacher_sampling_steps: int = 50
    student_sampling_steps: int = 5
    student_timesteps: tuple[int, ...] = (999, 749, 500, 250, 0)
    teacher_transition_counts: tuple[int, ...] = (12, 12, 12, 13)
    batch_size: int = 4
    num_query_points: int = 8192
    num_workers: int = 0
    epochs: int = 1
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    validation_noise_seed: int = 5000

    @property
    def learning_rate(self) -> float:
        return {1: 1.0e-5, 2: 3.0e-6, 3: 1.0e-6}.get(
            self.stage or 1, 1.0e-5
        )

    @property
    def noise_seed_start(self) -> int:
        return {1: 3000, 2: 4000, 3: 7000}.get(self.stage or 1, 3000)

    @property
    def trajectory_target_weights(self) -> tuple[float, ...]:
        if self.stage == 1:
            return (0.25, 0.25, 0.25, 0.25, 0.0)
        return (0.2375, 0.2375, 0.2375, 0.2375, 0.05)

    @property
    def trajectory_field_weights(self) -> tuple[float, ...]:
        return (1.0, 1.0, 1.0)

    @property
    def cfd_field_weights(self) -> tuple[float, ...]:
        return (1.0, 1.0, 1.0)

    @property
    def lambda_cfd(self) -> float:
        return 0.05 if self.stage == 3 else 0.0


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve(strict=False)


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _require_file(path: Path, role: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{role} is not a regular file: {path}")


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value.lower() in _AMBIGUOUS_RUN_IDS:
        raise ValueError(f"{name} cannot use ambiguous value: {value!r}")
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} contains unsafe characters: {value!r}; use letters, numbers, "
            "'.', '_' or '-'."
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_run_id(config: ProgressiveDistillationConfig) -> str:
    if config.run_id is not None:
        return _validate_component(config.run_id, "run_id")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_distill_progressive_stage{config.stage}"


def _print_resolved_config(config: ProgressiveDistillationConfig) -> None:
    values = asdict(config)
    def display_path(path: Optional[Path]) -> Optional[str]:
        return None if path is None else str(_resolve_repo_path(path))

    values.update(
        {
            "dataset_path": str(_resolve_repo_path(config.dataset_path)),
            "baseline_checkpoint": display_path(config.baseline_checkpoint),
            "stage1_checkpoint": display_path(config.stage1_checkpoint),
            "stage2_checkpoint": display_path(config.stage2_checkpoint),
            "results_root": str(_resolve_repo_path(config.results_root)),
            "learning_rate": config.learning_rate,
            "noise_seed_start": config.noise_seed_start,
            "trajectory_target_weights": config.trajectory_target_weights,
            "trajectory_field_weights": config.trajectory_field_weights,
            "cfd_field_weights": config.cfd_field_weights,
            "lambda_cfd": config.lambda_cfd,
        }
    )
    print(json.dumps(values, indent=2, sort_keys=True, default=str))


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[ProgressiveDistillationConfig, bool]:
    """Parse a side-effect-free preview configuration or a formal run request."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one progressive distillation stage. Without --run, only the "
            "resolved configuration is printed."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Enable data loading, checkpoint loading, artifact creation, and training.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3),
        help="Progressive stage; required together with --run.",
    )
    parser.add_argument("--device", default="cuda:1", help="Torch device, for example cuda:1 or cpu.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Existing HDF5 dataset path.")
    parser.add_argument(
        "--teacher-checkpoint",
        dest="baseline_checkpoint",
        default=str(DEFAULT_BASELINE_CHECKPOINT),
        help=(
            "Baseline diffusion checkpoint for the frozen teacher and Stage 1 initialization; "
            "loaded with weights_only=True."
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default=None,
        help=(
            "Current Python pipeline new-format Stage 1 initialization checkpoint for Stage 2. "
            "Historical notebook distillation checkpoints are unsupported."
        ),
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default=None,
        help=(
            "Current Python pipeline new-format Stage 2 initialization checkpoint for Stage 3. "
            "Historical notebook distillation checkpoints are unsupported."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory under which results/distill_progressive/<run_id> is created.",
    )
    parser.add_argument("--run-id", default=None, help="Optional unique filesystem-safe run identifier.")
    args = parser.parse_args(argv)
    if args.run and args.stage is None:
        parser.error("--stage is required when --run is supplied")
    config = ProgressiveDistillationConfig(
        stage=args.stage,
        dataset_path=Path(args.dataset),
        baseline_checkpoint=Path(args.baseline_checkpoint),
        stage1_checkpoint=Path(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        stage2_checkpoint=Path(args.stage2_checkpoint) if args.stage2_checkpoint else None,
        results_root=Path(args.results_root),
        device=args.device,
        run_id=args.run_id,
    )
    return config, bool(args.run)


def resolve_device(raw_device: str) -> torch.device:
    """Validate a requested device before reading HDF5 or checkpoint data."""

    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device: {raw_device!r}") from exc
    if device.type == "cuda":
        if device.index is None:
            raise ValueError(
                "CUDA device must include an explicit index, for example cuda:0"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {raw_device!r}, but CUDA is not available; use --device cpu"
            )
        count = int(torch.cuda.device_count())
        if device.index < 0 or device.index >= count:
            raise RuntimeError(
                f"Requested CUDA device index {device.index}, but only {count} device(s) are available"
            )
    elif device.type != "cpu":
        raise ValueError(f"Only CPU and explicitly indexed CUDA devices are supported, got {raw_device!r}")
    return device


def set_global_seed(seed: int) -> None:
    """Set global RNGs while preserving legacy global NumPy query sampling."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _trusted_checkpoint_path(path: str | Path, role: str) -> Path:
    resolved = _resolve_repo_path(path)
    try:
        resolved.relative_to(TRUSTED_CHECKPOINT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{role} must be an explicitly supplied file under {TRUSTED_CHECKPOINT_ROOT}"
        ) from exc
    _require_file(resolved, role)
    return resolved


def _stage_checkpoint_path(path: str | Path, role: str) -> Path:
    """Resolve an explicitly supplied current-stage checkpoint path."""

    resolved = _resolve_repo_path(path)
    _require_file(resolved, role)
    return resolved


def load_baseline_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load the trusted local baseline diffusion checkpoint."""

    resolved = _trusted_checkpoint_path(path, "Baseline diffusion checkpoint")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Baseline diffusion checkpoint must contain a mapping, "
            f"got {type(checkpoint).__name__}"
        )
    return checkpoint


def load_new_stage_checkpoint(path: str | Path, expected_stage: int) -> Mapping[str, Any]:
    """Load a current Python new-format checkpoint for one exact stage."""

    resolved = _stage_checkpoint_path(
        path, f"Current Python Stage {expected_stage} checkpoint"
    )
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Stage initialization checkpoint is not a current Python new-format "
            "checkpoint; historical notebook distillation checkpoints are unsupported."
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Current Python stage checkpoint must contain a mapping, "
            f"got {type(checkpoint).__name__}"
        )
    stored_stage = checkpoint.get("stage")
    if stored_stage is None:
        raise KeyError("Checkpoint is missing required field: stage")
    if int(stored_stage) != expected_stage:
        raise ValueError(
            f"Checkpoint stage {stored_stage!r} does not match requested stage {expected_stage}"
        )
    if checkpoint.get("schema_version") != "progressive_distillation_stage_v1":
        raise ValueError(
            "Stage initialization checkpoint is not a current Python new-format "
            "checkpoint; historical notebook distillation checkpoints are unsupported."
        )
    if checkpoint.get("stage_identity_source") != "explicit_new_checkpoint":
        raise ValueError(
            "Stage initialization checkpoint does not declare the current Python "
            "checkpoint identity source"
        )
    return checkpoint


def _require_field(checkpoint: Mapping[str, Any], name: str) -> Any:
    if name not in checkpoint:
        raise KeyError(f"Checkpoint is missing required field: {name}")
    return checkpoint[name]


def _as_list(value: Any, name: str) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Checkpoint field {name} must be a list-like value")
    return list(value)


def _compare_exact(name: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise ValueError(f"Checkpoint field {name!r} does not match the current protocol")


def _compare_numeric(name: str, expected: Any, actual: Any) -> None:
    try:
        expected_tensor = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
        actual_tensor = torch.as_tensor(actual, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"Checkpoint field {name!r} is not numeric") from exc
    if expected_tensor.shape != actual_tensor.shape or not torch.allclose(
        expected_tensor,
        actual_tensor,
        rtol=1.0e-6,
        atol=1.0e-8,
        equal_nan=False,
    ):
        raise ValueError(f"Checkpoint field {name!r} does not match the current protocol")


def _compare_state_dict(
    model: PointSetDiffusionDenoiser,
    state_dict: Any,
    field_name: str,
) -> None:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint field {field_name} must be a mapping")
    expected = model.state_dict()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected).difference(state_dict))
        unexpected = sorted(set(state_dict).difference(expected))
        raise ValueError(
            f"{field_name} keys do not match the canonical model; missing={missing}, unexpected={unexpected}"
        )
    expected_parameters = 0
    actual_parameters = 0
    for name, expected_tensor in expected.items():
        actual_tensor = state_dict[name]
        if not torch.is_tensor(actual_tensor):
            raise TypeError(f"{field_name}[{name!r}] must be a tensor")
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"{field_name}[{name!r}] shape {tuple(actual_tensor.shape)} does not match "
                f"{tuple(expected_tensor.shape)}"
            )
        expected_parameters += int(expected_tensor.numel())
        actual_parameters += int(actual_tensor.numel())
    if expected_parameters != actual_parameters:
        raise ValueError(f"{field_name} parameter count does not match the canonical model")


def _expected_diffusion_config(config: ProgressiveDistillationConfig) -> dict[str, Any]:
    return {
        "T": int(config.diffusion_steps),
        "beta_start": float(config.beta_start),
        "beta_end": float(config.beta_end),
    }


def validate_baseline_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    model: PointSetDiffusionDenoiser,
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    dataset_path: Path,
) -> dict[str, Any]:
    _compare_exact("model_config", dict(model_config), dict(_require_field(checkpoint, "model_config")))
    _compare_exact(
        "diffusion_config",
        dict(diffusion_config),
        dict(_require_field(checkpoint, "diffusion_config")),
    )
    for name, expected in (
        ("branch_channel_names", list(data["branch_channel_names"])),
        ("trunk_channel_names", list(data["trunk_channel_names"])),
        ("output_channel_names", list(data["output_channel_names"])),
        ("train_cases", list(split.train_cases)),
        ("val_cases", list(split.val_cases)),
        ("test_cases", list(split.test_cases)),
    ):
        _compare_exact(name, expected, _as_list(_require_field(checkpoint, name), name))
    stored_normalizer = _require_field(checkpoint, "y_normalizer")
    if not isinstance(stored_normalizer, Mapping):
        raise TypeError("Checkpoint y_normalizer must be a mapping")
    _compare_numeric(
        "y_normalizer.mean",
        normalizer.mean.detach().cpu(),
        _require_field(stored_normalizer, "mean"),
    )
    _compare_numeric(
        "y_normalizer.std",
        normalizer.std.detach().cpu(),
        _require_field(stored_normalizer, "std"),
    )
    _compare_numeric("local_aspect_mean", local_aspect_mean, _require_field(checkpoint, "local_aspect_mean"))
    _compare_numeric("local_aspect_std", local_aspect_std, _require_field(checkpoint, "local_aspect_std"))
    dataset_reference = Path(str(_require_field(checkpoint, "dataset_h5"))).expanduser().resolve(strict=False)
    if dataset_reference != dataset_path.resolve(strict=False):
        raise ValueError(
            f"Checkpoint dataset_h5 {dataset_reference} does not match {dataset_path.resolve(strict=False)}"
        )
    state_dict = _require_field(checkpoint, "model_state_dict")
    _compare_state_dict(model, state_dict, "model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    epoch = int(_require_field(checkpoint, "epoch"))
    val_loss = float(_require_field(checkpoint, "val_loss"))
    best_val_loss = float(
        checkpoint.get("best_val_loss", checkpoint.get("run_best_val_loss", float("nan")))
    )
    if not math.isfinite(val_loss) or not math.isfinite(best_val_loss):
        raise ValueError("Baseline checkpoint validation values must be finite")
    dataset_checksum = checkpoint.get("dataset_sha256")
    return {
        "path": _manifest_path(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": epoch,
        "validation_loss": val_loss,
        "best_validation_loss": best_val_loss,
        "dataset_path": _manifest_path(dataset_path),
        "dataset_sha256": dataset_checksum if isinstance(dataset_checksum, str) else None,
        "dataset_checksum_available": isinstance(dataset_checksum, str),
    }


def _split_indices_from_checkpoint(
    checkpoint: Mapping[str, Any],
    split: DiffusionCaseSplit,
) -> None:
    stored = _require_field(checkpoint, "split_indices")
    if not isinstance(stored, Mapping):
        raise TypeError("Checkpoint split_indices must be a mapping")
    expected = {
        "train_idx": list(split.train_indices),
        "val_idx": list(split.val_indices),
        "test_idx": list(split.test_indices),
    }
    for name, expected_indices in expected.items():
        actual = [
            int(value)
            for value in _as_list(_require_field(stored, name), f"split_indices.{name}")
        ]
        _compare_exact(f"split_indices.{name}", expected_indices, actual)


def _normalization_from_checkpoint(
    checkpoint: Mapping[str, Any],
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
) -> None:
    normalization = _require_field(checkpoint, "normalization")
    if not isinstance(normalization, Mapping):
        raise TypeError("Checkpoint normalization must be a mapping")
    _compare_numeric("normalization.target_mean", normalizer.mean.detach().cpu(), _require_field(normalization, "target_mean"))
    _compare_numeric("normalization.target_std", normalizer.std.detach().cpu(), _require_field(normalization, "target_std"))
    _compare_numeric("normalization.local_aspect_mean", local_aspect_mean, _require_field(normalization, "local_aspect_mean"))
    _compare_numeric("normalization.local_aspect_std", local_aspect_std, _require_field(normalization, "local_aspect_std"))


def _expected_segment_schedules(config: ProgressiveDistillationConfig, device: torch.device) -> list[torch.Tensor]:
    boundaries = tuple(int(value) for value in config.student_timesteps)
    if len(boundaries) != len(config.teacher_transition_counts) + 1:
        raise ValueError("Student timestep count must be one greater than segment count")
    if boundaries[0] != config.diffusion_steps - 1 or boundaries[-1] != 0:
        raise ValueError("Student timestep schedule must include the diffusion endpoints")
    if any(left <= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Student timestep schedule must be strictly descending")
    if sum(int(value) for value in config.teacher_transition_counts) != config.teacher_sampling_steps - 1:
        raise ValueError("Teacher transition counts must sum to teacher_sampling_steps - 1")
    return [
        build_integer_segment_schedule(
            boundaries[index],
            boundaries[index + 1],
            int(config.teacher_transition_counts[index]),
            device=device,
        )
        for index in range(len(config.teacher_transition_counts))
    ]


def _compare_schedule_list(name: str, actual: Any, expected: Sequence[torch.Tensor]) -> None:
    actual_list = _as_list(actual, name)
    if len(actual_list) != len(expected):
        raise ValueError(f"Checkpoint field {name!r} has the wrong number of segments")
    for index, (actual_segment, expected_segment) in enumerate(zip(actual_list, expected)):
        _compare_exact(
            f"{name}[{index}]",
            expected_segment.detach().cpu().tolist(),
            _as_list(actual_segment, f"{name}[{index}]"),
        )


def validate_stage_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    stage: int,
    model: PointSetDiffusionDenoiser,
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    dataset_identity: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    expected_initialization_path: Optional[Path],
    config: ProgressiveDistillationConfig,
    segment_schedules: Sequence[torch.Tensor],
) -> dict[str, Any]:
    _compare_exact(
        "schema_version",
        "progressive_distillation_stage_v1",
        _require_field(checkpoint, "schema_version"),
    )
    _compare_exact("stage", stage, _require_field(checkpoint, "stage"))
    stage_identity_source = _require_field(checkpoint, "stage_identity_source")
    _compare_exact("stage_identity_source", "explicit_new_checkpoint", stage_identity_source)
    _compare_exact("model_config", dict(model_config), dict(_require_field(checkpoint, "model_config")))
    _compare_exact("diffusion_config", dict(diffusion_config), dict(_require_field(checkpoint, "diffusion_config")))
    state_dict = _require_field(checkpoint, "student_model_state_dict")
    _compare_state_dict(model, state_dict, "student_model_state_dict")
    for name, expected in (
        ("branch_channel_names", list(data["branch_channel_names"])),
        ("trunk_channel_names", list(data["trunk_channel_names"])),
        ("output_channel_names", list(data["output_channel_names"])),
    ):
        _compare_exact(name, expected, _as_list(_require_field(checkpoint, name), name))
    stored_split = _require_field(checkpoint, "split_case_ids")
    if not isinstance(stored_split, Mapping):
        raise TypeError("Checkpoint split_case_ids must be a mapping")
    for name, expected in (
        ("train", list(split.train_cases)),
        ("val", list(split.val_cases)),
        ("test", list(split.test_cases)),
    ):
        _compare_exact(
            f"split_case_ids.{name}",
            expected,
            _as_list(_require_field(stored_split, name), f"split_case_ids.{name}"),
        )
    stored_dataset = _require_field(checkpoint, "dataset_identity")
    if not isinstance(stored_dataset, Mapping):
        raise TypeError("Checkpoint dataset_identity must be a mapping")
    for name in ("path", "resolved_path", "size_bytes", "sha256"):
        _compare_exact(
            f"dataset_identity.{name}",
            _require_field(dataset_identity, name),
            _require_field(stored_dataset, name),
        )
    _split_indices_from_checkpoint(checkpoint, split)
    _normalization_from_checkpoint(checkpoint, normalizer, local_aspect_mean, local_aspect_std)
    _compare_exact(
        "student_timesteps",
        list(config.student_timesteps),
        [int(value) for value in _as_list(_require_field(checkpoint, "student_timesteps"), "student_timesteps")],
    )
    _compare_schedule_list(
        "teacher_segment_schedules",
        _require_field(checkpoint, "teacher_segment_schedules"),
        segment_schedules,
    )
    _compare_exact(
        "student_sampling_steps",
        config.student_sampling_steps,
        int(_require_field(checkpoint, "student_sampling_steps")),
    )
    _compare_exact(
        "teacher_sampling_steps",
        config.teacher_sampling_steps,
        _require_field(checkpoint, "teacher_sampling_steps"),
    )
    _compare_exact(
        "teacher_transition_counts",
        list(config.teacher_transition_counts),
        [
            int(value)
            for value in _as_list(
                _require_field(checkpoint, "teacher_transition_counts"),
                "teacher_transition_counts",
            )
        ],
    )
    _compare_numeric(
        "trajectory_target_weights",
        list(config.trajectory_target_weights),
        _require_field(checkpoint, "trajectory_target_weights"),
    )
    _compare_numeric(
        "trajectory_field_weights",
        config.trajectory_field_weights,
        _require_field(checkpoint, "trajectory_field_weights"),
    )
    _compare_numeric(
        "cfd_field_weights",
        config.cfd_field_weights,
        _require_field(checkpoint, "cfd_field_weights"),
    )
    _compare_numeric("lambda_cfd", config.lambda_cfd, _require_field(checkpoint, "lambda_cfd"))
    _compare_exact(
        "query_sampling_protocol",
        "legacy_global_numpy",
        _require_field(checkpoint, "query_sampling_protocol"),
    )
    _compare_exact("global_seed", config.global_seed, _require_field(checkpoint, "global_seed"))
    _compare_exact("split_seed", config.split_seed, _require_field(checkpoint, "split_seed"))
    noise_seed_policy = _require_field(checkpoint, "noise_seed_policy")
    if not isinstance(noise_seed_policy, Mapping):
        raise TypeError("Checkpoint noise_seed_policy must be a mapping")
    _compare_exact("noise_seed_policy.kind", "per_batch_increment", _require_field(noise_seed_policy, "kind"))
    _compare_exact(
        "noise_seed_policy.training_seed_start",
        config.noise_seed_start,
        _require_field(noise_seed_policy, "training_seed_start"),
    )
    _compare_exact(
        "noise_seed_policy.validation_seed_start",
        config.validation_noise_seed,
        _require_field(noise_seed_policy, "validation_seed_start"),
    )
    _compare_exact("initialization_mode", "model_only", _require_field(checkpoint, "initialization_mode"))
    stored_baseline_identity = _require_field(checkpoint, "baseline_teacher_identity")
    if not isinstance(stored_baseline_identity, Mapping):
        raise TypeError("Checkpoint baseline_teacher_identity must be a mapping")
    for name in ("path", "sha256", "size_bytes"):
        _compare_exact(
            f"baseline_teacher_identity.{name}",
            _require_field(baseline_identity, name),
            _require_field(stored_baseline_identity, name),
        )
    stored_initialization_identity = _require_field(checkpoint, "initialization_checkpoint_identity")
    if not isinstance(stored_initialization_identity, Mapping):
        raise TypeError("Checkpoint initialization_checkpoint_identity must be a mapping")
    _compare_exact(
        "initialization_checkpoint_identity.initialization_mode",
        "model_only",
        _require_field(stored_initialization_identity, "initialization_mode"),
    )
    stored_initialization_path = _require_field(
        stored_initialization_identity, "path"
    )
    stored_initialization_sha256 = _require_field(
        stored_initialization_identity, "sha256"
    )
    if expected_initialization_path is not None:
        _compare_exact(
            "initialization_checkpoint_identity.path",
            _manifest_path(expected_initialization_path),
            stored_initialization_path,
        )
        _compare_exact(
            "initialization_checkpoint_identity.sha256",
            sha256_file(expected_initialization_path),
            stored_initialization_sha256,
        )
    elif (
        not isinstance(stored_initialization_path, str)
        or not stored_initialization_path
        or not isinstance(stored_initialization_sha256, str)
        or not stored_initialization_sha256
    ):
        raise ValueError(
            "Stage checkpoint initialization identity is incomplete and cannot be verified"
        )
    expected_source = "explicit_baseline_checkpoint" if stage == 1 else "explicit_new_checkpoint"
    _compare_exact(
        "initialization_checkpoint_identity.stage_identity_source",
        expected_source,
        _require_field(stored_initialization_identity, "stage_identity_source"),
    )
    if stage == 1:
        if "stage" in stored_initialization_identity:
            raise ValueError(
                "Stage 1 initialization identity must reference the baseline checkpoint"
            )
    else:
        _compare_exact(
            "initialization_checkpoint_identity.stage",
            stage - 1,
            _require_field(stored_initialization_identity, "stage"),
        )
    expected_objective = {
        "teacher_forcing": True,
        "autoregressive_student_rollout": stage == 3,
        "direct_cfd_supervision": stage == 3,
        "lambda_cfd": config.lambda_cfd,
    }
    stored_objective = _require_field(checkpoint, "objective")
    if not isinstance(stored_objective, Mapping):
        raise TypeError("Checkpoint objective must be a mapping")
    for name, expected in expected_objective.items():
        actual = _require_field(stored_objective, name)
        if isinstance(expected, float):
            _compare_numeric(f"objective.{name}", expected, actual)
        else:
            _compare_exact(f"objective.{name}", expected, actual)
    for name, expected in (
        ("teacher_forcing", True),
        ("autoregressive_student_rollout", stage == 3),
        ("direct_cfd_supervision", stage == 3),
    ):
        _compare_exact(name, expected, _require_field(checkpoint, name))
    optimizer_config = _require_field(checkpoint, "optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        raise TypeError("Checkpoint optimizer_config must be a mapping")
    _compare_exact("optimizer_config.name", "AdamW", _require_field(optimizer_config, "name"))
    for name, expected in (
        ("learning_rate", config.learning_rate),
        ("weight_decay", config.weight_decay),
        ("grad_clip", config.grad_clip),
    ):
        _compare_numeric(
            f"optimizer_config.{name}", expected, _require_field(optimizer_config, name)
        )
    _compare_exact("optimizer_config.scheduler", "none", _require_field(optimizer_config, "scheduler"))
    _compare_exact("rounding_policy", "round", _require_field(checkpoint, "rounding_policy"))
    global_schedule = build_ddim_timesteps(
        config.teacher_sampling_steps,
        config.diffusion_steps,
        device="cpu",
    )
    _compare_exact(
        "teacher_global_schedule",
        global_schedule.tolist(),
        [
            int(value)
            for value in _as_list(
                _require_field(checkpoint, "teacher_global_schedule"),
                "teacher_global_schedule",
            )
        ],
    )
    teacher_path = _require_field(checkpoint, "teacher_checkpoint_path")
    if _resolve_repo_path(teacher_path) != _resolve_repo_path(_require_field(baseline_identity, "absolute_path")):
        raise ValueError("Stage checkpoint teacher_checkpoint_path does not match the baseline checkpoint")
    initialization_checkpoint_path = _require_field(
        checkpoint, "initialization_checkpoint_path"
    )
    _compare_exact(
        "initialization_checkpoint_path",
        stored_initialization_path,
        initialization_checkpoint_path,
    )

    return {
        "stage": stage,
        "stage_identity_source": stage_identity_source,
        "path": _manifest_path(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(_require_field(checkpoint, "epoch")),
        "validation_loss": checkpoint.get("val_loss"),
        "best_validation_loss": checkpoint.get("best_val_loss"),
    }


def _build_data_loaders(
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    config: ProgressiveDistillationConfig,
) -> tuple[DataLoader, DataLoader]:
    if config.num_workers != 0:
        raise ValueError("num_workers must be 0 for the canonical legacy RNG path")
    train_dataset = DiffusionCellDataset(
        samples=data["samples"],
        sample_indices=split.train_indices,
        n_query_points=config.num_query_points,
        random_query=True,
        target_normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        branch_channel_names=data["branch_channel_names"],
    )
    val_dataset = DiffusionCellDataset(
        samples=data["samples"],
        sample_indices=split.val_indices,
        n_query_points=config.num_query_points,
        random_query=False,
        target_normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        branch_channel_names=data["branch_channel_names"],
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_diffusion_batch,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def build_teacher_and_student(
    *,
    config: ProgressiveDistillationConfig,
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    device: torch.device,
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    segment_schedules: Sequence[torch.Tensor],
    dataset_path: Path,
    dataset_identity: Mapping[str, Any],
) -> tuple[PointSetDiffusionDenoiser, PointSetDiffusionDenoiser, dict[str, Any], dict[str, Any]]:
    teacher, _ = build_diffusion_model(
        len(data["branch_channel_names"]),
        len(data["trunk_channel_names"]),
        len(data["output_channel_names"]),
    )
    baseline_path = _trusted_checkpoint_path(config.baseline_checkpoint, "Baseline checkpoint")
    baseline_checkpoint = load_baseline_checkpoint(baseline_path)
    baseline_identity = validate_baseline_checkpoint(
        baseline_checkpoint,
        baseline_path,
        model=teacher,
        model_config=model_config,
        diffusion_config=diffusion_config,
        data=data,
        split=split,
        normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        dataset_path=dataset_path,
    )
    baseline_identity["absolute_path"] = str(baseline_path)
    baseline_identity["resolved_path"] = str(baseline_path.resolve(strict=False))
    baseline_identity["size_bytes"] = baseline_path.stat().st_size

    student, _ = build_diffusion_model(
        len(data["branch_channel_names"]),
        len(data["trunk_channel_names"]),
        len(data["output_channel_names"]),
    )
    initialization_metadata: dict[str, Any] = {
        "initialization_mode": "model_only",
        "path": _manifest_path(baseline_path),
        "sha256": baseline_identity["sha256"],
        "stage_identity_source": "explicit_baseline_checkpoint",
    }
    if config.stage == 1:
        if config.stage1_checkpoint is not None or config.stage2_checkpoint is not None:
            raise ValueError("Stage 1 does not accept a stage initialization checkpoint")
        student.load_state_dict(teacher.state_dict(), strict=True)
    else:
        initialization_path = (
            config.stage1_checkpoint if config.stage == 2 else config.stage2_checkpoint
        )
        if initialization_path is None:
            raise ValueError(
                f"Stage {config.stage} requires a current Python Stage {config.stage - 1} checkpoint"
            )
        initialization_path = _stage_checkpoint_path(
            initialization_path, f"Stage {config.stage - 1} checkpoint"
        )
        initialization_checkpoint = load_new_stage_checkpoint(
            initialization_path, expected_stage=config.stage - 1
        )
        expected_source_path = (
            baseline_path if config.stage - 1 == 1 else config.stage1_checkpoint
        )
        stage_identity = validate_stage_checkpoint(
            initialization_checkpoint,
            initialization_path,
            stage=config.stage - 1,
            model=student,
            model_config=model_config,
            diffusion_config=diffusion_config,
            data=data,
            split=split,
            normalizer=normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            dataset_identity=dataset_identity,
            baseline_identity=baseline_identity,
            expected_initialization_path=expected_source_path,
            config=ProgressiveDistillationConfig(**{**asdict(config), "stage": config.stage - 1}),
            segment_schedules=segment_schedules,
        )
        student.load_state_dict(_require_field(initialization_checkpoint, "student_model_state_dict"), strict=True)
        initialization_metadata = {
            "initialization_mode": "model_only",
            "path": _manifest_path(initialization_path),
            "sha256": stage_identity["sha256"],
            "stage": config.stage - 1,
            "stage_identity_source": stage_identity["stage_identity_source"],
        }

    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    return teacher, student, baseline_identity, initialization_metadata


def _move_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    if len(batch) == 6:
        branch, query, target, query_batch_id, sample_idx, branch_mask = batch
    elif len(batch) == 5:
        branch, query, target, query_batch_id, sample_idx = batch
        branch_mask = None
    else:
        raise ValueError(f"Unexpected diffusion batch length: {len(batch)}")
    return (
        branch.to(device),
        query.to(device),
        target.to(device),
        query_batch_id.to(device),
        sample_idx,
        branch_mask.to(device) if branch_mask is not None else None,
    )


def _make_noise(shape: Sequence[int], device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype)


def _teacher_segment(
    teacher: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    x_start: torch.Tensor,
    segment_schedule: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    x_current = x_start.detach().clone()
    with torch.inference_mode():
        for index in range(int(segment_schedule.numel()) - 1):
            t_current = int(segment_schedule[index].item())
            t_next = int(segment_schedule[index + 1].item())
            t_query = torch.full(
                (query.shape[0],), t_current, device=query.device, dtype=torch.long
            )
            epsilon_pred = teacher(
                branch=branch,
                query=query,
                noisy_target=x_current,
                t_query=t_query,
                query_batch_id=query_batch_id,
                branch_mask=branch_mask,
            )
            x_current, _ = ddim_step(
                x_current,
                epsilon_pred,
                t_current,
                t_next,
                schedule.alphas_cumprod,
            )
            x_current = x_current.detach()
    return x_current


def generate_dynamic_distillation_targets(
    *,
    teacher: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    initial_noise: torch.Tensor,
    student_timesteps: Sequence[int],
    segment_schedules: Sequence[torch.Tensor],
    schedule: DiffusionSchedule,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Generate four segmented teacher endpoints plus the final clean target."""

    if len(student_timesteps) != len(segment_schedules) + 1:
        raise ValueError("Student boundaries and teacher segments have inconsistent lengths")
    teacher.eval()
    x_current = initial_noise.detach().clone()
    targets: list[dict[str, Any]] = []
    for jump_index, segment_schedule in enumerate(segment_schedules):
        t_start = int(student_timesteps[jump_index])
        t_end = int(student_timesteps[jump_index + 1])
        x_start = x_current.detach().clone()
        x_end = _teacher_segment(
            teacher,
            branch,
            query,
            query_batch_id,
            branch_mask,
            x_start,
            segment_schedule,
            schedule,
        )
        epsilon_target = equivalent_epsilon_target(
            x_start,
            x_end,
            t_start,
            t_end,
            schedule.alphas_cumprod,
        ).detach()
        targets.append(
            {
                "jump_index": jump_index,
                "target_type": "ddim_jump",
                "t_start": t_start,
                "t_end": t_end,
                "x_start": x_start,
                "x_end_teacher": x_end.detach(),
                "epsilon_target": epsilon_target,
                "n_teacher_transitions": int(segment_schedule.numel()) - 1,
            }
        )
        x_current = x_end.detach()

    x_t0_teacher = x_current.detach().clone()
    t_query_zero = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
    with torch.inference_mode():
        teacher_epsilon_t0 = teacher(
            branch=branch,
            query=query,
            noisy_target=x_t0_teacher,
            t_query=t_query_zero,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
        )
    teacher_clean_x0 = final_clean_projection(
        x_t0_teacher,
        teacher_epsilon_t0,
        0,
        schedule.alphas_cumprod,
    ).detach()
    targets.append(
        {
            "jump_index": len(segment_schedules),
            "target_type": "final_clean",
            "t_start": 0,
            "t_end": "clean",
            "x_start": x_t0_teacher,
            "x_end_teacher": teacher_clean_x0,
            "epsilon_target": teacher_epsilon_t0.detach(),
            "n_teacher_transitions": 1,
        }
    )
    return targets, teacher_clean_x0


def _predict_student_target(
    student: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    target_item: Mapping[str, Any],
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_start = target_item["x_start"]
    t_start = int(target_item["t_start"])
    t_query = torch.full((query.shape[0],), t_start, device=query.device, dtype=torch.long)
    epsilon_pred = student(
        branch=branch,
        query=query,
        noisy_target=x_start,
        t_query=t_query,
        query_batch_id=query_batch_id,
        branch_mask=branch_mask,
    )
    if target_item.get("target_type") == "final_clean":
        endpoint_pred = final_clean_projection(
            x_start, epsilon_pred, t_start, schedule.alphas_cumprod
        )
    else:
        endpoint_pred, _ = ddim_step(
            x_start,
            epsilon_pred,
            t_start,
            int(target_item["t_end"]),
            schedule.alphas_cumprod,
        )
    return epsilon_pred, endpoint_pred


def compute_corrected_trajectory_loss(
    *,
    student: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    corrected_targets: Sequence[Mapping[str, Any]],
    schedule: DiffusionSchedule,
    field_weights: Sequence[float],
    target_weights: Sequence[float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the teacher-forced five-evaluation trajectory objective."""

    if len(corrected_targets) != 5:
        raise ValueError("Corrected trajectory must contain exactly five targets")
    field_weight_tensor = torch.as_tensor(field_weights, device=query.device, dtype=query.dtype)
    target_weight_tensor = torch.as_tensor(target_weights, device=query.device, dtype=query.dtype)
    if field_weight_tensor.numel() != 3:
        raise ValueError("Trajectory field weights must contain three values")
    if target_weight_tensor.numel() != len(corrected_targets):
        raise ValueError("Trajectory target weights must match the target count")
    if torch.any(field_weight_tensor < 0) or torch.any(target_weight_tensor < 0):
        raise ValueError("Trajectory weights must be non-negative")
    if float(field_weight_tensor.sum()) <= 0.0 or float(target_weight_tensor.sum()) <= 0.0:
        raise ValueError("Trajectory weights must have positive sums")
    field_weight_tensor = field_weight_tensor / field_weight_tensor.sum()
    target_weight_tensor = target_weight_tensor / target_weight_tensor.sum()

    target_losses: list[torch.Tensor] = []
    epsilon_mse_fields: list[torch.Tensor] = []
    endpoint_mse_fields: list[torch.Tensor] = []
    for target_item in corrected_targets:
        epsilon_pred, endpoint_pred = _predict_student_target(
            student,
            branch,
            query,
            query_batch_id,
            branch_mask,
            target_item,
            schedule,
        )
        epsilon_target = torch.as_tensor(target_item["epsilon_target"], device=query.device, dtype=query.dtype).detach()
        endpoint_target = torch.as_tensor(target_item["x_end_teacher"], device=query.device, dtype=query.dtype).detach()
        epsilon_mse = torch.mean((epsilon_pred - epsilon_target) ** 2, dim=0)
        endpoint_mse = torch.mean((endpoint_pred - endpoint_target) ** 2, dim=0)
        epsilon_mse_fields.append(epsilon_mse)
        endpoint_mse_fields.append(endpoint_mse)
        target_losses.append(torch.sum(field_weight_tensor * epsilon_mse))
    target_losses_tensor = torch.stack(target_losses)
    total_loss = torch.sum(target_weight_tensor * target_losses_tensor)
    return total_loss, {
        "target_losses": target_losses_tensor.detach(),
        "epsilon_mse_fields": torch.stack(epsilon_mse_fields).detach(),
        "endpoint_mse_fields": torch.stack(endpoint_mse_fields).detach(),
    }


def differentiable_student_five_eval_rollout(
    *,
    student: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    initial_noise: torch.Tensor,
    student_timesteps: Sequence[int],
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run the five student evaluations without detaching the autograd path."""

    if len(student_timesteps) != 5:
        raise ValueError("The progressive student rollout requires five timesteps")
    x_current = initial_noise
    records: list[dict[str, Any]] = []
    for index in range(4):
        t_start = int(student_timesteps[index])
        t_end = int(student_timesteps[index + 1])
        t_query = torch.full((query.shape[0],), t_start, device=query.device, dtype=torch.long)
        epsilon_pred = student(
            branch=branch,
            query=query,
            noisy_target=x_current,
            t_query=t_query,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
        )
        x_next, predicted_x0 = ddim_step(
            x_current,
            epsilon_pred,
            t_start,
            t_end,
            schedule.alphas_cumprod,
        )
        records.append(
            {
                "evaluation_index": index,
                "type": "ddim_jump",
                "t_start": t_start,
                "t_end": t_end,
                "x_start": x_current,
                "epsilon_prediction": epsilon_pred,
                "predicted_x0": predicted_x0,
                "x_end": x_next,
            }
        )
        x_current = x_next
    t_query_zero = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
    epsilon_zero = student(
        branch=branch,
        query=query,
        noisy_target=x_current,
        t_query=t_query_zero,
        query_batch_id=query_batch_id,
        branch_mask=branch_mask,
    )
    clean_prediction = final_clean_projection(
        x_current,
        epsilon_zero,
        0,
        schedule.alphas_cumprod,
    )
    records.append(
        {
            "evaluation_index": 4,
            "type": "final_clean",
            "t_start": 0,
            "t_end": "clean",
            "x_start": x_current,
            "epsilon_prediction": epsilon_zero,
            "x_end": clean_prediction,
        }
    )
    return clean_prediction, records


def compute_direct_cfd_rollout_loss(
    clean_prediction: torch.Tensor,
    cfd_target_norm: torch.Tensor,
    field_weights: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized direct CFD field supervision for Stage 3."""

    if clean_prediction.shape != cfd_target_norm.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(clean_prediction.shape)} vs {tuple(cfd_target_norm.shape)}"
        )
    weights = torch.as_tensor(field_weights, device=clean_prediction.device, dtype=clean_prediction.dtype)
    if weights.numel() != clean_prediction.shape[1] or torch.any(weights < 0) or float(weights.sum()) <= 0.0:
        raise ValueError("CFD field weights must be non-negative and match the output width")
    per_field_mse = torch.mean((clean_prediction - cfd_target_norm) ** 2, dim=0)
    return torch.sum((weights / weights.sum()) * per_field_mse), per_field_mse


@torch.no_grad()
def evaluate_distillation_loss(
    *,
    stage: int,
    teacher: PointSetDiffusionDenoiser,
    student: PointSetDiffusionDenoiser,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    config: ProgressiveDistillationConfig,
    segment_schedules: Sequence[torch.Tensor],
    device: torch.device,
) -> float:
    """Evaluate fixed-noise validation trajectory loss for checkpoint selection."""

    teacher.eval()
    student.eval()
    total = 0.0
    count = 0
    stage_config = ProgressiveDistillationConfig(**{**asdict(config), "stage": stage})
    for batch_index, batch in enumerate(loader):
        branch, query, target, query_batch_id, _, branch_mask = _move_batch(batch, device)
        initial_noise = _make_noise(
            target.shape,
            device,
            target.dtype,
            config.validation_noise_seed + batch_index,
        )
        dynamic_targets, _ = generate_dynamic_distillation_targets(
            teacher=teacher,
            branch=branch,
            query=query,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
            initial_noise=initial_noise,
            student_timesteps=stage_config.student_timesteps,
            segment_schedules=segment_schedules,
            schedule=schedule,
        )
        trajectory_loss, _ = compute_corrected_trajectory_loss(
            student=student,
            branch=branch,
            query=query,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
            corrected_targets=dynamic_targets,
            schedule=schedule,
            field_weights=stage_config.trajectory_field_weights,
            target_weights=stage_config.trajectory_target_weights,
        )
        value = trajectory_loss
        if stage == 3:
            clean_prediction, _ = differentiable_student_five_eval_rollout(
                student=student,
                branch=branch,
                query=query,
                query_batch_id=query_batch_id,
                branch_mask=branch_mask,
                initial_noise=initial_noise,
                student_timesteps=stage_config.student_timesteps,
                schedule=schedule,
            )
            cfd_loss, _ = compute_direct_cfd_rollout_loss(
                clean_prediction,
                target,
                stage_config.cfd_field_weights,
            )
            value = value + stage_config.lambda_cfd * cfd_loss
        if not torch.isfinite(value):
            raise RuntimeError(f"Non-finite validation loss at batch {batch_index}")
        batch_size = int(branch.shape[0])
        total += float(value.item()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return total / count


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Optional[Path] = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def save_stage_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one new stage checkpoint inside the current run."""

    _atomic_torch_save(path, payload)


def _atomic_write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "stage",
        "epoch",
        "global_step",
        "batch_index",
        "train_loss",
        "trajectory_loss",
        "cfd_loss",
        "weighted_cfd_loss",
        "validation_loss",
        "gradient_norm",
        "learning_rate",
    ]
    temporary_path: Optional[Path] = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _identity(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity["path"] = _manifest_path(path)
    return identity


def _checkpoint_payload(
    *,
    stage: int,
    epoch: int,
    student: PointSetDiffusionDenoiser,
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    config: ProgressiveDistillationConfig,
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    dataset_identity: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    initialization_metadata: Mapping[str, Any],
    global_schedule: torch.Tensor,
    segment_schedules: Sequence[torch.Tensor],
    history: Sequence[Mapping[str, Any]],
    best_validation_loss: Optional[float],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    objective = {
        "teacher_forcing": True,
        "autoregressive_student_rollout": stage == 3,
        "direct_cfd_supervision": stage == 3,
        "lambda_cfd": config.lambda_cfd,
    }
    git = git_state(REPOSITORY_ROOT)
    return {
        "schema_version": "progressive_distillation_stage_v1",
        "stage": stage,
        "stage_identity_source": "explicit_new_checkpoint",
        "epoch": int(epoch),
        "student_model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in student.state_dict().items()
        },
        "model_config": dict(model_config),
        "diffusion_config": dict(diffusion_config),
        "branch_channel_names": list(data["branch_channel_names"]),
        "trunk_channel_names": list(data["trunk_channel_names"]),
        "output_channel_names": list(data["output_channel_names"]),
        "split_case_ids": {
            "train": list(split.train_cases),
            "val": list(split.val_cases),
            "test": list(split.test_cases),
        },
        "split_indices": {
            "train_idx": list(split.train_indices),
            "val_idx": list(split.val_indices),
            "test_idx": list(split.test_indices),
        },
        "normalization": {
            "target_mean": normalizer.mean.detach().cpu().clone(),
            "target_std": normalizer.std.detach().cpu().clone(),
            "local_aspect_mean": float(local_aspect_mean),
            "local_aspect_std": float(local_aspect_std),
        },
        "student_sampling_steps": int(config.student_sampling_steps),
        "student_timesteps": list(config.student_timesteps),
        "teacher_sampling_steps": int(config.teacher_sampling_steps),
        "teacher_global_schedule": global_schedule.detach().cpu().tolist(),
        "teacher_segment_schedules": [schedule.detach().cpu().tolist() for schedule in segment_schedules],
        "teacher_transition_counts": list(config.teacher_transition_counts),
        "rounding_policy": "round",
        "trajectory_field_weights": list(config.trajectory_field_weights),
        "trajectory_target_weights": list(config.trajectory_target_weights),
        "cfd_field_weights": list(config.cfd_field_weights),
        "lambda_cfd": config.lambda_cfd,
        "teacher_forcing": True,
        "autoregressive_student_rollout": stage == 3,
        "direct_cfd_supervision": stage == 3,
        "objective": objective,
        "baseline_teacher_identity": dict(baseline_identity),
        "initialization_checkpoint_identity": dict(initialization_metadata),
        "initialization_mode": "model_only",
        "teacher_checkpoint_path": baseline_identity["absolute_path"],
        "initialization_checkpoint_path": initialization_metadata.get("path"),
        "query_sampling_protocol": "legacy_global_numpy",
        "global_seed": config.global_seed,
        "split_seed": config.split_seed,
        "noise_seed_policy": {
            "kind": "per_batch_increment",
            "training_seed_start": config.noise_seed_start,
            "validation_seed_start": config.validation_noise_seed,
        },
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "scheduler": "none",
        },
        "optimizer_state_saved": False,
        "training_history": [dict(row) for row in history],
        "current_run_best_validation_loss": best_validation_loss,
        "dataset_identity": dict(dataset_identity),
        "git": git,
        "source_files": {
            "data_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "data.py"),
            "model_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "model.py"),
        "diffusion_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "diffusion.py"),
        "entrypoint": _identity(REPOSITORY_ROOT / "experiments" / "distill_progressive.py"),
        "protocol_notebook": _identity(
            REPOSITORY_ROOT / "train_point_diffusion_progressive_distillation.ipynb"
        ),
        },
        "optimizer_state_dict": None,
        "stage_learning_rate": config.learning_rate,
        "training_config": {
            "epochs": config.epochs,
            "stage": stage,
            "noise_seed_start": config.noise_seed_start,
            "teacher_forced_trajectory_loss": True,
            "autoregressive_student_rollout": stage == 3,
            "direct_cfd_supervision": stage == 3,
        },
        "optimizer_identity": type(optimizer).__name__,
    }


def train_stage(
    *,
    stage: int,
    teacher: PointSetDiffusionDenoiser,
    student: PointSetDiffusionDenoiser,
    train_loader: DataLoader,
    val_loader: DataLoader,
    schedule: DiffusionSchedule,
    segment_schedules: Sequence[torch.Tensor],
    global_schedule: torch.Tensor,
    config: ProgressiveDistillationConfig,
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    dataset_identity: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    initialization_metadata: Mapping[str, Any],
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    run_directory: Path,
) -> tuple[Path, Optional[Path], list[dict[str, Any]], Optional[float]]:
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history_path = run_directory / "training_history.csv"
    latest_path = run_directory / f"distill_stage{stage}_latest.pt"
    best_path = run_directory / f"distill_stage{stage}_best.pt"
    history: list[dict[str, Any]] = []
    best_validation_loss: Optional[float] = None
    global_batch_index = 0
    device = next(student.parameters()).device
    stage_config = ProgressiveDistillationConfig(**{**asdict(config), "stage": stage})

    for epoch in range(1, config.epochs + 1):
        student.train()
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            branch, query, target, query_batch_id, _, branch_mask = _move_batch(batch, device)
            initial_noise = _make_noise(
                target.shape,
                device,
                target.dtype,
                stage_config.noise_seed_start + global_batch_index,
            )
            dynamic_targets, _ = generate_dynamic_distillation_targets(
                teacher=teacher,
                branch=branch,
                query=query,
                query_batch_id=query_batch_id,
                branch_mask=branch_mask,
                initial_noise=initial_noise,
                student_timesteps=stage_config.student_timesteps,
                segment_schedules=segment_schedules,
                schedule=schedule,
            )
            optimizer.zero_grad(set_to_none=True)
            trajectory_loss, metrics = compute_corrected_trajectory_loss(
                student=student,
                branch=branch,
                query=query,
                query_batch_id=query_batch_id,
                branch_mask=branch_mask,
                corrected_targets=dynamic_targets,
                schedule=schedule,
                field_weights=stage_config.trajectory_field_weights,
                target_weights=stage_config.trajectory_target_weights,
            )
            cfd_loss = trajectory_loss.new_zeros(())
            if stage == 3:
                clean_prediction, _ = differentiable_student_five_eval_rollout(
                    student=student,
                    branch=branch,
                    query=query,
                    query_batch_id=query_batch_id,
                    branch_mask=branch_mask,
                    initial_noise=initial_noise,
                    student_timesteps=stage_config.student_timesteps,
                    schedule=schedule,
                )
                cfd_loss, _ = compute_direct_cfd_rollout_loss(
                    clean_prediction,
                    target,
                    stage_config.cfd_field_weights,
                )
            weighted_cfd_loss = stage_config.lambda_cfd * cfd_loss
            total_loss = trajectory_loss + weighted_cfd_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=config.grad_clip
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            row = {
                "stage": stage,
                "epoch": epoch,
                "global_step": global_batch_index + 1,
                "batch_index": batch_index,
                "train_loss": float(total_loss.detach().item()),
                "trajectory_loss": float(trajectory_loss.detach().item()),
                "cfd_loss": float(cfd_loss.detach().item()),
                "weighted_cfd_loss": float(weighted_cfd_loss.detach().item()),
                "validation_loss": None,
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "target_losses": metrics["target_losses"].detach().cpu().tolist(),
            }
            history.append(row)
            epoch_losses.append(row["train_loss"])
            global_batch_index += 1

        validation_loss = evaluate_distillation_loss(
            stage=stage,
            teacher=teacher,
            student=student,
            loader=val_loader,
            schedule=schedule,
            config=stage_config,
            segment_schedules=segment_schedules,
            device=device,
        )
        epoch_row = {
            "stage": stage,
            "epoch": epoch,
            "global_step": global_batch_index,
            "batch_index": None,
            "train_loss": float(np.mean(epoch_losses)),
            "trajectory_loss": float(np.mean([row["trajectory_loss"] for row in history if row["epoch"] == epoch])),
            "cfd_loss": float(np.mean([row["cfd_loss"] for row in history if row["epoch"] == epoch])),
            "weighted_cfd_loss": float(np.mean([row["weighted_cfd_loss"] for row in history if row["epoch"] == epoch])),
            "validation_loss": float(validation_loss),
            "gradient_norm": float(np.mean([row["gradient_norm"] for row in history if row["epoch"] == epoch])),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        if not math.isfinite(validation_loss):
            validation_loss = float("nan")
        is_best = math.isfinite(validation_loss) and (
            best_validation_loss is None or validation_loss < best_validation_loss
        )
        if is_best:
            best_validation_loss = float(validation_loss)
        payload = _checkpoint_payload(
            stage=stage,
            epoch=epoch,
            student=student,
            model_config=model_config,
            diffusion_config=diffusion_config,
            config=stage_config,
            data=data,
            split=split,
            normalizer=normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            dataset_identity=dataset_identity,
            baseline_identity=baseline_identity,
            initialization_metadata=initialization_metadata,
            global_schedule=global_schedule,
            segment_schedules=segment_schedules,
            history=history,
            best_validation_loss=best_validation_loss,
            optimizer=optimizer,
        )
        _atomic_write_history(history_path, history)
        save_stage_checkpoint(latest_path, payload)
        if is_best:
            save_stage_checkpoint(best_path, payload)

    return latest_path, best_path if best_validation_loss is not None else None, history, best_validation_loss


def _build_manifest(
    *,
    config: ProgressiveDistillationConfig,
    run_id: str,
    run_directory: Path,
    status: str,
    created_at_utc: str,
    started_at_utc: Optional[str],
    finished_at_utc: Optional[str],
    last_completed_epoch: Optional[int],
    failure_type: Optional[str],
    failure_message: Optional[str],
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    dataset_identity: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    initialization_metadata: Mapping[str, Any],
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    global_schedule: torch.Tensor,
    segment_schedules: Sequence[torch.Tensor],
    best_validation_loss: Optional[float],
) -> dict[str, Any]:
    source_files = {
        "data_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "data.py"),
        "model_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "model.py"),
        "diffusion_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "diffusion.py"),
        "provenance_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "provenance.py"),
        "artifacts_module": _identity(REPOSITORY_ROOT / "pidiffusion" / "artifacts.py"),
        "entrypoint": _identity(REPOSITORY_ROOT / "experiments" / "distill_progressive.py"),
        "protocol_notebook": _identity(
            REPOSITORY_ROOT / "train_point_diffusion_progressive_distillation.ipynb"
        ),
    }
    latest_path = run_directory / f"distill_stage{config.stage}_latest.pt"
    best_path = run_directory / f"distill_stage{config.stage}_best.pt"
    history_path = run_directory / "training_history.csv"
    return {
        "schema_version": "distill_progressive_v1",
        "run_id": run_id,
        "timestamp_utc": finished_at_utc or created_at_utc,
        "status": status,
        "created_at_utc": created_at_utc,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "last_completed_epoch": last_completed_epoch,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "git": git_state(REPOSITORY_ROOT),
        "source_files": source_files,
        "dataset": {
            **dict(dataset_identity),
            "split": {
                "method": "sorted_case_ids_default_rng",
                "seed": config.split_seed,
                "train_fraction": 0.8,
                "validation_fraction": 0.1,
                "train_cases": list(split.train_cases),
                "val_cases": list(split.val_cases),
                "test_cases": list(split.test_cases),
                "train_indices": list(split.train_indices),
                "val_indices": list(split.val_indices),
                "test_indices": list(split.test_indices),
            },
            "normalizer_source": "train_only",
            "target_mean": normalizer.mean.detach().cpu().tolist(),
            "target_std": normalizer.std.detach().cpu().tolist(),
            "local_aspect_mean": local_aspect_mean,
            "local_aspect_std": local_aspect_std,
            "channel_names": {
                "branch": list(data["branch_channel_names"]),
                "query": list(data["trunk_channel_names"]),
                "target": list(data["output_channel_names"]),
            },
        },
        "checkpoint": {
            "stage": config.stage,
            "initialization_mode": "model_only",
            "baseline_teacher": dict(baseline_identity),
            "initialization": dict(initialization_metadata),
            "model_config": dict(model_config),
            "diffusion_config": dict(diffusion_config),
            "current_run_best_validation_loss": best_validation_loss,
            "best": _identity(best_path) if best_path.exists() else None,
            "latest": _identity(latest_path) if latest_path.exists() else None,
        },
        "protocol": {
            "name": "progressive_distillation_round_reverse",
            "stage": config.stage,
            "teacher_sampling_steps": config.teacher_sampling_steps,
            "student_sampling_steps": config.student_sampling_steps,
            "student_timesteps": list(config.student_timesteps),
            "teacher_global_schedule": global_schedule.detach().cpu().tolist(),
            "teacher_segment_schedules": [schedule.detach().cpu().tolist() for schedule in segment_schedules],
            "teacher_transition_counts": list(config.teacher_transition_counts),
            "rounding_policy": "round",
            "trajectory_field_weights": list(config.trajectory_field_weights),
            "trajectory_target_weights": list(config.trajectory_target_weights),
            "cfd_field_weights": list(config.cfd_field_weights),
            "lambda_cfd": config.lambda_cfd,
            "teacher_forcing": True,
            "autoregressive_student_rollout": config.stage == 3,
            "direct_cfd_supervision": config.stage == 3,
            "query_sampling_protocol": "legacy_global_numpy",
            "validation_protocol": "dynamic_teacher_targets_with_fixed_noise",
            "full_resume": False,
        },
        "randomness": {
            "global_seed": config.global_seed,
            "split_seed": config.split_seed,
            "query_sampling": "legacy_global_numpy",
            "training_noise_seed_start": config.noise_seed_start,
            "validation_noise_seed_start": config.validation_noise_seed,
            "noise_seed_policy": "per_batch_increment",
            "dataloader_workers": config.num_workers,
        },
        "environment": runtime_environment(),
        "outputs": {
            "directory": _manifest_path(run_directory),
            "best_checkpoint": _identity(best_path) if best_path.exists() else None,
            "latest_checkpoint": _identity(latest_path) if latest_path.exists() else None,
            "training_history": _identity(history_path) if history_path.exists() else None,
        },
        "notes": [
            "This entrypoint implements only the approved progressive distillation protocol.",
            "Teacher global 50-step and segmented teacher-target schedules are recorded separately.",
            "Stage continuation is model-only; optimizer, scheduler, and RNG states are not restored.",
            "No sampling, reconstruction, plotting, test metrics, or unknown-interface inference is performed by this script.",
        ],
    }


def run_training(config: ProgressiveDistillationConfig) -> Path:
    """Load canonical data, train one approved stage, and publish isolated artifacts."""

    if config.stage not in (1, 2, 3):
        raise ValueError("A stage of 1, 2, or 3 is required for a formal run")
    device = resolve_device(config.device)
    dataset_path = _resolve_repo_path(config.dataset_path)
    baseline_path = _trusted_checkpoint_path(config.baseline_checkpoint, "Baseline checkpoint")
    if config.stage == 1 and (
        config.stage1_checkpoint is not None or config.stage2_checkpoint is not None
    ):
        raise ValueError("Stage 1 does not accept a stage initialization checkpoint")
    if config.stage == 2:
        if config.stage1_checkpoint is None:
            raise ValueError("Stage 2 requires --stage1-checkpoint")
        _stage_checkpoint_path(config.stage1_checkpoint, "Stage 1 checkpoint")
    if config.stage == 3:
        if config.stage2_checkpoint is None:
            raise ValueError("Stage 3 requires --stage2-checkpoint")
        _stage_checkpoint_path(config.stage2_checkpoint, "Stage 2 checkpoint")
    _require_file(dataset_path, "Dataset")
    if config.batch_size <= 0 or config.num_query_points <= 0 or config.epochs <= 0:
        raise ValueError("batch_size, num_query_points, and epochs must be positive")
    set_global_seed(config.global_seed)

    data = load_diffusion_dataset(dataset_path)
    split = build_case_split(data, split_seed=config.split_seed)
    normalizer, local_aspect_mean, local_aspect_std = fit_train_normalizers(data, split)
    dataset_identity = _identity(dataset_path)
    dataset_identity["checksum_available"] = dataset_identity.get("sha256") is not None
    dataset_identity["checksum_source"] = (
        "computed_from_current_dataset"
        if dataset_identity.get("sha256")
        else "unavailable"
    )
    train_loader, val_loader = _build_data_loaders(
        data,
        split,
        normalizer,
        local_aspect_mean,
        local_aspect_std,
        config,
    )
    model_probe, model_config = build_diffusion_model(
        len(data["branch_channel_names"]),
        len(data["trunk_channel_names"]),
        len(data["output_channel_names"]),
    )
    del model_probe
    schedule = build_linear_schedule(
        timesteps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    diffusion_config = _expected_diffusion_config(config)
    global_schedule = build_ddim_timesteps(
        config.teacher_sampling_steps,
        config.diffusion_steps,
        device=device,
    )
    segment_schedules = _expected_segment_schedules(config, device)
    teacher, student, baseline_identity, initialization_metadata = build_teacher_and_student(
        config=config,
        data=data,
        split=split,
        normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        device=device,
        model_config=model_config,
        diffusion_config=diffusion_config,
        segment_schedules=segment_schedules,
        dataset_path=dataset_path,
        dataset_identity=dataset_identity,
    )
    run_id = _build_run_id(config)
    results_root = _resolve_repo_path(config.results_root)
    run_directory = results_root / "distill_progressive" / run_id
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(exist_ok=False)
    created_at_utc = _utc_now()
    started_at_utc: Optional[str] = None
    last_completed_epoch: Optional[int] = None
    best_validation_loss: Optional[float] = None
    manifest_kwargs = {
        "config": config,
        "run_id": run_id,
        "run_directory": run_directory,
        "data": data,
        "split": split,
        "normalizer": normalizer,
        "local_aspect_mean": local_aspect_mean,
        "local_aspect_std": local_aspect_std,
        "dataset_identity": dataset_identity,
        "baseline_identity": baseline_identity,
        "initialization_metadata": initialization_metadata,
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "global_schedule": global_schedule,
        "segment_schedules": segment_schedules,
    }
    try:
        write_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="prepared",
                created_at_utc=created_at_utc,
                started_at_utc=None,
                finished_at_utc=None,
                last_completed_epoch=None,
                failure_type=None,
                failure_message=None,
                best_validation_loss=None,
            ),
        )
        started_at_utc = _utc_now()
        update_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="running",
                created_at_utc=created_at_utc,
                started_at_utc=started_at_utc,
                finished_at_utc=None,
                last_completed_epoch=None,
                failure_type=None,
                failure_message=None,
                best_validation_loss=None,
            ),
        )
        latest_path, best_path, history, best_validation_loss = train_stage(
            stage=config.stage,
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            schedule=schedule,
            segment_schedules=segment_schedules,
            global_schedule=global_schedule,
            config=config,
            data=data,
            split=split,
            normalizer=normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            dataset_identity=dataset_identity,
            baseline_identity=baseline_identity,
            initialization_metadata=initialization_metadata,
            model_config=model_config,
            diffusion_config=diffusion_config,
            run_directory=run_directory,
        )
        del latest_path, best_path, history
        last_completed_epoch = config.epochs
        update_manifest(
            run_directory,
            _build_manifest(
                **manifest_kwargs,
                status="completed",
                created_at_utc=created_at_utc,
                started_at_utc=started_at_utc,
                finished_at_utc=_utc_now(),
                last_completed_epoch=last_completed_epoch,
                failure_type=None,
                failure_message=None,
                best_validation_loss=best_validation_loss,
            ),
        )
    except BaseException as exc:
        if (run_directory / "manifest.json").exists():
            try:
                failure_type = type(exc).__name__
                failure_message = str(exc).splitlines()[0][:240] or "No exception message was provided."
                update_manifest(
                    run_directory,
                    _build_manifest(
                        **manifest_kwargs,
                        status="failed",
                        created_at_utc=created_at_utc,
                        started_at_utc=started_at_utc,
                        finished_at_utc=_utc_now(),
                        last_completed_epoch=last_completed_epoch,
                        failure_type=failure_type,
                        failure_message=failure_message,
                        best_validation_loss=best_validation_loss,
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
    run_directory = run_training(config)
    print(f"Progressive distillation run completed: {_manifest_path(run_directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
