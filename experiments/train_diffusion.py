"""Train the baseline point-set diffusion model."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

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
    build_linear_schedule,
    epsilon_prediction_loss,
)
from pidiffusion.model import PointSetDiffusionDenoiser  # noqa: E402
from pidiffusion.provenance import (  # noqa: E402
    file_identity,
    git_state,
    runtime_environment,
    sha256_file,
)


DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_controlpoints.h5"
)
DEFAULT_SOURCE_NOTEBOOK = (
    REPOSITORY_ROOT / "train_domain_channel_diffusion_iterative_prediction.ipynb"
)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AMBIGUOUS_RUN_IDS = {"latest", "final", "new", "updated"}


@dataclass
class DiffusionTrainingConfig:
    """Small typed configuration for one baseline diffusion run."""

    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    results_root: Path = field(default_factory=lambda: REPOSITORY_ROOT / "results")
    source_notebook: Path = field(default_factory=lambda: DEFAULT_SOURCE_NOTEBOOK)
    device: str = "cuda:1"
    global_seed: Optional[int] = None
    split_seed: int = 42
    batch_size: int = 4
    epochs: int = 250
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    scheduler_t_max: int = 250
    scheduler_eta_min: float = 1.0e-5
    use_scheduler: bool = True
    num_query_points: int = 8192
    num_workers: int = 0
    checkpoint_interval: int = 1
    validation_interval: int = 1
    initialize_from_checkpoint: Optional[Path] = None
    checkpoint_tag: str = "baseline"
    run_id: Optional[str] = None
    timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve(strict=False)


def _require_file(path: Path, role: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{role} is not a regular file: {path}")


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity_for_manifest(
    path: Path,
    *,
    include_checksum: bool = True,
) -> dict[str, Any]:
    if include_checksum:
        identity = file_identity(path)
    else:
        resolved = path.expanduser().resolve(strict=False)
        identity = {
            "path": str(path),
            "resolved_path": str(resolved),
            "exists": resolved.exists(),
            "size_bytes": None,
            "mtime_utc": None,
            "sha256": None,
        }
        if resolved.exists() and resolved.is_file():
            stat = resolved.stat()
            identity["size_bytes"] = int(stat.st_size)
            identity["mtime_utc"] = (
                datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
    identity["path"] = _manifest_path(path)
    if not include_checksum:
        identity["sha256"] = None
    return identity


def _short_failure(exc: BaseException) -> tuple[str, str]:
    message = str(exc).splitlines()[0].strip()
    if not message:
        message = "No exception message was provided."
    return type(exc).__name__, message[:240]


def _optional_identity_for_manifest(
    path: Path,
    *,
    include_checksum: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _identity_for_manifest(path, include_checksum=include_checksum)


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


def _build_run_id(config: DiffusionTrainingConfig) -> str:
    if config.run_id is not None:
        return _validate_component(config.run_id, "run_id")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = _validate_component(config.checkpoint_tag, "checkpoint_tag")
    return f"{timestamp}_baseline_diffusion_{tag}_splitseed{config.split_seed}"


def set_global_seed(seed: Optional[int]) -> None:
    """Set optional global RNGs without replacing legacy query sampling."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(raw_device: str) -> torch.device:
    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device: {raw_device!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {raw_device!r}, but CUDA is not available; use --device cpu"
            )
        if device.index is None:
            raise ValueError(
                "CUDA device must include an explicit index, for example cuda:0"
            )
        device_count = torch.cuda.device_count()
        if device.index < 0 or device.index >= device_count:
            raise RuntimeError(
                f"Requested CUDA device index {device.index}, but only "
                f"{device_count} device(s) are available"
            )
    return device


def _model_config(
    branch_input_dim: int,
    query_input_dim: int,
    target_dim: int,
) -> dict[str, int]:
    return {
        "branch_input_dim": int(branch_input_dim),
        "query_input_dim": int(query_input_dim),
        "target_dim": int(target_dim),
        "latent_dim": 128,
        "time_dim": 128,
        "branch_point_hidden_dim": 128,
        "branch_global_hidden_dim": 128,
        "denoiser_hidden_dim": 256,
        "denoiser_depth": 4,
    }


def build_diffusion_model(
    branch_input_dim: int,
    query_input_dim: int,
    target_dim: int,
) -> tuple[PointSetDiffusionDenoiser, dict[str, int]]:
    """Instantiate the canonical model and its checkpoint configuration."""

    model_config = _model_config(
        branch_input_dim=branch_input_dim,
        query_input_dim=query_input_dim,
        target_dim=target_dim,
    )
    model = PointSetDiffusionDenoiser(**model_config)
    return model, model_config


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def _require_checkpoint_field(
    checkpoint: Mapping[str, Any],
    field_name: str,
) -> Any:
    if field_name not in checkpoint:
        raise KeyError(f"Checkpoint is missing required field: {field_name}")
    return checkpoint[field_name]


def _compare_exact_checkpoint_field(
    field_name: str,
    expected: Any,
    actual: Any,
) -> None:
    if actual != expected:
        raise ValueError(
            f"Checkpoint field {field_name!r} does not match the current protocol"
        )


def _compare_float_checkpoint_field(
    field_name: str,
    expected: Any,
    actual: Any,
) -> None:
    expected_tensor = torch.as_tensor(expected, dtype=torch.float32)
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32)
    if expected_tensor.shape != actual_tensor.shape or not torch.allclose(
        expected_tensor,
        actual_tensor,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            f"Checkpoint field {field_name!r} does not match the current protocol"
        )


def _validate_checkpoint_protocol(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_config: Mapping[str, int],
    expected_diffusion_config: Mapping[str, Any],
    expected_output_channel_names: list[str],
    expected_branch_channel_names: list[str],
    expected_trunk_channel_names: list[str],
    expected_split: DiffusionCaseSplit,
    expected_normalizer: FeatureNormalizer,
    expected_local_aspect_mean: float,
    expected_local_aspect_std: float,
) -> None:
    stored_model_config = _require_checkpoint_field(checkpoint, "model_config")
    if not isinstance(stored_model_config, Mapping):
        raise TypeError("Checkpoint model_config must be a mapping")
    _compare_exact_checkpoint_field(
        "model_config",
        dict(expected_model_config),
        dict(stored_model_config),
    )

    stored_diffusion_config = _require_checkpoint_field(
        checkpoint, "diffusion_config"
    )
    if not isinstance(stored_diffusion_config, Mapping):
        raise TypeError("Checkpoint diffusion_config must be a mapping")
    _compare_exact_checkpoint_field(
        "diffusion_config",
        dict(expected_diffusion_config),
        dict(stored_diffusion_config),
    )

    for field_name, expected in (
        ("output_channel_names", expected_output_channel_names),
        ("branch_channel_names", expected_branch_channel_names),
        ("trunk_channel_names", expected_trunk_channel_names),
        ("train_cases", list(expected_split.train_cases)),
        ("val_cases", list(expected_split.val_cases)),
        ("test_cases", list(expected_split.test_cases)),
    ):
        actual = _require_checkpoint_field(checkpoint, field_name)
        _compare_exact_checkpoint_field(field_name, expected, list(actual))

    stored_normalizer = _require_checkpoint_field(checkpoint, "y_normalizer")
    if not isinstance(stored_normalizer, Mapping):
        raise TypeError("Checkpoint y_normalizer must be a mapping")
    _compare_float_checkpoint_field(
        "y_normalizer.mean",
        expected_normalizer.mean.detach().cpu(),
        _require_checkpoint_field(stored_normalizer, "mean"),
    )
    _compare_float_checkpoint_field(
        "y_normalizer.std",
        expected_normalizer.std.detach().cpu(),
        _require_checkpoint_field(stored_normalizer, "std"),
    )
    for field_name, expected in (
        ("local_aspect_mean", expected_local_aspect_mean),
        ("local_aspect_std", expected_local_aspect_std),
    ):
        actual = _require_checkpoint_field(checkpoint, field_name)
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=1.0e-6,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                f"Checkpoint field {field_name!r} does not match the current protocol"
            )


def initialize_from_checkpoint(
    model: PointSetDiffusionDenoiser,
    checkpoint_path: Path,
    device: torch.device,
    expected_model_config: Mapping[str, int],
    expected_diffusion_config: Mapping[str, Any],
    expected_output_channel_names: list[str],
    expected_branch_channel_names: list[str],
    expected_trunk_channel_names: list[str],
    expected_split: DiffusionCaseSplit,
    expected_normalizer: FeatureNormalizer,
    expected_local_aspect_mean: float,
    expected_local_aspect_std: float,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load model weights only after strict protocol compatibility checks."""

    del device
    checkpoint = _load_checkpoint(checkpoint_path)
    _require_checkpoint_field(checkpoint, "model_state_dict")
    _validate_checkpoint_protocol(
        checkpoint,
        expected_model_config=expected_model_config,
        expected_diffusion_config=expected_diffusion_config,
        expected_output_channel_names=expected_output_channel_names,
        expected_branch_channel_names=expected_branch_channel_names,
        expected_trunk_channel_names=expected_trunk_channel_names,
        expected_split=expected_split,
        expected_normalizer=expected_normalizer,
        expected_local_aspect_mean=expected_local_aspect_mean,
        expected_local_aspect_std=expected_local_aspect_std,
    )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model_state_dict must be a mapping")

    model.load_state_dict(state_dict, strict=True)
    for field_name in ("epoch", "val_loss"):
        _require_checkpoint_field(checkpoint, field_name)
    if "run_best_val_loss" in checkpoint:
        source_best_val_loss = checkpoint["run_best_val_loss"]
    elif "best_val_loss" in checkpoint:
        source_best_val_loss = checkpoint["best_val_loss"]
    else:
        raise KeyError(
            "Checkpoint is missing required field: run_best_val_loss"
        )
    if source_best_val_loss is None or not math.isfinite(float(source_best_val_loss)):
        raise ValueError("Checkpoint run_best_val_loss must be finite for initialization")
    initialization_metadata = {
        "initialization_mode": "model_only",
        "source_checkpoint_path": _manifest_path(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_val_loss": float(checkpoint["val_loss"]),
        "source_best_val_loss": float(source_best_val_loss),
        "source_dataset_path": checkpoint.get("dataset_h5"),
        "source_dataset_sha256": checkpoint.get("dataset_sha256"),
    }
    return checkpoint, initialization_metadata


def _unpack_batch(batch):
    if len(batch) == 6:
        return batch
    branch, query, target, query_batch_id, sample_idx = batch
    return branch, query, target, query_batch_id, sample_idx, None


def _move_batch(batch, device: torch.device):
    branch, query, target, query_batch_id, sample_idx, branch_mask = _unpack_batch(batch)
    branch = branch.to(device)
    query = query.to(device)
    target = target.to(device)
    query_batch_id = query_batch_id.to(device)
    if branch_mask is not None:
        branch_mask = branch_mask.to(device)
    return branch, query, target, query_batch_id, sample_idx, branch_mask


def train_one_epoch(
    model: PointSetDiffusionDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> float:
    """Run one stochastic baseline diffusion training epoch."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        branch, query, target, query_batch_id, _, branch_mask = _move_batch(
            batch, device
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = epsilon_prediction_loss(
            model=model,
            branch=branch,
            query=query,
            target=target,
            query_batch_id=query_batch_id,
            schedule=schedule,
            branch_mask=branch_mask,
        )
        loss.backward()
        if grad_clip is not None and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(branch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_diffusion_loss(
    model: PointSetDiffusionDenoiser,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
) -> float:
    """Evaluate the historical stochastic validation loss."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        branch, query, target, query_batch_id, _, branch_mask = _move_batch(
            batch, device
        )
        loss, _ = epsilon_prediction_loss(
            model=model,
            branch=branch,
            query=query,
            target=target,
            query_batch_id=query_batch_id,
            schedule=schedule,
            branch_mask=branch_mask,
        )
        batch_size = int(branch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def save_diffusion_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save one checkpoint payload within the current run."""

    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
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


def _history_rows(checkpoint: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if checkpoint is None or "history" not in checkpoint:
        return []
    rows = checkpoint["history"]
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "elapsed_sec",
        "lr",
        "phase",
    ]
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _build_manifest(
    *,
    config: DiffusionTrainingConfig,
    run_id: str,
    run_directory: Path,
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    model_config: Mapping[str, Any],
    schedule: DiffusionSchedule,
    initialization_path: Optional[Path],
    initialization_metadata: Mapping[str, Any],
    best_checkpoint_path: Path,
    latest_checkpoint_path: Path,
    history_path: Path,
    status: str,
    created_at_utc: str,
    started_at_utc: Optional[str],
    finished_at_utc: Optional[str],
    last_completed_epoch: Optional[int],
    failure_type: Optional[str],
    failure_message: Optional[str],
    run_best_val_loss: Optional[float],
    include_checksums: bool,
) -> dict[str, Any]:
    initialization_mode = (
        "model_only" if initialization_path is not None else "fresh"
    )
    source_paths = [
        ("source_notebook", config.source_notebook),
        ("dataset_loader_source", REPOSITORY_ROOT / "DeepONet" / "deeponet_fluent_dataset.py"),
        ("branch_model_source", REPOSITORY_ROOT / "DeepONet" / "fluent_deeponet.py"),
        ("data_module", REPOSITORY_ROOT / "pidiffusion" / "data.py"),
        ("model_module", REPOSITORY_ROOT / "pidiffusion" / "model.py"),
        ("diffusion_module", REPOSITORY_ROOT / "pidiffusion" / "diffusion.py"),
        ("provenance_module", REPOSITORY_ROOT / "pidiffusion" / "provenance.py"),
        ("artifacts_module", REPOSITORY_ROOT / "pidiffusion" / "artifacts.py"),
        ("training_entrypoint", REPOSITORY_ROOT / "experiments" / "train_diffusion.py"),
    ]
    source_files = [
        {
            "role": role,
            **_identity_for_manifest(path, include_checksum=include_checksums),
        }
        for role, path in source_paths
    ]
    best_identity = _optional_identity_for_manifest(
        best_checkpoint_path,
        include_checksum=include_checksums,
    )
    latest_identity = _optional_identity_for_manifest(
        latest_checkpoint_path,
        include_checksum=include_checksums,
    )
    history_identity = _optional_identity_for_manifest(
        history_path,
        include_checksum=include_checksums,
    )
    output_files = []
    for role, identity in (
        ("best_checkpoint", best_identity),
        ("latest_checkpoint", latest_identity),
        ("training_history", history_identity),
    ):
        if identity is not None:
            output_files.append({"role": role, **identity})
    source_checkpoint = None
    if initialization_path is not None:
        source_checkpoint = {
            **_identity_for_manifest(
                initialization_path,
                include_checksum=include_checksums,
            ),
            "source_checkpoint_sha256": initialization_metadata.get(
                "source_checkpoint_sha256"
            ),
            "source_epoch": initialization_metadata.get("source_epoch"),
            "source_val_loss": initialization_metadata.get("source_val_loss"),
            "source_best_val_loss": initialization_metadata.get(
                "source_best_val_loss"
            ),
            "source_dataset_path": initialization_metadata.get(
                "source_dataset_path"
            ),
            "source_dataset_sha256": initialization_metadata.get(
                "source_dataset_sha256"
            ),
            "dataset_checksum_available": (
                initialization_metadata.get("source_dataset_sha256") is not None
            ),
        }
    return {
        "schema_version": "1",
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
            **_identity_for_manifest(
                config.dataset_path,
                include_checksum=include_checksums,
            ),
            "split": {
                "method": "sorted_case_ids_default_rng",
                "seed": config.split_seed,
                "train_fraction": 0.8,
                "validation_fraction": 0.1,
                "train_cases": list(split.train_cases),
                "val_cases": list(split.val_cases),
                "test_cases": list(split.test_cases),
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
            "initialization_mode": initialization_mode,
            "initialization_source": source_checkpoint,
            "source_checkpoint_path": initialization_metadata.get(
                "source_checkpoint_path"
            ),
            "source_checkpoint_sha256": initialization_metadata.get(
                "source_checkpoint_sha256"
            ),
            "source_checkpoint_epoch": initialization_metadata.get("source_epoch"),
            "source_checkpoint_val_loss": initialization_metadata.get(
                "source_val_loss"
            ),
            "source_checkpoint_best_val_loss": initialization_metadata.get(
                "source_best_val_loss"
            ),
            "source_checkpoint_dataset_path": initialization_metadata.get(
                "source_dataset_path"
            ),
            "source_checkpoint_dataset_sha256": initialization_metadata.get(
                "source_dataset_sha256"
            ),
            "run_best_val_loss": run_best_val_loss,
            "best": best_identity,
            "latest": latest_identity,
            "model_config": dict(model_config),
            "diffusion_config": {
                "T": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
        },
        "protocol": {
            "name": "baseline_diffusion_training",
            "initialization_mode": initialization_mode,
            "validation_uses_random_timestep_and_noise": True,
            "query_sampling": "legacy_global_numpy",
            "sampling_used": False,
            "ddim_used": False,
            "test_evaluation_used": False,
            "official_method_selected": False,
        },
        "randomness": {
            "global_seed": config.global_seed,
            "split_seed": config.split_seed,
            "query_sampling": "legacy_global_numpy",
            "validation_random_timestep_and_noise": True,
            "ddim_initial_noise_seed": None,
        },
        "environment": runtime_environment(),
        "outputs": {
            "directory": _manifest_path(run_directory),
            "best_checkpoint": best_identity,
            "latest_checkpoint": latest_identity,
            "training_history": history_identity,
            "files": output_files,
        },
        "notes": [
            "Validation remains stochastic for historical protocol parity.",
            "Initialization loads model weights only; optimizer, scheduler, and RNG state are not restored.",
            "Sampling, reconstruction, and test evaluation are not part of this training run.",
        ],
    }


def run_training(config: DiffusionTrainingConfig) -> Path:
    """Assemble data and model, run baseline training, and publish artifacts."""

    config.dataset_path = _resolve_repo_path(config.dataset_path)
    config.results_root = _resolve_repo_path(config.results_root)
    config.source_notebook = _resolve_repo_path(config.source_notebook)
    initialization_path = (
        _resolve_repo_path(config.initialize_from_checkpoint)
        if config.initialize_from_checkpoint is not None
        else None
    )
    device = _resolve_device(config.device)
    _require_file(config.dataset_path, "Dataset")
    _require_file(config.source_notebook, "Source notebook")
    if initialization_path is not None:
        _require_file(initialization_path, "Initialization checkpoint")
    if config.num_workers != 0:
        raise ValueError("num_workers must be 0 for the canonical data path")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.num_query_points <= 0:
        raise ValueError("num_query_points must be positive")
    if config.validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if config.use_scheduler and config.scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive when the scheduler is enabled")

    set_global_seed(config.global_seed)
    data = load_diffusion_dataset(config.dataset_path)
    split = build_case_split(data, split_seed=config.split_seed)
    normalizer, local_aspect_mean, local_aspect_std = fit_train_normalizers(data, split)

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_diffusion_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_diffusion_batch,
    )

    branch_dim = len(data["branch_channel_names"])
    query_dim = len(data["trunk_channel_names"])
    target_dim = len(data["output_channel_names"])
    model, model_config = build_diffusion_model(branch_dim, query_dim, target_dim)
    model = model.to(device)
    schedule = build_linear_schedule(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    diffusion_config = {
        "T": schedule.timesteps,
        "beta_start": schedule.beta_start,
        "beta_end": schedule.beta_end,
    }
    loaded_checkpoint: Optional[Mapping[str, Any]] = None
    initialization_metadata: dict[str, Any] = {
        "initialization_mode": "fresh",
        "source_checkpoint_path": None,
        "source_checkpoint_sha256": None,
        "source_epoch": None,
        "source_val_loss": None,
        "source_best_val_loss": None,
        "source_dataset_path": None,
        "source_dataset_sha256": None,
    }
    if initialization_path is not None:
        loaded_checkpoint, initialization_metadata = initialize_from_checkpoint(
            model,
            initialization_path,
            device,
            model_config,
            diffusion_config,
            list(data["output_channel_names"]),
            list(data["branch_channel_names"]),
            list(data["trunk_channel_names"]),
            split,
            normalizer,
            local_aspect_mean,
            local_aspect_std,
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_t_max,
            eta_min=config.scheduler_eta_min,
        )

    history = _history_rows(loaded_checkpoint)
    global_step = int(loaded_checkpoint.get("global_step", 0)) if loaded_checkpoint else 0
    source_epoch = int(initialization_metadata["source_epoch"] or 0)
    start_epoch = source_epoch + 1
    end_epoch = source_epoch + config.epochs if loaded_checkpoint else config.epochs

    run_id = _build_run_id(config)
    run_directory = (
        config.results_root / "train_diffusion" / run_id
    ).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    best_checkpoint_path = run_directory / "diffusion_best.pt"
    latest_checkpoint_path = run_directory / "diffusion_latest.pt"
    history_path = run_directory / "training_history.csv"

    created_at_utc = _utc_now()
    prepared_manifest = _build_manifest(
        config=config,
        run_id=run_id,
        run_directory=run_directory,
        data=data,
        split=split,
        normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        model_config=model_config,
        schedule=schedule,
        initialization_path=initialization_path,
        initialization_metadata=initialization_metadata,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        history_path=history_path,
        status="prepared",
        created_at_utc=created_at_utc,
        started_at_utc=None,
        finished_at_utc=None,
        last_completed_epoch=None,
        failure_type=None,
        failure_message=None,
        run_best_val_loss=None,
        include_checksums=False,
    )
    write_manifest(run_directory, prepared_manifest)

    started_at_utc = _utc_now()
    last_completed_epoch: Optional[int] = None
    run_best_val_loss: Optional[float] = None
    try:
        update_manifest(
            run_directory,
            _build_manifest(
                config=config,
                run_id=run_id,
                run_directory=run_directory,
                data=data,
                split=split,
                normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                model_config=model_config,
                schedule=schedule,
                initialization_path=initialization_path,
                initialization_metadata=initialization_metadata,
                best_checkpoint_path=best_checkpoint_path,
                latest_checkpoint_path=latest_checkpoint_path,
                history_path=history_path,
                status="running",
                created_at_utc=created_at_utc,
                started_at_utc=started_at_utc,
                finished_at_utc=None,
                last_completed_epoch=None,
                failure_type=None,
                failure_message=None,
                run_best_val_loss=None,
                include_checksums=False,
            ),
        )

        for epoch in range(start_epoch, end_epoch + 1):
            epoch_start = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                schedule=schedule,
                device=device,
                optimizer=optimizer,
                grad_clip=config.grad_clip,
            )
            global_step += len(train_loader)
            if epoch % config.validation_interval == 0:
                val_loss = evaluate_diffusion_loss(
                    model=model,
                    loader=val_loader,
                    schedule=schedule,
                    device=device,
                )
            else:
                val_loss = float("nan")
            if scheduler is not None:
                scheduler.step()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            elapsed = float(time.time() - epoch_start)
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "elapsed_sec": elapsed,
                "lr": learning_rate,
                "phase": "baseline_diffusion",
            }
            history.append(row)
            _write_history(history_path, history)

            is_best = np.isfinite(val_loss) and (
                run_best_val_loss is None or val_loss < run_best_val_loss
            )
            if is_best:
                run_best_val_loss = float(val_loss)
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "run_best_val_loss": run_best_val_loss,
                "source_checkpoint_path": initialization_metadata.get(
                    "source_checkpoint_path"
                ),
                "source_checkpoint_sha256": initialization_metadata.get(
                    "source_checkpoint_sha256"
                ),
                "source_checkpoint_epoch": initialization_metadata.get("source_epoch"),
                "source_checkpoint_val_loss": initialization_metadata.get(
                    "source_val_loss"
                ),
                "source_checkpoint_best_val_loss": initialization_metadata.get(
                    "source_best_val_loss"
                ),
                "source_checkpoint_dataset_path": initialization_metadata.get(
                    "source_dataset_path"
                ),
                "source_checkpoint_dataset_sha256": initialization_metadata.get(
                    "source_dataset_sha256"
                ),
                "initialization_mode": initialization_metadata["initialization_mode"],
                "model_config": model_config,
                "diffusion_config": diffusion_config,
                "training_config": {
                    "phase": "baseline_diffusion",
                    "initialization_mode": initialization_metadata["initialization_mode"],
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "grad_clip": config.grad_clip,
                    "scheduler_t_max": config.scheduler_t_max,
                    "scheduler_eta_min": config.scheduler_eta_min,
                    "num_query_points": config.num_query_points,
                    "split_seed": config.split_seed,
                    "global_seed": config.global_seed,
                },
                "output_channel_names": list(data["output_channel_names"]),
                "branch_channel_names": list(data["branch_channel_names"]),
                "trunk_channel_names": list(data["trunk_channel_names"]),
                "y_normalizer": normalizer.state_dict(),
                "local_aspect_mean": local_aspect_mean,
                "local_aspect_std": local_aspect_std,
                "train_cases": list(split.train_cases),
                "val_cases": list(split.val_cases),
                "test_cases": list(split.test_cases),
                "history": history,
                "dataset_h5": str(config.dataset_path),
            }
            if epoch % config.checkpoint_interval == 0:
                save_diffusion_checkpoint(latest_checkpoint_path, payload)
            if is_best:
                save_diffusion_checkpoint(best_checkpoint_path, payload)
            last_completed_epoch = epoch
            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={learning_rate:.3e}"
            )

        if run_best_val_loss is None:
            raise RuntimeError(
                "No finite validation loss was observed; no best checkpoint was created"
            )
        finished_at_utc = _utc_now()
        manifest_path = update_manifest(
            run_directory,
            _build_manifest(
                config=config,
                run_id=run_id,
                run_directory=run_directory,
                data=data,
                split=split,
                normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                model_config=model_config,
                schedule=schedule,
                initialization_path=initialization_path,
                initialization_metadata=initialization_metadata,
                best_checkpoint_path=best_checkpoint_path,
                latest_checkpoint_path=latest_checkpoint_path,
                history_path=history_path,
                status="completed",
                created_at_utc=created_at_utc,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                last_completed_epoch=last_completed_epoch,
                failure_type=None,
                failure_message=None,
                run_best_val_loss=run_best_val_loss,
                include_checksums=True,
            ),
        )
    except Exception as exc:
        failure_type, failure_message = _short_failure(exc)
        try:
            update_manifest(
                run_directory,
                _build_manifest(
                    config=config,
                    run_id=run_id,
                    run_directory=run_directory,
                    data=data,
                    split=split,
                    normalizer=normalizer,
                    local_aspect_mean=local_aspect_mean,
                    local_aspect_std=local_aspect_std,
                    model_config=model_config,
                    schedule=schedule,
                    initialization_path=initialization_path,
                    initialization_metadata=initialization_metadata,
                    best_checkpoint_path=best_checkpoint_path,
                    latest_checkpoint_path=latest_checkpoint_path,
                    history_path=history_path,
                    status="failed",
                    created_at_utc=created_at_utc,
                    started_at_utc=started_at_utc,
                    finished_at_utc=_utc_now(),
                    last_completed_epoch=last_completed_epoch,
                    failure_type=failure_type,
                    failure_message=failure_message,
                    run_best_val_loss=run_best_val_loss,
                    include_checksums=False,
                ),
            )
        except Exception as manifest_exc:
            manifest_failure_type, manifest_failure_message = _short_failure(
                manifest_exc
            )
            print(
                "Warning: failed to publish failure manifest "
                f"({manifest_failure_type}: {manifest_failure_message})"
            )
        raise
    print("Completed baseline diffusion run:", run_directory)
    print("Manifest:", manifest_path)
    return run_directory


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the limited command-line override parser."""

    parser = argparse.ArgumentParser(
        description="Train the baseline point-set diffusion model."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start training; without this flag only the resolved configuration is printed",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_path",
        default=str(DEFAULT_DATASET_PATH),
        help="HDF5 dataset path; relative paths use the repository root",
    )
    parser.add_argument(
        "--results-root",
        default=str(REPOSITORY_ROOT / "results"),
        help="Results root; runs are stored under results/train_diffusion/<run_id>",
    )
    parser.add_argument(
        "--source-notebook",
        default=str(DEFAULT_SOURCE_NOTEBOOK),
        help="Notebook used as the canonical training source",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Torch device, for example cuda:1 or cpu",
    )
    parser.add_argument(
        "--seed",
        dest="global_seed",
        type=int,
        default=None,
        help="Optional global Python, NumPy, and Torch seed",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--initialize-from-checkpoint", default=None)
    parser.add_argument("--checkpoint-tag", default="baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--num-query-points",
        type=int,
        default=8192,
        help="Number of query points sampled per cell",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the fresh cosine scheduler",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DiffusionTrainingConfig:
    """Convert parser values into the typed training configuration."""

    return DiffusionTrainingConfig(
        dataset_path=_resolve_repo_path(args.dataset_path),
        results_root=_resolve_repo_path(args.results_root),
        source_notebook=_resolve_repo_path(args.source_notebook),
        device=args.device,
        global_seed=args.global_seed,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        initialize_from_checkpoint=(
            _resolve_repo_path(args.initialize_from_checkpoint)
            if args.initialize_from_checkpoint is not None
            else None
        ),
        checkpoint_tag=args.checkpoint_tag,
        run_id=args.run_id,
        num_query_points=args.num_query_points,
        use_scheduler=not args.no_scheduler,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and optionally execute one baseline diffusion run."""

    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    if not args.run:
        print("Resolved configuration:")
        for config_field in fields(config):
            print(f"  {config_field.name}: {getattr(config, config_field.name)}")
        print("Training was not started. Pass --run to start training.")
        return 0
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
