"""Train the canonical field-diffusion baseline on fixed train/validation datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pidiffusion.artifacts import update_manifest, write_manifest  # noqa: E402
from pidiffusion.data import (  # noqa: E402
    DiffusionCellDataset,
    FeatureNormalizer,
    collate_diffusion_batch,
    fit_subdomain_balanced_normalizers,
    load_diffusion_dataset,
    normalize_diffusion_branch,
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


DEFAULT_TRAIN_DATASET_PATH = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random10_train.h5"
)
DEFAULT_VAL_DATASET_PATH = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)
DEFAULT_SOURCE_NOTEBOOK = (
    REPOSITORY_ROOT / "train_domain_channel_diffusion_iterative_prediction.ipynb"
)

PROTOCOL_VERSION = "field_diffusion_baseline_v3"
VALIDATION_PLAN_VERSION = "fixed_validation_plan_v1"
NORMALIZER_CACHE_VERSION = "subdomain_balanced_normalizer_cache_v1"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AMBIGUOUS_RUN_IDS = {"latest", "final", "new", "updated"}


@dataclass
class DiffusionTrainingConfig:
    """Typed configuration for one canonical field-diffusion run."""

    train_dataset_path: Path = field(default_factory=lambda: DEFAULT_TRAIN_DATASET_PATH)
    val_dataset_path: Path = field(default_factory=lambda: DEFAULT_VAL_DATASET_PATH)
    results_root: Path = field(default_factory=lambda: REPOSITORY_ROOT / "results")
    source_notebook: Path = field(default_factory=lambda: DEFAULT_SOURCE_NOTEBOOK)
    device: str = "cuda:1"

    global_seed: int = 0
    validation_query_seed: int = 0
    validation_diffusion_seed: int = 0

    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 0.0
    boundary_loss_weight: float = 0.0

    scheduler: str = "plateau"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1.0e-4
    plateau_min_lr: float = 1.0e-5
    min_lr_early_stop_patience: int = 10

    num_query_points: int = 8192
    num_workers: int = 0
    checkpoint_interval: int = 1
    validation_interval: int = 1
    selection_metric: str = "interior_epsilon"
    progress_interval: int = 100

    normalizer_cache_path: Optional[Path] = None
    normalizer_from_checkpoint: Optional[Path] = None
    refresh_normalizer_cache: bool = False

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


def _optional_identity_for_manifest(
    path: Path,
    *,
    include_checksum: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _identity_for_manifest(path, include_checksum=include_checksum)


def _short_failure(exc: BaseException) -> tuple[str, str]:
    message = str(exc).splitlines()[0].strip()
    if not message:
        message = "No exception message was provided."
    return type(exc).__name__, message[:240]


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
    return f"{timestamp}_field_diffusion_{tag}_seed{config.global_seed}"


def set_global_seed(seed: int) -> None:
    """Set the training RNGs used for initialization, sampling, and shuffling."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    """Instantiate the fixed 382,083-parameter field-diffusion architecture."""

    model_config = _model_config(
        branch_input_dim=branch_input_dim,
        query_input_dim=query_input_dim,
        target_dim=target_dim,
    )
    model = PointSetDiffusionDenoiser(**model_config)
    return model, model_config


def _decode_hdf5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_dataset_protocol(path: Path) -> dict[str, Any]:
    keys = (
        "dataset_name",
        "dataset_role",
        "split_role",
        "interface_placement",
        "n_subdomains",
        "n_realizations",
        "split_manifest_path",
        "split_manifest_sha256",
        "dataset_builder_path",
        "dataset_builder_sha256",
        "generator_script_path",
        "generator_script_sha256",
        "dataset_random_seed",
        "decomposition_seed",
    )
    with h5py.File(path, "r") as handle:
        return {
            key: _decode_hdf5_attr(handle.attrs[key])
            for key in keys
            if key in handle.attrs
        }


def _validate_dataset_pair(
    train_path: Path,
    val_path: Path,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_protocol = _read_dataset_protocol(train_path)
    val_protocol = _read_dataset_protocol(val_path)

    if train_protocol.get("dataset_role") != "shared_randomized_training":
        raise ValueError(
            "Training dataset_role must be 'shared_randomized_training', got "
            f"{train_protocol.get('dataset_role')!r}"
        )
    if val_protocol.get("dataset_role") != "canonical_randomized_validation":
        raise ValueError(
            "Validation dataset_role must be 'canonical_randomized_validation', got "
            f"{val_protocol.get('dataset_role')!r}"
        )
    if val_protocol.get("split_role") != "validation":
        raise ValueError("Validation dataset split_role must be 'validation'")
    if train_protocol.get("interface_placement") != "random":
        raise ValueError("Training dataset must use randomized x-strip placement")
    if val_protocol.get("interface_placement") != "random":
        raise ValueError("Validation dataset must use randomized x-strip placement")
    if int(train_protocol.get("n_subdomains", -1)) != 10:
        raise ValueError("Training dataset must contain 10 subdomains per realization")
    if int(val_protocol.get("n_subdomains", -1)) != 10:
        raise ValueError("Validation dataset must contain 10 subdomains per realization")
    if int(train_protocol.get("n_realizations", -1)) != 10:
        raise ValueError("Training dataset must contain 10 realizations per channel")
    if int(val_protocol.get("n_realizations", -1)) != 5:
        raise ValueError("Validation dataset must contain 5 realizations per channel")

    for field_name in (
        "branch_channel_names",
        "trunk_channel_names",
        "output_channel_names",
    ):
        if list(train_data[field_name]) != list(val_data[field_name]):
            raise ValueError(
                f"Training and validation {field_name} do not match"
            )

    train_cases = {str(meta["case_id"]) for meta in train_data["metadata"]}
    val_cases = {str(meta["case_id"]) for meta in val_data["metadata"]}
    if len(train_cases) != 160:
        raise ValueError(f"Expected 160 training cases, got {len(train_cases)}")
    if len(val_cases) != 20:
        raise ValueError(f"Expected 20 validation cases, got {len(val_cases)}")
    overlap = train_cases & val_cases
    if overlap:
        raise ValueError(
            "Training and validation datasets overlap in case IDs: "
            + ", ".join(sorted(overlap)[:10])
        )
    if len(train_data["samples"]) != 16000:
        raise ValueError(
            f"Expected 16000 training samples, got {len(train_data['samples'])}"
        )
    if len(val_data["samples"]) != 1000:
        raise ValueError(
            f"Expected 1000 validation samples, got {len(val_data['samples'])}"
        )

    train_split_sha = train_protocol.get("split_manifest_sha256")
    val_split_sha = val_protocol.get("split_manifest_sha256")
    if not train_split_sha or train_split_sha != val_split_sha:
        raise ValueError(
            "Training and validation datasets must reference the same split manifest SHA256"
        )
    train_builder_sha = train_protocol.get("dataset_builder_sha256")
    val_builder_sha = val_protocol.get("dataset_builder_sha256")
    if not train_builder_sha or train_builder_sha != val_builder_sha:
        raise ValueError(
            "Training and validation datasets must use the same randomized dataset builder"
        )

    return train_protocol, val_protocol


def _sample_identity(metadata_row: Mapping[str, Any]) -> str:
    return (
        f"{metadata_row['case_id']}|"
        f"r{int(metadata_row['realization_id'])}|"
        f"s{int(metadata_row['subdomain_id'])}"
    )


def _stable_seed(base_seed: int, namespace: str, identity: str) -> int:
    payload = (
        f"{VALIDATION_PLAN_VERSION}|{int(base_seed)}|{namespace}|{identity}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


class FixedValidationDataset(Dataset):
    """Validation dataset with a fixed seeded query subset for every subdomain."""

    def __init__(
        self,
        *,
        samples: Sequence[Mapping[str, np.ndarray]],
        metadata: Sequence[Mapping[str, Any]],
        n_query_points: int,
        query_seed: int,
        target_normalizer: FeatureNormalizer,
        local_aspect_mean: float,
        local_aspect_std: float,
        branch_channel_names: Sequence[str],
    ):
        if len(samples) != len(metadata):
            raise ValueError("Validation sample and metadata counts must match")
        if n_query_points <= 0:
            raise ValueError("n_query_points must be positive")

        self.samples = list(samples)
        self.metadata = list(metadata)
        self.n_query_points = int(n_query_points)
        self.query_seed = int(query_seed)
        self.target_normalizer = target_normalizer
        self.local_aspect_mean = float(local_aspect_mean)
        self.local_aspect_std = float(local_aspect_std)
        self.branch_channel_names = list(branch_channel_names)

        self._query_indices: list[np.ndarray] = []
        digest = hashlib.sha256()
        digest.update(VALIDATION_PLAN_VERSION.encode("utf-8"))
        digest.update(str(self.query_seed).encode("utf-8"))

        for sample_index, (sample, meta) in enumerate(zip(self.samples, self.metadata)):
            query_count = int(np.asarray(sample["query"]).shape[0])
            selected_count = min(self.n_query_points, query_count)
            if selected_count == query_count:
                point_indices = np.arange(query_count, dtype=np.int32)
            else:
                identity = _sample_identity(meta)
                seed = _stable_seed(self.query_seed, "query_subset", identity)
                rng = np.random.default_rng(seed)
                point_indices = np.sort(
                    rng.choice(query_count, size=selected_count, replace=False)
                ).astype(np.int32, copy=False)

            self._query_indices.append(point_indices)
            digest.update(sample_index.to_bytes(8, "little", signed=False))
            digest.update(_sample_identity(meta).encode("utf-8"))
            digest.update(point_indices.tobytes(order="C"))

        self.query_plan_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_index = int(index)
        sample = self.samples[sample_index]
        point_indices = self._query_indices[sample_index]

        branch = normalize_diffusion_branch(
            np.asarray(sample["branch"], dtype=np.float32),
            branch_channel_names=self.branch_channel_names,
            target_normalizer=self.target_normalizer,
            local_aspect_mean=self.local_aspect_mean,
            local_aspect_std=self.local_aspect_std,
        )
        query = np.asarray(sample["query"], dtype=np.float32)[point_indices]
        target = np.asarray(sample["target"], dtype=np.float32)[point_indices]
        target = (
            self.target_normalizer.encode(target)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return (
            torch.from_numpy(branch).float(),
            torch.from_numpy(query).float(),
            torch.from_numpy(target).float(),
            torch.tensor(sample_index, dtype=torch.long),
        )


def _build_validation_timestep_plan(
    metadata: Sequence[Mapping[str, Any]],
    *,
    timesteps: int,
    diffusion_seed: int,
) -> tuple[np.ndarray, str]:
    """Assign well-distributed fixed timesteps by stable validation identity."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    n_samples = len(metadata)
    if n_samples == 0:
        raise ValueError("Validation metadata must not be empty")

    identities = [_sample_identity(row) for row in metadata]
    if len(set(identities)) != len(identities):
        raise ValueError("Validation sample identities must be unique")

    sorted_indices = sorted(range(n_samples), key=lambda i: identities[i])
    rng = np.random.default_rng(int(diffusion_seed))

    assigned = np.empty(n_samples, dtype=np.int64)
    position = 0
    while position < n_samples:
        permutation = rng.permutation(timesteps)
        take = min(timesteps, n_samples - position)
        for offset in range(take):
            original_index = sorted_indices[position + offset]
            assigned[original_index] = int(permutation[offset])
        position += take

    digest = hashlib.sha256()
    digest.update(VALIDATION_PLAN_VERSION.encode("utf-8"))
    digest.update(str(int(diffusion_seed)).encode("utf-8"))
    for index in sorted_indices:
        digest.update(identities[index].encode("utf-8"))
        digest.update(int(assigned[index]).to_bytes(8, "little", signed=False))

    return assigned, digest.hexdigest()


def _validation_noise_for_batch(
    *,
    metadata: Sequence[Mapping[str, Any]],
    sample_indices: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: Optional[torch.Tensor],
    branch: torch.Tensor,
    diffusion_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed per-sample interior and boundary Gaussian noise on CPU."""

    batch_indices = [int(value) for value in sample_indices.detach().cpu().tolist()]
    interior_parts: list[torch.Tensor] = []
    boundary_parts: list[torch.Tensor] = []
    max_branch_points = int(branch.shape[1])

    query_batch_cpu = query_batch_id.detach().cpu()
    branch_mask_cpu = branch_mask.detach().cpu() if branch_mask is not None else None

    for batch_id, sample_index in enumerate(batch_indices):
        identity = _sample_identity(metadata[sample_index])

        query_count = int((query_batch_cpu == batch_id).sum().item())
        target_generator = torch.Generator(device="cpu")
        target_generator.manual_seed(
            _stable_seed(diffusion_seed, "interior_noise", identity)
        )
        interior_parts.append(
            torch.randn((query_count, 3), generator=target_generator, dtype=torch.float32)
        )

        if branch_mask_cpu is None:
            valid_branch_points = max_branch_points
        else:
            valid_branch_points = int(branch_mask_cpu[batch_id].sum().item())
        boundary_generator = torch.Generator(device="cpu")
        boundary_generator.manual_seed(
            _stable_seed(diffusion_seed, "boundary_noise", identity)
        )
        valid_boundary_noise = torch.randn(
            (valid_branch_points, 3),
            generator=boundary_generator,
            dtype=torch.float32,
        )
        padded = torch.zeros((max_branch_points, 3), dtype=torch.float32)
        padded[:valid_branch_points] = valid_boundary_noise
        boundary_parts.append(padded)

    target_noise = torch.cat(interior_parts, dim=0).to(device=device)
    boundary_noise = torch.stack(boundary_parts, dim=0).to(device=device)
    return target_noise, boundary_noise


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def _require_checkpoint_field(checkpoint: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in checkpoint:
        raise KeyError(f"Checkpoint is missing required field: {field_name}")
    return checkpoint[field_name]


def _compare_exact_checkpoint_field(field_name: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"Checkpoint field {field_name!r} does not match the current protocol"
        )


def _compare_float_checkpoint_field(field_name: str, expected: Any, actual: Any) -> None:
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
    expected_normalizer: FeatureNormalizer,
    expected_local_aspect_mean: float,
    expected_local_aspect_std: float,
    expected_train_dataset_path: Path,
    expected_val_dataset_path: Path,
) -> None:
    _compare_exact_checkpoint_field(
        "training_protocol_version",
        PROTOCOL_VERSION,
        _require_checkpoint_field(checkpoint, "training_protocol_version"),
    )

    stored_model_config = _require_checkpoint_field(checkpoint, "model_config")
    if not isinstance(stored_model_config, Mapping):
        raise TypeError("Checkpoint model_config must be a mapping")
    _compare_exact_checkpoint_field(
        "model_config", dict(expected_model_config), dict(stored_model_config)
    )

    stored_diffusion_config = _require_checkpoint_field(checkpoint, "diffusion_config")
    if not isinstance(stored_diffusion_config, Mapping):
        raise TypeError("Checkpoint diffusion_config must be a mapping")
    _compare_exact_checkpoint_field(
        "diffusion_config", dict(expected_diffusion_config), dict(stored_diffusion_config)
    )

    for field_name, expected in (
        ("output_channel_names", expected_output_channel_names),
        ("branch_channel_names", expected_branch_channel_names),
        ("trunk_channel_names", expected_trunk_channel_names),
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
            float(actual), float(expected), rel_tol=1.0e-6, abs_tol=1.0e-7
        ):
            raise ValueError(
                f"Checkpoint field {field_name!r} does not match the current protocol"
            )

    _compare_exact_checkpoint_field(
        "train_dataset_h5",
        str(expected_train_dataset_path),
        str(_require_checkpoint_field(checkpoint, "train_dataset_h5")),
    )
    _compare_exact_checkpoint_field(
        "val_dataset_h5",
        str(expected_val_dataset_path),
        str(_require_checkpoint_field(checkpoint, "val_dataset_h5")),
    )


def initialize_from_checkpoint(
    model: PointSetDiffusionDenoiser,
    checkpoint_path: Path,
    *,
    expected_model_config: Mapping[str, int],
    expected_diffusion_config: Mapping[str, Any],
    expected_output_channel_names: list[str],
    expected_branch_channel_names: list[str],
    expected_trunk_channel_names: list[str],
    expected_normalizer: FeatureNormalizer,
    expected_local_aspect_mean: float,
    expected_local_aspect_std: float,
    expected_train_dataset_path: Path,
    expected_val_dataset_path: Path,
) -> dict[str, Any]:
    """Load model weights only after strict field-baseline compatibility checks."""

    checkpoint = _load_checkpoint(checkpoint_path)
    _validate_checkpoint_protocol(
        checkpoint,
        expected_model_config=expected_model_config,
        expected_diffusion_config=expected_diffusion_config,
        expected_output_channel_names=expected_output_channel_names,
        expected_branch_channel_names=expected_branch_channel_names,
        expected_trunk_channel_names=expected_trunk_channel_names,
        expected_normalizer=expected_normalizer,
        expected_local_aspect_mean=expected_local_aspect_mean,
        expected_local_aspect_std=expected_local_aspect_std,
        expected_train_dataset_path=expected_train_dataset_path,
        expected_val_dataset_path=expected_val_dataset_path,
    )

    state_dict = _require_checkpoint_field(checkpoint, "model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model_state_dict must be a mapping")
    model.load_state_dict(state_dict, strict=True)

    source_best = checkpoint.get("run_best_val_loss")
    if source_best is None or not math.isfinite(float(source_best)):
        raise ValueError("Checkpoint run_best_val_loss must be finite")

    return {
        "initialization_mode": "model_only",
        "source_checkpoint_path": _manifest_path(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_epoch": int(_require_checkpoint_field(checkpoint, "epoch")),
        "source_val_loss": float(_require_checkpoint_field(checkpoint, "val_loss")),
        "source_best_val_loss": float(source_best),
        "source_train_dataset_path": checkpoint.get("train_dataset_h5"),
        "source_val_dataset_path": checkpoint.get("val_dataset_h5"),
    }


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
    sample_idx = sample_idx.to(device)
    if branch_mask is not None:
        branch_mask = branch_mask.to(device)
    return branch, query, target, query_batch_id, sample_idx, branch_mask


def _global_grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_sum = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        grad_norm = float(parameter.grad.detach().norm(2).item())
        squared_sum += grad_norm * grad_norm
    return math.sqrt(squared_sum) if found else 0.0


def train_one_epoch(
    model: PointSetDiffusionDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    *,
    grad_clip: float,
    boundary_loss_weight: float,
    progress_interval: int = 0,
    epoch: Optional[int] = None,
) -> dict[str, float]:
    """Run one stochastic field-level diffusion training epoch."""

    model.train()
    total_objective = 0.0
    total_interior = 0.0
    total_boundary = 0.0
    total_samples = 0
    grad_norms: list[float] = []
    clipped_steps = 0
    steps = 0
    progress_start = time.perf_counter()
    total_loader_steps = len(loader)

    for batch in loader:
        branch, query, target, query_batch_id, _, branch_mask = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, details = epsilon_prediction_loss(
            model=model,
            branch=branch,
            query=query,
            target=target,
            query_batch_id=query_batch_id,
            schedule=schedule,
            branch_mask=branch_mask,
            boundary_loss_weight=boundary_loss_weight,
        )
        loss.backward()

        if grad_clip > 0.0:
            raw_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            raw_grad_norm = float(raw_grad_norm_tensor.item())
            if raw_grad_norm > grad_clip:
                clipped_steps += 1
        else:
            raw_grad_norm = _global_grad_norm(list(model.parameters()))

        if not math.isfinite(raw_grad_norm):
            raise RuntimeError("Encountered a non-finite gradient norm")
        grad_norms.append(raw_grad_norm)

        optimizer.step()

        batch_size = int(branch.shape[0])
        total_objective += float(loss.item()) * batch_size
        total_interior += float(details["interior_loss"].item()) * batch_size
        total_boundary += float(details["boundary_loss"].item()) * batch_size
        total_samples += batch_size
        steps += 1

        should_report = progress_interval > 0 and (
            steps % progress_interval == 0 or steps == total_loader_steps
        )
        if should_report:
            elapsed = time.perf_counter() - progress_start
            denominator = max(total_samples, 1)
            epoch_label = f"{int(epoch):04d}" if epoch is not None else "????"
            print(
                f"[Train] epoch {epoch_label} | step {steps:04d}/{total_loader_steps:04d} | "
                f"loss={total_objective / denominator:.6f} | "
                f"int={total_interior / denominator:.6f} | "
                f"grad={raw_grad_norm:.3e} | "
                f"{total_samples / max(elapsed, 1.0e-12):.1f} samples/s",
                flush=True,
            )

    denominator = max(total_samples, 1)
    grad_array = np.asarray(grad_norms, dtype=np.float64)
    return {
        "objective": total_objective / denominator,
        "interior": total_interior / denominator,
        "boundary": total_boundary / denominator,
        "grad_norm_mean": float(grad_array.mean()) if grad_array.size else 0.0,
        "grad_norm_p95": float(np.percentile(grad_array, 95.0)) if grad_array.size else 0.0,
        "grad_norm_max": float(grad_array.max()) if grad_array.size else 0.0,
        "clip_fraction": float(clipped_steps / max(steps, 1)),
        "steps": float(steps),
        "samples": float(total_samples),
    }


@torch.no_grad()
def evaluate_deterministic_validation(
    model: PointSetDiffusionDenoiser,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    *,
    metadata: Sequence[Mapping[str, Any]],
    timestep_plan: np.ndarray,
    diffusion_seed: int,
    boundary_loss_weight: float,
) -> dict[str, float]:
    """Evaluate fixed query/timestep/noise epsilon metrics on canonical validation."""

    model.eval()
    total_interior = 0.0
    total_boundary = 0.0
    total_objective = 0.0
    total_samples = 0

    for batch in loader:
        branch, query, target, query_batch_id, sample_idx, branch_mask = _move_batch(
            batch, device
        )
        sample_idx_cpu = sample_idx.detach().cpu().numpy().astype(np.int64)
        t_subdomain = torch.as_tensor(
            timestep_plan[sample_idx_cpu], dtype=torch.long, device=device
        )
        target_noise, boundary_noise = _validation_noise_for_batch(
            metadata=metadata,
            sample_indices=sample_idx,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
            branch=branch,
            diffusion_seed=diffusion_seed,
            device=device,
        )

        # Compute the boundary diagnostic on every validation run, even when the
        # training boundary weight is zero. The selection metric remains explicit.
        _, details = epsilon_prediction_loss(
            model=model,
            branch=branch,
            query=query,
            target=target,
            query_batch_id=query_batch_id,
            schedule=schedule,
            branch_mask=branch_mask,
            t_subdomain=t_subdomain,
            target_noise=target_noise,
            boundary_loss_weight=1.0,
            boundary_noise=boundary_noise,
        )

        interior = float(details["interior_loss"].item())
        boundary = float(details["boundary_loss"].item())
        objective = interior + float(boundary_loss_weight) * boundary
        batch_size = int(branch.shape[0])
        total_interior += interior * batch_size
        total_boundary += boundary * batch_size
        total_objective += objective * batch_size
        total_samples += batch_size

    denominator = max(total_samples, 1)
    return {
        "interior": total_interior / denominator,
        "boundary": total_boundary / denominator,
        "objective": total_objective / denominator,
        "samples": float(total_samples),
    }


def _selection_loss(metrics: Mapping[str, float], selection_metric: str) -> float:
    if selection_metric == "interior_epsilon":
        return float(metrics["interior"])
    if selection_metric == "training_objective":
        return float(metrics["objective"])
    raise ValueError(f"Unsupported selection metric: {selection_metric!r}")


def save_diffusion_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save one checkpoint payload within the current run."""

    temporary_path: Path | None = None
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


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "train_interior_loss",
        "train_boundary_loss",
        "val_loss",
        "val_interior_loss",
        "val_boundary_loss",
        "val_objective",
        "grad_norm_mean",
        "grad_norm_p95",
        "grad_norm_max",
        "clip_fraction",
        "elapsed_sec",
        "samples_per_sec",
        "peak_cuda_memory_gib",
        "lr",
        "phase",
    ]
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DiffusionTrainingConfig,
):
    if config.scheduler == "none":
        return None
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            threshold_mode="abs",
            min_lr=config.plateau_min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler!r}")


def _build_manifest(
    *,
    config: DiffusionTrainingConfig,
    run_id: str,
    run_directory: Path,
    train_data: Mapping[str, Any],
    val_data: Mapping[str, Any],
    train_protocol: Mapping[str, Any],
    val_protocol: Mapping[str, Any],
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    model_config: Mapping[str, Any],
    schedule: DiffusionSchedule,
    query_plan_sha256: str,
    timestep_plan_sha256: str,
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
    termination_reason: Optional[str] = None,
) -> dict[str, Any]:
    initialization_mode = "model_only" if initialization_path is not None else "fresh"
    source_paths = [
        ("historical_source_notebook", config.source_notebook),
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
        best_checkpoint_path, include_checksum=include_checksums
    )
    latest_identity = _optional_identity_for_manifest(
        latest_checkpoint_path, include_checksum=include_checksums
    )
    history_identity = _optional_identity_for_manifest(
        history_path, include_checksum=include_checksums
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
                initialization_path, include_checksum=include_checksums
            ),
            **dict(initialization_metadata),
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
        "termination_reason": termination_reason,
        "git": git_state(REPOSITORY_ROOT),
        "source_files": source_files,
        "dataset": {
            "train": {
                **_identity_for_manifest(
                    config.train_dataset_path, include_checksum=include_checksums
                ),
                "protocol_attributes": dict(train_protocol),
                "n_samples": len(train_data["samples"]),
                "n_cases": len({str(m["case_id"]) for m in train_data["metadata"]}),
            },
            "validation": {
                **_identity_for_manifest(
                    config.val_dataset_path, include_checksum=include_checksums
                ),
                "protocol_attributes": dict(val_protocol),
                "n_samples": len(val_data["samples"]),
                "n_cases": len({str(m["case_id"]) for m in val_data["metadata"]}),
            },
            "internal_split_used": False,
            "test_data_loaded": False,
            "normalizer_source": "entire_training_dataset_subdomain_balanced",
            "target_mean": normalizer.mean.detach().cpu().tolist(),
            "target_std": normalizer.std.detach().cpu().tolist(),
            "local_aspect_mean": local_aspect_mean,
            "local_aspect_std": local_aspect_std,
            "channel_names": {
                "branch": list(train_data["branch_channel_names"]),
                "query": list(train_data["trunk_channel_names"]),
                "target": list(train_data["output_channel_names"]),
            },
        },
        "checkpoint": {
            "initialization_mode": initialization_mode,
            "initialization_source": source_checkpoint,
            "run_best_val_loss": run_best_val_loss,
            "selection_metric": config.selection_metric,
            "best": best_identity,
            "latest": latest_identity,
            "model_config": dict(model_config),
            "diffusion_config": {
                "T": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
                "prediction_target": "epsilon",
            },
        },
        "protocol": {
            "name": PROTOCOL_VERSION,
            "diffusion_sample_unit": "one_subdomain_field",
            "training_timestep_policy": "one_independent_timestep_per_subdomain",
            "training_noise_policy": "iid_gaussian_per_query_point_and_output_component",
            "training_loss_aggregation": "mean_within_subdomain_then_equal_mean_across_subdomains",
            "boundary_loss": {
                "type": "masked_boundary_epsilon_denoising",
                "weight": config.boundary_loss_weight,
                "hard_clamping_used": False,
            },
            "optimization": {
                "optimizer": "AdamW",
                "initial_learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "gradient_clipping": config.grad_clip,
                "scheduler": config.scheduler,
                "plateau_monitor": config.selection_metric,
                "plateau_factor": config.plateau_factor,
                "plateau_patience": config.plateau_patience,
                "plateau_threshold": config.plateau_threshold,
                "plateau_threshold_mode": "abs",
                "plateau_min_lr": config.plateau_min_lr,
                "min_lr_early_stop_patience": config.min_lr_early_stop_patience,
                "min_lr_early_stop_threshold": config.plateau_threshold,
                "epochs_ceiling": config.epochs,
            },
            "validation": {
                "dataset": "fixed_randomized_xstrip_validation",
                "query_subset": "fixed_seeded_subset_without_replacement",
                "query_plan_sha256": query_plan_sha256,
                "timestep_policy": "balanced_permutation_assigned_by_stable_sample_identity",
                "timestep_plan_sha256": timestep_plan_sha256,
                "noise_policy": "fixed_iid_gaussian_per_stable_sample_identity",
                "boundary_diagnostic_always_computed": True,
                "selection_metric": config.selection_metric,
            },
            "sampling_used": False,
            "ddim_used": False,
            "test_evaluation_used": False,
        },
        "randomness": {
            "global_seed": config.global_seed,
            "validation_query_seed": config.validation_query_seed,
            "validation_diffusion_seed": config.validation_diffusion_seed,
            "training_query_sampling": "numpy_random_without_replacement_up_to_cap",
            "validation_plan_version": VALIDATION_PLAN_VERSION,
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
            "Best-checkpoint selection uses only the canonical randomized validation dataset.",
            "CP/AR1 test datasets are not loaded or evaluated by this training entrypoint.",
            "Initialization is model-only; optimizer, scheduler, history, and RNG state are fresh.",
            "Generated-field reverse-sampling metrics are intentionally outside this training loop.",
        ],
    }


def _checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: PointSetDiffusionDenoiser,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    val_loss: float,
    run_best_val_loss: float,
    config: DiffusionTrainingConfig,
    model_config: Mapping[str, Any],
    diffusion_config: Mapping[str, Any],
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    train_data: Mapping[str, Any],
    query_plan_sha256: str,
    timestep_plan_sha256: str,
    initialization_metadata: Mapping[str, Any],
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "training_protocol_version": PROTOCOL_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": float(train_metrics["objective"]),
        "train_interior_loss": float(train_metrics["interior"]),
        "train_boundary_loss": float(train_metrics["boundary"]),
        "val_loss": float(val_loss),
        "val_interior_loss": float(val_metrics["interior"]),
        "val_boundary_loss": float(val_metrics["boundary"]),
        "val_objective": float(val_metrics["objective"]),
        "run_best_val_loss": float(run_best_val_loss),
        "selection_metric": config.selection_metric,
        "initialization_metadata": dict(initialization_metadata),
        "model_config": dict(model_config),
        "diffusion_config": dict(diffusion_config),
        "training_config": {
            "phase": "field_diffusion_baseline",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "boundary_loss_weight": config.boundary_loss_weight,
            "batch_size": config.batch_size,
            "epochs_ceiling": config.epochs,
            "scheduler": config.scheduler,
            "plateau_factor": config.plateau_factor,
            "plateau_patience": config.plateau_patience,
            "plateau_threshold": config.plateau_threshold,
            "plateau_threshold_mode": "abs",
            "plateau_min_lr": config.plateau_min_lr,
            "min_lr_early_stop_patience": config.min_lr_early_stop_patience,
            "min_lr_early_stop_threshold": config.plateau_threshold,
            "num_query_points": config.num_query_points,
            "global_seed": config.global_seed,
            "validation_query_seed": config.validation_query_seed,
            "validation_diffusion_seed": config.validation_diffusion_seed,
        },
        "validation_plan": {
            "version": VALIDATION_PLAN_VERSION,
            "query_plan_sha256": query_plan_sha256,
            "timestep_plan_sha256": timestep_plan_sha256,
        },
        "output_channel_names": list(train_data["output_channel_names"]),
        "branch_channel_names": list(train_data["branch_channel_names"]),
        "trunk_channel_names": list(train_data["trunk_channel_names"]),
        "y_normalizer": normalizer.state_dict(),
        "normalizer_weighting": "subdomain_balanced",
        "local_aspect_mean": local_aspect_mean,
        "local_aspect_std": local_aspect_std,
        "train_dataset_h5": str(config.train_dataset_path),
        "val_dataset_h5": str(config.val_dataset_path),
        "history": list(history),
    }



def _default_normalizer_cache_path(config: DiffusionTrainingConfig) -> Path:
    return (
        config.results_root
        / "cache"
        / "normalizers"
        / f"{config.train_dataset_path.stem}.subdomain_balanced_v1.json"
    ).resolve(strict=False)


def _normalizer_cache_fingerprint(
    *,
    train_path: Path,
    train_data: Mapping[str, Any],
    train_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    stat = train_path.stat()
    protocol_keys = (
        "dataset_role",
        "split_manifest_sha256",
        "dataset_builder_sha256",
        "generator_script_sha256",
        "dataset_random_seed",
    )
    return {
        "resolved_path": str(train_path.resolve(strict=False)),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "n_samples": int(len(train_data["samples"])),
        "output_channel_names": list(train_data["output_channel_names"]),
        "protocol": {
            key: train_protocol.get(key)
            for key in protocol_keys
            if key in train_protocol
        },
    }


def _write_normalizer_cache(
    path: Path,
    *,
    fingerprint: Mapping[str, Any],
    normalizer: FeatureNormalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    source: Mapping[str, Any],
) -> None:
    payload = {
        "cache_version": NORMALIZER_CACHE_VERSION,
        "normalizer_weighting": "subdomain_balanced",
        "train_dataset_fingerprint": dict(fingerprint),
        "target_mean": [float(value) for value in normalizer.mean.detach().cpu().tolist()],
        "target_std": [float(value) for value in normalizer.std.detach().cpu().tolist()],
        "local_aspect_mean": float(local_aspect_mean),
        "local_aspect_std": float(local_aspect_std),
        "source": dict(source),
        "created_at_utc": _utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _load_normalizer_cache(
    path: Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> tuple[FeatureNormalizer, float, float, Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("cache_version") != NORMALIZER_CACHE_VERSION:
        raise ValueError(
            f"Unsupported normalizer cache version in {path}: "
            f"{payload.get('cache_version')!r}"
        )
    if payload.get("normalizer_weighting") != "subdomain_balanced":
        raise ValueError(f"Normalizer cache has wrong weighting: {path}")
    if payload.get("train_dataset_fingerprint") != dict(expected_fingerprint):
        raise ValueError(
            "Normalizer cache does not match the current training dataset fingerprint: "
            f"{path}"
        )

    normalizer = FeatureNormalizer.from_statistics(
        payload["target_mean"], payload["target_std"]
    )
    local_aspect_mean = float(payload["local_aspect_mean"])
    local_aspect_std = float(payload["local_aspect_std"])
    if not math.isfinite(local_aspect_mean):
        raise ValueError("Cached local_aspect_mean is non-finite")
    if not math.isfinite(local_aspect_std) or local_aspect_std <= 0.0:
        raise ValueError("Cached local_aspect_std must be finite and positive")
    return normalizer, local_aspect_mean, local_aspect_std, payload


def _normalizer_from_checkpoint(
    checkpoint_path: Path,
    *,
    train_dataset_path: Path,
    output_channel_names: Sequence[str],
) -> tuple[FeatureNormalizer, float, float, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("training_protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "Normalizer source checkpoint uses a different training protocol"
        )
    stored_train_path = Path(
        str(_require_checkpoint_field(checkpoint, "train_dataset_h5"))
    ).expanduser().resolve(strict=False)
    if stored_train_path != train_dataset_path.resolve(strict=False):
        raise ValueError(
            "Normalizer source checkpoint references a different training HDF5"
        )
    if checkpoint.get("normalizer_weighting") != "subdomain_balanced":
        raise ValueError(
            "Normalizer source checkpoint is not subdomain-balanced"
        )
    stored_outputs = list(
        _require_checkpoint_field(checkpoint, "output_channel_names")
    )
    if stored_outputs != list(output_channel_names):
        raise ValueError(
            "Normalizer source checkpoint output channels do not match the dataset"
        )
    y_normalizer = _require_checkpoint_field(checkpoint, "y_normalizer")
    if not isinstance(y_normalizer, Mapping):
        raise TypeError("Checkpoint y_normalizer must be a mapping")
    normalizer = FeatureNormalizer.from_state_dict(y_normalizer)
    local_aspect_mean = float(
        _require_checkpoint_field(checkpoint, "local_aspect_mean")
    )
    local_aspect_std = float(
        _require_checkpoint_field(checkpoint, "local_aspect_std")
    )
    return normalizer, local_aspect_mean, local_aspect_std, {
        "kind": "completed_training_checkpoint",
        "checkpoint_path": _manifest_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(_require_checkpoint_field(checkpoint, "epoch")),
    }


def _load_or_prepare_normalizer(
    *,
    config: DiffusionTrainingConfig,
    train_data: Mapping[str, Any],
    train_protocol: Mapping[str, Any],
) -> tuple[FeatureNormalizer, float, float, Path, str]:
    cache_path = (
        _resolve_repo_path(config.normalizer_cache_path)
        if config.normalizer_cache_path is not None
        else _default_normalizer_cache_path(config)
    )
    fingerprint = _normalizer_cache_fingerprint(
        train_path=config.train_dataset_path,
        train_data=train_data,
        train_protocol=train_protocol,
    )

    if cache_path.exists() and not config.refresh_normalizer_cache:
        normalizer, aspect_mean, aspect_std, _ = _load_normalizer_cache(
            cache_path,
            expected_fingerprint=fingerprint,
        )
        return normalizer, aspect_mean, aspect_std, cache_path, "cache"

    if config.normalizer_from_checkpoint is not None:
        source_checkpoint = _resolve_repo_path(config.normalizer_from_checkpoint)
        _require_file(source_checkpoint, "Normalizer source checkpoint")
        normalizer, aspect_mean, aspect_std, source = _normalizer_from_checkpoint(
            source_checkpoint,
            train_dataset_path=config.train_dataset_path,
            output_channel_names=train_data["output_channel_names"],
        )
        source_kind = "checkpoint"
    else:
        normalizer, aspect_mean, aspect_std = fit_subdomain_balanced_normalizers(
            train_data
        )
        source = {"kind": "full_training_dataset_scan"}
        source_kind = "computed"

    _write_normalizer_cache(
        cache_path,
        fingerprint=fingerprint,
        normalizer=normalizer,
        local_aspect_mean=aspect_mean,
        local_aspect_std=aspect_std,
        source=source,
    )
    return normalizer, aspect_mean, aspect_std, cache_path, source_kind

def _validate_config(config: DiffusionTrainingConfig) -> None:
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
    if config.progress_interval < 0:
        raise ValueError("progress_interval must be non-negative; use 0 to disable")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative; use 0 to disable")
    if config.boundary_loss_weight < 0.0:
        raise ValueError("boundary_loss_weight must be non-negative")
    if config.scheduler not in {"none", "plateau"}:
        raise ValueError("scheduler must be 'none' or 'plateau'")
    if not 0.0 < config.plateau_factor < 1.0:
        raise ValueError("plateau_factor must lie in (0, 1)")
    if config.plateau_patience < 0:
        raise ValueError("plateau_patience must be non-negative")
    if config.plateau_threshold < 0.0:
        raise ValueError("plateau_threshold must be non-negative")
    if config.plateau_min_lr <= 0.0:
        raise ValueError("plateau_min_lr must be positive")
    if config.min_lr_early_stop_patience <= 0:
        raise ValueError("min_lr_early_stop_patience must be positive")
    if config.selection_metric not in {"interior_epsilon", "training_objective"}:
        raise ValueError(
            "selection_metric must be 'interior_epsilon' or 'training_objective'"
        )
    if config.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not 0.0 < config.beta_start < 1.0:
        raise ValueError("beta_start must lie in (0, 1)")
    if not 0.0 < config.beta_end < 1.0:
        raise ValueError("beta_end must lie in (0, 1)")
    if config.beta_start >= config.beta_end:
        raise ValueError("beta_start must be smaller than beta_end")


def run_training(config: DiffusionTrainingConfig) -> Path:
    """Train with random10 train data and deterministic random5 validation only."""

    config.train_dataset_path = _resolve_repo_path(config.train_dataset_path)
    config.val_dataset_path = _resolve_repo_path(config.val_dataset_path)
    config.results_root = _resolve_repo_path(config.results_root)
    config.source_notebook = _resolve_repo_path(config.source_notebook)
    if config.normalizer_cache_path is not None:
        config.normalizer_cache_path = _resolve_repo_path(config.normalizer_cache_path)
    if config.normalizer_from_checkpoint is not None:
        config.normalizer_from_checkpoint = _resolve_repo_path(
            config.normalizer_from_checkpoint
        )
    initialization_path = (
        _resolve_repo_path(config.initialize_from_checkpoint)
        if config.initialize_from_checkpoint is not None
        else None
    )

    _validate_config(config)
    device = _resolve_device(config.device)
    _require_file(config.train_dataset_path, "Training dataset")
    _require_file(config.val_dataset_path, "Validation dataset")
    _require_file(config.source_notebook, "Historical source notebook")
    if initialization_path is not None:
        _require_file(initialization_path, "Initialization checkpoint")

    set_global_seed(config.global_seed)

    startup_start = time.perf_counter()
    stage_start = time.perf_counter()
    print("[Startup] Loading training dataset:", config.train_dataset_path, flush=True)
    train_data = load_diffusion_dataset(config.train_dataset_path)
    print(
        f"[Startup] Training dataset loaded in {time.perf_counter() - stage_start:.2f} s",
        flush=True,
    )

    stage_start = time.perf_counter()
    print("[Startup] Loading validation dataset:", config.val_dataset_path, flush=True)
    val_data = load_diffusion_dataset(config.val_dataset_path)
    print(
        f"[Startup] Validation dataset loaded in {time.perf_counter() - stage_start:.2f} s",
        flush=True,
    )

    stage_start = time.perf_counter()
    train_protocol, val_protocol = _validate_dataset_pair(
        config.train_dataset_path,
        config.val_dataset_path,
        train_data,
        val_data,
    )
    print(
        f"[Startup] Train/validation protocol checks passed in "
        f"{time.perf_counter() - stage_start:.2f} s",
        flush=True,
    )

    stage_start = time.perf_counter()
    normalizer, local_aspect_mean, local_aspect_std, normalizer_cache_path, normalizer_source = (
        _load_or_prepare_normalizer(
            config=config,
            train_data=train_data,
            train_protocol=train_protocol,
        )
    )
    print(
        f"[Startup] Normalizer ready from {normalizer_source} in "
        f"{time.perf_counter() - stage_start:.2f} s",
        flush=True,
    )
    print("[Startup] Normalizer cache:", normalizer_cache_path, flush=True)
    print(
        "[Startup] target mean:",
        [float(value) for value in normalizer.mean.detach().cpu().tolist()],
        flush=True,
    )
    print(
        "[Startup] target std :",
        [float(value) for value in normalizer.std.detach().cpu().tolist()],
        flush=True,
    )

    train_dataset = DiffusionCellDataset(
        samples=train_data["samples"],
        sample_indices=None,
        n_query_points=config.num_query_points,
        random_query=True,
        target_normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        branch_channel_names=train_data["branch_channel_names"],
    )
    val_dataset = FixedValidationDataset(
        samples=val_data["samples"],
        metadata=val_data["metadata"],
        n_query_points=config.num_query_points,
        query_seed=config.validation_query_seed,
        target_normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        branch_channel_names=val_data["branch_channel_names"],
    )

    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(config.global_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_diffusion_batch,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_diffusion_batch,
    )

    branch_dim = len(train_data["branch_channel_names"])
    query_dim = len(train_data["trunk_channel_names"])
    target_dim = len(train_data["output_channel_names"])
    model, model_config = build_diffusion_model(branch_dim, query_dim, target_dim)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 382083:
        raise RuntimeError(
            f"Canonical model parameter count changed: {parameter_count} != 382083"
        )
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
        "prediction_target": "epsilon",
        "timestep_unit": "subdomain",
    }

    timestep_plan, timestep_plan_sha256 = _build_validation_timestep_plan(
        val_data["metadata"],
        timesteps=config.timesteps,
        diffusion_seed=config.validation_diffusion_seed,
    )
    query_plan_sha256 = val_dataset.query_plan_sha256
    print(
        f"[Startup] Data/model preparation ready in {time.perf_counter() - startup_start:.2f} s",
        flush=True,
    )

    initialization_metadata: dict[str, Any] = {
        "initialization_mode": "fresh",
        "source_checkpoint_path": None,
        "source_checkpoint_sha256": None,
        "source_epoch": None,
        "source_val_loss": None,
        "source_best_val_loss": None,
        "source_train_dataset_path": None,
        "source_val_dataset_path": None,
    }
    if initialization_path is not None:
        initialization_metadata = initialize_from_checkpoint(
            model,
            initialization_path,
            expected_model_config=model_config,
            expected_diffusion_config=diffusion_config,
            expected_output_channel_names=list(train_data["output_channel_names"]),
            expected_branch_channel_names=list(train_data["branch_channel_names"]),
            expected_trunk_channel_names=list(train_data["trunk_channel_names"]),
            expected_normalizer=normalizer,
            expected_local_aspect_mean=local_aspect_mean,
            expected_local_aspect_std=local_aspect_std,
            expected_train_dataset_path=config.train_dataset_path,
            expected_val_dataset_path=config.val_dataset_path,
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config)

    history: list[dict[str, Any]] = []
    global_step = 0
    run_id = _build_run_id(config)
    run_directory = (config.results_root / "train_diffusion" / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    best_checkpoint_path = run_directory / "diffusion_best.pt"
    latest_checkpoint_path = run_directory / "diffusion_latest.pt"
    history_path = run_directory / "training_history.csv"

    print("Canonical field-diffusion training")
    print("  protocol             :", PROTOCOL_VERSION)
    print("  train samples        :", len(train_dataset))
    print("  validation samples   :", len(val_dataset))
    print("  model parameters     :", f"{parameter_count:,}")
    print("  query cap            :", config.num_query_points)
    print("  batch size           :", config.batch_size)
    print("  boundary loss weight :", config.boundary_loss_weight)
    print("  selection metric     :", config.selection_metric)
    print("  scheduler            :", config.scheduler)
    if config.scheduler == "plateau":
        print(
            "  plateau protocol     :",
            f"factor={config.plateau_factor}, patience={config.plateau_patience}, "
            f"threshold={config.plateau_threshold} abs, min_lr={config.plateau_min_lr}",
        )
        print(
            "  min-LR early stop    :",
            f"{config.min_lr_early_stop_patience} validation checks without "
            f">{config.plateau_threshold:g} improvement",
        )
    print("  progress interval    :", config.progress_interval)
    print("  normalizer source    :", normalizer_source)
    print("  normalizer cache     :", normalizer_cache_path)
    print("  validation query SHA :", query_plan_sha256)
    print("  validation time SHA  :", timestep_plan_sha256)

    created_at_utc = _utc_now()
    prepared_manifest = _build_manifest(
        config=config,
        run_id=run_id,
        run_directory=run_directory,
        train_data=train_data,
        val_data=val_data,
        train_protocol=train_protocol,
        val_protocol=val_protocol,
        normalizer=normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        model_config=model_config,
        schedule=schedule,
        query_plan_sha256=query_plan_sha256,
        timestep_plan_sha256=timestep_plan_sha256,
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
    manifest_path: Path | None = None
    min_lr_best_val_loss: Optional[float] = None
    min_lr_bad_validation_checks = 0
    termination_reason = "epochs_ceiling_reached"

    try:
        update_manifest(
            run_directory,
            _build_manifest(
                config=config,
                run_id=run_id,
                run_directory=run_directory,
                train_data=train_data,
                val_data=val_data,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                model_config=model_config,
                schedule=schedule,
                query_plan_sha256=query_plan_sha256,
                timestep_plan_sha256=timestep_plan_sha256,
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

        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            print(
                f"[Epoch {epoch:04d}/{config.epochs:04d}] training started",
                flush=True,
            )
            train_stage_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                schedule=schedule,
                device=device,
                optimizer=optimizer,
                grad_clip=config.grad_clip,
                boundary_loss_weight=config.boundary_loss_weight,
                progress_interval=config.progress_interval,
                epoch=epoch,
            )
            train_stage_elapsed = time.perf_counter() - train_stage_start
            print(
                f"[Epoch {epoch:04d}/{config.epochs:04d}] training completed in "
                f"{train_stage_elapsed:.2f} s",
                flush=True,
            )
            global_step += int(train_metrics["steps"])

            if epoch % config.validation_interval == 0:
                print(
                    f"[Epoch {epoch:04d}/{config.epochs:04d}] deterministic validation started",
                    flush=True,
                )
                val_stage_start = time.perf_counter()
                val_metrics = evaluate_deterministic_validation(
                    model=model,
                    loader=val_loader,
                    schedule=schedule,
                    device=device,
                    metadata=val_data["metadata"],
                    timestep_plan=timestep_plan,
                    diffusion_seed=config.validation_diffusion_seed,
                    boundary_loss_weight=config.boundary_loss_weight,
                )
                val_stage_elapsed = time.perf_counter() - val_stage_start
                print(
                    f"[Epoch {epoch:04d}/{config.epochs:04d}] deterministic validation "
                    f"completed in {val_stage_elapsed:.2f} s",
                    flush=True,
                )
                val_loss = _selection_loss(val_metrics, config.selection_metric)
            else:
                val_metrics = {
                    "interior": float("nan"),
                    "boundary": float("nan"),
                    "objective": float("nan"),
                }
                val_loss = float("nan")

            learning_rate_before_scheduler = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None and math.isfinite(val_loss):
                scheduler.step(val_loss)

            learning_rate = float(optimizer.param_groups[0]["lr"])
            lr_reduced = learning_rate < learning_rate_before_scheduler - 1.0e-15

            at_min_lr = (
                config.scheduler == "plateau"
                and learning_rate <= config.plateau_min_lr * (1.0 + 1.0e-12)
            )
            if at_min_lr and math.isfinite(val_loss):
                if min_lr_best_val_loss is None:
                    min_lr_best_val_loss = float(val_loss)
                    min_lr_bad_validation_checks = 0
                elif val_loss < min_lr_best_val_loss - config.plateau_threshold:
                    min_lr_best_val_loss = float(val_loss)
                    min_lr_bad_validation_checks = 0
                else:
                    min_lr_bad_validation_checks += 1
            elif not at_min_lr:
                min_lr_best_val_loss = None
                min_lr_bad_validation_checks = 0
            elapsed = float(time.time() - epoch_start)
            samples_per_sec = float(train_metrics["samples"] / max(elapsed, 1.0e-12))
            peak_cuda_memory_gib = (
                float(torch.cuda.max_memory_allocated(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            )

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(train_metrics["objective"]),
                "train_interior_loss": float(train_metrics["interior"]),
                "train_boundary_loss": float(train_metrics["boundary"]),
                "val_loss": float(val_loss),
                "val_interior_loss": float(val_metrics["interior"]),
                "val_boundary_loss": float(val_metrics["boundary"]),
                "val_objective": float(val_metrics["objective"]),
                "grad_norm_mean": float(train_metrics["grad_norm_mean"]),
                "grad_norm_p95": float(train_metrics["grad_norm_p95"]),
                "grad_norm_max": float(train_metrics["grad_norm_max"]),
                "clip_fraction": float(train_metrics["clip_fraction"]),
                "elapsed_sec": elapsed,
                "samples_per_sec": samples_per_sec,
                "peak_cuda_memory_gib": peak_cuda_memory_gib,
                "lr": learning_rate,
                "lr_reduced": bool(lr_reduced),
                "at_min_lr": bool(at_min_lr),
                "min_lr_bad_validation_checks": int(min_lr_bad_validation_checks),
                "phase": "field_diffusion_baseline",
            }
            history.append(row)
            _write_history(history_path, history)

            is_best = math.isfinite(val_loss) and (
                run_best_val_loss is None or val_loss < run_best_val_loss
            )
            if is_best:
                run_best_val_loss = float(val_loss)

            if run_best_val_loss is None:
                payload_best_value = float("inf")
            else:
                payload_best_value = float(run_best_val_loss)

            payload = _checkpoint_payload(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_loss=val_loss,
                run_best_val_loss=payload_best_value,
                config=config,
                model_config=model_config,
                diffusion_config=diffusion_config,
                normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                train_data=train_data,
                query_plan_sha256=query_plan_sha256,
                timestep_plan_sha256=timestep_plan_sha256,
                initialization_metadata=initialization_metadata,
                history=history,
            )

            if epoch % config.checkpoint_interval == 0:
                save_diffusion_checkpoint(latest_checkpoint_path, payload)
            if is_best:
                save_diffusion_checkpoint(best_checkpoint_path, payload)

            last_completed_epoch = epoch
            print(
                f"Epoch {epoch:04d} | "
                f"train={train_metrics['objective']:.6f} "
                f"(int={train_metrics['interior']:.6f}, bc={train_metrics['boundary']:.6f}) | "
                f"val_select={val_loss:.6f} "
                f"(int={val_metrics['interior']:.6f}, bc={val_metrics['boundary']:.6f}, "
                f"obj={val_metrics['objective']:.6f}) | "
                f"grad_p95={train_metrics['grad_norm_p95']:.3e} | "
                f"clip={train_metrics['clip_fraction']:.3f} | "
                f"lr={learning_rate:.3e} | "
                f"{samples_per_sec:.1f} samples/s | "
                f"peak={peak_cuda_memory_gib:.2f} GiB",
                flush=True,
            )

            if lr_reduced:
                print(
                    f"[Scheduler] epoch {epoch:04d} | learning rate reduced "
                    f"{learning_rate_before_scheduler:.3e} -> {learning_rate:.3e}",
                    flush=True,
                )

            if (
                at_min_lr
                and min_lr_bad_validation_checks
                >= config.min_lr_early_stop_patience
            ):
                termination_reason = (
                    "min_lr_plateau_early_stop:"
                    f"lr={learning_rate:.6g},"
                    f"bad_validation_checks={min_lr_bad_validation_checks},"
                    f"threshold_abs={config.plateau_threshold:.6g}"
                )
                print(
                    f"[Early stop] minimum LR {learning_rate:.3e} reached and "
                    f"{min_lr_bad_validation_checks} validation checks passed "
                    f"without >{config.plateau_threshold:g} improvement.",
                    flush=True,
                )
                break

        if run_best_val_loss is None:
            raise RuntimeError(
                "No finite deterministic validation loss was observed; "
                "no best checkpoint was created"
            )

        finished_at_utc = _utc_now()
        manifest_path = update_manifest(
            run_directory,
            _build_manifest(
                config=config,
                run_id=run_id,
                run_directory=run_directory,
                train_data=train_data,
                val_data=val_data,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
                model_config=model_config,
                schedule=schedule,
                query_plan_sha256=query_plan_sha256,
                timestep_plan_sha256=timestep_plan_sha256,
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
                termination_reason=termination_reason,
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
                    train_data=train_data,
                    val_data=val_data,
                    train_protocol=train_protocol,
                    val_protocol=val_protocol,
                    normalizer=normalizer,
                    local_aspect_mean=local_aspect_mean,
                    local_aspect_std=local_aspect_std,
                    model_config=model_config,
                    schedule=schedule,
                    query_plan_sha256=query_plan_sha256,
                    timestep_plan_sha256=timestep_plan_sha256,
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
                    termination_reason="failed",
                ),
            )
        except Exception as manifest_exc:
            manifest_failure_type, manifest_failure_message = _short_failure(manifest_exc)
            print(
                "Warning: failed to publish failure manifest "
                f"({manifest_failure_type}: {manifest_failure_message})"
            )
        raise

    print("Completed canonical field-diffusion run:", run_directory)
    print("Manifest:", manifest_path)
    return run_directory


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the explicit training command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Train field diffusion on random10_train with deterministic random5_val. "
            "The test datasets are never loaded by this entrypoint."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start training; without this flag only the resolved configuration is printed",
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_path",
        default=str(DEFAULT_TRAIN_DATASET_PATH),
        help="Randomized training HDF5 path",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_path",
        default=str(DEFAULT_VAL_DATASET_PATH),
        help="Canonical randomized validation HDF5 path",
    )
    parser.add_argument(
        "--results-root",
        default=str(REPOSITORY_ROOT / "results"),
        help="Results root; runs are stored under results/train_diffusion/<run_id>",
    )
    parser.add_argument(
        "--source-notebook",
        default=str(DEFAULT_SOURCE_NOTEBOOK),
        help="Historical source notebook retained for provenance",
    )
    parser.add_argument("--device", default="cuda:1")

    parser.add_argument("--seed", dest="global_seed", type=int, default=0)
    parser.add_argument("--validation-query-seed", type=int, default=0)
    parser.add_argument("--validation-diffusion-seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Global gradient-norm clip threshold; 0 disables clipping",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=0.0,
        help="Weight for masked boundary epsilon denoising",
    )

    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="plateau",
        help="Use no LR scheduler or ReduceLROnPlateau",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-threshold", type=float, default=1.0e-4)
    parser.add_argument("--plateau-min-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--min-lr-early-stop-patience",
        type=int,
        default=10,
        help=(
            "After the scheduler reaches plateau-min-lr, stop after this many "
            "validation checks without an absolute improvement larger than "
            "plateau-threshold"
        ),
    )

    parser.add_argument(
        "--num-query-points",
        type=int,
        default=8192,
        help="Maximum query points sampled per subdomain; smaller subdomains use all points",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print training progress every N optimizer steps; 0 disables step progress",
    )
    parser.add_argument(
        "--normalizer-cache",
        dest="normalizer_cache_path",
        default=None,
        help=(
            "Optional normalizer cache JSON path. Default: "
            "results/cache/normalizers/<train-stem>.subdomain_balanced_v1.json"
        ),
    )
    parser.add_argument(
        "--normalizer-from-checkpoint",
        default=None,
        help=(
            "If the cache is absent, seed it from a compatible completed checkpoint "
            "instead of rescanning all training targets"
        ),
    )
    parser.add_argument(
        "--refresh-normalizer-cache",
        action="store_true",
        help="Ignore an existing normalizer cache and recreate it",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("interior_epsilon", "training_objective"),
        default="interior_epsilon",
        help="Metric used for best checkpoint selection and plateau scheduling",
    )

    parser.add_argument("--initialize-from-checkpoint", default=None)
    parser.add_argument("--checkpoint-tag", default="baseline")
    parser.add_argument("--run-id", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> DiffusionTrainingConfig:
    """Convert parser values into the typed training configuration."""

    return DiffusionTrainingConfig(
        train_dataset_path=_resolve_repo_path(args.train_dataset_path),
        val_dataset_path=_resolve_repo_path(args.val_dataset_path),
        results_root=_resolve_repo_path(args.results_root),
        source_notebook=_resolve_repo_path(args.source_notebook),
        device=args.device,
        global_seed=args.global_seed,
        validation_query_seed=args.validation_query_seed,
        validation_diffusion_seed=args.validation_diffusion_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        boundary_loss_weight=args.boundary_loss_weight,
        scheduler=args.scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_min_lr=args.plateau_min_lr,
        min_lr_early_stop_patience=args.min_lr_early_stop_patience,
        num_query_points=args.num_query_points,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        selection_metric=args.selection_metric,
        progress_interval=args.progress_interval,
        normalizer_cache_path=(
            _resolve_repo_path(args.normalizer_cache_path)
            if args.normalizer_cache_path is not None
            else None
        ),
        normalizer_from_checkpoint=(
            _resolve_repo_path(args.normalizer_from_checkpoint)
            if args.normalizer_from_checkpoint is not None
            else None
        ),
        refresh_normalizer_cache=args.refresh_normalizer_cache,
        initialize_from_checkpoint=(
            _resolve_repo_path(args.initialize_from_checkpoint)
            if args.initialize_from_checkpoint is not None
            else None
        ),
        checkpoint_tag=args.checkpoint_tag,
        run_id=args.run_id,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and optionally execute one canonical field-diffusion run."""

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
