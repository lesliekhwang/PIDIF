"""Progressive nested-schedule distillation for PIDIF field diffusion.

This script distills the frozen field-diffusion baseline in two stages:

    Stage 1: nested 20-NFE teacher -> nested 10-NFE student
    Stage 2: distilled nested 10-NFE teacher -> nested 5-NFE student

The 20-NFE anchor schedule is the same round-and-reverse DDIM schedule used by
``evaluate_diffusion_generation.py``.  The 10-NFE and 5-NFE schedules are
strict nested subsets of that anchor.  Every student model evaluation therefore
matches exactly two teacher model evaluations, including the final clean
projection.

Training uses direct state matching in normalized p/u/v space:

    teacher: x_t -> x_mid -> x_target
    student: x_t ----------> x_target

For the final slot, ``x_target`` is the clean projection.  No CFD truth loss,
physics loss, boundary auxiliary loss, or test data is used by the distillation
objective.  CFD truth from the canonical randomized validation set is used only
for deterministic rollout validation and checkpoint selection.

Without ``--run``, the script performs CPU-only protocol validation and prints
the exact distillation plan.  It does not create a result directory or start
training.
"""

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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pidiffusion.artifacts import update_manifest, write_manifest
from pidiffusion.data import FeatureNormalizer, normalize_diffusion_branch
from pidiffusion.diffusion import (
    build_ddim_timesteps,
    build_linear_schedule,
    ddim_step,
    final_clean_projection,
)
from pidiffusion.model import PointSetDiffusionDenoiser
from pidiffusion.provenance import file_identity, git_state, runtime_environment


PROTOCOL_VERSION = "field_diffusion_progressive_nested_distillation_v1"
CHECKPOINT_SCHEMA_VERSION = "progressive_distillation_checkpoint_v1"
MANIFEST_SCHEMA_VERSION = "pidiffusion_run_manifest_v1"
SCHEDULE_FAMILY = "progressive_nested20"
ANCHOR_NFE = 20
STAGE1_TEACHER_NFE = 20
STAGE1_STUDENT_NFE = 10
STAGE2_TEACHER_NFE = 10
STAGE2_STUDENT_NFE = 5
EXPECTED_PARAMETER_COUNT = 382_083
EXPECTED_OUTPUTS = ("pressure", "u", "v")
EXPECTED_TRUNK = ("x_local", "y_local")
EXPECTED_TRAIN_SAMPLES = 16_000
EXPECTED_VAL_SAMPLES = 1_000

DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "results/train_diffusion/field_diffusion_baseline_long_seed0"
    / "diffusion_best.pt"
)
DEFAULT_TRAIN_H5 = (
    REPO_ROOT
    / "channel_diffusion_dataset/deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random10_train.h5"
)
DEFAULT_VAL_H5 = (
    REPO_ROOT
    / "channel_diffusion_dataset/deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

FIELD_UNITS = ("Pa", "m/s", "m/s")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Config:
    base_checkpoint: Path
    train_h5: Path
    val_h5: Path
    results_root: Path
    device: str
    run_id: str | None
    seed: int
    validation_sampling_seed: int
    batch_size: int
    query_points: int
    epochs_stage1: int
    epochs_stage2: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    min_lr: float
    min_lr_early_stop_patience: int
    validation_every: int
    num_workers: int
    max_train_samples: int | None
    max_batches_per_epoch: int | None
    validation_max_samples: int | None
    progress_every_batches: int
    progress_every_validation_samples: int
    stage1_only: bool
    stage2_only: bool
    stage1_checkpoint: Path | None


@dataclass(frozen=True)
class H5Protocol:
    path: Path
    dataset_role: str
    split_role: str
    n_samples: int
    branch_channel_names: tuple[str, ...]
    trunk_channel_names: tuple[str, ...]
    output_channel_names: tuple[str, ...]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class StageSchedule:
    stage: int
    teacher_nfe: int
    student_nfe: int
    teacher_sources: tuple[int, ...]
    student_sources: tuple[int, ...]


@dataclass
class ValidationSummary:
    balanced_norm_mse: float
    balanced_rmse_pressure: float
    balanced_rmse_u: float
    balanced_rmse_v: float
    n_samples: int
    n_points: int


class RandomQueryH5Dataset(Dataset):
    """Lazy HDF5 dataset with deterministic epoch-dependent query sampling."""

    def __init__(
        self,
        path: Path,
        indices: Sequence[int],
        branch_channel_names: Sequence[str],
        normalizer: FeatureNormalizer,
        local_aspect_mean: float,
        local_aspect_std: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        query_points: int,
        base_seed: int,
        stage: int,
    ) -> None:
        self.path = Path(path)
        self.indices = tuple(int(x) for x in indices)
        self.branch_channel_names = tuple(branch_channel_names)
        self.normalizer = normalizer
        self.local_aspect_mean = float(local_aspect_mean)
        self.local_aspect_std = float(local_aspect_std)
        self.target_mean = np.asarray(target_mean, dtype=np.float32).reshape(1, -1)
        self.target_std = np.asarray(target_std, dtype=np.float32).reshape(1, -1)
        self.query_points = int(query_points)
        self.base_seed = int(base_seed)
        self.stage = int(stage)
        self.epoch = 0
        self._handle: h5py.File | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _query_indices(self, n_query: int, sample_index: int) -> np.ndarray:
        if self.query_points <= 0 or n_query <= self.query_points:
            return np.arange(n_query, dtype=np.int64)
        seed = stable_int_seed(
            f"query:{self.base_seed}:{self.stage}:{self.epoch}:{sample_index}"
        )
        rng = np.random.default_rng(seed)
        values = rng.choice(
            n_query,
            size=self.query_points,
            replace=False,
        )
        values.sort()
        return values.astype(np.int64, copy=False)

    def __getitem__(self, position: int):
        sample_index = self.indices[position]
        group = self._h5()["samples"][str(sample_index)]

        branch = group["branch"][:].astype(np.float32)
        query_ds = group["query"]
        target_ds = group["target"]
        selection = self._query_indices(len(query_ds), sample_index)
        query = query_ds[selection].astype(np.float32)
        target = target_ds[selection].astype(np.float32)

        branch = normalize_diffusion_branch(
            branch,
            branch_channel_names=list(self.branch_channel_names),
            target_normalizer=self.normalizer,
            local_aspect_mean=self.local_aspect_mean,
            local_aspect_std=self.local_aspect_std,
        ).astype(np.float32, copy=False)
        target = (
            (target - self.target_mean) / self.target_std
        ).astype(np.float32, copy=False)

        if not (
            np.isfinite(branch).all()
            and np.isfinite(query).all()
            and np.isfinite(target).all()
        ):
            raise ValueError(f"Sample {sample_index} contains non-finite values")

        return branch, query, target, sample_index


def collate_random_query_batch(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    branch_dim = batch[0][0].shape[1]
    max_branch = max(item[0].shape[0] for item in batch)

    branch = np.zeros(
        (batch_size, max_branch, branch_dim),
        dtype=np.float32,
    )
    branch_mask = np.zeros(
        (batch_size, max_branch),
        dtype=np.bool_,
    )

    query_parts = []
    target_parts = []
    batch_id_parts = []
    sample_indices = []

    for batch_id, (b, q, y, sample_index) in enumerate(batch):
        n_branch = b.shape[0]
        branch[batch_id, :n_branch] = b
        branch_mask[batch_id, :n_branch] = True
        query_parts.append(q)
        target_parts.append(y)
        batch_id_parts.append(
            np.full(len(q), batch_id, dtype=np.int64)
        )
        sample_indices.append(sample_index)

    return (
        torch.from_numpy(branch),
        torch.from_numpy(np.concatenate(query_parts, axis=0)),
        torch.from_numpy(np.concatenate(target_parts, axis=0)),
        torch.from_numpy(np.concatenate(batch_id_parts, axis=0)),
        torch.tensor(sample_indices, dtype=torch.long),
        torch.from_numpy(branch_mask),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def stable_int_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def stable_sample_seed(base_seed: int, sample_index: int) -> int:
    return stable_int_seed(f"{base_seed}:{sample_index}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(text: str) -> torch.device:
    device = torch.device(text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
    return device


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def require_checkpoint_field(checkpoint: Mapping[str, Any], name: str) -> Any:
    if name not in checkpoint:
        raise KeyError(f"Checkpoint missing required field: {name}")
    return checkpoint[name]


def read_h5_protocol(path: Path) -> H5Protocol:
    with h5py.File(path, "r") as handle:
        attrs = {key: decode_attr(value) for key, value in handle.attrs.items()}
        n_samples = int(attrs.get("n_samples", len(handle["samples"])))
        branch_names = tuple(
            decode_text(attrs["branch_channel_names"]).split("\n")
        )
        trunk_names = tuple(
            decode_text(attrs["trunk_channel_names"]).split("\n")
        )
        output_names = tuple(
            decode_text(attrs["output_channel_names"]).split("\n")
        )
        return H5Protocol(
            path=path,
            dataset_role=decode_text(attrs.get("dataset_role", "")),
            split_role=decode_text(attrs.get("split_role", "")),
            n_samples=n_samples,
            branch_channel_names=branch_names,
            trunk_channel_names=trunk_names,
            output_channel_names=output_names,
            attrs=attrs,
        )


def checkpoint_path_field(checkpoint: Mapping[str, Any], name: str) -> Path:
    return Path(str(require_checkpoint_field(checkpoint, name))).expanduser().resolve(
        strict=False
    )


def validate_base_protocol(
    checkpoint: Mapping[str, Any],
    train_h5: Path,
    val_h5: Path,
    train_protocol: H5Protocol,
    val_protocol: H5Protocol,
) -> None:
    if "test" in str(train_h5).lower() or "test" in str(val_h5).lower():
        raise ValueError("Test datasets are forbidden in distillation")

    if train_protocol.n_samples != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_SAMPLES} training samples, "
            f"got {train_protocol.n_samples}"
        )
    if val_protocol.n_samples != EXPECTED_VAL_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_SAMPLES} validation samples, "
            f"got {val_protocol.n_samples}"
        )
    if "test" in train_protocol.split_role.lower():
        raise ValueError("Training HDF5 advertises a test split")
    if "test" in val_protocol.split_role.lower():
        raise ValueError("Validation HDF5 advertises a test split")
    if val_protocol.split_role.lower() != "validation":
        raise ValueError(
            f"Expected validation split_role, got {val_protocol.split_role!r}"
        )

    for protocol in (train_protocol, val_protocol):
        if protocol.output_channel_names != EXPECTED_OUTPUTS:
            raise ValueError(
                f"Unexpected output channels in {protocol.path.name}: "
                f"{protocol.output_channel_names}"
            )
        if protocol.trunk_channel_names != EXPECTED_TRUNK:
            raise ValueError(
                f"Unexpected trunk channels in {protocol.path.name}: "
                f"{protocol.trunk_channel_names}"
            )

    if train_protocol.branch_channel_names != val_protocol.branch_channel_names:
        raise ValueError("Training/validation branch channels differ")

    if checkpoint_path_field(checkpoint, "train_dataset_h5") != train_h5:
        raise ValueError("Base checkpoint training HDF5 does not match requested HDF5")
    if checkpoint_path_field(checkpoint, "val_dataset_h5") != val_h5:
        raise ValueError("Base checkpoint validation HDF5 does not match requested HDF5")

    if tuple(require_checkpoint_field(checkpoint, "branch_channel_names")) != (
        train_protocol.branch_channel_names
    ):
        raise ValueError("Checkpoint/branch channel mismatch")
    if tuple(require_checkpoint_field(checkpoint, "trunk_channel_names")) != (
        train_protocol.trunk_channel_names
    ):
        raise ValueError("Checkpoint/trunk channel mismatch")
    if tuple(require_checkpoint_field(checkpoint, "output_channel_names")) != (
        train_protocol.output_channel_names
    ):
        raise ValueError("Checkpoint/output channel mismatch")

    if require_checkpoint_field(checkpoint, "normalizer_weighting") != (
        "subdomain_balanced"
    ):
        raise ValueError("Base checkpoint normalizer must be subdomain-balanced")

    diffusion = require_checkpoint_field(checkpoint, "diffusion_config")
    if diffusion.get("prediction_target") != "epsilon":
        raise ValueError("Only epsilon-prediction checkpoints are supported")
    if int(diffusion["T"]) != 1000:
        raise ValueError(f"Expected T=1000, got {diffusion['T']}")

    model = PointSetDiffusionDenoiser(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Model parameter count {n_params:,} != {EXPECTED_PARAMETER_COUNT:,}"
        )


def build_stage_schedules(total_diffusion_steps: int) -> tuple[StageSchedule, StageSchedule]:
    anchor = build_ddim_timesteps(
        ANCHOR_NFE,
        total_diffusion_steps,
        device="cpu",
    ).cpu().tolist()
    nested10 = anchor[::2]
    nested5 = anchor[::4]

    if len(anchor) != 20 or len(nested10) != 10 or len(nested5) != 5:
        raise RuntimeError("Unexpected nested schedule lengths")
    if nested10 != anchor[::2] or nested5 != nested10[::2]:
        raise RuntimeError("Nested schedule construction failed")

    stage1 = StageSchedule(
        stage=1,
        teacher_nfe=20,
        student_nfe=10,
        teacher_sources=tuple(int(x) for x in anchor),
        student_sources=tuple(int(x) for x in nested10),
    )
    stage2 = StageSchedule(
        stage=2,
        teacher_nfe=10,
        student_nfe=5,
        teacher_sources=tuple(int(x) for x in nested10),
        student_sources=tuple(int(x) for x in nested5),
    )
    validate_stage_schedule(stage1)
    validate_stage_schedule(stage2)
    return stage1, stage2


def validate_stage_schedule(schedule: StageSchedule) -> None:
    teacher = schedule.teacher_sources
    student = schedule.student_sources
    if len(teacher) != 2 * len(student):
        raise ValueError(
            f"Stage {schedule.stage}: teacher NFE must be exactly 2x student NFE"
        )
    if tuple(teacher[::2]) != tuple(student):
        raise ValueError(
            f"Stage {schedule.stage}: student sources are not nested teacher sources"
        )
    if any(a <= b for a, b in zip(teacher[:-1], teacher[1:])):
        raise ValueError(f"Stage {schedule.stage}: teacher sources are not descending")
    if any(a <= b for a, b in zip(student[:-1], student[1:])):
        raise ValueError(f"Stage {schedule.stage}: student sources are not descending")


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    frozen: bool,
) -> PointSetDiffusionDenoiser:
    model = PointSetDiffusionDenoiser(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if frozen:
        model.requires_grad_(False)
        model.eval()
    return model.to(device)


def normalizer_numpy(
    checkpoint: Mapping[str, Any],
) -> tuple[FeatureNormalizer, np.ndarray, np.ndarray]:
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["y_normalizer"])
    state = checkpoint["y_normalizer"]
    mean = np.asarray(state["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(state["std"], dtype=np.float32).reshape(-1)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Expected 3-field target normalizer")
    if not np.all(std > 0.0):
        raise ValueError("Target normalizer contains non-positive std")
    return normalizer, mean, std


def move_batch(batch, device: torch.device):
    branch, query, target, query_batch_id, sample_indices, branch_mask = batch
    return (
        branch.to(device, non_blocking=True),
        query.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        query_batch_id.to(device, non_blocking=True),
        sample_indices.to(device, non_blocking=True),
        branch_mask.to(device, non_blocking=True),
    )


def predict_epsilon_ragged(
    model: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    x_t: torch.Tensor,
    t_branch: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: torch.Tensor | None,
) -> torch.Tensor:
    t_query = t_branch[query_batch_id]
    return model(
        branch=branch,
        query=query,
        noisy_target=x_t,
        t_query=t_query,
        query_batch_id=query_batch_id,
        branch_mask=branch_mask,
    )


def q_sample_per_query(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t_query: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    alpha = alphas_cumprod[t_query].reshape(-1, 1)
    return torch.sqrt(alpha) * x0 + torch.sqrt(1.0 - alpha) * noise


def ddim_step_per_query(
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
    t_current_query: torch.Tensor,
    t_next_query: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_t = alphas_cumprod[t_current_query].reshape(-1, 1)
    alpha_next = alphas_cumprod[t_next_query].reshape(-1, 1)
    x0_pred = (
        x_t - torch.sqrt(1.0 - alpha_t) * epsilon_pred
    ) / torch.sqrt(alpha_t)
    x_next = (
        torch.sqrt(alpha_next) * x0_pred
        + torch.sqrt(1.0 - alpha_next) * epsilon_pred
    )
    return x_next, x0_pred


def final_projection_per_query(
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
    t_query: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    alpha_t = alphas_cumprod[t_query].reshape(-1, 1)
    return (
        x_t - torch.sqrt(1.0 - alpha_t) * epsilon_pred
    ) / torch.sqrt(alpha_t)


def stage_slot_tensors(
    stage_schedule: StageSchedule,
    slot_branch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = slot_branch.device
    teacher = torch.tensor(
        stage_schedule.teacher_sources,
        dtype=torch.long,
        device=device,
    )
    student = torch.tensor(
        stage_schedule.student_sources,
        dtype=torch.long,
        device=device,
    )

    t_current = student[slot_branch]
    t_mid = teacher[2 * slot_branch + 1]
    final_slot = slot_branch == (stage_schedule.student_nfe - 1)

    next_slot = torch.clamp(
        slot_branch + 1,
        max=stage_schedule.student_nfe - 1,
    )
    t_end = student[next_slot]
    # The final student slot maps directly to clean.  Its DDIM alternative
    # branch is discarded, but keep the placeholder transition numerically
    # well-posed by setting t_end=t_mid instead of constructing a reverse
    # step back toward a noisier timestep.
    t_end = torch.where(final_slot, t_mid, t_end)
    return t_current, t_mid, t_end, final_slot


@torch.no_grad()
def teacher_two_eval_target(
    teacher: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    x_t: torch.Tensor,
    t_current_branch: torch.Tensor,
    t_mid_branch: torch.Tensor,
    t_end_branch: torch.Tensor,
    final_slot_branch: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: torch.Tensor | None,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    eps1 = predict_epsilon_ragged(
        teacher,
        branch,
        query,
        x_t,
        t_current_branch,
        query_batch_id,
        branch_mask,
    )
    x_mid, _ = ddim_step_per_query(
        x_t,
        eps1,
        t_current_branch[query_batch_id],
        t_mid_branch[query_batch_id],
        alphas_cumprod,
    )

    eps2 = predict_epsilon_ragged(
        teacher,
        branch,
        query,
        x_mid,
        t_mid_branch,
        query_batch_id,
        branch_mask,
    )
    x_end, _ = ddim_step_per_query(
        x_mid,
        eps2,
        t_mid_branch[query_batch_id],
        t_end_branch[query_batch_id],
        alphas_cumprod,
    )
    x_clean = final_projection_per_query(
        x_mid,
        eps2,
        t_mid_branch[query_batch_id],
        alphas_cumprod,
    )
    final_query = final_slot_branch[query_batch_id].reshape(-1, 1)
    return torch.where(final_query, x_clean, x_end)


def student_one_eval_state(
    student: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    x_t: torch.Tensor,
    t_current_branch: torch.Tensor,
    t_end_branch: torch.Tensor,
    final_slot_branch: torch.Tensor,
    query_batch_id: torch.Tensor,
    branch_mask: torch.Tensor | None,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    eps = predict_epsilon_ragged(
        student,
        branch,
        query,
        x_t,
        t_current_branch,
        query_batch_id,
        branch_mask,
    )
    x_end, _ = ddim_step_per_query(
        x_t,
        eps,
        t_current_branch[query_batch_id],
        t_end_branch[query_batch_id],
        alphas_cumprod,
    )
    x_clean = final_projection_per_query(
        x_t,
        eps,
        t_current_branch[query_batch_id],
        alphas_cumprod,
    )
    final_query = final_slot_branch[query_batch_id].reshape(-1, 1)
    return torch.where(final_query, x_clean, x_end)


def subdomain_balanced_state_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    query_batch_id: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    point_loss = ((prediction - target) ** 2).mean(dim=-1)
    loss_sum = prediction.new_zeros(batch_size)
    count = prediction.new_zeros(batch_size)
    loss_sum.scatter_add_(0, query_batch_id, point_loss)
    count.scatter_add_(0, query_batch_id, torch.ones_like(point_loss))
    return (loss_sum / count.clamp_min(1.0)).mean()


def validate_vectorized_ddim_algebra(alphas_cumprod: torch.Tensor) -> None:
    """Check vectorized training algebra against the frozen scalar DDIM core."""
    generator = torch.Generator(device=alphas_cumprod.device)
    generator.manual_seed(12345)
    x_t = torch.randn((7, 3), generator=generator, device=alphas_cumprod.device)
    epsilon = torch.randn((7, 3), generator=generator, device=alphas_cumprod.device)
    t_current = 789
    t_next = 578
    current_q = torch.full((7,), t_current, dtype=torch.long, device=x_t.device)
    next_q = torch.full((7,), t_next, dtype=torch.long, device=x_t.device)

    x_vec, x0_vec = ddim_step_per_query(
        x_t, epsilon, current_q, next_q, alphas_cumprod
    )
    x_core, x0_core = ddim_step(
        x_t=x_t,
        epsilon_pred=epsilon,
        t_current=t_current,
        t_next=t_next,
        alphas_cumprod=alphas_cumprod,
    )
    if not torch.allclose(x_vec, x_core, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("Vectorized DDIM transition does not match diffusion core")
    if not torch.allclose(x0_vec, x0_core, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("Vectorized x0 prediction does not match diffusion core")

    clean_vec = final_projection_per_query(
        x_t, epsilon, current_q, alphas_cumprod
    )
    clean_core = final_clean_projection(
        x_t=x_t,
        epsilon_pred=epsilon,
        timestep=t_current,
        alphas_cumprod=alphas_cumprod,
    )
    if not torch.allclose(clean_vec, clean_core, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("Vectorized clean projection does not match diffusion core")


def train_one_epoch(
    *,
    teacher: PointSetDiffusionDenoiser,
    student: PointSetDiffusionDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    stage_schedule: StageSchedule,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
    grad_clip: float,
    noise_generator: torch.Generator,
    max_batches: int | None,
    epoch: int,
    progress_every_batches: int,
) -> tuple[float, float, int]:
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_samples = 0
    grad_norm_max = 0.0
    n_batches = 0
    planned_batches = len(loader)
    if max_batches is not None:
        planned_batches = min(planned_batches, max_batches)
    epoch_started = time.perf_counter()

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        (
            branch,
            query,
            target,
            query_batch_id,
            _sample_indices,
            branch_mask,
        ) = move_batch(raw_batch, device)
        batch_size = int(branch.shape[0])

        slot_branch = torch.randint(
            low=0,
            high=stage_schedule.student_nfe,
            size=(batch_size,),
            device=device,
            generator=noise_generator,
        )
        (
            t_current_branch,
            t_mid_branch,
            t_end_branch,
            final_slot_branch,
        ) = stage_slot_tensors(stage_schedule, slot_branch)

        noise = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=device,
            generator=noise_generator,
        )
        x_t = q_sample_per_query(
            target,
            noise,
            t_current_branch[query_batch_id],
            alphas_cumprod,
        )

        teacher_target = teacher_two_eval_target(
            teacher=teacher,
            branch=branch,
            query=query,
            x_t=x_t,
            t_current_branch=t_current_branch,
            t_mid_branch=t_mid_branch,
            t_end_branch=t_end_branch,
            final_slot_branch=final_slot_branch,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
            alphas_cumprod=alphas_cumprod,
        )

        optimizer.zero_grad(set_to_none=True)
        student_state = student_one_eval_state(
            student=student,
            branch=branch,
            query=query,
            x_t=x_t,
            t_current_branch=t_current_branch,
            t_end_branch=t_end_branch,
            final_slot_branch=final_slot_branch,
            query_batch_id=query_batch_id,
            branch_mask=branch_mask,
            alphas_cumprod=alphas_cumprod,
        )
        loss = subdomain_balanced_state_mse(
            student_state,
            teacher_target,
            query_batch_id,
            batch_size,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite distillation loss")

        loss.backward()

        grad_sq = 0.0
        for parameter in student.parameters():
            if parameter.grad is not None:
                value = float(parameter.grad.detach().norm(2).item())
                grad_sq += value * value
        grad_norm = math.sqrt(grad_sq)
        grad_norm_max = max(grad_norm_max, grad_norm)

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        n_batches += 1

        if (
            progress_every_batches > 0
            and (
                n_batches % progress_every_batches == 0
                or n_batches == planned_batches
            )
        ):
            elapsed = time.perf_counter() - epoch_started
            avg_loss = total_loss / total_samples
            seconds_per_batch = elapsed / n_batches
            remaining = max(planned_batches - n_batches, 0)
            eta = remaining * seconds_per_batch
            print(
                f"[Stage {stage_schedule.stage}][Epoch {epoch:03d}] "
                f"train batch {n_batches:04d}/{planned_batches:04d} | "
                f"avg_state_mse={avg_loss:.8g} | "
                f"elapsed={elapsed / 60.0:.1f} min | "
                f"eta={eta / 60.0:.1f} min",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("No training samples were processed")

    return total_loss / total_samples, grad_norm_max, n_batches


def read_full_sample(
    handle: h5py.File,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = handle["samples"][str(sample_index)]
    branch = group["branch"][:].astype(np.float32)
    query = group["query"][:].astype(np.float32)
    target = group["target"][:].astype(np.float32)
    if target.shape != (query.shape[0], 3):
        raise ValueError(f"Invalid validation target shape: {target.shape}")
    return branch, query, target


@torch.inference_mode()
def predict_epsilon_single(
    model: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    x_t: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    n_query = int(query.shape[0])
    return model(
        branch=branch,
        query=query,
        noisy_target=x_t,
        t_query=torch.full(
            (n_query,), timestep, dtype=torch.long, device=query.device
        ),
        query_batch_id=torch.zeros(
            n_query, dtype=torch.long, device=query.device
        ),
        branch_mask=None,
    )


@torch.inference_mode()
def sample_student_schedule(
    model: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    initial_noise: torch.Tensor,
    student_sources: Sequence[int],
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    x_t = initial_noise.clone()
    for current, next_ in zip(student_sources[:-1], student_sources[1:]):
        epsilon = predict_epsilon_single(
            model,
            branch,
            query,
            x_t,
            int(current),
        )
        x_t, _ = ddim_step(
            x_t=x_t,
            epsilon_pred=epsilon,
            t_current=int(current),
            t_next=int(next_),
            alphas_cumprod=alphas_cumprod,
        )

    final_t = int(student_sources[-1])
    epsilon = predict_epsilon_single(model, branch, query, x_t, final_t)
    return final_clean_projection(
        x_t=x_t,
        epsilon_pred=epsilon,
        timestep=final_t,
        alphas_cumprod=alphas_cumprod,
    )


@torch.inference_mode()
def validate_rollout(
    *,
    model: PointSetDiffusionDenoiser,
    val_h5: Path,
    val_indices: Sequence[int],
    branch_channel_names: Sequence[str],
    normalizer: FeatureNormalizer,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    local_aspect_mean: float,
    local_aspect_std: float,
    student_sources: Sequence[int],
    alphas_cumprod: torch.Tensor,
    device: torch.device,
    sampling_seed: int,
    stage: int,
    epoch: int,
    progress_every_validation_samples: int,
) -> ValidationSummary:
    model.eval()
    norm_mse_sum = 0.0
    field_mse_sum = np.zeros(3, dtype=np.float64)
    n_samples = 0
    n_points = 0

    target_mean = np.asarray(target_mean, dtype=np.float32).reshape(1, 3)
    target_std = np.asarray(target_std, dtype=np.float32).reshape(1, 3)
    validation_started = time.perf_counter()
    total_validation_samples = len(val_indices)
    print(
        f"[Stage {stage}][Epoch {epoch:03d}] validation start | "
        f"samples={total_validation_samples} | "
        f"student_nfe={len(student_sources)}",
        flush=True,
    )

    with h5py.File(val_h5, "r") as handle:
        for position, sample_index in enumerate(val_indices, start=1):
            branch_raw, query_raw, truth = read_full_sample(handle, sample_index)
            branch_raw = normalize_diffusion_branch(
                branch_raw,
                branch_channel_names=list(branch_channel_names),
                target_normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
            ).astype(np.float32, copy=False)

            branch = torch.from_numpy(branch_raw).unsqueeze(0).to(device)
            query = torch.from_numpy(query_raw).to(device)

            generator = torch.Generator(device=device)
            generator.manual_seed(stable_sample_seed(sampling_seed, sample_index))
            initial_noise = torch.randn(
                (len(query_raw), 3),
                dtype=query.dtype,
                device=device,
                generator=generator,
            )

            pred_norm = sample_student_schedule(
                model,
                branch,
                query,
                initial_noise,
                student_sources,
                alphas_cumprod,
            )
            pred_norm_np = pred_norm.cpu().numpy().astype(np.float32)
            truth_norm = (truth - target_mean) / target_std
            norm_mse_sum += float(np.mean((pred_norm_np - truth_norm) ** 2))

            pred_phys = pred_norm_np * target_std + target_mean
            error = pred_phys - truth
            field_mse_sum += np.mean(error.astype(np.float64) ** 2, axis=0)
            n_samples += 1
            n_points += len(query_raw)

            if (
                progress_every_validation_samples > 0
                and (
                    position % progress_every_validation_samples == 0
                    or position == total_validation_samples
                )
            ):
                elapsed = time.perf_counter() - validation_started
                seconds_per_sample = elapsed / position
                remaining = max(total_validation_samples - position, 0)
                eta = remaining * seconds_per_sample
                print(
                    f"[Stage {stage}][Epoch {epoch:03d}] "
                    f"validation {position:04d}/{total_validation_samples:04d} | "
                    f"elapsed={elapsed:.1f} s | eta={eta:.1f} s",
                    flush=True,
                )

    if n_samples == 0:
        raise RuntimeError("No validation samples were evaluated")

    rmse = np.sqrt(field_mse_sum / n_samples)
    return ValidationSummary(
        balanced_norm_mse=norm_mse_sum / n_samples,
        balanced_rmse_pressure=float(rmse[0]),
        balanced_rmse_u=float(rmse[1]),
        balanced_rmse_v=float(rmse[2]),
        n_samples=n_samples,
        n_points=n_points,
    )


def choose_indices(n_samples: int, max_samples: int | None) -> tuple[int, ...]:
    if max_samples is None or max_samples >= n_samples:
        return tuple(range(n_samples))
    if max_samples <= 0:
        raise ValueError("Sample limit must be positive")
    values = np.linspace(0, n_samples - 1, num=max_samples).round().astype(np.int64)
    values = np.unique(values)
    if len(values) != max_samples:
        raise RuntimeError("Sample selection produced duplicates")
    return tuple(int(x) for x in values)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def stage_checkpoint_payload(
    *,
    base_checkpoint: Mapping[str, Any],
    model: PointSetDiffusionDenoiser,
    stage_schedule: StageSchedule,
    stage: int,
    epoch: int,
    global_step: int,
    validation: ValidationSummary,
    teacher_checkpoint_path: Path,
    initialization_checkpoint_path: Path,
    train_h5: Path,
    val_h5: Path,
    config: Config,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_protocol_version": PROTOCOL_VERSION,
        "checkpoint_type": "progressive_distillation_student",
        "schedule_family": SCHEDULE_FAMILY,
        "stage": int(stage),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "model_config": dict(base_checkpoint["model_config"]),
        "diffusion_config": dict(base_checkpoint["diffusion_config"]),
        "branch_channel_names": tuple(base_checkpoint["branch_channel_names"]),
        "trunk_channel_names": tuple(base_checkpoint["trunk_channel_names"]),
        "output_channel_names": tuple(base_checkpoint["output_channel_names"]),
        "y_normalizer": dict(base_checkpoint["y_normalizer"]),
        "local_aspect_mean": float(base_checkpoint["local_aspect_mean"]),
        "local_aspect_std": float(base_checkpoint["local_aspect_std"]),
        "normalizer_weighting": base_checkpoint["normalizer_weighting"],
        "train_dataset_h5": str(train_h5),
        "val_dataset_h5": str(val_h5),
        "teacher_checkpoint": str(teacher_checkpoint_path),
        "initialization_checkpoint": str(initialization_checkpoint_path),
        "teacher_nfe": stage_schedule.teacher_nfe,
        "student_nfe": stage_schedule.student_nfe,
        "teacher_timesteps": list(stage_schedule.teacher_sources),
        "student_timesteps": list(stage_schedule.student_sources),
        "distillation_objective": {
            "name": "subdomain_balanced_reverse_state_mse",
            "teacher_evaluations_per_student_evaluation": 2,
            "teacher_target": "two deterministic DDIM evaluations",
            "student_target": "one deterministic DDIM evaluation",
            "final_slot": "two-eval teacher clean projection matched by one-eval student clean projection",
            "truth_loss": False,
            "physics_loss": False,
            "boundary_auxiliary_loss": False,
            "space": "normalized pressure_u_v_state",
        },
        "validation_selection_metric": "subdomain_balanced_normalized_rollout_mse",
        "val_rollout_balanced_norm_mse": float(validation.balanced_norm_mse),
        "val_rollout_balanced_rmse_pressure": float(
            validation.balanced_rmse_pressure
        ),
        "val_rollout_balanced_rmse_u": float(validation.balanced_rmse_u),
        "val_rollout_balanced_rmse_v": float(validation.balanced_rmse_v),
        "seed": config.seed,
        "validation_sampling_seed": config.validation_sampling_seed,
        "query_sampling_policy": {
            "train_query_points": config.query_points,
            "deterministic_per_sample_epoch": True,
            "validation": "all query points",
        },
    }


def build_manifest(
    *,
    run_id: str,
    status: str,
    created_at: str,
    started_at: str | None,
    finished_at: str | None,
    config: Config,
    base_checkpoint_path: Path,
    train_protocol: H5Protocol,
    val_protocol: H5Protocol,
    stage1_schedule: StageSchedule,
    stage2_schedule: StageSchedule,
    outputs: Mapping[str, Any],
    failure: str | None = None,
) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        REPO_ROOT / "pidiffusion/data.py",
        REPO_ROOT / "pidiffusion/diffusion.py",
        REPO_ROOT / "pidiffusion/model.py",
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": created_at,
        "status": status,
        "git": git_state(REPO_ROOT),
        "source_files": [file_identity(path) for path in source_files],
        "checkpoint": {
            **file_identity(base_checkpoint_path),
            "role": "formal_base_field_diffusion_teacher",
        },
        "dataset": {
            "train": {
                **file_identity(config.train_h5),
                "dataset_role": train_protocol.dataset_role,
                "split_role": train_protocol.split_role,
                "n_samples": train_protocol.n_samples,
            },
            "validation": {
                **file_identity(config.val_h5),
                "dataset_role": val_protocol.dataset_role,
                "split_role": val_protocol.split_role,
                "n_samples": val_protocol.n_samples,
            },
            "test_access": "disabled",
        },
        "randomness": {
            "global_seed": config.seed,
            "training_noise_seed": config.seed,
            "validation_sampling_seed": config.validation_sampling_seed,
            "dataloader_seed_policy": "derived_from_global_seed_and_stage",
            "query_sampling_seed_policy": "derived_from_global_seed_stage_epoch_and_sample_index",
            "num_workers": config.num_workers,
        },
        "protocol": {
            "version": PROTOCOL_VERSION,
            "schedule_family": SCHEDULE_FAMILY,
            "anchor_nfe": ANCHOR_NFE,
            "stage1": {
                "teacher_nfe": stage1_schedule.teacher_nfe,
                "student_nfe": stage1_schedule.student_nfe,
                "teacher_timesteps": list(stage1_schedule.teacher_sources),
                "student_timesteps": list(stage1_schedule.student_sources),
            },
            "stage2": {
                "teacher_nfe": stage2_schedule.teacher_nfe,
                "student_nfe": stage2_schedule.student_nfe,
                "teacher_timesteps": list(stage2_schedule.teacher_sources),
                "student_timesteps": list(stage2_schedule.student_sources),
            },
            "objective": "direct normalized reverse-state matching",
            "teacher_gradient": "disabled",
            "student_initialization": "exact teacher weight copy at each stage",
            "truth_training_loss": False,
            "physics_training_loss": False,
            "boundary_auxiliary_loss": False,
            "checkpoint_selection": "deterministic validation rollout normalized MSE",
        },
        "training": {
            "seed": config.seed,
            "validation_sampling_seed": config.validation_sampling_seed,
            "batch_size": config.batch_size,
            "query_points": config.query_points,
            "epochs_stage1": config.epochs_stage1,
            "epochs_stage2": config.epochs_stage2,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": config.scheduler_factor,
                "patience": config.scheduler_patience,
                "threshold": config.scheduler_threshold,
                "min_lr": config.min_lr,
            },
            "min_lr_early_stop_patience": config.min_lr_early_stop_patience,
            "validation_every": config.validation_every,
            "max_train_samples": config.max_train_samples,
            "max_batches_per_epoch": config.max_batches_per_epoch,
            "validation_max_samples": config.validation_max_samples,
            "progress_every_batches": config.progress_every_batches,
            "progress_every_validation_samples": config.progress_every_validation_samples,
            "stage1_only": config.stage1_only,
        },
        "environment": runtime_environment(),
        "outputs": dict(outputs),
        "lifecycle": {
            "created_at_utc": created_at,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "failure": failure,
        },
    }


def stage_run(
    *,
    stage_schedule: StageSchedule,
    base_checkpoint: Mapping[str, Any],
    teacher_checkpoint_path: Path,
    initialization_checkpoint_path: Path,
    train_protocol: H5Protocol,
    val_protocol: H5Protocol,
    config: Config,
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    run_dir: Path,
    history_rows: list[dict[str, Any]],
    global_step_start: int,
) -> tuple[Path, int]:
    teacher_checkpoint = load_checkpoint(teacher_checkpoint_path)
    init_checkpoint = load_checkpoint(initialization_checkpoint_path)

    teacher = build_model_from_checkpoint(teacher_checkpoint, device, frozen=True)
    student = build_model_from_checkpoint(init_checkpoint, device, frozen=False)

    normalizer, target_mean, target_std = normalizer_numpy(base_checkpoint)
    train_indices = choose_indices(train_protocol.n_samples, config.max_train_samples)
    val_indices = choose_indices(val_protocol.n_samples, config.validation_max_samples)

    dataset = RandomQueryH5Dataset(
        path=config.train_h5,
        indices=train_indices,
        branch_channel_names=train_protocol.branch_channel_names,
        normalizer=normalizer,
        local_aspect_mean=float(base_checkpoint["local_aspect_mean"]),
        local_aspect_std=float(base_checkpoint["local_aspect_std"]),
        target_mean=target_mean,
        target_std=target_std,
        query_points=config.query_points,
        base_seed=config.seed,
        stage=stage_schedule.stage,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        stable_int_seed(f"loader:{config.seed}:stage:{stage_schedule.stage}")
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_random_query_batch,
        pin_memory=(device.type == "cuda"),
        generator=loader_generator,
        persistent_workers=(config.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode="abs",
        min_lr=config.min_lr,
    )

    stage_epochs = (
        config.epochs_stage1 if stage_schedule.stage == 1 else config.epochs_stage2
    )
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(
        stable_int_seed(f"noise:{config.seed}:stage:{stage_schedule.stage}")
    )

    latest_path = run_dir / f"stage{stage_schedule.stage}_latest.pt"
    best_path = run_dir / f"stage{stage_schedule.stage}_best.pt"
    history_path = run_dir / "distillation_history.csv"

    best_metric = float("inf")
    bad_at_min_lr = 0
    global_step = int(global_step_start)

    for epoch in range(1, stage_epochs + 1):
        dataset.set_epoch(epoch)
        train_loss, grad_norm_max, n_batches = train_one_epoch(
            teacher=teacher,
            student=student,
            loader=loader,
            optimizer=optimizer,
            stage_schedule=stage_schedule,
            alphas_cumprod=alphas_cumprod,
            device=device,
            grad_clip=config.grad_clip,
            noise_generator=noise_generator,
            max_batches=config.max_batches_per_epoch,
            epoch=epoch,
            progress_every_batches=config.progress_every_batches,
        )
        global_step += n_batches

        if epoch % config.validation_every != 0:
            print(
                f"[Stage {stage_schedule.stage}] epoch={epoch:03d} "
                f"train_state_mse={train_loss:.8g} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
            continue

        validation = validate_rollout(
            model=student,
            val_h5=config.val_h5,
            val_indices=val_indices,
            branch_channel_names=val_protocol.branch_channel_names,
            normalizer=normalizer,
            target_mean=target_mean,
            target_std=target_std,
            local_aspect_mean=float(base_checkpoint["local_aspect_mean"]),
            local_aspect_std=float(base_checkpoint["local_aspect_std"]),
            student_sources=stage_schedule.student_sources,
            alphas_cumprod=alphas_cumprod,
            device=device,
            sampling_seed=config.validation_sampling_seed,
            stage=stage_schedule.stage,
            epoch=epoch,
            progress_every_validation_samples=(
                config.progress_every_validation_samples
            ),
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "stage": stage_schedule.stage,
            "epoch": epoch,
            "global_step": global_step,
            "teacher_nfe": stage_schedule.teacher_nfe,
            "student_nfe": stage_schedule.student_nfe,
            "train_state_mse": train_loss,
            "val_rollout_balanced_norm_mse": validation.balanced_norm_mse,
            "val_pressure_balanced_rmse_pa": validation.balanced_rmse_pressure,
            "val_u_balanced_rmse_mps": validation.balanced_rmse_u,
            "val_v_balanced_rmse_mps": validation.balanced_rmse_v,
            "val_n_samples": validation.n_samples,
            "val_n_points": validation.n_points,
            "learning_rate": current_lr,
            "grad_norm_max": grad_norm_max,
        }
        history_rows.append(row)
        write_history(history_path, history_rows)

        payload = stage_checkpoint_payload(
            base_checkpoint=base_checkpoint,
            model=student,
            stage_schedule=stage_schedule,
            stage=stage_schedule.stage,
            epoch=epoch,
            global_step=global_step,
            validation=validation,
            teacher_checkpoint_path=teacher_checkpoint_path,
            initialization_checkpoint_path=initialization_checkpoint_path,
            train_h5=config.train_h5,
            val_h5=config.val_h5,
            config=config,
        )
        atomic_torch_save(payload, latest_path)

        improved = (
            validation.balanced_norm_mse
            < best_metric - config.scheduler_threshold
        )
        if improved:
            best_metric = validation.balanced_norm_mse
            atomic_torch_save(payload, best_path)

        scheduler.step(validation.balanced_norm_mse)
        new_lr = float(optimizer.param_groups[0]["lr"])

        at_min_lr = new_lr <= config.min_lr * (1.0 + 1.0e-12)
        if at_min_lr and not improved:
            bad_at_min_lr += 1
        else:
            bad_at_min_lr = 0

        print(
            f"[Stage {stage_schedule.stage}] epoch={epoch:03d} "
            f"train={train_loss:.8g} "
            f"val_norm_mse={validation.balanced_norm_mse:.8g} | "
            f"p={validation.balanced_rmse_pressure:.6g} Pa "
            f"u={validation.balanced_rmse_u:.6g} m/s "
            f"v={validation.balanced_rmse_v:.6g} m/s | "
            f"lr={new_lr:.3g} "
            f"best={best_metric:.8g}",
            flush=True,
        )

        if bad_at_min_lr >= config.min_lr_early_stop_patience:
            print(
                f"[Stage {stage_schedule.stage}] early stop: "
                f"min_lr={config.min_lr:g}, "
                f"bad_validation_checks={bad_at_min_lr}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            f"Stage {stage_schedule.stage} produced no best checkpoint"
        )

    del teacher
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_path, global_step



def validate_stage1_checkpoint_for_stage2(
    *,
    checkpoint_path: Path,
    base_checkpoint: Mapping[str, Any],
    config: Config,
    stage1_schedule: StageSchedule,
) -> None:
    """Fail closed unless the supplied checkpoint is the formal Stage-1 student."""
    checkpoint = load_checkpoint(checkpoint_path)

    expected_scalars = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_protocol_version": PROTOCOL_VERSION,
        "checkpoint_type": "progressive_distillation_student",
        "schedule_family": SCHEDULE_FAMILY,
        "stage": 1,
        "teacher_nfe": stage1_schedule.teacher_nfe,
        "student_nfe": stage1_schedule.student_nfe,
    }
    for key, expected in expected_scalars.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise ValueError(
                f"Invalid Stage-1 checkpoint field {key!r}: "
                f"expected {expected!r}, got {actual!r}"
            )

    expected_teacher_timesteps = list(stage1_schedule.teacher_sources)
    expected_student_timesteps = list(stage1_schedule.student_sources)

    if list(checkpoint.get("teacher_timesteps", [])) != expected_teacher_timesteps:
        raise ValueError(
            "Stage-1 teacher timestep schedule does not match the formal Nested20 schedule"
        )

    if list(checkpoint.get("student_timesteps", [])) != expected_student_timesteps:
        raise ValueError(
            "Stage-1 student timestep schedule does not match the formal Nested10 schedule"
        )

    for key in (
        "model_config",
        "diffusion_config",
        "normalizer_weighting",
    ):
        if checkpoint.get(key) != base_checkpoint.get(key):
            raise ValueError(
                f"Stage-1 checkpoint {key!r} does not match the formal base checkpoint"
            )

    for key in (
        "branch_channel_names",
        "trunk_channel_names",
        "output_channel_names",
    ):
        checkpoint_channels = tuple(checkpoint.get(key, ()))
        base_channels = tuple(base_checkpoint.get(key, ()))

        if checkpoint_channels != base_channels:
            raise ValueError(
                f"Stage-1 checkpoint {key!r} does not match the formal base checkpoint: "
                f"{checkpoint_channels!r} != {base_channels!r}"
            )

    checkpoint_train_h5 = resolve_path(Path(checkpoint["train_dataset_h5"]))
    checkpoint_val_h5 = resolve_path(Path(checkpoint["val_dataset_h5"]))

    if checkpoint_train_h5 != config.train_h5:
        raise ValueError(
            "Stage-1 checkpoint training dataset does not match --train-h5: "
            f"{checkpoint_train_h5} != {config.train_h5}"
        )

    if checkpoint_val_h5 != config.val_h5:
        raise ValueError(
            "Stage-1 checkpoint validation dataset does not match --val-h5: "
            f"{checkpoint_val_h5} != {config.val_h5}"
        )

    checkpoint_norm = checkpoint["y_normalizer"]
    base_norm = base_checkpoint["y_normalizer"]

    for key in ("mean", "std", "eps"):
        checkpoint_value = torch.as_tensor(checkpoint_norm[key]).detach().cpu()
        base_value = torch.as_tensor(base_norm[key]).detach().cpu()
        if not torch.equal(checkpoint_value, base_value):
            raise ValueError(
                f"Stage-1 checkpoint y_normalizer[{key!r}] "
                "does not match the formal base checkpoint"
            )

    for key in ("local_aspect_mean", "local_aspect_std"):
        checkpoint_value = float(checkpoint[key])
        base_value = float(base_checkpoint[key])
        if checkpoint_value != base_value:
            raise ValueError(
                f"Stage-1 checkpoint {key!r} does not match the formal base checkpoint"
            )

    print("Stage-1 checkpoint preflight: PASS")
    print("  path              :", checkpoint_path)
    print("  stage             :", checkpoint["stage"])
    print("  epoch             :", checkpoint["epoch"])
    print("  teacher NFE       :", checkpoint["teacher_nfe"])
    print("  student NFE       :", checkpoint["student_nfe"])
    print("  student timesteps :", checkpoint["student_timesteps"])

def print_plan(
    *,
    config: Config,
    base_checkpoint: Mapping[str, Any],
    train_protocol: H5Protocol,
    val_protocol: H5Protocol,
    stage1: StageSchedule,
    stage2: StageSchedule,
) -> None:
    print("Progressive field-diffusion distillation")
    print("  protocol          :", PROTOCOL_VERSION)
    print("  base checkpoint   :", config.base_checkpoint)
    print("  base epoch        :", base_checkpoint.get("epoch", "unknown"))
    print("  train HDF5        :", config.train_h5)
    print("  train samples     :", train_protocol.n_samples)
    print("  validation HDF5   :", config.val_h5)
    print("  validation samples:", val_protocol.n_samples)
    print("  test access       : disabled")
    print("  device            :", config.device)
    print("  objective         : direct normalized reverse-state matching")
    print("  truth train loss  : disabled")
    print("  physics loss      : disabled")
    print("  BC auxiliary      : disabled")
    print("  query points      :", config.query_points)
    print("  batch size        :", config.batch_size)
    print("  train progress    : every", config.progress_every_batches, "batches")
    print(
        "  val progress      : every",
        config.progress_every_validation_samples,
        "samples",
    )
    print("  learning rate     :", config.learning_rate)
    print("  stage 1 epochs    :", config.epochs_stage1)
    print("  stage 2 epochs    :", config.epochs_stage2)
    print("  schedule family   :", SCHEDULE_FAMILY)
    print("  anchor 20         :", list(stage1.teacher_sources))
    print("  nested 10         :", list(stage1.student_sources))
    print("  nested 5          :", list(stage2.student_sources))
    print("  stage 1           : 20 NFE -> 10 NFE")
    print("  stage 2           : 10 NFE -> 5 NFE")
    print("  final endpoint    : clean projection is part of exact 2:1 mapping")
    print("  checkpoint select : deterministic validation rollout normalized MSE")
    if config.max_train_samples is not None:
        print("  max train samples :", config.max_train_samples)
    if config.max_batches_per_epoch is not None:
        print("  max batches/epoch :", config.max_batches_per_epoch)
    if config.validation_max_samples is not None:
        print("  max val samples   :", config.validation_max_samples)
    if config.stage1_only:
        print("  stage 2           : disabled by --stage1-only")
    if config.stage2_only:
        print("  stage 1           : skipped by --stage2-only")
        print("  stage 1 checkpoint:", config.stage1_checkpoint)


def run_distillation(config: Config) -> Path:
    require_file(config.base_checkpoint, "base checkpoint")
    require_file(config.train_h5, "training HDF5")
    require_file(config.val_h5, "validation HDF5")
    if config.stage2_only:
        if config.stage1_checkpoint is None:
            raise ValueError("--stage2-only requires --stage1-checkpoint")
        require_file(config.stage1_checkpoint, "stage 1 checkpoint")

    base_checkpoint = load_checkpoint(config.base_checkpoint)
    train_protocol = read_h5_protocol(config.train_h5)
    val_protocol = read_h5_protocol(config.val_h5)
    validate_base_protocol(
        base_checkpoint,
        config.train_h5,
        config.val_h5,
        train_protocol,
        val_protocol,
    )

    diffusion = base_checkpoint["diffusion_config"]
    stage1, stage2 = build_stage_schedules(int(diffusion["T"]))

    if config.stage2_only:
        assert config.stage1_checkpoint is not None
        validate_stage1_checkpoint_for_stage2(
            checkpoint_path=config.stage1_checkpoint,
            base_checkpoint=base_checkpoint,
            config=config,
            stage1_schedule=stage1,
        )

    print_plan(
        config=config,
        base_checkpoint=base_checkpoint,
        train_protocol=train_protocol,
        val_protocol=val_protocol,
        stage1=stage1,
        stage2=stage2,
    )

    device = resolve_device(config.device)
    set_global_seed(config.seed)

    run_id = config.run_id
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if config.stage2_only:
            run_id = f"{stamp}_nested10_to5_seed{config.seed}"
        else:
            run_id = f"{stamp}_nested20_to10_to5_seed{config.seed}"
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"Unsafe run_id: {run_id!r}")

    run_dir = config.results_root / "distill_progressive" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    outputs = {
        "run_directory": str(run_dir),
        "history_csv": str(run_dir / "distillation_history.csv"),
        "stage1_best": (
            str(config.stage1_checkpoint)
            if config.stage2_only
            else str(run_dir / "stage1_best.pt")
        ),
        "stage1_latest": (
            None
            if config.stage2_only
            else str(run_dir / "stage1_latest.pt")
        ),
        "stage2_best": None if config.stage1_only else str(run_dir / "stage2_best.pt"),
        "stage2_latest": None if config.stage1_only else str(run_dir / "stage2_latest.pt"),
    }

    created_at = utc_now()
    write_manifest(
        run_dir,
        build_manifest(
            run_id=run_id,
            status="prepared",
            created_at=created_at,
            started_at=None,
            finished_at=None,
            config=config,
            base_checkpoint_path=config.base_checkpoint,
            train_protocol=train_protocol,
            val_protocol=val_protocol,
            stage1_schedule=stage1,
            stage2_schedule=stage2,
            outputs=outputs,
        ),
    )

    started_at = utc_now()
    update_manifest(
        run_dir,
        build_manifest(
            run_id=run_id,
            status="running",
            created_at=created_at,
            started_at=started_at,
            finished_at=None,
            config=config,
            base_checkpoint_path=config.base_checkpoint,
            train_protocol=train_protocol,
            val_protocol=val_protocol,
            stage1_schedule=stage1,
            stage2_schedule=stage2,
            outputs=outputs,
        ),
    )

    schedule = build_linear_schedule(
        timesteps=int(diffusion["T"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        device=device,
    )
    validate_vectorized_ddim_algebra(schedule.alphas_cumprod)

    history_rows: list[dict[str, Any]] = []
    global_step = 0

    try:
        if config.stage2_only:
            assert config.stage1_checkpoint is not None
            stage1_best = config.stage1_checkpoint
            print("\nStage 1 skipped.")
            print("Stage 2 teacher checkpoint:", stage1_best)
        else:
            stage1_best, global_step = stage_run(
                stage_schedule=stage1,
                base_checkpoint=base_checkpoint,
                teacher_checkpoint_path=config.base_checkpoint,
                initialization_checkpoint_path=config.base_checkpoint,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                config=config,
                device=device,
                alphas_cumprod=schedule.alphas_cumprod,
                run_dir=run_dir,
                history_rows=history_rows,
                global_step_start=global_step,
            )

        if not config.stage1_only:
            stage2_best, global_step = stage_run(
                stage_schedule=stage2,
                base_checkpoint=base_checkpoint,
                teacher_checkpoint_path=stage1_best,
                initialization_checkpoint_path=stage1_best,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                config=config,
                device=device,
                alphas_cumprod=schedule.alphas_cumprod,
                run_dir=run_dir,
                history_rows=history_rows,
                global_step_start=global_step,
            )
            outputs["stage2_best"] = str(stage2_best)

        update_manifest(
            run_dir,
            build_manifest(
                run_id=run_id,
                status="completed",
                created_at=created_at,
                started_at=started_at,
                finished_at=utc_now(),
                config=config,
                base_checkpoint_path=config.base_checkpoint,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                stage1_schedule=stage1,
                stage2_schedule=stage2,
                outputs=outputs,
            ),
        )

        print("\nCompleted:", run_dir)
        if config.stage2_only:
            print("Stage 1 source:", stage1_best)
        else:
            print("Stage 1 best:", stage1_best)
        if not config.stage1_only:
            print("Stage 2 best:", outputs["stage2_best"])
        return run_dir

    except Exception as exc:
        update_manifest(
            run_dir,
            build_manifest(
                run_id=run_id,
                status="failed",
                created_at=created_at,
                started_at=started_at,
                finished_at=utc_now(),
                config=config,
                base_checkpoint_path=config.base_checkpoint,
                train_protocol=train_protocol,
                val_protocol=val_protocol,
                stage1_schedule=stage1,
                stage2_schedule=stage2,
                outputs=outputs,
                failure=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Progressively distill the formal PIDIF field-diffusion baseline "
            "from nested 20 NFE to 10 NFE and then 5 NFE."
        )
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
    )
    parser.add_argument("--train-h5", type=Path, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--val-h5", type=Path, default=DEFAULT_VAL_H5)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-sampling-seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-points", type=int, default=8192)
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-threshold", type=float, default=1.0e-5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument(
        "--min-lr-early-stop-patience", type=int, default=10
    )
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Explicit smoke/debug limit. Omit for the formal 16000-sample training set.",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Explicit smoke/debug limit. Omit for a formal run.",
    )
    parser.add_argument(
        "--validation-max-samples",
        type=int,
        default=None,
        help="Explicit smoke/debug limit. Omit for all 1000 validation subdomains.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=25,
        help=(
            "Print training progress every N processed batches. "
            "Set 0 to disable batch progress printing."
        ),
    )
    parser.add_argument(
        "--progress-every-validation-samples",
        type=int,
        default=100,
        help=(
            "Print rollout-validation progress every N subdomains. "
            "Set 0 to disable validation progress printing."
        ),
    )
    parser.add_argument(
        "--validate-checkpoint-only",
        type=Path,
        default=None,
        help=(
            "Validate an existing distilled student checkpoint on the canonical "
            "validation rollout and exit without training."
        ),
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Train only the 20-to-10 stage.",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help=(
            "Train only the 10-to-5 stage from an existing "
            "Stage-1 progressive-distillation checkpoint."
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help=(
            "Existing Stage-1 10-NFE checkpoint used as both the frozen "
            "Stage-2 teacher and model-only student initialization. "
            "Required with --stage2-only."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually start GPU distillation. Without this flag only validate and print the plan.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    if args.stage1_only and args.stage2_only:
        raise ValueError(
            "--stage1-only and --stage2-only are mutually exclusive"
        )
    if args.stage2_only and args.stage1_checkpoint is None:
        raise ValueError(
            "--stage2-only requires --stage1-checkpoint"
        )
    if not args.stage2_only and args.stage1_checkpoint is not None:
        raise ValueError(
            "--stage1-checkpoint is only valid with --stage2-only"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.query_points <= 0:
        raise ValueError("--query-points must be positive")
    if args.epochs_stage1 <= 0 or args.epochs_stage2 <= 0:
        raise ValueError("Stage epoch counts must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.validation_every <= 0:
        raise ValueError("--validation-every must be positive")
    if args.progress_every_batches < 0:
        raise ValueError("--progress-every-batches must be >= 0")
    if args.progress_every_validation_samples < 0:
        raise ValueError(
            "--progress-every-validation-samples must be >= 0"
        )
    if not (0.0 < args.scheduler_factor < 1.0):
        raise ValueError("--scheduler-factor must be between 0 and 1")
    if args.min_lr <= 0.0 or args.min_lr > args.learning_rate:
        raise ValueError("--min-lr must be positive and <= learning rate")

    return Config(
        base_checkpoint=resolve_path(args.base_checkpoint),
        train_h5=resolve_path(args.train_h5),
        val_h5=resolve_path(args.val_h5),
        results_root=resolve_path(args.results_root),
        device=args.device,
        run_id=args.run_id,
        seed=args.seed,
        validation_sampling_seed=args.validation_sampling_seed,
        batch_size=args.batch_size,
        query_points=args.query_points,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_threshold=args.scheduler_threshold,
        min_lr=args.min_lr,
        min_lr_early_stop_patience=args.min_lr_early_stop_patience,
        validation_every=args.validation_every,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_batches_per_epoch=args.max_batches_per_epoch,
        validation_max_samples=args.validation_max_samples,
        progress_every_batches=args.progress_every_batches,
        progress_every_validation_samples=(
            args.progress_every_validation_samples
        ),
        stage1_only=args.stage1_only,
        stage2_only=args.stage2_only,
        stage1_checkpoint=(
            resolve_path(args.stage1_checkpoint)
            if args.stage1_checkpoint is not None
            else None
        ),
    )


def validate_checkpoint_only(
    config: Config, checkpoint_path: Path
) -> ValidationSummary:
    checkpoint_path = resolve_path(checkpoint_path)
    require_file(config.base_checkpoint, "base checkpoint")
    require_file(config.val_h5, "validation HDF5")
    require_file(checkpoint_path, "distilled checkpoint")

    base_checkpoint = load_checkpoint(config.base_checkpoint)
    train_protocol = read_h5_protocol(config.train_h5)
    val_protocol = read_h5_protocol(config.val_h5)
    validate_base_protocol(
        base_checkpoint,
        config.train_h5,
        config.val_h5,
        train_protocol,
        val_protocol,
    )

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("checkpoint_type") != "progressive_distillation_student":
        raise ValueError(
            "--validate-checkpoint-only requires a progressive-distillation "
            "student checkpoint"
        )
    if checkpoint_path_field(checkpoint, "train_dataset_h5") != config.train_h5:
        raise ValueError("Distilled checkpoint training HDF5 does not match")
    if checkpoint_path_field(checkpoint, "val_dataset_h5") != config.val_h5:
        raise ValueError("Distilled checkpoint validation HDF5 does not match")
    if checkpoint.get("schedule_family") != SCHEDULE_FAMILY:
        raise ValueError(
            f"Unexpected schedule family: {checkpoint.get('schedule_family')!r}"
        )

    stage1, stage2 = build_stage_schedules(
        int(base_checkpoint["diffusion_config"]["T"])
    )
    student_nfe = int(require_checkpoint_field(checkpoint, "student_nfe"))
    if student_nfe == stage1.student_nfe:
        schedule = stage1
    elif student_nfe == stage2.student_nfe:
        schedule = stage2
    else:
        raise ValueError(
            f"Unsupported distilled student_nfe={student_nfe}; expected 10 or 5"
        )
    if tuple(int(x) for x in checkpoint["student_timesteps"]) != tuple(
        schedule.student_sources
    ):
        raise ValueError("Checkpoint student timesteps do not match formal schedule")

    device = resolve_device(config.device)
    diffusion = base_checkpoint["diffusion_config"]
    linear_schedule = build_linear_schedule(
        timesteps=int(diffusion["T"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        device=device,
    )
    model = build_model_from_checkpoint(checkpoint, device, frozen=True)
    normalizer, target_mean, target_std = normalizer_numpy(base_checkpoint)
    val_indices = choose_indices(
        val_protocol.n_samples, config.validation_max_samples
    )

    print("Distilled-checkpoint validation only")
    print("  checkpoint        :", checkpoint_path)
    print("  checkpoint stage  :", checkpoint.get("stage", "unknown"))
    print("  checkpoint epoch  :", checkpoint.get("epoch", "unknown"))
    print("  student NFE       :", student_nfe)
    print("  student timesteps :", list(schedule.student_sources))
    print("  validation HDF5   :", config.val_h5)
    print("  validation selected:", len(val_indices))
    print("  sampling seed     :", config.validation_sampling_seed)
    print("  test access       : disabled")

    summary = validate_rollout(
        model=model,
        val_h5=config.val_h5,
        val_indices=val_indices,
        branch_channel_names=val_protocol.branch_channel_names,
        normalizer=normalizer,
        target_mean=target_mean,
        target_std=target_std,
        local_aspect_mean=float(base_checkpoint["local_aspect_mean"]),
        local_aspect_std=float(base_checkpoint["local_aspect_std"]),
        student_sources=schedule.student_sources,
        alphas_cumprod=linear_schedule.alphas_cumprod,
        device=device,
        sampling_seed=config.validation_sampling_seed,
        stage=int(checkpoint.get("stage", schedule.stage)),
        epoch=int(checkpoint.get("epoch", 0)),
        progress_every_validation_samples=(
            config.progress_every_validation_samples
        ),
    )

    print("\nValidation summary")
    print(
        f"  val_norm_mse={summary.balanced_norm_mse:.9g} | "
        f"p={summary.balanced_rmse_pressure:.6g} Pa | "
        f"u={summary.balanced_rmse_u:.8g} m/s | "
        f"v={summary.balanced_rmse_v:.8g} m/s"
    )
    print("  validation samples:", summary.n_samples)
    print("  validation points :", summary.n_points)
    return summary


def dry_run(config: Config) -> None:
    require_file(config.base_checkpoint, "base checkpoint")
    require_file(config.train_h5, "training HDF5")
    require_file(config.val_h5, "validation HDF5")

    checkpoint = load_checkpoint(config.base_checkpoint)
    train_protocol = read_h5_protocol(config.train_h5)
    val_protocol = read_h5_protocol(config.val_h5)
    validate_base_protocol(
        checkpoint,
        config.train_h5,
        config.val_h5,
        train_protocol,
        val_protocol,
    )
    stage1, stage2 = build_stage_schedules(int(checkpoint["diffusion_config"]["T"]))

    if config.stage2_only:
        assert config.stage1_checkpoint is not None
        require_file(config.stage1_checkpoint, "stage 1 checkpoint")
        validate_stage1_checkpoint_for_stage2(
            checkpoint_path=config.stage1_checkpoint,
            base_checkpoint=checkpoint,
            config=config,
            stage1_schedule=stage1,
        )

    diffusion = checkpoint["diffusion_config"]
    cpu_schedule = build_linear_schedule(
        timesteps=int(diffusion["T"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        device="cpu",
    )
    validate_vectorized_ddim_algebra(cpu_schedule.alphas_cumprod)
    print_plan(
        config=config,
        base_checkpoint=checkpoint,
        train_protocol=train_protocol,
        val_protocol=val_protocol,
        stage1=stage1,
        stage2=stage2,
    )
    print("\nDistillation was not started. Pass --run to execute it.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    if args.validate_checkpoint_only is not None:
        if args.run:
            raise ValueError(
                "Do not combine --validate-checkpoint-only with --run"
            )
        validate_checkpoint_only(config, args.validate_checkpoint_only)
    elif args.run:
        run_distillation(config)
    else:
        dry_run(config)


if __name__ == "__main__":
    main()
