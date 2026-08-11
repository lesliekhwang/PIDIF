"""Formal reverse-generation evaluation for the field-diffusion baseline.

Validation only:
- loads the frozen best field-diffusion checkpoint;
- reads the canonical random5 validation HDF5;
- generates p/u/v from pure Gaussian noise with deterministic eta=0 DDIM;
- compares physical-space predictions with Fluent ground truth;
- reports accuracy and reverse-sampling runtime;
- optionally stores predictions for later plotting.

Without --run, this script only validates inputs and prints the evaluation plan.
It intentionally refuses non-validation datasets so CP/AR1 test data cannot be
used while the sampling protocol is still being selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


PROTOCOL_VERSION = "field_diffusion_generation_eval_v3"
MANIFEST_SCHEMA_VERSION = "pidiffusion_run_manifest_v1"

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "results/train_diffusion/field_diffusion_baseline_long_seed0"
    / "diffusion_best.pt"
)
DEFAULT_VALIDATION_H5 = (
    REPO_ROOT
    / "channel_diffusion_dataset/deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

EXPECTED_DATASET_ROLE = "canonical_randomized_validation"
EXPECTED_SPLIT_ROLE = "validation"
EXPECTED_N_SAMPLES = 1000
EXPECTED_OUTPUTS = ("pressure", "u", "v")
EXPECTED_TRUNK = ("x_local", "y_local")
EXPECTED_PARAMETER_COUNT = 382_083

SCHEDULE_FAMILY_STANDARD = "standard"
SCHEDULE_FAMILY_PROGRESSIVE_NESTED20 = "progressive_nested20"
PROGRESSIVE_NESTED_ANCHOR_NFE = 20
PROGRESSIVE_NESTED_ALLOWED_NFE = (20, 10, 5)

FIELD_UNITS = {
    "pressure": "Pa",
    "u": "m/s",
    "v": "m/s",
}

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    validation_h5: Path
    results_root: Path
    device: str
    schedule_family: str
    sampling_steps: tuple[int, ...]
    sampling_seed: int
    sample_indices: tuple[int, ...] | None
    max_samples: int | None
    save_predictions: bool
    run_id: str | None
    progress_every: int


@dataclass
class FieldAccumulator:
    """Streaming aggregate statistics for one physical output field."""

    n_subdomains: int = 0
    n_points: int = 0
    subdomain_mse_sum: float = 0.0
    subdomain_mae_sum: float = 0.0
    sse: float = 0.0
    sae: float = 0.0
    true_sum: float = 0.0
    pred_sum: float = 0.0
    true_sq_sum: float = 0.0
    pred_sq_sum: float = 0.0
    true_pred_sum: float = 0.0

    def add(self, truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        y = np.asarray(truth, dtype=np.float64).reshape(-1)
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        if y.shape != p.shape or y.size == 0:
            raise ValueError("Invalid truth/prediction arrays")
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Truth/prediction contains non-finite values")

        error = p - y
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        corr = (
            float(np.corrcoef(y, p)[0, 1])
            if float(np.std(y)) > 0.0 and float(np.std(p)) > 0.0
            else float("nan")
        )

        self.n_subdomains += 1
        self.n_points += int(y.size)
        self.subdomain_mse_sum += mse
        self.subdomain_mae_sum += mae
        self.sse += ss_res
        self.sae += float(np.sum(np.abs(error)))
        self.true_sum += float(np.sum(y))
        self.pred_sum += float(np.sum(p))
        self.true_sq_sum += float(np.sum(y * y))
        self.pred_sq_sum += float(np.sum(p * p))
        self.true_pred_sum += float(np.sum(y * p))

        return {
            "rmse": math.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "correlation": corr,
        }

    def summary(self) -> dict[str, float | int]:
        if self.n_subdomains == 0 or self.n_points == 0:
            raise RuntimeError("No metrics were accumulated")

        balanced_rmse = math.sqrt(
            self.subdomain_mse_sum / self.n_subdomains
        )
        balanced_mae = self.subdomain_mae_sum / self.n_subdomains
        global_rmse = math.sqrt(self.sse / self.n_points)
        global_mae = self.sae / self.n_points

        true_centered_sq = (
            self.true_sq_sum - self.true_sum**2 / self.n_points
        )
        pred_centered_sq = (
            self.pred_sq_sum - self.pred_sum**2 / self.n_points
        )
        centered_cross = (
            self.true_pred_sum
            - self.true_sum * self.pred_sum / self.n_points
        )

        global_r2 = (
            1.0 - self.sse / true_centered_sq
            if true_centered_sq > 0.0
            else float("nan")
        )
        denominator = math.sqrt(
            max(true_centered_sq, 0.0)
            * max(pred_centered_sq, 0.0)
        )
        global_corr = (
            centered_cross / denominator
            if denominator > 0.0
            else float("nan")
        )

        return {
            "n_subdomains": self.n_subdomains,
            "n_points": self.n_points,
            "subdomain_balanced_rmse": balanced_rmse,
            "subdomain_balanced_mae": balanced_mae,
            "global_rmse": global_rmse,
            "global_mae": global_mae,
            "global_r2": global_r2,
            "global_correlation": global_corr,
        }


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


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    return checkpoint


def require_checkpoint_field(
    checkpoint: Mapping[str, Any],
    name: str,
) -> Any:
    if name not in checkpoint:
        raise KeyError(f"Checkpoint missing required field: {name}")
    return checkpoint[name]


def read_validation_protocol(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        return {
            "dataset_role": decode_text(attrs.get("dataset_role", "")),
            "split_role": decode_text(attrs.get("split_role", "")),
            "n_samples": int(attrs["n_samples"]),
            "n_subdomains": int(attrs.get("n_subdomains", -1)),
            "n_realizations": int(attrs.get("n_realizations", -1)),
            "branch_channel_names": tuple(
                decode_text(attrs["branch_channel_names"]).split("\n")
            ),
            "trunk_channel_names": tuple(
                decode_text(attrs["trunk_channel_names"]).split("\n")
            ),
            "output_channel_names": tuple(
                decode_text(attrs["output_channel_names"]).split("\n")
            ),
            "split_manifest_path": decode_text(
                attrs.get("split_manifest_path", "")
            ),
            "split_manifest_sha256": decode_text(
                attrs.get("split_manifest_sha256", "")
            ),
            "dataset_builder_path": decode_text(
                attrs.get("dataset_builder_path", "")
            ),
            "dataset_builder_sha256": decode_text(
                attrs.get("dataset_builder_sha256", "")
            ),
            "generator_script_path": decode_text(
                attrs.get("generator_script_path", "")
            ),
            "generator_script_sha256": decode_text(
                attrs.get("generator_script_sha256", "")
            ),
        }


def validate_protocol(
    checkpoint: Mapping[str, Any],
    validation_h5: Path,
    protocol: Mapping[str, Any],
    sampling_steps: Sequence[int],
) -> None:
    if protocol["dataset_role"] != EXPECTED_DATASET_ROLE:
        raise ValueError(
            "Validation-only evaluator rejected dataset_role="
            f"{protocol['dataset_role']!r}"
        )
    if protocol["split_role"] != EXPECTED_SPLIT_ROLE:
        raise ValueError(
            "Validation-only evaluator rejected split_role="
            f"{protocol['split_role']!r}"
        )
    if protocol["n_samples"] != EXPECTED_N_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_N_SAMPLES} validation samples, "
            f"got {protocol['n_samples']}"
        )
    if tuple(protocol["output_channel_names"]) != EXPECTED_OUTPUTS:
        raise ValueError("Unexpected validation output channels")
    if tuple(protocol["trunk_channel_names"]) != EXPECTED_TRUNK:
        raise ValueError("Unexpected validation trunk channels")

    checkpoint_val = Path(
        str(require_checkpoint_field(checkpoint, "val_dataset_h5"))
    ).expanduser().resolve(strict=False)
    if checkpoint_val != validation_h5.resolve(strict=False):
        raise ValueError(
            "Checkpoint validation HDF5 does not match requested HDF5"
        )

    if (
        require_checkpoint_field(checkpoint, "normalizer_weighting")
        != "subdomain_balanced"
    ):
        raise ValueError("Checkpoint normalizer is not subdomain-balanced")

    if tuple(require_checkpoint_field(
        checkpoint, "output_channel_names"
    )) != tuple(protocol["output_channel_names"]):
        raise ValueError("Checkpoint/output channel mismatch")
    if tuple(require_checkpoint_field(
        checkpoint, "trunk_channel_names"
    )) != tuple(protocol["trunk_channel_names"]):
        raise ValueError("Checkpoint/trunk channel mismatch")
    if tuple(require_checkpoint_field(
        checkpoint, "branch_channel_names"
    )) != tuple(protocol["branch_channel_names"]):
        raise ValueError("Checkpoint/branch channel mismatch")

    diffusion = require_checkpoint_field(checkpoint, "diffusion_config")
    if diffusion.get("prediction_target") != "epsilon":
        raise ValueError("Only epsilon-prediction checkpoints are supported")
    if diffusion.get("timestep_unit") != "subdomain":
        raise ValueError("Checkpoint timestep_unit must be 'subdomain'")

    total_steps = int(diffusion["T"])
    for steps in sampling_steps:
        if steps < 2 or steps > total_steps:
            raise ValueError(
                f"Invalid DDIM state count {steps} for T={total_steps}"
            )


def build_sampling_timesteps(
    *,
    sampling_steps: int,
    total_diffusion_steps: int,
    schedule_family: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Build one evaluation schedule without changing the diffusion core.

    ``standard`` preserves the existing independently spaced DDIM schedule.

    ``progressive_nested20`` uses the standard 20-NFE schedule as the
    immutable anchor and takes every 2nd or 4th model-evaluation source
    timestep to obtain exactly nested 10-NFE and 5-NFE schedules.  The
    sampler still performs a final clean projection from the last listed
    timestep, so the number of listed timesteps equals the number of model
    evaluations (NFE).
    """

    if schedule_family == SCHEDULE_FAMILY_STANDARD:
        return build_ddim_timesteps(
            sampling_steps,
            total_diffusion_steps,
            device=device,
        )

    if schedule_family != SCHEDULE_FAMILY_PROGRESSIVE_NESTED20:
        raise ValueError(
            f"Unsupported schedule family: {schedule_family!r}"
        )

    if sampling_steps not in PROGRESSIVE_NESTED_ALLOWED_NFE:
        raise ValueError(
            "progressive_nested20 only supports NFE values "
            f"{PROGRESSIVE_NESTED_ALLOWED_NFE}; got {sampling_steps}"
        )

    anchor = build_ddim_timesteps(
        PROGRESSIVE_NESTED_ANCHOR_NFE,
        total_diffusion_steps,
        device=device,
    )
    stride = PROGRESSIVE_NESTED_ANCHOR_NFE // sampling_steps
    nested = anchor[::stride].clone()

    if nested.numel() != sampling_steps:
        raise RuntimeError(
            "Nested DDIM schedule construction produced "
            f"{nested.numel()} timesteps for requested NFE={sampling_steps}"
        )
    if int(nested[0]) != total_diffusion_steps - 1:
        raise RuntimeError(
            "Nested DDIM schedule does not start at the highest timestep"
        )
    if nested.numel() > 1 and not torch.all(nested[:-1] > nested[1:]):
        raise RuntimeError(
            "Nested DDIM schedule is not strictly descending"
        )

    return nested


def resolve_device(text: str) -> torch.device:
    device = torch.device(text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but CUDA is unavailable"
            )
        if (
            device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Requested {device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return device


def build_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> PointSetDiffusionDenoiser:
    model = PointSetDiffusionDenoiser(
        **dict(require_checkpoint_field(checkpoint, "model_config"))
    )
    model.load_state_dict(
        require_checkpoint_field(checkpoint, "model_state_dict"),
        strict=True,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Model parameter count {parameter_count:,} != "
            f"{EXPECTED_PARAMETER_COUNT:,}"
        )
    model.requires_grad_(False)
    return model.to(device).eval()


def parse_sample_indices(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    values = tuple(
        int(part.strip())
        for part in text.split(",")
        if part.strip()
    )
    if not values:
        raise ValueError("--sample-indices is empty")
    if len(values) != len(set(values)):
        raise ValueError("--sample-indices contains duplicates")
    return values


def select_sample_indices(
    n_samples: int,
    explicit: tuple[int, ...] | None,
    max_samples: int | None,
) -> tuple[int, ...]:
    if explicit is not None and max_samples is not None:
        raise ValueError(
            "Use --sample-indices or --max-samples, not both"
        )

    if explicit is not None:
        if any(index < 0 or index >= n_samples for index in explicit):
            raise IndexError("A selected sample index is out of range")
        return explicit

    if max_samples is None or max_samples >= n_samples:
        return tuple(range(n_samples))
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    values = np.linspace(
        0,
        n_samples - 1,
        num=max_samples,
    ).round().astype(np.int64)
    values = np.unique(values)
    if len(values) != max_samples:
        raise RuntimeError("Subset selection produced duplicate indices")
    return tuple(int(value) for value in values)


def stable_sample_seed(base_seed: int, sample_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{sample_index}".encode("utf-8")
    ).digest()
    return (
        int.from_bytes(digest[:8], "little", signed=False)
        % (2**63 - 1)
    )


def read_sample(
    handle: h5py.File,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    group = handle["samples"][str(sample_index)]
    branch = group["branch"][:].astype(np.float32)
    query = group["query"][:].astype(np.float32)
    target = group["target"][:].astype(np.float32)

    metadata_group = handle["metadata"]
    metadata = {
        key: decode_scalar(metadata_group[key][sample_index])
        for key in metadata_group.keys()
    }

    if branch.ndim != 2:
        raise ValueError(f"Invalid branch shape: {branch.shape}")
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"Invalid query shape: {query.shape}")
    if target.shape != (query.shape[0], 3):
        raise ValueError(f"Invalid target shape: {target.shape}")
    if not (
        np.isfinite(branch).all()
        and np.isfinite(query).all()
        and np.isfinite(target).all()
    ):
        raise ValueError(f"Sample {sample_index} contains non-finite values")

    return branch, query, target, metadata


@torch.inference_mode()
def predict_epsilon(
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
            (n_query,),
            timestep,
            dtype=torch.long,
            device=query.device,
        ),
        query_batch_id=torch.zeros(
            n_query,
            dtype=torch.long,
            device=query.device,
        ),
        branch_mask=None,
    )


@torch.inference_mode()
def sample_ddim(
    model: PointSetDiffusionDenoiser,
    branch: torch.Tensor,
    query: torch.Tensor,
    initial_noise: torch.Tensor,
    schedule,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    x_t = initial_noise.clone()

    for current, next_ in zip(timesteps[:-1], timesteps[1:]):
        t_current = int(current.item())
        t_next = int(next_.item())
        epsilon = predict_epsilon(
            model,
            branch,
            query,
            x_t,
            t_current,
        )
        x_t, _ = ddim_step(
            x_t=x_t,
            epsilon_pred=epsilon,
            t_current=t_current,
            t_next=t_next,
            alphas_cumprod=schedule.alphas_cumprod,
        )

    final_t = int(timesteps[-1].item())
    epsilon = predict_epsilon(
        model,
        branch,
        query,
        x_t,
        final_t,
    )
    return final_clean_projection(
        x_t=x_t,
        epsilon_pred=epsilon,
        timestep=final_t,
        alphas_cumprod=schedule.alphas_cumprod,
    )


def timed_sample(
    model,
    branch,
    query,
    initial_noise,
    schedule,
    timesteps,
    device,
) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    prediction = sample_ddim(
        model,
        branch,
        query,
        initial_noise,
        schedule,
        timesteps,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return prediction, time.perf_counter() - start


@torch.inference_mode()
def warm_up(
    model,
    branch,
    query,
    target_dim,
    timestep,
    device,
) -> None:
    query = query[: min(2048, len(query))]
    x = torch.zeros(
        (len(query), target_dim),
        dtype=query.dtype,
        device=device,
    )
    _ = predict_epsilon(
        model,
        branch,
        query,
        x,
        timestep,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_validation_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return checkpoint-selection metadata for base or distilled checkpoints."""
    if "val_interior_loss" in checkpoint:
        return {
            "validation_metric_name": "val_interior_loss",
            "validation_metric_value": float(checkpoint["val_interior_loss"]),
            "val_interior_loss": float(checkpoint["val_interior_loss"]),
        }
    if "val_rollout_balanced_norm_mse" in checkpoint:
        return {
            "validation_metric_name": "val_rollout_balanced_norm_mse",
            "validation_metric_value": float(
                checkpoint["val_rollout_balanced_norm_mse"]
            ),
            "val_rollout_balanced_norm_mse": float(
                checkpoint["val_rollout_balanced_norm_mse"]
            ),
            "distillation_stage": checkpoint.get("stage"),
            "student_nfe": checkpoint.get("student_nfe"),
            "schedule_family": checkpoint.get("schedule_family"),
        }
    return {
        "validation_metric_name": None,
        "validation_metric_value": None,
    }


def build_provenance(
    config: Config,
    protocol: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        REPO_ROOT / "pidiffusion/data.py",
        REPO_ROOT / "pidiffusion/diffusion.py",
        REPO_ROOT / "pidiffusion/model.py",
    ]
    return {
        "git": git_state(REPO_ROOT),
        "source_files": [
            file_identity(path)
            for path in source_files
        ],
        "dataset": {
            **file_identity(config.validation_h5),
            "dataset_role": protocol["dataset_role"],
            "split_role": protocol["split_role"],
            "split_manifest_path": protocol["split_manifest_path"],
            "split_manifest_sha256": protocol[
                "split_manifest_sha256"
            ],
            "dataset_builder_path": protocol["dataset_builder_path"],
            "dataset_builder_sha256": protocol[
                "dataset_builder_sha256"
            ],
            "generator_script_path": protocol[
                "generator_script_path"
            ],
            "generator_script_sha256": protocol[
                "generator_script_sha256"
            ],
        },
        "checkpoint": {
            **file_identity(config.checkpoint),
            "training_protocol_version": checkpoint.get(
                "training_protocol_version"
            ),
            "epoch": int(
                require_checkpoint_field(checkpoint, "epoch")
            ),
            "global_step": int(
                require_checkpoint_field(checkpoint, "global_step")
            ),
            **checkpoint_validation_metadata(checkpoint),
            "checkpoint_type": checkpoint.get("checkpoint_type", "base_diffusion"),
            "model_config": dict(
                require_checkpoint_field(checkpoint, "model_config")
            ),
            "diffusion_config": dict(
                require_checkpoint_field(
                    checkpoint,
                    "diffusion_config",
                )
            ),
        },
        "environment": runtime_environment(),
    }


def make_manifest(
    *,
    run_id: str,
    status: str,
    created_at: str,
    started_at: str | None,
    finished_at: str | None,
    provenance: Mapping[str, Any],
    config: Config,
    selected_indices: Sequence[int],
    timesteps: Mapping[int, torch.Tensor],
    outputs: Mapping[str, str | None],
    failure: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": created_at,
        "status": status,
        "git": provenance["git"],
        "source_files": provenance["source_files"],
        "dataset": {
            **provenance["dataset"],
            "n_samples_selected": len(selected_indices),
            "selected_sample_indices": list(selected_indices),
        },
        "checkpoint": provenance["checkpoint"],
        "protocol": {
            "version": PROTOCOL_VERSION,
            "phase": "base_diffusion_generation_validation",
            "sampler": "DDIM",
            "eta": 0.0,
            "schedule_family": config.schedule_family,
            "progressive_nested_anchor_nfe": (
                PROGRESSIVE_NESTED_ANCHOR_NFE
                if config.schedule_family
                == SCHEDULE_FAMILY_PROGRESSIVE_NESTED20
                else None
            ),
            "sampling_steps": list(config.sampling_steps),
            "timestep_schedules": {
                str(steps): values.cpu().tolist()
                for steps, values in timesteps.items()
            },
            "ground_truth": (
                "Fluent p/u/v at identical validation query coordinates"
            ),
            "query_policy": "all CFD query points",
            "metric_policy": {
                "rmse_mae_primary": "subdomain_balanced",
                "rmse_mae_secondary": "all_point_global",
                "r2_correlation_primary": "all_point_global",
            },
            "runtime_scope": (
                "reverse sampling only; excludes I/O, normalization, "
                "metric computation, and output writing"
            ),
            "test_access": "disabled",
        },
        "randomness": {
            "sampling_seed": config.sampling_seed,
            "same_initial_noise_across_schedules": True,
            "sample_seed_rule": (
                "sha256('<seed>:<sample_index>')"
            ),
        },
        "environment": provenance["environment"],
        "outputs": dict(outputs),
        "lifecycle": {
            "created_at_utc": created_at,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "failure": failure,
        },
    }


def print_plan(
    config: Config,
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    selected_indices: Sequence[int],
    timestep_schedules: Mapping[int, torch.Tensor],
) -> None:
    print("Field-diffusion generation evaluation")
    print("  protocol         :", PROTOCOL_VERSION)
    print("  checkpoint       :", config.checkpoint)
    print("  checkpoint epoch :", checkpoint["epoch"])
    validation_meta = checkpoint_validation_metadata(checkpoint)
    metric_name = validation_meta["validation_metric_name"]
    metric_value = validation_meta["validation_metric_value"]
    if metric_name is None:
        print("  checkpoint metric:", "not recorded")
    else:
        print(
            "  checkpoint metric:",
            f"{metric_name}={metric_value:.9g}",
        )
    if checkpoint.get("checkpoint_type") == "progressive_distillation_student":
        print("  distill stage    :", checkpoint.get("stage"))
        print("  student NFE      :", checkpoint.get("student_nfe"))
    print("  validation HDF5  :", config.validation_h5)
    print("  dataset role     :", protocol["dataset_role"])
    print("  selected samples :", len(selected_indices))
    print("  device           :", config.device)
    print("  schedule family  :", config.schedule_family)
    print("  sampling seed    :", config.sampling_seed)
    print("  save predictions :", config.save_predictions)
    print("  DDIM schedules:")
    for steps, values in timestep_schedules.items():
        values = values.cpu().tolist()
        shown = (
            values
            if len(values) <= 12
            else values[:5] + ["..."] + values[-5:]
        )
        print(f"    {steps:4d}: {shown}")
    print("  test access      : disabled")


def run_evaluation(config: Config) -> Path:
    checkpoint = load_checkpoint(config.checkpoint)
    protocol = read_validation_protocol(config.validation_h5)
    validate_protocol(
        checkpoint,
        config.validation_h5,
        protocol,
        config.sampling_steps,
    )

    selected = select_sample_indices(
        protocol["n_samples"],
        config.sample_indices,
        config.max_samples,
    )
    device = resolve_device(config.device)

    diffusion = checkpoint["diffusion_config"]
    schedule = build_linear_schedule(
        timesteps=int(diffusion["T"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        device=device,
    )
    timestep_schedules = {
        steps: build_sampling_timesteps(
            sampling_steps=steps,
            total_diffusion_steps=schedule.timesteps,
            schedule_family=config.schedule_family,
            device=device,
        )
        for steps in config.sampling_steps
    }

    print_plan(
        config,
        checkpoint,
        protocol,
        selected,
        timestep_schedules,
    )

    run_id = config.run_id
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        step_text = "-".join(str(x) for x in config.sampling_steps)
        family_text = (
            "ddim"
            if config.schedule_family == SCHEDULE_FAMILY_STANDARD
            else config.schedule_family
        )
        run_id = (
            f"{stamp}_base_val_{family_text}{step_text}"
            f"_seed{config.sampling_seed}"
        )
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"Unsafe run_id: {run_id!r}")

    run_dir = (
        config.results_root
        / "evaluate_diffusion_generation"
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_path = run_dir / "summary_metrics.csv"
    sample_path = run_dir / "per_sample_metrics.csv"
    schedules_path = run_dir / "sampling_timesteps.json"
    predictions_path = (
        run_dir / "predictions.h5"
        if config.save_predictions
        else None
    )
    outputs = {
        "run_directory": str(run_dir),
        "summary_metrics_csv": str(summary_path),
        "per_sample_metrics_csv": str(sample_path),
        "sampling_timesteps_json": str(schedules_path),
        "predictions_h5": (
            str(predictions_path)
            if predictions_path is not None
            else None
        ),
    }

    with schedules_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                str(steps): values.cpu().tolist()
                for steps, values in timestep_schedules.items()
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    provenance = build_provenance(
        config,
        protocol,
        checkpoint,
    )
    created_at = utc_now()
    write_manifest(
        run_dir,
        make_manifest(
            run_id=run_id,
            status="prepared",
            created_at=created_at,
            started_at=None,
            finished_at=None,
            provenance=provenance,
            config=config,
            selected_indices=selected,
            timesteps=timestep_schedules,
            outputs=outputs,
        ),
    )

    started_at = utc_now()
    update_manifest(
        run_dir,
        make_manifest(
            run_id=run_id,
            status="running",
            created_at=created_at,
            started_at=started_at,
            finished_at=None,
            provenance=provenance,
            config=config,
            selected_indices=selected,
            timesteps=timestep_schedules,
            outputs=outputs,
        ),
    )

    model = build_model(checkpoint, device)
    normalizer = FeatureNormalizer.from_state_dict(
        checkpoint["y_normalizer"]
    ).to(device)
    fields = tuple(checkpoint["output_channel_names"])
    target_dim = len(fields)

    accumulators = {
        steps: {
            field: FieldAccumulator()
            for field in fields
        }
        for steps in config.sampling_steps
    }
    runtime = {
        steps: []
        for steps in config.sampling_steps
    }
    runtime_points = {
        steps: 0
        for steps in config.sampling_steps
    }
    per_sample_rows: list[dict[str, Any]] = []

    predictions_handle: h5py.File | None = None

    try:
        if predictions_path is not None:
            predictions_handle = h5py.File(predictions_path, "x")
            predictions_handle.attrs["protocol_version"] = (
                PROTOCOL_VERSION
            )
            predictions_handle.attrs["validation_h5"] = str(
                config.validation_h5
            )
            predictions_handle.attrs["checkpoint"] = str(
                config.checkpoint
            )
            predictions_handle.attrs["sampling_seed"] = (
                config.sampling_seed
            )
            predictions_handle.attrs["schedule_family"] = (
                config.schedule_family
            )
            if (
                config.schedule_family
                == SCHEDULE_FAMILY_PROGRESSIVE_NESTED20
            ):
                predictions_handle.attrs[
                    "progressive_nested_anchor_nfe"
                ] = PROGRESSIVE_NESTED_ANCHOR_NFE

        with h5py.File(config.validation_h5, "r") as handle:
            first_branch, first_query, _, _ = read_sample(
                handle,
                selected[0],
            )
            first_branch = normalize_diffusion_branch(
                first_branch,
                branch_channel_names=list(
                    checkpoint["branch_channel_names"]
                ),
                target_normalizer=normalizer,
                local_aspect_mean=float(
                    checkpoint["local_aspect_mean"]
                ),
                local_aspect_std=float(
                    checkpoint["local_aspect_std"]
                ),
            )
            warm_up(
                model,
                torch.from_numpy(first_branch).unsqueeze(0).to(device),
                torch.from_numpy(first_query).to(device),
                target_dim,
                schedule.timesteps - 1,
                device,
            )

            for position, sample_index in enumerate(selected, start=1):
                branch_raw, query_raw, truth, metadata = read_sample(
                    handle,
                    sample_index,
                )
                branch_raw = normalize_diffusion_branch(
                    branch_raw,
                    branch_channel_names=list(
                        checkpoint["branch_channel_names"]
                    ),
                    target_normalizer=normalizer,
                    local_aspect_mean=float(
                        checkpoint["local_aspect_mean"]
                    ),
                    local_aspect_std=float(
                        checkpoint["local_aspect_std"]
                    ),
                )

                branch = torch.from_numpy(
                    branch_raw
                ).unsqueeze(0).to(device)
                query = torch.from_numpy(query_raw).to(device)

                sample_seed = stable_sample_seed(
                    config.sampling_seed,
                    sample_index,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(sample_seed)
                initial_noise = torch.randn(
                    (len(query), target_dim),
                    dtype=query.dtype,
                    device=device,
                    generator=generator,
                )

                for steps in config.sampling_steps:
                    pred_std, seconds = timed_sample(
                        model,
                        branch,
                        query,
                        initial_noise,
                        schedule,
                        timestep_schedules[steps],
                        device,
                    )
                    pred = (
                        normalizer.decode(pred_std)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if not np.isfinite(pred).all():
                        raise ValueError(
                            f"Non-finite prediction for sample "
                            f"{sample_index}, DDIM-{steps}"
                        )

                    runtime[steps].append(seconds)
                    runtime_points[steps] += len(query_raw)

                    if predictions_handle is not None:
                        step_group = predictions_handle.require_group(
                            f"ddim_{steps}/samples"
                        )
                        sample_group = step_group.create_group(
                            str(sample_index)
                        )
                        sample_group.create_dataset(
                            "prediction",
                            data=pred,
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                        )
                        for key in (
                            "case_id",
                            "realization_id",
                            "subdomain_id",
                        ):
                            if key in metadata:
                                sample_group.attrs[key] = metadata[key]

                    for field_index, field in enumerate(fields):
                        metrics = accumulators[steps][field].add(
                            truth[:, field_index],
                            pred[:, field_index],
                        )
                        per_sample_rows.append(
                            {
                                "schedule_family": config.schedule_family,
                                "sampling_steps": steps,
                                "sample_index": sample_index,
                                "case_id": metadata.get("case_id", ""),
                                "realization_id": metadata.get(
                                    "realization_id", ""
                                ),
                                "subdomain_id": metadata.get(
                                    "subdomain_id", ""
                                ),
                                "field": field,
                                "unit": FIELD_UNITS[field],
                                "n_points": len(query_raw),
                                "rmse": metrics["rmse"],
                                "mae": metrics["mae"],
                                "r2": metrics["r2"],
                                "correlation": metrics[
                                    "correlation"
                                ],
                                "sampling_seconds": seconds,
                                "sample_seed": sample_seed,
                            }
                        )

                if predictions_handle is not None:
                    predictions_handle.flush()

                if (
                    config.progress_every > 0
                    and (
                        position % config.progress_every == 0
                        or position == len(selected)
                    )
                ):
                    print(
                        f"[Eval] {position:04d}/{len(selected):04d} "
                        "samples completed",
                        flush=True,
                    )

        if predictions_handle is not None:
            predictions_handle.close()
            predictions_handle = None

        write_csv(sample_path, per_sample_rows)

        reference_steps = max(config.sampling_steps)
        reference_runtime = sum(runtime[reference_steps])
        summary_rows: list[dict[str, Any]] = []

        for steps in config.sampling_steps:
            total_seconds = sum(runtime[steps])
            mean_seconds = float(np.mean(runtime[steps]))
            median_seconds = float(np.median(runtime[steps]))
            speedup = reference_runtime / total_seconds
            points_per_second = (
                runtime_points[steps] / total_seconds
            )

            for field in fields:
                summary_rows.append(
                    {
                        "schedule_family": config.schedule_family,
                        "sampling_steps": steps,
                        "field": field,
                        "unit": FIELD_UNITS[field],
                        **accumulators[steps][field].summary(),
                        "sampling_total_seconds": total_seconds,
                        "sampling_mean_seconds_per_subdomain": (
                            mean_seconds
                        ),
                        "sampling_median_seconds_per_subdomain": (
                            median_seconds
                        ),
                        "sampling_points_per_second": (
                            points_per_second
                        ),
                        "speedup_vs_max_steps_in_run": speedup,
                        "reference_sampling_steps": reference_steps,
                        "sampling_seed": config.sampling_seed,
                    }
                )

        write_csv(summary_path, summary_rows)

        finished_at = utc_now()
        update_manifest(
            run_dir,
            make_manifest(
                run_id=run_id,
                status="completed",
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
                provenance=provenance,
                config=config,
                selected_indices=selected,
                timesteps=timestep_schedules,
                outputs=outputs,
            ),
        )

        print("\nCompleted:", run_dir)
        print("Summary :", summary_path)

        print("\nAccuracy")
        for row in summary_rows:
            print(
                f"  DDIM-{row['sampling_steps']:<4d} "
                f"{row['field']:<10s} | "
                f"balanced RMSE="
                f"{row['subdomain_balanced_rmse']:.8g} "
                f"{row['unit']} | "
                f"global RMSE={row['global_rmse']:.8g} "
                f"{row['unit']} | "
                f"R²={row['global_r2']:.6f} | "
                f"corr={row['global_correlation']:.6f}"
            )

        print("\nRuntime")
        for steps in config.sampling_steps:
            total = sum(runtime[steps])
            mean = float(np.mean(runtime[steps]))
            print(
                f"  DDIM-{steps:<4d} | total={total:.3f} s | "
                f"mean/subdomain={mean:.6f} s | "
                f"speedup={reference_runtime / total:.3f}x "
                f"vs DDIM-{reference_steps}"
            )

        return run_dir

    except Exception as exc:
        if predictions_handle is not None:
            predictions_handle.close()

        update_manifest(
            run_dir,
            make_manifest(
                run_id=run_id,
                status="failed",
                created_at=created_at,
                started_at=started_at,
                finished_at=utc_now(),
                provenance=provenance,
                config=config,
                selected_indices=selected,
                timesteps=timestep_schedules,
                outputs=outputs,
                failure=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic DDIM field generation against "
            "Fluent ground truth on canonical random5 validation."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute evaluation; otherwise only print the plan.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--validation-h5",
        type=Path,
        default=DEFAULT_VALIDATION_H5,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--schedule-family",
        choices=(
            SCHEDULE_FAMILY_STANDARD,
            SCHEDULE_FAMILY_PROGRESSIVE_NESTED20,
        ),
        default=SCHEDULE_FAMILY_STANDARD,
        help=(
            "Sampling-grid family. 'standard' preserves the existing "
            "independently spaced DDIM schedules. 'progressive_nested20' "
            "uses the standard 20-NFE grid as the anchor and constructs "
            "exactly nested 20/10/5-NFE schedules for progressive "
            "distillation studies."
        ),
    )
    parser.add_argument(
        "--sampling-steps",
        nargs="+",
        type=int,
        default=[100, 50, 20, 10, 5],
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated fixed validation sample indices.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Evenly spaced deterministic validation subset; "
            "use for smoke tests or DDIM-1000 subset study."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    steps = tuple(int(value) for value in args.sampling_steps)
    if len(steps) != len(set(steps)):
        raise ValueError("--sampling-steps contains duplicates")
    if (
        args.schedule_family
        == SCHEDULE_FAMILY_PROGRESSIVE_NESTED20
        and any(
            step not in PROGRESSIVE_NESTED_ALLOWED_NFE
            for step in steps
        )
    ):
        raise ValueError(
            "--schedule-family progressive_nested20 requires "
            "--sampling-steps to be chosen from 20 10 5"
        )
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    config = Config(
        checkpoint=resolve_path(args.checkpoint),
        validation_h5=resolve_path(args.validation_h5),
        results_root=resolve_path(args.results_root),
        device=args.device,
        schedule_family=args.schedule_family,
        sampling_steps=steps,
        sampling_seed=int(args.sampling_seed),
        sample_indices=parse_sample_indices(args.sample_indices),
        max_samples=args.max_samples,
        save_predictions=bool(args.save_predictions),
        run_id=args.run_id,
        progress_every=int(args.progress_every),
    )

    if args.run:
        run_evaluation(config)
        return

    require_file(config.checkpoint, "Checkpoint")
    require_file(config.validation_h5, "Validation HDF5")

    checkpoint = load_checkpoint(config.checkpoint)
    protocol = read_validation_protocol(config.validation_h5)
    validate_protocol(
        checkpoint,
        config.validation_h5,
        protocol,
        config.sampling_steps,
    )

    selected = select_sample_indices(
        protocol["n_samples"],
        config.sample_indices,
        config.max_samples,
    )
    total_steps = int(checkpoint["diffusion_config"]["T"])
    timestep_schedules = {
        steps: build_sampling_timesteps(
            sampling_steps=steps,
            total_diffusion_steps=total_steps,
            schedule_family=config.schedule_family,
            device="cpu",
        )
        for steps in config.sampling_steps
    }

    print_plan(
        config,
        checkpoint,
        protocol,
        selected,
        timestep_schedules,
    )
    print("\nEvaluation was not started. Pass --run to execute it.")


if __name__ == "__main__":
    main()
