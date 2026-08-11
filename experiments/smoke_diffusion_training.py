"""Run a small GPU throughput and gradient diagnostic for field diffusion.

This script is diagnostic-only. It does not save checkpoints, does not run
validation, and does not read either test dataset. The default comparison uses
the canonical random10 training HDF5 with the same model, normalization,
query cap, optimizer, and epsilon objective as the formal training entrypoint.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pidiffusion.data import (  # noqa: E402
    DiffusionCellDataset,
    collate_diffusion_batch,
    fit_subdomain_balanced_normalizers,
    load_diffusion_dataset,
)
from pidiffusion.diffusion import (  # noqa: E402
    build_linear_schedule,
    epsilon_prediction_loss,
)
from pidiffusion.model import PointSetDiffusionDenoiser  # noqa: E402


DEFAULT_TRAIN_DATASET = (
    REPOSITORY_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random10_train.h5"
)


@dataclass(frozen=True)
class SmokeResult:
    batch_size: int
    status: str
    steps: int = 0
    samples: int = 0
    query_points: int = 0
    elapsed_seconds: float = math.nan
    seconds_per_step: float = math.nan
    samples_per_second: float = math.nan
    query_points_per_second: float = math.nan
    peak_allocated_gib: float = math.nan
    peak_reserved_gib: float = math.nan
    loss_mean: float = math.nan
    grad_norm_mean: float = math.nan
    grad_norm_p50: float = math.nan
    grad_norm_p95: float = math.nan
    grad_norm_p99: float = math.nan
    grad_norm_max: float = math.nan
    nonfinite_grad_steps: int = 0
    message: str = ""


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve(strict=False)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_cuda_device(raw_device: str) -> torch.device:
    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device: {raw_device!r}") from exc

    if device.type != "cuda":
        raise ValueError(
            "This diagnostic measures CUDA throughput and VRAM; use an explicit "
            "CUDA device such as --device cuda:1."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")
    if device.index is None:
        raise ValueError("CUDA device must include an explicit index, for example cuda:1.")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but torch sees only {torch.cuda.device_count()} CUDA device(s)."
        )
    return device


def _validate_training_dataset(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {path}")

    with h5py.File(path, "r") as handle:
        role = handle.attrs.get("dataset_role", "")
        placement = handle.attrs.get("interface_placement", "")
        n_samples = int(handle.attrs.get("n_samples", -1))
        n_subdomains = int(handle.attrs.get("n_subdomains", -1))
        n_realizations = int(handle.attrs.get("n_realizations", -1))

    if isinstance(role, bytes):
        role = role.decode("utf-8")
    if isinstance(placement, bytes):
        placement = placement.decode("utf-8")

    if role != "shared_randomized_training":
        raise ValueError(
            "Expected dataset_role='shared_randomized_training', got "
            f"{role!r}"
        )
    if placement != "random":
        raise ValueError(f"Expected interface_placement='random', got {placement!r}")
    if n_samples != 16000:
        raise ValueError(f"Expected 16000 training samples, got {n_samples}")
    if n_subdomains != 10:
        raise ValueError(f"Expected n_subdomains=10, got {n_subdomains}")
    if n_realizations != 10:
        raise ValueError(f"Expected n_realizations=10, got {n_realizations}")


def _build_model(branch_dim: int, query_dim: int, target_dim: int) -> PointSetDiffusionDenoiser:
    model = PointSetDiffusionDenoiser(
        branch_input_dim=int(branch_dim),
        query_input_dim=int(query_dim),
        target_dim=int(target_dim),
        latent_dim=128,
        time_dim=128,
        branch_point_hidden_dim=128,
        branch_global_hidden_dim=128,
        denoiser_hidden_dim=256,
        denoiser_depth=4,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 382083:
        raise RuntimeError(
            f"Canonical model parameter count changed: {parameter_count} != 382083"
        )
    return model


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


def _move_batch(batch, device: torch.device):
    if len(batch) == 6:
        branch, query, target, query_batch_id, sample_idx, branch_mask = batch
    else:
        branch, query, target, query_batch_id, sample_idx = batch
        branch_mask = None

    branch = branch.to(device)
    query = query.to(device)
    target = target.to(device)
    query_batch_id = query_batch_id.to(device)
    sample_idx = sample_idx.to(device)
    if branch_mask is not None:
        branch_mask = branch_mask.to(device)
    return branch, query, target, query_batch_id, sample_idx, branch_mask


def _make_loader(
    *,
    samples,
    sample_indices: np.ndarray,
    batch_size: int,
    num_query_points: int,
    target_normalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    branch_channel_names,
) -> DataLoader:
    dataset = DiffusionCellDataset(
        samples=samples,
        sample_indices=sample_indices,
        n_query_points=num_query_points,
        random_query=True,
        target_normalizer=target_normalizer,
        local_aspect_mean=local_aspect_mean,
        local_aspect_std=local_aspect_std,
        branch_channel_names=branch_channel_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_diffusion_batch,
        drop_last=False,
    )


def _run_batches(
    *,
    model: PointSetDiffusionDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule,
    device: torch.device,
    boundary_loss_weight: float,
    collect_metrics: bool,
) -> dict[str, object]:
    losses: list[float] = []
    grad_norms: list[float] = []
    nonfinite_grad_steps = 0
    samples_seen = 0
    query_points_seen = 0
    steps = 0

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
            boundary_loss_weight=boundary_loss_weight,
        )
        loss.backward()

        raw_grad_norm = _global_grad_norm(list(model.parameters()))
        if not math.isfinite(raw_grad_norm):
            nonfinite_grad_steps += 1
            raise RuntimeError("Encountered a non-finite gradient norm during smoke test.")

        optimizer.step()

        if collect_metrics:
            losses.append(float(loss.item()))
            grad_norms.append(raw_grad_norm)
            samples_seen += int(branch.shape[0])
            query_points_seen += int(query.shape[0])
            steps += 1

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "samples": samples_seen,
        "query_points": query_points_seen,
        "steps": steps,
    }


def _run_candidate(
    *,
    batch_size: int,
    data,
    warmup_indices: np.ndarray,
    measure_indices: np.ndarray,
    target_normalizer,
    local_aspect_mean: float,
    local_aspect_std: float,
    device: torch.device,
    seed: int,
    num_query_points: int,
    learning_rate: float,
    weight_decay: float,
    boundary_loss_weight: float,
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> SmokeResult:
    if len(warmup_indices) % batch_size != 0:
        raise ValueError(
            f"warmup sample count {len(warmup_indices)} must be divisible by batch size {batch_size}"
        )
    if len(measure_indices) % batch_size != 0:
        raise ValueError(
            f"measured sample count {len(measure_indices)} must be divisible by batch size {batch_size}"
        )

    torch.cuda.empty_cache()
    _set_seed(seed)

    model = None
    optimizer = None
    try:
        branch_dim = len(data["branch_channel_names"])
        query_dim = len(data["trunk_channel_names"])
        target_dim = len(data["output_channel_names"])

        model = _build_model(branch_dim, query_dim, target_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        schedule = build_linear_schedule(
            timesteps=int(timesteps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            device=device,
        )

        # Reset NumPy/Torch RNGs after model initialization so every candidate
        # sees the same sample order and reproducible query-subset stream.
        np.random.seed(int(seed) + 1000)
        torch.manual_seed(int(seed) + 2000)
        torch.cuda.manual_seed_all(int(seed) + 2000)

        warmup_loader = _make_loader(
            samples=data["samples"],
            sample_indices=warmup_indices,
            batch_size=batch_size,
            num_query_points=num_query_points,
            target_normalizer=target_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            branch_channel_names=data["branch_channel_names"],
        )
        model.train()
        _run_batches(
            model=model,
            loader=warmup_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            boundary_loss_weight=boundary_loss_weight,
            collect_metrics=False,
        )

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # Use a new but fixed stream for the measured section. Because sample
        # order is identical, each candidate uses the same query subset per
        # measured subdomain even though samples are grouped into different batches.
        np.random.seed(int(seed) + 3000)
        torch.manual_seed(int(seed) + 4000)
        torch.cuda.manual_seed_all(int(seed) + 4000)

        measured_loader = _make_loader(
            samples=data["samples"],
            sample_indices=measure_indices,
            batch_size=batch_size,
            num_query_points=num_query_points,
            target_normalizer=target_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            branch_channel_names=data["branch_channel_names"],
        )

        start = time.perf_counter()
        metrics = _run_batches(
            model=model,
            loader=measured_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            boundary_loss_weight=boundary_loss_weight,
            collect_metrics=True,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024.0**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0**3)

        grad = np.asarray(metrics["grad_norms"], dtype=np.float64)
        losses = np.asarray(metrics["losses"], dtype=np.float64)
        steps = int(metrics["steps"])
        samples_seen = int(metrics["samples"])
        query_points_seen = int(metrics["query_points"])

        return SmokeResult(
            batch_size=batch_size,
            status="PASS",
            steps=steps,
            samples=samples_seen,
            query_points=query_points_seen,
            elapsed_seconds=elapsed,
            seconds_per_step=elapsed / max(steps, 1),
            samples_per_second=samples_seen / max(elapsed, 1.0e-12),
            query_points_per_second=query_points_seen / max(elapsed, 1.0e-12),
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
            loss_mean=float(losses.mean()) if losses.size else math.nan,
            grad_norm_mean=float(grad.mean()) if grad.size else math.nan,
            grad_norm_p50=float(np.percentile(grad, 50.0)) if grad.size else math.nan,
            grad_norm_p95=float(np.percentile(grad, 95.0)) if grad.size else math.nan,
            grad_norm_p99=float(np.percentile(grad, 99.0)) if grad.size else math.nan,
            grad_norm_max=float(grad.max()) if grad.size else math.nan,
            nonfinite_grad_steps=int(metrics["nonfinite_grad_steps"]),
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return SmokeResult(
            batch_size=batch_size,
            status="OOM",
            message=str(exc).splitlines()[0][:240],
        )
    finally:
        del optimizer
        del model
        torch.cuda.empty_cache()


def _print_result(result: SmokeResult, total_memory_gib: float) -> None:
    print("\n" + "=" * 100)
    print(f"Batch size {result.batch_size}: {result.status}")
    if result.status != "PASS":
        print("message              :", result.message)
        return

    alloc_pct = 100.0 * result.peak_allocated_gib / max(total_memory_gib, 1.0e-12)
    reserved_pct = 100.0 * result.peak_reserved_gib / max(total_memory_gib, 1.0e-12)

    print("measured steps        :", result.steps)
    print("measured samples      :", result.samples)
    print("measured query points :", f"{result.query_points:,}")
    print("elapsed               :", f"{result.elapsed_seconds:.3f} s")
    print("seconds / step        :", f"{result.seconds_per_step:.5f}")
    print("samples / second      :", f"{result.samples_per_second:.2f}")
    print("query points / second :", f"{result.query_points_per_second:,.0f}")
    print(
        "peak allocated VRAM   :",
        f"{result.peak_allocated_gib:.3f} GiB ({alloc_pct:.1f}% of device)",
    )
    print(
        "peak reserved VRAM    :",
        f"{result.peak_reserved_gib:.3f} GiB ({reserved_pct:.1f}% of device)",
    )
    print("mean training loss    :", f"{result.loss_mean:.6f}")
    print("grad norm mean        :", f"{result.grad_norm_mean:.6f}")
    print("grad norm p50         :", f"{result.grad_norm_p50:.6f}")
    print("grad norm p95         :", f"{result.grad_norm_p95:.6f}")
    print("grad norm p99         :", f"{result.grad_norm_p99:.6f}")
    print("grad norm max         :", f"{result.grad_norm_max:.6f}")
    print("non-finite grad steps :", result.nonfinite_grad_steps)


def _print_summary(results: Sequence[SmokeResult], total_memory_gib: float) -> None:
    print("\n" + "=" * 132)
    print("SUMMARY")
    print("=" * 132)
    header = (
        f"{'batch':>6} {'status':>7} {'step s':>10} {'samples/s':>11} "
        f"{'qpoints/s':>13} {'alloc GiB':>10} {'alloc %':>8} "
        f"{'grad p50':>11} {'grad p95':>11} {'grad p99':>11} {'grad max':>11}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        if result.status != "PASS":
            print(f"{result.batch_size:6d} {result.status:>7}")
            continue
        alloc_pct = 100.0 * result.peak_allocated_gib / max(total_memory_gib, 1.0e-12)
        print(
            f"{result.batch_size:6d} "
            f"{result.status:>7} "
            f"{result.seconds_per_step:10.5f} "
            f"{result.samples_per_second:11.2f} "
            f"{result.query_points_per_second:13.0f} "
            f"{result.peak_allocated_gib:10.3f} "
            f"{alloc_pct:8.1f} "
            f"{result.grad_norm_p50:11.5f} "
            f"{result.grad_norm_p95:11.5f} "
            f"{result.grad_norm_p99:11.5f} "
            f"{result.grad_norm_max:11.5f}"
        )

    print(
        "\nNo batch size or clipping threshold is selected automatically. "
        "Use this diagnostic to choose them before formal training."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only field-diffusion GPU smoke test. No checkpoints are "
            "saved and no validation/test dataset is read."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the GPU diagnostic; without this flag only the configuration is printed.",
    )
    parser.add_argument(
        "--train-dataset",
        default=str(DEFAULT_TRAIN_DATASET),
        help="Canonical random10 training HDF5.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Batch sizes to compare.",
    )
    parser.add_argument("--num-query-points", type=int, default=8192)
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=64,
        help="Fixed number of warm-up subdomains per batch-size candidate.",
    )
    parser.add_argument(
        "--measure-samples",
        type=int,
        default=512,
        help="Fixed number of measured subdomains per batch-size candidate.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=12345,
        help="Seed selecting the fixed warm-up/measured training subdomains.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1.0e-4)
    parser.add_argument("--beta-end", type=float, default=2.0e-2)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.batch_sizes:
        raise ValueError("At least one batch size is required.")
    if any(value <= 0 for value in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must be unique.")
    if args.num_query_points <= 0:
        raise ValueError("num-query-points must be positive.")
    if args.warmup_samples <= 0 or args.measure_samples <= 0:
        raise ValueError("warmup-samples and measure-samples must be positive.")
    for batch_size in args.batch_sizes:
        if args.warmup_samples % batch_size != 0:
            raise ValueError(
                f"warmup-samples={args.warmup_samples} is not divisible by batch={batch_size}"
            )
        if args.measure_samples % batch_size != 0:
            raise ValueError(
                f"measure-samples={args.measure_samples} is not divisible by batch={batch_size}"
            )
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative.")
    if args.boundary_loss_weight < 0.0:
        raise ValueError("boundary-loss-weight must be non-negative.")
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    if not (0.0 < args.beta_start < 1.0 and 0.0 < args.beta_end < 1.0):
        raise ValueError("beta-start and beta-end must lie in (0, 1).")
    if args.beta_start >= args.beta_end:
        raise ValueError("beta-start must be smaller than beta-end.")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    train_path = _resolve_repo_path(args.train_dataset)

    print("Field-diffusion GPU smoke configuration")
    print("  train dataset        :", train_path)
    print("  device               :", args.device)
    print("  batch sizes          :", args.batch_sizes)
    print("  query cap            :", args.num_query_points)
    print("  warm-up samples      :", args.warmup_samples)
    print("  measured samples     :", args.measure_samples)
    print("  sample seed          :", args.sample_seed)
    print("  model/training seed  :", args.seed)
    print("  learning rate        :", args.learning_rate)
    print("  weight decay         :", args.weight_decay)
    print("  boundary loss weight :", args.boundary_loss_weight)
    print("  gradient clipping    : disabled")
    print("  scheduler            : none")
    print("  checkpoints          : none")
    print("  validation/test data : not loaded")

    if not args.run:
        print("\nGPU diagnostic was not started. Pass --run to execute it.")
        return 0

    _validate_training_dataset(train_path)
    device = _resolve_cuda_device(args.device)
    properties = torch.cuda.get_device_properties(device)
    total_memory_gib = properties.total_memory / (1024.0**3)

    print("\nCUDA device")
    print("  name                 :", properties.name)
    print("  total memory         :", f"{total_memory_gib:.2f} GiB")
    print("  torch CUDA           :", torch.version.cuda)

    print("\nLoading canonical training dataset...")
    data = load_diffusion_dataset(train_path)

    print("Fitting full train-only subdomain-balanced normalizers...")
    target_normalizer, local_aspect_mean, local_aspect_std = (
        fit_subdomain_balanced_normalizers(data)
    )
    print(
        "  target mean          :",
        [float(value) for value in target_normalizer.mean.tolist()],
    )
    print(
        "  target std           :",
        [float(value) for value in target_normalizer.std.tolist()],
    )
    print("  local aspect mean    :", local_aspect_mean)
    print("  local aspect std     :", local_aspect_std)

    needed = int(args.warmup_samples + args.measure_samples)
    if needed > len(data["samples"]):
        raise ValueError(
            f"Smoke test needs {needed} unique samples but dataset has only "
            f"{len(data['samples'])}."
        )

    sample_rng = np.random.default_rng(int(args.sample_seed))
    selected = sample_rng.choice(
        len(data["samples"]),
        size=needed,
        replace=False,
    ).astype(np.int64)
    warmup_indices = selected[: args.warmup_samples]
    measure_indices = selected[args.warmup_samples :]

    print("\nFixed diagnostic sample pool")
    print("  warm-up samples      :", len(warmup_indices))
    print("  measured samples     :", len(measure_indices))
    print("  overlap              :", len(set(warmup_indices) & set(measure_indices)))

    results: list[SmokeResult] = []
    for batch_size in args.batch_sizes:
        print("\n" + "#" * 100)
        print(f"Running batch size {batch_size}")
        result = _run_candidate(
            batch_size=int(batch_size),
            data=data,
            warmup_indices=warmup_indices,
            measure_indices=measure_indices,
            target_normalizer=target_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            device=device,
            seed=int(args.seed),
            num_query_points=int(args.num_query_points),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            boundary_loss_weight=float(args.boundary_loss_weight),
            timesteps=int(args.timesteps),
            beta_start=float(args.beta_start),
            beta_end=float(args.beta_end),
        )
        results.append(result)
        _print_result(result, total_memory_gib)

    _print_summary(results, total_memory_gib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
