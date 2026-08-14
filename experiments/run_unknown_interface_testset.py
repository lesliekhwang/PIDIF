#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "distill_progressive"
    / "distill_nested10_to5_stage2_5ep_seed0"
    / "stage2_best.pt"
)

DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "run_unknown_interface_testset"
)

DEFAULT_OPTIMIZER_SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "optimize_unknown_interfaces.py"
)

DEFAULT_EVALUATOR_SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "evaluate_unknown_interface_solution.py"
)

INTERFACE_STATES = (
    ("known", "known"),
    ("initial", "zero"),
    ("final", "physics"),
)

FIELDS = ("pressure", "u", "v")


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_case_tag(case_id: str) -> str:
    chars = []
    for char in str(case_id):
        if char.isalnum() or char in ("-", "_"):
            chars.append(char)
        else:
            chars.append("_")
    tag = "".join(chars).strip("_")
    if not tag:
        raise ValueError(f"Could not make safe tag from case_id={case_id!r}")
    return tag


def discover_case_realizations(
    dataset_path: Path,
    *,
    expected_subdomains: int = 10,
) -> list[dict[str, Any]]:
    """
    Discover complete (case_id, realization_id) channel-level groups directly
    from the HDF5 metadata table.

    Expected schema:
      metadata/case_id
      metadata/realization_id
      metadata/subdomain_id
      samples/<sample_index>/...
    """
    with h5py.File(dataset_path, "r") as handle:
        if "metadata" not in handle:
            raise KeyError("Missing HDF5 group: metadata")
        if "samples" not in handle:
            raise KeyError("Missing HDF5 group: samples")

        metadata = handle["metadata"]
        for key in ("case_id", "realization_id", "subdomain_id"):
            if key not in metadata:
                raise KeyError(f"Missing HDF5 metadata/{key}")

        case_ids = metadata["case_id"][:]
        realization_ids = metadata["realization_id"][:]
        subdomain_ids = metadata["subdomain_id"][:]

        n = len(case_ids)
        if len(realization_ids) != n or len(subdomain_ids) != n:
            raise RuntimeError("Metadata arrays have inconsistent lengths")

        groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)

        for sample_index in range(n):
            case_id = str(decode_scalar(case_ids[sample_index]))
            realization_id = int(decode_scalar(realization_ids[sample_index]))
            subdomain_id = int(decode_scalar(subdomain_ids[sample_index]))
            groups[(case_id, realization_id)].append(
                (subdomain_id, sample_index)
            )

        records: list[dict[str, Any]] = []

        for (case_id, realization_id), members in groups.items():
            members.sort(key=lambda item: item[0])
            subdomain_list = [item[0] for item in members]
            sample_indices = [item[1] for item in members]

            expected_ids = list(range(expected_subdomains))
            if subdomain_list != expected_ids:
                raise RuntimeError(
                    "Incomplete or unexpected channel decomposition for "
                    f"case={case_id}, realization={realization_id}: "
                    f"subdomains={subdomain_list}, expected={expected_ids}"
                )

            for sample_index in sample_indices:
                sample_key = str(sample_index)
                if sample_key not in handle["samples"]:
                    raise KeyError(f"Missing samples/{sample_key}")

            records.append(
                {
                    "case_id": case_id,
                    "realization_id": realization_id,
                    "sample_indices": sample_indices,
                    "subdomain_ids": subdomain_list,
                }
            )

    records.sort(
        key=lambda item: (
            item["case_id"],
            int(item["realization_id"]),
        )
    )
    return records


def run_help(python_executable: str, script_path: Path) -> str:
    completed = subprocess.run(
        [python_executable, str(script_path), "--help"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not read --help for {script_path}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def require_cli_flags(
    *,
    python_executable: str,
    script_path: Path,
    required_flags: Iterable[str],
    label: str,
) -> None:
    help_text = run_help(python_executable, script_path)
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(
            f"{label} CLI is incompatible with this batch runner.\n"
            f"Missing flags: {missing}\n\n"
            f"{script_path} --help output:\n{help_text}"
        )


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_logged_command(
    *,
    command: list[str],
    log_path: Path,
    cwd: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print(command_to_string(command))
    print(f"log: {log_path}")
    print("=" * 100)
    sys.stdout.flush()

    started = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(command_to_string(command) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return_code = process.wait()

    elapsed = time.perf_counter() - started

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with return code {return_code}: "
            f"{command_to_string(command)}\n"
            f"See log: {log_path}"
        )

    return elapsed


def optimization_complete(run_dir: Path) -> bool:
    required = (
        "config.json",
        "physics_loss_history.csv",
        "interface_states.npz",
        "summary.json",
    )
    return run_dir.is_dir() and all(
        (run_dir / name).is_file()
        for name in required
    )


def evaluation_complete(run_dir: Path) -> bool:
    required = (
        "config.json",
        "metrics_per_subdomain.csv",
        "metrics_summary.csv",
        "predictions.h5",
        "summary.json",
    )
    return run_dir.is_dir() and all(
        (run_dir / name).is_file()
        for name in required
    )


def verify_zero_initialization(optimization_dir: Path) -> None:
    states_path = optimization_dir / "interface_states.npz"
    with np.load(states_path) as data:
        if "z_initial_physical" not in data:
            raise KeyError(
                f"{states_path} does not contain z_initial_physical"
            )
        z_initial = np.asarray(
            data["z_initial_physical"],
            dtype=np.float64,
        )

    if z_initial.shape != (9, 256, 3):
        raise RuntimeError(
            f"Unexpected z_initial_physical shape {z_initial.shape}; "
            "expected (9, 256, 3)"
        )

    if not np.array_equal(z_initial, np.zeros_like(z_initial)):
        max_abs = float(np.max(np.abs(z_initial)))
        raise RuntimeError(
            "Initial interface state is not exactly physical zero. "
            f"max_abs={max_abs:.8e}"
        )


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size < 2:
        return float("nan")

    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if true_std == 0.0 or pred_std == 0.0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    residual = y_true - y_pred
    sse = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    sst = float(np.sum(centered * centered))

    if sst == 0.0:
        return float("nan")
    return float(1.0 - sse / sst)


def load_prediction_groups(
    predictions_path: Path,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    groups: list[tuple[int, np.ndarray, np.ndarray]] = []

    with h5py.File(predictions_path, "r") as handle:
        if "samples" not in handle:
            raise KeyError(f"Missing samples group in {predictions_path}")

        for key in handle["samples"].keys():
            group = handle["samples"][key]

            if "prediction" not in group or "target" not in group:
                raise KeyError(
                    f"Missing prediction/target under samples/{key}"
                )

            pred = np.asarray(
                group["prediction"][:],
                dtype=np.float64,
            )
            truth = np.asarray(
                group["target"][:],
                dtype=np.float64,
            )

            if pred.shape != truth.shape:
                raise RuntimeError(
                    f"Shape mismatch in samples/{key}: "
                    f"pred={pred.shape}, truth={truth.shape}"
                )
            if pred.ndim != 2 or pred.shape[1] != 3:
                raise RuntimeError(
                    f"Expected (N,3) prediction in samples/{key}, "
                    f"got {pred.shape}"
                )

            if "subdomain_id" in group.attrs:
                subdomain_id = int(group.attrs["subdomain_id"])
            else:
                subdomain_id = len(groups)

            groups.append((subdomain_id, pred, truth))

    groups.sort(key=lambda item: item[0])

    ids = [item[0] for item in groups]
    if ids != list(range(10)):
        raise RuntimeError(
            f"Expected subdomain IDs 0..9 in {predictions_path}, got {ids}"
        )

    return groups


def compute_channel_metrics(
    *,
    predictions_path: Path,
    testset_name: str,
    case_id: str,
    realization_id: int,
    state_label: str,
) -> list[dict[str, Any]]:
    groups = load_prediction_groups(predictions_path)

    rows: list[dict[str, Any]] = []

    for field_index, field_name in enumerate(FIELDS):
        per_subdomain_mse: list[float] = []
        per_subdomain_mae: list[float] = []
        per_subdomain_rel_l2: list[float] = []

        truth_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []

        for _, pred, truth in groups:
            y_pred = pred[:, field_index]
            y_true = truth[:, field_index]
            error = y_pred - y_true

            per_subdomain_mse.append(
                float(np.mean(error * error))
            )
            per_subdomain_mae.append(
                float(np.mean(np.abs(error)))
            )

            denominator = float(np.linalg.norm(y_true))
            if denominator > 0.0:
                rel_l2 = float(
                    np.linalg.norm(error) / denominator
                )
            else:
                rel_l2 = float("nan")
            per_subdomain_rel_l2.append(rel_l2)

            truth_parts.append(y_true)
            pred_parts.append(y_pred)

        y_true_all = np.concatenate(truth_parts)
        y_pred_all = np.concatenate(pred_parts)

        rows.append(
            {
                "testset": testset_name,
                "case_id": case_id,
                "realization_id": int(realization_id),
                "state": state_label,
                "field": field_name,
                "n_subdomains": 10,
                "n_points": int(y_true_all.size),
                "balanced_rmse": float(
                    math.sqrt(np.mean(per_subdomain_mse))
                ),
                "balanced_mae": float(
                    np.mean(per_subdomain_mae)
                ),
                "avg_relative_l2": float(
                    np.nanmean(per_subdomain_rel_l2)
                ),
                "global_r2": r2_score_numpy(
                    y_true_all,
                    y_pred_all,
                ),
                "global_corr": pearson_corr(
                    y_true_all,
                    y_pred_all,
                ),
                "max_abs_error": float(
                    np.max(np.abs(y_pred_all - y_true_all))
                ),
                "p995_abs_error": float(
                    np.quantile(
                        np.abs(y_pred_all - y_true_all),
                        0.995,
                    )
                ),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(
    rows: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    values = np.asarray(
        [float(row[key]) for row in rows],
        dtype=np.float64,
    )
    return values[np.isfinite(values)]


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_testset_summary(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = (
        "balanced_rmse",
        "balanced_mae",
        "avg_relative_l2",
        "global_r2",
        "global_corr",
        "max_abs_error",
        "p995_abs_error",
    )

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {
        "states": {},
        "avg_relative_l2_over_fields": {},
    }

    states = sorted({str(row["state"]) for row in rows})

    for state in states:
        summary_json["states"][state] = {}

        for field in FIELDS:
            subset = [
                row
                for row in rows
                if row["state"] == state
                and row["field"] == field
            ]

            if not subset:
                continue

            field_summary: dict[str, Any] = {}

            for metric in metrics:
                stats = summarize_values(
                    finite_values(subset, metric)
                )
                field_summary[metric] = stats

                summary_rows.append(
                    {
                        "state": state,
                        "field": field,
                        "metric": metric,
                        **stats,
                    }
                )

            summary_json["states"][state][field] = field_summary

        by_channel: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            if row["state"] != state:
                continue
            by_channel[
                (
                    str(row["case_id"]),
                    int(row["realization_id"]),
                )
            ].append(float(row["avg_relative_l2"]))

        channel_avg_values = np.asarray(
            [
                float(np.mean(values))
                for values in by_channel.values()
                if len(values) == 3
                and np.isfinite(values).all()
            ],
            dtype=np.float64,
        )

        avg_stats = summarize_values(channel_avg_values)
        summary_json["avg_relative_l2_over_fields"][state] = avg_stats

        summary_rows.append(
            {
                "state": state,
                "field": "avg_p_u_v",
                "metric": "avg_relative_l2",
                **avg_stats,
            }
        )

    return summary_rows, summary_json


def optimizer_command(
    *,
    python_executable: str,
    optimizer_script: Path,
    checkpoint: Path,
    dataset: Path,
    device: str,
    case_id: str,
    realization_id: int,
    results_root: Path,
    run_name: str,
    outer_max_iter: int,
    lbfgs_inner_max_iter: int,
    lr: float,
    tol: float,
    stagnation_patience: int,
    noise_seed_left: int,
    noise_seed_right: int,
) -> list[str]:
    return [
        python_executable,
        str(optimizer_script),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--device",
        device,
        "--case-id",
        case_id,
        "--realization-id",
        str(realization_id),
        "--outer-max-iter",
        str(outer_max_iter),
        "--lbfgs-inner-max-iter",
        str(lbfgs_inner_max_iter),
        "--lr",
        str(lr),
        "--tol",
        str(tol),
        "--stagnation-patience",
        str(stagnation_patience),
        "--noise-seed-left",
        str(noise_seed_left),
        "--noise-seed-right",
        str(noise_seed_right),
        "--results-root",
        str(results_root),
        "--run-name",
        run_name,
        "--run",
    ]


def evaluator_command(
    *,
    python_executable: str,
    evaluator_script: Path,
    optimization_run: Path,
    interface_state: str,
    device: str,
    results_root: Path,
    run_name: str,
) -> list[str]:
    return [
        python_executable,
        str(evaluator_script),
        "--optimization-run",
        str(optimization_run),
        "--interface-state",
        interface_state,
        "--device",
        device,
        "--results-root",
        str(results_root),
        "--run-name",
        run_name,
        "--run",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch unknown-interface physics optimization and A/B/C "
            "whole-field evaluation over one complete test HDF5 dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "One frozen test HDF5 dataset, e.g. control-points test "
            "or AR1 test."
        ),
    )
    parser.add_argument(
        "--testset-name",
        required=True,
        help="Short neutral label such as cp_test or ar1_test.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--optimizer-script",
        type=Path,
        default=DEFAULT_OPTIMIZER_SCRIPT,
    )
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=DEFAULT_EVALUATOR_SCRIPT,
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child scripts.",
    )

    # Frozen physics-optimization protocol.
    parser.add_argument("--outer-max-iter", type=int, default=50)
    parser.add_argument("--lbfgs-inner-max-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--stagnation-patience", type=int, default=10)
    parser.add_argument("--noise-seed-left", type=int, default=1000)
    parser.add_argument("--noise-seed-right", type=int, default=2000)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of discovered channel-level cases to process. "
            "Use --limit 1 for an end-to-end smoke run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing batch root by skipping only complete "
            "optimization/evaluation artifacts. Partial artifacts still fail."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Actually launch GPU optimization/evaluation. Without --run, "
            "only discover cases, validate CLIs, and print the plan."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    optimizer_script = args.optimizer_script.resolve()
    evaluator_script = args.evaluator_script.resolve()
    results_root = args.results_root.resolve()

    require_file(dataset, "dataset")
    require_file(checkpoint, "checkpoint")
    require_file(optimizer_script, "optimizer script")
    require_file(evaluator_script, "evaluator script")

    if args.outer_max_iter <= 0:
        raise ValueError("--outer-max-iter must be positive")
    if args.lbfgs_inner_max_iter <= 0:
        raise ValueError("--lbfgs-inner-max-iter must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.tol <= 0:
        raise ValueError("--tol must be positive")
    if args.stagnation_patience <= 0:
        raise ValueError("--stagnation-patience must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    # Fail closed if the child script APIs changed.
    require_cli_flags(
        python_executable=args.python,
        script_path=optimizer_script,
        label="optimizer",
        required_flags=(
            "--checkpoint",
            "--dataset",
            "--device",
            "--case-id",
            "--realization-id",
            "--outer-max-iter",
            "--lbfgs-inner-max-iter",
            "--lr",
            "--tol",
            "--stagnation-patience",
            "--noise-seed-left",
            "--noise-seed-right",
            "--results-root",
            "--run-name",
            "--run",
        ),
    )

    require_cli_flags(
        python_executable=args.python,
        script_path=evaluator_script,
        label="evaluator",
        required_flags=(
            "--optimization-run",
            "--interface-state",
            "--device",
            "--results-root",
            "--run-name",
        ),
    )

    cases = discover_case_realizations(dataset)
    if args.limit is not None:
        cases = cases[: args.limit]

    batch_root = results_root / args.testset_name
    optimization_root = batch_root / "optimizations"
    evaluation_root = batch_root / "evaluations"
    logs_root = batch_root / "logs"

    print()
    print("=" * 100)
    print("UNKNOWN-INTERFACE TEST-SET PLAN")
    print("=" * 100)
    print(f"Dataset              : {dataset}")
    print(f"Test-set name        : {args.testset_name}")
    print(f"Checkpoint           : {checkpoint}")
    print(f"Device               : {args.device}")
    print(f"Discovered channels  : {len(cases)}")
    print(f"Results root         : {batch_root}")
    print()
    print("Frozen optimizer protocol")
    print(f"  outer_max_iter     : {args.outer_max_iter}")
    print(f"  LBFGS inner max    : {args.lbfgs_inner_max_iter}")
    print(f"  lr                 : {args.lr:g}")
    print(f"  tol                : {args.tol:.3e}")
    print(f"  stagnation         : {args.stagnation_patience}")
    print(f"  noise seed left    : {args.noise_seed_left}")
    print(f"  noise seed right   : {args.noise_seed_right}")
    print()
    print("Per channel")
    print("  1. optimize all 9 unknown internal interfaces jointly")
    print("  2. evaluate known interface oracle")
    print("  3. evaluate zero-initialized/no-physics baseline")
    print("  4. evaluate physics-optimized unknown interfaces")
    print()
    for index, case in enumerate(cases):
        print(
            f"  [{index:02d}] "
            f"{case['case_id']} real={case['realization_id']} "
            f"samples={case['sample_indices']}"
        )
    print("=" * 100)

    if not args.run:
        print()
        print("DRY RUN ONLY")
        print(
            "No optimization or evaluation was launched. "
            "Add --run to execute."
        )
        return

    if batch_root.exists() and not args.resume:
        raise FileExistsError(
            f"Batch root already exists: {batch_root}\n"
            "Use a new --testset-name, or use --resume only if this is "
            "the same intended batch run."
        )

    batch_root.mkdir(parents=True, exist_ok=args.resume)
    optimization_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    config_path = batch_root / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(config_path)

    if not config_path.exists():
        config_record = {
            "protocol": "distilled5_unknown_interface_testset_v1",
            "testset_name": args.testset_name,
            "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_script": str(optimizer_script),
            "optimizer_script_sha256": sha256_file(optimizer_script),
            "evaluator_script": str(evaluator_script),
            "evaluator_script_sha256": sha256_file(evaluator_script),
            "device": args.device,
            "n_channel_level_cases": len(cases),
            "states": {
                "known": "CFD internal-interface oracle",
                "zero": (
                    "physical p=u=v=0 internal interfaces; "
                    "zero physics-optimization iterations"
                ),
                "physics": (
                    "truth-free physics-optimized unknown "
                    "internal-interface p/u/v"
                ),
            },
            "optimizer": {
                "outer_max_iter": args.outer_max_iter,
                "lbfgs_inner_max_iter": args.lbfgs_inner_max_iter,
                "lr": args.lr,
                "tol": args.tol,
                "stagnation_patience": args.stagnation_patience,
                "noise_seed_left": args.noise_seed_left,
                "noise_seed_right": args.noise_seed_right,
            },
            "aggregation": {
                "channel_equal_weight": True,
                "relative_l2": (
                    "For each field: mean of 10 subdomain "
                    "||pred-target||_2 / ||target||_2. "
                    "AvgRelL2 is then mean over p/u/v per channel."
                ),
                "balanced_rmse": (
                    "sqrt(mean of 10 subdomain MSE values)"
                ),
                "balanced_mae": (
                    "mean of 10 subdomain MAE values"
                ),
                "testset_summary": (
                    "mean/std across channel-level cases"
                ),
            },
            "cases": cases,
        }
        config_path.write_text(
            json.dumps(config_record, indent=2),
            encoding="utf-8",
        )

    all_metric_rows: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        realization_id = int(case["realization_id"])
        case_tag = safe_case_tag(case_id)
        base_name = f"{case_tag}_real{realization_id}"

        print()
        print("#" * 100)
        print(
            f"CASE {case_index}/{len(cases)}: "
            f"{case_id}, realization={realization_id}"
        )
        print("#" * 100)

        optimization_run_name = base_name
        optimization_dir = (
            optimization_root / optimization_run_name
        )

        if optimization_dir.exists():
            if not args.resume:
                raise FileExistsError(optimization_dir)
            if not optimization_complete(optimization_dir):
                raise RuntimeError(
                    "Existing optimization directory is incomplete; "
                    "refusing to overwrite or skip:\n"
                    f"{optimization_dir}"
                )
            print(
                f"[RESUME] complete optimization exists: "
                f"{optimization_dir}"
            )
        else:
            command = optimizer_command(
                python_executable=args.python,
                optimizer_script=optimizer_script,
                checkpoint=checkpoint,
                dataset=dataset,
                device=args.device,
                case_id=case_id,
                realization_id=realization_id,
                results_root=optimization_root,
                run_name=optimization_run_name,
                outer_max_iter=args.outer_max_iter,
                lbfgs_inner_max_iter=args.lbfgs_inner_max_iter,
                lr=args.lr,
                tol=args.tol,
                stagnation_patience=args.stagnation_patience,
                noise_seed_left=args.noise_seed_left,
                noise_seed_right=args.noise_seed_right,
            )
            run_logged_command(
                command=command,
                log_path=logs_root
                / f"{base_name}_optimization.log",
                cwd=PROJECT_ROOT,
            )

        if not optimization_complete(optimization_dir):
            raise RuntimeError(
                f"Optimization artifacts are incomplete: "
                f"{optimization_dir}"
            )

        verify_zero_initialization(optimization_dir)

        for interface_state, state_label in INTERFACE_STATES:
            evaluation_run_name = (
                f"{base_name}_{state_label}_interface"
            )
            evaluation_dir = (
                evaluation_root / evaluation_run_name
            )

            if evaluation_dir.exists():
                if not args.resume:
                    raise FileExistsError(evaluation_dir)
                if not evaluation_complete(evaluation_dir):
                    raise RuntimeError(
                        "Existing evaluation directory is incomplete; "
                        "refusing to overwrite or skip:\n"
                        f"{evaluation_dir}"
                    )
                print(
                    f"[RESUME] complete evaluation exists: "
                    f"{evaluation_dir}"
                )
            else:
                command = evaluator_command(
                    python_executable=args.python,
                    evaluator_script=evaluator_script,
                    optimization_run=optimization_dir,
                    interface_state=interface_state,
                    device=args.device,
                    results_root=evaluation_root,
                    run_name=evaluation_run_name,
                )
                run_logged_command(
                    command=command,
                    log_path=logs_root
                    / f"{base_name}_{state_label}_evaluation.log",
                    cwd=PROJECT_ROOT,
                )

            if not evaluation_complete(evaluation_dir):
                raise RuntimeError(
                    f"Evaluation artifacts are incomplete: "
                    f"{evaluation_dir}"
                )

            metric_rows = compute_channel_metrics(
                predictions_path=(
                    evaluation_dir / "predictions.h5"
                ),
                testset_name=args.testset_name,
                case_id=case_id,
                realization_id=realization_id,
                state_label=state_label,
            )
            all_metric_rows.extend(metric_rows)

        # Update aggregate files after every complete channel so that a long
        # test-set run leaves useful, inspectable progress.
        metrics_path = batch_root / "metrics_per_channel.csv"
        write_csv(metrics_path, all_metric_rows)

        summary_rows, summary_json = build_testset_summary(
            all_metric_rows
        )
        write_csv(
            batch_root / "metrics_summary.csv",
            summary_rows,
        )

        summary_payload = {
            "protocol": "distilled5_unknown_interface_testset_v1",
            "testset_name": args.testset_name,
            "completed_channel_level_cases": case_index,
            "planned_channel_level_cases": len(cases),
            **summary_json,
        }
        (
            batch_root / "summary.json"
        ).write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

    print()
    print("=" * 100)
    print("TEST-SET RUN COMPLETED")
    print("=" * 100)
    print(f"Channels completed     : {len(cases)}")
    print(
        f"Per-channel metrics    : "
        f"{batch_root / 'metrics_per_channel.csv'}"
    )
    print(
        f"Aggregate summary CSV  : "
        f"{batch_root / 'metrics_summary.csv'}"
    )
    print(
        f"Aggregate summary JSON : "
        f"{batch_root / 'summary.json'}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
