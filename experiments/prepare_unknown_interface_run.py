"""Prepare a provenance manifest; this is not an inference runner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pidiffusion.artifacts import (  # noqa: E402
    build_run_id,
    create_run_directory,
    write_manifest,
)
from pidiffusion.provenance import (  # noqa: E402
    file_identity,
    git_state,
    runtime_environment,
)


def _resolve_repo_path(raw_path: str) -> Path:
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


def _identity_for_manifest(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    identity["path"] = _manifest_path(path)
    return identity


def _build_manifest(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp_utc: datetime,
    dataset_path: Path,
    checkpoint_path: Path,
    source_notebook_path: Path,
    planned_directory: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "timestamp_utc": timestamp_utc.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "prepared",
        "git": git_state(REPOSITORY_ROOT),
        "source_files": [
            {
                "role": "unknown_interface_notebook",
                **_identity_for_manifest(source_notebook_path),
            }
        ],
        "dataset": {
            **_identity_for_manifest(dataset_path),
            "split": "test",
            "case_ids": [args.case_id],
            "normalizer_source": "stage3_checkpoint",
            "target_mean": None,
            "target_std": None,
            "local_aspect_mean": None,
            "local_aspect_std": None,
        },
        "checkpoint": {
            **_identity_for_manifest(checkpoint_path),
            "requested_stage": 3,
            "format_policy": "progressive_distillation_stage_v1_only",
            "checkpoint_tag": args.checkpoint_tag,
            "epoch": None,
        },
        "protocol": {
            "name": "consistency_physics_optimization",
            "method_name": "consistency_physics_optimization",
            "method_selection_basis": "fixed after development-stage CFD comparison",
            "model_role": "frozen_distilled_student",
            "trainable_variables": "internal_interface_pressure_u_v",
            "internal_interface_truth_used_as_model_input": False,
            "runtime_uses_cfd_truth": False,
            "posthoc_evaluation_enabled": False,
            "posthoc_evaluation_uses_cfd_truth": False,
            "metric_space": "physical",
            "metric_aggregation": "single_case_geometry_masked",
            "metric_mask": "inside_geometry",
        },
        "randomness": {
            "global_seed": args.global_seed,
            "inference_noise_seed": args.inference_noise_seed,
            "boundary_noise_seed": args.boundary_noise_seed,
            "ddim_steps": args.ddim_steps,
        },
        "environment": runtime_environment(),
        "outputs": {
            "directory": str(planned_directory),
            "files": [],
        },
        "notes": [
            "Prepared only; no model, checkpoint tensor, HDF5 contents, "
            "inference, sampling, optimization, or evaluation was executed."
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a protected PIDiffusion run directory and manifest. "
            "This command never runs inference."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path")
    parser.add_argument(
        "--source-notebook",
        required=True,
        help="Source notebook path used for this prepared run",
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--checkpoint-tag", required=True)
    parser.add_argument("--global-seed", required=True, type=int)
    parser.add_argument("--inference-noise-seed", required=True, type=int)
    parser.add_argument("--boundary-noise-seed", required=True, type=int)
    parser.add_argument("--ddim-steps", required=True, type=int)
    parser.add_argument(
        "--results-root",
        default=str(REPOSITORY_ROOT / "results"),
        help="Results root; relative paths are resolved from the repository root",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the run directory and prepared manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_path = _resolve_repo_path(args.dataset)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    source_notebook_path = _resolve_repo_path(args.source_notebook)
    for path, role in (
        (dataset_path, "dataset"),
        (checkpoint_path, "checkpoint"),
        (source_notebook_path, "source notebook"),
    ):
        _require_file(path, role)

    timestamp_utc = datetime.now(timezone.utc)
    run_id = build_run_id(
        protocol="unknown_interface",
        case_id=args.case_id,
        checkpoint_tag=args.checkpoint_tag,
        seed=args.global_seed,
        ddim_steps=args.ddim_steps,
        timestamp_utc=timestamp_utc,
    )
    results_root = _resolve_repo_path(args.results_root)
    planned_directory = results_root / "unknown_interface" / args.case_id / run_id
    manifest = _build_manifest(
        args=args,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        source_notebook_path=source_notebook_path,
        planned_directory=planned_directory,
    )

    print("mode:", "create" if args.create else "dry-run")
    print("repository_root:", REPOSITORY_ROOT)
    print("run_id:", run_id)
    print("planned_run_directory:", planned_directory)
    print("manifest_preview:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    if args.create:
        run_directory = create_run_directory(
            results_root,
            protocol="unknown_interface",
            case_id=args.case_id,
            run_id=run_id,
        )
        manifest["outputs"]["directory"] = str(run_directory)
        manifest_path = write_manifest(run_directory, manifest)
        print("created_run_directory:", run_directory)
        print("created_manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
