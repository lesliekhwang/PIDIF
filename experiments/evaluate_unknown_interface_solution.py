from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pidiffusion.data import normalize_diffusion_branch

from experiments.evaluate_diffusion_generation import (
    FIELD_UNITS,
    FieldAccumulator,
    SCHEDULE_FAMILY_PROGRESSIVE_NESTED20,
    build_model,
    build_sampling_timesteps,
    load_checkpoint,
    require_checkpoint_field,
    stable_sample_seed,
    timed_sample,
    warm_up,
)

from experiments.unknown_interface_diffusion_utils import (
    build_schedule,
    find_edge_rows,
    load_case_realization,
    load_target_normalizer,
    write_shared_z,
)


DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluate_unknown_interface_solution"
)

EXPECTED_FIELDS = [
    "pressure",
    "u",
    "v",
]

EXPECTED_NESTED5_TIMESTEPS = [
    999,
    789,
    578,
    368,
    158,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.generic):
        return value.item()

    return value


def load_truth_only(
    *,
    dataset_path: Path,
    sample_index: int,
) -> tuple[np.ndarray, dict]:
    """
    Load CFD target only during explicit post-hoc evaluation.

    Full-field generation must already be complete before this function
    is called.
    """
    with h5py.File(
        dataset_path,
        "r",
    ) as handle:
        group = handle["samples"][
            str(sample_index)
        ]

        truth = (
            group["target"][:]
            .astype(np.float32)
        )

        metadata_group = handle["metadata"]

        metadata = {
            key: decode_scalar(
                metadata_group[key][
                    sample_index
                ]
            )
            for key in metadata_group.keys()
        }

    if (
        truth.ndim != 2
        or truth.shape[1] != 3
    ):
        raise ValueError(
            f"Invalid target shape for sample "
            f"{sample_index}: {truth.shape}"
        )

    if not np.isfinite(
        truth
    ).all():
        raise ValueError(
            f"Non-finite CFD target for sample "
            f"{sample_index}"
        )

    return truth, metadata


def validate_interface_state(
    name: str,
    value: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> None:
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            f"Unexpected {name} shape: "
            f"{value.shape}; expected a 3-D "
            "(n_interfaces, n_points, 3) array"
        )

    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(
            f"Unexpected {name} shape: "
            f"{value.shape}; expected "
            f"{expected_shape}"
        )

    if not np.isfinite(
        value
    ).all():
        raise ValueError(
            f"{name} contains non-finite values"
        )


def interface_shape_from_optimization_config(
    optimization_config: dict,
) -> tuple[int, int, int]:
    interface_config = optimization_config.get(
        "interface"
    )

    if not isinstance(interface_config, dict):
        raise KeyError(
            "Optimization config is missing "
            "the interface section"
        )

    # Tolerate one historical accidental nested "interface" wrapper
    # while preferring the current flat schema.
    nested = interface_config.get("interface")
    if (
        isinstance(nested, dict)
        and "n_subdomains" not in interface_config
    ):
        interface_config = nested

    n_subdomains = int(
        interface_config["n_subdomains"]
    )
    n_internal_interfaces = int(
        interface_config.get(
            "n_internal_interfaces",
            n_subdomains - 1,
        )
    )
    n_interface_points = int(
        interface_config["points_per_interface"]
    )

    if n_subdomains < 2:
        raise ValueError(
            f"Invalid n_subdomains={n_subdomains}"
        )

    if n_internal_interfaces != n_subdomains - 1:
        raise ValueError(
            "Optimization interface metadata is inconsistent: "
            f"n_subdomains={n_subdomains}, "
            f"n_internal_interfaces={n_internal_interfaces}"
        )

    if n_interface_points <= 0:
        raise ValueError(
            "points_per_interface must be positive"
        )

    return (
        n_internal_interfaces,
        n_interface_points,
        3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimization-run",
        type=Path,
        required=True,
        help=(
            "Completed truth-free unknown-interface "
            "optimization run directory."
        ),
    )

    parser.add_argument(
        "--interface-state",
        choices=(
            "known",
            "initial",
            "final",
        ),
        default="final",
        help=(
            "Interface condition for full-field evaluation. "
            "'known' preserves CFD internal interface values from "
            "the dataset branch; 'initial' uses the truth-free "
            "physical-zero interface initialization; 'final' uses "
            "the physics-optimized interface state."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:1",
    )

    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help=(
            "Base seed for deterministic full-field sampling. "
            "This is independent of the fixed optimization "
            "left/right noise."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute post-hoc whole-field evaluation.",
    )

    args = parser.parse_args()

    optimization_run = (
        args.optimization_run
        .expanduser()
        .resolve()
    )

    optimization_config_path = (
        optimization_run
        / "config.json"
    )

    optimization_summary_path = (
        optimization_run
        / "summary.json"
    )

    interface_path = (
        optimization_run
        / "interface_states.npz"
    )

    for path in (
        optimization_config_path,
        optimization_summary_path,
        interface_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required optimization artifact "
                f"not found: {path}"
            )

    optimization_config = read_json(
        optimization_config_path
    )

    optimization_summary = read_json(
        optimization_summary_path
    )

    checkpoint_path = Path(
        optimization_config["checkpoint"]
    ).expanduser().resolve()

    dataset_path = Path(
        optimization_config["dataset"]
    ).expanduser().resolve()

    case_id = str(
        optimization_config["case_id"]
    )

    realization_id = int(
        optimization_config[
            "realization_id"
        ]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    expected_checkpoint_sha256 = (
        optimization_config.get(
            "checkpoint_sha256"
        )
    )

    actual_checkpoint_sha256 = (
        sha256_file(
            checkpoint_path
        )
    )

    if (
        expected_checkpoint_sha256
        is not None
        and actual_checkpoint_sha256
        != expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "Checkpoint SHA256 does not match "
            "the optimization artifact."
        )

    with np.load(
        interface_path
    ) as interface_data:
        required_keys = (
            "z_initial_normalized",
            "z_final_normalized",
            "field_names",
        )

        for key in required_keys:
            if key not in interface_data:
                raise KeyError(
                    f"interface_states.npz is "
                    f"missing {key}"
                )

        z_initial_normalized = (
            interface_data[
                "z_initial_normalized"
            ]
            .astype(np.float32)
            .copy()
        )

        z_final_normalized = (
            interface_data[
                "z_final_normalized"
            ]
            .astype(np.float32)
            .copy()
        )

        saved_field_names = [
            str(x)
            for x in interface_data[
                "field_names"
            ].tolist()
        ]

    expected_interface_shape = (
        interface_shape_from_optimization_config(
            optimization_config
        )
    )

    validate_interface_state(
        "z_initial_normalized",
        z_initial_normalized,
        expected_shape=expected_interface_shape,
    )

    validate_interface_state(
        "z_final_normalized",
        z_final_normalized,
        expected_shape=expected_interface_shape,
    )

    if saved_field_names != EXPECTED_FIELDS:
        raise RuntimeError(
            f"Unexpected interface fields: "
            f"{saved_field_names}"
        )

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, "
                "but CUDA is unavailable."
            )

        if (
            device.index is not None
            and device.index
            >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Requested {device}, but only "
                f"{torch.cuda.device_count()} "
                "CUDA devices are visible."
            )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    model = build_model(
        checkpoint,
        device,
    )

    normalizer = (
        load_target_normalizer(
            checkpoint
        )
        .to(device)
    )

    branch_channel_names = list(
        require_checkpoint_field(
            checkpoint,
            "branch_channel_names",
        )
    )

    output_channel_names = list(
        require_checkpoint_field(
            checkpoint,
            "output_channel_names",
        )
    )

    if output_channel_names != EXPECTED_FIELDS:
        raise RuntimeError(
            "Unexpected checkpoint output "
            f"channels: "
            f"{output_channel_names}"
        )

    fields = output_channel_names
    target_dim = len(fields)

    local_aspect_mean = float(
        checkpoint[
            "local_aspect_mean"
        ]
    )

    local_aspect_std = float(
        checkpoint[
            "local_aspect_std"
        ]
    )

    diffusion_config = dict(
        require_checkpoint_field(
            checkpoint,
            "diffusion_config",
        )
    )

    total_diffusion_steps = int(
        diffusion_config["T"]
    )

    schedule = build_schedule(
        checkpoint,
        device,
    )

    timesteps = (
        build_sampling_timesteps(
            sampling_steps=5,
            total_diffusion_steps=(
                total_diffusion_steps
            ),
            schedule_family=(
                SCHEDULE_FAMILY_PROGRESSIVE_NESTED20
            ),
            device=device,
        )
    )

    actual_timesteps = [
        int(x)
        for x in (
            timesteps.detach()
            .cpu()
            .tolist()
        )
    ]

    if (
        actual_timesteps
        != EXPECTED_NESTED5_TIMESTEPS
    ):
        raise RuntimeError(
            f"Unexpected Nested5 schedule: "
            f"{actual_timesteps}"
        )

    # ------------------------------------------------------------------
    # Stage 1:
    # Load geometry, branch conditions, and query coordinates only.
    #
    # load_case_realization() intentionally does not read CFD targets.
    # ------------------------------------------------------------------

    records = load_case_realization(
        dataset_path=dataset_path,
        case_id=case_id,
        realization_id=realization_id,
    )

    branches_normalized_np = (
        np.stack(
            [
                normalize_diffusion_branch(
                    branch=record[
                        "branch"
                    ],
                    branch_channel_names=(
                        branch_channel_names
                    ),
                    target_normalizer=(
                        normalizer
                    ),
                    local_aspect_mean=(
                        local_aspect_mean
                    ),
                    local_aspect_std=(
                        local_aspect_std
                    ),
                )
                for record in records
            ],
            axis=0,
        )
        .astype(np.float32)
    )

    left_rows, right_rows = (
        find_edge_rows(
            branches_normalized_np,
            branch_channel_names,
        )
    )

    n_subdomains = len(records)
    n_internal_interfaces = n_subdomains - 1
    n_interface_points = len(left_rows[0])

    actual_interface_shape = (
        n_internal_interfaces,
        n_interface_points,
        3,
    )

    if actual_interface_shape != expected_interface_shape:
        raise RuntimeError(
            "Dataset decomposition does not match the "
            "optimization interface metadata: "
            f"dataset expects {actual_interface_shape}, "
            f"optimization artifact expects "
            f"{expected_interface_shape}"
        )

    validate_interface_state(
        "z_initial_normalized",
        z_initial_normalized,
        expected_shape=actual_interface_shape,
    )

    validate_interface_state(
        "z_final_normalized",
        z_final_normalized,
        expected_shape=actual_interface_shape,
    )

    base_branch = (
        torch.from_numpy(
            branches_normalized_np
        )
        .to(device)
    )

    # ------------------------------------------------------------------
    # Select A/B/C interface condition.
    #
    # known:
    #   Preserve the CFD internal interface values already present in
    #   the original dataset branch. This is the oracle upper-bound
    #   known-interface baseline.
    #
    # initial:
    #   Overwrite all shared internal interfaces with the truth-free
    #   physical-zero initialization stored by the optimizer.
    #
    # final:
    #   Overwrite all shared internal interfaces with the final
    #   truth-free physics-optimized interface state.
    # ------------------------------------------------------------------

    if args.interface_state == "known":
        evaluation_branch = (
            base_branch.clone()
        )

        interface_description = (
            "CFD known internal interfaces "
            "(oracle baseline)"
        )

    elif args.interface_state == "initial":
        z_eval = (
            torch.from_numpy(
                z_initial_normalized
            )
            .to(device)
        )

        evaluation_branch = (
            write_shared_z(
                base_branch=base_branch,
                z=z_eval,
                left_rows=left_rows,
                right_rows=right_rows,
                names=branch_channel_names,
            )
        )

        interface_description = (
            "Truth-free physical-zero "
            "internal interfaces"
        )

    elif args.interface_state == "final":
        z_eval = (
            torch.from_numpy(
                z_final_normalized
            )
            .to(device)
        )

        evaluation_branch = (
            write_shared_z(
                base_branch=base_branch,
                z=z_eval,
                left_rows=left_rows,
                right_rows=right_rows,
                names=branch_channel_names,
            )
        )

        interface_description = (
            "Truth-free physics-optimized "
            "internal interfaces"
        )

    else:
        raise RuntimeError(
            f"Unsupported interface state: "
            f"{args.interface_state}"
        )

    if not torch.isfinite(
        evaluation_branch
    ).all():
        raise RuntimeError(
            "Non-finite evaluation branch."
        )

    if args.run_name is None:
        run_name = (
            f"{optimization_run.name}_"
            f"{args.interface_state}_interface"
        )
    else:
        run_name = args.run_name

    run_dir = (
        args.results_root
        / run_name
    )

    if (
        args.run
        and run_dir.exists()
    ):
        raise FileExistsError(
            f"Run directory already exists: "
            f"{run_dir}"
        )

    print()
    print("=" * 78)
    print(
        "Unknown-interface whole-field "
        "post-hoc evaluation"
    )
    print("=" * 78)
    print(
        f"Optimization run       : "
        f"{optimization_run}"
    )
    print(
        f"Checkpoint             : "
        f"{checkpoint_path}"
    )
    print(
        f"Dataset                : "
        f"{dataset_path}"
    )
    print(
        f"Case                   : "
        f"{case_id}"
    )
    print(
        f"Realization            : "
        f"{realization_id}"
    )
    print(
        f"Subdomains             : "
        f"{n_subdomains}"
    )
    print(
        f"Internal interfaces    : "
        f"{n_internal_interfaces}"
    )
    print(
        f"Points / interface     : "
        f"{n_interface_points}"
    )
    print(
        f"Device                 : "
        f"{device}"
    )
    print(
        f"Interface state        : "
        f"{args.interface_state}"
    )
    print(
        f"Interface definition   : "
        f"{interface_description}"
    )
    print(
        f"Nested5 timesteps      : "
        f"{actual_timesteps}"
    )
    print(
        f"Full-field sampling seed: "
        f"{args.sampling_seed}"
    )
    print(
        f"Optimization stop      : "
        f"{optimization_summary.get('stop_reason')}"
    )
    print(
        f"Optimized z shape      : "
        f"{z_final_normalized.shape}"
    )
    print(
        "CFD truth usage        : "
        "post-hoc metrics only"
    )

    if not args.run:
        print()
        print("DRY RUN ONLY")
        print(
            "Add --run to execute "
            "full-field generation and "
            "post-hoc CFD evaluation."
        )
        print("=" * 78)
        return

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    evaluation_config = {
        "protocol": (
            "unknown_interface_"
            "whole_field_ablation_v1"
        ),

        "optimization_run": str(
            optimization_run
        ),

        "optimization_config_sha256": (
            sha256_file(
                optimization_config_path
            )
        ),

        "optimization_summary_sha256": (
            sha256_file(
                optimization_summary_path
            )
        ),

        "interface_states_sha256": (
            sha256_file(
                interface_path
            )
        ),

        "checkpoint": str(
            checkpoint_path
        ),

        "checkpoint_sha256": (
            actual_checkpoint_sha256
        ),

        "dataset": str(
            dataset_path
        ),

        "case_id": case_id,

        "realization_id": (
            realization_id
        ),

        "interface": {
            "n_subdomains": int(
                n_subdomains
            ),
            "n_internal_interfaces": int(
                n_internal_interfaces
            ),
            "points_per_interface": int(
                n_interface_points
            ),
        },

        "device": str(device),

        "interface_state": (
            args.interface_state
        ),

        "interface_description": (
            interface_description
        ),

        "interface_state_definition": {
            "known": (
                "CFD internal interface values "
                "stored in the evaluation dataset "
                "branch; oracle known-interface "
                "baseline"
            ),
            "initial": (
                "Truth-free physical-zero "
                "interface initialization"
            ),
            "final": (
                "Truth-free physics-optimized "
                "interface state"
            ),
        },

        "sampling": {
            "schedule_family": (
                "progressive_nested20"
            ),

            "nfe": 5,

            "timesteps": (
                actual_timesteps
            ),

            "sampling_seed": int(
                args.sampling_seed
            ),

            "sample_seed_rule": (
                "stable_sample_seed("
                "base_seed, sample_index)"
            ),

            "optimization_noise_reused": (
                False
            ),
        },

        "truth_usage": {
            "used_in_interface_optimization": (
                False
            ),
            "used_in_full_field_generation": (
                args.interface_state
                == "known"
            ),
            "used_only_for_posthoc_metrics": (
                args.interface_state
                != "known"
            ),
        },
    }

    with (
        run_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            evaluation_config,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Stage 2:
    # Full-field generation.
    #
    # For initial/final interface states, CFD target is not read here.
    #
    # For known interface state, the branch intentionally contains CFD
    # internal interface values because this is the oracle baseline.
    # The full CFD field target is still not read until Stage 3.
    # ------------------------------------------------------------------

    first_query = (
        torch.from_numpy(
            records[0]["query"]
        )
        .to(device)
    )

    warm_up(
        model,
        evaluation_branch[
            0
        ].unsqueeze(0),
        first_query,
        target_dim,
        int(
            timesteps[0].item()
        ),
        device,
    )

    predictions: dict[
        int,
        np.ndarray,
    ] = {}

    sampling_seconds: dict[
        int,
        float,
    ] = {}

    sample_seeds: dict[
        int,
        int,
    ] = {}

    prediction_start = (
        time.perf_counter()
    )

    for position, record in enumerate(
        records,
        start=1,
    ):
        sample_index = int(
            record["sample_index"]
        )

        subdomain_id = int(
            record["subdomain_id"]
        )

        query_raw = (
            record["query"]
            .astype(np.float32)
        )

        query = (
            torch.from_numpy(
                query_raw
            )
            .to(device)
        )

        branch = (
            evaluation_branch[
                subdomain_id
            ]
            .unsqueeze(0)
        )

        sample_seed = (
            stable_sample_seed(
                args.sampling_seed,
                sample_index,
            )
        )

        generator = (
            torch.Generator(
                device=device
            )
        )

        generator.manual_seed(
            sample_seed
        )

        initial_noise = torch.randn(
            (
                len(query),
                target_dim,
            ),
            dtype=query.dtype,
            device=device,
            generator=generator,
        )

        pred_std, seconds = (
            timed_sample(
                model,
                branch,
                query,
                initial_noise,
                schedule,
                timesteps,
                device,
            )
        )

        pred = (
            normalizer.decode(
                pred_std
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        expected_shape = (
            len(query_raw),
            target_dim,
        )

        if pred.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected prediction shape "
                f"for sample {sample_index}: "
                f"{pred.shape}; expected "
                f"{expected_shape}"
            )

        if not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                f"Non-finite prediction "
                f"for sample "
                f"{sample_index}"
            )

        predictions[
            sample_index
        ] = pred

        sampling_seconds[
            sample_index
        ] = float(seconds)

        sample_seeds[
            sample_index
        ] = int(sample_seed)

        print(
            f"[Predict] "
            f"{position:02d}/{n_subdomains:02d} | "
            f"sample={sample_index} | "
            f"subdomain={subdomain_id} | "
            f"points={len(query_raw)} | "
            f"time={seconds:.4f}s",
            flush=True,
        )

    total_prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    print()
    print(
        "Full-field generation completed."
    )
    print(
        "Loading CFD full-field truth now "
        "for post-hoc evaluation."
    )

    # ------------------------------------------------------------------
    # Stage 3:
    # Full CFD field target is first accessed here.
    # ------------------------------------------------------------------

    accumulators = {
        field: FieldAccumulator()
        for field in fields
    }

    global_truth_parts = {
        field: []
        for field in fields
    }

    global_prediction_parts = {
        field: []
        for field in fields
    }

    per_subdomain_rows: list[
        dict
    ] = []

    predictions_path = (
        run_dir
        / "predictions.h5"
    )

    with h5py.File(
        predictions_path,
        "w",
    ) as output_handle:
        output_handle.attrs[
            "case_id"
        ] = case_id

        output_handle.attrs[
            "realization_id"
        ] = realization_id

        output_handle.attrs[
            "n_subdomains"
        ] = int(n_subdomains)

        output_handle.attrs[
            "n_internal_interfaces"
        ] = int(n_internal_interfaces)

        output_handle.attrs[
            "points_per_interface"
        ] = int(n_interface_points)

        output_handle.attrs[
            "interface_state"
        ] = args.interface_state

        output_handle.attrs[
            "interface_description"
        ] = interface_description

        output_handle.attrs[
            "sampling_seed"
        ] = int(
            args.sampling_seed
        )

        output_handle.attrs[
            "schedule_family"
        ] = (
            "progressive_nested20"
        )

        output_handle.attrs[
            "sampling_steps"
        ] = 5

        output_handle.attrs[
            "truth_usage"
        ] = (
            "posthoc_full_field_metrics"
        )

        output_handle.create_dataset(
            "evaluation_branch_normalized",
            data=(
                evaluation_branch
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        samples_group = (
            output_handle
            .create_group(
                "samples"
            )
        )

        for record in records:
            sample_index = int(
                record["sample_index"]
            )

            subdomain_id = int(
                record["subdomain_id"]
            )

            query_raw = (
                record["query"]
                .astype(np.float32)
            )

            prediction = (
                predictions[
                    sample_index
                ]
            )

            truth, metadata = (
                load_truth_only(
                    dataset_path=(
                        dataset_path
                    ),
                    sample_index=(
                        sample_index
                    ),
                )
            )

            if (
                truth.shape
                != prediction.shape
            ):
                raise RuntimeError(
                    f"Truth/prediction shape "
                    f"mismatch for sample "
                    f"{sample_index}: "
                    f"{truth.shape} vs "
                    f"{prediction.shape}"
                )

            sample_group = (
                samples_group
                .create_group(
                    str(
                        sample_index
                    )
                )
            )

            sample_group.attrs[
                "sample_index"
            ] = sample_index

            sample_group.attrs[
                "subdomain_id"
            ] = subdomain_id

            sample_group.attrs[
                "case_id"
            ] = str(
                metadata.get(
                    "case_id",
                    case_id,
                )
            )

            sample_group.attrs[
                "realization_id"
            ] = int(
                metadata.get(
                    "realization_id",
                    realization_id,
                )
            )

            sample_group.attrs[
                "sampling_seed"
            ] = sample_seeds[
                sample_index
            ]

            sample_group.attrs[
                "sampling_seconds"
            ] = sampling_seconds[
                sample_index
            ]

            sample_group.create_dataset(
                "query",
                data=query_raw,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            sample_group.create_dataset(
                "prediction",
                data=prediction,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            sample_group.create_dataset(
                "target",
                data=truth,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            sample_group.create_dataset(
                "absolute_error",
                data=(
                    np.abs(
                        prediction
                        - truth
                    )
                    .astype(np.float32)
                ),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            for (
                field_index,
                field,
            ) in enumerate(
                fields
            ):
                global_truth_parts[
                    field
                ].append(
                    truth[
                        :,
                        field_index,
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                )

                global_prediction_parts[
                    field
                ].append(
                    prediction[
                        :,
                        field_index,
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                )

                metrics = (
                    accumulators[
                        field
                    ].add(
                        truth[
                            :,
                            field_index,
                        ],
                        prediction[
                            :,
                            field_index,
                        ],
                    )
                )

                per_subdomain_rows.append(
                    {
                        "sample_index": (
                            sample_index
                        ),
                        "case_id": (
                            case_id
                        ),
                        "realization_id": (
                            realization_id
                        ),
                        "subdomain_id": (
                            subdomain_id
                        ),
                        "interface_state": (
                            args.interface_state
                        ),
                        "field": (
                            field
                        ),
                        "unit": (
                            FIELD_UNITS[
                                field
                            ]
                        ),
                        "n_points": (
                            len(
                                query_raw
                            )
                        ),
                        "rmse": (
                            metrics[
                                "rmse"
                            ]
                        ),
                        "mae": (
                            metrics[
                                "mae"
                            ]
                        ),
                        "r2": (
                            metrics[
                                "r2"
                            ]
                        ),
                        "correlation": (
                            metrics[
                                "correlation"
                            ]
                        ),
                        "sampling_seconds": (
                            sampling_seconds[
                                sample_index
                            ]
                        ),
                        "sample_seed": (
                            sample_seeds[
                                sample_index
                            ]
                        ),
                    }
                )

    per_subdomain_path = (
        run_dir
        / "metrics_per_subdomain.csv"
    )

    write_csv(
        per_subdomain_path,
        per_subdomain_rows,
    )

    summary_rows: list[
        dict
    ] = []

    sampling_total_seconds = float(
        sum(
            sampling_seconds.values()
        )
    )

    sampling_mean_seconds = float(
        np.mean(
            list(
                sampling_seconds.values()
            )
        )
    )

    sampling_median_seconds = float(
        np.median(
            list(
                sampling_seconds.values()
            )
        )
    )

    total_points = int(
        sum(
            len(
                record["query"]
            )
            for record in records
        )
    )

    for field in fields:
        metric_summary = (
            accumulators[
                field
            ].summary()
        )

        truth_all = np.concatenate(
            global_truth_parts[field]
        )
        prediction_all = np.concatenate(
            global_prediction_parts[field]
        )

        global_error = (
            prediction_all - truth_all
        )

        global_relative_l2 = float(
            np.linalg.norm(global_error)
            / max(
                float(np.linalg.norm(truth_all)),
                1.0e-12,
            )
        )

        summary_rows.append(
            {
                "case_id": (
                    case_id
                ),
                "realization_id": (
                    realization_id
                ),
                "interface_state": (
                    args.interface_state
                ),
                "field": (
                    field
                ),
                "unit": (
                    FIELD_UNITS[
                        field
                    ]
                ),
                **metric_summary,
                "global_relative_l2": (
                    global_relative_l2
                ),
                "sampling_steps": 5,
                "sampling_seed": (
                    args.sampling_seed
                ),
                "sampling_total_seconds": (
                    sampling_total_seconds
                ),
                "sampling_mean_seconds_per_subdomain": (
                    sampling_mean_seconds
                ),
                "sampling_median_seconds_per_subdomain": (
                    sampling_median_seconds
                ),
            }
        )

    summary_csv_path = (
        run_dir
        / "metrics_summary.csv"
    )

    write_csv(
        summary_csv_path,
        summary_rows,
    )

    summary_json = {
        "protocol": (
            "unknown_interface_"
            "whole_field_ablation_v1"
        ),

        "case_id": (
            case_id
        ),

        "realization_id": (
            realization_id
        ),

        "n_subdomains": int(
            n_subdomains
        ),

        "n_internal_interfaces": int(
            n_internal_interfaces
        ),

        "points_per_interface": int(
            n_interface_points
        ),

        "interface_state": (
            args.interface_state
        ),

        "interface_description": (
            interface_description
        ),

        "optimization": {
            "run": str(
                optimization_run
            ),

            "stop_reason": (
                optimization_summary.get(
                    "stop_reason"
                )
            ),

            "converged": (
                optimization_summary.get(
                    "converged"
                )
            ),

            "outer_iterations": (
                optimization_summary.get(
                    "outer_iterations"
                )
            ),

            "closure_calls": (
                optimization_summary.get(
                    "closure_calls"
                )
            ),
        },

        "sampling": {
            "schedule_family": (
                "progressive_nested20"
            ),

            "nfe": 5,

            "timesteps": (
                actual_timesteps
            ),

            "base_seed": int(
                args.sampling_seed
            ),

            "total_points": (
                total_points
            ),

            "total_sampling_seconds": (
                sampling_total_seconds
            ),

            "mean_seconds_per_subdomain": (
                sampling_mean_seconds
            ),

            "median_seconds_per_subdomain": (
                sampling_median_seconds
            ),

            "end_to_end_prediction_seconds": (
                total_prediction_seconds
            ),
        },

        "metrics": {
            row["field"]: {
                "unit": (
                    row["unit"]
                ),

                "subdomain_balanced_rmse": (
                    row[
                        "subdomain_balanced_rmse"
                    ]
                ),

                "subdomain_balanced_mae": (
                    row[
                        "subdomain_balanced_mae"
                    ]
                ),

                "global_rmse": (
                    row[
                        "global_rmse"
                    ]
                ),

                "global_mae": (
                    row[
                        "global_mae"
                    ]
                ),

                "global_r2": (
                    row[
                        "global_r2"
                    ]
                ),

                "global_correlation": (
                    row[
                        "global_correlation"
                    ]
                ),

                "global_relative_l2": (
                    row[
                        "global_relative_l2"
                    ]
                ),
            }
            for row in summary_rows
        },

        "truth_usage": {
            "used_in_interface_optimization": (
                False
            ),

            "known_interface_oracle": (
                args.interface_state
                == "known"
            ),

            "full_field_target_used_only_posthoc": (
                True
            ),
        },

        "artifacts": {
            "config": (
                "config.json"
            ),

            "per_subdomain_metrics": (
                "metrics_per_subdomain.csv"
            ),

            "summary_metrics": (
                "metrics_summary.csv"
            ),

            "predictions": (
                "predictions.h5"
            ),
        },
    }

    with (
        run_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary_json,
            handle,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        "Whole-channel post-hoc accuracy"
    )
    print("=" * 78)

    print(
        f"Interface state        : "
        f"{args.interface_state}"
    )

    for row in summary_rows:
        print(
            f"{row['field']:<10s} | "
            f"balanced RMSE="
            f"{row['subdomain_balanced_rmse']:.8g} "
            f"{row['unit']} | "
            f"balanced MAE="
            f"{row['subdomain_balanced_mae']:.8g} "
            f"{row['unit']} | "
            f"global R²="
            f"{row['global_r2']:.6f} | "
            f"corr="
            f"{row['global_correlation']:.6f} | "
            f"global Rel-L2="
            f"{100.0 * row['global_relative_l2']:.4f}%"
        )

    print()
    print(
        f"Sampling total         : "
        f"{sampling_total_seconds:.4f} s"
    )

    print(
        f"Mean / subdomain       : "
        f"{sampling_mean_seconds:.6f} s"
    )

    print(
        f"Saved run              : "
        f"{run_dir}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()