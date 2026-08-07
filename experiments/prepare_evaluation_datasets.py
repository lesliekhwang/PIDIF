"""Prepare deterministic validation/test datasets for DeepONet and diffusion.

This script creates four shared evaluation HDF5 files from the fixed case split:

- validation / control-point decomposition
- validation / AR1 equal-width decomposition
- test / control-point decomposition
- test / AR1 equal-width decomposition

The evaluation datasets use one deterministic decomposition per full channel.
They preserve the same branch/query/target schema as the finalized randomized
training dataset.

By default this script only validates the plan. Pass ``--run`` to generate
the HDF5 files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pidiffusion.diffusion_fluent_evaluation_dataset import (
    build_ar1_channel_dataset,
    build_control_point_channel_dataset,
)
from pidiffusion.provenance import sha256_file


DATASET_NAME = "channel_water"

SPLIT_PATH = (
    REPO_ROOT
    / "splits"
    / "channel_case_split_160train_20val_20test.json"
)

DESIGN_ROOT = (
    REPO_ROOT
    / "2d_geometry_specs"
    / "channel_water"
)

RUNS_ROOT = (
    REPO_ROOT
    / "runs_2d"
    / "channel_water"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
)

EVALUATION_BUILDER_PATH = (
    REPO_ROOT
    / "pidiffusion"
    / "diffusion_fluent_evaluation_dataset.py"
)

TRAINING_BUILDER_PATH = (
    REPO_ROOT
    / "pidiffusion"
    / "diffusion_fluent_dataset.py"
)

GENERATOR_PATH = Path(__file__).resolve()

N_INTERFACE_POINTS = 256
N_BOUNDARY_POINTS = 512

FIELD_MAP = {
    "pressure": "SV_P",
    "u": "SV_U",
    "v": "SV_V",
}

OUTPUT_FIELDS = (
    "pressure",
    "u",
    "v",
)

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

TRUNK_CHANNEL_NAMES = (
    "x_local",
    "y_local",
)

BC_KWARGS = {
    "inlet_v": 0.0,
    "outlet_pressure": 0.0,
    "wall_u": 0.0,
    "wall_v": 0.0,
}

EXPECTED_CASES_PER_SPLIT = 20
EXPECTED_SUBDOMAINS_PER_CASE = 10
EXPECTED_SAMPLES_PER_DATASET = 200

COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the fixed 160/20/20 case split."""

    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    required = {
        "train_cases",
        "validation_cases",
        "test_cases",
    }

    missing = required.difference(split)
    if missing:
        raise ValueError(
            f"Split manifest is missing keys: {sorted(missing)}"
        )

    train_cases = tuple(str(x) for x in split["train_cases"])
    validation_cases = tuple(
        str(x) for x in split["validation_cases"]
    )
    test_cases = tuple(str(x) for x in split["test_cases"])

    if len(train_cases) != 160:
        raise ValueError(
            f"Expected 160 training cases, got {len(train_cases)}"
        )

    if len(validation_cases) != EXPECTED_CASES_PER_SPLIT:
        raise ValueError(
            "Expected 20 validation cases, "
            f"got {len(validation_cases)}"
        )

    if len(test_cases) != EXPECTED_CASES_PER_SPLIT:
        raise ValueError(
            f"Expected 20 test cases, got {len(test_cases)}"
        )

    train_set = set(train_cases)
    validation_set = set(validation_cases)
    test_set = set(test_cases)

    if train_set & validation_set:
        raise ValueError("Train/validation case leakage")
    if train_set & test_set:
        raise ValueError("Train/test case leakage")
    if validation_set & test_set:
        raise ValueError("Validation/test case leakage")

    if len(train_set | validation_set | test_set) != 200:
        raise ValueError(
            "The fixed split must contain exactly 200 unique cases"
        )

    return split


def case_paths(case_id: str) -> dict[str, Path]:
    """Return raw design/mesh/solution paths for one channel."""

    return {
        "design": DESIGN_ROOT / f"{case_id}.json",
        "mesh": (
            RUNS_ROOT
            / case_id
            / f"{case_id}.msh.h5"
        ),
        "dat": (
            RUNS_ROOT
            / case_id
            / "case2d.dat.h5"
        ),
    }


def validate_raw_sources(case_ids: tuple[str, ...]) -> None:
    """Require every raw source file needed by the requested cases."""

    missing: list[Path] = []

    for case_id in case_ids:
        for path in case_paths(case_id).values():
            if not path.is_file():
                missing.append(path)

    if missing:
        preview = "\n".join(
            f"  {path}"
            for path in missing[:20]
        )
        raise FileNotFoundError(
            "Missing raw evaluation source files:\n"
            f"{preview}"
        )


def expected_subdomain_count(
    *,
    case_id: str,
    mode: str,
) -> int:
    """Compute the deterministic subdomain count from the design JSON."""

    design_path = case_paths(case_id)["design"]

    with design_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    metadata = config["metadata"]

    x_points = np.asarray(
        metadata["x_points_mm"],
        dtype=np.float64,
    ).reshape(-1)

    if x_points.size < 2:
        raise ValueError(
            f"{case_id}: x_points_mm must contain at least two values"
        )

    if not np.all(np.diff(x_points) > 0.0):
        raise ValueError(
            f"{case_id}: x_points_mm must be strictly increasing"
        )

    if mode == "control_points":
        return int(x_points.size - 1)

    if mode == "ar1":
        reference_length = float(metadata["L_mm"])
        if reference_length <= 0.0:
            raise ValueError(
                f"{case_id}: metadata.L_mm must be positive"
            )

        span = float(x_points[-1] - x_points[0])

        return max(
            1,
            int(round(span / reference_length)),
        )

    raise ValueError(f"Unknown evaluation mode: {mode}")


def validate_expected_counts(
    *,
    split_name: str,
    case_ids: tuple[str, ...],
    mode: str,
) -> int:
    """Validate current fixed protocol and return total sample count."""

    counts = [
        expected_subdomain_count(
            case_id=case_id,
            mode=mode,
        )
        for case_id in case_ids
    ]

    if len(counts) != EXPECTED_CASES_PER_SPLIT:
        raise ValueError(
            f"{split_name}/{mode}: expected 20 cases, got {len(counts)}"
        )

    bad = {
        case_id: count
        for case_id, count in zip(case_ids, counts)
        if count != EXPECTED_SUBDOMAINS_PER_CASE
    }

    if bad:
        raise ValueError(
            f"{split_name}/{mode}: current protocol expects exactly "
            f"{EXPECTED_SUBDOMAINS_PER_CASE} subdomains per channel; "
            f"mismatches={bad}"
        )

    total = int(sum(counts))

    if total != EXPECTED_SAMPLES_PER_DATASET:
        raise ValueError(
            f"{split_name}/{mode}: expected "
            f"{EXPECTED_SAMPLES_PER_DATASET} samples, got {total}"
        )

    return total


def evaluation_specs(
    split: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the four fixed evaluation dataset specifications."""

    validation_cases = tuple(
        str(x) for x in split["validation_cases"]
    )
    test_cases = tuple(
        str(x) for x in split["test_cases"]
    )

    return [
        {
            "split_name": "validation",
            "mode": "control_points",
            "case_ids": validation_cases,
            "builder": build_control_point_channel_dataset,
            "output_path": (
                OUTPUT_ROOT
                / "channel_deeponet_style_pressure_u_v_controlpoints_val.h5"
            ),
        },
        {
            "split_name": "validation",
            "mode": "ar1",
            "case_ids": validation_cases,
            "builder": build_ar1_channel_dataset,
            "output_path": (
                OUTPUT_ROOT
                / "channel_deeponet_style_pressure_u_v_ar1_val.h5"
            ),
        },
        {
            "split_name": "test",
            "mode": "control_points",
            "case_ids": test_cases,
            "builder": build_control_point_channel_dataset,
            "output_path": (
                OUTPUT_ROOT
                / "channel_deeponet_style_pressure_u_v_controlpoints_test.h5"
            ),
        },
        {
            "split_name": "test",
            "mode": "ar1",
            "case_ids": test_cases,
            "builder": build_ar1_channel_dataset,
            "output_path": (
                OUTPUT_ROOT
                / "channel_deeponet_style_pressure_u_v_ar1_test.h5"
            ),
        },
    ]


def _metadata_dtype(
    values: list[Any],
) -> tuple[np.dtype | h5py.Datatype, np.ndarray]:
    """Convert one metadata column to a stable HDF5-compatible array."""

    first = values[0]

    if isinstance(first, (str, bytes)):
        dtype = h5py.string_dtype(encoding="utf-8")
        array = np.asarray(
            [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in values
            ],
            dtype=object,
        )
        return dtype, array

    if isinstance(first, (bool, np.bool_)):
        array = np.asarray(values, dtype=np.bool_)
        return array.dtype, array

    if isinstance(first, (int, np.integer)):
        array = np.asarray(values, dtype=np.int64)
        return array.dtype, array

    if isinstance(first, (float, np.floating)):
        array = np.asarray(values, dtype=np.float64)
        return array.dtype, array

    raise TypeError(
        f"Unsupported metadata type: {type(first).__name__}"
    )


def write_metadata_group(
    handle: h5py.File,
    metadata_rows: list[dict[str, Any]],
) -> None:
    """Write column-oriented metadata compatible with the existing loader."""

    if not metadata_rows:
        raise ValueError("No metadata rows were produced")

    keys = tuple(metadata_rows[0].keys())

    for row_index, row in enumerate(metadata_rows):
        if tuple(row.keys()) != keys:
            raise ValueError(
                f"Metadata key order mismatch at row {row_index}"
            )

    group = handle.create_group("metadata")

    for key in keys:
        values = [row[key] for row in metadata_rows]
        dtype, array = _metadata_dtype(values)

        group.create_dataset(
            key,
            data=array,
            dtype=dtype,
        )


def initialize_hdf5_attributes(
    *,
    handle: h5py.File,
    split_name: str,
    mode: str,
    case_ids: tuple[str, ...],
    expected_samples: int,
) -> None:
    """Record protocol and provenance before any samples are written."""

    handle.attrs["schema_version"] = "deeponet_style_evaluation_v1"
    handle.attrs["dataset_name"] = DATASET_NAME
    handle.attrs["dataset_role"] = "shared_deterministic_evaluation"
    handle.attrs["split_name"] = split_name
    handle.attrs["evaluation_decomposition"] = mode
    handle.attrs["interface_placement"] = mode

    handle.attrs["n_cases"] = len(case_ids)
    handle.attrs["n_samples"] = expected_samples
    handle.attrs["n_realizations"] = 1
    handle.attrs["n_subdomains"] = EXPECTED_SUBDOMAINS_PER_CASE
    handle.attrs["expected_samples_per_case"] = (
        EXPECTED_SUBDOMAINS_PER_CASE
    )
    handle.attrs["expected_total_samples"] = expected_samples

    handle.attrs["n_interface_points"] = N_INTERFACE_POINTS
    handle.attrs["n_boundary_points"] = N_BOUNDARY_POINTS

    handle.attrs["branch_channel_names"] = "\n".join(
        BRANCH_CHANNEL_NAMES
    )
    handle.attrs["trunk_channel_names"] = "\n".join(
        TRUNK_CHANNEL_NAMES
    )
    handle.attrs["output_channel_names"] = "\n".join(
        OUTPUT_FIELDS
    )

    handle.attrs["split_manifest_path"] = str(
        SPLIT_PATH.resolve()
    )
    handle.attrs["split_manifest_sha256"] = sha256_file(
        SPLIT_PATH
    )

    handle.attrs["dataset_builder_path"] = str(
        EVALUATION_BUILDER_PATH.resolve()
    )
    handle.attrs["dataset_builder_sha256"] = sha256_file(
        EVALUATION_BUILDER_PATH
    )

    handle.attrs["dependency_training_builder_path"] = str(
        TRAINING_BUILDER_PATH.resolve()
    )
    handle.attrs["dependency_training_builder_sha256"] = sha256_file(
        TRAINING_BUILDER_PATH
    )

    handle.attrs["generator_script_path"] = str(
        GENERATOR_PATH
    )
    handle.attrs["generator_script_sha256"] = sha256_file(
        GENERATOR_PATH
    )

    handle.attrs["compression"] = COMPRESSION
    handle.attrs["compression_level"] = COMPRESSION_LEVEL


def validate_case_dataset(
    *,
    case_id: str,
    mode: str,
    dataset: Mapping[str, Any],
) -> None:
    """Validate one 10-subdomain deterministic builder result."""

    samples = dataset["samples"]
    metadata = dataset["metadata"]

    if len(samples) != EXPECTED_SUBDOMAINS_PER_CASE:
        raise ValueError(
            f"{case_id}/{mode}: expected 10 samples, "
            f"got {len(samples)}"
        )

    if len(metadata) != EXPECTED_SUBDOMAINS_PER_CASE:
        raise ValueError(
            f"{case_id}/{mode}: expected 10 metadata rows, "
            f"got {len(metadata)}"
        )

    if tuple(dataset["branch_channel_names"]) != BRANCH_CHANNEL_NAMES:
        raise ValueError(
            f"{case_id}/{mode}: unexpected branch schema"
        )

    if tuple(dataset["trunk_channel_names"]) != TRUNK_CHANNEL_NAMES:
        raise ValueError(
            f"{case_id}/{mode}: unexpected trunk schema"
        )

    if tuple(dataset["output_channel_names"]) != OUTPUT_FIELDS:
        raise ValueError(
            f"{case_id}/{mode}: unexpected output schema"
        )

    if str(dataset["interface_placement"]) != mode:
        raise ValueError(
            f"{case_id}/{mode}: builder reported "
            f"{dataset['interface_placement']!r}"
        )

    edges = np.asarray(dataset["x_edges"], dtype=np.float64)

    if edges.shape != (EXPECTED_SUBDOMAINS_PER_CASE + 1,):
        raise ValueError(
            f"{case_id}/{mode}: invalid edge count {edges.shape}"
        )

    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(
            f"{case_id}/{mode}: interfaces are not strictly increasing"
        )

    expected_subdomains = list(
        range(EXPECTED_SUBDOMAINS_PER_CASE)
    )

    observed_subdomains = [
        int(row["subdomain_id"])
        for row in metadata
    ]

    if observed_subdomains != expected_subdomains:
        raise ValueError(
            f"{case_id}/{mode}: subdomain IDs are not 0 through 9"
        )

    if {
        int(row["realization_id"])
        for row in metadata
    } != {0}:
        raise ValueError(
            f"{case_id}/{mode}: realization_id must be 0"
        )

    for sample_index, sample in enumerate(samples):
        branch = np.asarray(sample["branch"])
        query = np.asarray(sample["query"])
        target = np.asarray(sample["target"])

        if branch.shape != (
            2 * N_INTERFACE_POINTS
            + 2 * N_BOUNDARY_POINTS,
            len(BRANCH_CHANNEL_NAMES),
        ):
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                f"invalid branch shape {branch.shape}"
            )

        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                f"invalid query shape {query.shape}"
            )

        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                f"invalid target shape {target.shape}"
            )

        if query.shape[0] == 0:
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "empty query set"
            )

        if query.shape[0] != target.shape[0]:
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "query/target row mismatch"
            )

        if query.shape[0] != int(
            metadata[sample_index]["n_cells"]
        ):
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "metadata n_cells mismatch"
            )

        if not np.isfinite(branch).all():
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "branch contains non-finite values"
            )

        if not np.isfinite(query).all():
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "query contains non-finite values"
            )

        if not np.isfinite(target).all():
            raise ValueError(
                f"{case_id}/{mode}/sample {sample_index}: "
                "target contains non-finite values"
            )


def build_case(
    *,
    builder: Callable[..., dict[str, Any]],
    case_id: str,
) -> dict[str, Any]:
    """Run one deterministic evaluation builder."""

    paths = case_paths(case_id)

    return builder(
        case_id=case_id,
        design_path=paths["design"],
        mesh_path=paths["mesh"],
        dat_path=paths["dat"],
        n_interface_points=N_INTERFACE_POINTS,
        n_boundary_points=N_BOUNDARY_POINTS,
        inlet_v=BC_KWARGS["inlet_v"],
        outlet_pressure=BC_KWARGS["outlet_pressure"],
        wall_u=BC_KWARGS["wall_u"],
        wall_v=BC_KWARGS["wall_v"],
        field_map=FIELD_MAP,
    )


def write_one_dataset(
    *,
    split_name: str,
    mode: str,
    case_ids: tuple[str, ...],
    builder: Callable[..., dict[str, Any]],
    temporary_path: Path,
) -> None:
    """Stream one 20-channel evaluation dataset to a temporary HDF5."""

    expected_samples = validate_expected_counts(
        split_name=split_name,
        case_ids=case_ids,
        mode=mode,
    )

    metadata_rows: list[dict[str, Any]] = []
    sample_index = 0

    with h5py.File(temporary_path, "w") as handle:
        initialize_hdf5_attributes(
            handle=handle,
            split_name=split_name,
            mode=mode,
            case_ids=case_ids,
            expected_samples=expected_samples,
        )

        samples_group = handle.create_group("samples")

        for case_number, case_id in enumerate(
            case_ids,
            start=1,
        ):
            dataset = build_case(
                builder=builder,
                case_id=case_id,
            )

            validate_case_dataset(
                case_id=case_id,
                mode=mode,
                dataset=dataset,
            )

            for sample, metadata in zip(
                dataset["samples"],
                dataset["metadata"],
            ):
                group = samples_group.create_group(
                    str(sample_index)
                )

                for name in (
                    "branch",
                    "query",
                    "target",
                ):
                    group.create_dataset(
                        name,
                        data=np.asarray(sample[name]),
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_LEVEL,
                    )

                metadata_rows.append(
                    dict(metadata)
                )

                sample_index += 1

            print(
                f"[{case_number:02d}/{len(case_ids):02d}] "
                f"{split_name}/{mode} {case_id}: "
                f"{len(dataset['samples'])} samples "
                f"| total={sample_index}",
                flush=True,
            )

        if sample_index != expected_samples:
            raise ValueError(
                f"{split_name}/{mode}: wrote {sample_index} samples, "
                f"expected {expected_samples}"
            )

        write_metadata_group(
            handle,
            metadata_rows,
        )

        handle.attrs["n_samples"] = sample_index

        handle.flush()


def print_plan(
    specs: list[dict[str, Any]],
) -> None:
    """Print the complete deterministic evaluation plan."""

    print("Shared deterministic evaluation datasets")
    print(f"  dataset              : {DATASET_NAME}")
    print(f"  split manifest       : {SPLIT_PATH}")
    print(f"  design root          : {DESIGN_ROOT}")
    print(f"  runs root            : {RUNS_ROOT}")
    print(f"  evaluation builder   : {EVALUATION_BUILDER_PATH}")
    print(
        "  evaluation builder SHA256: "
        f"{sha256_file(EVALUATION_BUILDER_PATH)}"
    )
    print(
        "  training dependency SHA256: "
        f"{sha256_file(TRAINING_BUILDER_PATH)}"
    )
    print(f"  interface points     : {N_INTERFACE_POINTS}")
    print(f"  boundary points      : {N_BOUNDARY_POINTS}")
    print(f"  output fields        : {', '.join(OUTPUT_FIELDS)}")
    print()

    for spec in specs:
        split_name = spec["split_name"]
        mode = spec["mode"]
        case_ids = spec["case_ids"]
        output_path = spec["output_path"]

        expected_samples = validate_expected_counts(
            split_name=split_name,
            case_ids=case_ids,
            mode=mode,
        )

        status = (
            "exists"
            if output_path.exists()
            else "does not exist"
        )

        print(
            f"  {split_name:10s} / {mode:14s}"
            f" | cases={len(case_ids):2d}"
            f" | samples={expected_samples:3d}"
            f" | output={output_path.name}"
            f" | {status}"
        )

    print()
    print(
        "Protocol              : one deterministic decomposition "
        "per full channel"
    )
    print(
        "Expected total        : "
        f"{len(specs) * EXPECTED_SAMPLES_PER_DATASET} samples "
        "across four HDF5 files"
    )


def run_generation(
    specs: list[dict[str, Any]],
) -> None:
    """Generate all four files, then atomically publish each completed HDF5."""

    existing = [
        spec["output_path"]
        for spec in specs
        if spec["output_path"].exists()
    ]

    if existing:
        formatted = "\n".join(
            f"  {path}"
            for path in existing
        )
        raise FileExistsError(
            "Refusing to overwrite existing evaluation datasets:\n"
            f"{formatted}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_paths: list[tuple[Path, Path]] = []

    try:
        for spec in specs:
            output_path = spec["output_path"]

            temporary_path = (
                output_path.parent
                / f".{output_path.name}.tmp-{os.getpid()}"
            )

            if temporary_path.exists():
                raise FileExistsError(
                    f"Temporary output already exists: {temporary_path}"
                )

            temporary_paths.append(
                (temporary_path, output_path)
            )

            print()
            print("=" * 80)
            print(
                f"Building {spec['split_name']} / {spec['mode']}"
            )
            print("=" * 80)

            write_one_dataset(
                split_name=spec["split_name"],
                mode=spec["mode"],
                case_ids=spec["case_ids"],
                builder=spec["builder"],
                temporary_path=temporary_path,
            )

        for temporary_path, output_path in temporary_paths:
            os.replace(
                temporary_path,
                output_path,
            )

        print()
        print("Deterministic evaluation datasets completed")

        for _, output_path in temporary_paths:
            print(f"  {output_path}")

    finally:
        for temporary_path, _ in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic control-point and AR1 "
            "validation/test HDF5 datasets."
        )
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Generate all four evaluation datasets. "
            "Without --run, only validate and print the plan."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split = load_split_manifest(
        SPLIT_PATH
    )

    specs = evaluation_specs(split)

    all_case_ids = tuple(
        sorted(
            {
                case_id
                for spec in specs
                for case_id in spec["case_ids"]
            }
        )
    )

    validate_raw_sources(
        all_case_ids
    )

    print_plan(specs)

    print()
    print(
        "Raw source check       : "
        f"passed for {len(all_case_ids)} validation/test cases"
    )

    if not args.run:
        print("Action                 : validation only")
        return

    print("Action                 : generate four HDF5 files")

    run_generation(specs)


if __name__ == "__main__":
    main()
