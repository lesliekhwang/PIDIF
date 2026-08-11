"""Prepare the fixed randomized validation dataset for diffusion model selection.

This dataset uses the fixed 20 validation channels from the shared 160/20/20
case split. Each validation channel is decomposed five times with the same
random x-strip procedure used for the finalized randomized training dataset.
The decomposition RNG is fixed once for the full dataset so the resulting
1,000 validation subdomains are reproducible and can be frozen for model
selection.

By default this script validates the plan only. Pass ``--run`` to generate the
HDF5 file. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pidiffusion.diffusion_fluent_dataset import build_randomized_channel_dataset
from pidiffusion.provenance import sha256_file


DATASET_NAME = "channel_water"

DEFAULT_SPLIT_PATH = (
    REPO_ROOT
    / "splits"
    / "channel_case_split_160train_20val_20test.json"
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)

DESIGN_ROOT = REPO_ROOT / "2d_geometry_specs" / DATASET_NAME
RUNS_ROOT = REPO_ROOT / "runs_2d" / DATASET_NAME

N_VALIDATION_CASES = 20
N_SUBDOMAINS = 10
N_REALIZATIONS = 5
EXPECTED_SAMPLES_PER_CASE = N_SUBDOMAINS * N_REALIZATIONS
EXPECTED_TOTAL_SAMPLES = N_VALIDATION_CASES * EXPECTED_SAMPLES_PER_CASE

N_INTERFACE_POINTS = 256
N_BOUNDARY_POINTS = 512

INTERFACE_PLACEMENT = "random"
INTERFACE_JITTER = 0.0
MIN_SUBDOMAIN_WIDTH = 0.01

INSERT_SHARP_CONTROL_POINT_INTERFACES = False
HORIZONTAL_INTERFACE = False
HORIZONTAL_INTERFACE_JITTER = 0.0

# This seed controls only the randomized validation decompositions. A single
# RNG stream is created once and consumed continuously across all 20 cases.
DECOMPOSITION_SEED = 1

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

BC_KWARGS = {
    "inlet_v": 0.0,
    "outlet_pressure": 0.0,
    "wall_u": 0.0,
    "wall_v": 0.0,
}

EXPECTED_BRANCH_CHANNELS = (
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

EXPECTED_TRUNK_CHANNELS = (
    "x_local",
    "y_local",
)

COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the fixed channel-level split manifest."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Split manifest does not exist: {resolved_path}"
        )
    if not resolved_path.is_file():
        raise IsADirectoryError(
            f"Split manifest is not a regular file: {resolved_path}"
        )

    with resolved_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    required_keys = {
        "schema_version",
        "dataset_name",
        "counts",
        "train_cases",
        "validation_cases",
        "test_cases",
    }
    missing = sorted(required_keys.difference(split))
    if missing:
        raise ValueError(
            "Split manifest is missing required keys: " + ", ".join(missing)
        )

    if split["schema_version"] != "channel_case_split_v1":
        raise ValueError(
            "Unexpected split schema version: "
            f"{split['schema_version']!r}"
        )
    if split["dataset_name"] != DATASET_NAME:
        raise ValueError(
            "Split dataset does not match the expected dataset: "
            f"{split['dataset_name']!r} != {DATASET_NAME!r}"
        )

    train_cases = tuple(str(case_id) for case_id in split["train_cases"])
    validation_cases = tuple(
        str(case_id) for case_id in split["validation_cases"]
    )
    test_cases = tuple(str(case_id) for case_id in split["test_cases"])

    if len(train_cases) != 160:
        raise ValueError(
            f"Expected 160 training cases, got {len(train_cases)}"
        )
    if len(validation_cases) != N_VALIDATION_CASES:
        raise ValueError(
            f"Expected {N_VALIDATION_CASES} validation cases, "
            f"got {len(validation_cases)}"
        )
    if len(test_cases) != 20:
        raise ValueError(f"Expected 20 test cases, got {len(test_cases)}")

    train_set = set(train_cases)
    validation_set = set(validation_cases)
    test_set = set(test_cases)

    if len(train_set) != len(train_cases):
        raise ValueError("Training split contains duplicate case IDs")
    if len(validation_set) != len(validation_cases):
        raise ValueError("Validation split contains duplicate case IDs")
    if len(test_set) != len(test_cases):
        raise ValueError("Test split contains duplicate case IDs")

    if train_set & validation_set:
        raise ValueError("Training and validation cases overlap")
    if train_set & test_set:
        raise ValueError("Training and test cases overlap")
    if validation_set & test_set:
        raise ValueError("Validation and test cases overlap")
    if len(train_set | validation_set | test_set) != 200:
        raise ValueError(
            "The split must contain exactly 200 unique channel cases"
        )

    expected_counts = {
        "total": 200,
        "train": 160,
        "validation": 20,
        "test": 20,
    }
    if split["counts"] != expected_counts:
        raise ValueError(
            "Split counts do not match the expected fixed protocol: "
            f"{split['counts']}"
        )

    return split


def case_paths(case_id: str) -> dict[str, Path]:
    """Return the design, mesh, and solution paths for one channel."""

    return {
        "design": DESIGN_ROOT / f"{case_id}.json",
        "mesh": RUNS_ROOT / case_id / f"{case_id}.msh.h5",
        "dat": RUNS_ROOT / case_id / "case2d.dat.h5",
    }


def build_case_file_map(
    case_ids: Sequence[str],
) -> dict[str, dict[str, Path]]:
    """Build and validate the raw-file mapping for selected cases."""

    case_files: dict[str, dict[str, Path]] = {}
    missing_files: list[Path] = []

    for case_id in case_ids:
        paths = case_paths(case_id)
        case_files[case_id] = paths
        for path in paths.values():
            if not path.is_file():
                missing_files.append(path)

    if missing_files:
        preview = "\n".join(
            f"  - {path}" for path in missing_files[:20]
        )
        suffix = ""
        if len(missing_files) > 20:
            suffix = f"\n  ... and {len(missing_files) - 20} more"
        raise FileNotFoundError(
            "Required raw dataset files are missing:\n"
            f"{preview}{suffix}"
        )

    return case_files


def validate_case_dataset(
    *,
    case_id: str,
    dataset: Mapping[str, Any],
) -> None:
    """Validate one validation channel's five randomized decompositions."""

    samples = list(dataset["samples"])
    metadata = list(dataset["metadata"])

    if len(samples) != EXPECTED_SAMPLES_PER_CASE:
        raise ValueError(
            f"{case_id}: expected {EXPECTED_SAMPLES_PER_CASE} samples, "
            f"got {len(samples)}"
        )
    if len(metadata) != EXPECTED_SAMPLES_PER_CASE:
        raise ValueError(
            f"{case_id}: sample and metadata counts differ"
        )

    branch_names = tuple(dataset["branch_channel_names"])
    trunk_names = tuple(dataset["trunk_channel_names"])
    output_names = tuple(dataset["output_channel_names"])

    if branch_names != EXPECTED_BRANCH_CHANNELS:
        raise ValueError(
            f"{case_id}: unexpected branch channels: {branch_names}"
        )
    if trunk_names != EXPECTED_TRUNK_CHANNELS:
        raise ValueError(
            f"{case_id}: unexpected trunk channels: {trunk_names}"
        )
    if output_names != OUTPUT_FIELDS:
        raise ValueError(
            f"{case_id}: unexpected output channels: {output_names}"
        )
    if dataset["interface_placement"] != INTERFACE_PLACEMENT:
        raise ValueError(f"{case_id}: unexpected interface placement")
    if int(dataset["n_realizations"]) != N_REALIZATIONS:
        raise ValueError(f"{case_id}: unexpected realization count")
    if bool(dataset["insert_sharp_control_point_interfaces"]):
        raise ValueError(
            f"{case_id}: sharp control-point insertion must be disabled"
        )
    if bool(dataset["horizontal_interface"]):
        raise ValueError(
            f"{case_id}: horizontal interface must be disabled"
        )

    realization_to_subdomains: dict[int, list[int]] = {}

    for sample_index, (sample, meta) in enumerate(zip(samples, metadata)):
        metadata_case_id = str(meta["case_id"])
        if metadata_case_id != case_id:
            raise ValueError(
                f"{case_id}: metadata row {sample_index} belongs to "
                f"{metadata_case_id}"
            )

        realization_id = int(meta["realization_id"])
        subdomain_id = int(meta["subdomain_id"])
        realization_to_subdomains.setdefault(realization_id, []).append(
            subdomain_id
        )

        branch = np.asarray(sample["branch"])
        query = np.asarray(sample["query"])
        target = np.asarray(sample["target"])

        if branch.ndim != 2:
            raise ValueError(
                f"{case_id}: sample {sample_index} branch must be 2D"
            )
        if query.ndim != 2:
            raise ValueError(
                f"{case_id}: sample {sample_index} query must be 2D"
            )
        if target.ndim != 2:
            raise ValueError(
                f"{case_id}: sample {sample_index} target must be 2D"
            )
        if branch.shape[1] != len(EXPECTED_BRANCH_CHANNELS):
            raise ValueError(
                f"{case_id}: sample {sample_index} has an unexpected "
                "branch width"
            )
        if query.shape[1] != len(EXPECTED_TRUNK_CHANNELS):
            raise ValueError(
                f"{case_id}: sample {sample_index} has an unexpected "
                "query width"
            )
        if target.shape[1] != len(OUTPUT_FIELDS):
            raise ValueError(
                f"{case_id}: sample {sample_index} has an unexpected "
                "target width"
            )
        if query.shape[0] != target.shape[0]:
            raise ValueError(
                f"{case_id}: sample {sample_index} query/target row "
                "counts differ"
            )
        if query.shape[0] == 0:
            raise ValueError(
                f"{case_id}: sample {sample_index} contains no query points"
            )
        if not np.isfinite(branch).all():
            raise ValueError(
                f"{case_id}: sample {sample_index} contains non-finite "
                "branch values"
            )
        if not np.isfinite(query).all():
            raise ValueError(
                f"{case_id}: sample {sample_index} contains non-finite "
                "query values"
            )
        if not np.isfinite(target).all():
            raise ValueError(
                f"{case_id}: sample {sample_index} contains non-finite "
                "target values"
            )

    expected_realizations = set(range(N_REALIZATIONS))
    if set(realization_to_subdomains) != expected_realizations:
        raise ValueError(
            f"{case_id}: realization IDs are not exactly 0 through "
            f"{N_REALIZATIONS - 1}"
        )

    expected_subdomains = list(range(N_SUBDOMAINS))
    for realization_id in range(N_REALIZATIONS):
        observed = sorted(realization_to_subdomains[realization_id])
        if observed != expected_subdomains:
            raise ValueError(
                f"{case_id}: realization {realization_id} does not contain "
                "exactly subdomains 0 through 9"
            )


def write_metadata_group(
    handle: h5py.File,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write metadata using the existing shared HDF5 schema."""

    metadata_group = handle.create_group("metadata")
    keys = sorted(
        {
            key
            for row in metadata_rows
            for key in row.keys()
        }
    )

    numeric_types = (
        bool,
        int,
        float,
        np.bool_,
        np.integer,
        np.floating,
    )

    for key in keys:
        values = [row.get(key, "") for row in metadata_rows]

        if all(isinstance(value, numeric_types) for value in values):
            metadata_group.create_dataset(key, data=np.asarray(values))
            continue

        text_values = np.asarray(
            ["" if value is None else str(value) for value in values],
            dtype=h5py.string_dtype("utf-8"),
        )
        metadata_group.create_dataset(key, data=text_values)


def initialize_hdf5_attributes(
    *,
    handle: h5py.File,
    dataset: Mapping[str, Any],
    split_path: Path,
) -> None:
    """Write schema and provenance attributes to the output HDF5."""

    builder_path = (
        REPO_ROOT / "pidiffusion" / "diffusion_fluent_dataset.py"
    )
    generator_path = Path(__file__).resolve()

    handle.attrs["branch_channel_names"] = "\n".join(
        dataset["branch_channel_names"]
    )
    handle.attrs["trunk_channel_names"] = "\n".join(
        dataset["trunk_channel_names"]
    )
    handle.attrs["output_channel_names"] = "\n".join(
        dataset["output_channel_names"]
    )

    handle.attrs["n_interface_points"] = N_INTERFACE_POINTS
    handle.attrs["n_boundary_points"] = N_BOUNDARY_POINTS
    handle.attrs["include_interface_endpoints"] = False
    handle.attrs["horizontal_interface"] = HORIZONTAL_INTERFACE
    handle.attrs[
        "horizontal_interface_jitter"
    ] = HORIZONTAL_INTERFACE_JITTER

    handle.attrs["dataset_name"] = DATASET_NAME
    handle.attrs["dataset_role"] = "canonical_randomized_validation"
    handle.attrs["split_role"] = "validation"

    handle.attrs["interface_placement"] = INTERFACE_PLACEMENT
    handle.attrs["interface_jitter"] = INTERFACE_JITTER
    handle.attrs["n_subdomains"] = N_SUBDOMAINS
    handle.attrs["n_realizations"] = N_REALIZATIONS
    handle.attrs["min_subdomain_width"] = MIN_SUBDOMAIN_WIDTH
    handle.attrs[
        "insert_sharp_control_point_interfaces"
    ] = INSERT_SHARP_CONTROL_POINT_INTERFACES

    handle.attrs["decomposition_seed"] = DECOMPOSITION_SEED
    handle.attrs["decomposition_rng"] = "numpy.default_rng"
    handle.attrs[
        "decomposition_rng_scope"
    ] = "single_continuous_stream_across_all_validation_cases"

    handle.attrs["expected_validation_cases"] = N_VALIDATION_CASES
    handle.attrs[
        "expected_samples_per_case"
    ] = EXPECTED_SAMPLES_PER_CASE
    handle.attrs["expected_total_samples"] = EXPECTED_TOTAL_SAMPLES

    handle.attrs["split_manifest_path"] = str(split_path.resolve())
    handle.attrs["split_manifest_sha256"] = sha256_file(split_path)
    handle.attrs["dataset_builder_path"] = str(builder_path.resolve())
    handle.attrs["dataset_builder_sha256"] = sha256_file(builder_path)
    handle.attrs["generator_script_path"] = str(generator_path)
    handle.attrs["generator_script_sha256"] = sha256_file(generator_path)


def build_and_write_dataset(
    *,
    validation_cases: Sequence[str],
    case_files: Mapping[str, Mapping[str, Path]],
    split_path: Path,
    output_path: Path,
) -> None:
    """Build the fixed randomized validation dataset one channel at a time."""

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {resolved_output}"
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_output.with_name(
        f".{resolved_output.name}.tmp-{os.getpid()}"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    rng = np.random.default_rng(DECOMPOSITION_SEED)
    metadata_rows: list[dict[str, Any]] = []
    total_samples = 0
    reference_channels: tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ] | None = None

    try:
        with h5py.File(temporary_path, "w") as handle:
            samples_group = handle.create_group("samples")

            for case_number, case_id in enumerate(
                validation_cases,
                start=1,
            ):
                paths = case_files[case_id]
                dataset = build_randomized_channel_dataset(
                    case_id=case_id,
                    design_path=paths["design"],
                    mesh_path=paths["mesh"],
                    dat_path=paths["dat"],
                    n_subdomains=N_SUBDOMAINS,
                    n_realizations=N_REALIZATIONS,
                    n_interface_points=N_INTERFACE_POINTS,
                    n_boundary_points=N_BOUNDARY_POINTS,
                    min_subdomain_width=MIN_SUBDOMAIN_WIDTH,
                    rng=rng,
                    inlet_v=BC_KWARGS["inlet_v"],
                    outlet_pressure=BC_KWARGS["outlet_pressure"],
                    wall_u=BC_KWARGS["wall_u"],
                    wall_v=BC_KWARGS["wall_v"],
                    field_map=FIELD_MAP,
                )

                validate_case_dataset(case_id=case_id, dataset=dataset)

                current_channels = (
                    tuple(dataset["branch_channel_names"]),
                    tuple(dataset["trunk_channel_names"]),
                    tuple(dataset["output_channel_names"]),
                )

                if reference_channels is None:
                    reference_channels = current_channels
                    initialize_hdf5_attributes(
                        handle=handle,
                        dataset=dataset,
                        split_path=split_path,
                    )
                elif current_channels != reference_channels:
                    raise ValueError(
                        f"{case_id}: channel schema differs from previous cases"
                    )

                samples = list(dataset["samples"])
                metadata = list(dataset["metadata"])

                for sample, meta in zip(samples, metadata):
                    sample_group = samples_group.create_group(str(total_samples))
                    sample_group.create_dataset(
                        "branch",
                        data=np.asarray(sample["branch"], dtype=np.float32),
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_LEVEL,
                    )
                    sample_group.create_dataset(
                        "query",
                        data=np.asarray(sample["query"], dtype=np.float32),
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_LEVEL,
                    )
                    sample_group.create_dataset(
                        "target",
                        data=np.asarray(sample["target"], dtype=np.float32),
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_LEVEL,
                    )
                    metadata_rows.append(dict(meta))
                    total_samples += 1

                print(
                    f"[{case_number:02d}/{len(validation_cases):02d}] "
                    f"{case_id}: {len(samples)} samples | total={total_samples}",
                    flush=True,
                )

                del dataset
                del samples
                del metadata

            if total_samples != EXPECTED_TOTAL_SAMPLES:
                raise ValueError(
                    "Unexpected final sample count: "
                    f"expected={EXPECTED_TOTAL_SAMPLES}, actual={total_samples}"
                )
            if len(metadata_rows) != total_samples:
                raise ValueError(
                    "Metadata count does not match the final sample count"
                )

            write_metadata_group(handle, metadata_rows)
            handle.attrs["n_samples"] = total_samples
            handle.flush()

        os.replace(temporary_path, resolved_output)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print()
    print("Randomized validation dataset completed")
    print(f"  cases      : {len(validation_cases)}")
    print(f"  samples    : {total_samples}")
    print(f"  output     : {resolved_output}")


def print_plan(
    *,
    split_path: Path,
    output_path: Path,
    validation_cases: Sequence[str],
) -> None:
    """Print the exact randomized validation generation protocol."""

    builder_path = (
        REPO_ROOT / "pidiffusion" / "diffusion_fluent_dataset.py"
    )

    print("Canonical randomized validation dataset")
    print(f"  dataset              : {DATASET_NAME}")
    print(f"  split manifest       : {split_path.resolve()}")
    print(f"  split role           : validation")
    print(f"  validation cases     : {len(validation_cases)}")
    print(f"  partition            : {INTERFACE_PLACEMENT}")
    print(f"  subdomains           : {N_SUBDOMAINS}")
    print(f"  realizations/case    : {N_REALIZATIONS}")
    print(f"  samples/case         : {EXPECTED_SAMPLES_PER_CASE}")
    print(f"  expected samples     : {EXPECTED_TOTAL_SAMPLES}")
    print(f"  min width fraction   : {MIN_SUBDOMAIN_WIDTH}")
    print(
        f"  sharp CP insertion   : {INSERT_SHARP_CONTROL_POINT_INTERFACES}"
    )
    print(f"  horizontal split     : {HORIZONTAL_INTERFACE}")
    print(f"  interface points     : {N_INTERFACE_POINTS}")
    print(f"  boundary points      : {N_BOUNDARY_POINTS}")
    print(f"  output fields        : {', '.join(OUTPUT_FIELDS)}")
    print(f"  decomposition seed   : {DECOMPOSITION_SEED}")
    print("  RNG scope            : one continuous stream across all cases")
    print(f"  design root          : {DESIGN_ROOT}")
    print(f"  runs root            : {RUNS_ROOT}")
    print(f"  builder              : {builder_path}")
    print(f"  output               : {output_path.resolve()}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the 1,000-sample fixed randomized validation dataset "
            "used for diffusion checkpoint selection."
        )
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Fixed channel-level split JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output HDF5 dataset path.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Build and write the dataset. Without this flag, only validate "
            "the raw sources and print the plan."
        ),
    )
    return parser


def main() -> int:
    """Validate the plan and optionally build the validation dataset."""

    args = build_argument_parser().parse_args()
    split_path = args.split_file.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    split = load_split_manifest(split_path)
    validation_cases = tuple(
        str(case_id) for case_id in split["validation_cases"]
    )
    case_files = build_case_file_map(validation_cases)

    print_plan(
        split_path=split_path,
        output_path=output_path,
        validation_cases=validation_cases,
    )

    print()
    print(
        "Raw source check       : "
        f"passed for {len(validation_cases)} validation cases"
    )
    if output_path.exists():
        print("Output status          : already exists")
    else:
        print("Output status          : does not exist")

    if not args.run:
        print("Action                 : validation only")
        return 0

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {output_path}"
        )

    print("Action                 : build full dataset")
    print()
    build_and_write_dataset(
        validation_cases=validation_cases,
        case_files=case_files,
        split_path=split_path,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
