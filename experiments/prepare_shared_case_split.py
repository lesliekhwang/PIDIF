"""Prepare the fixed channel-level split shared by DeepONet and diffusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "splits"
    / "channel_case_split_160train_20val_20test.json"
)

EXPECTED_CASES = tuple(
    f"channel_{index:02d}"
    for index in range(200)
)

VALIDATION_CASES = (
    "channel_08",
    "channel_34",
    "channel_35",
    "channel_37",
    "channel_48",
    "channel_50",
    "channel_53",
    "channel_62",
    "channel_75",
    "channel_80",
    "channel_87",
    "channel_141",
    "channel_142",
    "channel_154",
    "channel_163",
    "channel_164",
    "channel_171",
    "channel_175",
    "channel_176",
    "channel_191",
)

TEST_CASES = (
    "channel_18",
    "channel_22",
    "channel_28",
    "channel_31",
    "channel_33",
    "channel_45",
    "channel_51",
    "channel_71",
    "channel_81",
    "channel_89",
    "channel_90",
    "channel_97",
    "channel_111",
    "channel_117",
    "channel_118",
    "channel_133",
    "channel_151",
    "channel_172",
    "channel_193",
    "channel_199",
)


def case_index(case_id: str) -> int:
    """Return the numeric suffix of a channel case identifier."""

    prefix = "channel_"

    if not case_id.startswith(prefix):
        raise ValueError(
            f"Invalid channel case identifier: {case_id!r}"
        )

    suffix = case_id[len(prefix):]

    if not suffix.isdigit():
        raise ValueError(
            f"Invalid channel case identifier: {case_id!r}"
        )

    return int(suffix)


def validate_case_group(
    case_ids: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate one group of case IDs and return numeric ordering."""

    normalized = tuple(str(case_id) for case_id in case_ids)

    for case_id in normalized:
        case_index(case_id)

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{label} contains duplicate case IDs"
        )

    unknown = set(normalized).difference(EXPECTED_CASES)
    if unknown:
        raise ValueError(
            f"{label} contains unexpected case IDs: "
            f"{sorted(unknown, key=case_index)}"
        )

    return tuple(
        sorted(
            normalized,
            key=case_index,
        )
    )


def build_fixed_split() -> dict[str, object]:
    """Build and validate the recovered 160/20/20 channel split."""

    validation_cases = validate_case_group(
        VALIDATION_CASES,
        label="Validation split",
    )
    test_cases = validate_case_group(
        TEST_CASES,
        label="Test split",
    )

    validation_set = set(validation_cases)
    test_set = set(test_cases)

    validation_test_overlap = validation_set & test_set
    if validation_test_overlap:
        raise ValueError(
            "Validation and test splits overlap: "
            f"{sorted(validation_test_overlap, key=case_index)}"
        )

    train_cases = tuple(
        case_id
        for case_id in EXPECTED_CASES
        if case_id not in validation_set
        and case_id not in test_set
    )

    train_cases = validate_case_group(
        train_cases,
        label="Training split",
    )

    train_set = set(train_cases)

    if train_set & validation_set:
        raise ValueError(
            "Training and validation splits overlap"
        )

    if train_set & test_set:
        raise ValueError(
            "Training and test splits overlap"
        )

    all_split_cases = (
        train_set
        | validation_set
        | test_set
    )

    if all_split_cases != set(EXPECTED_CASES):
        raise ValueError(
            "The split does not cover exactly "
            "channel_00 through channel_199"
        )

    expected_counts = {
        "train": 160,
        "validation": 20,
        "test": 20,
    }

    actual_counts = {
        "train": len(train_cases),
        "validation": len(validation_cases),
        "test": len(test_cases),
    }

    if actual_counts != expected_counts:
        raise ValueError(
            "Unexpected split counts: "
            f"expected={expected_counts}, "
            f"actual={actual_counts}"
        )

    return {
        "schema_version": "channel_case_split_v1",
        "dataset_name": "channel_water",
        "split_name": "160train_20val_20test",
        "source": (
            "DeepONet/train_deeponet.ipynb "
            "executed split output"
        ),
        "historical_split_seed": 0,
        "generation_policy": (
            "Exact recovered case lists; "
            "do not regenerate from filesystem order."
        ),
        "counts": {
            "total": len(EXPECTED_CASES),
            **actual_counts,
        },
        "train_cases": list(train_cases),
        "validation_cases": list(validation_cases),
        "test_cases": list(test_cases),
    }


def atomic_write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write a JSON file atomically without overwriting an existing split."""

    path = path.expanduser().resolve()

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split file: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the fixed 160/20/20 channel-level "
            "split shared by DeepONet and diffusion."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the split JSON manifest.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the validated split manifest.",
    )

    return parser


def main() -> int:
    """Validate the split and optionally write the JSON manifest."""

    args = build_argument_parser().parse_args()

    split = build_fixed_split()
    counts = split["counts"]
    output_path = args.output.expanduser().resolve()

    print("Shared channel case split")
    print(f"  dataset    : {split['dataset_name']}")
    print(f"  train      : {counts['train']}")
    print(f"  validation : {counts['validation']}")
    print(f"  test       : {counts['test']}")
    print(f"  total      : {counts['total']}")
    print(f"  output     : {output_path}")

    if not args.write:
        print("  action     : validation only")
        return 0

    atomic_write_json(
        output_path,
        split,
    )

    print("  action     : split manifest written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
