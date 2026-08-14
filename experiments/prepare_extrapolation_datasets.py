#!/usr/bin/env python3
"""Build deterministic OOD/extrapolation datasets for PIDiffusion.

This script is intentionally a thin wrapper around the validated deterministic
evaluation builder in ``pidiffusion.diffusion_fluent_evaluation_dataset``.

It does NOT modify the frozen training/evaluation builders and does NOT
reimplement Fluent interpolation or branch construction.

Source data
-----------
Geometry JSON:
    /home/nuoxu9/PIDIF/2d_geometry_specs/channel_water_ablation/channel_XXX.json

Fluent CFD:
    /home/nuoxu9/PIDIF/runs_2d/channel_water_ablation/channel_XXX/
        channel_XXX.msh.h5
        case2d.dat.h5

OOD groups
----------
- ar5_h010:   channel_000..009, L=0.10 mm, global AR=5
- ar20_h010:  channel_010..019, L=0.10 mm, global AR=20
- ar5_h020:   channel_020..029, L=0.20 mm, global AR=5
- ar20_h005:  channel_030..039, L=0.05 mm, global AR=20
- large_delta: channel_040..049, L=0.10 mm, global AR=10,
               larger wall perturbation

For every group, both deterministic decompositions can be generated:
- ar1: equal-width strips, nominal local aspect ratio ~1
- controlpoints: JSON control-point x coordinates used as strip edges

The formal branch/query/target schema is preserved. OOD data are evaluation
only and must never be used to fit model or normalization statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pidiffusion.diffusion_fluent_evaluation_dataset import (
    build_ar1_channel_dataset,
    build_control_point_channel_dataset,
)


REPO_ROOT = Path("/home/nuoxu9/PIDIF")
DEFAULT_SPEC_ROOT = REPO_ROOT / "2d_geometry_specs" / "channel_water_ablation"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs_2d" / "channel_water_ablation"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
)
DEFAULT_REFERENCE_H5 = (
    DEFAULT_OUTPUT_ROOT
    / "channel_deeponet_style_pressure_u_v_random10_train.h5"
)

N_INTERFACE_POINTS = 256
N_BOUNDARY_POINTS = 512


@dataclass(frozen=True)
class OODGroup:
    name: str
    start: int
    stop: int
    expected_l_mm: float
    expected_ar: int
    expected_subdomains: int
    family: str
    description: str

    @property
    def case_indices(self) -> range:
        return range(self.start, self.stop)


GROUPS: tuple[OODGroup, ...] = (
    OODGroup(
        name="ar5_h010",
        start=0,
        stop=10,
        expected_l_mm=0.10,
        expected_ar=5,
        expected_subdomains=5,
        family="global_aspect_ratio",
        description="H=0.10 mm, global AR=5",
    ),
    OODGroup(
        name="ar20_h010",
        start=10,
        stop=20,
        expected_l_mm=0.10,
        expected_ar=20,
        expected_subdomains=20,
        family="global_aspect_ratio",
        description="H=0.10 mm, global AR=20",
    ),
    OODGroup(
        name="ar5_h020",
        start=20,
        stop=30,
        expected_l_mm=0.20,
        expected_ar=5,
        expected_subdomains=5,
        family="geometry_scale",
        description="H=0.20 mm, global AR=5",
    ),
    OODGroup(
        name="ar20_h005",
        start=30,
        stop=40,
        expected_l_mm=0.05,
        expected_ar=20,
        expected_subdomains=20,
        family="geometry_scale",
        description="H=0.05 mm, global AR=20",
    ),
    OODGroup(
        name="large_delta",
        start=40,
        stop=50,
        expected_l_mm=0.10,
        expected_ar=10,
        expected_subdomains=10,
        family="shape_amplitude",
        description="H=0.10 mm, global AR=10, larger wall perturbation",
    ),
)

GROUP_BY_NAME = {g.name: g for g in GROUPS}
DECOMPOSITIONS = ("ar1", "controlpoints")


def _case_id(index: int) -> str:
    return f"channel_{int(index):03d}"


def _source_paths(
    *,
    case_id: str,
    spec_root: Path,
    runs_root: Path,
) -> tuple[Path, Path, Path]:
    design_path = spec_root / f"{case_id}.json"
    run_dir = runs_root / case_id
    mesh_path = run_dir / f"{case_id}.msh.h5"
    dat_path = run_dir / "case2d.dat.h5"
    return design_path, mesh_path, dat_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_source_case(
    *,
    group: OODGroup,
    case_index: int,
    spec_root: Path,
    runs_root: Path,
) -> dict[str, Any]:
    case_id = _case_id(case_index)
    design_path, mesh_path, dat_path = _source_paths(
        case_id=case_id,
        spec_root=spec_root,
        runs_root=runs_root,
    )

    missing = [
        str(path)
        for path in (design_path, mesh_path, dat_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{case_id}: missing required source files:\n  "
            + "\n  ".join(missing)
        )

    spec = _load_json(design_path)
    meta = spec.get("metadata", {})

    required = (
        "L_mm",
        "AR",
        "channel_length_mm",
        "x_points_mm",
        "Uin_mps",
    )
    missing_meta = [key for key in required if key not in meta]
    if missing_meta:
        raise KeyError(
            f"{case_id}: geometry metadata missing {missing_meta}"
        )

    l_mm = float(meta["L_mm"])
    ar = int(meta["AR"])
    channel_length_mm = float(meta["channel_length_mm"])
    x_points = np.asarray(meta["x_points_mm"], dtype=np.float64)

    if not math.isclose(
        l_mm,
        group.expected_l_mm,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{case_id}: L_mm={l_mm}, expected {group.expected_l_mm}"
        )

    if ar != group.expected_ar:
        raise ValueError(
            f"{case_id}: AR={ar}, expected {group.expected_ar}"
        )

    if x_points.ndim != 1 or x_points.size < 2:
        raise ValueError(f"{case_id}: invalid x_points_mm")

    if not np.all(np.isfinite(x_points)):
        raise ValueError(f"{case_id}: x_points_mm contains non-finite values")

    if not np.all(np.diff(x_points) > 0.0):
        raise ValueError(
            f"{case_id}: x_points_mm must be strictly increasing"
        )

    cp_subdomains = int(x_points.size - 1)
    if cp_subdomains != group.expected_subdomains:
        raise ValueError(
            f"{case_id}: control-point decomposition gives "
            f"{cp_subdomains} subdomains, expected "
            f"{group.expected_subdomains}"
        )

    ar1_subdomains = max(1, int(round(channel_length_mm / l_mm)))
    if ar1_subdomains != group.expected_subdomains:
        raise ValueError(
            f"{case_id}: AR1 decomposition gives "
            f"{ar1_subdomains} subdomains, expected "
            f"{group.expected_subdomains}"
        )

    return {
        "case_id": case_id,
        "case_index": int(case_index),
        "design_path": design_path,
        "mesh_path": mesh_path,
        "dat_path": dat_path,
        "l_mm": l_mm,
        "global_ar": ar,
        "channel_length_mm": channel_length_mm,
        "cp_subdomains": cp_subdomains,
        "ar1_subdomains": ar1_subdomains,
        "uin_mps": float(meta["Uin_mps"]),
    }


def _read_attr_names(value: Any) -> list[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).split("\n")


def _load_reference_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Reference HDF5 not found: {path}")

    with h5py.File(path, "r") as handle:
        n_samples = int(handle.attrs["n_samples"])
        if n_samples <= 0:
            raise ValueError(f"Reference HDF5 has no samples: {path}")

        sample0 = handle["samples"]["0"]
        return {
            "branch_channel_names": _read_attr_names(
                handle.attrs["branch_channel_names"]
            ),
            "trunk_channel_names": _read_attr_names(
                handle.attrs["trunk_channel_names"]
            ),
            "output_channel_names": _read_attr_names(
                handle.attrs["output_channel_names"]
            ),
            "n_interface_points": int(
                handle.attrs["n_interface_points"]
            ),
            "n_boundary_points": int(
                handle.attrs["n_boundary_points"]
            ),
            "branch_shape": tuple(sample0["branch"].shape),
            "query_dim": int(sample0["query"].shape[-1]),
            "target_dim": int(sample0["target"].shape[-1]),
        }


def _validate_channel_schema(
    *,
    channel: Mapping[str, Any],
    reference: Mapping[str, Any],
    case_id: str,
) -> None:
    if list(channel["branch_channel_names"]) != list(
        reference["branch_channel_names"]
    ):
        raise ValueError(
            f"{case_id}: branch channel names differ from reference"
        )

    if list(channel["trunk_channel_names"]) != list(
        reference["trunk_channel_names"]
    ):
        raise ValueError(
            f"{case_id}: trunk channel names differ from reference"
        )

    if list(channel["output_channel_names"]) != list(
        reference["output_channel_names"]
    ):
        raise ValueError(
            f"{case_id}: output channel names differ from reference"
        )

    if int(channel["n_interface_points"]) != int(
        reference["n_interface_points"]
    ):
        raise ValueError(
            f"{case_id}: n_interface_points differs from reference"
        )

    if int(channel["n_boundary_points"]) != int(
        reference["n_boundary_points"]
    ):
        raise ValueError(
            f"{case_id}: n_boundary_points differs from reference"
        )

    branch_cols = int(reference["branch_shape"][-1])
    query_dim = int(reference["query_dim"])
    target_dim = int(reference["target_dim"])

    for local_i, sample in enumerate(channel["samples"]):
        branch = np.asarray(sample["branch"])
        query = np.asarray(sample["query"])
        target = np.asarray(sample["target"])

        if branch.ndim != 2 or branch.shape[1] != branch_cols:
            raise ValueError(
                f"{case_id}/sample{local_i}: branch shape {branch.shape}"
            )
        if query.ndim != 2 or query.shape[1] != query_dim:
            raise ValueError(
                f"{case_id}/sample{local_i}: query shape {query.shape}"
            )
        if target.ndim != 2 or target.shape[1] != target_dim:
            raise ValueError(
                f"{case_id}/sample{local_i}: target shape {target.shape}"
            )
        if query.shape[0] != target.shape[0]:
            raise ValueError(
                f"{case_id}/sample{local_i}: query/target row mismatch"
            )
        if not np.all(np.isfinite(branch)):
            raise ValueError(
                f"{case_id}/sample{local_i}: non-finite branch values"
            )
        if not np.all(np.isfinite(query)):
            raise ValueError(
                f"{case_id}/sample{local_i}: non-finite query values"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError(
                f"{case_id}/sample{local_i}: non-finite target values"
            )


def _build_one_channel(
    *,
    decomposition: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    builders: dict[str, Callable[..., dict[str, Any]]] = {
        "ar1": build_ar1_channel_dataset,
        "controlpoints": build_control_point_channel_dataset,
    }
    builder = builders[decomposition]

    return builder(
        case_id=str(source["case_id"]),
        design_path=Path(source["design_path"]),
        mesh_path=Path(source["mesh_path"]),
        dat_path=Path(source["dat_path"]),
        n_interface_points=N_INTERFACE_POINTS,
        n_boundary_points=N_BOUNDARY_POINTS,
        inlet_v=0.0,
        outlet_pressure=0.0,
        wall_u=0.0,
        wall_v=0.0,
    )


def _merge_group_dataset(
    *,
    group: OODGroup,
    decomposition: str,
    sources: Sequence[Mapping[str, Any]],
    reference_schema: Mapping[str, Any],
) -> dict[str, Any]:
    samples: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []
    n_subdomains_per_case: list[int] = []

    branch_channel_names: list[str] | None = None
    trunk_channel_names: list[str] | None = None
    output_channel_names: list[str] | None = None

    for source in sources:
        case_id = str(source["case_id"])
        channel = _build_one_channel(
            decomposition=decomposition,
            source=source,
        )

        _validate_channel_schema(
            channel=channel,
            reference=reference_schema,
            case_id=case_id,
        )

        expected_n_sub = int(group.expected_subdomains)
        actual_n_sub = int(channel["n_subdomains"])
        if actual_n_sub != expected_n_sub:
            raise ValueError(
                f"{case_id}: builder returned {actual_n_sub} subdomains, "
                f"expected {expected_n_sub}"
            )

        if branch_channel_names is None:
            branch_channel_names = list(
                channel["branch_channel_names"]
            )
            trunk_channel_names = list(
                channel["trunk_channel_names"]
            )
            output_channel_names = list(
                channel["output_channel_names"]
            )

        for sample, meta in zip(
            channel["samples"],
            channel["metadata"],
            strict=True,
        ):
            row = dict(meta)
            row.update(
                {
                    "is_ood": 1,
                    "ood_group": group.name,
                    "ood_family": group.family,
                    "ood_description": group.description,
                    "decomposition_mode": decomposition,
                    "source_case_index": int(source["case_index"]),
                    "source_design_path": str(source["design_path"]),
                    "source_mesh_path": str(source["mesh_path"]),
                    "source_dat_path": str(source["dat_path"]),
                    "global_aspect_ratio": int(source["global_ar"]),
                    "channel_height_mm": float(source["l_mm"]),
                    "channel_length_mm": float(
                        source["channel_length_mm"]
                    ),
                    "n_subdomains_case": actual_n_sub,
                }
            )
            samples.append(sample)
            metadata.append(row)

        n_subdomains_per_case.append(actual_n_sub)

    assert branch_channel_names is not None
    assert trunk_channel_names is not None
    assert output_channel_names is not None

    expected_samples = len(sources) * group.expected_subdomains
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"{group.name}/{decomposition}: produced {len(samples)} "
            f"samples, expected {expected_samples}"
        )

    return {
        "samples": samples,
        "metadata": metadata,
        "branch_channel_names": branch_channel_names,
        "trunk_channel_names": trunk_channel_names,
        "output_channel_names": output_channel_names,
        "n_interface_points": N_INTERFACE_POINTS,
        "n_boundary_points": N_BOUNDARY_POINTS,
        "include_interface_endpoints": False,
        "horizontal_interface": False,
        "horizontal_interface_jitter": 0.0,
        "n_subdomains_per_case": n_subdomains_per_case,
        "n_cases": len(sources),
        "ood_group": group.name,
        "ood_family": group.family,
        "decomposition_mode": decomposition,
    }


def _save_dataset_h5(
    dataset: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Save the same variable-length branch/query/target schema used formally."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(dataset["samples"])
    metadata = list(dataset["metadata"])

    with h5py.File(output_path, "w") as handle:
        handle.attrs["branch_channel_names"] = "\n".join(
            dataset["branch_channel_names"]
        )
        handle.attrs["trunk_channel_names"] = "\n".join(
            dataset["trunk_channel_names"]
        )
        handle.attrs["output_channel_names"] = "\n".join(
            dataset["output_channel_names"]
        )
        handle.attrs["n_interface_points"] = int(
            dataset["n_interface_points"]
        )
        handle.attrs["n_boundary_points"] = int(
            dataset["n_boundary_points"]
        )
        handle.attrs["include_interface_endpoints"] = bool(
            dataset.get("include_interface_endpoints", False)
        )
        handle.attrs["horizontal_interface"] = bool(
            dataset.get("horizontal_interface", False)
        )
        handle.attrs["horizontal_interface_jitter"] = float(
            dataset.get("horizontal_interface_jitter", 0.0)
        )
        handle.attrs["n_samples"] = len(samples)
        handle.attrs["n_cases"] = int(dataset["n_cases"])
        handle.attrs["dataset_role"] = "ood_evaluation_only"
        handle.attrs["ood_group"] = str(dataset["ood_group"])
        handle.attrs["ood_family"] = str(dataset["ood_family"])
        handle.attrs["decomposition_mode"] = str(
            dataset["decomposition_mode"]
        )

        samples_group = handle.create_group("samples")
        for i, sample in enumerate(samples):
            sample_group = samples_group.create_group(str(i))
            sample_group.create_dataset(
                "branch",
                data=np.asarray(sample["branch"], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "query",
                data=np.asarray(sample["query"], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "target",
                data=np.asarray(sample["target"], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )

        metadata_group = handle.create_group("metadata")
        keys = sorted({key for row in metadata for key in row})
        for key in keys:
            values = [row.get(key, "") for row in metadata]

            numeric = all(
                isinstance(
                    value,
                    (int, float, np.integer, np.floating, bool, np.bool_),
                )
                for value in values
            )

            if numeric:
                metadata_group.create_dataset(
                    key,
                    data=np.asarray(values),
                )
            else:
                metadata_group.create_dataset(
                    key,
                    data=np.asarray(
                        values,
                        dtype=h5py.string_dtype("utf-8"),
                    ),
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_written_h5(
    *,
    path: Path,
    group: OODGroup,
    decomposition: str,
    reference_schema: Mapping[str, Any],
) -> dict[str, Any]:
    expected_samples = 10 * group.expected_subdomains

    with h5py.File(path, "r") as handle:
        n_samples = int(handle.attrs["n_samples"])
        if n_samples != expected_samples:
            raise RuntimeError(
                f"{path}: n_samples={n_samples}, expected {expected_samples}"
            )

        if int(handle.attrs["n_cases"]) != 10:
            raise RuntimeError(f"{path}: n_cases must be 10")

        if str(handle.attrs["dataset_role"]) != "ood_evaluation_only":
            raise RuntimeError(f"{path}: incorrect dataset_role")

        if str(handle.attrs["ood_group"]) != group.name:
            raise RuntimeError(f"{path}: incorrect ood_group")

        if str(handle.attrs["decomposition_mode"]) != decomposition:
            raise RuntimeError(f"{path}: incorrect decomposition_mode")

        branch_names = _read_attr_names(
            handle.attrs["branch_channel_names"]
        )
        trunk_names = _read_attr_names(
            handle.attrs["trunk_channel_names"]
        )
        output_names = _read_attr_names(
            handle.attrs["output_channel_names"]
        )

        if branch_names != reference_schema["branch_channel_names"]:
            raise RuntimeError(f"{path}: branch schema mismatch")
        if trunk_names != reference_schema["trunk_channel_names"]:
            raise RuntimeError(f"{path}: trunk schema mismatch")
        if output_names != reference_schema["output_channel_names"]:
            raise RuntimeError(f"{path}: output schema mismatch")

        if int(handle.attrs["n_interface_points"]) != N_INTERFACE_POINTS:
            raise RuntimeError(f"{path}: wrong n_interface_points")
        if int(handle.attrs["n_boundary_points"]) != N_BOUNDARY_POINTS:
            raise RuntimeError(f"{path}: wrong n_boundary_points")

        case_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["metadata"]["case_id"][:]
        ]
        subdomain_ids = handle["metadata"]["subdomain_id"][:]
        n_subdomains_case = handle["metadata"]["n_subdomains_case"][:]

        unique_cases = sorted(set(case_ids))
        if len(unique_cases) != 10:
            raise RuntimeError(
                f"{path}: expected 10 unique cases, got {len(unique_cases)}"
            )

        for case_id in unique_cases:
            idx = [
                i for i, value in enumerate(case_ids)
                if value == case_id
            ]
            found_subdomains = sorted(
                int(subdomain_ids[i]) for i in idx
            )
            expected_ids = list(range(group.expected_subdomains))
            if found_subdomains != expected_ids:
                raise RuntimeError(
                    f"{path}: {case_id} subdomain IDs "
                    f"{found_subdomains}, expected {expected_ids}"
                )
            if any(
                int(n_subdomains_case[i]) != group.expected_subdomains
                for i in idx
            ):
                raise RuntimeError(
                    f"{path}: {case_id} inconsistent n_subdomains_case"
                )

        sample0 = handle["samples"]["0"]
        branch_shape = tuple(sample0["branch"].shape)
        query_shape = tuple(sample0["query"].shape)
        target_shape = tuple(sample0["target"].shape)

        if branch_shape != tuple(reference_schema["branch_shape"]):
            raise RuntimeError(
                f"{path}: branch shape {branch_shape} differs from "
                f"reference {reference_schema['branch_shape']}"
            )
        if query_shape[-1] != int(reference_schema["query_dim"]):
            raise RuntimeError(f"{path}: query dimension mismatch")
        if target_shape[-1] != int(reference_schema["target_dim"]):
            raise RuntimeError(f"{path}: target dimension mismatch")

    return {
        "path": str(path),
        "n_cases": 10,
        "n_samples": expected_samples,
        "n_subdomains_per_case": group.expected_subdomains,
        "ood_group": group.name,
        "ood_family": group.family,
        "decomposition": decomposition,
        "sha256": _sha256(path),
    }


def _output_name(group: OODGroup, decomposition: str) -> str:
    return (
        "channel_deeponet_style_pressure_u_v_ood_"
        f"{group.name}_{decomposition}.h5"
    )


def _selected_groups(value: str) -> list[OODGroup]:
    if value == "all":
        return list(GROUPS)
    return [GROUP_BY_NAME[value]]


def _selected_decompositions(value: str) -> list[str]:
    if value == "both":
        return list(DECOMPOSITIONS)
    return [value]


def _print_plan(
    *,
    groups: Sequence[OODGroup],
    decompositions: Sequence[str],
    spec_root: Path,
    runs_root: Path,
    output_root: Path,
    reference_h5: Path,
) -> None:
    print("=" * 100)
    print("PIDiffusion OOD / extrapolation dataset preparation")
    print("=" * 100)
    print("Geometry specs :", spec_root)
    print("Fluent runs    :", runs_root)
    print("Output root    :", output_root)
    print("Reference HDF5 :", reference_h5)
    print("Interface pts  :", N_INTERFACE_POINTS)
    print("Boundary pts   :", N_BOUNDARY_POINTS)
    print("Normalization  : NOT computed from OOD data")
    print("Dataset role   : evaluation only")
    print()

    print(
        f"{'Group':<16} {'Cases':<14} {'H(mm)':>8} {'AR':>5} "
        f"{'Nsub':>6} {'Decompositions'}"
    )
    print("-" * 80)
    for group in groups:
        case_range = (
            f"{_case_id(group.start)}.."
            f"{_case_id(group.stop - 1)}"
        )
        print(
            f"{group.name:<16} {case_range:<14} "
            f"{group.expected_l_mm:>8.3f} "
            f"{group.expected_ar:>5d} "
            f"{group.expected_subdomains:>6d} "
            f"{','.join(decompositions)}"
        )

    print()
    for group in groups:
        for decomposition in decompositions:
            path = output_root / _output_name(group, decomposition)
            n_samples = 10 * group.expected_subdomains
            print(
                f"  {path.name}: 10 cases, "
                f"{group.expected_subdomains} subdomains/case, "
                f"{n_samples} samples"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic OOD/extrapolation HDF5 datasets "
            "for PIDiffusion."
        )
    )
    parser.add_argument(
        "--group",
        choices=["all", *GROUP_BY_NAME.keys()],
        default="all",
        help="OOD geometry group to prepare.",
    )
    parser.add_argument(
        "--decomposition",
        choices=["both", *DECOMPOSITIONS],
        default="both",
        help="Deterministic decomposition to build.",
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--reference-h5",
        type=Path,
        default=DEFAULT_REFERENCE_H5,
        help=(
            "Formal training HDF5 used only for schema parity checks. "
            "No normalization statistics are read or recomputed here."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing OOD HDF5 outputs.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually build and write HDF5 files. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    groups = _selected_groups(args.group)
    decompositions = _selected_decompositions(
        args.decomposition
    )

    spec_root = args.spec_root.resolve()
    runs_root = args.runs_root.resolve()
    output_root = args.output_root.resolve()
    reference_h5 = args.reference_h5.resolve()

    _print_plan(
        groups=groups,
        decompositions=decompositions,
        spec_root=spec_root,
        runs_root=runs_root,
        output_root=output_root,
        reference_h5=reference_h5,
    )

    print("\nValidating source files and geometry metadata...")
    source_by_group: dict[str, list[dict[str, Any]]] = {}

    for group in groups:
        sources = [
            _validate_source_case(
                group=group,
                case_index=case_index,
                spec_root=spec_root,
                runs_root=runs_root,
            )
            for case_index in group.case_indices
        ]
        source_by_group[group.name] = sources
        print(
            f"  {group.name}: PASS "
            f"({len(sources)} complete CFD cases)"
        )

    reference_schema = _load_reference_schema(reference_h5)

    print("\nReference schema:")
    print(
        "  branch channels   :",
        reference_schema["branch_channel_names"],
    )
    print(
        "  trunk channels    :",
        reference_schema["trunk_channel_names"],
    )
    print(
        "  output channels   :",
        reference_schema["output_channel_names"],
    )
    print(
        "  branch shape      :",
        reference_schema["branch_shape"],
    )
    print(
        "  interface points  :",
        reference_schema["n_interface_points"],
    )
    print(
        "  boundary points   :",
        reference_schema["n_boundary_points"],
    )

    if reference_schema["n_interface_points"] != N_INTERFACE_POINTS:
        raise RuntimeError(
            "Reference HDF5 n_interface_points does not match "
            f"{N_INTERFACE_POINTS}"
        )
    if reference_schema["n_boundary_points"] != N_BOUNDARY_POINTS:
        raise RuntimeError(
            "Reference HDF5 n_boundary_points does not match "
            f"{N_BOUNDARY_POINTS}"
        )

    if not args.run:
        print("\nDRY RUN ONLY")
        print("Source audit and reference-schema checks passed.")
        print("Add --run to build the OOD HDF5 files.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for group in groups:
        sources = source_by_group[group.name]

        for decomposition in decompositions:
            output_path = (
                output_root
                / _output_name(group, decomposition)
            )

            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}\n"
                    "Use --overwrite only if replacement is intended."
                )

            print("\n" + "=" * 100)
            print(
                f"BUILD {group.name} / {decomposition}"
            )
            print("=" * 100)

            dataset = _merge_group_dataset(
                group=group,
                decomposition=decomposition,
                sources=sources,
                reference_schema=reference_schema,
            )

            _save_dataset_h5(dataset, output_path)

            record = _validate_written_h5(
                path=output_path,
                group=group,
                decomposition=decomposition,
                reference_schema=reference_schema,
            )
            manifest_rows.append(record)

            print("WROTE :", output_path)
            print("CASES :", record["n_cases"])
            print("SAMPLES:", record["n_samples"])
            print("SHA256:", record["sha256"])
            print("STATUS: PASS")

    manifest_path = (
        output_root / "ood_extrapolation_dataset_manifest.json"
    )
    manifest = {
        "schema_version": "pidiffusion_ood_extrapolation_v1",
        "dataset_role": "ood_evaluation_only",
        "normalization_policy": (
            "Do not fit normalization on OOD data. "
            "Use the frozen training/checkpoint normalizers."
        ),
        "source_spec_root": str(spec_root),
        "source_runs_root": str(runs_root),
        "reference_h5": str(reference_h5),
        "n_interface_points": N_INTERFACE_POINTS,
        "n_boundary_points": N_BOUNDARY_POINTS,
        "outputs": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("OOD DATASET PREPARATION COMPLETED")
    print("=" * 100)
    print("Datasets :", len(manifest_rows))
    print("Manifest :", manifest_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
