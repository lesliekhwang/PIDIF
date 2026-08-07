#!/usr/bin/env python3
"""Generate SNI A/B/C solutions, partition their meshes with METIS, and save them."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from sni_eval_dataset import (
    EVAL_MESH_FILES,
    align_mesh_to_solution,
    build_sni_eval_dataset,
    load_sni_pickle,
    mesh_edges,
    metis_partition,
    read_eval_mesh,
    save_sni_eval_dataset_h5,
    unpack_sni_sample,
)

PDE_NAMES = (
    "laplace2d",
    "laplace2d_mixed",
    "darcy2d",
    "heat2d",
    "nonlinear_poisson2d",
)


def _default_sni_python() -> Path:
    candidate = Path.home() / ".miniconda3/envs/sni_fem/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable)


def _parse_mapping(values: Sequence[str], option: str) -> Dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} entries must use DOMAIN=PATH, got {value!r}")
        domain, path = value.split("=", 1)
        domain = domain.upper()
        if domain not in EVAL_MESH_FILES:
            raise ValueError(f"Unknown evaluation domain {domain!r}")
        result[domain] = Path(path).expanduser().resolve()
    return result


def _parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build DeepONet datasets from SNI evaluation geometries A, B, and C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sni-root", type=Path, default=Path("/home/hantianl/Documents/sni"))
    parser.add_argument("--sni-python", type=Path, default=_default_sni_python())
    parser.add_argument("--pde", choices=PDE_NAMES, default="laplace2d")
    parser.add_argument("--domains", nargs="+", choices=tuple(EVAL_MESH_FILES), default=list(EVAL_MESH_FILES))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=module_dir / "data" / "sni_eval_metis")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DOMAIN=PKL",
        help="Use an existing generate_eval.py pickle for a domain",
    )
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--n-subdomains", type=int, default=4)
    parser.add_argument("--n-partition-realizations", type=int, default=5)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument(
        "--metis-ufactor",
        type=int,
        default=30,
        help="Allowed METIS imbalance in 1/1000 units (30 means about 3 percent)",
    )
    parser.add_argument(
        "--edge-weight-jitter",
        type=float,
        default=0.10,
        help="Relative random jitter applied to graph edge weights",
    )
    parser.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="Do not require each METIS subdomain to be contiguous",
    )
    parser.add_argument("--gpmetis-executable", type=Path, default=None)
    return parser.parse_args()


def _generate_missing_sources(
    args: argparse.Namespace,
    source_paths: Dict[str, Path],
    source_dir: Path,
) -> None:
    missing = [domain for domain in args.domains if domain not in source_paths]
    if not missing:
        return
    script = args.sni_root / "data_generation" / "generate_eval.py"
    if not script.is_file():
        raise FileNotFoundError(f"SNI generate_eval.py not found: {script}")
    python = args.sni_python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"SNI FEM Python not found: {python}")
    source_dir.mkdir(parents=True, exist_ok=True)
    for domain in missing:
        expected = source_dir / f"{args.pde}_{domain}_{args.num_samples}_test.pkl"
        if args.regenerate or not expected.is_file():
            command = [
                str(python),
                str(script),
                "--pde",
                args.pde,
                "--domain",
                domain,
                "--num_samples",
                str(args.num_samples),
                "--output_dir",
                str(source_dir),
            ]
            print(f"Generating SNI {args.pde} solutions on geometry {domain}", flush=True)
            subprocess.run(command, cwd=args.sni_root, check=True)
        source_paths[domain] = expected


def _split_indices(
    n_samples: int,
    valid_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, List[int]]:
    if not 0.0 <= valid_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("valid_fraction and test_fraction must be in [0, 1)")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid_fraction + test_fraction must be < 1")
    order = rng.permutation(n_samples)
    n_valid = int(round(valid_fraction * n_samples))
    n_test = int(round(test_fraction * n_samples))
    if valid_fraction > 0.0 and n_samples >= 2:
        n_valid = max(n_valid, 1)
    if test_fraction > 0.0 and n_samples - n_valid >= 2:
        n_test = max(n_test, 1)
    if n_valid + n_test >= n_samples:
        raise ValueError("The requested split leaves no training samples")
    n_train = n_samples - n_valid - n_test
    return {
        "train": order[:n_train].astype(int).tolist(),
        "valid": order[n_train : n_train + n_valid].astype(int).tolist(),
        "test": order[n_train + n_valid :].astype(int).tolist(),
    }


def main() -> None:
    args = _parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.n_subdomains < 2:
        raise ValueError("n_subdomains must be at least 2")
    if args.n_partition_realizations < 1:
        raise ValueError("n_partition_realizations must be positive")

    args.sni_root = args.sni_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_dir = output_dir / "source"
    source_paths = _parse_mapping(args.source, "--source")
    unknown_sources = sorted(set(source_paths) - set(args.domains))
    if unknown_sources:
        raise ValueError(f"--source domains not selected by --domains: {unknown_sources}")
    _generate_missing_sources(args, source_paths, source_dir)

    source_by_domain = {}
    shape_data = {}
    partitions = {}
    split_by_domain = {}
    split_rng = np.random.default_rng(args.split_seed)
    gpmetis = args.gpmetis_executable
    if gpmetis is not None:
        gpmetis = gpmetis.expanduser().resolve()
        if not gpmetis.is_file():
            raise FileNotFoundError(f"gpmetis executable not found: {gpmetis}")

    for domain_index, domain in enumerate(args.domains):
        source_path = source_paths[domain]
        if not source_path.is_file():
            raise FileNotFoundError(f"SNI source pickle not found: {source_path}")
        source_samples = load_sni_pickle(source_path)
        sol, _, _ = unpack_sni_sample(source_samples[0])
        mesh_path = args.sni_root / "data" / "mesh" / EVAL_MESH_FILES[domain]
        mesh_points, triangles = read_eval_mesh(mesh_path)
        points, triangles = align_mesh_to_solution(mesh_points, triangles, sol[:, :2])
        edges = mesh_edges(triangles)
        source_by_domain[domain] = source_samples
        shape_data[domain] = {
            "points": points,
            "triangles": triangles,
            "edges": edges,
            "mesh_path": str(mesh_path),
        }
        split_by_domain[domain] = _split_indices(
            len(source_samples), args.valid_fraction, args.test_fraction, split_rng
        )
        partitions[domain] = []
        for realization_id in range(args.n_partition_realizations):
            seed = int(args.partition_seed + 100_000 * domain_index + realization_id)
            labels = metis_partition(
                points.shape[0],
                edges,
                args.n_subdomains,
                seed,
                gpmetis_executable=gpmetis,
                ufactor=args.metis_ufactor,
                contiguous=not args.allow_disconnected,
                edge_weight_jitter=args.edge_weight_jitter,
            )
            partitions[domain].append(
                {"labels": labels, "seed": seed, "n_parts": args.n_subdomains}
            )
        print(
            f"Geometry {domain}: {points.shape[0]} nodes, {triangles.shape[0]} triangles, "
            f"{args.n_partition_realizations} METIS realizations; sizes="
            f"{[np.bincount(part['labels'], minlength=args.n_subdomains).tolist() for part in partitions[domain]]}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "pde": args.pde,
        "domains": list(args.domains),
        "source_files": {key: str(value) for key, value in source_paths.items()},
        "mesh_files": {
            domain: str(args.sni_root / "data" / "mesh" / EVAL_MESH_FILES[domain])
            for domain in args.domains
        },
        "n_subdomains": args.n_subdomains,
        "n_partition_realizations": args.n_partition_realizations,
        "partition_seed": args.partition_seed,
        "metis_ufactor": args.metis_ufactor,
        "edge_weight_jitter": args.edge_weight_jitter,
        "contiguous": not args.allow_disconnected,
        "split_seed": args.split_seed,
        "splits": {},
    }
    for split_name in ("train", "valid", "test"):
        selected = {domain: split_by_domain[domain][split_name] for domain in args.domains}
        if not any(selected.values()):
            continue
        dataset = build_sni_eval_dataset(
            source_by_domain,
            selected,
            shape_data,
            partitions,
            pde_name=args.pde,
        )
        output_path = output_dir / f"{split_name}.h5"
        save_sni_eval_dataset_h5(dataset, output_path)
        manifest["splits"][split_name] = {
            "path": str(output_path),
            "source_sample_indices": selected,
            "n_subdomain_samples": len(dataset["samples"]),
        }
        print(f"Saved {len(dataset['samples'])} {split_name} subdomains to {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Saved generation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
