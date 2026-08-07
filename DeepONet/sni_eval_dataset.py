"""DeepONet datasets built from SNI's fixed A/B/C evaluation geometries.

The source pickle format is the one written by
``sni/data_generation/generate_eval.py``.  METIS partitions the evaluation
mesh vertex graph; every cut edge contributes its midpoint and interpolated
solution trace to both adjacent subdomains as an interface sensor.

This module is intentionally independent of ``deeponet_fluent_dataset``.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import meshio
import numpy as np
import torch
from scipy.spatial import cKDTree

PathLike = Union[str, Path]

EVAL_MESH_FILES = {
    "A": "A-schwarz.msh",
    "B": "B-holes.msh",
    "C": "C-bosch.msh",
}


def load_sni_pickle(path: PathLike) -> List[tuple]:
    path = Path(path)
    with path.open("rb") as stream:
        samples = pickle.load(stream)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}")
    return samples


def unpack_sni_sample(sample: tuple) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """Return ``(sol, scalar_parameters, coordinate_input_functions)``."""
    if not isinstance(sample, tuple) or len(sample) < 2:
        raise ValueError("An SNI sample must be a tuple with sol and input functions")
    sol = np.asarray(sample[0])
    input_functions = sample[-1]
    if sol.ndim != 2 or sol.shape[1] < 3:
        raise ValueError(f"sol must have shape (N, 2+C), got {sol.shape}")
    if not isinstance(input_functions, (list, tuple)) or not input_functions:
        raise ValueError("The last sample item must be a non-empty input-function list")
    inputs = [np.asarray(value) for value in input_functions]
    for value in inputs:
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError(f"Input functions must have shape (N, 2+C), got {value.shape}")
    parameters = []
    for value in sample[1:-1]:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(
                "Only scalar non-spatial parameters are supported; encode spatial "
                "parameters as coordinate input functions"
            )
        parameters.append(float(array.reshape(-1)[0]))
    return sol, parameters, inputs


def read_eval_mesh(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read mesh points and all triangular cells from an SNI Gmsh file."""
    mesh = meshio.read(Path(path))
    triangles = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    if not triangles:
        raise ValueError(f"No triangular cells found in {path}")
    return (
        np.asarray(mesh.points[:, :2], dtype=np.float64),
        np.concatenate(triangles, axis=0).astype(np.int64, copy=False),
    )


def align_mesh_to_solution(
    mesh_points: np.ndarray,
    triangles: np.ndarray,
    solution_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map Gmsh point indices to the node ordering stored by dolfinx in ``sol``."""
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    solution_points = np.asarray(solution_points, dtype=np.float64)
    if mesh_points.shape[0] != solution_points.shape[0]:
        raise ValueError(
            f"Mesh has {mesh_points.shape[0]} nodes but solution has "
            f"{solution_points.shape[0]} nodes"
        )
    distances, solution_index = cKDTree(solution_points).query(mesh_points, k=1)
    scale = max(float(np.ptp(solution_points, axis=0).max()), 1.0)
    # Generated pickles may have been stored as float32, while Gmsh stores
    # decimal coordinates at higher precision.
    tolerance = 1.0e-6 * scale
    if float(np.max(distances)) > tolerance:
        raise ValueError(
            f"Could not align mesh and solution nodes; maximum distance "
            f"{float(np.max(distances)):.3e} exceeds {tolerance:.3e}"
        )
    if np.unique(solution_index).size != solution_points.shape[0]:
        raise ValueError("Mesh-to-solution coordinate mapping is not one-to-one")
    aligned_triangles = solution_index[np.asarray(triangles, dtype=np.int64)]
    return solution_points.astype(np.float32), aligned_triangles.astype(np.int64)


def mesh_edges(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _gpmetis_labels(
    adjacency: Sequence[Mapping[int, int]],
    n_parts: int,
    seed: int,
    executable: PathLike,
    ufactor: int,
    contiguous: bool,
) -> np.ndarray:
    executable = Path(executable)
    with tempfile.TemporaryDirectory(prefix="sni-eval-metis-") as tmp:
        graph_path = Path(tmp) / "shape.graph"
        n_edges = sum(len(row) for row in adjacency) // 2
        with graph_path.open("w", encoding="utf-8") as stream:
            stream.write(f"{len(adjacency)} {n_edges} 1\n")
            for neighbors in adjacency:
                fields = []
                for neighbor, weight in sorted(neighbors.items()):
                    fields.extend((str(neighbor + 1), str(weight)))
                stream.write(" ".join(fields) + "\n")
        command = [
            str(executable),
            f"-ufactor={int(ufactor)}",
            "-ncuts=1",
            "-niter=10",
            f"-seed={int(seed)}",
            "-minconn",
        ]
        if contiguous:
            command.append("-contig")
        command.extend((str(graph_path), str(n_parts)))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"gpmetis failed with exit code {result.returncode}: {detail}")
        output = Path(f"{graph_path}.part.{n_parts}")
        if not output.is_file():
            raise RuntimeError(f"gpmetis did not create {output}")
        return np.loadtxt(output, dtype=np.int64, ndmin=1)


def metis_partition(
    n_nodes: int,
    edges: np.ndarray,
    n_parts: int,
    seed: int,
    gpmetis_executable: Optional[PathLike] = None,
    ufactor: int = 30,
    contiguous: bool = True,
    edge_weight_jitter: float = 0.10,
) -> np.ndarray:
    """Create a reproducible, balanced randomized METIS vertex partition.

    ``ufactor`` is METIS's permitted load imbalance in units of 1/1000, so the
    default 30 requests at most approximately 3% imbalance. Mild edge-weight
    jitter makes realizations differ without changing vertex-balance weights.
    """
    if n_parts < 2 or n_parts > n_nodes:
        raise ValueError(f"n_parts must be in [2, {n_nodes}], got {n_parts}")
    if int(ufactor) < 1:
        raise ValueError("ufactor must be positive")
    if not 0.0 <= float(edge_weight_jitter) <= 1.0:
        raise ValueError("edge_weight_jitter must be in [0, 1]")
    rng = np.random.default_rng(seed)
    adjacency: List[Dict[int, int]] = [dict() for _ in range(n_nodes)]
    weight_low = max(1, int(round(1000 * (1.0 - float(edge_weight_jitter)))))
    weight_high = max(weight_low + 1, int(round(1000 * (1.0 + float(edge_weight_jitter)))) + 1)
    for left, right in np.asarray(edges, dtype=np.int64):
        weight = 1000 if edge_weight_jitter == 0.0 else int(rng.integers(weight_low, weight_high))
        adjacency[int(left)][int(right)] = weight
        adjacency[int(right)][int(left)] = weight

    try:
        import pymetis  # type: ignore
    except ImportError:
        executable = gpmetis_executable or os.environ.get("GPMETIS_EXECUTABLE")
        if executable is None:
            executable = shutil.which("gpmetis")
        if executable is None:
            raise ImportError(
                "METIS requires PyMetis or gpmetis. Pass --gpmetis-executable "
                "or set GPMETIS_EXECUTABLE."
            ) from None
        labels = _gpmetis_labels(
            adjacency, n_parts, seed, executable, int(ufactor), bool(contiguous)
        )
    else:
        xadj = [0]
        adjncy: List[int] = []
        eweights: List[int] = []
        for neighbors in adjacency:
            for neighbor, weight in sorted(neighbors.items()):
                adjncy.append(neighbor)
                eweights.append(weight)
            xadj.append(len(adjncy))
        options = pymetis.Options(
            ufactor=int(ufactor),
            ncuts=1,
            niter=10,
            seed=int(seed),
            minconn=1,
            contig=int(bool(contiguous)),
        )
        _, labels = pymetis.part_graph(
            n_parts,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
            options=options,
        )

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unique = np.unique(labels)
    if labels.size != n_nodes or unique.size != n_parts:
        raise RuntimeError(
            f"METIS returned labels {unique.tolist()} for {labels.size}/{n_nodes} nodes"
        )
    counts = np.bincount(labels, minlength=n_parts)
    maximum_allowed = int(
        np.ceil((1.0 + float(ufactor) / 1000.0) * float(n_nodes) / float(n_parts))
    )
    if int(counts.max()) > maximum_allowed:
        raise RuntimeError(
            f"METIS violated the requested balance: counts={counts.tolist()}, "
            f"maximum allowed={maximum_allowed} for ufactor={ufactor}"
        )
    return labels


def infer_schema(source_samples: Sequence[tuple]) -> Dict[str, object]:
    """Infer one fixed branch/output layout for a source pickle."""
    sol, parameters, inputs = unpack_sni_sample(source_samples[0])
    n_roles = len(inputs)
    output_dim = sol.shape[1] - 2
    payload_dim = max(output_dim, *(value.shape[1] - 2 for value in inputs))
    for sample in source_samples[1:]:
        other_sol, other_parameters, other_inputs = unpack_sni_sample(sample)
        signature = (
            other_sol.shape[1] - 2,
            len(other_parameters),
            len(other_inputs),
            max(other_sol.shape[1] - 2, *(value.shape[1] - 2 for value in other_inputs)),
        )
        expected = (output_dim, len(parameters), n_roles, payload_dim)
        if signature != expected:
            raise ValueError(f"Inconsistent SNI sample schema: {signature} != {expected}")

    channel_names = ["x_local", "y_local"]
    channel_names += [f"input_{index}_mask" for index in range(n_roles)]
    channel_names += ["interface_mask"]
    channel_names += [f"payload_{index}" for index in range(payload_dim)]
    channel_names += [f"payload_{index}_valid" for index in range(payload_dim)]
    channel_names += [f"parameter_{index}" for index in range(len(parameters))]
    channel_names += ["local_aspect_ratio"]
    return {
        "n_input_roles": n_roles,
        "n_parameters": len(parameters),
        "payload_dim": payload_dim,
        "output_dim": output_dim,
        "branch_channel_names": channel_names,
        "trunk_channel_names": ["x_local", "y_local"],
        "output_channel_names": [f"solution_{index}" for index in range(output_dim)],
    }


def _map_function_nodes(function_xy: np.ndarray, solution_xy: np.ndarray) -> np.ndarray:
    distances, indices = cKDTree(np.asarray(solution_xy, dtype=np.float64)).query(
        np.asarray(function_xy, dtype=np.float64), k=1
    )
    scale = max(float(np.ptp(solution_xy, axis=0).max()), 1.0)
    if float(np.max(distances, initial=0.0)) > 1.0e-7 * scale:
        raise ValueError("An input-function point does not coincide with a solution node")
    return indices.astype(np.int64)


def build_partition_sample(
    source_sample: tuple,
    labels: np.ndarray,
    edges: np.ndarray,
    part_id: int,
    schema: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Convert one SNI solution and one METIS part into a DeepONet sample."""
    sol, parameters, input_functions = unpack_sni_sample(source_sample)
    solution_xy = sol[:, :2].astype(np.float64)
    target_all = sol[:, 2:].astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    part_nodes = np.flatnonzero(labels == int(part_id))
    if part_nodes.size == 0:
        raise ValueError(f"Partition {part_id} is empty")

    edges = np.asarray(edges, dtype=np.int64)
    cut = labels[edges[:, 0]] != labels[edges[:, 1]]
    cut_edges = edges[cut]
    touches = (labels[cut_edges[:, 0]] == part_id) | (labels[cut_edges[:, 1]] == part_id)
    local_cut_edges = cut_edges[touches]
    interface_xy = solution_xy[local_cut_edges].mean(axis=1)
    interface_values = target_all[local_cut_edges].mean(axis=1)

    local_inputs = []
    for role, function in enumerate(input_functions):
        function_nodes = _map_function_nodes(function[:, :2], solution_xy)
        keep = labels[function_nodes] == part_id
        if np.any(keep):
            local_inputs.append((role, function[keep]))

    geometry = [solution_xy[part_nodes]]
    geometry += [value[:, :2] for _, value in local_inputs]
    if interface_xy.size:
        geometry.append(interface_xy)
    geometry_points = np.concatenate(geometry, axis=0)
    origin = geometry_points.min(axis=0)
    span = np.maximum(geometry_points.max(axis=0) - origin, 1.0e-12)
    local_aspect = float(span[0] / span[1])

    n_roles = int(schema["n_input_roles"])
    payload_dim = int(schema["payload_dim"])
    n_parameters = int(schema["n_parameters"])
    branch_dim = len(schema["branch_channel_names"])

    def empty_features(n_rows: int, xy: np.ndarray) -> np.ndarray:
        features = np.zeros((n_rows, branch_dim), dtype=np.float32)
        features[:, :2] = ((np.asarray(xy) - origin) / span).astype(np.float32)
        if n_parameters:
            parameter_start = 3 + n_roles + 2 * payload_dim
            features[:, parameter_start : parameter_start + n_parameters] = np.asarray(
                parameters, dtype=np.float32
            )
        features[:, -1] = local_aspect
        return features

    branch_parts = []
    payload_start = 3 + n_roles
    valid_start = payload_start + payload_dim
    for role, function in local_inputs:
        features = empty_features(function.shape[0], function[:, :2])
        features[:, 2 + role] = 1.0
        payload = function[:, 2:]
        features[:, payload_start : payload_start + payload.shape[1]] = payload
        features[:, valid_start : valid_start + payload.shape[1]] = 1.0
        branch_parts.append(features)

    if interface_xy.size:
        features = empty_features(interface_xy.shape[0], interface_xy)
        features[:, 2 + n_roles] = 1.0
        features[:, payload_start : payload_start + target_all.shape[1]] = interface_values
        features[:, valid_start : valid_start + target_all.shape[1]] = 1.0
        branch_parts.append(features)
    if not branch_parts:
        raise ValueError(f"Partition {part_id} has no input or interface sensors")

    sample = {
        "branch": np.concatenate(branch_parts, axis=0).astype(np.float32),
        "query": ((solution_xy[part_nodes] - origin) / span).astype(np.float32),
        "target": target_all[part_nodes].astype(np.float32),
    }
    metadata = {
        "subdomain_id": int(part_id),
        "n_nodes": int(part_nodes.size),
        "n_interface_edges": int(local_cut_edges.shape[0]),
        "x_origin": float(origin[0]),
        "y_origin": float(origin[1]),
        "x_scale": float(span[0]),
        "y_scale": float(span[1]),
        "local_aspect_ratio": local_aspect,
    }
    return sample, metadata


def build_sni_eval_dataset(
    source_by_domain: Mapping[str, Sequence[tuple]],
    sample_indices_by_domain: Mapping[str, Sequence[int]],
    shape_data: Mapping[str, Mapping[str, object]],
    partitions: Mapping[str, Sequence[Mapping[str, object]]],
    pde_name: str,
) -> Dict[str, object]:
    """Build all subdomains for selected source samples and partition realizations."""
    all_source = [sample for values in source_by_domain.values() for sample in values]
    schema = infer_schema(all_source)
    samples = []
    metadata = []
    for domain, indices in sample_indices_by_domain.items():
        edges = np.asarray(shape_data[domain]["edges"], dtype=np.int64)
        for source_index in indices:
            for realization_id, partition in enumerate(partitions[domain]):
                labels = np.asarray(partition["labels"], dtype=np.int64)
                for part_id in range(int(partition["n_parts"])):
                    sample, row = build_partition_sample(
                        source_by_domain[domain][int(source_index)],
                        labels,
                        edges,
                        part_id,
                        schema,
                    )
                    row.update(
                        domain=domain,
                        source_sample_id=int(source_index),
                        realization_id=int(realization_id),
                        metis_seed=int(partition["seed"]),
                    )
                    samples.append(sample)
                    metadata.append(row)
    return {
        "samples": samples,
        "metadata": metadata,
        "shapes": shape_data,
        "partitions": partitions,
        "pde_name": pde_name,
        **schema,
    }


def save_sni_eval_dataset_h5(dataset: Mapping[str, object], path: PathLike) -> None:
    """Persist samples, original A/B/C shapes, and all METIS label arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "sni-eval-deeponet-v1"
        handle.attrs["pde_name"] = str(dataset["pde_name"])
        for key in ("branch_channel_names", "trunk_channel_names", "output_channel_names"):
            handle.attrs[key] = json.dumps(list(dataset[key]))
        handle.attrs["n_samples"] = len(dataset["samples"])

        samples_group = handle.create_group("samples")
        for index, (sample, metadata) in enumerate(
            zip(dataset["samples"], dataset["metadata"])
        ):
            group = samples_group.create_group(str(index))
            for key in ("branch", "query", "target"):
                group.create_dataset(
                    key,
                    data=np.asarray(sample[key], dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
            group.attrs["metadata"] = json.dumps(metadata)

        shapes_group = handle.create_group("shapes")
        for domain, shape in dataset["shapes"].items():
            group = shapes_group.create_group(domain)
            group.create_dataset("points", data=np.asarray(shape["points"], dtype=np.float32))
            group.create_dataset("triangles", data=np.asarray(shape["triangles"], dtype=np.int64))
            group.create_dataset("edges", data=np.asarray(shape["edges"], dtype=np.int64))
            realization_group = group.create_group("partitions")
            for realization_id, partition in enumerate(dataset["partitions"][domain]):
                part = realization_group.create_group(str(realization_id))
                part.create_dataset("labels", data=np.asarray(partition["labels"], dtype=np.int32))
                part.attrs["seed"] = int(partition["seed"])
                part.attrs["n_parts"] = int(partition["n_parts"])


def load_sni_eval_dataset_h5(path: PathLike) -> Dict[str, object]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("format", "")) != "sni-eval-deeponet-v1":
            raise ValueError(f"Not an SNI evaluation DeepONet dataset: {path}")
        samples = []
        metadata = []
        for index in range(int(handle.attrs["n_samples"])):
            group = handle["samples"][str(index)]
            samples.append({key: group[key][:].astype(np.float32) for key in ("branch", "query", "target")})
            metadata.append(json.loads(str(group.attrs["metadata"])))
        shapes = {}
        partitions = {}
        for domain, group in handle["shapes"].items():
            shapes[domain] = {
                "points": group["points"][:],
                "triangles": group["triangles"][:],
                "edges": group["edges"][:],
            }
            partitions[domain] = []
            part_group = group["partitions"]
            for key in sorted(part_group, key=int):
                part = part_group[key]
                partitions[domain].append(
                    {
                        "labels": part["labels"][:],
                        "seed": int(part.attrs["seed"]),
                        "n_parts": int(part.attrs["n_parts"]),
                    }
                )
        return {
            "samples": samples,
            "metadata": metadata,
            "shapes": shapes,
            "partitions": partitions,
            "pde_name": str(handle.attrs["pde_name"]),
            "branch_channel_names": json.loads(str(handle.attrs["branch_channel_names"])),
            "trunk_channel_names": json.loads(str(handle.attrs["trunk_channel_names"])),
            "output_channel_names": json.loads(str(handle.attrs["output_channel_names"])),
        }


class SNIEvalSubdomainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, np.ndarray]],
        branch_normalizer=None,
        target_normalizer=None,
        n_query_points: Optional[int] = None,
        random_query: bool = True,
    ):
        self.samples = list(samples)
        self.branch_normalizer = branch_normalizer
        self.target_normalizer = target_normalizer
        self.n_query_points = n_query_points
        self.random_query = bool(random_query)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[int(index)]
        branch = np.asarray(sample["branch"], dtype=np.float32)
        query = np.asarray(sample["query"], dtype=np.float32)
        target = np.asarray(sample["target"], dtype=np.float32)
        if self.n_query_points is not None and query.shape[0] > self.n_query_points:
            if self.random_query:
                selection = np.random.choice(query.shape[0], self.n_query_points, replace=False)
            else:
                selection = np.linspace(0, query.shape[0] - 1, self.n_query_points).astype(np.int64)
            query = query[selection]
            target = target[selection]
        if self.branch_normalizer is not None:
            branch = self.branch_normalizer.encode(branch).cpu().numpy().astype(np.float32)
        if self.target_normalizer is not None:
            target = self.target_normalizer.encode(target).cpu().numpy().astype(np.float32)
        return (
            torch.from_numpy(branch),
            torch.from_numpy(query),
            torch.from_numpy(target),
            torch.tensor(index, dtype=torch.long),
        )


def sni_eval_collate_fn(batch):
    branches, queries, targets, sample_indices = zip(*batch)
    batch_size = len(branches)
    max_branch = max(value.shape[0] for value in branches)
    branch_dim = branches[0].shape[1]
    branch = branches[0].new_zeros((batch_size, max_branch, branch_dim))
    branch_mask = torch.zeros((batch_size, max_branch), dtype=torch.bool)
    for index, value in enumerate(branches):
        branch[index, : value.shape[0]] = value
        branch_mask[index, : value.shape[0]] = True
    query_batch_id = torch.cat(
        [torch.full((value.shape[0],), index, dtype=torch.long) for index, value in enumerate(queries)]
    )
    return (
        branch,
        torch.cat(queries, dim=0),
        torch.cat(targets, dim=0),
        query_batch_id,
        torch.stack(sample_indices),
        branch_mask,
    )
