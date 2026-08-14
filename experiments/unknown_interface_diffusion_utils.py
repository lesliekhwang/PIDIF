from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from pidiffusion.data import FeatureNormalizer
from pidiffusion.diffusion import (
    ddim_step,
    final_clean_projection,
)
from experiments.evaluate_diffusion_generation import (
    require_checkpoint_field,
)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_target_normalizer(checkpoint):
    for key in (
        "target_normalizer",
        "normalizer",
        "y_normalizer",
        "output_normalizer",
    ):
        if key in checkpoint:
            print(f"Normalizer key       : {key}")
            return FeatureNormalizer.from_state_dict(
                checkpoint[key]
            )

    raise KeyError(
        f"No output normalizer found in checkpoint. "
        f"Keys: {sorted(checkpoint.keys())}"
    )


def build_schedule(checkpoint, device):
    config = dict(
        require_checkpoint_field(
            checkpoint,
            "diffusion_config",
        )
    )

    total_steps = int(config["T"])

    betas = torch.linspace(
        float(config["beta_start"]),
        float(config["beta_end"]),
        total_steps,
        dtype=torch.float32,
        device=device,
    )

    alphas_cumprod = torch.cumprod(
        1.0 - betas,
        dim=0,
    )

    return SimpleNamespace(
        alphas_cumprod=alphas_cumprod,
    )


def load_case_realization(
    dataset_path,
    case_id,
    realization_id,
):
    records = []

    with h5py.File(dataset_path, "r") as handle:
        n_samples = len(handle["samples"])

        for sample_idx in range(n_samples):
            this_case = decode_scalar(
                handle["metadata"]["case_id"][sample_idx]
            )

            this_realization = int(
                decode_scalar(
                    handle["metadata"]["realization_id"][sample_idx]
                )
            )

            if (
                this_case != case_id
                or this_realization != realization_id
            ):
                continue

            subdomain_id = int(
                decode_scalar(
                    handle["metadata"]["subdomain_id"][sample_idx]
                )
            )

            group = handle["samples"][str(sample_idx)]

            records.append(
                {
                    "sample_index": sample_idx,
                    "subdomain_id": subdomain_id,
                    "branch": (
                        group["branch"][:]
                        .astype(np.float32)
                    ),
                    "query": (
                        group["query"][:]
                        .astype(np.float32)
                    ),
                }
            )

    records.sort(
        key=lambda item: item["subdomain_id"]
    )

    if len(records) < 2:
        raise RuntimeError(
            f"Expected at least 2 subdomains, "
            f"found {len(records)}"
        )

    ids = [
        int(item["subdomain_id"])
        for item in records
    ]

    expected_ids = list(range(len(records)))

    if ids != expected_ids:
        raise RuntimeError(
            f"Unexpected subdomain IDs: {ids}; "
            f"expected {expected_ids}"
        )

    return records


def find_edge_rows(branches, names):
    ix = names.index("x_local")
    iy = names.index("y_local")
    interface_idx = names.index("interface_mask")

    n_subdomains = int(branches.shape[0])

    if n_subdomains < 2:
        raise RuntimeError(
            f"At least 2 subdomains are required, got {n_subdomains}"
        )

    left_rows = []
    right_rows = []
    n_interface_points = None

    for subdomain_id in range(n_subdomains):
        branch = branches[subdomain_id]

        edge_mask = branch[:, interface_idx] > 0.5

        left = np.flatnonzero(
            edge_mask
            & np.isclose(
                branch[:, ix],
                0.0,
                atol=1.0e-6,
            )
        )

        right = np.flatnonzero(
            edge_mask
            & np.isclose(
                branch[:, ix],
                1.0,
                atol=1.0e-6,
            )
        )

        left = left[
            np.argsort(branch[left, iy])
        ]
        right = right[
            np.argsort(branch[right, iy])
        ]

        if len(left) != len(right):
            raise RuntimeError(
                f"Subdomain {subdomain_id}: "
                f"left={len(left)}, right={len(right)}"
            )

        if n_interface_points is None:
            n_interface_points = int(len(left))

            if n_interface_points <= 0:
                raise RuntimeError(
                    "No interface points were found"
                )
        else:
            if (
                len(left) != n_interface_points
                or len(right) != n_interface_points
            ):
                raise RuntimeError(
                    f"Subdomain {subdomain_id}: "
                    f"expected {n_interface_points} interface points, "
                    f"got left={len(left)}, right={len(right)}"
                )

        left_rows.append(left)
        right_rows.append(right)

    for interface_id in range(n_subdomains - 1):
        y_right = branches[
            interface_id,
            right_rows[interface_id],
            iy,
        ]

        y_left = branches[
            interface_id + 1,
            left_rows[interface_id + 1],
            iy,
        ]

        if not np.array_equal(y_right, y_left):
            raise RuntimeError(
                f"Interface {interface_id} edge coordinates "
                "do not match exactly"
            )

    return left_rows, right_rows


def predict_epsilon_batched(
    model,
    branch,
    query,
    noisy_target,
    timestep,
    query_batch_id,
):
    return model(
        branch=branch,
        query=query,
        noisy_target=noisy_target,
        t_query=torch.full(
            (len(query),),
            int(timestep),
            dtype=torch.long,
            device=query.device,
        ),
        query_batch_id=query_batch_id,
        branch_mask=None,
    )


def sample_ddim_differentiable_batched(
    model,
    branch,
    query,
    initial_noise,
    query_batch_id,
    schedule,
    timesteps,
):
    x_t = initial_noise.clone()

    for current, next_ in zip(
        timesteps[:-1],
        timesteps[1:],
    ):
        t_current = int(current.item())
        t_next = int(next_.item())

        epsilon = predict_epsilon_batched(
            model=model,
            branch=branch,
            query=query,
            noisy_target=x_t,
            timestep=t_current,
            query_batch_id=query_batch_id,
        )

        x_t, _ = ddim_step(
            x_t=x_t,
            epsilon_pred=epsilon,
            t_current=t_current,
            t_next=t_next,
            alphas_cumprod=(
                schedule.alphas_cumprod
            ),
        )

    final_t = int(
        timesteps[-1].item()
    )

    epsilon = predict_epsilon_batched(
        model=model,
        branch=branch,
        query=query,
        noisy_target=x_t,
        timestep=final_t,
        query_batch_id=query_batch_id,
    )

    return final_clean_projection(
        x_t=x_t,
        epsilon_pred=epsilon,
        timestep=final_t,
        alphas_cumprod=(
            schedule.alphas_cumprod
        ),
    )


def write_shared_z(
    base_branch,
    z,
    left_rows,
    right_rows,
    names,
):
    branch = base_branch.clone()

    n_subdomains = int(branch.shape[0])
    n_internal_interfaces = n_subdomains - 1

    if n_subdomains < 2:
        raise ValueError(
            f"At least 2 subdomains are required, got {n_subdomains}"
        )

    if (
        len(left_rows) != n_subdomains
        or len(right_rows) != n_subdomains
    ):
        raise ValueError(
            "left_rows/right_rows do not match "
            f"n_subdomains={n_subdomains}"
        )

    n_interface_points = int(
        len(right_rows[0])
    )

    expected_shape = (
        n_internal_interfaces,
        n_interface_points,
        3,
    )

    if tuple(z.shape) != expected_shape:
        raise ValueError(
            f"Expected z shape {expected_shape}, "
            f"got {tuple(z.shape)}"
        )

    value_indices = torch.tensor(
        [
            names.index("boundary_pressure"),
            names.index("boundary_u"),
            names.index("boundary_v"),
        ],
        dtype=torch.long,
        device=branch.device,
    )

    known_indices = torch.tensor(
        [
            names.index("known_pressure"),
            names.index("known_u"),
            names.index("known_v"),
        ],
        dtype=torch.long,
        device=branch.device,
    )

    for interface_id in range(
        n_internal_interfaces
    ):
        upstream = interface_id
        downstream = interface_id + 1

        right = torch.as_tensor(
            right_rows[upstream],
            dtype=torch.long,
            device=branch.device,
        )

        left = torch.as_tensor(
            left_rows[downstream],
            dtype=torch.long,
            device=branch.device,
        )

        branch[
            upstream,
            right[:, None],
            value_indices[None, :],
        ] = z[interface_id]

        branch[
            downstream,
            left[:, None],
            value_indices[None, :],
        ] = z[interface_id]

        branch[
            upstream,
            right[:, None],
            known_indices[None, :],
        ] = 1.0

        branch[
            downstream,
            left[:, None],
            known_indices[None, :],
        ] = 1.0

    return branch


def build_edge_queries(
    branches_np,
    left_rows,
    right_rows,
    names,
    device,
):
    ix = names.index("x_local")
    iy = names.index("y_local")

    n_subdomains = int(
        branches_np.shape[0]
    )

    left_np = np.stack(
        [
            branches_np[
                subdomain_id,
                left_rows[subdomain_id],
            ][:, [ix, iy]]
            for subdomain_id in range(
                n_subdomains
            )
        ],
        axis=0,
    ).astype(np.float32)

    right_np = np.stack(
        [
            branches_np[
                subdomain_id,
                right_rows[subdomain_id],
            ][:, [ix, iy]]
            for subdomain_id in range(
                n_subdomains
            )
        ],
        axis=0,
    ).astype(np.float32)

    q_left = (
        torch.from_numpy(left_np)
        .to(device)
        .requires_grad_(True)
    )

    q_right = (
        torch.from_numpy(right_np)
        .to(device)
        .requires_grad_(True)
    )

    return q_left, q_right


def load_metadata_for_records(
    dataset_path: Path,
    records,
):
    metadata = []

    with h5py.File(
        dataset_path,
        "r",
    ) as handle:
        metadata_group = handle["metadata"]

        for record in records:
            sample_index = int(
                record["sample_index"]
            )

            metadata.append(
                {
                    key: decode_scalar(
                        metadata_group[key][
                            sample_index
                        ]
                    )
                    for key
                    in metadata_group.keys()
                }
            )

    return metadata
