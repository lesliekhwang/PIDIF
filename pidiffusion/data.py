"""Data loading, case splitting, normalization, and batching for baseline diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import h5py
import numpy as np
import torch


class FeatureNormalizer:
    """Channel-wise normalizer matching the historical notebook behavior."""

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        eps: float = 1.0e-6,
        skip_indices: Optional[Sequence[int]] = None,
    ):
        xt = torch.as_tensor(x, dtype=torch.float32)
        if xt.ndim < 2:
            raise ValueError(
                f"Expected at least 2 dimensions, got shape {tuple(xt.shape)}"
            )

        channels = int(xt.shape[-1])
        flat = xt.reshape(-1, channels)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(float(eps))

        if skip_indices is not None:
            for index in skip_indices:
                mean[int(index)] = 0.0
                std[int(index)] = 1.0

        self.mean = mean
        self.std = std
        self.eps = float(eps)

    def _shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple([1] * (x.ndim - 1) + [self.mean.numel()])

    def encode(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
        return (xt - self.mean.reshape(self._shape(xt))) / self.std.reshape(
            self._shape(xt)
        )

    def decode(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
        return xt * self.std.reshape(self._shape(xt)) + self.mean.reshape(
            self._shape(xt)
        )

    def to(self, device: Union[str, torch.device]) -> "FeatureNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": torch.tensor(float(self.eps)),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        obj = cls.__new__(cls)
        obj.mean = torch.as_tensor(state["mean"], dtype=torch.float32)
        obj.std = torch.as_tensor(state["std"], dtype=torch.float32)
        eps = state.get("eps", torch.tensor(1.0e-6))
        obj.eps = float(eps.item() if torch.is_tensor(eps) else eps)
        return obj


@dataclass(frozen=True)
class DiffusionCaseSplit:
    """Case-level split and the corresponding sample indices."""

    train_cases: tuple[str, ...]
    val_cases: tuple[str, ...]
    test_cases: tuple[str, ...]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


_REQUIRED_DATA_KEYS = {
    "samples",
    "metadata",
    "branch_channel_names",
    "trunk_channel_names",
    "output_channel_names",
    "n_interface_points",
    "n_boundary_points",
    "include_interface_endpoints",
    "horizontal_interface",
    "horizontal_interface_jitter",
}
_REQUIRED_METADATA_KEYS = {
    "case_id",
    "local_aspect_ratio",
    "subdomain_id",
    "realization_id",
}
_EXPECTED_BRANCH_CHANNEL_NAMES = (
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
_EXPECTED_TRUNK_CHANNEL_NAMES = ("x_local", "y_local")
_EXPECTED_OUTPUT_CHANNEL_NAMES = ("pressure", "u", "v")


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def _decode_attr_text(value: Any) -> str:
    value = _decode_scalar(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected a text attribute, got {type(value).__name__}")
    return value


def validate_diffusion_dataset(data: Mapping[str, Any]) -> None:
    """Validate the minimum schema required by the diffusion data path."""

    if not isinstance(data, Mapping):
        raise TypeError("Dataset must be a mapping")

    missing = sorted(_REQUIRED_DATA_KEYS.difference(data))
    if missing:
        raise ValueError("Dataset is missing required fields: " + ", ".join(missing))

    branch_names = list(data["branch_channel_names"])
    trunk_names = list(data["trunk_channel_names"])
    output_names = list(data["output_channel_names"])
    if not branch_names or not trunk_names or not output_names:
        raise ValueError("Channel name lists must not be empty")
    if any(not isinstance(name, str) or not name for name in branch_names + trunk_names + output_names):
        raise ValueError("Channel names must be non-empty strings")
    if tuple(branch_names) != _EXPECTED_BRANCH_CHANNEL_NAMES:
        raise ValueError(
            "Branch channel names do not match the canonical baseline order"
        )
    if tuple(trunk_names) != _EXPECTED_TRUNK_CHANNEL_NAMES:
        raise ValueError(
            "Trunk channel names do not match the canonical baseline order"
        )
    if tuple(output_names) != _EXPECTED_OUTPUT_CHANNEL_NAMES:
        raise ValueError(
            "Output channel names must be exactly pressure, u, v in that order"
        )

    samples = list(data["samples"])
    metadata = list(data["metadata"])
    if len(samples) != len(metadata):
        raise ValueError(
            f"Sample and metadata counts differ: {len(samples)} != {len(metadata)}"
        )

    for sample_index, (sample, meta) in enumerate(zip(samples, metadata)):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Sample {sample_index} must be a mapping")
        for field in ("branch", "query", "target"):
            if field not in sample:
                raise ValueError(f"Sample {sample_index} is missing {field!r}")
            array = np.asarray(sample[field])
            if array.ndim != 2:
                raise ValueError(
                    f"Sample {sample_index} field {field!r} must be 2D, got {array.ndim}D"
                )
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError(
                    f"Sample {sample_index} field {field!r} must have a floating dtype"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Sample {sample_index} field {field!r} contains non-finite values"
                )

        branch = np.asarray(sample["branch"])
        query = np.asarray(sample["query"])
        target = np.asarray(sample["target"])
        if query.shape[0] != target.shape[0]:
            raise ValueError(
                f"Sample {sample_index} query and target row counts differ: "
                f"{query.shape[0]} != {target.shape[0]}"
            )
        if branch.shape[1] != len(branch_names):
            raise ValueError(
                f"Sample {sample_index} branch width {branch.shape[1]} does not "
                f"match {len(branch_names)} channel names"
            )
        if query.shape[1] != len(trunk_names):
            raise ValueError(
                f"Sample {sample_index} query width {query.shape[1]} does not "
                f"match {len(trunk_names)} channel names"
            )
        if target.shape[1] != len(output_names):
            raise ValueError(
                f"Sample {sample_index} target width {target.shape[1]} does not "
                f"match {len(output_names)} channel names"
            )
        if not isinstance(meta, Mapping):
            raise ValueError(f"Metadata row {sample_index} must be a mapping")
        missing_meta = sorted(_REQUIRED_METADATA_KEYS.difference(meta))
        if missing_meta:
            raise ValueError(
                f"Metadata row {sample_index} is missing: " + ", ".join(missing_meta)
            )


def load_diffusion_dataset(path: Union[str, Path]) -> dict[str, Any]:
    """Load the existing variable-length HDF5 dataset into memory."""

    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise IsADirectoryError(f"Dataset path is not a regular file: {dataset_path}")

    with h5py.File(dataset_path, "r") as handle:
        n_samples = int(handle.attrs["n_samples"])
        samples: list[dict[str, np.ndarray]] = []
        for index in range(n_samples):
            sample_group = handle["samples"][str(index)]
            samples.append(
                {
                    "branch": sample_group["branch"][:].astype(np.float32),
                    "query": sample_group["query"][:].astype(np.float32),
                    "target": sample_group["target"][:].astype(np.float32),
                }
            )

        metadata: list[dict[str, Any]] = []
        metadata_group = handle["metadata"]
        metadata_keys = list(metadata_group.keys())
        for index in range(n_samples):
            row: dict[str, Any] = {}
            for key in metadata_keys:
                row[key] = _decode_scalar(metadata_group[key][index])
            metadata.append(row)

        data: dict[str, Any] = {
            "samples": samples,
            "metadata": metadata,
            "branch_channel_names": _decode_attr_text(
                handle.attrs["branch_channel_names"]
            ).split("\n"),
            "trunk_channel_names": _decode_attr_text(
                handle.attrs["trunk_channel_names"]
            ).split("\n"),
            "output_channel_names": _decode_attr_text(
                handle.attrs["output_channel_names"]
            ).split("\n"),
            "n_interface_points": int(handle.attrs["n_interface_points"]),
            "n_boundary_points": int(handle.attrs["n_boundary_points"]),
            "include_interface_endpoints": bool(
                handle.attrs.get("include_interface_endpoints", False)
            ),
            "horizontal_interface": bool(
                handle.attrs.get("horizontal_interface", False)
            ),
            "horizontal_interface_jitter": float(
                handle.attrs.get("horizontal_interface_jitter", 0.0)
            ),
        }

    validate_diffusion_dataset(data)
    return data


def build_case_split(
    data: Mapping[str, Any],
    split_seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> DiffusionCaseSplit:
    """Build the historical sorted case-level 80/10/10 split."""

    validate_diffusion_dataset(data)
    if train_fraction < 0.0 or val_fraction < 0.0:
        raise ValueError("Split fractions must be non-negative")
    if train_fraction + val_fraction > 1.0:
        raise ValueError("Train and validation fractions must sum to at most one")

    all_case_ids = sorted({str(meta["case_id"]) for meta in data["metadata"]})
    if not all_case_ids:
        raise ValueError("Dataset contains no case IDs")

    shuffled_cases = np.asarray(all_case_ids, dtype=object)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(shuffled_cases)

    n_cases = len(shuffled_cases)
    n_train = int(train_fraction * n_cases)
    n_val = int(val_fraction * n_cases)
    train_cases_set = set(str(case_id) for case_id in shuffled_cases[:n_train])
    val_cases_set = set(
        str(case_id) for case_id in shuffled_cases[n_train : n_train + n_val]
    )
    test_cases_set = set(str(case_id) for case_id in shuffled_cases[n_train + n_val :])

    train_indices = tuple(
        index
        for index, meta in enumerate(data["metadata"])
        if str(meta["case_id"]) in train_cases_set
    )
    val_indices = tuple(
        index
        for index, meta in enumerate(data["metadata"])
        if str(meta["case_id"]) in val_cases_set
    )
    test_indices = tuple(
        index
        for index, meta in enumerate(data["metadata"])
        if str(meta["case_id"]) in test_cases_set
    )

    return DiffusionCaseSplit(
        train_cases=tuple(sorted(train_cases_set)),
        val_cases=tuple(sorted(val_cases_set)),
        test_cases=tuple(sorted(test_cases_set)),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def fit_train_normalizers(
    data: Mapping[str, Any],
    split: DiffusionCaseSplit,
) -> tuple[FeatureNormalizer, float, float]:
    """Fit target and local-aspect normalizers using training samples only."""

    train_targets = np.concatenate(
        [data["samples"][index]["target"] for index in split.train_indices],
        axis=0,
    ).astype(np.float32)
    target_normalizer = FeatureNormalizer(train_targets)

    train_aspect = np.asarray(
        [data["metadata"][index]["local_aspect_ratio"] for index in split.train_indices],
        dtype=np.float32,
    )
    local_aspect_mean = float(train_aspect.mean())
    local_aspect_std = float(train_aspect.std() + 1.0e-12)
    return target_normalizer, local_aspect_mean, local_aspect_std


def normalize_diffusion_branch(
    branch: np.ndarray,
    branch_channel_names: Sequence[str],
    target_normalizer: Optional[FeatureNormalizer] = None,
    local_aspect_mean: Optional[float] = None,
    local_aspect_std: Optional[float] = None,
    zero_unknown_values: bool = True,
) -> np.ndarray:
    """Normalize branch values, known flags, coordinates, and local aspect."""

    names = list(branch_channel_names)
    output_fields = [
        name[len("boundary_") :]
        for name in names
        if name.startswith("boundary_")
    ]
    if not output_fields:
        raise ValueError("Branch channels contain no boundary fields")
    value_indices = [names.index(f"boundary_{field}") for field in output_fields]
    output_count = len(output_fields)
    out = np.asarray(branch, dtype=np.float32).copy()

    if target_normalizer is not None:
        mean = target_normalizer.mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
        std = target_normalizer.std.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if mean.size != output_count or std.size != output_count:
            raise ValueError(
                "Target normalizer width does not match boundary field count"
            )
        std = np.maximum(std, 1.0e-12)
        normalized_values = (out[:, value_indices] - mean.reshape(1, output_count)) / std.reshape(
            1, output_count
        )

        known_names = [f"known_{field}" for field in output_fields]
        if zero_unknown_values and all(name in names for name in known_names):
            known_indices = [names.index(name) for name in known_names]
            normalized_values = normalized_values * out[:, known_indices]
        out[:, value_indices] = normalized_values.astype(np.float32)

    if local_aspect_mean is not None and local_aspect_std is not None:
        if "local_aspect_ratio" not in names:
            raise ValueError("Branch channels contain no local_aspect_ratio field")
        aspect_index = names.index("local_aspect_ratio")
        out[:, aspect_index] = (
            out[:, aspect_index] - float(local_aspect_mean)
        ) / max(float(local_aspect_std), 1.0e-12)

    return out.astype(np.float32, copy=False)


class DiffusionCellDataset(torch.utils.data.Dataset):
    """PyTorch dataset for variable-length diffusion cell samples."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, np.ndarray]],
        sample_indices: Optional[Sequence[int]] = None,
        n_query_points: Optional[int] = 8192,
        random_query: bool = True,
        target_normalizer: Optional[FeatureNormalizer] = None,
        local_aspect_mean: Optional[float] = None,
        local_aspect_std: Optional[float] = None,
        branch_channel_names: Sequence[str] = (),
    ):
        self.samples = list(samples)
        if sample_indices is None:
            self.indices = np.arange(len(self.samples), dtype=np.int64)
        else:
            self.indices = np.asarray(sample_indices, dtype=np.int64)
        self.n_query_points = n_query_points
        self.random_query = bool(random_query)
        self.target_normalizer = target_normalizer
        self.local_aspect_mean = local_aspect_mean
        self.local_aspect_std = local_aspect_std
        self.branch_channel_names = list(branch_channel_names)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _sample_query_indices(self, n: int) -> np.ndarray:
        n = int(n)
        if self.n_query_points is None or int(self.n_query_points) >= n:
            return np.arange(n, dtype=np.int64)
        query_count = int(self.n_query_points)
        if self.random_query:
            return np.random.choice(n, size=query_count, replace=False)
        return np.linspace(0, n - 1, query_count).astype(np.int64)

    def __getitem__(self, index: int):
        sample_index = int(self.indices[int(index)])
        sample = self.samples[sample_index]
        branch = np.asarray(sample["branch"], dtype=np.float32)
        query_all = np.asarray(sample["query"], dtype=np.float32)
        target_all = np.asarray(sample["target"], dtype=np.float32)

        point_indices = self._sample_query_indices(query_all.shape[0])
        query = query_all[point_indices]
        target = target_all[point_indices]

        branch = normalize_diffusion_branch(
            branch,
            branch_channel_names=self.branch_channel_names,
            target_normalizer=self.target_normalizer,
            local_aspect_mean=self.local_aspect_mean,
            local_aspect_std=self.local_aspect_std,
        )
        if self.target_normalizer is not None:
            target = (
                self.target_normalizer.encode(target)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return (
            torch.from_numpy(branch).float(),
            torch.from_numpy(query).float(),
            torch.from_numpy(target).float(),
            torch.tensor(sample_index, dtype=torch.long),
        )


def collate_diffusion_batch(batch, return_branch_mask: bool = True):
    """Collate variable-size branch and query point sets."""

    branches, queries, targets, sample_indices = zip(*batch)
    batch_size = len(branches)
    branch_dim = branches[0].shape[1]
    max_branch_points = max(branch.shape[0] for branch in branches)

    for index, branch in enumerate(branches):
        if branch.ndim != 2:
            raise ValueError(f"Branch {index} has {branch.ndim} dimensions, expected 2")
        if branch.shape[1] != branch_dim:
            raise ValueError(
                f"Branch {index} has {branch.shape[1]} columns, expected {branch_dim}"
            )

    branch = branches[0].new_zeros((batch_size, max_branch_points, branch_dim))
    branch_mask = torch.zeros(
        (batch_size, max_branch_points), dtype=torch.bool, device=branches[0].device
    )
    for index, item in enumerate(branches):
        point_count = item.shape[0]
        branch[index, :point_count, :] = item
        branch_mask[index, :point_count] = True

    query_cat = torch.cat(queries, dim=0)
    target_cat = torch.cat(targets, dim=0)
    query_batch_id = torch.cat(
        [
            torch.full(
                (query.shape[0],),
                index,
                dtype=torch.long,
                device=query.device,
            )
            for index, query in enumerate(queries)
        ]
    )
    sample_idx = torch.stack(sample_indices, dim=0)

    if return_branch_mask:
        return branch, query_cat, target_cat, query_batch_id, sample_idx, branch_mask
    return branch, query_cat, target_cat, query_batch_id, sample_idx
