"""
Fluent-HDF5 to FNO dataset builder with local, subdomain-independent inputs.

This version adds input_mode='local_full16' and 'local_left12', which removes x_global,
global aspect_ratio, and subdomain_id.  It treats each split subdomain as an
independent local boundary-value sample.

It should live next to fno_fluent_dataset.py and can be used with the existing
fno_model_flexible.py / train_fluent_fno.py scripts.

Main modes recommended for independent subdomain training:
    input_mode='local_full16_broadcast' for a two-sided boundary-value model
    input_mode='local_left12_broadcast' for a left-to-right marching model

Saved arrays are NHWC by default:
    inputs:  (N, nx, ny, C_in)
    outputs: (N, nx, ny, 4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import json

import h5py
import numpy as np

from fno_fluent_dataset import build_case_subdomains

PathLike = Union[str, Path]
CaseFiles = Mapping[int, Mapping[str, PathLike]]

COORD5_CHANNELS = [
    "x_local",
    "y_norm",
    "x_global",
    "aspect_ratio_scaled",
    "subdomain_id_scaled",
]

# local_aspect_ratio is local geometry, not case identity:
#     local_aspect_ratio = subdomain_width / subdomain_height.
LOCAL_FULL16_CHANNELS = [
    "fluid_mask",
    "left_interface_mask",
    "right_interface_mask",
    "wall_mask",
    "local_aspect_ratio",
    "left_pressure",
    "left_temperature",
    "left_u",
    "left_v",
    "right_pressure",
    "right_temperature",
    "right_u",
    "right_v",
    "wall_temperature",
    "wall_u",
    "wall_v",
]

# Left-to-right/marching variant. It omits right interface variables so the
# right interface can be part of the prediction target instead of an input.
LOCAL_LEFT12_CHANNELS = [
    "fluid_mask",
    "left_interface_mask",
    "right_interface_mask",
    "wall_mask",
    "local_aspect_ratio",
    "left_pressure",
    "left_temperature",
    "left_u",
    "left_v",
    "wall_temperature",
    "wall_u",
    "wall_v",
]

OUTPUT_CHANNELS = ["pressure", "temperature", "u", "v"]


def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _to_nhwc(a_nchw: np.ndarray) -> np.ndarray:
    if a_nchw.ndim != 4:
        raise ValueError(f"Expected a 4-D NCHW array, got shape {a_nchw.shape}")
    return np.moveaxis(a_nchw, 1, -1)


def _field_index(channel_names: Sequence[str]) -> Dict[str, int]:
    aliases = {
        "pressure": ["pressure", "p", "SV_P"],
        "temperature": ["temperature", "t", "T", "SV_T"],
        "u": ["u", "x_velocity", "x-velocity", "SV_U"],
        "v": ["v", "y_velocity", "y-velocity", "SV_V"],
    }
    lower_to_idx = {str(name).lower(): i for i, name in enumerate(channel_names)}
    out: Dict[str, int] = {}
    for clean, names in aliases.items():
        found = None
        for name in names:
            key = str(name).lower()
            if key in lower_to_idx:
                found = lower_to_idx[key]
                break
        if found is None:
            raise ValueError(
                f"Could not find {clean!r} in output channels {list(channel_names)}"
            )
        out[clean] = int(found)
    return out


def make_local_full16_inputs_for_case(
    case: Mapping[str, object],
    profile_placement: str = "boundary",
) -> np.ndarray:
    """
    Build local, subdomain-independent boundary/interface input channels.

    Parameters
    ----------
    case
        One case returned by fno_fluent_dataset.build_case_subdomains().
    profile_placement
        'boundary': put interface values only on left/right grid boundaries.
        'broadcast': repeat left/right profiles across the full subdomain grid.
            This does not add new information, but sometimes makes FNO training
            easier because the interface traces are not confined to one column.

    Returns
    -------
    X
        Shape (n_subdomains, 16, nx, ny), with channels LOCAL_FULL16_CHANNELS.

    Notes
    -----
    left_* and right_* are artificial interface traces from the CFD solution.
    They are valid inputs only if those traces will also be known or estimated
    at inference time.  This mode is intended for a local boundary-value FNO,
    not for a global surrogate that only knows physical inlet/outlet BCs.
    """
    profile_placement = profile_placement.lower()
    if profile_placement not in {"boundary", "broadcast"}:
        raise ValueError("profile_placement must be 'boundary' or 'broadcast'")

    Y = np.asarray(case["Y"], dtype=np.float32)
    interfaces = np.asarray(case["interfaces"], dtype=np.float32)
    channel_names = list(case["output_channel_names"])
    metadata = list(case["metadata"])
    idx = _field_index(channel_names)

    if Y.ndim != 4:
        raise ValueError(f"case['Y'] must have shape (S,C,nx,ny), got {Y.shape}")
    if interfaces.ndim != 3:
        raise ValueError(f"case['interfaces'] must have shape (S+1,C,ny), got {interfaces.shape}")

    n_sub, _, nx, ny = Y.shape
    if interfaces.shape[0] != n_sub + 1 or interfaces.shape[-1] != ny:
        raise ValueError(f"Interface shape mismatch: Y={Y.shape}, interfaces={interfaces.shape}")
    if len(metadata) != n_sub:
        raise ValueError(f"Metadata length {len(metadata)} does not match n_subdomains {n_sub}")

    X = np.zeros((n_sub, len(LOCAL_FULL16_CHANNELS), nx, ny), dtype=np.float32)

    # Masks.
    X[:, 0, :, :] = 1.0       # fluid mask
    X[:, 1, 0, 1:-1] = 1.0       # left interface/inlet side
    X[:, 2, -1, 1:-1] = 1.0      # right interface/outlet side
    X[:, 3, :, 0] = 1.0       # bottom wall
    X[:, 3, :, -1] = 1.0      # top wall

    p_i = idx["pressure"]
    t_i = idx["temperature"]
    u_i = idx["u"]
    v_i = idx["v"]

    for s in range(n_sub):
        m = metadata[s]
        width = float(m["x_right_mm"]) - float(m["x_left_mm"])
        height = float(m["y_top_mm"]) - float(m["y_bottom_mm"])
        local_aspect = width / max(height, 1.0e-12)
        X[s, 4, :, :] = local_aspect

        left = interfaces[s]
        right = interfaces[s + 1]

        left_profiles = {
            5: left[p_i],
            6: left[t_i],
            7: left[u_i],
            8: left[v_i],
        }
        right_profiles = {
            9: right[p_i],
            10: right[t_i],
            11: right[u_i],
            12: right[v_i],
        }

        if profile_placement == "boundary":
            for ch, prof in left_profiles.items():
                X[s, ch, 0, 1:-1] = prof[1:-1]
            for ch, prof in right_profiles.items():
                X[s, ch, -1, 1:-1] = prof[1:-1]
        else:
            # Repeat y-profiles across x.  No extra physical information is
            # added, but it is often easier for convolutional/FNO models to use.
            for ch, prof in left_profiles.items():
                X[s, ch, :, 1:-1] = prof[None, 1:-1]
            for ch, prof in right_profiles.items():
                X[s, ch, :, 1:-1] = prof[None, 1:-1]

        # Sampled near-wall values from the gridded CFD field. )
        X[s, 13, :, 0] = Y[s, t_i, :, 0]
        X[s, 13, :, -1] = Y[s, t_i, :, -1]
        X[s, 14, :, 0] = Y[s, u_i, :, 0]
        X[s, 14, :, -1] = Y[s, u_i, :, -1]
        X[s, 15, :, 0] = Y[s, v_i, :, 0]
        X[s, 15, :, -1] = Y[s, v_i, :, -1]


    return X


def make_local_left12_inputs_for_case(
    case: Mapping[str, object],
    profile_placement: str = "boundary",
) -> np.ndarray:
    """
    Build local left-to-right boundary/interface input channels.

    This mode removes right_pressure/right_temperature/right_u/right_v from the
    input. Use it when the right interface should be predicted, not supplied.
    """
    profile_placement = profile_placement.lower()
    if profile_placement not in {"boundary", "broadcast"}:
        raise ValueError("profile_placement must be 'boundary' or 'broadcast'")

    Y = np.asarray(case["Y"], dtype=np.float32)
    interfaces = np.asarray(case["interfaces"], dtype=np.float32)
    channel_names = list(case["output_channel_names"])
    metadata = list(case["metadata"])
    idx = _field_index(channel_names)

    n_sub, _, nx, ny = Y.shape
    if interfaces.shape[0] != n_sub + 1 or interfaces.shape[-1] != ny:
        raise ValueError(f"Interface shape mismatch: Y={Y.shape}, interfaces={interfaces.shape}")
    if len(metadata) != n_sub:
        raise ValueError(f"Metadata length {len(metadata)} does not match n_subdomains {n_sub}")

    X = np.zeros((n_sub, len(LOCAL_LEFT12_CHANNELS), nx, ny), dtype=np.float32)
    X[:, 0, :, :] = 1.0       # fluid mask
    X[:, 1, 0, 1:-1] = 1.0       # left interface/inlet side
    X[:, 2, -1, 1:-1] = 1.0      # right side location, but no right values are given
    X[:, 3, :, 0] = 1.0       # bottom wall
    X[:, 3, :, -1] = 1.0      # top wall

    p_i = idx["pressure"]
    t_i = idx["temperature"]
    u_i = idx["u"]
    v_i = idx["v"]

    for s in range(n_sub):
        m = metadata[s]
        width = float(m["x_right_mm"]) - float(m["x_left_mm"])
        height = float(m["y_top_mm"]) - float(m["y_bottom_mm"])
        X[s, 4, :, :] = width / max(height, 1.0e-12)

        left = interfaces[s]
        left_profiles = {
            5: left[p_i],
            6: left[t_i],
            7: left[u_i],
            8: left[v_i],
        }
        if profile_placement == "boundary":
            for ch, prof in left_profiles.items():
                X[s, ch, 0, 1:-1] = prof[1:-1]
        else:
            for ch, prof in left_profiles.items():
                X[s, ch, :, 1:-1] = prof[None, 1:-1]

        X[s, 9, :, 0] = np.ones_like(X[s, 9, :, 0]) * 275
        X[s, 9, :, -1] = np.ones_like(X[s, 9, :, -1]) * 273
        X[s, 10, :, 0] = np.zeros_like(X[s, 10, :, 0])
        X[s, 10, :, -1] = np.zeros_like(X[s, 10, :, -1])
        X[s, 11, :, 0] = np.zeros_like(X[s, 11, :, 0])
        X[s, 11, :, -1] = np.zeros_like(X[s, 11, :, -1])

    return X


def build_fluent_fno_dataset(
    case_files: CaseFiles,
    ar_min: int = 10,
    ar_max: int = 20,
    nx: int = 64,
    ny: int = 64,
    n_subdomains: int = 10,
    input_mode: str = "local_full16_broadcast",
    layout: str = "NHWC",
) -> Dict[str, object]:
    """
    Build an in-memory FNO dataset.

    input_mode options:
        'local_full16'
            Recommended local mode. No x_global, no global AR, no subdomain ID.
            Full p/T/u/v traces are used on both left and right interfaces.
        'local_full16_broadcast'
            Same fields as local_full16, but p/T/u/v interface profiles are repeated across x.
            Same channels as local_full16, but left/right y-profiles are
            repeated through the interior to make them more visible to the FNO.
        'local_left12'
            Local left-to-right mode. No right interface field is given.
        'local_left12_broadcast'
            Same fields as local_left12, but left p/T/u/v interface profiles are repeated across x.
            Same as local_left12, but left profiles are repeated through x.
    """
    input_mode = input_mode.lower()
    layout = layout.upper()
    valid_modes = {"local_full16", "local_full16_broadcast", "local_left12", "local_left12_broadcast"}
    if input_mode not in valid_modes:
        raise ValueError(f"input_mode must be one of {sorted(valid_modes)}, got {input_mode!r}")
    if layout not in {"NHWC", "NCHW"}:
        raise ValueError("layout must be 'NHWC' or 'NCHW'")

    cases: Dict[int, Mapping[str, object]] = {}
    metadata: List[Mapping[str, object]] = []

    for ar in range(ar_min, ar_max + 1):
        if ar not in case_files:
            print(f"Skipping AR={ar}: not in case_files")
            continue
        print(f"Processing AR={ar}", flush=True)
        one = build_case_subdomains(
            mesh_h5=case_files[ar]["mesh"],
            dat_h5=case_files[ar]["dat"],
            aspect_ratio=ar,
            n_subdomains=n_subdomains,
            nx=nx,
            ny=ny,
        )
        cases[ar] = one
        metadata.extend(one["metadata"])

    if not cases:
        raise ValueError("No cases were processed. Check case_files and AR range.")

    X_parts: List[np.ndarray] = []
    Y_parts: List[np.ndarray] = []
    input_channel_names: Optional[List[str]] = None
    output_channel_names: Optional[List[str]] = None

    for ar in sorted(cases):
        case = cases[ar]
        X_coord = np.asarray(case["X"], dtype=np.float32)
        Y_case = np.asarray(case["Y"], dtype=np.float32)

        if output_channel_names is None:
            output_channel_names = list(case["output_channel_names"])

        if input_mode == "local_full16":
            X_case = make_local_full16_inputs_for_case(case, profile_placement="boundary")
            names = LOCAL_FULL16_CHANNELS
        elif input_mode == "local_full16_broadcast":
            X_case = make_local_full16_inputs_for_case(case, profile_placement="broadcast")
            names = LOCAL_FULL16_CHANNELS
        elif input_mode == "local_left12":
            X_case = make_local_left12_inputs_for_case(case, profile_placement="boundary")
            names = LOCAL_LEFT12_CHANNELS
        elif input_mode == "local_left12_broadcast":
            X_case = make_local_left12_inputs_for_case(case, profile_placement="broadcast")
            names = LOCAL_LEFT12_CHANNELS
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")
        
        if input_channel_names is None:
            input_channel_names = list(names)

        X_parts.append(X_case)
        Y_parts.append(Y_case)

    X_nchw = np.concatenate(X_parts, axis=0).astype(np.float32)
    Y_nchw = np.concatenate(Y_parts, axis=0).astype(np.float32)

    if layout == "NHWC":
        inputs = _to_nhwc(X_nchw)
        outputs = _to_nhwc(Y_nchw)
    else:
        inputs = X_nchw
        outputs = Y_nchw

    aspect_ratio = np.asarray([m["aspect_ratio"] for m in metadata], dtype=np.float32)
    subdomain_id = np.asarray([m["subdomain_id"] for m in metadata], dtype=np.int32)
    x_left_mm = np.asarray([m["x_left_mm"] for m in metadata], dtype=np.float32)
    x_right_mm = np.asarray([m["x_right_mm"] for m in metadata], dtype=np.float32)
    y_bottom_mm = np.asarray([m["y_bottom_mm"] for m in metadata], dtype=np.float32)
    y_top_mm = np.asarray([m["y_top_mm"] for m in metadata], dtype=np.float32)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "layout": layout,
        "input_mode": input_mode,
        "input_channel_names": input_channel_names or [],
        "output_channel_names": list(output_channel_names or []),
        "aspect_ratio": aspect_ratio,
        "subdomain_id": subdomain_id,
        "x_left_mm": x_left_mm,
        "x_right_mm": x_right_mm,
        "y_bottom_mm": y_bottom_mm,
        "y_top_mm": y_top_mm,
        "metadata": metadata,
        "cases": cases,
    }


def save_dataset_h5(dataset: Mapping[str, object], output_path: PathLike) -> None:
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("inputs", data=np.asarray(dataset["inputs"], dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("outputs", data=np.asarray(dataset["outputs"], dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
        for key in ["aspect_ratio", "subdomain_id", "x_left_mm", "x_right_mm", "y_bottom_mm", "y_top_mm"]:
            f.create_dataset(key, data=np.asarray(dataset[key]))
        f.attrs["layout"] = str(dataset["layout"])
        f.attrs["input_mode"] = str(dataset["input_mode"])
        f.attrs["input_channel_names_json"] = json.dumps(dataset["input_channel_names"])
        f.attrs["output_channel_names_json"] = json.dumps(dataset["output_channel_names"])


def _split_indices(
    dataset: Mapping[str, object],
    test_ars: Optional[Sequence[int]] = None,
    n_train: Optional[int] = None,
    test_fraction: float = 0.2,
    shuffle: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(np.asarray(dataset["inputs"]).shape[0])
    all_idx = np.arange(n)

    if test_ars is not None:
        ar = np.asarray(dataset["aspect_ratio"]).astype(int)
        test_mask = np.isin(ar, np.asarray(test_ars, dtype=int))
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        if len(test_idx) == 0:
            raise ValueError(f"No samples matched test_ars={list(test_ars)}")
        if len(train_idx) == 0:
            raise ValueError("No training samples remain after applying test_ars")
        return train_idx, test_idx

    rng = np.random.default_rng(seed)
    idx = all_idx.copy()
    if shuffle:
        rng.shuffle(idx)
    if n_train is None:
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be between 0 and 1")
        n_train = int(round(n * (1.0 - test_fraction)))
    n_train = int(n_train)
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"n_train must be in [1, {n - 1}], got {n_train}")
    return idx[:n_train], idx[n_train:]


def save_train_test_mat(
    dataset: Mapping[str, object],
    train_path: PathLike,
    test_path: PathLike,
    test_ars: Optional[Sequence[int]] = None,
    n_train: Optional[int] = None,
    test_fraction: float = 0.2,
    shuffle: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    train_path = _as_path(train_path)
    test_path = _as_path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = _split_indices(
        dataset,
        test_ars=test_ars,
        n_train=n_train,
        test_fraction=test_fraction,
        shuffle=shuffle,
        seed=seed,
    )

    def pack(idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "inputs": np.asarray(dataset["inputs"])[idx].astype(np.float32),
            "outputs": np.asarray(dataset["outputs"])[idx].astype(np.float32),
            "aspect_ratio": np.asarray(dataset["aspect_ratio"])[idx],
            "subdomain_id": np.asarray(dataset["subdomain_id"])[idx],
            "x_left_mm": np.asarray(dataset["x_left_mm"])[idx],
            "x_right_mm": np.asarray(dataset["x_right_mm"])[idx],
            "y_bottom_mm": np.asarray(dataset["y_bottom_mm"])[idx],
            "y_top_mm": np.asarray(dataset["y_top_mm"])[idx],
        }

    import scipy.io
    train_pack = pack(train_idx)
    test_pack = pack(test_idx)
    scipy.io.savemat(train_path, train_pack)
    scipy.io.savemat(test_path, test_pack)
    print(f"Saved train MAT: {train_path}  inputs={train_pack['inputs'].shape}")
    print(f"Saved test MAT : {test_path}  inputs={test_pack['inputs'].shape}")
    return train_idx, test_idx


def save_interfaces_npz(dataset: Mapping[str, object], output_path: PathLike) -> None:
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for ar, case in dataset["cases"].items():
        arrays[f"AR{int(ar)}_interfaces"] = np.asarray(case["interfaces"], dtype=np.float32)
        arrays[f"AR{int(ar)}_x_edges_mm"] = np.asarray(case["x_edges_mm"], dtype=np.float32)
        arrays[f"AR{int(ar)}_y_interface_mm"] = np.asarray(case["y_interface_mm"], dtype=np.float32)
    arrays["output_channel_names"] = np.asarray(dataset["output_channel_names"], dtype=object)
    np.savez_compressed(output_path, **arrays)


def build_and_save(
    case_files: CaseFiles,
    out_dir: PathLike,
    ar_min: int = 10,
    ar_max: int = 20,
    nx: int = 64,
    ny: int = 64,
    n_subdomains: int = 10,
    input_mode: str = "local_full16_broadcast",
    test_ars: Optional[Sequence[int]] = None,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Dict[str, object]:
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_fluent_fno_dataset(
        case_files=case_files,
        ar_min=ar_min,
        ar_max=ar_max,
        nx=nx,
        ny=ny,
        n_subdomains=n_subdomains,
        input_mode=input_mode,
        layout="NHWC",
    )

    stem = f"fluent_AR{ar_min}_{ar_max}_sub{n_subdomains}_s{nx}_{input_mode}"
    save_dataset_h5(dataset, out_dir / f"{stem}.h5")
    save_train_test_mat(
        dataset,
        train_path=out_dir / f"{stem}_train.mat",
        test_path=out_dir / f"{stem}_test.mat",
        test_ars=test_ars,
        test_fraction=test_fraction,
        seed=seed,
    )
    save_interfaces_npz(dataset, out_dir / f"{stem}_interfaces.npz")
    return dataset
