"""
CFD CSV → HDF5 dataset builder for FNO training.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import h5py
import torch
import scipy.io

# ============================================================
# CSV reading
# ============================================================

def read_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a CFD-exported CSV file.

    Supports two formats automatically:
      - ANSYS CFD-Post format  : has a [Data] section header
      - field_volume.csv format: plain CSV with column names
                                 (x, y, pressure, temperature, x-velocity / u, y-velocity / v)
    """
    with open(csv_path, "r") as f:
        raw_text = f.read()

    # ── Format A: ANSYS CFD-Post (has [Data] section) ────────────────────
    if "[Data]" in raw_text:
        lines = raw_text.splitlines()
        data_start = None
        for i, line in enumerate(lines):
            if "[Data]" in line:
                data_start = i + 1
                break

        df = pd.read_csv(
            csv_path,
            skiprows=data_start,
            header=None,
            engine="python",
        )
        df = df.iloc[:, :8].copy()
        df.columns = ["node", "x", "y", "z", "pressure", "temperature", "u", "v"]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df

    # ── Format B: plain CSV (field_volume.csv from PyFluent export) ──────
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "x-velocity": "u",
        "y-velocity": "v",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "Pressure": "pressure",
        "Temperature": "temperature",
    }
    df = df.rename(columns=rename_map)

    if "z" not in df.columns:
        df["z"] = 0.0
    if "node" not in df.columns:
        df.insert(0, "node", range(len(df)))

    required = ["node", "x", "y", "z", "pressure", "temperature", "u", "v"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing}\n"
                         f"  Available columns: {list(df.columns)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df[required]


# ============================================================
# Grid helpers
# ============================================================

def build_normalized_grid(nx: int = 128, ny: int = 128):
    """
    Return a uniform [0,1]×[0,1] grid.
    All cases use the same coordinate space regardless of physical domain size.

    Returns
    -------
    X_grid, Y_grid : ndarray of shape (ny, nx)
    """
    x_lin = np.linspace(0.0, 1.0, nx)
    y_lin = np.linspace(0.0, 1.0, ny)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    return X_grid, Y_grid


def normalize_coords(x: np.ndarray, y: np.ndarray):
    """
    Map physical (x, y) coordinates to [0,1]×[0,1].
    Uses the bounding box of the provided points.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    return x_norm, y_norm


# ============================================================
# Interpolation helpers
# ============================================================

def interpolate_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate scattered (x, y, values) onto a regular grid.
    NaNs from linear interpolation are filled with nearest-neighbor values.
    """
    grid_linear  = griddata((x, y), values, (X_grid, Y_grid), method="linear")
    grid_nearest = griddata((x, y), values, (X_grid, Y_grid), method="nearest")
    return np.where(np.isnan(grid_linear), grid_nearest, grid_linear)


# ============================================================
# Mask helpers  (KDTree-based — replaces O(N²) loop)
# ============================================================

def _grid_tolerance(X_grid: np.ndarray, Y_grid: np.ndarray, factor: float = 1.5) -> float:
    """Compute a spatial tolerance from the grid spacing."""
    dx = float(np.mean(np.diff(X_grid[0, :]))) if X_grid.shape[1] > 1 else 1.0
    dy = float(np.mean(np.diff(Y_grid[:, 0]))) if Y_grid.shape[0] > 1 else 1.0
    return factor * max(dx, dy)


def create_fluid_mask(
    df_domain: pd.DataFrame,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
) -> np.ndarray:
    """
    Binary mask: 1 where the grid point is inside the fluid domain, 0 elsewhere.
    Uses KDTree for O(N log N) nearest-neighbor search.
    """
    df_g = df_domain.groupby(["x", "y"], as_index=False).mean()
    x_norm, y_norm = normalize_coords(df_g["x"].to_numpy(), df_g["y"].to_numpy())

    tol = _grid_tolerance(X_grid, Y_grid)
    tree = KDTree(np.stack([x_norm, y_norm], axis=1))

    grid_pts = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    min_dist, _ = tree.query(grid_pts)

    mask = (min_dist.reshape(X_grid.shape) <= tol).astype(np.float32)
    return mask


def create_boundary_mask_and_field(
    df_bc: pd.DataFrame,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    field_name: str,
    domain_x: np.ndarray,
    domain_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a binary mask and boundary-value field on the regular grid.
    """
    if df_bc.empty or field_name not in df_bc.columns:
        empty = np.zeros(X_grid.shape, dtype=np.float32)
        return empty, empty

    df_g = df_bc.groupby(["x", "y"], as_index=False).mean()
    x_bc = df_g["x"].to_numpy()
    y_bc = df_g["y"].to_numpy()
    values = df_g[field_name].to_numpy()

    x_min, x_max = domain_x.min(), domain_x.max()
    y_min, y_max = domain_y.min(), domain_y.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_bc_norm = (x_bc - x_min) / x_range
    y_bc_norm = (y_bc - y_min) / y_range

    value_grid = griddata(
        (x_bc_norm, y_bc_norm), values, (X_grid, Y_grid), method="nearest"
    ).astype(np.float32)

    tol = _grid_tolerance(X_grid, Y_grid)
    tree = KDTree(np.stack([x_bc_norm, y_bc_norm], axis=1))
    grid_pts = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    min_dist, _ = tree.query(grid_pts)
    mask = (min_dist.reshape(X_grid.shape) <= tol).astype(np.float32)

    value_grid = np.where(mask > 0, value_grid, 0.0).astype(np.float32)
    return mask, value_grid


# ============================================================
# Single-case converter
# ============================================================

def case_csv_to_mat(
    domain_csv_path: str,
    inlet_csv_path: str,
    outlet_csv_path: str,
    walls_csv_path: str,
    output_path: str,
    nx: int = 128,
    ny: int = 128,
) -> None:
    """
    Build one training sample from four CSV files and save as a MAT file.

    Input channels (9):
        0  fluid_mask    — 1 inside fluid domain
        1  inlet_mask    — 1 on inlet boundary
        2  outlet_mask   — 1 on outlet boundary
        3  wall_mask     — 1 on wall boundaries
        4  inlet_u       — x-velocity on inlet
        5  inlet_v       — y-velocity on inlet
        6  inlet_T       — temperature on inlet
        7  wall_T        — temperature on walls
        8  wall_u
        9  wall_v
        10  outlet_p      — pressure on outlet

    Output channels (4):
        0  pressure
        1  temperature
        2  u  (x-velocity)
        3  v  (y-velocity)
    """

    df_domain = read_csv(domain_csv_path)
    df_inlet  = read_csv(inlet_csv_path)
    df_outlet = read_csv(outlet_csv_path)
    df_walls  = read_csv(walls_csv_path)

    X_grid, Y_grid = build_normalized_grid(nx=nx, ny=ny)

    df_dom_g = df_domain.groupby(["x", "y"], as_index=False).mean()
    x_dom = df_dom_g["x"].to_numpy()
    y_dom = df_dom_g["y"].to_numpy()

    x_dom_norm, y_dom_norm = normalize_coords(x_dom, y_dom)

    P_grid = interpolate_scalar_field(x_dom_norm, y_dom_norm,
                                      df_dom_g["pressure"].to_numpy(),    X_grid, Y_grid)
    T_grid = interpolate_scalar_field(x_dom_norm, y_dom_norm,
                                      df_dom_g["temperature"].to_numpy(), X_grid, Y_grid)
    U_grid = interpolate_scalar_field(x_dom_norm, y_dom_norm,
                                      df_dom_g["u"].to_numpy(),           X_grid, Y_grid)
    V_grid = interpolate_scalar_field(x_dom_norm, y_dom_norm,
                                      df_dom_g["v"].to_numpy(),           X_grid, Y_grid)

    fluid_mask = create_fluid_mask(df_domain, X_grid, Y_grid)

    inlet_mask,  inlet_u = create_boundary_mask_and_field(
        df_inlet,  X_grid, Y_grid, "u",           x_dom, y_dom)
    _,           inlet_v = create_boundary_mask_and_field(
        df_inlet,  X_grid, Y_grid, "v",           x_dom, y_dom)
    _,           inlet_T = create_boundary_mask_and_field(
        df_inlet,  X_grid, Y_grid, "temperature", x_dom, y_dom)

    outlet_mask, outlet_p = create_boundary_mask_and_field(
        df_outlet, X_grid, Y_grid, "pressure",    x_dom, y_dom)

    wall_mask,   wall_T = create_boundary_mask_and_field(
        df_walls,  X_grid, Y_grid, "temperature", x_dom, y_dom)
    _,  wall_u = create_boundary_mask_and_field(
        df_walls,  X_grid, Y_grid, "u",           x_dom, y_dom)
    _,  wall_v = create_boundary_mask_and_field(
        df_walls,  X_grid, Y_grid, "v",           x_dom, y_dom)
    
    inputs = np.stack([
        fluid_mask,
        inlet_mask,
        outlet_mask,
        wall_mask,
        inlet_u,
        inlet_v,
        inlet_T,
        wall_T,
        wall_u,
        wall_v,
        outlet_p,
    ], axis=-1).astype(np.float32)

    outputs = np.stack([
        P_grid, T_grid, U_grid, V_grid
    ], axis=-1).astype(np.float32)

    savemat(output_path, {
        "X_grid":  X_grid.astype(np.float32),
        "Y_grid":  Y_grid.astype(np.float32),
        "inputs":  inputs,
        "outputs": outputs,
    })

    print(f"[OK] Saved  {output_path}")
    print(f"     inputs  shape : {inputs.shape}")
    print(f"     outputs shape : {outputs.shape}")


# ============================================================
# Batch converter
# ============================================================

def convert_all_cases_to_mat(
    root_dir: str,
    output_dir: str,
    nx: int = 128,
    ny: int = 128,
) -> None:
    """
    Convert every case folder under root_dir into a MAT file.

    Expected layout (two naming conventions supported):

      Layout A — separate boundary files:
        root_dir/
          channel_01/
            channel_01.csv          ← domain
            channel_01_inlet.csv
            channel_01_outlet.csv
            channel_01_walls.csv

      Layout B — single field_volume.csv (PyFluent direct export):
        root_dir/
          channel_01/
            field_volume.csv        ← domain
            channel_01_inlet.csv
            channel_01_outlet.csv
            channel_01_walls.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    all_csv = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    skip_suffixes = ("_inlet", "_outlet", "_walls")

    domain_csvs = [
        p for p in all_csv
        if not any(os.path.splitext(os.path.basename(p))[0].endswith(s)
                   for s in skip_suffixes)
    ]

    if not domain_csvs:
        print(f"[WARN] No domain CSV files found under {root_dir}")
        return

    ok_count = 0
    skip_count = 0

    for domain_csv in domain_csvs:
        base_name = os.path.splitext(os.path.basename(domain_csv))[0]
        case_dir  = os.path.dirname(domain_csv)

        if base_name == "field_volume":
            case_name = os.path.basename(case_dir)
        else:
            case_name = base_name

        inlet_csv  = os.path.join(case_dir, case_name + "_inlet.csv")
        outlet_csv = os.path.join(case_dir, case_name + "_outlet.csv")
        walls_csv  = os.path.join(case_dir, case_name + "_walls.csv")

        missing = [p for p in [inlet_csv, outlet_csv, walls_csv]
                   if not os.path.exists(p)]
        if missing:
            print(f"[SKIP] {case_name}: missing {[os.path.basename(p) for p in missing]}")
            skip_count += 1
            continue

        output_mat = os.path.join(output_dir, case_name + ".mat")
        print(f"[...] Converting {case_name}")
        try:
            case_csv_to_mat(
                domain_csv, inlet_csv, outlet_csv, walls_csv,
                output_mat, nx=nx, ny=ny,
            )
            ok_count += 1
        except Exception as e:
            print(f"[FAIL] {case_name}: {e}")
            skip_count += 1

    print(f"\nDone. {ok_count} converted, {skip_count} skipped.")


# ============================================================
# Dataset assembler  — HDF5
# ============================================================
def build_dataset_from_mat(
    mat_dir: str,
    design_csv_path: str,
    output_dataset_path: str,
    chunk_cases: int = 10,
) -> None:
    import pandas as pd

    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {mat_dir}")

    df_design = pd.read_csv(design_csv_path)
    df_design["case"] = df_design["case"].astype(str)

    case_to_uin = dict(zip(df_design["case"], df_design["Uin_mps"]))

    # ───────────────────────────────
    # reference shape
    # ───────────────────────────────
    first = loadmat(mat_files[0])
    ref_in_shape  = first["inputs"].shape
    ref_out_shape = first["outputs"].shape

    H, W, C_in  = ref_in_shape
    _, _, C_out = ref_out_shape
    N = len(mat_files)

    print(f"[INFO] {N} MAT files found. Grid: {H}×{W}")
    print(f"[INFO] in_ch={C_in}, out_ch={C_out}")

    os.makedirs(os.path.dirname(os.path.abspath(output_dataset_path)), exist_ok=True)

    with h5py.File(output_dataset_path, "w") as hf:
        ds_in = hf.create_dataset(
            "inputs",
            shape=(N, H, W, C_in),
            dtype=np.float32,
            chunks=(chunk_cases, H, W, C_in),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        ds_out = hf.create_dataset(
            "outputs",
            shape=(N, H, W, C_out),
            dtype=np.float32,
            chunks=(chunk_cases, H, W, C_out),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        case_names = []
        uin_list   = []

        ok_count   = 0
        skip_count = 0

        for mat_file in mat_files:
            data    = loadmat(mat_file)
            inputs  = data["inputs"].astype(np.float32)
            outputs = data["outputs"].astype(np.float32)

            if inputs.shape != ref_in_shape or outputs.shape != ref_out_shape:
                print(f"[WARN] Shape mismatch in {mat_file} — skipping")
                skip_count += 1
                continue

            case_name = os.path.splitext(os.path.basename(mat_file))[0]

            if case_name not in case_to_uin:
                print(f"[WARN] {case_name} not found in designs.csv — skipping")
                skip_count += 1
                continue

            uin_value = case_to_uin[case_name]

            ds_in[ok_count]  = inputs
            ds_out[ok_count] = outputs

            case_names.append(case_name)
            uin_list.append(uin_value)

            ok_count += 1

            if ok_count % 50 == 0 or ok_count == N:
                print(f"  [{ok_count}/{N}] written ...")

        # resize
        if ok_count < N:
            ds_in.resize(ok_count, axis=0)
            ds_out.resize(ok_count, axis=0)

        # ───────────────────────────────
        # Save Metadata
        # ───────────────────────────────
        hf.create_dataset("case_names", data=np.array(case_names, dtype="S"))
        hf.create_dataset("uin", data=np.array(uin_list, dtype=np.float32))

        hf.attrs["n_cases"]  = ok_count
        hf.attrs["grid_h"]   = H
        hf.attrs["grid_w"]   = W
        hf.attrs["n_in_ch"]  = C_in
        hf.attrs["n_out_ch"] = C_out

        hf.attrs["in_channels"] = [
            "fluid_mask", "inlet_mask", "outlet_mask",
            "wall_mask", "inlet_u", "inlet_v",
            "inlet_T", "wall_T", "wall_u", "wall_v", "outlet_p"
        ]

        hf.attrs["out_channels"] = [
            "pressure", "temperature", "u", "v"
        ]

    print(f"\n[OK] Dataset saved → {output_dataset_path}")
    print(f"     inputs  shape : ({ok_count}, {H}, {W}, {C_in})")
    print(f"     outputs shape : ({ok_count}, {H}, {W}, {C_out})")

# ============================================================
# Train/Test split
# ============================================================

def load_mat_split_save(data_path, n_train=1000, train_path="train.mat", test_path="test.mat"):
    data = scipy.io.loadmat(data_path)
    
    X = torch.from_numpy(data["inputs"].astype("float32"))
    Y = torch.from_numpy(data["outputs"].astype("float32"))
    
    # train/test split
    x_train = X[:n_train]
    y_train = Y[:n_train]
    x_test = X[n_train:]
    y_test = Y[n_train:]
    
    scipy.io.savemat(train_path, {"inputs": x_train.numpy(), "outputs": y_train.numpy()})
    scipy.io.savemat(test_path, {"inputs": x_test.numpy(), "outputs": y_test.numpy()})
    
    print(f"Train data saved to {train_path}, shape: {x_train.shape}")
    print(f"Test data saved to {test_path}, shape: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test

def load_h5_split_save(data_path, n_train=1000, train_path="train.mat", test_path="test.mat"):
    with h5py.File(data_path, 'r') as f:
        inputs = f['inputs'][:]
        outputs = f['outputs'][:]
    
    X = torch.from_numpy(inputs.astype('float32'))
    Y = torch.from_numpy(outputs.astype('float32'))
    
    # train/test split
    x_train = X[:n_train]
    y_train = Y[:n_train]
    x_test = X[n_train:]
    y_test = Y[n_train:]

    scipy.io.savemat(train_path, {"inputs": x_train.numpy(), "outputs": y_train.numpy()})
    scipy.io.savemat(test_path, {"inputs": x_test.numpy(), "outputs": y_test.numpy()})
    
    print(f"Train data saved to {train_path}, shape: {x_train.shape}")
    print(f"Test data saved to {test_path}, shape: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test

def load_h5_split_by_uin(
    data_path,
    train_path="train.mat",
    test_gap_path="test_gap.mat",
    test_extrap_path="test_extrap.mat"
):
    with h5py.File(data_path, 'r') as f:
        inputs  = f['inputs'][:]
        outputs = f['outputs'][:]
        uin     = f['uin'][:]

    X = torch.from_numpy(inputs.astype('float32'))
    Y = torch.from_numpy(outputs.astype('float32'))

    # ───────────────────────────────
    # quantile 
    # ───────────────────────────────
    q5 = np.quantile(uin, 0.05)
    q10 = np.quantile(uin, 0.10)
    q45 = np.quantile(uin, 0.45)
    q55 = np.quantile(uin, 0.55)
    q90 = np.quantile(uin, 0.90)
    q95 = np.quantile(uin, 0.95)

    print(f"[INFO] Uin quantiles  5%: {q5:.4f} | 45%: {q45:.4f} | 55%: {q55:.4f} | 95%: {q95:.4f}")

    # # Train  : 10~45% + 55~90%
    # train_idx      = np.where(((uin >= q10) & (uin <= q45)) |
    #                            ((uin >= q55) & (uin <= q90)))[0]

    # # Test A : 45~55%  (gap interpolation)
    # test_gap_idx   = np.where((uin > q45) & (uin < q55))[0]

    # # Test B : 0~10% + 90~100%  (extrapolation)
    # test_extrap_idx = np.where((uin < q10) | (uin > q90))[0]

     # Train  : 5~45% + 55~95%
    train_idx      = np.where(((uin >= q5) & (uin <= q45)) |
                               ((uin >= q55) & (uin <= q95)))[0]

    # Test A : 45~55%  (gap interpolation)
    test_gap_idx   = np.where((uin > q45) & (uin < q55))[0]

    # Test B : 0~10% + 90~100%  (extrapolation)
    test_extrap_idx = np.where((uin < q5) | (uin > q95))[0]

    print(f"[INFO] Train       : {len(train_idx)}")
    print(f"[INFO] Test Gap    : {len(test_gap_idx)}")
    print(f"[INFO] Test Extrap : {len(test_extrap_idx)}")

    x_train      = X[train_idx]
    y_train      = Y[train_idx]
    x_test_gap   = X[test_gap_idx]
    y_test_gap   = Y[test_gap_idx]
    x_test_extrap = X[test_extrap_idx]
    y_test_extrap = Y[test_extrap_idx]

    scipy.io.savemat(train_path, {
        "inputs":  x_train.numpy(),
        "outputs": y_train.numpy(),
        "uin":     uin[train_idx].astype(np.float32)
    })

    scipy.io.savemat(test_gap_path, {
        "inputs":  x_test_gap.numpy(),
        "outputs": y_test_gap.numpy(),
        "uin":     uin[test_gap_idx].astype(np.float32)
    })

    scipy.io.savemat(test_extrap_path, {
        "inputs":  x_test_extrap.numpy(),
        "outputs": y_test_extrap.numpy(),
        "uin":     uin[test_extrap_idx].astype(np.float32)
    })

    print(f"\n[OK] Train       saved: {x_train.shape}")
    print(f"[OK] Test Gap    saved: {x_test_gap.shape}")
    print(f"[OK] Test Extrap saved: {x_test_extrap.shape}")

    # return x_train, y_train, x_test_gap, y_test_gap, x_test_extrap, y_test_extrap


if __name__ == "__main__":
    N  = 1000
    nx = 64
    ny = 64

    CSV_ROOT_DIR = "runs_2d/"
    MAT_DIR      = f"data/2d_s{nx}_mat_files"
    DATASET_PATH = f"data/2d_s{nx}.h5"

    DESIGN_CSV = "2d_geometry_specs/designs.csv"

    TRAIN_DATA_PATH = "data/2d_s64_train.mat"
    TEST_GAP_DATA_PATH = "data/2d_s64_test_gap.mat"
    TEST_EXTRAP_DATA_PATH = "data/2d_s64_test_extrap.mat"
    
    convert_all_cases_to_mat(
        root_dir=CSV_ROOT_DIR,
        output_dir=MAT_DIR,
        nx=nx,
        ny=ny,
    )

    build_dataset_from_mat(
        mat_dir=MAT_DIR,
        design_csv_path=DESIGN_CSV,
        output_dataset_path=DATASET_PATH,
        chunk_cases=10,
    )

    load_h5_split_by_uin(
        DATASET_PATH,
        train_path=TRAIN_DATA_PATH,
        test_gap_path=TEST_GAP_DATA_PATH,
        test_extrap_path=TEST_EXTRAP_DATA_PATH
    )



