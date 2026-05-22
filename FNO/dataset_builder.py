"""
CFD CSV -> MAT/HDF5 dataset builder for wavy-channel FNO/PIFNO training.

This version maps each physical wavy channel to a fixed computational domain:
    xi  = x / Lx
    eta = (y - h(x)) / Ly
where h(x) = A sin(2*pi*x/lambda + phase).

Assumption:
    The top and bottom walls are synchronized sinusoidal profiles, so the
    channel gap Ly is constant.
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.interpolate import griddata
from scipy.io import loadmat, savemat
from scipy.spatial import KDTree


# ============================================================
# CSV reading
# ============================================================

def read_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a CFD-exported CSV file.

    Supported formats:
      1. ANSYS CFD-Post format with a [Data] section.
      2. Plain CSV with columns such as x, y, pressure, temperature,
         x-velocity/u, and y-velocity/v.
    """
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    if not raw_text.strip():
        raise ValueError(f"Empty CSV file: {csv_path}")

    # Format A: ANSYS CFD-Post export.
    if "[Data]" in raw_text:
        lines = raw_text.splitlines()
        data_start = None
        for i, line in enumerate(lines):
            if "[Data]" in line:
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError(f"Could not locate [Data] section in {csv_path}")

        df = pd.read_csv(csv_path, skiprows=data_start, header=None, engine="python")
        df = df.iloc[:, :8].copy()
        df.columns = ["node", "x", "y", "z", "pressure", "temperature", "u", "v"]

    # Format B: plain CSV.
    else:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        rename_map = {
            "X": "x",
            "Y": "y",
            "Z": "z",
            "Pressure": "pressure",
            "Temperature": "temperature",
            "x-velocity": "u",
            "y-velocity": "v",
        }
        df = df.rename(columns=rename_map)

        if "z" not in df.columns:
            df["z"] = 0.0
        if "node" not in df.columns:
            df.insert(0, "node", range(len(df)))

    required = ["node", "x", "y", "z", "pressure", "temperature", "u", "v"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid numeric rows after parsing: {csv_path}")

    return df[required]


# ============================================================
# Computational grid and wavy-coordinate mapping
# ============================================================

def build_computational_grid(nx: int = 128, ny: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Return a uniform computational grid over [0, 1] x [0, 1]."""
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    Xi_grid, Eta_grid = np.meshgrid(xi, eta)
    return Xi_grid, Eta_grid


def wall_profile(x: np.ndarray, amp_mm: float, lam_mm: float, phase_rad: float) -> np.ndarray:
    """Bottom-wall profile h(x) in meters."""
    amp_m = amp_mm / 1000.0
    lam_m = lam_mm / 1000.0
    return amp_m * np.sin(2.0 * np.pi * x / lam_m + phase_rad)


def map_to_wavy_computational_coords(
    x: np.ndarray,
    y: np.ndarray,
    amp_mm: float,
    lam_mm: float,
    phase_rad: float,
    Lx: float = 0.050,
    Ly: float = 0.020,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map physical coordinates (x, y) to computational coordinates (xi, eta).

    xi  = x / Lx
    eta = (y - h(x)) / Ly
    """
    h = wall_profile(x, amp_mm, lam_mm, phase_rad)
    xi = x / Lx
    eta = (y - h) / Ly
    return xi, eta


# ============================================================
# Interpolation and masks
# ============================================================

def interpolate_scalar_field(
    xi: np.ndarray,
    eta: np.ndarray,
    values: np.ndarray,
    Xi_grid: np.ndarray,
    Eta_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate scattered values onto the computational grid."""
    grid_linear = griddata((xi, eta), values, (Xi_grid, Eta_grid), method="linear")
    grid_nearest = griddata((xi, eta), values, (Xi_grid, Eta_grid), method="nearest")
    return np.where(np.isnan(grid_linear), grid_nearest, grid_linear).astype(np.float32)


def grid_tolerance(Xi_grid: np.ndarray, Eta_grid: np.ndarray, factor: float = 1.5) -> float:
    """Compute a KDTree tolerance based on computational-grid spacing."""
    dxi = float(np.mean(np.diff(Xi_grid[0, :]))) if Xi_grid.shape[1] > 1 else 1.0
    deta = float(np.mean(np.diff(Eta_grid[:, 0]))) if Eta_grid.shape[0] > 1 else 1.0
    return factor * max(dxi, deta)


def make_mask_from_points(
    xi: np.ndarray,
    eta: np.ndarray,
    Xi_grid: np.ndarray,
    Eta_grid: np.ndarray,
) -> np.ndarray:
    """Create a binary mask on the computational grid from scattered points."""
    tol = grid_tolerance(Xi_grid, Eta_grid)
    tree = KDTree(np.stack([xi, eta], axis=1))
    grid_pts = np.stack([Xi_grid.ravel(), Eta_grid.ravel()], axis=1)
    min_dist, _ = tree.query(grid_pts)
    return (min_dist.reshape(Xi_grid.shape) <= tol).astype(np.float32)


def map_dataframe_points(
    df: pd.DataFrame,
    amp_mm: float,
    lam_mm: float,
    phase_rad: float,
    Lx: float,
    Ly: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Group duplicate physical points and map them to (xi, eta)."""
    df_g = df.groupby(["x", "y"], as_index=False).mean()
    xi, eta = map_to_wavy_computational_coords(
        df_g["x"].to_numpy(),
        df_g["y"].to_numpy(),
        amp_mm=amp_mm,
        lam_mm=lam_mm,
        phase_rad=phase_rad,
        Lx=Lx,
        Ly=Ly,
    )
    return xi, eta, df_g


def create_fluid_mask(
    df_domain: pd.DataFrame,
    Xi_grid: np.ndarray,
    Eta_grid: np.ndarray,
    amp_mm: float,
    lam_mm: float,
    phase_rad: float,
    Lx: float = 0.050,
    Ly: float = 0.020,
) -> np.ndarray:
    """Binary mask: 1 inside the fluid region, 0 outside."""
    xi, eta, _ = map_dataframe_points(df_domain, amp_mm, lam_mm, phase_rad, Lx, Ly)
    return make_mask_from_points(xi, eta, Xi_grid, Eta_grid)


def create_boundary_mask_and_field(
    df_bc: pd.DataFrame,
    Xi_grid: np.ndarray,
    Eta_grid: np.ndarray,
    field_name: str,
    amp_mm: float,
    lam_mm: float,
    phase_rad: float,
    Lx: float = 0.050,
    Ly: float = 0.020,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a boundary mask and prescribed boundary-value field."""
    if df_bc.empty or field_name not in df_bc.columns:
        empty = np.zeros(Xi_grid.shape, dtype=np.float32)
        return empty, empty

    xi, eta, df_g = map_dataframe_points(df_bc, amp_mm, lam_mm, phase_rad, Lx, Ly)
    values = df_g[field_name].to_numpy()

    value_grid = griddata((xi, eta), values, (Xi_grid, Eta_grid), method="nearest").astype(np.float32)
    mask = make_mask_from_points(xi, eta, Xi_grid, Eta_grid)
    value_grid = np.where(mask > 0, value_grid, 0.0).astype(np.float32)
    return mask, value_grid


# ============================================================
# Single-case conversion
# ============================================================

def case_csv_to_mat(
    domain_csv_path: str,
    inlet_csv_path: str,
    outlet_csv_path: str,
    walls_csv_path: str,
    output_path: str,
    amp_mm: float,
    lam_mm: float,
    phase_rad: float,
    Lx: float = 0.050,
    Ly: float = 0.020,
    nx: int = 128,
    ny: int = 128,
) -> None:
    """Build one MAT training sample from domain and boundary CSV files."""
    df_domain = read_csv(domain_csv_path)
    df_inlet = read_csv(inlet_csv_path)
    df_outlet = read_csv(outlet_csv_path)
    df_walls = read_csv(walls_csv_path)

    Xi_grid, Eta_grid = build_computational_grid(nx=nx, ny=ny)

    xi_dom, eta_dom, df_dom_g = map_dataframe_points(
        df_domain, amp_mm, lam_mm, phase_rad, Lx, Ly
    )

    # Output fields on the fixed computational grid.
    P_grid = interpolate_scalar_field(xi_dom, eta_dom, df_dom_g["pressure"].to_numpy(), Xi_grid, Eta_grid)
    T_grid = interpolate_scalar_field(xi_dom, eta_dom, df_dom_g["temperature"].to_numpy(), Xi_grid, Eta_grid)
    U_grid = interpolate_scalar_field(xi_dom, eta_dom, df_dom_g["u"].to_numpy(), Xi_grid, Eta_grid)
    V_grid = interpolate_scalar_field(xi_dom, eta_dom, df_dom_g["v"].to_numpy(), Xi_grid, Eta_grid)

    # Input masks and prescribed boundary-value fields.
    fluid_mask = create_fluid_mask(df_domain, Xi_grid, Eta_grid, amp_mm, lam_mm, phase_rad, Lx, Ly)

    inlet_mask, inlet_u = create_boundary_mask_and_field(
        df_inlet, Xi_grid, Eta_grid, "u", amp_mm, lam_mm, phase_rad, Lx, Ly
    )
    _, inlet_v = create_boundary_mask_and_field(
        df_inlet, Xi_grid, Eta_grid, "v", amp_mm, lam_mm, phase_rad, Lx, Ly
    )
    _, inlet_T = create_boundary_mask_and_field(
        df_inlet, Xi_grid, Eta_grid, "temperature", amp_mm, lam_mm, phase_rad, Lx, Ly
    )

    outlet_mask, outlet_p = create_boundary_mask_and_field(
        df_outlet, Xi_grid, Eta_grid, "pressure", amp_mm, lam_mm, phase_rad, Lx, Ly
    )

    wall_mask, wall_T = create_boundary_mask_and_field(
        df_walls, Xi_grid, Eta_grid, "temperature", amp_mm, lam_mm, phase_rad, Lx, Ly
    )
    _, wall_u = create_boundary_mask_and_field(
        df_walls, Xi_grid, Eta_grid, "u", amp_mm, lam_mm, phase_rad, Lx, Ly
    )
    _, wall_v = create_boundary_mask_and_field(
        df_walls, Xi_grid, Eta_grid, "v", amp_mm, lam_mm, phase_rad, Lx, Ly
    )

    inputs = np.stack(
        [
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
        ],
        axis=-1,
    ).astype(np.float32)

    outputs = np.stack([P_grid, T_grid, U_grid, V_grid], axis=-1).astype(np.float32)

    savemat(
        output_path,
        {
            "X_grid": Xi_grid.astype(np.float32),
            "Y_grid": Eta_grid.astype(np.float32),
            "inputs": inputs,
            "outputs": outputs,
        },
    )

    print(f"[OK] Saved {output_path}")
    print(f"     inputs  shape: {inputs.shape}")
    print(f"     outputs shape: {outputs.shape}")


# ============================================================
# Batch conversion
# ============================================================

def find_domain_csvs(root_dir: str) -> list[str]:
    """Find domain CSV files while excluding inlet/outlet/wall boundary CSVs."""
    all_csv = sorted(glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True))
    skip_suffixes = ("_inlet", "_outlet", "_walls")
    return [
        p for p in all_csv
        if not any(Path(p).stem.endswith(suffix) for suffix in skip_suffixes)
    ]


def infer_case_name(domain_csv: str) -> str:
    """Infer the case name from either channel_###.csv or field_volume.csv."""
    path = Path(domain_csv)
    return path.parent.name if path.stem == "field_volume" else path.stem


def convert_all_cases_to_mat(
    root_dir: str,
    output_dir: str,
    design_csv_path: str,
    nx: int = 128,
    ny: int = 128,
    Lx: float = 0.050,
    Ly: float = 0.020,
) -> None:
    """Convert all valid CFD case folders under root_dir into MAT files."""
    os.makedirs(output_dir, exist_ok=True)

    df_design = pd.read_csv(design_csv_path)
    df_design["case"] = df_design["case"].astype(str)
    design_by_case = df_design.set_index("case")

    domain_csvs = find_domain_csvs(root_dir)
    if not domain_csvs:
        print(f"[WARN] No domain CSV files found under {root_dir}")
        return

    ok_count = 0
    skip_count = 0

    for domain_csv in domain_csvs:
        case_name = infer_case_name(domain_csv)
        case_dir = os.path.dirname(domain_csv)

        inlet_csv = os.path.join(case_dir, f"{case_name}_inlet.csv")
        outlet_csv = os.path.join(case_dir, f"{case_name}_outlet.csv")
        walls_csv = os.path.join(case_dir, f"{case_name}_walls.csv")
        required_files = [domain_csv, inlet_csv, outlet_csv, walls_csv]

        missing = [p for p in required_files if not os.path.exists(p)]
        if missing:
            print(f"[SKIP] {case_name}: missing {[os.path.basename(p) for p in missing]}")
            skip_count += 1
            continue

        empty = [p for p in required_files if os.path.getsize(p) == 0]
        if empty:
            print(f"[SKIP] {case_name}: empty CSV {[os.path.basename(p) for p in empty]}")
            skip_count += 1
            continue

        if case_name not in design_by_case.index:
            print(f"[SKIP] {case_name}: geometry parameters not found in {design_csv_path}")
            skip_count += 1
            continue

        row = design_by_case.loc[case_name]
        amp_mm = float(row["A_mm"])
        lam_mm = float(row["lam_mm"])
        phase_rad = float(row["phase_rad"])

        output_mat = os.path.join(output_dir, f"{case_name}.mat")
        print(
            f"[...] Converting {case_name} | "
            f"A={amp_mm:.4f} mm, lambda={lam_mm:.4f} mm, phase={phase_rad:.4f} rad"
        )

        try:
            case_csv_to_mat(
                domain_csv_path=domain_csv,
                inlet_csv_path=inlet_csv,
                outlet_csv_path=outlet_csv,
                walls_csv_path=walls_csv,
                output_path=output_mat,
                amp_mm=amp_mm,
                lam_mm=lam_mm,
                phase_rad=phase_rad,
                Lx=Lx,
                Ly=Ly,
                nx=nx,
                ny=ny,
            )
            ok_count += 1
        except Exception as exc:
            print(f"[FAIL] {case_name}: {exc}")
            skip_count += 1

    print(f"\nDone. {ok_count} converted, {skip_count} skipped.")


# ============================================================
# HDF5 assembly
# ============================================================

def build_dataset_from_mat(
    mat_dir: str,
    design_csv_path: str,
    output_dataset_path: str,
    compression_opts: int = 4,
) -> None:
    """Assemble converted MAT samples into one HDF5 dataset."""
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {mat_dir}")

    df_design = pd.read_csv(design_csv_path)
    df_design["case"] = df_design["case"].astype(str)
    design_by_case = df_design.set_index("case")

    inputs_list = []
    outputs_list = []
    case_names = []
    uin_list = []
    amp_list = []
    lam_list = []
    phase_list = []

    ref_in_shape = None
    ref_out_shape = None

    for mat_file in mat_files:
        case_name = Path(mat_file).stem
        if case_name not in design_by_case.index:
            print(f"[WARN] {case_name} not found in designs.csv — skipping")
            continue

        data = loadmat(mat_file)
        inputs = data["inputs"].astype(np.float32)
        outputs = data["outputs"].astype(np.float32)

        if ref_in_shape is None:
            ref_in_shape = inputs.shape
            ref_out_shape = outputs.shape
        elif inputs.shape != ref_in_shape or outputs.shape != ref_out_shape:
            print(f"[WARN] Shape mismatch in {mat_file} — skipping")
            continue

        row = design_by_case.loc[case_name]
        inputs_list.append(inputs)
        outputs_list.append(outputs)
        case_names.append(case_name)
        uin_list.append(float(row["Uin_mps"]))
        amp_list.append(float(row["A_mm"]))
        lam_list.append(float(row["lam_mm"]))
        phase_list.append(float(row["phase_rad"]))

    if not inputs_list:
        raise RuntimeError("No valid MAT files were found for HDF5 assembly.")

    inputs_arr = np.stack(inputs_list, axis=0).astype(np.float32)
    outputs_arr = np.stack(outputs_list, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_dataset_path)), exist_ok=True)
    with h5py.File(output_dataset_path, "w") as hf:
        hf.create_dataset(
            "inputs",
            data=inputs_arr,
            compression="gzip",
            compression_opts=compression_opts,
            shuffle=True,
        )
        hf.create_dataset(
            "outputs",
            data=outputs_arr,
            compression="gzip",
            compression_opts=compression_opts,
            shuffle=True,
        )

        hf.create_dataset("case_names", data=np.array(case_names, dtype="S"))
        hf.create_dataset("uin", data=np.array(uin_list, dtype=np.float32))
        hf.create_dataset("amp", data=np.array(amp_list, dtype=np.float32))
        hf.create_dataset("lam", data=np.array(lam_list, dtype=np.float32))
        hf.create_dataset("phase", data=np.array(phase_list, dtype=np.float32))

        hf.attrs["n_cases"] = len(case_names)
        hf.attrs["grid_h"] = inputs_arr.shape[1]
        hf.attrs["grid_w"] = inputs_arr.shape[2]
        hf.attrs["n_in_ch"] = inputs_arr.shape[3]
        hf.attrs["n_out_ch"] = outputs_arr.shape[3]
        hf.attrs["in_channels"] = [
            "fluid_mask",
            "inlet_mask",
            "outlet_mask",
            "wall_mask",
            "inlet_u",
            "inlet_v",
            "inlet_T",
            "wall_T",
            "wall_u",
            "wall_v",
            "outlet_p",
        ]
        hf.attrs["out_channels"] = ["pressure", "temperature", "u", "v"]

    print(f"\n[OK] Dataset saved -> {output_dataset_path}")
    print(f"     inputs  shape: {inputs_arr.shape}")
    print(f"     outputs shape: {outputs_arr.shape}")


# ============================================================
# Train/test split
# ============================================================

def load_h5_split_by_uin(
    data_path: str,
    train_path: str = "train.mat",
    test_gap_path: str = "test_gap.mat",
    test_extrap_path: str = "test_extrap.mat",
) -> None:
    """Split HDF5 dataset by inlet-velocity quantiles and save MAT files."""
    with h5py.File(data_path, "r") as f:
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]
        uin = f["uin"][:]
        amp = f["amp"][:]
        lam = f["lam"][:]
        phase = f["phase"][:]

    X = torch.from_numpy(inputs.astype(np.float32))
    Y = torch.from_numpy(outputs.astype(np.float32))

    q5 = np.quantile(uin, 0.05)
    q45 = np.quantile(uin, 0.45)
    q55 = np.quantile(uin, 0.55)
    q95 = np.quantile(uin, 0.95)

    train_idx = np.where(((uin >= q5) & (uin <= q45)) | ((uin >= q55) & (uin <= q95)))[0]
    test_gap_idx = np.where((uin > q45) & (uin < q55))[0]
    test_extrap_idx = np.where((uin < q5) | (uin > q95))[0]

    print(f"[INFO] Uin quantiles 5%={q5:.4f}, 45%={q45:.4f}, 55%={q55:.4f}, 95%={q95:.4f}")
    print(f"[INFO] Train={len(train_idx)}, Test Gap={len(test_gap_idx)}, Test Extrap={len(test_extrap_idx)}")

    def save_split(path: str, indices: np.ndarray) -> None:
        scipy.io.savemat(
            path,
            {
                "inputs": X[indices].numpy(),
                "outputs": Y[indices].numpy(),
                "uin": uin[indices].astype(np.float32),
                "amp": amp[indices].astype(np.float32),
                "lam": lam[indices].astype(np.float32),
                "phase": phase[indices].astype(np.float32),
            },
        )
        print(f"[OK] Saved {path}: {X[indices].shape}")

    save_split(train_path, train_idx)
    save_split(test_gap_path, test_gap_idx)
    save_split(test_extrap_path, test_extrap_idx)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    nx = 64
    ny = 64

    CSV_ROOT_DIR = "runs_2d/"
    MAT_DIR = f"data/2d_s{nx}_mat_files"
    DATASET_PATH = f"data/2d_s{nx}.h5"
    DESIGN_CSV = "2d_geometry_specs/designs.csv"

    TRAIN_DATA_PATH = "data/2d_s64_train.mat"
    TEST_GAP_DATA_PATH = "data/2d_s64_test_gap.mat"
    TEST_EXTRAP_DATA_PATH = "data/2d_s64_test_extrap.mat"

    convert_all_cases_to_mat(
        root_dir=CSV_ROOT_DIR,
        output_dir=MAT_DIR,
        design_csv_path=DESIGN_CSV,
        nx=nx,
        ny=ny,
        Lx=0.050,
        Ly=0.020,
    )

    build_dataset_from_mat(
        mat_dir=MAT_DIR,
        design_csv_path=DESIGN_CSV,
        output_dataset_path=DATASET_PATH,
    )

    load_h5_split_by_uin(
        DATASET_PATH,
        train_path=TRAIN_DATA_PATH,
        test_gap_path=TEST_GAP_DATA_PATH,
        test_extrap_path=TEST_EXTRAP_DATA_PATH,
    )
