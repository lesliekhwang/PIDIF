# mesh_and_solve_2d.py

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import ansys.fluent.core as pyfluent
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_NPROCS = 4
DEFAULT_NITER = 200
DEFAULT_PRECISION = "double"
DEFAULT_UI_MODE = "hidden_gui"

BASE_DIR = Path("/home/nuoxu9/PIDIF")
DEFAULT_RUNS_ROOT = BASE_DIR / "runs_2d"


# ============================================================
# BASIC UTILS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def parse_scalar_file(path: Path) -> float | None:
    if not path.exists():
        return None

    txt = path.read_text().strip()
    if not txt:
        return None

    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if not tokens:
        return None

    try:
        return float(tokens[-1])
    except Exception:
        return None


# ============================================================
# SPEC HELPERS
# ============================================================

def get_case_info_from_spec(spec_json: Path) -> dict[str, Any]:
    spec = load_json(spec_json)
    meta = spec.get("metadata", {})

    case = spec["case"]
    uin_mps = float(meta.get("Uin_mps", 1.0))
    step_path = meta.get("target_geometry_file", str(spec_json.with_suffix(".step")))
    lx_m = float(meta["Lx_mm"]) / 1000.0
    ly_m = float(meta["Ly_mm"]) / 1000.0

    return {
        "case": case,
        "uin_mps": uin_mps,
        "step_path": str(step_path),
        "lx_m": lx_m,
        "ly_m": ly_m,
        "metadata": meta,
    }


# ============================================================
# PYFLUENT TASK HELPERS
# ============================================================

def set_task_state(task, state: dict[str, Any]) -> None:
    errs = []

    try:
        task.Arguments.set_state(state)
        return
    except Exception as e:
        errs.append(f"task.Arguments.set_state failed: {e}")

    for method_name in ["SetState", "setState"]:
        try:
            getattr(task.Arguments, method_name)(state)
            return
        except Exception as e:
            errs.append(f"task.Arguments.{method_name} failed: {e}")

    raise RuntimeError("\n".join(errs))


def execute_task(task) -> None:
    errs = []

    for method_name in ["Execute", "execute"]:
        try:
            getattr(task, method_name)()
            return
        except Exception as e:
            errs.append(f"{method_name} failed: {e}")

    raise RuntimeError("\n".join(errs))


def get_task(workflow, name: str):
    try:
        return workflow.TaskObject[name]
    except Exception as e:
        raise RuntimeError(f"Could not access workflow task '{name}': {e}")


# ============================================================
# 2D MESHING
# ============================================================

def launch_meshing(nprocs: int):
    log(f"[INFO] Launching Fluent meshing session with {nprocs} processes")
    return pyfluent.launch_fluent(
        mode="meshing",
        precision=DEFAULT_PRECISION,
        processor_count=nprocs,
        ui_mode=DEFAULT_UI_MODE,
    )


def initialize_2d_workflow(meshing) -> None:
    meshing.workflow.InitializeWorkflow(WorkflowType="2D Meshing")
    log("[INFO] Initialized workflow with {'WorkflowType': '2D Meshing'}")


def load_cad_geometry_2d(meshing, step_path: Path) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Load CAD Geometry")

    candidate_states = [
        {
            "FileName": str(step_path),
            "LengthUnit": "mm",
            "Refaceting": {"Refacet": False},
        },
        {
            "FileName": str(step_path),
            "LengthUnit": "mm",
        },
        {
            "FileName": str(step_path),
        },
    ]

    last_err = None
    for state in candidate_states:
        try:
            set_task_state(task, state)
            execute_task(task)
            log(f"[INFO] Loaded CAD geometry: {step_path}")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load CAD geometry from {step_path}. Last error: {last_err}")


def update_regions(meshing) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Update Regions")
    execute_task(task)
    log("[INFO] Updated regions")


def update_boundaries(meshing) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Update Boundaries")

    candidate_states = [
        {"SelectionType": "zone"},
        {"BoundaryLabelType": "zone"},
        {},
    ]

    last_err = None
    for state in candidate_states:
        try:
            if state:
                set_task_state(task, state)
            execute_task(task)
            log("[INFO] Updated boundaries")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to update boundaries. Last error: {last_err}")


def define_global_sizing(
    meshing,
    max_size_mm: float,
    min_size_mm: float,
    curvature_normal_angle_deg: float = 20.0,
) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Define Global Sizing")

    candidate_states = [
        {
            "CurvatureNormalAngle": float(curvature_normal_angle_deg),
            "MaxSize": float(max_size_mm),
            "MinSize": float(min_size_mm),
            "SizeFunctions": "Curvature",
        },
        {
            "CurvatureNormalAngle": float(curvature_normal_angle_deg),
            "MaxSize": float(max_size_mm),
            "MinSize": float(min_size_mm),
        },
        {
            "MaxSize": float(max_size_mm),
            "MinSize": float(min_size_mm),
        },
        {
            "MaxSize": float(max_size_mm),
        },
    ]

    last_err = None
    for state in candidate_states:
        try:
            set_task_state(task, state)
            execute_task(task)
            log(f"[INFO] Defined global sizing: max={max_size_mm} mm, min={min_size_mm} mm")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to define global sizing. Last error: {last_err}")


def generate_surface_mesh_2d(meshing) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Generate the Surface Mesh")

    candidate_states = [
        {
            "Surface2DPreferences": {
                "MergeEdgeZonesBasedOnLabels": "no",
                "MergeFaceZonesBasedOnLabels": "no",
                "ShowAdvancedOptions": True,
            }
        },
        {},
    ]

    last_err = None
    for state in candidate_states:
        try:
            if state:
                set_task_state(task, state)
            execute_task(task)
            log("[INFO] Generated 2D surface mesh")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to generate 2D surface mesh. Last error: {last_err}")


def export_fluent_2d_mesh(meshing, mesh_path: Path) -> None:
    ensure_dir(mesh_path.parent)

    workflow = meshing.workflow
    task = get_task(workflow, "Export Fluent 2D Mesh")

    candidate_states = [
        {"FileName": str(mesh_path)},
        {"Filename": str(mesh_path)},
    ]

    last_err = None
    for state in candidate_states:
        try:
            set_task_state(task, state)
            execute_task(task)
            log(f"[INFO] Exported Fluent 2D mesh: {mesh_path}")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to export Fluent 2D mesh to {mesh_path}. Last error: {last_err}")


def mesh_2d_from_step(
    step_path: Path,
    mesh_path: Path,
    nprocs: int,
    max_size_mm: float,
    min_size_mm: float,
) -> None:
    meshing = None
    try:
        meshing = launch_meshing(nprocs=nprocs)
        initialize_2d_workflow(meshing)
        load_cad_geometry_2d(meshing, step_path=step_path)
        update_regions(meshing)
        update_boundaries(meshing)
        define_global_sizing(
            meshing,
            max_size_mm=max_size_mm,
            min_size_mm=min_size_mm,
        )
        generate_surface_mesh_2d(meshing)
        export_fluent_2d_mesh(meshing, mesh_path=mesh_path)
    finally:
        if meshing is not None:
            try:
                meshing.exit()
            except Exception:
                pass


# ============================================================
# SOLVER SIDE
# ============================================================

def launch_solver(nprocs: int):
    log(f"[INFO] Launching Fluent solver session with {nprocs} processes")

    return pyfluent.launch_fluent(
        mode="solver",
        dimension=2,
        precision=DEFAULT_PRECISION,
        processor_count=nprocs,
        ui_mode=DEFAULT_UI_MODE,
    )


def mesh_check(solver) -> None:
    try:
        solver.settings.mesh.check()
        return
    except Exception as e:
        log(f"[WARN] settings.mesh.check failed: {e}")

    try:
        solver.tui.mesh.check()
    except Exception as e:
        log(f"[WARN] tui mesh.check failed: {e}")


def set_models_and_materials(solver) -> None:
    # enable energy equation
    try:
        solver.settings.setup.models.energy.enabled = True
        log("[INFO] Enabled energy equation")
    except Exception:
        try:
            solver.tui.define.models.energy("yes")
            log("[INFO] Enabled energy equation via TUI")
        except Exception as e:
            log(f"[WARN] Could not enable energy equation: {e}")
    
    # viscous model       
    try:
        solver.settings.setup.models.viscous.model = "laminar"
        log("[INFO] Set viscous model = laminar")
    except Exception as e:
        log(f"[WARN] Could not set laminar model explicitly: {e}")

    try:
        fluid_zones = list(solver.settings.setup.cell_zone_conditions.fluid.keys())
        log(f"[INFO] Fluid cell zones: {fluid_zones}")

        if fluid_zones:
            fluid_zone = fluid_zones[0]
            solver.settings.setup.cell_zone_conditions.fluid[fluid_zone].general.material = "air"
            log(f"[INFO] Assigned material 'air' to fluid zone '{fluid_zone}'")
    except Exception as e:
        log(f"[WARN] Could not inspect/assign fluid cell zone material: {e}")


def get_boundary_zone_names(solver) -> list[str]:
    try:
        names = list(solver.settings.setup.boundary_conditions.get_object_names())
    except Exception:
        try:
            names = list(solver.settings.setup.boundary_conditions.keys())
        except Exception as e:
            raise RuntimeError(f"Could not get boundary zone names: {e}")

    names = [str(n) for n in names if not str(n).startswith("interior:")]
    log(f"[INFO] Boundary names: {names}")
    return names


def get_zone_centroid(solver, zone_name: str) -> tuple[float, float, float]:
    c = solver.fields.reduction.centroid(locations=[zone_name], ctxt=solver)

    if hasattr(c, "__len__") and len(c) >= 3:
        return float(c[0]), float(c[1]), float(c[2])

    raise RuntimeError(f"Unexpected centroid return for zone '{zone_name}': {c}")


def classify_four_edge_boundaries(
    solver,
    lx_m: float,
    ly_m: float,
) -> dict[str, str]:
    names = get_boundary_zone_names(solver)

    if len(names) != 4:
        raise RuntimeError(
            f"Expected 4 boundary zones after refined STEP import, got {len(names)}: {names}"
        )

    zone_info = []
    for name in names:
        cx, cy, cz = get_zone_centroid(solver, name)
        zone_info.append({"name": name, "cx": cx, "cy": cy, "cz": cz})

    inlet = min(zone_info, key=lambda z: z["cx"])
    outlet = max(zone_info, key=lambda z: z["cx"])

    remaining = [z for z in zone_info if z["name"] not in {inlet["name"], outlet["name"]}]
    if len(remaining) != 2:
        raise RuntimeError(f"Expected 2 wall zones after inlet/outlet classification, got {remaining}")

    wall_bottom = min(remaining, key=lambda z: z["cy"])
    wall_top = max(remaining, key=lambda z: z["cy"])

    result = {
        "inlet": inlet["name"],
        "outlet": outlet["name"],
        "wall_bottom": wall_bottom["name"],
        "wall_top": wall_top["name"],
    }

    log(f"[INFO] Classified boundaries: {result}")
    return result


def convert_boundary_types(solver, inlet_zone: str, outlet_zone: str) -> None:
    bc = solver.settings.setup.boundary_conditions

    bc.set_zone_type(zone_list=[inlet_zone], new_type="velocity-inlet")
    bc.set_zone_type(zone_list=[outlet_zone], new_type="pressure-outlet")

    log(f"[INFO] Converted '{inlet_zone}' -> velocity-inlet")
    log(f"[INFO] Converted '{outlet_zone}' -> pressure-outlet")

def set_residual_targets(
    solver,
    continuity: float = 1e-15,
    x_velocity: float = 1e-15,
    y_velocity: float = 1e-15,
    energy: float = 1e-15,
) -> None:
    tried = []

    try:
        crit = solver.settings.solution.monitor.residual.equations
        crit["continuity"].absolute_criteria = continuity
        crit["x-velocity"].absolute_criteria = x_velocity
        crit["y-velocity"].absolute_criteria = y_velocity
        if "energy" in crit:
            crit["energy"].absolute_criteria = energy
        log(
            f"[INFO] Residual targets set via settings API: "
            f"continuity={continuity}, x={x_velocity}, y={y_velocity}, energy={energy}"
        )
        return
    except Exception as e:
        tried.append(f"settings residual equations failed: {e}")

    try:
        solver.tui.solve.monitors.residual.convergence_criteria(
            str(continuity),
            str(x_velocity),
            str(y_velocity),
            str(energy),
        )
        log("[INFO] Residual targets set via TUI")
        return
    except Exception as e:
        tried.append(f"TUI convergence_criteria failed: {e}")

    log("[WARN] Could not set residual targets:\n- " + "\n- ".join(tried))

def set_velocity_inlet(solver, inlet_name: str, uin_mps: float, temp_K: float = 300.0) -> None:

    inlet = solver.settings.setup.boundary_conditions.velocity_inlet[inlet_name]

    inlet.momentum.velocity_magnitude.value = float(uin_mps)

    try:
        inlet.thermal.temperature.value = float(temp_K)
    except Exception:
        pass

    log(f"[INFO] Set velocity inlet '{inlet_name}' = {uin_mps} m/s, T={temp_K} K")


def set_pressure_outlet(solver, outlet_name: str, gauge_pressure_pa: float = 0.0) -> None:
    outlet = solver.settings.setup.boundary_conditions.pressure_outlet[outlet_name]

    try:
        outlet.momentum.gauge_pressure = float(gauge_pressure_pa)
    except Exception:
        try:
            outlet.turbulence.gauge_pressure = float(gauge_pressure_pa)
        except Exception as e:
            raise RuntimeError(f"Failed to set pressure outlet '{outlet_name}': {e}")

    log(f"[INFO] Set pressure outlet '{outlet_name}' = {gauge_pressure_pa} Pa")

def set_wall_temperature(solver, wall_name: str, temp_K: float = 350.0) -> None:
    wall = solver.settings.setup.boundary_conditions.wall[wall_name]

    activation_errors = []

    for attr_name, value in [
        ("thermal_condition", "Temperature"),
        ("boundary_condition", "Temperature"),
    ]:
        try:
            setattr(wall.thermal, attr_name, value)
            log(f"[INFO] Activated wall thermal mode via {attr_name}='{value}' for '{wall_name}'")
            break
        except Exception as e:
            activation_errors.append(f"{attr_name}={value}: {e}")
    else:
        raise RuntimeError(
            f"Could not activate temperature BC for wall '{wall_name}'. Tried:\n- "
            + "\n- ".join(activation_errors)
        )

    value_errors = []

    try:
        wall.thermal.temperature.value = float(temp_K)
        log(f"[INFO] Set wall '{wall_name}' temperature = {temp_K} K via .value")
        return
    except Exception as e:
        value_errors.append(f"temperature.value: {e}")

    try:
        wall.thermal.temperature.set_state(float(temp_K))
        log(f"[INFO] Set wall '{wall_name}' temperature = {temp_K} K via set_state")
        return
    except Exception as e:
        value_errors.append(f"temperature.set_state: {e}")

    try:
        wall.thermal.temperature = float(temp_K)
        log(f"[INFO] Set wall '{wall_name}' temperature = {temp_K} K via direct assignment")
        return
    except Exception as e:
        value_errors.append(f"temperature=: {e}")

    raise RuntimeError(
        f"Temperature field for wall '{wall_name}' is still not writable after activation. Tried:\n- "
        + "\n- ".join(value_errors)
    )

def initialize_solution(solver) -> None:
    try:
        solver.tui.solve.initialize.hyb_initialization()
        log("[INFO] Hybrid initialization completed")
    except Exception:
        solver.tui.solve.initialize.initialize_flow()
        log("[INFO] Standard initialization completed")


def iterate_solver(solver, n_iter: int) -> None:
    solver.tui.solve.iterate(int(n_iter))
    log(f"[INFO] Solver iterated for {n_iter} steps")


def report_area_weighted_quantity(
    solver,
    zone_name: str,
    out_txt: Path,
    report_candidates: list[str],
    quantity_label: str,
) -> float:
    """
    Compute area-weighted average of a quantity on a boundary zone using the Results API.
    Writes the report to a file and returns the numeric value.
    """
    ensure_dir(out_txt.parent)

    tried = []
    results = solver.results

    for report_of in report_candidates:
        try:
            results.report.surface_integrals.area_weighted_avg(
                surface_names=[zone_name],
                report_of=report_of,
                write_to_file=True,
                file_name=str(out_txt),
            )

            val = parse_scalar_file(out_txt)
            if val is None:
                raise RuntimeError(f"Could not parse numeric value from {out_txt}")

            log(
                f"[INFO] Wrote area-weighted {quantity_label} for zone '{zone_name}' "
                f"using report_of='{report_of}' -> {out_txt} : {val:.12e}"
            )
            return val

        except Exception as e:
            tried.append(f"report_of='{report_of}': {e}")

    raise RuntimeError(
        f"Failed {quantity_label} report on zone '{zone_name}'. Tried:\n- "
        + "\n- ".join(tried)
    )

def report_area_weighted_pressure(solver, zone_name: str, out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["pressure", "static-pressure"],
        quantity_label="pressure",
    )

def report_area_weighted_temperature(solver, zone_name: str, out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["temperature", "static-temperature"],
        quantity_label="temperature",
    )

def report_area_weighted_velocity(solver, zone_name: str, out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["velocity-magnitude"],
        quantity_label="velocity",
    )

def write_residual_csv_from_monitor(solver, out_csv: Path):
    ensure_dir(out_csv.parent)

    iterations, data = solver.monitors.get_monitor_set_data(
        monitor_set_name="residual"
    )

    df = pd.DataFrame(data)
    df.insert(0, "iter", iterations)

    df.to_csv(out_csv, index=False)

    log(f"[INFO] Residual CSV written: {out_csv}")

def write_individual_residual_csvs(solver, out_dir: Path):
    """
    Export residual monitor history into individual CSV files
    for continuity, x-velocity, y-velocity, energy.
    """
    ensure_dir(out_dir)

    try:
        history = solver.monitor.get_monitor_set_data("residual")
    except Exception as e:
        log(f"[WARN] Could not access residual monitor history: {e}")
        return

    df = pd.DataFrame(history)

    mapping = {
        "continuity": "pressure.csv",
        "x-velocity": "x-velocity.csv",
        "y-velocity": "y-velocity.csv",
        "energy": "temperature.csv",
    }

    for key, filename in mapping.items():
        if key in df.columns:
            out_file = out_dir / filename
            pd.DataFrame({
                "iter": df["iter"],
                key: df[key],
            }).to_csv(out_file, index=False)

            log(f"[INFO] Saved residual history: {out_file}")

def save_residual_plot(residual_csv: Path, out_png: Path):
    if not residual_csv.exists():
        log(f"[WARN] Residual CSV not found: {residual_csv}")
        return

    try:
        df = pd.read_csv(residual_csv)
    except Exception as e:
        log(f"[WARN] Could not parse residual CSV {residual_csv}: {e}")
        return

    if "iter" not in df.columns:
        log(f"[WARN] Residual CSV has no 'iter' column: {residual_csv}")
        return

    plt.figure(figsize=(6, 4))

    for col in df.columns:
        if col.lower() != "iter":
            plt.semilogy(df["iter"], pd.to_numeric(df[col], errors="coerce"), label=col)

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Residual Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    log(f"[INFO] Saved residual plot: {out_png}")


def solve_2d_mesh(
    mesh_path: Path,
    out_dir: Path,
    uin_mps: float,
    nprocs: int,
    n_iter: int,
    lx_m: float,
    ly_m: float,
) -> dict[str, Any]:
    ensure_dir(out_dir)

    solver = None
    try:
        solver = launch_solver(nprocs=nprocs)

        solver.settings.file.read_mesh(file_name=str(mesh_path))
        log(f"[INFO] Read mesh: {mesh_path}")

        mesh_check(solver)
        set_models_and_materials(solver)

        boundary_map = classify_four_edge_boundaries(
            solver,
            lx_m=lx_m,
            ly_m=ly_m,
        )

        convert_boundary_types(
            solver,
            inlet_zone=boundary_map["inlet"],
            outlet_zone=boundary_map["outlet"],
        )

        set_velocity_inlet(solver, inlet_name=boundary_map["inlet"], uin_mps=uin_mps)
        set_pressure_outlet(solver, outlet_name=boundary_map["outlet"], gauge_pressure_pa=0.0)

        set_wall_temperature(solver, boundary_map["wall_top"], 350.0)
        set_wall_temperature(solver, boundary_map["wall_bottom"], 350.0)

        initialize_solution(solver)
        set_residual_targets(
            solver,
            continuity=1e-20,
            x_velocity=1e-20,
            y_velocity=1e-20,
            energy=1e-20,
        )
        iterate_solver(solver, n_iter=n_iter)

        residual_csv = out_dir / "residuals.csv"
        residual_png = out_dir / "residuals.png"
        write_residual_csv_from_monitor(solver, residual_csv)
        save_residual_plot(residual_csv, residual_png)
        write_individual_residual_csvs(solver, out_dir)

        pin_txt = out_dir / "pin.txt"
        pout_txt = out_dir / "pout.txt"
        tin_txt = out_dir / "tin.txt"
        tout_txt = out_dir / "tout.txt"
        vin_txt = out_dir / "vin.txt"
        vout_txt = out_dir / "vout.txt"
        case_data_path = out_dir / "case2d.cas.h5"

        pin = report_area_weighted_pressure(solver, boundary_map["inlet"], pin_txt)
        pout = report_area_weighted_pressure(solver, boundary_map["outlet"], pout_txt)
        dp = pin - pout

        tin = report_area_weighted_temperature(solver, boundary_map["inlet"], tin_txt)
        tout = report_area_weighted_temperature(solver, boundary_map["outlet"], tout_txt)

        vin = report_area_weighted_velocity(solver, boundary_map["inlet"], vin_txt)
        vout = report_area_weighted_velocity(solver, boundary_map["outlet"], vout_txt)

        wall_top_temp = 350.0
        wall_bottom_temp = 350.0

        postprocess_summary = {
            "case": mesh_path.stem.replace(".msh", ""),
            "boundaries": {
                "inlet": boundary_map["inlet"],
                "outlet": boundary_map["outlet"],
                "wall_top": boundary_map["wall_top"],
                "wall_bottom": boundary_map["wall_bottom"],
            },
            "inlet": {
                "pressure_pa": pin,
                "temperature_k": tin,
                "velocity_mps": vin,
                "txt_files": {
                    "pressure": str(pin_txt),
                    "temperature": str(tin_txt),
                    "velocity": str(vin_txt),
                },
            },
            "outlet": {
                "pressure_pa": pout,
                "temperature_k": tout,
                "velocity_mps": vout,
                "txt_files": {
                    "pressure": str(pout_txt),
                    "temperature": str(tout_txt),
                    "velocity": str(vout_txt),
                },
            },
            "wall_top": {
                "temperature_k": wall_top_temp,
            },
            "wall_bottom": {
                "temperature_k": wall_bottom_temp,
            },
            "derived": {
                "dp_pa": dp,
                "delta_t_k": tout - tin,
            },
            "residual_files": {
                "combined_csv": str(residual_csv),
                "plot_png": str(residual_png),
                "pressure_csv": str(out_dir / "pressure.csv"),
                "temperature_csv": str(out_dir / "temperature.csv"),
                "x_velocity_csv": str(out_dir / "x-velocity.csv"),
                "y_velocity_csv": str(out_dir / "y-velocity.csv"),
            },
        }

        post_json = out_dir / "postprocess_summary.json"
        with open(post_json, "w") as f:
            json.dump(postprocess_summary, f, indent=2)

        log(f"[INFO] Wrote postprocess summary: {post_json}")

        try:
            solver.settings.file.write_case_data(file_name=str(case_data_path))
        except Exception:
            solver.tui.file.write_case_data(str(case_data_path))
        log(f"[INFO] Wrote case/data: {case_data_path}")

        return {
            "mesh_path": str(mesh_path),
            "case_data_path": str(case_data_path),
            "pin_txt": str(pin_txt),
            "pout_txt": str(pout_txt),
            "pin_pa": pin,
            "pout_pa": pout,
            "dp_pa": dp,
            "inlet_name": boundary_map["inlet"],
            "outlet_name": boundary_map["outlet"],
            "wall_bottom_name": boundary_map["wall_bottom"],
            "wall_top_name": boundary_map["wall_top"],
            "tin_k": tin,
            "tout_k": tout,
            "vin_mps": vin,
            "vout_mps": vout,
            "tin_txt": str(tin_txt),
            "tout_txt": str(tout_txt),
            "vin_txt": str(vin_txt),
            "vout_txt": str(vout_txt),
            "residual_csv": str(residual_csv),
            "residual_png": str(residual_png),
            "postprocess_summary_json": str(post_json),
        }

    finally:
        if solver is not None:
            try:
                solver.exit()
            except Exception:
                pass

# ============================================================
# SINGLE CASE RUNNER
# ============================================================

def run_case_2d(
    spec_json: str | Path,
    step_path: str | Path,
    out_dir: str | Path,
    uin_mps: float | None = None,
    nprocs: int = DEFAULT_NPROCS,
    n_iter: int = DEFAULT_NITER,
    global_max_size_mm: float = 0.5,
    global_min_size_mm: float = 0.1,
) -> dict[str, Any]:
    spec_json = Path(spec_json)
    step_path = Path(step_path)
    out_dir = Path(out_dir)

    if not spec_json.exists():
        raise FileNotFoundError(f"Spec JSON not found: {spec_json}")
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    ensure_dir(out_dir)

    spec_info = get_case_info_from_spec(spec_json)
    case = spec_info["case"]
    if uin_mps is None:
        uin_mps = spec_info["uin_mps"]

    mesh_path = out_dir / f"{case}.msh.h5"
    summary_json = out_dir / "run_summary.json"

    started = time.time()

    mesh_2d_from_step(
        step_path=step_path,
        mesh_path=mesh_path,
        nprocs=nprocs,
        max_size_mm=global_max_size_mm,
        min_size_mm=global_min_size_mm,
    )

    solve_result = solve_2d_mesh(
        mesh_path=mesh_path,
        out_dir=out_dir,
        uin_mps=float(uin_mps),
        nprocs=nprocs,
        n_iter=n_iter,
        lx_m=spec_info["lx_m"],
        ly_m=spec_info["ly_m"],
    )

    result = {
        "case": case,
        "status": "ok",
        "spec_json": str(spec_json),
        "step_path": str(step_path),
        "uin_mps": float(uin_mps),
        "elapsed_sec": time.time() - started,
        **solve_result,
    }

    with open(summary_json, "w") as f:
        json.dump(result, f, indent=2)

    log(f"[OK] Finished case={case}  dp={result['dp_pa']}")
    return result

def _run_one_case_from_row(
    row: dict[str, str],
    runs_root: str,
    nprocs: int,
    n_iter: int,
    global_max_size_mm: float,
    global_min_size_mm: float,
) -> dict[str, Any]:
    case = row["case"]
    spec_json = Path(row["geometry_spec"])
    step_path = Path(row["target_geometry_file"])
    out_dir = Path(runs_root) / case
    uin = float(row.get("Uin_mps", 1.0))

    try:
        res = run_case_2d(
            spec_json=spec_json,
            step_path=step_path,
            out_dir=out_dir,
            uin_mps=uin,
            nprocs=nprocs,
            n_iter=n_iter,
            global_max_size_mm=global_max_size_mm,
            global_min_size_mm=global_min_size_mm,
        )
        return res
    except Exception as e:
        return {
            "case": case,
            "status": f"failed: {e}",
            "spec_json": str(spec_json),
            "step_path": str(step_path),
            "uin_mps": uin,
        }

# ============================================================
# BATCH MODE
# ============================================================

def batch_run_from_csv(
    designs_csv: str | Path,
    runs_root: str | Path,
    nprocs: int = DEFAULT_NPROCS,
    n_iter: int = DEFAULT_NITER,
    global_max_size_mm: float = 0.5,
    global_min_size_mm: float = 0.1,
    total_cores: int = 72,
    max_parallel_cases: int | None = None,
) -> Path:
    designs_csv = Path(designs_csv)
    runs_root = Path(runs_root)

    if not designs_csv.exists():
        raise FileNotFoundError(f"designs.csv not found: {designs_csv}")

    ensure_dir(runs_root)

    with open(designs_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {designs_csv}")

    if nprocs <= 0:
        raise ValueError("nprocs must be positive")

    auto_parallel = max(1, total_cores // nprocs)
    if max_parallel_cases is None:
        max_workers = min(len(rows), auto_parallel)
    else:
        max_workers = min(len(rows), max_parallel_cases, auto_parallel)

    log("=" * 72)
    log(f"[INFO] Total available cores: {total_cores}")
    log(f"[INFO] Cores per Fluent case: {nprocs}")
    log(f"[INFO] Parallel cases to launch: {max_workers}")
    log(f"[INFO] Theoretical core usage: {max_workers * nprocs}")
    log("=" * 72)

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(
                _run_one_case_from_row,
                row,
                str(runs_root),
                nprocs,
                n_iter,
                global_max_size_mm,
                global_min_size_mm,
            ): row["case"]
            for row in rows
        }

        for fut in as_completed(future_map):
            case = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
                if str(res.get("status", "")).startswith("failed:"):
                    log(f"[FAIL] case={case} reason={res['status']}")
                else:
                    log(f"[DONE] case={case} dp={res.get('dp_pa', '')}")
            except Exception as e:
                fail = {
                    "case": case,
                    "status": f"failed: {e}",
                }
                results.append(fail)
                log(f"[FAIL] case={case} reason={e}")

    results.sort(key=lambda r: r.get("case", ""))

    summary_csv = runs_root / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "case",
            "status",
            "spec_json",
            "step_path",
            "mesh_path",
            "case_data_path",
            "pin_txt",
            "pout_txt",
            "uin_mps",
            "pin_pa",
            "pout_pa",
            "dp_pa",
            "tin_txt",
            "tout_txt",
            "tin_k",
            "tout_k",
            "inlet_name",
            "outlet_name",
            "wall_bottom_name",
            "wall_top_name",
            "elapsed_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    log(f"[DONE] Wrote batch summary: {summary_csv}")
    return summary_csv

# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mesh and solve 2D STEP geometries using PyFluent.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json", type=str, help="Single geometry JSON spec path.")
    mode.add_argument("--csv", type=str, help="Batch designs.csv path.")

    p.add_argument("--step", type=str, default=None, help="Single-case STEP path.")
    p.add_argument("--out-dir", type=str, default=None, help="Single-case output directory.")
    p.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT), help="Batch runs root.")
    p.add_argument("--uin", type=float, default=None, help="Override inlet velocity for single case.")
    p.add_argument("--nprocs", type=int, default=DEFAULT_NPROCS)
    p.add_argument("--total-cores", type=int, default=72, help="Total CPU cores available on the machine. Used to choose how many cases to run in parallel.")
    p.add_argument("--max-parallel-cases", type=int, default=None, help="Optional hard cap on number of concurrent Fluent jobs.")
    p.add_argument("--niter", type=int, default=DEFAULT_NITER)
    p.add_argument("--max-size-mm", type=float, default=0.5, help="Maximum global mesh element size in mm. Smaller values produce finer meshes (more elements). Default: 0.5 mm.")
    p.add_argument("--min-size-mm", type=float, default=0.1, help="Minimum mesh element size in mm used for curvature/feature refinement. Must be smaller than --max-size-mm. Default: 0.1 mm.")

    return p


def main():
    args = build_argparser().parse_args()

    if args.json:
        spec_json = Path(args.json)
        spec_info = get_case_info_from_spec(spec_json)

        if args.step:
            step_path = Path(args.step)
        else:
            step_path = Path(spec_info["step_path"])

        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            out_dir = DEFAULT_RUNS_ROOT / spec_info["case"]

        res = run_case_2d(
            spec_json=spec_json,
            step_path=step_path,
            out_dir=out_dir,
            uin_mps=args.uin,
            nprocs=args.nprocs,
            n_iter=args.niter,
            global_max_size_mm=args.max_size_mm,
            global_min_size_mm=args.min_size_mm,
        )
        log(json.dumps(res, indent=2))

    elif args.csv:
        batch_run_from_csv(
            designs_csv=Path(args.csv),
            runs_root=Path(args.runs_root),
            nprocs=args.nprocs,
            n_iter=args.niter,
            global_max_size_mm=args.max_size_mm,
            global_min_size_mm=args.min_size_mm,
            total_cores=args.total_cores,
            max_parallel_cases=args.max_parallel_cases,
        )


if __name__ == "__main__":
    main()