# mesh_and_solve_2d.py

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
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

DEFAULT_NPROCS = 10
DEFAULT_NITER = 1000
DEFAULT_PRECISION = "double"
DEFAULT_UI_MODE = "hidden_gui"

# Fluent default air material properties (used for physics sanity checks).
AIR_DENSITY = 1.225          # kg/m^3
AIR_VISCOSITY = 1.7894e-5    # Pa*s

# Physics-check thresholds.
MASS_CONSERVATION_REL_TOL = 0.02   # 2% mismatch between inlet/outlet flux
REYNOLDS_LAMINAR_LIMIT = 2300.0    # below this the laminar model is appropriate

BASE_DIR = Path("/home/hantianl/Documents/PIDIF")
DEFAULT_RUNS_ROOT = BASE_DIR / "runs_2d" / "rand_channel_smooth"


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

def _segment_height_m(points: list[dict[str, Any]] | None, units: str = "mm") -> float | None:
    """Height (max y - min y) of an inlet/outlet segment, converted to meters."""
    if not points or len(points) < 2:
        return None
    ys = [float(p["y"]) for p in points]
    h = max(ys) - min(ys)
    if h <= 0.0:
        return None
    return h / 1000.0 if units.lower().strip() == "mm" else h


def get_case_info_from_spec(spec_json: Path) -> dict[str, Any]:
    spec = load_json(spec_json)
    meta = spec.get("metadata", {})
    units = spec.get("units", "mm")
    boundaries = spec.get("boundaries", {})

    case = spec["case"]
    inlet_profile = meta.get("inlet_velocity_profile", [])
    if not isinstance(inlet_profile, list):
        inlet_profile = []

    # Prefer explicit scalar velocity if present; otherwise derive from profile.
    if "Uin_mps" in meta:
        uin_mps = float(meta["Uin_mps"])
    elif inlet_profile:
        u_vals = [float(p.get("u_mps", 0.0)) for p in inlet_profile]
        uin_mps = float(sum(u_vals) / len(u_vals)) if u_vals else 1.0
    else:
        uin_mps = 1.0

    step_path = meta.get("target_geometry_file", str(spec_json.with_suffix(".step")))

    # Backward + forward compatibility:
    # old schema: Lx_mm/Ly_mm
    # new schema: L_mm
    if "Lx_mm" in meta and "Ly_mm" in meta:
        lx_m = float(meta["Lx_mm"]) / 1000.0
        ly_m = float(meta["Ly_mm"]) / 1000.0
    elif "L_mm" in meta:
        lx_m = float(meta["L_mm"]) / 1000.0
        ly_m = float(meta["L_mm"]) / 1000.0
    else:
        raise KeyError(
            "Spec metadata missing geometry size fields. "
            "Expected either (Lx_mm, Ly_mm) or L_mm."
        )

    # Inlet/outlet heights for mass-conservation and Reynolds-number checks.
    inlet_height_m = _segment_height_m(boundaries.get("inlet"), units)
    if inlet_height_m is None and "inlet_height_mm" in meta:
        inlet_height_m = float(meta["inlet_height_mm"]) / 1000.0
    outlet_height_m = _segment_height_m(boundaries.get("outlet"), units)

    return {
        "case": case,
        "uin_mps": uin_mps,
        "inlet_velocity_profile": inlet_profile,
        "inlet_velocity_scale_mps": float(meta.get("inlet_velocity_scale_mps", uin_mps)),
        "step_path": str(step_path),
        "lx_m": lx_m,
        "ly_m": ly_m,
        "inlet_height_m": inlet_height_m,
        "outlet_height_m": outlet_height_m,
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

def add_boundary_layers(meshing, 
                        name: str = "inflation-1",
                        n_layers: int = 5,
                        growth_rate: float | None = 1.2,
                        offset_method: str = "smooth-transition"
                        ) -> None:
    workflow = meshing.workflow
    task = get_task(workflow, "Add 2D Boundary Layers")

    candidate_states = [
        {
            "AddChild": "yes",
            "BLControlName": name,
            "NumberOfLayers": n_layers,
            "OffsetMethodType": offset_method,
        }
    ]
    
    if growth_rate is not None:
        for state in candidate_states:
            state["GrowthRate"] = growth_rate
        
    last_err = None
    for state in candidate_states:
        try:
            set_task_state(task, state)
            task.AddChildAndUpdate(DeferUpdate=True)
            log(f"[INFO] Added 2D boundary layers: {name} with {n_layers} layers, growth rate={growth_rate}, offset method={offset_method}")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to add 2D boundary layers. Last error: {last_err}")

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
    boundary_layers: list[dict[str, Any]] = [],
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
        # for boundary_layer in boundary_layers:
        #     add_boundary_layers(meshing, **boundary_layer)
        # face_zone_id_list = meshing.meshing_utilities.get_face_zones(filter="*")
        # cell_zone_id_list = meshing.meshing_utilities.get_cell_zones(filter="*")
        # edge_zone_id_list = meshing.meshing_utilities.get_edge_zones(filter="*")
        # for id in edge_zone_id_list:
        #     print(f"[INFO] Edge zone {id} type: {meshing.meshing_utilities.get_zone_type(zone_id=id)}")
        # for id in face_zone_id_list:
        #     print(f"[INFO] Face zone {id} type: {meshing.meshing_utilities.get_zone_type(zone_id=id)}")
        # for id in cell_zone_id_list:
        #     print(f"[INFO] Cell zone {id} type: {meshing.meshing_utilities.get_zone_type(zone_id=id)}")

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
    # keep energy equation disabled for isothermal runs
    try:
        solver.settings.setup.models.energy.enabled = False
        log("[INFO] Energy equation disabled")
    except Exception:
        try:
            solver.tui.define.models.energy("no")
            log("[INFO] Energy equation disabled via TUI")
        except Exception as e:
            log(f"[WARN] Could not explicitly disable energy equation: {e}")
    
    # viscous model       
    try:
        solver.settings.setup.models.viscous.model = "laminar"
        log("[INFO] Set viscous model = laminar")
    except Exception as e:
        log(f"[WARN] Could not set laminar model explicitly: {e}")
    
    # try:
    #     flow_scheme = solver.settings.solution.methods.p_v_coupling.flow_scheme
    #     flow_scheme.set_state("SIMPLE")
    #     log("[INFO] Set flow scheme = SIMPLE")
    # except Exception as e:
    #     log(f"[WARN] Could not set flow scheme explicitly: {e}")

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


def classify_channel_boundaries(
    solver,
    lx_m: float,
    ly_m: float,
) -> dict[str, Any]:
    names = get_boundary_zone_names(solver)

    if len(names) < 4:
        raise RuntimeError(
            f"Expected at least 4 boundary zones after STEP import, got {len(names)}: {names}"
        )

    zone_info = []
    for name in names:
        cx, cy, cz = get_zone_centroid(solver, name)
        zone_info.append({"name": name, "cx": cx, "cy": cy, "cz": cz})

    # For channel-like geometries, inlet/outlet are the most-left and most-right boundaries.
    inlet = min(zone_info, key=lambda z: z["cx"])
    outlet = max(zone_info, key=lambda z: z["cx"])

    remaining = [z for z in zone_info if z["name"] not in {inlet["name"], outlet["name"]}]
    if len(remaining) < 2:
        raise RuntimeError(f"Expected at least 2 wall zones after inlet/outlet classification, got {remaining}")

    # Connected-trapezoid geometries can create many wall segments.
    # Split them by centroid y into bottom/top groups.
    ys = [z["cy"] for z in remaining]
    y_mid = 0.5 * (min(ys) + max(ys))
    wall_bottom_zones = [z["name"] for z in remaining if z["cy"] <= y_mid]
    wall_top_zones = [z["name"] for z in remaining if z["cy"] > y_mid]

    # Fallback if centroids cluster awkwardly.
    if not wall_bottom_zones or not wall_top_zones:
        rem_sorted = sorted(remaining, key=lambda z: z["cy"])
        split_idx = max(1, len(rem_sorted) // 2)
        wall_bottom_zones = [z["name"] for z in rem_sorted[:split_idx]]
        wall_top_zones = [z["name"] for z in rem_sorted[split_idx:]]
        if not wall_top_zones:
            wall_top_zones = [wall_bottom_zones.pop()]

    wall_bottom = wall_bottom_zones[0]
    wall_top = wall_top_zones[0]

    result = {
        "inlet": inlet["name"],
        "outlet": outlet["name"],
        "wall_bottom": wall_bottom,
        "wall_top": wall_top,
        "inlet_zones": [inlet["name"]],
        "outlet_zones": [outlet["name"]],
        "wall_bottom_zones": wall_bottom_zones,
        "wall_top_zones": wall_top_zones,
    }

    log(f"[INFO] Classified boundaries: {result}")
    return result


def compute_physics_checks(
    uin_mps: float,
    vin_mps: float,
    vout_mps: float,
    pin_pa: float,
    pout_pa: float,
    inlet_height_m: float | None,
    outlet_height_m: float | None,
    density: float = AIR_DENSITY,
    viscosity: float = AIR_VISCOSITY,
) -> dict[str, Any]:
    """
    Compute real-world physics sanity checks for a converged 2D channel case:
      - pressure drop sign
      - mass conservation (volumetric flux per unit depth at inlet vs outlet)
      - Reynolds number and laminar-model validity
    """
    checks: dict[str, Any] = {}

    dp = pin_pa - pout_pa
    checks["dp_pa"] = dp
    checks["dp_positive"] = bool(dp > 0.0)

    # Reference velocity for flux/Reynolds: measured inlet velocity magnitude.
    u_ref = float(vin_mps) if vin_mps else float(uin_mps)

    # Mass conservation: in 2D, flux per unit depth = U_avg * height.
    if inlet_height_m and outlet_height_m:
        flux_in = u_ref * inlet_height_m
        flux_out = float(vout_mps) * outlet_height_m
        checks["mass_flux_in_per_depth_m2ps"] = flux_in
        checks["mass_flux_out_per_depth_m2ps"] = flux_out
        if flux_in != 0.0:
            rel_err = abs(flux_in - flux_out) / abs(flux_in)
            checks["mass_conservation_rel_error"] = rel_err
            checks["mass_conserved"] = bool(rel_err < MASS_CONSERVATION_REL_TOL)

    # Reynolds number based on inlet hydraulic diameter (parallel plates: D_h = 2*h).
    if inlet_height_m:
        d_h = 2.0 * inlet_height_m
        reynolds = density * u_ref * d_h / viscosity
        checks["reynolds_number"] = reynolds
        if reynolds < REYNOLDS_LAMINAR_LIMIT:
            regime = "laminar"
        elif reynolds < 2.0 * REYNOLDS_LAMINAR_LIMIT:
            regime = "transitional"
        else:
            regime = "turbulent"
        checks["flow_regime"] = regime
        checks["laminar_assumption_valid"] = bool(reynolds < REYNOLDS_LAMINAR_LIMIT)

    checks["physics_ok"] = bool(
        checks.get("dp_positive", False)
        and checks.get("mass_conserved", True)
        and checks.get("laminar_assumption_valid", True)
    )

    log(
        "[INFO] Physics checks: "
        f"dp={dp:.6g} Pa (positive={checks['dp_positive']}), "
        f"Re={checks.get('reynolds_number', float('nan')):.4g} "
        f"({checks.get('flow_regime', 'n/a')}), "
        f"mass_err={checks.get('mass_conservation_rel_error', float('nan')):.4g}, "
        f"physics_ok={checks['physics_ok']}"
    )
    return checks


def _zone_list(zones: str | list[str]) -> list[str]:
    return [zones] if isinstance(zones, str) else [str(z) for z in zones]


def convert_boundary_types(solver, inlet_zone: str | list[str], outlet_zone: str | list[str]) -> None:
    bc = solver.settings.setup.boundary_conditions

    inlet_list = _zone_list(inlet_zone)
    outlet_list = _zone_list(outlet_zone)

    bc.set_zone_type(zone_list=inlet_list, new_type="velocity-inlet")
    bc.set_zone_type(zone_list=outlet_list, new_type="pressure-outlet")

    log(f"[INFO] Converted {inlet_list} -> velocity-inlet")
    log(f"[INFO] Converted {outlet_list} -> pressure-outlet")

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
        log(
            f"[INFO] Residual targets set via settings API: "
            f"continuity={continuity}, x={x_velocity}, y={y_velocity}"
        )
        return
    except Exception as e:
        tried.append(f"settings residual equations failed: {e}")

    try:
        solver.tui.solve.monitors.residual.convergence_criteria(
            str(continuity),
            str(x_velocity),
            str(y_velocity),
        )
        log("[INFO] Residual targets set via TUI")
        return
    except Exception as e:
        tried.append(f"TUI convergence_criteria failed: {e}")

    log("[WARN] Could not set residual targets:\n- " + "\n- ".join(tried))

def set_velocity_inlet(
    solver,
    inlet_name: str,
    uin_mps: float,
    inlet_velocity_profile: list[dict[str, Any]] | None = None,
    temp_K: float = 300.0,
) -> None:

    inlet = solver.settings.setup.boundary_conditions.velocity_inlet[inlet_name]

    # Signature compatibility only (energy equation is disabled).
    _ = temp_K

    # Apply the actual nonuniform inlet profile when available.
    if inlet_velocity_profile and len(inlet_velocity_profile) >= 2:
        try:
            pts = sorted(
                [
                    (float(p["y_mm"]) / 1000.0, float(p["u_mps"]))
                    for p in inlet_velocity_profile
                ],
                key=lambda t: t[0],
            )
        except Exception as e:
            raise RuntimeError(f"Invalid inlet_velocity_profile format: {e}")

        profile_name = f"{inlet_name.replace('.', '_')}_u_profile"

        # with tempfile.NamedTemporaryFile(mode="w", suffix=".prof", delete=False) as tf:
        with open("./profile.prof", "w") as tf:
            prof_path = Path(tf.name)
            tf.write(f"(({profile_name} line {len(pts)})\n")
            tf.write("(x\n")
            for _y_m, _u in pts:
                tf.write("0.0\n")
            tf.write(")\n")
            tf.write("(y\n")
            for y_m, _u in pts:
                tf.write(f"{y_m:.16e}\n")
            tf.write(")\n")
            tf.write("(velocity-magnitude\n")
            for _y_m, u in pts:
                tf.write(f"{u:.16e}\n")
            tf.write(")\n")
            tf.write(")\n")

        try:
            solver.tui.file.read_profile(str(prof_path))
            log(f"[INFO] Read Fluent profile file {prof_path}")
            log(f"[INFO] profile: {solver.tui.define.profiles.list_profiles()}")
        except Exception as e:
            raise RuntimeError(f"Failed to read Fluent profile file {prof_path}: {e}")

        bind_errors: list[str] = []
        bound = False

        try:
            inlet.momentum.velocity_magnitude.option = "profile"
            inlet.momentum.velocity_magnitude.profile_name = profile_name
            inlet.momentum.velocity_magnitude.field_name = "velocity-magnitude"
            bound = True
        except Exception as e:
            log(f"[WARN] option/profile_name API failed: {e}")
            bind_errors.append(f"option/profile_name API failed: {e}")

        if not bound:
            try:
                inlet.momentum.velocity_magnitude.set_state(
                    {"option": "profile", "profile_name": profile_name, "field_name": "velocity-magnitude"}
                )
                bound = True
            except Exception as e:
                log(f"[WARN] set_state profile mapping failed: {e}")
                bind_errors.append(f"set_state profile mapping failed: {e}")

        if not bound:
            raise RuntimeError(
                "Loaded inlet profile but could not bind it to velocity inlet. Tried:\n- "
                + "\n- ".join(bind_errors)
            )

        log(
            f"[INFO] Set velocity inlet '{inlet_name}' from profile '{profile_name}' "
            f"with {len(pts)} points."
        )
        return

    # Scalar fallback only when profile data is unavailable.
    inlet.momentum.velocity_magnitude.value = float(uin_mps)
    log(f"[INFO] Set velocity inlet '{inlet_name}' = {uin_mps} m/s (scalar)")


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
    # Signature kept for compatibility; thermal wall BC is intentionally unused.
    _ = (solver, wall_name, temp_K)
    log(f"[INFO] Skipping wall temperature setup for '{wall_name}' (energy equation disabled)")

def initialize_solution(solver, inlet_zone_name: str) -> None:
    try:
        solver.tui.solve.initialize.hyb_initialization()
        log("[INFO] Hybrid initialization completed")
        inlet_vel = solver.tui.report.surface_integrals.area_weighted_average(
            zone_name=inlet_zone_name,
            quantity="velocity-magnitude",
        )
        log(f"[INFO] check inlet velocity: {inlet_vel}")
    except Exception:
        solver.tui.solve.initialize.initialize_flow()
        log("[INFO] Standard initialization completed")


def iterate_solver(solver, n_iter: int) -> None:
    solver.tui.solve.iterate(int(n_iter))
    log(f"[INFO] Solver iterated for {n_iter} steps")


def report_area_weighted_quantity(
    solver,
    zone_name: str | list[str],
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

    zone_names = _zone_list(zone_name)

    for report_of in report_candidates:
        try:
            results.report.surface_integrals.area_weighted_avg(
                surface_names=zone_names,
                report_of=report_of,
                write_to_file=True,
                file_name=str(out_txt),
            )

            val = parse_scalar_file(out_txt)
            if val is None:
                raise RuntimeError(f"Could not parse numeric value from {out_txt}")

            log(
                f"[INFO] Wrote area-weighted {quantity_label} for zone(s) {zone_names} "
                f"using report_of='{report_of}' -> {out_txt} : {val:.12e}"
            )
            return val

        except Exception as e:
            tried.append(f"report_of='{report_of}': {e}")

    raise RuntimeError(
        f"Failed {quantity_label} report on zone(s) {zone_names}. Tried:\n- "
        + "\n- ".join(tried)
    )

def report_area_weighted_pressure(solver, zone_name: str | list[str], out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["pressure", "static-pressure"],
        quantity_label="pressure",
    )

def report_area_weighted_temperature(solver, zone_name: str | list[str], out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["temperature", "static-temperature"],
        quantity_label="temperature",
    )

def report_area_weighted_velocity(solver, zone_name: str | list[str], out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["velocity-magnitude"],
        quantity_label="velocity",
    )

def report_area_weighted_x_velocity(solver, zone_name: str | list[str], out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["x-velocity"],
        quantity_label="x-velocity",
    )

def report_area_weighted_y_velocity(solver, zone_name: str | list[str], out_txt: Path) -> float:
    return report_area_weighted_quantity(
        solver=solver,
        zone_name=zone_name,
        out_txt=out_txt,
        report_candidates=["y-velocity"],
        quantity_label="y-velocity",
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
    for continuity, x-velocity, y-velocity.
    """
    ensure_dir(out_dir)

    try:
        history = solver.monitors.get_monitor_set_data("residual")
    except Exception as e:
        log(f"[WARN] Could not access residual monitor history: {e}")
        return

    df = pd.DataFrame(history)

    mapping = {
        "continuity": "pressure.csv",
        "x-velocity": "x-velocity.csv",
        "y-velocity": "y-velocity.csv",
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
    inlet_velocity_profile: list[dict[str, Any]] | None,
    nprocs: int,
    n_iter: int,
    lx_m: float,
    ly_m: float,
    inlet_height_m: float | None = None,
    outlet_height_m: float | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)

    solver = None
    try:
        solver = launch_solver(nprocs=nprocs)

        solver.settings.file.read_mesh(file_name=str(mesh_path))
        log(f"[INFO] Read mesh: {mesh_path}")

        mesh_check(solver)
        set_models_and_materials(solver)

        boundary_map = classify_channel_boundaries(
            solver,
            lx_m=lx_m,
            ly_m=ly_m,
        )

        convert_boundary_types(
            solver,
            inlet_zone=boundary_map["inlet_zones"],
            outlet_zone=boundary_map["outlet_zones"],
        )

        set_velocity_inlet(
            solver,
            inlet_name=boundary_map["inlet"],
            uin_mps=uin_mps,
            inlet_velocity_profile=inlet_velocity_profile,
        )
        set_pressure_outlet(solver, outlet_name=boundary_map["outlet"], gauge_pressure_pa=0.0)

        initialize_solution(solver, inlet_zone_name=boundary_map["inlet"])
        set_residual_targets(
            solver,
            continuity=1e-20,
            x_velocity=1e-20,
            y_velocity=1e-20,
        )
        iterate_solver(solver, n_iter=n_iter)

        residual_csv = out_dir / "residuals.csv"
        residual_png = out_dir / "residuals.png"
        write_residual_csv_from_monitor(solver, residual_csv)
        save_residual_plot(residual_csv, residual_png)
        write_individual_residual_csvs(solver, out_dir)

        pin_txt = out_dir / "pin.txt"
        pout_txt = out_dir / "pout.txt"
        vin_txt = out_dir / "vin.txt"
        vout_txt = out_dir / "vout.txt"
        vout_x_txt = out_dir / "vout_x.txt"
        vout_y_txt = out_dir / "vout_y.txt"
        vin_x_txt = out_dir / "vin_x.txt"
        vin_y_txt = out_dir / "vin_y.txt"
        case_data_path = out_dir / "case2d.cas.h5"

        pin = report_area_weighted_pressure(solver, boundary_map["inlet_zones"], pin_txt)
        pout = report_area_weighted_pressure(solver, boundary_map["outlet_zones"], pout_txt)
        dp = pin - pout

        vin = report_area_weighted_velocity(solver, boundary_map["inlet_zones"], vin_txt)
        vin_x = report_area_weighted_x_velocity(solver, boundary_map["inlet_zones"], vin_x_txt)
        vin_y = report_area_weighted_y_velocity(solver, boundary_map["inlet_zones"], vin_y_txt)
        vout = report_area_weighted_velocity(solver, boundary_map["outlet_zones"], vout_txt)
        vout_x = report_area_weighted_x_velocity(solver, boundary_map["outlet_zones"], vout_x_txt)
        vout_y = report_area_weighted_y_velocity(solver, boundary_map["outlet_zones"], vout_y_txt)

        physics_checks = compute_physics_checks(
            uin_mps=uin_mps,
            vin_mps=vin_x,
            vout_mps=vout_x,
            pin_pa=pin,
            pout_pa=pout,
            inlet_height_m=inlet_height_m,
            outlet_height_m=outlet_height_m,
        )

        postprocess_summary = {
            "case": mesh_path.stem.replace(".msh", ""),
            "boundaries": {
                "inlet": boundary_map["inlet"],
                "outlet": boundary_map["outlet"],
                "wall_top": boundary_map["wall_top"],
                "wall_bottom": boundary_map["wall_bottom"],
                "inlet_zones": boundary_map.get("inlet_zones", [boundary_map["inlet"]]),
                "outlet_zones": boundary_map.get("outlet_zones", [boundary_map["outlet"]]),
                "wall_top_zones": boundary_map.get("wall_top_zones", [boundary_map["wall_top"]]),
                "wall_bottom_zones": boundary_map.get("wall_bottom_zones", [boundary_map["wall_bottom"]]),
            },
            "inlet": {
                "pressure_pa": pin,
                "velocity_mps": vin,
                "txt_files": {
                    "pressure": str(pin_txt),
                    "velocity": str(vin_txt),
                    "x-velocity": str(vin_x_txt),
                    "y-velocity": str(vin_y_txt),
                },
            },
            "outlet": {
                "pressure_pa": pout,
                "velocity_mps": vout,
                "txt_files": {
                    "pressure": str(pout_txt),
                    "velocity": str(vout_txt),
                    "x-velocity": str(vout_x_txt),
                    "y-velocity": str(vout_y_txt),
                },
            },
            "derived": {
                "dp_pa": dp,
            },
            "physics_checks": physics_checks,
            "residual_files": {
                "combined_csv": str(residual_csv),
                "plot_png": str(residual_png),
                "pressure_csv": str(out_dir / "pressure.csv"),
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
            "vin_mps": vin,
            "vin_x_mps": vin_x,
            "vin_y_mps": vin_y,
            "vout_mps": vout,
            "vout_x_mps": vout_x,
            "vout_y_mps": vout_y,
            "vin_txt": str(vin_txt),
            "vout_txt": str(vout_txt),
            "reynolds_number": physics_checks.get("reynolds_number", ""),
            "flow_regime": physics_checks.get("flow_regime", ""),
            "mass_conservation_rel_error": physics_checks.get("mass_conservation_rel_error", ""),
            "physics_ok": physics_checks.get("physics_ok", ""),
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
    global_max_size_mm: float = 5e-3,
    global_min_size_mm: float = 1e-3,
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
    uin_for_run = float(uin_mps if uin_mps is not None else spec_info["uin_mps"])
    inlet_velocity_profile = spec_info.get("inlet_velocity_profile", [])

    mesh_path = out_dir / f"{case}.msh.h5"
    summary_json = out_dir / "run_summary.json"

    started = time.time()
    
    boundary_layers = [
        {
            "name": "inflation",
            "n_layers": 5,
            "growth_rate": 1.2,
            "offset_method": "smooth-transition",
        }
    ]

    mesh_2d_from_step(
        step_path=step_path,
        mesh_path=mesh_path,
        nprocs=nprocs,
        max_size_mm=global_max_size_mm,
        min_size_mm=global_min_size_mm,
        boundary_layers=boundary_layers,
    )

    solve_result = solve_2d_mesh(
        mesh_path=mesh_path,
        out_dir=out_dir,
        uin_mps=uin_for_run,
        inlet_velocity_profile=inlet_velocity_profile,
        nprocs=nprocs,
        n_iter=n_iter,
        lx_m=spec_info["lx_m"],
        ly_m=spec_info["ly_m"],
        inlet_height_m=spec_info.get("inlet_height_m"),
        outlet_height_m=spec_info.get("outlet_height_m"),
    )

    result = {
        "case": case,
        "status": "ok",
        "spec_json": str(spec_json),
        "step_path": str(step_path),
        "uin_mps": uin_for_run,
        "inlet_profile_points": len(inlet_velocity_profile),
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
    uin_raw = row.get("Uin_mps")
    uin = float(uin_raw) if (uin_raw is not None and str(uin_raw).strip() != "") else None

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
            "uin_mps": "" if uin is None else uin,
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
            "vin_mps",
            "vout_mps",
            "reynolds_number",
            "flow_regime",
            "mass_conservation_rel_error",
            "physics_ok",
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
    p.add_argument("--max-size-mm", type=float, default=0.2, help="Maximum global mesh element size in mm. Smaller values produce finer meshes (more elements). Default: 0.2 mm.")
    p.add_argument("--min-size-mm", type=float, default=0.02, help="Minimum mesh element size in mm used for curvature/feature refinement. Must be smaller than --max-size-mm. Default: 0.02 mm.")

    return p


def main():
    args = build_argparser().parse_args()
    
    ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"
    os.environ.setdefault("AWP_ROOT251", ANSYS_ROOT_V251)

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