import os
import json
from pathlib import Path
import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion
import ansys.fluent.core as pyfluent

ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"


def launch_meshing(nprocs=4):
    os.environ.setdefault("AWP_ROOT251", ANSYS_ROOT_V251)
    return pyfluent.launch_fluent(
        product_version=FluentVersion.v251,
        mode=pyfluent.FluentMode.MESHING,
        processor_count=nprocs,
        ui_mode="no_gui",
        precision="double",
    )


def safe_exit(session):
    for m in ("exit", "close"):
        if hasattr(session, m):
            try:
                getattr(session, m)()
                return
            except Exception:
                pass

def set_one_zone_per_face(wt):
    candidates = [
        ("import_geometry.cad_options.one_zone_per", "face"),
        ("import_geometry.cad_options.create_zones_per", "face"),
        ("import_geometry.cad_import_options.one_zone_per", "face"),
        ("import_geometry.cad_import_options.create_zones_per", "face"),
        ("import_geometry.import_options.one_zone_per", "face"),
        ("import_geometry.import_options.create_zones_per", "face"),
        ("import_geometry.cad_options.zone_method", "one-zone-per-face"),
    ]
    for path, val in candidates:
        obj = wt
        try:
            for token in path.split("."):
                obj = getattr(obj, token)
            obj.set_state(val)
            print(f"[WT] set {path} = {val}")
            return
        except Exception:
            pass

    try:
        print("[WT] import_geometry.Arguments.get_state():")
        print(wt.import_geometry.Arguments.get_state())
    except Exception:
        print("[WT] failed to print import_geometry.Arguments.get_state()")
    raise RuntimeError("Cannot find CAD import option for 'one zone per face' on this build.")


def mesh_step_to_solver(step_path: Path, nprocs=4,
                        length_unit="mm",
                        surf_max_size=0.3,
                        vol_fill="polyhedra",   # polyhedra, poly-hexcore, hexcore, tetrahedral
                        hex_max_cell_length=0.3):
    meshing = launch_meshing(nprocs=nprocs)
    wt = meshing.watertight()

    def step(msg, fn):
        print(f"[WT] {msg}", flush=True)
        return fn()

    set_one_zone_per_face(wt)

    step("import_geometry.set_state", lambda: (
        wt.import_geometry.file_name.set_state(str(step_path)),
        wt.import_geometry.length_unit.set_state(length_unit),
    ))
    step("import_geometry.execute", wt.import_geometry)

    # ---- Surface mesh
    step("create_surface_mesh.set_state", lambda: (
        wt.create_surface_mesh.cfd_surface_mesh_controls.max_size.set_state(float(surf_max_size)),
    ))
    step("create_surface_mesh.execute", wt.create_surface_mesh)

    # ---- Describe geometry
    step("describe_geometry.update_child_tasks(False)", lambda: (
        wt.describe_geometry.update_child_tasks(setup_type_changed=False),
    ))
    step("describe_geometry.set_setup_type", lambda: (
        wt.describe_geometry.setup_type.set_state(
            "The geometry consists of only fluid regions with no voids"
        ),
    ))
    step("describe_geometry.update_child_tasks(True)", lambda: (
        wt.describe_geometry.update_child_tasks(setup_type_changed=True),
    ))
    step("describe_geometry.execute", wt.describe_geometry)

    # ---- Boundaries/regions
    step("update_boundaries", wt.update_boundaries)
    step("update_regions", wt.update_regions)

    # ---- Volume mesh
    allowed = {"polyhedra", "poly-hexcore", "hexcore", "tetrahedral"}
    if vol_fill not in allowed:
        raise ValueError(f"vol_fill must be one of {sorted(allowed)}")

    step(f"create_volume_mesh.volume_fill={vol_fill}", lambda: (
        wt.create_volume_mesh.volume_fill.set_state(vol_fill),
    ))
    try:
        wt.create_volume_mesh.volume_fill_controls.hex_max_cell_length.set_state(float(hex_max_cell_length))
    except Exception:
        pass

    step("create_volume_mesh.execute", wt.create_volume_mesh)

    # ---- Switch to solver
    solver = meshing.switch_to_solver()

    # ---- Debug print boundary zones
    bcs = solver.settings.setup.boundary_conditions
    print("\n=== boundary_conditions keys ===")
    print(list(bcs.get_state().keys()))
    print("=== surfaces_info ===")
    print(solver.fields.field_info.get_surfaces_info())
    print("======================\n")

    return meshing, solver

def list_zones(solver):
    """Print zone names once to confirm inlet/outlet/wall names."""
    bcs = solver.settings.setup.boundary_conditions
    bcs.set_zone_type(zone_name="inlet", new_type="velocity-inlet")
    bcs.set_zone_type(zone_name="outlet", new_type="pressure-outlet")
    print("BC zones:", list(bcs.get_state().keys()))


def set_simple_laminar_incompressible(solver):
    bc = solver.settings.setup.boundary_conditions

    # Example velocity inlet
    bc.velocity_inlet["inlet"] = {
        "momentum": {
            "velocity_specification_method": "Magnitude, Normal to Boundary",
            "velocity": {"value": float(uin)},
        }
    }

    # Example pressure outlet
    bc.pressure_outlet["outlet"] = {
        "pressure": {"value": 0.0}
    }

def run_case(step_path: Path, out_dir: Path, uin: float, nprocs=4,
             inlet_name="inlet", outlet_name="outlet", n_iter=200):
    out_dir.mkdir(parents=True, exist_ok=True)

    meshing = None
    solver = None
    try:
        meshing, solver = mesh_step_to_solver(step_path, nprocs=nprocs)

        # DEBUG: print available surface names once
        print("Available surfaces:")
        print(solver.fields.field_info.get_surfaces_info())

        set_simple_laminar_incompressible(solver)
        apply_bcs_velocity_inlet_pressure_outlet(solver, inlet_name, outlet_name, uin)

        solver.settings.solution.initialization.hybrid_initialize()
        solver.settings.solution.run_calculation.iter_count = int(n_iter)
        solver.settings.solution.run_calculation.iterate()

        # Save case/data
        solver.settings.file.write_case(file_name=str(out_dir / "case.cas.h5"))
        solver.settings.file.write_data(file_name=str(out_dir / "data.dat.h5"))

        results = {
            "step": str(step_path),
            "uin": float(uin),
            "n_iter": int(n_iter),
            "inlet": inlet_name,
            "outlet": outlet_name,
        }
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        return results

    finally:
        if solver is not None:
            safe_exit(solver)
        if meshing is not None:
            safe_exit(meshing)

def compute_metrics(solver, inlet_name: str, outlet_name: str):
    """
    Returns:
      p_in, p_out: area-averaged absolute pressure [Pa] (Fluent internal units)
      dp: p_in - p_out
      mdot_in, mdot_out: mass flow rate [kg/s] (sign depends on surface normal)
      mdot_in_abs, mdot_out_abs: abs values
    """
    red = solver.fields.reduction

    inlet_loc = inlet_name
    outlet_loc = outlet_name
    try:
        inlet_loc = pyfluent.VelocityInlets(solver)[inlet_name]
    except Exception:
        pass
    try:
        outlet_loc = pyfluent.PressureOutlets(solver)[outlet_name]
    except Exception:
        pass

    # Area-averaged pressure on inlet/outlet
    p_in = float(red.area_average(expression="AbsolutePressure", locations=[inlet_loc]))
    p_out = float(red.area_average(expression="AbsolutePressure", locations=[outlet_loc]))
    dp = p_in - p_out

    # Mass flow rate through surface:
    # reduction supports mass_flow_integral(expression, locations)
    mdot_in = float(red.mass_flow_integral(expression="1", locations=[inlet_loc]))
    mdot_out = float(red.mass_flow_integral(expression="1", locations=[outlet_loc]))

    return {
        "p_in_Pa": p_in,
        "p_out_Pa": p_out,
        "dp_Pa": dp,
        "mdot_in_kgps": mdot_in,
        "mdot_out_kgps": mdot_out,
        "mdot_in_abs_kgps": abs(mdot_in),
        "mdot_out_abs_kgps": abs(mdot_out),
    }