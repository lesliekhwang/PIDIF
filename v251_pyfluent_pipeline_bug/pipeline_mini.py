import os
import json
from pathlib import Path
import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion

# CONFIG
ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"
BASE_CASE = Path("/home/nuoxu9/PIDIF/base/base_case.cas.h5")

# Utilities

def safe_exit(session):
    try:
        session.exit()
    except Exception:
        pass


def launch_solver_2d(nprocs=4):
    os.environ.setdefault("AWP_ROOT251", ANSYS_ROOT_V251)
    return pyfluent.launch_fluent(
        product_version=FluentVersion.v251,
        mode=pyfluent.FluentMode.SOLVER,
        processor_count=nprocs,
        ui_mode="no_gui",
        precision=pyfluent.Precision.DOUBLE,
        dimension=pyfluent.Dimension.TWO,
    )


def compute_metrics(solver, inlet_name="inlet", outlet_name="outlet"):
    red = solver.fields.reduction

    p_in = float(
        red.area_average(expression="AbsolutePressure",
                         locations=[inlet_name])
    )
    p_out = float(
        red.area_average(expression="AbsolutePressure",
                         locations=[outlet_name])
    )
    dp = p_in - p_out

    mdot_in = float(
        red.mass_flow_integral(expression="1",
                               locations=[inlet_name])
    )
    mdot_out = float(
        red.mass_flow_integral(expression="1",
                               locations=[outlet_name])
    )

    return {
        "p_in_Pa": p_in,
        "p_out_Pa": p_out,
        "dp_Pa": dp,
        "mdot_in_kgps": mdot_in,
        "mdot_out_kgps": mdot_out,
        "mdot_in_abs_kgps": abs(mdot_in),
        "mdot_out_abs_kgps": abs(mdot_out),
    }


# Phase 1 runner
def run_case_phase1(out_dir: Path,
                    uin: float,
                    nprocs=4,
                    inlet_name="inlet",
                    outlet_name="outlet",
                    n_iter=200):

    out_dir.mkdir(parents=True, exist_ok=True)

    solver = None
    try:
        solver = launch_solver_2d(nprocs=nprocs)

        # Read baseline case
        solver.settings.file.read_case(file_name=str(BASE_CASE))

        # Set inlet velocity
        bc = solver.settings.setup.boundary_conditions
        bc.velocity_inlet[inlet_name] = {
            "momentum": {
                "velocity_specification_method": "Magnitude, Normal to Boundary",
                "velocity": {"value": float(uin)},
            }
        }

        # Initialize + iterate
        solver.settings.solution.initialization.hybrid_initialize()
        solver.settings.solution.run_calculation.iter_count = int(n_iter)
        solver.settings.solution.run_calculation.iterate()

        # Metrics
        metrics = compute_metrics(solver,
                                  inlet_name=inlet_name,
                                  outlet_name=outlet_name)

        # Save
        solver.settings.file.write_case(file_name=str(out_dir / "case.cas.h5"))
        solver.settings.file.write_data(file_name=str(out_dir / "data.dat.h5"))

        results = {
            "uin": float(uin),
            "n_iter": int(n_iter),
            "inlet": inlet_name,
            "outlet": outlet_name,
            **metrics,
        }

        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2)
        )

        return results

    finally:
        if solver is not None:
            safe_exit(solver)