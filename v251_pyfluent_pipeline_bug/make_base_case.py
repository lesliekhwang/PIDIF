import os
from pathlib import Path
import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion

ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"

MESH_PATH = Path("/home/nuoxu9/PIDIF/v231/channel_00.msh")
OUT_CASE  = Path("/home/nuoxu9/PIDIF/base/base_case.cas.h5")

def launch_solver(nprocs=4, dim=pyfluent.Dimension.TWO):
    os.environ.setdefault("AWP_ROOT251", ANSYS_ROOT_V251)
    return pyfluent.launch_fluent(
        product_version=FluentVersion.v251,
        mode=pyfluent.FluentMode.SOLVER,
        processor_count=nprocs,
        ui_mode="no_gui",
        precision=pyfluent.Precision.DOUBLE,
        dimension=dim,  # <-- critical for 2D meshes :contentReference[oaicite:1]{index=1}
    )

def main():
    OUT_CASE.parent.mkdir(parents=True, exist_ok=True)
    assert MESH_PATH.is_file(), f"Missing mesh: {MESH_PATH}"

    # 1) Launch 2D solver (your mesh is dimension 2)
    solver = launch_solver(nprocs=4, dim=pyfluent.Dimension.TWO)

    try:
        # 2) Read mesh
        solver.settings.file.read_mesh(file_name=str(MESH_PATH))

        # 3) List available boundary surfaces (modern API)
        #    This avoids deprecated field_info and works after mesh load.
        surfs = solver.field_data.surfaces()
        print("=== surfaces() ===")
        for s in surfs:
            print(s)

        # 4) Save baseline case (even before you set BC types/values)
        solver.settings.file.write_case(file_name=str(OUT_CASE))
        print("Wrote:", OUT_CASE)

    finally:
        try:
            solver.exit()
        except Exception:
            pass

if __name__ == "__main__":
    main()