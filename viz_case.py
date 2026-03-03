import os
import argparse
from pathlib import Path

import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion
from ansys.fluent.core.solver import Graphics, Contour 

ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"


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


def pick_display_surface(surfaces: list[str]) -> str:
    for s in surfaces:
        if s.startswith("interior-"):
            return s

    for cand in ("fluid", "interior", "domain"):
        if cand in surfaces:
            return cand
    # last resort
    return surfaces[0] if surfaces else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Folder containing case.cas.h5 and data.dat.h5")
    ap.add_argument("--case", type=str, default=None, help="Path to .cas.h5")
    ap.add_argument("--data", type=str, default=None, help="Path to .dat.h5")
    ap.add_argument("--out_dir", type=str, default=None, help="Where to write PNGs (default: run_dir)")
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--surface", type=str, default=None, help="Surface name to plot on (optional)")
    ap.add_argument("--list_surfaces", action="store_true")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        case_path = run_dir / "case.cas.h5"
        data_path = run_dir / "data.dat.h5"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir
    else:
        if not args.case:
            raise SystemExit("Provide --run_dir OR --case (and optionally --data).")
        case_path = Path(args.case)
        data_path = Path(args.data) if args.data else None
        out_dir = Path(args.out_dir) if args.out_dir else case_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    solver = None
    try:
        solver = launch_solver_2d(nprocs=args.nprocs)

        # Load case
        solver.settings.file.read_case(file_name=str(case_path))
        if data_path and data_path.is_file():
            solver.settings.file.read_data(file_name=str(data_path))

        # Get surface names
        surfaces = solver.field_data.surfaces()
        if args.list_surfaces:
            print("=== surfaces() ===")
            for s in surfaces:
                print(s)

        surf = args.surface or pick_display_surface(surfaces)
        if not surf:
            raise RuntimeError("No surfaces found to plot. Try --list_surfaces.")

        graphics = Graphics(solver)
        graphics.picture.x_resolution = 1200
        graphics.picture.y_resolution = 450  

        # ---- Pressure contour (Absolute Pressure)
        p = Contour(solver, new_instance_name="p_contour")
        p.field = "absolute-pressure"
        p.surfaces_list = [surf]
        p.display()
        graphics.picture.save_picture(file_name=str(out_dir / "pressure.png"))

        # ---- Velocity magnitude contour
        v = Contour(solver, new_instance_name="v_contour")
        v.field = "velocity-magnitude"
        v.surfaces_list = [surf]
        v.display()
        graphics.picture.save_picture(file_name=str(out_dir / "velocity_mag.png"))

        print(f"[OK] Wrote: {out_dir/'pressure.png'}")
        print(f"[OK] Wrote: {out_dir/'velocity_mag.png'}")
        print(f"[INFO] Plotted on surface: {surf}")

    finally:
        if solver is not None:
            try:
                solver.exit()
            except Exception:
                pass


if __name__ == "__main__":
    main()