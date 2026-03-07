# visualize_case_2d.py

import os
import argparse
from pathlib import Path

import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion
from ansys.fluent.core.solver import Graphics, Contour

ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"


def launch_solver_2d(nprocs: int = 4):
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
    # Prefer interior/domain-like surfaces for full-field contours
    for s in surfaces:
        s_str = str(s)
        if s_str.startswith("interior:") or s_str.startswith("interior-"):
            return s_str

    for cand in ("fluid", "interior", "domain"):
        if cand in surfaces:
            return cand

    return str(surfaces[0]) if surfaces else ""


def read_case_and_data(solver, case_path: Path, data_path: Path | None):
    solver.settings.file.read_case(file_name=str(case_path))
    if data_path and data_path.is_file():
        solver.settings.file.read_data(file_name=str(data_path))


def save_contour(solver, graphics, surface_name: str, field_name: str, out_png: Path, instance_name: str):
    contour = Contour(solver, new_instance_name=instance_name)
    contour.field = field_name
    contour.surfaces_list = [surface_name]
    contour.display()
    graphics.picture.save_picture(file_name=str(out_png))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="/home/nuoxu9/PIDIF/runs_2d/channel_00",
        help="Folder containing the solved case/data files.",
    )
    ap.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional explicit path to .cas.h5 file.",
    )
    ap.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional explicit path to .dat.h5 file.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write PNGs. Default: run_dir",
    )
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument(
        "--surface",
        type=str,
        default=None,
        help="Surface name to plot on. Default: auto-detect.",
    )
    ap.add_argument("--list_surfaces", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case:
        case_path = Path(args.case)
    else:
        # Your current pipeline writes combined case+data to case2d.cas.h5
        case_candidates = [
            run_dir / "case2d.cas.h5",
            run_dir / "case.cas.h5",
        ]
        case_path = next((p for p in case_candidates if p.is_file()), None)
        if case_path is None:
            raise SystemExit(
                f"Could not find case file in {run_dir}. "
                f"Tried: {[str(p) for p in case_candidates]}"
            )

    if args.data:
        data_path = Path(args.data)
    else:
        # Optional separate data file
        data_candidates = [
            run_dir / "data.dat.h5",
            run_dir / "case2d.dat.h5",
        ]
        data_path = next((p for p in data_candidates if p.is_file()), None)

    solver = None
    try:
        solver = launch_solver_2d(nprocs=args.nprocs)
        read_case_and_data(solver, case_path=case_path, data_path=data_path)

        surfaces = solver.field_data.surfaces()
        if isinstance(surfaces, dict):
            surface_names = list(surfaces.keys())
        else:
            surface_names = list(surfaces)

        if args.list_surfaces:
            print("=== surfaces() ===")
            for s in surface_names:
                print(s)

        surf = args.surface or pick_display_surface(surface_names)
        if not surf:
            raise RuntimeError("No surfaces found to plot. Try --list_surfaces.")

        graphics = Graphics(solver)
        graphics.picture.x_resolution = 1400
        graphics.picture.y_resolution = 500

        # Pressure
        pressure_png = out_dir / "pressure.png"
        save_contour(
            solver=solver,
            graphics=graphics,
            surface_name=surf,
            field_name="absolute-pressure",
            out_png=pressure_png,
            instance_name="pressure_contour",
        )

        # Velocity magnitude
        velocity_png = out_dir / "velocity_mag.png"
        save_contour(
            solver=solver,
            graphics=graphics,
            surface_name=surf,
            field_name="velocity-magnitude",
            out_png=velocity_png,
            instance_name="velocity_contour",
        )

        # Temperature
        temperature_png = out_dir / "temperature.png"
        save_contour(
            solver=solver,
            graphics=graphics,
            surface_name=surf,
            field_name="temperature",
            out_png=temperature_png,
            instance_name="temperature_contour",
        )

        print(f"[OK] Wrote: {pressure_png}")
        print(f"[OK] Wrote: {velocity_png}")
        print(f"[OK] Wrote: {temperature_png}")
        print(f"[INFO] Plotted on surface: {surf}")

    finally:
        if solver is not None:
            try:
                solver.exit()
            except Exception:
                pass


if __name__ == "__main__":
    main()