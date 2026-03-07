# generate_2d_geometry_params.py

import csv
import json
import math
from pathlib import Path

import numpy as np


# =========================================
# CONFIG
# =========================================
BASE_DIR = Path("/home/nuoxu9/PIDIF")
OUT_DIR = BASE_DIR / "2d_geometry_specs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEP_DIR = BASE_DIR / "2d_geometry_step"
CSV_PATH = OUT_DIR / "designs.csv"

# reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# channel size in mm
Lx = 50.0
Ly = 20.0

# number of designs
N_CASES = 10

# waviness ranges
# keep amplitude conservative relative to Ly
A_MIN = 0.0
A_MAX = 1.0

LAM_MIN = 15.0
LAM_MAX = 30.0

PHASE_MIN = 0.0
PHASE_MAX = 2.0 * math.pi

# geometry sampling resolution for curve description
NPTS = 200

# simulation metadata
UIN_MIN = 0.05
UIN_MAX = 0.20


def make_case_name(i: int) -> str:
    return f"channel_{i:02d}"


def validate_inputs(
    lx: float,
    ly: float,
    amplitude: float,
    wavelength: float,
    npts: int,
) -> None:
    if lx <= 0.0:
        raise ValueError(f"Lx must be positive, got {lx}")
    if ly <= 0.0:
        raise ValueError(f"Ly must be positive, got {ly}")
    if wavelength <= 0.0:
        raise ValueError(f"Wavelength must be positive, got {wavelength}")
    if npts < 2:
        raise ValueError(f"npts must be >= 2, got {npts}")

    # safety bound for future use
    amax = 0.20 * ly
    if abs(amplitude) > amax:
        raise ValueError(
            f"Amplitude {amplitude} exceeds safety bound ±{amax} for Ly={ly}"
        )


def deduplicate_consecutive_points(pts, tol=1e-12):
    if not pts:
        return pts

    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def polygon_signed_area(poly):
    """
    Shoelace formula.
    Positive area => counterclockwise orientation.
    """
    if len(poly) < 3:
        return 0.0

    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def make_wall_curves(
    lx: float,
    ly: float,
    amplitude: float,
    wavelength: float,
    phase: float,
    npts: int = 200,
):
    """
    Build a constant-gap wavy channel:
      - bottom and top walls shift together
      - channel height remains constant everywhere

    Returns:
      pts_bot: bottom wall, left -> right
      pts_top: top wall, left -> right
      inlet:   left boundary, bottom -> top
      outlet:  right boundary, bottom -> top
      fluid_polygon_ccw: closed polygon loop in CCW order
                         bottom(left->right) + outlet(bottom->top)
                         + top(right->left) + inlet(top->bottom)
    """
    validate_inputs(lx, ly, amplitude, wavelength, npts)

    xs = np.linspace(0.0, lx, npts)

    offset = amplitude * np.sin(2.0 * math.pi * xs / wavelength + phase)

    y_bot = offset.copy()
    y_top = ly + offset

    # shift upward if bottom wall dips below y=0
    min_bot = float(np.min(y_bot))
    if min_bot < 0.0:
        shift = -min_bot
        y_bot += shift
        y_top += shift
    else:
        shift = 0.0

    pts_bot = [(float(x), float(y)) for x, y in zip(xs, y_bot)]
    pts_top = [(float(x), float(y)) for x, y in zip(xs, y_top)]

    pts_bot = deduplicate_consecutive_points(pts_bot)
    pts_top = deduplicate_consecutive_points(pts_top)

    inlet = [pts_bot[0], pts_top[0]]       # bottom -> top
    outlet = [pts_bot[-1], pts_top[-1]]    # bottom -> top

    # Build a CCW closed loop:
    # bottom left->right
    # then top right->left
    fluid_polygon_ccw = pts_bot + list(reversed(pts_top))
    fluid_polygon_ccw = deduplicate_consecutive_points(fluid_polygon_ccw)

    # Ensure CCW
    if polygon_signed_area(fluid_polygon_ccw) < 0.0:
        fluid_polygon_ccw = list(reversed(fluid_polygon_ccw))

    # basic validations
    if len(fluid_polygon_ccw) < 4:
        raise ValueError("Fluid polygon has too few points.")

    if abs(inlet[0][0] - inlet[1][0]) > 1e-9:
        raise ValueError("Inlet is not vertical.")
    if abs(outlet[0][0] - outlet[1][0]) > 1e-9:
        raise ValueError("Outlet is not vertical.")
    if inlet[1][1] <= inlet[0][1]:
        raise ValueError("Inlet has non-positive height.")
    if outlet[1][1] <= outlet[0][1]:
        raise ValueError("Outlet has non-positive height.")

    return pts_bot, pts_top, inlet, outlet, fluid_polygon_ccw, shift


def validate_geometry(
    pts_bot,
    pts_top,
    inlet,
    outlet,
    fluid_polygon,
    lx: float,
):
    if len(pts_bot) != len(pts_top):
        raise ValueError("Bottom and top wall point counts do not match.")

    # x should be monotonic increasing for both walls
    x_bot = [p[0] for p in pts_bot]
    x_top = [p[0] for p in pts_top]

    if any(x2 < x1 for x1, x2 in zip(x_bot[:-1], x_bot[1:])):
        raise ValueError("Bottom wall x-coordinates are not monotonic increasing.")
    if any(x2 < x1 for x1, x2 in zip(x_top[:-1], x_top[1:])):
        raise ValueError("Top wall x-coordinates are not monotonic increasing.")

    # top should always be above bottom
    for (xb, yb), (xt, yt) in zip(pts_bot, pts_top):
        if abs(xb - xt) > 1e-9:
            raise ValueError("Bottom/top x-coordinate mismatch.")
        if yt <= yb:
            raise ValueError("Top wall is not above bottom wall at some x.")

    # inlet/outlet should be located at x=0 and x=Lx
    if abs(inlet[0][0] - 0.0) > 1e-9 or abs(inlet[1][0] - 0.0) > 1e-9:
        raise ValueError("Inlet is not located at x=0.")
    if abs(outlet[0][0] - lx) > 1e-9 or abs(outlet[1][0] - lx) > 1e-9:
        raise ValueError(f"Outlet is not located at x={lx}.")

    # polygon orientation should be CCW
    area = polygon_signed_area(fluid_polygon)
    if area <= 0.0:
        raise ValueError("Fluid polygon is not CCW or has zero area.")


def point_list_to_dicts(pts):
    return [{"x": float(x), "y": float(y)} for x, y in pts]


def write_geometry_spec(
    out_json: Path,
    case: str,
    pts_bot,
    pts_top,
    inlet,
    outlet,
    fluid_polygon,
    meta: dict,
):
    payload = {
        "case": case,
        "units": "mm",
        "geometry_type": "2d_channel_constant_gap",
        "topology": {
            "fluid_region_type": "single_closed_polygon",
            "boundary_order_ccw": [
                "wall_bottom:left_to_right",
                "wall_top:right_to_left"
            ],
            "notes": (
                "fluid_polygon is the authoritative closed loop for downstream "
                "geometry creation. inlet/outlet are provided separately for "
                "named boundary reconstruction."
            ),
        },
        "boundaries": {
            "wall_bottom": point_list_to_dicts(pts_bot),      # left -> right
            "wall_top": point_list_to_dicts(pts_top),         # left -> right
            "inlet": point_list_to_dicts(inlet),              # bottom -> top
            "outlet": point_list_to_dicts(outlet),            # bottom -> top
            "fluid_polygon": point_list_to_dicts(fluid_polygon),
        },
        "metadata": meta,
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    rows = []

    for i in range(N_CASES):
        case = make_case_name(i)

        amp = float(rng.uniform(A_MIN, A_MAX))
        lam = float(rng.uniform(LAM_MIN, LAM_MAX))
        phase = float(rng.uniform(PHASE_MIN, PHASE_MAX))
        uin = float(rng.uniform(UIN_MIN, UIN_MAX))

        pts_bot, pts_top, inlet, outlet, fluid_polygon, shift = make_wall_curves(
            lx=Lx,
            ly=Ly,
            amplitude=amp,
            wavelength=lam,
            phase=phase,
            npts=NPTS,
        )

        validate_geometry(
            pts_bot=pts_bot,
            pts_top=pts_top,
            inlet=inlet,
            outlet=outlet,
            fluid_polygon=fluid_polygon,
            lx=Lx,
        )

        spec_path = OUT_DIR / f"{case}.json"

        meta = {
            "random_seed": SEED,
            "A_mm": amp,
            "lam_mm": lam,
            "phase_rad": phase,
            "Lx_mm": Lx,
            "Ly_mm": Ly,
            "npts": NPTS,
            "Uin_mps": uin,
            "y_shift_applied_mm": float(shift),
            "target_geometry_file": str(STEP_DIR / f"{case}.step"),
        }

        write_geometry_spec(
            out_json=spec_path,
            case=case,
            pts_bot=pts_bot,
            pts_top=pts_top,
            inlet=inlet,
            outlet=outlet,
            fluid_polygon=fluid_polygon,
            meta=meta,
        )

        rows.append({
            "case": case,
            "geometry_spec": str(spec_path),
            "target_geometry_file": str(STEP_DIR / f"{case}.step"),
            "A_mm": amp,
            "lam_mm": lam,
            "phase_rad": phase,
            "Lx_mm": Lx,
            "Ly_mm": Ly,
            "npts": NPTS,
            "Uin_mps": uin,
            "y_shift_applied_mm": float(shift),
        })

        print(f"[OK] Wrote geometry spec: {spec_path}")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "geometry_spec",
                "target_geometry_file",
                "A_mm",
                "lam_mm",
                "phase_rad",
                "Lx_mm",
                "Ly_mm",
                "npts",
                "Uin_mps",
                "y_shift_applied_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()