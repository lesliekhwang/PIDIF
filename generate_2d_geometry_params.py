# generate_2d_geometry_params.py

import csv
import json
from pathlib import Path

import numpy as np


# =========================================
# CONFIG
# =========================================
AR = 10

BASE_DIR = Path("/home/hantianl/Documents/PIDIF")
OUT_DIR = BASE_DIR / "2d_geometry_specs" / "rand_channel_smooth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEP_DIR = BASE_DIR / "2d_geometry_step" / "rand_channel_smooth"
CSV_PATH = OUT_DIR / "designs.csv"

# reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# square side length
L = 150  # mm
MIN_SUBDOMAIN_WIDTH = 0.2 * L

# number of designs
N_CASES = 100

# trapezoid offsets
DELTA = L * 0.1

# inlet velocity
UIN = 0.1 # m/s


def make_case_name(i: int) -> str:
    return f"channel_{i:02d}"


def validate_inputs(l: float) -> None:
    if l <= 0.0:
        raise ValueError(f"L must be positive, got {l}")


def sample_x_breakpoints(l: float, ar: int, min_width: float, rng_obj) -> list[float]:
    """
    Build AR+1 x locations over [0, L*AR]:
      - include 0 and L*AR
      - sample AR-1 interior points uniformly
    """
    if ar < 1:
        raise ValueError(f"AR must be >= 1, got {ar}")

    ref_len = float(l * ar)
    if ar == 1:
        return [0.0, ref_len]

    min_total = ar * min_width
    if min_total > ref_len:
        raise ValueError(
            "Infeasible subdomain constraints: "
            "ar * min_subdomain_width must be <= total length."
        )
    slack = ref_len - min_total
    extras = rng_obj.dirichlet(np.ones(ar, dtype=np.float64))
    widths = min_width + slack * extras
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    edges[-1] = ref_len
    return edges.astype(np.float64).tolist()


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


def make_piecewise_trapezoid_walls(l: float, x_points, deltas):
    """
    Build a long channel from connected trapezoids.
    At each x_i:
      bottom y = -delta_i
      top y    = L + delta_i
    so local height is:
      (L + delta_i) - (-delta_i) = L + 2*delta_i
    """
    validate_inputs(l)
    if len(x_points) != len(deltas):
        raise ValueError("x_points and deltas must have same length.")
    if len(x_points) < 2:
        raise ValueError("At least two x points are required.")

    pts_bot = [(float(x), float(-d)) for x, d in zip(x_points, deltas)]
    pts_top = [(float(x), float(l + d)) for x, d in zip(x_points, deltas)]

    inlet = [pts_bot[0], pts_top[0]]      # bottom -> top
    outlet = [pts_bot[-1], pts_top[-1]]   # bottom -> top

    # Bottom left->right and top right->left forms a CCW loop.
    fluid_polygon_ccw = pts_bot + list(reversed(pts_top))
    if polygon_signed_area(fluid_polygon_ccw) < 0.0:
        fluid_polygon_ccw = list(reversed(fluid_polygon_ccw))

    if inlet[1][1] <= inlet[0][1]:
        raise ValueError("Inlet has non-positive height.")
    if outlet[1][1] <= outlet[0][1]:
        raise ValueError("Outlet has non-positive height.")

    return pts_bot, pts_top, inlet, outlet, fluid_polygon_ccw


def validate_geometry(
    pts_bot,
    pts_top,
    inlet,
    outlet,
    fluid_polygon,
    channel_length: float,
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

    # inlet/outlet should be located at x=0 and x=channel_length
    if abs(inlet[0][0] - 0.0) > 1e-9 or abs(inlet[1][0] - 0.0) > 1e-9:
        raise ValueError("Inlet is not located at x=0.")
    if abs(outlet[0][0] - channel_length) > 1e-9 or abs(outlet[1][0] - channel_length) > 1e-9:
        raise ValueError(f"Outlet is not located at x={channel_length}.")

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
        "geometry_type": "2d_connected_trapezoid_channel",
        "topology": {
            "fluid_region_type": "single_closed_polygon",
            "boundary_order_ccw": [
                "wall_bottom:left_to_right",
                "wall_top:right_to_left",
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

        ar = AR
        x_points = sample_x_breakpoints(l=L, ar=ar, min_width=MIN_SUBDOMAIN_WIDTH, rng_obj=rng)
        deltas = [float(d) for d in rng.uniform(-DELTA, DELTA, size=ar + 1)]
        channel_length = float(L * ar)
        uin = UIN

        pts_bot, pts_top, inlet, outlet, fluid_polygon = make_piecewise_trapezoid_walls(
            l=L,
            x_points=x_points,
            deltas=deltas,
        )

        validate_geometry(
            pts_bot=pts_bot,
            pts_top=pts_top,
            inlet=inlet,
            outlet=outlet,
            fluid_polygon=fluid_polygon,
            channel_length=channel_length,
        )

        inlet_height = float(inlet[1][1] - inlet[0][1])
        outlet_height = float(outlet[1][1] - outlet[0][1])

        spec_path = OUT_DIR / f"{case}.json"

        meta = {
            "random_seed": SEED,
            "L_mm": L,
            "AR": ar,
            "channel_length_mm": channel_length,
            "x_points_mm": x_points,
            "deltas_mm": deltas,
            "inlet_height_mm": inlet_height,
            "outlet_height_mm": outlet_height,
            "Uin_mps": uin,
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
            "L_mm": L,
            "AR": ar,
            "channel_length_mm": channel_length,
            "x_points_mm": json.dumps(x_points),
            "deltas_mm": json.dumps(deltas),
            "inlet_height_mm": inlet_height,
            "Uin_mps": uin,
        })

        print(f"[OK] Wrote geometry spec: {spec_path}")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "geometry_spec",
                "target_geometry_file",
                "L_mm",
                "AR",
                "channel_length_mm",
                "x_points_mm",
                "deltas_mm",
                "inlet_height_mm",
                "Uin_mps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()