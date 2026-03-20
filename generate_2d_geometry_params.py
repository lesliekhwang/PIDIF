# generate_2d_geometry_params.py

import csv
import json
import math
from pathlib import Path

import numpy as np


# =========================================
# CONFIG
# =========================================
BASE_DIR = Path("/home/hantianl/Documents/PIDIF")
OUT_DIR = BASE_DIR / "2d_geometry_specs" / "trapezoid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEP_DIR = BASE_DIR / "2d_geometry_step" / "trapezoid"
CSV_PATH = OUT_DIR / "designs.csv"

# reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# square side length (200 um = 0.2 mm)
L = 0.2  # mm

# number of designs
N_CASES = 10

# trapezoid offsets
DELTA_MIN = -L * 0.2
DELTA_MAX = L * 0.2

# inlet beta-profile settings (symmetric: alpha = beta)
ALPHA_MIN = 1
ALPHA_MAX = 3
UIN_COEFF = 0.1  # m/s
INLET_PROFILE_NPTS = 201


def make_case_name(i: int) -> str:
    return f"trapezoid_{i:02d}"


def validate_inputs(l: float) -> None:
    if l <= 0.0:
        raise ValueError(f"L must be positive, got {l}")


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


def make_trapezoid_walls(l: float, delta1: float, delta2: float):
    """
    Start from an L by L square and offset y-coordinates:
      - top-left y    = L - delta1
      - bottom-left y = 0 + delta1
      - bottom-right y= 0 + delta2
      - top-right y   = L - delta2
    """
    validate_inputs(l)

    bl = (0.0, float(delta1))
    br = (float(l), float(delta2))
    tr = (float(l), float(l - delta2))
    tl = (0.0, float(l - delta1))

    pts_bot = [bl, br]  # left -> right
    pts_top = [tl, tr]  # left -> right
    inlet = [bl, tl]    # bottom -> top
    outlet = [br, tr]   # bottom -> top
    fluid_polygon_ccw = [bl, br, tr, tl]

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
    l: float,
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

    # inlet/outlet should be located at x=0 and x=L
    if abs(inlet[0][0] - 0.0) > 1e-9 or abs(inlet[1][0] - 0.0) > 1e-9:
        raise ValueError("Inlet is not located at x=0.")
    if abs(outlet[0][0] - l) > 1e-9 or abs(outlet[1][0] - l) > 1e-9:
        raise ValueError(f"Outlet is not located at x={l}.")

    # polygon orientation should be CCW
    area = polygon_signed_area(fluid_polygon)
    if area <= 0.0:
        raise ValueError("Fluid polygon is not CCW or has zero area.")


def point_list_to_dicts(pts):
    return [{"x": float(x), "y": float(y)} for x, y in pts]


def beta_pdf(y_norm: float, alpha: float, beta: float) -> float:
    """
    Beta PDF on y_norm in [0, 1].
    Uses endpoint values consistent with alpha,beta > 1 => zero at both ends.
    """
    if not (0.0 <= y_norm <= 1.0):
        raise ValueError(f"y_norm must be in [0,1], got {y_norm}")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"alpha and beta must be > 0, got {alpha}, {beta}")

    if y_norm in (0.0, 1.0):
        if alpha > 1.0 and beta > 1.0:
            return 0.0
        # fallback for the mathematically singular endpoint case
        return 0.0

    beta_fn = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return (y_norm ** (alpha - 1.0)) * ((1.0 - y_norm) ** (beta - 1.0)) / beta_fn


def build_inlet_velocity_profile(
    inlet_bottom_y: float,
    inlet_top_y: float,
    alpha: float,
    beta: float,
    coeff_mps: float,
    npts: int,
):
    """
    Build inlet velocity profile using normalized inlet coordinate y_norm.
    u_norm follows Beta(alpha, beta) PDF; u_mps = coeff_mps * u_norm.
    """
    if inlet_top_y <= inlet_bottom_y:
        raise ValueError("Inlet has non-positive height for profile generation.")
    if npts < 2:
        raise ValueError(f"npts must be >= 2, got {npts}")

    ys = np.linspace(inlet_bottom_y, inlet_top_y, npts)
    inlet_len = float(inlet_top_y - inlet_bottom_y)

    profile = []
    for y in ys:
        y_norm = float((y - inlet_bottom_y) / inlet_len)
        u_norm = float(beta_pdf(y_norm, alpha, beta))
        u_mps = float(coeff_mps * u_norm)
        profile.append(
            {
                "y_mm": float(y),
                "y_norm": y_norm,
                "u_norm": u_norm,
                "u_mps": u_mps,
            }
        )
    return profile


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
        "geometry_type": "2d_trapezoid_channel",
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

        delta1 = float(rng.uniform(DELTA_MIN, DELTA_MAX))
        delta2 = float(rng.uniform(DELTA_MIN, DELTA_MAX))
        alpha = float(rng.uniform(ALPHA_MIN, ALPHA_MAX))
        beta = alpha

        pts_bot, pts_top, inlet, outlet, fluid_polygon = make_trapezoid_walls(
            l=L,
            delta1=delta1,
            delta2=delta2,
        )

        validate_geometry(
            pts_bot=pts_bot,
            pts_top=pts_top,
            inlet=inlet,
            outlet=outlet,
            fluid_polygon=fluid_polygon,
            l=L,
        )

        # Normalize velocity magnitude based on left inlet length.
        left_inlet_len = float(L - 2.0 * delta1)
        inlet_scale = float(L / left_inlet_len)
        coeff_scaled = float(UIN_COEFF * inlet_scale)

        inlet_profile = build_inlet_velocity_profile(
            inlet_bottom_y=inlet[0][1],
            inlet_top_y=inlet[1][1],
            alpha=alpha,
            beta=beta,
            coeff_mps=coeff_scaled,
            npts=INLET_PROFILE_NPTS,
        )

        # Keep scalar Uin_mps for compatibility with downstream scripts:
        # average of profile over normalized inlet [0,1], where integral(pdf)=1.
        uin = coeff_scaled

        spec_path = OUT_DIR / f"{case}.json"

        meta = {
            "random_seed": SEED,
            "L_mm": L,
            "delta1_mm": delta1,
            "delta2_mm": delta2,
            "alpha": alpha,
            "beta": beta,
            "left_inlet_len_mm": left_inlet_len,
            "inlet_velocity_scale_mps": coeff_scaled,
            "inlet_velocity_profile": inlet_profile,
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
            "delta1_mm": delta1,
            "delta2_mm": delta2,
            "alpha": alpha,
            "beta": beta,
            "left_inlet_len_mm": left_inlet_len,
            "inlet_velocity_scale_mps": coeff_scaled,
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
                "delta1_mm",
                "delta2_mm",
                "alpha",
                "beta",
                "left_inlet_len_mm",
                "inlet_velocity_scale_mps",
                "Uin_mps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()