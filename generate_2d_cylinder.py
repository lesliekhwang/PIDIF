import csv
import json
from pathlib import Path

import numpy as np


# Geometry dimensions (mm)
CHANNEL_LENGTH = 1.0
CHANNEL_HEIGHT = 0.1
CYLINDER_DIAMETER = 0.025
CYLINDER_RADIUS = CYLINDER_DIAMETER / 2.0

BASE_DIR = Path("/home/hantianl/Documents/PIDIF")
OUT_DIR = BASE_DIR / "2d_geometry_specs" / "cylinder"
STEP_DIR = BASE_DIR / "2d_geometry_step" / "cylinder"

SEED = 42
START_CASE = 20
N_CASES = 80
UIN = 1.0  # m/s

if START_CASE == 0:
    csv_name = "designs.csv"
else:
    csv_name = f"designs_{START_CASE}_{START_CASE + N_CASES}.csv"
CSV_PATH = OUT_DIR / csv_name

def make_case_name(i: int) -> str:
    return f"channel_{i:03d}"


def point(x: float, y: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y)}


def validate_dimensions() -> None:
    if CHANNEL_LENGTH <= 0.0 or CHANNEL_HEIGHT <= 0.0:
        raise ValueError("Channel length and height must be positive.")
    if CYLINDER_DIAMETER <= 0.0:
        raise ValueError("Cylinder diameter must be positive.")
    if CYLINDER_DIAMETER >= CHANNEL_HEIGHT:
        raise ValueError("Cylinder diameter must be smaller than the channel height.")
    if 2.0 * CYLINDER_RADIUS >= CHANNEL_LENGTH:
        raise ValueError("Cylinder does not fit horizontally inside the rectangle.")


def make_geometry_spec(case: str, cylinder_x: float, cylinder_y: float) -> dict:
    """Return a rectangular 2D channel with one circular cylinder boundary."""
    if not CYLINDER_RADIUS <= cylinder_x <= CHANNEL_LENGTH - CYLINDER_RADIUS:
        raise ValueError("Cylinder does not fit horizontally inside the rectangle.")
    if not CYLINDER_RADIUS <= cylinder_y <= CHANNEL_HEIGHT - CYLINDER_RADIUS:
        raise ValueError("Cylinder does not fit vertically inside the rectangle.")

    bottom = [point(0.0, 0.0), point(CHANNEL_LENGTH, 0.0)]
    top = [point(0.0, CHANNEL_HEIGHT), point(CHANNEL_LENGTH, CHANNEL_HEIGHT)]
    inlet = [point(0.0, 0.0), point(0.0, CHANNEL_HEIGHT)]
    outlet = [point(CHANNEL_LENGTH, 0.0), point(CHANNEL_LENGTH, CHANNEL_HEIGHT)]
    rectangle = [
        point(0.0, 0.0),
        point(CHANNEL_LENGTH, 0.0),
        point(CHANNEL_LENGTH, CHANNEL_HEIGHT),
        point(0.0, CHANNEL_HEIGHT),
    ]

    target_step = STEP_DIR / f"{case}.step"
    return {
        "case": case,
        "units": "mm",
        "geometry_type": "2d_rectangular_channel_with_cylinder",
        "topology": {
            "fluid_region_type": "rectangle_with_one_circular_hole",
            "notes": (
                "The cylinder is represented by a circular inner boundary removed "
                "from the rectangular 2D fluid face."
            ),
        },
        "boundaries": {
            "wall_bottom": bottom,
            "wall_top": top,
            "inlet": inlet,
            "outlet": outlet,
            "fluid_polygon": rectangle,
            "cylinder": {
                "center": point(cylinder_x, cylinder_y),
                "diameter": CYLINDER_DIAMETER,
            },
        },
        "metadata": {
            "random_seed": SEED,
            "channel_length_mm": CHANNEL_LENGTH,
            "channel_height_mm": CHANNEL_HEIGHT,
            "cylinder_center_x_mm": float(cylinder_x),
            "cylinder_center_y_mm": float(cylinder_y),
            "cylinder_diameter_mm": CYLINDER_DIAMETER,
            "Uin_mps": UIN,
            "target_geometry_file": str(target_step),
        },
    }


def main() -> None:
    validate_dimensions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # Keep the complete cylinder inside the rectangle for every random sample.
    x_min = CYLINDER_RADIUS
    x_max = CHANNEL_LENGTH - CYLINDER_RADIUS - 0.3

    y_min = 0.03
    y_max = 0.07

    # Draw from a fresh stream and burn the draws for cases before START_CASE so
    # that the (x, y) sample for a given case index stays fixed regardless of
    # how the case range is batched across separate script runs.
    rng = np.random.default_rng(SEED)
    for _ in range(START_CASE):
        rng.uniform(x_min, x_max)
        rng.uniform(y_min, y_max)

    for i in range(START_CASE, START_CASE + N_CASES):
        case = make_case_name(i)
        cylinder_x = float(rng.uniform(x_min, x_max))
        cylinder_y = float(rng.uniform(y_min, y_max))
        payload = make_geometry_spec(case, cylinder_x, cylinder_y)
        spec_path = OUT_DIR / f"{case}.json"

        with open(spec_path, "w") as f:
            json.dump(payload, f, indent=2)

        metadata = payload["metadata"]
        rows.append({
            "case": case,
            "geometry_spec": str(spec_path),
            "target_geometry_file": metadata["target_geometry_file"],
            "channel_length_mm": CHANNEL_LENGTH,
            "channel_height_mm": CHANNEL_HEIGHT,
            "cylinder_center_x_mm": cylinder_x,
            "cylinder_center_y_mm": cylinder_y,
            "cylinder_diameter_mm": CYLINDER_DIAMETER,
            "Uin_mps": UIN,
        })
        print(f"[OK] Wrote geometry spec: {spec_path}")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
