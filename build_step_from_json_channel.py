
# build_step_from_json.py

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Wire
from OCC.Core.gp import gp_Pnt


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def point_from_dict(d: dict[str, Any]) -> tuple[float, float]:
    return float(d["x"]), float(d["y"])


def load_points(section: list[dict[str, Any]]) -> list[tuple[float, float]]:
    pts = [point_from_dict(p) for p in section]
    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")
    return pts


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def deduplicate_consecutive_points(
    pts: list[tuple[float, float]],
    tol: float = 1e-12,
) -> list[tuple[float, float]]:
    if not pts:
        return pts

    out = [pts[0]]
    for p in pts[1:]:
        if distance(p, out[-1]) > tol:
            out.append(p)
    return out


def signed_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def close_polygon_if_needed(
    pts: list[tuple[float, float]],
    tol: float = 1e-12,
) -> list[tuple[float, float]]:
    pts = deduplicate_consecutive_points(pts, tol=tol)
    if len(pts) < 3:
        raise ValueError("Polygon must have at least 3 unique points.")
    if distance(pts[0], pts[-1]) > tol:
        pts = pts + [pts[0]]
    return pts


def mm_to_model_units(val_mm: float, units: str) -> float:
    units = units.lower().strip()
    if units == "mm":
        return val_mm
    if units == "m":
        return val_mm / 1000.0
    raise ValueError(f"Unsupported units: {units}")


def convert_points_units(
    pts: list[tuple[float, float]],
    units: str,
) -> list[tuple[float, float]]:
    return [(mm_to_model_units(x, units), mm_to_model_units(y, units)) for x, y in pts]


def as_gp_pnt(p: tuple[float, float]) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), 0.0)


# ============================================================
# GEOMETRY VALIDATION
# ============================================================

def validate_wall_pair(
    bottom: list[tuple[float, float]],
    top: list[tuple[float, float]],
) -> None:
    if len(bottom) < 2 or len(top) < 2:
        raise ValueError("Top/bottom walls must each contain at least 2 points.")

    xb = [p[0] for p in bottom]
    xt = [p[0] for p in top]

    if any(x2 < x1 for x1, x2 in zip(xb[:-1], xb[1:])):
        raise ValueError("Bottom wall x is not monotonic increasing.")
    if any(x2 < x1 for x1, x2 in zip(xt[:-1], xt[1:])):
        raise ValueError("Top wall x is not monotonic increasing.")

    if abs(bottom[0][0] - top[0][0]) > 1e-9:
        raise ValueError("Bottom/top do not share same inlet x.")
    if abs(bottom[-1][0] - top[-1][0]) > 1e-9:
        raise ValueError("Bottom/top do not share same outlet x.")

    if top[0][1] <= bottom[0][1]:
        raise ValueError("Invalid inlet height.")
    if top[-1][1] <= bottom[-1][1]:
        raise ValueError("Invalid outlet height.")

    # Ensure top is always above bottom
    if len(bottom) == len(top):
        for (xb_i, yb_i), (xt_i, yt_i) in zip(bottom, top):
            if abs(xb_i - xt_i) > 1e-9:
                raise ValueError("Bottom/top x mismatch.")
            if yt_i <= yb_i:
                raise ValueError("Top wall not above bottom wall.")


def validate_fluid_polygon(poly: list[tuple[float, float]]) -> None:
    if len(poly) < 4:
        raise ValueError("Fluid polygon must contain at least 4 points including closure.")
    if distance(poly[0], poly[-1]) > 1e-9:
        raise ValueError("Fluid polygon must be closed (first point equals last point).")
    if abs(signed_area(poly[:-1])) <= 1e-14:
        raise ValueError("Fluid polygon has zero or near-zero area.")


# ============================================================
# OCC CURVE / EDGE BUILDERS
# ============================================================

def make_bspline_edge(
    pts: list[tuple[float, float]],
    label: str,
    linear: bool = False,
) :
    pts = deduplicate_consecutive_points(pts)
    if len(pts) < 2:
        raise ValueError(f"{label}: need at least 2 unique points for spline edge")

    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, (x, y) in enumerate(pts, start=1):
        arr.SetValue(i, gp_Pnt(float(x), float(y), 0.0))

    if linear:
        # DegMin=DegMax=1, C0 continuity => exact polyline as one curve.
        spline_builder = GeomAPI_PointsToBSpline(arr, 1, 1, GeomAbs_C0, 1.0e-9)
    else:
        spline_builder = GeomAPI_PointsToBSpline(arr)
    curve = spline_builder.Curve()
    if curve is None:
        raise RuntimeError(f"{label}: failed to construct BSpline curve")

    edge_builder = BRepBuilderAPI_MakeEdge(curve)
    if not edge_builder.IsDone():
        raise RuntimeError(f"{label}: failed to make edge from BSpline")

    return edge_builder.Edge()


def make_line_edge(
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str,
):
    edge_builder = BRepBuilderAPI_MakeEdge(as_gp_pnt(p1), as_gp_pnt(p2))
    if not edge_builder.IsDone():
        raise RuntimeError(f"{label}: failed to make straight edge")
    return edge_builder.Edge()


# ============================================================
# BUILD 4-EDGE WIRE/FACE
# ============================================================

def build_channel_wire_from_walls(
    bottom_pts: list[tuple[float, float]],
    top_pts: list[tuple[float, float]],
    linear_walls: bool = True,
) -> TopoDS_Wire:
    """
    Build exactly four edges:

        1) bottom spline   : left -> right
        2) outlet line     : bottom-right -> top-right
        3) top spline      : right -> left
        4) inlet line      : top-left -> bottom-left

    Each wall is a single edge, so the mesh gets exactly 4 boundary zones
    (inlet, outlet, wall_bottom, wall_top) with no per-segment fragmentation.

    `linear_walls=True` represents the walls as degree-1 B-splines so that
    connected-trapezoid corners stay sharp; straight (2-point) walls are
    unaffected.
    """
    validate_wall_pair(bottom_pts, top_pts)

    bottom_pts = deduplicate_consecutive_points(bottom_pts)
    top_pts = deduplicate_consecutive_points(top_pts)

    p_bl = bottom_pts[0]
    p_br = bottom_pts[-1]
    p_tl = top_pts[0]
    p_tr = top_pts[-1]

    # IMPORTANT:
    # top edge must run right -> left to keep one closed loop orientation
    top_rev = list(reversed(top_pts))

    e_bottom = make_bspline_edge(bottom_pts, "bottom wall", linear=linear_walls)
    e_outlet = make_line_edge(p_br, p_tr, "outlet")
    e_top = make_bspline_edge(top_rev, "top wall", linear=linear_walls)
    e_inlet = make_line_edge(p_tl, p_bl, "inlet")

    wire_builder = BRepBuilderAPI_MakeWire()
    wire_builder.Add(e_bottom)
    wire_builder.Add(e_outlet)
    wire_builder.Add(e_top)
    wire_builder.Add(e_inlet)

    if not wire_builder.IsDone():
        raise RuntimeError("Failed to build wire from 4 edges")

    wire = wire_builder.Wire()

    # Heal wire
    fixer = ShapeFix_Wire()
    fixer.Load(wire)
    fixer.Perform()
    wire = fixer.Wire()

    return wire


def build_channel_wire_from_polygon(
    fluid_poly_pts: list[tuple[float, float]],
) -> TopoDS_Wire:
    """
    Build a wire from piecewise-linear polygon segments.
    This preserves connected-trapezoid corner locations from generated JSON.
    """
    fluid_poly_pts = close_polygon_if_needed(fluid_poly_pts)

    # OpenCascade builds edges from consecutive points; do not duplicate closure point in loop logic.
    loop_pts = fluid_poly_pts[:-1]
    if signed_area(loop_pts) < 0.0:
        loop_pts = list(reversed(loop_pts))

    loop_pts = close_polygon_if_needed(loop_pts)
    validate_fluid_polygon(loop_pts)

    wire_builder = BRepBuilderAPI_MakeWire()
    for i in range(len(loop_pts) - 1):
        p1 = loop_pts[i]
        p2 = loop_pts[i + 1]
        edge = make_line_edge(p1, p2, label=f"poly_seg_{i}")
        wire_builder.Add(edge)

    if not wire_builder.IsDone():
        raise RuntimeError("Failed to build wire from fluid polygon segments")

    wire = wire_builder.Wire()

    fixer = ShapeFix_Wire()
    fixer.Load(wire)
    fixer.Perform()
    wire = fixer.Wire()
    return wire


def build_face_from_wire(wire: TopoDS_Wire) -> TopoDS_Face:
    face_builder = BRepBuilderAPI_MakeFace(wire)
    if not face_builder.IsDone():
        raise RuntimeError("Failed to build face from wire")

    face = face_builder.Face()

    face_fixer = ShapeFix_Face(face)
    face_fixer.Perform()
    face = face_fixer.Face()

    analyzer = BRepCheck_Analyzer(face)
    if not analyzer.IsValid():
        raise RuntimeError("Face is invalid after healing")

    return face


# ============================================================
# STEP EXPORT
# ============================================================

def export_step(shape, out_step: Path) -> None:
    out_step.parent.mkdir(parents=True, exist_ok=True)

    writer = STEPControl_Writer()

    Interface_Static.SetCVal("write.step.schema", "AP214")
    Interface_Static.SetCVal("write.surfacecurve.mode", "1")

    status = writer.Transfer(shape, STEPControl_AsIs)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP transfer failed")

    status = writer.Write(str(out_step))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP file: {out_step}")


# ============================================================
# SPEC PARSING
# ============================================================

def extract_case_geometry(spec: dict[str, Any]):
    case = spec.get("case", "unknown_case")
    units = spec.get("units", "mm")
    geometry_type = str(spec.get("geometry_type", "")).strip().lower()
    boundaries = spec["boundaries"]

    bottom_pts = top_pts = None
    fluid_poly_pts = None

    if "wall_bottom" in boundaries and "wall_top" in boundaries:
        bottom_pts = load_points(boundaries["wall_bottom"])
        top_pts = load_points(boundaries["wall_top"])
        bottom_pts = convert_points_units(bottom_pts, units)
        top_pts = convert_points_units(top_pts, units)

    if "fluid_polygon" in boundaries:
        fluid_poly_pts = load_points(boundaries["fluid_polygon"])
        fluid_poly_pts = convert_points_units(fluid_poly_pts, units)

    if bottom_pts is None and fluid_poly_pts is None:
        raise ValueError(
            "Spec must provide either wall_bottom/wall_top or fluid_polygon under boundaries."
        )

    return case, units, geometry_type, bottom_pts, top_pts, fluid_poly_pts


# ============================================================
# MAIN SINGLE BUILD
# ============================================================

def build_step_from_json(
    spec_path: Path,
    out_step: Path | None = None,
    verbose: bool = True,
) -> Path:
    spec = load_json(spec_path)
    case, units, geometry_type, bottom_pts, top_pts, fluid_poly_pts = extract_case_geometry(spec)

    if out_step is None:
        target = spec.get("metadata", {}).get("target_geometry_file")
        out_step = Path(target) if target else spec_path.with_suffix(".step")

    # Prefer the wall-based 4-edge build whenever wall point lists are present.
    # Each wall becomes a single (degree-1) edge, so the mesh has exactly 4
    # boundary zones (inlet, outlet, wall_bottom, wall_top) with sharp corners
    # preserved. The per-segment polygon builder is only a fallback.
    if bottom_pts is not None and top_pts is not None:
        wire = build_channel_wire_from_walls(bottom_pts, top_pts, linear_walls=True)
        constructed_edges = 4
        bottom_count = len(bottom_pts)
        top_count = len(top_pts)
    elif fluid_poly_pts is not None:
        wire = build_channel_wire_from_polygon(fluid_poly_pts)
        constructed_edges = len(close_polygon_if_needed(fluid_poly_pts)) - 1
        bottom_count = "n/a"
        top_count = "n/a"
    else:
        raise ValueError("Spec must provide wall_bottom/wall_top or fluid_polygon.")

    face = build_face_from_wire(wire)
    export_step(face, out_step)

    if verbose:
        log_lines = [
            f"[OK] case={case}",
            f"     units={units}",
            f"     geometry_type={geometry_type or 'unknown'}",
            f"     bottom_points={bottom_count}",
            f"     top_points={top_count}",
            f"     constructed_edges={constructed_edges}",
            f"     wrote STEP: {out_step}",
        ]
        print("\n".join(log_lines))

    return out_step


# ============================================================
# BATCH
# ============================================================

def build_from_csv(csv_path: Path, verbose: bool = True) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    results = []

    for row in rows:
        case = row.get("case", "unknown_case")
        spec_path = Path(row["geometry_spec"])
        out_step = Path(row["target_geometry_file"]) if row.get("target_geometry_file") else spec_path.with_suffix(".step")

        try:
            build_step_from_json(spec_path, out_step=out_step, verbose=verbose)
            results.append({
                "case": case,
                "geometry_spec": str(spec_path),
                "step_file": str(out_step),
                "status": "ok",
            })
        except Exception as e:
            print(f"[FAIL] case={case} reason={e}")
            results.append({
                "case": case,
                "geometry_spec": str(spec_path),
                "step_file": str(out_step),
                "status": f"failed: {e}",
            })

    step_dir = Path(rows[0]["target_geometry_file"]).parent if rows and rows[0].get("target_geometry_file") else csv_path.parent
    step_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = step_dir / "step_build_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "geometry_spec", "step_file", "status"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"[DONE] Wrote summary: {summary_csv}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build refined 2D STEP geometry from JSON spec using 4 CAD edges."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", type=str, help="Single geometry spec JSON")
    group.add_argument("--csv", type=str, help="Batch designs.csv")

    parser.add_argument("--out", type=str, default=None, help="Optional STEP output path for single case")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.json:
        spec_path = Path(args.json)
        out_step = Path(args.out) if args.out else None
        build_step_from_json(spec_path, out_step=out_step, verbose=True)

    elif args.csv:
        build_from_csv(Path(args.csv), verbose=True)


if __name__ == "__main__":
    main()