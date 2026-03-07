import os
import json
import numpy as np
import pandas as pd

# ---------- OCC ----------
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepCheck import BRepCheck_Analyzer
# -----------------------------
META_PATH = "meta.json"
CSV_PATH  = "params.csv"
OUT_DIR   = "step10"
N_SHOW    = 10
os.makedirs(OUT_DIR, exist_ok=True)

with open(META_PATH, "r") as f:
    meta = json.load(f)

df = pd.read_csv(CSV_PATH)

S = meta.get("settings", {})
Lx = float(S.get("Lx", 50.0))
Ly = float(S.get("Ly", 20.0))
margin = float(S.get("margin", 0.5))

# -----------------------------
def build_centerline(theta_row):
    Wc   = float(theta_row["Wc"])
    py   = float(theta_row["py"])
    Neff = float(theta_row["Neff"])
    sx   = float(theta_row["sx"])
    A    = float(theta_row["A"])
    lam  = float(theta_row["lam"])
    win  = float(theta_row["win"])
    wout = float(theta_row["wout"])

    N = int(np.clip(int(round(Neff)), 6, 24))
    ux = Lx - 2 * margin

    y0 = margin + 0.5 * Wc
    max_lanes_height = (N - 1) * py
    if y0 + max_lanes_height + 0.5 * Wc > Ly - margin:
        y0 = max(margin + 0.5 * Wc, (Ly - margin - 0.5 * Wc) - max_lanes_height)

    xL = 0.0
    xR = Lx

    pts = [(xL, y0)]
    y = y0
    direction = 1
    for k in range(N):
        pts.append((xR if direction == 1 else xL, y))
        if k != N - 1:
            y += py
            pts.append((pts[-1][0], y))
            direction *= -1

    # waviness on horizontal segments
    if A > 1e-9:
        pts_wavy = [pts[0]]
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            if abs(y1 - y0) < 1e-9:
                nseg = 15  # keep modest for robustness
                xs = np.linspace(x0, x1, nseg)
                ys = np.full_like(xs, y0) + A * np.sin(2 * np.pi * (xs - xL) / max(lam, 1e-6))
                pts_wavy.extend(list(zip(xs[1:], ys[1:])))
            else:
                pts_wavy.append((x1, y1))
        pts = pts_wavy

    return pts

def offset_polyline(points, offset):
    pts = np.asarray(points, dtype=float)
    out = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            t = pts[i+1] - pts[i]
        elif i == n-1:
            t = pts[i] - pts[i-1]
        else:
            t = pts[i+1] - pts[i-1]
        tn = np.linalg.norm(t)
        if tn < 1e-12:
            out.append((float(pts[i,0]), float(pts[i,1])))
            continue
        t = t / tn
        normal = np.array([-t[1], t[0]])
        p = pts[i] + offset * normal
        out.append((float(p[0]), float(p[1])))
    return out

def clamp_to_footprint(points, eps=1e-6):
    clamped = []
    for x, y in points:
        x = min(max(x, 0.0 + eps), Lx - eps)
        y = min(max(y, 0.0 + eps), Ly - eps)
        clamped.append((x, y))
    return clamped

def centerline_to_boundary(center_pts, Wc):
    left  = clamp_to_footprint(offset_polyline(center_pts, +Wc/2))
    right = clamp_to_footprint(offset_polyline(center_pts, -Wc/2))
    boundary = left + right[::-1]
    if boundary[0] != boundary[-1]:
        boundary.append(boundary[0])
    return boundary

def export_fluid_volume_to_step(boundary_pts_xy, Hc, out_step_path,
                               box_Lx, box_Ly, box_H,
                               unit="MM"):
    # 1) Channel solid (same as your current code)
    pts = [(float(x), float(y)) for (x, y) in boundary_pts_xy]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    poly = BRepBuilderAPI_MakePolygon()
    for (x, y) in pts:
        poly.Add(gp_Pnt(x, y, 0.0))
    poly.Close()
    wire = poly.Wire()

    face = BRepBuilderAPI_MakeFace(wire).Face()
    channel_solid = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, float(Hc))).Shape()

    # 2) Enclosure box (fluid “container”)
    box_solid = BRepPrimAPI_MakeBox(float(box_Lx), float(box_Ly), float(box_H)).Shape()

    # 3) Boolean cut: fluid = box - channel
    cut = BRepAlgoAPI_Cut(box_solid, channel_solid)
    cut.Build()
    if not cut.IsDone():
        raise RuntimeError("Boolean cut failed (box - channel).")
    fluid_solid = cut.Shape()

    # 4) Basic validity check
    if not BRepCheck_Analyzer(fluid_solid).IsValid():
        raise RuntimeError("Resulting fluid solid is invalid (check geometry/boolean).")

    # 5) Export STEP
    Interface_Static.SetCVal("write.step.schema", "AP203")
    Interface_Static.SetCVal("write.step.unit", unit)

    writer = STEPControl_Writer()
    writer.Transfer(fluid_solid, STEPControl_AsIs)
    status = writer.Write(out_step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {out_step_path}")
    
# -----------------------------
take = min(N_SHOW, len(df))
for i in range(take):
    row = df.iloc[i]
    design_id = int(row.get("design_id", i))

    center = build_centerline(row)
    boundary = centerline_to_boundary(center, Wc=float(row["Wc"]))

    out_step = os.path.join(OUT_DIR, f"channel_{design_id:02d}.step")
    box_H = float(row["Hc"])  # simplest: same height as channel

    export_fluid_volume_to_step(
        boundary_pts_xy=boundary,
        Hc=float(row["Hc"]),
        out_step_path=out_step,
        box_Lx=Lx,
        box_Ly=Ly,
        box_H=box_H,
        unit="MM",
    )
print(f"Exported {take} STEP files to: {OUT_DIR}/")