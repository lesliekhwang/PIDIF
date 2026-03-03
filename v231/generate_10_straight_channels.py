#!/usr/bin/env python3
"""
generate_10_channels_mixed.py

Outputs 10 simple 2D *fluid-domain* channels as STEP faces:
- 5 straight channels (different heights)
- 5 wavy channels (different amplitudes/wavelengths/phases)

Fluid domain = channel interior only (no outer box).
This makes inlet/outlet obvious:
- inlet  = left vertical edge at x=0
- outlet = right vertical edge at x=L
- walls  = top and bottom curves

Also writes step_out/designs.csv (case, type, H/A/lambda/phase, Uin).

Requires: pythonocc-core (OpenCascade)
"""

import os
import csv
import math
from dataclasses import dataclass
from typing import List, Optional

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Case:
    name: str
    kind: str            # "straight" or "wavy"
    L_mm: float
    H_mm: float          # straight height OR mean height for wavy
    Uin_mps: float
    A_mm: float = 0.0    # wavy amplitude (half-gap modulation)
    lam_mm: float = 1.0  # wavelength
    phase: float = 0.0   # phase in radians
    npts: int = 200      # discretization along x


# ----------------------------
# Geometry helpers
# ----------------------------
def export_step(shape, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Interface_Static.SetCVal("write.step.schema", "AP203")
    Interface_Static.SetCVal("write.step.unit", "MM")

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(out_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed: {out_path}")


def make_straight_channel_face(L: float, H: float):
    """
    Rectangle: (0,0)->(L,0)->(L,H)->(0,H)
    """
    poly = BRepBuilderAPI_MakePolygon()
    poly.Add(gp_Pnt(0.0, 0.0, 0.0))
    poly.Add(gp_Pnt(L, 0.0, 0.0))
    poly.Add(gp_Pnt(L, H, 0.0))
    poly.Add(gp_Pnt(0.0, H, 0.0))
    poly.Close()
    return BRepBuilderAPI_MakeFace(poly.Wire()).Face()


def make_wavy_channel_face(L: float, H_mean: float, A: float, lam: float, phase: float, npts: int = 200):
    """
    Build a closed polygon that approximates a wavy-top and wavy-bottom channel.

    Define half-gap variation as:
      y_top(x) = y_c + H_mean/2 + A * sin(2*pi*x/lam + phase)
      y_bot(x) = y_c - H_mean/2 - A * sin(2*pi*x/lam + phase)

    We center channel at y_c = 0 for simplicity, then shift upward so y_bot >= 0.
    """
    if A < 0:
        raise ValueError("A must be >= 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if npts < 30:
        raise ValueError("npts too small; use >= 30")

    # Center at y_c=0 then shift so bottom is >= 0 with margin
    # Worst-case bottom = -H/2 - A, so shift up by (H/2 + A)
    y_shift = (H_mean / 2.0) + A

    def s(x):
        return math.sin(2.0 * math.pi * x / lam + phase)

    xs = [L * i / (npts - 1) for i in range(npts)]

    top = [(x, y_shift + (H_mean / 2.0) + A * s(x)) for x in xs]
    bot = [(x, y_shift - (H_mean / 2.0) - A * s(x)) for x in xs]

    # Build polygon: go along bottom from x=0->L, then along top from x=L->0
    poly = BRepBuilderAPI_MakePolygon()
    for (x, y) in bot:
        poly.Add(gp_Pnt(x, y, 0.0))
    for (x, y) in reversed(top):
        poly.Add(gp_Pnt(x, y, 0.0))
    poly.Close()

    return BRepBuilderAPI_MakeFace(poly.Wire()).Face()


# ----------------------------
# Case set (10 designs)
# ----------------------------
def build_10_cases() -> List[Case]:
    L = 50.0  # mm, matches your sketch scale nicely

    # Straight: vary height only
    straight_H = [1.0, 1.2, 1.5, 1.8, 2.2]     # mm
    U_straight = [0.10, 0.20, 0.30, 0.40, 0.50]  # m/s

    # Wavy: keep mean height moderate; vary amplitude / wavelength / phase
    # Keep A <= ~0.2*H to avoid pinch points.
    wavy_specs = [
        # H_mean,  A,    lam, phase
        (1.5,     0.10,  20.0, 0.0),
        (1.5,     0.15,  25.0, 0.7),
        (1.8,     0.12,  15.0, 1.2),
        (1.8,     0.18,  30.0, 0.3),
        (2.2,     0.15,  18.0, 1.6),
    ]
    U_wavy = [0.10, 0.20, 0.30, 0.40, 0.50]

    cases: List[Case] = []
    idx = 0

    for H, U in zip(straight_H, U_straight):
        cases.append(Case(
            name=f"channel_{idx:02d}",
            kind="straight",
            L_mm=L,
            H_mm=H,
            Uin_mps=U,
        ))
        idx += 1

    for (H, A, lam, phase), U in zip(wavy_specs, U_wavy):
        cases.append(Case(
            name=f"channel_{idx:02d}",
            kind="wavy",
            L_mm=L,
            H_mm=H,
            Uin_mps=U,
            A_mm=A,
            lam_mm=lam,
            phase=phase,
            npts=250,
        ))
        idx += 1

    return cases


def main():
    out_dir = "step_out"
    os.makedirs(out_dir, exist_ok=True)

    cases = build_10_cases()

    csv_path = os.path.join(out_dir, "designs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "kind", "L_mm", "H_mean_mm", "A_mm", "lambda_mm", "phase_rad", "Uin_mps"])
        for c in cases:
            w.writerow([c.name, c.kind, c.L_mm, c.H_mm, c.A_mm, c.lam_mm, c.phase, c.Uin_mps])

    for c in cases:
        if c.kind == "straight":
            face = make_straight_channel_face(c.L_mm, c.H_mm)
        else:
            face = make_wavy_channel_face(c.L_mm, c.H_mm, c.A_mm, c.lam_mm, c.phase, c.npts)

        step_path = os.path.join(out_dir, f"{c.name}.step")
        export_step(face, step_path)
        print(f"[OK] {step_path}  ({c.kind}, H={c.H_mm} mm, A={c.A_mm} mm, lam={c.lam_mm} mm, U={c.Uin_mps} m/s)")

    print(f"\nDone.\n- STEP files: {out_dir}/channel_00.step ... channel_09.step\n- CSV: {csv_path}\n")


if __name__ == "__main__":
    main()