import math
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.GC import GC_MakeCircle
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepCheck import BRepCheck_Analyzer

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone


# ------------------------------
# Geometry params (mm)
Lx = 50.0
Ly = 20.0
y_mid = Ly * 0.5

# Sampling
NPTS = 220          # centerline discretization
NSEC = 60           # number of circular sections used for loft (robust)
EXT  = 0.5          # extend beyond [0,Lx] so inlet/outlet faces are clean
ZC   = 0.0          # pipe center z (mm). Keep 0 unless you want shift.

# ------------------------------
def is_valid(shape) -> bool:
    return BRepCheck_Analyzer(shape).IsValid()

def export_step(shape, out_path: str, unit="MM"):
    Interface_Static.SetCVal("write.step.schema", "AP203")
    Interface_Static.SetCVal("write.step.unit", unit)
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(out_path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {out_path}")

def make_centerline(kind: str, A: float, lam: float, phase: float, radius: float):
    """
    Returns xs, ys arrays for centerline in XY plane (z=ZC).
    Ensures y stays within [0, Ly] by clamping amplitude.
    """
    # keep centerline inside y bounds
    Amax = (Ly * 0.5 - radius - 0.2)
    A = min(abs(A), max(Amax, 0.0))
    if kind == "straight":
        A = 0.0

    xs = np.linspace(-EXT, Lx + EXT, NPTS)

    if A < 1e-12:
        ys = np.full_like(xs, y_mid)
    else:
        ys = y_mid + A * np.sin(2.0 * math.pi * xs / max(lam, 1e-6) + phase)

    return xs, ys

def make_tangent_dir(xs, ys, i: int) -> gp_Dir:
    """
    Tangent direction for section i using finite differences.
    """
    if i == 0:
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
    elif i == len(xs) - 1:
        dx = float(xs[-1] - xs[-2])
        dy = float(ys[-1] - ys[-2])
    else:
        dx = float(xs[i+1] - xs[i-1])
        dy = float(ys[i+1] - ys[i-1])

    n = math.sqrt(dx*dx + dy*dy)
    if n < 1e-12:
        # fallback
        dx, dy = 1.0, 0.0
        n = 1.0
    return gp_Dir(dx/n, dy/n, 0.0)

def make_circle_wire_at(p: gp_Pnt, tdir: gp_Dir, radius: float):
    """
    Circle lies in plane whose normal is the tangent (tdir).
    """
    ax2 = gp_Ax2(p, tdir)  # plane normal = tangent
    circ = GC_MakeCircle(ax2, float(radius)).Value()
    edge = BRepBuilderAPI_MakeEdge(circ).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    return wire

def build_pipe_solid_step(kind, A, lam, phase, radius, out_step):
    """
    Builds the FLUID domain as a pipe solid (no box).
    Uses ThruSections loft of many circle wires -> solid.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    xs, ys = make_centerline(kind, A, lam, phase, radius)

    # pick section indices (downsample)
    idxs = np.linspace(0, len(xs) - 1, NSEC).round().astype(int)
    idxs = np.unique(idxs)  # ensure increasing unique indices

    loft = BRepOffsetAPI_ThruSections(True, False)  # (isSolid=True, ruled=False)

    for i in idxs:
        p = gp_Pnt(float(xs[i]), float(ys[i]), float(ZC))
        tdir = make_tangent_dir(xs, ys, int(i))
        w = make_circle_wire_at(p, tdir, radius)
        loft.AddWire(w)

    loft.Build()
    if not loft.IsDone():
        raise RuntimeError("ThruSections loft failed")

    pipe_solid = loft.Shape()

    if not is_valid(pipe_solid):
        export_step(pipe_solid, "debug_pipe_invalid.step")
        raise RuntimeError("Pipe solid invalid (exported debug_pipe_invalid.step)")

    export_step(pipe_solid, out_step)
    print("Wrote", out_step)


# ---- quick test ----
build_pipe_solid_step(
    kind="wavy",
    A=0.1,
    lam=20.0,
    phase=0.0,
    radius=0.75,
    out_step="fluid_test.step",
)