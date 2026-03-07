import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import qmc
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -----------------------------
# Config (edit these)
# -----------------------------
@dataclass
class Settings:
    # Dataset size
    n_valid: int = 3000
    seed: int = 0

    # Footprint (mm)
    Lx: float = 50.0
    Ly: float = 20.0
    margin: float = 0.5

    # Manufacturing / meshing
    g_min: float = 0.15  # min clearance

    # Output
    out_csv: str = "params.csv"
    out_meta: str = "meta.json"

    # Sampling
    method: str = "sobol"  # "sobol" or "lhs"
    oversample_factor: float = 3.5 


S = Settings()


# -----------------------------
# 12D design vector θ
# -----------------------------
# θ order (all continuous):
# 0 Wc    channel width
# 1 Hc    channel height
# 2 tw    wall thickness
# 3 Rf    fillet radius
# 4 py    lane pitch
# 5 Neff  effective passes (later rounded)
# 6 sx    span ratio
# 7 A     waviness amplitude
# 8 lam   waviness wavelength
# 9 win   inlet header width
# 10 wout outlet header width
# 11 lh   header length

BOUNDS = {
    "Wc":   (0.15, 1.20),
    "Hc":   (0.10, 1.00),
    "tw":   (0.15, 1.20),
    "Rf":   (0.10, 1.50),
    "py":   (0.40, 2.50),     # lower bound will be tightened by constraints
    "Neff": (6.0, 24.0),
    "sx":   (0.60, 0.95),
    "A":    (0.00, 0.30),     # upper bound will be tightened by constraints (A <= 0.3*Hc)
    "lam":  (0.50, 12.0),     # lower bound will be tightened by constraints (lam >= 3A + 0.5)
    "win":  (0.50, 5.00),
    "wout": (0.50, 5.00),
    "lh":   (2.00, 15.0),
}

PARAMS = list(BOUNDS.keys())


# -----------------------------
# Constraint checks
# -----------------------------
def is_valid(theta: np.ndarray, settings: Settings) -> Tuple[bool, Dict[str, float]]:
    """
    Returns: (valid?, derived dict)
    """
    Wc, Hc, tw, Rf, py, Neff, sx, A, lam, win, wout, lh = theta.tolist()
    m = settings.margin
    gmin = settings.g_min

    # Derived integer topology 
    N = int(np.clip(int(round(Neff)), 6, 24))

    # A) pitch clears wall + min gap
    if py < (Wc + tw + gmin):
        return False, {}

    # B) aspect ratio range for mesh stability
    ar = Hc / max(Wc, 1e-9)
    if not (0.3 <= ar <= 5.0):
        return False, {}

    # C) fillet feasibility (simple conservative cap)
    # local clearance between adjacent lane centerlines is py - Wc
    clearance = py - Wc
    if clearance <= 0:
        return False, {}
    Rf_max = 0.5 * min(Wc, tw, clearance)
    if Rf > Rf_max:
        return False, {}

    # D) waviness constraints
    if A > 0.3 * Hc:
        return False, {}
    if lam < (3.0 * A + 0.5):
        return False, {}

    # E) lanes must fit in footprint height
    if (N - 1) * py + Wc + 2 * m > settings.Ly:
        return False, {}

    # F) span must fit in footprint width with margins + headers
    # reserve win and wout headers and require some active channel span.
    usable_x = settings.Lx - 2 * m
    if usable_x <= 0:
        return False, {}

    # Simple requirement: headers + meander span must fit
    # span = sx * (usable_x - win - wout) and must be positive
    span_base = usable_x - win - wout
    if span_base <= 2.0:  # need some space to route
        return False, {}
    span = sx * span_base
    if span < 5.0:  # avoid degenerate layouts
        return False, {}

    # Header length must fit in Ly direction (top view). Conservative:
    if lh + 2 * m > settings.Ly:
        return False, {}

    derived = {
        "N": float(N),
        "AR": float(ar),
        "Rf_max": float(Rf_max),
        "span": float(span),
    }
    return True, derived


# -----------------------------
# Sampling helpers
# -----------------------------
def unit_to_bounds(u: np.ndarray) -> np.ndarray:
    """Map u in [0,1]^d to bounds."""
    x = np.empty_like(u)
    for j, name in enumerate(PARAMS):
        lo, hi = BOUNDS[name]
        x[:, j] = lo + u[:, j] * (hi - lo)
    return x


def draw_candidates(n: int, seed: int, method: str) -> np.ndarray:
    d = len(PARAMS)
    if HAVE_SCIPY:
        if method.lower() == "sobol":
            # Sobol prefers power-of-2; we’ll just draw n and let SciPy handle it.
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            u = sampler.random(n)
        elif method.lower() == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            u = sampler.random(n)
        else:
            raise ValueError(f"Unknown method: {method}")
        return unit_to_bounds(u)

    # Fallback if SciPy is unavailable: plain RNG (less ideal than DOE)
    rng = np.random.default_rng(seed)
    u = rng.random((n, d))
    return unit_to_bounds(u)


# -----------------------------
# Main generation
# -----------------------------
def generate(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed)
    valid_rows: List[Dict[str, float]] = []

    # start with a big batch; if rejection rate is high, loop
    target = settings.n_valid
    batch0 = int(math.ceil(target * settings.oversample_factor))
    tries = 0
    next_seed = settings.seed

    while len(valid_rows) < target:
        tries += 1
        n_need = target - len(valid_rows)
        n_draw = max(batch0, int(math.ceil(n_need * settings.oversample_factor)))

        cand = draw_candidates(n_draw, seed=next_seed, method=settings.method)
        next_seed = int(rng.integers(0, 2**31 - 1))

        for i in range(cand.shape[0]):
            ok, derived = is_valid(cand[i], settings)
            if not ok:
                continue

            row = {name: float(cand[i, j]) for j, name in enumerate(PARAMS)}
            row.update(derived)
            valid_rows.append(row)

            if len(valid_rows) >= target:
                break

        # If acceptance is low, increase oversample_factor adaptively
        if tries >= 2 and len(valid_rows) < target:
            acc = len(valid_rows) / (tries * n_draw)
            if acc < 0.2:
                settings.oversample_factor = min(8.0, settings.oversample_factor * 1.4)

    df = pd.DataFrame(valid_rows[:target])
    # Helpful: stable design_id
    df.insert(0, "design_id", np.arange(len(df), dtype=int))
    return df


if __name__ == "__main__":
    df = generate(S)

    df.to_csv(S.out_csv, index=False)

    meta = {
        "settings": asdict(S),
        "params": PARAMS,
        "bounds": BOUNDS,
        "notes": {
            "Neff_to_N": "N = round(Neff) clipped to [6,24]",
            "units": "mm (assumed)",
            "sampler": "Sobol/LHS from SciPy if available; otherwise RNG fallback",
        },
        "constraints": [
            "py >= Wc + tw + g_min",
            "0.3 <= Hc/Wc <= 5.0",
            "Rf <= 0.5 * min(Wc, tw, py - Wc)",
            "A <= 0.3*Hc",
            "lam >= 3A + 0.5",
            "(N-1)*py + Wc + 2*margin <= Ly",
            "span = sx*(Lx-2*margin - win - wout) >= 5",
            "lh + 2*margin <= Ly",
        ],
    }
    with open(S.out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {S.out_csv} with {len(df)} valid designs")
    print(df.head(5).to_string(index=False))