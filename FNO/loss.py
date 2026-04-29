import torch
import torch.nn as nn

class LpLoss(nn.Module):
    def __init__(self, p=2, size_average=True):
        super().__init__()
        self.p = p
        self.size_average = size_average

    def forward(self, pred, target):
        diff = torch.norm(pred - target, p=self.p, dim=1)
        norm = torch.norm(target, p=self.p, dim=1)
        loss = diff / (norm + 1e-8)
        return loss.mean() if self.size_average else loss.sum()

def pde_residual(pred_phys, Lx, Ly):
    """
    pred_phys: (B, H, W, 4) — physical units
               channel order: [pressure, temperature, u, v]
    Lx, Ly   : physical domain length (m)

    Returns:
        continuity : (B, H, W)
        mom_u      : (B, H, W)
        mom_v      : (B, H, W)
        energy     : (B, H, W)
    """

    # ── unpack channels ──────────────────────────────────────
    p = pred_phys[..., 0]   # (B, H, W)
    T = pred_phys[..., 1]
    u = pred_phys[..., 2]
    v = pred_phys[..., 3]

    # ── air properties ───────────────────────────────────────
    rho = 1.225
    mu  = 1.789e-5
    cp  = 1006.0
    k   = 0.0242

    # ── spectral derivative helpers ──────────────────────────
    def ddx(f):
        """∂f/∂x using spectral (FFT) differentiation along W axis"""
        W = f.shape[2]
        # rfft along x-axis (dim=2)
        F = torch.fft.rfft(f, dim=2)
        # wavenumbers: 0, 1, ..., W//2
        kx = torch.fft.rfftfreq(W, d=1.0/W).to(f.device)   # shape (W//2+1,)
        kx = kx.view(1, 1, -1)                               # broadcast to (1,1,W//2+1)
        # multiply by i * 2π / Lx
        F_dx = F * (1j * 2.0 * torch.pi * kx / Lx)
        return torch.fft.irfft(F_dx, n=W, dim=2)

    def ddy(f):
        """∂f/∂y using spectral (FFT) differentiation along H axis"""
        H = f.shape[1]
        F = torch.fft.rfft(f, dim=1)
        ky = torch.fft.rfftfreq(H, d=1.0/H).to(f.device)   # shape (H//2+1,)
        ky = ky.view(1, -1, 1)                               # broadcast to (1,H//2+1,1)
        F_dy = F * (1j * 2.0 * torch.pi * ky / Ly)
        return torch.fft.irfft(F_dy, n=H, dim=1)

    def laplacian(f):
        """∇²f = ∂²f/∂x² + ∂²f/∂y² using spectral differentiation"""
        H, W = f.shape[1], f.shape[2]

        # ∂²f/∂x²
        Fx  = torch.fft.rfft(f, dim=2)
        kx  = torch.fft.rfftfreq(W, d=1.0/W).to(f.device).view(1, 1, -1)
        d2x = torch.fft.irfft(Fx * (-(2.0 * torch.pi * kx / Lx)**2), n=W, dim=2)

        # ∂²f/∂y²
        Fy  = torch.fft.rfft(f, dim=1)
        ky  = torch.fft.rfftfreq(H, d=1.0/H).to(f.device).view(1, -1, 1)
        d2y = torch.fft.irfft(Fy * (-(2.0 * torch.pi * ky / Ly)**2), n=H, dim=1)

        return d2x + d2y

    # ── 1. Continuity:  ∇·u = 0 ─────────────────────────────
    continuity = ddx(u) + ddy(v)

    # ── 2. Momentum (x):  ρ(u·∇)u = -∂p/∂x + μ∇²u ─────────
    mom_u = rho * (u * ddx(u) + v * ddy(u)) + ddx(p) - mu * laplacian(u)

    # ── 3. Momentum (y):  ρ(u·∇)v = -∂p/∂y + μ∇²v ─────────
    mom_v = rho * (u * ddx(v) + v * ddy(v)) + ddy(p) - mu * laplacian(v)

    # ── 4. Energy:  ρcₚ(u·∇T) = k∇²T ───────────────────────
    energy = rho * cp * (u * ddx(T) + v * ddy(T)) - k * laplacian(T)

    return continuity, mom_u, mom_v, energy