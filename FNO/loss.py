import torch
import torch.nn as nn
import torch.nn.functional as F

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


def pde_residual(pred_phys, uin, Lx=0.05, Ly=0.02):
    """
    Nondimensional PDE residuals for FNO-based convection problem.

    pred_phys: (B, H, W, 4)
               channel order: [pressure, temperature, u_velocity, v_velocity]

    Lx, Ly: physical domain size [m]

    Returns:
        continuity : (B, H, W)
        mom_u      : (B, H, W)
        mom_v      : (B, H, W)
        energy     : (B, H, W)
    """

    p = pred_phys[..., 0]
    T = pred_phys[..., 1]
    u = pred_phys[..., 2]
    v = pred_phys[..., 3]

    # Fluid density [kg/m^3]
    rho = 1.086

    # Kinematic viscosity [m^2/s]
    nu = 1.822e-5

    # Heat capacity [J/(kg·K)]
    cp = 1006.0

    # Thermal conductivity [W/(m·K)]
    k = 0.0280

    # Gravitational acceleration [m/s^2]
    g = 9.81

    # Convert uin to tensor for batch-wise broadcasting
    if not torch.is_tensor(uin):
        uin = torch.tensor(uin, device=pred_phys.device, dtype=pred_phys.dtype)
    else:
        uin = uin.to(device=pred_phys.device, dtype=pred_phys.dtype)

    # Case-specific reference velocity [m/s]
    if uin.ndim == 1:
        U_ref = uin.view(-1, 1, 1)
    else:
        U_ref = uin

    # Characteristic length for Re and Pe [m]
    L_ref = Lx

    # Characteristic distance for buoyancy term [m]
    d_ref = Ly

    # Reference/free-stream temperature [K]
    T_inf = 300.0

    # Heated surface temperature [K]
    T_s = 350.0

    # Reference temperature difference [K]
    dT_ref = T_s - T_inf

    # Dynamic pressure scale [Pa]
    p_ref = rho * U_ref**2

    # Thermal diffusivity [m^2/s]
    alpha = k / (rho * cp)

    # Reynolds number: inertia / viscous diffusion
    Re = U_ref * L_ref / nu

    # Prandtl number: momentum diffusivity / thermal diffusivity
    Pr = nu / alpha

    # Peclet number: thermal convection / thermal diffusion
    Pe = Re * Pr

    # Thermal expansion coefficient approximation [1/K]
    beta = 1.0 / T_inf

    # Richardson number: buoyancy / inertia
    Ri = g * beta * dT_ref * d_ref / (U_ref**2)

    # Inverse Reynolds number
    Re_ = 1.0 / Re

    # Inverse Peclet number
    Pe_ = 1.0 / Pe

    # Aspect ratio correction for normalized grid
    gamma = Lx / Ly

    # Nondimensional x-velocity
    U = u / U_ref

    # Nondimensional y-velocity
    V = v / U_ref

    # Nondimensional pressure
    P = p / p_ref

    # Nondimensional temperature
    T_nd = (T - T_inf) / dT_ref

    # First derivative with respect to normalized x-coordinate
    def ddx_nd(f):
        W = f.shape[2]
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0 / W).to(f.device).view(1, 1, -1)
        F_dx = F_fft * (1j * 2.0 * torch.pi * kx)
        return torch.fft.irfft(F_dx, n=W, dim=2)

    # First derivative with respect to normalized y-coordinate
    def ddy_nd(f):
        H = f.shape[1]
        F_fft = torch.fft.rfft(f, dim=1)
        ky = torch.fft.rfftfreq(H, d=1.0 / H).to(f.device).view(1, -1, 1)
        F_dy = F_fft * (1j * 2.0 * torch.pi * ky)
        return torch.fft.irfft(F_dy, n=H, dim=1)

    # Second derivative with respect to normalized x-coordinate
    def d2dx_nd(f):
        W = f.shape[2]
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0 / W).to(f.device).view(1, 1, -1)
        F_dxx = F_fft * (-(2.0 * torch.pi * kx) ** 2)
        return torch.fft.irfft(F_dxx, n=W, dim=2)

    # Second derivative with respect to normalized y-coordinate
    def d2dy_nd(f):
        H = f.shape[1]
        F_fft = torch.fft.rfft(f, dim=1)
        ky = torch.fft.rfftfreq(H, d=1.0 / H).to(f.device).view(1, -1, 1)
        F_dyy = F_fft * (-(2.0 * torch.pi * ky) ** 2)
        return torch.fft.irfft(F_dyy, n=H, dim=1)

    # Pressure gradient
    P_x = ddx_nd(P)
    P_y = ddy_nd(P)

    # x-velocity gradient
    U_x = ddx_nd(U)
    U_y = ddy_nd(U)

    # y-velocity gradient
    V_x = ddx_nd(V)
    V_y = ddy_nd(V)

    # Temperature gradient
    T_x = ddx_nd(T_nd)
    T_y = ddy_nd(T_nd)

    # x-velocity second derivative
    U_xx = d2dx_nd(U)
    U_yy = d2dy_nd(U)

    # y-velocity second derivative
    V_xx = d2dx_nd(V)
    V_yy = d2dy_nd(V)

    # Temperature second derivative
    T_xx = d2dx_nd(T_nd)
    T_yy = d2dy_nd(T_nd)

    # Continuity residual: incompressible mass conservation
    continuity = U_x + gamma * V_y

    # x-momentum residual: convection + pressure gradient + viscous diffusion
    mom_u = (
        U * U_x
        + gamma * V * U_y
        + P_x
        - Re_ * (U_xx + (gamma ** 2) * U_yy)
    )

    # y-momentum residual: convection + pressure gradient + diffusion + buoyancy
    mom_v = (
        U * V_x
        + gamma * V * V_y
        + gamma * P_y
        - Re_ * (V_xx + (gamma ** 2) * V_yy)
        + Ri * T_nd
    )

    # Energy residual: thermal convection + thermal diffusion
    energy = (
        U * T_x
        + gamma * V * T_y
        - Pe_ * (T_xx + (gamma ** 2) * T_yy)
    )

    return continuity, mom_u, mom_v, energy


def masked_residual_mse(residual, fluid_mask):
    """
    Compute PDE residual MSE only inside the fluid region.

    residual: (B, H, W)
    fluid_mask: (B, H, W)
    """

    # Boolean mask for fluid region
    mask = fluid_mask.bool()

    # Residual values only inside the fluid region
    selected = residual[mask]

    # Return zero if no fluid points are selected
    if selected.numel() == 0:
        return torch.tensor(0.0, device=residual.device, dtype=residual.dtype)

    # Mean squared residual against zero
    return F.mse_loss(selected, torch.zeros_like(selected))