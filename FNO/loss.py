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

def masked_data_rel_l2(pred, target, fluid_mask):
    """
    Relative L2 data loss only inside the fluid region.

    pred:       (B, H, W, C)
    target:     (B, H, W, C)
    fluid_mask: (B, H, W)
    """

    B = pred.shape[0]

    # Expand mask to all output channels
    mask = fluid_mask.bool().unsqueeze(-1)  # (B, H, W, 1)

    losses = []

    for b in range(B):
        pred_b = pred[b][mask[b].expand_as(pred[b])]
        target_b = target[b][mask[b].expand_as(target[b])]

        if target_b.numel() == 0:
            continue

        diff = torch.norm(pred_b - target_b, p=2)
        norm = torch.norm(target_b, p=2)

        losses.append(diff / (norm + 1e-8))

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return torch.stack(losses).mean()

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

def masked_mse(pred_value, target_value, mask):
    """
    pred_value   : (B, H, W)
    target_value : (B, H, W)
    mask         : (B, H, W)
    """
    mask = mask.bool()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_value.device, dtype=pred_value.dtype)

    selected_pred = pred_value[mask]
    selected_true = target_value[mask]

    return F.mse_loss(selected_pred, selected_true)

def pde_residual(pred, uin, Lx=0.05, Ly=0.02):
    """
    PDE residual for nondimensional FNO outputs.

    pred_nd: (B, H, W, 4)
             channel order: [P, T, U, V]

    uin: (B,) or scalar
         case-specific inlet velocity [m/s]

    Returns:
        continuity : (B, H, W)
        mom_u      : (B, H, W)
        mom_v      : (B, H, W)
        energy     : (B, H, W)
    """

    P = pred[..., 0]
    T = pred[..., 1]
    U = pred[..., 2]
    V = pred[..., 3]

    # Fluid density at 325 K [kg/m^3]
    rho = 1.086

    # Kinematic viscosity at 325 K [m^2/s]
    nu = 1.822e-5

    # Heat capacity at 325 K [J/(kg·K)]
    cp = 1006.0

    # Thermal conductivity at 325 K [W/(m·K)]
    k = 0.0280

    # Gravitational acceleration [m/s^2]
    g = 9.81

    # Convert uin to tensor for batch-wise broadcasting
    if not torch.is_tensor(uin):
        uin = torch.tensor(uin, device=pred.device, dtype=pred.dtype)
    else:
        uin = uin.to(device=pred.device, dtype=pred.dtype)

    # Case-specific reference velocity [m/s]
    if uin.ndim == 1:
        U_ref = uin.view(-1, 1, 1)
    else:
        U_ref = uin

    # Characteristic length for Re and Pe [m]
    L_ref = Ly

    # Characteristic distance for buoyancy term [m]
    d_ref = Ly

    # Reference/free-stream temperature [K]
    T_inf = 300.0

    # Heated surface temperature [K]
    T_s = 350.0

    # Reference temperature difference [K]
    dT_ref = T_s - T_inf

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

    # First derivative with respect to normalized x-coordinate (Check W, H)
    def ddx_nd(f):
        W = f.shape[2]
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0 / W).to(f.device).view(1, 1, -1)
        F_dx = F_fft * (1j * 2.0 * torch.pi * kx)
        result = torch.fft.irfft(F_dx, n=W, dim=2).real
        return result

    # First derivative with respect to normalized y-coordinate
    def ddy_nd(f):
        H = f.shape[1]
        F_fft = torch.fft.rfft(f, dim=1)
        ky = torch.fft.rfftfreq(H, d=1.0 / H).to(f.device).view(1, -1, 1)
        F_dy = F_fft * (1j * 2.0 * torch.pi * ky)
        result = torch.fft.irfft(F_dy, n=H, dim=1).real
        return result

    # Second derivative with respect to normalized x-coordinate
    def d2dx_nd(f):
        W = f.shape[2]
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0 / W).to(f.device).view(1, 1, -1)
        F_dxx = F_fft * (-(2.0 * torch.pi * kx) ** 2)
        result = torch.fft.irfft(F_dxx, n=W, dim=2).real
        return result

    # Second derivative with respect to normalized y-coordinate
    def d2dy_nd(f):
        H = f.shape[1]
        F_fft = torch.fft.rfft(f, dim=1)
        ky = torch.fft.rfftfreq(H, d=1.0 / H).to(f.device).view(1, -1, 1)
        F_dyy = F_fft * (-(2.0 * torch.pi * ky) ** 2)
        result = torch.fft.irfft(F_dyy, n=H, dim=1).real
        return result

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
    T_x = ddx_nd(T)
    T_y = ddy_nd(T)

    # x-velocity second derivative
    U_xx = d2dx_nd(U)
    U_yy = d2dy_nd(U)

    # y-velocity second derivative
    V_xx = d2dx_nd(V)
    V_yy = d2dy_nd(V)

    # Temperature second derivative
    T_xx = d2dx_nd(T)
    T_yy = d2dy_nd(T)

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
        + Ri * T
    )
 
    # mom_v = (
    #     U * V_x
    #     + gamma * V * V_y
    #     + gamma * P_y
    #     - Re_ * (V_xx + (gamma ** 2) * V_yy)
    # )
    
    # Energy residual: thermal convection + thermal diffusion
    energy = (
        U * T_x
        + gamma * V * T_y
        - Pe_ * (T_xx + (gamma ** 2) * T_yy)
    )

    return continuity, mom_u, mom_v, energy


def pde_residual_wavy(pred, uin, amp, lam, phase, Lx=0.05, Ly=0.02):
    P = pred[..., 0]
    T = pred[..., 1]
    U = pred[..., 2]
    V = pred[..., 3]

    B, H, W = P.shape
    gamma = Lx / Ly

    # Air properties @ 325 K
    rho, nu, cp, k = 1.086, 1.822e-5, 1006.0, 0.0280
    alpha = k / (rho * cp)
    g = 9.81
    T_inf = 300.0
    T_s   = 350.0
    dT_ref = T_s - T_inf
    beta   = 1.0 / T_inf
    d_ref  = Ly

    def to_tensor_3d(v, ref_tensor):
        if not torch.is_tensor(v):
            v = torch.tensor(v, device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            v = v.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return v.view(-1, 1, 1) if v.ndim == 1 else v.view(1, 1, 1) if v.ndim == 0 else v

    U_ref   = to_tensor_3d(uin,   P)
    amp_m   = to_tensor_3d(amp,   P) / 1000.0  # mm -> m
    lam_m   = to_tensor_3d(lam,   P) / 1000.0  # mm -> m
    phase_r = to_tensor_3d(phase, P)

    Re  = U_ref * Ly / nu
    Pe  = Re * (nu / alpha)
    Re_ = 1.0 / Re
    Pe_ = 1.0 / Pe
    Ri  = g * beta * dT_ref * d_ref / (U_ref ** 2)

    # ------------------------------------------------------------------
    # xi-direction: spectral derivatives (FFT)
    # Periodic in xi because the wavy channel is spatially periodic in x.
    # ------------------------------------------------------------------
    def d_dxi(f):
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0/W).to(f.device).view(1, 1, -1)
        return torch.fft.irfft(F_fft * (1j * 2.0 * torch.pi * kx), n=W, dim=2).real

    def d2_dxi2(f):
        F_fft = torch.fft.rfft(f, dim=2)
        kx = torch.fft.rfftfreq(W, d=1.0/W).to(f.device).view(1, 1, -1)
        return torch.fft.irfft(F_fft * (-(2.0 * torch.pi * kx)**2), n=W, dim=2).real

    # ------------------------------------------------------------------
    # eta-direction: finite difference derivatives (NOT periodic)
    # eta in [0,1] with no-slip walls at eta=0 and eta=1.
    # The channel flow profile is symmetric but not periodic —
    # using FFT here would assume eta=0 and eta=1 connect smoothly,
    # introducing Gibbs artifacts near the walls where accuracy matters most.
    # Using 2nd-order central differences in the interior,
    # and 2nd-order one-sided stencils at the two boundary rows.
    # ------------------------------------------------------------------
    deta = 1.0 / (H - 1)

    def d_deta(f):
        out = torch.empty_like(f)
        out[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * deta)
        out[:, 0,    :] = (-3.0*f[:, 0, :] + 4.0*f[:, 1,  :] - f[:, 2,  :]) / (2.0 * deta)
        out[:, -1,   :] = ( 3.0*f[:, -1,:] - 4.0*f[:, -2, :] + f[:, -3, :]) / (2.0 * deta)
        return out

    def d2_deta2(f):
        out = torch.empty_like(f)
        out[:, 1:-1, :] = (f[:, 2:, :] - 2.0*f[:, 1:-1, :] + f[:, :-2, :]) / (deta**2)
        out[:, 0,    :] = ( 2.0*f[:, 0, :] - 5.0*f[:, 1,  :] + 4.0*f[:, 2,  :] - f[:, 3,  :]) / (deta**2)
        out[:, -1,   :] = ( 2.0*f[:, -1,:] - 5.0*f[:, -2, :] + 4.0*f[:, -3, :] - f[:, -4, :]) / (deta**2)
        return out

    def d2_dxideta(f):
        # Apply spectral xi-derivative first (global, exact),
        # then FD eta-derivative (local, boundary-aware).
        return d_deta(d_dxi(f))

    # ------------------------------------------------------------------
    # Geometry: wall profile derivatives w.r.t. xi (normalized coordinate)
    # ------------------------------------------------------------------
    xi_grid = torch.linspace(0.0, 1.0, W, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    angle   = 2.0 * torch.pi * xi_grid * (Lx / lam_m) + phase_r

    k_xi        = 2.0 * torch.pi * Lx / lam_m                      # dimensionless wavenumber in xi
    h_prime_xi  =  amp_m * k_xi        * torch.cos(angle)           # dh/dxi  [m]
    h_pp_xi     = -amp_m * (k_xi**2)   * torch.sin(angle)           # d2h/dxi2 [m]

    deta_dx_star   = -h_prime_xi / Ly                                
    d2eta_dx2_star = -h_pp_xi    / Ly                               

    # ------------------------------------------------------------------
    # Transformed gradient operators
    # From chain rule on the mapping (xi, eta) -> (x*, y*):
    #   d/dx* = d/dxi + (deta/dxi) * d/deta
    #   d/dy* = gamma * d/deta
    # ------------------------------------------------------------------
    def grad_x(f):
        return d_dxi(f) + deta_dx_star * d_deta(f)

    def grad_y(f):
        return gamma * d_deta(f)

    # ------------------------------------------------------------------
    # Transformed Laplacian (full 4-term chain rule expansion)
    # ------------------------------------------------------------------
    def laplacian(f):
        term_xi2   = d2_dxi2(f)
        term_mixed = 2.0 * deta_dx_star * d2_dxideta(f)
        term_eta2  = (deta_dx_star**2 + gamma**2) * d2_deta2(f)
        term_curve = d2eta_dx2_star * d_deta(f)
        return term_xi2 + term_mixed + term_eta2 + term_curve

    # ------------------------------------------------------------------
    # Field derivatives
    # ------------------------------------------------------------------
    U_x, U_y = grad_x(U), grad_y(U)
    V_x, V_y = grad_x(V), grad_y(V)
    P_x, P_y = grad_x(P), grad_y(P)
    T_x, T_y = grad_x(T), grad_y(T)

    # ------------------------------------------------------------------
    # PDE residuals
    # ------------------------------------------------------------------
    continuity = U_x + V_y
    mom_u      = U * U_x + V * U_y + P_x - Re_ * laplacian(U)
    mom_v      = U * V_x + V * V_y + P_y - Re_ * laplacian(V) + Ri * T
    energy     = U * T_x + V * T_y - Pe_ * laplacian(T)

    return continuity, mom_u, mom_v, energy


def boundary_condition_loss(pred, x):
    """
    Boundary condition loss in nondimensional space.

    pred: (B, H, W, 4)
          channel order: [P, T, U, V]

    x:    (B, H, W, 11)
          physics-aware normalized input
    """

    # Boundary masks
    inlet_mask  = x[..., 1]
    outlet_mask = x[..., 2]
    wall_mask   = x[..., 3]

    # Predicted nondimensional fields
    P_pred = pred[..., 0]
    T_pred = pred[..., 1]
    U_pred = pred[..., 2]
    V_pred = pred[..., 3]

    # Input nondimensional boundary values
    inlet_U = x[..., 4]
    inlet_V = x[..., 5]
    inlet_T = x[..., 6]

    wall_T = x[..., 7]
    wall_U = x[..., 8]
    wall_V = x[..., 9]

    outlet_P = x[..., 10]

    # Inlet boundary loss
    inlet_loss = (
        masked_mse(U_pred, inlet_U, inlet_mask)
        + masked_mse(V_pred, inlet_V, inlet_mask)
        + masked_mse(T_pred, inlet_T, inlet_mask)
    )

    # Outlet boundary loss
    outlet_loss = masked_mse(P_pred, outlet_P, outlet_mask)

    # Wall boundary loss
    wall_loss = (
        masked_mse(U_pred, wall_U, wall_mask)
        + masked_mse(V_pred, wall_V, wall_mask)
        + masked_mse(T_pred, wall_T, wall_mask)
    )

    # Total boundary loss
    bc_loss = inlet_loss + outlet_loss + wall_loss

    return bc_loss, inlet_loss, outlet_loss, wall_loss