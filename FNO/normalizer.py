import torch

class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        self.std = torch.std(x, dim=(0, 1, 2), keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

class MinMaxNormalizer:
    """
    Min-max normalizer with configurable output range.

    feature_range:
        (0, 1)   -> normalize to [0, 1]
        (-1, 1)  -> normalize to [-1, 1]

    exclude_idx:
        channel indices that should remain unchanged.
        Useful for binary mask channels.
    """

    def __init__(self, x, feature_range=(-1.0, 1.0), exclude_idx=None, eps=1e-5):
        self.eps = eps
        self.a = float(feature_range[0])
        self.b = float(feature_range[1])

        if self.a >= self.b:
            raise ValueError("feature_range should be (min, max) with min < max")

        self.exclude_idx = list(exclude_idx) if exclude_idx is not None else []

        C = x.shape[-1]
        self.norm_idx = [i for i in range(C) if i not in self.exclude_idx]

        self.min = torch.amin(x[..., self.norm_idx], dim=(0, 1, 2), keepdim=True)
        self.max = torch.amax(x[..., self.norm_idx], dim=(0, 1, 2), keepdim=True)

    def encode(self, x):
        x_enc = x.clone()

        x_sel = x[..., self.norm_idx]
        x01 = (x_sel - self.min) / (self.max - self.min + self.eps)
        x_enc[..., self.norm_idx] = x01 * (self.b - self.a) + self.a

        return x_enc

    def decode(self, x):
        x_dec = x.clone()

        x_sel = x[..., self.norm_idx]
        x01 = (x_sel - self.a) / (self.b - self.a)
        x_dec[..., self.norm_idx] = x01 * (self.max - self.min + self.eps) + self.min

        return x_dec

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self
    
class PhysicsAwareInputNormalizer:
    """
    Physics-aware normalizer for input x with 11 channels:

    0  fluid_mask
    1  inlet_mask
    2  outlet_mask
    3  wall_mask
    4  inlet_u
    5  inlet_v
    6  inlet_T
    7  wall_T
    8  wall_u
    9  wall_v
    10 outlet_p
    """

    def __init__(self, rho=1.086, T_inf=300.0, T_s=350.0):
        # Fluid density [kg/m^3]
        self.rho = torch.tensor(float(rho))

        # Reference/free-stream temperature [K]
        self.T_inf = torch.tensor(float(T_inf))

        # Heated wall temperature [K]
        self.T_s = torch.tensor(float(T_s))

        # Reference temperature difference [K]
        self.dT_ref = torch.tensor(float(T_s - T_inf))

    def _reshape_uin(self, uin, ref):
        # Convert uin to tensor on the same device/dtype as ref
        if not torch.is_tensor(uin):
            uin = torch.tensor(uin, device=ref.device, dtype=ref.dtype)
        else:
            uin = uin.to(device=ref.device, dtype=ref.dtype)

        # Broadcast uin from (B,) to (B, 1, 1, 1)
        if uin.ndim == 1:
            uin = uin.view(-1, 1, 1, 1)

        # Broadcast scalar uin to all samples
        elif uin.ndim == 0:
            uin = uin.view(1, 1, 1, 1)

        return uin

    def encode(self, x, uin):
        """
        Convert physical input channels to nondimensional input channels.

        x:   (B, H, W, 11)
        uin: (B,) or scalar
        """
        x_enc = x.clone()

        # Case-specific reference velocity [m/s]
        U_ref = self._reshape_uin(uin, x)

        # Dynamic pressure scale [Pa]
        P_ref = self.rho.to(x.device, x.dtype) * U_ref**2

        # Reference temperature [K]
        T_inf = self.T_inf.to(x.device, x.dtype)

        # Reference temperature difference [K]
        dT_ref = self.dT_ref.to(x.device, x.dtype)

        # masks: 0, 1, 2, 3 stay unchanged

        # Nondimensional inlet x-velocity
        x_enc[..., 4:5] = x_enc[..., 4:5] / U_ref

        # Nondimensional inlet y-velocity
        x_enc[..., 5:6] = x_enc[..., 5:6] / U_ref

        # Nondimensional inlet temperature
        x_enc[..., 6:7] = (x_enc[..., 6:7] - T_inf) / dT_ref

        # Nondimensional wall temperature
        x_enc[..., 7:8] = (x_enc[..., 7:8] - T_inf) / dT_ref

        # Nondimensional wall x-velocity
        x_enc[..., 8:9] = x_enc[..., 8:9] / U_ref

        # Nondimensional wall y-velocity
        x_enc[..., 9:10] = x_enc[..., 9:10] / U_ref

        # Nondimensional outlet pressure
        x_enc[..., 10:11] = x_enc[..., 10:11] / P_ref

        return x_enc

    def decode(self, x, uin):
        """
        Convert nondimensional input channels back to physical input channels.

        x:   (B, H, W, 11)
        uin: (B,) or scalar
        """
        x_dec = x.clone()

        # Case-specific reference velocity [m/s]
        U_ref = self._reshape_uin(uin, x)

        # Dynamic pressure scale [Pa]
        P_ref = self.rho.to(x.device, x.dtype) * U_ref**2

        # Reference temperature [K]
        T_inf = self.T_inf.to(x.device, x.dtype)

        # Reference temperature difference [K]
        dT_ref = self.dT_ref.to(x.device, x.dtype)

        # masks: 0, 1, 2, 3 stay unchanged

        # Physical inlet x-velocity [m/s]
        x_dec[..., 4:5] = x_dec[..., 4:5] * U_ref

        # Physical inlet y-velocity [m/s]
        x_dec[..., 5:6] = x_dec[..., 5:6] * U_ref

        # Physical inlet temperature [K]
        x_dec[..., 6:7] = x_dec[..., 6:7] * dT_ref + T_inf

        # Physical wall temperature [K]
        x_dec[..., 7:8] = x_dec[..., 7:8] * dT_ref + T_inf

        # Physical wall x-velocity [m/s]
        x_dec[..., 8:9] = x_dec[..., 8:9] * U_ref

        # Physical wall y-velocity [m/s]
        x_dec[..., 9:10] = x_dec[..., 9:10] * U_ref

        # Physical outlet pressure [Pa]
        x_dec[..., 10:11] = x_dec[..., 10:11] * P_ref

        return x_dec
    
    def to(self, device):
        self.rho = self.rho.to(device)
        self.T_inf = self.T_inf.to(device)
        self.T_s = self.T_s.to(device)
        self.dT_ref = self.dT_ref.to(device)
        return self
    
class PhysicsAwareOutputNormalizer:
    """
    Normalizer for y output with 4 channels:

    0 pressure
    1 temperature
    2 u
    3 v
    """

    def __init__(self, rho=1.086, T_inf=300.0, T_s=350.0):
        self.rho = rho
        self.T_inf = T_inf
        self.T_s = T_s
        self.dT_ref = T_s - T_inf

    def _uin(self, uin, ref):
            if not torch.is_tensor(uin):
                uin = torch.tensor(uin, device=ref.device, dtype=ref.dtype)
            else:
                uin = uin.to(device=ref.device, dtype=ref.dtype)

            if uin.ndim == 1:
                uin = uin.view(-1, 1, 1)

            return uin

    def encode(self, y_phys, uin):
        uref = self._uin(uin, y_phys)

        p = y_phys[..., 0]
        T = y_phys[..., 1]
        u = y_phys[..., 2]
        v = y_phys[..., 3]

        P_nd = p / (self.rho * uref**2)
        T_nd = (T - self.T_inf) / self.dT_ref
        U_nd = u / uref
        V_nd = v / uref

        return torch.stack([P_nd, T_nd, U_nd, V_nd], dim=-1)

    def decode(self, y_nd, uin):
        uref = self._uin(uin, y_nd)

        P_nd = y_nd[..., 0]
        T_nd = y_nd[..., 1]
        U_nd = y_nd[..., 2]
        V_nd = y_nd[..., 3]

        p = P_nd * (self.rho * uref**2)
        T = T_nd * self.dT_ref + self.T_inf
        u = U_nd * uref
        v = V_nd * uref

        return torch.stack([p, T, u, v], dim=-1)

    def to(self, device):
        return self