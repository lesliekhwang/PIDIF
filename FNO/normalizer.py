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

class MinMaxNormalizerMinusOneToOne:
    def __init__(self, x, eps=1e-5):
        self.min = torch.amin(x, dim=(0,1,2), keepdim=True)
        self.max = torch.amax(x, dim=(0,1,2), keepdim=True)
        self.eps = eps

    def encode(self, x):
        return 2 * (x - self.min) / (self.max - self.min + self.eps) - 1

    def decode(self, x):
        return (x + 1) / 2 * (self.max - self.min + self.eps) + self.min

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self

class MinMaxNormalizerZeroToOne:
    def __init__(self, x, eps=1e-5):
        self.min = torch.amin(x, dim=(0, 1, 2), keepdim=True)  # (1,1,1,C)
        self.max = torch.amax(x, dim=(0, 1, 2), keepdim=True)  # (1,1,1,C)
        self.eps = eps

    def encode(self, x):
        return (x - self.min) / (self.max - self.min + self.eps)

    def decode(self, x):
        return x * (self.max - self.min + self.eps) + self.min

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self
    
class PhysicsAwareInputNormalizer:
    """
    Normalizer for x input with 11 channels:

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

    def __init__(self, U_ref=0.20, rho=1.225, dT_ref=50.0):
        self.U_ref = torch.tensor(float(U_ref))
        self.P_ref = torch.tensor(float(rho) * float(U_ref) ** 2)
        self.T_ref = torch.tensor(float(dT_ref))

    def to(self, device):
        self.U_ref = self.U_ref.to(device)
        self.P_ref = self.P_ref.to(device)
        self.T_ref = self.T_ref.to(device)
        return self

    def encode(self, x):
        x_enc = x.clone()

        # masks: 0, 1, 2, 3 stay unchanged

        # inlet velocity
        x_enc[..., 4:5] = x_enc[..., 4:5] / self.U_ref
        x_enc[..., 5:6] = x_enc[..., 5:6] / self.U_ref

        # inlet/wall temperature
        x_enc[..., 6:7] = x_enc[..., 6:7] / self.T_ref
        x_enc[..., 7:8] = x_enc[..., 7:8] / self.T_ref

        # wall velocity
        x_enc[..., 8:9] = x_enc[..., 8:9] / self.U_ref
        x_enc[..., 9:10] = x_enc[..., 9:10] / self.U_ref

        # outlet pressure
        x_enc[..., 10:11] = x_enc[..., 10:11] / self.P_ref

        return x_enc

    def decode(self, x):
        x_dec = x.clone()

        # masks: 0, 1, 2, 3 stay unchanged

        x_dec[..., 4:5] = x_dec[..., 4:5] * self.U_ref
        x_dec[..., 5:6] = x_dec[..., 5:6] * self.U_ref

        x_dec[..., 6:7] = x_dec[..., 6:7] * self.T_ref
        x_dec[..., 7:8] = x_dec[..., 7:8] * self.T_ref

        x_dec[..., 8:9] = x_dec[..., 8:9] * self.U_ref
        x_dec[..., 9:10] = x_dec[..., 9:10] * self.U_ref

        x_dec[..., 10:11] = x_dec[..., 10:11] * self.P_ref

        return x_dec
    
class PhysicsAwareOutputNormalizer:
    """
    Normalizer for y output with 4 channels:

    0 pressure
    1 temperature
    2 u
    3 v
    """

    def __init__(self, U_ref=0.20, rho=1.225, dT_ref=50.0):
        self.U_ref = torch.tensor(float(U_ref))
        self.P_ref = torch.tensor(float(rho) * float(U_ref) ** 2)
        self.T_ref = torch.tensor(float(dT_ref))

    def to(self, device):
        self.U_ref = self.U_ref.to(device)
        self.P_ref = self.P_ref.to(device)
        self.T_ref = self.T_ref.to(device)
        return self

    def encode(self, y):
        y_enc = y.clone()

        y_enc[..., 0:1] = y_enc[..., 0:1] / self.P_ref
        y_enc[..., 1:2] = y_enc[..., 1:2] / self.T_ref
        y_enc[..., 2:3] = y_enc[..., 2:3] / self.U_ref
        y_enc[..., 3:4] = y_enc[..., 3:4] / self.U_ref

        return y_enc

    def decode(self, y):
        y_dec = y.clone()

        y_dec[..., 0:1] = y_dec[..., 0:1] * self.P_ref
        y_dec[..., 1:2] = y_dec[..., 1:2] * self.T_ref
        y_dec[..., 2:3] = y_dec[..., 2:3] * self.U_ref
        y_dec[..., 3:4] = y_dec[..., 3:4] * self.U_ref

        return y_dec