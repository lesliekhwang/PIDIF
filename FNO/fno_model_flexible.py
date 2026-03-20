"""
Flexible 2-D FNO model for the Fluent subdomain datasets.

This is adapted from eval.py, with one important change:
FNO2d no longer hardcodes nn.Linear(13, width). It uses
input_channels + 2 grid channels, so it works with bc11, coords5,
or coords_bc16 input modes.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitGaussianNormalizer:
    """Channel-wise normalizer for NHWC tensors."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC tensor, got shape {tuple(x.shape)}")
        self.mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        self.std = torch.std(x, dim=(0, 1, 2), keepdim=True)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std, "eps": torch.tensor(float(self.eps))}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]):
        obj = cls.__new__(cls)
        obj.mean = state["mean"]
        obj.std = state["std"]
        eps = state.get("eps", torch.tensor(1e-5))
        obj.eps = float(eps.item() if torch.is_tensor(eps) else eps)
        return obj


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.weights1.size(1),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp2(F.gelu(self.mlp1(x)))


class FNO2d(nn.Module):
    """FNO that accepts channel-last NHWC input and returns NHWC output."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        padding: int = 9,
        append_grid: bool = True,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.padding = int(padding)
        self.append_grid = bool(append_grid)

        lifted_channels = self.input_channels + (2 if self.append_grid else 0)
        self.p = nn.Linear(lifted_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, self.output_channels, self.width * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NHWC input, got shape {tuple(x.shape)}")
        if self.append_grid:
            grid = self.get_grid(x.shape, x.device, x.dtype)
            x = torch.cat((x, grid), dim=-1)

        x = self.p(x).permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x[..., :-self.padding, :-self.padding]
        return self.q(x).permute(0, 2, 3, 1)

    @staticmethod
    def get_grid(shape: Tuple[int, int, int, int], device: torch.device, dtype: torch.dtype):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=dtype)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=dtype)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def config(self) -> Dict[str, int]:
        return {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "modes1": self.modes1,
            "modes2": self.modes2,
            "width": self.width,
            "padding": self.padding,
            "append_grid": self.append_grid,
        }
