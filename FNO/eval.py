import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import os

# ------------------ Utilities ------------------
class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x, dim=(0,1,2), keepdim=True)
        self.std = torch.std(x, dim=(0,1,2), keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# ------------------ Model ------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.weights1.size(1), x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        return self.mlp2(F.gelu(self.mlp1(x)))

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.padding = 9
        self.p = nn.Linear(13, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.q = MLP(width, 4, width*4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x).permute(0,3,1,2)
        x = F.pad(x, [0,self.padding,0,self.padding])

        for conv, w in [(self.conv0,self.w0),(self.conv1,self.w1),(self.conv2,self.w2)]:
            x = F.gelu(conv(x)+w(x))
        x = self.conv3(x)+self.w3(x)
        x = x[..., :-self.padding, :-self.padding]
        return self.q(x).permute(0,2,3,1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0,1,size_x,device=device).reshape(1,size_x,1,1).repeat(batchsize,1,size_y,1)
        gridy = torch.linspace(0,1,size_y,device=device).reshape(1,1,size_y,1).repeat(batchsize,size_x,1,1)
        return torch.cat((gridx,gridy), dim=-1)

# ------------------ Main ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load train/test data
train_data = scipy.io.loadmat("data/2d_N1000_s64_train.mat")
test_data = scipy.io.loadmat("data/2d_N1000_s64_test.mat")

x_train = torch.from_numpy(train_data["inputs"].astype("float32"))
y_train = torch.from_numpy(train_data["outputs"].astype("float32"))
x_test = torch.from_numpy(test_data["inputs"].astype("float32"))
y_test = torch.from_numpy(test_data["outputs"].astype("float32"))

# Normalizers
x_normalizer = UnitGaussianNormalizer(x_train).to(device)
y_normalizer = UnitGaussianNormalizer(y_train).to(device)

# x_test_encoded = x_normalizer.encode(x_test).to(device)
x_test = x_test.to(device)              
x_test_encoded = x_normalizer.encode(x_test)

model_paths = [
    "model/2d_N1000_ep500_batch20_s64_mode12_width32_lr0.001_fluidonly.pt"
]

for path in model_paths:
    model = FNO2d(modes1=12, modes2=12, width=32).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_test = model(x_test_encoded)
        pred_test = y_normalizer.decode(pred_test)

    # Save per model prediction
    save_path = path.replace("model/", "pred/").replace(".pt", ".mat")
    os.makedirs("pred", exist_ok=True)
    scipy.io.savemat(save_path, {"pred": pred_test.cpu().numpy(), "truth": y_test.numpy()})
    print(f"Saved prediction to {save_path}, shape: {pred_test.shape}")