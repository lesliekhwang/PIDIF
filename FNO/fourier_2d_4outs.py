import os
import csv
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.io import loadmat
from timeit import default_timer

# ----------------------------- Utilities -----------------------------
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

def load_dataset(data_path: str):
    """
    Auto-detect and load .mat or .h5/.hdf5 dataset files.
    Returns: X (torch.Tensor), Y (torch.Tensor)  — shape (N, H, W, C)
    """
    ext = os.path.splitext(data_path)[-1].lower()

    if ext == ".mat":
        data = loadmat(data_path)
        X = torch.from_numpy(data["inputs"].astype(np.float32))
        Y = torch.from_numpy(data["outputs"].astype(np.float32))

    elif ext in (".h5", ".hdf5"):
        with h5py.File(data_path, "r") as hf:
            X = torch.from_numpy(hf["inputs"][:].astype(np.float32))
            Y = torch.from_numpy(hf["outputs"][:].astype(np.float32))

    else:
        raise ValueError(f"Unsupported file format: {ext}  (supported: .mat, .h5, .hdf5)")

    return X, Y

# ----------------------------- Model -----------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.weights1.size(1), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
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
        self.p = nn.Linear(13, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 4, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x).permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for conv, w in [(self.conv0, self.w0), (self.conv1, self.w1), (self.conv2, self.w2)]:
            x = F.gelu(conv(x) + w(x))
        x = self.conv3(x) + self.w3(x)

        x = x[..., :-self.padding, :-self.padding]
        return self.q(x).permute(0, 2, 3, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

def run(cfg, device):
    print(f"\n{'='*60}")
    print(f"[START] {cfg['tag']}")
    print(f"{'='*60}")

    # config unpack
    train_path    = cfg['train_path']
    test_path     = cfg['test_path']
    ntrain        = cfg['ntrain']
    ntest         = cfg['ntest']
    modes         = cfg['modes']
    width         = cfg['width']
    batch_size    = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    epochs        = cfg['epochs']
    s             = cfg['s']
    tag           = cfg['tag']

    log_path   = f"pred/{tag}.csv"
    path_model = f"model/{tag}.pt"

    OUTPUT_COLS = ["pressure", "temperature", "x_velocity", "y_velocity"]

    os.makedirs("pred", exist_ok=True)
    # log_path = f"pred/2d_N{ntrain}_ep{epochs}_batch{batch_size}_s{s}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "time", "train_l2", "test_l2"]
        for col in OUTPUT_COLS:
            header.extend([f"{col}_mse", f"{col}_rel"])
        writer.writerow(header)

    # Data Load
    x_train, y_train = load_dataset(train_path)
    x_test, y_test = load_dataset(test_path) 

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)

    x_train_encoded = x_normalizer.encode(x_train)
    y_train_encoded = y_normalizer.encode(y_train)
    x_test_encoded = x_normalizer.encode(x_test)
    y_test_encoded = y_normalizer.encode(y_test)

    x_normalizer.to(device)
    y_normalizer.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_encoded, y_train_encoded), 
        batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_encoded, y_test_encoded, y_test), 
        batch_size=batch_size, shuffle=False
    )

    model = FNO2d(modes1=modes, modes2=modes, width=width).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    myloss = LpLoss(size_average=True)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        model.eval()
        test_l2, test_mse_ch, test_rel_ch = 0, torch.zeros(4).to(device), torch.zeros(4).to(device)
       
        with torch.no_grad():
            for x, y_n, y_p in test_loader:
                x, y_n, y_p = x.to(device), y_n.to(device), y_p.to(device)
                out_n = model(x)
                test_l2 += myloss(out_n.reshape(x.shape[0], -1), y_n.reshape(x.shape[0], -1)).item()
                out_p = y_normalizer.decode(out_n)

                for c in range(4):
                    test_mse_ch[c] += F.mse_loss(out_p[..., c], y_p[..., c]).item()
                    diff = torch.norm((out_p[..., c] - y_p[..., c]).reshape(x.shape[0], -1), p=2, dim=1)
                    norm = torch.norm(y_p[..., c].reshape(x.shape[0], -1), p=2, dim=1)
                    test_rel_ch[c] += (diff / (norm + 1e-8)).mean().item()

        t2 = default_timer()
        avg_train, avg_test = train_l2 / len(train_loader), test_l2 / len(test_loader)
        avg_mse, avg_rel = test_mse_ch / len(test_loader), test_rel_ch / len(test_loader)


        log_row = [ep, t2-t1, avg_train, avg_test]
        for i in range(4):
            log_row.extend([avg_mse[i].item(), avg_rel[i].item()])
        
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(log_row)

        print(f"Epoch {ep} | Time {t2-t1:.1f}s | Train {avg_train:.6f} | Test {avg_test:.6f}")
        print(f" > [Pres] MSE: {avg_mse[0]:.6f}, Rel: {avg_rel[0]:.6f} | "
              f"[Temp] MSE: {avg_mse[1]:.6f}, Rel: {avg_rel[1]:.6f} | "
              f"[VelX] MSE: {avg_mse[2]:.6f}, Rel: {avg_rel[2]:.6f} | "
              f"[VelY] MSE: {avg_mse[3]:.6f}, Rel: {avg_rel[3]:.6f}")
        
    torch.save(model.state_dict(), path_model)

    ################################################################
    # Prediction on test set (per-sample, batch_size=1)
    ################################################################
    pred = torch.zeros(y_test.shape)
    index = 0

    test_loader_single = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_encoded, y_test), 
        batch_size=1, shuffle=False
    )

    model.eval()
    print("Starting inference for saving .mat file...")
    with torch.no_grad():
        for x, y in test_loader_single:
            x = x.to(device)
            y = y.to(device)

            out = model(x) # (1, H, W, 4)
            out = y_normalizer.decode(out)

            pred[index] = out.cpu()[0]
            diff = torch.norm(out.reshape(1, -1) - y.reshape(1, -1), p=2)
            norm = torch.norm(y.reshape(1, -1), p=2)
            sample_l2 = (diff / (norm + 1e-8)).item()

            if index % 20 == 0: 
                print(f"Sample {index} | Rel L2: {sample_l2:.4f}")
            index += 1

    ################################################################
    # Calculate and Print Final Test Accuracy
    ################################################################
    total_avg_rel = torch.mean(avg_rel).item()
    total_accuracy = (1 - total_avg_rel) * 100
    
    print("\n" + "="*50)
    print(f" FINAL TEST REPORT (N={ntest} samples)")
    print("="*50)
    
    for i, name in enumerate(OUTPUT_COLS):
        ch_acc = (1 - avg_rel[i].item()) * 100
        print(f" [{name:11}] Rel_L2: {avg_rel[i].item():.6f}")
    print("="*50 + "\n")

    save_path = f"pred/{tag}.mat"
    scipy.io.savemat(save_path, mdict={
        "pred": pred.numpy(), 
        "truth": y_test.numpy()
    })
    print(f"Prediction results saved to {save_path}")

CONFIGS = [
    {'train_path': "data/2d_N1000_s64_train.mat", 
     'test_path': "data/2d_N1000_s64_test.mat", 
    'ntrain':1000, 'ntest':200,
    'batch_size':20, 
    'epochs':500,
    's':64,
    'modes':12,
    'width':32,
    'learning_rate':0.001,
    'tag': '2d_N1000_ep500_batch20_s64_mode12_width32_lr0.001'
    }
]

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i+1}/{len(CONFIGS)}] Running: {cfg['tag']}")
        run(cfg, device)