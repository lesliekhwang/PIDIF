import torch
import torch.nn.functional as F
import os
from utilities import *
from model import FNO2d
from loss import *
from normalizer import *

def evaluate(model, x_encoded, y_truth, y_normalizer, device, label=""):
    model.eval()
    with torch.no_grad():
        pred = y_normalizer.decode(model(x_encoded.to(device)))

    pred_cpu = pred.cpu()
    OUTPUT_COLS = ["pressure", "temperature", "x_velocity", "y_velocity"]

    print(f"\n[{label}] n={len(y_truth)}")
    for c, name in enumerate(OUTPUT_COLS):
        diff = torch.norm((pred_cpu[...,c] - y_truth[...,c]).reshape(len(y_truth), -1), p=2, dim=1)
        norm = torch.norm(y_truth[...,c].reshape(len(y_truth), -1), p=2, dim=1)
        rel  = (diff / (norm + 1e-8)).mean().item()
        print(f"  [{name:11}] Rel_L2: {rel:.6f}")

    return pred_cpu

def run(cfg, device):
    # ── data ────────────────────────────────────────────────
    x_train, y_train = load_data(cfg["train_path"])
    x_test, y_test = load_data(cfg["test_path"])

    # ── normalizer ──────────────────────────────────────────
    x_normalizer = UnitGaussianNormalizer(x_train).to(device)
    y_normalizer = UnitGaussianNormalizer(y_train).to(device)

    x_encoded = x_normalizer.encode(x_test.to(device))

    # ── model ───────────────────────────────────────────────
    model = FNO2d(modes1=cfg["modes"], modes2=cfg["modes"], width=cfg["width"]).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))

    os.makedirs("pred", exist_ok=True)
    tag = os.path.splitext(os.path.basename(cfg["model_path"]))[0]

    # ── eval ────────────────────────────────────────────────
    pred_extrap = evaluate(
        model, x_encoded, y_test,
        y_normalizer, device,
        label=f"[{cfg['label']}] Test"
    )

    save_path = f"pred/{tag}_extrap.mat"
    scipy.io.savemat(save_path, {
        "pred"  : pred_extrap.numpy(),
        "truth" : y_test.numpy()
    })
    print(f"[OK] Saved → {save_path}")

CONFIGS = [
    # {
    #     "model_path"     : "model/2d_ep50000_batch20_s64_mode25_width128_minmax_constantlr_0.001_wd1e-4.pt",
    #     "train_path"     : "data/2d_s64_train_v2.mat", 
    #     "test_path"      : "data/2d_s64_test_extrap_v2.mat",
    #     "modes"          : 25,
    #     "width"          : 128,
    #     "label"          : "2d_ep50000_batch20_s64_mode25_width128_minmax_constantlr_0.001_wd1e-4",
    # },
    {
        "model_path"     : "model/PIFNO_lambda0.001_minmax_ep10000.pt",
        "train_path"     : "data/2d_s64_train_v2.mat", 
        "test_path"      : "data/2d_s64_test_extrap_v2.mat",
        "modes"          : 25,
        "width"          : 128,
        "label"          : "PIFNO_lambda0.001_minmax_ep10000",
    },
]


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}] Running: {cfg['label']}")
        run(cfg, device)
