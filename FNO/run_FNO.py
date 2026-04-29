import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from timeit import default_timer
from utilities import *
from model import FNO2d
from loss import *
from normalizer import *

def run(cfg, device):
    print(f"\n{'=' * 60}")
    print(f"[START] {cfg['tag']}")
    print(f"{'=' * 60}")

    train_path = cfg['train_path']
    test_path = cfg['test_path']
    modes = cfg['modes']
    width = cfg['width']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    epochs = cfg['epochs']
    s = cfg['s']
    weight_decay = cfg['weight_decay']
    tag = cfg['tag']

    log_path = f"pred/{tag}.csv"
    path_model = f"model/{tag}.pt"

    OUTPUT_COLS = ["pressure", "temperature", "x_velocity", "y_velocity"]

    os.makedirs("pred", exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "time", "train_l2", "test_l2"]
        for col in OUTPUT_COLS:
            header.append(f"{col}_train_rel")
        for col in OUTPUT_COLS:
            header.extend([f"{col}_test_mse", f"{col}_test_rel"])
        writer.writerow(header)

    # Data Load
    x_train, y_train, uin_train = load_dataset(train_path)
    x_test, y_test, uin_test = load_dataset(test_path)
    ntrain = len(x_train)
    ntest = len(x_test)

    if cfg['normalizer'] == "unitguassian":
        x_normalizer = UnitGaussianNormalizer(x_train)
        y_normalizer = UnitGaussianNormalizer(y_train)
    elif cfg['normalizer'] == "minmax":
        x_normalizer = MinMaxNormalizerMinusOneToOne(x_train)
        y_normalizer = MinMaxNormalizerMinusOneToOne(y_train)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=200, T_mult=1, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    myloss = LpLoss(size_average=True)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()

        train_l2 = 0
        train_rel_ch = torch.zeros(4).to(device)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)

            loss = myloss(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            # scheduler.step()

            train_l2 += loss.item()

            # per-channel relative error (physical space)
            out_p = y_normalizer.decode(out)
            y_p = y_normalizer.decode(y)

            for c in range(4):
                diff = torch.norm((out_p[..., c] - y_p[..., c]).reshape(x.shape[0], -1), p=2, dim=1)
                norm = torch.norm(y_p[..., c].reshape(x.shape[0], -1), p=2, dim=1)
                train_rel_ch[c] += (diff / (norm + 1e-8)).mean().item()

        model.eval()
        test_l2 = 0
        test_mse_ch = torch.zeros(4).to(device)
        test_rel_ch = torch.zeros(4).to(device)

        with torch.no_grad():
            for x, y_n, y_p in test_loader:
                x, y_n, y_p = x.to(device), y_n.to(device), y_p.to(device)

                out_n = model(x)

                test_l2 += myloss(
                    out_n.reshape(x.shape[0], -1),
                    y_n.reshape(x.shape[0], -1)
                ).item()

                out_p = y_normalizer.decode(out_n)

                for c in range(4):
                    test_mse_ch[c] += F.mse_loss(out_p[..., c], y_p[..., c]).item()

                    diff = torch.norm((out_p[..., c] - y_p[..., c]).reshape(x.shape[0], -1), p=2, dim=1)
                    norm = torch.norm(y_p[..., c].reshape(x.shape[0], -1), p=2, dim=1)
                    test_rel_ch[c] += (diff / (norm + 1e-8)).mean().item()

        t2 = default_timer()

        avg_train = train_l2 / len(train_loader)
        avg_test = test_l2 / len(test_loader)

        avg_train_rel = train_rel_ch / len(train_loader)
        avg_mse = test_mse_ch / len(test_loader)
        avg_rel = test_rel_ch / len(test_loader)

        log_row = [ep, t2 - t1, avg_train, avg_test]

        for i in range(4):
            log_row.append(avg_train_rel[i].item())

        for i in range(4):
            log_row.extend([avg_mse[i].item(), avg_rel[i].item()])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(log_row)

        print(f"Epoch {ep} | Time {t2 - t1:.1f}s | Train {avg_train:.6f} | Test {avg_test:.6f}")

        print(f" > [Train Rel] "
              f"[Pres] {avg_train_rel[0]:.4f} | "
              f"[Temp] {avg_train_rel[1]:.4f} | "
              f"[VelX] {avg_train_rel[2]:.4f} | "
              f"[VelY] {avg_train_rel[3]:.4f}")

        print(f" > [Test] "
              f"[Pres] MSE: {avg_mse[0]:.6f}, Rel: {avg_rel[0]:.6f} | "
              f"[Temp] MSE: {avg_mse[1]:.6f}, Rel: {avg_rel[1]:.6f} | "
              f"[VelX] MSE: {avg_mse[2]:.6f}, Rel: {avg_rel[2]:.6f} | "
              f"[VelY] MSE: {avg_mse[3]:.6f}, Rel: {avg_rel[3]:.6f}")

        if (ep + 1) in [10000, 20000, 30000, 40000, 50000]:
            ckpt_path = f"model/{tag}_ep{ep+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[CHECKPOINT] Saved → {ckpt_path}")
            
    torch.save(model.state_dict(), path_model)

    ################################################################
    # Prediction on test set (per-sample, batch_size=1)
    ################################################################
    # pred = torch.zeros(y_test.shape)
    # index = 0

    # test_loader_single = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(x_test_encoded, y_test),
    #     batch_size=1, shuffle=False
    # )

    # model.eval()
    # print("Starting inference for saving .mat file...")
    # with torch.no_grad():
    #     for x, y in test_loader_single:
    #         x = x.to(device)
    #         y = y.to(device)

    #         out = model(x)  # (1, H, W, 4)
    #         out = y_normalizer.decode(out)

    #         pred[index] = out.cpu()[0]
    #         diff = torch.norm(out.reshape(1, -1) - y.reshape(1, -1), p=2)
    #         norm = torch.norm(y.reshape(1, -1), p=2)
    #         sample_l2 = (diff / (norm + 1e-8)).item()

    #         if index % 20 == 0:
    #             print(f"Sample {index} | Rel L2: {sample_l2:.4f}")
    #         index += 1

    ################################################################
    # Calculate and Print Final Test Accuracy
    ################################################################
    # print("\n" + "=" * 50)
    # print(f" FINAL TEST REPORT (N={ntest} samples)")
    # print("=" * 50)

    # for i, name in enumerate(OUTPUT_COLS):
    #     ch_acc = (1 - avg_rel[i].item()) * 100
    #     print(f" [{name:11}] Rel_L2: {avg_rel[i].item():.6f}")
    # print("=" * 50 + "\n")

    # save_path = f"pred/{tag}.mat"
    # scipy.io.savemat(save_path, {
    #     "pred": pred.numpy(),
    #     "truth": y_test.numpy(),
    #     "uin": uin_test if uin_test is not None else np.array([])
    # })

    # if uin_test is not None:
    #     mid = np.median(uin_test)
    #     low_idx  = uin_test < mid
    #     high_idx = uin_test >= mid
    #     print(f" [Low  Uin < {mid:.3f}] n={low_idx.sum()}")
    #     print(f" [High Uin >= {mid:.3f}] n={high_idx.sum()}")


CONFIGS = [
    {'train_path': "data/2d_s64_train_v2.mat",
     'test_path': "data/2d_s64_test_gap_v2.mat",
     'normalizer': 'minmax',
     'batch_size': 20,
     'epochs': 50000,
     's': 64,
     'modes': 25,
     'width': 128,
     'learning_rate': 0.001,
     'weight_decay': 1e-4,
     'tag': 'test'
     }
]

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}] Running: {cfg['tag']}")
        run(cfg, device)