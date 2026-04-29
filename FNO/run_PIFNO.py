import csv
import torch
import torch.nn.functional as F
from timeit import default_timer
from utilities import *
from model import FNO2d
from loss import *
from normalizer import *

def run(cfg, device):

    print(f"\n{'='*60}")
    print(f"[PIFNO TRAIN] {cfg['tag']}")
    print(f"{'='*60}")

    # -----------------------------
    # load dataset
    # -----------------------------
    x_train, y_train, uin_train = load_dataset(cfg['train_path'])
    x_test, y_test, uin_test = load_dataset(cfg['test_path'])
    
    fluid_mask_train = x_train[..., 0]
    fluid_mask_test  = x_test[..., 0]

    # -----------------------------
    # normalization
    # -----------------------------
    if cfg['normalizer'] == "unitguassian":
        x_normalizer = UnitGaussianNormalizer(x_train)
        y_normalizer = UnitGaussianNormalizer(y_train)
    elif cfg['normalizer'] == "minmax":
        x_normalizer = MinMaxNormalizerMinusOneToOne(x_train)
        y_normalizer = MinMaxNormalizerMinusOneToOne(y_train)
    else:
        x_normalizer = PhysicsAwareInputNormalizer(U_ref=0.20)
        y_normalizer = PhysicsAwareOutputNormalizer(U_ref=0.20)

    # print("y_normalizer std:", y_normalizer.std.squeeze())

    x_train_enc = x_normalizer.encode(x_train)
    y_train_enc = y_normalizer.encode(y_train)
    x_test_enc  = x_normalizer.encode(x_test)
    y_test_enc  = y_normalizer.encode(y_test)

    x_normalizer.to(device)
    y_normalizer.to(device)

    # print("x_train shape:", x_train.shape)
    # print("x_train_enc shape:", x_train_enc.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_train_enc shape:", y_train_enc.shape)
    # print("x_test shape:", x_test.shape)
    # print("x_test_enc shape:", x_test_enc.shape)

    # -----------------------------
    # dataloader
    # -----------------------------
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_enc, y_train_enc, fluid_mask_train),
        batch_size=cfg['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_enc, y_test_enc, y_test, fluid_mask_test),
        batch_size=cfg['batch_size'], shuffle=False
    )

    # -----------------------------
    # model
    # -----------------------------
    model = FNO2d(modes1=cfg['modes'], modes2=cfg['modes'], width=cfg['width']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    if cfg['scheduler'] == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    elif cfg['scheduler'] == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'] * len(train_loader))
    elif cfg['scheduler'] == "cosinewarmrestart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=200, T_mult=1, eta_min=1e-5)
    elif cfg['scheduler'] == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.5,patience=20,min_lr=1e-5)
    else:
        pass

    data_loss_fn = LpLoss()

    H = x_train.shape[1]
    W = x_train.shape[2]

    Lx = cfg.get("Lx", 0.050)
    Ly = cfg.get("Ly", 0.020)

    dx_phys = Lx / (W - 1)
    dy_phys = Ly / (H - 1)

    U_ref  = cfg.get("U_ref", 0.20)
    L_ref  = cfg.get("L_ref", Lx)
    dT_ref = cfg.get("dT_ref", 50.0)
    rho    = cfg.get("rho", 1.225)
    cp     = cfg.get("cp", 1006.0)

    cont_scale   = (U_ref / L_ref) ** 2
    mom_scale    = (rho * U_ref ** 2 / L_ref) ** 2
    energy_scale = (rho * cp * U_ref * dT_ref / L_ref) ** 2

    lambda_pde = cfg["lambda_pde"]

    ema_decay = 0.99
    ema_scale = {"cont": 1.0, "mou": 1.0, "mov": 1.0, "ene": 1.0}

    os.makedirs("pred", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    log_path = f"pred/{cfg['tag']}.csv"
    OUTPUT_COLS = ["pressure", "temperature", "x_velocity", "y_velocity"]

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "time", "data_loss", 
                  "pde_continuity", "pde_mom_u", "pde_mom_v", "pde_energy", "pde_total", "weighted_pde_loss",
                  "total_loss", "test_l2"]

        for col in OUTPUT_COLS:
            header.append(f"{col}_train_rel")
        for col in OUTPUT_COLS:
            header.extend([f"{col}_test_mse", f"{col}_test_rel"])
        writer.writerow(header)
    # ========================================================
    # TRAIN LOOP
    # ========================================================
    for ep in range(cfg['epochs']):
        model.train()
        t1 = default_timer()

        data_loss_sum  = 0.0
        pde_cont_sum   = 0.0
        pde_mou_sum    = 0.0
        pde_mov_sum    = 0.0
        pde_energy_sum = 0.0
        pde_loss_sum   = 0.0

        train_rel_ch = torch.zeros(4).to(device)

        for x, y, fluid_mask in train_loader:
            x, y, fluid_mask = x.to(device), y.to(device), fluid_mask.to(device)
            mask_sum = fluid_mask.sum() + 1e-8

            optimizer.zero_grad()

            # forward
            pred = model(x)

            # data loss (normalized space)
            data_loss = data_loss_fn(
                pred.reshape(x.shape[0], -1),
                y.reshape(x.shape[0], -1)
            )

            # physics loss (physical space)
            pred_phys = y_normalizer.decode(pred)
            continuity, mom_u, mom_v, energy = pde_residual(pred_phys, Lx, Ly)

            # Normalization

            with torch.no_grad():
                raw_cont = float(continuity.pow(2).mean().clamp(min=1e-10).item())
                raw_mou  = float(mom_u.pow(2).mean().clamp(min=1e-10).item())
                raw_mov  = float(mom_v.pow(2).mean().clamp(min=1e-10).item())
                raw_ene  = float(energy.pow(2).mean().clamp(min=1e-10).item())

            ema_scale["cont"] = ema_decay * ema_scale["cont"] + (1 - ema_decay) * raw_cont
            ema_scale["mou"]  = ema_decay * ema_scale["mou"]  + (1 - ema_decay) * raw_mou
            ema_scale["mov"]  = ema_decay * ema_scale["mov"]  + (1 - ema_decay) * raw_mov
            ema_scale["ene"]  = ema_decay * ema_scale["ene"]  + (1 - ema_decay) * raw_ene

            s_cont = max(ema_scale["cont"], 1e-10)
            s_mou  = max(ema_scale["mou"],  1e-10)
            s_mov  = max(ema_scale["mov"],  1e-10)
            s_ene  = max(ema_scale["ene"],  1e-10)

            pde_cont = (continuity.pow(2) * fluid_mask).sum() / mask_sum / s_cont
            pde_mou  = (mom_u.pow(2)      * fluid_mask).sum() / mask_sum / s_mou
            pde_mov  = (mom_v.pow(2)      * fluid_mask).sum() / mask_sum / s_mov
            pde_ene  = (energy.pow(2)     * fluid_mask).sum() / mask_sum / s_ene

            pde_loss = pde_cont + pde_mou + pde_mov + pde_ene

            # pde_cont = (continuity.pow(2) * fluid_mask).sum() / mask_sum / cont_scale
            # pde_mou  = (mom_u.pow(2)      * fluid_mask).sum() / mask_sum / mom_scale
            # pde_mov  = (mom_v.pow(2)      * fluid_mask).sum() / mask_sum / mom_scale
            # pde_ene  = (energy.pow(2)     * fluid_mask).sum() / mask_sum / energy_scale

            # pde_cont   = continuity.pow(2).mean() / cont_scale
            # pde_mou    = mom_u.pow(2).mean()      / mom_scale
            # pde_mov    = mom_v.pow(2).mean()      / mom_scale
            # pde_ene    = energy.pow(2).mean()     / energy_scale

            pde_cont_sum   += pde_cont.item()
            pde_mou_sum    += pde_mou.item()
            pde_mov_sum    += pde_mov.item()
            pde_energy_sum += pde_ene.item()

            # total loss
            warmup_ep = cfg.get("warmup_epochs", 0)
            if ep < warmup_ep:
                progress = ep / warmup_ep
                lambda_pde_current = cfg["lambda_pde"] * (1 - torch.cos(torch.tensor(progress * torch.pi)).item()) / 2
            else:
                lambda_pde_current = cfg["lambda_pde"]

            # print(f"lambda pde current: {lambda_pde_current}")
            loss = data_loss + lambda_pde_current * pde_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            # scheduler.step()

            data_loss_sum += data_loss.item()
            pde_loss_sum  += pde_loss.item()

            # per-channel train rel (physical space)
            y_phys = y_normalizer.decode(y)
            for c in range(4):
                diff = torch.norm((pred_phys[..., c] - y_phys[..., c]).reshape(x.shape[0], -1), p=2, dim=1)
                norm = torch.norm(y_phys[..., c].reshape(x.shape[0], -1), p=2, dim=1)
                train_rel_ch[c] += (diff / (norm + 1e-8)).mean().item()

        # ========================================================
        # EVAL
        # ========================================================
        model.eval()
        test_l2 = 0.0
        test_mse_ch  = torch.zeros(4).to(device)
        test_rel_ch  = torch.zeros(4).to(device)

        with torch.no_grad():
            for x, y_n, y_p, fluid_mask in test_loader:
                x, y_n, y_p = x.to(device), y_n.to(device), y_p.to(device)

                out_n = model(x)
                test_l2 += data_loss_fn(
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

        avg_data = data_loss_sum / len(train_loader)
        avg_pde  = pde_loss_sum  / len(train_loader)
        avg_test = test_l2       / len(test_loader)

        avg_train_rel = train_rel_ch / len(train_loader)
        avg_mse  = test_mse_ch  / len(test_loader)
        avg_rel  = test_rel_ch  / len(test_loader)

        avg_cont   = pde_cont_sum   / len(train_loader)
        avg_mou    = pde_mou_sum    / len(train_loader)
        avg_mov    = pde_mov_sum    / len(train_loader)
        avg_ene    = pde_energy_sum / len(train_loader)
        avg_pde    = avg_cont + avg_mou + avg_mov + avg_ene
        weighted_pde = lambda_pde * avg_pde
        total_loss = avg_data + weighted_pde

        # CSV row
        log_row = [
            ep, round(t2 - t1, 2), avg_data,
            avg_cont, avg_mou, avg_mov, avg_ene,
            avg_pde, weighted_pde,total_loss,avg_test
        ]

        for i in range(4):
            log_row.append(avg_train_rel[i].item())
        for i in range(4):
            log_row.extend([avg_mse[i].item(), avg_rel[i].item()])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(log_row)

        print(
            f"Epoch {ep} | Time {t2-t1:.1f}s | Data {avg_data:.5f} | PDE {avg_pde:.5f} | "
            f"λ_cur {lambda_pde_current:.6f} | weighted_PDE {lambda_pde_current * avg_pde:.5f} | "
            f"Test {avg_test:.5f}"
        )
        
    # ========================================================
    # SAVE
    # ========================================================
    torch.save(model.state_dict(), f"model/{cfg['tag']}.pt")
    print(f"[DONE] Model saved: model/{cfg['tag']}.pt")

    ################################################################
    # Prediction on test set (per-sample, batch_size=1)
    ################################################################
    pred = torch.zeros(y_test.shape)
    index = 0

    test_loader_single = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_enc, y_test),
        batch_size=1, shuffle=False
    )

    model.eval()
    print("Starting inference for saving .mat file...")
    with torch.no_grad():
        for x, y in test_loader_single:
            x = x.to(device)
            y = y.to(device)

            out = model(x)  # (1, H, W, 4)
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
    print("\n" + "=" * 50)
    print(f" FINAL TEST REPORT")
    print("=" * 50)

    for i, name in enumerate(OUTPUT_COLS):
        ch_acc = (1 - avg_rel[i].item()) * 100
        print(f" [{name:11}] Rel_L2: {avg_rel[i].item():.6f}")
    print("=" * 50 + "\n")

    save_path = f"pred/{cfg['tag']}.mat"
    scipy.io.savemat(save_path, {
        "pred": pred.numpy(),
        "truth": y_test.numpy(),
        "uin": uin_test if uin_test is not None else np.array([])
    })

CONFIGS = [
    {
        'train_path'    : 'data/2d_s64_train_v2.mat',
        'test_path'     : 'data/2d_s64_test_gap_v2.mat',
        'normalizer'    : 'minmax',
        'scheduler'     : "none",
        'modes'         : 25,
        'width'         : 128,
        'batch_size'    : 20,
        'epochs'        : 500,
        'learning_rate' : 1e-3,
        'weight_decay'  : 1e-4,
        'Lx'            : 0.050,
        'Ly'            : 0.020,
        'L_ref'         : 0.050,
        'U_ref'         : 0.20,
        'dT_ref'        : 50.0,
        'rho'           : 1.225,
        'cp'            : 1006.0,
        'lambda_pde'    : 0.001,
        'tag'           : 'PIFNO_lambda0.001_minmax_ep500_nowarmup'
    }
]

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}] Running: {cfg['tag']}")
        run(cfg, device)