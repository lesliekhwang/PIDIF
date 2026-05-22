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
    x_train, y_train, uin_train, amp_train, lam_train, phase_train = load_dataset(cfg['train_path'])
    x_test, y_test, uin_test, amp_test, lam_test, phase_test = load_dataset(cfg['test_path'])

    if not torch.is_tensor(uin_train):
        uin_train = torch.tensor(uin_train, dtype=torch.float32)
    else:
        uin_train = uin_train.float()

    if not torch.is_tensor(uin_test):
        uin_test = torch.tensor(uin_test, dtype=torch.float32)
    else:
        uin_test = uin_test.float()

    amp_train = torch.tensor(amp_train, dtype=torch.float32) if not torch.is_tensor(amp_train) else amp_train.float()
    lam_train = torch.tensor(lam_train, dtype=torch.float32) if not torch.is_tensor(lam_train) else lam_train.float()
    phase_train = torch.tensor(phase_train, dtype=torch.float32) if not torch.is_tensor(phase_train) else phase_train.float()

    amp_test = torch.tensor(amp_test, dtype=torch.float32) if not torch.is_tensor(amp_test) else amp_test.float()
    lam_test = torch.tensor(lam_test, dtype=torch.float32) if not torch.is_tensor(lam_test) else lam_test.float()
    phase_test = torch.tensor(phase_test, dtype=torch.float32) if not torch.is_tensor(phase_test) else phase_test.float()

    fluid_mask_train = x_train[..., 0]
    fluid_mask_test  = x_test[..., 0]

    # -----------------------------
    # normalization
    # -----------------------------
    x_normalizer = PhysicsAwareInputNormalizer()
    y_normalizer = PhysicsAwareOutputNormalizer()

    x_train_enc = x_normalizer.encode(x_train, uin_train)
    x_test_enc  = x_normalizer.encode(x_test, uin_test)
    y_train_enc = y_normalizer.encode(y_train, uin_train)
    y_test_enc  = y_normalizer.encode(y_test, uin_test)

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
        torch.utils.data.TensorDataset(x_train_enc, y_train_enc, fluid_mask_train, uin_train, amp_train, lam_train, phase_train),
        batch_size=cfg['batch_size'], shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_enc, y_test_enc, y_test, fluid_mask_test, uin_test, amp_test, lam_test, phase_test),
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

    Lx = cfg.get("Lx", 0.050)
    Ly = cfg.get("Ly", 0.020)

    lambda_data = cfg.get ("lambda_data", 1.0)
    lambda_pde = cfg["lambda_pde"]
    lambda_bc = cfg.get("lambda_bc", 1.0)

    os.makedirs("pred", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    log_path = f"pred/{cfg['tag']}.csv"
    OUTPUT_COLS = ["pressure", "temperature", "x_velocity", "y_velocity"]

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "time", "data_loss", "weighted_data_loss",
                  "inlet_loss", "wall_loss", "outlet_loss", "boundary_loss", "weighted_boundary_loss",
                  "pde_continuity", "pde_mom_u", "pde_mom_v", "pde_energy", "pde_total", "weighted_pde_loss",
                  "total_loss", "test_l2"]

        for col in OUTPUT_COLS:
            header.append(f"{col}_train_rel")
        for col in OUTPUT_COLS:
            header.extend([f"{col}_test_mse", f"{col}_test_rel"])
        writer.writerow(header)
    
    # with torch.no_grad():
    #     x_sample = x_train_enc[:20].to(device)
    #     y_sample = y_train_enc[:20].to(device)
    #     uin_sample = uin_train[:20].to(device)
    #     mask_sample = fluid_mask_train[:20].to(device)
    #     amp_sample = amp_train[:20].to(device)
    #     lam_sample = lam_train[:20].to(device)
    #     phase_sample = phase_train[:20].to(device)

        # cont, mu, mv, ene = pde_residual(y_sample, uin=uin_sample, Lx=Lx, Ly=Ly)
        
        # cont, mu, mv, ene = pde_residual_wavy(
        #     y_sample, uin=uin_sample, amp=amp_sample, lam=lam_sample, phase=phase_sample, Lx=Lx, Ly=Ly
        # )
        # print(f"[GT PDE residual]")
        # print(f"  Cont:   {masked_residual_mse(cont,  mask_sample).item():.4e}")
        # print(f"  Mom-u:  {masked_residual_mse(mu,    mask_sample).item():.4e}")
        # print(f"  Mom-v:  {masked_residual_mse(mv,    mask_sample).item():.4e}")
        # print(f"  Energy: {masked_residual_mse(ene,   mask_sample).item():.4e}")
        # print(f"y_train_enc stats:")
        # print(f"  P  min/max: {y_sample[...,0].min():.3f} / {y_sample[...,0].max():.3f}")
        # print(f"  T  min/max: {y_sample[...,1].min():.3f} / {y_sample[...,1].max():.3f}")
        # print(f"  U  min/max: {y_sample[...,2].min():.3f} / {y_sample[...,2].max():.3f}")
        # print(f"  V  min/max: {y_sample[...,3].min():.3f} / {y_sample[...,3].max():.3f}")
        # print(f"uin_sample: {uin_sample[:5]}")
        # print(f"uin_train raw:     {uin_train[:5]}")
        # print(f"uin_sample(device): {uin_sample[:5]}")
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
        bc_loss_sum = 0.0
        bc_inlet_sum = 0.0
        bc_outlet_sum = 0.0
        bc_wall_sum = 0.0

        train_rel_ch = torch.zeros(4).to(device)

        # for x, y, fluid_mask, uin_batch in train_loader:
        for x, y, fluid_mask, uin_batch, amp_batch, lam_batch, phase_batch in train_loader:
            x = x.to(device)
            y = y.to(device)
            fluid_mask = fluid_mask.to(device)
            uin_batch = uin_batch.to(device)
            amp_batch = amp_batch.to(device)
            lam_batch = lam_batch.to(device)
            phase_batch = phase_batch.to(device)

            optimizer.zero_grad()

            # forward
            pred = model(x)

            # data_loss = data_loss_fn(pred.reshape(x.shape[0], -1),y.reshape(x.shape[0], -1))
            data_loss = masked_data_rel_l2(pred, y, fluid_mask)

            # physics loss in nondimensional space
            continuity, mom_u, mom_v, energy = pde_residual_wavy(
                pred, uin=uin_batch, amp=amp_batch, lam=lam_batch, phase=phase_batch, Lx=Lx, Ly=Ly
            )

            pred_phys = y_normalizer.decode(pred, uin_batch)

            pde_cont = masked_residual_mse(continuity, fluid_mask)
            pde_mou  = masked_residual_mse(mom_u, fluid_mask)
            pde_mov  = masked_residual_mse(mom_v, fluid_mask)
            pde_ene  = masked_residual_mse(energy, fluid_mask)

            pde_loss = pde_cont + pde_mou + pde_mov + pde_ene

            bc_loss, bc_inlet, bc_outlet, bc_wall = boundary_condition_loss(pred, x)

            # total loss
            # warmup_ep = cfg["warmup_epochs"]
            # if ep < warmup_ep:
            #     progress = ep / warmup_ep
            #     lambda_pde_current = lambda_pde * (1 - torch.cos(torch.tensor(progress * torch.pi)).item()) / 2
            # else:
            #     lambda_pde_current = lambda_pde

            pde_start_epoch = cfg["pde_start_epoch"]
            if ep < pde_start_epoch:
                lambda_pde_current = 0.0
            else:
                lambda_pde_current = lambda_pde

            loss = lambda_data * data_loss + lambda_pde_current * pde_loss + lambda_bc * bc_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            # scheduler.step()

            data_loss_sum += data_loss.item()
            pde_loss_sum  += pde_loss.item()
            pde_cont_sum   += pde_cont.item()
            pde_mou_sum    += pde_mou.item()
            pde_mov_sum    += pde_mov.item()
            pde_energy_sum += pde_ene.item()
            bc_loss_sum += bc_loss.item()
            bc_inlet_sum += bc_inlet.item()
            bc_outlet_sum += bc_outlet.item()
            bc_wall_sum += bc_wall.item()

            # per-channel train rel (physical space)
            y_phys = y_normalizer.decode(y, uin_batch)
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
            # for x, y_n, y_p, fluid_mask, uin_batch in test_loader:
            for x, y_n, y_p, fluid_mask, uin_batch, amp_batch, lam_batch, phase_batch in test_loader:
                x = x.to(device)
                y_n = y_n.to(device)
                y_p = y_p.to(device)
                fluid_mask = fluid_mask.to(device)
                uin_batch = uin_batch.to(device)
                amp_batch = amp_batch.to(device)
                lam_batch = lam_batch.to(device)
                phase_batch = phase_batch.to(device)

                out_n = model(x)
                # test_l2 += data_loss_fn(out_n.reshape(x.shape[0], -1),y_n.reshape(x.shape[0], -1)).item()
                test_l2 += masked_data_rel_l2(out_n, y_n, fluid_mask).item()

                out_p = y_normalizer.decode(out_n, uin_batch)
                for c in range(4):
                    test_mse_ch[c] += F.mse_loss(out_p[..., c], y_p[..., c]).item()
                    diff = torch.norm((out_p[..., c] - y_p[..., c]).reshape(x.shape[0], -1), p=2, dim=1)
                    norm = torch.norm(y_p[..., c].reshape(x.shape[0], -1), p=2, dim=1)
                    test_rel_ch[c] += (diff / (norm + 1e-8)).mean().item()

        t2 = default_timer()

        avg_data = data_loss_sum / len(train_loader)
        avg_test = test_l2       / len(test_loader)

        avg_train_rel = train_rel_ch / len(train_loader)
        avg_mse = test_mse_ch  / len(test_loader)
        avg_rel = test_rel_ch  / len(test_loader)

        avg_cont = pde_cont_sum   / len(train_loader)
        avg_mou = pde_mou_sum    / len(train_loader)
        avg_mov = pde_mov_sum    / len(train_loader)
        avg_ene = pde_energy_sum / len(train_loader)
        avg_pde = pde_loss_sum  / len(train_loader)

        avg_bc_inlet = bc_inlet_sum / len(train_loader)
        avg_bc_outlet = bc_outlet_sum / len(train_loader)
        avg_bc_wall = bc_wall_sum / len(train_loader)
        avg_bc = bc_loss_sum / len(train_loader)
        
        weighted_data = lambda_data * avg_data
        weighted_pde = lambda_pde_current * avg_pde
        weighted_bc = lambda_bc * avg_bc

        total_loss = weighted_data + weighted_pde + weighted_bc

        # CSV row
        log_row = [ep, round(t2 - t1, 2), avg_data, weighted_data, 
                   avg_bc_inlet, avg_bc_wall, avg_bc_outlet, avg_bc, weighted_bc,
                   avg_cont, avg_mou, avg_mov, avg_ene, avg_pde, weighted_pde, total_loss, avg_test]

        for i in range(4):
            log_row.append(avg_train_rel[i].item())
        for i in range(4):
            log_row.extend([avg_mse[i].item(), avg_rel[i].item()])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(log_row)

        print(f"Epoch {ep} | Total {total_loss:.4f} | Data {avg_data:.4f} | PDE {avg_pde:.4f} | BC {avg_bc:.4f} | Test {avg_test:.4f}")

        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            print(
                f"  Weighted | "
                f"Data {weighted_data:.3f} | "
                f"PDE {weighted_pde:.3f} | "
                f"BC {weighted_bc:.3f}"
            )
            print(
                f"  PDE | "
                f"Cont {avg_cont:.3f} | "
                f"Mom-u {avg_mou:.3f} | "
                f"Mom-v {avg_mov:.3f} | "
                f"Energy {avg_ene:.3f}"
            )
            print(
                f"  BC  | "
                f"Inlet {avg_bc_inlet:.3f} | "
                f"Wall {avg_bc_wall:.3f} | "
                f"Outlet {avg_bc_outlet:.3f}"
            )
            print(
                f"  Rel | "
                f"P {avg_rel[0].item():.3f} | "
                f"T {avg_rel[1].item():.3f} | "
                f"U {avg_rel[2].item():.3f} | "
                f"V {avg_rel[3].item():.3f}"
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

    # test_loader_single = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(x_test_enc, y_test, uin_test),
    #     batch_size=1,
    #     shuffle=False
    # )
    test_loader_single = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_enc, y_test, uin_test, amp_test, lam_test, phase_test),
        batch_size=1,
        shuffle=False
    )

    model.eval()
    print("Starting inference for saving .mat file...")
    with torch.no_grad():
        # for x, y, uin_batch in test_loader_single:
        for x, y, uin_batch, amp_b, lam_b, phase_b in test_loader_single:
            x = x.to(device)
            y = y.to(device)
            uin_batch = uin_batch.to(device)

            out = model(x)  # (1, H, W, 4)
            out = y_normalizer.decode(out, uin_batch)

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
        'train_path'    : 'data/2d_s64_train_new.mat',
        'test_path'     : 'data/2d_s64_test_gap_new.mat',
        'scheduler'     : "none",
        'modes'         : 25,
        'width'         : 128,
        'batch_size'    : 20,
        'epochs'        : 1000,
        'warmup_epochs' : 0,
        'pde_start_epoch': 0,
        'learning_rate' : 1e-3,
        'weight_decay'  : 1e-4,
        'Lx'            : 0.050,
        'Ly'            : 0.020,
        'lambda_data'   : 1.0,
        'lambda_pde'    : 0.1,
        'lambda_bc'     : 0.0,
        'tag'           : 'test_new_v2'
    }
]

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}] Running: {cfg['tag']}")
        run(cfg, device)