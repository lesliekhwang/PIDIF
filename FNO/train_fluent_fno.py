"""Train the flexible FNO on Fluent subdomain MAT files."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from fno_model_flexible import FNO2d, UnitGaussianNormalizer


def _load_arrays(path: str):
    path_str = str(path)
    if path_str.endswith(".h5") or path_str.endswith(".hdf5"):
        import h5py
        with h5py.File(path_str, "r") as f:
            return {"inputs": f["inputs"][:], "outputs": f["outputs"][:]}
    import scipy.io
    return scipy.io.loadmat(path_str)


def load_mat_pair(train_mat: str, test_mat: str):
    train_data = _load_arrays(train_mat)
    test_data = _load_arrays(test_mat)
    x_train = torch.from_numpy(train_data["inputs"].astype("float32"))
    y_train = torch.from_numpy(train_data["outputs"].astype("float32"))
    x_test = torch.from_numpy(test_data["inputs"].astype("float32"))
    y_test = torch.from_numpy(test_data["outputs"].astype("float32"))
    return x_train, y_train, x_test, y_test


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    return torch.mean(torch.linalg.norm(pred_f - target_f, dim=1) / (torch.linalg.norm(target_f, dim=1) + eps))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train, y_train, x_test, y_test = load_mat_pair(args.train_mat, args.test_mat)

    if x_train.ndim != 4 or y_train.ndim != 4:
        raise ValueError("Expected channel-last MAT arrays shaped (N, nx, ny, C)")

    input_channels = x_train.shape[-1]
    output_channels = y_train.shape[-1]

    x_normalizer = UnitGaussianNormalizer(x_train).to(device)
    y_normalizer = UnitGaussianNormalizer(y_train).to(device)

    x_train_enc = x_normalizer.encode(x_train.to(device))
    y_train_enc = y_normalizer.encode(y_train.to(device))
    x_test_enc = x_normalizer.encode(x_test.to(device))
    y_test_dev = y_test.to(device)

    train_loader = DataLoader(
        TensorDataset(x_train_enc, y_train_enc),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = FNO2d(
        input_channels=input_channels,
        output_channels=output_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        padding=args.padding,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mse = torch.nn.MSELoss()

    print(f"device={device}")
    print(f"x_train={tuple(x_train.shape)} y_train={tuple(y_train.shape)}")
    print(f"x_test ={tuple(x_test.shape)} y_test ={tuple(y_test.shape)}")
    print(f"input_channels={input_channels} output_channels={output_channels}")

    best_rel = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_test_enc = model(x_test_enc)
                pred_test = y_normalizer.decode(pred_test_enc)
                test_rel = float(relative_l2(pred_test, y_test_dev).item())
            if test_rel < best_rel:
                best_rel = test_rel
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"epoch={epoch:04d} train_mse={train_loss:.6e} "
                f"test_rel_l2={test_rel:.6e} best={best_rel:.6e}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None and args.save_best:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config(),
        "x_normalizer": x_normalizer.state_dict(),
        "y_normalizer": y_normalizer.state_dict(),
        "best_test_rel_l2": best_rel,
        "train_mat": str(args.train_mat),
        "test_mat": str(args.test_mat),
    }
    torch.save(checkpoint, out_path)
    print(f"saved checkpoint: {out_path}")
    print(f"elapsed_sec={time.time() - start:.1f}")


def make_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--train-mat", required=True)
    p.add_argument("--test-mat", required=True)
    p.add_argument("--out", default="model/fluent_fno.pt")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--modes1", type=int, default=12)
    p.add_argument("--modes2", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--padding", type=int, default=9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--step-size", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(make_argparser().parse_args())
