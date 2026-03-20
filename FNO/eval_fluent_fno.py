"""Evaluate a flexible Fluent FNO checkpoint and save predictions to MAT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fno_model_flexible import FNO2d, UnitGaussianNormalizer


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "Expected a checkpoint created by train_fluent_fno.py with key "
            "'model_state_dict'. Old raw state_dict files do not contain "
            "normalizer statistics."
        )
    return ckpt


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)

    if str(args.test_mat).endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(args.test_mat, "r") as f:
            test_data = {"inputs": f["inputs"][:], "outputs": f["outputs"][:]}
            for key in ["aspect_ratio", "subdomain_id", "x_left_mm", "x_right_mm", "y_bottom_mm", "y_top_mm"]:
                if key in f:
                    test_data[key] = f[key][:]
    else:
        import scipy.io
        test_data = scipy.io.loadmat(args.test_mat)
    x_test = torch.from_numpy(test_data["inputs"].astype("float32")).to(device)
    y_test = torch.from_numpy(test_data["outputs"].astype("float32"))

    config = dict(ckpt["model_config"])
    model = FNO2d(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_normalizer = UnitGaussianNormalizer.from_state_dict(ckpt["x_normalizer"]).to(device)
    y_normalizer = UnitGaussianNormalizer.from_state_dict(ckpt["y_normalizer"]).to(device)

    with torch.no_grad():
        pred_enc = model(x_normalizer.encode(x_test))
        pred = y_normalizer.decode(pred_enc).cpu()

    out_path = Path(args.pred_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"pred": pred.numpy(), "truth": y_test.numpy()}
    for key in ["aspect_ratio", "subdomain_id", "x_left_mm", "x_right_mm", "y_bottom_mm", "y_top_mm"]:
        if key in test_data:
            save_dict[key] = test_data[key]
    import scipy.io
    scipy.io.savemat(out_path, save_dict)
    print(f"saved prediction: {out_path}, pred shape={tuple(pred.shape)}")


def make_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-mat", required=True)
    p.add_argument("--pred-path", default="pred/fluent_fno_pred.mat")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    evaluate(make_argparser().parse_args())
