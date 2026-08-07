#!/usr/bin/env python3
"""Train the existing point-set DeepONet on saved SNI A/B/C METIS subdomains."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fluent_deeponet import (
    DeepONet,
    FeatureNormalizer,
    evaluate_deeponet,
    train_deeponet_one_epoch,
)
from sni_eval_dataset import (
    SNIEvalSubdomainDataset,
    load_sni_eval_dataset_h5,
    sni_eval_collate_fn,
)


def _parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train DeepONet on SNI A/B/C random-METIS subdomains",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=module_dir / "data" / "sni_eval_metis")
    parser.add_argument("--train-data", type=Path, default=None)
    parser.add_argument("--valid-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-query-train", type=int, default=0, help="0 uses every subdomain node")
    parser.add_argument("--n-query-valid", type=int, default=0, help="0 uses every subdomain node")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _optional_count(value: int):
    return None if value <= 0 else int(value)


def _concatenate(samples, key):
    return np.concatenate(
        [np.asarray(sample[key], dtype=np.float32) for sample in samples], axis=0
    )


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("epochs, batch-size, and log-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = (args.train_data or args.data_dir / "train.h5").expanduser().resolve()
    valid_path = (args.valid_data or args.data_dir / "valid.h5").expanduser().resolve()
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(
            f"Expected {train_path} and {valid_path}; run generate_sni_eval_metis.py first"
        )
    train_data = load_sni_eval_dataset_h5(train_path)
    valid_data = load_sni_eval_dataset_h5(valid_path)
    for key in ("branch_channel_names", "trunk_channel_names", "output_channel_names"):
        if train_data[key] != valid_data[key]:
            raise ValueError(f"Train and validation {key} differ")

    branch_names = list(train_data["branch_channel_names"])
    skip_branch = [
        index
        for index, name in enumerate(branch_names)
        if name in {"x_local", "y_local"} or name.endswith("_mask") or name.endswith("_valid")
    ]
    branch_normalizer = FeatureNormalizer(
        _concatenate(train_data["samples"], "branch"), skip_indices=skip_branch
    )
    target_normalizer = FeatureNormalizer(_concatenate(train_data["samples"], "target"))

    train_dataset = SNIEvalSubdomainDataset(
        train_data["samples"],
        branch_normalizer=branch_normalizer,
        target_normalizer=target_normalizer,
        n_query_points=_optional_count(args.n_query_train),
        random_query=True,
    )
    valid_dataset = SNIEvalSubdomainDataset(
        valid_data["samples"],
        branch_normalizer=branch_normalizer,
        target_normalizer=target_normalizer,
        n_query_points=_optional_count(args.n_query_valid),
        random_query=False,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        collate_fn=sni_eval_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=sni_eval_collate_fn,
    )

    first = train_data["samples"][0]
    model = DeepONet(
        branch_input_dim=int(first["branch"].shape[1]),
        trunk_input_dim=int(first["query"].shape[1]),
        output_channels=int(first["target"].shape[1]),
        latent_dim=args.latent_dim,
        branch_point_hidden_dim=args.hidden_dim,
        branch_global_hidden_dim=args.hidden_dim,
        trunk_hidden_dim=args.hidden_dim,
        aggregation="mean",
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = model.to(device)
    # Keep the dataset's normalizer on CPU (including when DataLoader workers
    # are used) and use a separate device copy for metric decoding.
    target_normalizer_device = FeatureNormalizer.from_state_dict(
        target_normalizer.state_dict()
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.output_dir is None:
        stamp = datetime.now().strftime("sni_metis_%m%d%y_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "results" / stamp
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    history = []
    best_mse = float("inf")
    print(
        f"PDE={train_data['pde_name']} device={device} train={len(train_dataset)} "
        f"valid={len(valid_dataset)} branch_dim={first['branch'].shape[1]} "
        f"output_dim={first['target'].shape[1]}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        data_loss, _, _ = train_deeponet_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_bc=0.0,
            lambda_pde=0.0,
        )
        scheduler.step()
        metrics = evaluate_deeponet(
            model=model,
            loader=valid_loader,
            device=device,
            y_normalizer=target_normalizer_device,
        )
        row = {
            "epoch": epoch,
            "train_mse_normalized": float(data_loss),
            "valid_mse": float(metrics["mse"]),
            "valid_relative_l2": float(metrics["relative_l2"]),
            "valid_channel_relative_l2": np.asarray(
                metrics["channel_relative_l2"], dtype=np.float32
            ),
        }
        history.append(row)
        if row["valid_mse"] < best_mse:
            best_mse = row["valid_mse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config(),
                    "branch_normalizer": branch_normalizer.state_dict(),
                    "y_normalizer": target_normalizer_device.state_dict(),
                    "branch_channel_names": train_data["branch_channel_names"],
                    "trunk_channel_names": train_data["trunk_channel_names"],
                    "output_channel_names": train_data["output_channel_names"],
                    "pde_name": train_data["pde_name"],
                    "train_data": str(train_path),
                    "valid_data": str(valid_path),
                    "epoch": epoch,
                    "valid_metrics": row,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            channel_text = ", ".join(
                f"{name}={value:.3e}"
                for name, value in zip(
                    train_data["output_channel_names"], row["valid_channel_relative_l2"]
                )
            )
            print(
                f"Epoch {epoch:04d} | train normalized MSE={data_loss:.3e} | "
                f"valid MSE={row['valid_mse']:.3e} | "
                f"valid rel L2={row['valid_relative_l2']:.3e} | {channel_text}",
                flush=True,
            )

    np.savez(
        output_dir / "loss_history.npz",
        epoch=np.asarray([row["epoch"] for row in history], dtype=np.int32),
        train_mse_normalized=np.asarray(
            [row["train_mse_normalized"] for row in history], dtype=np.float32
        ),
        valid_mse=np.asarray([row["valid_mse"] for row in history], dtype=np.float32),
        valid_relative_l2=np.asarray(
            [row["valid_relative_l2"] for row in history], dtype=np.float32
        ),
    )
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        train_data=str(train_path),
        valid_data=str(valid_path),
        device=str(device),
        best_valid_mse=best_mse,
    )
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
