from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pidiffusion.data import normalize_diffusion_branch

from pidiffusion.interface_physics import (
    InterfacePhysicsConfig,
    auto_loss_scales,
    interface_physics_loss_torch,
    subdomain_scales_from_metadata,
)

from experiments.evaluate_diffusion_generation import (
    SCHEDULE_FAMILY_PROGRESSIVE_NESTED20,
    build_model,
    build_sampling_timesteps,
    load_checkpoint,
    require_checkpoint_field,
)

from experiments.unknown_interface_diffusion_utils import (
    build_edge_queries,
    build_schedule,
    find_edge_rows,
    load_case_realization,
    load_metadata_for_records,
    load_target_normalizer,
    sample_ddim_differentiable_batched,
    write_shared_z,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "distill_progressive"
    / "distill_nested10_to5_stage2_5ep_seed0"
    / "stage2_best.pt"
)

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "channel_deeponet_style_pressure_u_v_random5_val.h5"
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def make_fixed_noise(
    *,
    n_subdomains: int,
    n_points: int,
    target_dim: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    return torch.randn(
        (n_subdomains * n_points, target_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )


def rollout_edge_fixed_noise(
    *,
    model,
    branch: torch.Tensor,
    q_edge: torch.Tensor,
    fixed_noise: torch.Tensor,
    schedule,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    n_subdomains, n_points, _ = q_edge.shape

    query_flat = q_edge.reshape(
        n_subdomains * n_points,
        2,
    )

    query_batch_id = torch.arange(
        n_subdomains,
        dtype=torch.long,
        device=q_edge.device,
    ).repeat_interleave(n_points)

    if fixed_noise.shape != (
        n_subdomains * n_points,
        model.target_dim,
    ):
        raise ValueError(
            f"Fixed noise has shape {tuple(fixed_noise.shape)}, "
            f"expected "
            f"{(n_subdomains * n_points, model.target_dim)}"
        )

    output_flat = sample_ddim_differentiable_batched(
        model=model,
        branch=branch,
        query=query_flat,
        initial_noise=fixed_noise,
        query_batch_id=query_batch_id,
        schedule=schedule,
        timesteps=timesteps,
    )

    return output_flat.reshape(
        n_subdomains,
        n_points,
        model.target_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
    )
    parser.add_argument(
        "--case-id",
        default="channel_08",
    )
    parser.add_argument(
        "--realization-id",
        type=int,
        default=0,
    )

    # Senior parity:
    # outer loop max_iter = 50,
    # torch LBFGS internal max_iter = 20.
    parser.add_argument(
        "--outer-max-iter",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lbfgs-inner-max-iter",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-2,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--noise-seed-left",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--noise-seed-right",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute LBFGS optimization.",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "optimize_unknown_interfaces"
        ),
    ) 

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.run_name is None:
        run_name = (
            f"distilled5_senior_parity_"
            f"{args.case_id}_real{args.realization_id}"
        )
    else:
        run_name = args.run_name

    run_dir = args.results_root / run_name

    if args.run and run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}"
        )

    if args.outer_max_iter <= 0:
        raise ValueError(
            "--outer-max-iter must be positive"
        )

    if args.lbfgs_inner_max_iter <= 0:
        raise ValueError(
            "--lbfgs-inner-max-iter must be positive"
        )

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but CUDA is unavailable"
            )
        if (
            device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Requested {device}, but only "
                f"{torch.cuda.device_count()} CUDA devices "
                "are visible"
            )

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    model = build_model(
        checkpoint,
        device,
    )

    normalizer = load_target_normalizer(
        checkpoint
    ).to(device)

    branch_channel_names = list(
        require_checkpoint_field(
            checkpoint,
            "branch_channel_names",
        )
    )

    output_channel_names = list(
        require_checkpoint_field(
            checkpoint,
            "output_channel_names",
        )
    )

    if output_channel_names != [
        "pressure",
        "u",
        "v",
    ]:
        raise RuntimeError(
            f"Unexpected output channels: "
            f"{output_channel_names}"
        )

    local_aspect_mean = float(
        checkpoint["local_aspect_mean"]
    )
    local_aspect_std = float(
        checkpoint["local_aspect_std"]
    )

    records = load_case_realization(
        dataset_path=args.dataset,
        case_id=args.case_id,
        realization_id=args.realization_id,
    )

    metadata = load_metadata_for_records(
        args.dataset,
        records,
    )

    branches_np = np.stack(
        [
            normalize_diffusion_branch(
                branch=record["branch"],
                branch_channel_names=branch_channel_names,
                target_normalizer=normalizer,
                local_aspect_mean=local_aspect_mean,
                local_aspect_std=local_aspect_std,
            )
            for record in records
        ],
        axis=0,
    ).astype(np.float32)

    left_rows, right_rows = find_edge_rows(
        branches_np,
        branch_channel_names,
    )

    base_branch = torch.from_numpy(
        branches_np
    ).to(device)

    q_left_base, q_right_base = build_edge_queries(
        branches_np=branches_np,
        left_rows=left_rows,
        right_rows=right_rows,
        names=branch_channel_names,
        device=device,
    )

    # These base coordinates themselves are not optimization variables.
    q_left_base = q_left_base.detach()
    q_right_base = q_right_base.detach()

    output_mean = (
        normalizer.mean
        .detach()
        .to(device)
        .reshape(-1)
    )

    output_std = (
        normalizer.std
        .detach()
        .to(device)
        .reshape(-1)
    )

    # Senior notebook:
    # init_value=[0.0, 0.0, 0.0], init_noise_std=0.
    physical_init = torch.zeros(
        3,
        dtype=output_mean.dtype,
        device=device,
    )

    z_init_one = (
        physical_init - output_mean
    ) / output_std

    z_initial = (
        z_init_one
        .reshape(1, 1, 3)
        .expand(9, 256, 3)
        .clone()
        .detach()
    )

    z = torch.nn.Parameter(
        z_initial.clone()
    )

    schedule = build_schedule(
        checkpoint,
        device,
    )

    diffusion_config = dict(
        checkpoint["diffusion_config"]
    )

    timesteps = build_sampling_timesteps(
        sampling_steps=5,
        total_diffusion_steps=int(
            diffusion_config["T"]
        ),
        schedule_family=(
            SCHEDULE_FAMILY_PROGRESSIVE_NESTED20
        ),
        device=device,
    )

    physics_config = InterfacePhysicsConfig(
        viscosity=1.003e-3,
        length_unit_scale=1.0e-3,
        alpha_traction=0.1,
        alpha_flux=0.3,
        alpha_dirichlet=10.0,
        alpha_p=1.0,
        alpha_u=1.0,
        alpha_v=1.0,
        alpha_smooth=1.0e-4,
        alpha_value_l2=0.0,
        traction_scale=None,
        flux_scale=None,
    )

    x_scale, y_scale = subdomain_scales_from_metadata(
        metadata=metadata,
        n_subdomains=10,
        length_unit_scale=(
            physics_config.length_unit_scale
        ),
        device=device,
    )

    # Match the DeepONet reference: compute these scales once,
    # before optimization.
    traction_scale, flux_scale = auto_loss_scales(
        output_std=output_std,
        output_channel_names=output_channel_names,
        q_right=q_right_base,
        y_scale=y_scale,
        config=physics_config,
    )

    # Generate fixed noise ONCE.
    fixed_noise_left = make_fixed_noise(
        n_subdomains=10,
        n_points=256,
        target_dim=model.target_dim,
        seed=args.noise_seed_left,
        dtype=base_branch.dtype,
        device=device,
    )

    fixed_noise_right = make_fixed_noise(
        n_subdomains=10,
        n_points=256,
        target_dim=model.target_dim,
        seed=args.noise_seed_right,
        dtype=base_branch.dtype,
        device=device,
    )

    print()
    print("=" * 78)
    print("Distilled5 unknown-interface physics optimization")
    print("=" * 78)
    print(f"Checkpoint             : {args.checkpoint}")
    print(f"Dataset                : {args.dataset}")
    print(f"Case                   : {args.case_id}")
    print(f"Realization            : {args.realization_id}")
    print(f"Device                 : {device}")
    print(
        f"Nested5 timesteps      : "
        f"{timesteps.detach().cpu().tolist()}"
    )
    print(
        f"Noise seeds            : "
        f"left={args.noise_seed_left}, "
        f"right={args.noise_seed_right}"
    )
    print(
        f"Outer max iterations   : "
        f"{args.outer_max_iter}"
    )
    print(
        f"LBFGS inner max_iter   : "
        f"{args.lbfgs_inner_max_iter}"
    )
    print(f"LBFGS lr               : {args.lr:g}")
    print("LBFGS line search      : strong_wolfe")
    print(f"Tolerance              : {args.tol:.3e}")
    print(
        f"Traction scale         : "
        f"{traction_scale:.8e}"
    )
    print(
        f"Flux scale             : "
        f"{flux_scale:.8e}"
    )
    print(
        "Physical z init        : "
        "[0.0 Pa, 0.0 m/s, 0.0 m/s]"
    )
    print(
        f"Normalized z init      : "
        f"{z_init_one.detach().cpu().tolist()}"
    )

    if not args.run:
        print()
        print("DRY RUN ONLY")
        print(
            "Add --run to execute the LBFGS optimization."
        )
        print("=" * 78)
        return
    
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config_record = {
        "protocol": "distilled5_unknown_interface_physics_parity_v1",

        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),

        "dataset": str(args.dataset.resolve()),

        "case_id": args.case_id,
        "realization_id": int(args.realization_id),
        "device": str(device),

        "sampling": {
            "schedule_family": "progressive_nested20",
            "nfe": 5,
            "timesteps": [
                int(x)
                for x in timesteps.detach().cpu().tolist()
            ],
            "noise_seed_left": int(args.noise_seed_left),
            "noise_seed_right": int(args.noise_seed_right),
            "fixed_noise_during_optimization": True,
        },

        "optimizer": {
            "name": "lbfgs",
            "lr": float(args.lr),
            "outer_max_iter": int(args.outer_max_iter),
            "inner_max_iter": int(args.lbfgs_inner_max_iter),
            "line_search_fn": "strong_wolfe",
            "tol": float(args.tol),
            "stagnation_patience": int(args.stagnation_patience),
        },

        "physics": {
            "viscosity": float(physics_config.viscosity),
            "length_unit_scale": float(
                physics_config.length_unit_scale
            ),
            "alpha_traction": float(
                physics_config.alpha_traction
            ),
            "alpha_flux": float(
                physics_config.alpha_flux
            ),
            "alpha_dirichlet": float(
                physics_config.alpha_dirichlet
            ),
            "alpha_p": float(physics_config.alpha_p),
            "alpha_u": float(physics_config.alpha_u),
            "alpha_v": float(physics_config.alpha_v),
            "alpha_smooth": float(
                physics_config.alpha_smooth
            ),
            "alpha_value_l2": float(
                physics_config.alpha_value_l2
            ),
            "traction_scale": float(traction_scale),
            "flux_scale": float(flux_scale),
        },

        "interface": {
            "n_subdomains": 10,
            "n_internal_interfaces": 9,
            "points_per_interface": 256,
            "optimized_fields": [
                "pressure",
                "u",
                "v",
            ],
            "physical_initial_value": [
                0.0,
                0.0,
                0.0,
            ],
            "normalized_initial_value": (
                z_init_one.detach().cpu().tolist()
            ),
        },
    }

    with (
        run_dir / "config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            config_record,
            handle,
            indent=2,
        )

    optimizer = torch.optim.LBFGS(
        [z],
        lr=float(args.lr),
        max_iter=int(
            args.lbfgs_inner_max_iter
        ),
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def compute_physics_loss():
        # Fresh coordinate leaves for every closure.
        # Traction requires gradients wrt these coordinates.
        q_left = (
            q_left_base
            .detach()
            .clone()
            .requires_grad_(True)
        )

        q_right = (
            q_right_base
            .detach()
            .clone()
            .requires_grad_(True)
        )

        branch = write_shared_z(
            base_branch=base_branch,
            z=z,
            left_rows=left_rows,
            right_rows=right_rows,
            names=branch_channel_names,
        )

        out_left_norm = rollout_edge_fixed_noise(
            model=model,
            branch=branch,
            q_edge=q_left,
            fixed_noise=fixed_noise_left,
            schedule=schedule,
            timesteps=timesteps,
        )

        out_right_norm = rollout_edge_fixed_noise(
            model=model,
            branch=branch,
            q_edge=q_right,
            fixed_noise=fixed_noise_right,
            schedule=schedule,
            timesteps=timesteps,
        )

        loss, info = interface_physics_loss_torch(
            out_left_norm=out_left_norm,
            out_right_norm=out_right_norm,
            z_norm=z,
            q_left_base=q_left,
            q_right_base=q_right,
            x_scale=x_scale,
            y_scale=y_scale,
            output_mean=output_mean,
            output_std=output_std,
            output_channel_names=output_channel_names,
            optimized_output_channels=(0, 1, 2),
            config=physics_config,
            pressure_offsets=None,
            traction_scale=traction_scale,
            flux_scale=flux_scale,
        )

        return loss, info

    history = []
    prev_loss = None
    stagnant_iters = 0
    converged = False
    stop_reason = "max_iterations"

    total_start = time.perf_counter()

    for outer_iteration in range(
        1,
        args.outer_max_iter + 1,
    ):
        outer_start = time.perf_counter()
        calls_before = closure_calls

        def lbfgs_closure():
            nonlocal closure_calls

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, _ = compute_physics_loss()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite physics loss inside "
                    "LBFGS closure"
                )

            loss.backward()

            if z.grad is None:
                raise RuntimeError(
                    "z.grad is None inside LBFGS closure"
                )

            if not torch.isfinite(
                z.grad
            ).all():
                raise RuntimeError(
                    "Non-finite z gradient inside "
                    "LBFGS closure"
                )

            closure_calls += 1
            return loss

        optimizer.step(
            lbfgs_closure
        )

        # Match senior implementation: evaluate once after
        # optimizer.step() for history/stopping.
        loss, info = compute_physics_loss()

        row = {
            key: float(
                value.detach().cpu()
            )
            for key, value in info.items()
        }

        current_loss = row["loss"]

        convergence_metric = (
            row["traction"]
            + row["flux"]
            + row["dirichlet"]
        )

        converged = bool(
            convergence_metric
            <= float(args.tol)
        )

        if (
            prev_loss is not None
            and np.isclose(
                current_loss,
                prev_loss,
                rtol=0.0,
                atol=max(
                    float(args.tol),
                    1.0e-12,
                ),
            )
        ):
            stagnant_iters += 1
        else:
            stagnant_iters = 0

        prev_loss = current_loss

        loss_stagnant = (
            stagnant_iters
            >= args.stagnation_patience
        )

        outer_seconds = (
            time.perf_counter()
            - outer_start
        )

        calls_this_outer = (
            closure_calls - calls_before
        )

        row.update(
            {
                "outer_iteration": outer_iteration,
                "closure_calls_this_outer": (
                    calls_this_outer
                ),
                "closure_calls_total": closure_calls,
                "convergence_metric": (
                    convergence_metric
                ),
                "outer_seconds": outer_seconds,
            }
        )

        history.append(row)

        print(
            f"outer={outer_iteration:04d} | "
            f"loss={row['loss']:.6e} | "
            f"traction={row['traction']:.6e} | "
            f"flux={row['flux']:.6e} | "
            f"dirichlet={row['dirichlet']:.6e} | "
            f"smooth={row['smooth']:.6e} | "
            f"closures={calls_this_outer} | "
            f"time={outer_seconds:.2f}s",
            flush=True,
        )

        if converged:
            stop_reason = "physics_tolerance"
            print(
                "Stopping: senior-parity convergence "
                "criterion reached."
            )
            break

        if loss_stagnant:
            stop_reason = "stagnation"
            print(
                "Stopping: total loss unchanged for "
                f"{args.stagnation_patience} outer "
                "iterations."
            )
            break

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    z_final_norm = (
        z.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    mean_np = (
        output_mean.detach()
        .cpu()
        .numpy()
        .reshape(1, 1, 3)
    )

    std_np = (
        output_std.detach()
        .cpu()
        .numpy()
        .reshape(1, 1, 3)
    )

    z_final_phys = (
        z_final_norm * std_np
        + mean_np
    ).astype(np.float32)

    final_row = history[-1]

    history_path = (
        run_dir / "physics_loss_history.csv"
    )

    fieldnames = list(history[0].keys())

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)

    np.savez_compressed(
        run_dir / "interface_states.npz",

        z_initial_normalized=(
            z_initial
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        z_final_normalized=(
            z_final_norm
        ),

        z_initial_physical=np.zeros(
            (9, 256, 3),
            dtype=np.float32,
        ),

        z_final_physical=(
            z_final_phys
        ),

        output_mean=(
            output_mean
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        output_std=(
            output_std
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),

        field_names=np.asarray(
            ["pressure", "u", "v"]
        ),

        interface_ids=np.arange(
            9,
            dtype=np.int32,
        ),
    )

    summary = {
        "protocol": "distilled5_unknown_interface_senior_parity_v1",

        "case_id": args.case_id,
        "realization_id": int(args.realization_id),

        "stop_reason": stop_reason,
        "converged": bool(converged),

        "outer_iterations": int(len(history)),
        "closure_calls": int(closure_calls),
        "total_wall_seconds": float(total_seconds),

        "final": {
            "loss": float(final_row["loss"]),
            "traction": float(
                final_row["traction"]
            ),
            "flux": float(
                final_row["flux"]
            ),
            "dirichlet": float(
                final_row["dirichlet"]
            ),
            "smooth": float(
                final_row["smooth"]
            ),
            "value_l2": float(
                final_row["value_l2"]
            ),
            "convergence_metric": float(
                final_row["traction"]
                + final_row["flux"]
                + final_row["dirichlet"]
            ),
        },

        "artifacts": {
            "config": "config.json",
            "history": "physics_loss_history.csv",
            "interfaces": "interface_states.npz",
        },
    }

    with (
        run_dir / "summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("=" * 78)
    print("Optimization summary")
    print("=" * 78)
    print(
        f"Outer iterations done  : "
        f"{len(history)}"
    )
    print(
        f"Total closure calls    : "
        f"{closure_calls}"
    )
    print(
        f"Total wall time        : "
        f"{total_seconds:.2f} s"
    )
    print(
        f"Final loss             : "
        f"{final_row['loss']:.8e}"
    )
    print(
        f"Final traction         : "
        f"{final_row['traction']:.8e}"
    )
    print(
        f"Final flux             : "
        f"{final_row['flux']:.8e}"
    )
    print(
        f"Final dirichlet        : "
        f"{final_row['dirichlet']:.8e}"
    )
    print(
        f"Final smooth           : "
        f"{final_row['smooth']:.8e}"
    )
    print(
        f"Converged              : "
        f"{converged}"
    )
    print(
        f"Stop reason            : "
        f"{stop_reason}"
    )
    print(
        f"z finite               : "
        f"{bool(np.isfinite(z_final_phys).all())}"
    )
    print(
        f"Saved run              : "
        f"{run_dir}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
