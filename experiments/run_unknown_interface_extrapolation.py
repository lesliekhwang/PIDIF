#!/usr/bin/env python3
"""Run full OOD unknown-interface extrapolation experiments.

This script reuses experiments/run_unknown_interface_testset.py for every OOD
HDF5 listed in ood_extrapolation_dataset_manifest.json, then aggregates the
results into separate AR1 and control-point tables.

Prerequisite:
The existing unknown-interface optimizer/evaluator/test-set runner must already
support a dynamic number of subdomains (5 / 10 / 20).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "channel_diffusion_dataset"
    / "deeponet_style_dataset"
    / "ood_extrapolation_dataset_manifest.json"
)

CHECKPOINT = (
    ROOT
    / "results"
    / "distill_progressive"
    / "distill_nested10_to5_stage2_5ep_seed0"
    / "stage2_best.pt"
)

CHILD_RUNNER = ROOT / "experiments" / "run_unknown_interface_testset.py"
CHILD_RESULTS = ROOT / "results" / "run_unknown_interface_testset"
RESULTS_ROOT = ROOT / "results" / "run_unknown_interface_extrapolation"

GROUPS = [
    "ar5_h010",
    "ar20_h010",
    "ar5_h020",
    "ar20_h005",
    "large_delta",
]

GROUP_LABELS = {
    "ar5_h010": "AR=5, H=0.10 mm",
    "ar20_h010": "AR=20, H=0.10 mm",
    "ar5_h020": "AR=5, H=0.20 mm",
    "ar20_h005": "AR=20, H=0.05 mm",
    "large_delta": "Large delta",
}

DECOMPS = ["ar1", "controlpoints"]
METHODS = ["known", "zero", "physics"]
FIELDS = ["pressure", "u", "v"]


def decode(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def load_manifest(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for raw in data["outputs"]:
        row = dict(raw)
        decomp = row.get("decomposition", row.get("decomposition_mode"))
        if decomp == "control_points":
            decomp = "controlpoints"
        row["decomposition"] = decomp
        row["path"] = str(Path(row["path"]).resolve())
        rows.append(row)
    return rows


def audit_dataset(path: Path):
    with h5py.File(path, "r") as f:
        case_ids = [decode(x) for x in f["metadata"]["case_id"][:]]
        real_ids = f["metadata"]["realization_id"][:].astype(int)
        sub_ids = f["metadata"]["subdomain_id"][:].astype(int)

        grouped = {}
        for case, real, sub in zip(case_ids, real_ids, sub_ids, strict=True):
            grouped.setdefault((case, int(real)), []).append(int(sub))

        if len(grouped) != 10:
            raise RuntimeError(f"{path.name}: expected 10 channels")

        nsubs = []
        for key, ids in grouped.items():
            ids = sorted(ids)
            if ids != list(range(len(ids))):
                raise RuntimeError(
                    f"{path.name}: non-contiguous subdomains for {key}: {ids}"
                )
            nsubs.append(len(ids))

        if len(set(nsubs)) != 1:
            raise RuntimeError(f"{path.name}: inconsistent subdomain counts")

        return {
            "n_channels": len(grouped),
            "n_subdomains": nsubs[0],
            "n_interfaces": nsubs[0] - 1,
            "n_samples": int(f.attrs["n_samples"]),
        }


def child_name(group, decomp):
    return f"oodx_{decomp}_{group}"


def child_result_dir(group, decomp):
    return CHILD_RESULTS / child_name(group, decomp)


def child_command(
    dataset,
    group,
    decomp,
    checkpoint,
    device,
    resume,
    limit,
    run,
):
    cmd = [
        sys.executable,
        str(CHILD_RUNNER),
        "--dataset",
        str(dataset),
        "--testset-name",
        child_name(group, decomp),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
    ]
    if resume:
        cmd.append("--resume")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if run:
        cmd.append("--run")
    return cmd


def global_l2_column(df):
    for name in [
        "global_relative_l2",
        "global_rel_l2",
        "whole_channel_relative_l2",
        "whole_channel_rel_l2",
    ]:
        if name in df.columns:
            return name

    raise RuntimeError(
        "Child metrics_per_channel.csv has no whole-channel Global Relative L2. "
        "Please make evaluate_unknown_interface_solution.py and "
        "run_unknown_interface_testset.py save `global_relative_l2`. "
        "Do NOT use the old subdomain-averaged `avg_relative_l2` as a substitute."
    )


def load_child_metrics(group, decomp):
    path = child_result_dir(group, decomp) / "metrics_per_channel.csv"
    df = pd.read_csv(path)
    rel_col = global_l2_column(df)

    df = df[
        df["state"].isin(METHODS)
        & df["field"].isin(FIELDS)
    ].copy()

    df["geometry_ood"] = group
    df["geometry_label"] = GROUP_LABELS[group]
    df["decomposition"] = decomp
    df["method"] = df["state"]
    df["global_relative_l2"] = df[rel_col].astype(float)

    return df[
        [
            "geometry_ood",
            "geometry_label",
            "decomposition",
            "case_id",
            "realization_id",
            "method",
            "field",
            "n_subdomains",
            "n_points",
            "balanced_rmse",
            "global_relative_l2",
        ]
    ]


def make_per_channel(long_df):
    keys = [
        "geometry_ood",
        "geometry_label",
        "decomposition",
        "case_id",
        "realization_id",
        "method",
        "n_subdomains",
        "n_points",
    ]

    rows = []
    for key, part in long_df.groupby(keys, sort=False):
        by_field = {r["field"]: r for _, r in part.iterrows()}
        if any(field not in by_field for field in FIELDS):
            raise RuntimeError(f"Missing field for {key}")

        row = dict(zip(keys, key, strict=True))
        for field in FIELDS:
            row[f"{field}_rmse"] = float(by_field[field]["balanced_rmse"])
            row[f"{field}_global_rel_l2"] = float(
                by_field[field]["global_relative_l2"]
            )

        row["avg_global_rel_l2"] = np.mean(
            [row[f"{field}_global_rel_l2"] for field in FIELDS]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def make_table(channel_df, decomp):
    rows = []

    for group in GROUPS:
        for method in METHODS:
            part = channel_df[
                (channel_df["decomposition"] == decomp)
                & (channel_df["geometry_ood"] == group)
                & (channel_df["method"] == method)
            ]
            if part.empty:
                continue

            row = {
                "geometry_ood": group,
                "geometry_label": GROUP_LABELS[group],
                "method": method,
                "n_channels": len(part),
            }

            metrics = [
                "pressure_rmse",
                "u_rmse",
                "v_rmse",
                "pressure_global_rel_l2",
                "u_global_rel_l2",
                "v_global_rel_l2",
                "avg_global_rel_l2",
            ]

            for metric in metrics:
                row[f"{metric}_mean"] = part[metric].mean()
                row[f"{metric}_std"] = (
                    part[metric].std(ddof=1) if len(part) > 1 else 0.0
                )

            for field in FIELDS:
                row[f"{field}_global_rel_l2_percent_mean"] = (
                    100.0 * row[f"{field}_global_rel_l2_mean"]
                )
                row[f"{field}_global_rel_l2_percent_std"] = (
                    100.0 * row[f"{field}_global_rel_l2_std"]
                )

            row["avg_global_rel_l2_percent_mean"] = (
                100.0 * row["avg_global_rel_l2_mean"]
            )
            row["avg_global_rel_l2_percent_std"] = (
                100.0 * row["avg_global_rel_l2_std"]
            )
            row["oracle_gap_recovered_percent"] = np.nan
            rows.append(row)

    table = pd.DataFrame(rows)

    for group in GROUPS:
        known = (
            (table["geometry_ood"] == group)
            & (table["method"] == "known")
        )
        zero = (
            (table["geometry_ood"] == group)
            & (table["method"] == "zero")
        )
        physics = (
            (table["geometry_ood"] == group)
            & (table["method"] == "physics")
        )

        if zero.any():
            table.loc[zero, "oracle_gap_recovered_percent"] = 0.0

        if not (known.any() and zero.any() and physics.any()):
            continue

        lk = table.loc[known, "avg_global_rel_l2_mean"].iloc[0]
        lz = table.loc[zero, "avg_global_rel_l2_mean"].iloc[0]
        lp = table.loc[physics, "avg_global_rel_l2_mean"].iloc[0]

        denom = lz - lk
        if abs(denom) > 1e-12:
            table.loc[
                physics,
                "oracle_gap_recovered_percent",
            ] = 100.0 * (lz - lp) / denom

    return table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    results_root = args.results_root.resolve()

    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not CHILD_RUNNER.exists():
        raise FileNotFoundError(CHILD_RUNNER)

    records = load_manifest(manifest)
    by_key = {
        (r["decomposition"], r["ood_group"]): r
        for r in records
    }

    expected = [
        (decomp, group)
        for decomp in DECOMPS
        for group in GROUPS
    ]
    missing = [key for key in expected if key not in by_key]
    if missing:
        raise RuntimeError(f"Manifest missing datasets: {missing}")

    print("=" * 100)
    print("OOD UNKNOWN-INTERFACE EXTRAPOLATION")
    print("=" * 100)
    print("Manifest   :", manifest)
    print("Checkpoint :", checkpoint)
    print("Device     :", args.device)
    print("Mode       :", "RUN" if args.run else "DRY RUN")
    print()

    plan = []
    for decomp, group in expected:
        dataset = Path(by_key[(decomp, group)]["path"])
        audit = audit_dataset(dataset)

        print(
            f"{decomp:<14} {group:<14} | "
            f"channels={audit['n_channels']} | "
            f"subdomains={audit['n_subdomains']} | "
            f"interfaces={audit['n_interfaces']} | "
            f"samples={audit['n_samples']}"
        )

        cmd = child_command(
            dataset,
            group,
            decomp,
            checkpoint,
            args.device,
            args.resume,
            args.limit,
            args.run,
        )
        plan.append((decomp, group, dataset, cmd))

    if not args.run:
        print("\nDRY RUN ONLY. Add --run to execute.")
        return

    results_root.mkdir(parents=True, exist_ok=True)
    long_parts = []

    for i, (decomp, group, dataset, cmd) in enumerate(plan, start=1):
        print("\n" + "#" * 100)
        print(f"{i}/10  {decomp} / {group}")
        print("#" * 100)
        print(" ".join(cmd), flush=True)

        subprocess.run(cmd, cwd=ROOT, check=True)
        long_parts.append(load_child_metrics(group, decomp))

    long_df = pd.concat(long_parts, ignore_index=True)
    channel_df = make_per_channel(long_df)

    long_df.to_csv(results_root / "metrics_long.csv", index=False)
    channel_df.to_csv(
        results_root / "metrics_per_channel.csv",
        index=False,
    )

    table_ar1 = make_table(channel_df, "ar1")
    table_cp = make_table(channel_df, "controlpoints")

    table_ar1.to_csv(results_root / "table_ar1.csv", index=False)
    table_cp.to_csv(
        results_root / "table_controlpoints.csv",
        index=False,
    )

    summary = pd.concat(
        [
            table_ar1.assign(decomposition="ar1"),
            table_cp.assign(decomposition="controlpoints"),
        ],
        ignore_index=True,
    )
    summary.to_csv(
        results_root / "metrics_summary.csv",
        index=False,
    )

    summary_json = {
        "protocol": "distilled5_unknown_interface_extrapolation_v1",
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "device": args.device,
        "n_ood_channels": 50,
        "n_dataset_decomposition_pairs": 10,
        "main_metric": (
            "whole-channel Global Relative L2 from concatenated "
            "subdomain cell-center predictions/truth"
        ),
        "oracle_gap_recovered": (
            "100 * (L_zero - L_physics) / (L_zero - L_known)"
        ),
        "outputs": {
            "table_ar1": str(results_root / "table_ar1.csv"),
            "table_controlpoints": str(
                results_root / "table_controlpoints.csv"
            ),
            "metrics_per_channel": str(
                results_root / "metrics_per_channel.csv"
            ),
            "metrics_summary": str(
                results_root / "metrics_summary.csv"
            ),
        },
    }
    (results_root / "summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n"
    )

    print("\n" + "=" * 100)
    print("COMPLETED")
    print("=" * 100)
    print("AR1 table :", results_root / "table_ar1.csv")
    print("CP table  :", results_root / "table_controlpoints.csv")
    print("Summary   :", results_root / "metrics_summary.csv")


if __name__ == "__main__":
    main()
