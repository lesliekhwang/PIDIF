#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY = Path(
    "/home/nuoxu9/PIDIF/results/evaluate_diffusion_generation/"
    "base_val_ddim_full_seed0/summary_metrics.csv"
)

EXPECTED_STEPS = [5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the formal field-diffusion DDIM sampling study."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to summary_metrics.csv from evaluate_diffusion_generation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <summary parent>/figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf"],
        help="Figure formats to save.",
    )
    return parser.parse_args()


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        "Could not find any of these expected columns: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    steps_col = _resolve_column(
        df,
        ["sampling_steps", "ddim_steps", "steps", "num_sampling_steps"],
    )
    field_col = _resolve_column(df, ["field", "field_name"])
    balanced_rmse_col = _resolve_column(
        df,
        ["balanced_rmse", "subdomain_balanced_rmse", "rmse_balanced"],
    )
    runtime_col = _resolve_column(
        df,
        [
            "sampling_mean_seconds_per_subdomain",
            "mean_sampling_seconds",
            "mean_seconds_per_subdomain",
            "mean_runtime_seconds",
            "mean_runtime_per_subdomain",
        ],
    )

    out = df[[steps_col, field_col, balanced_rmse_col, runtime_col]].copy()
    out.columns = ["steps", "field", "balanced_rmse", "runtime"]
    out["steps"] = out["steps"].astype(int)

    observed_steps = sorted(out["steps"].unique().tolist())
    if observed_steps != EXPECTED_STEPS:
        raise ValueError(
            f"Expected DDIM steps {EXPECTED_STEPS}, found {observed_steps}."
        )

    expected_fields = {"pressure", "u", "v"}
    observed_fields = set(out["field"].astype(str))
    if observed_fields != expected_fields:
        raise ValueError(
            f"Expected fields {sorted(expected_fields)}, found {sorted(observed_fields)}."
        )

    counts = out.groupby(["steps", "field"]).size()
    if not (counts == 1).all():
        raise ValueError("Expected exactly one summary row per DDIM step and field.")

    return out.sort_values(["steps", "field"]).reset_index(drop=True)


def save_figure(fig, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {path}")


def field_series(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return df[df["field"] == field].sort_values("steps")


def plot_accuracy_vs_steps(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    specs = [
        ("pressure", "Pressure balanced RMSE [Pa]"),
        ("u", "u balanced RMSE [m/s]"),
        ("v", "v balanced RMSE [m/s]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for ax, (field, ylabel) in zip(axes, specs):
        d = field_series(df, field)
        ax.plot(d["steps"], d["balanced_rmse"], marker="o")
        ax.set_xscale("log")
        ax.set_xticks(EXPECTED_STEPS)
        ax.set_xticklabels([str(v) for v in EXPECTED_STEPS])
        ax.set_xlabel("DDIM sampling steps (NFE)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Field diffusion sampling accuracy vs DDIM sampling steps", fontsize=14)
    save_figure(fig, output_dir, "sampling_accuracy_vs_steps", formats)
    plt.close(fig)


def plot_runtime_vs_steps(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    runtime = (
        df.groupby("steps", as_index=False)["runtime"]
        .first()
        .sort_values("steps")
    )
    reference = float(runtime.loc[runtime["steps"] == 100, "runtime"].iloc[0])
    runtime["speedup"] = reference / runtime["runtime"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(runtime["steps"], runtime["runtime"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(EXPECTED_STEPS)
    ax.set_xticklabels([str(v) for v in EXPECTED_STEPS])
    ax.set_xlabel("DDIM sampling steps (NFE)")
    ax.set_ylabel("Mean sampling runtime / subdomain [s]")
    ax.set_title("Field diffusion runtime vs DDIM sampling steps")
    ax.grid(True, alpha=0.25)

    for _, row in runtime.iterrows():
        ax.annotate(
            f'{row["speedup"]:.2f}×',
            (row["steps"], row["runtime"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=9,
        )

    save_figure(fig, output_dir, "sampling_runtime_vs_steps", formats)
    plt.close(fig)


def plot_accuracy_runtime_tradeoff(
    df: pd.DataFrame, output_dir: Path, formats: list[str]
) -> None:
    specs = [
        ("pressure", "Pressure balanced RMSE [Pa]"),
        ("u", "u balanced RMSE [m/s]"),
        ("v", "v balanced RMSE [m/s]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for ax, (field, ylabel) in zip(axes, specs):
        d = field_series(df, field)
        ax.plot(d["runtime"], d["balanced_rmse"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Mean sampling runtime / subdomain [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        for _, row in d.iterrows():
            ax.annotate(
                str(int(row["steps"])),
                (row["runtime"], row["balanced_rmse"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    fig.suptitle("Field diffusion accuracy–runtime tradeoff", fontsize=14)
    save_figure(fig, output_dir, "sampling_accuracy_runtime_tradeoff", formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else summary_csv.parent / "figures"
    )

    df = load_summary(summary_csv)

    print(f"Summary : {summary_csv}")
    print(f"Output  : {output_dir}")
    print(f"Steps   : {EXPECTED_STEPS}")
    print("Fields  : pressure, u, v")

    plot_accuracy_vs_steps(df, output_dir, args.formats)
    plot_runtime_vs_steps(df, output_dir, args.formats)
    plot_accuracy_runtime_tradeoff(df, output_dir, args.formats)


if __name__ == "__main__":
    main()
