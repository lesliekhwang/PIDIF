#!/usr/bin/env python3
"""Plot formal progressive-distillation accuracy/runtime comparisons.

This script reads existing validation summary_metrics.csv files only.
It does not run inference, access the test set, or modify checkpoints.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path("/home/nuoxu9/PIDIF")

RESULTS_ROOT = (
    REPO_ROOT
    / "results"
    / "evaluate_diffusion_generation"
)

BASE_SUMMARY = (
    RESULTS_ROOT
    / "base_val_progressive_nested20_seed0"
    / "summary_metrics.csv"
)

DISTILLED10_SUMMARY = (
    RESULTS_ROOT
    / "distilled_nested10_val_seed0"
    / "summary_metrics.csv"
)

DISTILLED5_SUMMARY = (
    RESULTS_ROOT
    / "distilled_nested5_val_seed0"
    / "summary_metrics.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "distill_progressive"
)

FIELD_ORDER = ("pressure", "u", "v")

FIELD_LABELS = {
    "pressure": "Pressure",
    "u": "u",
    "v": "v",
}

FIELD_UNITS = {
    "pressure": "Pa",
    "u": "m/s",
    "v": "m/s",
}

MODEL_ORDER = (
    "Base Nested20",
    "Base Nested10",
    "Base Nested5",
    "Distilled Nested10",
    "Distilled Nested5",
)


@dataclass(frozen=True)
class EvaluationPoint:
    name: str
    family: str
    nfe: int
    balanced_rmse: dict[str, float]
    runtime_per_subdomain_s: float


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def find_column(
    fieldnames: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    normalized = {
        normalize_name(name): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Summary CSV is empty: {path}")

    return rows


def row_nfe(
    row: dict[str, str],
    fieldnames: Iterable[str],
) -> int | None:
    column = find_column(
        fieldnames,
        (
            "sampling_steps",
            "sampling_step",
            "steps",
            "nfe",
            "num_steps",
        ),
    )
    if column is not None:
        value = parse_int(row.get(column))
        if value is not None:
            return value

    for value in row.values():
        text = str(value).strip().upper()
        if text.startswith("DDIM-"):
            suffix = text.split("DDIM-", 1)[1]
            try:
                return int(suffix)
            except ValueError:
                pass

    return None


def row_field(
    row: dict[str, str],
    fieldnames: Iterable[str],
) -> str | None:
    column = find_column(
        fieldnames,
        (
            "field",
            "field_name",
            "variable",
            "channel",
            "output",
            "quantity",
        ),
    )

    if column is not None:
        value = normalize_name(row.get(column, ""))
        aliases = {
            "pressure": "pressure",
            "p": "pressure",
            "u": "u",
            "v": "v",
        }
        if value in aliases:
            return aliases[value]

    for value in row.values():
        normalized = normalize_name(str(value))
        if normalized in {"pressure", "p"}:
            return "pressure"
        if normalized == "u":
            return "u"
        if normalized == "v":
            return "v"

    return None


def row_balanced_rmse(
    row: dict[str, str],
    fieldnames: Iterable[str],
) -> float | None:
    column = find_column(
        fieldnames,
        (
            "subdomain_balanced_rmse",
            "balanced_rmse",
            "balanced_rmse_value",
            "rmse_balanced",
            "balanced_root_mean_squared_error",
        ),
    )
    if column is not None:
        return parse_float(row.get(column))

    return None


def find_runtime(
    rows: list[dict[str, str]],
    nfe: int,
) -> float:
    fieldnames = rows[0].keys()

    runtime_column = find_column(
        fieldnames,
        (
            "sampling_mean_seconds_per_subdomain",
            "mean_runtime_per_subdomain_s",
            "runtime_per_subdomain_s",
            "mean_subdomain_runtime_s",
            "mean_runtime_s",
            "runtime_mean_s",
        ),
    )

    if runtime_column is None:
        raise KeyError(
            "Could not find runtime-per-subdomain column in summary CSV. "
            f"Available columns: {list(fieldnames)}"
        )

    candidates = []

    for row in rows:
        parsed_nfe = row_nfe(row, fieldnames)
        if parsed_nfe == nfe:
            value = parse_float(row.get(runtime_column))
            if value is not None:
                candidates.append(value)

    if not candidates:
        raise ValueError(
            f"No runtime found for NFE={nfe}. "
            f"Runtime column: {runtime_column}"
        )

    return float(candidates[0])


def load_point(
    *,
    name: str,
    family: str,
    path: Path,
    nfe: int,
) -> EvaluationPoint:
    rows = read_summary_rows(path)
    fieldnames = rows[0].keys()

    rmse = {}

    for row in rows:
        parsed_nfe = row_nfe(row, fieldnames)
        parsed_field = row_field(row, fieldnames)

        if parsed_nfe != nfe:
            continue

        if parsed_field not in FIELD_ORDER:
            continue

        value = row_balanced_rmse(row, fieldnames)
        if value is not None:
            rmse[parsed_field] = value

    missing = [
        field
        for field in FIELD_ORDER
        if field not in rmse
    ]

    if missing:
        raise ValueError(
            f"Missing balanced RMSE values for {missing} "
            f"in {path} at NFE={nfe}.\n"
            f"Available columns: {list(fieldnames)}"
        )

    runtime = find_runtime(rows, nfe)

    return EvaluationPoint(
        name=name,
        family=family,
        nfe=nfe,
        balanced_rmse=rmse,
        runtime_per_subdomain_s=runtime,
    )


def load_all_points() -> list[EvaluationPoint]:
    return [
        load_point(
            name="Base Nested20",
            family="Base",
            path=BASE_SUMMARY,
            nfe=20,
        ),
        load_point(
            name="Base Nested10",
            family="Base",
            path=BASE_SUMMARY,
            nfe=10,
        ),
        load_point(
            name="Base Nested5",
            family="Base",
            path=BASE_SUMMARY,
            nfe=5,
        ),
        load_point(
            name="Distilled Nested10",
            family="Distilled",
            path=DISTILLED10_SUMMARY,
            nfe=10,
        ),
        load_point(
            name="Distilled Nested5",
            family="Distilled",
            path=DISTILLED5_SUMMARY,
            nfe=5,
        ),
    ]


def get_point(
    points: list[EvaluationPoint],
    name: str,
) -> EvaluationPoint:
    for point in points:
        if point.name == name:
            return point
    raise KeyError(name)


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )
    ax.tick_params(axis="both", labelsize=11)


def plot_accuracy_vs_nfe(
    points: list[EvaluationPoint],
    output_dir: Path,
) -> None:
    base = [
        get_point(points, "Base Nested5"),
        get_point(points, "Base Nested10"),
        get_point(points, "Base Nested20"),
    ]

    distilled = [
        get_point(points, "Distilled Nested5"),
        get_point(points, "Distilled Nested10"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.7),
        constrained_layout=True,
    )

    for ax, field in zip(axes, FIELD_ORDER):
        base_x = [point.nfe for point in base]
        base_y = [point.balanced_rmse[field] for point in base]

        distilled_x = [point.nfe for point in distilled]
        distilled_y = [
            point.balanced_rmse[field]
            for point in distilled
        ]

        ax.plot(
            base_x,
            base_y,
            marker="o",
            linewidth=2.0,
            markersize=7,
            label="Base nested DDIM",
        )

        ax.plot(
            distilled_x,
            distilled_y,
            marker="s",
            linestyle="--",
            linewidth=2.0,
            markersize=7,
            label="Progressively distilled",
        )

        ax.set_title(
            FIELD_LABELS[field],
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("NFE", fontsize=12)
        ax.set_ylabel(
            f"Balanced RMSE ({FIELD_UNITS[field]})",
            fontsize=12,
        )
        ax.set_xticks([5, 10, 20])
        setup_axis(ax)

    axes[0].legend(
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        "Progressive Distillation: Accuracy vs NFE",
        fontsize=17,
        fontweight="bold",
    )

    png_path = output_dir / "distillation_accuracy_vs_nfe.png"
    pdf_path = output_dir / "distillation_accuracy_vs_nfe.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def plot_runtime_vs_nfe(
    points: list[EvaluationPoint],
    output_dir: Path,
) -> None:
    base = [
        get_point(points, "Base Nested5"),
        get_point(points, "Base Nested10"),
        get_point(points, "Base Nested20"),
    ]

    distilled = [
        get_point(points, "Distilled Nested5"),
        get_point(points, "Distilled Nested10"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(
        [point.nfe for point in base],
        [point.runtime_per_subdomain_s for point in base],
        marker="o",
        linewidth=2.0,
        markersize=8,
        label="Base nested DDIM",
    )

    ax.plot(
        [point.nfe for point in distilled],
        [point.runtime_per_subdomain_s for point in distilled],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=8,
        label="Progressively distilled",
    )

    ax.set_title(
        "Inference Runtime vs NFE",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("NFE", fontsize=13)
    ax.set_ylabel("Runtime per subdomain (s)", fontsize=13)
    ax.set_xticks([5, 10, 20])
    setup_axis(ax)
    ax.legend(fontsize=11, frameon=True)

    fig.tight_layout()

    png_path = output_dir / "distillation_runtime_vs_nfe.png"
    pdf_path = output_dir / "distillation_runtime_vs_nfe.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def plot_accuracy_runtime_tradeoff(
    points: list[EvaluationPoint],
    output_dir: Path,
) -> None:
    base = [
        get_point(points, "Base Nested20"),
        get_point(points, "Base Nested10"),
        get_point(points, "Base Nested5"),
    ]

    distilled = [
        get_point(points, "Distilled Nested10"),
        get_point(points, "Distilled Nested5"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    ax.plot(
        [point.runtime_per_subdomain_s for point in base],
        [point.balanced_rmse["pressure"] for point in base],
        marker="o",
        linewidth=2.0,
        markersize=8,
        label="Base nested DDIM",
    )

    ax.plot(
        [
            point.runtime_per_subdomain_s
            for point in distilled
        ],
        [
            point.balanced_rmse["pressure"]
            for point in distilled
        ],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=8,
        label="Progressively distilled",
    )

    annotation_offsets = {
        "Base Nested20": (-55, 12),
        "Base Nested10": (8, 12),
        "Base Nested5": (8, 10),
        "Distilled Nested10": (8, 10),
        "Distilled Nested5": (-4, 14),
    }

    for point in points:
        dx, dy = annotation_offsets[point.name]
        ax.annotate(
            point.name,
            xy=(
                point.runtime_per_subdomain_s,
                point.balanced_rmse["pressure"],
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
        )

    ax.annotate(
        "better",
        xy=(0.008, 24.0),
        xytext=(0.013, 27.0),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
        },
        fontsize=11,
        fontstyle="italic",
    )

    ax.set_title(
        "Pressure Accuracy–Runtime Tradeoff",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Runtime per subdomain (s)", fontsize=13)
    ax.set_ylabel("Pressure balanced RMSE (Pa)", fontsize=13)

    setup_axis(ax)
    ax.legend(fontsize=11, frameon=True)

    fig.tight_layout()

    png_path = (
        output_dir
        / "distillation_accuracy_runtime_tradeoff.png"
    )
    pdf_path = (
        output_dir
        / "distillation_accuracy_runtime_tradeoff.pdf"
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def print_summary(points: list[EvaluationPoint]) -> None:
    print("\nLoaded formal validation results")
    print("=" * 100)

    header = (
        f"{'Model':<22}"
        f"{'NFE':>6}"
        f"{'Pressure RMSE':>18}"
        f"{'u RMSE':>16}"
        f"{'v RMSE':>16}"
        f"{'Runtime/subdomain':>22}"
    )
    print(header)
    print("-" * len(header))

    for name in MODEL_ORDER:
        point = get_point(points, name)

        print(
            f"{point.name:<22}"
            f"{point.nfe:>6d}"
            f"{point.balanced_rmse['pressure']:>18.6f}"
            f"{point.balanced_rmse['u']:>16.8f}"
            f"{point.balanced_rmse['v']:>16.8f}"
            f"{point.runtime_per_subdomain_s:>22.6f}"
        )

    base5 = get_point(points, "Base Nested5")
    distilled5 = get_point(points, "Distilled Nested5")
    base20 = get_point(points, "Base Nested20")

    pressure_improvement = (
        1.0
        - distilled5.balanced_rmse["pressure"]
        / base5.balanced_rmse["pressure"]
    ) * 100.0

    runtime_ratio_vs_base5 = (
        distilled5.runtime_per_subdomain_s
        / base5.runtime_per_subdomain_s
    )

    speedup_vs_base20 = (
        base20.runtime_per_subdomain_s
        / distilled5.runtime_per_subdomain_s
    )

    print("\nKey comparisons")
    print("=" * 100)
    print(
        "Distilled Nested5 pressure improvement vs Base Nested5: "
        f"{pressure_improvement:.2f}%"
    )
    print(
        "Distilled Nested5 runtime / Base Nested5 runtime: "
        f"{runtime_ratio_vs_base5:.3f}x"
    )
    print(
        "Distilled Nested5 speedup vs Base Nested20: "
        f"{speedup_vs_base20:.2f}x"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Input summaries")
    print("  Base       :", BASE_SUMMARY)
    print("  Distilled10:", DISTILLED10_SUMMARY)
    print("  Distilled5 :", DISTILLED5_SUMMARY)

    points = load_all_points()

    print_summary(points)

    plot_accuracy_vs_nfe(points, OUTPUT_DIR)
    plot_runtime_vs_nfe(points, OUTPUT_DIR)
    plot_accuracy_runtime_tradeoff(points, OUTPUT_DIR)

    print("\nDone.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()