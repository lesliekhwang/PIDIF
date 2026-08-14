from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.evaluate_diffusion_generation import (
    load_checkpoint,
    require_checkpoint_field,
)


DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluate_unknown_interface_solution"
)

DEFAULT_KNOWN_RUN = (
    DEFAULT_RESULTS_ROOT
    / "channel08_real0_known_interface"
)

DEFAULT_INITIAL_RUN = (
    DEFAULT_RESULTS_ROOT
    / "channel08_real0_zero_interface"
)

DEFAULT_FINAL_RUN = (
    DEFAULT_RESULTS_ROOT
    / "channel08_real0_physics_optimized_interface"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_RESULTS_ROOT
    / "channel08_real0_abc_wholefield_plots"
)


FIELDS = (
    "pressure",
    "u",
    "v",
)

FIELD_TITLES = {
    "pressure": "Pressure",
    "u": "u",
    "v": "v",
}

FIELD_UNITS = {
    "pressure": "Pa",
    "u": "m/s",
    "v": "m/s",
}


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def read_metadata(
    dataset_path: Path,
    sample_index: int,
) -> dict:
    with h5py.File(
        dataset_path,
        "r",
    ) as handle:
        group = handle["metadata"]

        return {
            key: decode_scalar(
                group[key][sample_index]
            )
            for key in group.keys()
        }


def load_run(
    run_dir: Path,
) -> dict:
    run_dir = (
        run_dir
        .expanduser()
        .resolve()
    )

    config_path = (
        run_dir
        / "config.json"
    )

    summary_path = (
        run_dir
        / "summary.json"
    )

    prediction_path = (
        run_dir
        / "predictions.h5"
    )

    for path in (
        config_path,
        summary_path,
        prediction_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing run artifact: {path}"
            )

    config = read_json(
        config_path
    )

    summary = read_json(
        summary_path
    )

    dataset_path = Path(
        config["dataset"]
    ).expanduser().resolve()

    samples = []

    with h5py.File(
        prediction_path,
        "r",
    ) as handle:
        sample_group = (
            handle["samples"]
        )

        for key in sample_group.keys():
            group = sample_group[key]

            sample_index = int(
                group.attrs.get(
                    "sample_index",
                    int(key),
                )
            )

            subdomain_id = int(
                group.attrs[
                    "subdomain_id"
                ]
            )

            samples.append(
                {
                    "sample_index": (
                        sample_index
                    ),
                    "subdomain_id": (
                        subdomain_id
                    ),
                    "query": (
                        group["query"][:]
                        .astype(np.float32)
                    ),
                    "prediction": (
                        group["prediction"][:]
                        .astype(np.float32)
                    ),
                    "target": (
                        group["target"][:]
                        .astype(np.float32)
                    ),
                }
            )

    samples.sort(
        key=lambda x: x[
            "subdomain_id"
        ]
    )

    if len(samples) != 10:
        raise RuntimeError(
            f"Expected 10 subdomains, "
            f"found {len(samples)} "
            f"in {run_dir}"
        )

    ids = [
        sample["subdomain_id"]
        for sample in samples
    ]

    if ids != list(range(10)):
        raise RuntimeError(
            f"Unexpected subdomain IDs: {ids}"
        )

    return {
        "run_dir": run_dir,
        "config": config,
        "summary": summary,
        "dataset_path": dataset_path,
        "samples": samples,
    }


def validate_alignment(
    reference: dict,
    other: dict,
    name: str,
) -> None:
    if (
        reference["dataset_path"].resolve()
        != other["dataset_path"].resolve()
    ):
        raise RuntimeError(
            f"{name}: dataset mismatch"
        )

    for ref, current in zip(
        reference["samples"],
        other["samples"],
    ):
        if (
            ref["sample_index"]
            != current["sample_index"]
        ):
            raise RuntimeError(
                f"{name}: sample index mismatch"
            )

        if (
            ref["subdomain_id"]
            != current["subdomain_id"]
        ):
            raise RuntimeError(
                f"{name}: subdomain ID mismatch"
            )

        if not np.array_equal(
            ref["query"],
            current["query"],
        ):
            raise RuntimeError(
                f"{name}: query mismatch for "
                f"sample {ref['sample_index']}"
            )

        if not np.array_equal(
            ref["target"],
            current["target"],
        ):
            raise RuntimeError(
                f"{name}: CFD target mismatch for "
                f"sample {ref['sample_index']}"
            )


def physical_query_coordinates(
    *,
    dataset_path: Path,
    samples: list[dict],
) -> tuple[
    dict[int, np.ndarray],
    list[float],
    dict[int, dict],
]:
    coordinates = {}
    metadata_by_subdomain = {}
    interface_x = []

    for sample in samples:
        sample_index = int(
            sample["sample_index"]
        )

        subdomain_id = int(
            sample["subdomain_id"]
        )

        metadata = read_metadata(
            dataset_path,
            sample_index,
        )

        metadata_by_subdomain[
            subdomain_id
        ] = metadata

        x_left_mm = float(
            metadata["x_left_mm"]
        )

        x_right_mm = float(
            metadata["x_right_mm"]
        )

        y_origin_mm = float(
            metadata[
                "y_local_origin_mm"
            ]
        )

        y_scale_mm = float(
            metadata[
                "y_local_scale_mm"
            ]
        )

        query = sample["query"]

        x_local = query[:, 0]
        y_local = query[:, 1]

        x_mm = (
            x_left_mm
            + x_local
            * (
                x_right_mm
                - x_left_mm
            )
        )

        y_mm = (
            y_origin_mm
            + y_local
            * y_scale_mm
        )

        coordinates[
            subdomain_id
        ] = np.column_stack(
            (
                x_mm,
                y_mm,
            )
        ).astype(np.float32)

        if subdomain_id < 9:
            interface_x.append(
                x_right_mm
            )

    if len(interface_x) != 9:
        raise RuntimeError(
            f"Expected 9 internal interfaces, "
            f"found {len(interface_x)}"
        )

    return (
        coordinates,
        interface_x,
        metadata_by_subdomain,
    )


def collapse_duplicate_x(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    order = np.argsort(x)

    x = np.asarray(
        x[order],
        dtype=np.float64,
    )

    y = np.asarray(
        y[order],
        dtype=np.float64,
    )

    unique_x, inverse = (
        np.unique(
            x,
            return_inverse=True,
        )
    )

    y_sum = np.zeros(
        len(unique_x),
        dtype=np.float64,
    )

    counts = np.zeros(
        len(unique_x),
        dtype=np.float64,
    )

    np.add.at(
        y_sum,
        inverse,
        y,
    )

    np.add.at(
        counts,
        inverse,
        1.0,
    )

    unique_y = (
        y_sum
        / counts
    )

    return (
        unique_x,
        unique_y,
    )


def reconstruct_channel_walls(
    *,
    dataset_path: Path,
    samples: list[dict],
    metadata_by_subdomain: dict[int, dict],
    branch_channel_names: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ix = branch_channel_names.index(
        "x_local"
    )

    iy = branch_channel_names.index(
        "y_local"
    )

    wall_idx = (
        branch_channel_names.index(
            "wall_mask"
        )
    )

    bottom_x = []
    bottom_y = []

    top_x = []
    top_y = []

    with h5py.File(
        dataset_path,
        "r",
    ) as handle:
        for sample in samples:
            sample_index = int(
                sample["sample_index"]
            )

            subdomain_id = int(
                sample["subdomain_id"]
            )

            metadata = (
                metadata_by_subdomain[
                    subdomain_id
                ]
            )

            branch = (
                handle["samples"][
                    str(sample_index)
                ]["branch"][:]
                .astype(np.float32)
            )

            wall_mask = (
                branch[:, wall_idx]
                > 0.5
            )

            wall = branch[
                wall_mask
            ]

            if len(wall) == 0:
                raise RuntimeError(
                    f"No wall points for "
                    f"sample {sample_index}"
                )

            x_left_mm = float(
                metadata[
                    "x_left_mm"
                ]
            )

            x_right_mm = float(
                metadata[
                    "x_right_mm"
                ]
            )

            y_origin_mm = float(
                metadata[
                    "y_local_origin_mm"
                ]
            )

            y_scale_mm = float(
                metadata[
                    "y_local_scale_mm"
                ]
            )

            x_local = wall[:, ix]
            y_local = wall[:, iy]

            x_mm = (
                x_left_mm
                + x_local
                * (
                    x_right_mm
                    - x_left_mm
                )
            )

            y_mm = (
                y_origin_mm
                + y_local
                * y_scale_mm
            )

            bottom_mask = (
                y_local <= 0.5
            )

            top_mask = (
                y_local > 0.5
            )

            bottom_x.append(
                x_mm[bottom_mask]
            )

            bottom_y.append(
                y_mm[bottom_mask]
            )

            top_x.append(
                x_mm[top_mask]
            )

            top_y.append(
                y_mm[top_mask]
            )

    bottom_x = np.concatenate(
        bottom_x
    )

    bottom_y = np.concatenate(
        bottom_y
    )

    top_x = np.concatenate(
        top_x
    )

    top_y = np.concatenate(
        top_y
    )

    bottom_x, bottom_y = (
        collapse_duplicate_x(
            bottom_x,
            bottom_y,
        )
    )

    top_x, top_y = (
        collapse_duplicate_x(
            top_x,
            top_y,
        )
    )

    return (
        bottom_x,
        bottom_y,
        top_x,
        top_y,
    )


def concatenate_coordinates(
    coordinates: dict[
        int,
        np.ndarray,
    ],
) -> np.ndarray:
    return np.concatenate(
        [
            coordinates[
                subdomain_id
            ]
            for subdomain_id
            in range(10)
        ],
        axis=0,
    )


def concatenate_values(
    *,
    run: dict,
    source: str,
) -> np.ndarray:
    values = []

    for sample in run[
        "samples"
    ]:
        if source == "target":
            array = sample[
                "target"
            ]

        elif source == "prediction":
            array = sample[
                "prediction"
            ]

        elif source == "absolute_error":
            array = np.abs(
                sample["prediction"]
                - sample["target"]
            )

        else:
            raise ValueError(
                f"Unknown source: {source}"
            )

        values.append(
            array
        )

    return np.concatenate(
        values,
        axis=0,
    ).astype(np.float32)


def make_grid(
    *,
    coordinates: np.ndarray,
    bottom_x: np.ndarray,
    bottom_y: np.ndarray,
    top_x: np.ndarray,
    top_y: np.ndarray,
    n_x: int,
    n_y: int,
):
    x_min = float(
        min(
            coordinates[:, 0].min(),
            bottom_x.min(),
            top_x.min(),
        )
    )

    x_max = float(
        max(
            coordinates[:, 0].max(),
            bottom_x.max(),
            top_x.max(),
        )
    )

    y_min = float(
        bottom_y.min()
    )

    y_max = float(
        top_y.max()
    )

    x_grid = np.linspace(
        x_min,
        x_max,
        n_x,
    )

    y_grid = np.linspace(
        y_min,
        y_max,
        n_y,
    )

    grid_x, grid_y = (
        np.meshgrid(
            x_grid,
            y_grid,
        )
    )

    bottom_curve = np.interp(
        x_grid,
        bottom_x,
        bottom_y,
    )

    top_curve = np.interp(
        x_grid,
        top_x,
        top_y,
    )

    geometry_mask = (
        (grid_y >= bottom_curve[None, :])
        & (
            grid_y
            <= top_curve[None, :]
        )
    )

    return {
        "x": x_grid,
        "y": y_grid,
        "X": grid_x,
        "Y": grid_y,
        "mask": geometry_mask,
        "bottom": bottom_curve,
        "top": top_curve,
        "extent": (
            x_min,
            x_max,
            y_min,
            y_max,
        ),
    }


def interpolate_fields(
    *,
    coordinates: np.ndarray,
    values: np.ndarray,
    grid: dict,
) -> np.ndarray:
    if (
        values.ndim != 2
        or values.shape[1] != 3
    ):
        raise ValueError(
            f"Expected values shape (N,3), "
            f"got {values.shape}"
        )

    if len(coordinates) != len(values):
        raise ValueError(
            "Coordinate/value length mismatch"
        )

    print(
        f"  griddata linear: "
        f"{len(values):,} points -> "
        f"{grid['X'].shape}"
    )

    linear = griddata(
        coordinates,
        values,
        (
            grid["X"],
            grid["Y"],
        ),
        method="linear",
    )

    missing = (
        grid["mask"]
        & ~np.all(
            np.isfinite(linear),
            axis=2,
        )
    )

    if np.any(missing):
        print(
            f"  nearest fill: "
            f"{int(missing.sum()):,} "
            "interior grid cells"
        )

        nearest = griddata(
            coordinates,
            values,
            (
                grid["X"],
                grid["Y"],
            ),
            method="nearest",
        )

        linear[
            missing
        ] = nearest[
            missing
        ]

    linear[
        ~grid["mask"]
    ] = np.nan

    return linear.astype(
        np.float32
    )


def field_truth_range(
    truth_grid: np.ndarray,
    field_index: int,
) -> tuple[float, float]:
    values = truth_grid[
        :,
        :,
        field_index,
    ]

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        raise RuntimeError(
            "No finite truth values"
        )

    vmin = float(
        np.min(finite)
    )

    vmax = float(
        np.max(finite)
    )

    if vmax <= vmin:
        vmax = vmin + 1.0

    return (
        vmin,
        vmax,
    )


def error_range(
    *,
    error_values: list[np.ndarray],
    field_index: int,
    quantile: float,
) -> tuple[float, float]:
    values = np.concatenate(
        [
            array[
                :,
                field_index,
            ]
            for array
            in error_values
        ]
    )

    if quantile >= 1.0:
        vmax = float(
            np.max(values)
        )
    else:
        vmax = float(
            np.quantile(
                values,
                quantile,
            )
        )

    if vmax <= 0.0:
        vmax = 1.0

    return (
        0.0,
        vmax,
    )


def decorate_axis(
    *,
    ax,
    grid: dict,
    interface_x: list[float],
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    ax.plot(
        grid["x"],
        grid["bottom"],
        color="black",
        linewidth=1.1,
        zorder=5,
    )

    ax.plot(
        grid["x"],
        grid["top"],
        color="black",
        linewidth=1.1,
        zorder=5,
    )

    for position in interface_x:
        ax.axvline(
            position,
            color="white",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
            zorder=6,
        )

    ax.set_xlim(
        grid["extent"][0],
        grid["extent"][1],
    )

    ax.set_ylim(
        grid["extent"][2],
        grid["extent"][3],
    )

    if show_xlabel:
        ax.set_xlabel(
            "x [mm]"
        )

    if show_ylabel:
        ax.set_ylabel(
            "y [mm]"
        )


def draw_field(
    *,
    ax,
    image: np.ndarray,
    field_index: int,
    grid: dict,
    vmin: float,
    vmax: float,
    interface_x: list[float],
):
    cmap = plt.get_cmap(
        "jet"
    ).copy()

    cmap.set_bad(
        "white"
    )

    artist = ax.imshow(
        np.ma.masked_invalid(
            image[
                :,
                :,
                field_index,
            ]
        ),
        extent=grid["extent"],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    decorate_axis(
        ax=ax,
        grid=grid,
        interface_x=interface_x,
    )

    return artist


def plot_whole_field_comparison(
    *,
    truth_grid: np.ndarray,
    known_grid: np.ndarray,
    initial_grid: np.ndarray,
    final_grid: np.ndarray,
    grid: dict,
    interface_x: list[float],
    output_dir: Path,
    dpi: int,
):
    rows = (
        (
            "CFD Truth",
            truth_grid,
        ),
        (
            "A  Known interface",
            known_grid,
        ),
        (
            "B  Zero interface",
            initial_grid,
        ),
        (
            "C  Physics-optimized",
            final_grid,
        ),
    )

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(
            24,
            10,
        ),
        constrained_layout=True,
    )

    for field_index, field in enumerate(
        FIELDS
    ):
        vmin, vmax = (
            field_truth_range(
                truth_grid,
                field_index,
            )
        )

        artist = None

        for row_index, (
            row_name,
            image,
        ) in enumerate(rows):
            ax = axes[
                row_index,
                field_index,
            ]

            artist = draw_field(
                ax=ax,
                image=image,
                field_index=field_index,
                grid=grid,
                vmin=vmin,
                vmax=vmax,
                interface_x=interface_x,
            )

            if row_index == 0:
                ax.set_title(
                    (
                        f"{FIELD_TITLES[field]} "
                        f"({FIELD_UNITS[field]})"
                    ),
                    fontsize=14,
                    fontweight="bold",
                )

            if field_index == 0:
                ax.text(
                    -0.18,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        colorbar = fig.colorbar(
            artist,
            ax=axes[:, field_index],
            fraction=0.018,
            pad=0.012,
        )

        colorbar.set_label(
            FIELD_UNITS[
                field
            ]
        )

    fig.suptitle(
        (
            "Unknown-interface whole-field comparison "
            "— channel_08, realization 0"
        ),
        fontsize=17,
        fontweight="bold",
    )

    for suffix in (
        "png",
        "pdf",
    ):
        path = (
            output_dir
            / (
                "unknown_interface_"
                "wholefield_comparison."
                f"{suffix}"
            )
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(
            f"Saved: {path}"
        )

    plt.close(fig)


def plot_error_comparison(
    *,
    known_error_grid: np.ndarray,
    initial_error_grid: np.ndarray,
    final_error_grid: np.ndarray,
    known_error_values: np.ndarray,
    initial_error_values: np.ndarray,
    final_error_values: np.ndarray,
    grid: dict,
    interface_x: list[float],
    output_dir: Path,
    error_quantile: float,
    dpi: int,
):
    rows = (
        (
            "A  Known interface",
            known_error_grid,
        ),
        (
            "B  Zero interface",
            initial_error_grid,
        ),
        (
            "C  Physics-optimized",
            final_error_grid,
        ),
    )

    raw_errors = [
        known_error_values,
        initial_error_values,
        final_error_values,
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            24,
            7.5,
        ),
        constrained_layout=True,
    )

    for field_index, field in enumerate(
        FIELDS
    ):
        vmin, vmax = error_range(
            error_values=raw_errors,
            field_index=field_index,
            quantile=error_quantile,
        )

        artist = None

        for row_index, (
            row_name,
            image,
        ) in enumerate(rows):
            ax = axes[
                row_index,
                field_index,
            ]

            artist = draw_field(
                ax=ax,
                image=image,
                field_index=field_index,
                grid=grid,
                vmin=vmin,
                vmax=vmax,
                interface_x=interface_x,
            )

            if row_index == 0:
                ax.set_title(
                    (
                        f"Absolute error — "
                        f"{FIELD_TITLES[field]} "
                        f"({FIELD_UNITS[field]})"
                    ),
                    fontsize=14,
                    fontweight="bold",
                )

            if field_index == 0:
                ax.text(
                    -0.18,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        colorbar = fig.colorbar(
            artist,
            ax=axes[:, field_index],
            fraction=0.018,
            pad=0.012,
        )

        colorbar.set_label(
            FIELD_UNITS[
                field
            ]
        )

    fig.suptitle(
        (
            "Unknown-interface absolute-error comparison "
            "— channel_08, realization 0"
        ),
        fontsize=17,
        fontweight="bold",
    )

    for suffix in (
        "png",
        "pdf",
    ):
        path = (
            output_dir
            / (
                "unknown_interface_"
                "absolute_error_comparison."
                f"{suffix}"
            )
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(
            f"Saved: {path}"
        )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--known-run",
        type=Path,
        default=DEFAULT_KNOWN_RUN,
    )

    parser.add_argument(
        "--initial-run",
        type=Path,
        default=DEFAULT_INITIAL_RUN,
    )

    parser.add_argument(
        "--final-run",
        type=Path,
        default=DEFAULT_FINAL_RUN,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--n-x",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--n-y",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--error-quantile",
        type=float,
        default=1.0,
        help=(
            "Upper error color range quantile. "
            "1.0 uses the true maximum; "
            "0.995 is useful for suppressing "
            "extreme plotting outliers."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if (
        args.n_x < 2
        or args.n_y < 2
    ):
        raise ValueError(
            "Grid dimensions must be >= 2"
        )

    if not (
        0.0
        < args.error_quantile
        <= 1.0
    ):
        raise ValueError(
            "--error-quantile must be "
            "in (0, 1]"
        )

    known = load_run(
        args.known_run
    )

    initial = load_run(
        args.initial_run
    )

    final = load_run(
        args.final_run
    )

    validate_alignment(
        known,
        initial,
        "initial",
    )

    validate_alignment(
        known,
        final,
        "final",
    )

    dataset_path = (
        known["dataset_path"]
    )

    (
        coordinates_by_subdomain,
        interface_x,
        metadata_by_subdomain,
    ) = physical_query_coordinates(
        dataset_path=dataset_path,
        samples=known["samples"],
    )

    coordinates = (
        concatenate_coordinates(
            coordinates_by_subdomain
        )
    )

    checkpoint_path = Path(
        known["config"]["checkpoint"]
    ).expanduser().resolve()

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    branch_channel_names = list(
        require_checkpoint_field(
            checkpoint,
            "branch_channel_names",
        )
    )

    (
        bottom_x,
        bottom_y,
        top_x,
        top_y,
    ) = reconstruct_channel_walls(
        dataset_path=dataset_path,
        samples=known["samples"],
        metadata_by_subdomain=(
            metadata_by_subdomain
        ),
        branch_channel_names=(
            branch_channel_names
        ),
    )

    grid = make_grid(
        coordinates=coordinates,
        bottom_x=bottom_x,
        bottom_y=bottom_y,
        top_x=top_x,
        top_y=top_y,
        n_x=args.n_x,
        n_y=args.n_y,
    )

    truth_values = concatenate_values(
        run=known,
        source="target",
    )

    known_prediction_values = (
        concatenate_values(
            run=known,
            source="prediction",
        )
    )

    initial_prediction_values = (
        concatenate_values(
            run=initial,
            source="prediction",
        )
    )

    final_prediction_values = (
        concatenate_values(
            run=final,
            source="prediction",
        )
    )

    known_error_values = (
        concatenate_values(
            run=known,
            source="absolute_error",
        )
    )

    initial_error_values = (
        concatenate_values(
            run=initial,
            source="absolute_error",
        )
    )

    final_error_values = (
        concatenate_values(
            run=final,
            source="absolute_error",
        )
    )

    print()
    print("=" * 78)
    print(
        "Unknown-interface griddata plotting"
    )
    print("=" * 78)

    print(
        f"Dataset                : "
        f"{dataset_path}"
    )

    print(
        f"Total query points     : "
        f"{len(coordinates):,}"
    )

    print(
        f"Plot grid              : "
        f"{args.n_y} × {args.n_x}"
    )

    print(
        f"Internal interfaces    : "
        f"{len(interface_x)}"
    )

    print(
        f"Interface x [mm]       : "
        f"{interface_x}"
    )

    print(
        f"x range [mm]           : "
        f"{grid['extent'][0]:.6f} to "
        f"{grid['extent'][1]:.6f}"
    )

    print(
        f"y range [mm]           : "
        f"{grid['extent'][2]:.6f} to "
        f"{grid['extent'][3]:.6f}"
    )

    print()
    print("[1/7] CFD truth")
    truth_grid = interpolate_fields(
        coordinates=coordinates,
        values=truth_values,
        grid=grid,
    )

    print("[2/7] Known-interface prediction")
    known_grid = interpolate_fields(
        coordinates=coordinates,
        values=known_prediction_values,
        grid=grid,
    )

    print("[3/7] Zero-interface prediction")
    initial_grid = interpolate_fields(
        coordinates=coordinates,
        values=initial_prediction_values,
        grid=grid,
    )

    print("[4/7] Physics-optimized prediction")
    final_grid = interpolate_fields(
        coordinates=coordinates,
        values=final_prediction_values,
        grid=grid,
    )

    print("[5/7] Known-interface absolute error")
    known_error_grid = interpolate_fields(
        coordinates=coordinates,
        values=known_error_values,
        grid=grid,
    )

    print("[6/7] Zero-interface absolute error")
    initial_error_grid = interpolate_fields(
        coordinates=coordinates,
        values=initial_error_values,
        grid=grid,
    )

    print("[7/7] Physics-optimized absolute error")
    final_error_grid = interpolate_fields(
        coordinates=coordinates,
        values=final_error_values,
        grid=grid,
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_whole_field_comparison(
        truth_grid=truth_grid,
        known_grid=known_grid,
        initial_grid=initial_grid,
        final_grid=final_grid,
        grid=grid,
        interface_x=interface_x,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    plot_error_comparison(
        known_error_grid=(
            known_error_grid
        ),
        initial_error_grid=(
            initial_error_grid
        ),
        final_error_grid=(
            final_error_grid
        ),
        known_error_values=(
            known_error_values
        ),
        initial_error_values=(
            initial_error_values
        ),
        final_error_values=(
            final_error_values
        ),
        grid=grid,
        interface_x=interface_x,
        output_dir=output_dir,
        error_quantile=(
            args.error_quantile
        ),
        dpi=args.dpi,
    )

    print()
    print("=" * 78)
    print("Completed")
    print("=" * 78)
    print(
        f"Output directory       : "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
