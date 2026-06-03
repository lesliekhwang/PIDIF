import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from deeponet_fluent_dataset import DeepONetCellDataset
from fluent_deeponet import predict_cell_sample

def local_query_to_physical(query_local, metadata):
    """
    Convert local subdomain query coordinates to physical coordinates.

    query_local:
        (N, 2), columns [x_local, y_local]

    metadata:
        one entry from data["metadata"]
    """
    query_local = np.asarray(query_local, dtype=np.float32)

    # x_local is normalized to [0, 1] across the subdomain width; y_local is
    # scaled by the reference length about the channel centerline.
    x_left = float(metadata["x_left_mm"])
    x_right = float(metadata["x_right_mm"])
    y_center = float(metadata["y_center_mm"])
    ref_length = float(metadata["reference_length"])

    x_phys = x_left + query_local[:, 0] * (x_right - x_left)
    y_phys = y_center + query_local[:, 1] * ref_length

    return x_phys, y_phys


def interpolate_to_plot_grid(
    x,
    y,
    z,
    n_x_plot=None,
    n_y_plot=100,
    method="linear",
):
    """
    Interpolate scattered cell-center values to a regular plot grid.

    This is only for visualization, not for training.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    if n_x_plot is None:
        ar_plot = (x_max - x_min) / max(y_max - y_min, 1.0e-12)
        n_x_plot = max(100, int(round(n_y_plot * ar_plot)))

    xi = np.linspace(x_min, x_max, int(n_x_plot))
    yi = np.linspace(y_min, y_max, int(n_y_plot))
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(
        (x, y),
        z,
        (Xi, Yi),
        method=method,
    )

    # Linear interpolation can produce NaNs near boundaries.
    # Fill those with nearest-neighbor interpolation for plotting.
    if np.any(np.isnan(Zi)):
        Zi_nearest = griddata(
            (x, y),
            z,
            (Xi, Yi),
            method="nearest",
        )
        Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)

    return Xi, Yi, Zi


def plot_prediction_imshow_from_points(
    x,
    y,
    pred,
    truth,
    field_name,
    field_idx,
    figsize=(24, 3.5),
    n_x_plot=None,
    n_y_plot=100,
    output_dir=None,
    filename_prefix="prediction",
    show=True,
):
    """
    Plot predicted field, CFD/truth field, and absolute error using griddata + imshow.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    pred = np.asarray(pred)
    truth = np.asarray(truth)

    _, _, z_pred = interpolate_to_plot_grid(
        x,
        y,
        pred[:, field_idx],
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        method="linear",
    )

    _, _, z_truth = interpolate_to_plot_grid(
        x,
        y,
        truth[:, field_idx],
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        method="linear",
    )

    z_error = np.abs(z_truth - z_pred)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    vmin = float(np.nanmin(truth[:, field_idx]))
    vmax = float(np.nanmax(truth[:, field_idx]))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        constrained_layout=True,
    )

    im1 = axes[0].imshow(
        z_truth,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="jet",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"CFD truth {field_name}", fontsize=16)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        z_pred,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="jet",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Prediction {field_name}", fontsize=16)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        z_error,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="jet",
        aspect="auto",
    )
    axes[2].set_title(f"{field_name} absolute error", fontsize=16)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im3, ax=axes[2])

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{filename_prefix}_{field_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes

@torch.no_grad()
def collect_predictions_for_data(
    model,
    data,
    device,
    y_normalizer,
    local_aspect_mean,
    local_aspect_std,
    sample_indices=None,
):
    """
    Predict selected samples and concatenate their physical coordinates,
    predictions, and truth values.
    """
    if sample_indices is None:
        sample_indices = range(len(data["samples"]))

    xs = []
    ys = []
    preds = []
    truths = []
    sample_ids = []

    for sid in sample_indices:
        sample = data["samples"][sid]
        metadata = data["metadata"][sid]

        pred_phys = predict_cell_sample(
            model=model,
            sample=sample,
            device=device,
            y_normalizer=y_normalizer,
            local_aspect_mean=local_aspect_mean,
            local_aspect_std=local_aspect_std,
            branch_channel_names=data["branch_channel_names"],
        )
        query_local = np.asarray(sample["query"], dtype=np.float32)
        truth_phys = np.asarray(sample["target"], dtype=np.float32)

        x_phys, y_phys = local_query_to_physical(query_local, metadata)

        xs.append(x_phys)
        ys.append(y_phys)
        preds.append(pred_phys)
        truths.append(truth_phys)
        sample_ids.append(np.full(len(x_phys), sid, dtype=np.int64))

    return {
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "pred": np.concatenate(preds, axis=0),
        "truth": np.concatenate(truths, axis=0),
        "sample_id": np.concatenate(sample_ids, axis=0),
    }


def select_samples_by_ar(data, ar, realization_id=0):
    selected = [
        i for i, m in enumerate(data["metadata"])
        if int(round(float(m["aspect_ratio"]))) == int(ar)
        and int(m.get("realization_id", 0)) == int(realization_id)
    ]

    selected = sorted(
        selected,
        key=lambda i: int(data["metadata"][i]["subdomain_id"]),
    )

    if not selected:
        raise ValueError(
            f"No samples found for AR={ar}, realization_id={realization_id}"
        )

    sub_ids = [
        int(data["metadata"][i]["subdomain_id"])
        for i in selected
    ]

    expected = list(range(len(selected)))
    if sub_ids != expected:
        raise ValueError(
            f"Selected subdomains are not consecutive 0..S-1. "
            f"Found subdomain IDs: {sub_ids}"
        )

    return selected