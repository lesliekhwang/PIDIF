import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_result_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['pred'], data['truth']

def plot_channel(pred_slice, truth_slice, channel_name, save_dir, sample_idx):
    plt.figure(figsize=(15, 4))

    cmap_phys = 'jet'
    cmap_err = 'inferno'

    # Ground Truth
    plt.subplot(1, 3, 1)
    plt.imshow(truth_slice, cmap=cmap_phys, origin='lower')
    plt.title(f"True {channel_name}")
    plt.colorbar()

    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(pred_slice, cmap=cmap_phys, origin='lower')
    plt.title(f"Pred {channel_name}")
    plt.colorbar()

    # Absolute Error
    plt.subplot(1, 3, 3)
    error = np.abs(truth_slice - pred_slice)
    plt.imshow(error, cmap=cmap_err, origin='lower')
    plt.title(f"{channel_name} Abs Error")
    plt.colorbar()

    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{save_dir}/sample_{sample_idx}_{channel_name}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"Saved: {save_name}")

def visualize_all_channels(sample_idx, pred, truth, save_base_path):
    channel_names = ["Pressure", "Temperature", "Vel_X", "Vel_Y"]
    
    print(f"--- Visualizing Sample Index: {sample_idx} ---")
    for i, name in enumerate(channel_names):
        p_slice = pred[sample_idx, :, :, i]
        t_slice = truth[sample_idx, :, :, i]

        plot_channel(p_slice, t_slice, name, save_base_path, sample_idx)

def visualize_combined_results(pred_data, truth_data, save_path, sample_idx):
    channel_names = ["Pressure", "Temperature", "Velocity (X)", "Velocity (Y)"]
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(4):
        gt = truth_data[sample_idx, :, :, i]
        pr = pred_data[sample_idx, :, :, i]
        
        err = np.abs(gt - pr)
        rel = np.abs(gt - pr) / (np.abs(gt) + 1e-8)
        rel_log = np.log10(rel + 1e-8)

        vmin = gt.min()
        vmax = gt.max()

        # print(f"[{channel_names[i]}] GT range: {vmin:.3e} ~ {vmax:.3e}")
        # print(f"[{channel_names[i]}] Pred range: {pr.min():.3e} ~ {pr.max():.3e}")

        row_axes = axes[i]
        
        # --- (1) Ground Truth ---
        im0 = row_axes[0].imshow(
            gt, cmap='jet', origin='lower',
            vmin=vmin, vmax=vmax
        )
        row_axes[0].set_title(f"True {channel_names[i]}", fontsize=12)
        
        # --- (2) Prediction ---
        im1 = row_axes[1].imshow(
            pr, cmap='jet', origin='lower',
            vmin=vmin, vmax=vmax
        )
        row_axes[1].set_title(f"Predicted {channel_names[i]}", fontsize=12)
        
        # --- (3) Absolute Error ---
        im2 = row_axes[2].imshow(
            err, cmap='inferno', origin='lower'
        )
        row_axes[2].set_title(f"Absolute Error ({channel_names[i]})", fontsize=12)
        
        # --- (4) Relative Error (log scale) ---
        im3 = row_axes[3].imshow(
            rel_log, cmap='inferno', origin='lower'
        )
        row_axes[3].set_title(f"Relative Error ({channel_names[i]})", fontsize=12)
        
        # --- Colorbars ---
        for j, im in enumerate([im0, im1, im2, im3]):
            divider = make_axes_locatable(row_axes[j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    os.makedirs(save_path, exist_ok=True)
    full_save_name = f"{save_path}/combined_sample_{sample_idx}.png"
    
    plt.savefig(full_save_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n--- Combined visualization saved to: {full_save_name} ---")

def plot_error_distribution(pred_data, truth_data, save_path):
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    ntest = pred_data.shape[0]
    channel_names = ["Pressure", "Temperature", "Velocity (X)", "Velocity (Y)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    errors = np.zeros((ntest, 4))
    
    for n in range(ntest):
        for c in range(4):
            diff_norm = np.linalg.norm(pred_data[n, ..., c] - truth_data[n, ..., c])
            truth_norm = np.linalg.norm(truth_data[n, ..., c])
            errors[n, c] = diff_norm / (truth_norm + 1e-8)

    total_errors = errors.mean(axis=1)

    # =========================
    # Channel-wise
    # =========================
    print("\n=== Error Summary (per channel) ===")
    for c in range(4):
        max_idx = np.argmax(errors[:, c])
        min_idx = np.argmin(errors[:, c])

        print(f"\n[{channel_names[c]}]")
        print(f"  Worst case  -> Sample {max_idx}, Error: {errors[max_idx, c]:.6e}")
        print(f"  Best case   -> Sample {min_idx}, Error: {errors[min_idx, c]:.6e}")
        print(f"  Average     -> Error: {np.mean(errors[:, c]):.6e}")

    # =========================
    # Total
    # =========================
    total_max_idx = np.argmax(total_errors)
    total_min_idx = np.argmin(total_errors)

    print("\n=== Overall Error Summary ===")
    print(f"  Worst overall -> Sample {total_max_idx}, Error: {total_errors[total_max_idx]:.6e}")
    print(f"  Best overall  -> Sample {total_min_idx}, Error: {total_errors[total_min_idx]:.6e}")
    print(f"  Average total -> Error: {np.mean(total_errors):.6e}")

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    
    for c in range(4):
        plt.plot(
            range(ntest),
            errors[:, c],
            label=f"{channel_names[c]} (Avg: {np.mean(errors[:, c]):.4f})",
            color=colors[c],
            alpha=0.8,
            linewidth=1.5
        )

    # total error
    # plt.plot(
    #     range(ntest),
    #     total_errors,
    #     label="Overall (Mean)",
    #     color="black",
    #     linestyle="--",
    #     linewidth=2
    # )

    plt.yscale('log')
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Relative L2 Error', fontsize=12)
    plt.title('Error Distribution across All Test Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/error_line_graph.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    
    print(f"\n--- Error line graph saved to: {save_file} ---")
    plt.show()

def plot_channel_scatter(pred_data, truth_data, save_path):
    channel_names = ["Pressure", "Temperature", "Vel_X", "Vel_Y"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        y_true = truth_data[..., i].flatten()
        y_pred = pred_data[..., i].flatten()
        
        # indices = np.random.choice(len(y_true), 10000, replace=False)
        
        axes[i].scatter(y_true, y_pred, alpha=0.1, s=1)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[i].set_title(f"{channel_names[i]}\nCorrelation: {np.corrcoef(y_true, y_pred)[0,1]:.4f}")
        axes[i].set_xlabel("True Value")
        axes[i].set_ylabel("Predicted Value")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/channel_scatter_analysis.png", dpi=300)
    print("Channel-wise scatter analysis saved.")
    plt.show()

def compute_global_ranges(pred_list, truth_list, sample_idx):
    stats = {
        "gt": [[], [], [], []],
        "pred": [[], [], [], []],
        "abs": [[], [], [], []],
        "rel": [[], [], [], []],
    }

    for pred, truth in zip(pred_list, truth_list):
        for c in range(4):
            gt = truth[sample_idx, :, :, c]
            pr = pred[sample_idx, :, :, c]

            err = np.abs(gt - pr)
            # rel = (gt - pr)**2 / (np.sum(gt**2) + 1e-8)
            rel = np.abs(gt - pr) / (np.abs(gt) + 1e-8)

            stats["gt"][c].append(gt)
            stats["pred"][c].append(pr)
            stats["abs"][c].append(err)
            stats["rel"][c].append(rel)

    ranges = {}

    for key in stats:
        ranges[key] = []
        for c in range(4):
            data = np.concatenate([x.flatten() for x in stats[key][c]])
            ranges[key].append((data.min(), data.max()))

    return ranges

def visualize_with_fixed_range_log(pred, truth, ranges, save_path, tag, sample_idx=10):
    channel_names = ["Pressure", "Temperature", "Velocity (X)", "Velocity (Y)"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(4):
        gt = truth[sample_idx, :, :, i]
        pr = pred[sample_idx, :, :, i]

        err = np.abs(gt - pr)
        rel = np.abs(gt - pr) / (np.abs(gt) + 1e-8)
        rel_log = np.log10(rel + 1e-8)

        gt_vmin, gt_vmax = ranges["gt"][i]
        abs_vmin, abs_vmax = ranges["abs"][i]
        rel_vmin, rel_vmax = ranges["rel"][i]

        rel_log_vmin = np.log10(rel_vmin + 1e-8)
        rel_log_vmax = np.log10(rel_vmax + 1e-8)

        ims = [
            (gt, "jet", (gt_vmin, gt_vmax)),
            (pr, "jet", (gt_vmin, gt_vmax)),
            (err, "inferno", (abs_vmin, abs_vmax)),
            (rel_log, "inferno", (rel_log_vmin, rel_log_vmax)),
        ]

        titles = [
            f"True {channel_names[i]}",
            f"Predicted {channel_names[i]}",
            f"Absolute Error ({channel_names[i]})",
            f"Relative Error ({channel_names[i]})"
        ]

        for j, (data, cmap, (vmin, vmax)) in enumerate(ims):
            im = axes[i, j].imshow(
                data,
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax
            )

            axes[i, j].set_title(titles[j], fontsize=12)

            divider = make_axes_locatable(axes[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/{tag}_sample_{sample_idx}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_file}")

if __name__ == "__main__":
    # mat_file = 'pred/2d_N1000_ep2000_batch20_s64_mode25_width128_lr0.001_wd1e-4.mat'
    # save_path = 'figures/2d_N1000_ep2000_batch20_s64_mode25_width128_lr0.001_wd1e-4'
    sample_idx = 10

    # pred_data, truth_data = load_result_data(mat_file)

    # visualize_combined_results(pred_data, truth_data, save_path, sample_idx)

    # visualize_all_channels(sample_idx, pred_data, truth_data, save_path)

    # for idx in [10, 20, 30]:
    #     visualize_all_channels(idx, pred_data, truth_data, save_path)

    # plot_error_distribution (pred_data, truth_data, save_path)

    # plot_channel_scatter (pred_data, truth_data, save_path)

    mat_file_1 = "pred/2d_N1000_ep2000_batch20_s64_mode25_width128_lr0.001_wd1e-4.mat"
    mat_file_2 = "pred/2d_N1000_ep2000_batch80_s64_mode25_width128_lr0.001_wd1e-4.mat"
    pred_data1, truth_data = load_result_data(mat_file_1)
    pred_data2, truth_data = load_result_data(mat_file_2)

    ranges = compute_global_ranges(
        [pred_data1, pred_data2],
        [truth_data, truth_data],
        sample_idx=sample_idx
    )

    visualize_with_fixed_range_log(pred_data1, truth_data, ranges, "figures/vis", "batch20")
    visualize_with_fixed_range_log(pred_data2, truth_data, ranges, "figures/vis", "batch80")