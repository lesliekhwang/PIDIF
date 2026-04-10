import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------
# Relative Error
# -----------------------------
def l2(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["train_l2"], label="Train L2")
    plt.plot(epochs, df["test_l2"], label="Test L2")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.title("Overall Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "learning_curve_total.png"), dpi=300)
    plt.show()

def l2_log(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["train_l2"], label="Train L2")
    plt.plot(epochs, df["test_l2"], label="Test L2")
    plt.yscale("log")
    ax = plt.gca()

    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.01, 0.1, 0.2, 0.3, 0.5), numticks=12))
    
    formatter = ticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)   

    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error (log scale)")
    plt.title("Overall Learning Curve")
    plt.legend()
    # plt.grid()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curve_total_log.png"), dpi=300)
    plt.show()

# -----------------------------
# Channel-wise MSE
# -----------------------------
def channel_wise_MSE(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["pressure_test_mse"], label="Pressure")
    plt.plot(epochs, df["temperature_test_mse"], label="Temperature")
    plt.plot(epochs, df["x_velocity_test_mse"], label="X Velocity")
    plt.plot(epochs, df["y_velocity_test_mse"], label="Y Velocity")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Channel-wise MSE")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "learning_curve_mse.png"), dpi=300)
    plt.show()

def channel_wise_MSE_log(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["pressure_test_mse"], label="Pressure")
    plt.plot(epochs, df["temperature_test_mse"], label="Temperature")
    plt.plot(epochs, df["x_velocity_test_mse"], label="X Velocity")
    plt.plot(epochs, df["y_velocity_test_mse"], label="Y Velocity")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("Channel-wise MSE")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "learning_curve_mse_log.png"), dpi=300)
    plt.show()

# -----------------------------
# Channel-wise Relative Error
# -----------------------------
def channel_wise_l2(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["pressure_test_rel"], label="Pressure")
    plt.plot(epochs, df["temperature_test_rel"], label="Temperature")
    plt.plot(epochs, df["x_velocity_test_rel"], label="X Velocity")
    plt.plot(epochs, df["y_velocity_test_rel"], label="Y Velocity")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.title("Channel-wise Relative Error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "learning_curve_relative.png"), dpi=300)
    plt.show()

def channel_wise_l2_log(df, save_dir):
    epochs = df["epoch"]

    plt.figure()
    plt.plot(epochs, df["pressure_test_rel"], label="Pressure")
    plt.plot(epochs, df["temperature_test_rel"], label="Temperature")
    plt.plot(epochs, df["x_velocity_test_rel"], label="X Velocity")
    plt.plot(epochs, df["y_velocity_test_rel"], label="Y Velocity")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error (log scale)")
    plt.title("Channel-wise Relative Error")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "learning_curve_relative_log.png"), dpi=300)
    plt.show()

# def test():
#     csv_files = [
#         "pred/2d_N1000_ep500_batch20_s64_mode6_width32_lr0.001_fluidonly.csv",
#         "pred/2d_N1000_ep500_batch20_s64_mode12_width32_lr0.001_fluidonly.csv",
#         "pred/2d_N1000_ep500_batch20_s64_mode20_width32_lr0.001_fluidonly.csv",
#         "pred/2d_N1000_ep500_batch20_s64_mode30_width32_lr0.001_fluidonly.csv",
#         "pred/2d_N1000_ep500_batch20_s64_mode40_width32_lr0.001_fluidonly.csv"
#     ]

#     models = ['mode6','mode12','mode20','mode30','mode40']

#     # csv_files = [
#     #     "pred/2d_N1000_ep500_batch20_s64_mode20_width8_lr0.001_fluidonly.csv",
#     #     "pred/2d_N1000_ep500_batch20_s64_mode20_width16_lr0.001_fluidonly.csv",
#     #     "pred/2d_N1000_ep500_batch20_s64_mode20_width32_lr0.001_fluidonly.csv",
#     #     "pred/2d_N1000_ep500_batch20_s64_mode20_width64_lr0.001_fluidonly.csv",
#     #     "pred/2d_N1000_ep500_batch20_s64_mode20_width128_lr0.001_fluidonly.csv"
#     # ]


#     # models = ['width8','width16','width32','width64','width128']

#     colors = ['b', 'r', 'g', 'm', 'c']

#     plt.figure(figsize=(10,6))

#     for i, file in enumerate(csv_files):
#         df = pd.read_csv(file)
        
#         plt.plot(df['epoch'], df['train_l2'], color=colors[i], label=f'{models[i]}')
#         # plt.plot(df['epoch'], df['test_l2'], color=colors[i], label=f'{models[i]}')

#     plt.yscale('log')
#     plt.xlabel('Epoch')
#     plt.ylabel('Relative Loss (Log scale)')
#     plt.title('Train Relative Loss Comparison')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('figures/mode_wise_train_loss_comparison.png')
#     # plt.show()


if __name__ == "__main__":
    # test()
    csv_path = "pred/2d_N1000_ep500_batch20_s64_mode25_width128_constantlr0.001_wd1e-4.csv"
    
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir = os.path.join("figures", base_name)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    l2(df, save_dir)
    l2_log(df, save_dir)
    # channel_wise_MSE(df, save_dir)
    # channel_wise_MSE_log(df, save_dir)
    # channel_wise_l2(df, save_dir)
    # channel_wise_l2_log(df, save_dir)