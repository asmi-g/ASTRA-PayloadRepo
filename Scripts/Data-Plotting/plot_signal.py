#plot_signal.py
"""
Visualize the reconstructed clean / noisy / filtered signal exported by
inference_with_signal_export.py.

Usage:
    python plot_signal.py                          # auto-picks the newest *_signal.csv in DATA_DIR
    python plot_signal.py path/to/some_signal.csv   # or point at a specific file
    python plot_signal.py --start 5000 --end 6000   # zoom into a sample range
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Data/")

def find_latest_signal_csv(data_dir):
    candidates = glob.glob(os.path.join(data_dir, "*_signal.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_signal.csv files found in {data_dir}")
    return max(candidates, key=os.path.getmtime)


def load_signal_data(csv_path):
    df = pd.read_csv(csv_path).rename(columns={'Index': 'sample_index', 'Clean Signal': 'clean_signal', 'Noisy Signal':'noisy_signal'})
    required_cols = {"sample_index", "clean_signal", "noisy_signal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # sort + dedupe in case of overlapping writes from multiple flush cycles
    df = df.drop_duplicates(subset="sample_index").sort_values("sample_index").reset_index(drop=True)
    return df


def compute_snr(clean, test):
    noise = test - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))


def plot_signals(df, start=None, end=None, save_path=None):
    if start is not None or end is not None:
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df["sample_index"] >= start
        if end is not None:
            mask &= df["sample_index"] <= end
        df = df[mask]

    if df.empty:
        raise ValueError("No data in the selected sample range.")

    x = df["sample_index"]
    clean = df["clean_signal"].to_numpy()
    noisy = df["noisy_signal"].to_numpy()
    filtered = df["filtered_signal"].to_numpy()

    print("clean")
    # Duplicate-row hypothesis: consecutive rows equal
    print((df["clean_signal"].diff() == 0).sum())
    # Periodicity hypothesis: value 1000 samples apart equal, but neighbors differ
    print((df["clean_signal"].diff(periods=1000) == 0).sum())
    print(df["clean_signal"].nunique(), "unique values out of", len(df))  # low unique count -> quantization
    print(df["clean_signal"].value_counts().head(5))
    print("noisy")
    # Duplicate-row hypothesis: consecutive rows equal
    print((df["noisy_signal"].diff() == 0).sum())
    # Periodicity hypothesis: value 1000 samples apart equal, but neighbors differ
    print((df["noisy_signal"].diff(periods=1000) == 0).sum())
    print(df["noisy_signal"].nunique(), "unique values out of", len(df))  # low unique count -> quantization
    print(df["noisy_signal"].value_counts())

    snr_before = compute_snr(clean, noisy)
    snr_after = compute_snr(clean, filtered)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, clean, color="black", linewidth=1)
    axes[0].set_title("Clean (reference) Signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(x, noisy, color="tab:red", linewidth=0.7)
    axes[1].set_title(f"Noisy Signal  (SNR = {snr_before:.2f} dB)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(x, filtered, color="tab:blue", linewidth=1)
    axes[2].set_title(f"Filtered (Denoised) Signal  (SNR = {snr_after:.2f} dB, "
                       f"Improvement = {snr_after - snr_before:.2f} dB)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Sample Index")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Overlay comparison plot
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(x, noisy, color="tab:red", linewidth=0.7, label="Noisy")
    ax2.plot(x, clean, color="black", alpha=0.4, linewidth=1.2, label="Clean")
    ax2.plot(x, filtered, color="tab:blue", linewidth=1.2, label="Filtered", linestyle="--")
    ax2.set_title("Overlay: Clean vs Noisy")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        fig.savefig(f"{base}_stacked{ext}", dpi=150)
        fig2.savefig(f"{base}_overlay{ext}", dpi=150)
        print(f"[INFO] Saved plots to {base}_stacked{ext} and {base}_overlay{ext}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot clean/noisy/filtered signal reconstruction.")
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to a *_signal.csv file. Defaults to the newest one in DATA_DIR.")
    parser.add_argument("--start", type=int, default=None, help="Start sample index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End sample index (inclusive).")
    parser.add_argument("--save", type=str, default=None,
                         help="Base path to save PNG outputs (e.g. plots/run1.png).")
    args = parser.parse_args()

    csv_path = args.csv_path or find_latest_signal_csv(DATA_DIR)
    print(f"[INFO] Loading signal data from {csv_path}")

    df = load_signal_data(csv_path)
    print(f"[INFO] Loaded {len(df)} samples, index range [{df['sample_index'].min()}, {df['sample_index'].max()}]")

    plot_signals(df, start=args.start, end=args.end, save_path=args.save)


if __name__ == "__main__":
    main()