#!/usr/bin/env python3
"""
verify_clean_noisy.py

Visual sanity check for a *_clean_noisy.csv produced by build_clean_noisy.py:
a zoomed-in view (waveform shape should be visible) and a broader view
(overall noise character / envelope). Saves PNGs instead of showing an
interactive window.

Usage:
    python verify_clean_noisy.py path/to/flight_signal_1_clean_noisy.csv [start] [n_zoom]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(csv_path, start=100, n_zoom=300, n_wide=500_000):
    df = pd.read_csv(csv_path)
    clean = df["Clean Signal"].to_numpy(dtype=np.float64)
    noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
    n = len(df)
    print(f"[INFO] loaded {n} samples from {csv_path}")

    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20))
    print(f"[INFO] clean std={clean.std():.6g}  noisy std={noisy.std():.6g}  noise std={noise.std():.6g}")
    print(f"[INFO] SNR (full file) = {snr:.2f} dB")

    # zoomed view -- should show the sinusoid shape (10 samples/cycle at 100kHz/1MHz)
    end_zoom = min(start + n_zoom, n)
    idx_z = np.arange(start, end_zoom)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx_z, noisy[start:end_zoom], color="tab:red", linewidth=0.8, label="Noisy", alpha=0.7)
    ax.plot(idx_z, clean[start:end_zoom], color="black", linewidth=1.3, label="Clean")
    ax.set_title(f"Zoomed view: samples [{start}, {end_zoom})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    zoom_path = "verify_clean_noisy_zoom.png"
    plt.savefig(zoom_path, dpi=150)
    print(f"[INFO] saved {zoom_path}")

    # wide view -- overall noise character / envelope, downsampled for plotting speed.
    # NOTE: naive stride-based downsampling (every Nth sample) aliases a fast
    # oscillating tone like the 100kHz carrier (only 10 samples/cycle at 1MHz)
    # into a spurious near-zero beat pattern. Use a min/max envelope per bin
    # instead, which represents the true amplitude envelope without aliasing.
    end_wide = min(n_wide, n)
    n_bins = min(20000, end_wide)
    bin_size = max(1, end_wide // n_bins)
    n_bins = end_wide // bin_size
    bin_idx = np.arange(n_bins) * bin_size

    def envelope(x):
        trimmed = x[:n_bins * bin_size].reshape(n_bins, bin_size)
        return trimmed.min(axis=1), trimmed.max(axis=1)

    clean_min, clean_max = envelope(clean[:end_wide])
    noisy_min, noisy_max = envelope(noisy[:end_wide])

    fig2, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].fill_between(bin_idx, clean_min, clean_max, color="black", linewidth=0)
    axes[0].set_title(f"Clean Signal envelope (first {end_wide} samples, {bin_size} samples/bin)")
    axes[1].fill_between(bin_idx, noisy_min, noisy_max, color="tab:red", linewidth=0)
    axes[1].set_title(f"Noisy Signal envelope (first {end_wide} samples, {bin_size} samples/bin)")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    wide_path = "verify_clean_noisy_wide.png"
    plt.savefig(wide_path, dpi=150)
    print(f"[INFO] saved {wide_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_clean_noisy.py path/to/flight_signal_1_clean_noisy.csv [start] [n_zoom]")
        sys.exit(1)
    csv_arg = sys.argv[1]
    start_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_zoom_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    main(csv_arg, start_arg, n_zoom_arg)
