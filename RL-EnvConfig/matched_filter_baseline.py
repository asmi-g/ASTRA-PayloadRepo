#!/usr/bin/env python3
"""
matched_filter_baseline.py

Diagnostic check (not a deployable filter): does the wavelet threshold
filter's ~0.2-0.4dB ceiling on flight_signal_1 reflect the noise severity
itself, or a limitation specific to per-1000-sample-window wavelet
thresholding? Tests two alternatives that exploit prior knowledge of the
transmitted signal, which wavelet thresholding never uses, sweeping
coherent-integration/block length as the key variable:

  1. MATCHED FILTER (oracle upper bound): estimates a real-valued gain via
     least-squares against the KNOWN CLEAN REFERENCE over each block, then
     reconstructs as A_hat * clean. This is not something a real deployed
     system denoising an unknown signal could do (it requires already having
     a perfect copy of what was sent) -- it answers "how much is recoverable
     in principle," not "how well could we actually do."

  2. NARROWBAND FILTER (realistic/generalizable): zeros out every FFT bin
     outside a band around the KNOWN CARRIER FREQUENCY (not the exact
     waveform), per block. This only assumes knowledge of where the signal
     lives in frequency -- realistic even for a real data-modulated signal
     with an unknown data payload but a known carrier -- so unlike (1) it IS
     a fair like-for-like alternative to wavelet thresholding.

Block length is swept up to 500,000 samples (one real flight segment's
length -- task 1) since coherence assumptions break down across a segment
boundary; beyond that, CFO/gain shift underneath you.

Uses flight_signal_1_clean_noisy.csv (build_clean_noisy.py's offline,
per-segment-precise reconstruction), matching the original static_filter_baseline.py
numbers -- this is a ceiling/diagnostic question, not a deployment-realism one,
so the most accurate available reconstruction is the right basis.

Usage:
    python matched_filter_baseline.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATED_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data/simulated_signal_match_hz.csv"))
FLIGHT_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data/flight_signal_1_clean_noisy.csv"))

SAMP_RATE = 1_000_000
F0_NOMINAL = 100_000
NARROWBAND_HALFWIDTH_HZ = 50_000  # matches inference.py's search_hz estimate, for consistency

BLOCK_LENGTHS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]


def snr_db(clean, test):
    noise = test - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def matched_filter_block(clean_block, noisy_block):
    denom = np.dot(clean_block, clean_block)
    if denom == 0:
        return np.zeros_like(clean_block)
    A_hat = np.dot(clean_block, noisy_block) / denom
    return A_hat * clean_block


def narrowband_filter_block(noisy_block, fs=SAMP_RATE, f0=F0_NOMINAL, halfwidth=NARROWBAND_HALFWIDTH_HZ):
    n = len(noisy_block)
    spectrum = np.fft.fft(noisy_block)
    freqs = np.fft.fftfreq(n, d=1 / fs)
    passband = np.abs(np.abs(freqs) - f0) < halfwidth
    spectrum[~passband] = 0
    return np.real(np.fft.ifft(spectrum))


def sweep_block_lengths(clean, noisy, label, block_lengths=BLOCK_LENGTHS):
    rows = []
    for n_block in block_lengths:
        n_blocks_total = len(clean) // n_block
        if n_blocks_total == 0:
            continue
        starts = np.arange(n_blocks_total) * n_block

        mf_snr_imps, nb_snr_imps = [], []
        for s in starts:
            c = clean[s:s + n_block]
            nz = noisy[s:s + n_block]
            snr_raw = snr_db(c, nz)

            mf_filt = matched_filter_block(c, nz)
            mf_snr_imps.append(snr_db(c, mf_filt) - snr_raw)

            nb_filt = narrowband_filter_block(nz)
            nb_snr_imps.append(snr_db(c, nb_filt) - snr_raw)

        rows.append({
            "label": label,
            "block_length": n_block,
            "n_blocks": n_blocks_total,
            "matched_filter_snr_improvement_db": np.mean(mf_snr_imps),
            "narrowband_filter_snr_improvement_db": np.mean(nb_snr_imps),
        })
        print(f"[{label}] block_length={n_block:>7}  n_blocks={n_blocks_total:>6}  "
              f"matched_filter={np.mean(mf_snr_imps):>8.3f} dB  "
              f"narrowband={np.mean(nb_snr_imps):>8.3f} dB")
    return pd.DataFrame(rows)


def load_signal(csv_path):
    df = pd.read_csv(csv_path)
    clean = df["Clean Signal"].to_numpy(dtype=np.float64)
    noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
    return clean, noisy


def plot_results(results_df, out_path="matched_filter_sweep.png"):
    labels = results_df["label"].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = results_df[results_df["label"] == label]
        ax.semilogx(sub["block_length"], sub["matched_filter_snr_improvement_db"],
                     "o-", label="matched filter (oracle, knows TX)")
        ax.semilogx(sub["block_length"], sub["narrowband_filter_snr_improvement_db"],
                     "s-", label="narrowband filter (knows carrier freq only)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{label}: SNR improvement vs. block length")
        ax.set_xlabel("block length (samples)")
        ax.set_ylabel("SNR improvement (dB)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    sim_clean, sim_noisy = load_signal(SIMULATED_CSV)
    flight_clean, flight_noisy = load_signal(FLIGHT_CSV)

    sim_results = sweep_block_lengths(sim_clean, sim_noisy, "Simulated")
    flight_results = sweep_block_lengths(flight_clean, flight_noisy, "Flight")

    all_results = pd.concat([sim_results, flight_results], ignore_index=True)
    plot_results(all_results)

    all_results.to_csv("matched_filter_baseline_results.csv", index=False)
    print("[INFO] saved matched_filter_baseline_results.csv")
    print()
    pd.set_option("display.width", 160)
    print(all_results.to_string(index=False))
