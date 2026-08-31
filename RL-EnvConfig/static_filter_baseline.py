#!/usr/bin/env python3
"""
static_filter_baseline.py

Task 6: find the optimal STATIC (fixed, non-adaptive) wavelet threshold_factor
for simulated_signal_match_hz.csv and flight_signal_1_clean_noisy.csv, and
report the resulting SNR improvement -- a baseline for "how would a normal,
non-adaptive filter do" to contrast against the RL agent's per-window
adaptive filtering (OFT/UN models).

Filter mechanics match StatelessDenoisingEnv.apply_filter in
custom_env_022025.py exactly (db4, level=5, periodization, VisuShrink-style
threshold, details-only thresholding, mean-preserving reconstruction), and
filtering is done window-by-window (window_size=1000, matching what the RL
agent actually processes per step) with ONE FIXED threshold_factor applied
to every window. This isolates "fixed vs. adaptive" as the only variable
being tested, rather than conflating it with whole-signal-at-once vs.
windowed filtering (which use different effective threshold scaling -- see
reward_funct_test.py's window-size fidelity fix).

The threshold_factor sweep range here is deliberately NOT constrained to the
RL agent's achievable action range -- this baseline represents "how well
could a competently-tuned static filter do," not "how well could a filter
limited to the RL agent's action space do."

For flight_signal_1 (non-stationary noise, per task 3), the search-phase
window sample is spread evenly across the whole file rather than taken from
just the first chunk, so the chosen threshold isn't biased toward whatever
segment happens to come first.

Usage:
    python static_filter_baseline.py
"""

import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_SIZE = 1000
WAVELET = 'db4'
LEVEL = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATED_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data/simulated_signal_match_hz.csv"))
FLIGHT_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data/flight_signal_1_clean_noisy.csv"))

SWEEP_RANGE = np.linspace(0.05, 30.0, 150)
SWEEP_MAX_WINDOWS = 2000  # subset for the search phase (speed); final report uses the FULL signal


def apply_filter(x, threshold_factor, wavelet=WAVELET, level=LEVEL):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="periodization")
    # Floored above 0 -- see custom_env_022025.py apply_filter for why an
    # unfloored sigma=0 (from heavily-quantized input) makes every
    # threshold_factor collapse to an identical degenerate all-zero output.
    sigma = max(np.median(np.abs(coeffs[-1])) / 0.6745, 1e-8) if coeffs[-1].size else 1e-8
    lam = threshold_factor * sigma * np.sqrt(2 * np.log(max(n, 2)))
    cA, details = coeffs[0], coeffs[1:]
    details = [pywt.threshold(c, lam, mode="soft") for c in details]
    y = pywt.waverec([cA] + details, wavelet, mode="periodization")[:n]
    y += (np.mean(x) - np.mean(y))
    return np.nan_to_num(y, copy=False)


def snr_db(clean, test):
    noise = test - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def window_starts(n_samples, window_size=WINDOW_SIZE, max_windows=None, spread=False):
    n_windows_total = n_samples // window_size
    if max_windows is None or n_windows_total <= max_windows:
        idx = np.arange(n_windows_total)
    elif spread:
        idx = np.linspace(0, n_windows_total - 1, max_windows).astype(int)
    else:
        idx = np.arange(max_windows)
    return idx * window_size


DEGENERATE_ENERGY_RATIO = 0.2  # matches reward_funct_test.py's degeneracy floor


def filter_at_starts(clean, noisy, threshold_factor, starts, window_size=WINDOW_SIZE):
    snr_improvements = np.empty(len(starts))
    energy_ratios = np.empty(len(starts))
    filtered_chunks = []
    for i, s in enumerate(starts):
        c = clean[s:s + window_size]
        nz = noisy[s:s + window_size]
        filt = apply_filter(nz, threshold_factor)
        snr_improvements[i] = snr_db(c, filt) - snr_db(c, nz)
        energy_ratios[i] = np.sum(filt ** 2) / (np.sum(nz ** 2) + 1e-12)
        filtered_chunks.append(filt)
    return snr_improvements, energy_ratios, filtered_chunks


def find_optimal_threshold(clean, noisy, label, sweep_range=SWEEP_RANGE, max_windows=SWEEP_MAX_WINDOWS):
    # A threshold_factor that zeros out every detail coefficient (collapsing
    # the output to a near-constant signal) trivially "improves" SNR by
    # destroying noise AND signal alike -- not real denoising. Excluded from
    # the optimum search the same way reward_funct_test.py's Pareto analysis
    # flags degenerate points (energy_ratio < 0.2), rather than letting the
    # sweep pick a threshold whose "improvement" is a numerical artifact.
    starts = window_starts(len(clean), max_windows=max_windows, spread=True)
    print(f"[{label}] sweeping {len(sweep_range)} threshold_factor values over {len(starts)} windows...")
    mean_improvements = np.empty(len(sweep_range))
    mean_energy_ratios = np.empty(len(sweep_range))
    for i, tf in enumerate(sweep_range):
        snr_imp, energy_ratios, _ = filter_at_starts(clean, noisy, tf, starts)
        mean_improvements[i] = snr_imp.mean()
        mean_energy_ratios[i] = energy_ratios.mean()

    non_degenerate = mean_energy_ratios >= DEGENERATE_ENERGY_RATIO
    if not non_degenerate.any():
        raise RuntimeError(f"[{label}] every threshold_factor in the sweep range is degenerate "
                            f"(energy_ratio < {DEGENERATE_ENERGY_RATIO}) -- widen/lower the sweep range.")
    n_degenerate = (~non_degenerate).sum()
    if n_degenerate:
        print(f"[{label}] excluded {n_degenerate}/{len(sweep_range)} degenerate threshold_factor values "
              f"(energy_ratio < {DEGENERATE_ENERGY_RATIO}) from the optimum search")

    candidate_improvements = np.where(non_degenerate, mean_improvements, -np.inf)
    best_idx = np.argmax(candidate_improvements)
    return sweep_range[best_idx], mean_improvements[best_idx], sweep_range, mean_improvements


def evaluate_full(clean, noisy, threshold_factor, label):
    """Final pass over the FULL signal (every window) at the chosen static threshold_factor."""
    starts = window_starts(len(clean), max_windows=None)
    snr_imp, energy_ratios, filtered_chunks = filter_at_starts(clean, noisy, threshold_factor, starts)
    filtered = np.concatenate(filtered_chunks)
    n = len(filtered)
    snr_raw = snr_db(clean[:n], noisy[:n])
    snr_filtered = snr_db(clean[:n], filtered)
    print(f"[{label}] FULL-SIGNAL static filter @ threshold_factor={threshold_factor:.3f}")
    print(f"    n_windows = {len(snr_imp)}  (n_samples = {n})")
    print(f"    SNR raw       = {snr_raw:.3f} dB")
    print(f"    SNR filtered  = {snr_filtered:.3f} dB")
    print(f"    SNR improvement (mean over windows) = {snr_imp.mean():.4f} dB  (std = {snr_imp.std():.4f})")
    return {
        "label": label,
        "threshold_factor": threshold_factor,
        "n_samples": n,
        "snr_raw_db": snr_raw,
        "snr_filtered_db": snr_filtered,
        "snr_improvement_mean_db": snr_imp.mean(),
        "snr_improvement_std_db": snr_imp.std(),
    }, filtered


def load_signal(csv_path):
    df = pd.read_csv(csv_path)
    clean = df["Clean Signal"].to_numpy(dtype=np.float64)
    noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
    return clean, noisy


def plot_sweep(results_by_label, out_path="static_filter_sweep.png"):
    fig, axes = plt.subplots(1, len(results_by_label), figsize=(7 * len(results_by_label), 5))
    if len(results_by_label) == 1:
        axes = [axes]
    for ax, (label, (best_tf, best_val, sweep_range, mean_improvements)) in zip(axes, results_by_label.items()):
        ax.plot(sweep_range, mean_improvements)
        ax.axvline(best_tf, color='r', linestyle='--', label=f'optimal={best_tf:.3f}')
        ax.set_title(f"{label}: static threshold_factor sweep")
        ax.set_xlabel("threshold_factor")
        ax.set_ylabel("mean SNR improvement (dB)")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] saved {out_path}")


def plot_waveform(clean, filtered, label, out_path, start=1000, n_samples=300):
    end = min(start + n_samples, len(clean))
    idx = np.arange(start, end)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, clean[start:end], color="black", linewidth=1.3, label="Clean")
    ax.plot(idx, filtered[start:end], color="tab:blue", linewidth=1.0, label="Filtered (static)")
    ax.set_title(f"{label}: static-filter output vs. clean, samples [{start}, {end})")
    ax.set_xlabel("Sample index")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    sim_clean, sim_noisy = load_signal(SIMULATED_CSV)
    flight_clean, flight_noisy = load_signal(FLIGHT_CSV)

    results_by_label = {}
    summary_rows = []

    for label, (clean, noisy) in [("Simulated", (sim_clean, sim_noisy)), ("Flight", (flight_clean, flight_noisy))]:
        best_tf, best_val, sweep_range, mean_improvements = find_optimal_threshold(clean, noisy, label)
        results_by_label[label] = (best_tf, best_val, sweep_range, mean_improvements)
        summary, filtered = evaluate_full(clean, noisy, best_tf, label)
        summary_rows.append(summary)
        plot_waveform(clean, filtered, label, f"static_filter_waveform_{label.lower()}.png")

    plot_sweep(results_by_label)

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 160)
    print()
    print(summary_df.to_string(index=False))
    summary_df.to_csv("static_filter_baseline_results.csv", index=False)
    print("[INFO] saved static_filter_baseline_results.csv")
