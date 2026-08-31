"""
Phase 0 + Phase A reward function testing for RL-based signal denoising.

Phase 0 (reward landscape, NO RL training):
    For a fixed noisy signal, sweep the filter parameter (wavelet soft-threshold)
    across a range and compute, at every point:
        - SNR_raw (no filtering)
        - SNR_filtered
        - delta_SNR
        - MSE (clean vs filtered)
        - correlation (clean vs filtered)
        - each candidate reward formula
    This tells you whether the reward landscape is sensible BEFORE spending any
    RL training compute on it.

Phase A (baseline comparison):
    For each reward formulation, find its best operating point (the threshold
    that maximizes that specific reward) and report delta_SNR / MSE / correlation
    there, alongside the untouched raw noisy signal. This is a proxy for "what
    would an agent converge to if it perfectly optimized this reward" -- it is
    NOT a substitute for actually training an agent (that's Phase C, mini-RL),
    but it is a cheap sanity check on what each reward formulation rewards.

Reward formulations compared (as requested):
    R1 = delta_SNR                              (SNR improvement only)
    R2 = delta_SNR - MSE                        (SNR improvement - MSE)
    R3 = delta_SNR - MSE + correlation          (SNR improvement - MSE + corr)

NOTE ON SCALE: delta_SNR is in dB (can be O(1-10)), MSE and correlation are on
different natural scales. This script intentionally uses raw (unweighted, w=1)
combinations for Phase 0/A so you can SEE the scale mismatch and decide whether
normalization or explicit weights (w1, w2, w3) are needed -- that decision is
exactly what Phase B (weight sweep) is for. Don't over-tune weights here.

Replace `apply_filter()` and `generate_signal()` with your actual environment's
filter and signal generator to make this directly representative of your
NoiseReductionEnv. The wavelet soft-threshold filter here is a stand-in.
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

RNG_SEED = 42
FS = 1e6            # sample rate (Hz)
DURATION = 1.0        # seconds
SIGNAL_FREQ = 100_000    # Hz, clean tone frequency
NOISE_STD = 0.6        # AWGN std dev added to clean signal
THRESHOLDS = np.linspace(0.0, 2.5, 31)   # filter parameter sweep, 0 = no filtering
WAVELET = "db4"
levels = 5

# Signal generation
def generate_signal():
    signal_length = 10_000
    x = np.arange(signal_length)
    t = x / 1e4  # F_SAMPLING = 10 kHz
    f_signal = 100  # Hz

    clean = np.sin(2 * np.pi * f_signal * t)
    white_noise = np.random.normal(0, np.sqrt(0.1), size=signal_length)
    pink_noise = generate_pink_noise(signal_length, white_noise)
    noisy = clean + white_noise + pink_noise
    noisy = add_bursts(noisy)

    return t, clean.astype(np.float64), noisy.astype(np.float64)

def generate_pink_noise(size, white_noise):
    freq = np.fft.fftfreq(size)
    pink_filter = np.sqrt(np.abs(freq) + 1e-6)
    spectrum = np.fft.fft(white_noise / np.sqrt(0.1))
    spectrum *= pink_filter
    pink_noise = np.fft.ifft(spectrum)
    return np.real(pink_noise)
    
def add_bursts(signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
    noisy_signal = signal.copy()
    for i in range(len(signal)):
        if np.random.rand() < burst_probability:
            burst_start = i
            burst_end = min(i + burst_duration, len(signal))
            noisy_signal[burst_start:burst_end] += np.random.uniform(-burst_amplitude, burst_amplitude)
    return noisy_signal



# Filter under test (swap this out for your actual environment's filter)
def apply_filter(window, threshold_factor, wavelet='db4', level=5):
    x = np.asarray(window, dtype=np.float64)
    n = len(x)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    lam = threshold_factor * sigma * np.sqrt(2*np.log(max(n,2)))
    cA, details = coeffs[0], coeffs[1:]
    details = [pywt.threshold(c, lam, mode="soft") for c in details]
    y = pywt.waverec([cA] + details, wavelet, mode="periodization")[:n]
    y += (np.mean(x) - np.mean(y))
    return np.nan_to_num(y, copy=False)

# Metrics
def snr_db(clean, test_signal):
    noise = test_signal - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-15) / (noise_power + 1e-15))

def compute_metrics(clean, noisy, filtered):
    snr_raw = snr_db(clean, noisy)
    snr_filt = snr_db(clean, filtered)
    delta_snr = snr_filt - snr_raw
    mse = float(np.mean((clean - filtered) ** 2))
    corr = float(np.corrcoef(clean, filtered)[0, 1])
    return {
        "snr_raw": snr_raw,
        "snr_filtered": snr_filt,
        "delta_snr": delta_snr,
        "mse": mse,
        "correlation": corr,
    }


def compute_rewards(m):
    r1 = m["delta_snr"]
    r2 = m["delta_snr"] - m["mse"]
    r3 = m["delta_snr"] - m["mse"] + m["correlation"]
    return {"R1_snr_only": r1, "R2_snr_minus_mse": r2, "R3_snr_minus_mse_plus_corr": r3}


# Phase 0: reward landscape sweep
def run_phase0():
    t, clean, noisy = generate_signal()
    rows = []
    for thr in THRESHOLDS:
        filtered = apply_filter(noisy, thr)
        m = compute_metrics(clean, noisy, filtered)
        r = compute_rewards(m)
        rows.append({"threshold": thr, **m, **r})
    df = pd.DataFrame(rows)
    df.to_csv("phase0_reward_landscape.csv", index=False)
    return t, clean, noisy, df


def plot_phase0(df, t, clean, noisy):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Metrics vs threshold
    ax = axes[0, 0]
    ax.plot(df["threshold"], df["delta_snr"], label="delta_SNR (dB)", marker="o", ms=3)
    ax.plot(df["threshold"], df["mse"], label="MSE", marker="s", ms=3)
    ax.plot(df["threshold"], df["correlation"], label="correlation", marker="^", ms=3)
    ax.set_xlabel("filter threshold")
    ax.set_ylabel("metric value")
    ax.set_title("Raw metrics vs. filter threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Reward formulas vs threshold
    ax = axes[0, 1]
    ax.plot(df["threshold"], df["R1_snr_only"], label="R1 = delta_SNR", marker="o", ms=3)
    ax.plot(df["threshold"], df["R2_snr_minus_mse"], label="R2 = delta_SNR - MSE", marker="s", ms=3)
    ax.plot(df["threshold"], df["R3_snr_minus_mse_plus_corr"], label="R3 = delta_SNR - MSE + corr", marker="^", ms=3)
    ax.set_xlabel("filter threshold")
    ax.set_ylabel("reward")
    ax.set_title("Candidate reward formulas vs. filter threshold")
    ax.axvline(df.loc[df["R1_snr_only"].idxmax(), "threshold"], color="C0", ls="--", alpha=0.5)
    ax.axvline(df.loc[df["R2_snr_minus_mse"].idxmax(), "threshold"], color="C1", ls="--", alpha=0.5)
    ax.axvline(df.loc[df["R3_snr_minus_mse_plus_corr"].idxmax(), "threshold"], color="C2", ls="--", alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # delta_SNR vs MSE trade-off (proto Pareto view)
    ax = axes[1, 0]
    sc = ax.scatter(df["mse"], df["delta_snr"], c=df["threshold"], cmap="viridis")
    ax.set_xlabel("MSE")
    ax.set_ylabel("delta_SNR (dB)")
    ax.set_title("delta_SNR vs. MSE across thresholds (color = threshold)")
    plt.colorbar(sc, ax=ax, label="threshold")
    ax.grid(alpha=0.3)

    # Example waveform at a mid threshold, for a visual sanity check
    ax = axes[1, 1]
    mid_thr = THRESHOLDS[len(THRESHOLDS) // 2]
    filtered = apply_filter(noisy, mid_thr)
    ax.plot(t, noisy, label="noisy", alpha=0.4)
    ax.plot(t, clean, label="clean", lw=2)
    ax.plot(t, filtered, label=f"filtered (thr={mid_thr:.2f})", lw=1.5)
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("time (s)")
    ax.set_title("Example waveform (zoomed)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


# Phase A: baseline comparison
def run_phase_a(df, clean, noisy):
    """
    For each reward formulation, report the metrics at the threshold that
    maximizes THAT formulation. This approximates "what an agent perfectly
    optimizing this reward would converge to."
    """
    raw_metrics = compute_metrics(clean, noisy, noisy)  # threshold=0, no filtering

    rows = [{
        "condition": "raw_noisy (no filtering)",
        "threshold": 0.0,
        "delta_snr": raw_metrics["delta_snr"],
        "mse": raw_metrics["mse"],
        "correlation": raw_metrics["correlation"],
    }]

    reward_cols = ["R1_snr_only", "R2_snr_minus_mse", "R3_snr_minus_mse_plus_corr"]
    labels = {
        "R1_snr_only": "SNR-only reward, best threshold",
        "R2_snr_minus_mse": "SNR - MSE reward, best threshold",
        "R3_snr_minus_mse_plus_corr": "SNR - MSE + corr reward, best threshold",
    }
    for col in reward_cols:
        best_idx = df[col].idxmax()
        best_row = df.loc[best_idx]
        rows.append({
            "condition": labels[col],
            "threshold": best_row["threshold"],
            "delta_snr": best_row["delta_snr"],
            "mse": best_row["mse"],
            "correlation": best_row["correlation"],
        })

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv("phaseA_baseline_comparison.csv", index=False)
    return baseline_df


# Main
if __name__ == "__main__":
    t, clean, noisy, df = run_phase0()
    plot_phase0(df, t, clean, noisy)
    baseline_df = run_phase_a(df, clean, noisy)

    print("=" * 70)
    print("PHASE 0: reward landscape sweep saved to")
    print("PHASE A: baseline comparison")
    print("=" * 70)
    print(baseline_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))