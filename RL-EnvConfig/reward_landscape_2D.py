"""
Phase 0 + Phase A reward function testing, 2D version (threshold x level).

Same purpose as before, but the filter parameter space is now 2D:
    threshold_factor (soft-threshold strength)
    level             (wavelet decomposition depth)

This matters because Phase 0 debugging showed the reward landscape plateaus
early if `level` is held fixed -- level was the real bottleneck, not
threshold_factor. This version sweeps both jointly so you can see the real
optimum and whether the two parameters interact (i.e. whether the best level
depends on the chosen threshold, and vice versa).
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.linspace(0.0, 2.5, 26)   # filter threshold_factor sweep
LEVELS = [1, 2, 3, 4, 5, 6]              # decomposition depth sweep
WAVELET = "db4"


# ---------------------------------------------------------------------------
# Signal generation (unchanged from your environment)
# ---------------------------------------------------------------------------
def generate_signal():
    signal_length = 10_000
    x = np.arange(signal_length)
    t = x / 1e6
    f_signal = 100_000

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


# ---------------------------------------------------------------------------
# Filter under test
# ---------------------------------------------------------------------------
def apply_filter(window, threshold_factor, wavelet='db4', level=1):
    x = np.asarray(window, dtype=np.float64)
    n = len(x)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    lam = threshold_factor * sigma * np.sqrt(2 * np.log(max(n, 2)))
    cA, details = coeffs[0], coeffs[1:]
    details = [pywt.threshold(c, lam, mode="soft") for c in details]
    y = pywt.waverec([cA] + details, wavelet, mode="periodization")[:n]
    y += (np.mean(x) - np.mean(y))
    return np.nan_to_num(y, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Phase 0: 2D reward landscape sweep (threshold x level)
# ---------------------------------------------------------------------------
def run_phase0():
    t, clean, noisy = generate_signal()
    rows = []
    for thr in THRESHOLDS:
        for lvl in LEVELS:
            filtered = apply_filter(noisy, thr, level=lvl)
            m = compute_metrics(clean, noisy, filtered)
            r = compute_rewards(m)
            rows.append({"threshold": thr, "level": lvl, **m, **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "phase0_reward_landscape_2d.csv", index=False)
    return t, clean, noisy, df


def plot_phase0(df, t, clean, noisy):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    reward_cols = ["R1_snr_only", "R2_snr_minus_mse", "R3_snr_minus_mse_plus_corr"]

    # (0,0): reward vs threshold, one line per level, for R3 (change if you want a different one)
    ax = axes[0, 0]
    for lvl in LEVELS:
        sub = df[df["level"] == lvl]
        ax.plot(sub["threshold"], sub["R3_snr_minus_mse_plus_corr"], marker="o", ms=3, label=f"level={lvl}")
    ax.set_xlabel("threshold_factor")
    ax.set_ylabel("R3 = delta_SNR - MSE + corr")
    ax.set_title("R3 vs. threshold, one line per decomposition level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1): heatmap of R3 over threshold x level
    ax = axes[0, 1]
    pivot = df.pivot(index="level", columns="threshold", values="R3_snr_minus_mse_plus_corr")
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis",
                    extent=[THRESHOLDS.min(), THRESHOLDS.max(), min(LEVELS), max(LEVELS)])
    ax.set_xlabel("threshold_factor")
    ax.set_ylabel("level")
    ax.set_title("R3 reward landscape (threshold x level)")
    plt.colorbar(im, ax=ax, label="R3 reward")
    best_idx = df["R3_snr_minus_mse_plus_corr"].idxmax()
    best = df.loc[best_idx]
    ax.scatter([best["threshold"]], [best["level"]], color="red", marker="*", s=150,
               label=f"best (thr={best['threshold']:.2f}, lvl={int(best['level'])})")
    ax.legend(fontsize=8)

    # (1,0): MSE heatmap over threshold x level, to see where distortion creeps back in
    ax = axes[1, 0]
    pivot_mse = df.pivot(index="level", columns="threshold", values="mse")
    im2 = ax.imshow(pivot_mse.values, aspect="auto", origin="lower", cmap="magma",
                     extent=[THRESHOLDS.min(), THRESHOLDS.max(), min(LEVELS), max(LEVELS)])
    ax.set_xlabel("threshold_factor")
    ax.set_ylabel("level")
    ax.set_title("MSE landscape (threshold x level) -- lower is better")
    plt.colorbar(im2, ax=ax, label="MSE")

    # (1,1): example waveform at the best (threshold, level) found for R3
    ax = axes[1, 1]
    filtered = apply_filter(noisy, best["threshold"], level=int(best["level"]))
    ax.plot(t, noisy, label="noisy", alpha=0.4)
    ax.plot(t, clean, label="clean", lw=2)
    ax.plot(t, filtered, label=f"filtered (thr={best['threshold']:.2f}, lvl={int(best['level'])})", lw=1.5)
    ax.set_xlim(0, 0.05)
    ax.set_xlabel("time (s)")
    ax.set_title("Example waveform at best (threshold, level) [zoomed]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return best


# ---------------------------------------------------------------------------
# Phase A: baseline comparison (now over both threshold AND level)
# ---------------------------------------------------------------------------
def run_phase_a(df, clean, noisy):
    raw_metrics = compute_metrics(clean, noisy, noisy)

    rows = [{
        "condition": "raw_noisy (no filtering)",
        "threshold": 0.0,
        "level": None,
        "delta_snr": raw_metrics["delta_snr"],
        "mse": raw_metrics["mse"],
        "correlation": raw_metrics["correlation"],
    }]

    reward_cols = ["R1_snr_only", "R2_snr_minus_mse", "R3_snr_minus_mse_plus_corr"]
    labels = {
        "R1_snr_only": "SNR-only reward, best (threshold, level)",
        "R2_snr_minus_mse": "SNR - MSE reward, best (threshold, level)",
        "R3_snr_minus_mse_plus_corr": "SNR - MSE + corr reward, best (threshold, level)",
    }
    for col in reward_cols:
        best_idx = df[col].idxmax()
        best_row = df.loc[best_idx]
        rows.append({
            "condition": labels[col],
            "threshold": best_row["threshold"],
            "level": int(best_row["level"]),
            "delta_snr": best_row["delta_snr"],
            "mse": best_row["mse"],
            "correlation": best_row["correlation"],
        })

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv("phaseA_baseline_comparison_2d.csv", index=False)
    return baseline_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    t, clean, noisy, df = run_phase0()
    best = plot_phase0(df, t, clean, noisy)
    baseline_df = run_phase_a(df, clean, noisy)

    print("=" * 70)
    print("PHASE 0: 2D reward landscape sweep saved to")
    print(f"  {OUT_DIR / 'phase0_reward_landscape_2d.csv'}")
    print(f"  {OUT_DIR / 'phase0_reward_landscape_2d.png'}")
    print()
    print(f"Best (threshold, level) for R3: threshold={best['threshold']:.2f}, level={int(best['level'])}")
    print()
    print("PHASE A: baseline comparison")
    print("=" * 70)
    print(baseline_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Saved to {OUT_DIR / 'phaseA_baseline_comparison_2d.csv'}")

    # Check whether best level is consistent across the threshold sweep -- if it
    # is NOT, that's evidence the agent needs to control level as a separate
    # action dimension rather than using a fixed level.
    best_level_per_threshold = df.loc[df.groupby("threshold")["R3_snr_minus_mse_plus_corr"].idxmax()]
    print()
    print("Best level at each threshold (check if this is constant or varies):")
    print(best_level_per_threshold[["threshold", "level"]].to_string(index=False))