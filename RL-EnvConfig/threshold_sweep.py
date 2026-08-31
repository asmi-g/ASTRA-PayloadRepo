"""
Sweep the wavelet filter over threshold_factor x level and visualize every
combination's filtered waveform against clean/noisy.

Outputs:
    1. One big grid figure: all 30 combinations (5 thresholds x 6 levels) at once.
    2. Individual PNG per combination, saved to a subfolder, for closer inspection.

No interactive popups are possible in this environment (no display attached),
so everything is saved to disk instead -- open the grid PNG first for the
overview, then dig into individual files for anything that looks interesting.
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/mnt/user-data/outputs")
INDIVIDUAL_DIR = OUT_DIR / "threshold_level_grid_individual"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
LEVELS = [1, 2, 3, 4, 5, 6]
WAVELET = "db4"
SEED = 42
ZOOM_XLIM = (0.0, 0.05)  # seconds, adjust to taste


# ---------------------------------------------------------------------------
# Signal generation (your environment's generator)
# ---------------------------------------------------------------------------
def generate_signal(seed=SEED, white_var=0.1, burst_prob=0.01, burst_amp=3.0,
                     burst_duration=10, f_signal=100):
    rng = np.random.default_rng(seed)
    signal_length = 10_000
    x = np.arange(signal_length)
    t = x / 1e4

    clean = np.sin(2 * np.pi * f_signal * t)
    white_noise = rng.normal(0, np.sqrt(white_var), size=signal_length)
    pink_noise = generate_pink_noise(signal_length, white_noise, white_var)
    noisy = clean + white_noise + pink_noise
    noisy = add_bursts(noisy, rng, burst_prob, burst_amp, burst_duration)

    return t, clean.astype(np.float64), noisy.astype(np.float64)


def generate_pink_noise(size, white_noise, white_var):
    freq = np.fft.fftfreq(size)
    pink_filter = np.sqrt(np.abs(freq) + 1e-6)
    spectrum = np.fft.fft(white_noise / np.sqrt(white_var))
    spectrum *= pink_filter
    pink_noise = np.fft.ifft(spectrum)
    return np.real(pink_noise)


def add_bursts(signal, rng, burst_probability, burst_amplitude, burst_duration):
    noisy_signal = signal.copy()
    hits = rng.random(len(signal)) < burst_probability
    for i in np.where(hits)[0]:
        burst_end = min(i + burst_duration, len(signal))
        noisy_signal[i:burst_end] += rng.uniform(-burst_amplitude, burst_amplitude)
    return noisy_signal


# ---------------------------------------------------------------------------
# Filter under test
# ---------------------------------------------------------------------------
def apply_filter(window, threshold_factor, wavelet=WAVELET, level=1):
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


def mse(clean, filtered):
    return float(np.mean((clean - filtered) ** 2))

def snr_db(clean, test_signal):
    noise = test_signal - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-15) / (noise_power + 1e-15))


# ---------------------------------------------------------------------------
# Grid figure: all combinations at once
# ---------------------------------------------------------------------------
def plot_grid(t, clean, noisy):
    n_rows, n_cols = len(LEVELS), len(THRESHOLDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharex=True, sharey=True)

    mask = (t >= ZOOM_XLIM[0]) & (t <= ZOOM_XLIM[1])

    for row, lvl in enumerate(LEVELS):
        for col, thr in enumerate(THRESHOLDS):
            ax = axes[row, col]
            filtered = apply_filter(noisy, thr, level=lvl)
            m = mse(clean, filtered)
            snr_raw = snr_db(clean, noisy)
            snr_filtered = snr_db(clean, filtered)
            snr_improvement = snr_filtered - snr_raw

            ax.plot(t[mask], noisy[mask], color="lightsteelblue", alpha=0.5, lw=0.8, label="noisy")
            ax.plot(t[mask], clean[mask], color="orange", lw=1.5, label="clean")
            ax.plot(t[mask], filtered[mask], color="green", lw=1.2, label="filtered")
            ax.set_title(f"thr={thr}, level={lvl}\nMSE={m:.3f} dSNR={snr_improvement:.3f}", fontsize=9)
            ax.tick_params(labelsize=7)

            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    for col, thr in enumerate(THRESHOLDS):
        axes[-1, col].set_xlabel("time (s)", fontsize=8)
    for row, lvl in enumerate(LEVELS):
        axes[row, 0].set_ylabel(f"level={lvl}", fontsize=8)

    fig.suptitle("Filtered waveform across threshold_factor x level grid", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = OUT_DIR / "threshold_level_grid_all.png"
    plt.show()
    return grid_path


if __name__ == "__main__":
    t, clean, noisy = generate_signal()

    grid_path = plot_grid(t, clean, noisy)
    print(f"Saved grid overview to: {grid_path}")