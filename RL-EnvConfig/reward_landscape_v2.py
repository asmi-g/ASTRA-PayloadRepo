"""
Step 1: Validate that level=5 is robustly the best decomposition level across
         several DIFFERENT noisy signals (varied seed + noise parameters), not
         just the one signal from the earlier 2D sweep.

Step 2: With level fixed at 5, run the 1D reward-formula comparison
         (R1 = delta_SNR, R2 = delta_SNR - MSE, R3 = delta_SNR - MSE + corr)
         across those same varied signals, so the comparison isn't a fluke of
         one random draw either.
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

THRESHOLDS = np.linspace(0.0, 2.5, 26)
LEVELS = [1, 2, 3, 4, 5, 6]
FIXED_LEVEL = 5
WAVELET = "db4"

# Signal variation grid: each entry is a distinct noise/signal configuration.
# Add/remove entries to widen or narrow the robustness check.
SIGNAL_CONFIGS = [
    dict(seed=1,  white_var=0.1,  burst_prob=0.01,  burst_amp=3.0, f_signal=100_000),
    dict(seed=2,  white_var=0.1,  burst_prob=0.01,  burst_amp=3.0, f_signal=100),
    dict(seed=3,  white_var=0.05, burst_prob=0.01,  burst_amp=3.0, f_signal=100),  # less white noise
    dict(seed=4,  white_var=0.2,  burst_prob=0.01,  burst_amp=3.0, f_signal=100_000),  # more white noise
    dict(seed=5,  white_var=0.1,  burst_prob=0.02,  burst_amp=3.0, f_signal=100),  # more frequent bursts
    dict(seed=6,  white_var=0.1,  burst_prob=0.01,  burst_amp=5.0, f_signal=100_000),  # bigger bursts
    dict(seed=7,  white_var=0.1,  burst_prob=0.01,  burst_amp=3.0, f_signal=50),   # lower signal freq
    dict(seed=8,  white_var=0.1,  burst_prob=0.01,  burst_amp=3.0, f_signal=200),  # higher signal freq
]


# ---------------------------------------------------------------------------
# Signal generation (parametrized version)
# ---------------------------------------------------------------------------
def generate_signal(seed, white_var=0.1, burst_prob=0.01, burst_amp=3.0,
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
# Filter
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
    return {"snr_raw": snr_raw, "snr_filtered": snr_filt, "delta_snr": delta_snr,
            "mse": mse, "correlation": corr}


def compute_rewards(m):
    r1 = m["delta_snr"]
    r2 = m["delta_snr"] - m["mse"]
    r3 = m["delta_snr"] - m["mse"] + m["correlation"]
    return {"R1_snr_only": r1, "R2_snr_minus_mse": r2, "R3_snr_minus_mse_plus_corr": r3}


# ---------------------------------------------------------------------------
# Step 1: robustness check -- is level 5 still best across varied signals?
# ---------------------------------------------------------------------------
def check_level_robustness():
    rows = []
    for cfg in SIGNAL_CONFIGS:
        t, clean, noisy = generate_signal(cfg["seed"], cfg["white_var"],
                                           cfg["burst_prob"], cfg["burst_amp"],
                                           f_signal=cfg["f_signal"])
        best_level_per_thr = []
        for thr in THRESHOLDS:
            if thr == 0.0:
                continue  # threshold=0 is a degenerate no-op case, skip for level comparison
            best_lvl, best_r3 = None, -np.inf
            for lvl in LEVELS:
                filtered = apply_filter(noisy, thr, level=lvl)
                m = compute_metrics(clean, noisy, filtered)
                r3 = m["delta_snr"] - m["mse"] + m["correlation"]
                if r3 > best_r3:
                    best_r3, best_lvl = r3, lvl
            best_level_per_thr.append(best_lvl)
        winning_level = pd.Series(best_level_per_thr).mode().iloc[0]
        rows.append({**cfg, "modal_best_level": winning_level,
                     "level5_win_rate": np.mean(np.array(best_level_per_thr) == 5)})
    df = pd.DataFrame(rows)
    df.to_csv("level_robustness_check.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Step 2: 1D reward comparison at fixed level=5, across varied signals
# ---------------------------------------------------------------------------
def run_reward_comparison():
    all_rows = []
    for cfg in SIGNAL_CONFIGS:
        t, clean, noisy = generate_signal(cfg["seed"], cfg["white_var"],
                                           cfg["burst_prob"], cfg["burst_amp"],
                                           f_signal=cfg["f_signal"])
        for thr in THRESHOLDS:
            filtered = apply_filter(noisy, thr, level=FIXED_LEVEL)
            m = compute_metrics(clean, noisy, filtered)
            r = compute_rewards(m)
            all_rows.append({"seed": cfg["seed"], "threshold": thr, **m, **r})
    df = pd.DataFrame(all_rows)
    df.to_csv(F"phase0_1d_level{FIXED_LEVEL}_multisignal.csv", index=False)
    return df


def plot_reward_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    reward_cols = ["R1_snr_only", "R2_snr_minus_mse", "R3_snr_minus_mse_plus_corr"]
    labels = ["R1 = delta_SNR", "R2 = delta_SNR - MSE", "R3 = delta_SNR - MSE + corr"]

    # Mean +/- std reward vs threshold, averaged across all signal configs
    ax = axes[0]
    for col, lab in zip(reward_cols, labels):
        grouped = df.groupby("threshold")[col].agg(["mean", "std"])
        ax.plot(grouped.index, grouped["mean"], label=lab, marker="o", ms=3)
        ax.fill_between(grouped.index, grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"], alpha=0.15)
    ax.set_xlabel(f"threshold_factor (level fixed = {FIXED_LEVEL})")
    ax.set_ylabel("reward (mean +/- std across signals)")
    ax.set_title(f"Reward comparison across {len(SIGNAL_CONFIGS)} varied signals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Where does each reward's argmax threshold land, per signal? (spread = instability)
    ax = axes[1]
    for i, (col, lab) in enumerate(zip(reward_cols, labels)):
        best_thrs = df.loc[df.groupby("seed")[col].idxmax(), "threshold"]
        ax.scatter([i] * len(best_thrs), best_thrs, alpha=0.7, label=lab)
    ax.set_xticks(range(len(reward_cols)))
    ax.set_xticklabels(["R1", "R2", "R3"])
    ax.set_ylabel("best threshold per signal")
    ax.set_title("Spread of optimal threshold across signals, per reward")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Step 1: checking level=5 robustness across varied signals...")
    rob_df = check_level_robustness()
    print(rob_df.to_string(index=False))

    print("Step 2: running 1D reward comparison at level=5 across the same signals...")
    reward_df = run_reward_comparison()
    plot_reward_comparison(reward_df)
