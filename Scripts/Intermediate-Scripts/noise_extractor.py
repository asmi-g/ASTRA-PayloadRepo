# noise_extractor_v3.py
# Reads pre-built Clean/Noisy columns directly from
# flight_signal_1_clean_noisy.csv — no IQ construction, no alignment
# step needed since the CSV already contains the aligned pair.

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import welch
from statsmodels.tsa.ar_model import AutoReg

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
CSV_PATH = r"c:\Users\imanq\Documents\Programs\GitHub\ASTRA-GeneralRepo\Data\flight_signal_1_clean_noisy.csv"
OUTPUT_MODEL_PATH = r"c:\Users\imanq\Documents\Programs\GitHub\ASTRA-GeneralRepo\Data\noise_model_fs1.pkl"
AR_LAGS = 50
PLOT = True

# -----------------------------------------------------------------------
# STEP 1: Load — columns are already Index, Time, Noisy Signal, Clean Signal
# -----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
clean = df['Clean Signal'].to_numpy(dtype=np.float64)
noisy = df['Noisy Signal'].to_numpy(dtype=np.float64)
print(f"Loaded {len(clean)} samples from {CSV_PATH}")

# -----------------------------------------------------------------------
# STEP 2: Noise residual — direct difference, no alignment needed
# -----------------------------------------------------------------------
noise_residual = noisy - clean
print(f"\nNoise residual stats:")
print(f"  mean: {np.mean(noise_residual):.6f}  std: {np.std(noise_residual):.6f}")
print(f"  min:  {np.min(noise_residual):.6f}   max: {np.max(noise_residual):.6f}")

# -----------------------------------------------------------------------
# STEP 3: Characterize noise color
# -----------------------------------------------------------------------
CHAR_SAMPLES = min(50_000, len(noise_residual))
noise_char = noise_residual[:CHAR_SAMPLES]
freqs, psd = welch(noise_char, nperseg=1024)

fft_noise = np.fft.rfft(noise_char, n=2 * len(noise_char))
autocorr = np.fft.irfft(fft_noise * np.conj(fft_noise))[:len(noise_char)]
autocorr /= autocorr[0]

psd_flatness = np.std(psd) / (np.mean(psd) + 1e-10)
print(f"\nPSD flatness: {psd_flatness:.4f}  |  Autocorr lag-1: {autocorr[1]:.4f}")

# -----------------------------------------------------------------------
# STEP 4: Fit AR model
# -----------------------------------------------------------------------
AR_FIT_SAMPLES = min(50_000, len(noise_residual))
noise_ar = noise_residual[:AR_FIT_SAMPLES]
ar_model = AutoReg(noise_ar, lags=AR_LAGS, old_names=False).fit()
ar_params = ar_model.params
ar_resid_std = np.std(ar_model.resid)
print(f"\nAR({AR_LAGS}) fitted. Innovation std: {ar_resid_std:.6f}")

# -----------------------------------------------------------------------
# STEP 5: RMS attenuation/envelope profile (on noisy signal)
# -----------------------------------------------------------------------
window_size = 1000
n_windows = len(noisy) // window_size
rms_profile = np.array([
    np.sqrt(np.mean(noisy[w*window_size:(w+1)*window_size] ** 2))
    for w in range(n_windows)
])
print(f"\nRMS profile — min/mean/max/std: "
      f"{rms_profile.min():.4f}/{rms_profile.mean():.4f}/"
      f"{rms_profile.max():.4f}/{rms_profile.std():.4f}")

# -----------------------------------------------------------------------
# STEP 6: Save model
# -----------------------------------------------------------------------
noise_model = {
    "ar_params": ar_params,
    "ar_resid_std": ar_resid_std,
    "ar_lags": AR_LAGS,
    "noise_std": np.std(noise_residual),
    "noise_mean": np.mean(noise_residual),
    "rms_profile": rms_profile,
    "psd_flatness": psd_flatness,
    "noise_residual_sample": noise_residual[:5000],
    # clean_std: lets a synthetic training env rescale bootstrapped noise to
    # match the REAL noise-to-signal severity ratio (noise_std/clean_std),
    # not just borrow noise's raw captured magnitude against an unrelated
    # (e.g. unit-amplitude) synthetic clean signal.
    "clean_std": np.std(clean),
}
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump(noise_model, f)
print(f"\nNoise model saved to {OUTPUT_MODEL_PATH}")

# -----------------------------------------------------------------------
# STEP 7: Diagnostics
# -----------------------------------------------------------------------
if PLOT:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Noise Extractor v3 — Diagnostics (pre-aligned CSV)")

    t = np.arange(min(5000, len(noise_residual)))
    axes[0, 0].plot(t, noise_residual[:len(t)], color='orange')
    axes[0, 0].set_title("Noise Residual (first 5000)")

    axes[0, 1].hist(noise_residual, bins=100, density=True)
    axes[0, 1].set_title("Noise Distribution — should be ~unimodal")

    axes[1, 0].semilogy(freqs, psd)
    axes[1, 0].set_title(f"Noise PSD (flatness={psd_flatness:.3f})")

    axes[1, 1].plot(autocorr[:200])
    axes[1, 1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1, 1].set_title("Noise Autocorrelation")

    plt.tight_layout()
    plt.savefig("Data/noise_extractor_fs1_diagnostics.png", dpi=120)
    plt.show()