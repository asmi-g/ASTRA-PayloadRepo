# noise_extractor_v4.py
# Reads pre-built Clean/Noisy columns from flight_signal_1_clean_noisy.csv.
#
# v4 change: SEGMENT 0 IS EXCLUDED. flight_signal_1 is 34 stitched captures
# (~500k samples each). Segment 0 is an anomalous startup acquisition -- its
# RX gain is 5x lower than every other segment (RX Magnitude 0.00787 vs
# 0.03937) and it carries a huge transient (segment excess kurtosis ~6500 vs
# ~10-30 for the rest). v3 fit its AR model on samples [:50000] and stored a
# residual snippet from [:5000] -- both entirely inside segment 0 -- so the
# old model was calibrated to the wrong receiver gain and an unrepresentative
# spike statistic. v4 fits the AR / colour / snippet on a representative
# mid-flight segment and computes the global stats + rms_profile over all
# segments EXCEPT segment 0.

import numpy as np
import pandas as pd
import pickle
from scipy.signal import welch
from statsmodels.tsa.ar_model import AutoReg

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
CSV_PATH = r"c:\Users\imanq\Documents\Programs\GitHub\ASTRA-GeneralRepo\Data\flight_signal_1_clean_noisy.csv"
OUTPUT_MODEL_PATH = r"c:\Users\imanq\Documents\Programs\GitHub\ASTRA-GeneralRepo\Data\noise_model_fs1.pkl"
AR_LAGS = 50
SKIP_SEGMENTS = 1          # exclude segment 0 (5x-low-gain startup outlier)
NOMINAL_SEG_LEN = 500_000  # fallback if Time-gap detection fails
CHAR_FRACTION = 0.5        # characterise the segment nearest this fraction through the usable range
CHAR_SAMPLES = 50_000      # samples used for AR fit / PSD / autocorr
SNIPPET_SAMPLES = 5_000    # raw residual excerpt stored for reference/diagnostics

# -----------------------------------------------------------------------
# STEP 1: Load (only the columns needed, to keep memory down)
# -----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, usecols=["Time", "Noisy Signal", "Clean Signal"])
t = df["Time"].to_numpy(dtype=np.float64)
clean = df["Clean Signal"].to_numpy(dtype=np.float64)
noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
noise_residual = noisy - clean
N = len(noise_residual)
print(f"Loaded {N} samples from {CSV_PATH}")

# -----------------------------------------------------------------------
# STEP 2: Segment boundaries (Time gaps between stitched captures)
# -----------------------------------------------------------------------
gap_idx = np.where(np.diff(t) > 0.5)[0] + 1
bounds = np.r_[0, gap_idx, N]
segs = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
if len(segs) < 2:  # gap detection failed -> fall back to fixed slicing
    segs = [(a, min(a + NOMINAL_SEG_LEN, N)) for a in range(0, N, NOMINAL_SEG_LEN)]
print(f"segments: {len(segs)}  (sizes {sorted(set(b - a for a, b in segs))[:5]}...)")

usable_segs = segs[SKIP_SEGMENTS:]
usable_idx = np.concatenate([np.arange(a, b) for a, b in usable_segs])
print(f"excluding first {SKIP_SEGMENTS} segment(s); {len(usable_segs)} usable, "
      f"{len(usable_idx)} samples")

# -----------------------------------------------------------------------
# STEP 3: Characterisation segment (representative, mid-flight)
# -----------------------------------------------------------------------
char_seg_pos = int(round(CHAR_FRACTION * (len(usable_segs) - 1)))
char_seg_index = SKIP_SEGMENTS + char_seg_pos
c_a, c_b = segs[char_seg_index]
char_noise = noise_residual[c_a:c_b][:CHAR_SAMPLES]
print(f"characterisation segment index {char_seg_index}  span [{c_a}:{c_b}]  "
      f"using first {len(char_noise)} samples")

freqs, psd = welch(char_noise, nperseg=1024)
psd_flatness = float(np.std(psd) / (np.mean(psd) + 1e-10))

fft_n = np.fft.rfft(char_noise, n=2 * len(char_noise))
autocorr = np.fft.irfft(fft_n * np.conj(fft_n))[:len(char_noise)]
autocorr /= autocorr[0]
print(f"PSD flatness: {psd_flatness:.4f} | autocorr lag-1: {autocorr[1]:.4f}")

# log-log PSD slope over 1 kHz-300 kHz (0 = white, -1 = pink, -2 = brown)
band = (freqs > 1e3 / (1e6 / 2)) & (freqs < 3e5 / (1e6 / 2))  # welch freqs are normalised (fs=1)
if band.sum() < 5:
    band = (freqs > 0.01) & (freqs < 0.45)
psd_slope = float(np.polyfit(np.log10(freqs[band]), np.log10(psd[band]), 1)[0])
print(f"PSD log-log slope (char seg): {psd_slope:.3f}")

# -----------------------------------------------------------------------
# STEP 4: AR fit on the characterisation segment
# -----------------------------------------------------------------------
ar_model = AutoReg(char_noise, lags=AR_LAGS, old_names=False).fit()
ar_params = ar_model.params
ar_resid_std = float(np.std(ar_model.resid))
ar_resid_excess_kurt = float(
    ((ar_model.resid - ar_model.resid.mean()) ** 4).mean() / np.std(ar_model.resid) ** 4 - 3.0)
print(f"AR({AR_LAGS}) fitted. innovation std {ar_resid_std:.6f}  "
      f"excess kurtosis {ar_resid_excess_kurt:.1f}  sum(phi) {ar_params[1:].sum():.4f}")

# -----------------------------------------------------------------------
# STEP 5: Global stats + RMS envelope profile -- usable segments only
# -----------------------------------------------------------------------
noise_usable = noise_residual[usable_idx]
clean_usable = clean[usable_idx]
noise_std = float(np.std(noise_usable))
noise_mean = float(np.mean(noise_usable))
clean_std = float(np.std(clean_usable))
global_excess_kurt = float(
    ((noise_usable - noise_mean) ** 4).mean() / noise_std ** 4 - 3.0)

window_size = 1000
rms_profile = np.concatenate([
    np.sqrt(np.mean(
        noisy[a:a + (b - a) // window_size * window_size]
        .reshape(-1, window_size) ** 2, axis=1))
    for a, b in usable_segs if (b - a) >= window_size
])
print(f"noise_std {noise_std:.5f}  noise_mean {noise_mean:.5f}  clean_std {clean_std:.6f}  "
      f"severity ratio {noise_std / clean_std:.2f}")
print(f"global excess kurtosis (usable) {global_excess_kurt:.1f}")
print(f"rms_profile: {len(rms_profile)} pts  min/mean/max "
      f"{rms_profile.min():.4f}/{rms_profile.mean():.4f}/{rms_profile.max():.4f}  "
      f"max/min {rms_profile.max() / rms_profile.min():.1f}")

# per-segment std spread (level non-stationarity across the flight)
seg_std = np.array([np.std(noise_residual[a:b]) for a, b in usable_segs])
print(f"per-usable-segment std: min {seg_std.min():.4f}  max {seg_std.max():.4f}  "
      f"ratio {seg_std.max() / seg_std.min():.2f}")

# -----------------------------------------------------------------------
# STEP 6: Save
# -----------------------------------------------------------------------
noise_model = {
    "ar_params": ar_params,
    "ar_resid_std": ar_resid_std,
    "ar_resid_excess_kurt": ar_resid_excess_kurt,
    "ar_lags": AR_LAGS,
    "noise_std": noise_std,
    "noise_mean": noise_mean,
    "clean_std": clean_std,
    "global_excess_kurt": global_excess_kurt,
    "rms_profile": rms_profile,
    "psd_flatness": psd_flatness,
    "psd_slope": psd_slope,
    "autocorr_lag1": float(autocorr[1]),
    "noise_residual_sample": noise_residual[c_a:c_a + SNIPPET_SAMPLES],
    # provenance
    "n_segments": len(segs),
    "excluded_segments": list(range(SKIP_SEGMENTS)),
    "char_segment_index": char_seg_index,
    "per_segment_std_ratio": float(seg_std.max() / seg_std.min()),
    "extractor_version": 4,
}
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(noise_model, f)
print(f"\nNoise model (v4, segment-0 excluded) saved to {OUTPUT_MODEL_PATH}")
