import numpy as np
import pandas as pd
import pywt
import ast
import matplotlib.pyplot as plt

# ---------- LOAD DATA ----------
ml = pd.read_csv("C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/ml_results.csv")
df = pd.read_csv("Data/flight_signal_2.csv")

print("threshold min:", np.min(ml["threshold_factor"]))
print("threshold max:", np.max(ml["threshold_factor"]))
print("threshold mean:", np.mean(ml["threshold_factor"]))

clean = df["TX Magnitude"].values.astype(np.float64)
noisy = df["RX Magnitude"].values.astype(np.float64)

# normalize exactly like environment
scale = np.max(np.abs(clean))
if scale > 0:
    clean = clean / scale
    noisy = noisy / scale

# ---------- FILTER FUNCTION ----------
def apply_filter(signal, wavelet='db4', level=1, threshold_factor=1.0):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

# ---------- SNR ----------
def calculate_snr(clean, signal):
    noise = signal - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

# ---------- REPLAY FILTERING CORRECTLY ----------
print("Reconstructing filtered signal using windows...")

filtered_full = noisy.copy()

for i, row in ml.iterrows():
    if i%100 == 0:
        print(f"applying filtering {i}...")

    # parse window string "(start, end)"
    start, end = ast.literal_eval(row["window"])

    tf = float(row["threshold_factor"])

    segment = filtered_full[start:end]
    filtered_segment = apply_filter(segment, threshold_factor=tf)

    # align length after wavelet reconstruction
    filtered_segment = filtered_segment[:len(segment)]

    filtered_full[start:end] = filtered_segment

# ---------- METRICS ----------
snr_noisy = calculate_snr(clean, noisy)
snr_filtered = calculate_snr(clean, filtered_full)

percent_snr_improvement = ((snr_filtered - snr_noisy) / abs(snr_noisy)) * 100

mse_noisy = np.mean((clean - noisy)**2)
mse_filtered = np.mean((clean - filtered_full)**2)
percent_mse_reduction = ((mse_noisy - mse_filtered) / mse_noisy) * 100

corr_noisy = np.corrcoef(clean, noisy)[0,1]
corr_filtered = np.corrcoef(clean, filtered_full)[0,1]
corr_gain = corr_filtered - corr_noisy

print("\n===== MODEL PERFORMANCE =====")
print("average snr improvement:", np.mean(ml["snr_improvement"]))
print("SNR noisy:", snr_noisy)
print("SNR filtered:", snr_filtered)
print("Percent SNR improvement:", percent_snr_improvement)

print("\nMSE noisy:", mse_noisy)
print("MSE filtered:", mse_filtered)
print("Percent MSE reduction:", percent_mse_reduction)

print("\nCorrelation noisy:", corr_noisy)
print("Correlation filtered:", corr_filtered)
print("Correlation gain:", corr_gain)
