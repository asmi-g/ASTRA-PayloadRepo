import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("Data/flight_signal_2.csv")

clean = pd.to_numeric(df["TX Magnitude"], errors="coerce").values
noisy = pd.to_numeric(df["RX Magnitude"], errors="coerce").values

clean = np.nan_to_num(clean)
noisy = np.nan_to_num(noisy)

n = min(len(clean), len(noisy))
clean = clean[:n]
noisy = noisy[:n]

# =====================================================
# 2. EXTRACT TRUE NOISE
# =====================================================

noise = noisy - clean
noise_bias = np.mean(noise)
noise = noise - noise_bias

print("bias removed:", noise_bias)
print("corrected noise std:", np.std(noise))


print("\nNoise statistics")
print("mean:", np.mean(noise))
print("std:", np.std(noise))
print("energy:", np.mean(noise**2))

# =====================================================
# 3. ANALYZE SPECTRAL PROFILE
# =====================================================

f_clean, psd_clean = welch(clean)
f_noise, psd_noise = welch(noise)

# Normalize PSD for shaping filter
psd_noise = psd_noise / np.max(psd_noise)

# =====================================================
# 4. HELPER: FFT AUTOCORRELATION
# =====================================================

def autocorr(x, lag=200):
    x = x - np.mean(x)
    fft = np.fft.fft(x, n=2*len(x))
    power = fft * np.conjugate(fft)
    result = np.fft.ifft(power).real[:len(x)]
    result /= result[0]
    return result[:lag]

# =====================================================
# 5. SYNTHETIC SIGNAL GENERATOR
# =====================================================

def generate_synthetic_signal(n_samples):
    """
    Generates clean + noise matching flight signal properties
    """

    # ---- Generate base clean signal using spectral shaping ----
    white = np.random.randn(n_samples)

    # Shape using clean spectrum
    fft_white = np.fft.rfft(white)
    freqs_fft = np.fft.rfftfreq(n_samples)

    # interpolate PSD to FFT resolution
    target_mag = np.sqrt(np.interp(freqs_fft, f_clean, psd_clean))
    shaped_fft = fft_white * target_mag

    synthetic_clean = np.fft.irfft(shaped_fft, n=n_samples)

    # Match amplitude distribution
    synthetic_clean = match_distribution(synthetic_clean, clean)

    # ---- Generate noise using learned spectrum ----
    white_noise = np.random.randn(n_samples)
    fft_noise = np.fft.rfft(white_noise)
    target_noise_mag = np.sqrt(np.interp(freqs_fft, f_noise, psd_noise))
    shaped_noise_fft = fft_noise * target_noise_mag

    synthetic_noise = np.fft.irfft(shaped_noise_fft, n=n_samples)

    synthetic_noise = match_distribution(synthetic_noise, noise)

    synthetic_noisy = synthetic_clean + synthetic_noise + noise_bias

    return synthetic_clean, synthetic_noisy

# =====================================================
# 6. DISTRIBUTION MATCHING
# =====================================================

def match_distribution(source, target):
    """
    Histogram matching via rank mapping
    """
    sorted_src = np.sort(source)
    sorted_tgt = np.sort(target)

    ranks = np.argsort(np.argsort(source))
    mapped = sorted_tgt[ranks % len(sorted_tgt)]

    return mapped

# =====================================================
# 7. VISUAL VALIDATION
# =====================================================

def compare_real_vs_synthetic():
    syn_clean, syn_noisy = generate_synthetic_signal(n)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(clean[:400],label = "flight")
    plt.plot(syn_clean[:400], label = "synthetic")
    plt.title("clean")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(noisy[:400],label = "flight")
    plt.plot(syn_noisy[:400], label = "synthetic")
    plt.title("noisy")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(12, 8))

    # Spectrum comparison
    plt.subplot(2, 2, 1)
    _, p_real = welch(noisy)
    _, p_syn = welch(syn_noisy)
    plt.semilogy(p_real, label="Real")
    plt.semilogy(p_syn, label="Synthetic")
    plt.title("Noisy Spectrum")
    plt.legend()

    # Noise distribution
    plt.subplot(2, 2, 2)
    plt.hist(noise, bins=100, alpha=0.5, label="Real")
    plt.hist(syn_noisy - syn_clean, bins=100, alpha=0.5, label="Synthetic")
    plt.title("Noise Distribution")
    plt.legend()

    # Clean autocorrelation
    plt.subplot(2, 2, 3)
    plt.plot(autocorr(clean), label="Real")
    plt.plot(autocorr(syn_clean), label="Synthetic")
    plt.title("Clean Autocorrelation")
    plt.legend()

    # Noise autocorrelation
    plt.subplot(2, 2, 4)
    plt.plot(autocorr(noise), label="Real")
    plt.plot(autocorr(syn_noisy - syn_clean), label="Synthetic")
    plt.title("Noise Autocorrelation")
    plt.legend()

    plt.tight_layout()
    plt.show()

compare_real_vs_synthetic()
