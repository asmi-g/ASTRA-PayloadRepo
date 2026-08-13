import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from scipy.signal import welch



# ---------- Load dataset ----------
df = pd.read_csv("Data/flight_signal_1.csv")
df = df.head(30000)

flight_clean = df["TX Magnitude"].values.astype(float)
flight_noisy = df["RX Magnitude"].values.astype(float)

n = len(df)

def generate_pink_noise(size, noise_power=0.1):
    white_noise = np.random.normal(0, np.sqrt(noise_power), size)

    # Frequency-domain pink noise shaping (1/f)
    freq = np.fft.fftfreq(size)
    pink_filter = np.sqrt(np.abs(freq))
    pink_filter[0] = 0  # avoid DC blow-up

    spectrum = np.fft.fft(white_noise)
    spectrum *= pink_filter

    pink_noise = np.fft.ifft(spectrum)
    return np.real(pink_noise)


def add_burst(signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
    noisy_signal = signal.copy()
    i = 0
    while i < len(signal):
        if np.random.rand() < burst_probability:
            end = min(i + burst_duration, len(signal))
            noisy_signal[i:end] += np.random.uniform(
                -burst_amplitude, burst_amplitude
            )
            i = end
        else:
            i += 1
    return noisy_signal

# ---------- Generate training-style data ----------
t = np.linspace(0, 2*np.pi, n)
training_clean = np.sin(t)
training_noisy = training_clean + np.random.normal(0, 0.3, size=n)
# pink_noise = generate_pink_noise(n, noise_power=0.1)
# pink_burst_signal = training_noisy + pink_noise
# training_noisy = add_burst(
#     pink_burst_signal,
#     burst_probability=0.005,
#     burst_amplitude=2.0,
#     burst_duration=15
# )

scale = np.max(np.abs(training_clean))
if scale > 0:
    training_clean /= scale
    training_noisy /= scale

flight_noise = flight_noisy - flight_clean
training_noise = training_noisy - training_clean
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(flight_noise,label = "flight")
plt.plot(training_noise, label = "training")
plt.title("noise")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(flight_clean,label = "flight")
plt.plot(training_clean, label = "training")
plt.title("clean)")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# ---------- Print basic statistics ----------
def print_stats(name, x):
    print(f"\n{name}")
    print("mean:", np.mean(x))
    print("std:", np.std(x))
    print("energy:", np.mean(x**2))

print("Signal Stats:")
print_stats("Training Clean", training_clean)
print_stats("Flight Clean", flight_clean)
print_stats("Training Noisy", training_noisy)
print_stats("Flight Noisy", flight_noisy)

# ---------- Autocorrelation ----------
def autocorr(x, lag=200):
    x = x - np.mean(x)
    n = len(x)

    # zero-pad to avoid circular correlation
    fft = np.fft.fft(x, n=2*n)
    power = fft * np.conjugate(fft)
    result = np.fft.ifft(power).real[:n]

    result /= result[0]  # normalize
    return result[:lag]

print("training clean signal range:", training_clean.min(), training_clean.max())
print("flight clean signal range:", flight_clean.min(), flight_clean.max())
print("training noisy signal range:", training_noisy.min(), training_noisy.max())
print("flight noisy signal range:", flight_noisy.min(), flight_noisy.max())

print("creating plot...")
# ---------- Create single figure ----------
plt.figure(figsize=(12, 8))

# Histogram (clean)
plt.subplot(2, 2, 1)

plt.hist(flight_clean, bins=100, label="Flight")
plt.hist(training_clean, bins=100, alpha=0.5, label="Training")

plt.title("Clean Distribution")
plt.legend()

# Spectrum (noisy)
plt.subplot(2, 2, 2)
f1, p1 = welch(training_noisy)
f2, p2 = welch(flight_noisy)
plt.semilogy(f1, p1, label="Training")
plt.semilogy(f2, p2, label="Flight")
plt.title("Noisy Spectrum")
plt.legend()

# Autocorrelation (clean)
plt.subplot(2, 2, 3)
plt.plot(autocorr(training_clean), label="Training")
plt.plot(autocorr(flight_clean), label="Flight")
plt.title("Clean Autocorrelation")
plt.legend()

# Autocorrelation (noisy)
plt.subplot(2, 2, 4)
plt.plot(autocorr(training_noisy), label="Training")
plt.plot(autocorr(flight_noisy), label="Flight")
plt.title("Noisy Autocorrelation")
plt.legend()

plt.tight_layout()
plt.show(block=True)

print("plot created")