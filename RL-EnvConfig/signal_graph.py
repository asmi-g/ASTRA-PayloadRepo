import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

N = 1000

df = pd.read_csv("Data/flight_signal_2.csv")
print(df.describe())
df = df.head(N)

clean_signal = np.sin(np.linspace(0, 2 * np.pi, N))
raw_signal = clean_signal + np.random.normal(0, 0.3, size=N)

scale = np.max(np.abs(clean_signal))
if scale > 0:
    clean_signal = clean_signal / scale
    raw_signal = raw_signal / scale

# Pink noise + burst signal
pink_noise = generate_pink_noise(N, noise_power=0.1)
pink_burst_signal = clean_signal + pink_noise
pink_burst_signal = add_burst(
    pink_burst_signal,
    burst_probability=0.005,
    burst_amplitude=2.0,
    burst_duration=15
)

# Normalize to same scale
scale = np.max(np.abs(pink_burst_signal))
if scale > 0:
    pink_burst_signal /= scale


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
# TX + clean signal
axes[0].plot(df['TX Magnitude'], label="TX Magnitude", alpha=0.7)
#axes[0].plot(clean_signal, label="Clean Signal", linewidth=2)
axes[0].set_title("TX Magnitude + Clean Signal")
axes[0].legend()
axes[0].grid(True)

# RX + noisy signal
axes[1].plot(df['RX Magnitude'], label="RX Magnitude", alpha=0.7)
# axes[1].plot(raw_signal, label="Noisy Signal", linewidth=2)
#axes[1].plot(pink_burst_signal, label="Pink Noise + Bursts", linewidth=2)
axes[1].set_title("RX Magnitude + Noisy Signal")
axes[1].legend()
axes[1].grid(True)

axes[1].set_xlabel("Sample Index")

plt.tight_layout()
plt.show()



# plt.subplot(2, 1, 1)
# plt.plot(df['TX Magnitude'])
# plt.title("TX Magnitude")

# plt.subplot(2, 1, 2)
# plt.plot(df['RX Magnitude'])
# plt.title("RX Magnitude")

# plt.tight_layout()
# plt.show()
