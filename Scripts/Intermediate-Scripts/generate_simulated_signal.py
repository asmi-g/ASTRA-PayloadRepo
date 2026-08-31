#!/usr/bin/env python3
"""
generate_simulated_signal.py

Batch (non-interactive) generator for a matched-Hz simulated clean/noisy
signal pair, using the SAME noise model (white noise + pink noise + random
amplitude bursts) as NoiseReductionEnv._generate_signals() in
astra_rev1/envs/custom_env_022025.py -- i.e. the noise model the OFT
("original flight type") RL model is trained on.

WHY: the original team's noise model / simulated validation signal used
F_SAMPLING=1e4, F_SIGNAL=100 Hz (see the old signal_simulation.py), which
does not match the real hardware's SAMP_RATE=1e6, F0=100e3 Hz (see
Scripts/SDR/TX.py). This script reuses the same noise-generation logic but
with hardware-matched parameters, so downstream comparisons (simulated vs.
flight_signal_1 in signal_analysis.py) aren't confounded by a frequency
mismatch on top of the noise-model mismatch under investigation.

Output columns match build_clean_noisy.py's *_clean_noisy.csv schema
(Index, Time, Noisy Signal, Clean Signal), so signal_analysis.py,
inference.py, and verify_clean_noisy.py all work on this file unmodified.

Usage:
    python generate_simulated_signal.py [signal_length] [output_path] [seed]
"""

import sys
import os
import numpy as np
import pandas as pd

SAMP_RATE = 1_000_000   # matches TX.py / RX.py samp_rate
F_SIGNAL = 100_000       # matches TX.py analog_sig_source_x_0 frequency
NOISE_POWER = 0.1        # matches custom_env_022025.py / signal_simulation.py
DEFAULT_SIGNAL_LENGTH = 500_000  # one flight-capture segment's worth of samples
DEFAULT_SEED = 42


def generate_pink_noise(size, white_noise, noise_power=NOISE_POWER):
    """Matches NoiseReductionEnv._generate_pink_noise exactly (which in turn
    matches the original signal_simulation.py logic: the caller must rescale
    the result by sqrt(noise_power), see generate_signals() below).

    NOTE: this filter (sqrt(|freq|), gain rising with frequency) actually
    produces blue noise, not true 1/f pink noise, despite the name. A true
    1/f fix was tried and reverted -- naive per-bin FFT division concentrates
    most of the energy in the single lowest nonzero frequency bin, making
    pink noise ~14-17x louder than white noise and unstably dependent on
    signal length. Kept as-is to match custom_env_022025.py / the original
    signal_simulation.py, documented as a known discrepancy for the paper."""
    freq = np.fft.fftfreq(size)
    pink_filter = np.sqrt(np.abs(freq))
    spectrum = np.fft.fft(white_noise / np.sqrt(noise_power))
    spectrum *= pink_filter
    pink_noise = np.fft.ifft(spectrum)
    return np.real(pink_noise)


def add_bursts(signal, burst_probability=0.01, burst_amplitude=3.0, burst_duration=10):
    """Matches NoiseReductionEnv._add_bursts exactly."""
    noisy_signal = signal.copy()
    for i in range(len(signal)):
        if np.random.rand() < burst_probability:
            burst_start = i
            burst_end = min(i + burst_duration, len(signal))
            noisy_signal[burst_start:burst_end] += np.random.uniform(-burst_amplitude, burst_amplitude)
    return noisy_signal


def generate_signals(signal_length, samp_rate=SAMP_RATE, f_signal=F_SIGNAL, noise_power=NOISE_POWER):
    """Matches NoiseReductionEnv._generate_signals exactly, parameterized
    by sample rate / signal frequency instead of the hardcoded 1e6/100_000
    (which happen to already be the matched-Hz values, but kept explicit
    here for clarity)."""
    x = np.arange(signal_length)
    t = x / samp_rate

    clean = np.sin(2 * np.pi * f_signal * t)
    white_noise = np.random.normal(0, np.sqrt(noise_power), size=signal_length)
    pink_noise = generate_pink_noise(signal_length, white_noise, noise_power) * np.sqrt(noise_power)
    noisy = clean + white_noise + pink_noise
    noisy = add_bursts(noisy)

    return clean, noisy


def main(signal_length=DEFAULT_SIGNAL_LENGTH, out_path=None, seed=DEFAULT_SEED):
    np.random.seed(seed)

    clean, noisy = generate_signals(signal_length)

    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power_actual = np.mean(noise ** 2)
    snr = 10 * np.log10((signal_power + 1e-20) / (noise_power_actual + 1e-20))

    print(f"[INFO] signal_length = {signal_length}")
    print(f"[INFO] samp_rate = {SAMP_RATE} Hz, f_signal = {F_SIGNAL} Hz (matched to TX.py)")
    print(f"[INFO] seed = {seed}")
    print(f"[INFO] clean std = {clean.std():.6g}, noisy std = {noisy.std():.6g}, noise std = {noise.std():.6g}")
    print(f"[INFO] SNR (raw) = {snr:.2f} dB")

    out_df = pd.DataFrame({
        "Index": np.arange(signal_length),
        "Time": np.arange(signal_length) / SAMP_RATE,
        "Noisy Signal": noisy,
        "Clean Signal": clean,
    })

    if out_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.normpath(os.path.join(script_dir, "../../Data/simulated_signal_match_hz.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")

    return out_df


if __name__ == "__main__":
    length_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SIGNAL_LENGTH
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SEED
    main(length_arg, out_arg, seed_arg)
