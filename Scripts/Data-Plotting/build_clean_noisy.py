#!/usr/bin/env python3
"""
build_clean_noisy.py

Offline, one-shot construction of a (clean, noisy) signal pair from a flight
CSV containing TX/RX Real+Imag columns -- analogous to the "Clean Signal" /
"Noisy Signal" pair used for the simulated data.

No windowing, no polling loop: reads the whole CSV once, fits alignment
once, writes the full-length clean/noisy arrays to a new CSV.

Usage:
    python build_clean_noisy.py /path/to/flight_signal_2.csv
"""

import sys
import numpy as np
import pandas as pd

SAMP_RATE = 1_000_000
F0_NOMINAL = 100_000
CALIB_LEN = 20_000       # samples used to fit CFO/gain
SCALE_PERCENTILE = 99.5  # robust scale, fit on RX


def estimate_cfo(tx, rx, fs, f0_nominal, search_hz=5000):
    """Frequency offset via phase slope of rx*conj(tx). No lag search needed:
    for a continuous single-tone tx, sample lag is degenerate with phase and
    gets absorbed by the gain fit below."""
    n = min(len(tx), len(rx))
    tx, rx = tx[:n], rx[:n]
    beat = rx * np.conj(tx)
    beat_f = np.fft.fft(beat * np.hanning(n))
    freqs = np.fft.fftfreq(n, d=1 / fs)
    mask = np.abs(freqs) < search_hz
    return freqs[mask][np.argmax(np.abs(beat_f[mask]))]


def estimate_gain(tx_cfo_corrected, rx):
    """Least-squares complex gain (amplitude + phase) mapping tx -> rx."""
    n = min(len(tx_cfo_corrected), len(rx))
    tx, rx = tx_cfo_corrected[:n], rx[:n]
    return np.vdot(tx, rx) / np.vdot(tx, tx)


def robust_scale(x, percentile=SCALE_PERCENTILE):
    return np.percentile(np.abs(x), percentile)


def build_clean_noisy(tx, rx, fs=SAMP_RATE, f0_nominal=F0_NOMINAL, calib_len=CALIB_LEN):
    n_full = min(len(tx), len(rx))
    tx, rx = tx[:n_full], rx[:n_full]

    n_calib = min(calib_len, n_full)
    t_calib = np.arange(n_calib) / fs

    df_hat = estimate_cfo(tx[:n_calib], rx[:n_calib], fs, f0_nominal)
    tx_cfo_calib = tx[:n_calib] * np.exp(1j * 2 * np.pi * df_hat * t_calib)
    A_hat = estimate_gain(tx_cfo_calib, rx[:n_calib])

    t_full = np.arange(n_full) / fs
    clean_complex_raw = A_hat * tx * np.exp(1j * 2 * np.pi * df_hat * t_full)

    scale = robust_scale(rx)
    if scale == 0:
        scale = robust_scale(clean_complex_raw) or 1.0

    clean_complex = clean_complex_raw / scale
    noisy_complex = rx / scale
    noise = noisy_complex - clean_complex

    snr = 10 * np.log10(np.sum(np.abs(clean_complex) ** 2) / (np.sum(np.abs(noise) ** 2) + 1e-20))

    return {
        "clean_complex": clean_complex,
        "noisy_complex": noisy_complex,
        "noise": noise,
        "df_hat": df_hat,
        "A_hat": A_hat,
        "scale": scale,
        "snr_db": snr,
        "n": n_full,
    }


def main(csv_path, out_path=None):
    df = pd.read_csv(csv_path)

    # flag (not fix) multi-cycle files: if "Index" resets partway through,
    # a single alignment fit over the whole file is not valid -- see chat.
    if "Index" in df.columns:
        resets = (df["Index"].diff() < 0).sum()
        if resets > 0:
            print(f"[WARNING] 'Index' column resets {resets} time(s) -- this file "
                  f"looks like multiple stitched captures. A single alignment fit "
                  f"assumes one continuous capture; results may be invalid past "
                  f"the first cycle.")

    tx = df["TX Real"].to_numpy() + 1j * df["TX Imag"].to_numpy()
    rx = df["RX Real"].to_numpy() + 1j * df["RX Imag"].to_numpy()

    result = build_clean_noisy(tx, rx)

    print(f"[INFO] n_samples     = {result['n']}")
    print(f"[INFO] df_hat        = {result['df_hat']:.2f} Hz")
    print(f"[INFO] A_hat         = {result['A_hat']}")
    print(f"[INFO] scale ({SCALE_PERCENTILE}pct) = {result['scale']:.6g}")
    print(f"[INFO] residual SNR  = {result['snr_db']:.2f} dB")

    # single real-valued column each, matching the simulated CSV's
    # Time,Noisy Signal,Clean Signal layout. Real part only -- this is the
    # same reduction inference.py already does (win_clean_complex.real)
    # before feeding windows to the env.
    out_df = pd.DataFrame({
        "Index": df["Index"].to_numpy()[: result["n"]] if "Index" in df.columns
                 else np.arange(result["n"]),
        "Time": np.arange(result["n"]) / SAMP_RATE,
        "Noisy Signal": result["noisy_complex"].real,
        "Clean Signal": result["clean_complex"].real,
    })

    if out_path is None:
        out_path = csv_path.rsplit(".", 1)[0] + "_clean_noisy.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")

    return result, out_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_clean_noisy.py /path/to/flight_signal.csv [output.csv]")
        sys.exit(1)
    csv_arg = sys.argv[1]
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(csv_arg, out_arg)