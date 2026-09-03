#!/usr/bin/env python3
"""
compare_signal_properties.py   (my-list items 1, 3; groundwork for 11)
=====================================================================
Computes a common set of time- and frequency-domain properties for one or
more (Clean Signal, Noisy Signal) CSVs and prints / saves a comparison table.
Used to characterise flight_signal_1 (item 1) and to compare the OAN / PFN
simulated signals against it (item 3).

Metrics per signal (noise := noisy - clean):
  n_samples, duration_s, n_segments (Time-gap detected)
  clean_std, clean_amp_est (sqrt(2)*std, i.e. tone amplitude), clean_peak_hz
  noise_std, noise_mean, severity_ratio (noise_std / clean_std)
  snr_raw_db (whole-signal), per-window SNR mean/std/min/max (window=1000)
  noise_excess_kurtosis, noise_psd_loglog_slope, noise_acf_lag1
  rolling_rms_drift_ratio (p99/p1 of 1000-sample RMS), burst_rate_per_ksample
  noisy_exact_zero_frac (ADC-quantisation / dropout indicator)

Reads only the two needed columns; safe on the multi-GB flight CSV.

Usage:
    python compare_signal_properties.py LABEL=path/to.csv [LABEL2=path2.csv ...]
    # no args -> the default OAN-sim / PFN-sim / flight_signal_1 comparison
"""

import os
import sys

import numpy as np
import pandas as pd

WINDOW = 1000
FS = 1_000_000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../Data"))

DEFAULT_SIGNALS = {
    "OAN_sim":  os.path.join(DATA_DIR, "simulated_signal_oan.csv"),
    "PFN_sim":  os.path.join(DATA_DIR, "simulated_signal_pfn.csv"),
    "flight_1": os.path.join(DATA_DIR, "flight_signal_1_clean_noisy.csv"),
}


def _welch_psd(x, nseg=8):
    n = len(x) // nseg
    if n < 16:
        return None, None
    acc = None
    for i in range(nseg):
        seg = x[i * n:(i + 1) * n]
        seg = seg - seg.mean()
        p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        acc = p if acc is None else acc + p
    psd = acc / nseg
    freqs = np.fft.rfftfreq(n, d=1 / FS)
    return freqs[1:], psd[1:]


def _loglog_slope(freqs, psd, f_lo=1e3, f_hi=3e5):
    m = (freqs >= f_lo) & (freqs <= f_hi) & (psd > 0)
    if m.sum() < 5:
        m = (freqs > freqs[0]) & (psd > 0)
    return float(np.polyfit(np.log10(freqs[m]), np.log10(psd[m]), 1)[0])


def _peak_hz(x):
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), d=1 / FS)
    return float(f[np.argmax(X)])


def _acf_lag1(x):
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0


def _rolling_rms(x, win):
    c = np.cumsum(np.insert(x ** 2, 0, 0.0))
    return np.sqrt((c[win:] - c[:-win]) / win)


def _burst_rate_per_ksample(noise):
    """Fraction-scaled count of samples > 5 sigma from a 50-sample local mean,
    grouped into runs; reported as runs per 1000 samples."""
    sm = np.convolve(noise, np.ones(50) / 50, mode="same")
    dev = noise - sm
    over = np.abs(dev) > 5 * np.std(dev)
    runs = int(np.sum(np.diff(over.astype(np.int8)) == 1) + (1 if over[0] else 0))
    return 1000.0 * runs / len(noise)


def analyse(label, path):
    if not os.path.exists(path):
        print(f"[skip] {label}: {path} not found")
        return None
    df = pd.read_csv(path, usecols=["Clean Signal", "Noisy Signal", "Time"]
                     if _has_time(path) else ["Clean Signal", "Noisy Signal"])
    clean = df["Clean Signal"].to_numpy(np.float64)
    noisy = df["Noisy Signal"].to_numpy(np.float64)
    noise = noisy - clean
    n = len(clean)

    if "Time" in df.columns:
        t = df["Time"].to_numpy(np.float64)
        duration = float(t[-1] - t[0])
        n_segments = int(np.sum(np.diff(t) > 0.5) + 1)
    else:
        duration = n / FS
        n_segments = 1

    nwin = n // WINDOW
    cw = clean[:nwin * WINDOW].reshape(-1, WINDOW)
    nw = noise[:nwin * WINDOW].reshape(-1, WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_snr = 10 * np.log10(np.mean(cw ** 2, axis=1) / np.mean(nw ** 2, axis=1))
    win_snr = win_snr[np.isfinite(win_snr)]

    freqs, psd = _welch_psd(noise)
    slope = _loglog_slope(freqs, psd) if freqs is not None else float("nan")

    rms = _rolling_rms(noise, WINDOW)
    drift = float(np.percentile(rms, 99) / max(np.percentile(rms, 1), 1e-12))

    xk = noise - noise.mean()
    excess_kurt = float((xk ** 4).mean() / (xk.std() ** 4) - 3.0)

    row = {
        "label": label,
        "n_samples": n,
        "duration_s": round(duration, 3),
        "n_segments": n_segments,
        "clean_std": clean.std(),
        "clean_amp_est": np.sqrt(2) * clean.std(),
        "clean_peak_hz": _peak_hz(clean[:min(n, 200_000)]),
        "noise_std": noise.std(),
        "noise_mean": noise.mean(),
        "severity_ratio": noise.std() / (clean.std() + 1e-20),
        "snr_raw_db": 10 * np.log10((np.mean(clean ** 2) + 1e-20) / (np.mean(noise ** 2) + 1e-20)),
        "win_snr_mean_db": float(win_snr.mean()),
        "win_snr_std_db": float(win_snr.std()),
        "win_snr_min_db": float(win_snr.min()),
        "win_snr_max_db": float(win_snr.max()),
        "noise_excess_kurtosis": excess_kurt,
        "noise_psd_loglog_slope": slope,
        "noise_acf_lag1": _acf_lag1(noise[:min(n, 500_000)]),
        "rolling_rms_drift_ratio": drift,
        "burst_rate_per_ksample": _burst_rate_per_ksample(noise),
        "noisy_exact_zero_frac": float(np.mean(noisy == 0.0)),
    }
    print(f"[ok] {label}: n={n:,} segs={n_segments} snr_raw={row['snr_raw_db']:.2f}dB "
          f"kurt={excess_kurt:.1f} slope={slope:.3f} drift={drift:.1f}")
    return row


def _has_time(path):
    head = pd.read_csv(path, nrows=0)
    return "Time" in head.columns


def main(pairs):
    rows = [r for label, path in pairs.items() if (r := analyse(label, path)) is not None]
    if not rows:
        print("nothing analysed")
        return
    out = pd.DataFrame(rows).set_index("label").T
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print("\n=== signal property comparison ===")
    print(out.to_string())
    dst = os.path.join(DATA_DIR, "signal_property_comparison.csv")
    out.to_csv(dst)
    print(f"\n[INFO] wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pairs = {}
        for arg in sys.argv[1:]:
            label, path = arg.split("=", 1)
            pairs[label] = path
    else:
        pairs = DEFAULT_SIGNALS
    main(pairs)
