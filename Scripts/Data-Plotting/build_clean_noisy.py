#!/usr/bin/env python3
"""
build_clean_noisy_v2.py

Same as build_clean_noisy.py, plus DC offset calibration, plus per-segment
alignment.

WHY (DC offset): rx (real hardware IQ) commonly carries a constant complex
offset from receiver-chain effects like LO self-mixing (classic in
zero-IF/direct-conversion front ends). clean_complex is a pure reconstructed
tone with ~zero mean, so that offset doesn't cancel in noise = noisy - clean
-- it just sits in the "noise" column as a huge constant, which dominates
any power-based noise statistic (SNR, kurtosis-adjacent power terms, etc.)
without being cosmic/channel noise at all.

FIX (DC offset): estimate the offset on the same calibration window used
for CFO/gain, subtract it from rx before CFO estimation, gain estimation,
and scaling, so it's calibrated out the same way frequency offset and gain
already are.

WHY (per-segment alignment): flight_signal_1.csv is not one continuous
capture -- its 'Index' column resets every 500,000 samples (0.5s at
SAMP_RATE), with a real multi-second time gap in 'Timestamp (ISO)' at each
reset. Empirically, fitting CFO/gain/DC-offset independently on each
500,000-sample segment gives wildly different results segment to segment
(CFO estimates swinging from -4350 Hz to +3600 Hz, gain phase spanning the
full +/-180 degrees) -- the signature of each segment being an independent
SDR acquisition with its own LO relock, not a phase-continuous stream. A
single global alignment fit (the old behavior) is only valid for the first
segment; downstream segments would get a "clean" reference built from the
wrong CFO/gain/DC-offset entirely.

FIX (per-segment alignment): detect segment boundaries from Index resets,
fit DC-offset/CFO/gain independently per segment (each using its own first
`calib_len` samples, with a local time origin reset to 0 at each segment
start), then concatenate. Falls back to the old single-fit behavior
automatically when there's only one segment (no resets detected).

Usage:
    python build_clean_noisy.py /path/to/flight_signal_1.csv
"""

import sys
import numpy as np
import pandas as pd

SAMP_RATE = 1_000_000
F0_NOMINAL = 100_000
CALIB_LEN = 20_000       # samples used to fit CFO/gain/DC-offset, per segment
SCALE_PERCENTILE = 99.5  # robust scale, fit on RX (post DC-removal, all segments pooled)
TIMESTAMP_COL = "Timestamp (ISO)"
CFO_SEARCH_HZ = 30_000   # search window for estimate_cfo -- widened from an earlier 5 kHz
                          # default after finding real per-segment CFO drift up to ~14 kHz
                          # (see build_clean_noisy.py history / paper notes: HackRF TCXO
                          # tolerance at 2.4 GHz is easily kHz-scale, and several segments
                          # show smoothly time-drifting peaks well outside +/-5 kHz)


def estimate_dc_offset(rx, n_calib):
    """Robust (median-based, so isolated bursts in the calib window don't
    skew it) complex DC offset. np.median applied separately to real/imag
    naturally gives an independent I/Q offset estimate, which matches how
    LO self-mixing / IQ imbalance actually behaves in hardware."""
    calib = rx[:n_calib]
    return complex(np.median(calib.real), np.median(calib.imag))


def estimate_cfo(tx, rx, fs, f0_nominal, search_hz=5000):
    n = min(len(tx), len(rx))
    tx, rx = tx[:n], rx[:n]
    beat = rx * np.conj(tx)
    beat_f = np.fft.fft(beat * np.hanning(n))
    freqs = np.fft.fftfreq(n, d=1 / fs)
    mask = np.abs(freqs) < search_hz
    return freqs[mask][np.argmax(np.abs(beat_f[mask]))]


def estimate_gain(tx_cfo_corrected, rx):
    n = min(len(tx_cfo_corrected), len(rx))
    tx, rx = tx_cfo_corrected[:n], rx[:n]
    return np.vdot(tx, rx) / np.vdot(tx, tx)


def robust_scale(x, percentile=SCALE_PERCENTILE):
    return np.percentile(np.abs(x), percentile)


def find_segment_boundaries(index_col):
    """Given the raw 'Index' column (may reset to 0 partway through if the
    file is multiple stitched captures), return a list of (start, end) row
    ranges, one per segment. A file with no resets returns a single segment
    covering the whole array."""
    index_col = np.asarray(index_col)
    reset_positions = np.where(np.diff(index_col) < 0)[0] + 1
    boundaries = [0] + list(reset_positions) + [len(index_col)]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def align_segment(tx, rx, fs, f0_nominal, calib_len):
    """Fit DC-offset/CFO/gain on one segment (local time origin = 0 at the
    start of this segment) and return the aligned clean/noisy arrays plus
    the fitted parameters, for diagnostics."""
    n = len(rx)
    n_calib = min(calib_len, n)

    dc_hat = estimate_dc_offset(rx, n_calib)
    rx = rx - dc_hat

    t_calib = np.arange(n_calib) / fs
    df_hat = estimate_cfo(tx[:n_calib], rx[:n_calib], fs, f0_nominal, search_hz=CFO_SEARCH_HZ)
    tx_cfo_calib = tx[:n_calib] * np.exp(1j * 2 * np.pi * df_hat * t_calib)
    A_hat = estimate_gain(tx_cfo_calib, rx[:n_calib])

    t_full = np.arange(n) / fs
    clean_complex_raw = A_hat * tx * np.exp(1j * 2 * np.pi * df_hat * t_full)

    return {
        "clean_complex_raw": clean_complex_raw,
        "noisy_complex_raw": rx,  # DC-removed, not yet globally scaled
        "df_hat": df_hat,
        "A_hat": A_hat,
        "dc_hat": dc_hat,
        "n": n,
    }


def build_clean_noisy(tx, rx, fs=SAMP_RATE, f0_nominal=F0_NOMINAL, calib_len=CALIB_LEN, segments=None):
    n_full = min(len(tx), len(rx))
    tx, rx = tx[:n_full], rx[:n_full]

    if segments is None:
        segments = [(0, n_full)]

    seg_results = []
    for start, end in segments:
        seg_results.append(align_segment(tx[start:end], rx[start:end], fs, f0_nominal, calib_len))

    clean_complex_raw = np.concatenate([s["clean_complex_raw"] for s in seg_results])
    noisy_complex_raw = np.concatenate([s["noisy_complex_raw"] for s in seg_results])

    scale = robust_scale(noisy_complex_raw)  # pooled across all segments, post DC-removal
    if scale == 0:
        scale = robust_scale(clean_complex_raw) or 1.0

    clean_complex = clean_complex_raw / scale
    noisy_complex = noisy_complex_raw / scale
    noise = noisy_complex - clean_complex

    snr = 10 * np.log10(np.sum(np.abs(clean_complex) ** 2) / (np.sum(np.abs(noise) ** 2) + 1e-20))

    return {
        "clean_complex": clean_complex,
        "noisy_complex": noisy_complex,
        "noise": noise,
        "scale": scale,
        "snr_db": snr,
        "n": n_full,
        "n_segments": len(segments),
        "segments": segments,
        "seg_results": seg_results,
    }


def main(csv_path, out_path=None):
    df = pd.read_csv(csv_path)

    segment_boundaries = None
    if "Index" in df.columns:
        segment_boundaries = find_segment_boundaries(df["Index"])
        if len(segment_boundaries) > 1:
            print(f"[WARNING] 'Index' column resets {len(segment_boundaries) - 1} time(s) -- this "
                  f"file looks like {len(segment_boundaries)} stitched captures. Fitting "
                  f"CFO/gain/DC-offset independently per segment instead of one global fit.")

    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format="ISO8601")
        if not df[TIMESTAMP_COL].is_monotonic_increasing:
            print(f"[WARNING] '{TIMESTAMP_COL}' is not already sorted -- sorting would "
                  f"invalidate the segment boundaries computed from 'Index'. Skipping sort.")
        # (No sort here: segment boundaries above are computed on file row order;
        # re-sorting afterward would silently desync them from the data.)
    else:
        print(f"[WARNING] '{TIMESTAMP_COL}' column not found -- skipping timestamp check.")

    if "Index" in df.columns:
        df = df.drop(columns=["Index"])
    df.insert(0, "Index", np.arange(len(df)))

    tx = df["TX Real"].to_numpy() + 1j * df["TX Imag"].to_numpy()
    rx = df["RX Real"].to_numpy() + 1j * df["RX Imag"].to_numpy()

    result = build_clean_noisy(tx, rx, segments=segment_boundaries)

    print(f"[INFO] n_samples     = {result['n']}")
    print(f"[INFO] n_segments    = {result['n_segments']}")
    print(f"[INFO] scale ({SCALE_PERCENTILE}pct, pooled) = {result['scale']:.6g}")
    print(f"[INFO] residual SNR (pooled) = {result['snr_db']:.2f} dB")

    print(f"[INFO] per-segment fit (first {min(10, result['n_segments'])} shown):")
    print(f"       {'seg':>4} {'n':>8} {'dc_hat':>18} {'cfo_hz':>10} {'|A_hat|':>12} {'A_hat_deg':>10}")
    for i, (s, seg) in enumerate(zip(result["segments"], result["seg_results"])):
        if i >= 10:
            print(f"       ... ({result['n_segments'] - 10} more segments)")
            break
        print(f"       {i:>4} {seg['n']:>8} {seg['dc_hat']:>18.6g} {seg['df_hat']:>10.1f} "
              f"{abs(seg['A_hat']):>12.6g} {np.degrees(np.angle(seg['A_hat'])):>10.2f}")

    out_df = pd.DataFrame({
        "Index": df["Index"].to_numpy()[: result["n"]],
        "Time": np.arange(result["n"]) / SAMP_RATE,
        "Noisy Signal": result["noisy_complex"].real,
        "Clean Signal": result["clean_complex"].real,
        'TX Magnitude': df['TX Magnitude'],
        'RX Magntiude': df['RX Magnitude']
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
