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

import os
import sys
import numpy as np
import pandas as pd

SAMP_RATE = 1_000_000
F0_NOMINAL = 100_000
CALIB_LEN = 20_000       # samples used to fit CFO/gain/DC-offset, per segment
SCALE_PERCENTILE = 99.5  # robust scale, fit on RX (post DC-removal, all segments pooled)
TIMESTAMP_COL = "Timestamp (ISO)"
GAP_FILL_PERIOD_S = 10.0  # spacing of explicit zero rows dropped into each inter-segment
                          # dead gap in the *_gapfilled.csv companion file (viz only)
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


def build_gap_filled(out_df, segments, period_s=GAP_FILL_PERIOD_S):
    """Return a copy of out_df with explicit all-zero rows inserted into every
    inter-segment dead gap, roughly every `period_s` seconds.

    The real-timed out_df represents each dead gap only as a jump in the 'Time'
    column (segment ends at t=39.99 s, next row is t=60.48 s). A plotter then
    draws a straight diagonal across the gap, which reads as if signal existed
    there. These filler rows pin the trace to zero across the gap instead --
    "the flight is still going, we're just not recording" -- so gaps show as
    flat dead stretches. This file is a visualization aid only; the RL/analysis
    pipeline consumes the gap-free *_clean_noisy.csv, not this one.
    """
    if segments is None or len(segments) < 2:
        return out_df.copy(), 0

    t = out_df["Time"].to_numpy()
    zero_cols = ["Noisy Signal", "Clean Signal", "TX Magnitude", "RX Magntiude"]
    pieces = []
    n_filler = 0
    for k, (start, end) in enumerate(segments):
        pieces.append(out_df.iloc[start:end])
        if k == len(segments) - 1:
            break
        gap_start = t[end - 1]
        gap_end = t[segments[k + 1][0]]
        fill_t = np.arange(gap_start + period_s, gap_end, period_s)
        if fill_t.size == 0:                       # gap shorter than period_s
            fill_t = np.array([0.5 * (gap_start + gap_end)])
        n_filler += fill_t.size
        filler = pd.DataFrame({"Time": fill_t})
        for col in zero_cols:
            filler[col] = 0.0
        pieces.append(filler.reindex(columns=out_df.columns))

    gf = pd.concat(pieces, ignore_index=True)
    gf["Index"] = np.arange(len(gf))              # single contiguous index over real + filler rows
    return gf, n_filler


def main(csv_path, out_path=None):
    df = pd.read_csv(csv_path)

    segment_boundaries = None
    if "Index" in df.columns:
        segment_boundaries = find_segment_boundaries(df["Index"])
        if len(segment_boundaries) > 1:
            print(f"[WARNING] 'Index' column resets {len(segment_boundaries) - 1} time(s) -- this "
                  f"file looks like {len(segment_boundaries)} stitched captures. Fitting "
                  f"CFO/gain/DC-offset independently per segment instead of one global fit.")

    real_time = None  # elapsed-seconds axis taken from the capture's own wall-clock timestamps
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], format="ISO8601")
        if not df[TIMESTAMP_COL].is_monotonic_increasing:
            print(f"[WARNING] '{TIMESTAMP_COL}' is not already sorted -- sorting would "
                  f"invalidate the segment boundaries computed from 'Index'. Skipping sort.")
        # (No sort here: segment boundaries above are computed on file row order;
        # re-sorting afterward would silently desync them from the data.)
        #
        # Real time axis: seconds elapsed since the first sample's timestamp. This makes
        # the output a 1:1 time-domain map of the input -- row i's clean/noisy value sits
        # at the same wall-clock instant as row i's TX/RX values -- and preserves BOTH the
        # true capture duration and the multi-second dead gaps between stitched segments
        # (across a gap 'Time' just jumps forward with no rows in between). The old
        # np.arange(n)/SAMP_RATE axis silently collapsed the 34 segments + gaps into one
        # contiguous ~17 s stream.
        real_time = (df[TIMESTAMP_COL] - df[TIMESTAMP_COL].iloc[0]).dt.total_seconds().to_numpy()
    else:
        print(f"[WARNING] '{TIMESTAMP_COL}' column not found -- falling back to synthetic "
              f"np.arange(n)/SAMP_RATE time axis (real duration and inter-segment gaps NOT "
              f"preserved).")

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

    n_out = result["n"]
    if real_time is not None:
        time_axis = real_time[:n_out]
    else:
        time_axis = np.arange(n_out) / SAMP_RATE

    out_df = pd.DataFrame({
        "Index": df["Index"].to_numpy()[:n_out],
        "Time": time_axis,
        "Noisy Signal": result["noisy_complex"].real,
        "Clean Signal": result["clean_complex"].real,
        'TX Magnitude': df['TX Magnitude'].to_numpy()[:n_out],
        'RX Magntiude': df['RX Magnitude'].to_numpy()[:n_out],
    })

    if out_path is None:
        out_path = csv_path.rsplit(".", 1)[0] + "_clean_noisy.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")

    # --- confirm the rebuilt file covers the same length of time as the input ---
    if real_time is not None:
        orig_span = float(real_time[-1] - real_time[0])          # full input capture
        written = pd.read_csv(out_path, usecols=["Time"])["Time"]  # re-read from disk
        new_span = float(written.iloc[-1] - written.iloc[0])
        delta = abs(orig_span - new_span)
        match = delta < 1e-6
        print(f"[CHECK] input   {os.path.basename(csv_path)}: {len(real_time)} rows, "
              f"span {orig_span:.6f} s ({orig_span / 60:.4f} min)")
        print(f"[CHECK] rebuilt {os.path.basename(out_path)}: {len(written)} rows, "
              f"span {new_span:.6f} s ({new_span / 60:.4f} min)")
        print(f"[CHECK] same length of time: {match}  (|delta| = {delta:.2e} s)")
        if not match:
            print("[CHECK] WARNING: rebuilt file does NOT span the same time as the input "
                  "(likely a TX/RX length mismatch truncated the output).")
        if result["n_segments"] > 1 and result["segments"] is not None:
            gaps = [real_time[s1] - real_time[e0 - 1]
                    for (_s0, e0), (s1, _e1) in zip(result["segments"][:-1], result["segments"][1:])]
            print(f"[CHECK] {result['n_segments']} segments, {len(gaps)} inter-segment dead "
                  f"gaps retained in 'Time' (each {min(gaps):.2f}-{max(gaps):.2f} s, "
                  f"{sum(gaps):.2f} s total dead time)")
    else:
        print("[CHECK] no timestamp column in input -- real-time span cannot be verified.")

    # --- companion file: dead gaps filled with explicit zero rows for plotting ---
    # Does NOT touch out_path above; writes a separate *_gapfilled.csv.
    if real_time is not None and result["n_segments"] > 1 and result["segments"] is not None:
        gf_df, n_filler = build_gap_filled(out_df, result["segments"])
        gf_path = out_path.rsplit(".", 1)[0] + "_gapfilled.csv"
        gf_df.to_csv(gf_path, index=False)
        print(f"[INFO] wrote {gf_path}")

        gf_written = pd.read_csv(gf_path, usecols=["Time"])["Time"]
        gf_span = float(gf_written.iloc[-1] - gf_written.iloc[0])
        gf_delta = abs(orig_span - gf_span)
        print(f"[CHECK] gapfilled {os.path.basename(gf_path)}: {len(gf_written)} rows "
              f"({n_filler} zero filler rows added), span {gf_span:.6f} s "
              f"({gf_span / 60:.4f} min)")
        print(f"[CHECK] same length of time as input: {gf_delta < 1e-6}  "
              f"(|delta| = {gf_delta:.2e} s)")
    else:
        print("[INFO] no inter-segment gaps to fill -- skipping *_gapfilled.csv")

    return result, out_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_clean_noisy.py /path/to/flight_signal.csv [output.csv]")
        sys.exit(1)
    csv_arg = sys.argv[1]
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(csv_arg, out_arg)
