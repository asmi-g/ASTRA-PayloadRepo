#!/usr/bin/env python3
"""
static_filter_realtime.py  (my-list item 10)
============================================
Non-adaptive (fixed threshold_factor) wavelet-denoising baseline, run through
the SAME real-time signal construction as inference.py -- so its numbers are
directly comparable to the RL agent's, with alignment method held constant.

Two modes:

  MODE = "sweep"  -> sweep threshold_factor over SWEEP_RANGE, filtering a
                     subset of windows spread across the file, and report the
                     value that maximises mean per-window SNR improvement
                     (degenerate all-shrink thresholds excluded). Writes a
                     <ts>_<tag>_static_sweep.csv. Run this FIRST for each
                     data source, then paste the winner into BEST_TF below.

  MODE = "apply"  -> filter EVERY window at BEST_TF[DATA_SOURCE] and write a
                     <ts>_<tag>_static_results_tf<...>.csv with the same
                     columns inference.py produces (window, threshold_factor,
                     snr_improvement, signal_loss, correlation, running_mean),
                     plus a printed mean +/- std summary.

DATA_SOURCE:
  "oan_sim" / "pfn_sim" / "simulated" / "un_sim"
        -> pre-built real-valued Clean/Noisy columns, used directly, only a
           scalar amplitude normalisation (matches inference.py's non-flight
           path).
  "flight"
        -> raw TX/RX IQ from flight_signal_1.csv, complex clean/noisy rebuilt
           live with periodic CFO/gain/DC recalibration (verbatim copy of
           inference.py's alignment; that file is the source of truth).

Filter mechanics come straight from StatelessDenoisingEnv.apply_filter
(db4, level 5, periodization, VisuShrink threshold, details-only, mean
preserving), i.e. the exact filter the RL agent drives -- no reimplementation.

Usage:  edit MODE / DATA_SOURCE / BEST_TF below, then
        cd RL-EnvConfig && python static_filter_realtime.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from astra_rev1.envs.custom_env_022025 import StatelessDenoisingEnv

# ============================ configuration ============================
# MODE / DATA_SOURCE / (apply-mode tf) can be overridden on the command line:
#   python static_filter_realtime.py [MODE] [DATA_SOURCE] [tf]
# The optional 3rd arg, in apply mode only, overrides BEST_TF[DATA_SOURCE].
MODE        = "sweep"       # "sweep" or "apply"
DATA_SOURCE = "oan_sim"     # "oan_sim" | "pfn_sim" | "opbn_sim" | "simulated" | "un_sim" | "flight"
TF_OVERRIDE = None
if len(sys.argv) > 1:
    MODE = sys.argv[1]
if len(sys.argv) > 2:
    DATA_SOURCE = sys.argv[2]
if len(sys.argv) > 3:
    TF_OVERRIDE = float(sys.argv[3])

WINDOW_SIZE = 1000          # must match the trained models / inference.py
STRIDE      = 100           # must match inference.py
LEVEL       = 5             # wavelet decomposition level (matches inference.py flevel)

# threshold_factor search grid. Deliberately NOT clamped to the RL action
# range [0.05, 2.5]: this baseline answers "how well could a competently tuned
# fixed filter do", not "how well could a fixed filter restricted to the
# agent's knob range do". (The in-range best is also reported for reference.)
SWEEP_RANGE      = np.linspace(0.05, 6.0, 80)
SWEEP_MAX_WINDOWS = 3000     # windows used in sweep mode (spread across the file)
# apply mode: None = filter EVERY window of the file (full pass). An int caps
# it to that many windows, spread evenly across the file. Sim signals have
# ~5000 windows either way; flight is ~170k windows (~a few minutes).
APPLY_MAX_WINDOWS = None

# Best static threshold_factor per data source, from a prior MODE="sweep" run.
# Leave None until swept; "apply" mode raises if the entry is still None.
BEST_TF = {
    "oan_sim":   0.2759,   # sweep 20260901_015956 (seed 20260904, w/ flare), +1.01 dB
    "pfn_sim":   0.4266,   # sweep 20260901_020001 (seed 20260907, realrate),  +6.63 dB
    "opbn_sim":  None,     # set from the sweep, or pass as 3rd CLI arg
    "simulated": None,
    "un_sim":    None,
    # flight sweep 20260901_011827: SNR improvement is MONOTONIC in tf across
    # the whole [0.05, 6.0] range (no interior optimum) and NO threshold is
    # degenerate (energy ratio stays ~0.97 even at tf=6) -- the filter is
    # near-inert on the 95%-exact-zero flight signal. Value below is the best
    # WITHIN the RL action range [0.05, 2.5]; the unconstrained sweep just
    # rises to its ceiling (tf=6 -> +0.246 dB). See item10 notes.
    "flight":    2.4601,   # +0.124 dB mean SNR improvement (RL-range max)
}

DEGENERATE_ENERGY_RATIO = 0.2   # matches reward_funct_test.py / static_filter_baseline.py

# flight real-time alignment constants -- identical to inference.py
SAMP_RATE        = 1_000_000
CALIB_SIZE_FLIGHT = 20_000
CALIB_SIZE_SIM    = 1_000
RECAL_INTERVAL    = 100_000
SCALE_EWMA_ALPHA  = 0.3
CFO_SEARCH_HZ     = 30_000
# =====================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(script_dir, "..", "Data"))

_SIM_SOURCES = {
    "oan_sim":   ("simulated_signal_oan.csv",            "oan_sim"),
    "pfn_sim":   ("simulated_signal_pfn.csv",            "pfn_sim"),
    "opbn_sim":  ("simulated_signal_opbn.csv",           "opbn_sim"),
    "simulated": ("simulated_signal_match_hz.csv",       "sim"),
    "un_sim":    ("simulated_signal_un_noise_model.csv", "un_sim"),
}
IS_FLIGHT = DATA_SOURCE == "flight"
if IS_FLIGHT:
    CSV_PATH, RUN_TAG = os.path.join(DATA_DIR, "flight_signal_1.csv"), "fs1"
elif DATA_SOURCE in _SIM_SOURCES:
    _fname, RUN_TAG = _SIM_SOURCES[DATA_SOURCE]
    CSV_PATH = os.path.join(DATA_DIR, _fname)
else:
    raise ValueError(f"unknown DATA_SOURCE {DATA_SOURCE!r}")

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

denoiser = StatelessDenoisingEnv(window_size=WINDOW_SIZE, level=LEVEL)


# ---------------------------------------------------------------------------
# alignment helpers -- verbatim from inference.py (source of truth). Kept as a
# copy rather than an import because importing inference.py executes its
# module-level polling loop.
# ---------------------------------------------------------------------------
def estimate_alignment(tx, rx, fs, n0=0, search_hz=CFO_SEARCH_HZ):
    n = min(len(tx), len(rx))
    tx, rx = tx[:n], rx[:n]

    dc_hat = complex(np.median(rx.real), np.median(rx.imag))
    rx = rx - dc_hat

    beat = rx * np.conj(tx)
    beat_f = np.fft.fft(beat * np.hanning(n))
    freqs = np.fft.fftfreq(n, d=1 / fs)
    mag = np.abs(beat_f)

    in_band = np.abs(freqs) < search_hz
    k = int(np.argmax(np.where(in_band, mag, -np.inf)))

    bin_hz = fs / n
    km1, kp1 = (k - 1) % n, (k + 1) % n
    a, b, c = np.log(mag[km1] + 1e-20), np.log(mag[k] + 1e-20), np.log(mag[kp1] + 1e-20)
    denom = a - 2.0 * b + c
    delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5)) if denom != 0 else 0.0
    df_hat = freqs[k] + delta * bin_hz

    t = np.arange(n0, n0 + n) / fs
    tx_cfo = tx * np.exp(1j * 2 * np.pi * df_hat * t)
    A_hat = np.vdot(tx_cfo, rx) / np.vdot(tx_cfo, tx_cfo)

    scale = np.percentile(np.abs(rx), 99.5)
    if scale == 0:
        scale = np.percentile(np.abs(A_hat * tx_cfo), 99.5) or 1.0

    return {"df_hat": df_hat, "A_hat": A_hat, "dc_hat": dc_hat, "fs": fs, "scale": scale}


def apply_alignment(tx_window, params, n_start_idx):
    df_hat, A_hat, fs = params["df_hat"], params["A_hat"], params["fs"]
    n = np.arange(n_start_idx, n_start_idx + len(tx_window))
    return A_hat * tx_window * np.exp(1j * 2 * np.pi * df_hat * n / fs)


def snr_db(clean, test):
    noise = test - clean
    return 10 * np.log10((np.mean(clean ** 2) + 1e-10) / (np.mean(noise ** 2) + 1e-10))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_source():
    """Return (clean_or_tx, noisy_or_rx, is_complex). For sim sources these are
    the real-valued Clean/Noisy columns; for flight they are the complex TX/RX
    IQ streams (alignment happens per window)."""
    if IS_FLIGHT:
        df = pd.read_csv(CSV_PATH, usecols=["TX Real", "TX Imag", "RX Real", "RX Imag"])
        tx = df["TX Real"].to_numpy(np.float64) + 1j * df["TX Imag"].to_numpy(np.float64)
        rx = df["RX Real"].to_numpy(np.float64) + 1j * df["RX Imag"].to_numpy(np.float64)
        return tx, rx, True
    df = pd.read_csv(CSV_PATH, usecols=["Clean Signal", "Noisy Signal"])
    clean = df["Clean Signal"].to_numpy(np.float64)
    noisy = df["Noisy Signal"].to_numpy(np.float64)
    return clean, noisy, False


# ---------------------------------------------------------------------------
# window generator -- mirrors inference.py's per-window construction
# ---------------------------------------------------------------------------
def iter_windows(a, b, is_complex, window_starts):
    """Yield (start_idx, win_clean_real, win_noisy_real) for each start in
    window_starts, applying the same scaling / live alignment inference.py does."""
    fs = SAMP_RATE
    if not is_complex:
        # non-flight: scalar amplitude normalisation from the first CALIB_SIZE_SIM
        clean_calib = a[:CALIB_SIZE_SIM]
        noisy_calib = b[:CALIB_SIZE_SIM]
        scale = max(np.max(np.abs(clean_calib)), np.max(np.abs(noisy_calib))) or 1.0
        for s in window_starts:
            wc = a[s:s + WINDOW_SIZE].astype(np.float64) / scale
            wn = b[s:s + WINDOW_SIZE].astype(np.float64) / scale
            yield s, wc, wn
        return

    # flight: initial calibration at absolute sample 0, then trailing-window
    # recalibration every RECAL_INTERVAL with an EWMA'd amplitude scale.
    params = estimate_alignment(a[:CALIB_SIZE_FLIGHT], b[:CALIB_SIZE_FLIGHT], fs, n0=0)
    next_recal = (CALIB_SIZE_FLIGHT - 1) + RECAL_INTERVAL
    for s in window_starts:
        i = s + WINDOW_SIZE - 1  # index of the newest sample in this window
        if i >= next_recal:
            cstart = max(0, i - CALIB_SIZE_FLIGHT + 1)
            prev_scale = params["scale"]
            params = estimate_alignment(a[cstart:i + 1], b[cstart:i + 1], fs, n0=cstart)
            params["scale"] = SCALE_EWMA_ALPHA * params["scale"] + (1 - SCALE_EWMA_ALPHA) * prev_scale
            next_recal = i + RECAL_INTERVAL
        tx_w = a[s:s + WINDOW_SIZE]
        rx_w = b[s:s + WINDOW_SIZE]
        wc = apply_alignment(tx_w, params, n_start_idx=s).real
        wn = (rx_w - params["dc_hat"]).real
        scale = params["scale"]
        if scale > 0:
            wc = wc / scale
            wn = wn / scale
        yield s, wc, wn


def all_window_starts(n_samples, cap=None, spread=False):
    n_windows = 1 + (n_samples - WINDOW_SIZE) // STRIDE
    if cap is None or n_windows <= cap:
        idx = np.arange(n_windows)
    elif spread:
        idx = np.unique(np.linspace(0, n_windows - 1, cap).astype(int))
    else:
        idx = np.arange(cap)
    return idx * STRIDE


# ---------------------------------------------------------------------------
def run_windows(a, b, is_complex, starts, tf):
    """Filter every window in `starts` at fixed threshold_factor tf. Returns a
    dict of per-window arrays."""
    snr_imp, sig_loss, corr, e_ratio = [], [], [], []
    for _, wc, wn in iter_windows(a, b, is_complex, starts):
        if len(wn) < WINDOW_SIZE:
            break
        filt = denoiser.apply_filter(wn, tf)
        snr_imp.append(snr_db(wc, filt) - snr_db(wc, wn))
        sig_loss.append(np.log1p(np.mean((filt - wc) ** 2)))
        c = np.corrcoef(filt, wc)[0, 1]
        corr.append(0.0 if np.isnan(c) else c)
        e_ratio.append(np.sum(filt ** 2) / (np.sum(wn ** 2) + 1e-12))
    return {k: np.asarray(v) for k, v in
            dict(snr_improvement=snr_imp, signal_loss=sig_loss,
                 correlation=corr, energy_ratio=e_ratio).items()}


def do_sweep(a, b, is_complex):
    n = len(a)
    starts = all_window_starts(n, cap=SWEEP_MAX_WINDOWS, spread=True)
    print(f"[sweep] {DATA_SOURCE}: {len(SWEEP_RANGE)} threshold_factors x {len(starts)} windows")
    rows = []
    for j, tf in enumerate(SWEEP_RANGE):
        r = run_windows(a, b, is_complex, starts, tf)
        rows.append({
            "threshold_factor": tf,
            "snr_improvement": r["snr_improvement"].mean(),
            "signal_loss": r["signal_loss"].mean(),
            "correlation": r["correlation"].mean(),
            "energy_ratio": r["energy_ratio"].mean(),
        })
        if j % 10 == 0:
            print(f"  {j + 1}/{len(SWEEP_RANGE)}  tf={tf:.3f}  "
                  f"snr_imp={rows[-1]['snr_improvement']:.4f}  "
                  f"e_ratio={rows[-1]['energy_ratio']:.3f}")
    df = pd.DataFrame(rows)
    ok = df["energy_ratio"] >= DEGENERATE_ENERGY_RATIO
    if not ok.any():
        raise RuntimeError("every threshold_factor is degenerate -- widen/lower SWEEP_RANGE")

    cand = df[ok]
    best = cand.loc[cand["snr_improvement"].idxmax()]
    in_range = cand[cand["threshold_factor"] <= 2.5]
    best_ir = in_range.loc[in_range["snr_improvement"].idxmax()] if len(in_range) else None

    out = os.path.join(DATA_DIR, f"{TS}_{RUN_TAG}_static_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\n[sweep] excluded {int((~ok).sum())}/{len(df)} degenerate thresholds")
    print(f"[sweep] BEST (unconstrained): threshold_factor={best['threshold_factor']:.4f}  "
          f"snr_improvement={best['snr_improvement']:.4f} dB  "
          f"signal_loss={best['signal_loss']:.4f}  correlation={best['correlation']:.4f}")
    if best_ir is not None:
        print(f"[sweep] BEST (<=2.5, RL knob range): threshold_factor={best_ir['threshold_factor']:.4f}  "
              f"snr_improvement={best_ir['snr_improvement']:.4f} dB")
    print(f"[sweep] wrote {out}")
    print(f"\n>>> set BEST_TF[{DATA_SOURCE!r}] = {best['threshold_factor']:.4f} and rerun with MODE='apply'")


def do_apply(a, b, is_complex):
    tf = TF_OVERRIDE if TF_OVERRIDE is not None else BEST_TF.get(DATA_SOURCE)
    if tf is None:
        raise SystemExit(f"BEST_TF[{DATA_SOURCE!r}] is None and no tf CLI arg given -- "
                         f"run MODE='sweep' first, then pass the winner as the 3rd arg or paste it into BEST_TF")
    n = len(a)
    # spread=True: when APPLY_MAX_WINDOWS caps below the true window count
    # (flight), sample evenly across the whole file rather than taking only the
    # leading windows. For the sim signals the count is below the cap so every
    # window is used regardless.
    starts = all_window_starts(n, cap=APPLY_MAX_WINDOWS, spread=True)
    print(f"[apply] {DATA_SOURCE}: filtering {len(starts)} windows at threshold_factor={tf:.4f}")
    r = run_windows(a, b, is_complex, starts, tf)
    si = r["snr_improvement"]
    running = np.cumsum(si) / np.arange(1, len(si) + 1)

    res = pd.DataFrame({
        "window": [f"({s}, {s + WINDOW_SIZE - 1})" for s in starts[:len(si)]],
        "threshold_factor": tf,
        "snr_improvement": si,
        "signal_loss": r["signal_loss"],
        "correlation": r["correlation"],
        "running_mean": running,
    })
    out = os.path.join(DATA_DIR, f"{TS}_{RUN_TAG}_static_results_tf{tf:.3f}.csv")
    res.to_csv(out, index=False)

    def ms(x):
        return f"{x.mean():.4f} +/- {x.std():.4f}"
    print(f"\n[apply] {DATA_SOURCE}  n_windows={len(si)}  threshold_factor={tf:.4f}")
    print(f"        snr_improvement (dB) : {ms(si)}")
    print(f"        signal_loss          : {ms(r['signal_loss'])}")
    print(f"        correlation          : {ms(r['correlation'])}")
    print(f"[apply] wrote {out}")


def main():
    t0 = time.time()
    print(f"[INFO] MODE={MODE}  DATA_SOURCE={DATA_SOURCE}  csv={CSV_PATH}")
    a, b, is_complex = load_source()
    print(f"[INFO] loaded {len(a)} samples (complex={is_complex})")
    if MODE == "sweep":
        do_sweep(a, b, is_complex)
    elif MODE == "apply":
        do_apply(a, b, is_complex)
    else:
        raise ValueError(f"MODE must be 'sweep' or 'apply', got {MODE!r}")
    print(f"[INFO] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
