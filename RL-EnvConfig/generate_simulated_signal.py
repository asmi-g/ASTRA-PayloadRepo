#!/usr/bin/env python3
"""
generate_simulated_signal.py
============================
Writes a held-out evaluation signal drawn from the SAME generator a given model
trains on. One script, one --mode switch; merges the former
generate_simulated_signal_opbn.py and generate_simulated_signal_pfn.py.

    --mode opbn -> Data/simulated_signal_opbn.csv
        NoiseReductionEnv.generate_signals_pink_bursts() with default params --
        an always-on WHITE floor plus true-1/f (pink) energy appearing ONLY
        inside a Poisson-count (mean 0.5) of short additive burst episodes, plus
        short +/- impulse hits. No continuous pink, no solar-flare envelope.
        Because n_bursts ~ Poisson(0.5) means ~40% of single draws contain no
        pink burst at all, this mode scans a fixed seed list and takes the first
        draw that actually contains a burst episode. The chosen seed is printed.

    --mode pfn  -> Data/simulated_signal_pfn.csv
        NoiseReductionEnv.generate_accurate_flight_signals() with default params,
        which parametrically rebuilds the flight_signal_1 residual from
        Data/noise_model_fs1.pkl (AR(50) colour + Bernoulli-Gaussian heavy-tailed
        innovations calibrated to the measured AR-innovation kurtosis + the
        measured rms_profile drift envelope, rescaled to the real noise/clean
        severity ratio). Uses a fixed representative seed (no scan).

train_sb.py calls each generator method with no arguments, so this file uses its
defaults too; only the RNG seed differs.

Output columns (Index, Time, Noisy Signal, Clean Signal) match the other
simulated signals, so inference.py (DATA_SOURCE="opbn_sim" / "pfn_sim"),
static_filter_realtime.py and the analysis scripts consume them unchanged.

Usage:  cd RL-EnvConfig && python generate_simulated_signal.py --mode {opbn,pfn}
        [--length N] [--out PATH] [--seed S]   # explicit --seed skips any scan
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from astra_rev1.envs import NoiseReductionEnv

DEFAULT_SIGNAL_LENGTH = 500_000
WINDOW_SIZE = 1000

# --- OPBN: sets default burst_len = 20 * window_size. Scanned in order; first
# accepted is used. seed 20260944 is the first draw in range(20260910, 20260970)
# with a clear pink-burst episode (4 runs, peak/median rolling RMS 3.49, 1.6% of
# windows elevated). Pinned via ordering.
OPBN_CANDIDATE_SEEDS = [20260944] + [s for s in range(20260910, 20260970) if s != 20260944]

# --- PFN: with envelope_mode="realrate" (the default), each draw takes a random
# contiguous slice of the flight rms_profile, so the realised noise kurtosis and
# drift depend on where that slice lands. Across seeds 20260902..20260915 the
# finished-noise excess kurtosis is 9-14 and drift (p99/p1) is 1.5-2.2 for 5 of
# 6 -- right on the AR-innovation target (~14) and the per-segment drift ratio
# (~1.7) -- with one spiky-slice outlier (~150 / ~5.6). seed 20260907
# (kurt 14.4, drift 2.2) is a representative pick and is the default.
PFN_DEFAULT_SEED = 20260907        # != train_sb.py SEED and != the OAN eval seed
NOISE_MODEL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../Data/noise_model_fs1.pkl"))


def _rolling_rms(noise):
    c = np.cumsum(np.insert(noise ** 2, 0, 0.0))
    return np.sqrt((c[WINDOW_SIZE:] - c[:-WINDOW_SIZE]) / WINDOW_SIZE)


def _burst_stats(noise):
    """Rolling 1000-sample noise RMS. A pink-burst episode (pink std = 4x white
    std over ~20k samples) lifts local RMS to ~3-4x the flat-white baseline;
    a burst-free draw sits at peak/median ~1.5-1.8."""
    rms = _rolling_rms(noise)
    med = np.median(rms)
    elevated = rms > 2.0 * med
    n_runs = int(np.sum(np.diff(elevated.astype(np.int8)) == 1) + (1 if elevated[0] else 0))
    return n_runs, float(rms.max() / (med + 1e-12)), float(elevated.mean())


def _excess_kurtosis(noise):
    x = noise - noise.mean()
    return float((x ** 4).mean() / (x.std() ** 4) - 3.0)


# --------------------------------------------------------------------------- OPBN

def _opbn_generate(seed, signal_length):
    np.random.seed(seed)
    env = NoiseReductionEnv(signal_length=signal_length, window_size=WINDOW_SIZE,
                            mode="train", signal_gen="pink_bursts")
    env.signal_length = signal_length          # pin: bypass per-episode length redraw
    return env.generate_signals_pink_bursts()  # defaults == training call


def _opbn_run(signal_length, seed):
    seeds = [seed] if seed is not None else OPBN_CANDIDATE_SEEDS
    if seed is not None:
        print(f"[INFO] explicit seed {seed} -- skipping burst scan")

    chosen = None
    for s in seeds:
        clean, noisy = _opbn_generate(s, signal_length)
        n_runs, peak_ratio, frac = _burst_stats(noisy - clean)
        ok = (n_runs >= 1) and (peak_ratio >= 2.8) and (frac >= 0.01)
        print(f"[scan] seed {s}: burst_runs={n_runs} peak/median_RMS={peak_ratio:.2f} "
              f"elevated_frac={frac:.3f} -> {'ACCEPT' if ok else 'skip'}")
        if ok or seed is not None:
            chosen = s
            break
    if chosen is None:
        raise SystemExit(f"no seed in {seeds[0]}..{seeds[-1]} produced a pink-burst episode; "
                         f"widen OPBN_CANDIDATE_SEEDS or raise n_bursts")

    clean, noisy = _opbn_generate(chosen, signal_length)
    noise = noisy - clean
    n_runs, peak_ratio, frac = _burst_stats(noise)
    snr = 10 * np.log10((np.mean(clean ** 2) + 1e-20) / (np.mean(noise ** 2) + 1e-20))

    print(f"\n[INFO] CHOSEN SEED   = {chosen}   (pin this)")
    print(f"[INFO] signal_length = {signal_length}")
    print(f"[INFO] clean std {clean.std():.6g} | noisy std {noisy.std():.6g} | noise std {noise.std():.6g}")
    print(f"[INFO] noise/clean ratio = {noise.std() / clean.std():.4f}")
    print(f"[INFO] SNR (raw)     = {snr:.2f} dB   noise excess kurtosis = {_excess_kurtosis(noise):.1f}")
    print(f"[INFO] pink-burst: {n_runs} run(s), peak/median rolling RMS = {peak_ratio:.2f}, "
          f"{frac:.1%} of windows elevated")
    return clean, noisy


# ---------------------------------------------------------------------------- PFN

def _pfn_run(signal_length, seed):
    seed = PFN_DEFAULT_SEED if seed is None else seed
    np.random.seed(seed)

    env = NoiseReductionEnv(signal_length=signal_length, window_size=WINDOW_SIZE,
                            mode="train", noise_model_path=NOISE_MODEL_PATH)
    env.signal_length = signal_length   # pin: bypass the per-episode length redraw

    clean, noisy = env.generate_accurate_flight_signals(seed=seed)

    noise = noisy - clean
    snr = 10 * np.log10((np.mean(clean ** 2) + 1e-20) / (np.mean(noise ** 2) + 1e-20))
    rms = _rolling_rms(noise)
    drift_ratio = float(np.percentile(rms, 99) / max(np.percentile(rms, 1), 1e-12))

    print(f"[INFO] signal_length   = {signal_length}")
    print(f"[INFO] seed            = {seed}")
    print(f"[INFO] noise_model     = {NOISE_MODEL_PATH}")
    print(f"[INFO] clean std {clean.std():.6g} | noisy std {noisy.std():.6g} | noise std {noise.std():.6g}")
    print(f"[INFO] noise/clean ratio = {noise.std() / clean.std():.4f}  (pkl target ~47.5)")
    print(f"[INFO] noise excess kurtosis = {_excess_kurtosis(noise):.1f}")
    print(f"[INFO] rolling-RMS drift ratio (p99/p1) = {drift_ratio:.1f}")
    print(f"[INFO] SNR (raw)       = {snr:.2f} dB")
    return clean, noisy


# --------------------------------------------------------------------------- main

MODES = {
    "opbn": (_opbn_run, "simulated_signal_opbn.csv"),
    "pfn":  (_pfn_run,  "simulated_signal_pfn.csv"),
}


def main(mode, signal_length=DEFAULT_SIGNAL_LENGTH, out_path=None, seed=None):
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    runner, default_name = MODES[mode]

    clean, noisy = runner(signal_length, seed)

    out_df = pd.DataFrame({
        "Index": np.arange(signal_length),
        "Time": np.arange(signal_length) / 1_000_000,
        "Noisy Signal": noisy,
        "Clean Signal": clean,
    })
    if out_path is None:
        out_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../Data", default_name))
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")
    return out_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=sorted(MODES),
                   help="which training generator to draw the eval signal from")
    p.add_argument("--length", type=int, default=DEFAULT_SIGNAL_LENGTH,
                   help=f"signal length in samples (default {DEFAULT_SIGNAL_LENGTH})")
    p.add_argument("--out", default=None,
                   help="output CSV path (default Data/simulated_signal_<mode>.csv)")
    p.add_argument("--seed", type=int, default=None,
                   help="explicit RNG seed; for --mode opbn this skips the burst scan")
    args = p.parse_args()
    main(args.mode, args.length, args.out, args.seed)
