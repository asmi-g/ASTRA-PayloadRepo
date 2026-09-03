#!/usr/bin/env python3
"""
generate_simulated_signal_oan.py   (my-list item 2)
==================================================
Writes Data/simulated_signal_oan.csv: a held-out evaluation signal drawn from
the SAME generator the OAN model trains on --
NoiseReductionEnv.generate_accurate_signals() with default parameters (white +
true 1/f pink normalised to the noise-power budget, Poisson(0.3) FRED-shaped
solar-radio-burst episodes, short impulsive bursts). train_sb.py calls that
method with no arguments, so this file uses its defaults too; only the RNG
seed differs, making this a genuine fresh draw rather than a replay of a
training episode.

SEED SEARCH: the flare-onset count per signal is Poisson(0.3), so ~74% of
single draws contain no flare at all. Since the point of the OAN eval signal
is to exercise the non-stationary / burst response, this script scans a fixed
list of candidate seeds and picks the first whose realized signal contains at
least MIN_FLARES flare stretches (rolling-RMS excursions well above the
white/pink baseline). The chosen seed is printed and should be pinned.

Output columns (Index, Time, Noisy Signal, Clean Signal) match the other
simulated signals, so inference.py (DATA_SOURCE="oan_sim"),
static_filter_realtime.py and the signal-analysis scripts consume it
unchanged.

Usage:  cd RL-EnvConfig && python generate_simulated_signal_oan.py
        [signal_length] [output_path] [seed]
        # passing an explicit seed skips the search and uses it as-is
"""

import os
import sys

import numpy as np
import pandas as pd

from astra_rev1.envs import NoiseReductionEnv

DEFAULT_SIGNAL_LENGTH = 500_000   # matches simulated_signal_match_hz.csv for comparability
WINDOW_SIZE = 1000                # sets the default flare length (30 * window_size)
MIN_FLARES = 1                    # require at least this many realized flare stretches
# Scanned in order; the first accepted is used. seed 20260904 is the first in
# this list whose 500k-sample draw contains a solar-flare episode (4 runs,
# peak/median rolling RMS 3.19, 1.4% of windows elevated). Pinned as the
# de-facto default via the ordering.
CANDIDATE_SEEDS = list(range(20260904, 20260961)) + list(range(20260901, 20260904))


def _flare_stats(noise):
    """Rolling 1000-sample noise RMS diagnostics. Empirically (scan over 60
    seeds) a signal with a solar-flare episode shows peak/median rolling RMS
    ~3.0-3.7 and >~3% of windows above 2x median; a flare-free signal
    (white + 1/f pink + short bursts only) stays at peak/median ~1.5-1.8 and
    ~0% elevated windows. The FRED envelope's nominal 6x gain on the noise std
    averages down to ~3x over the 1000-sample RMS window, hence the modest
    peak ratio."""
    c = np.cumsum(np.insert(noise ** 2, 0, 0.0))
    rms = np.sqrt((c[WINDOW_SIZE:] - c[:-WINDOW_SIZE]) / WINDOW_SIZE)
    med = np.median(rms)
    flaring = rms > 2.0 * med
    n_runs = int(np.sum(np.diff(flaring.astype(np.int8)) == 1) + (1 if flaring[0] else 0))
    peak_ratio = float(rms.max() / (med + 1e-12))
    flaring_frac = float(flaring.mean())
    return n_runs, peak_ratio, flaring_frac, rms


def _generate(seed, signal_length):
    np.random.seed(seed)
    env = NoiseReductionEnv(signal_length=signal_length, window_size=WINDOW_SIZE, mode="train")
    env.signal_length = signal_length            # pin: bypass per-episode length redraw
    clean, noisy = env.generate_accurate_signals()  # defaults == training call
    return clean, noisy


def main(signal_length=DEFAULT_SIGNAL_LENGTH, out_path=None, seed=None):
    if seed is not None:
        seeds_to_try = [seed]
        print(f"[INFO] explicit seed {seed} given -- skipping flare search")
    else:
        seeds_to_try = CANDIDATE_SEEDS

    chosen = None
    for s in seeds_to_try:
        clean, noisy = _generate(s, signal_length)
        n_runs, peak_ratio, flaring_frac, _ = _flare_stats(noisy - clean)
        # flare present: sustained rolling-RMS elevation (peak/median >~2.8 AND
        # >=1% of windows elevated). Flare-free draws sit at peak ~1.6, ~0%.
        ok = (n_runs >= MIN_FLARES) and (peak_ratio >= 2.8) and (flaring_frac >= 0.01)
        print(f"[scan] seed {s}: flare_runs={n_runs} peak/median_RMS={peak_ratio:.2f} "
              f"flaring_frac={flaring_frac:.3f} -> {'ACCEPT' if ok else 'skip'}")
        if ok or seed is not None:
            chosen = s
            break

    if chosen is None:
        raise SystemExit(f"no candidate seed in {seeds_to_try[0]}..{seeds_to_try[-1]} produced "
                         f">= {MIN_FLARES} flare(s); widen CANDIDATE_SEEDS or raise n_flares")

    clean, noisy = _generate(chosen, signal_length)
    noise = noisy - clean
    n_runs, peak_ratio, flaring_frac, _ = _flare_stats(noise)
    snr = 10 * np.log10((np.mean(clean ** 2) + 1e-20) / (np.mean(noise ** 2) + 1e-20))

    print(f"\n[INFO] CHOSEN SEED   = {chosen}   (pin this)")
    print(f"[INFO] signal_length = {signal_length}")
    print(f"[INFO] clean std {clean.std():.6g} | noisy std {noisy.std():.6g} | noise std {noise.std():.6g}")
    print(f"[INFO] noise/clean ratio = {noise.std() / clean.std():.4f}")
    print(f"[INFO] SNR (raw)     = {snr:.2f} dB")
    print(f"[INFO] flare: {n_runs} run(s), peak/median rolling RMS = {peak_ratio:.2f}, "
          f"{flaring_frac:.1%} of windows elevated")

    out_df = pd.DataFrame({
        "Index": np.arange(signal_length),
        "Time": np.arange(signal_length) / 1_000_000,
        "Noisy Signal": noisy,
        "Clean Signal": clean,
    })
    if out_path is None:
        out_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../Data/simulated_signal_oan.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")
    return out_df


if __name__ == "__main__":
    length_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SIGNAL_LENGTH
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(length_arg, out_arg, seed_arg)
