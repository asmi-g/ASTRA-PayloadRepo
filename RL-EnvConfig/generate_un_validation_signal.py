#!/usr/bin/env python3
"""
generate_un_validation_signal.py

Generates a held-out validation signal from the SAME noise generation path
UN was trained on (NoiseReductionEnv._generate_signals_from_noise_model:
block-bootstrap of the real flight-extracted residual, rescaled to the real
noise-to-signal severity ratio -- see task 9), analogous to how
simulated_signal_match_hz.csv validates OFT against its own (synthetic
white+blue+burst) training distribution.

This is NOT the same data OFT/the reward sweep were tuned against, and NOT
literally any specific training episode's draw (fresh random block-bootstrap
draw, different seed) -- it's an in-distribution held-out check for UN
specifically, to separate "does UN generalize poorly because of a train/test
distribution mismatch" from "even matched to its own training distribution,
real noise severity limits what the filter can do."

Usage:
    python generate_un_validation_signal.py [signal_length] [output_path] [seed]
"""

import sys
import os
import numpy as np
import pandas as pd
from astra_rev1.envs import NoiseReductionEnv

DEFAULT_SIGNAL_LENGTH = 500_000  # matches simulated_signal_match_hz.csv for comparability
DEFAULT_SEED = 123  # different from generate_simulated_signal.py's 42 -- genuinely a
                     # different draw, not a rerun of the same "fixed validation set" idea
                     # applied to a different noise model
NOISE_MODEL_PATH_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/noise_model_fs1.pkl")
)


def main(signal_length=DEFAULT_SIGNAL_LENGTH, out_path=None, seed=DEFAULT_SEED,
         noise_model_path=NOISE_MODEL_PATH_DEFAULT):
    np.random.seed(seed)

    env = NoiseReductionEnv(signal_length=signal_length, window_size=1000, mode="train",
                             noise_model_path=noise_model_path)
    # Bypass the constructor's signal_length randomization (see custom_env_022025.py --
    # kept intentionally for training variety, but we want an exact, known-length
    # validation file here) and generate directly at the requested length.
    env.signal_length = signal_length
    clean, noisy = env._generate_signals_from_noise_model()

    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20))

    print(f"[INFO] signal_length = {signal_length}")
    print(f"[INFO] noise_model_path = {noise_model_path}")
    print(f"[INFO] seed = {seed}")
    print(f"[INFO] clean std = {clean.std():.6g}, noisy std = {noisy.std():.6g}, "
          f"noise std = {noise.std():.6g}")
    print(f"[INFO] noise/clean ratio = {noise.std()/clean.std():.3f} "
          f"(real flight ratio target: ~47.8)")
    print(f"[INFO] SNR (raw) = {snr:.2f} dB")

    out_df = pd.DataFrame({
        "Index": np.arange(signal_length),
        "Time": np.arange(signal_length) / 1_000_000,
        "Noisy Signal": noisy,
        "Clean Signal": clean,
    })

    if out_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.normpath(os.path.join(script_dir, "../Data/simulated_signal_un_noise_model.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path}")

    return out_df


if __name__ == "__main__":
    length_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SIGNAL_LENGTH
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    seed_arg = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SEED
    main(length_arg, out_arg, seed_arg)
