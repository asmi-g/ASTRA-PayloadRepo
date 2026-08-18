import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from astra_rev1.envs import NoiseReductionEnv


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 10

# Threshold values corresponding to the continuous action range
THRESHOLDS = np.linspace(0.5, 2.5, 21)

# Weight combinations to test
ALPHAS = [0.25, 0.5, 0.75, 1.0]
BETAS  = [0.75, 1.0, 1.25, 1.5]
GAMMAS = [0.0, 0.25, 0.5, 0.75]

# Number of signal windows to test
N_WINDOWS = 1000

np.random.seed(42)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = NoiseReductionEnv(
    window_size=WINDOW_SIZE,
    mode="train"
)

# Generate signal using YOUR environment's generator
clean_signal, noisy_signal = env._generate_signals()


# Apply the same training normalization as your environment
scale = np.max(np.abs(clean_signal))

if scale > 0:
    clean_signal = clean_signal / scale
    noisy_signal = noisy_signal / scale


# ============================================================
# SELECT WINDOWS
# ============================================================

max_start = len(clean_signal) - WINDOW_SIZE

window_starts = np.random.choice(
    max_start,
    size=min(N_WINDOWS, max_start),
    replace=False
)


# ============================================================
# CALCULATE REWARD COMPONENTS
# ============================================================

results = []

for start in window_starts:

    clean_window = clean_signal[
        start:start + WINDOW_SIZE
    ]

    noisy_window = noisy_signal[
        start:start + WINDOW_SIZE
    ]

    for threshold in THRESHOLDS:

        # Use the SAME filtering implementation as your env
        filtered_window = env.denoiser.apply_filter(
            noisy_window,
            threshold
        )

        # -------------------------
        # SNR
        # -------------------------

        snr_raw = env.denoiser._snr(
            clean_window,
            noisy_window
        )

        snr_filtered = env.denoiser._snr(
            clean_window,
            filtered_window
        )

        snr_improvement = (
            snr_filtered - snr_raw
        )

        # -------------------------
        # Signal loss
        # -------------------------

        signal_loss = np.log1p(
            np.mean(
                (filtered_window - clean_window) ** 2
            )
        )

        # -------------------------
        # Correlation
        # -------------------------

        if (
            np.std(clean_window) < 1e-12
            or
            np.std(filtered_window) < 1e-12
        ):
            correlation = 0.0
        else:
            correlation = np.corrcoef(
                filtered_window,
                clean_window
            )[0, 1]

            if not np.isfinite(correlation):
                correlation = 0.0

        results.append([
            start,
            threshold,
            snr_improvement,
            signal_loss,
            correlation
        ])


results = np.array(results)


# ============================================================
# SWEEP REWARD WEIGHTS
# ============================================================

weight_results = []

for alpha, beta, gamma in product(
    ALPHAS,
    BETAS,
    GAMMAS
):

    # Columns:
    # 0 = window
    # 1 = threshold
    # 2 = SNR improvement
    # 3 = signal loss
    # 4 = correlation

    reward = (
        alpha * results[:, 2]
        - beta * results[:, 3]
        + gamma * results[:, 4]
    )

    # Find best threshold separately for each window
    best_rewards = []
    best_thresholds = []

    for start in window_starts:

        mask = results[:, 0] == start

        window_rewards = reward[mask]
        window_thresholds = results[mask, 1]

        best_idx = np.argmax(window_rewards)

        best_rewards.append(
            window_rewards[best_idx]
        )

        best_thresholds.append(
            window_thresholds[best_idx]
        )

    weight_results.append([
        alpha,
        beta,
        gamma,
        np.mean(best_rewards),
        np.mean(best_thresholds),
        np.std(best_thresholds)
    ])


# ============================================================
# DISPLAY RESULTS
# ============================================================

weight_results = np.array(weight_results)

# Sort by average best reward
weight_results = weight_results[
    np.argsort(weight_results[:, 3])[::-1]
]

print("\n======================================")
print("TOP REWARD WEIGHT COMBINATIONS")
print("======================================")

print(
    " alpha   beta   gamma   avg_reward   "
    "avg_threshold   threshold_std"
)

for row in weight_results[:10]:

    print(
        f"{row[0]:6.2f} "
        f"{row[1]:6.2f} "
        f"{row[2]:6.2f} "
        f"{row[3]:11.4f} "
        f"{row[4]:14.3f} "
        f"{row[5]:14.3f}"
    )


# ============================================================
# CURRENT REWARD
# ============================================================

current = weight_results[
    (weight_results[:, 0] == 0.5) &
    (weight_results[:, 1] == 1.25) &
    (weight_results[:, 2] == 0.25)
]

print("\n======================================")
print("CURRENT REWARD")
print("======================================")

print(
    "alpha=0.5, beta=1.25, gamma=0.25"
)

if len(current) > 0:

    print(
        f"Average best reward: "
        f"{current[0, 3]:.4f}"
    )

    print(
        f"Average best threshold: "
        f"{current[0, 4]:.3f}"
    )

    print(
        f"Threshold std: "
        f"{current[0, 5]:.3f}"
    )