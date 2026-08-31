"""
Reward Function Sweep Tool (v2 — with Pareto frontier analysis)
=================================================================
Tests candidate reward functions against a wavelet-threshold denoising filter
WITHOUT training an RL model.

WHAT'S NEW vs v1:
- Computes the Pareto frontier of (snr_improvement, signal_loss) directly from
  the threshold_factor sweep, independent of any reward function. This is the
  ground truth for "what tradeoffs are even achievable."
- Checks whether each reward function's chosen threshold_factor lands ON that
  frontier (efficient) or is DOMINATED (strictly worse on both axes than some
  other achievable point — a sign the reward function is defective).
- Adds a weight-sweep utility: for a linear reward of the form
      w1 * snr_improvement - w2 * signal_loss (+ w3 * correlation)
  it sweeps the weight ratio and records which threshold_factor gets picked
  at each ratio. This traces out which part of the frontier is *reachable*
  by any linear weighting — concave regions of the frontier are NOT reachable
  by any weight combo, which tells you if you need a non-linear reward shape
  (e.g. a hard floor / ratio / log term) instead of just retuning weights.

HOW TO USE:
1. Load your clean and noisy signals into `clean_signal` and `noisy_signal`.
2. Define candidate reward functions in REWARD_CONFIGS.
3. Run the script.
"""

import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD YOUR SIGNAL DATA HERE
# ============================================================
# Reward-function tuning is done purely against our own simulated noise
# model (simulated_signal_match_hz.csv) -- deliberately NOT the flight
# signal, since this sweep represents a decision that would have been made
# before the flight signal was ever obtained.

def load_simulated_signal():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(script_dir, "../Data/simulated_signal_match_hz.csv"))
    df = pd.read_csv(csv_path)
    clean = df["Clean Signal"].to_numpy(dtype=np.float64)
    noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
    return clean, noisy

clean_signal, noisy_signal = load_simulated_signal()


# ============================================================
# 2. WAVELET FILTER (matches StatelessDenoisingEnv.apply_filter in
#    custom_env_022025.py exactly -- same wavelet/level/mode, same
#    VisuShrink-style threshold scaling, only the detail coefficients are
#    thresholded (not the approximation coefficients), same mean-preserving
#    readjustment. A sweep against a different filter than the one the RL
#    agent actually controls wouldn't tell you anything about the agent's
#    real reward landscape.)
# ============================================================

def apply_filter(signal, threshold_factor, wavelet='db4', level=5):
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="periodization")
    # Floored above 0 -- see custom_env_022025.py apply_filter for why an
    # unfloored sigma=0 (from heavily-quantized input) makes every
    # threshold_factor collapse to an identical degenerate all-zero output.
    sigma = max(np.median(np.abs(coeffs[-1])) / 0.6745, 1e-8) if coeffs[-1].size else 1e-8
    lam = threshold_factor * sigma * np.sqrt(2 * np.log(max(n, 2)))
    cA, details = coeffs[0], coeffs[1:]
    details = [pywt.threshold(c, lam, mode="soft") for c in details]
    y = pywt.waverec([cA] + details, wavelet, mode="periodization")[:n]
    y += (np.mean(x) - np.mean(y))
    return np.nan_to_num(y, copy=False)


# ============================================================
# 3. METRICS
# ============================================================

def calculate_snr(clean, signal):
    noise = signal - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

def calculate_signal_loss(clean, filtered):
    # Matches StatelessDenoisingEnv.step's signal_loss term exactly
    # (log1p-compressed MSE, not raw MSE).
    return np.log1p(np.mean((filtered - clean)**2))

def calculate_mae_normalized(clean, filtered):
    return np.mean(np.abs(filtered - clean)) / (np.std(clean) + 1e-10)

def calculate_correlation(clean, filtered):
    if np.std(filtered) < 1e-10:
        return 0.0
    return np.corrcoef(clean, filtered)[0, 1]

def calculate_output_energy_ratio(raw, filtered):
    raw_energy = np.sum(raw ** 2) + 1e-10
    return np.sum(filtered ** 2) / raw_energy


# ============================================================
# 4. REWARD FUNCTIONS TO TEST
# ============================================================

REWARD_CONFIGS = {
    "original": lambda m: (
        5 * m["snr_improvement"]
    ),
    "twoterm": lambda m: (
        1 * m["snr_improvement"] - 1 * m["signal_loss"]
        ),
    "threeterm_v1": lambda m: (
        1 * m["snr_improvement"] - 1 * m["signal_loss"] + 1 * m["correlation"]
    ),
    "threeterm_final": lambda m: (  # matches StatelessDenoisingEnv.step's live reward
        1 * m["snr_improvement"] - 1.25 * m["signal_loss"] + 0.25 * m["correlation"]
    ),
    # Add your own candidates here:
    # "my_new_reward": lambda m: (...),
}


# ============================================================
# 5. SWEEP LOGIC
# ============================================================

def sweep_threshold_factor(clean, noisy, threshold_range=None, n_points=200, window_size=1000):
    # threshold_factor default range matches the RL agent's actual achievable
    # range: NoiseReductionEnv maps action in [-1,1] to threshold_factor in
    # [0.05, 2.0] (see custom_env_022025.py StatelessDenoisingEnv.step), so
    # sweeping outside that range would surface "optimal" points the agent
    # can never actually select.
    if threshold_range is None:
        threshold_range = np.linspace(0.05, 2.0, n_points)

    # Filter window-by-window (matching window_size=1000, the size the RL
    # agent actually filters per step -- see train_sb.py), not on the whole
    # signal at once. apply_filter's threshold includes a sqrt(2*log(n))
    # term, so filtering the full 500k-sample signal in one shot would use a
    # ~38% larger effective threshold multiplier than the agent ever
    # experiences per-window, shifting where the "optimal" threshold_factor
    # appears to be. Non-overlapping windows (not the training stride of 10)
    # are used here purely for sweep speed; this still captures the same
    # per-window threshold-scaling behavior the agent is subject to.
    n_windows = len(clean) // window_size

    rows = []
    for tf in threshold_range:
        snr_improvements, signal_losses, maes, correlations, energy_ratios = [], [], [], [], []
        for w in range(n_windows):
            c = clean[w * window_size:(w + 1) * window_size]
            nz = noisy[w * window_size:(w + 1) * window_size]
            filtered = apply_filter(nz, threshold_factor=tf)
            snr_improvements.append(calculate_snr(c, filtered) - calculate_snr(c, nz))
            signal_losses.append(calculate_signal_loss(c, filtered))
            maes.append(calculate_mae_normalized(c, filtered))
            correlations.append(calculate_correlation(c, filtered))
            energy_ratios.append(calculate_output_energy_ratio(nz, filtered))
        rows.append({
            "threshold_factor": tf,
            "snr_improvement": np.mean(snr_improvements),
            "signal_loss": np.mean(signal_losses),
            "mae": np.mean(maes),
            "correlation": np.mean(correlations),
            "energy_ratio": np.mean(energy_ratios),
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. PARETO FRONTIER (ground truth, no reward function involved)
# ============================================================
# A point is on the frontier if no other point has BOTH higher
# snr_improvement AND lower signal_loss (i.e. it is not dominated).

def compute_pareto_frontier(sweep_df):
    snr = sweep_df["snr_improvement"].values
    loss = sweep_df["signal_loss"].values
    n = len(sweep_df)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        # dominated if some other point j has snr[j] >= snr[i] and loss[j] <= loss[i],
        # with at least one strict inequality
        better_or_equal_snr = snr >= snr[i]
        better_or_equal_loss = loss <= loss[i]
        strictly_better = (snr > snr[i]) | (loss < loss[i])
        dominators = better_or_equal_snr & better_or_equal_loss & strictly_better
        is_dominated[i] = dominators.any()

    out = sweep_df.copy()
    out["on_pareto_frontier"] = ~is_dominated
    return out


def nearest_frontier_distance(sweep_df_with_frontier, threshold_factor):
    """How far (in normalized snr/loss space) is a given threshold_factor's
    point from the nearest point actually on the frontier. 0 = on frontier."""
    row = sweep_df_with_frontier.iloc[
        (sweep_df_with_frontier["threshold_factor"] - threshold_factor).abs().argmin()
    ]
    if row["on_pareto_frontier"]:
        return 0.0

    frontier = sweep_df_with_frontier[sweep_df_with_frontier["on_pareto_frontier"]]
    snr_span = sweep_df_with_frontier["snr_improvement"].max() - sweep_df_with_frontier["snr_improvement"].min() + 1e-10
    loss_span = sweep_df_with_frontier["signal_loss"].max() - sweep_df_with_frontier["signal_loss"].min() + 1e-10

    d_snr = (frontier["snr_improvement"] - row["snr_improvement"]) / snr_span
    d_loss = (frontier["signal_loss"] - row["signal_loss"]) / loss_span
    dist = np.sqrt(d_snr ** 2 + d_loss ** 2)
    return float(dist.min())


# ============================================================
# 7. EVALUATE REWARD FUNCTIONS AGAINST THE FRONTIER
# ============================================================

def evaluate_reward_configs(sweep_df, reward_configs):
    frontier_df = compute_pareto_frontier(sweep_df)
    results = []
    for name, reward_fn in reward_configs.items():
        rewards = sweep_df.apply(lambda row: reward_fn(row.to_dict()), axis=1)
        best_idx = rewards.idxmax()
        best_row = sweep_df.loc[best_idx]
        tf = best_row["threshold_factor"]

        is_degenerate = best_row["signal_loss"] > 0.8 or best_row["energy_ratio"] < 0.2
        dist_to_frontier = nearest_frontier_distance(frontier_df, tf)
        on_frontier = dist_to_frontier == 0.0

        results.append({
            "reward_function": name,
            "best_threshold_factor": round(tf, 3),
            "snr_improvement": round(best_row["snr_improvement"], 4),
            "signal_loss": round(best_row["signal_loss"], 4),
            "correlation": round(best_row["correlation"], 4),
            "energy_ratio": round(best_row["energy_ratio"], 4),
            "ON_PARETO_FRONTIER": "yes" if on_frontier else "NO (dominated)",
            "normalized_dist_to_frontier": round(dist_to_frontier, 4),
            "DEGENERATE": "YES" if is_degenerate else "no",
        })
    return pd.DataFrame(results), frontier_df


# ============================================================
# 8. WEIGHT SWEEP — what part of the frontier is reachable
#    by a LINEAR reward at all?
# ============================================================
# For reward = w * snr_improvement - (1-w) * signal_loss, sweep w in [0,1]
# and record which threshold_factor is picked at each w. This traces the
# frontier region reachable by linear scalarization. Gaps mean concave
# regions of the frontier that NO weight combination can reach — that's
# a sign you need a non-linear reward term (hard floor, ratio, log, etc.),
# not just different weights.

def sweep_reward_weights(sweep_df, n_weights=101):
    weights = np.linspace(0.0, 1.0, n_weights)
    rows = []
    for w in weights:
        reward = w * sweep_df["snr_improvement"] - (1 - w) * sweep_df["signal_loss"]
        best_idx = reward.idxmax()
        best_row = sweep_df.loc[best_idx]
        rows.append({
            "weight_on_snr": round(w, 3),
            "threshold_factor": round(best_row["threshold_factor"], 3),
            "snr_improvement": round(best_row["snr_improvement"], 4),
            "signal_loss": round(best_row["signal_loss"], 4),
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. PLOTS
# ============================================================

def plot_frontier(frontier_df, results_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    dominated = frontier_df[~frontier_df["on_pareto_frontier"]]
    frontier = frontier_df[frontier_df["on_pareto_frontier"]].sort_values("threshold_factor")

    ax.scatter(dominated["signal_loss"], dominated["snr_improvement"],
               s=10, color="lightgray", label="dominated (achievable but inefficient)")
    ax.plot(frontier["signal_loss"], frontier["snr_improvement"],
            "o-", color="tab:blue", markersize=4, label="Pareto frontier")

    colors = plt.cm.tab10.colors
    for i, row in results_df.iterrows():
        tf = row["best_threshold_factor"]
        match = frontier_df.iloc[(frontier_df["threshold_factor"] - tf).abs().argmin()]
        ax.scatter(match["signal_loss"], match["snr_improvement"],
                   marker="*", s=250, color=colors[i % len(colors)],
                   edgecolor="black", zorder=5, label=row["reward_function"])

    ax.set_xlabel("signal_loss (lower is better)")
    ax.set_ylabel("snr_improvement (higher is better)")
    ax.set_title("Achievable tradeoffs across threshold_factor\n(stars = each reward function's chosen point)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("reward_sweep_frontier.png", dpi=150)
    print("[INFO] Saved plot: reward_sweep_frontier.png")
    plt.show()


def plot_reward_landscapes(sweep_df, reward_configs):
    n = len(reward_configs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, reward_fn) in zip(axes, reward_configs.items()):
        rewards = sweep_df.apply(lambda row: reward_fn(row.to_dict()), axis=1)
        ax.plot(sweep_df["threshold_factor"], rewards, label="reward")
        best_idx = rewards.idxmax()
        ax.axvline(sweep_df.loc[best_idx, "threshold_factor"], color="red",
                   linestyle="--", label="reward-optimal threshold_factor")
        ax.set_title(name)
        ax.set_ylabel("reward")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("threshold_factor")
    plt.tight_layout()
    plt.savefig("reward_sweep_landscapes.png", dpi=150)
    print("[INFO] Saved plot: reward_sweep_landscapes.png")
    plt.show()


def plot_filtered_signals(clean, noisy, results_df, n_samples=400, window_size=1000):
    """One subplot per reward function: clean vs noisy vs filtered-at-that-
    reward-function's-chosen-threshold_factor, overlaid. n_samples limits how
    many points are plotted so the waveform shape stays readable. Filters
    only the first window_size samples (matching the window-based sweep),
    not the whole signal at once -- apply_filter's threshold scales with
    the length of what it's given."""
    n = len(results_df)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    idx = np.arange(min(n_samples, window_size, len(clean)))

    for ax, (_, row) in zip(axes, results_df.iterrows()):
        tf = row["best_threshold_factor"]
        filtered = apply_filter(noisy[:window_size], threshold_factor=tf)

        ax.plot(idx, noisy[idx], color="lightgray", linewidth=0.8, label="noisy")
        ax.plot(idx, clean[idx], color="black", linewidth=1.2, label="clean")
        ax.plot(idx, filtered[idx], color="tab:red", linewidth=1.2, label="filtered")

        ax.set_title(f"{row['reward_function']}  (threshold_factor={tf:.3f}, "
                     f"snr_improvement={row['snr_improvement']:.2f}, "
                     f"signal_loss={row['signal_loss']:.3f})")
        ax.set_ylabel("amplitude")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("sample index")
    plt.tight_layout()
    plt.savefig("reward_sweep_filtered_signals.png", dpi=150)
    print("[INFO] Saved plot: reward_sweep_filtered_signals.png")
    plt.show()


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    print("Sweeping threshold_factor and computing metrics...")
    sweep_df = sweep_threshold_factor(clean_signal, noisy_signal)

    print("\nEvaluating reward functions against the Pareto frontier...\n")
    results_df, frontier_df = evaluate_reward_configs(sweep_df, REWARD_CONFIGS)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(results_df.to_string(index=False))

    plot_reward_landscapes(sweep_df, REWARD_CONFIGS)
    plot_frontier(frontier_df, results_df)
    plot_filtered_signals(clean_signal, noisy_signal, results_df)

    print("\nSweeping linear reward weight (snr vs signal_loss) to trace reachable frontier...")
    weight_sweep_df = sweep_reward_weights(sweep_df)
    weight_sweep_df.to_csv("weight_sweep_results.csv", index=False)
    print("Saved weight sweep table to weight_sweep_results.csv")

    results_df.to_csv("reward_sweep_results.csv", index=False)
    print("Saved full results table to reward_sweep_results.csv")
    print("(Plots were shown interactively via plt.show(), not saved to disk.)")