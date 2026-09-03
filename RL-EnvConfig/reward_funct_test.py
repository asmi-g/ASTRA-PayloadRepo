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
# SIGNAL_SOURCE (env var REWARD_SWEEP_SOURCE):
#   "stationary"    -> simulated_signal_match_hz.csv: the stationary white +
#                      blue + burst model (matches signal_simulation.py and
#                      custom_env_022025._generate_signals). This is the
#                      pre-flight tuning signal; section 11 on it is the
#                      baseline. Reward-function tuning is done purely
#                      against our own simulated model, not the flight
#                      signal, since this sweep represents a decision made
#                      before the flight signal was ever obtained.
#   "nonstationary" -> generate_nonstationary_cosmic_signal(): the SAME
#                      cosmic-noise model, but noise power AND burst rate
#                      drift along the signal so the optimal threshold_factor
#                      genuinely moves over time. signal_simulation.py has
#                      no non-stationarity of its own, so this is a
#                      purpose-built variant for testing whether the
#                      jump-penalty / prev-action analysis behaves
#                      differently under drift.
#
# The real flight _clean_noisy.csv files are NOT an option here: their RX
# column is ~97% exact zeros (8-bit ADC quantization, tone buried ~34 dB
# under noise), which floors the wavelet sigma estimate in every window and
# makes every threshold_factor collapse to the identity filter -- a
# degeneracy, not a noise-structure result.

def load_simulated_signal():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(script_dir, "../Data/simulated_signal_match_hz.csv"))
    df = pd.read_csv(csv_path)
    clean = df["Clean Signal"].to_numpy(dtype=np.float64)
    noisy = df["Noisy Signal"].to_numpy(dtype=np.float64)
    return clean, noisy


def generate_nonstationary_cosmic_signal(
        n_samples=300_000,
        f_sampling=1e6,
        f_signal=100_000,
        base_noise_power=0.1,
        drift_low=0.5,
        drift_high=3.0,
        n_segments=6,
        burst_prob_base=0.01,
        burst_prob_storm=0.05,
        n_storms=4,
        storm_len=15_000,
        burst_amplitude=3.0,
        burst_duration=10,
        seed=0):
    """Non-stationary version of the GCR / cosmic-interference model in
    signal_simulation.py and custom_env_022025._generate_signals() (white +
    sqrt(|f|)-shaped 'pink' + random bursts). Those are fully stationary --
    fixed NOISE_POWER, fixed burst rate. Here two things vary along the
    signal:

      * NOISE POWER: a slow smooth envelope (low-pass random walk) times the
        white+pink std, scaled to span [drift_low, drift_high] x base --
        models GCR flux drifting with altitude / geomagnetic cutoff over a
        flight -- plus n_segments abrupt multiplicative level shifts that
        model the real flight file's stitched-capture boundaries (per-segment
        oscillator/gain state made noise std jump ~2x across flight_signal_1).
      * BURST RATE: n_storms windows of elevated burst probability
        (burst_prob_storm vs burst_prob_base) -- models solar-particle /
        air-shower activity clustering in time rather than being uniform.

    Clean tone, the sqrt(|f|) 'pink' filter, burst amplitude/duration and the
    sqrt(base_noise_power) scaling all match the stationary model exactly, so
    a stationary-vs-this comparison isolates the effect of non-stationarity.
    Returns (clean, noisy), both float64, comparable-scale (no ADC step)."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_samples)
    t = x / f_sampling
    clean = np.sin(2 * np.pi * f_signal * t)

    # slow smooth envelope: low-pass a random walk, map to [drift_low, drift_high]
    walk = np.cumsum(rng.normal(0, 1, n_samples))
    k = max(1, n_samples // 20)
    walk = np.convolve(walk, np.ones(k) / k, mode="same")
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-12)
    envelope = drift_low + (drift_high - drift_low) * walk

    # abrupt "stitched segment" level shifts
    seg_bounds = np.linspace(0, n_samples, n_segments + 1).astype(int)
    for s in range(n_segments):
        envelope[seg_bounds[s]:seg_bounds[s + 1]] *= rng.uniform(0.7, 1.6)

    # white + blue ("pink") noise, then modulated by the envelope
    white = rng.normal(0, np.sqrt(base_noise_power), n_samples)
    pink_filter = np.sqrt(np.abs(np.fft.fftfreq(n_samples)))
    spectrum = np.fft.fft(white / np.sqrt(base_noise_power)) * pink_filter
    pink = np.real(np.fft.ifft(spectrum)) * np.sqrt(base_noise_power)
    noisy = clean + (white + pink) * envelope

    # time-varying burst rate: baseline everywhere, elevated inside storms
    burst_prob = np.full(n_samples, burst_prob_base)
    for _ in range(n_storms):
        start = int(rng.integers(0, max(1, n_samples - storm_len)))
        burst_prob[start:start + storm_len] = burst_prob_storm
    draws = rng.random(n_samples)
    for i in range(n_samples):  # matches _add_bursts: per-sample check, bursts may stack
        if draws[i] < burst_prob[i]:
            end = min(i + burst_duration, n_samples)
            noisy[i:end] += rng.uniform(-burst_amplitude, burst_amplitude)

    return clean.astype(np.float64), noisy.astype(np.float64)


SIGNAL_SOURCE = os.environ.get("REWARD_SWEEP_SOURCE", "stationary")
if SIGNAL_SOURCE == "nonstationary":
    clean_signal, noisy_signal = generate_nonstationary_cosmic_signal()
else:
    clean_signal, noisy_signal = load_simulated_signal()
print(f"[INFO] SIGNAL_SOURCE = {SIGNAL_SOURCE} | {len(clean_signal)} samples")


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


# ------------------------------------------------------------
# Weights for the action-smoothness (jump-penalty) analysis in section 11.
# These FIX the snr / signal_loss / correlation tradeoff before the jump
# penalty is layered on top. Set them from the frontier + weight-sweep
# results (sections 6-8): pick the point on the Pareto frontier you
# actually want -- since the goal is "improve SNR without giving up signal
# fidelity", bias toward a lower-signal_loss frontier point rather than the
# raw snr_improvement max. Defaults mirror threeterm_final.
# ------------------------------------------------------------
W_SNR = 1.0
W_LOSS = 1.25
W_CORR = 0.25

# Action range the live RL env exposes (custom_env_022025.py). The jump
# penalty is measured in ACTION units (a in [-1, 1]), not threshold_factor
# units, so it matches the term you would actually add to the RL reward.
ACTION_LOW_TF, ACTION_HIGH_TF = 0.05, 2.5

# Windows for the trajectory analysis OVERLAP at the real inference stride
# (inference.py), not the non-overlapping windows the fast sweep uses. At
# stride 100 / window 1000 adjacent windows already share 90% of their
# samples, which is itself a strong smoother; independent windows would
# overstate the jitter and make the penalty look more useful than it is.
TRAJECTORY_STRIDE = 100
TRAJECTORY_WINDOW_CAP = 2500    # analyse at most this many consecutive windows
TRAJECTORY_N_THRESHOLDS = 90


# ============================================================
# 5. SWEEP LOGIC
# ============================================================

def sweep_threshold_factor(clean, noisy, threshold_range=None, n_points=200, window_size=1000):
    # threshold_factor default range matches the RL agent's actual achievable
    # range: NoiseReductionEnv maps action in [-1,1] to threshold_factor in
    # [0.05, 2.5] (custom_env_022025.py StatelessDenoisingEnv.step -- the live
    # env was widened from [0.5, 2.5] to [0.05, 2.5]; this file previously
    # still said 2.0). Sweeping outside that range would surface "optimal"
    # points the agent can never actually select.
    if threshold_range is None:
        threshold_range = np.linspace(0.05, 2.5, n_points)

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
# 11. ACTION-SMOOTHNESS / JUMP-PENALTY ANALYSIS (no RL training)
# ============================================================
# The sweep above collapses each threshold_factor to ONE number per metric
# (mean over windows), then picks a single global argmax threshold. That
# cannot evaluate a "penalize large action jumps" term: with one global
# threshold there is no action *trajectory*, so every delta-a is zero and
# the penalty is identically zero for every candidate.
#
# To screen the jump penalty WITHOUT training an RL model, build the full
# per-window reward curve r_w(tf), then simulate the policy an RL agent
# would converge to. Because consecutive windows are (near-)independent of
# the current action -- white/blue noise + 10-sample bursts, all
# correlation lengths << the 1000-sample window -- the task is essentially
# a contextual bandit, and a memoryless-plus-prev_action SAC policy at
# convergence approximates the GREEDY CAUSAL one-step smoother:
#     a_w = argmax_a [ r_w(a) - lam * (a - a_{w-1})^2 ]
# applied window by window. dp_global_smoother() additionally solves the
# globally optimal smoothed trajectory by DP (Viterbi over the threshold
# grid) as an upper bound on what ANY prev_action-aware policy (including a
# lookahead / recurrent one) could achieve.
#
# Sweeping lam traces the COST OF SMOOTHNESS: how much mean snr_improvement
# / signal_loss is given up to reduce mean |delta action|. If SNR/loss stay
# flat until lam is large, the penalty is ~free and worth adding; if any
# lam > 0 immediately moves you off the lam=0 Pareto point, it is not free
# and the independent-per-window policy is better.

def sweep_threshold_factor_per_window(clean, noisy, thresholds,
                                      window_size=1000, stride=TRAJECTORY_STRIDE,
                                      window_cap=TRAJECTORY_WINDOW_CAP):
    """For every (window, threshold_factor) pair, the snr_improvement /
    signal_loss / correlation of that window filtered at that threshold.
    Returns (snr_mat, loss_mat, corr_mat), each shape (n_windows, len(thresholds))."""
    n_windows = 1 + (len(clean) - window_size) // stride
    n_windows = min(n_windows, window_cap)
    T = len(thresholds)
    print(f"  per-window sweep: {n_windows} windows x {T} thresholds "
          f"(stride={stride}, window={window_size})")

    snr_mat = np.empty((n_windows, T))
    loss_mat = np.empty((n_windows, T))
    corr_mat = np.empty((n_windows, T))

    for ti, tf in enumerate(thresholds):
        if ti % 10 == 0:
            print(f"    threshold {ti + 1}/{T}")
        for w in range(n_windows):
            s = w * stride
            c = clean[s:s + window_size]
            nz = noisy[s:s + window_size]
            filt = apply_filter(nz, threshold_factor=tf)
            snr_mat[w, ti] = calculate_snr(c, filt) - calculate_snr(c, nz)
            loss_mat[w, ti] = calculate_signal_loss(c, filt)
            corr_mat[w, ti] = calculate_correlation(c, filt)
    return snr_mat, loss_mat, corr_mat


def greedy_causal_smoother(reward_mat, action_grid, lam):
    """The policy a converged memoryless+prev_action SAC agent approximates
    on a near-bandit task: walk windows in order, at each pick the action
    maximising immediate reward minus lam*(a - a_prev)^2."""
    W, T = reward_mat.shape
    choice = np.empty(W, dtype=int)
    choice[0] = int(np.argmax(reward_mat[0]))
    for w in range(1, W):
        penalty = lam * (action_grid - action_grid[choice[w - 1]]) ** 2
        choice[w] = int(np.argmax(reward_mat[w] - penalty))
    return choice


def dp_global_smoother(reward_mat, action_grid, lam):
    """Globally optimal smoothed trajectory (Viterbi over the threshold
    grid): an upper bound on what ANY prev_action-aware policy could get,
    since it is allowed to look ahead. The greedy smoother is the realistic
    analog; the gap between them bounds how much a recurrent policy adds."""
    W, T = reward_mat.shape
    diff2 = (action_grid[:, None] - action_grid[None, :]) ** 2  # (prev, cur)
    value = reward_mat[0].copy()
    back = np.empty((W, T), dtype=int)
    for w in range(1, W):
        cand = value[:, None] - lam * diff2 + reward_mat[w][None, :]
        back[w] = np.argmax(cand, axis=0)
        value = np.max(cand, axis=0)
    choice = np.empty(W, dtype=int)
    choice[-1] = int(np.argmax(value))
    for w in range(W - 1, 0, -1):
        choice[w - 1] = back[w, choice[w]]
    return choice


def evaluate_trajectory(choice, snr_mat, loss_mat, corr_mat, action_grid):
    w_idx = np.arange(len(choice))
    actions = action_grid[choice]
    dz = np.diff(actions)
    return {
        "snr_improvement": float(np.mean(snr_mat[w_idx, choice])),
        "signal_loss": float(np.mean(loss_mat[w_idx, choice])),
        "correlation": float(np.mean(corr_mat[w_idx, choice])),
        "mean_abs_delta_action": float(np.mean(np.abs(dz))) if len(dz) else 0.0,
        "max_abs_delta_action": float(np.max(np.abs(dz))) if len(dz) else 0.0,
        "std_action": float(np.std(actions)),
    }


def sweep_jump_penalty(snr_mat, loss_mat, corr_mat, action_grid, thresholds,
                       w_snr=W_SNR, w_loss=W_LOSS, w_corr=W_CORR, lambdas=None):
    """For fixed (w_snr, w_loss, w_corr), sweep the jump-penalty weight lam
    and record what the resulting action trajectory costs on each axis.
    The lam=0 row is the independent-per-window baseline."""
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.logspace(-3, 1, 21)])

    reward_mat = w_snr * snr_mat - w_loss * loss_mat + w_corr * corr_mat
    base_choice = greedy_causal_smoother(reward_mat, action_grid, 0.0)
    base = evaluate_trajectory(base_choice, snr_mat, loss_mat, corr_mat, action_grid)

    rows = []
    traj = {}
    for lam in lambdas:
        for method, solver in (("greedy", greedy_causal_smoother),
                               ("dp", dp_global_smoother)):
            choice = solver(reward_mat, action_grid, lam)
            ev = evaluate_trajectory(choice, snr_mat, loss_mat, corr_mat, action_grid)
            rows.append({
                "lambda": round(float(lam), 6),
                "method": method,
                **{k: round(v, 5) for k, v in ev.items()},
                "d_snr_vs_base": round(ev["snr_improvement"] - base["snr_improvement"], 5),
                "d_loss_vs_base": round(ev["signal_loss"] - base["signal_loss"], 5),
            })
            if method == "greedy":
                traj[float(lam)] = thresholds[choice]
    return pd.DataFrame(rows), base, traj


def plot_jump_penalty_sweep(sweep_rows):
    g = sweep_rows[sweep_rows["method"] == "greedy"].sort_values("lambda")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.plot(g["lambda"], g["snr_improvement"], "o-", color="tab:green", label="snr_improvement")
    ax1.plot(g["lambda"], g["signal_loss"], "s-", color="tab:red", label="signal_loss")
    ax1.set_xlabel("jump-penalty weight lambda  (0 = independent per window)")
    ax1.set_ylabel("snr_improvement / signal_loss")
    ax1.legend(loc="center left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(g["lambda"], g["mean_abs_delta_action"], "^--", color="tab:blue",
             label="mean |delta action|")
    ax2.set_ylabel("mean |delta action|  (trajectory smoothness)")
    ax2.legend(loc="center right", fontsize=8)

    ax1.set_title("Cost of the action-jump penalty\n"
                  "(flat green/red until large lambda => smoothing is ~free)")
    plt.tight_layout()
    plt.savefig("jump_penalty_sweep.png", dpi=150)
    print("[INFO] Saved plot: jump_penalty_sweep.png")
    plt.show()


def plot_jump_penalty_trajectories(traj, lambdas_to_show=None):
    keys = sorted(traj.keys())
    if lambdas_to_show is None:
        lambdas_to_show = [keys[0], keys[len(keys) // 3],
                           keys[2 * len(keys) // 3], keys[-1]]
    fig, ax = plt.subplots(figsize=(9, 4))
    for lam in lambdas_to_show:
        ax.plot(traj[lam], linewidth=1.0, label=f"lambda={lam:.3g}")
    ax.set_xlabel(f"window index (stride = {TRAJECTORY_STRIDE} samples)")
    ax.set_ylabel("chosen threshold_factor")
    ax.set_title("Chosen threshold_factor trajectory vs jump-penalty weight")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("jump_penalty_trajectories.png", dpi=150)
    print("[INFO] Saved plot: jump_penalty_trajectories.png")
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

    # --- section 11: action-smoothness / jump-penalty screening ---
    RUN_JUMP_PENALTY_ANALYSIS = True
    if RUN_JUMP_PENALTY_ANALYSIS:
        print("\n" + "=" * 60)
        print("ACTION-SMOOTHNESS / JUMP-PENALTY ANALYSIS")
        print(f"fixed weights: snr={W_SNR}  signal_loss={W_LOSS}  correlation={W_CORR}")
        print("=" * 60)
        traj_thresholds = np.linspace(ACTION_LOW_TF, ACTION_HIGH_TF, TRAJECTORY_N_THRESHOLDS)
        action_grid = np.interp(traj_thresholds,
                                [ACTION_LOW_TF, ACTION_HIGH_TF], [-1.0, 1.0])

        snr_mat, loss_mat, corr_mat = sweep_threshold_factor_per_window(
            clean_signal, noisy_signal, traj_thresholds)

        jp_df, jp_base, jp_traj = sweep_jump_penalty(
            snr_mat, loss_mat, corr_mat, action_grid, traj_thresholds)

        print("\nBaseline (lambda = 0, independent per window):")
        for k, v in jp_base.items():
            print(f"  {k:24s} {v:.5f}")
        print("\nJump-penalty sweep (greedy causal smoother -- realistic analog):")
        print(jp_df[jp_df["method"] == "greedy"].to_string(index=False))
        print("\nJump-penalty sweep (DP global optimum -- lookahead upper bound):")
        print(jp_df[jp_df["method"] == "dp"].to_string(index=False))

        jp_df.to_csv("jump_penalty_sweep_results.csv", index=False)
        print("\nSaved: jump_penalty_sweep_results.csv")

        plot_jump_penalty_sweep(jp_df)
        plot_jump_penalty_trajectories(jp_traj)