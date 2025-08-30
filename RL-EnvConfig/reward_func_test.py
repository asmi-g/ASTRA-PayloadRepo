# sweep_threshold_reward.py
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ---------- your env semantics ----------
def snr_db(clean, test):
    noise = test - clean
    sp = np.mean(clean**2)
    npow = np.mean(noise**2)
    return 10 * np.log10((sp + 1e-10) / (npow + 1e-10))

def apply_filter_like_env(window, wavelet='db4', level=1, threshold_factor=1.5):
    coeffs = pywt.wavedec(window, wavelet, level=level, mode='periodization')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    lam = threshold_factor * sigma * np.sqrt(2.0 * np.log(max(len(window), 2)))
    coeffs_thr = [coeffs[0]] + [pywt.threshold(c, lam, mode='soft') for c in coeffs[1:]]
    y = pywt.waverec(coeffs_thr, wavelet, mode='periodization')[:len(window)]
    y += (np.mean(window) - np.mean(y))  # keep DC
    return y

def reward_like_env(clean, noisy, filtered, threshold):
    snr_raw = snr_db(clean, noisy)
    snr_flt = snr_db(clean, filtered)
    snr_improvement = snr_flt - snr_raw
    signal_loss = np.log1p(np.mean((filtered - clean)**2))
    # robust correlation (guard zero variance)
    if np.std(clean) < 1e-12 or np.std(filtered) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(filtered, clean)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    # your weights
    reward = (0.8 * snr_improvement) - (1 * signal_loss) + (0.25 * corr)
    reward -= 0.1 * abs(threshold - 1.5)  # center bias you added
    return reward, snr_improvement, signal_loss, corr

# ---------- simple synthetic generator (matches your style) ----------
def generate_pair(n=1000, fs=1e4, f0=100, noise_power=0.1,
                  burst_p=0.01, burst_amp=3.0, burst_len=10, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    clean = np.sin(2*np.pi*f0*t)

    white = rng.normal(0, np.sqrt(noise_power), size=n)
    # pink-ish by 1/sqrt(f) shaping
    freqs = np.fft.rfftfreq(n, d=1/fs)
    H = np.sqrt(np.maximum(freqs, 1.0/(n/fs)))  # avoid 0
    spec = np.fft.rfft(white) * H
    pink = np.fft.irfft(spec, n=n)

    noisy = clean + white + pink
    # random bursts
    for i in range(n - burst_len):
        if rng.rand() < burst_p:
            noisy[i:i+burst_len] += rng.uniform(-burst_amp, burst_amp)
    return clean.astype(np.float64), noisy.astype(np.float64)

# ---------- run a sweep on a random window ----------
def run_sweep(window_size=10, level=1, wavelet='db4',
              thresholds=np.linspace(0.5, 2.5, 51), seed=0):
    clean, noisy = generate_pair(n=4000, seed=seed)
    # pick a window with enough variance
    rng = np.random.RandomState(seed+1)
    i = rng.randint(0, len(clean) - window_size)
    cwin = clean[i:i+window_size]
    nwin = noisy[i:i+window_size]

    rewards, gains, mses, corrs = [], [], [], []
    for th in thresholds:
        flt = apply_filter_like_env(nwin, wavelet=wavelet, level=level, threshold_factor=th)
        r, g, m, c = reward_like_env(cwin, nwin, flt, th)
        rewards.append(r); gains.append(g); mses.append(m); corrs.append(c)

    rewards = np.array(rewards); gains = np.array(gains)
    mses = np.array(mses); corrs = np.array(corrs)
    best_idx = int(np.nanargmax(rewards))
    best_th = float(thresholds[best_idx])

    # ---- plots ----
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0,0].plot(thresholds, rewards, label='Reward', lw=2)
    axs[0,0].axvline(best_th, ls='--', c='k'); axs[0,0].legend(); axs[0,0].set_title('Reward vs Threshold')
    axs[0,1].plot(thresholds, gains, label='SNR gain (dB)')
    axs[0,1].axvline(best_th, ls='--', c='k'); axs[0,1].legend(); axs[0,1].set_title('SNR Gain vs Threshold')
    axs[1,0].plot(thresholds, mses, label='log1p(MSE)')
    axs[1,0].axvline(best_th, ls='--', c='k'); axs[1,0].legend(); axs[1,0].set_title('Loss term vs Threshold')
    axs[1,1].plot(thresholds, corrs, label='corr(filtered, clean)')
    axs[1,1].axvline(best_th, ls='--', c='k'); axs[1,1].legend(); axs[1,1].set_title('Correlation vs Threshold')
    for ax in axs.ravel(): ax.set_xlabel('Threshold factor')
    plt.tight_layout(); plt.show()

    print(f"Best threshold on this window: {best_th:.3f}")
    print(f"Reward at best: {rewards[best_idx]:.3f}, SNR gain: {gains[best_idx]:.3f} dB, corr: {corrs[best_idx]:.3f}")
    return thresholds, rewards, gains, mses, corrs, best_th

if __name__ == "__main__":
    run_sweep()



# import numpy as np
# import matplotlib.pyplot as plt

# # Define a sample clean and noisy window
# np.random.seed(0)
# window_size = 10
# clean = np.sin(np.linspace(0, np.pi, window_size))
# noise = np.random.normal(0, 0.3, window_size)
# noisy = clean + noise

# # Define wavelet denoising function
# import pywt
# def wavelet_denoise(window, threshold_factor, wavelet='db4', level=1):
#     coeffs = pywt.wavedec(window, wavelet, level=level)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     threshold = threshold_factor * sigma
#     coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
#     return pywt.waverec(coeffs, wavelet)[:len(window)]

# # Define reward function
# def compute_reward(clean, noisy, threshold_factor, alpha, beta, gamma):
#     filtered = wavelet_denoise(noisy, threshold_factor)
#     snr_raw = 10 * np.log10(np.mean(clean**2) / (np.mean((noisy - clean)**2) + 1e-10))
#     snr_filtered = 10 * np.log10(np.mean(clean**2) / (np.mean((filtered - clean)**2) + 1e-10))
#     snr_gain = snr_filtered - snr_raw
#     signal_loss = np.mean((filtered - clean)**2)
#     correlation = np.corrcoef(filtered, clean)[0, 1]
#     reward = alpha * snr_gain - beta * signal_loss + gamma * correlation
#     return reward

# # Threshold factors to test
# threshold_factors = np.linspace(0.1, 5.0, 100)

# # Sweep through different (alpha, beta, gamma) values
# param_sets = [
#     (0.5, 0.5, 0.5),
#     (0.25, 0.5, 0.5),
#     (0.75, 0.25, 0.5),
#     (0.5, 0.5, 0.75),
#     (1.0, 1.0, 1.0),
# ]

# # Plot reward vs threshold for each parameter set
# plt.figure(figsize=(12, 6))
# for alpha, beta, gamma in param_sets:
#     rewards = [compute_reward(clean, noisy, tf, alpha, beta, gamma) for tf in threshold_factors]
#     plt.plot(threshold_factors, rewards, label=f"α={alpha}, β={beta}, γ={gamma}")

# plt.xlabel("Threshold Factor")
# plt.ylabel("Reward")
# plt.title("Reward vs Threshold Factor for Different α, β, γ Combinations")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from astra_rev1.envs import NoiseReductionEnv  # Update path as needed
# from stable_baselines3 import SAC
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# env = Monitor(NoiseReductionEnv())
# model = SAC("MlpPolicy", env, verbose=0)
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
# print(f"Mean Reward: {mean_reward}, Std: {std_reward}")

# # Setup
# env = NoiseReductionEnv(window_size=10)
# clean_signal, noisy_signal = env._generate_signals()

# window_starts = np.linspace(0, len(clean_signal) - env.window_size, 5, dtype=int)  # 5 sample windows
# threshold_factors = np.linspace(0.5, 2.5, 50)

# results = []

# for t in window_starts:
#     clean_win = clean_signal[t:t+env.window_size]
#     noisy_win = noisy_signal[t:t+env.window_size]
    
#     rewards = []
#     for tf in threshold_factors:
#         action = [np.interp(tf, [0.5, 2.5], [-1.0, 1.0])]  # inverse mapping
#         _, reward, _, info = env.denoiser.step(noisy_win, action, clean_win)
#         rewards.append(reward)
    
#     results.append((t, rewards))

# # --- Plot ---
# plt.figure(figsize=(10, 6))
# for t, rewards in results:
#     plt.plot(threshold_factors, rewards, label=f"Window @ {t}")
# plt.xlabel("Threshold Factor")
# plt.ylabel("Reward")
# plt.title("Reward vs Threshold Factor Across Signal Windows")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Generate a batch of test windows
# clean_signal, noisy_signal = env._generate_signals()
# window_size = 10

# thresholds = np.linspace(0.5, 2.5, 20)
# rewards_per_threshold = []

# for thresh in thresholds:
#     reward_trace = []

#     for i in range(len(clean_signal) - window_size):
#         clean_window = clean_signal[i:i+window_size]
#         noisy_window = noisy_signal[i:i+window_size]
#         filtered_window = denoise_with_threshold(noisy_window, thresh)

#         snr_raw = compute_snr(clean_window, noisy_window)
#         snr_filtered = compute_snr(clean_window, filtered_window)
#         improvement = snr_filtered - snr_raw
#         loss = np.mean((filtered_window - clean_window)**2)
#         normalized_loss = loss / (np.max(clean_window)**2 + 1e-6)
#         corr = np.corrcoef(filtered_window, clean_window)[0, 1]
#         if np.isnan(corr): corr = 0.0

#         reward = (
#             1 * improvement - 1.5 * normalized_loss + 0.75 * corr
#         )
#         reward_trace.append(reward)

#     rewards_per_threshold.append(np.mean(reward_trace))

# # Plot threshold vs average reward
# plt.plot(thresholds, rewards_per_threshold)
# plt.xlabel("Threshold")
# plt.ylabel("Avg Reward")
# plt.title("Reward Landscape vs. Threshold")
# plt.show()

