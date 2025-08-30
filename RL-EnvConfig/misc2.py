import numpy as np
import matplotlib.pyplot as plt
from astra_rev1.envs import NoiseReductionEnv

# Initialize environment
env = NoiseReductionEnv(window_size=10)
env.reset()

# Number of windows to average over
num_windows = 500  # <-- SET THIS HERE

threshold_factors = np.linspace(0.5, 2.5, 50)
all_rewards = np.zeros_like(threshold_factors)
all_snr_improvements = np.zeros_like(threshold_factors)
all_mses = np.zeros_like(threshold_factors)

# Loop over multiple windows
for _ in range(num_windows):
    # Choose a random valid window
    while len(env.clean) < env.window_size:
        print(len(env.clean))
        env.reset()
    max_start = len(env.clean) - env.window_size
    start = 0

    clean_window = env.clean[start:start + env.window_size]
    noisy_window = env.noisy[start:start + env.window_size]

    rewards, snr_improvements, mses = [], [], []

    # Evaluate across thresholds
    for threshold in threshold_factors:
        action = np.interp(threshold, [0.5, 2.5], [-1.0, 1.0])
        env.set_signal_window(clean_window, noisy_window)
        _, reward, _, info = env.step(np.array([action], dtype=np.float32))

        rewards.append(info["reward"])
        snr_improvements.append(info["SNR_filtered"] - info["SNR_raw"])
        mses.append(np.mean((info["filtered_signal"] - clean_window) ** 2))

    all_rewards += np.array(rewards)
    all_snr_improvements += np.array(snr_improvements)
    all_mses += np.array(mses)

# Compute averages
avg_rewards = all_rewards / num_windows
avg_snr_improvements = all_snr_improvements / num_windows
avg_mses = all_mses / num_windows

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(threshold_factors, avg_rewards, color='red')
axs[0].set_title("Avg Reward vs Threshold")
axs[0].set_xlabel("Threshold Factor")
axs[0].set_ylabel("Average Reward")

axs[1].plot(threshold_factors, avg_snr_improvements, color='blue')
axs[1].set_title("Avg SNR Improvement vs Threshold")
axs[1].set_xlabel("Threshold Factor")
axs[1].set_ylabel("Average SNR Improvement (dB)")

axs[2].plot(threshold_factors, avg_mses, color='green')
axs[2].set_title("Avg MSE vs Threshold")
axs[2].set_xlabel("Threshold Factor")
axs[2].set_ylabel("Average MSE")

plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from astra_rev1.envs import NoiseReductionEnv

# # Initialize environment
# env = NoiseReductionEnv(window_size=10)
# env.reset()

# # Parameters
# threshold_factors = np.linspace(0.5, 5, 50)
# window_stride = 10  # stride between windows
# window_indices = range(0, len(env.clean) - env.window_size, window_stride)

# # Initialize accumulators
# all_rewards = np.zeros_like(threshold_factors)
# all_snr_improvements = np.zeros_like(threshold_factors)
# all_mses = np.zeros_like(threshold_factors)
# valid_window_count = 0

# # Sweep across signal windows
# for idx in window_indices:
#     clean_window = env.clean[idx:idx + env.window_size]
#     noisy_window = env.noisy[idx:idx + env.window_size]

#     if clean_window.shape[0] != env.window_size or noisy_window.shape[0] != env.window_size:
#         continue

#     rewards, snr_improvements, mses = [], [], []

#     for threshold in threshold_factors:
#         action = np.interp(threshold, [0.5, 2.5], [-1.0, 1.0])
#         env.set_signal_window(clean_window, noisy_window)
#         _, reward, _, info = env.step(np.array([action], dtype=np.float32))

#         rewards.append(info["reward"])
#         snr_improvements.append(info["SNR_filtered"] - info["SNR_raw"])
#         mses.append(np.mean((info["filtered_signal"] - clean_window) ** 2))

#     if len(rewards) == len(threshold_factors):
#         all_rewards += np.array(rewards)
#         all_snr_improvements += np.array(snr_improvements)
#         all_mses += np.array(mses)
#         valid_window_count += 1

# # Normalize
# if valid_window_count == 0:
#     raise ValueError("No valid windows found for evaluation.")

# all_rewards /= valid_window_count
# all_snr_improvements /= valid_window_count
# all_mses /= valid_window_count

# # Plot
# fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# axs[0].plot(threshold_factors, all_rewards, color='red')
# axs[0].set_title("Avg Reward vs Threshold")
# axs[0].set_xlabel("Threshold Factor")
# axs[0].set_ylabel("Reward")

# axs[1].plot(threshold_factors, all_snr_improvements, color='blue')
# axs[1].set_title("Avg SNR Improvement vs Threshold")
# axs[1].set_xlabel("Threshold Factor")
# axs[1].set_ylabel("SNR Improvement (dB)")

# axs[2].plot(threshold_factors, all_mses, color='green')
# axs[2].set_title("Avg MSE vs Threshold")
# axs[2].set_xlabel("Threshold Factor")
# axs[2].set_ylabel("Mean Squared Error")

# plt.tight_layout()
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from astra_rev1.envs import NoiseReductionEnv

# # Initialize environment
# env = NoiseReductionEnv(window_size=10)
# env.reset()

# # Select a stable window for testing
# start = 100
# clean_window = env.clean[start:start + env.window_size]
# noisy_window = env.noisy[start:start + env.window_size]

# threshold_factors = np.linspace(0.5, 2.5, 50)
# rewards, snr_improvements, mses = [], [], []

# # Evaluate over threshold range
# for threshold in threshold_factors:
#     action = np.interp(threshold, [0.5, 2.5], [-1.0, 1.0])
#     env.set_signal_window(clean_window, noisy_window)
#     _, reward, _, info = env.step(np.array([action], dtype=np.float32))

#     rewards.append(info["reward"])
#     snr_improvements.append(info["SNR_filtered"] - info["SNR_raw"])
#     mses.append(np.mean((info["filtered_signal"] - clean_window) ** 2))

# # Plot results
# fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# axs[0].plot(threshold_factors, rewards, color='red')
# axs[0].set_title("Reward vs Threshold")
# axs[0].set_xlabel("Threshold Factor")
# axs[0].set_ylabel("Reward")

# axs[1].plot(threshold_factors, snr_improvements, color='blue')
# axs[1].set_title("SNR Improvement vs Threshold")
# axs[1].set_xlabel("Threshold Factor")
# axs[1].set_ylabel("SNR Improvement (dB)")

# axs[2].plot(threshold_factors, mses, color='green')
# axs[2].set_title("MSE vs Threshold")
# axs[2].set_xlabel("Threshold Factor")
# axs[2].set_ylabel("Mean Squared Error")

# plt.tight_layout()
# plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt

# filename = 'C:/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/rx_tx_data.csv'
# df= pd.read_csv(filename)

# print(df.head())

# plt.figure(figsize=(12, 10))

# plt.subplot(3, 1, 1)
# plt.plot(df['RX Magnitude'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Magnitude'], label='Clean Signal', color='orange')
# plt.title('Noisy and Clean Signals')
# plt.legend()


# plt.subplot(3, 1, 2)
# plt.plot(df['RX Real'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Real'], label='Clean Signal', color='orange')
# plt.title('Real Component of Noisy and Clean Signals')
# plt.legend()


# plt.subplot(3, 1, 3)
# plt.plot(df['RX Imag'], label='Noisy Signal', color='blue')
# plt.plot(df['TX Imag'], label='Clean Signal', color='orange')
# plt.title('Imag Component of Noisy and Clean Signals')
# plt.legend()

# plt.tight_layout()
# plt.show()