import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv
import os
import time

# Load SAC model
custom_objects = {
    "lr_schedule": lambda x: 0.003,
    "clip_range": lambda x: 0.02
}

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(script_dir, "../models/sac_noise_reduction_080725_6am.zip"))
model = SAC.load(model_path, custom_objects=custom_objects)

# Initialize environment
env = NoiseReductionEnv()

# Parameters
window_size = 10

BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Scripts/SDR/Data/")
csv_path = os.path.join(DATA_DIR, "signal.csv")
#csv_path = os.path.normpath(os.path.join(script_dir, "../Data/simulated_signal_data.csv")) # for testing

poll_interval = 2      # seconds between polls
timeout_seconds = 10   # time to wait for new data before exiting

# Tracking
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []
snr_improvement = []
clean_signal_data = []
noisy_signal_data = []
filtered_signal_data = []
thresholds = []
mse = []
results_rows = []

last_processed_index = window_size - 1
last_update_time = time.time()
done = False

# --- before the loop ---
clean_signal_data    = []
noisy_signal_data    = []
filtered_signal_data = []

print("Waiting for data to appear...")

while (1):
    # Load the latest CSV
    try:
        df = pd.read_csv(csv_path).tail(5000).rename(columns={
            'TX Magnitude': 'Noisy Signal',
            'RX Magnitude': 'Clean Signal'
        })
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("signal.csv not ready. Retrying...")
        print(csv_path)
        time.sleep(poll_interval)
        continue

    # Check if there is enough data
    if len(df) < window_size:
        print("Not enough data yet...")
        time.sleep(poll_interval)
        continue

    # Check for new data
    if last_processed_index >= len(df):
        if time.time() - last_update_time > timeout_seconds:
            print("No new data detected for timeout period. Exiting.")
            break
        else:
            time.sleep(poll_interval)
            continue

    # New data is available
    last_update_time = time.time()

    while last_processed_index <= len(df) - 1:
        i = last_processed_index

        # build right-aligned windows that end at index i
        win_clean = df.iloc[i - window_size + 1 : i + 1]["Clean Signal"].to_numpy()
        win_noisy = df.iloc[i - window_size + 1 : i + 1]["Noisy Signal"].to_numpy()

        if i == window_size - 1:
            state = env.reset(clean_signal=win_clean, noisy_signal=win_noisy)

        # RL step
        state_in = np.expand_dims(state, 0)
        action, _ = model.predict(state_in, deterministic=True)
        next_state, reward, done, info = env.step(action)

        # logging
        snr_raw = info["SNR_raw"]
        snr_filtered = info["SNR_filtered"]
        filt = np.asarray(info["filtered_signal"])  # length == window_size
        t_factor = info["threshold_factor"]

        rewards.append(reward)
        thresholds.append(t_factor)
        snr_raw_list.append(snr_raw)
        snr_filtered_list.append(snr_filtered)
        snr_improvement.append(snr_filtered - snr_raw)

        # --- synced saving ---
        if i == window_size - 1:
            # first step: dump the whole window
            clean_signal_data.extend(win_clean.tolist())
            noisy_signal_data.extend(win_noisy.tolist())
            filtered_signal_data.extend(filt.tolist())
        else:
            # subsequent: append one sample (right anchor at i)
            clean_signal_data.append(float(df.iloc[i]["Clean Signal"]))
            noisy_signal_data.append(float(df.iloc[i]["Noisy Signal"]))
            filtered_signal_data.append(float(filt[-1]))

        print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Improvement: {snr_improvement[-1]:.2f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {np.mean(filtered_signal_data):.4f} | clean signal: {np.mean(win_clean):.4f} | threshold factor: {t_factor:.4f}")

        results_rows.append({
            "window": f"({i - window_size + 1}, {i})",
            "action": action,
            "reward": reward,
            "snr_improvement": snr_improvement[-1],
            "threshold_factor": t_factor
        })

        # prepare next window for the env (only if there *is* a next sample)
        if i + 1 < len(df):
            next_win_clean = np.r_[win_clean[1:], df.iloc[i + 1]["Clean Signal"]]
            next_win_noisy = np.r_[win_noisy[1:], df.iloc[i + 1]["Noisy Signal"]]
            env.set_signal_window(next_win_clean, next_win_noisy)

        state = next_state
        last_processed_index += 1

        if done:
            print(f"Early termination signaled at index {i}.")
            results_rows.append({"window": "(DONE)", "action": np.nan,
                                "reward": np.nan, "snr_improvement": np.nan,
                                "threshold_factor": np.nan})
            break  # or continue, depending on your env semantics
    
    time.sleep(poll_interval)

env.close()

# Save results
os.makedirs("Data", exist_ok=True)
pd.DataFrame(results_rows).to_csv("Data/results.csv", index=False)

print("Inference complete. Results saved.")

# snr_improvement = np.array(snr_improvement)
# x = np.arange(len(snr_improvement))
# coeffs = np.polyfit(x, snr_improvement, deg=1)
# trendline = np.polyval(coeffs, x)

# # --- Visualization ---
# plt.figure(figsize=(12, 6))

# # Signal comparison
# plt.subplot(3, 1, 1)
# plt.plot(clean_signal_data, label="Clean Signal", color="blue", alpha=0.8)
# plt.plot(noisy_signal_data, label="Noisy Signal", color="orange", alpha=0.5)
# plt.plot(filtered_signal_data, label="Filtered Signal", color="green", alpha=0.8)
# plt.title("Clean vs. Noisy vs. Filtered Signal with SAC window size 10")
# plt.ylabel("Signal Amplitude")
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(snr_improvement, label="SNR Improvement", color="red")
# plt.plot(x, trendline, label="Trendline", color="blue", linestyle='--')
# plt.axhline(y=0, color='black', linestyle='--')
# plt.ylabel("SNR Improvement")
# plt.title("SNR Improvement Over Time")
# plt.legend()

# # # MSE evolution
# # plt.subplot(3, 1, 3)
# # plt.plot(mse, label="MSE (SNR Filtered vs. Raw)", color="blue")
# # plt.xlabel("Time Step")
# # plt.ylabel("MSE")
# # plt.title("MSE Over Time")
# # plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(thresholds, label="Threshold over time", color="blue")
# plt.xlabel("Time Step")
# plt.ylabel("Threshold")
# plt.title("Threshold Over Time")
# plt.legend()

# plt.tight_layout()
# plt.show()
