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
model_path = os.path.normpath(os.path.join(script_dir, "../models/sac_denoise_framestack_081125_7pm.zip"))
model = SAC.load(model_path, custom_objects=custom_objects)

# Initialize environment
env = NoiseReductionEnv()

# Parameters
window_size = 52 #window size has to be 52; training changed

BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Scripts/SDR/Data/")
csv_path = os.path.join(DATA_DIR, "signal.csv")
# csv_path = os.path.normpath(os.path.join(script_dir, "../Data/simulated_signal_data.csv")) # for testing

poll_interval = 2      # seconds between polls
timeout_seconds = 120   # time to wait for new data before exiting

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

last_processed_index = window_size
last_update_time = time.time()
done = False

print("Waiting for data to appear...")
counter = 0

while (1):
    # Load the latest CSV
    try:
        df = pd.read_csv(csv_path).rename(columns={
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
            print("Waiting for new data...")
            time.sleep(poll_interval)
            continue

    # New data is available
    last_update_time = time.time()

    while last_processed_index < len(df):
        i = last_processed_index
        current_window_clean = df.iloc[i - window_size:i]["Clean Signal"].tolist()
        current_window_noisy = df.iloc[i - window_size:i]["Noisy Signal"].tolist()

        # For the first iteration, reset the environment
        if i == window_size:
            state = env.reset(clean_signal=np.array(current_window_clean),
                              noisy_signal=np.array(current_window_noisy))

        # Prepare state
        state = np.expand_dims(state, axis=0)
        action, _ = model.predict(state, deterministic=True)

        next_state, reward, done, info = env.step(action)
        scale = info["s"]

        snr_raw = info["SNR_raw"]
        snr_filtered = info["SNR_filtered"]
        filtered_signal = info["filtered_signal"]*scale
        t_factor = info["threshold_factor"]

        rewards.append(reward)
        thresholds.append(t_factor)
        snr_raw_list.append(snr_raw)
        snr_filtered_list.append(snr_filtered)
        snr_improvement.append(snr_filtered_list[-1] - snr_raw_list[-1])
        mse.append(np.square(np.subtract(snr_filtered, snr_raw)).mean())

        clean_signal_data.extend(current_window_clean)
        noisy_signal_data.extend(current_window_noisy)
        filtered_signal_data.extend(filtered_signal)

        if counter == 1000:
            counter = 0
            print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Improvement: {snr_improvement[-1]:.2f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {np.mean(filtered_signal):.4f} | clean signal: {np.mean(current_window_clean):.4f} | threshold factor: {t_factor:.4f}")
        else:
            counter = counter + 1

        results_rows.append({
            "window": f"({i - window_size}, {i})",
            "action": action,
            "reward": reward,
            "snr_improvement": snr_improvement,
            "threshold_factor": t_factor
        })

        # Update sliding window for environment
        current_window_clean.pop(0)
        current_window_noisy.pop(0)
        current_window_clean.append(df.iloc[i]["Clean Signal"])
        current_window_noisy.append(df.iloc[i]["Noisy Signal"])

        env.set_signal_window(np.array(current_window_clean), np.array(current_window_noisy))

        # Update state
        state = next_state

        # Increment index
        last_processed_index += 1

        if done:
            #print(f"Early termination signaled by environment at index {i}.")
            
            results_rows.append({
            "window": f"(DONE)",
            "action": np.NaN,
            "reward": np.NaN,
            "snr_improvement": np.NaN,
            "threshold_factor": np.NaN
            })
            

    time.sleep(poll_interval)

env.close()

# Save results
os.makedirs("Data", exist_ok=True)
#pd.DataFrame(results_rows).to_csv("Data/results.csv", index=False)

print("Inference complete. Results saved.")


snr_improvement = np.array(snr_improvement)
x = np.arange(len(snr_improvement))
coeffs = np.polyfit(x, snr_improvement, deg=1)
trendline = np.polyval(coeffs, x)

# --- Visualization ---
plt.figure(figsize=(12, 6))

# Signal comparison
plt.subplot(3, 1, 1)
plt.plot(clean_signal_data, label="Clean Signal", color="blue", alpha=0.8)
plt.plot(noisy_signal_data, label="Noisy Signal", color="orange", alpha=0.5)
plt.plot(filtered_signal_data, label="Filtered Signal", color="green", alpha=0.8)
plt.title("Clean vs. Noisy vs. Filtered Signal with SAC")
plt.ylabel("Signal Amplitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(snr_improvement, label="SNR Improvement", color="red")
plt.plot(x, trendline, label="Trendline", color="blue", linestyle='--')
plt.axhline(y=0, color='black', linestyle='--')
plt.ylabel("SNR Improvement")
plt.title("SNR Improvement Over Time")
plt.legend()

# # MSE evolution
# plt.subplot(3, 1, 3)
# plt.plot(mse, label="MSE (SNR Filtered vs. Raw)", color="blue")
# plt.xlabel("Time Step")
# plt.ylabel("MSE")
# plt.title("MSE Over Time")
# plt.legend()

plt.subplot(3, 1, 3)
plt.plot(thresholds, label="Threshold over time", color="blue")
plt.xlabel("Time Step")
plt.ylabel("Threshold")
plt.title("Threshold Over Time")
plt.legend()

plt.tight_layout()
plt.show()
