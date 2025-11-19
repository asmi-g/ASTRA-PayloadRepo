import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv
import os
import time
import shutil

# Load SAC model
custom_objects = {
    "lr_schedule": lambda x: 0.003,
    "clip_range": lambda x: 0.02
}

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(script_dir, "../models/sac_noise_reduction_080725_6am"))
model = SAC.load(model_path, custom_objects=custom_objects)
<<<<<<< HEAD

SD_RESULTS_PATH = "/media/nvidia/sdcard/ml_results.csv"  # Modify if SD label is different
SD_SIGNAL_PATH = "/media/nvidia/sdcard/sdr_data.csv"  # Adjust mount path if needed
csv_path = SD_SIGNAL_PATH


def save_results_direct():
    try:
        os.makedirs(os.path.dirname(SD_RESULTS_PATH), exist_ok=True)
        write_header = not os.path.exists(SD_RESULTS_PATH)

        df = pd.DataFrame(results_rows)
        df.to_csv(
            SD_RESULTS_PATH,
            mode='a',              # append instead of overwrite
            header=write_header,   # only write header if file doesn't exist
            index=False
        )

        print(f"[INFO] Results appended to {SD_RESULTS_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not sync to SD card: {e}")

def save_results_and_sync():
    os.makedirs("Data", exist_ok=True)
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv("Data/results.csv", index=False)
    save_to_sd("Data/results.csv", SD_RESULTS_PATH)

=======
SD_DIR = "/media/nvidia/sdcard/"
# Results CSV will be written directly here
SD_RESULTS_PATH = os.path.join(SD_DIR, "ml_results.csv")
# The incoming signal.csv is expected here too
csv_path = os.path.join(SD_DIR, "signal.csv")
>>>>>>> 60852406fe419520d80590df73e7b5af47775af2

# Initialize environment
env = NoiseReductionEnv()

# Parameters
<<<<<<< HEAD
window_size = 1000

BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"

=======
window_size = 10
>>>>>>> 60852406fe419520d80590df73e7b5af47775af2

poll_interval = 2      # seconds between polls
timeout_seconds = 120   # time to wait for new data before exiting


def save_results():  # Does this over write?
    df_out = pd.DataFrame(results_rows)
    os.makedirs(SD_DIR, exist_ok=True)
    df_out.to_csv(SD_RESULTS_PATH, index=False)


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

        if counter == 1000:
            counter = 0
            print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Improvement: {snr_improvement[-1]:.2f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {np.mean(filtered_signal_data):.4f} | clean signal: {np.mean(win_clean):.4f} | threshold factor: {t_factor:.4f}")
        else:
            counter = counter + 1

        results_rows.append({
            "window": f"({i - window_size + 1}, {i})",
            "action": action,
            "reward": reward,
            "snr_improvement": snr_improvement[-1],
            "threshold_factor": t_factor
        })

<<<<<<< HEAD
        
        if counter % 100 == 0:
            save_results_direct()
=======
        # prepare next window for the env (only if there *is* a next sample)
        if i + 1 < len(df):
            next_win_clean = np.r_[win_clean[1:], df.iloc[i + 1]["Clean Signal"]]
            next_win_noisy = np.r_[win_noisy[1:], df.iloc[i + 1]["Noisy Signal"]]
            env.set_signal_window(next_win_clean, next_win_noisy)
>>>>>>> 60852406fe419520d80590df73e7b5af47775af2

        state = next_state
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

<<<<<<< HEAD
save_results_direct()

=======
# Save results
# os.makedirs("Data", exist_ok=True)
# pd.DataFrame(results_rows).to_csv("Data/results.csv", index=False)
#save_to_sd("Data/results.csv", SD_RESULTS_PATH)
save_results()
>>>>>>> 60852406fe419520d80590df73e7b5af47775af2

print("Inference complete. Results saved.")
