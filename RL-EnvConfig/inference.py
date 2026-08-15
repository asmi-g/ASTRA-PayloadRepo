#inference.py (dev version, with signal reconstruction export)
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv
import os
import time
import shutil
from datetime import datetime

last_update_time = datetime.now()
timestamp_str = last_update_time.strftime("%Y%m%d_%H%M%S")

# Load SAC model
custom_objects = {
    "lr_schedule": lambda x: 0.003,
    "clip_range": lambda x: 0.02
}

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(script_dir, "../models/sac_noise_reduction_080725_6am"))
model = SAC.load(model_path, custom_objects=custom_objects)

BASE_DIR = "/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Data/")
csv_path = "/Users/imanq/Downloads/simulated_signal_data.csv"
#csv_path = os.path.join(DATA_DIR, "flight_signal_2.csv")
RESULTS_PATH = os.path.join(DATA_DIR, f"{timestamp_str}_results_ws10s1.csv")
SIGNAL_PATH = os.path.join(DATA_DIR, f"{timestamp_str}_signal_ws10s1.csv")


def save_results_direct():
    try:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        write_header = not os.path.exists(RESULTS_PATH)

        df = pd.DataFrame(results_rows)
        df.to_csv(
            RESULTS_PATH,
            mode='a',              # append instead of overwrite
            header=write_header,   # only write header if file doesn't exist
            index=False
        )

        #print(f"[INFO] Results appended to {RESULTS_PATH}")
        # clear the buffer so we don't re-append the same rows next flush
        results_rows.clear()
        
    except Exception as e:
        print(f"[ERROR] Could not save results: {e}")


def save_results_and_sync():
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(RESULTS_PATH, index=False)


# --- NEW: save the reconstructed signal, one row per original sample ---
def save_signal_data():
    try:
        os.makedirs(os.path.dirname(SIGNAL_PATH), exist_ok=True)
        write_header = not os.path.exists(SIGNAL_PATH)

        df_signal = pd.DataFrame({
            "sample_index": signal_index_data,
            "clean_signal": clean_signal_data,
            "noisy_signal": noisy_signal_data,
            "filtered_signal": filtered_signal_data,
        })
        df_signal.to_csv(
            SIGNAL_PATH,
            mode='a',
            header=write_header,
            index=False
        )

        #print(f"[INFO] Signal data appended to {SIGNAL_PATH}")

        # clear the buffers so we don't write the same rows twice next flush
        signal_index_data.clear()
        clean_signal_data.clear()
        noisy_signal_data.clear()
        filtered_signal_data.clear()
    except Exception as e:
        print(f"[ERROR] Could not save signal data: {e}")

# Parameters
window_size = 10
stride = 1

# Initialize environment
env = NoiseReductionEnv()

poll_interval = 2      # seconds between polls
timeout_seconds = 120   # time to wait for new data before exiting

# Tracking
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []
snr_improvement = []
signal_index_data = []  # --- NEW: original row index for each sample ---
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
        # --- column mapping from script 2 ---
        df = pd.read_csv(csv_path).rename(columns={
            'RX Magnitude': 'Noisy Signal',
            'TX Magnitude': 'Clean Signal'
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
        else:
            env.set_signal_window(win_clean, win_noisy)
            state = env._get_state()   # refresh state to match the window you just set
        # if i > window_size - 1 + 5:  # after a few iterations
        #     print("ARRAY STATE EQUALITY CHECK: ", np.array_equal(state, prev_state), np.max(np.abs(state - prev_state)))
        # prev_state = state.copy()

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
        # The denoiser only ever filters `filt.shape[0]` samples starting at
        # the FRONT of the current context window (self.t is fixed at 0),
        # not the newest sample. Compute the true absolute index of filt[-1]
        # so clean/noisy/filtered stay aligned and always the same length.
        n_filt = filt.shape[0]
        context_start = i - window_size + 1
        last_idx = context_start + n_filt - 1  # absolute row index for filt[-1]

        signal_index_data.append(last_idx)
        clean_signal_data.append(float(df.iloc[last_idx]["Clean Signal"]))
        noisy_signal_data.append(float(df.iloc[last_idx]["Noisy Signal"]))
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

        if counter % 100 == 0:
            save_results_direct()
            save_signal_data()  # --- NEW ---

        state = next_state
        last_processed_index += stride

        if done:
            results_rows.append({
                "window": f"(DONE)",
                "action": np.NaN,
                "reward": np.NaN,
                "snr_improvement": np.NaN,
                "threshold_factor": np.NaN
            })

    time.sleep(poll_interval)

env.close()

save_results_direct()
save_signal_data()  # --- NEW: flush any remaining buffered samples ---

print("Inference complete. Results saved.")
print(f"Signal data saved to {SIGNAL_PATH}")