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
model_path = os.path.normpath(os.path.join(script_dir, "../models/OFT_W10_080725_6AM_05_125_075.zip"))
#model_path = os.path.normpath(os.path.join(script_dir, "../models/best_model_20260816_161412_100000/BFT_W1000_081626_6PM"))
model = SAC.load(model_path, custom_objects=custom_objects)

BASE_DIR = "/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Data/")

# --- NEW: single switch to change data source. Nothing else needs to change. ---
# "flight"    -> uses TX/RX Real+Imag columns, runs alignment + shared scaling
# "simulated" -> uses pre-built "Clean Signal"/"Noisy Signal" columns directly,
#                no alignment or scaling needed (simulator already emits a
#                matched, comparable-scale clean/noisy pair)
DATA_SOURCE = "simulated"  # "flight" or "simulated"

if DATA_SOURCE == "simulated":
    csv_path = "/Users/imanq/Downloads/simulated_signal_data.csv"
    RUN_TAG = "sim"
else:
    csv_path = os.path.join(DATA_DIR, "flight_signal_2.csv")
    RUN_TAG = "fs2"

RESULTS_PATH = os.path.join(DATA_DIR, f"{timestamp_str}_{RUN_TAG}_results_OFT_W10_080725_6AM_05_125_075.csv")
SIGNAL_PATH = os.path.join(DATA_DIR, f"{timestamp_str}_{RUN_TAG}_signal_OFT_W10_080725_6AM_05_125_075.csv")

def estimate_alignment(tx, rx, fs, f0_nominal=100_000):
    n = min(len(tx), len(rx))
    tx, rx = tx[:n], rx[:n]

    RX_F = np.fft.fft(rx * np.hanning(n))
    freqs = np.fft.fftfreq(n, d=1/fs)
    mask = np.abs(freqs - f0_nominal) < 5000
    f0_actual = freqs[mask][np.argmax(np.abs(RX_F[mask]))]
    df_hat = f0_actual - f0_nominal

    t = np.arange(n) / fs
    rx_corrected = rx * np.exp(-1j * 2 * np.pi * df_hat * t)
    A_hat = np.vdot(tx, rx_corrected) / np.vdot(tx, tx)

    # robust scale: fit on RX only, using a high percentile instead of max() so a single spike (e.g. the -0.35 outlier in the RX Real plot) doesn't compress every other sample toward zero
    scale = np.percentile(np.abs(rx), 99.5)
    if scale == 0:
        aligned_clean_calib = (A_hat * tx * np.exp(1j * 2 * np.pi * df_hat * t))
        scale = np.percentile(np.abs(aligned_clean_calib), 99.5) or 1.0

    return {"df_hat": df_hat, "A_hat": A_hat, "fs": fs, "scale": scale}

def apply_alignment(tx_window, params, n_start_idx):
    df_hat, A_hat, fs = params["df_hat"], params["A_hat"], params["fs"]
    n = np.arange(n_start_idx, n_start_idx + len(tx_window))
    return A_hat * tx_window * np.exp(1j * 2 * np.pi * df_hat * n / fs)

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

# --- save the reconstructed signal, one row per original sample ---
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
CALIB_SIZE = 20_000        # samples used to estimate df/A/scale once (flight mode only)
if RUN_TAG == "sim":
    CALIB_SIZE = 1000
SAMP_RATE = 1_000_000    # matches TX.py / RX.py samp_rate
alignment_params = None  # initialize this above the outer while(1) loop, alongside last_processed_index

# Initialize environment
env = NoiseReductionEnv(window_size=window_size)

poll_interval = 2      # seconds between polls
timeout_seconds = 10   # time to wait for new data before exiting

# Tracking
actions = []
rewards = []
snr_raw_list = []
snr_filtered_list = []
snr_improvement = []
signal_index_data = []  # original row index for each sample
clean_signal_data = []
noisy_signal_data = []
filtered_signal_data = []
thresholds = []
mse = []
results_rows = []

last_processed_index = window_size - 1
last_update_time = time.time()
done = False

print(f"[INFO] DATA_SOURCE = {DATA_SOURCE} | csv_path = {csv_path}")
print("Waiting for data to appear...")
counter = 0

while (1):
    # Load the latest CSV
    try:
        df = pd.read_csv(csv_path)
        if DATA_SOURCE == "simulated":
            # --- simulated CSV already has real-valued, comparable-scale columns ---
            df['Clean Signal'] = df['Clean Signal'].astype(np.float64)
            df['Noisy Signal'] = df['Noisy Signal'].astype(np.float64)
        else:
            # --- flight CSV: build complex IQ, alignment happens later ---
            df['Clean Signal'] = df['TX Real'] + 1j * df['TX Imag']
            df['Noisy Signal'] = df['RX Real'] + 1j * df['RX Imag']
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

    if alignment_params is None:
        if DATA_SOURCE == "simulated":
            if len(df) < CALIB_SIZE:
                print("Waiting for enough data to calibrate scale...")
                time.sleep(poll_interval)
                continue
            clean_calib = df['Clean Signal'].to_numpy()[:CALIB_SIZE]
            noisy_calib = df['Noisy Signal'].to_numpy()[:CALIB_SIZE]
            scale = max(np.max(np.abs(clean_calib)), np.max(np.abs(noisy_calib)))
            alignment_params = {"df_hat": 0.0, "A_hat": 1.0 + 0j, "fs": SAMP_RATE, "scale": scale}
            print(f"[INFO] Simulated mode -- calibrated scale from first {CALIB_SIZE} samples: {alignment_params}")
        else:
            if len(df) < CALIB_SIZE:
                print("Waiting for enough data to calibrate alignment...")
                time.sleep(poll_interval)
                continue
            tx_calib = df['Clean Signal'].to_numpy()[:CALIB_SIZE]
            rx_calib = df['Noisy Signal'].to_numpy()[:CALIB_SIZE]
            alignment_params = estimate_alignment(tx_calib, rx_calib, SAMP_RATE)

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

        start = i - window_size + 1

        if DATA_SOURCE == "simulated":
            # --- direct slice, no alignment/scaling needed ---
            win_clean = df.iloc[start:i + 1]["Clean Signal"].to_numpy().astype(np.float64)
            win_noisy = df.iloc[start:i + 1]["Noisy Signal"].to_numpy().astype(np.float64)
            scale = alignment_params["scale"]
            if scale > 0:
                win_clean = win_clean / scale
                win_noisy = win_noisy / scale
        else:
            # --- flight signal: unchanged calculation from before ---
            win_tx_complex = df.iloc[start:i + 1]["Clean Signal"].to_numpy()
            win_rx_complex = df.iloc[start:i + 1]["Noisy Signal"].to_numpy()

            win_clean_complex = apply_alignment(win_tx_complex, alignment_params, n_start_idx=start)

            win_clean = win_clean_complex.real
            win_noisy = win_rx_complex.real

            # scale clean and noisy by the SAME factor (computed once at calibration)
            # so their ratio -- the real SNR -- is preserved, rather than each being
            # normalized independently.
            scale = alignment_params["scale"]
            if scale > 0:
                win_clean = win_clean / scale
                win_noisy = win_noisy / scale

        if i == window_size - 1:
            state = env.reset(clean_signal=win_clean, noisy_signal=win_noisy)
        else:
            env.set_signal_window(win_clean, win_noisy)
            state = env._get_state()   # refresh state to match the window you just set

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
        running_mean = np.cumsum(snr_improvement) / np.arange(1, len(snr_improvement)+1)

        # --- synced saving ---
        # The denoiser filters `filt.shape[0]` samples starting at the FRONT
        # of the current context window, so filt[-1] corresponds to the
        # absolute row index `last_idx` below.
        n_filt = filt.shape[0]
        context_start = i - window_size + 1
        last_idx = context_start + n_filt - 1  # absolute row index for filt[-1]

        # --- FIXED: save every sample the model actually filtered this step,
        # not just the newest one. When stride > 1, the window still contains
        # `stride` new samples we haven't logged yet (as long as stride <=
        # window_size); pull all of them from the tail of win_clean/win_noisy/filt
        # instead of only the last entry.
        n_new = min(stride, window_size, n_filt)
        for k in range(n_new):
            offset = n_new - 1 - k          # counts down: n_new-1, ..., 0
            idx = last_idx - offset
            signal_index_data.append(idx)
            clean_signal_data.append(float(win_clean[-(offset + 1)]))
            noisy_signal_data.append(float(win_noisy[-(offset + 1)]))
            filtered_signal_data.append(float(filt[-(offset + 1)]))

        if counter == 1000:
            counter = 0
            print(f"Rows {i-window_size, i} | Action: {action} | Reward: {reward:.4f} | SNR Improvement: {snr_improvement[-1]:.2f} | SNR Raw: {snr_raw:.2f} | SNR Filtered: {snr_filtered:.2f} | Done: {done} | filtered signal: {np.mean(filtered_signal_data):.4f} | clean signal: {np.mean(win_clean):.4f} | threshold factor: {t_factor:.4f} | running mean: {running_mean[-1]}")
        else:
            counter = counter + 1

        results_rows.append({
            "window": f"({i - window_size + 1}, {i})",
            "action": action,
            "reward": reward,
            "snr_improvement": snr_improvement[-1],
            "threshold_factor": t_factor,
            "running_mean": running_mean[-1]
        })

        if counter % 100 == 0:
            save_results_direct()
            save_signal_data()

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
save_signal_data()  # flush any remaining buffered samples

print("Inference complete. Results saved.")
print(f"Signal data saved to {SIGNAL_PATH}")