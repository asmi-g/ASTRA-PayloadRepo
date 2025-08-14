import os, time
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import SAC
from astra_rev1.envs import NoiseReductionEnv  # fixed-size obs
import matplotlib.pyplot as plt  # (optional)

# ---------------- Config ----------------
WINDOW_SIZE = 10
POLL_INTERVAL = 2
TIMEOUT_SECONDS = 10
N_STACK = 4  # must match training

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
DATA_DIR   = os.path.join(BASE_DIR, "Scripts/SDR/Data/")
CSV_PATH = os.path.join(DATA_DIR, "signal.csv")
#CSV_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data/simulated_signal_data.csv")) # for testing
MODEL_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../models/sac_noise_reduction_081225_9pm.zip"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Load ----------------
model = SAC.load(MODEL_PATH, custom_objects={"lr_schedule": lambda x: 0.003, "clip_range": lambda x: 0.02})
env = NoiseReductionEnv()

# ---------------- State / buffers ----------------
frame_buf = deque(maxlen=N_STACK)
last_processed = WINDOW_SIZE - 1
last_update = time.time()
done = False
log_every = 1000
log_counter = 0

# metrics / output
rewards, thresholds = [], []
snr_raw_list, snr_filtered_list, snr_improvement = [], [], []
clean_signal_data, noisy_signal_data, filtered_signal_data = [], [], []
results_rows = []

# ---------------- Helpers ----------------
def stack_obs(obs_1d: np.ndarray) -> np.ndarray:
    if not frame_buf:
        for _ in range(N_STACK):
            frame_buf.append(obs_1d)
    else:
        frame_buf.append(obs_1d)
    return np.concatenate(frame_buf, axis=0)

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"TX Magnitude": "Noisy Signal", "RX Magnitude": "Clean Signal"})
    if not {"Clean Signal", "Noisy Signal"}.issubset(df.columns):
        raise ValueError("CSV missing required columns.")
    return df

# ---------------- Main ----------------
print("Waiting for data to appear...")

while True:
    try:
        df = read_csv(CSV_PATH)
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
        print("signal.csv not ready or invalid. Retrying...\n", CSV_PATH)
        time.sleep(POLL_INTERVAL)
        continue

    if len(df) < WINDOW_SIZE:
        print("Not enough data yet...")
        time.sleep(POLL_INTERVAL)
        continue

    if last_processed >= len(df):
        if time.time() - last_update > TIMEOUT_SECONDS:
            print("No new data detected for timeout period. Exiting.")
            break
        print("Waiting for new data...")
        time.sleep(POLL_INTERVAL)
        continue

    last_update = time.time()

    while last_processed <= len(df) - 1:
        i = last_processed

        # right-aligned window [i-W+1, i]
        win_clean = df.iloc[i - WINDOW_SIZE + 1 : i + 1]["Clean Signal"].to_numpy()
        win_noisy = df.iloc[i - WINDOW_SIZE + 1 : i + 1]["Noisy Signal"].to_numpy()

        # initialize env & stack on first window
        if i == WINDOW_SIZE - 1:
            state = env.reset(clean_signal=win_clean, noisy_signal=win_noisy)
            state_in = stack_obs(np.asarray(state, dtype=np.float32).reshape(-1))
        else:
            state_in = stack_obs(np.asarray(state, dtype=np.float32).reshape(-1))

        # predict + step
        action, _ = model.predict(state_in, deterministic=True)
        next_state, reward, done, info = env.step(action)

        # info / metrics
        snr_raw = info.get("SNR_raw")
        snr_flt = info.get("SNR_filtered")
        filt    = np.asarray(info.get("filtered_signal", [])) * np.max(np.abs(win_clean))
        thr     = info.get("threshold_factor", np.nan)

        rewards.append(float(reward))
        thresholds.append(float(thr) if thr is not None else np.nan)
        snr_raw_list.append(float(snr_raw) if snr_raw is not None else np.nan)
        snr_filtered_list.append(float(snr_flt) if snr_flt is not None else np.nan)
        snr_improvement.append((snr_flt - snr_raw) if (snr_raw is not None and snr_flt is not None) else np.nan)

        # synced saving
        if i == WINDOW_SIZE - 1:
            clean_signal_data.extend(win_clean.tolist())
            noisy_signal_data.extend(win_noisy.tolist())
            filtered_signal_data.extend((filt if filt.size == WINDOW_SIZE else np.resize(filt, WINDOW_SIZE)).tolist())
        else:
            clean_signal_data.append(float(df.iloc[i]["Clean Signal"]))
            noisy_signal_data.append(float(df.iloc[i]["Noisy Signal"]))
            filtered_signal_data.append(float(filt[-1]) if filt.size else np.nan)

        # periodic log
        if (log_counter % log_every) == 0:
            si = snr_improvement[-1] if snr_improvement else np.nan
            sr = snr_raw_list[-1] if snr_raw_list else np.nan
            sf = snr_filtered_list[-1] if snr_filtered_list else np.nan
            fmean = float(np.mean(filt)) if filt.size else np.nan
            print(f"Rows {(i - WINDOW_SIZE, i)} | a={action} r={reward:.4f} ΔSNR={si:.2f} "
                  f"SNR(raw,flt)=({sr:.2f},{sf:.2f}) done={done} filt_mean={fmean:.4f} thr={thr:.4f}")
        log_counter += 1

        results_rows.append({
            "window": f"({i - WINDOW_SIZE + 1}, {i})",
            "action": float(action[0]) if np.ndim(action) else float(action),
            "reward": float(reward),
            "snr_improvement": float(snr_improvement[-1]),
            "threshold_factor": float(thr) if thr is not None else np.nan
        })

        # advance to next window
        if i + 1 < len(df):
            next_win_clean = np.r_[win_clean[1:], df.iloc[i + 1]["Clean Signal"]]
            next_win_noisy = np.r_[win_noisy[1:], df.iloc[i + 1]["Noisy Signal"]]
            env.set_signal_window(next_win_clean, next_win_noisy)

        state = next_state
        last_processed += 1

        if done:  # mark, then continue streaming
            results_rows.append({"window": "(DONE)", "action": np.nan, "reward": np.nan,
                                 "snr_improvement": np.nan, "threshold_factor": np.nan})
            done = False

    time.sleep(POLL_INTERVAL)

env.close()

# save results
pd.DataFrame(results_rows).to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
print("Inference complete. Results saved.")

snr_improvement = np.array(snr_improvement)
valid_improvements = [v for v in snr_improvement if not np.isnan(v)]

if valid_improvements:
    avg_snr_imp = np.mean(valid_improvements)
    print(f"Average SNR Improvement: {avg_snr_imp:.2f} dB over {len(valid_improvements)} steps")
else:
    print("No valid SNR improvement values recorded.")


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
