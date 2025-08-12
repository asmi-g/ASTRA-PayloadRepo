import time
import sys
import os
import csv
import shutil
from datetime import datetime

sys.path.append('/home/nvidia/.local/lib/python3.6/site-packages')

try:
    import board
except NotImplementedError as e:
    print("Board module not available. Ensure sensor is connected and the correct libraries are installed.")
    board = None
    sys.exit(1)

import busio
from Adafruit_MCP9808 import MCP9808

# === Directories ===
BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
DATA_DIR = os.path.join(BASE_DIR, "Scripts/SystemTesting/Data/")
SD_LOG_PATH = "/media/nvidia/sdcard/temperature_log.csv"

# === Timestamped local log file ===
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
LOCAL_LOG_PATH = os.path.join(DATA_DIR, f"temperature_log_{timestamp}.csv")

print(f"[INFO] Local log: {LOCAL_LOG_PATH}")
print(f"[INFO] SD card target: {SD_LOG_PATH}")

# === Init I2C and sensor ===
i2c = busio.I2C(board.SCL, board.SDA)
sensor = MCP9808.MCP9808(busnum=1)


def save_to_sd(local_path, sd_path):
    try:
        os.makedirs(os.path.dirname(sd_path), exist_ok=True)
        shutil.copy2(local_path, sd_path)
        print(f"[INFO] Synced {local_path} -> {sd_path}")
    except Exception as e:
        print(f"[ERROR] Could not sync to SD card: {e}")


def log_temperature(log_file, interval=5):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Temperature (C)"])
        print("Logging temperature. Press Ctrl+C to stop.")
        try:
            while True:
                temperature = sensor.readTempC()
                timestamp = datetime.now().isoformat()
                writer.writerow([timestamp, f"{temperature:.2f}"])
                file.flush()

                print(f"{timestamp} - Temperature: {temperature:.2f} °C")

                # Save to SD on each write (or modify to sync every N seconds)
                save_to_sd(log_file, SD_LOG_PATH)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("[INFO] Temperature logging stopped.")


if __name__ == "__main__":
    log_temperature(LOCAL_LOG_PATH, interval=5)
