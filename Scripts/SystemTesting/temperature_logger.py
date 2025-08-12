# First we need to make sure the Jetson is set up for I2C
# Make sure I2C is enabled (e.g. /dev/i2c-1 is available). You can check with:
# ls /dev/i2c-*
# You should see something like /dev/i2c-1.

import time
import sys
import os
import shutil


sys.path.append('/home/nvidia/.local/lib/python3.6/site-packages')

try:
  import board
except NotImplementedError as e:
  print("Board module not available. Ensure sensor is corrected and the correct libraries are installed.")
  board = None
  sys.exit(1)
  
import busio
from Adafruit_MCP9808 import MCP9808

BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, "Scripts/SDR/Data/")
os.makedirs(DATA_DIR, exist_ok=True)

SD_FILE_PATH = "/media/nvidia/sdcard/temperature_log.csv"


timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
TEMP_LOG_CSV = os.path.join(DATA_DIR, "Data/temperature_log_{}.csv".format(timestamp))
print(f"Temp logger CSV: {TEMP_LOG_CSV}")


# Initialize I2C connection (Jetson TX2 usually uses I2C bus 1)
i2c = busio.I2C(board.SCL, board.SDA)
# Initialize MCP9808 sensor
sensor = MCP9808.MCP9808(busnum=1)

def save_to_sd(local_path, sd_path):
    try:
        os.makedirs(os.path.dirname(sd_path), exist_ok=True)
        shutil.copy2(local_path, sd_path)
        print(f"[INFO] Synced {local_path} -> {sd_path}")
    except Exception as e:
        print(f"[ERROR] Could not sync to SD card: {e}")


# Function to log temperature
def log_temperature(log_file, interval=5):
  """
  Logs temperature readings from the MCP9808 sensor.

  :param log_file: Path to the log file
  :param interval: Time interval between readings in seconds
  """
  os.makedirs(os.path.dirname(log_file), exist_ok=True)
  with open(log_file, 'a') as file:
    file.write("Timestamp,Temperature (C)\n")  # Write header
    print("Logging temperature. Press Ctrl+C to stop.")
    try:
      while True:
        # Read temperature
        temperature = sensor.readTempC()
        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Log to file
        file.write("{}, {:.2f}\n".format(timestamp, temperature))
        file.flush()  # write to disk immediately
        print("{} - Temperature: {:.2f} C".format(timestamp, temperature))
        save_to_sd(log_file,SD_FILE_PATH)
        # Wait for the specified interval
        time.sleep(interval)
    except KeyboardInterrupt:
      print("Logging stopped.")

# Main function
if __name__ == "__main__":
  log_interval = 5  # Set logging interval in seconds
  log_temperature(TEMP_LOG_CSV, interval=5)
