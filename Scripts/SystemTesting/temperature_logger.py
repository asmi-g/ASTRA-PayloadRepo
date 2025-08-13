# First we need to make sure the Jetson is set up for I2C
# Make sure I2C is enabled (e.g. /dev/i2c-1 is available). You can check with:
# ls /dev/i2c-*
# You should see something like /dev/i2c-1.

import time
import os
import sys
sys.path.append('/home/nvidia/.local/lib/python3.6/site-packages')

import board
import busio
# from Adafruit_MCP9808 import MCP9808
import Adafruit_MCP9808.MCP9808 as MCP9808


def find_mcp9808():
  # Initialize I2C connection (Jetson TX2 usually uses I2C bus 1)
  # i2c = busio.I2C(board.SCL_1, board.SDA_1)
  try:
    # Initialize MCP9808 sensor
    print("Checking now")
    sensor = MCP9808.MCP9808(busnum=1)
    print("Called sensor")
    sensor.begin()
    print(sensor.readTempC())
    return sensor
  except (ValueError, OSError):
    return None

# Function to log temperature
def log_temperature(log_file, interval=1):
  """
  Logs temperature readings from the MCP9808 sensor.

  :param log_file: Path to the log file
  :param interval: Time interval between readings in seconds
  """
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
        # Wait for the specified interval
        time.sleep(interval)
    except KeyboardInterrupt:
      print("Logging stopped.")

# Main function
if __name__ == "__main__":
  sensor = find_mcp9808()
  if sensor is None:
    print("No sensor detected.")
  else:
    print("Temp sensor detected!")
  timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
  log_file_path = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/Scripts/SDR/Data/temperature_log_{}.csv".format(timestamp)
  # log_interval = 5  # Set logging interval in seconds
  # log_temperature(log_file_path, log_interval)
