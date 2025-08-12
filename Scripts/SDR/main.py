import subprocess
import multiprocessing as mp
import time
import csv
import sys
import os
import signal
import platform
from datetime import datetime

# CUDA check
try:
    import cupy as np # CuPy is a GPU-accelerated library similar to NumPy
    print("CuPy: CUDA is available")
    print("Using CuPy (GPU)")
except ImportError:
    import numpy as np
    print("Using NumPy (CPU fallback)")
    try:
        import torch
        print(torch.__version__)
        print('PyTorch CUDA available:', torch.cuda.is_available())
        print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
        print("PyTorch CUDA:", torch.cuda.is_available())
    except:
        print("CUDA not available")


#TODO: start timmer so we start logging after 1-2 hours?

BASE_DIR = "/home/nvidia/Projects/ASTRA/ASTRA-GeneralRepo/"
print(f"Base directory: {BASE_DIR}")

#TODO: add in error handling in case this folder is deleted (aka create the folder if it doesn't exist)
DATA_DIR = os.path.join(BASE_DIR, "Scripts/SDR/Data/")
print(f"Data directory: {DATA_DIR}")

TX_SCRIPT = os.path.join(BASE_DIR, "Scripts/SDR/TX.py")
print(f"TX script: {TX_SCRIPT}")

RX_SCRIPT = os.path.join(BASE_DIR, "Scripts/SDR/RX.py")
print(f"RX script: {RX_SCRIPT}")

ML_SCRIPT = os.path.join(BASE_DIR, "RL-EnvConfig/inference.py")
print(f"ML script: {ML_SCRIPT}")

TEMP_LOGGER_SCRIPT = os.path.join(BASE_DIR, "Scripts/SystemTesting/temperature_logger.py")
print(f"Temp logger script: {TEMP_LOGGER_SCRIPT}")

CSV_FILE_PATH = os.path.join(DATA_DIR, "signal.csv")
print(f"CSV file path: {CSV_FILE_PATH}")

RUNTIME_SECONDS = 10  # duration to run TX/RX per cycle

# TO DO
# - Fix: "sink :warning: Soapy sink error: TIMEOUT"
# - Integrate Chelsea's comments from previous pr
# - Integrate AM scripts, address throttle block error and rerun on WSL
# - Integrate timed operation for flight

EXPECTED_SERIALS = [
    "0000000000000000a18c63dc2a8a8313",
    "0000000000000000a18c63dc2a7f8313"
]

def get_soapy_devices():
    try:
        output = subprocess.check_output(["SoapySDRUtil", "--find"]).decode()
        return output
    except subprocess.CalledProcessError:
        return ""

def needs_reset(soapy_output):
    count = soapy_output.count("HackRF One")
    serials = [line for line in soapy_output.splitlines() if "serial" in line.lower()]
    unique_serials = set(s.strip().split('=')[-1] for s in serials)

    if count < 2:
        return True
    if len(unique_serials) < 2:
        return True
    if not all(expected in soapy_output for expected in EXPECTED_SERIALS):
        return True
    return False

def reset_hackrfs():
    print("[INFO] Resetting HackRF devices via CPLD JTAG...")
    subprocess.call(["hackrf_cpldjtag", "--reset"])
    time.sleep(3)

def can_open_hackrf(serial):
    try:
        out = subprocess.check_output(["hackrf_info"]).decode()
        return serial in out
    except:
        return False

def check_sdrs():
    attempt = 1
    while True:
        print(f"[INFO] Checking HackRF devices (Attempt {attempt})...")
        devices = get_soapy_devices()

        if needs_reset(devices):
            print("[WARNING] HackRFs not enumerated properly. Attempting reset.")
            reset_hackrfs()
            time.sleep(2)
            attempt += 1
            continue

        # Both devices enumerated — now check if we can open them
        print("[INFO] HackRFs found. Checking openability...")
        time.sleep(1)  # give some USB time

        all_open = all(can_open_hackrf(serial) for serial in EXPECTED_SERIALS)
        if all_open:
            print("[INFO] HackRF devices recognized AND openable.")
            break
        else:
            print("[WARNING] HackRF devices cannot be opened. Resetting again.")
            reset_hackrfs()
            time.sleep(2)
            attempt += 1



def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Failed to install some packages. Continuing anyway...")


def run_script(script_path):
    if platform.system() == "Windows":
        return subprocess.Popen(["python", script_path])
    else:
        return subprocess.Popen(["python3", script_path], preexec_fn=os.setsid)


def terminate_process(proc):
    if proc.poll() is not None:
        # Process already terminated
        return

    try:
        if platform.system() == "Windows":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process already exited
        pass



def save_to_csv(rx_file_path, tx_file_path, csv_file_path):
    rx_data = np.fromfile(open(rx_file_path), dtype=np.complex64)
    tx_data = np.fromfile(open(tx_file_path), dtype=np.complex64)

    tx_data_last = tx_data[-500000:] if len(tx_data) >= 500000 else tx_data
    rx_data_last = rx_data[-500000:] if len(rx_data) >= 500000 else rx_data

    write_header = not os.path.exists(csv_file_path)

    #TODO: add try except. 
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Index", "Timestamp (ISO)", 
                             "TX Real", "TX Imag", "TX Magnitude", 
                             "RX Real", "RX Imag", "RX Magnitude"])

        for i in range(len(tx_data_last)):
            timestamp = datetime.now().isoformat()
            tx = tx_data_last[i]
            rx = rx_data_last[i] if i < len(rx_data_last) else 0
            writer.writerow([
                i, timestamp,
                np.real(tx), np.imag(tx), np.abs(tx),
                np.real(rx), np.imag(rx), np.abs(rx)
            ])


def SDR_cycle():
    print("[INFO] Checking SDRs before starting TX/RX...")
    check_sdrs()  # <-- Add this line to verify and reset HackRFs if needed


    print("Launching TX and RX scripts...")
    tx_proc = run_script(TX_SCRIPT)
    rx_proc = run_script(RX_SCRIPT)

    # Wait briefly to allow for early crash
    time.sleep(3)
    if tx_proc.poll() is not None or rx_proc.poll() is not None:
        print("[ERROR] One or both SDR scripts crashed at startup. Retrying...")
        terminate_process(tx_proc)
        terminate_process(rx_proc)
        reset_hackrfs()
        time.sleep(2)
        return  # Exit this cycle early    

    print(f"Running for {RUNTIME_SECONDS} seconds...")
    time.sleep(RUNTIME_SECONDS)

    print("Terminating scripts...")
    terminate_process(tx_proc)
    terminate_process(rx_proc)

    print("Saving to CSV...")
    save_to_csv(os.path.join(DATA_DIR, "rxdata.dat"),
                os.path.join(DATA_DIR, "txdata.dat"),
                CSV_FILE_PATH)
    with open(CSV_FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    print(f"CSV has {row_count} rows")

    print("Cycle complete.\n")


def main():
    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Launch temperature logging script
    print("Launching temperature logger...")
    temp_logger_proc = run_script(TEMP_LOGGER_SCRIPT)

    # Launch ML inference script once
    print("Launching ML model...")
    ml_proc = run_script(ML_SCRIPT)

    # SDR capture loop
    try:
        while True:
            SDR_cycle()
            time.sleep(2)  # Optional delay between cycles
    except KeyboardInterrupt:
        print("Terminating Model Operation...")
        terminate_process(ml_proc)

        print("Terminating temperature logging...")
        terminate_process(temp_logger_proc)


if __name__ == "__main__":
    main()