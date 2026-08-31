import pandas as pd
import matplotlib.pyplot as plt

#csv_path = r"D:\signal.csv"
csv_path = r"c:\Users\imanq\Documents\Programs\GitHub\ASTRA-GeneralRepo\Data\flight_signal_1.csv"
chunksize = 200_000
downsample_factor = 200

usecols = [
    "Index",
    "TX Real", "TX Imag", "TX Magnitude",
    "RX Real", "RX Imag", "RX Magnitude"
]

df_list = []

print("Loading and downsampling...")

for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols):

    # Ensure Index is numeric
    chunk["Index"] = pd.to_numeric(chunk["Index"], errors="coerce")

    # Drop rows where Index failed to convert
    chunk = chunk.dropna(subset=["Index"])

    # Downsample every N rows
    chunk_ds = chunk.iloc[::downsample_factor]

    df_list.append(chunk_ds)

df = pd.concat(df_list, ignore_index=True)

# Sort to prevent zig-zag plotting
df = df.sort_values("Index").reset_index(drop=True)

print(f"Reduced dataset to {len(df)} rows after downsampling.")

# ---------------- Plotting ----------------

plt.figure(figsize=(14, 10))
x = df["Index"]

plt.subplot(3, 2, 1)
plt.plot(x, df["TX Magnitude"], color="blue")
plt.title("TX Magnitude"); plt.grid()

plt.subplot(3, 2, 2)
plt.plot(x, df["TX Real"], color="cyan")
plt.title("TX Real"); plt.grid()

plt.subplot(3, 2, 3)
plt.plot(x, df["TX Imag"], color="purple")
plt.title("TX Imag"); plt.grid()

plt.subplot(3, 2, 4)
plt.plot(x, df["RX Magnitude"], color="red")
plt.title("RX Magnitude"); plt.grid()

plt.subplot(3, 2, 5)
plt.plot(x, df["RX Real"], color="orange")
plt.title("RX Real"); plt.grid()

plt.subplot(3, 2, 6)
plt.plot(x, df["RX Imag"], color="green")
plt.title("RX Imag"); plt.grid()

plt.tight_layout()
plt.savefig("sdr_results_flight_signal_1.png", dpi=150)
plt.show()
