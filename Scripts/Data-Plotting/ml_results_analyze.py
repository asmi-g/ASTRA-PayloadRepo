import pandas as pd
import ast
import matplotlib.pyplot as plt

# ---- Load CSV ----
# If your CSV has tab separation like the sample, use sep="\t"
#df = pd.read_csv("/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/20260813_081622_results.csv", sep=",")
df = pd.read_csv("/Users/imanq/Documents/Programs/GitHub/ASTRA-GeneralRepo/Data/20260816_230337_fs2_results_ws10s1_OFT_save.csv", sep = ",")
print(df.head())
# ---- Parse complex columns ----
# Convert "(0, 9)" → tuple, "[[-1.]]" → float
def parse_window(w):
    try:
        return ast.literal_eval(w)
    except (ValueError, SyntaxError):
        return None  # marks the "(DONE)" rows

df["window"] = df["window"].apply(parse_window)
df["action"] = df["action"].apply(lambda x: float(ast.literal_eval(x)[0][0]))

# Convert scientific notation strings to float if needed
df["snr_improvement"] = pd.to_numeric(df["snr_improvement"], errors="coerce")
df["running_mean"] = pd.to_numeric(df["running_mean"], errors="coerce")
df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
df["threshold_factor"] = pd.to_numeric(df["threshold_factor"], errors="coerce")

# ---- Create an index for plotting (window start, or row number) ----
df["window_start"] = df["window"].apply(lambda w: w[0])

# ---- Plot multiple metrics ----
plt.figure(figsize=(12, 8))

# --------- 1. Action ------------
plt.subplot(4, 1, 1)
plt.plot(df["window_start"], df["threshold_factor"], color="blue")
plt.title("Window vs threshold_factor")
plt.xlabel("Window Start")
plt.ylabel("threshold_factor")
plt.grid(True)

# --------- 2. Reward ------------
plt.subplot(4, 1, 2)
plt.plot(df["window_start"], df["reward"], color="green")
plt.title("Window vs Reward")
plt.xlabel("Window Start")
plt.ylabel("Reward")
plt.grid(True)

# --------- 3. SNR Improvement ------------
plt.subplot(4, 1, 3)
plt.plot(df["window_start"], df["snr_improvement"], color="red")
plt.title("Window vs SNR Improvement")
plt.xlabel("Window Start")
plt.ylabel("SNR Improvement")
plt.grid(True)

# --------- 4. Threshold Factor ------------
plt.subplot(4, 1, 4)
plt.plot(df["window_start"], df["running_mean"], color="green")
plt.title("Window vs running mean")
plt.xlabel("Window Start")
plt.ylabel("running mean")
plt.grid(True)

plt.tight_layout()
plt.show()

