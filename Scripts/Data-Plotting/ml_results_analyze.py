#ml_results_analyze.py
"""
Visualize the per-window RL inference metrics exported by inference.py
(the *_results_*.csv files: window, action, reward, snr_improvement,
threshold_factor, running_mean).

Usage:
    python ml_results_analyze.py                          # newest *_results_*.csv in DATA_DIR
    python ml_results_analyze.py path/to/some_results.csv  # a specific file
    python ml_results_analyze.py some_results.csv --save plots/run.png
"""

import argparse
import ast
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(BASE_DIR, "Data")


def find_latest_results_csv(data_dir):
    candidates = glob.glob(os.path.join(data_dir, "*_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_results_*.csv files found in {data_dir}")
    return max(candidates, key=os.path.getmtime)


def parse_scalar(x):
    """action is logged as a numpy repr like '[[-0.658]]' -- pull the first float out."""
    if isinstance(x, (int, float)):
        return float(x)
    try:
        v = ast.literal_eval(x)
        v = np.asarray(v, dtype=float).ravel()
        return float(v[0]) if v.size else np.nan
    except (ValueError, SyntaxError, TypeError):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return float(m.group(0)) if m else np.nan


def parse_window_start(w):
    try:
        return ast.literal_eval(w)[0]
    except (ValueError, SyntaxError, TypeError):
        m = re.search(r"\d+", str(w))
        return int(m.group(0)) if m else np.nan


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    # drop any trailing "(DONE)" / malformed marker rows
    df = df[df["window"].astype(str).str.contains(r"\d")].copy()

    df["window_start"] = df["window"].apply(parse_window_start)
    df["action"] = df["action"].apply(parse_scalar)
    for col in ("reward", "snr_improvement", "threshold_factor", "running_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("window_start").reset_index(drop=True)
    return df


def plot_results(df, title, save_path=None):
    x = df["window_start"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title)

    axes[0].plot(x, df["action"], color="tab:blue")
    axes[0].set_ylabel("action")
    axes[0].set_title("Action (raw policy output)")

    axes[1].plot(x, df["reward"], color="tab:green")
    axes[1].set_ylabel("reward")
    axes[1].set_title(f"Reward  (mean = {df['reward'].mean():.4f})")

    axes[2].plot(x, df["snr_improvement"], color="tab:red", label="per-window")
    if "running_mean" in df.columns:
        axes[2].plot(x, df["running_mean"], color="black", lw=1.2, label="running mean")
        axes[2].legend(loc="best")
    axes[2].set_ylabel("SNR improvement (dB)")
    axes[2].set_title(f"SNR improvement  (final running mean = {df['running_mean'].iloc[-1]:.4f} dB)"
                      if "running_mean" in df.columns else "SNR improvement")

    axes[3].plot(x, df["threshold_factor"], color="tab:blue")
    axes[3].set_ylabel("threshold_factor")
    axes[3].set_xlabel("window start (sample index)")
    axes[3].set_title(f"threshold_factor  (mean = {df['threshold_factor'].mean():.4f})")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] saved {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot RL inference per-window metrics.")
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to a *_results_*.csv file. Defaults to the newest one in DATA_DIR.")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the PNG (e.g. plots/run.png). If omitted, shows interactively.")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (useful for batch saving).")
    args = parser.parse_args()

    csv_path = args.csv_path or find_latest_results_csv(DATA_DIR)
    print(f"[INFO] Loading results from {csv_path}")

    df = load_results(csv_path)
    print(f"[INFO] {len(df)} windows, window_start range "
          f"[{df['window_start'].min()}, {df['window_start'].max()}]")

    plot_results(df, title=os.path.basename(csv_path), save_path=args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
