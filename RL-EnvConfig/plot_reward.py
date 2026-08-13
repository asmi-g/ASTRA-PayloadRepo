import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot the reward column from a results CSV.")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent / "Data" / "081325_results.csv"),
        help="Path to the results CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "Data" / "081325_reward_plot.png"),
        help="Path to save the plot image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    df = pd.read_csv(csv_path)
    if "reward" not in df.columns:
        raise ValueError(f"'reward' column not found in {csv_path}")

    rewards = df["reward"].astype(float)
    x = range(len(df))

    plt.figure(figsize=(10, 5))
    plt.plot(x, rewards, marker="o", linewidth=2, color="tab:blue", label="Reward")
    plt.title("Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
