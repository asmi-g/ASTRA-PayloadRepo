#!/usr/bin/env python3
"""
plot_eval_curve.py   (my-list items 5, 6, 8, 9 -- model selection support)
========================================================================
Reads the EvalCallback log written during training
(models/best_<RUN>/evaluations.npz) and:

  * prints the eval-reward-vs-timestep curve (mean +/- std over the eval
    episodes at each checkpoint),
  * reports the best eval timestep, the final eval, and the gap between them,
  * flags whether the curve had plateaued by the end (so "keep the final
    model" is safe) or was still climbing / had collapsed,
  * saves a PNG.

`evaluations.npz` (created by stable_baselines3.common.callbacks.EvalCallback)
contains:
    timesteps   (n_evals,)              training step of each evaluation
    results     (n_evals, n_episodes)   episode returns
    ep_lengths  (n_evals, n_episodes)

Usage:
    python plot_eval_curve.py [path/to/best_<RUN>    OR    .../evaluations.npz]
    # no arg -> newest models/best_* directory
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models"))


def _resolve(arg):
    if arg is None:
        cands = sorted(glob.glob(os.path.join(MODELS_DIR, "best_*")), key=os.path.getmtime)
        if not cands:
            raise SystemExit(f"no best_* dirs in {MODELS_DIR}; pass a path explicitly")
        arg = cands[-1]
    if os.path.isdir(arg):
        arg = os.path.join(arg, "evaluations.npz")
    if not os.path.exists(arg):
        raise SystemExit(f"not found: {arg}")
    return arg


def main(arg=None):
    npz_path = _resolve(arg)
    run = os.path.basename(os.path.dirname(npz_path))
    d = np.load(npz_path)
    ts = d["timesteps"]
    res = d["results"]                 # (n_evals, n_episodes)
    mean = res.mean(axis=1)
    std = res.std(axis=1)

    print(f"[INFO] {npz_path}")
    print(f"[INFO] run = {run}   {len(ts)} evaluations, "
          f"{res.shape[1]} episodes each, up to {ts[-1]:,} steps\n")
    print(f"{'step':>12}  {'mean_reward':>12}  {'std':>10}")
    for t, m, s in zip(ts, mean, std):
        print(f"{t:>12,}  {m:>12.4f}  {s:>10.4f}")

    best_i = int(np.argmax(mean))
    final_i = len(mean) - 1
    print(f"\n[best ] step {ts[best_i]:,}  mean {mean[best_i]:.4f} +/- {std[best_i]:.4f}")
    print(f"[final] step {ts[final_i]:,}  mean {mean[final_i]:.4f} +/- {std[final_i]:.4f}")
    gap = mean[best_i] - mean[final_i]
    print(f"[gap  ] best - final = {gap:.4f}  "
          f"({'final ~= best, plateau reached' if gap <= std[final_i] else 'FINAL IS BELOW BEST -- inspect for late collapse'})")

    # crude plateau check: slope of the last third vs the overall span
    if len(mean) >= 6:
        tail = slice(2 * len(mean) // 3, None)
        tail_slope = np.polyfit(ts[tail], mean[tail], 1)[0]
        full_span = mean.max() - mean.min() + 1e-12
        rel = tail_slope * (ts[-1] - ts[tail.start or 0]) / full_span
        print(f"[trend] last-third trend covers {rel:+.1%} of the full reward span "
              f"({'flat -> plateaued' if abs(rel) < 0.15 else 'still moving'})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, mean, "o-", color="tab:blue", label="eval mean reward")
    ax.fill_between(ts, mean - std, mean + std, color="tab:blue", alpha=0.2, label="+/- 1 std")
    ax.axvline(ts[best_i], color="tab:green", ls="--", label=f"best @ {ts[best_i]:,}")
    ax.set_xlabel("training timestep")
    ax.set_ylabel("evaluation episode reward")
    ax.set_title(f"Eval learning curve — {run}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = os.path.join(MODELS_DIR, f"eval_curve_{run}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\n[INFO] saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
