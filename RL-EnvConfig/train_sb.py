# train_sb.py
# ---------------------------------------------------------------------------
# Trains one SAC model against one of the two corrected noise schemes.
#
#   SCHEME = "OAN"  (original accurate noise)
#       NoiseReductionEnv.generate_accurate_signals(): white + true 1/f pink
#       noise normalised to the noise-power budget, rare FRED-shaped
#       solar-radio-burst episodes (Poisson mean 0.3 per signal), and short
#       impulsive bursts. No external noise model.
#
#   SCHEME = "PFN"  (parametric flight noise)
#       NoiseReductionEnv.generate_accurate_flight_signals(): parametric
#       rebuild of the flight_signal_1 residual from Data/noise_model_fs1.pkl
#       -- AR(50) colour + heavy-tailed (Bernoulli-Gaussian) innovations
#       calibrated to the measured AR-innovation kurtosis + the measured
#       rms_profile drift envelope, rescaled to the real noise/clean
#       severity ratio.
#
#   SCHEME = "OPBN"  (original pink-burst noise)
#       NoiseReductionEnv.generate_signals_pink_bursts(): always-on WHITE floor
#       at the noise-power budget, with true-1/f pink energy appearing ONLY
#       inside a Poisson-count (mean 0.5) of short additive burst episodes,
#       plus short impulsive hits. No continuous pink, no flare envelope.
#
# Everything else about the run (timesteps, window, signal-length policy, SAC
# hyperparameters, seed, eval protocol) is identical between the schemes, so a
# comparison isolates the training noise scheme as the only variable. Edit
# SCHEME to switch; there are no environment variables.
# ---------------------------------------------------------------------------

import os
import time
import random
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from astra_rev1.envs import NoiseReductionEnv

# ========================= run configuration =========================
SCHEME = "OPBN"               # "OAN", "PFN", or "OPBN"

TIMESTEPS        = 1_000_000   # total SAC training steps
TRAIN_SIGNAL_LEN = 500_000     # nominal; env redraws U[len/5, len*2) per episode
EVAL_SIGNAL_LEN  = 150_000     # nominal for eval episodes (also redrawn per episode)
WINDOW_SIZE      = 1000

EVAL_FREQ            = 50_000   # run an evaluation this often (steps) -> 20 evals over 1M
N_EVAL_EPISODES      = 10
MAX_NO_IMPROVE_EVALS = 6        # early-stop patience, in evals, on no eval-reward gain

# Off-policy update-to-data ratio. SAC's default is 1 gradient update per env
# step (train_freq=1, gradient_steps=1) -> ~1e6 gradient updates for a 1M-step
# run, ~13 h on this CPU-only box. TRAIN_FREQ=(4,"step") with GRADIENT_STEPS=1
# is 1 update per 4 env steps -> ~250k updates, ~3-4x faster (~7 h), while the
# environment-experience budget stays a full 1M steps. Both must be set: with
# GRADIENT_STEPS at SB3's -1 default, train_freq=(4,"step") would do 4 updates
# per trigger and the ratio would still be 1:1. Lower update-to-data ratio is
# a standard off-policy knob; the cost is mild sample-inefficiency (not
# instability), expected to be small here given the near-bandit reward
# landscape.
TRAIN_FREQ     = (4, "step")
GRADIENT_STEPS = 1

SEED = 42
# ====================================================================

assert SCHEME in ("OAN", "PFN", "OPBN"), f"SCHEME must be 'OAN', 'PFN', or 'OPBN', got {SCHEME!r}"

np.random.seed(SEED)
random.seed(SEED)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(script_dir, "../models"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join("Training", "Logs"), exist_ok=True)

noise_model_path = (
    os.path.normpath(os.path.join(script_dir, "../Data/noise_model_fs1.pkl"))
    if SCHEME == "PFN" else None
)
SIGNAL_GEN = "pink_bursts" if SCHEME == "OPBN" else "accurate"

RUN = f"{SCHEME}_W{WINDOW_SIZE}_{timestamp_str}_{TIMESTEPS}"
print(f"[INFO] SCHEME={SCHEME}  RUN={RUN}  noise_model_path={noise_model_path}")


def make_env(sig_len):
    return Monitor(NoiseReductionEnv(
        signal_length=sig_len, window_size=WINDOW_SIZE, mode="train",
        noise_model_path=noise_model_path, signal_gen=SIGNAL_GEN))


train_env = make_env(TRAIN_SIGNAL_LEN)
eval_env = make_env(EVAL_SIGNAL_LEN)
check_env(train_env, warn=True)

# --- best model (on eval reward) + early stopping ---
stop_cb = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=MAX_NO_IMPROVE_EVALS, min_evals=8, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(MODELS_DIR, f"best_{RUN}"),
    log_path=os.path.join(MODELS_DIR, f"best_{RUN}"),
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    callback_after_eval=stop_cb,
)

# --- rolling checkpoints (cheap insurance if the run is killed) ---
checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=MODELS_DIR, name_prefix=f"ckpt_{RUN}")


class SaveRecentAfter(BaseCallback):
    """After `start_after` timesteps, save the current model every `every`
    steps to models/recent_<RUN>_<step>.zip. Complements EvalCallback's
    best-model save: best_ = highest eval reward so far, recent_ = latest
    policy regardless of eval, so a manual stop still leaves a fresh model."""
    def __init__(self, start_after, every, save_dir, run, verbose=1):
        super().__init__(verbose)
        self.start_after, self.every = start_after, every
        self.save_dir, self.run = save_dir, run
        self._last = 0

    def _on_step(self):
        if (self.num_timesteps >= self.start_after
                and self.num_timesteps - self._last >= self.every):
            p = os.path.join(self.save_dir, f"recent_{self.run}_{self.num_timesteps}")
            self.model.save(p)
            self._last = self.num_timesteps
            if self.verbose:
                print(f"[recent-save] {p}.zip @ {self.num_timesteps} steps")
        return True


recent_callback = SaveRecentAfter(
    start_after=500_000, every=100_000, save_dir=MODELS_DIR, run=RUN)

start_time = time.time()
model = SAC("MlpPolicy", train_env, verbose=1, seed=SEED,
            train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS)

model.learn(
    total_timesteps=TIMESTEPS,
    callback=[eval_callback, checkpoint_callback, recent_callback],
)

model.save(os.path.join(MODELS_DIR, RUN))
print(f"[INFO] final model saved to {os.path.join(MODELS_DIR, RUN)}.zip")
print(f"[INFO] best model in {os.path.join(MODELS_DIR, f'best_{RUN}')}/best_model.zip")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
