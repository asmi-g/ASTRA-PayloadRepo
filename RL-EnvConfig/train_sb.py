
# train_noise_reduction_fixedobs.py
import os, time, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Import the new fixed-obs env
from astra_rev1.envs import NoiseReductionEnv

# -------------------------------
# Config
# -------------------------------
SEED = 42
TOTAL_STEPS = 70_000
LOG_DIR = "Training/Logs"
MODEL_DIR = "models"
BEST_DIR = os.path.join(MODEL_DIR, "best_model")
N_STACK = 4  # number of frames to stack (stacked features length = 12 * N_STACK)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
np.random.seed(SEED)

# -------------------------------
# Callbacks
# -------------------------------
class LogActionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        try:
            last_obs = self.model._last_obs
            action, _ = self.model.predict(last_obs, deterministic=False)
            self.logger.record("train/sample_action_mean", float(np.mean(action)))
        except Exception as e:
            if self.verbose:
                print(f"[LogActionCallback] Failed to record action: {e}")

# -------------------------------
# Env builders (training/eval)
# -------------------------------
def make_env():
    # The env internally synthesizes windows when reset() is called without signals.
    # Observation space is fixed (12,), so it plays nicely with frame stacking.
    env = NoiseReductionEnv()
    env = Monitor(env)
    return env

# Vectorized + Frame Stacked envs
train_env = DummyVecEnv([make_env])
train_env = VecFrameStack(train_env, n_stack=N_STACK, channels_order="last")

eval_env = DummyVecEnv([make_env])
eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="last")

# -------------------------------
# Model
# -------------------------------
model = SAC(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
    ent_coef="auto_0.5",
    buffer_size=500_000,
    learning_starts=10_000,
    train_freq=64,
    gradient_steps=64,
    batch_size=256,
)

# -------------------------------
# Callbacks
# -------------------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    eval_freq=2000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=MODEL_DIR,
    name_prefix="sac_checkpoint",
)

callback = CallbackList([eval_callback, checkpoint_callback, LogActionCallback(verbose=1)])

# -------------------------------
# Train
# -------------------------------
start = time.time()
model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

# -------------------------------
# Save & Evaluate
# -------------------------------
model_path = os.path.join(MODEL_DIR, f"sac_denoise_fixedobs_framestack_{int(start)}")
model.save(model_path)
print(f"\nModel saved to: {model_path}")

mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final Eval: {mean_r:.2f} ± {std_r:.2f}")

train_env.close()
eval_env.close()
print(f"Total runtime: {time.time() - start:.1f}s")