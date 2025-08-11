# train_noise_reduction.py
# train_noise_reduction.py
import os
import time
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.env_checker import check_env

from astra_rev1.envs import NoiseReductionEnv  # your env file

# ---------- Config ----------
SEED = 42
TOTAL_TIMESTEPS = 100_000
LOG_DIR = os.path.join("Training", "Logs")
MODEL_DIR = "models"
BEST_DIR = os.path.join(MODEL_DIR, "best_model")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
np.random.seed(SEED)


# ---------- Small wrappers to force training mode on reset ----------
class TrainResetWrapper(NoiseReductionEnv):
    """Ensure every SB3 reset() runs the env in training mode (sliding windows)."""
    def reset(self):
        # No args because SB3 calls reset() without kwargs. We forward training=True inside.
        return super().reset(training=True)

class EvalResetWrapper(NoiseReductionEnv):
    """Separate eval env (also sliding). You can switch to training=False to freeze window."""
    def reset(self):
        return super().reset(training=True)


# ---------- Build envs ----------
train_env = Monitor(TrainResetWrapper(signal_length=4000, window_size=10, training=True, seed=SEED))
check_env(train_env, warn=True)

eval_env = Monitor(EvalResetWrapper(signal_length=4000, window_size=10, training=True, seed=SEED + 1))

# ---------- Custom callback: log a sample action ----------
class LogActionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Keep training going
        return True

    def _on_rollout_end(self) -> None:
        try:
            # Use the most recent observation seen by the model to get a sample action
            last_obs = self.model._last_obs
            action, _ = self.model.predict(last_obs, deterministic=False)
            self.logger.record("train/sample_action", float(action[0]))
        except Exception as e:
            if self.verbose:
                print(f"[LogActionCallback] Failed to record action: {e}")


# ---------- SAC model ----------
model = SAC(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
    # Exploration: a bit stronger than default to avoid “extremes only”
    ent_coef="auto_0.5",
    # Some stable defaults for this lightweight, stationary task
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,        # update every step
    gradient_steps=2,    # small but steady updates
)

# ---------- Callbacks ----------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
    eval_freq=2_000,       # evaluate every N steps
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=MODEL_DIR,
    name_prefix="sac_checkpoint",
)

log_action_callback = LogActionCallback(verbose=1)

callback = CallbackList([eval_callback, checkpoint_callback, log_action_callback])

# ---------- Train ----------
start_time = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# ---------- Save ----------
model_name = f"sac_noise_reduction_{int(start_time)}"
save_path = os.path.join(MODEL_DIR, model_name)
model.save(save_path)
print(f"\nModel saved to: {save_path}")

# ---------- Final evaluation ----------
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final Evaluation: Mean Reward = {mean_r:.2f} ± {std_r:.2f}")

# If you also want per-episode rewards:
ep_rewards = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True)
print("Per-episode rewards:", ep_rewards)

train_env.close()
eval_env.close()

elapsed = time.time() - start_time
print(f"\nTotal runtime: {elapsed:.2f} s")

# 
# def generate_synthetic_signal(signal_type=None, noise_level=None, length=100):
#     t = np.linspace(0, 1, length)
#     signal_type = signal_type or random.choice(["sine", "square", "sawtooth", "random"])
#     noise_level = noise_level if noise_level is not None else np.random.uniform(0.2, 0.5)

#     if signal_type == "sine":
#         clean = np.sin(2 * np.pi * 5 * t)
#     elif signal_type == "square":
#         clean = np.sign(np.sin(2 * np.pi * 5 * t))
#     elif signal_type == "sawtooth":
#         clean = 2 * (t * 5 % 1) - 1
#     elif signal_type == "random":
#         clean = np.random.uniform(-1, 1, size=length)
#     else:
#         raise ValueError("Unknown signal type!")

#     noisy = clean + np.random.normal(0, noise_level, size=length)
#     return clean, noisy


# # Initialize model
# model = SAC(
#     "MlpPolicy",
#     env,
#     ent_coef="auto",  # 🔄 CHANGED: Encourage exploration via entropy bonus
#     action_noise=action_noise,  # 🔄 CHANGED: Added action noise
#     learning_rate=1e-4,
#     buffer_size=100000,
#     batch_size=128,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=4,  # 🔄 CHANGED: More frequent updates per step
#     verbose=1,
#     tensorboard_log="logs/tensorboard"
# )
# from stable_baselines3.common.logger import configure

# model._logger = configure(folder="logs/tensorboard", format_strings=["stdout", "tensorboard"])
# model._current_progress_remaining = 1.0  # Full training progress at start


# WINDOW_SIZE = 10
# SIGNAL_LENGTH = 100
# TOTAL_EPISODES = 5000

# # Training Loop
# for episode in range(TOTAL_EPISODES):
#     clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)

#     for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
#         clean_window = clean_full[i:i + WINDOW_SIZE]
#         noisy_window = noisy_full[i:i + WINDOW_SIZE]

#         if i == 0:
#             obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
#         else:
#             env.set_signal_window(clean_window, noisy_window)

#         action, _ = model.predict(obs, deterministic=False)
#         next_obs, reward, done, _, info = env.step(action)

#         # Manually store transition
#         model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
#         model.train(batch_size=model.batch_size, gradient_steps=1)

#         obs = next_obs

#     if episode % 100 == 0:
#         print(f"Step {episode} | Action: {action[0]:.4f} | Threshold: {env.threshold_factor:.3f} | Reward: {reward:.3f}") 


# # Callback for custom logging
# # callback = NoiseReductionLogger()
# # TOTAL_TIMESTEPS = 1000
# # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# model.save("models/sac_noise_reduction")
# print("Model saved to 'models/sac_noise_reduction'")

# # # Training parameters
# # WINDOW_SIZE = 10
# # SIGNAL_LENGTH = 100
# # TOTAL_EPISODES = 1000

# # # 🔄 CHANGED: Use sliding window over new signal every episode
# # for episode in range(TOTAL_EPISODES):
# #     clean_full, noisy_full = generate_synthetic_signal(length=SIGNAL_LENGTH)
# #     total_reward = 0

# #     for i in range(SIGNAL_LENGTH - WINDOW_SIZE):
# #         clean_window = clean_full[i:i + WINDOW_SIZE]
# #         noisy_window = noisy_full[i:i + WINDOW_SIZE]

# #         if i == 0:
# #             obs, _ = env.reset(clean_signal=clean_window, noisy_signal=noisy_window)
# #         else:
# #             env.set_signal_window(clean_window, noisy_window)

# #         action, _ = model.predict(obs)
# #         obs, reward, done, _, info = env.step(action)
# #         total_reward += reward

# #     # 🔄 CHANGED: Better logging for debugging
# #     if episode % 100 == 0:
# #         print(f"Episode {episode} | Total Reward: {total_reward:.2f} "
# #               f"| Threshold: {info['threshold_factor']:.3f} | Action: {action}")

# # Save the model
