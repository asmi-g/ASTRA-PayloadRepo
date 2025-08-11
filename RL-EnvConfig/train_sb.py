# train_noise_reduction.py
# train_noise_reduction.py
# train_noise_reduction_framestack.py
import time, os, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from astra_rev1.envs import NoiseReductionEnv

# -------------------------------
# Config
# -------------------------------
SEED = 42
TOTAL_STEPS = 100_000
LOG_DIR = "Training/Logs"
MODEL_DIR = "models"
BEST_DIR = os.path.join(MODEL_DIR, "best_model")
N_STACK = 4  # <- number of frames to stack (try 4–8)

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
            # last obs from vec env
            last_obs = self.model._last_obs
            action, _ = self.model.predict(last_obs, deterministic=False)
            # action may be shape (n_envs, act_dim)
            self.logger.record("train/sample_action", float(np.mean(action)))
        except Exception as e:
            if self.verbose:
                print(f"[LogActionCallback] Failed to record action: {e}")

# -------------------------------
# Env builders (training/eval)
# -------------------------------
def make_env(training: bool):
    # Monitor must wrap the raw env *inside* DummyVecEnv factories.
    return Monitor(NoiseReductionEnv(training=training))

# Vectorized + Frame Stacked envs
train_env = DummyVecEnv([lambda: make_env(True)])
train_env = VecFrameStack(train_env, n_stack=N_STACK, channels_order="last")

eval_env = DummyVecEnv([lambda: make_env(False)])
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
    ent_coef="auto_0.5",          # a bit more exploration early on
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
    eval_freq=2000,               # a bit less frequent once rollouts are longer
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=MODEL_DIR,
    name_prefix="sac_checkpoint"
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
model_path = os.path.join(MODEL_DIR, f"sac_denoise_framestack_{int(start)}")
model.save(model_path)
print(f"\nModel saved to: {model_path}")

mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Final Eval: {mean_r:.2f} ± {std_r:.2f}")

train_env.close()
eval_env.close()
print(f"Total runtime: {time.time() - start:.1f}s")

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
