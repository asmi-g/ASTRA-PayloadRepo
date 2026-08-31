# train_un.py
# Same as train_sb.py, but trains on the flight-extracted noise model
# (noise_model_fs1.pkl, task 8/9) instead of the synthetic white+pink+burst
# model -- this is the "UN" (updated noise) model.
import time
start_time = time.time()
import os
import numpy as np
import random
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from astra_rev1.envs import NoiseReductionEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

last_update_time = datetime.now()
timestamp_str = last_update_time.strftime("%Y%m%d_%H%M%S")
TIMESTEPS = 100_000

seed = 42
np.random.seed(seed)

NOISE_MODEL_PATH = "../Data/noise_model_fs1.pkl"

train_env = Monitor(NoiseReductionEnv(signal_length=10_000, window_size=1000, mode="train", noise_model_path=NOISE_MODEL_PATH))
eval_env = Monitor(NoiseReductionEnv(signal_length=10_000, window_size=1000, mode="train", noise_model_path=NOISE_MODEL_PATH))
check_env(train_env, warn=True)

log_path = os.path.join('Training', 'Logs')
os.makedirs(log_path, exist_ok=True)
os.makedirs("../models", exist_ok=True)

model = SAC("MlpPolicy", train_env, verbose=1, seed=seed) #tensorboard_log=log_path
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f'../models/best_model_UN_{timestamp_str}_{TIMESTEPS}',
    #log_path='Training/Logs',
    eval_freq=1000,
    n_eval_episodes=3,
    deterministic=True,
    render=False
)
# distinct name_prefix from train_sb.py's 'sac_checkpoint' -- the top-level
# models/ folder already has old sac_checkpoint_*_steps.zip files from a
# prior historical run; reusing that prefix here would silently overwrite them.
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../models/', name_prefix='un_sac_checkpoint')

model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, checkpoint_callback])

model.save(f"../models/UN_W1000_{timestamp_str}_{TIMESTEPS}")
print("Model saved to 'models/UN_W1000_...'")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean reward: {mean_reward} ± {std_reward}")

episode_rewards = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True)
print("Evaluation rewards over episodes: ", episode_rewards)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
