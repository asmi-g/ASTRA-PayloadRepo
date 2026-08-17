# train_sb.py
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

train_env = Monitor(NoiseReductionEnv(signal_length=10000, window_size=1000, mode="train"))
eval_env = Monitor(NoiseReductionEnv(signal_length=10000, window_size=1000, mode="train"))
check_env(train_env, warn=True)

log_path = os.path.join('Training', 'Logs')
os.makedirs(log_path, exist_ok=True)
os.makedirs("../models", exist_ok=True)

model = SAC("MlpPolicy", train_env, verbose=1, seed=seed) #tensorboard_log=log_path
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f'models/best_model_{timestamp_str}_{TIMESTEPS}',
    #log_path='Training/Logs',
    eval_freq=20000,
    n_eval_episodes=3,
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='models/', name_prefix='sac_checkpoint')

model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, checkpoint_callback])

model.save(f"../models/sac_noise_reduction_{timestamp_str}_")
print("Model saved to 'models/sac_noise_reduction'")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean reward: {mean_reward} ± {std_reward}")

episode_rewards = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True)
print("Evaluation rewards over episodes: ", episode_rewards)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal runtime: {elapsed_time:.2f} seconds")