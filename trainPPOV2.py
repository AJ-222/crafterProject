import gym as old_gym
from gym.envs.registration import register
from shimmy import GymV21CompatibilityV0
import crafter
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import gmean

from stable_baselines3.common.vec_env import DummyVecEnv
from ppoAgentV2 import train_ppo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = './logdir/ppo_lstm_run'

print("--------------------")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU.")
print("--------------------")

os.makedirs(LOG_DIR, exist_ok=True)
register(id='CrafterPartial-v1', entry_point='crafter:Env')
env = old_gym.make('CrafterPartial-v1')
env = crafter.Recorder(
    env,
    LOG_DIR,
    save_stats=True,
    save_video=False,
    save_episode=False,
)
env = GymV21CompatibilityV0(env=env)

env = DummyVecEnv([lambda: env])
TOTAL_TIMESTEPS = 1000000
train_ppo(env=env, total_timesteps=TOTAL_TIMESTEPS, log_dir=LOG_DIR)

print("\n--- Training Complete ---")
env.close()

print("\n--- Starting Evaluation ---")
stats_path = os.path.join(LOG_DIR, 'stats.jsonl')
try:
    df = pd.read_json(stats_path, lines=True)

    avg_cumulative_reward = df['reward'].mean()
    print(f"\nAverage Cumulative Reward: {avg_cumulative_reward:.2f}")

    avg_survival_time = df['length'].mean()
    print(f"Average Survival Time: {avg_survival_time:.2f} steps")

    print("\nAchievement Unlock Rates:")
    achievement_cols = sorted([col for col in df.columns if 'achievement_' in col])
    unlock_rates = []
    if not achievement_cols:
        print("  No achievement data found.")
    else:
        for ach in achievement_cols:
            rate = df[ach].apply(lambda x: 1 if x > 0 else 0).mean()
            
            unlock_rates.append(rate)
            ach_name = ach.replace('achievement_', '').replace('_', ' ').title()
            print(f"  - {ach_name:<25}: {rate:.2%}")

    if unlock_rates:
        geo_mean = gmean(np.array(unlock_rates) + 1e-9)
        print(f"\nGeometric Mean of Unlock Rates: {geo_mean:.5f}")
except FileNotFoundError:
    print(f"ERROR: Could not find stats file at '{stats_path}'.")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")