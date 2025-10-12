# train.py

# Student Number:
# Student Name:

# Student Number:
# Student Name:

import gym as old_gym
import stable_baselines3
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register


import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import gmean

from reinforceV1 import PolicyNetwork, train
from gymnasium.envs.registration import register

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = './logdir/reinforce_base_run' 

#GPU CHECK
print("--------------------")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU.")
print("--------------------")

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
args = parser.parse_args()

register(id='CrafterReward-v1',entry_point=crafter.Env)

env = old_gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)
env = GymV21CompatibilityV0(env=env)  


env = crafter.Recorder(
  env,
  LOG_DIR,
  save_stats=True,
  save_video=False,
  save_episode=False,
)
#########################################################
#setup done
#########################################################
#hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
NUM_EPISODES = 500
FEATURE_DIM = 512
#########################################################
#training
#########################################################
#REINFORCE with CNN policy network
num_actions = env.action_space.n
policy_network = PolicyNetwork(num_actions=num_actions, feature_dim=FEATURE_DIM)

episode_rewards, episode_losses = train(
    env=env,
    policy_net=policy_network,
    num_episodes=NUM_EPISODES,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    device=DEVICE
)
#########################################################
#end reinforce
##########################################################



print("\n--- Training Complete ---")
env.close()

##########################################################
#Evaluation
##########################################################

print("\n--- Starting Evaluation ---")
stats_path = os.path.join(LOG_DIR, 'stats.jsonl')

try:
    df = pd.read_json(stats_path, lines=True)

    #Cumulative Reward
    avg_cumulative_reward = df['reward'].mean()
    print(f"\nAverage Cumulative Reward: {avg_cumulative_reward:.2f}")

    #Survival Time
    avg_survival_time = df['length'].mean()
    print(f"Average Survival Time: {avg_survival_time:.2f} steps")

    #Achievement Unlock Rate 
    print("\nAchievement Unlock Rates:")
    achievement_cols = sorted([col for col in df.columns if 'achievement_' in col])
    unlock_rates = []
    if not achievement_cols:
        print("  No achievement data found.")
    else:
        for ach in achievement_cols:
            rate = df[ach].mean()
            unlock_rates.append(rate)
            ach_name = ach.replace('achievement_', '').replace('_', ' ').title()
            print(f"  - {ach_name:<25}: {rate:.2%}")

    #mean of rates
    if unlock_rates:
        geo_mean = gmean(np.array(unlock_rates) + 1e-9)
        print(f"\nGeometric Mean of Unlock Rates: {geo_mean:.5f}")

except FileNotFoundError:
    print(f"ERROR: Could not find stats file at '{stats_path}'.")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")

#save and plot
model_path = os.path.join(LOG_DIR, 'reinforce_model.pth')
torch.save(policy_network.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

#reward
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title("Total Reward per Episode (Vanilla REINFORCE)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig(os.path.join(LOG_DIR, 'reward_plot.png'))
plt.show()

#loss
plt.figure(figsize=(10, 5))
plt.plot(episode_losses)
plt.title("Training Loss per Episode (Vanilla REINFORCE)")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(LOG_DIR, 'loss_plot.png'))
plt.show()