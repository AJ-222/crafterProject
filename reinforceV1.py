import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import crafter
import numpy as np
import matplotlib.pyplot as plt


#############helpers#############
def arrayToTensor(obs, device):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    obs_tensor = obs_tensor.permute(2, 0, 1)
    obs_tensor = obs_tensor.unsqueeze(0)    
    return obs_tensor




###########################################
#feature extractory
class baseCNN(nn.Module):
    def __init__(self, feature_dim=512):
        super(baseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=feature_dim)

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.flatten(x)
        
        features = F.relu(self.fc1(x))
        
        return features
#########################################
#use CNN for actions
class PolicyNetwork(nn.Module):
    def __init__(self, num_actions, feature_dim=512):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = baseCNN(feature_dim=feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)

    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
###############################
#REINFORCE training loop
def train(env, policy_net, num_episodes, learning_rate, gamma, device):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    policy_net.to(device)
    all_rewards = []
    all_losses = []
    print(f"Starting training on {device} for {num_episodes} episodes...")
    for episode in range(num_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = arrayToTensor(state, device)
            
            action_probs = policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            saved_log_probs.append(dist.log_prob(action))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
            
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        policy_loss = [-log_prob * R for log_prob, R in zip(saved_log_probs, returns)]

        optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f}")
            
    return all_rewards