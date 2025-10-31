import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

CLIP_GRAD_NORM = 0.5
ENTROPY_COEF = 0.01 
# -----------------------------------

#############helpers#############
def arrayToTensor(obs, device):
    obs = np.array(obs, dtype=np.uint8)
    if obs.shape != (64, 64, 3):
        raise ValueError(f"Received invalid observation shape: {obs.shape}. Expected (64, 64, 3).")
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    obs_tensor = obs_tensor.permute(2, 0, 1)
    obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor

###########################################
#feature extraction
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
#A2C
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_actions, feature_dim=512):
        super(ActorCriticNetwork, self).__init__()
        self.feature_extractor = baseCNN(feature_dim=feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)
    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(features)
        return action_probs, state_value

###############################
def train(env, policy_net, num_episodes, learning_rate, gamma, device):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    policy_net.to(device)
    all_rewards = []
    all_losses = []
    print(f"Starting Actor-Critic (Tuned) training on {device} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        saved_log_probs = []
        saved_values = []
        rewards = []
        saved_entropies = []

        state, info = env.reset()
        done = False
        while not done:
            state_tensor = arrayToTensor(state, device)
            action_probs, state_value = policy_net(state_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            saved_log_probs.append(dist.log_prob(action))
            saved_values.append(state_value)
            saved_entropies.append(dist.entropy()) 
            
            state, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
            
        returns = torch.tensor(returns, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = (returns - returns.mean())

        saved_values = torch.cat(saved_values).squeeze()
        min_len = min(len(returns), len(saved_values), len(saved_log_probs), len(saved_entropies))
        returns = returns[:min_len]
        saved_values = saved_values[:min_len]
        saved_log_probs = saved_log_probs[:min_len]
        saved_entropies = torch.stack(saved_entropies[:min_len])

        advantage = returns - saved_values
        
        policy_loss = [-log_prob * A for log_prob, A in zip(saved_log_probs, advantage.detach())]
        policy_loss_sum = torch.stack(policy_loss).sum() 

        value_loss = F.mse_loss(saved_values, returns)
        entropy_loss = -saved_entropies.mean()
        loss = policy_loss_sum + 0.5 * value_loss + ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        all_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), CLIP_GRAD_NORM)
        
        optimizer.step()
        
        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f}")
            
    return all_rewards, all_losses