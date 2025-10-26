import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

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
class ActorCriticLSTMNetwork(nn.Module):
    def __init__(self, num_actions, feature_dim=512, lstm_hidden_size=256):
        super(ActorCriticLSTMNetwork, self).__init__()
        self.feature_extractor = baseCNN(feature_dim=feature_dim)
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm_hidden_size = lstm_hidden_size
        
        self.policy_head = nn.Linear(lstm_hidden_size, num_actions)
        self.value_head = nn.Linear(lstm_hidden_size, 1)

    def forward(self, obs, hidden_state):
        features = self.feature_extractor(obs)
        features_seq = features.unsqueeze(1) 

        lstm_out, new_hidden_state = self.lstm(features_seq, hidden_state)
        lstm_out_flat = lstm_out.squeeze(1)
        
        action_logits = self.policy_head(lstm_out_flat)
        action_probs = F.softmax(action_logits, dim=-1) 
        state_value = self.value_head(lstm_out_flat)    
        
        return action_probs, state_value, new_hidden_state

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(device))

###############################
def train(env, policy_net, num_episodes, learning_rate, gamma, device):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    policy_net.to(device)
    all_rewards = []
    all_losses = []
    print(f"Starting Actor-Critic LSTM training on {device} for {num_episodes} episodes...") 

    for episode in range(num_episodes):
        saved_log_probs = []
        saved_values = []
        rewards = []

        state, info = env.reset()
        hidden_state = policy_net.init_hidden(batch_size=1, device=device)

        done = False
        while not done:
            state_tensor = arrayToTensor(state, device)

            action_probs, state_value, next_hidden_state = policy_net(state_tensor, (hidden_state[0].detach(), hidden_state[1].detach()))
            hidden_state = next_hidden_state 

            dist = Categorical(action_probs)
            action = dist.sample()

            saved_log_probs.append(dist.log_prob(action))
            saved_values.append(state_value.squeeze())

            state, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        saved_values = torch.stack(saved_values)

        advantage = returns - saved_values

        policy_loss = [-log_prob * A for log_prob, A in zip(saved_log_probs, advantage.detach())]
        policy_loss_sum = torch.stack(policy_loss).sum()

        value_loss = F.mse_loss(saved_values, returns)
        loss = policy_loss_sum + 0.5 * value_loss

        optimizer.zero_grad()
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f}")

    return all_rewards, all_losses