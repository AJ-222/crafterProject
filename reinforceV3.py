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