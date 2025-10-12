import torch
import torch.nn as nn
import torch.nn.functional as F

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