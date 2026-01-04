import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartAttackCNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, out_channels=16, kernel_size=3, padding='same')
        self.pool = nn.MaxPool1d(2)
        self.flattened_size = (in_features // 2) * 16
        self.fc1 = nn.Linear((in_features // 2) * 16, 1)
        self.fc1 = nn.Linear(self.flattened_size, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension: [batch, 1, features]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
