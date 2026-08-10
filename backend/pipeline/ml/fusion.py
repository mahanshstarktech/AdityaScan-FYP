import torch
import torch.nn as nn

class NaiveBaselineModel(nn.Module):
    def __init__(self, in_features=10, hidden_dim=64, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
