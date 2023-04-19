import torch.nn as nn
import torch.nn.functional as F


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)  # No activation
        return z
