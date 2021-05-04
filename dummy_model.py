import torch
import torch.nn as nn


class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 3, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        return self.sigmoid(h)
