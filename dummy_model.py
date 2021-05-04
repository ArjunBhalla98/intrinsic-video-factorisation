import torch
import torch.nn as nn


class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.conv1(x)
        return self.tanh(h)
