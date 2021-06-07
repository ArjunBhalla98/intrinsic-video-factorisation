# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision
# import numpy as np
# import math

from imports import *

class Disc(nn.Module):
    
    def __init__(self, num_group=16, input_channel=3):
        super(Disc, self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(input_channel, 64, 5, stride=2, padding=2),
                nn.LeakyReLU()) # 512 -> 256

        self.conv1 = nn.Sequential(
                nn.Conv2d(64, 128, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=128),
                nn.LeakyReLU()) # 256 -> 128
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=256),
                nn.LeakyReLU()) # 128 -> 64

        self.conv3 = nn.Sequential(
                nn.Conv2d(256, 512, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=512),
                nn.LeakyReLU()) # 64 -> 32

        self.fc = nn.Linear(512 * 32 * 48, 1)

    def forward(self, input):
        
        x = self.conv3(self.conv2(self.conv1(self.conv0(input))))
        
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

 
class PatchDisc(nn.Module):
    
    def __init__(self, num_group=16, input_channel=3):
        super(PatchDisc, self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(input_channel, 64, 5, stride=2, padding=2),
                nn.LeakyReLU()) # 512 -> 256

        self.conv1 = nn.Sequential(
                nn.Conv2d(64, 128, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=128),
                nn.LeakyReLU()) # 256 -> 128
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=256),
                nn.LeakyReLU()) # 128 -> 64

        self.conv3 = nn.Sequential(
                nn.Conv2d(256, 256, 5, stride=2, padding=2),
                nn.GroupNorm(num_groups=num_group, num_channels=256),
                nn.LeakyReLU()) # 64 -> 32

        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 1, 5, stride=2, padding=2),
                nn.Sigmoid()) # 64 -> 32

    def forward(self, input):
        
        x = self.conv3(self.conv2(self.conv1(self.conv0(input))))
        x = self.conv4(x)

        return x

                       