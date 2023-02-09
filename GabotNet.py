# GaborNet

## Getting started

import torch
import torch.nn as nn
from torch.nn import functional as F
from GaborNet import GaborConv2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaborNN(nn.Module):
    def __init__(self):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=1, out_channels=96, kernel_size=(11, 11))
        self.c1 = nn.Conv2d(96, 384, (3,3))
        self.fc1 = nn.Linear(384*3*3, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(self.g0(x))
        x = nn.MaxPool2d()(x)
        x = F.leaky_relu(self.c1(x))
        x = nn.MaxPool2d()(x)
        x = x.view(-1, 384*3*3)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = GaborNN().to(device)
