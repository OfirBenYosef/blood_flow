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
        self.g1 = GaborConv2d(3, 32, kernel_size=(5, 5), stride=1)
        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=1)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=1)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=1)
        x = nn.Dropout2d()(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = torch.sigmoid(self.fc3(x))
        return x

