import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.main = nn.Sequential(*layers)

    def forward(self, t):
        return self.main(t)
