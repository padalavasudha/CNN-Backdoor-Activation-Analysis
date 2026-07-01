"""
CNN architecture used for the sunflower / not-sunflower classification task.

This is the *target* model: the one that gets trained on (potentially
poisoned) data, and whose internal activations we later probe for
backdoor signatures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Small 3-conv-layer CNN for 64x64 RGB binary classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


# Layer names exposed for activation hooking, in forward-pass order.
HOOKABLE_LAYERS = ["conv1", "conv2", "conv3", "fc1"]
