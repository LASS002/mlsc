"""
model.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-01-29
Explicação: SimpleCNN model architecture definition.
How to use: Used internally by train.py
Licença: AGPL3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input is 64x64 grayscale images (1 channel)

        # First block: Conv2d -> ReLU -> MaxPool2d
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Output: 64x64 -> MaxPool(2) -> 32x32

        # Second block: Conv2d -> ReLU -> MaxPool2d
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        # Output: 32x32 -> MaxPool(2) -> 16x16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten -> Linear -> Output
        # Feature map size: 32 channels * 16 * 16 = 8192
        self.fc = nn.Linear(
            32 * 16 * 16, 2
        )  # 2 outputs for CrossEntropy (Circle, Square)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # Linear -> Output
        x = self.fc(x)
        return x
