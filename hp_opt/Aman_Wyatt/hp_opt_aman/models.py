"""
models.py
---------
Well-commented PyTorch models for Fashion-MNIST.
Contains:
  - MLP: simple fully-connected baseline
  - SmallCNN: compact convolutional network

Both output raw logits for 10 classes (CrossEntropyLoss expects logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (fully-connected network).

    Expected input shape: (B, 1, 28, 28)
    This module flattens the image into 784 features and applies two hidden
    layers with ReLU + Dropout. The final layer outputs 10 logits.
    """

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, dropout: float = 0.2, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                    # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),  # logits for 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """A compact CNN suitable for 28x28 grayscale images (e.g., Fashion-MNIST).

    Architecture:
      - Conv(1->32) + ReLU
      - Conv(32->32) + ReLU
      - MaxPool(2)            # 28x28 -> 14x14
      - Conv(32->64) + ReLU
      - Conv(64->64) + ReLU
      - MaxPool(2)            # 14x14 -> 7x7
      - Dropout
      - Flatten
      - Linear(64*7*7 -> 128) + ReLU + Dropout
      - Linear(128 -> 10)
    """

    def __init__(self, dropout: float = 0.2, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # keep 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x