#!/usr/bin/env python3
import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------------------
# Repro + Device
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow fast path
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Models
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=256, dropout=0.2, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, dropout=0.2, num_classes=10, fc_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
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
            nn.Linear(64 * 7 * 7, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Training / Evaluation
# ---------------------------
@dataclass
class TrainConfig:
    model_type: str = "cnn"  # "cnn" or "mlp"
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 42
    val_split: float = 0.1
    num_workers: int = 4
    data_dir: str = "./data"
    save_path: str = "./fashion_mnist_baseline.pt"
    # HPO knobs
    optimizer: str = "adam"    # "adam","adamw","sgd"
    momentum: float = 0.9      # used if optimizer == "sgd"
    hidden_dim: int = 256      # MLP only
    cnn_fc_dim: int = 128      # CNN only


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def get_dataloaders(cfg: TrainConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Precomputed for Fashion-MNIST
    ])

    full_train = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=transform)

    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(cfg.seed))

    test_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def bui
