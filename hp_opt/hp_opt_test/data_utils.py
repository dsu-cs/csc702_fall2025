"""
data_utils.py
-------------
All the training configuration, data pipeline, and small utilities.
Contains:
  - TrainConfig dataclass (hyperparameters and paths)
  - set_seed, get_device, accuracy helpers
  - get_dataloaders for Fashion-MNIST (with normalization)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class TrainConfig:
    """Holds hyperparameters and runtime options.

    Only these values should be considered *tunable hyperparameters* for search:
      - model_type, batch_size, epochs, lr, weight_decay, dropout
    The rest control reproducibility or I/O.
    """
    model_type: str = "cnn"      # "cnn" or "mlp"
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


# ---------------------------
# Reproducibility & Device
# ---------------------------
def set_seed(seed: int = 42) -> None:
    """Set Python/PyTorch random seeds.

    Note: We keep cudnn.benchmark on for speed (non-deterministic fast path).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return the best available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Metrics
# ---------------------------
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute average accuracy for a batch.

    Args:
        logits: (B, C) raw, unnormalized scores.
        targets: (B,) ground-truth class indices.
    Returns:
        Scalar float accuracy in [0, 1].
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------
# Data pipeline
# ---------------------------
def get_dataloaders(cfg: TrainConfig):
    """Create train/val/test DataLoaders for Fashion-MNIST.

    Applies standard normalization statistics for Fashion-MNIST.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Pre-computed dataset stats
    ])

    # Full training set (60k samples)
    full_train = datasets.FashionMNIST(cfg.data_dir, train=True, download=True, transform=transform)

    # Deterministic train/val split
    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Test set (10k samples)
    test_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=transform)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader