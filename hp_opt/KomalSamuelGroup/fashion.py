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
    def __init__(self, dropout=0.2, num_classes=10):
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
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
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
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    test_ds = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(cfg: TrainConfig):
    if cfg.model_type == "mlp":
        return MLP(dropout=cfg.dropout)
    elif cfg.model_type == "cnn":
        return SmallCNN(dropout=cfg.dropout)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc += accuracy(logits.detach(), labels.detach()) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(logits, labels) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def train_eval(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg.__dict__}, cfg.save_path)
        print(f"Saved best model to: {cfg.save_path} (val_acc={best_val_acc:.4f})")

    # Final test evaluation with best state
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST baseline (PyTorch)")
    p.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"], help="Model type")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="./fashion_mnist_baseline.pt")
    return p.parse_args()

# Bayesian optimization using scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def objective(params):
    lr, weight_decay, dropout, batch_size, model_type = params

    cfg = TrainConfig(
        model_type=model_type,
        batch_size=int(batch_size),
        epochs=5,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        seed=42,
        val_split=0.1,
        num_workers=4,
        data_dir="./data",
        save_path="./best_model.pt"
    )

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, va_acc = evaluate(model,val_loader, criterion, device)
        if va_acc > best_val_acc:
            best_val_acc = va_acc


    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")

    return -best_val_acc

search_space = [
    Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
    Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
    Real(0.1, 0.5, name="dropout"),
    Categorical([16, 32, 64, 128, 256, 512], name="batch_size"),
    Categorical(["cnn", "mlp"], name="model_type")
]

def main():
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        n_random_starts=5,
        random_state=42,
    )

    print("Best hyperparameters:")
    print("  lr:", result.x[0])
    print("  weight_decay:", result.x[1])
    print("  dropout:", result.x[2])
    print("  batch_size:", result.x[3])
    print("  model_type:", result.x[4])
    print("Best validation acc:", -result.fun)


#Random Search
import random
from itertools import product

def random_search(n_trials=20):
    search_space = {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3],
        "weight_decay": [1e-6, 1e-5, 1e-4],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "batch_size": [64, 128, 256],
        "model_type": ["cnn", "mlp"]
    }

    all_combinations = list(product(
        search_space["lr"],
        search_space["weight_decay"],
        search_space["dropout"],
        search_space["batch_size"],
        search_space["model_type"]
    ))

    sampled_combinations = random.sample(all_combinations, min(n_trials, len(all_combinations)))

    best_val_acc = 0.0
    best_cfg = None

    for i, (lr, wd, dropout, bs, model_type) in enumerate(sampled_combinations):
        print(f"\n🔍 Trial {i+1}/{n_trials}")
        cfg = TrainConfig(
            model_type=model_type,
            batch_size=bs,
            epochs=5,
            lr=lr,
            weight_decay=wd,
            dropout=dropout,
            seed=42,
            val_split=0.1,
            num_workers=4,
            data_dir="./data",
            save_path=f"./random_model_trial_{i+1}.pt"
        )

        set_seed(cfg.seed)
        device = get_device()

        train_loader, val_loader, _ = get_dataloaders(cfg)
        model = build_model(cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_trial_acc = 0.0
        for epoch in range(cfg.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            best_trial_acc = max(best_trial_acc, val_acc)

        print(f"Trial {i+1} val_acc: {best_trial_acc:.4f}")

        if best_trial_acc > best_val_acc:
            best_val_acc = best_trial_acc
            best_cfg = cfg

    print("\n✅ Best config (Random Search):")
    print(best_cfg)
    print(f"Best val_acc: {best_val_acc:.4f}")

#Grid Search
def grid_search():
    search_space = {
        "lr": [1e-3, 5e-3],
        "weight_decay": [1e-5, 1e-4],
        "dropout": [0.2, 0.4],
        "batch_size": [128],
        "model_type": ["cnn", "mlp"]
    }

    grid = list(product(
        search_space["lr"],
        search_space["weight_decay"],
        search_space["dropout"],
        search_space["batch_size"],
        search_space["model_type"]
    ))

    best_val_acc = 0.0
    best_cfg = None

    for i, (lr, wd, dropout, bs, model_type) in enumerate(grid):
        print(f"\n🔍 Grid Trial {i+1}/{len(grid)}")
        cfg = TrainConfig(
            model_type=model_type,
            batch_size=bs,
            epochs=5,
            lr=lr,
            weight_decay=wd,
            dropout=dropout,
            seed=42,
            val_split=0.1,
            num_workers=4,
            data_dir="./data",
            save_path=f"./grid_model_trial_{i+1}.pt"
        )

        set_seed(cfg.seed)
        device = get_device()

        train_loader, val_loader, _ = get_dataloaders(cfg)
        model = build_model(cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_trial_acc = 0.0
        for epoch in range(cfg.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            best_trial_acc = max(best_trial_acc, val_acc)

        print(f"Grid Trial {i+1} val_acc: {best_trial_acc:.4f}")

        if best_trial_acc > best_val_acc:
            best_val_acc = best_trial_acc
            best_cfg = cfg

    print("\n✅ Best config (Grid Search):")
    print(best_cfg)
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
    # random_search()
    # grid_search()



# Random Search = Train loss 0.2158 acc 0.9210 | Val loss 0.2073 acc 0.9240
# Grid Search = Test loss 0.2206 acc 0.9192
# Bayesian = Test loss 0.2200 acc 0.9182  