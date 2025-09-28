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


def build_model(cfg: TrainConfig):
    if cfg.model_type == "mlp":
        return MLP(dropout=cfg.dropout, hidden_dim=cfg.hidden_dim)
    elif cfg.model_type == "cnn":
        return SmallCNN(dropout=cfg.dropout, fc_dim=cfg.cnn_fc_dim)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def make_optimizer(cfg: TrainConfig, params):
    opt = cfg.optimizer.lower()
    if opt == "adam":
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adamw":
        return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "sgd":
        return optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


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


# HPO-friendly train/eval that reports per-epoch validation accuracy
def train_eval_hpo(cfg: TrainConfig, trial=None):
    set_seed(cfg.seed)
    device = get_device()

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    optimizer = make_optimizer(cfg, model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| Val loss {va_loss:.4f} acc {va_acc:.4f}")

        if trial is not None:
            import optuna  # local import to avoid hard dependency if not using HPO
            trial.report(va_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Final test evaluation with best state
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")

    return best_val_acc, te_acc, best_state


# ---------------------------
# Optuna objective & driver
# ---------------------------
def objective(trial):
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner

    # Search space
    model_type = trial.suggest_categorical("model_type", ["cnn", "mlp"])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Model-specific knobs
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384]) if model_type == "mlp" else 256
    cnn_fc_dim = trial.suggest_categorical("cnn_fc_dim", [64, 128, 192]) if model_type == "cnn" else 128
    momentum = trial.suggest_float("momentum", 0.5, 0.95) if optimizer == "sgd" else 0.9

    # Fewer epochs per trial; pruner will stop weak ones early
    epochs = trial.suggest_categorical("epochs", [5, 7, 9])

    cfg = TrainConfig(
        model_type=model_type,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        seed=42,
        val_split=0.1,
        num_workers=4,
        data_dir="./data",
        save_path="./fashion_mnist_baseline.pt",
        optimizer=optimizer,
        momentum=momentum,
        hidden_dim=hidden_dim,
        cnn_fc_dim=cnn_fc_dim,
    )

    best_val_acc, test_acc, _ = train_eval_hpo(cfg, trial=trial)
    return best_val_acc  # maximize val accuracy


def run_hpo(n_trials: int = 20, seed: int = 42):
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=5)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best trial:", study.best_trial.number)
    print("Best value (val_acc):", study.best_value)
    print("Best params:", study.best_trial.params)
    return study


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST baseline (PyTorch + Optuna)")
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
    # HPO
    p.add_argument("--hpo_trials", type=int, default=0, help="Run Optuna with N trials (0 = no HPO)")
    return p.parse_args()


def main():
    args = parse_args()

    # HPO path
    if args.hpo_trials and args.hpo_trials > 0:
        study = run_hpo(args.hpo_trials, seed=args.seed)
        params = study.best_trial.params
        cfg = TrainConfig(
            model_type=params.get("model_type", "cnn"),
            batch_size=params.get("batch_size", 128),
            epochs=params.get("epochs", 7),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
            dropout=params.get("dropout", 0.2),
            seed=args.seed,
            val_split=0.1,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            save_path="./fashion_mnist_best.pt",
            optimizer=params.get("optimizer", "adam"),
            momentum=params.get("momentum", 0.9),
            hidden_dim=params.get("hidden_dim", 256),
            cnn_fc_dim=params.get("cnn_fc_dim", 128),
        )
        best_val_acc, test_acc, best_state = train_eval_hpo(cfg, trial=None)
        if best_state is not None:
            torch.save({"model_state_dict": best_state, "config": cfg.__dict__}, cfg.save_path)
        print(f"[BEST] Val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f} | Saved: {cfg.save_path}")
        return

    # Regular training path (no HPO)
    cfg = TrainConfig(
        model_type=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        val_split=args.val_split,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        save_path=args.save_path,
    )
    best_val_acc, test_acc, best_state = train_eval_hpo(cfg, trial=None)
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg.__dict__}, cfg.save_path)
    print(f"[RUN] Val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f} | Saved: {cfg.save_path}")


if __name__ == "__main__":
    main()
