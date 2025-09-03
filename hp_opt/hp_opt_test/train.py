"""
train.py
--------
Main training script with:
  - Training/evaluation loop (with mixed precision support)
  - Optional schedulers: OneCycleLR, CosineAnnealingLR
  - (a) Random Search tuner
  - (b) Bayesian Optimization via Optuna (if installed)
  - (c) Learning Rate Finder (LR range test) + LR suggestion

Usage examples:
  # plain train with cosine scheduler
  python train.py --mode train --scheduler cosine --epochs 10

  # random search (25 trials)
  python train.py --mode random_search --trials 25

  # optuna bayesian optimization (30 trials)
  python train.py --mode optuna --trials 30

  # LR finder then OneCycle training
  python train.py --mode lr_finder --lr_start 1e-5 --lr_end 5e-1 --iters 200
  # then copy the suggested LR, or pass --use_suggested_lr to auto-run OneCycle
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

# Local modules
from data_utils import TrainConfig, set_seed, get_device, accuracy, get_dataloaders
from models import MLP, SmallCNN


# ---------------------------
# Model factory
# ---------------------------

def build_model(cfg: TrainConfig) -> nn.Module:
    """Create MLP or CNN based on cfg.model_type."""
    if cfg.model_type == "mlp":
        return MLP(dropout=cfg.dropout)
    elif cfg.model_type == "cnn":
        return SmallCNN(dropout=cfg.dropout)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


# ---------------------------
# Core train/eval loop
# ---------------------------

def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
    """Evaluate average loss and accuracy over a dataset."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(logits, labels) * imgs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def train_eval(
    cfg: TrainConfig,
    scheduler_name: Optional[str] = None,  # None | 'onecycle' | 'cosine'
    use_amp: bool = True,
    return_metrics: bool = True,
) -> Tuple[float, float]:
    """Train a model and report (best_val_acc, test_acc).
    If return_metrics=False, still trains and prints metrics, but returns (0,0).
    """
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional LR schedulers
    if scheduler_name == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,         # treat cfg.lr as the peak LR
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_state = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # OneCycle steps every batch; Cosine steps every epoch
            if scheduler is not None and scheduler_name == 'onecycle':
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            running_acc += accuracy(logits.detach(), labels.detach()) * imgs.size(0)

        n_tr = len(train_loader.dataset)
        tr_loss = running_loss / n_tr
        tr_acc = running_acc / n_tr

        # ---- Validate ----
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None and scheduler_name == 'cosine':
            scheduler.step()

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        # Track best validation model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save best checkpoint (optional)
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg.__dict__}, cfg.save_path)
        print(f"Saved best model to: {cfg.save_path} (val_acc={best_val_acc:.4f})")

    # Evaluate on test set using best state
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f}")

    return (best_val_acc, te_acc) if return_metrics else (0.0, 0.0)


# ---------------------------
# (a) Random Search tuner
# ---------------------------

def log_uniform(a: float, b: float) -> float:
    return math.exp(random.uniform(math.log(a), math.log(b)))


def random_search(n_trials: int = 25) -> TrainConfig:
    """Randomly sample hyperparameters and keep the best by val accuracy."""
    best_cfg, best_val = None, -1.0
    for t in range(1, n_trials + 1):
        cfg = TrainConfig(
            model_type=random.choice(["cnn", "mlp"]),
            batch_size=random.choice([64, 128, 256]),
            epochs=5,  # short for search; retrain longer later
            lr=log_uniform(1e-4, 3e-2),
            weight_decay=log_uniform(1e-6, 1e-2),
            dropout=random.uniform(0.0, 0.5),
        )
        print(f"\n=== Random Search Trial {t}/{n_trials} ===\n{cfg}")
        val_acc, _ = train_eval(cfg, scheduler_name=None, use_amp=True, return_metrics=True)
        if val_acc > best_val:
            best_cfg, best_val = cfg, val_acc
            print(f"** New best val_acc={best_val:.4f} with cfg={best_cfg}")
    print("\nRandom Search Best:\n", best_cfg, "\nVal Acc:", best_val)
    return best_cfg


# ---------------------------
# (b) Bayesian Optimization (Optuna/TPE)
# ---------------------------
try:
    import optuna
except Exception as e:
    optuna = None
    print("Optuna not available:", e)


def optuna_objective(trial: 'optuna.trial.Trial') -> float:
    cfg = TrainConfig(
        model_type=trial.suggest_categorical("model", ["cnn", "mlp"]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        epochs=5,
        lr=trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
    )
    val_acc, _ = train_eval(cfg, scheduler_name=None, use_amp=True, return_metrics=True)
    return val_acc


def run_optuna(n_trials: int = 30):
    if optuna is None:
        print("Please install Optuna: pip install optuna")
        return None
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials)
    print("\nOptuna Best Params:", study.best_trial.params)
    print("Best Val Acc:", study.best_value)
    return study


# ---------------------------
# (c) LR Range Test (Finder) + Suggestion
# ---------------------------
@torch.no_grad()
def _eval_batch_loss(model: nn.Module, criterion, imgs, labels, device: torch.device) -> float:
    model.eval()
    imgs, labels = imgs.to(device), labels.to(device)
    logits = model(imgs)
    loss = criterion(logits, labels)
    return loss.item()


def lr_range_test(
    base_cfg: TrainConfig,
    lr_start: float = 1e-6,
    lr_end: float = 1.0,
    num_iters: int = 200,
    stop_ratio: float = 4.0,
) -> List[tuple]:
    """Increase LR exponentially from lr_start to lr_end over num_iters; record loss.

    Stops early if loss exceeds stop_ratio * best_loss.
    Returns a list of (lr, loss) pairs.
    """
    set_seed(base_cfg.seed)
    device = get_device()

    # Data subset for speed
    from torch.utils.data import random_split, DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(base_cfg.data_dir, train=True, download=True, transform=transform)
    subset_size = min(10000, len(full_train))
    subset, _ = random_split(full_train, [subset_size, len(full_train) - subset_size], generator=torch.Generator().manual_seed(base_cfg.seed))
    loader = DataLoader(subset, batch_size=base_cfg.batch_size, shuffle=True, num_workers=base_cfg.num_workers, pin_memory=True)

    model = build_model(base_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_start, weight_decay=base_cfg.weight_decay)

    mult = (lr_end / lr_start) ** (1 / max(1, num_iters - 1))
    lr = lr_start

    losses: List[tuple] = []
    best_loss = float('inf')
    it = 0

    for imgs, labels in loader:
        if it >= num_iters:
            break
        it += 1
        for g in optimizer.param_groups:
            g['lr'] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs.to(device, non_blocking=True))
        loss = criterion(logits, labels.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()

        l = loss.item()
        losses.append((lr, l))
        if l < best_loss:
            best_loss = l
        if l > stop_ratio * best_loss:
            print(f"Stopping LR test early: loss {l:.4f} > {stop_ratio}× best {best_loss:.4f}")
            break
        lr *= mult

    return losses


def suggest_lr_from_range(losses: List[tuple]) -> Optional[float]:
    """Heuristic: choose a conservative LR near the sharpest descent.

    Implementation: pick the smallest LR whose loss <= 1.1 * min_loss.
    """
    if not losses:
        return None
    min_loss = min(l for _, l in losses)
    target = min_loss * 1.1
    candidates = [lr for lr, l in losses if l <= target]
    return min(candidates) if candidates else losses[-1][0]


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST training & tuning (PyTorch)")
    p.add_argument("--mode", type=str, default="train", choices=["train", "random_search", "optuna", "lr_finder"], help="What to run")
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

    # Scheduler and LR finder specific
    p.add_argument("--scheduler", type=str, default=None, choices=[None, "onecycle", "cosine"], help="Optional LR scheduler")
    p.add_argument("--trials", type=int, default=25, help="Trials for random_search/optuna")
    p.add_argument("--lr_start", type=float, default=1e-6, help="LR finder start")
    p.add_argument("--lr_end", type=float, default=1.0, help="LR finder end")
    p.add_argument("--iters", type=int, default=200, help="LR finder iterations")
    p.add_argument("--use_suggested_lr", action="store_true", help="If set during lr_finder, auto-train with OneCycle using suggested LR")
    return p.parse_args()


def main():
    args = parse_args()

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

    if args.mode == "train":
        train_eval(cfg, scheduler_name=args.scheduler, use_amp=True, return_metrics=False)

    elif args.mode == "random_search":
        best_cfg = random_search(n_trials=args.trials)
        print("\nRetraining best config longer (e.g., epochs=15)...")
        best_cfg.epochs = max(best_cfg.epochs, 15)
        train_eval(best_cfg, scheduler_name='cosine', use_amp=True, return_metrics=False)

    elif args.mode == "optuna":
        study = run_optuna(n_trials=args.trials)
        if study is not None:
            # Build cfg from best params and retrain longer
            params = study.best_trial.params
            best_cfg = TrainConfig(
                model_type=params.get('model', 'cnn'),
                batch_size=params.get('batch_size', 128),
                epochs=15,
                lr=params.get('lr', 1e-3),
                weight_decay=params.get('weight_decay', 1e-4),
                dropout=params.get('dropout', 0.2),
            )
            train_eval(best_cfg, scheduler_name='cosine', use_amp=True, return_metrics=False)

    elif args.mode == "lr_finder":
        losses = lr_range_test(cfg, lr_start=args.lr_start, lr_end=args.lr_end, num_iters=args.iters)
        # Print a quick textual summary
        if losses:
            best_lr = suggest_lr_from_range(losses)
            print(f"Suggested LR from finder: {best_lr:.3e}")
            if args.use_suggested_lr and best_lr is not None:
                cfg.lr = best_lr
                # OneCycle pairs well with LR finder
                train_eval(cfg, scheduler_name='onecycle', use_amp=True, return_metrics=False)
        else:
            print("LR finder returned no losses; try increasing iters or adjusting range.")


if __name__ == "__main__":
    main()