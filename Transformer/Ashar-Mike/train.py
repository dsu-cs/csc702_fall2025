
# ============================
# train.py
# ============================
#Ashar and Mike
"""
Training entrypoint for the non‑LLM Transformer text classifier.
Usage example (CPU‑friendly):

python train.py \
  --epochs 3 \
  --d_model 128 \
  --nhead 4 \
  --nlayers 2 \
  --ff 256 \
  --batch_size 128 \
  --lr 5e-4

Each testing result will display and best result is captured in checkpoint.pt
"""
import argparse
import random
from typing import Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from model import TransformerTextClassifier
from data import get_dataloaders


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for X, y in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bsz = y.size(0)
        running_loss += loss.item() * bsz
        running_acc += accuracy(logits.detach(), y) * bsz
        n += bsz
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for X, y in tqdm(loader, desc="eval", leave=False):
            logits = model(X)
            loss = criterion(logits, y)
            bsz = y.size(0)
            running_loss += loss.item() * bsz
            running_acc += accuracy(logits, y) * bsz
            n += bsz
    return running_loss / n, running_acc / n

# Main driver routine.  Parameters can be passed in, all parameters have default values
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=4)
    parser.add_argument('--ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--min_freq', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='checkpoint.pt')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, vocab, pad_idx = get_dataloaders(batch_size=args.batch_size, device=device, min_freq=args.min_freq)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    model = TransformerTextClassifier(
        vocab_size=vocab_size,
        num_classes=4,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.ff,
        nlayers=args.nlayers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        print(f"Train loss {tr_loss:.4f} | acc {tr_acc:.4f} || Test loss {te_loss:.4f} | acc {te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab.get_stoi(),
                'pad_idx': pad_idx,
                'args': vars(args)
            }, args.save_path)
            print(f"Saved new best checkpoint to {args.save_path}")

    print(f"Best test acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()
