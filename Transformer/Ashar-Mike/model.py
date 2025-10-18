# model.py
# Ashar and Mike
"""
Model definitions for a vanilla Transformer (non‑LLM) text classifier.
- Learnable token embeddings
- Sinusoidal positional encoding
- TransformerEncoder stack (PyTorch)
- Mean pooling (mask‑aware) for classification head
"""
import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T, :]


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        nlayers: int = 4,
        dropout: float = 0.1,
        pad_idx: int = 1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        key_padding_mask = (x == self.pad_idx)  # [B, T]
        h = self.token_emb(x)                   # [B, T, D]
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        # mask‑aware mean pooling
        lengths = (~key_padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (h * (~key_padding_mask).unsqueeze(-1)).sum(dim=1) / lengths
        logits = self.classifier(pooled)
        return logits
