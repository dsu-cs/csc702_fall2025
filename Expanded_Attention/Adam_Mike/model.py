# model.py
# Adam and Mike (and Ashar)
"""
This is the Transformer model with Positional Encoding.

We're borrowing from Ashar and Mike's model definition from last week, but 
replacing the attention mechanism with a MultiHeadAttention module borrowed 
from https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html.
"""
import math
from typing import Optional

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import copy

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attn_mask,  # Pass it through
            dropout_p=self.dropout, 
            is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        # Create positional encoding matrix
        pos_matrix = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scaling = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_matrix[:, 0::2] = torch.sin(positions * scaling)
        pos_matrix[:, 1::2] = torch.cos(positions * scaling)
        self.register_buffer('positional_encoding', pos_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding[:seq_len, :]
    
class TransformerEncoderLayer(nn.Module):
    """Custom encoder layer using your MultiHeadAttention."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            E_q=d_model, E_k=d_model, E_v=d_model,
            E_total=d_model, nheads=nhead, dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal=False):
        # Self-attention with residual
        attn_mask = None
        if src_key_padding_mask is not None:
            # [B, T] -> [B, 1, 1, T] -> broadcasts to [B, nheads, T, T]
            attn_mask = src_key_padding_mask[:, None, None, :]
        
        src2 = self.self_attn(src, src, src, attn_mask=attn_mask, is_causal=is_causal)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal=False):
        output = src
        for mod in self.layers:
            output = mod(output, attn_mask=attn_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerSentimentAnalysis(nn.Module):
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
        self.nhead = nhead
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.encoder = TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        padding_mask = (x == self.pad_idx)  # [B, T]
        h = self.token_emb(x)
        h = self.pos_enc(h)
        
        # Create attention mask for padding
        # SDPA expects False for valid positions, True for masked
        attn_mask = None
        if padding_mask.any():
            # Expand to [B, 1, 1, T] so it broadcasts to [B, nheads, T, T]
            # In SDPA, True = masked (don't attend), False = valid
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        
        h = self.encoder(h, attn_mask=attn_mask)
        h = self.norm(h)

        # Mask-aware mean pooling
        mask = ~padding_mask  # [B, T] - True for valid tokens
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, d_model]
        
        logits = self.classifier(pooled)
        return logits
