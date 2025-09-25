import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        head_dim = d_model // nhead
        self.scale = math.sqrt(head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.head_dim = head_dim

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.size()
        # project
        q = self.q_proj(q).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        # score
        scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        # attend
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        y = self.norm1(x)
        y = self.attn(y, y, y, mask=mask)
        x = x + self.drop1(y)

        y = self.norm2(x)
        y = self.ff(y)
        return x + self.drop2(y)


class TransformerBlock(nn.Module):
    """Stacked transformer encoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class CausalSelfAttention(nn.Module):
    """Self-attention with causal mask."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        mask = ~mask  # causal
        return self.attn(x, x, x, mask=mask.unsqueeze(0))
