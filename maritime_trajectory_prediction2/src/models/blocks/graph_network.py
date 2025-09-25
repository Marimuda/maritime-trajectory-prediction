# File: src/models/blocks/graph_network.py
"""
Raw graph building blocks for maritime AIS fusion.
Contains:
- CosineAttention
- SpatialGatedBlock
- PolylineSubgraphEncoder
- MaritimeGraphNetwork (as nn.Module)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class CosineAttention(nn.Module):
    """Compute attention weights via cosine similarity."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Optional learnable projection
        self.W = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]
        # similarity: [B, N, N]
        normed = F.normalize(x @ self.W, dim=-1)
        sim = torch.matmul(normed, normed.transpose(-2, -1))
        attn = torch.sigmoid(sim)  # gating weights
        # reduce to per-node scalar by mean
        return attn.mean(dim=1, keepdim=True)  # [B, 1, N]


class SpatialGatedBlock(nn.Module):
    """Graph convolution with learned spatial gating."""

    def __init__(self, dim: int):
        super().__init__()
        self.cos_att = CosineAttention(dim)
        self.gcn = GCNConv(dim, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        # x: [B*N, dim], edge_index: [2, E]
        att = self.cos_att(x)  # [B,1,N]
        g = self.gcn(x, edge_index)
        # elementwise gate (broadcast att over features)
        return g * att.squeeze(1).unsqueeze(-1)


class PolylineSubgraphEncoder(nn.Module):
    """Encode per-vessel AIS polyline into latent features."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        # x: [B*N, features], edge_index: [2, E]
        x = F.relu(self.conv1(x, edge_index))
        return F.relu(self.conv2(x, edge_index))


class MaritimeGraphNetwork(nn.Module):
    """Combine polyline encoding and spatial gating."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.poly_encoder = PolylineSubgraphEncoder(input_dim, hidden_dim)
        self.spatial_gate = SpatialGatedBlock(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        h = self.poly_encoder(x, edge_index)
        return self.spatial_gate(h, edge_index)
