import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class CosineAttention(nn.Module):
    """
    Computes attention weights via cosine similarity.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)  # [B, N, N]
        attn = torch.sigmoid(sim @ x)  # [B, N, D]
        return attn


class PolylineSubgraphEncoder(nn.Module):
    """
    Encodes each AIS polyline as a subgraph via GCN.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(4, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        # data.x: [total_nodes, 4], data.edge_index
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return x


class SpatialGatedBlock(nn.Module):
    """
    Applies spatial gating with cosine attention and a GCN layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.cos_att = CosineAttention(dim)
        self.gcn = GCNConv(dim, dim)

    def forward(self, x: torch.Tensor, edge_index) -> torch.Tensor:
        att = self.cos_att(x)  # [B, N, D]
        g = self.gcn(x, edge_index)  # [total_nodes, D]
        return g * att


class MaritimeGraphNetwork(nn.Module):
    """
    Combines subgraph encoding and spatial gating for AIS fusion.
    """

    def __init__(self, polyline_dim: int):
        super().__init__()
        self.encoder = PolylineSubgraphEncoder(polyline_dim)
        self.spatial_gate = SpatialGatedBlock(polyline_dim)

    def forward(self, batch):
        # batch: torch_geometric.data.Batch
        x = self.encoder(batch)
        x = self.spatial_gate(x, batch.edge_index)
        return x
