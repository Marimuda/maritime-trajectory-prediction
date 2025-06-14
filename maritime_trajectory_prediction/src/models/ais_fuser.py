import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch
from src.models.transformer_blocks import TransformerBlock

class MaritimeGraphNetwork(pl.LightningModule):
    """PyG implementation of maritime graphical representations"""
    def __init__(self, polyline_dim=256, cluster_eps=0.015):
        super().__init__()
        self.polyline_encoder = PolylineSubgraphEncoder(polyline_dim)
        self.spatial_gate = SpatialGatedBlock(polyline_dim)
        
    def forward(self, data):
        # Process polyline subgraphs
        polylines = self.polyline_encoder(data)
        # Apply spatial gating
        return self.spatial_gate(polylines, data.edge_index)

class PolylineSubgraphEncoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(4, hidden_dim)  # Input: [d_start, d_end, attributes]
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return x

class SpatialGatedBlock(torch.nn.Module):
    """Implements cosAtt mechanism from the paper"""
    def __init__(self, dim):
        super().__init__()
        self.cos_att = CosineAttention(dim)
        self.gcn = GCNConv(dim, dim)
        
    def forward(self, x, edge_index):
        att = self.cos_att(x)
        g = self.gcn(x, edge_index)
        return g * att  # Hadamard product

class CosineAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(dim, dim))
        
    def forward(self, x):
        sim = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        return torch.sigmoid(sim @ x)

class AISFuserLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Maritime Graph Network
        self.graph_net = MaritimeGraphNetwork(
            polyline_dim=config.graph.polyline_dim,
            cluster_eps=config.graph.cluster_eps
        )
        
        # Temporal Transformer
        self.transformer = TransformerBlock(
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers
        )
        
        # SSL Weather Fusion
        self.weather_proj = torch.nn.Linear(config.weather_dim, config.transformer.d_model)
        self.ssl_head = torch.nn.Sequential(
            torch.nn.Linear(2*config.transformer.d_model, config.transformer.d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(config.transformer.d_model, 1)
        )
        
        # Output layers
        self.classifier = torch.nn.Linear(
            config.graph.polyline_dim + config.transformer.d_model,
            config.num_classes
        )

    def forward(self, data):
        # Process maritime graph
        graph_features = self.graph_net(data)
        
        # Process temporal sequence
        temporal_features = self.transformer(data.x)
        
        # Weather SSL fusion
        weather_emb = self.weather_proj(data.weather)
        ssl_loss = self._compute_ssl(temporal_features, weather_emb)
        
        # Combine features
        combined = torch.cat([graph_features, temporal_features], dim=-1)
        return self.classifier(combined), ssl_loss

    def training_step(self, batch, batch_idx):
        pred, ssl_loss = self(batch)
        main_loss = F.cross_entropy(pred, batch.y)
        total_loss = main_loss + self.hparams.ssl_beta * ssl_loss
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
    def _compute_ssl(self, temporal_features, weather_emb):
        """Compute SSL loss between temporal features and weather embeddings
        
        This implements a contrastive loss where temporal features and corresponding
        weather embeddings should be closer in embedding space than non-corresponding pairs.
        
        Args:
            temporal_features: Temporal features from transformer [batch_size, seq_len, d_model]
            weather_emb: Weather embeddings [batch_size, seq_len, d_model]
            
        Returns:
            SSL loss value
        """
        # Get the last temporal features (prediction point)
        temporal_last = temporal_features[:, -1, :]
        weather_last = weather_emb[:, -1, :]
        
        # Project features for SSL
        batch_size = temporal_last.size(0)
        
        # Concatenate features for SSL head
        combined = torch.cat([temporal_last, weather_last], dim=-1)
        ssl_logits = self.ssl_head(combined).squeeze(-1)
        
        # Create positive and negative pairs
        # For simplicity, we use a binary classification approach
        # where each sample is a positive pair (should be close)
        # In a more complex implementation, we would use negative sampling
        positive_targets = torch.ones_like(ssl_logits)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(ssl_logits, positive_targets)
        
        return loss
