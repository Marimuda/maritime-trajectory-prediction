"""
Baseline models for maritime trajectory prediction, anomaly detection, and vessel interaction tasks.

This module implements three baseline models:
1. TrajectoryLSTM - LSTM-based trajectory prediction
2. AnomalyAutoencoder - Autoencoder-based anomaly detection  
3. VesselGCN - Graph Convolutional Network for vessel interactions

Each model includes appropriate architectures, loss functions, and evaluation metrics
for their respective maritime tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TrajectoryLSTM(nn.Module):
    """
    Baseline LSTM model for maritime trajectory prediction.
    
    Predicts future vessel positions based on historical trajectory data.
    Uses bidirectional LSTM with attention mechanism for better temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,  # lat, lon, sog, cog
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        logger.info(f"Initialized TrajectoryLSTM: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, bidirectional={bidirectional}")
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for trajectory prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths for packed sequences
            
        Returns:
            Predicted trajectories of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        if lengths is not None:
            # Pack sequences for variable length handling
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = lstm_out + attn_out
        
        # Output projection
        predictions = self.output_projection(lstm_out)
        
        return predictions
    
    def predict_future(self, x: torch.Tensor, future_steps: int = 5) -> torch.Tensor:
        """
        Predict future trajectory points using autoregressive generation.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            future_steps: Number of future steps to predict
            
        Returns:
            Future predictions of shape (batch_size, future_steps, output_dim)
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            predictions = []
            
            # Use the last prediction as input for next step
            current_input = x
            
            for _ in range(future_steps):
                # Predict next step
                pred = self.forward(current_input)
                next_pred = pred[:, -1:, :]  # Take last timestep
                predictions.append(next_pred)
                
                # Update input for next iteration (simplified - in practice would need feature engineering)
                # For now, just use the predicted position and assume other features remain constant
                if current_input.shape[2] > self.output_dim:
                    # Keep non-predicted features from last timestep
                    last_features = current_input[:, -1:, self.output_dim:]
                    next_input = torch.cat([next_pred, last_features], dim=2)
                else:
                    next_input = next_pred
                
                # Slide window: remove first timestep, add new prediction
                current_input = torch.cat([current_input[:, 1:, :], next_input], dim=1)
            
            return torch.cat(predictions, dim=1)


class AnomalyAutoencoder(nn.Module):
    """
    Baseline autoencoder model for maritime anomaly detection.
    
    Detects anomalous vessel behavior by learning to reconstruct normal patterns.
    High reconstruction error indicates potential anomalies.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: List[int] = [128, 96],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Initialized AnomalyAutoencoder: input_dim={input_dim}, "
                   f"encoding_dim={encoding_dim}, hidden_dims={hidden_dims}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, encoding)
        """
        # Handle sequence input by flattening
        original_shape = x.shape
        if len(original_shape) == 3:  # (batch, seq, features)
            batch_size, seq_len, features = original_shape
            x = x.view(batch_size * seq_len, features)
            
        encoding = self.encode(x)
        reconstruction = self.decode(encoding)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            reconstruction = reconstruction.view(batch_size, seq_len, -1)
            # For encoding, we want to return the mean across sequence
            encoding = encoding.view(batch_size, seq_len, -1).mean(dim=1)  # Average across sequence
        
        return reconstruction, encoding
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            
            # Compute reconstruction error
            mse_error = F.mse_loss(reconstruction, x, reduction='none')
            
            # Aggregate error across features
            if len(mse_error.shape) == 3:  # Sequence data
                anomaly_scores = mse_error.mean(dim=(1, 2))  # Average over seq and features
            else:
                anomaly_scores = mse_error.mean(dim=1)  # Average over features
            
            return anomaly_scores


class VesselGCN(nn.Module):
    """
    Baseline Graph Convolutional Network for vessel interaction modeling.
    
    Models vessel interactions and predicts collision risks or coordination patterns.
    Uses spatial and temporal vessel relationships for graph construction.
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,  # Risk score or interaction strength
        dropout: float = 0.2,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.aggregation = aggregation
        
        # Node feature transformation
        self.node_transform = nn.Linear(node_features, hidden_dim)
        
        # Edge feature transformation
        self.edge_transform = nn.Linear(edge_features, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        logger.info(f"Initialized VesselGCN: node_features={node_features}, "
                   f"edge_features={edge_features}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            node_features: Node features of shape (batch_size, num_nodes, node_features)
            edge_features: Edge features of shape (batch_size, num_nodes, num_nodes, edge_features)
            adjacency_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Node predictions of shape (batch_size, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Transform input features
        x = self.node_transform(node_features)  # (batch, nodes, hidden)
        edge_attr = self.edge_transform(edge_features)  # (batch, nodes, nodes, hidden)
        
        # Apply GCN layers
        for i, (gcn_layer, layer_norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            # GCN forward pass
            x_new = gcn_layer(x, edge_attr, adjacency_matrix)
            
            # Residual connection and layer norm
            if i > 0:  # Skip connection after first layer
                x_new = x + x_new
            x = layer_norm(x_new)
        
        # Output prediction
        output = self.output_layers(x)
        
        return output
    
    def predict_interactions(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict vessel interactions and collision risks.
        
        Returns:
            Dictionary with interaction predictions and risk scores
        """
        self.eval()
        with torch.no_grad():
            # Get node predictions
            node_outputs = self.forward(node_features, edge_features, adjacency_matrix)
            
            # Compute pairwise interaction scores
            batch_size, num_nodes, _ = node_outputs.shape
            
            # Expand for pairwise computation
            node_i = node_outputs.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch, nodes, nodes, output)
            node_j = node_outputs.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch, nodes, nodes, output)
            
            # Compute interaction scores (simple dot product)
            interaction_scores = (node_i * node_j).sum(dim=-1)  # (batch, nodes, nodes)
            
            # Apply adjacency mask (only consider connected vessels)
            interaction_scores = interaction_scores * adjacency_matrix
            
            # Compute collision risk (higher scores = higher risk)
            collision_risks = torch.sigmoid(interaction_scores)
            
            return {
                'node_predictions': node_outputs,
                'interaction_scores': interaction_scores,
                'collision_risks': collision_risks,
                'max_risk_per_vessel': collision_risks.max(dim=2)[0]  # Max risk for each vessel
            }


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer with edge features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Node transformation
        self.node_linear = nn.Linear(input_dim, output_dim)
        
        # Edge-aware message passing
        self.edge_linear = nn.Linear(input_dim, output_dim)
        self.message_linear = nn.Linear(output_dim * 2, output_dim)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GCN layer.
        
        Args:
            node_features: (batch, nodes, input_dim)
            edge_features: (batch, nodes, nodes, input_dim)
            adjacency_matrix: (batch, nodes, nodes)
            
        Returns:
            Updated node features: (batch, nodes, output_dim)
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Transform node features
        node_transformed = self.node_linear(node_features)  # (batch, nodes, output_dim)
        
        # Message passing with edge features
        messages = []
        
        for i in range(num_nodes):
            # Get neighbors for node i
            neighbors_mask = adjacency_matrix[:, i, :]  # (batch, nodes)
            
            # Node i features repeated for all neighbors
            node_i_features = node_transformed[:, i:i+1, :].expand(-1, num_nodes, -1)  # (batch, nodes, output_dim)
            
            # Edge features from i to all neighbors
            edge_i_features = self.edge_linear(edge_features[:, i, :, :])  # (batch, nodes, output_dim)
            
            # Combine node and edge features
            combined_features = torch.cat([node_i_features, edge_i_features], dim=-1)  # (batch, nodes, 2*output_dim)
            messages_i = self.message_linear(combined_features)  # (batch, nodes, output_dim)
            
            # Apply neighbor mask and aggregate
            messages_i = messages_i * neighbors_mask.unsqueeze(-1)  # Mask non-neighbors
            aggregated_message = messages_i.sum(dim=1, keepdim=True)  # (batch, 1, output_dim)
            
            messages.append(aggregated_message)
        
        # Stack messages for all nodes
        all_messages = torch.cat(messages, dim=1)  # (batch, nodes, output_dim)
        
        # Combine with original node features
        output = node_transformed + all_messages
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


# Model factory function
def create_baseline_model(task: str, **kwargs) -> nn.Module:
    """
    Factory function to create baseline models for different tasks.
    
    Args:
        task: Task name ('trajectory_prediction', 'anomaly_detection', 'vessel_interaction')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model
    """
    if task == 'trajectory_prediction':
        return TrajectoryLSTM(**kwargs)
    elif task == 'anomaly_detection':
        return AnomalyAutoencoder(**kwargs)
    elif task == 'vessel_interaction':
        return VesselGCN(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: "
                        f"'trajectory_prediction', 'anomaly_detection', 'vessel_interaction'")


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    # Test TrajectoryLSTM
    print("Testing TrajectoryLSTM...")
    model1 = create_baseline_model('trajectory_prediction', input_dim=13, hidden_dim=64, output_dim=4)
    x1 = torch.randn(2, 10, 13)  # batch=2, seq=10, features=13
    out1 = model1(x1)
    print(f"TrajectoryLSTM output shape: {out1.shape}")
    
    # Test AnomalyAutoencoder
    print("\nTesting AnomalyAutoencoder...")
    model2 = create_baseline_model('anomaly_detection', input_dim=13, encoding_dim=32)
    x2 = torch.randn(2, 10, 13)  # batch=2, seq=10, features=13
    recon2, enc2 = model2(x2)
    print(f"AnomalyAutoencoder reconstruction shape: {recon2.shape}")
    print(f"AnomalyAutoencoder encoding shape: {enc2.shape}")
    
    # Test VesselGCN
    print("\nTesting VesselGCN...")
    model3 = create_baseline_model('vessel_interaction', node_features=10, edge_features=5, hidden_dim=32)
    nodes3 = torch.randn(2, 5, 10)  # batch=2, nodes=5, features=10
    edges3 = torch.randn(2, 5, 5, 5)  # batch=2, nodes=5, nodes=5, edge_features=5
    adj3 = torch.randint(0, 2, (2, 5, 5)).float()  # batch=2, nodes=5, nodes=5
    out3 = model3(nodes3, edges3, adj3)
    print(f"VesselGCN output shape: {out3.shape}")
    
    print("\nAll baseline models created successfully!")

