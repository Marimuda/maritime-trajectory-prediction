"""
Anomaly Transformer for Maritime Vessel Behavior Anomaly Detection

Implementation of the Anomaly Transformer (ICLR 2022) adapted for maritime AIS data.
This model uses anomaly-attention mechanism with association discrepancy for detecting
abnormal vessel behaviors in maritime traffic.

Paper: "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"
Authors: Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long
Venue: ICLR 2022 (Spotlight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class AnomalyAttention(nn.Module):
    """
    Anomaly-Attention mechanism that computes association discrepancy.
    
    This attention mechanism learns to distinguish between normal and anomalous
    patterns by measuring the discrepancy in attention distributions.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temperature parameter for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of anomaly attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attended output [batch_size, seq_len, d_model]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
            association: Association discrepancy [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Compute association discrepancy
        association = self._compute_association_discrepancy(attention, x)
        
        return output, attention, association
    
    def _compute_association_discrepancy(self, attention: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute association discrepancy for anomaly detection.
        
        Args:
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            association: Association discrepancy [batch_size, seq_len]
        """
        batch_size, n_heads, seq_len, _ = attention.shape
        
        # Average attention across heads
        avg_attention = attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Compute prior association (uniform distribution)
        prior = torch.ones(batch_size, seq_len, seq_len, device=attention.device) / seq_len
        
        # Compute KL divergence between attention and prior
        # Use a more stable computation
        eps = 1e-8
        avg_attention_stable = avg_attention + eps
        prior_stable = prior + eps
        
        # KL divergence: sum over last dimension (attention targets)
        kl_div = (avg_attention_stable * (avg_attention_stable / prior_stable).log()).sum(dim=-1)
        
        return kl_div


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with anomaly attention.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.anomaly_attention = AnomalyAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Layer output [batch_size, seq_len, d_model]
            attention: Attention weights
            association: Association discrepancy
        """
        # Anomaly attention with residual connection
        attn_output, attention, association = self.anomaly_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention, association


class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer model for maritime vessel behavior anomaly detection.
    
    This model uses a transformer architecture with anomaly-attention mechanism
    to detect abnormal patterns in vessel trajectories and behaviors.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Initialize Anomaly Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection for reconstruction
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Anomaly score computation
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Anomaly Transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed input [batch_size, seq_len, input_dim]
                - anomaly_scores: Anomaly scores [batch_size, seq_len]
                - association_discrepancy: Association discrepancy [batch_size, seq_len]
                - attention_weights: All attention weights from layers
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
            x_proj = x_proj + pos_enc
        
        x_proj = self.dropout(x_proj)
        
        # Pass through transformer layers
        hidden = x_proj
        all_attention = []
        total_association = torch.zeros(batch_size, seq_len, device=x.device)
        
        for layer in self.encoder_layers:
            hidden, attention, association = layer(hidden, mask)
            all_attention.append(attention)
            total_association += association
        
        # Average association discrepancy across layers
        avg_association = total_association / self.n_layers
        
        # Reconstruction
        reconstruction = self.output_projection(hidden)
        
        # Anomaly scores
        anomaly_scores = self.anomaly_scorer(hidden).squeeze(-1)
        
        return {
            'reconstruction': reconstruction,
            'anomaly_scores': anomaly_scores,
            'association_discrepancy': avg_association,
            'attention_weights': all_attention,
            'hidden_states': hidden
        }
    
    def compute_anomaly_criterion(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute anomaly detection criterion (loss components).
        
        Args:
            outputs: Model outputs
            targets: Target tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary of loss components
        """
        reconstruction = outputs['reconstruction']
        association = outputs['association_discrepancy']
        
        # Reconstruction loss - average over sequence and features
        recon_loss = F.mse_loss(reconstruction, targets, reduction='none').mean(dim=(1, 2))  # [batch_size]
        
        # Association discrepancy loss - average over sequence
        assoc_loss = association.mean(dim=1)  # [batch_size]
        
        # Combined anomaly criterion
        anomaly_criterion = recon_loss + assoc_loss
        
        return {
            'reconstruction_loss': recon_loss,
            'association_loss': assoc_loss,
            'anomaly_criterion': anomaly_criterion,
            'total_loss': anomaly_criterion.mean()
        }
    
    def detect_anomalies(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies in input sequences.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            threshold: Anomaly threshold
            
        Returns:
            Dictionary containing anomaly detection results
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Compute anomaly scores based on reconstruction error and association
            recon_error = F.mse_loss(outputs['reconstruction'], x, reduction='none').mean(dim=-1)
            association = outputs['association_discrepancy']
            
            # Combined anomaly score
            combined_score = (recon_error + association) / 2
            
            # Binary anomaly detection
            anomalies = (combined_score > threshold).float()
            
            return {
                'anomaly_scores': combined_score,
                'binary_anomalies': anomalies,
                'reconstruction_error': recon_error,
                'association_discrepancy': association,
                'confidence': torch.abs(combined_score - threshold)
            }


class AnomalyTransformerTrainer:
    """
    Trainer class for Anomaly Transformer with minimax training strategy.
    """
    
    def __init__(
        self,
        model: AnomalyTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_assoc: float = 1.0,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: AnomalyTransformer model
            learning_rate: Learning rate
            weight_decay: Weight decay
            lambda_assoc: Weight for association loss
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_assoc = lambda_assoc
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Input batch [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Compute losses
        loss_dict = self.model.compute_anomaly_criterion(outputs, batch)
        
        # Total loss with association weight
        total_loss = (
            loss_dict['reconstruction_loss'].mean() + 
            self.lambda_assoc * loss_dict['association_loss'].mean()
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': loss_dict['reconstruction_loss'].mean().item(),
            'association_loss': loss_dict['association_loss'].mean().item()
        }
    
    def validate_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(batch)
            loss_dict = self.model.compute_anomaly_criterion(outputs, batch)
            
            return {
                'val_total_loss': loss_dict['total_loss'].item(),
                'val_reconstruction_loss': loss_dict['reconstruction_loss'].mean().item(),
                'val_association_loss': loss_dict['association_loss'].mean().item()
            }


def create_anomaly_transformer(
    input_dim: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    **kwargs
) -> AnomalyTransformer:
    """
    Factory function to create AnomalyTransformer model.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        **kwargs: Additional model parameters
        
    Returns:
        AnomalyTransformer model
    """
    return AnomalyTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        **kwargs
    )


# Maritime-specific configurations
MARITIME_ANOMALY_CONFIG = {
    'small': {
        'input_dim': 13,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_len': 100
    },
    'medium': {
        'input_dim': 13,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len': 200
    },
    'large': {
        'input_dim': 13,
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 8,
        'd_ff': 3072,
        'dropout': 0.1,
        'max_seq_len': 500
    }
}


def create_maritime_anomaly_transformer(size: str = 'medium') -> AnomalyTransformer:
    """
    Create AnomalyTransformer configured for maritime applications.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        
    Returns:
        Configured AnomalyTransformer model
    """
    if size not in MARITIME_ANOMALY_CONFIG:
        raise ValueError(f"Unknown size: {size}. Available: {list(MARITIME_ANOMALY_CONFIG.keys())}")
    
    config = MARITIME_ANOMALY_CONFIG[size]
    return create_anomaly_transformer(**config)


if __name__ == "__main__":
    # Test the model
    print("Testing Anomaly Transformer...")
    
    # Create model
    model = create_maritime_anomaly_transformer('small')
    
    # Test input
    batch_size, seq_len, input_dim = 4, 50, 13
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Anomaly scores shape: {outputs['anomaly_scores'].shape}")
    print(f"Association discrepancy shape: {outputs['association_discrepancy'].shape}")
    
    # Test anomaly detection
    anomaly_results = model.detect_anomalies(x, threshold=0.5)
    print(f"Detected anomalies: {anomaly_results['binary_anomalies'].sum().item()}")
    
    print("✅ Anomaly Transformer test completed successfully!")

