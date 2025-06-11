"""
Motion Transformer (MTR) for Maritime Trajectory Prediction

Implementation of the Motion Transformer adapted for maritime vessel trajectory prediction.
This model uses a two-step optimization approach: global intention localization and 
local movement refinement to generate multimodal trajectory predictions.

Paper: "Motion Transformer with Global Intention Localization and Local Movement Refinement"
Authors: Shaoshuai Shi, Li Jiang, Dengxin Dai, Bernt Schiele
Venue: NeurIPS 2022 (Waymo Open Dataset Motion Prediction Challenge Winner)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for transformer.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with cross-attention.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class ContextEncoder(nn.Module):
    """
    Context encoder for processing historical trajectory data.
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 4, 
                 n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Encoded context [batch_size, seq_len, d_model]
        """
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x


class MotionDecoder(nn.Module):
    """
    Motion decoder with learnable query vectors for multimodal prediction.
    """
    
    def __init__(self, d_model: int = 256, n_queries: int = 6, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1,
                 prediction_horizon: int = 30, output_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim
        
        # Learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(n_queries, d_model))
        
        # Positional encoding for queries
        self.query_pos_encoding = nn.Parameter(torch.randn(prediction_horizon, d_model))
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection for trajectory prediction
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Confidence/probability head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, context: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            context: Encoded context [batch_size, seq_len, d_model]
            context_mask: Optional context mask
            
        Returns:
            Dictionary containing predictions and confidences
        """
        batch_size = context.size(0)
        
        # Expand query embeddings for batch
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional encoding to queries for each time step
        query_sequence = []
        for t in range(self.prediction_horizon):
            pos_queries = queries + self.query_pos_encoding[t:t+1, :].unsqueeze(0)
            query_sequence.append(pos_queries)
        
        # Stack to create full query sequence
        # [batch_size, prediction_horizon, n_queries, d_model]
        query_sequence = torch.stack(query_sequence, dim=1)
        
        # Reshape for decoder processing
        # [batch_size, prediction_horizon * n_queries, d_model]
        query_flat = query_sequence.view(batch_size, -1, self.d_model)
        
        # Pass through decoder layers
        decoder_output = query_flat
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, context, memory_mask=context_mask)
        
        # Reshape back to [batch_size, prediction_horizon, n_queries, d_model]
        decoder_output = decoder_output.view(batch_size, self.prediction_horizon, self.n_queries, self.d_model)
        
        # Generate trajectory predictions
        trajectories = self.trajectory_head(decoder_output)
        # [batch_size, prediction_horizon, n_queries, output_dim]
        
        # Generate confidence scores
        confidences = self.confidence_head(decoder_output.mean(dim=1))
        # [batch_size, n_queries, 1]
        
        return {
            'trajectories': trajectories,
            'confidences': confidences.squeeze(-1),
            'query_features': decoder_output
        }


class MotionTransformer(nn.Module):
    """
    Motion Transformer for maritime trajectory prediction.
    
    Implements a two-step approach:
    1. Global intention localization via learnable queries
    2. Local movement refinement via transformer decoder
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_queries: int = 6,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        prediction_horizon: int = 30,
        output_dim: int = 4,  # lat, lon, sog, cog
        max_context_len: int = 100
    ):
        """
        Initialize Motion Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_queries: Number of learnable query vectors (motion modes)
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            n_heads: Number of attention heads
            d_ff: Feed forward dimension
            dropout: Dropout rate
            prediction_horizon: Number of future time steps to predict
            output_dim: Output feature dimension
            max_context_len: Maximum context sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_queries = n_queries
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Motion decoder
        self.motion_decoder = MotionDecoder(
            d_model=d_model,
            n_queries=n_queries,
            n_layers=decoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
            output_dim=output_dim
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, 
                context_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Motion Transformer.
        
        Args:
            x: Input context tensor [batch_size, context_len, input_dim]
            context_mask: Optional context mask
            
        Returns:
            Dictionary containing:
                - trajectories: Predicted trajectories [batch_size, pred_horizon, n_queries, output_dim]
                - confidences: Mode confidences [batch_size, n_queries]
                - context_features: Encoded context features
        """
        # Encode context
        context_features = self.context_encoder(x, context_mask)
        
        # Decode motion
        motion_outputs = self.motion_decoder(context_features, context_mask)
        
        return {
            'trajectories': motion_outputs['trajectories'],
            'confidences': motion_outputs['confidences'],
            'context_features': context_features,
            'query_features': motion_outputs['query_features']
        }
    
    def predict_best_trajectory(self, x: torch.Tensor, 
                               context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict the most likely trajectory (highest confidence mode).
        
        Args:
            x: Input context tensor
            context_mask: Optional context mask
            
        Returns:
            Best trajectory [batch_size, pred_horizon, output_dim]
        """
        outputs = self.forward(x, context_mask)
        
        # Get the mode with highest confidence
        best_mode_idx = outputs['confidences'].argmax(dim=-1)  # [batch_size]
        
        # Extract best trajectory for each sample in batch
        batch_size = x.size(0)
        best_trajectories = []
        
        for b in range(batch_size):
            best_traj = outputs['trajectories'][b, :, best_mode_idx[b], :]
            best_trajectories.append(best_traj)
        
        return torch.stack(best_trajectories, dim=0)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     loss_type: str = 'best_of_n') -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth trajectories [batch_size, pred_horizon, output_dim]
            loss_type: Type of loss ('best_of_n', 'gmm', 'weighted')
            
        Returns:
            Dictionary of loss components
        """
        trajectories = outputs['trajectories']  # [batch_size, pred_horizon, n_queries, output_dim]
        confidences = outputs['confidences']    # [batch_size, n_queries]
        
        batch_size, pred_horizon, n_queries, output_dim = trajectories.shape
        
        # Expand targets for comparison with all modes
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, n_queries, -1)
        
        if loss_type == 'best_of_n':
            # Best-of-N loss: minimize loss of closest prediction
            trajectory_errors = F.mse_loss(trajectories, targets_expanded, reduction='none')
            trajectory_errors = trajectory_errors.mean(dim=(1, 3))  # [batch_size, n_queries]
            
            # Find best mode for each sample
            best_mode_errors, best_mode_idx = trajectory_errors.min(dim=-1)
            
            # Regression loss on best mode
            regression_loss = best_mode_errors.mean()
            
            # Classification loss to encourage correct mode selection
            classification_loss = F.cross_entropy(confidences, best_mode_idx)
            
            total_loss = regression_loss + 0.1 * classification_loss
            
            return {
                'total_loss': total_loss,
                'regression_loss': regression_loss,
                'classification_loss': classification_loss,
                'best_mode_errors': best_mode_errors
            }
        
        elif loss_type == 'weighted':
            # Weighted loss using confidence scores
            trajectory_errors = F.mse_loss(trajectories, targets_expanded, reduction='none')
            trajectory_errors = trajectory_errors.mean(dim=(1, 3))  # [batch_size, n_queries]
            
            # Weight errors by confidence
            weighted_errors = trajectory_errors * confidences
            regression_loss = weighted_errors.sum(dim=-1).mean()
            
            # Regularization to prevent confidence collapse
            confidence_reg = -torch.log(confidences + 1e-8).mean()
            
            total_loss = regression_loss + 0.01 * confidence_reg
            
            return {
                'total_loss': total_loss,
                'regression_loss': regression_loss,
                'confidence_regularization': confidence_reg
            }
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class MotionTransformerTrainer:
    """
    Trainer class for Motion Transformer.
    """
    
    def __init__(
        self,
        model: MotionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = 'best_of_n',
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: MotionTransformer model
            learning_rate: Learning rate
            weight_decay: Weight decay
            loss_type: Type of loss function
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        
        # Optimizer with learning rate warm-up
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=1000,  # Will be updated based on actual training steps
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
    def train_step(self, context: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            context: Context tensor [batch_size, context_len, input_dim]
            targets: Target trajectories [batch_size, pred_horizon, output_dim]
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(context)
        
        # Compute loss
        loss_dict = self.model.compute_loss(outputs, targets, self.loss_type)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in loss_dict.items()}
    
    def validate_step(self, context: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            context: Context tensor
            targets: Target trajectories
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(context)
            loss_dict = self.model.compute_loss(outputs, targets, self.loss_type)
            
            # Additional metrics
            best_trajectory = self.model.predict_best_trajectory(context)
            ade = F.mse_loss(best_trajectory, targets, reduction='none').mean(dim=-1).sqrt().mean()
            fde = F.mse_loss(best_trajectory[:, -1], targets[:, -1], reduction='none').mean(dim=-1).sqrt().mean()
            
            loss_dict.update({
                'val_ade': ade,
                'val_fde': fde
            })
            
            return {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in loss_dict.items()}


def create_motion_transformer(
    input_dim: int,
    d_model: int = 256,
    n_queries: int = 6,
    prediction_horizon: int = 30,
    **kwargs
) -> MotionTransformer:
    """
    Factory function to create MotionTransformer model.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        n_queries: Number of motion modes
        prediction_horizon: Prediction horizon
        **kwargs: Additional model parameters
        
    Returns:
        MotionTransformer model
    """
    return MotionTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_queries=n_queries,
        prediction_horizon=prediction_horizon,
        **kwargs
    )


# Maritime-specific configurations
MARITIME_MTR_CONFIG = {
    'small': {
        'input_dim': 4,
        'd_model': 128,
        'n_queries': 4,
        'encoder_layers': 2,
        'decoder_layers': 3,
        'n_heads': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'prediction_horizon': 10,
        'output_dim': 4
    },
    'medium': {
        'input_dim': 4,
        'd_model': 256,
        'n_queries': 6,
        'encoder_layers': 4,
        'decoder_layers': 6,
        'n_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'prediction_horizon': 30,
        'output_dim': 4
    },
    'large': {
        'input_dim': 4,
        'd_model': 512,
        'n_queries': 8,
        'encoder_layers': 6,
        'decoder_layers': 8,
        'n_heads': 16,
        'd_ff': 2048,
        'dropout': 0.1,
        'prediction_horizon': 60,
        'output_dim': 4
    }
}


def create_maritime_motion_transformer(size: str = 'medium') -> MotionTransformer:
    """
    Create MotionTransformer configured for maritime applications.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        
    Returns:
        Configured MotionTransformer model
    """
    if size not in MARITIME_MTR_CONFIG:
        raise ValueError(f"Unknown size: {size}. Available: {list(MARITIME_MTR_CONFIG.keys())}")
    
    config = MARITIME_MTR_CONFIG[size]
    return create_motion_transformer(**config)


if __name__ == "__main__":
    # Test the model
    print("Testing Motion Transformer...")
    
    # Create model
    model = create_maritime_motion_transformer('small')
    
    # Test input
    batch_size, context_len, input_dim = 4, 20, 13
    pred_horizon = 10
    
    context = torch.randn(batch_size, context_len, input_dim)
    targets = torch.randn(batch_size, pred_horizon, 4)
    
    # Forward pass
    outputs = model(context)
    
    print(f"Context shape: {context.shape}")
    print(f"Trajectories shape: {outputs['trajectories'].shape}")
    print(f"Confidences shape: {outputs['confidences'].shape}")
    
    # Test best trajectory prediction
    best_traj = model.predict_best_trajectory(context)
    print(f"Best trajectory shape: {best_traj.shape}")
    
    # Test loss computation
    loss_dict = model.compute_loss(outputs, targets)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    
    print("✅ Motion Transformer test completed successfully!")

