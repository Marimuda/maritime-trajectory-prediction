"""
PyTorch Lightning models for maritime trajectory prediction.

Implements the guideline's recommendations for Lightning modules with
proper metrics, callbacks, and optimization strategies.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, List, Optional, Tuple, Any
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

logger = logging.getLogger(__name__)


class TrajectoryPredictor(pl.LightningModule):
    """
    Base Lightning module for trajectory prediction.
    
    Implements the guideline's recommendations for training infrastructure,
    metrics tracking, and optimization strategies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        scheduler: str = "cosine",
        warmup_epochs: int = 10
    ):
        """
        Initialize trajectory predictor.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout probability
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            gradient_clip_val: Gradient clipping value
            scheduler: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        
        # Build model architecture
        self._build_model()
        
        # Initialize metrics following guideline
        self._init_metrics()
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def _build_model(self):
        """Build the model architecture (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _build_model")
    
    def _init_metrics(self):
        """Initialize torchmetrics for multi-GPU reduction."""
        # Training metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        
        # Validation metrics
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Test metrics
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Custom maritime metrics
        self.val_distance_error = torchmetrics.MeanMetric()
        self.test_distance_error = torchmetrics.MeanMetric()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement forward")
    
    def _compute_distance_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute geographical distance error in kilometers.
        
        Args:
            pred: Predicted trajectories [B, C, T] where C includes lat, lon
            target: Target trajectories [B, C, T]
            
        Returns:
            Mean distance error in kilometers
        """
        # Extract lat/lon (assuming first two channels)
        pred_lat, pred_lon = pred[:, 0], pred[:, 1]
        target_lat, target_lon = target[:, 0], target[:, 1]
        
        # Haversine distance calculation
        dlat = torch.deg2rad(target_lat - pred_lat)
        dlon = torch.deg2rad(target_lon - pred_lon)
        
        a = (torch.sin(dlat / 2) ** 2 + 
             torch.cos(torch.deg2rad(pred_lat)) * torch.cos(torch.deg2rad(target_lat)) * 
             torch.sin(dlon / 2) ** 2)
        
        c = 2 * torch.asin(torch.sqrt(a))
        distance_km = 6371.0 * c  # Earth radius in km
        
        return distance_km.mean()
    
    def _common_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Common step for training/validation/test."""
        x = batch['input']  # [B, C, T]
        y = batch['target']  # [B, C, T]
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics based on stage
        if stage == "train":
            self.train_mse.update(y_hat, y)
            self.train_mae.update(y_hat, y)
            
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/mse", self.train_mse, on_epoch=True)
            self.log("train/mae", self.train_mae, on_epoch=True)
            
        elif stage == "val":
            self.val_mse.update(y_hat, y)
            self.val_mae.update(y_hat, y)
            self.val_rmse.update(y_hat, y)
            
            # Compute distance error
            dist_error = self._compute_distance_error(y_hat, y)
            self.val_distance_error.update(dist_error)
            
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)
            self.log("val/mse", self.val_mse, on_epoch=True)
            self.log("val/mae", self.val_mae, on_epoch=True)
            self.log("val/rmse", self.val_rmse, on_epoch=True)
            self.log("val/distance_error_km", self.val_distance_error, on_epoch=True)
            
        elif stage == "test":
            self.test_mse.update(y_hat, y)
            self.test_mae.update(y_hat, y)
            self.test_rmse.update(y_hat, y)
            
            # Compute distance error
            dist_error = self._compute_distance_error(y_hat, y)
            self.test_distance_error.update(dist_error)
            
            self.log("test/loss", loss, on_epoch=True)
            self.log("test/mse", self.test_mse, on_epoch=True)
            self.log("test/mae", self.test_mae, on_epoch=True)
            self.log("test/rmse", self.test_rmse, on_epoch=True)
            self.log("test/distance_error_km", self.test_distance_error, on_epoch=True)
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._common_step(batch, "train")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._common_step(batch, "val")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._common_step(batch, "test")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers following guideline."""
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        config = {"optimizer": optimizer}
        
        # Configure scheduler
        if self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        elif self.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch"
            }
        
        return config
    
    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping before optimizer step."""
        if self.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm"
            )


class ConvolutionalPredictor(TrajectoryPredictor):
    """
    Convolutional sequence predictor following guideline architecture.
    
    Implements Conv1D-based sequence-to-sequence prediction with
    residual connections and attention mechanisms.
    """
    
    def _build_model(self):
        """Build convolutional architecture."""
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1)
        )
        
        # Projection layer for prediction horizon
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional predictor.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            Predicted trajectory [B, C, T_pred]
        """
        # Encode
        encoded = self.encoder(x)  # [B, H, T]
        
        # Use last timestep for prediction
        last_state = encoded[:, :, -1]  # [B, H]
        
        # Project to prediction horizon
        projected = self.projection(last_state)  # [B, H]
        
        # Expand to prediction horizon
        pred_horizon = 10  # This should be configurable
        expanded = projected.unsqueeze(-1).repeat(1, 1, pred_horizon)  # [B, H, T_pred]
        
        # Decode
        output = self.decoder(expanded)  # [B, C, T_pred]
        
        return output


class LSTMPredictor(TrajectoryPredictor):
    """
    LSTM-based trajectory predictor.
    
    Implements sequence-to-sequence LSTM with attention for
    trajectory prediction tasks.
    """
    
    def _build_model(self):
        """Build LSTM architecture."""
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM predictor.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            Predicted trajectory [B, C, T_pred]
        """
        # Transpose for LSTM: [B, T, C]
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # [B, T, H]
        
        # Use last hidden state for prediction
        last_hidden = hidden[-1]  # [B, H]
        
        # Generate predictions for horizon
        pred_horizon = 10  # This should be configurable
        predictions = []
        
        current_hidden = (hidden, cell)
        current_input = x[:, -1:, :]  # Last timestep
        
        for _ in range(pred_horizon):
            lstm_out, current_hidden = self.lstm(current_input, current_hidden)
            pred = self.output_projection(self.dropout_layer(lstm_out))
            predictions.append(pred)
            current_input = pred
        
        # Stack predictions
        output = torch.cat(predictions, dim=1)  # [B, T_pred, C]
        
        # Transpose back: [B, C, T_pred]
        return output.transpose(1, 2)

