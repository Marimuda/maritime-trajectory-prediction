# File: src/models/ais_fuser.py
"""
LightningModule for AIS fusion using graph and transformer blocks.
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

# Reusable blocks
from .blocks.ais_fuser_blocks import MaritimeGraphNetwork
from .blocks.transformer_blocks import TransformerBlock


class AISFuserLightning(pl.LightningModule):
    """
    Lightning wrapper for AIS fusion: graph encoding + temporal transformer + SSL fusion + classification.
    """

    def __init__(self, config: Any = None, **kwargs):
        super().__init__()
        # Handle Hydra config or manual kwargs
        if config is not None:
            self.save_hyperparameters(config)
            cfg = config
        else:
            self.save_hyperparameters(kwargs)

            class Cfg:
                pass

            cfg = Cfg()
            for k, v in kwargs.items():
                setattr(cfg, k, v)
        # Graph network
        self.graph_net = MaritimeGraphNetwork(polyline_dim=cfg.graph.polyline_dim)
        # Temporal transformer for sequence data
        self.transformer = TransformerBlock(
            d_model=cfg.transformer.d_model,
            nhead=cfg.transformer.nhead,
            num_layers=cfg.transformer.num_layers,
            dim_feedforward=cfg.transformer.dim_feedforward
            if hasattr(cfg.transformer, "dim_feedforward")
            else cfg.transformer.d_model * 4,
            dropout=cfg.transformer.dropout
            if hasattr(cfg.transformer, "dropout")
            else 0.1,
        )
        # Weather SSL projection
        self.weather_proj = nn.Linear(cfg.weather_dim, cfg.transformer.d_model)
        self.ssl_head = nn.Sequential(
            nn.Linear(2 * cfg.transformer.d_model, cfg.transformer.d_model),
            nn.ReLU(),
            nn.Linear(cfg.transformer.d_model, 1),
        )
        # Classification head
        total_dim = cfg.graph.polyline_dim + cfg.transformer.d_model
        self.classifier = nn.Linear(total_dim, cfg.num_classes)
        # SSL weight
        self.ssl_beta = getattr(cfg, "ssl_beta", 1.0)

    def forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Graph representation
        graph_feats = self.graph_net(batch)
        # Temporal encoding requires batch.x [B, seq_len, d_model]
        temp_feats = self.transformer(batch.x, None)
        # SSL fusion: project weather and compute SSL loss
        weather_emb = self.weather_proj(batch.weather)
        ssl_loss = self._compute_ssl(temp_feats, weather_emb)
        # Combine graph + temporal for classification
        combined = torch.cat([graph_feats, temp_feats], dim=-1)
        logits = self.classifier(combined)
        return logits, ssl_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, ssl_loss = self(batch)
        main_loss = F.cross_entropy(logits, batch.y)
        loss = main_loss + self.ssl_beta * ssl_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_main_loss", main_loss)
        self.log("train_ssl_loss", ssl_loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, ssl_loss = self(batch)
        main_loss = F.cross_entropy(logits, batch.y)
        loss = main_loss + self.ssl_beta * ssl_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_main_loss", main_loss)
        self.log("val_ssl_loss", ssl_loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, ssl_loss = self(batch)
        main_loss = F.cross_entropy(logits, batch.y)
        loss = main_loss + self.ssl_beta * ssl_loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr
            if hasattr(self.hparams, "lr")
            else self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
            if hasattr(self.hparams, "weight_decay")
            else 0.0,
        )
        return optimizer

    def _compute_ssl(
        self, temp_feats: torch.Tensor, weather_emb: torch.Tensor
    ) -> torch.Tensor:
        # Use last timestep features for SSL
        t_last = temp_feats[:, -1, :]
        w_last = weather_emb[:, -1, :]
        combined = torch.cat([t_last, w_last], dim=-1)
        ssl_logits = self.ssl_head(combined).squeeze(-1)
        targets = torch.ones_like(ssl_logits)
        return F.binary_cross_entropy_with_logits(ssl_logits, targets)
