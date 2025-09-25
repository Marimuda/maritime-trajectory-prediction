# File: src/models/traisformer.py
"""
Transformer-based trajectory predictor using discretized AIS features.
Integrates transformer building blocks for concise implementation.
"""

import pytorch_lightning as pl
import torch
from torch import nn

# Reusable transformer components
from .blocks.transformer_blocks import (
    PositionalEncoding,
    TransformerBlock,
)


class TrAISformer(pl.LightningModule):
    """
    Transformer-based trajectory predictor using discretized features.

    Args:
        d_model: model dimension (must be divisible by 4)
        nhead: number of attention heads
        num_layers: number of encoder layers
        dim_feedforward: feedforward network dimension
        dropout: dropout probability
        lat_bins: number of latitude bins
        lon_bins: number of longitude bins
        sog_bins: number of speed-over-ground bins
        cog_bins: number of course-over-ground bins
        learning_rate: optimizer learning rate
        weight_decay: optimizer weight decay
        pred_horizon: prediction horizon (unused here)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        lat_bins: int,
        lon_bins: int,
        sog_bins: int,
        cog_bins: int,
        learning_rate: float,
        weight_decay: float,
        pred_horizon: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Embedding dimension per feature
        embed_dim = d_model // 4
        # Feature embeddings
        self.lat_embed = nn.Embedding(lat_bins, embed_dim)
        self.lon_embed = nn.Embedding(lon_bins, embed_dim)
        self.sog_embed = nn.Embedding(sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(cog_bins, embed_dim)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer block (stacks multiple layers)
        self.transformer = TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # Classification heads
        self.lat_head = nn.Linear(d_model, lat_bins)
        self.lon_head = nn.Linear(d_model, lon_bins)
        self.sog_head = nn.Linear(d_model, sog_bins)
        self.cog_head = nn.Linear(d_model, cog_bins)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def embed_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed and concatenate discrete feature indices from batch dict.
        Expects keys 'lat_idx','lon_idx','sog_idx','cog_idx'.
        Returns: [B, seq_len, d_model]
        """
        l = self.lat_embed(batch["lat_idx"])
        lo = self.lon_embed(batch["lon_idx"])
        s = self.sog_embed(batch["sog_idx"])
        c = self.cog_embed(batch["cog_idx"])
        return torch.cat([l, lo, s, c], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.embed_inputs(batch)
        x = self.pos_encoder(x)
        x = self.transformer(x, batch.get("mask"))
        return {
            "lat": self.lat_head(x),
            "lon": self.lon_head(x),
            "sog": self.sog_head(x),
            "cog": self.cog_head(x),
        }

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch)
        loss = sum(
            self.loss_fn(
                logits[k].view(-1, getattr(self.hparams, f"{k}_bins")),
                batch[f"next_{k}_idx"].view(-1),
            )
            for k in ["lat", "lon", "sog", "cog"]
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
