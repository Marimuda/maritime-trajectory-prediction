"""
LightningModule: Bidirectional LSTM with attention for trajectory prediction.

Supports optional maritime safety-aware loss for collision avoidance training.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class TrajectoryLSTMLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_maritime_safety: bool = False,
        collision_weight: float = 10.0,
        feasibility_weight: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=self.hparams.input_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers > 1 else 0.0,
            bidirectional=self.hparams.bidirectional,
            batch_first=True,
        )
        lstm_out = self.hparams.hidden_dim * (2 if self.hparams.bidirectional else 1)
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out,
            num_heads=8,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_out)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim),
        )
        self.loss_fn = nn.MSELoss()
        self.use_maritime_safety = use_maritime_safety

        # Import maritime safety loss if needed
        if self.use_maritime_safety:
            from ...loss.trajectory_loss import maritime_safety_loss

            self.maritime_safety_loss = maritime_safety_loss

    def forward(self, x: torch.Tensor, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_p, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        else:
            out, _ = self.lstm(x)
        out = self.norm(out)
        attn_out, _ = self.attn(out, out, out)
        return self.proj(out + attn_out)

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        preds = self(x, batch.get("lengths"))

        # Use maritime safety loss if enabled and data available
        if self.use_maritime_safety and "neighbors" in batch:
            loss, loss_components = self.maritime_safety_loss(
                preds,
                y,
                neighbor_trajectories=batch.get("neighbors"),
                vessel_specs=batch.get("vessel_specs"),
                collision_weight=self.hparams.collision_weight,
                feasibility_weight=self.hparams.feasibility_weight,
            )
            # Log individual components
            for name, value in loss_components.items():
                self.log(f"train/{name}", value, on_epoch=True)
        else:
            loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        preds = self(x, batch.get("lengths"))

        # Use maritime safety loss if enabled and data available
        if self.use_maritime_safety and "neighbors" in batch:
            loss, loss_components = self.maritime_safety_loss(
                preds,
                y,
                neighbor_trajectories=batch.get("neighbors"),
                vessel_specs=batch.get("vessel_specs"),
                collision_weight=self.hparams.collision_weight,
                feasibility_weight=self.hparams.feasibility_weight,
            )
            # Log individual components
            for name, value in loss_components.items():
                self.log(f"val/{name}", value, on_epoch=True)

            # Log collision risk specifically for monitoring
            if "collision_risk" in loss_components:
                self.log(
                    "val_collision_risk",
                    loss_components["collision_risk"],
                    on_epoch=True,
                    prog_bar=True,
                )
        else:
            loss = self.loss_fn(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", F.l1_loss(preds, y), on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
