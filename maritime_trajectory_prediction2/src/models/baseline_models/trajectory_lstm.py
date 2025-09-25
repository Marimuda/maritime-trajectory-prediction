"""
LightningModule: Bidirectional LSTM with attention for trajectory prediction.
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
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        preds = self(x, batch.get("lengths"))
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_epoch=True)
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
