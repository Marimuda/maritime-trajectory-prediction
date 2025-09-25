"""
LightningModule: Autoencoder for anomaly detection.
"""

import pytorch_lightning as pl
import torch
from torch import nn


class AnomalyAutoencoderLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: list[int] = [128, 96],
        dropout: float = 0.2,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        act = act_map[self.hparams.activation]
        # encoder
        layers = []
        prev = self.hparams.input_dim
        for h in self.hparams.hidden_dims:
            layers += [
                nn.Linear(prev, h),
                act,
                nn.Dropout(self.hparams.dropout),
                nn.BatchNorm1d(h),
            ]
            prev = h
        layers.append(nn.Linear(prev, self.hparams.encoding_dim))
        self.encoder = nn.Sequential(*layers)
        # decoder
        layers = []
        prev = self.hparams.encoding_dim
        for h in reversed(self.hparams.hidden_dims):
            layers += [
                nn.Linear(prev, h),
                act,
                nn.Dropout(self.hparams.dropout),
                nn.BatchNorm1d(h),
            ]
            prev = h
        layers.append(nn.Linear(prev, self.hparams.input_dim))
        self.decoder = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        orig = x.shape
        if x.dim() == 3:
            B, T, F = orig
            x_flat = x.view(B * T, F)
        else:
            x_flat = x
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        if x.dim() == 3:
            recon = recon.view(B, T, F)
            z = z.view(B, T, -1).mean(dim=1)
        return recon, z

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        recon, _ = self(x)
        loss = self.loss_fn(recon, x)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        recon, _ = self(x)
        loss = self.loss_fn(recon, x)
        self.log("val_loss", loss, on_epoch=True)
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
