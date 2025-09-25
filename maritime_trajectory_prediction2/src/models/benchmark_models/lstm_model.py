"""
LightningModule: LSTM baseline for AIS trajectory prediction.
"""

import pytorch_lightning as pl
import torch
from torch import nn


class LSTMModelLightning(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers > 1 else 0.0,
            batch_first=True,
        )
        # FC
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, tuple) else batch
        return self(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
