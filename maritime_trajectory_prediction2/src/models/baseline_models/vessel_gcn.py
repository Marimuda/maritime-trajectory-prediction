"""
LightningModule: Graph Convolutional Network for vessel interaction.
"""

import pytorch_lightning as pl
import torch
from torch import nn


class VesselGCNLightning(pl.LightningModule):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.node_in = nn.Linear(self.hparams.node_features, self.hparams.hidden_dim)
        self.edge_in = nn.Linear(self.hparams.edge_features, self.hparams.hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    self.hparams.hidden_dim * 2 if i else self.hparams.hidden_dim,
                    self.hparams.hidden_dim,
                )
                for i in range(self.hparams.num_layers)
            ]
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.hparams.dropout)
        self.out = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dim // 2, self.hparams.output_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, node_feat, edge_feat, adj):
        x = self.node_in(node_feat)
        e = self.edge_in(edge_feat)
        for layer in self.layers:
            cat = torch.cat([x, e], dim=-1)
            x = self.drop(self.act(layer(cat)))
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, e, adj, y = (
            batch["node_feat"],
            batch["edge_feat"],
            batch["adj"],
            batch["target"],
        )
        preds = self(x, e, adj)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, e, adj, y = (
            batch["node_feat"],
            batch["edge_feat"],
            batch["adj"],
            batch["target"],
        )
        preds = self(x, e, adj)
        loss = self.loss_fn(preds, y)
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
