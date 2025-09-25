# File: src/models/motion_transformer.py
"""
LightningModule wrapper for the Motion Transformer.
Integrates the MotionTransformer blocks with PyTorch Lightning training logic.
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .blocks.motion_transformer import MotionTransformer


class MotionTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the MotionTransformer model.
    """

    def __init__(self, config: Any = None, **kwargs):
        super().__init__()
        # Handle Hydra config or manual kwargs
        if config is not None:
            self.save_hyperparameters(config)
            cfg = config
        else:
            # Flatten kwargs into hparams
            self.save_hyperparameters(kwargs)

            class SimpleConfig:
                def __init__(self, **k):
                    for key, val in k.items():
                        setattr(self, key, val)

            cfg = SimpleConfig(**kwargs)

        # Instantiate core MotionTransformer
        self.model = MotionTransformer(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            n_queries=cfg.n_queries,
            encoder_layers=cfg.encoder_layers,
            decoder_layers=cfg.decoder_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            prediction_horizon=cfg.prediction_horizon,
            output_dim=cfg.output_dim,
        )
        # Loss type for multi-modal
        self.loss_type = getattr(cfg, "loss_type", "best_of_n")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        return self.model(x, context_mask=mask)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]  # [B, seq_len, input_dim]
        targets = batch["targets"]  # [B, pred_horizon, output_dim]
        outputs = self(inputs)
        loss_dict = self.model.compute_loss(outputs, targets, self.loss_type)
        self.log("train_loss", loss_dict["total_loss"], prog_bar=True)
        return loss_dict["total_loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        outputs = self(inputs)
        loss_dict = self.model.compute_loss(outputs, targets, self.loss_type)
        # Compute ADE/FDE on best trajectory
        best = self.model.predict_best_trajectory(inputs)
        ade = F.mse_loss(best, targets, reduction="none").mean(dim=-1).sqrt().mean()
        fde = F.mse_loss(best[:, -1], targets[:, -1], reduction="none").sqrt().mean()
        self.log("val_loss", loss_dict["total_loss"], prog_bar=True)
        self.log("val_ade", ade)
        self.log("val_fde", fde)
        return loss_dict["total_loss"]

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        outputs = self(inputs)
        loss_dict = self.model.compute_loss(outputs, targets, self.loss_type)
        best = self.model.predict_best_trajectory(inputs)
        ade = F.mse_loss(best, targets, reduction="none").mean(dim=-1).sqrt().mean()
        fde = F.mse_loss(best[:, -1], targets[:, -1], reduction="none").sqrt().mean()
        self.log("test_loss", loss_dict["total_loss"])
        self.log("test_ade", ade)
        self.log("test_fde", fde)
        return loss_dict["total_loss"]

    def configure_optimizers(self) -> dict:
        lr = self.hparams.get("learning_rate", getattr(self.hparams, "lr", 1e-4))
        wd = self.hparams.get(
            "weight_decay", getattr(self.hparams, "weight_decay", 0.0)
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        # OneCycleLR with estimated steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class MotionTransformerTrainer:
    """
    Trainer class for Motion Transformer following existing repository pattern.
    """

    def __init__(
        self,
        model: MotionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        """
        Initialize trainer following existing pattern.

        Args:
            model: MotionTransformer model
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def training_step(self, batch):
        """Training step following existing pattern."""
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        outputs = self.model(inputs)
        loss_dict = self.model.compute_loss(outputs, targets, "best_of_n")

        loss = loss_dict["total_loss"]
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Maritime-specific configurations following existing pattern
MARITIME_MTR_CONFIG = {
    "small": {
        "input_dim": 4,
        "d_model": 128,
        "n_queries": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "prediction_horizon": 10,
        "output_dim": 4,
    },
    "medium": {
        "input_dim": 4,
        "d_model": 256,
        "n_queries": 8,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "n_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "prediction_horizon": 10,
        "output_dim": 4,
    },
    "large": {
        "input_dim": 4,
        "d_model": 512,
        "n_queries": 16,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "n_heads": 16,
        "d_ff": 2048,
        "dropout": 0.1,
        "prediction_horizon": 10,
        "output_dim": 4,
    },
}


def create_motion_transformer(**kwargs) -> MotionTransformer:
    """
    Factory to create core MotionTransformer block (respects existing architecture).
    """
    return MotionTransformer(**kwargs)


def create_maritime_motion_transformer(size: str = "medium") -> MotionTransformer:
    """
    Create maritime-configured Motion Transformer following existing pattern.
    """
    if size not in MARITIME_MTR_CONFIG:
        raise ValueError(
            f"Unknown size: {size}. Available: {list(MARITIME_MTR_CONFIG.keys())}"
        )
    config = MARITIME_MTR_CONFIG[size]
    return MotionTransformer(**config)


def create_motion_transformer_lightning(**kwargs) -> MotionTransformerLightning:
    """
    Factory to create the LightningModule with given parameters.
    """
    return MotionTransformerLightning(None, **kwargs)
