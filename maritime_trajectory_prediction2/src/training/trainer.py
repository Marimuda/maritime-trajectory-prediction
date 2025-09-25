"""
Training module wrapping PyTorch Lightning trainer.

Provides a unified training interface following the CLAUDE blueprint,
consolidating training logic from multiple scripts.
"""

import logging
import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

logger = logging.getLogger(__name__)


class LightningTrainerWrapper:
    """
    Wrapper around PyTorch Lightning trainer with configuration-driven setup.

    Consolidates training logic and provides performance optimizations
    following the blueprint recommendations.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the trainer wrapper.

        Args:
            config: Training configuration
        """
        self.config = config
        self.trainer_config = config.trainer
        self.logger_config = config.logger
        self.callbacks_config = config.callbacks

        # Setup environment optimizations
        self._setup_environment()

        # Create trainer components
        self.callbacks = self._create_callbacks()
        self.loggers = self._create_loggers()

        # Create PyTorch Lightning trainer
        self.trainer = self._create_trainer()

        logger.info(
            f"Trainer initialized with accelerator: {self.trainer_config.accelerator}"
        )

    def _setup_environment(self):
        """Setup environment for optimal performance."""
        # Enable performance optimizations
        if self.trainer_config.get("compile", False):
            trainer = pl.Trainer(compile_model=True)

        if self.trainer_config.get("benchmark", True):
            torch.backends.cudnn.benchmark = True
            logger.info("CUDNN benchmarking enabled")

        # Configure CUDA settings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Use memory pool for better allocation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")

        # Configure CPU settings
        torch.set_num_threads(min(8, torch.get_num_threads()))

    def _create_callbacks(self) -> list[pl.Callback]:
        """Create training callbacks based on configuration."""
        callbacks = []

        # Early stopping
        if self.callbacks_config.early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.callbacks_config.monitor,
                patience=self.callbacks_config.patience,
                mode=self.callbacks_config.mode,
                verbose=True,
                strict=True,
            )
            callbacks.append(early_stopping)
            logger.info(
                f"Early stopping enabled: monitor={self.callbacks_config.monitor}, patience={self.callbacks_config.patience}"
            )

        # Model checkpointing
        checkpoint_dir = Path(self.callbacks_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val/loss:.4f}",
            monitor=self.callbacks_config.monitor,
            mode=self.callbacks_config.mode,
            save_top_k=self.callbacks_config.save_top_k,
            save_last=self.callbacks_config.save_last,
            verbose=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            f"Model checkpointing enabled: save_top_k={self.callbacks_config.save_top_k}"
        )

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(
            logging_interval=self.callbacks_config.lr_logging_interval
        )
        callbacks.append(lr_monitor)

        # Device stats monitoring
        if self.callbacks_config.device_stats:
            device_stats = DeviceStatsMonitor()
            callbacks.append(device_stats)

        # Rich progress bar
        if self.callbacks_config.rich_progress:
            progress_bar = RichProgressBar()
            callbacks.append(progress_bar)

        return callbacks

    def _create_loggers(self) -> list[pl.loggers.Logger]:
        """Create experiment loggers based on configuration."""
        loggers = []

        # Create logs directory
        logs_dir = Path(self.logger_config.save_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard logger
        if self.logger_config.type == "tensorboard":
            tb_logger = TensorBoardLogger(
                save_dir=logs_dir,
                name=self.logger_config.name,
                version=self.logger_config.version,
                default_hp_metric=getattr(
                    self.logger_config, "tb_default_hp_metric", False
                ),
            )
            loggers.append(tb_logger)
            logger.info(
                f"TensorBoard logging enabled: {logs_dir / self.logger_config.name}"
            )

        # Weights & Biases logger
        elif self.logger_config.type == "wandb":
            try:
                wandb_logger = WandbLogger(
                    project=self.logger_config.wandb_project,
                    name=self.logger_config.name,
                    save_dir=logs_dir,
                    tags=getattr(self.logger_config, "wandb_tags", []),
                )
                loggers.append(wandb_logger)
                logger.info(
                    f"Weights & Biases logging enabled: project={self.logger_config.wandb_project}"
                )
            except ImportError:
                logger.warning("wandb not installed, falling back to TensorBoard")
                tb_logger = TensorBoardLogger(
                    save_dir=logs_dir,
                    name=self.logger_config.name,
                    version=self.logger_config.version,
                )
                loggers.append(tb_logger)

        return loggers

    def _create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer with configuration."""
        trainer_kwargs = {
            # Performance settings
            "accelerator": self.trainer_config.accelerator,
            "devices": self.trainer_config.devices,
            "precision": self.trainer_config.precision,
            # Training parameters
            "max_epochs": self.trainer_config.max_epochs,
            "gradient_clip_val": self.trainer_config.gradient_clip_val,
            "gradient_clip_algorithm": "norm",
            # Callbacks and logging
            "callbacks": self.callbacks,
            "logger": self.loggers,
            # Validation and checkpointing
            "check_val_every_n_epoch": self.trainer_config.check_val_every_n_epoch,
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            # Performance monitoring
            "enable_progress_bar": True,
            "enable_model_summary": True,
            # Reproducibility and performance
            "deterministic": self.trainer_config.deterministic,
            "benchmark": self.trainer_config.benchmark,
        }

        # Add optional validation check interval
        if (
            hasattr(self.trainer_config, "val_check_interval")
            and self.trainer_config.val_check_interval
        ):
            trainer_kwargs["val_check_interval"] = (
                self.trainer_config.val_check_interval
            )

        return pl.Trainer(**trainer_kwargs)

    def fit(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> None:
        """
        Train the model.

        Args:
            model: Lightning module to train
            datamodule: Lightning data module
        """
        logger.info("Starting training...")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Max epochs: {self.trainer_config.max_epochs}")
        logger.info(f"Precision: {self.trainer_config.precision}")
        logger.info(f"Accelerator: {self.trainer_config.accelerator}")
        logger.info(f"Devices: {self.trainer_config.devices}")

        self.trainer.fit(model, datamodule)

        logger.info("Training completed!")

        # Log best model path if available
        if hasattr(self.trainer.checkpoint_callback, "best_model_path"):
            best_path = self.trainer.checkpoint_callback.best_model_path
            logger.info(f"Best model saved to: {best_path}")

    def test(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> list[dict[str, Any]]:
        """
        Test the model.

        Args:
            model: Lightning module to test
            datamodule: Lightning data module

        Returns:
            Test results
        """
        logger.info("Starting testing...")
        results = self.trainer.test(model, datamodule)
        logger.info("Testing completed!")
        return results

    def validate(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> list[dict[str, Any]]:
        """
        Validate the model.

        Args:
            model: Lightning module to validate
            datamodule: Lightning data module

        Returns:
            Validation results
        """
        logger.info("Starting validation...")
        results = self.trainer.validate(model, datamodule)
        logger.info("Validation completed!")
        return results

    def predict(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> list[Any]:
        """
        Generate predictions.

        Args:
            model: Lightning module for prediction
            datamodule: Lightning data module

        Returns:
            Prediction results
        """
        logger.info("Starting prediction...")
        results = self.trainer.predict(model, datamodule)
        logger.info("Prediction completed!")
        return results

    def tune(
        self, model: pl.LightningModule, datamodule: pl.LightningDataModule
    ) -> dict[str, Any]:
        """
        Auto-tune model hyperparameters.

        Args:
            model: Lightning module to tune
            datamodule: Lightning data module

        Returns:
            Tuning results
        """
        logger.info("Starting hyperparameter tuning...")

        # Learning rate finder
        lr_finder = self.trainer.tuner.lr_find(model, datamodule)

        # Batch size finder
        batch_size_finder = self.trainer.tuner.scale_batch_size(model, datamodule)

        results = {
            "optimal_lr": lr_finder.suggestion(),
            "optimal_batch_size": batch_size_finder,
        }

        logger.info(f"Tuning results: {results}")
        return results


def create_trainer(config: DictConfig) -> LightningTrainerWrapper:
    """
    Factory function to create a trainer wrapper.

    Args:
        config: Configuration object

    Returns:
        Configured trainer wrapper
    """
    return LightningTrainerWrapper(config)


class BaselineTrainer:
    """
    Simple trainer for baseline models that don't use PyTorch Lightning.

    Provides a consistent interface for non-Lightning models.
    """

    def __init__(self, model: torch.nn.Module, config: DictConfig):
        """
        Initialize baseline trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.trainer_config = config.trainer

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup loss function
        self.criterion = self._create_criterion()

        logger.info(f"Baseline trainer initialized on device: {self.device}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.trainer_config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.trainer_config.learning_rate,
                weight_decay=self.trainer_config.weight_decay,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.trainer_config.learning_rate,
                weight_decay=self.trainer_config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler based on configuration."""
        scheduler_name = self.trainer_config.get("scheduler")

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.trainer_config.max_epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.trainer_config.max_epochs // 3, gamma=0.1
            )
        else:
            return None

    def _create_criterion(self) -> torch.nn.Module:
        """Create loss function based on task."""
        task = self.config.model.task

        if task == "trajectory_prediction":
            return torch.nn.MSELoss()
        elif task == "anomaly_detection":
            return torch.nn.BCEWithLogitsLoss()
        elif task == "vessel_interaction":
            return torch.nn.CrossEntropyLoss()
        else:
            return torch.nn.MSELoss()  # Default

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, list | tuple):
                batch_items = [
                    b.to(self.device) if torch.is_tensor(b) else b for b in batch
                ]
                inputs, targets = batch_items[0], batch_items[1]
            else:
                inputs = batch.to(self.device)
                targets = inputs  # For unsupervised tasks

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.trainer_config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.gradient_clip_val
                )

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        return {"loss": avg_loss}

    def validate_epoch(self, dataloader) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, list | tuple):
                    batch_items = [
                        b.to(self.device) if torch.is_tensor(b) else b for b in batch
                    ]
                    inputs, targets = batch_items[0], batch_items[1]
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # For unsupervised tasks

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    def fit(self, train_loader, val_loader=None) -> dict[str, list[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.trainer_config.max_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])

            # Validation
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                history["val_loss"].append(val_metrics["loss"])

                logger.info(
                    f"Epoch {epoch+1}/{self.trainer_config.max_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.trainer_config.max_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}"
                )

        return history
