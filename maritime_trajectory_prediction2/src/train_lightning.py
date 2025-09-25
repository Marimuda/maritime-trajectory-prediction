"""
Training script with PyTorch Lightning CLI integration.

Implements the guideline's recommendations for experiment management,
reproducibility, and performance optimization.
"""

import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global file paths (following guideline recommendation)
DATA_DIR = Path("./data")
ZARR_PATH = DATA_DIR / "ais_positions.zarr"
STATIC_ZARR_PATH = DATA_DIR / "ais_static.zarr"
CHECKPOINT_DIR = Path("./checkpoints")
LOGS_DIR = Path("./logs")

# Create directories
for dir_path in [DATA_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)


class AISLightningCLI(LightningCLI):
    """
    Custom Lightning CLI for AIS trajectory prediction.

    Implements the guideline's recommendations for experiment configuration,
    callbacks, and performance optimization.
    """

    def add_arguments_to_parser(self, parser):
        """Add custom arguments to the CLI parser."""
        # Data arguments
        parser.add_argument("--zarr_path", type=str, default=str(ZARR_PATH))
        parser.add_argument(
            "--static_zarr_path", type=str, default=str(STATIC_ZARR_PATH)
        )

        # Performance arguments
        parser.add_argument("--enable_tf32", type=bool, default=True)
        parser.add_argument("--compile_model", type=bool, default=False)
        parser.add_argument("--mixed_precision", type=str, default="16-mixed")

        # Experiment tracking
        parser.add_argument("--experiment_name", type=str, default="ais_trajectory")
        parser.add_argument("--use_wandb", type=bool, default=False)
        parser.add_argument("--wandb_project", type=str, default="maritime-trajectory")

    def before_instantiate_classes(self) -> None:
        """Setup before instantiating classes."""
        # Enable performance optimizations following guideline
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for faster training")

        # Set memory allocation strategy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Use memory pool for better allocation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    def configure_callbacks(self):
        """Configure callbacks following guideline recommendations."""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val/loss", patience=10, mode="min", verbose=True, strict=True
        )
        callbacks.append(early_stopping)

        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Device stats monitoring
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

        # Rich progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)

        return callbacks

    def configure_logger(self):
        """Configure experiment loggers."""
        loggers = []

        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=LOGS_DIR,
            name=self.config.experiment_name,
            version=None,  # Auto-increment
            default_hp_metric=False,
        )
        loggers.append(tb_logger)

        # Weights & Biases logger (optional)
        if self.config.use_wandb:
            try:
                wandb_logger = WandbLogger(
                    project=self.config.wandb_project,
                    name=self.config.experiment_name,
                    save_dir=LOGS_DIR,
                )
                loggers.append(wandb_logger)
                logger.info("Enabled Weights & Biases logging")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")

        return loggers

    def configure_trainer(self):
        """Configure trainer with performance optimizations."""
        trainer_config = {
            # Performance optimizations
            "precision": self.config.mixed_precision,
            "accelerator": "auto",
            "devices": "auto",
            "strategy": "auto",
            # Callbacks and logging
            "callbacks": self.configure_callbacks(),
            "logger": self.configure_logger(),
            # Training configuration
            "max_epochs": 100,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            # Validation and checkpointing
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            # Performance monitoring
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "profiler": "simple",
            # Reproducibility
            "deterministic": False,  # Set to True for full reproducibility
            "benchmark": True,  # Enable cudnn benchmarking for performance
        }

        return trainer_config


def setup_environment():
    """Setup environment for optimal performance."""
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)

    # Configure multiprocessing
    if hasattr(torch.multiprocessing, "set_sharing_strategy"):
        torch.multiprocessing.set_sharing_strategy("file_system")

    # Configure CUDA settings
    if torch.cuda.is_available():
        # Enable memory pool
        torch.cuda.empty_cache()

        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)

        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"Current device: {torch.cuda.current_device()}")

    # Configure CPU settings
    torch.set_num_threads(min(8, torch.get_num_threads()))

    logger.info("Environment setup complete")


def main():
    """Main training function."""
    # Setup environment
    setup_environment()

    # Import models and data modules
    from ..data.lightning_datamodule import AISLightningDataModule
    from ..models.lightning_models import ConvolutionalPredictor

    # Create CLI
    cli = AISLightningCLI(
        model_class=ConvolutionalPredictor,
        datamodule_class=AISLightningDataModule,
        save_config_callback=None,  # Disable automatic config saving
        auto_configure_optimizers=False,  # Use model's configure_optimizers
        parser_kwargs={
            "prog": "AIS Trajectory Prediction Training",
            "description": "Train maritime trajectory prediction models with PyTorch Lightning",
        },
    )


if __name__ == "__main__":
    main()
