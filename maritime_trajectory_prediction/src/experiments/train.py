"""
Training experiment module.

Migrates train_lightning.py logic into the unified experiment structure
following the CLAUDE blueprint.
"""

import logging
import os
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from ..models.factory import create_model
from ..training.trainer import create_trainer

logger = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig):
    """
    Setup environment for optimal performance and reproducibility.

    Args:
        cfg: Configuration object
    """
    # Set random seeds
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set random seed to {cfg.seed}")

    # Configure PyTorch performance optimizations
    if cfg.trainer.get("compile", False):
        logger.info("PyTorch compilation enabled")

    if cfg.trainer.get("benchmark", True):
        torch.backends.cudnn.benchmark = True
        logger.info("CUDNN benchmarking enabled")

        # Enable TF32 for faster training on compatible hardware
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for faster training")

    # Set memory allocation strategy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Use memory pool for better allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"Current device: {torch.cuda.current_device()}")

    # Configure CPU settings
    torch.set_num_threads(min(8, torch.get_num_threads()))
    logger.info(f"PyTorch threads: {torch.get_num_threads()}")


def create_data_module(cfg: DictConfig):
    """
    Create Lightning data module from configuration.

    Args:
        cfg: Configuration object

    Returns:
        Configured data module
    """
    # Try to import the data module
    try:
        from ..data.lightning_datamodule import AISLightningDataModule

        datamodule = AISLightningDataModule(
            zarr_path=cfg.data.get("processed_dir", "data/processed")
            + "/ais_positions.zarr",
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            window_size=cfg.data.sequence_length,
            prediction_horizon=cfg.data.prediction_horizon,
            features=cfg.data.feature_columns,
        )

        logger.info("Created Lightning data module")
        logger.info(
            f"  Data directory: {cfg.data.get('processed_dir', 'data/processed')}"
        )
        logger.info(f"  Batch size: {cfg.data.batch_size}")
        logger.info(f"  Sequence length: {cfg.data.sequence_length}")
        logger.info(f"  Prediction horizon: {cfg.data.prediction_horizon}")

        return datamodule

    except ImportError as e:
        logger.error(f"Could not import AISLightningDataModule: {e}")
        logger.info("Creating fallback data module...")

        # Create a simple fallback data module
        from ..data.datamodule import create_simple_datamodule

        return create_simple_datamodule(cfg)


def create_lightning_model(cfg: DictConfig) -> pl.LightningModule:
    """
    Create Lightning model from configuration.

    Args:
        cfg: Configuration object

    Returns:
        Lightning model instance
    """
    # Create the base model using our factory
    model = create_model(cfg.model)

    # Check if it's already a Lightning module
    if isinstance(model, pl.LightningModule):
        return model

    # Wrap in Lightning module if needed
    try:
        from ..models.lightning_models import LightningModelWrapper

        lightning_model = LightningModelWrapper(
            model=model,
            learning_rate=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            optimizer=cfg.trainer.optimizer,
            scheduler=cfg.trainer.get("scheduler"),
            task=cfg.model.task,
        )

        logger.info("Wrapped model in Lightning module")
        return lightning_model

    except ImportError as e:
        logger.error("Could not import LightningModelWrapper")
        raise RuntimeError(
            "Model is not a Lightning module and wrapper not available"
        ) from e


def log_model_info(model: pl.LightningModule, cfg: DictConfig):
    """
    Log model information and configuration.

    Args:
        model: Lightning model
        cfg: Configuration object
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model Information:")
    logger.info(f"  Class: {model.__class__.__name__}")
    logger.info(f"  Type: {cfg.model.type}")
    logger.info(f"  Task: {cfg.model.task}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

    # Log architecture details
    if hasattr(model, "hparams"):
        logger.info(f"  Hyperparameters: {dict(model.hparams)}")


def run_training(cfg: DictConfig) -> dict[str, Any]:
    """
    Main training function called by Hydra dispatch.

    Consolidates the training logic from train_lightning.py into the unified system.

    Args:
        cfg: Hydra configuration object

    Returns:
        Training results dictionary
    """
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # Setup environment for optimal performance
    setup_environment(cfg)

    # Log configuration summary
    logger.info("Training Configuration:")
    logger.info(f"  Model: {cfg.model.type}")
    logger.info(f"  Task: {cfg.model.task}")
    logger.info(f"  Max epochs: {cfg.trainer.max_epochs}")
    logger.info(f"  Learning rate: {cfg.trainer.learning_rate}")
    logger.info(f"  Batch size: {cfg.data.batch_size}")
    logger.info(f"  Precision: {cfg.trainer.precision}")
    logger.info(f"  Accelerator: {cfg.trainer.accelerator}")
    logger.info(f"  Devices: {cfg.trainer.devices}")

    try:
        # Create data module
        logger.info("Setting up data module...")
        datamodule = create_data_module(cfg)

        # Create model
        logger.info("Creating model...")
        model = create_lightning_model(cfg)

        # Log model information
        log_model_info(model, cfg)

        # Create trainer
        logger.info("Setting up trainer...")
        trainer_wrapper = create_trainer(cfg)

        # Setup data module
        logger.info("Preparing data...")
        datamodule.setup()

        # Log data information
        logger.info("Data Information:")
        if hasattr(datamodule, "train_dataset"):
            logger.info(f"  Training samples: {len(datamodule.train_dataset)}")
        if hasattr(datamodule, "val_dataset"):
            logger.info(f"  Validation samples: {len(datamodule.val_dataset)}")
        if hasattr(datamodule, "test_dataset"):
            logger.info(f"  Test samples: {len(datamodule.test_dataset)}")

        # Auto-tune hyperparameters if requested
        if cfg.get("auto_tune", False):
            logger.info("Auto-tuning hyperparameters...")
            tune_results = trainer_wrapper.tune(model, datamodule)
            logger.info(f"Auto-tune results: {tune_results}")

        # Train the model
        logger.info("Starting training...")
        logger.info("=" * 60)

        trainer_wrapper.fit(model, datamodule)

        logger.info("=" * 60)
        logger.info("Training completed!")

        # Test the model if test data is available
        test_results = None
        if hasattr(datamodule, "test_dataloader") and datamodule.test_dataloader():
            logger.info("Running final test evaluation...")
            test_results = trainer_wrapper.test(model, datamodule)
            logger.info(f"Test results: {test_results}")

        # Get training metrics
        trainer = trainer_wrapper.trainer
        training_results = {
            "best_model_path": trainer.checkpoint_callback.best_model_path
            if trainer.checkpoint_callback
            else None,
            "best_model_score": trainer.checkpoint_callback.best_model_score.item()
            if trainer.checkpoint_callback
            and trainer.checkpoint_callback.best_model_score
            else None,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "test_results": test_results,
        }

        # Log final results
        if training_results["best_model_path"]:
            logger.info(f"Best model saved to: {training_results['best_model_path']}")
        if training_results["best_model_score"] is not None:
            logger.info(
                f"Best validation score: {training_results['best_model_score']:.4f}"
            )

        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)

        # Return metric for hyperparameter optimization
        return training_results

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if cfg.get("debug", False):
            logger.exception("Full traceback:")
        raise


# Legacy main function for backward compatibility
def main(config: DictConfig) -> dict[str, Any]:
    """Legacy main function maintaining backward compatibility."""
    return run_training(config)


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="../../configs", config_name="experiment/base")
    def legacy_main(config: DictConfig):
        return main(config)

    legacy_main()
