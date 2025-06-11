"""
Enhanced training script for SOTA models integration.

This script provides unified training for both baseline and SOTA models
with proper configuration management, logging, and model selection.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
import yaml
from torch import nn
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models import create_model
from models.anomaly_transformer import AnomalyTransformerTrainer
from models.motion_transformer import MotionTransformerTrainer

# Note: Baseline trainers will be created as simple wrappers
# from data.lightning_datamodule import AISDataModule  # Skip for now due to import issues

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleBaselineTrainer:
    """Simple trainer wrapper for baseline models."""

    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-5, device="cpu"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

    def train_step(self, *args):
        """Simple training step."""
        self.optimizer.zero_grad()

        if len(args) == 1:
            # Anomaly detection or single input
            context = args[0]
            outputs = self.model(context)
            loss = self.criterion(outputs, context)  # Reconstruction loss
        else:
            # Trajectory prediction
            context, targets = args
            outputs = self.model(context)
            loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return {"total_loss": loss.item()}

    def validate_step(self, *args):
        """Simple validation step."""
        with torch.no_grad():
            if len(args) == 1:
                context = args[0]
                outputs = self.model(context)
                loss = self.criterion(outputs, context)
            else:
                context, targets = args
                outputs = self.model(context)
                loss = self.criterion(outputs, targets)

        return {"total_loss": loss.item()}


class SOTATrainingConfig:
    """Configuration management for SOTA model training."""

    DEFAULT_CONFIG = {
        "model": {
            "type": "motion_transformer",  # or 'anomaly_transformer', 'baseline'
            "task": "trajectory_prediction",  # for baseline models
            "size": "medium",  # for maritime configs
            "custom_params": {},
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": 100,
            "patience": 10,
            "gradient_clip": 1.0,
            "loss_type": "best_of_n",  # for motion transformer
            "validation_freq": 1,
        },
        "data": {
            "sequence_length": 30,
            "prediction_horizon": 10,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "num_workers": 4,
            "pin_memory": True,
        },
        "logging": {
            "use_wandb": False,
            "project_name": "maritime-sota",
            "experiment_name": None,
            "log_freq": 10,
            "save_freq": 5,
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        },
    }

    def __init__(self, config_path: str | None = None):
        """Initialize configuration."""
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                self._update_config(self.config, user_config)

    def _update_config(self, base_config: dict, update_config: dict):
        """Recursively update configuration."""
        for key, value in update_config.items():
            if (
                key in base_config
                and isinstance(base_config[key], dict)
                and isinstance(value, dict)
            ):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def save(self, path: str):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)


class SOTATrainer:
    """Unified trainer for SOTA and baseline models."""

    def __init__(self, config: SOTATrainingConfig):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories
        self._setup_directories()

        # Initialize logging
        self._setup_logging()

        # Create model
        self.model = self._create_model()

        # Create trainer
        self.trainer = self._create_trainer()

        # Setup data
        self.data_module = self._setup_data()

        logger.info(
            f"Initialized SOTATrainer with {self.config.get('model.type')} model"
        )
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        logger.info(f"Device: {self.device}")

    def _setup_directories(self):
        """Setup output directories."""
        for dir_key in ["output_dir", "checkpoint_dir", "log_dir"]:
            dir_path = self.config.get(f"paths.{dir_key}")
            os.makedirs(dir_path, exist_ok=True)

    def _setup_logging(self):
        """Setup experiment logging."""
        if self.config.get("logging.use_wandb"):
            experiment_name = self.config.get("logging.experiment_name")
            if not experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_type = self.config.get("model.type")
                experiment_name = f"{model_type}_{timestamp}"

            wandb.init(
                project=self.config.get("logging.project_name"),
                name=experiment_name,
                config=self.config.config,
            )

    def _create_model(self):
        """Create model based on configuration."""
        model_type = self.config.get("model.type")

        if model_type in ["anomaly_transformer", "motion_transformer"]:
            # SOTA models
            custom_params = self.config.get("model.custom_params", {})

            if model_type == "anomaly_transformer":
                from models.anomaly_transformer import (
                    create_maritime_anomaly_transformer,
                )

                size = self.config.get("model.size", "medium")
                model = create_maritime_anomaly_transformer(size)
            else:  # motion_transformer
                from models.motion_transformer import create_motion_transformer, MARITIME_MTR_CONFIG

                size = self.config.get("model.size", "medium")
                config = MARITIME_MTR_CONFIG[size].copy()
                
                # Override with training configuration parameters
                config['prediction_horizon'] = self.config.get("data.prediction_horizon", 10)
                
                # Get input dimension from data if available
                if hasattr(self, 'data_module') and hasattr(self.data_module, 'input_dim'):
                    config['input_dim'] = self.data_module.input_dim
                
                model = create_motion_transformer(**config)

            # Apply custom parameters if provided
            if custom_params:
                logger.info(f"Applying custom parameters: {custom_params}")
                # Note: This would require model reconstruction with new params
                # For now, we log the intent

        elif model_type == "baseline":
            # Baseline models
            task = self.config.get("model.task")
            custom_params = self.config.get("model.custom_params", {})
            model = create_model("baseline", task=task, **custom_params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def _create_trainer(self):
        """Create appropriate trainer for the model."""
        model_type = self.config.get("model.type")
        learning_rate = self.config.get("training.learning_rate")
        weight_decay = self.config.get("training.weight_decay")

        if model_type == "anomaly_transformer":
            return AnomalyTransformerTrainer(
                model=self.model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=self.device,
            )

        elif model_type == "motion_transformer":
            loss_type = self.config.get("training.loss_type", "best_of_n")
            return MotionTransformerTrainer(
                model=self.model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_type=loss_type,
                device=self.device,
            )

        elif model_type == "baseline":
            # Create simple baseline trainer wrapper
            return SimpleBaselineTrainer(
                model=self.model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=self.device,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _setup_data(self):
        """Setup data module."""
        logger.info("Setting up data module for real AIS data")

        from src.data.datamodule import AISDataModule

        data_dir = self.config.get("paths.data_dir", "./data")

        # Find AIS data file in the data directory
        data_path = None

        data_path_obj = Path(data_dir)

        # Look for common AIS data file patterns
        # Look for processed data first, then raw data
        processed_dir = data_path_obj / "processed"
        if processed_dir.exists():
            for pattern in [
                "**/combined_ais_all.parquet",
                "**/combined_ais_all.csv",
                "**/*.parquet",
                "**/*.csv",
            ]:
                files = list(processed_dir.glob(pattern))
                if files:
                    data_path = str(files[0])
                    logger.info(f"Found processed AIS data file: {data_path}")
                    break

        if not data_path:
            # Fallback to raw data
            for pattern in ["**/100k_ais.log", "**/100k_ais", "*.csv", "*.json"]:
                files = list(data_path_obj.glob(pattern))
                if files:
                    data_path = str(files[0])
                    logger.info(f"Found raw AIS data file: {data_path}")
                    break

        if not data_path:
            raise FileNotFoundError(f"No AIS data files found in {data_dir}")

        batch_size = self.config.get("training.batch_size", 32)
        sequence_length = self.config.get("data.sequence_length", 30)
        prediction_horizon = self.config.get("data.prediction_horizon", 10)

        self.data_module = AISDataModule(
            data_path=data_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            num_workers=self.config.get("data.num_workers", 4),
        )

        return self.data_module

    def train(self):
        """Main training loop."""
        max_epochs = self.config.get("training.max_epochs")
        patience = self.config.get("training.patience")
        validation_freq = self.config.get("training.validation_freq")
        log_freq = self.config.get("logging.log_freq")
        save_freq = self.config.get("logging.save_freq")

        # Setup data
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(
            f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}"
        )

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_losses = []

            # Add progress bar for training
            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                            desc=f"Epoch {epoch}/{max_epochs}", leave=False)
            
            for batch_idx, batch in train_pbar:
                # Handle different batch formats for different models
                loss_dict = self._train_step(batch)
                train_losses.append(loss_dict["total_loss"])
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'avg_loss': f"{sum(train_losses)/len(train_losses):.4f}"
                })

                # Logging
                if batch_idx % log_freq == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_dict['total_loss']:.4f}"
                    )

                    if self.config.get("logging.use_wandb"):
                        wandb.log(
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "train_loss": loss_dict["total_loss"],
                                **{
                                    f"train_{k}": v
                                    for k, v in loss_dict.items()
                                    if k != "total_loss"
                                },
                            }
                        )

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Validation phase
            if epoch % validation_freq == 0:
                val_loss = self._validate(val_loader)

                logger.info(
                    f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if self.config.get("logging.use_wandb"):
                    wandb.log(
                        {
                            "epoch": epoch,
                            "avg_train_loss": avg_train_loss,
                            "val_loss": val_loss,
                        }
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Regular checkpoint saving
            if epoch % save_freq == 0:
                self._save_checkpoint(epoch, avg_train_loss)

        logger.info("Training completed")

        if self.config.get("logging.use_wandb"):
            wandb.finish()

    def _train_step(self, batch):
        """Single training step."""
        model_type = self.config.get("model.type")

        # Extract input and target from AIS datamodule format
        input_data = batch["input"].to(self.device)  # [B, T, F]
        target_data = batch["target"].to(self.device)  # [B, T_pred, F]

        if model_type == "anomaly_transformer":
            # Anomaly transformer expects single input for reconstruction
            return self.trainer.train_step(input_data)

        elif model_type == "motion_transformer":
            # Motion transformer expects context and targets
            return self.trainer.train_step(input_data, target_data)

        elif model_type == "baseline":
            # Baseline models - delegate to trainer
            task = self.config.get("model.task")
            if task == "anomaly_detection":
                return self.trainer.train_step(input_data)
            else:  # trajectory_prediction or vessel_interaction
                return self.trainer.train_step(input_data, target_data)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _validate(self, val_loader):
        """Validation phase."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch in val_pbar:
                loss_dict = self._validation_step(batch)
                val_loss = loss_dict.get("total_loss", loss_dict.get("val_total_loss", 0))
                val_losses.append(val_loss)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'val_loss': f"{val_loss:.4f}",
                    'avg_val_loss': f"{sum(val_losses)/len(val_losses):.4f}"
                })

        return sum(val_losses) / len(val_losses)

    def _validation_step(self, batch):
        """Single validation step."""
        model_type = self.config.get("model.type")

        # Extract input and target from AIS datamodule format
        input_data = batch["input"].to(self.device)  # [B, T, F]
        target_data = batch["target"].to(self.device)  # [B, T_pred, F]

        if model_type == "anomaly_transformer":
            return self.trainer.validate_step(input_data)

        elif model_type == "motion_transformer":
            return self.trainer.validate_step(input_data, target_data)

        elif model_type == "baseline":
            task = self.config.get("model.task")
            if task == "anomaly_detection":
                return self.trainer.validate_step(input_data)
            else:
                return self.trainer.validate_step(input_data, target_data)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get("paths.checkpoint_dir")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "loss": loss,
            "config": self.config.config,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch} with loss {loss:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint["epoch"], checkpoint["loss"]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SOTA models for maritime trajectory prediction"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["anomaly_transformer", "motion_transformer", "baseline"],
        help="Model type to train",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["trajectory_prediction", "anomaly_detection", "vessel_interaction"],
        help="Task for baseline models",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size for SOTA models",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load configuration
    config = SOTATrainingConfig(args.config)

    # Override with command line arguments
    if args.model_type:
        config.config["model"]["type"] = args.model_type
    if args.task:
        config.config["model"]["task"] = args.task
    if args.size:
        config.config["model"]["size"] = args.size
    if args.epochs:
        config.config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config.config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config.config["training"]["learning_rate"] = args.learning_rate
    if args.data_dir:
        config.config["paths"]["data_dir"] = args.data_dir
    if args.output_dir:
        config.config["paths"]["output_dir"] = args.output_dir
    if args.use_wandb:
        config.config["logging"]["use_wandb"] = True

    # Create trainer
    trainer = SOTATrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Save configuration
    config_save_path = os.path.join(config.get("paths.output_dir"), "config.yaml")
    config.save(config_save_path)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
