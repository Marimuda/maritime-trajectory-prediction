#!/usr/bin/env python3
"""
Training scripts for maritime baseline models.

This module provides comprehensive training scripts for:
1. TrajectoryLSTM - Maritime trajectory prediction
2. AnomalyAutoencoder - Vessel behavior anomaly detection
3. VesselGCN - Vessel interaction and collision prediction

Each script includes proper data loading, training loops, validation,
and comprehensive evaluation with maritime-specific metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.multi_task_processor import AISMultiTaskProcessor
from src.data.pipeline import DataPipeline, DatasetConfig, MLTask
from src.models.baseline_models import (
    AnomalyAutoencoder,
    TrajectoryLSTM,
    VesselGCN,
    create_baseline_model,
)
from src.models.metrics import (
    MaritimeLossFunctions,
    create_loss_function,
    create_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineTrainer:
    """
    Base trainer class for maritime baseline models.

    Provides common functionality for training, validation, and evaluation
    across different maritime tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        save_dir: str = "./checkpoints",
    ):
        self.model = model
        self.task = task
        self.device = self._setup_device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup loss function and metrics
        self.loss_function = create_loss_function(task)
        self.metrics = create_metrics(task)

        # Training state
        self.current_epoch = 0
        self.best_metric = float("inf") if task == "trajectory_prediction" else 0.0
        self.training_history = []

        logger.info(
            f"Initialized {task} trainer with {sum(p.numel() for p in model.parameters())} parameters"
        )

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU device")

        return torch.device(device)

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        self.metrics.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = [
                item.to(self.device) if torch.is_tensor(item) else item
                for item in batch
            ]

            # Forward pass
            self.optimizer.zero_grad()
            loss, predictions, targets = self._forward_pass(batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Track metrics
            epoch_losses.append(loss.item())
            if predictions is not None and targets is not None:
                self._update_metrics(predictions, targets)

            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics["loss"] = np.mean(epoch_losses)

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        self.metrics.reset()

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = [
                    item.to(self.device) if torch.is_tensor(item) else item
                    for item in batch
                ]

                # Forward pass
                loss, predictions, targets = self._forward_pass(batch)

                # Track metrics
                epoch_losses.append(loss.item())
                if predictions is not None and targets is not None:
                    self._update_metrics(predictions, targets)

        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics["loss"] = np.mean(epoch_losses)

        return epoch_metrics

    def _forward_pass(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Task-specific forward pass. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _forward_pass")

    def _update_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with predictions and targets."""
        self.metrics.update(predictions, targets)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_best: bool = True,
    ) -> dict[str, list[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_best: Whether to save best model

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate_epoch(val_loader)

            # Track history
            epoch_history = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "time": time.time() - start_time,
            }
            self.training_history.append(epoch_history)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Check for improvement
            current_metric = self._get_primary_metric(val_metrics)
            is_better = self._is_better_metric(current_metric, self.best_metric)

            if is_better:
                self.best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0

                if save_best:
                    self.save_checkpoint("best_model.pth")
                    logger.info(
                        f"  New best model saved (metric: {current_metric:.4f})"
                    )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break

        # Load best model if saved
        if save_best and os.path.exists(self.save_dir / "best_model.pth"):
            self.load_checkpoint("best_model.pth")
            logger.info("Loaded best model for final evaluation")

        return self._format_training_history()

    def _get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Get primary metric for model selection. Override in subclasses."""
        return metrics["loss"]

    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best. Override in subclasses."""
        return current < best  # Default: lower is better

    def _format_training_history(self) -> dict[str, list[float]]:
        """Format training history for analysis."""
        history = {"train_loss": [], "val_loss": [], "epochs": [], "times": []}

        for epoch_data in self.training_history:
            history["epochs"].append(epoch_data["epoch"])
            history["train_loss"].append(epoch_data["train"]["loss"])
            history["val_loss"].append(epoch_data["val"]["loss"])
            history["times"].append(epoch_data["time"])

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.training_history = checkpoint["training_history"]


class TrajectoryTrainer(BaselineTrainer):
    """Trainer for trajectory prediction models."""

    def __init__(self, model: TrajectoryLSTM, **kwargs):
        super().__init__(model, "trajectory_prediction", **kwargs)

    def _forward_pass(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for trajectory prediction."""
        inputs, targets = batch[0], batch[1]

        # Model prediction
        predictions = self.model(inputs)

        # Compute loss
        loss = MaritimeLossFunctions.trajectory_loss(predictions, targets)

        return loss, predictions, targets

    def _get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Use ADE as primary metric for trajectory prediction."""
        return metrics.get("ade_km", metrics["loss"])


class AnomalyTrainer(BaselineTrainer):
    """Trainer for anomaly detection models."""

    def __init__(self, model: AnomalyAutoencoder, **kwargs):
        super().__init__(model, "anomaly_detection", **kwargs)

    def _forward_pass(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for anomaly detection."""
        inputs = batch[0]
        labels = batch[1] if len(batch) > 1 else None

        # Model prediction
        reconstructions, encodings = self.model(inputs)

        # Compute loss
        loss = MaritimeLossFunctions.anomaly_loss(reconstructions, inputs, labels)

        # Compute anomaly scores for metrics
        anomaly_scores = self.model.compute_anomaly_score(inputs)

        return loss, anomaly_scores, labels

    def _get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Use F1 score as primary metric for anomaly detection."""
        return metrics.get("f1_score", 0.0)

    def _is_better_metric(self, current: float, best: float) -> bool:
        """Higher F1 score is better."""
        return current > best


class InteractionTrainer(BaselineTrainer):
    """Trainer for vessel interaction models."""

    def __init__(self, model: VesselGCN, **kwargs):
        super().__init__(model, "vessel_interaction", **kwargs)

    def _forward_pass(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for vessel interaction."""
        node_features, edge_features, adjacency, collision_labels = batch

        # Model prediction
        predictions = self.model(node_features, edge_features, adjacency)

        # Extract collision predictions (assuming last dimension is collision risk)
        collision_predictions = predictions[:, :, -1].flatten()
        collision_targets = collision_labels.flatten()

        # Compute loss
        loss = MaritimeLossFunctions.interaction_loss(
            collision_predictions, collision_targets
        )

        return loss, collision_predictions, collision_targets

    def _get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Use collision F1 score as primary metric."""
        return metrics.get("collision_f1", 0.0)

    def _is_better_metric(self, current: float, best: float) -> bool:
        """Higher F1 score is better."""
        return current > best


def create_trainer(task: str, model: nn.Module, **kwargs) -> BaselineTrainer:
    """
    Factory function to create appropriate trainer for different tasks.

    Args:
        task: Task name ('trajectory_prediction', 'anomaly_detection', 'vessel_interaction')
        model: Model to train
        **kwargs: Trainer-specific parameters

    Returns:
        Initialized trainer
    """
    if task == "trajectory_prediction":
        return TrajectoryTrainer(model, **kwargs)
    elif task == "anomaly_detection":
        return AnomalyTrainer(model, **kwargs)
    elif task == "vessel_interaction":
        return InteractionTrainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


def load_and_prepare_data(
    data_path: str,
    task: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare data for training.

    Args:
        data_path: Path to data file
        task: Task name
        batch_size: Batch size for data loaders
        val_split: Validation split ratio
        test_split: Test split ratio

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Loading data for {task} from {data_path}")

    # Initialize data pipeline
    processor = AISMultiTaskProcessor([task])
    pipeline = DataPipeline()

    # Load and process data
    df = processor.process_file(data_path)

    # Create dataset configuration
    config = DatasetConfig(
        task=MLTask(task.upper()),
        sequence_length=10,
        prediction_horizon=5,
        min_trajectory_length=15,
        validation_split=val_split,
        test_split=test_split,
        random_seed=42,
    )

    # Build dataset
    dataset = pipeline.build_dataset(df, MLTask(task.upper()), config)

    # Create data loaders
    train_data = TensorDataset(
        torch.FloatTensor(dataset["train"]["X"]),
        torch.FloatTensor(dataset["train"]["y"]),
    )
    val_data = TensorDataset(
        torch.FloatTensor(dataset["validation"]["X"]),
        torch.FloatTensor(dataset["validation"]["y"]),
    )
    test_data = TensorDataset(
        torch.FloatTensor(dataset["test"]["X"]), torch.FloatTensor(dataset["test"]["y"])
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    logger.info(
        f"Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples"
    )

    return train_loader, val_loader, test_loader


def train_baseline_model(
    task: str,
    data_path: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: str = "./results",
) -> dict[str, Any]:
    """
    Train a baseline model for a specific task.

    Args:
        task: Task name
        data_path: Path to training data
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Output directory for results

    Returns:
        Training results and metrics
    """
    logger.info(f"Training {task} baseline model...")

    # Create output directory
    output_path = Path(output_dir) / task
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_and_prepare_data(
        data_path, task, **training_config.get("data", {})
    )

    # Create model
    model = create_baseline_model(task, **model_config)

    # Create trainer
    trainer = create_trainer(
        task,
        model,
        save_dir=output_path / "checkpoints",
        **training_config.get("trainer", {}),
    )

    # Train model
    history = trainer.train(
        train_loader, val_loader, **training_config.get("training", {})
    )

    # Evaluate on test set
    test_metrics = trainer.validate_epoch(test_loader)

    # Save results
    results = {
        "task": task,
        "model_config": model_config,
        "training_config": training_config,
        "training_history": history,
        "test_metrics": test_metrics,
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }

    # Save results to file
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Training completed. Results saved to {output_path}")
    logger.info(f"Test metrics: {test_metrics}")

    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train maritime baseline models")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["trajectory_prediction", "anomaly_detection", "vessel_interaction"],
        help="Task to train",
    )
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Default configurations
    model_configs = {
        "trajectory_prediction": {
            "input_dim": 4,
            "hidden_dim": 128,
            "num_layers": 2,
            "output_dim": 4,
        },
        "anomaly_detection": {
            "input_dim": 4,
            "encoding_dim": 64,
            "hidden_dims": [128, 96],
        },
        "vessel_interaction": {
            "node_features": 10,
            "edge_features": 5,
            "hidden_dim": 128,
            "num_layers": 3,
        },
    }

    training_config = {
        "data": {"batch_size": args.batch_size, "val_split": 0.2, "test_split": 0.1},
        "trainer": {
            "learning_rate": args.lr,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
        },
        "training": {"num_epochs": args.epochs, "patience": 10, "save_best": True},
    }

    # Load custom config if provided
    if args.config:
        with open(args.config) as f:
            custom_config = json.load(f)
            model_configs[args.task].update(custom_config.get("model", {}))
            training_config.update(custom_config.get("training", {}))

    # Train model
    results = train_baseline_model(
        args.task, args.data, model_configs[args.task], training_config, args.output
    )

    print("Training completed successfully!")
    print(f"Results saved to: {args.output}/{args.task}")


if __name__ == "__main__":
    main()
