#!/usr/bin/env python3
"""
Comprehensive baseline training script for maritime trajectory prediction.

Trains all baseline models with consistent experimental setup for scientific evaluation.
Includes classical ML, deep learning, and physics-based approaches.
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import AISDataModule

# from src.metrics.interaction_metrics import InteractionMetrics  # Not available yet
from src.metrics.operational.ops_metrics import OperationalMetrics
from src.metrics.trajectory_metrics import TrajectoryMetrics
from src.models.anomaly_transformer import create_maritime_anomaly_transformer
from src.models.baseline_models import (
    AnomalyAutoencoder,
    TrajectoryLSTM,
    VesselGCN,
    create_ct_lightning,
    create_cv_lightning,
    create_imm_lightning,
)
from src.models.motion_transformer import create_maritime_motion_transformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BaselineExperimentConfig:
    """Configuration for baseline experiments."""

    def __init__(self, args):
        self.data_path = Path(args.data_path)
        self.output_dir = Path(args.output_dir)
        self.experiment_name = args.experiment_name
        self.seed = args.seed
        self.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.num_workers = args.num_workers
        self.use_maritime_safety = args.use_maritime_safety

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.experiment_name
        self.results_dir = self.output_dir / "results" / self.experiment_name
        self.logs_dir = self.output_dir / "logs" / self.experiment_name

        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.save_config()

    def save_config(self):
        """Save experiment configuration."""
        config_dict = {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "timestamp": datetime.now().isoformat(),
            "use_maritime_safety": self.use_maritime_safety,
        }

        with open(self.results_dir / "experiment_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)


class TransformerBatchWrapper:
    """Wrapper that transforms batch keys for transformer models."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            # Transform keys from 'input'/'target' to 'inputs'/'targets'
            transformed_batch = {
                "inputs": batch["input"],
                "targets": batch["target"],
            }
            # Keep other keys as well
            for key in batch:
                if key not in ["input", "target"]:
                    transformed_batch[key] = batch[key]
            yield transformed_batch

    def __len__(self):
        return len(self.dataloader)


class MockMaritimeDataset(Dataset):
    """Mock dataset for testing when real data is not available."""

    def __init__(self, n_samples=1000, seq_len=50, n_features=4, n_neighbors=5):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_neighbors = n_neighbors

        # Generate synthetic maritime data
        np.random.seed(42)
        self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic maritime trajectory data."""
        data = []
        for _ in range(self.n_samples):
            # Generate smooth trajectory using cumulative sum
            trajectory = np.zeros((self.seq_len, self.n_features))

            # Latitude and longitude (small changes)
            trajectory[:, 0] = np.cumsum(np.random.randn(self.seq_len) * 0.001)
            trajectory[:, 1] = np.cumsum(np.random.randn(self.seq_len) * 0.001)

            # Speed (10-20 knots with some variation)
            trajectory[:, 2] = 15 + np.random.randn(self.seq_len) * 2
            trajectory[:, 2] = np.clip(trajectory[:, 2], 5, 25)

            # Course (0-360 degrees)
            trajectory[:, 3] = np.cumsum(np.random.randn(self.seq_len) * 5) % 360

            # Neighbors
            neighbors = np.random.randn(self.n_neighbors, self.seq_len, self.n_features)

            # Vessel specs [length, beam, max_turn_rate, max_accel]
            vessel_specs = np.array([200.0, 30.0, 5.0, 0.5])

            data.append(
                {
                    "input": trajectory[:-1],
                    "target": trajectory[1:],
                    "neighbors": neighbors[:, 1:],
                    "vessel_specs": vessel_specs,
                    "lengths": self.seq_len - 1,
                }
            )

        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input": torch.FloatTensor(sample["input"]),
            "target": torch.FloatTensor(sample["target"]),
            "neighbors": torch.FloatTensor(sample["neighbors"]),
            "vessel_specs": torch.FloatTensor(sample["vessel_specs"]),
            "lengths": sample["lengths"],
        }


class BaselineTrainer:
    """Trainer for all baseline models."""

    def __init__(self, config: BaselineExperimentConfig):
        self.config = config
        self.results = {}

        # Set random seeds for reproducibility
        pl.seed_everything(config.seed)

        # Initialize metrics
        self.traj_metrics = TrajectoryMetrics()
        # self.interaction_metrics = InteractionMetrics()  # Not available yet
        self.operational_metrics = OperationalMetrics()

    def get_baseline_models(self) -> dict[str, Any]:
        """Get all baseline models to train."""
        models = {}

        # 1. Classical Baselines
        # logger.info("Initializing classical baseline models...")
        # Note: Classical models require scikit-learn based implementations
        # Commented out for now as they're not in the current codebase

        # # Random Forest
        # models['random_forest'] = {
        #     'model': RandomForestTrajectory(
        #         n_estimators=100,
        #         max_depth=10,
        #         random_state=self.config.seed
        #     ),
        #     'type': 'classical'
        # }

        # # Support Vector Regression
        # models['svr'] = {
        #     'model': SVRTrajectory(
        #         kernel='rbf',
        #         gamma='scale',
        #         epsilon=0.1
        #     ),
        #     'type': 'classical'
        # }

        # 2. Physics-based Baselines
        logger.info("Initializing physics-based models...")

        # Kalman Filter variants
        # Note: prediction_horizon must match data module (10 steps ahead)
        # Kalman filters don't use learning_rate, dim_x, or dim_z parameters
        # State dimensions are determined by the motion model type
        models["kalman_cv"] = {
            "model": create_cv_lightning(prediction_horizon=10),
            "type": "lightning",
        }

        models["kalman_ct"] = {
            "model": create_ct_lightning(prediction_horizon=10),
            "type": "lightning",
        }

        models["kalman_imm"] = {
            "model": create_imm_lightning(prediction_horizon=10),
            "type": "lightning",
        }

        # 3. Deep Learning Baselines
        logger.info("Initializing deep learning models...")

        # LSTM with attention
        models["lstm_attention"] = {
            "model": TrajectoryLSTM(
                input_dim=4,
                hidden_dim=128,
                num_layers=2,
                output_dim=4,
                dropout=0.2,
                bidirectional=True,
                use_maritime_safety=self.config.use_maritime_safety,
                collision_weight=10.0,
                feasibility_weight=5.0,
            ),
            "type": "lightning",
        }

        # Standard LSTM (unidirectional, no attention)
        models["lstm_basic"] = {
            "model": TrajectoryLSTM(
                input_dim=4,
                hidden_dim=64,
                num_layers=2,
                output_dim=4,
                dropout=0.2,
                bidirectional=False,
                use_maritime_safety=False,  # Basic version without safety
            ),
            "type": "lightning",
        }

        # Graph Neural Network
        models["vessel_gcn"] = {
            "model": VesselGCN(
                node_features=4,
                edge_features=4,
                hidden_dim=64,
                output_dim=4,
                num_layers=3,
                dropout=0.2,
            ),
            "type": "lightning",
        }

        # Anomaly Detection Autoencoder
        models["autoencoder"] = {
            "model": AnomalyAutoencoder(
                input_dim=4, encoding_dim=16, hidden_dims=[64, 32], dropout=0.2
            ),
            "type": "lightning",
        }

        # 4. Advanced Transformer Models
        logger.info("Initializing transformer models...")

        # Motion Transformer (multi-modal trajectory prediction)
        models["motion_transformer"] = {
            "model": create_maritime_motion_transformer("small"),
            "type": "lightning",
            "requires_transformer_batch": True,
        }

        # Anomaly Transformer
        models["anomaly_transformer"] = {
            "model": create_maritime_anomaly_transformer("small"),
            "type": "lightning",
            "requires_transformer_batch": True,
        }

        return models

    def get_data_loaders(self):
        """Get data loaders for training and validation."""
        data_path = self.config.data_path

        if data_path.exists():
            logger.info(f"Loading real data from {data_path}")
            logger.info(f"File size: {data_path.stat().st_size / (1024**2):.1f} MB")

            # Use comprehensive AISDataModule with automatic preprocessing
            datamodule = AISDataModule(
                data_path=str(data_path),
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                sequence_length=50,
                prediction_horizon=10,
                auto_preprocess=True,  # Automatically preprocess raw logs if needed
            )

            # Preprocessing stage (raw → Zarr if needed)
            logger.info("Running prepare_data() - preprocessing if needed...")
            datamodule.prepare_data()

            # Setup stage (load data, create sequences, split)
            logger.info("Running setup() - loading and creating sequences...")
            datamodule.setup("fit")

            logger.info("✓ Data loaded successfully")
            logger.info(f"  - Train samples: {len(datamodule.train_dataset)}")
            logger.info(f"  - Val samples: {len(datamodule.val_dataset)}")
            logger.info(f"  - Input dimension: {datamodule.input_dim}")

            return datamodule.train_dataloader(), datamodule.val_dataloader()
        else:
            logger.warning(f"Data path {data_path} not found. Using synthetic data.")
            logger.warning("To use real data, set: --data_path path/to/your/data.log")

            # Use mock data
            train_dataset = MockMaritimeDataset(n_samples=1000)
            val_dataset = MockMaritimeDataset(n_samples=200)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

            return train_loader, val_loader

    def train_lightning_model(self, model, model_name: str, train_loader, val_loader):
        """Train a PyTorch Lightning model."""
        logger.info(f"Training {model_name}...")

        # Check if this is a Kalman filter model
        is_kalman = "kalman" in model_name.lower()

        # Callbacks
        callbacks = []

        # Checkpoint callback - monitors val_loss for all models
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir / model_name,
            filename="{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping - only for gradient-based models
        # Kalman filters don't improve over epochs (fitting happens immediately)
        if not is_kalman:
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, mode="min", verbose=True
            )
            callbacks.append(early_stopping)

        # Learning rate monitor - only for gradient-based models
        # Kalman filters have no optimizer, so no learning rate to monitor
        if not is_kalman:
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            callbacks.append(lr_monitor)

        # Loggers
        tb_logger = TensorBoardLogger(
            save_dir=self.config.logs_dir, name=model_name, version=None
        )

        csv_logger = CSVLogger(
            save_dir=self.config.logs_dir, name=model_name, version=None
        )

        # Trainer - Kalman models only need 1 epoch (auto-fit on first batch)
        # Neural models need full training
        max_epochs = 1 if is_kalman else self.config.max_epochs

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if self.config.use_gpu else "cpu",
            devices=1,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            deterministic=True,
            log_every_n_steps=10,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Get best checkpoint
        best_model_path = checkpoint_callback.best_model_path

        # Load best model for evaluation
        model = model.__class__.load_from_checkpoint(best_model_path)

        # Evaluate
        results = trainer.validate(model, val_loader)

        return model, results[0] if results else {}

    def train_classical_model(self, model, model_name: str, train_loader, val_loader):
        """Train a classical ML model."""
        logger.info(f"Training {model_name}...")

        # Convert data to numpy arrays
        X_train, y_train = [], []
        for batch in train_loader:
            X_train.append(batch["input"].numpy())
            y_train.append(batch["target"].numpy())

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Reshape for classical models (flatten sequences)
        n_samples, seq_len, n_features = X_train.shape
        X_train = X_train.reshape(n_samples, -1)
        y_train = y_train.reshape(n_samples, -1)

        # Train
        model.fit(X_train, y_train)

        # Validate
        X_val, y_val = [], []
        for batch in val_loader:
            X_val.append(batch["input"].numpy())
            y_val.append(batch["target"].numpy())

        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        X_val = X_val.reshape(X_val.shape[0], -1)
        y_val = y_val.reshape(y_val.shape[0], -1)

        # Predict and evaluate
        y_pred = model.predict(X_val)

        # Compute metrics
        mse = np.mean((y_pred - y_val) ** 2)
        mae = np.mean(np.abs(y_pred - y_val))

        results = {"val_loss": float(mse), "val_mae": float(mae)}

        # Save model
        import joblib

        model_path = self.config.checkpoint_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} to {model_path}")

        return model, results

    def train_all_baselines(self):
        """Train all baseline models."""
        logger.info("=" * 60)
        logger.info("Starting comprehensive baseline training")
        logger.info("=" * 60)

        # Get data loaders
        train_loader, val_loader = self.get_data_loaders()

        # Get all models
        models = self.get_baseline_models()

        # Train each model
        results_summary = {}

        for model_name, model_info in models.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*40}")

            try:
                # Use wrapped dataloaders for transformer models
                if model_info.get("requires_transformer_batch", False):
                    train_loader_wrapped = TransformerBatchWrapper(train_loader)
                    val_loader_wrapped = TransformerBatchWrapper(val_loader)
                else:
                    train_loader_wrapped = train_loader
                    val_loader_wrapped = val_loader

                if model_info["type"] == "lightning":
                    model, results = self.train_lightning_model(
                        model_info["model"],
                        model_name,
                        train_loader_wrapped,
                        val_loader_wrapped,
                    )
                elif model_info["type"] == "classical":
                    model, results = self.train_classical_model(
                        model_info["model"],
                        model_name,
                        train_loader_wrapped,
                        val_loader_wrapped,
                    )
                else:
                    logger.error(f"Unknown model type: {model_info['type']}")
                    continue

                results_summary[model_name] = results
                val_loss = results.get("val_loss", "N/A")
                if isinstance(val_loss, float):
                    logger.info(
                        f"✓ {model_name} training complete. Val loss: {val_loss:.4f}"
                    )
                else:
                    logger.info(
                        f"✓ {model_name} training complete. Val loss: {val_loss}"
                    )

            except Exception as e:
                logger.error(f"✗ Failed to train {model_name}: {str(e)}")
                results_summary[model_name] = {
                    "val_loss": "Failed",
                    "val_mae": "Failed",
                    "error": str(e),
                }
                continue

        # Save results summary
        self.save_results_summary(results_summary)

        return results_summary

    def save_results_summary(self, results):
        """Save comprehensive results summary."""
        # Create results DataFrame
        results_df = pd.DataFrame(results).T

        # Save as CSV
        csv_path = self.config.results_dir / "baseline_results.csv"
        results_df.to_csv(csv_path)
        logger.info(f"Saved results to {csv_path}")

        # Save as JSON with additional metadata
        results_json = {
            "experiment": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
                "seed": self.config.seed,
                "use_maritime_safety": self.config.use_maritime_safety,
            },
            "results": results,
        }

        json_path = self.config.results_dir / "baseline_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)

        # Print summary table
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE TRAINING SUMMARY")
        logger.info("=" * 60)

        # Sort by val_loss
        valid_results = {k: v for k, v in results.items() if "val_loss" in v}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]["val_loss"])

        logger.info(f"{'Model':<25} {'Val Loss':<12} {'Val MAE':<12}")
        logger.info("-" * 50)

        for model_name, metrics in sorted_results:
            val_loss = metrics.get("val_loss", "N/A")
            val_mae = metrics.get("val_mae", "N/A")

            if isinstance(val_loss, float) and isinstance(val_mae, float):
                logger.info(f"{model_name:<25} {val_loss:<12.6f} {val_mae:<12.6f}")
            else:
                logger.info(f"{model_name:<25} {str(val_loss):<12} {str(val_mae):<12}")

        logger.info("=" * 60)

        # Report best model
        if sorted_results:
            best_model = sorted_results[0][0]
            best_loss = sorted_results[0][1]["val_loss"]
            if isinstance(best_loss, float):
                logger.info(
                    f"\n🏆 Best Model: {best_model} (Val Loss: {best_loss:.6f})"
                )
            else:
                logger.info(f"\n🏆 Best Model: {best_model} (Val Loss: {best_loss})")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train all baseline models for maritime trajectory prediction"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/combined_aiscatcher_large.log",
        help="Path to training data (raw log, CSV, Parquet, or Zarr format) - defaults to 6.2GB dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Output directory for results and checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--use_maritime_safety",
        action="store_true",
        help="Use maritime safety-aware loss functions",
    )

    args = parser.parse_args()

    # Create configuration
    config = BaselineExperimentConfig(args)

    # Create trainer
    trainer = BaselineTrainer(config)

    # Train all baselines
    trainer.train_all_baselines()

    logger.info("\n✅ All baseline training complete!")
    logger.info(f"Results saved to: {config.results_dir}")
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")
    logger.info(f"Logs saved to: {config.logs_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
