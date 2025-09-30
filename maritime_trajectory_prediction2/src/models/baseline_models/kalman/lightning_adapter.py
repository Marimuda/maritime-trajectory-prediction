"""
PyTorch Lightning adapter for Kalman filter baseline models.

This module provides a Lightning wrapper to integrate Kalman filter baselines
with the existing PyTorch Lightning training infrastructure.
"""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from .imm import MaritimeIMMFilter
from .models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from .protocols import (
    IMMConfig,
    MaritimeConstraints,
    MotionModelConfig,
    TrajectoryBaseline,
)
from .tuning import tune_maritime_baseline


class KalmanBaselineLightning(pl.LightningModule):
    """
    PyTorch Lightning adapter for Kalman filter baseline models.

    This wrapper allows Kalman baseline models to be used within the PyTorch Lightning
    framework for consistent evaluation and comparison with neural models.
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        model_type: str = "imm",
        auto_tune: bool = False,
        prediction_horizon: int = 5,
        **kwargs,
    ):
        """
        Initialize Lightning wrapper for Kalman baseline.

        Args:
            config: Model configuration dictionary
            model_type: Type of baseline model ('imm', 'cv', 'ct', 'nca')
            auto_tune: Whether to automatically tune hyperparameters
            prediction_horizon: Default prediction horizon
            **kwargs: Additional configuration parameters
        """
        super().__init__()

        self.save_hyperparameters()

        # Store configuration
        self.model_type = model_type.lower()
        self.auto_tune = auto_tune
        self.prediction_horizon = prediction_horizon

        # Create underlying baseline model
        self.baseline_model = self._create_baseline_model(config or {})

        # Store training data for potential hyperparameter tuning
        self.training_sequences: list[np.ndarray] = []
        self.is_fitted = False

        # Metrics tracking
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []

    def _create_baseline_model(self, config: dict[str, Any]) -> TrajectoryBaseline:
        """Create the underlying baseline model based on configuration."""
        # Parse configuration
        if self.model_type == "imm":
            # Create IMM configuration
            motion_config = MotionModelConfig(**config.get("motion_config", {}))
            constraints = MaritimeConstraints(**config.get("constraints", {}))

            imm_config = IMMConfig(
                motion_config=motion_config,
                constraints=constraints,
                **config.get("imm_config", {}),
            )

            return MaritimeIMMFilter(imm_config)

        elif self.model_type == "cv":
            motion_config = MotionModelConfig(**config.get("motion_config", {}))
            constraints = MaritimeConstraints(**config.get("constraints", {}))
            return ConstantVelocityModel(motion_config, constraints)

        elif self.model_type == "ct":
            motion_config = MotionModelConfig(**config.get("motion_config", {}))
            constraints = MaritimeConstraints(**config.get("constraints", {}))
            return CoordinatedTurnModel(motion_config, constraints)

        elif self.model_type == "nca":
            motion_config = MotionModelConfig(**config.get("motion_config", {}))
            constraints = MaritimeConstraints(**config.get("constraints", {}))
            return NearlyConstantAccelModel(motion_config, constraints)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def setup(self, stage: str | None = None) -> None:
        """Setup the model for training/validation/testing."""
        if (
            stage == "fit"
            and self.auto_tune
            and not self.is_fitted
            and len(self.training_sequences) > 0
        ):
            print(f"Auto-tuning {self.model_type} baseline...")

            tuning_results = tune_maritime_baseline(
                self.training_sequences,
                model_type=self.model_type,
                prediction_horizon=self.prediction_horizon,
                max_iterations=20,  # Reduced for Lightning integration
            )

            # Update model with tuned configuration
            if "optimized_config" in tuning_results:
                optimized_config = tuning_results["optimized_config"]
                self.baseline_model = self._create_baseline_model(
                    {
                        "motion_config": optimized_config.__dict__
                        if hasattr(optimized_config, "__dict__")
                        else optimized_config
                    }
                )

            print(
                f"Tuning completed with score: {tuning_results.get('best_score', 'N/A')}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - not applicable for Kalman baselines.

        This method is required by Lightning but not used for baseline models
        since they follow a different prediction protocol.
        """
        # Return dummy tensor to satisfy Lightning interface
        return torch.zeros((x.shape[0], self.prediction_horizon, 2))

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step - collect data for baseline model fitting.

        Since Kalman baselines don't use gradient-based training, this method
        primarily collects trajectory sequences for later fitting.
        """
        # Extract trajectory data from batch
        # Assuming batch contains 'trajectory' and 'metadata' keys
        if "trajectory" in batch:
            trajectories = batch["trajectory"].cpu().numpy()

            # Convert batch to list of sequences
            for traj in trajectories:
                # Remove padding if present (assuming -1 padding)
                valid_mask = traj[:, 0] != -1  # Assuming first column is latitude
                if np.any(valid_mask):
                    valid_traj = traj[valid_mask]
                    MIN_VIABLE_SEQUENCE_LENGTH = 2
                    if (
                        len(valid_traj) >= MIN_VIABLE_SEQUENCE_LENGTH
                    ):  # Minimum viable sequence length
                        self.training_sequences.append(valid_traj)

        # Return dummy loss (baseline models don't use gradient-based training)
        return torch.tensor(0.0, requires_grad=True)

    def on_train_epoch_end(self) -> None:
        """End of training epoch - fit baseline model on collected data."""
        if len(self.training_sequences) > 0 and not self.is_fitted:
            print(
                f"Fitting {self.model_type} baseline on {len(self.training_sequences)} sequences..."
            )

            try:
                self.baseline_model.fit(self.training_sequences)
                self.is_fitted = True
                print("Baseline model fitting completed.")
            except Exception as e:
                print(f"Warning: Baseline model fitting failed: {e}")
                # Continue with default model

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step - evaluate baseline model predictions.
        """
        if not self.is_fitted:
            # Return dummy loss if model not fitted yet
            return torch.tensor(1.0)

        try:
            # Extract validation trajectories
            trajectories = batch["trajectory"].cpu().numpy()

            total_loss = 0.0
            n_predictions = 0

            for traj in trajectories:
                # Remove padding
                valid_mask = traj[:, 0] != -1
                if not np.any(valid_mask):
                    continue

                valid_traj = traj[valid_mask]
                if len(valid_traj) < self.prediction_horizon + 2:
                    continue

                # Split trajectory for prediction
                input_length = len(valid_traj) - self.prediction_horizon
                input_seq = valid_traj[:input_length]
                target_seq = valid_traj[
                    input_length : input_length + self.prediction_horizon
                ]

                try:
                    # Make prediction
                    result = self.baseline_model.predict(
                        input_seq,
                        horizon=self.prediction_horizon,
                        return_uncertainty=False,
                    )

                    # Calculate MSE loss in geographic coordinates
                    predictions = result.predictions
                    targets = target_seq[: len(predictions), :2]  # lat, lon only

                    # Simple Euclidean distance in degrees (approximation)
                    mse = np.mean((predictions - targets) ** 2)
                    total_loss += mse
                    n_predictions += 1

                except Exception:
                    # Skip this prediction if it fails
                    continue

            if n_predictions > 0:
                avg_loss = total_loss / n_predictions
                self.log("val_loss", avg_loss)
                return torch.tensor(avg_loss)
            else:
                return torch.tensor(1.0)

        except Exception as e:
            print(f"Validation step error: {e}")
            return torch.tensor(1.0)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step - similar to validation but for final evaluation."""
        return self.validation_step(batch, batch_idx)

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Prediction step - generate predictions using baseline model.
        """
        if not self.is_fitted:
            # Return zeros if model not fitted
            batch_size = batch["trajectory"].shape[0]
            return torch.zeros((batch_size, self.prediction_horizon, 2))

        trajectories = batch["trajectory"].cpu().numpy()
        batch_predictions = []

        for traj in trajectories:
            # Remove padding
            valid_mask = traj[:, 0] != -1
            if not np.any(valid_mask):
                # Return zeros for invalid trajectory
                batch_predictions.append(np.zeros((self.prediction_horizon, 2)))
                continue

            valid_traj = traj[valid_mask]
            MIN_PREDICTION_SEQUENCE_LENGTH = 2
            if len(valid_traj) < MIN_PREDICTION_SEQUENCE_LENGTH:
                batch_predictions.append(np.zeros((self.prediction_horizon, 2)))
                continue

            try:
                result = self.baseline_model.predict(
                    valid_traj,
                    horizon=self.prediction_horizon,
                    return_uncertainty=False,
                )
                batch_predictions.append(result.predictions)
            except Exception:
                # Fallback to zeros if prediction fails
                batch_predictions.append(np.zeros((self.prediction_horizon, 2)))

        return torch.tensor(np.array(batch_predictions), dtype=torch.float32)

    def configure_optimizers(self):
        """
        Configure optimizers - not applicable for Kalman baselines.

        Return None since Kalman filters don't use gradient-based optimization.
        """
        return None

    def get_baseline_model(self) -> TrajectoryBaseline:
        """Get the underlying baseline model."""
        return self.baseline_model

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information."""
        base_info = {
            "model_type": f"KalmanBaseline_{self.model_type.upper()}",
            "lightning_wrapper": True,
            "is_fitted": self.is_fitted,
            "auto_tune": self.auto_tune,
            "prediction_horizon": self.prediction_horizon,
            "n_training_sequences": len(self.training_sequences),
        }

        # Add baseline model info
        if self.baseline_model:
            base_info["baseline_info"] = self.baseline_model.get_model_info()

        return base_info


# Factory functions for different baseline types
def create_imm_lightning(
    config: dict[str, Any] = None, **kwargs
) -> KalmanBaselineLightning:
    """Create IMM baseline with Lightning wrapper."""
    return KalmanBaselineLightning(config or {}, model_type="imm", **kwargs)


def create_cv_lightning(
    config: dict[str, Any] = None, **kwargs
) -> KalmanBaselineLightning:
    """Create Constant Velocity baseline with Lightning wrapper."""
    return KalmanBaselineLightning(config or {}, model_type="cv", **kwargs)


def create_ct_lightning(
    config: dict[str, Any] = None, **kwargs
) -> KalmanBaselineLightning:
    """Create Coordinated Turn baseline with Lightning wrapper."""
    return KalmanBaselineLightning(config or {}, model_type="ct", **kwargs)


def create_nca_lightning(
    config: dict[str, Any] = None, **kwargs
) -> KalmanBaselineLightning:
    """Create Nearly Constant Acceleration baseline with Lightning wrapper."""
    return KalmanBaselineLightning(config or {}, model_type="nca", **kwargs)
