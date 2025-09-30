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

# Constants for data validation
MAX_LATITUDE_DEGREES = 90.0
MAX_LONGITUDE_DEGREES = 180.0
MIN_SEQUENCE_LENGTH_FOR_PREDICTION = 2


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
        Forward pass - not applicable for Kalman filter baselines.

        Kalman filters don't use neural network forward passes. Use predict_step()
        or call baseline_model.predict() directly instead.

        Raises:
            NotImplementedError: Kalman filters don't support forward() interface
        """
        raise NotImplementedError(
            "Kalman filter baselines don't support forward() pass. "
            "Use predict_step() or baseline_model.predict() instead."
        )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step - collect data for baseline model fitting.

        Kalman baselines don't use gradient-based training. This method extracts
        trajectory sequences from AISDataModule batch format for later fitting/tuning.

        Batch format from AISDataModule:
            - "input": (batch_size, seq_len, features) - historical trajectory
            - "target": (batch_size, pred_horizon, 4) - future positions [lat, lon, sog, cog]
        """
        # Handle AISDataModule batch format
        if "input" not in batch or "target" not in batch:
            raise ValueError(
                f"Expected 'input' and 'target' keys in batch, got: {list(batch.keys())}"
            )

        # Extract input and target sequences
        input_seq = batch["input"].cpu().numpy()  # (batch, seq_len, features)
        target_seq = batch["target"].cpu().numpy()  # (batch, pred_horizon, 4)

        # Extract lat/lon from first 2 columns and combine input + target for full trajectory
        # Input features order: [lat, lon, sog, cog, heading, turn, ...derived...]
        # Target features order: [lat, lon, sog, cog]
        for i in range(len(input_seq)):
            # Get lat/lon from input (first 2 columns)
            input_latlon = input_seq[i, :, :2]  # (seq_len, 2)

            # Get lat/lon from target (first 2 columns)
            target_latlon = target_seq[i, :, :2]  # (pred_horizon, 2)

            # Combine to form full trajectory
            full_trajectory = np.concatenate([input_latlon, target_latlon], axis=0)

            # Validate trajectory (no NaN, reasonable lat/lon bounds)
            if (
                not np.any(np.isnan(full_trajectory))
                and np.all(np.abs(full_trajectory[:, 0]) <= MAX_LATITUDE_DEGREES)
                and np.all(np.abs(full_trajectory[:, 1]) <= MAX_LONGITUDE_DEGREES)
            ):
                MIN_VIABLE_SEQUENCE_LENGTH = 2
                if len(full_trajectory) >= MIN_VIABLE_SEQUENCE_LENGTH:
                    self.training_sequences.append(full_trajectory)

        # Fit immediately on first batch if not using auto_tune
        # (fit() is lightweight - just sets coordinate transform reference)
        if (
            not self.auto_tune
            and not self.is_fitted
            and len(self.training_sequences) > 0
        ):
            self.baseline_model.fit(
                self.training_sequences[:1]
            )  # Only need one sequence
            self.is_fitted = True

        # Kalman filters don't use gradient-based training
        # Since configure_optimizers returns None (manual optimization mode),
        # Lightning doesn't use this value for backprop - it's just for logging
        # Return number of sequences collected for monitoring
        return torch.tensor(float(len(self.training_sequences)), dtype=torch.float32)

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
    ) -> dict[str, float] | None:
        """
        Validation step - evaluate Kalman baseline predictions.

        Uses input sequence to predict target sequence and compares with ground truth.
        Returns proper maritime distance metrics, not dummy values.
        """
        if not self.is_fitted:
            # Model not fitted yet - skip this validation step
            # Lightning validation_step can return None to skip the step
            return None

        # Extract sequences from AISDataModule format
        input_seq = batch["input"].cpu().numpy()  # (batch, seq_len, features)
        target_seq = batch["target"].cpu().numpy()  # (batch, pred_horizon, 4)

        total_loss = 0.0
        n_predictions = 0

        for i in range(len(input_seq)):
            # Extract lat/lon from input (first 2 columns)
            input_latlon = input_seq[i, :, :2]  # (seq_len, 2)

            # Extract lat/lon targets (first 2 columns)
            target_latlon = target_seq[i, :, :2]  # (pred_horizon, 2)

            # Validate input
            if (
                np.any(np.isnan(input_latlon))
                or np.any(np.isnan(target_latlon))
                or len(input_latlon) < MIN_SEQUENCE_LENGTH_FOR_PREDICTION
            ):
                continue

            try:
                # Predict using Kalman filter
                result = self.baseline_model.predict(
                    input_latlon,
                    horizon=self.prediction_horizon,
                    return_uncertainty=False,
                )

                # Get predictions (may be shorter than horizon if prediction fails)
                predictions = result.predictions  # (actual_horizon, 2)
                actual_horizon = len(predictions)

                if actual_horizon == 0:
                    continue

                # Match targets to prediction length
                targets = target_latlon[:actual_horizon]  # (actual_horizon, 2)

                # Calculate proper maritime distance using Haversine formula
                # Import here to avoid circular dependencies
                from src.utils.maritime_utils import MaritimeUtils

                # Calculate distance for each predicted point (vectorized)
                distances_nm = np.array(
                    [
                        MaritimeUtils.calculate_distance(
                            pred[0], pred[1], tgt[0], tgt[1]
                        )
                        for pred, tgt in zip(predictions, targets, strict=False)
                    ]
                )

                # Convert nautical miles to meters (1 NM = 1852 meters)
                distances_m = distances_nm * 1852.0

                # Calculate MSE in meters squared
                mse_m2 = np.mean(distances_m**2)
                total_loss += mse_m2
                n_predictions += 1

            except Exception as e:
                # Skip failed predictions (don't log every failure)
                if batch_idx == 0 and i == 0:  # Log first failure only
                    print(f"Prediction failed: {e}")
                continue

        if n_predictions > 0:
            # Average MSE in meters squared
            avg_mse_m2 = total_loss / n_predictions

            # Calculate RMSE in meters (more interpretable)
            rmse_m = np.sqrt(avg_mse_m2)

            # Log both MSE and RMSE for completeness
            self.log("val_loss", avg_mse_m2, prog_bar=False)  # MSE for Lightning
            self.log("val_rmse_m", rmse_m, prog_bar=True)  # RMSE for humans
            self.log("val_n_predictions", float(n_predictions), prog_bar=False)

            return {"val_loss": avg_mse_m2, "val_rmse_m": rmse_m}
        else:
            # No valid predictions - return None instead of dummy value
            return None

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step - similar to validation but for final evaluation."""
        return self.validation_step(batch, batch_idx)

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Prediction step - generate predictions using Kalman baseline model.

        Uses AISDataModule batch format with "input" key containing historical trajectory.
        Returns predictions as (batch_size, prediction_horizon, 2) tensor of [lat, lon].
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot predict with unfitted model. Call fit() or run training first."
            )

        # Extract input sequences (historical trajectory)
        input_seq = batch["input"].cpu().numpy()  # (batch, seq_len, features)

        batch_predictions = []

        for i in range(len(input_seq)):
            # Extract lat/lon from first 2 columns
            input_latlon = input_seq[i, :, :2]  # (seq_len, 2)

            # Validate input
            if (
                np.any(np.isnan(input_latlon))
                or len(input_latlon) < MIN_SEQUENCE_LENGTH_FOR_PREDICTION
            ):
                raise ValueError(
                    f"Invalid input sequence at batch index {i}: "
                    f"contains NaN or too short (< {MIN_SEQUENCE_LENGTH_FOR_PREDICTION} points)"
                )

            try:
                # Make prediction using Kalman filter
                result = self.baseline_model.predict(
                    input_latlon,
                    horizon=self.prediction_horizon,
                    return_uncertainty=False,
                )
                batch_predictions.append(result.predictions)
            except Exception as e:
                raise RuntimeError(
                    f"Kalman prediction failed for batch index {i}: {e}"
                ) from e

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
