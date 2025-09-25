"""
PyTorch Lightning wrapper for Kalman filter baselines.

This wrapper enables seamless integration of Kalman baselines with the
existing PyTorch Lightning training infrastructure and evaluation framework.
"""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .imm import MaritimeIMMFilter
from .protocols import BaselineResult, IMMConfig


class KalmanBaselineLightning(pl.LightningModule):
    """
    Lightning wrapper for Kalman filter baselines.

    This wrapper allows Kalman baselines to be used within the existing
    PyTorch Lightning training and evaluation infrastructure, even though
    the underlying models are not neural networks.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Lightning wrapper for Kalman baseline.

        Args:
            config: Configuration dictionary for the Kalman baseline
        """
        super().__init__()
        self.save_hyperparameters()

        # Parse configuration
        self.kalman_config = self._parse_config(config or {})

        # Initialize Kalman baseline
        self.kalman_filter = MaritimeIMMFilter(self.kalman_config)

        # Storage for training trajectories (Kalman filters don't "train" in the traditional sense)
        self.training_trajectories = []

        # Metrics storage for evaluation
        self.validation_predictions = []
        self.validation_targets = []

    def _parse_config(self, config: dict[str, Any]) -> IMMConfig:
        """
        Parse configuration dictionary into IMMConfig.

        Args:
            config: Raw configuration dictionary

        Returns:
            Parsed IMMConfig instance
        """
        # Handle different configuration formats
        if isinstance(config, IMMConfig):
            return config

        # Extract relevant parameters from config
        reference_point = config.get("reference_point")
        max_speed_knots = config.get("max_speed_knots", 50.0)
        transition_stay_prob = config.get("transition_stay_probability", 0.95)

        # Create IMMConfig
        from .imm import create_default_imm_config

        return create_default_imm_config(
            reference_point=reference_point,
            max_speed_knots=max_speed_knots,
            transition_probability_stay=transition_stay_prob,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass - predict using Kalman filter.

        Args:
            batch: Batch containing input sequences

        Returns:
            Predictions as torch tensor
        """
        # Extract input sequence
        if "input" in batch:
            input_seq = batch["input"]
        elif "x" in batch:
            input_seq = batch["x"]
        else:
            raise ValueError("Batch must contain 'input' or 'x' key")

        # Convert to numpy for Kalman filter
        if isinstance(input_seq, torch.Tensor):
            input_seq = input_seq.detach().cpu().numpy()

        predictions = []

        # Process each sequence in batch
        batch_size = input_seq.shape[0]
        for i in range(batch_size):
            sequence = input_seq[i]  # Shape: [seq_len, features]

            # Determine prediction horizon
            horizon = batch["target"].shape[1] if "target" in batch else 1

            try:
                # Make prediction using Kalman filter
                result = self.kalman_filter.predict(sequence, horizon=horizon)
                predictions.append(result.predictions)
            except Exception:
                # Fallback: return last known position repeated
                last_pos = sequence[-1, :2]  # Assume first 2 features are lat, lon
                pred = np.tile(last_pos, (horizon, 1))
                predictions.append(pred)

        # Convert back to torch tensor
        predictions = np.array(predictions)
        return torch.from_numpy(predictions).float()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """
        Training step for Kalman baseline.

        Since Kalman filters don't train in the traditional sense,
        we collect training trajectories for later fitting.
        """
        # Extract and store training trajectories
        if "input" in batch:
            input_seq = batch["input"].detach().cpu().numpy()

            # Store trajectories for later fitting
            for i in range(input_seq.shape[0]):
                sequence = input_seq[i]
                self.training_trajectories.append(sequence)

        # Return dummy loss (Kalman filters don't have trainable parameters)
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step - evaluate Kalman predictions.
        """
        # Make predictions
        predictions = self(batch)

        # Extract targets
        if "target" in batch:
            targets = batch["target"]
        elif "y" in batch:
            targets = batch["y"]
        else:
            raise ValueError("Batch must contain 'target' or 'y' key")

        # Compute validation loss (e.g., MSE)
        loss = torch.nn.functional.mse_loss(predictions, targets)

        # Store for epoch-end metrics
        self.validation_predictions.extend(predictions.detach().cpu().numpy())
        self.validation_targets.extend(targets.detach().cpu().numpy())

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_training_epoch_end(self):
        """
        End of training epoch - fit Kalman filter on collected trajectories.
        """
        if self.training_trajectories and not self.kalman_filter.is_initialized:
            try:
                # Fit Kalman filter on collected trajectories
                self.kalman_filter.fit(self.training_trajectories)
                print(
                    f"Fitted Kalman filter on {len(self.training_trajectories)} trajectories"
                )
            except Exception as e:
                print(f"Warning: Failed to fit Kalman filter: {e}")

        # Clear trajectories to avoid memory growth
        self.training_trajectories = []

    def on_validation_epoch_end(self):
        """
        End of validation epoch - compute additional metrics.
        """
        if self.validation_predictions and self.validation_targets:
            predictions = np.array(self.validation_predictions)
            targets = np.array(self.validation_targets)

            # Compute additional metrics
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))

            # Log additional metrics
            self.log("val_mae", mae, on_epoch=True, prog_bar=True)
            self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)

            # Clear stored predictions
            self.validation_predictions = []
            self.validation_targets = []

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """
        Test step - similar to validation step.
        """
        predictions = self(batch)

        if "target" in batch:
            targets = batch["target"]
        elif "y" in batch:
            targets = batch["y"]
        else:
            raise ValueError("Batch must contain 'target' or 'y' key")

        # Compute test loss
        loss = torch.nn.functional.mse_loss(predictions, targets)

        # Compute additional test metrics
        mae = torch.nn.functional.l1_loss(predictions, targets)

        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True)

        return {"test_loss": loss, "predictions": predictions, "targets": targets}

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """
        Prediction step for inference.
        """
        predictions = self(batch)

        return {"predictions": predictions, "batch_idx": batch_idx}

    def configure_optimizers(self):
        """
        Configure optimizers (not needed for Kalman filters).
        """
        # Return dummy optimizer since Kalman filters don't train
        # This is required by Lightning but won't be used
        dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        return torch.optim.Adam([dummy_param], lr=1e-3)

    def get_baseline_result(
        self,
        sequence: torch.Tensor | np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
    ) -> BaselineResult:
        """
        Get native baseline result from Kalman filter.

        Args:
            sequence: Input sequence
            horizon: Prediction horizon
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            BaselineResult with predictions and metadata
        """
        # Convert tensor to numpy if needed
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().numpy()

        return self.kalman_filter.predict(
            sequence, horizon=horizon, return_uncertainty=return_uncertainty
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive model information.
        """
        info = self.kalman_filter.get_model_info()
        info.update(
            {
                "lightning_wrapper": True,
                "wrapper_class": self.__class__.__name__,
                "hyperparameters": self.hparams,
            }
        )
        return info

    @property
    def is_baseline_initialized(self) -> bool:
        """Check if the underlying baseline model is initialized."""
        return self.kalman_filter.is_initialized

    def fit_baseline_on_dataset(self, dataloader: DataLoader):
        """
        Fit the Kalman baseline on a complete dataset.

        Args:
            dataloader: DataLoader with trajectory sequences
        """
        trajectories = []

        # Collect all trajectories from dataloader
        for batch in dataloader:
            if "input" in batch:
                input_seq = batch["input"]
            elif "x" in batch:
                input_seq = batch["x"]
            else:
                continue

            # Convert to numpy and collect
            input_seq = input_seq.detach().cpu().numpy()
            for i in range(input_seq.shape[0]):
                trajectories.append(input_seq[i])

        # Fit Kalman filter
        if trajectories:
            self.kalman_filter.fit(trajectories)
            print(f"Fitted Kalman baseline on {len(trajectories)} trajectories")

    def enable_uncertainty_prediction(self):
        """Enable uncertainty estimation in predictions."""
        # This could be implemented by storing a flag and modifying forward()
        # For now, uncertainty is available via get_baseline_result()
        pass


class KalmanEvaluationMixin:
    """
    Mixin class for enhanced Kalman baseline evaluation.

    This can be mixed with the existing evaluation framework
    to provide specialized evaluation for Kalman baselines.
    """

    def evaluate_kalman_baseline(
        self,
        model: KalmanBaselineLightning,
        dataloader: DataLoader,
        return_uncertainty: bool = False,
    ) -> dict[str, Any]:
        """
        Specialized evaluation for Kalman baselines.

        Args:
            model: KalmanBaselineLightning model
            dataloader: Evaluation data
            return_uncertainty: Whether to compute uncertainty metrics

        Returns:
            Dictionary with evaluation results
        """
        model.eval()

        predictions = []
        targets = []
        uncertainties = []
        model_infos = []

        with torch.no_grad():
            for batch in dataloader:
                # Get standard predictions
                batch_preds = model(batch)
                batch_targets = batch["target"] if "target" in batch else batch["y"]

                predictions.extend(batch_preds.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())

                # Get enhanced results with uncertainty if requested
                if return_uncertainty:
                    input_seq = batch["input"] if "input" in batch else batch["x"]
                    for i in range(input_seq.shape[0]):
                        try:
                            result = model.get_baseline_result(
                                input_seq[i],
                                horizon=batch_targets.shape[1],
                                return_uncertainty=True,
                            )
                            if result.uncertainty is not None:
                                uncertainties.append(result.uncertainty)
                            model_infos.append(result.model_info)
                        except Exception:
                            # Fallback if baseline fails
                            continue

        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute standard metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        results = {"mse": mse, "mae": mae, "rmse": rmse, "n_samples": len(predictions)}

        # Add uncertainty-based metrics if available
        if uncertainties:
            uncertainties = np.array(uncertainties)

            # Compute uncertainty statistics
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_trace = np.mean(
                [np.trace(cov) for cov in uncertainties.reshape(-1, 2, 2)]
            )

            results.update(
                {
                    "mean_uncertainty": mean_uncertainty,
                    "uncertainty_trace": uncertainty_trace,
                    "uncertainty_samples": len(uncertainties),
                }
            )

        # Add model-specific information
        if model_infos:
            # Aggregate model probabilities if available
            model_probs = []
            dominant_models = []

            for info in model_infos:
                if "final_model_probabilities" in info:
                    model_probs.append(info["final_model_probabilities"])
                if "dominant_model" in info:
                    dominant_models.append(info["dominant_model"])

            if model_probs:
                mean_model_probs = np.mean(model_probs, axis=0)
                results["mean_model_probabilities"] = {
                    "CV": mean_model_probs[0],
                    "CT": mean_model_probs[1],
                    "NCA": mean_model_probs[2],
                }

            if dominant_models:
                from collections import Counter

                model_counts = Counter(dominant_models)
                results["dominant_model_distribution"] = dict(model_counts)

        return results
