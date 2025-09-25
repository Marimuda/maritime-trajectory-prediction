"""
Base infrastructure for classical ML baselines with minimal resistance integration.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from ..kalman.protocols import BaselineResult, TrajectoryBaseline


@dataclass
class ClassicalMLConfig:
    """Configuration for classical ML baselines."""

    # Model parameters
    model_type: str = "svr"  # "svr", "rf", "gbm"

    # Feature engineering
    use_rolling_features: bool = True
    rolling_windows: list[int] = field(default_factory=lambda: [3, 5, 10])
    use_lag_features: bool = True
    lag_steps: list[int] = field(default_factory=lambda: [1, 2, 3])
    use_diff_features: bool = True

    # Time-aware CV
    cv_splits: int = 5
    cv_gap: int = 0  # Gap between train and test to prevent leakage

    # Delta prediction
    predict_deltas: bool = True  # Predict changes vs absolutes

    # Parallelization
    n_jobs: int = -1  # Use all cores


class ClassicalMLBaseline(TrajectoryBaseline):
    """
    Abstract base class for classical ML baselines.
    Implements TrajectoryBaseline protocol with sklearn integration.
    """

    def __init__(self, config: ClassicalMLConfig = None):
        self.config = config or ClassicalMLConfig()
        self.models = {}  # Store per-horizon models
        self.feature_pipeline = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = None

    @abstractmethod
    def _create_base_model(self) -> BaseEstimator:
        """Create the underlying sklearn model."""
        pass

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "ClassicalMLBaseline":
        """
        Fit classical ML model using time-aware feature engineering.

        Args:
            sequences: List of [seq_len, n_features] trajectories
            metadata: Optional vessel metadata
            **kwargs: Additional parameters

        Returns:
            Fitted model instance
        """
        # Convert sequences to tabular format with features
        X, y = self._sequences_to_tabular(sequences)

        # Fit scaler and transform features
        X_transformed = self.scaler.fit_transform(X)
        self.feature_dim = X_transformed.shape[1]

        # Fit separate model for each prediction horizon
        max_horizon = kwargs.get("max_horizon", 12)

        for horizon in range(1, max_horizon + 1):
            # Extract targets for this horizon
            y_horizon = self._extract_horizon_targets(sequences, horizon)

            if len(y_horizon) > 0:
                # Create and fit model
                model = self._create_horizon_model()

                # Align X with y_horizon length
                X_horizon = X_transformed[: len(y_horizon)]
                model.fit(X_horizon, y_horizon)
                self.models[horizon] = model

        self.is_fitted = True
        return self

    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs,
    ) -> np.ndarray | BaselineResult:
        """
        Predict future trajectory using fitted classical ML model.

        Args:
            sequence: Input sequence [seq_len, n_features]
            horizon: Number of steps to predict
            return_uncertainty: Return confidence intervals

        Returns:
            Predictions or BaselineResult with uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Multi-step prediction strategy
        predictions = []
        current_seq = sequence.copy()

        for h in range(1, horizon + 1):
            # Transform sequence to features
            X = self._sequence_to_features(current_seq)
            X_scaled = self.scaler.transform(X.reshape(1, -1))

            if h in self.models:
                # Use horizon-specific model
                pred = self.models[h].predict(X_scaled)
            else:
                # Use nearest available model
                nearest_h = min(self.models.keys(), key=lambda k: abs(k - h))
                pred = self.models[nearest_h].predict(X_scaled)

            predictions.append(pred[0])

            # Update sequence for next prediction (autoregressive)
            current_seq = np.vstack([current_seq[1:], pred[0]])

        predictions = np.array(predictions)

        # Convert deltas to absolute positions if needed
        if self.config.predict_deltas:
            predictions = self._integrate_deltas(sequence, predictions)

        if return_uncertainty:
            # Estimate uncertainty using model-specific method
            uncertainty = self._estimate_uncertainty(sequence, predictions)

            return BaselineResult(
                predictions=predictions,
                uncertainty=uncertainty,
                metadata={"model_type": self.config.model_type},
                model_info=self.get_model_info(),
            )

        return predictions

    def _sequences_to_tabular(
        self, sequences: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert sequences to tabular format with engineered features."""
        feature_rows = []
        target_rows = []

        for seq in sequences:
            # Extract features for each timestep that has enough history
            min_history = max(self.config.lag_steps) if self.config.lag_steps else 1

            for t in range(min_history, len(seq) - 1):
                features = self._extract_features_at_timestep(seq, t)

                # Target is next timestep
                if self.config.predict_deltas:
                    target = seq[t + 1] - seq[t]
                else:
                    target = seq[t + 1]

                feature_rows.append(features)
                target_rows.append(target)

        if not feature_rows:
            return np.array([]), np.array([])

        return np.array(feature_rows), np.array(target_rows)

    def _extract_features_at_timestep(self, sequence: np.ndarray, t: int) -> np.ndarray:
        """Extract engineered features at timestep t."""
        features = []

        # Current state
        features.extend(sequence[t])

        # Lag features
        if self.config.use_lag_features:
            for lag in self.config.lag_steps:
                if t - lag >= 0:
                    features.extend(sequence[t - lag])
                else:
                    features.extend(np.zeros_like(sequence[0]))

        # Difference features
        if self.config.use_diff_features:
            if t > 0:
                features.extend(sequence[t] - sequence[t - 1])
            else:
                features.extend(np.zeros_like(sequence[0]))

        # Rolling statistics
        if self.config.use_rolling_features:
            for window in self.config.rolling_windows:
                start_idx = max(0, t - window + 1)
                window_data = sequence[start_idx : t + 1]

                # Mean
                features.extend(np.mean(window_data, axis=0))
                # Std
                if len(window_data) > 1:
                    features.extend(np.std(window_data, axis=0))
                else:
                    features.extend(np.zeros_like(sequence[0]))

        return np.array(features)

    def _sequence_to_features(self, sequence: np.ndarray) -> np.ndarray:
        """Convert single sequence to feature vector."""
        # Use last valid timestep for prediction
        t = len(sequence) - 1
        return self._extract_features_at_timestep(sequence, t)

    def _extract_horizon_targets(
        self, sequences: list[np.ndarray], horizon: int
    ) -> np.ndarray:
        """Extract targets for specific horizon."""
        targets = []

        for seq in sequences:
            min_history = max(self.config.lag_steps) if self.config.lag_steps else 1

            for t in range(min_history, len(seq) - horizon):
                if self.config.predict_deltas:
                    # Sum of deltas over horizon
                    target = seq[t + horizon] - seq[t]
                else:
                    target = seq[t + horizon]
                targets.append(target)

        if not targets:
            return np.array([])

        return np.array(targets)

    def _integrate_deltas(self, sequence: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Convert delta predictions to absolute positions."""
        # Start from last observed position
        last_pos = sequence[-1]
        absolutes = np.zeros_like(deltas)

        for i in range(len(deltas)):
            absolutes[i] = last_pos + deltas[i]
            last_pos = absolutes[i]

        return absolutes

    def _create_horizon_model(self):
        """Create model for specific horizon (can be overridden)."""
        base_model = self._create_base_model()
        # Wrap in MultiOutputRegressor for multi-dimensional targets
        return MultiOutputRegressor(base_model, n_jobs=self.config.n_jobs)

    @abstractmethod
    def _estimate_uncertainty(
        self, sequence: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Estimate prediction uncertainty (model-specific)."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Return model configuration and state."""
        return {
            "model_type": self.config.model_type,
            "n_horizons": len(self.models),
            "is_fitted": self.is_fitted,
            "feature_dim": self.feature_dim,
            "config": {
                "use_rolling_features": self.config.use_rolling_features,
                "rolling_windows": self.config.rolling_windows,
                "use_lag_features": self.config.use_lag_features,
                "lag_steps": self.config.lag_steps,
                "use_diff_features": self.config.use_diff_features,
                "predict_deltas": self.config.predict_deltas,
                "cv_splits": self.config.cv_splits,
                "cv_gap": self.config.cv_gap,
                "n_jobs": self.config.n_jobs,
            },
        }
