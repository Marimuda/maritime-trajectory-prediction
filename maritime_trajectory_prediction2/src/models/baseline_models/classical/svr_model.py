"""
Support Vector Regression baseline with maritime-specific optimizations.
"""

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from .base import ClassicalMLBaseline, ClassicalMLConfig


class SVRBaseline(ClassicalMLBaseline):
    """
    Support Vector Regression for trajectory prediction.

    Uses RBF kernel with automatic gamma scaling and epsilon-insensitive loss.
    Handles multi-output via sklearn's MultiOutputRegressor.
    """

    def __init__(
        self,
        config: ClassicalMLConfig = None,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        cache_size: int = 500,  # MB for kernel cache
    ):
        """
        Initialize SVR baseline model.

        Args:
            config: ClassicalMLConfig instance
            C: Regularization parameter
            epsilon: Epsilon-tube for regression
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            cache_size: Kernel cache size in MB
        """
        super().__init__(config or ClassicalMLConfig(model_type="svr"))
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.cache_size = cache_size

        # Use RobustScaler for SVR (more robust to outliers)
        self.scaler = RobustScaler()

    def _create_base_model(self):
        """Create SVR model with optimized parameters."""
        return SVR(
            C=self.C,
            epsilon=self.epsilon,
            kernel=self.kernel,
            gamma=self.gamma,
            cache_size=self.cache_size,
            max_iter=10000,
            tol=1e-3,
        )

    def _create_horizon_model(self):
        """Create multi-output SVR for trajectory prediction."""
        base_svr = self._create_base_model()

        # Wrap in MultiOutputRegressor for multi-dimensional targets
        return MultiOutputRegressor(base_svr, n_jobs=self.config.n_jobs)

    def _estimate_uncertainty(
        self, sequence: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Estimate prediction uncertainty for SVR.

        Uses distance to support vectors as proxy for uncertainty.
        For simplicity, uses a heuristic based on prediction horizon.
        """
        # Simple heuristic: uncertainty grows with prediction horizon
        horizon = len(predictions)
        base_uncertainty = 0.1  # Base uncertainty in km

        uncertainty = np.zeros_like(predictions)
        for h in range(horizon):
            # Linear growth of uncertainty with some noise
            uncertainty[h] = (
                base_uncertainty * (1 + 0.1 * h) * np.ones(predictions.shape[1])
            )

            # Add distance-based component if we have position features
            MIN_POSITION_FEATURES = 2
            if predictions.shape[1] >= MIN_POSITION_FEATURES:  # Has lat/lon
                # Distance from last observed point
                dist = np.linalg.norm(predictions[h, :2] - sequence[-1, :2])
                uncertainty[h, :2] *= 1 + dist / 10.0  # Scale by distance

        return uncertainty

    def tune_hyperparameters(
        self, sequences: list[np.ndarray], param_grid: dict[str, list] = None
    ) -> dict[str, Any]:
        """
        Tune SVR hyperparameters using time-aware cross-validation.

        Args:
            sequences: Training sequences
            param_grid: Parameters to search

        Returns:
            Best parameters found
        """
        if param_grid is None:
            param_grid = {
                "estimator__C": [0.1, 1.0, 10.0],
                "estimator__epsilon": [0.01, 0.1, 0.5],
                "estimator__gamma": ["scale", "auto", 0.001, 0.01],
            }

        # Convert sequences to tabular
        X, y = self._sequences_to_tabular(sequences)

        if len(X) == 0:
            print("Warning: No valid training samples for tuning")
            return {}

        X_scaled = self.scaler.fit_transform(X)

        # Create base model for tuning
        base_model = MultiOutputRegressor(self._create_base_model())

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits, gap=self.config.cv_gap)

        # Grid search with time-aware CV
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=self.config.n_jobs,
            verbose=1,
        )

        search.fit(X_scaled, y)

        # Update model parameters
        best_params = search.best_params_
        for key, value in best_params.items():
            if "estimator__" in key:
                param_name = key.replace("estimator__", "")
                setattr(self, param_name, value)

        print(f"Best SVR params: {best_params}")
        print(f"Best CV score: {search.best_score_:.4f}")

        return best_params

    def get_model_complexity(self) -> dict[str, Any]:
        """
        Get model complexity metrics.

        Returns:
            Dict with support vector counts and other complexity measures
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        complexity = {}
        for horizon, model in self.models.items():
            # MultiOutputRegressor contains individual SVR models
            if hasattr(model, "estimators_"):
                sv_counts = [len(est.support_vectors_) for est in model.estimators_]
                complexity[f"horizon_{horizon}"] = {
                    "n_support_vectors": sv_counts,
                    "mean_support_vectors": np.mean(sv_counts),
                }

        return complexity
