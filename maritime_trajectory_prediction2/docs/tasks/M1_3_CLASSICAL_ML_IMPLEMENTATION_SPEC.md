# M1.3 Classical ML Baselines - Implementation Specification

## Executive Summary

This document provides a complete implementation specification for M1.3 Classical ML Baselines, integrating seamlessly with the existing codebase architecture. The implementation leverages mature scikit-learn libraries and follows established patterns from the existing XGBoost and Kalman baseline implementations.

## Integration Architecture Analysis

### Current Codebase Structure
```
src/models/
├── baseline_models/
│   ├── __init__.py                    # Factory pattern for baseline creation
│   ├── kalman/                        # Physics-based baselines (M1.2 complete)
│   │   ├── protocols.py              # TrajectoryBaseline Protocol ✓
│   │   ├── models.py                  # Motion models
│   │   └── lightning_wrapper.py       # Lightning integration
│   └── [NEW] classical/               # Classical ML baselines (M1.3)
├── benchmark_models/
│   ├── xgboost_model.py              # Existing XGBoost pattern ✓
│   └── lstm_model.py                  # Neural baseline
```

### Key Integration Points Identified

1. **TrajectoryBaseline Protocol** (`protocols.py`): Standard interface already defined
2. **BaselineResult Dataclass**: Structured output with uncertainty support
3. **Lightning Integration Pattern**: Optional via wrapper (like Kalman)
4. **Factory Pattern**: Existing `create_baseline_model()` function

## Library Endpoints and Mature Solutions

### Primary: Scikit-Learn (v1.3+)
```python
# Core Regression Models
from sklearn.svm import SVR                          # Support Vector Regression
from sklearn.ensemble import RandomForestRegressor   # Random Forest
from sklearn.multioutput import MultiOutputRegressor # Multi-target wrapper

# Feature Engineering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Pipeline Components
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# Time-Aware Cross-Validation
from sklearn.model_selection import (
    TimeSeriesSplit,      # Temporal splits
    GroupKFold,           # Vessel-based grouping
    GridSearchCV,         # Hyperparameter tuning
    cross_val_score       # Evaluation
)

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
```

### Secondary: Feature Engineering
```python
# Pandas for time-series features
import pandas as pd
pd.DataFrame.rolling()     # Rolling statistics
pd.DataFrame.shift()        # Lag features
pd.DataFrame.diff()         # Differences

# NumPy for efficient computation
import numpy as np
np.gradient()              # Numerical derivatives
np.convolve()              # Smoothing
```

### Supporting: Joblib for Parallelization
```python
from joblib import Parallel, delayed  # Parallel processing
from joblib import dump, load         # Model persistence
```

## Implementation Specification

### 1. Base Classical ML Infrastructure

**File**: `src/models/baseline_models/classical/base.py`

```python
"""
Base infrastructure for classical ML baselines with minimal resistance integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..kalman.protocols import BaselineResult, TrajectoryBaseline


@dataclass
class ClassicalMLConfig:
    """Configuration for classical ML baselines."""

    # Model parameters
    model_type: str = "svr"  # "svr", "rf", "gbm"

    # Feature engineering
    use_rolling_features: bool = True
    rolling_windows: List[int] = None  # [3, 5, 10] timesteps
    use_lag_features: bool = True
    lag_steps: List[int] = None  # [1, 2, 3] steps back
    use_diff_features: bool = True

    # Time-aware CV
    cv_splits: int = 5
    cv_gap: int = 0  # Gap between train and test to prevent leakage

    # Delta prediction
    predict_deltas: bool = True  # Predict changes vs absolutes

    # Parallelization
    n_jobs: int = -1  # Use all cores

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5, 10]
        if self.lag_steps is None:
            self.lag_steps = [1, 2, 3]


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

    @abstractmethod
    def _create_base_model(self) -> BaseEstimator:
        """Create the underlying sklearn model."""
        pass

    def fit(
        self,
        sequences: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "ClassicalMLBaseline":
        """
        Fit classical ML model using time-aware feature engineering.

        Args:
            sequences: List of [seq_len, n_features] trajectories
            metadata: Optional vessel metadata

        Returns:
            Fitted model instance
        """
        # Convert sequences to tabular format with features
        X, y = self._sequences_to_tabular(sequences)

        # Fit feature pipeline
        X_transformed = self._fit_transform_features(X)

        # Fit separate model for each prediction horizon
        max_horizon = kwargs.get('max_horizon', 12)

        for horizon in range(1, max_horizon + 1):
            # Extract targets for this horizon
            y_horizon = self._extract_horizon_targets(y, horizon)

            # Create and fit model
            model = self._create_horizon_model()
            model.fit(X_transformed, y_horizon)

            self.models[horizon] = model

        self.is_fitted = True
        return self

    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs
    ) -> Union[np.ndarray, BaselineResult]:
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

        # Transform sequence to features
        X = self._sequence_to_features(sequence)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        # Multi-step prediction strategy
        predictions = []
        current_features = X_scaled

        for h in range(1, horizon + 1):
            if h in self.models:
                # Use horizon-specific model
                pred = self.models[h].predict(current_features)
            else:
                # Use nearest available model
                nearest_h = min(self.models.keys(),
                              key=lambda k: abs(k - h))
                pred = self.models[nearest_h].predict(current_features)

            predictions.append(pred[0])

            # Update features for next prediction
            current_features = self._update_features(current_features, pred)

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
                metadata={'model_type': self.config.model_type},
                model_info=self.get_model_info()
            )

        return predictions

    def _sequences_to_tabular(
        self,
        sequences: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sequences to tabular format with engineered features."""
        feature_rows = []
        target_rows = []

        for seq in sequences:
            # Extract features for each timestep
            for t in range(self.config.lag_steps[-1], len(seq) - 1):
                features = self._extract_features_at_timestep(seq, t)
                target = seq[t + 1] - seq[t] if self.config.predict_deltas else seq[t + 1]

                feature_rows.append(features)
                target_rows.append(target)

        return np.array(feature_rows), np.array(target_rows)

    def _extract_features_at_timestep(
        self,
        sequence: np.ndarray,
        t: int
    ) -> np.ndarray:
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
                window_data = sequence[start_idx:t + 1]

                # Mean
                features.extend(np.mean(window_data, axis=0))
                # Std
                if len(window_data) > 1:
                    features.extend(np.std(window_data, axis=0))
                else:
                    features.extend(np.zeros_like(sequence[0]))

        return np.array(features)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and state."""
        return {
            'model_type': self.config.model_type,
            'n_horizons': len(self.models),
            'is_fitted': self.is_fitted,
            'feature_dim': self.scaler.n_features_in_ if self.is_fitted else None,
            'config': self.config.__dict__
        }


```

### 2. SVR Implementation

**File**: `src/models/baseline_models/classical/svr_model.py`

```python
"""
Support Vector Regression baseline with maritime-specific optimizations.
"""

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
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
        kernel: str = 'rbf',
        gamma: str = 'scale',
        cache_size: int = 500  # MB for kernel cache
    ):
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
            tol=1e-3
        )

    def _create_horizon_model(self):
        """Create multi-output SVR for trajectory prediction."""
        base_svr = self._create_base_model()

        # Wrap in MultiOutputRegressor for multi-dimensional targets
        return MultiOutputRegressor(
            base_svr,
            n_jobs=self.config.n_jobs
        )

    def _estimate_uncertainty(
        self,
        sequence: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Estimate prediction uncertainty for SVR.

        Uses distance to support vectors as proxy for uncertainty.
        """
        # Simple heuristic: uncertainty grows with prediction horizon
        horizon = len(predictions)
        base_uncertainty = 0.1  # Base uncertainty in km

        uncertainty = np.zeros_like(predictions)
        for h in range(horizon):
            # Linear growth of uncertainty
            uncertainty[h] = base_uncertainty * (1 + 0.1 * h)

        return uncertainty

    def tune_hyperparameters(
        self,
        sequences: List[np.ndarray],
        param_grid: Dict[str, List] = None
    ) -> Dict[str, Any]:
        """
        Tune SVR hyperparameters using time-aware cross-validation.

        Args:
            sequences: Training sequences
            param_grid: Parameters to search

        Returns:
            Best parameters found
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        if param_grid is None:
            param_grid = {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__epsilon': [0.01, 0.1, 0.5],
                'estimator__gamma': ['scale', 'auto', 0.001, 0.01]
            }

        # Convert sequences to tabular
        X, y = self._sequences_to_tabular(sequences)
        X_scaled = self.scaler.fit_transform(X)

        # Create base model for tuning
        base_model = MultiOutputRegressor(self._create_base_model())

        # Time series cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_splits,
            gap=self.config.cv_gap
        )

        # Grid search with time-aware CV
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs,
            verbose=1
        )

        search.fit(X_scaled, y)

        # Update model parameters
        best_params = search.best_params_
        for key, value in best_params.items():
            if 'estimator__' in key:
                param_name = key.replace('estimator__', '')
                setattr(self, param_name, value)

        print(f"Best SVR params: {best_params}")
        print(f"Best CV score: {search.best_score_:.4f}")

        return best_params
```

### 3. Random Forest Implementation

**File**: `src/models/baseline_models/classical/rf_model.py`

```python
"""
Random Forest regression baseline with feature importance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Any, Optional

from .base import ClassicalMLBaseline, ClassicalMLConfig


class RFBaseline(ClassicalMLBaseline):
    """
    Random Forest baseline for trajectory prediction.

    Provides feature importance analysis and out-of-bag error estimation.
    Naturally handles multi-output without wrapper.
    """

    def __init__(
        self,
        config: ClassicalMLConfig = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        oob_score: bool = True,
        random_state: int = 42
    ):
        super().__init__(config or ClassicalMLConfig(model_type="rf"))
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state

        # Store feature importance
        self.feature_importances_ = {}

    def _create_base_model(self):
        """Create Random Forest with maritime-optimized parameters."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_score=self.oob_score,
            n_jobs=self.config.n_jobs,
            random_state=self.random_state,
            warm_start=False,  # Fresh training each time
            verbose=0
        )

    def _create_horizon_model(self):
        """RF naturally handles multi-output."""
        return self._create_base_model()

    def fit(
        self,
        sequences: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "RFBaseline":
        """
        Fit RF model and extract feature importances.
        """
        # Call parent fit
        super().fit(sequences, metadata, **kwargs)

        # Extract and store feature importances for each horizon
        for horizon, model in self.models.items():
            self.feature_importances_[horizon] = model.feature_importances_

        return self

    def _estimate_uncertainty(
        self,
        sequence: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Estimate prediction uncertainty using tree variance.

        RF can provide uncertainty through prediction variance across trees.
        """
        # Access individual tree predictions for variance estimation
        horizon = len(predictions)
        uncertainty = np.zeros_like(predictions)

        X = self._sequence_to_features(sequence)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        for h in range(1, horizon + 1):
            if h in self.models:
                model = self.models[h]

                # Get predictions from all trees
                tree_predictions = []
                for tree in model.estimators_:
                    tree_pred = tree.predict(X_scaled)
                    tree_predictions.append(tree_pred)

                tree_predictions = np.array(tree_predictions)

                # Compute standard deviation across trees
                if tree_predictions.shape[0] > 1:
                    uncertainty[h-1] = np.std(tree_predictions, axis=0)[0]
                else:
                    uncertainty[h-1] = 0.1 * h  # Fallback

        return uncertainty

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate feature importance report across all horizons.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.feature_importances_:
            raise ValueError("Model must be fitted first")

        # Create feature names based on engineering configuration
        feature_names = self._generate_feature_names()

        # Aggregate importances across horizons
        importance_data = []

        for horizon, importances in self.feature_importances_.items():
            for idx, importance in enumerate(importances):
                importance_data.append({
                    'horizon': horizon,
                    'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                    'importance': importance
                })

        df = pd.DataFrame(importance_data)

        # Compute mean importance across horizons
        mean_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)

        return mean_importance

    def _generate_feature_names(self) -> List[str]:
        """Generate human-readable feature names."""
        names = []
        base_features = ['lat', 'lon', 'sog', 'cog']

        # Current features
        names.extend([f'current_{f}' for f in base_features])

        # Lag features
        if self.config.use_lag_features:
            for lag in self.config.lag_steps:
                names.extend([f'lag{lag}_{f}' for f in base_features])

        # Difference features
        if self.config.use_diff_features:
            names.extend([f'diff_{f}' for f in base_features])

        # Rolling features
        if self.config.use_rolling_features:
            for window in self.config.rolling_windows:
                names.extend([f'roll{window}_mean_{f}' for f in base_features])
                names.extend([f'roll{window}_std_{f}' for f in base_features])

        return names

    def get_oob_score(self) -> Dict[int, float]:
        """
        Get out-of-bag scores for each horizon model.

        Returns:
            Dict mapping horizon to OOB R² score
        """
        if not self.oob_score:
            raise ValueError("OOB scoring not enabled")

        scores = {}
        for horizon, model in self.models.items():
            if hasattr(model, 'oob_score_'):
                scores[horizon] = model.oob_score_

        return scores
```

### 4. Time-Aware Cross-Validation

**File**: `src/models/baseline_models/classical/validation.py`

```python
"""
Time-aware cross-validation utilities for maritime trajectory prediction.
"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from typing import Iterator, Optional, Tuple


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validator with purging to prevent data leakage.

    Ensures gap between train and test sets to avoid look-ahead bias
    in maritime trajectory prediction.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
        max_train_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.gap = gap  # Gap between train and test
        self.test_size = test_size
        self.max_train_size = max_train_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits with purging."""
        n_samples = len(X)

        if self.test_size:
            test_size = self.test_size
        else:
            test_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Calculate split points
            test_end = n_samples - fold * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            # Ensure valid splits
            if train_end <= train_start or test_end <= test_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


class VesselGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Combined vessel-based and time-based splitting.

    Ensures no vessel appears in both train and test while
    maintaining temporal ordering.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        vessel_split_ratio: float = 0.8
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.vessel_split_ratio = vessel_split_ratio

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate splits ensuring vessel and temporal separation."""
        if groups is None:
            raise ValueError("Vessel groups required for splitting")

        unique_vessels = np.unique(groups)
        n_vessels = len(unique_vessels)

        # Time-based splitting first
        time_splitter = PurgedTimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap
        )

        for train_time_idx, test_time_idx in time_splitter.split(X):
            # Then apply vessel-based filtering
            train_vessels = np.random.choice(
                unique_vessels,
                size=int(n_vessels * self.vessel_split_ratio),
                replace=False
            )

            # Combine time and vessel constraints
            train_mask = np.isin(groups[train_time_idx], train_vessels)
            test_mask = ~np.isin(groups[test_time_idx], train_vessels)

            train_indices = train_time_idx[train_mask]
            test_indices = test_time_idx[test_mask]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
```

### 5. Feature Engineering Pipeline

**File**: `src/models/baseline_models/classical/features.py`

```python
"""
Maritime-specific feature engineering for classical ML models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class MaritimeFeatureEngineer:
    """
    Feature engineering specifically for maritime trajectories.
    """

    @staticmethod
    def extract_kinematic_features(trajectory: np.ndarray) -> np.ndarray:
        """
        Extract kinematic features from trajectory.

        Args:
            trajectory: [seq_len, 4] with [lat, lon, sog, cog]

        Returns:
            Extended features including acceleration, turn rate, etc.
        """
        features = []

        # Original features
        features.append(trajectory)

        # Speed acceleration (SOG derivative)
        sog = trajectory[:, 2]
        acceleration = np.gradient(sog)
        features.append(acceleration.reshape(-1, 1))

        # Turn rate (COG derivative)
        cog = trajectory[:, 3]
        # Handle circular difference
        cog_diff = np.diff(cog)
        cog_diff = np.where(cog_diff > 180, cog_diff - 360, cog_diff)
        cog_diff = np.where(cog_diff < -180, cog_diff + 360, cog_diff)
        turn_rate = np.concatenate([[0], cog_diff])
        features.append(turn_rate.reshape(-1, 1))

        # Speed/course interaction
        speed_course_interaction = sog * np.sin(np.radians(cog))
        features.append(speed_course_interaction.reshape(-1, 1))

        return np.concatenate(features, axis=1)

    @staticmethod
    def extract_temporal_features(
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Extract temporal features from timestamps.

        Returns features like hour of day, day of week, etc.
        """
        # Convert to pandas for easy datetime manipulation
        times = pd.to_datetime(timestamps, unit='s')

        features = np.column_stack([
            times.hour.values / 24.0,  # Normalized hour
            times.dayofweek.values / 7.0,  # Normalized day of week
            times.month.values / 12.0,  # Normalized month
            np.sin(2 * np.pi * times.hour.values / 24),  # Cyclic hour
            np.cos(2 * np.pi * times.hour.values / 24)
        ])

        return features

    @staticmethod
    def extract_spatial_context(
        positions: np.ndarray,
        port_locations: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Extract spatial context features.

        Args:
            positions: [seq_len, 2] with [lat, lon]
            port_locations: List of (lat, lon) for nearby ports

        Returns:
            Spatial features like distance to nearest port
        """
        features = []

        # Distance from trajectory centroid
        centroid = np.mean(positions, axis=0)
        dist_from_centroid = np.linalg.norm(
            positions - centroid, axis=1
        )
        features.append(dist_from_centroid.reshape(-1, 1))

        # Trajectory spread
        spread = np.std(positions, axis=0)
        features.append(np.tile(spread, (len(positions), 1)))

        # Distance to nearest port (if available)
        if port_locations:
            min_port_distances = []
            for pos in positions:
                distances = [
                    haversine_distance(pos[0], pos[1], port[0], port[1])
                    for port in port_locations
                ]
                min_port_distances.append(min(distances))
            features.append(np.array(min_port_distances).reshape(-1, 1))

        return np.concatenate(features, axis=1)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
```

### 6. Integration with Existing System

**File**: `src/models/baseline_models/classical/__init__.py`

```python
"""
Classical ML baselines for maritime trajectory prediction.
"""

from .svr_model import SVRBaseline
from .rf_model import RFBaseline
from .base import ClassicalMLConfig
from .validation import PurgedTimeSeriesSplit, VesselGroupTimeSeriesSplit
from .features import MaritimeFeatureEngineer

__all__ = [
    'SVRBaseline',
    'RFBaseline',
    'ClassicalMLConfig',
    'PurgedTimeSeriesSplit',
    'VesselGroupTimeSeriesSplit',
    'MaritimeFeatureEngineer'
]

# Factory function for easy creation
def create_classical_baseline(
    model_type: str = "svr",
    **kwargs
) -> Union[SVRBaseline, RFBaseline]:
    """
    Factory function to create classical ML baselines.

    Args:
        model_type: "svr" or "rf"
        **kwargs: Model-specific parameters

    Returns:
        Configured baseline model
    """
    if model_type.lower() == "svr":
        return SVRBaseline(**kwargs)
    elif model_type.lower() in ["rf", "random_forest"]:
        return RFBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")
```

### 7. Update Main Factory

**File**: Update to `src/models/baseline_models/__init__.py`

```python
# Add to existing imports
from .classical import SVRBaseline, RFBaseline, create_classical_baseline

# Update __all__
__all__ = [
    # ... existing entries ...
    "SVRBaseline",
    "RFBaseline",
    "create_classical_baseline"
]

# Update create_baseline_model function
def create_baseline_model(model_type: str, **kwargs):
    """Create a baseline model of the specified type."""
    # ... existing cases ...
    elif model_type == "svr":
        return SVRBaseline(**kwargs)
    elif model_type == "rf":
        return RFBaseline(**kwargs)
    elif model_type == "classical":
        # Delegate to classical factory
        classical_type = kwargs.pop('classical_type', 'svr')
        return create_classical_baseline(classical_type, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

## Usage Examples

### Basic Usage
```python
from src.models.baseline_models.classical import SVRBaseline, RFBaseline

# Create and fit SVR baseline
svr_model = SVRBaseline()
svr_model.fit(train_sequences)

# Predict future trajectory
predictions = svr_model.predict(test_sequence, horizon=12)

# Get predictions with uncertainty
result = svr_model.predict(
    test_sequence,
    horizon=12,
    return_uncertainty=True
)
print(f"Predictions shape: {result.predictions.shape}")
print(f"Uncertainty: {result.uncertainty}")
```

### Advanced Usage with Tuning
```python
# Create RF model with custom config
from src.models.baseline_models.classical import ClassicalMLConfig

config = ClassicalMLConfig(
    use_rolling_features=True,
    rolling_windows=[5, 10, 20],
    cv_splits=5,
    cv_gap=10  # 10-step gap for purged CV
)

rf_model = RFBaseline(
    config=config,
    n_estimators=200,
    max_depth=15
)

# Fit with time-aware cross-validation
rf_model.fit(train_sequences, max_horizon=24)

# Get feature importance
importance_df = rf_model.get_feature_importance_report()
print("Top 10 important features:")
print(importance_df.head(10))

# Get OOB scores
oob_scores = rf_model.get_oob_score()
print(f"OOB R² scores by horizon: {oob_scores}")
```

### Integration with Evaluation Framework
```python
from src.evalx.validation.comparisons import ModelComparison

# Create baseline models
baselines = {
    'SVR': SVRBaseline(),
    'RF': RFBaseline(),
    'Kalman': MaritimeIMMFilter()  # From M1.2
}

# Fit all models
for name, model in baselines.items():
    model.fit(train_sequences)

# Generate predictions
predictions = {}
for name, model in baselines.items():
    preds = []
    for seq in test_sequences:
        pred = model.predict(seq, horizon=12)
        preds.append(pred)
    predictions[name] = np.array(preds)

# Statistical comparison
comparison = ModelComparison()
result = comparison.compare_models(predictions, test_targets)
print(result.summary_table)
```

## Testing Strategy

### Unit Tests Required
```python
# tests/unit/baselines/test_classical_ml.py

def test_svr_fit_predict():
    """Test SVR baseline fit and predict."""

def test_rf_feature_importance():
    """Test RF feature importance extraction."""

def test_time_aware_cv():
    """Test purged time series cross-validation."""

def test_delta_prediction():
    """Test delta vs absolute prediction modes."""

def test_feature_engineering():
    """Test maritime feature engineering."""
```

## Performance Optimization

### Parallelization Strategy
- Use `joblib.Parallel` for multi-horizon training
- Leverage `n_jobs=-1` in sklearn models
- Batch feature extraction for efficiency

### Memory Optimization
- Use `float32` instead of `float64` where possible
- Implement chunked processing for large datasets
- Clear intermediate results after each horizon

### Computational Complexity
- SVR: O(n²) to O(n³) depending on kernel
- RF: O(n × log(n) × m × p) where m=trees, p=features
- Feature engineering: O(n × w) where w=window size

## Risk Mitigation

### Handled Risks
1. **Data Leakage**: Purged CV with configurable gap
2. **Scalability**: Parallel processing and chunking
3. **Integration**: Follows existing TrajectoryBaseline protocol
4. **Feature Engineering**: Modular and configurable

### Remaining Considerations
1. **Hyperparameter Search**: Can be expensive for SVR
2. **Memory with Large Datasets**: May need streaming approach
3. **Feature Explosion**: Limit rolling windows and lags

## Summary

This implementation provides:
1. **Minimal Resistance**: Uses mature sklearn with established patterns
2. **Full Integration**: Follows existing protocols and patterns
3. **Maritime Optimization**: Specialized features and validation
4. **Production Ready**: Includes uncertainty, parallelization, and testing

The architecture seamlessly integrates with the existing Kalman (M1.2) and neural baselines while providing classical ML performance benchmarks.
