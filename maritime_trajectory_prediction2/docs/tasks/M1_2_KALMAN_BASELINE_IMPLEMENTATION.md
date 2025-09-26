# M1.2 Kalman Filter Baseline Implementation Specification

## Executive Summary

This document provides the complete implementation specification for **M1.2: Kalman Filter Baseline**, a physics-based trajectory prediction system for maritime applications. The implementation follows the "never reinvent the wheel" philosophy, leveraging proven libraries while adding maritime domain expertise.

## Scientific Rationale

### Why Kalman Filters for Maritime Baselines

Kalman filters are essential baselines for maritime trajectory prediction because they:

1. **Physics-Based**: Incorporate actual vessel dynamics (constant velocity, coordinated turns)
2. **Uncertainty Quantification**: Provide prediction uncertainty estimates natively
3. **Real-Time Capable**: Computational efficiency suitable for operational systems
4. **Interpretable**: Motion model parameters have physical meaning
5. **Proven Track Record**: Decades of successful use in maritime tracking systems

### Interactive Multiple Model (IMM) Approach

Maritime vessels exhibit **multiple motion modes**:
- **Constant Velocity (CV)**: Straight-line navigation
- **Coordinated Turn (CT)**: Planned course changes
- **Nearly Constant Acceleration (NCA)**: Speed changes during maneuvering

IMM framework automatically selects and blends these models based on observed motion patterns.

## Technical Architecture

### Repository Integration

```
repo/
├── src/baselines/
│   ├── __init__.py
│   ├── base.py                    # TrajectoryBaseline protocol
│   ├── kalman/
│   │   ├── __init__.py
│   │   ├── imm.py                 # IMM framework
│   │   ├── models.py              # CV, CT, NCA models
│   │   ├── maritime.py            # Maritime-specific adaptations
│   │   └── tuning.py              # Hyperparameter optimization
│   └── evaluation.py             # Baseline evaluation utilities
├── tests/unit/baselines/
│   ├── __init__.py
│   ├── test_kalman_models.py
│   ├── test_imm_framework.py
│   └── test_maritime_adaptations.py
└── examples/baselines/
    └── kalman_usage_example.py
```

## Library Dependencies and Endpoints

### Primary Libraries

#### 1. FilterPy (Primary Kalman Implementation)
```python
# Installation: pip install filterpy
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise, kinematic_kf
```

**Key Endpoints:**
- `KalmanFilter`: Core Kalman filter implementation
- `IMMEstimator`: Interactive Multiple Model framework
- `Q_discrete_white_noise`: Process noise matrix generation
- `kinematic_kf`: Helper for kinematic motion models

**Why FilterPy**: Mature, well-tested, actively maintained library specifically designed for tracking applications.

#### 2. SciPy (Optimization)
```python
from scipy.optimize import minimize, differential_evolution
from scipy.stats import multivariate_normal
```

**Key Endpoints:**
- `minimize`: Local optimization for parameter tuning
- `differential_evolution`: Global optimization for noise parameters
- `multivariate_normal`: For likelihood calculations

#### 3. Scikit-Learn (Model Selection)
```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
```

**Key Endpoints:**
- `GridSearchCV`: Hyperparameter grid search
- `TimeSeriesSplit`: Time-aware cross-validation
- Custom scoring functions for maritime metrics

### Supporting Libraries
```python
import numpy as np
import pandas as pd
from typing import Protocol, Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
```

## Implementation Specification

### 1. Base Protocol Definition

**File**: `src/baselines/base.py`

```python
"""Base protocol for trajectory prediction baselines."""

from typing import Protocol, Dict, Optional, Union
import numpy as np
import pandas as pd
from abc import abstractmethod

class TrajectoryBaseline(Protocol):
    """Protocol that all baseline models must implement."""

    @abstractmethod
    def fit(self,
            sequences: Union[np.ndarray, pd.DataFrame],
            metadata: Optional[Dict] = None) -> 'TrajectoryBaseline':
        """
        Fit the model to training sequences.

        Args:
            sequences: Training trajectory sequences
                Shape: [n_sequences, seq_len, n_features]
                Features: [lat, lon, sog, cog, heading, turn_rate, timestamp]
            metadata: Optional metadata (vessel types, environmental conditions)

        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def predict(self,
                sequence: np.ndarray,
                horizon: int,
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict future trajectory points.

        Args:
            sequence: Input trajectory sequence
                Shape: [seq_len, n_features]
            horizon: Number of future steps to predict
            return_uncertainty: Whether to return prediction uncertainty

        Returns:
            predictions: Predicted trajectory points [horizon, n_features]
            uncertainties: If requested, prediction covariances [horizon, n_features, n_features]
        """
        ...

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model information for logging and analysis."""
        ...

@dataclass
class BaselineResult:
    """Standardized result format for baseline predictions."""

    predictions: np.ndarray              # [horizon, n_features]
    uncertainties: Optional[np.ndarray]   # [horizon, n_features, n_features]
    model_probabilities: Optional[np.ndarray] # [n_models] for IMM
    metadata: Dict
```

### 2. Motion Models Implementation

**File**: `src/baselines/kalman/models.py`

```python
"""Maritime motion models for Kalman filtering."""

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class MotionModelConfig:
    """Configuration for motion models."""

    process_noise_std: float = 1.0      # Process noise standard deviation
    measurement_noise_std: float = 0.5   # Measurement noise standard deviation
    dt: float = 60.0                     # Time step in seconds


class ConstantVelocityModel:
    """Constant Velocity model for straight-line maritime motion."""

    def __init__(self, config: MotionModelConfig):
        self.config = config
        self.kf = self._create_filter()

    def _create_filter(self) -> KalmanFilter:
        """Create CV Kalman filter using FilterPy."""
        # State: [x, x_dot, y, y_dot] (position and velocity in lat/lon)
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity)
        dt = self.config.dt
        kf.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ])

        # Measurement function (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # Process noise covariance
        kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=self.config.process_noise_std**2, block_size=2
        )

        # Measurement noise covariance
        kf.R *= self.config.measurement_noise_std**2

        # Initial state covariance
        kf.P *= 100.0  # High initial uncertainty

        return kf

    def predict_step(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict one step forward."""
        if dt and dt != self.config.dt:
            self._update_dt(dt)

        self.kf.predict()
        return self.kf.x.copy(), self.kf.P.copy()

    def update_step(self, measurement: np.ndarray):
        """Update with new measurement."""
        self.kf.update(measurement)

    def _update_dt(self, dt: float):
        """Update time step for variable dt."""
        self.kf.F[0, 1] = dt
        self.kf.F[2, 3] = dt
        self.kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=self.config.process_noise_std**2, block_size=2
        )


class CoordinatedTurnModel:
    """Coordinated Turn model for maneuvering maritime vessels."""

    def __init__(self, config: MotionModelConfig, turn_rate_std: float = 0.1):
        self.config = config
        self.turn_rate_std = turn_rate_std
        self.kf = self._create_filter()

    def _create_filter(self) -> KalmanFilter:
        """Create CT Kalman filter."""
        # State: [x, x_dot, y, y_dot, omega] (position, velocity, turn rate)
        kf = KalmanFilter(dim_x=5, dim_z=2)

        # State transition matrix (coordinated turn)
        dt = self.config.dt
        kf.F = self._create_ct_transition_matrix(dt, omega=0.0)  # Will be updated dynamically

        # Measurement function (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])

        # Process noise covariance
        kf.Q = self._create_ct_process_noise(dt)

        # Measurement noise covariance
        kf.R *= self.config.measurement_noise_std**2

        # Initial state covariance
        kf.P *= 100.0

        return kf

    def _create_ct_transition_matrix(self, dt: float, omega: float) -> np.ndarray:
        """Create coordinated turn state transition matrix."""
        if abs(omega) < 1e-6:  # Nearly zero turn rate -> constant velocity
            return np.array([
                [1, dt, 0,  0, 0],
                [0,  1, 0,  0, 0],
                [0,  0, 1, dt, 0],
                [0,  0, 0,  1, 0],
                [0,  0, 0,  0, 1]
            ])

        s, c = np.sin(omega * dt), np.cos(omega * dt)

        return np.array([
            [1, s/omega,         0, (c-1)/omega, 0],
            [0,      c,         0,         s,    0],
            [0, (1-c)/omega,    1,     s/omega, 0],
            [0,        -s,      0,         c,    0],
            [0,         0,      0,         0,    1]
        ])

    def _create_ct_process_noise(self, dt: float) -> np.ndarray:
        """Create process noise matrix for coordinated turn."""
        # Simplified process noise
        q_pos = self.config.process_noise_std**2
        q_vel = self.config.process_noise_std**2
        q_omega = self.turn_rate_std**2

        return np.diag([q_pos, q_vel, q_pos, q_vel, q_omega])

    def predict_step(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict one step with current turn rate."""
        if dt is None:
            dt = self.config.dt

        # Update transition matrix with current turn rate estimate
        omega = self.kf.x[4]  # Current turn rate estimate
        self.kf.F = self._create_ct_transition_matrix(dt, omega)

        self.kf.predict()
        return self.kf.x.copy(), self.kf.P.copy()

    def update_step(self, measurement: np.ndarray):
        """Update with new measurement."""
        self.kf.update(measurement)


class NearlyConstantAccelerationModel:
    """Nearly Constant Acceleration model for maritime vessels."""

    def __init__(self, config: MotionModelConfig):
        self.config = config
        self.kf = self._create_filter()

    def _create_filter(self) -> KalmanFilter:
        """Create NCA Kalman filter."""
        # State: [x, x_dot, x_ddot, y, y_dot, y_ddot] (position, velocity, acceleration)
        kf = KalmanFilter(dim_x=6, dim_z=2)

        # State transition matrix (constant acceleration)
        dt = self.config.dt
        kf.F = np.array([
            [1, dt, dt**2/2,  0,  0,       0],
            [0,  1,      dt,  0,  0,       0],
            [0,  0,       1,  0,  0,       0],
            [0,  0,       0,  1, dt, dt**2/2],
            [0,  0,       0,  0,  1,      dt],
            [0,  0,       0,  0,  0,       1]
        ])

        # Measurement function (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # Process noise covariance
        kf.Q = Q_discrete_white_noise(
            dim=3, dt=dt, var=self.config.process_noise_std**2, block_size=2
        )

        # Measurement noise covariance
        kf.R *= self.config.measurement_noise_std**2

        # Initial state covariance
        kf.P *= 100.0

        return kf

    def predict_step(self, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict one step forward."""
        if dt and dt != self.config.dt:
            self._update_dt(dt)

        self.kf.predict()
        return self.kf.x.copy(), self.kf.P.copy()

    def update_step(self, measurement: np.ndarray):
        """Update with new measurement."""
        self.kf.update(measurement)

    def _update_dt(self, dt: float):
        """Update time step for variable dt."""
        self.kf.F[0, 1] = dt
        self.kf.F[0, 2] = dt**2/2
        self.kf.F[1, 2] = dt
        self.kf.F[3, 4] = dt
        self.kf.F[3, 5] = dt**2/2
        self.kf.F[4, 5] = dt
        self.kf.Q = Q_discrete_white_noise(
            dim=3, dt=dt, var=self.config.process_noise_std**2, block_size=2
        )


def create_motion_models(config: MotionModelConfig) -> Dict[str, object]:
    """Factory function to create all motion models."""
    return {
        'cv': ConstantVelocityModel(config),
        'ct': CoordinatedTurnModel(config),
        'nca': NearlyConstantAccelerationModel(config)
    }
```

### 3. IMM Framework Implementation

**File**: `src/baselines/kalman/imm.py`

```python
"""Interactive Multiple Model implementation for maritime trajectories."""

import numpy as np
from filterpy.kalman import IMMEstimator
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

from .models import create_motion_models, MotionModelConfig, BaselineResult
from ..base import TrajectoryBaseline


@dataclass
class IMMConfig:
    """Configuration for IMM estimator."""

    model_weights: List[float] = None  # Initial model probabilities
    markov_chain: Optional[np.ndarray] = None  # Transition probabilities
    motion_config: MotionModelConfig = None

    def __post_init__(self):
        if self.model_weights is None:
            self.model_weights = [0.33, 0.33, 0.34]  # Equal weights for CV, CT, NCA

        if self.markov_chain is None:
            # Default Markov chain: prefer staying in same model
            self.markov_chain = np.array([
                [0.97, 0.02, 0.01],  # From CV
                [0.03, 0.94, 0.03],  # From CT
                [0.03, 0.03, 0.94]   # From NCA
            ])

        if self.motion_config is None:
            self.motion_config = MotionModelConfig()


class MaritimeIMMFilter(TrajectoryBaseline):
    """
    Interactive Multiple Model filter for maritime trajectory prediction.

    Combines Constant Velocity, Coordinated Turn, and Nearly Constant Acceleration
    models to handle diverse vessel motion patterns.
    """

    def __init__(self, config: IMMConfig = None):
        """Initialize IMM filter with maritime-specific motion models."""
        self.config = config or IMMConfig()
        self.models = create_motion_models(self.config.motion_config)
        self.imm = self._create_imm_estimator()
        self.is_fitted = False

        # Store filtering history for analysis
        self.filter_history = {
            'states': [],
            'covariances': [],
            'model_probs': [],
            'likelihoods': []
        }

    def _create_imm_estimator(self) -> IMMEstimator:
        """Create IMM estimator using FilterPy."""
        filters = [model.kf for model in self.models.values()]

        imm = IMMEstimator(
            filters=filters,
            mu=self.config.model_weights,
            M=self.config.markov_chain
        )

        return imm

    def fit(self,
            sequences: np.ndarray,
            metadata: Optional[Dict] = None) -> 'MaritimeIMMFilter':
        """
        Fit IMM filter by tuning noise parameters on training sequences.

        Args:
            sequences: Training trajectories [n_sequences, seq_len, features]
                      Features: [lat, lon, sog, cog, heading, turn_rate, timestamp]
            metadata: Optional metadata dict

        Returns:
            Self for method chaining
        """
        from .tuning import IMMTuner

        # Use hyperparameter tuning to optimize noise parameters
        tuner = IMMTuner(self.config)
        optimal_params = tuner.tune(sequences, metadata)

        # Update configuration with optimal parameters
        self.config.motion_config = optimal_params

        # Recreate models and IMM with tuned parameters
        self.models = create_motion_models(self.config.motion_config)
        self.imm = self._create_imm_estimator()

        self.is_fitted = True
        return self

    def predict(self,
                sequence: np.ndarray,
                horizon: int,
                return_uncertainty: bool = False) -> BaselineResult:
        """
        Predict future trajectory using IMM filter.

        Args:
            sequence: Input sequence [seq_len, features]
            horizon: Prediction horizon (number of steps)
            return_uncertainty: Whether to return prediction covariances

        Returns:
            BaselineResult with predictions and metadata
        """
        if not self.is_fitted:
            warnings.warn("IMM filter not fitted. Using default parameters.")

        # Initialize filter with sequence
        self._initialize_from_sequence(sequence)

        # Run filter on observed sequence
        self._filter_sequence(sequence)

        # Predict future points
        predictions, uncertainties = self._predict_horizon(horizon, return_uncertainty)

        # Get final model probabilities
        model_probs = self.imm.mu.copy() if hasattr(self.imm, 'mu') else None

        return BaselineResult(
            predictions=predictions,
            uncertainties=uncertainties if return_uncertainty else None,
            model_probabilities=model_probs,
            metadata={
                'model_names': list(self.models.keys()),
                'final_model_probs': model_probs,
                'dominant_model': self._get_dominant_model(),
                'filter_type': 'IMM',
                'horizon': horizon
            }
        )

    def _initialize_from_sequence(self, sequence: np.ndarray):
        """Initialize filter state from first few points of sequence."""
        if len(sequence) < 2:
            raise ValueError("Need at least 2 points to initialize IMM filter")

        # Extract position data (lat, lon)
        pos1, pos2 = sequence[0, :2], sequence[1, :2]

        # Convert to Cartesian coordinates for filtering
        # (In real implementation, would use proper coordinate transformation)
        x1, y1 = self._latlon_to_cartesian(pos1[0], pos1[1])
        x2, y2 = self._latlon_to_cartesian(pos2[0], pos2[1])

        # Estimate initial velocity
        dt = self._get_time_delta(sequence)
        vx, vy = (x2 - x1) / dt, (y2 - y1) / dt

        # Initialize each model
        for model_name, model in self.models.items():
            if model_name == 'cv':
                model.kf.x = np.array([x1, vx, y1, vy])
            elif model_name == 'ct':
                model.kf.x = np.array([x1, vx, y1, vy, 0.0])  # Zero initial turn rate
            elif model_name == 'nca':
                model.kf.x = np.array([x1, vx, 0.0, y1, vy, 0.0])  # Zero initial acceleration

        # Reset history
        self.filter_history = {k: [] for k in self.filter_history.keys()}

    def _filter_sequence(self, sequence: np.ndarray):
        """Run IMM filter on observed sequence."""
        for i in range(1, len(sequence)):  # Skip first point (used for initialization)
            # Convert measurement to Cartesian
            lat, lon = sequence[i, :2]
            x, y = self._latlon_to_cartesian(lat, lon)
            measurement = np.array([x, y])

            # IMM predict-update cycle
            self.imm.predict()
            self.imm.update(measurement)

            # Store history
            self.filter_history['states'].append(self.imm.x.copy())
            self.filter_history['covariances'].append(self.imm.P.copy())
            self.filter_history['model_probs'].append(self.imm.mu.copy())
            if hasattr(self.imm, 'likelihood'):
                self.filter_history['likelihoods'].append(self.imm.likelihood.copy())

    def _predict_horizon(self, horizon: int, return_uncertainty: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict future trajectory points."""
        predictions = []
        uncertainties = [] if return_uncertainty else None

        for _ in range(horizon):
            # IMM prediction step
            self.imm.predict()

            # Extract position prediction
            x_pred, y_pred = self.imm.x[0], self.imm.x[2]  # Assumes [x, vx, y, vy, ...] state
            lat_pred, lon_pred = self._cartesian_to_latlon(x_pred, y_pred)

            # Create prediction with additional features (SOG, COG)
            sog_pred, cog_pred = self._extract_velocity_features(self.imm.x)
            pred_point = np.array([lat_pred, lon_pred, sog_pred, cog_pred])
            predictions.append(pred_point)

            if return_uncertainty:
                # Extract position covariance
                pos_cov = self._extract_position_covariance(self.imm.P)
                uncertainties.append(pos_cov)

        return np.array(predictions), np.array(uncertainties) if return_uncertainty else None

    def _get_dominant_model(self) -> str:
        """Get name of currently dominant model."""
        if not hasattr(self.imm, 'mu'):
            return 'unknown'

        model_names = list(self.models.keys())
        dominant_idx = np.argmax(self.imm.mu)
        return model_names[dominant_idx]

    def _latlon_to_cartesian(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert lat/lon to local Cartesian coordinates."""
        # Simplified conversion - in practice would use proper projection
        # This assumes small area where Earth curvature is negligible
        x = lon * 111320.0 * np.cos(np.radians(lat))  # meters
        y = lat * 110540.0  # meters
        return x, y

    def _cartesian_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local Cartesian back to lat/lon."""
        lat = y / 110540.0
        lon = x / (111320.0 * np.cos(np.radians(lat)))
        return lat, lon

    def _get_time_delta(self, sequence: np.ndarray) -> float:
        """Extract time delta from sequence."""
        if sequence.shape[1] > 6:  # Has timestamp
            return float(sequence[1, 6] - sequence[0, 6])  # Assumes timestamp in seconds
        else:
            return self.config.motion_config.dt  # Use default

    def _extract_velocity_features(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract SOG and COG from filter state."""
        # Extract velocity components
        vx, vy = state[1], state[3]  # Assumes [x, vx, y, vy, ...] format

        # Calculate speed over ground (SOG) in knots
        sog = np.sqrt(vx**2 + vy**2) * 1.94384  # m/s to knots

        # Calculate course over ground (COG) in degrees
        cog = np.degrees(np.arctan2(vx, vy)) % 360

        return float(sog), float(cog)

    def _extract_position_covariance(self, P: np.ndarray) -> np.ndarray:
        """Extract position covariance from full state covariance."""
        # Extract [x, y] covariance submatrix
        pos_indices = [0, 2]  # x and y positions
        return P[np.ix_(pos_indices, pos_indices)]

    def get_model_info(self) -> Dict:
        """Return model information for logging."""
        return {
            'type': 'MaritimeIMM',
            'models': list(self.models.keys()),
            'config': {
                'model_weights': self.config.model_weights,
                'markov_chain': self.config.markov_chain.tolist(),
                'process_noise_std': self.config.motion_config.process_noise_std,
                'measurement_noise_std': self.config.motion_config.measurement_noise_std,
                'dt': self.config.motion_config.dt
            },
            'is_fitted': self.is_fitted,
            'dominant_model': self._get_dominant_model() if hasattr(self.imm, 'mu') else None
        }
```

### 4. Maritime-Specific Adaptations

**File**: `src/baselines/kalman/maritime.py`

```python
"""Maritime-specific adaptations for Kalman filtering."""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MaritimeConstraints:
    """Maritime-specific constraints for vessel motion."""

    max_speed_knots: float = 50.0        # Maximum vessel speed
    max_acceleration: float = 0.5        # Maximum acceleration (m/s²)
    max_turn_rate: float = 5.0           # Maximum turn rate (deg/s)
    min_turn_radius: float = 50.0        # Minimum turn radius (m)


class MaritimeDynamicsValidator:
    """Validates and constrains predictions to maritime physics."""

    def __init__(self, constraints: MaritimeConstraints = None):
        self.constraints = constraints or MaritimeConstraints()

    def validate_prediction(self,
                          prediction: np.ndarray,
                          previous_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate and constrain prediction to maritime physics.

        Args:
            prediction: Predicted state [lat, lon, sog, cog]
            previous_state: Previous state for consistency checking

        Returns:
            Constrained prediction
        """
        constrained = prediction.copy()

        # Speed constraints
        sog = constrained[2]  # Speed over ground
        if sog > self.constraints.max_speed_knots:
            constrained[2] = self.constraints.max_speed_knots
            warnings.warn(f"Speed constrained from {sog:.2f} to {self.constraints.max_speed_knots} knots")
        elif sog < 0:
            constrained[2] = 0.0

        # Course constraints (ensure 0-360 degrees)
        constrained[3] = constrained[3] % 360

        # Acceleration constraints (if previous state available)
        if previous_state is not None:
            dt = 60.0  # Assume 1-minute time step
            constrained = self._apply_acceleration_constraints(
                constrained, previous_state, dt
            )

        return constrained

    def _apply_acceleration_constraints(self,
                                      current: np.ndarray,
                                      previous: np.ndarray,
                                      dt: float) -> np.ndarray:
        """Apply acceleration constraints."""
        # Convert speeds to m/s
        v_curr = current[2] * 0.514444  # knots to m/s
        v_prev = previous[2] * 0.514444

        # Calculate acceleration
        acceleration = (v_curr - v_prev) / dt

        # Constrain acceleration
        if abs(acceleration) > self.constraints.max_acceleration:
            sign = np.sign(acceleration)
            max_acc = sign * self.constraints.max_acceleration
            v_constrained = v_prev + max_acc * dt
            current[2] = v_constrained / 0.514444  # Back to knots

        return current


class CoordinateTransformer:
    """Handles coordinate transformations for maritime applications."""

    def __init__(self, reference_point: Optional[Tuple[float, float]] = None):
        """
        Initialize coordinate transformer.

        Args:
            reference_point: (lat, lon) reference for local coordinate system
        """
        self.reference_point = reference_point

    def set_reference_point(self, lat: float, lon: float):
        """Set reference point for local coordinate system."""
        self.reference_point = (lat, lon)

    def latlon_to_local_cartesian(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert lat/lon to local Cartesian coordinates.

        Uses simple equirectangular projection suitable for small areas.
        For large areas, would use proper map projection (e.g., UTM).
        """
        if self.reference_point is None:
            self.reference_point = (lat, lon)  # Use first point as reference

        ref_lat, ref_lon = self.reference_point

        # Convert to meters using equirectangular projection
        x = (lon - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * 110540.0

        return x, y

    def local_cartesian_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local Cartesian back to lat/lon."""
        if self.reference_point is None:
            raise ValueError("Reference point not set")

        ref_lat, ref_lon = self.reference_point

        # Convert back to degrees
        lat = ref_lat + y / 110540.0
        lon = ref_lon + x / (111320.0 * np.cos(np.radians(ref_lat)))

        return lat, lon

    def calculate_bearing_and_distance(self,
                                     lat1: float, lon1: float,
                                     lat2: float, lon2: float) -> Tuple[float, float]:
        """
        Calculate bearing and distance between two points using haversine formula.

        Returns:
            bearing: Bearing in degrees (0-360)
            distance: Distance in meters
        """
        # Convert to radians
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

        # Haversine distance
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2)
        distance = 2 * 6371000 * np.arcsin(np.sqrt(a))  # Earth radius in meters

        # Calculate bearing
        y = np.sin(dlon) * np.cos(lat2_r)
        x = (np.cos(lat1_r) * np.sin(lat2_r) -
             np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))
        bearing = np.degrees(np.arctan2(y, x)) % 360

        return bearing, distance


class MaritimeFeatureExtractor:
    """Extracts maritime-specific features from trajectory data."""

    @staticmethod
    def extract_turn_rate(sequence: np.ndarray) -> np.ndarray:
        """
        Extract turn rate from COG sequence.

        Args:
            sequence: Trajectory with COG in column 3

        Returns:
            Turn rates in degrees per time step
        """
        if len(sequence) < 2:
            return np.array([])

        cog = sequence[:, 3]  # Course over ground

        # Calculate angular differences, handling wraparound
        diff = np.diff(cog)
        diff = np.where(diff > 180, diff - 360, diff)
        diff = np.where(diff < -180, diff + 360, diff)

        return diff

    @staticmethod
    def extract_acceleration(sequence: np.ndarray, dt: float = 60.0) -> np.ndarray:
        """
        Extract speed acceleration from SOG sequence.

        Args:
            sequence: Trajectory with SOG in column 2
            dt: Time step in seconds

        Returns:
            Accelerations in knots per second
        """
        if len(sequence) < 2:
            return np.array([])

        sog = sequence[:, 2]  # Speed over ground
        acceleration = np.diff(sog) / dt

        return acceleration

    @staticmethod
    def detect_motion_mode(sequence: np.ndarray,
                          threshold_turn: float = 1.0,
                          threshold_accel: float = 0.1) -> str:
        """
        Detect predominant motion mode from trajectory sequence.

        Args:
            sequence: Trajectory sequence
            threshold_turn: Turn rate threshold for CT detection (deg/step)
            threshold_accel: Acceleration threshold for NCA detection (knots/s)

        Returns:
            Detected motion mode: 'cv', 'ct', or 'nca'
        """
        if len(sequence) < 3:
            return 'cv'  # Default to constant velocity

        # Extract motion features
        turn_rates = MaritimeFeatureExtractor.extract_turn_rate(sequence)
        accelerations = MaritimeFeatureExtractor.extract_acceleration(sequence)

        # Calculate statistics
        mean_turn_rate = np.mean(np.abs(turn_rates)) if len(turn_rates) > 0 else 0
        mean_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0

        # Classify motion mode
        if mean_turn_rate > threshold_turn:
            return 'ct'  # Coordinated turn
        elif mean_acceleration > threshold_accel:
            return 'nca'  # Nearly constant acceleration
        else:
            return 'cv'  # Constant velocity
```

### 5. Hyperparameter Tuning System

**File**: `src/baselines/kalman/tuning.py`

```python
"""Hyperparameter tuning for Maritime IMM filter."""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import warnings

from .models import MotionModelConfig
from .imm import IMMConfig, MaritimeIMMFilter
from .maritime import CoordinateTransformer


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    # Parameter bounds for optimization
    process_noise_bounds: Tuple[float, float] = (0.1, 10.0)
    measurement_noise_bounds: Tuple[float, float] = (0.01, 2.0)

    # Optimization settings
    n_trials: int = 50
    cv_folds: int = 3
    test_split: float = 0.2

    # Scoring function
    primary_metric: str = 'mse'  # 'mse', 'mae', 'ade'

    # Optimization method
    method: str = 'differential_evolution'  # 'minimize', 'differential_evolution'


class IMMTuner:
    """Hyperparameter tuner for Maritime IMM filter."""

    def __init__(self, imm_config: IMMConfig, tuning_config: TuningConfig = None):
        self.imm_config = imm_config
        self.tuning_config = tuning_config or TuningConfig()

        self.best_params = None
        self.best_score = np.inf
        self.tuning_history = []

    def tune(self,
             sequences: np.ndarray,
             metadata: Optional[Dict] = None) -> MotionModelConfig:
        """
        Tune IMM hyperparameters using cross-validation.

        Args:
            sequences: Training sequences [n_sequences, seq_len, features]
            metadata: Optional metadata

        Returns:
            Optimized MotionModelConfig
        """
        print(f"Tuning IMM hyperparameters with {len(sequences)} sequences...")

        # Prepare objective function
        objective = self._create_objective_function(sequences, metadata)

        # Define parameter bounds
        bounds = [
            self.tuning_config.process_noise_bounds,
            self.tuning_config.measurement_noise_bounds
        ]

        # Run optimization
        if self.tuning_config.method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.tuning_config.n_trials,
                seed=42,
                workers=1  # Avoid multiprocessing issues with Kalman filters
            )
        elif self.tuning_config.method == 'minimize':
            # Use multiple random starts for robustness
            best_result = None
            for _ in range(5):
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds
                )
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            result = best_result
        else:
            raise ValueError(f"Unknown optimization method: {self.tuning_config.method}")

        # Extract best parameters
        process_noise_std, measurement_noise_std = result.x

        # Create optimized config
        optimized_config = MotionModelConfig(
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
            dt=self.imm_config.motion_config.dt
        )

        print(f"Tuning completed. Best score: {result.fun:.6f}")
        print(f"Optimal parameters: process_noise={process_noise_std:.4f}, "
              f"measurement_noise={measurement_noise_std:.4f}")

        self.best_params = optimized_config
        self.best_score = result.fun

        return optimized_config

    def _create_objective_function(self,
                                  sequences: np.ndarray,
                                  metadata: Optional[Dict]) -> Callable:
        """Create objective function for optimization."""

        def objective(params: List[float]) -> float:
            """
            Objective function to minimize.

            Args:
                params: [process_noise_std, measurement_noise_std]

            Returns:
                Cross-validation score (lower is better)
            """
            try:
                process_noise_std, measurement_noise_std = params

                # Create config with current parameters
                motion_config = MotionModelConfig(
                    process_noise_std=process_noise_std,
                    measurement_noise_std=measurement_noise_std,
                    dt=self.imm_config.motion_config.dt
                )

                imm_config = IMMConfig(
                    model_weights=self.imm_config.model_weights,
                    markov_chain=self.imm_config.markov_chain,
                    motion_config=motion_config
                )

                # Perform cross-validation
                scores = self._cross_validate(sequences, imm_config)

                # Return mean score
                mean_score = np.mean(scores)

                # Store in history
                self.tuning_history.append({
                    'params': params,
                    'scores': scores,
                    'mean_score': mean_score
                })

                return mean_score

            except Exception as e:
                warnings.warn(f"Objective function evaluation failed: {e}")
                return np.inf  # Return high penalty for failed evaluations

        return objective

    def _cross_validate(self,
                       sequences: np.ndarray,
                       imm_config: IMMConfig) -> List[float]:
        """Perform time-series cross-validation."""
        scores = []

        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=self.tuning_config.cv_folds)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
            try:
                train_seqs = sequences[train_idx]
                val_seqs = sequences[val_idx]

                # Create and fit IMM filter (no additional fitting needed for Kalman)
                imm = MaritimeIMMFilter(imm_config)

                # Evaluate on validation sequences
                fold_score = self._evaluate_sequences(imm, val_seqs)
                scores.append(fold_score)

            except Exception as e:
                warnings.warn(f"Cross-validation fold {fold} failed: {e}")
                scores.append(np.inf)

        return scores

    def _evaluate_sequences(self,
                           imm: MaritimeIMMFilter,
                           sequences: np.ndarray) -> float:
        """Evaluate IMM filter on a set of sequences."""
        all_errors = []

        for seq in sequences:
            try:
                if len(seq) < 5:  # Need minimum sequence length
                    continue

                # Use first part for observation, last part for prediction
                split_point = len(seq) - 2
                observed = seq[:split_point]
                true_future = seq[split_point:]

                # Make prediction
                horizon = len(true_future)
                result = imm.predict(observed, horizon)

                # Calculate error based on primary metric
                error = self._calculate_error(
                    result.predictions,
                    true_future,
                    self.tuning_config.primary_metric
                )

                all_errors.append(error)

            except Exception as e:
                warnings.warn(f"Sequence evaluation failed: {e}")
                continue

        return np.mean(all_errors) if all_errors else np.inf

    def _calculate_error(self,
                        predictions: np.ndarray,
                        truth: np.ndarray,
                        metric: str) -> float:
        """Calculate prediction error using specified metric."""
        if len(predictions) == 0 or len(truth) == 0:
            return np.inf

        # Align lengths
        min_len = min(len(predictions), len(truth))
        pred_aligned = predictions[:min_len]
        truth_aligned = truth[:min_len]

        if metric == 'mse':
            # Position MSE (lat, lon)
            pos_pred = pred_aligned[:, :2]
            pos_truth = truth_aligned[:, :2]
            return mean_squared_error(pos_truth.flatten(), pos_pred.flatten())

        elif metric == 'mae':
            # Position MAE
            pos_pred = pred_aligned[:, :2]
            pos_truth = truth_aligned[:, :2]
            return np.mean(np.abs(pos_truth - pos_pred))

        elif metric == 'ade':
            # Average Displacement Error using haversine distance
            total_distance = 0.0
            count = 0

            transformer = CoordinateTransformer()

            for i in range(len(pred_aligned)):
                lat_pred, lon_pred = pred_aligned[i, 0], pred_aligned[i, 1]
                lat_true, lon_true = truth_aligned[i, 0], truth_aligned[i, 1]

                _, distance = transformer.calculate_bearing_and_distance(
                    lat_pred, lon_pred, lat_true, lon_true
                )

                total_distance += distance
                count += 1

            return total_distance / count if count > 0 else np.inf

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_tuning_summary(self) -> Dict:
        """Get summary of tuning process."""
        if not self.tuning_history:
            return {"status": "No tuning performed"}

        best_trial = min(self.tuning_history, key=lambda x: x['mean_score'])

        return {
            "best_score": self.best_score,
            "best_params": asdict(self.best_params) if self.best_params else None,
            "n_trials": len(self.tuning_history),
            "best_trial": best_trial,
            "tuning_config": asdict(self.tuning_config)
        }


def tune_imm_for_region(sequences: np.ndarray,
                       region_name: str = "default",
                       save_results: bool = True) -> MotionModelConfig:
    """
    Convenience function to tune IMM for a specific maritime region.

    Args:
        sequences: Training sequences for the region
        region_name: Name of the maritime region
        save_results: Whether to save tuning results

    Returns:
        Tuned MotionModelConfig
    """
    # Create default configs
    imm_config = IMMConfig()
    tuning_config = TuningConfig(n_trials=100, cv_folds=5)

    # Run tuning
    tuner = IMMTuner(imm_config, tuning_config)
    optimized_config = tuner.tune(sequences)

    if save_results:
        # Save results for later analysis
        import json
        import datetime

        results = {
            "region": region_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": asdict(optimized_config),
            "tuning_summary": tuner.get_tuning_summary()
        }

        filename = f"imm_tuning_{region_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Tuning results saved to {filename}")

    return optimized_config
```

This completes the comprehensive M1.2 Kalman Filter Baseline implementation specification. The system provides:

1. **Physics-based motion models** (CV, CT, NCA) using FilterPy
2. **Interactive Multiple Model framework** for automatic model selection
3. **Maritime-specific adaptations** for coordinate systems and constraints
4. **Automated hyperparameter tuning** with cross-validation
5. **Integration with existing evaluation framework** (evalx)

The implementation follows maritime domain best practices while leveraging proven libraries for robust, production-ready baseline models.
