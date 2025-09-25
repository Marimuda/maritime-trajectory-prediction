"""
Interactive Multiple Model (IMM) framework for maritime trajectory prediction.

The IMM framework automatically switches between multiple motion models based on
which model best explains the current vessel behavior. This is particularly useful
for maritime trajectories where vessels may exhibit different motion patterns
(straight-line navigation, turning, accelerating/decelerating).
"""

from typing import Any

import numpy as np
from filterpy.kalman import IMMEstimator

from .coordinates import MaritimeCoordinateTransform
from .models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from .protocols import (
    BaselineResult,
    IMMConfig,
    MaritimeConstraints,
    TrajectoryBaseline,
)


class MaritimeIMMFilter(TrajectoryBaseline):
    """
    Maritime-specific Interactive Multiple Model filter.

    Combines multiple motion models (CV, CT, NCA) and automatically selects
    the most appropriate model based on the vessel's current behavior.
    """

    def __init__(self, config: IMMConfig = None):
        """
        Initialize IMM filter with maritime motion models.

        Args:
            config: IMM configuration parameters
        """
        self.config = config or IMMConfig()
        self.coordinate_transform = MaritimeCoordinateTransform()
        self.is_initialized = False

        # Initialize motion models
        self.cv_model = ConstantVelocityModel(
            self.config.motion_config, self.config.constraints
        )
        self.ct_model = CoordinatedTurnModel(
            self.config.motion_config, self.config.constraints
        )
        self.nca_model = NearlyConstantAccelModel(
            self.config.motion_config, self.config.constraints
        )

        self.models = [self.cv_model, self.ct_model, self.nca_model]
        self.model_names = ["CV", "CT", "NCA"]

        # IMM estimator will be initialized during fit/predict
        self.imm: IMMEstimator | None = None

    def _setup_imm_estimator(self):
        """Initialize the IMM estimator with motion models."""
        # Extract Kalman filters from motion models
        filters = [model.kf for model in self.models]

        # Ensure all filters are initialized
        for i, kf in enumerate(filters):
            if kf is None:
                raise RuntimeError(f"Model {i} ({self.model_names[i]}) not initialized")

        # Create IMM estimator
        self.imm = IMMEstimator(
            filters=filters,
            mu=self.config.initial_model_probabilities,
            M=self.config.transition_probabilities,
        )

    def _align_state_vectors(self):
        """
        Align state vectors of different models for IMM.

        Different models have different state dimensions:
        - CV: [x, vx, y, vy] (4D)
        - CT: [x, vx, y, vy, ω] (5D)
        - NCA: [x, vx, ax, y, vy, ay] (6D)

        We'll use a common 6D state space and project accordingly.
        """
        # Common state: [x, vx, ax, y, vy, ay]
        # CV projects to: [x, vx, 0, y, vy, 0]
        # CT projects to: [x, vx, 0, y, vy, 0] (ignore ω for alignment)
        # NCA uses full: [x, vx, ax, y, vy, ay]

        target_dim = 6  # Use NCA's full state space

        for model in self.models:
            current_dim = model.kf.dim_x

            if current_dim < target_dim:
                # Expand state vector
                old_x = model.kf.x.copy()
                old_P = model.kf.P.copy()
                old_F = model.kf.F.copy()
                old_Q = model.kf.Q.copy()

                # Create new expanded matrices
                model.kf.dim_x = target_dim
                model.kf.x = np.zeros(target_dim)
                model.kf.P = (
                    np.eye(target_dim) * 1e6
                )  # Large uncertainty for new states
                model.kf.F = np.eye(target_dim)
                model.kf.Q = np.eye(target_dim) * 1e-6

                CV_MODEL_DIM = 4
                CT_MODEL_DIM = 5

                if current_dim == CV_MODEL_DIM:  # CV model
                    # Map [x, vx, y, vy] to [x, vx, 0, y, vy, 0]
                    model.kf.x[[0, 1, 3, 4]] = old_x
                    model.kf.P[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_P
                    model.kf.F[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_F
                    model.kf.Q[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_Q

                elif current_dim == CT_MODEL_DIM:  # CT model
                    # Map [x, vx, y, vy, ω] to [x, vx, 0, y, vy, 0] (ignore ω)
                    model.kf.x[[0, 1, 3, 4]] = old_x[[0, 1, 2, 3]]
                    model.kf.P[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_P[
                        np.ix_([0, 1, 2, 3], [0, 1, 2, 3])
                    ]
                    model.kf.F[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_F[
                        np.ix_([0, 1, 2, 3], [0, 1, 2, 3])
                    ]
                    model.kf.Q[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] = old_Q[
                        np.ix_([0, 1, 2, 3], [0, 1, 2, 3])
                    ]

            # Update measurement matrix for new state dimension
            model.kf.H = np.zeros((2, target_dim))
            model.kf.H[0, 0] = 1.0  # x position
            model.kf.H[1, 3] = 1.0  # y position

    def _extract_model_predictions(
        self, horizon: int, prediction_dt: float
    ) -> list[np.ndarray]:
        """
        Extract predictions from each model for model probability computation.

        Args:
            horizon: Number of prediction steps
            prediction_dt: Time step for predictions

        Returns:
            List of prediction arrays for each model
        """
        model_predictions = []

        for _i, model in enumerate(self.models):
            # Create a copy of the model's filter state
            model_kf = model.kf
            saved_x = model_kf.x.copy()
            saved_P = model_kf.P.copy()

            predictions = []
            for _step in range(horizon):
                # Update transition matrices based on model type
                if isinstance(model, ConstantVelocityModel):
                    model_kf.F[0, 1] = prediction_dt
                    model_kf.F[3, 4] = prediction_dt
                elif isinstance(model, CoordinatedTurnModel | NearlyConstantAccelModel):
                    model._update_transition_matrix(prediction_dt)

                # Update process noise
                model._update_process_noise(prediction_dt)

                # Predict
                model_kf.predict()

                # Apply constraints
                model_kf.x = model._enforce_constraints(model_kf.x)

                # Extract position
                predictions.append([model_kf.x[0], model_kf.x[3]])

            # Restore original state
            model_kf.x = saved_x
            model_kf.P = saved_P

            model_predictions.append(np.array(predictions))

        return model_predictions

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "MaritimeIMMFilter":
        """
        Fit the IMM filter on trajectory sequences.

        Args:
            sequences: List of trajectory sequences [seq_len, 2+] with lat/lon
            metadata: Optional metadata for each sequence
        """
        if not sequences:
            raise ValueError("No sequences provided for fitting")

        # Use first sequence to set up coordinate transform
        first_sequence = sequences[0]
        MIN_COLUMNS_REQUIRED = 2
        if first_sequence.shape[1] < MIN_COLUMNS_REQUIRED:
            raise ValueError("Sequences must have at least 2 columns (lat, lon)")

        # Set up coordinate transform
        if self.config.reference_point is not None:
            self.coordinate_transform.set_reference_point(*self.config.reference_point)
        else:
            self.coordinate_transform.auto_set_reference(first_sequence[:, :2])

        # Fit individual models
        for model in self.models:
            model.coordinate_transform = self.coordinate_transform
            model.fit(sequences, metadata, **kwargs)

        # Align state vectors for IMM compatibility
        self._align_state_vectors()

        # Set up IMM estimator
        self._setup_imm_estimator()

        self.is_initialized = True
        return self

    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs,
    ) -> BaselineResult:
        """
        Predict future trajectory using IMM framework.

        Args:
            sequence: Input sequence [seq_len, n_features] with positions and timestamps
            horizon: Number of steps to predict
            return_uncertainty: Whether to return uncertainty estimates
        """
        if not self.is_initialized:
            # Auto-fit on this sequence
            self.fit([sequence])

        MIN_POINTS_PREDICT = 2
        if sequence.shape[0] < MIN_POINTS_PREDICT:
            raise ValueError("Need at least 2 points for prediction")

        # Extract positions and timestamps
        positions = sequence[:, :2]
        TIMESTAMP_COLUMN_INDEX = 3
        if sequence.shape[1] >= TIMESTAMP_COLUMN_INDEX:
            timestamps = sequence[:, 2]
        else:
            timestamps = np.arange(len(positions), dtype=float)

        # Convert to local coordinates
        local_positions = self.coordinate_transform.to_local(positions)

        # Initialize IMM with best guess from individual models
        # Use CV model initialization as starting point

        # Initialize all models with recent trajectory data
        for model in self.models:
            if isinstance(model, ConstantVelocityModel):
                # Initialize CV model
                MIN_POINTS_VELOCITY_INIT = 2
                if len(local_positions) >= MIN_POINTS_VELOCITY_INIT:
                    dt = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0
                    dt = max(dt, 1e-6)

                    vel_x = (local_positions[-1, 0] - local_positions[-2, 0]) / dt
                    vel_y = (local_positions[-1, 1] - local_positions[-2, 1]) / dt

                    # Map to 6D state: [x, vx, ax, y, vy, ay]
                    model.kf.x = np.array(
                        [
                            local_positions[-1, 0],
                            vel_x,
                            0,
                            local_positions[-1, 1],
                            vel_y,
                            0,
                        ]
                    )

            elif isinstance(model, CoordinatedTurnModel):
                # Initialize CT model with turn rate estimate
                MIN_POINTS_TURN_RATE = 3
                if len(positions) >= MIN_POINTS_TURN_RATE:
                    _, turn_rates = (
                        self.coordinate_transform.compute_heading_and_turn_rate(
                            positions, timestamps
                        )
                    )
                    _initial_turn_rate = (
                        np.mean(turn_rates[-2:]) if len(turn_rates) > 0 else 0.0
                    )
                else:
                    _initial_turn_rate = 0.0

                velocities = self.coordinate_transform.compute_velocity_local(
                    positions, timestamps
                )
                if len(velocities) > 0:
                    # Map to 6D state
                    model.kf.x = np.array(
                        [
                            local_positions[-1, 0],
                            velocities[-1, 0],
                            0,
                            local_positions[-1, 1],
                            velocities[-1, 1],
                            0,
                        ]
                    )

            elif isinstance(model, NearlyConstantAccelModel):
                # Initialize NCA model with acceleration estimate
                MIN_POINTS_TURN_RATE = 3
                if len(positions) >= MIN_POINTS_TURN_RATE:
                    accelerations = (
                        self.coordinate_transform.compute_acceleration_local(
                            positions, timestamps
                        )
                    )
                    initial_accel = (
                        accelerations[-1]
                        if len(accelerations) > 0
                        else np.array([0.0, 0.0])
                    )
                else:
                    initial_accel = np.array([0.0, 0.0])

                velocities = self.coordinate_transform.compute_velocity_local(
                    positions, timestamps
                )
                if len(velocities) > 0:
                    # Full 6D state
                    model.kf.x = np.array(
                        [
                            local_positions[-1, 0],
                            velocities[-1, 0],
                            initial_accel[0],
                            local_positions[-1, 1],
                            velocities[-1, 1],
                            initial_accel[1],
                        ]
                    )

        # Process recent observations to calibrate the IMM
        calibration_points = min(5, len(local_positions))
        for i in range(len(local_positions) - calibration_points, len(local_positions)):
            if i <= 0:
                continue

            dt = timestamps[i] - timestamps[i - 1] if len(timestamps) > i else 1.0
            dt = max(dt, 1e-6)

            # Update each model's transition matrix
            for _j, model in enumerate(self.models):
                if isinstance(model, ConstantVelocityModel):
                    model.kf.F[0, 1] = dt
                    model.kf.F[3, 4] = dt
                elif isinstance(model, CoordinatedTurnModel):
                    # Use simplified linear approximation for IMM
                    model.kf.F[0, 1] = dt
                    model.kf.F[3, 4] = dt
                elif isinstance(model, NearlyConstantAccelModel):
                    model._update_transition_matrix(dt)

                model._update_process_noise(dt)

            # Run IMM predict-update cycle
            self.imm.predict()
            self.imm.update(local_positions[i])

        # Now predict future steps
        predictions_local = []
        uncertainties = []
        model_probabilities = []

        # Use last known time delta for predictions
        prediction_dt = (
            np.mean(np.diff(timestamps[-3:])) if len(timestamps) > 1 else 1.0
        )
        prediction_dt = max(prediction_dt, 1e-6)

        for _step in range(horizon):
            # Update transition matrices for all models
            for _j, model in enumerate(self.models):
                if isinstance(model, ConstantVelocityModel | CoordinatedTurnModel):
                    model.kf.F[0, 1] = prediction_dt
                    model.kf.F[3, 4] = prediction_dt
                elif isinstance(model, NearlyConstantAccelModel):
                    model._update_transition_matrix(prediction_dt)

                model._update_process_noise(prediction_dt)

            # IMM predict step
            self.imm.predict()

            # Apply constraints to mixed state estimate
            self.imm.x = self.models[0]._enforce_constraints(self.imm.x)

            # Extract position prediction
            pred_x, pred_y = self.imm.x[0], self.imm.x[3]
            predictions_local.append([pred_x, pred_y])

            # Store model probabilities
            model_probabilities.append(self.imm.mu.copy())

            if return_uncertainty:
                # Extract position uncertainty from mixed covariance
                pos_cov = np.array(
                    [
                        [self.imm.P[0, 0], self.imm.P[0, 3]],
                        [self.imm.P[3, 0], self.imm.P[3, 3]],
                    ]
                )
                uncertainties.append(pos_cov)

        # Convert predictions back to geographic coordinates
        predictions_local = np.array(predictions_local)
        predictions_geo = self.coordinate_transform.to_geographic(predictions_local)

        # Create result with IMM-specific information
        model_info = self.get_model_info()
        model_info["model_probabilities"] = np.array(model_probabilities)
        model_info["final_model_probabilities"] = self.imm.mu.copy()
        model_info["dominant_model"] = self.model_names[np.argmax(self.imm.mu)]

        result = BaselineResult(
            predictions=predictions_geo,
            model_info=model_info,
            confidence=np.max(
                model_probabilities, axis=1
            ),  # Use max model probability as confidence
        )

        if return_uncertainty:
            result.uncertainty = np.array(uncertainties)

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            "model_type": "MaritimeIMM",
            "component_models": self.model_names,
            "config": self.config.__dict__,
            "is_initialized": self.is_initialized,
            "coordinate_transform": self.coordinate_transform.get_reference_info(),
        }

        if self.imm is not None:
            info.update(
                {
                    "current_model_probabilities": self.imm.mu.copy(),
                    "dominant_model": self.model_names[np.argmax(self.imm.mu)],
                    "transition_matrix": self.config.transition_probabilities.copy(),
                    "state_dimension": self.imm.x.shape[0]
                    if hasattr(self.imm, "x")
                    else None,
                }
            )

        return info

    def get_model_probabilities_history(self) -> np.ndarray | None:
        """
        Get history of model probabilities if available.

        Returns:
            Array of shape [n_steps, n_models] with model probabilities over time
        """
        # This would require storing history during prediction
        # Implementation depends on whether we want to store this information
        return None

    def set_model_transition_probabilities(self, transition_matrix: np.ndarray):
        """
        Update model transition probabilities.

        Args:
            transition_matrix: New transition probability matrix [n_models, n_models]
        """
        if transition_matrix.shape != (len(self.models), len(self.models)):
            raise ValueError(
                f"Transition matrix must be {len(self.models)}x{len(self.models)}"
            )

        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Each row of transition matrix must sum to 1.0")

        self.config.transition_probabilities = transition_matrix.copy()

        # Update IMM if already initialized
        if self.imm is not None:
            self.imm.M = transition_matrix.copy()


def create_default_imm_config(
    reference_point: tuple[float, float] | None = None,
    max_speed_knots: float = 50.0,
    transition_probability_stay: float = 0.95,
) -> IMMConfig:
    """
    Create a default IMM configuration for maritime applications.

    Args:
        reference_point: (lat, lon) reference point for coordinate system
        max_speed_knots: Maximum vessel speed in knots
        transition_probability_stay: Probability of staying in same model

    Returns:
        Configured IMMConfig instance
    """
    # Create transition matrix with specified stay probability
    n_models = 3  # CV, CT, NCA
    transition_prob = (1.0 - transition_probability_stay) / (n_models - 1)

    transition_matrix = np.full((n_models, n_models), transition_prob)
    np.fill_diagonal(transition_matrix, transition_probability_stay)

    return IMMConfig(
        transition_probabilities=transition_matrix,
        initial_model_probabilities=np.ones(n_models) / n_models,
        reference_point=reference_point,
        constraints=MaritimeConstraints(max_speed_knots=max_speed_knots),
    )
