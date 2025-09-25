"""
Maritime motion models for Kalman filtering.

Implements various motion models suitable for vessel trajectory prediction:
- Constant Velocity (CV): For straight-line navigation
- Coordinated Turn (CT): For planned course changes
- Nearly Constant Acceleration (NCA): For speed changes and maneuvering
"""

from typing import Any

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from .coordinates import MaritimeCoordinateTransform
from .protocols import (
    BaselineResult,
    MaritimeConstraints,
    MotionModelConfig,
    TrajectoryBaseline,
)


class BaseMotionModel:
    """
    Base class for maritime motion models.

    Provides common functionality for all motion models including
    coordinate transformation, constraint enforcement, and uncertainty estimation.
    """

    def __init__(self, config: MotionModelConfig, constraints: MaritimeConstraints):
        self.config = config
        self.constraints = constraints
        self.coordinate_transform = MaritimeCoordinateTransform()
        self.kf: KalmanFilter | None = None
        self.is_initialized = False

    def _enforce_constraints(self, state: np.ndarray) -> np.ndarray:
        """
        Enforce physical constraints on state vector.

        Args:
            state: State vector to constrain

        Returns:
            Constrained state vector
        """
        constrained_state = state.copy()

        # Speed constraint (apply to velocity components)
        max_speed_ms = (
            self.constraints.max_speed_knots * 0.514444
        )  # Convert knots to m/s

        STATE_WITH_VELOCITY = 4
        if len(state) >= STATE_WITH_VELOCITY:  # Has velocity components
            vx, vy = state[1], state[3]
            current_speed = np.sqrt(vx**2 + vy**2)

            if current_speed > max_speed_ms:
                scale_factor = max_speed_ms / current_speed
                constrained_state[1] *= scale_factor  # vx
                constrained_state[3] *= scale_factor  # vy

        # Acceleration constraint (if applicable)
        STATE_WITH_ACCELERATION = 6
        if len(state) >= STATE_WITH_ACCELERATION:  # Has acceleration components
            ax, ay = state[4], state[5]
            current_accel = np.sqrt(ax**2 + ay**2)

            if current_accel > self.constraints.max_acceleration_ms2:
                scale_factor = self.constraints.max_acceleration_ms2 / current_accel
                constrained_state[4] *= scale_factor  # ax
                constrained_state[5] *= scale_factor  # ay

        # Turn rate constraint (if applicable)
        STATE_WITH_TURN_RATE = 7
        if len(state) >= STATE_WITH_TURN_RATE:  # Has turn rate
            max_turn_rate_rad_s = np.radians(self.constraints.max_turn_rate_deg_s)
            if abs(state[6]) > max_turn_rate_rad_s:
                constrained_state[6] = np.sign(state[6]) * max_turn_rate_rad_s

        return constrained_state

    def _get_measurement_matrix(self, state_dim: int) -> np.ndarray:
        """Get measurement matrix for position observations."""
        H = np.zeros((2, state_dim))
        H[0, 0] = 1.0  # x position
        H[1, 2] = 1.0  # y position
        return H

    def _get_measurement_noise_matrix(self) -> np.ndarray:
        """Get measurement noise covariance matrix."""
        pos_noise = self.config.position_measurement_noise
        return np.array([[pos_noise**2, 0], [0, pos_noise**2]])

    def get_model_info(self) -> dict[str, Any]:
        """Get model configuration information."""
        return {
            "model_type": self.__class__.__name__,
            "config": self.config.__dict__,
            "constraints": self.constraints.__dict__,
            "is_initialized": self.is_initialized,
            "state_dim": self.kf.dim_x if self.kf is not None else None,
        }


class ConstantVelocityModel(BaseMotionModel, TrajectoryBaseline):
    """
    Constant Velocity motion model.

    State vector: [x, vx, y, vy]
    - x, y: Position in local coordinates (meters)
    - vx, vy: Velocity components (m/s)
    """

    def __init__(
        self, config: MotionModelConfig = None, constraints: MaritimeConstraints = None
    ):
        super().__init__(
            config or MotionModelConfig(), constraints or MaritimeConstraints()
        )
        self._setup_kalman_filter()

    def _setup_kalman_filter(self):
        """Initialize the Kalman filter for constant velocity model."""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity)
        # x_{k+1} = x_k + vx_k * dt
        # vx_{k+1} = vx_k
        # y_{k+1} = y_k + vy_k * dt
        # vy_{k+1} = vy_k
        dt = 1.0  # Will be updated dynamically
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        # Measurement matrix (observe position only)
        self.kf.H = self._get_measurement_matrix(4)

        # Measurement noise
        self.kf.R = self._get_measurement_noise_matrix()

        # Initial state covariance
        pos_var = self.config.initial_position_uncertainty**2
        vel_var = self.config.initial_velocity_uncertainty**2
        self.kf.P = np.diag([pos_var, vel_var, pos_var, vel_var])

    def _update_process_noise(self, dt: float):
        """Update process noise matrix for given time step."""
        # Process noise for constant velocity model
        # Accounts for unmodeled acceleration
        q_vel = self.config.velocity_process_noise**2

        # Discrete white noise for position-velocity pairs
        q_block = Q_discrete_white_noise(2, dt, q_vel)

        self.kf.Q = np.block([[q_block, np.zeros((2, 2))], [np.zeros((2, 2)), q_block]])

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "ConstantVelocityModel":
        """
        Fit the CV model (mainly sets up coordinate transform).

        Args:
            sequences: List of trajectory sequences [seq_len, 2] with lat/lon
            metadata: Optional metadata for each sequence
        """
        if not sequences:
            raise ValueError("No sequences provided for fitting")

        # Use first sequence to set up coordinate transform
        first_sequence = sequences[0]
        if first_sequence.size == 0:
            raise ValueError("Cannot fit with empty sequence")
        MIN_COLUMNS_REQUIRED = 2
        MIN_DIMENSIONS = 2
        if len(first_sequence.shape) < MIN_DIMENSIONS or first_sequence.shape[1] < MIN_COLUMNS_REQUIRED:
            raise ValueError("Sequences must have at least 2 columns (lat, lon)")

        self.coordinate_transform.auto_set_reference(first_sequence[:, :2])
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
        Predict future trajectory points using CV model.

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

        # Extract positions (first 2 columns) and timestamps (last column or indices)
        positions = sequence[:, :2]  # lat, lon

        # Try to extract timestamps, fallback to indices if not available
        TIMESTAMP_COLUMN_INDEX = 3
        if sequence.shape[1] >= TIMESTAMP_COLUMN_INDEX:
            timestamps = sequence[:, 2]
        else:
            timestamps = np.arange(len(positions), dtype=float)

        # Convert to local coordinates
        local_positions = self.coordinate_transform.to_local(positions)

        # Initialize state from last few points
        MIN_POINTS_VELOCITY_ESTIMATE = 2
        if len(local_positions) >= MIN_POINTS_VELOCITY_ESTIMATE:
            # Estimate initial velocity
            dt = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1.0
            dt = max(dt, 1e-6)  # Avoid division by zero

            vel_x = (local_positions[-1, 0] - local_positions[-2, 0]) / dt
            vel_y = (local_positions[-1, 1] - local_positions[-2, 1]) / dt

            # Initialize state: [x, vx, y, vy]
            self.kf.x = np.array(
                [local_positions[-1, 0], vel_x, local_positions[-1, 1], vel_y]
            )
        else:
            self.kf.x = np.array([local_positions[-1, 0], 0, local_positions[-1, 1], 0])

        # Process a few recent observations to calibrate the filter
        calibration_points = min(5, len(local_positions))
        for i in range(len(local_positions) - calibration_points, len(local_positions)):
            if i <= 0:
                continue

            dt = timestamps[i] - timestamps[i - 1] if len(timestamps) > i else 1.0
            dt = max(dt, 1e-6)

            # Update transition matrix with actual dt
            self.kf.F[0, 1] = dt
            self.kf.F[2, 3] = dt

            # Update process noise
            self._update_process_noise(dt)

            # Predict and update
            self.kf.predict()
            self.kf.update(local_positions[i])

        # Now predict future steps
        predictions_local = []
        uncertainties = []

        # Use last known time delta for future predictions
        if len(timestamps) > 1:
            prediction_dt = np.mean(
                np.diff(timestamps[-3:])
            )  # Use mean of last few intervals
        else:
            prediction_dt = 1.0

        prediction_dt = max(prediction_dt, 1e-6)

        for _step in range(horizon):
            # Update matrices for prediction time step
            self.kf.F[0, 1] = prediction_dt
            self.kf.F[2, 3] = prediction_dt
            self._update_process_noise(prediction_dt)

            # Predict next state
            self.kf.predict()

            # Apply constraints
            self.kf.x = self._enforce_constraints(self.kf.x)

            # Extract position prediction
            pred_x, pred_y = self.kf.x[0], self.kf.x[2]
            predictions_local.append([pred_x, pred_y])

            if return_uncertainty:
                # Extract position uncertainty
                pos_cov = np.array(
                    [
                        [self.kf.P[0, 0], self.kf.P[0, 2]],
                        [self.kf.P[2, 0], self.kf.P[2, 2]],
                    ]
                )
                uncertainties.append(pos_cov)

        # Convert predictions back to geographic coordinates
        predictions_local = np.array(predictions_local)
        predictions_geo = self.coordinate_transform.to_geographic(predictions_local)

        result = BaselineResult(
            predictions=predictions_geo, model_info=self.get_model_info()
        )

        if return_uncertainty:
            result.uncertainty = np.array(uncertainties)

        return result


class CoordinatedTurnModel(BaseMotionModel, TrajectoryBaseline):
    """
    Coordinated Turn motion model.

    State vector: [x, vx, y, vy, ω]
    - x, y: Position in local coordinates (meters)
    - vx, vy: Velocity components (m/s)
    - ω: Turn rate (rad/s)
    """

    def __init__(
        self, config: MotionModelConfig = None, constraints: MaritimeConstraints = None
    ):
        super().__init__(
            config or MotionModelConfig(), constraints or MaritimeConstraints()
        )
        self._setup_kalman_filter()

    def _setup_kalman_filter(self):
        """Initialize the Kalman filter for coordinated turn model."""
        self.kf = KalmanFilter(dim_x=5, dim_z=2)

        # State transition matrix will be updated dynamically for nonlinear turn model
        # For small angles, we can use linearized approximation
        self.kf.F = np.eye(5)  # Will be updated in _update_transition_matrix

        # Measurement matrix (observe position only)
        self.kf.H = self._get_measurement_matrix(5)

        # Measurement noise
        self.kf.R = self._get_measurement_noise_matrix()

        # Initial state covariance
        pos_var = self.config.initial_position_uncertainty**2
        vel_var = self.config.initial_velocity_uncertainty**2
        turn_var = self.config.initial_turn_rate_uncertainty**2

        self.kf.P = np.diag([pos_var, vel_var, pos_var, vel_var, turn_var])

    def _update_transition_matrix(self, dt: float):
        """Update state transition matrix for coordinated turn."""
        # For coordinated turn model with turn rate ω:
        # x_{k+1} = x_k + (vx_k * sin(ω*dt) + vy_k * (1-cos(ω*dt))) / ω
        # vx_{k+1} = vx_k * cos(ω*dt) - vy_k * sin(ω*dt)
        # y_{k+1} = y_k + (vy_k * sin(ω*dt) - vx_k * (1-cos(ω*dt))) / ω
        # vy_{k+1} = vy_k * cos(ω*dt) + vx_k * sin(ω*dt)
        # ω_{k+1} = ω_k

        # For linearization, we use current state estimate
        MIN_OMEGA_THRESHOLD = 1e-6
        omega = (
            self.kf.x[4]
            if abs(self.kf.x[4]) > MIN_OMEGA_THRESHOLD
            else MIN_OMEGA_THRESHOLD
        )

        sin_omega_dt = np.sin(omega * dt)
        cos_omega_dt = np.cos(omega * dt)
        one_minus_cos = 1 - cos_omega_dt

        self.kf.F = np.array(
            [
                [1, sin_omega_dt / omega, 0, one_minus_cos / omega, 0],
                [0, cos_omega_dt, 0, -sin_omega_dt, 0],
                [0, -one_minus_cos / omega, 1, sin_omega_dt / omega, 0],
                [0, sin_omega_dt, 0, cos_omega_dt, 0],
                [0, 0, 0, 0, 1],
            ]
        )

    def _update_process_noise(self, dt: float):
        """Update process noise matrix for coordinated turn model."""
        q_pos = self.config.position_process_noise**2
        q_vel = self.config.velocity_process_noise**2
        q_turn = self.config.turn_rate_process_noise**2

        # Process noise affects primarily velocity and turn rate
        self.kf.Q = np.diag(
            [
                q_pos * dt**2,  # x position
                q_vel * dt,  # x velocity
                q_pos * dt**2,  # y position
                q_vel * dt,  # y velocity
                q_turn * dt,  # turn rate
            ]
        )

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "CoordinatedTurnModel":
        """Fit the CT model."""
        if not sequences:
            raise ValueError("No sequences provided for fitting")

        first_sequence = sequences[0]
        if first_sequence.size == 0:
            raise ValueError("Cannot fit with empty sequence")
        MIN_COLUMNS_REQUIRED = 2
        MIN_DIMENSIONS = 2
        if len(first_sequence.shape) < MIN_DIMENSIONS or first_sequence.shape[1] < MIN_COLUMNS_REQUIRED:
            raise ValueError("Sequences must have at least 2 columns (lat, lon)")

        self.coordinate_transform.auto_set_reference(first_sequence[:, :2])
        self.is_initialized = True

        return self

    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs,
    ) -> BaselineResult:
        """Predict future trajectory using coordinated turn model."""
        if not self.is_initialized:
            self.fit([sequence])

        MIN_POINTS_CT_PREDICT = 3
        if sequence.shape[0] < MIN_POINTS_CT_PREDICT:
            raise ValueError(
                f"CT model needs at least {MIN_POINTS_CT_PREDICT} points to estimate turn rate"
            )

        positions = sequence[:, :2]
        TIMESTAMP_COLUMN_INDEX = 3
        if sequence.shape[1] >= TIMESTAMP_COLUMN_INDEX:
            timestamps = sequence[:, 2]
        else:
            timestamps = np.arange(len(positions), dtype=float)

        # Convert to local coordinates and compute kinematics
        local_positions = self.coordinate_transform.to_local(positions)
        velocities = self.coordinate_transform.compute_velocity_local(
            positions, timestamps
        )

        # Estimate turn rate from recent data
        MIN_POINTS_TURN_RATE_CALC = 3
        if len(positions) >= MIN_POINTS_TURN_RATE_CALC:
            _, turn_rates = self.coordinate_transform.compute_heading_and_turn_rate(
                positions, timestamps
            )
            initial_turn_rate = np.mean(turn_rates[-3:]) if len(turn_rates) > 0 else 0.0
        else:
            initial_turn_rate = 0.0

        # Initialize state
        if len(velocities) > 0:
            self.kf.x = np.array(
                [
                    local_positions[-1, 0],  # x
                    velocities[-1, 0],  # vx
                    local_positions[-1, 1],  # y
                    velocities[-1, 1],  # vy
                    initial_turn_rate,  # ω
                ]
            )
        else:
            self.kf.x = np.array(
                [local_positions[-1, 0], 0, local_positions[-1, 1], 0, 0]
            )

        # Calibrate filter with recent observations
        calibration_points = min(3, len(local_positions))
        for i in range(len(local_positions) - calibration_points, len(local_positions)):
            if i <= 0:
                continue

            dt = timestamps[i] - timestamps[i - 1] if len(timestamps) > i else 1.0
            dt = max(dt, 1e-6)

            self._update_transition_matrix(dt)
            self._update_process_noise(dt)

            self.kf.predict()
            self.kf.update(local_positions[i])

        # Predict future states
        predictions_local = []
        uncertainties = []

        prediction_dt = (
            np.mean(np.diff(timestamps[-3:])) if len(timestamps) > 1 else 1.0
        )
        prediction_dt = max(prediction_dt, 1e-6)

        for _step in range(horizon):
            self._update_transition_matrix(prediction_dt)
            self._update_process_noise(prediction_dt)

            self.kf.predict()
            self.kf.x = self._enforce_constraints(self.kf.x)

            pred_x, pred_y = self.kf.x[0], self.kf.x[2]
            predictions_local.append([pred_x, pred_y])

            if return_uncertainty:
                pos_cov = np.array(
                    [
                        [self.kf.P[0, 0], self.kf.P[0, 2]],
                        [self.kf.P[2, 0], self.kf.P[2, 2]],
                    ]
                )
                uncertainties.append(pos_cov)

        predictions_local = np.array(predictions_local)
        predictions_geo = self.coordinate_transform.to_geographic(predictions_local)

        result = BaselineResult(
            predictions=predictions_geo, model_info=self.get_model_info()
        )

        if return_uncertainty:
            result.uncertainty = np.array(uncertainties)

        return result


class NearlyConstantAccelModel(BaseMotionModel, TrajectoryBaseline):
    """
    Nearly Constant Acceleration motion model.

    State vector: [x, vx, ax, y, vy, ay]
    - x, y: Position in local coordinates (meters)
    - vx, vy: Velocity components (m/s)
    - ax, ay: Acceleration components (m/s²)
    """

    def __init__(
        self, config: MotionModelConfig = None, constraints: MaritimeConstraints = None
    ):
        super().__init__(
            config or MotionModelConfig(), constraints or MaritimeConstraints()
        )
        self._setup_kalman_filter()

    def _setup_kalman_filter(self):
        """Initialize the Kalman filter for nearly constant acceleration model."""
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        # State transition matrix for constant acceleration
        # x_{k+1} = x_k + vx_k*dt + 0.5*ax_k*dt²
        # vx_{k+1} = vx_k + ax_k*dt
        # ax_{k+1} = ax_k
        dt = 1.0  # Will be updated dynamically
        self.kf.F = np.array(
            [
                [1, dt, 0.5 * dt**2, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt**2],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement matrix (observe position only)
        self.kf.H = self._get_measurement_matrix(6)

        # Measurement noise
        self.kf.R = self._get_measurement_noise_matrix()

        # Initial state covariance
        pos_var = self.config.initial_position_uncertainty**2
        vel_var = self.config.initial_velocity_uncertainty**2
        acc_var = self.config.initial_acceleration_uncertainty**2

        self.kf.P = np.diag([pos_var, vel_var, acc_var, pos_var, vel_var, acc_var])

    def _update_transition_matrix(self, dt: float):
        """Update state transition matrix with actual time step."""
        self.kf.F[0, 1] = dt
        self.kf.F[0, 2] = 0.5 * dt**2
        self.kf.F[1, 2] = dt
        self.kf.F[3, 4] = dt
        self.kf.F[3, 5] = 0.5 * dt**2
        self.kf.F[4, 5] = dt

    def _update_process_noise(self, dt: float):
        """Update process noise matrix for NCA model."""
        q_acc = self.config.acceleration_process_noise**2

        # Discrete white noise for position-velocity-acceleration triplets
        q_block = Q_discrete_white_noise(3, dt, q_acc)

        self.kf.Q = np.block([[q_block, np.zeros((3, 3))], [np.zeros((3, 3)), q_block]])

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "NearlyConstantAccelModel":
        """Fit the NCA model."""
        if not sequences:
            raise ValueError("No sequences provided for fitting")

        first_sequence = sequences[0]
        if first_sequence.size == 0:
            raise ValueError("Cannot fit with empty sequence")
        MIN_COLUMNS_REQUIRED = 2
        MIN_DIMENSIONS = 2
        if len(first_sequence.shape) < MIN_DIMENSIONS or first_sequence.shape[1] < MIN_COLUMNS_REQUIRED:
            raise ValueError("Sequences must have at least 2 columns (lat, lon)")

        self.coordinate_transform.auto_set_reference(first_sequence[:, :2])
        self.is_initialized = True

        return self

    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs,
    ) -> BaselineResult:
        """Predict future trajectory using NCA model."""
        if not self.is_initialized:
            self.fit([sequence])

        MIN_POINTS_NCA_PREDICT = 3
        if sequence.shape[0] < MIN_POINTS_NCA_PREDICT:
            raise ValueError(
                f"NCA model needs at least {MIN_POINTS_NCA_PREDICT} points to estimate acceleration"
            )

        positions = sequence[:, :2]
        TIMESTAMP_COLUMN_INDEX = 3
        if sequence.shape[1] >= TIMESTAMP_COLUMN_INDEX:
            timestamps = sequence[:, 2]
        else:
            timestamps = np.arange(len(positions), dtype=float)

        # Convert to local coordinates and compute kinematics
        local_positions = self.coordinate_transform.to_local(positions)
        velocities = self.coordinate_transform.compute_velocity_local(
            positions, timestamps
        )

        # Compute accelerations
        MIN_POINTS_TURN_RATE_CALC = 3
        if len(positions) >= MIN_POINTS_TURN_RATE_CALC:
            accelerations = self.coordinate_transform.compute_acceleration_local(
                positions, timestamps
            )
            initial_accel = (
                accelerations[-1] if len(accelerations) > 0 else np.array([0.0, 0.0])
            )
        else:
            initial_accel = np.array([0.0, 0.0])

        # Initialize state [x, vx, ax, y, vy, ay]
        if len(velocities) > 0:
            self.kf.x = np.array(
                [
                    local_positions[-1, 0],  # x
                    velocities[-1, 0],  # vx
                    initial_accel[0],  # ax
                    local_positions[-1, 1],  # y
                    velocities[-1, 1],  # vy
                    initial_accel[1],  # ay
                ]
            )
        else:
            self.kf.x = np.array(
                [local_positions[-1, 0], 0, 0, local_positions[-1, 1], 0, 0]
            )

        # Calibrate filter with recent observations
        calibration_points = min(5, len(local_positions))
        for i in range(len(local_positions) - calibration_points, len(local_positions)):
            if i <= 0:
                continue

            dt = timestamps[i] - timestamps[i - 1] if len(timestamps) > i else 1.0
            dt = max(dt, 1e-6)

            self._update_transition_matrix(dt)
            self._update_process_noise(dt)

            self.kf.predict()
            self.kf.update(local_positions[i])

        # Predict future states
        predictions_local = []
        uncertainties = []

        prediction_dt = (
            np.mean(np.diff(timestamps[-3:])) if len(timestamps) > 1 else 1.0
        )
        prediction_dt = max(prediction_dt, 1e-6)

        for _step in range(horizon):
            self._update_transition_matrix(prediction_dt)
            self._update_process_noise(prediction_dt)

            self.kf.predict()
            self.kf.x = self._enforce_constraints(self.kf.x)

            pred_x, pred_y = self.kf.x[0], self.kf.x[3]
            predictions_local.append([pred_x, pred_y])

            if return_uncertainty:
                pos_cov = np.array(
                    [
                        [self.kf.P[0, 0], self.kf.P[0, 3]],
                        [self.kf.P[3, 0], self.kf.P[3, 3]],
                    ]
                )
                uncertainties.append(pos_cov)

        predictions_local = np.array(predictions_local)
        predictions_geo = self.coordinate_transform.to_geographic(predictions_local)

        result = BaselineResult(
            predictions=predictions_geo, model_info=self.get_model_info()
        )

        if return_uncertainty:
            result.uncertainty = np.array(uncertainties)

        return result
