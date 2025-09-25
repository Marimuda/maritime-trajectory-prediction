"""
Base protocols and data structures for maritime trajectory baseline models.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class BaselineResult:
    """
    Standard result format for baseline model predictions.

    Attributes:
        predictions: Array of shape [batch_size, horizon, n_features] or [horizon, n_features]
        uncertainty: Optional uncertainty estimates (same shape as predictions)
        metadata: Additional information about the prediction
        confidence: Per-step confidence scores [0, 1]
        model_info: Information about which model(s) were used
    """

    predictions: np.ndarray
    uncertainty: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
    confidence: np.ndarray | None = None
    model_info: dict[str, Any] | None = None


@dataclass
class MaritimeConstraints:
    """Maritime-specific physical constraints for vessel motion."""

    max_speed_knots: float = 50.0  # Maximum realistic vessel speed
    max_acceleration_ms2: float = 0.5  # Maximum acceleration (m/s²)
    max_turn_rate_deg_s: float = 5.0  # Maximum turn rate (degrees/second)
    min_position_uncertainty: float = 10.0  # Minimum position uncertainty (meters)
    max_prediction_horizon_hours: float = 24.0  # Maximum prediction horizon


@dataclass
class MotionModelConfig:
    """Configuration for individual motion models."""

    # Process noise parameters
    position_process_noise: float = 0.1  # Position process noise (m)
    velocity_process_noise: float = 0.01  # Velocity process noise (m/s)
    acceleration_process_noise: float = 0.001  # Acceleration process noise (m/s²)
    turn_rate_process_noise: float = 0.1  # Turn rate process noise (rad/s)

    # Measurement noise parameters
    position_measurement_noise: float = 10.0  # Position measurement noise (m)
    velocity_measurement_noise: float = 0.1  # Velocity measurement noise (m/s)

    # Initial state uncertainty
    initial_position_uncertainty: float = 100.0  # Initial position uncertainty (m)
    initial_velocity_uncertainty: float = 5.0  # Initial velocity uncertainty (m/s)
    initial_acceleration_uncertainty: float = (
        1.0  # Initial acceleration uncertainty (m/s²)
    )
    initial_turn_rate_uncertainty: float = 0.1  # Initial turn rate uncertainty (rad/s)


@dataclass
class IMMConfig:
    """Configuration for Interactive Multiple Model framework."""

    # Model transition probabilities (Markov chain)
    # Default: 95% stay in same model, 5% split between other models
    transition_probabilities: np.ndarray | None = None

    # Initial model probabilities
    initial_model_probabilities: np.ndarray | None = None

    # Motion model configurations
    motion_config: MotionModelConfig | None = None

    # Maritime constraints
    constraints: MaritimeConstraints | None = None

    # Coordinate system settings
    use_local_coordinates: bool = True  # Convert lat/lon to local Cartesian
    reference_point: tuple[float, float] | None = None  # (lat, lon) reference

    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.motion_config is None:
            self.motion_config = MotionModelConfig()

        if self.constraints is None:
            self.constraints = MaritimeConstraints()

        if self.transition_probabilities is None:
            # Default 3-model transition matrix (CV, CT, NCA)
            self.transition_probabilities = np.array(
                [
                    [0.95, 0.025, 0.025],  # From CV
                    [0.025, 0.95, 0.025],  # From CT
                    [0.025, 0.025, 0.95],  # From NCA
                ]
            )

        if self.initial_model_probabilities is None:
            n_models = len(self.transition_probabilities)
            self.initial_model_probabilities = np.ones(n_models) / n_models


class TrajectoryBaseline(Protocol):
    """
    Protocol for maritime trajectory baseline models.

    This defines the standard interface that all baseline models should implement
    for consistent evaluation and comparison.
    """

    @abstractmethod
    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "TrajectoryBaseline":
        """
        Fit/train the baseline model on trajectory sequences.

        Args:
            sequences: List of trajectory sequences, each of shape [seq_len, n_features]
            metadata: Optional metadata for each sequence (vessel type, etc.)
            **kwargs: Additional model-specific parameters

        Returns:
            self: Fitted model instance
        """
        ...

    @abstractmethod
    def predict(
        self,
        sequence: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
        **kwargs,
    ) -> np.ndarray | BaselineResult:
        """
        Predict future trajectory points given historical sequence.

        Args:
            sequence: Input sequence of shape [seq_len, n_features]
            horizon: Number of future steps to predict
            return_uncertainty: Whether to return uncertainty estimates
            **kwargs: Additional prediction parameters

        Returns:
            predictions: If return_uncertainty=False, returns np.ndarray of shape [horizon, n_features]
                        If return_uncertainty=True, returns BaselineResult with uncertainty info
        """
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model configuration and state.

        Returns:
            Dictionary containing model information
        """
        ...


@dataclass
class TrajectorySequence:
    """
    Container for a single vessel trajectory sequence.
    """

    # Core trajectory data
    positions: np.ndarray  # Shape [seq_len, 2] - lat, lon in degrees
    timestamps: np.ndarray  # Shape [seq_len] - Unix timestamps

    # Derived kinematic features (computed if available)
    velocities: np.ndarray | None = (
        None  # Shape [seq_len, 2] - m/s in local coordinates
    )
    accelerations: np.ndarray | None = (
        None  # Shape [seq_len, 2] - m/s² in local coordinates
    )
    headings: np.ndarray | None = None  # Shape [seq_len] - radians from North
    turn_rates: np.ndarray | None = None  # Shape [seq_len] - rad/s
    speeds: np.ndarray | None = None  # Shape [seq_len] - m/s ground speed

    # AIS-specific features
    sog: np.ndarray | None = None  # Speed Over Ground from AIS
    cog: np.ndarray | None = None  # Course Over Ground from AIS
    heading_ais: np.ndarray | None = None  # AIS heading

    # Vessel metadata
    mmsi: str | None = None
    vessel_type: int | None = None
    length: float | None = None
    width: float | None = None

    # Quality indicators
    position_accuracy: np.ndarray | None = None  # Position accuracy flags
    data_source: np.ndarray | None = None  # Data source indicators

    def __post_init__(self):
        """Validate sequence data and compute derived features if needed."""
        if len(self.positions) != len(self.timestamps):
            raise ValueError("Positions and timestamps must have same length")

        # Ensure minimum sequence length
        MIN_SEQUENCE_LENGTH = 2
        if len(self.positions) < MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence must have at least {MIN_SEQUENCE_LENGTH} points"
            )

        # Sort by timestamp if not already sorted
        if not np.all(self.timestamps[:-1] <= self.timestamps[1:]):
            sort_idx = np.argsort(self.timestamps)
            self.positions = self.positions[sort_idx]
            self.timestamps = self.timestamps[sort_idx]

            # Sort other arrays if they exist
            for attr_name in [
                "sog",
                "cog",
                "heading_ais",
                "position_accuracy",
                "data_source",
            ]:
                attr_value = getattr(self, attr_name)
                if attr_value is not None:
                    setattr(self, attr_name, attr_value[sort_idx])

    @property
    def duration_hours(self) -> float:
        """Get sequence duration in hours."""
        return (self.timestamps[-1] - self.timestamps[0]) / 3600.0

    @property
    def mean_time_delta_seconds(self) -> float:
        """Get mean time between consecutive points in seconds."""
        return np.mean(np.diff(self.timestamps))

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Get bounding box (min_lat, min_lon, max_lat, max_lon)."""
        min_lat, min_lon = self.positions.min(axis=0)
        max_lat, max_lon = self.positions.max(axis=0)
        return min_lat, min_lon, max_lat, max_lon
