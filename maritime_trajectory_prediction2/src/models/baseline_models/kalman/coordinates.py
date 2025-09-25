"""
Coordinate transformation utilities for maritime trajectory prediction.

Provides conversions between geographic (lat/lon) and local Cartesian coordinates
optimized for maritime trajectory prediction tasks.
"""

import numpy as np


class MaritimeCoordinateTransform:
    """
    Coordinate transformer for maritime trajectories.

    Converts between geographic coordinates (latitude/longitude) and local
    Cartesian coordinates suitable for Kalman filtering. Uses a simplified
    equirectangular projection centered on the region of interest.
    """

    def __init__(self, reference_point: tuple[float, float] | None = None):
        """
        Initialize coordinate transformer.

        Args:
            reference_point: (lat, lon) reference point for local coordinates.
                           If None, will be set from first trajectory.
        """
        self.reference_lat = None
        self.reference_lon = None
        self.meters_per_degree_lat = None
        self.meters_per_degree_lon = None

        if reference_point is not None:
            self.set_reference_point(reference_point[0], reference_point[1])

    def set_reference_point(self, lat: float, lon: float) -> None:
        """
        Set the reference point for coordinate transformations.

        Args:
            lat: Reference latitude in degrees
            lon: Reference longitude in degrees
        """
        self.reference_lat = lat
        self.reference_lon = lon

        # Calculate meters per degree at reference latitude
        # Using standard Earth radius and latitude-dependent longitude scaling
        earth_radius_m = 6371000.0  # Earth radius in meters

        self.meters_per_degree_lat = (np.pi / 180.0) * earth_radius_m
        self.meters_per_degree_lon = (
            (np.pi / 180.0) * earth_radius_m * np.cos(np.radians(lat))
        )

    def auto_set_reference(self, positions: np.ndarray) -> None:
        """
        Automatically set reference point from trajectory positions.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees
        """
        if positions.size == 0:
            raise ValueError("Cannot set reference from empty position array")

        # Use centroid of positions as reference
        center_lat = np.mean(positions[:, 0])
        center_lon = np.mean(positions[:, 1])
        self.set_reference_point(center_lat, center_lon)

    def to_local(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert geographic coordinates to local Cartesian coordinates.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees

        Returns:
            Array of shape [n_points, 2] with [x, y] in meters from reference point
        """
        if self.reference_lat is None:
            self.auto_set_reference(positions)

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        lat_diff = positions[:, 0] - self.reference_lat
        lon_diff = positions[:, 1] - self.reference_lon

        x = lon_diff * self.meters_per_degree_lon
        y = lat_diff * self.meters_per_degree_lat

        return np.column_stack([x, y])

    def to_geographic(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert local Cartesian coordinates to geographic coordinates.

        Args:
            positions: Array of shape [n_points, 2] with [x, y] in meters

        Returns:
            Array of shape [n_points, 2] with [lat, lon] in degrees
        """
        if self.reference_lat is None:
            raise ValueError(
                "Reference point not set. Call set_reference_point() first."
            )

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        x = positions[:, 0]
        y = positions[:, 1]

        lon_diff = x / self.meters_per_degree_lon
        lat_diff = y / self.meters_per_degree_lat

        lon = self.reference_lon + lon_diff
        lat = self.reference_lat + lat_diff

        return np.column_stack([lat, lon])

    def compute_velocity_local(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity in local coordinates from position time series.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees
            timestamps: Array of shape [n_points] with Unix timestamps

        Returns:
            Array of shape [n_points-1, 2] with [vx, vy] in m/s
        """
        if len(positions) != len(timestamps):
            raise ValueError("Positions and timestamps must have same length")

        MIN_POINTS_VELOCITY = 2
        if len(positions) < MIN_POINTS_VELOCITY:
            raise ValueError(
                f"Need at least {MIN_POINTS_VELOCITY} points to compute velocity"
            )

        # Convert to local coordinates
        local_positions = self.to_local(positions)

        # Compute differences
        pos_diff = np.diff(local_positions, axis=0)
        time_diff = np.diff(timestamps)

        # Handle zero time differences
        time_diff = np.where(time_diff == 0, 1e-6, time_diff)

        # Compute velocities
        velocities = pos_diff / time_diff[:, np.newaxis]

        return velocities

    def compute_acceleration_local(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration in local coordinates from position time series.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees
            timestamps: Array of shape [n_points] with Unix timestamps

        Returns:
            Array of shape [n_points-2, 2] with [ax, ay] in m/s²
        """
        MIN_POINTS_ACCELERATION = 3
        if len(positions) < MIN_POINTS_ACCELERATION:
            raise ValueError(
                f"Need at least {MIN_POINTS_ACCELERATION} points to compute acceleration"
            )

        # First compute velocities
        velocities = self.compute_velocity_local(positions, timestamps)

        # Then compute acceleration from velocity differences
        vel_diff = np.diff(velocities, axis=0)
        time_diff = np.diff(timestamps[:-1])  # Time between velocity midpoints

        # Handle zero time differences
        time_diff = np.where(time_diff == 0, 1e-6, time_diff)

        accelerations = vel_diff / time_diff[:, np.newaxis]

        return accelerations

    def compute_heading_and_turn_rate(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute heading and turn rate from position time series.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees
            timestamps: Array of shape [n_points] with Unix timestamps

        Returns:
            Tuple of (headings, turn_rates):
                - headings: Array of shape [n_points-1] with heading in radians from North
                - turn_rates: Array of shape [n_points-2] with turn rate in rad/s
        """
        MIN_POINTS_HEADING = 2
        if len(positions) < MIN_POINTS_HEADING:
            raise ValueError(
                f"Need at least {MIN_POINTS_HEADING} points to compute heading"
            )

        # Get velocities in local coordinates
        velocities = self.compute_velocity_local(positions, timestamps)

        # Compute headings (angle from North, clockwise positive)
        headings = np.arctan2(velocities[:, 0], velocities[:, 1])

        MIN_POINTS_TURN_RATE = 3
        if len(positions) < MIN_POINTS_TURN_RATE:
            return headings, np.array([])

        # Compute turn rates
        heading_diff = np.diff(headings)

        # Handle angle wraparound (-π to π)
        heading_diff = np.where(
            heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff
        )
        heading_diff = np.where(
            heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff
        )

        time_diff = np.diff(timestamps[:-1])  # Time between heading midpoints
        time_diff = np.where(time_diff == 0, 1e-6, time_diff)

        turn_rates = heading_diff / time_diff

        return headings, turn_rates

    def validate_trajectory(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        max_speed_ms: float = 25.7,  # ~50 knots
        max_acceleration_ms2: float = 0.5,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Validate trajectory for physical plausibility.

        Args:
            positions: Array of shape [n_points, 2] with [lat, lon] in degrees
            timestamps: Array of shape [n_points] with Unix timestamps
            max_speed_ms: Maximum allowed speed in m/s
            max_acceleration_ms2: Maximum allowed acceleration in m/s²

        Returns:
            Tuple of (valid_mask, stats):
                - valid_mask: Boolean array indicating valid points
                - stats: Dictionary with validation statistics
        """
        n_points = len(positions)
        valid_mask = np.ones(n_points, dtype=bool)
        stats = {
            "total_points": n_points,
            "invalid_speed": 0,
            "invalid_acceleration": 0,
            "time_gaps": 0,
        }

        MIN_POINTS_VALIDATION = 2
        if n_points < MIN_POINTS_VALIDATION:
            return valid_mask, stats

        # Check velocities
        try:
            velocities = self.compute_velocity_local(positions, timestamps)
            speeds = np.linalg.norm(velocities, axis=1)
            invalid_speed_mask = speeds > max_speed_ms

            # Mark both points involved in invalid velocity as invalid
            for i in np.where(invalid_speed_mask)[0]:
                valid_mask[i] = False
                valid_mask[i + 1] = False
                stats["invalid_speed"] += 1

        except Exception:
            # If velocity computation fails, mark all as potentially invalid
            pass

        # Check accelerations (if enough points)
        MIN_POINTS_ACCEL_CHECK = 3
        if n_points >= MIN_POINTS_ACCEL_CHECK:
            try:
                accelerations = self.compute_acceleration_local(positions, timestamps)
                accel_magnitudes = np.linalg.norm(accelerations, axis=1)
                invalid_accel_mask = accel_magnitudes > max_acceleration_ms2

                # Mark points involved in invalid acceleration as invalid
                for i in np.where(invalid_accel_mask)[0]:
                    valid_mask[i] = False
                    valid_mask[i + 1] = False
                    valid_mask[i + 2] = False
                    stats["invalid_acceleration"] += 1

            except Exception:
                pass

        # Check for large time gaps (>1 hour)
        time_diffs = np.diff(timestamps)
        MAX_TIME_GAP_SECONDS = 3600  # 1 hour
        large_gaps = time_diffs > MAX_TIME_GAP_SECONDS
        for i in np.where(large_gaps)[0]:
            valid_mask[i + 1] = False
            stats["time_gaps"] += 1

        stats["valid_points"] = np.sum(valid_mask)
        stats["validity_rate"] = stats["valid_points"] / n_points if n_points > 0 else 0

        return valid_mask, stats

    def get_reference_info(self) -> dict:
        """Get information about the current coordinate system."""
        return {
            "reference_lat": self.reference_lat,
            "reference_lon": self.reference_lon,
            "meters_per_degree_lat": self.meters_per_degree_lat,
            "meters_per_degree_lon": self.meters_per_degree_lon,
            "is_initialized": self.reference_lat is not None,
        }
