"""
Maritime-specific feature engineering for classical ML models.
"""

import numpy as np
import pandas as pd


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
        # Constants for feature indices
        SOG_INDEX = 2
        COG_INDEX = 3
        ANGLE_WRAP_THRESHOLD = 180

        features = []

        # Original features
        features.append(trajectory)

        # Speed acceleration (SOG derivative)
        if trajectory.shape[1] > SOG_INDEX:  # Has SOG
            sog = trajectory[:, SOG_INDEX]
            acceleration = np.gradient(sog)
            features.append(acceleration.reshape(-1, 1))

        # Turn rate (COG derivative)
        if trajectory.shape[1] > COG_INDEX:  # Has COG
            cog = trajectory[:, COG_INDEX]
            # Handle circular difference
            cog_diff = np.diff(cog)
            cog_diff = np.where(
                cog_diff > ANGLE_WRAP_THRESHOLD, cog_diff - 360, cog_diff
            )
            cog_diff = np.where(
                cog_diff < -ANGLE_WRAP_THRESHOLD, cog_diff + 360, cog_diff
            )
            turn_rate = np.concatenate([[0], cog_diff])
            features.append(turn_rate.reshape(-1, 1))

            # Speed/course interaction
            if trajectory.shape[1] > SOG_INDEX:  # Has both SOG and COG
                speed_course_interaction = sog * np.sin(np.radians(cog))
                features.append(speed_course_interaction.reshape(-1, 1))

        return np.concatenate(features, axis=1)

    @staticmethod
    def extract_temporal_features(timestamps: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from timestamps.

        Returns features like hour of day, day of week, etc.
        """
        # Convert to pandas for easy datetime manipulation
        times = pd.to_datetime(timestamps, unit="s")

        features = np.column_stack(
            [
                times.hour.values / 24.0,  # Normalized hour
                times.dayofweek.values / 7.0,  # Normalized day of week
                times.month.values / 12.0,  # Normalized month
                np.sin(2 * np.pi * times.hour.values / 24),  # Cyclic hour sin
                np.cos(2 * np.pi * times.hour.values / 24),  # Cyclic hour cos
                np.sin(2 * np.pi * times.dayofweek.values / 7),  # Cyclic day sin
                np.cos(2 * np.pi * times.dayofweek.values / 7),  # Cyclic day cos
            ]
        )

        return features

    @staticmethod
    def extract_spatial_context(
        positions: np.ndarray, port_locations: list[tuple[float, float]] | None = None
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
        dist_from_centroid = np.linalg.norm(positions - centroid, axis=1)
        features.append(dist_from_centroid.reshape(-1, 1))

        # Trajectory spread (measure of how spread out the trajectory is)
        spread = np.std(positions, axis=0)
        features.append(np.tile(spread, (len(positions), 1)))

        # Cumulative distance traveled
        if len(positions) > 1:
            distances = []
            cumulative_dist = 0
            distances.append(0)
            for i in range(1, len(positions)):
                dist = haversine_distance(
                    positions[i - 1, 0],
                    positions[i - 1, 1],
                    positions[i, 0],
                    positions[i, 1],
                )
                cumulative_dist += dist
                distances.append(cumulative_dist)
            features.append(np.array(distances).reshape(-1, 1))

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

    @staticmethod
    def extract_statistical_features(
        sequence: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        Extract statistical features over sliding windows.

        Args:
            sequence: Input sequence [seq_len, n_features]
            window_size: Size of sliding window

        Returns:
            Statistical features for each timestep
        """
        seq_len, n_features = sequence.shape
        stat_features = []

        for i in range(seq_len):
            # Define window
            start = max(0, i - window_size + 1)
            end = i + 1
            window = sequence[start:end]

            # Compute statistics for each feature
            features = []
            for f in range(n_features):
                feature_window = window[:, f]

                # Basic statistics
                features.extend(
                    [
                        np.mean(feature_window),
                        np.std(feature_window) if len(feature_window) > 1 else 0,
                        np.min(feature_window),
                        np.max(feature_window),
                        np.median(feature_window),
                    ]
                )

                # Trend (linear regression slope)
                if len(feature_window) > 1:
                    x = np.arange(len(feature_window))
                    slope, _ = np.polyfit(x, feature_window, 1)
                    features.append(slope)
                else:
                    features.append(0)

            stat_features.append(features)

        return np.array(stat_features)

    @staticmethod
    def create_interaction_features(features: np.ndarray) -> np.ndarray:
        """
        Create interaction features between existing features.

        Args:
            features: Input features [n_samples, n_features]

        Returns:
            Extended features with interactions
        """
        # Constants for minimum feature requirements
        MIN_FEATURES_FOR_VELOCITY = 4
        MIN_FEATURES_FOR_POSITION = 2

        n_samples, n_features = features.shape
        interactions = []

        # Speed-direction interaction
        if n_features >= MIN_FEATURES_FOR_VELOCITY:  # Has lat, lon, sog, cog
            sog_idx, cog_idx = 2, 3
            sog = features[:, sog_idx]
            cog = features[:, cog_idx]

            # Velocity components
            vx = sog * np.cos(np.radians(cog))
            vy = sog * np.sin(np.radians(cog))
            interactions.append(vx.reshape(-1, 1))
            interactions.append(vy.reshape(-1, 1))

        # Position change rate
        if n_features >= MIN_FEATURES_FOR_POSITION:  # Has lat, lon
            lat_change = np.gradient(features[:, 0])
            lon_change = np.gradient(features[:, 1])
            interactions.append(lat_change.reshape(-1, 1))
            interactions.append(lon_change.reshape(-1, 1))

        if interactions:
            return np.concatenate([features] + interactions, axis=1)
        else:
            return features


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in km.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing between two points.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360
