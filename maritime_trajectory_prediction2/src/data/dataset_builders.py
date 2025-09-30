"""
Task-specific dataset builders for different ML applications.
"""

import logging

import numpy as np
import pandas as pd

from ..utils.maritime_utils import MaritimeUtils
from .pipeline import BaseDatasetBuilder
from .schema import FeatureGroups

logger = logging.getLogger(__name__)


class TrajectoryPredictionBuilder(BaseDatasetBuilder):
    """Dataset builder for vessel trajectory prediction tasks."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for trajectory prediction."""
        import time

        self.logger.info(f"Building features from {len(df):,} records...")
        features_df = df.copy()

        # Core position and movement features
        feature_columns = FeatureGroups.EXTENDED_TRAJECTORY[
            :6
        ]  # lat, lon, sog, cog, heading, turn

        # Add derived features with timing
        start = time.time()
        self.logger.info("  → Adding temporal features (hour, day, month)...")
        features_df = self._add_temporal_features(features_df)
        self.logger.info(f"    ✓ Temporal features added in {time.time()-start:.1f}s")

        start = time.time()
        self.logger.info(
            "  → Adding movement features (speed change, course change, distance)..."
        )
        features_df = self._add_movement_features(features_df)
        self.logger.info(f"    ✓ Movement features added in {time.time()-start:.1f}s")

        start = time.time()
        self.logger.info("  → Adding spatial features (distance from center)...")
        features_df = self._add_spatial_features(features_df)
        self.logger.info(f"    ✓ Spatial features added in {time.time()-start:.1f}s")

        # Select and order features
        available_features = [
            col for col in feature_columns if col in features_df.columns
        ]
        derived_features = [
            col
            for col in features_df.columns
            if col.startswith(("temporal_", "movement_", "spatial_"))
        ]

        final_features = available_features + derived_features
        self.logger.info(f"  ✓ Total features: {len(final_features)} columns")
        return features_df[["mmsi", "time"] + final_features]

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build targets for trajectory prediction (future positions)."""
        targets_df = df[["mmsi", "time"] + FeatureGroups.BASIC_TRAJECTORY].copy()
        return targets_df

    def create_sequences(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create input sequences and future position targets."""
        sequences_X = []
        sequences_y = []

        # Group by vessel
        for _mmsi, vessel_data in features_df.groupby("mmsi"):
            vessel_df = vessel_data.sort_values("time").reset_index(drop=True)

            if (
                len(vessel_df)
                < self.config.sequence_length + self.config.prediction_horizon
            ):
                continue

            # Extract feature columns (exclude mmsi and time)
            feature_cols = [
                col for col in vessel_df.columns if col not in ["mmsi", "time"]
            ]
            vessel_features = vessel_df[feature_cols].values

            # Create sequences
            for i in range(
                len(vessel_features)
                - self.config.sequence_length
                - self.config.prediction_horizon
                + 1
            ):
                # Input sequence
                X_seq = vessel_features[i : i + self.config.sequence_length]

                # Target sequence (future positions)
                target_start = i + self.config.sequence_length
                target_end = target_start + self.config.prediction_horizon
                # Target sequence - use schema-defined trajectory features
                target_feature_count = len(FeatureGroups.BASIC_TRAJECTORY)
                y_seq = vessel_features[target_start:target_end, :target_feature_count]

                sequences_X.append(X_seq)
                sequences_y.append(y_seq)

        if not sequences_X:
            return np.array([]), np.array([])

        X = np.array(sequences_X)
        y = np.array(sequences_y)

        self.logger.info(
            f"Created {len(X)} trajectory sequences with shape X: {X.shape}, y: {y.shape}"
        )
        return X, y

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if "time" in df.columns:
            df["temporal_hour"] = df["time"].dt.hour
            df["temporal_day_of_week"] = df["time"].dt.dayofweek
            df["temporal_month"] = df["time"].dt.month
        return df

    def _add_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add movement-derived features (VECTORIZED for performance)."""
        import numpy as np

        # Speed change rate (already vectorized - pandas groupby().diff())
        if "sog" in df.columns:
            df["movement_speed_change"] = df.groupby("mmsi")["sog"].diff()

        # Course change rate (already vectorized)
        if "cog" in df.columns:
            df["movement_course_change"] = df.groupby("mmsi")["cog"].diff()

        # Distance traveled (VECTORIZED version - massive speedup!)
        if all(col in df.columns for col in ["lat", "lon"]):
            # Sort by vessel and time to ensure consecutive points
            df = df.sort_values(["mmsi", "time"]).reset_index(drop=True)

            # Shift coordinates to get previous points (vectorized!)
            df["_prev_lat"] = df.groupby("mmsi")["lat"].shift(1)
            df["_prev_lon"] = df.groupby("mmsi")["lon"].shift(1)

            # Vectorized Haversine distance calculation
            # Formula: d = 2 * R * arcsin(sqrt(sin²((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin²((lon2-lon1)/2)))
            R = 6371.0  # Earth radius in km

            # Convert to radians (vectorized)
            lat1 = np.radians(df["_prev_lat"])
            lon1 = np.radians(df["_prev_lon"])
            lat2 = np.radians(df["lat"])
            lon2 = np.radians(df["lon"])

            # Haversine formula (all vectorized operations!)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c

            # First point in each vessel has no previous point (NaN -> 0)
            df["movement_distance"] = distance.fillna(0.0)

            # Clean up temporary columns
            df = df.drop(columns=["_prev_lat", "_prev_lon"])

        return df

    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial context features."""
        if all(col in df.columns for col in ["lat", "lon"]):
            # Distance from center of area
            center_lat = df["lat"].mean()
            center_lon = df["lon"].mean()

            df["spatial_distance_from_center"] = MaritimeUtils.calculate_distance(
                df["lat"], df["lon"], center_lat, center_lon
            )

        return df


class AnomalyDetectionBuilder(BaseDatasetBuilder):
    """
    Dataset builder for maritime anomaly detection using unsupervised learning.

    This builder creates sequences for autoencoder-based anomaly detection,
    where the model learns to reconstruct normal behavior. Anomalies are
    detected by reconstruction error, not hard-coded rules.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for anomaly detection."""
        features_df = df.copy()

        # Core features
        feature_columns = [
            "lat",
            "lon",
            "sog",
            "cog",
            "heading",
            "turn",
            "status",
            "shiptype",
            "to_bow",
            "to_stern",
            "to_port",
            "to_starboard",
        ]

        # Add behavioral features
        features_df = self._add_behavioral_features(features_df)
        features_df = self._add_statistical_features(features_df)
        features_df = self._add_contextual_features(features_df)

        # Select available features
        available_features = [
            col for col in feature_columns if col in features_df.columns
        ]
        derived_features = [
            col
            for col in features_df.columns
            if col.startswith(("behavioral_", "statistical_", "contextual_"))
        ]

        final_features = available_features + derived_features
        return features_df[["mmsi", "time"] + final_features]

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build targets for unsupervised anomaly detection.

        For autoencoder-based anomaly detection, the target is the input itself
        (reconstruction task). The model learns to reconstruct normal patterns,
        and anomalies are detected by high reconstruction error.

        Returns:
            DataFrame with same features as input (for reconstruction)
        """
        # For unsupervised anomaly detection (autoencoders), target = input
        # The model learns to reconstruct normal behavior
        return df.copy()

    def create_sequences(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for unsupervised anomaly detection.

        For autoencoders, both input (X) and target (y) are the same sequences.
        The model learns to reconstruct the input, and anomalies are detected
        by high reconstruction error during inference.
        """
        sequences_X = []
        sequences_y = []

        for _mmsi, vessel_data in features_df.groupby("mmsi"):
            vessel_df = vessel_data.sort_values("time").reset_index(drop=True)

            if len(vessel_df) < self.config.sequence_length:
                continue

            # Extract feature columns
            feature_cols = [
                col for col in vessel_df.columns if col not in ["mmsi", "time"]
            ]
            vessel_features = vessel_df[feature_cols].values

            # Create sequences where target = input (reconstruction)
            for i in range(len(vessel_features) - self.config.sequence_length + 1):
                X_seq = vessel_features[i : i + self.config.sequence_length]
                y_seq = X_seq.copy()  # Target is same as input for reconstruction

                sequences_X.append(X_seq)
                sequences_y.append(y_seq)

        if not sequences_X:
            return np.array([]), np.array([])

        X = np.array(sequences_X)
        y = np.array(sequences_y)

        self.logger.info(
            f"Created {len(X)} anomaly detection sequences with shape X: {X.shape}, y: {y.shape}"
        )
        return X, y

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral pattern features."""
        # Speed patterns
        if "sog" in df.columns:
            df["behavioral_speed_std"] = df.groupby("mmsi")["sog"].transform("std")
            df["behavioral_speed_mean"] = df.groupby("mmsi")["sog"].transform("mean")

        # Course patterns
        if "cog" in df.columns:
            df["behavioral_course_std"] = df.groupby("mmsi")["cog"].transform("std")

        # Turn patterns
        if "turn" in df.columns:
            df["behavioral_turn_abs_mean"] = df.groupby("mmsi")["turn"].transform(
                lambda x: np.abs(x).mean()
            )

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features over time windows."""
        # Rolling statistics for speed
        if "sog" in df.columns:
            df["statistical_speed_rolling_mean"] = df.groupby("mmsi")["sog"].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            df["statistical_speed_rolling_std"] = df.groupby("mmsi")["sog"].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )

        return df

    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features."""
        # Time-based context
        if "time" in df.columns:
            # Define time thresholds
            NIGHT_START_HOUR = 18
            NIGHT_END_HOUR = 6
            WEEKEND_START_DAY = 5  # Saturday (0=Monday)

            df["contextual_is_night"] = (df["time"].dt.hour < NIGHT_END_HOUR) | (
                df["time"].dt.hour > NIGHT_START_HOUR
            )
            df["contextual_is_weekend"] = df["time"].dt.dayofweek >= WEEKEND_START_DAY

        return df


class GraphNetworkBuilder(BaseDatasetBuilder):
    """Dataset builder for graph neural network applications."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build node and edge features for graph networks."""
        features_df = df.copy()

        # Node features
        features_df = self._add_node_features(features_df)
        features_df = self._add_edge_features(features_df)
        features_df = self._add_graph_features(features_df)

        return features_df

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build targets for graph network tasks."""
        # For demonstration: predict vessel interactions
        targets_df = df[["mmsi", "time"]].copy()
        targets_df["interaction_score"] = self._calculate_interaction_scores(df)
        return targets_df

    def create_sequences(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create graph sequences with node and edge features."""
        # This is a simplified implementation
        # In practice, you'd create proper graph structures

        sequences_X = []
        sequences_y = []

        # Group by time windows
        time_windows = pd.Grouper(key="time", freq="10min")
        MIN_NODES_FOR_GRAPH = 2  # Minimum nodes to form a graph

        for _time_group, window_df in features_df.groupby(time_windows):
            if len(window_df) < MIN_NODES_FOR_GRAPH:
                continue

            # Create node feature matrix
            node_features = self._create_node_features(window_df)

            # Create adjacency matrix
            adjacency = self._create_adjacency_matrix(window_df)

            # Create edge features
            edge_features = self._create_edge_features(window_df)

            # Combine into graph representation
            graph_features = {
                "nodes": node_features,
                "adjacency": adjacency,
                "edges": edge_features,
            }

            # Target: future interaction patterns
            targets = self._create_graph_targets(window_df)

            sequences_X.append(graph_features)
            sequences_y.append(targets)

        # Convert to arrays (simplified - in practice you'd use graph libraries)
        if sequences_X:
            # Flatten for demonstration
            X = np.array(
                [
                    np.concatenate([g["nodes"].flatten(), g["adjacency"].flatten()])
                    for g in sequences_X
                ]
            )
            y = np.array(sequences_y)
        else:
            X = np.array([])
            y = np.array([])

        self.logger.info(
            f"Created {len(X)} graph sequences with shape X: {X.shape}, y: {y.shape}"
        )
        return X, y

    def _add_node_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add node-specific features."""
        # Vessel characteristics as node features
        if "shiptype" in df.columns:
            df["node_vessel_type"] = df["shiptype"]

        if all(col in df.columns for col in ["to_bow", "to_stern"]):
            df["node_vessel_length"] = df["to_bow"] + df["to_stern"]

        return df

    def _add_edge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add edge-specific features."""
        # Distance-based edge features
        df["edge_proximity"] = 0.0  # Placeholder
        return df

    def _add_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add graph-level features."""
        # Graph density, clustering, etc.
        df["graph_density"] = 0.0  # Placeholder
        return df

    def _calculate_interaction_scores(self, df: pd.DataFrame) -> pd.Series:
        """Calculate vessel interaction scores."""
        # Simplified interaction scoring
        return pd.Series(0.0, index=df.index)

    def _create_node_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create node feature matrix."""
        node_cols = ["lat", "lon", "sog", "cog"]
        available_cols = [col for col in node_cols if col in df.columns]
        return df[available_cols].values

    def _create_adjacency_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create adjacency matrix based on proximity (VECTORIZED for performance)."""
        n_nodes = len(df)
        adjacency = np.zeros((n_nodes, n_nodes))

        if all(col in df.columns for col in ["lat", "lon"]):
            # Extract positions
            lats = df["lat"].values
            lons = df["lon"].values

            # Vectorized pairwise Haversine distance calculation using broadcasting
            # Convert to radians
            lats_rad = np.radians(lats)
            lons_rad = np.radians(lons)

            # Broadcasting: reshape to (N, 1) and (1, N) for pairwise computation
            lat1 = lats_rad[:, np.newaxis]  # Shape: (N, 1)
            lat2 = lats_rad[np.newaxis, :]  # Shape: (1, N)
            lon1 = lons_rad[:, np.newaxis]  # Shape: (N, 1)
            lon2 = lons_rad[np.newaxis, :]  # Shape: (1, N)

            # Haversine formula (all pairwise distances at once!)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(a))

            # Distance matrix in km
            R = 6371.0  # Earth radius in km
            distance_matrix = R * c  # Shape: (N, N)

            # Create adjacency based on distance threshold
            PROXIMITY_THRESHOLD_KM = 5.0  # Vessels within 5km are connected
            adjacency = (distance_matrix < PROXIMITY_THRESHOLD_KM).astype(int)

            # Remove self-loops (diagonal should be 0)
            np.fill_diagonal(adjacency, 0)

        return adjacency

    def _create_edge_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create edge feature matrix."""
        n_nodes = len(df)
        # Simplified: just distance features
        return np.zeros((n_nodes, n_nodes, 1))

    def _create_graph_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Create graph-level targets."""
        # Simplified: predict number of interactions
        return np.array([len(df)])


class CollisionAvoidanceBuilder(BaseDatasetBuilder):
    """Dataset builder for collision avoidance systems."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for collision avoidance."""
        features_df = df.copy()

        # High-precision movement features
        feature_columns = [
            "lat",
            "lon",
            "sog",
            "cog",
            "heading",
            "turn",
            "accuracy",
            "second",
            "maneuver",
            "raim",
        ]

        # Add collision-specific features
        features_df = self._add_collision_features(features_df)
        features_df = self._add_risk_features(features_df)

        available_features = [
            col for col in feature_columns if col in features_df.columns
        ]
        derived_features = [
            col
            for col in features_df.columns
            if col.startswith(("collision_", "risk_"))
        ]

        final_features = available_features + derived_features
        return features_df[["mmsi", "time"] + final_features]

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build collision risk targets."""
        targets_df = df[["mmsi", "time"]].copy()
        targets_df["collision_risk"] = self._calculate_collision_risk(df)
        targets_df["time_to_collision"] = self._calculate_time_to_collision(df)
        return targets_df

    def create_sequences(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for collision avoidance."""
        sequences_X = []
        sequences_y = []

        # Build targets
        targets_df = self.build_targets(features_df)

        for mmsi, vessel_data in features_df.groupby("mmsi"):
            vessel_df = vessel_data.sort_values("time").reset_index(drop=True)
            vessel_targets = (
                targets_df[targets_df["mmsi"] == mmsi]
                .sort_values("time")
                .reset_index(drop=True)
            )

            if len(vessel_df) < self.config.sequence_length:
                continue

            feature_cols = [
                col for col in vessel_df.columns if col not in ["mmsi", "time"]
            ]
            vessel_features = vessel_df[feature_cols].values

            target_cols = ["collision_risk", "time_to_collision"]
            vessel_targets_values = vessel_targets[target_cols].values

            for i in range(len(vessel_features) - self.config.sequence_length + 1):
                X_seq = vessel_features[i : i + self.config.sequence_length]
                y_seq = vessel_targets_values[i + self.config.sequence_length - 1]

                sequences_X.append(X_seq)
                sequences_y.append(y_seq)

        if not sequences_X:
            return np.array([]), np.array([])

        X = np.array(sequences_X)
        y = np.array(sequences_y)

        self.logger.info(
            f"Created {len(X)} collision avoidance sequences with shape X: {X.shape}, y: {y.shape}"
        )
        return X, y

    def _add_collision_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add collision-specific features."""
        # Relative motion features would be added here
        # This requires multi-vessel analysis
        df["collision_relative_speed"] = 0.0  # Placeholder
        df["collision_relative_bearing"] = 0.0  # Placeholder
        return df

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk assessment features."""
        # Risk factors based on speed, maneuverability, etc.
        if "sog" in df.columns:
            df["risk_speed_factor"] = np.clip(
                df["sog"] / 20.0, 0, 1
            )  # Normalize to 0-1

        return df

    def _calculate_collision_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate collision risk scores."""
        # Simplified risk calculation
        return pd.Series(0.0, index=df.index)

    def _calculate_time_to_collision(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time to closest point of approach."""
        # Simplified calculation
        return pd.Series(float("inf"), index=df.index)
