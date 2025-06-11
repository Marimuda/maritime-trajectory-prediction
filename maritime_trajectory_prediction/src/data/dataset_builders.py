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
        features_df = df.copy()

        # Core position and movement features
        feature_columns = FeatureGroups.EXTENDED_TRAJECTORY[
            :6
        ]  # lat, lon, sog, cog, heading, turn

        # Add derived features
        features_df = self._add_temporal_features(features_df)
        features_df = self._add_movement_features(features_df)
        features_df = self._add_spatial_features(features_df)

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
        for mmsi, vessel_df in features_df.groupby("mmsi"):
            vessel_df = vessel_df.sort_values("time").reset_index(drop=True)

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
        """Add movement-derived features."""
        # Speed change rate
        if "sog" in df.columns:
            df["movement_speed_change"] = df.groupby("mmsi")["sog"].diff()

        # Course change rate
        if "cog" in df.columns:
            df["movement_course_change"] = df.groupby("mmsi")["cog"].diff()

        # Distance traveled
        if all(col in df.columns for col in ["lat", "lon"]):
            df["movement_distance"] = 0.0
            for mmsi, vessel_df in df.groupby("mmsi"):
                if len(vessel_df) > 1:
                    # Calculate distances between consecutive points
                    distances = []
                    for i in range(1, len(vessel_df)):
                        dist = MaritimeUtils.calculate_distance(
                            vessel_df.iloc[i - 1]["lat"],
                            vessel_df.iloc[i - 1]["lon"],
                            vessel_df.iloc[i]["lat"],
                            vessel_df.iloc[i]["lon"],
                        )
                        distances.append(dist)

                    # First point has 0 distance
                    all_distances = [0.0] + distances
                    df.loc[vessel_df.index, "movement_distance"] = all_distances

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
    """Dataset builder for maritime anomaly detection."""

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
        """Build targets for anomaly detection (anomaly labels)."""
        targets_df = df[["mmsi", "time"]].copy()

        # Create synthetic anomaly labels based on heuristics
        targets_df["anomaly_speed"] = self._detect_speed_anomalies(df)
        targets_df["anomaly_course"] = self._detect_course_anomalies(df)
        targets_df["anomaly_position"] = self._detect_position_anomalies(df)
        targets_df["anomaly_overall"] = (
            targets_df["anomaly_speed"]
            | targets_df["anomaly_course"]
            | targets_df["anomaly_position"]
        )

        return targets_df

    def create_sequences(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for anomaly detection."""
        sequences_X = []
        sequences_y = []

        # Build targets
        targets_df = self.build_targets(features_df)

        for mmsi, vessel_df in features_df.groupby("mmsi"):
            vessel_df = vessel_df.sort_values("time").reset_index(drop=True)
            vessel_targets = (
                targets_df[targets_df["mmsi"] == mmsi]
                .sort_values("time")
                .reset_index(drop=True)
            )

            if len(vessel_df) < self.config.sequence_length:
                continue

            # Extract feature columns
            feature_cols = [
                col for col in vessel_df.columns if col not in ["mmsi", "time"]
            ]
            vessel_features = vessel_df[feature_cols].values

            # Extract target columns
            target_cols = ["anomaly_overall"]
            vessel_targets_values = vessel_targets[target_cols].values

            # Create sequences
            for i in range(len(vessel_features) - self.config.sequence_length + 1):
                X_seq = vessel_features[i : i + self.config.sequence_length]
                y_seq = vessel_targets_values[
                    i + self.config.sequence_length - 1
                ]  # Current anomaly label

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
            df["contextual_is_night"] = (df["time"].dt.hour < 6) | (
                df["time"].dt.hour > 18
            )
            df["contextual_is_weekend"] = df["time"].dt.dayofweek >= 5

        return df

    def _detect_speed_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect speed-based anomalies."""
        if "sog" not in df.columns:
            return pd.Series(False, index=df.index)

        # Anomaly: speed > 30 knots or speed < 0
        return (df["sog"] > 30) | (df["sog"] < 0)

    def _detect_course_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect course-based anomalies."""
        if "cog" not in df.columns:
            return pd.Series(False, index=df.index)

        # Anomaly: rapid course changes > 90 degrees
        course_change = df.groupby("mmsi")["cog"].diff().abs()
        return course_change > 90

    def _detect_position_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect position-based anomalies."""
        if not all(col in df.columns for col in ["lat", "lon"]):
            return pd.Series(False, index=df.index)

        # Anomaly: position jumps > 10 km in one time step
        anomalies = pd.Series(False, index=df.index)

        for mmsi, vessel_df in df.groupby("mmsi"):
            if len(vessel_df) > 1:
                # Calculate distance between consecutive points
                lats = vessel_df["lat"].values
                lons = vessel_df["lon"].values
                distances = []
                for i in range(1, len(lats)):
                    dist = MaritimeUtils.calculate_distance(
                        lats[i - 1], lons[i - 1], lats[i], lons[i]
                    )
                    distances.append(dist)
                # Assuming 1-minute intervals, 10 km jump is anomalous
                distances = np.array(distances)
                vessel_anomalies = np.zeros(len(vessel_df), dtype=bool)
                vessel_anomalies[1:] = (
                    distances > 10.0
                )  # First point can't be anomalous
                anomalies.loc[vessel_df.index] = vessel_anomalies

        return anomalies


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

        for time_group, window_df in features_df.groupby(time_windows):
            if len(window_df) < 2:  # Need at least 2 nodes for a graph
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
        """Create adjacency matrix based on proximity."""
        n_nodes = len(df)
        adjacency = np.zeros((n_nodes, n_nodes))

        if all(col in df.columns for col in ["lat", "lon"]):
            positions = df[["lat", "lon"]].values

            # Create edges based on distance threshold (e.g., 5 km)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    distance = MaritimeUtils.calculate_distance(
                        positions[i, 0],
                        positions[i, 1],
                        positions[j, 0],
                        positions[j, 1],
                    )
                    if distance < 5.0:  # 5 km threshold
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1

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

        for mmsi, vessel_df in features_df.groupby("mmsi"):
            vessel_df = vessel_df.sort_values("time").reset_index(drop=True)
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
