"""
Failure mining framework for identifying and analyzing worst-performing cases.

This module provides tools to:
- Extract worst-performing prediction cases
- Cluster failures by feature patterns
- Generate interpretable failure characterizations
- Create actionable case study documentation
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class FailureCase:
    """Individual failure case with context and analysis."""

    case_id: str
    error_magnitude: float
    features: dict[str, Any]
    prediction: np.ndarray
    target: np.ndarray
    metadata: dict[str, Any]
    cluster_id: int | None = None
    feature_vector: np.ndarray | None = None


@dataclass
class FailureCluster:
    """Cluster of similar failure cases."""

    cluster_id: int
    n_cases: int
    mean_error: float
    std_error: float
    dominant_features: dict[str, Any]
    representative_cases: list[FailureCase]
    characterization: str
    feature_importance: dict[str, float] | None = None


@dataclass
class FailureMiningResult:
    """Result of failure mining analysis."""

    worst_cases: list[FailureCase]
    clusters: list[FailureCluster]
    cluster_assignments: np.ndarray
    silhouette_score: float
    feature_names: list[str]
    summary_statistics: dict[str, Any]


class FailureMiner:
    """
    Identify and characterize worst-performing prediction cases.

    This class implements systematic failure analysis to understand
    where and why models fail, providing actionable insights for
    model improvement through:

    1. Extracting top-k worst performing cases
    2. Clustering failures in feature space
    3. Characterizing cluster patterns
    4. Generating interpretable failure descriptions

    Example:
        ```python
        miner = FailureMiner(k_worst=100, n_clusters=5)

        failure_result = miner.mine_failures(
            errors=prediction_errors,
            features=feature_matrix,
            metadata=sample_metadata
        )

        # Access worst cases
        worst_cases = failure_result.worst_cases

        # Access failure clusters
        for cluster in failure_result.clusters:
            print(f"Cluster {cluster.cluster_id}: {cluster.characterization}")
        ```
    """

    # Distance thresholds for port proximity (in km)
    VERY_CLOSE_TO_PORT_KM = 5
    NEAR_PORT_KM = 20

    # Error magnitude thresholds
    HIGH_ERROR_THRESHOLD = 1.0
    CRITICAL_ERROR_THRESHOLD = 2.0

    # Vessel type codes
    VESSEL_TYPE_FISHING = 30

    def __init__(
        self,
        k_worst: int = 100,
        n_clusters: int = 5,
        clustering_algorithm: str = "kmeans",
        random_state: int | None = 42,
    ):
        """
        Initialize FailureMiner.

        Args:
            k_worst: Number of worst cases to extract for analysis
            n_clusters: Number of clusters for failure grouping
            clustering_algorithm: Clustering algorithm ('kmeans', 'agglomerative')
            random_state: Random seed for reproducibility
        """
        self.k_worst = k_worst
        self.n_clusters = n_clusters
        self.clustering_algorithm = clustering_algorithm
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clusterer = None

    def mine_failures(
        self,
        errors: np.ndarray,
        features: np.ndarray,
        metadata: dict[str, Any],
        predictions: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        k_worst: int | None = None,
    ) -> FailureMiningResult:
        """
        Extract and cluster worst-performing cases.

        Args:
            errors: Per-sample error magnitudes shape [n_samples]
            features: Feature matrix shape [n_samples, n_features]
            metadata: Dictionary with sample metadata
            predictions: Model predictions shape [n_samples, ...] (optional)
            targets: Ground truth targets shape [n_samples, ...] (optional)
            k_worst: Override default k_worst parameter

        Returns:
            FailureMiningResult with worst cases and clusters
        """
        if k_worst is None:
            k_worst = self.k_worst

        # Validate inputs
        if len(errors) != len(features):
            raise ValueError(
                f"Errors length {len(errors)} != features length {len(features)}"
            )

        if k_worst > len(errors):
            warnings.warn(
                f"k_worst ({k_worst}) > n_samples ({len(errors)}), using all samples",
                stacklevel=2,
            )
            k_worst = len(errors)

        # Extract worst cases
        worst_indices = self._extract_worst_cases(errors, k_worst)
        worst_errors = errors[worst_indices]
        worst_features = features[worst_indices]

        # Create failure cases
        worst_cases = self._create_failure_cases(
            worst_indices, worst_errors, worst_features, metadata, predictions, targets
        )

        # Cluster failures
        if k_worst >= self.n_clusters:
            clusters, cluster_assignments, silhouette = self._cluster_failures(
                worst_features, worst_cases
            )
        else:
            # Not enough cases for clustering
            clusters = []
            cluster_assignments = np.zeros(len(worst_cases))
            silhouette = 0.0
            warnings.warn(
                f"Too few worst cases ({k_worst}) for clustering ({self.n_clusters} clusters)",
                stacklevel=2,
            )

        # Generate feature names
        feature_names = self._generate_feature_names(features.shape[1], metadata)

        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(worst_cases, clusters)

        return FailureMiningResult(
            worst_cases=worst_cases,
            clusters=clusters,
            cluster_assignments=cluster_assignments,
            silhouette_score=silhouette,
            feature_names=feature_names,
            summary_statistics=summary_stats,
        )

    def _extract_worst_cases(self, errors: np.ndarray, k_worst: int) -> np.ndarray:
        """Extract indices of k worst performing cases."""
        # Get indices of worst cases (largest errors)
        worst_indices = np.argpartition(errors, -k_worst)[-k_worst:]

        # Sort by error magnitude (largest first)
        sorted_indices = worst_indices[np.argsort(errors[worst_indices])[::-1]]

        return sorted_indices

    def _create_failure_cases(
        self,
        indices: np.ndarray,
        errors: np.ndarray,
        features: np.ndarray,
        metadata: dict[str, Any],
        predictions: np.ndarray | None = None,
        targets: np.ndarray | None = None,
    ) -> list[FailureCase]:
        """Create FailureCase objects for worst performing samples."""
        failure_cases = []

        for i, idx in enumerate(indices):
            # Extract metadata for this sample
            sample_metadata = {}
            for key, values in metadata.items():
                if isinstance(values, list | np.ndarray) and len(values) > idx:
                    sample_metadata[key] = values[idx]

            # Extract feature dictionary
            feature_dict = {
                f"feature_{j}": features[i, j] for j in range(features.shape[1])
            }

            # Add metadata features if available
            if "vessel_type" in sample_metadata:
                feature_dict["vessel_type"] = sample_metadata["vessel_type"]
            if "distance_to_port_km" in sample_metadata:
                feature_dict["port_distance"] = sample_metadata["distance_to_port_km"]

            failure_case = FailureCase(
                case_id=f"failure_{idx}_{i}",
                error_magnitude=errors[i],
                features=feature_dict,
                prediction=predictions[idx]
                if predictions is not None
                else np.array([]),
                target=targets[idx] if targets is not None else np.array([]),
                metadata=sample_metadata,
                feature_vector=features[i],
            )

            failure_cases.append(failure_case)

        return failure_cases

    def _cluster_failures(
        self, features: np.ndarray, failure_cases: list[FailureCase]
    ) -> tuple[list[FailureCluster], np.ndarray, float]:
        """Cluster failure cases in feature space."""
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Perform clustering
        if self.clustering_algorithm == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
        else:
            raise ValueError(
                f"Unsupported clustering algorithm: {self.clustering_algorithm}"
            )

        cluster_assignments = self.clusterer.fit_predict(features_scaled)

        # Compute silhouette score
        try:
            silhouette = silhouette_score(features_scaled, cluster_assignments)
        except Exception:
            silhouette = 0.0

        # Create cluster objects
        clusters = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_cases = [
                case for i, case in enumerate(failure_cases) if cluster_mask[i]
            ]

            if len(cluster_cases) > 0:
                cluster = self._create_failure_cluster(
                    cluster_id, cluster_cases, features_scaled[cluster_mask]
                )
                clusters.append(cluster)

        return clusters, cluster_assignments, silhouette

    def _create_failure_cluster(
        self,
        cluster_id: int,
        cluster_cases: list[FailureCase],
        cluster_features: np.ndarray,
    ) -> FailureCluster:
        """Create a FailureCluster object."""
        errors = [case.error_magnitude for case in cluster_cases]
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # Find representative cases (closest to centroid)
        if len(cluster_features) > 0:
            centroid = np.mean(cluster_features, axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_indices = np.argsort(distances)[: min(3, len(cluster_cases))]
            representative_cases = [cluster_cases[i] for i in closest_indices]
        else:
            representative_cases = cluster_cases[:3]

        # Analyze dominant features
        dominant_features = self._analyze_dominant_features(cluster_cases)

        # Generate characterization
        characterization = self._generate_cluster_characterization(
            cluster_id, cluster_cases, dominant_features
        )

        return FailureCluster(
            cluster_id=cluster_id,
            n_cases=len(cluster_cases),
            mean_error=mean_error,
            std_error=std_error,
            dominant_features=dominant_features,
            representative_cases=representative_cases,
            characterization=characterization,
        )

    def _analyze_dominant_features(
        self, cluster_cases: list[FailureCase]
    ) -> dict[str, Any]:
        """Analyze dominant features in a cluster."""
        if not cluster_cases:
            return {}

        dominant_features = {}

        # Analyze vessel types
        vessel_types = []
        port_distances = []

        for case in cluster_cases:
            if "vessel_type" in case.features:
                vessel_types.append(case.features["vessel_type"])
            if "port_distance" in case.features:
                port_distances.append(case.features["port_distance"])

        # Most common vessel type
        if vessel_types:
            vessel_type_counts = pd.Series(vessel_types).value_counts()
            dominant_features["dominant_vessel_type"] = vessel_type_counts.index[0]
            dominant_features["vessel_type_percentage"] = vessel_type_counts.iloc[
                0
            ] / len(vessel_types)

        # Average port distance
        if port_distances:
            dominant_features["mean_port_distance"] = np.mean(port_distances)
            dominant_features["std_port_distance"] = np.std(port_distances)

        # Feature statistics
        feature_matrix = np.array(
            [
                case.feature_vector
                for case in cluster_cases
                if case.feature_vector is not None
            ]
        )
        if len(feature_matrix) > 0:
            dominant_features["feature_means"] = np.mean(feature_matrix, axis=0)
            dominant_features["feature_stds"] = np.std(feature_matrix, axis=0)

        return dominant_features

    def _generate_cluster_characterization(
        self,
        cluster_id: int,
        cluster_cases: list[FailureCase],
        dominant_features: dict[str, Any],
    ) -> str:
        """Generate human-readable characterization of cluster."""
        characterization_parts = [f"Cluster {cluster_id} ({len(cluster_cases)} cases)"]

        # Vessel type characterization
        if "dominant_vessel_type" in dominant_features:
            vtype = dominant_features["dominant_vessel_type"]
            percentage = dominant_features["vessel_type_percentage"]
            characterization_parts.append(f"{percentage:.1%} {vtype} vessels")

        # Port distance characterization
        if "mean_port_distance" in dominant_features:
            mean_dist = dominant_features["mean_port_distance"]
            if mean_dist < self.VERY_CLOSE_TO_PORT_KM:
                distance_desc = "very close to port"
            elif mean_dist < self.NEAR_PORT_KM:
                distance_desc = "near port"
            else:
                distance_desc = "far from port"
            characterization_parts.append(distance_desc)

        # Error characterization
        mean_error = np.mean([case.error_magnitude for case in cluster_cases])
        if mean_error > self.CRITICAL_ERROR_THRESHOLD:  # Assuming error units
            error_desc = "very high prediction errors"
        elif mean_error > self.HIGH_ERROR_THRESHOLD:
            error_desc = "high prediction errors"
        else:
            error_desc = "moderate prediction errors"
        characterization_parts.append(error_desc)

        return " - ".join(characterization_parts)

    def _generate_feature_names(
        self, n_features: int, metadata: dict[str, Any]
    ) -> list[str]:
        """Generate feature names for interpretation."""
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Add known feature names if available
        if "feature_names" in metadata:
            provided_names = metadata["feature_names"]
            if len(provided_names) == n_features:
                feature_names = provided_names

        return feature_names

    def _compute_summary_statistics(
        self, worst_cases: list[FailureCase], clusters: list[FailureCluster]
    ) -> dict[str, Any]:
        """Compute summary statistics for mining result."""
        errors = [case.error_magnitude for case in worst_cases]

        summary = {
            "n_worst_cases": len(worst_cases),
            "mean_worst_error": np.mean(errors),
            "std_worst_error": np.std(errors),
            "min_worst_error": np.min(errors),
            "max_worst_error": np.max(errors),
            "n_clusters": len(clusters),
            "cluster_sizes": [cluster.n_cases for cluster in clusters],
        }

        if clusters:
            cluster_errors = [cluster.mean_error for cluster in clusters]
            summary["cluster_error_range"] = (
                np.min(cluster_errors),
                np.max(cluster_errors),
            )

        return summary

    def generate_case_cards(self, failure_cases: list[FailureCase]) -> list[dict]:
        """
        Generate detailed case study cards for failure analysis.

        Args:
            failure_cases: List of FailureCase objects to document

        Returns:
            List of case study dictionaries with detailed analysis
        """
        case_cards = []

        for case in failure_cases:
            card = {
                "case_id": case.case_id,
                "error_magnitude": case.error_magnitude,
                "cluster_id": case.cluster_id,
                "features_summary": self._summarize_case_features(case),
                "metadata_summary": self._summarize_case_metadata(case),
                "prediction_target_diff": self._analyze_prediction_difference(case),
                "recommendations": self._generate_case_recommendations(case),
            }
            case_cards.append(card)

        return case_cards

    def _summarize_case_features(self, case: FailureCase) -> dict[str, Any]:
        """Summarize key features for a case."""
        summary = {}

        # Extract key features
        if "vessel_type" in case.features:
            summary["vessel_type"] = case.features["vessel_type"]
        if "port_distance" in case.features:
            summary["port_distance_km"] = case.features["port_distance"]

        # Feature vector statistics if available
        if case.feature_vector is not None:
            summary["feature_norm"] = np.linalg.norm(case.feature_vector)
            summary["feature_mean"] = np.mean(case.feature_vector)
            summary["feature_std"] = np.std(case.feature_vector)

        return summary

    def _summarize_case_metadata(self, case: FailureCase) -> dict[str, Any]:
        """Summarize metadata for a case."""
        # Return subset of metadata for case card
        relevant_keys = ["mmsi", "timestamp", "vessel_name", "nav_status"]
        return {k: v for k, v in case.metadata.items() if k in relevant_keys}

    def _analyze_prediction_difference(self, case: FailureCase) -> dict[str, Any]:
        """Analyze difference between prediction and target."""
        if case.prediction.size == 0 or case.target.size == 0:
            return {"available": False}

        diff = case.prediction - case.target

        return {
            "available": True,
            "mean_absolute_diff": np.mean(np.abs(diff)),
            "max_absolute_diff": np.max(np.abs(diff)),
            "diff_norm": np.linalg.norm(diff),
            "relative_error": np.linalg.norm(diff)
            / (np.linalg.norm(case.target) + 1e-8),
        }

    def _generate_case_recommendations(self, case: FailureCase) -> list[str]:
        """Generate recommendations for addressing this failure case."""
        recommendations = []

        # Vessel-specific recommendations
        if "vessel_type" in case.features:
            vtype = case.features["vessel_type"]
            if vtype in [70, 71]:  # Cargo vessels
                recommendations.append(
                    "Consider cargo-specific movement patterns in training"
                )
            elif vtype in [80, 81]:  # Tankers
                recommendations.append(
                    "Add tanker-specific constraints for turning behavior"
                )
            elif vtype == self.VESSEL_TYPE_FISHING:  # Fishing
                recommendations.append("Model irregular fishing vessel patterns")

        # Distance-based recommendations
        if "port_distance" in case.features:
            distance = case.features["port_distance"]
            if distance < self.VERY_CLOSE_TO_PORT_KM:
                recommendations.append("Improve near-port maneuvering predictions")
            elif distance > self.NEAR_PORT_KM:
                recommendations.append("Better open-water trajectory modeling needed")

        # Error magnitude recommendations
        if case.error_magnitude > self.CRITICAL_ERROR_THRESHOLD:
            recommendations.append("Critical case - investigate data quality")
            recommendations.append("Consider ensemble methods for difficult cases")

        if not recommendations:
            recommendations.append(
                "Further analysis needed to identify improvement strategies"
            )

        return recommendations
