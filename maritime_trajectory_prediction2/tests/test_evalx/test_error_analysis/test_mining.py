"""
Comprehensive tests for the FailureMiner framework.

Tests cover:
- Failure case extraction
- Clustering analysis
- Case study generation
- Maritime-specific characterizations
- Error handling and edge cases
"""

import numpy as np
import pandas as pd
import pytest

from src.evalx.error_analysis.mining import (
    FailureCase,
    FailureCluster,
    FailureMiner,
    FailureMiningResult,
)


class TestFailureMiner:
    """Test suite for FailureMiner class."""

    @pytest.fixture
    def miner(self):
        """Create FailureMiner instance for testing."""
        return FailureMiner(k_worst=20, n_clusters=3, random_state=42)

    @pytest.fixture
    def sample_failure_data(self):
        """Generate sample data for failure mining tests."""
        np.random.seed(42)  # Reproducible tests

        n_samples = 50
        n_features = 4

        # Generate errors with some clear worst cases
        errors = np.random.exponential(scale=0.5, size=n_samples)
        errors[-10:] = np.random.uniform(
            2.0, 5.0, size=10
        )  # Make last 10 clearly worst

        # Generate features with some structure for clustering
        features = np.random.randn(n_samples, n_features)

        # Add structure to worst cases
        features[-10:-5] += [2, 0, 1, -1]  # Cluster 1
        features[-5:] += [-1, 2, -1, 1]  # Cluster 2

        # Create realistic metadata
        metadata = {
            "vessel_type": np.random.choice(
                [70, 71, 80, 81, 30, 60, 0], size=n_samples
            ),
            "distance_to_port_km": np.random.uniform(0, 50, size=n_samples),
            "mmsi": np.arange(1000, 1000 + n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1H"),
        }

        # Generate synthetic predictions and targets
        predictions = np.random.randn(n_samples, 3, 2) * 0.1
        targets = np.random.randn(n_samples, 3, 2) * 0.1

        return errors, features, metadata, predictions, targets

    def test_basic_failure_mining(self, miner, sample_failure_data):
        """Test basic failure mining functionality."""
        errors, features, metadata, predictions, targets = sample_failure_data

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            predictions=predictions,
            targets=targets,
        )

        # Check result structure
        assert isinstance(result, FailureMiningResult)
        assert len(result.worst_cases) == 20  # k_worst
        assert isinstance(result.clusters, list)
        assert len(result.cluster_assignments) == 20
        assert isinstance(result.silhouette_score, float)
        assert isinstance(result.feature_names, list)
        assert isinstance(result.summary_statistics, dict)

        # Check worst cases are actually worst
        worst_errors = [case.error_magnitude for case in result.worst_cases]
        assert all(
            e1 >= e2
            for e1, e2 in zip(worst_errors[:-1], worst_errors[1:], strict=False)
        )  # Descending order

        # Check that worst cases have highest errors
        all_errors_sorted = np.sort(errors)[::-1]
        expected_worst_errors = all_errors_sorted[:20]
        np.testing.assert_array_almost_equal(
            sorted(worst_errors, reverse=True), expected_worst_errors
        )

    def test_failure_case_creation(self, miner, sample_failure_data):
        """Test FailureCase object creation."""
        errors, features, metadata, predictions, targets = sample_failure_data

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            predictions=predictions,
            targets=targets,
            k_worst=5,
        )

        # Check failure cases
        for _i, case in enumerate(result.worst_cases):
            assert isinstance(case, FailureCase)
            assert isinstance(case.case_id, str)
            assert case.error_magnitude > 0
            assert isinstance(case.features, dict)
            assert isinstance(case.metadata, dict)
            assert case.prediction.shape == (3, 2)  # Shape from sample data
            assert case.target.shape == (3, 2)
            assert case.feature_vector is not None
            assert case.feature_vector.shape == (4,)  # n_features

    def test_clustering_analysis(self, miner, sample_failure_data):
        """Test failure clustering functionality."""
        errors, features, metadata, predictions, targets = sample_failure_data

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            k_worst=15,  # Enough for clustering
        )

        # Check clustering results
        assert len(result.clusters) <= miner.n_clusters
        assert len(result.cluster_assignments) == 15

        # Check cluster objects
        for cluster in result.clusters:
            assert isinstance(cluster, FailureCluster)
            assert cluster.cluster_id >= 0
            assert cluster.n_cases > 0
            assert cluster.mean_error > 0
            assert cluster.std_error >= 0
            assert isinstance(cluster.dominant_features, dict)
            assert isinstance(cluster.representative_cases, list)
            assert isinstance(cluster.characterization, str)
            assert len(cluster.representative_cases) <= 3

        # Check that all cases are assigned to clusters
        assigned_cases = sum(cluster.n_cases for cluster in result.clusters)
        assert assigned_cases == 15

    def test_case_study_generation(self, miner, sample_failure_data):
        """Test case study card generation."""
        errors, features, metadata, predictions, targets = sample_failure_data

        result = miner.mine_failures(
            errors=errors, features=features, metadata=metadata, k_worst=5
        )

        # Generate case cards
        case_cards = miner.generate_case_cards(result.worst_cases)

        assert len(case_cards) == 5

        for card in case_cards:
            assert isinstance(card, dict)
            assert "case_id" in card
            assert "error_magnitude" in card
            assert "features_summary" in card
            assert "metadata_summary" in card
            assert "prediction_target_diff" in card
            assert "recommendations" in card

            # Check card content
            assert card["error_magnitude"] > 0
            assert isinstance(card["features_summary"], dict)
            assert isinstance(card["metadata_summary"], dict)
            assert isinstance(card["prediction_target_diff"], dict)
            assert isinstance(card["recommendations"], list)
            assert len(card["recommendations"]) > 0

    def test_maritime_characterization(self, miner):
        """Test maritime-specific cluster characterization."""
        # Create controlled data with clear maritime patterns
        errors = np.array([3.0, 2.8, 2.5, 1.0, 0.8])
        features = np.array(
            [
                [1, 2, 3, 4],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [0, 0, 0, 0],
                [0.1, 0.1, 0.1, 0.1],
            ]
        )

        metadata = {
            "vessel_type": np.array([70, 70, 71, 80, 81]),  # Mostly cargo, some tanker
            "distance_to_port_km": np.array(
                [2.0, 3.0, 4.0, 25.0, 30.0]
            ),  # Close and far
            "mmsi": np.arange(1000, 1005),
        }

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            k_worst=5,
            predictions=np.random.randn(5, 2, 2),
            targets=np.random.randn(5, 2, 2),
        )

        # Check that characterizations include maritime information
        for cluster in result.clusters:
            characterization = cluster.characterization.lower()

            # Should mention maritime concepts
            maritime_terms = ["vessel", "cargo", "tanker", "port", "cluster", "error"]
            has_maritime_terms = any(
                term in characterization for term in maritime_terms
            )
            assert (
                has_maritime_terms
            ), f"Characterization missing maritime terms: {characterization}"

    def test_summary_statistics(self, miner, sample_failure_data):
        """Test summary statistics generation."""
        errors, features, metadata, predictions, targets = sample_failure_data

        result = miner.mine_failures(
            errors=errors, features=features, metadata=metadata, k_worst=15
        )

        stats = result.summary_statistics

        # Check required statistics
        required_keys = [
            "n_worst_cases",
            "mean_worst_error",
            "std_worst_error",
            "min_worst_error",
            "max_worst_error",
            "n_clusters",
            "cluster_sizes",
        ]
        for key in required_keys:
            assert key in stats

        # Check values
        assert stats["n_worst_cases"] == 15
        assert stats["mean_worst_error"] > 0
        assert stats["std_worst_error"] >= 0
        assert stats["min_worst_error"] > 0
        assert stats["max_worst_error"] >= stats["min_worst_error"]
        assert stats["n_clusters"] >= 0
        assert isinstance(stats["cluster_sizes"], list)

        if result.clusters:
            assert "cluster_error_range" in stats
            assert isinstance(stats["cluster_error_range"], tuple)
            assert len(stats["cluster_error_range"]) == 2

    def test_input_validation(self, miner):
        """Test input validation and error handling."""
        # Test mismatched array lengths
        errors = np.array([1, 2, 3])
        features = np.array([[1, 2], [3, 4]])  # Wrong length
        metadata = {"vessel_type": [70, 80, 30]}

        with pytest.raises(ValueError, match="Errors length .* != features length"):
            miner.mine_failures(errors, features, metadata)

    def test_insufficient_samples_for_clustering(self, miner):
        """Test behavior when k_worst is too small for clustering."""
        errors = np.array([1.0, 0.8])
        features = np.array([[1, 2], [3, 4]])
        metadata = {"vessel_type": [70, 80]}

        with pytest.warns(UserWarning, match="Too few worst cases"):
            result = miner.mine_failures(
                errors=errors,
                features=features,
                metadata=metadata,
                k_worst=2,  # Less than n_clusters=3
            )

        assert len(result.clusters) == 0
        assert result.silhouette_score == 0.0

    def test_k_worst_exceeds_samples(self, miner):
        """Test behavior when k_worst exceeds number of samples."""
        errors = np.array([1.0, 0.8, 0.5])
        features = np.array([[1, 2], [3, 4], [5, 6]])
        metadata = {"vessel_type": [70, 80, 30]}

        with pytest.warns(UserWarning, match="k_worst .* > n_samples"):
            result = miner.mine_failures(
                errors=errors,
                features=features,
                metadata=metadata,
                k_worst=10,  # More than 3 samples
            )

        assert len(result.worst_cases) == 3  # Should use all samples

    def test_without_predictions_targets(self, miner, sample_failure_data):
        """Test mining without predictions and targets."""
        errors, features, metadata, _, _ = sample_failure_data

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            # No predictions or targets
            k_worst=10,
        )

        # Should still work
        assert len(result.worst_cases) == 10

        # Check that failure cases have empty arrays for predictions/targets
        for case in result.worst_cases:
            assert case.prediction.size == 0
            assert case.target.size == 0

        # Case cards should handle missing predictions/targets
        case_cards = miner.generate_case_cards(result.worst_cases[:3])
        for card in case_cards:
            assert not card["prediction_target_diff"]["available"]

    def test_feature_name_generation(self, miner):
        """Test feature name generation."""
        errors = np.array([2.0, 1.5, 1.0])
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test without provided feature names
        metadata = {"vessel_type": [70, 80, 30]}
        result = miner.mine_failures(errors, features, metadata, k_worst=3)

        expected_names = ["feature_0", "feature_1", "feature_2"]
        assert result.feature_names == expected_names

        # Test with provided feature names
        metadata["feature_names"] = ["lat", "lon", "speed"]
        result = miner.mine_failures(errors, features, metadata, k_worst=3)
        assert result.feature_names == ["lat", "lon", "speed"]

    def test_cluster_representative_cases(self, miner):
        """Test selection of representative cases in clusters."""
        # Create data where we can control clustering
        errors = np.array([3.0, 2.8, 2.5, 2.3, 2.0])

        # Two clear clusters in feature space
        features = np.array(
            [
                [0, 0],  # Cluster 1 center
                [0.1, 0.1],  # Cluster 1
                [0.2, 0.2],  # Cluster 1
                [5, 5],  # Cluster 2 center
                [5.1, 5.1],  # Cluster 2
            ]
        )

        metadata = {"vessel_type": [70, 70, 70, 80, 80], "mmsi": np.arange(1000, 1005)}

        result = miner.mine_failures(
            errors=errors,
            features=features,
            metadata=metadata,
            k_worst=5,
            predictions=np.random.randn(5, 2, 2),
            targets=np.random.randn(5, 2, 2),
        )

        # Check that clusters have representative cases
        for cluster in result.clusters:
            assert len(cluster.representative_cases) <= 3
            assert len(cluster.representative_cases) <= cluster.n_cases

            # Representative cases should be actual failure cases
            for rep_case in cluster.representative_cases:
                assert isinstance(rep_case, FailureCase)
                assert rep_case in result.worst_cases

    def test_vessel_specific_recommendations(self, miner):
        """Test vessel-specific recommendation generation."""
        # Create cases with specific vessel types
        errors = np.array([2.0, 1.8, 1.5])
        features = np.array([[1, 2], [3, 4], [5, 6]])
        metadata = {
            "vessel_type": [70, 80, 30],  # Cargo, Tanker, Fishing
            "distance_to_port_km": [2, 15, 25],
            "mmsi": [1001, 1002, 1003],
        }

        result = miner.mine_failures(
            errors=errors, features=features, metadata=metadata, k_worst=3
        )

        case_cards = miner.generate_case_cards(result.worst_cases)

        # Check that recommendations are vessel-specific
        cargo_card = next(
            card for card in case_cards if card["case_id"].endswith("0_0")
        )  # First case
        tanker_card = next(
            card for card in case_cards if card["case_id"].endswith("1_1")
        )  # Second case
        fishing_card = next(
            card for card in case_cards if card["case_id"].endswith("2_2")
        )  # Third case

        # Check for vessel-specific recommendations
        cargo_recs = " ".join(cargo_card["recommendations"]).lower()
        tanker_recs = " ".join(tanker_card["recommendations"]).lower()
        fishing_recs = " ".join(fishing_card["recommendations"]).lower()

        # Should have different recommendations based on vessel type
        assert "cargo" in cargo_recs or "movement patterns" in cargo_recs
        assert "tanker" in tanker_recs or "turning behavior" in tanker_recs
        assert "fishing" in fishing_recs or "irregular" in fishing_recs


class TestFailureCase:
    """Test FailureCase data class."""

    def test_failure_case_creation(self):
        """Test FailureCase creation and attributes."""
        case = FailureCase(
            case_id="test_case_1",
            error_magnitude=1.5,
            features={"vessel_type": 70, "port_distance": 5.0},
            prediction=np.array([[1, 2], [3, 4]]),
            target=np.array([[1.1, 2.1], [3.1, 4.1]]),
            metadata={"mmsi": 123456},
            cluster_id=2,
            feature_vector=np.array([0.5, 1.0, -0.5]),
        )

        assert case.case_id == "test_case_1"
        assert case.error_magnitude == 1.5
        assert case.features["vessel_type"] == 70
        assert case.cluster_id == 2
        assert np.array_equal(case.feature_vector, [0.5, 1.0, -0.5])


class TestFailureCluster:
    """Test FailureCluster data class."""

    def test_failure_cluster_creation(self):
        """Test FailureCluster creation and attributes."""
        case1 = FailureCase("case1", 1.5, {}, np.array([]), np.array([]), {})
        case2 = FailureCase("case2", 1.8, {}, np.array([]), np.array([]), {})

        cluster = FailureCluster(
            cluster_id=1,
            n_cases=2,
            mean_error=1.65,
            std_error=0.15,
            dominant_features={"vessel_type": 70},
            representative_cases=[case1, case2],
            characterization="Test cluster",
        )

        assert cluster.cluster_id == 1
        assert cluster.n_cases == 2
        assert cluster.mean_error == 1.65
        assert cluster.std_error == 0.15
        assert cluster.dominant_features["vessel_type"] == 70
        assert len(cluster.representative_cases) == 2
        assert cluster.characterization == "Test cluster"


class TestFailureMiningResult:
    """Test FailureMiningResult data class."""

    def test_failure_mining_result_creation(self):
        """Test FailureMiningResult creation and attributes."""
        case1 = FailureCase("case1", 1.5, {}, np.array([]), np.array([]), {})
        cluster1 = FailureCluster(1, 1, 1.5, 0.0, {}, [case1], "Test")

        result = FailureMiningResult(
            worst_cases=[case1],
            clusters=[cluster1],
            cluster_assignments=np.array([0]),
            silhouette_score=0.8,
            feature_names=["feat1", "feat2"],
            summary_statistics={"n_cases": 1},
        )

        assert len(result.worst_cases) == 1
        assert len(result.clusters) == 1
        assert len(result.cluster_assignments) == 1
        assert result.silhouette_score == 0.8
        assert result.feature_names == ["feat1", "feat2"]
        assert result.summary_statistics["n_cases"] == 1
