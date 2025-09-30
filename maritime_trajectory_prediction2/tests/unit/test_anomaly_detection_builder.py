"""
Unit tests for AnomalyDetectionBuilder.

Tests the behavior of the anomaly detection dataset builder to ensure it
creates proper reconstruction targets for unsupervised anomaly detection
(autoencoders), without hard-coded rule-based anomaly labels.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builders import AnomalyDetectionBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig


@pytest.fixture
def anomaly_config():
    """Create config for anomaly detection builder."""
    return DatasetConfig(
        task=MLTask.ANOMALY_DETECTION,
        sequence_length=20,
        prediction_horizon=0,  # Not used for anomaly detection
        validation_split=0.2,
        test_split=0.1,
        random_seed=42,
    )


@pytest.fixture
def vessel_trajectory_data():
    """Create sample vessel trajectory data for testing."""
    np.random.seed(42)
    n_points = 100

    data = {
        "mmsi": [123456] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 59.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "lon": 10.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "sog": 10.0 + np.random.randn(n_points) * 1.0,  # ~10 knots
        "cog": 45.0 + np.random.randn(n_points) * 5.0,  # ~45 degrees
        "heading": 45.0 + np.random.randn(n_points) * 5.0,
        "turn": np.random.randn(n_points) * 2.0,
    }

    return pd.DataFrame(data)


@pytest.mark.unit
class TestAnomalyDetectionBuilderTargets:
    """Test that targets are correctly built for reconstruction."""

    def test_build_targets_returns_same_structure_as_input(
        self, anomaly_config, vessel_trajectory_data
    ):
        """Test that build_targets returns same data structure for reconstruction."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        # Build features
        features_df = builder.build_features(vessel_trajectory_data)

        # Build targets
        targets_df = builder.build_targets(features_df)

        # Targets should have same structure as features (reconstruction task)
        assert set(features_df.columns) == set(targets_df.columns)
        assert len(features_df) == len(targets_df)

    def test_build_targets_no_synthetic_labels(
        self, anomaly_config, vessel_trajectory_data
    ):
        """Test that no hard-coded anomaly labels are created."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)
        targets_df = builder.build_targets(features_df)

        # Should NOT have synthetic anomaly columns
        assert "anomaly_speed" not in targets_df.columns
        assert "anomaly_course" not in targets_df.columns
        assert "anomaly_position" not in targets_df.columns
        assert "anomaly_overall" not in targets_df.columns

    def test_build_targets_preserves_feature_values(
        self, anomaly_config, vessel_trajectory_data
    ):
        """Test that target values match input feature values."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)
        targets_df = builder.build_targets(features_df)

        # For reconstruction, targets should equal inputs
        pd.testing.assert_frame_equal(features_df, targets_df)


@pytest.mark.unit
class TestAnomalyDetectionBuilderSequences:
    """Test sequence creation for reconstruction."""

    def test_create_sequences_target_equals_input(
        self, anomaly_config, vessel_trajectory_data
    ):
        """Test that target sequences equal input sequences for reconstruction."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)
        X, y = builder.create_sequences(features_df)

        # For reconstruction, y should equal X
        assert X.shape == y.shape

        # Check equality element-wise, handling NaN and mixed dtypes (float + bool)
        # Flatten arrays for easier comparison
        X_flat = X.flatten()
        y_flat = y.flatten()

        for i in range(len(X_flat)):
            x_val, y_val = X_flat[i], y_flat[i]
            # Handle NaN comparison
            if isinstance(x_val, float) and isinstance(y_val, float):
                if np.isnan(x_val) and np.isnan(y_val):
                    continue  # Both NaN, consider equal
                assert x_val == y_val, f"Mismatch at index {i}: {x_val} != {y_val}"
            else:
                assert x_val == y_val, f"Mismatch at index {i}: {x_val} != {y_val}"

    def test_create_sequences_correct_shape(
        self, anomaly_config, vessel_trajectory_data
    ):
        """Test that sequences have correct shape."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)
        X, y = builder.create_sequences(features_df)

        # Check shape: (n_sequences, sequence_length, n_features)
        assert X.ndim == 3
        assert y.ndim == 3
        assert X.shape[1] == anomaly_config.sequence_length
        assert y.shape[1] == anomaly_config.sequence_length

        # Number of features should match
        assert X.shape[2] == y.shape[2]

    def test_create_sequences_non_empty(self, anomaly_config, vessel_trajectory_data):
        """Test that sequences are created when data is sufficient."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)
        X, y = builder.create_sequences(features_df)

        # Should create sequences from 100 points with seq_len=20
        assert len(X) > 0
        assert len(y) > 0

    def test_create_sequences_empty_when_insufficient_data(self, anomaly_config):
        """Test that no sequences are created when data is too short."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        # Create very short trajectory (less than sequence_length)
        short_data = pd.DataFrame(
            {
                "mmsi": [123456] * 10,
                "time": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "lat": [59.0] * 10,
                "lon": [10.0] * 10,
                "sog": [10.0] * 10,
                "cog": [45.0] * 10,
            }
        )

        features_df = builder.build_features(short_data)
        X, y = builder.create_sequences(features_df)

        # Should return empty arrays
        assert len(X) == 0
        assert len(y) == 0


@pytest.mark.unit
class TestAnomalyDetectionBuilderFeatures:
    """Test feature engineering for anomaly detection."""

    def test_add_behavioral_features(self, anomaly_config, vessel_trajectory_data):
        """Test that behavioral features are added correctly."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)

        # Should have behavioral features
        behavioral_cols = [
            col for col in features_df.columns if col.startswith("behavioral_")
        ]
        assert len(behavioral_cols) > 0

        # Check specific behavioral features
        assert "behavioral_speed_std" in features_df.columns
        assert "behavioral_speed_mean" in features_df.columns

    def test_add_statistical_features(self, anomaly_config, vessel_trajectory_data):
        """Test that statistical features are added correctly."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)

        # Should have statistical features
        statistical_cols = [
            col for col in features_df.columns if col.startswith("statistical_")
        ]
        assert len(statistical_cols) > 0

        # Check specific statistical features (rolling stats)
        assert "statistical_speed_rolling_mean" in features_df.columns
        assert "statistical_speed_rolling_std" in features_df.columns

    def test_add_contextual_features(self, anomaly_config, vessel_trajectory_data):
        """Test that contextual features are added correctly."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)

        # Should have contextual features
        contextual_cols = [
            col for col in features_df.columns if col.startswith("contextual_")
        ]
        assert len(contextual_cols) > 0

        # Check specific contextual features
        assert "contextual_is_night" in features_df.columns
        assert "contextual_is_weekend" in features_df.columns

    def test_features_no_nan_values(self, anomaly_config, vessel_trajectory_data):
        """Test that feature engineering doesn't introduce excessive NaN values."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        features_df = builder.build_features(vessel_trajectory_data)

        # Some NaN values are expected (e.g., first point in rolling stats)
        # but should not be excessive
        feature_cols = [
            col for col in features_df.columns if col not in ["mmsi", "time"]
        ]
        nan_ratio = features_df[feature_cols].isna().sum().sum() / (
            len(features_df) * len(feature_cols)
        )

        # Less than 5% NaN values is acceptable
        assert nan_ratio < 0.05


@pytest.mark.unit
class TestAnomalyDetectionBuilderMultipleVessels:
    """Test builder behavior with multiple vessels."""

    def test_sequences_from_multiple_vessels(self, anomaly_config):
        """Test that sequences are created correctly for multiple vessels."""
        builder = AnomalyDetectionBuilder(anomaly_config)

        # Create data for 3 vessels
        n_points = 50
        data_list = []

        for mmsi in [111111, 222222, 333333]:
            vessel_data = {
                "mmsi": [mmsi] * n_points,
                "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
                "lat": 59.0 + np.random.randn(n_points) * 0.01,
                "lon": 10.0 + np.random.randn(n_points) * 0.01,
                "sog": 10.0 + np.random.randn(n_points) * 1.0,
                "cog": 45.0 + np.random.randn(n_points) * 5.0,
            }
            data_list.append(pd.DataFrame(vessel_data))

        multi_vessel_data = pd.concat(data_list, ignore_index=True)

        features_df = builder.build_features(multi_vessel_data)
        X, y = builder.create_sequences(features_df)

        # Should create sequences for each vessel
        assert len(X) > 0
        # Each vessel with 50 points and seq_len=20 should produce ~30 sequences
        # So 3 vessels * 30 = ~90 sequences
        assert len(X) > 80  # Allow some variance

        # Targets should equal inputs - check element-wise with NaN handling
        assert X.shape == y.shape
        X_flat = X.flatten()
        y_flat = y.flatten()
        for i in range(len(X_flat)):
            x_val, y_val = X_flat[i], y_flat[i]
            if isinstance(x_val, float) and isinstance(y_val, float):
                if np.isnan(x_val) and np.isnan(y_val):
                    continue
                assert x_val == y_val
            else:
                assert x_val == y_val
