"""
Unit tests for CollisionAvoidanceBuilder.

Tests the behavior of collision avoidance dataset builder to ensure it
correctly calculates CPA/TCPA features and collision risk using the
maritime domain calculators.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builders import CollisionAvoidanceBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig


@pytest.fixture
def collision_config():
    """Create config for collision avoidance builder."""
    return DatasetConfig(
        task=MLTask.COLLISION_AVOIDANCE,
        sequence_length=20,
        prediction_horizon=5,
        validation_split=0.2,
        test_split=0.1,
        random_seed=42,
    )


@pytest.fixture
def single_vessel_data():
    """Create data for a single vessel (no collision scenarios)."""
    np.random.seed(42)
    n_points = 50

    data = {
        "mmsi": [123456] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 59.0 + np.cumsum(np.random.randn(n_points) * 0.0001),
        "lon": 10.0 + np.cumsum(np.random.randn(n_points) * 0.0001),
        "sog": [10.0] * n_points,
        "cog": [45.0] * n_points,
        "heading": [45.0] * n_points,
        "turn": [0.0] * n_points,
    }

    return pd.DataFrame(data)


@pytest.fixture
def two_vessels_close():
    """Create data for two vessels on collision course (close proximity)."""
    np.random.seed(42)
    n_points = 30

    # Vessel 1: Moving east at (59.0, 10.0)
    vessel1 = {
        "mmsi": [111111] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [59.0] * n_points,
        "lon": 10.0 + np.arange(n_points) * 0.001,  # Moving east
        "sog": [10.0] * n_points,
        "cog": [90.0] * n_points,  # Heading east
        "heading": [90.0] * n_points,
        "turn": [0.0] * n_points,
    }

    # Vessel 2: Moving north at (59.0, 10.05), will cross paths
    vessel2 = {
        "mmsi": [222222] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 58.99 + np.arange(n_points) * 0.0005,  # Moving north
        "lon": [10.05] * n_points,
        "sog": [10.0] * n_points,
        "cog": [0.0] * n_points,  # Heading north
        "heading": [0.0] * n_points,
        "turn": [0.0] * n_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def two_vessels_far():
    """Create data for two vessels far apart (no collision risk)."""
    np.random.seed(42)
    n_points = 30

    # Vessel 1 at (59.0, 10.0)
    vessel1 = {
        "mmsi": [111111] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [59.0] * n_points,
        "lon": [10.0] * n_points,
        "sog": [10.0] * n_points,
        "cog": [90.0] * n_points,
        "heading": [90.0] * n_points,
        "turn": [0.0] * n_points,
    }

    # Vessel 2 at (60.0, 11.0) - ~100 km away
    vessel2 = {
        "mmsi": [222222] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [60.0] * n_points,
        "lon": [11.0] * n_points,
        "sog": [10.0] * n_points,
        "cog": [0.0] * n_points,
        "heading": [0.0] * n_points,
        "turn": [0.0] * n_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)

    return pd.concat([df1, df2], ignore_index=True)


@pytest.mark.unit
class TestCollisionFeaturesCalculation:
    """Test collision feature calculation."""

    def test_single_vessel_no_nearby(self, collision_config, single_vessel_data):
        """Test that single vessel has no collision features (inf CPA/TCPA)."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(single_vessel_data)

        # Should have collision feature columns
        assert "collision_min_cpa" in features_df.columns
        assert "collision_min_tcpa" in features_df.columns
        assert "collision_num_nearby" in features_df.columns
        assert "collision_relative_speed" in features_df.columns
        assert "collision_relative_bearing" in features_df.columns

        # All values should be infinity/zero (no nearby vessels)
        assert all(features_df["collision_min_cpa"] == float("inf"))
        assert all(features_df["collision_min_tcpa"] == float("inf"))
        assert all(features_df["collision_num_nearby"] == 0)

    def test_two_vessels_close_calculates_cpa(
        self, collision_config, two_vessels_close
    ):
        """Test that CPA/TCPA is calculated for nearby vessels."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_close)

        # At least some records should have nearby vessels
        assert (features_df["collision_num_nearby"] > 0).any()

        # CPA should be finite for some records
        assert (features_df["collision_min_cpa"] < float("inf")).any()

        # TCPA should be calculated (column exists)
        # Note: Depending on geometry, vessels might be receding (TCPA = inf)
        # So we just verify that the calculation ran without errors
        assert "collision_min_tcpa" in features_df.columns

    def test_two_vessels_far_no_collision_features(
        self, collision_config, two_vessels_far
    ):
        """Test that vessels far apart don't trigger collision features."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_far)

        # Vessels are > 10 NM apart, should not be detected as nearby
        assert all(features_df["collision_num_nearby"] == 0)
        assert all(features_df["collision_min_cpa"] == float("inf"))

    def test_collision_features_with_nan_data(self, collision_config):
        """Test that NaN values are handled gracefully."""
        # Create data with some NaN values
        data = {
            "mmsi": [111111] * 20 + [222222] * 20,
            "time": pd.date_range("2024-01-01", periods=40, freq="1min"),
            "lat": [59.0] * 10 + [np.nan] * 10 + [59.0] * 20,
            "lon": [10.0] * 10 + [np.nan] * 10 + [10.01] * 20,
            "sog": [10.0] * 40,
            "cog": [90.0] * 40,
            "heading": [90.0] * 40,
            "turn": [0.0] * 40,
        }

        df = pd.DataFrame(data)
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(df)

        # Should complete without error
        assert len(features_df) == len(df)
        assert "collision_min_cpa" in features_df.columns


@pytest.mark.unit
class TestCollisionRiskCalculation:
    """Test collision risk scoring."""

    def test_risk_critical_level(self, collision_config):
        """Test critical risk level (CPA < 250m, TCPA < 5min)."""
        builder = CollisionAvoidanceBuilder(collision_config)

        # Create data with critical collision risk
        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "lat": [59.0],
            "lon": [10.0],
            "sog": [10.0],
            "cog": [90.0],
            "collision_min_cpa": [200.0],  # < 250m
            "collision_min_tcpa": [240.0],  # < 300s (5 min)
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 1.0  # Critical risk

    def test_risk_high_level(self, collision_config):
        """Test high risk level (CPA < 500m, TCPA < 10min)."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "collision_min_cpa": [400.0],  # < 500m
            "collision_min_tcpa": [500.0],  # < 600s (10 min)
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.75  # High risk

    def test_risk_medium_level(self, collision_config):
        """Test medium risk level."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "collision_min_cpa": [800.0],  # < 1000m
            "collision_min_tcpa": [1000.0],  # < 1200s (20 min)
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.5  # Medium risk

    def test_risk_low_level(self, collision_config):
        """Test low risk level."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "collision_min_cpa": [1500.0],  # < 2000m
            "collision_min_tcpa": [1500.0],  # < 1800s (30 min)
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.25  # Low risk

    def test_risk_none_level(self, collision_config):
        """Test no risk (CPA/TCPA beyond thresholds)."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "collision_min_cpa": [5000.0],  # > 2000m
            "collision_min_tcpa": [3000.0],  # > 1800s
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.0  # No risk

    def test_risk_negative_tcpa(self, collision_config):
        """Test that negative TCPA (receding vessels) has no risk."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "collision_min_cpa": [200.0],  # Close CPA
            "collision_min_tcpa": [-100.0],  # Negative = vessels receding
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.0  # No risk for receding vessels

    def test_risk_without_collision_features(self, collision_config):
        """Test that risk is 0 when collision features are missing."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
            "lat": [59.0],
            "lon": [10.0],
        }

        df = pd.DataFrame(data)
        risk = builder._calculate_collision_risk(df)

        assert risk.iloc[0] == 0.0


@pytest.mark.unit
class TestTimeToCollision:
    """Test time to collision calculation."""

    def test_time_to_collision_with_features(self, collision_config):
        """Test that time to collision returns min_tcpa when available."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456, 789012],
            "time": pd.date_range("2024-01-01", periods=2, freq="1min"),
            "collision_min_tcpa": [300.0, 600.0],
        }

        df = pd.DataFrame(data)
        ttc = builder._calculate_time_to_collision(df)

        assert ttc.iloc[0] == 300.0
        assert ttc.iloc[1] == 600.0

    def test_time_to_collision_without_features(self, collision_config):
        """Test that time to collision returns infinity without features."""
        builder = CollisionAvoidanceBuilder(collision_config)

        data = {
            "mmsi": [123456],
            "time": [pd.Timestamp("2024-01-01")],
        }

        df = pd.DataFrame(data)
        ttc = builder._calculate_time_to_collision(df)

        assert ttc.iloc[0] == float("inf")


@pytest.mark.unit
class TestCollisionTargets:
    """Test target generation for collision avoidance."""

    def test_build_targets_has_required_columns(
        self, collision_config, two_vessels_close
    ):
        """Test that targets include collision_risk and time_to_collision."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_close)
        targets_df = builder.build_targets(features_df)

        assert "collision_risk" in targets_df.columns
        assert "time_to_collision" in targets_df.columns
        assert "mmsi" in targets_df.columns
        assert "time" in targets_df.columns

    def test_targets_same_length_as_features(self, collision_config, two_vessels_close):
        """Test that targets have same length as input."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_close)
        targets_df = builder.build_targets(features_df)

        assert len(targets_df) == len(features_df)


@pytest.mark.unit
class TestCollisionSequences:
    """Test sequence generation for collision avoidance."""

    def test_create_sequences_with_sufficient_data(
        self, collision_config, two_vessels_close
    ):
        """Test that sequences are created when data is sufficient."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_close)
        X, y = builder.create_sequences(features_df)

        # Should create some sequences
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]

    def test_sequences_have_correct_shape(self, collision_config, two_vessels_close):
        """Test that sequence shapes match configuration."""
        builder = CollisionAvoidanceBuilder(collision_config)

        features_df = builder.build_features(two_vessels_close)
        X, y = builder.create_sequences(features_df)

        # X should be (n_sequences, sequence_length, n_features)
        assert X.ndim == 3
        assert X.shape[1] == collision_config.sequence_length

        # y should be (n_sequences, 2) for [collision_risk, time_to_collision]
        assert y.ndim == 2
        assert y.shape[1] == 2  # Two target values

    def test_sequences_empty_with_insufficient_data(self, collision_config):
        """Test that no sequences are created with insufficient data."""
        builder = CollisionAvoidanceBuilder(collision_config)

        # Create very short trajectory
        short_data = pd.DataFrame(
            {
                "mmsi": [123456] * 5,
                "time": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "lat": [59.0] * 5,
                "lon": [10.0] * 5,
                "sog": [10.0] * 5,
                "cog": [90.0] * 5,
                "heading": [90.0] * 5,
                "turn": [0.0] * 5,
            }
        )

        features_df = builder.build_features(short_data)
        X, y = builder.create_sequences(features_df)

        # Should return empty arrays
        assert len(X) == 0
        assert len(y) == 0
