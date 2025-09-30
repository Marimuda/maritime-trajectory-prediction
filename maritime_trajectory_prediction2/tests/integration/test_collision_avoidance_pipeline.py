"""
Integration tests for collision avoidance pipeline.

Tests the complete pipeline from data processing through feature calculation,
ensuring that real CPA/TCPA calculations work correctly in the full context.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builders import CollisionAvoidanceBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig


@pytest.fixture
def multi_vessel_scenario():
    """Create realistic multi-vessel scenario with potential collisions."""
    np.random.seed(42)

    # Vessel 1: Moving east
    vessel1_points = 60
    vessel1 = {
        "mmsi": [111111] * vessel1_points,
        "time": pd.date_range("2024-01-01", periods=vessel1_points, freq="1min"),
        "lat": [59.0] * vessel1_points,
        "lon": 10.0 + np.arange(vessel1_points) * 0.001,
        "sog": [10.0] * vessel1_points,
        "cog": [90.0] * vessel1_points,
        "heading": [90.0] * vessel1_points,
        "turn": [0.0] * vessel1_points,
        "accuracy": [1] * vessel1_points,
        "second": [0] * vessel1_points,
        "maneuver": [0] * vessel1_points,
        "raim": [False] * vessel1_points,
    }

    # Vessel 2: Moving north, crossing path of vessel 1
    vessel2_points = 60
    vessel2 = {
        "mmsi": [222222] * vessel2_points,
        "time": pd.date_range("2024-01-01", periods=vessel2_points, freq="1min"),
        "lat": 58.99 + np.arange(vessel2_points) * 0.0005,
        "lon": [10.03] * vessel2_points,
        "sog": [12.0] * vessel2_points,
        "cog": [0.0] * vessel2_points,
        "heading": [0.0] * vessel2_points,
        "turn": [0.0] * vessel2_points,
        "accuracy": [1] * vessel2_points,
        "second": [0] * vessel2_points,
        "maneuver": [0] * vessel2_points,
        "raim": [False] * vessel2_points,
    }

    # Vessel 3: Far away, no collision risk
    vessel3_points = 60
    vessel3 = {
        "mmsi": [333333] * vessel3_points,
        "time": pd.date_range("2024-01-01", periods=vessel3_points, freq="1min"),
        "lat": [60.0] * vessel3_points,
        "lon": [11.0] * vessel3_points,
        "sog": [8.0] * vessel3_points,
        "cog": [180.0] * vessel3_points,
        "heading": [180.0] * vessel3_points,
        "turn": [0.0] * vessel3_points,
        "accuracy": [1] * vessel3_points,
        "second": [0] * vessel3_points,
        "maneuver": [0] * vessel3_points,
        "raim": [False] * vessel3_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)
    df3 = pd.DataFrame(vessel3)

    return pd.concat([df1, df2, df3], ignore_index=True)


@pytest.mark.integration
class TestCollisionAvoidancePipeline:
    """Integration tests for complete collision avoidance pipeline."""

    def test_full_pipeline_data_to_sequences(self, multi_vessel_scenario):
        """Test complete pipeline from raw data to sequences."""
        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = CollisionAvoidanceBuilder(config)

        # Build features (includes collision feature calculation)
        features_df = builder.build_features(multi_vessel_scenario)

        # Verify collision features were calculated
        assert "collision_min_cpa" in features_df.columns
        assert "collision_min_tcpa" in features_df.columns
        assert "collision_num_nearby" in features_df.columns
        assert "collision_relative_speed" in features_df.columns
        assert "collision_relative_bearing" in features_df.columns

        # Verify that at least some vessels detected nearby vessels
        # (Vessels 1 and 2 should detect each other at some point)
        assert (features_df["collision_num_nearby"] > 0).any()

        # Create sequences
        X, y = builder.create_sequences(features_df)

        # Verify sequences were created
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]

        # Verify sequence shapes
        assert X.shape[1] == config.sequence_length
        assert y.shape[1] == 2  # [collision_risk, time_to_collision]

    def test_collision_features_realistic_values(self, multi_vessel_scenario):
        """Test that collision features have realistic values."""
        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = CollisionAvoidanceBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)

        # CPA should be non-negative
        assert (features_df["collision_min_cpa"] >= 0).all()

        # TCPA should be either positive or infinity
        assert (
            (features_df["collision_min_tcpa"] > 0)
            | (features_df["collision_min_tcpa"] == float("inf"))
        ).all()

        # Number of nearby vessels should be non-negative integer
        assert (features_df["collision_num_nearby"] >= 0).all()
        assert features_df["collision_num_nearby"].dtype in [int, np.int64, np.int32]

        # Relative speed should be non-negative
        assert (features_df["collision_relative_speed"] >= 0).all()

        # Relative bearing should be in [0, 360)
        nearby_mask = features_df["collision_num_nearby"] > 0
        if nearby_mask.any():
            bearings = features_df.loc[nearby_mask, "collision_relative_bearing"]
            assert (bearings >= 0).all()
            assert (bearings < 360).all()

    def test_collision_risk_targets(self, multi_vessel_scenario):
        """Test that collision risk targets are calculated correctly."""
        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = CollisionAvoidanceBuilder(config)

        # Build features and targets
        features_df = builder.build_features(multi_vessel_scenario)
        targets_df = builder.build_targets(features_df)

        # Verify target columns exist
        assert "collision_risk" in targets_df.columns
        assert "time_to_collision" in targets_df.columns

        # Verify risk scores are in valid range [0, 1]
        assert (targets_df["collision_risk"] >= 0).all()
        assert (targets_df["collision_risk"] <= 1).all()

        # Verify time to collision is positive or infinity
        assert (
            (targets_df["time_to_collision"] > 0)
            | (targets_df["time_to_collision"] == float("inf"))
        ).all()

    def test_vessels_detect_each_other(self, multi_vessel_scenario):
        """Test that nearby vessels detect each other."""
        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = CollisionAvoidanceBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)

        # Separate by vessel
        vessel1_df = features_df[features_df["mmsi"] == 111111]
        vessel2_df = features_df[features_df["mmsi"] == 222222]
        vessel3_df = features_df[features_df["mmsi"] == 333333]

        # Vessels 1 and 2 should detect each other at some point
        # (they are crossing paths)
        assert (vessel1_df["collision_num_nearby"] > 0).any()
        assert (vessel2_df["collision_num_nearby"] > 0).any()

        # Vessel 3 is far away, should not detect others
        assert (vessel3_df["collision_num_nearby"] == 0).all()

    def test_collision_risk_levels(self, multi_vessel_scenario):
        """Test that different risk levels are assigned appropriately."""
        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = CollisionAvoidanceBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)
        targets_df = builder.build_targets(features_df)

        # Should have a mix of risk levels
        unique_risks = targets_df["collision_risk"].unique()

        # At minimum, should have 0 risk (vessel 3 is far away)
        assert 0.0 in unique_risks

        # May or may not have other risk levels depending on vessel geometry
        # Just verify that risk calculation is working (not all same value)
        assert len(unique_risks) > 1

    def test_cpa_tcpa_calculator_integration(self):
        """Test that CPA/TCPA calculator from maritime module is used correctly."""
        from src.maritime.cpa_tcpa import CPACalculator, VesselState

        # Create simple two-vessel scenario with overlapping timestamps
        # Vessel 111 and 222 both have positions at time t0 and t1
        times = pd.date_range("2024-01-01", periods=2, freq="1min")

        data = {
            "mmsi": [111, 222, 111, 222],
            "time": [times[0], times[0], times[1], times[1]],  # Both vessels at each time
            "lat": [59.0, 59.0, 59.0, 59.001],
            "lon": [10.0, 10.01, 10.001, 10.01],
            "sog": [10.0, 10.0, 10.0, 10.0],
            "cog": [90.0, 270.0, 90.0, 270.0],  # Approaching head-on
            "heading": [90.0, 270.0, 90.0, 270.0],
            "turn": [0.0, 0.0, 0.0, 0.0],
        }

        df = pd.DataFrame(data)

        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=2,
            prediction_horizon=1,
        )

        builder = CollisionAvoidanceBuilder(config)
        features_df = builder.build_features(df)

        # The CPACalculator should have been used to calculate CPA/TCPA
        # Verify that calculation happened (vessels should detect each other)
        assert (features_df["collision_num_nearby"] > 0).any()

    def test_performance_with_many_vessels(self):
        """Test that collision feature calculation completes in reasonable time."""
        import time

        # Create scenario with 5 vessels
        n_vessels = 5
        n_points = 30
        data_list = []

        for i in range(n_vessels):
            vessel_data = {
                "mmsi": [100000 + i] * n_points,
                "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
                "lat": 59.0 + i * 0.01 + np.random.randn(n_points) * 0.0001,
                "lon": 10.0 + i * 0.01 + np.random.randn(n_points) * 0.0001,
                "sog": [10.0] * n_points,
                "cog": [90.0] * n_points,
                "heading": [90.0] * n_points,
                "turn": [0.0] * n_points,
            }
            data_list.append(pd.DataFrame(vessel_data))

        df = pd.concat(data_list, ignore_index=True)

        config = DatasetConfig(
            task=MLTask.COLLISION_AVOIDANCE,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = CollisionAvoidanceBuilder(config)

        start_time = time.time()
        features_df = builder.build_features(df)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds for 5 vessels)
        assert elapsed_time < 30

        # Should have calculated features
        assert "collision_min_cpa" in features_df.columns
        assert len(features_df) == len(df)