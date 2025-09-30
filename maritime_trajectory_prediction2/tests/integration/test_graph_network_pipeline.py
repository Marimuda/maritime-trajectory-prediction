"""
Integration tests for graph network pipeline.

Tests the complete pipeline from data processing through graph construction,
ensuring that edge features, graph metrics, and interaction scores work
correctly in the full context.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builders import GraphNetworkBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig


@pytest.fixture
def multi_vessel_scenario():
    """Create realistic multi-vessel scenario for graph network."""
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
    }

    # Vessel 3: Moving around nearby
    vessel3_points = 60
    vessel3 = {
        "mmsi": [333333] * vessel3_points,
        "time": pd.date_range("2024-01-01", periods=vessel3_points, freq="1min"),
        "lat": 59.01 + np.random.randn(vessel3_points) * 0.0001,
        "lon": 10.01 + np.random.randn(vessel3_points) * 0.0001,
        "sog": [8.0] * vessel3_points,
        "cog": [180.0] * vessel3_points,
        "heading": [180.0] * vessel3_points,
        "turn": [0.0] * vessel3_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)
    df3 = pd.DataFrame(vessel3)

    return pd.concat([df1, df2, df3], ignore_index=True)


@pytest.mark.integration
class TestGraphNetworkPipeline:
    """Integration tests for complete graph network pipeline."""

    def test_full_pipeline_data_to_sequences(self, multi_vessel_scenario):
        """Test complete pipeline from raw data to graph sequences."""
        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = GraphNetworkBuilder(config)

        # Build features (includes edge and graph features)
        features_df = builder.build_features(multi_vessel_scenario)

        # Verify edge features were calculated
        assert "edge_avg_distance" in features_df.columns
        assert "edge_min_distance" in features_df.columns
        assert "edge_avg_rel_velocity" in features_df.columns
        assert "edge_min_tcpa" in features_df.columns
        assert "edge_degree" in features_df.columns

        # Verify graph features were calculated
        assert "graph_density" in features_df.columns
        assert "graph_avg_degree" in features_df.columns
        assert "graph_clustering" in features_df.columns

        # Verify that at least some vessels have edges
        assert (features_df["edge_degree"] > 0).any()

        # Verify that at least some time windows have graph metrics
        assert (features_df["graph_density"] > 0).any()

        # Create sequences
        X, y = builder.create_sequences(features_df)

        # Verify sequences were created
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]

    def test_edge_features_realistic_values(self, multi_vessel_scenario):
        """Test that edge features have realistic values."""
        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = GraphNetworkBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)

        # Filter to vessels with edges
        has_edges = features_df["edge_degree"] > 0

        if not has_edges.any():
            pytest.skip("No vessels with edges found")

        edge_df = features_df[has_edges]

        # Distances should be non-negative and finite
        assert (edge_df["edge_min_distance"] >= 0).all()
        assert (edge_df["edge_avg_distance"] >= 0).all()
        assert (edge_df["edge_min_distance"] < float("inf")).all()

        # Relative velocity should be non-negative
        assert (edge_df["edge_avg_rel_velocity"] >= 0).all()

        # Degree should be positive
        assert (edge_df["edge_degree"] > 0).all()

    def test_graph_metrics_realistic_values(self, multi_vessel_scenario):
        """Test that graph metrics have realistic values."""
        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = GraphNetworkBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)

        # Filter to time windows with graph metrics
        has_graph = features_df["graph_density"] > 0

        if not has_graph.any():
            pytest.skip("No graph metrics found")

        graph_df = features_df[has_graph]

        # Density should be in [0, 1]
        assert (graph_df["graph_density"] >= 0).all()
        assert (graph_df["graph_density"] <= 1).all()

        # Average degree should be non-negative
        assert (graph_df["graph_avg_degree"] >= 0).all()

        # Clustering should be in [0, 1]
        assert (graph_df["graph_clustering"] >= 0).all()
        assert (graph_df["graph_clustering"] <= 1).all()

    def test_interaction_scores_calculation(self, multi_vessel_scenario):
        """Test that interaction scores are calculated correctly."""
        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = GraphNetworkBuilder(config)

        # Build features and targets
        features_df = builder.build_features(multi_vessel_scenario)
        targets_df = builder.build_targets(features_df)

        # Verify target columns exist
        assert "interaction_score" in targets_df.columns

        # Verify scores are in valid range [0, 1]
        assert (targets_df["interaction_score"] >= 0).all()
        assert (targets_df["interaction_score"] <= 1).all()

        # At least some vessels should have non-zero interaction
        assert (targets_df["interaction_score"] > 0).any()

    def test_vessels_form_network(self, multi_vessel_scenario):
        """Test that nearby vessels form a connected network."""
        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = GraphNetworkBuilder(config)
        features_df = builder.build_features(multi_vessel_scenario)

        # Separate by vessel
        vessel1_df = features_df[features_df["mmsi"] == 111111]
        vessel2_df = features_df[features_df["mmsi"] == 222222]
        vessel3_df = features_df[features_df["mmsi"] == 333333]

        # All vessels should have edges at some point (they're all close together)
        assert (vessel1_df["edge_degree"] > 0).any()
        assert (vessel2_df["edge_degree"] > 0).any()
        assert (vessel3_df["edge_degree"] > 0).any()

    def test_performance_with_many_vessels(self):
        """Test that graph feature calculation completes in reasonable time."""
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
            }
            data_list.append(pd.DataFrame(vessel_data))

        df = pd.concat(data_list, ignore_index=True)

        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=20,
            prediction_horizon=5,
        )

        builder = GraphNetworkBuilder(config)

        start_time = time.time()
        features_df = builder.build_features(df)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 30 seconds for 5 vessels)
        assert elapsed_time < 30

        # Should have calculated features
        assert "edge_degree" in features_df.columns
        assert "graph_density" in features_df.columns
        assert "graph_clustering" in features_df.columns
        assert len(features_df) == len(df)

    def test_cpa_calculator_integration(self):
        """Test that CPA/TCPA calculator from maritime module is used correctly."""
        from src.maritime.cpa_tcpa import CPACalculator

        # Create simple two-vessel scenario with overlapping timestamps
        times = pd.date_range("2024-01-01", periods=30, freq="1min")

        # Vessel 1: Moving east
        vessel1 = {
            "mmsi": [111] * 30,
            "time": times,
            "lat": [59.0] * 30,
            "lon": [10.0 + i * 0.001 for i in range(30)],
            "sog": [10.0] * 30,
            "cog": [90.0] * 30,
        }

        # Vessel 2: Stationary to the east, will be approached
        vessel2 = {
            "mmsi": [222] * 30,
            "time": times,
            "lat": [59.0] * 30,
            "lon": [10.01] * 30,
            "sog": [10.0] * 30,
            "cog": [270.0] * 30,
        }

        df1 = pd.DataFrame(vessel1)
        df2 = pd.DataFrame(vessel2)
        df = pd.concat([df1, df2], ignore_index=True)

        config = DatasetConfig(
            task=MLTask.GRAPH_NEURAL_NETWORKS,
            sequence_length=2,
            prediction_horizon=1,
        )

        builder = GraphNetworkBuilder(config)
        features_df = builder.build_features(df)

        # The CPACalculator should have been used to calculate edge features
        # Verify that calculation happened (vessels should detect each other)
        assert (features_df["edge_degree"] > 0).any()
        assert (features_df["edge_min_tcpa"] < float("inf")).any()