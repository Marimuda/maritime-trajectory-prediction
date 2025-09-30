"""
Unit tests for GraphNetworkBuilder.

Tests the behavior of graph network dataset builder to ensure it correctly
calculates edge features, graph metrics, and interaction scores.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builders import GraphNetworkBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig


@pytest.fixture
def graph_config():
    """Create config for graph network builder."""
    return DatasetConfig(
        task=MLTask.GRAPH_NEURAL_NETWORKS,
        sequence_length=20,
        prediction_horizon=5,
        validation_split=0.2,
        test_split=0.1,
        random_seed=42,
    )


@pytest.fixture
def single_vessel_data():
    """Create data for a single vessel (no edges/graph)."""
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
    """Create data for two vessels close together."""
    np.random.seed(42)
    n_points = 30

    # Vessel 1: Moving east at (59.0, 10.0)
    vessel1 = {
        "mmsi": [111111] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [59.0] * n_points,
        "lon": 10.0 + np.arange(n_points) * 0.001,  # Moving east
        "sog": [10.0] * n_points,
        "cog": [90.0] * n_points,
        "heading": [90.0] * n_points,
        "turn": [0.0] * n_points,
    }

    # Vessel 2: Moving north, nearby
    vessel2 = {
        "mmsi": [222222] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 58.99 + np.arange(n_points) * 0.0005,  # Moving north
        "lon": [10.02] * n_points,
        "sog": [10.0] * n_points,
        "cog": [0.0] * n_points,
        "heading": [0.0] * n_points,
        "turn": [0.0] * n_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def three_vessels_network():
    """Create data for three vessels forming a network."""
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
    }

    # Vessel 2 at (59.01, 10.01) - close to vessel 1
    vessel2 = {
        "mmsi": [222222] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [59.01] * n_points,
        "lon": [10.01] * n_points,
        "sog": [12.0] * n_points,
        "cog": [180.0] * n_points,
    }

    # Vessel 3 at (59.005, 10.005) - close to both 1 and 2
    vessel3 = {
        "mmsi": [333333] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": [59.005] * n_points,
        "lon": [10.005] * n_points,
        "sog": [8.0] * n_points,
        "cog": [270.0] * n_points,
    }

    df1 = pd.DataFrame(vessel1)
    df2 = pd.DataFrame(vessel2)
    df3 = pd.DataFrame(vessel3)

    return pd.concat([df1, df2, df3], ignore_index=True)


@pytest.mark.unit
class TestEdgeFeatures:
    """Test edge feature calculation."""

    def test_single_vessel_no_edges(self, graph_config, single_vessel_data):
        """Test that single vessel has no edge features (all infinity/zero)."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(single_vessel_data)

        # Should have edge feature columns
        assert "edge_avg_distance" in features_df.columns
        assert "edge_min_distance" in features_df.columns
        assert "edge_avg_rel_velocity" in features_df.columns
        assert "edge_min_tcpa" in features_df.columns
        assert "edge_degree" in features_df.columns

        # All values should be infinity/zero (no nearby vessels)
        assert all(features_df["edge_avg_distance"] == float("inf"))
        assert all(features_df["edge_min_distance"] == float("inf"))
        assert all(features_df["edge_avg_rel_velocity"] == 0.0)
        assert all(features_df["edge_min_tcpa"] == float("inf"))
        assert all(features_df["edge_degree"] == 0)

    def test_two_vessels_close_have_edges(self, graph_config, two_vessels_close):
        """Test that nearby vessels have edge features calculated."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(two_vessels_close)

        # At least some records should have edges
        assert (features_df["edge_degree"] > 0).any()

        # Edge distances should be finite for vessels with edges
        has_edges = features_df["edge_degree"] > 0
        assert (features_df.loc[has_edges, "edge_min_distance"] < float("inf")).all()
        assert (features_df.loc[has_edges, "edge_avg_distance"] < float("inf")).all()

    def test_edge_features_realistic_values(self, graph_config, two_vessels_close):
        """Test that edge features have realistic values."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(two_vessels_close)

        # Filter to vessels with edges
        has_edges = features_df["edge_degree"] > 0
        if not has_edges.any():
            pytest.skip("No vessels with edges found")

        edge_df = features_df[has_edges]

        # Distances should be non-negative
        assert (edge_df["edge_min_distance"] >= 0).all()
        assert (edge_df["edge_avg_distance"] >= 0).all()

        # Relative velocity should be non-negative
        assert (edge_df["edge_avg_rel_velocity"] >= 0).all()

        # TCPA should be positive or infinity
        assert (
            (edge_df["edge_min_tcpa"] > 0) | (edge_df["edge_min_tcpa"] == float("inf"))
        ).all()

        # Degree should be positive integer
        assert (edge_df["edge_degree"] > 0).all()
        assert edge_df["edge_degree"].dtype in [int, np.int64, np.int32]


@pytest.mark.unit
class TestGraphFeatures:
    """Test graph-level feature calculation."""

    def test_single_vessel_no_graph_metrics(self, graph_config, single_vessel_data):
        """Test that single vessel has zero graph metrics (need 3+ vessels)."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(single_vessel_data)

        # Should have graph feature columns
        assert "graph_density" in features_df.columns
        assert "graph_avg_degree" in features_df.columns
        assert "graph_clustering" in features_df.columns

        # All values should be zero (no graph)
        assert all(features_df["graph_density"] == 0.0)
        assert all(features_df["graph_avg_degree"] == 0.0)
        assert all(features_df["graph_clustering"] == 0.0)

    def test_three_vessels_have_graph_metrics(self, graph_config, three_vessels_network):
        """Test that three vessels get graph metrics calculated."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(three_vessels_network)

        # At least some records should have graph metrics
        assert (features_df["graph_density"] > 0).any()

    def test_graph_metrics_valid_ranges(self, graph_config, three_vessels_network):
        """Test that graph metrics are in valid ranges."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(three_vessels_network)

        # Density should be in [0, 1]
        assert (features_df["graph_density"] >= 0).all()
        assert (features_df["graph_density"] <= 1).all()

        # Average degree should be non-negative
        assert (features_df["graph_avg_degree"] >= 0).all()

        # Clustering should be in [0, 1]
        assert (features_df["graph_clustering"] >= 0).all()
        assert (features_df["graph_clustering"] <= 1).all()


@pytest.mark.unit
class TestInteractionScores:
    """Test vessel interaction score calculation."""

    def test_single_vessel_zero_interaction(self, graph_config, single_vessel_data):
        """Test that single vessel has zero interaction score."""
        builder = GraphNetworkBuilder(graph_config)

        targets_df = builder.build_targets(single_vessel_data)

        # Should have interaction_score column
        assert "interaction_score" in targets_df.columns

        # All values should be zero (no other vessels)
        assert all(targets_df["interaction_score"] == 0.0)

    def test_two_vessels_close_high_interaction(self, graph_config, two_vessels_close):
        """Test that close vessels have high interaction scores."""
        builder = GraphNetworkBuilder(graph_config)

        targets_df = builder.build_targets(two_vessels_close)

        # At least some vessels should have non-zero interaction
        assert (targets_df["interaction_score"] > 0).any()

    def test_interaction_scores_valid_range(self, graph_config, two_vessels_close):
        """Test that interaction scores are in [0, 1] range."""
        builder = GraphNetworkBuilder(graph_config)

        targets_df = builder.build_targets(two_vessels_close)

        # Scores should be in [0, 1]
        assert (targets_df["interaction_score"] >= 0).all()
        assert (targets_df["interaction_score"] <= 1).all()


@pytest.mark.unit
class TestGraphSequences:
    """Test graph sequence generation."""

    def test_create_sequences_with_sufficient_data(
        self, graph_config, three_vessels_network
    ):
        """Test that graph sequences are created when data is sufficient."""
        builder = GraphNetworkBuilder(graph_config)

        features_df = builder.build_features(three_vessels_network)
        X, y = builder.create_sequences(features_df)

        # Should create some sequences
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]

    def test_sequences_empty_with_insufficient_data(self, graph_config):
        """Test that no sequences are created with insufficient data."""
        builder = GraphNetworkBuilder(graph_config)

        # Create single data point (below min 2 nodes requirement)
        short_data = pd.DataFrame(
            {
                "mmsi": [123456],
                "time": pd.date_range("2024-01-01", periods=1, freq="1min"),
                "lat": [59.0],
                "lon": [10.0],
                "sog": [10.0],
                "cog": [90.0],
            }
        )

        features_df = builder.build_features(short_data)
        X, y = builder.create_sequences(features_df)

        # Should return empty arrays (need min 2 nodes per time window)
        assert len(X) == 0
        assert len(y) == 0