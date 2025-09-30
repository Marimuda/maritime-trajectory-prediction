"""
Unit tests for parallel sequence generation in AISDataModule.

Tests that the module-level _process_vessel_sequences function can be
pickled and used with multiprocessing.Pool correctly.
"""

import numpy as np
import pandas as pd
import pytest
from multiprocessing import Pool

from src.data.ais_datamodule import _process_vessel_sequences


@pytest.fixture
def vessel_trajectory_data():
    """Create sample vessel trajectory data."""
    np.random.seed(42)
    n_points = 100

    data = {
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 59.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "lon": 10.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "sog": 10.0 + np.random.randn(n_points) * 1.0,
        "cog": 45.0 + np.random.randn(n_points) * 5.0,
    }

    return pd.DataFrame(data)


@pytest.mark.unit
class TestProcessVesselSequences:
    """Test the module-level _process_vessel_sequences function."""

    def test_function_is_picklable(self):
        """Test that the function can be pickled (required for multiprocessing)."""
        import pickle

        # This should not raise an error
        pickled = pickle.dumps(_process_vessel_sequences)
        unpickled = pickle.loads(pickled)  # nosec B301 - testing pickling, not deserializing untrusted data

        # Verify it's still callable
        assert callable(unpickled)

    def test_process_single_vessel(self, vessel_trajectory_data):
        """Test processing a single vessel."""
        mmsi = 123456
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        args = (mmsi, vessel_trajectory_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # Should create sequences
        assert len(sequences) > 0

        # Each sequence should have required keys
        for seq in sequences:
            assert "input_sequence" in seq
            assert "target_sequence" in seq
            assert "mmsi" in seq
            assert "segment_id" in seq

            # Verify shapes
            assert len(seq["input_sequence"]) == seq_len
            assert len(seq["target_sequence"]) == pred_horizon
            assert seq["mmsi"] == mmsi

    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty list."""
        # Create very short trajectory
        short_data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "lat": [59.0] * 10,
            "lon": [10.0] * 10,
            "sog": [10.0] * 10,
            "cog": [90.0] * 10,
        })

        mmsi = 123456
        seq_len = 50  # Longer than available data
        pred_horizon = 10
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        args = (mmsi, short_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # Should return empty list
        assert len(sequences) == 0

    def test_correct_sequence_dimensions(self, vessel_trajectory_data):
        """Test that sequences have correct dimensions."""
        mmsi = 123456
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon"]  # Fewer target features

        args = (mmsi, vessel_trajectory_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # Check first sequence
        seq = sequences[0]

        # Input has all features
        assert seq["input_sequence"].shape == (seq_len, len(feature_cols))

        # Target has only target features
        assert seq["target_sequence"].shape == (pred_horizon, len(target_features))

    def test_multiprocessing_with_pool(self, vessel_trajectory_data):
        """Test that function works with multiprocessing.Pool."""
        # Create multiple vessels
        vessel1 = vessel_trajectory_data.copy()
        vessel2 = vessel_trajectory_data.copy()

        mmsi1 = 111111
        mmsi2 = 222222
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        # Prepare arguments
        vessel_args = [
            (mmsi1, vessel1, seq_len, pred_horizon, feature_cols, target_features),
            (mmsi2, vessel2, seq_len, pred_horizon, feature_cols, target_features),
        ]

        # Process with Pool (this is what was failing before)
        with Pool(processes=2) as pool:
            results = pool.map(_process_vessel_sequences, vessel_args)

        # Verify results
        assert len(results) == 2
        assert len(results[0]) > 0  # Vessel 1 sequences
        assert len(results[1]) > 0  # Vessel 2 sequences

        # Verify MMSIs are correct
        assert all(seq["mmsi"] == mmsi1 for seq in results[0])
        assert all(seq["mmsi"] == mmsi2 for seq in results[1])

    def test_segment_ids_are_placeholders(self, vessel_trajectory_data):
        """Test that segment_id is -1 (placeholder for global assignment)."""
        mmsi = 123456
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        args = (mmsi, vessel_trajectory_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # All segment_ids should be -1 (placeholder)
        assert all(seq["segment_id"] == -1 for seq in sequences)

    def test_sequences_are_sorted_by_time(self, vessel_trajectory_data):
        """Test that sequences are created from time-sorted data."""
        # Shuffle data
        shuffled_data = vessel_trajectory_data.sample(frac=1, random_state=42).reset_index(drop=True)

        mmsi = 123456
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        args = (mmsi, shuffled_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # Should still create valid sequences (function sorts internally)
        assert len(sequences) > 0

    def test_sliding_window_creates_overlapping_sequences(self, vessel_trajectory_data):
        """Test that sliding window creates overlapping sequences."""
        mmsi = 123456
        seq_len = 20
        pred_horizon = 5
        feature_cols = ["lat", "lon", "sog", "cog"]
        target_features = ["lat", "lon", "sog", "cog"]

        args = (mmsi, vessel_trajectory_data, seq_len, pred_horizon, feature_cols, target_features)

        sequences = _process_vessel_sequences(args)

        # Expected number of sequences
        expected_count = len(vessel_trajectory_data) - seq_len - pred_horizon + 1
        assert len(sequences) == expected_count

        # Sequences should be overlapping (second sequence starts 1 step after first)
        if len(sequences) >= 2:
            seq1_first_lat = sequences[0]["input_sequence"]["lat"].iloc[0]
            seq2_first_lat = sequences[1]["input_sequence"]["lat"].iloc[0]

            # First point of seq2 should be second point of seq1
            seq1_second_lat = sequences[0]["input_sequence"]["lat"].iloc[1]
            assert abs(seq2_first_lat - seq1_second_lat) < 1e-6