"""
Integration tests for anomaly detection pipeline.

Tests the complete pipeline from data processing through model training,
ensuring that the unsupervised anomaly detection (autoencoder-based) works
correctly without hard-coded rules.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset_builders import AnomalyDetectionBuilder
from src.data.multi_task_processor import MLTask
from src.data.pipeline import DatasetConfig
from src.models.baseline_models.anomaly_autoencoder import AnomalyAutoencoderLightning


@pytest.fixture
def vessel_data():
    """Create sample vessel trajectory data."""
    np.random.seed(42)
    n_points = 200

    data = {
        "mmsi": [123456] * n_points,
        "time": pd.date_range("2024-01-01", periods=n_points, freq="1min"),
        "lat": 59.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "lon": 10.0 + np.cumsum(np.random.randn(n_points) * 0.001),
        "sog": 10.0 + np.random.randn(n_points) * 1.0,
        "cog": 45.0 + np.random.randn(n_points) * 5.0,
        "heading": 45.0 + np.random.randn(n_points) * 5.0,
        "turn": np.random.randn(n_points) * 2.0,
    }

    return pd.DataFrame(data)


@pytest.mark.integration
class TestAnomalyDetectionPipeline:
    """Integration tests for complete anomaly detection pipeline."""

    def test_full_pipeline_data_to_sequences(self, vessel_data):
        """Test complete pipeline from raw data to sequences."""
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=20,
            prediction_horizon=0,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = AnomalyDetectionBuilder(config)

        # Build features
        features_df = builder.build_features(vessel_data)
        assert len(features_df) > 0
        assert "mmsi" in features_df.columns
        assert "time" in features_df.columns

        # Create sequences
        X, y = builder.create_sequences(features_df)

        # Verify sequences were created
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape == y.shape

        # Verify reconstruction property (target equals input)
        assert X.shape[1] == config.sequence_length
        X_flat = X.flatten()
        y_flat = y.flatten()
        mismatches = 0
        for i in range(len(X_flat)):
            x_val, y_val = X_flat[i], y_flat[i]
            if isinstance(x_val, float) and isinstance(y_val, float):
                if not (np.isnan(x_val) and np.isnan(y_val)) and x_val != y_val:
                    mismatches += 1
            elif x_val != y_val:
                mismatches += 1

        assert mismatches == 0, f"Found {mismatches} mismatches between X and y"

    def test_autoencoder_training_with_builder_output(self, vessel_data):
        """Test that autoencoder can train on data from builder."""
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=20,
            prediction_horizon=0,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = AnomalyDetectionBuilder(config)

        # Build features and sequences
        features_df = builder.build_features(vessel_data)
        X, y = builder.create_sequences(features_df)

        # Convert to tensors
        # Need to convert mixed-type array to float
        X_numeric = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    val = X[i, j, k]
                    if isinstance(val, bool):
                        X_numeric[i, j, k] = float(val)
                    elif isinstance(val, int | float):
                        if np.isnan(val):
                            X_numeric[i, j, k] = 0.0  # Fill NaN with 0 for model
                        else:
                            X_numeric[i, j, k] = float(val)
                    else:
                        X_numeric[i, j, k] = 0.0

        X_tensor = torch.FloatTensor(X_numeric)

        # Create autoencoder model
        input_dim = X_tensor.shape[2]
        model = AnomalyAutoencoderLightning(
            input_dim=input_dim,
            encoding_dim=32,
            hidden_dims=[64, 48],
            dropout=0.2,
            learning_rate=1e-3,
        )

        # Test forward pass
        with torch.no_grad():
            # Create batch
            batch = {"input": X_tensor[:5]}  # Use first 5 sequences

            # Forward pass (reconstruction)
            recon, encoding = model(batch["input"])

            # Check output shapes
            assert recon.shape == batch["input"].shape
            assert encoding.shape[0] == batch["input"].shape[0]

        # Test training step
        model.train()
        batch = {"input": X_tensor[:5]}
        loss = model.training_step(batch, batch_idx=0)

        # Loss should be computed (MSE between input and reconstruction)
        assert loss is not None
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # MSE loss should be non-negative

    def test_reconstruction_error_as_anomaly_score(self, vessel_data):
        """Test that reconstruction error can be used for anomaly detection."""
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=20,
            prediction_horizon=0,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = AnomalyDetectionBuilder(config)

        # Build features and sequences
        features_df = builder.build_features(vessel_data)
        X, y = builder.create_sequences(features_df)

        # Convert to numeric tensors
        X_numeric = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    val = X[i, j, k]
                    if isinstance(val, bool):
                        X_numeric[i, j, k] = float(val)
                    elif isinstance(val, int | float):
                        X_numeric[i, j, k] = 0.0 if np.isnan(val) else float(val)
                    else:
                        X_numeric[i, j, k] = 0.0

        X_tensor = torch.FloatTensor(X_numeric)

        # Create and "train" autoencoder (just forward pass)
        input_dim = X_tensor.shape[2]
        model = AnomalyAutoencoderLightning(
            input_dim=input_dim,
            encoding_dim=32,
            hidden_dims=[64, 48],
        )

        model.eval()
        with torch.no_grad():
            # Get reconstructions for all sequences
            recon, _ = model(X_tensor)

            # Calculate reconstruction error (MSE) for each sequence
            recon_errors = torch.mean((X_tensor - recon) ** 2, dim=(1, 2))

            # Verify we can compute anomaly scores
            assert recon_errors.shape[0] == X_tensor.shape[0]
            assert torch.all(torch.isfinite(recon_errors))
            assert torch.all(recon_errors >= 0)

            # High reconstruction error = anomaly
            # Low reconstruction error = normal

            # For a well-trained model, we could threshold on recon_errors
            # to identify anomalies (e.g., top 5% highest errors)
            threshold = torch.quantile(recon_errors, 0.95)
            anomalies = recon_errors > threshold

            # Should identify some anomalies
            assert torch.sum(anomalies) > 0
            assert torch.sum(anomalies) < len(anomalies)  # Not all are anomalies

    def test_no_hard_coded_labels_used(self, vessel_data):
        """Verify that no hard-coded anomaly labels are used in training."""
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=20,
            prediction_horizon=0,
            validation_split=0.2,
            test_split=0.1,
            random_seed=42,
        )

        builder = AnomalyDetectionBuilder(config)

        # Build features and targets
        features_df = builder.build_features(vessel_data)
        targets_df = builder.build_targets(features_df)

        # Targets should NOT contain synthetic anomaly columns
        synthetic_columns = [
            "anomaly_speed",
            "anomaly_course",
            "anomaly_position",
            "anomaly_overall",
        ]
        for col in synthetic_columns:
            assert (
                col not in targets_df.columns
            ), f"Found hard-coded anomaly column: {col}"

        # Targets should be same as features (for reconstruction)
        assert set(features_df.columns) == set(targets_df.columns)
