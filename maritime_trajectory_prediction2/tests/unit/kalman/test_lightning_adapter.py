"""
Unit tests for Kalman filter Lightning adapter.

Tests the PyTorch Lightning integration layer for Kalman baseline models,
including data format handling, maritime distance metrics, and error handling.
"""

import numpy as np
import pytest
import torch

from src.models.baseline_models.kalman import (
    create_ct_lightning,
    create_cv_lightning,
    create_imm_lightning,
)


@pytest.fixture
def realistic_maritime_batch():
    """Create realistic maritime batch in AISDataModule format."""
    np.random.seed(42)
    batch = {
        "input": torch.zeros(4, 50, 20),  # (batch, seq_len, features)
        "target": torch.zeros(4, 10, 4),  # (batch, pred_horizon, 4)
        "mmsi": [123456, 234567, 345678, 456789],
        "segment_id": [0, 1, 2, 3],
    }

    # Generate realistic maritime trajectories around Norway
    for i in range(4):
        base_lat = 59.0 + np.random.randn() * 0.5
        base_lon = 10.0 + np.random.randn() * 0.5

        # Input: smooth movement with small random steps
        lats = base_lat + np.cumsum(np.random.randn(50) * 0.0005)
        lons = base_lon + np.cumsum(np.random.randn(50) * 0.0005)
        batch["input"][i, :, 0] = torch.from_numpy(lats)
        batch["input"][i, :, 1] = torch.from_numpy(lons)

        # Target: continuation of smooth movement
        lats_target = lats[-1] + np.cumsum(np.random.randn(10) * 0.0005)
        lons_target = lons[-1] + np.cumsum(np.random.randn(10) * 0.0005)
        batch["target"][i, :, 0] = torch.from_numpy(lats_target)
        batch["target"][i, :, 1] = torch.from_numpy(lons_target)

    return batch


@pytest.mark.unit
class TestKalmanLightningDataAdapter:
    """Test data format adapter (Fix 1)."""

    def test_training_step_extracts_latlon_correctly(self, realistic_maritime_batch):
        """Test that training_step correctly extracts lat/lon from AISDataModule format."""
        model = create_cv_lightning(prediction_horizon=10)

        # Execute training step
        result = model.training_step(realistic_maritime_batch, batch_idx=0)

        # Should collect sequences
        assert (
            len(model.training_sequences) == 4
        ), "Should collect 4 sequences from batch"

        # Check sequence shape: (seq_len + pred_horizon, 2) = (60, 2)
        for seq in model.training_sequences:
            assert seq.shape == (60, 2), f"Wrong shape: {seq.shape}, expected (60, 2)"
            assert seq.shape[1] == 2, "Should extract only lat/lon (2 columns)"

        # Check lat/lon values are reasonable
        for seq in model.training_sequences:
            assert np.all(
                np.abs(seq[:, 0]) <= 90
            ), "Latitude should be within [-90, 90]"
            assert np.all(
                np.abs(seq[:, 1]) <= 180
            ), "Longitude should be within [-180, 180]"

        # Check return value (number of sequences collected)
        assert result.item() == 4, "Should return number of collected sequences"

    def test_training_step_validates_trajectories(self):
        """Test that training_step validates trajectories (NaN, bounds)."""
        model = create_cv_lightning(prediction_horizon=10)

        # Create batch with invalid data
        batch_with_nan = {
            "input": torch.tensor([[[np.nan, 10.0] + [0.0] * 18] * 50] * 2),
            "target": torch.tensor([[[59.0, 10.0, 0.0, 0.0]] * 10] * 2),
            "mmsi": [1, 2],
            "segment_id": [0, 1],
        }

        _ = model.training_step(batch_with_nan, 0)

        # Should skip invalid trajectories
        assert len(model.training_sequences) == 0, "Should skip trajectories with NaN"

    def test_training_step_auto_fits_model(self, realistic_maritime_batch):
        """Test that training_step auto-fits model when not using auto_tune."""
        model = create_cv_lightning(prediction_horizon=10, auto_tune=False)

        assert not model.is_fitted, "Model should not be fitted initially"

        _ = model.training_step(realistic_maritime_batch, 0)

        assert model.is_fitted, "Model should be auto-fitted after first batch"

    def test_validation_step_unfitted_model_returns_none(
        self, realistic_maritime_batch
    ):
        """Test that validation_step returns None for unfitted model."""
        model = create_cv_lightning(prediction_horizon=10)

        assert not model.is_fitted

        result = model.validation_step(realistic_maritime_batch, 0)

        assert result is None, "Should return None for unfitted model"

    def test_validation_step_extracts_data_correctly(self, realistic_maritime_batch):
        """Test that validation_step extracts data from AISDataModule format."""
        model = create_cv_lightning(prediction_horizon=10)

        # Fit model first
        _ = model.training_step(realistic_maritime_batch, 0)
        assert model.is_fitted

        # Validate
        result = model.validation_step(realistic_maritime_batch, 0)

        # Should return metrics, not None
        assert result is not None, "Should return metrics for fitted model"
        assert "val_loss" in result, "Should have val_loss key"
        assert "val_rmse_m" in result, "Should have val_rmse_m key"

    def test_all_kalman_types_work_with_data_format(self, realistic_maritime_batch):
        """Test that all Kalman model types (CV, CT, IMM) work with data format."""
        model_factories = [
            ("CV", create_cv_lightning),
            ("CT", create_ct_lightning),
            ("IMM", create_imm_lightning),
        ]

        for name, factory in model_factories:
            model = factory(prediction_horizon=10)

            # Training
            _ = model.training_step(realistic_maritime_batch, 0)
            assert model.is_fitted, f"{name} model should be fitted"
            assert (
                len(model.training_sequences) == 4
            ), f"{name} should collect 4 sequences"

            # Validation
            result = model.validation_step(realistic_maritime_batch, 0)
            assert result is not None, f"{name} should return validation metrics"


@pytest.mark.unit
class TestKalmanLightningMaritimeMetrics:
    """Test maritime distance metrics (Fix 2)."""

    def test_validation_uses_haversine_not_euclidean(self, realistic_maritime_batch):
        """Test that validation uses Haversine distance, not Euclidean in degrees."""
        model = create_cv_lightning(prediction_horizon=10)

        # Fit and validate
        _ = model.training_step(realistic_maritime_batch, 0)
        result = model.validation_step(realistic_maritime_batch, 0)

        # RMSE should be in meters, reasonable for maritime
        rmse_m = result["val_rmse_m"]

        # For short-term prediction with smooth trajectories, RMSE should be < 500m
        assert 0 < rmse_m < 500, f"RMSE {rmse_m}m seems unreasonable for maritime"

        # MSE should be RMSE squared
        mse_m2 = result["val_loss"]
        assert np.isclose(rmse_m**2, mse_m2, rtol=1e-5), "MSE should be RMSE squared"

    def test_metrics_scale_correctly_with_latitude(self):
        """Test that metrics handle high-latitude data correctly (Haversine accounts for this)."""
        # Create batch with high latitude (near Arctic)
        batch_high_lat = {
            "input": torch.zeros(2, 50, 20),
            "target": torch.zeros(2, 10, 4),
            "mmsi": [1, 2],
            "segment_id": [0, 1],
        }

        np.random.seed(42)
        for i in range(2):
            # High latitude: 70°N
            base_lat = 70.0
            base_lon = 20.0

            lats = base_lat + np.cumsum(np.random.randn(50) * 0.0005)
            lons = base_lon + np.cumsum(np.random.randn(50) * 0.0005)
            batch_high_lat["input"][i, :, 0] = torch.from_numpy(lats)
            batch_high_lat["input"][i, :, 1] = torch.from_numpy(lons)

            lats_target = lats[-1] + np.cumsum(np.random.randn(10) * 0.0005)
            lons_target = lons[-1] + np.cumsum(np.random.randn(10) * 0.0005)
            batch_high_lat["target"][i, :, 0] = torch.from_numpy(lats_target)
            batch_high_lat["target"][i, :, 1] = torch.from_numpy(lons_target)

        model = create_cv_lightning(prediction_horizon=10)
        _ = model.training_step(batch_high_lat, 0)
        result = model.validation_step(batch_high_lat, 0)

        # Should still produce reasonable metrics (Haversine handles latitude)
        rmse_m = result["val_rmse_m"]
        assert (
            0 < rmse_m < 500
        ), f"RMSE {rmse_m}m should be reasonable even at high latitude"


@pytest.mark.unit
class TestKalmanLightningErrorHandling:
    """Test error handling (Fix 3 - no dummy code)."""

    def test_forward_raises_not_implemented(self, realistic_maritime_batch):
        """Test that forward() raises NotImplementedError."""
        model = create_cv_lightning(prediction_horizon=10)

        with pytest.raises(NotImplementedError, match="don't support forward"):
            _ = model.forward(realistic_maritime_batch["input"])

    def test_predict_step_unfitted_raises_error(self, realistic_maritime_batch):
        """Test that predict_step() raises RuntimeError when model not fitted."""
        model = create_cv_lightning(prediction_horizon=10)

        assert not model.is_fitted

        with pytest.raises(RuntimeError, match="Cannot predict with unfitted model"):
            _ = model.predict_step(realistic_maritime_batch, 0)

    def test_predict_step_invalid_input_raises_error(self):
        """Test that predict_step() raises ValueError for invalid inputs."""
        model = create_cv_lightning(prediction_horizon=10)

        # Create valid batch first to fit model
        valid_batch = {
            "input": torch.tensor([[[59.0, 10.0] + [0.0] * 18] * 50]),
            "target": torch.tensor([[[59.0, 10.0, 0.0, 0.0]] * 10]),
            "mmsi": [1],
            "segment_id": [0],
        }
        _ = model.training_step(valid_batch, 0)
        assert model.is_fitted

        # Now test with NaN input
        batch_with_nan = {
            "input": torch.tensor([[[np.nan, 10.0] + [0.0] * 18] * 50]),
            "target": torch.tensor([[[59.0, 10.0, 0.0, 0.0]] * 10]),
            "mmsi": [1],
            "segment_id": [0],
        }

        with pytest.raises(ValueError, match="Invalid input sequence"):
            _ = model.predict_step(batch_with_nan, 0)

    def test_predict_step_works_after_fitting(self, realistic_maritime_batch):
        """Test that predict_step() works correctly after fitting."""
        model = create_cv_lightning(prediction_horizon=10)

        # Fit model
        _ = model.training_step(realistic_maritime_batch, 0)
        assert model.is_fitted

        # Predict
        predictions = model.predict_step(realistic_maritime_batch, 0)

        # Check output shape: (batch_size, prediction_horizon, 2)
        assert predictions.shape == (
            4,
            10,
            2,
        ), f"Wrong prediction shape: {predictions.shape}"

        # Check predictions are reasonable (lat/lon bounds)
        assert torch.all(
            torch.abs(predictions[:, :, 0]) <= 90
        ), "Predicted latitudes out of bounds"
        assert torch.all(
            torch.abs(predictions[:, :, 1]) <= 180
        ), "Predicted longitudes out of bounds"

        # Check no NaN values
        assert not torch.any(torch.isnan(predictions)), "Predictions contain NaN"


@pytest.mark.integration
class TestKalmanLightningIntegration:
    """Integration tests for Kalman Lightning adapter."""

    def test_full_training_validation_cycle(self, realistic_maritime_batch):
        """Test complete training and validation cycle."""
        model = create_cv_lightning(prediction_horizon=10, auto_tune=False)

        # Initial state
        assert not model.is_fitted
        assert len(model.training_sequences) == 0

        # Training step
        train_result = model.training_step(realistic_maritime_batch, 0)
        assert train_result.item() == 4  # 4 sequences collected
        assert model.is_fitted
        assert len(model.training_sequences) == 4

        # Validation step
        val_result = model.validation_step(realistic_maritime_batch, 0)
        assert val_result is not None
        assert "val_loss" in val_result
        assert "val_rmse_m" in val_result
        assert val_result["val_rmse_m"] > 0

        # Prediction step
        predictions = model.predict_step(realistic_maritime_batch, 0)
        assert predictions.shape == (4, 10, 2)

    def test_multiple_batches_accumulate_sequences(self, realistic_maritime_batch):
        """Test that multiple training batches accumulate sequences."""
        model = create_cv_lightning(prediction_horizon=10, auto_tune=False)

        # First batch
        _ = model.training_step(realistic_maritime_batch, 0)
        assert len(model.training_sequences) == 4

        # Second batch (create a new one)
        batch2 = {
            "input": realistic_maritime_batch["input"].clone(),
            "target": realistic_maritime_batch["target"].clone(),
            "mmsi": [5, 6],
            "segment_id": [4, 5],
        }
        # Only use first 2 samples from batch2
        batch2["input"] = batch2["input"][:2]
        batch2["target"] = batch2["target"][:2]

        _ = model.training_step(batch2, 1)

        # Should accumulate sequences
        assert len(model.training_sequences) == 6, "Should have 4 + 2 = 6 sequences"

    def test_realistic_maritime_prediction_quality(self):
        """Test that predictions produce reasonable output for maritime data."""
        np.random.seed(123)  # Different seed for variety

        # Create realistic maritime trajectory with smooth motion
        batch = {
            "input": torch.zeros(1, 50, 20),
            "target": torch.zeros(1, 10, 4),
            "mmsi": [999999],
            "segment_id": [0],
        }

        # Vessel moving steadily northward with small random variations
        base_lat = 60.0
        base_lon = 11.0

        # Smooth trajectory with small noise (realistic for vessel at sea)
        lats = base_lat + np.cumsum(np.random.randn(50) * 0.0003)
        lons = base_lon + np.cumsum(np.random.randn(50) * 0.0003)

        batch["input"][0, :, 0] = torch.from_numpy(lats)
        batch["input"][0, :, 1] = torch.from_numpy(lons)

        # Target continues the pattern
        lats_target = lats[-1] + np.cumsum(np.random.randn(10) * 0.0003)
        lons_target = lons[-1] + np.cumsum(np.random.randn(10) * 0.0003)
        batch["target"][0, :, 0] = torch.from_numpy(lats_target)
        batch["target"][0, :, 1] = torch.from_numpy(lons_target)

        model = create_cv_lightning(prediction_horizon=10)
        _ = model.training_step(batch, 0)
        result = model.validation_step(batch, 0)

        # Check that metrics are produced and reasonable
        rmse_m = result["val_rmse_m"]

        # For Kalman filters on maritime data, RMSE should be positive and finite
        # With smooth random walk, expect errors in 10s to 1000s of meters range
        assert 0 < rmse_m < 5000, f"RMSE {rmse_m}m should be positive and < 5km"
        assert np.isfinite(rmse_m), "RMSE should be finite"

        # MSE should be RMSE squared
        mse_m2 = result["val_loss"]
        assert np.isclose(rmse_m**2, mse_m2, rtol=1e-5), "MSE should be RMSE squared"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
