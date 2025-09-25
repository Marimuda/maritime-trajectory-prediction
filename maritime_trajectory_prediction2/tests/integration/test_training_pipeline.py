"""
Integration test to verify the complete training pipeline works end-to-end.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datamodule import AISDataModule


class TestTrainingPipeline:
    """Test the complete training pipeline integration."""

    @pytest.fixture
    def sample_ais_data(self):
        """Create sample AIS data for testing."""
        # Create realistic sample data
        data = []
        for vessel_id in [123456789, 987654321]:
            for i in range(100):  # 100 points per vessel
                data.append(
                    {
                        "mmsi": vessel_id,
                        "time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                        "lat": 62.0 + np.random.normal(0, 0.01),
                        "lon": -6.7 + np.random.normal(0, 0.01),
                        "sog": np.random.uniform(0, 15),
                        "cog": np.random.uniform(0, 360),
                        "heading": np.random.uniform(0, 360),
                        "turn": np.random.normal(0, 5),
                        "nav_status": 0,
                        "msg_type": 1,
                        "accuracy": True,
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_data_file(self, sample_ais_data):
        """Create temporary data file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".parquet", delete=False
        ) as f:
            sample_ais_data.to_parquet(f.name)
            yield f.name
        os.unlink(f.name)

    def test_datamodule_creation(self, temp_data_file):
        """Test that AISDataModule can be created and setup."""
        datamodule = AISDataModule(
            data_path=temp_data_file,
            batch_size=4,
            sequence_length=10,
            prediction_horizon=5,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        # Setup should work without errors
        datamodule.setup()

        # Check data loaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0

    def test_data_batch_format(self, temp_data_file):
        """Test that data batches have the expected format."""
        datamodule = AISDataModule(
            data_path=temp_data_file,
            batch_size=4,
            sequence_length=10,
            prediction_horizon=5,
            num_workers=0,
        )

        datamodule.setup()
        train_loader = datamodule.train_dataloader()

        # Get a sample batch
        batch = next(iter(train_loader))

        # Check batch structure
        assert isinstance(batch, dict)
        assert "inputs" in batch
        assert "targets" in batch

        # Check tensor shapes
        input_tensor = batch["inputs"]
        target_tensor = batch["targets"]

        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)

        # Check dimensions: [batch_size, sequence_length, features]
        assert len(input_tensor.shape) == 3
        assert len(target_tensor.shape) == 3

        batch_size, seq_len, input_features = input_tensor.shape
        _, pred_horizon, target_features = target_tensor.shape

        assert batch_size <= 4  # batch_size or smaller for last batch
        assert seq_len == 10
        assert pred_horizon == 5
        assert input_features >= 4  # At least lat, lon, sog, cog
        assert target_features == 4  # Exactly lat, lon, sog, cog

        print(f"Input shape: {input_tensor.shape}")
        print(f"Target shape: {target_tensor.shape}")
        print(f"Input features: {input_features}, Target features: {target_features}")

    def test_model_forward_pass(self, temp_data_file):
        """Test that model can process data batches."""
        datamodule = AISDataModule(
            data_path=temp_data_file,
            batch_size=4,
            sequence_length=10,
            prediction_horizon=5,
            num_workers=0,
        )

        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Get actual dimensions from data
        input_dim = batch["inputs"].shape[-1]
        prediction_horizon = batch["targets"].shape[1]  # Get from actual data

        # Create model with correct dimensions
        from src.models.motion_transformer import (
            MARITIME_MTR_CONFIG,
            create_motion_transformer,
        )

        config = MARITIME_MTR_CONFIG["small"].copy()
        config["input_dim"] = input_dim
        config["prediction_horizon"] = prediction_horizon
        model = create_motion_transformer(**config)

        model.eval()

        # Test forward pass
        with torch.no_grad():
            input_tensor = batch["inputs"]
            target_tensor = batch["targets"]

            # Forward pass should not error
            outputs = model(input_tensor)

            # Check output structure
            assert isinstance(outputs, dict)
            assert "trajectories" in outputs
            assert "confidences" in outputs

            trajectories = outputs["trajectories"]
            confidences = outputs["confidences"]

            # Check output shapes
            batch_size, pred_horizon, n_queries, output_dim = trajectories.shape
            assert pred_horizon == prediction_horizon  # Should match data
            assert output_dim == 4  # lat, lon, sog, cog
            assert confidences.shape == (batch_size, n_queries)

            # Validate output shapes match target expectations
            assert (
                trajectories.shape[-1] == target_tensor.shape[-1]
            ), "Output and target feature dimensions must match"
            assert (
                trajectories.shape[1] == target_tensor.shape[1]
            ), "Output and target prediction horizons must match"

            print(f"Trajectories shape: {trajectories.shape}")
            print(f"Confidences shape: {confidences.shape}")

    def test_loss_computation(self, temp_data_file):
        """Test that loss computation works."""
        datamodule = AISDataModule(
            data_path=temp_data_file,
            batch_size=4,
            sequence_length=10,
            prediction_horizon=5,
            num_workers=0,
        )

        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Get actual dimensions from data
        input_dim = batch["inputs"].shape[-1]
        prediction_horizon = batch["targets"].shape[1]

        # Create model
        from src.models.motion_transformer import (
            MARITIME_MTR_CONFIG,
            create_motion_transformer,
        )

        config = MARITIME_MTR_CONFIG["small"].copy()
        config["input_dim"] = input_dim
        config["prediction_horizon"] = prediction_horizon
        model = create_motion_transformer(**config)

        model.train()

        input_tensor = batch["inputs"]
        target_tensor = batch["targets"]

        # Forward pass
        outputs = model(input_tensor)

        # Loss computation should work
        loss_dict = model.compute_loss(outputs, target_tensor)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert torch.is_tensor(loss_dict["total_loss"])
        assert loss_dict["total_loss"].requires_grad

        print(f"Loss: {loss_dict['total_loss'].item():.4f}")

    def test_feature_alignment(self, temp_data_file):
        """Test that features are properly aligned between input and target."""
        datamodule = AISDataModule(
            data_path=temp_data_file,
            batch_size=4,
            sequence_length=10,
            prediction_horizon=5,
            num_workers=0,
        )

        datamodule.setup()

        # Check what features are actually being used
        if hasattr(datamodule, "dataset") and hasattr(datamodule.dataset, "features"):
            print(f"Dataset features: {datamodule.dataset.features}")

        batch = next(iter(datamodule.train_dataloader()))

        # Print actual tensor statistics
        input_tensor = batch["inputs"]
        target_tensor = batch["targets"]

        print(f"Input tensor - shape: {input_tensor.shape}")
        print(
            f"Input tensor - min/max: {input_tensor.min():.3f}/{input_tensor.max():.3f}"
        )
        print(
            f"Input tensor - mean/std: {input_tensor.mean():.3f}/{input_tensor.std():.3f}"
        )

        print(f"Target tensor - shape: {target_tensor.shape}")
        print(
            f"Target tensor - min/max: {target_tensor.min():.3f}/{target_tensor.max():.3f}"
        )
        print(
            f"Target tensor - mean/std: {target_tensor.mean():.3f}/{target_tensor.std():.3f}"
        )

        # Check for NaN values
        assert not torch.isnan(input_tensor).any(), "Input tensor contains NaN values"
        assert not torch.isnan(target_tensor).any(), "Target tensor contains NaN values"
