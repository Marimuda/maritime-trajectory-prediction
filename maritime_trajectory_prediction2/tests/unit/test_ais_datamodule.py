"""
Unit tests for AISDataModule.

Tests individual components and methods of the comprehensive
PyTorch Lightning DataModule for AIS maritime trajectory data.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.ais_datamodule import AISDataModule, AISTrajectoryDataset

# ==============================================================================
# Dataset Tests
# ==============================================================================


@pytest.mark.unit
class TestAISTrajectoryDataset:
    """Unit tests for AISTrajectoryDataset."""

    def test_init(self, trajectory_factory):
        """Test dataset initialization."""
        sequences = [trajectory_factory(seed=i) for i in range(10)]
        dataset = AISTrajectoryDataset(sequences=sequences)

        assert len(dataset) == 10
        assert dataset.sequence_length == 20
        assert dataset.prediction_horizon == 5
        assert dataset.input_dim == 4
        assert dataset.output_dim == 4

    def test_init_with_numpy_arrays(self):
        """Test dataset initialization with numpy arrays (optimized format)."""
        # Create sequences with numpy arrays instead of DataFrames
        sequences = []
        for i in range(10):
            np.random.seed(i)
            sequences.append(
                {
                    "input_sequence": np.random.randn(20, 4).astype(np.float32),
                    "target_sequence": np.random.randn(5, 4).astype(np.float32),
                    "mmsi": 123456789 + i,
                    "segment_id": i,
                }
            )

        dataset = AISTrajectoryDataset(sequences=sequences)

        assert len(dataset) == 10
        assert dataset.sequence_length == 20
        assert dataset.prediction_horizon == 5
        assert dataset.input_dim == 4
        assert dataset.output_dim == 4

    def test_getitem_with_numpy_arrays(self):
        """Test __getitem__ with numpy array sequences (optimized format)."""
        # Create sequence with numpy arrays
        sequence = {
            "input_sequence": np.random.randn(20, 4).astype(np.float32),
            "target_sequence": np.random.randn(5, 4).astype(np.float32),
            "mmsi": 123456789,
            "segment_id": 0,
        }

        dataset = AISTrajectoryDataset([sequence])
        item = dataset[0]

        # Check types
        assert isinstance(item["input"], torch.Tensor)
        assert isinstance(item["target"], torch.Tensor)
        assert item["input"].dtype == torch.float32
        assert item["target"].dtype == torch.float32

        # Check shapes
        assert item["input"].shape == (20, 4)
        assert item["target"].shape == (5, 4)

    def test_rejects_dataframe_input(self):
        """Test that dataset properly rejects DataFrame input with clear error."""
        # Create sequence with DataFrames (old format)
        sequence = {
            "input_sequence": pd.DataFrame(np.random.randn(20, 4)),
            "target_sequence": pd.DataFrame(np.random.randn(5, 4)),
            "mmsi": 123456789,
            "segment_id": 0,
        }

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError, match="Expected numpy arrays.*Use AISDataModule"):
            AISTrajectoryDataset([sequence])

    def test_getitem_returns_correct_format(self, trajectory_factory):
        """Test that __getitem__ returns correct batch format."""
        sequences = [trajectory_factory()]
        dataset = AISTrajectoryDataset(sequences)

        item = dataset[0]

        # Check keys match model expectations (singular, not plural)
        assert "input" in item
        assert "target" in item
        assert "mmsi" in item
        assert "segment_id" in item

        # Check types
        assert isinstance(item["input"], torch.Tensor)
        assert isinstance(item["target"], torch.Tensor)

    def test_getitem_correct_shapes(self, trajectory_factory):
        """Test that tensors have correct shapes."""
        seq_len = 20
        pred_horizon = 5
        n_features = 4

        sequences = [
            trajectory_factory(sequence_length=seq_len, prediction_horizon=pred_horizon)
        ]
        dataset = AISTrajectoryDataset(sequences)

        item = dataset[0]

        assert item["input"].shape == (seq_len, n_features)
        assert item["target"].shape == (pred_horizon, n_features)

    def test_getitem_tensor_dtype(self, trajectory_factory):
        """Test that tensors are float32."""
        sequences = [trajectory_factory()]
        dataset = AISTrajectoryDataset(sequences)

        item = dataset[0]

        assert item["input"].dtype == torch.float32
        assert item["target"].dtype == torch.float32

    def test_multiple_sequences(self, trajectory_factory):
        """Test dataset with multiple sequences."""
        n_sequences = 50
        sequences = [trajectory_factory(seed=i) for i in range(n_sequences)]
        dataset = AISTrajectoryDataset(sequences)

        assert len(dataset) == n_sequences

        # Test random access
        item_0 = dataset[0]
        item_25 = dataset[25]
        item_49 = dataset[49]

        # All should be valid
        assert item_0["input"].shape[0] == 20
        assert item_25["input"].shape[0] == 20
        assert item_49["input"].shape[0] == 20


# ==============================================================================
# DataModule Initialization Tests
# ==============================================================================


@pytest.mark.unit
class TestAISDataModuleInit:
    """Unit tests for AISDataModule initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Test initialization with default parameters."""
        data_file = tmp_path / "test_data.csv"
        data_file.touch()

        dm = AISDataModule(data_path=str(data_file))

        assert dm.data_path == data_file
        assert dm.batch_size == 32
        assert dm.sequence_length == 20
        assert dm.prediction_horizon == 5
        assert dm.train_split == 0.7
        assert dm.val_split == 0.15
        assert dm.test_split == 0.15
        assert dm.auto_preprocess is True

    def test_init_with_custom_params(self, tmp_path):
        """Test initialization with custom parameters."""
        data_file = tmp_path / "test_data.csv"
        data_file.touch()

        dm = AISDataModule(
            data_path=str(data_file),
            batch_size=64,
            sequence_length=30,
            prediction_horizon=10,
            features=["lat", "lon"],
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            num_workers=4,
            auto_preprocess=False,
        )

        assert dm.batch_size == 64
        assert dm.sequence_length == 30
        assert dm.prediction_horizon == 10
        assert dm.features == ["lat", "lon"]
        assert dm.train_split == 0.8
        assert dm.val_split == 0.1
        assert dm.test_split == 0.1
        assert dm.num_workers == 4
        assert dm.auto_preprocess is False

    def test_hyperparameters_saved(self, tmp_path):
        """Test that hyperparameters are saved for Lightning."""
        data_file = tmp_path / "test_data.csv"
        data_file.touch()

        dm = AISDataModule(
            data_path=str(data_file),
            batch_size=64,
            sequence_length=30,
        )

        # Lightning should save hyperparameters
        assert hasattr(dm, "hparams")


# ==============================================================================
# Preprocessing Detection Tests
# ==============================================================================


@pytest.mark.unit
class TestPreprocessingDetection:
    """Unit tests for preprocessing detection logic."""

    def test_needs_preprocessing_raw_log(self, tmp_path):
        """Test detection of raw log file needing preprocessing."""
        raw_log = tmp_path / "ais_data.log"
        raw_log.write_text("test log data")

        dm = AISDataModule(data_path=str(raw_log))

        assert dm._needs_preprocessing() is True

    def test_needs_preprocessing_zarr_exists(self, tmp_path):
        """Test that existing Zarr skips preprocessing."""
        raw_log = tmp_path / "ais_data.log"
        raw_log.write_text("test log data")

        # Create processed Zarr directory
        zarr_path = tmp_path / "processed" / "ais_positions.zarr"
        zarr_path.mkdir(parents=True)

        dm = AISDataModule(
            data_path=str(raw_log),
            processed_zarr_path=str(zarr_path),
        )

        assert dm._needs_preprocessing() is False

    def test_needs_preprocessing_force_reprocess(self, tmp_path):
        """Test force reprocessing even when Zarr exists."""
        raw_log = tmp_path / "ais_data.log"
        raw_log.write_text("test log data")

        zarr_path = tmp_path / "processed" / "ais_positions.zarr"
        zarr_path.mkdir(parents=True)

        dm = AISDataModule(
            data_path=str(raw_log),
            processed_zarr_path=str(zarr_path),
            force_reprocess=True,
        )

        assert dm._needs_preprocessing() is True

    def test_needs_preprocessing_csv_no_preprocess(self, tmp_path):
        """Test that CSV files don't need preprocessing."""
        csv_file = tmp_path / "ais_data.csv"
        csv_file.write_text(
            "mmsi,timestamp,lat,lon,sog,cog\n123456,2023-01-01,59.0,10.0,10.5,90.0\n"
        )

        dm = AISDataModule(data_path=str(csv_file))

        assert dm._needs_preprocessing() is False

    def test_needs_preprocessing_parquet_no_preprocess(self, tmp_path):
        """Test that Parquet files don't need preprocessing."""
        parquet_file = tmp_path / "ais_data.parquet"

        # Create a minimal parquet file
        df = pd.DataFrame(
            {
                "mmsi": [123456],
                "timestamp": [pd.Timestamp("2023-01-01")],
                "lat": [59.0],
                "lon": [10.0],
                "sog": [10.5],
                "cog": [90.0],
            }
        )
        df.to_parquet(parquet_file)

        dm = AISDataModule(data_path=str(parquet_file))

        assert dm._needs_preprocessing() is False


# ==============================================================================
# Effective Data Path Tests
# ==============================================================================


@pytest.mark.unit
class TestEffectiveDataPath:
    """Unit tests for _get_effective_data_path method."""

    def test_effective_path_uses_processed_when_exists(self, tmp_path):
        """Test that processed Zarr path is used when it exists."""
        raw_log = tmp_path / "raw" / "ais_data.log"
        raw_log.parent.mkdir(parents=True)
        raw_log.write_text("test")

        zarr_path = tmp_path / "processed" / "ais_positions.zarr"
        zarr_path.mkdir(parents=True)

        dm = AISDataModule(
            data_path=str(raw_log),
            processed_zarr_path=str(zarr_path),
        )

        effective_path = dm._get_effective_data_path()
        assert effective_path == zarr_path

    def test_effective_path_uses_original_when_no_processed(self, tmp_path):
        """Test that original path is used when no processed Zarr exists."""
        csv_file = tmp_path / "ais_data.csv"
        csv_file.write_text("test")

        dm = AISDataModule(data_path=str(csv_file))

        effective_path = dm._get_effective_data_path()
        assert effective_path == csv_file


# ==============================================================================
# Sequence Building Tests
# ==============================================================================


@pytest.mark.unit
class TestSequenceBuilding:
    """Unit tests for sequence building logic."""

    @patch("src.data.ais_datamodule.CacheManager")
    def test_build_sequences_uses_cache(
        self, mock_cache_manager, tmp_path, ais_data_factory
    ):
        """Test that sequence building uses cache when available."""
        csv_file = tmp_path / "ais_data.csv"
        ais_data = ais_data_factory(n_vessels=2, n_points_per_vessel=30)
        ais_data.to_csv(csv_file, index=False)

        # Mock cache hit
        mock_sequences = [
            {
                "input_sequence": pd.DataFrame(),
                "target_sequence": pd.DataFrame(),
                "mmsi": 0,
                "segment_id": 0,
            }
        ]
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = mock_sequences
        mock_cache_manager.return_value = mock_cache_instance

        dm = AISDataModule(data_path=str(csv_file))
        sequences = dm._build_sequences(csv_file)

        # Should return cached sequences
        assert sequences == mock_sequences
        mock_cache_instance.get.assert_called_once()

    @patch("src.data.ais_datamodule.CacheManager")
    def test_build_sequences_creates_and_caches(
        self, mock_cache_manager, tmp_path, ais_data_factory
    ):
        """Test that sequences are created and cached on cache miss."""
        csv_file = tmp_path / "ais_data.csv"
        ais_data = ais_data_factory(n_vessels=2, n_points_per_vessel=50)
        ais_data.to_csv(csv_file, index=False)

        # Mock cache miss
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache_manager.return_value = mock_cache_instance

        dm = AISDataModule(
            data_path=str(csv_file), sequence_length=10, prediction_horizon=5
        )

        with (
            patch.object(dm, "_load_processed_data", return_value=ais_data),
            patch.object(dm, "_create_sequences_from_data", return_value=[]),
        ):
            dm._build_sequences(csv_file)

            # Should call create_sequences
            mock_cache_instance.put.assert_called_once()


# ==============================================================================
# Data Loading Tests
# ==============================================================================


@pytest.mark.unit
class TestDataLoading:
    """Unit tests for data loading from different formats."""

    def test_load_csv(self, tmp_path, ais_data_factory):
        """Test loading data from CSV."""
        csv_file = tmp_path / "ais_data.csv"
        ais_data = ais_data_factory(n_vessels=2, n_points_per_vessel=20)
        ais_data.to_csv(csv_file, index=False)

        dm = AISDataModule(data_path=str(csv_file))
        loaded_data = dm._load_processed_data(csv_file)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 40  # 2 vessels * 20 points
        assert "mmsi" in loaded_data.columns
        assert "lat" in loaded_data.columns

    def test_load_parquet(self, tmp_path, ais_data_factory):
        """Test loading data from Parquet."""
        parquet_file = tmp_path / "ais_data.parquet"
        ais_data = ais_data_factory(n_vessels=2, n_points_per_vessel=20)
        ais_data.to_parquet(parquet_file, index=False)

        dm = AISDataModule(data_path=str(parquet_file))
        loaded_data = dm._load_processed_data(parquet_file)

        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 40
        assert "mmsi" in loaded_data.columns


# ==============================================================================
# DataLoader Tests
# ==============================================================================


@pytest.mark.unit
class TestDataLoaders:
    """Unit tests for DataLoader creation."""

    def test_train_dataloader_creation(self, trajectory_factory):
        """Test train dataloader is created correctly."""
        sequences = [trajectory_factory(seed=i) for i in range(100)]

        # Mock setup by directly setting datasets
        dm = AISDataModule(data_path="dummy.csv")
        dm.train_dataset = AISTrajectoryDataset(sequences)

        train_loader = dm.train_dataloader()

        assert train_loader.batch_size == 32
        assert train_loader.dataset == dm.train_dataset
        assert (
            len(train_loader) == 4
        )  # 100 samples / 32 batch_size = 4 batches (rounded up)

    def test_val_dataloader_creation(self, trajectory_factory):
        """Test validation dataloader is created correctly."""
        sequences = [trajectory_factory(seed=i) for i in range(50)]

        dm = AISDataModule(data_path="dummy.csv", batch_size=16)
        dm.val_dataset = AISTrajectoryDataset(sequences)

        val_loader = dm.val_dataloader()

        assert val_loader.batch_size == 16
        assert val_loader.dataset == dm.val_dataset
        assert len(val_loader) == 4  # 50 / 16 = 4

    def test_test_dataloader_creation(self, trajectory_factory):
        """Test test dataloader is created correctly."""
        sequences = [trajectory_factory(seed=i) for i in range(30)]

        dm = AISDataModule(data_path="dummy.csv", batch_size=8)
        dm.test_dataset = AISTrajectoryDataset(sequences)

        test_loader = dm.test_dataloader()

        assert test_loader.batch_size == 8
        assert test_loader.dataset == dm.test_dataset
        assert len(test_loader) == 4  # 30 / 8 = 4

    def test_dataloader_batch_format(self, trajectory_factory):
        """Test that dataloaders produce correctly formatted batches."""
        sequences = [trajectory_factory(seed=i) for i in range(10)]

        dm = AISDataModule(data_path="dummy.csv", batch_size=2)
        dm.train_dataset = AISTrajectoryDataset(sequences)

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check batch keys
        assert "input" in batch
        assert "target" in batch
        assert "mmsi" in batch
        assert "segment_id" in batch

        # Check batch dimensions
        assert batch["input"].shape[0] == 2  # batch_size
        assert batch["input"].shape[1] == 20  # sequence_length
        assert batch["input"].shape[2] == 4  # n_features
        assert batch["target"].shape[0] == 2
        assert batch["target"].shape[1] == 5  # prediction_horizon
        assert batch["target"].shape[2] == 4
