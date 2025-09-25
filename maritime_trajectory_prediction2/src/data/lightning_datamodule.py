"""
PyTorch Lightning DataModule for AIS trajectory data.

Implements the guideline's recommendations for windowed datasets,
efficient data loading, and multi-GPU training support.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# Calculate optimal number of workers (leave 4 cores free)
OPTIMAL_WORKERS = max(1, os.cpu_count() - 4)

logger = logging.getLogger(__name__)


class AISWindowDataset(Dataset):
    """
    Windowed AIS dataset for trajectory sequence learning.

    Implements the guideline's windowing strategy with dynamic padding,
    static data integration, and efficient memory usage.
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 60,
        stride: int = 10,
        prediction_horizon: int = 10,
        split: str = "train",
        attach_static: bool = True,
        static_data_path: str | None = None,
        scaler: tuple[np.ndarray, np.ndarray] | None = None,
        features: list[str] = None,
    ):
        """
        Initialize windowed AIS dataset.

        Args:
            zarr_path: Path to main trajectory Zarr store
            window_size: Length of input sequence
            stride: Stride between windows
            prediction_horizon: Length of prediction sequence
            split: Data split ('train', 'val', 'test')
            attach_static: Whether to include static vessel data
            static_zarr_path: Path to static vessel data
            scaler: Tuple of (mean, std) for normalization
            features: List of feature names to include
        """
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.attach_static = attach_static
        self.scaler = scaler

        # Default features
        self.features = features or ["lat", "lon", "sog", "cog", "heading"]

        # Load main dataset
        self.ds = xr.open_zarr(zarr_path, chunks={"time": window_size, "mmsi": 1})

        # Load static data if requested
        if attach_static and static_zarr_path:
            self.static_ds = xr.open_zarr(static_zarr_path)
        else:
            self.static_ds = None

        # Apply temporal split
        self._apply_temporal_split()

        # Generate window indices
        self._generate_window_indices()

        logger.info(f"Initialized {split} dataset with {len(self.indices)} windows")

    def _apply_temporal_split(self):
        """Apply temporal data splitting following guideline."""
        # Use fixed cutoff date for reproducible splits
        cut_date = pd.Timestamp("2025-06-01", tz="UTC")

        if self.split == "train":
            self.ds = self.ds.sel(time=slice(None, cut_date - pd.Timedelta("1s")))
        elif self.split == "val":
            val_end = cut_date + pd.Timedelta("7d")
            self.ds = self.ds.sel(time=slice(cut_date, val_end))
        else:  # test
            test_start = cut_date + pd.Timedelta("7d")
            self.ds = self.ds.sel(time=slice(test_start, None))

    def _generate_window_indices(self):
        """Generate valid window indices for the dataset."""
        self.indices = []
        min_sequence_length = self.window_size + self.prediction_horizon

        for mmsi in self.ds.mmsi.values:
            vessel_data = self.ds.sel(mmsi=mmsi)

            # Find valid time indices (non-NaN positions)
            valid_mask = vessel_data.lat.notnull() & vessel_data.lon.notnull()
            valid_times = vessel_data.time.where(valid_mask, drop=True)

            if len(valid_times) < min_sequence_length:
                continue

            # Generate window start indices
            for i in range(0, len(valid_times) - min_sequence_length + 1, self.stride):
                self.indices.append((mmsi, i))

    def __len__(self) -> int:
        """Return number of valid windows."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a windowed sequence sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input and target tensors
        """
        mmsi, time_idx = self.indices[idx]

        # Extract vessel data
        vessel_data = self.ds.sel(mmsi=mmsi)

        # Get valid time indices
        valid_mask = vessel_data.lat.notnull() & vessel_data.lon.notnull()
        valid_times = vessel_data.time.where(valid_mask, drop=True)

        # Extract sequences
        input_times = valid_times[time_idx : time_idx + self.window_size]
        target_times = valid_times[
            time_idx + self.window_size : time_idx
            + self.window_size
            + self.prediction_horizon
        ]

        # Get feature data
        input_data = []
        target_data = []

        for feature in self.features:
            if feature in vessel_data:
                input_vals = vessel_data[feature].sel(time=input_times).values
                target_vals = vessel_data[feature].sel(time=target_times).values

                input_data.append(input_vals)
                target_data.append(target_vals)

        # Stack features
        input_tensor = np.stack(input_data, axis=0).astype(np.float32)  # [C, T]
        target_tensor = np.stack(target_data, axis=0).astype(np.float32)  # [C, T]

        # Add static features if available
        if self.static_ds is not None and mmsi in self.static_ds.mmsi:
            static_data = self.static_ds.sel(mmsi=mmsi)
            static_features = []

            for var in static_data.data_vars:
                static_features.append(float(static_data[var].values))

            if static_features:
                static_tensor = np.array(static_features, dtype=np.float32)
                # Repeat static features across time dimension
                static_expanded = np.repeat(
                    static_tensor[:, None], self.window_size, axis=1
                )
                input_tensor = np.concatenate([input_tensor, static_expanded], axis=0)

        # Apply normalization if provided
        if self.scaler is not None:
            mean, std = self.scaler
            input_tensor = (input_tensor - mean[:, None]) / std[:, None]
            target_tensor = (target_tensor - mean[: len(target_tensor), None]) / std[
                : len(target_tensor), None
            ]

        return {
            "input": torch.from_numpy(input_tensor),
            "target": torch.from_numpy(target_tensor),
            "mmsi": torch.tensor(mmsi, dtype=torch.long),
            "time_idx": torch.tensor(time_idx, dtype=torch.long),
        }


class AISLightningDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AIS trajectory data.

    Implements the guideline's recommendations for efficient data loading,
    multi-GPU support, and performance optimization.
    """

    def __init__(
        self,
        zarr_path: str,
        static_zarr_path: str | None = None,
        window_size: int = 60,
        stride: int = 10,
        prediction_horizon: int = 10,
        batch_size: int = 32,
        num_workers: int = OPTIMAL_WORKERS,
        pin_memory: bool = True,
        prefetch_factor: int = 8,
        features: list[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize Lightning DataModule.

        Args:
            zarr_path: Path to main trajectory Zarr store
            static_zarr_path: Path to static vessel data
            window_size: Length of input sequence
            stride: Stride between windows
            prediction_horizon: Length of prediction sequence
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            prefetch_factor: Number of batches to prefetch
            features: List of feature names to include
            normalize: Whether to apply normalization
        """
        super().__init__()
        self.save_hyperparameters()

        self.zarr_path = zarr_path
        self.static_zarr_path = static_zarr_path
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_workers = min(num_workers, OPTIMAL_WORKERS)  # Cap at optimal workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.features = features
        self.normalize = normalize

        self.scaler = None

    def setup(self, stage: str | None = None):
        """Setup datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            # Create training dataset
            self.train_dataset = AISWindowDataset(
                zarr_path=self.zarr_path,
                static_zarr_path=self.static_zarr_path,
                window_size=self.window_size,
                stride=self.stride,
                prediction_horizon=self.prediction_horizon,
                split="train",
                features=self.features,
                scaler=self.scaler,
            )

            # Create validation dataset
            self.val_dataset = AISWindowDataset(
                zarr_path=self.zarr_path,
                static_zarr_path=self.static_zarr_path,
                window_size=self.window_size,
                stride=self.stride,
                prediction_horizon=self.prediction_horizon,
                split="val",
                features=self.features,
                scaler=self.scaler,
            )

            # Compute normalization statistics from training data
            if self.normalize and self.scaler is None:
                self._compute_normalization_stats()

        if stage == "test" or stage is None:
            # Create test dataset
            self.test_dataset = AISWindowDataset(
                zarr_path=self.zarr_path,
                static_zarr_path=self.static_zarr_path,
                window_size=self.window_size,
                stride=self.stride,
                prediction_horizon=self.prediction_horizon,
                split="test",
                features=self.features,
                scaler=self.scaler,
            )

    def _compute_normalization_stats(self):
        """Compute normalization statistics from training data."""
        logger.info("Computing normalization statistics...")

        # Sample subset of training data for statistics
        sample_size = min(1000, len(self.train_dataset))
        sample_indices = np.random.choice(
            len(self.train_dataset), sample_size, replace=False
        )

        all_features = []
        for idx in sample_indices:
            sample = self.train_dataset[idx]
            all_features.append(sample["input"].numpy())

        # Compute mean and std across all samples
        stacked_features = np.stack(all_features, axis=0)  # [N, C, T]
        mean = np.mean(stacked_features, axis=(0, 2))  # [C]
        std = np.std(stacked_features, axis=(0, 2))  # [C]

        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

        self.scaler = (mean, std)

        # Update datasets with scaler
        self.train_dataset.scaler = self.scaler
        self.val_dataset.scaler = self.scaler

        logger.info(f"Computed normalization: mean={mean}, std={std}")

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
        )
