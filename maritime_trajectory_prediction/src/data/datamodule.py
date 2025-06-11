"""
AIS Data Module for PyTorch Lightning training.

This module provides data loading and preprocessing functionality
for AIS trajectory prediction models.
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .maritime_message_processor import AISProcessor
from .schema import FeatureGroups


class AISDataset(Dataset):
    """
    Dataset class for AIS trajectory data.
    """

    def __init__(
        self,
        sequences: list[dict],
        features: list[str] = None,
        sequence_length: int = 20,
        prediction_horizon: int = 5,
    ):
        """
        Initialize AIS dataset.

        Args:
            sequences: List of trajectory sequences
            features: List of feature names to use
            sequence_length: Length of input sequences
            prediction_horizon: Number of future points to predict
        """
        self.sequences = sequences
        self.features = features or FeatureGroups.BASIC_TRAJECTORY
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]

        input_seq = sequence_data["input_sequence"]
        target_seq = sequence_data["target_sequence"]

        # Extract features
        input_features = input_seq[self.features].values.astype(np.float32)
        target_features = target_seq[self.features].values.astype(np.float32)

        return {
            "input": torch.tensor(input_features),
            "target": torch.tensor(target_features),
            "mmsi": sequence_data["mmsi"],
            "segment_id": sequence_data["segment_id"],
        }


class AISDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AIS trajectory prediction.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        sequence_length: int = 20,
        prediction_horizon: int = 5,
        features: list[str] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        num_workers: int = 4,
    ):
        """
        Initialize AIS DataModule.

        Args:
            data_path: Path to AIS data file
            batch_size: Batch size for training
            sequence_length: Length of input sequences
            prediction_horizon: Number of future points to predict
            features: List of feature names to use
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or FeatureGroups.BASIC_TRAJECTORY
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        self.processor = AISProcessor()

    def setup(self, stage: str | None = None):
        """
        Setup datasets for training, validation, and testing.
        """
        # Load data
        if self.data_path.endswith(".parquet"):
            processed_data = pd.read_parquet(self.data_path)
        elif self.data_path.endswith(".csv"):
            processed_data = pd.read_csv(self.data_path)
        else:
            # For raw log files
            from .maritime_message_processor import load_ais_data

            raw_data = load_ais_data(self.data_path)
            processed_data = self.processor.clean_ais_data(raw_data)

        # Extract sequences using trajectory builder
        from .dataset_builders import TrajectoryPredictionBuilder
        from .pipeline import DatasetConfig, MLTask

        config = DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
        )

        # Rename timestamp to time for compatibility
        if (
            "timestamp" in processed_data.columns
            and "time" not in processed_data.columns
        ):
            processed_data = processed_data.rename(columns={"timestamp": "time"})

        builder = TrajectoryPredictionBuilder(config)
        features_df = builder.build_features(processed_data)
        X, y = builder.create_sequences(features_df)

        # Get the actual feature columns used by the builder (exclude mmsi, time)
        actual_features = [
            col for col in features_df.columns if col not in ["mmsi", "time"]
        ]
        target_features = FeatureGroups.BASIC_TRAJECTORY  # y has 4 features

        # Convert to list of sequence dictionaries for the dataset
        sequences = []
        for i in range(len(X)):
            sequences.append(
                {
                    "input_sequence": pd.DataFrame(X[i], columns=actual_features),
                    "target_sequence": pd.DataFrame(y[i], columns=target_features),
                    "mmsi": i,  # Placeholder
                    "segment_id": i,
                }
            )

        # Split data
        n_sequences = len(sequences)
        n_train = int(n_sequences * self.train_split)
        n_val = int(n_sequences * self.val_split)

        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train : n_train + n_val]
        test_sequences = sequences[n_train + n_val :]

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = AISDataset(
                train_sequences,
                features=self.features,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
            )
            self.val_dataset = AISDataset(
                val_sequences,
                features=self.features,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
            )

        if stage == "test" or stage is None:
            self.test_dataset = AISDataset(
                test_sequences,
                features=self.features,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
