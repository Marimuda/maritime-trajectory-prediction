"""
Time-aware cross-validation utilities for maritime trajectory prediction.
"""

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validator with purging to prevent data leakage.

    Ensures gap between train and test sets to avoid look-ahead bias
    in maritime trajectory prediction.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: int | None = None,
        max_train_size: int | None = None,
    ):
        """
        Initialize purged time series cross-validator.

        Args:
            n_splits: Number of splits
            gap: Number of samples to exclude between train and test
            test_size: Size of test set (if None, computed automatically)
            max_train_size: Maximum size of training set
        """
        self.n_splits = n_splits
        self.gap = gap  # Gap between train and test
        self.test_size = test_size
        self.max_train_size = max_train_size

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits with purging."""
        n_samples = len(X)

        if self.test_size:
            test_size = self.test_size
        else:
            # Automatically determine test size
            test_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Calculate split points
            test_end = n_samples - fold * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            # Ensure valid splits
            if train_end <= train_start or test_end <= test_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class VesselGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Combined vessel-based and time-based splitting.

    Ensures no vessel appears in both train and test while
    maintaining temporal ordering.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        vessel_split_ratio: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize vessel-aware time series cross-validator.

        Args:
            n_splits: Number of splits
            gap: Gap between train and test
            vessel_split_ratio: Fraction of vessels for training
            random_state: Random seed for vessel selection
        """
        self.n_splits = n_splits
        self.gap = gap
        self.vessel_split_ratio = vessel_split_ratio
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate splits ensuring vessel and temporal separation."""
        if groups is None:
            raise ValueError("Vessel groups required for splitting")

        # Set random seed
        np.random.seed(self.random_state)

        unique_vessels = np.unique(groups)
        n_vessels = len(unique_vessels)

        # Time-based splitting first
        time_splitter = PurgedTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)

        for train_time_idx, test_time_idx in time_splitter.split(X):
            # Then apply vessel-based filtering
            train_vessels = np.random.choice(
                unique_vessels,
                size=int(n_vessels * self.vessel_split_ratio),
                replace=False,
            )

            # Combine time and vessel constraints
            train_mask = np.isin(groups[train_time_idx], train_vessels)
            test_mask = ~np.isin(groups[test_time_idx], train_vessels)

            train_indices = train_time_idx[train_mask]
            test_indices = test_time_idx[test_mask]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class BlockingTimeSeriesSplit(BaseCrossValidator):
    """
    Time series split that ensures contiguous blocks.

    Useful for trajectory data where maintaining sequence
    continuity is important.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        train_size: int = None,
        gap: int = 0,
    ):
        """
        Initialize blocking time series cross-validator.

        Args:
            n_splits: Number of splits
            test_size: Size of each test block
            train_size: Size of each train block
            gap: Gap between blocks
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate blocked train/test splits."""
        n_samples = len(X)

        if self.test_size is None:
            self.test_size = n_samples // (2 * self.n_splits)

        if self.train_size is None:
            self.train_size = n_samples - self.test_size * self.n_splits

        indices = np.arange(n_samples)

        # Create blocks
        for i in range(self.n_splits):
            # Test block
            test_start = i * (self.test_size + self.gap)
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            # Train block (everything before test with gap)
            train_end = test_start - self.gap
            train_start = max(0, train_end - self.train_size) if self.train_size else 0

            if train_end <= train_start:
                continue

            yield indices[train_start:train_end], indices[test_start:test_end]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits
