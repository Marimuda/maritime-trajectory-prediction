"""Cross-validation protocols for maritime trajectory data."""

from collections.abc import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold as SKGroupKFold
from sklearn.model_selection import TimeSeriesSplit as SKTimeSeriesSplit


class TimeSeriesSplit:
    """Time series cross-validation with maritime-specific considerations."""

    def __init__(
        self, n_splits: int = 5, min_gap_minutes: int = 60, test_size: int | None = None
    ):
        """
        Initialize time series cross-validation.

        Args:
            n_splits: Number of cross-validation folds
            min_gap_minutes: Minimum gap between train/test to avoid leakage
            test_size: Size of test set (if None, uses equal splits)
        """
        self.n_splits = n_splits
        self.min_gap_minutes = min_gap_minutes
        self.test_size = test_size
        self.sklearn_splitter = SKTimeSeriesSplit(
            n_splits=n_splits, test_size=test_size
        )

    def split(
        self, df: pd.DataFrame, time_col: str = "timestamp"
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate time-series splits with gap enforcement.

        Args:
            df: DataFrame with time series data
            time_col: Name of timestamp column

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in DataFrame")

        # Sort by time to ensure chronological order
        df_sorted = df.sort_values(time_col).reset_index(drop=True)

        # Use sklearn TimeSeriesSplit as base
        for train_idx, test_idx in self.sklearn_splitter.split(df_sorted):
            # Enforce minimum gap between train and test
            if self.min_gap_minutes > 0:
                train_end_time = df_sorted.iloc[train_idx[-1]][time_col]
                test_start_time = df_sorted.iloc[test_idx[0]][time_col]

                # Calculate gap in minutes
                if hasattr(train_end_time, "total_seconds"):  # timedelta
                    gap_minutes = (
                        test_start_time - train_end_time
                    ).total_seconds() / 60
                else:  # assume unix timestamp
                    gap_minutes = (test_start_time - train_end_time) / 60

                # Filter test set to respect minimum gap
                if gap_minutes < self.min_gap_minutes:
                    min_test_time = train_end_time + pd.Timedelta(
                        minutes=self.min_gap_minutes
                    )
                    valid_test_mask = (
                        df_sorted.iloc[test_idx][time_col] >= min_test_time
                    )
                    test_idx = test_idx[valid_test_mask.values]

            if len(test_idx) > 0:  # Only yield if test set is not empty
                yield train_idx, test_idx


class GroupKFold:
    """Group K-Fold cross-validation for maritime vessels."""

    def __init__(
        self, n_splits: int = 5, shuffle: bool = True, random_state: int | None = None
    ):
        """
        Initialize group-based cross-validation.

        Args:
            n_splits: Number of cross-validation folds
            shuffle: Whether to shuffle groups before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.sklearn_splitter = SKGroupKFold(n_splits=n_splits)

    def split(
        self, df: pd.DataFrame, group_col: str = "mmsi"
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate group-based splits ensuring no group appears in both train and test.

        Args:
            df: DataFrame with grouped data
            group_col: Name of group column (e.g., 'mmsi' for vessels)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in DataFrame")

        groups = df[group_col].values
        X_dummy = np.zeros((len(df), 1))  # Dummy features for sklearn compatibility

        for train_idx, test_idx in self.sklearn_splitter.split(X_dummy, groups=groups):
            yield train_idx, test_idx

    def get_group_distribution(
        self, df: pd.DataFrame, group_col: str = "mmsi"
    ) -> pd.DataFrame:
        """Get distribution of groups across folds."""
        groups = df[group_col].unique()
        fold_assignments = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(df, group_col)):
            test_groups = set(df.iloc[test_idx][group_col].unique())
            for group in groups:
                if group in test_groups:
                    fold_assignments.append(
                        {"group": group, "fold": fold_idx, "split": "test"}
                    )
                else:
                    fold_assignments.append(
                        {"group": group, "fold": fold_idx, "split": "train"}
                    )

        return pd.DataFrame(fold_assignments)


def maritime_cv_split(
    df: pd.DataFrame,
    split_type: str = "temporal",
    n_splits: int = 5,
    time_col: str = "timestamp",
    group_col: str = "mmsi",
    min_gap_minutes: int = 60,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Maritime-specific cross-validation splitting.

    Args:
        df: DataFrame with maritime trajectory data
        split_type: 'temporal', 'vessel', or 'combined'
        n_splits: Number of cross-validation folds
        time_col: Name of timestamp column
        group_col: Name of vessel identifier column
        min_gap_minutes: Minimum gap for temporal splits
        random_state: Random seed

    Returns:
        List of (train_indices, test_indices) tuples

    Example:
        >>> splits = maritime_cv_split(df, split_type='vessel', n_splits=5)
        >>> for fold, (train_idx, test_idx) in enumerate(splits):
        ...     print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")
    """
    if split_type == "temporal":
        splitter = TimeSeriesSplit(n_splits=n_splits, min_gap_minutes=min_gap_minutes)
        return list(splitter.split(df, time_col))

    elif split_type == "vessel":
        splitter = GroupKFold(n_splits=n_splits, random_state=random_state)
        return list(splitter.split(df, group_col))

    elif split_type == "combined":
        # Nested CV: outer vessel-based, inner temporal
        vessel_splitter = GroupKFold(n_splits=n_splits, random_state=random_state)
        combined_splits = []

        for train_idx, test_idx in vessel_splitter.split(df, group_col):
            # Further split training data temporally
            train_df = df.iloc[train_idx]
            if len(train_df) > min(100, len(train_df) // 2):  # Only if sufficient data
                temporal_splitter = TimeSeriesSplit(
                    n_splits=2, min_gap_minutes=min_gap_minutes
                )
                temporal_splits = list(temporal_splitter.split(train_df, time_col))
                if temporal_splits:
                    # Use first temporal split for training
                    inner_train_idx, inner_val_idx = temporal_splits[0]
                    actual_train_idx = train_idx[inner_train_idx]
                    combined_splits.append((actual_train_idx, test_idx))
                else:
                    combined_splits.append((train_idx, test_idx))
            else:
                combined_splits.append((train_idx, test_idx))

        return combined_splits

    else:
        raise ValueError("split_type must be 'temporal', 'vessel', or 'combined'")


def validate_split_quality(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    time_col: str = "timestamp",
    group_col: str = "mmsi",
) -> dict:
    """
    Validate quality of cross-validation splits.

    Args:
        df: DataFrame with data
        splits: List of (train_idx, test_idx) tuples
        time_col: Timestamp column name
        group_col: Group column name

    Returns:
        Dictionary with validation metrics
    """
    metrics = {
        "n_splits": len(splits),
        "train_sizes": [],
        "test_sizes": [],
        "temporal_overlaps": 0,
        "group_overlaps": 0,
        "time_gaps_minutes": [],
    }

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        metrics["train_sizes"].append(len(train_idx))
        metrics["test_sizes"].append(len(test_idx))

        # Check group overlap
        train_groups = set(train_df[group_col].unique())
        test_groups = set(test_df[group_col].unique())
        if len(train_groups & test_groups) > 0:
            metrics["group_overlaps"] += 1

        # Check temporal overlap
        train_time_range = (train_df[time_col].min(), train_df[time_col].max())
        test_time_range = (test_df[time_col].min(), test_df[time_col].max())

        if (
            train_time_range[1] >= test_time_range[0]
            and train_time_range[0] <= test_time_range[1]
        ):
            metrics["temporal_overlaps"] += 1

        # Calculate time gap
        if train_time_range[1] < test_time_range[0]:
            if hasattr(train_time_range[1], "total_seconds"):
                gap_minutes = (
                    test_time_range[0] - train_time_range[1]
                ).total_seconds() / 60
            else:
                gap_minutes = (test_time_range[0] - train_time_range[1]) / 60
            metrics["time_gaps_minutes"].append(gap_minutes)

    # Summary statistics
    metrics["avg_train_size"] = np.mean(metrics["train_sizes"])
    metrics["avg_test_size"] = np.mean(metrics["test_sizes"])
    metrics["train_test_ratio"] = metrics["avg_train_size"] / metrics["avg_test_size"]

    if metrics["time_gaps_minutes"]:
        metrics["avg_time_gap_minutes"] = np.mean(metrics["time_gaps_minutes"])
        metrics["min_time_gap_minutes"] = np.min(metrics["time_gaps_minutes"])
    else:
        metrics["avg_time_gap_minutes"] = None
        metrics["min_time_gap_minutes"] = None

    return metrics
