"""
Complete PyTorch Lightning DataModule for AIS maritime trajectory data.

Handles the full pipeline:
1. Raw AIS logs → Preprocessing → Zarr format (prepare_data)
2. Zarr → Sequences → Train/Val/Test splits (setup)
3. Sequences → DataLoaders (train_dataloader, val_dataloader, test_dataloader)

Features:
- Automatic preprocessing detection and execution
- Multi-level caching for efficiency
- Supports raw logs, CSV, Parquet, and Zarr formats
- Proper Lightning lifecycle (prepare_data, setup, dataloaders)
- Multi-GPU compatible
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .cache_manager import CacheManager
from .maritime_message_processor import AISProcessor
from .schema import FeatureGroups

# Calculate optimal number of workers (leave 4 cores free)
OPTIMAL_WORKERS = max(1, os.cpu_count() - 4)

logger = logging.getLogger(__name__)


def _process_vessel_sequences(args):
    """
    Process a single vessel to create sequences (module-level for multiprocessing).

    This function is defined at module level so it can be pickled by multiprocessing.Pool.
    Local/nested functions cannot be pickled, which was causing the original crash.

    Args:
        args: Tuple of (mmsi, vessel_features_array, seq_len, pred_horizon, target_feature_count)
              vessel_features_array is already a numpy array (pre-extracted to avoid serialization overhead)

    Returns:
        List of sequence dictionaries for this vessel
    """
    mmsi, vessel_features, seq_len, pred_horizon, target_feature_count = args

    # Check if vessel has enough data
    if len(vessel_features) < seq_len + pred_horizon:
        return []

    # Create sliding window sequences for this vessel
    vessel_sequences = []

    for i in range(len(vessel_features) - seq_len - pred_horizon + 1):
        # Input sequence (numpy array - no DataFrame overhead!)
        X_seq = vessel_features[i : i + seq_len]

        # Target sequence (future positions, basic features only)
        target_start = i + seq_len
        target_end = target_start + pred_horizon
        y_seq = vessel_features[target_start:target_end, :target_feature_count]

        # Store as numpy arrays - 100x+ faster than DataFrame creation
        # DataFrames will be created on-demand in __getitem__ only when needed
        vessel_sequences.append(
            {
                "input_sequence": X_seq,  # numpy array
                "target_sequence": y_seq,  # numpy array
                "mmsi": int(mmsi),
                "segment_id": -1,  # Will be assigned globally after merging
            }
        )

    return vessel_sequences


class AISTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for AIS trajectory sequences.

    Expects sequences with numpy float32 arrays for optimal ML performance.
    Arrays are converted to tensors using zero-copy torch.from_numpy() in __getitem__.

    Note: Use AISDataModule to create datasets - it handles format normalization
    automatically (including migration of old DataFrame caches to numpy format).
    """

    def __init__(self, sequences: list[dict]):
        """
        Initialize AIS trajectory dataset.

        Args:
            sequences: List of trajectory sequences, each containing:
                - input_sequence: numpy float32 array (seq_len, features)
                - target_sequence: numpy float32 array (pred_horizon, features)
                - mmsi: Vessel identifier
                - segment_id: Trajectory segment identifier
        """
        self.sequences = sequences

        # Store metadata from first sequence
        if sequences:
            first_input = sequences[0]["input_sequence"]
            first_target = sequences[0]["target_sequence"]

            # Validate format (fail fast with clear error message)
            if not isinstance(first_input, np.ndarray):
                raise TypeError(
                    f"Expected numpy arrays, got {type(first_input)}. "
                    "Use AISDataModule which handles format normalization."
                )

            # Simple dimension extraction (no type checking!)
            self.input_dim = first_input.shape[1]
            self.output_dim = first_target.shape[1]
            self.sequence_length = first_input.shape[0]
            self.prediction_horizon = first_target.shape[0]
        else:
            self.input_dim = 0
            self.output_dim = 0
            self.sequence_length = 0
            self.prediction_horizon = 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sequence (HOT PATH - optimized for performance).

        This method is called millions of times during training, so we:
        - Avoid all type checking (format ensured at construction)
        - Use zero-copy torch.from_numpy() instead of torch.tensor()
        - Skip dtype conversion (arrays already float32)

        Returns:
            dict: Batch item with:
                - input: torch.Tensor (seq_len, features) - zero-copy view of numpy array
                - target: torch.Tensor (pred_horizon, features) - zero-copy view
                - mmsi: int - vessel identifier
                - segment_id: int - trajectory segment identifier
        """
        sequence_data = self.sequences[idx]

        # Zero-copy conversion: torch.from_numpy() creates a tensor view
        # This is 3-5x faster than torch.tensor() which copies data
        return {
            "input": torch.from_numpy(sequence_data["input_sequence"]),
            "target": torch.from_numpy(sequence_data["target_sequence"]),
            "mmsi": sequence_data["mmsi"],
            "segment_id": sequence_data["segment_id"],
        }


class AISDataModule(pl.LightningDataModule):
    """
    Complete PyTorch Lightning DataModule for AIS trajectory prediction.

    Automatically handles:
    - Raw log preprocessing (if needed)
    - Data loading from multiple formats (raw logs, CSV, Parquet, Zarr)
    - Sequence generation with caching
    - Train/val/test splitting
    - DataLoader creation

    Example:
        ```python
        # Option 1: With raw log file (will auto-preprocess)
        dm = AISDataModule(
            data_path="data/raw/combined_aiscatcher.log",
            batch_size=32,
            sequence_length=20,
            prediction_horizon=5
        )

        # Option 2: With already processed Zarr
        dm = AISDataModule(
            data_path="data/processed/ais_positions.zarr",
            batch_size=32
        )

        dm.prepare_data()  # Preprocessing (if needed)
        dm.setup("fit")    # Load and create sequences
        train_loader = dm.train_dataloader()
        ```
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
        num_workers: int = OPTIMAL_WORKERS,
        pin_memory: bool = True,
        # Preprocessing options
        auto_preprocess: bool = True,
        processed_zarr_path: str | None = None,
        force_reprocess: bool = False,
    ):
        """
        Initialize AIS DataModule.

        Args:
            data_path: Path to AIS data (raw log, CSV, Parquet, or Zarr)
            batch_size: Batch size for training
            sequence_length: Length of input sequences
            prediction_horizon: Number of future points to predict
            features: List of feature names to use
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU transfer
            auto_preprocess: Automatically preprocess raw data if needed
            processed_zarr_path: Path to save/load processed Zarr (default: data/processed/ais_positions.zarr)
            force_reprocess: Force reprocessing even if processed data exists
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or FeatureGroups.BASIC_TRAJECTORY
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = min(num_workers, OPTIMAL_WORKERS)
        self.pin_memory = pin_memory

        # Preprocessing config
        self.auto_preprocess = auto_preprocess
        self.force_reprocess = force_reprocess
        self.processed_zarr_path = (
            Path(processed_zarr_path)
            if processed_zarr_path
            else Path("data/processed/ais_positions.zarr")
        )

        # Initialize processors
        self.processor = AISProcessor()
        self.cache_manager = CacheManager()

        # Runtime state
        self.input_dim = None

    def prepare_data(self):
        """
        Prepare data (preprocessing stage).

        Called only once and on a single process (rank 0 in multi-GPU).
        Handles raw log → processed format conversion if needed.
        """
        logger.info("=" * 70)
        logger.info("DataModule: prepare_data() stage")
        logger.info("=" * 70)
        logger.info(f"Input data path: {self.data_path}")
        logger.info(f"Processed data path: {self.processed_zarr_path}")

        # Check if data needs preprocessing
        if self._needs_preprocessing():
            if not self.auto_preprocess:
                raise ValueError(
                    f"Raw data detected at {self.data_path} but auto_preprocess=False. "
                    f"Set auto_preprocess=True or preprocess manually."
                )

            logger.info("⚙️  Starting preprocessing pipeline...")
            self._preprocess_raw_to_zarr()
            logger.info(f"✓ Preprocessing complete: {self.processed_zarr_path}")
        else:
            effective_path = self._get_effective_data_path()
            logger.info("✓ Using existing data (no preprocessing needed)")
            logger.info(f"  Data source: {effective_path}")

        logger.info("=" * 70)

    def _needs_preprocessing(self) -> bool:
        """Check if raw data needs preprocessing."""
        # If force reprocess, always preprocess
        if self.force_reprocess:
            logger.info("force_reprocess=True, will reprocess data")
            return True

        # Check for existing preprocessed data (both .zarr and .parquet)
        # Try zarr path first
        if self.processed_zarr_path.exists():
            size_mb = sum(
                f.stat().st_size
                for f in self.processed_zarr_path.rglob("*")
                if f.is_file()
            ) / (1024**2)
            logger.info(
                f"✓ Found existing processed Zarr at {self.processed_zarr_path} ({size_mb:.1f} MB)"
            )
            return False

        # Try parquet alternative (same name but .parquet extension)
        parquet_alternative = (
            self.processed_zarr_path.parent / f"{self.processed_zarr_path.stem}.parquet"
        )
        if parquet_alternative.exists():
            size_mb = parquet_alternative.stat().st_size / (1024**2)
            logger.info(
                f"✓ Found existing processed Parquet at {parquet_alternative} ({size_mb:.1f} MB)"
            )
            # Update the path to use existing parquet
            self.processed_zarr_path = parquet_alternative
            return False

        # Check what format the input data is
        if self.data_path.suffix == ".log" or str(self.data_path).endswith(".log.gz"):
            logger.info(f"Raw log file detected: {self.data_path.name}")
            logger.info("No existing preprocessed data found - preprocessing required")
            return True

        # If data_path is already Zarr, no preprocessing needed
        if self.data_path.suffix == ".zarr" or (
            self.data_path.is_dir() and ".zarr" in str(self.data_path)
        ):
            logger.info(f"Input is already Zarr format: {self.data_path}")
            return False

        # CSV/Parquet can be used directly, no preprocessing needed
        if self.data_path.suffix in [".csv", ".parquet"]:
            logger.info(
                f"Input is already {self.data_path.suffix} format: {self.data_path}"
            )
            return False

        return False

    def _preprocess_raw_to_zarr(self):
        """Preprocess raw AIS logs and build sequences (full preprocessing pipeline)."""
        import pickle

        from .preprocess import parse_ais_catcher_log

        logger.info(f"Step 1/3: Parsing raw log file: {self.data_path}")
        logger.info(f"File size: {self.data_path.stat().st_size / (1024**3):.2f} GB")

        # Parse the single log file directly (no multiprocessing nesting)
        try:
            position_df, static_df = parse_ais_catcher_log(str(self.data_path))

            if position_df.empty:
                raise RuntimeError(f"No valid AIS data parsed from {self.data_path}")

            logger.info(f"✓ Parsed {len(position_df)} position records")
            logger.info(f"✓ Parsed {len(static_df)} static records")

            # Ensure output directory exists
            self.processed_zarr_path.parent.mkdir(parents=True, exist_ok=True)

            # Step 2: Save raw parquet (for reference)
            logger.info("Step 2/3: Saving processed data...")
            parquet_path = (
                self.processed_zarr_path.parent
                / f"{self.processed_zarr_path.stem}.parquet"
            )
            position_df.to_parquet(parquet_path, index=False)
            logger.info(f"✓ Saved parquet at {parquet_path}")

            # Step 3: Build and save sequences (the expensive part)
            logger.info(
                "Step 3/3: Building sequences (this will take a few minutes)..."
            )
            sequences = self._build_sequences_from_df(position_df)

            # Save sequences to disk
            sequences_path = (
                self.processed_zarr_path.parent
                / f"{self.processed_zarr_path.stem}_sequences.pkl"
            )
            with open(sequences_path, "wb") as f:
                pickle.dump(sequences, f)

            logger.info(f"✓ Saved {len(sequences)} sequences to {sequences_path}")
            logger.info(f"  (Size: {sequences_path.stat().st_size / (1024**2):.1f} MB)")

            # Update the path to use parquet
            self.processed_zarr_path = parquet_path
            self._sequences_path = sequences_path

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to preprocess raw logs: {e}") from e

    def _get_effective_data_path(self) -> Path:
        """Get the effective data path to use for loading."""
        # Check if processed data exists (could be .zarr or .parquet)
        if self.processed_zarr_path.exists():
            return self.processed_zarr_path

        # Check for parquet alternative
        parquet_alternative = (
            self.processed_zarr_path.parent / f"{self.processed_zarr_path.stem}.parquet"
        )
        if parquet_alternative.exists():
            return parquet_alternative

        # Otherwise use the original data_path
        return self.data_path

    def setup(self, stage: str | None = None):
        """
        Setup datasets (data loading and sequence generation stage).

        Called on every process in distributed training.
        Loads data, generates sequences, and creates train/val/test splits.
        """
        logger.info("=" * 70)
        logger.info(f"DataModule: setup(stage={stage})")
        logger.info("=" * 70)

        effective_data_path = self._get_effective_data_path()
        logger.info(f"Loading from: {effective_data_path}")

        # Build sequences using multi-level caching
        sequences = self._build_sequences(effective_data_path)

        if not sequences:
            raise ValueError(f"No sequences generated from {effective_data_path}")

        logger.info(f"Generated {len(sequences)} sequences")

        # Store input dimension for model configuration
        # Sequences are guaranteed to be numpy arrays (from _ensure_numpy_format)
        if sequences:
            first_seq = sequences[0]["input_sequence"]
            first_target = sequences[0]["target_sequence"]

            # Simple dimension extraction (no type checking needed!)
            self.input_dim = first_seq.shape[1]
            target_dim = first_target.shape[1]

            logger.info(
                f"Input dimension: {self.input_dim} (includes derived features)"
            )
            logger.info(f"Target dimension: {target_dim}")

        # Split data (temporal split preferred for time series)
        n_sequences = len(sequences)
        n_train = int(n_sequences * self.train_split)
        n_val = int(n_sequences * self.val_split)

        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train : n_train + n_val]
        test_sequences = sequences[n_train + n_val :]

        logger.info(
            f"Split: train={len(train_sequences)}, val={len(val_sequences)}, test={len(test_sequences)}"
        )

        # Create datasets (sequences already have properly formatted DataFrames)
        if stage == "fit" or stage is None:
            self.train_dataset = AISTrajectoryDataset(train_sequences)
            self.val_dataset = AISTrajectoryDataset(val_sequences)
            logger.info(f"✓ Created train dataset: {len(self.train_dataset)} samples")
            logger.info(f"✓ Created val dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            self.test_dataset = AISTrajectoryDataset(test_sequences)
            logger.info(f"✓ Created test dataset: {len(self.test_dataset)} samples")

        logger.info("=" * 70)

    def _build_sequences(self, data_path: Path):
        """Load pre-built sequences from disk (already built during preprocessing)."""
        import pickle

        # Check for pre-built sequences (saved during preprocessing)
        sequences_path = data_path.parent / f"{data_path.stem}_sequences.pkl"

        if sequences_path.exists():
            logger.info(f"✓ Found pre-built sequences at {sequences_path}")
            logger.info(
                f"  Loading sequences (size: {sequences_path.stat().st_size / (1024**2):.1f} MB)..."
            )
            with open(sequences_path, "rb") as f:
                sequences = pickle.load(f)  # nosec B301 - loading our own cached data, not untrusted input
            logger.info(f"✓ Loaded {len(sequences)} pre-built sequences")
        else:
            # If no pre-built sequences, build them now (fallback for CSV/Parquet input)
            logger.warning(
                "No pre-built sequences found - building from scratch (this will be slow)..."
            )
            logger.warning(
                "Consider running preprocessing separately to save time on future runs"
            )

            # Load processed data
            processed_data = self._load_processed_data(data_path)

            # Build sequences
            sequences = self._build_sequences_from_df(processed_data)

            # Save for future use
            logger.info(f"Saving sequences to {sequences_path} for future runs...")
            with open(sequences_path, "wb") as f:
                pickle.dump(sequences, f)

            logger.info(f"✓ Built and saved {len(sequences)} sequences")

        # Ensure sequences are in optimal numpy format (migrate old DataFrame caches)
        sequences = self._ensure_numpy_format(sequences)

        return sequences

    def _ensure_numpy_format(self, sequences: list[dict]) -> list[dict]:
        """
        Ensure all sequences use numpy float32 arrays for optimal ML performance.

        This method provides backward compatibility with cached DataFrame sequences
        while maintaining clean, type-homogeneous code paths throughout the rest
        of the pipeline.

        Args:
            sequences: List of sequence dicts (may contain DataFrames or numpy arrays)

        Returns:
            List of sequence dicts with numpy float32 arrays
        """
        if not sequences:
            return sequences

        # Check format of first sequence
        first_input = sequences[0]["input_sequence"]

        # Already numpy arrays - check dtype
        if isinstance(first_input, np.ndarray):
            if first_input.dtype == np.float32:
                logger.info("✓ Sequences already in optimal numpy float32 format")
                return sequences

            # Fix dtype only
            logger.info(f"Normalizing sequence dtype: {first_input.dtype} → float32...")
            return self._normalize_dtype(sequences)

        # Migrate from DataFrame format (old cached sequences)
        logger.info("Migrating cached sequences: DataFrame → numpy float32 array...")
        import time

        start_time = time.time()
        migrated = self._migrate_dataframe_sequences(sequences)
        migration_time = time.time() - start_time

        logger.info(
            f"✓ Migrated {len(migrated)} sequences to numpy format in {migration_time:.1f}s"
        )
        logger.info("  Future loads will be faster with numpy format")

        return migrated

    def _normalize_dtype(self, sequences: list[dict]) -> list[dict]:
        """Ensure all numpy arrays use float32 dtype (optimal for ML)."""
        normalized = []

        for seq in sequences:
            input_seq = seq["input_sequence"]
            target_seq = seq["target_sequence"]

            # Convert to float32 if needed
            if input_seq.dtype != np.float32:
                input_seq = input_seq.astype(np.float32)
            if target_seq.dtype != np.float32:
                target_seq = target_seq.astype(np.float32)

            normalized.append(
                {
                    "input_sequence": input_seq,
                    "target_sequence": target_seq,
                    "mmsi": seq["mmsi"],
                    "segment_id": seq["segment_id"],
                }
            )

        return normalized

    def _migrate_dataframe_sequences(self, sequences: list[dict]) -> list[dict]:
        """
        Migrate sequences from DataFrame format to numpy arrays.

        This handles old cached sequences that used DataFrames.
        New sequences are created directly as numpy arrays for better performance.
        """
        migrated = []

        for seq in sequences:
            input_seq = seq["input_sequence"]
            target_seq = seq["target_sequence"]

            # Convert DataFrames to numpy arrays
            if isinstance(input_seq, pd.DataFrame):
                input_seq = input_seq.values.astype(np.float32)
            elif isinstance(input_seq, np.ndarray):
                input_seq = input_seq.astype(np.float32)
            else:
                raise TypeError(
                    f"Expected DataFrame or numpy array for input_sequence, got {type(input_seq)}"
                )

            if isinstance(target_seq, pd.DataFrame):
                target_seq = target_seq.values.astype(np.float32)
            elif isinstance(target_seq, np.ndarray):
                target_seq = target_seq.astype(np.float32)
            else:
                raise TypeError(
                    f"Expected DataFrame or numpy array for target_sequence, got {type(target_seq)}"
                )

            migrated.append(
                {
                    "input_sequence": input_seq,
                    "target_sequence": target_seq,
                    "mmsi": seq["mmsi"],
                    "segment_id": seq["segment_id"],
                }
            )

        return migrated

    def _build_sequences_from_df(self, df: pd.DataFrame):
        """Build sequences from DataFrame (used during preprocessing)."""
        return self._create_sequences_from_data(df)

    def _load_processed_data(self, data_path: Path):
        """Load processed data from various formats."""
        logger.info(f"Loading data from {data_path}...")

        import time

        start_time = time.time()

        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        elif ".zarr" in str(data_path):
            # Load from Zarr using xarray
            import xarray as xr

            ds = xr.open_zarr(data_path)
            # Convert to pandas
            df = ds.to_dataframe().reset_index()
        else:
            # For raw logs - check cache or process
            from .maritime_message_processor import load_ais_data

            raw_data = load_ais_data(str(data_path))
            df = self.processor.clean_ais_data(raw_data)

        load_time = time.time() - start_time
        logger.info(f"✓ Loaded {len(df):,} records in {load_time:.1f}s")

        # Ensure timestamp column is datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["timestamp"]
        ):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _create_sequences_from_data(self, processed_data: pd.DataFrame):
        """Create sequences from processed data."""
        import time

        from .dataset_builders import TrajectoryPredictionBuilder
        from .pipeline import DatasetConfig, MLTask
        from .schema import FeatureGroups

        logger.info(f"Building sequences from {len(processed_data):,} records...")

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

        # Step 1: Build features (adds derived features)
        logger.info(
            "Step 1/3: Building derived features (temporal, movement, spatial)..."
        )
        start_time = time.time()
        builder = TrajectoryPredictionBuilder(config)
        features_df = builder.build_features(processed_data)
        feature_time = time.time() - start_time
        logger.info(
            f"✓ Built features in {feature_time:.1f}s ({len(features_df.columns)} columns)"
        )

        # Step 2: Create sequences (WITH MMSI tracking, parallel processing)
        logger.info(
            f"Step 2/3: Creating sequences (seq_len={self.sequence_length}, pred_horizon={self.prediction_horizon})..."
        )
        start_time = time.time()

        # Get actual feature columns from the builder (includes derived features)
        feature_cols = [
            col for col in features_df.columns if col not in ["mmsi", "time"]
        ]

        # Targets use only basic trajectory features (as per builder design)
        target_features = FeatureGroups.BASIC_TRAJECTORY

        # Update self.features to include all input features (for proper input_dim)
        self.input_features = feature_cols
        self.target_features = target_features

        # Process vessels in parallel
        from multiprocessing import Pool, cpu_count

        # Optimal: 8-32 workers (more causes context switching overhead)
        # Even on large machines, excessive workers hurt performance
        num_workers = min(32, max(8, cpu_count() // 4))
        logger.info(f"Using {num_workers} workers for parallel sequence generation")

        # Pre-extract numpy arrays to avoid serializing DataFrames (major speedup!)
        # Sort each vessel's data by time and extract features as numpy array
        logger.info("Pre-processing vessel data for parallel execution...")
        vessel_args = []
        target_feature_count = len(target_features)

        for mmsi, vessel_df in features_df.groupby("mmsi"):
            # Sort by time and extract numpy array ONCE
            sorted_vessel_df = vessel_df.sort_values("time").reset_index(drop=True)
            vessel_features = sorted_vessel_df[feature_cols].values

            vessel_args.append(
                (
                    int(mmsi),
                    vessel_features,  # numpy array - fast serialization!
                    self.sequence_length,
                    self.prediction_horizon,
                    target_feature_count,
                )
            )

        logger.info(f"✓ Prepared {len(vessel_args)} vessels for processing")

        # Process in parallel using module-level function (picklable)
        from tqdm import tqdm

        logger.info(
            f"Processing {len(vessel_args)} vessels with {num_workers} workers..."
        )
        with Pool(processes=num_workers) as pool:
            # Use imap for progress tracking instead of map
            results = list(
                tqdm(
                    pool.imap(_process_vessel_sequences, vessel_args),
                    total=len(vessel_args),
                    desc="Generating sequences",
                    unit="vessel",
                    ncols=100,
                )
            )

        # Merge results and assign global sequence IDs
        sequences = []
        sequence_id = 0
        for vessel_sequences in results:
            for seq in vessel_sequences:
                seq["segment_id"] = sequence_id
                sequences.append(seq)
                sequence_id += 1

        sequence_time = time.time() - start_time
        logger.info(f"✓ Created {len(sequences):,} sequences in {sequence_time:.1f}s")

        if len(sequences) == 0:
            logger.warning(
                "No sequences generated - check data quality and length requirements"
            )

        return sequences

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
