"""
Core data pipeline architecture for AIS maritime data processing.
"""
import abc
import logging
import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from datetime import datetime, timedelta

from .multi_task_processor import AISMultiTaskProcessor, MLTask

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    task: MLTask
    sequence_length: int = 10
    prediction_horizon: int = 5
    min_trajectory_length: int = 20
    spatial_bounds: Optional[Dict[str, float]] = None
    temporal_bounds: Optional[Dict[str, datetime]] = None
    vessel_types: Optional[List[int]] = None
    sampling_rate: str = '1min'
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class DatasetMetadata:
    """Metadata for generated datasets."""
    task: str
    config: DatasetConfig
    num_samples: int
    num_vessels: int
    temporal_range: Tuple[datetime, datetime]
    spatial_bounds: Dict[str, float]
    feature_columns: List[str]
    target_columns: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


class DatasetFormat(Enum):
    """Supported dataset output formats."""
    PANDAS = "pandas"
    XARRAY = "xarray"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    PARQUET = "parquet"
    ZARR = "zarr"


class BaseDatasetBuilder(abc.ABC):
    """Abstract base class for dataset builders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.metadata = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abc.abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for the specific task."""
        pass
    
    @abc.abstractmethod
    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build targets for the specific task."""
        pass
    
    @abc.abstractmethod
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets."""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data quality."""
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return False
        
        required_columns = ['mmsi', 'time', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        return True
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply task-specific data filtering."""
        filtered_df = df.copy()
        
        # Spatial filtering
        if self.config.spatial_bounds:
            bounds = self.config.spatial_bounds
            spatial_mask = (
                (filtered_df['latitude'] >= bounds.get('min_lat', -90)) &
                (filtered_df['latitude'] <= bounds.get('max_lat', 90)) &
                (filtered_df['longitude'] >= bounds.get('min_lon', -180)) &
                (filtered_df['longitude'] <= bounds.get('max_lon', 180))
            )
            filtered_df = filtered_df[spatial_mask]
            self.logger.info(f"Spatial filtering: {len(df)} -> {len(filtered_df)} records")
        
        # Temporal filtering
        if self.config.temporal_bounds:
            bounds = self.config.temporal_bounds
            temporal_mask = (
                (filtered_df['time'] >= bounds.get('start_time', pd.Timestamp.min)) &
                (filtered_df['time'] <= bounds.get('end_time', pd.Timestamp.max))
            )
            filtered_df = filtered_df[temporal_mask]
            self.logger.info(f"Temporal filtering: {len(df)} -> {len(filtered_df)} records")
        
        # Vessel type filtering
        if self.config.vessel_types and 'shiptype' in filtered_df.columns:
            type_mask = filtered_df['shiptype'].isin(self.config.vessel_types)
            filtered_df = filtered_df[type_mask]
            self.logger.info(f"Vessel type filtering: {len(df)} -> {len(filtered_df)} records")
        
        return filtered_df
    
    def resample_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample vessel trajectories to regular time intervals."""
        if df.empty:
            return df
        
        resampled_dfs = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            try:
                # Skip vessels with insufficient data
                if len(vessel_df) < self.config.min_trajectory_length:
                    self.logger.debug(f"Skipping vessel {mmsi}: only {len(vessel_df)} records")
                    continue
                
                # Sort by time
                vessel_df = vessel_df.sort_values('time')
                
                # Check time span
                time_span = vessel_df['time'].max() - vessel_df['time'].min()
                if time_span.total_seconds() < 300:  # Less than 5 minutes
                    self.logger.debug(f"Skipping vessel {mmsi}: time span too short ({time_span})")
                    continue
                
                # Set time as index for resampling
                vessel_df = vessel_df.set_index('time')
                
                # Resample to 1-minute intervals with forward fill
                resampled = vessel_df.resample('1min').first()
                
                # Forward fill missing values
                resampled = resampled.fillna(method='ffill')
                
                # Drop rows with remaining NaN values
                resampled = resampled.dropna()
                
                # Check if we still have enough data
                if len(resampled) >= self.config.min_trajectory_length:
                    resampled['mmsi'] = mmsi  # Restore mmsi column
                    resampled_dfs.append(resampled.reset_index())
                else:
                    self.logger.debug(f"Vessel {mmsi} dropped after resampling: {len(resampled)} records")
                    
            except Exception as e:
                self.logger.warning(f"Failed to resample vessel {mmsi}: {e}")
                continue
        
        if resampled_dfs:
            result = pd.concat(resampled_dfs, ignore_index=True)
            self.logger.info(f"Resampling: {df['mmsi'].nunique()} -> {result['mmsi'].nunique()} vessels")
            self.logger.info(f"Records: {len(df)} -> {len(result)}")
            return result
        else:
            self.logger.warning("No vessels remaining after resampling")
            return pd.DataFrame()
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split dataset into train/validation/test sets."""
        np.random.seed(self.config.random_seed)
        
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Calculate split indices
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        splits = {
            'train': (X[train_idx], y[train_idx]),
            'validation': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }
        
        self.logger.info(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return splits
    
    def build_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build complete dataset for the task."""
        self.logger.info(f"Building dataset for {self.config.task.value}")
        
        # Validate input data
        if not self.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Filter and preprocess data
        filtered_df = self.filter_data(df)
        if filtered_df.empty:
            raise ValueError("No data remaining after filtering")
        
        # Resample trajectories
        resampled_df = self.resample_trajectories(filtered_df)
        if resampled_df.empty:
            raise ValueError("No trajectories remaining after resampling")
        
        # Build features and targets
        features_df = self.build_features(resampled_df)
        targets_df = self.build_targets(resampled_df)
        
        # Create sequences
        X, y = self.create_sequences(features_df)
        
        if len(X) == 0:
            raise ValueError("No sequences created")
        
        # Split dataset
        splits = self.split_dataset(X, y)
        
        # Create metadata
        self.metadata = DatasetMetadata(
            task=self.config.task.value,
            config=self.config,
            num_samples=len(X),
            num_vessels=resampled_df['mmsi'].nunique(),
            temporal_range=(resampled_df['time'].min(), resampled_df['time'].max()),
            spatial_bounds={
                'min_lat': resampled_df['latitude'].min(),
                'max_lat': resampled_df['latitude'].max(),
                'min_lon': resampled_df['longitude'].min(),
                'max_lon': resampled_df['longitude'].max()
            },
            feature_columns=list(features_df.columns),
            target_columns=list(targets_df.columns) if hasattr(targets_df, 'columns') else ['target']
        )
        
        return {
            'splits': splits,
            'metadata': self.metadata,
            'features_df': features_df,
            'targets_df': targets_df,
            'raw_sequences': (X, y)
        }


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, processor: AISMultiTaskProcessor):
        self.processor = processor
        self.builders = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_builder(self, task: MLTask, builder_class: type):
        """Register a dataset builder for a specific task."""
        self.builders[task] = builder_class
        self.logger.info(f"Registered builder for {task.value}: {builder_class.__name__}")
    
    def process_raw_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Process raw AIS data file."""
        self.logger.info(f"Processing raw data: {file_path}")
        df = self.processor.process_file(file_path)
        self.logger.info(f"Processed {len(df)} records from {file_path}")
        return df
    
    def build_dataset(self, df: pd.DataFrame, task: MLTask, config: DatasetConfig) -> Dict[str, Any]:
        """Build dataset for specific task."""
        if task not in self.builders:
            raise ValueError(f"No builder registered for task: {task.value}")
        
        # Get task-specific data
        task_df = self.processor.get_task_specific_dataset(df, task)
        
        # Initialize builder and build dataset
        builder = self.builders[task](config)
        dataset = builder.build_dataset(task_df)
        
        return dataset
    
    def export_dataset(self, dataset: Dict[str, Any], output_dir: Path, 
                      formats: List[DatasetFormat] = None) -> Dict[str, Path]:
        """Export dataset in specified formats."""
        if formats is None:
            formats = [DatasetFormat.PARQUET, DatasetFormat.NUMPY]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        metadata = dataset['metadata']
        splits = dataset['splits']
        
        for format_type in formats:
            format_dir = output_dir / format_type.value
            format_dir.mkdir(exist_ok=True)
            
            if format_type == DatasetFormat.PARQUET:
                # Export as Parquet files
                for split_name, (X, y) in splits.items():
                    # Convert to DataFrame for Parquet export
                    features_df = pd.DataFrame(X.reshape(X.shape[0], -1))
                    targets_df = pd.DataFrame(y.reshape(y.shape[0], -1))
                    
                    features_file = format_dir / f"{split_name}_features.parquet"
                    targets_file = format_dir / f"{split_name}_targets.parquet"
                    
                    features_df.to_parquet(features_file)
                    targets_df.to_parquet(targets_file)
                    
                    exported_files[f"{format_type.value}_{split_name}_features"] = features_file
                    exported_files[f"{format_type.value}_{split_name}_targets"] = targets_file
            
            elif format_type == DatasetFormat.NUMPY:
                # Export as NumPy arrays
                for split_name, (X, y) in splits.items():
                    features_file = format_dir / f"{split_name}_features.npy"
                    targets_file = format_dir / f"{split_name}_targets.npy"
                    
                    np.save(features_file, X)
                    np.save(targets_file, y)
                    
                    exported_files[f"{format_type.value}_{split_name}_features"] = features_file
                    exported_files[f"{format_type.value}_{split_name}_targets"] = targets_file
            
            elif format_type == DatasetFormat.XARRAY:
                # Export as xarray Dataset
                for split_name, (X, y) in splits.items():
                    # Create xarray Dataset
                    ds = xr.Dataset({
                        'features': (['sample', 'time', 'feature'], X),
                        'targets': (['sample', 'time_target', 'target'], y)
                    })
                    
                    # Add metadata as attributes
                    ds.attrs.update({
                        'task': metadata.task,
                        'num_samples': metadata.num_samples,
                        'num_vessels': metadata.num_vessels,
                        'created_at': metadata.created_at.isoformat(),
                        'version': metadata.version
                    })
                    
                    dataset_file = format_dir / f"{split_name}_dataset.nc"
                    ds.to_netcdf(dataset_file)
                    
                    exported_files[f"{format_type.value}_{split_name}"] = dataset_file
        
        # Export metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            # Convert metadata to JSON-serializable format
            metadata_dict = {
                'task': metadata.task,
                'num_samples': metadata.num_samples,
                'num_vessels': metadata.num_vessels,
                'temporal_range': [
                    metadata.temporal_range[0].isoformat(),
                    metadata.temporal_range[1].isoformat()
                ],
                'spatial_bounds': metadata.spatial_bounds,
                'feature_columns': metadata.feature_columns,
                'target_columns': metadata.target_columns,
                'created_at': metadata.created_at.isoformat(),
                'version': metadata.version,
                'config': {
                    'task': metadata.config.task.value,
                    'sequence_length': metadata.config.sequence_length,
                    'prediction_horizon': metadata.config.prediction_horizon,
                    'min_trajectory_length': metadata.config.min_trajectory_length,
                    'sampling_rate': metadata.config.sampling_rate,
                    'validation_split': metadata.config.validation_split,
                    'test_split': metadata.config.test_split,
                    'random_seed': metadata.config.random_seed
                }
            }
            json.dump(metadata_dict, f, indent=2)
        
        exported_files['metadata'] = metadata_file
        
        self.logger.info(f"Exported dataset to {output_dir} in {len(formats)} formats")
        return exported_files
    
    def load_dataset(self, dataset_dir: Path, format_type: DatasetFormat = DatasetFormat.NUMPY) -> Dict[str, Any]:
        """Load previously exported dataset."""
        dataset_dir = Path(dataset_dir)
        
        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        # Load data based on format
        format_dir = dataset_dir / format_type.value
        if not format_dir.exists():
            raise FileNotFoundError(f"Format directory not found: {format_dir}")
        
        splits = {}
        
        if format_type == DatasetFormat.NUMPY:
            for split_name in ['train', 'validation', 'test']:
                features_file = format_dir / f"{split_name}_features.npy"
                targets_file = format_dir / f"{split_name}_targets.npy"
                
                if features_file.exists() and targets_file.exists():
                    X = np.load(features_file)
                    y = np.load(targets_file)
                    splits[split_name] = (X, y)
        
        elif format_type == DatasetFormat.PARQUET:
            for split_name in ['train', 'validation', 'test']:
                features_file = format_dir / f"{split_name}_features.parquet"
                targets_file = format_dir / f"{split_name}_targets.parquet"
                
                if features_file.exists() and targets_file.exists():
                    features_df = pd.read_parquet(features_file)
                    targets_df = pd.read_parquet(targets_file)
                    
                    # Convert back to original shape (this is a simplification)
                    X = features_df.values
                    y = targets_df.values
                    splits[split_name] = (X, y)
        
        elif format_type == DatasetFormat.XARRAY:
            for split_name in ['train', 'validation', 'test']:
                dataset_file = format_dir / f"{split_name}_dataset.nc"
                
                if dataset_file.exists():
                    ds = xr.open_dataset(dataset_file)
                    X = ds['features'].values
                    y = ds['targets'].values
                    splits[split_name] = (X, y)
                    ds.close()
        
        self.logger.info(f"Loaded dataset from {dataset_dir} in {format_type.value} format")
        
        return {
            'splits': splits,
            'metadata': metadata_dict
        }

