"""
Data validation, quality checks, and export utilities for AIS datasets.
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataValidator:
    """Comprehensive data validation for AIS datasets."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_dataset(self, df: pd.DataFrame, task: str = None) -> ValidationResult:
        """Perform comprehensive dataset validation."""
        errors = []
        warnings = []
        metrics = {}
        
        # Basic structure validation
        structure_result = self._validate_structure(df)
        errors.extend(structure_result['errors'])
        warnings.extend(structure_result['warnings'])
        metrics.update(structure_result['metrics'])
        
        # Data quality validation
        quality_result = self._validate_data_quality(df)
        errors.extend(quality_result['errors'])
        warnings.extend(quality_result['warnings'])
        metrics.update(quality_result['metrics'])
        
        # Maritime-specific validation
        maritime_result = self._validate_maritime_data(df)
        errors.extend(maritime_result['errors'])
        warnings.extend(maritime_result['warnings'])
        metrics.update(maritime_result['metrics'])
        
        # Task-specific validation
        if task:
            task_result = self._validate_task_specific(df, task)
            errors.extend(task_result['errors'])
            warnings.extend(task_result['warnings'])
            metrics.update(task_result['metrics'])
        
        # Temporal validation
        temporal_result = self._validate_temporal_data(df)
        errors.extend(temporal_result['errors'])
        warnings.extend(temporal_result['warnings'])
        metrics.update(temporal_result['metrics'])
        
        # Spatial validation
        spatial_result = self._validate_spatial_data(df)
        errors.extend(spatial_result['errors'])
        warnings.extend(spatial_result['warnings'])
        metrics.update(spatial_result['metrics'])
        
        is_valid = len(errors) == 0 or not self.strict_mode
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic dataset structure."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("Dataset is empty")
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check required columns
        required_columns = ['mmsi', 'time', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
            warnings.append("Time column is not datetime type")
        
        # Record metrics
        metrics.update({
            'total_records': len(df),
            'total_columns': len(df.columns),
            'unique_vessels': df['mmsi'].nunique() if 'mmsi' in df.columns else 0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check for null values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        # Handle empty DataFrame case
        if len(df) == 0 or len(df.columns) == 0:
            null_percentage = 0
            duplicate_percentage = 0
        else:
            null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
            
            # Check for duplicate records
            if 'mmsi' in df.columns and 'time' in df.columns:
                duplicates = df.duplicated(subset=['mmsi', 'time']).sum()
                duplicate_percentage = (duplicates / len(df)) * 100
            else:
                duplicate_percentage = 0
        
        if null_percentage > 50:
            errors.append(f"High null percentage: {null_percentage:.1f}%")
        elif null_percentage > 20:
            warnings.append(f"Moderate null percentage: {null_percentage:.1f}%")
        
        if duplicate_percentage > 10:
            warnings.append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
        
        # Check data ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['latitude', 'longitude', 'sog', 'cog']:
                continue  # These are checked in maritime validation
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                q1, q99 = col_data.quantile([0.01, 0.99])
                outliers = ((col_data < q1) | (col_data > q99)).sum()
                outlier_percentage = (outliers / len(col_data)) * 100
                
                if outlier_percentage > 5:
                    warnings.append(f"High outlier percentage in {col}: {outlier_percentage:.1f}%")
        
        metrics.update({
            'null_percentage': null_percentage,
            'duplicate_percentage': duplicate_percentage,
            'null_counts_by_column': null_counts.to_dict()
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_maritime_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate maritime-specific data constraints."""
        errors = []
        warnings = []
        metrics = {}
        
        # Validate MMSI
        if 'mmsi' in df.columns:
            invalid_mmsi = df['mmsi'].apply(lambda x: not self._is_valid_mmsi(x)).sum()
            if invalid_mmsi > 0:
                warnings.append(f"Invalid MMSI values: {invalid_mmsi}")
        
        # Validate coordinates
        if 'latitude' in df.columns:
            invalid_lat = ((df['latitude'] < -90) | (df['latitude'] > 90)).sum()
            if invalid_lat > 0:
                errors.append(f"Invalid latitude values: {invalid_lat}")
        
        if 'longitude' in df.columns:
            invalid_lon = ((df['longitude'] < -180) | (df['longitude'] > 180)).sum()
            if invalid_lon > 0:
                errors.append(f"Invalid longitude values: {invalid_lon}")
        
        # Validate speed over ground
        if 'sog' in df.columns:
            invalid_sog = ((df['sog'] < 0) | (df['sog'] > 102.2)).sum()  # Max AIS speed
            if invalid_sog > 0:
                warnings.append(f"Invalid SOG values: {invalid_sog}")
        
        # Validate course over ground
        if 'cog' in df.columns:
            invalid_cog = ((df['cog'] < 0) | (df['cog'] >= 360)).sum()
            if invalid_cog > 0:
                warnings.append(f"Invalid COG values: {invalid_cog}")
        
        # Validate heading
        if 'heading' in df.columns:
            invalid_heading = ((df['heading'] < 0) | (df['heading'] >= 360)).sum()
            if invalid_heading > 0:
                warnings.append(f"Invalid heading values: {invalid_heading}")
        
        # Validate turn rate
        if 'turn' in df.columns:
            invalid_turn = ((df['turn'] < -127) | (df['turn'] > 127)).sum()
            if invalid_turn > 0:
                warnings.append(f"Invalid turn rate values: {invalid_turn}")
        
        metrics.update({
            'invalid_mmsi': invalid_mmsi if 'mmsi' in df.columns else 0,
            'invalid_coordinates': (invalid_lat if 'latitude' in df.columns else 0) + 
                                 (invalid_lon if 'longitude' in df.columns else 0),
            'invalid_movement': (invalid_sog if 'sog' in df.columns else 0) + 
                              (invalid_cog if 'cog' in df.columns else 0)
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_task_specific(self, df: pd.DataFrame, task: str) -> Dict[str, Any]:
        """Validate task-specific requirements."""
        errors = []
        warnings = []
        metrics = {}
        
        if task == 'trajectory_prediction':
            # Need sufficient trajectory length
            if 'mmsi' in df.columns:
                trajectory_lengths = df.groupby('mmsi').size()
                short_trajectories = (trajectory_lengths < 10).sum()
                if short_trajectories > trajectory_lengths.count() * 0.5:
                    warnings.append(f"Many short trajectories: {short_trajectories}")
        
        elif task == 'anomaly_detection':
            # Need vessel characteristics
            if 'shiptype' not in df.columns:
                warnings.append("Missing vessel type information for anomaly detection")
        
        elif task == 'graph_neural_networks':
            # Need multiple vessels in time windows
            if 'mmsi' in df.columns and 'time' in df.columns:
                time_groups = df.groupby(pd.Grouper(key='time', freq='10min'))['mmsi'].nunique()
                sparse_windows = (time_groups < 2).sum()
                if sparse_windows > len(time_groups) * 0.5:
                    warnings.append(f"Many sparse time windows for graph construction: {sparse_windows}")
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_temporal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal data consistency."""
        errors = []
        warnings = []
        metrics = {}
        
        if 'time' not in df.columns:
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check time ordering
        if not df['time'].is_monotonic_increasing:
            warnings.append("Time series is not monotonically increasing")
        
        # Check for time gaps
        if 'mmsi' in df.columns:
            large_gaps = 0
            for mmsi, vessel_df in df.groupby('mmsi'):
                if len(vessel_df) > 1:
                    time_diffs = vessel_df['time'].diff().dt.total_seconds()
                    gaps = (time_diffs > 3600).sum()  # Gaps > 1 hour
                    large_gaps += gaps
            
            if large_gaps > 0:
                warnings.append(f"Large time gaps detected: {large_gaps}")
        
        # Check temporal range
        time_range = df['time'].max() - df['time'].min()
        metrics.update({
            'temporal_range_hours': time_range.total_seconds() / 3600,
            'time_gaps': large_gaps if 'mmsi' in df.columns else 0
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _validate_spatial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate spatial data consistency."""
        errors = []
        warnings = []
        metrics = {}
        
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
        
        # Check spatial bounds
        lat_range = df['latitude'].max() - df['latitude'].min()
        lon_range = df['longitude'].max() - df['longitude'].min()
        
        # Check for unrealistic position jumps
        if 'mmsi' in df.columns:
            large_jumps = 0
            for mmsi, vessel_df in df.groupby('mmsi'):
                if len(vessel_df) > 1:
                    # Calculate distances between consecutive points
                    lats = vessel_df['latitude'].values
                    lons = vessel_df['longitude'].values
                    
                    for i in range(1, len(lats)):
                        # Haversine distance approximation
                        dlat = np.radians(lats[i] - lats[i-1])
                        dlon = np.radians(lons[i] - lons[i-1])
                        a = (np.sin(dlat/2)**2 + 
                             np.cos(np.radians(lats[i-1])) * np.cos(np.radians(lats[i])) * 
                             np.sin(dlon/2)**2)
                        distance = 2 * 6371 * np.arcsin(np.sqrt(a))  # km
                        
                        if distance > 100:  # 100 km jump
                            large_jumps += 1
            
            if large_jumps > 0:
                warnings.append(f"Large position jumps detected: {large_jumps}")
        
        metrics.update({
            'spatial_extent_lat': lat_range,
            'spatial_extent_lon': lon_range,
            'position_jumps': large_jumps if 'mmsi' in df.columns else 0
        })
        
        return {'errors': errors, 'warnings': warnings, 'metrics': metrics}
    
    def _is_valid_mmsi(self, mmsi: Any) -> bool:
        """Check if MMSI is valid."""
        if pd.isna(mmsi):
            return False
        
        try:
            mmsi_int = int(mmsi)
            # MMSI should be 9 digits
            return 100000000 <= mmsi_int <= 999999999
        except (ValueError, TypeError):
            return False


class QualityChecker:
    """Advanced quality checking for AIS datasets."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_trajectory_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check trajectory-specific quality metrics."""
        quality_metrics = {}
        
        if 'mmsi' not in df.columns:
            return quality_metrics
        
        trajectory_stats = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('time')
            
            stats = {
                'mmsi': mmsi,
                'length': len(vessel_df),
                'duration_hours': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'course_changes': 0,
                'position_accuracy': 0
            }
            
            if len(vessel_df) > 1:
                # Duration
                duration = vessel_df['time'].iloc[-1] - vessel_df['time'].iloc[0]
                stats['duration_hours'] = duration.total_seconds() / 3600
                
                # Speed statistics
                if 'sog' in vessel_df.columns:
                    stats['avg_speed'] = vessel_df['sog'].mean()
                    stats['max_speed'] = vessel_df['sog'].max()
                
                # Course changes
                if 'cog' in vessel_df.columns:
                    course_diffs = vessel_df['cog'].diff().abs()
                    stats['course_changes'] = (course_diffs > 10).sum()
                
                # Position accuracy
                if 'accuracy' in vessel_df.columns:
                    stats['position_accuracy'] = vessel_df['accuracy'].mean()
            
            trajectory_stats.append(stats)
        
        trajectory_df = pd.DataFrame(trajectory_stats)
        
        quality_metrics.update({
            'num_trajectories': len(trajectory_df),
            'avg_trajectory_length': trajectory_df['length'].mean(),
            'avg_trajectory_duration': trajectory_df['duration_hours'].mean(),
            'short_trajectories': (trajectory_df['length'] < 10).sum(),
            'long_trajectories': (trajectory_df['length'] > 1000).sum(),
            'high_speed_trajectories': (trajectory_df['max_speed'] > 30).sum()
        })
        
        return quality_metrics
    
    def check_coverage_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check spatial and temporal coverage quality."""
        coverage_metrics = {}
        
        if not all(col in df.columns for col in ['latitude', 'longitude', 'time']):
            return coverage_metrics
        
        # Spatial coverage
        lat_range = df['latitude'].max() - df['latitude'].min()
        lon_range = df['longitude'].max() - df['longitude'].min()
        
        # Temporal coverage
        time_range = df['time'].max() - df['time'].min()
        
        # Coverage density
        if 'mmsi' in df.columns:
            # Vessels per square degree
            area = lat_range * lon_range
            vessels_per_area = df['mmsi'].nunique() / max(area, 0.01)
            
            # Records per hour
            records_per_hour = len(df) / max(time_range.total_seconds() / 3600, 1)
        else:
            vessels_per_area = 0
            records_per_hour = 0
        
        coverage_metrics.update({
            'spatial_extent_deg2': lat_range * lon_range,
            'temporal_extent_hours': time_range.total_seconds() / 3600,
            'vessels_per_deg2': vessels_per_area,
            'records_per_hour': records_per_hour
        })
        
        return coverage_metrics


class DatasetExporter:
    """Export datasets in various formats with metadata."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def export_to_parquet(self, dataset: Dict[str, Any], name: str) -> Dict[str, Path]:
        """Export dataset to Parquet format."""
        parquet_dir = self.output_dir / "parquet" / name
        parquet_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        splits = dataset['splits']
        
        for split_name, (X, y) in splits.items():
            # Convert arrays to DataFrames
            feature_df = pd.DataFrame(
                X.reshape(X.shape[0], -1),
                columns=[f'feature_{i}' for i in range(X.shape[1] * X.shape[2])]
            )
            target_df = pd.DataFrame(
                y.reshape(y.shape[0], -1),
                columns=[f'target_{i}' for i in range(y.shape[1] if len(y.shape) > 1 else 1)]
            )
            
            # Export files
            feature_file = parquet_dir / f"{split_name}_features.parquet"
            target_file = parquet_dir / f"{split_name}_targets.parquet"
            
            feature_df.to_parquet(feature_file, compression='snappy')
            target_df.to_parquet(target_file, compression='snappy')
            
            exported_files[f"{split_name}_features"] = feature_file
            exported_files[f"{split_name}_targets"] = target_file
        
        # Export metadata
        metadata_file = parquet_dir / "metadata.json"
        self._export_metadata(dataset['metadata'], metadata_file)
        exported_files['metadata'] = metadata_file
        
        self.logger.info(f"Exported dataset to Parquet: {parquet_dir}")
        return exported_files
    
    def export_to_zarr(self, dataset: Dict[str, Any], name: str) -> Dict[str, Path]:
        """Export dataset to Zarr format."""
        zarr_dir = self.output_dir / "zarr" / name
        zarr_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        splits = dataset['splits']
        metadata = dataset['metadata']
        
        for split_name, (X, y) in splits.items():
            # Create xarray Dataset
            ds = xr.Dataset({
                'features': (['sample', 'time', 'feature'], X),
                'targets': (['sample', 'target_time', 'target'], y)
            })
            
            # Add metadata as attributes
            ds.attrs.update({
                'task': metadata.task,
                'split': split_name,
                'num_samples': len(X),
                'created_at': metadata.created_at.isoformat()
            })
            
            # Export to Zarr
            zarr_file = zarr_dir / f"{split_name}.zarr"
            ds.to_zarr(zarr_file, mode='w')
            exported_files[split_name] = zarr_file
        
        # Export metadata
        metadata_file = zarr_dir / "metadata.json"
        self._export_metadata(metadata, metadata_file)
        exported_files['metadata'] = metadata_file
        
        self.logger.info(f"Exported dataset to Zarr: {zarr_dir}")
        return exported_files
    
    def export_to_hdf5(self, dataset: Dict[str, Any], name: str) -> Dict[str, Path]:
        """Export dataset to HDF5 format."""
        hdf5_dir = self.output_dir / "hdf5" / name
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        splits = dataset['splits']
        
        for split_name, (X, y) in splits.items():
            hdf5_file = hdf5_dir / f"{split_name}.h5"
            
            with pd.HDFStore(hdf5_file, mode='w') as store:
                # Store features
                feature_df = pd.DataFrame(
                    X.reshape(X.shape[0], -1),
                    columns=[f'feature_{i}' for i in range(X.shape[1] * X.shape[2])]
                )
                store.put('features', feature_df, format='table')
                
                # Store targets
                target_df = pd.DataFrame(
                    y.reshape(y.shape[0], -1),
                    columns=[f'target_{i}' for i in range(y.shape[1] if len(y.shape) > 1 else 1)]
                )
                store.put('targets', target_df, format='table')
                
                # Store metadata
                metadata_df = pd.DataFrame([{
                    'task': dataset['metadata'].task,
                    'num_samples': len(X),
                    'created_at': dataset['metadata'].created_at.isoformat()
                }])
                store.put('metadata', metadata_df, format='table')
            
            exported_files[split_name] = hdf5_file
        
        self.logger.info(f"Exported dataset to HDF5: {hdf5_dir}")
        return exported_files
    
    def _export_metadata(self, metadata, file_path: Path):
        """Export metadata to JSON file."""
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
            'version': metadata.version
        }
        
        with open(file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

