"""
Enhanced AIS data processor with xarray and Zarr backend.

This module implements the recommendations from the AisXarrayPtlGuideline
for robust data processing, storage, and validation.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import numcodecs
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from ..utils.maritime_utils import MaritimeUtils

logger = logging.getLogger(__name__)


class AISDataProcessor:
    """
    Enhanced AIS data processor with xarray/Zarr backend.
    
    Implements industrial-scale data processing patterns following
    the AisXarrayPtlGuideline recommendations.
    """
    
    # ITU-R M.1371 sentinel values
    SENTINEL_VALUES = {
        'latitude': 91.0,
        'longitude': 181.0,
        'sog': 102.3,  # knots
        'heading': 511,
        'draught': 25.5,  # meters
        'nav_status': 15
    }
    
    # CF-compliant field mappings
    FIELD_MAPPING = {
        'lat': 'latitude',
        'lon': 'longitude', 
        'speed': 'sog',
        'course': 'cog'
    }
    
    def __init__(
        self,
        zarr_path: Optional[str] = None,
        chunk_size: Dict[str, int] = None,
        compression_level: int = 5
    ):
        """
        Initialize the enhanced AIS data processor.
        
        Args:
            zarr_path: Path to Zarr store for data persistence
            chunk_size: Chunking strategy for xarray datasets
            compression_level: Zstd compression level (1-22)
        """
        self.zarr_path = Path(zarr_path) if zarr_path else Path("./ais_data.zarr")
        self.chunk_size = chunk_size or {"time": 1440, "mmsi": 1}  # 1-day × 1-vessel
        self.compression_level = compression_level
        
        # Configure compressor following guideline
        self.compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=compression_level,
            shuffle=numcodecs.Blosc.BITSHUFFLE
        )
        
        self.utils = MaritimeUtils()
        
    def parse_ais_message(self, message: Dict) -> Optional[Dict]:
        """
        Parse and normalize a single AIS message.
        
        Implements the guideline's parsing strategy with sentinel handling
        and CF-compliant field renaming.
        
        Args:
            message: Raw AIS message dictionary
            
        Returns:
            Normalized message or None if invalid
        """
        if not isinstance(message, dict) or 'type' not in message:
            return None
            
        # Apply field mappings
        normalized = message.copy()
        for old_key, new_key in self.FIELD_MAPPING.items():
            if old_key in normalized:
                normalized[new_key] = normalized.pop(old_key)
        
        # Handle sentinel values → NaN
        for field, sentinel in self.SENTINEL_VALUES.items():
            if field in normalized and normalized[field] == sentinel:
                normalized[field] = np.nan
                
        # Classify message type
        msg_type = normalized.get('type', 0)
        normalized['msg_class'] = {
            1: "A_pos", 2: "A_pos", 3: "A_pos",
            4: "Base_pos", 5: "Static",
            18: "B_pos", 21: "ATON", 24: "StaticB"
        }.get(msg_type, "Other")
        
        # Validate critical fields
        if not self._validate_message(normalized):
            return None
            
        return normalized
    
    def _validate_message(self, message: Dict) -> bool:
        """
        Validate AIS message against ITU-R M.1371 specifications.
        
        Args:
            message: Normalized AIS message
            
        Returns:
            True if message is valid
        """
        # MMSI validation (100M - 799M range)
        mmsi = message.get('mmsi', 0)
        if not (100_000_000 <= mmsi <= 799_999_999):
            return False
            
        # Position validation
        lat = message.get('latitude')
        lon = message.get('longitude')
        if lat is not None and not (-90 <= lat <= 90):
            return False
        if lon is not None and not (-180 <= lon <= 180):
            return False
            
        # Speed validation
        sog = message.get('sog')
        if sog is not None and not (0 <= sog <= 102.2):
            return False
            
        return True
    
    def process_dataframe_to_xarray(
        self, 
        df: pd.DataFrame,
        time_column: str = 'timestamp'
    ) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset with proper chunking.
        
        Args:
            df: Input DataFrame with AIS data
            time_column: Name of timestamp column
            
        Returns:
            xarray Dataset with optimized chunking
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], utc=True)
        
        # Filter position messages only
        position_df = df[df['msg_class'].str.contains('pos', na=False)].copy()
        
        if position_df.empty:
            logger.warning("No position messages found in DataFrame")
            return xr.Dataset()
        
        # Set multi-index for xarray conversion
        position_df = position_df.set_index([time_column, 'mmsi'])
        
        # Convert to xarray and unstack
        ds = xr.Dataset.from_dataframe(position_df)
        ds = ds.unstack(time_column)
        
        # Apply chunking strategy
        ds = ds.chunk(self.chunk_size)
        
        # Add metadata
        ds.attrs.update({
            'title': 'AIS Trajectory Data',
            'source': 'Maritime Trajectory Prediction System',
            'created': datetime.utcnow().isoformat(),
            'conventions': 'CF-1.8',
            'processing_level': 'L2'
        })
        
        return ds
    
    def save_to_zarr(
        self, 
        dataset: xr.Dataset,
        mode: str = 'w',
        consolidated: bool = True
    ) -> None:
        """
        Save xarray Dataset to Zarr store with optimized encoding.
        
        Args:
            dataset: xarray Dataset to save
            mode: Write mode ('w', 'a', 'r+')
            consolidated: Whether to consolidate metadata
        """
        # Define encoding for all variables
        encoding = {}
        for var in dataset.data_vars:
            encoding[var] = {
                'compressor': self.compressor,
                'dtype': 'float32'  # Reduce precision for storage efficiency
            }
        
        # Save to Zarr
        dataset.to_zarr(
            self.zarr_path,
            mode=mode,
            consolidated=consolidated,
            encoding=encoding
        )
        
        logger.info(f"Dataset saved to {self.zarr_path}")
    
    def load_from_zarr(self) -> xr.Dataset:
        """
        Load xarray Dataset from Zarr store.
        
        Returns:
            Loaded xarray Dataset
        """
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.zarr_path}")
        
        ds = xr.open_zarr(self.zarr_path, chunks=self.chunk_size)
        logger.info(f"Dataset loaded from {self.zarr_path}")
        return ds
    
    def create_temporal_split(
        self,
        dataset: xr.Dataset,
        train_end: str,
        val_days: int = 7,
        test_days: int = 7
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """
        Create temporal train/validation/test splits.
        
        Args:
            dataset: Input xarray Dataset
            train_end: End date for training data (ISO format)
            val_days: Number of days for validation
            test_days: Number of days for test
            
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        train_end_dt = pd.Timestamp(train_end, tz='UTC')
        val_end_dt = train_end_dt + pd.Timedelta(days=val_days)
        
        # Create splits
        train_ds = dataset.sel(time=slice(None, train_end_dt - pd.Timedelta('1s')))
        val_ds = dataset.sel(time=slice(train_end_dt, val_end_dt))
        test_ds = dataset.sel(time=slice(val_end_dt, None))
        
        logger.info(f"Created temporal splits: train={len(train_ds.time)}, "
                   f"val={len(val_ds.time)}, test={len(test_ds.time)}")
        
        return train_ds, val_ds, test_ds
    
    def get_dataset_statistics(self, dataset: xr.Dataset) -> Dict:
        """
        Compute comprehensive dataset statistics.
        
        Args:
            dataset: xarray Dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_vessels': len(dataset.mmsi),
            'n_timestamps': len(dataset.time),
            'time_range': {
                'start': str(dataset.time.min().values),
                'end': str(dataset.time.max().values)
            },
            'spatial_bounds': {
                'lat_min': float(dataset.latitude.min().values),
                'lat_max': float(dataset.latitude.max().values),
                'lon_min': float(dataset.longitude.min().values),
                'lon_max': float(dataset.longitude.max().values)
            },
            'data_completeness': {}
        }
        
        # Compute completeness for each variable
        for var in dataset.data_vars:
            total_cells = dataset[var].size
            valid_cells = dataset[var].count().values
            stats['data_completeness'][var] = float(valid_cells / total_cells)
        
        return stats
    
    def optimize_chunks(
        self,
        dataset: xr.Dataset,
        target_chunk_size_mb: int = 128
    ) -> Dict[str, int]:
        """
        Optimize chunk sizes based on data characteristics.
        
        Args:
            dataset: xarray Dataset
            target_chunk_size_mb: Target chunk size in MB
            
        Returns:
            Optimized chunk sizes
        """
        # Estimate memory usage per element
        bytes_per_element = 4  # float32
        target_elements = (target_chunk_size_mb * 1024 * 1024) // bytes_per_element
        
        # Calculate optimal chunks
        time_chunk = min(1440, len(dataset.time))  # Max 1 day
        mmsi_chunk = max(1, target_elements // time_chunk)
        mmsi_chunk = min(mmsi_chunk, len(dataset.mmsi))
        
        optimized_chunks = {
            'time': time_chunk,
            'mmsi': mmsi_chunk
        }
        
        logger.info(f"Optimized chunks: {optimized_chunks}")
        return optimized_chunks

