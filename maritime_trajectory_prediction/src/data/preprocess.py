"""
Data preprocessing script following AisXarrayPtlGuideline recommendations.

Implements the complete pipeline from raw AIS logs to analysis-ready
xarray datasets with Zarr storage backend.
"""
import logging
import json
import re
import orjson
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..data.xarray_processor import AISDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global file paths (following guideline recommendation)
RAW_DATA_DIR = Path("./data/raw")
PROCESSED_DATA_DIR = Path("./data/processed")
ZARR_PATH = PROCESSED_DATA_DIR / "ais_positions.zarr"
STATIC_ZARR_PATH = PROCESSED_DATA_DIR / "ais_static.zarr"
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet_chunks"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PARQUET_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class AISLogParser:
    """
    AIS log parser following the guideline's parsing strategy.
    
    Implements fast JSON parsing, sentinel value handling,
    and ITU-R M.1371 validation.
    """
    
    def __init__(self):
        """Initialize the AIS log parser."""
        # Regex pattern for log line validation
        self.pattern = re.compile(r"^\d{4}-\d{2}-\d{2} .* - ")
        
        # ITU-R M.1371 sentinel values
        self.sentinel_values = {
            'lat': 91.0,
            'lon': 181.0,
            'speed': 102.3,
            'heading': 511,
            'draught': 25.5
        }
        
        # CF-compliant field mappings
        self.field_mapping = {
            'lat': 'latitude',
            'lon': 'longitude',
            'speed': 'sog',
            'course': 'cog'
        }
        
        # Message type classifications
        self.msg_classes = {
            1: "A_pos", 2: "A_pos", 3: "A_pos",
            4: "Base_pos", 5: "Static",
            18: "B_pos", 21: "ATON", 24: "StaticB"
        }
    
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single AIS log line following guideline strategy.
        
        Args:
            line: Raw log line string
            
        Returns:
            Parsed and normalized record or None if invalid
        """
        # Fast reject non-matching lines
        if not self.pattern.match(line):
            return None
        
        # Split timestamp and payload
        try:
            _, payload = line.split(" - ", 1)
        except ValueError:
            return None
        
        # Skip non-JSON lines (engine chatter)
        if not payload.startswith("{"):
            return None
        
        # Parse JSON using orjson (≈2× faster than json)
        try:
            rec = orjson.loads(payload)
        except (orjson.JSONDecodeError, ValueError):
            return None
        
        # Validate required fields
        if not isinstance(rec, dict) or 'type' not in rec:
            return None
        
        # Parse timestamp
        if 'rxtime' in rec:
            try:
                rec['time'] = pd.to_datetime(
                    rec['rxtime'], 
                    format="%Y%m%d%H%M%S", 
                    utc=True
                )
            except (ValueError, TypeError):
                return None
        else:
            return None
        
        # Apply CF-compliant field mappings
        for old_key, new_key in self.field_mapping.items():
            if old_key in rec:
                rec[new_key] = rec.pop(old_key)
        
        # Handle sentinel values → NaN
        for field, sentinel in self.sentinel_values.items():
            if field in rec and rec[field] == sentinel:
                rec[field] = np.nan
        
        # Additional range validation
        self._validate_ranges(rec)
        
        # Classify message type
        msg_type = rec.get('type', 0)
        rec['msg_class'] = self.msg_classes.get(msg_type, "Other")
        
        return rec
    
    def _validate_ranges(self, rec: Dict) -> None:
        """Validate and clean field ranges."""
        # Latitude validation
        if 'latitude' in rec and rec['latitude'] is not None:
            if not (-90 <= rec['latitude'] <= 90):
                rec['latitude'] = np.nan
        
        # Longitude validation
        if 'longitude' in rec and rec['longitude'] is not None:
            if not (-180 <= rec['longitude'] <= 180):
                rec['longitude'] = np.nan
        
        # Speed validation
        if 'sog' in rec and rec['sog'] is not None:
            if rec['sog'] < 0 or rec['sog'] >= 102.2:
                rec['sog'] = np.nan
        
        # MMSI validation
        if 'mmsi' in rec and rec['mmsi'] is not None:
            if not (100_000_000 <= rec['mmsi'] <= 799_999_999):
                return None  # Invalid MMSI, reject record
    
    def parse_file(self, file_path: Path) -> List[Dict]:
        """
        Parse an entire AIS log file.
        
        Args:
            file_path: Path to AIS log file
            
        Returns:
            List of parsed records
        """
        records = []
        
        logger.info(f"Parsing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = self.parse_line(line)
                    if record:
                        records.append(record)
                    
                    # Progress logging
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines, {len(records)} valid records")
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
        
        logger.info(f"Completed parsing: {len(records)} valid records from {file_path}")
        return records


def process_raw_logs_to_parquet(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PARQUET_DIR
) -> List[Path]:
    """
    Process raw AIS logs to intermediate Parquet files.
    
    Args:
        input_dir: Directory containing raw log files
        output_dir: Directory for output Parquet files
        
    Returns:
        List of created Parquet file paths
    """
    parser = AISLogParser()
    parquet_files = []
    
    # Find all log files
    log_files = list(input_dir.glob("*.log")) + list(input_dir.glob("*.log.gz"))
    
    if not log_files:
        logger.warning(f"No log files found in {input_dir}")
        return []
    
    for log_file in log_files:
        logger.info(f"Processing {log_file}")
        
        # Parse file
        records = parser.parse_file(log_file)
        
        if not records:
            logger.warning(f"No valid records in {log_file}")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Schema validation
        try:
            assert df['latitude'].between(-90, 90, na=True).all()
            assert df['longitude'].between(-180, 180, na=True).all()
            assert df['mmsi'].between(100_000_000, 799_999_999, na=True).all()
        except AssertionError as e:
            logger.error(f"Schema validation failed for {log_file}: {e}")
            continue
        
        # Save to Parquet (day-level partitioning)
        if 'time' in df.columns:
            df['date'] = df['time'].dt.date
            
            for date, date_df in df.groupby('date'):
                parquet_path = output_dir / f"ais_{date}.parquet"
                
                # Append to existing file or create new
                if parquet_path.exists():
                    existing_df = pd.read_parquet(parquet_path)
                    combined_df = pd.concat([existing_df, date_df], ignore_index=True)
                    combined_df.to_parquet(parquet_path, index=False)
                else:
                    date_df.to_parquet(parquet_path, index=False)
                
                if parquet_path not in parquet_files:
                    parquet_files.append(parquet_path)
        
        logger.info(f"Saved {len(df)} records from {log_file}")
    
    logger.info(f"Created {len(parquet_files)} Parquet files")
    return parquet_files


def create_xarray_datasets(
    parquet_dir: Path = PARQUET_DIR,
    zarr_path: Path = ZARR_PATH,
    static_zarr_path: Path = STATIC_ZARR_PATH
) -> None:
    """
    Create xarray datasets from Parquet files and save to Zarr.
    
    Args:
        parquet_dir: Directory containing Parquet files
        zarr_path: Output path for position data Zarr store
        static_zarr_path: Output path for static data Zarr store
    """
    processor = AISDataProcessor(str(zarr_path))
    
    # Find all Parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"No Parquet files found in {parquet_dir}")
        return
    
    # Load and combine all Parquet files
    logger.info("Loading Parquet files...")
    dfs = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total records")
    
    # Separate position and static data
    position_df = combined_df[combined_df['msg_class'].str.contains('pos', na=False)]
    static_df = combined_df[combined_df['msg_class'] == 'Static']
    
    logger.info(f"Position records: {len(position_df)}")
    logger.info(f"Static records: {len(static_df)}")
    
    # Process position data to xarray
    if not position_df.empty:
        logger.info("Creating position dataset...")
        position_ds = processor.process_dataframe_to_xarray(position_df, 'time')
        
        # Save to Zarr
        processor.save_to_zarr(position_ds, mode='w')
        logger.info(f"Position dataset saved to {zarr_path}")
        
        # Print dataset statistics
        stats = processor.get_dataset_statistics(position_ds)
        logger.info(f"Dataset statistics: {stats}")
    
    # Process static data
    if not static_df.empty and len(static_df) > 0:
        logger.info("Creating static dataset...")
        
        # Keep latest record per MMSI
        static_df = static_df.sort_values('time').groupby('mmsi').last().reset_index()
        
        # Select relevant static fields
        static_fields = ['mmsi', 'shipname', 'shiptype', 'dim_bow', 'dim_stern', 
                        'dim_port', 'dim_starboard', 'callsign', 'destination']
        available_fields = [f for f in static_fields if f in static_df.columns]
        
        if available_fields:
            static_subset = static_df[available_fields].set_index('mmsi')
            static_ds = xr.Dataset.from_dataframe(static_subset)
            
            # Save static data to Zarr
            static_ds.to_zarr(static_zarr_path, mode='w', consolidated=True)
            logger.info(f"Static dataset saved to {static_zarr_path}")


def main():
    """Main preprocessing pipeline."""
    logger.info("Starting AIS data preprocessing pipeline")
    
    # Step 1: Parse raw logs to Parquet
    logger.info("Step 1: Parsing raw logs to Parquet...")
    parquet_files = process_raw_logs_to_parquet()
    
    if not parquet_files:
        logger.error("No Parquet files created. Check raw data directory.")
        return
    
    # Step 2: Create xarray datasets
    logger.info("Step 2: Creating xarray datasets...")
    create_xarray_datasets()
    
    logger.info("Preprocessing pipeline completed successfully!")
    logger.info(f"Position data: {ZARR_PATH}")
    logger.info(f"Static data: {STATIC_ZARR_PATH}")


if __name__ == "__main__":
    main()

