"""
Data preprocessing script following the CLAUDE blueprint.

Consolidates AIS data processing logic from multiple scripts into a unified
CLI-callable module. Supports both AIS-catcher log parsing and xarray dataset
creation with Zarr storage backend.
"""

import json
import logging
import multiprocessing as mp
import os
import re
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from tqdm import tqdm

from ..data.xarray_processor import AISDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Calculate optimal number of workers (leave 4 cores free)
OPTIMAL_WORKERS = max(1, os.cpu_count() - 4)
logger.info(f"Using {OPTIMAL_WORKERS} worker processes (total cores: {os.cpu_count()})")

# Global file paths (following guideline recommendation)
RAW_DATA_DIR = Path("./data/raw")
PROCESSED_DATA_DIR = Path("./data/processed")
ZARR_PATH = PROCESSED_DATA_DIR / "ais_positions.zarr"
STATIC_ZARR_PATH = PROCESSED_DATA_DIR / "ais_static.zarr"
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet_chunks"

# Constants for validation
MIN_LATITUDE = -90
MAX_LATITUDE = 90
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180
MAX_SPEED_SOG = 102.2

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
            "lat": 91.0,
            "lon": 181.0,
            "speed": 102.3,
            "heading": 511,
            "draught": 25.5,
        }

        # CF-compliant field mappings
        self.field_mapping = {
            "lat": "latitude",
            "lon": "longitude",
            "speed": "sog",
            "course": "cog",
        }

        # Message type classifications
        self.msg_classes = {
            1: "A_pos",
            2: "A_pos",
            3: "A_pos",
            4: "Base_pos",
            5: "Static",
            18: "B_pos",
            21: "ATON",
            24: "StaticB",
        }

    def parse_line(self, line: str) -> dict | None:
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
        if not isinstance(rec, dict) or "type" not in rec:
            return None

        # Parse timestamp
        if "rxtime" in rec:
            try:
                rec["time"] = pd.to_datetime(
                    rec["rxtime"], format="%Y%m%d%H%M%S", utc=True
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
        rec = self._validate_ranges(rec)
        if rec is None:
            return None

        # Classify message type
        msg_type = rec.get("type", 0)
        rec["msg_class"] = self.msg_classes.get(msg_type, "Other")

        return rec

    def _validate_ranges(self, rec: dict) -> dict | None:
        """Validate and clean field ranges."""
        # Latitude validation
        if (
            "latitude" in rec
            and rec["latitude"] is not None
            and not (MIN_LATITUDE <= rec["latitude"] <= MAX_LATITUDE)
        ):
            rec["latitude"] = np.nan

        # Longitude validation
        if (
            "longitude" in rec
            and rec["longitude"] is not None
            and not (MIN_LONGITUDE <= rec["longitude"] <= MAX_LONGITUDE)
        ):
            rec["longitude"] = np.nan

        # Speed validation
        if (
            "sog" in rec
            and rec["sog"] is not None
            and (rec["sog"] < 0 or rec["sog"] >= MAX_SPEED_SOG)
        ):
            rec["sog"] = np.nan

        # MMSI validation
        if (
            "mmsi" in rec
            and rec["mmsi"] is not None
            and not (100_000_000 <= rec["mmsi"] <= 799_999_999)
        ):
            return None  # Invalid MMSI, reject record

        return rec

    def parse_file(self, file_path: Path) -> list[dict]:
        """
        Parse an entire AIS log file using multiprocessing.

        Args:
            file_path: Path to AIS log file

        Returns:
            List of parsed records
        """
        logger.info(f"Parsing file: {file_path}")

        try:
            # Read file into chunks for parallel processing
            chunk_size = 10000  # Lines per chunk
            chunks = []

            with open(file_path, encoding="utf-8") as f:
                chunk = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        chunk.append(line)

                    if len(chunk) >= chunk_size:
                        chunks.append(chunk)
                        chunk = []

                # Add remaining lines
                if chunk:
                    chunks.append(chunk)

            logger.info(f"Split file into {len(chunks)} chunks for parallel processing")

            # Process chunks in parallel
            with mp.Pool(processes=OPTIMAL_WORKERS) as pool:
                chunk_results = pool.map(self._parse_chunk, chunks)

            # Combine results
            records = []
            for chunk_records in chunk_results:
                records.extend(chunk_records)

            logger.info(
                f"Completed parsing: {len(records)} valid records from {file_path}"
            )
            return records

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []

    def _parse_chunk(self, lines: list[str]) -> list[dict]:
        """
        Parse a chunk of lines in parallel.

        Args:
            lines: List of log lines to parse

        Returns:
            List of parsed records from this chunk
        """
        records = []
        for line in lines:
            record = self.parse_line(line)
            if record:
                records.append(record)
        return records


def _process_single_log_file(args: tuple[Path, Path]) -> list[Path]:
    """
    Process a single log file to Parquet. Used for multiprocessing.

    Args:
        args: Tuple of (log_file_path, output_dir)

    Returns:
        List of created Parquet file paths
    """
    log_file, output_dir = args
    parser = AISLogParser()
    parquet_files = []

    logger.info(f"Processing {log_file}")

    # Parse file
    records = parser.parse_file(log_file)

    if not records:
        logger.warning(f"No valid records in {log_file}")
        return []

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Schema validation
    try:
        assert df["latitude"].between(-90, 90, na=True).all()
        assert df["longitude"].between(-180, 180, na=True).all()
        assert df["mmsi"].between(100_000_000, 799_999_999, na=True).all()
    except AssertionError as e:
        logger.error(f"Schema validation failed for {log_file}: {e}")
        return []

    # Save to Parquet (day-level partitioning)
    if "time" in df.columns:
        df["date"] = df["time"].dt.date

        for date, date_df in df.groupby("date"):
            parquet_path = output_dir / f"ais_{date}_{log_file.stem}.parquet"
            date_df.to_parquet(parquet_path, index=False)
            parquet_files.append(parquet_path)

    logger.info(f"Saved {len(df)} records from {log_file}")
    return parquet_files


def process_raw_logs_to_parquet(
    input_dir: Path = RAW_DATA_DIR, output_dir: Path = PARQUET_DIR
) -> list[Path]:
    """
    Process raw AIS logs to intermediate Parquet files using multiprocessing.

    Args:
        input_dir: Directory containing raw log files
        output_dir: Directory for output Parquet files

    Returns:
        List of created Parquet file paths
    """
    # Find all log files
    log_files = list(input_dir.glob("*.log")) + list(input_dir.glob("*.log.gz"))

    if not log_files:
        logger.warning(f"No log files found in {input_dir}")
        return []

    logger.info(
        f"Processing {len(log_files)} log files using {OPTIMAL_WORKERS} workers"
    )

    # Prepare arguments for parallel processing
    file_args = [(log_file, output_dir) for log_file in log_files]

    # Process files in parallel
    with mp.Pool(processes=OPTIMAL_WORKERS) as pool:
        results = pool.map(_process_single_log_file, file_args)

    # Combine results and merge files by date
    all_parquet_files = []
    for file_list in results:
        all_parquet_files.extend(file_list)

    # Merge files by date to avoid having multiple files per date
    logger.info("Merging daily Parquet files...")
    merged_files = _merge_daily_parquet_files(all_parquet_files, output_dir)

    logger.info(f"Created {len(merged_files)} merged Parquet files")
    return merged_files


def _merge_daily_parquet_files(
    parquet_files: list[Path], output_dir: Path
) -> list[Path]:
    """
    Merge multiple parquet files for the same date into single files.

    Args:
        parquet_files: List of parquet file paths to merge
        output_dir: Output directory for merged files

    Returns:
        List of merged parquet file paths
    """
    # Group files by date
    date_groups = {}
    for pf in parquet_files:
        # Extract date from filename (format: ais_YYYY-MM-DD_*.parquet)
        date_str = pf.name.split("_")[1]  # Get date part
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(pf)

    merged_files = []
    for date_str, files in date_groups.items():
        if len(files) == 1:
            # Single file, just rename it
            final_path = output_dir / f"ais_{date_str}.parquet"
            files[0].rename(final_path)
            merged_files.append(final_path)
        else:
            # Multiple files, merge them
            dfs = [pd.read_parquet(f) for f in files]
            combined_df = pd.concat(dfs, ignore_index=True)

            final_path = output_dir / f"ais_{date_str}.parquet"
            combined_df.to_parquet(final_path, index=False)
            merged_files.append(final_path)

            # Remove temporary files
            for f in files:
                f.unlink()

    return merged_files


def create_xarray_datasets(
    parquet_dir: Path = PARQUET_DIR,
    zarr_path: Path = ZARR_PATH,
    static_zarr_path: Path = STATIC_ZARR_PATH,
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
    position_df = combined_df[combined_df["msg_class"].str.contains("pos", na=False)]
    static_df = combined_df[combined_df["msg_class"] == "Static"]

    logger.info(f"Position records: {len(position_df)}")
    logger.info(f"Static records: {len(static_df)}")

    # Process position data to xarray
    if not position_df.empty:
        logger.info("Creating position dataset...")
        position_ds = processor.process_dataframe_to_xarray(position_df, "time")

        # Save to Zarr
        processor.save_to_zarr(position_ds, mode="w")
        logger.info(f"Position dataset saved to {zarr_path}")

        # Print dataset statistics
        stats = processor.get_dataset_statistics(position_ds)
        logger.info(f"Dataset statistics: {stats}")

    # Process static data
    if not static_df.empty and len(static_df) > 0:
        logger.info("Creating static dataset...")

        # Keep latest record per MMSI
        static_df = static_df.sort_values("time").groupby("mmsi").last().reset_index()

        # Select relevant static fields
        static_fields = [
            "mmsi",
            "shipname",
            "shiptype",
            "dim_bow",
            "dim_stern",
            "dim_port",
            "dim_starboard",
            "callsign",
            "destination",
        ]
        available_fields = [f for f in static_fields if f in static_df.columns]

        if available_fields:
            static_subset = static_df[available_fields].set_index("mmsi")
            static_ds = xr.Dataset.from_dataframe(static_subset)

            # Save static data to Zarr
            static_ds.to_zarr(static_zarr_path, mode="w", consolidated=True)
            logger.info(f"Static dataset saved to {static_zarr_path}")


# AIS-catcher processing functions (consolidated from scripts/process_ais_catcher.py)

EARTH_RADIUS_KM = 6371.0
MAX_SPEED_KNOTS = 50.0
MIN_SPEED_KNOTS = 0.5
POSITION_MESSAGE_TYPES = [1, 2, 3, 18, 19]
STATIC_MESSAGE_TYPES = [5, 24]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial bearing between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
    bearing = atan2(y, x)
    return (degrees(bearing) + 360) % 360


def is_valid_mmsi(mmsi) -> bool:
    """Check if MMSI is valid for a vessel."""
    if not mmsi:
        return False
    mmsi_str = str(mmsi)
    if len(mmsi_str) != 9:
        return False
    if mmsi_str.startswith("00") or mmsi_str.startswith("99"):
        return False
    return True


def is_valid_position(lat, lon) -> bool:
    """Check if position is valid."""
    if lat is None or lon is None:
        return False
    if abs(lat) > 90 or abs(lon) > 180:
        return False
    if lat == 91.0 or lon == 181.0:
        return False
    return True


def is_valid_speed(speed) -> bool:
    """Check if speed is valid."""
    if speed is None:
        return False
    if speed < 0 or speed > MAX_SPEED_KNOTS:
        return False
    if abs(speed - 102.3) < 0.1:
        return False
    return True


def is_valid_course(course) -> bool:
    """Check if course is valid."""
    if course is None:
        return False
    if course < 0 or course >= 360:
        return False
    return True


def parse_ais_catcher_log(
    log_file: str, max_records: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse AIS-catcher log file into DataFrames for position and static data.

    Args:
        log_file: Path to the AIS-catcher log file
        max_records: Maximum number of records to process (for testing)

    Returns:
        Tuple of (position_df, static_df) with parsed AIS data
    """
    position_records = []
    static_records = []

    error_counts = {
        "malformed_lines": 0,
        "invalid_mmsi": 0,
        "invalid_position": 0,
        "invalid_speed": 0,
        "invalid_course": 0,
    }
    message_type_counts = {}

    logger.info(f"Starting to parse AIS log file: {log_file}")

    logger.info(f"Processing file with multiprocessing using {OPTIMAL_WORKERS} workers")

    # Parse the file with progress bar
    with open(log_file) as f:
        if total_lines:
            pbar = tqdm(total=total_lines, desc="Parsing AIS data")
        else:
            pbar = tqdm(desc="Parsing AIS data")

        lines_processed = 0
        for line in f:
            try:
                # Extract JSON part (after timestamp)
                parts = line.strip().split(" - ", 1)
                if len(parts) != 2:
                    error_counts["malformed_lines"] += 1
                    continue

                timestamp_str = parts[0]
                json_str = parts[1]

                # Parse JSON
                data = json.loads(json_str)

                # Only process AIS messages
                if data.get("class") != "AIS":
                    continue

                # Get MMSI
                mmsi = data.get("mmsi")
                if not is_valid_mmsi(mmsi):
                    error_counts["invalid_mmsi"] += 1
                    continue

                # Process dynamic position messages
                msg_type = data.get("type")

                # Count message types for statistics
                if msg_type in message_type_counts:
                    message_type_counts[msg_type] += 1
                else:
                    message_type_counts[msg_type] = 1

                if msg_type in POSITION_MESSAGE_TYPES:
                    lat = data.get("lat")
                    lon = data.get("lon")

                    # Validate position
                    if not is_valid_position(lat, lon):
                        error_counts["invalid_position"] += 1
                        continue

                    # Extract dynamic data
                    speed = data.get("speed", 0)
                    course = data.get("course", 0)
                    heading = data.get("heading")

                    # Validate speed and course
                    if not is_valid_speed(speed):
                        error_counts["invalid_speed"] += 1
                        speed = 0

                    if not is_valid_course(course):
                        error_counts["invalid_course"] += 1
                        course = 0

                    # Get navigation status
                    nav_status = data.get("status")

                    # Create position record
                    record = {
                        "mmsi": mmsi,
                        "timestamp": timestamp_str,
                        "lat": lat,
                        "lon": lon,
                        "sog": speed,
                        "cog": course,
                        "heading": heading,
                        "nav_status": nav_status,
                        "msg_type": msg_type,
                    }

                    position_records.append(record)

                # Process static vessel information
                elif msg_type in STATIC_MESSAGE_TYPES:
                    if msg_type == 5:
                        record = {
                            "mmsi": mmsi,
                            "timestamp": timestamp_str,
                            "name": data.get("shipname"),
                            "call_sign": data.get("callsign"),
                            "imo": data.get("imo"),
                            "ship_type": data.get("shiptype"),
                            "ship_type_text": data.get("shiptype_text"),
                            "length": data.get("to_bow", 0) + data.get("to_stern", 0),
                            "width": data.get("to_port", 0)
                            + data.get("to_starboard", 0),
                            "draft": data.get("draught"),
                            "destination": data.get("destination"),
                            "eta": data.get("eta"),
                            "msg_type": msg_type,
                        }
                        static_records.append(record)

                    elif msg_type == 24:
                        partno = data.get("partno")

                        record = {
                            "mmsi": mmsi,
                            "timestamp": timestamp_str,
                            "msg_type": msg_type,
                            "partno": partno,
                        }

                        if partno == 0:
                            record["name"] = data.get("shipname")
                        elif partno == 1:
                            record["call_sign"] = data.get("callsign")
                            record["ship_type"] = data.get("shiptype")
                            record["ship_type_text"] = data.get("shiptype_text")
                            record["length"] = data.get("to_bow", 0) + data.get(
                                "to_stern", 0
                            )
                            record["width"] = data.get("to_port", 0) + data.get(
                                "to_starboard", 0
                            )

                        static_records.append(record)

            except (json.JSONDecodeError, KeyError):
                error_counts["malformed_lines"] += 1
                continue

            # Update progress bar
            lines_processed += 1
            pbar.update(1)

            # Log progress periodically
            if lines_processed % 100000 == 0:
                logger.info(
                    f"Processed {lines_processed} lines, found {len(position_records)} position reports and {len(static_records)} static messages"
                )

            # Limit records if needed
            if max_records and lines_processed >= max_records:
                logger.info(f"Reached maximum records limit ({max_records})")
                break

        pbar.close()

    # Log parsing statistics
    logger.info(f"Parsing complete: {lines_processed} lines processed")
    logger.info(
        f"Found {len(position_records)} position reports and {len(static_records)} static messages"
    )
    logger.info(f"Error counts: {error_counts}")
    logger.info(f"Message type distribution: {message_type_counts}")

    # Convert to DataFrames
    position_df = pd.DataFrame(position_records)
    static_df = pd.DataFrame(static_records)

    logger.info("Converting timestamps to datetime format")

    # Convert timestamps to datetime
    if not position_df.empty:
        try:
            position_df["timestamp"] = pd.to_datetime(
                position_df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f"
            )
        except ValueError:
            logger.warning(
                "Timestamp format didn't match expected pattern, using flexible parser"
            )
            position_df["timestamp"] = pd.to_datetime(position_df["timestamp"])
        position_df = position_df.sort_values(["mmsi", "timestamp"])

    if not static_df.empty:
        try:
            static_df["timestamp"] = pd.to_datetime(
                static_df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f"
            )
        except ValueError:
            logger.warning(
                "Timestamp format didn't match expected pattern, using flexible parser"
            )
            static_df["timestamp"] = pd.to_datetime(static_df["timestamp"])
        static_df = static_df.sort_values(["mmsi", "timestamp"])

    logger.info("AIS log parsing completed successfully")

    return position_df, static_df


def _process_vessel_trajectories(
    args: tuple[int, pd.DataFrame, int, int],
) -> list[pd.DataFrame]:
    """
    Process trajectories for a single vessel. Used for multiprocessing.

    Args:
        args: Tuple of (mmsi, vessel_data, max_gap_minutes, min_points)

    Returns:
        List of trajectory DataFrames for this vessel
    """
    mmsi, group, max_gap_minutes, min_points = args
    trajectories = []

    # Sort by timestamp
    group = group.sort_values("timestamp").reset_index(drop=True)

    # Calculate time differences
    group["time_diff"] = group["timestamp"].diff().dt.total_seconds() / 60.0

    # Find gaps larger than threshold
    gap_indices = np.where(group["time_diff"] > max_gap_minutes)[0]

    if len(gap_indices) == 0:
        # Single trajectory
        if len(group) >= min_points:
            trajectories.append(group.copy())
    else:
        # Multiple trajectories
        start_idx = 0

        for gap_idx in gap_indices:
            segment = group.iloc[start_idx:gap_idx].copy()
            if len(segment) >= min_points:
                trajectories.append(segment)
            start_idx = gap_idx

        # Last segment
        segment = group.iloc[start_idx:].copy()
        if len(segment) >= min_points:
            trajectories.append(segment)

    return trajectories


def create_trajectories(
    position_df: pd.DataFrame, max_gap_minutes: int = 30, min_points: int = 6
) -> list[pd.DataFrame]:
    """
    Create vessel trajectories from position data using multiprocessing.

    Args:
        position_df: DataFrame with vessel position data
        max_gap_minutes: Maximum gap in minutes to consider same trajectory
        min_points: Minimum number of points per trajectory

    Returns:
        List of trajectory DataFrames
    """
    if position_df.empty:
        logger.warning("Empty position DataFrame, no trajectories to create")
        return []

    logger.info(
        f"Creating trajectories with max_gap={max_gap_minutes} minutes, min_points={min_points}"
    )

    # Group by vessel MMSI
    grouped = position_df.groupby("mmsi")
    num_vessels = len(grouped)
    logger.info(
        f"Processing {num_vessels} unique vessels using {OPTIMAL_WORKERS} workers"
    )

    # Prepare arguments for parallel processing
    vessel_args = [
        (mmsi, group, max_gap_minutes, min_points) for mmsi, group in grouped
    ]

    # Process vessels in parallel
    with mp.Pool(processes=OPTIMAL_WORKERS) as pool:
        vessel_results = pool.map(_process_vessel_trajectories, vessel_args)

    # Combine results
    trajectories = []
    for vessel_trajectories in vessel_results:
        trajectories.extend(vessel_trajectories)

    logger.info(f"Created {len(trajectories)} trajectories")
    return trajectories


def add_derived_features(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a trajectory."""
    traj = trajectory.copy()

    # Time delta in seconds
    traj["delta_time"] = traj["time_diff"] * 60.0  # Convert from minutes to seconds

    # Calculate distance between consecutive points
    distances = []
    bearings = []

    for i in range(len(traj)):
        if i == 0:
            distances.append(0.0)
            bearings.append(None)
        else:
            lat1, lon1 = traj.iloc[i - 1]["lat"], traj.iloc[i - 1]["lon"]
            lat2, lon2 = traj.iloc[i]["lat"], traj.iloc[i]["lon"]

            dist = haversine_distance(lat1, lon1, lat2, lon2)
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)

            distances.append(dist)
            bearings.append(bearing)

    traj["distance_km"] = distances
    traj["bearing"] = bearings

    # Convert distance to nautical miles
    traj["distance_nm"] = traj["distance_km"] / 1.852

    # Calculate speed changes
    traj["speed_delta"] = traj["sog"].diff()

    # Calculate acceleration (knots per minute)
    traj["acceleration"] = traj["speed_delta"] / traj["time_diff"]

    # Calculate course changes
    traj["course_delta"] = traj["cog"].diff()
    traj.loc[traj["course_delta"] > 180, "course_delta"] -= 360
    traj.loc[traj["course_delta"] < -180, "course_delta"] += 360

    # Calculate turn rate (degrees per minute)
    traj["turn_rate"] = traj["course_delta"] / traj["time_diff"]

    return traj


def filter_outliers(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers from a trajectory."""
    traj = trajectory.copy()

    if len(traj) < 3:
        return traj

    # Flag outliers
    traj["is_outlier"] = False

    # Check for position jumps
    for i in range(1, len(traj)):
        if traj.iloc[i]["delta_time"] > 0:
            implied_speed_knots = traj.iloc[i]["distance_nm"] / (
                traj.iloc[i]["delta_time"] / 3600
            )

            if implied_speed_knots > MAX_SPEED_KNOTS * 1.5:
                traj.loc[traj.index[i], "is_outlier"] = True

    # Check for unrealistic accelerations
    traj.loc[abs(traj["acceleration"]) > 2, "is_outlier"] = True

    # Check for unrealistic turn rates
    traj.loc[abs(traj["turn_rate"]) > 30, "is_outlier"] = True

    # Remove outliers
    return traj[~traj["is_outlier"]].copy()


def export_to_formats(
    trajectories: list[pd.DataFrame], output_prefix: str
) -> list[str]:
    """Export trajectories to various formats."""
    if not trajectories:
        return []

    output_files = []

    # Create output directory if it doesn't exist
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to Parquet (flat table)
    all_trajectories = pd.concat(trajectories, ignore_index=True)
    parquet_path = f"{output_prefix}_all.parquet"
    all_trajectories.to_parquet(parquet_path, index=False)
    output_files.append(parquet_path)

    # Export to CSV (flat table)
    csv_path = f"{output_prefix}_all.csv"
    all_trajectories.to_csv(csv_path, index=False)
    output_files.append(csv_path)

    # Export static data separately
    static_data = []

    for traj in trajectories:
        if traj.empty:
            continue

        mmsi = traj["mmsi"].iloc[0]

        # Extract static data from first row
        static_row = {"mmsi": mmsi}

        for col in [
            "name",
            "call_sign",
            "imo",
            "ship_type",
            "ship_type_text",
            "length",
            "width",
            "draft",
            "destination",
            "eta",
        ]:
            if col in traj.columns:
                static_row[col] = traj[col].iloc[0]

        static_data.append(static_row)

    static_df = pd.DataFrame(static_data)
    static_csv_path = f"{output_prefix}_static.csv"
    static_df.to_csv(static_csv_path, index=False)
    output_files.append(static_csv_path)

    return output_files


def run_preprocess(cfg: DictConfig) -> dict[str, any]:
    """
    Main preprocessing function called by Hydra dispatch.

    Args:
        cfg: Hydra configuration object

    Returns:
        Results dictionary
    """
    logger.info("Starting AIS data preprocessing pipeline")

    # Extract configuration
    data_cfg = cfg.data
    input_file = data_cfg.get("file")
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if input_file and Path(input_file).exists():
        # Process specific AIS log file
        logger.info(f"Processing AIS-catcher log file: {input_file}")

        # Parse AIS data
        position_df, static_df = parse_ais_catcher_log(input_file)

        logger.info(
            f"Parsed {len(position_df)} position messages and {len(static_df)} static messages"
        )

        # Create trajectories
        trajectories = create_trajectories(
            position_df,
            max_gap_minutes=data_cfg.get("max_gap_minutes", 30),
            min_points=data_cfg.get("min_points", 6),
        )

        # Add derived features and filter outliers using multiprocessing
        logger.info(
            f"Processing {len(trajectories)} trajectories using {OPTIMAL_WORKERS} workers"
        )

        with mp.Pool(processes=OPTIMAL_WORKERS) as pool:
            enriched_trajectories = pool.map(_process_trajectory_features, trajectories)

        # Export data
        output_prefix = processed_dir / "processed_ais"
        logger.info(f"Exporting {len(enriched_trajectories)} trajectories...")
        output_files = export_to_formats(enriched_trajectories, str(output_prefix))

        logger.info("Exported data to:")
        for f in output_files:
            logger.info(f"  - {f}")

        results = {
            "input_file": input_file,
            "position_records": len(position_df),
            "static_records": len(static_df),
            "trajectories_created": len(enriched_trajectories),
            "output_files": output_files,
        }

    else:
        # Process all log files in raw directory
        logger.info("Processing all log files in raw directory...")

        # Step 1: Parse raw logs to Parquet
        parquet_files = process_raw_logs_to_parquet(
            raw_dir, processed_dir / "parquet_chunks"
        )

        if not parquet_files:
            logger.error("No Parquet files created. Check raw data directory.")
            return {"error": "No Parquet files created"}

        # Step 2: Create xarray datasets
        logger.info("Creating xarray datasets...")
        zarr_path = processed_dir / "ais_positions.zarr"
        static_zarr_path = processed_dir / "ais_static.zarr"

        create_xarray_datasets(
            parquet_dir=processed_dir / "parquet_chunks",
            zarr_path=zarr_path,
            static_zarr_path=static_zarr_path,
        )

        results = {
            "parquet_files": len(parquet_files),
            "zarr_path": str(zarr_path),
            "static_zarr_path": str(static_zarr_path),
        }

    logger.info("Preprocessing pipeline completed successfully!")
    return results


def main():
    """Legacy main function for backward compatibility."""
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
