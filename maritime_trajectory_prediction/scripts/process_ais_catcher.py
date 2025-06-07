#!/usr/bin/env python3
"""
Script to process AIS-catcher log files for maritime trajectory prediction.

This script implements the AIS data pipeline described in the AIS data documentation.
It processes AIS messages from a log file, filters relevant message types,
reconstructs vessel trajectories, engineers features, and outputs the data
in formats suitable for machine learning.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt, atan2, degrees
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Constants
EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers
MAX_SPEED_KNOTS = 50.0  # Maximum realistic speed
MIN_SPEED_KNOTS = 0.5  # Minimum speed to consider a vessel moving
MAX_GAP_MINUTES = 30  # Maximum gap in minutes to consider same trajectory
MIN_POINTS = 6  # Minimum points per trajectory
POSITION_MESSAGE_TYPES = [1, 2, 3, 18, 19]  # Dynamic position report types
STATIC_MESSAGE_TYPES = [5, 24]  # Static vessel info message types

# Setup logging
def setup_logging(level=logging.INFO):
    """Configure logging for the AIS processing pipeline."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ais_processor")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (in degrees)
        lat2, lon2: Coordinates of second point (in degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return EARTH_RADIUS_KM * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing between two points.
    
    Args:
        lat1, lon1: Coordinates of first point (in degrees)
        lat2, lon2: Coordinates of second point (in degrees)
        
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate bearing
    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
    bearing = atan2(y, x)
    
    # Convert to degrees and normalize
    bearing_deg = (degrees(bearing) + 360) % 360
    
    return bearing_deg

def knots_to_kmh(knots):
    """Convert speed in knots to kilometers per hour"""
    return knots * 1.852

def kmh_to_knots(kmh):
    """Convert speed in kilometers per hour to knots"""
    return kmh / 1.852

def is_valid_mmsi(mmsi):
    """Check if MMSI is valid for a vessel"""
    if not mmsi:
        return False
    
    # MMSI should be a 9-digit number
    mmsi_str = str(mmsi)
    if len(mmsi_str) != 9:
        return False
    
    # Exclude certain patterns (like 00... or 99... for base stations)
    if mmsi_str.startswith('00') or mmsi_str.startswith('99'):
        return False
    
    return True

def is_valid_position(lat, lon):
    """Check if position is valid"""
    if lat is None or lon is None:
        return False
    
    # Check if within valid ranges
    if abs(lat) > 90 or abs(lon) > 180:
        return False
    
    # Check for sentinel values
    if lat == 91.0 or lon == 181.0:
        return False
    
    return True

def is_valid_speed(speed):
    """Check if speed is valid"""
    if speed is None:
        return False
    
    # Check if within valid range
    if speed < 0 or speed > MAX_SPEED_KNOTS:
        return False
    
    # Check for sentinel values (102.3 knots indicates "not available")
    if abs(speed - 102.3) < 0.1:
        return False
    
    return True

def is_valid_course(course):
    """Check if course is valid"""
    if course is None:
        return False
    
    # Check if within valid range
    # 360 or 361 may indicate "not available"
    if course < 0 or course >= 360:
        return False
    
    return True

def parse_ais_catcher_log(log_file, logger=None, max_records=None):
    """
    Parse AIS-catcher log file into DataFrames for position and static data.
    
    Args:
        log_file: Path to the AIS-catcher log file
        logger: Logger object for detailed logging
        max_records: Maximum number of records to process (for testing)
        
    Returns:
        Tuple of (position_df, static_df) with parsed AIS data
    """
    if logger is None:
        logger = logging.getLogger("ais_processor")
    
    position_records = []
    static_records = []
    
    error_counts = {
        "malformed_lines": 0,
        "invalid_mmsi": 0,
        "invalid_position": 0,
        "invalid_speed": 0,
        "invalid_course": 0
    }
    message_type_counts = {}
    
    logger.info(f"Starting to parse AIS log file: {log_file}")
    
    # Count number of lines in file for progress bar
    if max_records is None:
        try:
            with open(log_file, 'r') as f:
                total_lines = sum(1 for _ in f)
                logger.info(f"File contains {total_lines} lines")
        except Exception as e:
            logger.warning(f"Could not count lines in file: {e}")
            total_lines = None
    else:
        total_lines = max_records
        logger.info(f"Processing up to {max_records} records")
    
    # Parse the file with progress bar
    with open(log_file, 'r') as f:
        # Create a progress bar
        if total_lines:
            pbar = tqdm(total=total_lines, desc="Parsing AIS data")
        else:
            pbar = tqdm(desc="Parsing AIS data")
        
        lines_processed = 0
        for line in f:
            try:
                # Extract JSON part (after timestamp)
                parts = line.strip().split(' - ', 1)
                if len(parts) != 2:
                    error_counts["malformed_lines"] += 1
                    continue
                    
                timestamp_str = parts[0]
                json_str = parts[1]
                
                # Parse JSON
                data = json.loads(json_str)
                
                # Only process AIS messages
                if data.get('class') != 'AIS':
                    continue
                
                # Get MMSI
                mmsi = data.get('mmsi')
                if not is_valid_mmsi(mmsi):
                    error_counts["invalid_mmsi"] += 1
                    continue
                
                # Process dynamic position messages
                msg_type = data.get('type')
                
                # Count message types for statistics
                if msg_type in message_type_counts:
                    message_type_counts[msg_type] += 1
                else:
                    message_type_counts[msg_type] = 1
                
                if msg_type in POSITION_MESSAGE_TYPES:
                    lat = data.get('lat')
                    lon = data.get('lon')
                    
                    # Validate position
                    if not is_valid_position(lat, lon):
                        error_counts["invalid_position"] += 1
                        continue
                    
                    # Extract dynamic data
                    speed = data.get('speed', 0)  # Speed over ground in knots
                    course = data.get('course', 0)  # Course over ground in degrees
                    heading = data.get('heading')  # True heading in degrees
                    
                    # Validate speed and course
                    if not is_valid_speed(speed):
                        error_counts["invalid_speed"] += 1
                        # Set to 0 if invalid but position is valid
                        speed = 0
                    
                    if not is_valid_course(course):
                        error_counts["invalid_course"] += 1
                        # Set to 0 if invalid but position is valid
                        course = 0
                    
                    # For stationary vessels, course might be unreliable
                    if speed < MIN_SPEED_KNOTS:
                        # Don't trust course for very low speeds
                        pass
                    
                    # Get navigation status (for Class A)
                    nav_status = data.get('status')
                    
                    # Create position record
                    record = {
                        'mmsi': mmsi,
                        'timestamp': timestamp_str,
                        'lat': lat,
                        'lon': lon,
                        'sog': speed,
                        'cog': course,
                        'heading': heading,
                        'nav_status': nav_status,
                        'msg_type': msg_type
                    }
                    
                    position_records.append(record)
                
                # Process static vessel information
                elif msg_type in STATIC_MESSAGE_TYPES:
                    # Type 5 - Class A static and voyage data
                    if msg_type == 5:
                        record = {
                            'mmsi': mmsi,
                            'timestamp': timestamp_str,
                            'name': data.get('shipname'),
                            'call_sign': data.get('callsign'),
                            'imo': data.get('imo'),
                            'ship_type': data.get('shiptype'),
                            'ship_type_text': data.get('shiptype_text'),
                            'length': data.get('to_bow', 0) + data.get('to_stern', 0),
                            'width': data.get('to_port', 0) + data.get('to_starboard', 0),
                            'draft': data.get('draught'),
                            'destination': data.get('destination'),
                            'eta': data.get('eta'),
                            'msg_type': msg_type
                        }
                        static_records.append(record)
                    
                    # Type 24 - Class B static data
                    elif msg_type == 24:
                        # Part A (shipname) or Part B (ship type, dimensions)
                        partno = data.get('partno')
                        
                        record = {
                            'mmsi': mmsi,
                            'timestamp': timestamp_str,
                            'msg_type': msg_type,
                            'partno': partno
                        }
                        
                        # Part A - Ship name
                        if partno == 0:
                            record['name'] = data.get('shipname')
                        
                        # Part B - Ship type and dimensions
                        elif partno == 1:
                            record['call_sign'] = data.get('callsign')
                            record['ship_type'] = data.get('shiptype')
                            record['ship_type_text'] = data.get('shiptype_text')
                            record['length'] = data.get('to_bow', 0) + data.get('to_stern', 0)
                            record['width'] = data.get('to_port', 0) + data.get('to_starboard', 0)
                        
                        static_records.append(record)
                    
            except (json.JSONDecodeError, KeyError) as e:
                error_counts["malformed_lines"] += 1
                # Skip malformed lines
                continue
            
            # Update progress bar
            lines_processed += 1
            pbar.update(1)
            
            # Log progress periodically
            if lines_processed % 100000 == 0:
                logger.info(f"Processed {lines_processed} lines, found {len(position_records)} position reports and {len(static_records)} static messages")
            
            # Limit records if needed
            if max_records and lines_processed >= max_records:
                logger.info(f"Reached maximum records limit ({max_records})")
                break
        
        pbar.close()
    
    # Log parsing statistics
    logger.info(f"Parsing complete: {lines_processed} lines processed")
    logger.info(f"Found {len(position_records)} position reports and {len(static_records)} static messages")
    logger.info(f"Error counts: {error_counts}")
    logger.info(f"Message type distribution: {message_type_counts}")
    
    # Convert to DataFrames
    position_df = pd.DataFrame(position_records)
    static_df = pd.DataFrame(static_records)
    
    logger.info("Converting timestamps to datetime format")
    
    # Convert timestamps to datetime
    if not position_df.empty:
        # First try to parse with a specific format, falling back to dateutil
        try:
            position_df['timestamp'] = pd.to_datetime(position_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')
        except ValueError:
            logger.warning("Timestamp format didn't match expected pattern, using flexible parser")
            position_df['timestamp'] = pd.to_datetime(position_df['timestamp'])
        position_df = position_df.sort_values(['mmsi', 'timestamp'])
    
    if not static_df.empty:
        try:
            static_df['timestamp'] = pd.to_datetime(static_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')
        except ValueError:
            logger.warning("Timestamp format didn't match expected pattern, using flexible parser")
            static_df['timestamp'] = pd.to_datetime(static_df['timestamp'])
        static_df = static_df.sort_values(['mmsi', 'timestamp'])
    
    logger.info("AIS log parsing completed successfully")
    
    return position_df, static_df

def process_static_data(static_df):
    """
    Process static vessel data to create a single record per vessel.
    
    Args:
        static_df: DataFrame with static vessel information
        
    Returns:
        DataFrame with one row per MMSI containing latest static info
    """
    if static_df.empty:
        return pd.DataFrame()
    
    # Process Type 5 messages (Class A)
    type5_df = static_df[static_df['msg_type'] == 5].copy()
    
    # Process Type 24 messages (Class B)
    type24_df = static_df[static_df['msg_type'] == 24].copy()
    
    # For Type 24, we need to combine Part A and Part B
    if not type24_df.empty:
        # Part A - Ship name
        part_a = type24_df[type24_df['partno'] == 0].copy()
        part_a = part_a.sort_values('timestamp').groupby('mmsi').last().reset_index()
        
        # Part B - Ship type and dimensions
        part_b = type24_df[type24_df['partno'] == 1].copy()
        part_b = part_b.sort_values('timestamp').groupby('mmsi').last().reset_index()
        
        # Merge Part A and Part B
        if not part_a.empty and not part_b.empty:
            type24_combined = pd.merge(
                part_a[['mmsi', 'name']],
                part_b[['mmsi', 'call_sign', 'ship_type', 'ship_type_text', 'length', 'width']],
                on='mmsi',
                how='outer'
            )
        elif not part_a.empty:
            type24_combined = part_a
        elif not part_b.empty:
            type24_combined = part_b
        else:
            type24_combined = pd.DataFrame()
    else:
        type24_combined = pd.DataFrame()
    
    # Combine Type 5 and Type 24
    if not type5_df.empty:
        # Get latest Type 5 message per vessel
        type5_latest = type5_df.sort_values('timestamp').groupby('mmsi').last().reset_index()
        
        # Select relevant columns
        type5_cols = ['mmsi', 'name', 'call_sign', 'imo', 'ship_type', 'ship_type_text', 
                      'length', 'width', 'draft', 'destination', 'eta']
        type5_latest = type5_latest[type5_cols].copy()
    else:
        type5_latest = pd.DataFrame()
    
    # Combine all static data
    if not type5_latest.empty and not type24_combined.empty:
        # Prefer Type 5 over Type 24 if both exist
        static_combined = pd.concat([type24_combined, type5_latest])
        static_combined = static_combined.groupby('mmsi').last().reset_index()
    elif not type5_latest.empty:
        static_combined = type5_latest
    elif not type24_combined.empty:
        static_combined = type24_combined
    else:
        static_combined = pd.DataFrame()
    
    return static_combined

def create_trajectories(position_df, max_gap_minutes=MAX_GAP_MINUTES, min_points=MIN_POINTS, logger=None):
    """
    Create vessel trajectories from position data.
    
    Args:
        position_df: DataFrame with vessel position data
        max_gap_minutes: Maximum gap in minutes to consider same trajectory
        min_points: Minimum number of points per trajectory
        logger: Logger object for detailed logging
        
    Returns:
        List of trajectory DataFrames
    """
    if logger is None:
        logger = logging.getLogger("ais_processor")
    
    if position_df.empty:
        logger.warning("Empty position DataFrame, no trajectories to create")
        return []
    
    logger.info(f"Creating trajectories with max_gap={max_gap_minutes} minutes, min_points={min_points}")
    
    # Group by vessel MMSI
    grouped = position_df.groupby('mmsi')
    num_vessels = len(grouped)
    logger.info(f"Processing {num_vessels} unique vessels")
    
    trajectories = []
    trajectory_stats = {
        "total_vessels": num_vessels,
        "vessels_with_trajectories": 0,
        "trajectory_lengths": [],
        "max_points": 0,
        "min_points": float('inf'),
        "total_segments": 0
    }
    
    # Process each vessel with progress bar
    for mmsi, group in tqdm(grouped, desc="Creating vessel trajectories", total=num_vessels):
        # Sort by timestamp
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate time differences
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds() / 60.0
        
        # Find gaps larger than threshold
        gap_indices = np.where(group['time_diff'] > max_gap_minutes)[0]
        
        vessel_segments = 0
        
        if len(gap_indices) == 0:
            # Single trajectory
            if len(group) >= min_points:
                trajectories.append(group.copy())
                vessel_segments += 1
                trajectory_stats["trajectory_lengths"].append(len(group))
                trajectory_stats["max_points"] = max(trajectory_stats["max_points"], len(group))
                trajectory_stats["min_points"] = min(trajectory_stats["min_points"], len(group))
        else:
            # Multiple trajectories
            start_idx = 0
            
            for gap_idx in gap_indices:
                segment = group.iloc[start_idx:gap_idx].copy()
                if len(segment) >= min_points:
                    trajectories.append(segment)
                    vessel_segments += 1
                    trajectory_stats["trajectory_lengths"].append(len(segment))
                    trajectory_stats["max_points"] = max(trajectory_stats["max_points"], len(segment))
                    trajectory_stats["min_points"] = min(trajectory_stats["min_points"], len(segment))
                start_idx = gap_idx
            
            # Last segment
            segment = group.iloc[start_idx:].copy()
            if len(segment) >= min_points:
                trajectories.append(segment)
                vessel_segments += 1
                trajectory_stats["trajectory_lengths"].append(len(segment))
                trajectory_stats["max_points"] = max(trajectory_stats["max_points"], len(segment))
                trajectory_stats["min_points"] = min(trajectory_stats["min_points"], len(segment))
        
        if vessel_segments > 0:
            trajectory_stats["vessels_with_trajectories"] += 1
            trajectory_stats["total_segments"] += vessel_segments
            
            # Log more details for vessels with many segments
            if vessel_segments > 5:
                logger.debug(f"Vessel MMSI {mmsi} has {vessel_segments} trajectory segments")
    
    # Calculate statistics
    if trajectory_stats["trajectory_lengths"]:
        trajectory_stats["avg_points"] = sum(trajectory_stats["trajectory_lengths"]) / len(trajectory_stats["trajectory_lengths"])
    else:
        trajectory_stats["min_points"] = 0
        trajectory_stats["avg_points"] = 0
    
    # Log trajectory statistics
    logger.info(f"Created {len(trajectories)} trajectories from {trajectory_stats['vessels_with_trajectories']} vessels")
    logger.info(f"Trajectory statistics: {trajectory_stats['vessels_with_trajectories']}/{trajectory_stats['total_vessels']} vessels have trajectories")
    logger.info(f"Points per trajectory: min={trajectory_stats['min_points']}, max={trajectory_stats['max_points']}, avg={trajectory_stats['avg_points']:.1f}")
    logger.info(f"Average segments per vessel: {trajectory_stats['total_segments']/max(1, trajectory_stats['vessels_with_trajectories']):.1f}")
    
    return trajectories

def add_derived_features(trajectory):
    """
    Add derived features to a trajectory.
    
    Args:
        trajectory: DataFrame with a vessel trajectory
        
    Returns:
        DataFrame with added features
    """
    traj = trajectory.copy()
    
    # Time delta in seconds
    traj['delta_time'] = traj['time_diff'] * 60.0  # Convert from minutes to seconds
    
    # Calculate distance between consecutive points
    distances = []
    bearings = []
    
    for i in range(len(traj)):
        if i == 0:
            distances.append(0.0)
            bearings.append(None)
        else:
            lat1, lon1 = traj.iloc[i-1]['lat'], traj.iloc[i-1]['lon']
            lat2, lon2 = traj.iloc[i]['lat'], traj.iloc[i]['lon']
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            
            distances.append(dist)
            bearings.append(bearing)
    
    traj['distance_km'] = distances
    traj['bearing'] = bearings
    
    # Convert distance to nautical miles
    traj['distance_nm'] = traj['distance_km'] / 1.852
    
    # Calculate speed changes
    traj['speed_delta'] = traj['sog'].diff()
    
    # Calculate acceleration (knots per minute)
    traj['acceleration'] = traj['speed_delta'] / traj['time_diff']
    
    # Calculate course changes
    # Handle circular nature of course (0-360 degrees)
    traj['course_delta'] = traj['cog'].diff()
    traj.loc[traj['course_delta'] > 180, 'course_delta'] -= 360
    traj.loc[traj['course_delta'] < -180, 'course_delta'] += 360
    
    # Calculate turn rate (degrees per minute)
    traj['turn_rate'] = traj['course_delta'] / traj['time_diff']
    
    # Calculate course drift (difference between heading and course)
    traj['course_drift'] = None
    mask = ~traj['heading'].isna()
    traj.loc[mask, 'course_drift'] = (traj.loc[mask, 'heading'] - traj.loc[mask, 'cog']).abs()
    traj.loc[traj['course_drift'] > 180, 'course_drift'] = 360 - traj.loc[traj['course_drift'] > 180, 'course_drift']
    
    # Calculate bearing deviation (difference between bearing and course)
    traj['bearing_deviation'] = None
    mask = ~traj['bearing'].isna()
    traj.loc[mask, 'bearing_deviation'] = (traj.loc[mask, 'bearing'] - traj.loc[mask, 'cog']).abs()
    traj.loc[traj['bearing_deviation'] > 180, 'bearing_deviation'] = 360 - traj.loc[traj['bearing_deviation'] > 180, 'bearing_deviation']
    
    return traj

def filter_outliers(trajectory):
    """
    Filter outliers from a trajectory.
    
    Args:
        trajectory: DataFrame with a vessel trajectory
        
    Returns:
        Filtered DataFrame
    """
    traj = trajectory.copy()
    
    if len(traj) < 3:
        return traj
    
    # Flag outliers
    traj['is_outlier'] = False
    
    # Check for position jumps
    # If the distance implies a speed much higher than MAX_SPEED_KNOTS, it's likely an outlier
    for i in range(1, len(traj)):
        if traj.iloc[i]['delta_time'] > 0:
            implied_speed_knots = traj.iloc[i]['distance_nm'] / (traj.iloc[i]['delta_time'] / 3600)
            
            if implied_speed_knots > MAX_SPEED_KNOTS * 1.5:  # Allow some buffer
                traj.loc[traj.index[i], 'is_outlier'] = True
    
    # Check for unrealistic accelerations (more than 2 knots per minute)
    traj.loc[abs(traj['acceleration']) > 2, 'is_outlier'] = True
    
    # Check for unrealistic turn rates (more than 30 degrees per minute for large vessels)
    # This could be refined based on vessel type/size
    traj.loc[abs(traj['turn_rate']) > 30, 'is_outlier'] = True
    
    # For zero speed, course changes are unreliable
    mask = (traj['sog'] < MIN_SPEED_KNOTS) & (abs(traj['course_delta']) > 5)
    traj.loc[mask, 'course_delta'] = 0
    traj.loc[mask, 'turn_rate'] = 0
    
    # Remove outliers
    return traj[~traj['is_outlier']].copy()

def enrich_with_static_data(trajectories, static_df):
    """
    Enrich trajectories with static vessel data.
    
    Args:
        trajectories: List of trajectory DataFrames
        static_df: DataFrame with static vessel information
        
    Returns:
        List of enriched trajectory DataFrames
    """
    if static_df.empty or not trajectories:
        return trajectories
    
    enriched_trajectories = []
    
    for traj in trajectories:
        if traj.empty:
            continue
        
        mmsi = traj['mmsi'].iloc[0]
        
        # Find static data for this vessel
        vessel_static = static_df[static_df['mmsi'] == mmsi]
        
        if not vessel_static.empty:
            # Get static fields
            static_data = vessel_static.iloc[0].to_dict()
            
            # Add static fields to trajectory
            for key, value in static_data.items():
                if key not in traj.columns:
                    traj[key] = value
        
        enriched_trajectories.append(traj)
    
    return enriched_trajectories

def export_to_formats(trajectories, output_prefix):
    """
    Export trajectories to various formats.
    
    Args:
        trajectories: List of trajectory DataFrames
        output_prefix: Prefix for output files
        
    Returns:
        Paths to the exported files
    """
    if not trajectories:
        return []
    
    output_files = []
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Export to Parquet (flat table)
    all_trajectories = pd.concat(trajectories, ignore_index=True)
    parquet_path = f"{output_prefix}_all.parquet"
    all_trajectories.to_parquet(parquet_path, index=False)
    output_files.append(parquet_path)
    
    # 2. Export to CSV (flat table)
    csv_path = f"{output_prefix}_all.csv"
    all_trajectories.to_csv(csv_path, index=False)
    output_files.append(csv_path)
    
    # 3. Export to JSON Lines (one line per vessel)
    jsonl_path = f"{output_prefix}_vessels.jsonl"
    
    # JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(jsonl_path, 'w') as f:
        for traj in trajectories:
            if traj.empty:
                continue
                
            mmsi = int(traj['mmsi'].iloc[0])  # Convert to standard Python int
            
            # Extract static data from first row
            static_cols = ['name', 'call_sign', 'imo', 'ship_type', 'ship_type_text',
                          'length', 'width', 'draft', 'destination', 'eta']
            
            static_data = {}
            for col in static_cols:
                if col in traj.columns:
                    val = traj[col].iloc[0]
                    if not pd.isna(val):
                        # Convert numpy types to Python native types
                        if isinstance(val, (np.integer, np.int64)):
                            static_data[col] = int(val)
                        elif isinstance(val, (np.floating, np.float64)):
                            static_data[col] = float(val)
                        else:
                            static_data[col] = val
            
            # Extract trajectory data
            traj_data = []
            for _, row in traj.iterrows():
                # Convert timestamp to string
                timestamp_str = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                
                point = {
                    'timestamp': timestamp_str,
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'sog': float(row['sog']),
                    'cog': float(row['cog'])
                }
                
                # Add optional fields if available
                for field in ['heading', 'nav_status', 'delta_time', 'distance_km',
                             'distance_nm', 'speed_delta', 'acceleration', 
                             'course_delta', 'turn_rate']:
                    if field in row and not pd.isna(row[field]):
                        # Convert numpy types to Python native types
                        if isinstance(row[field], (np.integer, np.int64)):
                            point[field] = int(row[field])
                        elif isinstance(row[field], (np.floating, np.float64)):
                            point[field] = float(row[field])
                        else:
                            point[field] = row[field]
                
                traj_data.append(point)
            
            # Create vessel object
            vessel_obj = {
                'mmsi': mmsi,
                'static': static_data,
                'trajectory': traj_data
            }
            
            # Write to file using custom encoder for NumPy types
            f.write(json.dumps(vessel_obj, cls=NumpyEncoder) + '\n')
    
    output_files.append(jsonl_path)
    
    # 4. Export static data separately
    static_data = []
    
    for traj in trajectories:
        if traj.empty:
            continue
            
        mmsi = traj['mmsi'].iloc[0]
        
        # Extract static data from first row
        static_row = {'mmsi': mmsi}
        
        for col in ['name', 'call_sign', 'imo', 'ship_type', 'ship_type_text',
                   'length', 'width', 'draft', 'destination', 'eta']:
            if col in traj.columns:
                static_row[col] = traj[col].iloc[0]
        
        static_data.append(static_row)
    
    static_df = pd.DataFrame(static_data)
    static_csv_path = f"{output_prefix}_static.csv"
    static_df.to_csv(static_csv_path, index=False)
    output_files.append(static_csv_path)
    
    return output_files

def plot_trajectories(trajectories, output_file=None, max_trajectories=10):
    """
    Plot vessel trajectories on a map.
    
    Args:
        trajectories: List of trajectory DataFrames
        output_file: Path to save the plot image
        max_trajectories: Maximum number of trajectories to plot
        
    Returns:
        Figure and axes objects
    """
    if not trajectories:
        return None, None
    
    # Limit number of trajectories to avoid cluttered plot
    if len(trajectories) > max_trajectories:
        print(f"Limiting visualization to {max_trajectories} trajectories")
        trajectories_to_plot = trajectories[:max_trajectories]
    else:
        trajectories_to_plot = trajectories
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each trajectory with a different color
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories_to_plot)))
    
    for i, traj in enumerate(trajectories_to_plot):
        if traj.empty:
            continue
            
        # Get vessel info for label
        mmsi = traj['mmsi'].iloc[0]
        vessel_name = traj['name'].iloc[0] if 'name' in traj.columns and not pd.isna(traj['name'].iloc[0]) else f"MMSI: {mmsi}"
        
        # Plot trajectory
        ax.plot(traj['lon'], traj['lat'], '-', color=colors[i], linewidth=2, alpha=0.7, label=vessel_name)
        
        # Mark start and end points
        ax.plot(traj['lon'].iloc[0], traj['lat'].iloc[0], 'o', color=colors[i], markersize=8)
        ax.plot(traj['lon'].iloc[-1], traj['lat'].iloc[-1], 's', color=colors[i], markersize=8)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Trajectories')
    ax.grid(True)
    
    # Add legend if not too many trajectories
    if len(trajectories_to_plot) <= 20:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description="Process AIS-catcher log files for maritime trajectory prediction")
    parser.add_argument("log_file", help="Path to AIS-catcher log file")
    parser.add_argument("--output", "-o", default="processed_ais", help="Output file prefix")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS, help="Minimum points per trajectory")
    parser.add_argument("--max-gap", type=int, default=MAX_GAP_MINUTES, help="Maximum gap in minutes between points")
    parser.add_argument("--visualize", action="store_true", help="Visualize trajectories")
    parser.add_argument("--max-trajectories", type=int, default=10, help="Maximum trajectories to visualize")
    parser.add_argument("--max-records", type=int, help="Maximum records to process (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logger = setup_logging(level=log_level)
    
    logger.info(f"Starting AIS data processing pipeline")
    logger.info(f"Processing AIS-catcher log file: {args.log_file}")
    logger.info(f"Parameters: min_points={args.min_points}, max_gap={args.max_gap}")
    
    # Parse AIS data
    position_df, static_df = parse_ais_catcher_log(
        args.log_file, 
        logger=logger, 
        max_records=args.max_records
    )
    
    logger.info(f"Parsed {len(position_df)} position messages and {len(static_df)} static messages")
    
    # Process static data
    logger.info("Processing static vessel data...")
    static_combined = process_static_data(static_df)
    
    logger.info(f"Combined static data for {len(static_combined)} vessels")
    
    # Create trajectories
    trajectories = create_trajectories(
        position_df, 
        max_gap_minutes=args.max_gap, 
        min_points=args.min_points,
        logger=logger
    )
    
    # Calculate derived features
    enriched_trajectories = []
    
    logger.info(f"Adding derived features and filtering outliers for {len(trajectories)} trajectories")
    
    for i, traj in enumerate(tqdm(trajectories, desc="Processing trajectories")):
        # Add derived features
        traj_with_features = add_derived_features(traj)
        
        # Filter outliers
        filtered_traj = filter_outliers(traj_with_features)
        
        # Log if significant filtering occurred
        if len(filtered_traj) < len(traj_with_features) * 0.9:  # More than 10% filtered
            logger.debug(f"Trajectory {i}: {len(traj_with_features) - len(filtered_traj)} outliers filtered out " +
                         f"({len(filtered_traj)}/{len(traj_with_features)} points remain)")
        
        enriched_trajectories.append(filtered_traj)
    
    # Enrich with static data
    logger.info("Enriching trajectories with static vessel data...")
    final_trajectories = enrich_with_static_data(enriched_trajectories, static_combined)
    
    # Export data
    logger.info(f"Exporting {len(final_trajectories)} trajectories...")
    output_files = export_to_formats(final_trajectories, args.output)
    
    logger.info(f"Exported data to:")
    for f in output_files:
        logger.info(f"  - {f}")
    
    # Display statistics
    logger.info("\nTrajectory Statistics:")
    lengths = [len(t) for t in final_trajectories]
    if lengths:
        logger.info(f"Min points: {min(lengths)}")
        logger.info(f"Max points: {max(lengths)}")
        logger.info(f"Avg points: {sum(lengths)/len(lengths):.1f}")
    
    if not position_df.empty:
        logger.info(f"Total vessels: {position_df['mmsi'].nunique()}")
    
    # Visualize if requested
    if args.visualize:
        logger.info("\nVisualizing trajectories...")
        fig, ax = plot_trajectories(
            final_trajectories, 
            output_file=f"{args.output}_trajectories.png",
            max_trajectories=args.max_trajectories
        )
        
        # Save and show
        logger.info(f"Trajectory visualization saved to {args.output}_trajectories.png")
        
        # Try to display if running in X environment
        try:
            logger.info("Displaying trajectory visualization...")
            plt.show()
        except Exception as e:
            logger.warning(f"Could not display visualization: {e}")
    
    logger.info("AIS data processing completed successfully!")

if __name__ == "__main__":
    main()