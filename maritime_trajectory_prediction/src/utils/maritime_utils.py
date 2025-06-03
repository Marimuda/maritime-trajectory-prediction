import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from datetime import timedelta

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return r * c

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate bearing
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = np.arctan2(y, x)
    
    # Convert to degrees
    bearing = np.degrees(bearing)
    
    # Normalize to 0-360
    bearing = (bearing + 360) % 360
    
    return bearing

def knots_to_kmh(knots):
    """Convert speed in knots to kilometers per hour"""
    return knots * 1.852

def kmh_to_knots(kmh):
    """Convert speed in kilometers per hour to knots"""
    return kmh / 1.852

def calculate_trajectory_features(trajectory):
    """
    Calculate trajectory features like speed, course, etc.
    
    Args:
        trajectory: DataFrame with lat, lon, timestamp columns
        
    Returns:
        DataFrame with additional features
    """
    # Copy to avoid modifying original
    traj = trajectory.copy()
    
    # Sort by timestamp
    traj = traj.sort_values('timestamp')
    
    # Calculate time differences in seconds
    traj['dt'] = traj['timestamp'].diff().dt.total_seconds()
    
    # Calculate spatial differences
    traj['dlat'] = traj['lat'].diff()
    traj['dlon'] = traj['lon'].diff()
    
    # Calculate distance in kilometers
    distances = []
    for i in range(len(traj)):
        if i == 0:
            distances.append(0)
        else:
            lat1, lon1 = traj.iloc[i-1]['lat'], traj.iloc[i-1]['lon']
            lat2, lon2 = traj.iloc[i]['lat'], traj.iloc[i]['lon']
            distances.append(haversine_distance(lat1, lon1, lat2, lon2))
    
    traj['distance_km'] = distances
    
    # Calculate speed in km/h
    traj['speed_kmh'] = traj['distance_km'] / (traj['dt'] / 3600)
    
    # Calculate speed in knots
    traj['speed_knots'] = kmh_to_knots(traj['speed_kmh'])
    
    # Calculate bearing
    bearings = []
    for i in range(len(traj)):
        if i == 0:
            bearings.append(0)
        else:
            lat1, lon1 = traj.iloc[i-1]['lat'], traj.iloc[i-1]['lon']
            lat2, lon2 = traj.iloc[i]['lat'], traj.iloc[i]['lon']
            bearings.append(bearing(lat1, lon1, lat2, lon2))
    
    traj['bearing'] = bearings
    
    # Calculate bearing changes
    traj['dbearing'] = traj['bearing'].diff()
    
    # Normalize bearing changes to -180 to 180
    traj['dbearing'] = (traj['dbearing'] + 180) % 360 - 180
    
    return traj

def interpolate_trajectory(trajectory, interval_minutes=10):
    """
    Interpolate a trajectory to a fixed time interval
    
    Args:
        trajectory: DataFrame with lat, lon, timestamp columns
        interval_minutes: Time interval for interpolation
        
    Returns:
        DataFrame with interpolated trajectory
    """
    # Copy to avoid modifying original
    traj = trajectory.copy()
    
    # Sort by timestamp
    traj = traj.sort_values('timestamp')
    
    # Create new time range
    start_time = traj['timestamp'].min()
    end_time = traj['timestamp'].max()
    
    # Create new index with fixed interval
    new_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{interval_minutes}min'
    )
    
    # Create new DataFrame with interpolated values
    new_traj = pd.DataFrame(index=new_index)
    new_traj.index.name = 'timestamp'
    new_traj = new_traj.reset_index()
    
    # Interpolate lat and lon
    traj_indexed = traj.set_index('timestamp')
    new_traj['lat'] = np.interp(
        new_traj['timestamp'].astype(int) / 10**9,
        traj_indexed.index.astype(int) / 10**9,
        traj_indexed['lat']
    )
    new_traj['lon'] = np.interp(
        new_traj['timestamp'].astype(int) / 10**9,
        traj_indexed.index.astype(int) / 10**9,
        traj_indexed['lon']
    )
    
    # Recalculate features
    new_traj = calculate_trajectory_features(new_traj)
    
    return new_traj

def create_trajectory_segments(ais_df, max_gap_minutes=30, min_points=6):
    """
    Create trajectory segments from AIS data
    
    Args:
        ais_df: DataFrame with MMSI, lat, lon, timestamp columns
        max_gap_minutes: Maximum time gap between points to consider same trajectory
        min_points: Minimum number of points for a valid trajectory
        
    Returns:
        List of DataFrames, each representing a trajectory segment
    """
    # Group by vessel ID
    grouped = ais_df.groupby('mmsi')
    
    trajectories = []
    
    for mmsi, group in grouped:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Calculate time gaps
        time_gaps = group['timestamp'].diff()
        
        # Find gaps larger than threshold
        gap_indices = np.where(time_gaps > timedelta(minutes=max_gap_minutes))[0]
        
        if len(gap_indices) == 0:
            # Single trajectory
            if len(group) >= min_points:
                trajectories.append(group)
        else:
            # Multiple segments
            start_idx = 0
            
            for gap_idx in gap_indices:
                segment = group.iloc[start_idx:gap_idx]
                if len(segment) >= min_points:
                    trajectories.append(segment)
                start_idx = gap_idx
            
            # Last segment
            segment = group.iloc[start_idx:]
            if len(segment) >= min_points:
                trajectories.append(segment)
    
    return trajectories

def discretize_value(value, min_val, max_val, num_bins):
    """
    Discretize a continuous value into a bin index
    
    Args:
        value: Value to discretize
        min_val: Minimum value of range
        max_val: Maximum value of range
        num_bins: Number of bins for discretization
        
    Returns:
        Bin index (0 to num_bins-1)
    """
    if value <= min_val:
        return 0
    if value >= max_val:
        return num_bins - 1
    
    bin_size = (max_val - min_val) / num_bins
    bin_idx = int((value - min_val) / bin_size)
    
    return min(bin_idx, num_bins - 1)

def create_four_hot_encoding(lat, lon, sog, cog, config):
    """
    Create four-hot encoding for a single AIS point
    
    Args:
        lat: Latitude value
        lon: Longitude value
        sog: Speed over ground (knots)
        cog: Course over ground (degrees)
        config: Configuration with bin specifications
        
    Returns:
        Dictionary of one-hot vectors for each attribute
    """
    # Get bin indices
    lat_idx = discretize_value(lat, config.lat_min, config.lat_max, config.lat_bins)
    lon_idx = discretize_value(lon, config.lon_min, config.lon_max, config.lon_bins)
    sog_idx = discretize_value(sog, config.sog_min, config.sog_max, config.sog_bins)
    cog_idx = discretize_value(cog, 0, 360, config.cog_bins)
    
    # Create one-hot vectors
    lat_one_hot = np.zeros(config.lat_bins)
    lat_one_hot[lat_idx] = 1
    
    lon_one_hot = np.zeros(config.lon_bins)
    lon_one_hot[lon_idx] = 1
    
    sog_one_hot = np.zeros(config.sog_bins)
    sog_one_hot[sog_idx] = 1
    
    cog_one_hot = np.zeros(config.cog_bins)
    cog_one_hot[cog_idx] = 1
    
    return {
        'lat': lat_one_hot,
        'lon': lon_one_hot,
        'sog': sog_one_hot,
        'cog': cog_one_hot,
        'indices': (lat_idx, lon_idx, sog_idx, cog_idx)
    }
