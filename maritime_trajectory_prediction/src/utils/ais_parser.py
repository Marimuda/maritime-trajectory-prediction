import json
import pandas as pd
import os
from pathlib import Path

def load_processed_ais_data(file_path):
    """
    Load processed AIS data from Parquet, CSV or JSONL files.
    
    Args:
        file_path: Path to the processed AIS data file
        
    Returns:
        DataFrame with processed AIS data or list of trajectories
    """
    # Check file extension
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.parquet':
        # Load Parquet file
        df = pd.read_parquet(file_path)
        return df
    
    elif extension == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    elif extension == '.jsonl':
        # Load JSONL file as list of trajectories
        trajectories = []
        
        with open(file_path, 'r') as f:
            for line in f:
                vessel_data = json.loads(line.strip())
                
                mmsi = vessel_data.get('mmsi')
                static_data = vessel_data.get('static', {})
                trajectory_points = vessel_data.get('trajectory', [])
                
                if not trajectory_points:
                    continue
                
                # Convert trajectory points to DataFrame
                traj_df = pd.DataFrame(trajectory_points)
                
                # Convert timestamp to datetime
                if 'timestamp' in traj_df.columns:
                    traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
                
                # Add MMSI
                traj_df['mmsi'] = mmsi
                
                # Add static data
                for key, value in static_data.items():
                    traj_df[key] = value
                
                trajectories.append(traj_df)
        
        return trajectories
    
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def get_vessel_trajectories(ais_data, mmsi=None):
    """
    Extract vessel trajectories from processed AIS data.
    
    Args:
        ais_data: DataFrame with processed AIS data or list of trajectories
        mmsi: Optional MMSI to filter for a specific vessel
        
    Returns:
        List of trajectory DataFrames
    """
    # If already a list of trajectories
    if isinstance(ais_data, list):
        trajectories = ais_data
        
        # Filter by MMSI if specified
        if mmsi is not None:
            trajectories = [t for t in trajectories if t['mmsi'].iloc[0] == mmsi]
        
        return trajectories
    
    # If DataFrame, group by MMSI
    trajectories = []
    
    if 'mmsi' not in ais_data.columns:
        raise ValueError("AIS data must contain an 'mmsi' column")
    
    # Filter by MMSI if specified
    if mmsi is not None:
        vessel_data = ais_data[ais_data['mmsi'] == mmsi]
        
        # Group by trajectory segment if available
        if 'segment_id' in vessel_data.columns:
            for segment_id, segment in vessel_data.groupby('segment_id'):
                trajectories.append(segment.copy())
        else:
            trajectories.append(vessel_data.copy())
    else:
        # Group by MMSI
        for vessel_mmsi, vessel_data in ais_data.groupby('mmsi'):
            # Group by trajectory segment if available
            if 'segment_id' in vessel_data.columns:
                for segment_id, segment in vessel_data.groupby('segment_id'):
                    trajectories.append(segment.copy())
            else:
                trajectories.append(vessel_data.copy())
    
    return trajectories

def prepare_trajectory_for_prediction(trajectory, sequence_length=20, features=None):
    """
    Prepare a trajectory for prediction by extracting a sequence of the specified length.
    
    Args:
        trajectory: DataFrame with a vessel trajectory
        sequence_length: Number of points to include in the sequence
        features: List of features to include (defaults to position and motion features)
        
    Returns:
        numpy array with the sequence data
    """
    if len(trajectory) < sequence_length:
        raise ValueError(f"Trajectory has fewer than {sequence_length} points")
    
    # Default features if not specified
    if features is None:
        features = ['lat', 'lon', 'sog', 'cog', 'distance_km', 'speed_delta', 'course_delta']
    
    # Get the latest sequence_length points
    sequence = trajectory.iloc[-sequence_length:].copy()
    
    # Extract features
    feature_data = sequence[features].values
    
    return feature_data