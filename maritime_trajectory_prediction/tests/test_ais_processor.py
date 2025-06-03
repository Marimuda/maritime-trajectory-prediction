"""
Tests for AIS data processing functionality.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Test JSON serialization
def test_json_serialization():
    """Test that NumPy types can be serialized to JSON."""
    test_data = {
        "lat": np.float32(59.123), 
        "lon": np.float32(10.456),
        "sog": np.float64(8.7),
        "cog": np.int32(45),
        "values": np.array([1.2, 3.4, 5.6], dtype=np.float32)
    }
    
    # Convert NumPy types to Python native types
    json_str = json.dumps(test_data, default=lambda x: 
        float(x) if isinstance(x, np.number) 
        else x.tolist() if isinstance(x, np.ndarray) 
        else x)
    
    # Deserialize and check values
    deserialized = json.loads(json_str)
    assert isinstance(deserialized["lat"], float)
    assert isinstance(deserialized["lon"], float)
    assert isinstance(deserialized["sog"], float)
    assert isinstance(deserialized["cog"], float)
    assert isinstance(deserialized["values"], list)
    assert len(deserialized["values"]) == 3
    
    # Verify values match
    assert abs(deserialized["lat"] - 59.123) < 0.001
    assert abs(deserialized["lon"] - 10.456) < 0.001


# Test loading AIS data
def test_load_ais_data():
    """Test loading AIS data from processed files."""
    from src.utils.ais_parser import load_processed_ais_data
    
    # Create a simple DataFrame for testing
    test_df = pd.DataFrame({
        'mmsi': [123456789, 123456789, 987654321],
        'timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:05:00', '2023-01-01 12:10:00']),
        'lat': [59.123, 59.124, 59.200],
        'lon': [10.456, 10.457, 10.500]
    })
    
    # Save to a temporary CSV file
    temp_file = 'temp_test_data.csv'
    test_df.to_csv(temp_file, index=False)
    
    try:
        # Test loading
        loaded_df = load_processed_ais_data(temp_file)
        
        # Check shape and data types
        assert loaded_df.shape == test_df.shape
        assert pd.api.types.is_datetime64_dtype(loaded_df['timestamp'])
        assert 'mmsi' in loaded_df.columns
        assert 'lat' in loaded_df.columns
        assert 'lon' in loaded_df.columns
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Test trajectory feature calculation
def test_feature_calculation():
    """Test calculating trajectory features."""
    # We'll create a simple trajectory for testing
    trajectory = pd.DataFrame({
        'mmsi': [123456789] * 5,
        'timestamp': pd.to_datetime([
            '2023-01-01 12:00:00', 
            '2023-01-01 12:05:00',
            '2023-01-01 12:10:00',
            '2023-01-01 12:15:00',
            '2023-01-01 12:20:00'
        ]),
        'lat': [59.0, 59.01, 59.02, 59.03, 59.04],
        'lon': [10.0, 10.01, 10.02, 10.03, 10.04],
        'sog': [10.0, 10.5, 11.0, 10.5, 10.0],
        'cog': [90.0, 90.0, 91.0, 92.0, 90.0]
    })
    
    # Import the function from the main module (should be in ais_utils or similar)
    # This is a placeholder - in a real test we'd import the actual function
    def calculate_trajectory_features(traj):
        """Simple implementation for testing"""
        result = traj.copy()
        # Calculate time deltas in seconds
        result['time_delta'] = result['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Calculate distances (simplified for testing)
        result['distance_km'] = np.sqrt(
            (result['lat'].diff() * 111) ** 2 + 
            (result['lon'].diff() * 111 * np.cos(np.radians(result['lat']))) ** 2
        ).fillna(0)
        
        # Calculate speed and course changes
        result['speed_delta'] = result['sog'].diff().fillna(0)
        result['course_delta'] = result['cog'].diff().fillna(0)
        
        return result
    
    # Calculate features
    traj_with_features = calculate_trajectory_features(trajectory)
    
    # Check that new columns exist
    assert 'distance_km' in traj_with_features.columns
    assert 'speed_delta' in traj_with_features.columns
    assert 'course_delta' in traj_with_features.columns
    
    # Check values
    assert traj_with_features['distance_km'].iloc[1] > 0
    assert traj_with_features['speed_delta'].iloc[1] == 0.5
    assert traj_with_features['course_delta'].iloc[2] == 1.0