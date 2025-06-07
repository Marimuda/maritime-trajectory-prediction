"""
Updated tests for AIS data processing functionality.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import from the refactored package structure
from maritime_trajectory_prediction.src.utils.ais_parser import AISParser
from maritime_trajectory_prediction.src.utils.maritime_utils import MaritimeUtils
from maritime_trajectory_prediction.src.data.ais_processor import AISProcessor


class TestJSONSerialization:
    """Test JSON serialization functionality."""
    
    def test_json_serialization(self):
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


class TestAISParser:
    """Test AIS parser functionality."""
    
    def test_load_csv_data(self, tmp_path):
        """Test loading AIS data from CSV files."""
        parser = AISParser()
        
        # Create a simple DataFrame for testing
        test_df = pd.DataFrame({
            'mmsi': [123456789, 123456789, 987654321],
            'timestamp': ['2023-01-01 12:00:00', '2023-01-01 12:05:00', '2023-01-01 12:10:00'],
            'lat': [59.123, 59.124, 59.200],
            'lon': [10.456, 10.457, 10.500]
        })
        
        # Save to a temporary CSV file
        temp_file = tmp_path / 'test_data.csv'
        test_df.to_csv(temp_file, index=False)
        
        # Test loading
        loaded_df = parser.load_processed_ais_data(temp_file)
        
        # Check shape and data types
        assert loaded_df.shape == test_df.shape
        assert pd.api.types.is_datetime64_dtype(loaded_df['timestamp'])
        assert 'mmsi' in loaded_df.columns
        assert 'lat' in loaded_df.columns
        assert 'lon' in loaded_df.columns
    
    def test_get_vessel_trajectories(self):
        """Test extracting vessel trajectories."""
        parser = AISParser()
        
        # Create test data
        test_df = pd.DataFrame({
            'mmsi': [123456789, 123456789, 987654321, 987654321],
            'timestamp': pd.to_datetime([
                '2023-01-01 12:00:00', '2023-01-01 12:05:00',
                '2023-01-01 12:10:00', '2023-01-01 12:15:00'
            ]),
            'lat': [59.0, 59.01, 59.1, 59.11],
            'lon': [10.0, 10.01, 10.1, 10.11]
        })
        
        # Get trajectories
        trajectories = parser.get_vessel_trajectories(test_df)
        
        # Should have 2 trajectories (one per MMSI)
        assert len(trajectories) == 2
        assert all(len(traj) == 2 for traj in trajectories)


class TestMaritimeUtils:
    """Test maritime utility functions."""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test known distance (approximately)
        lat1, lon1 = 59.0, 10.0
        lat2, lon2 = 59.01, 10.01
        
        distance = MaritimeUtils.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Should be approximately 1.3 km
        assert 1.0 < distance < 2.0
    
    def test_calculate_distances(self):
        """Test distance calculation for trajectory."""
        lats = np.array([59.0, 59.01, 59.02])
        lons = np.array([10.0, 10.01, 10.02])
        
        distances = MaritimeUtils.calculate_distances(lats, lons)
        
        # First distance should be 0
        assert distances[0] == 0
        # Other distances should be positive
        assert all(d > 0 for d in distances[1:])
    
    def test_calculate_bearing(self):
        """Test bearing calculation."""
        lat1, lon1 = 59.0, 10.0
        lat2, lon2 = 59.01, 10.01
        
        bearing = MaritimeUtils.calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Should be between 0 and 360
        assert 0 <= bearing <= 360
    
    def test_is_valid_position(self):
        """Test position validation."""
        assert MaritimeUtils.is_valid_position(59.0, 10.0)
        assert not MaritimeUtils.is_valid_position(91.0, 10.0)  # Invalid latitude
        assert not MaritimeUtils.is_valid_position(59.0, 181.0)  # Invalid longitude
    
    def test_is_valid_speed(self):
        """Test speed validation."""
        assert MaritimeUtils.is_valid_speed(10.0)
        assert not MaritimeUtils.is_valid_speed(-1.0)  # Negative speed
        assert not MaritimeUtils.is_valid_speed(100.0)  # Too fast


class TestAISProcessor:
    """Test AIS data processor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = AISProcessor()
        
        assert processor.min_points_per_trajectory == 10
        assert processor.max_time_gap_hours == 6.0
        assert processor.min_speed_knots == 0.1
        assert processor.max_speed_knots == 50.0
    
    def test_clean_ais_data(self):
        """Test data cleaning functionality."""
        processor = AISProcessor()
        
        # Create test data with some invalid entries
        test_df = pd.DataFrame({
            'mmsi': [123456789, 123456789, 123456789, 123456789],
            'lat': [59.0, 91.0, 59.02, 59.03],  # One invalid latitude
            'lon': [10.0, 10.01, 181.0, 10.03],  # One invalid longitude
            'sog': [10.0, 10.5, 100.0, 10.0],  # One invalid speed
            'cog': [90.0, 90.0, 91.0, 92.0]
        })
        
        cleaned_df = processor.clean_ais_data(test_df)
        
        # Should have only 2 valid records (first and last)
        # Second has invalid lat, third has invalid lon and speed
        assert len(cleaned_df) == 2
        assert cleaned_df.iloc[0]['lat'] == 59.0
        assert cleaned_df.iloc[1]['lat'] == 59.03
    
    def test_get_trajectory_sequences(self):
        """Test trajectory sequence extraction."""
        processor = AISProcessor()
        
        # Create test data with enough points
        test_df = pd.DataFrame({
            'mmsi': [123456789] * 30,
            'segment_id': [0] * 30,
            'lat': np.linspace(59.0, 59.1, 30),
            'lon': np.linspace(10.0, 10.1, 30),
            'sog': [10.0] * 30,
            'cog': [90.0] * 30
        })
        
        sequences = processor.get_trajectory_sequences(
            test_df, sequence_length=10, prediction_horizon=5
        )
        
        # Should have sequences
        assert len(sequences) > 0
        
        # Check sequence structure
        seq = sequences[0]
        assert 'input_sequence' in seq
        assert 'target_sequence' in seq
        assert 'mmsi' in seq
        assert 'segment_id' in seq
        
        # Check sequence lengths
        assert len(seq['input_sequence']) == 10
        assert len(seq['target_sequence']) == 5

