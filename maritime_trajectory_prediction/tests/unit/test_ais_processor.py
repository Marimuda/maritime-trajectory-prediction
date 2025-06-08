"""
Comprehensive pytest tests for AIS data processor.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime

from src.data.ais_processor import AISProcessor, load_ais_data, preprocess_ais_data


class TestAISProcessor:
    """Test suite for AISProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create AISProcessor instance for testing."""
        return AISProcessor()
    
    @pytest.fixture
    def sample_ais_line(self):
        """Sample AIS log line for testing."""
        return '2025-05-08 11:10:19,454 - {"class":"AIS","device":"AIS-catcher","version":61,"driver":1,"hardware":"RTL2838UHIDIR","rxtime":"20250508101019","scaled":true,"channel":"B","nmea":["!AIVDM,1,1,,B,13Lh?20P00OPwK4SNhbrq?vV0<0G,0*09"],"signalpower":-15.704244,"ppm":-4.629630,"type":1,"repeat":0,"mmsi":231477000,"status":0,"status_text":"Under way using engine","turn_unscaled":-128,"turn":-128,"speed":0.000000,"accuracy":false,"lon":-6.774024,"lat":62.006901,"course":278.800018,"heading":511,"second":19,"maneuver":0,"raim":false,"radio":49175}'
    
    @pytest.fixture
    def invalid_lines(self):
        """Invalid log lines for testing."""
        return [
            "2025-05-08 11:10:20,199 - [AIS engine v0.61 #0-0] received: 6 msgs",  # Engine chatter
            "Invalid line format",  # No timestamp
            "2025-05-08 11:10:19,454 - {invalid json}",  # Invalid JSON
            "2025-05-08 11:10:19,454 - {}",  # Empty JSON
            "2025-05-08 11:10:19,454 - {\"type\":1}",  # Missing required fields
        ]
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        data = {
            'mmsi': [231477000, 231226000, 257510500],
            'latitude': [62.006901, 62.006180, 61.966694],
            'longitude': [-6.774024, -6.772121, -6.492732],
            'sog': [0.0, 0.0, 9.2],
            'cog': [278.8, 0.0, 15.3],
            'time': pd.to_datetime(['2025-05-08 10:10:19', '2025-05-08 10:10:20', '2025-05-08 10:10:21'], utc=True),
            'msg_class': ['A_pos', 'A_pos', 'A_pos']
        }
        return pd.DataFrame(data)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert hasattr(processor, 'pattern')
        assert hasattr(processor, 'sentinel_values')
        assert hasattr(processor, 'field_mapping')
        assert hasattr(processor, 'msg_classes')
        assert processor.stats['lines_processed'] == 0
    
    def test_parse_valid_line(self, processor, sample_ais_line):
        """Test parsing a valid AIS line."""
        result = processor.parse_line(sample_ais_line)
        
        assert result is not None
        assert isinstance(result, dict)
        assert result['mmsi'] == 231477000
        assert result['type'] == 1
        assert result['msg_class'] == 'A_pos'
        assert 'time' in result
        assert 'latitude' in result  # CF-compliant field name
        assert 'longitude' in result  # CF-compliant field name
        assert processor.stats['valid_records'] == 1
    
    def test_parse_invalid_lines(self, processor, invalid_lines):
        """Test parsing invalid lines."""
        for line in invalid_lines:
            result = processor.parse_line(line)
            assert result is None
        
        # Check that stats are updated correctly
        assert processor.stats['lines_processed'] == len(invalid_lines)
        assert processor.stats['valid_records'] == 0
        assert processor.stats['filtered_records'] + processor.stats['error_records'] == len(invalid_lines)
    
    def test_field_mapping(self, processor):
        """Test CF-compliant field mapping."""
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":62.0,"lon":-6.7,"speed":5.0,"course":90.0}'
        result = processor.parse_line(line)
        
        assert result is not None
        assert 'latitude' in result
        assert 'longitude' in result
        assert 'sog' in result
        assert 'cog' in result
        assert 'lat' not in result  # Original field should be removed
        assert 'lon' not in result
        assert 'speed' not in result
        assert 'course' not in result
    
    def test_sentinel_value_handling(self, processor):
        """Test sentinel value handling."""
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":91.0,"lon":181.0,"speed":102.3,"heading":511}'
        result = processor.parse_line(line)
        
        assert result is not None
        assert pd.isna(result['latitude'])  # lat=91 should become NaN
        assert pd.isna(result['longitude'])  # lon=181 should become NaN
        assert pd.isna(result['sog'])  # speed=102.3 should become NaN
        assert pd.isna(result['heading'])  # heading=511 should become NaN
    
    def test_mmsi_validation(self, processor):
        """Test MMSI validation."""
        # Valid MMSIs
        valid_mmsis = [231477000, 2311500, 992310001]  # Vessel, base station, ATON
        
        for mmsi in valid_mmsis:
            line = f'2025-05-08 11:10:19,454 - {{"rxtime":"20250508101019","type":1,"mmsi":{mmsi},"lat":62.0,"lon":-6.7}}'
            result = processor.parse_line(line)
            assert result is not None, f"Valid MMSI {mmsi} should be accepted"
        
        # Invalid MMSI
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":99999999,"lat":62.0,"lon":-6.7}'
        result = processor.parse_line(line)
        assert result is None, "Invalid MMSI should be rejected"
    
    def test_range_validation(self, processor):
        """Test coordinate and value range validation."""
        # Invalid latitude
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":95.0,"lon":-6.7}'
        result = processor.parse_line(line)
        assert result is not None
        assert pd.isna(result['latitude'])
        
        # Invalid longitude
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":62.0,"lon":185.0}'
        result = processor.parse_line(line)
        assert result is not None
        assert pd.isna(result['longitude'])
        
        # Invalid speed
        line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":62.0,"lon":-6.7,"speed":105.0}'
        result = processor.parse_line(line)
        assert result is not None
        assert pd.isna(result['sog'])
    
    def test_message_classification(self, processor):
        """Test message type classification."""
        test_cases = [
            (1, "A_pos"),
            (2, "A_pos"),
            (3, "A_pos"),
            (4, "Base_pos"),
            (5, "Static"),
            (18, "B_pos"),
            (21, "ATON"),
            (24, "StaticB"),
            (99, "Other")
        ]
        
        for msg_type, expected_class in test_cases:
            line = f'2025-05-08 11:10:19,454 - {{"rxtime":"20250508101019","type":{msg_type},"mmsi":231477000,"lat":62.0,"lon":-6.7}}'
            result = processor.parse_line(line)
            assert result is not None
            assert result['msg_class'] == expected_class
    
    def test_process_file_with_real_data(self, processor):
        """Test processing real log file."""
        log_path = Path("data/raw/log_snipit.log")
        if log_path.exists():
            df = processor.process_file(log_path)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'mmsi' in df.columns
            assert 'time' in df.columns
            assert 'latitude' in df.columns
            assert 'longitude' in df.columns
            assert 'msg_class' in df.columns
            
            # Check statistics
            stats = processor.get_statistics()
            assert stats['lines_processed'] > 0
            assert stats['valid_records'] > 0
            assert stats['valid_records'] == len(df)
    
    def test_process_empty_file(self, processor):
        """Test processing empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            df = processor.process_file(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_clean_ais_data(self, processor, sample_dataframe):
        """Test AIS data cleaning."""
        # Add some invalid data
        invalid_df = sample_dataframe.copy()
        invalid_df.loc[len(invalid_df)] = {
            'mmsi': np.nan,
            'latitude': np.nan,
            'longitude': np.nan,
            'sog': 5.0,
            'cog': 90.0,
            'time': pd.Timestamp.now(tz='UTC'),
            'msg_class': 'A_pos'
        }
        invalid_df.loc[len(invalid_df)] = {
            'mmsi': 99999999,  # Invalid MMSI
            'latitude': 62.0,
            'longitude': -6.7,
            'sog': 5.0,
            'cog': 90.0,
            'time': pd.Timestamp.now(tz='UTC'),
            'msg_class': 'A_pos'
        }
        
        cleaned_df = processor.clean_ais_data(invalid_df)
        
        assert len(cleaned_df) == len(sample_dataframe)  # Only original valid records
        assert not cleaned_df['latitude'].isna().any()
        assert not cleaned_df['longitude'].isna().any()
        assert cleaned_df['mmsi'].apply(lambda x: (100_000_000 <= x <= 799_999_999) or 
                                                  (2_000_000 <= x <= 9_999_999) or 
                                                  (990_000_000 <= x <= 999_999_999)).all()
    
    def test_get_statistics(self, processor, sample_ais_line):
        """Test statistics tracking."""
        initial_stats = processor.get_statistics()
        assert all(v == 0 for v in initial_stats.values())
        
        processor.parse_line(sample_ais_line)
        stats = processor.get_statistics()
        
        assert stats['lines_processed'] == 1
        assert stats['valid_records'] == 1
        assert stats['filtered_records'] == 0
        assert stats['error_records'] == 0


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_load_ais_data_function(self):
        """Test load_ais_data function."""
        log_path = Path("data/raw/log_snipit.log")
        if log_path.exists():
            df = load_ais_data(log_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_preprocess_ais_data_function(self):
        """Test preprocess_ais_data function."""
        # Create sample data with some invalid records
        data = {
            'mmsi': [231477000, np.nan, 99999999],
            'latitude': [62.0, np.nan, 62.1],
            'longitude': [-6.7, np.nan, -6.8],
            'sog': [5.0, 3.0, 2.0],
            'time': pd.to_datetime(['2025-05-08 10:10:19', '2025-05-08 10:10:20', '2025-05-08 10:10:21'], utc=True)
        }
        df = pd.DataFrame(data)
        
        cleaned_df = preprocess_ais_data(df)
        
        assert len(cleaned_df) == 1  # Only first record should remain
        assert cleaned_df.iloc[0]['mmsi'] == 231477000
    
    def test_preprocess_empty_dataframe(self):
        """Test preprocessing empty DataFrame."""
        empty_df = pd.DataFrame()
        result = preprocess_ais_data(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

