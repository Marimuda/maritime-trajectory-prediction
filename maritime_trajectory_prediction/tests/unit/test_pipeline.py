"""
Comprehensive tests for the data pipeline components.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from maritime_trajectory_prediction.src.data.pipeline import (
    DataPipeline, BaseDatasetBuilder, DatasetConfig, DatasetMetadata, DatasetFormat
)
from maritime_trajectory_prediction.src.data.builders import (
    TrajectoryPredictionBuilder, AnomalyDetectionBuilder, 
    GraphNetworkBuilder, CollisionAvoidanceBuilder
)
from maritime_trajectory_prediction.src.data.validation import (
    DataValidator, QualityChecker, DatasetExporter, ValidationResult
)
from maritime_trajectory_prediction.src.data.multi_task_processor import MLTask, AISMultiTaskProcessor


class TestDatasetConfig:
    """Test DatasetConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig(task=MLTask.TRAJECTORY_PREDICTION)
        
        assert config.task == MLTask.TRAJECTORY_PREDICTION
        assert config.sequence_length == 10
        assert config.prediction_horizon == 5
        assert config.min_trajectory_length == 20
        assert config.validation_split == 0.2
        assert config.test_split == 0.1
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=15,
            prediction_horizon=3,
            min_trajectory_length=30,
            spatial_bounds={'min_lat': 60.0, 'max_lat': 65.0},
            vessel_types=[30, 31, 32]
        )
        
        assert config.task == MLTask.ANOMALY_DETECTION
        assert config.sequence_length == 15
        assert config.prediction_horizon == 3
        assert config.spatial_bounds['min_lat'] == 60.0
        assert config.vessel_types == [30, 31, 32]


class TestBaseDatasetBuilder:
    """Test BaseDatasetBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample AIS data for testing."""
        np.random.seed(42)
        n_records = 100
        n_vessels = 5
        
        data = []
        base_time = datetime(2024, 1, 1)
        
        for vessel_id in range(n_vessels):
            mmsi = 123456000 + vessel_id
            for i in range(n_records // n_vessels):
                data.append({
                    'mmsi': mmsi,
                    'time': base_time + timedelta(minutes=i),
                    'latitude': 62.0 + np.random.normal(0, 0.01),
                    'longitude': -6.5 + np.random.normal(0, 0.01),
                    'sog': np.random.uniform(0, 20),
                    'cog': np.random.uniform(0, 360),
                    'heading': np.random.uniform(0, 360),
                    'turn': np.random.uniform(-10, 10),
                    'status': np.random.choice([0, 1, 2, 3]),
                    'shiptype': np.random.choice([30, 31, 32])
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=5,
            prediction_horizon=2,
            min_trajectory_length=10
        )
    
    def test_validate_data_valid(self, sample_data, config):
        """Test data validation with valid data."""
        builder = TrajectoryPredictionBuilder(config)
        assert builder.validate_data(sample_data) is True
    
    def test_validate_data_empty(self, config):
        """Test data validation with empty DataFrame."""
        builder = TrajectoryPredictionBuilder(config)
        empty_df = pd.DataFrame()
        assert builder.validate_data(empty_df) is False
    
    def test_validate_data_missing_columns(self, config):
        """Test data validation with missing required columns."""
        builder = TrajectoryPredictionBuilder(config)
        incomplete_df = pd.DataFrame({'mmsi': [123456789], 'time': [datetime.now()]})
        assert builder.validate_data(incomplete_df) is False
    
    def test_filter_data_spatial(self, sample_data, config):
        """Test spatial filtering."""
        config.spatial_bounds = {
            'min_lat': 61.9, 'max_lat': 62.1,
            'min_lon': -6.6, 'max_lon': -6.4
        }
        
        builder = TrajectoryPredictionBuilder(config)
        filtered_df = builder.filter_data(sample_data)
        
        assert len(filtered_df) <= len(sample_data)
        assert filtered_df['latitude'].min() >= 61.9
        assert filtered_df['latitude'].max() <= 62.1
    
    def test_filter_data_temporal(self, sample_data, config):
        """Test temporal filtering."""
        mid_time = sample_data['time'].median()
        config.temporal_bounds = {
            'start_time': mid_time,
            'end_time': sample_data['time'].max()
        }
        
        builder = TrajectoryPredictionBuilder(config)
        filtered_df = builder.filter_data(sample_data)
        
        assert len(filtered_df) <= len(sample_data)
        assert filtered_df['time'].min() >= mid_time
    
    def test_resample_trajectories(self, sample_data, config):
        """Test trajectory resampling."""
        builder = TrajectoryPredictionBuilder(config)
        resampled_df = builder.resample_trajectories(sample_data)
        
        assert len(resampled_df) > 0
        assert 'mmsi' in resampled_df.columns
        
        # Check that each vessel has consistent time intervals
        for mmsi, vessel_df in resampled_df.groupby('mmsi'):
            if len(vessel_df) > 1:
                time_diffs = vessel_df['time'].diff().dropna()
                # Should be approximately 1 minute intervals
                assert all(abs(td.total_seconds() - 60) < 10 for td in time_diffs)
    
    def test_split_dataset(self, config):
        """Test dataset splitting."""
        builder = TrajectoryPredictionBuilder(config)
        
        # Create dummy data
        X = np.random.rand(100, 10, 5)
        y = np.random.rand(100, 5, 2)
        
        splits = builder.split_dataset(X, y)
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        train_X, train_y = splits['train']
        val_X, val_y = splits['validation']
        test_X, test_y = splits['test']
        
        # Check shapes
        assert train_X.shape[0] == train_y.shape[0]
        assert val_X.shape[0] == val_y.shape[0]
        assert test_X.shape[0] == test_y.shape[0]
        
        # Check total samples
        total_samples = train_X.shape[0] + val_X.shape[0] + test_X.shape[0]
        assert total_samples == 100


class TestTrajectoryPredictionBuilder:
    """Test TrajectoryPredictionBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample trajectory data."""
        np.random.seed(42)
        data = []
        base_time = datetime(2024, 1, 1)
        
        # Create one vessel with sufficient data
        mmsi = 123456789
        for i in range(50):
            data.append({
                'mmsi': mmsi,
                'time': base_time + timedelta(minutes=i),
                'latitude': 62.0 + i * 0.001,
                'longitude': -6.5 + i * 0.001,
                'sog': 10 + np.random.normal(0, 1),
                'cog': 45 + np.random.normal(0, 5),
                'heading': 45 + np.random.normal(0, 5),
                'turn': np.random.normal(0, 2)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=10,
            prediction_horizon=5,
            min_trajectory_length=20
        )
    
    def test_build_features(self, sample_data, config):
        """Test feature building."""
        builder = TrajectoryPredictionBuilder(config)
        features_df = builder.build_features(sample_data)
        
        assert len(features_df) == len(sample_data)
        assert 'mmsi' in features_df.columns
        assert 'time' in features_df.columns
        assert 'latitude' in features_df.columns
        assert 'longitude' in features_df.columns
        
        # Check for derived features
        derived_cols = [col for col in features_df.columns if col.startswith(('temporal_', 'movement_', 'spatial_'))]
        assert len(derived_cols) > 0
    
    def test_build_targets(self, sample_data, config):
        """Test target building."""
        builder = TrajectoryPredictionBuilder(config)
        targets_df = builder.build_targets(sample_data)
        
        assert len(targets_df) == len(sample_data)
        assert 'mmsi' in targets_df.columns
        assert 'time' in targets_df.columns
        assert 'latitude' in targets_df.columns
        assert 'longitude' in targets_df.columns
    
    def test_create_sequences(self, sample_data, config):
        """Test sequence creation."""
        builder = TrajectoryPredictionBuilder(config)
        features_df = builder.build_features(sample_data)
        
        X, y = builder.create_sequences(features_df)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == config.sequence_length
        assert y.shape[1] == config.prediction_horizon
    
    def test_build_dataset_complete(self, sample_data, config):
        """Test complete dataset building."""
        builder = TrajectoryPredictionBuilder(config)
        dataset = builder.build_dataset(sample_data)
        
        assert 'splits' in dataset
        assert 'metadata' in dataset
        assert 'features_df' in dataset
        assert 'targets_df' in dataset
        
        # Check metadata
        metadata = dataset['metadata']
        assert metadata.task == MLTask.TRAJECTORY_PREDICTION.value
        assert metadata.num_samples > 0
        assert metadata.num_vessels > 0


class TestAnomalyDetectionBuilder:
    """Test AnomalyDetectionBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with potential anomalies."""
        np.random.seed(42)
        data = []
        base_time = datetime(2024, 1, 1)
        
        mmsi = 123456789
        for i in range(30):
            # Add some anomalous data points
            if i == 10:
                # Speed anomaly
                sog = 50  # Very high speed
            elif i == 20:
                # Course anomaly
                cog = 180 if i == 19 else 90  # Sudden 90-degree turn
            else:
                sog = 10 + np.random.normal(0, 1)
                cog = 45 + np.random.normal(0, 5)
            
            data.append({
                'mmsi': mmsi,
                'time': base_time + timedelta(minutes=i),
                'latitude': 62.0 + i * 0.001,
                'longitude': -6.5 + i * 0.001,
                'sog': sog,
                'cog': cog,
                'heading': cog + np.random.normal(0, 2),
                'turn': np.random.normal(0, 2),
                'status': 0,
                'shiptype': 30
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=5,
            min_trajectory_length=10
        )
    
    def test_build_features(self, sample_data, config):
        """Test anomaly detection feature building."""
        builder = AnomalyDetectionBuilder(config)
        features_df = builder.build_features(sample_data)
        
        assert len(features_df) == len(sample_data)
        
        # Check for behavioral features
        behavioral_cols = [col for col in features_df.columns if col.startswith('behavioral_')]
        assert len(behavioral_cols) > 0
        
        # Check for statistical features
        statistical_cols = [col for col in features_df.columns if col.startswith('statistical_')]
        assert len(statistical_cols) > 0
    
    def test_build_targets(self, sample_data, config):
        """Test anomaly target building."""
        builder = AnomalyDetectionBuilder(config)
        targets_df = builder.build_targets(sample_data)
        
        assert len(targets_df) == len(sample_data)
        assert 'anomaly_speed' in targets_df.columns
        assert 'anomaly_course' in targets_df.columns
        assert 'anomaly_position' in targets_df.columns
        assert 'anomaly_overall' in targets_df.columns
        
        # Should detect the speed anomaly we inserted
        assert targets_df['anomaly_speed'].sum() > 0
    
    def test_detect_speed_anomalies(self, sample_data, config):
        """Test speed anomaly detection."""
        builder = AnomalyDetectionBuilder(config)
        speed_anomalies = builder._detect_speed_anomalies(sample_data)
        
        # Should detect the high speed we inserted
        assert speed_anomalies.sum() > 0
    
    def test_create_sequences(self, sample_data, config):
        """Test anomaly detection sequence creation."""
        builder = AnomalyDetectionBuilder(config)
        features_df = builder.build_features(sample_data)
        
        X, y = builder.create_sequences(features_df)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == config.sequence_length


class TestDataValidator:
    """Test DataValidator functionality."""
    
    @pytest.fixture
    def valid_data(self):
        """Create valid AIS data."""
        return pd.DataFrame({
            'mmsi': [123456789, 987654321],
            'time': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
            'latitude': [62.0, 62.1],
            'longitude': [-6.5, -6.4],
            'sog': [10.0, 12.0],
            'cog': [45.0, 50.0],
            'heading': [45, 50],
            'turn': [0, 1]
        })
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid AIS data."""
        return pd.DataFrame({
            'mmsi': [123456789, 999999999],
            'time': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
            'latitude': [95.0, 62.1],  # Invalid latitude
            'longitude': [-6.5, -200.0],  # Invalid longitude
            'sog': [50.0, 12.0],  # Invalid speed
            'cog': [400.0, 50.0],  # Invalid course
            'heading': [45, 50],
            'turn': [0, 1]
        })
    
    def test_validate_valid_data(self, valid_data):
        """Test validation with valid data."""
        validator = DataValidator(strict_mode=True)
        result = validator.validate_dataset(valid_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert 'total_records' in result.metrics
        assert result.metrics['total_records'] == 2
    
    def test_validate_invalid_data(self, invalid_data):
        """Test validation with invalid data."""
        validator = DataValidator(strict_mode=True)
        result = validator.validate_dataset(invalid_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Should detect coordinate errors
        coordinate_errors = [err for err in result.errors if 'latitude' in err or 'longitude' in err]
        assert len(coordinate_errors) > 0
    
    def test_validate_empty_data(self):
        """Test validation with empty data."""
        validator = DataValidator(strict_mode=True)
        empty_df = pd.DataFrame()
        result = validator.validate_dataset(empty_df)
        
        assert result.is_valid is False
        assert any('empty' in err.lower() for err in result.errors)
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        validator = DataValidator(strict_mode=True)
        incomplete_df = pd.DataFrame({'mmsi': [123456789]})
        result = validator.validate_dataset(incomplete_df)
        
        assert result.is_valid is False
        assert any('missing' in err.lower() for err in result.errors)
    
    def test_validate_maritime_data(self, invalid_data):
        """Test maritime-specific validation."""
        validator = DataValidator()
        result = validator._validate_maritime_data(invalid_data)
        
        assert len(result['errors']) > 0
        assert len(result['warnings']) > 0
        assert 'invalid_coordinates' in result['metrics']
    
    def test_is_valid_mmsi(self):
        """Test MMSI validation."""
        validator = DataValidator()
        
        assert validator._is_valid_mmsi(123456789) is True
        assert validator._is_valid_mmsi(999999999) is True
        assert validator._is_valid_mmsi(12345678) is False  # Too short
        assert validator._is_valid_mmsi(1234567890) is False  # Too long
        assert validator._is_valid_mmsi('invalid') is False
        assert validator._is_valid_mmsi(None) is False


class TestQualityChecker:
    """Test QualityChecker functionality."""
    
    @pytest.fixture
    def trajectory_data(self):
        """Create trajectory data for quality checking."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        # Vessel 1: Good quality trajectory
        for i in range(100):
            data.append({
                'mmsi': 123456789,
                'time': base_time + timedelta(minutes=i),
                'latitude': 62.0 + i * 0.001,
                'longitude': -6.5 + i * 0.001,
                'sog': 10 + np.random.normal(0, 1),
                'cog': 45 + np.random.normal(0, 2),
                'accuracy': 1
            })
        
        # Vessel 2: Short trajectory
        for i in range(5):
            data.append({
                'mmsi': 987654321,
                'time': base_time + timedelta(minutes=i),
                'latitude': 61.0 + i * 0.001,
                'longitude': -7.0 + i * 0.001,
                'sog': 15,
                'cog': 90,
                'accuracy': 2
            })
        
        return pd.DataFrame(data)
    
    def test_check_trajectory_quality(self, trajectory_data):
        """Test trajectory quality checking."""
        checker = QualityChecker()
        quality_metrics = checker.check_trajectory_quality(trajectory_data)
        
        assert 'num_trajectories' in quality_metrics
        assert quality_metrics['num_trajectories'] == 2
        assert 'avg_trajectory_length' in quality_metrics
        assert 'short_trajectories' in quality_metrics
        assert quality_metrics['short_trajectories'] == 1  # One short trajectory
    
    def test_check_coverage_quality(self, trajectory_data):
        """Test coverage quality checking."""
        checker = QualityChecker()
        coverage_metrics = checker.check_coverage_quality(trajectory_data)
        
        assert 'spatial_extent_deg2' in coverage_metrics
        assert 'temporal_extent_hours' in coverage_metrics
        assert 'vessels_per_deg2' in coverage_metrics
        assert 'records_per_hour' in coverage_metrics


class TestDatasetExporter:
    """Test DatasetExporter functionality."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for export testing."""
        # Create dummy data
        X_train = np.random.rand(50, 10, 5)
        y_train = np.random.rand(50, 5, 2)
        X_val = np.random.rand(20, 10, 5)
        y_val = np.random.rand(20, 5, 2)
        X_test = np.random.rand(10, 10, 5)
        y_test = np.random.rand(10, 5, 2)
        
        splits = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        metadata = DatasetMetadata(
            task='trajectory_prediction',
            config=DatasetConfig(task=MLTask.TRAJECTORY_PREDICTION),
            num_samples=80,
            num_vessels=5,
            temporal_range=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            spatial_bounds={'min_lat': 60.0, 'max_lat': 65.0, 'min_lon': -8.0, 'max_lon': -5.0},
            feature_columns=['lat', 'lon', 'sog', 'cog', 'heading'],
            target_columns=['lat', 'lon']
        )
        
        return {
            'splits': splits,
            'metadata': metadata
        }
    
    def test_export_to_parquet(self, sample_dataset):
        """Test Parquet export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DatasetExporter(Path(temp_dir))
            exported_files = exporter.export_to_parquet(sample_dataset, "test_dataset")
            
            assert 'train_features' in exported_files
            assert 'train_targets' in exported_files
            assert 'validation_features' in exported_files
            assert 'validation_targets' in exported_files
            assert 'test_features' in exported_files
            assert 'test_targets' in exported_files
            assert 'metadata' in exported_files
            
            # Check files exist
            for file_path in exported_files.values():
                assert file_path.exists()
    
    def test_export_to_zarr(self, sample_dataset):
        """Test Zarr export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = DatasetExporter(Path(temp_dir))
            exported_files = exporter.export_to_zarr(sample_dataset, "test_dataset")
            
            assert 'train' in exported_files
            assert 'validation' in exported_files
            assert 'test' in exported_files
            assert 'metadata' in exported_files
            
            # Check files exist
            for file_path in exported_files.values():
                assert file_path.exists()


class TestDataPipeline:
    """Test DataPipeline integration."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create mock AIS processor."""
        processor = Mock(spec=AISMultiTaskProcessor)
        
        # Mock processed data
        sample_data = pd.DataFrame({
            'mmsi': [123456789] * 50,
            'time': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)],
            'latitude': [62.0 + i * 0.001 for i in range(50)],
            'longitude': [-6.5 + i * 0.001 for i in range(50)],
            'sog': [10 + np.random.normal(0, 1) for _ in range(50)],
            'cog': [45 + np.random.normal(0, 2) for _ in range(50)],
            'heading': [45 + np.random.normal(0, 2) for _ in range(50)],
            'turn': [np.random.normal(0, 1) for _ in range(50)]
        })
        
        processor.process_file.return_value = sample_data
        processor.get_task_specific_dataset.return_value = sample_data
        
        return processor
    
    def test_pipeline_initialization(self, mock_processor):
        """Test pipeline initialization."""
        pipeline = DataPipeline(mock_processor)
        
        assert pipeline.processor == mock_processor
        assert len(pipeline.builders) == 0
    
    def test_register_builder(self, mock_processor):
        """Test builder registration."""
        pipeline = DataPipeline(mock_processor)
        pipeline.register_builder(MLTask.TRAJECTORY_PREDICTION, TrajectoryPredictionBuilder)
        
        assert MLTask.TRAJECTORY_PREDICTION in pipeline.builders
        assert pipeline.builders[MLTask.TRAJECTORY_PREDICTION] == TrajectoryPredictionBuilder
    
    def test_process_raw_data(self, mock_processor):
        """Test raw data processing."""
        pipeline = DataPipeline(mock_processor)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_file:
            temp_file.write('{"mmsi": 123456789, "time": "2024-01-01T00:00:00"}\n')
            temp_file_path = temp_file.name
        
        try:
            df = pipeline.process_raw_data(temp_file_path)
            assert len(df) > 0
            mock_processor.process_file.assert_called_once_with(temp_file_path)
        finally:
            Path(temp_file_path).unlink()
    
    def test_build_dataset(self, mock_processor):
        """Test dataset building."""
        pipeline = DataPipeline(mock_processor)
        pipeline.register_builder(MLTask.TRAJECTORY_PREDICTION, TrajectoryPredictionBuilder)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'mmsi': [123456789] * 50,
            'time': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)],
            'latitude': [62.0 + i * 0.001 for i in range(50)],
            'longitude': [-6.5 + i * 0.001 for i in range(50)],
            'sog': [10 + np.random.normal(0, 1) for _ in range(50)],
            'cog': [45 + np.random.normal(0, 2) for _ in range(50)],
            'heading': [45 + np.random.normal(0, 2) for _ in range(50)],
            'turn': [np.random.normal(0, 1) for _ in range(50)]
        })
        
        config = DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=5,
            prediction_horizon=2,
            min_trajectory_length=10
        )
        
        dataset = pipeline.build_dataset(sample_data, MLTask.TRAJECTORY_PREDICTION, config)
        
        assert 'splits' in dataset
        assert 'metadata' in dataset
        assert dataset['metadata'].task == MLTask.TRAJECTORY_PREDICTION.value
    
    def test_export_dataset(self, mock_processor):
        """Test dataset export."""
        pipeline = DataPipeline(mock_processor)
        
        # Create sample dataset
        X = np.random.rand(10, 5, 3)
        y = np.random.rand(10, 2, 2)
        splits = {'train': (X, y)}
        
        metadata = DatasetMetadata(
            task='test_task',
            config=DatasetConfig(task=MLTask.TRAJECTORY_PREDICTION),
            num_samples=10,
            num_vessels=1,
            temporal_range=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            spatial_bounds={'min_lat': 60.0, 'max_lat': 65.0, 'min_lon': -8.0, 'max_lon': -5.0},
            feature_columns=['lat', 'lon', 'sog'],
            target_columns=['lat', 'lon']
        )
        
        dataset = {'splits': splits, 'metadata': metadata}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = pipeline.export_dataset(
                dataset, 
                Path(temp_dir), 
                formats=[DatasetFormat.NUMPY]
            )
            
            assert 'numpy_train_features' in exported_files
            assert 'numpy_train_targets' in exported_files
            assert 'metadata' in exported_files
            
            # Check files exist
            for file_path in exported_files.values():
                assert file_path.exists()


# Integration test fixtures
@pytest.fixture
def real_ais_data():
    """Create realistic AIS data for integration testing."""
    np.random.seed(42)
    data = []
    base_time = datetime(2024, 1, 1)
    
    # Create multiple vessels with realistic trajectories
    vessel_configs = [
        {'mmsi': 231000001, 'start_lat': 62.0, 'start_lon': -6.5, 'course': 45, 'speed': 12},
        {'mmsi': 231000002, 'start_lat': 62.1, 'start_lon': -6.4, 'course': 135, 'speed': 8},
        {'mmsi': 231000003, 'start_lat': 61.9, 'start_lon': -6.6, 'course': 270, 'speed': 15},
    ]
    
    for vessel in vessel_configs:
        for i in range(100):
            # Simulate realistic vessel movement
            time_offset = timedelta(minutes=i)
            distance_nm = vessel['speed'] * (i / 60)  # Distance in nautical miles
            
            # Convert to lat/lon changes (approximate)
            lat_change = distance_nm * np.cos(np.radians(vessel['course'])) / 60
            lon_change = distance_nm * np.sin(np.radians(vessel['course'])) / (60 * np.cos(np.radians(vessel['start_lat'])))
            
            data.append({
                'mmsi': vessel['mmsi'],
                'time': base_time + time_offset,
                'latitude': vessel['start_lat'] + lat_change + np.random.normal(0, 0.001),
                'longitude': vessel['start_lon'] + lon_change + np.random.normal(0, 0.001),
                'sog': vessel['speed'] + np.random.normal(0, 1),
                'cog': vessel['course'] + np.random.normal(0, 5),
                'heading': vessel['course'] + np.random.normal(0, 3),
                'turn': np.random.normal(0, 2),
                'status': np.random.choice([0, 1, 2]),
                'shiptype': np.random.choice([30, 31, 32, 70]),
                'to_bow': np.random.randint(10, 100),
                'to_stern': np.random.randint(10, 100),
                'to_port': np.random.randint(5, 20),
                'to_starboard': np.random.randint(5, 20),
                'accuracy': np.random.choice([0, 1, 2])
            })
    
    return pd.DataFrame(data)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_trajectory_prediction(self, real_ais_data):
        """Test complete trajectory prediction pipeline."""
        # Create processor
        processor = AISMultiTaskProcessor([MLTask.TRAJECTORY_PREDICTION])
        
        # Create pipeline
        pipeline = DataPipeline(processor)
        pipeline.register_builder(MLTask.TRAJECTORY_PREDICTION, TrajectoryPredictionBuilder)
        
        # Create configuration
        config = DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=10,
            prediction_horizon=5,
            min_trajectory_length=20,
            validation_split=0.2,
            test_split=0.1
        )
        
        # Build dataset
        dataset = pipeline.build_dataset(real_ais_data, MLTask.TRAJECTORY_PREDICTION, config)
        
        # Validate results
        assert 'splits' in dataset
        assert 'metadata' in dataset
        
        splits = dataset['splits']
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        # Check data shapes
        train_X, train_y = splits['train']
        assert len(train_X.shape) == 3  # (samples, time, features)
        assert len(train_y.shape) == 3  # (samples, time, targets)
        assert train_X.shape[1] == config.sequence_length
        assert train_y.shape[1] == config.prediction_horizon
        
        # Validate metadata
        metadata = dataset['metadata']
        assert metadata.task == MLTask.TRAJECTORY_PREDICTION.value
        assert metadata.num_samples > 0
        assert metadata.num_vessels == 3
    
    def test_end_to_end_anomaly_detection(self, real_ais_data):
        """Test complete anomaly detection pipeline."""
        # Add some anomalous data
        anomaly_data = real_ais_data.copy()
        anomaly_data.loc[50, 'sog'] = 50  # Speed anomaly
        anomaly_data.loc[100, 'cog'] = anomaly_data.loc[99, 'cog'] + 180  # Course anomaly
        
        # Create processor
        processor = AISMultiTaskProcessor([MLTask.ANOMALY_DETECTION])
        
        # Create pipeline
        pipeline = DataPipeline(processor)
        pipeline.register_builder(MLTask.ANOMALY_DETECTION, AnomalyDetectionBuilder)
        
        # Create configuration
        config = DatasetConfig(
            task=MLTask.ANOMALY_DETECTION,
            sequence_length=5,
            min_trajectory_length=20
        )
        
        # Build dataset
        dataset = pipeline.build_dataset(anomaly_data, MLTask.ANOMALY_DETECTION, config)
        
        # Validate results
        assert 'splits' in dataset
        splits = dataset['splits']
        train_X, train_y = splits['train']
        
        # Should have detected some anomalies
        assert train_y.sum() > 0
    
    def test_data_validation_integration(self, real_ais_data):
        """Test data validation integration."""
        validator = DataValidator(strict_mode=False)
        result = validator.validate_dataset(real_ais_data, task='trajectory_prediction')
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert 'total_records' in result.metrics
        assert result.metrics['total_records'] == len(real_ais_data)
        assert result.metrics['unique_vessels'] == 3
    
    def test_quality_checking_integration(self, real_ais_data):
        """Test quality checking integration."""
        checker = QualityChecker()
        
        trajectory_quality = checker.check_trajectory_quality(real_ais_data)
        coverage_quality = checker.check_coverage_quality(real_ais_data)
        
        assert trajectory_quality['num_trajectories'] == 3
        assert trajectory_quality['avg_trajectory_length'] == 100
        assert coverage_quality['temporal_extent_hours'] > 0
        assert coverage_quality['spatial_extent_deg2'] > 0
    
    def test_export_import_cycle(self, real_ais_data):
        """Test complete export/import cycle."""
        # Create processor and pipeline
        processor = AISMultiTaskProcessor([MLTask.TRAJECTORY_PREDICTION])
        pipeline = DataPipeline(processor)
        pipeline.register_builder(MLTask.TRAJECTORY_PREDICTION, TrajectoryPredictionBuilder)
        
        # Build dataset
        config = DatasetConfig(
            task=MLTask.TRAJECTORY_PREDICTION,
            sequence_length=5,
            prediction_horizon=2,
            min_trajectory_length=20
        )
        
        dataset = pipeline.build_dataset(real_ais_data, MLTask.TRAJECTORY_PREDICTION, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export dataset
            exported_files = pipeline.export_dataset(
                dataset, 
                Path(temp_dir),
                formats=[DatasetFormat.NUMPY, DatasetFormat.PARQUET]
            )
            
            # Load dataset back
            loaded_dataset = pipeline.load_dataset(Path(temp_dir), DatasetFormat.NUMPY)
            
            # Validate loaded data
            assert 'splits' in loaded_dataset
            assert 'metadata' in loaded_dataset
            
            original_train_X, original_train_y = dataset['splits']['train']
            loaded_train_X, loaded_train_y = loaded_dataset['splits']['train']
            
            # Check shapes match
            assert original_train_X.shape == loaded_train_X.shape
            assert original_train_y.shape == loaded_train_y.shape
            
            # Check data is approximately equal (allowing for floating point precision)
            np.testing.assert_array_almost_equal(original_train_X, loaded_train_X, decimal=6)
            np.testing.assert_array_almost_equal(original_train_y, loaded_train_y, decimal=6)

