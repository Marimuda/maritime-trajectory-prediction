"""
Pytest configuration and fixtures for maritime trajectory prediction tests.

This module provides the runtime substrate for the test corpus, including
deterministic seeding, factory fixtures, and test infrastructure.
"""
import os
import random
import tempfile
from pathlib import Path
from typing import Generator, Callable, Dict, Any

import numpy as np
import pandas as pd
import pytest
import torch


def pytest_configure():
    """Configure pytest with deterministic seeding."""
    seed = int(os.getenv("TEST_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests and enforce marker usage."""
    for item in items:
        # Auto-mark tests that might be slow
        if "integration" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Ensure all tests have at least one marker
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# Data Factory Fixtures
# ============================================================================

@pytest.fixture
def ais_data_factory():
    """Factory for generating AIS test data with configurable parameters."""
    def _make_ais_data(
        n_vessels: int = 3,
        n_points_per_vessel: int = 20,
        start_lat: float = 59.0,
        start_lon: float = 10.0,
        time_interval_minutes: int = 5,
        seed: int = 0
    ) -> pd.DataFrame:
        """Generate synthetic AIS data for testing."""
        np.random.seed(seed)
        
        data = []
        base_time = pd.Timestamp("2023-01-01 12:00:00")
        
        for vessel_id in range(n_vessels):
            mmsi = 123456789 + vessel_id
            
            # Generate trajectory with some randomness
            lat = start_lat + np.random.normal(0, 0.01, n_points_per_vessel).cumsum()
            lon = start_lon + np.random.normal(0, 0.01, n_points_per_vessel).cumsum()
            
            # Ensure valid coordinates
            lat = np.clip(lat, -90, 90)
            lon = np.clip(lon, -180, 180)
            
            for i in range(n_points_per_vessel):
                timestamp = base_time + pd.Timedelta(minutes=i * time_interval_minutes)
                
                data.append({
                    'mmsi': mmsi,
                    'timestamp': timestamp,
                    'lat': lat[i],
                    'lon': lon[i],
                    'sog': np.random.uniform(5, 15),  # Speed over ground
                    'cog': np.random.uniform(0, 360),  # Course over ground
                    'heading': np.random.uniform(0, 360),
                    'nav_status': np.random.choice([0, 1, 2, 3, 4, 5]),
                    'vessel_type': np.random.choice([30, 31, 32, 33, 34, 35])
                })
        
        return pd.DataFrame(data)
    
    return _make_ais_data


@pytest.fixture
def tensor_factory():
    """Factory for generating PyTorch tensors with deterministic seeding."""
    def _make_tensor(
        shape: tuple = (64, 64),
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Generate a tensor with specified properties."""
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)
    
    return _make_tensor


@pytest.fixture
def trajectory_factory():
    """Factory for generating trajectory sequences for model testing."""
    def _make_trajectory_sequence(
        sequence_length: int = 20,
        prediction_horizon: int = 5,
        n_features: int = 4,
        seed: int = 0
    ) -> Dict[str, Any]:
        """Generate a trajectory sequence for testing."""
        np.random.seed(seed)
        
        # Generate smooth trajectory
        t = np.linspace(0, 1, sequence_length + prediction_horizon)
        lat_base = 59.0 + 0.1 * np.sin(2 * np.pi * t)
        lon_base = 10.0 + 0.1 * np.cos(2 * np.pi * t)
        
        # Add some noise
        lat = lat_base + np.random.normal(0, 0.001, len(t))
        lon = lon_base + np.random.normal(0, 0.001, len(t))
        sog = 10 + 2 * np.sin(4 * np.pi * t) + np.random.normal(0, 0.5, len(t))
        cog = (90 + 30 * np.sin(3 * np.pi * t)) % 360
        
        # Create DataFrame
        df = pd.DataFrame({
            'lat': lat,
            'lon': lon,
            'sog': np.clip(sog, 0, 30),
            'cog': cog
        })
        
        # Split into input and target
        input_sequence = df.iloc[:sequence_length]
        target_sequence = df.iloc[sequence_length:sequence_length + prediction_horizon]
        
        return {
            'input_sequence': input_sequence,
            'target_sequence': target_sequence,
            'mmsi': 123456789,
            'segment_id': 0,
            'full_trajectory': df
        }
    
    return _make_trajectory_sequence


# ============================================================================
# File and Data Fixtures
# ============================================================================

@pytest.fixture
def tmp_ais_file(tmp_path, ais_data_factory):
    """Create a temporary AIS data file for testing."""
    ais_data = ais_data_factory(n_vessels=2, n_points_per_vessel=10)
    file_path = tmp_path / "test_ais_data.csv"
    ais_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_ais_data(ais_data_factory):
    """Provide sample AIS data for testing."""
    return ais_data_factory(n_vessels=3, n_points_per_vessel=15, seed=42)


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data fixtures."""
    return Path(__file__).parent / "data" / "fixtures"


# ============================================================================
# Model and Component Fixtures
# ============================================================================

@pytest.fixture
def ais_parser():
    """Provide an AISParser instance for testing."""
    from maritime_trajectory_prediction.src.utils.ais_parser import AISParser
    return AISParser()


@pytest.fixture
def maritime_utils():
    """Provide a MaritimeUtils instance for testing."""
    from maritime_trajectory_prediction.src.utils.maritime_utils import MaritimeUtils
    return MaritimeUtils()


@pytest.fixture
def ais_processor():
    """Provide an AISProcessor instance with test configuration."""
    from maritime_trajectory_prediction.src.data.maritime_message_processor import AISProcessor
    return AISProcessor(
        min_points_per_trajectory=5,  # Lower for testing
        max_time_gap_hours=12.0,
        min_speed_knots=0.1,
        max_speed_knots=50.0
    )


# ============================================================================
# Performance and Benchmarking Fixtures
# ============================================================================

@pytest.fixture
def benchmark_data(ais_data_factory):
    """Large dataset for performance benchmarking."""
    return ais_data_factory(
        n_vessels=100,
        n_points_per_vessel=1000,
        seed=123
    )


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_torch_cache():
    """Clean up PyTorch cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Ensure random state is reset after each test."""
    yield
    # Reset random states
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

