"""
Pytest configuration file for maritime trajectory prediction tests.
"""
import os
import pytest
from pathlib import Path

# Define fixtures that can be reused across tests
@pytest.fixture
def sample_ais_file():
    """Path to the sample AIS data file."""
    sample_path = Path(__file__).parent.parent / "data" / "sample_ais.log"
    if not sample_path.exists():
        pytest.skip(f"Sample AIS data file not found at {sample_path}")
    return sample_path

@pytest.fixture
def processed_ais_file():
    """Path to the processed AIS data file."""
    processed_path = Path(__file__).parent.parent / "data" / "processed_ais_all.parquet"
    if not processed_path.exists():
        pytest.skip(f"Processed AIS data file not found at {processed_path}")
    return processed_path