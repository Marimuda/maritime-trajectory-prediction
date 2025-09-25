"""
Unit tests for the CacheManager class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache_manager import CacheFormat, CacheLevel, CacheManager


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """CacheManager instance for testing."""
    return CacheManager(temp_cache_dir)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame(
        {
            "lat": [60.1, 60.2, 60.3],
            "lon": [5.1, 5.2, 5.3],
            "sog": [10.1, 10.2, 10.3],
            "cog": [45.0, 46.0, 47.0],
        }
    )


@pytest.fixture
def sample_source_file(temp_cache_dir):
    """Create a sample source file."""
    source_file = temp_cache_dir / "test_data.csv"
    source_file.write_text("test,data\n1,2\n3,4\n")
    return source_file


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache_manager = CacheManager(temp_cache_dir)

        assert cache_manager.cache_dir == temp_cache_dir
        assert cache_manager.version == "1.0"

        # Check that level directories are created
        for level in CacheLevel:
            assert (temp_cache_dir / level.value).exists()

    def test_file_hash_computation(self, cache_manager, sample_source_file):
        """Test file hash computation."""
        hash1 = cache_manager._compute_file_hash(sample_source_file)
        assert hash1 != ""
        assert len(hash1) == 32  # MD5 hex digest length

        # Hash should be consistent
        hash2 = cache_manager._compute_file_hash(sample_source_file)
        assert hash1 == hash2

        # Hash should change when file changes
        sample_source_file.write_text("different content")
        hash3 = cache_manager._compute_file_hash(sample_source_file)
        assert hash1 != hash3

    def test_params_hash_computation(self, cache_manager):
        """Test parameter hash computation."""
        params1 = {"a": 1, "b": 2, "c": [3, 4]}
        params2 = {"b": 2, "a": 1, "c": [3, 4]}  # Different order
        params3 = {"a": 1, "b": 3, "c": [3, 4]}  # Different values

        hash1 = cache_manager._compute_params_hash(params1)
        hash2 = cache_manager._compute_params_hash(params2)
        hash3 = cache_manager._compute_params_hash(params3)

        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different values should produce different hashes

    def test_cache_key_generation(self, cache_manager, sample_source_file):
        """Test cache key generation."""
        params = {"param1": "value1", "param2": 2}

        key1 = cache_manager._generate_cache_key(
            CacheLevel.RAW, [sample_source_file], params
        )
        key2 = cache_manager._generate_cache_key(
            CacheLevel.RAW, [sample_source_file], params
        )

        assert key1 == key2  # Should be consistent
        assert key1.startswith("raw_")

        # Different level should produce different key
        key3 = cache_manager._generate_cache_key(
            CacheLevel.CLEANED, [sample_source_file], params
        )
        assert key1 != key3

    def test_pickle_cache_put_get(self, cache_manager, sample_source_file, sample_data):
        """Test putting and getting data with pickle format."""
        params = {"test": "params"}

        # Put data in cache
        cache_key = cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params,
            CacheFormat.PICKLE,
        )

        assert cache_key is not None

        # Get data from cache
        retrieved_data = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PICKLE
        )

        assert retrieved_data is not None
        pd.testing.assert_frame_equal(sample_data, retrieved_data)

    def test_parquet_cache_put_get(
        self, cache_manager, sample_source_file, sample_data
    ):
        """Test putting and getting data with parquet format."""
        params = {"test": "params"}

        # Put data in cache
        cache_key = cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params,
            CacheFormat.PARQUET,
        )

        # Get data from cache
        retrieved_data = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PARQUET
        )

        assert retrieved_data is not None
        pd.testing.assert_frame_equal(sample_data, retrieved_data)

    def test_numpy_cache_put_get(self, cache_manager, sample_source_file):
        """Test putting and getting numpy arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        params = {"test": "numpy"}

        # Put data in cache
        cache_key = cache_manager.put(
            data, CacheLevel.FEATURES, [sample_source_file], params, CacheFormat.NUMPY
        )

        # Get data from cache
        retrieved_data = cache_manager.get(
            CacheLevel.FEATURES, [sample_source_file], params, CacheFormat.NUMPY
        )

        assert retrieved_data is not None
        np.testing.assert_array_equal(data, retrieved_data)

    def test_cache_miss(self, cache_manager, sample_source_file):
        """Test cache miss scenario."""
        params = {"nonexistent": "params"}

        result = cache_manager.get(
            CacheLevel.RAW, [sample_source_file], params, CacheFormat.PICKLE
        )

        assert result is None

    def test_cache_invalidation_on_file_change(
        self, cache_manager, sample_source_file, sample_data
    ):
        """Test that cache is invalidated when source file changes."""
        params = {"test": "invalidation"}

        # Put data in cache
        cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params,
            CacheFormat.PICKLE,
        )

        # Verify cache hit
        result1 = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PICKLE
        )
        assert result1 is not None

        # Modify source file
        sample_source_file.write_text("modified content")

        # Should now be cache miss
        result2 = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PICKLE
        )
        assert result2 is None

    def test_metadata_storage_retrieval(
        self, cache_manager, sample_source_file, sample_data
    ):
        """Test metadata storage and retrieval."""
        params = {"test": "metadata"}

        # Put data in cache
        cache_key = cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params,
            CacheFormat.PICKLE,
        )

        # Load metadata directly
        metadata = cache_manager._load_metadata(cache_key, CacheLevel.CLEANED)

        assert metadata is not None
        assert metadata.level == CacheLevel.CLEANED
        assert metadata.cache_key == cache_key
        assert str(sample_source_file) in metadata.source_files
        assert metadata.parameters == params
        assert metadata.format == CacheFormat.PICKLE
        assert metadata.size_bytes > 0

    def test_invalidate_specific_entry(
        self, cache_manager, sample_source_file, sample_data
    ):
        """Test invalidating a specific cache entry."""
        params = {"test": "invalidate"}

        # Put data in cache
        cache_key = cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params,
            CacheFormat.PICKLE,
        )

        # Verify it exists
        result1 = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PICKLE
        )
        assert result1 is not None

        # Invalidate
        cache_manager.invalidate(cache_key, CacheLevel.CLEANED)

        # Should now be cache miss
        result2 = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params, CacheFormat.PICKLE
        )
        assert result2 is None

    def test_clear_level(self, cache_manager, sample_source_file, sample_data):
        """Test clearing all entries at a specific level."""
        params1 = {"test": "clear1"}
        params2 = {"test": "clear2"}

        # Put multiple entries
        cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            params1,
            CacheFormat.PICKLE,
        )
        cache_manager.put(
            sample_data,
            CacheLevel.RAW,
            [sample_source_file],
            params2,
            CacheFormat.PICKLE,
        )

        # Clear one level
        count = cache_manager.clear_level(CacheLevel.CLEANED)
        assert count >= 2  # Data file + metadata file

        # CLEANED should be gone, RAW should remain
        result1 = cache_manager.get(
            CacheLevel.CLEANED, [sample_source_file], params1, CacheFormat.PICKLE
        )
        assert result1 is None

        result2 = cache_manager.get(
            CacheLevel.RAW, [sample_source_file], params2, CacheFormat.PICKLE
        )
        assert result2 is not None

    def test_get_info(self, cache_manager, sample_source_file, sample_data):
        """Test getting cache information."""
        # Start with empty cache
        info1 = cache_manager.get_info()
        assert info1["total"]["files"] == 0

        # Add some data
        cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            {"test": "info"},
            CacheFormat.PICKLE,
        )

        # Check updated info
        info2 = cache_manager.get_info()
        assert info2["total"]["files"] > 0
        assert info2["levels"]["cleaned"]["files"] > 0
        assert info2["levels"]["cleaned"]["size_mb"] > 0

    def test_list_entries(self, cache_manager, sample_source_file, sample_data):
        """Test listing cache entries."""
        # Start with empty cache
        entries1 = cache_manager.list_entries()
        assert len(entries1) == 0

        # Add some data
        cache_manager.put(
            sample_data,
            CacheLevel.CLEANED,
            [sample_source_file],
            {"test": "list"},
            CacheFormat.PICKLE,
        )

        # List all entries
        entries2 = cache_manager.list_entries()
        assert len(entries2) == 1
        assert entries2[0]["level"] == "cleaned"

        # List specific level
        entries3 = cache_manager.list_entries(CacheLevel.CLEANED)
        assert len(entries3) == 1

        entries4 = cache_manager.list_entries(CacheLevel.RAW)
        assert len(entries4) == 0

    def test_unsupported_format_error(self, cache_manager, sample_source_file):
        """Test error handling for unsupported formats."""
        # This should raise an error since we can't save a string as parquet
        with pytest.raises(ValueError, match="Cannot save.*as Parquet"):
            cache_manager.put(
                "string data",
                CacheLevel.RAW,
                [sample_source_file],
                {"test": "error"},
                CacheFormat.PARQUET,
            )

    def test_multiple_source_files(self, cache_manager, temp_cache_dir, sample_data):
        """Test caching with multiple source files."""
        # Create multiple source files
        file1 = temp_cache_dir / "source1.csv"
        file2 = temp_cache_dir / "source2.csv"
        file1.write_text("data1")
        file2.write_text("data2")

        params = {"test": "multi_source"}

        # Put data with multiple sources
        cache_key = cache_manager.put(
            sample_data, CacheLevel.CLEANED, [file1, file2], params, CacheFormat.PICKLE
        )

        # Should be able to retrieve
        result = cache_manager.get(
            CacheLevel.CLEANED, [file1, file2], params, CacheFormat.PICKLE
        )
        assert result is not None

        # Change one file - should invalidate cache
        file1.write_text("modified data1")

        result2 = cache_manager.get(
            CacheLevel.CLEANED, [file1, file2], params, CacheFormat.PICKLE
        )
        assert result2 is None
