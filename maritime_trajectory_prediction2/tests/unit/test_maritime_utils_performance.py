"""
Unit tests for MaritimeUtils distance calculation with performance verification.

Tests both correctness and performance, especially for scalar-to-Series broadcasting
which was the source of the 1296+ second slowdown.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.utils.maritime_utils import MaritimeUtils


@pytest.mark.unit
class TestDistanceCalculationCorrectness:
    """Test correctness of distance calculations."""

    def test_scalar_to_scalar(self):
        """Test distance between two scalar points."""
        # Oslo to Bergen (approximately 300 km = 162 NM)
        oslo_lat, oslo_lon = 59.9139, 10.7522
        bergen_lat, bergen_lon = 60.3913, 5.3221

        distance = MaritimeUtils.calculate_distance(
            oslo_lat, oslo_lon, bergen_lat, bergen_lon
        )

        # Should be around 162 nautical miles
        assert 150 < distance < 180
        assert isinstance(distance, float)

    def test_series_to_series(self):
        """Test distance between two Series (pairwise)."""
        # Create two trajectories
        lats1 = pd.Series([59.0, 59.1, 59.2])
        lons1 = pd.Series([10.0, 10.1, 10.2])
        lats2 = pd.Series([59.0, 59.1, 59.2])
        lons2 = pd.Series([10.0, 10.1, 10.2])

        distances = MaritimeUtils.calculate_distance(lats1, lons1, lats2, lons2)

        # Distance from same points should be ~0
        assert isinstance(distances, pd.Series)
        assert len(distances) == 3
        assert all(distances < 0.1)  # Very close to 0

    def test_scalar_to_series_broadcasting(self):
        """Test broadcasting scalar point to Series (the critical performance case)."""
        # Center point (scalar)
        center_lat = 59.0
        center_lon = 10.0

        # Multiple points (Series)
        n_points = 1000
        lats = pd.Series([59.0 + i * 0.01 for i in range(n_points)])
        lons = pd.Series([10.0 + i * 0.01 for i in range(n_points)])

        distances = MaritimeUtils.calculate_distance(
            lats, lons, center_lat, center_lon
        )

        # Should return Series of same length
        assert isinstance(distances, pd.Series)
        assert len(distances) == n_points

        # First point should be ~0 (same as center)
        assert distances.iloc[0] < 0.1

        # Distances should increase monotonically (moving away from center)
        assert distances.iloc[-1] > distances.iloc[0]

    def test_series_to_scalar_broadcasting(self):
        """Test broadcasting Series to scalar point."""
        # Multiple points (Series)
        lats = pd.Series([59.0, 59.1, 59.2])
        lons = pd.Series([10.0, 10.1, 10.2])

        # Target point (scalar)
        target_lat = 59.0
        target_lon = 10.0

        distances = MaritimeUtils.calculate_distance(
            target_lat, target_lon, lats, lons
        )

        # Should return Series
        assert isinstance(distances, pd.Series)
        assert len(distances) == 3

        # First point should be ~0
        assert distances.iloc[0] < 0.1

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        # Scalar case
        dist = MaritimeUtils.calculate_distance(np.nan, 10.0, 59.0, 10.0)
        assert pd.isna(dist)

        # Series case
        lats = pd.Series([59.0, np.nan, 59.2])
        lons = pd.Series([10.0, 10.1, 10.2])

        distances = MaritimeUtils.calculate_distance(lats, lons, 59.0, 10.0)

        # Should handle NaN gracefully
        assert isinstance(distances, pd.Series)
        assert len(distances) == 3


@pytest.mark.unit
class TestDistanceCalculationPerformance:
    """Test performance of distance calculations."""

    def test_scalar_to_series_performance(self):
        """Test that scalar-to-Series broadcasting is fast (the critical fix)."""
        # This is the case that was taking 1296+ seconds before the fix
        # Create a large dataset (simulating real preprocessing)
        n_points = 100000  # 100k points
        center_lat = 59.0
        center_lon = 10.0

        # Create Series
        lats = pd.Series(np.random.randn(n_points) * 0.1 + 59.0)
        lons = pd.Series(np.random.randn(n_points) * 0.1 + 10.0)

        # Time the calculation
        start_time = time.time()
        distances = MaritimeUtils.calculate_distance(lats, lons, center_lat, center_lon)
        elapsed_time = time.time() - start_time

        # Should complete in under 1 second for 100k points
        # (was taking ~1296s for 500k points before fix, scaled down)
        assert elapsed_time < 1.0, f"Too slow: {elapsed_time:.2f}s for 100k points"

        # Verify correctness
        assert isinstance(distances, pd.Series)
        assert len(distances) == n_points
        assert all(distances >= 0)  # All distances should be non-negative

    def test_series_to_series_performance(self):
        """Test pairwise distance performance."""
        n_points = 50000

        lats1 = pd.Series(np.random.randn(n_points) * 0.1 + 59.0)
        lons1 = pd.Series(np.random.randn(n_points) * 0.1 + 10.0)
        lats2 = pd.Series(np.random.randn(n_points) * 0.1 + 59.1)
        lons2 = pd.Series(np.random.randn(n_points) * 0.1 + 10.1)

        start_time = time.time()
        distances = MaritimeUtils.calculate_distance(lats1, lons1, lats2, lons2)
        elapsed_time = time.time() - start_time

        # Should be fast
        assert elapsed_time < 1.0, f"Too slow: {elapsed_time:.2f}s"

        # Verify correctness
        assert isinstance(distances, pd.Series)
        assert len(distances) == n_points

    def test_multiple_scalar_broadcasts(self):
        """Test repeated scalar broadcasts (simulating feature engineering)."""
        # Simulate calculating distance from center for multiple vessels
        n_vessels = 10
        points_per_vessel = 10000

        center_lat = 59.0
        center_lon = 10.0

        start_time = time.time()

        for _ in range(n_vessels):
            lats = pd.Series(np.random.randn(points_per_vessel) * 0.1 + 59.0)
            lons = pd.Series(np.random.randn(points_per_vessel) * 0.1 + 10.0)

            distances = MaritimeUtils.calculate_distance(lats, lons, center_lat, center_lon)

            assert len(distances) == points_per_vessel

        elapsed_time = time.time() - start_time

        # 10 vessels × 10k points = 100k total calculations
        # Should complete in under 2 seconds
        assert elapsed_time < 2.0, f"Too slow: {elapsed_time:.2f}s for 100k points"


@pytest.mark.unit
class TestDistanceCalculationAccuracy:
    """Test accuracy of Haversine formula implementation."""

    def test_known_distances(self):
        """Test against known real-world distances."""
        # New York to London (approximately 3000 NM)
        ny_lat, ny_lon = 40.7128, -74.0060
        london_lat, london_lon = 51.5074, -0.1278

        distance = MaritimeUtils.calculate_distance(ny_lat, ny_lon, london_lat, london_lon)

        # Should be around 3000 nautical miles (±100)
        assert 2900 < distance < 3100

    def test_equator_distance(self):
        """Test distance along equator."""
        # 1 degree of longitude at equator ≈ 60 NM
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 1.0

        distance = MaritimeUtils.calculate_distance(lat1, lon1, lat2, lon2)

        # Should be around 60 nautical miles
        assert 58 < distance < 62

    def test_zero_distance(self):
        """Test that distance from point to itself is zero."""
        lat, lon = 59.0, 10.0

        distance = MaritimeUtils.calculate_distance(lat, lon, lat, lon)

        assert distance < 0.001  # Essentially zero

    def test_symmetry(self):
        """Test that distance(A, B) = distance(B, A)."""
        lat1, lon1 = 59.0, 10.0
        lat2, lon2 = 60.0, 11.0

        dist_ab = MaritimeUtils.calculate_distance(lat1, lon1, lat2, lon2)
        dist_ba = MaritimeUtils.calculate_distance(lat2, lon2, lat1, lon1)

        assert abs(dist_ab - dist_ba) < 0.001

    def test_short_distances(self):
        """Test accuracy for very short distances."""
        # Two points 100 meters apart at 59°N
        lat1 = 59.0
        lon1 = 10.0
        lat2 = 59.0 + (100 / 111320)  # ~100 meters north
        lon2 = 10.0

        distance_nm = MaritimeUtils.calculate_distance(lat1, lon1, lat2, lon2)
        distance_m = distance_nm * 1852  # Convert NM to meters

        # Should be around 100 meters (±10m)
        assert 90 < distance_m < 110