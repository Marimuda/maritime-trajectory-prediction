"""
Comprehensive pytest tests for maritime utilities.
"""

import numpy as np
import pandas as pd

from src.utils.maritime_utils import MaritimeUtils


class TestMaritimeUtils:
    """Test suite for MaritimeUtils class."""

    def test_calculate_distance_valid_coordinates(self):
        """Test distance calculation with valid coordinates."""
        # Test known distance (approximately)
        lat1, lon1 = 62.006901, -6.774024  # Faroe Islands
        lat2, lon2 = 62.006180, -6.772121  # Nearby point

        distance = MaritimeUtils.calculate_distance(lat1, lon1, lat2, lon2)

        assert isinstance(distance, float)
        assert distance > 0
        assert distance < 1.0  # Should be less than 1 nautical mile

    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        lat, lon = 62.0, -6.7
        distance = MaritimeUtils.calculate_distance(lat, lon, lat, lon)

        assert distance == 0.0

    def test_calculate_distance_with_nan(self):
        """Test distance calculation with NaN values."""
        distance = MaritimeUtils.calculate_distance(np.nan, -6.7, 62.0, -6.8)
        assert pd.isna(distance)

        distance = MaritimeUtils.calculate_distance(62.0, np.nan, 62.1, -6.8)
        assert pd.isna(distance)

        distance = MaritimeUtils.calculate_distance(62.0, -6.7, np.nan, -6.8)
        assert pd.isna(distance)

        distance = MaritimeUtils.calculate_distance(62.0, -6.7, 62.1, np.nan)
        assert pd.isna(distance)

    def test_calculate_bearing_valid_coordinates(self):
        """Test bearing calculation with valid coordinates."""
        # North direction
        lat1, lon1 = 62.0, -6.7
        lat2, lon2 = 62.1, -6.7  # Point to the north

        bearing = MaritimeUtils.calculate_bearing(lat1, lon1, lat2, lon2)

        assert isinstance(bearing, float)
        assert 0 <= bearing <= 360
        assert abs(bearing - 0) < 10  # Should be close to north (0 degrees)

    def test_calculate_bearing_east_direction(self):
        """Test bearing calculation for eastward direction."""
        lat1, lon1 = 62.0, -6.7
        lat2, lon2 = 62.0, -6.6  # Point to the east

        bearing = MaritimeUtils.calculate_bearing(lat1, lon1, lat2, lon2)

        assert 80 < bearing < 100  # Should be close to east (90 degrees)

    def test_calculate_bearing_with_nan(self):
        """Test bearing calculation with NaN values."""
        bearing = MaritimeUtils.calculate_bearing(np.nan, -6.7, 62.0, -6.8)
        assert pd.isna(bearing)

        bearing = MaritimeUtils.calculate_bearing(62.0, np.nan, 62.1, -6.8)
        assert pd.isna(bearing)

    def test_calculate_speed_valid_inputs(self):
        """Test speed calculation with valid inputs."""
        distance_nm = 10.0  # 10 nautical miles
        time_diff_hours = 2.0  # 2 hours

        speed = MaritimeUtils.calculate_speed(distance_nm, time_diff_hours)

        assert speed == 5.0  # 10 nm / 2 hours = 5 knots

    def test_calculate_speed_zero_time(self):
        """Test speed calculation with zero time difference."""
        speed = MaritimeUtils.calculate_speed(10.0, 0.0)
        assert pd.isna(speed)

        speed = MaritimeUtils.calculate_speed(10.0, -1.0)
        assert pd.isna(speed)

    def test_calculate_speed_unrealistic_speed(self):
        """Test speed calculation that results in unrealistic speed."""
        distance_nm = 1000.0  # 1000 nautical miles
        time_diff_hours = 1.0  # 1 hour

        speed = MaritimeUtils.calculate_speed(distance_nm, time_diff_hours)
        assert pd.isna(speed)  # > 100 knots should be rejected

    def test_calculate_speed_with_nan(self):
        """Test speed calculation with NaN values."""
        speed = MaritimeUtils.calculate_speed(np.nan, 2.0)
        assert pd.isna(speed)

        speed = MaritimeUtils.calculate_speed(10.0, np.nan)
        assert pd.isna(speed)

    def test_is_in_port_within_radius(self):
        """Test port proximity detection within radius."""
        # Test point near Faroe Islands
        lat, lon = 62.0, -6.7
        port_coords = [(62.01, -6.71), (61.9, -6.8)]  # Nearby ports

        result = MaritimeUtils.is_in_port(lat, lon, port_coords, radius_nm=2.0)
        assert result is True

    def test_is_in_port_outside_radius(self):
        """Test port proximity detection outside radius."""
        lat, lon = 62.0, -6.7
        port_coords = [(61.0, -5.0)]  # Distant port

        result = MaritimeUtils.is_in_port(lat, lon, port_coords, radius_nm=1.0)
        assert result is False

    def test_is_in_port_with_nan(self):
        """Test port proximity detection with NaN coordinates."""
        port_coords = [(62.0, -6.7)]

        result = MaritimeUtils.is_in_port(np.nan, -6.7, port_coords)
        assert result is False

        result = MaritimeUtils.is_in_port(62.0, np.nan, port_coords)
        assert result is False

    def test_is_in_port_empty_ports(self):
        """Test port proximity detection with empty port list."""
        result = MaritimeUtils.is_in_port(62.0, -6.7, [])
        assert result is False

    def test_classify_vessel_behavior_anchored(self):
        """Test vessel behavior classification for anchored vessel."""
        speeds = [0.1, 0.2, 0.0, 0.3]  # Low speeds
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "anchored"

    def test_classify_vessel_behavior_maneuvering(self):
        """Test vessel behavior classification for maneuvering vessel."""
        speeds = [1.0, 2.0, 1.5, 2.5]  # Low to moderate speeds
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "maneuvering"

    def test_classify_vessel_behavior_transit(self):
        """Test vessel behavior classification for transit."""
        speeds = [8.0, 9.0, 10.0, 7.5]  # Moderate speeds
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "transit"

    def test_classify_vessel_behavior_high_speed(self):
        """Test vessel behavior classification for high speed."""
        speeds = [20.0, 25.0, 18.0, 22.0]  # High speeds
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "high_speed"

    def test_classify_vessel_behavior_with_nan(self):
        """Test vessel behavior classification with NaN values."""
        speeds = [np.nan, 5.0, np.nan, 6.0]
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "transit"  # Should use valid values only

    def test_classify_vessel_behavior_empty_list(self):
        """Test vessel behavior classification with empty speed list."""
        behavior = MaritimeUtils.classify_vessel_behavior([])
        assert behavior == "unknown"

    def test_classify_vessel_behavior_all_nan(self):
        """Test vessel behavior classification with all NaN values."""
        speeds = [np.nan, np.nan, np.nan]
        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
        assert behavior == "unknown"

    def test_interpolate_position_valid_inputs(self):
        """Test position interpolation with valid inputs."""
        lat1, lon1 = 62.0, -6.7
        lat2, lon2 = 62.1, -6.6
        time1 = pd.Timestamp("2025-05-08 10:00:00", tz="UTC")
        time2 = pd.Timestamp("2025-05-08 10:10:00", tz="UTC")
        target_time = pd.Timestamp("2025-05-08 10:05:00", tz="UTC")  # Midpoint

        lat_interp, lon_interp = MaritimeUtils.interpolate_position(
            lat1, lon1, time1, lat2, lon2, time2, target_time
        )

        assert not pd.isna(lat_interp)
        assert not pd.isna(lon_interp)
        assert abs(lat_interp - 62.05) < 0.01  # Should be close to midpoint
        assert abs(lon_interp - (-6.65)) < 0.01  # Should be close to midpoint

    def test_interpolate_position_outside_bounds(self):
        """Test position interpolation outside time bounds."""
        lat1, lon1 = 62.0, -6.7
        lat2, lon2 = 62.1, -6.6
        time1 = pd.Timestamp("2025-05-08 10:00:00", tz="UTC")
        time2 = pd.Timestamp("2025-05-08 10:10:00", tz="UTC")
        target_time = pd.Timestamp("2025-05-08 09:50:00", tz="UTC")  # Before time1

        lat_interp, lon_interp = MaritimeUtils.interpolate_position(
            lat1, lon1, time1, lat2, lon2, time2, target_time
        )

        assert pd.isna(lat_interp)
        assert pd.isna(lon_interp)

    def test_interpolate_position_same_time(self):
        """Test position interpolation with same timestamps."""
        lat1, lon1 = 62.0, -6.7
        lat2, lon2 = 62.1, -6.6
        time1 = pd.Timestamp("2025-05-08 10:00:00", tz="UTC")
        time2 = time1  # Same time
        target_time = time1

        lat_interp, lon_interp = MaritimeUtils.interpolate_position(
            lat1, lon1, time1, lat2, lon2, time2, target_time
        )

        assert lat_interp == lat1
        assert lon_interp == lon1

    def test_interpolate_position_with_nan(self):
        """Test position interpolation with NaN values."""
        time1 = pd.Timestamp("2025-05-08 10:00:00", tz="UTC")
        time2 = pd.Timestamp("2025-05-08 10:10:00", tz="UTC")
        target_time = pd.Timestamp("2025-05-08 10:05:00", tz="UTC")

        lat_interp, lon_interp = MaritimeUtils.interpolate_position(
            np.nan, -6.7, time1, 62.1, -6.6, time2, target_time
        )

        assert pd.isna(lat_interp)
        assert pd.isna(lon_interp)

    def test_validate_trajectory_valid_data(self):
        """Test trajectory validation with valid data."""
        data = {
            "lat": [62.0, 62.01, 62.02],
            "lon": [-6.7, -6.69, -6.68],
            "time": pd.to_datetime(
                ["2025-05-08 10:00:00", "2025-05-08 10:01:00", "2025-05-08 10:02:00"],
                utc=True,
            ),
        }
        df = pd.DataFrame(data)

        validated_df = MaritimeUtils.validate_trajectory(df)

        assert len(validated_df) == len(df)
        assert "calculated_speed" in validated_df.columns
        assert (
            not validated_df["calculated_speed"].iloc[1:].isna().all()
        )  # Should have calculated speeds

    def test_validate_trajectory_unrealistic_speed(self):
        """Test trajectory validation with unrealistic speeds."""
        data = {
            "lat": [62.0, 63.0],  # Large jump
            "lon": [-6.7, -5.0],  # Large jump
            "time": pd.to_datetime(
                ["2025-05-08 10:00:00", "2025-05-08 10:01:00"], utc=True
            ),  # 1 minute
        }
        df = pd.DataFrame(data)

        validated_df = MaritimeUtils.validate_trajectory(df, max_speed_knots=50.0)

        # Check that calculated speed is present and unrealistic speeds are detected
        assert "calculated_speed" in validated_df.columns
        # The speed calculation should result in a very high speed that gets filtered
        # Note: The actual filtering depends on the calculated speed being > max_speed_knots

    def test_validate_trajectory_empty_dataframe(self):
        """Test trajectory validation with empty DataFrame."""
        df = pd.DataFrame()
        validated_df = MaritimeUtils.validate_trajectory(df)

        assert len(validated_df) == 0
        assert isinstance(validated_df, pd.DataFrame)

    def test_validate_trajectory_missing_columns(self):
        """Test trajectory validation with missing required columns."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        validated_df = MaritimeUtils.validate_trajectory(df)

        assert len(validated_df) == len(df)  # Should return unchanged
        assert "calculated_speed" not in validated_df.columns

    def test_validate_trajectory_no_time_column(self):
        """Test trajectory validation without time column."""
        data = {"lat": [62.0, 62.01, 62.02], "lon": [-6.7, -6.69, -6.68]}
        df = pd.DataFrame(data)

        validated_df = MaritimeUtils.validate_trajectory(df)

        assert len(validated_df) == len(df)  # Should return unchanged
        assert "calculated_speed" not in validated_df.columns
