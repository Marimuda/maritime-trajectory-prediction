"""
Fixed maritime utilities with proper implementations.
"""

import logging

import numpy as np
import pandas as pd
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class MaritimeUtils:
    """Maritime utility functions for AIS data processing."""

    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using geodesic distance.
        Handles both scalar and Series inputs.

        Args:
            lat1, lon1: First point coordinates (scalar or Series)
            lat2, lon2: Second point coordinates (scalar or Series)

        Returns:
            Distance in nautical miles (scalar or Series)
        """
        try:
            # Check if any input is a Series
            if any(isinstance(x, pd.Series) for x in [lat1, lon1, lat2, lon2]):
                # Convert all to Series for vectorized operation
                if not isinstance(lat1, pd.Series):
                    # If lat2/lon2 are Series, create Series of same length
                    if isinstance(lat2, pd.Series):
                        lat1 = pd.Series([lat1] * len(lat2), index=lat2.index)
                        lon1 = pd.Series([lon1] * len(lat2), index=lat2.index)
                    else:
                        lat1 = pd.Series([lat1])
                        lon1 = pd.Series([lon1])
                        lat2 = pd.Series([lat2])
                        lon2 = pd.Series([lon2])
                
                # Create DataFrame for easier vectorization
                df = pd.DataFrame({
                    'lat1': lat1, 'lon1': lon1, 
                    'lat2': lat2, 'lon2': lon2
                })
                
                def calc_row_distance(row):
                    if pd.isna(row['lat1']) or pd.isna(row['lon1']) or \
                       pd.isna(row['lat2']) or pd.isna(row['lon2']):
                        return np.nan
                    try:
                        point1 = (row['lat1'], row['lon1'])
                        point2 = (row['lat2'], row['lon2'])
                        distance_km = geodesic(point1, point2).kilometers
                        return distance_km * 0.539957  # Convert to nautical miles
                    except:
                        return np.nan
                
                result = df.apply(calc_row_distance, axis=1)
                return result if len(result) > 1 else result.iloc[0]
            
            # Handle scalar inputs
            else:
                # Handle NaN values
                if any(pd.isna([lat1, lon1, lat2, lon2])):
                    return np.nan

                # Calculate geodesic distance
                point1 = (lat1, lon1)
                point2 = (lat2, lon2)
                distance_km = geodesic(point1, point2).kilometers

                # Convert to nautical miles
                distance_nm = distance_km * 0.539957

                return distance_nm

        except Exception as e:
            logger.warning(f"Error calculating distance: {e}")
            return np.nan

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing between two points.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Bearing in degrees (0-360)
        """
        try:
            # Handle NaN values
            if any(pd.isna([lat1, lon1, lat2, lon2])):
                return np.nan

            # Convert to radians
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            dlon_rad = np.radians(lon2 - lon1)

            # Calculate bearing
            y = np.sin(dlon_rad) * np.cos(lat2_rad)
            x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
                lat2_rad
            ) * np.cos(dlon_rad)

            bearing_rad = np.arctan2(y, x)
            bearing_deg = np.degrees(bearing_rad)

            # Normalize to 0-360
            bearing_deg = (bearing_deg + 360) % 360

            return bearing_deg

        except Exception as e:
            logger.warning(f"Error calculating bearing: {e}")
            return np.nan

    @staticmethod
    def calculate_speed(distance_nm: float, time_diff_hours: float) -> float:
        """
        Calculate speed from distance and time.

        Args:
            distance_nm: Distance in nautical miles
            time_diff_hours: Time difference in hours

        Returns:
            Speed in knots
        """
        try:
            if pd.isna(distance_nm) or pd.isna(time_diff_hours) or time_diff_hours <= 0:
                return np.nan

            speed_knots = distance_nm / time_diff_hours

            # Sanity check - reject unrealistic speeds
            if speed_knots > 100:  # > 100 knots is unrealistic for most vessels
                return np.nan

            return speed_knots

        except Exception as e:
            logger.warning(f"Error calculating speed: {e}")
            return np.nan

    @staticmethod
    def is_in_port(
        lat: float,
        lon: float,
        port_coords: list[tuple[float, float]],
        radius_nm: float = 1.0,
    ) -> bool:
        """
        Check if coordinates are within port area.

        Args:
            lat, lon: Point coordinates
            port_coords: List of (lat, lon) tuples for port centers
            radius_nm: Port radius in nautical miles

        Returns:
            True if within any port area
        """
        try:
            if pd.isna(lat) or pd.isna(lon):
                return False

            for port_lat, port_lon in port_coords:
                distance = MaritimeUtils.calculate_distance(
                    lat, lon, port_lat, port_lon
                )
                if not pd.isna(distance) and distance <= radius_nm:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking port proximity: {e}")
            return False

    @staticmethod
    def classify_vessel_behavior(
        speeds: list[float], threshold_knots: float = 0.5
    ) -> str:
        """
        Classify vessel behavior based on speed patterns.

        Args:
            speeds: List of speed values in knots
            threshold_knots: Speed threshold for stationary classification

        Returns:
            Behavior classification string
        """
        try:
            # Filter out NaN values
            valid_speeds = [s for s in speeds if not pd.isna(s)]

            if not valid_speeds:
                return "unknown"

            avg_speed = np.mean(valid_speeds)
            max_speed = np.max(valid_speeds)

            if avg_speed < threshold_knots:
                return "anchored"
            elif avg_speed < 3.0:
                return "maneuvering"
            elif avg_speed < 15.0:
                return "transit"
            else:
                return "high_speed"

        except Exception as e:
            logger.warning(f"Error classifying vessel behavior: {e}")
            return "unknown"

    @staticmethod
    def interpolate_position(
        lat1: float,
        lon1: float,
        time1: pd.Timestamp,
        lat2: float,
        lon2: float,
        time2: pd.Timestamp,
        target_time: pd.Timestamp,
    ) -> tuple[float, float]:
        """
        Interpolate position between two points at a target time.

        Args:
            lat1, lon1, time1: First position and time
            lat2, lon2, time2: Second position and time
            target_time: Time for interpolation

        Returns:
            Interpolated (lat, lon) coordinates
        """
        try:
            # Handle NaN values
            if any(pd.isna([lat1, lon1, lat2, lon2])) or pd.isna(target_time):
                return np.nan, np.nan

            # Check if target time is within bounds
            if target_time < min(time1, time2) or target_time > max(time1, time2):
                return np.nan, np.nan

            # Calculate time ratios
            total_time = (time2 - time1).total_seconds()
            if total_time == 0:
                return lat1, lon1

            elapsed_time = (target_time - time1).total_seconds()
            ratio = elapsed_time / total_time

            # Linear interpolation
            lat_interp = lat1 + ratio * (lat2 - lat1)
            lon_interp = lon1 + ratio * (lon2 - lon1)

            return lat_interp, lon_interp

        except Exception as e:
            logger.warning(f"Error interpolating position: {e}")
            return np.nan, np.nan

    @staticmethod
    def validate_trajectory(
        df: pd.DataFrame, max_speed_knots: float = 50.0
    ) -> pd.DataFrame:
        """
        Validate and clean trajectory data.

        Args:
            df: DataFrame with trajectory data
            max_speed_knots: Maximum realistic speed in knots

        Returns:
            Cleaned DataFrame
        """
        try:
            if df.empty or "lat" not in df.columns or "lon" not in df.columns:
                return df

            # Sort by time if available
            if "time" in df.columns:
                df = df.sort_values("time")

            # Calculate speeds between consecutive points
            if len(df) > 1 and "time" in df.columns:
                speeds = []
                for i in range(1, len(df)):
                    prev_row = df.iloc[i - 1]
                    curr_row = df.iloc[i]

                    distance = MaritimeUtils.calculate_distance(
                        prev_row["lat"],
                        prev_row["lon"],
                        curr_row["lat"],
                        curr_row["lon"],
                    )

                    time_diff = (
                        curr_row["time"] - prev_row["time"]
                    ).total_seconds() / 3600
                    speed = MaritimeUtils.calculate_speed(distance, time_diff)
                    speeds.append(speed)

                # Mark unrealistic speeds
                speeds = [np.nan] + speeds  # First point has no previous speed
                df["calculated_speed"] = speeds

                # Filter out points with unrealistic speeds
                mask = pd.isna(df["calculated_speed"]) | (
                    df["calculated_speed"] <= max_speed_knots
                )
                df = df[mask]

            return df

        except Exception as e:
            logger.error(f"Error validating trajectory: {e}")
            return df
