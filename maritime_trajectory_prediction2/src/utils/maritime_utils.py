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
        Calculate distance between two points using vectorized Haversine formula.
        Handles both scalar and Series/array inputs with efficient numpy broadcasting.

        Args:
            lat1, lon1: First point coordinates (scalar, Series, or array)
            lat2, lon2: Second point coordinates (scalar, Series, or array)

        Returns:
            Distance in nautical miles (scalar, Series, or array)
        """
        try:
            # Determine if we have Series inputs
            is_series = any(isinstance(x, pd.Series) for x in [lat1, lon1, lat2, lon2])

            if is_series:
                # Efficient handling: identify which inputs are Series and use numpy broadcasting
                result_index = None

                # Convert Series to numpy arrays, keep scalars as is
                if isinstance(lat1, pd.Series):
                    result_index = lat1.index
                    lat1_arr = lat1.values
                    lon1_arr = lon1.values if isinstance(lon1, pd.Series) else lon1
                    # Use numpy broadcasting for scalar lat2/lon2
                    lat2_arr = lat2.values if isinstance(lat2, pd.Series) else lat2
                    lon2_arr = lon2.values if isinstance(lon2, pd.Series) else lon2
                elif isinstance(lat2, pd.Series):
                    result_index = lat2.index
                    lat2_arr = lat2.values
                    lon2_arr = lon2.values if isinstance(lon2, pd.Series) else lon2
                    # Use numpy broadcasting for scalar lat1/lon1
                    lat1_arr = lat1
                    lon1_arr = lon1
                else:
                    # Edge case: lon1 or lon2 is Series but lat is not
                    # Find the Series to get result index
                    for x in [lon1, lon2]:
                        if isinstance(x, pd.Series):
                            result_index = x.index
                            break
                    lat1_arr = lat1.values if isinstance(lat1, pd.Series) else lat1
                    lon1_arr = lon1.values if isinstance(lon1, pd.Series) else lon1
                    lat2_arr = lat2.values if isinstance(lat2, pd.Series) else lat2
                    lon2_arr = lon2.values if isinstance(lon2, pd.Series) else lon2
            else:
                # All scalars
                if any(pd.isna([lat1, lon1, lat2, lon2])):
                    return np.nan
                lat1_arr = lat1
                lon1_arr = lon1
                lat2_arr = lat2
                lon2_arr = lon2
                result_index = None

            # Vectorized Haversine formula
            # Formula: d = 2 * R * arcsin(sqrt(sin²((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin²((lon2-lon1)/2)))
            R = 6371.0  # Earth radius in km

            # Convert to radians (vectorized, numpy handles broadcasting automatically)
            lat1_rad = np.radians(lat1_arr)
            lon1_rad = np.radians(lon1_arr)
            lat2_rad = np.radians(lat2_arr)
            lon2_rad = np.radians(lon2_arr)

            # Haversine formula (all vectorized operations with automatic broadcasting!)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_km = R * c

            # Convert to nautical miles (1 km = 0.539957 NM)
            distance_nm = distance_km * 0.539957

            # Return in appropriate format
            if is_series and result_index is not None:
                return pd.Series(distance_nm, index=result_index)
            elif isinstance(distance_nm, np.ndarray) and distance_nm.size == 1:
                return float(distance_nm)
            elif np.isscalar(distance_nm):
                return float(distance_nm)
            else:
                return distance_nm

        except Exception as e:
            logger.warning(f"Error calculating distance: {e}")
            if is_series and result_index is not None:
                return pd.Series(np.nan, index=result_index)
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
        Validate and clean trajectory data (VECTORIZED for performance).

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
                df = df.sort_values("time").reset_index(drop=True)

            # Calculate speeds between consecutive points (VECTORIZED)
            if len(df) > 1 and "time" in df.columns:
                # Get previous positions using shift (vectorized!)
                df["_prev_lat"] = df["lat"].shift(1)
                df["_prev_lon"] = df["lon"].shift(1)
                df["_prev_time"] = df["time"].shift(1)

                # Vectorized distance calculation
                distances = MaritimeUtils.calculate_distance(
                    df["_prev_lat"],
                    df["_prev_lon"],
                    df["lat"],
                    df["lon"]
                )

                # Vectorized time difference (in hours)
                time_diffs = (df["time"] - df["_prev_time"]).dt.total_seconds() / 3600

                # Vectorized speed calculation (distance/time)
                # Handle division by zero and NaN values
                speeds = np.where(
                    (time_diffs > 0) & ~pd.isna(distances),
                    distances / time_diffs,
                    np.nan
                )

                df["calculated_speed"] = speeds

                # Clean up temporary columns
                df = df.drop(columns=["_prev_lat", "_prev_lon", "_prev_time"])

                # Filter out points with unrealistic speeds
                mask = pd.isna(df["calculated_speed"]) | (
                    df["calculated_speed"] <= max_speed_knots
                )
                df = df[mask]

            return df

        except Exception as e:
            logger.error(f"Error validating trajectory: {e}")
            return df
