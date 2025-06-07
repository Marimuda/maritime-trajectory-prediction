"""
Maritime utility functions for AIS data processing and analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
import math
from geopy.distance import geodesic


class MaritimeUtils:
    """
    Utility class for maritime-specific calculations and operations.
    """
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in decimal degrees
            lat2, lon2: Latitude and longitude of second point in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
    
    @staticmethod
    def calculate_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Calculate distances between consecutive points in a trajectory.
        
        Args:
            lats: Array of latitudes
            lons: Array of longitudes
            
        Returns:
            Array of distances in kilometers (first element is 0)
        """
        distances = np.zeros(len(lats))
        
        for i in range(1, len(lats)):
            distances[i] = MaritimeUtils.haversine_distance(
                lats[i-1], lons[i-1], lats[i], lons[i]
            )
        
        return distances
    
    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the bearing between two points.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in decimal degrees
            lat2, lon2: Latitude and longitude of second point in decimal degrees
            
        Returns:
            Bearing in degrees (0-360)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        
        # Convert to degrees and normalize to 0-360
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    @staticmethod
    def calculate_speed(distance_km: float, time_hours: float) -> float:
        """
        Calculate speed from distance and time.
        
        Args:
            distance_km: Distance in kilometers
            time_hours: Time in hours
            
        Returns:
            Speed in knots
        """
        if time_hours <= 0:
            return 0.0
        
        speed_kmh = distance_km / time_hours
        speed_knots = speed_kmh * 0.539957  # Convert km/h to knots
        
        return speed_knots
    
    @staticmethod
    def normalize_course(course: float) -> float:
        """
        Normalize course to 0-360 degrees.
        
        Args:
            course: Course in degrees
            
        Returns:
            Normalized course (0-360 degrees)
        """
        return course % 360
    
    @staticmethod
    def course_difference(course1: float, course2: float) -> float:
        """
        Calculate the difference between two courses, accounting for wraparound.
        
        Args:
            course1: First course in degrees
            course2: Second course in degrees
            
        Returns:
            Course difference in degrees (-180 to 180)
        """
        diff = course2 - course1
        
        # Handle wraparound
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
            
        return diff
    
    @staticmethod
    def is_valid_position(lat: float, lon: float) -> bool:
        """
        Check if a position is valid.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            
        Returns:
            True if position is valid
        """
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    
    @staticmethod
    def is_valid_speed(speed_knots: float, max_speed: float = 50.0) -> bool:
        """
        Check if a speed is valid for a maritime vessel.
        
        Args:
            speed_knots: Speed in knots
            max_speed: Maximum valid speed in knots
            
        Returns:
            True if speed is valid
        """
        return 0 <= speed_knots <= max_speed
    
    @staticmethod
    def is_valid_course(course: float) -> bool:
        """
        Check if a course is valid.
        
        Args:
            course: Course in degrees
            
        Returns:
            True if course is valid
        """
        return 0 <= course <= 360
    
    @staticmethod
    def interpolate_position(lat1: float, lon1: float, lat2: float, lon2: float, 
                           fraction: float) -> Tuple[float, float]:
        """
        Interpolate position between two points.
        
        Args:
            lat1, lon1: First position
            lat2, lon2: Second position
            fraction: Interpolation fraction (0-1)
            
        Returns:
            Interpolated latitude and longitude
        """
        # Simple linear interpolation (for short distances)
        lat = lat1 + fraction * (lat2 - lat1)
        lon = lon1 + fraction * (lon2 - lon1)
        
        return lat, lon
    
    @staticmethod
    def calculate_trajectory_features(trajectory: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive features for a trajectory.
        
        Args:
            trajectory: DataFrame with lat, lon, timestamp columns
            
        Returns:
            DataFrame with additional feature columns
        """
        traj = trajectory.copy()
        
        # Calculate distances
        distances = MaritimeUtils.calculate_distances(
            traj['lat'].values, traj['lon'].values
        )
        traj['distance_km'] = distances
        
        # Calculate bearings
        bearings = np.zeros(len(traj))
        for i in range(1, len(traj)):
            bearings[i] = MaritimeUtils.calculate_bearing(
                traj.iloc[i-1]['lat'], traj.iloc[i-1]['lon'],
                traj.iloc[i]['lat'], traj.iloc[i]['lon']
            )
        traj['bearing'] = bearings
        
        # Calculate speeds if timestamp is available
        if 'timestamp' in traj.columns:
            time_diffs = traj['timestamp'].diff().dt.total_seconds() / 3600  # hours
            speeds = np.zeros(len(traj))
            
            for i in range(1, len(traj)):
                if time_diffs.iloc[i] > 0:
                    speeds[i] = MaritimeUtils.calculate_speed(
                        distances[i], time_diffs.iloc[i]
                    )
            
            traj['calculated_speed_knots'] = speeds
            traj['time_diff_hours'] = time_diffs
        
        # Calculate course changes
        if len(traj) > 2:
            course_changes = np.zeros(len(traj))
            for i in range(2, len(traj)):
                course_changes[i] = MaritimeUtils.course_difference(
                    bearings[i-1], bearings[i]
                )
            traj['course_change'] = course_changes
        
        return traj

