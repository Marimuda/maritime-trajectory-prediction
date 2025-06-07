"""
Metrics for evaluating trajectory prediction models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


class TrajectoryMetrics:
    """
    Metrics for evaluating trajectory prediction performance.
    """
    
    @staticmethod
    def rmse_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return math.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
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
    def distance_error(true_positions: np.ndarray, pred_positions: np.ndarray) -> np.ndarray:
        """
        Calculate distance errors between true and predicted positions.
        
        Args:
            true_positions: Array of shape (n_samples, 2) with [lat, lon]
            pred_positions: Array of shape (n_samples, 2) with [lat, lon]
            
        Returns:
            Array of distance errors in kilometers
        """
        errors = []
        
        for i in range(len(true_positions)):
            true_lat, true_lon = true_positions[i]
            pred_lat, pred_lon = pred_positions[i]
            
            error = TrajectoryMetrics.haversine_distance(
                true_lat, true_lon, pred_lat, pred_lon
            )
            errors.append(error)
        
        return np.array(errors)
    
    @staticmethod
    def average_displacement_error(true_positions: np.ndarray, pred_positions: np.ndarray) -> float:
        """
        Calculate Average Displacement Error (ADE).
        
        Args:
            true_positions: Array of shape (n_samples, 2) with [lat, lon]
            pred_positions: Array of shape (n_samples, 2) with [lat, lon]
            
        Returns:
            ADE in kilometers
        """
        distance_errors = TrajectoryMetrics.distance_error(true_positions, pred_positions)
        return np.mean(distance_errors)
    
    @staticmethod
    def final_displacement_error(true_positions: np.ndarray, pred_positions: np.ndarray) -> float:
        """
        Calculate Final Displacement Error (FDE).
        
        Args:
            true_positions: Array of shape (n_samples, 2) with [lat, lon]
            pred_positions: Array of shape (n_samples, 2) with [lat, lon]
            
        Returns:
            FDE in kilometers
        """
        if len(true_positions) == 0:
            return 0.0
        
        true_final = true_positions[-1]
        pred_final = pred_positions[-1]
        
        return TrajectoryMetrics.haversine_distance(
            true_final[0], true_final[1], pred_final[0], pred_final[1]
        )
    
    @staticmethod
    def trajectory_metrics(true_trajectories: List[np.ndarray], 
                         pred_trajectories: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate comprehensive trajectory prediction metrics.
        
        Args:
            true_trajectories: List of true trajectory arrays
            pred_trajectories: List of predicted trajectory arrays
            
        Returns:
            Dictionary with metric values
        """
        if len(true_trajectories) != len(pred_trajectories):
            raise ValueError("Number of true and predicted trajectories must match")
        
        ade_values = []
        fde_values = []
        lat_rmse_values = []
        lon_rmse_values = []
        lat_mae_values = []
        lon_mae_values = []
        
        for true_traj, pred_traj in zip(true_trajectories, pred_trajectories):
            if len(true_traj) != len(pred_traj):
                # Truncate to minimum length
                min_len = min(len(true_traj), len(pred_traj))
                true_traj = true_traj[:min_len]
                pred_traj = pred_traj[:min_len]
            
            if len(true_traj) == 0:
                continue
            
            # Calculate ADE and FDE
            ade = TrajectoryMetrics.average_displacement_error(true_traj, pred_traj)
            fde = TrajectoryMetrics.final_displacement_error(true_traj, pred_traj)
            
            ade_values.append(ade)
            fde_values.append(fde)
            
            # Calculate coordinate-wise errors
            true_lats = true_traj[:, 0]
            true_lons = true_traj[:, 1]
            pred_lats = pred_traj[:, 0]
            pred_lons = pred_traj[:, 1]
            
            lat_rmse = TrajectoryMetrics.rmse_error(true_lats, pred_lats)
            lon_rmse = TrajectoryMetrics.rmse_error(true_lons, pred_lons)
            lat_mae = TrajectoryMetrics.mae_error(true_lats, pred_lats)
            lon_mae = TrajectoryMetrics.mae_error(true_lons, pred_lons)
            
            lat_rmse_values.append(lat_rmse)
            lon_rmse_values.append(lon_rmse)
            lat_mae_values.append(lat_mae)
            lon_mae_values.append(lon_mae)
        
        return {
            'ADE_km': np.mean(ade_values) if ade_values else 0.0,
            'FDE_km': np.mean(fde_values) if fde_values else 0.0,
            'Latitude_RMSE': np.mean(lat_rmse_values) if lat_rmse_values else 0.0,
            'Longitude_RMSE': np.mean(lon_rmse_values) if lon_rmse_values else 0.0,
            'Latitude_MAE': np.mean(lat_mae_values) if lat_mae_values else 0.0,
            'Longitude_MAE': np.mean(lon_mae_values) if lon_mae_values else 0.0,
            'num_trajectories': len(true_trajectories)
        }
    
    @staticmethod
    def speed_accuracy(true_speeds: np.ndarray, pred_speeds: np.ndarray) -> Dict[str, float]:
        """
        Calculate speed prediction accuracy metrics.
        
        Args:
            true_speeds: True speed values in knots
            pred_speeds: Predicted speed values in knots
            
        Returns:
            Dictionary with speed metrics
        """
        return {
            'Speed_RMSE_knots': TrajectoryMetrics.rmse_error(true_speeds, pred_speeds),
            'Speed_MAE_knots': TrajectoryMetrics.mae_error(true_speeds, pred_speeds),
            'Speed_MAPE_%': np.mean(np.abs((true_speeds - pred_speeds) / (true_speeds + 1e-8))) * 100
        }
    
    @staticmethod
    def course_accuracy(true_courses: np.ndarray, pred_courses: np.ndarray) -> Dict[str, float]:
        """
        Calculate course prediction accuracy metrics.
        
        Args:
            true_courses: True course values in degrees
            pred_courses: Predicted course values in degrees
            
        Returns:
            Dictionary with course metrics
        """
        # Handle circular nature of course (0° = 360°)
        course_errors = []
        
        for true_course, pred_course in zip(true_courses, pred_courses):
            error = abs(true_course - pred_course)
            # Take the smaller angle
            error = min(error, 360 - error)
            course_errors.append(error)
        
        course_errors = np.array(course_errors)
        
        return {
            'Course_MAE_degrees': np.mean(course_errors),
            'Course_RMSE_degrees': np.sqrt(np.mean(course_errors**2))
        }

