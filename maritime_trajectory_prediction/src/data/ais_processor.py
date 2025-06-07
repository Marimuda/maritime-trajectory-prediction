"""
AIS data processor for maritime trajectory prediction.

This module provides functionality to process raw AIS data into formats
suitable for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta

from ..utils.maritime_utils import MaritimeUtils
from ..utils.ais_parser import AISParser

logger = logging.getLogger(__name__)


class AISProcessor:
    """
    Processes raw AIS data for trajectory prediction models.
    
    This class handles data cleaning, feature engineering, and trajectory
    segmentation for AIS data.
    """
    
    def __init__(self, 
                 min_points_per_trajectory: int = 10,
                 max_time_gap_hours: float = 6.0,
                 min_speed_knots: float = 0.1,
                 max_speed_knots: float = 50.0):
        """
        Initialize AIS processor.
        
        Args:
            min_points_per_trajectory: Minimum number of points required for a valid trajectory
            max_time_gap_hours: Maximum time gap between points before splitting trajectory
            min_speed_knots: Minimum valid speed in knots
            max_speed_knots: Maximum valid speed in knots
        """
        self.min_points_per_trajectory = min_points_per_trajectory
        self.max_time_gap_hours = max_time_gap_hours
        self.min_speed_knots = min_speed_knots
        self.max_speed_knots = max_speed_knots
        
        self.maritime_utils = MaritimeUtils()
        self.ais_parser = AISParser()
        
    def load_ais_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load AIS data from various file formats.
        
        Args:
            file_path: Path to AIS data file
            
        Returns:
            DataFrame with raw AIS data
        """
        return self.ais_parser.load_processed_ais_data(file_path)
    
    def clean_ais_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw AIS data by removing invalid entries.
        
        Args:
            df: Raw AIS DataFrame
            
        Returns:
            Cleaned AIS DataFrame
        """
        logger.info(f"Cleaning AIS data with {len(df)} initial records")
        
        # Remove records with invalid coordinates
        df = df.dropna(subset=['lat', 'lon'])
        df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
        
        # Remove records with invalid speeds
        if 'sog' in df.columns:
            df = df[df['sog'].between(self.min_speed_knots, self.max_speed_knots)]
        
        # Remove records with invalid course
        if 'cog' in df.columns:
            df = df[df['cog'].between(0, 360) | df['cog'].isna()]
        
        # Sort by MMSI and timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values(['mmsi', 'timestamp'])
        
        logger.info(f"Cleaned data contains {len(df)} records")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for trajectory prediction.
        
        Args:
            df: Cleaned AIS DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for trajectory prediction")
        
        df = df.copy()
        
        # Calculate time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        
        # Calculate distance and speed features for each vessel
        for mmsi, vessel_data in df.groupby('mmsi'):
            vessel_idx = df['mmsi'] == mmsi
            
            # Calculate distances between consecutive points
            distances = self.maritime_utils.calculate_distances(
                vessel_data['lat'].values,
                vessel_data['lon'].values
            )
            
            # Calculate time differences
            if 'timestamp' in vessel_data.columns:
                time_diffs = vessel_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
                
                # Calculate derived speed
                derived_speeds = distances / time_diffs.fillna(1)  # km/h
                derived_speeds = derived_speeds * 0.539957  # Convert to knots
                
                df.loc[vessel_idx, 'time_diff_hours'] = time_diffs
                df.loc[vessel_idx, 'derived_speed_knots'] = derived_speeds
            
            df.loc[vessel_idx, 'distance_km'] = distances
            
            # Calculate course changes
            if 'cog' in vessel_data.columns:
                course_changes = vessel_data['cog'].diff()
                # Handle wraparound (e.g., 359° to 1°)
                course_changes = np.where(course_changes > 180, course_changes - 360, course_changes)
                course_changes = np.where(course_changes < -180, course_changes + 360, course_changes)
                df.loc[vessel_idx, 'course_change'] = course_changes
        
        return df
    
    def segment_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment trajectories based on time gaps and other criteria.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with trajectory segments
        """
        logger.info("Segmenting trajectories")
        
        df = df.copy()
        df['segment_id'] = 0
        
        segment_counter = 0
        
        for mmsi, vessel_data in df.groupby('mmsi'):
            vessel_idx = df['mmsi'] == mmsi
            
            # Find time gaps that exceed threshold
            if 'time_diff_hours' in vessel_data.columns:
                time_gaps = vessel_data['time_diff_hours'] > self.max_time_gap_hours
                
                # Create segment boundaries
                segment_boundaries = time_gaps.cumsum()
                
                # Assign segment IDs
                for segment_num in segment_boundaries.unique():
                    segment_mask = (segment_boundaries == segment_num)
                    segment_size = segment_mask.sum()
                    
                    # Only keep segments with minimum number of points
                    if segment_size >= self.min_points_per_trajectory:
                        df.loc[vessel_idx & segment_mask, 'segment_id'] = segment_counter
                        segment_counter += 1
                    else:
                        # Mark small segments for removal
                        df.loc[vessel_idx & segment_mask, 'segment_id'] = -1
            else:
                # If no timestamp, treat entire vessel track as one segment
                if len(vessel_data) >= self.min_points_per_trajectory:
                    df.loc[vessel_idx, 'segment_id'] = segment_counter
                    segment_counter += 1
                else:
                    df.loc[vessel_idx, 'segment_id'] = -1
        
        # Remove segments marked for deletion
        df = df[df['segment_id'] >= 0]
        
        logger.info(f"Created {segment_counter} trajectory segments")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for AIS data.
        
        Args:
            df: Raw AIS DataFrame
            
        Returns:
            Preprocessed DataFrame ready for model training
        """
        logger.info("Starting AIS data preprocessing pipeline")
        
        # Clean data
        df = self.clean_ais_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Segment trajectories
        df = self.segment_trajectories(df)
        
        logger.info("AIS data preprocessing completed")
        return df
    
    def get_trajectory_sequences(self, 
                               df: pd.DataFrame, 
                               sequence_length: int = 20,
                               prediction_horizon: int = 5) -> List[Dict]:
        """
        Extract trajectory sequences for model training.
        
        Args:
            df: Preprocessed AIS DataFrame
            sequence_length: Length of input sequences
            prediction_horizon: Number of future points to predict
            
        Returns:
            List of trajectory sequences
        """
        sequences = []
        
        for segment_id in df['segment_id'].unique():
            segment_data = df[df['segment_id'] == segment_id].copy()
            
            if len(segment_data) < sequence_length + prediction_horizon:
                continue
            
            # Extract sequences with sliding window
            for i in range(len(segment_data) - sequence_length - prediction_horizon + 1):
                input_seq = segment_data.iloc[i:i + sequence_length]
                target_seq = segment_data.iloc[i + sequence_length:i + sequence_length + prediction_horizon]
                
                sequences.append({
                    'input_sequence': input_seq,
                    'target_sequence': target_seq,
                    'mmsi': segment_data['mmsi'].iloc[0],
                    'segment_id': segment_id
                })
        
        logger.info(f"Extracted {len(sequences)} trajectory sequences")
        return sequences

