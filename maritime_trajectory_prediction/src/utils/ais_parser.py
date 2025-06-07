"""
AIS parser module for loading and processing AIS data files.
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import List, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AISParser:
    """
    Parser for AIS data in various formats.
    """
    
    def __init__(self):
        """Initialize AIS parser."""
        pass
    
    def load_processed_ais_data(self, file_path: Union[str, Path]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load processed AIS data from Parquet, CSV or JSONL files.
        
        Args:
            file_path: Path to the processed AIS data file
            
        Returns:
            DataFrame with processed AIS data or list of trajectories
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        extension = file_path.suffix.lower()
        
        if extension == '.parquet':
            return self._load_parquet(file_path)
        elif extension == '.csv':
            return self._load_csv(file_path)
        elif extension == '.jsonl':
            return self._load_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file."""
        logger.info(f"Loading Parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} records from Parquet file")
        return df
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file."""
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records from CSV file")
        return df
    
    def _load_jsonl(self, file_path: Path) -> List[pd.DataFrame]:
        """Load JSONL file as list of trajectories."""
        logger.info(f"Loading JSONL file: {file_path}")
        trajectories = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    vessel_data = json.loads(line.strip())
                    
                    mmsi = vessel_data.get('mmsi')
                    static_data = vessel_data.get('static', {})
                    trajectory_points = vessel_data.get('trajectory', [])
                    
                    if not trajectory_points:
                        continue
                    
                    # Convert trajectory points to DataFrame
                    traj_df = pd.DataFrame(trajectory_points)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in traj_df.columns:
                        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
                    
                    # Add MMSI
                    traj_df['mmsi'] = mmsi
                    
                    # Add static data
                    for key, value in static_data.items():
                        traj_df[key] = value
                    
                    trajectories.append(traj_df)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(trajectories)} trajectories from JSONL file")
        return trajectories
    
    def get_vessel_trajectories(self, 
                              ais_data: Union[pd.DataFrame, List[pd.DataFrame]], 
                              mmsi: int = None) -> List[pd.DataFrame]:
        """
        Extract vessel trajectories from processed AIS data.
        
        Args:
            ais_data: DataFrame with processed AIS data or list of trajectories
            mmsi: Optional MMSI to filter for a specific vessel
            
        Returns:
            List of trajectory DataFrames
        """
        # If already a list of trajectories
        if isinstance(ais_data, list):
            trajectories = ais_data
            
            # Filter by MMSI if specified
            if mmsi is not None:
                trajectories = [t for t in trajectories if t['mmsi'].iloc[0] == mmsi]
            
            return trajectories
        
        # If DataFrame, group by MMSI
        trajectories = []
        
        if 'mmsi' not in ais_data.columns:
            raise ValueError("AIS data must contain an 'mmsi' column")
        
        # Filter by MMSI if specified
        if mmsi is not None:
            vessel_data = ais_data[ais_data['mmsi'] == mmsi]
            
            # Group by trajectory segment if available
            if 'segment_id' in vessel_data.columns:
                for segment_id, segment in vessel_data.groupby('segment_id'):
                    trajectories.append(segment.copy())
            else:
                trajectories.append(vessel_data.copy())
        else:
            # Group by MMSI
            for vessel_mmsi, vessel_data in ais_data.groupby('mmsi'):
                # Group by trajectory segment if available
                if 'segment_id' in vessel_data.columns:
                    for segment_id, segment in vessel_data.groupby('segment_id'):
                        trajectories.append(segment.copy())
                else:
                    trajectories.append(vessel_data.copy())
        
        return trajectories
    
    def prepare_trajectory_for_prediction(self, 
                                        trajectory: pd.DataFrame, 
                                        sequence_length: int = 20, 
                                        features: List[str] = None) -> Dict[str, Any]:
        """
        Prepare a trajectory for prediction by extracting a sequence of the specified length.
        
        Args:
            trajectory: DataFrame with a vessel trajectory
            sequence_length: Number of points to include in the sequence
            features: List of features to include (defaults to position and motion features)
            
        Returns:
            Dictionary with sequence data and metadata
        """
        if len(trajectory) < sequence_length:
            raise ValueError(f"Trajectory has fewer than {sequence_length} points")
        
        # Default features if not specified
        if features is None:
            features = ['lat', 'lon', 'sog', 'cog', 'distance_km', 'speed_delta', 'course_delta']
        
        # Filter features that exist in the trajectory
        available_features = [f for f in features if f in trajectory.columns]
        
        if not available_features:
            raise ValueError(f"None of the specified features {features} are available in the trajectory")
        
        # Get the latest sequence_length points
        sequence = trajectory.iloc[-sequence_length:].copy()
        
        # Extract features
        feature_data = sequence[available_features].values
        
        return {
            'features': feature_data,
            'feature_names': available_features,
            'mmsi': trajectory['mmsi'].iloc[0] if 'mmsi' in trajectory.columns else None,
            'sequence_length': sequence_length,
            'timestamps': sequence['timestamp'].tolist() if 'timestamp' in sequence.columns else None
        }

