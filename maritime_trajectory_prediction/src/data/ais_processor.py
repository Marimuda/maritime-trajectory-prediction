import pandas as pd
import numpy as np
from datetime import timedelta
import torch
from sklearn.preprocessing import MinMaxScaler

class AISProcessor:
    """Base class for AIS data processing"""
    def __init__(self, config):
        """
        Initialize AIS processor with configuration
        
        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
    
    def process(self, data):
        """
        Process AIS data
        
        Args:
            data: Raw AIS data
            
        Returns:
            Processed data
        """
        raise NotImplementedError("Subclasses must implement this method")

class AISDataProcessor(AISProcessor):
    """Process AIS data for trajectory prediction"""
    def __init__(self, config):
        super().__init__(config)
        
        # Store scaling parameters
        self.lat_min = config.get("lat_min", None)
        self.lat_max = config.get("lat_max", None)
        self.lon_min = config.get("lon_min", None)
        self.lon_max = config.get("lon_max", None)
        
        # Create scalers
        self.lat_scaler = MinMaxScaler()
        self.lon_scaler = MinMaxScaler()
        self.sog_scaler = MinMaxScaler()
        self.cog_scaler = MinMaxScaler()
    
    def process(self, data):
        """
        Process AIS data for trajectory prediction
        
        Args:
            data: DataFrame with AIS data
            
        Returns:
            Processed data
        """
        # 1. Clean data
        cleaned_data = self._clean_data(data)
        
        # 2. Create trajectories
        trajectories = self._create_trajectories(cleaned_data)
        
        # 3. Filter trajectories
        filtered_trajectories = self._filter_trajectories(trajectories)
        
        # 4. Create sequences
        input_sequences, target_sequences = self._create_sequences(filtered_trajectories)
        
        # 5. Scale features
        scaled_inputs, scaled_targets = self._scale_features(input_sequences, target_sequences)
        
        return {
            'input_sequences': scaled_inputs,
            'target_sequences': scaled_targets,
            'trajectories': filtered_trajectories
        }
    
    def _clean_data(self, data):
        """
        Clean AIS data
        
        Args:
            data: DataFrame with AIS data
            
        Returns:
            Cleaned data
        """
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Filter out vessels that are not moving
        df = df[df['sog'] > 0.5]  # Speed over 0.5 knots
        
        # Filter out unrealistic speeds
        df = df[df['sog'] < 50]  # Max realistic speed
        
        # Sort by MMSI and timestamp
        df = df.sort_values(['mmsi', 'timestamp'])
        
        return df
    
    def _create_trajectories(self, data):
        """
        Create trajectories from AIS data
        
        Args:
            data: Cleaned AIS data
            
        Returns:
            List of trajectory DataFrames
        """
        # Group by vessel ID
        grouped = data.groupby('mmsi')
        
        trajectories = []
        
        for mmsi, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate time differences
            time_diffs = group['timestamp'].diff()
            
            # Find gaps larger than threshold (e.g., 30 minutes)
            gap_indices = np.where(time_diffs > pd.Timedelta(minutes=30))[0]
            
            if len(gap_indices) == 0:
                # Single trajectory
                if len(group) >= 6:  # Minimum length
                    trajectories.append(group)
            else:
                # Multiple trajectories
                start_idx = 0
                for idx in gap_indices:
                    segment = group.iloc[start_idx:idx]
                    if len(segment) >= 6:  # Minimum length
                        trajectories.append(segment)
                    start_idx = idx
                
                # Last segment
                segment = group.iloc[start_idx:]
                if len(segment) >= 6:  # Minimum length
                    trajectories.append(segment)
        
        return trajectories
    
    def _filter_trajectories(self, trajectories):
        """
        Filter trajectories based on criteria
        
        Args:
            trajectories: List of trajectory DataFrames
            
        Returns:
            Filtered list of trajectories
        """
        filtered = []
        
        for traj in trajectories:
            # Check minimum length
            if len(traj) < self.config.min_seq_len:
                continue
            
            # Check for abnormal speeds
            if traj['sog'].max() > self.config.max_speed:
                continue
            
            # Add additional filters as needed
            
            filtered.append(traj)
        
        return filtered
    
    def _create_sequences(self, trajectories):
        """
        Create input/target sequences from trajectories
        
        Args:
            trajectories: List of trajectory DataFrames
            
        Returns:
            Input and target sequences
        """
        input_seq_len = self.config.input_seq_len
        target_seq_len = self.config.target_seq_len
        
        input_sequences = []
        target_sequences = []
        
        for traj in trajectories:
            # Need enough points for both input and target
            if len(traj) < input_seq_len + target_seq_len:
                continue
            
            # Create sliding window sequences
            for i in range(len(traj) - input_seq_len - target_seq_len + 1):
                input_idx = slice(i, i + input_seq_len)
                target_idx = slice(i + input_seq_len, i + input_seq_len + target_seq_len)
                
                input_seq = traj[['lat', 'lon', 'sog', 'cog']].iloc[input_idx].values
                target_seq = traj[['lat', 'lon', 'sog', 'cog']].iloc[target_idx].values
                
                input_sequences.append(input_seq)
                target_sequences.append(target_seq)
        
        return np.array(input_sequences), np.array(target_sequences)
    
    def _scale_features(self, inputs, targets):
        """
        Scale features to appropriate ranges
        
        Args:
            inputs: Input sequences
            targets: Target sequences
            
        Returns:
            Scaled inputs and targets
        """
        # Reshape for scaling
        input_shape = inputs.shape
        target_shape = targets.shape
        
        inputs_reshaped = inputs.reshape(-1, inputs.shape[-1])
        targets_reshaped = targets.reshape(-1, targets.shape[-1])
        
        # Set scaling ranges if not provided
        if self.lat_min is None:
            self.lat_min = inputs_reshaped[:, 0].min()
            self.lat_max = inputs_reshaped[:, 0].max()
            
        if self.lon_min is None:
            self.lon_min = inputs_reshaped[:, 1].min()
            self.lon_max = inputs_reshaped[:, 1].max()
        
        # Scale each feature
        lat_inputs = inputs_reshaped[:, 0].reshape(-1, 1)
        lon_inputs = inputs_reshaped[:, 1].reshape(-1, 1)
        sog_inputs = inputs_reshaped[:, 2].reshape(-1, 1)
        cog_inputs = inputs_reshaped[:, 3].reshape(-1, 1)
        
        lat_targets = targets_reshaped[:, 0].reshape(-1, 1)
        lon_targets = targets_reshaped[:, 1].reshape(-1, 1)
        sog_targets = targets_reshaped[:, 2].reshape(-1, 1)
        cog_targets = targets_reshaped[:, 3].reshape(-1, 1)
        
        # Fit and transform
        lat_inputs_scaled = self.lat_scaler.fit_transform(lat_inputs)
        lon_inputs_scaled = self.lon_scaler.fit_transform(lon_inputs)
        sog_inputs_scaled = self.sog_scaler.fit_transform(sog_inputs)
        cog_inputs_scaled = self.cog_scaler.fit_transform(cog_inputs)
        
        lat_targets_scaled = self.lat_scaler.transform(lat_targets)
        lon_targets_scaled = self.lon_scaler.transform(lon_targets)
        sog_targets_scaled = self.sog_scaler.transform(sog_targets)
        cog_targets_scaled = self.cog_scaler.transform(cog_targets)
        
        # Combine scaled features
        inputs_scaled = np.column_stack([
            lat_inputs_scaled, lon_inputs_scaled, 
            sog_inputs_scaled, cog_inputs_scaled
        ])
        
        targets_scaled = np.column_stack([
            lat_targets_scaled, lon_targets_scaled, 
            sog_targets_scaled, cog_targets_scaled
        ])
        
        # Reshape back to original shape
        inputs_scaled = inputs_scaled.reshape(input_shape)
        targets_scaled = targets_scaled.reshape(target_shape)
        
        return inputs_scaled, targets_scaled
    
    def inverse_scale(self, scaled_data):
        """
        Inverse scale data back to original range
        
        Args:
            scaled_data: Scaled data
            
        Returns:
            Original scale data
        """
        # Reshape for scaling
        data_shape = scaled_data.shape
        data_reshaped = scaled_data.reshape(-1, scaled_data.shape[-1])
        
        # Extract features
        lat_scaled = data_reshaped[:, 0].reshape(-1, 1)
        lon_scaled = data_reshaped[:, 1].reshape(-1, 1)
        sog_scaled = data_reshaped[:, 2].reshape(-1, 1)
        cog_scaled = data_reshaped[:, 3].reshape(-1, 1)
        
        # Inverse transform
        lat = self.lat_scaler.inverse_transform(lat_scaled)
        lon = self.lon_scaler.inverse_transform(lon_scaled)
        sog = self.sog_scaler.inverse_transform(sog_scaled)
        cog = self.cog_scaler.inverse_transform(cog_scaled)
        
        # Combine features
        data_original = np.column_stack([lat, lon, sog, cog])
        
        # Reshape back to original shape
        data_original = data_original.reshape(data_shape)
        
        return data_original

class FourHotAISProcessor(AISProcessor):
    """Process AIS data using four-hot encoding from TrAISformer paper"""
    def __init__(self, config):
        super().__init__(config)
        
        # Grid parameters
        self.lat_min = config.lat_min
        self.lat_max = config.lat_max
        self.lat_bins = config.lat_bins
        
        self.lon_min = config.lon_min
        self.lon_max = config.lon_max
        self.lon_bins = config.lon_bins
        
        self.sog_min = config.sog_min
        self.sog_max = config.sog_max
        self.sog_bins = config.sog_bins
        
        self.cog_bins = config.cog_bins
    
    def process(self, data):
        """
        Process AIS data using four-hot encoding
        
        Args:
            data: DataFrame with AIS data
            
        Returns:
            Processed data with four-hot encoding
        """
        # 1. Clean data
        cleaned_data = self._clean_data(data)
        
        # 2. Create trajectories
        trajectories = self._create_trajectories(cleaned_data)
        
        # 3. Filter trajectories
        filtered_trajectories = self._filter_trajectories(trajectories)
        
        # 4. Create encoded sequences
        input_encodings, target_encodings = self._create_four_hot_sequences(filtered_trajectories)
        
        return {
            'input_encodings': input_encodings,
            'target_encodings': target_encodings,
            'trajectories': filtered_trajectories
        }
    
    def _clean_data(self, data):
        """
        Clean AIS data
        
        Args:
            data: DataFrame with AIS data
            
        Returns:
            Cleaned data
        """
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Filter out vessels that are not moving
        df = df[df['sog'] > 0.5]  # Speed over 0.5 knots
        
        # Filter out unrealistic speeds
        df = df[df['sog'] < 50]  # Max realistic speed
        
        # Sort by MMSI and timestamp
        df = df.sort_values(['mmsi', 'timestamp'])
        
        return df
    
    def _create_trajectories(self, data):
        """Same as AISDataProcessor._create_trajectories"""
        # Group by vessel ID
        grouped = data.groupby('mmsi')
        
        trajectories = []
        
        for mmsi, group in grouped:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate time differences
            time_diffs = group['timestamp'].diff()
            
            # Find gaps larger than threshold (e.g., 30 minutes)
            gap_indices = np.where(time_diffs > pd.Timedelta(minutes=30))[0]
            
            if len(gap_indices) == 0:
                # Single trajectory
                if len(group) >= 6:  # Minimum length
                    trajectories.append(group)
            else:
                # Multiple trajectories
                start_idx = 0
                for idx in gap_indices:
                    segment = group.iloc[start_idx:idx]
                    if len(segment) >= 6:  # Minimum length
                        trajectories.append(segment)
                    start_idx = idx
                
                # Last segment
                segment = group.iloc[start_idx:]
                if len(segment) >= 6:  # Minimum length
                    trajectories.append(segment)
        
        return trajectories
    
    def _filter_trajectories(self, trajectories):
        """Same as AISDataProcessor._filter_trajectories"""
        filtered = []
        
        for traj in trajectories:
            # Check minimum length
            if len(traj) < self.config.min_seq_len:
                continue
            
            # Check for abnormal speeds
            if traj['sog'].max() > self.config.max_speed:
                continue
            
            filtered.append(traj)
        
        return filtered
    
    def _discretize_value(self, value, min_val, max_val, num_bins):
        """
        Discretize a continuous value into a bin index
        
        Args:
            value: Value to discretize
            min_val: Minimum value of range
            max_val: Maximum value of range
            num_bins: Number of bins for discretization
            
        Returns:
            Bin index (0 to num_bins-1)
        """
        if value <= min_val:
            return 0
        if value >= max_val:
            return num_bins - 1
        
        bin_size = (max_val - min_val) / num_bins
        bin_idx = int((value - min_val) / bin_size)
        
        return min(bin_idx, num_bins - 1)
    
    def _create_four_hot_encoding(self, lat, lon, sog, cog):
        """
        Create four-hot encoding for a single AIS point
        
        Args:
            lat: Latitude value
            lon: Longitude value
            sog: Speed over ground (knots)
            cog: Course over ground (degrees)
            
        Returns:
            Tuple of indices for each attribute
        """
        # Get bin indices
        lat_idx = self._discretize_value(lat, self.lat_min, self.lat_max, self.lat_bins)
        lon_idx = self._discretize_value(lon, self.lon_min, self.lon_max, self.lon_bins)
        sog_idx = self._discretize_value(sog, self.sog_min, self.sog_max, self.sog_bins)
        cog_idx = self._discretize_value(cog, 0, 360, self.cog_bins)
        
        return (lat_idx, lon_idx, sog_idx, cog_idx)
    
    def _create_four_hot_sequences(self, trajectories):
        """
        Create four-hot encoded sequences from trajectories
        
        Args:
            trajectories: List of trajectory DataFrames
            
        Returns:
            Input and target sequences with four-hot encoding
        """
        input_seq_len = self.config.input_seq_len
        target_seq_len = self.config.target_seq_len
        
        input_encodings = []
        target_encodings = []
        
        for traj in trajectories:
            # Need enough points for both input and target
            if len(traj) < input_seq_len + target_seq_len:
                continue
            
            # Create sliding window sequences
            for i in range(len(traj) - input_seq_len - target_seq_len + 1):
                input_idx = slice(i, i + input_seq_len)
                target_idx = slice(i + input_seq_len, i + input_seq_len + target_seq_len)
                
                input_traj = traj.iloc[input_idx]
                target_traj = traj.iloc[target_idx]
                
                # Create four-hot encodings
                input_enc = []
                for _, row in input_traj.iterrows():
                    encoding = self._create_four_hot_encoding(
                        row['lat'], row['lon'], row['sog'], row['cog']
                    )
                    input_enc.append(encoding)
                
                target_enc = []
                for _, row in target_traj.iterrows():
                    encoding = self._create_four_hot_encoding(
                        row['lat'], row['lon'], row['sog'], row['cog']
                    )
                    target_enc.append(encoding)
                
                input_encodings.append(input_enc)
                target_encodings.append(target_enc)
        
        # Convert to torch tensors
        input_encodings = torch.tensor(input_encodings, dtype=torch.long)
        target_encodings = torch.tensor(target_encodings, dtype=torch.long)
        
        return input_encodings, target_encodings
    
    def bin_to_value(self, bin_idx, feature_type):
        """
        Convert bin index back to continuous value
        
        Args:
            bin_idx: Bin index
            feature_type: Feature type ('lat', 'lon', 'sog', 'cog')
            
        Returns:
            Continuous value
        """
        if feature_type == 'lat':
            min_val, max_val, num_bins = self.lat_min, self.lat_max, self.lat_bins
        elif feature_type == 'lon':
            min_val, max_val, num_bins = self.lon_min, self.lon_max, self.lon_bins
        elif feature_type == 'sog':
            min_val, max_val, num_bins = self.sog_min, self.sog_max, self.sog_bins
        elif feature_type == 'cog':
            min_val, max_val, num_bins = 0, 360, self.cog_bins
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        bin_size = (max_val - min_val) / num_bins
        value = min_val + (bin_idx + 0.5) * bin_size  # Use bin center
        
        return value
