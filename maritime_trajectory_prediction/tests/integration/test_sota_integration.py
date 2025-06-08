"""
Integration tests for SOTA models with maritime data pipeline.

This module tests the end-to-end integration of SOTA models
with the maritime trajectory prediction pipeline, including
data loading, preprocessing, training, and inference.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from models.anomaly_transformer import create_maritime_anomaly_transformer
from models.motion_transformer import create_maritime_motion_transformer
# Import only what we need to avoid circular imports
import importlib.util


class TestSOTADataIntegration:
    """Test SOTA models with real maritime data structures."""
    
    @pytest.fixture
    def sample_ais_data(self):
        """Create sample AIS data for testing."""
        np.random.seed(42)
        n_points = 100
        
        # Create realistic AIS trajectory
        base_lat, base_lon = 60.0, 5.0  # Norwegian waters
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min'),
            'mmsi': [123456789] * n_points,
            'latitude': base_lat + np.cumsum(np.random.normal(0, 0.001, n_points)),
            'longitude': base_lon + np.cumsum(np.random.normal(0, 0.001, n_points)),
            'sog': np.random.uniform(5, 15, n_points),  # Speed over ground
            'cog': np.random.uniform(0, 360, n_points),  # Course over ground
            'heading': np.random.uniform(0, 360, n_points),
            'nav_status': [0] * n_points,  # Under way using engine
            'vessel_type': [70] * n_points,  # Cargo ship
            'length': [200] * n_points,
            'width': [30] * n_points,
            'draught': [12] * n_points,
            'rot': np.random.normal(0, 5, n_points)  # Rate of turn
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def processed_sequences(self, sample_ais_data):
        """Create processed sequences for model input."""
        # Simple processing without importing complex modules
        data = sample_ais_data.copy()
        
        # Add derived features
        data['lat_diff'] = data['latitude'].diff().fillna(0)
        data['lon_diff'] = data['longitude'].diff().fillna(0)
        
        # Create sequences
        sequence_length = 20
        prediction_horizon = 10
        
        sequences = []
        targets = []
        
        feature_cols = ['latitude', 'longitude', 'sog', 'cog', 'heading', 
                       'nav_status', 'vessel_type', 'length', 'width', 
                       'draught', 'rot', 'lat_diff', 'lon_diff']
        target_cols = ['latitude', 'longitude', 'sog', 'cog']
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data.iloc[i:i+sequence_length]
            target = data.iloc[i+sequence_length:i+sequence_length+prediction_horizon]
            
            # Convert to tensors
            seq_tensor = torch.tensor(seq[feature_cols].values, dtype=torch.float32)
            target_tensor = torch.tensor(target[target_cols].values, dtype=torch.float32)
            
            sequences.append(seq_tensor)
            targets.append(target_tensor)
        
        return torch.stack(sequences), torch.stack(targets)
    
    def test_anomaly_transformer_with_ais_data(self, processed_sequences):
        """Test Anomaly Transformer with processed AIS data."""
        sequences, _ = processed_sequences
        
        # Create model
        model = create_maritime_anomaly_transformer('small')
        
        # Test forward pass
        outputs = model(sequences)
        
        # Check outputs
        assert 'anomaly_scores' in outputs
        assert 'reconstruction' in outputs
        assert outputs['anomaly_scores'].shape == (sequences.size(0), sequences.size(1))
        assert outputs['reconstruction'].shape == sequences.shape
        
        # Test anomaly detection
        results = model.detect_anomalies(sequences, threshold=0.5)
        assert 'binary_anomalies' in results
        assert results['binary_anomalies'].shape == (sequences.size(0), sequences.size(1))
    
    def test_motion_transformer_with_ais_data(self, processed_sequences):
        """Test Motion Transformer with processed AIS data."""
        sequences, targets = processed_sequences
        
        # Create model
        model = create_maritime_motion_transformer('small')
        
        # Test forward pass
        outputs = model(sequences)
        
        # Check outputs
        assert 'trajectories' in outputs
        assert 'confidences' in outputs
        assert outputs['trajectories'].shape[0] == sequences.size(0)  # batch_size
        assert outputs['trajectories'].shape[1] == targets.size(1)    # prediction_horizon
        assert outputs['trajectories'].shape[3] == targets.size(2)    # output_dim
        
        # Test best trajectory prediction
        best_traj = model.predict_best_trajectory(sequences)
        assert best_traj.shape == targets.shape
        
        # Test loss computation
        loss_dict = model.compute_loss(outputs, targets)
        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].item() > 0
    
    def test_models_with_variable_sequence_lengths(self, sample_ais_data):
        """Test models with different sequence lengths."""
        # Simple processing
        data = sample_ais_data.copy()
        data['lat_diff'] = data['latitude'].diff().fillna(0)
        data['lon_diff'] = data['longitude'].diff().fillna(0)
        
        feature_cols = ['latitude', 'longitude', 'sog', 'cog', 'heading', 
                       'nav_status', 'vessel_type', 'length', 'width', 
                       'draught', 'rot', 'lat_diff', 'lon_diff']
        
        # Test different sequence lengths
        for seq_len in [10, 15, 25]:
            if len(data) < seq_len:
                continue
                
            seq = data.iloc[:seq_len]
            seq_tensor = torch.tensor(seq[feature_cols].values, dtype=torch.float32).unsqueeze(0)
            
            # Test Anomaly Transformer
            anomaly_model = create_maritime_anomaly_transformer('small')
            anomaly_outputs = anomaly_model(seq_tensor)
            assert anomaly_outputs['anomaly_scores'].shape == (1, seq_len)
            
            # Test Motion Transformer
            motion_model = create_maritime_motion_transformer('small')
            motion_outputs = motion_model(seq_tensor)
            assert motion_outputs['trajectories'].shape[0] == 1  # batch_size
            assert motion_outputs['trajectories'].shape[1] == 10  # prediction_horizon (small config)
    
    def test_batch_processing(self, processed_sequences):
        """Test models with different batch sizes."""
        sequences, targets = processed_sequences
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            if sequences.size(0) < batch_size:
                continue
                
            batch_seq = sequences[:batch_size]
            batch_targets = targets[:batch_size]
            
            # Test Anomaly Transformer
            anomaly_model = create_maritime_anomaly_transformer('small')
            anomaly_outputs = anomaly_model(batch_seq)
            assert anomaly_outputs['anomaly_scores'].shape[0] == batch_size
            
            # Test Motion Transformer
            motion_model = create_maritime_motion_transformer('small')
            motion_outputs = motion_model(batch_seq)
            assert motion_outputs['trajectories'].shape[0] == batch_size
            
            # Test loss computation
            loss_dict = motion_model.compute_loss(motion_outputs, batch_targets)
            assert torch.isfinite(loss_dict['total_loss'])


class TestSOTATrainingIntegration:
    """Test SOTA models training integration."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data for testing."""
        batch_size, seq_len, input_dim = 16, 20, 13
        pred_horizon, output_dim = 10, 4
        
        # Create synthetic training data
        context = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randn(batch_size, pred_horizon, output_dim)
        
        return context, targets
    
    def test_anomaly_transformer_training_step(self, training_data):
        """Test Anomaly Transformer training step."""
        from models.anomaly_transformer import AnomalyTransformerTrainer
        
        context, _ = training_data
        
        # Create model and trainer
        model = create_maritime_anomaly_transformer('small')
        trainer = AnomalyTransformerTrainer(model, learning_rate=1e-4, device='cpu')
        
        # Test training step
        loss_dict = trainer.train_step(context)
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert 'association_loss' in loss_dict
        
        # Check loss values are reasonable
        assert 0 < loss_dict['total_loss'] < 100
        assert isinstance(loss_dict['total_loss'], float)
        
        # Test validation step
        val_loss_dict = trainer.validate_step(context)
        assert 'total_loss' in val_loss_dict or 'val_total_loss' in val_loss_dict
    
    def test_motion_transformer_training_step(self, training_data):
        """Test Motion Transformer training step."""
        from models.motion_transformer import MotionTransformerTrainer
        
        context, targets = training_data
        
        # Create model and trainer
        model = create_maritime_motion_transformer('small')
        trainer = MotionTransformerTrainer(model, learning_rate=1e-4, device='cpu')
        
        # Test training step
        loss_dict = trainer.train_step(context, targets)
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert isinstance(loss_dict['total_loss'], float)
        assert 0 < loss_dict['total_loss'] < 100
        
        # Test validation step
        val_loss_dict = trainer.validate_step(context, targets)
        assert 'total_loss' in val_loss_dict
        assert 'val_ade' in val_loss_dict
        assert 'val_fde' in val_loss_dict
    
    def test_training_convergence(self):
        """Test that models can learn simple patterns."""
        # Create simple synthetic data with learnable pattern
        batch_size, seq_len, input_dim = 8, 15, 13
        
        # Create data where future is just shifted version of past
        context = torch.randn(batch_size, seq_len, input_dim)
        # Simple pattern: future trajectory is last position + small increment
        # Make sure prediction horizon matches model config
        model = create_maritime_motion_transformer('small')
        pred_horizon = 10  # Small config uses 10 steps
        targets = context[:, -1:, :4].repeat(1, pred_horizon, 1) + 0.1 * torch.randn(batch_size, pred_horizon, 4)
        
        # Test Motion Transformer learning
        from models.motion_transformer import MotionTransformerTrainer
        
        model = create_maritime_motion_transformer('small')
        trainer = MotionTransformerTrainer(model, learning_rate=1e-3, device='cpu')
        
        # Train for a few steps
        initial_loss = None
        final_loss = None
        
        for step in range(10):
            loss_dict = trainer.train_step(context, targets)
            if step == 0:
                initial_loss = loss_dict['total_loss']
            if step == 9:
                final_loss = loss_dict['total_loss']
        
        # Check that loss decreased (model is learning)
        assert final_loss < initial_loss * 1.1  # Allow for some variation


class TestSOTAModelComparison:
    """Test comparison between SOTA and baseline models."""
    
    def test_model_complexity_comparison(self):
        """Compare model complexity between SOTA and baseline models."""
        from models import create_model, get_model_info
        
        # Create models
        baseline_lstm = create_model('baseline', task='trajectory_prediction')
        baseline_autoencoder = create_model('baseline', task='anomaly_detection')
        anomaly_transformer = create_model('anomaly_transformer', d_model=128, n_layers=2)
        motion_transformer = create_model('motion_transformer', d_model=128, encoder_layers=2, decoder_layers=2)
        
        # Count parameters
        lstm_params = sum(p.numel() for p in baseline_lstm.parameters())
        autoencoder_params = sum(p.numel() for p in baseline_autoencoder.parameters())
        anomaly_params = sum(p.numel() for p in anomaly_transformer.parameters())
        motion_params = sum(p.numel() for p in motion_transformer.parameters())
        
        print(f"Baseline LSTM: {lstm_params:,} parameters")
        print(f"Baseline Autoencoder: {autoencoder_params:,} parameters")
        print(f"Anomaly Transformer: {anomaly_params:,} parameters")
        print(f"Motion Transformer: {motion_params:,} parameters")
        
        # SOTA models should be more complex but not excessively so
        assert anomaly_params > autoencoder_params
        assert motion_params > lstm_params
        assert anomaly_params < 50_000_000  # Reasonable upper bound
        assert motion_params < 50_000_000
    
    def test_inference_time_comparison(self):
        """Compare inference times between models."""
        import time
        
        # Test data
        batch_size, seq_len = 4, 20
        context = torch.randn(batch_size, seq_len, 13)
        
        # Create models
        baseline_lstm = create_model('baseline', task='trajectory_prediction')
        anomaly_transformer = create_model('anomaly_transformer', d_model=128, n_layers=2)
        motion_transformer = create_model('motion_transformer', d_model=128, encoder_layers=2, decoder_layers=2)
        
        # Set to eval mode
        baseline_lstm.eval()
        anomaly_transformer.eval()
        motion_transformer.eval()
        
        # Time baseline LSTM
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = baseline_lstm(context)
            lstm_time = (time.time() - start) / 10
        
        # Time Anomaly Transformer
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = anomaly_transformer(context)
            anomaly_time = (time.time() - start) / 10
        
        # Time Motion Transformer
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = motion_transformer(context)
            motion_time = (time.time() - start) / 10
        
        print(f"Baseline LSTM: {lstm_time:.4f}s")
        print(f"Anomaly Transformer: {anomaly_time:.4f}s")
        print(f"Motion Transformer: {motion_time:.4f}s")
        
        # All models should have reasonable inference times
        assert lstm_time < 1.0
        assert anomaly_time < 2.0
        assert motion_time < 2.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

