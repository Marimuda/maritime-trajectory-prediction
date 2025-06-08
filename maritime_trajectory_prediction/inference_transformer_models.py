"""
Inference script for SOTA models.

This script provides unified inference capabilities for both baseline and SOTA models
with support for real-time processing, batch inference, and model comparison.
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import create_model, get_model_info
from models.anomaly_transformer import create_maritime_anomaly_transformer
from models.motion_transformer import create_maritime_motion_transformer
from data.ais_processor import AISProcessor
from utils.maritime_utils import MaritimeUtils
from utils.visualization import TrajectoryVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SOTAInference:
    """Unified inference engine for SOTA and baseline models."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and configuration
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
        
        # Create model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup data processor
        self.processor = AISProcessor()
        
        # Setup visualizer
        self.visualizer = TrajectoryVisualizer()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model type: {self.config.get('model', {}).get('type', 'unknown')}")
        logger.info(f"Device: {self.device}")
    
    def _create_model(self):
        """Create model based on configuration."""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type')
        
        if model_type == 'anomaly_transformer':
            size = model_config.get('size', 'medium')
            return create_maritime_anomaly_transformer(size).to(self.device)
        
        elif model_type == 'motion_transformer':
            size = model_config.get('size', 'medium')
            return create_maritime_motion_transformer(size).to(self.device)
        
        elif model_type == 'baseline':
            task = model_config.get('task')
            custom_params = model_config.get('custom_params', {})
            return create_model('baseline', task=task, **custom_params).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def preprocess_data(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input data for inference.
        
        Args:
            data: Input data (DataFrame, numpy array, or tensor)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(data, pd.DataFrame):
            # Process AIS DataFrame
            processed_data = self.processor.process_dataframe(data)
            
            # Extract features
            feature_cols = ['latitude', 'longitude', 'sog', 'cog', 'heading', 
                           'nav_status', 'vessel_type', 'length', 'width', 
                           'draught', 'rot', 'lat_diff', 'lon_diff']
            
            tensor_data = torch.tensor(processed_data[feature_cols].values, dtype=torch.float32)
            
            # Add batch dimension if needed
            if tensor_data.dim() == 2:
                tensor_data = tensor_data.unsqueeze(0)
        
        elif isinstance(data, np.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32)
            if tensor_data.dim() == 2:
                tensor_data = tensor_data.unsqueeze(0)
        
        elif isinstance(data, torch.Tensor):
            tensor_data = data.float()
            if tensor_data.dim() == 2:
                tensor_data = tensor_data.unsqueeze(0)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return tensor_data.to(self.device)
    
    def predict_anomalies(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor], 
                         threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect anomalies in vessel trajectories.
        
        Args:
            data: Input trajectory data
            threshold: Anomaly detection threshold
            
        Returns:
            Dictionary containing anomaly detection results
        """
        model_type = self.config.get('model', {}).get('type')
        
        if model_type != 'anomaly_transformer' and not (model_type == 'baseline' and 
                                                       self.config.get('model', {}).get('task') == 'anomaly_detection'):
            raise ValueError(f"Model type {model_type} does not support anomaly detection")
        
        # Preprocess data
        input_tensor = self.preprocess_data(data)
        
        with torch.no_grad():
            if model_type == 'anomaly_transformer':
                results = self.model.detect_anomalies(input_tensor, threshold=threshold)
            else:  # baseline anomaly detection
                outputs = self.model(input_tensor)
                # Convert baseline outputs to anomaly detection format
                anomaly_scores = torch.sigmoid(outputs).cpu().numpy()
                binary_anomalies = (anomaly_scores > threshold).astype(int)
                
                results = {
                    'anomaly_scores': torch.tensor(anomaly_scores),
                    'binary_anomalies': torch.tensor(binary_anomalies),
                    'reconstruction_error': torch.zeros_like(torch.tensor(anomaly_scores)),
                    'confidence': torch.tensor(anomaly_scores)
                }
        
        # Convert to numpy for easier handling
        numpy_results = {}
        for key, value in results.items():
            if torch.is_tensor(value):
                numpy_results[key] = value.cpu().numpy()
            else:
                numpy_results[key] = value
        
        return numpy_results
    
    def predict_trajectories(self, context_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                           prediction_horizon: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict future vessel trajectories.
        
        Args:
            context_data: Historical trajectory data
            prediction_horizon: Number of future steps to predict
            
        Returns:
            Dictionary containing trajectory predictions
        """
        model_type = self.config.get('model', {}).get('type')
        
        if model_type not in ['motion_transformer'] and not (model_type == 'baseline' and 
                                                           self.config.get('model', {}).get('task') == 'trajectory_prediction'):
            raise ValueError(f"Model type {model_type} does not support trajectory prediction")
        
        # Preprocess data
        input_tensor = self.preprocess_data(context_data)
        
        with torch.no_grad():
            if model_type == 'motion_transformer':
                outputs = self.model(input_tensor)
                
                # Get best trajectory
                best_trajectory = self.model.predict_best_trajectory(input_tensor)
                
                results = {
                    'trajectories': outputs['trajectories'].cpu().numpy(),
                    'confidences': outputs['confidences'].cpu().numpy(),
                    'best_trajectory': best_trajectory.cpu().numpy(),
                    'context_features': outputs['context_features'].cpu().numpy()
                }
            
            else:  # baseline trajectory prediction
                outputs = self.model(input_tensor)
                
                results = {
                    'trajectories': outputs.cpu().numpy(),
                    'best_trajectory': outputs.cpu().numpy(),
                    'confidences': np.ones((input_tensor.size(0), 1))  # Single mode
                }
        
        return results
    
    def batch_inference(self, data_list: List[Union[pd.DataFrame, np.ndarray, torch.Tensor]],
                       task: str = 'auto', **kwargs) -> List[Dict[str, Any]]:
        """
        Perform batch inference on multiple data samples.
        
        Args:
            data_list: List of input data samples
            task: Task type ('anomaly_detection', 'trajectory_prediction', or 'auto')
            **kwargs: Additional arguments for specific tasks
            
        Returns:
            List of prediction results
        """
        if task == 'auto':
            model_type = self.config.get('model', {}).get('type')
            if model_type == 'anomaly_transformer':
                task = 'anomaly_detection'
            elif model_type == 'motion_transformer':
                task = 'trajectory_prediction'
            else:
                task = self.config.get('model', {}).get('task', 'trajectory_prediction')
        
        results = []
        
        for i, data in enumerate(data_list):
            try:
                if task == 'anomaly_detection':
                    result = self.predict_anomalies(data, **kwargs)
                elif task == 'trajectory_prediction':
                    result = self.predict_trajectories(data, **kwargs)
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def real_time_inference(self, data_stream, task: str = 'auto', 
                          callback=None, **kwargs):
        """
        Perform real-time inference on streaming data.
        
        Args:
            data_stream: Iterator or generator yielding data samples
            task: Task type
            callback: Optional callback function for results
            **kwargs: Additional arguments
        """
        if task == 'auto':
            model_type = self.config.get('model', {}).get('type')
            if model_type == 'anomaly_transformer':
                task = 'anomaly_detection'
            elif model_type == 'motion_transformer':
                task = 'trajectory_prediction'
            else:
                task = self.config.get('model', {}).get('task', 'trajectory_prediction')
        
        logger.info(f"Starting real-time inference for task: {task}")
        
        for i, data in enumerate(data_stream):
            start_time = time.time()
            
            try:
                if task == 'anomaly_detection':
                    result = self.predict_anomalies(data, **kwargs)
                elif task == 'trajectory_prediction':
                    result = self.predict_trajectories(data, **kwargs)
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                inference_time = time.time() - start_time
                result['inference_time'] = inference_time
                result['sample_id'] = i
                
                if callback:
                    callback(result)
                else:
                    logger.info(f"Sample {i}: Inference time: {inference_time:.4f}s")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
    
    def compare_models(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                      other_model_paths: List[str], task: str = 'auto') -> Dict[str, Any]:
        """
        Compare predictions from multiple models.
        
        Args:
            data: Input data
            other_model_paths: List of paths to other model checkpoints
            task: Task type
            
        Returns:
            Comparison results
        """
        # Get prediction from current model
        if task == 'auto':
            model_type = self.config.get('model', {}).get('type')
            if model_type == 'anomaly_transformer':
                task = 'anomaly_detection'
            elif model_type == 'motion_transformer':
                task = 'trajectory_prediction'
            else:
                task = self.config.get('model', {}).get('task', 'trajectory_prediction')
        
        if task == 'anomaly_detection':
            current_result = self.predict_anomalies(data)
        else:
            current_result = self.predict_trajectories(data)
        
        results = {
            'current_model': {
                'path': 'current',
                'type': self.config.get('model', {}).get('type'),
                'result': current_result
            },
            'other_models': []
        }
        
        # Compare with other models
        for model_path in other_model_paths:
            try:
                other_inference = SOTAInference(model_path)
                
                if task == 'anomaly_detection':
                    other_result = other_inference.predict_anomalies(data)
                else:
                    other_result = other_inference.predict_trajectories(data)
                
                results['other_models'].append({
                    'path': model_path,
                    'type': other_inference.config.get('model', {}).get('type'),
                    'result': other_result
                })
                
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {e}")
                results['other_models'].append({
                    'path': model_path,
                    'error': str(e)
                })
        
        return results
    
    def export_predictions(self, results: Dict[str, Any], output_path: str, 
                          format: str = 'csv'):
        """
        Export prediction results to file.
        
        Args:
            results: Prediction results
            output_path: Output file path
            format: Output format ('csv', 'json', 'npz')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'csv':
            # Convert results to DataFrame
            if 'anomaly_scores' in results:
                # Anomaly detection results
                df = pd.DataFrame({
                    'anomaly_score': results['anomaly_scores'].flatten(),
                    'binary_anomaly': results['binary_anomalies'].flatten(),
                    'confidence': results['confidence'].flatten()
                })
            elif 'best_trajectory' in results:
                # Trajectory prediction results
                best_traj = results['best_trajectory']
                if best_traj.ndim == 3:  # [batch, time, features]
                    batch_size, time_steps, features = best_traj.shape
                    data_dict = {}
                    for t in range(time_steps):
                        for f in range(features):
                            data_dict[f'step_{t}_feature_{f}'] = best_traj[:, t, f]
                    df = pd.DataFrame(data_dict)
                else:
                    df = pd.DataFrame(best_traj)
            
            df.to_csv(output_path, index=False)
        
        elif format == 'json':
            import json
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif format == 'npz':
            np.savez(output_path, **results)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported results to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='SOTA model inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config-path', type=str, help='Path to model configuration')
    parser.add_argument('--data-path', type=str, required=True, help='Path to input data')
    parser.add_argument('--output-path', type=str, help='Path to save results')
    parser.add_argument('--task', type=str, choices=['anomaly_detection', 'trajectory_prediction', 'auto'],
                       default='auto', help='Inference task')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly detection threshold')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'npz'], default='csv',
                       help='Output format')
    parser.add_argument('--compare-models', type=str, nargs='+', help='Paths to other models for comparison')
    parser.add_argument('--real-time', action='store_true', help='Real-time inference mode')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = SOTAInference(args.model_path, args.config_path)
    
    # Load data
    if args.data_path.endswith('.csv'):
        data = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.parquet'):
        data = pd.read_parquet(args.data_path)
    elif args.data_path.endswith('.npz'):
        data = np.load(args.data_path)['data']
    else:
        raise ValueError(f"Unsupported data format: {args.data_path}")
    
    # Perform inference
    if args.compare_models:
        results = inference.compare_models(data, args.compare_models, args.task)
    elif args.real_time:
        # For real-time, we'll process data in chunks
        def data_stream():
            chunk_size = args.batch_size
            for i in range(0, len(data), chunk_size):
                yield data.iloc[i:i+chunk_size] if isinstance(data, pd.DataFrame) else data[i:i+chunk_size]
        
        inference.real_time_inference(data_stream(), args.task, threshold=args.threshold)
        return
    else:
        if args.task == 'anomaly_detection' or (args.task == 'auto' and 
                                               inference.config.get('model', {}).get('type') == 'anomaly_transformer'):
            results = inference.predict_anomalies(data, threshold=args.threshold)
        else:
            results = inference.predict_trajectories(data)
    
    # Export results
    if args.output_path:
        if args.compare_models:
            # Save comparison results
            import json
            with open(args.output_path, 'w') as f:
                # Convert numpy arrays to lists for JSON
                json_results = {}
                for key, value in results.items():
                    if key == 'current_model' or key == 'other_models':
                        json_results[key] = value  # These should already be serializable
                    else:
                        json_results[key] = value
                json.dump(json_results, f, indent=2, default=str)
        else:
            inference.export_predictions(results, args.output_path, args.format)
    
    logger.info("Inference completed successfully")


if __name__ == '__main__':
    main()

