import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pytorch_lightning as pl
from src.utils.metrics import haversine_distance

class TrajectoryEvaluator:
    def __init__(self, model, data_module, config):
        """
        Initialize the trajectory evaluator
        
        Args:
            model: Trained model for prediction
            data_module: Data module containing test data
            config: Evaluation configuration
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        
        # Put model in evaluation mode
        self.model.eval()
        
    def generate_trajectories(self, input_sequence, n_samples=100, max_steps=12):
        """
        Generate multiple trajectory predictions using stochastic sampling
        
        Args:
            input_sequence: Starting sequence for prediction
            n_samples: Number of trajectories to generate
            max_steps: Maximum number of prediction steps
            
        Returns:
            List of predicted trajectories
        """
        self.model.eval()
        with torch.no_grad():
            # Create n_samples copies of the input
            batch_inputs = [input_sequence for _ in range(n_samples)]
            
            # Initialize output trajectories
            trajectories = []
            
            for sample_idx in range(n_samples):
                # Copy input as starting point
                traj = input_sequence.clone()
                
                # Generate future steps
                for step in range(max_steps):
                    # Get next step prediction
                    next_step = self.model.predict_step(traj)
                    
                    # Append to trajectory
                    traj = torch.cat([traj, next_step.unsqueeze(0)], dim=0)
                
                trajectories.append(traj)
            
            return trajectories
    
    def best_of_n_evaluation(self, test_dataset, n_samples=100, horizons=[6, 12, 18, 24]):
        """
        Evaluate using best-of-N methodology at different prediction horizons
        
        Args:
            test_dataset: Dataset with test examples
            n_samples: Number of trajectories to sample per test case
            horizons: List of future horizons (timesteps) to evaluate
            
        Returns:
            Dictionary of metrics at each horizon
        """
        results = {horizon: [] for horizon in horizons}
        
        for i, (input_seq, target_seq) in enumerate(test_dataset):
            # Generate multiple trajectories
            predicted_trajectories = self.generate_trajectories(
                input_seq, n_samples=n_samples, max_steps=max(horizons)
            )
            
            # Evaluate at each horizon
            for horizon in horizons:
                # Extract trajectories up to this horizon
                pred_positions = [traj[:horizon, :2] for traj in predicted_trajectories]  # lat, lon
                true_position = target_seq[:horizon, :2]  # lat, lon
                
                # Calculate errors for each sample
                errors = []
                for pred in pred_positions:
                    # Calculate Haversine distance
                    error = haversine_distance(
                        pred[-1, 0], pred[-1, 1],
                        true_position[-1, 0], true_position[-1, 1]
                    )
                    errors.append(error)
                
                # Get best trajectory (minimum error)
                min_error = min(errors)
                results[horizon].append(min_error)
        
        # Compute statistics
        metrics = {}
        for horizon in horizons:
            horizon_errors = results[horizon]
            metrics[f"horizon_{horizon}"] = {
                "mean": np.mean(horizon_errors),
                "median": np.median(horizon_errors),
                "p90": np.percentile(horizon_errors, 90),
                "max": np.max(horizon_errors)
            }
        
        return metrics
    
    def visualize_predictions(self, input_seq, target_seq, n_samples=10, save_path=None):
        """
        Visualize multiple predicted trajectories against the ground truth
        
        Args:
            input_seq: Input sequence for prediction
            target_seq: Ground truth future trajectory
            n_samples: Number of trajectories to visualize
            save_path: Path to save the visualization
        """
        # Generate predictions
        predicted_trajectories = self.generate_trajectories(
            input_seq, n_samples=n_samples, max_steps=len(target_seq)
        )
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot input sequence
        plt.plot(
            input_seq[:, 1].cpu().numpy(), 
            input_seq[:, 0].cpu().numpy(), 
            'b-', linewidth=2, label='Input'
        )
        
        # Plot ground truth
        plt.plot(
            target_seq[:, 1].cpu().numpy(), 
            target_seq[:, 0].cpu().numpy(), 
            'g-', linewidth=2, label='Ground Truth'
        )
        
        # Plot predicted trajectories
        for i, traj in enumerate(predicted_trajectories):
            if i == 0:
                plt.plot(
                    traj[:, 1].cpu().numpy(), 
                    traj[:, 0].cpu().numpy(), 
                    'r-', alpha=0.3, label='Predictions'
                )
            else:
                plt.plot(
                    traj[:, 1].cpu().numpy(), 
                    traj[:, 0].cpu().numpy(), 
                    'r-', alpha=0.3
                )
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('AIS Trajectory Prediction')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
