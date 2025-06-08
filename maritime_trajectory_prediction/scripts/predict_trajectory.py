#!/usr/bin/env python3
"""
Script to predict vessel trajectories using trained models.
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from maritime_trajectory_prediction.src.utils.ais_parser import load_processed_ais_data, get_vessel_trajectories, prepare_trajectory_for_prediction
from maritime_trajectory_prediction.src.models.model_factory import load_model
from maritime_trajectory_prediction.src.utils.visualization import plot_prediction_vs_ground_truth


def predict_trajectory(model, input_sequence, num_steps=12, num_samples=10):
    """
    Predict future trajectory using a trained model.
    
    Args:
        model: Trained trajectory prediction model
        input_sequence: Input sequence data
        num_steps: Number of steps to predict
        num_samples: Number of trajectory samples to generate
        
    Returns:
        List of predicted trajectories
    """
    model.eval()
    
    # Convert input to tensor if needed
    if not isinstance(input_sequence, torch.Tensor):
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
    
    # Add batch dimension if needed
    if len(input_sequence.shape) == 2:
        input_sequence = input_sequence.unsqueeze(0)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_sequence = input_sequence.to(device)
    
    # Generate multiple trajectory samples
    with torch.no_grad():
        predicted_trajectories = []
        
        for _ in range(num_samples):
            # For TrAISformer or similar models that support stochastic sampling
            if hasattr(model, 'predict_trajectory'):
                pred_traj = model.predict_trajectory(input_sequence, steps=num_steps)
            else:
                # For simpler models that predict one step at a time
                current_input = input_sequence.clone()
                predictions = []
                
                for _ in range(num_steps):
                    # Predict next step
                    next_step = model(current_input)
                    
                    # Extract prediction (assume model returns [batch, features])
                    pred = next_step[0] if isinstance(next_step, tuple) else next_step
                    
                    # Add to predictions
                    predictions.append(pred[:, -1, :])  # Last time step
                    
                    # Update input for next step (remove oldest, add prediction)
                    if len(current_input.shape) == 3:  # [batch, seq_len, features]
                        new_input = torch.cat([
                            current_input[:, 1:, :],  # Remove oldest step
                            pred[:, -1, :].unsqueeze(1)  # Add prediction as new step
                        ], dim=1)
                        current_input = new_input
                
                # Combine predictions into trajectory
                pred_traj = torch.cat([p.unsqueeze(1) for p in predictions], dim=1)
            
            # Convert to numpy and add to results
            predicted_trajectories.append(pred_traj.cpu().numpy())
    
    return predicted_trajectories

def evaluate_predictions(predicted_trajectories, ground_truth):
    """
    Evaluate trajectory predictions against ground truth.
    
    Args:
        predicted_trajectories: List of predicted trajectories
        ground_truth: Ground truth trajectory
        
    Returns:
        Dictionary of evaluation metrics
    """
    from maritime_trajectory_prediction.src.utils.metrics import haversine_distance, rmse_error
    
    if not predicted_trajectories:
        return {}
    
    # Extract positions (lat, lon) from ground truth
    gt_positions = ground_truth[:, :2] if len(ground_truth.shape) == 2 else ground_truth[:, :, :2]
    
    # Calculate errors for each predicted trajectory
    errors = []
    
    for pred_traj in predicted_trajectories:
        # Extract positions from prediction
        pred_positions = pred_traj[0] if len(pred_traj.shape) == 3 else pred_traj
        pred_positions = pred_positions[:, :2] if len(pred_positions.shape) == 2 else pred_positions[:, :, :2]
        
        # Limit to minimum length
        min_len = min(len(gt_positions), len(pred_positions))
        gt_pos = gt_positions[:min_len]
        pred_pos = pred_positions[:min_len]
        
        # Calculate RMSE
        rmse = rmse_error(torch.tensor(pred_pos), torch.tensor(gt_pos))
        
        # Calculate Haversine distances at each step
        distances = []
        for i in range(min_len):
            dist = haversine_distance(
                gt_pos[i, 0], gt_pos[i, 1],
                pred_pos[i, 0], pred_pos[i, 1]
            )
            distances.append(dist)
        
        errors.append({
            'rmse': rmse.item(),
            'distances': distances,
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances)
        })
    
    # Find best trajectory (minimum average distance)
    best_idx = np.argmin([e['mean_distance'] for e in errors])
    best_error = errors[best_idx]
    
    # Compute overall metrics
    metrics = {
        'best_rmse': best_error['rmse'],
        'best_mean_distance': best_error['mean_distance'],
        'best_max_distance': best_error['max_distance'],
        'best_trajectory_idx': best_idx,
        'mean_rmse': np.mean([e['rmse'] for e in errors]),
        'min_rmse': np.min([e['rmse'] for e in errors]),
        'distances_by_horizon': {
            f"horizon_{i+1}": np.mean([e['distances'][i] if i < len(e['distances']) else np.nan 
                                      for e in errors]) 
            for i in range(max([len(e['distances']) for e in errors]))
        }
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Predict vessel trajectories using trained models")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data", required=True, help="Path to processed AIS data file")
    parser.add_argument("--mmsi", type=int, help="MMSI of vessel to predict (if not specified, uses first vessel)")
    parser.add_argument("--steps", type=int, default=12, help="Number of steps to predict")
    parser.add_argument("--samples", type=int, default=10, help="Number of trajectory samples to generate")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--output", help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Load data
    print(f"Loading AIS data from {args.data}")
    ais_data = load_processed_ais_data(args.data)
    
    # Get vessel trajectories
    trajectories = get_vessel_trajectories(ais_data, mmsi=args.mmsi)
    
    if not trajectories:
        print("No trajectories found")
        return
    
    # Use first trajectory if MMSI not specified
    if args.mmsi is None:
        trajectory = trajectories[0]
        mmsi = trajectory['mmsi'].iloc[0]
        print(f"Using first trajectory (MMSI: {mmsi})")
    else:
        # Find trajectory for specified MMSI
        matching_trajectories = [t for t in trajectories if t['mmsi'].iloc[0] == args.mmsi]
        if not matching_trajectories:
            print(f"No trajectory found for MMSI {args.mmsi}")
            return
        
        trajectory = matching_trajectories[0]
        mmsi = args.mmsi
    
    # Get vessel info
    vessel_name = trajectory['name'].iloc[0] if 'name' in trajectory.columns and not pd.isna(trajectory['name'].iloc[0]) else f"MMSI: {mmsi}"
    vessel_type = trajectory['ship_type_text'].iloc[0] if 'ship_type_text' in trajectory.columns and not pd.isna(trajectory['ship_type_text'].iloc[0]) else "Unknown"
    
    print(f"Vessel: {vessel_name}")
    print(f"Type: {vessel_type}")
    print(f"Trajectory length: {len(trajectory)} points")
    
    # Split trajectory for testing (use last N points as ground truth)
    test_size = min(args.steps, len(trajectory) // 3)  # Use up to 1/3 of trajectory for testing
    train_trajectory = trajectory.iloc[:-test_size].copy()
    test_trajectory = trajectory.iloc[-test_size:].copy()
    
    print(f"Using {len(train_trajectory)} points for input, {len(test_trajectory)} points for validation")
    
    # Prepare input sequence
    try:
        input_sequence = prepare_trajectory_for_prediction(train_trajectory)
        print(f"Input sequence shape: {input_sequence.shape}")
    except ValueError as e:
        print(f"Error preparing input sequence: {e}")
        return
    
    # Prepare ground truth
    features = ['lat', 'lon', 'sog', 'cog', 'distance_km', 'speed_delta', 'course_delta']
    ground_truth = test_trajectory[features].values
    
    # Predict trajectory
    print(f"Generating {args.samples} trajectory predictions for {args.steps} steps")
    predicted_trajectories = predict_trajectory(
        model,
        input_sequence,
        num_steps=args.steps,
        num_samples=args.samples
    )
    
    # Evaluate predictions
    metrics = evaluate_predictions(predicted_trajectories, ground_truth)
    
    print("\nPrediction Metrics:")
    print(f"Best RMSE: {metrics.get('best_rmse', 'N/A'):.4f}")
    print(f"Best Mean Distance: {metrics.get('best_mean_distance', 'N/A'):.4f} km")
    print(f"Best Max Distance: {metrics.get('best_max_distance', 'N/A'):.4f} km")
    
    # Print distance by horizon
    print("\nDistance by Horizon:")
    for horizon, distance in metrics.get('distances_by_horizon', {}).items():
        print(f"{horizon}: {distance:.4f} km")
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing predictions...")
        
        # Get best trajectory
        best_idx = metrics.get('best_trajectory_idx', 0)
        best_prediction = predicted_trajectories[best_idx]
        
        # Extract positions
        input_positions = input_sequence[:, :2]
        gt_positions = ground_truth[:, :2]
        pred_positions = best_prediction[0, :, :2] if len(best_prediction.shape) == 3 else best_prediction[:, :2]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot input trajectory
        plt.plot(input_positions[:, 1], input_positions[:, 0], 'b-', linewidth=2, label='Input Trajectory')
        
        # Plot ground truth
        plt.plot(gt_positions[:, 1], gt_positions[:, 0], 'g-', linewidth=2, label='Ground Truth')
        
        # Plot prediction
        plt.plot(pred_positions[:, 1], pred_positions[:, 0], 'r-', linewidth=2, label='Prediction')
        
        # Plot all other predictions with transparency
        for i, pred_traj in enumerate(predicted_trajectories):
            if i != best_idx:
                pred_pos = pred_traj[0, :, :2] if len(pred_traj.shape) == 3 else pred_traj[:, :2]
                plt.plot(pred_pos[:, 1], pred_pos[:, 0], 'r-', alpha=0.2)
        
        # Add labels and legend
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajectory Prediction for {vessel_name}')
        plt.grid(True)
        plt.legend()
        
        # Save if output specified
        if args.output:
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {args.output}")
        
        plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()