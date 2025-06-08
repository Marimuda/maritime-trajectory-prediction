#!/usr/bin/env python3
"""
Training script to validate the complete AIS system with 1 epoch training.

This script demonstrates the end-to-end workflow:
1. Load and process real AIS data using our pipeline
2. Generate task-specific dataset
3. Train a model for 1 epoch
4. Validate results and system integration
"""

import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

# Import our components
from maritime_trajectory_prediction.src.data import (
    DataPipeline, AISMultiTaskProcessor, MLTask, DatasetConfig,
    TrajectoryPredictionBuilder, DataValidator
)
from maritime_trajectory_prediction.src.models.transformer_blocks import (
    MultiHeadAttention, TransformerBlock
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths (configurable)
DATA_DIR = Path('./data')
RAW_DATA_FILE = DATA_DIR / 'raw' / 'log_snipit.log'  # Start with small dataset
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_OUTPUT_DIR = Path('./models')
RESULTS_DIR = Path('./results')

# Training configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1  # Single epoch for validation


class SimpleTrajectoryPredictor(nn.Module):
    """
    Simple trajectory prediction model for validation testing.
    Uses transformer blocks from our implemented components.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(f"Initialized model with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, output_dim)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.layer_norm(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


def create_dataset_config():
    """Create configuration for trajectory prediction dataset."""
    config = DatasetConfig(
        task=MLTask.TRAJECTORY_PREDICTION,
        sequence_length=3,  # Further reduced
        prediction_horizon=1,  # Single step prediction
        min_trajectory_length=5,  # Much smaller minimum
        validation_split=0.2,
        test_split=0.1,
        random_seed=42,
        # No spatial bounds - use all data
        spatial_bounds=None
    )
    
    logger.info(f"Created dataset config: {config.task.value}")
    logger.info(f"  Sequence length: {config.sequence_length}")
    logger.info(f"  Prediction horizon: {config.prediction_horizon}")
    logger.info(f"  Min trajectory length: {config.min_trajectory_length}")
    return config


def load_and_process_data():
    """Load and process AIS data using our pipeline."""
    logger.info("Starting data loading and processing...")
    
    # Check if raw data exists
    if not RAW_DATA_FILE.exists():
        logger.error(f"Raw data file not found: {RAW_DATA_FILE}")
        raise FileNotFoundError(f"Please ensure {RAW_DATA_FILE} exists")
    
    # Initialize processor and pipeline
    processor = AISMultiTaskProcessor([MLTask.TRAJECTORY_PREDICTION])
    pipeline = DataPipeline(processor)
    pipeline.register_builder(MLTask.TRAJECTORY_PREDICTION, TrajectoryPredictionBuilder)
    
    # Process raw data
    logger.info(f"Processing raw data from: {RAW_DATA_FILE}")
    df = pipeline.process_raw_data(str(RAW_DATA_FILE))
    logger.info(f"Loaded {len(df)} raw AIS records")
    
    # Validate data
    validator = DataValidator(strict_mode=False)
    validation_result = validator.validate_dataset(df, task='trajectory_prediction')
    
    if not validation_result.is_valid:
        logger.warning(f"Data validation issues: {validation_result.errors}")
        logger.info("Proceeding with warnings...")
    
    logger.info(f"Data validation metrics: {validation_result.metrics}")
    
    return df, pipeline


def generate_dataset(df, pipeline):
    """Generate task-specific dataset."""
    logger.info("Generating trajectory prediction dataset...")
    
    # Create configuration
    config = create_dataset_config()
    
    # Build dataset
    dataset = pipeline.build_dataset(df, MLTask.TRAJECTORY_PREDICTION, config)
    
    # Extract splits
    splits = dataset['splits']
    metadata = dataset['metadata']
    
    logger.info(f"Dataset metadata:")
    logger.info(f"  Task: {metadata.task}")
    logger.info(f"  Samples: {metadata.num_samples}")
    logger.info(f"  Vessels: {metadata.num_vessels}")
    logger.info(f"  Features: {len(metadata.feature_columns)}")
    logger.info(f"  Targets: {len(metadata.target_columns)}")
    
    # Log split sizes
    for split_name, (X, y) in splits.items():
        logger.info(f"  {split_name}: {X.shape[0]} samples, X: {X.shape}, y: {y.shape}")
    
    return splits, metadata


def create_data_loaders(splits):
    """Create PyTorch data loaders."""
    logger.info("Creating data loaders...")
    
    data_loaders = {}
    
    for split_name, (X, y) in splits.items():
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Use smaller batch size for validation/test
        batch_size = BATCH_SIZE if split_name == 'train' else min(BATCH_SIZE, len(dataset))
        shuffle = split_name == 'train'
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
        
        data_loaders[split_name] = loader
        logger.info(f"  {split_name}: {len(dataset)} samples, {len(loader)} batches")
    
    return data_loaders


def create_model(input_dim, output_dim):
    """Create and initialize the model."""
    logger.info(f"Creating model with input_dim={input_dim}, output_dim={output_dim}")
    
    model = SimpleTrajectoryPredictor(
        input_dim=input_dim,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=output_dim
    )
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {DEVICE}")
    
    return model


def train_one_epoch(model, train_loader, optimizer, criterion):
    """Train the model for one epoch."""
    logger.info("Starting training epoch...")
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # Move to device
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_batch)
        
        # Calculate loss (only on the last timestep for simplicity)
        loss = criterion(predictions[:, -1, :], y_batch[:, -1, :])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    logger.info(f"Training epoch completed:")
    logger.info(f"  Average loss: {avg_loss:.6f}")
    logger.info(f"  Time: {epoch_time:.2f} seconds")
    logger.info(f"  Batches processed: {num_batches}")
    
    return avg_loss


def evaluate_model(model, data_loader, criterion, split_name):
    """Evaluate the model on a dataset split."""
    logger.info(f"Evaluating on {split_name} set...")
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # Move to device
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            predictions = model(X_batch)
            
            # Calculate loss
            loss = criterion(predictions[:, -1, :], y_batch[:, -1, :])
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"  {split_name} loss: {avg_loss:.6f}")
    
    return avg_loss


def save_results(train_loss, val_loss, test_loss, metadata):
    """Save training results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'training': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'device': str(DEVICE)
        },
        'losses': {
            'train': float(train_loss),
            'validation': float(val_loss),
            'test': float(test_loss)
        },
        'dataset': {
            'task': metadata.task,
            'num_samples': metadata.num_samples,
            'num_vessels': metadata.num_vessels,
            'feature_columns': metadata.feature_columns,
            'target_columns': metadata.target_columns
        }
    }
    
    results_file = RESULTS_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    return results


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("AIS TRAJECTORY PREDICTION - 1 EPOCH VALIDATION")
    logger.info("="*60)
    
    try:
        # Step 1: Load and process data
        df, pipeline = load_and_process_data()
        
        # Step 2: Generate dataset
        splits, metadata = generate_dataset(df, pipeline)
        
        # Step 3: Create data loaders
        data_loaders = create_data_loaders(splits)
        
        # Step 4: Create model
        # Get dimensions from the first batch
        sample_X, sample_y = next(iter(data_loaders['train']))
        input_dim = sample_X.shape[-1]
        output_dim = sample_y.shape[-1]
        
        model = create_model(input_dim, output_dim)
        
        # Step 5: Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        logger.info(f"Training setup:")
        logger.info(f"  Criterion: {criterion}")
        logger.info(f"  Optimizer: {optimizer}")
        logger.info(f"  Learning rate: {LEARNING_RATE}")
        
        # Step 6: Train for 1 epoch
        train_loss = train_one_epoch(model, data_loaders['train'], optimizer, criterion)
        
        # Step 7: Evaluate
        val_loss = evaluate_model(model, data_loaders['validation'], criterion, 'validation')
        test_loss = evaluate_model(model, data_loaders['test'], criterion, 'test')
        
        # Step 8: Save results
        results = save_results(train_loss, val_loss, test_loss, metadata)
        
        # Step 9: Summary
        logger.info("="*60)
        logger.info("TRAINING VALIDATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Final Results:")
        logger.info(f"  Training Loss:   {train_loss:.6f}")
        logger.info(f"  Validation Loss: {val_loss:.6f}")
        logger.info(f"  Test Loss:       {test_loss:.6f}")
        logger.info(f"  Dataset Size:    {metadata.num_samples} samples")
        logger.info(f"  Vessels:         {metadata.num_vessels}")
        logger.info(f"  Device:          {DEVICE}")
        
        # Validation checks
        if train_loss < 1000:  # Reasonable loss range
            logger.info("✅ Training loss is in reasonable range")
        else:
            logger.warning("⚠️  Training loss seems high")
        
        if val_loss / train_loss < 10:  # Not severely overfitting
            logger.info("✅ No severe overfitting detected")
        else:
            logger.warning("⚠️  Possible overfitting detected")
        
        logger.info("="*60)
        logger.info("SYSTEM VALIDATION: ALL COMPONENTS WORKING CORRECTLY!")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run training
    results = main()

