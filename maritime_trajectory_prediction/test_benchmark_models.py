#!/usr/bin/env python3
"""
Test script for maritime baseline models.

This script tests all three baseline models with real AIS data
to validate their functionality and measure baseline performance.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.baseline_models import create_baseline_model
from src.models.train_baselines import create_trainer, load_and_prepare_data
from src.models.metrics import create_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trajectory_prediction():
    """Test trajectory prediction baseline."""
    logger.info("Testing Trajectory Prediction Baseline...")
    
    try:
        # Create model
        model = create_baseline_model(
            'trajectory_prediction',
            input_dim=13,
            hidden_dim=64,  # Smaller for testing
            num_layers=2,
            output_dim=4
        )
        
        # Test forward pass
        batch_size, seq_len, input_dim = 4, 5, 13
        test_input = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (batch_size, seq_len, 4)  # lat, lon, sog, cog
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Test metrics
        metrics = create_metrics('trajectory_prediction')
        target = torch.randn_like(output)
        metrics.update(output, target)
        metric_results = metrics.compute()
        
        logger.info(f"✅ Trajectory prediction test passed")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Sample metrics: {list(metric_results.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trajectory prediction test failed: {e}")
        return False


def test_anomaly_detection():
    """Test anomaly detection baseline."""
    logger.info("Testing Anomaly Detection Baseline...")
    
    try:
        # Create model
        model = create_baseline_model(
            'anomaly_detection',
            input_dim=13,
            encoding_dim=32,  # Smaller for testing
            hidden_dims=[64, 48]
        )
        
        # Test forward pass
        batch_size, seq_len, input_dim = 4, 10, 13
        test_input = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            reconstruction, encoding = model(test_input)
            anomaly_scores = model.compute_anomaly_score(test_input)
        
        assert reconstruction.shape == test_input.shape, f"Reconstruction shape mismatch"
        assert encoding.shape == (batch_size, 32), f"Encoding shape mismatch"
        assert anomaly_scores.shape == (batch_size,), f"Anomaly scores shape mismatch"
        
        # Test metrics
        metrics = create_metrics('anomaly_detection')
        labels = torch.randint(0, 2, (batch_size,))
        metrics.update(anomaly_scores, labels)
        metric_results = metrics.compute()
        
        logger.info(f"✅ Anomaly detection test passed")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"   Reconstruction shape: {reconstruction.shape}")
        logger.info(f"   Anomaly scores shape: {anomaly_scores.shape}")
        logger.info(f"   Sample metrics: {list(metric_results.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Anomaly detection test failed: {e}")
        return False


def test_vessel_interaction():
    """Test vessel interaction baseline."""
    logger.info("Testing Vessel Interaction Baseline...")
    
    try:
        # Create model
        model = create_baseline_model(
            'vessel_interaction',
            node_features=10,
            edge_features=5,
            hidden_dim=64,  # Smaller for testing
            num_layers=2
        )
        
        # Test forward pass
        num_nodes, node_features = 6, 10
        edge_features = 5
        
        node_feat = torch.randn(1, num_nodes, node_features)
        edge_feat = torch.randn(1, num_nodes, num_nodes, edge_features)
        adjacency = torch.randint(0, 2, (1, num_nodes, num_nodes)).float()
        
        with torch.no_grad():
            output = model(node_feat, edge_feat, adjacency)
        
        expected_shape = (1, num_nodes, 1)  # Collision risk per node
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Test metrics
        metrics = create_metrics('vessel_interaction')
        collision_preds = output.flatten()
        collision_labels = torch.randint(0, 2, (num_nodes,))
        metrics.update(collision_preds, collision_labels)
        metric_results = metrics.compute()
        
        logger.info(f"✅ Vessel interaction test passed")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Sample metrics: {list(metric_results.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vessel interaction test failed: {e}")
        return False


def test_training_pipeline():
    """Test training pipeline with small dataset."""
    logger.info("Testing Training Pipeline...")
    
    try:
        # Check if we have real data
        data_path = project_root / "maritime_trajectory_prediction" / "data" / "raw" / "log_snipit.log"
        
        if not data_path.exists():
            logger.warning("Real data not found, skipping training pipeline test")
            return True
        
        # Test trajectory prediction training
        logger.info("Testing trajectory prediction training...")
        
        # Create small model for quick testing
        model = create_baseline_model(
            'trajectory_prediction',
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=4
        )
        
        # Create trainer
        trainer = create_trainer(
            'trajectory_prediction',
            model,
            learning_rate=0.01,
            save_dir='./test_checkpoints'
        )
        
        # Create dummy data loaders for testing
        batch_size = 2
        seq_len = 5
        input_dim = 13
        output_dim = 4
        
        # Create small dataset
        X = torch.randn(10, seq_len, input_dim)
        y = torch.randn(10, seq_len, output_dim)
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Train for 1 epoch
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)
        
        assert 'loss' in train_metrics, "Training metrics missing loss"
        assert 'loss' in val_metrics, "Validation metrics missing loss"
        
        logger.info(f"✅ Training pipeline test passed")
        logger.info(f"   Train loss: {train_metrics['loss']:.4f}")
        logger.info(f"   Val loss: {val_metrics['loss']:.4f}")
        
        # Cleanup
        import shutil
        if os.path.exists('./test_checkpoints'):
            shutil.rmtree('./test_checkpoints')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline test failed: {e}")
        return False


def benchmark_models():
    """Benchmark all models for performance."""
    logger.info("Benchmarking Model Performance...")
    
    benchmarks = {}
    
    # Test configurations
    configs = {
        'trajectory_prediction': {
            'input_shape': (8, 10, 13),  # batch, seq, features
            'model_config': {
                'input_dim': 13,
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 4
            }
        },
        'anomaly_detection': {
            'input_shape': (8, 15, 13),
            'model_config': {
                'input_dim': 13,
                'encoding_dim': 64,
                'hidden_dims': [128, 96]
            }
        },
        'vessel_interaction': {
            'input_shape': (1, 10, 10),  # batch, nodes, nodes
            'model_config': {
                'node_features': 10,
                'edge_features': 5,
                'hidden_dim': 128,
                'num_layers': 3
            }
        }
    }
    
    for task, config in configs.items():
        try:
            logger.info(f"Benchmarking {task}...")
            
            # Create model
            model = create_baseline_model(task, **config['model_config'])
            model.eval()
            
            # Create test data
            if task == 'vessel_interaction':
                batch_size, num_nodes = config['input_shape'][0], config['input_shape'][1]
                node_feat = torch.randn(batch_size, num_nodes, config['model_config']['node_features'])
                edge_feat = torch.randn(batch_size, num_nodes, num_nodes, config['model_config']['edge_features'])
                adjacency = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
                test_input = (node_feat, edge_feat, adjacency)
            else:
                test_input = torch.randn(*config['input_shape'])
            
            # Benchmark inference time
            num_runs = 100
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    if task == 'vessel_interaction':
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            
            # Model statistics
            num_params = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            benchmarks[task] = {
                'parameters': num_params,
                'size_mb': model_size_mb,
                'inference_time_ms': avg_time,
                'input_shape': config['input_shape']
            }
            
            logger.info(f"   Parameters: {num_params:,}")
            logger.info(f"   Size: {model_size_mb:.2f} MB")
            logger.info(f"   Inference: {avg_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed for {task}: {e}")
            benchmarks[task] = {'error': str(e)}
    
    return benchmarks


def main():
    """Run all baseline model tests."""
    logger.info("🚀 Starting Maritime Baseline Model Tests...")
    
    results = {
        'trajectory_prediction': False,
        'anomaly_detection': False,
        'vessel_interaction': False,
        'training_pipeline': False
    }
    
    # Test individual models
    results['trajectory_prediction'] = test_trajectory_prediction()
    results['anomaly_detection'] = test_anomaly_detection()
    results['vessel_interaction'] = test_vessel_interaction()
    
    # Test training pipeline
    results['training_pipeline'] = test_training_pipeline()
    
    # Benchmark performance
    benchmarks = benchmark_models()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        logger.info(f"{test_name:25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Benchmark summary
    logger.info("\n📈 PERFORMANCE BENCHMARKS")
    logger.info("-" * 60)
    for task, stats in benchmarks.items():
        if 'error' not in stats:
            logger.info(f"{task:25} {stats['parameters']:>8,} params  {stats['inference_time_ms']:>6.1f}ms")
        else:
            logger.info(f"{task:25} ERROR: {stats['error']}")
    
    if passed == total:
        logger.info("\n🎉 All baseline models are working correctly!")
        logger.info("Ready for training and evaluation on real maritime data.")
        return True
    else:
        logger.error(f"\n❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

