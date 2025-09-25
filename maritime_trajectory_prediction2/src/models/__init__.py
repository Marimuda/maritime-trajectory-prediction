"""
Updated models module __init__.py with baseline models, metrics, and SOTA models.

This module provides comprehensive baseline models for maritime AI tasks
with appropriate metrics and loss functions, plus state-of-the-art models.
"""

# SOTA Models
from .anomaly_transformer import (
    MARITIME_ANOMALY_CONFIG,
    AnomalyTransformer,
    AnomalyTransformerTrainer,
    create_anomaly_transformer,
    create_maritime_anomaly_transformer,
)
from .baseline_models import (
    AnomalyAutoencoder,
    TrajectoryLSTM,
    VesselGCN,
    create_baseline_model,
)
from .metrics import (
    AnomalyDetectionMetrics,
    MaritimeLossFunctions,
    TrajectoryPredictionMetrics,
    VesselInteractionMetrics,
    create_loss_function,
    create_metrics,
)
from .motion_transformer import (
    MARITIME_MTR_CONFIG,
    MotionTransformer,
    MotionTransformerTrainer,
    create_maritime_motion_transformer,
    create_motion_transformer,
)
from .train_baselines import (
    AnomalyTrainer,
    BaselineTrainer,
    InteractionTrainer,
    TrajectoryTrainer,
    create_trainer,
    train_baseline_model,
)

# Version information
__version__ = "1.1.0"

# Public API
__all__ = [
    # Baseline Models
    "TrajectoryLSTM",
    "AnomalyAutoencoder",
    "VesselGCN",
    "create_baseline_model",
    # SOTA Models
    "AnomalyTransformer",
    "AnomalyTransformerTrainer",
    "create_anomaly_transformer",
    "create_maritime_anomaly_transformer",
    "MARITIME_ANOMALY_CONFIG",
    "MotionTransformer",
    "MotionTransformerTrainer",
    "create_motion_transformer",
    "create_maritime_motion_transformer",
    "MARITIME_MTR_CONFIG",
    # Metrics and Loss Functions
    "TrajectoryPredictionMetrics",
    "AnomalyDetectionMetrics",
    "VesselInteractionMetrics",
    "MaritimeLossFunctions",
    "create_metrics",
    "create_loss_function",
    # Training Infrastructure
    "BaselineTrainer",
    "TrajectoryTrainer",
    "AnomalyTrainer",
    "InteractionTrainer",
    "create_trainer",
    "train_baseline_model",
]

# Model registry for easy access
BASELINE_MODELS = {
    "trajectory_prediction": TrajectoryLSTM,
    "anomaly_detection": AnomalyAutoencoder,
    "vessel_interaction": VesselGCN,
}

# SOTA Models registry
SOTA_MODELS = {
    "anomaly_transformer": AnomalyTransformer,
    "motion_transformer": MotionTransformer,
}

# Combined model registry
ALL_MODELS = {**BASELINE_MODELS, **SOTA_MODELS}

# Metrics registry
METRICS_REGISTRY = {
    "trajectory_prediction": TrajectoryPredictionMetrics,
    "anomaly_detection": AnomalyDetectionMetrics,
    "vessel_interaction": VesselInteractionMetrics,
}

# Default configurations for each model
DEFAULT_CONFIGS = {
    "trajectory_prediction": {
        "input_dim": 4,
        "hidden_dim": 128,
        "num_layers": 2,
        "output_dim": 4,
        "bidirectional": True,
        "dropout": 0.2,
    },
    "anomaly_detection": {
        "input_dim": 4,
        "encoding_dim": 64,
        "hidden_dims": [128, 96],
        "activation": "relu",
        "dropout": 0.2,
    },
    "vessel_interaction": {
        "node_features": 10,
        "edge_features": 5,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "aggregation": "mean",
    },
    "anomaly_transformer": {
        "input_dim": 4,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 200,
    },
    "motion_transformer": {
        "input_dim": 4,
        "d_model": 256,
        "n_queries": 6,
        "encoder_layers": 4,
        "decoder_layers": 6,
        "n_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "prediction_horizon": 30,
        "output_dim": 4,
    },
}


def create_model(model_type: str, task: str = None, **kwargs):
    """
    Unified model creation function for both baseline and SOTA models.

    Args:
        model_type: Type of model ('baseline' or specific SOTA model name)
        task: Task name (required for baseline models)
        **kwargs: Model configuration parameters

    Returns:
        Model instance
    """
    if model_type == "baseline":
        if task is None:
            raise ValueError("Task must be specified for baseline models")

        # Get default config and update with kwargs
        config = DEFAULT_CONFIGS[task].copy()
        config.update(kwargs)
        return create_baseline_model(task, **config)

    elif model_type == "anomaly_transformer":
        config = DEFAULT_CONFIGS["anomaly_transformer"].copy()
        config.update(kwargs)
        return create_anomaly_transformer(**config)
    elif model_type == "motion_transformer":
        config = DEFAULT_CONFIGS["motion_transformer"].copy()
        config.update(kwargs)
        return create_motion_transformer(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model_type: str, task: str = None) -> dict:
    """
    Get comprehensive information about a model.

    Args:
        model_type: Model type ('baseline' or SOTA model name)
        task: Task name (for baseline models)

    Returns:
        Dictionary with model information
    """
    if model_type == "baseline":
        if task is None:
            raise ValueError("Task must be specified for baseline models")
        if task not in BASELINE_MODELS:
            raise ValueError(
                f"Unknown task: {task}. Available: {list(BASELINE_MODELS.keys())}"
            )

        model_class = BASELINE_MODELS[task]
        config = DEFAULT_CONFIGS[task]

        # Create model instance to get parameter count
        model = create_baseline_model(task, **config)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        return {
            "model_type": "baseline",
            "task": task,
            "model_class": model_class.__name__,
            "parameters": param_count,
            "size_mb": round(model_size_mb, 2),
            "default_config": config,
            "description": model_class.__doc__.split("\n")[1].strip()
            if model_class.__doc__
            else "No description available",
        }

    elif model_type == "anomaly_transformer":
        config = DEFAULT_CONFIGS["anomaly_transformer"]
        model = create_anomaly_transformer(**config)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        return {
            "model_type": "sota",
            "model_name": "anomaly_transformer",
            "model_class": "AnomalyTransformer",
            "parameters": param_count,
            "size_mb": round(model_size_mb, 2),
            "default_config": config,
            "description": "State-of-the-art anomaly detection with anomaly-attention mechanism",
        }

    elif model_type == "motion_transformer":
        config = DEFAULT_CONFIGS["motion_transformer"]
        model = create_motion_transformer(**config)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        return {
            "model_type": "sota",
            "model_name": "motion_transformer",
            "model_class": "MotionTransformer",
            "parameters": param_count,
            "size_mb": round(model_size_mb, 2),
            "default_config": config,
            "description": "State-of-the-art trajectory prediction with multimodal outputs",
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def list_available_models() -> dict:
    """
    List all available models with their information.

    Returns:
        Dictionary mapping model identifiers to model information
    """
    models = {}

    # Baseline models
    for task in BASELINE_MODELS:
        models[f"baseline_{task}"] = get_model_info("baseline", task)

    # SOTA models
    for model_name in SOTA_MODELS:
        models[model_name] = get_model_info(model_name)

    return models


# Lazy loading for heavy dependencies
def __getattr__(name: str):
    """Lazy loading for optional components."""
    if name == "lightning_models":
        try:
            from . import lightning_models

            return lightning_models
        except ImportError:
            raise ImportError(
                "PyTorch Lightning not available. Install with: pip install pytorch-lightning"
            )

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Module-level convenience functions
def quick_train(task: str, data_path: str, epochs: int = 50, **kwargs):
    """
    Quick training function for baseline models.

    Args:
        task: Task name
        data_path: Path to training data
        epochs: Number of training epochs
        **kwargs: Additional configuration parameters

    Returns:
        Training results
    """
    model_config = DEFAULT_CONFIGS[task].copy()
    model_config.update(kwargs.get("model_config", {}))

    training_config = {
        "data": {"batch_size": 32, "val_split": 0.2, "test_split": 0.1},
        "trainer": {"learning_rate": 0.001, "weight_decay": 1e-5},
        "training": {"num_epochs": epochs, "patience": 10, "save_best": True},
    }
    training_config.update(kwargs.get("training_config", {}))

    return train_baseline_model(
        task=task,
        data_path=data_path,
        model_config=model_config,
        training_config=training_config,
        output_dir=kwargs.get("output_dir", "./results"),
    )


def quick_evaluate(task: str, model_path: str, data_path: str, **kwargs):
    """
    Quick evaluation function for trained baseline models.

    Args:
        task: Task name
        model_path: Path to trained model
        data_path: Path to evaluation data
        **kwargs: Additional parameters

    Returns:
        Evaluation metrics
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Load model
    model = create_baseline_model(task, **DEFAULT_CONFIGS[task])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Create trainer for evaluation
    trainer = create_trainer(task, model)

    # Load data (simplified for quick evaluation)
    # In practice, you'd use the full data pipeline
    test_data = torch.randn(100, 10, 4)  # Dummy data
    test_targets = torch.randn(100, 10, 4)  # Dummy targets

    test_dataset = TensorDataset(test_data, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate
    metrics = trainer.validate_epoch(test_loader)

    return metrics


if __name__ == "__main__":
    # Print available models
    print("Available Maritime Baseline Models:")
    print("=" * 50)

    for task, info in list_available_models().items():
        print(f"\n{task.upper().replace('_', ' ')}")
        print(f"  Model: {info['model_class']}")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Size: {info['size_mb']} MB")
        print(f"  Description: {info['description']}")
