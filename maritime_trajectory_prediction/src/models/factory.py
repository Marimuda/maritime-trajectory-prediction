"""
Unified model factory for the maritime trajectory prediction system.

Consolidates all model instantiation logic following the CLAUDE blueprint.
Provides a single entry point for creating any model described in configurations.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

# Import config types for type hints
from ..config.dataclasses import ModelConfig, ModelType, TaskType

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Unified model factory for creating all supported models.

    Supports both baseline and SOTA models with consistent interface.
    """

    def __init__(self):
        """Initialize the model factory."""
        self._model_registry = {}
        self._register_models()

    def _register_models(self):
        """Register all available models in the factory."""

        # Trajectory prediction models
        self._model_registry.update(
            {
                ModelType.TRAISFORMER: {
                    "class": "src.models.traisformer.TrAISformer",
                    "task": TaskType.TRAJECTORY_PREDICTION,
                    "category": "sota",
                },
                ModelType.AIS_FUSER: {
                    "class": "src.models.ais_fuser.AISFuserLightning",
                    "task": TaskType.TRAJECTORY_PREDICTION,
                    "category": "sota",
                },
                ModelType.LSTM: {
                    "class": "src.models.baseline_models.LSTMPredictor",
                    "task": TaskType.TRAJECTORY_PREDICTION,
                    "category": "baseline",
                },
                ModelType.XGBOOST: {
                    "class": "src.models.baseline_models.XGBoostPredictor",
                    "task": TaskType.TRAJECTORY_PREDICTION,
                    "category": "baseline",
                },
                ModelType.MOTION_TRANSFORMER: {
                    "class": "src.models.motion_transformer.MaritimeMotionTransformer",
                    "task": TaskType.TRAJECTORY_PREDICTION,
                    "category": "sota",
                },
                ModelType.ANOMALY_TRANSFORMER: {
                    "class": "src.models.anomaly_transformer.MaritimeAnomalyTransformer",
                    "task": TaskType.ANOMALY_DETECTION,
                    "category": "sota",
                },
            }
        )

    def create_model(self, config: DictConfig | ModelConfig) -> Any:
        """
        Create a model instance from configuration.

        Args:
            config: Model configuration (DictConfig or ModelConfig)

        Returns:
            Instantiated model

        Raises:
            ValueError: If model type is not supported
        """
        # Convert to ModelConfig if needed
        if isinstance(config, DictConfig):
            model_config = ModelConfig(**config)
        else:
            model_config = config

        model_type = model_config.type

        if model_type not in self._model_registry:
            available_types = list(self._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {available_types}"
            )

        model_info = self._model_registry[model_type]

        # Validate task compatibility
        if model_config.task != model_info["task"]:
            logger.warning(
                f"Model {model_type} is designed for {model_info['task']} but configured for {model_config.task}"
            )

        # Dynamic import and instantiation
        class_path = model_info["class"]
        model_class = self._import_class(class_path)

        logger.info(f"Creating {model_info['category']} model: {model_type}")

        # Create model with appropriate parameters
        try:
            if model_info["category"] == "baseline":
                return self._create_baseline_model(model_class, model_config)
            else:
                return self._create_sota_model(model_class, model_config)
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise

    def _create_baseline_model(self, model_class: type, config: ModelConfig) -> Any:
        """Create baseline model with standard parameters."""

        # Standard baseline parameters
        baseline_params = {
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "task": config.task.value,
        }

        # Add model-specific custom parameters
        if config.custom_params:
            baseline_params.update(config.custom_params)

        return model_class(**baseline_params)

    def _create_sota_model(self, model_class: type, config: ModelConfig) -> Any:
        """Create SOTA model with advanced parameters."""

        # Standard SOTA parameters
        sota_params = {
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "prediction_horizon": config.prediction_horizon,
        }

        # Add multimodal parameters for trajectory prediction
        if config.task == TaskType.TRAJECTORY_PREDICTION:
            sota_params["num_modes"] = config.num_modes

        # Add model-specific custom parameters
        if config.custom_params:
            sota_params.update(config.custom_params)

        return model_class(**sota_params)

    def _import_class(self, class_path: str) -> type:
        """Dynamically import a class from its path."""
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {class_path}: {e}")
            raise ImportError(f"Cannot import {class_path}: {e}")

    def load_model(
        self,
        checkpoint_path: str | Path,
        config: DictConfig | ModelConfig | None = None,
    ) -> Any:
        """
        Load a model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            config: Optional configuration for model creation

        Returns:
            Loaded model instance
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if config is not None:
            # Create model from config then load weights
            model = self.create_model(config)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # Load additional metadata if available
            if hasattr(model, "load_metadata") and "metadata" in checkpoint:
                model.load_metadata(checkpoint["metadata"])

            return model
        else:
            # Try to load using PyTorch Lightning if available
            try:
                import pytorch_lightning as pl

                return pl.LightningModule.load_from_checkpoint(checkpoint_path)
            except ImportError:
                logger.error("PyTorch Lightning not available and no config provided")
                raise ValueError(
                    "Must provide config when PyTorch Lightning is not available"
                )

    def get_model_class(self, model_type: ModelType) -> type:
        """
        Get model class by type.

        Args:
            model_type: Type of model

        Returns:
            Model class
        """
        if model_type not in self._model_registry:
            available_types = list(self._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {available_types}"
            )

        class_path = self._model_registry[model_type]["class"]
        return self._import_class(class_path)

    def get_model_info(self, model_type: ModelType) -> dict[str, Any]:
        """
        Get information about a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary with model information
        """
        if model_type not in self._model_registry:
            available_types = list(self._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {available_types}"
            )

        return self._model_registry[model_type].copy()

    def list_models(
        self, task: TaskType | None = None, category: str | None = None
    ) -> dict[ModelType, dict[str, Any]]:
        """
        List available models with optional filtering.

        Args:
            task: Filter by task type
            category: Filter by category ('baseline' or 'sota')

        Returns:
            Dictionary of available models
        """
        filtered_models = {}

        for model_type, model_info in self._model_registry.items():
            # Apply filters
            if task is not None and model_info["task"] != task:
                continue
            if category is not None and model_info["category"] != category:
                continue

            filtered_models[model_type] = model_info.copy()

        return filtered_models

    def create_baseline_model(self, task: str, **kwargs) -> Any:
        """
        Legacy function for creating baseline models.

        Args:
            task: Task type as string
            **kwargs: Model parameters

        Returns:
            Baseline model instance
        """
        logger.warning(
            "create_baseline_model is deprecated. Use create_model with ModelConfig instead."
        )

        # Map task string to enum
        task_mapping = {
            "trajectory_prediction": TaskType.TRAJECTORY_PREDICTION,
            "anomaly_detection": TaskType.ANOMALY_DETECTION,
            "vessel_interaction": TaskType.VESSEL_INTERACTION,
        }

        task_enum = task_mapping.get(task)
        if task_enum is None:
            raise ValueError(f"Unknown task: {task}")

        # Create a temporary config
        config = ModelConfig(
            type=ModelType.LSTM
            if task_enum == TaskType.TRAJECTORY_PREDICTION
            else ModelType.ANOMALY_TRANSFORMER,
            task=task_enum,
            custom_params=kwargs,
        )

        return self.create_model(config)


# Global factory instance
_model_factory = ModelFactory()


# Convenience functions for backward compatibility
def create_model(config: DictConfig | ModelConfig) -> Any:
    """Create a model instance from configuration."""
    return _model_factory.create_model(config)


def load_model(
    checkpoint_path: str | Path, config: DictConfig | ModelConfig | None = None
) -> Any:
    """Load a model from checkpoint."""
    return _model_factory.load_model(checkpoint_path, config)


def get_model_class(model_type: ModelType) -> type:
    """Get model class by type."""
    return _model_factory.get_model_class(model_type)


def get_model_info(model_type: ModelType) -> dict[str, Any]:
    """Get information about a model type."""
    return _model_factory.get_model_info(model_type)


def list_models(
    task: TaskType | None = None, category: str | None = None
) -> dict[ModelType, dict[str, Any]]:
    """List available models with optional filtering."""
    return _model_factory.list_models(task, category)


def create_baseline_model(task: str, **kwargs) -> Any:
    """Legacy function for creating baseline models."""
    return _model_factory.create_baseline_model(task, **kwargs)
