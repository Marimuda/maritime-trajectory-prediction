"""Configuration module for the maritime trajectory prediction system."""

from .dataclasses import (
    CallbacksConfig,
    DataConfig,
    ExperimentConfig,
    LoggerConfig,
    ModeConfig,
    ModelConfig,
    PredictConfig,
    RootConfig,
    TrainerConfig,
)
from .store import register_configs

__all__ = [
    "RootConfig",
    "ModeConfig",
    "DataConfig",
    "ModelConfig",
    "TrainerConfig",
    "LoggerConfig",
    "CallbacksConfig",
    "ExperimentConfig",
    "PredictConfig",
    "register_configs",
]
