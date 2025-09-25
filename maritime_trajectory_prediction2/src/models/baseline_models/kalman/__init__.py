"""
Kalman Filter baseline models for maritime trajectory prediction.

This package provides physics-based Kalman filtering approaches for vessel
trajectory prediction, including:
- Multiple motion models (Constant Velocity, Coordinated Turn, Nearly Constant Acceleration)
- Interactive Multiple Model (IMM) framework for automatic model switching
- Maritime-specific adaptations and constraints
- Hyperparameter tuning system
"""

from .imm import MaritimeIMMFilter
from .models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from .protocols import BaselineResult, TrajectoryBaseline

__all__ = [
    "TrajectoryBaseline",
    "BaselineResult",
    "ConstantVelocityModel",
    "CoordinatedTurnModel",
    "NearlyConstantAccelModel",
    "MaritimeIMMFilter",
]
