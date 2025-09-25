"""
Classical ML baselines for maritime trajectory prediction.

This module provides traditional machine learning approaches using
scikit-learn for trajectory prediction baselines.
"""

from .base import ClassicalMLBaseline, ClassicalMLConfig
from .features import MaritimeFeatureEngineer
from .rf_model import RFBaseline
from .svr_model import SVRBaseline
from .validation import PurgedTimeSeriesSplit, VesselGroupTimeSeriesSplit

__all__ = [
    "ClassicalMLBaseline",
    "ClassicalMLConfig",
    "SVRBaseline",
    "RFBaseline",
    "PurgedTimeSeriesSplit",
    "VesselGroupTimeSeriesSplit",
    "MaritimeFeatureEngineer",
    "create_classical_baseline",
]


def create_classical_baseline(
    model_type: str = "svr", config: ClassicalMLConfig = None, **kwargs
):
    """
    Factory function to create classical ML baselines.

    Args:
        model_type: "svr" or "rf"
        config: ClassicalMLConfig instance
        **kwargs: Additional model-specific parameters

    Returns:
        Configured baseline model instance
    """
    config = config or ClassicalMLConfig()

    if model_type.lower() == "svr":
        return SVRBaseline(config=config, **kwargs)
    elif model_type.lower() in ["rf", "random_forest"]:
        return RFBaseline(config=config, **kwargs)
    else:
        raise ValueError(
            f"Unknown classical model type: {model_type}. Choose 'svr' or 'rf'"
        )
