"""
Baseline models for maritime trajectory prediction.

This module provides baseline implementations for various maritime AI tasks:
- Trajectory prediction using LSTM
- Anomaly detection using autoencoder
- Vessel interaction modeling using GCN
- Physics-based Kalman filter baselines
"""

from .anomaly_autoencoder import AnomalyAutoencoderLightning as AnomalyAutoencoder
from .trajectory_lstm import TrajectoryLSTMLightning as TrajectoryLSTM
from .vessel_gcn import VesselGCNLightning as VesselGCN
from .kalman import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    MaritimeIMMFilter,
    NearlyConstantAccelModel,
    create_imm_lightning,
    create_cv_lightning,
    create_ct_lightning,
    create_nca_lightning,
)

__all__ = [
    "AnomalyAutoencoder",
    "TrajectoryLSTM",
    "VesselGCN",
    "ConstantVelocityModel",
    "CoordinatedTurnModel",
    "MaritimeIMMFilter",
    "NearlyConstantAccelModel",
    "create_imm_lightning",
    "create_cv_lightning",
    "create_ct_lightning",
    "create_nca_lightning",
]

def create_baseline_model(model_type: str, **kwargs):
    """Create a baseline model of the specified type."""
    if model_type == "trajectory_prediction":
        return TrajectoryLSTM(**kwargs)
    elif model_type == "anomaly_detection":
        return AnomalyAutoencoder(**kwargs)
    elif model_type == "vessel_interaction":
        return VesselGCN(**kwargs)
    elif model_type == "kalman_cv":
        return ConstantVelocityModel(**kwargs)
    elif model_type == "kalman_ct":
        return CoordinatedTurnModel(**kwargs)
    elif model_type == "kalman_nca":
        return NearlyConstantAccelModel(**kwargs)
    elif model_type == "kalman_imm":
        return MaritimeIMMFilter(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")