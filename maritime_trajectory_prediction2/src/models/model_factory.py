"""
Unified factory for creating and loading models in the maritime trajectory prediction system.
Supports Hydra-based instantiation and manual instantiation by type.
"""

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from .ais_fuser import AISFuserLightning
from .benchmark_models import LSTMModel, XGBoostModel
from .motion_transformer import MotionTransformerLightning

# Import your model classes
from .baseline_models.kalman import (
    MaritimeIMMFilter,
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel
)
from .baseline_models.kalman.lightning_adapter import (
    KalmanBaselineLightning,
    create_imm_lightning,
    create_cv_lightning,
    create_ct_lightning,
    create_nca_lightning
)
from .traisformer import TrAISformer

logger = logging.getLogger(__name__)


def create_model(config: DictConfig | dict[str, Any]) -> pl.LightningModule:
    """
    Instantiate a model from a Hydra DictConfig or a manual config dict.

    Args:
        config: Hydra DictConfig with _target_ field or dict with 'type' key

    Returns:
        Instantiated LightningModule
    """
    # Hydra instantiation path
    if isinstance(config, DictConfig) and config.get("_target_"):
        logger.info(f"Instantiating model via Hydra: {config._target_}")
        return instantiate(config)

    # Manual instantiation
    # Allow plain dict or OmegaConf object without _target_
    model_type = (
        config.get("type")
        if isinstance(config, dict)
        else getattr(config, "type", None)
    )
    if model_type is None:
        raise ValueError("Config must specify 'type' for manual instantiation")
    model_type = model_type.lower()

    if model_type == "traisformer":
        return TrAISformer(config)
    elif model_type == "ais_fuser":
        return AISFuserLightning(config)
    elif model_type == "lstm":
        return LSTMModel(config)
    elif model_type == "xgboost":
        return XGBoostModel(config)
    elif model_type == "motion_transformer":
        return MotionTransformerLightning(config)
    elif model_type in ["kalman", "imm", "kalman_imm", "maritime_imm"]:
        return create_imm_lightning(config)
    elif model_type in ["kalman_cv", "constant_velocity"]:
        return create_cv_lightning(config)
    elif model_type in ["kalman_ct", "coordinated_turn"]:
        return create_ct_lightning(config)
    elif model_type in ["kalman_nca", "nearly_constant_accel"]:
        return create_nca_lightning(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(
    checkpoint_path: str | Path, config: DictConfig | dict[str, Any] | None = None
) -> pl.LightningModule:
    """
    Load a model from checkpoint, optionally using config for instantiation.

    Args:
        checkpoint_path: Path to checkpoint (.ckpt or .pth)
        config: Optional config for manual instantiation

    Returns:
        Loaded model with weights
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info(f"Loading checkpoint: {path}")

    ckpt = torch.load(path, map_location="cpu")
    # Manual instantiation then load
    if config is not None:
        model = create_model(config)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        return model

    # Delegate to Lightning
    return pl.LightningModule.load_from_checkpoint(str(path))


def get_model_class(model_type: str) -> Any:
    """
    Retrieve the class for a given model_type string.

    Args:
        model_type: Type identifier

    Returns:
        Model class
    """
    mt = model_type.lower()
    if mt == "traisformer":
        return TrAISformer
    if mt == "ais_fuser":
        return AISFuserLightning
    if mt == "lstm":
        return LSTMModel
    if mt == "xgboost":
        return XGBoostModel
    if mt == "motion_transformer":
        return MotionTransformerLightning
    if mt in ["kalman", "imm", "kalman_imm", "maritime_imm"]:
        return KalmanBaselineLightning
    if mt in ["kalman_cv", "constant_velocity"]:
        return KalmanBaselineLightning
    if mt in ["kalman_ct", "coordinated_turn"]:
        return KalmanBaselineLightning
    if mt in ["kalman_nca", "nearly_constant_accel"]:
        return KalmanBaselineLightning
    raise ValueError(f"Unknown model type: {model_type}")
