"""Training module for the maritime trajectory prediction system."""

from .trainer import LightningTrainerWrapper, create_trainer

__all__ = ["create_trainer", "LightningTrainerWrapper"]
