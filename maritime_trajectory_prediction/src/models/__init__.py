"""
Machine learning models for maritime trajectory prediction.
"""

# Only import lightweight transformer blocks directly
from .transformer_blocks import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerBlock,
    CausalSelfAttention,
)

# Define public API
__all__ = [
    "TrAISformer",
    "AISFuser",
    "BaselineModels",
    "ModelFactory",
    "PositionalEncoding",
    "MultiHeadAttention", 
    "TransformerEncoderLayer",
    "TransformerBlock",
    "CausalSelfAttention",
]

# Lazy loading for heavy ML models to avoid importing torch/pytorch-lightning at package import
import importlib
import sys
from types import ModuleType

_lazy_models = {
    "TrAISformer": f"{__name__}.traisformer",
    "AISFuser": f"{__name__}.ais_fuser",
    "BaselineModels": f"{__name__}.baselines",
    "ModelFactory": f"{__name__}.factory",
}

def __getattr__(name: str) -> ModuleType:
    """Lazy loading of ML models to defer heavy imports."""
    if name in _lazy_models:
        module_path = _lazy_models[name]
        module = importlib.import_module(module_path)
        # Get the specific class from the module
        attr = getattr(module, name)
        setattr(sys.modules[__name__], name, attr)
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    """Return available attributes."""
    return sorted(__all__)

# Logging setup
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

