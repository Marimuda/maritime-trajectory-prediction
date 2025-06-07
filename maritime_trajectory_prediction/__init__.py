"""
Maritime Trajectory Prediction Package

A comprehensive package for predicting maritime vessel trajectories using 
transformer-based models and AIS (Automatic Identification System) data.
"""

# Version management following PEP 562
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.1.0+local"

# Metadata
__author__ = "Jákup Svøðstein"
__email__ = "jakupsv@setur.fo"

# Selective re-exports for flat API - only core, stable components
from .src.data.ais_processor import AISProcessor
from .src.data.datamodule import AISDataModule

# Define public API
__all__ = [
    "AISProcessor",
    "AISDataModule",
    "__version__",
    "__author__",
    "__email__",
]

# Lazy loading for heavy submodules (models, visualization)
import importlib
import sys
from types import ModuleType

_lazy_submodules = {
    "models": f"{__name__}.src.models",
    "utils": f"{__name__}.src.utils", 
    "experiments": f"{__name__}.src.experiments",
}

def __getattr__(name: str) -> ModuleType:
    """Lazy loading of heavy submodules."""
    if name in _lazy_submodules:
        module = importlib.import_module(_lazy_submodules[name])
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    """Return available attributes including lazy-loaded modules."""
    return sorted(__all__ + list(_lazy_submodules))

# Logging setup - NullHandler to prevent warnings
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

