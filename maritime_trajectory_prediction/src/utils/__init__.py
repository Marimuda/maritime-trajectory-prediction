"""
Utility functions and helper modules for maritime trajectory prediction.
"""

# Core utilities - always available
from .ais_parser import AISParser
from .maritime_utils import MaritimeUtils

# Define public API
__all__ = [
    "AISParser",
    "MaritimeUtils",
    "TrajectoryMetrics", 
    "TrajectoryVisualizer",
]

# Lazy loading for visualization and metrics (may have heavy dependencies)
import importlib
import sys
from types import ModuleType

_lazy_utils = {
    "TrajectoryMetrics": f"{__name__}.metrics",
    "TrajectoryVisualizer": f"{__name__}.visualization",
}

def __getattr__(name: str) -> ModuleType:
    """Lazy loading of utility modules with heavy dependencies."""
    if name in _lazy_utils:
        module_path = _lazy_utils[name]
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

