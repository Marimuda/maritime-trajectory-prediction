"""
Core source modules for maritime trajectory prediction.
"""

# Define public API - only expose stable submodules
__all__ = [
    "data",
    "models",
    "utils",
    "experiments",
]

# Lazy loading for submodules to avoid heavy imports
import importlib
import sys
from types import ModuleType

_submodules = {
    "data": f"{__name__}.data",
    "models": f"{__name__}.models",
    "utils": f"{__name__}.utils",
    "experiments": f"{__name__}.experiments",
}


def __getattr__(name: str) -> ModuleType:
    """Lazy loading of submodules."""
    if name in _submodules:
        module = importlib.import_module(_submodules[name])
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return available attributes."""
    return sorted(__all__)


# Logging setup
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
