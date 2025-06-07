"""
Data processing and loading modules for maritime trajectory prediction.
"""

# Core data processing - always available
from .ais_processor import AISProcessor

# Define public API
__all__ = [
    "AISProcessor",
    "AISDataModule", 
    "GraphProcessor",
]

# Lazy loading for potentially heavy modules
import importlib
import sys
from types import ModuleType

_lazy_modules = {
    "AISDataModule": f"{__name__}.datamodule",
    "GraphProcessor": f"{__name__}.graph_processor",
}

def __getattr__(name: str) -> ModuleType:
    """Lazy loading of data modules."""
    if name in _lazy_modules:
        module_path = _lazy_modules[name]
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

