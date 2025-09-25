"""
Experiment management and training modules.
"""

# Define public API
__all__ = [
    "TrainingManager",
    "EvaluationManager",
    "SweepRunner",
]

# Lazy loading for all experiment modules (heavy ML dependencies)
import importlib
import sys
from types import ModuleType

_lazy_experiments = {
    "TrainingManager": f"{__name__}.train",
    "EvaluationManager": f"{__name__}.evaluation",
    "SweepRunner": f"{__name__}.sweep_runner",
}


def __getattr__(name: str) -> ModuleType:
    """Lazy loading of experiment modules to defer heavy ML imports."""
    if name in _lazy_experiments:
        module_path = _lazy_experiments[name]
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
