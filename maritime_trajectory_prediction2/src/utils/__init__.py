"""
Fixed utility functions and helpers.
"""

from .maritime_utils import MaritimeUtils

try:
    from .ais_parser import AISParser
except ImportError:
    AISParser = None

try:
    from .metrics import TrajectoryMetrics
except ImportError:
    TrajectoryMetrics = None

try:
    from .visualization import TrajectoryVisualizer
except ImportError:
    TrajectoryVisualizer = None

__all__ = ["MaritimeUtils"]

# Add optional components if available
if AISParser:
    __all__.append("AISParser")

if TrajectoryMetrics:
    __all__.append("TrajectoryMetrics")

if TrajectoryVisualizer:
    __all__.append("TrajectoryVisualizer")
