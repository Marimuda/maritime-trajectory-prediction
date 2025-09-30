"""
Operational metrics for maritime trajectory prediction.

This module provides metrics that maritime operators and regulators
care about for real-world deployment assessment.
"""

from .benchmarking import ThroughputBenchmark
from .ops_metrics import (
    CoverageResult,
    OperationalMetrics,
    ThroughputResult,
    WarningEvent,
    WarningTimeStats,
)

__all__ = [
    "WarningEvent",
    "OperationalMetrics",
    "ThroughputResult",
    "CoverageResult",
    "WarningTimeStats",
    "ThroughputBenchmark",
]
