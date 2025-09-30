"""
Error Analysis Framework for Maritime Trajectory Prediction.

This module provides comprehensive error analysis tools including:
- Performance slicing by maritime conditions
- Failure mining and clustering analysis
- Horizon curve analysis with confidence intervals
- Automated case study generation

Key Components:
- ErrorSlicer: Slice performance by vessel type, traffic density, etc.
- FailureMiner: Identify and cluster worst-performing cases
- HorizonAnalyzer: Analyze error progression over prediction horizons
- CaseStudyGenerator: Generate actionable failure documentation

Example Usage:
```python
from evalx.error_analysis import ErrorSlicer, FailureMiner, HorizonAnalyzer

# Performance slicing
slicer = ErrorSlicer()
sliced_results = slicer.slice_errors(predictions, targets, metadata)

# Failure mining
miner = FailureMiner()
failure_clusters = miner.mine_failures(errors, features, metadata)

# Horizon analysis
analyzer = HorizonAnalyzer()
horizon_curves = analyzer.analyze_horizon_errors(predictions, targets)
```
"""

from .case_studies import CaseStudy, CaseStudyGenerator
from .horizon import HorizonAnalyzer, HorizonCurve
from .mining import FailureCase, FailureCluster, FailureMiner
from .slicers import ErrorSlicer, SliceConfig, SliceResult

__all__ = [
    "ErrorSlicer",
    "SliceConfig",
    "SliceResult",
    "FailureMiner",
    "FailureCluster",
    "FailureCase",
    "HorizonAnalyzer",
    "HorizonCurve",
    "CaseStudyGenerator",
    "CaseStudy",
]
