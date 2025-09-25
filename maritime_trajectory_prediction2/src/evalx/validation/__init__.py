"""Cross-validation and model comparison protocols for maritime data."""

from .comparisons import ComparisonResult, ModelComparison
from .protocols import GroupKFold, TimeSeriesSplit, maritime_cv_split

__all__ = [
    "TimeSeriesSplit",
    "GroupKFold",
    "maritime_cv_split",
    "ModelComparison",
    "ComparisonResult",
]
