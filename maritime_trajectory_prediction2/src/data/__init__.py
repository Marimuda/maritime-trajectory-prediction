"""
Updated data module __init__.py with pipeline components.
"""

from .dataset_builders import (
    AnomalyDetectionBuilder,
    CollisionAvoidanceBuilder,
    GraphNetworkBuilder,
    TrajectoryPredictionBuilder,
)
from .maritime_message_processor import AISProcessor
from .multi_task_processor import AISMultiTaskProcessor, MLTask
from .pipeline import (
    BaseDatasetBuilder,
    DataPipeline,
    DatasetConfig,
    DatasetFormat,
    DatasetMetadata,
)
from .validation import DatasetExporter, DataValidator, QualityChecker, ValidationResult

__all__ = [
    # Core processors
    "AISProcessor",
    "AISMultiTaskProcessor",
    "MLTask",
    # Pipeline components
    "DataPipeline",
    "BaseDatasetBuilder",
    "DatasetConfig",
    "DatasetMetadata",
    "DatasetFormat",
    # Task-specific builders
    "TrajectoryPredictionBuilder",
    "AnomalyDetectionBuilder",
    "GraphNetworkBuilder",
    "CollisionAvoidanceBuilder",
    # Validation and quality
    "DataValidator",
    "QualityChecker",
    "DatasetExporter",
    "ValidationResult",
]


# Lazy loading for heavy components
def __getattr__(name):
    if name == "xarray_processor":
        from .xarray_processor import AISDataProcessor

        return AISDataProcessor
    elif name == "lightning_datamodule":
        from .lightning_datamodule import AISLightningDataModule

        return AISLightningDataModule
    elif name == "preprocess":
        from .preprocess import preprocess_ais_logs

        return preprocess_ais_logs
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
