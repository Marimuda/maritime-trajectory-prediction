# M1.4 Baseline Suite Integration & Validation Specification

## Executive Summary

This document specifies M1.4: Baseline Suite Integration & Validation, which unifies all baseline models (M1.1 Statistical, M1.2 Kalman, M1.3 Classical ML) into a comprehensive evaluation framework with statistical validation against neural models.

## Integration Architecture Analysis

### Current Baseline Landscape
```
src/models/baseline_models/
├── __init__.py                    # Factory pattern with 7 model types ✓
├── kalman/                        # M1.2: Physics-based (4 models) ✓
│   ├── protocols.py              # TrajectoryBaseline protocol ✓
│   ├── models.py                 # CV, CT, NCA implementations
│   └── imm.py                    # IMM framework
├── classical/                    # M1.3: ML-based (2 models) ✓
│   ├── base.py                   # ClassicalMLBaseline
│   ├── svr_model.py             # Support Vector Regression
│   └── rf_model.py              # Random Forest
└── [Neural baselines]            # TrajectoryLSTM, VesselGCN, AnomalyAutoencoder
```

### Existing Evaluation Infrastructure
```
src/evalx/                        # M1.1: Statistical Framework ✓
├── validation/
│   ├── protocols.py             # Time-aware CV with maritime considerations
│   └── comparisons.py           # ModelComparison with bootstrap CI + statistical tests
├── stats/
│   ├── bootstrap.py             # BootstrapCI with 9999 resamples
│   └── tests.py                 # Paired t-test, Wilcoxon, Cliff's delta
└── metrics/
    └── enhanced_metrics.py      # Maritime-specific metrics
```

## Library Dependencies and Endpoints

### Primary Libraries (Mature Solutions)

#### 1. Performance Profiling
```python
# Memory profiling
import psutil
import tracemalloc
from memory_profiler import profile

# CPU profiling
import cProfile
import line_profiler
import time

# GPU profiling (if available)
import gpustat  # pip install gpustat
import pynvml   # pip install nvidia-ml-py
```

#### 2. Parallel Processing
```python
# Joblib for parallel evaluation
from joblib import Parallel, delayed, Memory

# Multiprocessing for model isolation
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
```

#### 3. Data Management
```python
# Efficient data handling
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

# Progress tracking
from tqdm import tqdm
```

#### 4. Visualization and Reporting
```python
# Statistical visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Performance dashboards
import plotly.graph_objects as go
import plotly.express as px

# Report generation
from jinja2 import Template
```

## M1.4 Implementation Specification

### 1. Unified Baseline Registry

**File**: `src/models/baseline_models/registry.py`

```python
"""
Unified registry for all baseline models with discovery and management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import importlib
import inspect
import json

from .kalman.protocols import TrajectoryBaseline, BaselineResult


@dataclass
class BaselineInfo:
    """Information about a registered baseline model."""

    name: str
    model_class: Type[TrajectoryBaseline]
    category: str  # "physics", "classical_ml", "neural", "ensemble"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    paper_reference: Optional[str] = None
    computational_complexity: Optional[str] = None  # O(n), O(n²), etc.
    supports_uncertainty: bool = False
    supports_multi_horizon: bool = False
    requires_gpu: bool = False


class BaselineRegistry:
    """
    Central registry for all baseline models with automatic discovery.

    Provides unified interface for baseline model management, discovery,
    and instantiation across different categories.
    """

    def __init__(self):
        self._baselines: Dict[str, BaselineInfo] = {}
        self._categories: Dict[str, List[str]] = {}
        self._auto_discover()

    def register(
        self,
        name: str,
        model_class: Type[TrajectoryBaseline],
        category: str,
        description: str,
        **kwargs
    ) -> None:
        """Register a baseline model."""

        # Validate model class implements protocol
        if not self._implements_protocol(model_class):
            raise ValueError(f"Model {model_class} must implement TrajectoryBaseline protocol")

        baseline_info = BaselineInfo(
            name=name,
            model_class=model_class,
            category=category,
            description=description,
            **kwargs
        )

        self._baselines[name] = baseline_info

        # Update categories
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def _auto_discover(self):
        """Automatically discover and register baseline models."""

        # Register Kalman models (M1.2)
        from .kalman import (
            ConstantVelocityModel,
            CoordinatedTurnModel,
            MaritimeIMMFilter,
            NearlyConstantAccelModel
        )

        self.register(
            name="kalman_cv",
            model_class=ConstantVelocityModel,
            category="physics",
            description="Constant Velocity Kalman filter for straight-line motion",
            computational_complexity="O(n)",
            supports_uncertainty=True,
            supports_multi_horizon=True,
            paper_reference="Kalman, R.E. (1960). A new approach to linear filtering"
        )

        self.register(
            name="kalman_ct",
            model_class=CoordinatedTurnModel,
            category="physics",
            description="Coordinated Turn model for maneuvering vessels",
            computational_complexity="O(n)",
            supports_uncertainty=True,
            supports_multi_horizon=True
        )

        self.register(
            name="kalman_nca",
            model_class=NearlyConstantAccelModel,
            category="physics",
            description="Nearly Constant Acceleration model",
            computational_complexity="O(n)",
            supports_uncertainty=True,
            supports_multi_horizon=True
        )

        self.register(
            name="kalman_imm",
            model_class=MaritimeIMMFilter,
            category="physics",
            description="Interactive Multiple Model with CV/CT/NCA switching",
            computational_complexity="O(n*m²)",  # m = number of models
            supports_uncertainty=True,
            supports_multi_horizon=True,
            paper_reference="Bar-Shalom, Y. (2001). Estimation with Applications"
        )

        # Register Classical ML models (M1.3)
        from .classical import SVRBaseline, RFBaseline

        self.register(
            name="svr",
            model_class=SVRBaseline,
            category="classical_ml",
            description="Support Vector Regression with RBF kernel",
            computational_complexity="O(n²) to O(n³)",
            supports_uncertainty=True,
            supports_multi_horizon=True,
            paper_reference="Vapnik, V. (1995). The Nature of Statistical Learning Theory"
        )

        self.register(
            name="random_forest",
            model_class=RFBaseline,
            category="classical_ml",
            description="Random Forest with feature importance analysis",
            computational_complexity="O(n*log(n)*m*p)",  # m=trees, p=features
            supports_uncertainty=True,
            supports_multi_horizon=True,
            paper_reference="Breiman, L. (2001). Random Forests"
        )

        # Register Neural baselines
        from . import TrajectoryLSTM, VesselGCN, AnomalyAutoencoder

        self.register(
            name="trajectory_lstm",
            model_class=TrajectoryLSTM,
            category="neural",
            description="LSTM for sequential trajectory prediction",
            computational_complexity="O(n*d²)",  # d = hidden dimension
            supports_uncertainty=False,
            supports_multi_horizon=True,
            requires_gpu=True
        )

    def get_baseline(self, name: str) -> BaselineInfo:
        """Get baseline information by name."""
        if name not in self._baselines:
            available = list(self._baselines.keys())
            raise ValueError(f"Baseline '{name}' not found. Available: {available}")
        return self._baselines[name]

    def list_baselines(
        self,
        category: Optional[str] = None,
        supports_uncertainty: Optional[bool] = None,
        requires_gpu: Optional[bool] = None
    ) -> List[BaselineInfo]:
        """List baselines with optional filtering."""

        baselines = list(self._baselines.values())

        if category is not None:
            baselines = [b for b in baselines if b.category == category]

        if supports_uncertainty is not None:
            baselines = [b for b in baselines if b.supports_uncertainty == supports_uncertainty]

        if requires_gpu is not None:
            baselines = [b for b in baselines if b.requires_gpu == requires_gpu]

        return baselines

    def create_baseline(
        self,
        name: str,
        **kwargs
    ) -> TrajectoryBaseline:
        """Create baseline model instance."""

        baseline_info = self.get_baseline(name)

        try:
            return baseline_info.model_class(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create baseline '{name}': {e}")

    def get_categories(self) -> Dict[str, List[str]]:
        """Get all categories and their baselines."""
        return self._categories.copy()

    def _implements_protocol(self, model_class: Type) -> bool:
        """Check if model class implements TrajectoryBaseline protocol."""
        required_methods = {'fit', 'predict', 'get_model_info'}

        class_methods = set(name for name, _ in inspect.getmembers(
            model_class, predicate=inspect.isfunction
        ))

        return required_methods.issubset(class_methods)

    def export_registry(self, filepath: Path) -> None:
        """Export registry to JSON file."""

        registry_data = {}
        for name, info in self._baselines.items():
            registry_data[name] = {
                'category': info.category,
                'description': info.description,
                'parameters': info.parameters,
                'tags': info.tags,
                'paper_reference': info.paper_reference,
                'computational_complexity': info.computational_complexity,
                'supports_uncertainty': info.supports_uncertainty,
                'supports_multi_horizon': info.supports_multi_horizon,
                'requires_gpu': info.requires_gpu,
                'model_class': f"{info.model_class.__module__}.{info.model_class.__name__}"
            }

        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)


# Global registry instance
BASELINE_REGISTRY = BaselineRegistry()
```

### 2. Comprehensive Evaluation Pipeline

**File**: `src/models/baseline_models/evaluation.py`

```python
"""
Comprehensive evaluation pipeline for baseline models.
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tracemalloc
import psutil
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ..evalx.validation.protocols import maritime_cv_split, validate_split_quality
from ..evalx.validation.comparisons import ModelComparison, ComparisonResult
from ..evalx.metrics.enhanced_metrics import TrajectoryMetrics
from .registry import BASELINE_REGISTRY, BaselineInfo
from .kalman.protocols import TrajectoryBaseline, BaselineResult


@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluation."""

    # Cross-validation settings
    cv_strategy: str = "temporal"  # "temporal", "vessel", "combined"
    n_splits: int = 5
    min_gap_minutes: int = 60

    # Evaluation settings
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    metrics: List[str] = field(default_factory=lambda: ["ADE", "FDE", "RMSE", "MAE"])

    # Performance profiling
    profile_memory: bool = True
    profile_time: bool = True
    track_gpu: bool = False

    # Statistical testing
    confidence_level: float = 0.95
    n_bootstrap: int = 9999
    correction_method: str = "holm"

    # Parallel processing
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""

    model_name: str
    category: str

    # Prediction metrics
    metrics: Dict[str, np.ndarray]  # metric_name -> scores per fold
    horizon_metrics: Dict[int, Dict[str, np.ndarray]]  # horizon -> metrics

    # Performance profiling
    avg_training_time: float = 0.0
    avg_prediction_time: float = 0.0
    peak_memory_mb: float = 0.0

    # Model characteristics
    model_info: Dict[str, Any] = field(default_factory=dict)
    supports_uncertainty: bool = False

    # Error analysis
    failure_rate: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    config: EvaluationConfig
    models_evaluated: List[str]

    # Individual model performance
    individual_results: Dict[str, ModelPerformance]

    # Comparative analysis
    comparison_results: Dict[str, ComparisonResult]  # per horizon/metric

    # Summary statistics
    summary_table: pd.DataFrame
    performance_table: pd.DataFrame

    # Cross-validation quality
    cv_quality_metrics: Dict[str, Any]

    # Runtime information
    total_runtime_minutes: float
    evaluation_timestamp: str


class BaselineEvaluationPipeline:
    """
    Comprehensive pipeline for evaluating baseline models.

    Integrates with existing evalx framework and provides unified
    evaluation across all baseline categories.
    """

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.trajectory_metrics = TrajectoryMetrics()
        self.model_comparison = ModelComparison(
            confidence_level=self.config.confidence_level,
            n_bootstrap=self.config.n_bootstrap,
            correction_method=self.config.correction_method
        )

    def evaluate_baselines(
        self,
        sequences: List[np.ndarray],
        baseline_names: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> EvaluationReport:
        """
        Evaluate multiple baseline models.

        Args:
            sequences: List of trajectory sequences [seq_len, n_features]
            baseline_names: Names of baselines to evaluate (if None, evaluates all)
            metadata: Optional metadata for each sequence

        Returns:
            Comprehensive evaluation report
        """

        start_time = time.time()

        # Determine baselines to evaluate
        if baseline_names is None:
            # Evaluate all available baselines
            available_baselines = BASELINE_REGISTRY.list_baselines()
            baseline_names = [b.name for b in available_baselines]

        if self.config.verbose:
            print(f"Evaluating {len(baseline_names)} baselines on {len(sequences)} sequences")
            print(f"Baselines: {baseline_names}")

        # Prepare data for cross-validation
        df = self._sequences_to_dataframe(sequences, metadata)

        # Generate cross-validation splits
        cv_splits = maritime_cv_split(
            df,
            split_type=self.config.cv_strategy,
            n_splits=self.config.n_splits,
            min_gap_minutes=self.config.min_gap_minutes
        )

        # Validate split quality
        cv_quality = validate_split_quality(df, cv_splits)

        if self.config.verbose:
            print(f"Generated {len(cv_splits)} CV splits with quality metrics:")
            for key, value in cv_quality.items():
                if not key.endswith('_sizes') and not key.endswith('_minutes'):
                    print(f"  {key}: {value}")

        # Evaluate each baseline
        individual_results = {}

        for baseline_name in tqdm(baseline_names, desc="Evaluating baselines"):
            try:
                performance = self._evaluate_single_baseline(
                    baseline_name, sequences, cv_splits, metadata
                )
                individual_results[baseline_name] = performance

                if self.config.verbose:
                    print(f"✓ {baseline_name}: {performance.category} baseline completed")

            except Exception as e:
                warnings.warn(f"Failed to evaluate {baseline_name}: {e}")
                continue

        # Comparative analysis across horizons
        comparison_results = {}
        for horizon in self.config.prediction_horizons:
            horizon_results = {}

            for baseline_name, performance in individual_results.items():
                if horizon in performance.horizon_metrics:
                    horizon_results[baseline_name] = performance.horizon_metrics[horizon]

            if len(horizon_results) >= 2:
                comparison = self.model_comparison.compare_models(
                    horizon_results, self.config.metrics
                )
                comparison_results[f"horizon_{horizon}"] = comparison

        # Generate summary tables
        summary_table = self._create_summary_table(individual_results)
        performance_table = self._create_performance_table(individual_results)

        # Calculate total runtime
        total_runtime = (time.time() - start_time) / 60.0

        return EvaluationReport(
            config=self.config,
            models_evaluated=list(individual_results.keys()),
            individual_results=individual_results,
            comparison_results=comparison_results,
            summary_table=summary_table,
            performance_table=performance_table,
            cv_quality_metrics=cv_quality,
            total_runtime_minutes=total_runtime,
            evaluation_timestamp=pd.Timestamp.now().isoformat()
        )

    def _evaluate_single_baseline(
        self,
        baseline_name: str,
        sequences: List[np.ndarray],
        cv_splits: List,
        metadata: Optional[List[Dict[str, Any]]]
    ) -> ModelPerformance:
        """Evaluate a single baseline model."""

        baseline_info = BASELINE_REGISTRY.get_baseline(baseline_name)

        # Performance tracking
        training_times = []
        prediction_times = []
        peak_memory = 0.0

        # Metrics per horizon
        horizon_metrics = {h: {} for h in self.config.prediction_horizons}

        # Overall metrics (averaged across horizons)
        overall_metrics = {metric: [] for metric in self.config.metrics}

        failures = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            try:
                # Create model instance
                model = BASELINE_REGISTRY.create_baseline(baseline_name)

                # Prepare data
                train_sequences = [sequences[i] for i in train_idx]
                test_sequences = [sequences[i] for i in test_idx]

                train_metadata = [metadata[i] for i in train_idx] if metadata else None

                # Training phase with profiling
                if self.config.profile_memory:
                    tracemalloc.start()

                train_start = time.time()
                model.fit(train_sequences, train_metadata)
                train_time = time.time() - train_start
                training_times.append(train_time)

                if self.config.profile_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    peak_memory = max(peak_memory, peak / (1024 * 1024))  # MB
                    tracemalloc.stop()

                # Prediction and evaluation phase
                pred_start = time.time()

                # Evaluate on each horizon
                for horizon in self.config.prediction_horizons:
                    horizon_scores = {metric: [] for metric in self.config.metrics}

                    for test_seq in test_sequences:
                        if len(test_seq) <= horizon:
                            continue

                        # Split sequence
                        input_seq = test_seq[:-horizon]
                        true_future = test_seq[-horizon:]

                        try:
                            # Predict
                            result = model.predict(input_seq, horizon, return_uncertainty=True)

                            if isinstance(result, BaselineResult):
                                predictions = result.predictions
                            else:
                                predictions = result

                            # Compute metrics
                            metrics_dict = self.trajectory_metrics.compute_metrics(
                                predictions, true_future
                            )

                            for metric in self.config.metrics:
                                if metric in metrics_dict:
                                    horizon_scores[metric].append(metrics_dict[metric])

                        except Exception as e:
                            failures.append(f"Prediction failed: {e}")
                            continue

                    # Store horizon-specific scores
                    for metric in self.config.metrics:
                        if horizon_scores[metric]:  # Only if we have scores
                            if metric not in horizon_metrics[horizon]:
                                horizon_metrics[horizon][metric] = []
                            horizon_metrics[horizon][metric].extend(horizon_scores[metric])

                pred_time = time.time() - pred_start
                prediction_times.append(pred_time)

            except Exception as e:
                failures.append(f"Fold {fold_idx} failed: {e}")
                continue

        # Aggregate metrics across horizons for overall performance
        for horizon, metrics_dict in horizon_metrics.items():
            for metric, scores in metrics_dict.items():
                if scores:
                    # Average this horizon's contribution to overall metric
                    overall_metrics[metric].extend(scores)

        # Convert to numpy arrays
        for metric in overall_metrics:
            overall_metrics[metric] = np.array(overall_metrics[metric])

        for horizon in horizon_metrics:
            for metric in horizon_metrics[horizon]:
                if horizon_metrics[horizon][metric]:
                    horizon_metrics[horizon][metric] = np.array(horizon_metrics[horizon][metric])

        # Calculate failure rate
        total_attempts = len(cv_splits) * len(self.config.prediction_horizons)
        failure_rate = len(failures) / max(total_attempts, 1)

        return ModelPerformance(
            model_name=baseline_name,
            category=baseline_info.category,
            metrics=overall_metrics,
            horizon_metrics=horizon_metrics,
            avg_training_time=np.mean(training_times) if training_times else 0.0,
            avg_prediction_time=np.mean(prediction_times) if prediction_times else 0.0,
            peak_memory_mb=peak_memory,
            model_info=model.get_model_info() if hasattr(model, 'get_model_info') else {},
            supports_uncertainty=baseline_info.supports_uncertainty,
            failure_rate=failure_rate,
            failure_reasons=failures[:10]  # Keep first 10 failure reasons
        )

    def _sequences_to_dataframe(
        self,
        sequences: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Convert sequences to DataFrame for cross-validation."""

        rows = []
        for seq_idx, seq in enumerate(sequences):
            seq_metadata = metadata[seq_idx] if metadata else {}
            mmsi = seq_metadata.get('mmsi', f'vessel_{seq_idx}')

            for point_idx, point in enumerate(seq):
                row = {
                    'sequence_id': seq_idx,
                    'point_id': point_idx,
                    'mmsi': mmsi,
                    'timestamp': point_idx * 60,  # Assume 1-minute intervals
                    'lat': point[0] if len(point) > 0 else 0.0,
                    'lon': point[1] if len(point) > 1 else 0.0
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _create_summary_table(self, results: Dict[str, ModelPerformance]) -> pd.DataFrame:
        """Create summary table of all results."""

        rows = []
        for model_name, performance in results.items():
            for metric in self.config.metrics:
                if metric in performance.metrics and len(performance.metrics[metric]) > 0:
                    scores = performance.metrics[metric]
                    row = {
                        'Model': model_name,
                        'Category': performance.category,
                        'Metric': metric,
                        'Mean': np.mean(scores),
                        'Std': np.std(scores, ddof=1),
                        'Min': np.min(scores),
                        'Max': np.max(scores),
                        'Count': len(scores)
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def _create_performance_table(self, results: Dict[str, ModelPerformance]) -> pd.DataFrame:
        """Create performance profiling table."""

        rows = []
        for model_name, performance in results.items():
            row = {
                'Model': model_name,
                'Category': performance.category,
                'Avg_Training_Time_s': performance.avg_training_time,
                'Avg_Prediction_Time_s': performance.avg_prediction_time,
                'Peak_Memory_MB': performance.peak_memory_mb,
                'Supports_Uncertainty': performance.supports_uncertainty,
                'Failure_Rate': performance.failure_rate
            }
            rows.append(row)

        return pd.DataFrame(rows)
```

### 3. Performance Profiling System

**File**: `src/models/baseline_models/profiling.py`

```python
"""
Performance profiling system for baseline models.
"""

import time
import psutil
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional
import functools

import numpy as np


@dataclass
class ProfileResult:
    """Results from performance profiling."""

    execution_time: float  # seconds
    peak_memory_mb: float
    cpu_percent: float
    memory_percent: float

    # Detailed timing
    fit_time: Optional[float] = None
    predict_time: Optional[float] = None

    # Model-specific metrics
    model_size_mb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None


class PerformanceProfiler:
    """Profiler for baseline model performance."""

    def __init__(self, track_memory: bool = True, track_cpu: bool = True):
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self._process = psutil.Process()

    @contextmanager
    def profile(self) -> Generator[ProfileResult, None, None]:
        """Context manager for profiling execution."""

        # Start monitoring
        if self.track_memory:
            tracemalloc.start()

        start_time = time.time()
        start_cpu_times = self._process.cpu_times()
        start_memory = self._process.memory_info().rss

        try:
            # Create result object that will be populated
            result = ProfileResult(
                execution_time=0.0,
                peak_memory_mb=0.0,
                cpu_percent=0.0,
                memory_percent=0.0
            )

            yield result

        finally:
            # Calculate final metrics
            end_time = time.time()
            result.execution_time = end_time - start_time

            # Memory metrics
            if self.track_memory:
                current, peak = tracemalloc.get_traced_memory()
                result.peak_memory_mb = peak / (1024 * 1024)
                tracemalloc.stop()

            # CPU metrics
            if self.track_cpu:
                end_cpu_times = self._process.cpu_times()
                cpu_time = (end_cpu_times.user - start_cpu_times.user +
                           end_cpu_times.system - start_cpu_times.system)
                result.cpu_percent = (cpu_time / result.execution_time) * 100 if result.execution_time > 0 else 0

            # Memory percentage
            end_memory = self._process.memory_info().rss
            result.memory_percent = ((end_memory - start_memory) / start_memory * 100
                                   if start_memory > 0 else 0)


def profile_model(profiler: PerformanceProfiler = None):
    """Decorator for profiling model methods."""

    if profiler is None:
        profiler = PerformanceProfiler()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile() as profile_result:
                result = func(*args, **kwargs)

            # Attach profiling info to result if possible
            if hasattr(result, '__dict__'):
                result._profile_info = profile_result

            return result
        return wrapper
    return decorator


class BenchmarkSuite:
    """Benchmark suite for comparing baseline models."""

    def __init__(self):
        self.profiler = PerformanceProfiler()

    def benchmark_baseline(
        self,
        model,
        train_sequences: list,
        test_sequences: list,
        horizons: list = [1, 3, 6, 12]
    ) -> Dict[str, Any]:
        """Benchmark a single baseline model."""

        results = {
            'model_name': model.__class__.__name__,
            'training': {},
            'prediction': {},
            'memory': {},
            'throughput': {}
        }

        # Training benchmark
        with self.profiler.profile() as train_profile:
            model.fit(train_sequences)

        results['training'] = {
            'time_seconds': train_profile.execution_time,
            'memory_mb': train_profile.peak_memory_mb,
            'cpu_percent': train_profile.cpu_percent
        }

        # Prediction benchmark for each horizon
        for horizon in horizons:
            horizon_results = []

            with self.profiler.profile() as pred_profile:
                for seq in test_sequences[:100]:  # Limit for benchmarking
                    if len(seq) > horizon:
                        input_seq = seq[:-horizon]
                        try:
                            prediction = model.predict(input_seq, horizon)
                            horizon_results.append(len(input_seq))
                        except:
                            continue

            results['prediction'][f'horizon_{horizon}'] = {
                'time_seconds': pred_profile.execution_time,
                'memory_mb': pred_profile.peak_memory_mb,
                'throughput_sequences_per_sec': len(horizon_results) / pred_profile.execution_time if pred_profile.execution_time > 0 else 0,
                'sequences_processed': len(horizon_results)
            }

        return results
```

### 4. Integration with Existing Factory

**File**: Update to `src/models/baseline_models/__init__.py`

```python
# Add imports for M1.4
from .registry import BASELINE_REGISTRY, BaselineInfo
from .evaluation import BaselineEvaluationPipeline, EvaluationConfig
from .profiling import PerformanceProfiler, BenchmarkSuite

# Update __all__
__all__ = [
    # ... existing exports ...
    "BASELINE_REGISTRY",
    "BaselineInfo",
    "BaselineEvaluationPipeline",
    "EvaluationConfig",
    "PerformanceProfiler",
    "BenchmarkSuite"
]

# Update factory function
def create_baseline_model(model_type: str, **kwargs):
    """Create a baseline model using the unified registry."""
    return BASELINE_REGISTRY.create_baseline(model_type, **kwargs)

# Convenience functions
def evaluate_all_baselines(sequences, **kwargs):
    """Evaluate all registered baselines."""
    pipeline = BaselineEvaluationPipeline()
    return pipeline.evaluate_baselines(sequences, **kwargs)

def list_available_baselines(category=None):
    """List available baselines with optional category filter."""
    return BASELINE_REGISTRY.list_baselines(category=category)
```

## Usage Examples

### Basic Usage
```python
from src.models.baseline_models import (
    BASELINE_REGISTRY,
    BaselineEvaluationPipeline,
    EvaluationConfig
)

# List available baselines
physics_baselines = BASELINE_REGISTRY.list_baselines(category="physics")
print(f"Physics-based baselines: {[b.name for b in physics_baselines]}")

# Create and run comprehensive evaluation
config = EvaluationConfig(
    cv_strategy="temporal",
    n_splits=5,
    prediction_horizons=[1, 3, 6, 12],
    metrics=["ADE", "FDE", "RMSE"],
    profile_memory=True,
    n_bootstrap=9999
)

pipeline = BaselineEvaluationPipeline(config)
report = pipeline.evaluate_baselines(train_sequences)

# Access results
print(f"Evaluated {len(report.models_evaluated)} baselines")
print(f"Best model for ADE: {report.comparison_results['horizon_12'].best_model['ADE']}")
print(report.summary_table.head())
```

### Performance Comparison
```python
# Compare specific baseline categories
classical_names = ["svr", "random_forest"]
physics_names = ["kalman_cv", "kalman_imm"]

classical_report = pipeline.evaluate_baselines(
    sequences, baseline_names=classical_names
)

physics_report = pipeline.evaluate_baselines(
    sequences, baseline_names=physics_names
)

# Statistical comparison
from src.evalx.validation.comparisons import ModelComparison

comparison = ModelComparison()
combined_results = {
    **classical_report.individual_results,
    **physics_report.individual_results
}

comparison_result = comparison.compare_models(
    {name: perf.metrics for name, perf in combined_results.items()}
)
```

## Testing Strategy

### Unit Tests Required
```python
# tests/unit/baseline_models/test_m1_4_integration.py

def test_baseline_registry():
    """Test baseline registration and discovery."""

def test_evaluation_pipeline():
    """Test comprehensive evaluation pipeline."""

def test_performance_profiling():
    """Test performance profiling accuracy."""

def test_statistical_validation():
    """Test integration with evalx framework."""
```

## Performance Optimization

### Parallel Processing
- Use `joblib.Parallel` for cross-validation folds
- Process different baselines in parallel when possible
- Batch predictions for efficiency

### Memory Management
- Use `tracemalloc` for accurate memory profiling
- Clear model states between evaluations
- Implement result streaming for large evaluations

### Caching Strategy
```python
from joblib import Memory

# Cache expensive computations
memory = Memory('cache_dir', verbose=0)

@memory.cache
def cached_model_evaluation(model_params, data_hash, config_hash):
    # Expensive evaluation logic
    pass
```

## Summary

M1.4 provides:

1. **Unified Registry**: Central management of all baseline models
2. **Comprehensive Evaluation**: Statistical validation with bootstrap CI
3. **Performance Profiling**: Memory, CPU, and throughput analysis
4. **Statistical Validation**: Integration with evalx framework
5. **Minimal Resistance**: Uses mature libraries (pandas, scikit-learn, joblib)

The implementation seamlessly integrates M1.1 (statistical validation), M1.2 (Kalman), and M1.3 (Classical ML) into a production-ready evaluation framework with comprehensive reporting capabilities.
