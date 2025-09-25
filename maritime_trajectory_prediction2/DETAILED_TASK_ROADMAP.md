# Maritime AI Research: Detailed Task Roadmap

## Overview

This document provides a comprehensive, prioritized task breakdown for transforming the maritime trajectory prediction system into a publication-ready research platform. Tasks are organized by milestones with detailed specifications, acceptance criteria, and implementation guidelines.

## Milestone Structure

- **M1 (2 weeks)**: Statistical Rigor & Baseline Suite
- **M2 (4 weeks)**: Error Analysis & Domain Validation
- **M3 (6-8 weeks)**: Uncertainty Quantification & Multi-Region
- **M4 (10-12 weeks)**: Publication-Ready Reporting

---

# MILESTONE 1: Statistical Rigor & Baseline Suite (2 weeks)

## Task M1.1: Statistical Validation Framework
**Priority**: Critical | **Estimated Time**: 3-4 days | **Dependencies**: None

### Objective
Create robust statistical testing infrastructure for model comparison with proper confidence intervals and significance testing.

### Implementation Details

#### Subtask M1.1a: Bootstrap Confidence Intervals
**Branch**: `feature/bootstrap-ci`

```python
# File: evalx/stats/bootstrap.py
class BootstrapCI:
    """
    Bias-corrected and accelerated (BCa) confidence intervals.
    Uses existing scipy.stats implementation where possible.
    """

    def __init__(self, n_bootstrap: int = 2000, alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def bca_ci(self, data: np.ndarray, statistic_func: callable) -> Tuple[float, float]:
        """
        Compute BCa confidence interval for given statistic.

        Uses sklearn.utils.resample for bootstrap sampling.
        Uses scipy.stats for bias correction and acceleration.
        """
        pass  # Implementation using scipy.stats.bootstrap
```

**Dependencies to Use**:
- `scipy.stats.bootstrap` (primary)
- `sklearn.utils.resample` (fallback)
- `numpy.percentile`

**Tests**:
```python
# tests/unit/test_bootstrap_ci.py
def test_bootstrap_ci_coverage():
    """Test that 95% CI contains true parameter 95% of times."""

def test_bootstrap_ci_known_distribution():
    """Test CI on normal distribution with known parameters."""

def test_bootstrap_ci_small_samples():
    """Test behavior with small sample sizes."""
```

#### Subtask M1.1b: Model Comparison Tests
**Branch**: `feature/model-comparison-tests`

```python
# File: evalx/stats/tests.py
from scipy import stats
from typing import Dict, Any

def paired_t_test(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, Any]:
    """
    Paired t-test for model comparison.
    Uses scipy.stats.ttest_rel.
    """

def wilcoxon_signed_rank(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, Any]:
    """
    Non-parametric alternative to paired t-test.
    Uses scipy.stats.wilcoxon.
    """

def cliffs_delta(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, float]:
    """
    Effect size measure (Cliff's delta).
    Pure NumPy implementation.
    """

def multiple_comparison_correction(p_values: List[float], method: str = "holm") -> np.ndarray:
    """
    Multiple comparison correction.
    Uses statsmodels.stats.multitest.
    """
```

**Dependencies to Use**:
- `scipy.stats.ttest_rel`
- `scipy.stats.wilcoxon`
- `statsmodels.stats.multitest.multipletests`

**Tests**:
```python
def test_paired_t_test_identical_samples():
def test_wilcoxon_small_differences():
def test_cliffs_delta_effect_sizes():
def test_multiple_comparison_bonferroni():
```

### Acceptance Criteria
- [ ] Bootstrap CI has ≥95% coverage on simulated data
- [ ] All statistical tests pass unit tests
- [ ] Integration with existing metrics (ADE, FDE)
- [ ] Documentation with usage examples

---

## Task M1.2: Kalman Filter Baseline
**Priority**: Critical | **Estimated Time**: 4-5 days | **Dependencies**: None

### Objective
Implement physics-based Kalman filter baseline for trajectory prediction with proper tuning.

### Implementation Details

#### Subtask M1.2a: IMM Kalman Filter
**Branch**: `feature/kalman-baseline`

```python
# File: baselines/kalman.py
from filterpy.kalman import IMM, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

class IMMKalmanBaseline:
    """
    Interacting Multiple Model Kalman Filter for vessel trajectory prediction.

    Models:
    - Constant Velocity (CV)
    - Constant Turn Rate (CT)
    - Nearly Constant Acceleration (NCA)
    """

    def __init__(self, process_noise_std: float = 1.0):
        # Use FilterPy for proven implementation
        self.filters = self._create_filters()
        self.imm = IMM(filters=self.filters, mu=[0.33, 0.33, 0.34])

    def _create_filters(self):
        """Create CV, CT, NCA filters using FilterPy."""
        pass

    def fit(self, sequences: np.ndarray, meta: dict) -> None:
        """Tune process/measurement noise via cross-validation."""
        pass

    def predict(self, sequence: np.ndarray, horizon: int) -> np.ndarray:
        """Predict future trajectory points."""
        pass
```

**Dependencies to Use**:
- `filterpy` (primary Kalman filter library)
- `sklearn.model_selection.GridSearchCV` for hyperparameter tuning
- `scipy.optimize.minimize` for noise parameter optimization

**Tests**:
```python
def test_kalman_constant_velocity():
    """Test CV model on synthetic straight-line trajectory."""

def test_kalman_circular_motion():
    """Test CT model on synthetic turning trajectory."""

def test_kalman_noise_tuning():
    """Test that noise tuning improves performance."""
```

#### Subtask M1.2b: ARIMA Baseline
**Branch**: `feature/arima-baseline`

```python
# File: baselines/arima.py
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

class ARIMABaseline:
    """
    ARIMA model for speed prediction + dead reckoning for position.
    Uses automatic model selection via AIC.
    """

    def __init__(self):
        self.sog_model = None
        self.cog_model = None

    def fit(self, sequences: np.ndarray, meta: dict) -> None:
        """Auto-select ARIMA orders using AIC."""
        # Use statsmodels auto_arima equivalent
        pass

    def predict(self, sequence: np.ndarray, horizon: int) -> np.ndarray:
        """Predict SOG/COG then integrate to position."""
        pass
```

**Dependencies to Use**:
- `statsmodels.tsa.arima.model.ARIMA`
- `pmdarima.auto_arima` for automatic order selection
- `statsmodels.tsa.seasonal.seasonal_decompose`

### Acceptance Criteria
- [ ] IMM Kalman achieves reasonable performance on synthetic data
- [ ] ARIMA baseline handles seasonal patterns
- [ ] Both models implement standard `TrajBaseline` protocol
- [ ] Hyperparameter tuning via cross-validation
- [ ] Performance comparison with LSTM baseline

---

## Task M1.3: Classical ML Baselines
**Priority**: High | **Estimated Time**: 3-4 days | **Dependencies**: M1.2

### Objective
Implement Support Vector Regression and Random Forest baselines with proper time-aware cross-validation.

### Implementation Details

#### Subtask M1.3a: SVR/RF Implementation
**Branch**: `feature/classical-ml-baselines`

```python
# File: baselines/svr_rf.py
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

class SVRBaseline:
    """
    Support Vector Regression for trajectory prediction.
    Predicts Δlat, Δlon, Δcog with proper time-aware CV.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf'))
        ])
        self.tscv = TimeSeriesSplit(n_splits=5, gap=24*3600)  # 1-day gap

    def fit(self, sequences: np.ndarray, meta: dict) -> None:
        """Fit with time-aware cross-validation to avoid leakage."""
        pass

class RFBaseline:
    """Random Forest baseline with time-aware features."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
```

**Dependencies to Use**:
- `sklearn.svm.SVR`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.model_selection.TimeSeriesSplit`
- `sklearn.model_selection.GridSearchCV`

**Tests**:
```python
def test_svr_time_aware_cv():
    """Test that CV respects temporal order."""

def test_rf_feature_importance():
    """Test feature importance extraction."""

def test_delta_prediction_integration():
    """Test that Δlat, Δlon integration works correctly."""
```

### Acceptance Criteria
- [ ] Time-aware CV prevents data leakage
- [ ] Both models handle multivariate output (lat, lon, sog, cog)
- [ ] Hyperparameter optimization via GridSearchCV
- [ ] Feature importance analysis for RF
- [ ] Performance metrics comparable to literature baselines

---

# MILESTONE 2: Error Analysis & Domain Validation (4 weeks)

## Task M2.1: Error Analysis Framework
**Priority**: High | **Estimated Time**: 5-6 days | **Dependencies**: M1.1

### Objective
Create comprehensive error analysis tools for understanding model failures and performance patterns.

### Implementation Details

#### Subtask M2.1a: Performance Slicing
**Branch**: `feature/error-slicing`

```python
# File: evalx/error_analysis/slicers.py
from dataclasses import dataclass
from typing import Dict, List, Callable

@dataclass
class SliceConfig:
    name: str
    slicer_func: Callable
    bins: List[str]

class ErrorSlicer:
    """
    Slice performance by various maritime conditions.
    """

    def __init__(self):
        self.slices = {
            'vessel_type': SliceConfig(
                name='Vessel Type',
                slicer_func=self._slice_by_vessel_type,
                bins=['cargo', 'tanker', 'fishing', 'passenger', 'other']
            ),
            'traffic_density': SliceConfig(
                name='Traffic Density',
                slicer_func=self._slice_by_traffic_density,
                bins=['low', 'medium', 'high']
            ),
            'distance_to_port': SliceConfig(
                name='Port Proximity',
                slicer_func=self._slice_by_port_distance,
                bins=['<5km', '5-20km', '>20km']
            ),
            'prediction_horizon': SliceConfig(
                name='Horizon Steps',
                slicer_func=self._slice_by_horizon,
                bins=[f'step_{i}' for i in range(1, 13)]
            )
        }

    def slice_errors(self, predictions: np.ndarray, targets: np.ndarray,
                    metadata: Dict) -> Dict[str, Dict]:
        """Compute error metrics for each slice."""
        pass
```

**Dependencies to Use**:
- `pandas.cut` for binning
- `numpy.percentile` for quantile-based slicing
- `sklearn.cluster.KMeans` for failure clustering
- `matplotlib.pyplot` for visualization

#### Subtask M2.1b: Failure Mining
**Branch**: `feature/failure-mining`

```python
# File: evalx/error_analysis/mining.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class FailureMiner:
    """
    Identify and characterize worst-performing cases.
    """

    def mine_failures(self, errors: np.ndarray, features: np.ndarray,
                      metadata: Dict, k_worst: int = 100) -> Dict:
        """
        1. Extract top-k worst cases by error magnitude
        2. Cluster failure cases in feature space
        3. Label clusters with dominant characteristics
        4. Generate failure case cards
        """
        pass

    def generate_case_cards(self, failure_cases: List[Dict]) -> List[Dict]:
        """Generate detailed case studies for worst failures."""
        pass
```

**Tests**:
```python
def test_error_slicing_coverage():
    """Test that all samples are assigned to slices."""

def test_failure_clustering_stability():
    """Test that clustering is stable across runs."""

def test_horizon_curve_generation():
    """Test that horizon curves are generated correctly."""
```

### Acceptance Criteria
- [ ] Error slicing covers all major maritime conditions
- [ ] Failure mining identifies interpretable clusters
- [ ] Horizon curves show error progression with confidence intervals
- [ ] Case cards provide actionable failure analysis

---

## Task M2.2: Maritime Domain Validation
**Priority**: Critical | **Estimated Time**: 6-7 days | **Dependencies**: None

### Objective
Implement maritime-specific validation metrics including CPA/TCPA computation and COLREGS compliance.

### Implementation Details

#### Subtask M2.2a: CPA/TCPA Implementation
**Branch**: `feature/cpa-tcpa-validation`

```python
# File: maritime/cpa_tcpa.py
import numpy as np
from typing import Tuple, Dict

def cpa_tcpa(pos1: np.ndarray, vel1: np.ndarray,
             pos2: np.ndarray, vel2: np.ndarray) -> Tuple[float, float]:
    """
    Compute Closest Point of Approach (CPA) and Time to CPA (TCPA).

    Args:
        pos1, pos2: Position vectors [lat, lon] in degrees
        vel1, vel2: Velocity vectors [sog*cos(cog), sog*sin(cog)] in m/s

    Returns:
        (cpa_distance_meters, tcpa_minutes)
    """
    # Convert lat/lon to local Cartesian coordinates
    # Use relative velocity vector for CPA calculation
    # Handle edge cases (parallel tracks, past CPA)
    pass

def batch_cpa_tcpa(trajectories: np.ndarray) -> np.ndarray:
    """
    Vectorized CPA/TCPA for trajectory pairs.

    Args:
        trajectories: Shape [n_pairs, 2, seq_len, 4]  # [vessel1, vessel2, time, features]

    Returns:
        cpa_tcpa_results: Shape [n_pairs, seq_len, 2]  # [cpa, tcpa] per timestep
    """
    pass

class CPAValidator:
    """Validate trajectory predictions against CPA/TCPA ground truth."""

    def __init__(self, warning_thresholds: Dict[str, float]):
        self.thresholds = warning_thresholds  # {'cpa_meters': 500, 'tcpa_minutes': 10}

    def validate_predictions(self, pred_trajectories: np.ndarray,
                           true_trajectories: np.ndarray) -> Dict[str, float]:
        """
        Compute CPA/TCPA prediction accuracy and warning metrics.
        """
        pass
```

**Dependencies to Use**:
- `numpy` for vectorized calculations
- `pyproj` for geodetic coordinate transformations
- `geopy.distance.distance` for haversine calculations

#### Subtask M2.2b: COLREGS Compliance
**Branch**: `feature/colregs-validation`

```python
# File: maritime/colregs.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

class COLREGSRule(Enum):
    RULE_13_OVERTAKING = "rule_13_overtaking"
    RULE_14_HEAD_ON = "rule_14_head_on"
    RULE_15_CROSSING = "rule_15_crossing"
    RULE_16_GIVE_WAY = "rule_16_give_way"
    RULE_17_STAND_ON = "rule_17_stand_on"

@dataclass
class EncounterSituation:
    rule: COLREGSRule
    give_way_vessel: int
    stand_on_vessel: int
    confidence: float
    parameters: Dict

class COLREGSValidator:
    """
    Parametric COLREGS compliance checking.

    Note: This is a simplified implementation for research purposes.
    Real-world COLREGS compliance requires expert maritime knowledge.
    """

    def __init__(self, params: Dict[str, float]):
        self.params = params  # Configurable thresholds for rules

    def classify_encounter(self, vessel1_track: np.ndarray,
                         vessel2_track: np.ndarray) -> EncounterSituation:
        """Classify encounter type based on relative geometry."""
        pass

    def check_compliance(self, predicted_action: np.ndarray,
                        encounter: EncounterSituation) -> Dict[str, bool]:
        """Check if predicted trajectory complies with COLREGS."""
        pass
```

**Tests**:
```python
def test_cpa_tcpa_parallel_tracks():
    """Test CPA/TCPA for parallel vessel tracks."""

def test_cpa_tcpa_crossing_tracks():
    """Test CPA/TCPA for crossing vessel tracks."""

def test_colregs_head_on_situation():
    """Test COLREGS classification for head-on encounter."""

def test_colregs_crossing_situation():
    """Test COLREGS classification for crossing encounter."""
```

### Acceptance Criteria
- [ ] CPA/TCPA calculation matches maritime navigation software
- [ ] COLREGS classifier handles basic encounter types
- [ ] Warning time distribution analysis
- [ ] False alert rate computation at operational thresholds

---

## Task M2.3: Operational Metrics
**Priority**: High | **Estimated Time**: 3-4 days | **Dependencies**: M2.2

### Objective
Implement maritime operational metrics that operators and regulators care about.

### Implementation Details

#### Subtask M2.3a: Warning System Metrics
**Branch**: `feature/operational-metrics`

```python
# File: maritime/ops_metrics.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class WarningEvent:
    timestamp: float
    vessel_id: str
    threat_type: str
    warning_time: float  # minutes before threshold breach
    false_positive: bool
    severity: str

class OperationalMetrics:
    """
    Metrics that maritime operators care about for real deployments.
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def warning_time_distribution(self, events: List[WarningEvent]) -> Dict[str, float]:
        """
        Compute warning time statistics.

        Returns:
            - median_warning_time (minutes)
            - p10_warning_time (10th percentile)
            - mean_warning_time
        """
        pass

    def false_alert_rate(self, events: List[WarningEvent],
                        time_window_hours: float = 24) -> float:
        """
        Compute false positive rate per time period.
        Critical for operational acceptance.
        """
        pass

    def coverage_analysis(self, total_vessels: int,
                         handled_vessels: int) -> Dict[str, float]:
        """
        What percentage of maritime traffic can be reliably handled?
        """
        pass

    def throughput_benchmark(self, model, test_data: np.ndarray) -> Dict[str, float]:
        """
        Measure inference throughput (vessels/second) on different hardware.
        """
        pass
```

**Dependencies to Use**:
- `numpy.percentile` for percentile calculations
- `time.perf_counter` for timing benchmarks
- `psutil` for system resource monitoring
- `torch.profiler` for GPU profiling

**Tests**:
```python
def test_warning_time_calculation():
def test_false_alert_rate_computation():
def test_coverage_percentage_calculation():
def test_throughput_benchmarking():
```

### Acceptance Criteria
- [ ] Warning time metrics align with maritime operational requirements
- [ ] False alert rate calculation validated against synthetic scenarios
- [ ] Coverage analysis identifies system limitations
- [ ] Throughput benchmarks guide deployment decisions

---

# MILESTONE 3: Uncertainty & Multi-Region (6-8 weeks)

## Task M3.1: Conformal Prediction
**Priority**: High | **Estimated Time**: 5-6 days | **Dependencies**: M1.1

### Objective
Implement conformal prediction for uncertainty quantification with coverage guarantees.

### Implementation Details

#### Subtask M3.1a: Split Conformal Prediction
**Branch**: `feature/conformal-prediction`

```python
# File: evalx/uncertainty/conformal.py
from typing import Tuple, List, Optional
import numpy as np
from sklearn.model_selection import train_test_split

class SplitConformalRegression:
    """
    Split conformal prediction for trajectory forecasting.
    Provides distribution-free prediction intervals.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Miscoverage rate (10% for 90% coverage)
        self.quantile = None

    def calibrate(self, residuals_cal: np.ndarray) -> float:
        """
        Compute quantile from calibration residuals.

        Args:
            residuals_cal: Absolute residuals on calibration set

        Returns:
            Quantile threshold for prediction bands
        """
        n = len(residuals_cal)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(residuals_cal, quantile_level)
        return self.quantile

    def predict_intervals(self, point_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals around point predictions.

        Args:
            point_predictions: Shape [n_samples, horizon, n_features]

        Returns:
            lower_bounds, upper_bounds: Same shape as point_predictions
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")

        lower = point_predictions - self.quantile
        upper = point_predictions + self.quantile
        return lower, upper

class SequenceConformalRegression:
    """
    Conformal prediction adapted for sequential trajectory prediction.
    Handles temporal dependencies in prediction intervals.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantiles_per_step = None

    def calibrate_per_step(self, residuals_cal: np.ndarray) -> np.ndarray:
        """
        Compute step-specific quantiles for sequence prediction.

        Args:
            residuals_cal: Shape [n_samples, horizon, n_features]

        Returns:
            quantiles: Shape [horizon, n_features]
        """
        pass
```

**Dependencies to Use**:
- `numpy.quantile` for quantile computation
- `sklearn.model_selection.train_test_split` for calibration splits
- Existing `mapie` library for advanced conformal prediction if available

#### Subtask M3.1b: Online Conformal Prediction
**Branch**: `feature/online-conformal`

```python
# File: evalx/uncertainty/online_conformal.py
class OnlineConformalRegression:
    """
    Online conformal prediction for streaming AIS data.
    Updates prediction intervals as new data arrives.
    """

    def __init__(self, alpha: float = 0.1, window_size: int = 1000):
        self.alpha = alpha
        self.window_size = window_size
        self.residuals_window = []
        self.current_quantile = None

    def update(self, prediction: np.ndarray, true_value: np.ndarray) -> None:
        """Update residuals window with new prediction error."""
        pass

    def get_current_intervals(self, point_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction intervals with current quantile."""
        pass
```

**Tests**:
```python
def test_conformal_coverage_guarantee():
    """Test that empirical coverage meets nominal coverage."""

def test_sequence_conformal_per_step():
    """Test step-specific intervals for trajectory sequences."""

def test_online_conformal_adaptation():
    """Test that online method adapts to distribution shifts."""
```

### Acceptance Criteria
- [ ] Empirical coverage within ±2% of nominal coverage (e.g., 88%-92% for 90% target)
- [ ] Per-step intervals for trajectory sequences
- [ ] Online adaptation for streaming data
- [ ] Integration with existing trajectory models

---

## Task M3.2: Bayesian Uncertainty
**Priority**: Medium | **Estimated Time**: 4-5 days | **Dependencies**: M3.1

### Objective
Implement MC-Dropout and ensemble methods for predictive uncertainty estimation.

### Implementation Details

#### Subtask M3.2a: MC-Dropout Implementation
**Branch**: `feature/mc-dropout-uncertainty`

```python
# File: evalx/uncertainty/bayes.py
import torch
import torch.nn as nn
from typing import List, Tuple

class MCDropoutModel(nn.Module):
    """
    Wrapper for existing models to enable MC-Dropout uncertainty estimation.
    """

    def __init__(self, base_model: nn.Module, n_samples: int = 100):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples

    def enable_dropout_inference(self):
        """Enable dropout during inference for uncertainty estimation."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates.

        Returns:
            mean_prediction: Expected prediction
            prediction_std: Predictive standard deviation
        """
        self.enable_dropout_inference()

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_samples, batch_size, ...]
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

class DeepEnsemble:
    """
    Deep ensemble for uncertainty estimation.
    Trains multiple models with different initializations.
    """

    def __init__(self, model_factory: callable, n_models: int = 5):
        self.model_factory = model_factory
        self.n_models = n_models
        self.models = []

    def train_ensemble(self, train_loader, val_loader, epochs: int = 50):
        """Train ensemble of models with different seeds."""
        for i in range(self.n_models):
            torch.manual_seed(42 + i)  # Different initialization
            model = self.model_factory()
            # Train model...
            self.models.append(model)

    def predict_ensemble(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate ensemble predictions with uncertainty."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred
```

**Dependencies to Use**:
- `torch.nn.Dropout` (enable during inference)
- `torch.manual_seed` for reproducible ensemble training
- Existing model architectures

#### Subtask M3.2b: Uncertainty Calibration
**Branch**: `feature/uncertainty-calibration`

```python
# File: evalx/uncertainty/calibration.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

class UncertaintyCalibration:
    """
    Calibrate uncertainty estimates to match empirical error rates.
    """

    def __init__(self):
        self.calibrator = None

    def fit_calibration(self, predicted_std: np.ndarray, actual_errors: np.ndarray):
        """
        Fit calibration mapping from predicted uncertainty to actual error.

        Uses isotonic regression for monotonic calibration.
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(predicted_std.flatten(), actual_errors.flatten())

    def calibrate_uncertainties(self, predicted_std: np.ndarray) -> np.ndarray:
        """Apply calibration to uncertainty estimates."""
        if self.calibrator is None:
            raise ValueError("Must fit calibration first")

        return self.calibrator.predict(predicted_std.flatten()).reshape(predicted_std.shape)

    def plot_calibration_curve(self, predicted_std: np.ndarray,
                             actual_errors: np.ndarray, n_bins: int = 10) -> plt.Figure:
        """Generate calibration curve plot."""
        pass
```

**Tests**:
```python
def test_mc_dropout_variance_reduction():
    """Test that MC-dropout variance decreases with more samples."""

def test_ensemble_uncertainty_correlation():
    """Test that ensemble uncertainty correlates with prediction error."""

def test_calibration_improvement():
    """Test that calibration improves reliability diagram."""
```

### Acceptance Criteria
- [ ] MC-Dropout provides meaningful uncertainty estimates
- [ ] Ensemble uncertainty correlates with prediction errors
- [ ] Calibration improves reliability of uncertainty estimates
- [ ] Integration with trajectory prediction models

---

## Task M3.3: Multi-Region Framework
**Priority**: Medium | **Estimated Time**: 6-7 days | **Dependencies**: None

### Objective
Create framework for multi-region experiments and cross-region generalization studies.

### Implementation Details

#### Subtask M3.3a: Region Registry System
**Branch**: `feature/multi-region-framework`

```python
# File: datax/regions/registry.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import yaml

@dataclass
class RegionMetadata:
    name: str
    description: str
    geographic_bounds: Dict[str, float]  # {'min_lat': x, 'max_lat': y, ...}
    crs: str  # Coordinate Reference System
    traffic_density: str  # 'low', 'medium', 'high'
    weather_regime: str
    data_path: Path
    message_types: List[int]
    vessel_types: List[str]
    collection_period: Dict[str, str]  # {'start': '2023-01-01', 'end': '2023-12-31'}

class RegionRegistry:
    """
    Registry for maritime regions with standardized metadata.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("configs/regions/registry.yaml")
        self.regions = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Load region metadata from YAML file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f)
                for region_name, metadata in data.items():
                    self.regions[region_name] = RegionMetadata(**metadata)

    def register_region(self, region: RegionMetadata) -> None:
        """Add new region to registry."""
        self.regions[region.name] = region
        self._save_registry()

    def get_region(self, name: str) -> RegionMetadata:
        """Get region metadata by name."""
        if name not in self.regions:
            raise ValueError(f"Region '{name}' not found in registry")
        return self.regions[name]

    def list_regions(self) -> List[str]:
        """List all registered regions."""
        return list(self.regions.keys())
```

#### Subtask M3.3b: Cross-Region Data Loaders
**Branch**: `feature/cross-region-loaders`

```python
# File: datax/regions/loaders.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple

class BaseRegionLoader(ABC):
    """
    Abstract base class for region-specific data loaders.
    """

    @abstractmethod
    def load_raw_data(self, path: Path) -> pd.DataFrame:
        """Load raw AIS data for this region."""
        pass

    @abstractmethod
    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to standard schema (13D/16D features)."""
        pass

    @abstractmethod
    def apply_region_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply region-specific normalization."""
        pass

class FaroeIslandsLoader(BaseRegionLoader):
    """Loader for Faroe Islands AIS data (existing format)."""

    def load_raw_data(self, path: Path) -> pd.DataFrame:
        # Existing implementation
        pass

class BalticSeaLoader(BaseRegionLoader):
    """Loader for Baltic Sea AIS data (future implementation)."""

    def load_raw_data(self, path: Path) -> pd.DataFrame:
        # Future: Handle different AIS message formats
        pass

class MultiRegionDataModule:
    """
    DataModule that can handle multiple regions with consistent interfaces.
    """

    def __init__(self, region_configs: Dict[str, dict]):
        self.region_configs = region_configs
        self.loaders = {}
        self._initialize_loaders()

    def _initialize_loaders(self):
        """Initialize appropriate loaders for each region."""
        loader_map = {
            'faroe_islands': FaroeIslandsLoader,
            'baltic_sea': BalticSeaLoader,
            # Add more regions as implemented
        }

        for region_name, config in self.region_configs.items():
            if region_name in loader_map:
                self.loaders[region_name] = loader_map[region_name]()

    def load_region_data(self, region_name: str) -> pd.DataFrame:
        """Load and standardize data for specified region."""
        if region_name not in self.loaders:
            raise ValueError(f"No loader for region: {region_name}")

        loader = self.loaders[region_name]
        config = self.region_configs[region_name]

        # Load, standardize, and normalize
        raw_data = loader.load_raw_data(Path(config['data_path']))
        standardized = loader.standardize_schema(raw_data)
        normalized = loader.apply_region_normalization(standardized)

        return normalized
```

#### Subtask M3.3c: Leave-One-Region-Out Experiments
**Branch**: `feature/loro-experiments`

```python
# File: datax/regions/loro_experiments.py
from sklearn.model_selection import LeaveOneGroupOut
from typing import List, Dict, Any
import numpy as np

class LeaveOneRegionOut:
    """
    Leave-One-Region-Out cross-validation for geographic generalization.
    """

    def __init__(self, regions: List[str]):
        self.regions = regions
        self.n_splits = len(regions)

    def split(self, datasets: Dict[str, Any]) -> List[Tuple[List[str], str]]:
        """
        Generate train/test splits for LORO validation.

        Returns:
            List of (train_regions, test_region) tuples
        """
        splits = []
        for test_region in self.regions:
            train_regions = [r for r in self.regions if r != test_region]
            splits.append((train_regions, test_region))
        return splits

    def evaluate_generalization(self, model, datasets: Dict[str, Any],
                              metrics: List[callable]) -> Dict[str, Dict]:
        """
        Run LORO evaluation and compute generalization metrics.
        """
        results = {}

        for train_regions, test_region in self.split(datasets):
            # Train on combined train_regions data
            # Test on test_region data
            # Compute metrics
            pass

        return results

# Hydra config for LORO experiments
# configs/experiment/traj_loro.yaml
"""
defaults:
  - base
  - _self_

experiment_type: "leave_one_region_out"
regions:
  - faroe_islands
  - baltic_sea
  - mediterranean

evaluation:
  metrics:
    - ade_km
    - fde_km
    - rmse_course_circ

cross_validation:
  type: "loro"
  n_repeats: 3
"""
```

**Tests**:
```python
def test_region_registry_crud():
    """Test region registration and retrieval."""

def test_data_loader_standardization():
    """Test that different regions produce consistent schema."""

def test_loro_split_generation():
    """Test LORO split generation."""
```

### Acceptance Criteria
- [ ] Region registry supports metadata for multiple regions
- [ ] Data loaders produce consistent schema across regions
- [ ] LORO experiments can be configured via Hydra
- [ ] Cross-region generalization metrics implemented

---

# MILESTONE 4: Publication-Ready Reporting (10-12 weeks)

## Task M4.1: Automated Report Generation
**Priority**: Medium | **Estimated Time**: 7-8 days | **Dependencies**: M1.1, M2.1, M3.1

### Objective
Create automated report generation system for publication-ready artifacts.

### Implementation Details

#### Subtask M4.1a: LaTeX Template System
**Branch**: `feature/latex-report-generation`

```python
# File: reporting/templates/latex_generator.py
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Any
import pandas as pd

class LaTeXReportGenerator:
    """
    Generate LaTeX sections for research papers.
    """

    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path("reporting/templates")
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))

    def generate_methods_section(self, config: Dict[str, Any]) -> str:
        """
        Generate Methods section with model configurations.

        Template: methods.tex.jinja2
        """
        template = self.env.get_template("methods.tex.jinja2")
        return template.render(**config)

    def generate_results_section(self, results: Dict[str, Any]) -> str:
        """
        Generate Results section with tables and statistical tests.

        Template: results.tex.jinja2
        """
        template = self.env.get_template("results.tex.jinja2")

        # Process results for LaTeX formatting
        processed_results = self._process_results_for_latex(results)

        return template.render(**processed_results)

    def _process_results_for_latex(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process numerical results for LaTeX formatting."""
        processed = {}

        # Format confidence intervals
        for metric_name, metric_data in results.get('metrics', {}).items():
            if 'mean' in metric_data and 'ci' in metric_data:
                mean = metric_data['mean']
                ci_lower, ci_upper = metric_data['ci']
                processed[f'{metric_name}_formatted'] = f"{mean:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"

        # Format p-values with significance stars
        for test_name, p_value in results.get('p_values', {}).items():
            if p_value < 0.001:
                processed[f'{test_name}_formatted'] = f"{p_value:.4f}***"
            elif p_value < 0.01:
                processed[f'{test_name}_formatted'] = f"{p_value:.3f}**"
            elif p_value < 0.05:
                processed[f'{test_name}_formatted'] = f"{p_value:.3f}*"
            else:
                processed[f'{test_name}_formatted'] = f"{p_value:.3f}"

        return processed

# Template: reporting/templates/methods.tex.jinja2
"""
\\section{Methods}

\\subsection{Dataset}
Our evaluation uses {{ dataset.name }} containing {{ dataset.n_vessels }} unique vessels
over {{ dataset.duration_days }} days, totaling {{ dataset.n_messages }} AIS messages.

\\subsection{Models}
We evaluate the following approaches:
\\begin{itemize}
{% for model in models %}
\\item \\textbf{ {{- model.name -}} }: {{ model.description }}
    {% if model.parameters %}
    ({{ model.parameters.n_params }} parameters)
    {% endif %}
{% endfor %}
\\end{itemize}

\\subsection{Evaluation Protocol}
Models are evaluated using {{ evaluation.cv_type }} with {{ evaluation.n_folds }} folds.
Metrics include {{ evaluation.metrics|join(', ') }}.
Statistical significance is assessed via {{ evaluation.statistical_test }}
with {{ evaluation.alpha }} significance level.
"""

# Template: reporting/templates/results.tex.jinja2
"""
\\section{Results}

\\subsection{Trajectory Prediction Performance}

Table~\\ref{tab:trajectory_results} shows the performance of all models
on trajectory prediction task.

\\begin{table}[h]
\\centering
\\caption{Trajectory prediction results (mean ± 95\\% CI)}
\\label{tab:trajectory_results}
\\begin{tabular}{lcccc}
\\toprule
Model & ADE (km) & FDE (km) & Course RMSE (°) & Inference (ms) \\\\
\\midrule
{% for result in trajectory_results %}
{{ result.model_name }} &
{{ result.ade_formatted }} &
{{ result.fde_formatted }} &
{{ result.course_rmse_formatted }} &
{{ result.inference_time_formatted }} \\\\
{% endfor %}
\\bottomrule
\\end{tabular}
\\end{table}
"""
```

**Dependencies to Use**:
- `jinja2` for template rendering
- `pandas` for data processing
- `pathlib` for file handling

#### Subtask M4.1b: Figure Generation Pipeline
**Branch**: `feature/automated-figure-generation`

```python
# File: reporting/figures/generator.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

class FigureGenerator:
    """
    Generate publication-quality figures with consistent styling.
    """

    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        plt.style.use(style)
        self.figure_dir = Path("outputs/figures")
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def plot_horizon_curves(self, results: Dict[str, Any],
                          save_path: str = "horizon_curves.png") -> Path:
        """
        Generate ADE/FDE vs prediction horizon curves with confidence intervals.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ADE curve
        for model_name, metrics in results.items():
            horizons = metrics['horizons']
            ade_means = metrics['ade_means']
            ade_cis = metrics['ade_cis']

            ax1.plot(horizons, ade_means, 'o-', label=model_name, linewidth=2)
            ax1.fill_between(horizons,
                           [ci[0] for ci in ade_cis],
                           [ci[1] for ci in ade_cis],
                           alpha=0.3)

        ax1.set_xlabel('Prediction Horizon (steps)')
        ax1.set_ylabel('ADE (km)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # FDE curve (similar implementation)
        # ...

        plt.tight_layout()
        save_full_path = self.figure_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_full_path

    def plot_calibration_curves(self, predictions: np.ndarray,
                              uncertainties: np.ndarray,
                              actual_errors: np.ndarray,
                              save_path: str = "calibration_curves.png") -> Path:
        """
        Generate uncertainty calibration plots.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Calibration curve
        n_bins = 10
        bin_boundaries = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        observed_frequencies = []
        expected_frequencies = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
            if mask.sum() > 0:
                observed_freq = (actual_errors[mask] <= uncertainties[mask]).mean()
                expected_freq = (bin_lower + bin_upper) / 2 / uncertainties.max()

                observed_frequencies.append(observed_freq)
                expected_frequencies.append(expected_freq)

        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(expected_frequencies, observed_frequencies, 'ro-', label='Model')
        ax1.set_xlabel('Expected Frequency')
        ax1.set_ylabel('Observed Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Reliability histogram
        ax2.hist(uncertainties, bins=20, alpha=0.7, density=True)
        ax2.set_xlabel('Predicted Uncertainty')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_full_path = self.figure_dir / save_path
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_full_path
```

### Tests**:
```python
def test_latex_methods_generation():
    """Test LaTeX methods section generation."""

def test_results_table_formatting():
    """Test that results tables format correctly."""

def test_figure_generation_consistency():
    """Test that figures are generated with consistent styling."""
```

### Acceptance Criteria
- [ ] LaTeX templates generate valid paper sections
- [ ] Figures have publication-quality formatting
- [ ] All results include proper confidence intervals and p-values
- [ ] Generated reports are reproducible from run artifacts

---

## Task M4.2: Interpretability Framework (Coarse-grained)
**Priority**: Medium | **Estimated Time**: 5-6 days | **Dependencies**: M1.2, M2.1

### Objective
Implement interpretability tools for understanding model behavior and feature importance.

### Implementation Overview
- SHAP integration for tabular features (anomaly detection)
- Attention visualization for transformer models
- Feature importance analysis for baseline models
- Graph attention heatmaps for vessel interaction models

**Key Libraries**: `shap`, `captum`, `matplotlib`, existing attention mechanisms

**Acceptance Criteria**:
- [ ] Feature importance rankings for each model type
- [ ] Attention visualizations saved as publication figures
- [ ] SHAP values computed for anomaly detection features

---

## Task M4.3: Reproducibility Infrastructure (Coarse-grained)
**Priority**: High | **Estimated Time**: 4-5 days | **Dependencies**: All previous

### Objective
Ensure complete reproducibility of experimental results with versioning and seeding.

### Implementation Overview
- Seed management across PyTorch, NumPy, CUDA
- Run manifest generation (git hash, data hash, config hash, environment)
- Data versioning integration
- Environment reproducibility (Docker/conda)

**Key Libraries**: `wandb`, `dvc`, `hydra`, `docker`

**Acceptance Criteria**:
- [ ] All randomness sources controlled
- [ ] Run manifests link results to exact experimental conditions
- [ ] Results reproducible across different hardware

---

# Implementation Guidelines & Best Practices

## Git Workflow Standards

### Branch Naming Convention
- `feature/task-name` for new features
- `bugfix/issue-description` for bug fixes
- `refactor/component-name` for refactoring

### Commit Message Format
```
type: brief description

- Detailed change description
- List specific modifications
- Include relevant issue/task references

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Pre-commit Requirements
- All tests pass
- Code formatted with `ruff` or `black`
- Type hints validated with `mypy`
- Documentation updated for public APIs

## Testing Strategy

### Unit Test Coverage
- **Critical Path**: >90% coverage for statistical functions, metrics, core algorithms
- **Integration Tests**: End-to-end pipeline tests with small datasets
- **Property Tests**: Use `hypothesis` for statistical property testing
- **Performance Tests**: Benchmark critical paths for regression detection

### Test Organization
```python
# Example test structure
def test_function_name_expected_behavior():
    """Test that function behaves correctly under normal conditions."""

def test_function_name_edge_case():
    """Test function behavior with edge case inputs."""

def test_function_name_error_handling():
    """Test that function raises appropriate errors for invalid inputs."""

# Use parameterized tests for multiple scenarios
@pytest.mark.parametrize("input,expected", [(1, 2), (2, 4), (3, 6)])
def test_function_name_multiple_inputs(input, expected):
    assert function_name(input) == expected
```

## Library Selection Guidelines

### Statistical Computing
- **Primary**: `scipy.stats` for statistical tests
- **Secondary**: `statsmodels` for advanced models
- **Avoid**: Custom implementations of well-established statistical methods

### Machine Learning
- **Classical ML**: `sklearn` (preferred for baselines)
- **Time Series**: `statsmodels`, `pmdarima`
- **Filtering**: `filterpy` for Kalman filters
- **Uncertainty**: `mapie` for conformal prediction if available

### Visualization
- **Primary**: `matplotlib` with `seaborn` styling
- **Interactive**: `plotly` for exploratory analysis only
- **Publication**: Consistent styling with rcParams

### Data Processing
- **Core**: `pandas`, `numpy`
- **Geospatial**: `pyproj`, `geopy` for coordinate transformations
- **Performance**: `numba` for computational bottlenecks only

## Code Quality Standards

### Documentation Requirements
- All public functions must have docstrings with Args/Returns
- Type hints required for all function signatures
- README files for each major module
- Example usage in docstrings for complex functions

### Error Handling
- Use specific exception types
- Provide actionable error messages
- Validate inputs at function boundaries
- Handle edge cases explicitly

### Performance Considerations
- Profile before optimizing
- Vectorize operations with NumPy where possible
- Cache expensive computations appropriately
- Use appropriate data structures for scale

## Development Dependencies

### Essential Packages
```yaml
# pyproject.toml additions
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers --cov=src --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

# New dependencies for M1-M4
scipy = "^1.11.0"          # Statistical functions
scikit-learn = "^1.3.0"    # Classical ML baselines
statsmodels = "^0.14.0"    # Time series, advanced stats
filterpy = "^1.4.5"        # Kalman filter implementation
jinja2 = "^3.1.0"          # Template rendering
matplotlib = "^3.7.0"      # Plotting
seaborn = "^0.12.0"        # Statistical plots
shap = "^0.42.0"           # Model interpretability
pyproj = "^3.6.0"          # Coordinate transformations
geopy = "^2.3.0"           # Geographic calculations
```

### Development Tools
- `pytest` with coverage reporting
- `ruff` or `black` for formatting
- `mypy` for type checking
- `pre-commit` hooks for quality gates

---

# Task Dependencies & Scheduling

## Critical Path Analysis
1. **M1.1** (Statistical Framework) → **M2.1** (Error Analysis) → **M4.1** (Reporting)
2. **M1.2** (Kalman Baseline) → **M2.2** (Domain Validation)
3. **M3.1** (Conformal Prediction) → **M3.2** (Bayesian Uncertainty)
4. **M3.3** (Multi-Region) runs parallel to uncertainty tasks

## Parallelization Opportunities
- **M1.2** and **M1.3** can be developed in parallel (different baselines)
- **M2.2** and **M2.3** can be developed in parallel (different validation types)
- **M3.1** and **M3.2** can be developed in parallel (different uncertainty methods)

## Risk Mitigation
- **High-Risk Tasks**: M3.3 (Multi-region) due to data dependencies
- **Fallback Plans**: Use synthetic multi-region data if real data unavailable
- **Testing Strategy**: Implement with toy datasets first, then scale to real data

---

This roadmap provides a structured path from the current state to a publication-ready maritime AI research platform. Each task is designed to be an independent work unit that can be developed, tested, and merged following the specified git workflow.