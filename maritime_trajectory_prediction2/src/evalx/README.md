# EvalX: Statistical Evaluation Framework for Maritime AI

EvalX is a comprehensive statistical evaluation framework designed for maritime trajectory prediction and related AI tasks. It provides robust statistical analysis, hypothesis testing, and model comparison capabilities with specific considerations for maritime domain challenges.

## Key Features

### 🔬 Statistical Rigor
- **Bootstrap Confidence Intervals**: BCa, percentile, and basic methods using scipy
- **Significance Testing**: Paired t-tests, Wilcoxon signed-rank, Cliff's delta
- **Multiple Comparison Correction**: Bonferroni, Holm, FDR methods
- **Effect Size Analysis**: Cohen's d, Cliff's delta with interpretations

### 🚢 Maritime-Specific Protocols
- **Cross-Validation**: Time-series and vessel-based splitting with leakage prevention
- **Domain Metrics**: Haversine distance, circular course handling
- **Temporal Considerations**: Gap enforcement, chronological validation
- **Vessel Generalization**: Group-based splitting by MMSI

### ⚡ Deep Learning Integration
- **PyTorch Lightning**: Seamless integration with Lightning modules
- **TorchMetrics**: Enhanced wrappers with statistical capabilities
- **Automated Collection**: Per-sample metric collection for CI computation
- **Production Ready**: Minimal overhead, robust error handling

## Quick Start

### Installation

```bash
# Core dependencies
pip install numpy pandas scipy statsmodels scikit-learn

# Optional for enhanced features
pip install pytorch-lightning torchmetrics matplotlib
```

### Basic Usage

```python
from evalx.stats import bootstrap_ci, paired_t_test
from evalx.validation import ModelComparison
import numpy as np

# Bootstrap confidence intervals
ade_scores = np.array([1.2, 1.1, 1.3, 1.0, 1.15])
result = bootstrap_ci(ade_scores, confidence_level=0.95)
print(f"ADE: {result.statistic_value:.3f} km")
print(f"95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}] km")

# Model comparison
lstm_scores = np.array([1.2, 1.1, 1.3, 1.0, 1.15])
transformer_scores = np.array([0.9, 0.8, 1.0, 0.7, 0.85])

test_result = paired_t_test(lstm_scores, transformer_scores)
print(f"p-value: {test_result.p_value:.4f}")
print(f"Effect size: {test_result.effect_size:.3f} ({test_result.effect_size_interpretation})")
```

### Cross-Validation

```python
from evalx.validation.protocols import maritime_cv_split
import pandas as pd

# Maritime-specific cross-validation
splits = maritime_cv_split(
    df,
    split_type='vessel',  # or 'temporal', 'combined'
    n_splits=5,
    min_gap_minutes=60
)

for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")
```

### Comprehensive Model Comparison

```python
from evalx.validation.comparisons import ModelComparison

# Simulate cross-validation results
results = {
    'LSTM': {'ADE': np.array([1.2, 1.1, 1.3, 1.0, 1.15])},
    'Transformer': {'ADE': np.array([0.9, 0.8, 1.0, 0.7, 0.85])}
}

comparison = ModelComparison(confidence_level=0.95)
comp_result = comparison.compare_models(results)

print("Summary Table:")
print(comp_result.summary_table)

print(f"Best model: {comp_result.best_model['ADE']}")
```

### Lightning Integration

```python
from evalx.metrics import StatisticalMetricWrapper
from torchmetrics import MetricCollection
from metrics.trajectory_metrics import ADE, FDE

# Enhance Lightning module with statistical metrics
class TrajectoryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # ... model definition

        # Add statistical capabilities to metrics
        self.val_metrics = MetricCollection({
            'ade': StatisticalMetricWrapper(ADE()),
            'fde': StatisticalMetricWrapper(FDE())
        })

    def validation_step(self, batch, batch_idx):
        # ... validation logic
        self.val_metrics.update(y_pred, y_true)

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()

        for name, result in results.items():
            self.log(f'val_{name}', result.value)
            if result.bootstrap_ci:
                ci = result.bootstrap_ci.confidence_interval
                self.log(f'val_{name}_ci_lower', ci[0])
                self.log(f'val_{name}_ci_upper', ci[1])
```

## Module Structure

```
evalx/
├── stats/                    # Statistical analysis tools
│   ├── bootstrap.py         # Bootstrap confidence intervals
│   ├── tests.py            # Significance tests
│   └── corrections.py      # Multiple comparison correction
├── validation/              # Cross-validation and comparison
│   ├── protocols.py        # CV splitting strategies
│   └── comparisons.py      # Model comparison framework
├── metrics/                # Enhanced metrics with statistics
│   └── enhanced_metrics.py # Statistical metric wrappers
└── examples/               # Usage examples
    ├── basic_usage.py      # Core functionality demo
    └── lightning_integration.py # PyTorch Lightning examples
```

## Maritime Domain Considerations

### Temporal Leakage Prevention
- **Gap Enforcement**: Configurable time gaps between train/test splits
- **Chronological Order**: Ensures proper temporal sequence in splits
- **Future Data**: Prevents accidental use of future information

### Vessel Generalization
- **MMSI-based Splitting**: Ensures no vessel appears in both train and test
- **Group K-Fold**: Proper cross-validation for vessel-specific models
- **Leakage Detection**: Automatic validation of split quality

### Maritime Metrics
- **Haversine Distance**: Great-circle distance for ADE/FDE computation
- **Circular Statistics**: Proper handling of course angles (0°/360° wraparound)
- **Domain Units**: Results in maritime-appropriate units (km, degrees, knots)

## Advanced Features

### Bootstrap Methods
- **BCa (Bias-Corrected and Accelerated)**: Most robust method when applicable
- **Percentile**: Simple quantile-based intervals
- **Basic**: Reflection method for symmetric distributions
- **Automatic Fallback**: Graceful degradation when BCa fails

### Multiple Comparison Correction
- **Family-Wise Error Rate**: Bonferroni, Holm methods
- **False Discovery Rate**: Benjamini-Hochberg, Benjamini-Yekutieli
- **Automatic Selection**: Recommended methods for different scenarios

### Effect Size Interpretation
- **Cohen's d**: Standardized effect size with interpretation
- **Cliff's delta**: Non-parametric effect size for ordinal data
- **Practical Significance**: Domain-informed thresholds

## Examples

### Running Examples

```bash
# Basic functionality
python src/evalx/examples/basic_usage.py

# Lightning integration (requires pytorch-lightning)
python src/evalx/examples/lightning_integration.py
```

### Expected Output

The basic usage example demonstrates:
1. Bootstrap confidence intervals for different statistics
2. Parametric and non-parametric significance tests
3. Multiple comparison correction methods
4. Cross-validation protocols
5. Comprehensive model comparison workflow

## Testing

```bash
# Run comprehensive test suite
python -m pytest tests/unit/evalx/ -v

# Run specific test modules
python -m pytest tests/unit/evalx/test_bootstrap.py -v
python -m pytest tests/unit/evalx/test_statistical_tests.py -v
python -m pytest tests/unit/evalx/test_validation.py -v
```

The test suite includes:
- **Statistical Accuracy**: Verification against known distributions
- **Edge Cases**: Handling of degenerate cases and errors
- **Integration**: End-to-end workflow testing
- **Maritime Scenarios**: Domain-specific test cases

## Performance Considerations

### Bootstrap Sampling
- **Default**: 9999 resamples (good balance of accuracy/speed)
- **Fast**: 999 resamples for development/testing
- **High Precision**: 19999+ resamples for publication results

### Memory Usage
- **Per-Sample Collection**: Optional for statistical analysis
- **Batch Processing**: Efficient handling of large datasets
- **Lazy Evaluation**: Deferred computation of expensive operations

### Caching
- **Result Caching**: Bootstrap distributions can be cached
- **Reproducibility**: Fixed random seeds for consistent results
- **Parallel**: Support for multi-threaded bootstrap sampling

## Contributing

### Development Setup

```bash
# Clone and install development dependencies
git clone <repository>
cd maritime_trajectory_prediction2

# Install in development mode
pip install -e .

# Install testing dependencies
pip install pytest pytest-cov
```

### Code Standards
- **Type Hints**: Full typing throughout the codebase
- **Docstrings**: NumPy-style documentation
- **Testing**: Comprehensive test coverage
- **Linting**: Follows project coding standards

### Adding New Features

1. **Statistical Methods**: Add to `src/evalx/stats/`
2. **CV Protocols**: Extend `src/evalx/validation/protocols.py`
3. **Metrics**: Enhance `src/evalx/metrics/enhanced_metrics.py`
4. **Tests**: Add comprehensive tests in `tests/unit/evalx/`

## Citation

If you use EvalX in your research, please cite:

```bibtex
@software{evalx2024,
  title={EvalX: Statistical Evaluation Framework for Maritime AI},
  author={Maritime Trajectory Prediction Team},
  year={2024},
  url={https://github.com/your-repo/maritime_trajectory_prediction}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Related Work

- **Maritime AI**: Integration with maritime trajectory prediction models
- **TorchMetrics**: Enhanced statistical capabilities for PyTorch metrics
- **SciPy Stats**: Leverages robust statistical implementations
- **Cross-Validation**: Maritime-specific extensions to scikit-learn protocols

## Support

For questions and support:
- **Documentation**: See `src/evalx/examples/` for detailed usage examples
- **Issues**: Report bugs and feature requests via GitHub issues
- **Tests**: Run the test suite for implementation validation
