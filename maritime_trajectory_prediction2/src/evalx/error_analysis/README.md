# M2.1 Error Analysis Framework

A comprehensive maritime trajectory prediction error analysis framework providing advanced performance slicing, failure mining, and horizon curve analysis capabilities.

## Overview

The M2.1 Error Analysis Framework consists of three main components:

1. **Performance Slicing (M2.1a)** - Systematic error analysis across maritime dimensions
2. **Failure Mining (M2.1b)** - Intelligent worst-case identification and clustering
3. **Horizon Curve Analysis** - Temporal error progression analysis

## Quick Start

```python
from src.evalx.error_analysis import ErrorSlicer, FailureMiner, HorizonAnalyzer

# Initialize analyzers
slicer = ErrorSlicer(confidence_level=0.95, n_bootstrap=100)
miner = FailureMiner(k_worst=50, n_clusters=5)
horizon_analyzer = HorizonAnalyzer(confidence_level=0.95, n_bootstrap=100)

# Performance slicing analysis
slice_results = slicer.slice_errors(
    predictions=predictions,     # shape: [n_samples, horizon, features]
    targets=targets,
    metadata=metadata,
    error_metric='mae',
    include_bootstrap=True
)

# Failure mining analysis
sample_errors = compute_sample_errors(predictions, targets)
feature_matrix = create_feature_matrix(metadata)

mining_results = miner.mine_failures(
    errors=sample_errors,
    features=feature_matrix,
    metadata=metadata
)

# Horizon curve analysis
horizon_results = horizon_analyzer.analyze_horizon_errors(
    predictions=predictions,
    targets=targets,
    error_metric='mae',
    include_bootstrap=True
)
```

## Components

### 1. Performance Slicing (`ErrorSlicer`)

Systematic error analysis across maritime-specific dimensions:

#### Slicing Dimensions

- **Vessel Type**: Cargo, tanker, fishing, passenger, other
- **Traffic Density**: Low (<5 vessels), medium (5-15), high (>15)
- **Port Distance**: Close (<5km), medium (5-20km), far (>20km)
- **Prediction Horizon**: Step-wise error analysis

#### Features

- Bootstrap confidence intervals for statistical rigor
- Custom slicing dimension support
- Comprehensive summary statistics
- Empty bin handling with graceful degradation

#### Example

```python
# Basic slicing analysis
results = slicer.slice_errors(
    predictions=predictions,
    targets=targets,
    metadata=metadata,
    error_metric='mae'
)

# Access vessel type performance
vessel_results = results['vessel_type']
cargo_performance = vessel_results['cargo']
print(f"Cargo vessels: {cargo_performance.n_samples} samples, "
      f"MAE: {cargo_performance.mean_error:.4f}")

# Generate summary table
summary_df = slicer.get_summary_statistics(results)
print(summary_df)
```

#### Custom Slicing Dimensions

```python
def slice_by_speed(metadata, errors):
    speeds = metadata['sog_knots']
    return np.where(speeds < 10, 'slow',
                   np.where(speeds < 20, 'medium', 'fast'))

slicer.add_custom_slice(
    slice_name='speed',
    slicer_func=slice_by_speed,
    bins=['slow', 'medium', 'fast'],
    description='Performance by vessel speed'
)
```

### 2. Failure Mining (`FailureMiner`)

Intelligent identification and analysis of worst-performing cases:

#### Features

- K-means clustering of failure patterns
- Maritime-specific characterizations
- Case study generation with recommendations
- Silhouette scoring for cluster quality

#### Core Classes

- `FailureCase`: Individual failure with features and context
- `FailureCluster`: Group of similar failures with characterization
- `FailureMiningResult`: Complete analysis results

#### Example

```python
# Mine worst performers
mining_results = miner.mine_failures(
    errors=sample_errors,
    features=feature_matrix,
    metadata=metadata
)

# Analyze failure clusters
for cluster in mining_results.clusters:
    print(f"{cluster.characterization}")
    print(f"  Cases: {cluster.n_cases}, Error: {cluster.mean_error:.4f}")

# Generate case study cards
case_cards = miner.generate_case_cards(mining_results.worst_cases[:10])

for card in case_cards:
    print(f"Case {card['case_id']}:")
    print(f"  Error: {card['error_magnitude']:.4f}")
    for rec in card['recommendations']:
        print(f"  - {rec}")
```

### 3. Horizon Curve Analysis (`HorizonAnalyzer`)

Advanced temporal error progression analysis:

#### Features

- Degradation rate computation (linear/exponential)
- Critical step identification
- Bootstrap confidence intervals
- Matplotlib plotting integration
- Comparative analysis support

#### Core Classes

- `HorizonPoint`: Single prediction step analysis
- `HorizonCurve`: Complete horizon analysis
- `HorizonAnalysisResult`: Extended analysis with metadata

#### Example

```python
# Analyze horizon degradation
horizon_results = horizon_analyzer.analyze_horizon_errors(
    predictions=predictions,
    targets=targets,
    error_metric='mae',
    include_bootstrap=True
)

# Access degradation metrics
print(f"Degradation rate: {horizon_results.degradation_rate:.4f}")
print(f"Critical steps: {horizon_results.critical_steps}")

# Plot horizon curve
fig = horizon_analyzer.plot_horizon_curve(
    horizon_results,
    title="Model Error Progression",
    show_confidence_bands=True
)

# Compare multiple models
comparison = horizon_analyzer.compare_horizon_curves([
    (model1_curve, "Model 1"),
    (model2_curve, "Model 2")
])
```

## Maritime Domain Integration

The framework is specifically designed for maritime trajectory prediction with:

### Vessel Type Mapping

```python
# AIS vessel type codes mapped to categories
VESSEL_TYPE_MAPPING = {
    70, 71, 72, 73, 74: 'cargo',      # Cargo vessels
    80, 81, 82, 83, 84, 85, 89: 'tanker',  # Tankers
    30: 'fishing',                     # Fishing vessels
    60-69: 'passenger',               # Passenger vessels
    0: 'other'                        # Unknown/other
}
```

### Maritime-Specific Recommendations

The failure mining component generates contextual recommendations:

- **Cargo vessels**: "Consider cargo-specific movement patterns"
- **Tankers**: "Add tanker-specific constraints for turning behavior"
- **Near-port scenarios**: "Improve near-port maneuvering predictions"
- **High-traffic areas**: "Better traffic interaction modeling needed"

### COLREGS Integration

Future versions will integrate with:
- Collision avoidance regulations (COLREGS)
- Closest Point of Approach (CPA) analysis
- Time to CPA (TCPA) computations

## Statistical Rigor

### Bootstrap Confidence Intervals

All components support bootstrap resampling for robust uncertainty quantification:

```python
# Configure bootstrap parameters
analyzer = ErrorSlicer(
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=42
)

# Results include confidence intervals
result = analyzer.slice_errors(..., include_bootstrap=True)
ci_lower, ci_upper = result.bootstrap_ci.confidence_interval
```

### Error Metrics

Supported error metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

## Integration Examples

### Complete Analysis Pipeline

```python
def complete_maritime_analysis(predictions, targets, metadata):
    """Complete maritime trajectory analysis pipeline."""

    # Initialize all analyzers
    slicer = ErrorSlicer(confidence_level=0.95, n_bootstrap=100)
    miner = FailureMiner(k_worst=100, n_clusters=6)
    horizon_analyzer = HorizonAnalyzer(confidence_level=0.95, n_bootstrap=100)

    # 1. Performance slicing
    slice_results = slicer.slice_errors(
        predictions=predictions,
        targets=targets,
        metadata=metadata,
        include_bootstrap=True
    )

    # 2. Horizon analysis
    horizon_results = horizon_analyzer.analyze_horizon_errors(
        predictions=predictions,
        targets=targets,
        include_bootstrap=True
    )

    # 3. Failure mining
    sample_errors = compute_sample_errors(predictions, targets)
    feature_matrix = create_feature_matrix(metadata)

    mining_results = miner.mine_failures(
        errors=sample_errors,
        features=feature_matrix,
        metadata=metadata
    )

    # Generate comprehensive report
    return {
        'performance_slices': slice_results,
        'horizon_analysis': horizon_results,
        'failure_mining': mining_results,
        'summary_statistics': slicer.get_summary_statistics(slice_results),
        'case_studies': miner.generate_case_cards(mining_results.worst_cases[:20])
    }
```

### Research Workflow Integration

```python
def evaluate_maritime_model(model, test_loader, device):
    """Evaluate maritime model with comprehensive analysis."""

    model.eval()
    all_predictions = []
    all_targets = []
    all_metadata = []

    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch.to(device))
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_metadata.append(batch.metadata)

    # Combine batch results
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metadata = combine_metadata(all_metadata)

    # Run complete analysis
    analysis_results = complete_maritime_analysis(predictions, targets, metadata)

    # Generate research insights
    return generate_research_insights(analysis_results)
```

## Testing

The framework includes comprehensive testing:

- **Unit Tests**: 53 individual component tests
- **Integration Tests**: 8 end-to-end pipeline tests
- **Test Coverage**: All critical paths and edge cases

Run tests:
```bash
# Run all error analysis tests
pytest tests/test_evalx/test_error_analysis/ -v

# Run specific component tests
pytest tests/test_evalx/test_error_analysis/test_slicers.py -v
pytest tests/test_evalx/test_error_analysis/test_mining.py -v
pytest tests/test_evalx/test_error_analysis/test_horizon.py -v

# Run integration tests
pytest tests/test_evalx/test_error_analysis/test_integration.py -v
```

## Configuration

### Hydra Configuration

```yaml
# conf/error_analysis/default.yaml
error_analysis:
  slicing:
    confidence_level: 0.95
    n_bootstrap: 100
    default_error_metric: 'mae'

  mining:
    k_worst: 50
    n_clusters: 5
    clustering_algorithm: 'kmeans'

  horizon:
    confidence_level: 0.95
    n_bootstrap: 100
    critical_step_threshold: 0.1
```

## Performance Considerations

### Memory Usage

- Bootstrap resampling: O(n_bootstrap × n_samples)
- Failure clustering: O(k_worst × n_features)
- Horizon analysis: O(n_samples × horizon)

### Computational Complexity

- Slicing analysis: O(n_samples × n_slices)
- K-means clustering: O(k × n_iterations × k_worst)
- Bootstrap CI: O(n_bootstrap × metric_computation)

### Optimization Tips

```python
# For large datasets, reduce bootstrap samples
slicer = ErrorSlicer(n_bootstrap=50)  # Instead of 1000

# Use subset for failure mining
miner = FailureMiner(k_worst=100)  # Instead of 1000

# Skip bootstrap for development
results = analyzer.analyze_errors(..., include_bootstrap=False)
```

## Future Enhancements

### Planned Features (M2.2+)

- **COLREGS Integration**: Collision avoidance rule analysis
- **CPA/TCPA Analysis**: Closest point of approach computations
- **Multi-model Comparison**: Statistical comparison frameworks
- **Temporal Analysis**: Time-series specific error patterns
- **Spatial Clustering**: Geographic error pattern analysis

### Extension Points

```python
# Custom error metrics
def custom_maritime_error(predictions, targets, metadata=None):
    """Custom maritime-specific error metric."""
    # Implementation here
    return errors

# Custom failure characterization
def custom_characterization(cluster_cases, dominant_features):
    """Custom failure cluster characterization."""
    # Implementation here
    return characterization_string

# Custom slicing dimensions
def custom_slice_function(metadata, errors):
    """Custom slicing logic."""
    # Implementation here
    return bin_assignments
```

## Contributing

### Code Structure

```
src/evalx/error_analysis/
├── __init__.py          # Main module interface
├── slicers.py           # Performance slicing framework
├── mining.py            # Failure mining framework
├── horizon.py           # Horizon curve analysis
└── README.md            # This documentation

tests/test_evalx/test_error_analysis/
├── test_slicers.py      # Slicing component tests
├── test_mining.py       # Mining component tests
├── test_horizon.py      # Horizon analysis tests
└── test_integration.py  # Integration tests
```

### Development Guidelines

1. **Maritime Focus**: All features should consider maritime domain specifics
2. **Statistical Rigor**: Include bootstrap confidence intervals where applicable
3. **Comprehensive Testing**: Maintain >95% test coverage
4. **Documentation**: Include docstrings and usage examples
5. **Performance**: Consider memory and computational efficiency

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{maritime_error_analysis_2024,
  title={Maritime Trajectory Prediction Error Analysis Framework},
  author={Maritime-PyG Development Team},
  year={2024},
  version={2.1.0},
  url={https://github.com/maritime-pyg/maritime-trajectory-prediction}
}
```

## License

This framework is part of the Maritime-PyG project and follows the same licensing terms.
