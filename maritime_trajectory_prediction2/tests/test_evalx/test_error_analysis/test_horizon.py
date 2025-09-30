"""
Comprehensive tests for the HorizonAnalyzer framework.

Tests cover:
- Horizon curve generation
- Degradation rate computation
- Critical step identification
- Sliced horizon analysis
- Statistical comparison
- Error handling and edge cases
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evalx.error_analysis.horizon import (
    HorizonAnalysisResult,
    HorizonAnalyzer,
    HorizonCurve,
    HorizonPoint,
)


class TestHorizonAnalyzer:
    """Test suite for HorizonAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create HorizonAnalyzer instance for testing."""
        return HorizonAnalyzer(
            confidence_level=0.95, n_bootstrap=100, critical_step_threshold=0.1
        )

    @pytest.fixture
    def sample_horizon_data(self):
        """Generate sample data for horizon analysis tests."""
        np.random.seed(42)  # Reproducible tests

        n_samples = 30
        horizon_length = 8
        n_features = 2  # lat, lon

        # Create predictions and targets with increasing error over horizon
        predictions = np.random.randn(n_samples, horizon_length, n_features) * 0.1
        targets = np.random.randn(n_samples, horizon_length, n_features) * 0.1

        # Add systematic error increase over horizon
        for step in range(horizon_length):
            error_multiplier = 1.0 + 0.2 * step  # Increasing error
            noise = np.random.randn(n_samples, n_features) * 0.05 * error_multiplier
            predictions[:, step, :] += noise

        # Create metadata
        metadata = {
            "vessel_type": np.random.choice([70, 71, 80, 81, 30], size=n_samples),
            "distance_to_port_km": np.random.uniform(0, 30, size=n_samples),
            "mmsi": np.arange(1000, 1000 + n_samples),
        }

        return predictions, targets, metadata

    def test_basic_horizon_analysis(self, analyzer, sample_horizon_data):
        """Test basic horizon curve generation."""
        predictions, targets, metadata = sample_horizon_data

        curve = analyzer.analyze_horizon_errors(
            predictions=predictions, targets=targets, error_metric="mae"
        )

        # Check curve structure
        assert isinstance(curve, HorizonCurve)
        assert len(curve.points) == predictions.shape[1]  # One point per horizon step
        assert curve.error_metric == "mae"
        assert curve.total_samples == predictions.shape[0]

        # Check curve properties
        assert len(curve.steps) == predictions.shape[1]
        assert len(curve.mean_errors) == predictions.shape[1]
        assert len(curve.std_errors) == predictions.shape[1]

        # Steps should be 1-indexed and sequential
        expected_steps = list(range(1, predictions.shape[1] + 1))
        assert curve.steps == expected_steps

        # Check that errors generally increase (due to our synthetic data)
        assert curve.mean_errors[-1] > curve.mean_errors[0]  # Final > initial
        assert curve.degradation_rate > 0  # Should be positive degradation

    def test_horizon_points_structure(self, analyzer, sample_horizon_data):
        """Test HorizonPoint object structure."""
        predictions, targets, metadata = sample_horizon_data

        curve = analyzer.analyze_horizon_errors(predictions, targets)

        for i, point in enumerate(curve.points):
            assert isinstance(point, HorizonPoint)
            assert point.step == i + 1  # 1-indexed
            assert point.mean_error >= 0  # Non-negative
            assert point.std_error >= 0  # Non-negative
            assert point.n_samples == predictions.shape[0]
            assert point.bootstrap_ci is not None  # Should have CI
            assert point.per_sample_errors is not None
            assert len(point.per_sample_errors) == predictions.shape[0]

    def test_error_metrics(self, analyzer, sample_horizon_data):
        """Test different error metrics (MAE, MSE, RMSE)."""
        predictions, targets, metadata = sample_horizon_data

        metrics = ["mae", "mse", "rmse"]

        curves = {}
        for metric in metrics:
            curve = analyzer.analyze_horizon_errors(
                predictions, targets, error_metric=metric, include_bootstrap=False
            )
            curves[metric] = curve

            assert curve.error_metric == metric
            assert all(error >= 0 for error in curve.mean_errors)

        # Check that different metrics give different but reasonable results
        mae_final = curves["mae"].mean_errors[-1]
        mse_final = curves["mse"].mean_errors[-1]
        rmse_final = curves["rmse"].mean_errors[-1]

        # Note: RMSE != sqrt(MSE) because:
        # RMSE = mean(sqrt(individual_MSEs))
        # sqrt(MSE) = sqrt(mean(individual_MSEs))
        # These differ due to Jensen's inequality, so we just check reasonableness
        assert rmse_final > 0 and np.isfinite(rmse_final)

        # All metrics should be non-negative and finite
        assert mae_final >= 0 and np.isfinite(mae_final)
        assert mse_final >= 0 and np.isfinite(mse_final)
        assert rmse_final >= 0 and np.isfinite(rmse_final)

        # For small errors, MSE < MAE (since squaring < 1 values makes them smaller)
        # For large errors, MSE > MAE (since squaring > 1 values makes them larger)
        # We just verify the relationship is reasonable without strict ordering

    def test_degradation_rate_computation(self, analyzer):
        """Test degradation rate computation with known patterns."""
        # Create controlled data with known degradation pattern
        n_samples, horizon_length, n_features = 20, 5, 2

        predictions = np.zeros((n_samples, horizon_length, n_features))
        targets = np.zeros((n_samples, horizon_length, n_features))

        # Create linear error increase
        for step in range(horizon_length):
            step_error = 0.5 + 0.1 * step  # Linear increase
            predictions[:, step, :] = targets[:, step, :] + step_error

        curve = analyzer.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )

        # Should have positive degradation rate
        assert curve.degradation_rate > 0

    def test_critical_steps_identification(self, analyzer):
        """Test identification of critical steps."""
        # Create data with a clear step increase
        n_samples, horizon_length, n_features = 20, 6, 2

        predictions = np.zeros((n_samples, horizon_length, n_features))
        targets = np.zeros((n_samples, horizon_length, n_features))

        # Create error pattern with spike at step 4
        base_errors = [0.1, 0.12, 0.13, 0.25, 0.26, 0.27]  # Big jump at step 4
        for step in range(horizon_length):
            predictions[:, step, :] = targets[:, step, :] + base_errors[step]

        curve = analyzer.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )

        # Should identify step 4 as critical (>10% relative increase)
        assert 4 in curve.critical_steps

    def test_comprehensive_horizon_analysis(self, analyzer, sample_horizon_data):
        """Test comprehensive analysis with slicing."""
        predictions, targets, metadata = sample_horizon_data

        result = analyzer.comprehensive_horizon_analysis(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slice_by=["vessel_type"],
            error_metric="mae",
        )

        # Check result structure
        assert isinstance(result, HorizonAnalysisResult)
        assert isinstance(result.overall_curve, HorizonCurve)
        assert result.sliced_curves is not None
        assert result.comparative_analysis is not None
        assert result.summary_statistics is not None

        # Check sliced curves
        assert "vessel_type" in result.sliced_curves
        vessel_curves = result.sliced_curves["vessel_type"]

        # Should have curves for different vessel types
        assert len(vessel_curves) > 0
        for _bin_name, curve in vessel_curves.items():
            assert isinstance(curve, HorizonCurve)
            assert curve.error_metric == "mae"

        # Check comparative analysis
        comp_analysis = result.comparative_analysis
        assert "overall_degradation_rate" in comp_analysis
        assert "overall_critical_steps" in comp_analysis

        # Check summary statistics
        summary = result.summary_statistics
        required_keys = [
            "total_samples",
            "horizon_length",
            "error_metric",
            "initial_error",
            "final_error",
            "degradation_rate",
        ]
        for key in required_keys:
            assert key in summary

    def test_horizon_curve_comparison(self, analyzer):
        """Test statistical comparison of horizon curves."""
        # Create two different curves for comparison
        n_samples = 25

        # Curve 1: Slower degradation
        predictions1 = np.random.randn(n_samples, 5, 2) * 0.1
        targets1 = np.random.randn(n_samples, 5, 2) * 0.1
        for step in range(5):
            predictions1[:, step, :] += 0.05 * step

        # Curve 2: Faster degradation
        predictions2 = np.random.randn(n_samples, 5, 2) * 0.1
        targets2 = np.random.randn(n_samples, 5, 2) * 0.1
        for step in range(5):
            predictions2[:, step, :] += 0.1 * step

        curve1 = analyzer.analyze_horizon_errors(predictions1, targets1, "mae")
        curve2 = analyzer.analyze_horizon_errors(predictions2, targets2, "mae")

        # Compare curves
        curves = {"curve1": curve1, "curve2": curve2}
        comparison = analyzer.compare_horizon_curves(curves)

        assert "curve1_vs_curve2" in comparison
        comp_result = comparison["curve1_vs_curve2"]

        assert "step_comparisons" in comp_result
        assert "overall_better" in comp_result
        assert "degradation_rate_diff" in comp_result

        # Curve1 should have better (lower) degradation rate
        assert comp_result["degradation_rate_diff"] < 0  # curve1 - curve2 < 0
        assert comp_result["overall_better"] == "curve1"

    def test_plotting_functionality(self, analyzer, sample_horizon_data):
        """Test horizon curve plotting functionality."""
        predictions, targets, metadata = sample_horizon_data

        curve = analyzer.analyze_horizon_errors(predictions, targets)

        # Test plotting without errors
        fig = analyzer.plot_horizon_curve(curve, title="Test Curve")

        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 1

        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Prediction Step"
        assert "MAE" in ax.get_ylabel()  # Should include error metric
        assert ax.get_title() == "Test Curve"

        # Clean up
        plt.close(fig)

    def test_input_validation(self, analyzer):
        """Test input validation and error handling."""
        # Test shape mismatch
        predictions = np.random.randn(10, 5, 2)
        targets = np.random.randn(15, 5, 2)  # Different n_samples

        with pytest.raises(ValueError, match="Predictions shape .* != targets shape"):
            analyzer.analyze_horizon_errors(predictions, targets)

        # Test invalid error metric
        predictions = np.random.randn(10, 5, 2)
        targets = np.random.randn(10, 5, 2)

        with pytest.raises(ValueError, match="Unknown error metric"):
            analyzer.analyze_horizon_errors(
                predictions, targets, error_metric="invalid"
            )

    def test_bootstrap_confidence_intervals(self, analyzer, sample_horizon_data):
        """Test bootstrap confidence interval computation."""
        predictions, targets, metadata = sample_horizon_data

        curve = analyzer.analyze_horizon_errors(
            predictions, targets, include_bootstrap=True
        )

        # Check that bootstrap CIs are computed
        for point in curve.points:
            assert point.bootstrap_ci is not None
            ci = point.bootstrap_ci.confidence_interval
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert ci[0] <= point.mean_error <= ci[1]  # Mean should be within CI

        # Check confidence intervals property
        cis = curve.confidence_intervals
        assert len(cis) == len(curve.points)
        for ci in cis:
            assert ci is not None
            assert isinstance(ci, tuple)

    def test_without_bootstrap(self, analyzer, sample_horizon_data):
        """Test analysis without bootstrap confidence intervals."""
        predictions, targets, metadata = sample_horizon_data

        curve = analyzer.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )

        # Bootstrap CIs should be None
        for point in curve.points:
            assert point.bootstrap_ci is None

        # Confidence intervals should all be None
        cis = curve.confidence_intervals
        assert all(ci is None for ci in cis)

    def test_small_sample_size(self, analyzer):
        """Test behavior with small sample sizes."""
        # Very small sample
        predictions = np.random.randn(3, 4, 2) * 0.1
        targets = np.random.randn(3, 4, 2) * 0.1

        # Should still work but without bootstrap CI due to small sample
        curve = analyzer.analyze_horizon_errors(predictions, targets)

        assert len(curve.points) == 4
        assert curve.total_samples == 3

        # Bootstrap CIs might be None or present depending on implementation
        for point in curve.points:
            # With n=3 < 10, bootstrap CI should be None
            assert point.bootstrap_ci is None

    def test_sliced_analysis_with_insufficient_samples(self, analyzer):
        """Test sliced analysis when some slices have too few samples."""
        # Small dataset with imbalanced vessel types
        predictions = np.random.randn(8, 4, 2) * 0.1
        targets = np.random.randn(8, 4, 2) * 0.1

        metadata = {
            "vessel_type": [70, 70, 70, 70, 70, 80, 30, 60],  # Imbalanced
            "distance_to_port_km": np.random.uniform(0, 30, size=8),
        }

        result = analyzer.comprehensive_horizon_analysis(
            predictions, targets, metadata, slice_by=["vessel_type"]
        )

        # Should have some vessel type curves but maybe not all
        vessel_curves = result.sliced_curves["vessel_type"]

        # Cargo vessels (70) should have a curve (5 samples >= 5 minimum)
        assert "cargo" in vessel_curves

        # Other vessel types might not have curves due to insufficient samples

    def test_degradation_rate_edge_cases(self, analyzer):
        """Test degradation rate computation edge cases."""
        # Test with constant errors
        n_samples = 10
        predictions = np.ones((n_samples, 4, 2)) * 0.5
        targets = np.zeros((n_samples, 4, 2))

        curve = analyzer.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )

        # Should have near-zero degradation rate for constant error
        assert abs(curve.degradation_rate) < 0.01

        # Test with single step
        predictions_single = predictions[:, :1, :]
        targets_single = targets[:, :1, :]

        curve_single = analyzer.analyze_horizon_errors(
            predictions_single, targets_single, include_bootstrap=False
        )

        assert curve_single.degradation_rate == 0.0

    def test_critical_step_threshold_parameter(self):
        """Test different critical step thresholds."""
        # Test with different thresholds
        analyzer_strict = HorizonAnalyzer(critical_step_threshold=0.05)  # 5%
        analyzer_lenient = HorizonAnalyzer(critical_step_threshold=0.2)  # 20%

        # Create data with moderate error increases
        n_samples = 15
        predictions = np.zeros((n_samples, 5, 2))
        targets = np.zeros((n_samples, 5, 2))

        # 10% increase at step 3
        errors = [0.1, 0.11, 0.121, 0.15, 0.16]
        for step in range(5):
            predictions[:, step, :] = targets[:, step, :] + errors[step]

        curve_strict = analyzer_strict.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )
        curve_lenient = analyzer_lenient.analyze_horizon_errors(
            predictions, targets, include_bootstrap=False
        )

        # Strict threshold should identify more critical steps
        assert len(curve_strict.critical_steps) >= len(curve_lenient.critical_steps)

    def test_summary_statistics_completeness(self, analyzer, sample_horizon_data):
        """Test completeness of summary statistics."""
        predictions, targets, metadata = sample_horizon_data

        result = analyzer.comprehensive_horizon_analysis(
            predictions, targets, metadata, slice_by=["vessel_type"]
        )

        summary = result.summary_statistics

        # Required statistics
        required_keys = [
            "total_samples",
            "horizon_length",
            "error_metric",
            "initial_error",
            "final_error",
            "error_range",
            "degradation_rate",
            "n_critical_steps",
        ]

        for key in required_keys:
            assert key in summary

        # Check value types and ranges
        assert isinstance(summary["total_samples"], int)
        assert summary["total_samples"] > 0
        assert isinstance(summary["horizon_length"], int)
        assert summary["horizon_length"] > 0
        assert isinstance(summary["error_range"], tuple)
        assert len(summary["error_range"]) == 2

        # Check slice summaries if present
        if "slice_summaries" in summary:
            slice_summaries = summary["slice_summaries"]
            assert isinstance(slice_summaries, dict)

            for _slice_dim, slice_summary in slice_summaries.items():
                assert "n_bins_analyzed" in slice_summary
                assert "degradation_rates" in slice_summary
                assert "final_errors" in slice_summary


class TestHorizonCurve:
    """Test HorizonCurve data class and properties."""

    def test_horizon_curve_properties(self):
        """Test HorizonCurve properties."""
        # Create sample points
        points = [
            HorizonPoint(step=1, mean_error=0.1, std_error=0.02, n_samples=50),
            HorizonPoint(step=2, mean_error=0.15, std_error=0.03, n_samples=50),
            HorizonPoint(step=3, mean_error=0.2, std_error=0.04, n_samples=50),
        ]

        curve = HorizonCurve(
            points=points,
            error_metric="mae",
            total_samples=50,
            degradation_rate=0.05,
            critical_steps=[3],
        )

        # Test properties
        assert curve.steps == [1, 2, 3]
        assert curve.mean_errors == [0.1, 0.15, 0.2]
        assert curve.std_errors == [0.02, 0.03, 0.04]
        assert curve.confidence_intervals == [None, None, None]  # No bootstrap CIs

    def test_horizon_curve_with_bootstrap(self):
        """Test HorizonCurve with bootstrap confidence intervals."""
        from src.evalx.stats.bootstrap import BootstrapResult

        # Create mock bootstrap result
        bootstrap_result = BootstrapResult(
            confidence_interval=(0.08, 0.12),
            confidence_level=0.95,
            method="BCa",
            n_resamples=100,
            statistic_value=0.1,
            bootstrap_distribution=np.array([0.09, 0.1, 0.11]),
        )

        points = [
            HorizonPoint(
                step=1,
                mean_error=0.1,
                std_error=0.02,
                n_samples=50,
                bootstrap_ci=bootstrap_result,
            )
        ]

        curve = HorizonCurve(
            points=points,
            error_metric="mae",
            total_samples=50,
            degradation_rate=0.05,
            critical_steps=[],
        )

        cis = curve.confidence_intervals
        assert len(cis) == 1
        assert cis[0] == (0.08, 0.12)


class TestHorizonPoint:
    """Test HorizonPoint data class."""

    def test_horizon_point_creation(self):
        """Test HorizonPoint creation and attributes."""
        point = HorizonPoint(
            step=5,
            mean_error=1.5,
            std_error=0.3,
            n_samples=100,
            per_sample_errors=np.array([1.2, 1.8, 1.4]),
        )

        assert point.step == 5
        assert point.mean_error == 1.5
        assert point.std_error == 0.3
        assert point.n_samples == 100
        assert point.bootstrap_ci is None
        np.testing.assert_array_equal(point.per_sample_errors, [1.2, 1.8, 1.4])


class TestHorizonAnalysisResult:
    """Test HorizonAnalysisResult data class."""

    def test_horizon_analysis_result_creation(self):
        """Test HorizonAnalysisResult creation."""
        # Create minimal curve for testing
        points = [HorizonPoint(step=1, mean_error=0.1, std_error=0.02, n_samples=20)]
        curve = HorizonCurve(
            points=points,
            error_metric="mae",
            total_samples=20,
            degradation_rate=0.0,
            critical_steps=[],
        )

        result = HorizonAnalysisResult(
            overall_curve=curve,
            comparative_analysis={"test": "value"},
            summary_statistics={"samples": 20},
        )

        assert isinstance(result.overall_curve, HorizonCurve)
        assert result.sliced_curves is None
        assert result.comparative_analysis == {"test": "value"}
        assert result.summary_statistics == {"samples": 20}
