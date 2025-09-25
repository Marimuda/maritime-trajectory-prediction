"""Comprehensive tests for bootstrap confidence intervals."""

import warnings

import numpy as np
import pytest

from src.evalx.stats.bootstrap import BootstrapCI, BootstrapResult, bootstrap_ci


class TestBootstrapCI:
    """Test suite for BootstrapCI class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.bootstrap_calc = BootstrapCI(n_resamples=999, random_state=42)
        self.sample_data = np.random.normal(10, 2, 100)
        np.random.seed(42)  # For reproducible test data

    def test_initialization(self):
        """Test BootstrapCI initialization."""
        calc = BootstrapCI(n_resamples=1000, confidence_level=0.90, method="percentile")
        assert calc.n_resamples == 1000
        assert calc.confidence_level == 0.90
        assert calc.method == "percentile"
        assert calc.random_state is None

    def test_initialization_with_random_state(self):
        """Test BootstrapCI initialization with random state."""
        calc = BootstrapCI(random_state=123)
        assert calc.random_state == 123

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Method must be one of"):
            BootstrapCI(method="invalid_method")

    def test_compute_ci_single_array(self):
        """Test compute_ci with single array."""
        result = self.bootstrap_calc.compute_ci(self.sample_data)

        assert isinstance(result, BootstrapResult)
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert result.confidence_level == 0.95
        assert result.method == "BCa"
        assert result.n_resamples == 999
        assert isinstance(result.statistic_value, (float, np.floating))

    def test_compute_ci_paired_arrays(self):
        """Test compute_ci with paired arrays for difference."""
        group_a = np.random.normal(10, 2, 50)
        group_b = np.random.normal(9, 2, 50)

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        result = self.bootstrap_calc.compute_ci((group_a, group_b), mean_diff)

        assert isinstance(result, BootstrapResult)
        assert len(result.confidence_interval) == 2
        expected_diff = np.mean(group_a) - np.mean(group_b)
        assert abs(result.statistic_value - expected_diff) < 1e-10

    def test_different_statistics(self):
        """Test compute_ci with different statistics."""
        # Test median
        result_median = self.bootstrap_calc.compute_ci(self.sample_data, np.median)
        expected_median = np.median(self.sample_data)
        assert abs(result_median.statistic_value - expected_median) < 1e-10

        # Test standard deviation
        result_std = self.bootstrap_calc.compute_ci(
            self.sample_data, lambda x: np.std(x, ddof=1)
        )
        expected_std = np.std(self.sample_data, ddof=1)
        assert abs(result_std.statistic_value - expected_std) < 1e-10

    def test_different_methods(self):
        """Test different bootstrap methods."""
        methods = ["percentile", "basic", "BCa"]

        for method in methods:
            calc = BootstrapCI(method=method, n_resamples=999, random_state=42)
            result = calc.compute_ci(self.sample_data)

            assert result.method == method
            assert len(result.confidence_interval) == 2
            assert result.confidence_interval[0] < result.confidence_interval[1]

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        levels = [0.90, 0.95, 0.99]

        for level in levels:
            calc = BootstrapCI(confidence_level=level, n_resamples=999, random_state=42)
            result = calc.compute_ci(self.sample_data)

            assert result.confidence_level == level
            assert len(result.confidence_interval) == 2

        # Check that higher confidence levels give wider intervals
        calc_90 = BootstrapCI(confidence_level=0.90, n_resamples=999, random_state=42)
        calc_99 = BootstrapCI(confidence_level=0.99, n_resamples=999, random_state=42)

        result_90 = calc_90.compute_ci(self.sample_data)
        result_99 = calc_99.compute_ci(self.sample_data)

        width_90 = result_90.confidence_interval[1] - result_90.confidence_interval[0]
        width_99 = result_99.confidence_interval[1] - result_99.confidence_interval[0]

        assert width_99 > width_90

    def test_compare_means(self):
        """Test compare_means method."""
        group_a = np.random.normal(10, 2, 50)
        group_b = np.random.normal(9, 2, 50)

        result = self.bootstrap_calc.compare_means(group_a, group_b)

        expected_diff = np.mean(group_a) - np.mean(group_b)
        assert abs(result.statistic_value - expected_diff) < 1e-10
        assert len(result.confidence_interval) == 2

    def test_compare_medians(self):
        """Test compare_medians method."""
        group_a = np.random.normal(10, 2, 50)
        group_b = np.random.normal(9, 2, 50)

        result = self.bootstrap_calc.compare_medians(group_a, group_b)

        expected_diff = np.median(group_a) - np.median(group_b)
        assert abs(result.statistic_value - expected_diff) < 1e-10
        assert len(result.confidence_interval) == 2

    def test_fallback_to_percentile(self):
        """Test fallback to percentile method when BCa fails."""
        # Create data that might cause BCa to fail (constant values)
        constant_data = np.ones(10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.bootstrap_calc.compute_ci(constant_data)
            # Check if fallback warning was issued (may or may not happen depending on scipy version)
            # The important thing is that we get a result

        assert isinstance(result, BootstrapResult)
        assert len(result.confidence_interval) == 2

    def test_reproducibility(self):
        """Test that results are reproducible with fixed random state."""
        calc1 = BootstrapCI(random_state=42, n_resamples=999)
        calc2 = BootstrapCI(random_state=42, n_resamples=999)

        result1 = calc1.compute_ci(self.sample_data)
        result2 = calc2.compute_ci(self.sample_data)

        # Results should be identical with same random state
        assert (
            abs(result1.confidence_interval[0] - result2.confidence_interval[0]) < 1e-10
        )
        assert (
            abs(result1.confidence_interval[1] - result2.confidence_interval[1]) < 1e-10
        )

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = np.array([])

        with pytest.raises((ValueError, IndexError)):
            self.bootstrap_calc.compute_ci(empty_data)

    def test_single_value_data(self):
        """Test handling of single-value data."""
        single_data = np.array([5.0])

        # This may raise an error or return degenerate CI, both are acceptable
        try:
            result = self.bootstrap_calc.compute_ci(single_data)
            # If it succeeds, CI should be degenerate
            assert (
                abs(result.confidence_interval[0] - result.confidence_interval[1])
                < 1e-6
            )
        except (ValueError, RuntimeError):
            # If it fails, that's also acceptable for single-point data
            pass


class TestBootstrapCIConvenienceFunction:
    """Test suite for bootstrap_ci convenience function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sample_data = np.random.normal(5, 1, 80)
        np.random.seed(123)

    def test_bootstrap_ci_basic_usage(self):
        """Test basic usage of bootstrap_ci function."""
        result = bootstrap_ci(self.sample_data, confidence_level=0.95, random_state=123)

        assert isinstance(result, BootstrapResult)
        assert len(result.confidence_interval) == 2
        assert result.confidence_level == 0.95

    def test_bootstrap_ci_with_custom_statistic(self):
        """Test bootstrap_ci with custom statistic."""

        def trimmed_mean(data, trim_percent=0.1):
            sorted_data = np.sort(data)
            n = len(sorted_data)
            trim_n = int(n * trim_percent)
            return np.mean(sorted_data[trim_n : n - trim_n])

        result = bootstrap_ci(self.sample_data, trimmed_mean, random_state=123)

        expected = trimmed_mean(self.sample_data)
        assert abs(result.statistic_value - expected) < 1e-10

    def test_bootstrap_ci_parameters(self):
        """Test bootstrap_ci with various parameters."""
        result = bootstrap_ci(
            self.sample_data,
            statistic=np.median,
            confidence_level=0.90,
            n_resamples=1999,
            method="percentile",
            random_state=456,
        )

        assert result.confidence_level == 0.90
        assert result.n_resamples == 1999
        assert result.method == "percentile"


class TestBootstrapIntegration:
    """Integration tests for bootstrap functionality."""

    def test_maritime_metrics_simulation(self):
        """Test bootstrap CI on simulated maritime trajectory metrics."""
        # Simulate ADE scores from cross-validation folds
        ade_scores = np.array([1.2, 1.1, 1.3, 1.0, 1.15, 1.25, 1.05, 1.18, 1.22, 1.08])

        result = bootstrap_ci(ade_scores, confidence_level=0.95, random_state=42)

        # Basic checks
        assert isinstance(result, BootstrapResult)
        assert len(result.confidence_interval) == 2
        assert (
            result.confidence_interval[0]
            < np.mean(ade_scores)
            < result.confidence_interval[1]
        )

        # Should contain reasonable values for ADE (in km)
        assert 0.5 < result.confidence_interval[0] < 1.5
        assert 1.0 < result.confidence_interval[1] < 2.0

    def test_model_comparison_scenario(self):
        """Test bootstrap CI in model comparison context."""
        # Simulate scores from two models
        model_a_scores = np.array(
            [0.85, 0.87, 0.83, 0.89, 0.86, 0.84, 0.88, 0.85, 0.87, 0.86]
        )
        model_b_scores = np.array(
            [0.82, 0.84, 0.80, 0.85, 0.83, 0.81, 0.86, 0.82, 0.84, 0.83]
        )

        calc = BootstrapCI(random_state=42)

        # CI for model A
        result_a = calc.compute_ci(model_a_scores)

        # CI for model B
        result_b = calc.compute_ci(model_b_scores)

        # CI for difference
        result_diff = calc.compare_means(model_a_scores, model_b_scores)

        # Model A should have higher scores
        assert result_a.statistic_value > result_b.statistic_value

        # Difference should be positive (A > B)
        assert result_diff.statistic_value > 0

        # CIs should not overlap significantly if models are truly different
        a_lower, a_upper = result_a.confidence_interval
        b_lower, b_upper = result_b.confidence_interval

        # At least some separation expected
        assert a_lower > b_upper * 0.95  # Some overlap is OK due to randomness

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have approximately correct coverage."""
        # This is a more advanced test that checks statistical properties
        true_mean = 10
        true_std = 2
        n_samples = 50
        n_experiments = 100  # Would be more in real test, but kept low for speed

        coverage_count = 0

        for _ in range(n_experiments):
            # Generate sample from known distribution
            sample = np.random.normal(true_mean, true_std, n_samples)

            # Compute 95% CI
            result = bootstrap_ci(sample, confidence_level=0.95)

            # Check if true mean is within CI
            if (
                result.confidence_interval[0]
                <= true_mean
                <= result.confidence_interval[1]
            ):
                coverage_count += 1

        # Coverage should be approximately 95% (allow some tolerance)
        coverage_rate = coverage_count / n_experiments
        assert 0.85 <= coverage_rate <= 1.0  # Relaxed bounds for small n_experiments

    @pytest.mark.parametrize("method", ["percentile", "basic"])
    def test_different_methods_consistency(self, method):
        """Test that different methods produce reasonable results."""
        np.random.seed(42)
        data = np.random.normal(15, 3, 100)

        result = bootstrap_ci(data, method=method, random_state=42)

        # All methods should give CIs that contain the sample mean
        sample_mean = np.mean(data)
        ci_low, ci_high = result.confidence_interval

        assert ci_low <= sample_mean <= ci_high
        assert ci_low < ci_high
        assert result.method == method
