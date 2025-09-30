"""
Comprehensive tests for the ErrorSlicer performance slicing framework.

Tests cover:
- Basic slicing functionality
- All slicing dimensions (vessel type, traffic density, port distance, horizon)
- Bootstrap confidence intervals
- Error handling and edge cases
- Custom slicing dimensions
- Summary statistics generation
"""

import numpy as np
import pandas as pd
import pytest

from src.evalx.error_analysis.slicers import ErrorSlicer, SliceConfig, SliceResult


class TestErrorSlicer:
    """Test suite for ErrorSlicer class."""

    @pytest.fixture
    def slicer(self):
        """Create ErrorSlicer instance for testing."""
        return ErrorSlicer(
            confidence_level=0.95, n_bootstrap=100
        )  # Small n_bootstrap for speed

    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction data for testing."""
        np.random.seed(42)  # Reproducible tests

        n_samples = 50
        horizon = 5
        n_features = 2  # lat, lon

        # Generate synthetic predictions and targets
        predictions = np.random.randn(n_samples, horizon, n_features) * 0.1
        targets = np.random.randn(n_samples, horizon, n_features) * 0.1

        # Create realistic metadata
        metadata = {
            "vessel_type": np.random.choice(
                [70, 71, 80, 81, 30, 60, 0], size=n_samples
            ),
            "vessel_count_5km": np.random.randint(0, 20, size=n_samples),
            "distance_to_port_km": np.random.uniform(0, 50, size=n_samples),
            "mmsi": np.arange(1000, 1000 + n_samples),
        }

        return predictions, targets, metadata

    def test_basic_slicing_functionality(self, slicer, sample_data):
        """Test basic error slicing functionality."""
        predictions, targets, metadata = sample_data

        # Test basic slicing
        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            error_metric="mae",
            include_bootstrap=False,
        )

        # Check structure
        assert isinstance(results, dict)
        assert "vessel_type" in results
        assert "traffic_density" in results
        assert "distance_to_port" in results

        # Check vessel type results
        vessel_results = results["vessel_type"]
        assert isinstance(vessel_results, dict)
        assert "cargo" in vessel_results
        assert "tanker" in vessel_results
        assert "other" in vessel_results

        # Check result objects
        for bin_name, result in vessel_results.items():
            assert isinstance(result, SliceResult)
            assert result.slice_name == "vessel_type"
            assert result.bin_name == bin_name
            assert isinstance(result.n_samples, int)
            assert result.n_samples >= 0

    def test_vessel_type_slicing(self, slicer, sample_data):
        """Test vessel type slicing logic."""
        predictions, targets, metadata = sample_data

        # Test with known vessel types
        metadata["vessel_type"] = np.array(
            [70, 80, 30, 60, 0, 71, 81]
        )  # cargo, tanker, fishing, passenger, other
        predictions = predictions[:7]
        targets = targets[:7]

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["vessel_type"],
            include_bootstrap=False,
        )

        vessel_results = results["vessel_type"]

        # Check that we have samples in appropriate bins
        expected_bins = ["cargo", "tanker", "fishing", "passenger", "other"]
        for bin_name in expected_bins:
            assert bin_name in vessel_results

        # Verify mapping logic
        assert vessel_results["cargo"].n_samples == 2  # vessel types 70, 71
        assert vessel_results["tanker"].n_samples == 2  # vessel types 80, 81
        assert vessel_results["fishing"].n_samples == 1  # vessel type 30
        assert vessel_results["passenger"].n_samples == 1  # vessel type 60
        assert vessel_results["other"].n_samples == 1  # vessel type 0

    def test_traffic_density_slicing(self, slicer, sample_data):
        """Test traffic density slicing with vessel count."""
        predictions, targets, metadata = sample_data

        # Create known traffic density scenario
        metadata["vessel_count_5km"] = np.array(
            [1, 2, 10, 15, 20] * 10
        )  # Pattern for testing

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["traffic_density"],
            include_bootstrap=False,
        )

        traffic_results = results["traffic_density"]

        # Check that all density bins have samples
        assert traffic_results["low"].n_samples > 0
        assert traffic_results["medium"].n_samples > 0
        assert traffic_results["high"].n_samples > 0

        # Total samples should equal input
        total_samples = sum(result.n_samples for result in traffic_results.values())
        assert total_samples == len(predictions)

    def test_port_distance_slicing(self, slicer, sample_data):
        """Test port distance slicing logic."""
        predictions, targets, metadata = sample_data

        # Create known distance scenarios
        metadata["distance_to_port_km"] = np.array(
            [2, 4, 8, 15, 25, 35] * 8 + [1, 10]
        )  # Known pattern

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["distance_to_port"],
            include_bootstrap=False,
        )

        port_results = results["distance_to_port"]

        # Check bins
        assert "<5km" in port_results
        assert "5-20km" in port_results
        assert ">20km" in port_results

        # Verify logic - check a few known cases
        close_count = port_results["<5km"].n_samples
        medium_count = port_results["5-20km"].n_samples
        far_count = port_results[">20km"].n_samples

        assert close_count > 0  # Should have some < 5km
        assert medium_count > 0  # Should have some 5-20km
        assert far_count > 0  # Should have some > 20km
        assert close_count + medium_count + far_count == len(predictions)

    def test_horizon_step_analysis(self, slicer, sample_data):
        """Test horizon step analysis functionality."""
        predictions, targets, metadata = sample_data

        # Test horizon analysis
        horizon_results = slicer.analyze_horizon_steps(
            predictions=predictions,
            targets=targets,
            error_metric="mae",
            include_bootstrap=False,
        )

        # Check structure
        assert isinstance(horizon_results, dict)
        assert len(horizon_results) == predictions.shape[1]  # One result per step

        # Check individual steps
        for step in range(1, predictions.shape[1] + 1):
            step_name = f"step_{step}"
            assert step_name in horizon_results

            result = horizon_results[step_name]
            assert isinstance(result, SliceResult)
            assert result.slice_name == "prediction_horizon"
            assert result.bin_name == step_name
            assert result.n_samples == predictions.shape[0]
            assert not np.isnan(result.mean_error)
            assert not np.isnan(result.std_error)

    def test_error_metrics(self, slicer, sample_data):
        """Test different error metrics (MAE, MSE, RMSE)."""
        predictions, targets, metadata = sample_data

        metrics = ["mae", "mse", "rmse"]

        for metric in metrics:
            results = slicer.slice_errors(
                predictions=predictions,
                targets=targets,
                metadata=metadata,
                error_metric=metric,
                slicing_dimensions=["vessel_type"],
                include_bootstrap=False,
            )

            # Should get valid results
            assert "vessel_type" in results
            cargo_result = results["vessel_type"]["cargo"]

            if cargo_result.n_samples > 0:
                assert not np.isnan(cargo_result.mean_error)
                assert (
                    cargo_result.mean_error >= 0
                )  # All metrics should be non-negative

    def test_bootstrap_confidence_intervals(self, slicer, sample_data):
        """Test bootstrap confidence interval computation."""
        predictions, targets, metadata = sample_data

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["vessel_type"],
            include_bootstrap=True,
        )

        vessel_results = results["vessel_type"]

        # Check that bootstrap CIs are computed for bins with sufficient samples
        for _bin_name, result in vessel_results.items():
            if result.n_samples >= 10:  # Should have CI
                assert result.bootstrap_ci is not None
                assert hasattr(result.bootstrap_ci, "confidence_interval")
                ci_lower, ci_upper = result.bootstrap_ci.confidence_interval
                assert ci_lower <= result.mean_error <= ci_upper

    def test_empty_bin_handling(self, slicer):
        """Test handling of empty bins."""
        # Create data where some bins will be empty
        predictions = np.random.randn(5, 3, 2) * 0.1
        targets = np.random.randn(5, 3, 2) * 0.1

        # Metadata that will create empty bins
        metadata = {
            "vessel_type": np.array([70, 70, 70, 70, 70]),  # Only cargo vessels
            "distance_to_port_km": np.array([1, 2, 3, 4, 5]),  # All close to port
        }

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            include_bootstrap=False,
        )

        vessel_results = results["vessel_type"]

        # Cargo should have samples
        assert vessel_results["cargo"].n_samples == 5

        # Other vessel types should be empty
        for vessel_type in ["tanker", "fishing", "passenger"]:
            result = vessel_results[vessel_type]
            assert result.n_samples == 0
            assert np.isnan(result.mean_error)
            assert np.isnan(result.std_error)
            assert result.bootstrap_ci is None

    def test_missing_metadata_handling(self, slicer, sample_data):
        """Test error handling for missing metadata."""
        predictions, targets, _ = sample_data

        # Test with missing vessel_type
        incomplete_metadata = {"distance_to_port_km": np.random.uniform(0, 50, size=50)}

        with pytest.warns(UserWarning, match="Failed to compute slice vessel_type"):
            results = slicer.slice_errors(
                predictions=predictions,
                targets=targets,
                metadata=incomplete_metadata,
                slicing_dimensions=["vessel_type", "distance_to_port"],
            )

        # Should still get results for distance_to_port
        assert "distance_to_port" in results
        assert "vessel_type" not in results

    def test_custom_slicing_dimension(self, slicer, sample_data):
        """Test adding and using custom slicing dimensions."""
        predictions, targets, metadata = sample_data

        # Add custom speed-based slicing
        def slice_by_speed(meta, errors):
            # Mock speed data
            speeds = np.random.uniform(0, 30, size=len(errors))
            return np.where(
                speeds < 10, "slow", np.where(speeds < 20, "medium", "fast")
            )

        # Add custom slice
        slicer.add_custom_slice(
            slice_name="speed",
            slicer_func=slice_by_speed,
            bins=["slow", "medium", "fast"],
            description="Performance by vessel speed",
        )

        # Test custom slicing
        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["speed"],
            include_bootstrap=False,
        )

        assert "speed" in results
        speed_results = results["speed"]
        assert "slow" in speed_results
        assert "medium" in speed_results
        assert "fast" in speed_results

        # Check total samples
        total_samples = sum(result.n_samples for result in speed_results.values())
        assert total_samples == len(predictions)

    def test_summary_statistics(self, slicer, sample_data):
        """Test summary statistics generation."""
        predictions, targets, metadata = sample_data

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            include_bootstrap=True,
        )

        # Generate summary
        summary_df = slicer.get_summary_statistics(results)

        # Check structure
        assert isinstance(summary_df, pd.DataFrame)
        expected_columns = [
            "slice_dimension",
            "bin",
            "n_samples",
            "mean_error",
            "std_error",
            "ci_lower",
            "ci_upper",
        ]
        assert all(col in summary_df.columns for col in expected_columns)

        # Check content
        assert len(summary_df) > 0
        assert summary_df["n_samples"].sum() > 0

        # Check that we have results for different dimensions
        dimensions = summary_df["slice_dimension"].unique()
        assert "vessel_type" in dimensions
        assert "traffic_density" in dimensions
        assert "distance_to_port" in dimensions

    def test_input_validation(self, slicer):
        """Test input validation and error handling."""
        # Test shape mismatch
        predictions = np.random.randn(10, 5, 2)
        targets = np.random.randn(15, 5, 2)  # Different n_samples
        metadata = {"vessel_type": np.arange(10)}

        with pytest.raises(ValueError, match="Predictions shape .* != targets shape"):
            slicer.slice_errors(predictions, targets, metadata)

        # Test invalid error metric
        predictions = np.random.randn(10, 5, 2)
        targets = np.random.randn(10, 5, 2)

        with pytest.raises(ValueError, match="Unknown error metric"):
            slicer.slice_errors(predictions, targets, metadata, error_metric="invalid")

        # Test invalid slicing dimension
        with pytest.raises(ValueError, match="Invalid slicing dimensions"):
            slicer.slice_errors(
                predictions,
                targets,
                metadata,
                slicing_dimensions=["nonexistent_dimension"],
            )

    @pytest.mark.parametrize("error_metric", ["mae", "mse", "rmse"])
    def test_error_computation_consistency(self, slicer, error_metric):
        """Test that error computations are consistent and reasonable."""
        # Create controlled test case
        predictions = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.5, 2.5], [3.5, 4.5]]])
        targets = np.array([[[1.1, 2.1], [3.1, 4.1]], [[1.4, 2.4], [3.4, 4.4]]])

        metadata = {
            "vessel_type": np.array([70, 80]),  # cargo, tanker
            "distance_to_port_km": np.array([2.0, 15.0]),
        }

        results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            error_metric=error_metric,
            include_bootstrap=False,
        )

        # Check that errors are reasonable
        vessel_results = results["vessel_type"]
        for result in vessel_results.values():
            if result.n_samples > 0:
                assert result.mean_error >= 0  # Errors should be non-negative
                assert result.std_error >= 0  # Std should be non-negative
                assert not np.isnan(result.mean_error)
                assert not np.isnan(result.std_error)


class TestSliceResult:
    """Test SliceResult data class."""

    def test_slice_result_creation(self):
        """Test SliceResult creation and attributes."""
        result = SliceResult(
            slice_name="vessel_type",
            bin_name="cargo",
            n_samples=100,
            mean_error=1.5,
            std_error=0.8,
        )

        assert result.slice_name == "vessel_type"
        assert result.bin_name == "cargo"
        assert result.n_samples == 100
        assert result.mean_error == 1.5
        assert result.std_error == 0.8
        assert result.bootstrap_ci is None
        assert result.per_sample_errors is None
        assert result.metadata is None


class TestSliceConfig:
    """Test SliceConfig data class."""

    def test_slice_config_creation(self):
        """Test SliceConfig creation and attributes."""

        def dummy_slicer(metadata, errors):
            return np.array(["bin1"] * len(errors))

        config = SliceConfig(
            name="Test Slice",
            slicer_func=dummy_slicer,
            bins=["bin1", "bin2"],
            description="Test slicing dimension",
        )

        assert config.name == "Test Slice"
        assert config.slicer_func == dummy_slicer
        assert config.bins == ["bin1", "bin2"]
        assert config.description == "Test slicing dimension"
