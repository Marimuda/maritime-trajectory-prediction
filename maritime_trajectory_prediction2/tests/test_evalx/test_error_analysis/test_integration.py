"""
Integration tests for the complete M2.1 Error Analysis Framework.

This module tests the full pipeline integration between all error analysis components:
- ErrorSlicer performance analysis
- FailureMiner worst-case identification
- HorizonAnalyzer temporal degradation analysis
- End-to-end maritime trajectory analysis workflow
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.evalx.error_analysis import ErrorSlicer, FailureMiner, HorizonAnalyzer


class TestM21Integration:
    """Integration tests for the complete M2.1 Error Analysis Framework."""

    @pytest.fixture
    def comprehensive_dataset(self):
        """Generate comprehensive maritime trajectory dataset for integration testing."""
        np.random.seed(42)

        n_samples = 200
        horizon = 8
        n_features = 2  # lat, lon

        # Generate predictions and targets with realistic maritime patterns
        base_predictions = np.random.randn(n_samples, horizon, n_features) * 0.05
        base_targets = np.random.randn(n_samples, horizon, n_features) * 0.05

        # Add horizon-dependent error increase (realistic degradation)
        horizon_multiplier = np.linspace(1.0, 2.5, horizon)
        predictions = base_predictions * horizon_multiplier[None, :, None]
        targets = base_targets

        # Create comprehensive maritime metadata
        vessel_types = np.random.choice(
            [70, 71, 80, 81, 30, 60, 0],
            size=n_samples,
            p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05],
        )

        metadata = {
            "vessel_type": vessel_types,
            "vessel_count_5km": np.random.randint(0, 25, size=n_samples),
            "distance_to_port_km": np.random.uniform(0, 60, size=n_samples),
            "mmsi": np.arange(100000, 100000 + n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1h"),
            "vessel_name": [f"VESSEL_{i:03d}" for i in range(n_samples)],
            "sog_knots": np.random.uniform(0, 25, size=n_samples),
            "cog_degrees": np.random.uniform(0, 360, size=n_samples),
        }

        return predictions, targets, metadata

    def test_complete_error_analysis_pipeline(self, comprehensive_dataset):
        """Test the complete error analysis pipeline integration."""
        predictions, targets, metadata = comprehensive_dataset

        # Initialize all analyzers
        slicer = ErrorSlicer(confidence_level=0.95, n_bootstrap=50)
        miner = FailureMiner(k_worst=50, n_clusters=4)
        horizon_analyzer = HorizonAnalyzer(confidence_level=0.95, n_bootstrap=50)

        # 1. Performance Slicing Analysis
        slice_results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            error_metric="mae",
            include_bootstrap=True,
        )

        # Validate slicing results
        assert "vessel_type" in slice_results
        assert "traffic_density" in slice_results
        assert "distance_to_port" in slice_results

        # Check that we have meaningful results
        vessel_results = slice_results["vessel_type"]
        assert "cargo" in vessel_results
        assert vessel_results["cargo"].n_samples > 0

        # 2. Horizon Analysis
        horizon_results = horizon_analyzer.analyze_horizon_errors(
            predictions=predictions,
            targets=targets,
            error_metric="mae",
            include_bootstrap=True,
        )

        # Validate horizon analysis
        assert len(horizon_results.points) == predictions.shape[1]
        assert horizon_results.degradation_rate != 0
        assert len(horizon_results.critical_steps) >= 0

        # 3. Failure Mining on worst performers
        # Compute per-sample errors for mining
        sample_errors = np.mean(np.linalg.norm(predictions - targets, axis=-1), axis=1)
        feature_matrix = np.column_stack(
            [
                metadata["vessel_type"],
                metadata["distance_to_port_km"],
                metadata["vessel_count_5km"],
            ]
        )

        mining_results = miner.mine_failures(
            errors=sample_errors,
            features=feature_matrix,
            metadata=metadata,
            predictions=predictions,
            targets=targets,
        )

        # Validate mining results
        assert len(mining_results.worst_cases) == 50
        assert len(mining_results.clusters) <= 4
        assert (
            mining_results.silhouette_score >= -1
            and mining_results.silhouette_score <= 1
        )

        # 4. Integration verification - cross-reference results
        # Check that worst cases from mining have high errors
        worst_case_errors = [
            case.error_magnitude for case in mining_results.worst_cases
        ]
        assert all(
            error >= np.percentile(sample_errors, 75) for error in worst_case_errors
        )

        # Check that horizon degradation is captured properly
        horizon_errors = [point.mean_error for point in horizon_results.points]
        assert (
            horizon_errors[-1] > horizon_errors[0]
        )  # Error should increase with horizon

        # Generate comprehensive summary
        slice_summary = slicer.get_summary_statistics(slice_results)
        case_cards = miner.generate_case_cards(mining_results.worst_cases[:10])

        # Validate summary outputs
        assert isinstance(slice_summary, pd.DataFrame)
        assert len(slice_summary) > 0
        assert len(case_cards) == 10
        assert all("case_id" in card for card in case_cards)

    def test_cross_analysis_consistency(self, comprehensive_dataset):
        """Test consistency between different analysis methods."""
        predictions, targets, metadata = comprehensive_dataset

        slicer = ErrorSlicer()
        horizon_analyzer = HorizonAnalyzer()

        # Compute errors using both methods
        _ = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            error_metric="mae",
            include_bootstrap=False,
        )

        horizon_slice_results = slicer.analyze_horizon_steps(
            predictions=predictions,
            targets=targets,
            error_metric="mae",
            include_bootstrap=False,
        )

        horizon_analysis = horizon_analyzer.analyze_horizon_errors(
            predictions=predictions,
            targets=targets,
            error_metric="mae",
            include_bootstrap=False,
        )

        # Check consistency between horizon methods
        horizon_slice_step1 = horizon_slice_results["step_1"].mean_error
        horizon_analysis_step1 = horizon_analysis.points[0].mean_error

        # Should be approximately equal (within numerical precision)
        assert abs(horizon_slice_step1 - horizon_analysis_step1) < 0.001

    def test_maritime_specific_insights(self, comprehensive_dataset):
        """Test maritime domain-specific insights generation."""
        predictions, targets, metadata = comprehensive_dataset

        # Focus on vessel type analysis
        slicer = ErrorSlicer()
        miner = FailureMiner(k_worst=30, n_clusters=3)

        # Analyze performance by vessel type
        vessel_performance = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            slicing_dimensions=["vessel_type"],
            error_metric="mae",
            include_bootstrap=False,
        )

        vessel_results = vessel_performance["vessel_type"]

        # Verify maritime vessel types are properly categorized
        expected_types = ["cargo", "tanker", "fishing", "passenger", "other"]
        for vtype in expected_types:
            assert vtype in vessel_results

        # Mine failures with maritime context
        sample_errors = np.mean(np.linalg.norm(predictions - targets, axis=-1), axis=1)
        feature_matrix = np.column_stack(
            [
                metadata["vessel_type"],
                metadata["distance_to_port_km"],
                metadata["vessel_count_5km"],
            ]
        )

        mining_results = miner.mine_failures(
            errors=sample_errors, features=feature_matrix, metadata=metadata
        )

        # Generate case cards with maritime recommendations
        case_cards = miner.generate_case_cards(mining_results.worst_cases[:5])

        # Verify maritime-specific recommendations are generated
        all_recommendations = []
        for card in case_cards:
            all_recommendations.extend(card["recommendations"])

        # Should contain maritime-specific advice
        maritime_keywords = [
            "vessel",
            "cargo",
            "tanker",
            "fishing",
            "port",
            "maneuvering",
        ]
        recommendation_text = " ".join(all_recommendations).lower()
        assert any(keyword in recommendation_text for keyword in maritime_keywords)

    def test_statistical_rigor_integration(self, comprehensive_dataset):
        """Test statistical rigor across the integrated pipeline."""
        predictions, targets, metadata = comprehensive_dataset

        # Test with bootstrap enabled across all components
        slicer = ErrorSlicer(confidence_level=0.95, n_bootstrap=30)
        horizon_analyzer = HorizonAnalyzer(confidence_level=0.95, n_bootstrap=30)

        # Slice analysis with bootstrap
        slice_results = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            include_bootstrap=True,
        )

        # Horizon analysis with bootstrap
        horizon_results = horizon_analyzer.analyze_horizon_errors(
            predictions=predictions, targets=targets, include_bootstrap=True
        )

        # Verify bootstrap confidence intervals
        vessel_results = slice_results["vessel_type"]
        for _vessel_type, result in vessel_results.items():
            if result.n_samples >= 10:  # Sufficient samples for CI
                assert result.bootstrap_ci is not None
                ci_lower, ci_upper = result.bootstrap_ci.confidence_interval
                assert ci_lower <= result.mean_error <= ci_upper
                assert ci_lower < ci_upper  # Non-degenerate interval

        # Verify horizon bootstrap CIs
        for point in horizon_results.points:
            if point.bootstrap_ci is not None:
                ci_lower, ci_upper = point.bootstrap_ci.confidence_interval
                assert ci_lower <= point.mean_error <= ci_upper

    def test_error_handling_integration(self, comprehensive_dataset):
        """Test error handling across integrated components."""
        predictions, targets, metadata = comprehensive_dataset

        # Test with missing metadata
        incomplete_metadata = {
            "vessel_type": metadata["vessel_type"]
            # Missing traffic density and port distance
        }

        slicer = ErrorSlicer()

        # Should handle missing metadata gracefully
        with pytest.warns(UserWarning):
            slice_results = slicer.slice_errors(
                predictions=predictions, targets=targets, metadata=incomplete_metadata
            )

        # Should still get vessel_type results
        assert "vessel_type" in slice_results

        # Test with mismatched shapes
        bad_predictions = predictions[:50]  # Different number of samples

        with pytest.raises(ValueError, match="Predictions shape"):
            slicer.slice_errors(bad_predictions, targets, metadata)

    def test_performance_scalability(self, comprehensive_dataset):
        """Test performance with larger datasets."""
        predictions, targets, metadata = comprehensive_dataset

        # Create larger dataset
        large_predictions = np.tile(predictions, (5, 1, 1))  # 1000 samples
        large_targets = np.tile(targets, (5, 1, 1))

        # Extend metadata
        large_metadata = {}
        for key, values in metadata.items():
            if isinstance(values, np.ndarray):
                large_metadata[key] = np.tile(values, 5)
            elif isinstance(values, list):
                large_metadata[key] = values * 5
            else:
                large_metadata[key] = values

        # Test that analysis completes in reasonable time
        slicer = ErrorSlicer(n_bootstrap=10)  # Reduced bootstrap for speed
        miner = FailureMiner(k_worst=100, n_clusters=5)

        # Should complete without timeout
        slice_results = slicer.slice_errors(
            predictions=large_predictions,
            targets=large_targets,
            metadata=large_metadata,
            include_bootstrap=False,  # Skip bootstrap for speed
        )

        assert len(slice_results) > 0

        # Test mining scalability
        sample_errors = np.mean(
            np.linalg.norm(large_predictions - large_targets, axis=-1), axis=1
        )
        feature_matrix = np.column_stack(
            [
                large_metadata["vessel_type"],
                large_metadata["distance_to_port_km"],
                large_metadata["vessel_count_5km"],
            ]
        )

        mining_results = miner.mine_failures(
            errors=sample_errors, features=feature_matrix, metadata=large_metadata
        )

        assert len(mining_results.worst_cases) == 100
        assert mining_results.silhouette_score >= -1


class TestM21ErrorAnalysisWorkflows:
    """Test realistic maritime error analysis workflows."""

    def test_post_training_evaluation_workflow(self):
        """Test typical post-training model evaluation workflow."""
        # Simulate model evaluation results
        np.random.seed(123)
        n_samples = 150
        horizon = 6

        # Create model predictions with realistic error patterns
        predictions = np.random.randn(n_samples, horizon, 2) * 0.03
        targets = np.random.randn(n_samples, horizon, 2) * 0.03

        # Add systematic errors for specific vessel types
        vessel_types = np.random.choice([70, 80, 30], size=n_samples)

        # Tankers (80) have higher errors in later horizon steps
        tanker_mask = vessel_types == 80
        predictions[tanker_mask, 3:, :] += 0.05

        metadata = {
            "vessel_type": vessel_types,
            "distance_to_port_km": np.random.uniform(1, 40, size=n_samples),
            "vessel_count_5km": np.random.randint(0, 15, size=n_samples),
            "model_name": ["TestModel"] * n_samples,
            "test_split": ["val"] * n_samples,
        }

        # Run complete evaluation workflow
        slicer = ErrorSlicer()
        miner = FailureMiner(k_worst=30)
        horizon_analyzer = HorizonAnalyzer()

        # 1. Overall performance slicing
        performance_slices = slicer.slice_errors(
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            include_bootstrap=True,
        )

        # 2. Horizon degradation analysis
        horizon_analysis = horizon_analyzer.analyze_horizon_errors(
            predictions=predictions, targets=targets, include_bootstrap=True
        )

        # 3. Failure mining for model improvement insights
        sample_errors = np.mean(np.linalg.norm(predictions - targets, axis=-1), axis=1)
        feature_matrix = np.column_stack(
            [
                metadata["vessel_type"],
                metadata["distance_to_port_km"],
                metadata["vessel_count_5km"],
            ]
        )

        failure_analysis = miner.mine_failures(
            errors=sample_errors,
            features=feature_matrix,
            metadata=metadata,
            predictions=predictions,
            targets=targets,
        )

        # Verify workflow outputs provide actionable insights
        assert len(performance_slices) >= 3  # Multiple slicing dimensions
        assert horizon_analysis.degradation_rate != 0  # Detectable degradation
        assert len(failure_analysis.worst_cases) == 30

        # Generate actionable reports
        summary_df = slicer.get_summary_statistics(performance_slices)
        case_cards = miner.generate_case_cards(failure_analysis.worst_cases[:10])

        assert len(summary_df) > 0
        assert len(case_cards) == 10

        # Verify tanker performance issue was detected
        vessel_performance = performance_slices["vessel_type"]
        if (
            "tanker" in vessel_performance
            and vessel_performance["tanker"].n_samples > 0
        ):
            tanker_error = vessel_performance["tanker"].mean_error
            cargo_error = (
                vessel_performance["cargo"].mean_error
                if "cargo" in vessel_performance
                else 0
            )

            # Tanker error should be higher due to our systematic bias
            if vessel_performance["cargo"].n_samples > 0:
                assert tanker_error >= cargo_error * 0.8  # Allow some variance

    @patch("matplotlib.pyplot.show")
    def test_visualization_integration(self, mock_show):
        """Test integration with visualization components."""
        # Create test data with clear horizon trends
        predictions = np.random.randn(100, 5, 2) * 0.02
        targets = np.random.randn(100, 5, 2) * 0.02

        # Add increasing error with horizon
        for h in range(5):
            predictions[:, h, :] += h * 0.01

        _ = {
            "vessel_type": np.random.choice([70, 80], size=100),
            "distance_to_port_km": np.random.uniform(5, 30, size=100),
            "vessel_count_5km": np.random.randint(2, 12, size=100),
        }

        # Test horizon plotting
        horizon_analyzer = HorizonAnalyzer()
        horizon_results = horizon_analyzer.analyze_horizon_errors(
            predictions=predictions, targets=targets, include_bootstrap=True
        )

        # Test plotting functionality
        fig = horizon_analyzer.plot_horizon_curve(
            horizon_results, title="Test Horizon Analysis", show_confidence_bands=True
        )

        assert fig is not None
        # Check that we have axes and they contain plots
        axes = fig.get_axes()
        assert len(axes) > 0

        # Check that lines were plotted
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) > 0  # Should have at least the main error curve
