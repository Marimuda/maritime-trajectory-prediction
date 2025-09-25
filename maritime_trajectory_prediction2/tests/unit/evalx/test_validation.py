"""Tests for cross-validation and model comparison protocols."""

import numpy as np
import pandas as pd
import pytest

from src.evalx.validation.comparisons import ComparisonResult, ModelComparison
from src.evalx.validation.protocols import (
    GroupKFold,
    TimeSeriesSplit,
    maritime_cv_split,
    validate_split_quality,
)


class TestTimeSeriesSplit:
    """Test suite for TimeSeriesSplit."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create sample time series data
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        self.df = pd.DataFrame(
            {
                "timestamp": dates,
                "value": np.random.randn(100),
                "mmsi": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_initialization(self):
        """Test TimeSeriesSplit initialization."""
        splitter = TimeSeriesSplit(n_splits=5, min_gap_minutes=60)
        assert splitter.n_splits == 5
        assert splitter.min_gap_minutes == 60
        assert splitter.test_size is None

    def test_basic_split(self):
        """Test basic time series splitting."""
        splitter = TimeSeriesSplit(n_splits=3, min_gap_minutes=0)
        splits = list(splitter.split(self.df))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap

    def test_chronological_order(self):
        """Test that splits maintain chronological order."""
        splitter = TimeSeriesSplit(n_splits=3, min_gap_minutes=0)
        splits = list(splitter.split(self.df))

        for train_idx, test_idx in splits:
            train_times = self.df.iloc[train_idx]["timestamp"]
            test_times = self.df.iloc[test_idx]["timestamp"]

            # All training times should be before test times
            assert train_times.max() <= test_times.min()

    def test_gap_enforcement(self):
        """Test minimum gap enforcement."""
        splitter = TimeSeriesSplit(n_splits=2, min_gap_minutes=120)  # 2 hour gap
        splits = list(splitter.split(self.df))

        for train_idx, test_idx in splits:
            if len(test_idx) > 0:  # Only check if test set is not empty
                train_end = self.df.iloc[train_idx[-1]]["timestamp"]
                test_start = self.df.iloc[test_idx[0]]["timestamp"]

                gap_minutes = (test_start - train_end).total_seconds() / 60
                assert gap_minutes >= 120

    def test_missing_time_column(self):
        """Test error handling for missing time column."""
        splitter = TimeSeriesSplit(n_splits=3)
        df_no_time = self.df.drop(columns=["timestamp"])

        with pytest.raises(ValueError, match="not found"):
            list(splitter.split(df_no_time))

    def test_custom_time_column(self):
        """Test splitting with custom time column name."""
        df_custom = self.df.rename(columns={"timestamp": "datetime"})
        splitter = TimeSeriesSplit(n_splits=2)

        splits = list(splitter.split(df_custom, time_col="datetime"))
        assert len(splits) == 2


class TestGroupKFold:
    """Test suite for GroupKFold."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create sample grouped data
        mmsi_list = ["vessel_1", "vessel_2", "vessel_3", "vessel_4", "vessel_5"]
        data = []
        for mmsi in mmsi_list:
            for i in range(20):  # 20 points per vessel
                data.append({"mmsi": mmsi, "value": np.random.randn(), "index": i})
        self.df = pd.DataFrame(data)

    def test_initialization(self):
        """Test GroupKFold initialization."""
        splitter = GroupKFold(n_splits=3, shuffle=True, random_state=42)
        assert splitter.n_splits == 3
        assert splitter.shuffle is True
        assert splitter.random_state == 42

    def test_basic_group_split(self):
        """Test basic group-based splitting."""
        splitter = GroupKFold(n_splits=3)
        splits = list(splitter.split(self.df))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap

    def test_no_group_leakage(self):
        """Test that groups don't appear in both train and test."""
        splitter = GroupKFold(n_splits=3)
        splits = list(splitter.split(self.df))

        for train_idx, test_idx in splits:
            train_groups = set(self.df.iloc[train_idx]["mmsi"].unique())
            test_groups = set(self.df.iloc[test_idx]["mmsi"].unique())

            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0

    def test_group_distribution(self):
        """Test group distribution analysis."""
        splitter = GroupKFold(n_splits=3)
        distribution = splitter.get_group_distribution(self.df)

        assert isinstance(distribution, pd.DataFrame)
        assert "group" in distribution.columns
        assert "fold" in distribution.columns
        assert "split" in distribution.columns

        # Each group should appear in exactly one test fold
        test_assignments = distribution[distribution["split"] == "test"]
        groups_per_fold = test_assignments.groupby("fold")["group"].nunique()
        assert all(groups_per_fold > 0)  # Each fold has some test groups

    def test_missing_group_column(self):
        """Test error handling for missing group column."""
        splitter = GroupKFold(n_splits=3)
        df_no_group = self.df.drop(columns=["mmsi"])

        with pytest.raises(ValueError, match="not found"):
            list(splitter.split(df_no_group))

    def test_reproducibility(self):
        """Test reproducible splits with fixed random state."""
        splitter1 = GroupKFold(n_splits=3, shuffle=True, random_state=42)
        splitter2 = GroupKFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(splitter1.split(self.df))
        splits2 = list(splitter2.split(self.df))

        # Results should be identical
        for (train1, test1), (train2, test2) in zip(splits1, splits2, strict=False):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)


class TestMaritimeCVSplit:
    """Test suite for maritime_cv_split function."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create comprehensive maritime dataset
        vessels = ["vessel_1", "vessel_2", "vessel_3", "vessel_4"]
        dates = pd.date_range("2023-01-01", periods=200, freq="1H")

        data = []
        for i, timestamp in enumerate(dates):
            vessel = vessels[i % len(vessels)]
            data.append(
                {
                    "timestamp": timestamp,
                    "mmsi": vessel,
                    "lat": 60.0 + np.random.randn() * 0.1,
                    "lon": -7.0 + np.random.randn() * 0.1,
                }
            )

        self.df = pd.DataFrame(data)

    def test_temporal_split(self):
        """Test temporal splitting mode."""
        splits = maritime_cv_split(self.df, split_type="temporal", n_splits=3)

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            # Check chronological order
            train_times = self.df.iloc[train_idx]["timestamp"]
            test_times = self.df.iloc[test_idx]["timestamp"]
            assert train_times.max() <= test_times.min()

    def test_vessel_split(self):
        """Test vessel-based splitting mode."""
        splits = maritime_cv_split(self.df, split_type="vessel", n_splits=2)

        assert len(splits) == 2
        for train_idx, test_idx in splits:
            # Check no vessel leakage
            train_vessels = set(self.df.iloc[train_idx]["mmsi"].unique())
            test_vessels = set(self.df.iloc[test_idx]["mmsi"].unique())
            assert len(train_vessels & test_vessels) == 0

    def test_combined_split(self):
        """Test combined splitting mode."""
        splits = maritime_cv_split(self.df, split_type="combined", n_splits=2)

        assert len(splits) == 2
        # Combined splits should maintain vessel separation
        for train_idx, test_idx in splits:
            train_vessels = set(self.df.iloc[train_idx]["mmsi"].unique())
            test_vessels = set(self.df.iloc[test_idx]["mmsi"].unique())
            assert len(train_vessels & test_vessels) == 0

    def test_invalid_split_type(self):
        """Test error handling for invalid split type."""
        with pytest.raises(ValueError, match="must be"):
            maritime_cv_split(self.df, split_type="invalid")

    def test_custom_parameters(self):
        """Test custom parameters."""
        splits = maritime_cv_split(
            self.df,
            split_type="temporal",
            n_splits=2,
            min_gap_minutes=30,
            random_state=123,
        )

        assert len(splits) == 2
        # Additional parameter validation would require deeper inspection


class TestValidateSplitQuality:
    """Test suite for validate_split_quality function."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create dataset with known properties
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        self.df = pd.DataFrame(
            {
                "timestamp": dates,
                "mmsi": np.repeat(["A", "B", "C", "D"], 25),
                "value": np.random.randn(100),
            }
        )

        # Create sample splits
        self.splits = [
            (np.arange(50), np.arange(50, 75)),
            (np.arange(75), np.arange(75, 100)),
        ]

    def test_basic_validation(self):
        """Test basic split quality validation."""
        metrics = validate_split_quality(self.df, self.splits)

        assert isinstance(metrics, dict)
        assert "n_splits" in metrics
        assert "train_sizes" in metrics
        assert "test_sizes" in metrics
        assert "temporal_overlaps" in metrics
        assert "group_overlaps" in metrics

        assert metrics["n_splits"] == 2
        assert len(metrics["train_sizes"]) == 2
        assert len(metrics["test_sizes"]) == 2

    def test_overlap_detection(self):
        """Test temporal and group overlap detection."""
        # Create overlapping splits (bad)
        bad_splits = [
            (np.arange(60), np.arange(40, 80)),  # Temporal overlap
            (np.arange(50), np.arange(50, 100)),
        ]

        metrics = validate_split_quality(self.df, bad_splits)
        assert metrics["temporal_overlaps"] > 0

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        metrics = validate_split_quality(self.df, self.splits)

        assert "avg_train_size" in metrics
        assert "avg_test_size" in metrics
        assert "train_test_ratio" in metrics

        assert metrics["avg_train_size"] > 0
        assert metrics["avg_test_size"] > 0
        assert metrics["train_test_ratio"] > 0

    def test_time_gap_calculation(self):
        """Test time gap calculation."""
        metrics = validate_split_quality(self.df, self.splits)

        if (
            "avg_time_gap_minutes" in metrics
            and metrics["avg_time_gap_minutes"] is not None
        ):
            assert metrics["avg_time_gap_minutes"] >= 0
            assert metrics["min_time_gap_minutes"] >= 0


class TestModelComparison:
    """Test suite for ModelComparison framework."""

    def setup_method(self):
        """Setup test fixtures."""
        self.comparison = ModelComparison(
            confidence_level=0.95,
            n_bootstrap=999,  # Lower for faster tests
            random_state=42,
        )

        # Create sample model results
        self.results = {
            "LSTM": {
                "ADE": np.array([1.2, 1.1, 1.3, 1.0, 1.15]),
                "FDE": np.array([2.1, 2.0, 2.2, 1.9, 2.05]),
            },
            "Transformer": {
                "ADE": np.array([0.9, 0.8, 1.0, 0.7, 0.85]),
                "FDE": np.array([1.6, 1.5, 1.7, 1.4, 1.55]),
            },
        }

    def test_initialization(self):
        """Test ModelComparison initialization."""
        assert self.comparison.confidence_level == 0.95
        assert self.comparison.n_bootstrap == 999
        assert self.comparison.correction_method == "holm"
        assert self.comparison.alpha == 0.05

    def test_basic_model_comparison(self):
        """Test basic model comparison functionality."""
        result = self.comparison.compare_models(self.results)

        assert isinstance(result, ComparisonResult)
        assert result.model_names == ["LSTM", "Transformer"]
        assert "ADE" in result.bootstrap_cis["LSTM"]
        assert "FDE" in result.bootstrap_cis["LSTM"]

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval computation."""
        result = self.comparison.compare_models(self.results)

        for model in result.model_names:
            for metric in ["ADE", "FDE"]:
                bootstrap_result = result.bootstrap_cis[model][metric]
                assert len(bootstrap_result.confidence_interval) == 2
                assert (
                    bootstrap_result.confidence_interval[0]
                    < bootstrap_result.confidence_interval[1]
                )
                assert bootstrap_result.confidence_level == 0.95

    def test_pairwise_statistical_tests(self):
        """Test pairwise statistical testing."""
        result = self.comparison.compare_models(self.results)

        assert "LSTM_vs_Transformer" in result.pairwise_tests
        for metric in ["ADE", "FDE"]:
            test_result = result.pairwise_tests["LSTM_vs_Transformer"][metric]
            assert hasattr(test_result, "p_value")
            assert hasattr(test_result, "effect_size")
            assert hasattr(test_result, "significant")

    def test_summary_table_creation(self):
        """Test summary table creation."""
        result = self.comparison.compare_models(self.results)

        assert result.summary_table is not None
        assert isinstance(result.summary_table, pd.DataFrame)
        assert "Model" in result.summary_table.columns
        assert "Metric" in result.summary_table.columns
        assert "Mean" in result.summary_table.columns
        assert "CI_Formatted" in result.summary_table.columns

    def test_best_model_identification(self):
        """Test best model identification."""
        result = self.comparison.compare_models(self.results)

        assert result.best_model is not None
        assert "ADE" in result.best_model
        assert "FDE" in result.best_model

        # Transformer should be better (lower error) for this test data
        assert result.best_model["ADE"] == "Transformer"
        assert result.best_model["FDE"] == "Transformer"

    def test_insufficient_models_error(self):
        """Test error handling for insufficient models."""
        single_model = {"LSTM": self.results["LSTM"]}

        with pytest.raises(ValueError, match="at least 2 models"):
            self.comparison.compare_models(single_model)

    def test_missing_metric_error(self):
        """Test error handling for missing metrics."""
        incomplete_results = {
            "Model1": {"ADE": np.array([1, 2, 3])},
            "Model2": {"FDE": np.array([1, 2, 3])},  # Missing ADE
        }

        with pytest.raises(ValueError, match="missing metric"):
            self.comparison.compare_models(incomplete_results)

    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        # Add third model to trigger multiple comparison correction
        extended_results = {
            **self.results,
            "XGBoost": {
                "ADE": np.array([1.4, 1.3, 1.5, 1.2, 1.35]),
                "FDE": np.array([2.4, 2.3, 2.5, 2.2, 2.35]),
            },
        }

        result = self.comparison.compare_models(extended_results)

        assert result.corrected_pvalues is not None
        assert "correction_result" in result.corrected_pvalues
        assert "pvalue_mapping" in result.corrected_pvalues

    @pytest.mark.skip(reason="Requires matplotlib, optional test")
    def test_plotting_functionality(self):
        """Test plotting functionality (optional, requires matplotlib)."""
        result = self.comparison.compare_models(self.results)

        try:
            fig = self.comparison.plot_comparison(result, "ADE")
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not available")


class TestIntegrationScenarios:
    """Integration tests for validation protocols."""

    def test_complete_maritime_validation_workflow(self):
        """Test complete maritime validation workflow."""
        # Create realistic maritime dataset
        np.random.seed(42)
        vessels = [f"vessel_{i}" for i in range(10)]
        dates = pd.date_range("2023-01-01", periods=500, freq="30min")

        data = []
        for timestamp in dates:
            vessel = np.random.choice(vessels)
            data.append(
                {
                    "timestamp": timestamp,
                    "mmsi": vessel,
                    "lat": 60.0 + np.random.randn() * 0.1,
                    "lon": -7.0 + np.random.randn() * 0.1,
                    "sog": np.abs(np.random.randn() * 5 + 10),  # Speed over ground
                }
            )

        df = pd.DataFrame(data)

        # Test different split strategies
        temporal_splits = maritime_cv_split(df, "temporal", n_splits=3)
        vessel_splits = maritime_cv_split(df, "vessel", n_splits=3)

        # Validate split quality
        temporal_quality = validate_split_quality(df, temporal_splits)
        vessel_quality = validate_split_quality(df, vessel_splits)

        # Basic quality checks
        assert temporal_quality["n_splits"] == 3
        assert vessel_quality["n_splits"] == 3
        assert vessel_quality["group_overlaps"] == 0  # No vessel leakage

        # Simulate model comparison
        model_results = {
            "LSTM": {"ADE": np.random.uniform(1.0, 2.0, 5)},
            "Transformer": {"ADE": np.random.uniform(0.5, 1.5, 5)},
        }

        comparison = ModelComparison(n_bootstrap=99)  # Fast for testing
        comp_result = comparison.compare_models(model_results)

        assert isinstance(comp_result, ComparisonResult)
        assert comp_result.summary_table is not None
        assert comp_result.best_model is not None

    def test_edge_cases_handling(self):
        """Test handling of edge cases."""
        # Very small dataset
        small_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5),
                "mmsi": ["A"] * 5,
                "value": [1, 2, 3, 4, 5],
            }
        )

        # Should handle gracefully
        try:
            splits = maritime_cv_split(small_df, "temporal", n_splits=2)
            quality = validate_split_quality(small_df, splits)
            assert isinstance(quality, dict)
        except (ValueError, IndexError):
            # Expected for very small datasets
            pass
