"""Tests for statistical significance tests."""

import numpy as np
import pytest

from src.evalx.stats.tests import (
    StatTestResult,
    cliffs_delta,
    mcnemar_test,
    paired_t_test,
    wilcoxon_test,
)


class TestPairedTTest:
    """Test suite for paired t-test."""

    def test_basic_paired_t_test(self):
        """Test basic paired t-test functionality."""
        group_a = np.array([1.2, 1.1, 1.3, 1.0, 1.2])
        group_b = np.array([1.0, 0.9, 1.1, 0.8, 1.0])

        result = paired_t_test(group_a, group_b)

        assert isinstance(result, StatTestResult)
        assert result.test_name == "Paired t-test"
        assert isinstance(result.statistic, float | np.floating)
        assert isinstance(result.p_value, float | np.floating)
        assert isinstance(result.effect_size, float | np.floating)
        assert result.effect_size_interpretation in [
            "negligible",
            "small",
            "medium",
            "large",
        ]
        assert isinstance(result.significant, bool)
        assert result.alpha == 0.05

    def test_paired_t_test_different_alternatives(self):
        """Test paired t-test with different alternative hypotheses."""
        group_a = np.array([1.5, 1.4, 1.6, 1.3, 1.5])
        group_b = np.array([1.0, 0.9, 1.1, 0.8, 1.0])

        # Two-sided test
        result_two = paired_t_test(group_a, group_b, alternative="two-sided")
        assert result_two.additional_info["alternative"] == "two-sided"

        # Greater test (A > B)
        result_greater = paired_t_test(group_a, group_b, alternative="greater")
        assert result_greater.additional_info["alternative"] == "greater"

        # Less test (A < B)
        result_less = paired_t_test(group_a, group_b, alternative="less")
        assert result_less.additional_info["alternative"] == "less"

        # Greater should have smaller p-value for this data
        assert result_greater.p_value <= result_two.p_value

    def test_paired_t_test_unequal_lengths(self):
        """Test that unequal length groups raise ValueError."""
        group_a = np.array([1, 2, 3])
        group_b = np.array([1, 2])

        with pytest.raises(ValueError, match="equal length"):
            paired_t_test(group_a, group_b)

    def test_paired_t_test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        # Small effect
        group_a = np.array([1.00, 1.02, 0.98, 1.03, 0.97])
        group_b = np.array([1.00, 1.00, 1.00, 1.00, 1.00])
        result = paired_t_test(group_a, group_b)
        assert result.effect_size_interpretation in ["negligible", "small"]

        # Large effect
        group_a = np.array([2.0, 2.1, 2.0, 2.1, 2.0])
        group_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = paired_t_test(group_a, group_b)
        assert abs(result.effect_size) > 0.8  # Should be large
        assert result.effect_size_interpretation == "large"

    def test_paired_t_test_custom_alpha(self):
        """Test paired t-test with custom alpha level."""
        group_a = np.array([1.2, 1.1, 1.3, 1.0, 1.2])
        group_b = np.array([1.0, 0.9, 1.1, 0.8, 1.0])

        result = paired_t_test(group_a, group_b, alpha=0.01)
        assert result.alpha == 0.01


class TestWilcoxonTest:
    """Test suite for Wilcoxon signed-rank test."""

    def test_basic_wilcoxon_test(self):
        """Test basic Wilcoxon test functionality."""
        group_a = np.array([1.2, 1.1, 1.3, 1.0, 1.2, 1.4, 1.1])
        group_b = np.array([1.0, 0.9, 1.1, 0.8, 1.0, 1.2, 0.9])

        result = wilcoxon_test(group_a, group_b)

        assert isinstance(result, StatTestResult)
        assert result.test_name == "Wilcoxon signed-rank test"
        assert isinstance(result.statistic, float | np.floating)
        assert isinstance(result.p_value, float | np.floating)
        assert isinstance(result.significant, bool)

    def test_wilcoxon_test_different_alternatives(self):
        """Test Wilcoxon test with different alternatives."""
        group_a = np.array([2, 3, 1, 4, 2, 3, 2])
        group_b = np.array([1, 2, 1, 2, 1, 2, 1])

        result_two = wilcoxon_test(group_a, group_b, alternative="two-sided")
        result_greater = wilcoxon_test(group_a, group_b, alternative="greater")

        assert result_two.additional_info["alternative"] == "two-sided"
        assert result_greater.additional_info["alternative"] == "greater"

    def test_wilcoxon_test_unequal_lengths(self):
        """Test that unequal length groups raise ValueError."""
        group_a = np.array([1, 2, 3])
        group_b = np.array([1, 2])

        with pytest.raises(ValueError, match="equal length"):
            wilcoxon_test(group_a, group_b)

    def test_wilcoxon_test_identical_groups(self):
        """Test Wilcoxon test with identical groups."""
        group_a = np.array([1, 2, 3, 4, 5])
        group_b = np.array([1, 2, 3, 4, 5])

        # This may raise a warning or return a result, both are acceptable
        result = wilcoxon_test(group_a, group_b)

        # P-value should be high (not significant)
        if not np.isnan(result.p_value):
            assert result.p_value > 0.05

    def test_wilcoxon_effect_size(self):
        """Test effect size calculation for Wilcoxon test."""
        # Create data with sufficient size for effect size calculation
        group_a = np.random.normal(1.5, 0.5, 25)
        group_b = np.random.normal(1.0, 0.5, 25)
        np.random.seed(42)

        result = wilcoxon_test(group_a, group_b)

        if result.effect_size is not None:
            assert isinstance(result.effect_size, float | np.floating)
            assert result.effect_size_interpretation in [
                "negligible",
                "small",
                "medium",
                "large",
            ]


class TestCliffssDelta:
    """Test suite for Cliff's delta effect size."""

    def test_basic_cliffs_delta(self):
        """Test basic Cliff's delta functionality."""
        group_a = np.array([3, 4, 5, 6, 7])
        group_b = np.array([1, 2, 3, 4, 5])

        result = cliffs_delta(group_a, group_b)

        assert isinstance(result, StatTestResult)
        assert result.test_name == "Cliff's delta"
        assert isinstance(result.statistic, float | np.floating)
        assert result.p_value is None  # Cliff's delta is not a significance test
        assert isinstance(result.effect_size, float | np.floating)
        assert result.effect_size_interpretation in [
            "negligible",
            "small",
            "medium",
            "large",
        ]

        # For this data, A > B, so delta should be positive
        assert result.effect_size > 0

    def test_cliffs_delta_identical_groups(self):
        """Test Cliff's delta with identical groups."""
        group_a = np.array([1, 2, 3, 4, 5])
        group_b = np.array([1, 2, 3, 4, 5])

        result = cliffs_delta(group_a, group_b)

        # Delta should be 0 for identical groups
        assert abs(result.effect_size) < 1e-10
        assert result.effect_size_interpretation == "negligible"

    def test_cliffs_delta_interpretation_boundaries(self):
        """Test interpretation boundaries for Cliff's delta."""
        # Create groups with known Cliff's delta values
        group_b = np.array([1, 2, 3, 4, 5])

        # Test negligible effect (delta < 0.147)
        group_a_negligible = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = cliffs_delta(group_a_negligible, group_b)
        assert result.effect_size_interpretation in ["negligible", "small"]

        # Test large effect
        group_a_large = np.array([10, 11, 12, 13, 14])
        result = cliffs_delta(group_a_large, group_b)
        assert result.effect_size_interpretation in ["medium", "large"]
        assert result.effect_size > 0.474

    def test_cliffs_delta_negative_effect(self):
        """Test Cliff's delta with negative effect (A < B)."""
        group_a = np.array([1, 2, 3])
        group_b = np.array([4, 5, 6])

        result = cliffs_delta(group_a, group_b)

        # Delta should be negative since A < B
        assert result.effect_size < 0

    def test_cliffs_delta_additional_info(self):
        """Test additional info in Cliff's delta result."""
        group_a = np.array([3, 4, 5])
        group_b = np.array([1, 2])

        result = cliffs_delta(group_a, group_b)

        assert "n1" in result.additional_info
        assert "n2" in result.additional_info
        assert "dominance_count" in result.additional_info
        assert result.additional_info["n1"] == 3
        assert result.additional_info["n2"] == 2


class TestMcNemarTest:
    """Test suite for McNemar test."""

    def test_basic_mcnemar_test(self):
        """Test basic McNemar test functionality."""
        # Contingency table: [[both_correct, A_correct_B_wrong], [A_wrong_B_correct, both_wrong]]
        table = np.array([[50, 5], [10, 35]])

        result = mcnemar_test(table)

        assert isinstance(result, StatTestResult)
        assert result.test_name == "McNemar test"
        assert isinstance(result.statistic, float | np.floating)
        assert isinstance(result.p_value, float | np.floating)
        assert isinstance(result.significant, bool)

    def test_mcnemar_test_small_sample(self):
        """Test McNemar test with small sample (uses exact test)."""
        # Small sample - should use exact binomial test
        table = np.array([[10, 2], [3, 5]])

        result = mcnemar_test(table)

        assert "test_type" in result.additional_info
        assert result.additional_info["test_type"] == "exact"
        assert result.additional_info["discordant_pairs"] == 5  # 2 + 3

    def test_mcnemar_test_large_sample(self):
        """Test McNemar test with large sample (uses chi-square)."""
        # Large sample - should use chi-square approximation
        table = np.array([[100, 15], [20, 65]])

        result = mcnemar_test(table)

        assert "test_type" in result.additional_info
        assert result.additional_info["test_type"] == "chi-square"
        assert result.additional_info["discordant_pairs"] == 35  # 15 + 20

    def test_mcnemar_test_invalid_table(self):
        """Test McNemar test with invalid contingency table."""
        # Wrong shape
        table = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="2x2"):
            mcnemar_test(table)

    def test_mcnemar_test_no_difference(self):
        """Test McNemar test when there's no difference between models."""
        # Equal discordant pairs
        table = np.array([[50, 10], [10, 30]])

        result = mcnemar_test(table)

        # Should not be significant
        assert result.p_value > 0.05
        assert not result.significant

    def test_mcnemar_test_custom_alpha(self):
        """Test McNemar test with custom alpha level."""
        table = np.array([[50, 5], [15, 30]])

        result = mcnemar_test(table, alpha=0.01)
        assert result.alpha == 0.01

    def test_mcnemar_test_additional_info(self):
        """Test additional information in McNemar test result."""
        table = np.array([[40, 8], [12, 20]])

        result = mcnemar_test(table)

        assert "model1_better" in result.additional_info
        assert "model2_better" in result.additional_info
        assert "contingency_table" in result.additional_info

        assert result.additional_info["model1_better"] == 8
        assert result.additional_info["model2_better"] == 12
        assert np.array_equal(
            result.additional_info["contingency_table"], table.tolist()
        )


class TestIntegrationScenarios:
    """Integration tests for statistical tests in maritime model comparison."""

    def test_maritime_model_comparison_workflow(self):
        """Test typical maritime model comparison workflow."""
        # Simulate ADE scores from cross-validation (transformer clearly better)
        lstm_ade = np.array([1.5, 1.4, 1.6, 1.3, 1.5, 1.7, 1.2, 1.8])
        transformer_ade = np.array([0.8, 0.7, 0.9, 0.6, 0.8, 0.9, 0.5, 1.0])

        # Perform paired t-test
        t_result = paired_t_test(lstm_ade, transformer_ade)

        # Should be significant (transformer much better)
        assert t_result.p_value < 0.05
        assert t_result.significant

        # Effect size should be large
        assert abs(t_result.effect_size) > 0.5

        # Also test with Wilcoxon (non-parametric)
        w_result = wilcoxon_test(lstm_ade, transformer_ade)

        # Should also be significant
        assert w_result.p_value < 0.05

        # Cliff's delta for effect size (transformer vs LSTM)
        cliff_result = cliffs_delta(transformer_ade, lstm_ade)

        # Should show large negative effect (transformer better, lower ADE)
        assert cliff_result.effect_size < -0.5
        assert cliff_result.effect_size_interpretation in ["medium", "large"]

    def test_anomaly_detection_comparison(self):
        """Test statistical comparison for anomaly detection models."""
        # Simulate confusion matrices for two anomaly detection models
        # Format: [[true_neg, false_pos], [false_neg, true_pos]]

        # Create contingency table for McNemar test
        # [[both_correct, A_correct_B_wrong], [A_wrong_B_correct, both_wrong]]
        model_comparison = np.array([[85, 5], [8, 2]])

        result = mcnemar_test(model_comparison)

        # Model A appears better (fewer errors when B is correct)
        assert result.additional_info["model1_better"] == 5
        assert result.additional_info["model2_better"] == 8

    def test_small_sample_robustness(self):
        """Test statistical tests with small samples."""
        small_sample_a = np.array([1.1, 1.2, 1.0])
        small_sample_b = np.array([0.9, 1.0, 0.8])

        # All tests should handle small samples gracefully
        t_result = paired_t_test(small_sample_a, small_sample_b)
        w_result = wilcoxon_test(small_sample_a, small_sample_b)
        c_result = cliffs_delta(small_sample_a, small_sample_b)

        # All should return valid results
        assert isinstance(t_result.p_value, float | np.floating)
        assert isinstance(w_result.p_value, float | np.floating)
        assert isinstance(c_result.effect_size, float | np.floating)

    def test_no_difference_scenario(self):
        """Test behavior when models perform identically."""
        identical_a = np.array([1.0, 1.0, 1.0, 1.0])
        identical_b = np.array([1.0, 1.0, 1.0, 1.0])

        t_result = paired_t_test(identical_a, identical_b)
        c_result = cliffs_delta(identical_a, identical_b)

        # Should not be significant
        assert not t_result.significant
        assert t_result.p_value > 0.05

        # Effect size should be negligible
        assert abs(c_result.effect_size) < 0.01
        assert c_result.effect_size_interpretation == "negligible"
