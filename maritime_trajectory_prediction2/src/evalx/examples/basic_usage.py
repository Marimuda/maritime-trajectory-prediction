"""
Basic usage examples for the evalx statistical evaluation framework.

This script demonstrates the core functionality of evalx including:
- Bootstrap confidence intervals
- Statistical significance tests
- Cross-validation protocols
- Model comparison workflows

Run this script to see the framework in action with maritime trajectory data examples.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evalx.stats import bootstrap_ci, cliffs_delta, paired_t_test, wilcoxon_test
from evalx.stats.corrections import multiple_comparison_correction
from evalx.validation.comparisons import ModelComparison
from evalx.validation.protocols import maritime_cv_split


def example_bootstrap_confidence_intervals():
    """Demonstrate bootstrap confidence intervals for maritime metrics."""
    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS EXAMPLE")
    print("=" * 60)

    # Simulate ADE (Average Displacement Error) scores from cross-validation
    print("\n1. Single Model Evaluation")
    print("-" * 30)

    lstm_ade_scores = np.array([1.2, 1.1, 1.3, 1.0, 1.15, 1.25, 1.05, 1.18])
    print(f"LSTM ADE scores: {lstm_ade_scores}")

    # Compute 95% confidence interval
    result = bootstrap_ci(lstm_ade_scores, confidence_level=0.95, random_state=42)

    print("\nBootstrap Results:")
    print(f"  Mean ADE: {result.statistic_value:.3f} km")
    print(
        f"  95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}] km"
    )
    print(
        f"  CI Width: {result.confidence_interval[1] - result.confidence_interval[0]:.3f} km"
    )
    print(f"  Method: {result.method}")
    print(f"  Bootstrap samples: {result.n_resamples}")

    # Different confidence levels
    print("\n2. Different Confidence Levels")
    print("-" * 30)

    for confidence in [0.90, 0.95, 0.99]:
        result = bootstrap_ci(
            lstm_ade_scores, confidence_level=confidence, random_state=42
        )
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        print(
            f"  {int(confidence*100)}% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}] (width: {ci_width:.3f})"
        )

    print("\n3. Different Statistics")
    print("-" * 30)

    # Bootstrap CI for median
    median_result = bootstrap_ci(lstm_ade_scores, statistic=np.median, random_state=42)
    print(f"  Median: {median_result.statistic_value:.3f} km")
    print(
        f"  95% CI: [{median_result.confidence_interval[0]:.3f}, {median_result.confidence_interval[1]:.3f}] km"
    )

    # Bootstrap CI for standard deviation
    std_result = bootstrap_ci(
        lstm_ade_scores, statistic=lambda x: np.std(x, ddof=1), random_state=42
    )
    print(f"  Std Dev: {std_result.statistic_value:.3f} km")
    print(
        f"  95% CI: [{std_result.confidence_interval[0]:.3f}, {std_result.confidence_interval[1]:.3f}] km"
    )


def example_statistical_tests():
    """Demonstrate statistical significance testing for model comparison."""
    print("\n\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING EXAMPLE")
    print("=" * 60)

    # Simulate model comparison scenario
    lstm_scores = np.array([1.2, 1.1, 1.3, 1.0, 1.15, 1.25, 1.05])
    transformer_scores = np.array([0.9, 0.8, 1.0, 0.7, 0.85, 0.95, 0.75])

    print(f"\nLSTM ADE scores:        {lstm_scores}")
    print(f"Transformer ADE scores: {transformer_scores}")
    print(
        f"Mean difference: {np.mean(lstm_scores) - np.mean(transformer_scores):.3f} km"
    )

    print("\n1. Paired t-test (Parametric)")
    print("-" * 30)

    t_result = paired_t_test(lstm_scores, transformer_scores)
    print(f"  Test statistic: {t_result.statistic:.3f}")
    print(f"  P-value: {t_result.p_value:.6f}")
    print(f"  Significant at α=0.05: {t_result.significant}")
    print(
        f"  Effect size (Cohen's d): {t_result.effect_size:.3f} ({t_result.effect_size_interpretation})"
    )

    print("\n2. Wilcoxon Signed-Rank Test (Non-parametric)")
    print("-" * 30)

    w_result = wilcoxon_test(lstm_scores, transformer_scores)
    print(f"  Test statistic: {w_result.statistic:.3f}")
    print(f"  P-value: {w_result.p_value:.6f}")
    print(f"  Significant at α=0.05: {w_result.significant}")
    if w_result.effect_size is not None:
        print(
            f"  Effect size: {w_result.effect_size:.3f} ({w_result.effect_size_interpretation})"
        )

    print("\n3. Cliff's Delta (Effect Size)")
    print("-" * 30)

    cliff_result = cliffs_delta(lstm_scores, transformer_scores)
    print(
        f"  Cliff's δ: {cliff_result.effect_size:.3f} ({cliff_result.effect_size_interpretation})"
    )
    print(
        f"  Interpretation: Transformer has {abs(cliff_result.effect_size):.1%} probability of lower error"
    )


def example_multiple_comparison_correction():
    """Demonstrate multiple comparison correction."""
    print("\n\n" + "=" * 60)
    print("MULTIPLE COMPARISON CORRECTION EXAMPLE")
    print("=" * 60)

    # Simulate multiple model comparisons
    p_values = [0.01, 0.04, 0.03, 0.08, 0.002, 0.12, 0.006]
    comparison_names = [
        "LSTM vs Trans",
        "LSTM vs XGBoost",
        "Trans vs XGBoost",
        "LSTM vs SVR",
        "Trans vs SVR",
        "XGBoost vs SVR",
        "LSTM vs Kalman",
    ]

    print(f"\nOriginal p-values from {len(p_values)} comparisons:")
    for name, p_val in zip(comparison_names, p_values, strict=False):
        significant = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else ""
        )
        print(f"  {name:<20}: {p_val:.4f} {significant}")

    print(
        f"\nSignificant at α=0.05 (uncorrected): {sum(p < 0.05 for p in p_values)}/{len(p_values)}"
    )

    # Apply different correction methods
    methods = ["bonferroni", "holm", "fdr_bh"]

    for method in methods:
        print(f"\n{method.upper()} Correction:")
        print("-" * 30)

        result = multiple_comparison_correction(p_values, alpha=0.05, method=method)

        print(f"  Corrected α: {result.original_alpha:.3f}")
        print(
            f"  Significant comparisons: {result.n_significant}/{result.n_comparisons}"
        )

        for name, orig_p, corr_p, sig in zip(
            comparison_names,
            p_values,
            result.corrected_pvalues,
            result.significant,
            strict=False,
        ):
            sig_marker = (
                "***"
                if corr_p < 0.001
                else "**"
                if corr_p < 0.01
                else "*"
                if sig
                else ""
            )
            print(f"  {name:<20}: {orig_p:.4f} → {corr_p:.4f} {sig_marker}")


def example_cross_validation_protocols():
    """Demonstrate maritime-specific cross-validation protocols."""
    print("\n\n" + "=" * 60)
    print("CROSS-VALIDATION PROTOCOLS EXAMPLE")
    print("=" * 60)

    # Create sample maritime dataset
    np.random.seed(42)
    vessels = [f"vessel_{i:03d}" for i in range(5)]
    dates = pd.date_range("2023-01-01", periods=200, freq="30min")

    data = []
    for timestamp in dates:
        vessel = np.random.choice(vessels)
        data.append(
            {
                "timestamp": timestamp,
                "mmsi": vessel,
                "lat": 60.0 + np.random.randn() * 0.1,
                "lon": -7.0 + np.random.randn() * 0.1,
                "sog": np.abs(np.random.randn() * 5 + 10),
            }
        )

    df = pd.DataFrame(data)
    print(f"\nDataset: {len(df)} AIS messages from {df['mmsi'].nunique()} vessels")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\n1. Temporal Cross-Validation")
    print("-" * 30)

    temporal_splits = maritime_cv_split(
        df, split_type="temporal", n_splits=3, min_gap_minutes=60
    )
    print(f"  Number of splits: {len(temporal_splits)}")

    for i, (train_idx, test_idx) in enumerate(temporal_splits):
        train_time_range = (
            df.iloc[train_idx]["timestamp"].min(),
            df.iloc[train_idx]["timestamp"].max(),
        )
        test_time_range = (
            df.iloc[test_idx]["timestamp"].min(),
            df.iloc[test_idx]["timestamp"].max(),
        )
        gap = (test_time_range[0] - train_time_range[1]).total_seconds() / 3600  # hours

        print(
            f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}, Gap={gap:.1f}h"
        )

    print("\n2. Vessel-Based Cross-Validation")
    print("-" * 30)

    vessel_splits = maritime_cv_split(
        df, split_type="vessel", n_splits=3, random_state=42
    )
    print(f"  Number of splits: {len(vessel_splits)}")

    for i, (train_idx, test_idx) in enumerate(vessel_splits):
        train_vessels = set(df.iloc[train_idx]["mmsi"].unique())
        test_vessels = set(df.iloc[test_idx]["mmsi"].unique())
        overlap = len(train_vessels & test_vessels)

        print(
            f"  Fold {i+1}: Train={len(train_idx)} ({len(train_vessels)} vessels), "
            f"Test={len(test_idx)} ({len(test_vessels)} vessels), Overlap={overlap}"
        )


def example_comprehensive_model_comparison():
    """Demonstrate comprehensive model comparison workflow."""
    print("\n\n" + "=" * 60)
    print("COMPREHENSIVE MODEL COMPARISON EXAMPLE")
    print("=" * 60)

    # Simulate cross-validation results for multiple models and metrics
    np.random.seed(42)

    models = ["LSTM", "Transformer", "XGBoost", "Kalman"]
    metrics = ["ADE", "FDE", "Course_RMSE"]
    n_folds = 5

    # Generate realistic performance data
    performance_ranges = {
        "LSTM": {"ADE": (1.0, 1.4), "FDE": (1.8, 2.4), "Course_RMSE": (8, 12)},
        "Transformer": {"ADE": (0.7, 1.1), "FDE": (1.4, 1.9), "Course_RMSE": (6, 10)},
        "XGBoost": {"ADE": (1.1, 1.5), "FDE": (2.0, 2.6), "Course_RMSE": (9, 13)},
        "Kalman": {"ADE": (1.3, 1.8), "FDE": (2.2, 2.8), "Course_RMSE": (12, 16)},
    }

    results = {}
    for model in models:
        results[model] = {}
        for metric in metrics:
            min_val, max_val = performance_ranges[model][metric]
            results[model][metric] = np.random.uniform(min_val, max_val, n_folds)

    print("\nModel Performance (Mean ± Std):")
    print("-" * 40)
    for model in models:
        print(f"\n{model}:")
        for metric in metrics:
            scores = results[model][metric]
            print(f"  {metric:<12}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    print("\n" + "=" * 40)
    print("STATISTICAL COMPARISON")
    print("=" * 40)

    # Perform comprehensive comparison
    comparison = ModelComparison(
        confidence_level=0.95,
        n_bootstrap=1999,
        correction_method="holm",
        random_state=42,
    )

    comp_result = comparison.compare_models(results)

    print("\nBest Models per Metric:")
    print("-" * 30)
    for metric, best_model in comp_result.best_model.items():
        best_score = np.mean(results[best_model][metric])
        print(f"  {metric:<12}: {best_model} ({best_score:.3f})")

    print("\nSummary Table (first few rows):")
    print("-" * 30)
    print(comp_result.summary_table.head(8).to_string(index=False))

    print("\nPairwise Comparisons (ADE metric):")
    print("-" * 30)
    for comparison_name, test_results in comp_result.pairwise_tests.items():
        if "ADE" in test_results:
            test = test_results["ADE"]
            sig_marker = (
                "***"
                if test.p_value < 0.001
                else "**"
                if test.p_value < 0.01
                else "*"
                if test.significant
                else ""
            )
            print(
                f"  {comparison_name:<20}: p={test.p_value:.4f} {sig_marker}, "
                f"δ={test.effect_size:.3f} ({test.effect_size_interpretation})"
            )

    if comp_result.corrected_pvalues:
        correction_result = comp_result.corrected_pvalues["correction_result"]
        print(f"\nMultiple Comparison Correction ({correction_result.method}):")
        print(
            f"  Original significant: {sum(1 for tests in comp_result.pairwise_tests.values() for test in tests.values() if test.significant)}"
        )
        print(
            f"  Corrected significant: {correction_result.n_significant}/{correction_result.n_comparisons}"
        )


def main():
    """Run all examples."""
    print("EVALX STATISTICAL FRAMEWORK - USAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the key features of the evalx framework")
    print("for statistical evaluation of maritime trajectory prediction models.")

    try:
        example_bootstrap_confidence_intervals()
        example_statistical_tests()
        example_multiple_comparison_correction()
        example_cross_validation_protocols()
        example_comprehensive_model_comparison()

        print("\n\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - src/evalx/stats/ for statistical functions")
        print("  - src/evalx/validation/ for cross-validation protocols")
        print("  - tests/unit/evalx/ for comprehensive test suite")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("  pip install numpy pandas scipy statsmodels")
        raise


if __name__ == "__main__":
    main()
