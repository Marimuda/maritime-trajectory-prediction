"""Model comparison framework with statistical testing."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.bootstrap import BootstrapCI, BootstrapResult
from ..stats.corrections import CorrectionResult, multiple_comparison_correction
from ..stats.tests import StatTestResult, cliffs_delta, paired_t_test, wilcoxon_test


@dataclass
class ComparisonResult:
    """Result of model comparison analysis."""

    model_names: list[str]
    metrics: dict[str, np.ndarray]  # metric_name -> array of per-fold scores
    bootstrap_cis: dict[
        str, dict[str, BootstrapResult]
    ]  # model -> metric -> bootstrap result
    pairwise_tests: dict[
        str, dict[str, StatTestResult]
    ]  # comparison -> metric -> test result
    corrected_pvalues: dict[str, CorrectionResult] | None = None
    summary_table: pd.DataFrame | None = None
    best_model: dict[str, str] | None = None  # metric -> best model name


class ModelComparison:
    """Framework for comparing multiple models with statistical rigor."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 9999,
        correction_method: str = "holm",
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        """
        Initialize model comparison framework.

        Args:
            confidence_level: Confidence level for bootstrap CIs
            n_bootstrap: Number of bootstrap resamples
            correction_method: Multiple comparison correction method
            alpha: Significance level
            random_state: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.correction_method = correction_method
        self.alpha = alpha
        self.random_state = random_state
        self.bootstrap_calculator = BootstrapCI(
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    def compare_models(
        self,
        results: dict[str, dict[str, np.ndarray]],
        metrics: list[str] | None = None,
    ) -> ComparisonResult:
        """
        Compare multiple models across multiple metrics.

        Args:
            results: Nested dict {model_name: {metric_name: scores_array}}
            metrics: List of metrics to compare (if None, uses all)

        Returns:
            ComparisonResult with comprehensive comparison analysis

        Example:
            >>> results = {
            ...     'LSTM': {'ADE': np.array([1.2, 1.1, 1.3, 1.0, 1.2]),
            ...              'FDE': np.array([2.1, 2.0, 2.2, 1.9, 2.1])},
            ...     'Transformer': {'ADE': np.array([0.9, 0.8, 1.0, 0.7, 0.9]),
            ...                     'FDE': np.array([1.6, 1.5, 1.7, 1.4, 1.6])}
            ... }
            >>> comparison = ModelComparison()
            >>> result = comparison.compare_models(results)
            >>> print(result.summary_table)
        """
        model_names = list(results.keys())
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for comparison")

        # Determine metrics to analyze
        if metrics is None:
            all_metrics = set()
            for model_results in results.values():
                all_metrics.update(model_results.keys())
            metrics = sorted(list(all_metrics))

        # Validate that all models have all metrics
        for model_name in model_names:
            for metric in metrics:
                if metric not in results[model_name]:
                    raise ValueError(f"Model '{model_name}' missing metric '{metric}'")

        # Organize metrics data
        metrics_data = {}
        for metric in metrics:
            metric_scores = np.array([results[model][metric] for model in model_names])
            metrics_data[metric] = metric_scores

        # Compute bootstrap confidence intervals
        bootstrap_cis = {}
        for model_name in model_names:
            bootstrap_cis[model_name] = {}
            for metric in metrics:
                scores = results[model_name][metric]
                bootstrap_result = self.bootstrap_calculator.compute_ci(scores)
                bootstrap_cis[model_name][metric] = bootstrap_result

        # Perform pairwise comparisons
        pairwise_tests = {}
        all_pvalues = {}

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{model_a}_vs_{model_b}"
                    pairwise_tests[comparison_key] = {}

                    for metric in metrics:
                        scores_a = results[model_a][metric]
                        scores_b = results[model_b][metric]

                        # Paired t-test (parametric)
                        t_test = paired_t_test(scores_a, scores_b, self.alpha)

                        # Wilcoxon test (non-parametric)
                        wilcoxon = wilcoxon_test(scores_a, scores_b, self.alpha)

                        # Cliff's delta (effect size)
                        cliff_delta = cliffs_delta(scores_a, scores_b)

                        # Store the most appropriate test result
                        # Use Wilcoxon if sample size is small or data not normal
                        n_samples = len(scores_a)
                        if n_samples < 20:
                            primary_test = wilcoxon
                        else:
                            primary_test = t_test

                        # Combine results
                        combined_result = StatTestResult(
                            test_name=f"{primary_test.test_name} + Cliff's δ",
                            statistic=primary_test.statistic,
                            p_value=primary_test.p_value,
                            effect_size=cliff_delta.effect_size,
                            effect_size_interpretation=cliff_delta.effect_size_interpretation,
                            significant=primary_test.significant,
                            alpha=self.alpha,
                            additional_info={
                                "t_test": t_test,
                                "wilcoxon": wilcoxon,
                                "cliffs_delta": cliff_delta,
                                "n_samples": n_samples,
                            },
                        )

                        pairwise_tests[comparison_key][metric] = combined_result

                        # Collect p-values for multiple comparison correction
                        pvalue_key = f"{comparison_key}_{metric}"
                        all_pvalues[pvalue_key] = primary_test.p_value

        # Apply multiple comparison correction
        corrected_pvalues = None
        if len(all_pvalues) > 1:
            try:
                correction_result = multiple_comparison_correction(
                    list(all_pvalues.values()),
                    alpha=self.alpha,
                    method=self.correction_method,
                )
                corrected_pvalues = {
                    "correction_result": correction_result,
                    "pvalue_mapping": list(all_pvalues.keys()),
                }
            except Exception as e:
                warnings.warn(f"Multiple comparison correction failed: {e}")

        # Create summary table
        summary_table = self._create_summary_table(
            model_names, metrics, bootstrap_cis, results
        )

        # Determine best models per metric
        best_model = {}
        for metric in metrics:
            metric_means = []
            for model_name in model_names:
                mean_score = np.mean(results[model_name][metric])
                metric_means.append((mean_score, model_name))

            # For error metrics (ADE, FDE, RMSE), lower is better
            # For accuracy metrics, higher is better
            # Assume lower is better by default (most maritime metrics are errors)
            metric_means.sort()
            best_model[metric] = metric_means[0][1]

        return ComparisonResult(
            model_names=model_names,
            metrics=metrics_data,
            bootstrap_cis=bootstrap_cis,
            pairwise_tests=pairwise_tests,
            corrected_pvalues=corrected_pvalues,
            summary_table=summary_table,
            best_model=best_model,
        )

    def _create_summary_table(
        self,
        model_names: list[str],
        metrics: list[str],
        bootstrap_cis: dict[str, dict[str, BootstrapResult]],
        raw_results: dict[str, dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Create summary table with means, CIs, and significance indicators."""
        rows = []

        for metric in metrics:
            for model_name in model_names:
                scores = raw_results[model_name][metric]
                bootstrap_result = bootstrap_cis[model_name][metric]

                ci_low, ci_high = bootstrap_result.confidence_interval

                row = {
                    "Metric": metric,
                    "Model": model_name,
                    "Mean": np.mean(scores),
                    "Std": np.std(scores, ddof=1),
                    "CI_Low": ci_low,
                    "CI_High": ci_high,
                    "CI_Width": ci_high - ci_low,
                    "N": len(scores),
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # Add formatted CI column
        df["CI_Formatted"] = df.apply(
            lambda row: f"[{row['CI_Low']:.3f}, {row['CI_High']:.3f}]", axis=1
        )

        # Reorder columns
        column_order = [
            "Metric",
            "Model",
            "Mean",
            "Std",
            "CI_Formatted",
            "CI_Low",
            "CI_High",
            "N",
        ]
        df = df[column_order]

        return df.sort_values(["Metric", "Mean"])

    def plot_comparison(
        self, comparison_result: ComparisonResult, metric: str, figsize: tuple = (10, 6)
    ) -> "matplotlib.figure.Figure":
        """
        Create comparison plot for a specific metric.

        Args:
            comparison_result: Result from compare_models()
            metric: Metric to plot
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if (
            metric
            not in comparison_result.bootstrap_cis[comparison_result.model_names[0]]
        ):
            raise ValueError(f"Metric '{metric}' not found in comparison results")

        fig, ax = plt.subplots(figsize=figsize)

        model_names = comparison_result.model_names
        positions = range(len(model_names))

        means = []
        ci_lows = []
        ci_highs = []

        for model_name in model_names:
            bootstrap_result = comparison_result.bootstrap_cis[model_name][metric]
            means.append(bootstrap_result.statistic_value)
            ci_low, ci_high = bootstrap_result.confidence_interval
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        # Create error bars
        yerr = [
            np.array(means) - np.array(ci_lows),
            np.array(ci_highs) - np.array(means),
        ]

        ax.errorbar(positions, means, yerr=yerr, fmt="o", capsize=5, capthick=2)
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric}")
        ax.set_title(
            f"{metric} Comparison with {int(comparison_result.bootstrap_cis[model_names[0]][metric].confidence_level*100)}% CI"
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
