"""
Horizon curve analysis for understanding error progression over prediction steps.

This module provides tools to:
- Analyze how errors change across prediction horizons
- Generate horizon curves with confidence intervals
- Identify critical prediction steps where errors increase
- Provide statistical analysis of error degradation patterns
"""

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..stats.bootstrap import BootstrapCI, BootstrapResult


@dataclass
class HorizonPoint:
    """Single point on a horizon curve."""

    step: int
    mean_error: float
    std_error: float
    n_samples: int
    bootstrap_ci: BootstrapResult | None = None
    per_sample_errors: np.ndarray | None = None


@dataclass
class HorizonCurve:
    """Horizon curve showing error progression over prediction steps."""

    points: list[HorizonPoint]
    error_metric: str
    total_samples: int
    degradation_rate: float
    critical_steps: list[int]
    metadata: dict[str, Any] | None = None

    @property
    def steps(self) -> list[int]:
        """Get list of prediction steps."""
        return [point.step for point in self.points]

    @property
    def mean_errors(self) -> list[float]:
        """Get list of mean errors."""
        return [point.mean_error for point in self.points]

    @property
    def std_errors(self) -> list[float]:
        """Get list of standard deviations."""
        return [point.std_error for point in self.points]

    @property
    def confidence_intervals(self) -> list[tuple[float, float] | None]:
        """Get list of confidence intervals."""
        return [
            point.bootstrap_ci.confidence_interval if point.bootstrap_ci else None
            for point in self.points
        ]


@dataclass
class HorizonAnalysisResult:
    """Result of comprehensive horizon analysis."""

    overall_curve: HorizonCurve
    sliced_curves: dict[str, dict[str, HorizonCurve]] | None = None
    comparative_analysis: dict[str, Any] | None = None
    summary_statistics: dict[str, Any] | None = None


class HorizonAnalyzer:
    """
    Analyze error progression across prediction horizons.

    This class implements comprehensive horizon curve analysis to understand
    how prediction quality degrades over time, identify critical prediction
    steps, and provide statistical analysis of error patterns.

    Key Features:
    - Generate horizon curves with bootstrap confidence intervals
    - Identify critical steps where errors increase significantly
    - Analyze error degradation rates and patterns
    - Support for sliced analysis (by vessel type, conditions, etc.)
    - Statistical comparison of horizon curves

    Example:
        ```python
        analyzer = HorizonAnalyzer(confidence_level=0.95)

        # Basic horizon analysis
        curve = analyzer.analyze_horizon_errors(predictions, targets)

        # Access curve properties
        print(f"Degradation rate: {curve.degradation_rate:.3f}")
        print(f"Critical steps: {curve.critical_steps}")

        # Comprehensive analysis with slicing
        result = analyzer.comprehensive_horizon_analysis(
            predictions, targets, metadata, slice_by=['vessel_type']
        )
        ```
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        critical_step_threshold: float = 0.1,
    ):
        """
        Initialize HorizonAnalyzer.

        Args:
            confidence_level: Confidence level for bootstrap intervals
            n_bootstrap: Number of bootstrap resamples
            critical_step_threshold: Relative error increase threshold for critical steps
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.critical_step_threshold = critical_step_threshold
        self.bootstrap_ci = BootstrapCI(
            confidence_level=confidence_level, n_resamples=n_bootstrap
        )

    def analyze_horizon_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        error_metric: str = "mae",
        include_bootstrap: bool = True,
    ) -> HorizonCurve:
        """
        Generate comprehensive horizon curve analysis.

        Args:
            predictions: Model predictions shape [n_samples, horizon, features]
            targets: Ground truth targets shape [n_samples, horizon, features]
            error_metric: Error metric ('mae', 'mse', 'rmse')
            include_bootstrap: Whether to compute bootstrap confidence intervals

        Returns:
            HorizonCurve object with detailed analysis
        """
        # Validate inputs
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} != targets shape {targets.shape}"
            )

        n_samples, horizon_length, n_features = predictions.shape

        # Compute step-wise errors
        horizon_points = []

        for step in range(horizon_length):
            step_errors = self._compute_step_errors(
                predictions[:, step, :], targets[:, step, :], error_metric
            )

            # Compute statistics
            mean_error = np.mean(step_errors)
            std_error = np.std(step_errors)

            # Bootstrap CI
            bootstrap_result = None
            MIN_BOOTSTRAP_SAMPLES = 10
            if include_bootstrap and n_samples >= MIN_BOOTSTRAP_SAMPLES:
                try:
                    bootstrap_result = self.bootstrap_ci.compute_ci(
                        step_errors, np.mean
                    )
                except Exception as e:
                    warnings.warn(
                        f"Bootstrap CI failed for step {step + 1}: {e}", stacklevel=2
                    )

            horizon_point = HorizonPoint(
                step=step + 1,  # 1-indexed steps
                mean_error=mean_error,
                std_error=std_error,
                n_samples=n_samples,
                bootstrap_ci=bootstrap_result,
                per_sample_errors=step_errors,
            )

            horizon_points.append(horizon_point)

        # Analyze degradation and critical steps
        degradation_rate = self._compute_degradation_rate(horizon_points)
        critical_steps = self._identify_critical_steps(horizon_points)

        # Create metadata
        metadata = {
            "error_metric": error_metric,
            "horizon_length": horizon_length,
            "n_features": n_features,
            "include_bootstrap": include_bootstrap,
            "confidence_level": self.confidence_level,
        }

        return HorizonCurve(
            points=horizon_points,
            error_metric=error_metric,
            total_samples=n_samples,
            degradation_rate=degradation_rate,
            critical_steps=critical_steps,
            metadata=metadata,
        )

    def comprehensive_horizon_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any] | None = None,
        slice_by: list[str] | None = None,
        error_metric: str = "mae",
    ) -> HorizonAnalysisResult:
        """
        Perform comprehensive horizon analysis with optional slicing.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Sample metadata for slicing
            slice_by: List of slicing dimensions
            error_metric: Error metric to use

        Returns:
            HorizonAnalysisResult with comprehensive analysis
        """
        # Overall horizon curve
        overall_curve = self.analyze_horizon_errors(predictions, targets, error_metric)

        # Sliced analysis if requested
        sliced_curves = None
        if slice_by and metadata:
            sliced_curves = self._compute_sliced_horizon_curves(
                predictions, targets, metadata, slice_by, error_metric
            )

        # Comparative analysis
        comparative_analysis = self._compute_comparative_analysis(
            overall_curve, sliced_curves
        )

        # Summary statistics
        summary_statistics = self._compute_horizon_summary_statistics(
            overall_curve, sliced_curves
        )

        return HorizonAnalysisResult(
            overall_curve=overall_curve,
            sliced_curves=sliced_curves,
            comparative_analysis=comparative_analysis,
            summary_statistics=summary_statistics,
        )

    def compare_horizon_curves(
        self, curves: dict[str, HorizonCurve], statistical_test: str = "bootstrap"
    ) -> dict[str, Any]:
        """
        Statistically compare multiple horizon curves.

        Args:
            curves: Dictionary mapping curve names to HorizonCurve objects
            statistical_test: Type of statistical test ('bootstrap', 't_test')

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}

        curve_names = list(curves.keys())

        for i, name1 in enumerate(curve_names):
            for name2 in curve_names[i + 1 :]:
                curve1 = curves[name1]
                curve2 = curves[name2]

                # Perform step-wise comparison
                step_comparisons = []
                for point1, point2 in zip(curve1.points, curve2.points, strict=False):
                    if point1.step == point2.step:
                        step_comparison = self._compare_horizon_points(
                            point1, point2, statistical_test
                        )
                        step_comparisons.append(step_comparison)

                comparison_key = f"{name1}_vs_{name2}"
                comparison_results[comparison_key] = {
                    "step_comparisons": step_comparisons,
                    "overall_better": self._determine_overall_better_curve(
                        curve1, curve2
                    ),
                    "degradation_rate_diff": curve1.degradation_rate
                    - curve2.degradation_rate,
                }

        return comparison_results

    def plot_horizon_curve(
        self,
        curve: HorizonCurve,
        title: str | None = None,
        show_confidence_bands: bool = True,
        figsize: tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot a horizon curve with optional confidence bands.

        Args:
            curve: HorizonCurve to plot
            title: Plot title
            show_confidence_bands: Whether to show confidence intervals
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        steps = curve.steps
        mean_errors = curve.mean_errors

        # Plot main curve
        ax.plot(
            steps,
            mean_errors,
            "b-",
            linewidth=2,
            label=f"Mean {curve.error_metric.upper()}",
        )

        # Plot confidence bands if available
        if show_confidence_bands and curve.confidence_intervals:
            ci_lower = []
            ci_upper = []

            for ci in curve.confidence_intervals:
                if ci is not None:
                    ci_lower.append(ci[0])
                    ci_upper.append(ci[1])
                else:
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)

            ax.fill_between(
                steps,
                ci_lower,
                ci_upper,
                alpha=0.3,
                color="blue",
                label=f"{self.confidence_level*100:.0f}% Confidence Interval",
            )

        # Highlight critical steps
        if curve.critical_steps:
            critical_indices = [
                step - 1 for step in curve.critical_steps if step <= len(mean_errors)
            ]
            critical_errors = [mean_errors[i] for i in critical_indices]
            ax.scatter(
                curve.critical_steps,
                critical_errors,
                color="red",
                s=50,
                zorder=5,
                label=f"Critical Steps (>{self.critical_step_threshold*100:.0f}% increase)",
            )

        ax.set_xlabel("Prediction Step")
        ax.set_ylabel(f"{curve.error_metric.upper()} Error")
        ax.set_title(
            title or f"Horizon Curve - {curve.error_metric.upper()} Error Progression"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add degradation rate annotation
        ax.text(
            0.02,
            0.98,
            f"Degradation Rate: {curve.degradation_rate:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        plt.tight_layout()
        return fig

    def _compute_step_errors(
        self, step_predictions: np.ndarray, step_targets: np.ndarray, metric: str
    ) -> np.ndarray:
        """Compute errors for a specific prediction step."""
        if metric == "mae":
            errors = np.mean(np.abs(step_predictions - step_targets), axis=1)
        elif metric == "mse":
            errors = np.mean((step_predictions - step_targets) ** 2, axis=1)
        elif metric == "rmse":
            # RMSE should be sqrt of MSE for consistency
            mse_errors = np.mean((step_predictions - step_targets) ** 2, axis=1)
            errors = np.sqrt(mse_errors)
        else:
            raise ValueError(f"Unknown error metric: {metric}")

        return errors

    def _compute_degradation_rate(self, horizon_points: list[HorizonPoint]) -> float:
        """Compute overall error degradation rate."""
        MIN_POINTS_FOR_RATE = 2
        if len(horizon_points) < MIN_POINTS_FOR_RATE:
            return 0.0

        # Simple linear regression on log errors to get exponential rate
        steps = np.array([point.step for point in horizon_points])
        errors = np.array([point.mean_error for point in horizon_points])

        # Avoid log of zero/negative errors
        if np.any(errors <= 0):
            # Use relative change instead
            return (errors[-1] - errors[0]) / (steps[-1] - steps[0])

        log_errors = np.log(errors)

        # Linear regression: log(error) = rate * step + intercept
        coeffs = np.polyfit(steps, log_errors, 1)
        degradation_rate = coeffs[0]

        return degradation_rate

    def _identify_critical_steps(self, horizon_points: list[HorizonPoint]) -> list[int]:
        """Identify steps where error increases significantly."""
        MIN_POINTS_FOR_CRITICAL = 2
        if len(horizon_points) < MIN_POINTS_FOR_CRITICAL:
            return []

        critical_steps = []
        errors = [point.mean_error for point in horizon_points]

        for i in range(1, len(errors)):
            if errors[i - 1] > 0:  # Avoid division by zero
                relative_increase = (errors[i] - errors[i - 1]) / errors[i - 1]
                if relative_increase > self.critical_step_threshold:
                    critical_steps.append(horizon_points[i].step)

        return critical_steps

    def _compute_sliced_horizon_curves(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any],
        slice_by: list[str],
        error_metric: str,
    ) -> dict[str, dict[str, HorizonCurve]]:
        """Compute horizon curves for different slices."""
        # Import ErrorSlicer for slicing functionality
        from .slicers import ErrorSlicer

        sliced_curves = {}

        # Create dummy errors for slicing (we'll compute horizon-specific errors)
        dummy_errors = np.ones(len(predictions))

        slicer = ErrorSlicer()

        for slice_dim in slice_by:
            if slice_dim in slicer.slices:
                slice_curves = {}

                # Get slice assignments
                try:
                    slicer_func = slicer.slices[slice_dim].slicer_func
                    slice_assignments = slicer_func(metadata, dummy_errors)

                    # Compute curve for each bin
                    for bin_name in slicer.slices[slice_dim].bins:
                        bin_mask = slice_assignments == bin_name

                        if np.any(bin_mask):
                            bin_predictions = predictions[bin_mask]
                            bin_targets = targets[bin_mask]

                            MIN_BIN_SAMPLES = 5
                            MIN_BOOTSTRAP_SAMPLES = 10
                            if (
                                len(bin_predictions) >= MIN_BIN_SAMPLES
                            ):  # Minimum samples for meaningful analysis
                                bin_curve = self.analyze_horizon_errors(
                                    bin_predictions,
                                    bin_targets,
                                    error_metric,
                                    include_bootstrap=len(bin_predictions)
                                    >= MIN_BOOTSTRAP_SAMPLES,
                                )
                                slice_curves[bin_name] = bin_curve

                    sliced_curves[slice_dim] = slice_curves

                except Exception as e:
                    warnings.warn(
                        f"Failed to compute sliced curves for {slice_dim}: {e}",
                        stacklevel=2,
                    )

        return sliced_curves

    def _compute_comparative_analysis(
        self,
        overall_curve: HorizonCurve,
        sliced_curves: dict[str, dict[str, HorizonCurve]] | None,
    ) -> dict[str, Any]:
        """Compute comparative analysis between curves."""
        analysis = {
            "overall_degradation_rate": overall_curve.degradation_rate,
            "overall_critical_steps": overall_curve.critical_steps,
        }

        if sliced_curves:
            slice_degradation_rates = {}
            slice_critical_steps = {}

            for slice_dim, curves in sliced_curves.items():
                rates = {
                    bin_name: curve.degradation_rate
                    for bin_name, curve in curves.items()
                }
                critical_steps = {
                    bin_name: curve.critical_steps for bin_name, curve in curves.items()
                }

                slice_degradation_rates[slice_dim] = rates
                slice_critical_steps[slice_dim] = critical_steps

            analysis["slice_degradation_rates"] = slice_degradation_rates
            analysis["slice_critical_steps"] = slice_critical_steps

            # Find best and worst performing slices
            for slice_dim, curves in sliced_curves.items():
                final_errors = {
                    bin_name: curve.mean_errors[-1]
                    for bin_name, curve in curves.items()
                }

                if final_errors:
                    best_slice = min(final_errors.items(), key=lambda x: x[1])
                    worst_slice = max(final_errors.items(), key=lambda x: x[1])

                    analysis[f"{slice_dim}_best_performing"] = {
                        "bin": best_slice[0],
                        "final_error": best_slice[1],
                    }
                    analysis[f"{slice_dim}_worst_performing"] = {
                        "bin": worst_slice[0],
                        "final_error": worst_slice[1],
                    }

        return analysis

    def _compute_horizon_summary_statistics(
        self,
        overall_curve: HorizonCurve,
        sliced_curves: dict[str, dict[str, HorizonCurve]] | None,
    ) -> dict[str, Any]:
        """Compute summary statistics for horizon analysis."""
        summary = {
            "total_samples": overall_curve.total_samples,
            "horizon_length": len(overall_curve.points),
            "error_metric": overall_curve.error_metric,
            "initial_error": overall_curve.mean_errors[0],
            "final_error": overall_curve.mean_errors[-1],
            "error_range": (
                min(overall_curve.mean_errors),
                max(overall_curve.mean_errors),
            ),
            "degradation_rate": overall_curve.degradation_rate,
            "n_critical_steps": len(overall_curve.critical_steps),
        }

        if sliced_curves:
            slice_summaries = {}
            for slice_dim, curves in sliced_curves.items():
                slice_summary = {
                    "n_bins_analyzed": len(curves),
                    "degradation_rates": {
                        name: curve.degradation_rate for name, curve in curves.items()
                    },
                    "final_errors": {
                        name: curve.mean_errors[-1] for name, curve in curves.items()
                    },
                }
                slice_summaries[slice_dim] = slice_summary

            summary["slice_summaries"] = slice_summaries

        return summary

    def _compare_horizon_points(
        self, point1: HorizonPoint, point2: HorizonPoint, test_type: str
    ) -> dict[str, Any]:
        """Compare two horizon points statistically."""
        comparison = {
            "step": point1.step,
            "mean_diff": point1.mean_error - point2.mean_error,
            "relative_diff": (point1.mean_error - point2.mean_error) / point2.mean_error
            if point2.mean_error > 0
            else np.inf,
        }

        if test_type == "bootstrap" and point1.bootstrap_ci and point2.bootstrap_ci:
            # Check if confidence intervals overlap
            ci1 = point1.bootstrap_ci.confidence_interval
            ci2 = point2.bootstrap_ci.confidence_interval

            overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
            comparison["ci_overlap"] = overlap
            comparison["significant_diff"] = not overlap

        return comparison

    def _determine_overall_better_curve(
        self, curve1: HorizonCurve, curve2: HorizonCurve
    ) -> str:
        """Determine which curve performs better overall."""
        # Compare based on final error and degradation rate
        final_error_diff = curve1.mean_errors[-1] - curve2.mean_errors[-1]
        degradation_rate_diff = curve1.degradation_rate - curve2.degradation_rate

        # Lower final error and lower degradation rate are better
        if final_error_diff < 0 and degradation_rate_diff < 0:
            return "curve1"
        elif final_error_diff > 0 and degradation_rate_diff > 0:
            return "curve2"
        else:
            # Mixed results - use weighted combination
            weighted_score1 = final_error_diff + 0.5 * degradation_rate_diff
            return "curve1" if weighted_score1 < 0 else "curve2"
