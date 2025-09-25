"""Enhanced metrics system with statistical evaluation integration."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torchmetrics import Metric

from ..stats.bootstrap import BootstrapCI, BootstrapResult
from ..validation.comparisons import ComparisonResult, ModelComparison


@dataclass
class MetricResult:
    """Enhanced metric result with statistical information."""

    value: float
    bootstrap_ci: BootstrapResult | None = None
    per_sample_values: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class StatisticalMetricWrapper(nn.Module):
    """Wrapper that adds statistical analysis capabilities to torchmetrics."""

    def __init__(
        self,
        metric: Metric,
        collect_samples: bool = True,
        bootstrap_ci: bool = True,
        confidence_level: float = 0.95,
        n_bootstrap: int = 9999,
    ):
        """
        Initialize statistical metric wrapper.

        Args:
            metric: Base torchmetrics Metric
            collect_samples: Whether to collect per-sample values
            bootstrap_ci: Whether to compute bootstrap confidence intervals
            confidence_level: Confidence level for bootstrap CIs
            n_bootstrap: Number of bootstrap resamples
        """
        super().__init__()
        self.metric = metric
        self.collect_samples = collect_samples
        self.bootstrap_ci = bootstrap_ci
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap

        # Storage for per-sample values
        self.per_sample_values = [] if collect_samples else None

        # Bootstrap calculator
        if bootstrap_ci:
            self.bootstrap_calculator = BootstrapCI(
                n_resamples=n_bootstrap, confidence_level=confidence_level
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric and collect per-sample values if enabled."""
        # Update base metric
        self.metric.update(preds, target)

        # Collect per-sample values for statistical analysis
        if self.collect_samples:
            # Compute per-sample metric values
            sample_values = self._compute_per_sample(preds, target)
            if sample_values is not None:
                self.per_sample_values.extend(sample_values.cpu().numpy().flatten())

    def _compute_per_sample(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute per-sample metric values. Override for custom metrics."""
        # Default implementation for distance-based metrics
        if (
            hasattr(self.metric, "_haversine")
            or "ADE" in str(type(self.metric))
            or "FDE" in str(type(self.metric))
        ):
            return self._compute_distance_per_sample(preds, target)
        elif "Course" in str(type(self.metric)):
            return self._compute_course_per_sample(preds, target)
        else:
            # For other metrics, try to compute element-wise differences
            try:
                return torch.mean((preds - target) ** 2, dim=-1)
            except:
                warnings.warn(
                    f"Could not compute per-sample values for {type(self.metric)}"
                )
                return None

    def _compute_distance_per_sample(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample distance values."""
        from ...metrics.trajectory_metrics import _haversine

        if preds.shape[-1] >= 2 and target.shape[-1] >= 2:
            # Position data [B, T, 2] or [B, 2]
            if len(preds.shape) == 3:
                # Trajectory data - compute mean distance per trajectory
                lat1, lon1 = preds[..., 0], preds[..., 1]
                lat2, lon2 = target[..., 0], target[..., 1]
                distances = _haversine(lat1, lon1, lat2, lon2)
                return torch.mean(distances, dim=-1)  # [B]
            else:
                # Single point data [B, 2]
                lat1, lon1 = preds[:, 0], preds[:, 1]
                lat2, lon2 = target[:, 0], target[:, 1]
                return _haversine(lat1, lon1, lat2, lon2)
        return None

    def _compute_course_per_sample(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample course error."""
        diff = preds - target
        diff = torch.where(diff > 180, diff - 360, diff)
        diff = torch.where(diff < -180, diff + 360, diff)

        if len(diff.shape) > 1:
            return torch.sqrt(torch.mean(diff**2, dim=-1))  # RMS per sample
        else:
            return torch.abs(diff)

    def compute(self) -> MetricResult:
        """Compute metric with statistical analysis."""
        # Get base metric value
        base_value = self.metric.compute()

        # Convert to numpy for statistical analysis
        if isinstance(base_value, torch.Tensor):
            base_value_np = base_value.cpu().item()
        else:
            base_value_np = float(base_value)

        # Compute bootstrap CI if requested and samples available
        bootstrap_result = None
        if (
            self.bootstrap_ci
            and self.per_sample_values
            and len(self.per_sample_values) > 1
        ):
            try:
                sample_array = np.array(self.per_sample_values)
                bootstrap_result = self.bootstrap_calculator.compute_ci(sample_array)
            except Exception as e:
                warnings.warn(f"Bootstrap CI computation failed: {e}")

        return MetricResult(
            value=base_value_np,
            bootstrap_ci=bootstrap_result,
            per_sample_values=np.array(self.per_sample_values)
            if self.per_sample_values
            else None,
            metadata={
                "n_samples": len(self.per_sample_values)
                if self.per_sample_values
                else 0,
                "metric_type": str(type(self.metric)),
            },
        )

    def reset(self):
        """Reset metric state."""
        self.metric.reset()
        if self.per_sample_values is not None:
            self.per_sample_values.clear()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        """Forward pass for compatibility."""
        return self.metric(preds, target)


class MetricCollector:
    """Collects and manages multiple statistical metrics."""

    def __init__(
        self,
        metrics: dict[str, Metric],
        enable_statistical: bool = True,
        confidence_level: float = 0.95,
    ):
        """
        Initialize metric collector.

        Args:
            metrics: Dictionary of metric_name -> Metric
            enable_statistical: Whether to enable statistical analysis
            confidence_level: Confidence level for statistical analysis
        """
        self.metrics = {}
        self.enable_statistical = enable_statistical
        self.confidence_level = confidence_level

        # Wrap metrics with statistical capabilities
        for name, metric in metrics.items():
            if enable_statistical:
                self.metrics[name] = StatisticalMetricWrapper(
                    metric=metric.clone() if hasattr(metric, "clone") else metric,
                    confidence_level=confidence_level,
                )
            else:
                self.metrics[name] = metric

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(preds, target)

    def compute(self) -> dict[str, float | MetricResult]:
        """Compute all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, StatisticalMetricWrapper):
                results[name] = metric.compute()
            else:
                value = metric.compute()
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                results[name] = float(value)
        return results

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def get_statistical_summary(self) -> dict[str, dict]:
        """Get statistical summary of all metrics."""
        results = self.compute()
        summary = {}

        for name, result in results.items():
            if isinstance(result, MetricResult):
                summary[name] = {
                    "value": result.value,
                    "n_samples": result.metadata.get("n_samples", 0)
                    if result.metadata
                    else 0,
                }

                if result.bootstrap_ci:
                    ci = result.bootstrap_ci.confidence_interval
                    summary[name].update(
                        {
                            "ci_lower": ci[0],
                            "ci_upper": ci[1],
                            "ci_width": ci[1] - ci[0],
                            "confidence_level": result.bootstrap_ci.confidence_level,
                        }
                    )

                if (
                    result.per_sample_values is not None
                    and len(result.per_sample_values) > 0
                ):
                    values = result.per_sample_values
                    summary[name].update(
                        {
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "median": np.median(values),
                            "q25": np.percentile(values, 25),
                            "q75": np.percentile(values, 75),
                        }
                    )
            else:
                summary[name] = {"value": result}

        return summary


class EvaluationRunner:
    """High-level runner for model evaluation with statistical analysis."""

    def __init__(
        self,
        metrics: dict[str, Metric],
        confidence_level: float = 0.95,
        comparison_method: str = "holm",
    ):
        """
        Initialize evaluation runner.

        Args:
            metrics: Dictionary of metric_name -> Metric
            confidence_level: Confidence level for statistical analysis
            comparison_method: Method for multiple comparison correction
        """
        self.base_metrics = metrics
        self.confidence_level = confidence_level
        self.comparison_method = comparison_method
        self.model_comparator = ModelComparison(
            confidence_level=confidence_level, correction_method=comparison_method
        )

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> dict[str, MetricResult]:
        """
        Evaluate a single model with statistical analysis.

        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on

        Returns:
            Dictionary of metric results with statistical information
        """
        if device is None:
            device = next(model.parameters()).device

        # Initialize metrics collector
        collector = MetricCollector(
            metrics=self.base_metrics,
            enable_statistical=True,
            confidence_level=self.confidence_level,
        )

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Assume batch is (features, targets) or similar
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    predictions = model(inputs)
                    collector.update(predictions, targets)
                else:
                    warnings.warn("Unexpected batch format, skipping batch")
                    continue

        return collector.compute()

    def compare_models(
        self,
        models: dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
        n_runs: int = 1,
    ) -> ComparisonResult:
        """
        Compare multiple models with statistical significance testing.

        Args:
            models: Dictionary of model_name -> model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            n_runs: Number of evaluation runs (for robustness)

        Returns:
            ComparisonResult with statistical analysis
        """
        model_results = {}

        for run in range(n_runs):
            for model_name, model in models.items():
                if model_name not in model_results:
                    model_results[model_name] = {}

                # Evaluate model
                run_results = self.evaluate_model(model, dataloader, device)

                # Store results for each metric
                for metric_name, metric_result in run_results.items():
                    if metric_name not in model_results[model_name]:
                        model_results[model_name][metric_name] = []

                    # Store scalar value for statistical comparison
                    if isinstance(metric_result, MetricResult):
                        model_results[model_name][metric_name].append(
                            metric_result.value
                        )
                    else:
                        model_results[model_name][metric_name].append(
                            float(metric_result)
                        )

        # Convert lists to numpy arrays for comparison
        for model_name in model_results:
            for metric_name in model_results[model_name]:
                model_results[model_name][metric_name] = np.array(
                    model_results[model_name][metric_name]
                )

        # Perform statistical comparison
        return self.model_comparator.compare_models(model_results)
