"""Bootstrap confidence interval implementation using scipy.stats.bootstrap."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval computation."""

    confidence_interval: tuple[float, float]
    confidence_level: float
    method: str
    n_resamples: int
    statistic_value: float
    bootstrap_distribution: np.ndarray | None = None


class BootstrapCI:
    """Bootstrap confidence interval calculator using scipy.stats.bootstrap."""

    def __init__(
        self,
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        method: str = "BCa",
        random_state: int | None = None,
    ):
        """
        Initialize bootstrap CI calculator.

        Args:
            n_resamples: Number of bootstrap resamples (default: 9999)
            confidence_level: Confidence level (default: 0.95)
            method: Bootstrap method ('percentile', 'basic', 'BCa')
            random_state: Random seed for reproducibility
        """
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state

        # Validate method
        valid_methods = ["percentile", "basic", "BCa"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def compute_ci(
        self, data: np.ndarray | tuple, statistic: Callable = np.mean, **kwargs
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Data array or tuple of arrays for paired statistics
            statistic: Statistical function to bootstrap (default: mean)
            **kwargs: Additional arguments passed to scipy.stats.bootstrap

        Returns:
            BootstrapResult with confidence interval and metadata
        """
        # Ensure data is tuple of arrays for scipy.stats.bootstrap
        if isinstance(data, np.ndarray):
            data = (data,)
        elif not isinstance(data, tuple):
            data = tuple(data)

        # Convert method name for scipy
        method_map = {"percentile": "percentile", "basic": "basic", "BCa": "BCa"}
        scipy_method = method_map[self.method]

        # Configure bootstrap
        rng = np.random.default_rng(self.random_state)

        try:
            # Compute bootstrap CI using scipy
            res = stats.bootstrap(
                data,
                statistic,
                n_resamples=self.n_resamples,
                confidence_level=self.confidence_level,
                method=scipy_method,
                random_state=rng,
                **kwargs,
            )

            # Compute original statistic value
            statistic_value = statistic(*data)

            return BootstrapResult(
                confidence_interval=(
                    res.confidence_interval.low,
                    res.confidence_interval.high,
                ),
                confidence_level=self.confidence_level,
                method=self.method,
                n_resamples=self.n_resamples,
                statistic_value=statistic_value,
                bootstrap_distribution=res.bootstrap_distribution
                if hasattr(res, "bootstrap_distribution")
                else None,
            )

        except Exception as e:
            # Fallback to percentile method if BCa fails
            if self.method == "BCa":
                warnings.warn(
                    f"BCa method failed ({e}), falling back to percentile method"
                )
                fallback_ci = BootstrapCI(
                    n_resamples=self.n_resamples,
                    confidence_level=self.confidence_level,
                    method="percentile",
                    random_state=self.random_state,
                )
                return fallback_ci.compute_ci(data, statistic, **kwargs)
            else:
                raise

    def compare_means(
        self, group_a: np.ndarray, group_b: np.ndarray
    ) -> BootstrapResult:
        """Bootstrap CI for difference in means between two groups."""

        def mean_diff(a, b):
            return np.mean(a) - np.mean(b)

        return self.compute_ci((group_a, group_b), mean_diff)

    def compare_medians(
        self, group_a: np.ndarray, group_b: np.ndarray
    ) -> BootstrapResult:
        """Bootstrap CI for difference in medians between two groups."""

        def median_diff(a, b):
            return np.median(a) - np.median(b)

        return self.compute_ci((group_a, group_b), median_diff)


def bootstrap_ci(
    data: np.ndarray | tuple,
    statistic: Callable = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    method: str = "BCa",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Convenience function for computing bootstrap confidence intervals.

    Args:
        data: Data array or tuple of arrays
        statistic: Statistical function to bootstrap
        confidence_level: Confidence level (0.95 = 95% CI)
        n_resamples: Number of bootstrap resamples
        method: Bootstrap method ('percentile', 'basic', 'BCa')
        random_state: Random seed

    Returns:
        BootstrapResult with confidence interval and metadata

    Example:
        >>> import numpy as np
        >>> from evalx.stats import bootstrap_ci
        >>> data = np.random.normal(10, 2, 100)
        >>> result = bootstrap_ci(data, confidence_level=0.95)
        >>> print(f"Mean: {result.statistic_value:.2f}")
        >>> print(f"95% CI: ({result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f})")
    """
    calculator = BootstrapCI(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
    )
    return calculator.compute_ci(data, statistic)
