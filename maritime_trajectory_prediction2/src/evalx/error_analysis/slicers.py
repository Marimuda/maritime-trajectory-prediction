"""
Performance slicing framework for maritime trajectory prediction error analysis.

This module provides tools to slice prediction errors by various maritime conditions
to understand model performance patterns and identify systematic failures.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..stats.bootstrap import BootstrapCI, BootstrapResult


@dataclass
class SliceConfig:
    """Configuration for a performance slice."""

    name: str
    slicer_func: Callable
    bins: list[str]
    description: str | None = None


@dataclass
class SliceResult:
    """Result of performance slicing analysis."""

    slice_name: str
    bin_name: str
    n_samples: int
    mean_error: float
    std_error: float
    bootstrap_ci: BootstrapResult | None = None
    per_sample_errors: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class ErrorSlicer:
    """
    Slice prediction performance by various maritime conditions.

    This class implements systematic performance analysis across different
    maritime conditions to identify where models fail and understand
    performance patterns.

    Key slicing dimensions:
    - Vessel type (cargo, tanker, fishing, passenger, other)
    - Traffic density (low, medium, high)
    - Distance to port (<5km, 5-20km, >20km)
    - Prediction horizon (step-wise analysis)

    Example:
        ```python
        slicer = ErrorSlicer()
        results = slicer.slice_errors(predictions, targets, metadata)

        # Access specific slice
        vessel_results = results['vessel_type']
        cargo_performance = vessel_results['cargo']
        ```
    """

    # Vessel type codes
    VESSEL_TYPE_FISHING = 30

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        """
        Initialize ErrorSlicer.

        Args:
            confidence_level: Confidence level for bootstrap intervals
            n_bootstrap: Number of bootstrap resamples
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = BootstrapCI(
            confidence_level=confidence_level, n_resamples=n_bootstrap
        )

        # Configure standard slicing dimensions
        self.slices = {
            "vessel_type": SliceConfig(
                name="Vessel Type",
                slicer_func=self._slice_by_vessel_type,
                bins=["cargo", "tanker", "fishing", "passenger", "other"],
                description="Performance by vessel type classification",
            ),
            "traffic_density": SliceConfig(
                name="Traffic Density",
                slicer_func=self._slice_by_traffic_density,
                bins=["low", "medium", "high"],
                description="Performance by local traffic density",
            ),
            "distance_to_port": SliceConfig(
                name="Port Proximity",
                slicer_func=self._slice_by_port_distance,
                bins=["<5km", "5-20km", ">20km"],
                description="Performance by distance to nearest port",
            ),
            "prediction_horizon": SliceConfig(
                name="Horizon Steps",
                slicer_func=self._slice_by_horizon,
                bins=[f"step_{i}" for i in range(1, 13)],
                description="Performance by prediction time horizon",
            ),
        }

    def slice_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any],
        error_metric: str = "mae",
        include_bootstrap: bool = True,
        slicing_dimensions: list[str] | None = None,
    ) -> dict[str, dict[str, SliceResult]]:
        """
        Slice prediction errors by maritime conditions.

        Args:
            predictions: Model predictions shape [n_samples, horizon, features]
            targets: Ground truth targets shape [n_samples, horizon, features]
            metadata: Dictionary containing slicing metadata
            error_metric: Error metric to use ('mae', 'mse', 'rmse')
            include_bootstrap: Whether to compute bootstrap confidence intervals
            slicing_dimensions: Specific dimensions to slice (None for all)

        Returns:
            Nested dictionary of slice results:
            {slice_name: {bin_name: SliceResult}}
        """
        # Validate inputs
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} != targets shape {targets.shape}"
            )

        # Compute per-sample errors
        errors = self._compute_errors(predictions, targets, error_metric)

        # Select slicing dimensions
        if slicing_dimensions is None:
            slicing_dimensions = list(self.slices.keys())

        # Validate requested dimensions
        invalid_dims = set(slicing_dimensions) - set(self.slices.keys())
        if invalid_dims:
            raise ValueError(f"Invalid slicing dimensions: {invalid_dims}")

        results = {}

        for slice_name in slicing_dimensions:
            slice_config = self.slices[slice_name]

            try:
                # Apply slicing function
                slice_assignments = slice_config.slicer_func(metadata, errors)

                # Compute results for each bin
                bin_results = {}
                for bin_name in slice_config.bins:
                    bin_mask = slice_assignments == bin_name

                    if np.any(bin_mask):
                        bin_errors = errors[bin_mask]

                        # Compute basic statistics
                        mean_error = np.mean(bin_errors)
                        std_error = np.std(bin_errors)
                        n_samples = len(bin_errors)

                        # Compute bootstrap CI if requested
                        bootstrap_result = None
                        MIN_BOOTSTRAP_SAMPLES = 10
                        if include_bootstrap and n_samples >= MIN_BOOTSTRAP_SAMPLES:
                            try:
                                bootstrap_result = self.bootstrap_ci.compute_ci(
                                    bin_errors, np.mean
                                )
                            except Exception as e:
                                warnings.warn(
                                    f"Bootstrap CI failed for {slice_name}/{bin_name}: {e}",
                                    stacklevel=2,
                                )

                        bin_results[bin_name] = SliceResult(
                            slice_name=slice_name,
                            bin_name=bin_name,
                            n_samples=n_samples,
                            mean_error=mean_error,
                            std_error=std_error,
                            bootstrap_ci=bootstrap_result,
                            per_sample_errors=bin_errors,
                            metadata={"slice_config": slice_config},
                        )
                    else:
                        # Empty bin
                        bin_results[bin_name] = SliceResult(
                            slice_name=slice_name,
                            bin_name=bin_name,
                            n_samples=0,
                            mean_error=np.nan,
                            std_error=np.nan,
                            bootstrap_ci=None,
                            per_sample_errors=None,
                            metadata={"slice_config": slice_config, "empty_bin": True},
                        )

                results[slice_name] = bin_results

            except Exception as e:
                warnings.warn(
                    f"Failed to compute slice {slice_name}: {e}", stacklevel=2
                )
                continue

        return results

    def _compute_errors(
        self, predictions: np.ndarray, targets: np.ndarray, metric: str
    ) -> np.ndarray:
        """
        Compute per-sample error values.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metric: Error metric ('mae', 'mse', 'rmse')

        Returns:
            Per-sample errors shape [n_samples]
        """
        # Compute differences
        diff = predictions - targets

        if metric == "mae":
            # Mean absolute error per sample
            errors = np.mean(np.abs(diff), axis=(1, 2))
        elif metric == "mse":
            # Mean squared error per sample
            errors = np.mean(diff**2, axis=(1, 2))
        elif metric == "rmse":
            # Root mean squared error per sample
            errors = np.sqrt(np.mean(diff**2, axis=(1, 2)))
        else:
            raise ValueError(f"Unknown error metric: {metric}")

        return errors

    def _slice_by_vessel_type(self, metadata: dict, errors: np.ndarray) -> np.ndarray:
        """Slice by vessel type classification."""
        if "vessel_type" not in metadata:
            raise KeyError("'vessel_type' not found in metadata")

        vessel_types = metadata["vessel_type"]
        if isinstance(vessel_types, list | np.ndarray):
            vessel_types = np.array(vessel_types)
        else:
            raise ValueError("vessel_type must be array-like")

        # Map vessel type codes to categories
        # Using standard AIS vessel type categories
        def map_vessel_type(vtype):
            if pd.isna(vtype) or vtype == 0:
                return "other"
            elif vtype in [70, 71, 72, 73, 74]:  # Cargo vessels
                return "cargo"
            elif vtype in [80, 81, 82, 83, 84, 85, 89]:  # Tankers
                return "tanker"
            elif vtype == ErrorSlicer.VESSEL_TYPE_FISHING:  # Fishing
                return "fishing"
            elif vtype in [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:  # Passenger
                return "passenger"
            else:
                return "other"

        # Apply mapping
        slice_assignments = np.array([map_vessel_type(vt) for vt in vessel_types])

        return slice_assignments

    def _slice_by_traffic_density(
        self, metadata: dict, errors: np.ndarray
    ) -> np.ndarray:
        """Slice by local traffic density."""
        if "traffic_density" in metadata:
            # Use provided traffic density if available
            traffic_density = np.array(metadata["traffic_density"])
        elif "vessel_count_5km" in metadata:
            # Compute from vessel count in 5km radius
            vessel_counts = np.array(metadata["vessel_count_5km"])

            # Define density thresholds (can be made configurable)
            low_threshold = np.percentile(vessel_counts, 33)
            high_threshold = np.percentile(vessel_counts, 67)

            traffic_density = np.where(
                vessel_counts <= low_threshold,
                "low",
                np.where(vessel_counts >= high_threshold, "high", "medium"),
            )
        else:
            raise KeyError(
                "Neither 'traffic_density' nor 'vessel_count_5km' found in metadata"
            )

        return traffic_density

    def _slice_by_port_distance(self, metadata: dict, errors: np.ndarray) -> np.ndarray:
        """Slice by distance to nearest port."""
        if "distance_to_port_km" not in metadata:
            raise KeyError("'distance_to_port_km' not found in metadata")

        distances = np.array(metadata["distance_to_port_km"])

        # Define distance bins
        NEAR_PORT_KM = 5.0
        MID_RANGE_KM = 20.0
        slice_assignments = np.where(
            distances < NEAR_PORT_KM,
            "<5km",
            np.where(distances < MID_RANGE_KM, "5-20km", ">20km"),
        )

        return slice_assignments

    def _slice_by_horizon(self, metadata: dict, errors: np.ndarray) -> np.ndarray:
        """Slice by prediction horizon step."""
        if "horizon_step" not in metadata:
            # If not provided, assume we want to analyze by horizon step
            # This requires different handling - we need step-wise errors
            raise NotImplementedError(
                "Horizon slicing requires step-wise error computation - use analyze_horizon_steps method"
            )

        horizon_steps = np.array(metadata["horizon_step"])
        slice_assignments = np.array([f"step_{step}" for step in horizon_steps])

        return slice_assignments

    def analyze_horizon_steps(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        error_metric: str = "mae",
        include_bootstrap: bool = True,
    ) -> dict[str, SliceResult]:
        """
        Analyze errors across prediction horizon steps.

        Args:
            predictions: Model predictions shape [n_samples, horizon, features]
            targets: Ground truth targets shape [n_samples, horizon, features]
            error_metric: Error metric to use
            include_bootstrap: Whether to compute bootstrap CIs

        Returns:
            Dictionary mapping step names to SliceResult objects
        """
        n_samples, horizon_length, n_features = predictions.shape

        results = {}

        for step in range(horizon_length):
            step_name = f"step_{step + 1}"

            # Extract predictions and targets for this step
            step_preds = predictions[:, step, :]
            step_targets = targets[:, step, :]

            # Compute step-specific errors
            if error_metric == "mae":
                step_errors = np.mean(np.abs(step_preds - step_targets), axis=1)
            elif error_metric == "mse":
                step_errors = np.mean((step_preds - step_targets) ** 2, axis=1)
            elif error_metric == "rmse":
                step_errors = np.sqrt(np.mean((step_preds - step_targets) ** 2, axis=1))
            else:
                raise ValueError(f"Unknown error metric: {error_metric}")

            # Compute statistics
            mean_error = np.mean(step_errors)
            std_error = np.std(step_errors)
            n_step_samples = len(step_errors)

            # Bootstrap CI
            bootstrap_result = None
            MIN_BOOTSTRAP_SAMPLES = 10
            if include_bootstrap and n_step_samples >= MIN_BOOTSTRAP_SAMPLES:
                try:
                    bootstrap_result = self.bootstrap_ci.compute_ci(
                        step_errors, np.mean
                    )
                except Exception as e:
                    warnings.warn(
                        f"Bootstrap CI failed for {step_name}: {e}", stacklevel=2
                    )

            results[step_name] = SliceResult(
                slice_name="prediction_horizon",
                bin_name=step_name,
                n_samples=n_step_samples,
                mean_error=mean_error,
                std_error=std_error,
                bootstrap_ci=bootstrap_result,
                per_sample_errors=step_errors,
                metadata={"horizon_step": step + 1},
            )

        return results

    def add_custom_slice(
        self,
        slice_name: str,
        slicer_func: Callable,
        bins: list[str],
        description: str | None = None,
    ) -> None:
        """
        Add a custom slicing dimension.

        Args:
            slice_name: Name for the slicing dimension
            slicer_func: Function to compute slice assignments
            bins: List of bin names
            description: Optional description
        """
        self.slices[slice_name] = SliceConfig(
            name=slice_name, slicer_func=slicer_func, bins=bins, description=description
        )

    def get_summary_statistics(
        self, slice_results: dict[str, dict[str, SliceResult]]
    ) -> pd.DataFrame:
        """
        Generate summary statistics table from slice results.

        Args:
            slice_results: Results from slice_errors method

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        for slice_name, bin_results in slice_results.items():
            for bin_name, result in bin_results.items():
                if result.n_samples > 0:
                    row = {
                        "slice_dimension": slice_name,
                        "bin": bin_name,
                        "n_samples": result.n_samples,
                        "mean_error": result.mean_error,
                        "std_error": result.std_error,
                        "ci_lower": result.bootstrap_ci.confidence_interval[0]
                        if result.bootstrap_ci
                        else None,
                        "ci_upper": result.bootstrap_ci.confidence_interval[1]
                        if result.bootstrap_ci
                        else None,
                    }
                    rows.append(row)

        return pd.DataFrame(rows)
