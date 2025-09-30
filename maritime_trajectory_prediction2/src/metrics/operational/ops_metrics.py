"""
Operational metrics for maritime trajectory prediction systems.

Implements metrics that matter for real-world deployment:
- Warning time distributions
- False alert rates
- System coverage analysis
- Throughput benchmarks
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WarningEvent:
    """Represents a warning event in the maritime system."""

    timestamp: float
    vessel_id: str
    threat_type: str  # 'collision', 'grounding', 'restricted_area'
    warning_time: float  # minutes before threshold breach
    false_positive: bool
    severity: str  # 'low', 'medium', 'high', 'critical'
    cpa_distance: float | None = None  # meters
    tcpa_time: float | None = None  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "vessel_id": self.vessel_id,
            "threat_type": self.threat_type,
            "warning_time": self.warning_time,
            "false_positive": self.false_positive,
            "severity": self.severity,
            "cpa_distance": self.cpa_distance,
            "tcpa_time": self.tcpa_time,
            "metadata": self.metadata,
        }


@dataclass
class WarningTimeStats:
    """Statistics about warning time distribution."""

    median_warning_time: float
    mean_warning_time: float
    std_warning_time: float
    p10_warning_time: float  # 10th percentile
    p25_warning_time: float  # 25th percentile
    p75_warning_time: float  # 75th percentile
    p90_warning_time: float  # 90th percentile
    min_warning_time: float
    max_warning_time: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "median_warning_time": self.median_warning_time,
            "mean_warning_time": self.mean_warning_time,
            "std_warning_time": self.std_warning_time,
            "p10_warning_time": self.p10_warning_time,
            "p25_warning_time": self.p25_warning_time,
            "p75_warning_time": self.p75_warning_time,
            "p90_warning_time": self.p90_warning_time,
            "min_warning_time": self.min_warning_time,
            "max_warning_time": self.max_warning_time,
            "n_samples": self.n_samples,
        }


@dataclass
class CoverageResult:
    """Results of coverage analysis."""

    total_vessels: int
    handled_vessels: int
    coverage_percentage: float
    unhandled_reasons: dict[str, int]  # Reason -> count
    vessel_type_coverage: dict[str, float]  # Vessel type -> coverage %
    spatial_coverage: dict[str, float] | None = None  # Region -> coverage %

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_vessels": self.total_vessels,
            "handled_vessels": self.handled_vessels,
            "coverage_percentage": self.coverage_percentage,
            "unhandled_reasons": self.unhandled_reasons,
            "vessel_type_coverage": self.vessel_type_coverage,
            "spatial_coverage": self.spatial_coverage,
        }


@dataclass
class ThroughputResult:
    """Results of throughput benchmarking."""

    vessels_per_second: float
    batches_per_second: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    gpu_utilization: float | None = None
    memory_usage_mb: float = 0.0
    batch_size: int = 1
    hardware_info: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vessels_per_second": self.vessels_per_second,
            "batches_per_second": self.batches_per_second,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage_mb": self.memory_usage_mb,
            "batch_size": self.batch_size,
            "hardware_info": self.hardware_info,
        }


class OperationalMetrics:
    """
    Metrics that maritime operators care about for real deployments.

    These metrics focus on practical deployment concerns like warning times,
    false alert rates, and system throughput.
    """

    def __init__(self, thresholds: dict[str, float] | None = None):
        """
        Initialize operational metrics calculator.

        Args:
            thresholds: Dictionary of operational thresholds
                - min_warning_time_minutes: Minimum acceptable warning time
                - max_false_alert_rate: Maximum acceptable false alert rate
                - min_coverage_percentage: Minimum coverage requirement
        """
        self.thresholds = thresholds or {
            "min_warning_time_minutes": 5.0,
            "max_false_alert_rate": 0.1,
            "min_coverage_percentage": 90.0,
        }

    def warning_time_distribution(
        self, events: list[WarningEvent], severity_filter: str | None = None
    ) -> WarningTimeStats:
        """
        Compute warning time statistics.

        Args:
            events: List of warning events
            severity_filter: Optional filter by severity level

        Returns:
            WarningTimeStats with distribution statistics
        """
        if not events:
            raise ValueError("No events provided for analysis")

        # Filter events if needed
        filtered_events = events
        if severity_filter:
            filtered_events = [e for e in events if e.severity == severity_filter]

        if not filtered_events:
            raise ValueError(f"No events with severity '{severity_filter}'")

        # Extract warning times
        warning_times = np.array([e.warning_time for e in filtered_events])

        # Compute statistics
        stats = WarningTimeStats(
            median_warning_time=np.median(warning_times),
            mean_warning_time=np.mean(warning_times),
            std_warning_time=np.std(warning_times),
            p10_warning_time=np.percentile(warning_times, 10),
            p25_warning_time=np.percentile(warning_times, 25),
            p75_warning_time=np.percentile(warning_times, 75),
            p90_warning_time=np.percentile(warning_times, 90),
            min_warning_time=np.min(warning_times),
            max_warning_time=np.max(warning_times),
            n_samples=len(warning_times),
        )

        # Check against thresholds
        if stats.median_warning_time < self.thresholds["min_warning_time_minutes"]:
            warnings.warn(
                f"Median warning time ({stats.median_warning_time:.1f} min) "
                f"below threshold ({self.thresholds['min_warning_time_minutes']} min)",
                stacklevel=2,
            )

        return stats

    def false_alert_rate(
        self,
        events: list[WarningEvent],
        time_window_hours: float = 24.0,
        by_severity: bool = False,
    ) -> dict[str, float]:
        """
        Compute false positive rate per time period.

        Critical metric for operational acceptance - too many false alerts
        lead to operator fatigue and system distrust.

        Args:
            events: List of warning events
            time_window_hours: Time window for rate calculation
            by_severity: If True, compute rates per severity level

        Returns:
            Dictionary with false alert rates
        """
        if not events:
            return {"overall_rate": 0.0, "total_events": 0, "false_positives": 0}

        # Count false positives
        false_positives = [e for e in events if e.false_positive]
        total_false = len(false_positives)
        total_events = len(events)

        # Overall rate
        overall_rate = total_false / total_events if total_events > 0 else 0.0

        result = {
            "overall_rate": overall_rate,
            "false_per_hour": total_false / time_window_hours,
            "total_events": total_events,
            "false_positives": total_false,
            "true_positives": total_events - total_false,
        }

        # Per-severity rates if requested
        if by_severity:
            severities = {e.severity for e in events}
            for severity in severities:
                severity_events = [e for e in events if e.severity == severity]
                severity_false = [e for e in severity_events if e.false_positive]
                if severity_events:
                    result[f"{severity}_rate"] = len(severity_false) / len(
                        severity_events
                    )

        # Check against threshold
        if overall_rate > self.thresholds["max_false_alert_rate"]:
            warnings.warn(
                f"False alert rate ({overall_rate:.2%}) exceeds "
                f"threshold ({self.thresholds['max_false_alert_rate']:.2%})",
                stacklevel=2,
            )

        return result

    def coverage_analysis(
        self,
        total_vessels: int,
        handled_vessels: int,
        unhandled_reasons: dict[str, int] | None = None,
        vessel_types: dict[str, tuple[int, int]] | None = None,
    ) -> CoverageResult:
        """
        Analyze what percentage of maritime traffic can be reliably handled.

        Args:
            total_vessels: Total number of vessels in dataset/region
            handled_vessels: Number successfully processed
            unhandled_reasons: Dictionary of reason -> count for failures
            vessel_types: Dict of vessel_type -> (total, handled) counts

        Returns:
            CoverageResult with detailed coverage analysis
        """
        if total_vessels <= 0:
            raise ValueError("Total vessels must be positive")

        coverage_pct = (handled_vessels / total_vessels) * 100

        # Compute per-vessel-type coverage if provided
        vessel_type_coverage = {}
        if vessel_types:
            for vtype, (v_total, v_handled) in vessel_types.items():
                if v_total > 0:
                    vessel_type_coverage[vtype] = (v_handled / v_total) * 100

        result = CoverageResult(
            total_vessels=total_vessels,
            handled_vessels=handled_vessels,
            coverage_percentage=coverage_pct,
            unhandled_reasons=unhandled_reasons or {},
            vessel_type_coverage=vessel_type_coverage,
        )

        # Check against threshold
        if coverage_pct < self.thresholds["min_coverage_percentage"]:
            warnings.warn(
                f"Coverage ({coverage_pct:.1f}%) below minimum "
                f"requirement ({self.thresholds['min_coverage_percentage']}%)",
                stacklevel=2,
            )

        return result

    def compute_severity(
        self, cpa_distance: float, tcpa_time: float, threat_type: str
    ) -> str:
        """
        Determine warning severity based on CPA/TCPA and threat type.

        Args:
            cpa_distance: Closest point of approach in meters
            tcpa_time: Time to CPA in seconds
            threat_type: Type of threat

        Returns:
            Severity level: 'critical', 'high', 'medium', or 'low'
        """
        # Critical thresholds
        CRITICAL_CPA = 100  # meters
        CRITICAL_TCPA = 120  # seconds (2 minutes)

        # High thresholds
        HIGH_CPA = 300  # meters
        HIGH_TCPA = 300  # seconds (5 minutes)

        # Medium thresholds
        MEDIUM_CPA = 500  # meters
        MEDIUM_TCPA = 600  # seconds (10 minutes)

        if threat_type == "collision":
            if cpa_distance < CRITICAL_CPA and tcpa_time < CRITICAL_TCPA:
                return "critical"
            elif cpa_distance < HIGH_CPA and tcpa_time < HIGH_TCPA:
                return "high"
            elif cpa_distance < MEDIUM_CPA and tcpa_time < MEDIUM_TCPA:
                return "medium"
            else:
                return "low"
        # Simplified for other threat types
        elif tcpa_time < CRITICAL_TCPA:
            return "critical"
        elif tcpa_time < HIGH_TCPA:
            return "high"
        elif tcpa_time < MEDIUM_TCPA:
            return "medium"
        else:
            return "low"

    def analyze_warning_patterns(
        self, events: list[WarningEvent], time_bins_hours: int = 1
    ) -> pd.DataFrame:
        """
        Analyze temporal patterns in warnings.

        Args:
            events: List of warning events
            time_bins_hours: Size of time bins for aggregation

        Returns:
            DataFrame with temporal analysis
        """
        if not events:
            return pd.DataFrame()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([e.to_dict() for e in events])

        # Convert timestamps to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

        # Create time bins
        df["time_bin"] = pd.to_datetime(df["datetime"]).dt.floor(f"{time_bins_hours}h")

        # Aggregate by time bin
        temporal_stats = (
            df.groupby("time_bin")
            .agg(
                {
                    "vessel_id": "count",  # Number of warnings
                    "false_positive": "mean",  # False positive rate
                    "warning_time": ["mean", "median", "std"],  # Warning time stats
                    "severity": lambda x: (x == "critical").mean(),  # Critical rate
                }
            )
            .round(3)
        )

        temporal_stats.columns = [
            "warning_count",
            "false_positive_rate",
            "mean_warning_time",
            "median_warning_time",
            "std_warning_time",
            "critical_rate",
        ]

        return temporal_stats
