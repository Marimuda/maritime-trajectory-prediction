"""
Tests for operational metrics module.

Verifies warning time analysis, false alert computation, and coverage metrics.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.metrics.operational.ops_metrics import (
    CoverageResult,
    OperationalMetrics,
    WarningEvent,
    WarningTimeStats,
)


class TestWarningEvent:
    """Test WarningEvent dataclass."""

    def test_warning_event_creation(self):
        """Test creating a warning event."""
        event = WarningEvent(
            timestamp=1234567890.0,
            vessel_id="VESSEL001",
            threat_type="collision",
            warning_time=8.5,
            false_positive=False,
            severity="high",
            cpa_distance=250.0,
            tcpa_time=510.0,
        )

        assert event.timestamp == 1234567890.0
        assert event.vessel_id == "VESSEL001"
        assert event.threat_type == "collision"
        assert event.warning_time == 8.5
        assert not event.false_positive
        assert event.severity == "high"

    def test_warning_event_to_dict(self):
        """Test converting warning event to dictionary."""
        event = WarningEvent(
            timestamp=1234567890.0,
            vessel_id="VESSEL001",
            threat_type="collision",
            warning_time=8.5,
            false_positive=False,
            severity="high",
            metadata={"region": "north_sea"},
        )

        event_dict = event.to_dict()
        assert event_dict["vessel_id"] == "VESSEL001"
        assert event_dict["metadata"]["region"] == "north_sea"


class TestOperationalMetrics:
    """Test OperationalMetrics class."""

    @pytest.fixture
    def ops_metrics(self):
        """Create operational metrics instance."""
        return OperationalMetrics(
            thresholds={
                "min_warning_time_minutes": 5.0,
                "max_false_alert_rate": 0.15,
                "min_coverage_percentage": 85.0,
            }
        )

    @pytest.fixture
    def sample_events(self):
        """Create sample warning events."""
        events = []
        np.random.seed(42)

        for i in range(100):
            events.append(
                WarningEvent(
                    timestamp=1234567890.0 + i * 60,
                    vessel_id=f"VESSEL{i:03d}",
                    threat_type=np.random.choice(["collision", "grounding"]),
                    warning_time=np.random.uniform(2, 15),  # 2-15 minutes
                    false_positive=np.random.random() < 0.1,  # 10% false positives
                    severity=np.random.choice(["low", "medium", "high", "critical"]),
                    cpa_distance=np.random.uniform(50, 1000),
                    tcpa_time=np.random.uniform(60, 900),
                )
            )

        return events

    def test_warning_time_distribution(self, ops_metrics, sample_events):
        """Test warning time distribution analysis."""
        stats = ops_metrics.warning_time_distribution(sample_events)

        assert isinstance(stats, WarningTimeStats)
        assert stats.n_samples == 100
        assert (
            stats.min_warning_time
            <= stats.median_warning_time
            <= stats.max_warning_time
        )
        assert stats.p10_warning_time <= stats.p90_warning_time
        assert stats.mean_warning_time > 0
        assert stats.std_warning_time > 0

    def test_warning_time_distribution_with_filter(self, ops_metrics, sample_events):
        """Test warning time distribution with severity filter."""
        stats = ops_metrics.warning_time_distribution(
            sample_events, severity_filter="high"
        )

        high_events = [e for e in sample_events if e.severity == "high"]
        assert stats.n_samples == len(high_events)

    def test_warning_time_threshold_check(self, ops_metrics):
        """Test warning time threshold validation."""
        # Create events with short warning times
        short_warning_events = [
            WarningEvent(
                timestamp=1234567890.0 + i,
                vessel_id=f"VESSEL{i:03d}",
                threat_type="collision",
                warning_time=3.0,  # Below 5 minute threshold
                false_positive=False,
                severity="high",
            )
            for i in range(10)
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ops_metrics.warning_time_distribution(short_warning_events)
            assert len(w) == 1
            assert "below threshold" in str(w[0].message)

    def test_false_alert_rate(self, ops_metrics, sample_events):
        """Test false alert rate computation."""
        result = ops_metrics.false_alert_rate(sample_events, time_window_hours=24.0)

        assert "overall_rate" in result
        assert "false_per_hour" in result
        assert "total_events" in result
        assert "false_positives" in result
        assert "true_positives" in result

        assert result["total_events"] == 100
        assert result["false_positives"] + result["true_positives"] == 100
        assert 0 <= result["overall_rate"] <= 1

    def test_false_alert_rate_by_severity(self, ops_metrics, sample_events):
        """Test false alert rate broken down by severity."""
        result = ops_metrics.false_alert_rate(
            sample_events, time_window_hours=24.0, by_severity=True
        )

        # Should have rates for each severity level
        for severity in ["low", "medium", "high", "critical"]:
            assert f"{severity}_rate" in result
            assert 0 <= result[f"{severity}_rate"] <= 1

    def test_false_alert_threshold_check(self, ops_metrics):
        """Test false alert rate threshold validation."""
        # Create events with high false positive rate
        high_fp_events = [
            WarningEvent(
                timestamp=1234567890.0 + i,
                vessel_id=f"VESSEL{i:03d}",
                threat_type="collision",
                warning_time=10.0,
                false_positive=(i < 20),  # 20% false positives
                severity="high",
            )
            for i in range(100)
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ops_metrics.false_alert_rate(high_fp_events)
            assert len(w) == 1
            assert "exceeds threshold" in str(w[0].message)

    def test_coverage_analysis(self, ops_metrics):
        """Test coverage analysis."""
        result = ops_metrics.coverage_analysis(
            total_vessels=1000,
            handled_vessels=920,
            unhandled_reasons={"no_ais": 50, "out_of_range": 30},
            vessel_types={
                "cargo": (400, 380),
                "tanker": (200, 190),
                "fishing": (300, 250),
                "other": (100, 100),
            },
        )

        assert isinstance(result, CoverageResult)
        assert result.total_vessels == 1000
        assert result.handled_vessels == 920
        assert result.coverage_percentage == 92.0

        # Check unhandled reasons
        assert result.unhandled_reasons["no_ais"] == 50
        assert result.unhandled_reasons["out_of_range"] == 30

        # Check vessel type coverage
        assert result.vessel_type_coverage["cargo"] == 95.0
        assert result.vessel_type_coverage["tanker"] == 95.0
        assert result.vessel_type_coverage["fishing"] < 90.0  # Lower coverage

    def test_coverage_threshold_check(self, ops_metrics):
        """Test coverage threshold validation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ops_metrics.coverage_analysis(
                total_vessels=1000,
                handled_vessels=800,  # 80% - below 85% threshold
            )
            assert len(w) == 1
            assert "below minimum requirement" in str(w[0].message)

    def test_compute_severity(self, ops_metrics):
        """Test severity computation based on CPA/TCPA."""
        # Test critical severity
        assert ops_metrics.compute_severity(50, 60, "collision") == "critical"

        # Test high severity
        assert ops_metrics.compute_severity(200, 200, "collision") == "high"

        # Test medium severity
        assert ops_metrics.compute_severity(400, 400, "collision") == "medium"

        # Test low severity
        assert ops_metrics.compute_severity(1000, 1000, "collision") == "low"

    def test_analyze_warning_patterns(self, ops_metrics, sample_events):
        """Test temporal pattern analysis."""
        df = ops_metrics.analyze_warning_patterns(sample_events, time_bins_hours=1)

        assert isinstance(df, pd.DataFrame)
        assert "warning_count" in df.columns
        assert "false_positive_rate" in df.columns
        assert "mean_warning_time" in df.columns
        assert "critical_rate" in df.columns

    def test_empty_events_handling(self, ops_metrics):
        """Test handling of empty event lists."""
        # Warning time distribution should raise error
        with pytest.raises(ValueError, match="No events provided"):
            ops_metrics.warning_time_distribution([])

        # False alert rate should return zero rates
        result = ops_metrics.false_alert_rate([])
        assert result["overall_rate"] == 0.0
        assert result["total_events"] == 0

        # Pattern analysis should return empty DataFrame
        df = ops_metrics.analyze_warning_patterns([])
        assert df.empty

    def test_invalid_coverage_input(self, ops_metrics):
        """Test invalid input handling for coverage analysis."""
        with pytest.raises(ValueError, match="must be positive"):
            ops_metrics.coverage_analysis(
                total_vessels=0,  # Invalid
                handled_vessels=0,
            )


class TestWarningTimeStats:
    """Test WarningTimeStats dataclass."""

    def test_warning_time_stats_creation(self):
        """Test creating warning time statistics."""
        stats = WarningTimeStats(
            median_warning_time=8.0,
            mean_warning_time=8.5,
            std_warning_time=2.5,
            p10_warning_time=5.0,
            p25_warning_time=6.5,
            p75_warning_time=10.0,
            p90_warning_time=12.0,
            min_warning_time=3.0,
            max_warning_time=15.0,
            n_samples=100,
        )

        assert stats.median_warning_time == 8.0
        assert stats.n_samples == 100

    def test_warning_time_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = WarningTimeStats(
            median_warning_time=8.0,
            mean_warning_time=8.5,
            std_warning_time=2.5,
            p10_warning_time=5.0,
            p25_warning_time=6.5,
            p75_warning_time=10.0,
            p90_warning_time=12.0,
            min_warning_time=3.0,
            max_warning_time=15.0,
            n_samples=100,
        )

        stats_dict = stats.to_dict()
        assert stats_dict["median_warning_time"] == 8.0
        assert stats_dict["n_samples"] == 100
        assert len(stats_dict) == 10  # All fields


class TestIntegrationScenarios:
    """Test realistic operational scenarios."""

    def test_collision_avoidance_scenario(self):
        """Test metrics for collision avoidance system."""
        ops_metrics = OperationalMetrics()

        # Simulate collision warning events over 24 hours
        events = []
        for hour in range(24):
            for _ in range(np.random.poisson(5)):  # Average 5 warnings per hour
                events.append(
                    WarningEvent(
                        timestamp=hour * 3600 + np.random.uniform(0, 3600),
                        vessel_id=f"VESSEL{np.random.randint(1, 100):03d}",
                        threat_type="collision",
                        warning_time=np.random.gamma(2, 3),  # Gamma distribution
                        false_positive=np.random.random() < 0.08,  # 8% false positive
                        severity=np.random.choice(
                            ["low", "medium", "high", "critical"],
                            p=[0.5, 0.3, 0.15, 0.05],
                        ),
                        cpa_distance=np.random.exponential(300),
                        tcpa_time=np.random.uniform(60, 600),
                    )
                )

        # Analyze warning performance
        stats = ops_metrics.warning_time_distribution(events)
        false_rate = ops_metrics.false_alert_rate(events, time_window_hours=24)

        # Verify realistic ranges
        assert 3 <= stats.median_warning_time <= 12  # Reasonable warning time
        assert false_rate["overall_rate"] < 0.15  # Acceptable false positive rate
        assert false_rate["false_per_hour"] < 1.0  # Less than 1 false alert per hour

    def test_high_traffic_scenario(self):
        """Test coverage in high traffic areas."""
        ops_metrics = OperationalMetrics()

        # Simulate Dover Strait scenario (300+ vessels)
        total_vessels = 350
        handled_vessels = 315  # 90% coverage

        # Different vessel types in the strait
        vessel_distribution = {
            "cargo": (150, 140),  # Most common
            "tanker": (80, 75),
            "passenger": (30, 28),
            "fishing": (40, 32),  # Lower coverage for fishing vessels
            "other": (50, 40),
        }

        unhandled_reasons = {
            "no_ais": 15,  # Small vessels without AIS
            "poor_signal": 10,  # Signal quality issues
            "out_of_range": 10,  # Edge of coverage area
        }

        result = ops_metrics.coverage_analysis(
            total_vessels=total_vessels,
            handled_vessels=handled_vessels,
            unhandled_reasons=unhandled_reasons,
            vessel_types=vessel_distribution,
        )

        assert result.coverage_percentage == 90.0
        assert result.vessel_type_coverage["cargo"] > 90  # Good cargo coverage
        assert result.vessel_type_coverage["fishing"] < 85  # Lower fishing coverage
