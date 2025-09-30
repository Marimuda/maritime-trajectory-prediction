"""
Unit tests for CPA/TCPA calculations.

Tests cover various maritime encounter scenarios with known mathematical outcomes.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from maritime.cpa_tcpa import CPACalculator, CPAResult, CPAValidator, VesselState


class TestVesselState:
    """Test VesselState dataclass functionality."""

    def test_vessel_state_creation(self):
        """Test creating VesselState instance."""
        vessel = VesselState(lat=55.0, lon=12.0, sog=10.0, cog=45.0)

        assert vessel.lat == 55.0
        assert vessel.lon == 12.0
        assert vessel.sog == 10.0
        assert vessel.cog == 45.0
        assert vessel.timestamp is None

    def test_vessel_state_to_dict(self):
        """Test converting VesselState to dictionary."""
        vessel = VesselState(lat=55.0, lon=12.0, sog=10.0, cog=45.0)
        result = vessel.to_dict()

        expected = {
            "lat": 55.0,
            "lon": 12.0,
            "sog": 10.0,
            "cog": 45.0,
            "timestamp": None,
        }

        assert result == expected


class TestCPAResult:
    """Test CPAResult dataclass functionality."""

    def test_cpa_result_creation(self):
        """Test creating CPAResult instance."""
        result = CPAResult(
            cpa_distance=100.0,
            tcpa_time=300.0,
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        assert result.cpa_distance == 100.0
        assert result.tcpa_time == 300.0
        assert result.encounter_type == "approaching"
        assert result.warning_level == "medium"

    def test_cpa_result_to_dict(self):
        """Test converting CPAResult to dictionary."""
        result = CPAResult(
            cpa_distance=100.0,
            tcpa_time=300.0,
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        result_dict = result.to_dict()

        assert result_dict["cpa_distance_m"] == 100.0
        assert result_dict["tcpa_time_s"] == 300.0
        assert result_dict["tcpa_time_min"] == 5.0
        assert result_dict["encounter_type"] == "approaching"
        assert result_dict["warning_level"] == "medium"


class TestCPACalculator:
    """Test CPA/TCPA calculation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = CPACalculator(
            cpa_warning_threshold=500.0, tcpa_warning_threshold=600.0
        )

    def test_knots_to_mps_conversion(self):
        """Test knots to meters per second conversion."""
        # 1 knot = 0.514444 m/s
        assert abs(self.calculator._knots_to_mps(1.0) - 0.514444) < 1e-5
        assert abs(self.calculator._knots_to_mps(10.0) - 5.14444) < 1e-4

    def test_degrees_to_velocity_components(self):
        """Test conversion from course/speed to velocity components."""
        # North (0°): vx=0, vy=speed
        vx, vy = self.calculator._degrees_to_velocity_components(10.0, 0.0)
        assert abs(vx) < 1e-10  # Should be essentially 0
        assert abs(vy - 10.0) < 1e-10

        # East (90°): vx=speed, vy=0
        vx, vy = self.calculator._degrees_to_velocity_components(10.0, 90.0)
        assert abs(vx - 10.0) < 1e-10
        assert abs(vy) < 1e-10

        # South (180°): vx=0, vy=-speed
        vx, vy = self.calculator._degrees_to_velocity_components(10.0, 180.0)
        assert abs(vx) < 1e-10
        assert abs(vy + 10.0) < 1e-10

        # West (270°): vx=-speed, vy=0
        vx, vy = self.calculator._degrees_to_velocity_components(10.0, 270.0)
        assert abs(vx + 10.0) < 1e-10
        assert abs(vy) < 1e-10

    def test_warning_level_determination(self):
        """Test warning level determination logic."""
        # Critical warning: close distance, short time
        level = self.calculator._determine_warning_level(200.0, 300.0)
        assert level == "critical"

        # High warning
        level = self.calculator._determine_warning_level(350.0, 400.0)
        assert level == "high"

        # Medium warning
        level = self.calculator._determine_warning_level(450.0, 500.0)
        assert level == "medium"

        # Low warning
        level = self.calculator._determine_warning_level(700.0, 800.0)
        assert level == "low"

        # No warning: far distance
        level = self.calculator._determine_warning_level(1000.0, 300.0)
        assert level == "none"

        # No warning: receding (negative TCPA)
        level = self.calculator._determine_warning_level(200.0, -300.0)
        assert level == "none"

    @patch("maritime.cpa_tcpa.PYPROJ_AVAILABLE", False)
    def test_approximate_coordinate_conversion(self):
        """Test approximate coordinate conversion when pyproj is not available."""
        calc = CPACalculator()

        # Test conversion (approximate)
        x, y = calc._geographic_to_local(12.0, 55.0)  # lon, lat
        assert x != 0 and y != 0  # Should produce some values

        # Test inverse conversion
        lon, lat = calc._local_to_geographic(x, y)
        # Should be approximately the same (within reasonable tolerance for approximation)
        assert abs(lon - 12.0) < 0.1
        assert abs(lat - 55.0) < 0.1

    def test_head_on_collision_scenario(self):
        """Test head-on collision scenario with known outcome."""
        # Two vessels approaching head-on
        # Vessel 1: at (0, 0), moving North at 10 knots
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)

        # Vessel 2: at (0, 0.01), moving South at 10 knots (opposite direction)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Should be approaching
        assert result.encounter_type == "approaching"

        # TCPA should be positive (approaching)
        assert result.tcpa_time > 0

        # CPA distance should be very small (head-on collision)
        assert result.cpa_distance < 10.0  # Very close approach

    def test_perpendicular_crossing_scenario(self):
        """Test perpendicular crossing scenario."""
        # Vessel 1: at (0, 0), moving East at 10 knots
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=90.0)

        # Vessel 2: at (0.01, -0.01), moving North at 10 knots
        vessel2 = VesselState(lat=-0.01, lon=0.01, sog=10.0, cog=0.0)

        result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Should be approaching
        assert result.encounter_type == "approaching"

        # TCPA should be positive
        assert result.tcpa_time > 0

        # Should have some finite CPA distance
        assert result.cpa_distance > 0

    def test_parallel_same_speed_scenario(self):
        """Test parallel vessels with same speed."""
        # Vessel 1: at (0, 0), moving North at 10 knots
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)

        # Vessel 2: at (0, 0.001), moving North at 10 knots (parallel)
        vessel2 = VesselState(lat=0.0, lon=0.001, sog=10.0, cog=0.0)

        result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Should be parallel encounter
        assert result.encounter_type == "parallel"

        # TCPA should be 0 (maintaining constant distance)
        assert abs(result.tcpa_time) < 1e-6

        # CPA distance should be the current separation
        assert result.cpa_distance > 0

    def test_receding_vessels_scenario(self):
        """Test vessels moving away from each other."""
        # Vessel 1: at (0, 0), moving North at 10 knots
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)

        # Vessel 2: at (0, -0.01), moving South at 10 knots (moving away)
        vessel2 = VesselState(lat=-0.01, lon=0.0, sog=10.0, cog=180.0)

        result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Should be receding
        assert result.encounter_type == "receding"

        # TCPA should be negative (in the past)
        assert result.tcpa_time < 0

        # Warning level should be 'none' for receding vessels
        assert result.warning_level == "none"

    def test_stationary_vessel_scenario(self):
        """Test encounter with stationary vessel."""
        # Vessel 1: at (0, 0), stationary
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=0.0, cog=0.0)

        # Vessel 2: at (0, 0.01), moving towards vessel 1
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Should be approaching
        assert result.encounter_type == "approaching"

        # TCPA should be positive
        assert result.tcpa_time > 0

        # Should have very small CPA distance (direct approach)
        assert result.cpa_distance < 50.0


class TestCPAValidator:
    """Test CPA validation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = CPAValidator(tolerance_distance=50.0, tolerance_time=30.0)

    def test_accurate_prediction_validation(self):
        """Test validation of accurate prediction."""
        predicted = CPAResult(
            cpa_distance=100.0,
            tcpa_time=300.0,
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        actual = CPAResult(
            cpa_distance=110.0,
            tcpa_time=320.0,  # Within tolerance
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        result = self.validator.validate_prediction(predicted, actual)

        assert result["distance_error_m"] == 10.0
        assert result["time_error_s"] == 20.0
        assert result["distance_accurate"] is True
        assert result["time_accurate"] is True
        assert result["overall_accurate"] is True

    def test_inaccurate_prediction_validation(self):
        """Test validation of inaccurate prediction."""
        predicted = CPAResult(
            cpa_distance=100.0,
            tcpa_time=300.0,
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        actual = CPAResult(
            cpa_distance=200.0,
            tcpa_time=400.0,  # Outside tolerance
            cpa_lat1=55.0,
            cpa_lon1=12.0,
            cpa_lat2=55.1,
            cpa_lon2=12.1,
            encounter_type="approaching",
            warning_level="medium",
        )

        result = self.validator.validate_prediction(predicted, actual)

        assert result["distance_error_m"] == 100.0
        assert result["time_error_s"] == 100.0
        assert result["distance_accurate"] is False
        assert result["time_accurate"] is False
        assert result["overall_accurate"] is False

    def test_validation_statistics_computation(self):
        """Test computation of aggregate validation statistics."""
        validation_results = [
            {
                "distance_error_m": 10.0,
                "time_error_s": 20.0,
                "distance_accurate": True,
                "time_accurate": True,
                "overall_accurate": True,
            },
            {
                "distance_error_m": 100.0,
                "time_error_s": 100.0,
                "distance_accurate": False,
                "time_accurate": False,
                "overall_accurate": False,
            },
        ]

        stats = self.validator.compute_validation_statistics(validation_results)

        assert stats["distance_accuracy_rate"] == 0.5
        assert stats["time_accuracy_rate"] == 0.5
        assert stats["overall_accuracy_rate"] == 0.5
        assert stats["mean_distance_error_m"] == 55.0
        assert stats["mean_time_error_s"] == 60.0

    def test_empty_validation_statistics(self):
        """Test handling of empty validation results."""
        stats = self.validator.compute_validation_statistics([])
        assert stats == {}


class TestCPACalculatorVectorized:
    """Test vectorized CPA/TCPA calculations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = CPACalculator(
            cpa_warning_threshold=500.0, tcpa_warning_threshold=600.0
        )

    def test_empty_vessel_arrays(self):
        """Test handling of empty vessel arrays."""
        result = self.calculator.calculate_cpa_tcpa_vectorized([], [])
        assert result == []

    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched array lengths."""
        vessels1 = [VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)]
        vessels2 = []

        with pytest.raises(ValueError, match="must have the same length"):
            self.calculator.calculate_cpa_tcpa_vectorized(vessels1, vessels2)

    def test_single_pair_vectorized_vs_basic(self):
        """Test that vectorized calculation matches basic for single pair."""
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        # Basic calculation
        basic_result = self.calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Vectorized calculation
        vectorized_results = self.calculator.calculate_cpa_tcpa_vectorized(
            [vessel1], [vessel2]
        )

        assert len(vectorized_results) == 1
        vectorized_result = vectorized_results[0]

        # Results should be very close (within numerical precision)
        assert abs(basic_result.cpa_distance - vectorized_result.cpa_distance) < 1e-6
        assert abs(basic_result.tcpa_time - vectorized_result.tcpa_time) < 1e-6
        assert basic_result.encounter_type == vectorized_result.encounter_type
        assert basic_result.warning_level == vectorized_result.warning_level

    def test_multiple_pairs_vectorized(self):
        """Test vectorized calculation with multiple vessel pairs."""
        # Create multiple test scenarios
        vessels1 = [
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # Head-on
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=90.0),  # Crossing
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # Parallel same speed
        ]

        vessels2 = [
            VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0),  # Head-on
            VesselState(lat=-0.01, lon=0.01, sog=10.0, cog=0.0),  # Crossing
            VesselState(lat=0.0, lon=0.001, sog=10.0, cog=0.0),  # Parallel same speed
        ]

        results = self.calculator.calculate_cpa_tcpa_vectorized(vessels1, vessels2)

        assert len(results) == 3

        # Check that each result has proper structure
        for result in results:
            assert isinstance(result, CPAResult)
            assert result.cpa_distance >= 0
            assert result.encounter_type in ["approaching", "receding", "parallel"]
            assert result.warning_level in ["none", "low", "medium", "high", "critical"]

        # Specific checks for known scenarios
        assert results[0].encounter_type == "approaching"  # Head-on
        assert results[1].encounter_type == "approaching"  # Crossing
        assert results[2].encounter_type == "parallel"  # Parallel same speed

    def test_vectorized_vs_basic_consistency(self):
        """Test that vectorized results match basic calculations for multiple pairs."""
        # Create test scenarios
        vessels1 = [
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),
            VesselState(lat=0.0, lon=0.0, sog=15.0, cog=45.0),
            VesselState(lat=0.01, lon=0.01, sog=5.0, cog=270.0),
        ]

        vessels2 = [
            VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0),
            VesselState(lat=-0.01, lon=0.01, sog=12.0, cog=0.0),
            VesselState(lat=0.0, lon=0.0, sog=8.0, cog=90.0),
        ]

        # Calculate using both methods
        basic_results = []
        for v1, v2 in zip(vessels1, vessels2, strict=False):
            basic_results.append(self.calculator.calculate_cpa_tcpa_basic(v1, v2))

        vectorized_results = self.calculator.calculate_cpa_tcpa_vectorized(
            vessels1, vessels2
        )

        # Compare results
        assert len(basic_results) == len(vectorized_results)

        for basic, vectorized in zip(basic_results, vectorized_results, strict=False):
            # Allow for small numerical differences
            assert abs(basic.cpa_distance - vectorized.cpa_distance) < 1e-3
            assert abs(basic.tcpa_time - vectorized.tcpa_time) < 1e-3
            assert basic.encounter_type == vectorized.encounter_type
            assert basic.warning_level == vectorized.warning_level

    def test_vectorized_warning_levels(self):
        """Test vectorized warning level determination."""
        # Create scenarios with known warning levels
        vessels1 = [
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # Critical
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # High
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # Medium
            VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0),  # None (receding)
        ]

        vessels2 = [
            VesselState(lat=0.002, lon=0.0, sog=10.0, cog=180.0),  # Very close approach
            VesselState(lat=0.003, lon=0.0, sog=10.0, cog=180.0),  # Close approach
            VesselState(lat=0.004, lon=0.0, sog=10.0, cog=180.0),  # Medium approach
            VesselState(lat=-0.01, lon=0.0, sog=10.0, cog=180.0),  # Receding
        ]

        results = self.calculator.calculate_cpa_tcpa_vectorized(vessels1, vessels2)

        # Check warning levels (approximately)
        # All approaching vessels should have some level of warning
        assert results[0].warning_level in ["critical", "high"]  # Very close
        assert results[1].warning_level in [
            "critical",
            "high",
            "medium",
        ]  # Close (more flexible)
        assert results[2].warning_level in [
            "critical",
            "high",
            "medium",
            "low",
        ]  # Medium (flexible)
        assert results[3].warning_level == "none"  # Receding

    @patch("maritime.cpa_tcpa.PYPROJ_AVAILABLE", False)
    def test_vectorized_without_pyproj(self):
        """Test vectorized calculations work without pyproj."""
        calc = CPACalculator()

        vessels1 = [VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)]
        vessels2 = [VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)]

        results = calc.calculate_cpa_tcpa_vectorized(vessels1, vessels2)

        assert len(results) == 1
        assert results[0].encounter_type == "approaching"
        # For approximate calculations, CPA distance might be very small but should be finite
        assert results[0].cpa_distance >= 0  # Allow zero for approximate calculations


if __name__ == "__main__":
    pytest.main([__file__])
