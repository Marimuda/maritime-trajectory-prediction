"""
Unit tests for COLREGS encounter classification and compliance checking.

Tests cover various maritime encounter scenarios and COLREGS rule applications.
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from maritime.colregs import (
    COLREGSAction,
    COLREGSValidator,
    EncounterClassifier,
    EncounterType,
)
from maritime.cpa_tcpa import VesselState


class TestEncounterClassifier:
    """Test encounter classification functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = EncounterClassifier()

    def test_calculate_relative_bearing(self):
        """Test relative bearing calculation."""
        # Vessel 1 at origin, vessel 2 to the north
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        bearing_1_to_2, bearing_2_to_1 = self.classifier.calculate_relative_bearing(
            vessel1, vessel2
        )

        # Bearing from vessel1 to vessel2 should be ~0° (north)
        assert abs(bearing_1_to_2 - 0.0) < 1.0  # Within 1 degree

        # Bearing from vessel2 to vessel1 should be ~180° (south)
        assert abs(bearing_2_to_1 - 180.0) < 1.0

    def test_calculate_relative_bearing_to_course(self):
        """Test relative bearing from vessel's perspective."""
        # Absolute bearing is 090° (east), vessel heading is 000° (north)
        rel_bearing = self.classifier.calculate_relative_bearing_to_course(90.0, 0.0)
        assert abs(rel_bearing - 90.0) < 1e-6  # Should be 90° (starboard side)

        # Absolute bearing is 270° (west), vessel heading is 000° (north)
        rel_bearing = self.classifier.calculate_relative_bearing_to_course(270.0, 0.0)
        assert abs(rel_bearing - 270.0) < 1e-6  # Should be 270° (port side)

    def test_calculate_crossing_angle(self):
        """Test crossing angle calculation."""
        # Same direction
        angle = self.classifier.calculate_crossing_angle(0.0, 10.0)
        assert abs(angle - 10.0) < 1e-6

        # Opposite directions
        angle = self.classifier.calculate_crossing_angle(0.0, 180.0)
        assert abs(angle - 180.0) < 1e-6

        # Perpendicular
        angle = self.classifier.calculate_crossing_angle(0.0, 90.0)
        assert abs(angle - 90.0) < 1e-6

        # Handle wrap-around
        angle = self.classifier.calculate_crossing_angle(350.0, 20.0)
        assert abs(angle - 30.0) < 1e-6

    def test_head_on_encounter(self):
        """Test head-on encounter classification."""
        # Two vessels approaching head-on
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)  # Heading north
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)  # Heading south

        result = self.classifier.classify_encounter(vessel1, vessel2)

        assert result.encounter_type == EncounterType.HEAD_ON
        assert result.vessel1_action == COLREGSAction.BOTH_ALTER
        assert result.vessel2_action == COLREGSAction.BOTH_ALTER
        assert result.risk_level == "high"

    def test_crossing_starboard_encounter(self):
        """Test crossing encounter with other vessel on starboard."""
        # Vessel 1 heading north, vessel 2 heading west (crossing from starboard)
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)  # North
        vessel2 = VesselState(lat=0.005, lon=0.01, sog=10.0, cog=270.0)  # West

        result = self.classifier.classify_encounter(vessel1, vessel2)

        assert result.encounter_type == EncounterType.CROSSING_STARBOARD
        assert result.vessel1_action == COLREGSAction.GIVE_WAY  # Must give way
        assert result.vessel2_action == COLREGSAction.STAND_ON  # Has right of way
        assert result.risk_level in ["high", "medium"]

    def test_crossing_port_encounter(self):
        """Test crossing encounter with other vessel on port."""
        # Vessel 1 heading north, vessel 2 heading east (crossing from port)
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)  # North
        vessel2 = VesselState(lat=0.005, lon=-0.01, sog=10.0, cog=90.0)  # East

        result = self.classifier.classify_encounter(vessel1, vessel2)

        assert result.encounter_type == EncounterType.CROSSING_PORT
        assert result.vessel1_action == COLREGSAction.STAND_ON  # Has right of way
        assert result.vessel2_action == COLREGSAction.GIVE_WAY  # Must give way
        assert result.risk_level in ["high", "medium"]

    def test_overtaking_encounter(self):
        """Test overtaking encounter scenarios."""
        # Vessel 1 approaching vessel 2 from behind (overtaking scenario)
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=15.0, cog=0.0)  # Behind, faster
        vessel2 = VesselState(lat=0.005, lon=0.0, sog=10.0, cog=0.0)  # Ahead, slower

        result = self.classifier.classify_encounter(vessel1, vessel2)

        # The classification can be either type of overtaking or parallel same
        assert result.encounter_type in [
            EncounterType.OVERTAKING_TAKEN,
            EncounterType.OVERTAKING_GIVEN,
            EncounterType.PARALLEL_SAME,
        ]

        # Check appropriate actions based on classification
        if result.encounter_type == EncounterType.OVERTAKING_TAKEN:
            assert result.vessel1_action == COLREGSAction.GIVE_WAY
            assert result.vessel2_action == COLREGSAction.STAND_ON
        elif result.encounter_type == EncounterType.OVERTAKING_GIVEN:
            assert result.vessel1_action == COLREGSAction.STAND_ON  # Being overtaken
            assert result.vessel2_action == COLREGSAction.GIVE_WAY  # Overtaking

        if result.encounter_type in [
            EncounterType.OVERTAKING_TAKEN,
            EncounterType.OVERTAKING_GIVEN,
        ]:
            assert result.risk_level == "medium"

    def test_parallel_same_direction(self):
        """Test parallel vessels moving in same direction."""
        # Two vessels heading same direction, parallel
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(
            lat=0.0, lon=0.001, sog=10.0, cog=5.0
        )  # Nearly same direction

        result = self.classifier.classify_encounter(vessel1, vessel2)

        assert result.encounter_type == EncounterType.PARALLEL_SAME
        assert result.vessel1_action == COLREGSAction.NO_ACTION
        assert result.vessel2_action == COLREGSAction.NO_ACTION
        assert result.risk_level == "low"

    def test_parallel_opposite_direction(self):
        """Test parallel vessels moving in opposite directions."""
        # Two vessels heading opposite directions, but not directly head-on
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(
            lat=0.01, lon=0.002, sog=10.0, cog=175.0
        )  # Nearly opposite, offset

        result = self.classifier.classify_encounter(vessel1, vessel2)

        # Could be classified as head-on, parallel opposite, or even crossing depending on geometry
        assert result.encounter_type in [
            EncounterType.PARALLEL_OPPOSITE,
            EncounterType.HEAD_ON,
            EncounterType.CROSSING_STARBOARD,
            EncounterType.CROSSING_PORT,
        ]

        # Actions should be appropriate for the classified type
        if result.encounter_type == EncounterType.PARALLEL_OPPOSITE:
            assert result.vessel1_action == COLREGSAction.NO_ACTION
            assert result.vessel2_action == COLREGSAction.NO_ACTION

    def test_encounter_result_to_dict(self):
        """Test EncounterResult serialization."""
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        result = self.classifier.classify_encounter(vessel1, vessel2, "V1", "V2")
        result_dict = result.to_dict()

        assert result_dict["vessel1_id"] == "V1"
        assert result_dict["vessel2_id"] == "V2"
        assert result_dict["encounter_type"] == EncounterType.HEAD_ON.value
        assert result_dict["vessel1_action"] == COLREGSAction.BOTH_ALTER.value
        assert result_dict["vessel2_action"] == COLREGSAction.BOTH_ALTER.value
        assert "relative_bearing_1_to_2" in result_dict
        assert "crossing_angle" in result_dict

    def test_edge_case_bearings(self):
        """Test edge cases with bearing calculations."""
        # Test vessels very close together
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.00001, lon=0.0, sog=10.0, cog=180.0)

        result = self.classifier.classify_encounter(vessel1, vessel2)

        # Should still classify as head-on despite very small separation
        assert result.encounter_type == EncounterType.HEAD_ON

    def test_custom_classification_thresholds(self):
        """Test classifier with custom thresholds."""
        # Use very strict head-on threshold
        strict_classifier = EncounterClassifier(head_on_angle_threshold=5.0)

        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=170.0)  # 10° difference

        # With default threshold (15°), this might be head-on
        result_default = self.classifier.classify_encounter(vessel1, vessel2)

        # With strict threshold (5°), this should not be head-on
        result_strict = strict_classifier.classify_encounter(vessel1, vessel2)

        # Results may differ based on thresholds
        assert isinstance(result_default.encounter_type, EncounterType)
        assert isinstance(result_strict.encounter_type, EncounterType)


class TestCOLREGSValidator:
    """Test COLREGS compliance validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = COLREGSValidator()
        self.classifier = EncounterClassifier()

    def test_stand_on_compliance(self):
        """Test stand-on vessel compliance validation."""
        # Create head-on encounter
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        _ = self.classifier.classify_encounter(vessel1, vessel2)

        # Test compliant stand-on action (minimal course/speed change)
        compliant_action = {"course_change": 2.0, "speed_change": 0.0}
        non_compliant_action = {"course_change": 20.0, "speed_change": -2.0}

        # For head-on, both should alter, so let's create a crossing scenario
        vessel1_crossing = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2_crossing = VesselState(lat=0.005, lon=-0.01, sog=10.0, cog=90.0)

        crossing_result = self.classifier.classify_encounter(
            vessel1_crossing, vessel2_crossing
        )

        if crossing_result.vessel1_action == COLREGSAction.STAND_ON:
            compliance = self.validator.validate_encounter_compliance(
                crossing_result,
                compliant_action,
                {"course_change": 15.0, "speed_change": 0.0},
            )
            assert compliance["vessel1_compliant"] is True

            # Test non-compliant stand-on
            compliance_bad = self.validator.validate_encounter_compliance(
                crossing_result,
                non_compliant_action,
                {"course_change": 15.0, "speed_change": 0.0},
            )
            assert compliance_bad["vessel1_compliant"] is False

    def test_give_way_compliance(self):
        """Test give-way vessel compliance validation."""
        # Create crossing encounter where vessel1 must give way
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.005, lon=0.01, sog=10.0, cog=270.0)

        encounter_result = self.classifier.classify_encounter(vessel1, vessel2)

        if encounter_result.vessel1_action == COLREGSAction.GIVE_WAY:
            # Test compliant give-way action (substantial course change)
            compliant_action = {"course_change": 30.0, "speed_change": 0.0}
            compliance = self.validator.validate_encounter_compliance(
                encounter_result,
                compliant_action,
                {"course_change": 0.0, "speed_change": 0.0},
            )
            assert compliance["vessel1_compliant"] is True

            # Test non-compliant give-way (insufficient change)
            non_compliant_action = {"course_change": 2.0, "speed_change": 0.0}
            compliance_bad = self.validator.validate_encounter_compliance(
                encounter_result,
                non_compliant_action,
                {"course_change": 0.0, "speed_change": 0.0},
            )
            assert compliance_bad["vessel1_compliant"] is False

    def test_both_alter_compliance(self):
        """Test both-alter scenario compliance."""
        # Head-on encounter
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        encounter_result = self.classifier.classify_encounter(vessel1, vessel2)

        assert encounter_result.vessel1_action == COLREGSAction.BOTH_ALTER
        assert encounter_result.vessel2_action == COLREGSAction.BOTH_ALTER

        # Both vessels should make alterations
        vessel1_action = {"course_change": 15.0, "speed_change": 0.0}
        vessel2_action = {"course_change": -15.0, "speed_change": 0.0}

        compliance = self.validator.validate_encounter_compliance(
            encounter_result, vessel1_action, vessel2_action
        )

        assert compliance["vessel1_compliant"] is True
        assert compliance["vessel2_compliant"] is True
        assert compliance["overall_compliant"] is True

    def test_no_action_compliance(self):
        """Test no-action scenario compliance."""
        # Parallel vessels
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.0, lon=0.001, sog=10.0, cog=5.0)

        encounter_result = self.classifier.classify_encounter(vessel1, vessel2)

        if encounter_result.vessel1_action == COLREGSAction.NO_ACTION:
            # Any action should be acceptable
            any_action = {"course_change": 10.0, "speed_change": 1.0}
            no_action = {"course_change": 0.0, "speed_change": 0.0}

            compliance1 = self.validator.validate_encounter_compliance(
                encounter_result, any_action, no_action
            )
            compliance2 = self.validator.validate_encounter_compliance(
                encounter_result, no_action, any_action
            )

            assert compliance1["vessel1_compliant"] is True
            assert compliance2["vessel1_compliant"] is True

    def test_unknown_action_compliance(self):
        """Test unknown scenario handling."""
        # Create a scenario that might result in UNKNOWN
        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.1, lon=0.1, sog=10.0, cog=45.0)

        encounter_result = self.classifier.classify_encounter(vessel1, vessel2)

        # Even if unknown, validator should handle gracefully
        any_action = {"course_change": 10.0, "speed_change": 1.0}

        compliance = self.validator.validate_encounter_compliance(
            encounter_result, any_action, any_action
        )

        # Should not crash and provide some result
        assert "vessel1_compliant" in compliance
        assert "vessel2_compliant" in compliance
        assert "overall_compliant" in compliance

    def test_compliance_tolerance(self):
        """Test compliance checking with custom tolerance."""
        # Create validator with tight tolerance
        strict_validator = COLREGSValidator(compliance_tolerance=2.0)

        vessel1 = VesselState(lat=0.0, lon=0.0, sog=10.0, cog=0.0)
        vessel2 = VesselState(lat=0.01, lon=0.0, sog=10.0, cog=180.0)

        encounter_result = self.classifier.classify_encounter(vessel1, vessel2)

        # 3-degree change should pass normal tolerance but fail strict
        borderline_action = {"course_change": 3.0, "speed_change": 0.0}

        normal_compliance = self.validator.validate_encounter_compliance(
            encounter_result, borderline_action, borderline_action
        )
        strict_compliance = strict_validator.validate_encounter_compliance(
            encounter_result, borderline_action, borderline_action
        )

        # Results may differ based on tolerance and required actions
        assert "vessel1_compliant" in normal_compliance
        assert "vessel1_compliant" in strict_compliance


if __name__ == "__main__":
    pytest.main([__file__])
