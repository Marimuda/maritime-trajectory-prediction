"""
Integration tests for M2.2 Maritime Domain Validation.

Tests the complete pipeline of CPA/TCPA calculations combined with
encounter classification and COLREGS compliance checking.
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
from maritime.cpa_tcpa import CPACalculator, CPAValidator, VesselState


class TestMaritimeIntegration:
    """Integration tests for complete maritime domain validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cpa_calculator = CPACalculator(
            cpa_warning_threshold=500.0, tcpa_warning_threshold=600.0
        )
        self.cpa_validator = CPAValidator(tolerance_distance=50.0, tolerance_time=30.0)
        self.encounter_classifier = EncounterClassifier()
        self.colregs_validator = COLREGSValidator()

    def test_head_on_collision_scenario(self):
        """Test complete head-on collision analysis pipeline."""
        # Two vessels approaching head-on
        vessel1 = VesselState(lat=55.0, lon=12.0, sog=15.0, cog=0.0)  # North
        vessel2 = VesselState(lat=55.01, lon=12.0, sog=12.0, cog=180.0)  # South

        # Step 1: CPA/TCPA Analysis
        cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        assert cpa_result.encounter_type == "approaching"
        assert cpa_result.tcpa_time > 0  # Approaching
        assert cpa_result.cpa_distance < 100  # Very close approach
        assert cpa_result.warning_level in ["critical", "high"]

        # Step 2: Encounter Classification
        encounter_result = self.encounter_classifier.classify_encounter(
            vessel1, vessel2, "vessel1", "vessel2"
        )

        assert encounter_result.encounter_type == EncounterType.HEAD_ON
        assert encounter_result.vessel1_action == COLREGSAction.BOTH_ALTER
        assert encounter_result.vessel2_action == COLREGSAction.BOTH_ALTER
        assert encounter_result.risk_level == "high"

        # Step 3: Add CPA result to encounter analysis
        encounter_result.cpa_result = cpa_result

        # Step 4: Validate compliance (both vessels alter course)
        vessel1_action = {"course_change": -20.0, "speed_change": -2.0}
        vessel2_action = {"course_change": 20.0, "speed_change": -1.5}

        compliance = self.colregs_validator.validate_encounter_compliance(
            encounter_result, vessel1_action, vessel2_action
        )

        assert compliance["vessel1_compliant"] is True
        assert compliance["vessel2_compliant"] is True
        assert compliance["overall_compliant"] is True

        # Step 5: Integration validation
        result_dict = encounter_result.to_dict()
        assert result_dict["cpa_distance"] == cpa_result.cpa_distance
        assert result_dict["tcpa_time"] == cpa_result.tcpa_time

    def test_crossing_encounter_with_cpa_analysis(self):
        """Test crossing encounter with integrated CPA/TCPA analysis."""
        # Vessel 1 heading north, vessel 2 crossing from starboard
        vessel1 = VesselState(lat=55.0, lon=12.0, sog=12.0, cog=0.0)  # North
        vessel2 = VesselState(lat=55.005, lon=12.01, sog=10.0, cog=270.0)  # West

        # CPA/TCPA Analysis
        cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Encounter Classification
        encounter_result = self.encounter_classifier.classify_encounter(
            vessel1, vessel2, "cargo_ship", "fishing_vessel"
        )
        encounter_result.cpa_result = cpa_result

        # Validate the integrated analysis
        assert encounter_result.encounter_type == EncounterType.CROSSING_STARBOARD
        assert encounter_result.vessel1_action == COLREGSAction.GIVE_WAY
        assert encounter_result.vessel2_action == COLREGSAction.STAND_ON

        # CPA should indicate collision risk
        if cpa_result.cpa_distance < 500.0 and cpa_result.tcpa_time < 600.0:
            assert cpa_result.warning_level in ["critical", "high", "medium"]

        # Test compliance scenarios
        # Scenario A: Vessel 1 gives way properly
        compliant_action_v1 = {"course_change": 30.0, "speed_change": -3.0}
        stand_on_action_v2 = {"course_change": 0.0, "speed_change": 0.0}

        compliance_good = self.colregs_validator.validate_encounter_compliance(
            encounter_result, compliant_action_v1, stand_on_action_v2
        )
        assert compliance_good["vessel1_compliant"] is True
        assert compliance_good["vessel2_compliant"] is True

        # Scenario B: Vessel 1 fails to give way
        non_compliant_action_v1 = {"course_change": 2.0, "speed_change": 0.0}

        compliance_bad = self.colregs_validator.validate_encounter_compliance(
            encounter_result, non_compliant_action_v1, stand_on_action_v2
        )
        assert compliance_bad["vessel1_compliant"] is False
        assert compliance_bad["overall_compliant"] is False

    def test_overtaking_scenario_integration(self):
        """Test overtaking scenario with complete maritime analysis."""
        # Fast vessel overtaking slower vessel
        slower_vessel = VesselState(lat=55.0, lon=12.0, sog=8.0, cog=45.0)
        faster_vessel = VesselState(lat=54.999, lon=11.999, sog=15.0, cog=45.0)

        # CPA/TCPA Analysis
        cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(
            faster_vessel, slower_vessel
        )

        # Encounter Classification
        encounter_result = self.encounter_classifier.classify_encounter(
            faster_vessel, slower_vessel, "fast_ferry", "cargo_vessel"
        )
        encounter_result.cpa_result = cpa_result

        # Validate overtaking detection and rules
        if encounter_result.encounter_type in [
            EncounterType.OVERTAKING_TAKEN,
            EncounterType.OVERTAKING_GIVEN,
        ]:
            # Check that appropriate vessel is designated to give way
            if encounter_result.encounter_type == EncounterType.OVERTAKING_TAKEN:
                assert encounter_result.vessel1_action == COLREGSAction.GIVE_WAY
                assert encounter_result.vessel2_action == COLREGSAction.STAND_ON
            else:  # OVERTAKING_GIVEN
                assert encounter_result.vessel1_action == COLREGSAction.STAND_ON
                assert encounter_result.vessel2_action == COLREGSAction.GIVE_WAY

            assert encounter_result.risk_level == "medium"

            # Test compliance
            give_way_action = {"course_change": 25.0, "speed_change": -2.0}
            stand_on_action = {"course_change": 0.0, "speed_change": 0.0}

            if encounter_result.vessel1_action == COLREGSAction.GIVE_WAY:
                actions = (give_way_action, stand_on_action)
            else:
                actions = (stand_on_action, give_way_action)

            compliance = self.colregs_validator.validate_encounter_compliance(
                encounter_result, actions[0], actions[1]
            )
            assert compliance["overall_compliant"] is True

    def test_vectorized_multi_encounter_analysis(self):
        """Test vectorized analysis of multiple simultaneous encounters."""
        # Create multiple encounter scenarios
        vessel1_list = [
            VesselState(lat=55.0, lon=12.0, sog=10.0, cog=0.0),  # Head-on scenario
            VesselState(lat=55.0, lon=12.0, sog=12.0, cog=90.0),  # Crossing scenario
            VesselState(lat=55.0, lon=12.0, sog=15.0, cog=45.0),  # Overtaking scenario
        ]

        vessel2_list = [
            VesselState(lat=55.01, lon=12.0, sog=10.0, cog=180.0),  # Head-on
            VesselState(lat=54.99, lon=12.01, sog=10.0, cog=0.0),  # Crossing
            VesselState(lat=55.005, lon=12.005, sog=8.0, cog=45.0),  # Being overtaken
        ]

        # Vectorized CPA/TCPA analysis
        cpa_results = self.cpa_calculator.calculate_cpa_tcpa_vectorized(
            vessel1_list, vessel2_list
        )

        assert len(cpa_results) == 3

        # Individual encounter classification
        encounter_results = []
        for i, (v1, v2) in enumerate(zip(vessel1_list, vessel2_list, strict=False)):
            encounter_result = self.encounter_classifier.classify_encounter(
                v1, v2, f"vessel1_{i}", f"vessel2_{i}"
            )
            encounter_result.cpa_result = cpa_results[i]
            encounter_results.append(encounter_result)

        # Validate each encounter
        expected_types = [
            EncounterType.HEAD_ON,
            [
                EncounterType.CROSSING_STARBOARD,
                EncounterType.CROSSING_PORT,
            ],  # Could be either
            [
                EncounterType.OVERTAKING_TAKEN,
                EncounterType.OVERTAKING_GIVEN,
                EncounterType.PARALLEL_SAME,
            ],
        ]

        for _i, (encounter, expected) in enumerate(
            zip(encounter_results, expected_types, strict=False)
        ):
            if isinstance(expected, list):
                assert encounter.encounter_type in expected
            else:
                assert encounter.encounter_type == expected

            # All should have valid CPA results
            assert encounter.cpa_result is not None
            assert encounter.cpa_result.cpa_distance >= 0
            assert encounter.cpa_result.encounter_type in [
                "approaching",
                "receding",
                "parallel",
            ]

    def test_warning_system_integration(self):
        """Test integrated warning system with CPA/TCPA and COLREGS."""
        # Critical collision scenario
        vessel1 = VesselState(lat=55.0, lon=12.0, sog=20.0, cog=90.0)  # Fast eastbound
        vessel2 = VesselState(
            lat=55.002, lon=12.003, sog=15.0, cog=180.0
        )  # Fast southbound

        # Analysis pipeline
        cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)
        encounter_result = self.encounter_classifier.classify_encounter(
            vessel1, vessel2, "high_speed_craft", "passenger_ferry"
        )
        encounter_result.cpa_result = cpa_result

        # Warning assessment
        warnings = []

        # CPA-based warnings
        if cpa_result.warning_level in ["critical", "high"]:
            warnings.append(
                {
                    "type": "collision_warning",
                    "severity": cpa_result.warning_level,
                    "cpa_distance": cpa_result.cpa_distance,
                    "tcpa_time": cpa_result.tcpa_time,
                }
            )

        # COLREGS-based warnings
        if encounter_result.risk_level in ["high", "medium"]:
            warnings.append(
                {
                    "type": "colregs_warning",
                    "encounter_type": encounter_result.encounter_type.value,
                    "vessel1_action": encounter_result.vessel1_action.value,
                    "vessel2_action": encounter_result.vessel2_action.value,
                }
            )

        # Validate warning system
        assert len(warnings) >= 1  # Should generate at least one warning

        collision_warnings = [w for w in warnings if w["type"] == "collision_warning"]
        colregs_warnings = [w for w in warnings if w["type"] == "colregs_warning"]

        # For close encounters, should have both types
        if cpa_result.cpa_distance < 200.0:
            assert len(collision_warnings) > 0

        if encounter_result.encounter_type != EncounterType.PARALLEL_SAME:
            assert len(colregs_warnings) > 0

    def test_prediction_validation_workflow(self):
        """Test workflow for validating trajectory predictions in maritime context."""
        # Scenario: Validate predicted vs actual encounter outcomes
        vessel1 = VesselState(lat=55.0, lon=12.0, sog=10.0, cog=45.0)
        vessel2 = VesselState(lat=55.005, lon=12.005, sog=12.0, cog=225.0)

        # Ground truth encounter
        actual_cpa = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)

        # Simulated prediction (with some error)
        predicted_cpa = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)
        predicted_cpa.cpa_distance += 25.0  # Add 25m error
        predicted_cpa.tcpa_time += 15.0  # Add 15s error

        # Validate prediction accuracy
        validation_result = self.cpa_validator.validate_prediction(
            predicted_cpa, actual_cpa
        )

        # Check validation metrics
        assert "distance_error_m" in validation_result
        assert "time_error_s" in validation_result
        assert "overall_accurate" in validation_result

        # Integrate with encounter classification
        encounter_result = self.encounter_classifier.classify_encounter(
            vessel1, vessel2, "predicted", "actual"
        )

        # Should classify as some type of encounter
        assert encounter_result.encounter_type != EncounterType.UNKNOWN
        assert encounter_result.risk_level in ["none", "low", "medium", "high"]

    def test_edge_case_integration(self):
        """Test integration with edge cases and boundary conditions."""
        # Edge case: Vessels very close together
        vessel1 = VesselState(lat=55.000000, lon=12.000000, sog=5.0, cog=0.0)
        vessel2 = VesselState(lat=55.000001, lon=12.000000, sog=5.0, cog=180.0)

        try:
            cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(vessel1, vessel2)
            encounter_result = self.encounter_classifier.classify_encounter(
                vessel1, vessel2
            )

            # Should handle gracefully without crashing
            assert cpa_result is not None
            assert encounter_result is not None
            assert cpa_result.cpa_distance >= 0

        except Exception as e:
            pytest.fail(f"Integration should handle edge cases gracefully: {e}")

        # Edge case: Stationary vessels
        vessel1_stationary = VesselState(lat=55.0, lon=12.0, sog=0.0, cog=0.0)
        vessel2_stationary = VesselState(lat=55.001, lon=12.0, sog=0.0, cog=90.0)

        try:
            cpa_result = self.cpa_calculator.calculate_cpa_tcpa_basic(
                vessel1_stationary, vessel2_stationary
            )
            encounter_result = self.encounter_classifier.classify_encounter(
                vessel1_stationary, vessel2_stationary
            )

            # Should handle stationary vessels
            assert cpa_result is not None
            assert encounter_result is not None

        except Exception as e:
            pytest.fail(f"Should handle stationary vessels: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
