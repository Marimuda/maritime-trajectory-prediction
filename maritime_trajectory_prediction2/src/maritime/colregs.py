"""
COLREGS (International Regulations for Preventing Collisions at Sea) implementation.

This module provides encounter classification and basic COLREGS compliance checking
for maritime vessel interactions.

Key Components:
- EncounterClassifier: Classifies vessel encounters (head-on, crossing, overtaking)
- COLREGSValidator: Basic COLREGS compliance checking
- Relative bearing and geometric calculations for maritime scenarios
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .cpa_tcpa import CPAResult, VesselState


class EncounterType(Enum):
    """Types of vessel encounters according to COLREGS."""

    HEAD_ON = "head_on"
    CROSSING_STARBOARD = "crossing_starboard"  # Other vessel crossing from starboard
    CROSSING_PORT = "crossing_port"  # Other vessel crossing from port
    OVERTAKING_GIVEN = "overtaking_given"  # This vessel being overtaken
    OVERTAKING_TAKEN = "overtaking_taken"  # This vessel overtaking
    PARALLEL_SAME = "parallel_same"  # Same direction, parallel
    PARALLEL_OPPOSITE = "parallel_opposite"  # Opposite direction, parallel
    UNKNOWN = "unknown"


class COLREGSAction(Enum):
    """COLREGS-defined actions for encounters."""

    STAND_ON = "stand_on"  # Maintain course and speed
    GIVE_WAY = "give_way"  # Alter course/speed to avoid
    BOTH_ALTER = "both_alter"  # Both vessels should alter (head-on)
    NO_ACTION = "no_action"  # No specific action required
    UNKNOWN = "unknown"


@dataclass
class EncounterResult:
    """Result of encounter classification and COLREGS analysis."""

    vessel1_id: str
    vessel2_id: str
    encounter_type: EncounterType
    relative_bearing_1_to_2: float  # Bearing from vessel 1 to vessel 2 (degrees)
    relative_bearing_2_to_1: float  # Bearing from vessel 2 to vessel 1 (degrees)
    crossing_angle: float  # Angle between vessel courses (degrees)
    vessel1_action: COLREGSAction  # Required action for vessel 1
    vessel2_action: COLREGSAction  # Required action for vessel 2
    risk_level: str  # 'high', 'medium', 'low', 'none'
    cpa_result: CPAResult | None = None  # Associated CPA/TCPA analysis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vessel1_id": self.vessel1_id,
            "vessel2_id": self.vessel2_id,
            "encounter_type": self.encounter_type.value,
            "relative_bearing_1_to_2": self.relative_bearing_1_to_2,
            "relative_bearing_2_to_1": self.relative_bearing_2_to_1,
            "crossing_angle": self.crossing_angle,
            "vessel1_action": self.vessel1_action.value,
            "vessel2_action": self.vessel2_action.value,
            "risk_level": self.risk_level,
            "cpa_distance": self.cpa_result.cpa_distance if self.cpa_result else None,
            "tcpa_time": self.cpa_result.tcpa_time if self.cpa_result else None,
        }


class EncounterClassifier:
    """
    Classifies maritime vessel encounters according to COLREGS definitions.

    Uses relative bearings and course angles to determine encounter type
    and applicable maritime rules.
    """

    # Bearing angle constants
    STARBOARD_MIN = 0
    STARBOARD_MAX = 180
    PORT_MIN = 180
    PORT_MAX = 360

    # Crossing risk angle thresholds
    CROSSING_HIGH_RISK_MIN_1 = 60
    CROSSING_HIGH_RISK_MAX_1 = 120
    CROSSING_HIGH_RISK_MIN_2 = 240
    CROSSING_HIGH_RISK_MAX_2 = 300

    def __init__(
        self,
        head_on_angle_threshold: float = 15.0,  # degrees
        crossing_angle_range: tuple[float, float] = (15.0, 165.0),  # degrees
        overtaking_angle_threshold: float = 135.0,
    ):  # degrees
        """
        Initialize encounter classifier.

        Args:
            head_on_angle_threshold: Maximum angle difference for head-on classification
            crossing_angle_range: Angle range (min, max) for crossing encounters
            overtaking_angle_threshold: Minimum angle from behind for overtaking
        """
        self.head_on_threshold = head_on_angle_threshold
        self.crossing_min_angle, self.crossing_max_angle = crossing_angle_range
        self.overtaking_threshold = overtaking_angle_threshold

    def calculate_relative_bearing(
        self, vessel1: VesselState, vessel2: VesselState
    ) -> tuple[float, float]:
        """
        Calculate relative bearings between two vessels.

        Args:
            vessel1: First vessel state
            vessel2: Second vessel state

        Returns:
            Tuple of (bearing_1_to_2, bearing_2_to_1) in degrees [0, 360)
        """
        # Calculate bearing from vessel1 to vessel2
        lat1, lon1 = np.radians(vessel1.lat), np.radians(vessel1.lon)
        lat2, lon2 = np.radians(vessel2.lat), np.radians(vessel2.lon)

        # Calculate bearing using spherical trigonometry
        dlon = lon2 - lon1

        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing_1_to_2 = np.degrees(np.arctan2(y, x))
        bearing_1_to_2 = (bearing_1_to_2 + 360) % 360

        # Calculate reverse bearing
        bearing_2_to_1 = (bearing_1_to_2 + 180) % 360

        return bearing_1_to_2, bearing_2_to_1

    def calculate_relative_bearing_to_course(
        self, absolute_bearing: float, vessel_course: float
    ) -> float:
        """
        Calculate relative bearing from vessel's perspective.

        Args:
            absolute_bearing: Absolute bearing to target (degrees)
            vessel_course: Vessel's course over ground (degrees)

        Returns:
            Relative bearing from vessel's bow [0, 360) degrees
            0° = ahead, 90° = starboard, 180° = behind, 270° = port
        """
        relative_bearing = (absolute_bearing - vessel_course + 360) % 360
        return relative_bearing

    def calculate_crossing_angle(
        self, vessel1_course: float, vessel2_course: float
    ) -> float:
        """
        Calculate the crossing angle between two vessel courses.

        Args:
            vessel1_course: Course of vessel 1 (degrees)
            vessel2_course: Course of vessel 2 (degrees)

        Returns:
            Crossing angle in degrees [0, 180]
        """
        angle_diff = abs(vessel1_course - vessel2_course)
        crossing_angle = min(angle_diff, 360 - angle_diff)
        return crossing_angle

    def classify_encounter(
        self,
        vessel1: VesselState,
        vessel2: VesselState,
        vessel1_id: str = "vessel1",
        vessel2_id: str = "vessel2",
    ) -> EncounterResult:
        """
        Classify encounter between two vessels according to COLREGS.

        Args:
            vessel1: State of first vessel
            vessel2: State of second vessel
            vessel1_id: Identifier for vessel 1
            vessel2_id: Identifier for vessel 2

        Returns:
            EncounterResult with classification and COLREGS analysis
        """
        # Calculate bearings and angles
        bearing_1_to_2, bearing_2_to_1 = self.calculate_relative_bearing(
            vessel1, vessel2
        )

        # Relative bearings from each vessel's perspective
        rel_bearing_1_to_2 = self.calculate_relative_bearing_to_course(
            bearing_1_to_2, vessel1.cog
        )
        rel_bearing_2_to_1 = self.calculate_relative_bearing_to_course(
            bearing_2_to_1, vessel2.cog
        )

        # Crossing angle between courses
        crossing_angle = self.calculate_crossing_angle(vessel1.cog, vessel2.cog)

        # Classify encounter type
        encounter_type = self._classify_encounter_type(
            rel_bearing_1_to_2, rel_bearing_2_to_1, crossing_angle
        )

        # Determine COLREGS actions
        vessel1_action, vessel2_action = self._determine_colregs_actions(
            encounter_type, rel_bearing_1_to_2, rel_bearing_2_to_1
        )

        # Assess risk level (simplified)
        risk_level = self._assess_risk_level(
            encounter_type, rel_bearing_1_to_2, rel_bearing_2_to_1
        )

        return EncounterResult(
            vessel1_id=vessel1_id,
            vessel2_id=vessel2_id,
            encounter_type=encounter_type,
            relative_bearing_1_to_2=rel_bearing_1_to_2,
            relative_bearing_2_to_1=rel_bearing_2_to_1,
            crossing_angle=crossing_angle,
            vessel1_action=vessel1_action,
            vessel2_action=vessel2_action,
            risk_level=risk_level,
        )

    def _classify_encounter_type(
        self,
        rel_bearing_1_to_2: float,
        rel_bearing_2_to_1: float,
        crossing_angle: float,
    ) -> EncounterType:
        """Classify encounter type based on relative bearings and crossing angle."""

        # Overtaking scenarios (check first, as they can have similar courses)
        if self._is_overtaking(rel_bearing_1_to_2):
            return EncounterType.OVERTAKING_TAKEN  # Vessel 1 overtaking vessel 2
        elif self._is_overtaking(rel_bearing_2_to_1):
            return (
                EncounterType.OVERTAKING_GIVEN
            )  # Vessel 1 being overtaken by vessel 2

        # Head-on: vessels approaching nearly head-on with both seeing each other ahead
        if (
            crossing_angle >= (180 - self.head_on_threshold)
            and self._is_nearly_ahead(rel_bearing_1_to_2)
            and self._is_nearly_ahead(rel_bearing_2_to_1)
        ):
            return EncounterType.HEAD_ON

        # Parallel courses (same direction)
        if crossing_angle <= self.head_on_threshold:
            return EncounterType.PARALLEL_SAME

        # Parallel courses (opposite direction) - but not head-on
        if crossing_angle >= (180 - self.head_on_threshold) and not (
            self._is_nearly_ahead(rel_bearing_1_to_2)
            and self._is_nearly_ahead(rel_bearing_2_to_1)
        ):
            return EncounterType.PARALLEL_OPPOSITE

        # Crossing scenarios
        if self.crossing_min_angle <= crossing_angle <= self.crossing_max_angle:
            # Determine which side the other vessel is crossing from
            if self._is_starboard_side(rel_bearing_1_to_2):
                return EncounterType.CROSSING_STARBOARD
            elif self._is_port_side(rel_bearing_1_to_2):
                return EncounterType.CROSSING_PORT

        return EncounterType.UNKNOWN

    def _is_nearly_ahead(self, relative_bearing: float) -> bool:
        """Check if relative bearing is nearly straight ahead."""
        return relative_bearing <= self.head_on_threshold or relative_bearing >= (
            360 - self.head_on_threshold
        )

    def _is_overtaking(self, relative_bearing: float) -> bool:
        """Check if relative bearing indicates overtaking scenario."""
        return (
            self.overtaking_threshold
            <= relative_bearing
            <= (360 - self.overtaking_threshold)
        )

    def _is_starboard_side(self, relative_bearing: float) -> bool:
        """Check if relative bearing is on starboard side."""
        return self.STARBOARD_MIN < relative_bearing < self.STARBOARD_MAX

    def _is_port_side(self, relative_bearing: float) -> bool:
        """Check if relative bearing is on port side."""
        return self.PORT_MIN < relative_bearing < self.PORT_MAX

    def _determine_colregs_actions(
        self,
        encounter_type: EncounterType,
        rel_bearing_1_to_2: float,
        rel_bearing_2_to_1: float,
    ) -> tuple[COLREGSAction, COLREGSAction]:
        """Determine required COLREGS actions for each vessel."""

        if encounter_type == EncounterType.HEAD_ON:
            return COLREGSAction.BOTH_ALTER, COLREGSAction.BOTH_ALTER

        elif encounter_type == EncounterType.CROSSING_STARBOARD:
            # Other vessel on starboard side has right of way
            return COLREGSAction.GIVE_WAY, COLREGSAction.STAND_ON

        elif encounter_type == EncounterType.CROSSING_PORT:
            # This vessel has right of way
            return COLREGSAction.STAND_ON, COLREGSAction.GIVE_WAY

        elif encounter_type == EncounterType.OVERTAKING_GIVEN:
            # Being overtaken - maintain course
            return COLREGSAction.STAND_ON, COLREGSAction.GIVE_WAY

        elif encounter_type == EncounterType.OVERTAKING_TAKEN:
            # Overtaking - give way
            return COLREGSAction.GIVE_WAY, COLREGSAction.STAND_ON

        elif encounter_type in [
            EncounterType.PARALLEL_SAME,
            EncounterType.PARALLEL_OPPOSITE,
        ]:
            return COLREGSAction.NO_ACTION, COLREGSAction.NO_ACTION

        else:
            return COLREGSAction.UNKNOWN, COLREGSAction.UNKNOWN

    def _assess_risk_level(
        self,
        encounter_type: EncounterType,
        rel_bearing_1_to_2: float,
        rel_bearing_2_to_1: float,
    ) -> str:
        """Assess collision risk level based on encounter type and geometry."""

        if encounter_type == EncounterType.HEAD_ON:
            return "high"

        elif encounter_type in [
            EncounterType.CROSSING_STARBOARD,
            EncounterType.CROSSING_PORT,
        ]:
            # Higher risk for closer to perpendicular crossings
            if (
                self.CROSSING_HIGH_RISK_MIN_1
                <= rel_bearing_1_to_2
                <= self.CROSSING_HIGH_RISK_MAX_1
                or self.CROSSING_HIGH_RISK_MIN_2
                <= rel_bearing_1_to_2
                <= self.CROSSING_HIGH_RISK_MAX_2
            ):
                return "high"
            else:
                return "medium"

        elif encounter_type in [
            EncounterType.OVERTAKING_GIVEN,
            EncounterType.OVERTAKING_TAKEN,
        ]:
            return "medium"

        elif encounter_type in [
            EncounterType.PARALLEL_SAME,
            EncounterType.PARALLEL_OPPOSITE,
        ]:
            return "low"

        else:
            return "none"


class COLREGSValidator:
    """
    Validates vessel behavior against COLREGS rules.

    Provides basic compliance checking for maritime encounters.
    """

    # Speed change tolerance in knots
    SPEED_CHANGE_TOLERANCE = 0.1

    def __init__(self, compliance_tolerance: float = 5.0):  # degrees
        """
        Initialize COLREGS validator.

        Args:
            compliance_tolerance: Tolerance for compliance checking (degrees)
        """
        self.compliance_tolerance = compliance_tolerance
        self.encounter_classifier = EncounterClassifier()

    def validate_encounter_compliance(
        self,
        encounter_result: EncounterResult,
        vessel1_actual_action: dict[str, Any],
        vessel2_actual_action: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate actual vessel actions against COLREGS requirements.

        Args:
            encounter_result: Result from encounter classification
            vessel1_actual_action: Actual action taken by vessel 1
                                 {'course_change': float, 'speed_change': float}
            vessel2_actual_action: Actual action taken by vessel 2

        Returns:
            Dictionary with compliance assessment
        """
        compliance_result = {
            "vessel1_compliant": self._check_action_compliance(
                encounter_result.vessel1_action, vessel1_actual_action
            ),
            "vessel2_compliant": self._check_action_compliance(
                encounter_result.vessel2_action, vessel2_actual_action
            ),
            "overall_compliant": False,
        }

        compliance_result["overall_compliant"] = (
            compliance_result["vessel1_compliant"]
            and compliance_result["vessel2_compliant"]
        )

        return compliance_result

    def _check_action_compliance(
        self, required_action: COLREGSAction, actual_action: dict[str, Any]
    ) -> bool:
        """Check if actual action complies with required COLREGS action."""

        course_change = actual_action.get("course_change", 0.0)
        speed_change = actual_action.get("speed_change", 0.0)

        if required_action == COLREGSAction.STAND_ON:
            # Should maintain course and speed (minimal changes)
            return (
                abs(course_change) <= self.compliance_tolerance
                and abs(speed_change) <= self.SPEED_CHANGE_TOLERANCE
            )  # Speed tolerance in knots

        elif required_action == COLREGSAction.GIVE_WAY:
            # Should make substantial course or speed change
            return (
                abs(course_change) > self.compliance_tolerance
                or abs(speed_change) > self.SPEED_CHANGE_TOLERANCE
            )

        elif required_action == COLREGSAction.BOTH_ALTER:
            # Should make some alteration
            return (
                abs(course_change) > self.compliance_tolerance
                or abs(speed_change) > self.SPEED_CHANGE_TOLERANCE
            )

        elif required_action == COLREGSAction.NO_ACTION:
            return True  # Any action is acceptable

        else:  # UNKNOWN
            return True  # Cannot assess unknown scenarios
