"""
CPA/TCPA (Closest Point of Approach / Time to Closest Point of Approach) calculations.

This module provides maritime-specific calculations for vessel encounter analysis,
including closest point of approach distance and time predictions.

Mathematical Foundation:
- CPA: Minimum distance between two moving vessels
- TCPA: Time until vessels reach closest approach
- Handles coordinate transformations from geographic to local coordinates
- Supports vectorized operations for multiple vessel pairs
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyproj

    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    warnings.warn(
        "pyproj not available. Geographic coordinate transformations will be limited.",
        stacklevel=2,
    )


@dataclass
class VesselState:
    """Represents the state of a vessel at a given time."""

    lat: float  # Latitude in decimal degrees
    lon: float  # Longitude in decimal degrees
    sog: float  # Speed over ground in knots
    cog: float  # Course over ground in degrees
    timestamp: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "lat": self.lat,
            "lon": self.lon,
            "sog": self.sog,
            "cog": self.cog,
            "timestamp": self.timestamp,
        }


@dataclass
class CPAResult:
    """Result of CPA/TCPA calculation."""

    cpa_distance: float  # CPA distance in meters
    tcpa_time: float  # TCPA time in seconds
    cpa_lat1: float  # Latitude of vessel 1 at CPA
    cpa_lon1: float  # Longitude of vessel 1 at CPA
    cpa_lat2: float  # Latitude of vessel 2 at CPA
    cpa_lon2: float  # Longitude of vessel 2 at CPA
    encounter_type: str  # Type of encounter (e.g., 'approaching', 'receding')
    warning_level: str  # Warning level based on thresholds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cpa_distance_m": self.cpa_distance,
            "tcpa_time_s": self.tcpa_time,
            "tcpa_time_min": self.tcpa_time / 60.0,
            "cpa_lat1": self.cpa_lat1,
            "cpa_lon1": self.cpa_lon1,
            "cpa_lat2": self.cpa_lat2,
            "cpa_lon2": self.cpa_lon2,
            "encounter_type": self.encounter_type,
            "warning_level": self.warning_level,
        }


class CPACalculator:
    """
    Calculates CPA/TCPA for maritime vessel encounters.

    Supports both single vessel pair calculations and vectorized operations
    for multiple vessel pairs simultaneously.
    """

    # Minimum relative speed squared to consider vessels as moving (m^2/s^2)
    MIN_RELATIVE_SPEED_SQ = 1e-10

    def __init__(
        self,
        cpa_warning_threshold: float = 500.0,  # meters
        tcpa_warning_threshold: float = 600.0,  # seconds (10 minutes)
        coordinate_system: str = "wgs84",
        local_projection: str = "utm",
    ):
        """
        Initialize CPA calculator.

        Args:
            cpa_warning_threshold: Distance threshold for CPA warnings (meters)
            tcpa_warning_threshold: Time threshold for TCPA warnings (seconds)
            coordinate_system: Input coordinate system ('wgs84')
            local_projection: Local projection for calculations ('utm', 'mercator')
        """
        self.cpa_warning_threshold = cpa_warning_threshold
        self.tcpa_warning_threshold = tcpa_warning_threshold
        self.coordinate_system = coordinate_system
        self.local_projection = local_projection

        # Initialize coordinate transformer
        self._setup_coordinate_transformer()

    def _setup_coordinate_transformer(self):
        """Setup coordinate transformation for accurate distance calculations."""
        if not PYPROJ_AVAILABLE:
            warnings.warn(
                "pyproj not available. Using approximate calculations.", stacklevel=2
            )
            self.transformer = None
            self.inverse_transformer = None
            return

        # For now, use a simple approach - will be enhanced based on data region
        # In practice, should determine UTM zone based on data location
        self.wgs84 = pyproj.CRS("EPSG:4326")  # WGS84
        self.utm = pyproj.CRS(
            "EPSG:32633"
        )  # UTM Zone 33N (example - should be dynamic)

        self.transformer = pyproj.Transformer.from_crs(
            self.wgs84, self.utm, always_xy=True
        )
        self.inverse_transformer = pyproj.Transformer.from_crs(
            self.utm, self.wgs84, always_xy=True
        )

    def _geographic_to_local(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert geographic coordinates to local projection."""
        if self.transformer is None:
            # Approximate conversion using simple scaling
            # This is a fallback - not accurate for precise maritime calculations
            x = lon * 111320 * np.cos(np.radians(lat))  # Approximate meters
            y = lat * 110540  # Approximate meters
            return x, y

        return self.transformer.transform(lon, lat)

    def _local_to_geographic(self, x: float, y: float) -> tuple[float, float]:
        """Convert local coordinates back to geographic."""
        if self.inverse_transformer is None:
            # Approximate inverse conversion
            lat = y / 110540
            lon = x / (111320 * np.cos(np.radians(lat)))
            return lon, lat

        return self.inverse_transformer.transform(x, y)

    def _knots_to_mps(self, speed_knots: float) -> float:
        """Convert speed from knots to meters per second."""
        return speed_knots * 0.514444

    def _degrees_to_velocity_components(
        self, speed_mps: float, course_degrees: float
    ) -> tuple[float, float]:
        """Convert speed and course to velocity components."""
        # Convert course to standard mathematical angle (counter-clockwise from east)
        # Maritime course: 0° = North, 90° = East
        # Mathematical angle: 0° = East, 90° = North
        angle_rad = np.radians(90 - course_degrees)

        vx = speed_mps * np.cos(angle_rad)  # East component
        vy = speed_mps * np.sin(angle_rad)  # North component

        return vx, vy

    def calculate_cpa_tcpa_basic(
        self, vessel1: VesselState, vessel2: VesselState
    ) -> CPAResult:
        """
        Calculate CPA/TCPA for two vessels using basic geometric approach.

        Args:
            vessel1: State of first vessel
            vessel2: State of second vessel

        Returns:
            CPAResult with calculated CPA distance, TCPA time, and encounter details
        """
        # Convert coordinates to local projection
        x1, y1 = self._geographic_to_local(vessel1.lon, vessel1.lat)
        x2, y2 = self._geographic_to_local(vessel2.lon, vessel2.lat)

        # Convert speeds and courses to velocity components
        speed1_mps = self._knots_to_mps(vessel1.sog)
        speed2_mps = self._knots_to_mps(vessel2.sog)

        vx1, vy1 = self._degrees_to_velocity_components(speed1_mps, vessel1.cog)
        vx2, vy2 = self._degrees_to_velocity_components(speed2_mps, vessel2.cog)

        # Calculate relative position and velocity
        dx = x2 - x1  # Relative position
        dy = y2 - y1
        dvx = vx2 - vx1  # Relative velocity
        dvy = vy2 - vy1

        # Calculate TCPA
        relative_speed_squared = dvx * dvx + dvy * dvy

        if (
            relative_speed_squared < self.MIN_RELATIVE_SPEED_SQ
        ):  # Vessels have same velocity (parallel, same speed)
            # Vessels are moving in parallel with same speed
            tcpa = 0.0  # They maintain constant distance
            cpa_distance = np.sqrt(dx * dx + dy * dy)
            encounter_type = "parallel"
        else:
            # Time to closest approach
            tcpa = -(dx * dvx + dy * dvy) / relative_speed_squared

            # Position at closest approach
            cpa_dx = dx + dvx * tcpa
            cpa_dy = dy + dvy * tcpa
            cpa_distance = np.sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy)

            encounter_type = "approaching" if tcpa > 0 else "receding"

        # Calculate positions at CPA
        cpa_x1 = x1 + vx1 * tcpa
        cpa_y1 = y1 + vy1 * tcpa
        cpa_x2 = x2 + vx2 * tcpa
        cpa_y2 = y2 + vy2 * tcpa

        # Convert back to geographic coordinates
        cpa_lon1, cpa_lat1 = self._local_to_geographic(cpa_x1, cpa_y1)
        cpa_lon2, cpa_lat2 = self._local_to_geographic(cpa_x2, cpa_y2)

        # Determine warning level
        warning_level = self._determine_warning_level(cpa_distance, tcpa)

        return CPAResult(
            cpa_distance=cpa_distance,
            tcpa_time=tcpa,
            cpa_lat1=cpa_lat1,
            cpa_lon1=cpa_lon1,
            cpa_lat2=cpa_lat2,
            cpa_lon2=cpa_lon2,
            encounter_type=encounter_type,
            warning_level=warning_level,
        )

    def _determine_warning_level(self, cpa_distance: float, tcpa_time: float) -> str:
        """Determine warning level based on CPA distance and TCPA time."""
        if tcpa_time <= 0:
            return "none"  # Vessels are receding

        if (
            cpa_distance <= self.cpa_warning_threshold
            and tcpa_time <= self.tcpa_warning_threshold
        ):
            if cpa_distance <= self.cpa_warning_threshold * 0.5:
                return "critical"
            elif cpa_distance <= self.cpa_warning_threshold * 0.75:
                return "high"
            else:
                return "medium"
        elif (
            cpa_distance <= self.cpa_warning_threshold * 1.5
            and tcpa_time <= self.tcpa_warning_threshold * 1.5
        ):
            return "low"
        else:
            return "none"

    def calculate_cpa_tcpa_vectorized(
        self, vessels1: list | np.ndarray, vessels2: list | np.ndarray
    ) -> list:
        """
        Calculate CPA/TCPA for multiple vessel pairs using vectorized operations.

        Args:
            vessels1: Array/list of VesselState objects for first vessels
            vessels2: Array/list of VesselState objects for second vessels

        Returns:
            List of CPAResult objects for each vessel pair

        Note:
            This method provides significant performance improvements for large
            numbers of vessel encounters by using vectorized NumPy operations.
        """
        if len(vessels1) != len(vessels2):
            raise ValueError("vessels1 and vessels2 must have the same length")

        if len(vessels1) == 0:
            return []

        # Convert to arrays for vectorized operations
        n_pairs = len(vessels1)

        # Extract coordinates and convert to local projection
        lats1 = np.array([v.lat for v in vessels1])
        lons1 = np.array([v.lon for v in vessels1])
        lats2 = np.array([v.lat for v in vessels2])
        lons2 = np.array([v.lon for v in vessels2])

        # Vectorized coordinate conversion
        if self.transformer is not None:
            # Use pyproj for accurate conversion (batch processing)
            x1_array, y1_array = self.transformer.transform(lons1, lats1)
            x2_array, y2_array = self.transformer.transform(lons2, lats2)
        else:
            # Fallback to approximate conversion
            x1_array = lons1 * 111320 * np.cos(np.radians(lats1))
            y1_array = lats1 * 110540
            x2_array = lons2 * 111320 * np.cos(np.radians(lats2))
            y2_array = lats2 * 110540

        # Extract speeds and courses
        sog1 = np.array([v.sog for v in vessels1])
        cog1 = np.array([v.cog for v in vessels1])
        sog2 = np.array([v.sog for v in vessels2])
        cog2 = np.array([v.cog for v in vessels2])

        # Convert to velocity components (vectorized)
        speed1_mps = sog1 * 0.514444
        speed2_mps = sog2 * 0.514444

        # Vectorized course to velocity conversion
        angle1_rad = np.radians(90 - cog1)
        angle2_rad = np.radians(90 - cog2)

        vx1 = speed1_mps * np.cos(angle1_rad)
        vy1 = speed1_mps * np.sin(angle1_rad)
        vx2 = speed2_mps * np.cos(angle2_rad)
        vy2 = speed2_mps * np.sin(angle2_rad)

        # Relative position and velocity (vectorized)
        dx = x2_array - x1_array
        dy = y2_array - y1_array
        dvx = vx2 - vx1
        dvy = vy2 - vy1

        # Calculate TCPA (vectorized)
        relative_speed_squared = dvx * dvx + dvy * dvy

        # Handle parallel vessels (same velocity)
        parallel_mask = relative_speed_squared < self.MIN_RELATIVE_SPEED_SQ
        tcpa_array = np.zeros(n_pairs)

        # For non-parallel vessels
        non_parallel_mask = ~parallel_mask
        if np.any(non_parallel_mask):
            tcpa_array[non_parallel_mask] = (
                -(
                    dx[non_parallel_mask] * dvx[non_parallel_mask]
                    + dy[non_parallel_mask] * dvy[non_parallel_mask]
                )
                / relative_speed_squared[non_parallel_mask]
            )

        # Calculate CPA distance (vectorized)
        cpa_dx = dx + dvx * tcpa_array
        cpa_dy = dy + dvy * tcpa_array
        cpa_distances = np.sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy)

        # For parallel vessels, use current distance
        cpa_distances[parallel_mask] = np.sqrt(
            dx[parallel_mask] ** 2 + dy[parallel_mask] ** 2
        )

        # Calculate positions at CPA (vectorized)
        cpa_x1 = x1_array + vx1 * tcpa_array
        cpa_y1 = y1_array + vy1 * tcpa_array
        cpa_x2 = x2_array + vx2 * tcpa_array
        cpa_y2 = y2_array + vy2 * tcpa_array

        # Convert back to geographic coordinates (vectorized)
        if self.inverse_transformer is not None:
            cpa_lons1, cpa_lats1 = self.inverse_transformer.transform(cpa_x1, cpa_y1)
            cpa_lons2, cpa_lats2 = self.inverse_transformer.transform(cpa_x2, cpa_y2)
        else:
            # Approximate inverse conversion
            cpa_lats1 = cpa_y1 / 110540
            cpa_lons1 = cpa_x1 / (111320 * np.cos(np.radians(cpa_lats1)))
            cpa_lats2 = cpa_y2 / 110540
            cpa_lons2 = cpa_x2 / (111320 * np.cos(np.radians(cpa_lats2)))

        # Determine encounter types and warning levels (vectorized)
        encounter_types = np.where(
            parallel_mask,
            "parallel",
            np.where(tcpa_array > 0, "approaching", "receding"),
        )

        warning_levels = self._determine_warning_levels_vectorized(
            cpa_distances, tcpa_array
        )

        # Create result objects
        results = []
        for i in range(n_pairs):
            result = CPAResult(
                cpa_distance=float(cpa_distances[i]),
                tcpa_time=float(tcpa_array[i]),
                cpa_lat1=float(cpa_lats1[i]),
                cpa_lon1=float(cpa_lons1[i]),
                cpa_lat2=float(cpa_lats2[i]),
                cpa_lon2=float(cpa_lons2[i]),
                encounter_type=encounter_types[i],
                warning_level=warning_levels[i],
            )
            results.append(result)

        return results

    def _determine_warning_levels_vectorized(
        self, cpa_distances: np.ndarray, tcpa_times: np.ndarray
    ) -> np.ndarray:
        """Vectorized warning level determination."""
        # Initialize all as 'none'
        warning_levels = np.full(len(cpa_distances), "none", dtype=object)

        # Receding vessels (negative TCPA) already have 'none' as default

        # Approaching vessels within thresholds
        approaching_mask = (
            (tcpa_times > 0)
            & (cpa_distances <= self.cpa_warning_threshold)
            & (tcpa_times <= self.tcpa_warning_threshold)
        )

        # Critical warning
        critical_mask = approaching_mask & (
            cpa_distances <= self.cpa_warning_threshold * 0.5
        )
        warning_levels[critical_mask] = "critical"

        # High warning
        high_mask = (
            approaching_mask
            & ~critical_mask
            & (cpa_distances <= self.cpa_warning_threshold * 0.75)
        )
        warning_levels[high_mask] = "high"

        # Medium warning
        medium_mask = approaching_mask & ~critical_mask & ~high_mask
        warning_levels[medium_mask] = "medium"

        # Low warning (extended thresholds)
        low_mask = (
            (tcpa_times > 0)
            & ~approaching_mask
            & (cpa_distances <= self.cpa_warning_threshold * 1.5)
            & (tcpa_times <= self.tcpa_warning_threshold * 1.5)
        )
        warning_levels[low_mask] = "low"

        return warning_levels


class CPAValidator:
    """
    Validates CPA/TCPA predictions against ground truth encounters.

    Used for evaluating the accuracy of trajectory prediction models
    in maritime encounter scenarios.
    """

    def __init__(
        self,
        tolerance_distance: float = 50.0,  # meters
        tolerance_time: float = 30.0,
    ):  # seconds
        """
        Initialize CPA validator.

        Args:
            tolerance_distance: Acceptable error in CPA distance prediction (meters)
            tolerance_time: Acceptable error in TCPA time prediction (seconds)
        """
        self.tolerance_distance = tolerance_distance
        self.tolerance_time = tolerance_time

    def validate_prediction(
        self, predicted_cpa: CPAResult, actual_cpa: CPAResult
    ) -> dict[str, Any]:
        """
        Validate CPA/TCPA prediction against actual encounter.

        Args:
            predicted_cpa: Predicted CPA result
            actual_cpa: Actual CPA result

        Returns:
            Dictionary with validation metrics
        """
        distance_error = abs(predicted_cpa.cpa_distance - actual_cpa.cpa_distance)
        time_error = abs(predicted_cpa.tcpa_time - actual_cpa.tcpa_time)

        distance_accurate = distance_error <= self.tolerance_distance
        time_accurate = time_error <= self.tolerance_time

        overall_accurate = distance_accurate and time_accurate

        return {
            "distance_error_m": distance_error,
            "time_error_s": time_error,
            "distance_accurate": distance_accurate,
            "time_accurate": time_accurate,
            "overall_accurate": overall_accurate,
            "distance_error_percentage": (distance_error / actual_cpa.cpa_distance)
            * 100,
            "time_error_percentage": (time_error / abs(actual_cpa.tcpa_time)) * 100
            if actual_cpa.tcpa_time != 0
            else 0,
        }

    def compute_validation_statistics(self, validation_results: list) -> dict[str, Any]:
        """
        Compute aggregate validation statistics from multiple validations.

        Args:
            validation_results: List of validation result dictionaries

        Returns:
            Dictionary with aggregate statistics
        """
        if not validation_results:
            return {}

        distance_errors = [r["distance_error_m"] for r in validation_results]
        time_errors = [r["time_error_s"] for r in validation_results]

        accuracy_rates = {
            "distance_accuracy_rate": np.mean(
                [r["distance_accurate"] for r in validation_results]
            ),
            "time_accuracy_rate": np.mean(
                [r["time_accurate"] for r in validation_results]
            ),
            "overall_accuracy_rate": np.mean(
                [r["overall_accurate"] for r in validation_results]
            ),
        }

        error_statistics = {
            "mean_distance_error_m": np.mean(distance_errors),
            "std_distance_error_m": np.std(distance_errors),
            "median_distance_error_m": np.median(distance_errors),
            "mean_time_error_s": np.mean(time_errors),
            "std_time_error_s": np.std(time_errors),
            "median_time_error_s": np.median(time_errors),
        }

        return {**accuracy_rates, **error_statistics}
