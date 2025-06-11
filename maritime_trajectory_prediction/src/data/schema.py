"""
Centralized data schema and column definitions for AIS data.

This module provides a single source of truth for all column names,
data types, and field mappings used throughout the system.
"""


class ColumnNames:
    """Centralized column name definitions."""

    # Core identification
    MMSI = "mmsi"
    TIME = "time"

    # Position coordinates (use short names to match actual data)
    LAT = "lat"
    LON = "lon"

    # Movement
    SOG = "sog"  # Speed over ground
    COG = "cog"  # Course over ground
    HEADING = "heading"
    TURN = "turn"

    # Status and metadata
    NAV_STATUS = "nav_status"
    MSG_TYPE = "msg_type"
    ACCURACY = "accuracy"

    # Vessel characteristics
    SHIP_TYPE = "ship_type"
    SHIP_TYPE_TEXT = "ship_type_text"
    LENGTH = "length"
    WIDTH = "width"
    DRAFT = "draft"
    NAME = "name"
    CALL_SIGN = "call_sign"
    IMO = "imo"
    DESTINATION = "destination"

    # Derived features
    DISTANCE_KM = "distance_km"
    SPEED_DELTA = "speed_delta"
    COURSE_DELTA = "course_delta"
    TIME_DIFF = "time_diff"
    BEARING = "bearing"


class FeatureGroups:
    """Predefined feature groups for different tasks."""

    # Core position and movement features
    POSITION = [ColumnNames.LAT, ColumnNames.LON]
    MOVEMENT = [ColumnNames.SOG, ColumnNames.COG, ColumnNames.HEADING]
    # Basic trajectory for prediction targets (position + basic movement)
    BASIC_TRAJECTORY = [
        ColumnNames.LAT,
        ColumnNames.LON,
        ColumnNames.SOG,
        ColumnNames.COG,
    ]
    # Full trajectory including heading for input features
    FULL_TRAJECTORY = POSITION + MOVEMENT

    # Extended trajectory features
    EXTENDED_TRAJECTORY = BASIC_TRAJECTORY + [
        ColumnNames.TURN,
        ColumnNames.DISTANCE_KM,
        ColumnNames.SPEED_DELTA,
        ColumnNames.COURSE_DELTA,
    ]

    # Required columns for data validation
    REQUIRED_CORE = [ColumnNames.MMSI, ColumnNames.TIME] + POSITION

    # All available features
    ALL_FEATURES = [
        ColumnNames.LAT,
        ColumnNames.LON,
        ColumnNames.SOG,
        ColumnNames.COG,
        ColumnNames.HEADING,
        ColumnNames.TURN,
        ColumnNames.NAV_STATUS,
        ColumnNames.MSG_TYPE,
        ColumnNames.ACCURACY,
        ColumnNames.DISTANCE_KM,
        ColumnNames.SPEED_DELTA,
        ColumnNames.COURSE_DELTA,
        ColumnNames.TIME_DIFF,
        ColumnNames.BEARING,
    ]


class DataTypes:
    """Standard data types for columns."""

    COLUMN_DTYPES = {
        ColumnNames.MMSI: "int64",
        ColumnNames.TIME: "datetime64[ns]",
        ColumnNames.LAT: "float64",
        ColumnNames.LON: "float64",
        ColumnNames.SOG: "float64",
        ColumnNames.COG: "float64",
        ColumnNames.HEADING: "float64",
        ColumnNames.TURN: "float64",
        ColumnNames.NAV_STATUS: "int8",
        ColumnNames.MSG_TYPE: "int8",
        ColumnNames.ACCURACY: "int8",
        ColumnNames.SHIP_TYPE: "int16",
        ColumnNames.LENGTH: "float64",
        ColumnNames.WIDTH: "float64",
        ColumnNames.DRAFT: "float64",
        ColumnNames.DISTANCE_KM: "float64",
        ColumnNames.SPEED_DELTA: "float64",
        ColumnNames.COURSE_DELTA: "float64",
        ColumnNames.TIME_DIFF: "float64",
        ColumnNames.BEARING: "float64",
    }


class ValidationRanges:
    """Valid ranges for different columns."""

    RANGES = {
        ColumnNames.LAT: (-90, 90),
        ColumnNames.LON: (-180, 180),
        ColumnNames.SOG: (0, 102.2),  # Max AIS speed
        ColumnNames.COG: (0, 360),
        ColumnNames.HEADING: (0, 360),
        ColumnNames.TURN: (-127, 127),
        ColumnNames.NAV_STATUS: (0, 15),
        ColumnNames.MSG_TYPE: (1, 27),
        ColumnNames.ACCURACY: (0, 1),
    }


# Legacy field mappings for backward compatibility
LEGACY_MAPPINGS = {
    "latitude": ColumnNames.LAT,
    "longitude": ColumnNames.LON,
    "speed": ColumnNames.SOG,
    "course": ColumnNames.COG,
}


def get_feature_list(task: str = "trajectory_prediction") -> list[str]:
    """Get appropriate feature list for a given task."""
    if task == "trajectory_prediction":
        return FeatureGroups.EXTENDED_TRAJECTORY
    elif task == "anomaly_detection":
        return FeatureGroups.ALL_FEATURES
    elif task == "collision_avoidance":
        return FeatureGroups.BASIC_TRAJECTORY + [ColumnNames.TURN, ColumnNames.ACCURACY]
    else:
        return FeatureGroups.BASIC_TRAJECTORY


def get_required_columns() -> list[str]:
    """Get list of required columns for basic functionality."""
    return FeatureGroups.REQUIRED_CORE


def validate_column_names(df_columns: list[str]) -> dict[str, list[str]]:
    """Validate dataframe columns against schema."""
    required = set(get_required_columns())
    available = set(df_columns)

    missing = list(required - available)
    extra = list(
        available - set(FeatureGroups.ALL_FEATURES + FeatureGroups.REQUIRED_CORE)
    )

    return {"missing": missing, "extra": extra, "valid": len(missing) == 0}
