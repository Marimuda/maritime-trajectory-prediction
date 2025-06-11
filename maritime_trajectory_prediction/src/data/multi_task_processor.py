"""
Enhanced AIS Multi-Task Processor with message type preservation.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from .schema import ColumnNames

logger = logging.getLogger(__name__)


class MLTask(Enum):
    """Supported ML tasks for AIS data processing."""

    TRAJECTORY_PREDICTION = "trajectory_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    GRAPH_NEURAL_NETWORKS = "graph_neural_networks"
    COLLISION_AVOIDANCE = "collision_avoidance"
    PORT_OPERATIONS = "port_operations"
    ENVIRONMENTAL_MONITORING = "environmental_monitoring"
    SEARCH_AND_RESCUE = "search_and_rescue"


@dataclass
class MessageTypeConfig:
    """Configuration for AIS message type processing."""

    # Core position reports - always preserve
    CORE_POSITION_TYPES: set[int] = frozenset({1, 2, 3, 18, 19, 27})

    # Infrastructure and context
    CONTEXT_TYPES: set[int] = frozenset({4, 21})

    # Static and voyage data
    METADATA_TYPES: set[int] = frozenset({5, 24})

    # Safety and emergency
    SAFETY_TYPES: set[int] = frozenset({6, 7, 8, 9, 12, 13, 14, 15, 16})

    # All supported types
    ALL_TYPES: set[int] = (
        CORE_POSITION_TYPES | CONTEXT_TYPES | METADATA_TYPES | SAFETY_TYPES
    )


@dataclass
class FieldConfig:
    """Field extraction configuration for different tasks."""

    # Universal fields for all tasks
    UNIVERSAL_FIELDS: set[str] = frozenset(
        {
            ColumnNames.MMSI,
            ColumnNames.TIME,
            ColumnNames.LAT,
            ColumnNames.LON,
            "message_type",
            "rxtime",
            "channel",
        }
    )

    # Task-specific field groups
    TRAJECTORY_FIELDS: set[str] = frozenset(
        {
            "sog",
            "cog",
            "heading",
            "turn",
            "status",
            "accuracy",
            "second",
            "maneuver",
            "raim",
        }
    )

    ANOMALY_FIELDS: set[str] = frozenset(
        {
            "shipname",
            "shiptype",
            "to_bow",
            "to_stern",
            "to_port",
            "to_starboard",
            "destination",
            "eta",
            "draught",
        }
    )

    GRAPH_FIELDS: set[str] = frozenset(
        {"callsign", "imo", "aid_type", "name", "off_position", "virtual_aid"}
    )

    SAFETY_FIELDS: set[str] = frozenset({"safety_text", "urgency", "response_required"})


class AISMultiTaskProcessor:
    """Enhanced AIS processor with multi-task support and message type preservation."""

    def __init__(self, target_tasks: list[MLTask | str] = None):
        """
        Initialize the multi-task processor.

        Args:
            target_tasks: List of ML tasks to optimize for
        """
        self.target_tasks = self._parse_tasks(
            target_tasks or [MLTask.TRAJECTORY_PREDICTION]
        )
        self.msg_config = MessageTypeConfig()
        self.field_config = FieldConfig()

        # Compute required message types and fields
        self.required_message_types = self._compute_required_message_types()
        self.required_fields = self._compute_required_fields()

        # Statistics tracking
        self.stats = {
            "lines_processed": 0,
            "valid_records": 0,
            "filtered_records": 0,
            "error_records": 0,
            "message_type_counts": {},
            "task_specific_records": {task.value: 0 for task in self.target_tasks},
        }

        logger.info(
            f"Initialized AIS Multi-Task Processor for tasks: {[t.value for t in self.target_tasks]}"
        )
        logger.info(f"Required message types: {sorted(self.required_message_types)}")

    def _parse_tasks(self, tasks: list[MLTask | str]) -> list[MLTask]:
        """Parse task list into MLTask enums."""
        parsed_tasks = []
        for task in tasks:
            if isinstance(task, str):
                try:
                    parsed_tasks.append(MLTask(task))
                except ValueError:
                    logger.warning(f"Unknown task: {task}")
            elif isinstance(task, MLTask):
                parsed_tasks.append(task)
        return parsed_tasks

    def _compute_required_message_types(self) -> set[int]:
        """Compute required message types based on target tasks."""
        required_types = set()

        for task in self.target_tasks:
            if task in [
                MLTask.TRAJECTORY_PREDICTION,
                MLTask.COLLISION_AVOIDANCE,
                MLTask.ENVIRONMENTAL_MONITORING,
            ]:
                # Core position data + context
                required_types.update(self.msg_config.CORE_POSITION_TYPES)
                required_types.update(self.msg_config.CONTEXT_TYPES)

            elif task in [MLTask.ANOMALY_DETECTION, MLTask.PORT_OPERATIONS]:
                # Position + metadata for behavior analysis
                required_types.update(self.msg_config.CORE_POSITION_TYPES)
                required_types.update(self.msg_config.CONTEXT_TYPES)
                required_types.update(self.msg_config.METADATA_TYPES)

            elif task == MLTask.GRAPH_NEURAL_NETWORKS:
                # All types for comprehensive network modeling
                required_types.update(self.msg_config.ALL_TYPES)

            elif task == MLTask.SEARCH_AND_RESCUE:
                # Position + safety messages
                required_types.update(self.msg_config.CORE_POSITION_TYPES)
                required_types.update(self.msg_config.CONTEXT_TYPES)
                required_types.update(self.msg_config.SAFETY_TYPES)

        return required_types

    def _compute_required_fields(self) -> set[str]:
        """Compute required fields based on target tasks."""
        required_fields = set(self.field_config.UNIVERSAL_FIELDS)

        for task in self.target_tasks:
            if task in [MLTask.TRAJECTORY_PREDICTION, MLTask.COLLISION_AVOIDANCE]:
                required_fields.update(self.field_config.TRAJECTORY_FIELDS)

            elif task in [MLTask.ANOMALY_DETECTION, MLTask.PORT_OPERATIONS]:
                required_fields.update(self.field_config.TRAJECTORY_FIELDS)
                required_fields.update(self.field_config.ANOMALY_FIELDS)

            elif task == MLTask.GRAPH_NEURAL_NETWORKS:
                required_fields.update(self.field_config.TRAJECTORY_FIELDS)
                required_fields.update(self.field_config.GRAPH_FIELDS)

            elif task == MLTask.SEARCH_AND_RESCUE:
                required_fields.update(self.field_config.TRAJECTORY_FIELDS)
                required_fields.update(self.field_config.SAFETY_FIELDS)

        return required_fields

    def parse_line(self, line: str) -> dict | None:
        """Parse a single AIS log line with task-aware filtering."""
        self.stats["lines_processed"] += 1

        # Handle None or non-string input
        if not isinstance(line, str):
            self.stats["error_records"] += 1
            return None

        # Skip engine status lines
        if "AIS engine" in line:
            self.stats["filtered_records"] += 1
            return None

        try:
            # Split timestamp and payload
            _, payload = line.split(" - ", 1)
            data = json.loads(payload)

            # Check if message type is required
            msg_type = data.get("type")
            if msg_type not in self.required_message_types:
                self.stats["filtered_records"] += 1
                return None

            # Extract and validate core fields
            parsed_record = self._extract_fields(data)
            if parsed_record is None:
                self.stats["error_records"] += 1
                return None

            # Update statistics
            self.stats["valid_records"] += 1
            self.stats["message_type_counts"][msg_type] = (
                self.stats["message_type_counts"].get(msg_type, 0) + 1
            )

            # Update task-specific counters
            for task in self.target_tasks:
                if self._is_relevant_for_task(parsed_record, task):
                    self.stats["task_specific_records"][task.value] += 1

            return parsed_record

        except (ValueError, json.JSONDecodeError, IndexError, KeyError):
            self.stats["error_records"] += 1
            return None

    def _extract_fields(self, data: dict) -> dict | None:
        """Extract relevant fields based on task requirements."""
        try:
            # Start with universal fields
            record = {}

            # Core identification
            record["mmsi"] = data.get("mmsi")
            record["message_type"] = data.get("type")
            record["rxtime"] = data.get("rxtime")
            record["channel"] = data.get("channel")

            # Position data (with field mapping)
            if "lat" in data:
                record[ColumnNames.LAT] = data["lat"]
            if "lon" in data:
                record[ColumnNames.LON] = data["lon"]

            # Movement data
            if "speed" in data:
                record["sog"] = data["speed"]
            if "course" in data:
                record["cog"] = data["course"]
            if "heading" in data:
                record["heading"] = data["heading"]
            if "turn" in data:
                record["turn"] = data["turn"]

            # Navigation status
            if "status" in data:
                record["status"] = data["status"]
            if "accuracy" in data:
                record["accuracy"] = data["accuracy"]
            if "second" in data:
                record["second"] = data["second"]
            if "maneuver" in data:
                record["maneuver"] = data["maneuver"]
            if "raim" in data:
                record["raim"] = data["raim"]

            # Static and voyage data (for anomaly detection, port operations)
            if any(
                task in [MLTask.ANOMALY_DETECTION, MLTask.PORT_OPERATIONS]
                for task in self.target_tasks
            ):
                if "shipname" in data:
                    record["shipname"] = data["shipname"]
                if "shiptype" in data:
                    record["shiptype"] = data["shiptype"]
                if "to_bow" in data:
                    record["to_bow"] = data["to_bow"]
                if "to_stern" in data:
                    record["to_stern"] = data["to_stern"]
                if "to_port" in data:
                    record["to_port"] = data["to_port"]
                if "to_starboard" in data:
                    record["to_starboard"] = data["to_starboard"]
                if "destination" in data:
                    record["destination"] = data["destination"]
                if "eta" in data:
                    record["eta"] = data["eta"]
                if "draught" in data:
                    record["draught"] = data["draught"]

            # Graph network fields
            if MLTask.GRAPH_NEURAL_NETWORKS in self.target_tasks:
                if "callsign" in data:
                    record["callsign"] = data["callsign"]
                if "imo" in data:
                    record["imo"] = data["imo"]
                if "aid_type" in data:
                    record["aid_type"] = data["aid_type"]
                if "name" in data:
                    record["name"] = data["name"]
                if "off_position" in data:
                    record["off_position"] = data["off_position"]
                if "virtual_aid" in data:
                    record["virtual_aid"] = data["virtual_aid"]

            # Safety fields (for SAR)
            if MLTask.SEARCH_AND_RESCUE in self.target_tasks:
                if "safety_text" in data:
                    record["safety_text"] = data["safety_text"]

            # Parse timestamp
            if "rxtime" in data:
                try:
                    timestamp_str = data["rxtime"]
                    if len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                        year = int(timestamp_str[:4])
                        month = int(timestamp_str[4:6])
                        day = int(timestamp_str[6:8])
                        hour = int(timestamp_str[8:10])
                        minute = int(timestamp_str[10:12])
                        second = int(timestamp_str[12:14])

                        # Basic validation
                        if (
                            1900 <= year <= 2100
                            and 1 <= month <= 12
                            and 1 <= day <= 31
                            and 0 <= hour <= 23
                            and 0 <= minute <= 59
                            and 0 <= second <= 59
                        ):
                            record["time"] = pd.Timestamp(
                                year, month, day, hour, minute, second, tz="UTC"
                            )
                        else:
                            record["time"] = pd.NaT
                    else:
                        record["time"] = pd.NaT
                except (ValueError, TypeError):
                    record["time"] = pd.NaT
            else:
                record["time"] = pd.NaT

            # Validate essential fields
            if record.get("mmsi") is None:
                return None

            # Classify message for task relevance
            record["msg_class"] = self._classify_message(record)

            return record

        except Exception as e:
            logger.debug(f"Field extraction error: {e}")
            return None

    def _classify_message(self, record: dict) -> str:
        """Classify message type for ML tasks."""
        msg_type = record.get("message_type")

        if msg_type in {1, 2, 3}:
            return "A_pos"  # Class A position report
        elif msg_type in {18, 19}:
            return "B_pos"  # Class B position report
        elif msg_type == 4:
            return "base_station"
        elif msg_type == 5:
            return "static_voyage"
        elif msg_type == 21:
            return "aton"  # Aid to navigation
        elif msg_type == 24:
            return "static_data"
        elif msg_type in {6, 7, 8, 12, 13, 14}:
            return "safety_comm"
        elif msg_type == 9:
            return "sar_aircraft"
        elif msg_type == 27:
            return "long_range"
        else:
            return "other"

    def _is_relevant_for_task(self, record: dict, task: MLTask) -> bool:
        """Check if record is relevant for specific task."""
        msg_class = record.get("msg_class", "")

        if task == MLTask.TRAJECTORY_PREDICTION:
            return msg_class in {"A_pos", "B_pos", "long_range"}
        elif task == MLTask.ANOMALY_DETECTION:
            return msg_class in {"A_pos", "B_pos", "static_voyage", "long_range"}
        elif task == MLTask.GRAPH_NEURAL_NETWORKS:
            return True  # All message types relevant
        elif task == MLTask.COLLISION_AVOIDANCE:
            return msg_class in {"A_pos", "B_pos"}
        elif task == MLTask.PORT_OPERATIONS:
            return msg_class in {"A_pos", "B_pos", "static_voyage"}
        elif task == MLTask.ENVIRONMENTAL_MONITORING:
            return msg_class in {"A_pos", "B_pos", "long_range"}
        elif task == MLTask.SEARCH_AND_RESCUE:
            return msg_class in {"A_pos", "B_pos", "sar_aircraft", "safety_comm"}

        return False

    def process_file(
        self, file_path: str | Path, chunk_size: int = 10000
    ) -> pd.DataFrame:
        """Process AIS log file with task-aware filtering."""
        file_path = Path(file_path)
        logger.info(f"Processing file: {file_path}")

        records = []

        with open(file_path) as f:
            for i, line in enumerate(f):
                record = self.parse_line(line.strip())
                if record:
                    records.append(record)

                # Progress logging
                if (i + 1) % chunk_size == 0:
                    logger.info(
                        f"Processed {i + 1:,} lines, {len(records):,} valid records"
                    )

        logger.info(
            f"Completed processing: {len(records):,} valid records from {file_path}"
        )
        logger.info(f"Stats: {self.stats}")

        if records:
            df = pd.DataFrame(records)
            return self._post_process_dataframe(df)
        else:
            return pd.DataFrame()

    def _post_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the DataFrame for task optimization."""
        if df.empty:
            return df

        # Sort by time and MMSI for trajectory tasks
        if MLTask.TRAJECTORY_PREDICTION in self.target_tasks:
            df = df.sort_values(["mmsi", "time"])

        # Add derived features for anomaly detection
        if MLTask.ANOMALY_DETECTION in self.target_tasks:
            df = self._add_anomaly_features(df)

        # Add graph features
        if MLTask.GRAPH_NEURAL_NETWORKS in self.target_tasks:
            df = self._add_graph_features(df)

        return df

    def _add_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for anomaly detection."""
        # Speed anomaly indicators
        if "sog" in df.columns:
            df["speed_anomaly"] = df["sog"] > 30.0  # Unrealistic speed

        # Position jump detection
        if all(col in df.columns for col in [ColumnNames.LAT, ColumnNames.LON]):
            df["position_jump"] = False  # Placeholder for position jump detection

        return df

    def _add_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for graph neural networks."""
        # Node type classification
        df["node_type"] = (
            df["msg_class"]
            .map(
                {
                    "A_pos": "vessel_a",
                    "B_pos": "vessel_b",
                    "base_station": "infrastructure",
                    "aton": "infrastructure",
                    "static_voyage": "metadata",
                    "static_data": "metadata",
                }
            )
            .fillna("other")
        )

        return df

    def get_task_specific_dataset(self, df: pd.DataFrame, task: MLTask) -> pd.DataFrame:
        """Extract task-specific dataset from processed DataFrame."""
        if df.empty:
            return df

        # Filter records relevant to the task
        mask = df.apply(
            lambda row: self._is_relevant_for_task(row.to_dict(), task), axis=1
        )
        task_df = df[mask].copy()

        # Task-specific processing
        if task == MLTask.TRAJECTORY_PREDICTION:
            # Keep only position reports, sort by vessel and time
            task_df = task_df[
                task_df["msg_class"].isin(["A_pos", "B_pos", "long_range"])
            ]
            task_df = task_df.sort_values(["mmsi", "time"])

        elif task == MLTask.GRAPH_NEURAL_NETWORKS:
            # Include all message types, add graph-specific features
            task_df = self._prepare_graph_dataset(task_df)

        return task_df

    def _prepare_graph_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset specifically for graph neural networks."""
        # Separate vessel nodes and infrastructure nodes
        vessel_mask = df["node_type"].isin(["vessel_a", "vessel_b"])
        infrastructure_mask = df["node_type"] == "infrastructure"

        # Add temporal edges (vessel trajectories)
        df["has_temporal_edge"] = vessel_mask

        # Add spatial proximity indicators
        df["has_spatial_edge"] = False  # Placeholder for spatial edge detection

        return df

    def get_statistics(self) -> dict:
        """Get processing statistics."""
        stats = self.stats.copy()

        # Add task-specific statistics
        stats["task_coverage"] = {}
        for task in self.target_tasks:
            task_records = stats["task_specific_records"][task.value]
            total_records = stats["valid_records"]
            coverage = (task_records / total_records * 100) if total_records > 0 else 0
            stats["task_coverage"][task.value] = f"{coverage:.1f}%"

        return stats
