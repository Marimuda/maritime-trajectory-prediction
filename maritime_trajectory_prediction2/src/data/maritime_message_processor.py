"""
Fixed AIS data processor with proper imports and real data handling.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ColumnNames, ValidationRanges

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

logger = logging.getLogger(__name__)


class AISProcessor:
    """
    Enhanced AIS data processor following guideline recommendations.
    """

    def __init__(self):
        """Initialize the AIS processor."""
        # Regex pattern for log line validation
        self.pattern = re.compile(r"^\d{4}-\d{2}-\d{2} .* - ")

        # ITU-R M.1371 sentinel values
        self.sentinel_values = {
            "lat": 91.0,
            "lon": 181.0,
            "speed": 102.3,
            "heading": 511,
            "draught": 25.5,
        }

        # Field mappings to standardized column names
        self.field_mapping = {
            "lat": ColumnNames.LAT,
            "lon": ColumnNames.LON,
            "speed": ColumnNames.SOG,
            "course": ColumnNames.COG,
        }

        # Message type classifications
        self.msg_classes = {
            1: "A_pos",
            2: "A_pos",
            3: "A_pos",
            4: "Base_pos",
            5: "Static",
            18: "B_pos",
            21: "ATON",
            24: "StaticB",
        }

        self.stats = {
            "lines_processed": 0,
            "valid_records": 0,
            "filtered_records": 0,
            "error_records": 0,
        }

    def parse_line(self, line: str) -> dict | None:
        """Parse a single AIS log line."""
        self.stats["lines_processed"] += 1

        # Handle None or non-string input
        if not isinstance(line, str):
            self.stats["error_records"] += 1
            return None

        # Fast reject non-matching lines
        if not self.pattern.match(line):
            self.stats["filtered_records"] += 1
            return None

        # Split timestamp and payload
        try:
            _, payload = line.split(" - ", 1)
        except ValueError:
            self.stats["error_records"] += 1
            return None

        # Skip non-JSON lines (engine chatter)
        if not payload.startswith("{"):
            self.stats["filtered_records"] += 1
            return None

        # Parse JSON
        try:
            if HAS_ORJSON:
                rec = orjson.loads(payload)
            else:
                rec = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            self.stats["error_records"] += 1
            return None

        # Validate required fields
        if not isinstance(rec, dict) or "type" not in rec:
            self.stats["error_records"] += 1
            return None

        # Parse timestamp
        if "rxtime" in rec:
            try:
                rec["time"] = pd.to_datetime(
                    rec["rxtime"], format="%Y%m%d%H%M%S", utc=True
                )
            except (ValueError, TypeError):
                self.stats["error_records"] += 1
                return None
        else:
            self.stats["error_records"] += 1
            return None

        # Apply CF-compliant field mappings
        for old_key, new_key in self.field_mapping.items():
            if old_key in rec:
                rec[new_key] = rec.pop(old_key)

        # Handle sentinel values → NaN (use mapped field names)
        sentinel_mappings = {
            ColumnNames.LAT: 91.0,
            ColumnNames.LON: 181.0,
            ColumnNames.SOG: 102.3,
            "heading": 511,
            "draught": 25.5,
        }
        for field, sentinel in sentinel_mappings.items():
            if field in rec and rec[field] == sentinel:
                rec[field] = np.nan

        # Additional range validation
        if not self._validate_ranges(rec):
            self.stats["error_records"] += 1
            return None

        # Classify message type
        msg_type = rec.get("type", 0)
        rec["msg_class"] = self.msg_classes.get(msg_type, "Other")

        self.stats["valid_records"] += 1
        return rec

    def _validate_ranges(self, rec: dict) -> bool:
        """Validate and clean field ranges."""
        # Position validation using schema ranges
        lat_range = ValidationRanges.RANGES[ColumnNames.LAT]
        if ColumnNames.LAT in rec and rec[ColumnNames.LAT] is not None:
            if not (lat_range[0] <= rec[ColumnNames.LAT] <= lat_range[1]):
                rec[ColumnNames.LAT] = np.nan

        lon_range = ValidationRanges.RANGES[ColumnNames.LON]
        if ColumnNames.LON in rec and rec[ColumnNames.LON] is not None:
            if not (lon_range[0] <= rec[ColumnNames.LON] <= lon_range[1]):
                rec[ColumnNames.LON] = np.nan

        # Speed validation
        if "sog" in rec and rec["sog"] is not None:
            if rec["sog"] < 0 or rec["sog"] >= 102.2:
                rec["sog"] = np.nan

        # MMSI validation - Allow special ranges
        if "mmsi" in rec and rec["mmsi"] is not None:
            mmsi = rec["mmsi"]
            # Standard vessel range: 100M-799M
            # Base station range: 2M-9M (coastal stations)
            # Aid to navigation: 990M-999M
            if not (
                (100_000_000 <= mmsi <= 799_999_999)  # Standard vessels
                or (2_000_000 <= mmsi <= 9_999_999)  # Base stations/coastal
                or (990_000_000 <= mmsi <= 999_999_999)
            ):  # Aids to navigation
                logger.debug(f"Rejecting record with invalid MMSI: {mmsi}")
                return False

        return True

    def process_file(self, file_path: str | Path) -> pd.DataFrame:
        """Process an entire AIS log file."""
        file_path = Path(file_path)
        records = []

        logger.info(f"Processing file: {file_path}")

        # Reset stats
        self.stats = {
            "lines_processed": 0,
            "valid_records": 0,
            "filtered_records": 0,
            "error_records": 0,
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    record = self.parse_line(line)
                    if record:
                        records.append(record)

                    # Progress logging
                    if line_num % 1000 == 0:
                        logger.info(
                            f"Processed {line_num} lines, {len(records)} valid records"
                        )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

        logger.info(
            f"Completed processing: {len(records)} valid records from {file_path}"
        )
        logger.info(f"Stats: {self.stats}")

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def clean_ais_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate AIS data."""
        if df.empty:
            return df

        original_count = len(df)

        # Remove records with invalid coordinates
        df = df.dropna(subset=[ColumnNames.LAT, ColumnNames.LON])

        # Remove records with invalid MMSI
        def is_valid_mmsi(mmsi):
            if pd.isna(mmsi):
                return False
            return (
                (100_000_000 <= mmsi <= 799_999_999)
                or (2_000_000 <= mmsi <= 9_999_999)
                or (990_000_000 <= mmsi <= 999_999_999)
            )

        df = df[df["mmsi"].apply(is_valid_mmsi)]

        # Sort by time
        if "time" in df.columns:
            df = df.sort_values("time")

        logger.info(f"Cleaned data: {len(df)}/{original_count} records retained")

        return df

    def get_statistics(self) -> dict:
        """Get processing statistics."""
        return self.stats.copy()


def load_ais_data(file_path: str | Path) -> pd.DataFrame:
    """Load AIS data from file."""
    processor = AISProcessor()
    return processor.process_file(file_path)


def preprocess_ais_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess AIS data."""
    processor = AISProcessor()
    return processor.clean_ais_data(df)
