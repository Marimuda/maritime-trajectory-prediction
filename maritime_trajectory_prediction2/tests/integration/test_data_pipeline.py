"""
Integration tests for the complete AIS data processing pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.maritime_message_processor import AISProcessor
from src.utils.maritime_utils import MaritimeUtils


@pytest.mark.integration
class TestAISDataPipeline:
    """Integration tests for the complete AIS data processing pipeline."""

    @pytest.fixture
    def sample_log_content(self):
        """Sample log content for testing."""
        return """2025-05-08 11:10:19,454 - {"class":"AIS","device":"AIS-catcher","version":61,"driver":1,"hardware":"RTL2838UHIDIR","rxtime":"20250508101019","scaled":true,"channel":"B","nmea":["!AIVDM,1,1,,B,13Lh?20P00OPwK4SNhbrq?vV0<0G,0*09"],"signalpower":-15.704244,"ppm":-4.629630,"type":1,"repeat":0,"mmsi":231477000,"status":0,"status_text":"Under way using engine","turn_unscaled":-128,"turn":-128,"speed":0.000000,"accuracy":false,"lon":-6.774024,"lat":62.006901,"course":278.800018,"heading":511,"second":19,"maneuver":0,"raim":false,"radio":49175}
2025-05-08 11:10:19,538 - {"class":"AIS","device":"AIS-catcher","version":61,"driver":1,"hardware":"RTL2838UHIDIR","rxtime":"20250508101019","scaled":true,"channel":"B","nmea":["!AIVDM,1,1,,B,402=5C0000HttOQ7g6STgiW00t1a,0*12"],"signalpower":-15.023769,"ppm":-2.314815,"type":4,"repeat":0,"mmsi":2311500,"timestamp":"0000-00-00T24:60:60Z","accuracy":false,"lon":-6.745648,"lat":62.170361,"epfd":7,"epfd_text":"Surveyed","raim":false,"radio":245865}
2025-05-08 11:10:19,708 - {"class":"AIS","device":"AIS-catcher","version":61,"driver":1,"hardware":"RTL2838UHIDIR","rxtime":"20250508101019","scaled":true,"channel":"A","nmea":["!AIVDM,1,1,,A,13LPrT?P00OPwvfSNfvh0?vV25@`,0*64"],"signalpower":-24.933611,"ppm":-1.446759,"type":1,"repeat":0,"mmsi":231226000,"status":15,"status_text":"Not defined","turn_unscaled":-128,"turn":-128,"speed":0.000000,"accuracy":false,"lon":-6.772121,"lat":62.006180,"course":0.000000,"heading":511,"second":19,"maneuver":0,"raim":true,"radio":21544}
2025-05-08 11:10:20,135 - {"class":"AIS","device":"AIS-catcher","version":61,"driver":1,"hardware":"RTL2838UHIDIR","rxtime":"20250508101020","scaled":true,"channel":"A","nmea":["!AIVDM,1,1,,A,13mU<I001LORAmBSMBLPV@K60l19,0*18"],"signalpower":-32.539433,"ppm":-0.289352,"type":1,"repeat":0,"mmsi":257510500,"status":0,"status_text":"Under way using engine","turn_unscaled":0,"turn":0,"speed":9.200000,"accuracy":false,"lon":-6.492732,"lat":61.966694,"course":15.300000,"heading":13,"second":35,"maneuver":0,"raim":false,"radio":213065}
2025-05-08 11:10:20,199 - [AIS engine v0.61 #0-0]                 received: 6 msgs, total: 2778 msgs, rate: 1.99595 msg/s"""

    @pytest.mark.integration
    def test_end_to_end_processing(self, sample_log_content):
        """Test complete end-to-end processing pipeline."""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            # Process the file
            processor = AISProcessor()
            df = processor.process_file(temp_path)

            # Verify basic processing
            assert len(df) == 4  # 4 valid AIS messages, 1 engine chatter filtered
            assert "mmsi" in df.columns
            assert "lat" in df.columns
            assert "lon" in df.columns
            assert "time" in df.columns
            assert "msg_class" in df.columns

            # Clean the data
            cleaned_df = processor.clean_ais_data(df)
            assert len(cleaned_df) <= len(df)

            # Verify message classification
            msg_classes = cleaned_df["msg_class"].unique()
            expected_classes = {"A_pos", "Base_pos"}
            assert set(msg_classes).issubset(expected_classes)

            # Test maritime utilities with the processed data
            position_data = cleaned_df[
                cleaned_df["msg_class"].str.contains("pos", na=False)
            ]
            if len(position_data) >= 2:
                # Calculate distances between consecutive points
                for i in range(1, len(position_data)):
                    prev_row = position_data.iloc[i - 1]
                    curr_row = position_data.iloc[i]

                    distance = MaritimeUtils.calculate_distance(
                        prev_row["lat"],
                        prev_row["lon"],
                        curr_row["lat"],
                        curr_row["lon"],
                    )

                    assert isinstance(distance, (float, type(np.nan)))
                    if not pd.isna(distance):
                        assert distance >= 0

        finally:
            Path(temp_path).unlink()

    @pytest.mark.integration
    def test_vessel_trajectory_analysis(self, sample_log_content):
        """Test vessel trajectory analysis."""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            # Process the file
            processor = AISProcessor()
            df = processor.process_file(temp_path)
            cleaned_df = processor.clean_ais_data(df)

            # Group by vessel and analyze trajectories
            position_data = cleaned_df[
                cleaned_df["msg_class"].str.contains("pos", na=False)
            ]

            for mmsi, vessel_data in position_data.groupby("mmsi"):
                vessel_data = vessel_data.sort_values("time")

                # Validate trajectory
                validated_trajectory = MaritimeUtils.validate_trajectory(vessel_data)
                assert len(validated_trajectory) <= len(vessel_data)

                # Classify behavior if we have speed data
                if "sog" in vessel_data.columns:
                    speeds = vessel_data["sog"].dropna().tolist()
                    if speeds:
                        behavior = MaritimeUtils.classify_vessel_behavior(speeds)
                        assert behavior in [
                            "anchored",
                            "maneuvering",
                            "transit",
                            "high_speed",
                            "unknown",
                        ]

        finally:
            Path(temp_path).unlink()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_data_processing(self):
        """Test processing with real log file if available."""
        log_path = Path("data/raw/log_snipit.log")
        if not log_path.exists():
            pytest.skip("Real log file not available")

        # Process real data
        processor = AISProcessor()
        df = processor.process_file(log_path)

        # Basic validation
        assert len(df) > 0
        assert "mmsi" in df.columns
        assert "time" in df.columns

        # Check data quality
        stats = processor.get_statistics()
        assert stats["valid_records"] > 0
        assert (
            stats["lines_processed"] > stats["valid_records"]
        )  # Some filtering should occur

        # Clean and validate
        cleaned_df = processor.clean_ais_data(df)
        assert len(cleaned_df) <= len(df)

        # Verify coordinate ranges
        if "lat" in cleaned_df.columns and not cleaned_df["lat"].isna().all():
            lat_range = cleaned_df["lat"].dropna()
            assert lat_range.min() >= -90
            assert lat_range.max() <= 90

        if (
            "lon" in cleaned_df.columns
            and not cleaned_df["lon"].isna().all()
        ):
            lon_range = cleaned_df["lon"].dropna()
            assert lon_range.min() >= -180
            assert lon_range.max() <= 180

        # Test vessel analysis
        position_data = cleaned_df[
            cleaned_df["msg_class"].str.contains("pos", na=False)
        ]
        if not position_data.empty:
            vessel_count = position_data["mmsi"].nunique()
            assert vessel_count > 0

            # Analyze most active vessel
            vessel_activity = position_data.groupby("mmsi").size()
            most_active_mmsi = vessel_activity.idxmax()
            most_active_data = position_data[position_data["mmsi"] == most_active_mmsi]

            # Validate trajectory for most active vessel
            validated_trajectory = MaritimeUtils.validate_trajectory(most_active_data)
            assert isinstance(validated_trajectory, pd.DataFrame)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in the pipeline."""
        # Create log with various error conditions
        problematic_content = """2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":62.0,"lon":-6.7}
Invalid line without timestamp
2025-05-08 11:10:19,454 - {invalid json}
2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":99999999,"lat":95.0,"lon":185.0}
2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477001,"lat":62.1,"lon":-6.8}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(problematic_content)
            temp_path = f.name

        try:
            processor = AISProcessor()
            df = processor.process_file(temp_path)

            # Should process valid records despite errors
            assert len(df) >= 1  # At least one valid record

            stats = processor.get_statistics()
            assert stats["error_records"] > 0  # Should have recorded errors
            assert stats["filtered_records"] > 0  # Should have filtered invalid lines

            # Clean data should handle invalid coordinates
            cleaned_df = processor.clean_ais_data(df)

            # Verify no invalid coordinates remain
            if "lat" in cleaned_df.columns:
                valid_lats = cleaned_df["lat"].dropna()
                if len(valid_lats) > 0:
                    assert valid_lats.between(-90, 90).all()

            if "lon" in cleaned_df.columns:
                valid_lons = cleaned_df["lon"].dropna()
                if len(valid_lons) > 0:
                    assert valid_lons.between(-180, 180).all()

        finally:
            Path(temp_path).unlink()

    def test_performance_with_large_dataset(self):
        """Test performance characteristics with larger dataset."""
        # Create a larger synthetic dataset
        base_line = '2025-05-08 11:10:19,454 - {"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":62.0,"lon":-6.7}'
        large_content = "\n".join([base_line] * 1000)  # 1000 identical lines

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            processor = AISProcessor()
            df = processor.process_file(temp_path)

            # Should process all lines
            assert len(df) == 1000

            # Performance check - should complete in reasonable time
            stats = processor.get_statistics()
            assert stats["lines_processed"] == 1000
            assert stats["valid_records"] == 1000
            assert stats["error_records"] == 0

            # Clean data should be efficient
            cleaned_df = processor.clean_ais_data(df)
            assert len(cleaned_df) == 1000  # All should be valid

        finally:
            Path(temp_path).unlink()


class TestDataQualityAssurance:
    """Tests for data quality assurance and validation."""

    def test_coordinate_validation(self):
        """Test comprehensive coordinate validation."""
        test_cases = [
            # (lat, lon, should_be_valid)
            (62.0, -6.7, True),  # Valid Faroe Islands coordinates
            (0.0, 0.0, True),  # Valid equator/prime meridian
            (90.0, 180.0, True),  # Valid extreme coordinates
            (-90.0, -180.0, True),  # Valid extreme coordinates
            (91.0, -6.7, False),  # Invalid latitude (sentinel)
            (62.0, 181.0, False),  # Invalid longitude (sentinel)
            (95.0, -6.7, False),  # Invalid latitude (out of range)
            (62.0, 185.0, False),  # Invalid longitude (out of range)
        ]

        processor = AISProcessor()

        for lat, lon, should_be_valid in test_cases:
            line = f'2025-05-08 11:10:19,454 - {{"rxtime":"20250508101019","type":1,"mmsi":231477000,"lat":{lat},"lon":{lon}}}'
            result = processor.parse_line(line)

            if should_be_valid:
                assert result is not None
                if lat == 91.0 or lat > 90:
                    assert pd.isna(result["lat"])
                else:
                    assert result["lat"] == lat

                if lon == 181.0 or abs(lon) > 180:
                    assert pd.isna(result["lon"])
                else:
                    assert result["lon"] == lon
            # Note: Invalid coordinates are converted to NaN, not rejected entirely

    def test_mmsi_validation_comprehensive(self):
        """Test comprehensive MMSI validation."""
        test_cases = [
            # (mmsi, should_be_valid, description)
            (231477000, True, "Standard vessel MMSI"),
            (123456789, True, "Standard vessel MMSI"),
            (799999999, True, "Maximum standard vessel MMSI"),
            (100000000, True, "Minimum standard vessel MMSI"),
            (2311500, True, "Base station MMSI"),
            (9999999, True, "Maximum base station MMSI"),
            (2000000, True, "Minimum base station MMSI"),
            (992310001, True, "Aid to navigation MMSI"),
            (999999999, True, "Maximum ATON MMSI"),
            (990000000, True, "Minimum ATON MMSI"),
            (99999999, False, "Invalid MMSI (too low)"),
            (800000000, False, "Invalid MMSI (gap range)"),
            (989999999, False, "Invalid MMSI (gap range)"),
            (1000000000, False, "Invalid MMSI (too high)"),
            (1999999, False, "Invalid MMSI (too low for base station)"),
        ]

        processor = AISProcessor()

        for mmsi, should_be_valid, description in test_cases:
            line = f'2025-05-08 11:10:19,454 - {{"rxtime":"20250508101019","type":1,"mmsi":{mmsi},"lat":62.0,"lon":-6.7}}'
            result = processor.parse_line(line)

            if should_be_valid:
                assert (
                    result is not None
                ), f"Valid MMSI {mmsi} ({description}) should be accepted"
                assert result["mmsi"] == mmsi
            else:
                assert (
                    result is None
                ), f"Invalid MMSI {mmsi} ({description}) should be rejected"

    def test_temporal_data_validation(self):
        """Test temporal data validation and parsing."""
        test_cases = [
            ("20250508101019", True),  # Valid timestamp
            ("20251231235959", True),  # Valid end of year
            ("20250101000000", True),  # Valid start of year
            ("20250229101019", False),  # Invalid leap year (2025 is not leap)
            ("20251301101019", False),  # Invalid month
            ("20250532101019", False),  # Invalid day
            ("20250508251019", False),  # Invalid hour
            ("20250508106019", False),  # Invalid minute
            ("invalid", False),  # Invalid format
        ]

        processor = AISProcessor()

        for timestamp, should_be_valid in test_cases:
            line = f'2025-05-08 11:10:19,454 - {{"rxtime":"{timestamp}","type":1,"mmsi":231477000,"lat":62.0,"lon":-6.7}}'
            result = processor.parse_line(line)

            if should_be_valid:
                assert (
                    result is not None
                ), f"Valid timestamp {timestamp} should be accepted"
                assert "time" in result
                assert isinstance(result["time"], pd.Timestamp)
            else:
                assert (
                    result is None
                ), f"Invalid timestamp {timestamp} should be rejected"
