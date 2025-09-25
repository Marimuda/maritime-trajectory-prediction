"""
Tests for maritime coordinate transformation utilities.
"""

import numpy as np
import pytest

from src.models.baseline_models.kalman.coordinates import MaritimeCoordinateTransform


class TestMaritimeCoordinateTransform:
    """Test suite for maritime coordinate transformations."""

    def test_initialization(self):
        """Test basic initialization."""
        # Without reference point
        transform = MaritimeCoordinateTransform()
        assert transform.reference_lat is None
        assert transform.reference_lon is None

        # With reference point
        ref_point = (60.0, -7.0)
        transform = MaritimeCoordinateTransform(ref_point)
        assert transform.reference_lat == 60.0
        assert transform.reference_lon == -7.0

    def test_set_reference_point(self):
        """Test setting reference point."""
        transform = MaritimeCoordinateTransform()
        transform.set_reference_point(55.0, -10.0)

        assert transform.reference_lat == 55.0
        assert transform.reference_lon == -10.0
        assert transform.meters_per_degree_lat is not None
        assert transform.meters_per_degree_lon is not None

    def test_auto_set_reference(self):
        """Test automatic reference point setting."""
        positions = np.array([[60.0, -7.0], [60.1, -7.1], [59.9, -6.9]])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        # Reference should be approximately the centroid
        assert abs(transform.reference_lat - 60.0) < 0.1
        assert abs(transform.reference_lon - (-7.0)) < 0.1

    def test_coordinate_conversion_round_trip(self):
        """Test that coordinate conversions are invertible."""
        positions = np.array([[60.0, -7.0], [60.1, -7.1], [59.9, -6.9]])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        # Convert to local and back
        local_pos = transform.to_local(positions)
        restored_pos = transform.to_geographic(local_pos)

        np.testing.assert_allclose(positions, restored_pos, rtol=1e-10)

    def test_single_point_conversion(self):
        """Test conversion of single points."""
        transform = MaritimeCoordinateTransform((60.0, -7.0))

        # Single point as 1D array
        single_point = np.array([60.1, -7.1])
        local = transform.to_local(single_point)
        restored = transform.to_geographic(local)

        np.testing.assert_allclose(single_point, restored.flatten(), rtol=1e-10)

    def test_velocity_computation(self):
        """Test velocity computation from trajectory."""
        # Simple trajectory: vessel moving east at constant speed
        positions = np.array([[60.0, -7.0], [60.0, -6.9], [60.0, -6.8]])
        timestamps = np.array([0.0, 10.0, 20.0])  # 10-second intervals

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        velocities = transform.compute_velocity_local(positions, timestamps)

        # Should have 2 velocity vectors (n-1)
        assert velocities.shape == (2, 2)

        # Velocity should be primarily in x-direction (eastward)
        assert np.all(velocities[:, 0] > 0)  # Positive x velocity
        assert np.all(np.abs(velocities[:, 1]) < 0.1)  # Minimal y velocity

    def test_acceleration_computation(self):
        """Test acceleration computation from trajectory."""
        # Accelerating trajectory
        positions = np.array(
            [
                [60.0, -7.0],
                [60.0, -6.99],
                [60.0, -6.96],  # Increasing speed
                [60.0, -6.91],
            ]
        )
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        accelerations = transform.compute_acceleration_local(positions, timestamps)

        # Should have 2 acceleration vectors (n-2)
        assert accelerations.shape == (2, 2)

        # Should show positive acceleration in x-direction
        assert np.all(accelerations[:, 0] > 0)

    def test_heading_and_turn_rate_computation(self):
        """Test heading and turn rate computation."""
        # Trajectory with a turn
        positions = np.array(
            [
                [60.0, -7.0],  # Start
                [60.1, -7.0],  # Move north
                [60.1, -6.9],  # Turn east
                [60.0, -6.9],  # Turn south
            ]
        )
        timestamps = np.array([0.0, 10.0, 20.0, 30.0])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        headings, turn_rates = transform.compute_heading_and_turn_rate(
            positions, timestamps
        )

        # Should have n-1 headings and n-2 turn rates
        assert len(headings) == 3
        assert len(turn_rates) == 2

        # Turn rates should show the vessel is turning
        assert np.any(np.abs(turn_rates) > 0)

    def test_trajectory_validation(self):
        """Test trajectory validation for physical plausibility."""
        # Valid trajectory
        valid_positions = np.array(
            [
                [60.0, -7.0],
                [60.001, -7.001],  # Reasonable movement
                [60.002, -7.002],
            ]
        )
        timestamps = np.array([0.0, 10.0, 20.0])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(valid_positions)

        valid_mask, stats = transform.validate_trajectory(valid_positions, timestamps)

        assert np.all(valid_mask)  # All points should be valid
        assert stats["validity_rate"] == 1.0

        # Invalid trajectory with impossible speed
        invalid_positions = np.array(
            [
                [60.0, -7.0],
                [61.0, -7.0],  # 1 degree latitude in 1 second (~111 km/s)
                [62.0, -7.0],
            ]
        )
        timestamps = np.array([0.0, 1.0, 2.0])

        valid_mask, stats = transform.validate_trajectory(
            invalid_positions, timestamps, max_speed_ms=30.0
        )

        assert not np.all(valid_mask)  # Some points should be invalid
        assert stats["invalid_speed"] > 0

    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectories."""
        transform = MaritimeCoordinateTransform()

        with pytest.raises(ValueError):
            transform.auto_set_reference(np.array([]))

        with pytest.raises(ValueError):
            transform.compute_velocity_local(np.array([]), np.array([]))

    def test_coordinate_system_info(self):
        """Test coordinate system information retrieval."""
        transform = MaritimeCoordinateTransform()

        info = transform.get_reference_info()
        assert info["is_initialized"] is False

        transform.set_reference_point(60.0, -7.0)
        info = transform.get_reference_info()

        assert info["is_initialized"] is True
        assert info["reference_lat"] == 60.0
        assert info["reference_lon"] == -7.0
        assert info["meters_per_degree_lat"] is not None

    def test_large_time_gaps_detection(self):
        """Test detection of large time gaps in trajectories."""
        positions = np.array([[60.0, -7.0], [60.001, -7.001], [60.002, -7.002]])
        # Large gap between second and third points
        timestamps = np.array([0.0, 10.0, 4000.0])  # >1 hour gap

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        valid_mask, stats = transform.validate_trajectory(positions, timestamps)

        assert stats["time_gaps"] > 0
        assert not valid_mask[-1]  # Last point should be marked invalid due to gap

    def test_coordinate_accuracy_near_poles(self):
        """Test coordinate accuracy for high-latitude positions."""
        # High latitude positions
        positions = np.array([[80.0, -7.0], [80.001, -7.001], [80.002, -7.002]])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        # Conversion should still work but with different scaling
        local_pos = transform.to_local(positions)
        restored_pos = transform.to_geographic(local_pos)

        np.testing.assert_allclose(positions, restored_pos, rtol=1e-8)

        # Longitude scaling should be much smaller at high latitudes
        assert transform.meters_per_degree_lon < transform.meters_per_degree_lat

    def test_zero_time_intervals(self):
        """Test handling of zero time intervals."""
        positions = np.array([[60.0, -7.0], [60.001, -7.001], [60.002, -7.002]])
        # Duplicate timestamps
        timestamps = np.array([0.0, 0.0, 10.0])

        transform = MaritimeCoordinateTransform()
        transform.auto_set_reference(positions)

        # Should handle zero time intervals gracefully
        velocities = transform.compute_velocity_local(positions, timestamps)

        assert velocities.shape == (2, 2)
        assert np.all(np.isfinite(velocities))

    def test_minimum_sequence_length(self):
        """Test minimum sequence length requirements."""
        transform = MaritimeCoordinateTransform()

        # Single point should raise error
        single_point = np.array([[60.0, -7.0]])
        timestamps = np.array([0.0])

        with pytest.raises(ValueError):
            transform.compute_velocity_local(single_point, timestamps)

        # Two points should work for velocity
        two_points = np.array([[60.0, -7.0], [60.001, -7.001]])
        timestamps = np.array([0.0, 10.0])

        velocities = transform.compute_velocity_local(two_points, timestamps)
        assert velocities.shape == (1, 2)

        # Need at least 3 points for acceleration
        with pytest.raises(ValueError):
            transform.compute_acceleration_local(two_points, timestamps)
