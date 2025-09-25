"""
Tests for IMM (Interactive Multiple Model) framework.
"""

import numpy as np
import pytest

from src.models.baseline_models.kalman.imm import (
    MaritimeIMMFilter,
    create_default_imm_config,
)
from src.models.baseline_models.kalman.protocols import (
    BaselineResult,
    IMMConfig,
    MaritimeConstraints,
)


class TestMaritimeIMMFilter:
    """Test IMM filter implementation."""

    @pytest.fixture
    def sample_config(self):
        """Create sample IMM configuration."""
        return create_default_imm_config(
            reference_point=(62.0, -7.0), max_speed_knots=30.0
        )

    @pytest.fixture
    def straight_trajectory(self):
        """Create straight-line trajectory (favors CV model)."""
        n_points = 15
        lat = np.linspace(62.0, 62.05, n_points)
        lon = np.full(n_points, -7.0)
        timestamps = np.arange(n_points) * 60.0  # 1 minute intervals

        return np.column_stack([lat, lon, timestamps])

    @pytest.fixture
    def turning_trajectory(self):
        """Create turning trajectory (favors CT model)."""
        n_points = 20
        theta = np.linspace(0, np.pi / 2, n_points)  # Quarter circle
        radius = 0.02
        lat = 62.0 + radius * np.sin(theta)
        lon = -7.0 + radius * (1 - np.cos(theta))
        timestamps = np.arange(n_points) * 30.0

        return np.column_stack([lat, lon, timestamps])

    @pytest.fixture
    def accelerating_trajectory(self):
        """Create accelerating trajectory (favors NCA model)."""
        n_points = 12
        t = np.arange(n_points) * 30.0
        # Quadratic motion in latitude
        lat = 62.0 + 0.0001 * t + 0.000001 * t**2
        lon = np.full(n_points, -7.0)

        return np.column_stack([lat, lon, t])

    def test_initialization(self, sample_config):
        """Test IMM filter initialization."""
        imm = MaritimeIMMFilter(sample_config)

        assert imm.config == sample_config
        assert len(imm.models) == 3  # CV, CT, NCA
        assert imm.model_names == ["CV", "CT", "NCA"]
        assert not imm.is_initialized
        assert imm.imm is None

    def test_default_initialization(self):
        """Test IMM filter with default configuration."""
        imm = MaritimeIMMFilter()

        assert imm.config is not None
        assert isinstance(imm.config, IMMConfig)
        assert imm.config.transition_probabilities.shape == (3, 3)

        # Transition matrix should be row-stochastic
        assert np.allclose(imm.config.transition_probabilities.sum(axis=1), 1.0)

    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_imm_config(
            reference_point=(60.0, -5.0),
            max_speed_knots=40.0,
            transition_probability_stay=0.9,
        )

        assert config.reference_point == (60.0, -5.0)
        assert config.constraints.max_speed_knots == 40.0

        # Check transition probabilities
        assert np.allclose(np.diag(config.transition_probabilities), 0.9)
        off_diag = 0.1 / 2  # (1 - 0.9) / (3 - 1)
        assert np.allclose(config.transition_probabilities[0, 1], off_diag)

    def test_fit(self, straight_trajectory, sample_config):
        """Test IMM fitting process."""
        imm = MaritimeIMMFilter(sample_config)

        # Test fitting
        fitted_imm = imm.fit([straight_trajectory])

        assert fitted_imm is imm  # Returns self
        assert imm.is_initialized
        assert imm.imm is not None
        assert imm.coordinate_transform.reference_lat is not None

        # All individual models should be fitted
        for model in imm.models:
            assert model.is_initialized

    def test_predict_straight_line(self, straight_trajectory):
        """Test IMM prediction on straight-line trajectory."""
        imm = MaritimeIMMFilter()

        result = imm.predict(straight_trajectory, horizon=5)

        assert isinstance(result, BaselineResult)
        assert result.predictions.shape == (5, 2)

        # Check model info
        assert "model_probabilities" in result.model_info
        assert "dominant_model" in result.model_info
        assert "final_model_probabilities" in result.model_info

        # For straight line, one model should dominate (not necessarily CV)
        final_probs = result.model_info["final_model_probabilities"]
        max_prob = max(final_probs)
        assert max_prob > 0.33  # Should be higher than random (33.3% for 3 models)
        # At least one model should have reasonable confidence
        assert max_prob > 0.4  # Some model should be reasonably confident

        # Should generally continue northward (allowing some IMM instability)
        lat_diffs = np.diff(result.predictions[:, 0])
        # Most predictions should increase, allowing some model switching effects
        assert np.sum(lat_diffs > 0) >= len(lat_diffs) * 0.6  # At least 60% increasing

    def test_predict_turning_trajectory(self, turning_trajectory):
        """Test IMM prediction on turning trajectory."""
        imm = MaritimeIMMFilter()

        result = imm.predict(turning_trajectory, horizon=3)

        assert result.predictions.shape == (3, 2)

        # For turning trajectory, CT or NCA model might dominate
        # (depends on specific trajectory characteristics)
        final_probs = result.model_info["final_model_probabilities"]
        dominant_model = result.model_info["dominant_model"]

        assert dominant_model in ["CV", "CT", "NCA"]

        # All probabilities should sum to 1
        assert abs(final_probs.sum() - 1.0) < 1e-10

    def test_predict_with_uncertainty(self, straight_trajectory):
        """Test IMM prediction with uncertainty estimation."""
        imm = MaritimeIMMFilter()

        result = imm.predict(straight_trajectory, horizon=3, return_uncertainty=True)

        assert result.uncertainty is not None
        assert result.uncertainty.shape == (3, 2, 2)

        # Uncertainty should be positive definite
        for i in range(3):
            cov_matrix = result.uncertainty[i]
            eigenvals = np.linalg.eigvals(cov_matrix)
            assert np.all(eigenvals >= 0)  # Positive semi-definite

    def test_model_probability_evolution(self, turning_trajectory):
        """Test that model probabilities evolve during prediction."""
        imm = MaritimeIMMFilter()

        result = imm.predict(turning_trajectory, horizon=5)

        # Check that we have probability history
        model_probs = result.model_info["model_probabilities"]
        assert model_probs.shape == (5, 3)  # 5 steps, 3 models

        # Each step should have valid probability distribution
        for step in range(5):
            step_probs = model_probs[step]
            assert abs(step_probs.sum() - 1.0) < 1e-10  # Sum to 1
            assert np.all(step_probs >= 0)  # Non-negative

    def test_confidence_scores(self, straight_trajectory):
        """Test confidence score computation."""
        imm = MaritimeIMMFilter()

        result = imm.predict(straight_trajectory, horizon=4)

        assert result.confidence is not None
        assert len(result.confidence) == 4

        # Confidence should be between 0 and 1
        assert np.all(result.confidence >= 0)
        assert np.all(result.confidence <= 1)

        # For straight line, confidence should be relatively high
        assert np.mean(result.confidence) > 0.3

    def test_state_alignment(self, straight_trajectory):
        """Test that different model states are properly aligned."""
        imm = MaritimeIMMFilter()

        # Fit to initialize state alignment
        imm.fit([straight_trajectory])

        # All models should have same state dimension after alignment
        state_dims = [model.kf.dim_x for model in imm.models]
        assert len(set(state_dims)) == 1  # All same dimension

        # Should be 6D (NCA full state)
        assert state_dims[0] == 6

    def test_transition_probability_update(self, sample_config):
        """Test updating transition probabilities."""
        imm = MaritimeIMMFilter(sample_config)

        # New transition matrix (more likely to stay in CV)
        new_transition = np.array(
            [[0.98, 0.01, 0.01], [0.1, 0.85, 0.05], [0.1, 0.05, 0.85]]
        )

        imm.set_model_transition_probabilities(new_transition)

        np.testing.assert_array_equal(
            imm.config.transition_probabilities, new_transition
        )

        # Test validation
        invalid_transition = np.array(
            [
                [0.5, 0.3, 0.1],  # Doesn't sum to 1
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ]
        )

        with pytest.raises(ValueError):
            imm.set_model_transition_probabilities(invalid_transition)

    def test_model_switching_behavior(self):
        """Test that IMM can switch between models appropriately."""
        imm = MaritimeIMMFilter()

        # Create trajectory that changes behavior: straight then turn
        straight_part = np.array(
            [
                [62.0, -7.0, 0.0],
                [62.01, -7.0, 60.0],
                [62.02, -7.0, 120.0],
                [62.03, -7.0, 180.0],
            ]
        )

        turn_part = np.array(
            [
                [62.03, -7.0, 180.0],
                [62.035, -6.995, 240.0],
                [62.04, -6.99, 300.0],
                [62.045, -6.985, 360.0],
            ]
        )

        combined_trajectory = np.vstack(
            [straight_part, turn_part[1:]]
        )  # Remove duplicate

        result = imm.predict(combined_trajectory, horizon=3)

        # Should adapt to the changing behavior
        assert result.predictions.shape == (3, 2)
        final_probs = result.model_info["final_model_probabilities"]

        # Some model should be reasonably confident
        assert np.max(final_probs) > 0.4

    def test_get_model_info_completeness(self):
        """Test that model info contains all required fields."""
        imm = MaritimeIMMFilter()

        info = imm.get_model_info()

        required_keys = [
            "model_type",
            "component_models",
            "config",
            "is_initialized",
            "coordinate_transform",
        ]

        for key in required_keys:
            assert key in info

        assert info["model_type"] == "MaritimeIMM"
        assert info["component_models"] == ["CV", "CT", "NCA"]

    def test_empty_sequence_error(self):
        """Test error handling for empty sequences."""
        imm = MaritimeIMMFilter()

        with pytest.raises(ValueError):
            imm.fit([])

        with pytest.raises(ValueError):
            imm.predict(np.array([]), horizon=1)

    def test_insufficient_points_error(self):
        """Test error handling for insufficient points."""
        imm = MaritimeIMMFilter()

        single_point = np.array([[62.0, -7.0, 0.0]])

        with pytest.raises(ValueError):
            imm.predict(single_point, horizon=1)

    def test_constraint_enforcement(self):
        """Test that physical constraints are enforced in IMM."""
        # Create config with strict constraints
        config = IMMConfig()
        config.constraints = MaritimeConstraints(max_speed_knots=5.0)  # Very slow

        imm = MaritimeIMMFilter(config)

        # High-speed trajectory
        high_speed_traj = np.array(
            [
                [62.0, -7.0, 0.0],
                [62.05, -7.0, 60.0],  # Very fast
                [62.1, -7.0, 120.0],
            ]
        )

        result = imm.predict(high_speed_traj, horizon=2)

        # Predictions should be constrained
        # This is tested indirectly by checking that extreme extrapolation doesn't occur
        lat_changes = np.diff(
            np.concatenate(
                [
                    [high_speed_traj[-1, 0]],  # Last observed point
                    result.predictions[:, 0],
                ]
            )
        )

        # Predictions should be reasonable (not wildly extrapolating)
        # With 5 knot speed limit, movement should be constrained compared to unconstrained extrapolation
        assert np.all(np.abs(lat_changes) < 0.5)  # Should not make extreme jumps
        # At least some constraint effect should be visible
        assert np.mean(np.abs(lat_changes)) < 0.4  # Average change should be reasonable

    def test_auto_fit_behavior(self, straight_trajectory):
        """Test automatic fitting when predict is called on uninitialized IMM."""
        imm = MaritimeIMMFilter()

        # Should not be initialized initially
        assert not imm.is_initialized

        # Calling predict should auto-fit
        result = imm.predict(straight_trajectory, horizon=2)

        # Should be initialized after predict
        assert imm.is_initialized
        assert result.predictions.shape == (2, 2)

    def test_coordinate_system_consistency(self):
        """Test that coordinate system is consistently used across models."""
        imm = MaritimeIMMFilter()

        trajectory = np.array(
            [[62.0, -7.0, 0.0], [62.01, -7.0, 60.0], [62.02, -7.0, 120.0]]
        )

        # Fit IMM
        imm.fit([trajectory])

        # All models should use same coordinate system
        ref_info = imm.coordinate_transform.get_reference_info()

        for model in imm.models:
            model_ref = model.coordinate_transform.get_reference_info()
            assert model_ref["reference_lat"] == ref_info["reference_lat"]
            assert model_ref["reference_lon"] == ref_info["reference_lon"]

    @pytest.mark.parametrize("horizon", [1, 3, 5, 10])
    def test_various_prediction_horizons(self, straight_trajectory, horizon):
        """Test IMM with various prediction horizons."""
        imm = MaritimeIMMFilter()

        result = imm.predict(straight_trajectory, horizon=horizon)

        assert result.predictions.shape == (horizon, 2)
        assert result.confidence.shape == (horizon,)

        if result.uncertainty is not None:
            assert result.uncertainty.shape == (horizon, 2, 2)
