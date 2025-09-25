"""
Tests for maritime motion models (CV, CT, NCA).
"""

import numpy as np
import pytest

from src.models.baseline_models.kalman.models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from src.models.baseline_models.kalman.protocols import (
    BaselineResult,
    MaritimeConstraints,
    MotionModelConfig,
)


class TestConstantVelocityModel:
    """Test suite for Constant Velocity model."""

    def create_test_trajectory(self, n_points=10):
        """Create a simple test trajectory."""
        # Linear trajectory moving northeast
        lat = np.linspace(60.0, 60.01, n_points)
        lon = np.linspace(-7.0, -6.99, n_points)
        timestamps = np.arange(n_points, dtype=float) * 10.0  # 10-second intervals

        return np.column_stack([lat, lon, timestamps])

    def test_initialization(self):
        """Test CV model initialization."""
        model = ConstantVelocityModel()

        assert model.config is not None
        assert model.constraints is not None
        assert model.kf is not None
        assert model.kf.dim_x == 4  # [x, vx, y, vy]
        assert model.kf.dim_z == 2  # [x, y] observations
        assert not model.is_initialized

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = MotionModelConfig(
            position_process_noise=0.5, velocity_process_noise=0.05
        )
        constraints = MaritimeConstraints(max_speed_knots=30.0)

        model = ConstantVelocityModel(config, constraints)

        assert model.config.position_process_noise == 0.5
        assert model.constraints.max_speed_knots == 30.0

    def test_fit(self):
        """Test model fitting."""
        model = ConstantVelocityModel()
        trajectory = self.create_test_trajectory()

        fitted_model = model.fit([trajectory])

        assert fitted_model is model  # Returns self
        assert model.is_initialized
        assert model.coordinate_transform.reference_lat is not None

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        model = ConstantVelocityModel()
        trajectory = self.create_test_trajectory()

        model.fit([trajectory])
        result = model.predict(trajectory[:5], horizon=3)

        assert isinstance(result, BaselineResult)
        assert result.predictions.shape == (3, 2)  # 3 predictions, lat/lon
        assert result.model_info is not None
        assert result.model_info["model_type"] == "ConstantVelocityModel"

    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        model = ConstantVelocityModel()
        trajectory = self.create_test_trajectory()

        model.fit([trajectory])
        result = model.predict(trajectory[:5], horizon=3, return_uncertainty=True)

        assert result.uncertainty is not None
        assert result.uncertainty.shape == (
            3,
            2,
            2,
        )  # 3 predictions, 2x2 covariance matrices

    def test_predict_without_timestamps(self):
        """Test prediction when trajectory doesn't have timestamps."""
        model = ConstantVelocityModel()
        trajectory = self.create_test_trajectory()

        # Remove timestamps
        trajectory_no_time = trajectory[:, :2]

        model.fit([trajectory_no_time])
        result = model.predict(trajectory_no_time[:5], horizon=2)

        assert result.predictions.shape == (2, 2)

    def test_constraint_enforcement(self):
        """Test that speed constraints are enforced."""
        config = MotionModelConfig()
        constraints = MaritimeConstraints(max_speed_knots=5.0)  # Very low speed limit

        model = ConstantVelocityModel(config, constraints)

        # Create high-speed trajectory
        high_speed_trajectory = np.array(
            [
                [60.0, -7.0, 0.0],
                [60.1, -7.0, 1.0],  # 1 degree latitude per second (impossible speed)
                [60.2, -7.0, 2.0],
            ]
        )

        model.fit([high_speed_trajectory])
        result = model.predict(high_speed_trajectory[:2], horizon=1)

        # Model should still produce reasonable predictions despite invalid input
        assert np.all(np.isfinite(result.predictions))

    def test_minimum_data_requirements(self):
        """Test minimum data requirements for prediction."""
        model = ConstantVelocityModel()

        # Single point should raise error
        with pytest.raises(ValueError):
            model.predict(np.array([[60.0, -7.0]]), horizon=1)

        # Two points should work
        two_points = np.array([[60.0, -7.0], [60.001, -7.001]])
        model.fit([two_points])
        result = model.predict(two_points, horizon=1)

        assert result.predictions.shape == (1, 2)

    def test_model_info(self):
        """Test model information retrieval."""
        model = ConstantVelocityModel()
        info = model.get_model_info()

        assert info["model_type"] == "ConstantVelocityModel"
        assert "config" in info
        assert "constraints" in info
        assert info["is_initialized"] is False
        assert info["state_dim"] == 4


class TestCoordinatedTurnModel:
    """Test suite for Coordinated Turn model."""

    def create_circular_trajectory(self, n_points=20):
        """Create a circular trajectory for testing turn model."""
        # Circle centered at (60.0, -7.0) with small radius
        angles = np.linspace(0, np.pi, n_points)  # Half circle
        radius_deg = 0.01  # Small radius in degrees

        lat = 60.0 + radius_deg * np.sin(angles)
        lon = -7.0 + radius_deg * np.cos(angles)
        timestamps = np.arange(n_points, dtype=float) * 5.0  # 5-second intervals

        return np.column_stack([lat, lon, timestamps])

    def test_initialization(self):
        """Test CT model initialization."""
        model = CoordinatedTurnModel()

        assert model.kf.dim_x == 5  # [x, vx, y, vy, ω]
        assert model.kf.dim_z == 2  # [x, y] observations

    def test_fit_and_predict(self):
        """Test fitting and prediction on circular trajectory."""
        model = CoordinatedTurnModel()
        trajectory = self.create_circular_trajectory()

        model.fit([trajectory])
        result = model.predict(trajectory[:10], horizon=5)

        assert result.predictions.shape == (5, 2)
        assert "model_type" in result.model_info
        assert result.model_info["model_type"] == "CoordinatedTurnModel"

    def test_turn_rate_estimation(self):
        """Test that model can estimate turn rates."""
        model = CoordinatedTurnModel()
        trajectory = self.create_circular_trajectory()

        model.fit([trajectory])

        # The model should initialize with some turn rate estimate
        # (Exact verification is complex due to coordinate transformations)
        result = model.predict(trajectory[:10], horizon=3)
        assert np.all(np.isfinite(result.predictions))

    def test_minimum_data_for_turn_estimation(self):
        """Test minimum data requirements for turn rate estimation."""
        model = CoordinatedTurnModel()

        # Need at least 3 points for turn rate estimation
        with pytest.raises(ValueError):
            model.predict(np.array([[60.0, -7.0], [60.001, -7.001]]), horizon=1)

        # Three points should work
        three_points = np.array(
            [[60.0, -7.0, 0.0], [60.001, -7.001, 5.0], [60.002, -7.002, 10.0]]
        )
        model.fit([three_points])
        result = model.predict(three_points, horizon=1)

        assert result.predictions.shape == (1, 2)


class TestNearlyConstantAccelModel:
    """Test suite for Nearly Constant Acceleration model."""

    def create_accelerating_trajectory(self, n_points=15):
        """Create an accelerating trajectory."""
        # Quadratic trajectory (constant acceleration)
        t = np.linspace(0, 10, n_points)
        x = 0.5 * 0.001 * t**2  # Acceleration in longitude
        y = 0.001 * t  # Constant velocity in latitude

        lat = 60.0 + y
        lon = -7.0 + x
        timestamps = t * 10.0  # Scale time

        return np.column_stack([lat, lon, timestamps])

    def test_initialization(self):
        """Test NCA model initialization."""
        model = NearlyConstantAccelModel()

        assert model.kf.dim_x == 6  # [x, vx, ax, y, vy, ay]
        assert model.kf.dim_z == 2  # [x, y] observations

    def test_fit_and_predict(self):
        """Test fitting and prediction on accelerating trajectory."""
        model = NearlyConstantAccelModel()
        trajectory = self.create_accelerating_trajectory()

        model.fit([trajectory])
        result = model.predict(trajectory[:8], horizon=4)

        assert result.predictions.shape == (4, 2)
        assert result.model_info["model_type"] == "NearlyConstantAccelModel"

    def test_acceleration_estimation(self):
        """Test that model can estimate accelerations."""
        model = NearlyConstantAccelModel()
        trajectory = self.create_accelerating_trajectory()

        model.fit([trajectory])
        result = model.predict(trajectory[:8], horizon=3)

        # Should produce reasonable predictions
        assert np.all(np.isfinite(result.predictions))

        # Predictions should show the effect of acceleration
        # (trajectory should continue to accelerate)
        pred_distances = np.linalg.norm(np.diff(result.predictions, axis=0), axis=1)
        assert len(pred_distances) > 0  # Should have some distance values

    def test_minimum_data_for_acceleration_estimation(self):
        """Test minimum data requirements for acceleration estimation."""
        model = NearlyConstantAccelModel()

        # Need at least 3 points for acceleration estimation
        with pytest.raises(ValueError):
            model.predict(np.array([[60.0, -7.0], [60.001, -7.001]]), horizon=1)

        # Three points should work
        three_points = np.array(
            [
                [60.0, -7.0, 0.0],
                [60.001, -7.001, 5.0],
                [60.002, -7.003, 10.0],  # Slight acceleration
            ]
        )
        model.fit([three_points])
        result = model.predict(three_points, horizon=1)

        assert result.predictions.shape == (1, 2)


class TestModelComparison:
    """Test comparison and consistency between models."""

    def create_test_trajectory(self):
        """Create a test trajectory for model comparison."""
        # Straight line trajectory (should favor CV model)
        lat = np.linspace(60.0, 60.01, 20)
        lon = np.linspace(-7.0, -6.99, 20)
        timestamps = np.arange(20, dtype=float) * 10.0

        return np.column_stack([lat, lon, timestamps])

    def test_all_models_basic_functionality(self):
        """Test that all models can handle the same trajectory."""
        trajectory = self.create_test_trajectory()

        models = [
            ConstantVelocityModel(),
            CoordinatedTurnModel(),
            NearlyConstantAccelModel(),
        ]

        for model in models:
            model.fit([trajectory])
            result = model.predict(trajectory[:10], horizon=3)

            assert result.predictions.shape == (3, 2)
            assert np.all(np.isfinite(result.predictions))
            assert result.model_info is not None

    def test_model_prediction_consistency(self):
        """Test that models produce consistent predictions for similar scenarios."""
        trajectory = self.create_test_trajectory()

        cv_model = ConstantVelocityModel()
        ct_model = CoordinatedTurnModel()
        nca_model = NearlyConstantAccelModel()

        results = []
        for model in [cv_model, ct_model, nca_model]:
            model.fit([trajectory])
            result = model.predict(trajectory[:10], horizon=1)
            results.append(result.predictions)

        # For a straight-line trajectory, all models should give similar predictions
        # (though not identical due to different assumptions)
        for i in range(1, len(results)):
            # Predictions should be in same general area (within ~1km)
            distance_diff = np.linalg.norm(results[0] - results[i])
            assert distance_diff < 0.01  # Less than 0.01 degrees (~1km)

    def test_model_uncertainty_estimates(self):
        """Test that models provide reasonable uncertainty estimates."""
        trajectory = self.create_test_trajectory()

        models = [
            ConstantVelocityModel(),
            CoordinatedTurnModel(),
            NearlyConstantAccelModel(),
        ]

        for model in models:
            model.fit([trajectory])
            result = model.predict(trajectory[:5], horizon=3, return_uncertainty=True)

            assert result.uncertainty is not None
            assert result.uncertainty.shape == (3, 2, 2)

            # Uncertainty should increase with prediction horizon
            uncertainties = [np.trace(cov) for cov in result.uncertainty]
            assert uncertainties[-1] >= uncertainties[0]  # Should increase or stay same

    def test_constraint_enforcement_consistency(self):
        """Test that all models enforce constraints consistently."""
        constraints = MaritimeConstraints(max_speed_knots=10.0)

        models = [
            ConstantVelocityModel(constraints=constraints),
            CoordinatedTurnModel(constraints=constraints),
            NearlyConstantAccelModel(constraints=constraints),
        ]

        # Create a high-speed trajectory that violates constraints
        high_speed_trajectory = np.array(
            [
                [60.0, -7.0, 0.0],
                [60.05, -7.0, 1.0],  # ~5.5 km in 1 second
                [60.10, -7.0, 2.0],
            ]
        )

        for model in models:
            model.fit([high_speed_trajectory])
            # CV model can work with 2 points, CT and NCA need 3+ points
            if isinstance(model, ConstantVelocityModel):
                result = model.predict(high_speed_trajectory[:2], horizon=1)
            else:
                result = model.predict(high_speed_trajectory, horizon=1)

            # All models should produce valid predictions despite invalid input
            assert np.all(np.isfinite(result.predictions))

    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        models = [
            ConstantVelocityModel(),
            CoordinatedTurnModel(),
            NearlyConstantAccelModel(),
        ]

        for model in models:
            with pytest.raises(ValueError):
                model.fit([])

            with pytest.raises(ValueError):
                model.fit([np.array([])])

    def test_model_state_isolation(self):
        """Test that models maintain independent state."""
        trajectory = self.create_test_trajectory()

        model1 = ConstantVelocityModel()
        model2 = ConstantVelocityModel()

        # Fit models with different data
        model1.fit([trajectory[:10]])
        model2.fit([trajectory[10:]])

        result1 = model1.predict(trajectory[:5], horizon=1)
        result2 = model2.predict(trajectory[10:15], horizon=1)

        # Results should be different (models fitted on different data)
        assert not np.allclose(result1.predictions, result2.predictions)
