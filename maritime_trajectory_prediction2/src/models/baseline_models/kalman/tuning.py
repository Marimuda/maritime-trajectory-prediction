"""
Hyperparameter tuning system for maritime Kalman filter baselines.

Provides automated optimization of process noise, measurement noise,
and other model parameters using cross-validation and maritime-specific
evaluation metrics.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import TimeSeriesSplit

from .coordinates import MaritimeCoordinateTransform
from .imm import MaritimeIMMFilter
from .models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from .protocols import (
    IMMConfig,
    MaritimeConstraints,
    MotionModelConfig,
)


class ParameterBounds:
    """Parameter bounds for optimization."""

    # Process noise bounds (log scale)
    POSITION_PROCESS_NOISE = (1e-3, 1.0)
    VELOCITY_PROCESS_NOISE = (1e-4, 0.1)
    ACCELERATION_PROCESS_NOISE = (1e-5, 0.01)
    TURN_RATE_PROCESS_NOISE = (1e-3, 1.0)

    # Measurement noise bounds (log scale)
    POSITION_MEASUREMENT_NOISE = (1.0, 100.0)
    VELOCITY_MEASUREMENT_NOISE = (0.01, 10.0)

    # Initial uncertainty bounds (log scale)
    INITIAL_POSITION_UNCERTAINTY = (10.0, 1000.0)
    INITIAL_VELOCITY_UNCERTAINTY = (0.1, 50.0)
    INITIAL_ACCELERATION_UNCERTAINTY = (0.01, 10.0)
    INITIAL_TURN_RATE_UNCERTAINTY = (0.01, 1.0)


class MaritimeMetrics:
    """Maritime-specific evaluation metrics for tuning."""

    @staticmethod
    def haversine_distance(
        lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Haversine distance between points.

        Args:
            lat1, lon1: First points (degrees)
            lat2, lon2: Second points (degrees)

        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km

        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    @staticmethod
    def average_displacement_error(
        predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """
        Calculate Average Displacement Error (ADE) in kilometers.

        Args:
            predictions: Predicted positions [horizon, 2] (lat, lon)
            ground_truth: True positions [horizon, 2] (lat, lon)

        Returns:
            ADE in kilometers
        """
        if len(predictions) != len(ground_truth):
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]

        distances = MaritimeMetrics.haversine_distance(
            predictions[:, 0], predictions[:, 1], ground_truth[:, 0], ground_truth[:, 1]
        )

        return np.mean(distances)

    @staticmethod
    def final_displacement_error(
        predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """
        Calculate Final Displacement Error (FDE) in kilometers.

        Args:
            predictions: Predicted positions [horizon, 2] (lat, lon)
            ground_truth: True positions [horizon, 2] (lat, lon)

        Returns:
            FDE in kilometers
        """
        if len(predictions) == 0 or len(ground_truth) == 0:
            return float("inf")

        return (
            MaritimeMetrics.haversine_distance(
                predictions[-1, 0],
                predictions[-1, 1],
                ground_truth[-1, 0],
                ground_truth[-1, 1],
            )[0]
            if len(predictions[-1:]) > 0
            else float("inf")
        )


class KalmanTuner:
    """
    Hyperparameter tuner for individual Kalman filter models.
    """

    def __init__(
        self,
        model_class: type,
        metric_fn: Callable = None,
        cv_splits: int = 3,
        optimization_method: str = "differential_evolution",
        random_state: int = 42,
    ):
        """
        Initialize tuner for a specific model class.

        Args:
            model_class: Model class to tune (CV, CT, or NCA)
            metric_fn: Evaluation metric function (default: ADE)
            cv_splits: Number of cross-validation splits
            optimization_method: Optimization algorithm
            random_state: Random seed
        """
        self.model_class = model_class
        self.metric_fn = metric_fn or MaritimeMetrics.average_displacement_error
        self.cv_splits = cv_splits
        self.optimization_method = optimization_method
        self.random_state = random_state

        self.coordinate_transform = MaritimeCoordinateTransform()
        self.best_config = None
        self.best_score = float("inf")
        self.optimization_history = []

    def _get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for the specific model."""
        bounds = [
            ParameterBounds.POSITION_PROCESS_NOISE,
            ParameterBounds.VELOCITY_PROCESS_NOISE,
            ParameterBounds.POSITION_MEASUREMENT_NOISE,
            ParameterBounds.INITIAL_POSITION_UNCERTAINTY,
            ParameterBounds.INITIAL_VELOCITY_UNCERTAINTY,
        ]

        if self.model_class == NearlyConstantAccelModel:
            bounds.extend(
                [
                    ParameterBounds.ACCELERATION_PROCESS_NOISE,
                    ParameterBounds.INITIAL_ACCELERATION_UNCERTAINTY,
                ]
            )
        elif self.model_class == CoordinatedTurnModel:
            bounds.extend(
                [
                    ParameterBounds.TURN_RATE_PROCESS_NOISE,
                    ParameterBounds.INITIAL_TURN_RATE_UNCERTAINTY,
                ]
            )

        return bounds

    def _params_to_config(self, params: np.ndarray) -> MotionModelConfig:
        """Convert optimization parameters to model configuration."""
        config = MotionModelConfig()

        config.position_process_noise = params[0]
        config.velocity_process_noise = params[1]
        config.position_measurement_noise = params[2]
        config.initial_position_uncertainty = params[3]
        config.initial_velocity_uncertainty = params[4]

        NCA_PARAM_COUNT = 7
        CT_PARAM_COUNT = 7

        if (
            self.model_class == NearlyConstantAccelModel
            and len(params) >= NCA_PARAM_COUNT
        ):
            config.acceleration_process_noise = params[5]
            config.initial_acceleration_uncertainty = params[6]
        elif self.model_class == CoordinatedTurnModel and len(params) >= CT_PARAM_COUNT:
            config.turn_rate_process_noise = params[5]
            config.initial_turn_rate_uncertainty = params[6]

        return config

    def _evaluate_parameters(
        self,
        params: np.ndarray,
        sequences: list[np.ndarray],
        prediction_horizon: int = 5,
    ) -> float:
        """
        Evaluate parameter configuration using cross-validation.

        Args:
            params: Parameter vector to evaluate
            sequences: List of trajectory sequences
            prediction_horizon: Number of steps to predict

        Returns:
            Average cross-validation score
        """
        try:
            config = self._params_to_config(params)
            constraints = MaritimeConstraints()

            cv_scores = []

            # Time series cross-validation
            for seq in sequences:
                if len(seq) < prediction_horizon + 5:  # Need minimum data
                    continue

                tscv = TimeSeriesSplit(
                    n_splits=min(self.cv_splits, len(seq) // (prediction_horizon + 2))
                )

                for train_idx, test_idx in tscv.split(seq):
                    if len(test_idx) < prediction_horizon:
                        continue

                    try:
                        # Create and fit model
                        model = self.model_class(config, constraints)
                        train_data = seq[train_idx]

                        # Fit model
                        model.fit([train_data])

                        # Make prediction
                        test_input = seq[test_idx[0] : test_idx[0] + prediction_horizon]
                        MIN_INPUT_LENGTH = 2
                        if len(test_input) < MIN_INPUT_LENGTH:
                            continue

                        result = model.predict(test_input[:-1], horizon=1)
                        prediction = result.predictions

                        # Get ground truth
                        ground_truth = seq[
                            test_idx[0] + len(test_input) - 1 : test_idx[0]
                            + len(test_input)
                        ]

                        if len(ground_truth) == 0:
                            continue

                        # Evaluate prediction
                        score = self.metric_fn(
                            prediction, ground_truth[: len(prediction)]
                        )

                        if np.isfinite(score):
                            cv_scores.append(score)

                    except Exception as e:
                        # Skip this fold if evaluation fails
                        warnings.warn(f"Evaluation failed: {e}", stacklevel=2)
                        continue

            if not cv_scores:
                return float("inf")

            mean_score = np.mean(cv_scores)
            self.optimization_history.append((params.copy(), mean_score))

            return mean_score

        except Exception as e:
            warnings.warn(f"Parameter evaluation failed: {e}", stacklevel=2)
            return float("inf")

    def tune(
        self,
        sequences: list[np.ndarray],
        prediction_horizon: int = 5,
        max_iterations: int = 50,
    ) -> MotionModelConfig:
        """
        Tune hyperparameters for the model.

        Args:
            sequences: Training sequences [seq_len, n_features]
            prediction_horizon: Number of steps to predict ahead
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized model configuration
        """
        if not sequences:
            raise ValueError("No sequences provided for tuning")

        # Set up coordinate transform
        all_positions = np.vstack([seq[:, :2] for seq in sequences])
        self.coordinate_transform.auto_set_reference(all_positions)

        # Update sequences to include timestamps if not present
        processed_sequences = []
        for traj_seq in sequences:
            TIMESTAMP_COLUMN_COUNT = 3
            if traj_seq.shape[1] < TIMESTAMP_COLUMN_COUNT:
                # Add dummy timestamps
                timestamps = np.arange(len(traj_seq), dtype=float)
                processed_traj = np.column_stack([traj_seq, timestamps])
            else:
                processed_traj = traj_seq
            processed_sequences.append(processed_traj)

        # Set up optimization
        bounds = self._get_parameter_bounds()

        def objective(params):
            return self._evaluate_parameters(
                params, processed_sequences, prediction_horizon
            )

        # Run optimization
        if self.optimization_method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iterations,
                seed=self.random_state,
                disp=True,
                atol=1e-6,
                tol=1e-6,
            )
        else:
            # Use L-BFGS-B as fallback
            x0 = np.array([np.mean(bound) for bound in bounds])
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iterations},
            )

        if result.success or hasattr(result, "x"):
            self.best_config = self._params_to_config(result.x)
            self.best_score = result.fun
        else:
            # Fallback to default configuration
            self.best_config = MotionModelConfig()
            self.best_score = float("inf")
            warnings.warn(
                "Optimization failed, using default configuration", stacklevel=2
            )

        return self.best_config

    def get_tuning_results(self) -> dict[str, Any]:
        """Get detailed tuning results."""
        return {
            "model_class": self.model_class.__name__,
            "best_config": self.best_config.__dict__ if self.best_config else None,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "n_evaluations": len(self.optimization_history),
        }


class IMMTuner:
    """
    Hyperparameter tuner for IMM framework.
    """

    def __init__(
        self,
        metric_fn: Callable = None,
        cv_splits: int = 3,
        tune_individual_models: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize IMM tuner.

        Args:
            metric_fn: Evaluation metric function
            cv_splits: Number of cross-validation splits
            tune_individual_models: Whether to tune individual model parameters
            random_state: Random seed
        """
        self.metric_fn = metric_fn or MaritimeMetrics.average_displacement_error
        self.cv_splits = cv_splits
        self.tune_individual_models = tune_individual_models
        self.random_state = random_state

        self.individual_tuners = {}
        self.best_config = None
        self.best_score = float("inf")

    def tune(
        self,
        sequences: list[np.ndarray],
        prediction_horizon: int = 5,
        max_iterations: int = 30,
    ) -> IMMConfig:
        """
        Tune IMM hyperparameters.

        Args:
            sequences: Training sequences
            prediction_horizon: Prediction horizon for evaluation
            max_iterations: Maximum optimization iterations per model

        Returns:
            Optimized IMM configuration
        """
        if not sequences:
            raise ValueError("No sequences provided for tuning")

        # Step 1: Tune individual models if requested
        if self.tune_individual_models:
            print("Tuning individual motion models...")

            model_classes = [
                ConstantVelocityModel,
                CoordinatedTurnModel,
                NearlyConstantAccelModel,
            ]

            for model_class in model_classes:
                print(f"Tuning {model_class.__name__}...")
                tuner = KalmanTuner(
                    model_class,
                    self.metric_fn,
                    self.cv_splits,
                    random_state=self.random_state,
                )

                try:
                    tuner.tune(sequences, prediction_horizon, max_iterations)
                    self.individual_tuners[model_class.__name__] = tuner
                    print(
                        f"{model_class.__name__} tuned with score: {tuner.best_score:.4f}"
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to tune {model_class.__name__}: {e}", stacklevel=2
                    )
                    self.individual_tuners[model_class.__name__] = None

        # Step 2: Create IMM configuration
        if self.tune_individual_models and self.individual_tuners:
            # Use best individual model configurations
            best_motion_config = None
            best_score = float("inf")

            for _name, tuner in self.individual_tuners.items():
                if tuner and tuner.best_score < best_score:
                    best_motion_config = tuner.best_config
                    best_score = tuner.best_score

            if best_motion_config is None:
                best_motion_config = MotionModelConfig()
        else:
            best_motion_config = MotionModelConfig()

        # Step 3: Optimize transition probabilities (simplified)
        # For now, use default transition matrix
        # Could be extended to optimize transition probabilities

        self.best_config = IMMConfig(
            motion_config=best_motion_config, constraints=MaritimeConstraints()
        )

        # Step 4: Evaluate final IMM configuration
        try:
            imm_filter = MaritimeIMMFilter(self.best_config)

            # Simple evaluation on a subset of sequences
            evaluation_scores = []

            for seq in sequences[: min(5, len(sequences))]:  # Limit for efficiency
                if len(seq) < prediction_horizon + 5:
                    continue

                try:
                    # Split sequence
                    split_point = len(seq) - prediction_horizon
                    train_seq = seq[:split_point]
                    test_seq = seq[split_point:]

                    # Fit and predict
                    imm_filter.fit([train_seq])
                    result = imm_filter.predict(train_seq[-5:], prediction_horizon)

                    # Evaluate
                    score = self.metric_fn(
                        result.predictions, test_seq[: len(result.predictions), :2]
                    )
                    if np.isfinite(score):
                        evaluation_scores.append(score)

                except Exception:
                    continue

            if evaluation_scores:
                self.best_score = np.mean(evaluation_scores)

        except Exception as e:
            warnings.warn(f"IMM evaluation failed: {e}", stacklevel=2)

        return self.best_config

    def get_tuning_results(self) -> dict[str, Any]:
        """Get comprehensive tuning results."""
        results = {
            "imm_config": self.best_config.__dict__ if self.best_config else None,
            "imm_score": self.best_score,
            "individual_model_results": {},
        }

        for name, tuner in self.individual_tuners.items():
            if tuner:
                results["individual_model_results"][name] = tuner.get_tuning_results()

        return results


def tune_maritime_baseline(
    sequences: list[np.ndarray],
    model_type: str = "imm",
    prediction_horizon: int = 5,
    max_iterations: int = 50,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Convenience function to tune maritime baseline models.

    Args:
        sequences: Training sequences [seq_len, n_features] with lat, lon, [timestamp]
        model_type: Type of model to tune ('cv', 'ct', 'nca', 'imm')
        prediction_horizon: Number of steps to predict ahead
        max_iterations: Maximum optimization iterations
        random_state: Random seed

    Returns:
        Dictionary with tuning results and optimized configuration
    """
    if model_type.lower() == "imm":
        tuner = IMMTuner(random_state=random_state)
        config = tuner.tune(sequences, prediction_horizon, max_iterations)
        results = tuner.get_tuning_results()
        results["optimized_config"] = config

    else:
        model_class_map = {
            "cv": ConstantVelocityModel,
            "ct": CoordinatedTurnModel,
            "nca": NearlyConstantAccelModel,
        }

        if model_type.lower() not in model_class_map:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = model_class_map[model_type.lower()]
        tuner = KalmanTuner(model_class, random_state=random_state)
        config = tuner.tune(sequences, prediction_horizon, max_iterations)
        results = tuner.get_tuning_results()
        results["optimized_config"] = config

    return results
