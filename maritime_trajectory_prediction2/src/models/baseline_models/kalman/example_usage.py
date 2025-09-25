"""
Example usage of maritime Kalman filter baseline models.

This script demonstrates how to use the different Kalman filter baseline models
for maritime trajectory prediction, including IMM, CV, CT, and NCA models.
"""

import matplotlib.pyplot as plt
import numpy as np

from .imm import MaritimeIMMFilter
from .models import (
    ConstantVelocityModel,
    CoordinatedTurnModel,
    NearlyConstantAccelModel,
)
from .protocols import IMMConfig, MaritimeConstraints, MotionModelConfig
from .tuning import tune_maritime_baseline

# Constants for linting compliance
SECONDS_PER_HOUR = 3600
ACCELERATION_FACTOR = 0.5
TRAJECTORY_TYPE_ACCELERATING = 2


def create_synthetic_trajectories(n_trajectories: int = 5) -> list[np.ndarray]:
    """
    Create synthetic maritime trajectories for demonstration.

    Returns:
        List of trajectory arrays [seq_len, 3] with [lat, lon, timestamp]
    """
    np.random.seed(42)
    trajectories = []

    for i in range(n_trajectories):
        # Base position around Faroe Islands
        base_lat = 62.0 + np.random.normal(0, 0.5)
        base_lon = -7.0 + np.random.normal(0, 1.0)

        # Trajectory length
        seq_len = np.random.randint(20, 50)
        timestamps = np.arange(seq_len, dtype=float) * 300.0  # 5-minute intervals

        if i == 0:
            # Straight line trajectory (favors CV model)
            direction = np.random.uniform(0, 2 * np.pi)
            speed_deg_per_hour = 0.01  # ~1 km/hour in degrees

            lats = base_lat + np.cumsum(
                np.cos(direction) * speed_deg_per_hour * (timestamps / SECONDS_PER_HOUR)
            )
            lons = base_lon + np.cumsum(
                np.sin(direction) * speed_deg_per_hour * (timestamps / SECONDS_PER_HOUR)
            )

        elif i == 1:
            # Circular trajectory (favors CT model)
            radius = 0.01  # degrees
            angular_velocity = 2 * np.pi / (seq_len * 300)  # One circle over trajectory

            angles = timestamps * angular_velocity
            lats = base_lat + radius * np.sin(angles)
            lons = base_lon + radius * np.cos(angles)

        elif i == TRAJECTORY_TYPE_ACCELERATING:
            # Accelerating trajectory (favors NCA model)
            initial_speed = 0.005
            acceleration = 0.0001

            # Quadratic motion
            distances = (
                initial_speed * (timestamps / SECONDS_PER_HOUR)
                + ACCELERATION_FACTOR * acceleration * (timestamps / SECONDS_PER_HOUR) ** 2
            )
            direction = np.random.uniform(0, 2 * np.pi)

            lats = base_lat + distances * np.cos(direction)
            lons = base_lon + distances * np.sin(direction)

        else:
            # Mixed behavior trajectory (favors IMM)
            lats = [base_lat]
            lons = [base_lon]

            for t in range(1, seq_len):
                # Change behavior every ~10 steps
                phase = (t // 10) % 3

                if phase == 0:  # Straight line
                    lat_step = 0.001 * np.cos(t * 0.1)
                    lon_step = 0.001 * np.sin(t * 0.1)
                elif phase == 1:  # Turn
                    lat_step = 0.001 * np.cos(t * 0.5)
                    lon_step = 0.001 * np.sin(t * 0.5)
                else:  # Accelerate
                    lat_step = 0.001 * (1 + 0.1 * t)
                    lon_step = 0.001 * (1 + 0.1 * t)

                lats.append(lats[-1] + lat_step)
                lons.append(lons[-1] + lon_step)

            lats = np.array(lats)
            lons = np.array(lons)

        # Combine into trajectory
        trajectory = np.column_stack([lats, lons, timestamps])
        trajectories.append(trajectory)

    return trajectories


def demonstrate_individual_models():
    """Demonstrate individual motion models (CV, CT, NCA)."""
    print("=" * 60)
    print("INDIVIDUAL MODEL DEMONSTRATION")
    print("=" * 60)

    # Create test trajectories
    trajectories = create_synthetic_trajectories(3)

    models = {
        "Constant Velocity": ConstantVelocityModel(),
        "Coordinated Turn": CoordinatedTurnModel(),
        "Nearly Constant Accel": NearlyConstantAccelModel(),
    }

    for model_name, model in models.items():
        print(f"\n{model_name} Model:")
        print("-" * 30)

        # Fit model
        model.fit(trajectories)

        # Make predictions on test trajectory
        test_trajectory = trajectories[0]
        input_seq = test_trajectory[:15]  # Use first 15 points
        horizon = 5

        result = model.predict(input_seq, horizon, return_uncertainty=True)

        print(f"  Input sequence length: {len(input_seq)}")
        print(f"  Predictions shape: {result.predictions.shape}")
        print(f"  Has uncertainty: {result.uncertainty is not None}")
        print(f"  Model info: {result.model_info['model_type']}")

        # Show sample prediction
        print(f"  Sample prediction (lat, lon): {result.predictions[0]}")


def demonstrate_imm_framework():
    """Demonstrate IMM framework with model switching."""
    print("\n\n" + "=" * 60)
    print("IMM FRAMEWORK DEMONSTRATION")
    print("=" * 60)

    # Create IMM configuration
    config = IMMConfig(
        motion_config=MotionModelConfig(), constraints=MaritimeConstraints()
    )

    # Create IMM filter
    imm_filter = MaritimeIMMFilter(config)

    # Generate mixed-behavior trajectory
    trajectories = create_synthetic_trajectories(1)
    test_trajectory = trajectories[0]

    print(f"Fitting IMM on trajectory with {len(test_trajectory)} points...")

    # Fit IMM
    imm_filter.fit([test_trajectory])

    # Make prediction
    input_seq = test_trajectory[:20]
    result = imm_filter.predict(input_seq, horizon=8, return_uncertainty=True)

    print("IMM Prediction Results:")
    print(f"  Predictions shape: {result.predictions.shape}")
    print(f"  Final model probabilities: {result.model_info['final_probabilities']}")
    print(f"  Dominant model: {result.model_info['dominant_model']}")
    print(f"  Model switches: {result.model_info['n_switches']}")
    print(f"  Confidence scores: {result.confidence}")


def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n\n" + "=" * 60)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("=" * 60)

    # Create training trajectories
    trajectories = create_synthetic_trajectories(8)

    print(f"Tuning on {len(trajectories)} trajectories...")

    # Tune IMM model
    print("\nTuning IMM model...")
    tuning_results = tune_maritime_baseline(
        trajectories,
        model_type="imm",
        prediction_horizon=3,
        max_iterations=10,  # Reduced for demo
    )

    print("Tuning Results:")
    print(f"  Best score: {tuning_results['imm_score']:.4f}")
    print(f"  Optimized config available: {'optimized_config' in tuning_results}")

    if "individual_model_results" in tuning_results:
        for model_name, results in tuning_results["individual_model_results"].items():
            if results:
                print(f"  {model_name}: {results['best_score']:.4f}")


def demonstrate_model_comparison():
    """Compare different baseline models on the same data."""
    print("\n\n" + "=" * 60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)

    # Create test trajectories
    trajectories = create_synthetic_trajectories(3)
    test_trajectory = trajectories[0]

    # Create all models
    models = {
        "CV": ConstantVelocityModel(),
        "CT": CoordinatedTurnModel(),
        "NCA": NearlyConstantAccelModel(),
        "IMM": MaritimeIMMFilter(),
    }

    # Fit all models
    for _name, model in models.items():
        model.fit(trajectories)

    # Compare predictions
    input_seq = test_trajectory[:15]
    horizon = 5

    print(f"Comparing models on {len(input_seq)}-point input sequence:")
    print("Model | Final Position (lat, lon) | Distance from Truth")
    print("-" * 60)

    ground_truth = test_trajectory[len(input_seq) : len(input_seq) + horizon]
    final_truth = ground_truth[-1, :2] if len(ground_truth) > 0 else input_seq[-1, :2]

    for name, model in models.items():
        try:
            result = model.predict(input_seq, horizon)
            final_pred = result.predictions[-1]

            # Calculate distance (approximate)
            distance = (
                np.sqrt(
                    (final_pred[0] - final_truth[0]) ** 2
                    + (final_pred[1] - final_truth[1]) ** 2
                )
                * 111.32
            )  # Convert to km

            print(
                f"{name:5} | ({final_pred[0]:.4f}, {final_pred[1]:.4f}) | {distance:.2f} km"
            )

        except Exception as e:
            print(f"{name:5} | Error: {e}")


def visualize_predictions(save_plot: bool = False):
    """Create visualization of model predictions."""
    print("\n\n" + "=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)

    # Create test trajectory
    trajectories = create_synthetic_trajectories(1)
    test_trajectory = trajectories[0]

    # Create and fit IMM model
    imm_model = MaritimeIMMFilter()
    imm_model.fit([test_trajectory])

    # Split trajectory
    split_point = len(test_trajectory) - 8
    input_seq = test_trajectory[:split_point]
    true_future = test_trajectory[split_point:]

    # Make prediction
    result = imm_model.predict(
        input_seq, horizon=len(true_future), return_uncertainty=True
    )

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot historical trajectory
    plt.plot(input_seq[:, 1], input_seq[:, 0], "b-o", label="Historical", markersize=4)

    # Plot true future
    plt.plot(
        true_future[:, 1], true_future[:, 0], "g-o", label="True Future", markersize=4
    )

    # Plot predictions
    plt.plot(
        result.predictions[:, 1],
        result.predictions[:, 0],
        "r-s",
        label="IMM Prediction",
        markersize=6,
    )

    # Plot uncertainty ellipses if available
    if result.uncertainty is not None:
        for _i, (pred, cov) in enumerate(
            zip(result.predictions, result.uncertainty, strict=False)
        ):
            # Simple uncertainty visualization
            std_x = np.sqrt(cov[1, 1])  # longitude uncertainty
            _std_y = np.sqrt(cov[0, 0])  # latitude uncertainty

            ellipse = plt.Circle(
                (pred[1], pred[0]), radius=2 * std_x, fill=False, color="red", alpha=0.3
            )
            plt.gca().add_patch(ellipse)

    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title("Maritime Trajectory Prediction with IMM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    if save_plot:
        plt.savefig("kalman_prediction_demo.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'kalman_prediction_demo.png'")

    plt.show()

    # Show model probability evolution
    if result.model_info and "model_probabilities" in result.model_info:
        plt.figure(figsize=(10, 6))

        probabilities = np.array(result.model_info["model_probabilities"])
        model_names = result.model_info["model_names"]

        for i, name in enumerate(model_names):
            plt.plot(probabilities[:, i], label=name, linewidth=2)

        plt.xlabel("Prediction Step")
        plt.ylabel("Model Probability")
        plt.title("IMM Model Probability Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        if save_plot:
            plt.savefig("imm_probabilities_demo.png", dpi=300, bbox_inches="tight")
            print("Model probabilities plot saved as 'imm_probabilities_demo.png'")

        plt.show()


def main():
    """Run all demonstration examples."""
    print("MARITIME KALMAN FILTER BASELINE DEMONSTRATION")
    print("=" * 60)

    try:
        demonstrate_individual_models()
        demonstrate_imm_framework()
        demonstrate_model_comparison()
        demonstrate_hyperparameter_tuning()

        # Visualization
        visualize_predictions(save_plot=False)

        print("\n\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nKey takeaways:")
        print(
            "- Individual models (CV, CT, NCA) are specialized for different motion types"
        )
        print("- IMM automatically switches between models based on vessel behavior")
        print("- Hyperparameter tuning improves prediction accuracy")
        print("- All models provide uncertainty estimates for robust prediction")
        print(
            "- Integration with PyTorch Lightning enables comparison with neural models"
        )

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
