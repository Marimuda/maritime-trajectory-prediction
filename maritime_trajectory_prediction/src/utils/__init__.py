"""Utility modules for maritime trajectory prediction"""

from .metrics import (
    haversine_distance,
    HaversineDistanceMetric,
    rmse_error,
    mae_error,
    rmse_haversine,
)
from .maritime_utils import (
    haversine_distance,
    bearing,
    knots_to_kmh,
    kmh_to_knots,
    calculate_trajectory_features,
    interpolate_trajectory,
    create_trajectory_segments,
    discretize_value,
    create_four_hot_encoding,
)
from .visualization import (
    plot_trajectory,
    plot_multiple_trajectories,
    create_interactive_map,
    plot_density_map,
    plot_prediction_vs_ground_truth,
    plot_maritime_graph,
    visualize_attention_weights,
    plot_prediction_uncertainty,
    create_animation,
)
from .ais_parser import (
    load_processed_ais_data,
    get_vessel_trajectories,
    prepare_trajectory_for_prediction,
)

__all__ = [
    # metrics
    "haversine_distance",
    "HaversineDistanceMetric",
    "rmse_error",
    "mae_error",
    "rmse_haversine",
    # maritime utils
    "bearing",
    "knots_to_kmh",
    "kmh_to_knots",
    "calculate_trajectory_features",
    "interpolate_trajectory",
    "create_trajectory_segments",
    "discretize_value",
    "create_four_hot_encoding",
    # visualization
    "plot_trajectory",
    "plot_multiple_trajectories",
    "create_interactive_map",
    "plot_density_map",
    "plot_prediction_vs_ground_truth",
    "plot_maritime_graph",
    "visualize_attention_weights",
    "plot_prediction_uncertainty",
    "create_animation",
    # ais parsing
    "load_processed_ais_data",
    "get_vessel_trajectories",
    "prepare_trajectory_for_prediction",
]
