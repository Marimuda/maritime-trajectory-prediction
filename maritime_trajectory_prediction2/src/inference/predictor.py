"""
Unified prediction/inference module.

Consolidates predict_trajectory.py and inference_transformer_models.py
into a single, configuration-driven prediction system following the CLAUDE blueprint.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from ..data.maritime_message_processor import AISProcessor
from ..models.model_factory import load_model
from ..utils.maritime_utils import MaritimeUtils

# Constants for tensor dimensions and data structure
TENSOR_2D = 2  # 2D tensor dimension (for input data without batch dimension)
TENSOR_3D = 3  # 3D tensor dimension (batch, sequence, features)
COORDINATE_PAIR_SIZE = 2  # Number of coordinate features (lat, lon)
POWER_OF_TWO = 2  # Exponent for squared operations (MSE calculation)
HAVERSINE_MULTIPLIER = 2  # Multiplier in Haversine distance formula

logger = logging.getLogger(__name__)


class UnifiedPredictor:
    """
    Unified prediction engine consolidating multiple inference approaches.

    Supports both simple trajectory prediction and advanced SOTA model inference
    with real-time processing capabilities.
    """

    def __init__(self, model_path: str, config: DictConfig | None = None):
        """
        Initialize the prediction engine.

        Args:
            model_path: Path to trained model checkpoint
            config: Optional model configuration
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Setup data processor
        self.processor = AISProcessor() if hasattr(self, "_has_ais_processor") else None
        self.maritime_utils = (
            MaritimeUtils() if hasattr(self, "_has_maritime_utils") else None
        )

        logger.info(f"Predictor initialized with model from {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model type: {type(self.model).__name__}")

    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        try:
            if self.config:
                # Load with specific config
                return load_model(self.model_path, self.config)
            else:
                # Try to load checkpoint and extract config
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )  # nosec B614

                if "config" in checkpoint and checkpoint["config"]:
                    # Load with embedded config
                    return load_model(self.model_path, checkpoint["config"])
                else:
                    # Load without config (PyTorch Lightning format)
                    return load_model(self.model_path)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_data(
        self, data: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocess input data for inference.

        Args:
            data: Input data (DataFrame, numpy array, or tensor)

        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(data, pd.DataFrame):
            # Process AIS DataFrame
            if self.processor:
                processed_data = self.processor.process_dataframe(data)

                # Extract features based on model requirements
                feature_cols = [
                    "latitude",
                    "longitude",
                    "sog",
                    "cog",
                    "heading",
                    "nav_status",
                    "vessel_type",
                    "length",
                    "width",
                    "draught",
                    "rot",
                    "lat_diff",
                    "lon_diff",
                ]

                available_cols = [
                    col for col in feature_cols if col in processed_data.columns
                ]
                tensor_data = torch.tensor(
                    processed_data[available_cols].values, dtype=torch.float32
                )
            else:
                # Basic feature extraction
                feature_cols = ["lat", "lon", "sog", "cog", "heading", "nav_status"]
                available_cols = [col for col in feature_cols if col in data.columns]
                tensor_data = torch.tensor(
                    data[available_cols].values, dtype=torch.float32
                )

            # Add batch dimension if needed
            if tensor_data.dim() == TENSOR_2D:
                tensor_data = tensor_data.unsqueeze(0)

        elif isinstance(data, np.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32)
            if tensor_data.dim() == TENSOR_2D:
                tensor_data = tensor_data.unsqueeze(0)

        elif isinstance(data, torch.Tensor):
            tensor_data = data.float()
            if tensor_data.dim() == TENSOR_2D:
                tensor_data = tensor_data.unsqueeze(0)

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return tensor_data.to(self.device)

    def predict_trajectory(
        self,
        input_sequence: pd.DataFrame | np.ndarray | torch.Tensor,
        num_steps: int = 12,
        num_samples: int = 10,
    ) -> dict[str, Any]:
        """
        Predict vessel trajectory using the loaded model.

        Args:
            input_sequence: Input sequence data
            num_steps: Number of steps to predict
            num_samples: Number of trajectory samples to generate

        Returns:
            Dictionary containing predictions and metadata
        """
        # Preprocess input
        input_tensor = self.preprocess_data(input_sequence)

        predictions = []
        inference_times = []

        with torch.no_grad():
            for _ in range(num_samples):
                start_time = time.time()

                # Check if model has specialized prediction method
                if hasattr(self.model, "predict_trajectory"):
                    pred_traj = self.model.predict_trajectory(
                        input_tensor, steps=num_steps
                    )
                elif hasattr(self.model, "predict_best_trajectory"):
                    pred_traj = self.model.predict_best_trajectory(input_tensor)
                else:
                    # Standard autoregressive prediction
                    pred_traj = self._autoregressive_prediction(input_tensor, num_steps)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Convert to numpy
                if torch.is_tensor(pred_traj):
                    pred_traj = pred_traj.cpu().numpy()

                predictions.append(pred_traj)

        # Find best prediction (could be based on confidence or other criteria)
        best_idx = 0  # Simple: use first prediction as best
        if len(predictions) > 1 and hasattr(self.model, "compute_confidence"):
            confidences = [self.model.compute_confidence(pred) for pred in predictions]
            best_idx = np.argmax(confidences)

        results = {
            "predictions": predictions,
            "best_prediction": predictions[best_idx],
            "best_trajectory_idx": best_idx,
            "num_samples": num_samples,
            "inference_times": inference_times,
            "avg_inference_time": np.mean(inference_times),
            "input_shape": input_tensor.shape,
            "prediction_horizon": num_steps,
        }

        return results

    def _autoregressive_prediction(
        self, input_tensor: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        Perform autoregressive prediction for models without specialized methods.

        Args:
            input_tensor: Input sequence tensor
            num_steps: Number of steps to predict

        Returns:
            Predicted trajectory tensor
        """
        current_input = input_tensor.clone()
        predictions = []

        for _ in range(num_steps):
            # Predict next step
            output = self.model(current_input)

            # Handle different output formats
            if isinstance(output, tuple):
                next_step = output[0]  # Take first element if tuple
            elif isinstance(output, dict):
                next_step = output.get(
                    "trajectories", output.get("prediction", list(output.values())[0])
                )
            else:
                next_step = output

            # Extract prediction for next timestep
            if len(next_step.shape) == TENSOR_3D:  # [batch, seq, features]
                pred = next_step[:, -1, :]  # Last time step
            else:
                pred = next_step

            predictions.append(pred)

            # Update input for next step (sliding window)
            if len(current_input.shape) == TENSOR_3D:  # [batch, seq_len, features]
                new_input = torch.cat(
                    [
                        current_input[:, 1:, :],  # Remove oldest step
                        pred.unsqueeze(1),  # Add prediction as new step
                    ],
                    dim=1,
                )
                current_input = new_input

        # Combine predictions into trajectory
        return torch.cat([p.unsqueeze(1) for p in predictions], dim=1)

    def predict_anomalies(
        self, data: pd.DataFrame | np.ndarray | torch.Tensor, threshold: float = 0.5
    ) -> dict[str, Any]:
        """
        Detect anomalies in vessel trajectories.

        Args:
            data: Input trajectory data
            threshold: Anomaly detection threshold

        Returns:
            Dictionary containing anomaly detection results
        """
        # Check if model supports anomaly detection
        if not hasattr(self.model, "detect_anomalies"):
            logger.warning(
                "Model does not have specialized anomaly detection method. Using standard prediction."
            )

            # Use standard prediction and compute anomaly scores
            input_tensor = self.preprocess_data(data)

            with torch.no_grad():
                outputs = self.model(input_tensor)

                # Convert outputs to anomaly scores (simple heuristic)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Compute reconstruction error or use direct output
                if hasattr(self.model, "compute_reconstruction_error"):
                    anomaly_scores = self.model.compute_reconstruction_error(
                        input_tensor, outputs
                    )
                else:
                    # Simple anomaly scoring based on prediction confidence
                    anomaly_scores = torch.sigmoid(outputs.mean(dim=-1))

                binary_anomalies = (anomaly_scores > threshold).float()

                results = {
                    "anomaly_scores": anomaly_scores.cpu().numpy(),
                    "binary_anomalies": binary_anomalies.cpu().numpy(),
                    "threshold": threshold,
                    "detection_rate": float(binary_anomalies.mean()),
                    "confidence": anomaly_scores.cpu().numpy(),
                }
        else:
            # Use specialized anomaly detection
            input_tensor = self.preprocess_data(data)

            with torch.no_grad():
                results = self.model.detect_anomalies(input_tensor, threshold=threshold)

                # Convert tensors to numpy
                for key, value in results.items():
                    if torch.is_tensor(value):
                        results[key] = value.cpu().numpy()

        return results

    def batch_inference(
        self,
        data_list: list[pd.DataFrame | np.ndarray | torch.Tensor],
        task: str = "trajectory_prediction",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Perform batch inference on multiple data samples.

        Args:
            data_list: List of input data samples
            task: Task type ('trajectory_prediction' or 'anomaly_detection')
            **kwargs: Additional arguments for specific tasks

        Returns:
            List of prediction results
        """
        results = []

        logger.info(
            f"Starting batch inference on {len(data_list)} samples for task: {task}"
        )

        for i, data in enumerate(data_list):
            try:
                if task == "anomaly_detection":
                    result = self.predict_anomalies(data, **kwargs)
                else:
                    result = self.predict_trajectory(data, **kwargs)

                result["sample_id"] = i
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(data_list)} samples")

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                results.append({"sample_id": i, "error": str(e)})

        logger.info("Batch inference completed")
        return results

    def real_time_inference(
        self, data_stream, task: str = "trajectory_prediction", callback=None, **kwargs
    ):
        """
        Perform real-time inference on streaming data.

        Args:
            data_stream: Iterator or generator yielding data samples
            task: Task type
            callback: Optional callback function for results
            **kwargs: Additional arguments
        """
        logger.info(f"Starting real-time inference for task: {task}")

        for i, data in enumerate(data_stream):
            start_time = time.time()

            try:
                if task == "anomaly_detection":
                    result = self.predict_anomalies(data, **kwargs)
                else:
                    result = self.predict_trajectory(data, **kwargs)

                inference_time = time.time() - start_time
                result["inference_time"] = inference_time
                result["sample_id"] = i
                result["timestamp"] = time.time()

                if callback:
                    callback(result)
                else:
                    logger.info(f"Sample {i}: Inference time: {inference_time:.4f}s")

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")

    def evaluate_predictions(
        self, predicted_trajectories: list[np.ndarray], ground_truth: np.ndarray
    ) -> dict[str, Any]:
        """
        Evaluate trajectory predictions against ground truth.

        Args:
            predicted_trajectories: List of predicted trajectories
            ground_truth: Ground truth trajectory

        Returns:
            Dictionary of evaluation metrics
        """
        if not predicted_trajectories:
            return {}

        # Extract positions (lat, lon) from ground truth
        if len(ground_truth.shape) == TENSOR_3D:
            gt_positions = ground_truth[0, :, :COORDINATE_PAIR_SIZE]  # Take first batch
        else:
            gt_positions = ground_truth[:, :COORDINATE_PAIR_SIZE]

        errors = []

        for pred_traj in predicted_trajectories:
            # Extract positions from prediction
            if len(pred_traj.shape) == TENSOR_3D:
                pred_positions = pred_traj[
                    0, :, :COORDINATE_PAIR_SIZE
                ]  # Take first batch
            else:
                pred_positions = pred_traj[:, :COORDINATE_PAIR_SIZE]

            # Limit to minimum length
            min_len = min(len(gt_positions), len(pred_positions))
            gt_pos = gt_positions[:min_len]
            pred_pos = pred_positions[:min_len]

            # Calculate metrics
            mse = np.mean((pred_pos - gt_pos) ** POWER_OF_TWO)
            rmse = np.sqrt(mse)

            # Calculate distances at each step
            distances = []
            for i in range(min_len):
                dist = self._haversine_distance(
                    gt_pos[i, 0], gt_pos[i, 1], pred_pos[i, 0], pred_pos[i, 1]
                )
                distances.append(dist)

            errors.append(
                {
                    "mse": mse,
                    "rmse": rmse,
                    "distances": distances,
                    "mean_distance": np.mean(distances),
                    "max_distance": np.max(distances),
                }
            )

        # Find best trajectory
        best_idx = np.argmin([e["mean_distance"] for e in errors])
        best_error = errors[best_idx]

        # Compute overall metrics
        metrics = {
            "best_mse": best_error["mse"],
            "best_rmse": best_error["rmse"],
            "best_mean_distance": best_error["mean_distance"],
            "best_max_distance": best_error["max_distance"],
            "best_trajectory_idx": best_idx,
            "mean_rmse": np.mean([e["rmse"] for e in errors]),
            "std_rmse": np.std([e["rmse"] for e in errors]),
            "min_rmse": np.min([e["rmse"] for e in errors]),
            "num_predictions": len(predicted_trajectories),
        }

        return metrics

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate haversine distance between two points."""
        from math import asin, cos, radians, sin, sqrt

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            sin(dlat / HAVERSINE_MULTIPLIER) ** POWER_OF_TWO
            + cos(lat1) * cos(lat2) * sin(dlon / HAVERSINE_MULTIPLIER) ** POWER_OF_TWO
        )
        c = HAVERSINE_MULTIPLIER * asin(sqrt(a))

        # Earth radius in kilometers
        return 6371.0 * c

    def export_predictions(
        self, results: dict[str, Any], output_path: str, format: str = "csv"
    ):
        """
        Export prediction results to file.

        Args:
            results: Prediction results
            output_path: Output file path
            format: Output format ('csv', 'json', 'npz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            # Convert results to DataFrame
            if "predictions" in results:
                # Trajectory predictions
                best_traj = results["best_prediction"]
                if len(best_traj.shape) == TENSOR_3D:
                    best_traj = best_traj[0]  # Take first batch

                df = pd.DataFrame(
                    best_traj,
                    columns=[f"feature_{i}" for i in range(best_traj.shape[1])],
                )
                df.to_csv(output_path, index=False)

            elif "anomaly_scores" in results:
                # Anomaly detection results
                df = pd.DataFrame(
                    {
                        "anomaly_score": results["anomaly_scores"].flatten(),
                        "binary_anomaly": results["binary_anomalies"].flatten(),
                        "confidence": results.get(
                            "confidence", results["anomaly_scores"]
                        ).flatten(),
                    }
                )
                df.to_csv(output_path, index=False)

        elif format == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value

            with open(output_path, "w") as f:
                json.dump(json_results, f, indent=2)

        elif format == "npz":
            # Save as compressed numpy arrays
            np.savez_compressed(
                output_path,
                **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
            )

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported results to {output_path}")

    def visualize_predictions(
        self,
        input_data: np.ndarray,
        predictions: dict[str, Any],
        ground_truth: np.ndarray | None = None,
        output_path: str | None = None,
    ):
        """
        Visualize trajectory predictions.

        Args:
            input_data: Input trajectory data
            predictions: Prediction results
            ground_truth: Optional ground truth data
            output_path: Optional path to save visualization
        """
        plt.figure(figsize=(12, 8))

        # Extract position data
        if len(input_data.shape) == TENSOR_3D:
            input_positions = input_data[0, :, :COORDINATE_PAIR_SIZE]
        else:
            input_positions = input_data[:, :COORDINATE_PAIR_SIZE]

        # Plot input trajectory
        plt.plot(
            input_positions[:, 1],
            input_positions[:, 0],
            "b-",
            linewidth=2,
            label="Input Trajectory",
            marker="o",
        )

        # Plot predictions
        if "best_prediction" in predictions:
            best_pred = predictions["best_prediction"]
            if len(best_pred.shape) == TENSOR_3D:
                best_pred = best_pred[0]

            pred_positions = best_pred[:, :COORDINATE_PAIR_SIZE]
            plt.plot(
                pred_positions[:, 1],
                pred_positions[:, 0],
                "r-",
                linewidth=2,
                label="Best Prediction",
                marker="s",
            )

            # Plot other predictions with transparency
            if "predictions" in predictions and len(predictions["predictions"]) > 1:
                for i, pred in enumerate(predictions["predictions"]):
                    if i != predictions.get("best_trajectory_idx", 0):
                        pred_data = pred[0] if len(pred.shape) == TENSOR_3D else pred
                        pred_pos = pred_data[:, :COORDINATE_PAIR_SIZE]
                        plt.plot(pred_pos[:, 1], pred_pos[:, 0], "r-", alpha=0.3)

        # Plot ground truth if available
        if ground_truth is not None:
            if len(ground_truth.shape) == TENSOR_3D:
                gt_positions = ground_truth[0, :, :COORDINATE_PAIR_SIZE]
            else:
                gt_positions = ground_truth[:, :COORDINATE_PAIR_SIZE]

            plt.plot(
                gt_positions[:, 1],
                gt_positions[:, 0],
                "g-",
                linewidth=2,
                label="Ground Truth",
                marker="^",
            )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Vessel Trajectory Prediction")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        plt.show()


def run_prediction(cfg: DictConfig) -> dict[str, Any]:
    """
    Main prediction function called by Hydra dispatch.

    Args:
        cfg: Hydra configuration object

    Returns:
        Prediction results dictionary
    """
    logger.info("Starting prediction pipeline")

    # Extract configuration
    predict_cfg = cfg.predict
    model_cfg = cfg.model

    # Validate inputs
    if not model_cfg.checkpoint_path:
        raise ValueError("Model checkpoint path not specified")

    if not predict_cfg.input_file:
        raise ValueError("Input file not specified")

    # Initialize predictor
    logger.info(f"Loading model from: {model_cfg.checkpoint_path}")
    predictor = UnifiedPredictor(model_cfg.checkpoint_path, model_cfg)

    # Load input data
    input_file = Path(predict_cfg.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading input data from: {input_file}")

    if input_file.suffix == ".csv":
        data = pd.read_csv(input_file)
    elif input_file.suffix == ".parquet":
        data = pd.read_parquet(input_file)
    elif input_file.suffix == ".npz":
        data = np.load(input_file)["data"]
    else:
        raise ValueError(f"Unsupported input format: {input_file.suffix}")

    # Determine task based on model configuration
    task = (
        "anomaly_detection"
        if model_cfg.task == "anomaly_detection"
        else "trajectory_prediction"
    )

    # Perform prediction
    logger.info(f"Running {task} with {predict_cfg.num_samples} samples")

    if task == "anomaly_detection":
        results = predictor.predict_anomalies(
            data, threshold=predict_cfg.get("threshold", 0.5)
        )
    else:
        results = predictor.predict_trajectory(
            data,
            num_steps=model_cfg.prediction_horizon,
            num_samples=predict_cfg.num_samples,
        )

    # Export results if output file specified
    if predict_cfg.output_file:
        logger.info(f"Exporting results to: {predict_cfg.output_file}")
        predictor.export_predictions(
            results, predict_cfg.output_file, predict_cfg.format
        )

    # Visualize if requested
    if predict_cfg.visualize and task == "trajectory_prediction":
        logger.info("Creating visualization...")
        viz_path = None
        if predict_cfg.output_file:
            viz_path = str(Path(predict_cfg.output_file).with_suffix(".png"))

        predictor.visualize_predictions(data, results, output_path=viz_path)

    # Log summary
    logger.info("Prediction completed successfully!")
    logger.info(f"Task: {task}")
    logger.info(f"Input shape: {results.get('input_shape', 'N/A')}")

    if task == "trajectory_prediction":
        logger.info(f"Generated {results['num_samples']} trajectory samples")
        logger.info(f"Average inference time: {results['avg_inference_time']:.4f}s")
    else:
        logger.info(f"Detection rate: {results['detection_rate']:.2%}")

    return results
