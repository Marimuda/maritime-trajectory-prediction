#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained baseline models.

Generates metrics and visualizations suitable for scientific paper presentation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.maritime.cpa_tcpa import CPATCPACalculator

# from src.metrics.interaction_metrics import InteractionMetrics  # Not available yet
from src.metrics.operational.ops_metrics import OperationalMetrics, WarningEvent
from src.metrics.trajectory_metrics import TrajectoryMetrics
from train_all_baselines import MockMaritimeDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BaselineEvaluator:
    """Comprehensive evaluator for trained baseline models."""

    def __init__(self, experiment_dir: Path, data_path: Path | None = None):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Load experiment config
        config_path = self.results_dir / "experiment_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Initialize metrics
        self.traj_metrics = TrajectoryMetrics()
        # self.interaction_metrics = InteractionMetrics()  # Not available yet
        self.operational_metrics = OperationalMetrics()
        self.cpa_calculator = CPATCPACalculator()

        # Load test data
        self.test_loader = self._load_test_data(data_path)

        # Results storage
        self.evaluation_results = {}

    def _load_test_data(self, data_path: Path | None = None):
        """Load test dataset."""
        if data_path and data_path.exists():
            logger.info(f"Loading test data from {data_path}")
            # Load real test data
            # (Implementation would load actual test split)
            pass
        else:
            logger.info("Using synthetic test data")
            test_dataset = MockMaritimeDataset(n_samples=500, seq_len=50)
            return DataLoader(test_dataset, batch_size=32, shuffle=False)

    def evaluate_all_models(self) -> dict[str, dict[str, float]]:
        """Evaluate all trained models."""
        logger.info("=" * 60)
        logger.info("Starting comprehensive model evaluation")
        logger.info("=" * 60)

        # Find all model checkpoints
        model_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model_name = model_dir.name
            logger.info(f"\nEvaluating {model_name}...")

            try:
                # Load model based on type
                if model_name in ["random_forest", "svr"]:
                    # Classical model
                    model_path = self.checkpoint_dir / f"{model_name}.joblib"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        results = self.evaluate_classical_model(model, model_name)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
                        continue
                else:
                    # Lightning model - find best checkpoint
                    ckpt_files = list(model_dir.glob("*.ckpt"))
                    if not ckpt_files:
                        logger.warning(f"No checkpoints found in {model_dir}")
                        continue

                    # Use last.ckpt or best checkpoint
                    ckpt_path = model_dir / "last.ckpt"
                    if not ckpt_path.exists() and ckpt_files:
                        ckpt_path = ckpt_files[0]

                    results = self.evaluate_lightning_model(ckpt_path, model_name)

                self.evaluation_results[model_name] = results
                logger.info(f"✓ {model_name} evaluation complete")

            except Exception as e:
                logger.error(f"✗ Failed to evaluate {model_name}: {str(e)}")
                self.evaluation_results[model_name] = {"error": str(e)}

        # Save comprehensive results
        self.save_evaluation_results()

        # Generate plots
        self.generate_evaluation_plots()

        # Generate LaTeX tables
        self.generate_latex_tables()

        return self.evaluation_results

    def evaluate_lightning_model(
        self, checkpoint_path: Path, model_name: str
    ) -> dict[str, float]:
        """Evaluate a PyTorch Lightning model."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Dynamically load the model class based on name
        if "lstm" in model_name.lower():
            from src.models.baseline_models import TrajectoryLSTM

            model = TrajectoryLSTM.load_from_checkpoint(checkpoint_path)
        elif "kalman" in model_name.lower():
            from src.models.baseline_models.kalman.lightning_wrapper import (
                KalmanBaselineLightning,
            )

            model = KalmanBaselineLightning.load_from_checkpoint(checkpoint_path)
        elif "gcn" in model_name.lower():
            from src.models.baseline_models import VesselGCN

            model = VesselGCN.load_from_checkpoint(checkpoint_path)
        elif "autoencoder" in model_name.lower():
            from src.models.baseline_models import AnomalyAutoencoder

            model = AnomalyAutoencoder.load_from_checkpoint(checkpoint_path)
        elif "motion_transformer" in model_name.lower():
            from src.models import MotionTransformer

            model = MotionTransformer.load_from_checkpoint(checkpoint_path)
        elif "anomaly_transformer" in model_name.lower():
            from src.models import AnomalyTransformer

            model = AnomalyTransformer.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        model.eval()

        # Compute comprehensive metrics
        results = self._compute_comprehensive_metrics(model, model_name)

        return results

    def evaluate_classical_model(self, model: Any, model_name: str) -> dict[str, float]:
        """Evaluate a classical ML model."""
        results = {}

        # Prepare test data
        X_test, y_test = [], []
        for batch in self.test_loader:
            X_test.append(batch["input"].numpy())
            y_test.append(batch["target"].numpy())

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        # Reshape for classical models
        n_samples, seq_len, n_features = X_test.shape
        X_test_flat = X_test.reshape(n_samples, -1)

        # Predict
        y_pred_flat = model.predict(X_test_flat)
        y_pred = y_pred_flat.reshape(n_samples, seq_len - 1, n_features)

        # Compute metrics
        results["mse"] = float(np.mean((y_pred - y_test) ** 2))
        results["mae"] = float(np.mean(np.abs(y_pred - y_test)))
        results["rmse"] = float(np.sqrt(results["mse"]))

        # Position-specific metrics
        position_error = np.sqrt(np.mean((y_pred[:, :, :2] - y_test[:, :, :2]) ** 2))
        results["position_rmse"] = float(position_error)

        # Speed and course metrics
        speed_error = np.mean(np.abs(y_pred[:, :, 2] - y_test[:, :, 2]))
        results["speed_mae"] = float(speed_error)

        course_diff = np.abs(y_pred[:, :, 3] - y_test[:, :, 3])
        course_diff = np.minimum(course_diff, 360 - course_diff)  # Circular
        results["course_mae"] = float(np.mean(course_diff))

        return results

    def _compute_comprehensive_metrics(
        self, model: Any, model_name: str
    ) -> dict[str, float]:
        """Compute comprehensive metrics for a model."""
        results = {}
        predictions = []
        targets = []
        neighbors = []

        # Collect predictions
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Evaluating {model_name}"):
                inputs = batch["input"]
                target = batch["target"]

                # Get predictions based on model type
                if hasattr(model, "predict"):
                    pred = model.predict(inputs)
                elif hasattr(model, "forward"):
                    pred = model(inputs)
                else:
                    pred = model(inputs)

                # Handle different output formats
                if isinstance(pred, dict):
                    if "trajectories" in pred:
                        pred = pred["trajectories"][:, 0]  # Take first mode
                    elif "reconstruction" in pred:
                        pred = pred["reconstruction"]

                predictions.append(pred.cpu() if torch.is_tensor(pred) else pred)
                targets.append(target.cpu() if torch.is_tensor(target) else target)

                if "neighbors" in batch:
                    neighbors.append(batch["neighbors"].cpu())

        # Stack all predictions
        predictions = (
            torch.cat(predictions, dim=0)
            if torch.is_tensor(predictions[0])
            else np.concatenate(predictions)
        )
        targets = (
            torch.cat(targets, dim=0)
            if torch.is_tensor(targets[0])
            else np.concatenate(targets)
        )

        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.numpy()
        if torch.is_tensor(targets):
            targets = targets.numpy()

        # 1. Trajectory Metrics
        results["mse"] = float(np.mean((predictions - targets) ** 2))
        results["mae"] = float(np.mean(np.abs(predictions - targets)))
        results["rmse"] = float(np.sqrt(results["mse"]))

        # ADE (Average Displacement Error)
        position_errors = np.linalg.norm(
            predictions[:, :, :2] - targets[:, :, :2], axis=-1
        )
        results["ade"] = float(np.mean(position_errors))

        # FDE (Final Displacement Error)
        final_errors = np.linalg.norm(
            predictions[:, -1, :2] - targets[:, -1, :2], axis=-1
        )
        results["fde"] = float(np.mean(final_errors))

        # 2. Maritime-specific Metrics
        results["position_rmse_km"] = float(
            np.sqrt(np.mean((predictions[:, :, :2] - targets[:, :, :2]) ** 2)) * 111
        )  # Convert to km
        results["speed_mae_knots"] = float(
            np.mean(np.abs(predictions[:, :, 2] - targets[:, :, 2]))
        )

        # Course error (circular)
        course_diff = np.abs(predictions[:, :, 3] - targets[:, :, 3])
        course_diff = np.minimum(course_diff, 360 - course_diff)
        results["course_mae_degrees"] = float(np.mean(course_diff))

        # 3. Prediction Horizon Analysis
        for horizon in [1, 5, 10, 20]:
            if horizon < predictions.shape[1]:
                horizon_error = np.linalg.norm(
                    predictions[:, horizon, :2] - targets[:, horizon, :2], axis=-1
                )
                results[f"error_{horizon}step"] = float(np.mean(horizon_error))

        # 4. Collision Risk Metrics (if neighbors available)
        if neighbors:
            neighbors = torch.cat(neighbors, dim=0).numpy()
            collision_metrics = self._compute_collision_metrics(
                predictions, targets, neighbors
            )
            results.update(collision_metrics)

        # 5. Operational Metrics
        warning_events = self._generate_warning_events(
            predictions, targets, neighbors if neighbors else None
        )
        if warning_events:
            try:
                warning_stats = self.operational_metrics.warning_time_distribution(
                    warning_events
                )
                results["median_warning_time"] = warning_stats.median_warning_time
                results["p90_warning_time"] = warning_stats.p90_warning_time
            except Exception:
                pass  # Warning stats calculation failed, skip this metric

            false_alert_stats = self.operational_metrics.false_alert_rate(
                warning_events
            )
            results["false_alert_rate"] = false_alert_stats["overall_rate"]

        return results

    def _compute_collision_metrics(
        self, predictions: np.ndarray, targets: np.ndarray, neighbors: np.ndarray
    ) -> dict[str, float]:
        """Compute collision-related metrics."""
        metrics = {}

        CPA_COLLISION_THRESHOLD_M = 500  # Meters for collision risk
        cpa_errors = []
        tcpa_errors = []
        collision_detected = 0
        collision_missed = 0
        total_encounters = 0

        for b in range(min(10, predictions.shape[0])):  # Sample for efficiency
            for n in range(neighbors.shape[1]):
                # Calculate predicted CPA/TCPA
                pred_cpa, pred_tcpa, _ = self.cpa_calculator.calculate_cpa_tcpa(
                    predictions[b, :, :2],
                    neighbors[b, n, :, :2],
                    predictions[b, :, 2:4],
                    neighbors[b, n, :, 2:4],
                )

                # Calculate ground truth CPA/TCPA
                gt_cpa, gt_tcpa, _ = self.cpa_calculator.calculate_cpa_tcpa(
                    targets[b, :, :2],
                    neighbors[b, n, :, :2],
                    targets[b, :, 2:4],
                    neighbors[b, n, :, 2:4],
                )

                if gt_cpa is not None and pred_cpa is not None:
                    cpa_errors.append(abs(pred_cpa - gt_cpa))
                    total_encounters += 1

                if gt_tcpa is not None and pred_tcpa is not None:
                    tcpa_errors.append(abs(pred_tcpa - gt_tcpa))

                # Check collision detection (CPA < threshold)
                if gt_cpa is not None and gt_cpa < CPA_COLLISION_THRESHOLD_M:
                    if pred_cpa is not None and pred_cpa < CPA_COLLISION_THRESHOLD_M:
                        collision_detected += 1
                    else:
                        collision_missed += 1

        if cpa_errors:
            metrics["cpa_error_mean"] = float(np.mean(cpa_errors))
            metrics["cpa_error_std"] = float(np.std(cpa_errors))

        if tcpa_errors:
            metrics["tcpa_error_mean"] = float(np.mean(tcpa_errors))

        if collision_detected + collision_missed > 0:
            metrics["collision_recall"] = collision_detected / (
                collision_detected + collision_missed
            )

        metrics["total_encounters_evaluated"] = total_encounters

        return metrics

    def _generate_warning_events(
        self, predictions: np.ndarray, targets: np.ndarray, neighbors: np.ndarray | None
    ) -> list[WarningEvent]:
        """Generate warning events for operational metrics."""
        CPA_WARNING_THRESHOLD_M = 1000  # Meters for warning generation
        events = []

        if neighbors is None:
            return events

        for b in range(min(10, predictions.shape[0])):
            for n in range(neighbors.shape[1]):
                pred_cpa, pred_tcpa, _ = self.cpa_calculator.calculate_cpa_tcpa(
                    predictions[b, :, :2],
                    neighbors[b, n, :, :2],
                    predictions[b, :, 2:4],
                    neighbors[b, n, :, 2:4],
                )

                if pred_cpa is not None and pred_cpa < CPA_WARNING_THRESHOLD_M:
                    gt_cpa, _, _ = self.cpa_calculator.calculate_cpa_tcpa(
                        targets[b, :, :2],
                        neighbors[b, n, :, :2],
                        targets[b, :, 2:4],
                        neighbors[b, n, :, 2:4],
                    )

                    false_positive = gt_cpa is None or gt_cpa > CPA_WARNING_THRESHOLD_M

                    event = WarningEvent(
                        timestamp=float(b),
                        vessel_id=f"vessel_{b}",
                        threat_type="collision",
                        warning_time=pred_tcpa if pred_tcpa else 0,
                        false_positive=false_positive,
                        severity=self.operational_metrics.compute_severity(
                            pred_cpa, pred_tcpa * 60 if pred_tcpa else 0, "collision"
                        ),
                        cpa_distance=pred_cpa,
                        tcpa_time=pred_tcpa * 60 if pred_tcpa else 0,
                    )
                    events.append(event)

        return events

    def save_evaluation_results(self):
        """Save comprehensive evaluation results."""
        # Create results DataFrame
        results_df = pd.DataFrame(self.evaluation_results).T

        # Save as CSV
        csv_path = self.results_dir / "evaluation_results.csv"
        results_df.to_csv(csv_path)
        logger.info(f"Saved evaluation results to {csv_path}")

        # Save as JSON with metadata
        results_json = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.config.get("experiment_name", "unknown"),
            "n_test_samples": len(self.test_loader.dataset) if self.test_loader else 0,
            "results": self.evaluation_results,
        }

        json_path = self.results_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)

        # Print summary
        self._print_results_summary(results_df)

    def _print_results_summary(self, results_df: pd.DataFrame):
        """Print formatted results summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 80)

        # Key metrics to display
        key_metrics = [
            "rmse",
            "mae",
            "ade",
            "fde",
            "position_rmse_km",
            "collision_recall",
        ]

        # Filter available metrics
        available_metrics = [m for m in key_metrics if m in results_df.columns]

        if not available_metrics:
            available_metrics = results_df.columns[:6]  # Take first 6 metrics

        # Create summary table
        summary_df = results_df[available_metrics].round(4)

        # Sort by primary metric (RMSE or MAE)
        sort_metric = "rmse" if "rmse" in summary_df.columns else summary_df.columns[0]
        summary_df = summary_df.sort_values(sort_metric)

        logger.info("\n" + summary_df.to_string())

        # Identify best models
        logger.info("\n" + "=" * 80)
        logger.info("BEST MODELS BY METRIC:")
        logger.info("-" * 80)

        for metric in available_metrics:
            if metric in results_df.columns:
                best_model = (
                    results_df[metric].idxmin()
                    if "error" in metric or "rmse" in metric or "mae" in metric
                    else results_df[metric].idxmax()
                )
                best_value = results_df.loc[best_model, metric]
                logger.info(f"{metric:20s}: {best_model:25s} ({best_value:.4f})")

    def generate_evaluation_plots(self):
        """Generate comprehensive evaluation plots."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to plot")
            return

        # Create results DataFrame
        results_df = pd.DataFrame(self.evaluation_results).T

        # 1. Overall Performance Comparison
        self._plot_overall_performance(results_df)

        # 2. Prediction Horizon Analysis
        self._plot_horizon_analysis(results_df)

        # 3. Maritime-specific Metrics
        self._plot_maritime_metrics(results_df)

        # 4. Model Ranking Heatmap
        self._plot_model_ranking(results_df)

        logger.info(f"Saved all plots to {self.plots_dir}")

    def _plot_overall_performance(self, results_df: pd.DataFrame):
        """Plot overall performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ["rmse", "mae", "ade", "fde"]
        titles = ["RMSE", "MAE", "ADE", "FDE"]

        for ax, metric, title in zip(axes.flat, metrics, titles, strict=False):
            if metric in results_df.columns:
                data = results_df[metric].sort_values()
                ax.barh(data.index, data.values)
                ax.set_xlabel(title)
                ax.set_title(f"{title} Comparison")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "overall_performance.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_horizon_analysis(self, results_df: pd.DataFrame):
        """Plot prediction error vs horizon."""
        horizon_cols = [col for col in results_df.columns if "step" in col]

        if not horizon_cols:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in results_df.index:
            horizons = []
            errors = []
            for col in horizon_cols:
                if col in results_df.columns and not pd.isna(
                    results_df.loc[model, col]
                ):
                    horizon = int(col.replace("error_", "").replace("step", ""))
                    horizons.append(horizon)
                    errors.append(results_df.loc[model, col])

            if horizons:
                ax.plot(horizons, errors, marker="o", label=model)

        ax.set_xlabel("Prediction Horizon (steps)")
        ax.set_ylabel("Prediction Error")
        ax.set_title("Prediction Error vs Horizon")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "horizon_analysis.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_maritime_metrics(self, results_df: pd.DataFrame):
        """Plot maritime-specific metrics."""
        maritime_metrics = [
            "position_rmse_km",
            "speed_mae_knots",
            "course_mae_degrees",
            "collision_recall",
        ]
        available = [m for m in maritime_metrics if m in results_df.columns]

        if not available:
            return

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))

        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available, strict=False):
            data = results_df[metric].dropna().sort_values()
            ax.barh(data.index, data.values, color="skyblue")
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "maritime_metrics.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_model_ranking(self, results_df: pd.DataFrame):
        """Plot model ranking heatmap."""
        # Rank models for each metric
        ranking_df = pd.DataFrame(index=results_df.index)

        for col in results_df.columns:
            if results_df[col].dtype in [np.float64, np.int64]:
                # Lower is better for error metrics
                if any(
                    term in col
                    for term in ["error", "rmse", "mae", "mse", "fde", "ade"]
                ):
                    ranking_df[col] = results_df[col].rank()
                # Higher is better for accuracy/recall metrics
                else:
                    ranking_df[col] = results_df[col].rank(ascending=False)

        # Average rank
        ranking_df["avg_rank"] = ranking_df.mean(axis=1)
        ranking_df = ranking_df.sort_values("avg_rank")

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            ranking_df.T,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            cbar_kws={"label": "Rank (lower is better)"},
        )
        ax.set_title("Model Performance Ranking Across Metrics")
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "model_ranking_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def generate_latex_tables(self):
        """Generate LaTeX tables for paper."""
        if not self.evaluation_results:
            return

        results_df = pd.DataFrame(self.evaluation_results).T

        # Main results table
        latex_path = self.results_dir / "latex_tables.tex"
        with open(latex_path, "w") as f:
            # Table 1: Main Performance Metrics
            f.write("% Table 1: Main Performance Metrics\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Baseline Model Performance on Maritime Trajectory Prediction}\n"
            )
            f.write("\\label{tab:baseline_performance}\n")

            # Select key metrics
            key_metrics = ["rmse", "mae", "ade", "fde", "position_rmse_km"]
            available = [m for m in key_metrics if m in results_df.columns]

            if available:
                table_df = results_df[available].round(3)
                table_df = table_df.sort_values(available[0])

                # Convert to LaTeX
                latex_str = table_df.to_latex(escape=False)
                f.write(latex_str)
            f.write("\\end{table}\n\n")

            # Table 2: Maritime-Specific Metrics
            f.write("% Table 2: Maritime-Specific Metrics\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Maritime Domain-Specific Performance}\n")
            f.write("\\label{tab:maritime_metrics}\n")

            maritime_metrics = [
                "collision_recall",
                "false_alert_rate",
                "median_warning_time",
            ]
            available = [m for m in maritime_metrics if m in results_df.columns]

            if available:
                table_df = results_df[available].round(3)
                latex_str = table_df.to_latex(escape=False)
                f.write(latex_str)
            f.write("\\end{table}\n")

        logger.info(f"LaTeX tables saved to {latex_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained baseline models")

    parser.add_argument(
        "experiment_dir", type=str, help="Directory containing trained models"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data")

    args = parser.parse_args()

    # Create evaluator
    evaluator = BaselineEvaluator(args.experiment_dir, args.data_path)

    # Run evaluation
    evaluator.evaluate_all_models()

    logger.info("\n✅ Evaluation complete!")
    logger.info(f"Results saved to: {evaluator.results_dir}")
    logger.info(f"Plots saved to: {evaluator.plots_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
