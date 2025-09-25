"""
Lightning integration example for the evalx statistical evaluation framework.

This example demonstrates how to integrate evalx with PyTorch Lightning
for comprehensive model evaluation in maritime trajectory prediction.
"""

import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import pytorch_lightning as pl
    from torchmetrics import MetricCollection

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. This example requires Lightning.")

from evalx.metrics.enhanced_metrics import EvaluationRunner, StatisticalMetricWrapper
from metrics.trajectory_metrics import ADE, FDE, CourseRMSE, RMSEPosition


class MockTrajectoryModel(nn.Module):
    """Mock trajectory prediction model for demonstration."""

    def __init__(
        self, input_size: int = 13, hidden_size: int = 64, output_size: int = 4
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]  # [batch, hidden]
        # Predict next position and movement
        output = self.linear(last_output)  # [batch, 4] - lat, lon, sog, cog
        return output


class LightningTrajectoryModel(pl.LightningModule):
    """Lightning wrapper for trajectory prediction model with evalx integration."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        enable_statistical_metrics: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Setup metrics with statistical analysis
        base_metrics = {
            "ade": ADE(),
            "fde": FDE(),
            "rmse_pos": RMSEPosition(),
            "rmse_course": CourseRMSE(),
        }

        if enable_statistical_metrics:
            # Wrap metrics with statistical capabilities
            self.train_metrics = MetricCollection(
                {
                    name: StatisticalMetricWrapper(metric.clone())
                    for name, metric in base_metrics.items()
                }
            )
            self.val_metrics = MetricCollection(
                {
                    name: StatisticalMetricWrapper(metric.clone())
                    for name, metric in base_metrics.items()
                }
            )
        else:
            self.train_metrics = MetricCollection(base_metrics).clone(prefix="train_")
            self.val_metrics = MetricCollection(base_metrics).clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # Assume y contains [lat, lon, sog, cog]
        loss = nn.functional.mse_loss(y_hat, y)

        # Update metrics
        position_pred = y_hat[:, :2]  # lat, lon
        position_true = y[:, :2]
        course_pred = y_hat[:, 3]  # cog
        course_true = y[:, 3]

        # Update position metrics (need to add sequence dimension)
        pos_pred_seq = position_pred.unsqueeze(1)  # [B, 1, 2]
        pos_true_seq = position_true.unsqueeze(1)  # [B, 1, 2]

        self.train_metrics["ade"].update(pos_pred_seq, pos_true_seq)
        self.train_metrics["fde"].update(pos_pred_seq, pos_true_seq)
        self.train_metrics["rmse_pos"].update(pos_pred_seq, pos_true_seq)
        self.train_metrics["rmse_course"].update(course_pred, course_true)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = nn.functional.mse_loss(y_hat, y)

        # Update validation metrics
        position_pred = y_hat[:, :2]
        position_true = y[:, :2]
        course_pred = y_hat[:, 3]
        course_true = y[:, 3]

        pos_pred_seq = position_pred.unsqueeze(1)
        pos_true_seq = position_true.unsqueeze(1)

        self.val_metrics["ade"].update(pos_pred_seq, pos_true_seq)
        self.val_metrics["fde"].update(pos_pred_seq, pos_true_seq)
        self.val_metrics["rmse_pos"].update(pos_pred_seq, pos_true_seq)
        self.val_metrics["rmse_course"].update(course_pred, course_true)

        self.log("val_loss", loss)

    def on_train_epoch_end(self):
        # Compute and log training metrics
        train_results = self.train_metrics.compute()

        for name, result in train_results.items():
            if hasattr(result, "value"):  # Statistical wrapper result
                self.log(f"train_{name}", result.value)
                if result.bootstrap_ci:
                    ci = result.bootstrap_ci.confidence_interval
                    self.log(f"train_{name}_ci_lower", ci[0])
                    self.log(f"train_{name}_ci_upper", ci[1])
                    self.log(f"train_{name}_ci_width", ci[1] - ci[0])
            else:  # Regular metric result
                self.log(f"train_{name}", result)

        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        # Compute and log validation metrics
        val_results = self.val_metrics.compute()

        for name, result in val_results.items():
            if hasattr(result, "value"):  # Statistical wrapper result
                self.log(f"val_{name}", result.value)
                if result.bootstrap_ci:
                    ci = result.bootstrap_ci.confidence_interval
                    self.log(f"val_{name}_ci_lower", ci[0])
                    self.log(f"val_{name}_ci_upper", ci[1])
                    self.log(f"val_{name}_ci_width", ci[1] - ci[0])
            else:  # Regular metric result
                self.log(f"val_{name}", result)

        self.val_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def create_sample_maritime_data(n_samples: int = 1000, seq_len: int = 10):
    """Create sample maritime trajectory data."""
    np.random.seed(42)

    # Generate synthetic AIS-like data
    n_features = 13  # AIS features
    n_targets = 4  # lat, lon, sog, cog

    # Input sequences [batch, seq_len, features]
    X = torch.randn(n_samples, seq_len, n_features)

    # Target values [batch, targets]
    # Simulate realistic ranges for maritime data
    lat = torch.normal(60.0, 0.5, (n_samples,))  # Latitude around Faroe Islands
    lon = torch.normal(-7.0, 1.0, (n_samples,))  # Longitude around Faroe Islands
    sog = torch.abs(torch.normal(10.0, 3.0, (n_samples,)))  # Speed over ground (knots)
    cog = torch.uniform(0.0, 360.0, (n_samples,))  # Course over ground (degrees)

    y = torch.stack([lat, lon, sog, cog], dim=1)

    return X, y


def example_lightning_training_with_statistics():
    """Demonstrate Lightning training with statistical metrics."""
    print("=" * 60)
    print("LIGHTNING TRAINING WITH STATISTICAL METRICS")
    print("=" * 60)

    # Create sample data
    X, y = create_sample_maritime_data(n_samples=500)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    base_model = MockTrajectoryModel(input_size=13, hidden_size=64, output_size=4)
    lightning_model = LightningTrajectoryModel(
        base_model, learning_rate=1e-3, enable_statistical_metrics=True
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=3,  # Short for demo
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,  # Disable logging for clean output
    )

    print("\nTraining model with statistical metrics...")
    trainer.fit(lightning_model, train_loader, val_loader)

    print("\nTraining completed!")

    # Extract final metrics
    final_results = lightning_model.val_metrics.compute()

    print("\nFinal Validation Results:")
    print("-" * 30)
    for name, result in final_results.items():
        if hasattr(result, "value"):
            print(f"{name}: {result.value:.4f}")
            if result.bootstrap_ci:
                ci = result.bootstrap_ci.confidence_interval
                print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")


def example_model_comparison_with_evaluation_runner():
    """Demonstrate model comparison using EvaluationRunner."""
    print("\n\n" + "=" * 60)
    print("MODEL COMPARISON WITH EVALUATION RUNNER")
    print("=" * 60)

    # Create test data
    X, y = create_sample_maritime_data(n_samples=200)
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Create different model architectures
    models = {
        "LSTM_Small": MockTrajectoryModel(13, 32, 4),
        "LSTM_Medium": MockTrajectoryModel(13, 64, 4),
        "LSTM_Large": MockTrajectoryModel(13, 128, 4),
    }

    # Define evaluation metrics
    metrics = {"ade": ADE(), "fde": FDE(), "rmse_pos": RMSEPosition()}

    # Create evaluation runner
    evaluator = EvaluationRunner(
        metrics=metrics, confidence_level=0.95, comparison_method="holm"
    )

    print(f"Evaluating {len(models)} models on {len(test_dataset)} samples...")

    # Simulate multiple evaluation runs for robustness
    n_runs = 3
    print(f"Running {n_runs} evaluation rounds for statistical robustness...")

    # Compare models
    comparison_result = evaluator.compare_models(
        models=models, dataloader=test_loader, n_runs=n_runs
    )

    print("\nModel Comparison Results:")
    print("-" * 40)

    # Display best models per metric
    print("Best Models:")
    for metric, best_model in comparison_result.best_model.items():
        print(f"  {metric}: {best_model}")

    # Display summary table
    print("\nDetailed Results:")
    print(comparison_result.summary_table.round(4).to_string(index=False))

    # Display pairwise comparisons
    print("\nPairwise Comparisons (ADE):")
    print("-" * 30)
    for comparison, results in comparison_result.pairwise_tests.items():
        if "ade" in results:
            test = results["ade"]
            sig_marker = (
                "***"
                if test.p_value < 0.001
                else "**"
                if test.p_value < 0.01
                else "*"
                if test.significant
                else ""
            )
            print(f"  {comparison}: p={test.p_value:.4f} {sig_marker}")

    # Multiple comparison correction results
    if comparison_result.corrected_pvalues:
        correction = comparison_result.corrected_pvalues["correction_result"]
        print(f"\nMultiple Comparison Correction ({correction.method}):")
        print(
            f"  Significant comparisons: {correction.n_significant}/{correction.n_comparisons}"
        )


def example_cross_validation_integration():
    """Demonstrate cross-validation with Lightning and evalx."""
    print("\n\n" + "=" * 60)
    print("CROSS-VALIDATION INTEGRATION")
    print("=" * 60)

    # This would typically integrate with the maritime_cv_split function
    # For this example, we'll simulate the process

    print("Cross-validation integration would involve:")
    print("1. Using maritime_cv_split() to create CV folds")
    print("2. Training Lightning models on each fold")
    print("3. Collecting metrics with statistical wrappers")
    print("4. Performing model comparison across folds")
    print("\nExample workflow:")

    workflow_code = """
    # Create CV splits
    cv_splits = maritime_cv_split(df, split_type='vessel', n_splits=5)

    # Collect results across folds
    fold_results = {}

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        # Create fold-specific data loaders
        train_loader = create_dataloader(df.iloc[train_idx])
        val_loader = create_dataloader(df.iloc[val_idx])

        # Train model
        model = LightningTrajectoryModel(base_model)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model, train_loader, val_loader)

        # Collect statistical metrics
        fold_metrics = model.val_metrics.compute()
        fold_results[f'fold_{fold}'] = fold_metrics

    # Compare models using collected results
    comparison = ModelComparison()
    final_results = comparison.compare_models(fold_results)
    """

    print(workflow_code)


def main():
    """Run Lightning integration examples."""
    if not LIGHTNING_AVAILABLE:
        print("PyTorch Lightning is required for this example.")
        print("Install with: pip install pytorch-lightning torchmetrics")
        return

    print("EVALX LIGHTNING INTEGRATION EXAMPLES")
    print("=" * 60)

    try:
        example_lightning_training_with_statistics()
        example_model_comparison_with_evaluation_runner()
        example_cross_validation_integration()

        print("\n\n" + "=" * 60)
        print("LIGHTNING INTEGRATION EXAMPLES COMPLETED!")
        print("=" * 60)
        print("\nKey Integration Points:")
        print("- StatisticalMetricWrapper: Adds CI computation to torchmetrics")
        print("- EvaluationRunner: High-level model comparison interface")
        print("- Lightning compatibility: Seamless integration with training loops")
        print("- Cross-validation support: Works with maritime_cv_split protocols")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
