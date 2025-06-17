"""
Benchmark experiment module.

Migrates test_benchmark_models.py logic into the unified experiment structure
following the CLAUDE blueprint.
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

from ..data.datamodule import create_simple_datamodule
from ..models.factory import create_model
from ..training.trainer import create_trainer

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """
    Comprehensive model benchmarking engine.

    Migrates functionality from test_benchmark_models.py into the unified system.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the model benchmarker.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.benchmark_config = config.benchmark
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Results storage
        self.results = {
            "model_tests": {},
            "training_tests": {},
            "performance_benchmarks": {},
            "system_info": self._get_system_info(),
        }

        logger.info(f"Benchmark engine initialized on device: {self.device}")

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                    "gpu_count": torch.cuda.device_count(),
                }
            )

        return info

    def _create_dummy_data(
        self, batch_size: int = 32, sequence_length: int = 30
    ) -> torch.Tensor:
        """Create dummy data for testing."""
        # Generate realistic maritime AIS data
        data = torch.randn(batch_size, sequence_length, 4)  # lat, lon, sog, cog

        # Add some realistic constraints
        data[:, :, 0] = torch.sigmoid(data[:, :, 0]) * 90  # latitude: -90 to 90
        data[:, :, 1] = (
            torch.sigmoid(data[:, :, 1]) * 360 - 180
        )  # longitude: -180 to 180
        data[:, :, 2] = torch.relu(data[:, :, 2]) * 25  # speed: 0 to 25 knots
        data[:, :, 3] = torch.sigmoid(data[:, :, 3]) * 360  # course: 0 to 360 degrees

        return data.to(self.device)

    def test_trajectory_prediction_model(self) -> dict[str, Any]:
        """Test trajectory prediction model."""
        logger.info("Testing trajectory prediction model...")

        test_results = {
            "success": False,
            "error": None,
            "metrics": {},
            "inference_time": None,
        }

        try:
            # Create trajectory prediction model
            model_config = self.config.model.copy()
            model_config.task = "trajectory_prediction"
            model_config.type = "TRAISFORMER"

            model = create_model(model_config)
            model = model.to(self.device)
            model.eval()

            # Test with dummy data
            batch_size = 16
            sequence_length = 30
            input_data = self._create_dummy_data(batch_size, sequence_length)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                if hasattr(model, "predict_trajectory"):
                    output = model.predict_trajectory(input_data, steps=12)
                else:
                    output = model(input_data)

            inference_time = time.time() - start_time

            # Validate output
            if torch.is_tensor(output):
                output_shape = output.shape
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()

                test_results.update(
                    {
                        "success": True,
                        "metrics": {
                            "output_shape": list(output_shape),
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                            "output_mean": float(output.mean()),
                            "output_std": float(output.std()),
                        },
                        "inference_time": inference_time,
                    }
                )

                logger.info(
                    f"Trajectory prediction test PASSED - Output shape: {output_shape}"
                )
            else:
                test_results["error"] = "Model output is not a tensor"

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Trajectory prediction test FAILED: {e}")

        return test_results

    def test_anomaly_detection_model(self) -> dict[str, Any]:
        """Test anomaly detection model."""
        logger.info("Testing anomaly detection model...")

        test_results = {
            "success": False,
            "error": None,
            "metrics": {},
            "inference_time": None,
        }

        try:
            # Create anomaly detection model
            model_config = self.config.model.copy()
            model_config.task = "anomaly_detection"
            model_config.type = "ANOMALY_TRANSFORMER"

            model = create_model(model_config)
            model = model.to(self.device)
            model.eval()

            # Test with dummy data
            batch_size = 16
            sequence_length = 30
            input_data = self._create_dummy_data(batch_size, sequence_length)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                if hasattr(model, "detect_anomalies"):
                    output = model.detect_anomalies(input_data)
                else:
                    output = model(input_data)

            inference_time = time.time() - start_time

            # Validate output
            if torch.is_tensor(output):
                output_shape = output.shape
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()

                # For anomaly detection, check if output is between 0 and 1
                output_sigmoid = torch.sigmoid(output)
                in_range = ((output_sigmoid >= 0) & (output_sigmoid <= 1)).all().item()

                test_results.update(
                    {
                        "success": True,
                        "metrics": {
                            "output_shape": list(output_shape),
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                            "in_valid_range": in_range,
                            "anomaly_rate": float(output_sigmoid.mean()),
                        },
                        "inference_time": inference_time,
                    }
                )

                logger.info(
                    f"Anomaly detection test PASSED - Output shape: {output_shape}"
                )
            else:
                test_results["error"] = "Model output is not a tensor"

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Anomaly detection test FAILED: {e}")

        return test_results

    def test_vessel_interaction_model(self) -> dict[str, Any]:
        """Test vessel interaction model."""
        logger.info("Testing vessel interaction model...")

        test_results = {
            "success": False,
            "error": None,
            "metrics": {},
            "inference_time": None,
        }

        try:
            # Create vessel interaction model
            model_config = self.config.model.copy()
            model_config.task = "vessel_interaction"
            model_config.type = "AIS_FUSER"

            model = create_model(model_config)
            model = model.to(self.device)
            model.eval()

            # Test with dummy data
            batch_size = 16
            sequence_length = 30
            input_data = self._create_dummy_data(batch_size, sequence_length)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                output = model(input_data)

            inference_time = time.time() - start_time

            # Validate output
            if torch.is_tensor(output):
                output_shape = output.shape
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()

                test_results.update(
                    {
                        "success": True,
                        "metrics": {
                            "output_shape": list(output_shape),
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                            "output_mean": float(output.mean()),
                            "output_std": float(output.std()),
                        },
                        "inference_time": inference_time,
                    }
                )

                logger.info(
                    f"Vessel interaction test PASSED - Output shape: {output_shape}"
                )
            else:
                test_results["error"] = "Model output is not a tensor"

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Vessel interaction test FAILED: {e}")

        return test_results

    def test_training_pipeline(self) -> dict[str, Any]:
        """Test the complete training pipeline."""
        logger.info("Testing training pipeline...")

        test_results = {
            "success": False,
            "error": None,
            "metrics": {},
            "training_time": None,
        }

        try:
            # Create minimal training configuration
            train_config = self.config.copy()
            train_config.trainer.max_epochs = 2  # Just a quick test
            train_config.data.batch_size = 8
            train_config.trainer.check_val_every_n_epoch = 1

            # Create model and trainer
            model = create_model(train_config.model)
            trainer_wrapper = create_trainer(train_config)

            # Create simple data module with dummy data
            datamodule = create_simple_datamodule(train_config)

            # Measure training time
            start_time = time.time()

            # Run training
            trainer_wrapper.fit(model, datamodule)

            training_time = time.time() - start_time

            # Check if training completed
            trainer = trainer_wrapper.trainer

            test_results.update(
                {
                    "success": True,
                    "metrics": {
                        "completed_epochs": trainer.current_epoch,
                        "global_step": trainer.global_step,
                        "model_parameters": sum(p.numel() for p in model.parameters()),
                        "device_used": str(trainer.strategy.root_device),
                    },
                    "training_time": training_time,
                }
            )

            logger.info(
                f"Training pipeline test PASSED - Completed {trainer.current_epoch + 1} epochs"
            )

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Training pipeline test FAILED: {e}")

        return test_results

    def benchmark_performance(self) -> dict[str, Any]:
        """Benchmark computational performance across different configurations."""
        logger.info("Benchmarking computational performance...")

        benchmark_results = {}

        # Test different batch sizes
        batch_sizes = self.benchmark_config.batch_sizes
        sequence_lengths = self.benchmark_config.sequence_lengths
        iterations = self.benchmark_config.iterations
        warmup_iterations = self.benchmark_config.warmup_iterations

        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                config_name = f"batch_{batch_size}_seq_{seq_len}"
                logger.info(f"Benchmarking configuration: {config_name}")

                try:
                    # Create model for benchmarking
                    model = create_model(self.config.model)
                    model = model.to(self.device)
                    model.eval()

                    # Create test data
                    input_data = self._create_dummy_data(batch_size, seq_len)

                    # Warmup runs
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            _ = model(input_data)

                    # Benchmark runs
                    times = []
                    memory_usage = []

                    for _ in range(iterations):
                        # Clear cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        # Measure memory before
                        if torch.cuda.is_available():
                            mem_before = torch.cuda.memory_allocated()

                        start_time = time.time()

                        with torch.no_grad():
                            output = model(input_data)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        end_time = time.time()

                        # Measure memory after
                        if torch.cuda.is_available():
                            mem_after = torch.cuda.memory_allocated()
                            memory_usage.append(mem_after - mem_before)

                        times.append(end_time - start_time)

                    benchmark_results[config_name] = {
                        "batch_size": batch_size,
                        "sequence_length": seq_len,
                        "mean_time": np.mean(times),
                        "std_time": np.std(times),
                        "min_time": np.min(times),
                        "max_time": np.max(times),
                        "throughput": batch_size / np.mean(times),  # samples per second
                        "mean_memory": np.mean(memory_usage) if memory_usage else 0,
                        "iterations": iterations,
                    }

                    logger.info(
                        f"  Mean time: {np.mean(times):.4f}s, Throughput: {batch_size / np.mean(times):.1f} samples/s"
                    )

                except Exception as e:
                    logger.error(f"Benchmark failed for {config_name}: {e}")
                    benchmark_results[config_name] = {"error": str(e)}

        return benchmark_results

    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        logger.info("Generating benchmark report...")

        report = f"""
# Model Benchmark Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Information
- **Device**: {self.results['system_info']['device']}
- **PyTorch Version**: {self.results['system_info']['pytorch_version']}
- **Python Version**: {self.results['system_info']['python_version']}
"""

        if "cuda_version" in self.results["system_info"]:
            report += f"""
- **CUDA Version**: {self.results['system_info']['cuda_version']}
- **GPU**: {self.results['system_info']['gpu_name']}
- **GPU Memory**: {self.results['system_info']['gpu_memory']}
"""

        report += "\n## Model Tests\n"

        # Model test results
        if "model_tests" in self.results:
            for test_name, result in self.results["model_tests"].items():
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                report += f"\n### {test_name.replace('_', ' ').title()}\n"
                report += f"**Status**: {status}\n"

                if result["success"]:
                    report += f"**Inference Time**: {result['inference_time']:.4f}s\n"
                    if "metrics" in result:
                        for metric, value in result["metrics"].items():
                            report += (
                                f"- **{metric.replace('_', ' ').title()}**: {value}\n"
                            )
                else:
                    report += f"**Error**: {result['error']}\n"

        # Training test results
        if "training_tests" in self.results and self.results["training_tests"]:
            report += "\n## Training Pipeline Test\n"
            result = self.results["training_tests"]
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            report += f"**Status**: {status}\n"

            if result["success"]:
                report += f"**Training Time**: {result['training_time']:.2f}s\n"
                if "metrics" in result:
                    for metric, value in result["metrics"].items():
                        report += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
            else:
                report += f"**Error**: {result['error']}\n"

        # Performance benchmark results
        if (
            "performance_benchmarks" in self.results
            and self.results["performance_benchmarks"]
        ):
            report += "\n## Performance Benchmarks\n"

            benchmarks = self.results["performance_benchmarks"]

            # Create summary table
            report += "\n### Performance Summary\n"
            report += "| Configuration | Mean Time (s) | Throughput (samples/s) | Memory (MB) |\n"
            report += "|---------------|---------------|------------------------|-------------|\n"

            for config_name, metrics in benchmarks.items():
                if "error" not in metrics:
                    mean_time = metrics["mean_time"]
                    throughput = metrics["throughput"]
                    memory_mb = (
                        metrics["mean_memory"] / 1e6
                        if metrics["mean_memory"] > 0
                        else 0
                    )
                    report += f"| {config_name} | {mean_time:.4f} | {throughput:.1f} | {memory_mb:.1f} |\n"

        report += "\n## Conclusions\n"

        # Analyze results
        passed_tests = sum(
            1
            for result in self.results.get("model_tests", {}).values()
            if result["success"]
        )
        total_tests = len(self.results.get("model_tests", {}))

        if passed_tests == total_tests and total_tests > 0:
            report += "- ✅ All model tests passed successfully\n"
        else:
            report += f"- ⚠️  {passed_tests}/{total_tests} model tests passed\n"

        if self.results.get("training_tests", {}).get("success"):
            report += "- ✅ Training pipeline is functional\n"
        elif "training_tests" in self.results:
            report += "- ❌ Training pipeline has issues\n"

        report += "- 📊 Performance benchmarks completed\n"
        report += "- 🚀 System is ready for maritime trajectory prediction tasks\n"

        return report

    def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting comprehensive model benchmarking")
        logger.info("=" * 60)

        # Model tests
        logger.info("Running model tests...")
        self.results["model_tests"] = {
            "trajectory_prediction": self.test_trajectory_prediction_model(),
            "anomaly_detection": self.test_anomaly_detection_model(),
            "vessel_interaction": self.test_vessel_interaction_model(),
        }

        # Training pipeline test (optional)
        if self.benchmark_config.get("test_training", True):
            logger.info("Running training pipeline test...")
            self.results["training_tests"] = self.test_training_pipeline()

        # Performance benchmarks
        if self.benchmark_config.get("test_performance", True):
            logger.info("Running performance benchmarks...")
            self.results["performance_benchmarks"] = self.benchmark_performance()

        # Generate report
        report = self.generate_benchmark_report()

        logger.info("=" * 60)
        logger.info("Benchmarking completed successfully!")

        return {"results": self.results, "report": report}


def run_benchmarking(cfg: DictConfig) -> dict[str, Any]:
    """
    Main benchmarking function called by Hydra dispatch.

    Args:
        cfg: Hydra configuration object

    Returns:
        Benchmark results dictionary
    """
    logger.info("Starting benchmarking pipeline")
    logger.info("=" * 60)

    # Initialize benchmarker
    benchmarker = ModelBenchmarker(cfg)

    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark()

    # Save results if output directory specified
    if cfg.benchmark.get("output_dir"):
        output_dir = Path(cfg.benchmark.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save benchmark results
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in results["results"].items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value

            json.dump(json_results, f, indent=2)

        logger.info(f"Benchmark results saved to: {results_file}")

        # Save report
        report_file = output_dir / "benchmark_report.md"
        with open(report_file, "w") as f:
            f.write(results["report"])

        logger.info(f"Benchmark report saved to: {report_file}")

        # Generate plots if requested
        if cfg.benchmark.get("generate_plots", True):
            _generate_benchmark_plots(results["results"], output_dir)

    # Log summary
    logger.info("=" * 60)
    logger.info("Benchmarking completed successfully!")
    logger.info("=" * 60)

    # Print key results
    if "model_tests" in results["results"]:
        passed = sum(
            1 for r in results["results"]["model_tests"].values() if r["success"]
        )
        total = len(results["results"]["model_tests"])
        logger.info(f"Model Tests: {passed}/{total} passed")

    if "performance_benchmarks" in results["results"]:
        benchmarks = results["results"]["performance_benchmarks"]
        if benchmarks:
            best_config = min(
                benchmarks.keys(),
                key=lambda k: benchmarks[k].get("mean_time", float("inf")),
            )
            best_time = benchmarks[best_config].get("mean_time", 0)
            logger.info(f"Best Performance: {best_config} ({best_time:.4f}s)")

    return results


def _generate_benchmark_plots(results: dict[str, Any], output_dir: Path):
    """Generate benchmark visualization plots."""
    try:
        logger.info("Generating benchmark plots...")

        if "performance_benchmarks" in results and results["performance_benchmarks"]:
            benchmarks = results["performance_benchmarks"]

            # Filter valid results
            valid_benchmarks = {k: v for k, v in benchmarks.items() if "error" not in v}

            if valid_benchmarks:
                # Performance comparison plot
                plt.figure(figsize=(12, 8))

                configs = list(valid_benchmarks.keys())
                times = [valid_benchmarks[c]["mean_time"] for c in configs]
                throughputs = [valid_benchmarks[c]["throughput"] for c in configs]

                # Subplot 1: Inference time
                plt.subplot(2, 2, 1)
                plt.bar(range(len(configs)), times)
                plt.xlabel("Configuration")
                plt.ylabel("Mean Inference Time (s)")
                plt.title("Inference Time by Configuration")
                plt.xticks(range(len(configs)), configs, rotation=45)

                # Subplot 2: Throughput
                plt.subplot(2, 2, 2)
                plt.bar(range(len(configs)), throughputs)
                plt.xlabel("Configuration")
                plt.ylabel("Throughput (samples/s)")
                plt.title("Throughput by Configuration")
                plt.xticks(range(len(configs)), configs, rotation=45)

                # Subplot 3: Memory usage (if available)
                memory_usage = [
                    valid_benchmarks[c]["mean_memory"] / 1e6 for c in configs
                ]
                if any(m > 0 for m in memory_usage):
                    plt.subplot(2, 2, 3)
                    plt.bar(range(len(configs)), memory_usage)
                    plt.xlabel("Configuration")
                    plt.ylabel("Memory Usage (MB)")
                    plt.title("Memory Usage by Configuration")
                    plt.xticks(range(len(configs)), configs, rotation=45)

                plt.tight_layout()
                plt.savefig(
                    output_dir / "benchmark_performance.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                logger.info("Performance plots saved")

    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")
