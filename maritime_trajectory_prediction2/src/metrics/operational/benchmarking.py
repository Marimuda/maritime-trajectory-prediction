"""
Throughput benchmarking for maritime trajectory prediction models.

Provides hardware-aware performance measurement for deployment planning.
"""

import platform
import time
from typing import Any

import numpy as np
import psutil

from .ops_metrics import ThroughputResult

# Try to import PyTorch for GPU benchmarking
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ThroughputBenchmark:
    """
    Benchmark inference throughput on different hardware configurations.

    Measures real-world performance metrics for deployment decisions.
    """

    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        """
        Initialize throughput benchmark.

        Args:
            warmup_iterations: Number of warmup runs before measurement
            benchmark_iterations: Number of iterations for benchmarking
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def benchmark_model(
        self,
        model: Any,
        test_data: np.ndarray,
        batch_sizes: list[int] | None = None,
        use_gpu: bool = False,
    ) -> dict[int, ThroughputResult]:
        """
        Benchmark model throughput with different batch sizes.

        Args:
            model: Model to benchmark (should have predict/forward method)
            test_data: Test data shape [n_samples, seq_len, features]
            batch_sizes: List of batch sizes to test
            use_gpu: Whether to use GPU if available

        Returns:
            Dictionary mapping batch_size to ThroughputResult
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]

        results = {}
        n_samples = test_data.shape[0]

        for batch_size in batch_sizes:
            if batch_size > n_samples:
                continue

            # Prepare batched data
            n_batches = n_samples // batch_size
            batches = [
                test_data[i * batch_size : (i + 1) * batch_size]
                for i in range(n_batches)
            ]

            # Run benchmark
            if TORCH_AVAILABLE and use_gpu and torch.cuda.is_available():
                result = self._benchmark_gpu(model, batches, batch_size)
            else:
                result = self._benchmark_cpu(model, batches, batch_size)

            results[batch_size] = result

        return results

    def _benchmark_cpu(
        self, model: Any, batches: list[np.ndarray], batch_size: int
    ) -> ThroughputResult:
        """Benchmark on CPU."""
        # Warmup
        for _ in range(min(self.warmup_iterations, len(batches))):
            _ = self._run_inference(model, batches[0])

        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Benchmark
        latencies = []
        start_time = time.perf_counter()

        for i in range(min(self.benchmark_iterations, len(batches))):
            batch = batches[i % len(batches)]
            batch_start = time.perf_counter()
            _ = self._run_inference(model, batch)
            batch_end = time.perf_counter()
            latencies.append((batch_end - batch_start) * 1000)  # ms

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = mem_after - mem_before

        # Calculate metrics
        n_processed = min(self.benchmark_iterations, len(batches))
        batches_per_second = n_processed / total_time
        vessels_per_second = batches_per_second * batch_size

        latencies_array = np.array(latencies)
        mean_latency = np.mean(latencies_array)
        p95_latency = np.percentile(latencies_array, 95)
        p99_latency = np.percentile(latencies_array, 99)

        return ThroughputResult(
            vessels_per_second=vessels_per_second,
            batches_per_second=batches_per_second,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            memory_usage_mb=memory_increase,
            batch_size=batch_size,
            hardware_info=self._get_hardware_info(use_gpu=False),
        )

    def _benchmark_gpu(
        self, model: Any, batches: list[np.ndarray], batch_size: int
    ) -> ThroughputResult:
        """Benchmark on GPU."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU benchmarking")

        device = torch.device("cuda")

        # Convert model to GPU if needed
        if hasattr(model, "to"):
            model = model.to(device)

        # Convert batches to GPU tensors
        gpu_batches = [torch.from_numpy(batch).float().to(device) for batch in batches]

        # Warmup
        for _ in range(min(self.warmup_iterations, len(gpu_batches))):
            with torch.no_grad():
                _ = self._run_inference(model, gpu_batches[0])

        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        for i in range(min(self.benchmark_iterations, len(gpu_batches))):
            batch = gpu_batches[i % len(gpu_batches)]

            torch.cuda.synchronize()
            batch_start = time.perf_counter()

            with torch.no_grad():
                _ = self._run_inference(model, batch)

            torch.cuda.synchronize()
            batch_end = time.perf_counter()

            latencies.append((batch_end - batch_start) * 1000)  # ms

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # GPU metrics
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        gpu_util = self._estimate_gpu_utilization()

        # Calculate metrics
        n_processed = min(self.benchmark_iterations, len(gpu_batches))
        batches_per_second = n_processed / total_time
        vessels_per_second = batches_per_second * batch_size

        latencies_array = np.array(latencies)
        mean_latency = np.mean(latencies_array)
        p95_latency = np.percentile(latencies_array, 95)
        p99_latency = np.percentile(latencies_array, 99)

        return ThroughputResult(
            vessels_per_second=vessels_per_second,
            batches_per_second=batches_per_second,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            gpu_utilization=gpu_util,
            memory_usage_mb=peak_memory,
            batch_size=batch_size,
            hardware_info=self._get_hardware_info(use_gpu=True),
        )

    def _run_inference(self, model: Any, batch: Any) -> Any:
        """Run inference on a batch."""
        if hasattr(model, "predict"):
            return model.predict(batch)
        elif hasattr(model, "forward"):
            return model.forward(batch)
        elif callable(model):
            return model(batch)
        else:
            raise ValueError("Model must have predict/forward method or be callable")

    def _get_hardware_info(self, use_gpu: bool) -> dict[str, str]:
        """Get hardware information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": str(psutil.cpu_count()),
            "ram_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
        }

        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "N/A"

        return info

    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization (simplified)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0

        # This is a simplified estimate
        # Real GPU utilization requires nvidia-ml-py or similar
        return 75.0  # Placeholder estimate

    def analyze_scaling(self, results: dict[int, ThroughputResult]) -> dict[str, Any]:
        """
        Analyze how performance scales with batch size.

        Args:
            results: Dictionary of batch_size -> ThroughputResult

        Returns:
            Scaling analysis with recommendations
        """
        if not results:
            return {}

        batch_sizes = sorted(results.keys())
        throughputs = [results[bs].vessels_per_second for bs in batch_sizes]
        latencies = [results[bs].mean_latency_ms for bs in batch_sizes]

        # Find optimal batch size (best throughput with reasonable latency)
        optimal_batch = None
        max_throughput = 0
        latency_threshold = 100  # ms

        for bs in batch_sizes:
            if (
                results[bs].mean_latency_ms <= latency_threshold
                and results[bs].vessels_per_second > max_throughput
            ):
                max_throughput = results[bs].vessels_per_second
                optimal_batch = bs

        # Calculate scaling efficiency
        if len(batch_sizes) > 1:
            # Compare largest to smallest batch size
            scaling_factor = throughputs[-1] / throughputs[0]
            batch_increase = batch_sizes[-1] / batch_sizes[0]
            scaling_efficiency = scaling_factor / batch_increase
        else:
            scaling_efficiency = 1.0

        return {
            "optimal_batch_size": optimal_batch,
            "max_throughput": max_throughput,
            "scaling_efficiency": scaling_efficiency,
            "batch_sizes": batch_sizes,
            "throughputs": throughputs,
            "latencies": latencies,
            "recommendation": self._get_scaling_recommendation(
                optimal_batch, scaling_efficiency
            ),
        }

    def _get_scaling_recommendation(
        self, optimal_batch: int | None, scaling_efficiency: float
    ) -> str:
        """Generate deployment recommendation based on results."""
        POOR_SCALING_THRESHOLD = 0.5

        if optimal_batch is None:
            return (
                "No batch size meets latency requirements. Consider model optimization."
            )
        elif optimal_batch == 1:
            return "Single-sample inference is optimal. Consider model simplification for better batching."
        elif scaling_efficiency < POOR_SCALING_THRESHOLD:
            return f"Poor scaling efficiency ({scaling_efficiency:.2f}). Batch size {optimal_batch} recommended."
        else:
            return f"Good scaling. Batch size {optimal_batch} provides best throughput/latency balance."
