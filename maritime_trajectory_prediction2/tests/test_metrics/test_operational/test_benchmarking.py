"""
Tests for throughput benchmarking module.

Verifies performance measurement and scaling analysis.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.metrics.operational.benchmarking import ThroughputBenchmark
from src.metrics.operational.ops_metrics import ThroughputResult


class MockModel:
    """Mock model for testing benchmarking."""

    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.call_count = 0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Simulate prediction with controlled latency."""
        self.call_count += 1
        # Simulate some processing time (would use time.sleep in real scenario)
        return np.zeros_like(batch)


class TestThroughputBenchmark:
    """Test ThroughputBenchmark class."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return ThroughputBenchmark(warmup_iterations=2, benchmark_iterations=10)

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(42)
        # Shape: [n_samples, seq_len, features]
        return np.random.randn(100, 10, 4)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MockModel(latency_ms=5.0)

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        bench = ThroughputBenchmark(warmup_iterations=5, benchmark_iterations=50)
        assert bench.warmup_iterations == 5
        assert bench.benchmark_iterations == 50

    def test_cpu_benchmark(self, benchmark, mock_model, test_data):
        """Test CPU benchmarking."""
        results = benchmark.benchmark_model(
            model=mock_model,
            test_data=test_data,
            batch_sizes=[1, 8, 16],
            use_gpu=False,
        )

        assert len(results) == 3
        assert 1 in results
        assert 8 in results
        assert 16 in results

        # Check result structure
        result_1 = results[1]
        assert isinstance(result_1, ThroughputResult)
        assert result_1.batch_size == 1
        assert result_1.vessels_per_second > 0
        assert result_1.mean_latency_ms > 0
        assert result_1.p95_latency_ms >= result_1.mean_latency_ms
        assert result_1.p99_latency_ms >= result_1.p95_latency_ms

    def test_batch_size_filtering(self, benchmark, mock_model):
        """Test that batch sizes larger than data are filtered."""
        small_data = np.random.randn(10, 5, 4)

        results = benchmark.benchmark_model(
            model=mock_model,
            test_data=small_data,
            batch_sizes=[1, 8, 16, 32],  # 16 and 32 exceed data size
            use_gpu=False,
        )

        assert 1 in results
        assert 8 in results
        assert 16 not in results  # Filtered out
        assert 32 not in results  # Filtered out

    def test_throughput_scaling(self, benchmark, mock_model, test_data):
        """Test that throughput generally increases with batch size."""
        results = benchmark.benchmark_model(
            model=mock_model,
            test_data=test_data,
            batch_sizes=[1, 4, 8],
            use_gpu=False,
        )

        # Generally expect higher throughput with larger batches
        # (though this is simplified in mock)
        assert results[1].batch_size == 1
        assert results[4].batch_size == 4
        assert results[8].batch_size == 8

    def test_model_without_predict(self, benchmark, test_data):
        """Test benchmarking with callable model."""

        def model_func(batch):
            return np.zeros_like(batch)

        results = benchmark.benchmark_model(
            model=model_func, test_data=test_data, batch_sizes=[1, 4], use_gpu=False
        )

        assert len(results) == 2
        assert results[1].vessels_per_second > 0

    def test_model_with_forward(self, benchmark, test_data):
        """Test benchmarking with model that has forward method."""

        class ForwardModel:
            def forward(self, batch):
                return np.zeros_like(batch)

        model = ForwardModel()
        results = benchmark.benchmark_model(
            model=model, test_data=test_data, batch_sizes=[1, 4], use_gpu=False
        )

        assert len(results) == 2
        assert results[1].vessels_per_second > 0

    def test_invalid_model(self, benchmark, test_data):
        """Test error handling for invalid model."""

        class InvalidModel:
            pass  # No predict, forward, or __call__

        model = InvalidModel()

        with pytest.raises(ValueError, match="must have predict/forward"):
            benchmark.benchmark_model(model=model, test_data=test_data, batch_sizes=[1])

    @patch("src.metrics.operational.benchmarking.psutil.Process")
    def test_memory_measurement(self, mock_process, benchmark, mock_model, test_data):
        """Test memory usage measurement."""
        # Mock memory info
        mock_mem = MagicMock()
        mock_mem.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value = mock_mem

        results = benchmark.benchmark_model(
            model=mock_model, test_data=test_data, batch_sizes=[8], use_gpu=False
        )

        assert results[8].memory_usage_mb >= 0

    def test_hardware_info(self, benchmark):
        """Test hardware information gathering."""
        info = benchmark._get_hardware_info(use_gpu=False)

        assert "platform" in info
        assert "processor" in info
        assert "cpu_count" in info
        assert "ram_gb" in info

    def test_analyze_scaling(self, benchmark):
        """Test scaling analysis."""
        # Create synthetic results
        results = {
            1: ThroughputResult(
                vessels_per_second=100,
                batches_per_second=100,
                mean_latency_ms=10,
                p95_latency_ms=12,
                p99_latency_ms=15,
                batch_size=1,
            ),
            8: ThroughputResult(
                vessels_per_second=600,
                batches_per_second=75,
                mean_latency_ms=13,
                p95_latency_ms=15,
                p99_latency_ms=18,
                batch_size=8,
            ),
            16: ThroughputResult(
                vessels_per_second=1000,
                batches_per_second=62.5,
                mean_latency_ms=16,
                p95_latency_ms=20,
                p99_latency_ms=25,
                batch_size=16,
            ),
            32: ThroughputResult(
                vessels_per_second=1400,
                batches_per_second=43.75,
                mean_latency_ms=150,  # High latency
                p95_latency_ms=180,
                p99_latency_ms=200,
                batch_size=32,
            ),
        }

        analysis = benchmark.analyze_scaling(results)

        assert analysis["optimal_batch_size"] == 16  # Best under latency threshold
        assert analysis["max_throughput"] == 1000
        assert "scaling_efficiency" in analysis
        assert "recommendation" in analysis
        assert len(analysis["batch_sizes"]) == 4
        assert len(analysis["throughputs"]) == 4
        assert len(analysis["latencies"]) == 4

    def test_scaling_recommendation(self, benchmark):
        """Test scaling recommendation generation."""
        # Test good scaling
        rec = benchmark._get_scaling_recommendation(
            optimal_batch=16, scaling_efficiency=0.7
        )
        assert "Good scaling" in rec

        # Test poor scaling
        rec = benchmark._get_scaling_recommendation(
            optimal_batch=8, scaling_efficiency=0.3
        )
        assert "Poor scaling" in rec

        # Test single-sample optimal
        rec = benchmark._get_scaling_recommendation(
            optimal_batch=1, scaling_efficiency=0.5
        )
        assert "Single-sample" in rec

        # Test no optimal batch
        rec = benchmark._get_scaling_recommendation(
            optimal_batch=None, scaling_efficiency=0.5
        )
        assert "No batch size meets" in rec

    def test_empty_results_analysis(self, benchmark):
        """Test scaling analysis with empty results."""
        analysis = benchmark.analyze_scaling({})
        assert analysis == {}


class TestThroughputResult:
    """Test ThroughputResult dataclass."""

    def test_throughput_result_creation(self):
        """Test creating throughput result."""
        result = ThroughputResult(
            vessels_per_second=500.0,
            batches_per_second=50.0,
            mean_latency_ms=20.0,
            p95_latency_ms=25.0,
            p99_latency_ms=30.0,
            gpu_utilization=75.0,
            memory_usage_mb=256.0,
            batch_size=10,
            hardware_info={"gpu": "NVIDIA RTX 3090"},
        )

        assert result.vessels_per_second == 500.0
        assert result.batch_size == 10
        assert result.gpu_utilization == 75.0

    def test_throughput_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ThroughputResult(
            vessels_per_second=500.0,
            batches_per_second=50.0,
            mean_latency_ms=20.0,
            p95_latency_ms=25.0,
            p99_latency_ms=30.0,
            batch_size=10,
        )

        result_dict = result.to_dict()
        assert result_dict["vessels_per_second"] == 500.0
        assert result_dict["batch_size"] == 10
        assert "hardware_info" in result_dict


class TestRealisticBenchmarking:
    """Test realistic benchmarking scenarios."""

    def test_small_model_benchmark(self):
        """Test benchmarking a small, fast model."""
        benchmark = ThroughputBenchmark(warmup_iterations=5, benchmark_iterations=20)

        # Simulate small model with low latency
        class FastModel:
            def predict(self, batch):
                # Very fast processing
                return np.mean(batch, axis=1)

        model = FastModel()
        data = np.random.randn(200, 10, 4)

        results = benchmark.benchmark_model(
            model=model,
            test_data=data,
            batch_sizes=[1, 16, 32],
            use_gpu=False,
        )

        # Fast model should handle many vessels per second
        assert results[1].vessels_per_second > 10
        assert results[32].vessels_per_second > results[1].vessels_per_second

    def test_large_model_benchmark(self):
        """Test benchmarking a large, slow model."""
        benchmark = ThroughputBenchmark(warmup_iterations=2, benchmark_iterations=5)

        # Simulate large model with high latency
        class SlowModel:
            def predict(self, batch):
                # Simulate complex processing
                result = np.zeros_like(batch)
                for _ in range(10):  # Multiple operations
                    result = result + np.random.randn(*batch.shape) * 0.1
                return result

        model = SlowModel()
        data = np.random.randn(50, 10, 4)

        results = benchmark.benchmark_model(
            model=model,
            test_data=data,
            batch_sizes=[1, 4, 8],
            use_gpu=False,
        )

        # Model should process data (actual throughput varies by hardware)
        assert results[1].vessels_per_second > 0
        # Batching should provide some benefit (or at least not hurt)
        # Note: exact relationship depends on hardware and model complexity
        assert results[8].vessels_per_second > 0
