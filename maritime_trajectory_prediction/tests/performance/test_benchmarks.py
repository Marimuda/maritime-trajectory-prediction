"""
Performance benchmarks for maritime trajectory prediction components.

These tests measure performance characteristics and detect regressions
in critical path operations.
"""
import pytest
import time
import numpy as np
import pandas as pd
from memory_profiler import profile
import psutil
import os

from maritime_trajectory_prediction.src.utils.ais_parser import AISParser
from maritime_trajectory_prediction.src.utils.maritime_utils import MaritimeUtils
from maritime_trajectory_prediction.src.data.ais_processor import AISProcessor


class TestDataLoadingPerformance:
    """Benchmark data loading operations."""
    
    @pytest.mark.perf
    @pytest.mark.slow
    def test_large_csv_loading_performance(self, tmp_path, ais_data_factory, benchmark):
        """Benchmark loading large CSV files."""
        # Arrange
        large_dataset = ais_data_factory(
            n_vessels=1000,
            n_points_per_vessel=1000,
            seed=42
        )
        data_file = tmp_path / "large_dataset.csv"
        large_dataset.to_csv(data_file, index=False)
        
        parser = AISParser()
        
        # Act & Assert
        result = benchmark(parser.load_processed_ais_data, data_file)
        
        assert len(result) == len(large_dataset)
        # Benchmark will automatically track timing
    
    @pytest.mark.perf
    def test_trajectory_extraction_performance(self, benchmark_data, benchmark):
        """Benchmark trajectory extraction from large datasets."""
        # Arrange
        parser = AISParser()
        
        # Act & Assert
        trajectories = benchmark(parser.get_vessel_trajectories, benchmark_data)
        
        assert len(trajectories) > 0
        # Should complete within reasonable time (tracked by benchmark)


class TestDataProcessingPerformance:
    """Benchmark data processing operations."""
    
    @pytest.mark.perf
    def test_data_cleaning_performance(self, benchmark_data, benchmark):
        """Benchmark data cleaning operations."""
        # Arrange
        processor = AISProcessor()
        
        # Act & Assert
        cleaned_data = benchmark(processor.clean_ais_data, benchmark_data)
        
        assert len(cleaned_data) <= len(benchmark_data)
    
    @pytest.mark.perf
    def test_sequence_generation_performance(self, ais_data_factory, benchmark):
        """Benchmark sequence generation for model training."""
        # Arrange
        large_trajectory = ais_data_factory(
            n_vessels=1,
            n_points_per_vessel=10000,
            seed=42
        )
        processor = AISProcessor()
        
        # Act & Assert
        sequences = benchmark(
            processor.get_trajectory_sequences,
            large_trajectory,
            sequence_length=50,
            prediction_horizon=10
        )
        
        assert len(sequences) > 0


class TestMathematicalOperationsPerformance:
    """Benchmark mathematical computations."""
    
    @pytest.mark.perf
    def test_haversine_distance_vectorized_performance(self, benchmark):
        """Benchmark vectorized haversine distance calculations."""
        # Arrange
        n_points = 100000
        lats1 = np.random.uniform(-90, 90, n_points)
        lons1 = np.random.uniform(-180, 180, n_points)
        lats2 = np.random.uniform(-90, 90, n_points)
        lons2 = np.random.uniform(-180, 180, n_points)
        
        utils = MaritimeUtils()
        
        def calculate_distances():
            distances = []
            for i in range(len(lats1)):
                dist = utils.haversine_distance(lats1[i], lons1[i], lats2[i], lons2[i])
                distances.append(dist)
            return distances
        
        # Act & Assert
        distances = benchmark(calculate_distances)
        
        assert len(distances) == n_points
        assert all(d >= 0 for d in distances)
    
    @pytest.mark.perf
    def test_trajectory_distance_calculation_performance(self, benchmark):
        """Benchmark trajectory distance calculations."""
        # Arrange
        n_points = 50000
        lats = np.random.uniform(58, 61, n_points)  # Norway region
        lons = np.random.uniform(8, 12, n_points)
        
        utils = MaritimeUtils()
        
        # Act & Assert
        distances = benchmark(utils.calculate_distances, lats, lons)
        
        assert len(distances) == n_points
        assert distances[0] == 0.0


class TestMemoryUsagePerformance:
    """Test memory usage characteristics."""
    
    @pytest.mark.perf
    @pytest.mark.slow
    def test_memory_usage_large_dataset(self, ais_data_factory):
        """Test memory usage with large datasets."""
        # Arrange
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Act
        large_dataset = ais_data_factory(
            n_vessels=500,
            n_points_per_vessel=2000,
            seed=42
        )
        
        processor = AISProcessor()
        cleaned_data = processor.clean_ais_data(large_dataset)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del large_dataset, cleaned_data
        
        # Assert
        assert memory_increase < 1000  # Should not use more than 1GB additional memory
    
    @pytest.mark.perf
    def test_memory_efficiency_sequence_generation(self, ais_data_factory):
        """Test memory efficiency of sequence generation."""
        # Arrange
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trajectory_data = ais_data_factory(
            n_vessels=10,
            n_points_per_vessel=1000,
            seed=42
        )
        
        processor = AISProcessor()
        
        # Act
        sequences = processor.get_trajectory_sequences(
            trajectory_data,
            sequence_length=100,
            prediction_horizon=20
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Assert
        assert len(sequences) > 0
        assert memory_increase < 500  # Should not use more than 500MB additional memory


class TestScalabilityPerformance:
    """Test performance scaling characteristics."""
    
    @pytest.mark.perf
    @pytest.mark.slow
    def test_processing_time_scaling(self, ais_data_factory):
        """Test how processing time scales with data size."""
        # Arrange
        data_sizes = [100, 500, 1000, 2000]
        processing_times = []
        
        processor = AISProcessor()
        
        for size in data_sizes:
            # Act
            test_data = ais_data_factory(
                n_vessels=10,
                n_points_per_vessel=size,
                seed=42
            )
            
            start_time = time.time()
            cleaned_data = processor.clean_ais_data(test_data)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Clean up
            del test_data, cleaned_data
        
        # Assert
        # Processing time should scale reasonably (not exponentially)
        assert all(t > 0 for t in processing_times)
        
        # Check that scaling is reasonable (not worse than quadratic)
        time_ratios = [processing_times[i+1] / processing_times[i] for i in range(len(processing_times)-1)]
        size_ratios = [data_sizes[i+1] / data_sizes[i] for i in range(len(data_sizes)-1)]
        
        # Time scaling should not be much worse than linear
        for time_ratio, size_ratio in zip(time_ratios, size_ratios):
            assert time_ratio <= size_ratio * 2  # Allow up to 2x linear scaling
    
    @pytest.mark.perf
    def test_concurrent_processing_performance(self, ais_data_factory):
        """Test performance under concurrent processing scenarios."""
        import threading
        import queue
        
        # Arrange
        test_data = ais_data_factory(n_vessels=20, n_points_per_vessel=100, seed=42)
        results_queue = queue.Queue()
        
        def process_data():
            processor = AISProcessor()
            start_time = time.time()
            cleaned_data = processor.clean_ais_data(test_data.copy())
            processing_time = time.time() - start_time
            results_queue.put(processing_time)
        
        # Act - Run multiple threads
        threads = []
        n_threads = 4
        
        start_time = time.time()
        for _ in range(n_threads):
            thread = threading.Thread(target=process_data)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        processing_times = []
        while not results_queue.empty():
            processing_times.append(results_queue.get())
        
        # Assert
        assert len(processing_times) == n_threads
        assert all(t > 0 for t in processing_times)
        
        # Concurrent processing should not be much slower than sequential
        avg_concurrent_time = np.mean(processing_times)
        
        # Sequential baseline
        processor = AISProcessor()
        sequential_start = time.time()
        processor.clean_ais_data(test_data.copy())
        sequential_time = time.time() - sequential_start
        
        # Concurrent should not be more than 3x slower per thread
        assert avg_concurrent_time <= sequential_time * 3


class TestRegressionBenchmarks:
    """Benchmark tests to detect performance regressions."""
    
    @pytest.mark.perf
    def test_baseline_processing_performance(self, ais_data_factory, benchmark):
        """Establish baseline performance metrics."""
        # Arrange
        baseline_data = ais_data_factory(
            n_vessels=50,
            n_points_per_vessel=200,
            seed=42
        )
        
        processor = AISProcessor()
        
        # Act & Assert
        result = benchmark(processor.clean_ais_data, baseline_data)
        
        assert len(result) <= len(baseline_data)
        
        # Benchmark framework will track timing and detect regressions
        # when run with --benchmark-compare or --benchmark-histogram
    
    @pytest.mark.perf
    def test_baseline_trajectory_extraction_performance(self, ais_data_factory, benchmark):
        """Establish baseline for trajectory extraction performance."""
        # Arrange
        baseline_data = ais_data_factory(
            n_vessels=20,
            n_points_per_vessel=100,
            seed=42
        )
        
        parser = AISParser()
        
        # Act & Assert
        trajectories = benchmark(parser.get_vessel_trajectories, baseline_data)
        
        assert len(trajectories) > 0
    
    @pytest.mark.perf
    def test_baseline_mathematical_operations_performance(self, benchmark):
        """Establish baseline for mathematical operations."""
        # Arrange
        n_points = 10000
        lats = np.random.uniform(58, 61, n_points)
        lons = np.random.uniform(8, 12, n_points)
        
        utils = MaritimeUtils()
        
        # Act & Assert
        distances = benchmark(utils.calculate_distances, lats, lons)
        
        assert len(distances) == n_points

