"""
Integration tests for unified CLI system following CLAUDE blueprint.

Tests that all major use-cases are accessible through the main.py entry point
and that the refactor maintains backward compatibility.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Test configuration imports
def test_config_structure():
    """Test that configuration structure is properly set up."""
    from config.dataclasses import ModeType, RootConfig

    # Check all required modes exist
    expected_modes = ["preprocess", "train", "evaluate", "predict", "benchmark"]
    available_modes = [mode.value for mode in ModeType]

    for mode in expected_modes:
        assert mode in available_modes, f"Mode {mode} not found in {available_modes}"

    # Test RootConfig instantiation
    sample_config = {
        "mode": "train",
        "model": {"name": "test"},
        "trainer": {"max_epochs": 10},
        "data": {"batch_size": 32},
    }

    # Should not raise exception
    root_config = RootConfig(**sample_config)
    assert root_config.mode == "train"


def test_dispatch_registry():
    """Test that dispatch registry maps all modes to correct functions."""
    from config.dataclasses import ModeType

    # Expected mapping from main.py
    expected_registry = {
        ModeType.PREPROCESS: "src.data.preprocess.run_preprocess",
        ModeType.TRAIN: "src.experiments.train.run_training",
        ModeType.EVALUATE: "src.experiments.evaluation.run_evaluation",
        ModeType.PREDICT: "src.inference.predictor.run_prediction",
        ModeType.BENCHMARK: "src.experiments.benchmark.run_benchmarking",
    }

    # Verify all modes have corresponding functions
    for _mode, function_path in expected_registry.items():
        module_path, function_name = function_path.rsplit(".", 1)

        # Import module and check function exists
        import importlib

        try:
            module = importlib.import_module(module_path)
            assert hasattr(
                module, function_name
            ), f"Function {function_name} not found in {module_path}"
        except ImportError as e:
            pytest.fail(f"Cannot import {module_path}: {e}")


def test_entry_point_signatures():
    """Test that all entry point functions have correct signatures."""
    entry_points = [
        "src.data.preprocess.run_preprocess",
        "src.experiments.train.run_training",
        "src.experiments.evaluation.run_evaluation",
        "src.inference.predictor.run_prediction",
        "src.experiments.benchmark.run_benchmarking",
    ]

    import importlib
    import inspect

    for function_path in entry_points:
        module_path, function_name = function_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)

            # Check signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Should accept cfg parameter
            assert "cfg" in params, f"Function {function_path} missing 'cfg' parameter"

            # Should return dict or Any
            return_annotation = sig.return_annotation
            assert (
                return_annotation != inspect.Signature.empty
            ), f"Function {function_path} missing return annotation"

        except ImportError as e:
            pytest.fail(f"Cannot import {function_path}: {e}")


def test_config_files_exist():
    """Test that all required configuration files exist."""
    config_dir = Path(__file__).parent.parent.parent / "configs"

    # Main config
    assert (config_dir / "config.yaml").exists(), "Main config.yaml missing"

    # Mode configs
    mode_dir = config_dir / "mode"
    assert mode_dir.exists(), "Mode directory missing"

    required_modes = [
        "train.yaml",
        "evaluate.yaml",
        "predict.yaml",
        "preprocess.yaml",
        "benchmark.yaml",
    ]
    for mode_file in required_modes:
        assert (mode_dir / mode_file).exists(), f"Mode config {mode_file} missing"

    # Data configs
    data_dir = config_dir / "data"
    assert data_dir.exists(), "Data directory missing"
    assert (data_dir / "ais_processed.yaml").exists(), "AIS processed config missing"

    # Model configs
    model_dir = config_dir / "model"
    assert model_dir.exists(), "Model directory missing"

    # Trainer configs
    trainer_dir = config_dir / "trainer"
    assert trainer_dir.exists(), "Trainer directory missing"
    assert (trainer_dir / "gpu.yaml").exists(), "GPU trainer config missing"
    assert (trainer_dir / "cpu.yaml").exists(), "CPU trainer config missing"


def test_deprecated_scripts_removed():
    """Test that deprecated scripts have been removed per CLAUDE blueprint."""
    project_root = Path(__file__).parent.parent.parent

    deprecated_scripts = [
        "predict_trajectory.py",
        "inference_transformer_models.py",
        "train_simple_model.py",
        "evaluate_transformer_models.py",
        "test_benchmark_models.py",
        "scripts/predict_trajectory.py",
    ]

    for script in deprecated_scripts:
        script_path = project_root / script
        assert not script_path.exists(), f"Deprecated script {script} still exists - should be removed per CLAUDE blueprint"


def test_unified_cli_structure():
    """Test that unified CLI structure matches CLAUDE blueprint."""
    project_root = Path(__file__).parent.parent.parent

    # Main entry point exists
    assert (project_root / "main.py").exists(), "main.py entry point missing"

    # Source structure exists
    src_dir = project_root / "src"
    assert src_dir.exists(), "src directory missing"

    required_modules = [
        "config/__init__.py",
        "config/dataclasses.py",
        "config/store.py",
        "data/preprocess.py",
        "models/factory.py",
        "training/trainer.py",
        "inference/predictor.py",
        "experiments/train.py",
        "experiments/evaluation.py",
        "experiments/benchmark.py",
    ]

    for module_path in required_modules:
        full_path = src_dir / module_path
        assert full_path.exists(), f"Required module {module_path} missing"


@pytest.mark.integration
def test_preprocess_mode_accessible():
    """Test that preprocess mode is accessible."""
    try:
        from src.data.preprocess import run_preprocess

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.data.get.return_value = "test_value"
        mock_cfg.data.__contains__ = lambda self, key: True

        # Should be callable (may fail due to missing data, but should be importable)
        assert callable(run_preprocess), "run_preprocess not callable"

    except ImportError as e:
        pytest.fail(f"Cannot import preprocess module: {e}")


@pytest.mark.integration
def test_train_mode_accessible():
    """Test that train mode is accessible."""
    try:
        from src.experiments.train import run_training

        # Should be callable
        assert callable(run_training), "run_training not callable"

    except ImportError as e:
        pytest.fail(f"Cannot import train module: {e}")


@pytest.mark.integration
def test_predict_mode_accessible():
    """Test that predict mode is accessible."""
    try:
        from src.inference.predictor import run_prediction

        # Should be callable
        assert callable(run_prediction), "run_prediction not callable"

    except ImportError as e:
        pytest.fail(f"Cannot import prediction module: {e}")


@pytest.mark.integration
def test_evaluate_mode_accessible():
    """Test that evaluate mode is accessible."""
    try:
        from src.experiments.evaluation import run_evaluation

        # Should be callable
        assert callable(run_evaluation), "run_evaluation not callable"

    except ImportError as e:
        pytest.fail(f"Cannot import evaluation module: {e}")


@pytest.mark.integration
def test_benchmark_mode_accessible():
    """Test that benchmark mode is accessible."""
    try:
        from src.experiments.benchmark import run_benchmarking

        # Should be callable
        assert callable(run_benchmarking), "run_benchmarking not callable"

    except ImportError as e:
        pytest.fail(f"Cannot import benchmark module: {e}")


def test_claude_blueprint_compliance():
    """Test that implementation follows CLAUDE blueprint requirements."""
    project_root = Path(__file__).parent.parent.parent

    # 1. Single entry point
    assert (project_root / "main.py").exists(), "Single main.py entry point missing"

    # 2. Hydra configuration
    assert (project_root / "configs" / "config.yaml").exists(), "Hydra config missing"

    # 3. Modular structure
    src_dir = project_root / "src"
    required_dirs = ["config", "data", "models", "training", "inference", "experiments"]
    for dir_name in required_dirs:
        assert (src_dir / dir_name).exists(), f"Required directory {dir_name} missing"

    # 4. Deprecated scripts removed
    deprecated_scripts = [
        "predict_trajectory.py",
        "inference_transformer_models.py",
        "train_simple_model.py",
        "evaluate_transformer_models.py",
        "test_benchmark_models.py",
    ]
    for script in deprecated_scripts:
        assert not (
            project_root / script
        ).exists(), f"Deprecated script {script} not removed"

    # 5. Configuration groups
    config_dir = project_root / "configs"
    required_groups = ["mode", "data", "model", "trainer", "logger", "callbacks"]
    for group in required_groups:
        assert (config_dir / group).exists(), f"Configuration group {group} missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
