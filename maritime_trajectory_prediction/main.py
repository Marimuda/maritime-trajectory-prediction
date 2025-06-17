#!/usr/bin/env python3
"""
Maritime Trajectory Prediction - Unified CLI Entry Point

This is the single entry point for all maritime trajectory prediction tasks
following the CLAUDE blueprint architecture. It uses Hydra for configuration
management and dispatches to specialized modules based on the mode.

Usage Examples:
    # Training
    python main.py mode=train model=traisformer data=ais_processed

    # Hyperparameter sweep
    python main.py experiment=traisformer_sweep --multirun

    # Real-time inference
    python main.py mode=predict model.checkpoint_path=outputs/best.pt \
                   predict.input_file=data.parquet predict.output_file=results.csv

    # Preprocessing
    python main.py mode=preprocess data.file=data/raw/ais.log \
                   data.processed_dir=data/processed

    # Evaluation
    python main.py mode=evaluate model.checkpoint_path=outputs/best.pt \
                   data=ais_processed

    # Benchmarking
    python main.py mode=benchmark
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import config registration
from config import register_configs  # noqa: E402
from config.dataclasses import ModeType, RootConfig  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig):
    """
    Setup environment for optimal performance and reproducibility.

    Args:
        cfg: Hydra configuration
    """
    # Set random seeds
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set random seed to {cfg.seed}")

    # Configure PyTorch performance optimizations
    if cfg.trainer.get("compile", False):
        logger.info("PyTorch compilation enabled")

    if cfg.trainer.get("benchmark", True):
        import torch

        torch.backends.cudnn.benchmark = True
        logger.info("CUDNN benchmarking enabled")

    # Setup working directory
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.runtime.output_dir
    logger.info(f"Working directory: {work_dir}")

    # Create necessary directories
    for dir_name in ["checkpoints", "logs", "outputs"]:
        dir_path = Path(work_dir) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: DictConfig) -> bool:
    """
    Validate configuration before execution.

    Args:
        cfg: Hydra configuration

    Returns:
        True if configuration is valid
    """
    try:
        # Validate using pydantic model
        RootConfig(**cfg)
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def dispatch_mode(cfg: DictConfig) -> Any:
    """
    Dispatch to the appropriate module based on mode.

    Args:
        cfg: Hydra configuration

    Returns:
        Result from the dispatched function
    """
    mode = cfg.mode

    # Define the registry of mode handlers
    registry = {
        ModeType.PREPROCESS: "src.data.preprocess.run_preprocess",
        ModeType.TRAIN: "src.experiments.train.run_training",
        ModeType.EVALUATE: "src.experiments.evaluation.run_evaluation",
        ModeType.PREDICT: "src.inference.predictor.run_prediction",
        ModeType.BENCHMARK: "src.experiments.benchmark.run_benchmarking",
    }

    if mode not in registry:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes: {list(registry.keys())}"
        )

    function_path = registry[mode]
    logger.info(f"Dispatching to {function_path} for mode '{mode}'")

    try:
        # Use Hydra's call utility to invoke the function
        result = call(function_path, cfg)
        logger.info(f"Mode '{mode}' completed successfully")
        return result
    except Exception as e:
        logger.error(f"Mode '{mode}' failed with error: {e}")
        raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """
    Main entry point for the maritime trajectory prediction system.

    Args:
        cfg: Hydra configuration object

    Returns:
        Optional metric value for hyperparameter optimization
    """
    try:
        # Print configuration in debug mode
        if cfg.get("debug", False):
            logger.info("Configuration:")
            logger.info(OmegaConf.to_yaml(cfg))

        # Validate configuration
        if not validate_config(cfg):
            return None

        # Setup environment
        setup_environment(cfg)

        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")

        # Import and log PyTorch information
        try:
            import torch

            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            logger.warning("PyTorch not available")

        # Dispatch to appropriate mode handler
        result = dispatch_mode(cfg)

        # Return metric for hyperparameter optimization
        if isinstance(result, dict) and "metric" in result:
            return result["metric"]

        return result

    except Exception as e:
        logger.error(f"Application failed with error: {e}")
        if cfg.get("debug", False):
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    # Register configurations with Hydra
    register_configs()

    # Run main application
    main()
