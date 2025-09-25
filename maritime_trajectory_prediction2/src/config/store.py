"""
Hydra ConfigStore registration.

Registers all configuration dataclasses with Hydra's ConfigStore
for structured configuration and validation.
"""

from hydra.core.config_store import ConfigStore

from .dataclasses import (
    AcceleratorType,
    BenchmarkConfig,
    CallbacksConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoggerConfig,
    LoggerType,
    ModelConfig,
    ModelType,
    ModeType,
    PredictConfig,
    PreprocessConfig,
    RootConfig,
    TaskType,
    TrainerConfig,
)


def register_configs():
    """Register all configuration schemas with Hydra ConfigStore."""
    cs = ConfigStore.instance()

    # Register root config
    cs.store(name="config", node=RootConfig)

    # Register mode configs
    cs.store(group="mode", name="train", node={"mode": ModeType.TRAIN})
    cs.store(group="mode", name="preprocess", node={"mode": ModeType.PREPROCESS})
    cs.store(group="mode", name="evaluate", node={"mode": ModeType.EVALUATE})
    cs.store(group="mode", name="predict", node={"mode": ModeType.PREDICT})
    cs.store(group="mode", name="benchmark", node={"mode": ModeType.BENCHMARK})

    # Register data configs
    cs.store(
        group="data",
        name="ais_processed",
        node=DataConfig(
            file="data/processed/combined_ais_all.parquet",
            sequence_length=30,
            prediction_horizon=10,
            batch_size=32,
            feature_columns=[
                "lat",
                "lon",
                "sog",
                "cog",
                "heading",
                "nav_status",
                "vessel_type",
                "length",
                "width",
                "draught",
                "rot",
            ],
            target_columns=["lat", "lon", "sog", "cog"],
        ),
    )

    cs.store(
        group="data",
        name="ais_small",
        node=DataConfig(
            file="data/processed/test_ais_all.parquet",
            sequence_length=10,
            prediction_horizon=5,
            batch_size=16,
            feature_columns=[
                "lat",
                "lon",
                "sog",
                "cog",
                "heading",
                "nav_status",
                "vessel_type",
                "length",
                "width",
                "draught",
                "rot",
            ],
            target_columns=["lat", "lon", "sog", "cog"],
        ),
    )

    # Register model configs
    cs.store(
        group="model",
        name="traisformer",
        node=ModelConfig(
            type=ModelType.TRAISFORMER,
            task=TaskType.TRAJECTORY_PREDICTION,
            input_dim=11,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            output_dim=4,
            num_modes=3,
        ),
    )

    cs.store(
        group="model",
        name="ais_fuser",
        node=ModelConfig(
            type=ModelType.AIS_FUSER,
            task=TaskType.TRAJECTORY_PREDICTION,
            input_dim=11,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            output_dim=4,
            num_modes=1,
        ),
    )

    cs.store(
        group="model",
        name="lstm",
        node=ModelConfig(
            type=ModelType.LSTM,
            task=TaskType.TRAJECTORY_PREDICTION,
            input_dim=11,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
            output_dim=4,
        ),
    )

    cs.store(
        group="model",
        name="xgboost",
        node=ModelConfig(
            type=ModelType.XGBOOST,
            task=TaskType.TRAJECTORY_PREDICTION,
            input_dim=11,
            output_dim=4,
        ),
    )

    cs.store(
        group="model",
        name="motion_transformer",
        node=ModelConfig(
            type=ModelType.MOTION_TRANSFORMER,
            task=TaskType.TRAJECTORY_PREDICTION,
            input_dim=11,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            output_dim=4,
            num_modes=5,
        ),
    )

    cs.store(
        group="model",
        name="anomaly_transformer",
        node=ModelConfig(
            type=ModelType.ANOMALY_TRANSFORMER,
            task=TaskType.ANOMALY_DETECTION,
            input_dim=11,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            output_dim=1,
        ),
    )

    # Register trainer configs
    cs.store(
        group="trainer",
        name="cpu",
        node=TrainerConfig(
            accelerator=AcceleratorType.CPU,
            devices=1,
            precision=32,
            max_epochs=50,
            learning_rate=1e-3,
            compile=False,
        ),
    )

    cs.store(
        group="trainer",
        name="gpu",
        node=TrainerConfig(
            accelerator=AcceleratorType.GPU,
            devices=1,
            precision="16-mixed",
            max_epochs=100,
            learning_rate=1e-3,
            compile=True,
            benchmark=True,
        ),
    )

    cs.store(
        group="trainer",
        name="multi_gpu",
        node=TrainerConfig(
            accelerator=AcceleratorType.GPU,
            devices=-1,  # Use all available GPUs
            precision="16-mixed",
            max_epochs=100,
            learning_rate=1e-3,
            compile=True,
            benchmark=True,
        ),
    )

    # Register logger configs
    cs.store(
        group="logger",
        name="tensorboard",
        node=LoggerConfig(
            type=LoggerType.TENSORBOARD,
            save_dir="logs",
            name="maritime_trajectory",
            tb_log_graph=False,
            tb_default_hp_metric=False,
        ),
    )

    cs.store(
        group="logger",
        name="wandb",
        node=LoggerConfig(
            type=LoggerType.WANDB,
            save_dir="logs",
            name="maritime_trajectory",
            wandb_project="maritime-trajectory-prediction",
            wandb_tags=["trajectory", "maritime", "ais"],
        ),
    )

    # Register callback configs
    cs.store(
        group="callbacks",
        name="default",
        node=CallbacksConfig(
            early_stopping=True,
            patience=10,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            checkpoint_dir="checkpoints",
            lr_logging_interval="epoch",
            rich_progress=True,
            device_stats=True,
        ),
    )

    cs.store(
        group="callbacks",
        name="minimal",
        node=CallbacksConfig(
            early_stopping=False,
            patience=5,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            checkpoint_dir="checkpoints",
            lr_logging_interval="step",
            rich_progress=False,
            device_stats=False,
        ),
    )

    # Register experiment configs
    cs.store(
        group="experiment",
        name="base",
        node=ExperimentConfig(
            name="base",
            sweep_enabled=False,
            num_trials=20,
            search_space={},
            pruning_enabled=True,
        ),
    )

    cs.store(
        group="experiment",
        name="traisformer_sweep",
        node=ExperimentConfig(
            name="traisformer_sweep",
            sweep_enabled=True,
            num_trials=50,
            search_space={
                "model.hidden_dim": [128, 256, 512],
                "model.num_layers": [4, 6, 8],
                "model.num_heads": [4, 8, 16],
                "model.dropout": [0.1, 0.2, 0.3],
                "trainer.learning_rate": [1e-4, 1e-3, 1e-2],
            },
            pruning_enabled=True,
        ),
    )

    cs.store(
        group="experiment",
        name="ais_fuser_sweep",
        node=ExperimentConfig(
            name="ais_fuser_sweep",
            sweep_enabled=True,
            num_trials=30,
            search_space={
                "model.hidden_dim": [64, 128, 256],
                "model.num_layers": [2, 4, 6],
                "model.num_heads": [2, 4, 8],
                "trainer.learning_rate": [1e-4, 1e-3, 1e-2],
            },
            pruning_enabled=True,
        ),
    )

    # Register prediction configs
    cs.store(
        group="predict",
        name="default",
        node=PredictConfig(
            num_samples=10,
            batch_size=32,
            format="csv",
            visualize=False,
            max_trajectories=10,
        ),
    )

    cs.store(
        group="predict",
        name="batch",
        node=PredictConfig(
            num_samples=1, batch_size=128, format="parquet", visualize=False
        ),
    )

    cs.store(
        group="predict",
        name="interactive",
        node=PredictConfig(
            num_samples=10,
            batch_size=16,
            format="json",
            visualize=True,
            max_trajectories=5,
        ),
    )

    # Register preprocess configs
    cs.store(
        group="preprocess",
        name="default",
        node=PreprocessConfig(
            input_dir="data/raw",
            output_dir="data/processed",
            batch_size=1000,
            sequence_length=30,
            prediction_horizon=12,
            min_trajectory_length=50,
            feature_engineering=True,
            normalize_features=True,
            parallel_processing=True,
            num_workers=4,
        ),
    )

    cs.store(
        group="preprocess",
        name="fast",
        node=PreprocessConfig(
            input_dir="data/raw",
            output_dir="data/processed",
            batch_size=5000,
            sequence_length=20,
            prediction_horizon=10,
            min_trajectory_length=30,
            feature_engineering=False,
            normalize_features=False,
            parallel_processing=True,
            num_workers=8,
        ),
    )

    # Register evaluation configs
    cs.store(
        group="evaluation",
        name="default",
        node=EvaluationConfig(
            data_path=None,
            output_dir="outputs/evaluation",
            metrics=["trajectory_prediction", "anomaly_detection"],
            save_predictions=True,
            save_visualizations=True,
            generate_report=True,
            batch_size=32,
            num_samples=50,
            confidence_threshold=0.5,
        ),
    )

    cs.store(
        group="evaluation",
        name="comprehensive",
        node=EvaluationConfig(
            data_path=None,
            output_dir="outputs/evaluation",
            metrics=[
                "trajectory_prediction",
                "anomaly_detection",
                "computational_performance",
            ],
            save_predictions=True,
            save_visualizations=True,
            generate_report=True,
            batch_size=16,
            num_samples=100,
            confidence_threshold=0.5,
        ),
    )

    # Register benchmark configs
    cs.store(
        group="benchmark",
        name="default",
        node=BenchmarkConfig(
            models=None,
            metrics=["accuracy", "speed", "memory_usage"],
            output_dir="outputs/benchmarks",
            generate_plots=True,
            save_detailed_results=True,
            iterations=10,
            warmup_iterations=3,
            batch_sizes=[1, 4, 8, 16, 32],
            sequence_lengths=[10, 20, 30, 50],
        ),
    )

    cs.store(
        group="benchmark",
        name="performance",
        node=BenchmarkConfig(
            models=None,
            metrics=["speed", "memory_usage", "throughput"],
            output_dir="outputs/benchmarks",
            generate_plots=True,
            save_detailed_results=True,
            iterations=20,
            warmup_iterations=5,
            batch_sizes=[1, 8, 32, 128],
            sequence_lengths=[20, 50, 100],
        ),
    )
