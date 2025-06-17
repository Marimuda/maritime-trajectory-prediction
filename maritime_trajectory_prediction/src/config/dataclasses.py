"""
Pydantic dataclasses for configuration validation.

Implements the blueprint's recommendation for structured configuration
with validation and defaults using pydantic.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ModeType(str, Enum):
    """Supported operation modes."""

    PREPROCESS = "preprocess"
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    BENCHMARK = "benchmark"


class ModelType(str, Enum):
    """Supported model types."""

    TRAISFORMER = "traisformer"
    AIS_FUSER = "ais_fuser"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    MOTION_TRANSFORMER = "motion_transformer"
    ANOMALY_TRANSFORMER = "anomaly_transformer"


class TaskType(str, Enum):
    """Supported ML tasks."""

    TRAJECTORY_PREDICTION = "trajectory_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    VESSEL_INTERACTION = "vessel_interaction"


class AcceleratorType(str, Enum):
    """Supported accelerator types."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class LoggerType(str, Enum):
    """Supported logger types."""

    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


# Mode configuration
class ModeConfig(BaseModel):
    """Configuration for operation mode."""

    mode: ModeType = ModeType.TRAIN


# Data configuration
class DataConfig(BaseModel):
    """Configuration for data loading and processing."""

    # Data paths
    file: str | None = None
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Processing parameters
    sequence_length: int = 30
    prediction_horizon: int = 10
    min_trajectory_length: int = 10
    max_gap_minutes: int = 30
    min_points: int = 6

    # Splits
    validation_split: float = 0.2
    test_split: float = 0.1

    # Batch processing
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Features
    feature_columns: list[str] = Field(
        default_factory=lambda: [
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
        ]
    )
    target_columns: list[str] = Field(
        default_factory=lambda: ["lat", "lon", "sog", "cog"]
    )

    # Spatial bounds (optional)
    spatial_bounds: dict[str, float] | None = None

    @validator("validation_split", "test_split")
    def validate_split_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Split values must be between 0 and 1")
        return v


# Model configuration
class ModelConfig(BaseModel):
    """Configuration for models."""

    type: ModelType = ModelType.TRAISFORMER
    task: TaskType = TaskType.TRAJECTORY_PREDICTION

    # Architecture parameters
    input_dim: int = 11
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Task-specific parameters
    output_dim: int = 4
    num_modes: int = 3  # For multimodal prediction
    prediction_horizon: int = 10

    # Model-specific parameters
    custom_params: dict[str, Any] = Field(default_factory=dict)

    # Checkpoint path for loading
    checkpoint_path: str | None = None

    @validator("dropout")
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout must be between 0 and 1")
        return v


# Training configuration
class TrainerConfig(BaseModel):
    """Configuration for training."""

    # Accelerator settings
    accelerator: AcceleratorType = AcceleratorType.AUTO
    devices: str | int | list[int] = "auto"
    precision: str | int = "16-mixed"

    # Training parameters
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Optimization
    optimizer: str = "adam"
    scheduler: str | None = "cosine"
    gradient_clip_val: float = 1.0

    # Performance
    compile: bool = False
    deterministic: bool = False
    benchmark: bool = True

    # Validation
    check_val_every_n_epoch: int = 1
    val_check_interval: float | None = None

    @validator("learning_rate", "weight_decay")
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Learning rate and weight decay must be positive")
        return v


# Logger configuration
class LoggerConfig(BaseModel):
    """Configuration for experiment logging."""

    type: LoggerType = LoggerType.TENSORBOARD

    # Common settings
    save_dir: str = "logs"
    name: str = "maritime_trajectory"
    version: str | None = None

    # TensorBoard specific
    tb_log_graph: bool = False
    tb_default_hp_metric: bool = False

    # Wandb specific
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = Field(default_factory=list)


# Callbacks configuration
class CallbacksConfig(BaseModel):
    """Configuration for training callbacks."""

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor: str = "val/loss"
    mode: str = "min"

    # Model checkpointing
    save_top_k: int = 3
    save_last: bool = True
    checkpoint_dir: str = "checkpoints"

    # Learning rate monitoring
    lr_logging_interval: str = "epoch"

    # Rich progress bar
    rich_progress: bool = True

    # Device stats monitoring
    device_stats: bool = True


# Experiment configuration
class ExperimentConfig(BaseModel):
    """Configuration for experiments and hyperparameter sweeps."""

    name: str = "base"

    # Sweep configuration
    sweep_enabled: bool = False
    num_trials: int = 20

    # Search space (simplified representation)
    search_space: dict[str, Any] = Field(default_factory=dict)

    # Pruning
    pruning_enabled: bool = True

    # Study storage
    study_name: str | None = None
    storage_url: str | None = None


# Prediction configuration
class PredictConfig(BaseModel):
    """Configuration for prediction/inference."""

    # Input/output
    input_file: str | None = None
    output_file: str | None = None

    # Prediction parameters
    num_samples: int = 10
    batch_size: int = 32

    # Output format
    format: str = "csv"  # csv, json, npz

    # Visualization
    visualize: bool = False
    max_trajectories: int = 10


# Preprocessing configuration
class PreprocessConfig(BaseModel):
    """Configuration for data preprocessing."""

    # Input/output directories
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"

    # Processing parameters
    batch_size: int = 1000
    sequence_length: int = 30
    prediction_horizon: int = 12
    min_trajectory_length: int = 50
    max_trajectory_length: int = 1000

    # Feature engineering
    feature_engineering: bool = True
    normalize_features: bool = True
    save_statistics: bool = True

    # Parallel processing
    parallel_processing: bool = True
    num_workers: int = 4


# Evaluation configuration
class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    # Data settings
    data_path: str | None = None
    output_dir: str = "outputs/evaluation"

    # Evaluation metrics
    metrics: list[str] = Field(
        default_factory=lambda: ["trajectory_prediction", "anomaly_detection"]
    )

    # Output settings
    save_predictions: bool = True
    save_visualizations: bool = True
    generate_report: bool = True

    # Processing parameters
    batch_size: int = 32
    num_samples: int = 50
    confidence_threshold: float = 0.5


# Benchmark configuration
class BenchmarkConfig(BaseModel):
    """Configuration for model benchmarking."""

    # Models to benchmark
    models: list[str] | None = None

    # Benchmark metrics
    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "speed", "memory_usage"]
    )

    # Output settings
    output_dir: str = "outputs/benchmarks"
    generate_plots: bool = True
    save_detailed_results: bool = True

    # Benchmark parameters
    iterations: int = 10
    warmup_iterations: int = 3
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 8, 16, 32])
    sequence_lengths: list[int] = Field(default_factory=lambda: [10, 20, 30, 50])


# Root configuration
class RootConfig(BaseModel):
    """Root configuration combining all components."""

    # Core components
    mode: ModeType = ModeType.TRAIN
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    # Mode-specific components
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # Global settings
    seed: int = 42
    verbose: bool = True
    debug: bool = False

    # Hydra settings
    hydra_run_dir: str = "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    hydra_sweep_dir: str = "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Prevent additional fields
        validate_assignment = True
        use_enum_values = True
