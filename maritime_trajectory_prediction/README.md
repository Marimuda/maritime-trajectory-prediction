# Maritime Trajectory Prediction

A comprehensive PyTorch Lightning framework for vessel trajectory prediction using AIS (Automatic Identification System) data. This implementation includes state-of-the-art models for maritime trajectory forecasting, incorporating both transformer-based and graph neural network approaches.

## Models

### TrAISformer

A transformer-based model for vessel trajectory prediction based on the paper [TrAISformer](https://github.com/CIA-Oceanix/TrAISformer). Key features:

- Discrete four-hot encoding for maritime data
- Causal self-attention mechanism
- Stochastic trajectory sampling
- Best-of-N evaluation methodology

### AISFuser

A graph-enhanced model combining GNN and transformer architectures. Key features:

- Maritime graph network with PyTorch Geometric
- Spatial gated block with cosine attention
- SSL weather fusion with multi-task learning
- Hybrid architecture combining GNN and transformer components

### Baseline Models

- **LSTM**: Sequence-to-sequence prediction with LSTM networks
- **XGBoost**: Traditional ML approach for trajectory prediction

## Features

- **PyTorch Lightning** integration for scalable, organized training
- **PyTorch Geometric** for maritime graph networks
- **Hydra** configuration system for flexible experiment management
- **Weights & Biases** and **TensorBoard** for experiment tracking
- **Hyperparameter optimization** with Optuna
- **Maritime-specific metrics** including Haversine distance calculations
- **Visualization tools** for trajectory analysis and comparison

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/maritime-trajectory-prediction.git
cd maritime-trajectory-prediction

# Install dependencies
pip install -r requirements.txt

# Set up environment
python scripts/setup_environment.py
```

## Project Structure

```
maritime_trajectory_prediction/
├── configs/              # Hydra configuration files
│   ├── callbacks/        # Callback configurations
│   ├── data/             # Data configurations
│   ├── experiment/       # Experiment configurations
│   ├── hparams_search/   # Hyperparameter search configurations
│   ├── logger/           # Logger configurations
│   ├── model/            # Model configurations
│   └── trainer/          # Trainer configurations
├── src/
│   ├── data/             # Data processing modules
│   │   ├── ais_processor.py    # AIS data processing
│   │   ├── datamodule.py       # Lightning data modules
│   │   └── graph_processor.py  # Maritime graph creation
│   ├── experiments/      # Training and evaluation scripts
│   │   ├── evaluation.py       # Model evaluation
│   │   ├── sweep_runner.py     # Hyperparameter optimization
│   │   └── train.py            # Training script
│   ├── models/           # Model implementations
│   │   ├── ais_fuser.py        # AISFuser model
│   │   ├── baselines.py        # LSTM and XGBoost models
│   │   ├── factory.py          # Model factory
│   │   └── traisformer.py      # TrAISformer model
│   └── utils/            # Utility functions
│       ├── maritime_utils.py   # Maritime-specific utilities
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Visualization tools
├── scripts/              # Utility scripts
└── README.md
```

## Usage

### Data Preparation

Before training, prepare your AIS data:

```python
from src.data.ais_processor import AISDataProcessor

# Configure processor
config = {
    'min_seq_len': 6,
    'max_speed': 50,
    'input_seq_len': 10,
    'target_seq_len': 6
}

# Process data
processor = AISDataProcessor(config)
processed_data = processor.process(raw_ais_data)
```

### Training a Model

```bash
# Train TrAISformer with default configuration
python src/experiments/train.py model=traisformer

# Train AISFuser with custom parameters
python src/experiments/train.py model=ais_fuser data.batch_size=64 trainer.devices=2
```

### Hyperparameter Optimization

```bash
# Run hyperparameter search for TrAISformer
python src/experiments/train.py -m experiment=traisformer_sweep

# Run hyperparameter search for AISFuser
python src/experiments/train.py -m experiment=ais_fuser_sweep
```

### Model Evaluation

```bash
# Evaluate a trained model
python src/experiments/evaluation.py model=traisformer checkpoint=/path/to/checkpoint.ckpt

# Generate visualizations
python src/experiments/evaluation.py model=traisformer checkpoint=/path/to/checkpoint.ckpt visualize=true
```

### Running Batch Experiments

```bash
# Run multiple experiments in sequence
./scripts/run_experiments.sh
```

## Key Features

### Four-Hot Encoding

The TrAISformer model uses a novel four-hot encoding scheme for discretizing continuous AIS attributes:

```python
# Example configuration
config = {
    'lat_min': 50.0, 'lat_max': 60.0, 'lat_bins': 250,
    'lon_min': 0.0, 'lon_max': 10.0, 'lon_bins': 270,
    'sog_min': 0.0, 'sog_max': 30.0, 'sog_bins': 30,
    'cog_bins': 72  # 5° resolution for 360°
}

# Create encoding
from src.utils.maritime_utils import create_four_hot_encoding
encoding = create_four_hot_encoding(lat, lon, sog, cog, config)
```

### Maritime Graph Networks

The AISFuser model incorporates geographic information using graph neural networks:

```python
# Example graph processing
from src.data.graph_processor import AISGraphProcessor

processor = AISGraphProcessor(
    dp_epsilon=0.03,    # Douglas-Peucker epsilon
    dbscan_eps=1.5,     # DBSCAN epsilon (in km)
    min_samples=100     # Min samples for clustering
)

# Process trajectories into a maritime graph
graph = processor.process(trajectories)
```

### Evaluation Metrics

Maritime-specific metrics for trajectory evaluation:

```python
from src.utils.metrics import haversine_distance, rmse_haversine

# Calculate distance between predicted and actual positions
error = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)

# RMSE over a trajectory
rmse = rmse_haversine(pred_lats, pred_lons, true_lats, true_lons)
```

## Extending the Framework

### Adding a New Model

1. Create a new model file in `src/models/`
2. Implement the model using PyTorch Lightning
3. Add a configuration in `configs/model/`
4. Register the model in the model factory

Example:

```python
# src/models/my_model.py
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        # Implement model architecture
        
    def forward(self, x):
        # Forward pass
        
    def training_step(self, batch, batch_idx):
        # Training logic
        
    def configure_optimizers(self):
        # Optimizer configuration
```

### Creating Custom Experiments

Use the Hydra configuration system to define new experiments:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - base
  - _self_
  - override model: my_model

# Custom parameters
my_parameter: value
```

## Citation

If you use this code in your research, please cite:

```
@article{
  title={Maritime Trajectory Prediction with PyTorch Lightning},
  author={Your Name},
  year={2023}
}
```

## License

MIT