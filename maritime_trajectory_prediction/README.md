# Maritime Trajectory Prediction

This repository contains a PyTorch Lightning implementation of advanced trajectory prediction models for maritime vessel tracking.

## Models

- **TrAISformer**: A transformer-based model for vessel trajectory prediction based on the paper [TrAISformer](https://github.com/CIA-Oceanix/TrAISformer)
- **AISFuser**: A graph-enhanced model for maritime trajectory prediction combining GNN and transformer architectures

## Features

- **PyTorch Lightning** integration for scalable training
- **PyTorch Geometric** for maritime graph networks
- **Hydra** configuration system
- **Weights & Biases** logging and experiment tracking
- **Hyperparameter optimization** with Optuna

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/maritime-trajectory-prediction.git
cd maritime-trajectory-prediction

# Install dependencies
pip install -r requirements.txt
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
│   ├── experiments/      # Training and evaluation scripts
│   ├── models/           # Model implementations
│   └── utils/            # Utility functions
├── scripts/              # Utility scripts
└── README.md
```

## Usage

### Training a Model

```bash
# Train TrAISformer
python src/experiments/train.py model=traisformer

# Train AISFuser
python src/experiments/train.py model=ais_fuser
```

### Hyperparameter Optimization

```bash
# Run hyperparameter search for TrAISformer
python src/experiments/train.py -m experiment=traisformer_sweep

# Run hyperparameter search for AISFuser
python src/experiments/train.py -m experiment=ais_fuser_sweep
```

### Evaluation

```bash
# Evaluate a trained model
python src/experiments/evaluation.py model=traisformer checkpoint=/path/to/checkpoint.ckpt
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
