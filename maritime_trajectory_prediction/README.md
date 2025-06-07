# Maritime Trajectory Prediction

A comprehensive Python package for predicting maritime vessel trajectories using transformer-based models and AIS (Automatic Identification System) data.

## Features

- **TrAISformer**: Transformer-based model specifically designed for AIS trajectory prediction
- **AISFuser**: Multi-modal fusion model combining AIS data with additional maritime features
- **Data Processing**: Comprehensive AIS data preprocessing and feature engineering
- **Evaluation**: Built-in metrics and visualization tools for trajectory prediction assessment
- **Experiments**: Configurable training and hyperparameter optimization framework

## Installation

### From Source

```bash
git clone https://github.com/jakupsv/maritime-trajectory-prediction.git
cd maritime-trajectory-prediction
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from maritime_trajectory_prediction import TrAISformer, AISProcessor

# Load and process AIS data
processor = AISProcessor()
data = processor.load_ais_data("path/to/ais_data.csv")
processed_data = processor.preprocess(data)

# Initialize and train model
model = TrAISformer(
    d_model=256,
    nhead=8,
    num_layers=6,
    sequence_length=100
)

# Train the model
model.fit(processed_data)

# Make predictions
predictions = model.predict(test_data)
```

### Command Line Interface

```bash
# Process AIS data
ais-process --input data/raw_ais.csv --output data/processed_ais.csv

# Train and predict trajectories
ais-predict --model traisformer --data data/processed_ais.csv --output predictions.csv
```

## Project Structure

```
maritime_trajectory_prediction/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # ML models (TrAISformer, AISFuser, etc.)
│   ├── utils/          # Utility functions
│   └── experiments/    # Training and evaluation
├── scripts/            # Command line tools
├── tests/              # Unit tests
└── configs/            # Configuration files
```

## Models

### TrAISformer

A transformer-based architecture specifically designed for maritime trajectory prediction:

- Multi-head self-attention for capturing temporal dependencies
- Positional encoding adapted for geographical coordinates
- Causal masking for autoregressive prediction

### AISFuser

A multi-modal fusion model that combines:

- AIS trajectory data
- Vessel characteristics
- Environmental conditions
- Port and route information

## Configuration

The package uses Hydra for configuration management. Configuration files are located in the `configs/` directory:

- `configs/model/`: Model configurations
- `configs/data/`: Data processing configurations  
- `configs/experiment/`: Training configurations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{maritime_trajectory_prediction,
  author = {Svøðstein, Jákup},
  title = {Maritime Trajectory Prediction: Transformer-based Models for AIS Data},
  year = {2024},
  url = {https://github.com/jakupsv/maritime-trajectory-prediction}
}
```

## Contact

Jákup Svøðstein - jakupsv@setur.fo

Project Link: https://github.com/jakupsv/maritime-trajectory-prediction

