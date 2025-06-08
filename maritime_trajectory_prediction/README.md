# Maritime Trajectory Prediction

A comprehensive Python package for predicting maritime vessel trajectories using state-of-the-art transformer-based models and AIS (Automatic Identification System) data.

## 🚀 Features

### State-of-the-Art Models
- **Anomaly Transformer** (ICLR 2022): Advanced anomaly detection with attention mechanisms
- **Motion Transformer** (NeurIPS 2022): Multimodal trajectory prediction with transformer architecture
- **Baseline Models**: LSTM, Autoencoder, and GCN models for comparison

### Core Capabilities
- **Real-time Anomaly Detection**: Detect unusual vessel behavior patterns
- **Multimodal Trajectory Prediction**: Generate multiple possible future paths
- **Maritime Data Processing**: Comprehensive AIS data preprocessing and feature engineering
- **Performance Evaluation**: Built-in metrics and visualization tools
- **Production Ready**: Unified training and inference pipelines

### Advanced Features
- **Attention Mechanisms**: Interpretable attention weights for model understanding
- **Scalable Architecture**: Efficient batch processing for operational deployment
- **Configuration Management**: YAML-based configuration with command-line overrides
- **Comprehensive Testing**: Unit, integration, and performance tests
- **Visualization Tools**: Publication-quality plots and dashboards

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

## 📊 Quick Start

### SOTA Model Usage

```python
from models import create_model

# Create SOTA models
anomaly_model = create_model('anomaly_transformer', size='medium')
motion_model = create_model('motion_transformer', size='small')

# Anomaly detection
results = anomaly_model.detect_anomalies(trajectory_data, threshold=0.5)
print(f"Anomalies detected: {results['binary_anomalies'].sum()}")

# Trajectory prediction
predictions = motion_model.predict_best_trajectory(context_data)
print(f"Predicted trajectory shape: {predictions.shape}")
```

### Training Models

```bash
# Train Anomaly Transformer
python train_sota.py \
  --model-type anomaly_transformer \
  --size medium \
  --epochs 50 \
  --batch-size 16

# Train Motion Transformer
python train_sota.py \
  --model-type motion_transformer \
  --size small \
  --epochs 100 \
  --batch-size 32
```

### Inference Pipeline

```bash
# Anomaly detection
python inference_sota.py \
  --model-path checkpoints/anomaly_transformer/best_model.pt \
  --data-path data/trajectories.csv \
  --task anomaly_detection

# Trajectory prediction
python inference_sota.py \
  --model-path checkpoints/motion_transformer/best_model.pt \
  --data-path data/trajectories.csv \
  --task trajectory_prediction
```

### Legacy Usage (Baseline Models)

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

## 🏗️ Project Structure

```
maritime_trajectory_prediction/
├── src/
│   ├── data/           # Data processing and pipeline modules
│   ├── models/         # SOTA and baseline models
│   │   ├── anomaly_transformer.py    # Anomaly Transformer implementation
│   │   ├── motion_transformer.py     # Motion Transformer implementation
│   │   └── baseline_models.py        # LSTM, Autoencoder, GCN models
│   ├── utils/          # Maritime utilities and metrics
│   └── experiments/    # Training and evaluation frameworks
├── scripts/            # Command line tools and examples
├── tests/              # Comprehensive test suite
├── configs/            # YAML configuration files
├── docs/               # Documentation
├── examples/           # Usage examples and tutorials
└── validation_results/ # Model validation reports and plots
```

## 🤖 Models

### State-of-the-Art Models

#### Anomaly Transformer (ICLR 2022)
- **Novel Anomaly-Attention**: Computes association discrepancy for anomaly detection
- **Minimax Training**: Amplifies differences between normal and anomalous patterns
- **Maritime Adaptation**: Optimized for vessel behavior analysis
- **Performance**: 100% detection rate with real-time inference

#### Motion Transformer (NeurIPS 2022)
- **Multimodal Prediction**: Generates multiple trajectory hypotheses
- **Learnable Queries**: 4-8 query vectors for different motion modes
- **Best-of-N Training**: Optimizes for closest prediction to ground truth
- **Uncertainty Handling**: Confidence scores for each prediction mode

### Baseline Models

#### TrAISformer (Legacy)
A transformer-based architecture specifically designed for maritime trajectory prediction:
- Multi-head self-attention for capturing temporal dependencies
- Positional encoding adapted for geographical coordinates
- Causal masking for autoregressive prediction

#### AISFuser
A multi-modal fusion model that combines:
- AIS trajectory data
- Vessel characteristics
- Environmental conditions
- Port and route information

## ⚙️ Configuration

The package uses YAML-based configuration management with command-line overrides:

```yaml
# configs/sota_configs.yaml
model:
  type: "motion_transformer"
  size: "medium"
  custom_params:
    d_model: 256
    n_queries: 6

training:
  batch_size: 16
  learning_rate: 1e-4
  max_epochs: 100

data:
  sequence_length: 30
  prediction_horizon: 10
```

Configuration directories:
- `configs/model/`: Model-specific configurations
- `configs/data/`: Data processing configurations  
- `configs/experiment/`: Training and evaluation configurations

## 📊 Performance Benchmarks

### Validation Results

| Model | Task | Performance | Inference Time |
|-------|------|-------------|----------------|
| Anomaly Transformer | Anomaly Detection | 100% Detection Rate | 5-27ms |
| Motion Transformer | Trajectory Prediction | ADE: 63.99 | 4-25ms |
| Baseline Autoencoder | Anomaly Detection | 50% Detection Rate | 0.3ms |
| Baseline LSTM | Trajectory Prediction | ADE: 62.36 | 2.1ms |

### Model Complexity

| Model | Parameters | Memory Usage | Throughput |
|-------|------------|--------------|------------|
| Anomaly Transformer (Small) | 3.2M | ~50MB | 192 samples/s |
| Motion Transformer (Small) | 1.2M | ~25MB | 233 samples/s |
| Baseline Models | 42K-839K | ~5-15MB | 483-3,086 samples/s |

## 📚 Documentation

- **[SOTA Models Guide](docs/SOTA_MODELS.md)**: Comprehensive documentation for state-of-the-art models
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Training Guide](docs/TRAINING.md)**: Model training best practices
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production deployment instructions

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v

# All tests
pytest tests/ -v --cov=src
```

## 🚀 Examples

Check the `examples/` directory for:
- **Training Examples**: Complete training workflows
- **Inference Examples**: Real-time and batch inference
- **Visualization Examples**: Creating plots and dashboards
- **Configuration Examples**: Custom model configurations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/jakupsv/maritime-trajectory-prediction.git
cd maritime-trajectory-prediction

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{maritime_trajectory_prediction,
  author = {Svøðstein, Jákup},
  title = {Maritime Trajectory Prediction: State-of-the-Art Transformer Models for AIS Data},
  year = {2024},
  url = {https://github.com/jakupsv/maritime-trajectory-prediction},
  note = {Includes Anomaly Transformer (ICLR 2022) and Motion Transformer (NeurIPS 2022)}
}
```

### Research References

- **Anomaly Transformer**: Xu et al. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" ICLR 2022
- **Motion Transformer**: Shi et al. "Motion Transformer with Global Intention Localization and Local Movement Refinement" NeurIPS 2022

## 📞 Contact

**Jákup Svøðstein**  
Email: jakupsv@setur.fo  
GitHub: [@jakupsv](https://github.com/jakupsv)

## 🙏 Acknowledgments

- Research teams behind the Anomaly Transformer and Motion Transformer papers
- Maritime domain experts who provided insights for model adaptation
- Open source community for foundational tools and libraries

## 📈 Changelog

### v1.0.0 (2024-06-08)
- ✅ **SOTA Integration**: Added Anomaly Transformer and Motion Transformer
- ✅ **Unified Pipeline**: Complete training and inference workflows
- ✅ **Comprehensive Testing**: Unit, integration, and performance tests
- ✅ **Real Data Validation**: Validated with maritime AIS data
- ✅ **Production Ready**: Scalable architecture for deployment
- ✅ **Documentation**: Complete API and usage documentation

### v0.2.0 (Previous)
- Added baseline models (LSTM, Autoencoder, GCN)
- Implemented data processing pipeline
- Added evaluation metrics and visualization tools

### v0.1.0 (Initial)
- Basic TrAISformer implementation
- AIS data preprocessing utilities
- Initial project structure

Project Link: https://github.com/jakupsv/maritime-trajectory-prediction

