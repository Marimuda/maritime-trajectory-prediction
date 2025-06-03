#\!/bin/bash
# Script to run multiple experiments with different configurations

# Set base directory
BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $BASE_DIR

# Create output directory
mkdir -p outputs

# Log start time
echo "Starting experiments at $(date)"

# Run TrAISformer experiment
echo "Running TrAISformer experiment..."
python src/experiments/train.py model=traisformer

# Run AISFuser experiment
echo "Running AISFuser experiment..."
python src/experiments/train.py model=ais_fuser

# Run hyperparameter sweep for TrAISformer
echo "Running TrAISformer hyperparameter sweep..."
python src/experiments/train.py -m experiment=traisformer_sweep

# Run hyperparameter sweep for AISFuser
echo "Running AISFuser hyperparameter sweep..."
python src/experiments/train.py -m experiment=ais_fuser_sweep

# Log end time
echo "All experiments completed at $(date)"
