#!/bin/bash

# Script to run baseline LSTM and TrAISformer models sequentially
# Created for maritime trajectory prediction comparison

set -e  # Exit on any error

echo "=========================================="
echo "Maritime Trajectory Prediction Comparison"
echo "Running Baseline LSTM and TrAISformer"
echo "=========================================="
echo ""

# Get start time
start_time=$(date)
echo "Starting experiments at $start_time"
echo ""

# Function to check if command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ $1 failed"
        exit 1
    fi
}

# Function to log experiment start
log_experiment_start() {
    echo "----------------------------------------"
    echo "Starting: $1"
    echo "Time: $(date)"
    echo "----------------------------------------"
}

# Function to log experiment end
log_experiment_end() {
    echo "----------------------------------------"
    echo "Completed: $1"
    echo "Time: $(date)"
    echo "----------------------------------------"
    echo ""
}

# Baseline LSTM Model
log_experiment_start "Baseline LSTM Training"
python main.py mode=train model=lstm data=ais_processed
check_success "Baseline LSTM training"
log_experiment_end "Baseline LSTM Training"

# TrAISformer Model
log_experiment_start "TrAISformer Training"
python main.py mode=train model=traisformer data=ais_processed
check_success "TrAISformer training"
log_experiment_end "TrAISformer Training"

# Final summary
end_time=$(date)
echo "=========================================="
echo "All experiments completed successfully!"
echo ""
echo "Started at:  $start_time"
echo "Finished at: $end_time"
echo ""
echo "Models trained:"
echo "  1. Baseline LSTM"
echo "  2. TrAISformer"
echo ""
echo "Check wandb dashboard for results:"
echo "  Project: maritime_trajectory_prediction"
echo "=========================================="
