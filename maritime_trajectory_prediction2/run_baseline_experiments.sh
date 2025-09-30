#!/bin/bash

# Run comprehensive baseline experiments for maritime trajectory prediction
# This script trains and evaluates all baseline models

set -e  # Exit on error

echo "==========================================="
echo "Maritime Trajectory Prediction"
echo "Comprehensive Baseline Experiments"
echo "==========================================="

# Configuration
EXPERIMENT_NAME="maritime_baselines_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="experiments"
MAX_EPOCHS=200  # Full training for scientific paper
BATCH_SIZE=32
USE_GPU=""  # Add --use_gpu if GPU available

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling GPU acceleration"
    USE_GPU="--use_gpu"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "- Experiment Name: $EXPERIMENT_NAME"
echo "- Output Directory: $OUTPUT_DIR"
echo "- Max Epochs: $MAX_EPOCHS"
echo "- Batch Size: $BATCH_SIZE"
echo "- GPU: ${USE_GPU:-Disabled}"
echo ""

# Step 1: Train all baseline models
echo "==========================================="
echo "Step 1: Training Baseline Models"
echo "==========================================="

python3 train_all_baselines.py \
    --output_dir $OUTPUT_DIR \
    --experiment_name $EXPERIMENT_NAME \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_maritime_safety \
    $USE_GPU

echo ""
echo "✓ Training complete!"
echo ""

# Step 2: Evaluate trained models
echo "==========================================="
echo "Step 2: Evaluating Trained Models"
echo "==========================================="

python3 evaluate_baselines.py \
    $OUTPUT_DIR/$EXPERIMENT_NAME

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 3: Display results summary
echo "==========================================="
echo "Step 3: Results Summary"
echo "==========================================="

RESULTS_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/results"

if [ -f "$RESULTS_DIR/evaluation_results.csv" ]; then
    echo "Top 5 Models by RMSE:"
    echo ""
    head -n 6 "$RESULTS_DIR/evaluation_results.csv" | column -t -s,
fi

echo ""
echo "==========================================="
echo "✅ EXPERIMENT COMPLETE!"
echo "==========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Checkpoints saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints"
echo "Plots saved to: $RESULTS_DIR/plots"
echo "LaTeX tables saved to: $RESULTS_DIR/latex_tables.tex"
echo ""
echo "For detailed analysis, check:"
echo "- $RESULTS_DIR/evaluation_results.json"
echo "- $RESULTS_DIR/plots/*.png"
echo ""
