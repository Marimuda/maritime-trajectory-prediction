#!/bin/bash

# SOTA Model Training and Inference Examples
# This script demonstrates how to use the SOTA training and inference pipeline

echo "🚀 SOTA Model Training and Inference Examples"
echo "=============================================="

# Create necessary directories
mkdir -p data/test outputs checkpoints logs configs

# Example 1: Train Anomaly Transformer (small config for testing)
echo ""
echo "📋 Example 1: Training Anomaly Transformer"
echo "python train_sota.py \\"
echo "  --model-type anomaly_transformer \\"
echo "  --size small \\"
echo "  --epochs 5 \\"
echo "  --batch-size 8 \\"
echo "  --learning-rate 1e-3 \\"
echo "  --data-dir ./data/test \\"
echo "  --output-dir ./outputs/anomaly_test"

# Example 2: Train Motion Transformer
echo ""
echo "📋 Example 2: Training Motion Transformer"
echo "python train_sota.py \\"
echo "  --model-type motion_transformer \\"
echo "  --size medium \\"
echo "  --epochs 10 \\"
echo "  --batch-size 16 \\"
echo "  --learning-rate 1e-4 \\"
echo "  --data-dir ./data/processed \\"
echo "  --output-dir ./outputs/motion_transformer \\"
echo "  --use-wandb"

# Example 3: Train Baseline Model
echo ""
echo "📋 Example 3: Training Baseline Model"
echo "python train_sota.py \\"
echo "  --model-type baseline \\"
echo "  --task trajectory_prediction \\"
echo "  --epochs 20 \\"
echo "  --batch-size 32 \\"
echo "  --learning-rate 1e-3 \\"
echo "  --data-dir ./data/processed \\"
echo "  --output-dir ./outputs/baseline"

# Example 4: Train with Configuration File
echo ""
echo "📋 Example 4: Training with Configuration File"
echo "python train_sota.py --config configs/sota_configs.yaml"

# Example 5: Resume Training from Checkpoint
echo ""
echo "📋 Example 5: Resume Training"
echo "python train_sota.py \\"
echo "  --config configs/motion_transformer.yaml \\"
echo "  --resume checkpoints/motion_transformer/best_model.pt"

echo ""
echo "🔍 INFERENCE EXAMPLES"
echo "===================="

# Example 6: Anomaly Detection Inference
echo ""
echo "📋 Example 6: Anomaly Detection"
echo "python inference_sota.py \\"
echo "  --model-path checkpoints/anomaly_transformer/best_model.pt \\"
echo "  --data-path data/test_trajectories.csv \\"
echo "  --task anomaly_detection \\"
echo "  --threshold 0.7 \\"
echo "  --output-path results/anomalies.csv \\"
echo "  --format csv"

# Example 7: Trajectory Prediction Inference
echo ""
echo "📋 Example 7: Trajectory Prediction"
echo "python inference_sota.py \\"
echo "  --model-path checkpoints/motion_transformer/best_model.pt \\"
echo "  --data-path data/test_trajectories.csv \\"
echo "  --task trajectory_prediction \\"
echo "  --output-path results/predictions.json \\"
echo "  --format json"

# Example 8: Batch Inference
echo ""
echo "📋 Example 8: Batch Inference"
echo "python inference_sota.py \\"
echo "  --model-path checkpoints/motion_transformer/best_model.pt \\"
echo "  --data-path data/large_dataset.parquet \\"
echo "  --batch-size 64 \\"
echo "  --output-path results/batch_predictions.npz \\"
echo "  --format npz"

# Example 9: Model Comparison
echo ""
echo "📋 Example 9: Model Comparison"
echo "python inference_sota.py \\"
echo "  --model-path checkpoints/motion_transformer/best_model.pt \\"
echo "  --data-path data/test_trajectories.csv \\"
echo "  --compare-models \\"
echo "    checkpoints/baseline/best_model.pt \\"
echo "    checkpoints/anomaly_transformer/best_model.pt \\"
echo "  --output-path results/model_comparison.json"

# Example 10: Real-time Inference
echo ""
echo "📋 Example 10: Real-time Inference"
echo "python inference_sota.py \\"
echo "  --model-path checkpoints/motion_transformer/best_model.pt \\"
echo "  --data-path data/streaming_data.csv \\"
echo "  --real-time \\"
echo "  --batch-size 1"

echo ""
echo "⚙️  CONFIGURATION EXAMPLES"
echo "========================="

# Example 11: Create Custom Configuration
echo ""
echo "📋 Example 11: Custom Configuration"
cat << 'EOF'
# Create custom_config.yaml
model:
  type: "motion_transformer"
  size: "large"
  custom_params:
    d_model: 512
    n_queries: 8
    prediction_horizon: 50

training:
  batch_size: 8
  learning_rate: 5e-5
  max_epochs: 200
  patience: 25

data:
  sequence_length: 60
  prediction_horizon: 50

logging:
  use_wandb: true
  project_name: "maritime-sota-large"
  experiment_name: "large_motion_transformer_v1"

paths:
  data_dir: "./data/large_dataset"
  output_dir: "./outputs/large_motion_transformer"
EOF

echo ""
echo "🧪 TESTING EXAMPLES"
echo "=================="

# Example 12: Quick Test Training
echo ""
echo "📋 Example 12: Quick Test (5 minutes)"
echo "python train_sota.py \\"
echo "  --model-type motion_transformer \\"
echo "  --size small \\"
echo "  --epochs 3 \\"
echo "  --batch-size 4 \\"
echo "  --learning-rate 1e-3 \\"
echo "  --data-dir ./data/test \\"
echo "  --output-dir ./outputs/quick_test"

# Example 13: Model Information
echo ""
echo "📋 Example 13: Model Information"
echo "python -c \\"
echo "\"from models import get_model_info, list_available_models; \\"
echo "print('Available models:', list_available_models()); \\"
echo "print('Motion Transformer info:', get_model_info('motion_transformer'))\""

echo ""
echo "📊 MONITORING AND EVALUATION"
echo "============================"

# Example 14: Monitor Training with TensorBoard
echo ""
echo "📋 Example 14: Monitor with TensorBoard"
echo "tensorboard --logdir logs/ --port 6006"

# Example 15: Evaluate Model Performance
echo ""
echo "📋 Example 15: Performance Evaluation"
echo "python -c \\"
echo "\"from inference_sota import SOTAInference; \\"
echo "import time; \\"
echo "inference = SOTAInference('checkpoints/best_model.pt'); \\"
echo "# Add your evaluation code here\""

echo ""
echo "🔧 TROUBLESHOOTING"
echo "=================="

echo ""
echo "📋 Common Issues and Solutions:"
echo "1. CUDA out of memory: Reduce batch_size"
echo "2. Import errors: Check Python path and dependencies"
echo "3. Data loading issues: Verify data format and paths"
echo "4. Training not converging: Adjust learning rate or model size"
echo "5. Inference errors: Check model compatibility with data"

echo ""
echo "📋 Performance Tips:"
echo "1. Use GPU for training: CUDA_VISIBLE_DEVICES=0"
echo "2. Increase num_workers for faster data loading"
echo "3. Use mixed precision: --fp16 (if implemented)"
echo "4. Monitor GPU memory: nvidia-smi"
echo "5. Use Weights & Biases for experiment tracking"

echo ""
echo "✅ Setup complete! You can now run the examples above."
echo "📚 For more details, check the documentation in docs/"

