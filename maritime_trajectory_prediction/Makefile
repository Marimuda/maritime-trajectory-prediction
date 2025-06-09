# Maritime Trajectory Prediction - Development Makefile
#
# Common tasks for development, testing, and training

.PHONY: help clean install dev-install test test-fast test-slow test-integration test-perf lint format check train build data setup

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := maritime_trajectory_prediction
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
MODELS_DIR := models
CHECKPOINTS_DIR := checkpoints
LOGS_DIR := logs
OUTPUTS_DIR := outputs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)Maritime Trajectory Prediction - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*(Setup|Install)/ {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*(Clean|Format|Lint|Check)/ {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*Test/ {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Training & Data:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*(Train|Build|Data)/ {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Setup and installation targets
setup: clean install setup-dirs setup-git ## Setup: Complete project setup
	@echo "$(GREEN)✓ Project setup complete!$(NC)"

install: ## Install: Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev-install: ## Install: Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov pytest-benchmark pre-commit ruff black isort mypy bandit
	pre-commit install

setup-dirs: ## Setup: Create necessary directories
	@echo "$(BLUE)Creating project directories...$(NC)"
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/interim
	@mkdir -p $(MODELS_DIR) $(CHECKPOINTS_DIR) $(LOGS_DIR) $(OUTPUTS_DIR)
	@mkdir -p results/plots results/metrics
	@touch $(DATA_DIR)/.gitkeep $(MODELS_DIR)/.gitkeep $(LOGS_DIR)/.gitkeep

setup-git: ## Setup: Configure git hooks and settings
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		pre-commit install --hook-type commit-msg; \
	else \
		echo "$(YELLOW)Warning: pre-commit not found, skipping git hooks setup$(NC)"; \
	fi

# Cleaning targets
clean: clean-cache clean-build clean-test clean-logs ## Clean: Remove all cache and build files
	@echo "$(GREEN)✓ All clean tasks completed!$(NC)"

clean-cache: ## Clean: Remove Python cache files
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

clean-build: ## Clean: Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .coverage htmlcov/

clean-test: ## Clean: Remove test artifacts
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/ .coverage coverage.xml
	@find . -name "*.coverage" -delete

clean-logs: ## Clean: Remove log files
	@echo "$(BLUE)Cleaning log files...$(NC)"
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf wandb/

clean-models: ## Clean: Remove model checkpoints (WARNING: Irreversible!)
	@echo "$(RED)WARNING: This will delete all model checkpoints!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@rm -rf $(CHECKPOINTS_DIR)/* $(MODELS_DIR)/*
	@echo "$(GREEN)✓ Model files cleaned$(NC)"

# Code quality targets
lint: ## Lint: Run ruff linting
	@echo "$(BLUE)Running ruff linting...$(NC)"
	@ruff check $(SRC_DIR) $(TEST_DIR) --fix

format: ## Format: Format code with ruff
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	@ruff format $(SRC_DIR) $(TEST_DIR)

check: lint format ## Check: Run all code quality checks
	@echo "$(GREEN)✓ Code quality checks completed!$(NC)"

# Testing targets
test: ## Test: Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-fast: ## Test: Run only fast unit tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "unit and not slow" --cov=$(SRC_DIR)

test-slow: ## Test: Run slow/integration tests
	@echo "$(BLUE)Running slow tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "slow" --cov=$(SRC_DIR)

test-integration: ## Test: Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "integration" --cov=$(SRC_DIR)

test-perf: ## Test: Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "perf" --benchmark-only --benchmark-sort=mean

test-coverage: ## Test: Generate detailed coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# Training and model targets
train: ## Train: Run default training (Motion Transformer)
	@echo "$(BLUE)Starting default training (Motion Transformer)...$(NC)"
	$(PYTHON) train_transformer_models.py \
		--model-type motion_transformer \
		--size medium \
		--epochs 50 \
		--batch-size 32 \
		--learning-rate 1e-4 \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUTS_DIR)

train-anomaly: ## Train: Train anomaly detection model
	@echo "$(BLUE)Training anomaly detection model...$(NC)"
	$(PYTHON) train_transformer_models.py \
		--model-type anomaly_transformer \
		--size medium \
		--epochs 30 \
		--batch-size 32 \
		--learning-rate 1e-4 \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUTS_DIR)

train-baseline: ## Train: Train baseline LSTM model
	@echo "$(BLUE)Training baseline model...$(NC)"
	$(PYTHON) train_transformer_models.py \
		--model-type baseline \
		--task trajectory_prediction \
		--epochs 50 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUTS_DIR)

train-wandb: ## Train: Train with Weights & Biases logging
	@echo "$(BLUE)Training with W&B logging...$(NC)"
	$(PYTHON) train_transformer_models.py \
		--model-type motion_transformer \
		--size medium \
		--epochs 50 \
		--batch-size 32 \
		--use-wandb \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUTS_DIR)

train-sweep: ## Train: Run hyperparameter sweep
	@echo "$(BLUE)Running hyperparameter sweep...$(NC)"
	$(PYTHON) scripts/run_experiments.sh

# Build targets
build: clean check test-fast train ## Build: Complete build pipeline
	@echo "$(GREEN)✓ Build pipeline completed successfully!$(NC)"

build-ci: clean check test ## Build: CI/CD build pipeline
	@echo "$(GREEN)✓ CI build pipeline completed!$(NC)"

# Data processing targets
data: data-download data-process data-validate ## Data: Complete data pipeline
	@echo "$(GREEN)✓ Data pipeline completed!$(NC)"

data-download: ## Data: Download sample data
	@echo "$(BLUE)Downloading sample data...$(NC)"
	@if [ ! -f $(DATA_DIR)/raw/100k_ais ]; then \
		echo "$(YELLOW)Sample data not found. Please place AIS data in $(DATA_DIR)/raw/$(NC)"; \
	else \
		echo "$(GREEN)✓ Data found in $(DATA_DIR)/raw/$(NC)"; \
	fi

data-process: ## Data: Process raw AIS data
	@echo "$(BLUE)Processing AIS data...$(NC)"
	$(PYTHON) scripts/process_ais_catcher.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed

data-validate: ## Data: Validate processed data
	@echo "$(BLUE)Validating processed data...$(NC)"
	$(PYTHON) -c "from src.data.validation import DataValidator; \
		import pandas as pd; \
		validator = DataValidator(); \
		print('Data validation completed')"

data-visualize: ## Data: Create data visualizations
	@echo "$(BLUE)Creating data visualizations...$(NC)"
	$(PYTHON) visualize_results.py --output results/plots/

# Evaluation targets
evaluate: ## Evaluate: Run model evaluation
	@echo "$(BLUE)Evaluating trained models...$(NC)"
	$(PYTHON) evaluate_transformer_models.py \
		--checkpoints-dir $(CHECKPOINTS_DIR) \
		--data-dir $(DATA_DIR) \
		--output-dir results/

inference: ## Inference: Run model inference
	@echo "$(BLUE)Running model inference...$(NC)"
	$(PYTHON) inference_transformer_models.py \
		--model-path $(CHECKPOINTS_DIR)/best_model.pt \
		--data-path $(DATA_DIR)/processed \
		--output results/predictions.csv

# Utility targets
requirements: ## Generate requirements.txt from current environment
	@echo "$(BLUE)Generating requirements.txt...$(NC)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✓ Requirements updated$(NC)"

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "  Python version: $$(python3 --version)"
	@echo "  Working directory: $$(pwd)"
	@echo "  Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "  Git status: $$(git status --porcelain | wc -l) modified files"
	@echo ""
	@echo "$(BLUE)Directory Sizes:$(NC)"
	@echo "  Data: $$(du -sh $(DATA_DIR) 2>/dev/null || echo 'N/A')"
	@echo "  Models: $$(du -sh $(MODELS_DIR) 2>/dev/null || echo 'N/A')"
	@echo "  Logs: $$(du -sh $(LOGS_DIR) 2>/dev/null || echo 'N/A')"
	@echo "  Checkpoints: $$(du -sh $(CHECKPOINTS_DIR) 2>/dev/null || echo 'N/A')"

info: ## Show detailed project information
	@echo "$(BLUE)Maritime Trajectory Prediction Project$(NC)"
	@echo "======================================"
	@echo ""
	@echo "$(GREEN)Available Models:$(NC)"
	@echo "  • Motion Transformer (SOTA trajectory prediction)"
	@echo "  • Anomaly Transformer (SOTA anomaly detection)"
	@echo "  • Baseline LSTM (traditional approach)"
	@echo ""
	@echo "$(GREEN)Key Features:$(NC)"
	@echo "  • Real-time AIS data processing"
	@echo "  • Multi-task learning support"
	@echo "  • Comprehensive test suite"
	@echo "  • Performance benchmarking"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  1. make setup          # Initial setup"
	@echo "  2. make data           # Process data"
	@echo "  3. make train          # Train model"
	@echo "  4. make evaluate       # Evaluate results"

# Development workflow shortcuts
dev: clean format lint test-fast ## Dev: Quick development cycle
	@echo "$(GREEN)✓ Development cycle completed!$(NC)"

ci: clean check test ## CI: Continuous integration pipeline
	@echo "$(GREEN)✓ CI pipeline completed!$(NC)"

release: clean check test build ## Release: Prepare for release
	@echo "$(GREEN)✓ Release preparation completed!$(NC)"

# Docker targets (if needed)
docker-build: ## Docker: Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Docker: Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -v $(PWD)/data:/app/data $(PROJECT_NAME):latest

# Monitoring targets
monitor-train: ## Monitor: Watch training logs
	@echo "$(BLUE)Monitoring training logs...$(NC)"
	@if [ -f $(LOGS_DIR)/training.log ]; then \
		tail -f $(LOGS_DIR)/training.log; \
	else \
		echo "$(YELLOW)No training logs found$(NC)"; \
	fi

monitor-system: ## Monitor: Show system resource usage
	@echo "$(BLUE)System Resource Usage:$(NC)"
	@echo "CPU: $$(top -bn1 | grep "Cpu(s)" | awk '{print $$2}' | cut -d'%' -f1)%"
	@echo "Memory: $$(free -h | awk '/^Mem:/{print $$3 "/" $$2}')"
	@echo "Disk: $$(df -h . | awk 'NR==2{print $$3 "/" $$2 " (" $$5 ")"}')"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU: $$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"; \
	fi

# Backup targets
backup: ## Backup: Create backup of important files
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups
	@tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz \
		--exclude=__pycache__ \
		--exclude=.git \
		--exclude=data/raw \
		--exclude=checkpoints \
		src/ tests/ configs/ *.py *.md Makefile requirements.txt
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"
