# Test Markers Guide

This project uses pytest markers to categorize and filter tests for efficient testing workflows.

## Available Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `unit` | Fast isolated unit tests | Quick feedback during development |
| `slow` | Tests that take >5s or use significant resources | Run separately from unit tests |
| `integration` | Tests that traverse process boundaries | End-to-end validation |
| `perf` | Performance benchmarking tests | Measure performance characteristics |
| `gpu` | Tests that require CUDA context | Hardware-specific testing |
| `regression` | Tests that codify historical defects | Prevent regressions |
| `hypothesis` | Property-based tests using hypothesis | Automated test case generation |

## Usage Examples

### Run only fast unit tests (recommended for development):
```bash
pytest -m "unit and not slow"
```

### Run slow tests separately:
```bash
pytest -m "slow"
```

### Run integration tests:
```bash
pytest -m "integration"
```

### Run unit tests excluding integration and slow tests:
```bash
pytest -m "unit and not integration and not slow"
```

### Run performance benchmarks:
```bash
pytest -m "perf"
```

### Run all tests except slow ones:
```bash
pytest -m "not slow"
```

### Get test timing breakdown:
```bash
pytest --durations=10
```

## Test Categories Overview

### ✅ **Unit Tests (28 tests)** - Fast, Isolated
- Configuration validation
- Data filtering and validation
- Feature/target building
- Dataset splitting

### 🔄 **Integration Tests (5 tests)** - End-to-End
- Complete pipeline workflows
- Export/import cycles
- Multi-component interactions

### ⚡ **Slow Tests (9 tests)** - Resource Intensive
- Large dataset processing
- File I/O operations
- Complex transformations

### 🏎️ **Performance Tests** - Benchmarking
- Memory usage profiling
- Processing speed measurement
- Scalability testing

## CI/CD Recommendations

```yaml
# Fast feedback
- pytest -m "unit and not slow"

# Comprehensive validation
- pytest -m "integration"
- pytest -m "slow"

# Performance monitoring
- pytest -m "perf" --benchmark-only
```

## Current Test Distribution

- **Total**: 37 tests
- **Unit**: 28 tests (~76%)
- **Integration**: 5 tests (~14%)
- **Slow**: 9 tests (~24%)
- **Pass Rate**: 100% ✅

Run `pytest --markers` to see all available markers.
