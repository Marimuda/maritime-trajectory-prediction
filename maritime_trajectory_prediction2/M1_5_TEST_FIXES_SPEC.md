# M1.5: Test Infrastructure Fixes & Code Quality

## Overview

M1.5 addresses the remaining test failures and linting issues identified during the sanity check, ensuring a clean codebase before M2 development.

## Problem Analysis

**Linting Issues (29 errors):**
1. **F821 Undefined names:** Missing imports for `create_maritime_motion_transformer`, `TrajectoryMetrics`, `MARITIME_MTR_CONFIG`, `MotionTransformerTrainer`, `create_motion_transformer`
2. **SIM105:** Replace try/except/pass with `contextlib.suppress()`
3. **UP038:** Use `X | Y` instead of `(X, Y)` in isinstance calls

**Test Failures:**
1. **14 failed evalx tests:** Statistical precision/numerical issues in bootstrap confidence intervals and significance tests
2. **Multiple Kalman test failures:** Coordinate transformation and IMM filter edge cases

## Implementation Plan

### Phase 1: Fix Import Issues (Priority 1)

**Root Cause:** Tests import functions that don't exist or have different names.

**Solutions:**

1. **Motion Transformer Tests:**
   - `create_maritime_motion_transformer` → Create factory function in motion_transformer.py
   - `MARITIME_MTR_CONFIG` → Create config constants
   - `MotionTransformerTrainer` → Import from correct module or create adapter
   - `create_motion_transformer` → Create basic factory function

2. **Auxiliary Component Tests:**
   - `TrajectoryMetrics` → Create or import correct class from metrics module

### Phase 2: Fix Evalx Statistical Tests (Priority 2)

**Identified Issues:**
```
TestPairedTTest::test_basic_paired_t_test - FAILED
TestWilcoxonTest::test_basic_wilcoxon_test - FAILED
TestMcNemarTest::test_basic_mcnemar_test - FAILED
```

**Root Causes:**
1. **Numerical precision:** Small datasets causing division by zero or near-zero denominators
2. **Test expectations:** Hard-coded expected values don't account for floating-point precision
3. **Edge cases:** Empty or single-value datasets not handled properly

**Solutions:**
1. Use `pytest.approx()` for floating-point comparisons
2. Add proper edge case handling in statistical functions
3. Increase tolerance for statistical test comparisons
4. Fix bootstrap confidence interval edge cases

### Phase 3: Fix Kalman Filter Tests (Priority 2)

**Identified Failures:**
```
TestMaritimeIMMFilter::test_fit - FAILED
TestMaritimeIMMFilter::test_predict_straight_line - FAILED
TestMaritimeIMMFilter::test_predict_turning_trajectory - FAILED
```

**Root Causes:**
1. **Coordinate system issues:** Lat/lon to local coordinate conversion edge cases
2. **Sequence length validation:** Tests using insufficient data points
3. **IMM initialization:** Model probability and transition matrix issues

**Solutions:**
1. Fix coordinate transformation validation
2. Ensure test data meets minimum requirements
3. Improve IMM filter initialization robustness

### Phase 4: Code Quality Improvements (Priority 3)

**Linting Fixes:**
1. Replace `try/except/pass` with `contextlib.suppress()`
2. Update `isinstance()` calls to use `|` syntax
3. Add missing imports and fix undefined names

## Implementation Strategy

### Step 1: Create Missing Factory Functions

```python
# In src/models/motion_transformer.py
MARITIME_MTR_CONFIG = {
    "small": {"input_dim": 4, "d_model": 128, "n_queries": 4, "encoder_layers": 2, "decoder_layers": 2},
    "medium": {"input_dim": 4, "d_model": 256, "n_queries": 8, "encoder_layers": 4, "decoder_layers": 4},
    "large": {"input_dim": 4, "d_model": 512, "n_queries": 16, "encoder_layers": 6, "decoder_layers": 6}
}

def create_maritime_motion_transformer(size: str = "medium"):
    """Create maritime-configured Motion Transformer."""
    if size not in MARITIME_MTR_CONFIG:
        raise ValueError(f"Unknown size: {size}. Available: {list(MARITIME_MTR_CONFIG.keys())}")
    config = MARITIME_MTR_CONFIG[size]
    return MotionTransformer(**config)

def create_motion_transformer(**kwargs):
    """Create Motion Transformer with custom config."""
    return MotionTransformer(**kwargs)
```

### Step 2: Fix Evalx Statistical Issues

```python
# In src/evalx/stats/tests.py
def paired_t_test(group1, group2, alternative="two-sided", alpha=0.05):
    # Add numerical stability checks
    if len(group1) <= 1 or len(group2) <= 1:
        return TestResult(statistic=np.nan, p_value=1.0, effect_size=0.0, significant=False)

    differences = np.array(group1) - np.array(group2)
    if np.all(differences == 0) or np.std(differences, ddof=1) < 1e-10:
        return TestResult(statistic=0.0, p_value=1.0, effect_size=0.0, significant=False)
```

### Step 3: Fix Kalman Filter Edge Cases

```python
# In src/models/baseline_models/kalman/models.py
def fit(self, sequences: list[np.ndarray], **kwargs):
    if not sequences or len(sequences) == 0:
        raise ValueError("No training sequences provided")

    # Validate sequence lengths
    for i, seq in enumerate(sequences):
        if len(seq) < self.MIN_SEQUENCE_LENGTH:
            warnings.warn(f"Sequence {i} too short ({len(seq)} < {self.MIN_SEQUENCE_LENGTH}), skipping")
            continue
```

## Success Criteria

1. **Linting:** Zero ruff/flake8 errors
2. **Import Tests:** All test files can be imported successfully
3. **Evalx Tests:** 14 statistical tests pass with proper numerical handling
4. **Kalman Tests:** All IMM and coordinate transformation tests pass
5. **Overall Test Coverage:** >90% of working tests pass

## Testing Strategy

### Incremental Testing
1. Fix imports → Test collection
2. Fix evalx → Run evalx test suite
3. Fix Kalman → Run Kalman test suite
4. Full test suite → Comprehensive validation

### Validation Commands
```bash
# Test import resolution
python -m pytest --collect-only -q

# Test specific components
python -m pytest tests/unit/evalx/ -v
python -m pytest tests/unit/kalman/ -v

# Full validation
python -m pytest tests/unit/ -x --tb=short
```

## Risk Assessment

**Low Risk:**
- Import fixes are straightforward
- Statistical test improvements are well-documented

**Medium Risk:**
- Kalman filter coordinate transformations are complex
- Evalx numerical precision may require extensive testing

**Mitigation:**
- Implement fixes incrementally
- Maintain backward compatibility
- Add comprehensive edge case testing

---

M1.5 will deliver a clean, fully-tested codebase ready for M2 development with zero shortcuts and proper error handling.
