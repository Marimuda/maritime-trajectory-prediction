# M2 Parallel Development Workflow - Git Worktree Setup

## Overview

Successfully created git worktree split for **parallel development** of M2.1 and M2.2 with **minimal merge conflicts**. This enables simultaneous work on error analysis and maritime validation components.

## Worktree Structure

```
/home/jakup/repo/maritime-trajectory-prediction/
├── maritime_trajectory_prediction2/                    [main branch - integration point]
├── maritime-m2.1-error-analysis/maritime_trajectory_prediction2/    [feature/m2.1-error-analysis]
└── maritime-m2.2-maritime-validation/maritime_trajectory_prediction2/ [feature/m2.2-maritime-validation]
```

## Branch Layout

- **`main`**: Integration branch (current: `9f450c2`)
- **`feature/m2.1-error-analysis`**: M2.1 Error Analysis Framework
- **`feature/m2.2-maritime-validation`**: M2.2 Maritime Domain Validation

## Development Commands

### Switch Between Worktrees

```bash
# Work on M2.1 Error Analysis
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.1-error-analysis/maritime_trajectory_prediction2

# Work on M2.2 Maritime Validation
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.2-maritime-validation/maritime_trajectory_prediction2

# Return to main integration branch
cd /home/jakup/repo/maritime-trajectory-prediction/maritime_trajectory_prediction2
```

### Check Worktree Status

```bash
cd /home/jakup/repo/maritime-trajectory-prediction
git worktree list
```

## Conflict Minimization Strategy

### M2.1 Error Analysis - Safe File Zones
**Primary Development Areas** (minimal conflict risk):
```
src/evalx/error_analysis/          [NEW - M2.1 exclusive]
├── __init__.py
├── slicers.py                     # ErrorSlicer class
├── mining.py                      # FailureMiner class
├── horizon_curves.py              # Horizon analysis
└── case_studies.py                # Failure case documentation

configs/evaluation/                [NEW - M2.1 exclusive]
├── error_analysis.yaml            # M2.1 configuration
└── slicing_configs.yaml           # Slicing parameters

tests/test_evalx/                  [EXTEND - minimal conflict]
├── test_error_analysis/           # M2.1 tests
└── test_slicing.py                # Unit tests
```

### M2.2 Maritime Validation - Safe File Zones
**Primary Development Areas** (minimal conflict risk):
```
src/maritime/                      [NEW - M2.2 exclusive]
├── __init__.py
├── cpa_tcpa.py                    # CPA/TCPA calculations
├── colregs.py                     # COLREGS compliance
├── domain_metrics.py              # Maritime metrics
└── validators.py                  # Domain validation

configs/maritime/                  [NEW - M2.2 exclusive]
├── domain_validation.yaml         # M2.2 configuration
└── cpa_tcpa_params.yaml           # CPA parameters

tests/test_maritime/               [NEW - M2.2 exclusive]
├── test_cpa_tcpa.py               # CPA/TCPA tests
└── test_colregs.py                # COLREGS tests
```

### Shared Risk Areas (Coordinate Changes)
**Files requiring coordination between M2.1 and M2.2**:

```
requirements.txt                   [MEDIUM RISK - coordinate additions]
- M2.1 may add: joblib>=1.3.0 (parallel processing)
- M2.2 will add: pyproj>=3.5.0, shapely>=2.0.0, networkx>=3.1

src/utils/maritime_utils.py        [MEDIUM RISK - coordinate extensions]
- M2.1: May extend for error analysis utilities
- M2.2: Will extend for CPA/TCPA utilities

configs/config.yaml               [LOW RISK - separate sections]
- M2.1: Add evaluation section
- M2.2: Add maritime section

src/metrics/                      [LOW RISK - different files]
- M2.1: Extend enhanced_metrics.py
- M2.2: Create operational/ subdirectory
```

## Parallel Development Workflow

### Phase 1: Independent Development (Days 1-5)

#### M2.1 Developer Commands
```bash
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.1-error-analysis/maritime_trajectory_prediction2

# Start M2.1a: Performance Slicing (Day 1-2)
git checkout feature/m2.1-error-analysis
mkdir -p src/evalx/error_analysis
mkdir -p configs/evaluation

# Development work here...
git add src/evalx/error_analysis/
git commit -m "feat: M2.1a implement performance slicing framework"

# Continue with M2.1b: Failure Mining (Day 3-5)
# Development work...
git commit -m "feat: M2.1b implement failure mining and clustering"
```

#### M2.2 Developer Commands
```bash
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.2-maritime-validation/maritime_trajectory_prediction2

# Start M2.2a: CPA/TCPA Implementation (Day 1-3)
git checkout feature/m2.2-maritime-validation
mkdir -p src/maritime
mkdir -p configs/maritime

# Development work here...
git add src/maritime/
git commit -m "feat: M2.2a implement CPA/TCPA calculations"

# Continue with M2.2b: COLREGS Compliance (Day 4-5)
# Development work...
git commit -m "feat: M2.2b implement COLREGS compliance validation"
```

### Phase 2: Integration (Day 6)

#### Merge M2.1 First
```bash
cd /home/jakup/repo/maritime-trajectory-prediction/maritime_trajectory_prediction2

# Switch to main
git checkout main

# Merge M2.1 error analysis
git merge feature/m2.1-error-analysis --no-ff -m "feat: integrate M2.1 error analysis framework

- Performance slicing by maritime conditions
- Failure mining and clustering analysis
- Horizon curve generation with confidence intervals
- Automated failure case study generation"

# Test integration
make test
```

#### Merge M2.2 Second
```bash
# Merge M2.2 maritime validation
git merge feature/m2.2-maritime-validation --no-ff -m "feat: integrate M2.2 maritime domain validation

- CPA/TCPA calculation and validation
- COLREGS compliance checking framework
- Maritime-specific domain metrics
- Vessel encounter classification"

# Resolve any conflicts (should be minimal with this strategy)
# Test full integration
make test
make ci
```

### Phase 3: M2.3 Development (Day 7-10)
```bash
# M2.3 depends on M2.2, so develop on main after integration
git checkout main
mkdir -p src/metrics/operational
mkdir -p configs/metrics

# Implement operational metrics
git commit -m "feat: M2.3 implement operational metrics system"
```

## Conflict Resolution Protocol

### If Conflicts Occur in Shared Files

#### requirements.txt Conflicts
```bash
# Resolution strategy: Combine all additions
# M2.1 additions + M2.2 additions = merged requirements

# Example merged requirements.txt:
# M2: Error Analysis & Maritime Validation
pyproj>=3.5.0        # M2.2: Geodetic transformations
shapely>=2.0.0        # M2.2: Geometric operations
networkx>=3.1         # M2.2: Graph analysis
joblib>=1.3.0         # M2.1: Parallel processing
```

#### maritime_utils.py Conflicts
```bash
# Resolution strategy: Create separate utility classes
class ErrorAnalysisUtils:    # M2.1 utilities
    pass

class MaritimeDomainUtils:   # M2.2 utilities
    pass

class MaritimeUtils:         # Existing utilities
    pass
```

#### config.yaml Conflicts
```bash
# Resolution strategy: Separate configuration sections
defaults:
  - evaluation: error_analysis    # M2.1
  - maritime: domain_validation   # M2.2
  - metrics: operational         # M2.3
```

## Testing Strategy per Worktree

### M2.1 Testing Commands
```bash
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.1-error-analysis/maritime_trajectory_prediction2

# Run M2.1-specific tests
python -m pytest tests/test_evalx/test_error_analysis/ -v
python -m pytest tests/test_evalx/test_slicing.py -v

# Run integration tests
python -m pytest tests/integration/test_m2_1_integration.py -v
```

### M2.2 Testing Commands
```bash
cd /home/jakup/repo/maritime-trajectory-prediction/maritime-m2.2-maritime-validation/maritime_trajectory_prediction2

# Run M2.2-specific tests
python -m pytest tests/test_maritime/ -v
python -m pytest tests/test_maritime/test_cpa_tcpa.py -v

# Run integration tests
python -m pytest tests/integration/test_m2_2_integration.py -v
```

## Development Coordination

### Daily Sync Protocol
1. **Morning**: Check worktree list and sync with main if needed
2. **End of Day**: Push progress to respective feature branches
3. **Communication**: Coordinate any shared file modifications

### Shared File Modification Protocol
1. **Announce**: Before modifying shared files, announce in team chat
2. **Minimal Changes**: Make smallest possible changes to shared files
3. **Test**: Verify changes don't break existing functionality
4. **Document**: Document all shared file modifications

## Emergency Recovery

### If Worktree Gets Corrupted
```bash
# Remove corrupted worktree
git worktree remove maritime-m2.1-error-analysis --force

# Recreate worktree
cd /home/jakup/repo/maritime-trajectory-prediction
git worktree add maritime-m2.1-error-analysis feature/m2.1-error-analysis
```

### If Branch Gets Out of Sync
```bash
# In the specific worktree
git fetch origin
git rebase origin/main

# Resolve conflicts and continue
git rebase --continue
```

## Cleanup After M2 Completion

```bash
# After successful M2 integration, clean up worktrees
cd /home/jakup/repo/maritime-trajectory-prediction

git worktree remove maritime-m2.1-error-analysis
git worktree remove maritime-m2.2-maritime-validation

# Delete feature branches (optional)
git branch -d feature/m2.1-error-analysis
git branch -d feature/m2.2-maritime-validation
```

## Summary

✅ **Worktree Split Complete**
- M2.1 Error Analysis: `/home/jakup/repo/maritime-trajectory-prediction/maritime-m2.1-error-analysis/`
- M2.2 Maritime Validation: `/home/jakup/repo/maritime-trajectory-prediction/maritime-m2.2-maritime-validation/`
- Integration Point: `/home/jakup/repo/maritime-trajectory-prediction/maritime_trajectory_prediction2/`

✅ **Conflict Minimization**
- 90%+ of development in separate file trees
- Clear coordination protocol for shared files
- Integration strategy to minimize merge conflicts

✅ **Parallel Development Ready**
- Independent development environments
- Separate testing and configuration
- Clear merge sequence for integration

This setup enables **efficient parallel development** while maintaining **code quality and integration safety**.
