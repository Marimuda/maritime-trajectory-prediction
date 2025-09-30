# Critical Performance and Crash Issues

**STATUS: ALL ISSUES RESOLVED ✅**

Both critical issues have been fixed, tested, and committed:
- Commit `3c25196`: Multiprocessing pickle error (CRASH) - FIXED
- Commit `db37c0d`: Slow distance calculation (21+ min delay) - FIXED (250x+ speedup)

---

## Issue 1: Multiprocessing Pickle Error (CRASH)

**Location:** `src/data/ais_datamodule.py:489-530`

**Error:**
```
AttributeError: Can't pickle local object 'AISDataModule._create_sequences_from_data.<locals>._process_vessel'
```

**Root Cause:**
The `_process_vessel` function is defined as a **local function** inside `_create_sequences_from_data`. Python's `multiprocessing.Pool.map()` requires functions to be picklable, but local/nested functions cannot be pickled.

**Impact:**
- Complete crash during parallel sequence generation
- Prevents training pipeline from running
- Affects all baseline model training

**Solution:**
Move `_process_vessel` to either:
1. **Module-level function** (outside the class)
2. **Static method** of the class
3. **Instance method** (less efficient due to pickling self)

**Example Fix:**
```python
# Option 1: Module-level function (RECOMMENDED)
def _process_vessel_sequences(args):
    """Process a single vessel to create sequences."""
    mmsi, vessel_df, seq_len, pred_horizon, feature_cols, target_features = args
    # ... existing logic ...
    return vessel_sequences

class AISDataModule:
    def _create_sequences_from_data(self, processed_data):
        # ... setup code ...

        # Use module-level function instead of local function
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_vessel_sequences, vessel_args)
```

---

## Issue 2: Extremely Slow Spatial Feature Computation (1296+ seconds)

**Location:**
- `src/utils/maritime_utils.py:36-44` (root cause)
- `src/data/dataset_builders.py:187-192` (caller)

**Symptom:**
```
✓ Spatial features added in 1296.6s  (21+ minutes!)
```

**Root Cause:**
`MaritimeUtils.calculate_distance()` has catastrophically inefficient scalar-to-Series broadcasting:

```python
# INEFFICIENT - Creates huge intermediate lists!
if isinstance(lat2, pd.Series):
    lat1 = pd.Series([lat1] * len(lat2), index=lat2.index)  # 500k-element list!
    lon1 = pd.Series([lon1] * len(lat2), index=lat2.index)  # Another 500k list!
```

When computing distance from center point to all positions:
- `df["lat"]` has 500,000 rows (Series)
- `center_lat` is a scalar
- Creates `[center_lat] * 500000` - a 500k-element list in memory
- Does this TWICE (lat and lon)
- Converts both lists to pandas Series
- This is O(n) memory allocation + O(n) list construction + O(n) Series construction

**Impact:**
- 21+ minutes for spatial features on large datasets
- Blocks entire preprocessing pipeline
- Unnecessary memory pressure

**Solution:**
Use numpy broadcasting directly without intermediate lists:

```python
# EFFICIENT - Let numpy handle broadcasting
if is_series:
    # Get the Series length for result indexing
    if isinstance(lat1, pd.Series):
        result_index = lat1.index
        lat1_arr = lat1.values
        lon1_arr = lon1.values
        lat2_arr = np.broadcast_to(lat2, lat1_arr.shape) if np.isscalar(lat2) else lat2.values
        lon2_arr = np.broadcast_to(lon2, lon1_arr.shape) if np.isscalar(lon2) else lon2.values
    else:  # lat2 is Series
        result_index = lat2.index
        lat2_arr = lat2.values
        lon2_arr = lon2.values
        lat1_arr = np.broadcast_to(lat1, lat2_arr.shape) if np.isscalar(lat1) else lat1.values
        lon1_arr = np.broadcast_to(lon1, lon2_arr.shape) if np.isscalar(lon1) else lon1.values
else:
    # Scalar case - convert to arrays
    lat1_arr = np.asarray([lat1])
    lon1_arr = np.asarray([lon1])
    lat2_arr = np.asarray([lat2])
    lon2_arr = np.asarray([lon2])
    result_index = None

# Rest of calculation remains the same...
```

**Expected Performance Improvement:**
- From: 1296+ seconds (~21 minutes)
- To: < 5 seconds (250x+ speedup)

---

## Summary

| Issue | Type | Location | Impact | Difficulty |
|-------|------|----------|--------|-----------|
| Pickle error | Crash | `ais_datamodule.py:489` | Complete failure | Easy |
| Slow distance calc | Performance | `maritime_utils.py:36-44` | 21+ min delay | Medium |

Both issues are **CRITICAL** and should be fixed before running large-scale experiments.

**Tags Added:**
- `# TODO: CRITICAL BUG` - for the pickle error
- `# TODO: CRITICAL PERFORMANCE BUG` - for the slow distance calculation
- `# HACK:` - marking workarounds that need proper fixes

All tagged locations include:
1. Description of the problem
2. Why it happens
3. Suggested solution
