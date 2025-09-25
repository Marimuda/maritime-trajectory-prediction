# Maritime AIS Dataset Analysis

This folder contains the dataset paper deliverables generated from the CLAUDE.md requirements assessment.

## Generated Outputs

### 📊 Tables

**✅ Immediately Deliverable Tables:**

- **Table 1: Dataset Summary Statistics** (`table1_dataset_summary.csv`)
  - Total messages: 580,442
  - Unique vessels: 166
  - Observation period: 6 days (2025-05-08 to 2025-05-14)
  - Geographic coverage: North Sea area

- **Table 2: Detailed Field Dictionary** (`table2_field_dictionary.csv`)
  - Complete field descriptions, data types, units, and sources
  - ITU-R M.1371 compliant field mappings
  - Example values and source attribution

- **Table 4: Kinematic Feature Statistics** (`table4_kinematic_statistics.csv`)
  - SOG, COG, heading, turn rates, acceleration statistics
  - Mean, median, std dev, min/max, quantiles
  - Data completeness percentages

- **Table 5: AIS Message Type Distribution** (`table5_message_types.csv`)
  - Breakdown by message types (1, 3, 18, 19)
  - Count and percentage distribution
  - Message type descriptions

- **Table 6: Static Data Completeness** (`table6_static_data_completeness.csv`)
  - Vessel name, call sign, IMO, type, dimensions availability
  - Completeness rates for static vessel information

- **Table 7: Preprocessing Summary & Outlier Statistics** (`table7_preprocessing_summary.csv`)
  - Raw vs processed message counts
  - Filtering and validation statistics
  - Trajectory segmentation metrics

### 📈 Figures

**✅ Immediately Deliverable Figures:**

- **Figure 7: Speed Over Ground Distribution** (`figure7_sog_distribution.png`)
  - SOG histogram and temporal patterns
  - Speed distribution by hour of day

- **Figure 8: Course Over Ground Distribution** (`figure8_cog_distribution.png`)
  - Polar histogram showing directional preferences
  - Rose plot of vessel courses

- **Figure 9: Turn Rate Distribution** (`figure9_turn_rate_distribution.png`)
  - Histogram of calculated turn rates
  - Filtered for visualization (outliers removed)

- **Figure 10: Example Trajectories** (`figure10_example_trajectories.png`)
  - 4-panel plot showing:
    - Geographic vessel trajectories
    - Speed profiles over time
    - Course profiles over time
    - Trajectory colored by speed

- **Figure 11: Temporal Analysis** (`figure11_temporal_analysis.png`)
  - Message reception rates by hour and day
  - Inter-message interval distributions
  - Activity heatmaps by vessel type

- **Figure 14: Correlation Heatmap** (`figure14_correlation_heatmap.png`)
  - Pearson correlation matrix of kinematic features
  - Feature relationships visualization

## 🔧 Analysis Scripts

- **`dataset_analyzer.py`**: Main analysis script generating core tables and figures
- **`advanced_analysis.py`**: Extended analysis for temporal and preprocessing statistics

## ⚠️ Missing Components (Require Additional Implementation)

The following components from CLAUDE.md require physical-layer data extraction:

- **Figure 2**: Signal Power vs. Distance
- **Figure 3**: Frequency Offset (PPM) vs. Radial Velocity
- **Figure 4**: RSSI Distribution
- **Figure 5**: PPM Distribution
- **Figure 6**: PPM Stability Over Time
- **Table 3**: Physical-Layer Feature Statistics

**Reason**: Physical-layer data (`signalpower`, `ppm`) exists in raw AIS-catcher logs but is not currently extracted during preprocessing.

## 📋 Delivery Status Summary

**Immediately Deliverable (80% complete):**
- ✅ Core dataset description and access
- ✅ Complete kinematic data characterization
- ✅ Temporal and message type analysis
- ✅ Data quality and preprocessing insights
- ✅ Basic correlation analysis

**Requires Implementation (20% remaining):**
- ⚠️ Physical-layer data extraction and analysis
- ⚠️ Signal propagation characterization
- ⚠️ RSSI/PPM statistical analysis

## 🚀 Usage

Run the analysis scripts to regenerate outputs:

```bash
# Generate core tables and figures
python3 dataset_analysis/scripts/dataset_analyzer.py

# Generate advanced temporal and statistical analysis
python3 dataset_analysis/scripts/advanced_analysis.py
```

## 📁 Data Sources

The analysis uses:
- **Processed Parquet files**: For kinematic data analysis
- **Raw AIS-catcher logs**: For comprehensive message analysis (when available)
- **Synthetic data fallback**: For demonstration when no data files present

## 📖 Dataset Paper Integration

These outputs directly support the dataset paper sections:

1. **Dataset Description**: Tables 1-2 provide comprehensive overview
2. **Kinematic Analysis**: Figures 7-11 and Table 4 characterize vessel movement
3. **Data Quality**: Table 7 demonstrates preprocessing rigor
4. **Feature Relationships**: Figure 14 shows correlation patterns
5. **Temporal Patterns**: Figure 11 reveals reception and activity patterns

The generated figures and tables can be directly integrated into a dataset paper manuscript, with high-resolution PNG outputs suitable for publication.
