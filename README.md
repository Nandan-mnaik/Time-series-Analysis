# Cyclone Machine Data Analysis Pipeline

A comprehensive time series analysis pipeline for cyclone machine operational data, featuring preprocessing, exploratory analysis, shutdown detection, operational state clustering, anomaly detection, and forecasting capabilities.[1]

## Overview

This project implements an end-to-end data analysis system for monitoring and analyzing cyclone machine sensor data. The pipeline processes 5-minute interval time series data from multiple temperature and pressure sensors to extract operational insights, detect anomalies, and forecast future behavior.[1]

### Key Features

- Automated time series preprocessing with outlier detection using MAD (Median Absolute Deviation)
- Multi-variable shutdown and idle period detection
- K-means clustering for operational state identification
- Isolation Forest-based anomaly detection with severity ranking
- Ridge regression forecasting with time series cross-validation
- Comprehensive visualization suite for all analysis stages

## Requirements

### Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
tqdm
scikit-learn
openpyxl
```

Install all dependencies using:

```bash
pip install numpy pandas scipy matplotlib seaborn tqdm scikit-learn openpyxl
```

## Input Data

### Expected Format

- **File**: `data.xlsx` (Excel format)
- **Timestamp Column**: Auto-detected (supports `timestamp`, `time`, `datetime`, etc.)
- **Sampling Frequency**: 5-minute intervals
- **Required Variables**:
  - `Cyclone_Inlet_Gas_Temp`
  - `Cyclone_Gas_Outlet_Temp`
  - `Cyclone_Outlet_Gas_draft`
  - `Cyclone_cone_draft`
  - `Cyclone_Inlet_Draft`
  - `Cyclone_Material_Temp`

## Project Structure

```
.
├── Project.ipynb              # Main analysis notebook
├── data.xlsx                  # Input data file
├── Task1/
│   ├── outputs/              # Generated analysis results (CSV/JSON)
│   └── plots/                # Visualization outputs (PNG)
```

## Usage

### Running the Analysis

1. **Setup Environment**: Ensure all dependencies are installed
2. **Prepare Data**: Place `data.xlsx` in the project root directory
3. **Execute Notebook**: Run all cells in `Project.ipynb` sequentially

```bash
jupyter notebook Project.ipynb
```

### Configuration

The pipeline is configurable through the `CONFIG` dictionary in the notebook:

- **Data Parameters**: Input file path, timestamp column, sampling frequency
- **Outlier Detection**: MAD method with k=5.0 threshold
- **Shutdown Rules**: Minimum idle duration, variance thresholds
- **Clustering**: K-means parameters (k=3-6), DBSCAN settings
- **Anomaly Detection**: Contamination rate (2%), event duration thresholds
- **Forecasting**: Target variable, horizon (12 steps = 60 minutes), train/test split

## Pipeline Stages

### 1. Data Loading and Preprocessing

- Loads Excel data and validates format
- Auto-detects timestamp column
- Enforces fixed 5-minute intervals
- Handles missing timestamps with forward-fill (up to 30 minutes)
- Applies MAD-based outlier winsorization

### 2. Exploratory Data Analysis

- Generates comprehensive summary statistics
- Correlation analysis across all variables
- Distribution analysis and visualization
- Time series plots for all sensor variables

### 3. Shutdown Detection

- Multi-criteria detection using:
  - Near-zero draft conditions (threshold: 0.02)
  - Flat temperature variance (threshold: 1.0 std)
- Minimum shutdown duration: 30 minutes
- Outputs shutdown event timeline and statistics

### 4. Operational State Clustering

- Feature engineering: rolling statistics (12 & 72 steps), lag features (1, 2, 12 steps), deltas
- K-means clustering on active (non-shutdown) periods
- Optimal k selection via silhouette score
- Identifies 4-5 distinct operational states
- Generates state transition matrices

### 5. Anomaly Detection

- Isolation Forest algorithm per operational state
- Contamination rate: 2%
- Event merging (gap ≤ 10 minutes)
- Severity scoring based on anomaly scores
- Identifies top contributing variables per anomaly

### 6. Time Series Forecasting

- Target: `Cyclone_Inlet_Gas_Temp`
- Horizon: 12 steps (60 minutes ahead)
- Ridge regression with lag features
- Time series cross-validation
- Outputs predictions with confidence intervals

## Outputs

### Generated Files

**Task1/outputs/**
- `shutdown_events.csv` - Detected shutdown periods with durations
- `operational_states.csv` - Clustering results with state assignments
- `state_summary.csv` - Statistics per operational state
- `anomaly_events.csv` - Detected anomalies with severity rankings
- `forecast_results.csv` - Predictions and actual values
- `insights_summary.json` - Comprehensive analysis summary

**Task1/plots/**
- `01_timeseries_overview.png` - All variables over time
- `02_correlation_heatmap.png` - Cross-variable correlations
- `03_distributions.png` - Variable distributions
- `04_shutdown_timeline.png` - Shutdown events visualization
- `05_cluster_elbow.png` - K-means elbow curve
- `06_cluster_timeline.png` - Operational states over time
- `07_state_transitions.png` - State transition matrix
- `08_anomaly_timeline.png` - Detected anomalies
- `09_anomaly_severity.png` - Severity distribution
- `10_forecast_results.png` - Forecast vs actual comparison

## Key Insights

The analysis pipeline identifies:
- **Operational States**: Typically 4-5 distinct states including normal operation, high-load, low-load, and transitional states
- **Shutdown Patterns**: Long-duration shutdowns (up to 57,000+ minutes) and frequent short maintenance stops
- **Anomalies**: Rare events with high severity scores indicating potential equipment issues or unusual operating conditions
- **Predictive Accuracy**: 60-minute ahead temperature forecasts for proactive monitoring

## Technical Details

### Feature Engineering

- **Rolling Statistics**: 12-step (1 hour) and 72-step (6 hour) windows for mean and standard deviation
- **Lag Features**: 1, 2, and 12-step lags capturing short and medium-term dependencies
- **Delta Features**: First-order differences for rate-of-change analysis

### Performance Optimization

The notebook includes optimized implementations for large datasets:
- Silhouette score computation optimization note (O(n²) complexity warning)
- Efficient event detection algorithms
- Memory-efficient rolling window computations

## Limitations and Notes

- The silhouette score function in clustering has O(n²) time complexity and may be slow for very large datasets (>100K active samples)
- Forward-fill imputation limited to 30 minutes to avoid unrealistic data propagation
- Anomaly detection assumes 2% contamination rate (adjustable in CONFIG)
- Forecasting limited to 60 minutes ahead with Ridge regression (expandable to ARIMA/LSTM for longer horizons)

## License

This project is provided as-is for data analysis purposes.
