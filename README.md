
---

````markdown
# 🔧 Cyclone Machine Data Analysis
> A comprehensive time-series analysis pipeline for industrial cyclone sensor data featuring shutdown detection, operational state clustering, contextual anomaly detection, and predictive forecasting.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📊 Overview
This project performs advanced analysis on 3 years of cyclone sensor data (~370,000 records at 5-minute intervals) to extract actionable insights for predictive maintenance and operational optimization. The analysis pipeline includes data preprocessing, exploratory analysis, machine learning-based state segmentation, contextual anomaly detection, and forecasting capabilities.

### Key Features
- **Automated Shutdown Detection** – Multi-criteria detection of idle periods with downtime analytics  
- **Operational State Clustering** – K-Means/DBSCAN segmentation with optimized feature engineering  
- **Context-Aware Anomaly Detection** – State-specific Isolation Forest with root cause analysis  
- **Short-Horizon Forecasting** – 1-hour ahead predictions with Ridge regression  
- **Comprehensive Visualizations** – 20+ automated plots for insights and reporting  

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| **Time Period** | 3 years (2017-2019) |
| **Sampling Frequency** | 5-minute intervals |
| **Total Records** | ~378,580 data points |
| **Data Format** | CSV/Excel |

### Variables
- `Cyclone_Inlet_Gas_Temp` – Temperature of hot gas at cyclone inlet (°C)  
- `Cyclone_Gas_Outlet_Temp` – Temperature of hot gas at cyclone outlet (°C)  
- `Cyclone_Outlet_Gas_draft` – Gas draft (pressure) at cyclone outlet  
- `Cyclone_cone_draft` – Gas draft (pressure) at cyclone cone section  
- `Cyclone_Inlet_Draft` – Gas draft (pressure) at cyclone inlet  
- `Cyclone_Material_Temp` – Temperature of material at cyclone outlet (°C)  

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
````

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/cyclone-machine-analysis.git
   cd cyclone-machine-analysis
   ```

2. **Install dependencies**

   ```bash
   pip install numpy pandas scipy matplotlib seaborn scikit-learn openpyxl tqdm
   ```

3. **Add your data**

   ```bash
   # Place your data file in the project root
   cp /path/to/your/data.xlsx ./data.xlsx
   ```

4. **Run the analysis**

   ```bash
   # Open Jupyter notebook
   jupyter notebook Project.ipynb

   # Or run as Python script (if converted)
   python task1_analysis.py
   ```

---

## 📁 Project Structure

```
Task1/
├── outputs/                              # Analysis results (CSV files)
│   ├── shutdown_periods.csv            # Detected shutdown events
│   ├── anomalous_periods.csv           # Anomalies with severity scores
│   ├── clusters_summary.csv            # Operational state statistics
│   ├── forecasts.csv                   # Forecast predictions vs actuals
│   ├── summary_stats.csv               # Descriptive statistics
│   ├── correlations.csv                # Variable correlation matrix
│   ├── forecast_metrics.json           # Model performance metrics
│   └── insights_summary.txt            # Key findings and recommendations
├── plots/                                # Visualizations (PNG files)
│   ├── correlation_matrix.png          # Variable correlation heatmap
│   ├── eda_week_slice.png              # One-week data overview
│   ├── eda_year_slice.png              # One-year data overview
│   ├── year_with_shutdowns.png         # Annual view with shutdown bands
│   ├── shutdowns_by_month.png          # Shutdown frequency distribution
│   ├── operational_states_timeline.png # State segmentation timeline
│   ├── state_distribution.png          # Cluster distribution pie chart
│   ├── state_transitions.png           # State transition heatmap
│   ├── anomaly_event_*.png             # Top anomaly details (5 plots)
│   ├── anomaly_frequency.png           # Anomaly temporal distribution
│   ├── forecast_comparison.png         # Model vs baseline predictions
│   └── forecast_error_analysis.png     # Error distribution analysis
├── Project.ipynb                         # Main analysis notebook
└── README.md                             # This file
```

---

## ⚙️ Configuration

Customize analysis parameters by modifying the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    "input_csv": "data.xlsx",
    "timestamp_col": "timestamp",
    "freq_minutes": 5,

    "outliers": {
        "method": "mad",
        "k": 5.0
    },

    "shutdown_rules": {
        "min_idle_minutes": 30,
        "max_var_window_min": 30,
        "near_zero_thresholds": {
            "Cyclone_Outlet_Gas_draft": 0.02,
            "Cyclone_cone_draft": 0.02
        },
        "flat_std_thresholds": {
            "Cyclone_Inlet_Gas_Temp": 1.0,
            "Cyclone_Gas_Outlet_Temp": 1.0
        }
    },

    "cluster": {
        "max_k": 6,
        "min_k": 3,
        "dbscan": {
            "eps": 0.8,
            "min_samples": 200
        }
    },

    "anomaly": {
        "contamination": 0.02,
        "merge_gap_minutes": 10,
        "min_event_minutes": 10
    },

    "forecast": {
        "target": "Cyclone_Inlet_Gas_Temp",
        "horizon_steps": 12,
        "train_ratio": 0.8,
        "model": "ridge"
    },

    "random_state": 42
}
```

---

## 🔬 Analysis Pipeline

### 1️⃣ Data Preparation & Exploratory Analysis

* Auto-detect timestamp column and enforce strict 5-minute indexing
* Handle missing values using forward-fill (up to 30 minutes)
* Robust outlier treatment with Median Absolute Deviation (MAD)
* Generate summary statistics and correlation matrix
* Create time-series visualizations (weekly and yearly slices)

**Outputs**: `summary_stats.csv`, `correlations.csv`, `correlation_matrix.png`, `eda_*.png`

---

### 2️⃣ Shutdown / Idle Period Detection

**Multi-Criteria Detection:**

* Near-zero draft conditions (both sensors < 0.02)
* Flat temperature patterns (rolling std < 1.0°C for 30-min windows)
* Minimum duration filter (events ≥30 minutes)

**Analytics:**

* Total downtime calculation (~7,178 hours over 3 years)
* Machine availability metrics (~77.2%)
* Shutdown frequency by month

**Outputs**: `shutdown_periods.csv`, `year_with_shutdowns.png`, `shutdowns_by_month.png`

---

### 3️⃣ Machine State Segmentation (Clustering)

**Feature Engineering:**

* Raw sensor values
* Rolling statistics (1-hour & 6-hour windows)
* Lag features (1, 2, 12 steps)
* First-order differences (deltas)

**Clustering Algorithms:**

* **K-Means**: Optimized with elbow method for large datasets (>10k points)
* **DBSCAN**: Density-based for irregular patterns
* StandardScaler normalization

**Per-State Analytics:**

* Summary statistics (mean, std, percentiles)
* Behavior characterization
* Duration profiling and event counts
* State transition analysis

**Outputs**: `clusters_summary.csv`, `operational_states_timeline.png`, `state_distribution.png`, `state_transitions.png`

---

### 4️⃣ Contextual Anomaly Detection + Root Cause Analysis

**State-Aware Detection:**

* Separate Isolation Forest per operational state
* Context-specific anomaly scoring (relative to state distribution)
* Event merging (within 10-minute gaps)
* Minimum duration filter (≥10 minutes)

**Root Cause Analysis:**

* Compute per-variable z-scores during anomaly windows
* Rank variables by absolute mean z-score
* Report top 3 contributors per event
* Severity scoring across all variables

**Outputs**: `anomalous_periods.csv`, `anomaly_event_*.png`, `anomaly_frequency.png`

---

### 5️⃣ Short-Horizon Forecasting

**Target:** `Cyclone_Inlet_Gas_Temp` for next 1 hour (12 steps)

**Models:**

* **Ridge Regression** (α=1.0) with lag and rolling features
* **Persistence Baseline** (previous value)

**Features:**

* Lags: 1, 2, 3, 6, 12, 24 steps (5 min to 2 hours)
* Rolling mean (12-step window)
* Rolling std (12-step window)

**Evaluation:** RMSE and MAE on 20% held-out test set
**Typical Results:**

* Ridge RMSE: ~12.3°C
* Baseline RMSE: ~16.4°C
* **Improvement: ~25.2%**

**Outputs**: `forecasts.csv`, `forecast_metrics.json`, `forecast_comparison.png`, `forecast_error_analysis.png`

---

### 6️⃣ Insights & Storytelling

**Automated Insights:**

* Connection between shutdowns, clusters, and anomalies
* Efficiency metrics by operational state
* Temporal patterns in anomaly occurrence
* Forecasting performance analysis

**Actionable Recommendations:**

* Monitoring rules and alert triggers
* Predictive maintenance strategies
* Additional data collection suggestions
* Operational optimization opportunities

**Output:** `insights_summary.txt`

---

## 📈 Key Results

| Metric                        | Value                 |
| ----------------------------- | --------------------- |
| **Machine Availability**      | 77.2%                 |
| **Total Downtime**            | 7,178 hours (3 years) |
| **Shutdown Events**           | 268 events detected   |
| **Operational States**        | 5 distinct patterns   |
| **Anomaly Rate**              | 0.69 events/day       |
| **Forecast RMSE Improvement** | 25.2% over baseline   |

---

## 📊 Sample Visualizations

### Shutdown Detection

* Full-year timeline with highlighted shutdown periods
* Monthly shutdown frequency distribution

### Clustering Analysis

* Operational states timeline (color-coded)
* State distribution pie chart
* State transition heatmap

### Anomaly Detection

* Top 5 anomaly events with context windows
* Temporal distribution of anomalies
* Variable contribution analysis

### Forecasting

* Model vs baseline comparison
* Error distribution and temporal analysis

---

## 🛠️ Requirements

### Core Dependencies

```bash
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
openpyxl>=3.0.0
tqdm>=4.50.0
```

### Optional Dependencies

```bash
jupyter>=1.0.0           # For notebook interface
statsmodels>=0.12.0      # For advanced statistical analysis
```

---

## 🐛 Troubleshooting

**Issue:** `FileNotFoundError` when loading data
**Solution:** Ensure `data.xlsx` is in project root or update `CONFIG["input_csv"]`

**Issue:** Memory errors with large datasets
**Solution:** Use optimized clustering or sample large data subsets

**Issue:** Unicode encoding errors
**Solution:** All file operations use `encoding='utf-8'`

**Issue:** Slow clustering performance
**Solution:** Reduce `CONFIG["cluster"]["max_k"]` for speed

**Issue:** No anomalies detected
**Solution:** Increase `CONFIG["anomaly"]["contamination"]` (default = 0.02)

---

## 📝 Deliverables Checklist

All outputs are generated automatically in the `Task1/` folder:

* [x] `outputs/shutdown_periods.csv`
* [x] `outputs/anomalous_periods.csv`
* [x] `outputs/clusters_summary.csv`
* [x] `outputs/forecasts.csv`
* [x] `outputs/summary_stats.csv`
* [x] `outputs/correlations.csv`
* [x] `outputs/forecast_metrics.json`
* [x] `outputs/insights_summary.txt`
* [x] `plots/` (20+ visualization files)

---

## 🔍 Methodology Highlights

* **Shutdown Detection:** Multi-criteria approach combining near-zero draft, flat temperature, and duration filters.
* **Clustering:** Rolling stats, lags, and deltas; optimized via elbow method.
* **Anomaly Detection:** Context-aware Isolation Forest trained per operational state.
* **Forecasting:** Ridge regression with lag features, evaluated vs baseline.

---

## 🎯 Future Enhancements

* [ ] Real-time streaming support
* [ ] LSTM/GRU forecasting models
* [ ] Interactive dashboards (Plotly/Dash)
* [ ] Automated alert system
* [ ] Multi-variable forecasting
* [ ] Seasonal decomposition analysis
* [ ] Causal inference modeling

---

## 📚 References

**Libraries:**

* NumPy, Pandas – Data manipulation
* Scikit-learn – ML (clustering, regression, anomaly detection)
* Matplotlib, Seaborn – Visualization
* SciPy – Statistical analysis

**Techniques:**

* Median Absolute Deviation (MAD)
* K-Means & DBSCAN clustering
* Isolation Forest anomaly detection
* Ridge regression forecasting
* Silhouette & elbow methods

---

## 🤝 Contributing

1. Review notebook structure and inline comments
2. Test on a data subset before pushing
3. Ensure all outputs generate correctly
4. Update documentation for config changes

---

## 📞 Contact & Support

* Review inline code comments in `Project.ipynb`
* Check parameters in `CONFIG` dictionary
* Examine function docstrings for clarity

---

## 📜 License

This project is provided for educational and research purposes. Modify configuration parameters for your specific use case.

---

## 🏆 Project Status

**Status:** ✅ Complete
**Last Updated:** October 2025
**Version:** 1.0.0

---

## 🙏 Acknowledgments

* Developed for industrial cyclone maintenance optimization
* Implements best practices in time-series and predictive maintenance
* Optimized for large-scale sensor data processing

---

**Made with ❤️ for predictive maintenance and operational excellence**

```

---

